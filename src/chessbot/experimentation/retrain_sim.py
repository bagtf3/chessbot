"""Read-only retrain simulation: does a dynamic LR, patience and up to
MAX_EPOCHS passes beat the one-pass selfplay retrain?

Draws one retrain's shards from a run's primary and replay buffers (the
runner's draw) plus HOLDOUT_SHARDS extra primary shards, all read once and
shared by every arm. Three holdout sets are sampled from the pooled primary
records by source tag: val_rand (any source), val_xc0 and test (xc0 only);
the rest trains. Games straddle records, so the holdouts are not
game-disjoint.

Arms, all from the run's model and train_ckpts state:
  A   static run LR, 1 epoch
  C   static LR_HI, 1 epoch
  Bx  warmup to LR_HI, patience on val_xc0, LR lowerings, 1-3 epochs
  Br  as Bx, patience on val_rand

Nothing on disk is written except under OUT_ROOT/<stamp>_seed<seed>: shards
are read in place, the model and optimizer / scaler / LW state are only
loaded, and every file read is checked unchanged at the end.
"""
import argparse
import csv
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from chessbot.config import Config
from chessbot.replay_buffer import (
    read_shard, list_shard_files, sample_retrain_primary,
    stack_policies, RETRAIN_PRIMARY_SHARDS, RETRAIN_REPLAY_SHARDS)
from chessbot.retrain_worker import records_to_arrays
from chessbot.train_pytorch import (
    load_pt_model, decay_groups, normalize_bounded_mean, pt_opt_state_path,
    pt_scaler_state_path, retrain_lw_state_path, retrain_lw_from_state,
    VRAM_CHUNK, GATE_LEGAL_POS_W, GATE_SHARE, GATE_W_FLOOR)

RUNS_DIR = r"D:\Data\chessbot_data\selfplay_runs"
OUT_ROOT = r"D:\Data\chessbot_data\experiments\retrain_sim"
RUN = "m137_8c8t_selfplay10"
# selfplay9's retrain LR
LR_HI = 3.54e-05

VAL_RECORDS = 20_480
TEST_RECORDS = 10_240
# extra primary shards drawn so the holdouts cost the training set nothing
HOLDOUT_SHARDS = 5
LOAD_THREADS = 8
EVAL_CHUNK = 4096

EVAL_EVERY = 100
PATIENCE = 4
MAX_LOWERINGS = 2
LR_MIN_FRAC = 0.1
WARMUP_STEPS = 20
MIN_EPOCHS = 1
MAX_EPOCHS = 3
# a later eval replaces the kept best within BEST_TOL of it, never past
# LIFETIME_TOL of the lowest loss seen
BEST_TOL = 0.002
LIFETIME_TOL = 0.005

VAL_SETS = ("val_xc0", "val_rand")


def load_shards(paths):
    with ThreadPoolExecutor(LOAD_THREADS) as ex:
        return [r for rs in ex.map(read_shard, paths) for r in rs]


def file_stamps(paths):
    return {p: (os.path.getsize(p), os.path.getmtime(p)) for p in paths}


def draw(run_dir, seed):
    """RETRAIN_PRIMARY_SHARDS + HOLDOUT_SHARDS primary shards (the runner's
    draw) and RETRAIN_REPLAY_SHARDS replay shards, read once. The holdouts
    are sampled from the pooled primary records: val_rand from any source,
    then val_xc0 and test from the remaining xc0. Everything else trains.
    Returns (shard paths, training records, holdout sets)."""
    random.seed(seed)
    primary = list_shard_files(os.path.join(run_dir, "primary_buffer"))
    replay = list_shard_files(os.path.join(run_dir, "replay_buffer"))
    p_paths = sample_retrain_primary(primary, RETRAIN_PRIMARY_SHARDS + HOLDOUT_SHARDS)
    r_paths = random.sample(replay, RETRAIN_REPLAY_SHARDS)
    recs = load_shards(p_paths)

    idx = list(range(len(recs)))
    random.shuffle(idx)
    rand_idx = idx[:VAL_RECORDS]
    xc0 = [i for i in idx[VAL_RECORDS:] if recs[i][6] == "xc0"]
    need = VAL_RECORDS + TEST_RECORDS
    if len(xc0) < need:
        raise RuntimeError(f"only {len(xc0)} xc0 records for {need} holdouts")
    sets = {"val_rand": [recs[i] for i in rand_idx],
            "val_xc0": [recs[i] for i in xc0[:VAL_RECORDS]],
            "test": [recs[i] for i in xc0[VAL_RECORDS:need]]}
    held = set(rand_idx) | set(xc0[:need])
    train = [r for i, r in enumerate(recs) if i not in held]
    train += load_shards(r_paths)
    return p_paths + r_paths, train, sets


def holdout_tensors(records):
    """Per-set arrays for scoring, weights pinned the way training pins them."""
    X, P, Y, vwht, pwht = records_to_arrays(records)
    pol = stack_policies(P, index=None)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_p = np.where(pol > 0, pol * np.log(pol), 0.0).sum(1)
        h_y = np.where(Y > 0, Y * np.log(Y), 0.0).sum(1)
    return {"X": X, "P": pol, "Y": Y, "h_p": h_p, "h_y": h_y,
            "vw": normalize_bounded_mean(vwht), "pw": normalize_bounded_mean(pwht)}


def set_train_mode(model):
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()


def fp16_model(cfg, state=None):
    """The run's model, or state loaded into it, as an fp16 eval copy (the
    retrain worker's predict_fp16 setup)."""
    model, _ = load_pt_model(cfg.model_path)
    if state is not None:
        model.load_state_dict(state)
    return model.to(torch.device("cuda")).half().eval()


def evaluate(model, h, policy_lw, value_lw):
    """Weighted holdout loss as trained (policy + value, no gate term) and
    unweighted KL(target || net) sidecars for policy and value. model is an
    fp16 eval copy."""
    device = torch.device("cuda")
    ce_p, ce_v = [], []
    with torch.no_grad():
        for i in range(0, len(h["X"]), EVAL_CHUNK):
            x = torch.from_numpy(h["X"][i:i + EVAL_CHUNK]).long().to(device)
            p = torch.from_numpy(h["P"][i:i + EVAL_CHUNK]).to(device)
            y = torch.from_numpy(h["Y"][i:i + EVAL_CHUNK]).to(device)
            logits, v = model(x)
            ce_p.append(-(p * F.log_softmax(logits.float(), -1)).sum(-1).cpu())
            ce_v.append(-(y * F.log_softmax(v.float(), -1)).sum(-1).cpu())
    ce_p = torch.cat(ce_p).double().numpy()
    ce_v = torch.cat(ce_v).double().numpy()
    return {"loss": float(policy_lw * (h["pw"] * ce_p).mean()
                          + value_lw * (h["vw"] * ce_v).mean()),
            "klp": float((ce_p + h["h_p"]).mean()),
            "klv": float((ce_v + h["h_y"]).mean())}


def restore_opt(model, cfg, lr):
    """AdamW, GradScaler and LW weights from the run's train_ckpts, loaded
    only; the files are never written."""
    device = torch.device("cuda")
    opt = torch.optim.AdamW(decay_groups(model, cfg.retrain_weight_decay), lr=lr,
                            betas=(0.9, cfg.adam_beta2), fused=True)
    flat = torch.optim.Adam(model.parameters())
    flat.load_state_dict(torch.load(pt_opt_state_path(cfg.model_path, cfg.run_dir),
                                    map_location="cpu"))
    for p, state in flat.state.items():
        opt.state[p] = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in state.items()}
        if "step" in opt.state[p]:
            opt.state[p]["step"] = opt.state[p]["step"].float()
    scaler = GradScaler("cuda", init_scale=2**15)
    scaler.load_state_dict(torch.load(
        pt_scaler_state_path(cfg.model_path, cfg.run_dir), map_location="cpu"))
    lw = torch.load(retrain_lw_state_path(cfg.model_path, cfg.run_dir),
                    map_location="cpu")
    return opt, scaler, retrain_lw_from_state(lw)


class Tracker:
    """Best-checkpoint keeper with BEST_TOL / LIFETIME_TOL and a patience count."""

    def __init__(self, name, start):
        self.name = name
        self.low = self.best = start
        self.best_step, self.best_state = 0, None
        self.bad = 0

    def update(self, loss, step, model):
        """'*' new low, '+' kept as best within tolerance, ' ' neither."""
        mark = " "
        if loss < self.low:
            self.low, self.bad, mark = loss, 0, "*"
        else:
            self.bad += 1
        if loss <= self.best * (1 + BEST_TOL) and loss <= self.low * (1 + LIFETIME_TOL):
            self.best, self.best_step = loss, step
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            mark = "*" if mark == "*" else "+"
        return mark


def batches(n, epoch_seed):
    """VRAM_CHUNK slices of one seeded permutation, as retrain_pt walks them."""
    perm = np.random.default_rng(epoch_seed).permutation(n)
    for c in range(0, n, VRAM_CHUNK):
        yield perm[c:c + VRAM_CHUNK]


def train_arm(arm, cfg, data, hold, seed, lr_hi, out):
    """One arm from the run's starting weights. Returns its trackers and
    the end-of-run weights."""
    device = torch.device("cuda")
    model, arch = load_pt_model(cfg.model_path)
    model = model.to(device)
    set_train_mode(model)
    eval_model = fp16_model(cfg)
    dynamic = arm in ("Bx", "Br")
    lr_max = cfg.learning_rate if arm == "A" else lr_hi
    lr_min = lr_max * LR_MIN_FRAC
    lr = lr_min if dynamic else lr_max
    opt, scaler, (policy_lw, value_lw) = restore_opt(model, cfg, lr)
    has_gate = hasattr(model, "gate_raw")
    has_proj = hasattr(model, "project_params")
    drive = {"Bx": "val_xc0", "Br": "val_rand"}.get(arm)
    trackers = {s: Tracker(s, hold["pre"][s]["loss"]) for s in VAL_SETS}

    X, P, Y, vwht, pwht = data
    n = len(X)
    per_epoch = sum(math.ceil(len(c) / cfg.retrain_batch_size)
                    for c in batches(n, 0))
    max_epochs = MAX_EPOCHS if dynamic else 1
    lowerings, step, loss_sum, loss_n, stop = 0, 0, 0.0, 0, False
    log = open(os.path.join(out, f"evals_{arm}.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["arm", "step", "epoch", "lr", "train_loss"]
               + [f"{s}_{k}" for s in VAL_SETS for k in ("loss", "klp", "klv")]
               + ["patience", "lowerings"])
    print(f"\n[{arm}] lr_max {lr_max:.3g}  lr_min {lr_min:.3g}  "
          f"{per_epoch} steps/epoch  max {max_epochs} epoch(s)"
          + (f"  patience on {drive}" if drive else ""), flush=True)
    print(f"[{arm}] loss is the weighted holdout loss as trained: * new low, "
          f"+ kept as best within tolerance, d best = loss - prior low")
    print(f"[{arm}] pat = checks since the {drive or 'val_xc0'} low"
          + (", ** = LR cut or stop" if drive else " (info only)"))
    print(f"{'step':>6} {'ep':>5} {'lr':>8} {'train':>7} | "
          + " | ".join(f"{s:>8}  {'d best':>8} {'KLp':>6} {'KLv':>6}"
                       for s in VAL_SETS)
          + f" | {'pat':>3}   | {'s':>4} {'tot s':>6}", flush=True)
    print(f"{0:>6} {0:>5.2f} {lr:>8.2e} {'-':>7} | "
          + " | ".join(f"{hold['pre'][s]['loss']:>8.4f}  {'-':>8} "
                       f"{hold['pre'][s]['klp']:>6.3f} {hold['pre'][s]['klv']:>6.3f}"
                       for s in VAL_SETS)
          + f" | {'-':>3}   | {'-':>4} {'-':>6}", flush=True)
    run_t0 = t0 = time.time()

    for ep in range(max_epochs):
        for chunk in batches(n, seed * 1000 + ep):
            X_t = torch.from_numpy(X[chunk]).long().to(device)
            P_t = torch.from_numpy(
                stack_policies([P[i] for i in chunk], index=None)).to(device)
            Y_t = torch.from_numpy(Y[chunk]).to(device)
            vw_t = torch.from_numpy(vwht[chunk]).to(device)
            pw_t = torch.from_numpy(pwht[chunk]).to(device)
            for s in range(0, len(chunk), cfg.retrain_batch_size):
                e = s + cfg.retrain_batch_size
                xb, pb, yb = X_t[s:e], P_t[s:e], Y_t[s:e]
                if dynamic and step < WARMUP_STEPS:
                    lr = lr_min + (lr_max - lr_min) * (step + 1) / WARMUP_STEPS
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad()
                with autocast("cuda"):
                    logits, v = model(xb)
                    p_loss = (-(pb * F.log_softmax(logits, -1)).sum(-1)
                              * pw_t[s:e]).mean()
                    v_loss = (-(yb * F.log_softmax(v, -1)).sum(-1)
                              * vw_t[s:e]).mean()
                    loss = policy_lw * p_loss + value_lw * v_loss
                    if has_gate:
                        legal = pb > 0
                        g_w = torch.where(legal, GATE_LEGAL_POS_W, 1.0)
                        g_w = g_w * (g_w.shape[-1] / g_w.sum(-1, keepdim=True))
                        g_loss = F.binary_cross_entropy_with_logits(
                            model.gate_raw.float(), (~legal).float(),
                            weight=g_w, reduction="none").sum(-1).mean()
                        gate_w = GATE_SHARE * loss.detach() / g_loss.detach().clamp(
                            min=GATE_W_FLOOR)
                        loss = loss + gate_w * g_loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.retrain_clip_norm).item()
                scaler.step(opt)
                scaler.update()
                step += 1
                if np.isfinite(norm):
                    if has_proj:
                        model.project_params()
                    loss_sum += loss.item()
                    loss_n += 1

                if step % EVAL_EVERY and step != per_epoch * (ep + 1):
                    continue
                eval_model.load_state_dict(model.state_dict())
                res = {sname: evaluate(eval_model, hold[sname], policy_lw, value_lw)
                       for sname in VAL_SETS}
                cols = []
                for sname, t in trackers.items():
                    delta = res[sname]["loss"] - t.low
                    mark = t.update(res[sname]["loss"], step, model)
                    cols.append(f"{res[sname]['loss']:>8.4f}{mark} {delta:>+8.4f} "
                                f"{res[sname]['klp']:>6.3f} {res[sname]['klv']:>6.3f}")
                epochs_done = step / per_epoch
                pat = trackers[drive or "val_xc0"].bad
                flip = bool(drive) and pat >= PATIENCE
                train_loss = loss_sum / max(1, loss_n)
                loss_sum, loss_n = 0.0, 0
                print(f"{step:>6} {epochs_done:>5.2f} {lr:>8.2e} {train_loss:>7.4f} | "
                      + " | ".join(cols)
                      + f" | {pat:>3}{'**' if flip else '  '} | "
                      f"{time.time() - t0:>4.0f} {time.time() - run_t0:>6.0f}",
                      flush=True)
                t0 = time.time()
                w.writerow([arm, step, round(epochs_done, 3), lr, train_loss]
                           + [res[sname][k] for sname in VAL_SETS
                              for k in ("loss", "klp", "klv")]
                           + [pat, lowerings])
                log.flush()
                if flip:
                    if lowerings < MAX_LOWERINGS:
                        lowerings += 1
                        trackers[drive].bad = 0
                        print(f"[{arm}] {drive} stalled {PATIENCE} checks: lr "
                              f"{lr:.2e} -> {lr / math.sqrt(1 / LR_MIN_FRAC):.2e}  "
                              f"({lowerings}/{MAX_LOWERINGS})", flush=True)
                        lr /= math.sqrt(1 / LR_MIN_FRAC)
                    elif epochs_done >= MIN_EPOCHS:
                        print(f"[{arm}] {drive} stalled at the LR floor, "
                              f"stopping at step {step}", flush=True)
                        stop = True
                        break
            if stop:
                break
        if stop:
            break
    log.close()
    end_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model, opt, eval_model
    torch.cuda.empty_cache()
    return trackers, end_state, arch, (policy_lw, value_lw)


def score_state(cfg, state, h, lws):
    model = fp16_model(cfg, state)
    res = evaluate(model, h, *lws)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--arms", default="A,C,Bx,Br")
    args = ap.parse_args()

    run_dir = os.path.join(RUNS_DIR, args.run)
    cfg = Config.from_yaml(os.path.join(run_dir, "config.yaml"))
    lr_hi = LR_HI
    out = os.path.join(OUT_ROOT, time.strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}")
    assert os.path.commonpath([out, OUT_ROOT]) == OUT_ROOT
    os.makedirs(os.path.join(out, "models"))
    print(f"[sim] {args.run} seed {args.seed} -> {out}", flush=True)
    print("[sim] selfplay must be stopped: a live retrain moves these shards",
          flush=True)

    t0 = time.time()
    ckpts = [cfg.model_path[:-3] + ".pt",
             pt_opt_state_path(cfg.model_path, cfg.run_dir),
             pt_scaler_state_path(cfg.model_path, cfg.run_dir),
             retrain_lw_state_path(cfg.model_path, cfg.run_dir)]
    paths, records, sets = draw(run_dir, args.seed)
    watched = file_stamps(paths + ckpts)
    random.Random(args.seed).shuffle(records)
    X, P, Y, vwht, pwht = records_to_arrays(records)
    del records
    vwht = normalize_bounded_mean(vwht).astype(np.float32)
    pwht = normalize_bounded_mean(pwht).astype(np.float32)
    data = (X, P, Y, vwht, pwht)
    hold = {s: holdout_tensors(r) for s, r in sets.items()}
    print(f"[sim] {RETRAIN_PRIMARY_SHARDS + HOLDOUT_SHARDS} primary + "
          f"{RETRAIN_REPLAY_SHARDS} replay shards: train {len(X):,}, "
          + ", ".join(f"{s} {len(r):,}" for s, r in sets.items())
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(out, "shards.txt"), "w") as f:
        f.write("\n".join(paths) + "\n")

    model = fp16_model(cfg)
    lws = retrain_lw_from_state(torch.load(
        retrain_lw_state_path(cfg.model_path, cfg.run_dir), map_location="cpu"))
    hold["pre"] = {s: evaluate(model, hold[s], *lws) for s in VAL_SETS + ("test",)}
    del model
    torch.cuda.empty_cache()
    print("[sim] pre-retrain: " + "  ".join(
        f"{s} {v['loss']:.4f} KLp {v['klp']:.4f} KLv {v['klv']:.4f}"
        for s, v in hold["pre"].items()), flush=True)

    rows = [("pre-retrain", "-", hold["pre"]["test"])]
    for arm in args.arms.split(","):
        trackers, end_state, arch, lws = train_arm(
            arm, cfg, data, hold, args.seed, lr_hi, out)
        picks = []
        if arm in ("A", "C"):
            picks.append((f"{arm} end of epoch 1", "-", end_state))
            for s, t in trackers.items():
                if t.best_state is not None:
                    picks.append((f"{arm} best {s}", t.best_step, t.best_state))
        else:
            t = trackers["val_xc0" if arm == "Bx" else "val_rand"]
            # no accepted eval leaves the starting weights as the best
            picks.append((f"{arm} best", t.best_step, t.best_state))
        for name, at, state in picks:
            if state is not None:
                path = os.path.join(out, "models",
                                    name.replace(" ", "_") + ".pt")
                torch.save({"model": state, "arch": arch}, path)
            rows.append((name, at, score_state(cfg, state, hold["test"], lws)))

    print(f"\n[sim] test set ({len(sets['test']):,} xc0)")
    print(f"{'model':<26}{'step':>7}{'loss':>9}{'KLp':>9}{'KLv':>9}")
    with open(os.path.join(out, "test_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "loss", "klp", "klv"])
        for name, at, r in rows:
            print(f"{name:<26}{str(at):>7}{r['loss']:>9.4f}{r['klp']:>9.4f}"
                  f"{r['klv']:>9.4f}")
            w.writerow([name, at, r["loss"], r["klp"], r["klv"]])

    changed = [p for p, s in watched.items() if file_stamps([p])[p] != s]
    print(f"\n[sim] source files unchanged: {len(watched) - len(changed)} of "
          f"{len(watched)}" + ("" if not changed else f"  CHANGED: {changed}"))


if __name__ == "__main__":
    main()
