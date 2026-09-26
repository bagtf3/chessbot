"""OOM-robust PyTorch bootstrap trainer, sparse pkl.gz shards (no TensorFlow).

Trains directly on the 4-tuple sparse shards produced by build_pretrain_shards.py
(xc0h, sparse_policy, wdl_target, source). Data is served by an async 256k-record
pool: a loader thread cycles shards into a ring buffer, a prefetch thread draws
random 10240-record epoch bundles and keeps the next one ready. Everything else
(cosine LR, SWA, resume, unified eval_progress.csv, checkpoint/export) mirrors the
tfrecord trainer; this is meant to replace it once proven.

Usage:
    python scripts/train/bootstrap_model_async_sparse_pkl.py
        [--model m13-7-8c8t-d256] [--run-tag TAG | --run-dir DIR]
        [--shard-dir DIR] [--max-epoch 26001] [--val-shards 20] [--swa]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import pickle
import queue
import random
import re
import threading
import time

import numpy as np
import pandas as pd
import scipy.special

import math

from chessbot.pretrain import (
    EPOCH_SIZE, VAL_FRACTION, LR_DECAY_EPOCHS, LR_STEP_SIZE,
    lr_for_epoch, save_plot, split_train_val as split_train_val_frac,
)
from chessbot.model import VARIANTS, PT_BUILDERS
from chessbot.replay_buffer import POLICY_DIM

PT_BATCH_SIZE      = 512
PT_STEPS_PER_EPOCH = EPOCH_SIZE // PT_BATCH_SIZE   # 20
PT_ADAM_BETA2      = 0.95
PROFILE_STATE      = {"on": os.environ.get("TRAIN_PROFILE") == "1", "done": False}
DROPOUT_DUTY       = 0.25   # per-site per-step chance dropout runs, for models with set_dropout
# smartgate illegality BCE on m.gate_raw (target = policy_t == 0): a legal elem
# weighs GATE_LEGAL_POS_W x an illegal one, weights rescaled per row to sum to
# 1858; the term is pinned to GATE_SHARE * (policy + value) while gate_loss is
# above GATE_W_FLOOR, then holds that weight fixed so residual errors aren't amplified
GATE_LEGAL_POS_W   = 5.0
GATE_SHARE         = 0.05
GATE_W_FLOOR       = 5.0
PT_WEIGHT_DECAY    = 0.008   # AdamW decoupled; applied to 2D+ weights only

BUFFER_CAP     = 256_000       # shuffle pool high-water mark
REFILL_BAND    = 56_000        # refill once the pool drains below high - band (-> 200k)
VAL_BUFFER_CAP = 96_000

# cosine LR floor and ceiling
LR_MIN = 1e-4
LR_MAX = 1e-3
LR_HOLD_EPOCHS = 1000   # flat at LR_MAX after warmup until here, then cosine
# reforge (--reforge-from EPOCH): linear LR_MIN -> lr_mid over REFORGE_RAMP_EPOCHS,
# no hold, then stepped cosine lr_mid -> LR_MIN, flat at LR_MIN for the last
# LR_DECAY_EPOCHS
REFORGE_LR_MID      = 3e-4
REFORGE_RAMP_EPOCHS = 100

CHECKPOINT_EVERY = 200
HARD_SAVE_EVERY = 2000   # persistent checkpoint, never overwritten/deleted; manual cleanup only
SAVE_PLOTS = False   # progress png is slow to render; csv has everything
SWA_WINDOW = 100
SWA_EVERY  = 20

DEFAULT_MODEL     = "m13-7-8c8t-d256"
DEFAULT_RUN_TAG   = "val_test_multi"
DEFAULT_MAX_EPOCH = 26001
DEFAULT_SHARD_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\pretrain_shards"

# per-draw subsampling (train pool only): drawish positions (soft D past the
# threshold) are dropped with DRAW_DROP_P, then decided positions (any WDL
# element past DECIDED_THRESH) with DECIDED_DROP_P. Records are revisited across
# passes, so a drop is not permanent. The drop is the only downweighting.
DRAW_P_THRESH  = 0.6
DRAW_DROP_P    = 0.175
DECIDED_THRESH = 0.9
DECIDED_DROP_P = 0.125
DRAW_OVERDRAW  = 1.15   # pool must hold this many epochs before a draw

GRAD_REPORT_EVERY = 10   # report epochs: full grad scan + extremes printout
# grad scanning (nan_to_num + clip) arms off after CLIP_CALM_STEPS consecutive
# steps with <= CLIP_CALM_MAX clips; while off it runs only on report epochs and
# re-arms if a report epoch would have clipped >= CLIP_REARM_CLIPS steps
CLIP_CALM_STEPS  = 1000
CLIP_CALM_MAX    = 2
CLIP_REARM_CLIPS = 10
FP16_MIN = 5.960464477539063e-8
FP16_MAX = 65504.0
HOTSPOT_SCAN_EVERY = 20   # hotspot stats + gate EMAs are tracked and printed on these epochs only
EVAL_EVERY = 20

UNIFIED_COLS = [
    "model_epoch", "value_mse", "value_corr", "value_ce", "policy_ce",
    "chess_loss", "uniform_ce", "ce_gain", "top1_exact", "top1_mass", "top3_mass",
    "top5_mass", "n_samples", "mass_on_legal", "avg_top_prob",
    "avg_top_prob_target",
    "train_loss", "gn_mean", "exp_prob_model", "exp_prob_uniform",
    "prob_on_others",
]
# explicit float dtypes so an all-NaN column (e.g. train_loss at epoch 0) never
# lands as object and trips pandas' concat dtype-inference warning
EVAL_DTYPES = {c: "float64" for c in UNIFIED_COLS if c not in ("model_epoch", "n_samples")}


# ---------------------------------------------------------------------------
# Checkpoint / CSV utilities
# ---------------------------------------------------------------------------

def ckpt_path(run_dir, name, epoch):
    return os.path.join(run_dir, f"{name}_pt_ckpt{epoch:04d}.pt")


def hard_ckpt_path(run_dir, name, epoch):
    return os.path.join(run_dir, f"{name}_pt_hard{epoch:04d}.pt")


def swa_ckpt_path(run_dir, name, epoch):
    return os.path.join(run_dir, f"{name}_pt_swa{epoch:04d}.pt")


def swa_epochs(max_epoch):
    final = max_epoch - 1
    eps = [final - k * SWA_EVERY for k in range(SWA_WINDOW // SWA_EVERY)]
    return sorted(e for e in eps if e >= 0)


def average_state_dicts(sds):
    import torch
    out = {}
    for k in sds[0]:
        vals = [sd[k] for sd in sds]
        if vals[0].is_floating_point():
            out[k] = torch.stack([v.float() for v in vals]).mean(0).to(vals[0].dtype)
        else:
            out[k] = vals[-1]
    return out


def normalize_progress_df(df):
    if "model_epoch" not in df.columns and "training_epoch" in df.columns:
        df = df.rename(columns={"training_epoch": "model_epoch"})
    extra = [c for c in df.columns if c not in UNIFIED_COLS]
    df = df.reindex(columns=UNIFIED_COLS + extra)
    df["chess_loss"] = df["chess_loss"].fillna(df["policy_ce"] + 4.0 * df["value_ce"])
    return df


def save_progress_csv(df, path):
    try:
        df.round(4).to_csv(path, index=False)
    except OSError as e:
        print(f"[csv] write skipped, file busy ({os.path.basename(path)}): {e}",
              flush=True)


def save_plot_safe(*args, **kwargs):
    if not SAVE_PLOTS:
        return
    try:
        save_plot(*args, **kwargs)
    except OSError as e:
        print(f"[plot] write skipped, file busy: {e}", flush=True)


def progress_csv_path(run_dir, name, unified):
    if unified:
        return os.path.join(run_dir, "eval_progress.csv")
    return os.path.join(run_dir, f"{name}_pt_eval_progress.csv")


def find_last_checkpoint(run_dir, name):
    pat = re.compile(rf"^{re.escape(name)}_pt_ckpt(\d+)\.pt$")
    best = -1
    try:
        for fname in os.listdir(run_dir):
            m = pat.match(fname)
            if m:
                best = max(best, int(m.group(1)))
    except FileNotFoundError:
        pass
    return best


def delete_old_checkpoints(run_dir, name, keep_epoch):
    pat = re.compile(rf"^{re.escape(name)}_pt_ckpt(\d+)\.pt$")
    try:
        for fname in os.listdir(run_dir):
            m = pat.match(fname)
            if m and int(m.group(1)) != keep_epoch:
                os.remove(os.path.join(run_dir, fname))
                print(f"[ckpt] removed old checkpoint {fname}")
    except FileNotFoundError:
        pass


def trim_progress_csv(progress_file, max_training_epoch, unified=False):
    if not os.path.exists(progress_file):
        return
    df = pd.read_csv(progress_file)
    epoch_col = next(
        (c for c in ("model_epoch", "training_epoch") if c in df.columns), None)
    if epoch_col is None:
        if not unified:
            os.remove(progress_file)
        return
    df = df[df[epoch_col] <= max_training_epoch]
    if df.empty and not unified:
        os.remove(progress_file)
    else:
        df.to_csv(progress_file, index=False)


def get_resume_epoch(run_dir, name, progress_file, unified=False):
    os.makedirs(run_dir, exist_ok=True)
    last_ckpt = find_last_checkpoint(run_dir, name)
    if last_ckpt < 0:
        if os.path.exists(progress_file):
            bak = progress_file + ".bak"
            os.replace(progress_file, bak)
            print(f"[recovery] no checkpoints found; fresh start, "
                  f"old csv -> {os.path.basename(bak)}")
        return 0
    trim_progress_csv(progress_file, last_ckpt, unified=unified)
    print(f"[recovery] last checkpoint epoch={last_ckpt} -> resuming {last_ckpt + 1}")
    return last_ckpt + 1


def reforge_lr(ep, start, max_epoch, lr_mid, ramp):
    k = ep - start
    if k <= ramp:
        return LR_MIN + (lr_mid - LR_MIN) * max(k, 0) / ramp
    decay_end = max_epoch - LR_DECAY_EPOCHS
    if ep >= decay_end:
        return LR_MIN
    total = max(1, (decay_end - start - ramp) // LR_STEP_SIZE)
    t = ((k - ramp) // LR_STEP_SIZE) / total
    return LR_MIN + 0.5 * (lr_mid - LR_MIN) * (1.0 + math.cos(math.pi * t))


def lr_at(ep, wargs):
    if wargs.get("reforge_from") is not None:
        return reforge_lr(ep, wargs["reforge_from"], wargs["max_epoch"],
                          wargs["lr_mid"], wargs["ramp_epochs"])
    return lr_for_epoch(ep, wargs["max_epoch"], lr_min=LR_MIN, lr_max=LR_MAX,
                        hold_epochs=LR_HOLD_EPOCHS)


def retrain_opt_state(model, opt_sd):
    """AdamW decay/no-decay state -> retrain_pt's single-group Adam layout."""
    import torch
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    src = torch.optim.AdamW([{"params": decay}, {"params": no_decay}])
    src.load_state_dict(opt_sd)
    dst = torch.optim.Adam(model.parameters())
    for p in model.parameters():
        if p in src.state:
            dst.state[p] = src.state[p]
    return dst.state_dict()


def make_adam(model, lr):
    import torch
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)   # no decay on bias/norm
    groups = [
        {"params": decay,    "weight_decay": PT_WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, PT_ADAM_BETA2), fused=True)


def saved_param_ids(model, saved, old_sd):
    """{param name: saved optimizer id} for the model that wrote `saved`. Walks
    old_sd in save order (= named_parameters order, buffers interleaved); groups
    follow make_adam (decay = ndim >= 2). A key still in today's model is a param
    iff it is one today; a removed key counts as a param when its shape matches
    the next saved Adam state in its group."""
    params = dict(model.named_parameters())
    buffers = set(model.state_dict()) - set(params)
    ids = [list(spg["params"]) for spg in saved["param_groups"]]
    pos = [0] * len(ids)
    out = {}
    for k, t in old_sd.items():
        g = 0 if t.ndim >= 2 else 1
        if pos[g] >= len(ids[g]):
            continue
        sid = ids[g][pos[g]]
        if k in params:
            is_param = True
        elif k in buffers:
            is_param = False
        else:
            st = saved["state"].get(sid)
            is_param = st is not None and st["exp_avg"].shape == t.shape
        if is_param:
            out[k] = sid
            pos[g] += 1
    if pos != [len(g) for g in ids]:
        raise RuntimeError(f"optimizer remap walked {pos} of {[len(g) for g in ids]}")
    return out


def remap_opt_state(model, opt, saved, old_sd):
    """Realign saved optimizer state by param name across added and removed
    params. Params new since the save start fresh; removed ones are dropped."""
    name_of = {id(p): n for n, p in model.named_parameters()}
    old_id = saved_param_ids(model, saved, old_sd)
    state, groups, nid = {}, [], 0
    for pg, spg in zip(opt.param_groups, saved["param_groups"]):
        ids = []
        for p in pg["params"]:
            sid = old_id.get(name_of[id(p)])
            if sid is not None and sid in saved["state"]:
                state[nid] = saved["state"][sid]
            ids.append(nid)
            nid += 1
        groups.append({**spg, "params": ids})
    return {"state": state, "param_groups": groups}


def load_matching(model, sd):
    """Load sd by name. A tensor whose shape changed, or that sd lacks, keeps its
    fresh init; sd entries the model lacks are dropped. The shape check runs as
    a per-module pre-hook, so it sees keys after any module's own load
    conversion. Returns (reshaped, missing, unexpected) key lists."""
    reshaped = []

    def drop_reshaped(module, state_dict, prefix, *args):
        own = list(module.named_parameters(recurse=False))
        own += list(module.named_buffers(recurse=False))
        for local, t in own:
            key = prefix + local
            if key in state_dict and state_dict[key].shape != t.shape:
                reshaped.append(key)
                del state_dict[key]

    handles = [m.register_load_state_dict_pre_hook(drop_reshaped)
               for m in model.modules()]
    res = model.load_state_dict(sd, strict=False)
    for h in handles:
        h.remove()
    missing = [k for k in res.missing_keys if k not in reshaped]
    return reshaped, missing, list(res.unexpected_keys)


def print_key_list(label, keys, limit=12):
    if keys:
        shown = ", ".join(keys[:limit]) + (" ..." if len(keys) > limit else "")
        print(f"[worker] {label} ({len(keys)}): {shown}", flush=True)


def load_pt_model(path, name, device, lr):
    import torch
    from torch.amp import GradScaler
    cfg    = VARIANTS[name]
    model  = PT_BUILDERS[name](cfg).to(device)
    opt    = make_adam(model, lr)
    scaler = GradScaler("cuda", init_scale=2 ** 15)
    ckpt = torch.load(path, map_location="cpu")
    reshaped, missing, unexpected = load_matching(model, ckpt["model"])
    print_key_list("re-init, shape changed", reshaped)
    print_key_list("re-init, new in model", missing)
    print_key_list("dropped, not in model", unexpected)
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if "optimizer" in ckpt:
        wds = [pg["weight_decay"] for pg in opt.param_groups]
        saved = ckpt["optimizer"]
        sizes = [len(pg["params"]) for pg in opt.param_groups]
        if unexpected or sizes != [len(pg["params"]) for pg in saved["param_groups"]]:
            saved = remap_opt_state(model, opt, saved, ckpt["model"])
            print("[worker] optimizer state remapped by name for added/removed params")
        opt.load_state_dict(saved)
        for mod in model.modules():   # sign-converted on load: stale momentum
            if getattr(mod, "converted", False):
                for p in mod.parameters():
                    opt.state.pop(p, None)
                print(f"[worker] reset Adam state for converted {type(mod).__name__}")
        for pg, wd in zip(opt.param_groups, wds):
            pg["lr"] = lr
            pg["weight_decay"] = wd
            pg["fused"] = True
            pg["foreach"] = None
            for p in pg["params"]:   # param reshaped since the save: fresh Adam state
                if p in opt.state and opt.state[p]["exp_avg"].shape != p.shape:
                    del opt.state[p]
                    print(f"[worker] reset Adam state for reshaped param {tuple(p.shape)}")
        for p, st in opt.state.items():   # fused Adam wants step as fp32 on the param device
            if "step" in st:
                st["step"] = st["step"].to(device=p.device, dtype=torch.float32)
    return model, opt, scaler


def save_pt_ckpt(model, opt, scaler, epoch, name, path):
    import torch
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "arch": name,
    }, path)
    print(f"[ckpt] epoch {epoch} -> {path}")


def pt_clipnorm_for_epoch(ep):
    if ep < 10:  return 1.0
    if ep < 20:  return 2.5
    if ep < 30:  return 5.0
    return 10.0


def ema_summary(ema):
    """(min, max, mean, p25, p50, p75) of an EMA tensor."""
    import torch
    e = ema.detach().float().flatten()
    p25, p50, p75 = torch.quantile(e, torch.tensor([0.25, 0.5, 0.75], device=e.device)).tolist()
    return np.array([float(e.min()), float(e.max()), float(e.mean()), p25, p50, p75])


def print_gate_ema(model):
    """Gate-EMA six-number readout for mean/std/range, each averaged across
    every block carrying gate_*_ema buffers. Silent for models without any."""
    blocks = [b for b in model.modules() if hasattr(b, "gate_range_ema")]
    if not blocks:
        return
    for label in ("mean", "std", "range"):
        s = np.mean([ema_summary(getattr(b, f"gate_{label}_ema")) for b in blocks], axis=0)
        print(f"  GE gate-EMA {label:<5}  x{len(blocks)}  min {s[0]:5.2f}  max {s[1]:5.2f}"
              f"  mean {s[2]:5.2f}  p25 {s[3]:5.2f}  p50 {s[4]:5.2f}  p75 {s[5]:5.2f}",
              flush=True)


def print_attn_gate_ema(model):
    """AG rows: attention-gate EMAs (1 = identity) averaged across gated blocks."""
    mods = [m for m in model.modules() if hasattr(m, "ag_range_ema")]
    if not mods:
        return
    for label in ("mean", "std", "range"):
        s = np.mean([ema_summary(getattr(m, f"ag_{label}_ema")) for m in mods], axis=0)
        print(f"  AG gate-EMA {label:<5}  x{len(mods)}  min {s[0]:5.2f}  max {s[1]:5.2f}"
              f"  mean {s[2]:5.2f}  p25 {s[3]:5.2f}  p50 {s[4]:5.2f}  p75 {s[5]:5.2f}",
              flush=True)


def print_prelu(model):
    """PR row: every PReLU slope pooled, plus counts below 0 and at the clamps."""
    import torch
    ps = [m for m in model.modules() if isinstance(m, torch.nn.PReLU)]
    if not ps:
        return
    a = torch.cat([m.weight.detach().float() for m in ps])
    print(f"  PR slope          x{len(ps)}  min {a.min():6.3f}  max {a.max():6.3f}"
          f"  mean {a.mean():6.3f}  std {a.std():6.3f}  <0 {int((a < 0).sum())}"
          f"  floor {int((a <= -0.499).sum())}  ceil {int((a >= 0.499).sum())}",
          flush=True)


def gfmt(v):
    return f"{v:.6f}" if v == 0 or abs(v) >= 1e-5 else f"{v:.2e}"


def grad_report(model, backward_scale):
    """Per-parameter grad extremes after unscale_: min/max L2 over 2D+ weights,
    min nonzero / max scaled |grad| with their names, and the count of scaled
    grads that are nonzero yet at or below the fp16 subnormal floor."""
    import torch
    names, l2s = [], []
    min_grad, min_name = float("inf"), ""
    max_grad, max_name = 0.0, ""
    n_fp16_min = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if p.ndim >= 2:
            names.append(name)
            l2s.append(g.norm())
        ga = g.abs() * backward_scale
        nz = ga[ga > 0]
        n_fp16_min += int((nz <= FP16_MIN).sum())
        gmax = float(ga.max())
        if gmax > max_grad:
            max_grad, max_name = gmax, name
        if nz.numel():
            gmin = float(nz.min())
            if gmin < min_grad:
                min_grad, min_name = gmin, name
    pgn = torch.stack(l2s)
    lo, hi = int(pgn.argmin()), int(pgn.argmax())
    return {
        "min_l2": float(pgn[lo]), "min_l2_name": names[lo],
        "max_l2": float(pgn[hi]), "max_l2_name": names[hi],
        "min_grad": min_grad, "min_grad_name": min_name,
        "max_grad": max_grad, "max_grad_name": max_name,
        "n_fp16_min": n_fp16_min,
    }


def print_grad_report(r):
    min_box = "[x]" if r["min_grad"] < FP16_MIN else "[ ]"
    max_box = "[x]" if r["max_grad"] > FP16_MAX else "[ ]"
    nw = max(len(r["min_l2_name"]), len(r["min_grad_name"]))
    vw = max(len(gfmt(r["max_l2"])), len(gfmt(r["max_grad"])))
    print(f"  grad L2         {gfmt(r['min_l2'])} {r['min_l2_name']:<{nw}}"
          f"  ->     {gfmt(r['max_l2']):<{vw}}  {r['max_l2_name']}", flush=True)
    print(f"  scaled grad {min_box} {gfmt(r['min_grad'])} {r['min_grad_name']:<{nw}}"
          f"  -> {max_box} {gfmt(r['max_grad']):<{vw}}  {r['max_grad_name']}"
          f"  n<=fp16_min {r['n_fp16_min']}", flush=True)


# ---------------------------------------------------------------------------
# Async sparse-shard data layer
# ---------------------------------------------------------------------------

def densify_batch(sparse_list):
    """Scatter a list of (idx, vals) sparse policies into one [n, POLICY_DIM]
    array with a single fancy-index assignment."""
    lens = np.fromiter((len(sp[0]) for sp in sparse_list), dtype=np.int64,
                       count=len(sparse_list))
    rows = np.repeat(np.arange(len(sparse_list)), lens)
    cols = np.concatenate([np.asarray(sp[0], dtype=np.int64) for sp in sparse_list])
    vals = np.concatenate([np.asarray(sp[1], dtype=np.float32) for sp in sparse_list])
    d = np.zeros((len(sparse_list), POLICY_DIM), dtype=np.float32)
    d[rows, cols] = vals
    return d


def extract_shard(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)   # list of (xc0h, policy_sparse, wdl, src)


class AsyncShardPool:
    """Async draining shuffle buffer.

    A loader thread streams shards (reshuffled each pass) into a pool up to
    `high`, refilling whenever it drains below `low`. A prefetch thread draws a
    random epoch_size WITHOUT replacement and, when `drain=True`, REMOVES them
    from the pool -- so over one pass of the shard stream every record is served
    exactly once (uniform exposure), and the run's total exposure is just the
    number of passes. Val drains the same way so eval marches through the whole
    held-out set over the run. `drain=False` leaves a static pool that resamples
    without removal. Both threads run in the background and stage the next
    `n_ready` bundles.
    """

    def __init__(self, shard_files, high, low, epoch_size, drain,
                 n_ready=2, name="train", subsample=True):
        self.files      = list(shard_files)
        self.subsample  = subsample
        self.high       = high
        self.low        = max(epoch_size, min(low, high))
        self.epoch_size = epoch_size
        self.draw_size  = int(epoch_size * DRAW_OVERDRAW)
        self.drain      = drain
        self.name       = name
        self.pool = []
        self.meta = np.zeros((0, 2), dtype=np.float32)   # per record: (D, max WDL)
        self.size = 0
        self.lock = threading.Lock()
        self.loader_done = threading.Event()
        self.stop_evt    = threading.Event()
        self.ready = queue.Queue(maxsize=n_ready)
        self.loader_t = threading.Thread(target=self.loader, daemon=True)
        self.pref_t   = threading.Thread(target=self.prefetch, daemon=True)

    def start(self):
        self.loader_t.start()
        self.pref_t.start()

    def shard_stream(self):
        while not self.stop_evt.is_set():
            order = list(range(len(self.files)))
            random.shuffle(order)
            for i in order:
                if self.stop_evt.is_set():
                    return
                try:
                    yield extract_shard(self.files[i])
                except Exception as e:
                    print(f"[pool-{self.name}] read fail "
                          f"{os.path.basename(self.files[i])}: {e}", flush=True)

    def fill_to_high(self, stream):
        for recs in stream:
            if self.stop_evt.is_set():
                return
            wdl = np.array([r[2] for r in recs], dtype=np.float32)
            meta = np.stack([wdl[:, 1], wdl.max(1)], axis=1)
            with self.lock:
                self.pool.extend(recs)
                self.meta = np.concatenate([self.meta, meta])
                self.size = len(self.pool)
                full = self.size >= self.high
            if full:
                return

    def loader(self):
        stream = self.shard_stream()
        self.fill_to_high(stream)
        if not self.drain:
            self.loader_done.set()   # static pool: fill once, no rotation
            return
        while not self.stop_evt.is_set():
            with self.lock:
                below = self.size < self.low
            if below:
                self.fill_to_high(stream)
            else:
                time.sleep(0.05)

    def sample_bundle(self):
        """Walk a random permutation of the pool, dropping drawish records with
        DRAW_DROP_P and decided ones with DECIDED_DROP_P (subsample=True only),
        until epoch_size survivors. Everything walked (kept or dropped) leaves the
        pool; the untouched tail stays. Pure numpy up to the final record gather."""
        with self.lock:
            n = self.size
            perm = np.random.permutation(n)
            if self.subsample:
                d, mx = self.meta[perm, 0], self.meta[perm, 1]
                dropped = (d > DRAW_P_THRESH) & (np.random.rand(n) < DRAW_DROP_P)
                dropped |= ~dropped & (mx > DECIDED_THRESH) & (np.random.rand(n) < DECIDED_DROP_P)
                surv = np.flatnonzero(~dropped)[:self.epoch_size]
            else:
                surv = np.arange(min(self.epoch_size, n))
            walked = surv[-1] + 1
            keep = perm[surv]
            recs = [self.pool[i] for i in keep]
            if self.drain:
                stay = np.ones(n, dtype=bool)
                stay[perm[:walked]] = False
                self.pool = [self.pool[i] for i in np.flatnonzero(stay)]
                self.meta = self.meta[stay]
                self.size = len(self.pool)
        enc = np.stack([r[0] for r in recs]).astype(np.int64)
        pol = densify_batch([r[1] for r in recs])
        val = np.stack([r[2] for r in recs]).astype(np.float32)
        w   = np.ones(len(recs), dtype=np.float32)
        return {"enc_in": enc, "policy_logits": pol, "value_out": val, "weight": w}

    def prefetch(self):
        while not self.stop_evt.is_set():
            with self.lock:
                enough = self.size >= self.draw_size
            if enough or self.loader_done.is_set():
                break
            time.sleep(0.1)
        while not self.stop_evt.is_set():
            while not self.stop_evt.is_set():
                with self.lock:
                    enough = self.size >= self.draw_size
                if enough or self.loader_done.is_set():
                    break
                time.sleep(0.05)   # loader catching back up after a drain
            b = self.sample_bundle()
            while not self.stop_evt.is_set():
                try:
                    self.ready.put(b, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def get(self):
        return self.ready.get()

    def stop(self):
        self.stop_evt.set()
        try:
            while True:
                self.ready.get_nowait()
        except queue.Empty:
            pass


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def new_clip_state():
    return {"on": True, "window_steps": 0, "window_clips": 0}


def update_clip_state(state, clipped, report):
    """Advance the arm/disarm window with this epoch's per-step clip flags.
    Returns a log line when the state flips, else None."""
    if state["on"]:
        for c in clipped:
            state["window_steps"] += 1
            state["window_clips"] += int(c)
            if state["window_clips"] > CLIP_CALM_MAX:
                state["window_steps"] = state["window_clips"] = 0
        if state["window_steps"] >= CLIP_CALM_STEPS:
            state["on"] = False
            return (f"[clip] off: {CLIP_CALM_STEPS} steps with "
                    f"<= {CLIP_CALM_MAX} clips; scanning every {GRAD_REPORT_EVERY} epochs")
    elif report and int(clipped.sum()) >= CLIP_REARM_CLIPS:
        state["on"] = True
        state["window_steps"] = state["window_clips"] = 0
        return f"[clip] on: {int(clipped.sum())}/{len(clipped)} steps clipped on a report epoch"
    return None



def print_profile(prof):
    """on_trace_ready for the one-shot TRAIN_PROFILE window."""
    PROFILE_STATE["done"] = True
    ka = prof.key_averages()
    print(ka.table(sort_by="cuda_time_total", row_limit=30), flush=True)
    print(ka.table(sort_by="self_cpu_time_total", row_limit=20), flush=True)


def fit_epoch(model, opt, scaler, bundle, max_norm, device, clip_state,
              report=False):
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    enc      = torch.as_tensor(bundle["enc_in"],        dtype=torch.long).to(device)
    policy_t = torch.as_tensor(bundle["policy_logits"], dtype=torch.float32).to(device)
    value_t  = torch.as_tensor(bundle["value_out"],     dtype=torch.float32).to(device)
    w        = torch.as_tensor(bundle["weight"],        dtype=torch.float32).to(device)

    n    = enc.shape[0]
    perm = torch.randperm(n, device=device)
    model.train()
    losses     = []   # (p, v) per step, kept on device; one sync at the end
    grad_norms = []
    n_batches  = 0
    grad_rep   = None
    scan = clip_state["on"] or report
    has_hot = hasattr(model, "hotspot_end_step")
    has_drop = hasattr(model, "set_dropout")
    has_gate = hasattr(model, "gate_raw")
    has_proj = hasattr(model, "project_params")
    g_losses = []
    if PROFILE_STATE["on"] and not PROFILE_STATE["done"] and "prof" not in PROFILE_STATE:
        from torch.profiler import profile, schedule, ProfilerActivity
        PROFILE_STATE["prof"] = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=21, active=10, repeat=1),
            on_trace_ready=print_profile,
        )
        PROFILE_STATE["prof"].__enter__()
    prof = PROFILE_STATE.get("prof")

    for start in range(0, n, PT_BATCH_SIZE):
        if prof is not None:
            prof.step()
        idx = perm[start:start + PT_BATCH_SIZE]
        if has_drop:
            model.set_dropout(DROPOUT_DUTY)
        opt.zero_grad(set_to_none=True)
        with autocast("cuda"):
            pol, val = model(enc[idx])
            p_loss = (F.cross_entropy(pol, policy_t[idx], reduction="none") * w[idx]).mean()
            v_loss = (F.cross_entropy(val, value_t[idx], reduction="none") * w[idx]).mean()
            loss = p_loss + v_loss
            if has_gate:
                legal = policy_t[idx] > 0
                g_elem_w = torch.where(legal, GATE_LEGAL_POS_W, 1.0)
                g_elem_w = g_elem_w * (g_elem_w.shape[-1] / g_elem_w.sum(-1, keepdim=True))
                g_loss = F.binary_cross_entropy_with_logits(
                    model.gate_raw.float(),
                    (~legal).float(),
                    weight=g_elem_w,
                    reduction="none",
                ).sum(-1).mean()
                gate_w = GATE_SHARE * loss.detach() / g_loss.detach().clamp(min=GATE_W_FLOOR)
                loss = loss + gate_w * g_loss
                g_losses.append(g_loss.detach())
        if has_hot:
            model.hotspot_end_step()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)   # records found_inf here, so the scrub below can't mask it
        for p in model.parameters():   # always on, never gated by clip_state
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if scan:
            if report and start + PT_BATCH_SIZE >= n:
                grad_rep = grad_report(model, scaler.get_scale())
            grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
        scaler.step(opt)
        scaler.update()
        if has_proj:
            model.project_params()
        losses.append((p_loss.detach(), v_loss.detach()))
        n_batches += 1

    if prof is not None and PROFILE_STATE["done"]:
        prof.__exit__(None, None, None)
        del PROFILE_STATE["prof"]

    pv = torch.tensor(losses, device=device).float().cpu().numpy()
    total_p, total_v = pv[:, 0].mean(), pv[:, 1].mean()
    if scan:
        gn = torch.stack(grad_norms).float().cpu().numpy()
        clipped = (gn > max_norm) | np.isnan(gn)
        flip = update_clip_state(clip_state, clipped, report)
    else:
        gn, clipped, flip = np.array([np.nan]), np.zeros(0, dtype=bool), None
    grad_stats = {
        "gn_mean": gn.mean(), "gn_median": np.median(gn), "gn_min": gn.min(),
        "gn_max": gn.max(), "gn_clips": int(clipped.sum()),
        "gn_steps": n_batches, "scanned": scan, "report": grad_rep, "flip": flip,
        "g_loss": torch.stack(g_losses).float().mean().item() if g_losses else None,
    }
    return total_p, total_v, total_p + total_v, grad_stats


def epoch_diagnostics(model, scaler, gns, hot_scan):
    """Grad-norm, gate-health, hotspot, gate-param and grad-report printouts on
    hotspot/report epochs."""
    if gns["scanned"] and (hot_scan or gns["report"] is not None):
        print(f"[grad norms ] mean: {gns['gn_mean']:.3f}  "
              f"median: {gns['gn_median']:.3f}  min: {gns['gn_min']:.3f}  "
              f"max: {gns['gn_max']:.3f}  clips: {gns['gn_clips']}/{gns['gn_steps']}")
    if gns["flip"]:
        print(gns["flip"], flush=True)
    if hot_scan:
        print(flush=True)
        print_gate_ema(model)
        print_attn_gate_ema(model)
        print_prelu(model)
        print(flush=True)
        model.set_hotspot_scan(False)
        stats = model.hotspot_stats()
        rows = {k: (f"{v[2]:.2f}", f"{v[1]:.2f}", f"{v[0]:.2f}",
                    ", ".join(map(str, v[3])))
                for k, v in stats.items()}
        nw = max(len(k) for k in rows) + 1
        w = [max(len(r[i]) for r in rows.values()) for i in range(3)]
        for k, r in rows.items():
            print(f"[check {k:<{nw}}] avg act: {r[0]:>{w[0]}}  avg max: {r[1]:>{w[1]}}  "
                  f"abs max: {r[2]:>{w[2]}}  arg max: {r[3]}", flush=True)
        if hasattr(model, "gate_param_stats"):
            b_min, b_mean, b_max, g_scale = model.gate_param_stats()
            print(f"[gate params] bias min: {b_min:.3f}  mean: {b_mean:.3f}  "
                  f"max: {b_max:.3f}  scale: {g_scale:.3f}", flush=True)
    if gns["report"] is not None:
        print_grad_report(gns["report"])
        print(f"  grad scale      {scaler.get_scale():.0f}", flush=True)


def do_eval(model, name, epoch, bundle, eval_df, progress_file, plot_file, device,
            train_loss=None, gn_mean=None):
    import torch
    from torch.amp import autocast
    from chessbot.utils import batch_policy_metrics, print_validation

    enc = torch.as_tensor(bundle["enc_in"], dtype=torch.long).to(device)
    model.eval()
    all_pol, all_val = [], []
    with torch.no_grad(), autocast("cuda"):
        for start in range(0, enc.shape[0], PT_BATCH_SIZE):
            pol, val = model(enc[start:start + PT_BATCH_SIZE])
            all_pol.append(pol.float().cpu().numpy())
            all_val.append(val.float().cpu().numpy())

    pol_preds  = np.concatenate(all_pol, axis=0)
    val_logits = np.concatenate(all_val, axis=0)
    target_wdl = bundle["value_out"]
    pstack     = bundle["policy_logits"]
    mstack     = (pstack > 0).astype(np.int32)

    val_wdl = scipy.special.softmax(val_logits, axis=1)
    val_q   = val_wdl[:, 0] - val_wdl[:, 2]
    tgt_q   = target_wdl[:, 0] - target_wdl[:, 2]

    value_mse  = np.mean((val_q - tgt_q) ** 2)
    value_corr = np.corrcoef(val_q, tgt_q)[0, 1]
    value_ce   = -np.mean(
        np.sum(target_wdl * np.log(np.clip(val_wdl, 1e-9, None)), axis=1))
    pol_stats  = batch_policy_metrics(pol_preds, pstack, mstack)

    row = {
        "model_epoch": epoch,
        "value_mse": value_mse, "value_corr": value_corr, "value_ce": value_ce,
        **pol_stats,
        "chess_loss": pol_stats["policy_ce"] + 4.0 * value_ce,
        "n_samples": int(enc.shape[0]),
        "avg_top_prob_target": float(pstack.max(axis=1).mean()),
        "train_loss": train_loss, "gn_mean": gn_mean,
    }
    new_row = pd.DataFrame([row]).reindex(columns=UNIFIED_COLS).astype(EVAL_DTYPES)
    eval_df = new_row if eval_df is None else pd.concat([eval_df, new_row], ignore_index=True)
    save_progress_csv(eval_df, progress_file)

    print_validation(epoch, {
        "value_mse": value_mse, "value_corr": value_corr, "value_ce": value_ce,
        **pol_stats,
    })
    save_plot_safe(eval_df, f"{name}  [PT]", epoch, plot_file, tgt_q, val_q)
    return eval_df, tgt_q, val_q


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker_main(wargs):
    import gc
    import torch
    from dotenv import load_dotenv
    load_dotenv()
    from chessbot.utils import format_time

    name        = wargs["name"]
    run_dir     = wargs["run_dir"]
    train_files = wargs["train_files"]
    val_files   = wargs["val_files"]
    start_epoch = wargs["start_epoch"]
    end_epoch   = wargs["end_epoch"]
    max_epoch   = wargs["max_epoch"]
    unified     = wargs.get("unified_csv", False)
    buffer_cap  = wargs.get("buffer_cap", BUFFER_CAP)
    swa_set     = set(swa_epochs(max_epoch)) if wargs.get("swa") else set()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True   # fixed shapes: batch never varies
    progress_file = wargs.get("progress_file") or progress_csv_path(run_dir, name, unified)
    plot_file     = os.path.join(run_dir, f"{name}_pt_plot.png")

    print(f"\n{'#' * 72}")
    print(f"  {name}  epochs {start_epoch}..{end_epoch - 1}  ".center(72, "#"))
    print(f"{'#' * 72}\n")

    lr0 = lr_at(start_epoch, wargs)
    if start_epoch == 0:
        from torch.amp import GradScaler
        cfg    = VARIANTS[name]
        model  = PT_BUILDERS[name](cfg).to(device)
        opt    = make_adam(model, lr0)
        scaler = GradScaler("cuda", init_scale=2 ** 15)
        print(f"[worker] fresh init  lr={lr0:.4e}")
    else:
        last_ckpt = find_last_checkpoint(run_dir, name)
        if last_ckpt < 0:
            raise RuntimeError(f"start_epoch={start_epoch} but no checkpoint in {run_dir}")
        load_from = ckpt_path(run_dir, name, last_ckpt)
        print(f"[worker] loading {load_from}")
        model, opt, scaler = load_pt_model(load_from, name, device, lr=lr0)

    eval_df = None
    if os.path.exists(progress_file):
        existing = pd.read_csv(progress_file)
        eval_df = normalize_progress_df(existing) if not existing.empty else None

    print(f"[pkl] train_shards={len(train_files)}  val_shards={len(val_files)}"
          f"  batch={PT_BATCH_SIZE}  steps/epoch={PT_STEPS_PER_EPOCH}"
          f"  buffer={buffer_cap:,}")

    train_pool = AsyncShardPool(
        train_files, high=buffer_cap, low=buffer_cap - REFILL_BAND,
        epoch_size=EPOCH_SIZE, drain=True, n_ready=2, name="train")
    val_pool = AsyncShardPool(
        val_files, high=VAL_BUFFER_CAP, low=VAL_BUFFER_CAP - REFILL_BAND,
        epoch_size=EPOCH_SIZE, drain=True, n_ready=1, name="val", subsample=False)
    train_pool.start()
    val_pool.start()


    current_lr = None
    clip_state = new_clip_state()
    begin = time.time()
    epoch_times = []
    t_fetch = t_fit = t_eval = 0.0
    total_samples = window_samples = 0
    window_loss_sum = window_gn_sum = 0.0
    window_count = window_gn_count = 0
    last_tgt_q = last_val_q = None

    for ep in range(start_epoch, end_epoch):
        target_lr = lr_at(ep, wargs)
        if target_lr != current_lr:
            current_lr = target_lr
            for pg in opt.param_groups:
                pg["lr"] = target_lr
            print(f"[lr update] epoch {ep}: lr={target_lr:.4e}")

        epoch_start = time.time()
        print("-" * 89)

        if ep % EVAL_EVERY == 0 or ep == end_epoch - 1:
            t0 = time.time()
            val_bundle = val_pool.get()
            avg_loss = window_loss_sum / window_count if window_count > 0 else None
            avg_gn   = window_gn_sum / window_gn_count if window_gn_count > 0 else None
            eval_df, last_tgt_q, last_val_q = do_eval(
                model, name, ep, val_bundle, eval_df, progress_file, plot_file,
                device, train_loss=avg_loss, gn_mean=avg_gn)
            window_loss_sum = window_gn_sum = 0.0
            window_count = window_gn_count = 0
            t_eval += time.time() - t0

        t0 = time.time()
        bundle = train_pool.get()
        t_fetch += time.time() - t0

        t0 = time.time()
        max_norm = pt_clipnorm_for_epoch(ep)
        hot_scan = ep % HOTSPOT_SCAN_EVERY == 0 and hasattr(model, "set_hotspot_scan")
        if hot_scan:
            model.set_hotspot_scan(True)
        p_loss, v_loss, t_loss, gns = fit_epoch(
            model, opt, scaler, bundle, max_norm, device, clip_state,
            report=(ep % GRAD_REPORT_EVERY == 0))
        t_fit += time.time() - t0

        n_this = int(bundle["enc_in"].shape[0])
        total_samples  += n_this
        window_samples += n_this
        window_loss_sum += t_loss
        window_count    += 1
        if gns["scanned"]:
            window_gn_sum   += gns["gn_mean"]
            window_gn_count += 1

        if ep == start_epoch and ep % EVAL_EVERY == 0 and eval_df is not None:
            eval_df.loc[eval_df.index[-1], "train_loss"] = t_loss
            eval_df.loc[eval_df.index[-1], "gn_mean"]   = gns["gn_mean"]
            save_progress_csv(eval_df, progress_file)

        g_str = f"gate_loss: {gns['g_loss']:.2f}  " if gns["g_loss"] is not None else ""
        print(f"[epoch {ep:4d}] policy_loss: {p_loss:.2f}  "
              f"value_loss: {v_loss:.2f}  {g_str}total: {t_loss:.2f}  "
              f"samples: {total_samples:,}", flush=True)
        epoch_diagnostics(model, scaler, gns, hot_scan)

        is_ckpt = (ep % CHECKPOINT_EVERY == 0 and ep > 0) or ep == end_epoch - 1 or ep == max_epoch - 1
        if is_ckpt:
            save_pt_ckpt(model, opt, scaler, ep, name, ckpt_path(run_dir, name, ep))
            delete_old_checkpoints(run_dir, name, keep_epoch=ep)
            save_plot_safe(eval_df, f"{name}  [PT]", ep, plot_file, last_tgt_q, last_val_q)

        if ep % HARD_SAVE_EVERY == 0 and ep > 0:
            save_pt_ckpt(model, opt, scaler, ep, name, hard_ckpt_path(run_dir, name, ep))
            print(f"[ckpt] hard save at epoch {ep} (persistent, not auto-deleted)")

        if ep in swa_set:
            sp = swa_ckpt_path(run_dir, name, ep)
            torch.save({"model": model.state_dict(), "arch": name}, sp)
            print(f"[swa] snapshot -> {os.path.basename(sp)}")

        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        print(f"[time check] last: {format_time(elapsed)}  "
              f"avg: {format_time(np.mean(epoch_times))}  "
              f"total: {format_time(time.time() - begin)}")

        if ep % EVAL_EVERY == 0 and ep > start_epoch:
            print(f"[timing/{EVAL_EVERY}ep] fetch: {t_fetch:.2f}s  fit: {t_fit:.2f}s  "
                  f"eval: {t_eval:.2f}s  samples_window: {window_samples:,}  "
                  f"samples_total: {total_samples:,}")
            t_fetch = t_fit = t_eval = 0.0
            window_samples = 0
        print()

    train_pool.stop()
    val_pool.stop()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[worker] {name}: block complete (epochs {start_epoch}..{end_epoch - 1})")


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

def list_shards(shard_dir):
    return sorted(glob.glob(os.path.join(shard_dir, "*.pkl.gz")))


def split_train_val_count(files, n_val, seed=0):
    r = random.Random(seed)
    files = list(files)
    r.shuffle(files)
    return files[n_val:], files[:n_val]


def seed_selfplay_buffers(run_dir, seed_files):
    """Seed the selfplay run's buffers from pretrain shards, converting our
    4-tuples to the selfplay 7-tuple format tagged SEED_SOURCE. Replay fills to
    REPLAY_SEED_SHARDS, primary to PRIMARY_SEED_SHARDS (short of the trigger so
    the pretrain->selfplay handoff is gradual), live to 80% of LIVE_BUFFER_SIZE.
    Held-out val shards are consumed first."""
    from chessbot.replay_buffer import (
        REPLAY_SEED_SHARDS, PRIMARY_SEED_SHARDS, LIVE_BUFFER_SIZE, SEED_SOURCE,
        new_shard_path, write_pkl_gz_shard, sync_live_buffer, find_live_buffer,
    )
    replay_dir  = os.path.join(run_dir, "replay_buffer")
    primary_dir = os.path.join(run_dir, "primary_buffer")
    os.makedirs(replay_dir, exist_ok=True)
    os.makedirs(primary_dir, exist_ok=True)

    src = iter(seed_files)
    status = {}

    def to_seed(recs):
        return [(r[0], None, r[1], r[2], 1.0, 1.0, SEED_SOURCE) for r in recs]

    def count_pkls(d):
        return sum(1 for f in os.listdir(d) if f.endswith(".pkl.gz"))

    def fill(out_dir, target):
        need = max(0, target - count_pkls(out_dir))
        wrote = 0
        for _ in range(need):
            path = next(src, None)
            if path is None:
                break
            write_pkl_gz_shard(to_seed(extract_shard(path)), new_shard_path(out_dir))
            wrote += 1
        return wrote

    r = fill(replay_dir, REPLAY_SEED_SHARDS)
    status["replay"] = f"{r} shards (target {REPLAY_SEED_SHARDS})"
    p = fill(primary_dir, PRIMARY_SEED_SHARDS)
    status["primary"] = f"{p} shards (target {PRIMARY_SEED_SHARDS})"

    if find_live_buffer(run_dir) is None:
        target = int(0.8 * LIVE_BUFFER_SIZE)
        recs = []
        while len(recs) < target:
            path = next(src, None)
            if path is None:
                break
            recs.extend(to_seed(extract_shard(path)))
        recs = recs[:target]
        sync_live_buffer(run_dir, recs)
        status["live"] = f"{len(recs):,} records (80% of {LIVE_BUFFER_SIZE:,})"
    else:
        status["live"] = "skipped (already present)"

    print("[seed] " + " | ".join(f"{k}: {v}" for k, v in status.items()), flush=True)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    from chessbot import SP_DIR

    parser = argparse.ArgumentParser(
        description="Sparse-pkl PyTorch bootstrap trainer for Xerces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--shard-dir", default=DEFAULT_SHARD_DIR)
    parser.add_argument("--max-epoch", type=int, default=DEFAULT_MAX_EPOCH)
    parser.add_argument("--val-shards", type=int, default=0,
                        help=f"held-out val shard count; 0 = use VAL_FRACTION "
                             f"({VAL_FRACTION:.0%}, seed-matched to pretrain)")
    parser.add_argument("--buffer-cap", type=int, default=BUFFER_CAP)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--reforge-from", type=int, default=None,
                        help="checkpoint epoch the reforge LR schedule starts at")
    parser.add_argument("--lr-mid", type=float, default=REFORGE_LR_MID)
    parser.add_argument("--ramp-epochs", type=int, default=REFORGE_RAMP_EPOCHS)
    parser.add_argument("--progress-csv", default=None,
                        help="csv filename inside the run dir (default: eval_progress.csv)")
    args = parser.parse_args()

    run_dir = args.run_dir or os.path.join(SP_DIR, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    unified = args.run_dir is not None
    name = args.model
    progress_file = (os.path.join(run_dir, args.progress_csv) if args.progress_csv
                     else progress_csv_path(run_dir, name, unified))

    print(f"[train] model={name}")
    print(f"[train] run_dir={run_dir}")
    print(f"[train] progress_csv={os.path.basename(progress_file)}")
    print(f"[train] shard_dir={args.shard_dir}")
    print(f"[train] max_epoch={args.max_epoch}  buffer={args.buffer_cap:,}")
    print(f"[train] batch={PT_BATCH_SIZE}  steps/epoch={PT_STEPS_PER_EPOCH}"
          f"  lr_range=[{LR_MIN:.1e}, {LR_MAX:.1e}]")
    if args.reforge_from is not None:
        print(f"[train] reforge from epoch {args.reforge_from}: lr {LR_MIN:.1e} -> "
              f"{args.lr_mid:.1e} over {args.ramp_epochs} epochs, cosine -> "
              f"{LR_MIN:.1e} by {args.max_epoch - LR_DECAY_EPOCHS}")

    all_files = list_shards(args.shard_dir)
    if not all_files:
        parser.error(f"no .pkl.gz shards found in {args.shard_dir}")
    if args.val_shards > 0:
        train_files, val_files = split_train_val_count(all_files, args.val_shards)
    else:
        train_files, val_files = split_train_val_frac(all_files)
    print(f"[train] train_shards={len(train_files)}  val_shards={len(val_files)}")

    resume = get_resume_epoch(run_dir, name, progress_file, unified=unified)

    wargs = {
        "name": name, "run_dir": run_dir,
        "train_files": train_files, "val_files": val_files,
        "start_epoch": resume, "end_epoch": args.max_epoch,
        "max_epoch": args.max_epoch, "progress_file": progress_file,
        "unified_csv": unified, "swa": args.swa, "buffer_cap": args.buffer_cap,
        "reforge_from": args.reforge_from, "lr_mid": args.lr_mid,
        "ramp_epochs": args.ramp_epochs,
    }
    worker_main(wargs)
    print(f"\n[train] training complete ({args.max_epoch} epochs)")

    last_ckpt = find_last_checkpoint(run_dir, name)
    if last_ckpt >= 0:
        import torch
        from chessbot.train_pytorch import save_pt_model as export_ts
        src = ckpt_path(run_dir, name, last_ckpt)
        print(f"[train] exporting final model from {os.path.basename(src)}")
        raw = torch.load(src, map_location="cpu")
        arch_name = raw.get("arch", name) if isinstance(raw, dict) else name
        model = PT_BUILDERS[arch_name](VARIANTS[arch_name])
        model.load_state_dict(raw["model"])
        run_tag = os.path.basename(run_dir)
        pt_path = os.path.join(run_dir, f"{run_tag}_model.pt")
        ts_path = os.path.join(run_dir, f"{run_tag}_model.ts")
        export_ts(model, ts_path, arch=arch_name, trace=False)
        print(f"[train] export complete -> {pt_path}")

        if "optimizer" in raw:
            from chessbot.train_pytorch import pt_opt_state_path
            opt_state_path = pt_opt_state_path(pt_path, run_dir)
            os.makedirs(os.path.dirname(opt_state_path), exist_ok=True)
            torch.save(retrain_opt_state(model, raw["optimizer"]), opt_state_path)
            print(f"[train] exported optimizer state -> {opt_state_path}")

        if args.swa:
            snaps = [swa_ckpt_path(run_dir, name, e) for e in swa_epochs(args.max_epoch)]
            found = [p for p in snaps if os.path.exists(p)]
            if not found:
                print("[swa] no snapshots found -- skipping SWA export")
            else:
                sds = [torch.load(p, map_location="cpu")["model"] for p in found]
                avg = average_state_dicts(sds)
                swa_model = PT_BUILDERS[arch_name](VARIANTS[arch_name])
                swa_model.load_state_dict(avg)
                swa_pt = os.path.join(run_dir, f"{run_tag}_model_SWA.pt")
                swa_ts = os.path.join(run_dir, f"{run_tag}_model_SWA.ts")
                export_ts(swa_model, swa_ts, arch=arch_name, trace=False)
                print(f"[swa] averaged {len(found)} snapshots -> {swa_pt}")

                # SWA becomes the trainer lineage selfplay loads and retrain
                # resumes from; else the non-averaged final model plays.
                export_ts(swa_model, ts_path, arch=arch_name, trace=False)
                print(f"[swa] trainer lineage set to SWA -> {pt_path}")
                final_ckpt = ckpt_path(run_dir, name, last_ckpt)
                if os.path.exists(swa_pt):
                    for p in found:
                        if p != final_ckpt:
                            os.remove(p)
                            print(f"[swa] removed snapshot {os.path.basename(p)}")

    if args.run_dir:
        seed_files = val_files + train_files   # held-out val consumed first
        seed_selfplay_buffers(run_dir, seed_files)
    else:
        print(f"[seed] no --run-dir (run_tag={args.run_tag}); "
              f"skipping selfplay buffer seeding", flush=True)


if __name__ == "__main__":
    main()
