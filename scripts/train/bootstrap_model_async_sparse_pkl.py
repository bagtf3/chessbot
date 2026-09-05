"""OOM-robust PyTorch bootstrap trainer, sparse pkl.gz shards (no TensorFlow).

Trains directly on the 4-tuple sparse shards produced by build_pretrain_shards.py
(xc0h, sparse_policy, wdl_target, source). Data is served by an async 256k-record
pool: a loader thread cycles shards into a ring buffer, a prefetch thread draws
random 10240-record epoch bundles and keeps the next one ready. Everything else
(cosine LR, SWA, resume, unified eval_progress.csv, checkpoint/export) mirrors the
tfrecord trainer; this is meant to replace it once proven.

Usage:
    python scripts/train/bootstrap_model_async_sparse_pkl.py
        [--model precond-mha-10c6t-d256] [--run-tag TAG | --run-dir DIR]
        [--shard-dir DIR] [--max-epoch 3501] [--val-shards 20] [--swa]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import math
import multiprocessing as mp
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

from chessbot.pretrain import (
    EPOCH_SIZE, PLOT_EVERY, LR_WARMUP_EPOCHS, VAL_FRACTION, lr_for_epoch,
    save_plot, split_train_val as split_train_val_frac,
)
from chessbot.model import VARIANTS, PT_BUILDERS
from chessbot.replay_buffer import POLICY_DIM

PT_BATCH_SIZE      = 512
PT_STEPS_PER_EPOCH = EPOCH_SIZE // PT_BATCH_SIZE   # 20
PT_ADAM_BETA2      = 0.999

BUFFER_CAP     = 256_000       # shuffle pool high-water mark
REFILL_BAND    = 56_000        # refill once the pool drains below high - band (-> 200k)
VAL_BUFFER_CAP = 96_000

# cosine LR: floor raised to 1e-4 (was 5e-5), ceiling unchanged
LR_MIN = 1e-4
LR_MAX = 3e-4

CHECKPOINT_EVERY = 100
SWA_WINDOW = 100
SWA_EVERY  = 20

DEFAULT_MODEL     = "precond-mha-10c6t-d256"
DEFAULT_RUN_TAG   = "val_test_multi"
DEFAULT_MAX_EPOCH = 3501
DEFAULT_SHARD_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\pretrain_shards"

LW_WARMUP_POLICY   = 1.0
LW_WARMUP_VALUE    = 2.0
LW_EMA_SPAN_EPOCHS = 20
LW_EMA_ALPHA       = 1.0 / LW_EMA_SPAN_EPOCHS
LW_RECOMPUTE_EVERY = 10
LW_TARGET_VALUE_MULT  = 4.0
LW_TARGET_POLICY_MULT = 1.0

UNIFIED_COLS = [
    "model_epoch", "value_mse", "value_corr", "value_ce", "policy_ce",
    "uniform_ce", "ce_gain", "top1_exact", "top1_mass", "top3_mass",
    "top5_mass", "n_samples", "mass_on_legal", "avg_top_prob",
    "avg_top_prob_target",
    "train_loss", "gn_mean", "exp_prob_model", "exp_prob_uniform",
    "prob_on_others",
]


# ---------------------------------------------------------------------------
# LW schedule
# ---------------------------------------------------------------------------

def lw_ema_update(ema, raw):
    return raw if ema is None else (1.0 - LW_EMA_ALPHA) * ema + LW_EMA_ALPHA * raw


def lw_for_epoch(ep, policy_ce_hat, value_ce_hat, active):
    if ep < LR_WARMUP_EPOCHS:
        return LW_WARMUP_POLICY, LW_WARMUP_VALUE, None, False
    osc_ep = ep - LR_WARMUP_EPOCHS
    if osc_ep % LW_RECOMPUTE_EVERY != 0:
        return active["policy_lw"], active["value_lw"], active["scale"], False
    current_total = LW_WARMUP_POLICY * policy_ce_hat + LW_WARMUP_VALUE * value_ce_hat
    target_total  = (LW_TARGET_VALUE_MULT * value_ce_hat
                     + LW_TARGET_POLICY_MULT * policy_ce_hat)
    scale = target_total / current_total
    return scale * LW_WARMUP_POLICY, scale * LW_WARMUP_VALUE, scale, True


# ---------------------------------------------------------------------------
# Checkpoint / CSV utilities
# ---------------------------------------------------------------------------

def ckpt_path(run_dir, name, epoch):
    return os.path.join(run_dir, f"{name}_pt_ckpt{epoch:04d}.pt")


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
    return df.reindex(columns=UNIFIED_COLS + extra)


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
        if os.path.exists(progress_file) and not unified:
            os.remove(progress_file)
            print(f"[recovery] no checkpoints found; cleared {progress_file}")
        return 0
    trim_progress_csv(progress_file, last_ckpt, unified=unified)
    print(f"[recovery] last checkpoint epoch={last_ckpt} -> resuming {last_ckpt + 1}")
    return last_ckpt + 1


def make_adam(model, lr):
    import torch
    return torch.optim.Adam(model.parameters(), lr=lr,
                            betas=(0.9, PT_ADAM_BETA2), weight_decay=1e-6)


def load_pt_model(path, name, device, lr):
    import torch
    from torch.amp import GradScaler
    cfg    = VARIANTS[name]
    model  = PT_BUILDERS[name](cfg).to(device)
    opt    = make_adam(model, lr)
    scaler = GradScaler("cuda")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
        for pg in opt.param_groups:
            pg["lr"] = lr
    return model, opt, scaler, ckpt.get("lw_state")


def save_pt_ckpt(model, opt, scaler, epoch, name, path, lw_state=None):
    import torch
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "arch": name,
        "lw_state": lw_state,
    }, path)
    print(f"[ckpt] epoch {epoch} -> {path}")


def pt_clipnorm_for_epoch(ep):
    if ep < 10:  return 1.0
    if ep < 20:  return 2.5
    if ep < 30:  return 5.0
    if ep < 40:  return 10.0
    return 20.0


# ---------------------------------------------------------------------------
# Async sparse-shard data layer
# ---------------------------------------------------------------------------

def densify(sp):
    idx, vals = sp
    d = np.zeros(POLICY_DIM, dtype=np.float32)
    if len(idx):
        d[np.asarray(idx, dtype=np.int32)] = np.asarray(vals, dtype=np.float32)
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
                 n_ready=2, name="train"):
        self.files      = list(shard_files)
        self.high       = high
        self.low        = max(epoch_size, min(low, high))
        self.epoch_size = epoch_size
        self.drain      = drain
        self.name       = name
        self.pool = []
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
            with self.lock:
                self.pool.extend(recs)
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
        with self.lock:
            n = self.size
            k = min(self.epoch_size, n)
            picked = np.random.choice(n, size=k, replace=False)
            recs = [self.pool[i] for i in picked]
            if self.drain:
                drop = set(picked.tolist())
                self.pool = [self.pool[i] for i in range(n) if i not in drop]
                self.size = len(self.pool)
        enc = np.stack([r[0] for r in recs]).astype(np.int64)
        pol = np.stack([densify(r[1]) for r in recs])
        val = np.stack([r[2] for r in recs]).astype(np.float32)
        w   = np.ones(len(recs), dtype=np.float32)
        return {"enc_in": enc, "policy_logits": pol, "value_out": val, "weight": w}

    def prefetch(self):
        while not self.stop_evt.is_set():
            with self.lock:
                enough = self.size >= self.epoch_size
            if enough or self.loader_done.is_set():
                break
            time.sleep(0.1)
        while not self.stop_evt.is_set():
            while not self.stop_evt.is_set():
                with self.lock:
                    enough = self.size >= self.epoch_size
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

def fit_epoch(model, opt, scaler, bundle, lw, max_norm, device):
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
    total_p = total_v = total = 0.0
    grad_norms = []
    clip_count = 0
    n_batches  = 0

    for start in range(0, n, PT_BATCH_SIZE):
        idx = perm[start:start + PT_BATCH_SIZE]
        opt.zero_grad(set_to_none=True)
        with autocast("cuda"):
            pol, val = model(enc[idx])
            p_loss = F.cross_entropy(pol, policy_t[idx]) * lw["policy_logits"]
            v_loss = (F.cross_entropy(val, value_t[idx], reduction="none")
                      * w[idx]).mean() * lw["value_out"]
            loss = p_loss + v_loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()
        grad_norms.append(raw_norm)
        if raw_norm > max_norm or math.isnan(raw_norm):
            clip_count += 1
        scaler.step(opt)
        scaler.update()
        total_p += p_loss.item()
        total_v += v_loss.item()
        total   += loss.item()
        n_batches += 1

    gn = np.array(grad_norms)
    grad_stats = {
        "gn_mean": gn.mean(), "gn_median": np.median(gn), "gn_min": gn.min(),
        "gn_max": gn.max(), "gn_clips": clip_count, "gn_steps": n_batches,
    }
    return total_p / n_batches, total_v / n_batches, total / n_batches, grad_stats


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
        "n_samples": int(enc.shape[0]),
        "avg_top_prob_target": float(pstack.max(axis=1).mean()),
        "train_loss": train_loss, "gn_mean": gn_mean,
    }
    new_row = pd.DataFrame([row]).reindex(columns=UNIFIED_COLS)
    eval_df = new_row if eval_df is None else pd.concat([eval_df, new_row], ignore_index=True)
    eval_df.round(4).to_csv(progress_file, index=False)

    print_validation(epoch, {
        "value_mse": value_mse, "value_corr": value_corr, "value_ce": value_ce,
        **pol_stats,
    })
    save_plot(eval_df, f"{name}  [PT]", epoch, plot_file, tgt_q, val_q)
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
    progress_file = wargs.get("progress_file") or progress_csv_path(run_dir, name, unified)
    plot_file     = os.path.join(run_dir, f"{name}_pt_plot.png")

    print(f"\n{'#' * 72}")
    print(f"  {name}  epochs {start_epoch}..{end_epoch - 1}  ".center(72, "#"))
    print(f"{'#' * 72}\n")

    lr0 = lr_for_epoch(start_epoch, max_epoch, lr_min=LR_MIN, lr_max=LR_MAX)
    if start_epoch == 0:
        from torch.amp import GradScaler
        cfg    = VARIANTS[name]
        model  = PT_BUILDERS[name](cfg).to(device)
        opt    = make_adam(model, lr0)
        scaler = GradScaler("cuda")
        lw_state = None
        print(f"[worker] fresh init  lr={lr0:.4e}")
    else:
        last_ckpt = find_last_checkpoint(run_dir, name)
        if last_ckpt < 0:
            raise RuntimeError(f"start_epoch={start_epoch} but no checkpoint in {run_dir}")
        load_from = ckpt_path(run_dir, name, last_ckpt)
        print(f"[worker] loading {load_from}")
        model, opt, scaler, lw_state = load_pt_model(load_from, name, device, lr=lr0)

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
        epoch_size=EPOCH_SIZE, drain=True, n_ready=1, name="val")
    train_pool.start()
    val_pool.start()

    if lw_state is not None:
        policy_ce_hat = lw_state["policy_ce_hat"]
        value_ce_hat  = lw_state["value_ce_hat"]
        active_lw = {"policy_lw": lw_state["policy_lw"],
                     "value_lw": lw_state["value_lw"],
                     "scale": lw_state.get("scale")}
        print(f"[worker] restored LW: policy_ce_hat={policy_ce_hat:.3f}  "
              f"value_ce_hat={value_ce_hat:.3f}  scale={active_lw['scale']}")
    else:
        policy_ce_hat = value_ce_hat = None
        active_lw = {"policy_lw": LW_WARMUP_POLICY, "value_lw": LW_WARMUP_VALUE,
                     "scale": None}

    current_lr = None
    begin = time.time()
    epoch_times = []
    t_fetch = t_fit = t_eval = 0.0
    total_samples = window_samples = 0
    window_loss_sum = window_gn_sum = 0.0
    window_count = 0
    last_tgt_q = last_val_q = None

    for ep in range(start_epoch, end_epoch):
        target_lr = lr_for_epoch(ep, max_epoch, lr_min=LR_MIN, lr_max=LR_MAX)
        if target_lr != current_lr:
            current_lr = target_lr
            for pg in opt.param_groups:
                pg["lr"] = target_lr
            print(f"[lr update] epoch {ep}: lr={target_lr:.4e}")

        epoch_start = time.time()
        print("-" * 89)

        if ep % PLOT_EVERY == 0 or ep == end_epoch - 1:
            t0 = time.time()
            val_bundle = val_pool.get()
            avg_loss = window_loss_sum / window_count if window_count > 0 else None
            avg_gn   = window_gn_sum   / window_count if window_count > 0 else None
            eval_df, last_tgt_q, last_val_q = do_eval(
                model, name, ep, val_bundle, eval_df, progress_file, plot_file,
                device, train_loss=avg_loss, gn_mean=avg_gn)
            window_loss_sum = window_gn_sum = 0.0
            window_count = 0
            t_eval += time.time() - t0

        t0 = time.time()
        bundle = train_pool.get()
        t_fetch += time.time() - t0

        policy_lw, value_lw, scale, recomputed = lw_for_epoch(
            ep, policy_ce_hat, value_ce_hat, active_lw)
        active_lw = {"policy_lw": policy_lw, "value_lw": value_lw, "scale": scale}
        lw = {"policy_logits": policy_lw, "value_out": value_lw}

        t0 = time.time()
        max_norm = pt_clipnorm_for_epoch(ep)
        p_loss, v_loss, t_loss, gns = fit_epoch(
            model, opt, scaler, bundle, lw, max_norm, device)
        t_fit += time.time() - t0

        policy_ce_hat = lw_ema_update(policy_ce_hat, p_loss / policy_lw)
        value_ce_hat  = lw_ema_update(value_ce_hat,  v_loss / value_lw)
        live_A = policy_ce_hat / value_ce_hat

        n_this = int(bundle["enc_in"].shape[0])
        total_samples  += n_this
        window_samples += n_this
        window_loss_sum += t_loss
        window_gn_sum   += gns["gn_mean"]
        window_count    += 1

        if ep == start_epoch and ep % PLOT_EVERY == 0 and eval_df is not None:
            eval_df.loc[eval_df.index[-1], "train_loss"] = t_loss
            eval_df.loc[eval_df.index[-1], "gn_mean"]   = gns["gn_mean"]
            eval_df.round(4).to_csv(progress_file, index=False)

        print(f"[epoch {ep:4d}] [{name}] policy_loss: {p_loss:.2f}  "
              f"value_loss: {v_loss:.2f}  total: {t_loss:.2f}  "
              f"samples: {total_samples:,}")
        print(f"[epoch {ep:4d}] [{name}] policy_ce_hat: {policy_ce_hat:.2f}  "
              f"value_ce_hat (raw): {value_ce_hat:.3f}  A = {live_A:.2f}")
        if recomputed:
            print(f"[lw-fixed RECOMPUTE] scale: {scale:.3f}  "
                  f"new LW: policy={policy_lw:.3f}  value={value_lw:.3f}\n")
        print(f"[grad norms ] mean: {gns['gn_mean']:.3f}  "
              f"median: {gns['gn_median']:.3f}  min: {gns['gn_min']:.3f}  "
              f"max: {gns['gn_max']:.3f}  clips: {gns['gn_clips']}/{gns['gn_steps']}")

        is_ckpt = ep % CHECKPOINT_EVERY == 0 or ep == end_epoch - 1 or ep == max_epoch - 1
        if is_ckpt:
            lw_state = {
                "policy_ce_hat": policy_ce_hat, "value_ce_hat": value_ce_hat,
                "policy_lw": active_lw["policy_lw"], "value_lw": active_lw["value_lw"],
                "scale": active_lw["scale"],
            }
            save_pt_ckpt(model, opt, scaler, ep, name, ckpt_path(run_dir, name, ep),
                         lw_state=lw_state)
            delete_old_checkpoints(run_dir, name, keep_epoch=ep)
            save_plot(eval_df, f"{name}  [PT]", ep, plot_file, last_tgt_q, last_val_q)

        if ep in swa_set:
            sp = swa_ckpt_path(run_dir, name, ep)
            torch.save({"model": model.state_dict(), "arch": name}, sp)
            print(f"[swa] snapshot -> {os.path.basename(sp)}")

        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        print(f"[time check] last: {format_time(elapsed)}  "
              f"avg: {format_time(np.mean(epoch_times))}  "
              f"total: {format_time(time.time() - begin)}")

        if ep % PLOT_EVERY == 0 and ep > start_epoch:
            print(f"[timing/{PLOT_EVERY}ep] fetch: {t_fetch:.2f}s  fit: {t_fit:.2f}s  "
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
    args = parser.parse_args()

    run_dir = args.run_dir or os.path.join(SP_DIR, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)
    unified = args.run_dir is not None
    name = args.model
    progress_file = progress_csv_path(run_dir, name, unified)

    print(f"[train] model={name}")
    print(f"[train] run_dir={run_dir}")
    print(f"[train] progress_csv={os.path.basename(progress_file)}")
    print(f"[train] shard_dir={args.shard_dir}")
    print(f"[train] max_epoch={args.max_epoch}  buffer={args.buffer_cap:,}")
    print(f"[train] batch={PT_BATCH_SIZE}  steps/epoch={PT_STEPS_PER_EPOCH}"
          f"  lr_range=[{LR_MIN:.1e}, {LR_MAX:.1e}]")

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
            torch.save(raw["optimizer"], opt_state_path)
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
