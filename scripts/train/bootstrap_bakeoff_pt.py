"""bootstrap_bakeoff_pt.py -- train the arms of the lc0-vs-xc0 encoding bake-off.

Reads the shared shuffled ab_*.tfrecord.gz dataset (both boards + shared 1858
targets) and trains a copy of 16m-precond-smartgate on EACH encoding in turn:
    xc0        -> 16m-precond-smartgate      input = xc0_board (64 int16 tokens)
    lc0        -> 16m-precond-smartgate-lc0  input = lc0_board (112,8,8, rule50 /99)
    lc0_nohmc    -> 16m-precond-smartgate-lc0  lc0_board with the halfmove-clock
                    plane (rule50, plane 109) zeroed
    lc0_onehist  -> 16m-precond-smartgate-lc0  lc0_board with 6 of 7 history
                    positions zeroed -- only the current position and the last
                    move survive
    lc0_curronly -> 16m-precond-smartgate-lc0  lc0_board with all 7 prior history
                    positions zeroed -- only the current position survives (hmc kept)
    lc0_norep    -> 16m-precond-smartgate-lc0  lc0_board with all 8 repetition
                    planes (offset 12 in each 13-block) zeroed
    lc0_nostm    -> 16m-precond-smartgate-lc0  lc0_board with the side-to-move
                    plane (108) zeroed -- STM-agnostic
    lc0_4hist    -> 16m-precond-smartgate-lc0  keep only positions 0..3 (current +
                    3 history); positions 4..7 are overwritten with a copy of 0..3
                    instead of zeroed, so the net sees no dead planes. HMC and STM
                    kept.

The lc0 ablations share the -lc0 model and input; they differ only by a fixed
plane mask applied at parse time, so a score gap is attributable to the missing
feature. Everything after the input stem is identical between arms, so a gap is
attributable to the encoding. A supervisor trains one encoding to completion in
its own spawned subprocess, then the next -- fresh CUDA context per arm, OOM/crash
resume from last checkpoint. Async epoch prefetch (background thread) overlaps data
load with GPU training, same as bootstrap_model_async_tfrec_pt.py. LR schedule and
epochs follow chessbot.pretrain. Checkpoints every 100 epochs; eval every PLOT_EVERY
rolls into a CSV. No plots are written -- build them off the CSV.

Each arm resumes from its own run_dir (bakeoff_<encoding>): a completed arm has a
checkpoint at max_epoch-1 and is skipped on rerun, so adding new encodings and
restarting trains only the new arms and leaves finished ones untouched.

Usage:
    python scripts/train/bootstrap_bakeoff_pt.py
    python scripts/train/bootstrap_bakeoff_pt.py --encodings xc0
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import queue
import re
import threading
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

import numpy as np
import pandas as pd
import scipy.special

from chessbot.pretrain import (
    EPOCH_SIZE, VAL_SHUFFLE_BUFFER, PLOT_EVERY, DEFAULT_MAX_EPOCH,
    POLICY_LW, VALUE_LW, lr_for_epoch,
    list_tfrecord_files, split_train_val,
)
from chessbot.model import VARIANTS, PT_BUILDERS

SHUFFLED_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\encoding_test\shuffled"

BATCH_SIZE       = 512
STEPS_PER_EPOCH  = EPOCH_SIZE // BATCH_SIZE     # 20
SHUFFLE_BUFFER   = 128_000
ADAM_BETA2       = 0.999
CHECKPOINT_EVERY = 100
ENCODINGS        = ["lc0", "xc0", "lc0_nohmc", "lc0_onehist", "lc0_curronly",
                    "lc0_norep", "lc0_nostm", "lc0_4hist", "lc0_flatlr",
                    "lc0_morelr"]

MODELS = {
    "xc0":           "16m-precond-smartgate",
    "lc0":           "16m-precond-smartgate-lc0",
    "lc0_nohmc":     "16m-precond-smartgate-lc0",
    "lc0_onehist":   "16m-precond-smartgate-lc0",
    "lc0_curronly":  "16m-precond-smartgate-lc0",
    "lc0_norep":     "16m-precond-smartgate-lc0",
    "lc0_nostm":     "16m-precond-smartgate-lc0",
    "lc0_4hist":     "16m-precond-smartgate-lc0",
    "lc0_flatlr":    "16m-precond-smartgate-lc0",
    "lc0_morelr":    "16m-precond-smartgate-lc0",
}

# encodings that feed the 112-plane lc0 board (vanilla + ablations)
LC0_ENCODINGS = {"lc0", "lc0_nohmc", "lc0_onehist", "lc0_curronly",
                 "lc0_norep", "lc0_nostm", "lc0_4hist", "lc0_flatlr",
                 "lc0_morelr"}

# LR-schedule ablation arms: vanilla lc0 board, but a non-standard cosine. Maps
# encoding -> (lr_min, lr_max). Any arm not listed uses the pretrain.py standard
# 5e-5..3e-4 (mean ~1.512e-4, 6:1 spread).
#   lc0_flatlr: flatter cosine at the same time-mean (peak-to-trough compressed).
#   lc0_morelr: standard 6:1 shape scaled up ~1.5x -> more total LR mass.
LR_OVERRIDE_ARMS = {
    "lc0_flatlr": (1.10e-4, 2.12e-4),   # 1.9:1 spread, mean ~1.512e-4 (matched)
    "lc0_morelr": (7.5e-5,  4.5e-4),    # 6:1 spread, mean ~2.27e-4 (~1.5x heavier)
}

# lc0 board layout: 8 positions x 13 planes (0..103), then 8 meta planes (104..111)
LC0_PLANES_PER_POS = 13
LC0_HISTORY_POS    = 8      # current + 7 prior positions
LC0_REP_OFFSET     = 12     # repetition plane within each 13-block
STM_PLANE          = 108    # side-to-move flag (lc0 input_format 1)
RULE50_PLANE       = 109    # halfmove clock, already /99 at build time


def lr_for_arm(encoding, ep, max_epoch):
    """LR schedule for an arm: LR-override arms use their (lr_min, lr_max); all
    others use the pretrain.py standard schedule."""
    if encoding in LR_OVERRIDE_ARMS:
        lr_min, lr_max = LR_OVERRIDE_ARMS[encoding]
        return lr_for_epoch(ep, max_epoch, lr_min=lr_min, lr_max=lr_max)
    return lr_for_epoch(ep, max_epoch)


def lc0_plane_mask(encoding):
    """Multiplicative (112,) float32 mask for an lc0 ablation arm, or None.

    lc0_nohmc    -> zero the rule50 (halfmove clock) plane.
    lc0_onehist  -> keep positions 0 (current) and 1 (last move); zero the 6 older
                    history positions (planes 26..103).
    lc0_curronly -> keep only position 0 (current); zero all 7 prior history
                    positions (planes 13..103). Halfmove clock is kept.
    lc0_norep    -> zero the repetition plane of every history position (offset 12
                    in each 13-block: planes 12, 25, 38, ... 103).
    lc0_nostm    -> zero the side-to-move plane (108): STM-agnostic.

    lc0_4hist is not a mask (it copies planes) and is handled in make_dataset.
    """
    if encoding == "lc0_nostm":
        mask = np.ones(112, np.float32)
        mask[STM_PLANE] = 0.0
        return mask
    if encoding == "lc0_nohmc":
        mask = np.ones(112, np.float32)
        mask[RULE50_PLANE] = 0.0
        return mask
    if encoding == "lc0_onehist":
        mask = np.ones(112, np.float32)
        mask[LC0_PLANES_PER_POS * 2:LC0_PLANES_PER_POS * LC0_HISTORY_POS] = 0.0
        return mask
    if encoding == "lc0_curronly":
        mask = np.ones(112, np.float32)
        mask[LC0_PLANES_PER_POS * 1:LC0_PLANES_PER_POS * LC0_HISTORY_POS] = 0.0
        return mask
    if encoding == "lc0_norep":
        mask = np.ones(112, np.float32)
        mask[LC0_REP_OFFSET:LC0_PLANES_PER_POS * LC0_HISTORY_POS:LC0_PLANES_PER_POS] = 0.0
        return mask
    return None


# ---------------------------------------------------------------------------
# data pipeline (shared bake-off schema) + async epoch prefetch
# ---------------------------------------------------------------------------

def make_dataset(files, encoding, shuffle_buffer):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    is_lc0 = encoding in LC0_ENCODINGS
    board_key = "lc0_board" if is_lc0 else "xc0_board"
    board_dtype = tf.float16 if is_lc0 else tf.int16
    mask_np = lc0_plane_mask(encoding)
    mask_t = None if mask_np is None else tf.constant(mask_np.reshape(112, 1, 1))
    copy_hist4 = encoding == "lc0_4hist"
    spec = {
        "lc0_board": tf.io.FixedLenFeature([], tf.string),
        "xc0_board": tf.io.FixedLenFeature([], tf.string),
        "policy":    tf.io.FixedLenFeature([], tf.string),
        "legal":     tf.io.FixedLenFeature([], tf.string),
        "wdl":       tf.io.FixedLenFeature([3], tf.float32),
    }

    def parse(raw):
        f = tf.io.parse_single_example(raw, spec)
        board = tf.io.decode_raw(f[board_key], board_dtype)
        if is_lc0:
            board = tf.reshape(tf.cast(board, tf.float32), (112, 8, 8))
            stm = tf.cast(board[STM_PLANE, 0, 0], tf.int32)   # read before transform
            if copy_hist4:
                # keep positions 0..3; overwrite 4..7 with a copy of 0..3 (no dead
                # planes), then restore the 8 meta planes (STM, HMC, castling ...)
                keep = board[:LC0_PLANES_PER_POS * 4]
                board = tf.concat(
                    [keep, keep, board[LC0_PLANES_PER_POS * LC0_HISTORY_POS:]], axis=0)
            elif mask_t is not None:
                board = board * mask_t
        else:
            board = tf.reshape(tf.cast(board, tf.int32), (64,))
            stm = tf.constant(0, tf.int32)                    # unused for xc0
        policy = tf.reshape(tf.io.decode_raw(f["policy"], tf.float32), (1858,))
        legal  = tf.reshape(tf.cast(tf.io.decode_raw(f["legal"], tf.uint8), tf.int32), (1858,))
        return board, policy, legal, f["wdl"], stm

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(len(files), reshuffle_each_iteration=True).repeat()
    ds = ds.interleave(
        lambda p: tf.data.TFRecordDataset(p, compression_type="GZIP"),
        cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


class EpochBufferThread(threading.Thread):
    """Prefetches whole-epoch bundles from a TF dataset in a background thread,
    so data load overlaps GPU training. Bundle keys: board/policy/legal/wdl/stm."""

    def __init__(self, dataset, steps_per_epoch, max_ready=2):
        super().__init__(daemon=True)
        self.dataset         = dataset
        self.steps_per_epoch = steps_per_epoch
        self.q               = queue.Queue(maxsize=max_ready)
        self.stop_event      = threading.Event()

    def stop(self):
        self.stop_event.set()

    def get(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def run(self):
        import tensorflow as tf
        try:
            it = iter(self.dataset)
            while not self.stop_event.is_set():
                parts = {"board": [], "policy": [], "legal": [], "wdl": [], "stm": []}
                for _ in range(self.steps_per_epoch):
                    if self.stop_event.is_set():
                        return
                    board, policy, legal, wdl, stm = next(it)
                    parts["board"].append(board)
                    parts["policy"].append(policy)
                    parts["legal"].append(legal)
                    parts["wdl"].append(wdl)
                    parts["stm"].append(stm)
                bundle = {k: tf.concat(v, axis=0).numpy() for k, v in parts.items()}
                self.q.put(bundle)
        except Exception as exc:
            try:
                self.q.put(exc, timeout=1.0)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

def ckpt_path(run_dir, name, epoch):
    return os.path.join(run_dir, f"{name}_pt_ckpt{epoch:04d}.pt")


def find_last_checkpoint(run_dir, name):
    pat = re.compile(rf"^{re.escape(name)}_pt_ckpt(\d{{4}})\.pt$")
    best = -1
    try:
        for fn in os.listdir(run_dir):
            m = pat.match(fn)
            if m:
                best = max(best, int(m.group(1)))
    except FileNotFoundError:
        pass
    return best


def delete_old_checkpoints(run_dir, name, keep):
    pat = re.compile(rf"^{re.escape(name)}_pt_ckpt(\d{{4}})\.pt$")
    for fn in os.listdir(run_dir):
        m = pat.match(fn)
        if m and int(m.group(1)) != keep:
            os.remove(os.path.join(run_dir, fn))
            print(f"[ckpt] removed old checkpoint {fn}")


def trim_progress_csv(progress_file, max_training_epoch):
    if not os.path.exists(progress_file):
        return
    df = pd.read_csv(progress_file)
    if "training_epoch" not in df.columns:
        os.remove(progress_file)
        return
    df = df[df["training_epoch"] <= max_training_epoch]
    if df.empty:
        os.remove(progress_file)
    else:
        df.to_csv(progress_file, index=False)


def get_resume_epoch(run_dir, name, progress_file):
    os.makedirs(run_dir, exist_ok=True)
    last_ckpt = find_last_checkpoint(run_dir, name)
    if last_ckpt < 0:
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"[recovery] no checkpoints found; cleared {progress_file}")
        return 0
    trim_progress_csv(progress_file, last_ckpt)
    resume = last_ckpt + 1
    print(f"[recovery] last checkpoint epoch={last_ckpt}  ->  resuming from {resume}")
    return resume


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------

def pt_clipnorm_for_epoch(ep):
    if ep < 10:  return 1.0
    if ep < 20:  return 2.5
    if ep < 30:  return 5.0
    if ep < 40:  return 10.0
    return 20.0


def to_model_in(board_np, encoding, device):
    import torch
    if encoding in LC0_ENCODINGS:
        return torch.as_tensor(board_np, dtype=torch.float32, device=device)
    return torch.as_tensor(board_np, dtype=torch.long, device=device)


def fit_epoch(model, opt, scaler, bundle, encoding, max_norm, device,
              weight_table=None):
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    board = to_model_in(bundle["board"], encoding, device)
    pol_t = torch.as_tensor(bundle["policy"], dtype=torch.float32, device=device)
    val_t = torch.as_tensor(bundle["wdl"],    dtype=torch.float32, device=device)
    stm_t = (torch.as_tensor(bundle["stm"], dtype=torch.long, device=device)
             if weight_table is not None else None)

    n    = board.shape[0]
    perm = torch.randperm(n, device=device)
    model.train()
    total_p = total_v = total = 0.0
    grad_norms = []
    clip_count = 0
    n_batches  = 0

    for start in range(0, n, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        opt.zero_grad(set_to_none=True)
        with autocast("cuda"):
            pol, val = model(board[idx])
            p_loss = F.cross_entropy(pol, pol_t[idx]) * POLICY_LW
            if weight_table is None:
                v_loss = F.cross_entropy(val, val_t[idx]) * VALUE_LW
            else:
                v_ce = F.cross_entropy(val, val_t[idx], reduction="none")
                w = weight_table[stm_t[idx], val_t[idx].argmax(1)]
                v_loss = (v_ce * w).sum() / w.sum() * VALUE_LW
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
        total_p += p_loss.item(); total_v += v_loss.item(); total += loss.item()
        n_batches += 1

    gn = np.array(grad_norms)
    grad_stats = {
        "gn_mean": gn.mean(), "gn_median": np.median(gn),
        "gn_min": gn.min(), "gn_max": gn.max(),
        "gn_clips": clip_count, "gn_steps": n_batches,
    }
    return total_p / n_batches, total_v / n_batches, total / n_batches, grad_stats


def do_eval(model, name, epoch, bundle, eval_df, progress_file, encoding, device,
            train_loss=None, gn_mean=None, weight_table=None):
    import torch
    from torch.amp import autocast
    from chessbot.utils import batch_policy_metrics, print_validation

    board = to_model_in(bundle["board"], encoding, device)
    model.eval()
    all_pol, all_val = [], []
    with torch.no_grad(), autocast("cuda"):
        for start in range(0, board.shape[0], BATCH_SIZE):
            pol, val = model(board[start:start + BATCH_SIZE])
            all_pol.append(pol.float().cpu().numpy())
            all_val.append(val.float().cpu().numpy())

    pol_preds  = np.concatenate(all_pol, axis=0)
    val_logits = np.concatenate(all_val, axis=0)
    target_wdl = bundle["wdl"]
    pstack     = bundle["policy"]
    mstack     = bundle["legal"]

    val_wdl    = scipy.special.softmax(val_logits, axis=1)
    val_q      = val_wdl[:, 0] - val_wdl[:, 2]
    tgt_q      = target_wdl[:, 0] - target_wdl[:, 2]

    # balanced arms are scored under the same STM-decorrelated weighting they
    # train on, so the comparison is not penalized for refusing the color bias.
    if weight_table is None:
        w = np.ones(len(val_q), np.float64)
    else:
        w = weight_table[bundle["stm"], target_wdl.argmax(1)].astype(np.float64)
    w = w / w.sum()

    def wmean(x):
        return float((w * x).sum())
    se         = (val_q - tgt_q) ** 2
    value_mse  = wmean(se)
    per_ce     = -np.sum(target_wdl * np.log(np.clip(val_wdl, 1e-9, None)), axis=1)
    value_ce   = wmean(per_ce)
    mx, my     = wmean(val_q), wmean(tgt_q)
    cov        = wmean((val_q - mx) * (tgt_q - my))
    vx, vy     = wmean((val_q - mx) ** 2), wmean((tgt_q - my) ** 2)
    value_corr = float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")
    pol_stats  = batch_policy_metrics(pol_preds, pstack, mstack)

    row = {
        "training_epoch": epoch,
        "train_loss": train_loss, "gn_mean": gn_mean,
        "value_mse": value_mse, "value_corr": value_corr, "value_ce": value_ce,
        **pol_stats,
    }
    new_row = pd.DataFrame([row])
    eval_df = new_row if eval_df is None else pd.concat([eval_df, new_row], ignore_index=True)
    eval_df.round(4).to_csv(progress_file, index=False)
    print_validation(epoch, {
        "value_mse": value_mse, "value_corr": value_corr, "value_ce": value_ce,
        **pol_stats,
    })
    return eval_df


# ---------------------------------------------------------------------------
# worker -- trains ONE encoding to completion in its own subprocess
# ---------------------------------------------------------------------------

def worker_main(wargs):
    import gc
    import sys
    import torch
    from torch.amp import GradScaler
    from dotenv import load_dotenv
    load_dotenv()
    from chessbot.utils import format_time

    encoding    = wargs["encoding"]
    name        = wargs["name"]
    run_dir     = wargs["run_dir"]
    train_files = wargs["train_files"]
    val_files   = wargs["val_files"]
    start_epoch = wargs["start_epoch"]
    max_epoch   = wargs["max_epoch"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress_file = os.path.join(run_dir, f"{name}_pt_eval_progress.csv")

    print(f"\n{'#' * 72}")
    print(f"  {name} [{encoding}]  epochs {start_epoch}..{max_epoch - 1}  ".center(72, "#"))
    print(f"{'#' * 72}\n")

    lr0 = lr_for_arm(encoding, start_epoch, max_epoch)
    model  = PT_BUILDERS[name](VARIANTS[name]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr0,
                              betas=(0.9, ADAM_BETA2), weight_decay=1e-6)
    scaler = GradScaler("cuda")

    if start_epoch == 0:
        print(f"[worker] fresh init  lr={lr0:.4e}")
    else:
        last_ckpt = find_last_checkpoint(run_dir, name)
        if last_ckpt < 0:
            raise RuntimeError(f"start_epoch={start_epoch} but no checkpoint in {run_dir}")
        load_from = ckpt_path(run_dir, name, last_ckpt)
        print(f"[worker] loading {load_from}")
        ck = torch.load(load_from, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        for pg in opt.param_groups:
            pg["lr"] = lr0

    eval_df = None
    if os.path.exists(progress_file):
        existing = pd.read_csv(progress_file)
        eval_df = existing if not existing.empty else None

    print(f"[tfrec] train={len(train_files)}  val={len(val_files)}"
          f"  batch={BATCH_SIZE}  steps/epoch={STEPS_PER_EPOCH}")

    train_ds = make_dataset(train_files, encoding, SHUFFLE_BUFFER)
    val_ds   = make_dataset(val_files, encoding, VAL_SHUFFLE_BUFFER)
    prefetcher     = EpochBufferThread(train_ds, STEPS_PER_EPOCH, max_ready=2)
    val_prefetcher = EpochBufferThread(val_ds,   STEPS_PER_EPOCH, max_ready=1)
    prefetcher.start()
    val_prefetcher.start()

    current_lr = None
    begin = time.time()
    epoch_times = []
    t_fetch = t_fit = t_eval = 0.0
    total_samples = window_samples = 0
    window_loss_sum = window_gn_sum = 0.0
    window_count = 0

    for ep in range(start_epoch, max_epoch):
        target_lr = lr_for_arm(encoding, ep, max_epoch)
        if target_lr != current_lr:
            current_lr = target_lr
            for pg in opt.param_groups:
                pg["lr"] = target_lr
            print(f"[lr update] epoch {ep}: lr={target_lr:.4e}")

        epoch_start = time.time()
        print("-" * 89)

        if ep % PLOT_EVERY == 0 or ep == max_epoch - 1:
            t0 = time.time()
            val_bundle = val_prefetcher.get()
            avg_loss = window_loss_sum / window_count if window_count > 0 else None
            avg_gn   = window_gn_sum   / window_count if window_count > 0 else None
            eval_df = do_eval(model, name, ep, val_bundle, eval_df, progress_file,
                              encoding, device, train_loss=avg_loss, gn_mean=avg_gn)
            window_loss_sum = window_gn_sum = 0.0
            window_count = 0
            t_eval += time.time() - t0

        t0 = time.time()
        bundle = prefetcher.get()
        t_fetch += time.time() - t0

        t0 = time.time()
        max_norm = pt_clipnorm_for_epoch(ep)
        p_loss, v_loss, t_loss, gns = fit_epoch(
            model, opt, scaler, bundle, encoding, max_norm, device)
        t_fit += time.time() - t0

        n_this = int(bundle["board"].shape[0])
        total_samples   += n_this
        window_samples  += n_this
        window_loss_sum += t_loss
        window_gn_sum   += gns["gn_mean"]
        window_count    += 1

        if ep == start_epoch and ep % PLOT_EVERY == 0 and eval_df is not None:
            eval_df.loc[eval_df.index[-1], "train_loss"] = t_loss
            eval_df.loc[eval_df.index[-1], "gn_mean"]    = gns["gn_mean"]
            eval_df.round(4).to_csv(progress_file, index=False)

        print(f"[epoch {ep:4d}] [{name}] "
              f"policy_loss: {p_loss:.4f}  value_loss: {v_loss:.4f}  "
              f"total: {t_loss:.4f}  samples: {total_samples:,}")
        print(f"[grad norms ] "
              f"mean: {gns['gn_mean']:.3f}  median: {gns['gn_median']:.3f}  "
              f"min: {gns['gn_min']:.3f}  max: {gns['gn_max']:.3f}  "
              f"clips: {gns['gn_clips']}/{gns['gn_steps']}")

        if ep % CHECKPOINT_EVERY == 0 or ep == max_epoch - 1:
            cp = ckpt_path(run_dir, name, ep)
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "arch": name}, cp)
            print(f"[ckpt] epoch {ep} -> {cp}")
            delete_old_checkpoints(run_dir, name, keep=ep)

        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        print(f"[time check] last: {format_time(elapsed)}  "
              f"avg: {format_time(np.mean(epoch_times))}  "
              f"total: {format_time(time.time() - begin)}")

        if ep % PLOT_EVERY == 0 and ep > start_epoch:
            print(f"[timing/{PLOT_EVERY}ep] fetch: {t_fetch:.2f}s  "
                  f"fit: {t_fit:.2f}s  eval: {t_eval:.2f}s  "
                  f"samples_window: {window_samples:,}  "
                  f"samples_total: {total_samples:,}")
            t_fetch = t_fit = t_eval = 0.0
            window_samples = 0

        print()

    prefetcher.stop()
    val_prefetcher.stop()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[worker] {name} [{encoding}]: complete ({max_epoch} epochs)")

    # tf.data + torch share one CUDA process; the repeating dataset's AUTOTUNE
    # threads never join, so normal shutdown deadlocks. Hard-exit(0) now that the
    # final checkpoint is written -- the supervisor reads exit code 0 as success.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


# ---------------------------------------------------------------------------
# supervisor -- trains one encoding then the other
# ---------------------------------------------------------------------------

def spawn_block(wargs):
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=worker_main, args=(wargs,), daemon=False)
    p.start()
    p.join()
    return p.exitcode if p.exitcode is not None else -1


def train_encoding(encoding, tfrec_dir, max_epoch):
    name = MODELS[encoding]
    parent = os.path.dirname(tfrec_dir.rstrip("/\\"))
    run_dir = os.path.join(parent, f"bakeoff_{encoding}")
    os.makedirs(run_dir, exist_ok=True)
    progress_file = os.path.join(run_dir, f"{name}_pt_eval_progress.csv")

    files = list_tfrecord_files(tfrec_dir)
    if not files:
        raise SystemExit(f"no .tfrecord.gz in {tfrec_dir}")
    train_files, val_files = split_train_val(files)

    print(f"\n{'=' * 72}")
    print(f"[supervisor] encoding={encoding}  model={name}")
    print(f"[supervisor] run_dir={run_dir}")
    print(f"[supervisor] tfrec_dir={tfrec_dir}")
    print(f"[supervisor] max_epoch={max_epoch}  batch={BATCH_SIZE}"
          f"  steps/epoch={STEPS_PER_EPOCH}")
    print(f"[supervisor] train_files={len(train_files)}  val_files={len(val_files)}")

    resume = get_resume_epoch(run_dir, name, progress_file)
    while resume < max_epoch:
        wargs = {
            "encoding": encoding, "name": name, "run_dir": run_dir,
            "train_files": train_files, "val_files": val_files,
            "start_epoch": resume, "max_epoch": max_epoch,
        }
        exit_code = spawn_block(wargs)
        if exit_code == 0:
            resume = max_epoch
            print(f"[supervisor] {encoding} complete")
        else:
            print(f"[supervisor] {encoding} worker crashed (exit={exit_code}), recovering ...")
            time.sleep(5)
            resume = get_resume_epoch(run_dir, name, progress_file)
            print(f"[supervisor] will retry {encoding} from epoch {resume}")


def main():
    from dotenv import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--encodings", nargs="+", choices=ENCODINGS, default=ENCODINGS)
    ap.add_argument("--tfrec-dir", default=SHUFFLED_DIR)
    ap.add_argument("--max-epoch", type=int, default=DEFAULT_MAX_EPOCH)
    args = ap.parse_args()

    for encoding in args.encodings:
        train_encoding(encoding, args.tfrec_dir, args.max_epoch)

    print(f"\n[supervisor] bake-off training complete: {', '.join(args.encodings)}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
