"""bootstrap_bakeoff_paired_pt.py -- breadth-first pair training of the two
STM-balanced bake-off arms (lc0_bal and lc0_nostm_bal) in a SINGLE process.

Experiment: instead of training each arm to completion in its own subprocess
(bootstrap_bakeoff_pt.py), train BOTH simultaneously, lockstepped by epoch:

    epoch e:  pull ONE lc0 batch
              fit lc0_bal        on it          (full 112-plane board)
              fit lc0_nostm_bal  on the SAME batch, STM plane (108) zeroed
    epoch e+1: pull a NEW batch, repeat ...

Both arms share the exact same lc0 board data, the same shuffled batch each
epoch, the same LR schedule, the same checkpoint cadence, and the same frozen
value-loss balance table (both are BALANCED_ENCODINGS). The ONLY difference is
the multiplicative plane mask: lc0_bal sees the full board, lc0_nostm_bal has
plane 108 zeroed. Fit calls never overlap -- lc0_bal finishes its epoch before
lc0_nostm_bal starts, so it is strictly sequential within a process, just
breadth-first across arms instead of depth-first.

Each arm keeps its OWN run_dir (bakeoff_lc0_bal / bakeoff_lc0_nostm_bal), its
own checkpoint, and its own eval-progress CSV, identical to the original script,
so the monitor and any downstream tooling see no difference. Logging is the same
per arm, with a blank line between the two arms' blocks so it stays readable.

This is a side experiment; the original bootstrap_bakeoff_pt.py is untouched.

Usage:
    python scripts/train/bootstrap_bakeoff_paired_pt.py
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

MODEL_NAME = "16m-precond-smartgate-lc0"

# lc0 board layout: 8 positions x 13 planes (0..103), then 8 meta planes (104..111)
STM_PLANE    = 108    # side-to-move flag (lc0 input_format 1)

# the two arms trained in lockstep. lc0_bal sees the full board; lc0_nostm_bal
# has the STM plane zeroed. both are value-loss balanced (same weight table).
ARM_TAGS = ["lc0_bal", "lc0_nostm_bal"]


def stm_zero_mask():
    """Multiplicative (112,1,1) float32 mask that zeroes only the STM plane."""
    m = np.ones(112, np.float32)
    m[STM_PLANE] = 0.0
    return m.reshape(112, 1, 1)


# ---------------------------------------------------------------------------
# data pipeline (shared bake-off schema, always full lc0 board) + prefetch
# ---------------------------------------------------------------------------

def make_dataset(files, shuffle_buffer):
    """Emit the FULL lc0 board (no mask); arms apply their own mask at fit time.
    Bundle fields: board (112,8,8) f32, policy, legal, wdl, stm (read plane 108)."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    spec = {
        "lc0_board": tf.io.FixedLenFeature([], tf.string),
        "xc0_board": tf.io.FixedLenFeature([], tf.string),
        "policy":    tf.io.FixedLenFeature([], tf.string),
        "legal":     tf.io.FixedLenFeature([], tf.string),
        "wdl":       tf.io.FixedLenFeature([3], tf.float32),
    }

    def parse(raw):
        f = tf.io.parse_single_example(raw, spec)
        board = tf.io.decode_raw(f["lc0_board"], tf.float16)
        board = tf.reshape(tf.cast(board, tf.float32), (112, 8, 8))
        stm = tf.cast(board[STM_PLANE, 0, 0], tf.int32)
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


def compute_balance_weights(files, n_sample=300_000):
    """Frozen (2,3) value-loss weight table W[stm, outcome] = P(o) / P(o|stm).

    Reweighting the value loss by this makes P(outcome|color) collapse onto the
    marginal P(outcome) for both colors, so the STM bit carries no information
    about the value target. Shared by both arms (both are balanced). One pass."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    spec = {"lc0_board": tf.io.FixedLenFeature([], tf.string),
            "wdl":       tf.io.FixedLenFeature([3], tf.float32)}

    def p(raw):
        f = tf.io.parse_single_example(raw, spec)
        b = tf.reshape(tf.cast(tf.io.decode_raw(f["lc0_board"], tf.float16),
                               tf.float32), (112, 8, 8))
        stm  = tf.cast(b[STM_PLANE, 0, 0], tf.int32)
        outc = tf.cast(tf.argmax(f["wdl"]), tf.int32)
        return stm, outc

    ds = tf.data.TFRecordDataset(files, compression_type="GZIP")
    ds = ds.map(p, num_parallel_calls=tf.data.AUTOTUNE).batch(8192)
    counts = np.zeros((2, 3), np.float64)
    seen = 0
    for stm_b, out_b in ds:
        stm_b = stm_b.numpy(); out_b = out_b.numpy()
        for c in (0, 1):
            m = stm_b == c
            for o in (0, 1, 2):
                counts[c, o] += int(np.sum(m & (out_b == o)))
        seen += len(stm_b)
        if seen >= n_sample:
            break
    p_o   = counts.sum(0) / counts.sum()
    p_o_c = counts / counts.sum(1, keepdims=True)
    w = (p_o[None, :] / p_o_c).astype(np.float32)
    return w, counts


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
# train / eval  (plane_mask distinguishes the two arms at fit time)
# ---------------------------------------------------------------------------

def pt_clipnorm_for_epoch(ep):
    if ep < 10:  return 1.0
    if ep < 20:  return 2.5
    if ep < 30:  return 5.0
    if ep < 40:  return 10.0
    return 20.0


def fit_epoch(model, opt, scaler, bundle, max_norm, device, weight_table,
              plane_mask):
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    board = torch.as_tensor(bundle["board"], dtype=torch.float32, device=device)
    if plane_mask is not None:
        board = board * plane_mask
    pol_t = torch.as_tensor(bundle["policy"], dtype=torch.float32, device=device)
    val_t = torch.as_tensor(bundle["wdl"],    dtype=torch.float32, device=device)
    stm_t = torch.as_tensor(bundle["stm"],    dtype=torch.long,    device=device)

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


def do_eval(model, name, epoch, bundle, eval_df, progress_file, device,
            weight_table, plane_mask, train_loss=None, gn_mean=None):
    import torch
    from torch.amp import autocast
    from chessbot.utils import batch_policy_metrics, print_validation

    board = torch.as_tensor(bundle["board"], dtype=torch.float32, device=device)
    if plane_mask is not None:
        board = board * plane_mask
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

    # both arms are balanced; score under the same STM-decorrelated weighting.
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
# worker -- trains BOTH arms in lockstep in one process
# ---------------------------------------------------------------------------

def build_arm(tag, run_dir, plane_mask_np, start_epoch, device):
    """Model/opt/scaler/state for one arm, loaded from its own checkpoint."""
    import torch
    from torch.amp import GradScaler

    lr0 = lr_for_epoch(start_epoch, DEFAULT_MAX_EPOCH)
    model  = PT_BUILDERS[MODEL_NAME](VARIANTS[MODEL_NAME]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr0,
                              betas=(0.9, ADAM_BETA2), weight_decay=1e-6)
    scaler = GradScaler("cuda")

    if start_epoch == 0:
        print(f"[arm {tag}] fresh init  lr={lr0:.4e}")
    else:
        last_ckpt = find_last_checkpoint(run_dir, MODEL_NAME)
        if last_ckpt < 0:
            raise RuntimeError(f"start_epoch={start_epoch} but no checkpoint in {run_dir}")
        load_from = ckpt_path(run_dir, MODEL_NAME, last_ckpt)
        print(f"[arm {tag}] loading {load_from}")
        ck = torch.load(load_from, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        for pg in opt.param_groups:
            pg["lr"] = lr0

    progress_file = os.path.join(run_dir, f"{MODEL_NAME}_pt_eval_progress.csv")
    eval_df = None
    if os.path.exists(progress_file):
        existing = pd.read_csv(progress_file)
        eval_df = existing if not existing.empty else None

    plane_mask_t = None
    if plane_mask_np is not None:
        plane_mask_t = torch.as_tensor(plane_mask_np, device=device)

    return {
        "tag": tag, "run_dir": run_dir, "progress_file": progress_file,
        "model": model, "opt": opt, "scaler": scaler, "eval_df": eval_df,
        "plane_mask": plane_mask_t,
        "window_loss_sum": 0.0, "window_gn_sum": 0.0, "window_count": 0,
    }


def worker_main(wargs):
    import gc
    import sys
    import torch
    from dotenv import load_dotenv
    load_dotenv()
    from chessbot.utils import format_time

    run_dirs    = wargs["run_dirs"]        # {tag: run_dir}
    train_files = wargs["train_files"]
    val_files   = wargs["val_files"]
    start_epoch = wargs["start_epoch"]
    max_epoch   = wargs["max_epoch"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#' * 72}")
    print(f"  paired {' + '.join(ARM_TAGS)}  epochs {start_epoch}..{max_epoch - 1}  "
          .center(72, "#"))
    print(f"{'#' * 72}\n")

    print(f"[tfrec] train={len(train_files)}  val={len(val_files)}"
          f"  batch={BATCH_SIZE}  steps/epoch={STEPS_PER_EPOCH}")

    # shared frozen balance table (both arms are value-loss balanced)
    weight_table_np, counts = compute_balance_weights(train_files)
    weight_table_t = torch.as_tensor(weight_table_np, device=device)
    with np.printoptions(precision=3, suppress=True):
        print(f"[balance] outcome counts (stm x W/D/L):\n{counts.astype(int)}")
        print(f"[balance] value-loss weights P(o)/P(o|stm):\n{weight_table_np}")

    masks = {"lc0_bal": None, "lc0_nostm_bal": stm_zero_mask()}
    arms = [build_arm(tag, run_dirs[tag], masks[tag], start_epoch, device)
            for tag in ARM_TAGS]

    # ONE dataset (full lc0 board); both arms consume the same epoch bundle.
    train_ds = make_dataset(train_files, SHUFFLE_BUFFER)
    val_ds   = make_dataset(val_files, VAL_SHUFFLE_BUFFER)
    prefetcher     = EpochBufferThread(train_ds, STEPS_PER_EPOCH, max_ready=2)
    val_prefetcher = EpochBufferThread(val_ds,   STEPS_PER_EPOCH, max_ready=1)
    prefetcher.start()
    val_prefetcher.start()

    current_lr = None
    begin = time.time()
    epoch_times = []
    t_fetch = t_fit = t_eval = 0.0
    total_samples = window_samples = 0

    for ep in range(start_epoch, max_epoch):
        target_lr = lr_for_epoch(ep, max_epoch)
        if target_lr != current_lr:
            current_lr = target_lr
            for a in arms:
                for pg in a["opt"].param_groups:
                    pg["lr"] = target_lr
            print(f"[lr update] epoch {ep}: lr={target_lr:.4e}")

        epoch_start = time.time()
        print("-" * 89)

        if ep % PLOT_EVERY == 0 or ep == max_epoch - 1:
            t0 = time.time()
            val_bundle = val_prefetcher.get()
            for a in arms:
                avg_loss = (a["window_loss_sum"] / a["window_count"]
                            if a["window_count"] > 0 else None)
                avg_gn   = (a["window_gn_sum"] / a["window_count"]
                            if a["window_count"] > 0 else None)
                print(f"[eval {a['tag']}]")
                a["eval_df"] = do_eval(
                    a["model"], MODEL_NAME, ep, val_bundle, a["eval_df"],
                    a["progress_file"], device, weight_table_np, a["plane_mask"],
                    train_loss=avg_loss, gn_mean=avg_gn)
                a["window_loss_sum"] = a["window_gn_sum"] = 0.0
                a["window_count"] = 0
                print()
            t_eval += time.time() - t0

        t0 = time.time()
        bundle = prefetcher.get()
        t_fetch += time.time() - t0

        max_norm = pt_clipnorm_for_epoch(ep)
        t0 = time.time()
        for a in arms:
            p_loss, v_loss, t_loss, gns = fit_epoch(
                a["model"], a["opt"], a["scaler"], bundle, max_norm, device,
                weight_table_t, a["plane_mask"])

            a["window_loss_sum"] += t_loss
            a["window_gn_sum"]   += gns["gn_mean"]
            a["window_count"]    += 1

            if ep == start_epoch and ep % PLOT_EVERY == 0 and a["eval_df"] is not None:
                a["eval_df"].loc[a["eval_df"].index[-1], "train_loss"] = t_loss
                a["eval_df"].loc[a["eval_df"].index[-1], "gn_mean"]    = gns["gn_mean"]
                a["eval_df"].round(4).to_csv(a["progress_file"], index=False)

            print(f"[epoch {ep:4d}] [{a['tag']}] "
                  f"policy_loss: {p_loss:.4f}  value_loss: {v_loss:.4f}  "
                  f"total: {t_loss:.4f}")
            print(f"[grad norms ] "
                  f"mean: {gns['gn_mean']:.3f}  median: {gns['gn_median']:.3f}  "
                  f"min: {gns['gn_min']:.3f}  max: {gns['gn_max']:.3f}  "
                  f"clips: {gns['gn_clips']}/{gns['gn_steps']}")
        t_fit += time.time() - t0

        n_this = int(bundle["board"].shape[0])
        total_samples  += n_this
        window_samples += n_this

        if ep % CHECKPOINT_EVERY == 0 or ep == max_epoch - 1:
            for a in arms:
                cp = ckpt_path(a["run_dir"], MODEL_NAME, ep)
                torch.save({"model": a["model"].state_dict(),
                            "optimizer": a["opt"].state_dict(),
                            "scaler": a["scaler"].state_dict(),
                            "epoch": ep, "arch": MODEL_NAME}, cp)
                print(f"[ckpt] {a['tag']} epoch {ep} -> {cp}")
                delete_old_checkpoints(a["run_dir"], MODEL_NAME, keep=ep)

        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        print(f"[time check] last: {format_time(elapsed)}  "
              f"avg: {format_time(np.mean(epoch_times))}  "
              f"total: {format_time(time.time() - begin)}  "
              f"samples: {total_samples:,}")

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
    print(f"[worker] paired {' + '.join(ARM_TAGS)}: complete ({max_epoch} epochs)")

    # tf.data + torch share one CUDA process; the repeating dataset's AUTOTUNE
    # threads never join, so normal shutdown deadlocks. Hard-exit(0) now that the
    # final checkpoints are written -- the supervisor reads exit code 0 as success.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


# ---------------------------------------------------------------------------
# supervisor
# ---------------------------------------------------------------------------

def spawn_block(wargs):
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=worker_main, args=(wargs,), daemon=False)
    p.start()
    p.join()
    return p.exitcode if p.exitcode is not None else -1


def paired_resume_epoch(run_dirs):
    """Min resume across both arms; they checkpoint together so this is normally
    the same for both. Trims each arm's CSV to its own last checkpoint."""
    resumes = []
    for tag in ARM_TAGS:
        run_dir = run_dirs[tag]
        progress_file = os.path.join(run_dir, f"{MODEL_NAME}_pt_eval_progress.csv")
        resumes.append(get_resume_epoch(run_dir, MODEL_NAME, progress_file))
    if resumes[0] != resumes[1]:
        print(f"[supervisor] WARNING arms out of sync (resumes={resumes}); "
              f"using min")
    return min(resumes)


def train_paired(tfrec_dir, max_epoch):
    parent = os.path.dirname(tfrec_dir.rstrip("/\\"))
    run_dirs = {tag: os.path.join(parent, f"bakeoff_{tag}") for tag in ARM_TAGS}
    for rd in run_dirs.values():
        os.makedirs(rd, exist_ok=True)

    files = list_tfrecord_files(tfrec_dir)
    if not files:
        raise SystemExit(f"no .tfrecord.gz in {tfrec_dir}")
    train_files, val_files = split_train_val(files)

    print(f"\n{'=' * 72}")
    print(f"[supervisor] paired arms: {', '.join(ARM_TAGS)}  model={MODEL_NAME}")
    for tag in ARM_TAGS:
        print(f"[supervisor] run_dir[{tag}]={run_dirs[tag]}")
    print(f"[supervisor] tfrec_dir={tfrec_dir}")
    print(f"[supervisor] max_epoch={max_epoch}  batch={BATCH_SIZE}"
          f"  steps/epoch={STEPS_PER_EPOCH}")
    print(f"[supervisor] train_files={len(train_files)}  val_files={len(val_files)}")

    resume = paired_resume_epoch(run_dirs)
    while resume < max_epoch:
        wargs = {
            "run_dirs": run_dirs, "train_files": train_files,
            "val_files": val_files, "start_epoch": resume, "max_epoch": max_epoch,
        }
        exit_code = spawn_block(wargs)
        if exit_code == 0:
            resume = max_epoch
            print(f"[supervisor] paired training complete")
        else:
            print(f"[supervisor] worker crashed (exit={exit_code}), recovering ...")
            time.sleep(5)
            resume = paired_resume_epoch(run_dirs)
            print(f"[supervisor] will retry from epoch {resume}")


def main():
    from dotenv import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrec-dir", default=SHUFFLED_DIR)
    ap.add_argument("--max-epoch", type=int, default=DEFAULT_MAX_EPOCH)
    args = ap.parse_args()

    train_paired(args.tfrec_dir, args.max_epoch)


if __name__ == "__main__":
    mp.freeze_support()
    main()
