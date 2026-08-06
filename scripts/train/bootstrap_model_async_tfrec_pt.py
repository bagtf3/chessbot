"""bootstrap_model_async_tfrec_pt.py — OOM-robust PyTorch bootstrap trainer

The supervisor spawns a fresh subprocess for each block of epochs.
On crash the supervisor rolls back to the last checkpoint and restarts.

PyTorch owns CUDA entirely.  TF is restricted to CPU (tfrecord reading only).

Pre-saved .pt source models must exist in MODEL_DIR before running.

Usage:
    python scripts/bootstrap_model_async_tfrec_pt.py [options]

Options:
    --model      Model name            (default: conv-pure)
    --run-tag    Sub-dir under SP_DIR  (default: val_test_multi)
    --run-dir    Explicit run dir (overrides --run-tag)
    --tfrec-dir  Path to .tfrecord.gz files  (env: BOOTSTRAP_TFREC_DIR)
    --max-epoch  Total epochs to train        (default: 2000)
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
import random
import re
import shutil
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

import numpy as np
import pandas as pd
import scipy.special

from chessbot.pretrain import (
    EPOCH_SIZE, VAL_SHUFFLE_BUFFER,
    PLOT_EVERY,
    LR_WARMUP_EPOCHS,
    # POLICY_LW, VALUE_LW,  # superseded by LW-twiddle experiment below
    lr_for_epoch,
    list_tfrecord_files, split_train_val,
    make_dataset, EpochBufferThread,
    save_plot,
)

from chessbot.replay_buffer import new_shard_path, write_pkl_gz_shard, SEED_SOURCE

SEED_REPLAY_SHARDS   = 64
SEED_PRIMARY_SHARDS  = 24
SEED_REMAINING_SHARDS = 8

from chessbot.model import VARIANTS, PT_BUILDERS

PT_BATCH_SIZE      = 512
PT_STEPS_PER_EPOCH = EPOCH_SIZE // PT_BATCH_SIZE   # 20
PT_ADAM_BETA2      = 0.999
PT_SHUFFLE_BUFFER  = 384_000

CHECKPOINT_EVERY  = 100

# --swa: snapshot the tail of the run and average it. SWA_EVERY apart, ending
# on the final epoch, spanning at most SWA_WINDOW epochs -> 4 snapshots plus
# the final one at the defaults.
SWA_WINDOW = 100
SWA_EVERY  = 20
DEFAULT_MODEL     = "precond-mha-10c6t-d256"
DEFAULT_RUN_TAG   = "val_test_multi"
DEFAULT_MAX_EPOCH = 3501
DEFAULT_TFREC_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\xc0hK6_combined\shuffled"

# ---------------------------------------------------------------------------
# LW-twiddle experiment (lw_twiddle branch, pretrain-only)
#
# Twiddling/oscillation paused for now -- fixed 1:2 policy:value shape
# throughout (policy_ce + 2*value_ce, same ratio warmup already uses), no
# adaptive A, no sin cycle. After warmup, track EMAs of the raw per-epoch
# CEs and every LW_RECOMPUTE_EVERY epochs rescale that fixed-shape pair by a
# single factor so the weighted total matches LW_TARGET_VALUE_MULT*
# value_ce_hat + LW_TARGET_POLICY_MULT*policy_ce_hat -- i.e. policy_ce +
# 2*value_ce, upweighted to the magnitude of policy_ce + 4*value_ce.
# ---------------------------------------------------------------------------
LW_WARMUP_POLICY   = 1.0
LW_WARMUP_VALUE    = 2.0

LW_EMA_SPAN_EPOCHS = 20             # 400 steps / 20 steps-per-epoch
LW_EMA_ALPHA       = 1.0 / LW_EMA_SPAN_EPOCHS

LW_RECOMPUTE_EVERY = 10             # epochs between LW recomputes (200 steps)

LW_TARGET_VALUE_MULT  = 4.0         # target total = this*value_ce_hat
LW_TARGET_POLICY_MULT = 1.0         #              + this*policy_ce_hat


def lw_ema_update(ema, raw):
    return raw if ema is None else (1.0 - LW_EMA_ALPHA) * ema + LW_EMA_ALPHA * raw


def lw_for_epoch(ep, policy_ce_hat, value_ce_hat, active):
    """
    Return (policy_lw, value_lw, scale, recomputed). `active` is the dict of
    values carried over from the last recompute; passed through unchanged
    on non-recompute epochs. Fixed 1:2 shape, rescaled to the target total.
    """
    if ep < LR_WARMUP_EPOCHS:
        return LW_WARMUP_POLICY, LW_WARMUP_VALUE, None, False

    osc_ep = ep - LR_WARMUP_EPOCHS
    if osc_ep % LW_RECOMPUTE_EVERY != 0:
        return active["policy_lw"], active["value_lw"], active["scale"], False

    current_total = LW_WARMUP_POLICY * policy_ce_hat + LW_WARMUP_VALUE * value_ce_hat
    target_total = (
        LW_TARGET_VALUE_MULT * value_ce_hat + LW_TARGET_POLICY_MULT * policy_ce_hat
    )
    scale = target_total / current_total
    policy_lw = scale * LW_WARMUP_POLICY
    value_lw  = scale * LW_WARMUP_VALUE

    return policy_lw, value_lw, scale, True


# Shared schema for the unified eval_progress.csv. The first 15 are selfplay's
# existing columns in their existing order -- Rescorer.aggregate_metrics writes
# exactly these and nothing else. The last 5 only pretraining can produce, so
# they sit at the end and stay blank once selfplay starts appending.
UNIFIED_COLS = [
    "model_epoch", "value_mse", "value_corr", "value_ce", "policy_ce",
    "uniform_ce", "ce_gain", "top1_exact", "top1_mass", "top3_mass",
    "top5_mass", "n_samples", "mass_on_legal", "avg_top_prob",
    "avg_top_prob_target",
    "train_loss", "gn_mean", "exp_prob_model", "exp_prob_uniform",
    "prob_on_others",
]


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def ckpt_path(run_dir: str, name: str, epoch: int) -> str:
    return os.path.join(run_dir, f"{name}_pt_ckpt{epoch:04d}.pt")


def swa_ckpt_path(run_dir: str, name: str, epoch: int) -> str:
    """Deliberately not the _pt_ckpt pattern -- delete_old_checkpoints prunes
    everything but the newest of those, and SWA snapshots have to survive."""
    return os.path.join(run_dir, f"{name}_pt_swa{epoch:04d}.pt")


def swa_epochs(max_epoch: int) -> list[int]:
    """Snapshot epochs: the final one, then back by SWA_EVERY within SWA_WINDOW."""
    final = max_epoch - 1
    eps = [final - k * SWA_EVERY for k in range(SWA_WINDOW // SWA_EVERY)]
    return sorted(e for e in eps if e >= 0)


def average_state_dicts(sds: list[dict]) -> dict:
    """Mean of float tensors; non-float entries (int index buffers like sl_idx)
    take the last value since averaging would corrupt the dtype. No BatchNorm
    anywhere in these models, so there are no running stats to recompute."""
    import torch

    out = {}
    for k in sds[0]:
        vals = [sd[k] for sd in sds]
        if vals[0].is_floating_point():
            out[k] = torch.stack([v.float() for v in vals]).mean(0).to(vals[0].dtype)
        else:
            out[k] = vals[-1]
    return out


def normalize_progress_df(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a loaded progress csv onto UNIFIED_COLS so resumed rows land in the
    right columns. Files written before unification key on training_epoch; that
    gets renamed rather than left to collide with model_epoch."""
    if "model_epoch" not in df.columns and "training_epoch" in df.columns:
        df = df.rename(columns={"training_epoch": "model_epoch"})
    extra = [c for c in df.columns if c not in UNIFIED_COLS]
    return df.reindex(columns=UNIFIED_COLS + extra)


def progress_csv_path(run_dir: str, name: str, unified: bool) -> str:
    """Unified mode writes straight into the selfplay run's eval_progress.csv so
    the first retrain appends to the same history. Otherwise the standalone
    pretrain file, unchanged."""
    if unified:
        return os.path.join(run_dir, "eval_progress.csv")
    return os.path.join(run_dir, f"{name}_pt_eval_progress.csv")


def find_last_checkpoint(run_dir: str, name: str) -> int:
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


def delete_old_checkpoints(run_dir: str, name: str, keep_epoch: int) -> None:
    pat = re.compile(rf"^{re.escape(name)}_pt_ckpt(\d+)\.pt$")
    try:
        for fname in os.listdir(run_dir):
            m = pat.match(fname)
            if m and int(m.group(1)) != keep_epoch:
                path = os.path.join(run_dir, fname)
                os.remove(path)
                print(f"[ckpt] removed old checkpoint {fname}")
    except FileNotFoundError:
        pass


def trim_progress_csv(progress_file: str, max_training_epoch: int,
                      unified: bool = False) -> None:
    if not os.path.exists(progress_file):
        return
    df = pd.read_csv(progress_file)
    # training_epoch is the pre-unification key; still accepted so an older
    # standalone run resumes instead of having its history thrown away
    epoch_col = next(
        (c for c in ("model_epoch", "training_epoch") if c in df.columns), None)
    if epoch_col is None:
        if unified:
            print(f"[recovery] {os.path.basename(progress_file)} has no epoch "
                  f"column; leaving it alone")
            return
        os.remove(progress_file)
        return
    df = df[df[epoch_col] <= max_training_epoch]
    if df.empty and not unified:
        os.remove(progress_file)
    else:
        df.to_csv(progress_file, index=False)


def get_resume_epoch(run_dir: str, name: str, progress_file: str,
                     unified: bool = False) -> int:
    os.makedirs(run_dir, exist_ok=True)
    last_ckpt = find_last_checkpoint(run_dir, name)

    if last_ckpt < 0:
        if os.path.exists(progress_file):
            # unified mode shares this file with selfplay, so never delete it --
            # a stale pretrain row is recoverable, a wiped run history is not
            if unified:
                print(f"[recovery] no checkpoints found, but "
                      f"{os.path.basename(progress_file)} is shared with "
                      f"selfplay -- leaving it in place. Clear it by hand for "
                      f"a true fresh start.")
            else:
                os.remove(progress_file)
                print(f"[recovery] no checkpoints found; cleared {progress_file}")
        return 0

    trim_progress_csv(progress_file, last_ckpt, unified=unified)
    resume = last_ckpt + 1
    print(
        f"[recovery] last checkpoint epoch={last_ckpt}"
        f"  ->  resuming from epoch {resume}"
    )
    return resume


# ---------------------------------------------------------------------------
# PT model load / save
# ---------------------------------------------------------------------------

def make_adam(model, lr: float):
    import torch
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, PT_ADAM_BETA2),
        weight_decay=1e-6,
    )


def load_pt_model(path: str, name: str, device, lr: float):
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
    try:
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            for pg in opt.param_groups:
                pg["lr"] = lr
    except Exception as e:
        print("[load] optimizer state incompatible (optimizer change?), starting fresh:", e)
        raise

    lw_state = ckpt.get("lw_state")
    return model, opt, scaler, lw_state


def save_pt_ckpt(model, opt, scaler, epoch: int, name: str, path: str,
                  lw_state: dict | None = None) -> None:
    import torch
    torch.save({
        "model":     model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler":    scaler.state_dict(),
        "epoch":     epoch,
        "arch":      name,
        "lw_state":  lw_state,
    }, path)
    print(f"[ckpt] epoch {epoch} -> {path}")


# ---------------------------------------------------------------------------
# Training and eval (called inside worker subprocess)
# ---------------------------------------------------------------------------

def pt_clipnorm_for_epoch(ep: int) -> float:
    if ep < 10:  return 1.0
    if ep < 20:  return 2.5
    if ep < 30:  return 5.0
    if ep < 40:  return 10.0
    return 20.0


def fit_epoch(model, opt, scaler, bundle, lw: dict, max_norm: float, device):
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    enc      = torch.as_tensor(bundle["enc_in"].numpy(),        dtype=torch.long).to(device)
    policy_t = torch.as_tensor(bundle["policy_logits"].numpy(), dtype=torch.float32).to(device)
    value_t  = torch.as_tensor(bundle["value_out"].numpy(),     dtype=torch.float32).to(device)
    w        = torch.as_tensor(bundle["weight"].numpy(),        dtype=torch.float32).to(device)

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
            v_loss = (
                F.cross_entropy(val, value_t[idx], reduction="none") * w[idx]
            ).mean() * lw["value_out"]
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
        total_p   += p_loss.item()
        total_v   += v_loss.item()
        total     += loss.item()
        n_batches += 1

    gn = np.array(grad_norms)
    grad_stats = {
        "gn_mean":  gn.mean(),
        "gn_median": np.median(gn),
        "gn_min":   gn.min(),
        "gn_max":   gn.max(),
        "gn_clips": clip_count,
        "gn_steps": n_batches,
    }
    return total_p / n_batches, total_v / n_batches, total / n_batches, grad_stats


def do_eval(model, name, epoch, bundle, eval_df, progress_file, plot_file, device,
            train_loss=None, gn_mean=None):
    import torch
    from torch.amp import autocast
    from chessbot.utils import batch_policy_metrics, print_validation

    enc = torch.as_tensor(bundle["enc_in"].numpy(), dtype=torch.long).to(device)
    model.eval()
    all_pol, all_val = [], []
    with torch.no_grad(), autocast("cuda"):
        for start in range(0, enc.shape[0], PT_BATCH_SIZE):
            pol, val = model(enc[start:start + PT_BATCH_SIZE])
            all_pol.append(pol.float().cpu().numpy())
            all_val.append(val.float().cpu().numpy())

    pol_preds  = np.concatenate(all_pol, axis=0)
    val_logits = np.concatenate(all_val, axis=0)
    target_wdl = bundle["value_out"].numpy()
    pstack     = bundle["policy_logits"].numpy()
    mstack     = (pstack > 0).astype(np.int32)

    val_wdl    = scipy.special.softmax(val_logits, axis=1)
    val_q      = val_wdl[:, 0] - val_wdl[:, 2]
    tgt_q      = target_wdl[:, 0] - target_wdl[:, 2]

    value_mse  = np.mean((val_q - tgt_q) ** 2)
    value_corr = np.corrcoef(val_q, tgt_q)[0, 1]
    value_ce   = -np.mean(
        np.sum(target_wdl * np.log(np.clip(val_wdl, 1e-9, None)), axis=1)
    )
    pol_stats  = batch_policy_metrics(pol_preds, pstack, mstack)

    # one row shape everywhere -- selfplay's schema. Only the filename changes
    # when this is not aimed at a run_dir.
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
# Worker — runs one block of EPOCHS_PER_WORKER epochs in a fresh subprocess
# ---------------------------------------------------------------------------

def worker_main(wargs: dict) -> None:
    import gc
    import torch
    from dotenv import load_dotenv
    load_dotenv()

    from chessbot import MODEL_DIR
    from chessbot.utils import format_time

    name        = wargs["name"]
    run_dir     = wargs["run_dir"]
    train_files = wargs["train_files"]
    val_files   = wargs["val_files"]
    start_epoch = wargs["start_epoch"]
    end_epoch   = wargs["end_epoch"]
    max_epoch   = wargs["max_epoch"]
    model_dir   = wargs.get("model_dir", MODEL_DIR)
    unified     = wargs.get("unified_csv", False)
    swa_set     = set(swa_epochs(max_epoch)) if wargs.get("swa") else set()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    progress_file = wargs.get("progress_file") or progress_csv_path(
        run_dir, name, unified)
    plot_file     = os.path.join(run_dir, f"{name}_pt_plot.png")

    print(f"\n{'#' * 72}")
    print(f"  {name}  epochs {start_epoch}..{end_epoch - 1}  ".center(72, "#"))
    print(f"{'#' * 72}\n")

    lr0 = lr_for_epoch(start_epoch, max_epoch)
    if start_epoch == 0:
        import torch
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
            raise RuntimeError(
                f"start_epoch={start_epoch} but no checkpoint found in {run_dir}"
            )
        load_from = ckpt_path(run_dir, name, last_ckpt)
        print(f"[worker] loading {load_from}")
        model, opt, scaler, lw_state = load_pt_model(load_from, name, device, lr=lr0)

    eval_df: pd.DataFrame | None = None
    if os.path.exists(progress_file):
        existing = pd.read_csv(progress_file)
        eval_df = normalize_progress_df(existing) if not existing.empty else None

    print(
        f"[tfrec] train={len(train_files)}  val={len(val_files)}"
        f"  batch={PT_BATCH_SIZE}  steps/epoch={PT_STEPS_PER_EPOCH}"
    )

    train_ds = make_dataset(train_files, PT_SHUFFLE_BUFFER, batch_size=PT_BATCH_SIZE)
    val_ds   = make_dataset(val_files, VAL_SHUFFLE_BUFFER, batch_size=PT_BATCH_SIZE)
    prefetcher     = EpochBufferThread(train_ds, PT_STEPS_PER_EPOCH, max_ready=2)
    val_prefetcher = EpochBufferThread(val_ds,   PT_STEPS_PER_EPOCH, max_ready=1)
    prefetcher.start()
    val_prefetcher.start()

    if lw_state is not None:
        policy_ce_hat = lw_state["policy_ce_hat"]
        value_ce_hat  = lw_state["value_ce_hat"]
        active_lw = {
            "policy_lw": lw_state["policy_lw"],
            "value_lw":  lw_state["value_lw"],
            "scale":     lw_state.get("scale"),
        }
        print(
            f"[worker] restored LW state: policy_ce_hat={policy_ce_hat:.3f}  "
            f"value_ce_hat={value_ce_hat:.3f}  scale={active_lw['scale']}"
        )
    else:
        policy_ce_hat: float | None = None
        value_ce_hat: float | None = None
        active_lw = {
            "policy_lw": LW_WARMUP_POLICY, "value_lw": LW_WARMUP_VALUE,
            "scale": None,
        }
        if start_epoch > LR_WARMUP_EPOCHS:
            print(
                "[worker] no LW state in checkpoint -- resuming with warmup LW "
                "until the next recompute"
            )
    current_lr: float | None = None
    begin          = time.time()
    epoch_times: list[float] = []
    t_fetch = t_fit = t_eval = 0.0
    total_samples  = 0
    window_samples = 0
    window_loss_sum = 0.0
    window_gn_sum   = 0.0
    window_count    = 0
    last_tgt_q: np.ndarray | None = None
    last_val_q: np.ndarray | None = None

    for ep in range(start_epoch, end_epoch):
        target_lr = lr_for_epoch(ep, max_epoch)
        if target_lr != current_lr:
            current_lr = target_lr
            for pg in opt.param_groups:
                pg["lr"] = target_lr
            print(f"[lr update] epoch {ep}: lr={target_lr:.4e}")

        epoch_start = time.time()
        print("-" * 89)

        if ep % PLOT_EVERY == 0 or ep == end_epoch - 1:
            t0 = time.time()
            val_bundle = val_prefetcher.get()
            avg_loss = window_loss_sum / window_count if window_count > 0 else None
            avg_gn   = window_gn_sum   / window_count if window_count > 0 else None
            eval_df, last_tgt_q, last_val_q = do_eval(
                model, name, ep, val_bundle, eval_df,
                progress_file, plot_file, device,
                train_loss=avg_loss, gn_mean=avg_gn,
            )
            window_loss_sum = 0.0
            window_gn_sum   = 0.0
            window_count    = 0
            t_eval += time.time() - t0

        t0 = time.time()
        bundle = prefetcher.get()
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

        n_this           = int(bundle["enc_in"].shape[0])
        total_samples   += n_this
        window_samples  += n_this
        window_loss_sum += t_loss
        window_gn_sum   += gns['gn_mean']
        window_count    += 1

        if ep == start_epoch and ep % PLOT_EVERY == 0 and eval_df is not None:
            eval_df.loc[eval_df.index[-1], 'train_loss'] = t_loss
            eval_df.loc[eval_df.index[-1], 'gn_mean']   = gns['gn_mean']
            eval_df.round(4).to_csv(progress_file, index=False)

        print(
            f"[epoch {ep:4d}] [{name}] "
            f"policy_loss: {p_loss:.2f}  value_loss: {v_loss:.2f}  "
            f"total: {t_loss:.2f}  samples: {total_samples:,}"
        )
        print(
            f"[epoch {ep:4d}] [{name}] "
            f"policy_ce_hat: {policy_ce_hat:.2f}  value_ce_hat (raw): {value_ce_hat:.3f}  "
            f"A = {live_A:.2f}"
        )
        if recomputed:
            print(
                f"[lw-fixed RECOMPUTE] scale: {scale:.3f}  "
                f"new LW: policy={policy_lw:.3f}  value={value_lw:.3f}"
            )
            print()
        print(
            f"[grad norms ] "
            f"mean: {gns['gn_mean']:.3f}  median: {gns['gn_median']:.3f}  "
            f"min: {gns['gn_min']:.3f}  max: {gns['gn_max']:.3f}  "
            f"clips: {gns['gn_clips']}/{gns['gn_steps']}"
        )

        is_ckpt = ep % CHECKPOINT_EVERY == 0 or ep == end_epoch - 1 or ep == max_epoch - 1
        if is_ckpt:
            cp = ckpt_path(run_dir, name, ep)
            lw_state = {
                "policy_ce_hat": policy_ce_hat,
                "value_ce_hat":  value_ce_hat,
                "policy_lw":     active_lw["policy_lw"],
                "value_lw":      active_lw["value_lw"],
                "scale":         active_lw["scale"],
            }
            save_pt_ckpt(model, opt, scaler, ep, name, cp, lw_state=lw_state)
            delete_old_checkpoints(run_dir, name, keep_epoch=ep)
            save_plot(eval_df, f"{name}  [PT]", ep, plot_file, last_tgt_q, last_val_q)

        if ep in swa_set:
            sp = swa_ckpt_path(run_dir, name, ep)
            torch.save({"model": model.state_dict(), "arch": name}, sp)
            print(f"[swa] snapshot {len(swa_set)} -> {os.path.basename(sp)}")

        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        print(
            f"[time check] last: {format_time(elapsed)}  "
            f"avg: {format_time(np.mean(epoch_times))}  "
            f"total: {format_time(time.time() - begin)}"
        )

        if ep % PLOT_EVERY == 0 and ep > start_epoch:
            print(
                f"[timing/{PLOT_EVERY}ep] fetch: {t_fetch:.2f}s  "
                f"fit: {t_fit:.2f}s  eval: {t_eval:.2f}s  "
                f"samples_window: {window_samples:,}  "
                f"samples_total: {total_samples:,}"
            )
            t_fetch = t_fit = t_eval = 0.0
            window_samples = 0

        print()

    prefetcher.stop()
    val_prefetcher.stop()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[worker] {name}: block complete (epochs {start_epoch}..{end_epoch - 1})")


# ---------------------------------------------------------------------------
# One-time end-of-pretrain seeding: drain the held-out validation split
# (never trained on, same distribution as the training set) into the
# selfplay run's buffers so the first ~10 retrains aren't starved of
# volume and the transition into self-correlated selfplay data is a
# gradual blend instead of a jolt. Only called when --run-dir is given
# (i.e. not the default val_test_multi dev/test run_tag).
# ---------------------------------------------------------------------------

def seed_selfplay_buffers(run_dir: str, val_files: list[str]) -> None:
    replay_dir     = os.path.join(run_dir, "replay_buffer")
    primary_dir    = os.path.join(run_dir, "primary_buffer")
    remaining_path = os.path.join(run_dir, "remaining_untrained.pkl")

    os.makedirs(replay_dir, exist_ok=True)
    os.makedirs(primary_dir, exist_ok=True)

    val_files = list(val_files)
    status = {}

    # replay: raw .tfrecord.gz copy, same convention as seed_replay_buffer /
    # the retrain-time historic boosters -- no re-encoding needed.
    existing_replay = os.listdir(replay_dir)
    replay_needed = max(0, SEED_REPLAY_SHARDS - len(existing_replay))
    if replay_needed == 0:
        status["replay"] = (
            f"skipped ({len(existing_replay)}/{SEED_REPLAY_SHARDS} already present)"
        )
    else:
        n_take = min(replay_needed, len(val_files))
        chosen = random.sample(val_files, n_take)
        for src in chosen:
            shutil.copy2(src, os.path.join(replay_dir, os.path.basename(src)))
        chosen_set = set(chosen)
        val_files = [f for f in val_files if f not in chosen_set]
        kind = "full fill" if len(existing_replay) == 0 else "partial fill (top up)"
        status["replay"] = f"{kind}: copied {n_take} tfrecord.gz files"
        if n_take < replay_needed:
            status["replay"] += f" (only {n_take}/{replay_needed} val files available)"
            status["primary"] = "skipped (no val_files left after replay)"
            status["remaining_untrained"] = "skipped (no val_files left after replay)"
            print("[seed] " + " | ".join(f"{k}: {v}" for k, v in status.items()))
            return

    # primary + remaining_untrained: drained/re-encoded via TF pipeline,
    # reuse=False, into pkl.gz shards matching the live selfplay schema.
    existing_primary = os.listdir(primary_dir)
    primary_needed = max(0, SEED_PRIMARY_SHARDS - len(existing_primary))
    remaining_exists = os.path.exists(remaining_path)

    ds_iter = None
    if primary_needed > 0 or not remaining_exists:
        ds = make_dataset(
            val_files, shuffle_buffer=2 * VAL_SHUFFLE_BUFFER,
            batch_size=EPOCH_SIZE, repeat=False,
        )
        ds_iter = iter(ds)

    def drain_epoch():
        inp, out, _ = next(ds_iter)
        enc = inp["enc_in"].numpy()
        pol = out["policy_logits"].numpy()
        val = out["value_out"].numpy()
        return [
            (enc[i], None, pol[i], val[i], 1.0, 1.0, SEED_SOURCE)
            for i in range(enc.shape[0])
        ]

    if primary_needed == 0:
        status["primary"] = (
            f"skipped ({len(existing_primary)}/{SEED_PRIMARY_SHARDS} already present)"
        )
        
    else:
        for _ in range(primary_needed):
            write_pkl_gz_shard(drain_epoch(), new_shard_path(primary_dir))
        kind = "full fill" if len(existing_primary) == 0 else "partial fill (top up)"
        status["primary"] = f"{kind}: wrote {primary_needed} shards"

    if remaining_exists:
        status["remaining_untrained"] = "skipped (already present)"
    else:
        remaining_records = []
        for _ in range(SEED_REMAINING_SHARDS):
            remaining_records.extend(drain_epoch())
        with open(remaining_path, "wb") as f:
            pickle.dump(remaining_records, f)
        status["remaining_untrained"] = (
            f"full fill: wrote {len(remaining_records):,} records"
        )

    print("[seed] " + " | ".join(f"{k}: {v}" for k, v in status.items()))


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

def spawn_block(wargs: dict) -> int:
    ctx = mp.get_context("spawn")
    p   = ctx.Process(target=worker_main, args=(wargs,), daemon=False)
    p.start()
    p.join()
    return p.exitcode if p.exitcode is not None else -1


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from chessbot import SP_DIR, MODEL_DIR

    parser = argparse.ArgumentParser(
        description="OOM-robust PyTorch bootstrap trainer for Xerces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",    default=DEFAULT_MODEL,   help="model name")
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG, help="run tag under SP_DIR")
    parser.add_argument(
        "--run-dir", default=None, help="explicit run dir (overrides --run-tag)"
    )
    parser.add_argument(
        "--tfrec-dir",
        default=DEFAULT_TFREC_DIR,
        help="path to .tfrecord.gz files",
    )
    parser.add_argument("--max-epoch", type=int, default=DEFAULT_MAX_EPOCH)
    parser.add_argument(
        "--swa", action="store_true",
        help=f"snapshot every {SWA_EVERY} epochs over the last {SWA_WINDOW} "
             f"and export their average as {{run_tag}}_model_SWA.pt/.ts",
    )
    args = parser.parse_args()

    if not args.tfrec_dir:
        parser.error("--tfrec-dir is required (or set $BOOTSTRAP_TFREC_DIR)")

    run_dir = args.run_dir or os.path.join(SP_DIR, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)

    # --run-dir means this directory is the selfplay run, so metrics go straight
    # into its eval_progress.csv and the first retrain continues the same
    # history. Same signal that gates seed_selfplay_buffers below.
    unified = args.run_dir is not None

    name = args.model
    progress_file = progress_csv_path(run_dir, name, unified)
    from chessbot.pretrain import LR_MIN, LR_MAX
    print(f"[train] model={name}")
    print(f"[train] run_dir={run_dir}")
    print(f"[train] progress_csv={os.path.basename(progress_file)}")
    print(f"[train] tfrec_dir={args.tfrec_dir}")
    print(f"[train] max_epoch={args.max_epoch}")
    print(f"[train] batch={PT_BATCH_SIZE}  steps/epoch={PT_STEPS_PER_EPOCH}"
          f"  lr_range=[{LR_MIN:.1e}, {LR_MAX:.1e}]")

    all_files = list_tfrecord_files(args.tfrec_dir)
    if not all_files:
        parser.error(f"no .tfrecord.gz files found in {args.tfrec_dir}")

    train_files, val_files = split_train_val(all_files)
    print(f"[train] train_files={len(train_files)}  val_files={len(val_files)}")

    resume = get_resume_epoch(run_dir, name, progress_file, unified=unified)

    wargs = {
        "name":          name,
        "run_dir":       run_dir,
        "train_files":   train_files,
        "val_files":     val_files,
        "model_dir":     MODEL_DIR,
        "start_epoch":   resume,
        "end_epoch":     args.max_epoch,
        "max_epoch":     args.max_epoch,
        "progress_file": progress_file,
        "unified_csv":   unified,
        "swa":           args.swa,
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
        cfg = VARIANTS[arch_name]
        model = PT_BUILDERS[arch_name](cfg)
        model.load_state_dict(raw["model"])
        run_tag   = os.path.basename(run_dir)
        pt_path   = os.path.join(run_dir, f"{run_tag}_model.pt")
        ts_path   = os.path.join(run_dir, f"{run_tag}_model.ts")
        torch.save(raw["model"], pt_path)
        export_ts(model, ts_path, arch=arch_name)
        print(f"[train] export complete -> {pt_path}")
        print(f"[train] export complete -> {ts_path}")

        # Lift-and-shift the pretraining optimizer state into selfplay's
        # retrain opt-state slot, so the first selfplay retrain warm-starts
        # from the same Adam moments the model actually converged under
        # instead of starting cold against an already-converged model.
        if "optimizer" in raw:
            from chessbot.train_pytorch import pt_opt_state_path
            opt_state_path = pt_opt_state_path(pt_path, run_dir)
            os.makedirs(os.path.dirname(opt_state_path), exist_ok=True)
            torch.save(raw["optimizer"], opt_state_path)
            print(f"[train] exported optimizer state -> {opt_state_path}")
        else:
            print("[train] no optimizer state in checkpoint -- "
                  "selfplay retrain will start fresh")

        if args.swa:
            snaps = [swa_ckpt_path(run_dir, name, e)
                     for e in swa_epochs(args.max_epoch)]
            found = [p for p in snaps if os.path.exists(p)]
            missing = len(snaps) - len(found)
            if not found:
                print("[swa] no snapshots found -- skipping SWA export")
            else:
                if missing:
                    print(f"[swa] {missing} of {len(snaps)} snapshots missing, "
                          f"averaging the {len(found)} that are present")
                sds = [torch.load(p, map_location="cpu")["model"] for p in found]
                avg = average_state_dicts(sds)

                swa_model = PT_BUILDERS[arch_name](VARIANTS[arch_name])
                swa_model.load_state_dict(avg)
                swa_pt = os.path.join(run_dir, f"{run_tag}_model_SWA.pt")
                swa_ts = os.path.join(run_dir, f"{run_tag}_model_SWA.ts")
                torch.save(avg, swa_pt)
                export_ts(swa_model, swa_ts, arch=arch_name)
                print(f"[swa] averaged {len(found)} snapshots -> {swa_pt}")
                print(f"[swa] averaged {len(found)} snapshots -> {swa_ts}")

                # only now, with both artifacts confirmed on disk, drop the
                # snapshots. The final training checkpoint is a _pt_ckpt file,
                # a different pattern, so it cannot be caught here -- assert it
                # anyway rather than trust that by inspection.
                final_ckpt = ckpt_path(run_dir, name, last_ckpt)
                if os.path.exists(swa_pt) and os.path.exists(swa_ts):
                    for p in found:
                        if p == final_ckpt:
                            continue
                        os.remove(p)
                        print(f"[swa] removed snapshot {os.path.basename(p)}")
                    if not os.path.exists(final_ckpt):
                        raise RuntimeError(
                            f"[swa] final checkpoint {final_ckpt} disappeared "
                            f"during snapshot cleanup")
                    print(f"[swa] final checkpoint intact: "
                          f"{os.path.basename(final_ckpt)}")
                else:
                    print("[swa] SWA .pt/.ts not both present -- keeping "
                          "snapshots")

    if args.run_dir:
        seed_selfplay_buffers(run_dir, val_files)
    else:
        print(
            f"[seed] no --run-dir given (run_tag={args.run_tag}); "
            "skipping selfplay buffer seeding"
        )


if __name__ == "__main__":
    main()
