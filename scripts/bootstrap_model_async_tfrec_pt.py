"""bootstrap_model_async_tfrec_pt.py — OOM-robust PyTorch bootstrap trainer

The supervisor spawns a fresh subprocess for each block of epochs.
On crash the supervisor rolls back to the last checkpoint and restarts.

PyTorch owns CUDA entirely.  TF is restricted to CPU (tfrecord reading only).

Pre-saved .pt source models must exist in MODEL_DIR before running.

Usage:
    python scripts/bootstrap_model_async_tfrec_pt.py [options]

Options:
    --model      Model name            (default: 16m-transformer)
    --run-tag    Sub-dir under SP_DIR  (default: val_test_multi)
    --run-dir    Explicit run dir (overrides --run-tag)
    --tfrec-dir  Path to .tfrecord.gz files  (env: BOOTSTRAP_TFREC_DIR)
    --max-epoch  Total epochs to train        (default: 2000)
    --lr         Peak learning rate           (default: 1e-2)
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.special

from chessbot.pretrain import (
    EPOCH_SIZE, SHUFFLE_BUFFER, VAL_SHUFFLE_BUFFER,
    PLOT_EVERY, DEFAULT_MAX_EPOCH,
    POLICY_LW, VALUE_LW, LR_WARMUP_EPOCHS,
    lr_for_epoch,
    list_tfrecord_files, split_train_val,
    make_dataset, EpochBufferThread,
    save_plot,
)
from chessbot.model import (
    VOCAB_SIZE, SEQ_LEN,
    VARIANTS,
    PT_BUILDERS,
)

# PT uses 2x batch vs TF -> same epoch size, 2x LR (linear scaling rule)
PT_BATCH_SIZE      = 512
PT_STEPS_PER_EPOCH = EPOCH_SIZE // PT_BATCH_SIZE   # 20
PT_LR_SCALE        = 2.0
PT_MOMENTUM        = 0.9
PT_WEIGHT_DECAY    = 1e-4

EPOCHS_PER_WORKER = 1000
CHECKPOINT_EVERY  = 20
DEFAULT_MODEL     = "16m-transformer"
DEFAULT_RUN_TAG   = "val_test_multi"



# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def ckpt_path(run_dir: str, name: str, epoch: int) -> str:
    return os.path.join(run_dir, f"{name}_pt_ckpt{epoch:04d}.pt")


def find_last_checkpoint(run_dir: str, name: str) -> int:
    pat = re.compile(rf"^{re.escape(name)}_pt_ckpt(\d{{4}})\.pt$")
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
    pat = re.compile(rf"^{re.escape(name)}_pt_ckpt(\d{{4}})\.pt$")
    try:
        for fname in os.listdir(run_dir):
            m = pat.match(fname)
            if m and int(m.group(1)) != keep_epoch:
                path = os.path.join(run_dir, fname)
                os.remove(path)
                print(f"[ckpt] removed old checkpoint {fname}")
    except FileNotFoundError:
        pass


def trim_progress_csv(progress_file: str, max_training_epoch: int) -> None:
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


def get_resume_epoch(run_dir: str, name: str) -> int:
    os.makedirs(run_dir, exist_ok=True)
    progress_file = os.path.join(run_dir, f"{name}_pt_eval_progress.csv")
    last_ckpt = find_last_checkpoint(run_dir, name)

    if last_ckpt < 0:
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"[recovery] no checkpoints found; cleared {progress_file}")
        return 0

    trim_progress_csv(progress_file, last_ckpt)
    resume = last_ckpt + 1
    print(
        f"[recovery] last checkpoint epoch={last_ckpt}"
        f"  ->  resuming from epoch {resume}"
    )
    return resume


# ---------------------------------------------------------------------------
# PT model load / save
# ---------------------------------------------------------------------------

def make_sgd(model, lr: float):
    import torch
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=PT_MOMENTUM,
        weight_decay=PT_WEIGHT_DECAY,
        nesterov=True,
    )


def load_pt_model(path: str, name: str, device, lr: float):
    import torch
    from torch.amp import GradScaler

    cfg    = VARIANTS[name]
    model  = PT_BUILDERS[name](cfg).to(device)
    opt    = make_sgd(model, lr)
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

    return model, opt, scaler


def save_pt_ckpt(model, opt, scaler, epoch: int, name: str, path: str) -> None:
    import torch
    torch.save({
        "model":     model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler":    scaler.state_dict(),
        "epoch":     epoch,
        "arch":      name,
    }, path)
    print(f"[ckpt] epoch {epoch} -> {path}")


# ---------------------------------------------------------------------------
# Training and eval (called inside worker subprocess)
# ---------------------------------------------------------------------------

def pt_clipnorm_for_epoch(ep: int) -> float:
    return 1.0 if ep < LR_WARMUP_EPOCHS else 2.5


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


def do_eval(model, name, epoch, bundle, eval_df, progress_file, plot_file, device):
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
    mstack     = bundle["mask"].numpy()

    val_wdl    = scipy.special.softmax(val_logits, axis=1)
    val_q      = val_wdl[:, 0] - val_wdl[:, 2]
    tgt_q      = target_wdl[:, 0] - target_wdl[:, 2]

    value_mse  = np.mean((val_q - tgt_q) ** 2)
    value_corr = np.corrcoef(val_q, tgt_q)[0, 1]
    value_ce   = -np.mean(
        np.sum(target_wdl * np.log(np.clip(val_wdl, 1e-9, None)), axis=1)
    )
    pol_stats  = batch_policy_metrics(pol_preds, pstack, mstack)

    row = {
        "training_epoch": epoch,
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

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    name        = wargs["name"]
    run_dir     = wargs["run_dir"]
    train_files = wargs["train_files"]
    val_files   = wargs["val_files"]
    start_epoch = wargs["start_epoch"]
    end_epoch   = wargs["end_epoch"]
    max_epoch   = wargs["max_epoch"]
    model_dir   = wargs.get("model_dir", MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    progress_file = os.path.join(run_dir, f"{name}_pt_eval_progress.csv")
    plot_file     = os.path.join(run_dir, f"{name}_pt_plot.png")

    print(f"\n{'#' * 72}")
    print(f"  {name}  epochs {start_epoch}..{end_epoch - 1}  ".center(72, "#"))
    print(f"{'#' * 72}\n")

    lr0 = lr_for_epoch(start_epoch, max_epoch, scale=PT_LR_SCALE)
    if start_epoch == 0:
        import torch
        from torch.amp import GradScaler
        cfg    = VARIANTS[name]
        model  = PT_BUILDERS[name](cfg).to(device)
        opt    = make_sgd(model, lr0)
        scaler = GradScaler("cuda")
        print(f"[worker] fresh init  lr={lr0:.4e}")
    else:
        last_ckpt = find_last_checkpoint(run_dir, name)
        if last_ckpt < 0:
            raise RuntimeError(
                f"start_epoch={start_epoch} but no checkpoint found in {run_dir}"
            )
        load_from = ckpt_path(run_dir, name, last_ckpt)
        print(f"[worker] loading {load_from}")
        model, opt, scaler = load_pt_model(load_from, name, device, lr=lr0)

    eval_df: pd.DataFrame | None = None
    if os.path.exists(progress_file):
        existing = pd.read_csv(progress_file)
        eval_df = existing if not existing.empty else None

    print(
        f"[tfrec] train={len(train_files)}  val={len(val_files)}"
        f"  batch={PT_BATCH_SIZE}  steps/epoch={PT_STEPS_PER_EPOCH}"
    )

    prefetcher = EpochBufferThread(
        make_dataset(train_files, SHUFFLE_BUFFER, batch_size=PT_BATCH_SIZE),
        PT_STEPS_PER_EPOCH, max_ready=2,
    )
    val_prefetcher = EpochBufferThread(
        make_dataset(val_files, VAL_SHUFFLE_BUFFER, batch_size=PT_BATCH_SIZE),
        PT_STEPS_PER_EPOCH, max_ready=1,
    )
    prefetcher.start()
    val_prefetcher.start()

    fixed_lw = {"policy_logits": POLICY_LW, "value_out": VALUE_LW}
    current_lr: float | None = None
    begin          = time.time()
    epoch_times: list[float] = []
    t_fetch = t_fit = t_eval = 0.0
    total_samples  = 0
    window_samples = 0
    last_tgt_q: np.ndarray | None = None
    last_val_q: np.ndarray | None = None

    for ep in range(start_epoch, end_epoch):
        target_lr = lr_for_epoch(ep, max_epoch, scale=PT_LR_SCALE)
        if target_lr != current_lr:
            current_lr = target_lr
            for pg in opt.param_groups:
                pg["lr"] = target_lr
            print(f"[lr update] epoch {ep}: lr={target_lr:.4e}")

        epoch_start = time.time()
        print("-" * 89)

        if ep % PLOT_EVERY == 0:
            t0 = time.time()
            val_bundle = val_prefetcher.get()
            eval_df, last_tgt_q, last_val_q = do_eval(
                model, name, ep, val_bundle, eval_df,
                progress_file, plot_file, device,
            )
            t_eval += time.time() - t0

        t0 = time.time()
        bundle = prefetcher.get()
        t_fetch += time.time() - t0

        t0 = time.time()
        max_norm = pt_clipnorm_for_epoch(ep)
        p_loss, v_loss, t_loss, gns = fit_epoch(model, opt, scaler, bundle, fixed_lw, max_norm, device)
        t_fit += time.time() - t0

        n_this          = int(bundle["enc_in"].shape[0])
        total_samples  += n_this
        window_samples += n_this

        print(
            f"[epoch {ep:4d}] [{name}] "
            f"policy_loss: {p_loss:.4f}  value_loss: {v_loss:.4f}  "
            f"total: {t_loss:.4f}  samples: {total_samples:,}"
        )
        print(
            f"[grad norms ] "
            f"mean: {gns['gn_mean']:.3f}  median: {gns['gn_median']:.3f}  "
            f"min: {gns['gn_min']:.3f}  max: {gns['gn_max']:.3f}  "
            f"clips: {gns['gn_clips']}/{gns['gn_steps']}"
        )

        is_ckpt = ep % CHECKPOINT_EVERY == 0 or ep == end_epoch - 1 or ep == max_epoch - 1
        if is_ckpt:
            cp = ckpt_path(run_dir, name, ep)
            save_pt_ckpt(model, opt, scaler, ep, name, cp)
            delete_old_checkpoints(run_dir, name, keep_epoch=ep)
            save_plot(eval_df, f"{name}  [PT]", ep, plot_file, last_tgt_q, last_val_q)

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
    parser.add_argument("--run-tag",  default=DEFAULT_RUN_TAG, help="run tag under SP_DIR")
    parser.add_argument(
        "--run-dir", default=None, help="explicit run dir (overrides --run-tag)"
    )
    parser.add_argument(
        "--tfrec-dir",
        default=os.getenv(
            "BOOTSTRAP_TFREC_DIR",
            r"C:\Users\Bryan\Data\chessbot_data\training_data\wdl",
        ),
        help="path to .tfrecord.gz files  (env: BOOTSTRAP_TFREC_DIR)",
    )
    parser.add_argument("--max-epoch", type=int,   default=DEFAULT_MAX_EPOCH)
    args = parser.parse_args()

    if not args.tfrec_dir:
        parser.error("--tfrec-dir is required (or set $BOOTSTRAP_TFREC_DIR)")

    run_dir = args.run_dir or os.path.join(SP_DIR, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)

    name = args.model
    print(f"[supervisor] model={name}")
    print(f"[supervisor] run_dir={run_dir}")
    print(f"[supervisor] tfrec_dir={args.tfrec_dir}")
    from chessbot.pretrain import LR_MIN, LR_MAX
    print(f"[supervisor] max_epoch={args.max_epoch}")
    print(f"[supervisor] batch={PT_BATCH_SIZE}  steps/epoch={PT_STEPS_PER_EPOCH}"
          f"  lr_range=[{LR_MIN * PT_LR_SCALE:.1e}, {LR_MAX * PT_LR_SCALE:.1e}]")

    all_files = list_tfrecord_files(args.tfrec_dir)
    if not all_files:
        parser.error(f"no .tfrecord.gz files found in {args.tfrec_dir}")

    train_files, val_files = split_train_val(all_files)
    print(f"[supervisor] train_files={len(train_files)}  val_files={len(val_files)}")

    resume = get_resume_epoch(run_dir, name)

    while resume < args.max_epoch:
        block_start = (resume // EPOCHS_PER_WORKER) * EPOCHS_PER_WORKER
        end_epoch   = min(block_start + EPOCHS_PER_WORKER, args.max_epoch)

        print(
            f"\n[supervisor] block {block_start // EPOCHS_PER_WORKER}"
            f"  epochs {resume}..{end_epoch - 1}"
        )

        wargs = {
            "name":        name,
            "run_dir":     run_dir,
            "train_files": train_files,
            "val_files":   val_files,
            "model_dir":   MODEL_DIR,
            "start_epoch": resume,
            "end_epoch":   end_epoch,
            "max_epoch":   args.max_epoch,
        }

        exit_code = spawn_block(wargs)

        if exit_code == 0:
            resume = end_epoch
            print(f"[supervisor] block complete  ->  resume={resume}")
        else:
            print(f"[supervisor] worker crashed (exit={exit_code}), recovering ...")
            time.sleep(5)
            resume = get_resume_epoch(run_dir, name)
            print(f"[supervisor] will retry from epoch {resume}")

    print(f"\n[supervisor] training complete ({args.max_epoch} epochs)")


if __name__ == "__main__":
    main()
