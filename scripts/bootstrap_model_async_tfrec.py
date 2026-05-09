#!/usr/bin/env python3
"""
bootstrap_model_async_tfrec.py — OOM-robust bootstrap trainer

The supervisor spawns a fresh subprocess for each 150-epoch block.
On crash / OOM the supervisor rolls back to the last checkpoint
(saved every 20 epochs) and restarts with a fresh random data stream.

Usage:
    python scripts/bootstrap_model_async_tfrec.py [options]

Options:
    --model      Model name          (default: 16m-conformer-interweaved)
    --run-tag    Sub-dir under SP_DIR (default: val_test_multi)
    --run-dir    Explicit run dir (overrides --run-tag)
    --tfrec-dir  Path to .tfrecord.gz files  (env: BOOTSTRAP_TFREC_DIR)
    --max-epoch  Total epochs to train       (default: 1001)
    --lr         Learning rate               (default: 2e-4)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import time

import numpy as np
import pandas as pd

from chessbot.pretrain import (
    BATCH_SIZE, STEPS_PER_EPOCH, SHUFFLE_BUFFER, VAL_SHUFFLE_BUFFER,
    PLOT_EVERY, DEFAULT_LR, DEFAULT_MAX_EPOCH,
    LW_SCHEDULE, lw_for_epoch,
    list_tfrecord_files, split_train_val,
    make_dataset, EpochBufferThread,
    moving_average, save_plot,
)

EPOCHS_PER_WORKER = 100
CHECKPOINT_EVERY  = 20    # must be a multiple of PLOT_EVERY
DEFAULT_MODEL     = "16m-conformer-interweaved"
DEFAULT_RUN_TAG   = "val_test_multi"


def ckpt_path(run_dir: str, name: str, epoch: int) -> str:
    return os.path.join(run_dir, f"{name}_ckpt{epoch:04d}.h5")


def delete_old_checkpoints(run_dir: str, name: str, keep_epoch: int) -> None:
    """Delete all checkpoint files except the one at keep_epoch."""
    pat = re.compile(rf"^{re.escape(name)}_ckpt(\d{{4}})\.h5$")
    try:
        for fname in os.listdir(run_dir):
            m = pat.match(fname)
            if m and int(m.group(1)) != keep_epoch:
                path = os.path.join(run_dir, fname)
                os.remove(path)
                print(f"[ckpt] removed old checkpoint {fname}")
    except FileNotFoundError:
        pass


def find_last_checkpoint(run_dir: str, name: str) -> int:
    """Return epoch number of the newest checkpoint file, or -1 if none."""
    pat = re.compile(rf"^{re.escape(name)}_ckpt(\d{{4}})\.h5$")
    best = -1
    try:
        for fname in os.listdir(run_dir):
            m = pat.match(fname)
            if m:
                best = max(best, int(m.group(1)))
    except FileNotFoundError:
        pass
    return best


def trim_progress_csv(progress_file: str, max_training_epoch: int) -> None:
    """Drop rows where training_epoch > max_training_epoch, overwrite in-place."""
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
    """Scan checkpoint files, trim the eval CSV to match, return next epoch."""
    os.makedirs(run_dir, exist_ok=True)
    progress_file = os.path.join(run_dir, f"{name}_eval_progress.csv")
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


def worker_main(wargs: dict) -> None:
    """Train for one block of epochs. All TF imports are local so the
    supervisor process never touches the GPU."""
    import gc
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import mixed_precision

    from chessbot import MODEL_DIR
    from chessbot.model import set_loss_weights
    from chessbot.utils import batch_policy_metrics, format_time, print_validation

    mixed_precision.set_global_policy("mixed_float16")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    name        = wargs["name"]
    run_dir     = wargs["run_dir"]
    train_files = wargs["train_files"]
    val_files   = wargs["val_files"]
    model_dir   = wargs.get("model_dir", MODEL_DIR)
    start_epoch = wargs["start_epoch"]
    end_epoch   = wargs["end_epoch"]
    lr          = wargs.get("lr", DEFAULT_LR)
    max_epoch   = wargs["max_epoch"]

    progress_file = os.path.join(run_dir, f"{name}_eval_progress.csv")
    plot_file     = os.path.join(run_dir, f"{name}_plot.png")

    if start_epoch == 0:
        load_from = os.path.join(model_dir, f"{name}_model.h5")
    else:
        last_ckpt = find_last_checkpoint(run_dir, name)
        if last_ckpt < 0:
            raise RuntimeError(
                f"start_epoch={start_epoch} but no checkpoint found in {run_dir}"
            )
        load_from = ckpt_path(run_dir, name, last_ckpt)

    print(f"\n{'#' * 72}")
    print(f"  {name}  epochs {start_epoch}..{end_epoch - 1}  ".center(72, "#"))
    print(f"{'#' * 72}\n")
    print(f"[worker] loading {load_from}")

    model = keras.models.load_model(load_from, compile=False)
    opt   = tf.keras.optimizers.Adam(learning_rate=lr)
    opt   = mixed_precision.LossScaleOptimizer(opt)
    model._default_opt = opt
    model._default_loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out":     tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    }
    set_loss_weights(model, {"policy_logits": 1.0, "value_out": 1.0}, jit=False)
    current_lw: dict | None = None

    train_ckpts_dir = os.path.join(run_dir, "train_ckpts")
    os.makedirs(train_ckpts_dir, exist_ok=True)
    epoch_var = tf.Variable(start_epoch, trainable=False, dtype=tf.int64)
    tf_ckpt   = tf.train.Checkpoint(model=model, optimizer=opt, epoch=epoch_var)
    manager   = tf.train.CheckpointManager(tf_ckpt, train_ckpts_dir, max_to_keep=2)
    if manager.latest_checkpoint:
        tf_ckpt.restore(manager.latest_checkpoint)
        print(f"[train_ckpt] restored from {manager.latest_checkpoint}")
    else:
        print("[train_ckpt] no prior optimizer checkpoint — starting fresh")

    eval_df: pd.DataFrame | None = None
    if os.path.exists(progress_file):
        existing = pd.read_csv(progress_file)
        eval_df = existing if not existing.empty else None

    print(
        f"[tfrec] train={len(train_files)}  val={len(val_files)}"
        f"  shuffle_buffer={SHUFFLE_BUFFER}"
    )

    prefetcher = EpochBufferThread(
        make_dataset(train_files, SHUFFLE_BUFFER), STEPS_PER_EPOCH, max_ready=2
    )
    val_prefetcher = EpochBufferThread(
        make_dataset(val_files, VAL_SHUFFLE_BUFFER), STEPS_PER_EPOCH, max_ready=1
    )
    prefetcher.start()
    val_prefetcher.start()

    def fit_epoch(bundle, lw: dict):
        x = {"enc_in": bundle["enc_in"]}
        if "mask" in model.input_names:
            x["mask"] = bundle["mask"]
        y = {
            "policy_logits": bundle["policy_logits"],
            "value_out":     bundle["value_out"],
        }
        sw = {
            "policy_logits": tf.ones_like(bundle["weight"]) * lw["policy_logits"],
            "value_out":     bundle["weight"] * lw["value_out"],
        }
        return model.fit(
            x=x, y=y, sample_weight=sw,
            batch_size=BATCH_SIZE, epochs=1, verbose=0, shuffle=False,
        )

    def do_eval(bundle, ep: int):
        nonlocal eval_df
        x = {"enc_in": bundle["enc_in"]}
        if "mask" in model.input_names:
            x["mask"] = bundle["mask"]

        preds      = model.predict(x, verbose=0, batch_size=64)
        pol_preds  = preds[0]
        val_logits = preds[1]                         # (N, 3) raw WDL logits
        pstack     = bundle["policy_logits"].numpy()
        target_wdl = bundle["value_out"].numpy()      # (N, 3)
        mstack     = bundle["mask"].numpy()

        shifted  = val_logits - val_logits.max(axis=1, keepdims=True)
        val_wdl  = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

        pred_q   = val_wdl[:, 0] - val_wdl[:, 2]
        target_q = target_wdl[:, 0] - target_wdl[:, 2]

        value_mse  = float(np.mean((pred_q - target_q) ** 2))
        value_corr = float(np.corrcoef(pred_q, target_q)[0, 1])
        value_ce   = float(-np.mean(
            np.sum(target_wdl * np.log(np.clip(val_wdl, 1e-9, None)), axis=1)
        ))

        pol_stats = batch_policy_metrics(pol_preds, pstack, mstack)
        row = {
            "training_epoch": ep,
            "value_mse": value_mse, "value_corr": value_corr, "value_ce": value_ce,
        }
        row.update(pol_stats)
        new_row = pd.DataFrame([row])
        eval_df = (
            new_row if eval_df is None
            else pd.concat([eval_df, new_row], ignore_index=True)
        )
        eval_df.round(4).to_csv(progress_file, index=False)

        print_metrics = {
            "value_mse": value_mse, "value_corr": value_corr, "value_ce": value_ce,
            **pol_stats
        }
        print_validation(ep, print_metrics)
        return eval_df, target_q, pred_q

    begin          = time.time()
    epoch_times: list[float] = []
    t_fetch = t_fit = t_eval = 0.0
    total_samples  = 0
    window_samples = 0
    last_ystack: np.ndarray | None    = None
    last_val_preds: np.ndarray | None = None

    for ep in range(start_epoch, end_epoch):
        target_lw = lw_for_epoch(ep)
        if target_lw is not current_lw:
            current_lw = target_lw
            lw_str = "  ".join(f"{k}={v}" for k, v in target_lw.items())
            print(f"[lw update] epoch {ep}: {lw_str}")

        epoch_start = time.time()
        print("-" * 89)

        if ep % PLOT_EVERY == 0:
            t0 = time.time()
            val_bundle = val_prefetcher.get()
            eval_df, last_ystack, last_val_preds = do_eval(val_bundle, ep)
            t_eval += time.time() - t0

        t0 = time.time()
        bundle = prefetcher.get()
        t_fetch += time.time() - t0

        t0   = time.time()
        hist = fit_epoch(bundle, target_lw)
        t_fit += time.time() - t0

        n_this          = int(bundle["enc_in"].shape[0])
        total_samples  += n_this
        window_samples += n_this

        h = hist.history
        print(
            f"[epoch {ep:4d}] [{name}] "
            f"policy_loss: {h.get('policy_logits_loss', [float('nan')])[0]:.4f}  "
            f"value_loss: {h.get('value_out_loss', [float('nan')])[0]:.4f}  "
            f"total: {h.get('loss', [float('nan')])[0]:.4f}  "
            f"samples: {total_samples:,}"
        )

        is_ckpt = ep % CHECKPOINT_EVERY == 0
        if is_ckpt or ep == end_epoch - 1 or ep == max_epoch - 1:
            cp = ckpt_path(run_dir, name, ep)
            model.save(cp)
            print(f"[ckpt] epoch {ep} -> {cp}")
            delete_old_checkpoints(run_dir, name, keep_epoch=ep)
            epoch_var.assign(ep)
            manager.save()
            print(f"[train_ckpt] optimizer state -> {manager.latest_checkpoint}")
            run_tag = os.path.basename(run_dir)
            save_plot(
                eval_df, f"{name}  [{run_tag}]", ep, plot_file,
                last_ystack, last_val_preds,
            )

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
    print(f"[worker] {name}: block complete (epochs {start_epoch}..{end_epoch - 1})")


def spawn_block(wargs: dict) -> int:
    """Spawn one block as a subprocess; return its exit code."""
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
        description="OOM-robust bootstrap trainer for Xerces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",     default=DEFAULT_MODEL,     help="model name")
    parser.add_argument(
        "--run-tag", default=DEFAULT_RUN_TAG, help="run tag under SP_DIR"
    )
    parser.add_argument(
        "--run-dir", default=None, help="explicit run dir (overrides --run-tag)"
    )
    parser.add_argument(
        "--tfrec-dir",
        default=os.getenv("BOOTSTRAP_TFREC_DIR", ""),
        help="path to .tfrecord.gz files  (env: BOOTSTRAP_TFREC_DIR)",
    )
    parser.add_argument(
        "--tfrec-dir-2",
        default="",
        help="second folder to blend 50/50 with --tfrec-dir",
    )
    parser.add_argument("--max-epoch", type=int,   default=1201)
    parser.add_argument("--lr",        type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    if not args.tfrec_dir:
        parser.error("--tfrec-dir is required (or set $BOOTSTRAP_TFREC_DIR)")

    run_dir = args.run_dir or os.path.join(SP_DIR, args.run_tag)
    os.makedirs(run_dir, exist_ok=True)

    name = args.model
    print(f"[supervisor] model={name}")
    print(f"[supervisor] run_dir={run_dir}")
    print(f"[supervisor] tfrec_dir={args.tfrec_dir}")
    if args.tfrec_dir_2:
        print(f"[supervisor] tfrec_dir_2={args.tfrec_dir_2}  (50/50 blend)")
    print(f"[supervisor] max_epoch={args.max_epoch}  lr={args.lr}")

    files1 = list_tfrecord_files(args.tfrec_dir)
    if not files1:
        parser.error(f"no .tfrecord.gz files found in {args.tfrec_dir}")

    if args.tfrec_dir_2:
        files2 = list_tfrecord_files(args.tfrec_dir_2)
        if not files2:
            parser.error(f"no .tfrecord.gz files found in {args.tfrec_dir_2}")
        n = min(len(files1), len(files2))
        rng = np.random.default_rng(42)
        sample1 = rng.choice(files1, size=n, replace=False).tolist()
        sample2 = rng.choice(files2, size=n, replace=False).tolist()
        all_files = sample1 + sample2
        rng.shuffle(all_files)
        print(f"[supervisor] blended: {n} from each dir -> {len(all_files)} total")
    else:
        all_files = files1

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
            "lr":          args.lr,
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
