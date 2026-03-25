#!/usr/bin/env python3
"""
bootstrap_model_async_tfrec.py — OOM-robust bootstrap trainer

The supervisor spawns a fresh subprocess for each 100-epoch block.
On crash / OOM the supervisor rolls back to the last checkpoint
(saved every 20 epochs) and restarts with a fresh random data stream.

Usage:
    python scripts/bootstrap_model_async_tfrec.py [options]

Options:
    --model      Model name          (default: 16m-frankenformer-interweaved)
    --run-tag    Sub-dir under SP_DIR (default: val_test_multi)
    --run-dir    Explicit run dir (overrides --run-tag)
    --tfrec-dir  Path to .tfrecord.gz files  (env: BOOTSTRAP_TFREC_DIR)
    --max-epoch  Total epochs to train       (default: 900)
    --lr         Learning rate               (default: 2e-4)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time

import numpy as np
import pandas as pd

BATCH_SIZE        = 256
EPOCH_SIZE        = 10_240
SHUFFLE_BUFFER     = 64_000
VAL_SHUFFLE_BUFFER = 16_000
STEPS_PER_EPOCH   = EPOCH_SIZE // BATCH_SIZE
VAL_FRACTION      = 0.05
VAL_SPLIT_SEED    = 42

EPOCHS_PER_WORKER = 150
CHECKPOINT_EVERY  = 20    # multiple of PLOT_EVERY so checkpoints always land on eval epochs
PLOT_EVERY        = 10

DEFAULT_LR        = 2e-4
DEFAULT_MAX_EPOCH = 920
DEFAULT_MODEL     = "16m-conformer-interweaved"
DEFAULT_RUN_TAG   = "val_test_multi"

LW_SCHEDULE: dict[int, dict[str, float]] = {
    0:   {"policy_logits": 0.10, "value_out": 0.10},
    1:   {"policy_logits": 0.75, "value_out": 1.50},
    10:  {"policy_logits": 1.25, "value_out": 2.50},
    160: {"policy_logits": 1.00, "value_out": 2.25},
    250: {"policy_logits": 0.75, "value_out": 1.50},
    480: {"policy_logits": 0.65, "value_out": 1.25},
    640: {"policy_logits": 0.50, "value_out": 1.00},
    700: {"policy_logits": 0.25, "value_out": 0.50},
}


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
    """Scan checkpoint files, trim the eval CSV to match, return next epoch to train."""
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
    print(f"[recovery] last checkpoint epoch={last_ckpt}  →  resuming from epoch {resume}")
    return resume


class EpochBufferThread(threading.Thread):
    """Prefetches epoch bundles from a TF dataset in a background thread."""

    def __init__(self, dataset, steps_per_epoch: int, max_ready: int = 2):
        super().__init__(daemon=True)
        self.dataset         = dataset
        self.steps_per_epoch = steps_per_epoch
        self._q              = queue.Queue(maxsize=max_ready)
        self._stop           = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def get(self):
        item = self._q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def run(self) -> None:
        import tensorflow as tf
        try:
            it = iter(self.dataset)
            while not self._stop.is_set():
                parts: dict[str, list] = {
                    "enc_in": [], "mask": [], "policy_logits": [],
                    "value_out": [], "weight": [],
                }
                for _ in range(self.steps_per_epoch):
                    if self._stop.is_set():
                        return
                    inp, out, wt = next(it)
                    parts["enc_in"].append(inp["enc_in"])
                    parts["mask"].append(inp["mask"])
                    parts["policy_logits"].append(out["policy_logits"])
                    parts["value_out"].append(out["value_out"])
                    parts["weight"].append(wt["policy_logits"])
                bundle = {k: tf.concat(v, axis=0) for k, v in parts.items()}
                self._q.put(bundle)
        except Exception as exc:
            try:
                self._q.put(exc, timeout=1.0)
            except queue.Full:
                pass


def worker_main(wargs: dict) -> None:
    """Train for one block of epochs. All TF imports are local so the
    supervisor process never touches the GPU."""
    import gc
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import mixed_precision
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mixed_precision.set_global_policy("mixed_float16")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    from chessbot import MODEL_DIR
    from chessbot.model import set_loss_weights
    from chessbot.utils import batch_policy_metrics, format_time, print_validation

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

    # load model
    if start_epoch == 0:
        load_from = os.path.join(model_dir, f"{name}_model.h5")
    else:
        last_ckpt = find_last_checkpoint(run_dir, name)
        if last_ckpt < 0:
            raise RuntimeError(f"start_epoch={start_epoch} but no checkpoint found in {run_dir}")
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
        "value_out":     tf.keras.losses.MeanSquaredError(),
    }
    set_loss_weights(model, {"policy_logits": 1.0, "value_out": 1.0}, jit=False)
    current_lw: dict | None = None

    # tf.train.Checkpoint for optimizer state persistence across block restarts
    train_ckpts_dir = os.path.join(run_dir, "train_ckpts")
    os.makedirs(train_ckpts_dir, exist_ok=True)
    epoch_var = tf.Variable(start_epoch, trainable=False, dtype=tf.int64)
    tf_ckpt   = tf.train.Checkpoint(model=model, optimizer=opt, epoch=epoch_var)
    manager   = tf.train.CheckpointManager(tf_ckpt, train_ckpts_dir, max_to_keep=2)
    if manager.latest_checkpoint:
        tf_ckpt.restore(manager.latest_checkpoint)
        print(f"[train_ckpt] restored optimizer state from {manager.latest_checkpoint}")
    else:
        print("[train_ckpt] no prior optimizer checkpoint — starting fresh")

    # load existing progress CSV
    eval_df: pd.DataFrame | None = None
    if os.path.exists(progress_file):
        existing = pd.read_csv(progress_file)
        eval_df = existing if not existing.empty else None

    # build dataset
    feature_spec = {
        "enc_in":        tf.io.FixedLenFeature([], tf.string),
        "mask":          tf.io.FixedLenFeature([], tf.string),
        "policy_logits": tf.io.FixedLenFeature([], tf.string),
        "value_out":     tf.io.FixedLenFeature([], tf.float32),
        "weight":        tf.io.FixedLenFeature([], tf.float32),
    }

    def parse_record(raw):
        feat   = tf.io.parse_single_example(raw, feature_spec)
        enc_in = tf.cast(
            tf.io.parse_tensor(feat["enc_in"], out_type=tf.int16), tf.int32
        )
        mask   = tf.io.parse_tensor(feat["mask"], out_type=tf.int32)
        policy = tf.io.parse_tensor(feat["policy_logits"], out_type=tf.float32)
        value  = tf.reshape(feat["value_out"], [1])
        weight = feat["weight"]
        return (
            {"enc_in": enc_in, "mask": mask},
            {"policy_logits": policy, "value_out": value},
            {"policy_logits": weight, "value_out": weight},
        )

    if not train_files:
        raise RuntimeError("train_files is empty")
    if not val_files:
        raise RuntimeError("val_files is empty")

    print(f"[tfrec] train={len(train_files)}  val={len(val_files)}  shuffle_buffer={SHUFFLE_BUFFER}")

    def make_dataset(file_list: list[str], shuffle_buffer: int) -> "tf.data.Dataset":
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        ds = ds.shuffle(len(file_list), reshuffle_each_iteration=True)
        ds = ds.repeat()
        ds = ds.interleave(
            lambda p: tf.data.TFRecordDataset(p, compression_type="GZIP"),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        ds = ds.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.batch(BATCH_SIZE, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    prefetcher     = EpochBufferThread(make_dataset(train_files, SHUFFLE_BUFFER),     STEPS_PER_EPOCH, max_ready=2)
    val_prefetcher = EpochBufferThread(make_dataset(val_files,   VAL_SHUFFLE_BUFFER), STEPS_PER_EPOCH, max_ready=1)
    prefetcher.start()
    val_prefetcher.start()

    def moving_average(arr, window: int) -> np.ndarray:
        s = pd.Series(arr)
        return s.rolling(window, center=True, min_periods=1).mean().to_numpy()

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

    def do_eval(bundle, ep: int) -> tuple[pd.DataFrame | None, dict | None]:
        nonlocal eval_df
        x = {"enc_in": bundle["enc_in"]}
        if "mask" in model.input_names:
            x["mask"] = bundle["mask"]

        preds      = model.predict(x, verbose=0, batch_size=64)
        pol_preds  = preds[0]
        val_preds  = preds[1].ravel()
        pstack     = bundle["policy_logits"].numpy()
        ystack     = bundle["value_out"].numpy().ravel()
        mstack     = bundle["mask"].numpy()

        pol_stats  = batch_policy_metrics(pol_preds, pstack, mstack)
        value_mse  = float(np.mean((val_preds - ystack) ** 2))
        value_corr = float(np.corrcoef(val_preds, ystack)[0, 1])

        row = {"training_epoch": ep, "value_mse": value_mse, "value_corr": value_corr}
        row.update(pol_stats)

        new_row = pd.DataFrame([row])
        eval_df = (
            new_row if eval_df is None
            else pd.concat([eval_df, new_row], ignore_index=True)
        )
        eval_df.round(4).to_csv(progress_file, index=False)

        print_metrics = {"value_mse": value_mse, "value_corr": value_corr, **pol_stats}
        print_validation(ep, print_metrics)

        return eval_df, {"ystack": ystack, "val_preds": val_preds}

    def save_plot(ep: int, scatter: dict | None) -> None:
        if eval_df is None or len(eval_df) < 2:
            return
        n     = len(eval_df)
        hide  = max(int(0.1 * n), 5)
        win   = min(max(3, int(n * 0.2)), 15)
        if win % 2 == 0:
            win += 1
        xs      = np.arange(n)
        xs_show = xs[hide:]
        if len(xs_show) < 2:
            return

        run_tag = os.path.basename(run_dir)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"{name}  [{run_tag}]  —  epoch {ep}")

        for ax, key, title in [
            (axes[0, 0], "policy_ce",  "policy CE (nats)"),
            (axes[0, 1], "ce_gain",    "CE gain vs uniform"),
            (axes[0, 2], "value_mse",  "value MSE"),
            (axes[1, 0], "value_corr", "value corr"),
            (axes[1, 1], "top1_exact", "top1 exact"),
        ]:
            if key not in eval_df.columns:
                continue
            raw = eval_df[key].to_numpy()[hide:]
            ma  = moving_average(eval_df[key].to_numpy(), window=win)[hide:]
            ax.plot(xs_show, raw, alpha=0.6, lw=1, label=key)
            ax.plot(xs_show, ma,  lw=2,      label=f"MA{win}")
            ax.set_title(title)
            ax.legend()

        ax_sc = axes[1, 2]
        if scatter is not None:
            ys, vp = scatter["ystack"], scatter["val_preds"]
            corr   = float(np.corrcoef(vp, ys)[0, 1])
            ax_sc.scatter(ys, vp, s=2, alpha=0.3)
            lims = [min(ys.min(), vp.min()), max(ys.max(), vp.max())]
            ax_sc.plot(lims, lims, "r--", lw=1)
            ax_sc.set_xlabel("target")
            ax_sc.set_ylabel("pred")
            ax_sc.set_title(f"value scatter  (r={corr:.3f})")

        fig.tight_layout()
        fig.savefig(plot_file, dpi=100)
        plt.close(fig)
        plt.close("all")
        print(f"[plot] saved → {plot_file}")

    # training loop
    begin          = time.time()
    epoch_times: list[float] = []
    t_fetch = t_fit = t_eval = 0.0
    total_samples  = 0
    window_samples = 0
    last_scatter: dict | None = None

    for ep in range(start_epoch, end_epoch):
        target_lw = LW_SCHEDULE[max(k for k in LW_SCHEDULE if k <= ep)]
        if target_lw is not current_lw:
            current_lw = target_lw
            lw_str = "  ".join(f"{k}={v}" for k, v in target_lw.items())
            print(f"[lw update] epoch {ep}: {lw_str}")

        epoch_start = time.time()

        print("-" * 89)

        if ep % PLOT_EVERY == 0:
            t0 = time.time()
            val_bundle = val_prefetcher.get()
            eval_df, last_scatter = do_eval(val_bundle, ep)
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
            print(f"[ckpt] epoch {ep} → {cp}")
            delete_old_checkpoints(run_dir, name, keep_epoch=ep)
            epoch_var.assign(ep)
            manager.save()
            print(f"[train_ckpt] optimizer state → {manager.latest_checkpoint}")
            save_plot(ep, last_scatter)

        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        print(
            f"[time check] last: {format_time(elapsed)}  "
            f"avg: {format_time(float(np.mean(epoch_times)))}  "
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
    parser.add_argument("--run-tag",   default=DEFAULT_RUN_TAG,   help="run tag under SP_DIR")
    parser.add_argument("--run-dir",   default=None,              help="explicit run dir (overrides --run-tag)")
    parser.add_argument(
        "--tfrec-dir",
        default=os.getenv("BOOTSTRAP_TFREC_DIR", ""),
        help="path to .tfrecord.gz files  (env: BOOTSTRAP_TFREC_DIR)",
    )
    parser.add_argument("--max-epoch", type=int,   default=DEFAULT_MAX_EPOCH)
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
    print(f"[supervisor] max_epoch={args.max_epoch}  lr={args.lr}")

    all_files = sorted(
        os.path.join(args.tfrec_dir, f)
        for f in os.listdir(args.tfrec_dir)
        if f.endswith(".tfrecord.gz")
    )
    if not all_files:
        parser.error(f"no .tfrecord.gz files found in {args.tfrec_dir}")

    rng     = np.random.default_rng(VAL_SPLIT_SEED)
    idx     = rng.permutation(len(all_files))
    n_val   = max(1, int(len(all_files) * VAL_FRACTION))
    val_files   = [all_files[i] for i in idx[:n_val]]
    train_files = [all_files[i] for i in idx[n_val:]]
    print(f"[supervisor] train_files={len(train_files)}  val_files={len(val_files)}")

    resume = get_resume_epoch(run_dir, name)

    while resume < args.max_epoch:
        block_start = (resume // EPOCHS_PER_WORKER) * EPOCHS_PER_WORKER
        end_epoch   = min(block_start + EPOCHS_PER_WORKER, args.max_epoch)

        print(f"\n[supervisor] block {block_start // EPOCHS_PER_WORKER}  epochs {resume}..{end_epoch - 1}")

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
            print(f"[supervisor] block complete  →  resume={resume}")
        else:
            print(f"[supervisor] worker crashed (exit={exit_code}), recovering ...")
            time.sleep(5)
            resume = get_resume_epoch(run_dir, name)
            print(f"[supervisor] will retry from epoch {resume}")

    print(f"\n[supervisor] training complete ({args.max_epoch} epochs)")


if __name__ == "__main__":
    main()
