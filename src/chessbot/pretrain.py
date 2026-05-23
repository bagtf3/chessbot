"""pretrain.py — shared constants, LW schedule, TFRecord pipeline,
and plot utilities used by both TF and PT bootstrap trainers.

TF is imported lazily (inside functions) so this module is safe to
import in the TF supervisor process without touching the GPU.
"""
from __future__ import annotations

import math
import os
import queue
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE         = 256
EPOCH_SIZE         = 10_240
SHUFFLE_BUFFER     = 256_000
VAL_SHUFFLE_BUFFER = 16_000
STEPS_PER_EPOCH    = EPOCH_SIZE // BATCH_SIZE
VAL_FRACTION       = 0.05
VAL_SPLIT_SEED     = 42
UNIFORM_BLEND      = 0.05
POLICY_MAX_CLIP    = 0.6
PLOT_EVERY         = 10
DEFAULT_LR         = 1e-4
DEFAULT_MAX_EPOCH  = 2000


# ---------------------------------------------------------------------------
# Loss-weight schedule — fixed 4:1 value:policy after warmup
# ---------------------------------------------------------------------------

POLICY_LW = 1.0
VALUE_LW  = 4.0

LW_SCHEDULE: dict[int, dict[str, float]] = {
    0:  {"policy_logits": 0.10, "value_out": 0.10},
    3:  {"policy_logits": 0.75, "value_out": 1.50},
    10: {"policy_logits": POLICY_LW, "value_out": VALUE_LW},
}


def lw_for_epoch(epoch: int) -> dict[str, float]:
    return LW_SCHEDULE[max(k for k in LW_SCHEDULE if k <= epoch)]


# ---------------------------------------------------------------------------
# Learning-rate schedule
#   0-9:   warmup LR_MIN -> LR_MAX
#   10-99: flat at LR_MAX
#   100+:  cosine steps every LR_STEP_SIZE epochs down to LR_MIN
#   last LR_DECAY_EPOCHS: flat at LR_MIN (finetune)
# ---------------------------------------------------------------------------

LR_MIN          = 1e-4
LR_MAX          = 5e-3
LR_WARMUP_EPOCHS = 3
LR_HOLD_EPOCHS   = 100
LR_DECAY_EPOCHS  = 300
LR_STEP_SIZE     = 100


def lr_for_epoch(ep: int, max_epoch: int = DEFAULT_MAX_EPOCH) -> float:
    decay_end = max_epoch - LR_DECAY_EPOCHS
    if ep < LR_WARMUP_EPOCHS:
        return LR_MIN + (LR_MAX - LR_MIN) * (ep / LR_WARMUP_EPOCHS)
    if ep < LR_HOLD_EPOCHS:
        return LR_MAX
    if ep >= decay_end:
        return LR_MIN
    step        = (ep - LR_HOLD_EPOCHS) // LR_STEP_SIZE
    total_steps = (decay_end - LR_HOLD_EPOCHS) // LR_STEP_SIZE
    t = (step + 1) / total_steps
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def list_tfrecord_files(tfrec_dir: str) -> list[str]:
    return sorted(
        os.path.join(tfrec_dir, f)
        for f in os.listdir(tfrec_dir)
        if f.endswith(".tfrecord.gz")
    )


def split_train_val(
    all_files: list[str],
) -> tuple[list[str], list[str]]:
    rng   = np.random.default_rng(VAL_SPLIT_SEED)
    idx   = rng.permutation(len(all_files))
    n_val = max(1, int(len(all_files) * VAL_FRACTION))
    val_files   = [all_files[i] for i in idx[:n_val]]
    train_files = [all_files[i] for i in idx[n_val:]]
    return train_files, val_files


# ---------------------------------------------------------------------------
# TFRecord data pipeline
# ---------------------------------------------------------------------------

def make_dataset(file_list: list[str], shuffle_buffer: int, batch_size: int = BATCH_SIZE):
    """Build a repeating, shuffled TF dataset from .tfrecord.gz files.

    Imports TF lazily so the supervisor process stays GPU-free.
    """
    import tensorflow as tf

    feature_spec = {
        "enc_in":        tf.io.FixedLenFeature([], tf.string),
        "mask":          tf.io.FixedLenFeature([], tf.string),
        "policy_logits": tf.io.FixedLenFeature([], tf.string),
        "value_out":     tf.io.FixedLenFeature([3], tf.float32),
        "weight":        tf.io.FixedLenFeature([], tf.float32),
    }

    def parse_record(raw):
        feat   = tf.io.parse_single_example(raw, feature_spec)
        enc_in = tf.cast(
            tf.io.parse_tensor(feat["enc_in"], out_type=tf.int16), tf.int32
        )
        mask    = tf.io.parse_tensor(feat["mask"], out_type=tf.int32)
        policy  = tf.io.parse_tensor(feat["policy_logits"], out_type=tf.float32)
        mask_f  = tf.cast(mask, tf.float32)
        n_legal = tf.reduce_sum(mask_f)
        policy  = (1.0 - UNIFORM_BLEND) * policy + UNIFORM_BLEND * (mask_f / n_legal)
        policy  = tf.minimum(policy, POLICY_MAX_CLIP)
        policy  = policy / tf.reduce_sum(policy)
        value   = feat["value_out"]
        weight  = feat["weight"]
        return (
            {"enc_in": enc_in, "mask": mask},
            {"policy_logits": policy, "value_out": value},
            {"policy_logits": weight, "value_out": weight},
        )

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
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class EpochBufferThread(threading.Thread):
    """Prefetches epoch bundles from a TF dataset in a background thread."""

    def __init__(self, dataset, steps_per_epoch: int, max_ready: int = 2):
        super().__init__(daemon=True)
        self.dataset         = dataset
        self.steps_per_epoch = steps_per_epoch
        self.q               = queue.Queue(maxsize=max_ready)
        self.stop_event      = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def get(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def run(self) -> None:
        import tensorflow as tf
        try:
            it = iter(self.dataset)
            while not self.stop_event.is_set():
                parts: dict[str, list] = {
                    "enc_in": [], "mask": [], "policy_logits": [],
                    "value_out": [], "weight": [],
                }
                for _ in range(self.steps_per_epoch):
                    if self.stop_event.is_set():
                        return
                    inp, out, wt = next(it)
                    parts["enc_in"].append(inp["enc_in"])
                    parts["mask"].append(inp["mask"])
                    parts["policy_logits"].append(out["policy_logits"])
                    parts["value_out"].append(out["value_out"])
                    parts["weight"].append(wt["policy_logits"])
                bundle = {k: tf.concat(v, axis=0) for k, v in parts.items()}
                self.q.put(bundle)
        except Exception as exc:
            try:
                self.q.put(exc, timeout=1.0)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def moving_average(arr, window: int) -> np.ndarray:
    return (
        pd.Series(arr)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def save_plot(
    eval_df: pd.DataFrame | None,
    label: str,
    ep: int,
    plot_file: str,
    ystack: np.ndarray | None = None,
    val_preds: np.ndarray | None = None,
) -> None:
    if eval_df is None or len(eval_df) < 2:
        return
    n    = len(eval_df)
    hide = max(int(0.1 * n), 5)
    win  = min(max(3, int(n * 0.2)), 15)
    if win % 2 == 0:
        win += 1
    xs_show = np.arange(n)[hide:]
    if len(xs_show) < 2:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{label}  —  epoch {ep}")

    for ax, key, title in [
        (axes[0, 0], "policy_ce",  "policy CE (nats)"),
        (axes[0, 1], "ce_gain",    "CE gain vs uniform"),
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

    # value MSE (left axis) + CE (right axis) on shared subplot
    ax_val = axes[0, 2]
    ax_ce  = ax_val.twinx()
    has_mse = "value_mse" in eval_df.columns
    has_ce  = "value_ce"  in eval_df.columns
    if has_mse:
        raw = eval_df["value_mse"].to_numpy()[hide:]
        ma  = moving_average(eval_df["value_mse"].to_numpy(), window=win)[hide:]
        ax_val.plot(xs_show, raw, alpha=0.6, lw=1, color="tab:blue",   label="MSE")
        ax_val.plot(xs_show, ma,  lw=2,            color="tab:blue",   label=f"MSE MA{win}")
        ax_val.set_ylabel("MSE (Q)", color="tab:blue")
        ax_val.tick_params(axis="y", labelcolor="tab:blue")
    if has_ce:
        raw = eval_df["value_ce"].to_numpy()[hide:]
        ma  = moving_average(eval_df["value_ce"].to_numpy(), window=win)[hide:]
        ax_ce.plot(xs_show, raw, alpha=0.6, lw=1, color="tab:orange", label="CE")
        ax_ce.plot(xs_show, ma,  lw=2,             color="tab:orange", label=f"CE MA{win}")
        ax_ce.set_ylabel("CE (nats)", color="tab:orange")
        ax_ce.tick_params(axis="y", labelcolor="tab:orange")
    ax_val.set_title("value MSE + CE")
    lines  = ax_val.get_legend_handles_labels()
    lines2 = ax_ce.get_legend_handles_labels()
    ax_val.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=7)

    ax_sc = axes[1, 2]
    if ystack is not None and val_preds is not None:
        corr = float(np.corrcoef(val_preds, ystack)[0, 1])
        ax_sc.scatter(ystack, val_preds, s=2, alpha=0.3)
        lims = [
            min(ystack.min(), val_preds.min()),
            max(ystack.max(), val_preds.max()),
        ]
        ax_sc.plot(lims, lims, "r--", lw=1)
        ax_sc.set_xlabel("target Q")
        ax_sc.set_ylabel("pred Q")
        ax_sc.set_title(f"value Q scatter  (r={corr:.3f})")

    fig.tight_layout()
    fig.savefig(plot_file, dpi=100)
    plt.close(fig)
    plt.close("all")
    print(f"[plot] saved -> {plot_file}")
