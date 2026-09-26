"""Retrain a checkpoint (gate re-init, one-off) in <target_dir>_repair<epoch> until
it matches the run's last-5-row CSV averages at --repair_epoch, then install it
as <arch>_pt_ckpt<repair_epoch>.pt in --target_dir. Rolling repair_latest.pt every 50.

Usage:
    python scripts/train/repair_sidestream.py --target_dir DIR
        --target_model FILE.pt --repair_epoch 6000 --max_repair_epochs 2000
        [--margin 0.01] [--checkin_every 10]
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from bootstrap_model_async_sparse_pkl import (
    AsyncShardPool, BUFFER_CAP, DEFAULT_MAX_EPOCH, DEFAULT_SHARD_DIR, EPOCH_SIZE,
    GRAD_REPORT_EVERY, HOTSPOT_SCAN_EVERY, LR_HOLD_EPOCHS, LR_MAX, LR_MIN,
    REFILL_BAND, VAL_BUFFER_CAP, ckpt_path, do_eval, epoch_diagnostics, fit_epoch,
    list_shards, load_matching, make_adam, new_clip_state, print_gate_ema,
    print_key_list, progress_csv_path, pt_clipnorm_for_epoch, save_pt_ckpt,
    split_train_val_count,
)
from chessbot.model import PT_BUILDERS, VARIANTS
from chessbot.pretrain import lr_for_epoch, split_train_val as split_train_val_frac

# metric -> True if higher is better
METRICS = {"policy_ce": False, "value_ce": False, "value_corr": True, "top1_exact": True}
TARGET_ROWS = 5
PASS_STREAK = 3
ROLLING_EVERY = 50   # repair epochs between repair_latest.pt saves
GATE_PREFIX = "gate_"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target_dir", required=True)
    p.add_argument("--target_model", required=True,
                   help="checkpoint file name inside --target_dir")
    p.add_argument("--repair_epoch", type=int, required=True)
    p.add_argument("--max_repair_epochs", type=int, required=True)
    p.add_argument("--margin", type=float, default=0.01,
                   help="fractional slack on each target (looser)")
    p.add_argument("--checkin_every", type=int, default=10)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--max_epoch", type=int, default=DEFAULT_MAX_EPOCH,
                   help="the main run's max epoch, for the LR schedule")
    p.add_argument("--shard_dir", default=DEFAULT_SHARD_DIR)
    p.add_argument("--val_shards", type=int, default=0,
                   help="same meaning as the main trainer's --val-shards")
    return p.parse_args()


def read_targets(target_dir, name, repair_epoch):
    path = progress_csv_path(target_dir, name, unified=True)
    if not os.path.exists(path):
        path = progress_csv_path(target_dir, name, unified=False)
    df = pd.read_csv(path)
    hits = df.index[df["model_epoch"] == repair_epoch]
    if len(hits) == 0:
        raise RuntimeError(f"no model_epoch={repair_epoch} row in {path}")
    pos = df.index.get_loc(hits[-1])
    rows = df.iloc[max(0, pos - (TARGET_ROWS - 1)):pos + 1]
    print(f"[repair] targets from {os.path.basename(path)} epochs "
          f"{rows['model_epoch'].astype(int).tolist()}", flush=True)
    return {m: float(rows[m].mean()) for m in METRICS}


def thresholds(targets, margin):
    return {m: t * (1 - margin) if METRICS[m] else t * (1 + margin)
            for m, t in targets.items()}


def print_targets(targets, bars, margin):
    print(f"\n[repair] {'metric':<11} {'last5 avg':>10} {'pass at':>10}  "
          f"(margin {margin:.1%})", flush=True)
    for m, t in targets.items():
        rule = ">=" if METRICS[m] else "<="
        print(f"[repair] {m:<11} {t:>10.4f} {rule} {bars[m]:>7.4f}", flush=True)
    print(flush=True)


def check(row, bars):
    """Per metric (value, bar, gap, ok); gap > 0 means short of the bar."""
    out = {}
    for m, bar in bars.items():
        v = float(row[m])
        gap = bar - v if METRICS[m] else v - bar
        out[m] = (v, bar, gap, gap <= 0)
    return out


def print_check(i, result, streak):
    cells = "  ".join(
        f"{m} {v:.4f} ({'ok' if ok else f'{gap:.4f}'})"
        for m, (v, bar, gap, ok) in result.items())
    print(f"[repair check {i:4d}] {cells}  streak {streak}/{PASS_STREAK}", flush=True)


def build_model(name, sd, device):
    model = PT_BUILDERS[name](VARIANTS[name]).to(device)
    gate_keys = [k for k in sd if k.startswith(GATE_PREFIX)]
    sd = {k: v for k, v in sd.items() if not k.startswith(GATE_PREFIX)}
    reshaped, missing, unexpected = load_matching(model, sd)
    print_key_list("gate re-init (dropped from ckpt)", gate_keys)
    print_key_list("re-init, shape changed", reshaped)
    print_key_list("re-init, new in model",
                   [k for k in missing if not k.startswith(GATE_PREFIX)])
    print_key_list("dropped, not in model", unexpected)
    return model


def main():
    import torch
    from torch.amp import GradScaler
    from dotenv import load_dotenv
    load_dotenv()

    args = parse_args()
    target_dir = os.path.abspath(args.target_dir)
    src = os.path.join(target_dir, args.target_model)
    side_dir = f"{target_dir.rstrip(os.sep)}_repair{args.repair_epoch}"
    os.makedirs(side_dir, exist_ok=True)

    rolling_path = os.path.join(side_dir, "repair_latest.pt")
    resume = os.path.exists(rolling_path)
    ckpt = torch.load(rolling_path if resume else src, map_location="cpu")
    name = ckpt.get("arch")
    if name not in PT_BUILDERS:
        raise RuntimeError(f"arch {name!r} not in PT_BUILDERS")
    out_path = ckpt_path(target_dir, name, args.repair_epoch)

    print(f"[repair] source   {rolling_path if resume else src}", flush=True)
    print(f"[repair] arch     {name}", flush=True)
    print(f"[repair] side dir {side_dir}", flush=True)
    print(f"[repair] output   {out_path}", flush=True)

    targets = read_targets(target_dir, name, args.repair_epoch)
    bars = thresholds(targets, args.margin)
    print_targets(targets, bars, args.margin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    lr_rep = lr_for_epoch(args.repair_epoch, args.max_epoch, lr_min=LR_MIN,
                          lr_max=LR_MAX, hold_epochs=LR_HOLD_EPOCHS)
    scaler = GradScaler("cuda", init_scale=2 ** 15)
    side_csv = os.path.join(side_dir, "repair_progress.csv")
    eval_df, start = None, 0
    if resume:
        # repair_latest already carries the repaired gate: full load, no re-init
        model = PT_BUILDERS[name](VARIANTS[name]).to(device)
        model.load_state_dict(ckpt["model"])
        opt = make_adam(model, lr_rep)
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start = ckpt["epoch"] + 1
        if os.path.exists(side_csv):
            eval_df = pd.read_csv(side_csv)
        print(f"[repair] resuming at repair epoch {start} (gate kept, Adam kept, "
              f"pass streak restarts at 0)", flush=True)
    else:
        model = build_model(name, ckpt["model"], device)
        opt = make_adam(model, lr_rep)
    del ckpt
    print(f"[repair] lr at epoch {args.repair_epoch}: {lr_rep:.4e}  "
          f"(linear warmup over {args.warmup} epochs)", flush=True)

    all_files = list_shards(args.shard_dir)
    if args.val_shards > 0:
        train_files, val_files = split_train_val_count(all_files, args.val_shards)
    else:
        train_files, val_files = split_train_val_frac(all_files)
    train_pool = AsyncShardPool(
        train_files, high=BUFFER_CAP, low=BUFFER_CAP - REFILL_BAND,
        epoch_size=EPOCH_SIZE, drain=True, n_ready=2, name="train")
    val_pool = AsyncShardPool(
        val_files, high=VAL_BUFFER_CAP, low=VAL_BUFFER_CAP - REFILL_BAND,
        epoch_size=EPOCH_SIZE, drain=True, n_ready=1, name="val", subsample=False)
    train_pool.start()
    val_pool.start()

    side_plot = os.path.join(side_dir, "repair_plot.png")
    max_norm = pt_clipnorm_for_epoch(args.repair_epoch)
    clip_state = new_clip_state()
    result, streak = None, 0
    begin = time.time()

    for i in range(start, args.max_repair_epochs):
        lr = lr_rep * min(1.0, (i + 1) / max(args.warmup, 1))
        for pg in opt.param_groups:
            pg["lr"] = lr
        t0 = time.time()
        hot_scan = i % HOTSPOT_SCAN_EVERY == 0 and hasattr(model, "set_hotspot_scan")
        if hot_scan:
            model.set_hotspot_scan(True)
        p_loss, v_loss, t_loss, gns = fit_epoch(
            model, opt, scaler, train_pool.get(), max_norm, device, clip_state,
            report=(i % GRAD_REPORT_EVERY == 0))
        g_str = f"gate_loss: {gns['g_loss']:.2f}  " if gns["g_loss"] is not None else ""
        print(f"[repair {i:4d}] policy_loss: {p_loss:.2f}  value_loss: {v_loss:.2f}  "
              f"{g_str}lr: {lr:.2e}  {time.time() - t0:.1f}s", flush=True)
        epoch_diagnostics(model, scaler, gns, hot_scan)

        last = i == args.max_repair_epochs - 1
        if (i + 1) % args.checkin_every == 0 or last:
            eval_df, _, _ = do_eval(model, name, i, val_pool.get(), eval_df,
                                    side_csv, side_plot, device, train_loss=t_loss)
            print_gate_ema(model)
            result = check(eval_df.iloc[-1], bars)
            streak = streak + 1 if all(r[3] for r in result.values()) else 0
            print_check(i, result, streak)
            if streak >= PASS_STREAK:
                break
        if (i + 1) % ROLLING_EVERY == 0:
            save_pt_ckpt(model, opt, scaler, i, name, rolling_path)

    train_pool.stop()
    val_pool.stop()
    hit = streak >= PASS_STREAK
    print(f"\n[repair] {i + 1 - start} repair epochs in {(time.time() - begin) / 60:.1f} min",
          flush=True)
    if hit:
        print(f"[repair] TARGET HIT: {PASS_STREAK} passing check-ins in a row", flush=True)
    else:
        print(f"[repair] TARGET NOT HIT after {i + 1} epochs; last check-in vs target:",
              flush=True)
        for m, (v, bar, gap, ok) in result.items():
            status = "ok" if ok else f"short by {gap:.4f}"
            print(f"[repair]   {m:<11} got {v:.4f}  target {targets[m]:.4f}  "
                  f"pass at {bar:.4f}  {status}", flush=True)
    save_pt_ckpt(model, opt, scaler, args.repair_epoch, name, out_path)
    print(f"[repair] installed -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
