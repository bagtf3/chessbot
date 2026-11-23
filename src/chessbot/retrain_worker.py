"""Retrain worker process. Spawned fresh (ctx.Process) for each retrain
cycle and torn down after -- a deliberate VRAM-reclaim policy, not an
artifact of the old subprocess-script approach it replaces. See
replay_buffer_redesign.md for the full design and message protocol.

Message protocol (msg_q, main -> worker):
  {"cmd": "preload"}
  {"cmd": "start"}
      Both sent back to back the moment the worker is spawned, which only
      happens once primary_buffer is full at PRIMARY_TRIGGER_SHARDS. The
      worker draws the replay and historic shards on preload and the primary
      and holdout shards on start. Once training has saved the model, the
      worker moves the trained primary shards to replay and deletes the drawn
      replay shards.
  {"cmd": "go"}
      Every selfplay worker has dropped its GPU sessions (or main gave up
      waiting). Training allocates only after this, or after GO_TIMEOUT_S.

Message protocol (result_q, worker -> main):
  {"cmd": "retrain_ready"}
      validation predicted, dataset built, about to train. Main pauses
      selfplay workers on receipt of this.
  {"cmd": "retrain_done", "ok": bool, "error": str or None,
   "train_stats": {"train_loss": float, "gn_mean": float}}
      training, save, and the validation report (console, csv rows with
      train_stats folded in, validation_latest.png) finished, or failed.
      Main rebuilds TRT and unpauses selfplay workers on receipt of this.
"""
import os
import queue
import random
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from chessbot.config import Config
from chessbot.opening_counts import apply_historic_dup_weights
from chessbot.replay_buffer import (
    discard_files, list_shard_files, read_shard, rotate_to_replay,
    sample_files, sample_holdout_primary, sample_records,
    sample_retrain_primary, stack_policies, write_pkl_gz_shard, POLICY_DIM,
    RETRAIN_HISTORIC_SHARDS, RETRAIN_HOLDOUT_SHARDS, RETRAIN_PRIMARY_SHARDS,
    RETRAIN_REPLAY_SHARDS, SHARD_SIZE, VAL_HISTORIC_RECORDS, VAL_LC0_RECORDS,
    VAL_PRIMARY_RECORDS, VAL_SF_RECORDS, SF_SOURCE,
)
from chessbot.eval_metrics import report_validation
from chessbot.utils import policy_metric_rows

# above the runner's RETRAIN_RELEASE_TIMEOUT_S so main's go normally wins
GO_TIMEOUT_S = 120.0
# validation rows predicted and scored per pass; the only dense
# (rows, POLICY_DIM) arrays ever held are this tall
VAL_CHUNK = 2048
# fit tracking: TRACK_RECORDS trained xc0 records and every record of the
# holdout shards (read only, never trained), each predicted with the starting
# and the trained weights. The worst row is the top WORST_FRAC of the trained
# slice by its starting loss.
TRACK_RECORDS = 2048
WORST_FRAC = 0.1
FIT_CSV = "fit_tracking.csv"


def load_records(paths):
    records = []
    for path in paths:
        records.extend(read_shard(path))
    return records


def shuffle_rewrite_primary(paths, records):
    """Shuffle already-loaded primary records across file boundaries and
    rewrite them, split evenly, back into the same filenames. Takes the records the caller already read rather than re-reading from
    disk. Returns the shuffled record list -- already the exact primary
    portion of the retrain set."""
    all_records = list(records)
    random.shuffle(all_records)
    bounds = np.linspace(0, len(all_records), len(paths) + 1).astype(int)
    for p, a, b in zip(paths, bounds[:-1], bounds[1:]):
        write_pkl_gz_shard(all_records[a:b], p)
    return all_records


def predict_fp16(model, X_list, batch_size):
    """(wdl (N, 3), policy logits (N, 1858)) float32 arrays."""
    import torch
    import torch.nn.functional as F
    device = torch.device("cuda")
    model = model.to(device).half().eval()
    n = len(X_list)
    wdl = np.zeros((n, 3), dtype=np.float32)
    logits = np.zeros((n, POLICY_DIM), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = np.stack(X_list[i:i + batch_size], axis=0)
            xb = torch.from_numpy(batch).long().to(device)
            pol_logits, val_out = model(xb)
            j = i + len(batch)
            logits[i:j] = pol_logits.float().cpu().numpy()
            wdl[i:j] = F.softmax(val_out.float(), dim=-1).cpu().numpy()
    return wdl, logits


def validate_slice(model, samples, batch_size):
    """Value predictions and per-row policy metrics for one validation slice,
    VAL_CHUNK rows at a time. Input to eval_metrics.report_validation."""
    wdl, policy, top_target = [], defaultdict(list), []
    for i in range(0, len(samples), VAL_CHUNK):
        part = samples[i:i + VAL_CHUNK]
        w, logits = predict_fp16(model, [r[0] for r in part], batch_size)
        labels = stack_policies(part)
        for k, v in policy_metric_rows(logits, labels, labels > 0).items():
            policy[k].append(v)
        top_target.append(labels.max(axis=1))
        wdl.append(w)
    return {
        "wdl_pred": np.concatenate(wdl) if wdl else np.zeros((0, 3), np.float32),
        "wdl_true": np.array([r[3] for r in samples],
                             dtype=np.float64).reshape(-1, 3),
        "source": [r[6] for r in samples],
        "policy": {k: np.concatenate(v) for k, v in policy.items()},
        "top_prob_target": np.concatenate(top_target) if top_target else np.zeros(0),
    }


def masked_logp(logits, mask):
    """Log-softmax over the masked (legal) moves; -inf elsewhere."""
    with np.errstate(invalid="ignore"):
        z = np.where(mask, logits, -np.inf)
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def kl_rows(p, logp, logq):
    """Per-row KL(p || q), summed only where p > 0."""
    with np.errstate(invalid="ignore"):
        return np.where(p > 0, p * (logp - logq), 0.0).sum(axis=1)


def safe_log(p):
    return np.log(np.where(p > 0, p, 1.0))


def fit_rows(samples, before, after):
    """Per-row fit metrics for one slice. KLp / KLv are KL(target || net)
    for policy / value with the starting (0) and trained (1) weights; move_p
    / move_v are KL(trained || starting) on the same positions, how far the
    net moved whatever the targets."""
    labels = stack_policies(samples)
    mask = labels > 0
    y = np.array([r[3] for r in samples], dtype=np.float64).reshape(-1, 3)
    (w0, l0), (w1, l1) = before, after
    lq0, lq1 = masked_logp(l0, mask), masked_logp(l1, mask)
    lt, ly = safe_log(labels), safe_log(y)
    lw0, lw1 = safe_log(w0.astype(np.float64)), safe_log(w1.astype(np.float64))
    return {
        "klp0": kl_rows(labels, lt, lq0), "klp1": kl_rows(labels, lt, lq1),
        "klv0": kl_rows(y, ly, lw0), "klv1": kl_rows(y, ly, lw1),
        "move_p": kl_rows(np.exp(lq1), lq1, lq0),
        "move_v": kl_rows(w1.astype(np.float64), lw1, lw0),
    }


def fit_summary(rows, worst=False):
    """Slice means. With worst, policy columns cover the top WORST_FRAC by
    starting KLp and value columns the top WORST_FRAC by starting KLv."""
    def top(key):
        n = max(1, int(round(WORST_FRAC * len(rows[key]))))
        return np.argsort(rows[key])[-n:]
    ip = top("klp0") if worst else slice(None)
    iv = top("klv0") if worst else slice(None)
    out = {k: float(rows[k][ip].mean()) for k in ("klp0", "klp1", "move_p")}
    out.update({k: float(rows[k][iv].mean()) for k in ("klv0", "klv1", "move_v")})
    out["n"] = len(rows["klp0"][ip])
    return out


def report_fit_tracking(cfg, epoch, slices, before, after):
    """Print the fit table and append one row to FIT_CSV in the run dir."""
    import csv
    if not slices["trained"] or not slices["held_out"]:
        print("[retrain] fit tracking skipped: no xc0 records", flush=True)
        return
    trained = fit_rows(slices["trained"], before["trained"], after["trained"])
    summary = {
        "trained": fit_summary(trained),
        "worst": fit_summary(trained, worst=True),
        "held_out": fit_summary(fit_rows(slices["held_out"], before["held_out"],
                                         after["held_out"])),
    }
    print(f"[retrain] {'fit':<10}{'n':>6}{'KLp0':>8}{'KLp1':>8}{'dKLp':>8}"
          f"{'KLv0':>8}{'KLv1':>8}{'dKLv':>8}{'moveP':>8}{'moveV':>8}")
    for name, s in summary.items():
        print(f"[retrain] {name:<10}{s['n']:>6}{s['klp0']:>8.3f}{s['klp1']:>8.3f}"
              f"{s['klp1'] - s['klp0']:>+8.3f}{s['klv0']:>8.3f}{s['klv1']:>8.3f}"
              f"{s['klv1'] - s['klv0']:>+8.3f}{s['move_p']:>8.4f}{s['move_v']:>8.4f}")

    row = {"epoch": epoch, "ts": int(time.time()), "lr": cfg.learning_rate}
    for name, s in summary.items():
        row[f"{name}_n"] = s["n"]
        for k in ("klp0", "klp1", "klv0", "klv1", "move_p", "move_v"):
            row[f"{name}_{k}"] = round(s[k], 6)
    path = os.path.join(cfg.run_dir, FIT_CSV)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def records_to_arrays(records):
    # x is int16 on every path and torch recasts on the way to the GPU
    # (.long() for token encodings, .float() for lc0 planes), so an int32
    # intermediate here just doubles the array for nothing
    X = np.stack([r[0] for r in records])
    # policies stay in record form (sparse or dense, mixed); retrain_pt
    # densifies one VRAM chunk at a time so the full (N, 1858) float32 stack
    # never exists
    P = [r[2] for r in records]
    Y = np.array([r[3] for r in records], dtype=np.float32)
    vwht = np.array([r[4] for r in records], dtype=np.float32)
    pwht = np.array([r[5] for r in records], dtype=np.float32)
    return X, P, Y, vwht, pwht


def wait_for_go(msg_q):
    """Block until main reports the fleet off the GPU. The timeout is a
    backstop only; main sends go after its own shorter one."""
    try:
        msg_q.get(timeout=GO_TIMEOUT_S)
    except queue.Empty:
        print("[retrain] no go from main, training anyway", flush=True)


def run_retrain_worker(run_dir, msg_q, result_q, epoch):
    try:
        retrain_worker_body(run_dir, msg_q, result_q, epoch)
    except Exception as e:
        result_q.put({"cmd": "retrain_done", "ok": False, "error": str(e)})
    # os._exit skips the queue's feeder thread; flush it or retrain_done is lost
    result_q.close()
    result_q.join_thread()
    sys.stdout.flush()
    sys.stderr.flush()
    # TF (used for historic tfrec.gz reads) hangs on normal interpreter exit
    # -- hard-exit instead, matching shuffle_xc0.py / bootstrap_model_async.
    os._exit(0)


def retrain_worker_body(run_dir, msg_q, result_q, epoch):
    import torch
    from chessbot.train_pytorch import load_pt_model, retrain_pt

    cfg = Config.from_yaml(os.path.join(run_dir, "config.yaml"))

    msg_q.get()
    replay_paths = sample_files(cfg.replay_buffer_dir, RETRAIN_REPLAY_SHARDS)
    replay_records = load_records(replay_paths)
    historic_records = apply_historic_dup_weights(load_records(
        sample_files(cfg.historic_dir, RETRAIN_HISTORIC_SHARDS)))

    msg_q.get()
    primary = list_shard_files(cfg.primary_buffer_dir)
    primary_paths = sample_retrain_primary(primary, RETRAIN_PRIMARY_SHARDS)

    model, arch = load_pt_model(cfg.model_path)
    primary_records_raw = load_records(primary_paths)
    held_out = load_records(sample_holdout_primary(
        primary, primary_paths, RETRAIN_HOLDOUT_SHARDS))

    # primary validation is pure xc0 only; replay, lc0, and blend records
    # still train via all_records below, just aren't validated on
    val_pool = [r for r in primary_records_raw if r[6] == 'xc0']
    val_primary = sample_records(val_pool, VAL_PRIMARY_RECORDS)
    tracked = sample_records(val_pool, TRACK_RECORDS)
    fit_slices = {"trained": tracked, "held_out": held_out}
    val_historic = sample_records(historic_records, VAL_HISTORIC_RECORDS)
    # plot-only slice, never validated against in the csv-logged metrics
    lc0_pool = [r for r in primary_records_raw if r[6].startswith('lc0')]
    val_lc0 = sample_records(lc0_pool, VAL_LC0_RECORDS)
    # SF-played plies: plot and console only, like lc0; empty slices are skipped
    sf_pool = [r for r in primary_records_raw if r[6] == SF_SOURCE]
    val_sf = sample_records(sf_pool, VAL_SF_RECORDS)

    bs = cfg.retrain_batch_size
    val = {name: validate_slice(model, samples, bs)
           for name, samples in (("primary", val_primary),
                                 ("historic", val_historic),
                                 ("lc0", val_lc0), ("sf", val_sf))}
    fit_before = {name: predict_fp16(model, [r[0] for r in s], bs)
                  for name, s in fit_slices.items()}

    del (model, val_pool, val_primary, val_historic, lc0_pool, val_lc0,
         sf_pool, val_sf)
    torch.cuda.empty_cache()

    primary_records = shuffle_rewrite_primary(primary_paths, primary_records_raw)
    del primary_records_raw

    all_records = primary_records + replay_records + historic_records
    del primary_records, replay_records, historic_records
    random.shuffle(all_records)
    X, P, Y, vwht, pwht = records_to_arrays(all_records)
    # X and the arrays are copies; P holds only the policy objects
    del all_records

    result_q.put({"cmd": "retrain_ready"})
    wait_for_go(msg_q)
    t0 = time.monotonic()

    model, arch = load_pt_model(cfg.model_path)
    args = SimpleNamespace(batch_size=cfg.retrain_batch_size)

    timings = {}
    train_stats = retrain_pt(cfg.model_path, X, P, Y, vwht, pwht, cfg, epoch, args,
                             timings=timings, model=model, arch=arch,
                             holdout=records_to_arrays(held_out))
    # the model is saved, so its training shards are spent
    rotate_to_replay(primary_paths, cfg.replay_buffer_dir)
    discard_files(replay_paths)
    print("[retrain] primary -> replay rotated", flush=True)

    dt = time.monotonic() - t0
    n = timings["samples"]
    print(f"[retrain] timing for epoch {epoch}: {dt:.1f}s  "
          f"{n / dt:.0f} samples/s  {dt * SHARD_SIZE / n:.2f}s per {SHARD_SIZE}",
          flush=True)

    del X, P, Y, vwht, pwht, model
    torch.cuda.empty_cache()
    model, _ = load_pt_model(cfg.model_path)
    fit_after = {name: predict_fp16(model, [r[0] for r in s], bs)
                 for name, s in fit_slices.items()}
    del model
    report_fit_tracking(cfg, epoch, fit_slices, fit_before, fit_after)
    report_validation(epoch, cfg.progress_csv_path, val, train_stats)
    sys.stdout.flush()

    result_q.put({"cmd": "retrain_done", "ok": True, "error": None,
                  "train_stats": train_stats})
