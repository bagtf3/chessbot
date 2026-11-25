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

from chessbot import FIT_CHECKIN_SHARD
from chessbot.config import Config
from chessbot.opening_counts import apply_historic_dup_weights
from chessbot.replay_buffer import (
    discard_files, is_correction_shard, list_shard_files, read_shard, rotate_to_replay,
    sample_files, sample_holdout_primary, sample_records,
    sample_retrain_primary, stack_policies, write_pkl_gz_shard,
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
# in-sample records scored at each patience eval: from the trained primary
# shards (correction shards excluded) and from the drawn replay shards
TRACK_RECORDS = 5_120
# retired fit check csv, still carried to new runs for its history
FIT_CSV = "fit_tracking.csv"
# training weights by the start weights' miss, KL(target || net), ranked over
# the whole training set per head: (top fraction, multiplier), a narrower
# tier overriding a wider one, and the bottom MISS_FLOOR (fraction,
# multiplier) for the smallest misses. A record whose stored weight is at or
# under KEEP_WEIGHT_MAX keeps it, times the multiplier; any other record takes
# the multiplier as its weight. retrain_pt then pins each head's mean to 1.
MISS_TIERS = ((0.25, 1.25), (0.15, 1.5), (0.10, 2.0), (0.05, 2.5))
MISS_FLOOR = (0.10, 0.5)
KEEP_WEIGHT_MAX = 0.5
# training-set records scored by score_tiers, start vs picked weights
TIER_RECORDS = 20_480


def tier_names():
    """Miss tier labels, bottom first: MISS_FLOOR, the middle, then one per
    MISS_TIERS entry, widest first."""
    fracs = [f for f, _ in MISS_TIERS] + [0.0]
    names = [f"bot{MISS_FLOOR[0] * 100:g}", "mid"]
    for hi, lo in zip(fracs, fracs[1:]):
        names.append(f"{lo * 100:g}-{hi * 100:g}" if lo else f"top{hi * 100:g}")
    return names


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


def miss_tiers(miss):
    """Each record's tier_names index from its start-weight miss."""
    tier = np.ones(len(miss), dtype=np.int8)
    tier[miss <= np.quantile(miss, MISS_FLOOR[0])] = 0
    for k, (frac, _) in enumerate(MISS_TIERS):
        tier[miss >= np.quantile(miss, 1.0 - frac)] = k + 2
    return tier


def miss_weights(tier, stored):
    """One head's training weights from its miss tiers (MISS_TIERS,
    MISS_FLOOR)."""
    mult = np.array([MISS_FLOOR[1], 1.0] + [m for _, m in MISS_TIERS])[tier]
    return np.where(stored <= KEEP_WEIGHT_MAX, stored * mult, mult).astype(np.float32)


def validate_slice(samples, wdl_pred, logits):
    """Value predictions and per-row policy metrics for one validation slice
    from its start-pass predictions, VAL_CHUNK rows at a time. Input to
    eval_metrics.report_validation."""
    wdl, policy, top_target = [], defaultdict(list), []
    for i in range(0, len(samples), VAL_CHUNK):
        part = samples[i:i + VAL_CHUNK]
        w = wdl_pred[i:i + VAL_CHUNK]
        labels = stack_policies(part)
        for k, v in policy_metric_rows(logits[i:i + VAL_CHUNK], labels,
                                       labels > 0).items():
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
    from chessbot.train_pytorch import load_pt_model, retrain_pt, start_pass

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
    corr_paths = [p for p in primary_paths if is_correction_shard(p)]
    normal_records = load_records([p for p in primary_paths if p not in corr_paths])
    primary_records = normal_records + load_records(corr_paths)
    held_out = load_records(sample_holdout_primary(
        primary, primary_paths, RETRAIN_HOLDOUT_SHARDS))
    checkin = load_records([FIT_CHECKIN_SHARD]) if FIT_CHECKIN_SHARD else None

    # every sampled set is indices into all_records: primary (correction
    # shards last), replay, historic
    all_records = primary_records + replay_records + historic_records
    n_norm, n_prim, n_rep = len(normal_records), len(primary_records), len(replay_records)
    del normal_records
    sources = [r[6] for r in all_records]

    def pick(idx, n, pred=None):
        pool = [i for i in idx if pred is None or pred(sources[i])]
        return sorted(sample_records(pool, n))

    prim_idx = range(n_prim)
    # primary validation is pure xc0 only; lc0 and SF-played plies are plot
    # and console only; empty slices are skipped
    val_idx = {
        "primary": pick(prim_idx, VAL_PRIMARY_RECORDS, lambda s: s == 'xc0'),
        "historic": pick(range(n_prim + n_rep, len(all_records)), VAL_HISTORIC_RECORDS),
        "lc0": pick(prim_idx, VAL_LC0_RECORDS, lambda s: s.startswith('lc0')),
        "sf": pick(prim_idx, VAL_SF_RECORDS, lambda s: s == SF_SOURCE),
    }
    tracked_idx = {"primary": pick(range(n_norm), TRACK_RECORDS),
                   "replay": pick(range(n_prim, n_prim + n_rep), TRACK_RECORDS)}
    tier_idx = pick(range(len(all_records)), TIER_RECORDS)

    X, P, Y, vwht, pwht = records_to_arrays(all_records)
    # one start-weight pass over the training set: the miss weights, the
    # validation slices, the tier sample and the tracked sets' start loss all
    # come from it
    keep = sorted(set(tier_idx).union(*val_idx.values()))
    sp = start_pass(model, X, P, Y, cfg.retrain_batch_size, cfg.encoding_type, keep)
    row = {i: k for k, i in enumerate(keep)}
    val = {}
    for name, idx in val_idx.items():
        rows = [row[i] for i in idx]
        val[name] = validate_slice([all_records[i] for i in idx],
                                   sp["wdl"][rows], sp["logits"][rows])
    tier_p = miss_tiers(sp["lp"] - sp["hp"])
    tier_v = miss_tiers(sp["lv"] - sp["hv"])
    pwht = miss_weights(tier_p, pwht)
    vwht = miss_weights(tier_v, vwht)
    rows = [row[i] for i in tier_idx]
    tiers = {"X": X[tier_idx], "Y": Y[tier_idx],
             "P": stack_policies([P[i] for i in tier_idx], index=None).astype(np.float16),
             "wdl0": sp["wdl"][rows], "logits0": sp["logits"][rows],
             "tier_p": tier_p[tier_idx], "tier_v": tier_v[tier_idx],
             "names": tier_names()}
    tracked = {k: ((X[idx], [P[i] for i in idx], Y[idx], vwht[idx], pwht[idx]),
                   sp["lp"][idx], sp["lv"][idx])
               for k, idx in tracked_idx.items()}
    del model, sp
    torch.cuda.empty_cache()

    # the shards keep their stored weights; the miss weights live only here
    shuffle_rewrite_primary(primary_paths, primary_records)
    # X and the arrays are copies; P holds only the policy objects
    del all_records, primary_records, replay_records, historic_records

    result_q.put({"cmd": "retrain_ready"})
    wait_for_go(msg_q)
    t0 = time.monotonic()

    model, arch = load_pt_model(cfg.model_path)
    args = SimpleNamespace(batch_size=cfg.retrain_batch_size)

    timings = {}
    # the held-out shard is scored unweighted
    hX, hP, hY, hv, hp = records_to_arrays(held_out)
    train_stats = retrain_pt(cfg.model_path, X, P, Y, vwht, pwht, cfg, epoch, args,
                             timings=timings, model=model, arch=arch,
                             holdout=(hX, hP, hY, np.ones_like(hv), np.ones_like(hp)),
                             tracked=tracked,
                             checkin=records_to_arrays(checkin) if checkin else None,
                             tiers=tiers)
    # the model is saved, so its training shards are spent
    rotate_to_replay(primary_paths, cfg.replay_buffer_dir)
    discard_files(replay_paths)

    dt = time.monotonic() - t0
    n = timings["samples"]
    print(f"[retrain] {epoch} done in {dt:.0f}s  {SHARD_SIZE * dt / n:.1f}s per "
          f"{SHARD_SIZE}", flush=True)

    del X, P, Y, vwht, pwht, model
    torch.cuda.empty_cache()
    report_validation(epoch, cfg.progress_csv_path, val, train_stats)
    sys.stdout.flush()

    result_q.put({"cmd": "retrain_done", "ok": True, "error": None,
                  "train_stats": train_stats})
