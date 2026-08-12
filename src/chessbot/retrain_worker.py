"""Retrain worker process. Spawned fresh (ctx.Process) for each retrain
cycle and torn down after -- a deliberate VRAM-reclaim policy, not an
artifact of the old subprocess-script approach it replaces. See
replay_buffer_redesign.md for the full design and message protocol.

Message protocol (msg_q, main -> worker):
  {"cmd": "preload", "replay": [paths], "historic": [paths]}
  {"cmd": "start", "primary": [paths]}
      Both sent back to back the moment the worker is spawned, which only
      happens once primary_buffer is full at PRIMARY_TRIGGER_SHARDS. They
      stayed two messages because the worker consumes them in order and the
      split costs nothing.

Message protocol (result_q, worker -> main):
  {"cmd": "retrain_ready"}
      validation predictions written, dataset built, about to train. Main
      pauses selfplay workers on receipt of this.

predictions_latest.pkl layout (consumed by Rescorer.aggregate_metrics):
  {"primary":  {"samples": [...], "preds": [...]},
   "historic": {"samples": [...], "preds": [...]}}
  {"cmd": "retrain_done", "ok": bool, "error": str or None,
   "train_stats": {"train_loss": float, "gn_mean": float}}
      training + save finished (or failed). Main unpauses selfplay workers
      immediately on receipt of this, before running aggregate_metrics, which
      folds train_stats into the validation rows.
"""
import os
import pickle
import random
import sys
import time
from types import SimpleNamespace

import numpy as np

from chessbot.config import Config
from chessbot.replay_buffer import (
    read_shard, sample_records, write_pkl_gz_shard,
    SHARD_SIZE, VAL_HISTORIC_RECORDS, VAL_PRIMARY_RECORDS,
)


def load_records(paths):
    records = []
    for path in paths:
        records.extend(read_shard(path))
    return records


def shuffle_rewrite_primary(paths, records, shard_size=SHARD_SIZE):
    """Shuffle already-loaded primary records across file boundaries and
    rewrite shard_size-sized chunks back into the same filenames. Takes the
    records the caller already read (for validation) rather than re-reading
    from disk. Returns the shuffled record list -- already the exact primary
    portion of the retrain set."""
    all_records = list(records)
    random.shuffle(all_records)
    for i, p in enumerate(paths):
        chunk = all_records[i * shard_size:(i + 1) * shard_size]
        write_pkl_gz_shard(chunk, p)
    return all_records


def predict_fp16(model, X_list, batch_size):
    import torch
    import torch.nn.functional as F
    device = torch.device("cuda")
    model = model.to(device).half().eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(X_list), batch_size):
            batch = np.stack(X_list[i:i + batch_size], axis=0)
            xb = torch.from_numpy(batch).long().to(device)
            pol_logits, val_out = model(xb)
            pol_np = pol_logits.float().cpu().numpy()
            wdl_np = F.softmax(val_out.float(), dim=-1).cpu().numpy()
            for j in range(len(pol_np)):
                results.append((wdl_np[j].tolist(), pol_np[j].tolist()))
    return results


def records_to_arrays(records):
    X = np.stack([r[0] for r in records]).astype(np.int32)
    P = np.stack([r[2] for r in records]).astype(np.float32)
    Y = np.array([r[3] for r in records], dtype=np.float32)
    vwht = np.array([r[4] for r in records], dtype=np.float32)
    pwht = np.array([r[5] for r in records], dtype=np.float32)
    return X, P, Y, vwht, pwht


def run_retrain_worker(run_dir, msg_q, result_q, epoch):
    try:
        retrain_worker_body(run_dir, msg_q, result_q, epoch)
    except Exception as e:
        result_q.put({"cmd": "retrain_done", "ok": False, "error": str(e)})
    sys.stdout.flush()
    sys.stderr.flush()
    # TF (used for historic tfrec.gz reads) hangs on normal interpreter exit
    # -- hard-exit instead, matching shuffle_xc0.py / bootstrap_model_async.
    os._exit(0)


def retrain_worker_body(run_dir, msg_q, result_q, epoch):
    import torch
    from chessbot.train_pytorch import load_pt_model, retrain_pt, update_player_ema

    cfg = Config.from_yaml(os.path.join(run_dir, "config.yaml"))

    preload_msg = msg_q.get()
    replay_records = load_records(preload_msg["replay"])
    historic_records = load_records(preload_msg["historic"])

    start_msg = msg_q.get()
    primary_paths = start_msg["primary"]

    model, arch = load_pt_model(cfg.model_path)
    primary_records_raw = load_records(primary_paths)

    # Two validation streams, both drawn from records already in RAM for
    # training: primary (mixed sources, tracks the live distribution) and
    # historic (pretrain distribution, continues the pretraining curve).
    # Replay is deliberately excluded -- it has already been trained on.
    val_primary = sample_records(primary_records_raw, VAL_PRIMARY_RECORDS)
    val_historic = sample_records(historic_records, VAL_HISTORIC_RECORDS)

    bs = cfg.retrain_batch_size
    preds_primary = predict_fp16(model, [r[0] for r in val_primary], batch_size=bs)
    preds_historic = predict_fp16(model, [r[0] for r in val_historic], batch_size=bs)

    pred_pkl_path = os.path.join(run_dir, "predictions_latest.pkl")
    payload = {
        "primary": {"samples": val_primary, "preds": preds_primary},
        "historic": {"samples": val_historic, "preds": preds_historic},
    }
    with open(pred_pkl_path + ".tmp", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(pred_pkl_path + ".tmp", pred_pkl_path)

    del model
    torch.cuda.empty_cache()

    primary_records = shuffle_rewrite_primary(primary_paths, primary_records_raw)

    all_records = primary_records + replay_records + historic_records
    random.shuffle(all_records)
    X, P, Y, vwht, pwht = records_to_arrays(all_records)

    result_q.put({"cmd": "retrain_ready"})
    time.sleep(10)

    model, arch = load_pt_model(cfg.model_path)
    args = SimpleNamespace(batch_size=cfg.retrain_batch_size)

    # snapshot before training. On the first retrain this seeds the EMA with
    # the weights that have been playing, so the average never starts cold --
    # this run began from a pretrain SWA, which is already a 5-way blend.
    seed_state = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()}

    train_stats = retrain_pt(cfg.model_path, X, P, Y, vwht, pwht, cfg, epoch, args,
                             model=model, arch=arch)

    update_player_ema(cfg.model_path, run_dir, model, arch, seed_state=seed_state)

    result_q.put({"cmd": "retrain_done", "ok": True, "error": None,
                  "train_stats": train_stats})
