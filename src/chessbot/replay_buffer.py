"""LiveBuffer + primary/replay/historic directory management for the retrain
data pipeline. See replay_buffer_redesign.md for the full design.

Record schema (used uniformly regardless of source file format):
    (x, mask, policy, Y, vwht, pwht, source)
- x: board encoding array (int, model-input token sequence)
- mask: legal move mask, unused downstream during retrain (kept for shape
  parity with live selfplay records)
- policy: float32[1858] policy target
- Y: float32[3] WDL target
- vwht, pwht: value/policy sample weights
- source: provenance tag string for real selfplay data; None for the old
  tfrec.gz historic booster shards read via read_tfrec_gz_shard, which
  carry no source; SEED_SOURCE for the bootstrap-seeded primary/
  remaining_untrained records (see seed_selfplay_buffers), which do carry
  one. For validation grouping (Rescorer.plot_source) SEED_SOURCE and None
  both map to "historic" -- they are pretrain-distribution records, not
  selfplay output. Anything starting with "lc0" maps to "lc0"; the
  lc0_pv/lc0_blunder/lc0_enrich distinction survives only in the
  print_sample_stats counters.
"""
import gzip
import os
import pickle
import random
import time
import uuid

import numpy as np

LIVE_BUFFER_SIZE = 122_880
SHARD_SIZE = 10_240
PRIMARY_TRIGGER_SHARDS = 64
RETRAIN_PRIMARY_SHARDS = 8
RETRAIN_REPLAY_SHARDS = 8
RETRAIN_HISTORIC_SHARDS = 8
REPLAY_SEED_SHARDS = 96
# initial primary fill, left short of the trigger so selfplay supplies the
# last few shards of the first cycle rather than retraining on pure seed
PRIMARY_SEED_SHARDS = 62
SEED_SOURCE = "historic_seeded"

# Validation draws, taken in-memory from records the retrain worker has
# already loaded for training -- no extra shard reads. Primary only; the
# replay buffer has been trained on already and is left alone.
VAL_PRIMARY_RECORDS = 20_480
VAL_HISTORIC_RECORDS = 10_240


def new_shard_path(out_dir, suffix=".pkl.gz"):
    name = f"{int(time.time())}-{uuid.uuid4().hex}{suffix}"
    return os.path.join(out_dir, name)


def write_pkl_gz_shard(records, path):
    tmp_path = path + ".tmp"
    with gzip.open(tmp_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def read_pkl_gz_shard(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def read_tfrec_gz_shard(path):
    """Read a historic bootstrap tfrecord.gz shard, normalized to the same
    7-tuple record schema as pkl.gz shards. Historic data carries no mask,
    no explicit sample weights (both default to 1.0), and no source tag."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "3")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    feature_spec = {
        "xc0h_board": tf.io.FixedLenFeature([], tf.string),
        "policy":     tf.io.FixedLenFeature([], tf.string),
        "wdl":        tf.io.FixedLenFeature([3], tf.float32),
    }

    records = []
    dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")
    for raw in dataset:
        feat = tf.io.parse_single_example(raw, feature_spec)
        x = np.frombuffer(feat["xc0h_board"].numpy(), dtype=np.int16).astype(np.int32)
        policy = np.frombuffer(feat["policy"].numpy(), dtype=np.float32).copy()
        policy = policy / policy.sum()
        Y = feat["wdl"].numpy().astype(np.float32)
        records.append((x, None, policy, Y, 1.0, 1.0, None))
    return records


def read_shard(path):
    lower = str(path).lower()
    if lower.endswith(".tfrecord.gz") or lower.endswith(".tfrec.gz"):
        return read_tfrec_gz_shard(path)
    return read_pkl_gz_shard(path)


def list_shard_files(dir_path):
    if not os.path.isdir(dir_path):
        return []
    exts = (".pkl.gz", ".tfrecord.gz", ".tfrec.gz")
    return sorted(
        os.path.join(dir_path, fn) for fn in os.listdir(dir_path)
        if fn.lower().endswith(exts)
    )


def sample_files(dir_path, n):
    files = list_shard_files(dir_path)
    if len(files) < n:
        raise RuntimeError(f"{dir_path}: need {n} shard files, found {len(files)}")
    return random.sample(files, n)


def sample_records(records, n):
    """Random subset without replacement, for validation draws. Returns the
    whole list (shuffled) if it holds fewer than n."""
    if len(records) <= n:
        return list(records)
    return random.sample(records, n)


def move_files(paths, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    moved = []
    for p in paths:
        dest = os.path.join(dest_dir, os.path.basename(p))
        os.replace(p, dest)
        moved.append(dest)
    return moved


def discard_files(paths):
    for p in paths:
        os.remove(p)


def seed_replay_buffer(historic_dir, replay_dir, n_files=REPLAY_SEED_SHARDS):
    """Copy n_files chosen at random from historic_dir into replay_dir.
    historic_dir is never written to or modified. No-op if replay_dir
    already has n_files or more (idempotent across restarts)."""
    os.makedirs(replay_dir, exist_ok=True)
    existing = list_shard_files(replay_dir)
    if len(existing) >= n_files:
        return existing

    historic_files = list_shard_files(historic_dir)
    if len(historic_files) < n_files:
        raise RuntimeError(
            f"historic dir {historic_dir}: need {n_files} files, "
            f"found {len(historic_files)}")

    needed = n_files - len(existing)
    chosen = random.sample(historic_files, needed)
    seeded = []
    for src in chosen:
        dest = os.path.join(replay_dir, os.path.basename(src))
        with open(src, "rb") as fin, open(dest + ".tmp", "wb") as fout:
            fout.write(fin.read())
        os.replace(dest + ".tmp", dest)
        seeded.append(dest)
    return existing + seeded


class LiveBuffer:
    """In-RAM record pool, replaces the flat rescorer.training_data list.
    Flushes shard_size records at random (without replacement) to
    primary_dir as pkl.gz whenever it fills to capacity."""

    def __init__(self, primary_dir, capacity=LIVE_BUFFER_SIZE, shard_size=SHARD_SIZE):
        self.primary_dir = primary_dir
        self.capacity = capacity
        self.shard_size = shard_size
        self.records = []

    def __len__(self):
        return len(self.records)

    def append(self, record):
        self.records.append(record)

    def extend(self, records):
        self.records.extend(records)

    def maybe_flush(self):
        """Write one shard if at capacity. Returns the written path, or
        None if not yet full. Call in a loop if bulk-loading (e.g. from
        remaining_untrained.pkl) could fill more than one shard at once."""
        if len(self.records) < self.capacity:
            return None

        idx = set(random.sample(range(len(self.records)), self.shard_size))
        chunk = [r for i, r in enumerate(self.records) if i in idx]
        self.records = [r for i, r in enumerate(self.records) if i not in idx]

        os.makedirs(self.primary_dir, exist_ok=True)
        path = new_shard_path(self.primary_dir)
        write_pkl_gz_shard(chunk, path)
        return path

    def flush_all_ready(self):
        """Drain every full shard currently available. Returns list of
        written paths."""
        written = []
        while True:
            path = self.maybe_flush()
            if path is None:
                break
            written.append(path)
        return written
