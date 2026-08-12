"""LiveBuffer + primary/replay/historic directory management for the retrain
data pipeline. See replay_buffer_redesign.md for the full design.

Record schema (used uniformly regardless of source file format):
    (x, mask, policy, Y, vwht, pwht, source)
- x: board encoding array (int, model-input token sequence)
- mask: always None. Records written before the sparse-policy change carry a
  1858-element list of 0/1 legal flags here; nothing reads it. The slot is
  kept so tuple arity stays 7 and no element is renumbered.
- policy: either a dense float32[1858] (old records, and records built from
  historic tfrec.gz) or a sparse (int16 indices, float32 values) pair. At
  ~1.5% density the sparse form is what makes shards cheap to write and
  cheap to unpickle. Detect per record with is_sparse_policy; densify at the
  point of use with densify_policy / stack_policies. Both forms are
  supported indefinitely -- there is no migration pass, and a single shard
  can hold a mix, because run_selfplay reloads remaining_untrained straight
  into the live buffer without normalising. Never densify inside read_shard:
  records staying sparse through the read is the whole point. Primary shards
  self-convert as shuffle_rewrite_primary rewrites them each retrain;
  replay and historic simply age out.
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

from chessbot.opening_counts import apply_repeat_weights

LIVE_BUFFER_SIZE = 163_840
SHARD_SIZE = 10_240
PRIMARY_TRIGGER_SHARDS = 96
RETRAIN_PRIMARY_SHARDS = 8
RETRAIN_REPLAY_SHARDS = 8
RETRAIN_HISTORIC_SHARDS = 8
REPLAY_SEED_SHARDS = 144
# initial primary fill, left short of the trigger so selfplay supplies the
# last few shards of the first cycle rather than retraining on pure seed
PRIMARY_SEED_SHARDS = 94
SEED_SOURCE = "historic_seeded"

# Validation draws, taken in-memory from records the retrain worker has
# already loaded for training -- no extra shard reads. Primary only; the
# replay buffer has been trained on already and is left alone.
VAL_PRIMARY_RECORDS = 20_480
VAL_HISTORIC_RECORDS = 10_240

# gzip's default is 9, which costs ~4x the write time of 6 for ~4% less disk.
# Shard writes are on the rescorer's thread and run ~96 times per retrain
# cycle, so the trade is heavily in favour of the faster level.
GZIP_LEVEL = 6

POLICY_DIM = 1858


def new_shard_path(out_dir, suffix=".pkl.gz"):
    name = f"{int(time.time())}-{uuid.uuid4().hex}{suffix}"
    return os.path.join(out_dir, name)


def write_pkl_gz_shard(records, path):
    tmp_path = path + ".tmp"
    with gzip.open(tmp_path, "wb", compresslevel=GZIP_LEVEL) as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def sparsify_policy(policy, dim=POLICY_DIM):
    """Dense float32[dim] -> (int16 indices, float32 values) over nonzeros.

    flatnonzero rather than `> 0`: a negative value must round-trip, not be
    silently dropped. int16 holds 1858 with room to spare, but assert so a
    future POLICY_DIM bump fails loudly instead of wrapping negative.
    """
    assert dim <= 32767, f"POLICY_DIM {dim} does not fit int16 indices"
    p = np.asarray(policy, dtype=np.float32)
    idx = np.flatnonzero(p).astype(np.int16)
    return idx, p[idx]


def is_sparse_policy(p):
    """Sparse records carry a 2-tuple; dense ones carry an ndarray."""
    return isinstance(p, tuple)


def densify_policy(p, dim=POLICY_DIM):
    """Either form -> dense float32[dim]."""
    if not is_sparse_policy(p):
        return np.asarray(p, dtype=np.float32)
    idx, vals = p
    out = np.zeros(dim, dtype=np.float32)
    out[idx.astype(np.int32)] = vals
    return out


def stack_policies(records, index=2, dim=POLICY_DIM):
    """(N, dim) dense stack from a list of records in either format.

    The single densify point. Scatters straight into one preallocated array
    rather than building N intermediates and stacking them.
    """
    out = np.zeros((len(records), dim), dtype=np.float32)
    for i, r in enumerate(records):
        p = r[index]
        if is_sparse_policy(p):
            idx, vals = p
            out[i, idx.astype(np.int32)] = vals
        else:
            out[i] = p
    return out


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
        x = np.frombuffer(feat["xc0h_board"].numpy(), dtype=np.int16).copy()
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


REMAINING_GZ = "remaining_untrained.pkl.gz"
REMAINING_PKL = "remaining_untrained.pkl"


def find_remaining_untrained(run_dir):
    """Path to the untrained-sample carryover, gz preferred, else the plain
    pkl older runs left behind. None when neither is present."""
    for name in (REMAINING_GZ, REMAINING_PKL):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p
    return None


def read_remaining_untrained(path):
    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def write_remaining_untrained(run_dir, records):
    """Always writes gz. Any plain .pkl left from an older run is removed so
    the two cannot disagree about which carryover is current."""
    path = os.path.join(run_dir, REMAINING_GZ)
    write_pkl_gz_shard(records, path)
    stale = os.path.join(run_dir, REMAINING_PKL)
    if os.path.exists(stale):
        os.remove(stale)
    return path


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

    def __init__(self, primary_dir, capacity=LIVE_BUFFER_SIZE,
                 shard_size=SHARD_SIZE, counts=None):
        self.primary_dir = primary_dir
        self.capacity = capacity
        self.shard_size = shard_size
        self.records = []
        # opening-repetition slot per record, parallel to self.records.
        # UNTRACKED means the position was past the ply window; None means no
        # key was recorded, and maybe_flush re-derives it from the record's own
        # x. Kept alongside rather than inside the record so the on-disk schema
        # stays a 7-tuple.
        self.keys = []
        self.counts = counts

    def __len__(self):
        return len(self.records)

    def append(self, record, key=None):
        self.records.append(record)
        self.keys.append(key)

    def extend(self, records, keys=None):
        self.records.extend(records)
        self.keys.extend(keys if keys is not None else [None] * len(records))

    def load_records(self, records):
        """Replace the pool wholesale. Used by the state-reload and
        remaining_untrained paths, whose records outlive the key list; None
        keys make maybe_flush re-derive each slot from the record's own x, so a
        restart weights its carryover identically to records that never left
        memory."""
        self.records = list(records)
        self.keys = [None] * len(self.records)

    def maybe_flush(self):
        """Write one shard if at capacity. Returns the written path, or
        None if not yet full. Call in a loop if bulk-loading (e.g. from
        remaining_untrained.pkl) could fill more than one shard at once.

        This is the only place opening reweighting happens. It must not move
        into write_pkl_gz_shard -- shuffle_rewrite_primary calls that too, and
        would compound the weighting every retrain cycle."""
        if len(self.records) < self.capacity:
            return None

        idx = set(random.sample(range(len(self.records)), self.shard_size))
        chunk = [r for i, r in enumerate(self.records) if i in idx]
        chunk_keys = [k for i, k in enumerate(self.keys) if i in idx]
        self.records = [r for i, r in enumerate(self.records) if i not in idx]
        self.keys = [k for i, k in enumerate(self.keys) if i not in idx]

        if self.counts is not None:
            chunk = apply_repeat_weights(chunk, chunk_keys, self.counts)
            self.counts.decay()

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
