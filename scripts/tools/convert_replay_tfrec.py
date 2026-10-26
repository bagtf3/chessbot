"""One-off: convert the tfrec shards left in a run's replay buffer to sparse
pkl.gz shards, dup-weighted the same way the historic draw is.

The replay buffer is a mix: pkl.gz shards written by the live buffer and
tfrecord.gz booster shards seeded at run start. The tfrec ones are read through
TF on every retrain, arrive dense, and carry flat 1.0 weights -- they have never
been through the repeat suppression the historic draw gets at
apply_historic_dup_weights. This reads all of them at once, applies that same
band table (1.00 / 0.85 / 0.66 / 0.33 by in-batch repeat count, mass-preserving),
and rewrites the lot as 10240-record sparse pkl.gz shards.

Weighting runs over groups of files rather than the whole buffer at once. The
band table is calibrated against in-batch repeat counts, so those counts scale
with how much data is weighed together -- pooling every shard would push
positions into harsher bands than the historic draw ever puts them in, since
that draw only ever sees RETRAIN_HISTORIC_SHARDS at a time. Two groups keeps
the population per weighing in the same neighbourhood. Each group is weighted
and rectified independently; the results are pooled and shuffled before
chunking, so a written shard is not a rewrite of any one input file.

The tfrec sources are deleted once their replacements are on disk -- leaving
them in place would double the data in the buffer.

Usage:
    python scripts/tools/convert_replay_tfrec.py <run_tag> [--apply]
"""
import argparse
import os
import random

from chessbot import SP_DIR
from chessbot.opening_counts import apply_historic_dup_weights
from chessbot.replay_buffer import (
    SHARD_SIZE, new_shard_path, read_shard, sparsify_policy,
    write_pkl_gz_shard,
)

TFREC_EXTS = (".tfrecord.gz", ".tfrec.gz")


def tfrec_files(replay_dir):
    return sorted(
        os.path.join(replay_dir, fn) for fn in os.listdir(replay_dir)
        if fn.lower().endswith(TFREC_EXTS)
    )


def to_sparse(record):
    """Dense tfrec record -> the sparse form the live buffer writes. Source
    stays None: these are pretrain-distribution records, and None is what the
    schema already uses to say so."""
    x, mask, policy, Y, vwht, pwht, source = record
    return (x, mask, sparsify_policy(policy), Y, vwht, pwht, source)


def split_groups(paths, n_groups):
    """Near-equal contiguous slices of an already-shuffled file list."""
    per = len(paths) / n_groups
    return [paths[round(i * per):round((i + 1) * per)] for i in range(n_groups)]


def weigh_group(paths, label):
    records = []
    for i, p in enumerate(paths, 1):
        chunk = read_shard(p)
        records.extend(chunk)
        print(f"[convert]   {label} {i:>3}/{len(paths)}  "
              f"{os.path.basename(p)}  {len(chunk)} records")

    weighted = apply_historic_dup_weights(records)
    n = len(weighted)
    vw = [r[4] for r in weighted]
    pw = [r[5] for r in weighted]
    print(f"[convert] {label}: {n} records  "
          f"vwht mean={sum(vw) / n:.3f} min={min(vw):.3f} max={max(vw):.3f}  "
          f"pwht mean={sum(pw) / n:.3f} min={min(pw):.3f} max={max(pw):.3f}")
    return weighted


def convert(run_dir, apply_changes, keep_source, n_groups, seed_val):
    replay_dir = os.path.join(run_dir, "replay_buffer")
    if not os.path.isdir(replay_dir):
        raise RuntimeError(f"no replay_buffer in {run_dir}")

    paths = tfrec_files(replay_dir)
    if not paths:
        print("[convert] no tfrec shards in the replay buffer, nothing to do")
        return

    rng = random.Random(seed_val)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    groups = split_groups(shuffled, n_groups)
    print(f"[convert] {len(paths)} tfrec shards in {replay_dir}")
    print(f"[convert] weighing in {n_groups} groups of "
          f"{[len(g) for g in groups]} files")

    pool = []
    for gi, group in enumerate(groups, 1):
        pool.extend(weigh_group(group, f"g{gi}"))

    n = len(pool)
    n_shards = n // SHARD_SIZE
    leftover = n - n_shards * SHARD_SIZE
    print(f"[convert] {n} records -> {n_shards} shards of {SHARD_SIZE}, "
          f"{leftover} left over (dropped)")

    if not apply_changes:
        print("[convert] dry run -- rerun with --apply to write")
        return

    rng.shuffle(pool)
    written = []
    for i in range(n_shards):
        chunk = [to_sparse(r) for r in
                 pool[i * SHARD_SIZE:(i + 1) * SHARD_SIZE]]
        path = new_shard_path(replay_dir)
        write_pkl_gz_shard(chunk, path)
        written.append(path)
    print(f"[convert] wrote {len(written)} pkl.gz shards")

    if keep_source:
        print(f"[convert] kept {len(paths)} tfrec shards in place -- the "
              f"buffer now holds both copies")
        return
    for p in paths:
        os.remove(p)
    print(f"[convert] removed {len(paths)} tfrec shards")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tag")
    ap.add_argument("--apply", action="store_true",
                    help="write shards; without it the script only reports")
    ap.add_argument("--keep-source", action="store_true",
                    help="leave the tfrec shards in place after writing")
    ap.add_argument("--groups", type=int, default=2,
                    help="independent weighing groups; more groups = milder "
                         "banding, since repeat counts scale with batch size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = os.path.join(SP_DIR, args.run_tag)
    convert(run_dir, args.apply, args.keep_source, args.groups, args.seed)


if __name__ == "__main__":
    main()
