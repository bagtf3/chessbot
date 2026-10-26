"""One-off: top the primary buffer up to --target shards by converting
historic tfrecord.gz into pkl.gz.

The primary buffer only reads pkl.gz (read_shard would hand a .tfrecord.gz to
TF, which works, but primary shards get rewritten in place by
shuffle_rewrite_primary and that always writes pkl.gz). So filling primary
from historic data means re-encoding, unlike the replay buffer, which takes
raw tfrecord.gz copies.

Records land with the SEED_SOURCE tag, matching seed_selfplay_buffers, so
validation groups them as historic.

The historic directory is read-only here -- files are re-encoded, never moved
or removed.

Usage:
  python scripts/tools/fill_primary_buffer.py <run_dir> --tfrec-dir <dir> \
      [--target 44] [--apply]
"""
import argparse
import os
import sys

from chessbot.pretrain import EPOCH_SIZE, VAL_SHUFFLE_BUFFER, list_tfrecord_files
from chessbot.replay_buffer import (
    PRIMARY_SEED_SHARDS, SEED_SOURCE, list_shard_files, new_shard_path,
    sparsify_policy, write_pkl_gz_shard,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tfrec-dir", default=os.getenv("BOOTSTRAP_TFREC_DIR", ""))
    ap.add_argument("--target", type=int, default=PRIMARY_SEED_SHARDS,
                    help="primary shard count to fill up to")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not args.tfrec_dir:
        ap.error("--tfrec-dir is required (or set $BOOTSTRAP_TFREC_DIR)")

    primary_dir = os.path.join(args.run_dir, "primary_buffer")
    os.makedirs(primary_dir, exist_ok=True)

    existing = list_shard_files(primary_dir)
    needed = max(0, args.target - len(existing))
    tfrec_files = list_tfrecord_files(args.tfrec_dir)

    print(f"[fill] primary {len(existing)}/{args.target}, need {needed} shards")
    print(f"[fill] historic source: {len(tfrec_files)} tfrecord.gz files")

    if needed == 0:
        print("[fill] nothing to do")
        return
    if not tfrec_files:
        raise SystemExit(f"[fill] no tfrecord.gz files in {args.tfrec_dir}")
    if not args.apply:
        print(f"[fill] dry run, would write {needed} pkl.gz shards of "
              f"{EPOCH_SIZE} records (pass --apply)")
        return

    from chessbot.pretrain import make_dataset

    ds = make_dataset(
        tfrec_files, shuffle_buffer=2 * VAL_SHUFFLE_BUFFER,
        batch_size=EPOCH_SIZE, repeat=False,
    )
    ds_iter = iter(ds)

    for i in range(needed):
        inp, out, _ = next(ds_iter)
        enc = inp["enc_in"].numpy()
        pol = out["policy_logits"].numpy()
        val = out["value_out"].numpy()
        records = [
            (enc[j], None, sparsify_policy(pol[j]), val[j], 1.0, 1.0, SEED_SOURCE)
            for j in range(enc.shape[0])
        ]
        path = new_shard_path(primary_dir)
        write_pkl_gz_shard(records, path)
        print(f"[fill] {i + 1}/{needed} wrote {len(records)} records -> "
              f"{os.path.basename(path)}")

    print(f"[fill] primary now {len(list_shard_files(primary_dir))} shards")
    sys.stdout.flush()
    # TF hangs on normal interpreter exit, same as the bootstrap script
    os._exit(0)


if __name__ == "__main__":
    main()
