"""Slowly move shard files from one buffer dir into another.

Drip-feeds rather than dumping everything at once so a live selfplay run
takes the donated data across several retrain cycles instead of one, which
keeps any single retrain from being dominated by the donor run's data.

Moves, never copies -- the source emptying is the stop condition.

Usage:
  python scripts/tools/drain_buffer.py <src_dir> <dst_dir> [--every 600]
"""
import argparse
import os
import random
import shutil
import sys
import time

from chessbot.replay_buffer import list_shard_files


def stamp():
    return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("dst_dir")
    ap.add_argument("--every", type=int, default=600,
                    help="seconds between moves (default 600)")
    args = ap.parse_args()

    if not os.path.isdir(args.src_dir):
        raise SystemExit(f"[drain] source not found: {args.src_dir}")
    os.makedirs(args.dst_dir, exist_ok=True)

    total = len(list_shard_files(args.src_dir))
    print(f"[drain] {stamp()} start: {total} shard(s) in {args.src_dir}")
    print(f"[drain] {stamp()} -> {args.dst_dir}, one every {args.every}s "
          f"(first move in {args.every}s, ETA ~{total * args.every / 60:.0f} min)")
    sys.stdout.flush()

    moved = 0
    while True:
        time.sleep(args.every)

        files = list_shard_files(args.src_dir)
        if not files:
            print(f"[drain] {stamp()} source empty -- moved {moved} shard(s), done")
            sys.stdout.flush()
            return

        src = random.choice(files)
        name = os.path.basename(src)
        dst = os.path.join(args.dst_dir, name)
        if os.path.exists(dst):
            # timestamp-uuid names, so this only happens if the same file was
            # already delivered; drop the duplicate rather than overwrite
            os.remove(src)
            print(f"[drain] {stamp()} {name} already in dest, removed from source")
            sys.stdout.flush()
            continue

        shutil.move(src, dst)
        moved += 1
        left = len(list_shard_files(args.src_dir))
        print(f"[drain] {stamp()} moved {name}  ({moved} done, {left} left, "
              f"dest now {len(list_shard_files(args.dst_dir))})")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
