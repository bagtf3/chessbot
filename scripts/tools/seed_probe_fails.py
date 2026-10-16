"""
One-shot: seed n_fails on every probe-pool record from its ratcheted sf_ms.

Run with the selfplay process STOPPED -- the runner holds the pool in RAM and
rewrites it on maybe_save_pool, so a live edit gets clobbered.

Usage: python seed_probe_fails.py [--dry-run]
"""
import argparse
import gzip
import os
import pickle
import shutil
import sys
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

from chessbot.blunder_replay import probe_fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pool", default=os.environ.get("BLUNDER_POSITIONS", ""))
    args = ap.parse_args()

    if not args.pool or not os.path.exists(args.pool):
        print(f"[seed] pool not found: {args.pool!r}", flush=True)
        sys.exit(1)

    with gzip.open(args.pool, "rb") as f:
        pool = pickle.load(f)
    print(f"[seed] loaded {len(pool):,} records from "
          f"{os.path.basename(args.pool)}", flush=True)

    # an n_fails written before seeding counted from zero and ignores the
    # position's history, so sf_ms wins wherever it is higher
    hist = Counter()
    seeded = 0
    for rec in pool.values():
        n = max(rec.get("n_fails", 0), probe_fails(rec))
        if not args.dry_run:
            rec["n_fails"] = n
        hist[n] += 1
        seeded += 1

    print(f"[seed] {'would seed' if args.dry_run else 'seeded'} "
          f"{seeded:,} records", flush=True)
    print("[seed] n_fails distribution:", flush=True)
    for k in sorted(hist):
        print(f"        {k:>3} -> {hist[k]:,}", flush=True)
    at_thresh = sum(v for k, v in hist.items() if k >= 3)
    print(f"[seed] {at_thresh:,} positions at 3+ fails "
          f"(reviewable on next failed probe)", flush=True)

    if args.dry_run:
        print("[seed] dry run, nothing written", flush=True)
        return

    backup = args.pool + ".bak_preseed"
    shutil.copy2(args.pool, backup)
    print(f"[seed] backup -> {os.path.basename(backup)}", flush=True)

    tmp = args.pool + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, args.pool)
    print(f"[seed] wrote {os.path.basename(args.pool)}", flush=True)


if __name__ == "__main__":
    main()
