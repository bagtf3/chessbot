"""
Reset the live blunder pool's probe history for a new selfplay run.

Keeps every position and every cached observation (the SF depth-ratchet
cache and the lc0 teacher move), and clears only the per-position probe
history: fail count, escalated SF budget, rediscovery count, and the epoch
bookkeeping. The new run's epoch counter restarts at 0, so last_seen is set
below the cooldown floor rather than to 0 -- otherwise probe_is_due stays
False for the first BRP_LAST_SEEN_MIN retrains and nothing gets probed.

Usage:
  python scripts/reset_blunder_pool.py --pool <path.pkl.gz> [--dry-run]
"""
import argparse
import datetime
import gzip
import os
import pickle
import shutil
import sys

from chessbot.blunder_replay import BRP_LAST_SEEN_MIN

KEEP_CACHE = ("deep_evals", "deep_depths", "deep_best_move", "deep_best_cp",
              "lc0_move")
DROP_KEYS = ("sf_ms",)
FRESH_LAST_SEEN = -(BRP_LAST_SEEN_MIN + 1)


def backup_pool(path):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.replace(".pkl.gz", f"_prereset_{stamp}.pkl.gz")
    shutil.copy2(path, dest)
    return dest


def reset_record(rec):
    """In-place history wipe for one pool entry. Returns the fields hit."""
    touched = []
    for key in DROP_KEYS:
        if key in rec:
            del rec[key]
            touched.append(key)
    for key, value in (("last_seen", FRESH_LAST_SEEN), ("n_fails", 0),
                       ("times_seen", 1), ("model_epoch", 0)):
        if rec.get(key) != value:
            touched.append(key)
        rec[key] = value
    return touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.environ.get("BLUNDER_POSITIONS", ""))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.pool:
        sys.exit("no pool path: pass --pool or set BLUNDER_POSITIONS")
    if not os.path.exists(args.pool):
        sys.exit(f"no pool at {args.pool!r}")

    with gzip.open(args.pool, "rb") as f:
        pool = pickle.load(f)
    print(f"[reset] loaded {len(pool)} positions from {args.pool}", flush=True)

    counts = {}
    cached = 0
    for rec in pool.values():
        for key in reset_record(rec):
            counts[key] = counts.get(key, 0) + 1
        if any(k in rec for k in KEEP_CACHE):
            cached += 1

    for key in sorted(counts):
        print(f"[reset]   {key:14s} changed on {counts[key]}", flush=True)
    print(f"[reset] cache obs kept on {cached} positions", flush=True)

    if args.dry_run:
        print("[reset] dry run, nothing written", flush=True)
        return

    print(f"[reset] backup -> {backup_pool(args.pool)}", flush=True)
    tmp = args.pool + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, args.pool)
    print(f"[reset] wrote {args.pool}", flush=True)


if __name__ == "__main__":
    main()
