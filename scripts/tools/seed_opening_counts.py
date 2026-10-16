"""Seed a run's opening-repetition counter from finished game logs.

The counter fills on its own within a few hundred games, so this is a
convenience rather than a requirement -- it mainly saves the first cycle from
training on an unsuppressed opening distribution, and exercises the load path
before a run starts leaning on it.

Counts are seeded raw, not rescaled to steady state. Scaling a small sample up
would push positions seen once into a heavy band on the strength of a single
observation; leaving them raw means the seed only asserts what it actually saw.
The effect is that hot openings arrive already suppressed and everything else
starts near 1.0 and climbs, which is the safe direction to be wrong in.

UHO games are skipped because they open at ply 16 and would contribute nothing
anyway, and validation games because their opening mix is a fixed book rather
than the training curriculum.

Usage:
    python scripts/tools/seed_opening_counts.py <run_tag> [--games 300]
"""
import argparse
import gzip
import itertools
import os
import pickle
import random

import numpy as np
import pyfastchess as pf

from chessbot import SP_DIR
from chessbot.opening_counts import (
    COUNTS_FILE, MAX_TRACKED_PLY, OpeningCounts, absolute_ply, band_weights,
)

SKIP_SCENARIOS = {"UHO"}


def encode_board(board, encoding_type, history_K):
    if encoding_type == "lc0":
        return board.lc0_features()
    if encoding_type == "xc0h":
        return board.history_tokens(history_K)
    return np.array(board.encode_64_tokens(), dtype=np.int16)


def load_log(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def seed(run_dir, n_games, pool_mult=6, seed_val=0):
    log_dir = os.path.join(run_dir, "game_logs")
    if not os.path.isdir(log_dir):
        raise RuntimeError(f"no game_logs in {run_dir}")

    # names only, and only as many as we need -- stat'ing a full game_logs
    # tree is minutes of IO. Filenames are uuids, so head order is unbiased.
    pool = [e.path for e in itertools.islice(
        os.scandir(log_dir), n_games * pool_mult)]
    random.Random(seed_val).shuffle(pool)

    counts = OpeningCounts(os.path.join(run_dir, COUNTS_FILE))
    used = skipped_uho = skipped_val = 0
    enc_seen = set()

    for path in pool:
        if used >= n_games:
            break
        g = load_log(path)
        if g.get("scenario") in SKIP_SCENARIOS:
            skipped_uho += 1
            continue
        if g.get("is_validation_run"):
            skipped_val += 1
            continue

        enc = g.get("encoding_type", "xc0h")
        hk = g.get("history_K", 6)
        enc_seen.add((enc, hk))

        board = pf.Board(g["start_fen"])
        for mv in g["moves_played"]:
            if absolute_ply(board) > MAX_TRACKED_PLY:
                break
            x = encode_board(board, enc, hk)
            counts.bump(counts.key(board, x))
            board.push_uci(mv)
        used += 1

    return counts, used, skipped_uho, skipped_val, enc_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tag")
    ap.add_argument("--games", type=int, default=300)
    args = ap.parse_args()

    run_dir = os.path.join(SP_DIR, args.run_tag)
    counts, used, s_uho, s_val, enc_seen = seed(run_dir, args.games)

    print(f"games used {used}  (skipped {s_uho} UHO, {s_val} validation)")
    print(f"encodings seen: {sorted(enc_seen)}")
    print(f"bumps {counts.n_bumps}  distinct slots {counts.occupancy()}")

    nz = np.flatnonzero(counts.counts)
    vals = counts.counts[nz]
    order = np.argsort(vals)[::-1]
    print("\ntop slots:")
    for i in order[:8]:
        print("  slot %-10d count %6.0f  weight %.2f"
              % (nz[i], vals[i], band_weights([vals[i]])[0]))

    w = band_weights(vals)
    print("\nseeded weight distribution over tracked positions:")
    for lo, hi in [(1.0, 1.0), (0.6, 0.99), (0.3, 0.59), (0.0, 0.29)]:
        m = (w >= lo) & (w <= hi)
        print("  wt %.2f-%.2f  %6d slots  %7.0f bumps"
              % (lo, hi, m.sum(), vals[m].sum()))

    path = counts.save()
    size = os.path.getsize(path) / 1e6
    back = OpeningCounts(path)
    back.load()
    ok = np.array_equal(counts.counts, back.counts)
    print(f"\nwrote {path}  ({size:.2f} MB)")
    print(f"reload verified: {ok}  slots {back.occupancy()}  "
          f"bumps {back.n_bumps}")


if __name__ == "__main__":
    main()
