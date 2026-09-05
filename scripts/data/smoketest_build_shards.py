"""Smoke test: read migrated game_logs and write training shards.

Reads migrated pkl.gz files in order, applies CPL <= 30 filter,
hashes xc0h for ply < 16 (accumulator timing proxy), then writes
retrain-ready shards to <run_dir>/.

Usage:
    python scripts/data/smoketest_build_shards.py --run_tag 16m_precond_run2
"""
import argparse
import gzip
import os
import pickle
import time

import numpy as np

from chessbot import SP_DIR
from chessbot.replay_buffer import write_pkl_gz_shard

MAX_SHARDS = 5
SHARD_SIZE = 10240


def policy_to_sparse(policy_list):
    if not policy_list:
        return (np.array([], dtype=np.int16), np.array([], dtype=np.float32))
    idxs = np.array([i for i, v in policy_list], dtype=np.int16)
    vals = np.array([v for i, v in policy_list], dtype=np.float32)
    return (idxs, vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_tag', required=True)
    args = ap.parse_args()

    run_dir = os.path.join(SP_DIR, args.run_tag)
    log_dir = os.path.join(run_dir, 'game_logs')

    files = sorted(f for f in os.listdir(log_dir) if f.endswith('.pkl.gz'))
    print(f"{args.run_tag}: {len(files):,} files", flush=True)

    buffer = []
    n_shards = 0
    n_plies_seen = 0
    n_plies_kept = 0
    n_games = 0
    t0 = time.time()

    for fname in files:
        path = os.path.join(log_dir, fname)
        try:
            with gzip.open(path, 'rb') as f:
                record = pickle.load(f)
        except Exception as e:
            print(f"  skip {fname}: {e}", flush=True)
            continue

        plies = record.get('plies', [])
        n_games += 1

        for ply_idx, ply in enumerate(plies):
            n_plies_seen += 1

            cpl = ply.get('cpl')
            if cpl is None or cpl > 30:
                continue

            xc0h = ply.get('xc0h')
            if xc0h is None:
                continue

            xc0h = np.asarray(xc0h, dtype=np.int16)

            if ply_idx < 16:
                _ = hash(xc0h.tobytes())

            wdl_target = np.asarray(ply['wdl_target'], dtype=np.float32)
            sparse_policy = policy_to_sparse(ply.get('policy', []))

            buffer.append((xc0h, None, sparse_policy, wdl_target, 1.0, 1.0, 'historic'))
            n_plies_kept += 1

            if len(buffer) >= SHARD_SIZE:
                path_out = os.path.join(run_dir, f'smoketest_shard_{n_shards:02d}.pkl.gz')
                write_pkl_gz_shard(buffer, path_out)
                elapsed = time.time() - t0
                pos_s    = n_plies_seen / elapsed
                games_s  = n_games / elapsed
                shards_m = (n_shards + 1) / (elapsed / 60)
                print(f"  shard {n_shards}: written  |  "
                      f"{pos_s:.0f} pos/s  {games_s:.1f} games/s  "
                      f"{shards_m:.2f} shards/min  |  "
                      f"{n_plies_seen:,} pos seen  {n_games:,} games", flush=True)
                buffer = []
                n_shards += 1
                if n_shards >= MAX_SHARDS:
                    break

        if n_shards >= MAX_SHARDS:
            break

    elapsed = time.time() - t0
    pos_s    = n_plies_seen / max(elapsed, 1e-6)
    games_s  = n_games / max(elapsed, 1e-6)
    shards_m = n_shards / max(elapsed / 60, 1e-6)
    print(f"\ndone: {n_shards} shards  {n_plies_seen:,} pos seen  "
          f"{n_plies_kept:,} kept ({100*n_plies_kept/max(n_plies_seen,1):.1f}%)  "
          f"{elapsed:.1f}s", flush=True)
    print(f"      {pos_s:.0f} pos/s  {games_s:.1f} games/s  "
          f"{shards_m:.2f} shards/min", flush=True)


if __name__ == '__main__':
    main()
