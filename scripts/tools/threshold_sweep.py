"""
Sweep |Q| and p_draw thresholds across game logs and analyze metrics.

Samples N games, matches with Stockfish analysis, and reports mean CPL, CE, KL
for each |Q| x p_draw threshold combination. Writes the grid to CSV.

Usage:
  python threshold_sweep.py --run_tag 18m_10c6t_SWA_selfplay2 --n_games 10000
"""
import argparse
import os
import pickle
import gzip
import glob
import random
import time
import numpy as np
import pandas as pd
from tabulate import tabulate

from chessbot import SP_DIR


def say(msg):
    """Unbuffered print -- stdout block-buffers when piped to a file."""
    print(msg, flush=True)


def main(run_tag, n_games=10000, q_thresholds=None, p_thresholds=None,
         out_path=None):
    if q_thresholds is None:
        q_thresholds = [0.3, 0.5, 0.7]
    if p_thresholds is None:
        p_thresholds = [0.3, 0.5, 0.7]

    run_dir = os.path.join(SP_DIR, run_tag)
    analysis_path = os.path.join(run_dir, "analyze_results_combined.pkl")
    game_logs_dir = os.path.join(run_dir, "game_logs")
    out_path = out_path or os.path.join(run_dir, "threshold_sweep.csv")

    t0 = time.time()
    say("Loading analyze_results_combined.pkl...")
    with open(analysis_path, "rb") as f:
        analysis = pickle.load(f)
    df_sf = analysis['df_all']

    # Index by game_id for fast lookup
    sf_by_game = {gid: grp for gid, grp in df_sf.groupby('game_id')}
    say(f"Loaded {len(sf_by_game)} unique games from SF analysis "
        f"({time.time() - t0:.0f}s)")

    game_files = sorted(glob.glob(os.path.join(game_logs_dir, "*_log.pkl.gz")))

    sample_size = min(n_games, len(game_files))
    sample_files = random.sample(game_files, sample_size)
    say(f"Sampling {sample_size} of {len(game_files)} games from logs\n")

    # Collect data matching game_id and move_num
    data_records = []
    t_loop = time.time()

    for i, game_file in enumerate(sample_files):
        if (i + 1) % 250 == 0:
            done = i + 1
            el = time.time() - t_loop
            eta = el / done * (sample_size - done)
            say(f"  {done}/{sample_size}  {done/el:.1f} games/s  "
                f"{len(data_records):,} positions  "
                f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m")

        try:
            with gzip.open(game_file, "rb") as f:
                game = pickle.load(f)
        except Exception:
            continue

        game_id = game.get("game_id")
        if game_id not in sf_by_game:
            continue

        sf_rows = sf_by_game[game_id]
        tree_data = game.get("tree_search_data", {})

        for ply_str, ply_data in tree_data.items():
            ply = int(ply_str)

            # Get model data
            wdl = ply_data.get("best_wdl")
            q_stm = ply_data.get("Q_stm")

            if wdl is None or q_stm is None:
                continue

            total = sum(wdl)
            if total == 0:
                continue

            p_draw = wdl[1] / total
            q = q_stm

            # Get SF data (move_num in SF df is 0-indexed ply)
            sf_row = sf_rows[sf_rows['move_num'] == ply]
            if sf_row.empty:
                continue

            sf_row = sf_row.iloc[0]
            cpl = sf_row['best_cp'] - sf_row['played_cp']
            ce = sf_row['ce']
            kl = sf_row['kl']

            data_records.append({
                'abs_q': abs(q),
                'p_draw': p_draw,
                'cpl': cpl,
                'ce': ce,
                'kl': kl,
            })

    say(f"\nCollected {len(data_records):,} matched positions "
        f"in {(time.time() - t_loop)/60:.1f}m")

    df = pd.DataFrame(data_records)

    say("\nCalculating metrics by threshold...\n")

    rows = []
    for q_thresh in q_thresholds:
        for p_thresh in p_thresholds:
            subset = df[(df['abs_q'] < q_thresh) & (df['p_draw'] < p_thresh)]
            rows.append({
                'abs_q_lt': q_thresh,
                'p_draw_lt': p_thresh,
                'n': len(subset),
                'pct_of_all': 100.0 * len(subset) / len(df) if len(df) else np.nan,
                'cpl': subset['cpl'].mean() if len(subset) else np.nan,
                'ce': subset['ce'].mean() if len(subset) else np.nan,
                'kl': subset['kl'].mean() if len(subset) else np.nan,
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)

    lookup = {(r['abs_q_lt'], r['p_draw_lt']): r for r in rows}
    grid = []
    for q_thresh in q_thresholds:
        line = [f"|Q| < {q_thresh}"]
        for p_thresh in p_thresholds:
            r = lookup[(q_thresh, p_thresh)]
            if not r['n']:
                line.append("-")
                continue
            line.append(
                f"n    {r['n']:>7,}\n"
                f"%    {r['pct_of_all']:>7.1f}\n"
                f"CPL  {r['cpl']:>7.1f}\n"
                f"CE   {r['ce']:>7.3f}\n"
                f"KL   {r['kl']:>7.3f}")
        grid.append(line)

    headers = [""] + [f"p_draw < {p}" for p in p_thresholds]
    say(tabulate(grid, headers=headers, tablefmt='grid'))

    say(f"\nbaseline (all {len(df):,} positions): "
        f"CPL {df.cpl.mean():.1f}  CE {df.ce.mean():.3f}  KL {df.kl.mean():.3f}")
    say(f"wrote {out_path}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_tag", required=True, help="Run tag (e.g. 18m_10c6t_SWA_selfplay2)")
    parser.add_argument("--n_games", type=int, default=10000,
                        help="Number of games to sample")
    parser.add_argument("--out", default=None,
                        help="CSV path (default <run_dir>/threshold_sweep.csv)")
    args = parser.parse_args()

    main(args.run_tag, args.n_games, out_path=args.out)
