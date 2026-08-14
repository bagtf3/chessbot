"""
Scan analyzed validation games for positions worth re-testing.

Pulls plies where xc0 made a real error in a game it lost or drew, and stores
enough to rebuild the position from STARTPOS plus the original search snapshot.
The deep SF pass that freezes the answer key runs later, on the pooled output.

Classes (V = best_cp, P = played_cp, both STM-POV; cpl = V - P):
  A  |V| <= 200 and cpl >= 60      tight position, real error
  B  V >= 120 and P < 50           winning chances thrown away
  F  V >= 1200 and P < 1000        mate available, not found

Class is not stored: V, P and cpl are, so the rules can be rerun or redefined
against the pool without rescanning.
"""

import argparse
import glob
import gzip
import hashlib
import os
import pickle

import chess
import numpy as np
import pandas as pd

from chessbot.review import load_game_index
from chessbot.validation import load_probe_pool, save_probe_pool

BASE = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs"
RUN_PATTERNS = ["18m_*", "16m_xc0hK6_*"]

A_ABS_V = 200
A_MIN_CPL = 60
B_MIN_V = 120
B_MAX_P = 50
F_MIN_V = 1200
F_MAX_P = 1000

PLY_COLS = [
    "move_num", "played_move", "most_visited_move", "best_move",
    "best_cp", "played_cp", "delta", "sims", "stop_reason", "stm",
    "kl", "ce", "game_id",
]

TREE_COLS = [
    "selection_method", "stop_reason", "sims", "avg_depth", "max_depth",
    "children_visited", "total_children", "Q_stm", "best_wdl",
]


def load_ply_frame(run_dir):
    """Per-ply analysis rows. Converted runs have parquet, others the pkl."""
    parquet = os.path.join(run_dir, "df_all.parquet")
    if os.path.exists(parquet):
        return pd.read_parquet(parquet)

    combined = os.path.join(run_dir, "analyze_results_combined.pkl")
    if not os.path.exists(combined):
        return None

    with open(combined, "rb") as f:
        return pickle.load(f)["df_all"]


def candidate_plies(df, run_dir):
    """Validation plies matching A, B or F in games xc0 lost or drew."""
    val = df[df["scenario"] == "paired_validation"]
    if not len(val):
        return None

    # df_all already carries stockfish_color; only the outcome is missing
    index = pd.DataFrame(load_game_index(run_dir))
    merged = val.merge(index[["game_id", "result"]], on="game_id", how="inner")

    xc0_white = ~merged["stockfish_color"].astype(bool)
    merged["xc0_result"] = np.where(
        xc0_white, merged["result"], -merged["result"]
    )
    merged = merged[merged["xc0_result"] <= 0]
    if not len(merged):
        return None

    v = merged["best_cp"]
    p = merged["played_cp"]
    cpl = merged["delta"]

    hit_a = (v.abs() <= A_ABS_V) & (cpl >= A_MIN_CPL)
    hit_b = (v >= B_MIN_V) & (p < B_MAX_P)
    hit_f = (v >= F_MIN_V) & (p < F_MAX_P)

    return merged[hit_a | hit_b | hit_f]


def log_path(run_dir, game_id):
    return os.path.join(run_dir, "game_logs", game_id + "_log.pkl.gz")


def build_records(run_tag, run_dir, hits):
    """Open each source log once and emit one record per candidate ply."""
    records = []
    for game_id, group in hits.groupby("game_id"):
        path = log_path(run_dir, game_id)
        if not os.path.exists(path):
            continue

        with gzip.open(path, "rb") as f:
            log = pickle.load(f)

        history = log["history_uci"]
        played = log["moves_played"]
        tree = log["tree_search_data"]
        offset = len(history) - len(played)

        for row in group.itertuples(index=False):
            ply = int(row.move_num)
            if ply >= len(played):
                continue

            path_uci = list(history[:offset + ply])
            board = chess.Board()
            for uci in path_uci:
                board.push_uci(uci)

            snap = tree[ply] if ply < len(tree) else {}
            rec = {
                "fen": board.fen(),
                "uci_path": path_uci,
                "start_ply": offset + ply,
                "run_tag": run_tag,
                "game_id": game_id,
                "move_num": ply,
                "model_epoch": log.get("model_epoch"),
                "xc0_result": float(row.xc0_result),
                "played_move": row.played_move,
                "most_visited_move": row.most_visited_move,
                "sf_best_move": row.best_move,
                "best_cp": int(row.best_cp),
                "played_cp": int(row.played_cp),
                "cpl": int(row.delta),
                # older runs predate these columns
                "kl": float(getattr(row, "kl", np.nan)),
                "ce": float(getattr(row, "ce", np.nan)),
                "candidate_moves": snap.get("candidate_moves"),
            }
            for col in TREE_COLS:
                rec[col] = snap.get(col)

            records.append(rec)

    return records


def scan_run(run_dir):
    run_tag = os.path.basename(run_dir.rstrip("/\\"))
    df = load_ply_frame(run_dir)
    if df is None:
        print(f"[skip] {run_tag}: no analysis frame")
        return []

    hits = candidate_plies(df, run_dir)
    del df
    if hits is None or not len(hits):
        print(f"[none] {run_tag}: no candidates")
        return []

    records = build_records(run_tag, run_dir, hits)
    print(f"[ok]   {run_tag}: {len(hits)} plies -> {len(records)} records "
          f"({hits['game_id'].nunique()} games)")
    return records


def merge_into_pool(new_records, pool_path):
    """
    Add positions the pool does not already hold.

    On a collision the existing record always wins: it may carry deep_evals,
    deep_depths, probes and a retirement streak earned over previous probes,
    and none of that is recoverable from a rescan. Duplicates inside the new
    batch collapse the same way.
    """
    pool = load_probe_pool(pool_path)
    have = {r["pos_id"] for r in pool}

    added, dupes = [], 0
    for rec in new_records:
        pos_id = hashlib.sha1(rec["fen"].encode()).hexdigest()[:16]
        if pos_id in have:
            dupes += 1
            continue
        rec["pos_id"] = pos_id
        have.add(pos_id)
        added.append(rec)

    save_probe_pool(pool + added, pool_path)
    return len(pool), len(added), dupes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="blunder_positions_raw.pkl.gz")
    ap.add_argument("--append", default=None,
                    help="merge into this existing pool instead of --out")
    ap.add_argument("--patterns", nargs="*", default=RUN_PATTERNS)
    args = ap.parse_args()

    run_dirs = []
    for pat in args.patterns:
        for path in sorted(glob.glob(os.path.join(args.base, pat))):
            if os.path.isdir(path) and path not in run_dirs:
                run_dirs.append(path)

    print(f"scanning {len(run_dirs)} run dirs")

    # when appending, stage to a sidecar and merge once at the end -- the live
    # pool is read by probe workers and must never hold a partial scan
    staging = args.append + ".incoming.pkl.gz" if args.append else args.out

    all_records = []
    for run_dir in run_dirs:
        all_records.extend(scan_run(run_dir))
        # checkpoint after each run: a scan is minutes long and one bad
        # frame should not cost the dirs already done
        with gzip.open(staging, "wb") as f:
            pickle.dump(all_records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\ntotal {len(all_records)} records -> {staging}")

    if args.append:
        before, added, dupes = merge_into_pool(all_records, args.append)
        print(f"[merge] pool {before} + {added} new "
              f"({dupes} already present) -> {before + added}")
        os.remove(staging)

    if all_records:
        df = pd.DataFrame([
            {k: r[k] for k in ("run_tag", "cpl", "best_cp", "played_cp")}
            for r in all_records
        ])
        print(df.groupby("run_tag")["cpl"].agg(["size", "median", "max"]))


if __name__ == "__main__":
    main()
