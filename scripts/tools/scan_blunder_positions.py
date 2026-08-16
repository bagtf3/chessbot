"""
Scan analyzed games for positions worth re-testing.

Pulls plies where a real error preceded a bad outcome for whoever made it, and
stores enough to rebuild the position from STARTPOS plus the original search
snapshot. The deep SF pass that freezes the answer key runs later, on the
pooled output.

Two modes, differing in which games and which outcome gate:

  --mode validation   paired_validation games only, gated on xc0 losing or
                      drawing the game. These never reach training, so the
                      probe measures generalisation.

  --mode selfplay     everything except paired_validation, gated per ply on
                      Z_stm (the result from the mover's perspective) because
                      xc0 plays both sides. These DO pass through training, so
                      the probe measures self-correction through the RL loop.

Classes (V = best_cp, P = played_cp, both STM-POV; cpl = V - P):
  A   |V| <= 200 and cpl >= 60     tight position, real error
  B   V >= 120 and P < 50          winning chances thrown away
  F   V >= 1200 and P < 1000       mate available, not found (validation only)
  G3  V >= 300 and P < 100         clearly winning, let it slip (selfplay only)

In selfplay mode A is restricted to Z_stm == -1: prod SF only reaches d14-15,
so an "error" the mover still converted is more likely measurement noise than
a real one, and A has samples to spare.

Class is not stored: V, P and cpl are, so the rules can be rerun or redefined
against the pool without rescanning.
"""

import argparse
import collections
import glob
import gzip
import os
import pickle
import random
import sys
import time

import chess
import numpy as np
import pandas as pd

from chessbot.review import load_game_index
from chessbot.blunder_replay import (
    BLUNDER_SCENARIOS, is_blunder_candidate, load_probe_pool, save_probe_pool,
    short_fen,
)

# stdout is fully buffered (not line-buffered) when redirected to a file, so
# a background run's progress prints -- including every \r counter below --
# sit invisible until the process exits. write_through makes every print land
# immediately, same effect as `python -u`.
sys.stdout.reconfigure(write_through=True)

BASE = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs"
RUN_PATTERNS = ["18m_*", "16m_xc0hK6_*"]

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


def candidate_plies_selfplay(df, run_dir):
    """
    Selfplay plies matching A/B/G3/F, gated per ply on Z_stm.

    xc0 plays both sides here, so the outcome gate cannot be game-level: it is
    Z_stm, the result from the perspective of whoever is to move (+1 the mover
    went on to win, 0 draw, -1 lost). Same rule function rescore.py's live
    ingestion uses (chessbot.blunder_replay.is_blunder_candidate), so this
    scan and the live hook can never drift apart.
    """
    # only scenarios that push moves from true STARTPOS. piece_odds,
    # piece_training and random_init seed a custom fastboard(fen) directly, so
    # history_uci for those never traces back to STARTPOS and build_records'
    # replay-from-STARTPOS would land on the wrong position.
    val = df[df["scenario"].isin(BLUNDER_SCENARIOS)]
    if not len(val):
        return None

    index = pd.DataFrame(load_game_index(run_dir))[["game_id", "result"]]
    index = index.drop_duplicates("game_id")
    merged = val.merge(index, on="game_id", how="inner")

    # result is white-POV; flip it for the side actually on move
    merged["z_stm"] = np.where(
        merged["stm"].astype(bool), merged["result"], -merged["result"]
    )

    hit, label = is_blunder_candidate(
        merged["best_cp"], merged["played_cp"], merged["delta"],
        merged["z_stm"],
    )
    counts = pd.Series(label[hit]).value_counts()
    print("       " + "  ".join(f"{k} {v:,}" for k, v in counts.items()))
    return merged[hit]


def candidate_plies(df, run_dir):
    """
    Validation plies matching A/B/G3/F in games xc0 lost or drew.

    xc0 has one fixed color for the whole game here, so the game-level
    xc0_result stands in directly for is_blunder_candidate's z_stm -- same
    rule function as the selfplay scan and the live rescore.py hook.
    """
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

    hit, label = is_blunder_candidate(
        merged["best_cp"], merged["played_cp"], merged["delta"],
        merged["xc0_result"],
    )
    counts = pd.Series(label[hit]).value_counts()
    print("       " + "  ".join(f"{k} {v:,}" for k, v in counts.items()))
    return merged[hit]


def log_path(run_dir, game_id):
    return os.path.join(run_dir, "game_logs", game_id + "_log.pkl.gz")


def build_records(run_tag, run_dir, hits):
    """Open each source log once and emit one record per candidate ply."""
    records = []
    groups = hits.groupby("game_id")
    n_games = groups.ngroups
    started = time.time()
    for i, (game_id, group) in enumerate(groups, 1):
        if i % 500 == 0 or i == n_games:
            rate = i / max(time.time() - started, 1e-9)
            print(f"  {run_tag}: {i:,}/{n_games:,} games "
                  f"({rate:.0f} games/s, {len(records):,} records so far)",
                  end="\r")
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
            scenario = getattr(row, "scenario", None)
            if scenario is None:
                # can't skip the rule this record qualified under without
                # knowing the scenario, so it's not usable -- drop it rather
                # than store a record we can't trust
                continue

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
                "scenario": scenario,
                "xc0_result": getattr(row, "xc0_result", np.nan),
                "z_stm": getattr(row, "z_stm", np.nan),
                "played_move": row.played_move,
                "most_visited_move": row.most_visited_move,
                "sf_best_move": row.best_move,
                "best_cp": row.best_cp,
                "played_cp": row.played_cp,
                "cpl": row.delta,
                # older runs predate these columns
                "kl": getattr(row, "kl", np.nan),
                "ce": getattr(row, "ce", np.nan),
                "candidate_moves": snap.get("candidate_moves"),
            }
            for col in TREE_COLS:
                rec[col] = snap.get(col)

            records.append(rec)

    print()
    return records


def scan_run(run_dir, mode="validation"):
    run_tag = os.path.basename(run_dir.rstrip("/\\"))
    df = load_ply_frame(run_dir)
    if df is None:
        print(f"[skip] {run_tag}: no analysis frame")
        return []

    if mode == "selfplay":
        hits = candidate_plies_selfplay(df, run_dir)
    else:
        hits = candidate_plies(df, run_dir)
    del df
    if hits is None or not len(hits):
        print(f"[none] {run_tag}: no candidates")
        return []

    records = build_records(run_tag, run_dir, hits)
    print(f"[ok]   {run_tag}: {len(hits)} plies -> {len(records)} records "
          f"({hits['game_id'].nunique()} games)")
    return records


def merge_into_pool(new_records, pool_path, target_size=None, max_add=None):
    """
    Add positions the pool does not already hold, bounded by a buffer.

    On a collision the existing record always wins: it may carry deep_evals,
    deep_depths, probes and an equiv_streak earned over real probes, none of
    which is recoverable from a rescan -- but the hit still counts, so
    times_seen tracks how often a position keeps resurfacing. Duplicates
    inside the new batch collapse the same way.

    When candidates exceed the room left under target_size (further capped
    by max_add), admission is a random sample of that size, not the
    first-N: scan order groups by run_tag, and sequential admission would
    silently bias which run_tags end up represented.
    """
    pool = load_probe_pool(pool_path)

    fresh = {}
    dupes = 0
    for rec in new_records:
        key = short_fen(rec["fen"])
        if key in pool:
            pool[key]["times_seen"] = pool[key].get("times_seen", 1) + 1
            dupes += 1
            continue
        if key in fresh:
            dupes += 1
            continue
        rec["times_seen"] = 1
        fresh[key] = rec

    keys = list(fresh.keys())
    budget = len(keys)
    if target_size is not None:
        budget = min(budget, max(0, target_size - len(pool)))
    if max_add is not None:
        budget = min(budget, max_add)

    if budget < len(keys):
        keys = random.sample(keys, budget)

    before = len(pool)
    for key in keys:
        pool[key] = fresh[key]

    save_probe_pool(pool, pool_path)
    return before, len(keys), dupes


def report_and_dedupe(records):
    """Keep one record per FEN, and show which positions recurred."""
    seen, out, dupes = {}, [], collections.Counter()
    for rec in records:
        fen = rec["fen"]
        dupes[fen] += 1
        if fen in seen:
            continue
        seen[fen] = rec
        out.append(rec)

    repeats = {f: n for f, n in dupes.items() if n > 1}
    print(f"\nunique fens {len(out):,} of {len(records):,} records; "
          f"{len(repeats):,} fens seen more than once")
    if repeats:
        hist = collections.Counter(repeats.values())
        print("  times seen | how many fens")
        for n in sorted(hist):
            print(f"  {n:>10} | {hist[n]:,}")
        print("\n  most repeated:")
        for fen, n in sorted(repeats.items(), key=lambda kv: -kv[1])[:10]:
            r = seen[fen]
            print(f"  {n:>3}x  cpl {r['cpl']:>4}  {r['played_move']} "
                  f"(best {r['sf_best_move']})  {fen}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="blunder_positions_raw.pkl.gz")
    ap.add_argument("--append", default=None,
                    help="merge into this existing pool instead of --out")
    ap.add_argument("--patterns", nargs="*", default=RUN_PATTERNS)
    ap.add_argument("--mode", choices=("validation", "selfplay"),
                    default="validation")
    ap.add_argument("--target_size", type=int, default=None,
                    help="soft cap on pool size; admission is random-"
                         "sampled down to the room left under this")
    ap.add_argument("--max_add", type=int, default=None,
                    help="hard cap on new positions admitted this run, "
                         "regardless of target_size headroom")
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
        all_records.extend(scan_run(run_dir, args.mode))
        # checkpoint after each run: a scan is minutes long and one bad
        # frame should not cost the dirs already done
        with gzip.open(staging, "wb") as f:
            pickle.dump(all_records, f, protocol=pickle.HIGHEST_PROTOCOL)

    all_records = report_and_dedupe(all_records)

    with gzip.open(staging, "wb") as f:
        pickle.dump(all_records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\ntotal {len(all_records)} records -> {staging}")

    if args.append:
        before, added, dupes = merge_into_pool(
            all_records, args.append,
            target_size=args.target_size, max_add=args.max_add,
        )
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
