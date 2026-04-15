"""
Scan game logs across one or more run_tags and count how often each position
(by short FEN: piece placement + side + castling + ep, no move counters) appears
within the first --max_plies plies.

For each unique short FEN, records the first-seen location (run_tag, game_id, ply)
so that part-2 (SF seeding) can reconstruct the full move history for accurate eval.

Usage:
    python get_position_frequencies.py \
        --run_tags val_test run2 \
        --scenarios startpos pre_opened pre_opened_mini \
        --max_plies 16 \
        --output pos_freq.pkl \
        --threads 8
"""
import argparse
import gzip
import json
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
SP_DIR = os.getenv("SP_DIR", "")


def short_fen(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def load_game(path: str) -> dict | None:
    try:
        p = path.lower()
        if p.endswith(".pkl.gz"):
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        elif p.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif p.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
    return None


def process_game(path: str, run_tag: str, scenarios: set[str] | None, max_plies: int):
    """
    Returns (entries, run_tag) where entries is a list of (short_fen, ply, game_id).
    Returns None on skip or error.
    """
    data = load_game(path)
    if data is None:
        return None

    scenario = data.get("scenario", "")
    if scenarios and scenario not in scenarios:
        return None

    game_id = str(data.get("game_id", ""))
    start_fen = data.get("start_fen", chess.STARTING_FEN)
    moves = data.get("moves_played", [])

    board = chess.Board(start_fen)
    results = []
    for ply, uci in enumerate(moves):
        if ply >= max_plies:
            break
        try:
            sf = short_fen(board)
            results.append((sf, ply, game_id))
            board.push_uci(uci)
        except Exception:
            break

    return results, run_tag


def find_game_files(run_tag: str) -> list[str]:
    run_dir = os.path.join(SP_DIR, run_tag)
    game_dir = os.path.join(run_dir, "game_logs")
    if not os.path.isdir(game_dir):
        print(f"[warn] game_logs not found for run_tag '{run_tag}': {game_dir}")
        return []
    files = []
    for name in os.listdir(game_dir):
        if name.endswith((".pkl", ".pkl.gz", ".json")):
            files.append(os.path.join(game_dir, name))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tags", nargs="+", required=True)
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="filter by scenario name(s); omit for all")
    parser.add_argument("--max_plies", type=int, default=16)
    parser.add_argument("--output", default="pos_freq.pkl")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    scenarios = set(args.scenarios) if args.scenarios else None

    all_files = []
    for tag in args.run_tags:
        files = find_game_files(tag)
        all_files.extend((f, tag) for f in files)

    print(f"found {len(all_files)} game files across {len(args.run_tags)} run tag(s)")
    if not all_files:
        print("nothing to do")
        return

    # freq_map: short_fen -> count
    # first_seen: short_fen -> (run_tag, game_id, ply)
    freq_map = defaultdict(int)
    first_seen = {}

    done = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(process_game, path, tag, scenarios, args.max_plies): (path, tag)
            for path, tag in all_files
        }
        for fut in as_completed(futures):
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(all_files)} files processed, "
                      f"{len(freq_map)} unique positions so far")
            try:
                result = fut.result()
            except Exception as e:
                print(f"[warn] worker error: {e}", file=sys.stderr)
                skipped += 1
                continue

            if result is None:
                skipped += 1
                continue

            entries, run_tag = result
            for sf, ply, game_id in entries:
                freq_map[sf] += 1
                if sf not in first_seen:
                    first_seen[sf] = (run_tag, game_id, ply)

    print(f"\n--- summary ---")
    print(f"game files processed : {done - skipped:,} / {len(all_files):,}")
    print(f"unique positions      : {len(freq_map):,}")
    total_instances = sum(freq_map.values())
    print(f"total ply-instances  : {total_instances:,}")

    sorted_positions = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)

    print(f"\ntop 15 positions by frequency:")
    for sf, count in sorted_positions[:15]:
        tag, gid, ply = first_seen[sf]
        print(f"  {count:>6,}x  ply={ply}  {sf[:50]}  ({tag}/{gid})")

    # build output: list of dicts sorted by count desc
    output = [
        {
            "short_fen": sf,
            "count": count,
            "run_tag": first_seen[sf][0],
            "game_id": first_seen[sf][1],
            "ply": first_seen[sf][2],
        }
        for sf, count in sorted_positions
    ]

    with open(args.output, "wb") as f:
        pickle.dump(output, f)

    print(f"\nsaved {len(output):,} entries to {args.output}")


if __name__ == "__main__":
    main()
