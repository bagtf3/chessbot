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
        --workers 8
"""
import argparse
import gzip
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pyfastchess import Board
from chessbot.review import load_game_index

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
SP_DIR = os.getenv("SP_DIR", "")

REPORT_INTERVAL = 500


def load_game(path: str) -> dict | None:
    try:
        if path.lower().endswith(".pkl.gz"):
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
    return None


def process_game(path: str, run_tag: str, game_id: str, result: float, max_plies: int):
    data = load_game(path)
    if data is None:
        return None

    start_fen = data.get("start_fen")
    moves = data.get("moves_played", [])

    board = Board(start_fen) if start_fen else Board()
    entries = []
    for ply, uci in enumerate(moves):
        if ply >= max_plies:
            break
        entries.append((board.fen(include_counters=False), ply, game_id))
        if not board.push_uci(uci):
            break

    return entries, run_tag, result


def worker_fn(args):
    jobs_chunk, max_plies, progress_queue = args
    freq_map = defaultdict(int)
    wdl_map = {}
    first_seen = {}
    skipped = 0

    for i, (path, tag, gid, res) in enumerate(jobs_chunk):
        result = process_game(path, tag, gid, res, max_plies)
        if result is None:
            skipped += 1
        else:
            entries, run_tag, game_result = result
            w = 1 if game_result > 0 else 0
            d = 1 if game_result == 0 else 0
            lv = 1 if game_result < 0 else 0
            for sf, ply, game_id in entries:
                freq_map[sf] += 1
                if sf in wdl_map:
                    wdl = wdl_map[sf]
                    wdl[0] += w; wdl[1] += d; wdl[2] += lv
                else:
                    wdl_map[sf] = [w, d, lv]
                if sf not in first_seen:
                    first_seen[sf] = (run_tag, game_id, ply)

        if (i + 1) % REPORT_INTERVAL == 0:
            progress_queue.put(REPORT_INTERVAL)

    remainder = len(jobs_chunk) % REPORT_INTERVAL
    if remainder:
        progress_queue.put(remainder)
    progress_queue.put(None)  # sentinel

    return dict(freq_map), wdl_map, first_seen, skipped


def collect_jobs(run_tags: list[str], scenarios: set[str] | None) -> list[tuple]:
    jobs = []
    for tag in run_tags:
        run_dir = os.path.join(SP_DIR, tag)
        index = load_game_index(run_dir)
        before = len(index)

        if scenarios:
            index = [g for g in index if g.get("scenario") in scenarios]
        index = [g for g in index if isinstance(g.get("pkl_file"), str)]
        index = [g for g in index if os.path.exists(g["pkl_file"])]

        print(f"  {tag}: {before} total -> {len(index)} after scenario+file filter")
        for g in index:
            jobs.append((g["pkl_file"], tag, str(g["game_id"]), float(g.get("result", 0))))

    return jobs


def split_chunks(lst, n):
    k, rem = divmod(len(lst), n)
    i = 0
    for j in range(n):
        size = k + (1 if j < rem else 0)
        yield lst[i:i + size]
        i += size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tags", nargs="+", required=True)
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="filter by scenario name(s); omit for all")
    parser.add_argument("--max_plies", type=int, default=16)
    parser.add_argument("--output", default="pos_freq.pkl")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    scenarios = set(args.scenarios) if args.scenarios else None

    jobs = collect_jobs(args.run_tags, scenarios)
    print(f"\n{len(jobs)} games to process across {len(args.run_tags)} run tag(s)")
    if not jobs:
        print("nothing to do")
        return

    n_workers = min(args.workers, len(jobs))
    chunks = list(split_chunks(jobs, n_workers))
    print(f"splitting across {n_workers} worker processes\n")

    manager = mp.Manager()
    progress_queue = manager.Queue()
    worker_args = [(chunk, args.max_plies, progress_queue) for chunk in chunks]

    t_start = time.time()
    last_report = t_start
    games_done = 0
    sentinels = 0

    with mp.Pool(processes=n_workers) as pool:
        async_result = pool.map_async(worker_fn, worker_args)

        while sentinels < n_workers:
            try:
                msg = progress_queue.get(timeout=0.5)
                if msg is None:
                    sentinels += 1
                else:
                    games_done += msg
            except Exception:
                pass

            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - t_start
                rate = games_done / elapsed if elapsed > 0 else 0
                remaining = len(jobs) - games_done
                print(f"  {games_done:,}/{len(jobs):,}  {remaining:,} remaining  "
                      f"{rate:.0f} games/s")
                last_report = now

        results = async_result.get()

    freq_map = defaultdict(int)
    wdl_map = {}
    first_seen = {}
    total_skipped = 0

    for w_freq, w_wdl, w_first, w_skip in results:
        total_skipped += w_skip
        for sf, cnt in w_freq.items():
            freq_map[sf] += cnt
        for sf, wdl in w_wdl.items():
            if sf in wdl_map:
                wdl_map[sf][0] += wdl[0]
                wdl_map[sf][1] += wdl[1]
                wdl_map[sf][2] += wdl[2]
            else:
                wdl_map[sf] = list(wdl)
        for sf, loc in w_first.items():
            if sf not in first_seen:
                first_seen[sf] = loc

    elapsed = time.time() - t_start
    print(f"\n--- summary ---")
    print(f"games processed      : {len(jobs) - total_skipped:,} / {len(jobs):,}")
    print(f"unique positions     : {len(freq_map):,}")
    print(f"total ply-instances  : {sum(freq_map.values()):,}")
    print(f"elapsed              : {elapsed:.1f}s  ({len(jobs)/elapsed:.0f} games/s)")

    sorted_positions = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)

    print(f"\ntop 15 positions by frequency:")
    for sf, count in sorted_positions[:15]:
        tag, gid, ply = first_seen[sf]
        print(f"  {count:>6,}x  ply={ply}  {sf[:50]}  ({tag}/{gid})")

    output = [
        {
            "short_fen": sf,
            "count": count,
            "w": wdl_map[sf][0],
            "d": wdl_map[sf][1],
            "l": wdl_map[sf][2],
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
