"""
Build a pre-populated SFCache from the position-frequency list produced by
get_position_frequencies.py.  Runs Stockfish at --depth (default 20) with
multipv=5 on every position whose count >= --min_count, using N SF worker
threads.  Saves a gzipped pickle that SFCache.load_seed() can merge at startup.

Usage:
    python seed_sf_cache.py \
        --freq_pkl pos_freq.pkl \
        --output "C:/Users/Bryan/Data/chessbot_data/opening_books/SFCache_prepop_depth20.pkl.gz" \
        --min_count 2 \
        --depth 20 \
        --threads 4
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from queue import Queue, Empty

import chess
import chess.engine
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
SF_LOC = os.getenv("SF_LOC", "")
SP_DIR = os.getenv("SP_DIR", "")

MATE_CP = 2500
CLIP_MAX = 1200
SEED_LAST_SEEN = 10 ** 9


def score_cp_stm(pov_score):
    s = pov_score.relative.score(mate_score=MATE_CP)
    return int(np.clip(s, -CLIP_MAX, CLIP_MAX))


def score_cp_white(pov_score):
    s = pov_score.white().score(mate_score=MATE_CP)
    return int(s)   # unclipped, matches rescore usage


def short_fen(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def load_game(path: str) -> dict | None:
    try:
        if path.endswith(".pkl.gz"):
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
    return None


def find_game_file(run_tag: str, game_id: str) -> str | None:
    game_dir = os.path.join(SP_DIR, run_tag, "game_logs")
    for ext in ("_log.pkl.gz", "_log.pkl", "_log.json", ".pkl.gz", ".pkl"):
        p = os.path.join(game_dir, game_id + ext)
        if os.path.exists(p):
            return p
    # fallback: glob for game_id prefix
    if os.path.isdir(game_dir):
        for name in os.listdir(game_dir):
            if game_id in name:
                return os.path.join(game_dir, name)
    return None


def reconstruct_board(game_data: dict, target_ply: int):
    """
    Returns (chess.Board at target_ply, cache_key) where
    cache_key = (short_fen, reps, halfmoves) matching the live rescorer logic.
    """
    start_fen = game_data.get("start_fen", chess.STARTING_FEN)
    moves = game_data.get("moves_played", [])

    board = chess.Board(start_fen)
    rep_counter = defaultdict(int)
    rep_counter[short_fen(board)] += 1

    for i in range(min(target_ply, len(moves))):
        board.push_uci(moves[i])
        rep_counter[short_fen(board)] += 1

    sf = short_fen(board)
    hmc = board.halfmove_clock
    halfmoves = 0 if hmc < 45 else hmc
    reps = 3 if rep_counter[sf] >= 3 else 0
    cache_key = (sf, reps, halfmoves)
    return board, cache_key


# ---------------------------------------------------------------------------
# SF worker thread
# ---------------------------------------------------------------------------

class SFWorker:
    def __init__(self, req_q: Queue, res_q: Queue, depth: int):
        self.req_q = req_q
        self.res_q = res_q
        self.depth = depth
        self.t = None

    def start(self):
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def join(self):
        if self.t:
            self.t.join()

    def run(self):
        eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
        eng.configure({"Hash": 256, "UCI_ShowWDL": True, "Threads": 1})
        limit = chess.engine.Limit(depth=self.depth)
        while True:
            item = self.req_q.get()
            if item is None:
                eng.quit()
                self.res_q.put(None)
                return
            cache_key, fen = item
            try:
                board = chess.Board(fen)
                info = eng.analyse(board, limit, info=chess.engine.INFO_ALL)
                self.res_q.put((cache_key, info, None))
            except Exception as e:
                self.res_q.put((cache_key, None, e))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq_pkl", default="pos_freq.pkl")
    parser.add_argument(
        "--output",
        default=r"C:\Users\Bryan\Data\chessbot_data\opening_books\SFCache_prepop_depth20.pkl.gz",
    )
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)

    args = parser.parse_args()

    if not SF_LOC:
        print("SF_LOC not set in environment / .env")
        sys.exit(1)

    with open(args.freq_pkl, "rb") as f:
        freq_list = pickle.load(f)

    candidates = [e for e in freq_list if e["count"] >= args.min_count]
    print(f"positions with count >= {args.min_count}: {len(candidates):,}")

    # load existing output to support resume
    cache_data: dict = {}
    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        try:
            with gzip.open(out_path, "rb") as f:
                saved = pickle.load(f)
            cache_data = saved.get("data", {})
            print(f"resuming: {len(cache_data):,} positions already cached")
        except Exception as e:
            print(f"[warn] could not load existing output: {e}")

    from itertools import groupby

    candidates.sort(key=lambda e: (e["run_tag"], e["game_id"]))

    req_q: Queue = Queue(maxsize=args.threads * 4)
    res_q: Queue = Queue()

    workers = [SFWorker(req_q, res_q, args.depth) for _ in range(args.threads)]
    for w in workers:
        w.start()

    # Reconstruct and submit in a background thread so SF workers start immediately.
    # Game files are loaded one at a time (grouped by game_id) and freed after use.
    submitted_count = [0]
    missing_files = [0]

    def submit_all():
        for (run_tag, game_id), group in groupby(
            candidates, key=lambda e: (e["run_tag"], e["game_id"])
        ):
            path = find_game_file(run_tag, game_id)
            entries = list(group)
            if path is None:
                missing_files[0] += len(entries)
                continue
            game_data = load_game(path)
            if game_data is None:
                missing_files[0] += len(entries)
                continue
            for entry in entries:
                try:
                    board, cache_key = reconstruct_board(game_data, entry["ply"])
                except Exception as e:
                    print(f"[warn] reconstruct failed for {game_id} ply {entry['ply']}: {e}")
                    continue
                if cache_key in cache_data:
                    continue
                req_q.put((cache_key, board.fen()))
                submitted_count[0] += 1
        for _ in workers:
            req_q.put(None)

    submit_thread = threading.Thread(target=submit_all, daemon=True)
    submit_thread.start()
    print(f"workers started, reconstructing and submitting positions...")

    done = 0
    errors = 0
    t_start = time.time()
    save_every = 500
    workers_done = 0

    while workers_done < len(workers):
        item = res_q.get()
        if item is None:
            workers_done += 1
            continue

        cache_key, info, err = item
        if err is not None:
            errors += 1
            continue
        pv = info.get("pv", []) if info else []
        if not pv:
            continue

        wdl_obj = info.get("wdl")
        wdl_val = (
            (wdl_obj.relative.wins - wdl_obj.relative.losses) / 1000.0
            if wdl_obj is not None else None
        )
        cache_data[cache_key] = {
            "best": (str(pv[0]), score_cp_stm(info["score"]),
                     score_cp_white(info["score"]), wdl_val),
            "others": {},
            "hits": 0,
            "last_seen": SEED_LAST_SEEN,
        }
        done += 1

        if done % 100 == 0:
            elapsed = time.time() - t_start
            rate = done / elapsed
            sub = submitted_count[0]
            eta_min = (sub - done) / rate / 60 if rate > 0 and sub > done else 0
            print(f"  {done:>6,}/{sub:,}  {rate:.1f} pos/s  ETA {eta_min:.0f}m  "
                  f"errors {errors}")

        if done % save_every == 0:
            save_cache(cache_data, out_path, args.depth)

    submit_thread.join()
    for w in workers:
        w.join()

    elapsed = time.time() - t_start
    print(f"\ndone: {done:,} positions in {elapsed/60:.1f}m  errors={errors}")
    save_cache(cache_data, out_path, args.depth)


def save_cache(cache_data: dict, path: str, depth: int):
    payload = {
        "data": cache_data,
        "depth": depth,
        "n_positions": len(cache_data),
        "saved_at": time.time(),
    }
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)
    print(f"  saved {len(cache_data):,} entries -> {path}")


if __name__ == "__main__":
    main()
