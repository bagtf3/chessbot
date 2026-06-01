"""
build_quota_opening_book.py

Quota-driven opening book generator for Xerces.

Opens from `pre_opened_uci_paths_upto2.pkl` as seed prefixes, then grows
each 2-ply prefix into 4-8 ply paths using Stockfish MultiPV, with
controlled sampling so the final distribution matches FIRST_MOVE_WEIGHTS
and REPLY_WEIGHTS rather than descendant count.

4 worker threads run in parallel, each owning one prefix at a time with
its own SF instance and local node/child counts. No shared state.

Output: pre_opened_uci_paths_quota_4to8.pkl
"""

import os
import math
import random
import pickle
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import chess
import chess.engine

load_dotenv()

SF_LOC = os.getenv("SF_LOC")
UPTO2_PATH = os.getenv(
    "PRE_OPENED_UCI_PATHS_MINI",
    "C:/Users/Bryan/Data/chessbot_data/pre_opened_uci_paths_upto2.pkl",
)
OUT_PATH = (
    "C:/Users/Bryan/Data/chessbot_data/pre_opened_uci_paths_quota_4to8.pkl"
)

TOTAL_PATHS = 6000
N_WORKERS   = 4

MULTIPV   = 8
SF_DEPTH  = 12
MAX_CPL   = 80
TEMP_CP   = 45
CHILD_CAP     = 0.35
UNIF_MIX      = 0.10
POL_EXP       = 0.75
MAX_ENDING_CP = 120

DEPTH_CHOICES = [4, 5, 6, 7, 8]
DEPTH_WEIGHTS = [0.15, 0.20, 0.25, 0.20, 0.20]

FIRST_MOVE_WEIGHTS = {
    "e2e4": 0.24,
    "d2d4": 0.24,
    "c2c4": 0.17,
    "g1f3": 0.13,
    "g2g3": 0.07,
    "b1c3": 0.04,
    "b2b3": 0.03,
    "e2e3": 0.025,
    "f2f4": 0.02,
    "d2d3": 0.015,
    "c2c3": 0.01,
}

REPLY_WEIGHTS = {
    "e2e4": {
        "c7c5": 0.18, "e7e5": 0.12, "e7e6": 0.14, "c7c6": 0.14,
        "d7d5": 0.12, "d7d6": 0.12, "g7g6": 0.08, "g8f6": 0.07,
        "b8c6": 0.015, "b7b6": 0.015,
    },
    "d2d4": {
        "g8f6": 0.26, "d7d5": 0.23, "e7e6": 0.13, "g7g6": 0.11,
        "c7c5": 0.10, "c7c6": 0.07, "f7f5": 0.06, "d7d6": 0.02,
        "b7b6": 0.01, "e7e5": 0.01,
    },
    "c2c4": {
        "e7e5": 0.24, "c7c5": 0.20, "g8f6": 0.20, "e7e6": 0.13,
        "g7g6": 0.10, "c7c6": 0.04, "d7d5": 0.04, "d7d6": 0.03,
        "b7b6": 0.02,
    },
    "g1f3": {
        "d7d5": 0.28, "g8f6": 0.24, "c7c5": 0.14, "e7e6": 0.12,
        "g7g6": 0.09, "c7c6": 0.05, "d7d6": 0.04, "f7f5": 0.02,
        "b7b6": 0.02,
    },
    "g2g3": {
        "d7d5": 0.24, "e7e5": 0.22, "g8f6": 0.18, "c7c5": 0.12,
        "g7g6": 0.09, "e7e6": 0.07, "c7c6": 0.04, "d7d6": 0.04,
    },
    "f2f4": {
        "d7d5": 0.30, "g8f6": 0.20, "e7e5": 0.18, "g7g6": 0.10,
        "c7c5": 0.08, "e7e6": 0.06, "d7d6": 0.04, "b7b6": 0.02,
        "c7c6": 0.02,
    },
    "c2c3": {
        "d7d5": 0.30, "e7e5": 0.24, "g8f6": 0.16, "c7c5": 0.10,
        "e7e6": 0.08, "g7g6": 0.05, "c7c6": 0.04, "d7d6": 0.03,
    },
    "b1c3": {
        "d7d5": 0.28, "e7e5": 0.24, "g8f6": 0.16, "c7c5": 0.10,
        "e7e6": 0.08, "g7g6": 0.05, "c7c6": 0.04, "d7d6": 0.03,
        "b8c6": 0.02,
    },
    "b2b3": {
        "d7d5": 0.28, "e7e5": 0.22, "g8f6": 0.18, "c7c5": 0.10,
        "e7e6": 0.08, "g7g6": 0.06, "c7c6": 0.04, "d7d6": 0.04,
    },
    "e2e3": {
        "d7d5": 0.32, "g8f6": 0.20, "e7e5": 0.14, "c7c5": 0.10,
        "e7e6": 0.08, "g7g6": 0.06, "c7c6": 0.05, "d7d6": 0.03,
        "f7f5": 0.02,
    },
    "d2d3": {
        "d7d5": 0.24, "e7e5": 0.22, "g8f6": 0.18, "c7c5": 0.12,
        "g7g6": 0.08, "e7e6": 0.07, "c7c6": 0.05, "d7d6": 0.04,
    },
}


def load_paths(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_paths(paths, path):
    with open(path, "wb") as f:
        pickle.dump(paths, f)
    print(f"Saved {len(paths)} paths -> {path}")


def normalize_weights(weights):
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def board_from_path(path):
    board = chess.Board()
    for uci in path:
        board.push(chess.Move.from_uci(uci))
    return board


def is_legal_path(path):
    board = chess.Board()
    for uci in path:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return False
        board.push(move)
    return True


def get_available_prefixes(upto2_paths):
    prefixes = set()
    for path in upto2_paths:
        if len(path) == 2:
            prefixes.add(tuple(path))
    return prefixes


def add_manual_legal_prefixes(prefixes):
    added = []
    for w1, replies in REPLY_WEIGHTS.items():
        for w2 in replies:
            pair = (w1, w2)
            if pair not in prefixes and is_legal_path(list(pair)):
                prefixes.add(pair)
                added.append(pair)
    if added:
        print(f"Added {len(added)} missing legal prefixes: {added}")
    return prefixes


def make_prefix_quotas(total_paths, prefixes):
    avail_first = {p[0] for p in prefixes}
    fw = {k: v for k, v in FIRST_MOVE_WEIGHTS.items() if k in avail_first}
    fw = normalize_weights(fw)

    raw = {}
    for prefix in prefixes:
        w1, w2 = prefix
        if w1 not in REPLY_WEIGHTS:
            continue
        rw = REPLY_WEIGHTS[w1]
        if w2 not in rw:
            continue
        avail_replies = {p[1] for p in prefixes if p[0] == w1}
        rw_avail = {k: v for k, v in rw.items() if k in avail_replies}
        rw_norm = normalize_weights(rw_avail)
        raw[prefix] = fw[w1] * rw_norm[w2]

    total_raw = sum(raw.values())
    quotas = {}
    remainder = 0.0
    assigned = 0
    for prefix, weight in sorted(raw.items()):
        exact = total_paths * weight / total_raw
        floor = int(exact)
        remainder += exact - floor
        quotas[prefix] = floor
        assigned += floor
        if remainder >= 1.0:
            quotas[prefix] += 1
            assigned += 1
            remainder -= 1.0

    shortfall = total_paths - assigned
    if shortfall > 0:
        top = max(raw, key=raw.get)
        quotas[top] = quotas.get(top, 0) + shortfall

    return quotas


def sample_depth():
    return random.choices(DEPTH_CHOICES, weights=DEPTH_WEIGHTS, k=1)[0]


def stockfish_top_moves(board, engine):
    limit = chess.engine.Limit(depth=SF_DEPTH)
    infos = engine.analyse(
        board, limit, multipv=MULTIPV, info=chess.engine.INFO_ALL
    )
    results = []
    for info in infos:
        if "pv" not in info or not info["pv"]:
            continue
        move = info["pv"][0].uci()
        score = info["score"].relative
        cp = score.score(mate_score=3000)
        if cp is None:
            continue
        results.append({"move": move, "cp": cp})
    return results


def filter_candidates(candidates):
    if not candidates:
        return []
    best_cp = candidates[0]["cp"]
    return [c for c in candidates if best_cp - c["cp"] <= MAX_CPL]


def make_candidate_weights(candidates, policy_priors=None):
    best_cp = candidates[0]["cp"]
    sf_weights = []
    for c in candidates:
        cpl = best_cp - c["cp"]
        w = math.exp(-cpl / TEMP_CP)
        if policy_priors is not None:
            prior = policy_priors.get(c["move"], 0.0)
            w *= prior ** POL_EXP
        sf_weights.append(w)

    total = sum(sf_weights)
    n = len(candidates)
    final = []
    for w in sf_weights:
        norm = w / total
        final.append((1 - UNIF_MIX) * norm + UNIF_MIX / n)
    return final


def apply_child_cap(weights, candidates, parent_key, child_counts, node_counts):
    parent_visits = node_counts[parent_key]
    if parent_visits == 0:
        return weights
    out = list(weights)
    for i, c in enumerate(candidates):
        share = child_counts[(parent_key, c["move"])] / parent_visits
        if share > CHILD_CAP:
            out[i] *= 0.05
    total = sum(out)
    if total == 0:
        return weights
    return [w / total for w in out]


def sample_weighted(weights):
    r = random.random()
    cumul = 0.0
    for i, w in enumerate(weights):
        cumul += w
        if r <= cumul:
            return i
    return len(weights) - 1


def grow_path(prefix, engine, node_counts, child_counts, policy_priors=None):
    target_depth = sample_depth()
    path = list(prefix)
    last_cp = 0

    while len(path) < target_depth:
        board = board_from_path(path)
        if board.is_game_over():
            break

        candidates = stockfish_top_moves(board, engine)
        candidates = filter_candidates(candidates)
        if not candidates:
            break

        weights = make_candidate_weights(candidates, policy_priors)
        parent_key = tuple(path)
        weights = apply_child_cap(
            weights, candidates, parent_key, child_counts, node_counts
        )

        idx = sample_weighted(weights)
        chosen = candidates[idx]["move"]
        last_cp = candidates[idx]["cp"]

        node_counts[parent_key] += 1
        child_counts[(parent_key, chosen)] += 1
        path.append(chosen)

    if len(path) < 4:
        return None, 0
    return tuple(path), last_cp


def generate_prefix_paths(prefix, quota, seen_fen, policy_priors=None):
    """
    Worker function: owns its own SF instance and local counters.
    seen_fen is shared across all workers for global FEN dedup.
    """
    engine = chess.engine.SimpleEngine.popen_uci(SF_LOC)
    engine.configure({"Hash": 64, "Threads": 1})

    node_counts  = collections.Counter()
    child_counts = collections.Counter()
    seen         = set()
    attempts     = 0
    max_attempts = quota * 80
    collected    = []

    while len(collected) < quota and attempts < max_attempts:
        attempts += 1
        path, final_cp = grow_path(prefix, engine, node_counts, child_counts,
                                   policy_priors)
        if path is None:
            continue
        seen.add(path)  # block re-traversal regardless of other filters
        if abs(final_cp) > MAX_ENDING_CP:
            continue
        final_epd = board_from_path(path).epd()
        if final_epd in seen_fen:
            continue
        seen_fen.add(final_epd)
        collected.append(path)

    engine.quit()
    return prefix, collected, attempts


def build_book(total_paths, upto2_paths, policy_priors=None):
    prefixes = get_available_prefixes(upto2_paths)
    prefixes = add_manual_legal_prefixes(prefixes)
    quotas   = make_prefix_quotas(total_paths, prefixes)

    print(f"\nPrefix quotas ({len(quotas)} prefixes):")
    for prefix in sorted(quotas, key=lambda p: -quotas[p]):
        print(f"  {list(prefix)}: {quotas[prefix]}")

    work = [(prefix, quota) for prefix, quota in quotas.items() if quota > 0]

    all_paths       = []
    shortfalls      = {}
    seen_fen        = set()
    n_prefixes_done = 0
    n_paths_done    = 0
    n_prefixes_total = len(work)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(
                generate_prefix_paths, prefix, quota, seen_fen, policy_priors
            ): prefix
            for prefix, quota in work
        }
        for future in as_completed(futures):
            prefix, collected, attempts = future.result()
            quota = quotas[prefix]
            if len(collected) < quota:
                shortfalls[prefix] = quota - len(collected)
            all_paths.extend(collected)
            n_prefixes_done += 1
            n_paths_done    += len(collected)
            pct_pre  = 100 * n_prefixes_done / n_prefixes_total
            pct_path = 100 * n_paths_done / total_paths

            len_counts = collections.Counter(len(p) for p in collected)
            n_col = max(len(collected), 1)
            len_str = "  ".join(
                f"{k} {len_counts[k]/n_col*100:.2f}%"
                for k in DEPTH_CHOICES
            )
            prefix_col   = f"{str(list(prefix)):<22}"
            result_col   = f"{len(collected):>4}/{quota:<4} ({attempts:>4} att)"
            progress_col = (f"{n_prefixes_done:>3}/{n_prefixes_total} pre"
                            f" ({pct_pre:>5.1f}%)"
                            f"  {n_paths_done:>5}/{total_paths} paths"
                            f" ({pct_path:>5.1f}%)")
            print(f"  {prefix_col}  {result_col}  |  {progress_col}"
                  f"  |  {len_str}")

    return all_paths, shortfalls


def audit_book(paths, shortfalls=None):
    print(f"\n{'='*60}")
    print(f"AUDIT: {len(paths)} total paths")
    print(f"{'='*60}")

    lengths = collections.Counter(len(p) for p in paths)
    print("\nLength distribution:")
    for k in sorted(lengths):
        print(f"  {k} plies: {lengths[k]}")

    total = len(paths)
    first_moves = collections.Counter(p[0] for p in paths)
    print("\nFirst move distribution:")
    for mv, cnt in first_moves.most_common():
        print(f"  {mv}: {cnt}  ({100*cnt/total:.1f}%)")

    prefixes2 = collections.Counter(p[:2] for p in paths)
    print("\nTop 2-ply prefix distribution:")
    for prefix, cnt in prefixes2.most_common(30):
        print(f"  {list(prefix)}: {cnt}  ({100*cnt/total:.1f}%)")

    prefixes3 = collections.Counter(p[:3] for p in paths)
    print("\nTop 20 three-ply prefixes:")
    for prefix, cnt in prefixes3.most_common(20):
        print(f"  {list(prefix)}: {cnt}")

    prefixes4 = collections.Counter(p[:4] for p in paths)
    print("\nTop 20 four-ply prefixes:")
    for prefix, cnt in prefixes4.most_common(20):
        print(f"  {list(prefix)}: {cnt}")

    if shortfalls:
        print("\nQuota shortfalls:")
        for prefix, deficit in shortfalls.items():
            print(f"  {list(prefix)}: -{deficit}")
    else:
        print("\nNo quota shortfalls.")


def main():
    upto2_paths = load_paths(UPTO2_PATH)
    print(f"Loaded {len(upto2_paths)} seed prefixes from {UPTO2_PATH}")

    paths, shortfalls = build_book(TOTAL_PATHS, upto2_paths)

    audit_book(paths, shortfalls)
    save_paths([list(p) for p in paths], OUT_PATH)


if __name__ == "__main__":
    main()
