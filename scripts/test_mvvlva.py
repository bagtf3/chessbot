# tools/test_mvvlva.py
import time
import random
import chess
from chessbot.utils import random_init

def mvvlva_python(board, move):
    mvvlva = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 105, 104, 103, 102, 101, 100],
        [0, 205, 204, 203, 202, 201, 200],
        [0, 305, 304, 303, 302, 301, 300],
        [0, 405, 404, 403, 402, 401, 400],
        [0, 505, 504, 503, 502, 501, 500],
        [0, 605, 604, 603, 602, 601, 600],
    ]

    from_square = move.from_square
    to_square = move.to_square
    attacker = board.piece_type_at(from_square)
    victim = board.piece_type_at(to_square)

    # En passant
    if victim is None:
        victim = 1
    return mvvlva[victim][attacker]


def py_mvvlva_from_fen(fen, uci):
    """Compute MVV-LVA using python-chess on the given FEN and UCI move.
    Mirrors your original Python logic: empty 'to' square => victim = pawn (1).
    """
    cb = chess.Board(fen)
    mv = chess.Move.from_uci(uci)
    return mvvlva_python(cb, mv)

def test_random_positions(n_positions=200, sample_moves=20):
    # Build the sample list first (so both timings operate on identical inputs)
    checks = []  # list of (fb, fen, uci)
    total_moves = 0

    for i in range(n_positions):
        fb = random_init(random.randint(0, 40))
        fen = fb.fen()
        legal = fb.legal_moves()
        if not legal:
            continue

        if sample_moves is None:
            moves_to_check = list(legal)
        else:
            moves_to_check = random.sample(legal, min(sample_moves, len(legal)))

        for uci in moves_to_check:
            checks.append((fb, fen, uci))
            total_moves += 1

    if not checks:
        print("No moves sampled.")
        return []

    # Optional warm-up (small): run first 10 checks on both implementations
    warmup = min(10, len(checks))
    for i in range(warmup):
        fb, fen, uci = checks[i]
        # python side
        _ = py_mvvlva_from_fen(fen, uci)
        # cpp side
        try:
            _ = fb.mvvlva(uci)
        except Exception:
            pass

    # Time the Python implementation (python-chess on FEN)
    t0 = time.perf_counter()
    py_results = []
    for fb, fen, uci in checks:
        py_results.append(py_mvvlva_from_fen(fen, uci))
    t1 = time.perf_counter()
    total_py = t1 - t0

    # Time the C++ implementation (fastboard.mvvlva)
    t2 = time.perf_counter()
    cpp_results = []
    for fb, fen, uci in checks:
        try:
            cpp_results.append(fb.mvvlva(uci))
        except Exception:
            cpp_results.append(None)
    t3 = time.perf_counter()
    total_cpp = t3 - t2

    # Verify correctness again (optional) and gather mismatch count
    mismatches = []
    for idx, (pyv, cppv) in enumerate(zip(py_results, cpp_results)):
        if pyv != cppv:
            fb, fen, uci = checks[idx]
            mismatches.append((fen, uci, pyv, cppv))

    # Report timing and correctness summary
    n = len(checks)
    print("MVV-LVA speed test")
    print("Positions sampled:", n_positions)
    print("Total moves checked:", n)
    print(f"Python total time: {total_py:.6f}s   -> {total_py*1000.0/n:.6f} ms/move   ({n/total_py:.2f} moves/sec)" if total_py>0 else "Python total time: 0")
    print(f"C++    total time: {total_cpp:.6f}s   -> {total_cpp*1000.0/n:.6f} ms/move   ({n/total_cpp:.2f} moves/sec)" if total_cpp>0 else "C++ total time: 0")
    ratio = (total_py / total_cpp) if (total_cpp > 0) else float("inf")
    print("Speedup (python / c++): {:.2f}x".format(ratio))

    print("Mismatches:", len(mismatches))
    if mismatches:
        print("First 10 mismatches (fen, move, py_score, cpp_score):")
        for item in mismatches[:10]:
            print(item)
    else:
        print("All results matched.")

    return {
        "n": n,
        "total_py": total_py,
        "total_cpp": total_cpp,
        "ms_per_py": (total_py*1000.0/n),
        "ms_per_cpp": (total_cpp*1000.0/n),
        "moves_per_sec_py": (n/total_py) if total_py>0 else float("inf"),
        "moves_per_sec_cpp": (n/total_cpp) if total_cpp>0 else float("inf"),
        "mismatches": mismatches
    }

if __name__ == "__main__":
    # example run
    stats = test_random_positions(n_positions=1000, sample_moves=20)
    from pprint import pprint
    pprint(stats)
