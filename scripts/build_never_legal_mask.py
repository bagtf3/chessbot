"""
build_never_legal_mask.py

Tests the sometimes-legal mask produced by pyfastchess.build_sometimes_legal_mask()
against real positions from legal_move_mask(), geometry spot-checks, and a Python
reference implementation to verify C++ and Python agree bit-for-bit.

Usage:
  python scripts/build_never_legal_mask.py

Index structure (from pyfastchess/src/backend.cpp):

  Square encoding:  sq = file + rank * 8   (a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63)
  STM-POV flip:     if black to move: sq_stm = sq_raw ^ 56

  Main region [0, 4096):
    flat = from_sq_stm * 64 + to_sq_stm

  Underpromotion region [4096, 4288):
    flat = 4096 + from_file + 8 * to_file + 64 * promo_type
    promo_type: 0=N  1=B  2=R
"""

import numpy as np
from collections import deque
import pyfastchess
from pyfastchess import Board
from chessbot.utils import random_init, make_random_move

POLICY_SIZE = 4288
MAIN_SIZE   = 4096
PROMO_SIZE  = 192


# ---------------------------------------------------------------------------
# Square helpers
# ---------------------------------------------------------------------------

def sq(name):
    return "abcdefgh".index(name[0]) + (int(name[1]) - 1) * 8


# ---------------------------------------------------------------------------
# Python reference implementation (used only for cross-check vs C++)
# ---------------------------------------------------------------------------

def build_sometimes_legal_mask_py():
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for from_sq in range(64):
        fr, ff = from_sq // 8, from_sq % 8
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            dr  = to_sq // 8 - fr
            df  = to_sq % 8  - ff
            adr = abs(dr)
            adf = abs(df)
            queen  = (dr == 0) or (df == 0) or (adr == adf)
            knight = (adr, adf) in ((1, 2), (2, 1))
            if queen or knight:
                mask[from_sq * 64 + to_sq] = True
    for from_file in range(8):
        for to_file in range(8):
            if abs(to_file - from_file) <= 1:
                for promo_type in range(3):
                    mask[4096 + from_file + 8 * to_file + 64 * promo_type] = True
    return mask


def print_mask_stats(mask):
    n_total = int(mask.sum())
    n_main  = int(mask[:MAIN_SIZE].sum())
    n_promo = int(mask[MAIN_SIZE:].sum())
    print(f"Sometimes-legal: {n_total} / {POLICY_SIZE}  "
          f"(main={n_main}/{MAIN_SIZE}  promo={n_promo}/{PROMO_SIZE})")
    print(f"Never-legal:     {POLICY_SIZE - n_total}")


# ---------------------------------------------------------------------------
# Cross-check: C++ vs Python reference
# ---------------------------------------------------------------------------

def test_cpp_matches_python(mask_cpp):
    print("\n--- C++ vs Python reference cross-check ---")
    mask_py = build_sometimes_legal_mask_py().astype(np.uint8)
    if np.array_equal(mask_cpp, mask_py):
        print("  C++ and Python masks agree on all 4288 indices")
    else:
        diff = np.where(mask_cpp != mask_py)[0]
        print(f"  FAIL: {len(diff)} disagreements at indices: {diff[:20]}")
        assert False, "C++ and Python sometimes-legal masks do not agree"


# ---------------------------------------------------------------------------
# Geometry spot-checks
# ---------------------------------------------------------------------------

def test_geometry(mask):
    print("\n--- main region geometry spot-checks ---")
    cases = [
        ("c1", "d5", False, "dr=4 df=1 - nothing moves like this"),
        ("a1", "b4", False, "dr=3 df=1 - not queen, not knight"),
        ("a1", "d2", False, "dr=1 df=3 - not queen, not knight"),
        ("a1", "h3", False, "dr=2 df=7 - not queen, not knight"),
        ("c3", "f5", False, "dr=2 df=3 - not queen, not knight"),
        ("b2", "d6", False, "dr=4 df=2 - not queen, not knight"),
        ("b2", "e4", False, "dr=2 df=3 - not queen, not knight"),
        ("a1", "c2", True,  "dr=1 df=2 - knight"),
        ("e2", "e4", True,  "rook/pawn double push"),
        ("g1", "f3", True,  "knight (2,1)"),
        ("g1", "h3", True,  "knight (2,1)"),
        ("a1", "h8", True,  "bishop long diagonal"),
        ("d1", "h5", True,  "bishop"),
        ("d4", "d8", True,  "rook"),
        ("e1", "g1", True,  "castling dest / king step"),
        ("a7", "a8", True,  "pawn push / queen promo - main region"),
        ("a7", "b8", True,  "pawn capture / queen promo capture - main region"),
    ]
    errors = 0
    for from_name, to_name, expected, desc in cases:
        f, t = sq(from_name), sq(to_name)
        idx  = f * 64 + t
        got  = bool(mask[idx])
        ok   = "OK  " if got == expected else "FAIL"
        if got != expected:
            errors += 1
        print(f"  {ok}  {from_name}->{to_name}  expected={expected} got={got}  {desc}")
    assert errors == 0, f"{errors} geometry spot-check(s) failed"
    print("  all main-region spot-checks passed")


def test_promo_geometry(mask):
    print("\n--- underpromotion region spot-checks ---")
    cases = [
        (0, 0, 0, True,  "a->a straight, knight"),
        (0, 0, 1, True,  "a->a straight, bishop"),
        (0, 0, 2, True,  "a->a straight, rook"),
        (0, 1, 0, True,  "a->b diagonal right, knight"),
        (7, 6, 2, True,  "h->g diagonal left, rook"),
        (3, 4, 1, True,  "d->e diagonal right, bishop"),
        (3, 3, 0, True,  "d->d straight, knight"),
        (0, 2, 0, False, "a->c - df=2, impossible"),
        (7, 5, 1, False, "h->f - df=2, impossible"),
        (3, 6, 2, False, "d->g - df=3, impossible"),
        (0, 7, 0, False, "a->h - df=7, impossible"),
        (4, 1, 1, False, "e->b - df=3, impossible"),
    ]
    errors = 0
    for ff, tf, pt, expected, desc in cases:
        idx = 4096 + ff + 8 * tf + 64 * pt
        got = bool(mask[idx])
        ok  = "OK  " if got == expected else "FAIL"
        if got != expected:
            errors += 1
        print(f"  {ok}  from_file={ff} to_file={tf} promo={pt}  "
              f"expected={expected} got={got}  {desc}")
    assert errors == 0, f"{errors} underpromo spot-check(s) failed"
    print("  all underpromo spot-checks passed")


# ---------------------------------------------------------------------------
# Position generators
# ---------------------------------------------------------------------------

SEED_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1",
    "rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "8/PPP5/8/8/8/8/8/K6k w - - 0 1",
    "k6K/8/8/8/8/8/ppp5/8 b - - 0 1",
    "r2q1rk1/pp2ppbp/2n2np1/2pp4/3P4/2N1PN2/PPQ1BPPP/R4RK1 b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
]


def collect_bfs_positions(seeds=None, max_positions=4000):
    visited = set()
    queue   = deque()
    for fen in (seeds or [Board().fen()]):
        if fen not in visited:
            visited.add(fen)
            queue.append(fen)
    result = []
    while queue and len(result) < max_positions:
        fen   = queue.popleft()
        result.append(fen)
        b     = Board(fen)
        for m in b.legal_moves():
            b2 = Board(fen)
            b2.push_uci(m)
            f2 = b2.fen()
            if f2 not in visited:
                visited.add(f2)
                queue.append(f2)
    return result


def collect_random_walk_positions(n_walks, ply_min, ply_max, walk_extension=10):
    import random
    fens = []
    for _ in range(n_walks):
        plies = random.randint(ply_min, ply_max)
        b = random_init(plies=plies)
        fens.append(b.fen())
        for _ in range(walk_extension):
            moves = b.legal_moves()
            if not moves:
                break
            make_random_move(b)
            fens.append(b.fen())
    return fens


# ---------------------------------------------------------------------------
# Coverage test
# ---------------------------------------------------------------------------

def test_all_legal_moves_in_mask(mask, positions, label=""):
    print(f"\n--- legal-move coverage{' (' + label + ')' if label else ''} "
          f"({len(positions)} positions) ---")
    violations   = []
    total_moves  = 0
    seen_indices = set()
    for fen in positions:
        b     = Board(fen)
        moves = b.legal_moves()
        if not moves:
            continue
        for uci, idx in zip(moves, b.moves_to_indices(moves)):
            total_moves += 1
            seen_indices.add(idx)
            if not mask[idx]:
                violations.append((fen, uci, int(idx)))
    n_sometimes = int(mask.sum())
    coverage    = 100.0 * len(seen_indices) / n_sometimes
    print(f"  moves checked:  {total_moves:,}")
    print(f"  unique indices: {len(seen_indices):,} / {n_sometimes}  "
          f"({coverage:.1f}% coverage)")
    if violations:
        print(f"  VIOLATIONS: {len(violations)}")
        for fen, uci, idx in violations[:10]:
            print(f"    idx={idx}  uci={uci}  fen={fen[:70]}")
    else:
        print("  no violations")
    assert not violations, f"{len(violations)} legal move(s) mapped to never-legal indices"
    return seen_indices


# ---------------------------------------------------------------------------
# Targeted tests
# ---------------------------------------------------------------------------

def test_stm_symmetry(mask):
    print("\n--- STM-flip symmetry ---")
    b_w = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    b_b = Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    idx_w = list(b_w.moves_to_indices(["e2e4"]))[0]
    idx_b = list(b_b.moves_to_indices(["e7e5"]))[0]
    print(f"  white e2e4 -> {idx_w}  (expected 796)")
    print(f"  black e7e5 -> {idx_b}  (expected 796)")
    assert idx_w == 796, f"white e2e4: expected 796, got {idx_w}"
    assert idx_b == 796, f"black e7e5: expected 796, got {idx_b}"
    assert mask[796]
    print("  STM symmetry OK")


def test_promo_indices(mask):
    print("\n--- promotion index values ---")
    b_w = Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
    moves_w   = b_w.legal_moves()
    indices_w = {m: i for m, i in zip(moves_w, b_w.moves_to_indices(moves_w))}

    b_b = Board("k6K/8/8/8/8/8/p7/8 b - - 0 1")
    moves_b   = b_b.legal_moves()
    indices_b = {m: i for m, i in zip(moves_b, b_b.moves_to_indices(moves_b))}

    expected = {
        "a7a8q": 48 * 64 + 56,
        "a7a8n": 4096 + 0 + 8*0 + 64*0,
        "a7a8b": 4096 + 0 + 8*0 + 64*1,
        "a7a8r": 4096 + 0 + 8*0 + 64*2,
    }
    mirror = {"a2a1q": "a7a8q", "a2a1n": "a7a8n",
              "a2a1b": "a7a8b", "a2a1r": "a7a8r"}

    errors = 0
    for uci, exp in expected.items():
        got = indices_w.get(uci)
        ok  = "OK  " if got == exp else "FAIL"
        if got != exp:
            errors += 1
        print(f"  {ok}  white {uci}: expected={exp} got={got}  in_mask={bool(mask[exp])}")
    for b_uci, w_uci in mirror.items():
        got = indices_b.get(b_uci)
        exp = expected[w_uci]
        ok  = "OK  " if got == exp else "FAIL"
        if got != exp:
            errors += 1
        print(f"  {ok}  black {b_uci}: expected={exp} got={got}  (mirror of {w_uci})")
    assert errors == 0, f"{errors} promotion index check(s) failed"
    print("  promotion index tests passed")


def test_castling_indices(mask):
    print("\n--- castling indices ---")
    b_w = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    b_b = Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    iw  = {m: i for m, i in zip(b_w.legal_moves(), b_w.moves_to_indices(b_w.legal_moves()))}
    ib  = {m: i for m, i in zip(b_b.legal_moves(), b_b.moves_to_indices(b_b.legal_moves()))}
    cases = [
        ("e1g1", iw, 4*64+6, "white KS castle"),
        ("e1c1", iw, 4*64+2, "white QS castle"),
        ("e8g8", ib, 4*64+6, "black KS castle (STM-flipped = white KS)"),
        ("e8c8", ib, 4*64+2, "black QS castle (STM-flipped = white QS)"),
    ]
    errors = 0
    for uci, idx_map, exp, desc in cases:
        got = idx_map.get(uci)
        ok  = "OK  " if got == exp else "FAIL"
        if got != exp:
            errors += 1
        print(f"  {ok}  {uci}: expected={exp} got={got}  in_mask={bool(mask[exp])}  {desc}")
    assert errors == 0, f"{errors} castling index check(s) failed"
    print("  castling index tests passed")


def test_en_passant(mask):
    print("\n--- en passant ---")
    b = Board("rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1")
    moves   = b.legal_moves()
    indices = {m: i for m, i in zip(moves, b.moves_to_indices(moves))}
    ep_idx  = indices.get("e5f6")
    assert ep_idx is not None, "e5f6 not in legal moves"
    assert ep_idx == 36*64+45, f"e5f6: expected {36*64+45}, got {ep_idx}"
    assert mask[ep_idx]
    print(f"  white e5f6 (e.p.) -> idx={ep_idx}  in_mask=True  OK")


def test_no_never_legal_ever_seen(mask, seen_indices):
    print("\n--- never-legal contamination check ---")
    legal_set = set(int(i) for i in np.where(mask)[0])
    bad = seen_indices - legal_set
    if bad:
        print(f"  FAIL: {len(bad)} never-legal indices appeared in real positions!")
        for idx in sorted(bad)[:10]:
            print(f"    idx={idx}")
        assert False, "never-legal contamination"
    print("  clean - 0 never-legal indices seen across all tested positions")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading sometimes-legal mask from pyfastchess.build_sometimes_legal_mask()...")
    mask = pyfastchess.build_sometimes_legal_mask().astype(bool)
    print_mask_stats(mask)
    assert mask.sum() == 1858, f"Expected 1858 sometimes-legal indices, got {mask.sum()}"

    test_cpp_matches_python(mask.astype(np.uint8))
    test_geometry(mask)
    test_promo_geometry(mask)

    print("\nCollecting BFS positions from seed FENs...")
    bfs_positions = collect_bfs_positions(seeds=SEED_FENS, max_positions=4000)
    print(f"  collected {len(bfs_positions)} positions")
    seen = test_all_legal_moves_in_mask(mask, bfs_positions, label="BFS")

    tiers = [
        ("early  ply 5-15",  2000,  5,  15, 10),
        ("middle ply 20-40", 2000, 20,  40, 15),
        ("late   ply 40-80", 2000, 40,  80, 20),
        ("deep   ply 80-150",1000, 80, 150, 20),
    ]
    for label, n_walks, pmin, pmax, ext in tiers:
        print(f"\nGenerating random walk positions ({label}, {n_walks} walks x ~{ext} steps)...")
        fens = collect_random_walk_positions(n_walks, pmin, pmax, walk_extension=ext)
        print(f"  generated {len(fens)} positions")
        new  = test_all_legal_moves_in_mask(mask, fens, label=label)
        prev = len(seen)
        seen |= new
        print(f"  cumulative unique indices: {len(seen)} / {int(mask.sum())}  "
              f"(+{len(seen) - prev} new)")

    test_stm_symmetry(mask)
    test_promo_indices(mask)
    test_castling_indices(mask)
    test_en_passant(mask)
    test_no_never_legal_ever_seen(mask, seen)

    print("\n=== ALL TESTS PASSED ===")
    print_mask_stats(mask)
    n = int(mask.sum())
    print(f"Final coverage: {len(seen)} / {n}  ({100.0*len(seen)/n:.1f}%)")


if __name__ == "__main__":
    main()
