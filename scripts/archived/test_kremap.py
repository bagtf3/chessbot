"""
kRemap validation tests (Python-side, no C++ changes required yet).

Tests from 1858CONVERSION.txt:
  Test 1 - kRemap table integrity
  Test 5 - perft legal move coverage across diverse positions
  Test 6 - underpromo coverage

All tests run against the current 4288-domain moves_to_indices (pre-C++ migration).
kRemap is applied in Python to verify correctness before it is baked into C++.
"""
import sys
import numpy as np
import pyfastchess

SENTINEL = 0xFFFF

def build_kremap(mask):
    """Build kRemap[4288] from the sometimes-legal mask.
    Valid slots get sequential indices 0..1857; never-legal slots get SENTINEL.
    """
    kremap = np.full(4288, SENTINEL, dtype=np.uint32)
    counter = 0
    for i, v in enumerate(mask):
        if v:
            kremap[i] = counter
            counter += 1
    return kremap


# ---------------------------------------------------------------------------
# Test positions: diverse coverage
# ---------------------------------------------------------------------------
POSITIONS = [
    # (label, fen)
    ("startpos",       "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("italian",        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("kiwipete",       "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("endgame_btm",    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1"),
    ("sicilian",       "rnbqkb1r/pp2pppp/3p1n2/2p5/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 5"),
    ("queens_gambit",  "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"),
    ("ruy_lopez",      "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("complex_mid",    "r2q1rk1/pp2ppbp/3p1np1/2pP4/2P1P3/2N2N2/PP3PPP/R1BQR1K1 b - - 0 12"),
    ("check_pos",      "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
    ("ep_pos",         "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
    ("ep_btm",         "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2"),
    ("promo_adj_w",    "8/3P4/8/8/8/8/8/k2K4 w - - 0 1"),
    ("promo_adj_b",    "K2k4/8/8/8/8/8/3p4/8 b - - 0 1"),
    ("underpromo_w",   "8/P7/8/8/8/8/8/k1K5 w - - 0 1"),
    ("underpromo_b",   "K1k5/8/8/8/8/8/p7/8 b - - 0 1"),
    ("promo_diag_w",   "1n6/P7/8/8/8/8/8/k1K5 w - - 0 1"),  # can also capture on b8
    ("castling",       "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ("castling_btm",   "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
    ("dense_mid",      "r1bqr1k1/1pp2ppp/p1np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 9"),
    ("late_endgame",   "8/8/4k3/4p3/4P3/4K3/8/8 w - - 0 1"),
]


def run_test_1(mask, kremap):
    print("Test 1: kRemap table integrity")
    errors = []

    # exactly 1858 valid entries
    valid = (kremap != SENTINEL)
    n_valid = int(valid.sum())
    if n_valid != 1858:
        errors.append(f"  expected 1858 valid entries, got {n_valid}")

    # valid values span exactly [0, 1857] with no gaps
    valid_vals = kremap[valid]
    if set(valid_vals.tolist()) != set(range(1858)):
        missing = set(range(1858)) - set(valid_vals.tolist())
        extra   = set(valid_vals.tolist()) - set(range(1858))
        if missing: errors.append(f"  missing indices: {sorted(missing)[:10]}...")
        if extra:   errors.append(f"  extra indices: {sorted(extra)[:10]}...")

    # no duplicates among valid entries
    if len(valid_vals) != len(set(valid_vals.tolist())):
        errors.append("  duplicate values in valid entries")

    # kRemap[sl_idx[i]] == i for all i
    sl_idx = np.where(mask)[0]
    assert len(sl_idx) == 1858
    for i, pos in enumerate(sl_idx):
        if kremap[pos] != i:
            errors.append(f"  kRemap[sl_idx[{i}]] = {kremap[pos]}, expected {i}")
            if len(errors) > 5:
                errors.append("  ... (truncated)")
                break

    if errors:
        print("  FAIL")
        for e in errors: print(e)
    else:
        print("  PASS")
    return len(errors) == 0


def run_test_5(kremap):
    print("Test 5: perft legal move coverage")
    errors = []
    for label, fen in POSITIONS:
        board = pyfastchess.Board(fen)
        moves = board.legal_moves()
        if not moves:
            continue
        indices_4288 = board.moves_to_indices(moves)
        indices_1858 = [kremap[i] for i in indices_4288]

        # no sentinel hits
        sentinels = [moves[j] for j, v in enumerate(indices_1858) if v == SENTINEL]
        if sentinels:
            errors.append(f"  [{label}] sentinel hit for moves: {sentinels}")

        # no collisions within position
        seen = {}
        for move, idx in zip(moves, indices_1858):
            if idx in seen:
                errors.append(f"  [{label}] collision: {move} and {seen[idx]} both -> {idx}")
            seen[idx] = move

        # all in [0, 1857]
        out_of_range = [v for v in indices_1858 if v != SENTINEL and v >= 1858]
        if out_of_range:
            errors.append(f"  [{label}] indices out of [0,1857]: {out_of_range[:5]}")

    if errors:
        print("  FAIL")
        for e in errors: print(e)
    else:
        print(f"  PASS ({len(POSITIONS)} positions)")
    return len(errors) == 0


def run_test_6(kremap):
    print("Test 6: underpromo coverage")
    errors = []

    underpromo_positions = [
        ("underpromo_w", "8/P7/8/8/8/8/8/k1K5 w - - 0 1",
         {"a7a8q": "queen_promo", "a7a8n": "knight_under", "a7a8b": "bishop_under", "a7a8r": "rook_under"}),
        ("underpromo_b", "K1k5/8/8/8/8/8/p7/8 b - - 0 1",
         {"a2a1q": "queen_promo", "a2a1n": "knight_under", "a2a1b": "bishop_under", "a2a1r": "rook_under"}),
        ("promo_diag_w", "1n6/P7/8/8/8/8/8/k1K5 w - - 0 1",
         {"a7b8q": "diag_queen", "a7b8n": "diag_knight", "a7b8b": "diag_bishop", "a7b8r": "diag_rook",
          "a7a8q": "straight_queen", "a7a8n": "straight_knight"}),
    ]

    for label, fen, expected_moves in underpromo_positions:
        board = pyfastchess.Board(fen)
        legal = set(board.legal_moves())

        present = {m: role for m, role in expected_moves.items() if m in legal}
        missing = {m: role for m, role in expected_moves.items() if m not in legal}
        if missing:
            errors.append(f"  [{label}] expected moves not legal: {missing}")
            continue

        moves_list = list(present.keys())
        idxs_4288 = board.moves_to_indices(moves_list)
        idxs_1858 = [kremap[i] for i in idxs_4288]

        # all valid
        for move, idx in zip(moves_list, idxs_1858):
            role = present[move]
            if idx == SENTINEL:
                errors.append(f"  [{label}] {move} ({role}) mapped to SENTINEL")
            elif idx >= 1858:
                errors.append(f"  [{label}] {move} ({role}) out of range: {idx}")

        # all distinct
        idx_set = set(idxs_1858)
        if len(idx_set) != len(idxs_1858):
            errors.append(f"  [{label}] collision among promo moves: {list(zip(moves_list, idxs_1858))}")

        # queen promo and underpromos land in different regions (sanity)
        # queen promos come from 4096 region (indices < 4096), underpromos from 192 region (>= 4096)
        for move, idx_4288 in zip(moves_list, idxs_4288):
            role = present[move]
            is_under = move[-1] in ('n', 'b', 'r') and len(move) == 5
            in_under_region = idx_4288 >= 4096
            if is_under and not in_under_region:
                errors.append(f"  [{label}] {move} is underpromo but 4288 idx={idx_4288} not in [4096,4287]")
            if not is_under and move[-1] == 'q' and in_under_region:
                errors.append(f"  [{label}] {move} is queen promo but 4288 idx={idx_4288} in underpromo region")

        if not errors:
            for move, idx_1858 in zip(moves_list, idxs_1858):
                role = present[move]
                print(f"    [{label}] {move:8s} ({role:15s}) -> 4288[{idxs_4288[moves_list.index(move)]:4d}] -> 1858[{idx_1858:4d}]")

    if errors:
        print("  FAIL")
        for e in errors: print(e)
    else:
        print("  PASS")
    return len(errors) == 0


def run_bonus_ep_test(kremap):
    """Verify en passant moves map to valid, collision-free 1858 indices."""
    print("Bonus: en passant coverage")
    errors = []
    ep_positions = [
        ("ep_w_f6", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
        ("ep_b_d3", "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2"),
        ("ep_multi", "8/8/8/pPp5/8/8/8/K1k5 w - a6 0 1"),
    ]
    for label, fen in ep_positions:
        board = pyfastchess.Board(fen)
        moves = board.legal_moves()
        ep_sq = board.enpassant_sq()
        ep_moves = [m for m in moves if len(m) == 4 and m[2:4] == ep_sq]
        if not ep_moves:
            continue
        idxs_4288 = board.moves_to_indices(ep_moves)
        idxs_1858 = [kremap[i] for i in idxs_4288]
        for move, idx in zip(ep_moves, idxs_1858):
            if idx == SENTINEL or idx >= 1858:
                errors.append(f"  [{label}] ep move {move} -> bad index {idx}")
        if len(set(idxs_1858)) != len(idxs_1858):
            errors.append(f"  [{label}] ep collision")
        else:
            for move, i4, i8 in zip(ep_moves, idxs_4288, idxs_1858):
                print(f"    [{label}] {move} -> 4288[{i4}] -> 1858[{i8}]")

    if errors:
        print("  FAIL")
        for e in errors: print(e)
    else:
        print("  PASS")
    return len(errors) == 0


if __name__ == "__main__":
    mask = pyfastchess.build_sometimes_legal_mask()
    kremap = build_kremap(mask)

    print(f"mask: {mask.shape}, nonzero: {int(mask.sum())}")
    print(f"kremap: {kremap.shape}, valid entries: {int((kremap != SENTINEL).sum())}")
    print()

    results = []
    results.append(run_test_1(mask, kremap))
    print()
    results.append(run_test_5(kremap))
    print()
    results.append(run_test_6(kremap))
    print()
    results.append(run_bonus_ep_test(kremap))
    print()

    if all(results):
        print("ALL TESTS PASSED")
    else:
        n_fail = results.count(False)
        print(f"{n_fail} TEST(S) FAILED")
        sys.exit(1)
