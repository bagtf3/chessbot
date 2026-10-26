"""
Tests 2, 3, 5 against the new convert1858 C++ build.
Run with default sys.path to get the new build from dist_1858.
"""
import sys
sys.path.insert(0, r"C:\Users\Bryan\repos\pyfastchess_1858\dist_1858")
import pfc1858 as pfc
import numpy as np

SENTINEL = 0xFFFF

# Python kRemap for cross-check
mask = pfc.build_sometimes_legal_mask()
kremap = np.full(4288, SENTINEL, dtype=np.uint32)
counter = 0
for i, v in enumerate(mask):
    if v:
        kremap[i] = counter
        counter += 1

POSITIONS = [
    ("startpos",     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete",     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("endgame_btm",  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1"),
    ("ep_pos",       "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
    ("underpromo_w", "8/P7/8/8/8/8/8/k1K5 w - - 0 1"),
    ("underpromo_b", "K1k5/8/8/8/8/8/p7/8 b - - 0 1"),
    ("castling",     "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ("castling_btm", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
    ("italian",      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("promo_diag_w", "1n6/P7/8/8/8/8/8/k1K5 w - - 0 1"),
]

print("=== Test 2: legal_move_mask bits land in [0, 1857] ===")
all_ok = True
for label, fen in POSITIONS:
    b = pfc.Board(fen)
    mask = np.array(b.legal_move_mask(), dtype=np.uint8)  # length 4288
    moves = b.legal_moves()
    n_legal = len(moves)
    bits_set   = int(mask.sum())
    bits_above = int(mask[1858:].sum())
    ok = bits_set == n_legal and bits_above == 0
    all_ok = all_ok and ok
    print(f"  [{label}] legal={n_legal}, mask_bits={bits_set}, above_1857={bits_above}, {'OK' if ok else 'FAIL'}")

print()
print("=== Test 3: moves_to_indices consistent with legal_move_mask ===")
for label, fen in POSITIONS:
    b = pfc.Board(fen)
    moves  = b.legal_moves()
    idxs   = b.moves_to_indices(moves)
    mask   = np.array(b.legal_move_mask(), dtype=np.uint8)
    # every index from moves_to_indices should be set in the mask
    mask_hits  = [mask[i] == 1 for i in idxs]
    # mask nonzero count should equal moves count
    count_ok   = mask.sum() == len(moves)
    # all in range
    range_ok   = all(0 <= i < 1858 for i in idxs)
    ok = all(mask_hits) and count_ok and range_ok
    all_ok = all_ok and ok
    print(f"  [{label}] {'OK' if ok else 'FAIL'}", end="")
    if not all(mask_hits): print(f" mask_miss={[moves[i] for i,h in enumerate(mask_hits) if not h]}", end="")
    if not count_ok: print(f" count_mismatch mask={mask.sum()} moves={len(moves)}", end="")
    print()

print()
print("=== Test 5: all legal moves map to valid, collision-free 1858 indices ===")
for label, fen in POSITIONS:
    b = pfc.Board(fen)
    moves = b.legal_moves()
    if not moves:
        continue
    idxs = b.moves_to_indices(moves)
    in_range = all(0 <= v < 1858 for v in idxs)
    no_coll  = len(set(idxs)) == len(idxs)
    ok = in_range and no_coll
    all_ok = all_ok and ok
    print(f"  [{label}] {len(moves)} moves, range=[{min(idxs)},{max(idxs)}], {'OK' if ok else 'FAIL'}")

print()
# spot-check underpromo encoding
print("=== Spot check: underpromo indices ===")
b = pfc.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
moves = b.legal_moves()
idxs  = b.moves_to_indices(moves)
for m, idx in sorted(zip(moves, idxs)):
    print(f"  {m} -> {idx}")

print()
print("ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
