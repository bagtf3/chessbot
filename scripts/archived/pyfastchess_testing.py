#%%
import pyfastchess as pfc
import chess
def check_equal(label, a, b):
    print(f"{label}: {'OK' if a == b else 'MISMATCH'}")
    if a != b:
        print("  got     :", a)
        print("  expected:", b)

# 1) Start from the standard start position
b = pfc.Board()                 # default ctor = standard start
start_fen = b.fen()
print("start FEN:", start_fen)

print("legal moves (first 10):", b.legal_moves()[:10])
n0 = len(b.legal_moves())
print("num legal moves at start:", n0)  # should be 20

# 2) Make a move via UCI
b.push_uci("e2e4")
fen_after = b.fen()
print("FEN after e2e4:", fen_after)

# 3) Unmake last move
b.unmake()
b.fen()
fen_back = b.fen()
check_equal("unmake returns to start FEN", fen_back, start_fen)

# 4) Construct from a specific FEN (not 'startpos' — use actual FEN)
custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"  # after e2e4
b2 = pfc.Board(custom_fen)
print("custom FEN round-trip:", b2.fen())
check_equal("round-trip FEN", b2.fen(), custom_fen)
    
assert not b2.push_uci("e7e5q")

# 6) Make/unmake on b2 and verify
moves = b2.legal_moves()
print("b2 legal moves (first 10):", moves[:10])
b2.push_uci(moves[0])
mid_fen = b2.fen()
b2.unmake()
check_equal("b2 unmake returns to custom FEN", b2.fen(), custom_fen)
#%%
import pickle
fen_path = "C:/Users/Bryan/Data/chessbot_data/fens2000.pkl"
with open(fen_path, "rb") as f:
    fen_list = pickle.load(f)

import time, random, statistics
import pyfastchess as pfc
import chess

# --------- config ----------
K_RANDOM_PER_BOARD = 10   # how many push/unmake pairs per board for the last test
SEED = 1337               # reproducibility
random.seed(SEED)

# Sanity check
assert isinstance(fen_list, list) and len(fen_list) >= 1 and isinstance(fen_list[0], str)

def time_block(fn, label):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label}: {dt:.4f}s")
    return dt, out

# ---------- 1) INIT speed ----------
def init_python_chess():
    for _ in range(3):
        boards = [chess.Board(fen) for fen in fen_list]
    return boards

def init_pyfastchess():
    for _ in range(3):
        boards = [pfc.Board(fen) for fen in fen_list]
    return boards

t_py_chess, boards_py = time_block(init_python_chess, "python-chess init  ")
t_pfc, boards_pfc = time_block(init_pyfastchess,    "pyfastchess init   ")

print(f"per-board init: python-chess {1e6*t_py_chess/len(fen_list):.1f} µs | pyfastchess {1e6*t_pfc/len(fen_list):.1f} µs")
print("-"*60)

# ---------- 2) FEN round-trip ----------
def fen_roundtrip_python_chess():
    # just generate FEN strings; equality to input may differ in spacing but it’s fine for timing
    return [b.fen() for b in boards_py]

def fen_roundtrip_pyfastchess():
    return [b.fen() for b in boards_pfc]

t_py_chess_fen, fens_py = time_block(fen_roundtrip_python_chess, "python-chess fen()")
t_pfc_fen, fens_pfc = time_block(fen_roundtrip_pyfastchess,      "pyfastchess fen()")

print(f"per-board fen:  python-chess {1e6*t_py_chess_fen/len(fen_list):.1f} µs | pyfastchess {1e6*t_pfc_fen/len(fen_list):.1f} µs")
print("-"*60)

# ---------- 3) Legal moves over all 2000 ----------
def legal_moves_python_chess():
    counts = []
    for b in boards_py:
        # Count without materializing a list is slightly faster:
        n = 0
        for _ in b.legal_moves:
            n += 1
        counts.append(n)
    return counts

def legal_moves_pyfastchess():
    counts = []
    for b in boards_pfc:
        # pyfastchess returns list[str] already
        counts.append(len(b.legal_moves()))
    return counts

t_py_chess_legal, counts_py = time_block(legal_moves_python_chess, "python-chess legal moves")
t_pfc_legal, counts_pfc = time_block(legal_moves_pyfastchess,      "pyfastchess legal moves ")

print(f"avg legal moves: python-chess {statistics.mean(counts_py):.2f} | pyfastchess {statistics.mean(counts_pfc):.2f}")
print(f"per-board legal: python-chess {1e6*t_py_chess_legal/len(fen_list):.1f} µs | pyfastchess {1e6*t_pfc_legal/len(fen_list):.1f} µs")
print("-"*60)

# ---------- 4) random push/unmake loops ----------
def rand_push_unmake_python_chess():
    rng = random.Random(SEED)
    # work on copies to avoid polluting earlier boards if you care
    for b in (bb.copy() for bb in boards_py):
        for _ in range(K_RANDOM_PER_BOARD):
            # need a concrete list to choose from
            lm = list(b.legal_moves)
            if not lm:
                break
            m = rng.choice(lm)
            b.push(m)
            b.pop()

def rand_push_unmake_pyfastchess():
    rng = random.Random(SEED)
    for b in boards_pfc:
        for _ in range(K_RANDOM_PER_BOARD):
            lm = b.legal_moves()
            if not lm:
                break
            m = rng.choice(lm)         # UCI string
            ok = b.push_uci(m)
            if not ok:
                print(f"bad move {m}")
                # If ever invalid (shouldn't be), just continue
                continue
            b.unmake()

t_py_chess_mk, _ = time_block(rand_push_unmake_python_chess, "python-chess random push/pop")
t_pfc_mk, _      = time_block(rand_push_unmake_pyfastchess,  "pyfastchess random push/unmake")

iters = len(fen_list) * K_RANDOM_PER_BOARD
print(f"per-iter push/unmake: python-chess {1e6*t_py_chess_mk/max(1,iters):.1f} µs | pyfastchess {1e6*t_pfc_mk/max(1,iters):.1f} µs")
print("-"*60)

# Optional sanity: compare counts for a few positions
mismatches = sum([1 for a,b in zip(counts_py, counts_pfc) if a != b])
print(f"legal-move count mismatches: {mismatches}/{len(fen_list)} (differences can happen with illegal FENs or variant flags)")

#%%
b = pfc.Board()
print("game_over:", b.is_game_over())   # expect ('none','none') in the start position
print("check after e2e4:", b.gives_check("e2e4"))  # should be False


b = pfc.Board()
moves = b.legal_moves()
assert moves, "no legal moves from startpos?"
ok = b.push_uci(moves[0])
assert ok is True
assert b.unmake() is True


b.history_uci()
b.history_size()
b.clear_history()
#%%
import copy
import random
import pyfastchess as pf

def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(msg or f"{a!r} != {b!r}")

# --- 1) startpos basics ---
b = pf.Board()
start_fen_full = b.fen(True)
start_fen_nc = b.fen(False)

print("start fen:", start_fen_full)
assert " w " in start_fen_full
assert_eq(b.side_to_move(), "w")
assert_eq(b.enpassant_sq(), "-")
assert b.castling_rights() in ("KQkq", "-")  # lib specific, but startpos should be KQkq
assert b.in_check() is False
assert b.is_game_over() == ("none", "none")
assert b.history_size() == 0
assert b.legal_moves(), "no legal moves from start?"

# --- 2) push/unmake round-trip ---
first = b.legal_moves()[0]
ok = b.push_uci(first)
assert ok is True
assert b.history_size() == 1
after_push = b.fen(True)
assert after_push != start_fen_full

ok = b.unmake()
assert ok is True
assert b.history_size() == 0
after_unmake = b.fen(True)
assert_eq(after_unmake, start_fen_full, "FEN mismatch after unmake")

# --- 3) history info works & clear_history ---
# play two moves
seq = random.sample(pf.Board().legal_moves(), 2)
for mv in seq:
    assert b.push_uci(mv)

hist = b.history_uci()
assert_eq(len(hist), 2)
assert hist == seq, f"history_uci mismatch: {hist} vs {seq}"

b.clear_history()
assert_eq(b.history_size(), 0)

# --- 4) copy/clone behave, copy is independent ---
b2 = pf.Board("rnbqkbnr/pppppppp/8/8/4Q3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 2")  # a legal random-ish FEN
b2_copyctor = pf.Board(b2)
b2_clone    = b2.clone()
b2_shallow  = copy.copy(b2)
b2_deep     = copy.deepcopy(b2)

for c in (b2_copyctor, b2_clone, b2_shallow, b2_deep):
    assert_eq(c.fen(True), b2.fen(True))

# mutate original, copies should NOT change
orig_fen = b2.fen(True)
mvs = b2.legal_moves()
if mvs:
    b2.push_uci(mvs[0])

for name, c in [("copyctor", b2_copyctor), ("clone", b2_clone), ("copy", b2_shallow), ("deepcopy", b2_deep)]:
    assert_eq(c.fen(True), orig_fen), f"{name} changed when original mutated"

# --- 5) gives_check: construct a simple position where a move clearly gives check ---
# FEN: White Q on e2, Black king on e8; move e2e7 should give check.
check_fen = "3k4/8/8/8/8/8/4Q3/4K3 w - - 0 1"
bc = pf.Board(check_fen)
assert bc.gives_check("e2d2") is True
# sanity: a quiet queen move that doesn't give check
assert bc.gives_check("e2e3") is False

# --- 6) mini playout + full rollback (no leaks) ---
b3 = pf.Board()
orig = b3.fen(True)
path = []
for _ in range(30):
    moves = b3.legal_moves()
    if not moves:
        break
    mv = random.choice(moves)
    b3.push_uci(mv)
    path.append(mv)

# roll back everything
while b3.history_size():
    assert b3.unmake() is True

assert_eq(b3.fen(True), orig, "state did not fully restore after rolling back")

print("✅ all tests passed.")


def winner_side(board: pf.Board):
    reason, result = board.is_game_over()
    if reason != "checkmate":
        return None
    # Side to move is the mated side; winner is the opposite
    return "b" if board.side_to_move() == "w" else "w"


def rollout():
    b = pf.Board()
    ply = 0
    
    while True:
        reason, result = b.is_game_over()
        if reason != "none":
            print(f"Game over: reason={reason}, result={result}")
            if reason == 'checkmate':
                print(f"{winner_side(b)} wins!")
            print(f"Total plies: {ply}")
            print("Final FEN:", b.fen(True))
            print("History size:", b.history_size())
            print("Last few moves:", b.history_uci()[-10:])
            break
    
        moves = b.legal_moves()
        if not moves:
            # should be covered by is_game_over, but just in case
            print("No legal moves but not flagged as game over?")
            print("FEN:", b.fen(True))
            break
    
        b.push_uci(random.choice(moves))
        if b.enpassant_sq() != '-':
            print(b.enpassant_sq())
        ply += 1
        
rollout()
#%%
import numpy as np, pyfastchess as pf
b = pf.Board()
x = b.get_piece_planes()
assert x[7,4,5] == 1 and x[0,4,11] == 1  # kings e1/e8


# plane order must match your C++: [P,N,B,R,Q,K, p,n,b,r,q,k]
PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q':10, 'k':11,
}

def truth_planes_from_python_chess(board: chess.Board) -> np.ndarray:
    """
    Build (8,8,14) HWC planes from python-chess board.
    Plane 0..11: pieces
    Plane 12: side-to-move (all 1 if white to move, else 0)
    Plane 13: castling rights encoded in 4 quadrants (K/Q/k/q)
    """
    x = np.zeros((8, 8, 14), dtype=np.uint8)

    # --- pieces ---
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        plane = PIECE_TO_PLANE[p.symbol()]
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        row = 7 - rank_idx  # row 0 = rank 8
        col = file_idx
        x[row, col, plane] = 1

    # --- side-to-move ---
    stm_val = 1 if board.turn == chess.WHITE else 0
    x[..., 12] = stm_val

    # --- castling rights ---
    wk = board.has_kingside_castling_rights(chess.WHITE)
    wq = board.has_queenside_castling_rights(chess.WHITE)
    bk = board.has_kingside_castling_rights(chess.BLACK)
    bq = board.has_queenside_castling_rights(chess.BLACK)

    for r in range(8):
        for c in range(8):
            if r < 4 and c < 4:
                x[r, c, 13] = 1 if wk else 0
            elif r < 4 and c >= 4:
                x[r, c, 13] = 1 if wq else 0
            elif r >= 4 and c < 4:
                x[r, c, 13] = 1 if bk else 0
            else:
                x[r, c, 13] = 1 if bq else 0

    return x

def random_legal_position(max_plies=120) -> chess.Board:
    """Start from startpos, push up to max_plies random legal moves (stop if game over)."""
    b = chess.Board()
    plies = random.randint(0, max_plies)
    for _ in range(plies):
        if b.is_game_over():
            break
        mv = random.choice(list(b.legal_moves))
        b.push(mv)
    return b

def verify_planes_once(max_plies=120, verbose=True) -> bool:
    # 1) make a random legal position
    b_py = random_legal_position(max_plies=max_plies)
    fen = b_py.fen()  # includes counters/ep/castling; fine

    # 2) construct pyfastchess board from the FEN and get planes
    b_pf = pf.Board(fen)
    planes_pf = b_pf.get_piece_planes()  # (8,8,12), uint8

    # 3) ground-truth planes from python-chess
    planes_true = truth_planes_from_python_chess(b_py)

    # 4) quick invariants
    assert planes_pf.shape == (8,8,14) and planes_pf.dtype == np.uint8
    assert planes_true.shape == (8,8,14)
    # number of ones should equal number of pieces
    n_pf = int(planes_pf.sum())
    n_true = int(planes_true.sum())
    if n_pf != n_true:
        if verbose:
            print("Count mismatch:", n_pf, "vs", n_true)
            print("FEN:", fen)
        return False

    # 5) exact match check
    equal = np.array_equal(planes_pf, planes_true)
    if not equal and verbose:
        diff = np.where(planes_pf != planes_true)
        mismatches = list(zip(diff[0].tolist(), diff[1].tolist(), diff[2].tolist()))
        print(f"{len(mismatches)} cell(s) differ. Example up to 10:")
        for r,c,k in mismatches[:10]:
            print(f"  at (row={r}, col={c}, plane={k}): pf={planes_pf[r,c,k]}, truth={planes_true[r,c,k]}")
        print("FEN:", fen)
    return bool(equal)

def verify_planes_many(trials=20, max_plies=120):
    ok = 0
    for i in range(trials):
        if verify_planes_once(max_plies=max_plies, verbose=False):
            ok += 1
        else:
            # re-run verbose to print the failure details
            print(f"Trial {i+1} failed:")
            verify_planes_once(max_plies=max_plies, verbose=True)
            break
    print(f"Passed {ok}/{trials} random positions.")
    return ok == trials


verify_planes_many(trials=100, max_plies=150)


# Precompute once
LOOKUP_RANKS = np.array([-(1 + p // 8) for p in range(64)], dtype=np.int8)
LOOKUP_FILES = np.array([p % 8 for p in range(64)], dtype=np.int8)

def get_rank_file(pieces):
    pieces = np.asarray(pieces, dtype=np.int32)
    return LOOKUP_RANKS[pieces], LOOKUP_FILES[pieces]


def get_board_state(board):
    state = np.zeros((64, 12), dtype=np.int8)

    for color in [chess.WHITE, chess.BLACK]:
        for piece in range(1, 7):
            channel = piece-1 if color else piece-1+6

            squares = np.fromiter(board.pieces(piece, color), dtype=np.int32)
            state[squares, channel] = 1
    
    state = state.reshape(8, 8, 12)    
    state =  np.flipud(state)
    return state


def get_planes(b):
    return b.get_piece_planes()

import time, gc, statistics, numpy as np

# --- timing helper (matches your style) ---
def time_block(fn, label, repeat=1):
    # light warmup
    fn()
    gc.collect()
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn()
    t1 = time.perf_counter()
    elapsed = (t1 - t0) / repeat
    print(f"{label:>28}: {elapsed:.4f} s")
    return elapsed, out

# --- benches ---
def bench_get_board_state():
    # returns a checksum to avoid dead-code elimination
    s = 0
    for b in boards_py:
        x = get_board_state(b)           # (8,8,12) HWC from python-chess
        s += int(x.sum())
    return s

def bench_get_planes():
    s = 0
    for b in boards_pfc:
        x = get_planes(b)                # (8,8,12) HWC from pyfastchess
        s += int(x.sum())
    return s

# --- run ---
n = len(boards_py)
assert n == len(boards_pfc), "boards_py and boards_pfc should be same length/order"

t_state, sum_state = time_block(bench_get_board_state, "get_board_state (py-chess)")
t_planes, sum_planes = time_block(bench_get_planes,     "get_planes (pyfastchess)  ")

print(f"per-board  get_board_state : {1e6*t_state/n:.1f} µs")
print(f"per-board  get_planes      : {1e6*t_planes/n:.1f} µs")
print(f"speedup (state/planes)     : {t_state/t_planes:.2f}×")
print("-"*60)

#%%

import pyfastchess as pfc
import random

b = pfc.Board()
print(b.get_piece_planes().shape)     # (8,8,14)
print(b.stacked_planes(5).shape)      # (8,8,70)

from pyfastchess import Board as fastboard
from collections import deque

# --- play some random plies and record planes ---
b = fastboard()
arrays = deque(maxlen=5)

for ply in range(15):
    moves = b.legal_moves()
    b.push_uci(random.choice(moves))

    # unwind terminal check...
    reason = ''
    while reason != 'none':
        b.push_uci(random.choice(b.legal_moves()))
        reason, result = b.is_game_over()
        if reason == 'none':
            b.unmake()

    arrays.append(b.get_piece_planes())  # deque auto-drops oldest

# --- build stacked planes ---
big_stack = b.stacked_planes(5)  # shape (8,8,70)

# --- assert consistency ---
C = 14
F = 5

# arrays is a deque of length ≤ 5, oldest → newest
N = len(arrays)

# Current planes (newest) should match last slice
assert np.array_equal(arrays[-1], big_stack[:, :, (F-1)*C:F*C])

# Walk backwards through the deque and compare
for back in range(1, F):
    if back < N:
        expected = arrays[-(back+1)]  # older frame
        actual   = big_stack[:, :, (F-1-back)*C:(F-back)*C]
        assert np.array_equal(expected, actual), f"Mismatch at back={back}"
    else:
        # Not enough history → should be zero planes
        actual = big_stack[:, :, (F-1-back)*C:(F-back)*C]
        assert np.count_nonzero(actual) == 0, f"Expected zeros at back={back}"

print("✅ stacked_planes matches deque history for all frames")


def random_legal(depth=15):
    b = fastboard()
    d = 0
    counter = 0
    while d < depth:
        # safety in case it gets caught in some weird corner
        counter += 1
        if counter > 100:
            # just try again
            return random_legal(depth=depth)
        
        moves = b.legal_moves()
        while not len(moves):
            b.unmake()
            moves = b.legal_moves()
            d -= 1

        b.push_uci(random.choice(moves))
        d += 1
    return b


random_legal_fastboards = [random_legal(15) for _ in range(2000)]
random_legal_fens = [b.fen() for b in random_legal_fastboards]
random_legal_chessboards = [chess.Board(f) for f in random_legal_fens]

# --- timing helper (matches your style) ---
def time_block(fn, label, repeat=3):
    # light warmup
    fn()
    gc.collect()
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn()
    t1 = time.perf_counter()
    elapsed = (t1 - t0) / repeat
    print(f"{label:>28}: {elapsed:.4f} s")
    return elapsed, out

# --- benches ---
def bench_get_board_state():
    # returns a checksum to avoid dead-code elimination
    s = 0
    for b in random_legal_chessboards:
        x = get_board_state(b)
        s += int(x.sum())
    return s

def bench_get_planes():
    s = 0
    for b in random_legal_fastboards:
        x = b.stacked_planes(5)
        s += int(x.sum())
    return s


t_state, sum_state = time_block(bench_get_board_state, "get_board_state (py-chess)")
t_planes, sum_planes = time_block(bench_get_planes,     "stacked_planes(5) (pyfastchess)  ")

print(f"per-board  get_board_state : {1e6*t_state/n:.1f} µs")
print(f"per-board  get_planes      : {1e6*t_planes/n:.1f} µs")
print(f"speedup (state/planes)     : {t_state/t_planes:.2f}×")
print("-"*60)

b = fastboard()
print(b.san("e2e4"))   # "e4"
b.push_uci("e2e4")
print(b.san("g8f6"))   # "Nf6"
b.push_uci("g8f6")
print(b.san("e4e5"))   # "e5"

import numpy as np
import chess

# Collapsed promotion: queen folds into 0 (same as "no promo")
_PROMO_COLLAPSED = {
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  0,   # fold Q -> 0
}

def move_to_labels_collapsed(board: chess.Board, move: chess.Move):
    """
    Match C++ labels with collapsed promotion scheme.
    Returns (from_idx, to_idx, piece_idx, promo_idx)
      - from_idx, to_idx: 0..63
      - piece_idx: P=0..K=5 (color-collapsed)
      - promo_idx: 0 = none/queen, 1 = knight, 2 = bishop, 3 = rook
    """
    from_idx = move.from_square
    to_idx   = move.to_square

    piece_type = board.piece_type_at(move.from_square)  # 1..6
    piece_idx  = piece_type - 1                         # 0..5

    promo_idx = 0
    if move.promotion is not None:
        promo_idx = _PROMO_COLLAPSED.get(move.promotion, 0)

    return from_idx, to_idx, piece_idx, promo_idx


def get_legal_labels_collapsed(board: chess.Board, moves):
    """
    Vectorized-ish helper over an iterable of chess.Move.
    Returns dict of numpy arrays matching your heads.
    """
    N = len(moves)
    f = np.zeros(N, dtype=np.int32)
    t = np.zeros(N, dtype=np.int32)
    pc = np.zeros(N, dtype=np.int32)
    pr = np.zeros(N, dtype=np.int32)

    for i, mv in enumerate(moves):
        f[i], t[i], pc[i], pr[i] = move_to_labels_collapsed(board, mv)

    return {"from": f, "to": t, "piece": pc, "promo": pr}

import chess

b = chess.Board()  # startpos
assert move_to_labels_collapsed(b, chess.Move.from_uci("e2e4")) == (12, 28, 0, 0)
assert move_to_labels_collapsed(b, chess.Move.from_uci("g1f3")) == (6, 21, 1, 0)
assert move_to_labels_collapsed(b, chess.Move.from_uci("e1g1")) == (4, 6, 5, 0)   # castle uses king dest in python-chess

# promotions (white)
b = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
assert move_to_labels_collapsed(b, chess.Move.from_uci("a7a8q")) == (48, 56, 0, 0)  # Q -> 0
assert move_to_labels_collapsed(b, chess.Move.from_uci("a7a8n")) == (48, 56, 0, 1)
assert move_to_labels_collapsed(b, chess.Move.from_uci("a7a8b")) == (48, 56, 0, 2)
assert move_to_labels_collapsed(b, chess.Move.from_uci("a7a8r")) == (48, 56, 0, 3)

# promotions (black)
# promotions (black)
b = chess.Board("K6k/8/8/8/8/8/7p/8 b - - 0 1")
assert move_to_labels_collapsed(b, chess.Move.from_uci("h2h1q")) == (15, 7, 0, 0)  # Q -> 0
assert move_to_labels_collapsed(b, chess.Move.from_uci("h2h1n")) == (15, 7, 0, 1)
assert move_to_labels_collapsed(b, chess.Move.from_uci("h2h1b")) == (15, 7, 0, 2)
assert move_to_labels_collapsed(b, chess.Move.from_uci("h2h1r")) == (15, 7, 0, 3)

#%%
# find_edge_case_fens.py
import random
import json
from typing import Callable, Dict, List, Optional, Tuple
import chess

# ---------- helpers ----------
def random_play_until(predicate: Callable[[chess.Board, chess.Move], bool],
                      seed: int = 1337,
                      max_trials: int = 200,
                      max_plies_per_trial: int = 200,
                      chess960: bool = False) -> Dict:
    """
    Repeatedly start games from the initial position and make random legal moves
    until a position is found that has at least one legal move satisfying `predicate`.
    Returns dict with 'fen' and 'moves' (list of UCI moves that satisfy).
    """
    rng = random.Random(seed)
    for trial in range(max_trials):
        b = chess.Board(chess960=chess960)
        for ply in range(max_plies_per_trial):
            legal = list(b.legal_moves)
            if not legal:
                break
            # Check if current position offers any target move
            matches = [mv for mv in legal if predicate(b, mv)]
            if matches:
                return {
                    "fen": b.fen(),                    # FEN BEFORE playing the move
                    "moves": [m.uci() for m in matches],
                    "trial": trial,
                    "ply": ply,
                }
            # Otherwise make a random legal move and continue
            mv = rng.choice(legal)
            b.push(mv)
    return {}

def is_castling(b: chess.Board, mv: chess.Move) -> bool:
    # python-chess knows castling explicitly
    return b.is_castling(mv)

def is_en_passant(b: chess.Board, mv: chess.Move) -> bool:
    return b.is_en_passant(mv)

def is_promotion(b: chess.Board, mv: chess.Move) -> bool:
    # Any promotion piece counts (QNRB)
    return mv.promotion is not None

# ---------- finders ----------
def find_castling(seed=1, max_trials=300, max_plies=300, chess960=False):
    return random_play_until(is_castling, seed, max_trials, max_plies, chess960)

def find_en_passant(seed=2, max_trials=400, max_plies=400, chess960=False):
    # EP is rarer under random play; give it more budget
    return random_play_until(is_en_passant, seed, max_trials, max_plies, chess960)

def find_promotion(seed=3, max_trials=300, max_plies=600, chess960=False):
    # Promotions often need more plies; give extra budget
    return random_play_until(is_promotion, seed, max_trials, max_plies, chess960)

# ---------- run all & emit ----------
def main(save_path: Optional[str] = None, chess960: bool = False):
    results = {
        "castling": find_castling(chess960=chess960),
        "en_passant": find_en_passant(chess960=chess960),
        "promotion": find_promotion(chess960=chess960),
        "meta": {"chess960": chess960},
    }

    # Pretty print (skip 'meta', guard missing keys)
    for k, v in results.items():
        if k == "meta":
            print(f"\n[META] chess960={v.get('chess960')}")
            continue
        if v and "fen" in v:
            print(f"\n[{k.upper()}] trial={v.get('trial')} ply={v.get('ply')}")
            print("FEN:  ", v["fen"])
            print("MOVES:", ", ".join(v.get("moves", [])[:10]))
        else:
            print(f"\n[{k.upper()}] not found within budget")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {save_path}")

    return results


res = main(save_path="C:/Users/Bryan/Data/chessbot_data/edge_case_fens.json", chess960=False)


import json, random
import chess
import numpy as np
from pyfastchess import Board as fastboard

# ---- gold labeler (python-chess) with COLLAPSED promotions ----
_PROMO = {chess.KNIGHT:1, chess.BISHOP:2, chess.ROOK:3, chess.QUEEN:0}

def gold_labels(board: chess.Board, mv: chess.Move):
    from_idx = mv.from_square
    to_idx   = mv.to_square
    piece_idx = board.piece_type_at(mv.from_square) - 1   # P..K -> 0..5
    promo_idx = 0 if mv.promotion is None else _PROMO.get(mv.promotion, 0)
    return (from_idx, to_idx, piece_idx, promo_idx)

# ---- core comparers ----
def compare_move(fen: str, uci: str):
    b_py = chess.Board(fen)
    b_fb = fastboard(fen)  # if your ctor differs, adjust (e.g., Board(); set_fen)
    gold = gold_labels(b_py, chess.Move.from_uci(uci))
    got  = b_fb.move_to_labels(uci)
    return got, gold

def compare_all_legal(fen: str, limit=None, verbose=False):
    b_py = chess.Board(fen)
    b_fb = fastboard(fen)
    moves = list(b_py.legal_moves)
    if limit:
        random.shuffle(moves); moves = moves[:limit]
    bad = []
    for mv in moves:
        uci = mv.uci()
        got  = b_fb.move_to_labels(uci)
        gold = gold_labels(b_py, mv)
        if got != gold:
            bad.append((uci, got, gold))
            if verbose:
                print(f"BAD {uci:6s}  got={got}  gold={gold}")
    return bad

# ---- edge-case suite using your 'res' dict or JSON on disk ----
def run_edge_case_suite(res_or_path, check_all_legal=False, legal_limit=None):
    if isinstance(res_or_path, str):
        with open(res_or_path, "r", encoding="utf-8") as f:
            res = json.load(f)
    else:
        res = res_or_path

    def run_bucket(name):
        bucket = res.get(name) or {}
        fen = bucket.get("fen")
        moves = bucket.get("moves", [])
        if not fen:
            print(f"{name.upper():>11}: not found"); return
        print(f"\n{name.upper():>11}: FEN ok, {len(moves)} target move(s)")
        # Check the target moves you discovered
        any_bad = False
        for u in moves:
            got, gold = compare_move(fen, u)
            if got == gold:
                print(f"  OK  {u:6s}  {got}")
            else:
                any_bad = True
                print(f"  BAD {u:6s}  got={got}  gold={gold}")
        # Optionally sweep all legal moves from this FEN
        if check_all_legal:
            bad = compare_all_legal(fen, limit=legal_limit, verbose=False)
            if bad:
                any_bad = True
                print(f"  LEGAL SWEEP: {len(bad)} mismatch(es); first few:")
                for uci, got, gold in bad[:10]:
                    print(f"    {uci:6s} got={got} gold={gold}")
            else:
                print("  LEGAL SWEEP: all matched ✅")
        if not any_bad:
            print("  ▶ bucket matched ✅")

    run_bucket("castling")
    run_bucket("en_passant")
    run_bucket("promotion")

# --------- run it ---------
# If you already have `res` in memory:
res = main(save_path="C:/Users/Bryan/Data/chessbot_data/edge_case_fens.json", chess960=False)
run_edge_case_suite(res, check_all_legal=True, legal_limit=None)

# Or load from disk:
# run_edge_case_suite(r"C:/Users/Bryan/Data/chessbot_data/edge_case_fens.json", check_all_legal=True)

from pyfastchess import Board as fastboard
import chess

b = fastboard()
print("fast legal move gen", b.legal_moves())

print(b.side_to_move()) # 'w' for white, 'b' for black
startpos = chess.Board().fen()
b2 = fastboard(startpos)

def random_init(plies=5):
    b = fastboard()
    p = 0
    counter = 0
    limit = int(3*plies)
    while p < plies:
        # safety in case it gets caught in some weird corner
        counter += 1
        if counter > limit:
            # really bad luck, just try again
            return random_init(plies=plies)
        
        moves = b.legal_moves()
        while not len(moves):
            b.unmake()
            moves = b.legal_moves()
            p -= 1

        b.push_uci(random.choice(moves))
        p += 1
    return b

rb = random_init(5)
random_fen = rb.fen()

rb10 = random_init(10)
print("all 10 moves:", rb10.history_uci())
for _ in range(3):
    rb10.unmake()
    print(rb10.history_size(), rb10.history_uci())

rb_clone = rb.clone()
print("move histories match", rb_clone.history_uci(), rb.history_uci())

# for machine learning
X = rb.stacked_planes(5)
print("shape of X (ml input):", X.shape)

model.summary()
print(model.input_names)
print(model.output_names)





