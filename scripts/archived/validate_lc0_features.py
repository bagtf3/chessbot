"""
Validate board.lc0_features() (C++) against Lc0Board.features() (Python).
2000 random positions, 10+ plies deep. Acceptance: 100% exact match.
"""

import sys, random
import numpy as np
import chess

sys.path.insert(0, r'C:\Users\Bryan\repos\chessbot\src')
from chessbot.lc0_utils import Lc0Board
from pyfastchess import Board as FastBoard, lc0_features_float

random.seed(0)
N        = 2000
MIN_PLY  = 10
MAX_PLY  = 60
failures = []

for trial in range(N):
    # build a random game
    b  = chess.Board()
    lb = Lc0Board()
    bf = FastBoard()

    n_moves = random.randint(MIN_PLY, MAX_PLY)
    for _ in range(n_moves):
        legal = list(b.legal_moves)
        if not legal:
            break
        mv = random.choice(legal)
        b.push(mv)
        lb.push(mv)
        bf.push_uci(mv.uci())

    if len(b.move_stack) < MIN_PLY:
        continue

    ref  = lb.features()                 # (112, 8, 8) float32 from Python
    got  = lc0_features_float(bf)        # (112, 8, 8) float32 from C++

    if not np.array_equal(ref, got):
        diff_planes = np.where(~np.all(ref == got, axis=(1, 2)))[0]
        failures.append({
            'trial':       trial,
            'fen':         b.fen(),
            'plies':       len(b.move_stack),
            'diff_planes': diff_planes.tolist(),
            'max_abs_err': float(np.max(np.abs(ref - got))),
        })

print(f'Tested {N} positions')
if not failures:
    print('PASS: all positions match exactly')
else:
    print(f'FAIL: {len(failures)} / {N} mismatches')
    for f in failures[:10]:
        print(f"  trial={f['trial']}  plies={f['plies']}"
              f"  planes={f['diff_planes']}  max_err={f['max_abs_err']:.6f}")
        print(f"    fen: {f['fen']}")
    if len(failures) > 10:
        print(f'  ... and {len(failures) - 10} more')
    sys.exit(1)


# %% timing comparison

import time

TIMED = 1000

# build paired lists so both methods see identical positions
boards_ch = []
boards_lb = []
boards_bf = []
rng = random.Random(99)
while len(boards_ch) < TIMED:
    b  = chess.Board()
    lb = Lc0Board()
    bf = FastBoard()
    for _ in range(rng.randint(MIN_PLY, MAX_PLY)):
        legal = list(b.legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        b.push(mv)
        lb.push(mv)
        bf.push_uci(mv.uci())
    if len(b.move_stack) >= MIN_PLY:
        boards_ch.append(b)
        boards_lb.append(lb)
        boards_bf.append(bf)

# warmup
for lb, bf in zip(boards_lb[:20], boards_bf[:20]):
    lb.features()
    lc0_features_float(bf)

# time Python Lc0Board
t0 = time.perf_counter()
for lb in boards_lb:
    lb.features()
t_py = time.perf_counter() - t0

# time C++ lc0_features
t0 = time.perf_counter()
for bf in boards_bf:
    lc0_features_float(bf)
t_cpp = time.perf_counter() - t0

print(f'\nTiming over {TIMED} positions:')
print(f'  Python Lc0Board.features(): {t_py/TIMED*1e6:.1f} us/pos  ({TIMED/t_py:.0f} pos/sec)')
print(f'  C++ lc0_features_float():   {t_cpp/TIMED*1e6:.1f} us/pos  ({TIMED/t_cpp:.0f} pos/sec)')
print(f'  Speedup: {t_py/t_cpp:.1f}x')
