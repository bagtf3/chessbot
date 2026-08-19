"""
Timing test: reconstruct Lc0Board from a chess.Board move stack and call features().
Simulates the on-demand conversion approach for rescore.py.
"""

# %% [1] Generate 1000 random positions >= 10 plies deep

import sys, time, random
import chess
import numpy as np

sys.path.insert(0, r'C:\Users\Bryan\repos\chessbot\src')
from chessbot.lc0_utils import Lc0Board

random.seed(42)
N = 1000
MIN_PLY = 10

positions = []   # list of (chess.Board with move_stack intact)
while len(positions) < N:
    b = chess.Board()
    n_moves = random.randint(MIN_PLY, 60)
    for _ in range(n_moves):
        legal = list(b.legal_moves)
        if not legal:
            break
        b.push(random.choice(legal))
    if len(b.move_stack) >= MIN_PLY:
        positions.append(b)

print(f'Generated {len(positions)} positions')
print(f'Ply depth: min={min(len(b.move_stack) for b in positions)}'
      f'  max={max(len(b.move_stack) for b in positions)}'
      f'  mean={np.mean([len(b.move_stack) for b in positions]):.1f}')


# %% [2] Time the conversion: replay last 8 moves into Lc0Board, call features()

def board_to_lc0_features(b: chess.Board) -> np.ndarray:
    """Reconstruct Lc0Board from chess.Board move stack (last 8 moves) and return features."""
    stack = list(b.move_stack)
    # walk back to position 8 moves ago
    n_replay = min(8, len(stack))
    replay_moves = stack[-n_replay:]

    # board at the start of the replay window
    tmp = b.copy()
    for _ in range(n_replay):
        tmp.pop()

    lb = Lc0Board(tmp)
    for mv in replay_moves:
        lb.push(mv)

    return lb.features()


# warmup
for b in positions[:10]:
    board_to_lc0_features(b)

t0 = time.perf_counter()
for b in positions:
    board_to_lc0_features(b)
elapsed = time.perf_counter() - t0

print(f'\n{N} conversions in {elapsed*1000:.1f} ms')
print(f'Per position: {elapsed/N*1e6:.1f} us')
print(f'Throughput:   {N/elapsed:.0f} positions/sec')
