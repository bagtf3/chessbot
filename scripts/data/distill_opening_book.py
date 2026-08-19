"""
Distill an opening book from the pre_opened 4-8 book using t1-large TRT.

Walks all 6211 opening lines from startpos. At each position along the way,
checks a shortfen cache (no clock fields). New positions are queued for
batch ORT inference (batch=128) through the t1-large .engine. Results are
stored as XC0-format 4288-dim policy probabilities + WDL.

Policy is softmaxed over legal moves only before mapping to XC0 slots.

Output: {shortfen: (xc0_policy float32(4288,), wdl float32(3,))}
Saved alongside the source opening book pkl.
"""

# %% [1] Imports and paths

import sys, os, pickle
import numpy as np
import chess

sys.path.insert(0, r'C:\Users\Bryan\repos\chessbot\src')
from chessbot.lc0_utils import Lc0Board, softmax, make_lc0_trt_session
from pyfastchess import Board as FastBoard

from xerces_training.parse_utils import uci_to_xerces_index

PRE_OPENED_PKL = os.getenv('PRE_OPENED_UCI_PATHS', '')
TRT_CACHE      = r'C:\Users\Bryan\Data\chessbot_data\lc0\trt_cache'
ONNX_PATH      = r'C:\Users\Bryan\Data\chessbot_data\lc0\onnx\t1-512x15x8h-distilled-swa-3395000.onnx'
MODEL_NAME     = 't1-big'
BATCH_SIZE     = 128
OUT_PKL        = r'C:\Users\Bryan\Data\chessbot_data\t1_large_book_evals.pkl'

print(f'Source book: {PRE_OPENED_PKL}')
print(f'Output:      {OUT_PKL}')


# %% [2] Load TRT session

sess = make_lc0_trt_session(ONNX_PATH, MODEL_NAME, TRT_CACHE)


# %% [3] Walk all opening lines and collect unique positions

print(f'\nLoading {PRE_OPENED_PKL} ...')
with open(PRE_OPENED_PKL, 'rb') as f:
    PATHS = pickle.load(f)
print(f'  {len(PATHS)} opening lines')

pos_features  = {}   # shortfen -> np.ndarray (112, 8, 8)
pos_lc0_idxs  = {}   # shortfen -> list of LC0 policy indices for legal moves
pos_xc0_idxs  = {}   # shortfen -> list of XC0 flat indices for legal moves

def record_position(b, lb, bf):
    sf = bf.fen(include_counters=False)
    if sf in pos_features:
        return
    legal     = [m.uci() for m in b.legal_moves]
    stm_white = b.turn == chess.WHITE
    pos_features[sf] = lb.features()
    pos_lc0_idxs[sf] = lb.policy_indices(legal)
    pos_xc0_idxs[sf] = [uci_to_xerces_index(u, stm_white=stm_white) for u in legal]

for line_idx, moves_uci in enumerate(PATHS):
    b  = chess.Board()
    lb = Lc0Board()
    bf = FastBoard()
    record_position(b, lb, bf)

    for m in moves_uci:
        move = chess.Move.from_uci(m)
        b.push(move)
        lb.push(move)
        bf.push_uci(m)
        record_position(b, lb, bf)

    if (line_idx + 1) % 1000 == 0:
        print(f'  lines {line_idx + 1}/{len(PATHS)}'
              f'  unique so far: {len(pos_features)}')

print(f'\nTotal unique positions: {len(pos_features)}')


# %% [4] Batch inference through t1-large TRT

sf_list      = list(pos_features.keys())
features_arr = np.stack([pos_features[sf] for sf in sf_list])   # (N, 112, 8, 8)
N            = len(sf_list)
n_batches    = (N + BATCH_SIZE - 1) // BATCH_SIZE

all_wdl    = np.zeros((N, 3),    dtype=np.float32)
all_logits = np.zeros((N, 1858), dtype=np.float32)

print(f'Running {n_batches} batches of up to {BATCH_SIZE}...')
for i in range(0, N, BATCH_SIZE):
    batch        = features_arr[i:i + BATCH_SIZE]
    wdl_raw, pol = sess.run(['/output/wdl', '/output/policy'],
                            {'/input/planes': batch})
    all_wdl[i:i + BATCH_SIZE]    = wdl_raw
    all_logits[i:i + BATCH_SIZE] = pol
    b_idx = i // BATCH_SIZE + 1
    if b_idx % 50 == 0 or b_idx == n_batches:
        print(f'  batch {b_idx}/{n_batches}')

all_wdl = softmax(all_wdl)
print('Inference done.')


# %% [5] Mask to legal moves, softmax, place into XC0 slots, save

print('Converting to XC0 policy format (legal-move masked softmax)...')
all_xc0 = np.zeros((N, 4288), dtype=np.float32)

for i, sf in enumerate(sf_list):
    lc0_idxs = pos_lc0_idxs[sf]
    xc0_idxs = pos_xc0_idxs[sf]
    if not lc0_idxs:
        continue
    logits = all_logits[i][lc0_idxs].astype(np.float64)
    e      = np.exp(logits - logits.max())
    probs  = (e / e.sum()).astype(np.float32)
    all_xc0[i][xc0_idxs] = probs

book = {sf: (all_xc0[i], all_wdl[i]) for i, sf in enumerate(sf_list)}

print(f'Saving {len(book)} entries to {OUT_PKL} ...')
with open(OUT_PKL, 'wb') as f:
    pickle.dump(book, f, protocol=4)
print('Done.')


# %% [6] Overwrite startpos, after 1.e4, after 1.d4 with 791556 priors

OVERWRITE_LINES = [
    ['e2e4'],     # after 1.e4
    ['d2d4'],     # after 1.d4
]

sess_791 = make_lc0_trt_session(
    r'C:\Users\Bryan\Data\chessbot_data\lc0\onnx\791556.onnx', '791556', TRT_CACHE)

for moves_uci in OVERWRITE_LINES:
    b  = chess.Board()
    lb = Lc0Board()
    bf = FastBoard()
    for m in moves_uci:
        move = chess.Move.from_uci(m)
        b.push(move)
        lb.push(move)
        bf.push_uci(m)
    sf = bf.fen(include_counters=False)

    feats        = lb.features()[None]
    wdl_raw, pol = sess_791.run(['/output/wdl', '/output/policy'],
                                {'/input/planes': feats})
    wdl          = softmax(wdl_raw)[0]
    pol          = pol[0]
    stm_white    = b.turn == chess.WHITE
    legal        = [m.uci() for m in b.legal_moves]
    lc0_idxs     = lb.policy_indices(legal)
    xc0_idxs     = [uci_to_xerces_index(u, stm_white=stm_white) for u in legal]
    logits       = pol[lc0_idxs].astype(np.float64)
    e            = np.exp(logits - logits.max())
    probs        = (e / e.sum()).astype(np.float32)
    xc0          = np.zeros(4288, dtype=np.float32)
    xc0[xc0_idxs] = probs
    book[sf]     = (xc0, wdl)
    label        = ' '.join(moves_uci) if moves_uci else 'startpos'
    print(f'Overwritten {label}  ({sf})')

with open(OUT_PKL, 'wb') as f:
    pickle.dump(book, f, protocol=4)
print(f'Saved {len(book)} entries to {OUT_PKL}')


# %% [8] Inspect

def show_policy(book, fen, label):
    b  = chess.Board(fen)
    sf = FastBoard(fen).fen(include_counters=False)
    if sf not in book:
        print(f'{label}: not in book ({sf})')
        return
    xc0, wdl  = book[sf]
    stm_white = b.turn == chess.WHITE
    legal     = [m.uci() for m in b.legal_moves]
    rows      = []
    for uci in legal:
        xi   = uci_to_xerces_index(uci, stm_white=stm_white)
        prob = float(xc0[xi])
        rows.append((uci, xi, prob))
    rows.sort(key=lambda r: -r[2])
    print(f'\n{label}  WDL=[{wdl[0]:.3f} {wdl[1]:.3f} {wdl[2]:.3f}]  '
          f'policy_sum={sum(r[2] for r in rows):.4f}')
    print(f'  {"move":<8}  {"xc0_idx":>8}  {"prob":>8}')
    for uci, xi, prob in rows:
        print(f'  {uci:<8}  {xi:>8}  {prob:>8.4f}')

show_policy(book, chess.Board().fen(), 'startpos')
show_policy(book, 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1', 'after 1.e4')


# %% [9] Compare all 5 LC0 models across 4 positions

MODELS_7 = [
    ('LD2',      r'C:\Users\Bryan\Data\chessbot_data\lc0\onnx\LD2.onnx'),
    ('791556',   r'C:\Users\Bryan\Data\chessbot_data\lc0\onnx\791556.onnx'),
    ('t1-small', r'C:\Users\Bryan\Data\chessbot_data\lc0\onnx\t1-256x10-distilled-swa-2432500.onnx'),
    ('t1-big',   r'C:\Users\Bryan\Data\chessbot_data\lc0\onnx\t1-512x15x8h-distilled-swa-3395000.onnx'),
    ('BT4',      r'C:\Users\Bryan\Data\chessbot_data\lc0\onnx\BT4-1024x15x32h-swa-6147500.onnx'),
]

POSITIONS_7 = [
    ('startpos',    []),
    ('after 1.e4',  ['e2e4']),
    ('after 1.d4',  ['d2d4']),
    ('after 1.e4 e5', ['e2e4', 'e7e5']),
]

print('Loading sessions...')
sessions_7 = {}
for name, onnx in MODELS_7:
    sessions_7[name] = make_lc0_trt_session(onnx, name, TRT_CACHE)


def top5_priors(sess, moves_uci):
    b  = chess.Board()
    lb = Lc0Board()
    for m in moves_uci:
        move = chess.Move.from_uci(m)
        b.push(move)
        lb.push(move)
    feats        = lb.features()[None]
    wdl_raw, pol = sess.run(['/output/wdl', '/output/policy'], {'/input/planes': feats})
    wdl          = softmax(wdl_raw)[0]
    pol          = pol[0]
    stm_white    = b.turn == chess.WHITE
    legal        = [m.uci() for m in b.legal_moves]
    lc0_idxs     = lb.policy_indices(legal)
    logits       = pol[lc0_idxs].astype(np.float64)
    e            = np.exp(logits - logits.max())
    probs        = e / e.sum()
    ranked       = sorted(zip(legal, probs), key=lambda r: -r[1])[:5]
    return wdl, ranked


for pos_label, moves_uci in POSITIONS_7:
    print(f'\n{"=" * 60}')
    print(f'  {pos_label}')
    print(f'{"=" * 60}')
    print(f'  {"model":<10}  {"WDL (W/D/L)":<26}  top-5 moves')
    print(f'  {"-" * 10}  {"-" * 26}  {"-" * 30}')
    for name, _ in MODELS_7:
        wdl, ranked = top5_priors(sessions_7[name], moves_uci)
        wdl_str  = f'[{wdl[0]:.3f} {wdl[1]:.3f} {wdl[2]:.3f}]'
        moves_str = '  '.join(f'{u}={p:.3f}' for u, p in ranked)
        print(f'  {name:<10}  {wdl_str:<26}  {moves_str}')
