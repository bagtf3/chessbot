"""
Validate t1-small TRT inference against lc0.exe priors on 50 pre_opened positions.

For each position: lc0.exe (nodes=800) reports policy priors (P_pct) per legal move.
ORT TRT runs the same model and produces policy logits -> softmax over legal moves.
Both should agree closely since they use the same weights.
"""

# %% [1] Imports and paths

import sys, os, pickle, random
import numpy as np
import chess

sys.path.insert(0, r'C:\Users\Bryan\repos\chessbot\src')
from chessbot import LC0_WEIGHTS
from chessbot.engines import Lc0Session
from chessbot.lc0_utils import Lc0Board, softmax, make_lc0_trt_session

PRE_OPENED_PKL = os.getenv('PRE_OPENED_UCI_PATHS', '')
TRT_CACHE      = r'C:\Users\Bryan\Data\chessbot_data\lc0\trt_cache'
ONNX_PATH      = r'C:\Users\Bryan\Data\chessbot_data\lc0\onnx\t1-256x10-distilled-swa-2432500.onnx'
MODEL_NAME     = 't1-small'
N_POSITIONS    = 50
EXTRA_MOVES    = 3

print(f'LC0_WEIGHTS:    {LC0_WEIGHTS}')
print(f'PRE_OPENED_PKL: {PRE_OPENED_PKL}')


# %% [2] Load ORT TRT session

sess = make_lc0_trt_session(ONNX_PATH, MODEL_NAME, TRT_CACHE)


# %% [3] Sample positions from opening book

print(f'Loading {PRE_OPENED_PKL} ...')
with open(PRE_OPENED_PKL, 'rb') as f:
    PATHS = pickle.load(f)
print(f'  {len(PATHS)} lines')

random.seed(42)
sample = random.sample(PATHS, min(N_POSITIONS, len(PATHS)))

positions = []
for moves_uci in sample:
    b  = chess.Board()
    lb = Lc0Board()
    for m in moves_uci:
        move = chess.Move.from_uci(m)
        b.push(move)
        lb.push(move)
    for _ in range(EXTRA_MOVES):
        legal = list(b.legal_moves)
        if not legal:
            break
        move = random.choice(legal)
        b.push(move)
        lb.push(move)
    positions.append((b, lb))

print(f'Built {len(positions)} positions.')


# %% [4] ORT TRT batch inference

feats       = np.stack([lb.features() for _, lb in positions])
wdl_raw, pol_logits = sess.run(
    ['/output/wdl', '/output/policy'], {'/input/planes': feats})
wdl_ort = softmax(wdl_raw)

print(f'ORT inference done. batch={feats.shape[0]}, policy shape={pol_logits.shape}')


# %% [5] lc0.exe inference (nodes=800 for real policy + WDL)

lc0_rows_all = []
lc0_wdls     = []
with Lc0Session(cfg={'WeightsFile': LC0_WEIGHTS}) as lc0:
    for i, (b, _) in enumerate(positions):
        rows, wdl_lc0, pv = lc0.analyze(b, nodes=800)
        lc0_rows_all.append(rows)
        lc0_wdls.append(wdl_lc0)
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(positions)}')

print('lc0.exe done.')


# %% [6] Compare policy distributions

corrs, top1_agrees, kl_divs = [], [], []

for i, ((b, lb), rows) in enumerate(zip(positions, lc0_rows_all)):
    lc0_map = {r['move']: r.get('P_pct', 0.0) / 100.0
               for r in rows if r.get('P_pct') is not None}
    if not lc0_map:
        continue

    legal   = [m.uci() for m in b.legal_moves]
    idxs    = lb.policy_indices(legal)
    raw     = pol_logits[i][idxs].astype(np.float64)
    e       = np.exp(raw - raw.max())
    ort_arr = e / e.sum()
    ort_map = dict(zip(legal, ort_arr))

    common = [m for m in legal if m in lc0_map]
    if len(common) < 2:
        continue

    lc0_vec  = np.array([lc0_map[m] for m in common])
    ort_vec  = np.array([ort_map.get(m, 0.0) for m in common])
    lc0_vec  = lc0_vec / lc0_vec.sum()
    ort_vec  = ort_vec / ort_vec.sum() if ort_vec.sum() > 0 else ort_vec

    corr     = float(np.corrcoef(lc0_vec, ort_vec)[0, 1])
    top1_lc0 = max(lc0_map, key=lc0_map.get)
    top1_ort = legal[int(np.argmax(ort_arr))]
    eps      = 1e-9
    kl       = float(np.sum(lc0_vec * np.log((lc0_vec + eps) / (ort_vec + eps))))

    corrs.append(corr)
    top1_agrees.append(top1_lc0 == top1_ort)
    kl_divs.append(kl)

n = len(corrs)
print(f'\n  positions evaluated:  {n}')
print(f'  mean policy corr:     {np.mean(corrs):.4f}')
print(f'  min  policy corr:     {np.min(corrs):.4f}')
print(f'  top-1 agree:          {np.mean(top1_agrees):.3f}  ({sum(top1_agrees)}/{n})')
print(f'  mean KL (lc0||ort):   {np.mean(kl_divs):.4f}')
print(f'  max  KL (lc0||ort):   {np.max(kl_divs):.4f}')


# %% [7] WDL sample

print(f'\n{"#":<4}  {"lc0.exe WDL (W/D/L)":<26}  ORT TRT WDL')
for i, (lc0_wdl, ort_wdl) in enumerate(zip(lc0_wdls[:5], wdl_ort[:5])):
    lc0_str = f'[{lc0_wdl[0]:.3f} {lc0_wdl[1]:.3f} {lc0_wdl[2]:.3f}]' if lc0_wdl else '[n/a]'
    ort_str = f'[{ort_wdl[0]:.3f} {ort_wdl[1]:.3f} {ort_wdl[2]:.3f}]'
    print(f'{i:<4}  {lc0_str:<26}  {ort_str}')
