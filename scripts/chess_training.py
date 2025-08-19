import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import pickle, os, math, random, time
            
import matplotlib.pyplot as plt
plt.ion()

import io
import chess
import chess.pgn
import chess.engine

SF_LOC = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"


import chessbot.model as cbm
import chessbot.utils as cbu
import chessbot.features as cbf

from chessbot.encoding import (
    encode_board, build_training_targets_8x8x73, 
    legal_mask_8x8x73, idx_to_move_8x8x73
)


def ResidualBlock(width: int):
    def f(x):
        y = L.Conv2D(width, 3, padding="same", use_bias=False)(x)
        y = L.BatchNormalization()(y)
        y = L.ReLU()(y)
        y = L.Conv2D(width, 3, padding="same", use_bias=False)(y)
        y = L.BatchNormalization()(y)
        return L.ReLU()(L.Add()([x, y]))
    return f


def build_chess_model(in_planes=23, width=128, n_blocks=12):
    """Input: [8,8,in_planes]  ->  policy: [8,8,73] (logits),  value: [1] (sigmoid)"""
    inp = L.Input(shape=(8,8,in_planes), name="board_planes")

    # stem
    x = L.Conv2D(width, 3, padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # body
    for _ in range(n_blocks):
        x = ResidualBlock(width)(x)

    # policy head: 1x1 -> 73 planes (logits; don’t softmax here)
    p = L.Conv2D(73, 1, padding="same", use_bias=True, name="policy_logits")(x)  # [B,8,8,73]

    # value head: 1x1 -> 32, GAP, MLP -> 1 sigmoid
    v = L.Conv2D(32, 1, padding="same", use_bias=True)(x)
    v = L.GlobalAveragePooling2D()(v)
    v = L.Dense(256, activation="relu")(v)
    v = L.Dense(1, activation="sigmoid", name="value")(v)

    return keras.Model(inp, outputs=[p, v], name="Alpha0ish")






all_X = []
all_Yp = []
all_Yv = []

board = chess.Board()
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
    while len(all_X) < 250:
        if board.is_game_over():
            board = chess.Board()
            
        if not len(list(board.legal_moves)):
            board = chess.Board()
        
        # random init
        if len(board.move_stack) == 0:
            board.push(random.choice(list(board.legal_moves)))
        
        # always "white" pov
        to_score = board if board.turn else board.mirror()
        
        info_list = engine.analyse(
            to_score, multipv=50, limit=chess.engine.Limit(depth=4),
            info=chess.engine.Info.ALL
        )
        
        all_X.append(encode_board(to_score))
        yp, yv = build_training_targets_8x8x73(to_score, info_list)
        all_Yp.append(yp)
        all_Yv.append(yv)
        
        move_to_make = random.choice(info_list[:5])['pv'][0]
        if not board.turn:
            move_to_make = mirror_move(move_to_make)
        
        board.push(move_to_make)
    
        
X = np.stack(all_X, axis=0)            # [N,8,8,23]
Yp = np.stack(all_Yp, axis=0)          # [N,8,8,73]
Yv = np.stack(all_Yv, axis=0)          # [N,1]
print(X.shape, Yp.shape, Yv.shape)

batch_size = 64

def ds_from(X, Yp, Yv):
    # just one dataset, shuffle every epoch
    ds = tf.data.Dataset.from_tensor_slices((X, {"policy_logits": Yp, "value": Yv}))
    ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = ds_from(X, Yp, Yv)

# Train
history = model.fit(train_ds, epochs=8)

#%%

def masked_softmax_pick(policy_logits, legal_mask, temperature=1.0, sample=False):
    """
    policy_logits: np.ndarray [8,8,73] (raw logits from the net)
    legal_mask:    np.ndarray bool [8,8,73]
    temperature:   float > 0 (lower = sharper)
    sample:        if True, sample from the distribution; else argmax

    Returns:
      probs:  [8,8,73] np.float32, sums to 1 over legal entries
      best:   (fr, ff, pl) tuple for chosen move (None if no legal)
      pbest:  float, probability of chosen move (0 if none)
    """
    logits = np.asarray(policy_logits, dtype=np.float32).copy()
    mask   = np.asarray(legal_mask, dtype=bool)

    # handle no-legal-move edge case
    k = int(mask.sum())
    if k == 0:
        return np.zeros_like(logits, dtype=np.float32), None, 0.0

    # temperature + mask
    t = max(float(temperature), 1e-6)
    flat = logits.ravel() / t
    mflat = mask.ravel()
    flat[~mflat] = -1e9  # effectively -inf

    # stable softmax over legal entries only
    m = np.max(flat[mflat])
    exps = np.zeros_like(flat, dtype=np.float32)
    exps[mflat] = np.exp(flat[mflat] - m)
    Z = exps[mflat].sum()
    if not np.isfinite(Z) or Z <= 0.0:
        # fallback uniform over legal if something went weird
        probs = np.zeros_like(flat, dtype=np.float32)
        probs[mflat] = 1.0 / k
        probs = probs.reshape(8,8,73)
        # pick index
        choice = np.random.choice(np.where(mflat)[0]) if sample else np.where(mflat)[0][0]
        fr, ff, pl = np.unravel_index(choice, (8,8,73))
        return probs, (fr, ff, pl), float(probs.reshape(-1)[choice])

    probs_flat = exps / Z
    probs = probs_flat.reshape(8,8,73)

    # pick a move
    if sample:
        choice = np.random.choice(probs_flat.size, p=probs_flat)
    else:
        choice = int(np.argmax(probs_flat))
    fr, ff, pl = np.unravel_index(choice, (8,8,73))
    return probs, (fr, ff, pl), float(probs_flat[choice])


def make_move_with_model(board, model):
    original_color = board.turn
    to_score = board if original_color else board.mirror()
    test = [encode_board(to_score)]
    
    mask = legal_mask_8x8x73(to_score)
    pred = model.predict(np.stack(test, axis=0))
    policy_logits = pred[0]
    
    probs, (fr, ff, pl), score = masked_softmax_pick(policy_logits, mask)
    move = idx_to_move_8x8x73(fr, ff, pl, to_score)
    
    if not original_color:
        move = mirror_move(move)
    
    return move


move = make_move_with_model(board, model)
board.push(move)
