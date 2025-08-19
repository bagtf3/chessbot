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




move = make_move_with_model(board, model)
board.push(move)
