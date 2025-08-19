import os
import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import random            
import matplotlib.pyplot as plt
plt.ion()

import chess
import chess.engine

SF_LOC = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"

import chessbot.model as cbm
import chessbot.utils as cbu
import chessbot.features as cbf
import chessbot.encoding as cbe

# model, loss_weights = cbm.new_policy_value_model()

# all_X = []
# all_Y = []
# #total_pos = 0
# #all_evals = pd.DataFrame()

# board = chess.Board()
# with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
#     while total_pos < 80000:
#         if board.is_game_over():
#             board = cbu.random_init(1)
            
#         legal = list(board.legal_moves)
#         if not legal:
#             board = cbu.random_init(1)
        
#         # always "white" pov
#         to_score = board if board.turn else cbu.mirror_board(board)
        
#         info_list = engine.analyse(
#             to_score, multipv=50, limit=chess.engine.Limit(depth=5),
#             info=chess.engine.Info.ALL
#         )
        
#         info_list = [i for i in info_list if 'pv' in i and i['pv']]
#         if not info_list:
#             board = cbu.random_init(1)
#             continue
        
#         all_X.append(cbe.encode_board(to_score))
#         y = cbe.build_training_targets_8x8x73(to_score, info_list)

#         y.update(cbf.all_king_exposure_features(to_score))
#         y.update(cbf.all_piece_features(to_score))

#         y['material'] = cbf.get_piece_value_sum(to_score)
#         y['piece_to_move'] = cbe.piece_to_move_target(to_score, y['policy_logits'])
#         y['legal_moves'] = 1*cbe.legal_mask_8x8x73(to_score).astype(np.float32)
#         all_Y.append(y.copy())
#         total_pos += 1

#         move_to_make = random.choice(info_list[:5])['pv'][0]
#         if not board.turn:
#             move_to_make = cbu.mirror_move(move_to_make)
        
#         board.push(move_to_make)
        
#         # test and train     
#         if len(all_X) > 2000:
#             targets = cbm.prepare_targets(all_Y)
#             X = np.stack(all_X, axis=0, dtype=np.float32)
            
#             eval_df = cbu.score_game_data(model, X, targets)
#             all_evals = pd.concat([all_evals, eval_df])
#             cbu.plot_training_progress(all_evals)
            
            
#             # train (no val)
#             model.fit(X, targets, batch_size=128, epochs=4, shuffle=True)
#             all_X = []
#             all_Y = []

from multiprocessing import Process, Queue, Event
from chessbot.data_gen import data_worker

def train_loop(model, n_workers=4, batch_size=128):
    total_pos = 0
    n_trains = 0
    queue = Queue(maxsize=6000)
    stop_event = Event()
    procs = [
        Process(target=data_worker, args=(i, queue, stop_event))
        for i in range(n_workers)
    ]
    
    for p in procs: p.start()

    all_evals = []
    try:
        while total_pos < 10000:
            all_X, all_Y = [], []
            # accumulate a minibatch
            while len(all_X) < 1000:
                X, y = queue.get()
                all_X.append(X)
                all_Y.append(y)

            X = np.stack(all_X, axis=0, dtype=np.float32)
            targets = cbm.prepare_targets(all_Y)

            # metrics
            if n_trains % 5 == 4:
                eval_df = cbu.score_game_data(model, X, targets)
                all_evals.append(eval_df)
                cbu.plot_training_progress(pd.concat(all_evals))

            # train
            model.fit(X, targets, batch_size=batch_size, epochs=2, shuffle=True)
            n_trains += 1
            total_pos += len(all_X)
            
    finally:
        
        cbm.save_model(model, os.path.join(cbm.MODEL_DIR, "value_policy_model_v1.h5"))
        stop_event.set()
        for p in procs: p.join()


if __name__ == "__main__":
    model_path = os.path.join(cbm.MODEL_DIR, "value_policy_model_v1.h5")
    if os.path.exists(model_path):
        model = cbm.load_model(model_path)
    else:
        model, loss_weights = cbm.new_policy_value_model()
        new_weights = loss_weights.copy()
        new_weights['king_ray_exposure'] = 0.05
        new_weights['king_ring_pressure'] = 0.05
        new_weights['king_pawn_shield'] = 0.05
        new_weights['king_escape_square'] = 0.05
    
        new_weights['material'] = 0.05
        new_weights['policy_logits'] = 1.25
    
        cbm.set_head_weights(model, new_weights)
        
    # train
    train_loop(model)
