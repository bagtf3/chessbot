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

import os, pickle, numpy as np, time
import chessbot.model as cbm

DATA_DIR =  "C:/Users/Bryan/Data/chessbot_data/training_data/policy_value_data"

model_path = os.path.join(cbm.MODEL_DIR, "value_policy_model_v0.h5")

if os.path.exists(model_path):
    model = cbm.load_model(model_path)
else:
    # model, loss_weights = cbm.new_policy_value_model()
    # new_weights = loss_weights.copy()
    # new_weights['king_ray_exposure'] = 0.05
    # new_weights['king_ring_pressure'] = 0.05
    # new_weights['king_pawn_shield'] = 0.05
    # new_weights['king_escape_square'] = 0.05

    # new_weights['material'] = 0.05
    # new_weights['policy_logits'] = 3.0
    # new_weights['legal_moves'] = 3.0
    # not_pawns = [k for k in new_weights.keys() if 'pawn' not in k]
    
    # for k in [k for k in not_pawns if 'hanging' in k]:
    #     new_weights[k] = 0.5
        
    # for k in [k for k in not_pawns if 'en_prise' in k]:
    #     new_weights[k] = 0.5
        
    # for k in [k for k in not_pawns if 'un_defended' in k]:
    #     new_weights[k] = 0.5
    #new_weights['value'] = 1.0
    # for k in [k for k in not_pawns if 'lower' in k]:
    #     new_weights[k] = 0.5
    # for k in [k for k in not_pawns if 'queen' in k]:
    #     new_weights[k] = 0.65
    # new_weights['legal_moves'] = 2.0
    # new_weights['policy_logits'] = 4.0
    # cbm.set_head_weights(model, new_weights)

all_evals = []
seen_already = []
n_trains = 0


while True:
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")])
    files = [f for f in files if f not in seen_already]
    if not files:
        print("No new batches yet... waiting")
        time.sleep(5)
        continue

    batch_file = files[0]  # grab the oldest
    path = os.path.join(DATA_DIR, batch_file)
    with open(path, "rb") as f:
        all_X, all_Y = pickle.load(f)

    # prep data
    X = np.stack(all_X, dtype=np.float32)
    Y = cbm.prepare_targets(all_Y)

    eval_df = cbu.score_game_data(model, X, Y)
    all_evals.append(eval_df)
    if n_trains % 5 == 4:
        cbu.plot_training_progress(pd.concat(all_evals[5:]))

    print(f"Training on {batch_file} -> X {X.shape}")
    
    model.fit(X, Y, batch_size=128, epochs=2, shuffle=True)
    n_trains += 1
    if n_trains % 50 == 0:
        model_loc = os.path.join(cbm.MODEL_DIR, f"value_policy_model_v{n_trains}.h5")
        print(f"Saving to {model_loc}")
        cbm.save_model(model, model_loc)
        
    seen_already.append(batch_file)
    os.remove(path)  # delete after training

