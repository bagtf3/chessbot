import chess
from collections import defaultdict
import time
import numpy as np
import pandas as pd

import chessbot.encoding as cbe
import chessbot.utils as cbu


SF_LOC = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
PRED_CACHE = {}

#%%
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.shredder_fen = board.shredder_fen()
        self.priors = {}

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def average_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.5
    
    def expand(self):
        legal = list(self.board.legal_moves)
        for move in legal:
            if move not in self.children:
                new_board = self.board.copy()
                new_board.push(move)
                self.children[move] = MCTSNode(new_board, parent=self, move=move)

        # OPTIONAL: if you have a policy head, set node.priors here.
        # Otherwise, selection will uniform-fallback.
        # Example (uniform):
        n = len(self.children) or 1
        self.priors = {m: 1.0 / n for m in self.children}

    def best_child(self):
        return max(self.children.values(), key=lambda c: c.average_value())
    

def predict_value_cached(node, engine):
    key = node.board.shredder_fen()
    if key in PRED_CACHE:
        return PRED_CACHE[key]
    
    score = engine.analyse(node.board, limit=chess.engine.Limit(depth=5))
    v = cbe.pawns_to_winprob(cbe.score_to_cp_white(score['score']))
    PRED_CACHE[key] = v
    return v


def predict_value_cached_priors(node, engine):
    key = node.shredder_fen
    need_current_pred = key not in PRED_CACHE.keys()
    has_children = node.children
    
    if has_children:
        need_priors = hasattr(node, "priors") 
        need_priors = need_priors and node.children
        need_priors = need_priors and len(set(node.priors.values())) < 2
        
    else:
        need_priors = False
    
    if need_priors:
        all_priors_cached = True
        moves, evals = [], []
        eval_sum = 0
        for m, n in node.children.items():
            n_eval = PRED_CACHE.get(n.shredder_fen, None)
            if n_eval is None:
                all_priors_cached = False
                break
            else:
                moves.append(m)
                evals.append(n_eval)
                eval_sum += n_eval
     
    run_sf = need_current_pred or (need_priors and not all_priors_cached)
    
    if run_sf:
        n_moves = len(list(node.board.legal_moves))
        # might be a checkmate or something
        if not n_moves:
            return predict_value_cached(node, engine)
        
        sf_eval = engine.analyse(
            node.board, multipv=n_moves, limit=chess.engine.Limit(depth=5),
            info=chess.engine.Info.ALL
        )
    
    if need_current_pred:
        v = cbe.pawns_to_winprob(cbe.score_to_cp_white(sf_eval[0]['score']))
        PRED_CACHE[node.shredder_fen] = v
        
    else:
        v = PRED_CACHE[node.shredder_fen]
    
    if need_priors:
        if all_priors_cached:
            if eval_sum == 0:
                evals = [1.0 for _ in evals]
                eval_sum = len(evals)
            priors = {m:e/eval_sum for m, e in zip(moves, evals)}
            node.priors = priors
            
        else:
            moves, evals = [], []
            eval_sum = 0
            for e in sf_eval:
                ve = cbe.pawns_to_winprob(cbe.score_to_cp_white(e['score']))
            
                moves.append(e['pv'][0])
                evals.append(ve)
                eval_sum += ve
                
            if eval_sum == 0:
                evals = [1.0 for _ in evals]
                eval_sum = len(evals)
                
            priors = {m:e/eval_sum for m, e in zip(moves, evals)}
            node.priors = priors
            
    return v


def add_dirichlet_noise(priors, epsilon=0.25, alpha=0.3):
    moves = list(priors.keys())
    n = len(moves)
    noise = np.random.dirichlet([alpha] * n)
    return {m: (1 - epsilon) * priors[m] + epsilon * n_
            for m, n_ in zip(moves, noise)}


def select_child_puct(
    node, prefer_higher=True, c_puct=1.0, root_noise=False, epsilon=0.25, alpha=0.3
):
    """PUCT selection with optional Dirichlet noise at the root."""
    kids_items = list(node.children.items())
    if not kids_items:
        return None

    # Initialize priors if not set
    if not hasattr(node, "priors") or node.priors is None:
        n = len(kids_items)
        node.priors = {m: 1.0 / n for m, _ in kids_items}

        # Add root noise if requested
        if root_noise:
            node.priors = add_dirichlet_noise(node.priors, epsilon=epsilon, alpha=alpha)

    N = max(1, node.visits)  # parent visits
    best_child, best_score = None, -1e18
    for move, child in kids_items:
        Q = child.average_value() if prefer_higher else 1 - child.average_value()
        P = node.priors.get(move, 1.0 / len(kids_items))
        U = c_puct * P * (np.sqrt(N) / (1.0 + child.visits))
        score = Q + U
        if score > best_score:
            best_score, best_child = score, child
    return best_child


def simulate(engine, root, max_depth=128, c_puct=1.5):
    path = []
    node = root
    depth = 0
    
    # SELECTION + EXPANSION
    while depth < max_depth:
        root_noise = depth == 0
        path.append(node)

        if not node.children:
            node.expand()
            node = select_child_puct(
                node, prefer_higher=node.board.turn,
                c_puct=c_puct, root_noise=root_noise
            )
            
            break
        
        else:
            node = select_child_puct(
                node, prefer_higher=node.board.turn,
                c_puct=c_puct, root_noise=root_noise
            )
        
        # stop here if we reached a terminal state
        if node.board.is_game_over():
            break
        depth += 1
    
    if depth < 3:
        value = predict_value_cached_priors(node, engine)
    else:
        value = predict_value_cached(node, engine)

    # BACKPROPAGATION
    for i, node in enumerate(reversed(path)):
        node.visits += 1
        node.value_sum += value


def collect_policy_data(root, n_sims):
    to_df = []
    for move, child in root.children.items():
        to_df.append([str(move), child.average_value(), child.visits])
        
    cols = ['move', f'Q_{n_sims}', f'visits_{n_sims}']
    out_df = pd.DataFrame(to_df, columns=cols).set_index('move')
    return out_df


def collect_value_data(engine, root):
    n_moves = len(list(root.board.legal_moves))
    info_list = engine.analyse(
        root.board, multipv=n_moves, limit=chess.engine.Limit(depth=10),
        info=chess.engine.Info.ALL
    )
    
    to_df = []
    for mv in info_list:
        move_name = str(mv['pv'][0])
        v = cbe.pawns_to_winprob(cbe.score_to_cp_white(mv['score']))
        to_df.append([move_name, v])
        
    out_df = pd.DataFrame(to_df, columns=['move', 'SF_eval'])
    return out_df.set_index('move')
    

def choose_move(root, engine, board, max_sims=1000):
    print("Thinking... ")
    start = time.time()
    df_list = []
    df_list.append(collect_value_data(engine, root))
    
    for s in range(max_sims):
        simulate(engine, root, max_depth=128)
        
        if s+1 in [100, 200, 500, 1000]:
            df_list.append(collect_policy_data(root, s+1))
    
    stop = time.time()
    print(f"Completed {s} simulations in {round(stop-start, 2)} seconds")
    df = pd.concat(df_list, axis=1).sort_values("SF_eval")
    
    # Sort children by visit count (descending)
    sorted_children = sorted(root.children.items(),
        key=lambda x: x[1].visits,
        reverse=True
    )
    
    # Print top 5 candidates with SAN and model value of the child position
    print("\nTop candidate moves:")
    for move, node in sorted_children[:5]:
        san = root.board.san(move)
        v_child = predict_value_cached(node, engine)
        print(f"{move.uci():<6} ({san})  visits={node.visits:<4} "
              f"Q={node.average_value():.3f}  V(model)={v_child:.3f}")

    # Pick the best move (highest visits)
    best_move, best_node = sorted_children[0]
    best_san = root.board.san(best_move)
    print(f"\nChosen move: {best_move.uci()} ({best_san})  "
          f"(visits={best_node.visits}, Q={best_node.average_value():.3f})")
    
    return best_move, df, root


def advance_root(root, move):
    """Advance the root of the MCTS tree to the child that matches `move`."""
    if move in root.children:
        new_root = root.children[move]
        new_root.parent = None  # cut off old parent
        return new_root
    else:
        # Tree has no record of this move, start fresh
        return MCTSNode(root.board.copy())


#%%
import time
from IPython.display import display, clear_output, SVG
import chess.svg, chess.pgn

bot_color = np.random.uniform() < 0.5
board = chess.Board()
root = MCTSNode(board)
data = []
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
    while not board.is_game_over():
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=board, flipped=not bot_color)))
        
        sf_eval = engine.analyse(
            board, multipv=10, limit=chess.engine.Limit(depth=1),
            info=chess.engine.Info.ALL
        )
        
        if board.turn == bot_color:
            best_move, df, root = choose_move(root, engine, board, max_sims=1000)
            data.append(df)
            root = advance_root(root, best_move)
            print("\nStockfish top 3 moves: ")
            moves = sf_eval[:3]
            for i, mv in enumerate(moves):
                san = board.san(mv['pv'][0])
                sc = cbe.pawns_to_winprob(cbe.score_to_cp_white(mv['score']))
                
                print(f"{i+1}. {san}: {np.round(sc, 3)}")
                
            print()
            board.push(best_move)
            
        else:
            sf_move = sf_eval[0]['pv'][0]
            san = board.san(sf_move)
            print(f"\nStockfish plays {san}\n")
            time.sleep(0.5)
            board.push(sf_move)
            root = advance_root(root, sf_move)
        
#%%
# import time
# from IPython.display import display, clear_output, SVG
# import chess.svg, chess.pgn
# board = cbu.random_init(3)
# clear_output(wait=True)
# display(SVG(chess.svg.board(board=board, flipped=not root.board.turn)))

# with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
#     best_move, df = choose_move(engine, board)
# #PRED_CACHE = {}
# def predict_value_cached(model, board):
#     key = board.shredder_fen()
#     if key in PRED_CACHE:
#         return PRED_CACHE[key]
#     x = get_board_state(board).reshape(1, 8, 8, 9)
#     out = model.predict(x, verbose=0)
#     v = out[model.output_names.index('value')].item()
#     PRED_CACHE[key] = v
#     return v

# def choose_move(board, max_sims=1500, max_time=45):
#     root = MCTSNode(board.copy())
    
#     print("Thinking... ")
#     start = time.time()
#     n_sims = 0
#     while time.time() - start < max_time:
#         simulate(engine, root, max_depth=64)
#         n_sims+=1
        
#         # check to see if there is a clear winner that will not likely be beaten
#         if n_sims % 100 == 0:
#             if n_sims > 399:
#                 children = [v for k, v in root.children.items()]
#                 visits = sorted([c.visits for c in children])
                
#                 if visits[-1] >= 0.5* n_sims:
#                     break
#                 if (visits[-1] >= 0.4*n_sims) and (visits[-1] > 1.5*visits[-2]):
#                     break
#                 if visits[-1]-visits[-2] > 0.8 * (max_sims-n_sims):
#                     break
        
#         if n_sims >= max_sims:
#             break
            
#     stop = time.time()
#     print(f"Completed {n_sims} simulations in {round(stop-start, 2)} seconds")
    
#     if not root.children:
#         print("No legal moves.")
#         return None
    
#     # Sort children by visit count (descending)
#     sorted_children = sorted(root.children.items(),
#         key=lambda x: x[1].visits,
#         reverse=True
#     )

#     # Print top 5 candidates with SAN and model value of the child position
#     print("\nTop candidate moves:")
#     for move, node in sorted_children[:5]:
#         san = root.board.san(move)
#         v_child = predict_value_cached(model, node.board)*10
#         print(f"{move.uci():<6} ({san})  visits={node.visits:<4} "
#               f"Q={node.average_value():.3f}  V(model)={v_child:.3f}")

#     # Pick the best move (highest visits)
#     best_move, best_node = sorted_children[0]
#     best_san = root.board.san(best_move)
#     print(f"\nChosen move: {best_move.uci()} ({best_san})  "
#           f"(visits={best_node.visits}, Q={best_node.average_value():.3f})")
    
#     return best_move

def sort_candidates(evals, clr):
    if clr: 
        ranked = sorted(evals, key=lambda x: x[1], reverse=True)
    else:
        ranked = sorted(evals, key=lambda x: x[1], reverse5e=False)
        
    return ranked
    
def eval_current_state(board, model):
    current_state = np.zeros((1, 8, 8, 9))
    current_state[0] = get_board_state(board)
    current_pred = model.predict(current_state, verbose=0)
    
    if isinstance(current_pred, list):
        outs = model.output_names
        current_eval = current_pred[outs.index('value')]
        current_cmr = current_pred[outs.index('checkmate_radar')]
        current_queen_en_prise = current_pred[outs.index('queen_en_prise')]
        current_material = current_pred[outs.index('material')]
        
    print(f"Current Evaluation: {np.round(current_eval.item()*10, 4)}")
    print(f"Current CMR: {np.round(current_cmr.item(), 4)}")
    print(f"Current Materal: {np.round(current_material[0]*10, 4)}")
    print(f"Current QEP: {np.round(current_queen_en_prise[0], 4)}")
    
    

    
    



#%%
#model = tf.keras.models.load_model("C:/Users/Bryan/repos/chessbot/chess_model_multihead_v240_3.keras")
