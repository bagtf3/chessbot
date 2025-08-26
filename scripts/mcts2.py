import chess
import time
import numpy as np
import pandas as pd

import chessbot.encoding as cbe

import tensorflow as tf
print("Built with CUDA?", tf.test.is_built_with_cuda())
print("GPUs available:", tf.config.list_physical_devices('GPU'))

model = tf.keras.models.load_model("C:/Users/Bryan/Data/chessbot_data/models/chess_model_multihead_v240_3.keras")
SF_LOC = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
PRED_CACHE = {}
PRIORS_CACHE = {}
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
    
    def expand(self, engine, set_priors=True):
        legal = list(self.board.legal_moves)
        for move in legal:
            if move not in self.children:
                new_board = self.board.copy()
                new_board.push(move)
                self.children[move] = MCTSNode(new_board, parent=self, move=move)

        # OPTIONAL: if you have a policy head, set node.priors here.
        # Otherwise, selection will uniform-fallback.
        # Example (uniform):
        if len(self.children):
            if set_priors:
                self.priors = self.set_priors(engine)
            else:
                self.priors = {m:1/len(self.children) for m in self.children.keys()}
    
    def set_priors(self, engine):
        if self.shredder_fen in PRIORS_CACHE:
            return PRIORS_CACHE[self.shredder_fen]
        
        sf_eval = engine.analyse(
            self.board, multipv=len(self.children), limit=chess.engine.Limit(depth=2),
            info=chess.engine.Info.ALL
        )
        
        moves, evals = [], []
        for e in sf_eval:
            cp = cbe.score_to_cp_white(e['score'])
            ve = cbe.pawns_to_winprob(cp if self.board.turn else -1*cp)
            moves.append(e['pv'][0])
            evals.append(ve)
        evals_p = cbe.winprob_to_policy(evals, temperature=0.6)
        priors = {m:e for m, e in zip(moves, evals_p)}
        PRIORS_CACHE[self.shredder_fen] = priors
        return priors

    def best_child(self):
        return max(self.children.values(), key=lambda c: c.average_value())
    

def predict_value_cached(node, model):
    key = node.board.shredder_fen()
    if key in PRED_CACHE:
        return PRED_CACHE[key]
    
    candidates = [get_board_state(node.board)]
        
    # predict and take the best based on score
    preds = model.predict(np.stack(candidates, axis=0), verbose=0)
    score = preds[model.output_names.index('value')].item()
    v = cbe.pawns_to_winprob(score)
    PRED_CACHE[key] = v
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


def simulate(engine, model, root, max_depth=128, c_puct=2.0):
    path = []
    node = root
    depth = 0
    
    # SELECTION + EXPANSION
    while depth < max_depth:
        root_noise = depth == 0
        path.append(node)

        if not node.children:
            node.expand(engine, set_priors=depth < 3)
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
    

    value = predict_value_cached(node, model)

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
        root.board, multipv=n_moves, limit=chess.engine.Limit(depth=1),
        info=chess.engine.Info.ALL
    )
    
    to_df = []
    for mv in info_list:
        move_name = str(mv['pv'][0])
        v = cbe.pawns_to_winprob(cbe.score_to_cp_white(mv['score']))
        to_df.append([move_name, v])
        
    out_df = pd.DataFrame(to_df, columns=['move', 'SF_eval'])
    return out_df.set_index('move')
    

def choose_move(root, engine, model, board, max_sims=1000):
    print("Thinking... ")
    start = time.time()
    df_list = []
    df_list.append(collect_value_data(engine, root))
    
    for s in range(max_sims):
        simulate(engine, model, root, max_depth=128)
        
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
        v_child = predict_value_cached(node, model)
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


def get_rank_file(pieces):
    ranks = [-1*(1 + p // 8) for p in pieces]
    files = [p % 8 for p in pieces]
    return ranks, files


def get_board_state(board):
    board_state = np.zeros((8, 8, 9), dtype=int)
    attack_map = np.zeros((8, 8), dtype=int)
    value_map = np.zeros((8, 8), dtype=int)
    
    value_lookup = {1:1, 2:3, 3:3, 4:5, 5:9, 6:12}
    
    for color in [True, False]:
        value = 1 if color == True else -1
        for piece in [1, 2, 3, 4, 5, 6]:
            pieces = list(board.pieces(piece, color))
            ranks, files = get_rank_file(pieces)
            board_state[ranks, files, piece-1] = value
            value_map[ranks, files] = value * value_lookup[piece]
                
            # check the attacks
            for p in pieces:
                # ignore pins
                if board.is_pinned(color, p):
                    continue
                squares_attacked = list(board.attacks(p))
                attack_ranks, attack_files = get_rank_file(squares_attacked)
                attack_map[attack_ranks, attack_files] += value
    
    # game metadata
    game_data = [
        1 * board.turn,
        1 * board.has_castling_rights(chess.WHITE),
        1 * board.has_castling_rights(chess.BLACK),
        1 * board.is_insufficient_material(),
        1 * board.can_claim_draw(),
        board.halfmove_clock / 100,
        1 * board.is_repetition(3),
        1 * board.is_check(),
        1 * board.is_stalemate(),
        attack_map.sum(),
        value_map.sum(),
        1 * board.turn,
        value_map[np.abs(value_map) == 1].sum(),
        value_map[np.abs(value_map) <= 3].sum(), 
        value_map[np.abs(value_map) > 3].sum(),
        len(list(board.legal_moves)) / 10
    ]
    
    meta_plane = np.tile(np.array(game_data).reshape(4, 4), (2, 2))
    
    board_state[:, :, -3] = attack_map
    board_state[:, :, -2] = value_map
    board_state[:, :, -1] = meta_plane
    
    return board_state


def make_move_with_model(board, model, color, return_state=True):
    # generate the candidate moves
    legal_moves = list(board.legal_moves)    
    candidates = np.zeros((len(legal_moves), 8, 8, 9), dtype=int)
    data = []
    checkmates = np.zeros(len(legal_moves))
    draws = np.zeros(len(legal_moves))
    for i, lm in enumerate(legal_moves):
        board.push(lm)
        if board.is_checkmate():
            checkmates[i] = 1
            
        threefold = board.can_claim_threefold_repetition()
        if threefold | board.is_stalemate() | board.can_claim_draw():
            draws[i] = 1
        fen = board.fen()
        data.append([lm, fen])
        
        candidate_state = get_board_state(board)
        candidates[i] = candidate_state
        
        board.pop()
        
    # predict and take the best based on score
    preds = model.predict(candidates, verbose=0)
    if isinstance(preds, list):
        outs = model.output_names
        preds = preds[outs.index('value')]
    
    # update based on hyper obvious checks
    if draws.sum():
        preds[draws == 1] = 0.0
        
    if checkmates.sum():
        if color:
            preds[checkmates == 1] = 1000
        else:
            preds[checkmates == 1] = -1000
            
    # best for white, worst for black    
    sel = np.argmax if color else np.argmin
    move = legal_moves[sel(preds)]
    board.push(move)
    
    #optionally return the state if training
    out = {"board": board}
    if return_state:
        out['board_state'] = candidates[sel(preds)]
        
    return out

#%%
from IPython.display import display, clear_output, SVG
import chess.svg, chess.pgn, random
import chessbot.utils as cbu

bot_color = np.random.uniform() < 0.5
board = chess.Board()
board = cbu.random_init(4)
root = MCTSNode(board)
data = []
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
    #v = predict_value_cached_priors(root, engine)
    while not board.is_game_over():
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=board, flipped=not bot_color)))
        
        sf_eval = engine.analyse(
            board, multipv=10, limit=chess.engine.Limit(depth=1),
            info=chess.engine.Info.ALL
        )
        
        if board.turn == bot_color:
            best_move, df, root = choose_move(root, engine, model, board, max_sims=100)
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
            time.sleep(0.5)
            sf_move = sf_eval[0]['pv'][0]
            san = board.san(sf_move)
            print(f"\nStockfish plays {san}\n")
            #board.push(random.choice(list(board.legal_moves)))
            board.push(sf_move)
            root = advance_root(root, sf_move)





