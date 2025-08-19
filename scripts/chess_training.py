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

sf_loc = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"

import tensorflow as tf
print("Built with CUDA?", tf.test.is_built_with_cuda())
print("GPUs available:", tf.config.list_physical_devices('GPU'))


def plot_training_progress(all_evals):
    """
    Plots predicted vs true for each head in 3x3 grids, 9 at a time.
    """
    cols = list(all_evals.columns)
    chunk_size = 9
    n_chunks = math.ceil(len(cols) / chunk_size)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk = cols[start:end]

        fig, axes = plt.subplots(3, 3, figsize=(14, 8))
        axes = axes.flatten()

        for i, col in enumerate(chunk):
            ax = axes[i]
            ax.plot(all_evals[col].values, label=col)
            ax.set_title(col)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide unused axes
        for i in range(len(chunk), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()


def plot_pred_vs_true_grid(model, preds_list, y_true_dict, sample_limit=None):
    """
    Plots predicted vs true for each head in 3x3 grids, 9 at a time.
    """
    og_names = list(model.output_names)
    chunk_size = 9
    n_chunks = math.ceil(len(og_names) / chunk_size)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_names = og_names[start:end]

        fig, axes = plt.subplots(3, 3, figsize=(14, 8))
        axes = axes.flatten()

        for ax, name in zip(axes, chunk_names):
            if name not in y_true_dict:
                ax.set_visible(False)
                continue

            y_pred = np.asarray(preds_list[og_names.index(name)])
            y_true = np.asarray(y_true_dict[name])

            y_true = y_true.reshape(len(y_true), -1)
            y_pred = y_pred.reshape(len(y_pred), -1)

            if sample_limit is not None:
                y_true = y_true[:sample_limit]
                y_pred = y_pred[:sample_limit]

            # Scatter for each dim in the head
            for i in range(y_true.shape[1]):
                ax.scatter(y_true[:, i], y_pred[:, i], s=5, alpha=0.5)

            lo = float(min(y_true.min(), y_pred.min()))
            hi = float(max(y_true.max(), y_pred.max()))
            ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)

            ax.set_title(name)
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots in last chunk
        for i in range(len(chunk_names), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()
    

def score_game_data(model, X, Y_batch):
    outs = model.output_names
    preds = model.predict(X)

    plot_pred_vs_true_grid(model, preds, Y_batch, sample_limit=None)
    
    cols = ['total_loss'] + outs
    eval_df = pd.DataFrame(model.evaluate(X, Y_batch, verbose=0), index=cols).T
    
    return eval_df


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
    

def get_all_board_features(board):
    all_feats = all_king_exposure_features(board)
    all_feats.update(all_piece_features(board))
    
    wt, blk = get_piece_value_sum(board)
    all_feats['material'] = [wt, blk]
    
    return all_feats


def random_init(board, halfmoves=5):
    for _ in range(halfmoves):
        moves = list(board.legal_moves)
        move = np.random.choice(moves, 1)[0]
        board.push(move)
        
    return board


def build_training_arrays(data):
    head_names = data[0][1].keys()
    X = np.stack([sample[0] for sample in data], axis=0)

    Y = {}
    for head in head_names:
        vals = [sample[1][head] for sample in data]
        Y[head] = np.array(vals, dtype=np.float32)

    return X, Y


def get_board_val(board_score):
    # always look from whites perspective
    from_white = board_score.white()
    
    #check for mates
    if from_white.is_mate():
        return from_white.score(mate_score=100) / 100
    
    else:
        return np.clip(from_white.score() / 1000, -0.95, 0.95)
    
    
def analyze_board(board, engine):
    position_data = []
    
    # get some training data
    possible_moves = engine.analyse(
        board, multipv=10, limit=chess.engine.Limit(depth=12),
        info=chess.engine.Info.ALL
    )
    
    for move in possible_moves:
        board.push(move['pv'][0])
        this_X = get_board_state(board)
        
        this_Y = get_all_board_features(board)
        this_Y['value'] = get_board_val(move['score'])
        
        checkmate_radar = 0
        if move['score'].is_mate():
            if np.abs(move['score'].white().mate()) <= 5:
                checkmate_radar = move['score'].white().score(mate_score=6) / 2
        
        this_Y['checkmate_radar'] = checkmate_radar
        position_data.append([this_X, this_Y])
        board.pop()
        
    return position_data

#%%
model = tf.keras.models.load_model("C:/Users/Bryan/repos/chessbot/chess_model_multihead_v240_3.keras")

pickle_dir = "C:/Users/Bryan/repos/chessbot/training_data/multiheaded_bot_tournament"
pickles = os.listdir(pickle_dir)

all_bot_data = []
for p in pickles:
    file_path = os.path.join(pickle_dir, p)
    with open(file_path, "rb") as f:
        bot_data = pickle.load(f)
    all_bot_data += bot_data[:]
        
random.shuffle(all_bot_data)

data = []
training_data = []
all_evals = pd.DataFrame()
n_trains = 0
for i, bd in enumerate(all_bot_data):
    game = chess.pgn.read_game(io.StringIO(bd['pgn']))
    print(f"###### Game {i} ######")
    print(str(game))
    print(f"\nCurrent length of data: {len(data)}")
    print("\n#######################\n")
    
    # Create a board from it
    board = chess.Board()
    with chess.engine.SimpleEngine.popen_uci(sf_loc) as engine:
        data += analyze_board(board, engine)
        for move in game.mainline_moves():
            board.push(move)
            
            if board.is_game_over():
                break
            
            data += analyze_board(board, engine)
            
            if len(data) > 5000:
                X, Y_batch = build_training_arrays(data)
                eval_df = score_game_data(model, X, Y_batch)
                all_evals = pd.concat([all_evals, eval_df])
                plot_training_progress(all_evals)
                
                model.fit(X, Y_batch, epochs=2, batch_size=256)
                model.save("C:/Users/Bryan/repos/chessbot/chess_model_multihead_v240_3.keras")
                training_data += data[:]
                data = []

# # Params
chunk_size = 100000
output_dir = "C:/Users/Bryan/repos/chessbot/training_data/multiheaded_bot_pickled"
os.makedirs(output_dir, exist_ok=True)

# Shuffle the data so each chunk is random
random.shuffle(training_data)

# Split into chunks
for i in range(0, len(training_data), chunk_size):
    chunk = training_data[i:i+chunk_size]
    file_path = os.path.join(output_dir, f"chunk_{i}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(chunk, f)
    print(f"Saved {file_path} ({len(chunk)} items)")
#%%
def make_random_move(board):
    moves = list(board.legal_moves)
    board.push(np.random.choice(moves))
    return {"board": board}


def make_move_with_stockfish(board, engine, depth=1, top_n=0):
    legal_moves = engine.analyse(
        board, multipv=5, limit=chess.engine.Limit(depth=depth),
        info=chess.engine.Info.ALL
    )
    
    if top_n:
        legal_moves = legal_moves[:top_n]
        move = np.random.choice(legal_moves)['pv'][0]
        
    else:
        move = legal_moves[0]['pv'][0]
    
    board.push(move)
    
    # recover the score to save time
    sf_score = [l['score'] for l in legal_moves if l['pv'][0] == move][0]
    return {"board": board, "score": get_board_val(sf_score)}
    
    
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
    

class Analyzer(object):
    def __init__(self, engine, depth):
        self.engine = engine
        self.depth = depth
    
    def analyze(self, board):
        sf = self.engine.analyse(
            board, limit=chess.engine.Limit(depth=self.depth)
        )
        
        return get_board_val(sf["score"])
        
        
class ChessPlayer(object):
    def __init__(self, color, analyzer=None, return_data=False):
        self.return_data = return_data
        self.color = color
        self.analyzer = analyzer
        
    def analyze_board(self, board):
        if self.analyzer is None:
            raise Exception("No Analyzer given!")
        return self.analyzer.analyze(board)
    
    def add_return_data(self, result):
        if 'board_state' not in result.keys():
            result['board_state'] = get_board_state(result['board'])
        
        feats = get_all_board_features(result['board'])
        if 'score' not in result.keys():
            feats['value'] = self.analyze_board(result['board'])
        else:
            feats['value'] = result['score']
        
        result['score'] = feats
        
        return result
    
    def make_move(self, board):
        result = self._make_move(board)
        if self.return_data:
            result = self.add_return_data(result)
        return result
    
    def _make_move(self, board):
        raise NotImplementedError
        
        
class RandomPlayer(ChessPlayer):
    def _make_move(self, board):
        return make_random_move(board)


class NNPlayer(ChessPlayer):
    def __init__(self, model, color, analyzer=None, return_data=False):
        super().__init__(color, analyzer, return_data)
        self.model = model
    
    def _make_move(self, board):
        # can clean this up later. like having the function available while
        # tinkering. will be cleaner when converted to a method.
        return make_move_with_model(
            board, self.model, self.color, return_state=self.return_data
        )
            

class StockfishPlayer(ChessPlayer):
    def __init__(self, engine, depth=1, top_n=0, return_data=False, **kwargs):
        self.engine = engine
        self.depth = depth
        self.top_n = top_n
        self.return_data = return_data

    def _make_move(self, board):
        return make_move_with_stockfish(
            board, self.engine, depth=self.depth, top_n=self.top_n
        )


#%%
import time
from IPython.display import display, clear_output, SVG
import chess.svg, chess.pgn

def random_board():
    b = chess.Board()
    for _ in range(5):
        lms = list(b.legal_moves)
        random.shuffle(lms)
        b.push(lms[0])
    return b


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
    
    return current_eval

#%%

VAL = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}

def mvv_lva(board, move):
    cap = board.piece_at(move.to_square)
    att = board.piece_at(move.from_square)
    if not cap or not att: return 0.0
    return 0.02 * (VAL[cap.piece_type] - VAL[att.piece_type])


def is_check_after(board, move):
    board.push(move); chk = board.is_check(); board.pop(); return chk
    

def see_gain_simple(board, move):
    """Tiny SEE proxy: gained piece value minus attacker value if immediately recaptured."""
    if not board.is_capture(move): return 0.0
    att = board.piece_at(move.from_square); cap = board.piece_at(move.to_square)
    if not att or not cap: return 0.0
    # very crude: value captured - value attacker (optimistic but bounded)
    return max(0.0, 0.05 * (VAL[cap.piece_type] - VAL[att.piece_type]))

CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
EXT_CENTER = CENTER | {chess.C3, chess.C4, chess.C5, chess.C6,
                       chess.D3, chess.E3, chess.D6, chess.E6,
                       chess.F3, chess.F4, chess.F5, chess.F6}


def center_push_bonus(move):
    return 0.03 if move.to_square in CENTER else (0.015 if move.to_square in EXT_CENTER else 0.0)


def develop_minor_from_backrank(board, move):
    p = board.piece_at(move.from_square)
    if not p or p.piece_type not in (chess.KNIGHT, chess.BISHOP): return 0.0
    rank0 = 1 if p.color == chess.WHITE else 6  # 0-indexed ranks
    return 0.03 if chess.square_rank(move.from_square) == rank0 else 0.0


def promotion_bonus(move):
    return 0.1 if move.promotion else 0.0


def castling_bonus(move):
    return 0.08 if board.is_castling(move) else 0.0


def free_hanging_capture(board, move):
    """Small nudge if we capture a truly hanging piece (undefended)."""
    if not board.is_capture(move): return 0.0
    # simple: after our capture, can opponent recapture that square?
    board.push(move)
    their_attackers = board.attackers(not board.turn, move.to_square)
    board.pop()
    return 0.05 if len(their_attackers) == 0 else 0.0


def heuristic_priors(board, moves, temperature=1.0, eps=1e-3):
    """
    Returns dict: move -> prior probability (sums to 1).
    temperature<1.0 sharpens; >1.0 flattens.
    """
    scores = []
    if not moves:
        return {}
    
    for mv in moves:
        s = 0.0
        # Tactical
        s += mvv_lva(board, mv)
        s += 0.10 if is_check_after(board, mv) else 0.0
        s += see_gain_simple(board, mv)
        s += promotion_bonus(mv)
        s += castling_bonus(mv)

        # Positional-lite
        s += center_push_bonus(mv)
        s += develop_minor_from_backrank(board, mv)
        s += free_hanging_capture(board, mv)

        scores.append(s)

    # Softmax with temperature + floor
    if temperature <= 0: temperature = 1e-6
    mx = max(scores)
    logits = [ (sc - mx) / max(1e-6, temperature) for sc in scores ]  # stabilize
    exps = [ math.exp(x) for x in logits ]
    Z = sum(exps) + eps * len(exps)
    priors = { mv: (exps[i] + eps) / Z for i, mv in enumerate(moves) }
    return priors


def model_priors(board):
    moves = list(board.legal_moves)
    scores = []
    for mv in moves:
        board.push(mv)
        if board.is_game_over():
            scores.append(terminal_value_white(board))
        else:
            scores.append(predict_value_cached(model, board))
        board.pop()
    
    # normalize
    min_score = min(scores)
    scores = [s + min_score for s in scores]
    sum_score = sum(scores)
    scores = [s/sum_score for s in scores]
    priors = {m:s for m, s in zip(moves, scores)}
    return priors
    
    
#%%
#MCTS
import chess
from collections import defaultdict
import random
import numpy as np


class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def average_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
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


    def expand_with_priors(self, prior_type=None):
        legal_moves = list(self.board.legal_moves)
        
        if prior_type == 'model':
            priors = model_priors(self.board)
        else:
            priors = heuristic_priors(self.board, legal_moves, temperature=0.7, eps=1e-3)
        
        # Get raw scores for each move
        for move in legal_moves:
            if move not in self.children:
                new_board = self.board.copy()
                new_board.push(move)
                self.children[move] = MCTSNode(new_board, parent=self, move=move)
        
        # Normalize scores into probabilities
        self.priors = priors
        
    def best_child(self):
        return max(self.children.values(), key=lambda c: c.average_value())


#PRED_CACHE = {}
def predict_value_cached(model, board):
    key = board.shredder_fen()
    if key in PRED_CACHE:
        return PRED_CACHE[key]
    x = get_board_state(board).reshape(1, 8, 8, 9)
    out = model.predict(x, verbose=0)
    v = out[model.output_names.index('value')].item()
    PRED_CACHE[key] = v
    return v


def softmax(x):
    x = np.array(x)
    exps = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exps / np.sum(exps)


def select_child_softmax(node, tau=1.0):
    """Sample children by softmax over Q (average_value)."""
    kids = list(node.children.values())
    if not kids:
        return None
    q = np.array([k.average_value() for k in kids], dtype=float)
    # Stabilize: subtract max and divide by tau
    probs = softmax(q / max(1e-6, tau))
    return np.random.choice(kids, p=probs)


def select_child_puct(node, prefer_higher=True, c_puct=1.0):
    """PUCT with uniform priors (or attach your policy later)."""
    kids_items = list(node.children.items())
    if not kids_items:
        return None

    # Initialize uniform priors once per node
    if not hasattr(node, "priors") or node.priors is None:
        n = len(kids_items)
        node.priors = {m: 1.0 / n for m, _ in kids_items}

    N = max(1, node.visits)  # parent visits
    best_child, best_score = None, -1e18
    for move, child in kids_items:
        Q = child.average_value() if prefer_higher else -1*child.average_value()
        P = node.priors.get(move, 1.0 / len(kids_items))
        U = c_puct * P * (np.sqrt(N) / (1.0 + child.visits))
        score = Q + U
        if score > best_score:
            best_score, best_child = score, child
    return best_child


def terminal_value_white(board: chess.Board):
    """+1 if white wins, -1 if black wins, 0 draw; None if non-terminal."""
    if not board.is_game_over():
        return None
    outcome = board.outcome()
    if outcome.winner is None:
        val = 0.0
    val = 1.0 if outcome.winner == chess.WHITE else -1.0
    PRED_CACHE[board.shredder_fen()] = val
    return val


def simulate(model, root, max_depth=64):
    path = []
    node = root
    depth = 0
    
    # SELECTION + EXPANSION
    while depth < max_depth:
        path.append(node)

        if not node.children:
            node.expand_with_priors(prior_type="heuristic")
            
            node = select_child_puct(node, prefer_higher=node.board.turn, c_puct=2.0)
            break  # expand only 1 node per simulation
        
        # OPTION 1: PUCT
        node = select_child_puct(node, prefer_higher=node.board.turn, c_puct=2.0)
        # stop here if we reached a terminal state
        if node.board.is_game_over():
            break

        depth += 1
    
    value = predict_value_cached(model, node.board)

    # BACKPROPAGATION
    for i, node in enumerate(reversed(path)):
        node.visits += 1
        node.value_sum += value
        

def choose_move(board, max_sims=1500, max_time=45):
    root = MCTSNode(board.copy())
    
    print("Thinking... ")
    start = time.time()
    n_sims = 0
    while time.time() - start < max_time:
        simulate(model, root, max_depth=64)
        n_sims+=1
        
        # check to see if there is a clear winner that will not likely be beaten
        if n_sims % 100 == 0:
            if n_sims > 399:
                children = [v for k, v in root.children.items()]
                visits = sorted([c.visits for c in children])
                
                if visits[-1] >= 0.5* n_sims:
                    break
                if (visits[-1] >= 0.4*n_sims) and (visits[-1] > 1.5*visits[-2]):
                    break
                if visits[-1]-visits[-2] > 0.8 * (max_sims-n_sims):
                    break
        
        if n_sims >= max_sims:
            break
            
    stop = time.time()
    print(f"Completed {n_sims} simulations in {round(stop-start, 2)} seconds")
    
    if not root.children:
        print("No legal moves.")
        return None
    
    # Sort children by visit count (descending)
    sorted_children = sorted(root.children.items(),
        key=lambda x: x[1].visits,
        reverse=True
    )

    # Print top 5 candidates with SAN and model value of the child position
    print("\nTop candidate moves:")
    for move, node in sorted_children[:5]:
        san = root.board.san(move)
        v_child = predict_value_cached(model, node.board)*10
        print(f"{move.uci():<6} ({san})  visits={node.visits:<4} "
              f"Q={node.average_value():.3f}  V(model)={v_child:.3f}")

    # Pick the best move (highest visits)
    best_move, best_node = sorted_children[0]
    best_san = root.board.san(best_move)
    print(f"\nChosen move: {best_move.uci()} ({best_san})  "
          f"(visits={best_node.visits}, Q={best_node.average_value():.3f})")
    
    return best_move


#%%
#model = tf.keras.models.load_model("C:/Users/Bryan/repos/chessbot/chess_model_multihead_v240_3.keras")

bot_color = np.random.uniform() < 0.5
board = chess.Board()

with chess.engine.SimpleEngine.popen_uci(sf_loc) as engine:
    while not board.is_game_over():
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=board, flipped=not bot_color)))
        
        current_state = np.zeros((1, 8, 8, 9))
        current_state[0] = get_board_state(board)
        current_pred = model.predict(current_state, verbose=0)
        
        if isinstance(current_pred, list):
            outs = model.output_names
            current_eval = current_pred[outs.index('value')]
            current_cmr = current_pred[outs.index('checkmate_radar')]
            current_queen_en_prise = current_pred[outs.index('queen_en_prise')]
            current_material = current_pred[outs.index('material')]
        
        sf_eval = engine.analyse(
            board, multipv=10, limit=chess.engine.Limit(depth=2),
            info=chess.engine.Info.ALL
        )
        
        sf_score = np.round(get_board_val(sf_eval[0]['score']) * 10, 3)
        print(f"Current Evaluation: {np.round(current_eval.item()*10, 4)} vs Stockfish {sf_score}")
        # print(f"Current CMR: {np.round(current_cmr.item(), 4)}")
        # print(f"Current Materal: {np.round(current_material[0]*10, 4)}")
        # print(f"Current QEP: {np.round(current_queen_en_prise[0], 4)}")
        
        if board.turn == bot_color:
            best_move = choose_move(board, max_sims=1000, max_time=60)
            print("\nStockfish top 3 moves: ")
            moves = sf_eval[:3]
            for i, mv in enumerate(moves):
                san = board.san(mv['pv'][0])
                print(f"{i+1}. {san}: {np.round(10*get_board_val(mv['score']), 3)}")
                
            print()
            board.push(best_move)
            
        else:
            # print("\nYour move (e.g., e2e4): ", end="")
            # user_move = input().strip()
            
            # try:
            #     board.push_san(user_move) if len(user_move) <= 5 else board.push_uci(user_move)
                
            # except ValueError:
            #     print("Invalid move. Try again.")
            #     continue
            san = board.san(sf_eval[0]['pv'][0])
            print(f"\nStockfish plays {san}\n")
            time.sleep(0.3)
            board.push(sf_eval[0]['pv'][0])
        

clear_output(wait=True)
display(SVG(chess.svg.board(board=board, flipped=bot_color)))
time.sleep(0.2)
#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

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


def policy_ce_unmasked(y_true, y_pred):
    # y_true, y_pred: [B,8,8,73]
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    # From logits = True → applies softmax+log inside
    return keras.losses.categorical_crossentropy(y_true_f, y_pred_f, from_logits=True)

model = build_chess_model(in_planes=23, width=128, n_blocks=12)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss={
        "policy_logits": policy_ce_unmasked,
        "value": keras.losses.BinaryCrossentropy(),
    },
    loss_weights={"policy_logits": 1.0, "value": 0.5},
    metrics={"value": [keras.metrics.MeanAbsoluteError(), keras.metrics.AUC(name="AUC")]}
)

#%%
def mirror_move(mv: chess.Move) -> chess.Move:
    """Mirror a move through the board center (promotion piece is unchanged)."""
    return chess.Move(
        chess.square_mirror(mv.from_square),
        chess.square_mirror(mv.to_square),
        promotion=mv.promotion
    )

all_X = []
all_Yp = []
all_Yv = []

board = chess.Board()
with chess.engine.SimpleEngine.popen_uci(sf_loc) as engine:
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
