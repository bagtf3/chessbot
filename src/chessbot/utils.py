import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import math, random, time, pickle
            
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()

import seaborn as sns
from sklearn.metrics import confusion_matrix

import chess
import chess.engine

from chessbot import features as ft

import chess.svg
from IPython.display import SVG, display, clear_output

from pyfastchess import Board as fastboard
SF_LOC = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"


def show_board(board, flipped=False, sleep=0.1):
    clear_output(wait=True)
    display(SVG(chess.svg.board(board=board, flipped=flipped)))
    time.sleep(sleep)
    
    
def mirror_board(board):
    return board.mirror()


def mirror_move(mv: chess.Move) -> chess.Move:
    """Mirror a move through the board center (promotion piece is unchanged)."""
    return chess.Move(
        chess.square_mirror(mv.from_square),
        chess.square_mirror(mv.to_square),
        promotion=mv.promotion
    )


def make_random_move(board):
    moves = list(board.legal_moves)
    board.push(random.choice(moves))
    return board


def random_init(halfmoves=5):
    board = chess.Board()
    for _ in range(halfmoves):
        moves = list(board.legal_moves)
        # try again 
        if not len(moves):
            return random_init(halfmoves=halfmoves)
        
        board = make_random_move(board)
        
        # also try again
        if board.is_game_over():
            return random_init(halfmoves=halfmoves)
        
    return board


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
    return board


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


def plot_training_progress(all_evals, max_cols=4):
    """
    Plots training/eval metrics for each model output in a grid.
    Adapts rows/cols automatically, up to max_cols wide.
    """
    cols = list(all_evals.columns)

    # Always keep "value" last for consistency
    important = ['value']
    cols = [c for c in cols if c not in important] + important

    n_plots = len(cols)
    n_cols = min(max_cols, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        y = all_evals[col].values
        ax.plot(y, label=col)
        # moving average (MA5)
        ma = pd.Series(y).rolling(15, min_periods=1).mean().values
        ax.plot(ma, lw=2, alpha=0.6, label=f"{col} (MA5)")
        ax.set_title(col)
        ax.legend()

    # Hide any unused axes
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_pred_vs_true_grid(model, preds, y_true_dict):
    names = list(model.output_names)

    chunk_size = 9
    n_chunks = math.ceil(len(names) / chunk_size)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_names = names[start:end]

        fig, axes = plt.subplots(3, 3, figsize=(14, 8))
        axes = axes.flatten()

        for ax, name in zip(axes, chunk_names):
            if name not in y_true_dict:
                ax.set_visible(False)
                continue

            y_pred = np.asarray(preds[name])
            y_true = np.asarray(y_true_dict[name])

            # squeeze singleton dims
            if y_pred.ndim > 1 and y_pred.shape[-1] == 1:
                y_pred = y_pred.reshape(-1)
            if y_true.ndim > 1 and y_true.shape[-1] == 1 and name == "value":
                y_true = y_true.reshape(-1)

            if name == "value":
                # regression scatter
                yp = y_pred.reshape(-1)
                yt = y_true.reshape(-1)
                n = min(len(yp), len(yt))
                yp, yt = yp[:n], yt[:n]
                ax.scatter(yt, yp, s=8, alpha=0.5)
                lo = float(min(yt.min(), yp.min()))
                hi = float(max(yt.max(), yp.max()))
                ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
                ax.set_title("value")
                ax.set_xlabel("True")
                ax.set_ylabel("Pred")
                ax.grid(True, alpha=0.3)
                continue

            # classification heads: handle soft or sparse y_true
            pred_classes = y_pred.argmax(axis=1)

            if y_true.ndim == 2:
                true_classes = y_true.argmax(axis=1)
            else:
                true_classes = y_true.reshape(-1)

            pred_classes = pred_classes.reshape(-1)
            true_classes = true_classes.reshape(-1)

            n = min(len(true_classes), len(pred_classes))
            ax.scatter(true_classes[:n], pred_classes[:n], s=5, alpha=0.5)
            ax.set_title(name)
            ax.set_xlabel("True class")
            ax.set_ylabel("Pred class")
            ax.grid(True, alpha=0.3)

        for i in range(len(chunk_names), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()


def top_k_accuracy(y_true, y_pred, k=3):
    """
    Computes Top-K accuracy for classification heads.
    y_true : (N,) int labels
    y_pred : (N, C) logits or probs
    k      : how many top guesses to consider
    """
    # convert logits -> probs
    probs = tf.nn.softmax(y_pred, axis=-1).numpy()
    # indices of top-k per sample
    topk = np.argpartition(-probs, k, axis=1)[:, :k]
    # check if true label is in top-k
    correct = [y_true[i] in topk[i] for i in range(len(y_true))]
    return np.mean(correct)


def _to_sparse_labels(y):
    y = np.asarray(y)
    if y.ndim == 2:  # one-hot / soft
        return y.argmax(axis=1).astype(np.int64)
    return y.reshape(-1).astype(np.int64)


def _topk_from_logits(y_true_sparse, y_pred_logits, k=1):
    y_pred = np.asarray(y_pred_logits)
    topk = np.argpartition(-y_pred, kth=min(k, y_pred.shape[1]-1), axis=1)[:, :k]
    # count hits
    hits = (topk == y_true_sparse[:, None]).any(axis=1)
    return float(hits.mean())


def score_game_data(model, X, Y_batch):
    raw_preds = model.predict(X, batch_size=256, verbose=0)
    preds = {name: raw_preds[i] for i, name in enumerate(model.output_names)}

    # Plot overview grid
    plot_pred_vs_true_grid(model, preds, Y_batch)

    # Keras evaluate -> dataframe row
    cols = ['total_loss'] + model.output_names
    eval_df = pd.DataFrame(model.evaluate(X, Y_batch, verbose=0), index=cols).T

    # ----- Value head metrics -----
    yt_val = np.asarray(Y_batch['value']).reshape(-1)
    yp_val = np.asarray(preds['value']).reshape(-1)

    val_mse = float(np.mean((yt_val - yp_val) ** 2))

    # safe correlation (avoid NaNs if std=0)
    yt_std = yt_val.std()
    yp_std = yp_val.std()
    if yt_val.size > 1 and yt_std > 0 and yp_std > 0:
        corr = float(np.corrcoef(yt_val, yp_val)[0, 1])
    else:
        corr = 0.0

    # ----- Classification metrics -----
    cls_metrics = {}
    for head in [n for n in model.output_names if n != "value"]:
        y_true_sparse = _to_sparse_labels(Y_batch[head])
        y_pred_sparse = np.asarray(preds[head]).argmax(axis=1)
        acc = float((y_true_sparse == y_pred_sparse).mean())
        cls_metrics[head] = acc

    print("\n=== Extra Metrics ===")
    print(f"Value : MSE={val_mse:.4f}, Corr={corr:.3f}")

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
    all_feats = ft.all_king_exposure_features(board)
    all_feats.update(ft.all_piece_features(board))
    
    wt, blk = ft.get_piece_value_sum(board)
    all_feats['material'] = [wt, blk]
    
    return all_feats


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


from IPython.display import display, clear_output, SVG
import chess.svg, chess.pgn




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


PRED_CACHE = {}
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
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    y = np.exp(x)
    s = float(np.sum(y))
    return y / s if s > 0 else np.full_like(y, 1.0 / len(y))


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
        

def choose_move(board, model, max_sims=1500, max_time=45):
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


OPENING_BOOK = {
    "Ruy Lopez, Morphy Defense": [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"
    ],
    "Italian Game (Giuoco Piano)": [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"
    ],
    "Scotch Game": [
        "e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "g8f6"
    ],
    "Sicilian Defense, Najdorf": [
        "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"
    ],
    "Sicilian Defense, Dragon": [
        "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"
    ],
    "French Defense, Classical": [
        "e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7"
    ],
    "Caro-Kann, Advance": [
        "e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5", "c2c4", "e7e6"
    ],
    "Caro-Kann, Classical": [
        "e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"
    ],
    "Queen's Gambit Declined": [
        "d2d4", "d7d5", "c2c4", "e7e6", "g1f3", "g8f6", "b1c3", "c7c6"
    ],
    "Queen's Gambit Accepted": [
        "d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6"
    ],
    "Slav Defense": [
        "d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4"
    ],
    "Nimzo-Indian Defense": [
        "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"
    ],
    "King's Indian Defense": [
        "d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"
    ],
    "Grünfeld Defense": [
        "d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"
    ],
    "London System": [
        "d2d4", "d7d5", "c1f4", "g8f6", "e2e3", "c7c5", "c2c3", "b8c6"
    ],
    "English Opening, Four Knights": [
        "c2c4", "e7e5", "g1f3", "b8c6", "g2g3", "g8f6", "f1g2", "f8c5"
    ],
    "English Opening, Symmetrical": [
        "c2c4", "c7c5", "g1f3", "g8f6", "d2d4", "c5d4", "f3d4", "b8c6"
    ],
    "Scandinavian Defense": [
        "e2e4", "d7d5", "e4d5", "d8d5", "g1f3", "c8g4", "f1e2", "g4f3"
    ],
    "Pirc Defense": [
        "e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f2f4", "f8g7"
    ],
    "Modern Defense": [
        "e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "d7d6", "f2f4", "c7c5"
    ],
    # Ultra-canonical stubs
    "Double King Pawn (e4 e5)": ["e2e4", "e7e5"],
    "Double Queen Pawn (d4 d5)": ["d2d4", "d7d5"],
    "Sicilian Defense Stub (e4 c5)": ["e2e4", "c7c5"],
    "French Defense Stub (e4 e6)": ["e2e4", "e7e6"],
    "Caro-Kann Stub (e4 c6)": ["e2e4", "c7c6"],
    "Pirc Stub (e4 d6)": ["e2e4", "d7d6"],
    "Modern Stub (e4 g6)": ["e2e4", "g7g6"],
    "English Opening Stub (c4)": ["c2c4"],
    "Reti Stub (Nf3)": ["g1f3"],
    "Indian Defense Stub (d4 Nf6)": ["d2d4", "g8f6"],
    "Dutch Stub (d4 f5)": ["d2d4", "f7f5"],
    "Benoni Stub (d4 c5)": ["d2d4", "c7c5"],
    "Catalan Stub (d4 Nf6 c4 e6 g3)": ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"],
    "London Stub (d4 d5 Bf4)": ["d2d4", "d7d5", "c1f4"],
    "King’s Indian Stub (d4 Nf6 c4 g6)": ["d2d4", "g8f6", "c2c4"],
    "Grünfeld Stub (d4 Nf6 c4 g6 Nc3 d5)": ["d2d4", "g8f6", "c2c4", "g7g6"]
}


def get_opening(name=None):
    """
    Return a board set up in a chosen or random opening.
    """
    if name is None:
        name = random.choice(list(OPENING_BOOK.keys()))
    moves = OPENING_BOOK[name]
    
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return name, board


def get_all_openings():
    names, boards = [], []
    for O in OPENING_BOOK.keys():
        names.append(O)
        n, b = get_opening(O)
        boards.append(b)
    
    return names, boards


def random_endgame_board(max_tries=1000):
    """
    Create a random board with exactly 5 pieces:
    - 2 kings (one white, one black)
    - 3 random other pieces (any type, any color, random squares)
    
    Keeps retrying until a legal board is found.
    """
    piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

    for _ in range(max_tries):
        board = chess.Board.empty()
        board.turn = random.choice([chess.WHITE, chess.BLACK])

        # Place the two kings first
        king_squares = random.sample(chess.SQUARES, 2)
        board.set_piece_at(king_squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(king_squares[1], chess.Piece(chess.KING, chess.BLACK))
        occupied = set(king_squares)

        # Place 3 random other pieces
        for _ in range(3):
            sq = random.choice([s for s in chess.SQUARES if s not in occupied])
            occupied.add(sq)
            piece_type = random.choice(piece_types)
            color = random.choice([chess.WHITE, chess.BLACK])
            board.set_piece_at(sq, chess.Piece(piece_type, color))

        # Validate
        if board.is_valid():
            return board

    raise ValueError("Could not generate a valid endgame board in given tries")


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.3f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.2f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.2f}s"


import chess
import chess.engine
from collections import deque

def short_fen(fen):
    return " ".join(fen.split(" ")[:4])

def greedy_sf_tree_paths(n_positions=2000, multipv=4, max_depth=7, eval_thresh=150):
    """
    Return a list of UCI move lists (paths) from STARTPOS via greedy BFS.
    Includes STARTPOS as [] and all intermediate paths until n_positions reached.

    Args:
        n_positions : target number of positions (approx; includes STARTPOS)
        multipv     : how many top moves to expand per node
        max_depth   : Stockfish search depth for analysis
        eval_thresh : only expand moves within (best_cp - cp) <= eval_thresh
        sf_path     : path to stockfish binary; if None, uses chess.engine default resolution

    Returns:
        paths : list[list[str]] such as:
            [ [],
              ['e2e4'], ['d2d4'], ['c2c4'], ['g1f3'],
              ['e2e4','d7d5'], ... ]
    """
    # Seed: STARTPOS plus four common first moves
    start = chess.Board()
    seed_sans = ["e4", "d4", "Nf3", "c4"]

    paths = []                 # output paths
    seen = set()               # short-FEN dedup
    q = deque()                # queue of (board, path)

    # Add STARTPOS
    paths.append([])                       # []
    seen.add(short_fen(start.fen()))

    # Enqueue seeds
    for san in seed_sans:
        mv = start.parse_san(san)
        b2 = start.copy(); b2.push(mv)
        q.append( (b2, [mv.uci()]) )

    eng_kwargs = {}
    eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)

    try:
        while q and len(paths) < n_positions:
            board, path = q.popleft()
            key = short_fen(board.fen())
            if key in seen:
                continue

            seen.add(key)
            paths.append(path)

            if board.is_game_over():
                continue

            info = eng.analyse(board, chess.engine.Limit(depth=max_depth), multipv=multipv)
            if not info:
                continue
            
            best_cp = info[0]["score"].pov(board.turn).score(mate_score=10000)
            if best_cp is None:
                best_cp = 0

            for d in info:
                sc = d["score"].pov(board.turn).score(mate_score=10000)
                if sc is None:
                    continue
                if best_cp - sc > eval_thresh:
                    continue
                if "pv" not in d or not d["pv"]:
                    continue
                mv = d["pv"][0]
                b2 = board.copy()
                b2.push(mv)
                q.append( (b2, path + [mv.uci()]) )
    finally:
        eng.quit()

    return paths


uci_path_path =  r"C:/Users/Bryan/Data/chessbot_data/uci_paths3000.pkl"  
with open(uci_path_path, "rb") as f:
    paths = pickle.load(f)
    
def get_pre_opened_game():
    b = fastboard()
    moves_to_play = random.choice(paths)
    for mtp in moves_to_play:
        b.push_uci(mtp)
    return b


def ensure_df(df_or_dicts):
    if isinstance(df_or_dicts, pd.DataFrame):
        return df_or_dicts.copy()
    return pd.DataFrame(df_or_dicts).copy()


def plot_sf_simple(df):
    """
    Line plot over epochs showing:
      - wins (total)
      - draws (total)
      - draws as White
      - draws as Black
      - average game length (plies) on a secondary axis
    """
    df = ensure_df(df).sort_values("epoch").reset_index(drop=True)

    ep = df["epoch"].astype(int).values
    wins_total   = df.get("wins",         0).fillna(0).astype(int).values
    draws_total  = df.get("draws",        0).fillna(0).astype(int).values
    white_draws  = df.get("white_draws",  0).fillna(0).astype(int).values
    black_draws  = df.get("black_draws",  0).fillna(0).astype(int).values
    avg_ply      = df.get("avg_ply",    np.nan).values / 5

    fig, ax1 = plt.subplots(figsize=(10,4))

    # counts (left axis)
    ax1.plot(ep, wins_total,  lw=1.5, label="Wins (total)")
    ax1.plot(ep, draws_total, lw=1.5, label="Draws (total)")
    ax1.plot(ep, white_draws, lw=1.2, alpha=0.9, label="Draws as White")
    ax1.plot(ep, black_draws, lw=1.2, alpha=0.9, label="Draws as Black")
    ax1.plot(ep, avg_ply,     lw=1.5, color='gray', label="Avg plies / 5")
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Counts")
    ax1.grid(True, alpha=0.3)

    # combined legend
    l1, lab1 = ax1.get_legend_handles_labels()
    ax1.legend(l1, lab1, loc="upper left")

    plt.title("SF600 exhibitions: wins/draws and avg game length")
    plt.tight_layout()
    plt.show()