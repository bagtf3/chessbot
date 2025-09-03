import os
import chess
import chess.engine
import pandas as pd
import numpy as np
import random
import time
from collections import defaultdict

from IPython.display import display, clear_output, SVG
import chess.svg
import uuid

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras

from chessbot.utils import random_init
import chessbot.encoding as cbe
import chessbot.features as feats
import chessbot.utils as cbu

from chessbot import SF_LOC, ENDGAME_LOC
import chess.syzygy

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"

PIECE_TO_ID = {
    None: 0,                # empty square
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

# black pieces offset by +6
def square_to_id(piece):
    if piece is None:
        return 0
    base = PIECE_TO_ID[piece.piece_type]
    return base if piece.color == chess.WHITE else base + 6


def board_to_tokens(board: chess.Board):
    tokens = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        tokens.append(square_to_id(piece))
    # extra tokens
    tokens.append(int(board.turn))
    tokens.append(int(board.has_kingside_castling_rights(chess.WHITE)))
    tokens.append(int(board.has_queenside_castling_rights(chess.WHITE)))
    tokens.append(int(board.has_kingside_castling_rights(chess.BLACK)))
    tokens.append(int(board.has_queenside_castling_rights(chess.BLACK)))
    ep = board.ep_square if board.ep_square else 0
    tokens.append(ep % 64)  # encode en passant square (0 if none)
    return np.array(tokens, dtype=np.int32)

# -------------------------
# Core Transformer Stump
# -------------------------

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, name=None, **kwargs):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn_output, attn_scores = self.att(x, x, return_attention_scores=True)
        x = self.norm1(x + self.drop1(attn_output, training=training))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_output, training=training))
        return x, attn_scores

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
            "name": self.name,
        })
        return config


def make_transformer_stump(vocab_size=32, seq_len=70, d_model=64, num_heads=4, num_layers=2, ff_dim=128, drop_rate=0.1):
    tokens_in = layers.Input(shape=(seq_len,), dtype="int32", name="tokens")

    tok_emb = layers.Embedding(vocab_size, d_model, name="tok_emb")(tokens_in)
    pos_emb = layers.Embedding(seq_len, d_model, name="pos_emb")(tf.range(seq_len))
    x = tok_emb + pos_emb

    attn_maps = []
    for i in range(num_layers):
        x, scores = TransformerBlock(d_model, num_heads, ff_dim, rate=drop_rate, name=f"block_{i}")(x)
        attn_maps.append(scores)

    # sequence features (per-token), and pooled vector stump
    trunk_seq = x   # no Identity
    trunk_vec = layers.GlobalAveragePooling1D(name="trunk_vec")(x)
    stump = Model(tokens_in, [trunk_seq, trunk_vec], name="tf_stump")

    stump.attn_maps = attn_maps
    return stump

# -------------------------
# Heads (policy/value/aux)
# -------------------------

def value_head(trunk_vec, hidden=128, leak=0.05, name="value"):
    x = trunk_vec
    if hidden:
        x = layers.Dense(hidden, name=f"{name}_dense1")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
    out = layers.Dense(1, activation="tanh", name=name)(x)  # [-1,1] White POV
    return out, "mse"


def policy_factor_head(trunk_vec, prefix, hidden=256, leak=0.05):
    """Return 5 factorized Dense logits for a policy head."""
    x = trunk_vec
    if hidden:
        x = layers.Dense(hidden, name=f"{prefix}_dense1")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{prefix}_lrelu1")(x)

    from_logits = layers.Dense(64, name=f"{prefix}_from")(x)
    to_logits = layers.Dense(64, name=f"{prefix}_to")(x)
    piece_logits = layers.Dense(6,  name=f"{prefix}_piece")(x)
    cap_logits = layers.Dense(7,  name=f"{prefix}_cap")(x)
    promo_logits = layers.Dense(5,  name=f"{prefix}_promo")(x)

    return [from_logits, to_logits, piece_logits, cap_logits, promo_logits]


def make_big_model(
    seq_len=70, vocab_size=64, d_model=128, num_heads=4, num_layers=4, ff_dim=512
):

    stump = make_transformer_stump(
        vocab_size, seq_len, d_model, num_heads, num_layers, ff_dim
    )
    
    tokens_in = stump.input
    trunk_seq, trunk_vec = stump.output

    # Value head
    val_out, _ = value_head(trunk_vec, hidden=256)

    # Best/worst move heads (factorized)
    best_outputs  = policy_factor_head(trunk_vec, prefix="best")

    model = Model(tokens_in, [val_out] + best_outputs, name="big_tf")
    return model


def evaluate_terminal(board):
    if board.is_checkmate():
        return 1 if board.result() == '1-0' else -1
    
    elif board.is_game_over(claim_draw=True):
        return 0
    
    else:
        raise Exception("Non terminal board found!")


def blend_value_targets(all_Y, outcome, mode="replace", weight=0.5, schedule="linear"):
    """
    Adjusts the 'value' field in all_Y (list of dicts with per-position targets).

    Args:
      all_Y: list of dicts, each with a 'value' np.array([v], dtype=float32).
      outcome: final result from White POV: +1 (White win), -1 (Black win), 0 (draw).
      mode: 
        - "replace"   -> erase original scores, set all to harsh outcome.
        - "blend"     -> average harsh & original with constant weight.
        - "schedule"  -> blend with a weight that increases over plies.
      weight: float in [0,1] for blending (only used in "blend" mode).
      schedule: only used if mode=="schedule".
        - "linear": weights go 0 , 1 evenly across plies.
        - "fast": accelerate towards 1 faster (quadratic ramp).
    """
    all_Y = all_Y
    n = len(all_Y)
    if n == 0:
        return all_Y

    for i, Y in enumerate(all_Y):
        orig_val = float(Y['value'][0])
        harsh_val = float(outcome)

        if mode == "replace":
            new_val = harsh_val

        elif mode == "blend":
            new_val = (1 - weight) * orig_val + weight * harsh_val

        elif mode == "schedule":
            progress = i / (n - 1) if n > 1 else 1.0
            if schedule == "linear":
                w = progress                     # 0 , 1 evenly
            elif schedule == "fast":
                w = progress**2                  # accelerates later
            else:
                raise ValueError(f"unknown schedule {schedule}")
            new_val = (1 - w) * orig_val + w * harsh_val

        else:
            raise ValueError(f"unknown mode {mode}")

        # overwrite in place
        Y['value'] = np.array([new_val], dtype=np.float32)

    return all_Y


def load_model(model_loc):
    custom = {"TransformerBlock": TransformerBlock}
    model = keras.models.load_model(model_loc, custom_objects=custom)
    return model
#%%
#model = load_model(os.path.join(MODEL_DIR, "transformer_model_v400.h5"))
#model = make_big_model()
#model.summary()

def reload_model_100():
    losses = {
        "value": "mse",
        "best_from": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "best_to": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "best_piece": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "best_cap": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "worst_from": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "worst_to": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "worst_piece": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "worst_cap": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    }
    
    model = load_model(f"C:/Users/Bryan/Data/chessbot_data/models/transformer_model_v2-{100}.h5")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    loss_weights = {
        "value": 1.0,
    
        # Best move heads
        "best_from": 0.5,
        "best_to": 0.5,
        "best_piece": 0.5,
        "best_cap": 0.2,
    
        # Worst move heads
        "worst_from": 0.5,
        "worst_to": 0.5,
        "worst_piece": 0.5,
        "worst_cap": 0.2
    }
    
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights)
    return model

model = reload_model_100()
model.summary()
    
all_evals = pd.DataFrame()
#%%

def prepare_data_for_model(X, Y, model, W=None):
    inputs, outputs, weights = {}, {}, {}

    # Inputs
    for name, tensor in zip(model.input_names, model.inputs):
        vals = [x[name] for x in X]
        arr = np.array(vals, dtype=np.int32)
        inputs[name] = arr

    # Outputs
    for name, tensor in zip(model.output_names, model.outputs):
        vals = [y.get(name, None) for y in Y]
        # replace None with 0 (dummy class) if needed
        vals = [0 if v is None else v for v in vals]

        if name == "value":
            arr = np.array(vals, dtype=np.float32).reshape(-1, 1)
        else:
            arr = np.array(vals, dtype=np.int32)
        outputs[name] = arr

    # Sample weights (optional)
    if W is not None:
        for name, tensor in zip(model.output_names, model.outputs):
            vals = [w.get(name, 1.0) for w in W]   # default weight = 1.0
            arr = np.array(vals, dtype=np.float32)
            weights[name] = arr

    if W is not None:
        return inputs, outputs, weights
    else:
        return inputs, outputs


def prepare_X(X, model):
    if not isinstance(X, list):
        X = [X]
        
    inputs = {}

    # Inputs
    for name, tensor in zip(model.input_names, model.inputs):
        vals = [x[name] for x in X]   # grab each sample's tokens
        arr = np.array(vals, dtype=np.int32)
        inputs[name] = arr
    
    return inputs
        
        
def mirror_move(move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion
    )


PIECE_MAP = {
    None: 0, 
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}


def move_to_labels(board, move):
    """
    Convert a move into factorized labels for training.
    Returns: (from_idx, to_idx, piece_idx, cap_idx, promo_idx)
    """

    # From / To squares (0..63)
    from_idx = move.from_square
    to_idx   = move.to_square

    # Piece to move (0..5)
    piece = board.piece_type_at(move.from_square)
    piece_idx = PIECE_MAP[piece]

    # Capture (0..6)
    cap_idx = 0
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            cap_idx = PIECE_MAP.get(captured.piece_type, 0)
    
    promo_idx = PIECE_MAP[move.promotion]

    return from_idx, to_idx, piece_idx, cap_idx


def move_distance(moves, best, worst, 
                  w_from=1.0, w_to=1.0, w_piece=1.0,
                  alpha=1.0, beta=0.25):
    """
    Weighted distance scoring:
      alpha * (closeness to best) - beta * (closeness to worst)
    """
    moves = np.asarray(moves, dtype=float)
    best  = np.asarray(best, dtype=float)
    worst = np.asarray(worst, dtype=float)

    W = np.array([w_from, w_to, w_piece])

    d_best  = np.linalg.norm((moves - best)  * W, axis=1)
    d_worst = np.linalg.norm((moves - worst) * W, axis=1)

    return -alpha * d_best + beta * d_worst


engine = chess.engine.SimpleEngine.popen_uci(SF_LOC)
def collect_stockfish_data(engine, depth=np.random.randint(2, 6), n_init=None, n_rand=5):
    if n_init is not None:
        board = random_init(n_init)
    else:
        board = random_init(np.random.randint(0, 3))
        
    all_X, all_Y = [], []
    while not board.is_game_over():
        # always STM-POV implemented via always white-POV
        to_score = board if board.turn else board.mirror()
        X_tokens = board_to_tokens(to_score)
        all_X.append({"tokens": X_tokens})

        # Run Stockfish multipv analysis
        info_list = engine.analyse(
            to_score, multipv=len(list(to_score.legal_moves)),
            limit=chess.engine.Limit(depth=depth),
            info=chess.engine.Info.ALL
        )

        # TARGETS
        Y = {}

        # 1. Value head target
        # Scale Stockfish score to [-1, 1]
        Y["value"] = cbe.score_to_cp_white(info_list[0]["score"])

        # 2. Best move (from multipv=1)
        best_move = info_list[0]["pv"][0]
        from_idx, to_idx, piece_idx, cap_idx = move_to_labels(to_score, best_move)
        
        Y["best_from"] = from_idx
        Y["best_to"] = to_idx
        Y["best_piece"] = piece_idx
        Y["best_cap"] = cap_idx

        # 3. Worst move (lowest eval in multipv)
        worst_move = info_list[-1]["pv"][0]
        from_idx, to_idx, piece_idx, cap_idx = move_to_labels(to_score, worst_move)
        
        Y["worst_from"] = from_idx
        Y["worst_to"] = to_idx
        Y["worst_piece"] = piece_idx
        Y["worst_cap"] = cap_idx

        all_Y.append(Y)

        # Pick a move to play to continue the game (random among top-n_rand)
        move_to_make = random.choice(info_list[:n_rand])["pv"][0]
        if not board.turn:
            move_to_make = mirror_move(move_to_make)
            
        board.push(move_to_make)

    return all_X, all_Y
#%%

# phase 1: learn from stockfish evaluations
for outer in range(1, 101):
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
        all_X, all_Y = [], []
        for game in range(10):
            print(f"Outer {outer} game {game}: {len(all_X)} positions so far")
            X, Y = collect_stockfish_data(engine, n_rand=6+game)
            all_X += X
            all_Y += Y
        
        inputs, outputs = prepare_data_for_model(all_X, all_Y, model)
        
        eval_df = cbu.score_game_data(model, inputs, outputs)
        all_evals = pd.concat([all_evals, eval_df])
        
        if len(all_evals) and (len(all_evals) % 5 == 0):
           cbu.plot_training_progress(all_evals)
        
        # train (no val)
        model.fit(inputs, outputs, batch_size=512, epochs=2, shuffle=True)
        all_X = []
        all_Y = []
        
        if outer % 50 == 0:
            model.save(
                f"C:/Users/Bryan/Data/chessbot_data/models/transformer_model_v2-{outer}.h5"
            )
# #%%
# def score_to_cp_rel(board_score):
#     rel = board_score.relative
    
#     #check for mates
#     if rel.score() is None:
#         return np.clip(rel.score(mate_score=16), -10, 10) / 10
        
#     else:
#         return np.clip(rel.score() / 1000, -0.95, 0.95)


# def choose_move(model, board, sf_eval, top_n=3):
#     # mirror board for STM-POV
#     to_score = board if board.turn else board.mirror()
#     tokens = {"tokens": board_to_tokens(to_score)}
#     inputs = prepare_X([tokens], model)

#     # predictions -> dict
#     raw_preds = model.predict(inputs, verbose=0)
#     preds = {name: arr for name, arr in zip(model.output_names, raw_preds)}
    
#     value = raw_preds[model.output_names.index("value")].item()
#     if not board.turn:
#         value = -1*value
#     print(f"Model estimate of postion: {value: <.3}")

#     # best & worst triples
#     best = np.array([
#         preds["best_from"].argmax(axis=1)[0],
#         preds["best_to"].argmax(axis=1)[0],
#         preds["best_piece"].argmax(axis=1)[0],
#     ])
#     worst = np.array([
#         preds["worst_from"].argmax(axis=1)[0],
#         preds["worst_to"].argmax(axis=1)[0],
#         preds["worst_piece"].argmax(axis=1)[0],
#     ])

#     # legal moves → triples
#     legal_moves = []
#     moves_list = list(to_score.legal_moves)
    
#     for mv in moves_list:
#         piece = to_score.piece_type_at(mv.from_square) - 1
#         legal_moves.append([mv.from_square, mv.to_square, piece])
#     legal_moves = np.array(legal_moves)

#     # score moves
#     scores = move_distance(legal_moves, best, worst, alpha=1.0, beta=0.0)

#     # pick top-N
#     top_idx = np.argsort(-scores)[:top_n]
#     moves, scores = [moves_list[i] for i in top_idx], scores[top_idx]
    
#     if not board.turn:
#         moves = [mirror_move(mv) for mv in moves]
        
#     out = [[m, s] for m, s in zip(moves, scores)]
#     sf_moves = [s['pv'][0] for s in sf_eval]
#     sf_scores = [score_to_cp_rel(s['score']) for s in sf_eval]
    
#     print("\n--- Model vs Stockfish ---")
#     for i in range(min(top_n, len(out))):        
#         uci_move = out[i][0]
#         san_move = board.san(uci_move)
#         sf_index = sf_moves.index(uci_move)
#         sf_score = sf_scores[sf_index]
#         sf_rank = sf_index + 1
#         print(f"{i+1}. {uci_move} ({san_move}) SF: {sf_score: <.3} Rank: {sf_rank}")
        
#     print("--------------------------\n")
#     best_move = out[0][0]
#     return best_move
    
# #%%
# from IPython.display import display, clear_output, SVG
# import chess.svg, chess.pgn
# import chessbot.utils as cbu
# import chessbot.encoding as cbe
# import numpy as np

# bot_color = np.random.uniform() < 0.5
# board = chess.Board()
# with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
#     while not board.is_game_over():
#         clear_output(wait=True)
#         display(SVG(chess.svg.board(board=board, flipped=not bot_color)))
        
#         sf_eval = engine.analyse(
#             board, multipv=len(list(board.legal_moves)), limit=chess.engine.Limit(depth=3),
#             info=chess.engine.Info.ALL
#         )
        
#         print(f"Current Eval: {cbe.score_to_cp_white(sf_eval[0]['score']):.<3}")
#         time.sleep(0.75)
#         if board.turn == bot_color:
#             best_move = choose_move(model, board, sf_eval, top_n=5)
#             print()
#             board.push(best_move)
#             time.sleep(2.0)
            
#         else:
#             sf_move = random.choice(sf_eval[:3])['pv'][0]
#             san = board.san(sf_move)
#             board.push(sf_move)
#             time.sleep(2.0)
#%%
import json
from datetime import datetime

LOGFILE = "C:/Users/Bryan/Data/chessbot_data/self_play_stats.jsonl"
def log_game_stats(games, epoch=None):
    log_path = LOGFILE
    
    # Create/initialize empty file
    if not os.path.exists(log_path):
        os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)
        with open(LOGFILE, "w") as f:
            pass
    
    stats = {
        "timestamp": datetime.utcnow().isoformat(),
        "epoch": epoch,
        "n_games": len(games),
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "avg_length": 0,
    }
    lengths = []
    for g in games:
        if g.outcome > 0:
            stats["white_wins"] += 1
        elif g.outcome < 0:
            stats["black_wins"] += 1
        else:
            stats["draws"] += 1
        lengths.append(len(g.board.move_stack))

    if lengths:
        stats["avg_length"] = float(np.mean(lengths))

    # append to JSONL
    with open(log_path, "a") as f:
        f.write(json.dumps(stats) + "\n")

    return stats


def get_legal_labels(board, moves):
    N = len(moves)
    f, t, pc, cp = np.zeros(N, int), np.zeros(N, int), np.zeros(N, int), np.zeros(N, int)
    for i, mv in enumerate(moves):
        f[i], t[i], pc[i], cp[i] = move_to_labels(board, mv)
    return {"from":f, "to":t, "piece":pc, "cap":cp}

# legal_labels = get_legal_label(to_score, moves_list)
# priors_best, priots_worst = compute_move_priors( model_outputs, legal_labels)
def compute_move_priors(model_outputs, legal_labels, weights=None):
    """
    model_outputs   : dict of np.arrays (softmax probs) for each head
                      e.g. {"best_from": (64,), "best_to": (64,), ...}
    legal_labels    : dict of arrays, each shaped (N_moves,)
                      e.g. {"from": [12,14,...], "to": [...], "piece": [...], ...}
    weights         : dict of per-head importance, defaults tuned for chess
    
    Returns:
        priors_best  : np.array (N_moves,) priors from best head
        priors_worst : np.array (N_moves,) priors from worst head
    """
    if weights is None:
        weights = {"from":1.0, "to":1.0, "piece":0.5, "cap":0.25}

    N = len(legal_labels["from"])

    # Stack indices for advanced indexing
    # e.g. for "from" head: probs[from_indices]
    priors_best = np.zeros(N, dtype=np.float32)
    priors_worst = np.zeros(N, dtype=np.float32)

    for factor in ["from","to","piece","cap"]:
        # Get indices of legal moves for this factor
        idxs = legal_labels[factor]

        # Look up probs for each legal move at those indices
        p_best  = model_outputs[f"best_{factor}"][idxs]
        p_worst = model_outputs[f"worst_{factor}"][idxs]

        # Avoid log(0)
        p_best  = np.clip(p_best,  1e-9, 1.0)
        p_worst = np.clip(p_worst, 1e-9, 1.0)

        # Accumulate weighted log-likelihood
        priors_best  += weights[factor] * np.log(p_best)
        priors_worst += weights[factor] * np.log(p_worst)

    # Convert back from log-scores to priors
    priors_best  = np.exp(priors_best  - np.max(priors_best))
    priors_worst = np.exp(priors_worst - np.max(priors_worst))

    # Normalize
    priors_best  /= priors_best.sum()
    priors_worst /= priors_worst.sum()

    return priors_best, priors_worst


class SelfPlayGame:
    def __init__(self, n_init=0, move_limit=80, prevent_repetition=False, **kwargs):
        self.game_id = str(uuid.uuid4())
        self.board = random_init(n_init)
        self.init_fen = self.board.fen()
        self.move_limit = move_limit
        
        self.prevent_repetition = prevent_repetition
        self.pos_move_counter = defaultdict(int)
        self.pos_counter = defaultdict(int)
        self.move_counter = defaultdict(int)

        self.training_X = []
        self.training_Y = []
        self.sample_weights = []

        self.outcome = None
        self.game_complete = False

        self.mat_adv_counter = 0  # track 7-pt lead persistence
        self.draw_score = kwargs.get("draw_score", 0.0)

    def short_fen(self, fen=None):
        f = self.board.fen() if fen is None else fen
        return " ".join(f.split(" ")[:4])

    def check_termination(self):
        claim_draw = len(self.board.move_stack) > 60
        if self.board.is_game_over(claim_draw=claim_draw):
            print(self.game_id, "Game over detected:", self.board.result(), self.board.fen())
            self.outcome = evaluate_terminal(self.board)
            return True
    
        white_mat, black_mat = feats.get_piece_value_sum(self.board)
        mat_diff = 10*white_mat - 10*black_mat
        if abs(mat_diff) >= 9:
            self.mat_adv_counter += 1
        else:
            self.mat_adv_counter = 0
    
        if self.mat_adv_counter >= 7:
            print(self.game_id, "Material cutoff:", mat_diff)
            self.outcome = 1.0 if mat_diff > 0 else -1.0
            return True
    
        if self.board.fullmove_number > self.move_limit:
            print(self.game_id, "Move limit reached:", self.board.fullmove_number)
            self.outcome = 0.0
            return True
    
        return False

    def record_position(self):
        # STM-POV: mirror if Black to move
        to_score = self.board if self.board.turn else self.board.mirror()
        tokens = board_to_tokens(to_score)

        self.training_X.append({"tokens": tokens})
        
        # labels filled in later (after outcome known)
        self.training_Y.append({
            'value': None, 'best_from': None, 'best_to': None, 'best_piece': None,
            'best_cap': None, 'worst_from': None, 'worst_to': None, 'worst_piece': None,
            'worst_cap': None
        })
        
        self.sample_weights.append({
            'value': 0.1, 'best_from': 0.0, 'best_to': 0.0, 'best_piece': 0.0,
            'best_cap': 0.0, 'worst_from': 0.0, 'worst_to': 0.0, 'worst_piece': 0.0,
            'worst_cap': 0.0
        })

    def get_premove_data(self):
        if self.check_termination():
            self.finalize_outcome()
            return [], []
        
        # always collect
        self.record_position()
        
        meta = [self.game_id]
        X = self.training_X[-1:]
        # this gets sent to the batch, move made when preds are returned
        return X, meta    
    
    def make_move(self, preds, top_n=1):
        """
        Select and push a move given model predictions for this position.
        raw_preds: list/array of outputs for ONE sample (from the batch)
        """
        
        short_fen = self.short_fen()
        self.pos_counter[short_fen] += 1
        
        # extract best/worst anchors
        best = np.array([
            preds["best_from"].argmax(),
            preds["best_to"].argmax(),
            preds["best_piece"].argmax(),
        ])
        worst = np.array([
            preds["worst_from"].argmax(),
            preds["worst_to"].argmax(),
            preds["worst_piece"].argmax(),
        ])
    
        # build legal-move triples
        to_score = self.board if self.board.turn else self.board.mirror()
        moves_list = list(to_score.legal_moves)
        
        # prevent threefold by storing pos + move and if that combo has been encountered
        if self.prevent_repetition:
            repeated = []
            for lm in list(self.board.legal_moves):
                if self.pos_move_counter[(short_fen, lm)] >= 2:
                    repeated.append(lm)
                    
                # # check the san too
                if self.move_counter[self.board.san(lm)] >= 2:
                    repeated.append(lm)
            
            # if there are no non-repeating moves, end game in draw
            if len(repeated) == len(moves_list):
                self.outcome = self.draw_score
                self.finalize_outcome()
                return None, None
            
            if repeated:
                # flip and subset moves_list
                disallow = [m if self.board.turn else mirror_move(m) for m in repeated]
                moves_list = [m for m in moves_list if m not in disallow]
            
        legal_triples = []
        for mv in moves_list:
            piece = to_score.piece_type_at(mv.from_square) - 1
            legal_triples.append([mv.from_square, mv.to_square, piece])
        legal_triples = np.array(legal_triples)
    
        # score by distance
        scores = move_distance(
            legal_triples, best, worst,
            w_from=1.0, w_to=1.0, w_piece=1.0,  alpha=1.0, beta=0.01
        )
    
        # pick top-N
        top_idx = np.argsort(-scores)[:top_n]
        moves, scores = [moves_list[i] for i in top_idx], scores[top_idx]
    
        # unmirror back if it was Black’s turn
        if not self.board.turn:
            moves = [mirror_move(mv) for mv in moves]
        
        chosen = random.choice(moves) if top_n > 1 else moves[0]
        # push best move
        self.move_counter[self.board.san(chosen)] += 1
        self.board.push(chosen)
        self.pos_move_counter[(short_fen, chosen)] += 1
        
        # record move info in the last Y slot
        from_idx, to_idx, piece_idx, cap_idx = move_to_labels(self.board, chosen)
    
        self.training_Y[-1].update({
            "from": from_idx,
            "to": to_idx,
            "piece": piece_idx,
            "cap": cap_idx,
            "move": chosen,   # keep the actual move object too
        })
    
        return chosen, scores[moves.index(chosen)]
    
    def finalize_outcome(self):
        good_factors = ["best_from","best_to","best_piece","best_cap"]
        bad_factors = ["worst_from","worst_to","worst_piece","worst_cap"]
        stm = True  # White POV
        for x, y, w in zip(self.training_X, self.training_Y, self.sample_weights):
            # always set value from White’s perspective
            y["value"] = self.outcome if stm else -self.outcome
            
            if self.outcome == 0:
                w['value'] = 0.005
            
            elif (self.outcome > 0 and stm) or (self.outcome < 0 and not stm):
                # winning side → good move
                y["best_from"] = y.pop("from")
                y["best_to"]   = y.pop("to")
                y["best_piece"]= y.pop("piece")
                y["best_cap"]  = y.pop("cap")
                w.update({k:0.1 for k in good_factors})
            else:
                # losing side → bad move
                y["worst_from"] = y.pop("from")
                y["worst_to"]   = y.pop("to")
                y["worst_piece"]= y.pop("piece")
                y["worst_cap"]  = y.pop("cap")
                w.update({k:0.01 for k in bad_factors})
    
            stm = not stm
        self.game_complete = True
        
    def show_board(self, flipped=False, sleep=0.1):
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=self.board, flipped=flipped)))
        time.sleep(sleep)


def self_play_batch(model, n_games=32, show=True, return_games=False, **kwargs):
    games = []
    for _ in range(n_games):
        n_init = kwargs.get("n_init", np.random.randint(3, 15))
        games.append(SelfPlayGame(n_init=n_init, prevent_repetition=True))
                 
    all_finished = []
    train_X, train_Y, train_W = [], [], []
    while games:
        this_X, this_meta = [], []
    
        for g in games:
            if not g.game_complete:
                X, meta = g.get_premove_data()
                this_X += X
                this_meta += meta
    
        finished = [g for g in games if g.game_complete]
        all_finished += finished
        for g in finished:
            train_X += g.training_X
            train_Y += g.training_Y
            train_W += g.sample_weights
            
        games = [g for g in games if not g.game_complete]
    
        if not this_X:
            break
        
        # predict and store
        if this_X:
            to_predict = prepare_X(this_X, model)
            raw_preds = model.predict(to_predict, batch_size=256, verbose=0)
            preds = {name: arr for name, arr in zip(model.output_names, raw_preds)}
            
            preds_lookup = {}
            for i, g_id in enumerate(this_meta):
                preds_lookup[g_id] = {k:v[i] for k, v in preds.items()}
        
        # make the moves
        for game in games:
            game.make_move(preds_lookup[game.game_id], kwargs.get("top_n", 1))
        
        # show the game
        if show and games:
            game = games[0]
            games[0].show_board(sleep=0.0)
        
        # drain the list once more
        finished = [g for g in games if g.game_complete]
        all_finished += finished
        for g in finished:
            train_X += g.training_X
            train_Y += g.training_Y
            train_W += g.sample_weights
            
        games = [g for g in games if not g.game_complete]
    
    # we get here once its done with every game
    inputs, outputs, weights = prepare_data_for_model(train_X, train_Y, model, train_W)
    if return_games:
        return inputs, outputs, weights, all_finished
    
    else:
        return inputs, outputs, weights

i, o, w, g = self_play_batch(model, n_games=1, show=True, return_games=True)
#%%
#model = reload_model_100()
#%%
for outer in range(1, 50):
    inputs, outputs, weights, games = self_play_batch(
        model, n_games=128, show=True, return_games=True, top_n=2
    )
    
    stats = log_game_stats(games, epoch=outer)
    print(stats)

    eval_df = cbu.score_game_data(model, inputs, outputs)
    all_evals = pd.concat([all_evals, eval_df])
    
    if len(all_evals) and (len(all_evals) % 5 == 0):
        cbu.plot_training_progress(all_evals)
    
    # train (no val)
    model.fit(
        inputs, outputs, sample_weight=weights,
        batch_size=512, epochs=1, shuffle=True
    )

    if outer % 50 == 0:
        model.save(
            f"C:/Users/Bryan/Data/chessbot_data/models/transformer_model_selfplay_v{outer}.h5"
        )

#%%
# phase 2: stockfish eval on self play games
class ChessGame(object):
    def __init__(self, engine, tablebase, n_init=0, depth=1,
                 blend_outcomes=False, mode="self"):
        """
        mode = "self" , both sides use model
        mode = "vs_sf" , one side is the model, one side is Stockfish
        """
        self.game_id = str(uuid.uuid4())
        self.engine = engine
        self.tablebase = tablebase
        self.depth = depth
        self.board = random_init(n_init)
        self.move_limit = 180
        self.blend_outcomes = blend_outcomes
        self.training_X = []
        self.training_Y = []
        self.outcome = None
        self.game_complete = False
        self.endgame_table = ENDGAME_LOC
        self.mode = mode

        if mode == "vs_sf":
            # assign bot’s color at random (True = White, False = Black)
            self.bot_color = np.random.rand() < 0.5
        else:
            self.bot_color = None  # unused


    def short_fen(self, board):
        return " ".join(board.fen().split(" ")[:4])
        
    def five_or_less(self):
        return len(self.board.piece_map()) <= 5
    
    def lookup_endgame(self):
        result = np.clip(self.tablebase.probe_wdl(self.board), -1, 1)
        return result if self.board.turn else -1*result
        
    def finalize_outcome(self, outcome=None):
        self.game_complete = True
        if outcome is None:
            self.outcome = evaluate_terminal(self.board)
        else:
            self.outcome = outcome
        if self.blend_outcomes:
            self.training_Y = blend_value_targets(
                self.training_Y, self.outcome, mode="blend", weight=0.25
            )
            
    def get_X_data(self, board):
        X_tokens = board_to_tokens(board)
        legal_mask = cbe.legal_mask_8x8x73(board).reshape(-1)
        return {'policy_mask': legal_mask, "tokens": X_tokens}
    
    def get_Y_data(self, board):
        info_list = self.engine.analyse(
            board, multipv=len(list(board.legal_moves)),
            limit=chess.engine.Limit(depth=self.depth),
            info=chess.engine.Info.ALL
        )
        Y = cbe.build_training_targets_value_policy(board, info_list)
        Y.update(feats.all_king_exposure_features(board))
        Y.update(feats.all_piece_features(board))
        Y['material'] = feats.get_piece_value_sum(board)
        Y['piece_to_move'] = cbe.piece_to_move_target(board, Y['policy_logits'])
        return Y
        
    def get_premove_data(self):
        if self.board.is_game_over():
            self.finalize_outcome()
            return [], [], []
        
        # always collect
        self.training_X.append(self.get_X_data(self.board))
        self.training_Y.append(self.get_Y_data(self.board))
        
        # In vs_sf mode , only gather training data on bot's turn
        if self.mode == "vs_sf" and self.board.turn != self.bot_color:
            return [], [], []

        these_X, meta, endgames = [], [], []
        for mv in self.board.legal_moves:
            self.board.push(mv)
            if self.five_or_less():
                res = self.lookup_endgame()
                self.board.pop()
                endgames.append((self.game_id, self.board.san(mv), res))
                continue
            if self.board.is_game_over():
                self.finalize_outcome()
                return [], [], []
            these_X.append(self.get_X_data(self.board))
            self.board.pop()
            meta.append((self.game_id, self.board.san(mv)))
        return these_X, meta, endgames

    def make_move(self, move_preds, random_prob=0.05):
        if self.mode == "vs_sf" and self.board.turn != self.bot_color:
            # SF’s turn
            info = self.engine.analyse(
                self.board,
                multipv=1,
                limit=chess.engine.Limit(depth=self.depth+3),
                info=chess.engine.Info.ALL
            )
            move = info[0]['pv'][0]
            self.board.push(move)
        else:
            # Model’s turn (works for self-play too)
            if np.random.rand() <= random_prob:
                move = random.choice(list(self.board.legal_moves))
            else:
                if self.board.turn:
                    best = max(move_preds, key=lambda x: x[1])
                else:
                    best = min(move_preds, key=lambda x: x[1])
                san, score = best
                move = self.board.parse_san(san)
            self.board.push(move)

        if self.board.is_game_over():
            self.finalize_outcome()
            
    def return_training_data(self):
        return self.training_X, self.training_Y
    
    def show_board(self, flipped=False, sleep=0.1):
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=self.board, flipped=flipped)))
        time.sleep(sleep)
    

def batch_training_loop(engine, tablebase, n_games=32, **kwargs):
    # look for optional args
    show = kwargs.get("show", False)
    blend = kwargs.get("blend", False)
    n_init = kwargs.get("n_init", np.random.randint(10, 30))
    depth = kwargs.get("depth", 3)
    mode = kwargs.get("mode", 'self')
    
    chess_games = [
        ChessGame(
            engine, tablebase, n_init=n_init, depth=depth,
            blend_outcomes=blend, mode=mode
        )
        for _ in range(n_games)
    ]
    
    train_X, train_Y = [], []
    while chess_games:
        this_X, this_meta, this_endgames = [], [], []
    
        for game in chess_games:
            if not game.game_complete:
                X, meta, endgames = game.get_premove_data()
                this_X += X
                this_meta += meta
                this_endgames += endgames
    
        finished = [g for g in chess_games if g.game_complete]
        for g in finished:
            train_X += g.training_X
            train_Y += g.training_Y
        chess_games = [g for g in chess_games if not g.game_complete]
    
        if not this_X and not this_endgames:
            break
        
        if this_X:
            to_predict = prepare_X(this_X, model)
            preds = model.predict(to_predict, batch_size=768, verbose=0)['value']
            preds_lookup = defaultdict(list)
            for (g_id, move), pred in zip(this_meta, preds):
                preds_lookup[g_id].append([move, pred.item()])
        else:
            preds_lookup = defaultdict(list)
        
        for g_id, move, pred in this_endgames:
            preds_lookup[g_id].append([move, pred])
        
        for game in chess_games:
            game.make_move(preds_lookup[game.game_id])
        
        if show and chess_games:
            game = chess_games[0]
            flipped = False
            if game.bot_color is not None:
                flipped = not game.bot_color
            
            chess_games[0].show_board(flipped=flipped, sleep=0.0)
        
        finished = [g for g in chess_games if g.game_complete]
        for g in finished:
            train_X += g.training_X
            train_Y += g.training_Y
        chess_games = [g for g in chess_games if not g.game_complete]
        
    inputs, outputs = prepare_data_for_model(train_X, train_Y, model)
    return inputs, outputs


for outer in range(401, 501):
    with chess.syzygy.open_tablebase(ENDGAME_LOC) as tablebase:
        with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
            inputs, outputs = batch_training_loop(
                engine, tablebase, n_games=48, show=True, blend=True,
                n_init=4, mode="vs_sf", depth=5
            )
            
            eval_df = cbu.score_game_data(model, inputs, outputs)
            all_evals = pd.concat([all_evals, eval_df])
            
            if len(all_evals) and (len(all_evals) % 5 == 0):
                cbu.plot_training_progress(all_evals)
            
            # train (no val)
            model.fit(inputs, outputs, batch_size=512, epochs=2, shuffle=True)
            all_X = []
            all_Y = []
        
        if outer % 50 == 0:
            model.save(
                f"C:/Users/Bryan/Data/chessbot_data/models/transformer_model_v{outer}.h5"
            )







