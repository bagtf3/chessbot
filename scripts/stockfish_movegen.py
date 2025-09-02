import chess
import chess.engine
import chess.svg
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from IPython.display import display, clear_output, SVG

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras

from chessbot.utils import random_init
import chessbot.encoding as cbe
import chessbot.features as feats
from chessbot.model import masked_policy_ce

from chessbot import SF_LOC

import chess.syzygy


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


def policy_head(trunk_vec, n_moves, hidden=128, leak=0.05, name="policy", use_mask=True):
    x = trunk_vec
    if hidden:
        x = layers.Dense(hidden, name=f"{name}_dense1")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
    logits = layers.Dense(n_moves, name=f"{name}_logits")(x)
    if use_mask:
        mask_in = layers.Input(shape=(n_moves,), dtype="float32", name=f"{name}_mask")
        masked_logits = layers.Add(name=f"{name}_masked_logits")(
            [logits, (1.0 - mask_in) * (-1e9)]
        )
        
        probs = layers.Activation("softmax", name=name)(masked_logits)
        return probs, "categorical_crossentropy", mask_in
    else:
        probs = layers.Activation("softmax", name=name)(logits)
        return probs, "categorical_crossentropy", None


def count_vec_head(trunk_vec, name, shape, leak=0.05):
    x = layers.Dense(128, name=f"{name}_dense1")(trunk_vec)
    x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
    out = layers.Dense(shape, activation="softplus", name=name)(x)
    return out, "poisson"


def binary_head(trunk_vec, name, shape=1, leak=0.05):
    x = layers.Dense(128, name=f"{name}_dense1")(trunk_vec)
    x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
    out = layers.Dense(shape, activation="sigmoid", name=name)(x)
    return out, "binary_crossentropy"


def regression_head(trunk_vec, name, shape=1, leak=0.05):
    x = layers.Dense(128, name=f"{name}_dense1")(trunk_vec)
    x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
    out = layers.Dense(shape, activation="linear", name=name)(x)
    return out, "mse"


def probe_head(trunk_vec, name, leak=0.05):
    x = layers.Dense(128, name=f"{name}_dense1")(trunk_vec)
    x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
    # mirror your ResNet behavior
    if name == "hanging_opp_value":
        out = layers.Dense(1, activation="linear", name=name)(x); loss = "mse"
    else:
        out = layers.Dense(1, activation="sigmoid", name=name)(x)
        loss = "binary_crossentropy"
    det = layers.Lambda(lambda t: tf.stop_gradient(t), name=f"sg_{name}")(out)
    return out, loss, det


def piece_to_move_head(trunk_vec, leak=0.05, name="piece_to_move"):
    x = layers.Dense(128, name=f"{name}_dense1")(trunk_vec)
    x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
    out = layers.Dense(6, activation="softmax", name=name)(x)  # P,N,B,R,Q,K
    return out, tf.keras.losses.CategoricalCrossentropy()

# -------------------------
# Builder with aux configs
# -------------------------

def build_transformer_full_with_aux(
    vocab_size=32, seq_len=70, d_model=128, num_heads=4, num_layers=4, ff_dim=128,
    n_moves=4672, use_mask=True, aux_configs=None, aux_into_heads=False,
    aux_weight=0.1, value_weight=1.0, policy_weight=1.0, drop_rate=0.1
):
    
    if aux_configs is None:
        aux_configs = []

    stump = make_transformer_stump(
        vocab_size=vocab_size, seq_len=seq_len, d_model=d_model, num_heads=num_heads, 
        num_layers=num_layers, ff_dim=ff_dim, drop_rate=drop_rate
    )

    tokens_in = stump.input
    trunk_seq, trunk_vec = stump.output  # [B,seq,C], [B,C]

    # Collect probe dets for optional stop-grad concat
    probe_dets = []

    outputs = {}
    losses = {}
    loss_weights = {}

    # Main value & policy heads (heads see stump_vec; optionally augmented with probe_dets)
    head_base = trunk_vec

    # First pass: build aux heads; collect dets for concat if requested
    for kind, name, shape in aux_configs:
        if kind == "count":
            out, loss = count_vec_head(trunk_vec, name, shape)
            outputs[name] = out; losses[name] = loss; loss_weights[name] = aux_weight

        elif kind == "binary":
            out, loss = binary_head(trunk_vec, name, shape)
            outputs[name] = out; losses[name] = loss; loss_weights[name] = aux_weight

        elif kind == "regression":
            out, loss = regression_head(trunk_vec, name, shape)
            outputs[name] = out; losses[name] = loss; loss_weights[name] = aux_weight

        elif kind == "probe":
            out, loss, det = probe_head(trunk_vec, name)
            outputs[name] = out; losses[name] = loss; loss_weights[name] = aux_weight
            probe_dets.append(det)

        elif kind == "piece_to_move":
            out, loss = piece_to_move_head(trunk_vec, name=name)
            outputs[name] = out; losses[name] = loss; loss_weights[name] = aux_weight

        else:
            raise ValueError("unknown aux kind: " + str(kind))

    if aux_into_heads and probe_dets:
        head_base = layers.Concatenate(name="head_feat_concat")(
            [trunk_vec] + [layers.Lambda(lambda t: t)(d) for d in probe_dets]
        )

    # Value head
    value_out, v_loss = value_head(head_base, hidden=128, name="value")
    outputs["value"] = value_out; losses["value"] = v_loss
    loss_weights["value"] = value_weight

    # Policy head (masked softmax)
    policy_out, p_loss, mask_in = policy_head(
        head_base, n_moves=n_moves, hidden=128, name="policy", use_mask=use_mask
    )
    
    outputs["policy"] = policy_out; losses["policy"] = p_loss
    loss_weights["policy"] = policy_weight

    inputs = [tokens_in] + ([mask_in] if mask_in is not None else [])
    model = Model(inputs=inputs, outputs=outputs, name="transformer_with_aux")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4), loss=losses, loss_weights=loss_weights
    )
    
    model._head_losses = dict(losses)
    model._head_loss_weights = dict(loss_weights)
    model.attn_maps = stump.attn_maps
    return model


# 1) choose your aux heads
aux_configs = [
    # king exposure bundle (2 = [mine, theirs])
    ('count',  'king_ray_exposure',          2),
    ('count',  'king_ring_pressure',         2),
    ('count',  'king_pawn_shield',           2),
    ('count',  'king_escape_square',         2),

    # pawns
    ('count',  'pawn_undefended',            2),
    ('count',  'pawn_hanging',               2),
    ('count',  'pawn_en_prise',              2),

    # knights
    ('count',  'knight_undefended',          2),
    ('count',  'knight_hanging',             2),
    ('count',  'knight_en_prise',            2),
    ('count',  'knight_attacked_by_lower_value', 2),

    # bishops
    ('count',  'bishop_undefended',          2),
    ('count',  'bishop_hanging',             2),
    ('count',  'bishop_en_prise',            2),
    ('count',  'bishop_attacked_by_lower_value', 2),

    # rooks
    ('count',  'rook_undefended',            2),
    ('count',  'rook_hanging',               2),
    ('count',  'rook_en_prise',              2),
    ('count',  'rook_attacked_by_lower_value', 2),

    # queens (binary but per-side , shape=2 vector)
    ('binary', 'queen_undefended',           2),
    ('binary', 'queen_hanging',              2),
    ('binary', 'queen_en_prise',             2),
    ('binary', 'queen_attacked_by_lower_value', 2),

    # material (pair)
    ('count',  'material',                   2),
    ('piece_to_move', 'piece_to_move', 6)
]


# -------------------------
# Utilities: adjust head weights on the fly
# -------------------------

def set_head_weights(model, new_weights):
    curr = getattr(model, "_head_loss_weights", None)
    if curr is None:
        try:
            cfg = model.get_compile_config()
            curr = dict(cfg.get("loss_weights") or {})
        except Exception:
            curr = {o.name.split(':')[0].split("/")[0]: 1.0 for o in model.outputs}
    curr.update(new_weights)

    losses = getattr(model, "_head_losses", None)
    if losses is None:
        losses = {}
        for o in model.outputs:
            name = o.name.split(':')[0].split("/")[0]
            losses[name] = "mse" if o.shape[-1] != 1 else "mse"

    model.compile(optimizer=model.optimizer, loss=losses, loss_weights=curr)
    model._head_loss_weights = dict(curr)
    
    
new_weights = {
    'king_ray_exposure': 0.05,
    'king_ring_pressure': 0.1,
    'king_pawn_shield': 0.1,
    'king_escape_square': 0.1,
    'pawn_undefended': 0.1,
    'pawn_hanging': 0.1,
    'pawn_en_prise': 0.1,
    'knight_undefended': 0.1,
    'knight_hanging': 0.1,
    'knight_en_prise': 0.1,
    'knight_attacked_by_lower_value': 0.1,
    'bishop_undefended': 0.1,
    'bishop_hanging': 0.1,
    'bishop_en_prise': 0.1,
    'bishop_attacked_by_lower_value': 0.1,
    'rook_undefended': 0.1,
    'rook_hanging': 0.1,
    'rook_en_prise': 0.1,
    'rook_attacked_by_lower_value': 0.1,
    'queen_undefended': 0.2,
    'queen_hanging': 0.2,
    'queen_en_prise': 0.2,
    'queen_attacked_by_lower_value': 0.2,
    'material': 0.025,
    'piece_to_move': 0.05,
    'value': 3.0,
    'policy': 5.0
 }

ignore_aux = {k:0.0 for k in new_weights}
ignore_aux['value'] = 2.0
ignore_aux['policy'] = 2.0

#%%

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


def collect_stockfish_data(engine, depth=2, n_init=None, n_rand=5):
    if n_init is not None:
        board = random_init(n_init)
    else:
        board = random_init(np.random.randint(0, 3))
        
    all_X, all_Y = [], []
    while not board.is_game_over():
        X_tokens = board_to_tokens(board)
        legal_mask = cbe.legal_mask_8x8x73(board).reshape(-1)
        all_X.append({'policy_mask': legal_mask, "tokens": X_tokens})
        
        info_list = engine.analyse(
            board, multipv=len(list(board.legal_moves)),
            limit=chess.engine.Limit(depth=depth),
            info=chess.engine.Info.ALL
        )
        
        Y = cbe.build_training_targets_value_policy(board, info_list)
        Y.update(feats.all_king_exposure_features(board))
        Y.update(feats.all_piece_features(board))
        Y['material'] = feats.get_piece_value_sum(board)
        Y['piece_to_move'] = cbe.piece_to_move_target(board, Y['policy_logits'])
        all_Y.append(Y)
        
        move_to_make = random.choice(info_list[:n_rand])['pv'][0]
        
        board.push(move_to_make)
        
    return all_X, all_Y


def prepare_data_for_model(X, Y, model):
    inputs, outputs = {}, {}

    # Inputs
    for name, tensor in zip(model.input_names, model.inputs):
        vals = [x[name] for x in X]
        arr = np.array(vals)

        # Flatten mask to (B,4672)
        if name == "policy_mask":
            arr = arr.reshape(len(X), -1)
        # Ensure sequence length matches
        exp_len = tensor.shape[1]
        if arr.shape[1] != exp_len:
            raise ValueError(f"Input {name} has shape {arr.shape}, expected length {exp_len}")

        inputs[name] = arr

    # Outputs
    for name, tensor in zip(model.output_names, model.outputs):
        key = "policy_logits" if name == "policy" else name
        vals = [y[key] for y in Y]
        arr = np.array(vals)

        # Flatten policy targets
        if name == "policy":
            arr = arr.reshape(len(Y), -1)

        # Ensure trailing dim matches
        exp_dim = tensor.shape[1]
        if arr.shape[1] != exp_dim:
            raise ValueError(f"Output {name} has shape {arr.shape}, expected dim {exp_dim}")

        outputs[name] = arr

    return inputs, outputs


def prepare_X(X, model):
    inputs =  {}

    # Inputs
    for name, tensor in zip(model.input_names, model.inputs):
        vals = [x[name] for x in X]
        arr = np.array(vals)

        # Flatten mask to (B,4672)
        if name == "policy_mask":
            arr = arr.reshape(len(X), -1)

        # Ensure sequence length matches
        exp_len = tensor.shape[1]
        if arr.shape[1] != exp_len:
            raise ValueError(f"Input {name} has shape {arr.shape}, expected length {exp_len}")

        inputs[name] = arr
    
    return inputs


def load_model(model_loc):
    custom = {
        "masked_policy_ce": masked_policy_ce,
        "TransformerBlock": TransformerBlock,
    }
    model = keras.models.load_model(model_loc, custom_objects=custom)
    return model


def analyze_board(board, engine, multipv=5, depth=10):
    """
    Analyze a bulletchess.Board with Stockfish (via python-chess).
    Returns a list of dicts like python-chess's engine.analyse(multipv=n).
    """
    # Convert bulletchess -> python-chess
    if isinstance(board, Board):
        a_board = chess.Board(board.fen())
    else:
        a_board = board
        
    
    return engine.analyse(
        a_board,
        multipv=multipv,
        limit=chess.engine.Limit(depth=depth),
        info=chess.engine.Info.ALL
    )
    

def show_board(board, bot_color=True):
    s_board = chess.Board(board.fen()) if isinstance(board, Board) else board
    clear_output(wait=True)
    display(SVG(chess.svg.board(board=s_board, flipped=not bot_color)))
    
#%%
# build model
n_moves = 4672  # your move-index space
#model = build_transformer_full_with_aux(
#     vocab_size=16, seq_len=70, d_model=256, num_heads=4, num_layers=4, ff_dim=256,
#     n_moves=n_moves, use_mask=True, aux_configs=aux_configs,
#     aux_into_heads=False, aux_weight=0.05, value_weight=2.0, policy_weight=2.0
# )

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


import os
MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"
model = load_model(os.path.join(MODEL_DIR, "transformer_model_v400.h5"))
#set_head_weights(model, ignore_aux)
all_evals = pd.DataFrame()
#%%
MCTS_CACHE = {}

import gc


def evaluate_terminal(board):
    """
    Return +1 if White wins, -1 if Black wins, 0 for draw.
    Assumes `board` is a terminal position.
    """
    if board in CHECKMATE:
        # side to move has been checkmated
        return -1 if board.turn == WHITE else 1

    elif is_game_over(board):
        # stalemate, repetition, insufficient material, etc.
        return 0

    else:
        raise Exception("Non-terminal board passed to evaluate_terminal")


def is_game_over(board):
    return any([board in state for state in [CHECKMATE, DRAW, FORCED_DRAW]])


class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.fen = board.fen()
        self.short_fen = " ".join(self.fen.split(" ")[:4])
        self.parent = parent
        self.move = move
        self.board_turn = board.turn
        
        self.moves = []
        self.children = []
        self.N = np.zeros(0, dtype=np.int32)
        self.W = np.zeros(0, dtype=np.float32)
        self.P = np.zeros(0, dtype=np.float32)

        self.value = None
    
    def reconstruct_board(self):
        return Board.from_fen(self.fen)
    
    def average_values(self):
        return np.divide(self.W, self.N, out=np.zeros_like(self.W), where=self.N > 0)

    def get_short_fen(self, board):
        return " ".join(board.fen().split(" ")[:4])

    def is_fully_expanded(self):
        return len(self.children) == len(self.reconstruct_board().legal_moves())

    def average_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def predict_with_priors(self, model, legal_moves=None):
        # Cache check
        if self.short_fen in MCTS_CACHE:
            self.value, self.P = MCTS_CACHE[self.short_fen]
            return self.value
        
        tmp_board = self.reconstruct_board()
        if legal_moves is None:
            legal_moves = list(tmp_board.legal_moves())
    
        # Terminal state
        if is_game_over(tmp_board):
            self.value = 1.5*evaluate_terminal(tmp_board)
            n = len(legal_moves)
            
            self.P = np.array([], dtype=np.float32)
            if n > 0:
                self.P = np.full(n, 1.0 / n, dtype=np.float32) 
                
            MCTS_CACHE[self.short_fen] = (self.value, self.P)
            return self.value
    
        # --- NN eval ---
        # Encode board into model input
        tokens = board_to_tokens_bc(tmp_board)   # (65,) int32
        legal_mask = cbe.legal_mask_bc(tmp_board).reshape(-1)  # (4672,)
        X = {
            "tokens": np.expand_dims(tokens, 0),             # (1, 70) or whatever tokens len is
            "policy_mask": np.expand_dims(legal_mask, 0)     # (1, 4672)
        }
        preds = model.predict(X, verbose=0)
    
        # Value head
        self.value = float(preds["value"][0][0])
    
        # Policy head (raw 8x8x73)
        policy_logits = preds["policy"][0].reshape((8, 8, 73))
        
        # for testing
        # self.value = np.random.uniform(-1, 1)
        # policy_logits = np.random.uniform(0, 1, size=(8, 8, 73))
        
        # Align priors with legal moves
        priors = []
        for mv in legal_moves:
            fr, ff, pl = cbe.move_to_8x8x73(mv, tmp_board)
            priors.append(policy_logits[fr, ff, pl])
            
        self.P = np.array(priors, dtype=np.float32)
    
        # Normalize
        if self.P.sum() > 0:
            self.P /= self.P.sum()
        else:
            self.P = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)
    
        # Cache result
        MCTS_CACHE[self.short_fen] = (self.value, self.P)
        return self.value


    # def expand(self, model=None):
    #     if self.children:  # already expanded
    #         return self.value
        
    #     tmp_board = self.reconstruct_board()
    #     legal_moves = list(tmp_board.legal_moves())
    #     self.moves = legal_moves
    #     self.children = []

    #     self.value = self.predict_with_priors(model, legal_moves)

    #     self.N = np.zeros(len(legal_moves), dtype=np.int32)
    #     self.W = np.zeros(len(legal_moves), dtype=np.float32)

    #     # Create children
    #     for mv in legal_moves:
    #         tmp_board.apply(mv)
    #         self.children.append(MCTSNode(tmp_board, parent=self, move=mv))
    #         tmp_board.undo()

    #     return self.value
    
    def expand(self, model=None):
        if self.children:
            return self.value

        tmp_board = self.reconstruct_board()
        legal_moves = list(tmp_board.legal_moves())
        self.moves = legal_moves
        self.children = []

        self.value = self.predict_with_priors(model, legal_moves)

        self.N = np.zeros(len(legal_moves), dtype=np.int32)
        self.W = np.zeros(len(legal_moves), dtype=np.float32)

        # Create children
        for mv in legal_moves:
            tmp_board.apply(mv)
            self.children.append(MCTSNode(tmp_board, parent=self, move=mv))
            tmp_board.undo()

        # log memory after expansion
        print(f"Expanded {len(legal_moves)} moves, RSS MB:", proc.memory_info().rss / 1e6)
        n = gc.collect()
        print("GC collected:", n)
        print("Memory MB after:", proc.memory_info().rss / 1e6)

        return self.value


def add_dirichlet_noise(priors, epsilon=0.25, alpha=0.3):
    """
    priors: dict {move: prior_probability}
    returns a new dict with Dirichlet noise mixed in.
    """
    moves = list(priors.keys())
    n = len(moves)
    if n == 0:
        return priors
    noise = np.random.dirichlet([alpha] * n)
    return {m: (1 - epsilon) * priors[m] + epsilon * n_
            for m, n_ in zip(moves, noise)}


def select_child_puct(
    node, c_puct=1.5, prefer_higher=True, root_noise=False,
    epsilon=0.25, alpha=0.3
):
    if not node.children:
        return None

    # Optionally inject Dirichlet noise at the root
    priors = node.P.copy()
    if root_noise:
        priors_dict = {m: p for m, p in zip(node.moves, priors)}
        noisy_priors = add_dirichlet_noise(priors_dict, epsilon, alpha)
        priors = np.array([noisy_priors[m] for m in node.moves], dtype=np.float32)

    N_parent = max(1, np.sum(node.N))
    Q = node.average_values()
    if not prefer_higher:
        Q = -Q

    U = c_puct * priors * (np.sqrt(N_parent) / (1.0 + node.N))
    scores = Q + U

    # tie break with tiny random jitter
    idx = np.argmax(scores + 1e-6*np.random.randn(*scores.shape))
    return node.children[idx], idx


from bulletchess import WHITE, BLACK
# def simulate(model, root, max_depth=32, c_puct=1.5):
#     path = []
#     node = root
#     depth = 0

#     # selection/expansion
#     while depth < max_depth:
#         root_noise = depth == 0
        
#         if not node.children:
#             node.expand(model)
#             break
        
#         is_white = node.board.turn == WHITE
#         child = select_child_puct(
#             node, prefer_higher=is_white, c_puct=c_puct, root_noise=root_noise
#         )
        
#         idx = node.children.index(child)
#         path.append((node, idx))
#         node = child

#         if is_game_over(node.board):
#             break
        
#         depth += 1

#     value = node.predict_with_priors(model)

#     # BACKPROP
#     for parent, idx in path:
#         parent.N[idx] += 1
#         parent.W[idx] += value
        

def collect_leaves(root, max_depth=32, c_puct=1.5, epsilon=0.25, alpha=0.3):
    leaves = []
    paths = []

    node = root
    path = []
    depth = 0

    while depth < max_depth:
        if not node.children:  # leaf
            leaves.append((node, list(node.reconstruct_board().legal_moves())))
            paths.append(path)
            break

        prefer_higher = (node.board_turn == WHITE)
        root_noise = (depth == 0)  # ✅ only at root

        child, idx = select_child_puct(
            node,
            prefer_higher=prefer_higher,
            c_puct=c_puct,
            root_noise=root_noise,
            epsilon=epsilon,
            alpha=alpha
        )

        path.append((node, idx))
        node = child

        if is_game_over(node.reconstruct_board()):
            leaves.append((node, []))
            paths.append(path)
            break

        depth += 1

    return leaves, paths


def batch_evaluate(model, leaves):
    # Encode all boards into one batch
    tokens_batch = []
    masks_batch = []

    for node, legal_moves in leaves:
        tokens = board_to_tokens_bc(node.reconstruct_board())
        mask = cbe.legal_mask_bc(node.reconstruct_board()).reshape(-1)
        tokens_batch.append(tokens)
        masks_batch.append(mask)

    X = {
        "tokens": np.array(tokens_batch, dtype=np.int32),
        "policy_mask": np.array(masks_batch, dtype=np.float32)
    }
    preds = model.predict(X, verbose=0, batch_size=128)
    #preds = {'value': np.random.uniform(-1, 1, size=(len(leaves), 1)), "policy": np.random.uniform(0, 1, size=(len(leaves), 8*8*73))}

    # Scatter results back
    for (node, legal_moves), value, policy_flat in zip(leaves, preds["value"], preds["policy"]):
        node.value = float(value[0])
        node.moves = legal_moves
    
        if legal_moves:
            # reshape flat policy vector -> (8,8,73)
            policy = policy_flat.reshape(8, 8, 73)
    
            node.P = np.array(
                [policy[cbe.move_to_8x8x73(mv, node.reconstruct_board())] for mv in legal_moves],
                dtype=np.float32
            )
            # normalize
            s = node.P.sum()
            if s > 0:
                node.P /= s
            else:
                node.P = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)
        else:
            node.P = np.array([], dtype=np.float32)


def simulate_batch(model, root, n_sims=32, max_depth=32):
    # Ensure root is expanded once
    if not root.children:
        root.expand(model)

    all_paths = []
    leaves = []

    # 1. collect n_sims distinct leaves
    for _ in range(n_sims):
        new_leaves, paths = collect_leaves(root, max_depth=max_depth)
        leaves.extend(new_leaves)
        all_paths.extend(paths)

    # 2. batch evaluate
    batch_evaluate(model, leaves)
    gc.collect()

    # 3. backprop each path
    for (node, _), path in zip(leaves, all_paths):
        value = node.value
        for parent, idx in path:
            parent.N[idx] += 1
            parent.W[idx] += value


def choose_move(root, engine, model, board, max_sims=500):
    print("Thinking... ")
    start = time.time()
    n_sims = 16
    for s in range(max_sims):
        simulate_batch(model, root, n_sims=n_sims, max_depth=32)
    
    stop = time.time()
    print(f"Completed {n_sims * (s+1)} simulations in {round(stop-start, 2)} seconds")
    
    # Get stats
    visits = root.N
    Q = np.divide(root.W, root.N, out=np.zeros_like(root.W), where=root.N > 0)
    
    # Sort children by visit count (descending)
    order = np.argsort(-visits)  # minus = descending
    print("\nTop candidate moves:")
    for idx in order[:5]:
        move = root.moves[idx]
        node = root.children[idx]
        san = move.san(root.reconstruct_board())
        v_child = node.predict_with_priors(model)
        print(f"{move.uci():<6} ({san})  visits={visits[idx]:<4} "
              f"Q={Q[idx]:.3f}  V(model)={v_child:.3f}")
    
    # Pick the best move (highest visits)
    best_idx = order[0]
    best_move = root.moves[best_idx]
    best_san = best_move.san(root.reconstruct_board())
    print(f"\nChosen move: {best_move.uci()} ({best_san})  "
          f"(visits={visits[best_idx]}, Q={Q[best_idx]:.3f})")
    
    return best_move

#%%
bot_color = WHITE if np.random.uniform() < 0.5 else BLACK
board = Board()
root = MCTSNode(board)
data = []
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
    while not is_game_over(board):
        show_board(board, bot_color=bot_color == WHITE)
        sf_eval = analyze_board(board, engine)
        if board.turn == bot_color:
            best_move = choose_move(root, engine, model, board, max_sims=50)
            print("\nStockfish top 3 moves: ")
            moves = sf_eval[:3]
            for i, mv in enumerate(moves):
                bc_move = bulletchess.Move.from_uci(str(mv['pv'][0]))
                san = bc_move.san(board)
                sc = cbe.pawns_to_winprob(cbe.score_to_cp_white(mv['score']))
                print(f"{i+1}. {san}: {np.round(sc, 3)}")
                
            board.apply(best_move)
            del root
            root = MCTSNode(board)
            print()
            
        else:
            #sf_move = random.choice(sf_eval[:2])['pv'][0]
            sf_move = sf_eval[0]['pv'][0]
            bc_move = bulletchess.Move.from_uci(str(sf_move))
            san = bc_move.san(board)
            print(f"\nStockfish plays {san}\n")
            time.sleep(0.5)            
            board.apply(bc_move)
            del root
            root = MCTSNode(board)
        

#%%
import subprocess

proc = subprocess.Popen(
    [SF_LOC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1
)

def send(cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()

def read_fens(n):
    """Read until we’ve collected n FEN lines"""
    fens = []
    while len(fens) < n:
        line = proc.stdout.readline()
        if not line:
            continue
        print(line)
        line = line.strip()
        
        if line.startswith("Fen:"):
            fens.append(line.split("Fen:")[1].strip())
    return fens

# init
send("uci")
while True:
    line = proc.stdout.readline()
    if "uciok" in line:
        break

# moves to test
moves = ["e2e4", "d2d4", "c2c4", "g1f3"]

# build one big command block
cmds = []
for mv in moves:
    cmds.append(f"position startpos moves {mv}")
    cmds.append("d")

batch = "\n".join(cmds) + "\n"
proc.stdin.write(batch)
proc.stdin.flush()

# read back 4 FENs
fens = read_fens(len(moves))
for mv, fen in zip(moves, fens):
    print(mv, "->", fen)

# cleanup
send("quit")
proc.wait()

#%%
proc = subprocess.Popen(
    [SF_LOC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1
)

def send(cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()

def read_until(tokens):
    """Read until we see one of the tokens, return the line"""
    while True:
        line = proc.stdout.readline().strip()
        if not line:
            continue
        for tok in tokens:
            if tok in line:
                return line

def is_terminal(fen):
    # load pos
    send(f"position fen {fen}")

    # check legal moves
    send("go perft 1")
    moves = []
    while True:
        line = proc.stdout.readline().strip()
        if not line:
            continue
        if line.startswith("Nodes searched"):
            break
        if ":" in line:
            moves.append(line.split(":")[0].strip())

    # ask SF for eval
    send("go depth 1")
    eval_line = read_until(["score"])
    
    if "mate" in eval_line:
        mate_n = eval_line.split("mate")[1].split()[0]
        return True, f"mate in {mate_n}"
    elif not moves:
        return True, "stalemate"
    else:
        return False, "ongoing"

# --- test positions ---
positions = {
    "Fool's mate (white mated)": "rnb1kbnr/pppp1ppp/8/4p3/7q/5P2/PPPPP1PP/RNBQKBNR w KQkq - 1 3",
    "Stalemate (black)": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "Normal position": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
}

for name, fen in positions.items():
    term, result = is_terminal(fen)
    print(name, "->", result)

# cleanup
send("quit")
proc.wait()

