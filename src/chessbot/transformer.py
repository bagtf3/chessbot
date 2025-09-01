import chess
import chess.engine
import pandas as pd
import numpy as np
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt

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
from chessbot.model import masked_policy_ce

from chessbot import SF_LOC, ENDGAME_LOC

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


#%%
# build model
n_moves = 4672  # your move-index space
#model = build_transformer_full_with_aux(
#     vocab_size=16, seq_len=70, d_model=256, num_heads=4, num_layers=4, ff_dim=256,
#     n_moves=n_moves, use_mask=True, aux_configs=aux_configs,
#     aux_into_heads=False, aux_weight=0.05, value_weight=2.0, policy_weight=2.0
# )

import os
MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"
model = load_model(os.path.join(MODEL_DIR, "transformer_model_v400.h5"))
#set_head_weights(model, ignore_aux)
all_evals = pd.DataFrame()
#%%
# phase 1: learn from stockfish evaluations
for outer in range(301, 401):
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
        all_X, all_Y = [], []
        for game in range(10):
            print(f"Outer {outer} game {game}: {len(all_X)} positions so far")
            X, Y = collect_stockfish_data(engine, depth=3, n_init=20, n_rand=6+game)
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
                f"C:/Users/Bryan/Data/chessbot_data/models/transformer_model_v{outer}.h5"
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

#%%
#model.save("C:/Users/Bryan/Data/chessbot_data/models/transformer_model_initial.h5")

#%%  
    
def choose_move(model, board, sf_info, top_n=3):
    moves, all_X = [], []
    checkmate_found = False
    checkmate_move = None

    for move in list(board.legal_moves):
        board.push(move)
        if board.is_game_over():
            board.pop()
            checkmate_found = True
            checkmate_move = move
            break

        X_tokens = board_to_tokens(board)
        legal_mask = cbe.legal_mask_8x8x73(board).reshape(-1)
        all_X.append({'policy_mask': legal_mask, "tokens": X_tokens})
        moves.append(move)
        board.pop()
    
    if checkmate_found:
        print(f"Checkmate found , playing {board.san(checkmate_move)}")
        return checkmate_move

    # Model predictions
    X = prepare_X(all_X, model)
    preds = model.predict(X, verbose=0)['value'].reshape(-1)

    # Decide ranking direction
    if board.turn:  # White , higher is better
        top_idx = np.argsort(-preds)[:top_n]
    else:           # Black , lower is better
        top_idx = np.argsort(preds)[:top_n]

    # Pre-compute SF move ranking (dict: move , rank)
    sf_ranking = {}
    for i, entry in enumerate(sf_info, start=1):
        if "pv" in entry and entry["pv"]:
            mv = entry["pv"][0]
            sf_ranking[mv] = i

    print("\n--- Model vs Stockfish ---")
    chosen_move = None
    for rank, idx in enumerate(top_idx, start=1):
        mv = moves[idx]
        san = board.san(mv)
        val = preds[idx]

        # Stockfish CP and rank if available
        sf_rank = sf_ranking.get(mv, None)
        sf_score = None
        for entry in sf_info:
            if "pv" in entry and entry["pv"] and entry["pv"][0] == mv:
                sf_score = cbe.score_to_cp_white(entry["score"])
                break

        if sf_rank is not None:
            print(f"Model#{rank}: {san:5s} val={val:+.3f} | SF={sf_score:+.2f} (rank {sf_rank})")
        else:
            print(f"Model#{rank}: {san:5s} val={val:+.3f} | SF=--- (not in top list)")

        if rank == 1:
            chosen_move = mv

    print("--------------------------\n")
    return chosen_move

bot_color = np.random.uniform() < 0.5
board = random_init(2)
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
    while not board.is_game_over():
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=board, flipped=not bot_color)))
        time.sleep(0.75)
        
        sf_eval = engine.analyse(
            board, multipv=len(list(board.legal_moves)), limit=chess.engine.Limit(depth=1),
            info=chess.engine.Info.ALL
        )
        
        if board.turn == bot_color:
            best_move = choose_move(model, board, sf_eval)
            moves = sf_eval[:3]
            board.push(best_move)
            time.sleep(1.5)
            
        else:
            sf_move = sf_eval[0]['pv'][0]
            san = board.san(sf_move)
            board.push(sf_move)
            time.sleep(0.75)
#%%
MCTS_CACHE = {}
#%%
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move

        # Children bookkeeping
        self.moves = []        # list of moves in same order as arrays
        self.children = []     # list of MCTSNode objects
        self.N = np.zeros(0)   # visits
        self.W = np.zeros(0)   # total value
        self.P = np.zeros(0)   # priors

        self.value = None
        self.short_fen = " ".join(board.fen().split(" ")[:4])

    def average_values(self):
        return np.divide(self.W, self.N, out=np.zeros_like(self.W), where=self.N>0)

    
    def get_short_fen(self, board):
        return " ".join(board.fen().split(" ")[:4])
    
    def five_or_less(self, board):
        return len(board.piece_map()) <= 5
    
    def lookup_endgame(self):
        with chess.syzygy.open_tablebase(self.endgame_loc) as tablebase:
            # boost endgame scores by 1.5
            result = 1.5*np.clip(tablebase.probe_wdl(self.board), -1, 1)
        return result if self.board.turn else -1*result
        
    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def average_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def predict_with_priors(self, model, legal_moves=None):
        # Cache check
        if self.short_fen in MCTS_CACHE:
            self.value, self.P = MCTS_CACHE[self.short_fen]
            return self.value
    
        if legal_moves is None:
            legal_moves = list(self.board.legal_moves)
    
        # Terminal state
        if self.board.is_game_over():
            self.value = 2 * evaluate_terminal(self.board)
            n = len(legal_moves)
            self.P = np.full(n, 1.0 / n, dtype=np.float32) if n > 0 else np.array([], dtype=np.float32)
            MCTS_CACHE[self.short_fen] = (self.value, self.P)
            return self.value
    
        # Endgame tablebase
        if self.five_or_less(self.board):
            self.value = self.lookup_endgame()
            n = len(legal_moves)
            self.P = np.full(n, 1.0 / n, dtype=np.float32)
            MCTS_CACHE[self.short_fen] = (self.value, self.P)
            return self.value
    
        # NN eval
        legal_mask = cbe.legal_mask_8x8x73(self.board).reshape(-1)
        X = {'tokens': board_to_tokens(self.board), 'policy_mask': legal_mask}
        #preds = model.predict(prepare_X([X], model), verbose=0)
    
        #self.value = preds['value'][0][0]
        self.value = np.random.uniform(-1, 1)
        #policy = preds['policy'][0].reshape((8, 8, 73))
        policy = np.random.uniform(0, 1, size=(8, 8, 73))
    
        # Align priors with move order
        self.moves = legal_moves
        self.P = np.array(
            [policy[cbe.move_to_8x8x73(m, self.board)] for m in legal_moves],
            dtype=np.float32
        )
    
        # Cache
        MCTS_CACHE[self.short_fen] = (self.value, self.P)
        return self.value
    
    def expand(self, model):
        if self.children:  # already expanded
            return self.value

        legal_moves = list(self.board.legal_moves)
        self.moves = legal_moves
        self.children = []

        # Predict priors + value
        self.value = self.predict_with_priors(model, legal_moves)

        self.N = np.zeros(len(legal_moves), dtype=np.int32)
        self.W = np.zeros(len(legal_moves), dtype=np.float32)

        # Pre-create child slots (lazy init possible too)
        for mv in legal_moves:
            new_board = self.board.copy()
            new_board.push(mv)
            self.children.append(MCTSNode(new_board, parent=self, move=mv))

        return self.value


    def best_child(self):
        return max(self.children.values(), key=lambda c: c.average_value())


def add_dirichlet_noise(priors, epsilon=0.25, alpha=0.3):
    moves = list(priors.keys())
    n = len(moves)
    noise = np.random.dirichlet([alpha] * n)
    return {m: (1 - epsilon) * priors[m] + epsilon * n_
            for m, n_ in zip(moves, noise)}


def select_child_puct(
    node, c_puct=1.5, prefer_higher=True, root_noise=False, epsilon=0.25, alpha=0.3
):
    if not node.children:
        return None

    N_parent = max(1, np.sum(node.N))
    Q = node.average_values()
    if not prefer_higher:
        Q = -Q

    U = c_puct * node.P * (np.sqrt(N_parent) / (1.0 + node.N))
    scores = Q + U

    idx = np.argmax(scores)
    return node.children[idx]


def simulate(model, root, max_depth=32, c_puct=1.5):
    path = []
    node = root
    depth = 0

    # selection/expansion
    while depth < max_depth:
        root_noise = depth == 0
        
        if not node.children:
            node.expand(model)
            break

        child = select_child_puct(
            node, prefer_higher=node.board.turn, c_puct=c_puct, root_noise=root_noise
        )
        
        idx = node.children.index(child)
        path.append((node, idx))
        node = child

        if node.board.is_game_over():
            break
        
        depth += 1

    value = node.predict_with_priors(model)

    # BACKPROP
    for parent, idx in path:
        parent.N[idx] += 1
        parent.W[idx] += value
        

def choose_move(root, engine, model, board, max_sims=500):
    print("Thinking... ")
    start = time.time()
    
    for s in range(max_sims):
        simulate(model, root, max_depth=32)
    
    stop = time.time()
    print(f"Completed {s+1} simulations in {round(stop-start, 2)} seconds")
    
    # Get stats
    visits = root.N
    Q = np.divide(root.W, root.N, out=np.zeros_like(root.W), where=root.N > 0)
    
    # Sort children by visit count (descending)
    order = np.argsort(-visits)  # minus = descending
    print("\nTop candidate moves:")
    for idx in order[:5]:
        move = root.moves[idx]
        node = root.children[idx]
        san = root.board.san(move)
        v_child = node.predict_with_priors(model)
        print(f"{move.uci():<6} ({san})  visits={visits[idx]:<4} "
              f"Q={Q[idx]:.3f}  V(model)={v_child:.3f}")
    
    # Pick the best move (highest visits)
    best_idx = order[0]
    best_move = root.moves[best_idx]
    best_san = root.board.san(best_move)
    print(f"\nChosen move: {best_move.uci()} ({best_san})  "
          f"(visits={visits[best_idx]}, Q={Q[best_idx]:.3f})")
    
    return best_move, root


def advance_root(root, move):
    """Advance the MCTS root to the child corresponding to `move`."""
    if move in root.moves:
        idx = root.moves.index(move)
        new_root = root.children[idx]
        new_root.parent = None  # cut link
        return new_root
    else:
        print("MOVE NOT FOUND")
        # If move not in current root (e.g. unexpected opp move), rebuild from scratch
        new_board = root.board.copy()
        new_board.push(move)
        return MCTSNode(new_board)


from IPython.display import display, clear_output, SVG
import chess.svg, chess.pgn
import chessbot.utils as cbu

bot_color = np.random.uniform() < 0.9995
board = chess.Board()
board.push(chess.Move.from_uci("e2e4"))
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
            best_move, root = choose_move(root, engine, model, board, max_sims=500)
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
            # best_move, root = choose_move(root, engine, model, board, max_sims=100)
            # root = advance_root(root, best_move)
            # print("\nStockfish top 3 moves: ")
            # moves = sf_eval[:3]
            # for i, mv in enumerate(moves):
            #     san = board.san(mv['pv'][0])
            #     sc = cbe.pawns_to_winprob(cbe.score_to_cp_white(mv['score']))
                
            #     print(f"{i+1}. {san}: {np.round(sc, 3)}")
                
            # print()
            # board.push(best_move)
            sf_move = random.choice(sf_eval[:2])['pv'][0]
            san = board.san(sf_move)
            print(f"\nStockfish plays {san}\n")
            root = advance_root(root, sf_move)
            board.push(sf_move)
            time.sleep(0.5)

#%%
def plot_attention_on_board(board, attn_scores, square, head=0):
    vec = attn_scores[0, head, square, :64]
    mat = vec.reshape(8,8)
    plt.figure(figsize=(5,5))
    plt.imshow(mat, cmap="hot", origin="upper")
    plt.colorbar()
    piece = board.piece_at(square)
    piece_symbol = piece.symbol() if piece else "·"
    plt.title(f"Attention from {piece_symbol} at {chess.square_name(square)} (head {head})")
    plt.xticks(range(8), ["a","b","c","d","e","f","g","h"])
    plt.yticks(range(8), list(range(8,0,-1)))
    plt.show()


board = random_init(26)
knight_squares = list(board.pieces(chess.KNIGHT, chess.WHITE)) + \
                 list(board.pieces(chess.KNIGHT, chess.BLACK))
                 
king_squares = list(board.pieces(chess.KING, chess.WHITE)) + \
                 list(board.pieces(chess.KING, chess.BLACK))
                 
queen_squares = list(board.pieces(chess.QUEEN, chess.WHITE)) + \
                 list(board.pieces(chess.QUEEN, chess.BLACK))
                 
print("Knights at:", [chess.square_name(sq) for sq in knight_squares])

probe = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[4].output   # if this is your TransformerBlock
)

to_score = board if board.turn else board.mirror()
tokens = board_to_tokens(to_score)
legal_mask = cbe.legal_mask_8x8x73(to_score).reshape(-1)

this_X = {
    "policy_mask": legal_mask.reshape(1, -1),
    "tokens": tokens.reshape(1, -1)             
}

block_out, attn_scores = probe(this_X)
attn_scores = attn_scores.numpy()
    
# visualize for each knight, head 0
for sq in [chess.E4, chess.E5, chess.D4, chess.D5]:
    plot_attention_on_board(board, attn_scores, sq, head=0)
    
for sq in knight_squares:
    plot_attention_on_board(board, attn_scores, sq, head=0)
    
for sq in king_squares:
    plot_attention_on_board(board, attn_scores, sq, head=1)
    
for sq in queen_squares:
    plot_attention_on_board(board, attn_scores, sq, head=0)





