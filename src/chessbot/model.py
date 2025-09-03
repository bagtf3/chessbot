import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"
HE = "he_normal"

import tensorflow as tf
from tensorflow import keras

print("Built with CUDA?", tf.test.is_built_with_cuda())
print("GPUs available:", tf.config.list_physical_devices('GPU'))

from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, Add,
    GlobalAveragePooling2D, Dense, Lambda
)


def save_model(model, model_loc):
    model.save(model_loc, save_format='h5')
    
    
def load_model(model_loc):
    custom = {
        "masked_policy_ce": masked_policy_ce,
        'weighted_bce_legality': weighted_bce_legality
    }
    
    model = keras.models.load_model(model_loc, custom_objects=custom)
    
    return model
    
    
def masked_policy_ce(y_true, y_pred, eps=1e-7):
    """
    y_true: [B,8,8,73] target probs (illegal=0; legal>0; sums to 1 over legal)
    y_pred: [B,8,8,73] logits
    DIDNT WORK WORTH A DARN
    """
    B = tf.shape(y_true)[0]
    t = tf.reshape(y_true, [B, -1])        # targets
    z = tf.reshape(y_pred, [B, -1])        # logits

    # Build mask from targets
    m = tf.cast(t > 1e-6, z.dtype)          # [B,N]

    # (Optional but recommended) renormalize only over legal entries
    t_legal = t * m
    denom = tf.reduce_sum(t_legal, axis=1, keepdims=True)
    t_legal = t_legal / (denom + eps)

    # Masked log-softmax over legal entries only
    neg_inf = tf.cast(-1e9, z.dtype)
    z_masked = tf.where(m > 0, z, neg_inf)
    logp = z_masked - tf.reduce_logsumexp(z_masked, axis=1, keepdims=True)

    # Cross-entropy over legal entries
    loss = -tf.reduce_sum(t_legal * logp, axis=1)   # [B]
    return loss


def policy_ce_unmasked(y_true, y_pred):
    # y_true, y_pred: [B,8,8,73]
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    # From logits = True → applies softmax+log inside
    return keras.losses.categorical_crossentropy(y_true_f, y_pred_f, from_logits=True)


def masked_mse(y_true, y_pred):
    """
    y_true: [B,8,8,73] with 0 for illegal, winprob [0,1] for legal moves
    y_pred: [B,8,8,73] sigmoid outputs
    """
    mask = tf.cast(y_true > 0, tf.float32)  # 1 where legal, 0 where illegal
    diff = (y_true - y_pred) ** 2
    # only average over legal moves
    return tf.reduce_sum(diff * mask) / (tf.reduce_sum(mask) + 1e-7)


def res_block(x, filters, leak=0.05):
    skip = x
    y = Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer=HE)(x)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=leak)(y)
    y = Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer=HE)(y)
    y = BatchNormalization()(y)
    y = Add()([skip, y])
    return LeakyReLU(alpha=leak)(y)


def build_stump(input_shape=(8, 8, 23), width=128, n_blocks=4, leak=0.05):
    inputs = Input(shape=input_shape, name="board")
    x = Conv2D(width, 3, padding="same", use_bias=False, kernel_initializer=HE)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak)(x)
    
    for _ in range(n_blocks):
        x = res_block(x, width, leak=leak)
        
    x = Conv2D(width // 2, 1, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    
    trunk = LeakyReLU(alpha=leak, name="trunk")(x)
    return Model(inputs, trunk, name="stump")


def attach_policy_value_heads(trunk, leak=0.05):
    policy_logits = Conv2D(73, 1, padding="same", name="policy_logits")(trunk)
    v = Conv2D(32, 1, padding="same")(trunk)
    v = LeakyReLU(alpha=leak)(v)
    v = GlobalAveragePooling2D()(v)
    v = Dense(256)(v); v = LeakyReLU(alpha=leak)(v)
    value = Dense(1, activation="sigmoid", name="value")(v)
    return policy_logits, value


def count_vec_head(trunk_vec, name, shape):
    x = Dense(128, name=f'{name}_dense1')(trunk_vec)
    x = LeakyReLU(alpha=0.05, name=f'{name}_lrelu1')(x)
    out = Dense(shape, activation='softplus', name=name)(x)
    return out, 'poisson'


def binary_head(trunk_vec, name, shape=1):
    x = Dense(128, name=f'{name}_dense1')(trunk_vec)
    x = LeakyReLU(alpha=0.05, name=f'{name}_lrelu1')(x)
    out = Dense(shape, activation='sigmoid', name=name)(x)
    return out, 'binary_crossentropy'


def regression_head(trunk_vec, name, shape=1):
    x = Dense(128, name=f'{name}_dense1')(trunk_vec)
    x = LeakyReLU(alpha=0.05, name=f'{name}_lrelu1')(x)
    out = Dense(shape, activation='linear', name=name)(x)
    return out, 'mse'


def probe_head(trunk_vec, name):
    x = Dense(128, name=f'{name}_dense1')(trunk_vec)
    x = LeakyReLU(alpha=0.05, name=f'{name}_lrelu1')(x)
    if name == 'hanging_opp_value':
        out = Dense(1, activation='linear', name=name)(x); loss = 'mse'
    else:
        out = Dense(1, activation='sigmoid', name=name)(x); loss = 'binary_crossentropy'
    det = Lambda(lambda t: tf.stop_gradient(t), name=f'sg_{name}')(out)
    return out, loss, det


def weighted_bce_legality(pos_weight=300.0, neg_weight=1.0, eps=1e-7):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)            # [B,8,8,73]
        w = y_true * pos_weight + (1.0 - y_true) * neg_weight
        # elementwise BCE on probabilities (sigmoid head)
        bce = -(y_true * tf.math.log(y_pred + eps) +
                (1.0 - y_true) * tf.math.log(1.0 - y_pred + eps))  # [B,8,8,73]
        return tf.reduce_mean(w * bce)                    # scalar
    return loss


def legal_head(trunk_feat, name="legal_moves"):
    """
    Spatial legality map head: [B,8,8,73] with sigmoid.
    Target: 0/1 mask (same shape).
    """
    out = Conv2D(73, 1, padding="same", activation="sigmoid", name=name)(trunk_feat)
    loss = weighted_bce_legality(pos_weight=300.0, neg_weight=1.0)
    return out, loss


def piece_to_move_head(trunk_vec, leak=0.05, name="piece_to_move"):
    x = Dense(128, name=f"{name}_dense1")(trunk_vec)
    x = LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
    out = Dense(6, activation="softmax", name=name)(x)  # order: P,N,B,R,Q,K
    return out, CategoricalCrossentropy()


def build_full_with_aux(input_shape, aux_configs, width=128, n_blocks=8, leak=0.05):
    stump = build_stump(
        input_shape=input_shape, width=width, n_blocks=n_blocks, leak=leak
    )
    trunk_feat = stump.output                       # [8,8,C]
    trunk_vec  = GlobalAveragePooling2D(name="trunk_gap_for_aux")(trunk_feat)  # [B,C]

    # main heads
    policy_logits, value = attach_policy_value_heads(trunk_feat, leak=leak)

    outputs = {'policy_logits': policy_logits, 'value': value}
    losses  = {
        'policy_logits': masked_mse,
        'value': 'binary_crossentropy'
    }
    
    loss_weights = {'policy_logits': 1.0, 'value': 0.5}

    # aux heads
    for kind, name, shape in aux_configs:
        if kind == 'count':
            out, loss = count_vec_head(trunk_vec, name, shape)
            outputs[name] = out; losses[name] = loss; loss_weights[name] = 0.1
            
        elif kind == 'binary':
            out, loss = binary_head(trunk_vec, name, shape)
            outputs[name] = out; losses[name] = loss; loss_weights[name] = 0.1
            
        elif kind == 'regression':
            out, loss = regression_head(trunk_vec, name, shape)
            outputs[name] = out; losses[name] = loss; loss_weights[name] = 0.1
            
        elif kind == 'probe':
            out, loss, _det = probe_head(trunk_vec, name)
            outputs[name] = out; losses[name] = loss; loss_weights[name] = 0.1
            
        elif kind == 'legal':
            out, loss = legal_head(trunk_feat, name)  # spatial [8,8,73]
            outputs[name] = out; losses[name] = loss; loss_weights[name] = 0.2
            
        elif kind == 'piece_to_move':
            out, loss = piece_to_move_head(trunk_vec, name=name)
            outputs[name] = out; losses[name] = loss; loss_weights[name] = 0.1
        
        else:
            raise ValueError(f"unknown aux kind: {kind}")

    model = Model(stump.input, outputs, name="full_with_aux")
    return model, losses, loss_weights

# 2) choose heads now, but you can add more later
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

    # queens (binary but per-side → shape=2 vector)
    ('binary', 'queen_undefended',           2),
    ('binary', 'queen_hanging',              2),
    ('binary', 'queen_en_prise',             2),
    ('binary', 'queen_attacked_by_lower_value', 2),

    # material (pair)
    ('count',  'material',                   2),
    ('piece_to_move', 'piece_to_move', 6),

    # NEW: legality map aux head (spatial)
    ('legal',  'legal_moves',                None),
]


def new_policy_value_model():
    model, losses, loss_weights = build_full_with_aux(
        input_shape=(8, 8, 23), aux_configs=aux_configs
    )
    
    opt = keras.optimizers.Adam(3e-4)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights)
    return model, loss_weights


def build_value_policy_only(model, value_name="value", policy_name="policy_logits"):
    """
    Returns a new Model that takes the same input as `model`
    but outputs only [value, policy_logits].
    """
    return Model(
        inputs=model.input,
        outputs=[
            model.get_layer(value_name).output,
            model.get_layer(policy_name).output,
        ],
        name="value_policy_only",
    )


# utilities
def set_head_weights(model, new_weights):
    """
    new_weights: dict {head_name: weight}. Unmentioned heads keep current weight.
    Re-compiles the model with updated loss_weights.
    """
    # start from last known weights (fallback: equal weights)
    curr = getattr(model, "_head_loss_weights", None)
    if curr is None:
        # Try to pull from compile config (Keras 3+), else default
        try:
            cfg = model.get_compile_config()
            curr = dict(cfg.get("loss_weights") or {})
        except Exception:
            curr = {o.name.split(':')[0].split("/")[0]: 1.0 for o in model.outputs}

    curr.update(new_weights)

    # Recompile with same optimizer and losses, new weights
    losses = getattr(model, "_head_losses", None)
    if losses is None:
        # Best-effort recovery if not stashed
        losses = {}
        for o in model.outputs:
            name = o.name.split(':')[0].split("/")[0]
            # Guess a loss (won't be perfect, but avoids crash)
            losses[name] = "mse" if o.shape[-1] != 1 else "mse"

    model.compile(optimizer=model.optimizer, loss=losses, loss_weights=curr)
    model._head_loss_weights = dict(curr)


def freeze_heads(model, head_names, trainable=False):
    """
    Freeze/unfreeze all layers belonging to the given heads.
    A layer is considered part of a head if its name starts with the head name
    (e.g., 'value_dense1', 'white_queen_hanging_lrelu1', etc.).
    """
    prefixes = tuple(head_names)
    for layer in model.layers:
        if layer.name.startswith(prefixes):
            layer.trainable = trainable

    model.compile(
        optimizer=model.optimizer,
        loss=getattr(model, "_head_losses", None) or model.loss,
        loss_weights=getattr(model, "_head_loss_weights", None) or None
    )


def build_value_only(model, value_name='value'):
    return Model(model.input, model.get_layer(value_name).output, name='value_only')


# helpers
def _head_name(tensor):
    # e.g. "value/Tanh:0" -> "value"
    return tensor.name.split(':')[0].split('/')[0]


def _get_trunk_tensor(model, trunk_layer_name):
    return model.get_layer(trunk_layer_name).output


def _collect_losses_and_weights(model, out_names, default_loss="mse"):
    old_losses  = getattr(model, "_head_losses", {}) or {}
    old_weights = getattr(model, "_head_loss_weights", {}) or {}

    losses  = {n: old_losses.get(n, default_loss) for n in out_names}
    weights = {n: old_weights.get(n, 1.0)        for n in out_names}
    return losses, weights


def _recompile(model, losses, weights):
    lr = getattr(model.optimizer, 'learning_rate', 1e-3)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=losses, loss_weights=weights)
    model._head_losses = dict(losses)
    model._head_loss_weights = dict(weights)


def add_head(
    model,
    kind,                 # 'value' | 'binary' | 'count' | 'regression' | 'probe'
    name,
    shape=None,           # needed for 'count' and multi-dim heads
    trunk_layer_name='trunk_flat',
    loss_weight=1.0,
    **kwargs              # passthrough to builder (e.g., extra_feats for value_head)
):
    """
    Adds a new head on top of the existing trunk using your builders.
    Only the new head is freshly initialized; all other weights stay.
    """
    trunk = _get_trunk_tensor(model, trunk_layer_name)

    # build the requested head using your existing builders
    if kind == 'binary':
        out, loss_name = binary_head(trunk, name=name, shape=(shape or 1))
    elif kind == 'count':
        assert shape is not None, "add_head(kind='count') requires shape"
        out, loss_name = count_vec_head(trunk, name=name, shape=shape)
    elif kind == 'regression':
        out, loss_name = regression_head(trunk, name=name, shape=(shape or 1))
    elif kind == 'probe':
        out, loss_name, _ = probe_head(trunk, name=name)
    else:
        raise ValueError(f"Unknown head kind: {kind}")

    # create new model with the extra output
    new_outputs = list(model.outputs) + [out]
    new_model = Model(inputs=model.inputs, outputs=new_outputs, name=model.name)

    # rebuild losses/weights and add the new head’s entries
    out_names = [_head_name(o) for o in new_model.outputs]
    losses, weights = _collect_losses_and_weights(model, out_names)
    losses[name] = loss_name
    weights[name] = loss_weight

    _recompile(new_model, losses, weights)
    return new_model


def remove_head(model, name, relink_identity=False):
    """
    Removes a head from the model outputs. All other weights are preserved.
    """
    kept = []
    for t in model.outputs:
        if _head_name(t) == name:
            continue
        kept.append(t)

    if not kept:
        raise ValueError("Cannot remove the last remaining head.")

    # optionally relink via identity (usually unnecessary)
    if relink_identity:
        removed = [t for t in model.outputs if _head_name(t) == name]
        if removed:
            _ = Lambda(lambda z: z, name=f"{name}_removed_identity")(removed[0])

    new_model = Model(inputs=model.inputs, outputs=kept, name=model.name)

    out_names = [_head_name(o) for o in new_model.outputs]
    losses, weights = _collect_losses_and_weights(model, out_names)

    # ensure removed head is not in dicts
    losses.pop(name, None)
    weights.pop(name, None)

    _recompile(new_model, losses, weights)
    return new_model


def rename_head(model, old_name, new_name):
    """
    Renames an output head **without** rebuilding it (preserves its weights).
    Implemented by wrapping the old output with an identity Lambda that has `new_name`.
    """
    new_outputs = []
    replaced = False
    for t in model.outputs:
        if _head_name(t) == old_name:
            # identity layer with the new name keeps the same tensor values/weights
            new_t = Lambda(lambda z: z, name=new_name)(t)
            new_outputs.append(new_t)
            replaced = True
        else:
            new_outputs.append(t)

    if not replaced:
        raise ValueError(f"Head '{old_name}' not found.")

    new_model = Model(inputs=model.inputs, outputs=new_outputs, name=model.name)

    # transfer losses/weights, swapping the key
    out_names = [_head_name(o) for o in new_model.outputs]
    old_losses  = getattr(model, "_head_losses", {}) or {}
    old_weights = getattr(model, "_head_loss_weights", {}) or {}

    losses  = {}
    weights = {}
    for n in out_names:
        src = new_name if n == new_name else n
        if src == new_name and old_name in old_losses:
            # bring over the old head's settings under the new name
            losses[n]  = old_losses.get(old_name, "mse")
            weights[n] = old_weights.get(old_name, 1.0)
        else:
            losses[n]  = old_losses.get(n, "mse")
            weights[n] = old_weights.get(n, 1.0)

    _recompile(new_model, losses, weights)
    return new_model


def _to_float32(x):
    a = np.asarray(x)
    return a.astype(np.float32, copy=False)

def prepare_targets(all_Y):
    # which targets you expect (must match your model output names)
    POLICY_KEY = "policy_logits"   # your policy target name
    VALUE_KEY  = "value"           # scalar [1]
    LEGAL_KEY  = "legal_moves"     # [8,8,73] 0/1
    
    # vector heads that are length-2 (mine, theirs)
    VEC2_HEADS = [
        "king_ray_exposure","king_ring_pressure","king_pawn_shield","king_escape_square",
        "pawn_undefended","pawn_hanging","pawn_en_prise",
        "knight_undefended","knight_hanging","knight_en_prise",
        "knight_attacked_by_lower_value", "bishop_undefended","bishop_hanging",
        "bishop_en_prise","bishop_attacked_by_lower_value", "rook_undefended",
        "rook_hanging","rook_en_prise","rook_attacked_by_lower_value", "queen_undefended",
        "queen_hanging","queen_en_prise","queen_attacked_by_lower_value", "material"
    ]
    
    # other heads by shape
    PIECE_TO_MOVE_KEY = "piece_to_move"   # [6]    
    
    N = len(all_Y)
    # discover which keys are present (require same keys for all samples)
    keys = set(all_Y[0].keys())
    for d in all_Y[1:]:
        keys &= set(d.keys())
    # minimal required
    assert POLICY_KEY in keys and VALUE_KEY in keys, "Missing some policy/value"

    # --- stack core heads ---
    Yp = np.stack([_to_float32(d[POLICY_KEY]) for d in all_Y], axis=0)  # [N,8,8,73]
    Yv = np.stack([_to_float32(d[VALUE_KEY])  for d in all_Y], axis=0)  # [N,1] or [N]
    Yv = Yv.reshape(N, 1).astype(np.float32)

    targets = {POLICY_KEY: Yp, VALUE_KEY: Yv}

    # --- legality map if present ---
    if LEGAL_KEY in keys:
        Ym = np.stack([_to_float32(d[LEGAL_KEY]) for d in all_Y], axis=0)  # [N,8,8,73]
        targets[LEGAL_KEY] = Ym

    # --- piece_to_move if present ---
    if PIECE_TO_MOVE_KEY in keys:
        Ypiece = np.stack([_to_float32(d[PIECE_TO_MOVE_KEY]) for d in all_Y], axis=0)
        targets[PIECE_TO_MOVE_KEY] = Ypiece

    # --- all the 2-dim aux heads (convert list/tuple -> [2] float32) ---
    for name in VEC2_HEADS:
        if name in keys:
            Y = np.stack([_to_float32(all_Y[i][name]) for i in range(N)], axis=0)
            targets[name] = Y

    return targets


# PIECE_TO_ID = {
#     None: 0,                # empty square
#     chess.PAWN: 1,
#     chess.KNIGHT: 2,
#     chess.BISHOP: 3,
#     chess.ROOK: 4,
#     chess.QUEEN: 5,
#     chess.KING: 6,
# }

# # black pieces offset by +6
# def square_to_id(piece):
#     if piece is None:
#         return 0
#     base = PIECE_TO_ID[piece.piece_type]
#     return base if piece.color == chess.WHITE else base + 6


# def board_to_tokens(board: chess.Board):
#     tokens = []
#     for sq in chess.SQUARES:
#         piece = board.piece_at(sq)
#         tokens.append(square_to_id(piece))
#     # extra tokens
#     tokens.append(int(board.turn))
#     tokens.append(int(board.has_kingside_castling_rights(chess.WHITE)))
#     tokens.append(int(board.has_queenside_castling_rights(chess.WHITE)))
#     tokens.append(int(board.has_kingside_castling_rights(chess.BLACK)))
#     tokens.append(int(board.has_queenside_castling_rights(chess.BLACK)))
#     ep = board.ep_square if board.ep_square else 0
#     tokens.append(ep % 64)  # encode en passant square (0 if none)
#     return np.array(tokens, dtype=np.int32)

# # -------------------------
# # Core Transformer Stump
# # -------------------------

# class TransformerBlock(layers.Layer):
#     def __init__(self, d_model, num_heads, ff_dim, rate=0.1, name=None, **kwargs):
#         super().__init__(name=name)
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.ff_dim = ff_dim
#         self.rate = rate

#         self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
#         self.ffn = tf.keras.Sequential([
#             layers.Dense(ff_dim, activation="gelu"),
#             layers.Dense(d_model),
#         ])
#         self.norm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.norm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.drop1 = layers.Dropout(rate)
#         self.drop2 = layers.Dropout(rate)

#     def call(self, x, training=False):
#         attn_output, attn_scores = self.att(x, x, return_attention_scores=True)
#         x = self.norm1(x + self.drop1(attn_output, training=training))
#         ffn_output = self.ffn(x)
#         x = self.norm2(x + self.drop2(ffn_output, training=training))
#         return x, attn_scores

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "d_model": self.d_model,
#             "num_heads": self.num_heads,
#             "ff_dim": self.ff_dim,
#             "rate": self.rate,
#             "name": self.name,
#         })
#         return config


# def make_transformer_stump(vocab_size=32, seq_len=70, d_model=64, num_heads=4, num_layers=2, ff_dim=128, drop_rate=0.1):
#     tokens_in = layers.Input(shape=(seq_len,), dtype="int32", name="tokens")

#     tok_emb = layers.Embedding(vocab_size, d_model, name="tok_emb")(tokens_in)
#     pos_emb = layers.Embedding(seq_len, d_model, name="pos_emb")(tf.range(seq_len))
#     x = tok_emb + pos_emb

#     attn_maps = []
#     for i in range(num_layers):
#         x, scores = TransformerBlock(d_model, num_heads, ff_dim, rate=drop_rate, name=f"block_{i}")(x)
#         attn_maps.append(scores)

#     # sequence features (per-token), and pooled vector stump
#     trunk_seq = x   # no Identity
#     trunk_vec = layers.GlobalAveragePooling1D(name="trunk_vec")(x)
#     stump = Model(tokens_in, [trunk_seq, trunk_vec], name="tf_stump")

#     stump.attn_maps = attn_maps
#     return stump

# # -------------------------
# # Heads (policy/value/aux)
# # -------------------------

# def value_head(trunk_vec, hidden=128, leak=0.05, name="value"):
#     x = trunk_vec
#     if hidden:
#         x = layers.Dense(hidden, name=f"{name}_dense1")(x)
#         x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
#     out = layers.Dense(1, activation="tanh", name=name)(x)  # [-1,1] White POV
#     return out, "mse"


# def policy_head(trunk_vec, n_moves, hidden=128, leak=0.05, name="policy", use_mask=True):
#     x = trunk_vec
#     if hidden:
#         x = layers.Dense(hidden, name=f"{name}_dense1")(x)
#         x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
#     logits = layers.Dense(n_moves, name=f"{name}_logits")(x)
#     if use_mask:
#         mask_in = layers.Input(shape=(n_moves,), dtype="float32", name=f"{name}_mask")
#         masked_logits = layers.Add(name=f"{name}_masked_logits")(
#             [logits, (1.0 - mask_in) * (-1e9)]
#         )
        
#         probs = layers.Activation("softmax", name=name)(masked_logits)
#         return probs, "categorical_crossentropy", mask_in
#     else:
#         probs = layers.Activation("softmax", name=name)(logits)
#         return probs, "categorical_crossentropy", None


# def count_vec_head(trunk_vec, name, shape, leak=0.05):
#     x = layers.Dense(128, name=f"{name}_dense1")(trunk_vec)
#     x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
#     out = layers.Dense(shape, activation="softplus", name=name)(x)
#     return out, "poisson"


# def binary_head(trunk_vec, name, shape=1, leak=0.05):
#     x = layers.Dense(128, name=f"{name}_dense1")(trunk_vec)
#     x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
#     out = layers.Dense(shape, activation="sigmoid", name=name)(x)
#     return out, "binary_crossentropy"


# def regression_head(trunk_vec, name, shape=1, leak=0.05):
#     x = layers.Dense(128, name=f"{name}_dense1")(trunk_vec)
#     x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
#     out = layers.Dense(shape, activation="linear", name=name)(x)
#     return out, "mse"


# def probe_head(trunk_vec, name, leak=0.05):
#     x = layers.Dense(128, name=f"{name}_dense1")(trunk_vec)
#     x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
#     # mirror your ResNet behavior
#     if name == "hanging_opp_value":
#         out = layers.Dense(1, activation="linear", name=name)(x); loss = "mse"
#     else:
#         out = layers.Dense(1, activation="sigmoid", name=name)(x)
#         loss = "binary_crossentropy"
#     det = layers.Lambda(lambda t: tf.stop_gradient(t), name=f"sg_{name}")(out)
#     return out, loss, det


# def piece_to_move_head(trunk_vec, leak=0.05, name="piece_to_move"):
#     x = layers.Dense(128, name=f"{name}_dense1")(trunk_vec)
#     x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)
#     out = layers.Dense(6, activation="softmax", name=name)(x)  # P,N,B,R,Q,K
#     return out, tf.keras.losses.CategoricalCrossentropy()

# # -------------------------
# # Builder with aux configs
# # -------------------------

# def build_transformer_full_with_aux(
#     vocab_size=32, seq_len=70, d_model=128, num_heads=4, num_layers=4, ff_dim=128,
#     n_moves=4672, use_mask=True, aux_configs=None, aux_into_heads=False,
#     aux_weight=0.1, value_weight=1.0, policy_weight=1.0, drop_rate=0.1
# ):
    
#     if aux_configs is None:
#         aux_configs = []

#     stump = make_transformer_stump(
#         vocab_size=vocab_size, seq_len=seq_len, d_model=d_model, num_heads=num_heads, 
#         num_layers=num_layers, ff_dim=ff_dim, drop_rate=drop_rate
#     )

#     tokens_in = stump.input
#     trunk_seq, trunk_vec = stump.output  # [B,seq,C], [B,C]

#     # Collect probe dets for optional stop-grad concat
#     probe_dets = []

#     outputs = {}
#     losses = {}
#     loss_weights = {}

#     # Main value & policy heads (heads see stump_vec; optionally augmented with probe_dets)
#     head_base = trunk_vec

#     # First pass: build aux heads; collect dets for concat if requested
#     for kind, name, shape in aux_configs:
#         if kind == "count":
#             out, loss = count_vec_head(trunk_vec, name, shape)
#             outputs[name] = out; losses[name] = loss; loss_weights[name] = aux_weight

#         elif kind == "binary":
#             out, loss = binary_head(trunk_vec, name, shape)
#             outputs[name] = out; losses[name] = loss; loss_weights[name] = aux_weight

#         elif kind == "regression":
#             out, loss = regression_head(trunk_vec, name, shape)
#             outputs[name] = out; losses[name] = loss; loss_weights[name] = aux_weight

#         elif kind == "probe":
#             out, loss, det = probe_head(trunk_vec, name)
#             outputs[name] = out; losses[name] = loss; loss_weights[name] = aux_weight
#             probe_dets.append(det)

#         elif kind == "piece_to_move":
#             out, loss = piece_to_move_head(trunk_vec, name=name)
#             outputs[name] = out; losses[name] = loss; loss_weights[name] = aux_weight

#         else:
#             raise ValueError("unknown aux kind: " + str(kind))

#     if aux_into_heads and probe_dets:
#         head_base = layers.Concatenate(name="head_feat_concat")(
#             [trunk_vec] + [layers.Lambda(lambda t: t)(d) for d in probe_dets]
#         )

#     # Value head
#     value_out, v_loss = value_head(head_base, hidden=128, name="value")
#     outputs["value"] = value_out; losses["value"] = v_loss
#     loss_weights["value"] = value_weight

#     # Policy head (masked softmax)
#     policy_out, p_loss, mask_in = policy_head(
#         head_base, n_moves=n_moves, hidden=128, name="policy", use_mask=use_mask
#     )
    
#     outputs["policy"] = policy_out; losses["policy"] = p_loss
#     loss_weights["policy"] = policy_weight

#     inputs = [tokens_in] + ([mask_in] if mask_in is not None else [])
#     model = Model(inputs=inputs, outputs=outputs, name="transformer_with_aux")
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(3e-4), loss=losses, loss_weights=loss_weights
#     )
    
#     model._head_losses = dict(losses)
#     model._head_loss_weights = dict(loss_weights)
#     model.attn_maps = stump.attn_maps
#     return model


# # 1) choose your aux heads
# aux_configs = [
#     # king exposure bundle (2 = [mine, theirs])
#     ('count',  'king_ray_exposure',          2),
#     ('count',  'king_ring_pressure',         2),
#     ('count',  'king_pawn_shield',           2),
#     ('count',  'king_escape_square',         2),

#     # pawns
#     ('count',  'pawn_undefended',            2),
#     ('count',  'pawn_hanging',               2),
#     ('count',  'pawn_en_prise',              2),

#     # knights
#     ('count',  'knight_undefended',          2),
#     ('count',  'knight_hanging',             2),
#     ('count',  'knight_en_prise',            2),
#     ('count',  'knight_attacked_by_lower_value', 2),

#     # bishops
#     ('count',  'bishop_undefended',          2),
#     ('count',  'bishop_hanging',             2),
#     ('count',  'bishop_en_prise',            2),
#     ('count',  'bishop_attacked_by_lower_value', 2),

#     # rooks
#     ('count',  'rook_undefended',            2),
#     ('count',  'rook_hanging',               2),
#     ('count',  'rook_en_prise',              2),
#     ('count',  'rook_attacked_by_lower_value', 2),

#     # queens (binary but per-side , shape=2 vector)
#     ('binary', 'queen_undefended',           2),
#     ('binary', 'queen_hanging',              2),
#     ('binary', 'queen_en_prise',             2),
#     ('binary', 'queen_attacked_by_lower_value', 2),

#     # material (pair)
#     ('count',  'material',                   2),
#     ('piece_to_move', 'piece_to_move', 6)
# ]


# # -------------------------
# # Utilities: adjust head weights on the fly
# # -------------------------

# def set_head_weights(model, new_weights):
#     curr = getattr(model, "_head_loss_weights", None)
#     if curr is None:
#         try:
#             cfg = model.get_compile_config()
#             curr = dict(cfg.get("loss_weights") or {})
#         except Exception:
#             curr = {o.name.split(':')[0].split("/")[0]: 1.0 for o in model.outputs}
#     curr.update(new_weights)

#     losses = getattr(model, "_head_losses", None)
#     if losses is None:
#         losses = {}
#         for o in model.outputs:
#             name = o.name.split(':')[0].split("/")[0]
#             losses[name] = "mse" if o.shape[-1] != 1 else "mse"

#     model.compile(optimizer=model.optimizer, loss=losses, loss_weights=curr)
#     model._head_loss_weights = dict(curr)
    
    
# new_weights = {
#     'king_ray_exposure': 0.05,
#     'king_ring_pressure': 0.1,
#     'king_pawn_shield': 0.1,
#     'king_escape_square': 0.1,
#     'pawn_undefended': 0.1,
#     'pawn_hanging': 0.1,
#     'pawn_en_prise': 0.1,
#     'knight_undefended': 0.1,
#     'knight_hanging': 0.1,
#     'knight_en_prise': 0.1,
#     'knight_attacked_by_lower_value': 0.1,
#     'bishop_undefended': 0.1,
#     'bishop_hanging': 0.1,
#     'bishop_en_prise': 0.1,
#     'bishop_attacked_by_lower_value': 0.1,
#     'rook_undefended': 0.1,
#     'rook_hanging': 0.1,
#     'rook_en_prise': 0.1,
#     'rook_attacked_by_lower_value': 0.1,
#     'queen_undefended': 0.2,
#     'queen_hanging': 0.2,
#     'queen_en_prise': 0.2,
#     'queen_attacked_by_lower_value': 0.2,
#     'material': 0.025,
#     'piece_to_move': 0.05,
#     'value': 3.0,
#     'policy': 5.0
#  }

# ignore_aux = {k:0.0 for k in new_weights}
# ignore_aux['value'] = 2.0
# ignore_aux['policy'] = 2.0
