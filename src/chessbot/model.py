import os
import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import json
from pathlib import Path

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"
HE = "he_normal"
BIG_NEG = -1e9
EPS = 1e-9

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import unpack_x_y_sample_weight
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, Add,
    GlobalAveragePooling2D, Dense, Lambda
)


def load_model(model_loc):
    model = keras.models.load_model(model_loc)
    return model


def save_model(model, model_loc):
    model.save(model_loc, save_format='h5')


def res_block(x, filters, leak=0.05):
    skip = x

    # project skip if channel count doesnt match
    if x.shape[-1] != filters:
        skip = Conv2D(
            filters, 1, padding="same", use_bias=False, kernel_initializer=HE
        )(skip)
        skip = BatchNormalization()(skip)

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


def build_full_with_aux(input_shape, aux_configs, width=128, n_blocks=8, leak=0.05):
    stump = build_stump(
        input_shape=input_shape, width=width, n_blocks=n_blocks, leak=leak
    )
    trunk_feat = stump.output                       # [8,8,C]
    trunk_vec  = GlobalAveragePooling2D(name="trunk_gap_for_aux")(trunk_feat)  # [B,C]

    # main heads
    policy_logits, value = attach_policy_value_heads(trunk_feat, leak=leak)

    outputs = {'policy_logits': policy_logits, 'value': value}
    losses  = {'value': 'binary_crossentropy'}
    
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


#### Current Model ###
def value_head(trunk_vec, hidden=512, leak=0.05, name="value"):
    x = trunk_vec
    if hidden:
        # First hidden layer
        x = layers.Dense(hidden, name=f"{name}_dense1")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)

        # Second hidden layer at half size
        x = layers.Dense(hidden // 2, name=f"{name}_dense2")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu2")(x)

    out = layers.Dense(1, activation="tanh", name=name)(x)  # [-1,1] White POV
    return out, "mse"


def policy_factor_head(trunk_vec, prefix, hidden=512, leak=0.05):
    """Factorized move logits (no capture)."""
    x = trunk_vec
    if hidden:
        # First hidden layer
        x = layers.Dense(hidden, name=f"{prefix}_dense1")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{prefix}_lrelu1")(x)

        # Second hidden layer at half size
        x = layers.Dense(hidden // 2, name=f"{prefix}_dense2")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{prefix}_lrelu2")(x)

    # Raw logits, no activation
    from_logits   = layers.Dense(64, name=f"{prefix}_from")(x)
    to_logits     = layers.Dense(64, name=f"{prefix}_to")(x)
    piece_logits  = layers.Dense(6,  name=f"{prefix}_piece")(x)
    promo_logits  = layers.Dense(4,  name=f"{prefix}_promo")(x)

    return [from_logits, to_logits, piece_logits, promo_logits]


def build_conv_trunk(input_shape=(8, 8, 70), width=256, n_blocks=8, leak=0.05):
    """Convolutional trunk with residual blocks, returns feature map + pooled vector."""
    inputs = layers.Input(shape=input_shape, name="board")

    # fat first conv
    x = layers.Conv2D(
        2*width, 3, padding="same", use_bias=False, kernel_initializer=HE
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=leak)(x)
    
    # residual tower
    for _ in range(n_blocks):
        x = res_block(x, width, leak=leak)
    
    trunk_feat = layers.LeakyReLU(alpha=leak, name="trunk")(x)

    # global pooling: (8,8,width) -> (width,)
    trunk_vec = layers.GlobalAveragePooling2D(name="trunk_vec")(trunk_feat)

    return Model(inputs, [trunk_feat, trunk_vec], name="conv_trunk")


def make_conv_model(name, input_shape=(8, 8, 70), width=256, n_blocks=8):
    trunk = build_conv_trunk(input_shape=input_shape, width=width, n_blocks=n_blocks)
    inputs = trunk.input
    trunk_feat, trunk_vec = trunk.output

    # Heads
    val_out, _ = value_head(trunk_vec, hidden=512)
    best_outputs = policy_factor_head(trunk_vec, prefix="best", hidden=512)

    model = Model(inputs, [val_out] + best_outputs, name=name)

    losses = {
        "value": tf.keras.losses.MeanSquaredError(),
        "best_from": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "best_to":   tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "best_piece":tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "best_promo":tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    }

    loss_weights = {
        "value": 0.5,
        "best_from": 0.5,
        "best_to": 0.5,
        "best_piece": 0.5,
        "best_promo": 0.1,
    }

    opt = tf.keras.optimizers.Adam(1e-4)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights)
    return model


def make_fwd(model, warm_shapes=(64, 256)):
    """
    Returns a callable fwd(X_np) that:
      1) uses a compiled @tf.function on the given model with training=False
      2) accepts a NumPy batch [B,8,8,70] and returns a list of NumPy arrays
         in the same order as model.outputs
      3) warms up the common batch shapes once to stabilize autotune
    """
    @tf.function(input_signature=[tf.TensorSpec([None, 8, 8, 70], tf.float32)])
    def graph(x):
        return model(x, training=False)

    for B in warm_shapes:
        _ = graph(tf.zeros([B, 8, 8, 70], tf.float32))

    def fwd(x_np):
        x = tf.convert_to_tensor(x_np, dtype=tf.float32)
        outs = graph(x)
        if isinstance(outs, (list, tuple)):
            return [t.numpy() for t in outs]
        if isinstance(outs, dict):
            ordered = []
            for t in model.outputs:
                key = t.name.split(':')[0].split('/')[0]
                ordered.append(outs[key].numpy())
            return ordered
        return [outs.numpy()]
    return fwd
    

def make_fwd_batched(model, max_bs=1024, warm_shapes=(256, 512, 1024)):
    base = make_fwd(model, warm_shapes=warm_shapes)

    def fwd(X_np):
        B = X_np.shape[0]
        if B <= max_bs:
            return base(X_np)

        acc = None
        i = 0
        while i < B:
            j = min(i + max_bs, B)
            part = base(X_np[i:j])
            if acc is None:
                acc = [p for p in part]
            else:
                for k in range(len(acc)):
                    acc[k] = np.concatenate([acc[k], part[k]], axis=0)
            i = j
        return acc

    return fwd


def build_transformer_64pv(
    d_model=256, n_heads=8, n_layers=5, ff_dim=None, proj_dim=48,
    dropout=0.05, vocab_size=21, name="transformer_64pv"
):
    if ff_dim is None:
        ff_dim = d_model * 2

    enc_in = Input(shape=(64,), dtype="int32", name="enc_in")
    legal_in = Input(shape=(4096,), dtype="int32", name="legal_mask")

    token_emb = layers.Embedding(
        input_dim=vocab_size, output_dim=d_model, name="token_emb"
    )(enc_in)

    pos_indices = tf.range(start=0, limit=64, delta=1)
    
    pos_emb = layers.Embedding(
        input_dim=64, output_dim=d_model, name="pos_emb")(pos_indices)
    
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    x = token_emb + pos_emb

    for i in range(n_layers):
        # pre-LN transformer block
        ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x)
        att = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            name=f"mha_{i}")(ln1, ln1, ln1)
        
        att = layers.Dropout(dropout)(att)
        x = layers.Add()([x, att])
        
        ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x)
        ff = layers.Dense(ff_dim, activation="gelu", name=f"ff1_{i}")(ln2)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(d_model, name=f"ff2_{i}")(ff)
        ff = layers.Dropout(dropout)(ff)
        x = layers.Add()([x, ff])

    square_feats = x  # (B,64,d)

    def legal_summaries(mask_tensor, **kwargs):
        # mask_tensor may be int32; cast to float32 first
        m = tf.reshape(tf.cast(mask_tensor, tf.float32), (-1, 64, 64))
        from_count = tf.reduce_sum(m, axis=2)
        from_has = tf.cast(from_count > 0.0, tf.float32)
        from_count_norm = tf.math.log1p(from_count) / tf.math.log(64.0)
        return tf.expand_dims(from_has, axis=2), tf.expand_dims(
            from_count_norm, axis=2
        )


    from_has, from_count_norm = layers.Lambda(
        legal_summaries, name="from_legal_feats")(legal_in)

    # append legal features to each square token
    from_has_tiled = tf.cast(from_has, square_feats.dtype)
    from_count_tiled = tf.cast(from_count_norm, square_feats.dtype)
    square_aug = layers.Concatenate(axis=2, name="square_aug")(
        [square_feats, from_has_tiled, from_count_tiled]
    )

    from_proj = layers.Dense(proj_dim, name="from_proj")(square_aug)
    to_proj = layers.Dense(proj_dim, name="to_proj")(square_aug)

    pair_scores = layers.Lambda(
        lambda t: tf.einsum('bif,bjf->bij', t[0], t[1]),
        name="pair_scores")([from_proj, to_proj]
    )

    # per-from and per-to biases
    b_from = layers.Dense(1, name="b_from")(from_proj)
    b_to = layers.Dense(1, name="b_to")(to_proj)
    b_from = tf.squeeze(b_from, axis=2)[:, :, None]
    b_to = tf.squeeze(b_to, axis=2)[:, None, :]

    pair_scores = pair_scores + b_from + b_to

    policy_logits_raw = layers.Reshape((4096,), name="policy_logits_raw")(pair_scores)

    # hard mask inside model (do masking in float32 to avoid fp16 issues)
    def apply_mask(args):
        logits, mask = args
        logits32 = tf.cast(logits, tf.float32)
        mask32 = tf.cast(mask, tf.float32)
        big_neg = tf.constant(-1e6, dtype=tf.float32)
        masked = tf.where(mask32 > 0.5, logits32, big_neg)
        return tf.cast(masked, logits.dtype)

    policy_logits = layers.Lambda(apply_mask, name="policy_logits")(
        [policy_logits_raw, legal_in]
    )

    # value head
    v_pool = layers.GlobalAveragePooling1D(name="v_gap")(x)
    v = layers.Dense(d_model // 2, activation="relu", name="v_fc1")(v_pool)
    v = layers.Dense(d_model // 4, activation="relu", name="v_fc2")(v)
    value_out = layers.Dense(1, activation="tanh", name="value_out")(v)

    model = Model(
        inputs=[enc_in, legal_in],
        outputs=[policy_logits, value_out], name=name
    )
    
    # attach compact init params and loss_weights placeholder
    model.init_params = dict(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        proj_dim=proj_dim,
        dropout=dropout,
        vocab_size=vocab_size,
        name=name,
    )
    
    model.loss_weights = {"policy_logits": 1.0, "value_out": 2.0}
    model._default_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model._default_loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out": "mse"}
    
    model.compile(
        optimizer=model._default_opt,
        loss=model._default_loss_dict,
        loss_weights=model.loss_weights
    )
    
    return model


def save_transformer_model(model, path):
    path = Path(path)
    weights_path = path
    meta_path = path.with_suffix(".json")

    # save weights and metadata
    model.save_weights(str(weights_path), save_format="h5")

    payload = {
        "params": model.init_params,
        "weights_path": str(weights_path),
        "loss_weights": model.loss_weights
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_transformer_model(path):
    meta_path = Path(path)
    if meta_path.suffix != ".json":
        meta_path = meta_path.with_suffix(".json")

    with meta_path.open("r", encoding="utf8") as fh:
        payload = json.load(fh)

    params = payload["params"]
    weights_path = payload["weights_path"]
    loss_weights = payload.get("loss_weights")

    model = build_transformer_64pv(**params)
    model.load_weights(weights_path)
    model.init_params = params
    
    if loss_weights is not None:
        model = set_loss_weights(model, loss_weights)
        
    return model


def set_loss_weights(model, loss_weights):
    # Get a fresh optimizer (same config) to avoid double-wrapping
    opt_candidate = model._default_opt
    if isinstance(opt_candidate, tf.keras.mixed_precision.LossScaleOptimizer):
        base_opt = opt_candidate.inner_optimizer
    else:
        base_opt = opt_candidate

    # create a new optimizer instance with same config
    opt_cfg = tf.keras.optimizers.serialize(base_opt)
    opt = tf.keras.optimizers.deserialize(opt_cfg)

    model.compile(
        optimizer=opt,
        loss=model._default_loss_dict,
        loss_weights=loss_weights
    )
    model.loss_weights = loss_weights
    # remember the fresh optimizer for later use
    model._default_opt = opt
    return model


def build_conv_64pv(
    d_model_embed=128,
    vocab_size=21,
    filters=288,
    n_conv=16,
    proj_dim=96,
    name="conv_64pv"
):
    
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    
    DE, FL, VS = d_model_embed, filters, vocab_size

    enc_in = Input(shape=(64,), dtype="int32", name="enc_in")
    tok_emb = layers.Embedding(input_dim=VS, output_dim=DE, name="token_emb")(enc_in)
    x = layers.Reshape((8, 8, DE), name="to_2d")(tok_emb)

    x = layers.Conv2D(FL, 3, padding="same", activation="relu", name="conv_init")(x)
    
    # residual blocks
    for i in range(n_conv):
        y = layers.Conv2D(FL, 3, padding="same", activation=None, name=f"conv_{i}")(x)
        # light regularization
        if i % 2 == 1:
            y = layers.LayerNormalization(epsilon=1e-5, center=False, scale=False)(y)
            
        y = layers.LeakyReLU(alpha=0.01, name=f"lrelu_{i}")(y)
        x = layers.Add(name=f"res_{i}")([x, y])
    
    # expand to (B, 64, 512) then (B, 64, FL)
    x = layers.Conv2D(512, 1, padding="same", activation="relu", name="conv_expand")(x)
    x = layers.Conv2D(512, 3, padding="same", activation="relu", name="conv_mix")(x)
    
    x = layers.LayerNormalization(
        epsilon=1e-5, center=False, scale=False,
        dtype="float32", name="ln_x512"
    )(x)
    
    x = layers.Conv2D(FL, 1, padding="same", activation="relu", name="conv_project")(x)

    # feat for pairwise head: flatten 8x8 -> 64
    feat = layers.Reshape((64, FL), name="to_64_x")(x)  # (B,64,FL)

    # from/to projections and outer-product -> (B,64,64)
    from_proj = layers.Dense(proj_dim, name="from_proj")(feat)  # (B,64,proj)
    to_proj = layers.Dense(proj_dim, name="to_proj")(feat)      # (B,64,proj)
    pair_scores = tf.matmul(from_proj, to_proj, transpose_b=True)  # (B,64,64)

    # biases and base flatten
    b_from = layers.Dense(1, name="b_from")(from_proj)            # (B,64,1)
    b_from = layers.Reshape((64, 1), name="b_from_reshape")(b_from)
    b_to = layers.Dense(1, name="b_to")(to_proj)                 # (B,64,1)
    b_to = layers.Permute((2, 1), name="b_to_permute")(b_to)     # (B,1,64)
    pair_scores = layers.Add(name="add_biases")([pair_scores, b_from, b_to])
    
    # (B,4096)
    base_flat = layers.Reshape((64 * 64,), name="policy_base_flat")(pair_scores)

    # underpromo conv path: operate on 8x8 feature map x (B,8,8,FL)
    # produce 3 channels per board square (B,8,8,3)
    up_conv1 = layers.Conv2D(
        FL // 2, 2, padding="same", activation="relu", name="under_promo_conv1")(x)
    
    # (B,8,8,3)
    up_conv2 = layers.Conv2D(
        3, 1, padding="same", activation=None, name="under_promo_conv2")(up_conv1)

    # reshape to (B,64,3) then flatten to (B,192)
    up_flat_64_3 = layers.Reshape((64, 3), name="promo_64_3")(up_conv2)  # (B,64,3)
    up_flat = layers.Reshape((64 * 3,), name="promo_flat_raw")(up_flat_64_3)  # (B,192)

    # add a small learned bias/offset per promo slot: create a transform and add it
    # using Dense with zero kernel init so it's effectively a bias addition at start
    promo_bias = layers.Dense(64 * 3, use_bias=True, name="promo_bias_dense")(up_flat)
    promo_logits = layers.Add(name="promo_with_bias")([up_flat, promo_bias])  # (B,192)

    # concat base 4096 + promo 192 -> final logits (B,4288)
    policy_logits = layers.Concatenate(
        name="policy_logits")([base_flat, promo_logits])

    # value head
    v_pool = layers.GlobalAveragePooling1D(name="v_gap")(feat)
    v = layers.Dense(FL // 2, activation="relu", name="v_fc1")(v_pool)
    v = layers.Dense(FL // 4, activation="relu", name="v_fc2")(v)
    value_out = layers.Dense(1, activation="tanh", name="value_out")(v)

    model = Model(inputs=[enc_in], outputs=[policy_logits, value_out], name=name)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    opt = mixed_precision.LossScaleOptimizer(opt)
    
    loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out": "mse",
    }
    loss_weights = {"policy_logits": 1.0, "value_out": 1.0}
    model.compile(optimizer=opt, loss=loss_dict, loss_weights=loss_weights)

    return model, opt, loss_weights, loss_dict


def make_conv_infer(model, max_bs=1024, min_p=0.001, max_p=0.35, temp=1.0):
    """
    Returns fwd((enc_np, legal_np)) -> (probs_np, val_np).
    enc_np: int32 [B,64], legal_np: int32 [B,4288].
    min_p, max_p, temp are baked into the closure.
    """
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    BIG_NEG = tf.constant(-1e9, dtype=tf.float32)
    EPS = tf.constant(1e-12, dtype=tf.float32)

    # baked float32 constants for the graph
    min_p_c = tf.constant(float(min_p), dtype=tf.float32)
    max_p_c = tf.constant(float(max_p), dtype=tf.float32)
    temp_c = tf.constant(float(temp), dtype=tf.float32)

    @tf.function(input_signature=[
        tf.TensorSpec([None, 64], tf.int32),
        tf.TensorSpec([None, 4288], tf.int32),
    ], experimental_compile=True)
    def graph(enc, legal):
        # model returns (policy_logits, value)
        logits, value = model(enc, training=False)
        logits = tf.reshape(logits, [tf.shape(logits)[0], -1])  # (B,4288)

        # mask as float32
        mask = tf.cast(tf.reshape(legal, [tf.shape(logits)[0], -1]), tf.float32)

        # cast logits -> float32 for stable softmax under mixed precision
        logits32 = tf.cast(logits, tf.float32)

        # mask illegal moves with a large negative in float32
        masked_logits32 = tf.where(mask > 0.5, logits32, BIG_NEG)

        # temperature-stable softmax in float32
        scaled = masked_logits32 / temp_c
        row_max = tf.reduce_max(scaled, axis=1, keepdims=True)
        exp = tf.exp(scaled - row_max) * mask
        sumexp = tf.reduce_sum(exp, axis=1, keepdims=True)
        has_any = sumexp > 0.0
        probs = tf.where(has_any, exp / (sumexp + EPS), tf.zeros_like(exp))

        # clipping on legal slots and renormalize (float32)
        clipped = tf.where(mask > 0.5,
                           tf.clip_by_value(probs, min_p_c, max_p_c),
                           tf.zeros_like(probs))
        s = tf.reduce_sum(clipped, axis=1, keepdims=True)
        valid = s > EPS
        probs_final = tf.where(
            valid, clipped / (s + (1.0 - tf.cast(valid, tf.float32))),
            tf.zeros_like(clipped)
        )

        # cast value to float32
        value_f = tf.cast(value, tf.float32)
        return probs_final, value_f

    def base_fwd(pair):
        if not isinstance(pair, (list, tuple)):
            raise ValueError("pass (enc_np, legal_np) tuple")
        enc_np, legal_np = pair
        e_tf = tf.convert_to_tensor(enc_np, dtype=tf.int32)
        l_tf = tf.convert_to_tensor(legal_np, dtype=tf.int32)
        probs_tf, val_tf = graph(e_tf, l_tf)
        return probs_tf.numpy(), val_tf.numpy()

    if max_bs is None:
        return base_fwd

    def fwd(pair):
        enc_np, legal_np = pair
        B = int(enc_np.shape[0])
        if B <= max_bs:
            return base_fwd((enc_np, legal_np))
        parts = None
        i = 0
        while i < B:
            j = min(i + max_bs, B)
            p_probs, p_val = base_fwd((enc_np[i:j], legal_np[i:j]))
            if parts is None:
                parts = [p_probs, p_val]
            else:
                parts[0] = np.concatenate([parts[0], p_probs], axis=0)
                parts[1] = np.concatenate([parts[1], p_val],  axis=0)
            i = j
        return parts
    return fwd
