import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
plt.ion()

import chess
import chess.engine

import math, random
import os, pickle

#%%
from keras import regularizers
from keras import optimizers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import tensorflow as tf
print("Built with CUDA?", tf.test.is_built_with_cuda())
print("GPUs available:", tf.config.list_physical_devices('GPU'))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Add, Flatten, Dense, LeakyReLU,
    BatchNormalization, Lambda, Concatenate
)

# stump (core)
def res_block(x, filters, kernel_size):
    skip = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = LeakyReLU(alpha=0.02)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = Add()([x, skip])
    x = LeakyReLU(alpha=0.02)(x)
    return x


def build_stump(input_shape=(8, 8, 9)):
    inputs = Input(shape=input_shape, name='board')
    
    # Big first conv to grab wide patterns
    x = Conv2D(256, (3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.05)(x)
    
    # Residual tower
    x = res_block(x, 256, (3, 3))
    x = res_block(x, 256, (3, 3))
    
    # Downscale channels, preserve space
    x = Conv2D(128, (1, 1))(x)
    x = LeakyReLU(alpha=0.05)(x)
    
    # More convs
    x = Conv2D(128, (2, 2), padding='valid')(x)
    x = LeakyReLU(alpha=0.05)(x)
    
    x = Conv2D(64, (2, 2), padding='valid')(x)
    x = LeakyReLU(alpha=0.05)(x)
    
    trunk = Flatten(name='trunk_flat')(x)
    stump = Model(inputs=inputs, outputs=trunk, name='stump')
    return stump


# head builders
def value_head(trunk, name='value', extra_feats=None):
    x = trunk if extra_feats is None else Concatenate(name=f'{name}_in')(
        [trunk, extra_feats]
    )
    x = Dense(512, name=f'{name}_dense1')(x)
    x = LeakyReLU(alpha=0.05, name=f'{name}_lrelu1')(x)
    x = Dense(128, name=f'{name}_dense2')(x)
    x = LeakyReLU(alpha=0.05, name=f'{name}_lrelu2')(x)
    out = Dense(1, activation='tanh', name=name)(x)
    return out, 'mse'


def count_vec_head(trunk, name, shape):
    x = Dense(128, name=f'{name}_dense1')(trunk)
    x = LeakyReLU(alpha=0.05, name=f'{name}_lrelu1')(x)
    
    # softplus -> strictly positive; add small epsilon shift if you want near-zeros
    out = Dense(shape, activation='softplus', name=name)(x)
    
    # 'poisson' works well for sparse counts; 'mse' is fine too
    return out, 'poisson'


def binary_head(trunk, name, shape=1):
    x = Dense(128, name=f'{name}_dense1')(trunk)
    x = LeakyReLU(alpha=0.05, name=f'{name}_lrelu1')(x)
    out = Dense(shape, activation='sigmoid', name=name)(x)
    return out, 'binary_crossentropy'


def regression_head(trunk, name, shape=1):
    x = Dense(128, name=f'{name}_dense1')(trunk)
    x = LeakyReLU(alpha=0.05, name=f'{name}_lrelu1')(x)
    out = Dense(shape, activation='linear', name=name)(x)
    return out, 'mse'



def probe_head(trunk, name):
    """
    name: e.g. 'queen_in_danger', 'opp_queen_hanging', 'hanging_opp_value'
    Returns: (out_tensor, loss_name, detached_feature_tensor)
    """
    x = Dense(128, name=f'{name}_dense1')(trunk)
    x = LeakyReLU(alpha=0.05, name=f'{name}_lrelu1')(x)

    if name == 'hanging_opp_value':
        out = Dense(1, activation='linear', name=name)(x)
        loss = 'mse'
    else:
        out = Dense(1, activation='sigmoid', name=name)(x)
        loss = 'binary_crossentropy'

    det = Lambda(lambda t: tf.stop_gradient(t), name=f'sg_{name}')(out)
    return out, loss, det


def assemble_model(
    stump,
    heads_spec,
    optimizer=None,
    loss_weights=None,
    include_probe_feats_in_value=True
):
    """
    heads_spec: list of tuples, each a single head, e.g.:
      ('probe',  {'name':'queen_in_danger'})
      ('probe',  {'name':'opp_queen_hanging'})
      ('probe',  {'name':'hanging_opp_value'})
      ('value',  {'name':'value'})
      ('binary', {'name':'white_queen_hanging'})
      ('reg',    {'name':'some_vector_reg', 'shape':4})  # if you use vector reg
    """
    trunk = stump.output
    outputs = []
    losses  = {}
    probe_feats = []

    # First pass: build probes so value can consume their detached outputs
    for kind, cfg in heads_spec:
        if kind == 'probe':
            out, loss, det = probe_head(trunk, cfg['name'])
            outputs.append(out)
            losses[out.name.split(':')[0]] = loss
            probe_feats.append(det)

    # Concatenate all probe features (if any)
    probe_feat_tensor = None
    if probe_feats:
        if len(probe_feats) == 1:
            probe_feat_tensor = probe_feats[0]
        else:
            probe_feat_tensor = Concatenate(name='probe_feats')(probe_feats)

    # Second pass: other heads (value can take probe features)
    for kind, cfg in heads_spec:
        if kind == 'probe':
            continue

        if kind == 'value':
            out, loss = value_head(
                trunk, name=cfg.get('name', 'value'),
                extra_feats=probe_feat_tensor if include_probe_feats_in_value else None
            )

        elif kind == 'binary':
            out, loss = binary_head(trunk, name=cfg['name'], shape=cfg.get('shape', 1))
        
        elif kind == 'count':
            out, loss = count_vec_head(trunk, name=cfg['name'], shape=cfg['shape'])
    
        elif kind == 'regression':
            out, loss = regression_head(
                trunk, name=cfg['name'], shape=cfg.get('shape', 1)
            )

        else:
            raise ValueError(f'Unknown head kind: {kind}')
        
        outputs.append(out)
        losses[cfg['name']] = loss

    model = Model(inputs=stump.input, outputs=outputs, name='stump_with_heads')

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

    if loss_weights is None:
        loss_weights = {o.name.split(':')[0]: 1.0 for o in outputs}
    
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
    
    # keep our own copy since Keras doesn't expose loss_weights as an attr
    model._head_loss_weights = dict(loss_weights)
    model._head_losses = dict(losses)  # also stash losses so we can recompile cleanly

    return model


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


## example usage
# 1) build stump
stump = build_stump(input_shape=(8,8,9))

# 2) choose heads now, but you can add more later
heads_spec = [
    ('value',  {'name': 'value'}),

    # king exposure bundle (2 = [white, black])
    ('count',  {'name': 'king_ray_exposure',   'shape': 2}),
    ('count',  {'name': 'king_pawn_shield',    'shape': 2}),
    ('count',  {'name': 'king_escape_square',  'shape': 2}),
    ('count',  {'name': 'king_openfile',       'shape': 2}),
    ('count',  {'name': 'king_semi_openfile',  'shape': 2}),

    # pawns
    ('count',  {'name': 'pawn_hanging',        'shape': 2}),
    ('count',  {'name': 'pawn_free_to_take',   'shape': 2}),
    ('count',  {'name': 'pawn_en_prise',       'shape': 2}),

    # knights
    ('count',  {'name': 'knight_hanging',      'shape': 2}),
    ('count',  {'name': 'knight_free_to_take', 'shape': 2}),
    ('count',  {'name': 'knight_en_prise',     'shape': 2}),
    ('count',  {'name': 'knight_attacked_by_lower_value', 'shape': 2}),

    # bishops
    ('count',  {'name': 'bishop_hanging',      'shape': 2}),
    ('count',  {'name': 'bishop_free_to_take', 'shape': 2}),
    ('count',  {'name': 'bishop_en_prise',     'shape': 2}),
    ('count',  {'name': 'bishop_attacked_by_lower_value', 'shape': 2}),

    # rooks
    ('count',  {'name': 'rook_hanging',        'shape': 2}),
    ('count',  {'name': 'rook_free_to_take',   'shape': 2}),
    ('count',  {'name': 'rook_en_prise',       'shape': 2}),
    ('count',  {'name': 'rook_attacked_by_lower_value', 'shape': 2}),

    # queens
    ('binary',  {'name': 'queen_hanging',       'shape': 2}),
    ('binary',  {'name': 'queen_free_to_take',  'shape': 2}),
    ('binary',  {'name': 'queen_en_prise',      'shape': 2}),
    ('binary',  {'name': 'queen_attacked_by_lower_value', 'shape': 2}),

    # material (pair) + checkmate (scalar)
    ('count',  {'name': 'material',            'shape': 2}),
    ('binary', {'name': 'is_checkmate',        'shape': 1}),
]


# 3) compile with custom weights (toggle any head by setting weight=0.0)
loss_weights = {
    'value': 3.0,

    'king_ray_exposure': 0.3,
    'king_pawn_shield':  0.2,
    'king_escape_square':0.2,
    'king_openfile':     0.2,
    'king_semi_openfile':0.2,

    'pawn_hanging':                0.2,
    'pawn_free_to_take':           0.2,
    'pawn_en_prise':               0.2,

    'knight_hanging':              0.3,
    'knight_free_to_take':         0.3,
    'knight_en_prise':             0.3,
    'knight_attacked_by_lower_value': 0.3,

    'bishop_hanging':              0.3,
    'bishop_free_to_take':         0.3,
    'bishop_en_prise':             0.3,
    'bishop_attacked_by_lower_value': 0.3,

    'rook_hanging':                0.4,
    'rook_free_to_take':           0.4,
    'rook_en_prise':               0.4,
    'rook_attacked_by_lower_value':0.4,

    'queen_hanging':               0.6,
    'queen_free_to_take':          0.6,
    'queen_en_prise':              0.6,
    'queen_attacked_by_lower_value': 0.6,

    'material': 0.005,
    'checkmate_radar': 0.5,
}


# model = assemble_model(
#     stump, heads_spec, optimizer=tf.keras.optimizers.Adam(1e-3),
#     loss_weights=loss_weights
# )

#set_head_weights(model, loss_weights)

# 5) value-only submodel for play
#value_only = build_value_only(model, value_name='value')

def replace_is_checkmate_with_radar(
    model,
    trunk_layer_name='trunk_flat',
    new_head_name='checkmate_radar',
    new_head_units=1,
    new_head_weight=1.0,
    activation='linear',  # 'tanh' for [-1,1] mapping, 'linear' for unbounded
    loss='mse'             # regression loss
):
    """
    Remove the 'is_checkmate' head and add a fresh regression head `checkmate_radar`.
    Reuses all existing layers/weights; only the new head is randomly initialized.
    """
    # 1) Shared trunk
    trunk = model.get_layer(trunk_layer_name).output

    # 2) New regression head
    x = Dense(128, name=f'{new_head_name}_dense1')(trunk)
    x = LeakyReLU(alpha=0.05, name=f'{new_head_name}_lrelu1')(x)
    radar_out = Dense(new_head_units, activation=activation, name=new_head_name)(x)

    # 3) Keep outputs except the old head
    def head_name(t):
        return t.name.split(':')[0].split('/')[0]

    kept_outputs = [t for t in model.outputs if head_name(t) != 'is_checkmate']
    new_outputs = kept_outputs + [radar_out]

    new_model = Model(inputs=model.inputs, outputs=new_outputs, name='stump_with_heads')

    # 4) Build a correct loss dict for all outputs
    out_names = [head_name(o) for o in new_model.outputs]

    old_losses  = getattr(model, "_head_losses", {}) or {}
    old_weights = getattr(model, "_head_loss_weights", {}) or {}

    losses = {}
    weights = {}
    for n in out_names:
        losses[n]  = old_losses.get(n, "mse")
        weights[n] = old_weights.get(n, 1.0)

    # Force the new head’s loss/weight
    losses[new_head_name] = loss
    weights[new_head_name] = new_head_weight

    # 5) Fresh optimizer (no old state)
    lr = getattr(model.optimizer, 'learning_rate', 1e-3)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # 6) Compile and stash
    new_model.compile(optimizer=opt, loss=losses, loss_weights=weights)
    new_model._head_losses = dict(losses)
    new_model._head_loss_weights = dict(weights)

    return new_model


# ---------- helpers ----------

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

# ---------- core ops ----------

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
    if kind == 'value':
        out, loss_name = value_head(trunk, name=name, **kwargs)
    elif kind == 'binary':
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



# model = tf.keras.models.load_model("C:/Users/Bryan/repos/chessbot/chess_model_multihead_v240_2.keras")

# new_model = remove_head(model, "king_semi_openfile")
# new_model = remove_head(new_model, "king_openfile")
# new_model = remove_head(new_model, "is_checkmate")

# new_model = add_head(new_model, 'regression', 'checkmate_radar', 1, loss_weight=0.75)
# # You already have `model` trained in memory
# #model = replace_is_checkmate_with_radar(model)

# # sanity check: list output names
# print([t.name.split(':')[0] for t in new_model.outputs])
# ...should include 'checkmate_radar' and no 'is_checkmate'

#model.save("C:/Users/Bryan/repos/chessbot/chess_model_multihead_v240_2_2.keras")
