import os
import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"
HE = "he_normal"
BIG_NEG = -1e9
EPS = 1e-9

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
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


class MaskedPolicyModel(tf.keras.Model):
    def __init__(self, core):
        super().__init__()
        
        self.core = core
        self.policy_loss_tracker = tf.keras.metrics.Mean(name="policy_loss")
        self.value_loss_tracker = tf.keras.metrics.Mean(name="value_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.policy_loss_tracker,
                self.value_loss_tracker]

    def compile(self, optimizer, policy_loss_weight=1.0, value_loss_weight=1.0,
                **kwargs):
        super().compile(**kwargs)
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight

    def train_step(self, data):
        (boards, legal_mask), y = data
        labels = y["policy"]
        values = y["value"]

        with tf.GradientTape() as tape:
            logits, v_pred = self.core([boards, legal_mask], training=True)
            B = tf.shape(logits)[0]
            logits = tf.reshape(logits, [B, -1])
            labels = tf.reshape(labels, [B, -1])
            mask = tf.cast(tf.reshape(legal_mask, [B, -1]), logits.dtype)

            masked_logits = tf.where(mask > 0.5,
                                     logits,
                                     tf.ones_like(logits) * BIG_NEG)
            logp = tf.nn.log_softmax(masked_logits, axis=1)

            label_sums = tf.reduce_sum(labels, axis=1, keepdims=True)
            labels_norm = labels / (label_sums + EPS)

            per_sample_ce = -tf.reduce_sum(labels_norm * logp, axis=1)
            valid = tf.squeeze(label_sums > EPS, axis=1)
            policy_loss = tf.reduce_sum(
                tf.where(valid, per_sample_ce, tf.zeros_like(per_sample_ce))
            ) / (tf.reduce_sum(tf.cast(valid, tf.float32)) + EPS)

            value_loss = tf.reduce_mean(tf.square(v_pred - values))
            total_loss = (self.policy_loss_weight * policy_loss +
                          self.value_loss_weight * value_loss)

        grads = tape.gradient(total_loss, self.core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.core.trainable_variables))

        self.policy_loss_tracker.update_state(policy_loss)
        self.value_loss_tracker.update_state(value_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {"loss": self.total_loss_tracker.result(),
                "policy_loss": self.policy_loss_tracker.result(),
                "value_loss": self.value_loss_tracker.result()}

    def test_step(self, data):
        (boards, legal_mask), y = data
        labels = y["policy"]
        values = y["value"]

        logits, v_pred = self.core([boards, legal_mask], training=False)
        B = tf.shape(logits)[0]
        logits = tf.reshape(logits, [B, -1])
        labels = tf.reshape(labels, [B, -1])
        mask = tf.cast(tf.reshape(legal_mask, [B, -1]), logits.dtype)

        masked_logits = tf.where(mask > 0.5,
                                 logits,
                                 tf.ones_like(logits) * BIG_NEG)
        logp = tf.nn.log_softmax(masked_logits, axis=1)

        label_sums = tf.reduce_sum(labels, axis=1, keepdims=True)
        labels_norm = labels / (label_sums + EPS)
        per_sample_ce = -tf.reduce_sum(labels_norm * logp, axis=1)
        valid = tf.squeeze(label_sums > EPS, axis=1)
        policy_loss = tf.reduce_sum(
            tf.where(valid, per_sample_ce, tf.zeros_like(per_sample_ce))
        ) / (tf.reduce_sum(tf.cast(valid, tf.float32)) + EPS)

        value_loss = tf.reduce_mean(tf.square(v_pred - values))
        total_loss = (self.policy_loss_weight * policy_loss +
                      self.value_loss_weight * value_loss)

        self.policy_loss_tracker.update_state(policy_loss)
        self.value_loss_tracker.update_state(value_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {"loss": self.total_loss_tracker.result(),
                "policy_loss": self.policy_loss_tracker.result(),
                "value_loss": self.value_loss_tracker.result()}
    
    def call(self, inputs, training):
        return self.core(inputs, training=training)
    
    def save(self, path):
        self.core.save(path)

    @classmethod
    def from_saved(cls, path):
        """
        Load a MaskedPolicyModel previously saved with save(path).
        Returns a built subclass instance (not compiled).
        """
        # load the functional core (works for SavedModel dir or .h5 file)
        core_loaded = tf.keras.models.load_model(path)
        inst = cls(core_loaded)
        # try to build so model.summary() is usable immediately
        try:
            inst.build(input_shape=[(None, 8, 8, 29), (None, 4096)])
        except Exception:
            # if build fails (nonstandard shapes), ignore — user can call build/call
            pass
        return inst

    @classmethod
    def build_core(self, n_blocks, name):
        board_in = Input(shape=(8, 8, 29), name="board")
        legal_in = Input(shape=(4096,), name="legal_mask")

        x = layers.Conv2D(256, 3, padding="same", name="conv0")(board_in)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        # Residual-ish trunk: N blocks of 256 filters (wide & deep)
        for i in range(n_blocks):
            r = layers.Conv2D(256, 3, padding="same", name=f"b{i}_c1")(x)
            r = layers.BatchNormalization()(r)
            r = layers.LeakyReLU()(r)
            r = layers.Conv2D(256, 3, padding="same", name=f"b{i}_c2")(r)
            r = layers.BatchNormalization()(r)
            x = layers.Add()([x, r])
            x = layers.LeakyReLU()(x)

        trunk = x  # shape (B, 8, 8, 256)

        # Policy head: 1x1 conv -> 64 channels -> flatten to 4096 logits
        p = layers.Conv2D(64, 1, padding="same", name="policy_conv1x1")(trunk)
        policy_logits = layers.Reshape((4096,), name="policy_logits")(p)

        # Value head: GAP -> MLP -> tanh in [-1,1]
        v = layers.GlobalAveragePooling2D(name="gap")(trunk)
        v = layers.Dense(1024, activation="relu", name="v_fc1")(v)
        v = layers.Dense(256, activation="relu", name="v_fc2")(v)
        v_out = layers.Dense(1, activation="tanh", name="value_out")(v)

        core = Model(
            inputs=[board_in, legal_in],
            outputs=[policy_logits, v_out],
            name=name
        )
        
        return core

# from chessbot import MODEL_DIR

# model_loc = MODEL_DIR + "conv_stm_pov_test.h5"
# core = MaskedPolicyModel.build_core(n_blocks=10, name="stm_pov_v1")

# core.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-4),
#     loss=["categorical_crossentropy", "mse"],
#     loss_weights=[1.0, 1.0]
# )

# model = MaskedPolicyModel(core)
# model.save(model_loc)