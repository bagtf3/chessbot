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
from tensorflow.keras.initializers import Zeros, Constant
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


def set_loss_weights(model, loss_weights, steps_per_execution=1, jit=False):
    # Get a fresh optimizer (same config) to avoid double-wrapping
    opt_candidate = model._default_opt
    if isinstance(opt_candidate, tf.keras.mixed_precision.LossScaleOptimizer):
        base_opt = opt_candidate.inner_optimizer
    else:
        base_opt = opt_candidate

    # create a new optimizer instance with same config
    opt_cfg = tf.keras.optimizers.serialize(base_opt)
    opt = tf.keras.optimizers.deserialize(opt_cfg)

    # re-wrap with LossScaleOptimizer when training in mixed float16
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(
        optimizer=opt,
        loss=model._default_loss_dict,
        loss_weights=loss_weights,
        steps_per_execution=steps_per_execution,
        jit_compile=jit,
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


def res_block_layernorm(x, channels, leak=0.01, name=None):
    """
    Residual block with Conv2D -> LayerNorm -> activation -> Conv2D ->
    LayerNorm -> add. This avoids BatchNorm and is stable for small
    batch sizes.
    """
    nm = "" if name is None else name + "_"
    conv1 = layers.Conv2D(
        channels, 3, padding="same", use_bias=False,
        kernel_initializer=HE, name=nm + "conv1"
    )(x)

    ln1 = layers.LayerNormalization(epsilon=1e-5, name=nm + "ln1")(conv1)
    act1 = layers.LeakyReLU(alpha=leak, name=nm + "lrelu1")(ln1)

    conv2 = layers.Conv2D(
        channels, 3, padding="same", use_bias=False,
        kernel_initializer=HE, name=nm + "conv2"
    )(act1)

    ln2 = layers.LayerNormalization(epsilon=1e-5, name=nm + "ln2")(conv2)

    out = layers.Add(name=nm + "add")([x, ln2])
    out = layers.LeakyReLU(alpha=leak, name=nm + "lrelu_out")(out)
    return out


def build_conv_flat_64x67(
    d_model_embed=128,
    vocab_size=21,
    filters=256,
    n_blocks=9,
    name="conv_flat_64x67",
):
    """
    Conv-heavy model that uses LayerNormalization in the trunk rather
    than BatchNormalization. This is better for small or variable batch
    sizes commonly seen in selfplay.
    """
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")

    de, fl, vs = d_model_embed, filters, vocab_size

    enc_in = Input(shape=(64,), dtype="int32", name="enc_in")
    token_emb = layers.Embedding(input_dim=vs, output_dim=de, name="token_emb")(enc_in)
    x = layers.Reshape((8, 8, de), name="to_2d")(token_emb)

    x = layers.Conv2D(
        fl, 3, padding="same", use_bias=False,
        kernel_initializer=HE, name="conv_init"
    )(x)

    # drop BatchNorm here, rely on LayerNorm inside blocks
    x = layers.LeakyReLU(alpha=0.01, name="lrelu_init")(x)

    for i in range(n_blocks):
        x = res_block_layernorm(x, fl, leak=0.01, name=f"res{i}")

    x = layers.Conv2D(fl, 1, padding="same", use_bias=False, name="conv_mix_1x1")(x)
    x = layers.LayerNormalization(epsilon=1e-5, name="ln_mix")(x)
    x = layers.LeakyReLU(alpha=0.01, name="lrelu_mix")(x)

    policy_map = layers.Conv2D(
        67, 1, padding="same", activation=None, name="policy_conv_67"
    )(x)

    feat_64_67 = layers.Reshape((64, 67), name="to_64_67")(policy_map)
    policy_logits = layers.Reshape((64 * 67,), name="policy_logits")(feat_64_67)

    v_pool = layers.GlobalAveragePooling2D(name="v_gap")(x)
    v = layers.Dense(fl // 2, activation="relu", name="v_fc1")(v_pool)
    v = layers.Dense(fl // 4, activation="relu", name="v_fc2")(v)
    value_out = layers.Dense(1, activation="tanh", dtype="float32", name="value_out")(v)

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


def warm_conv_infer(graph, max_bs):
    for _ in range(100):
        rep_mask = (np.random.rand(max_bs, 4288) < 0.02).astype(np.int32)
        rep_enc = (np.random.rand(max_bs, 64) < 0.32).astype(np.int32)
        _ = graph(tf.convert_to_tensor(rep_enc), tf.convert_to_tensor(rep_mask))


def make_conv_infer(
    model,
    max_bs=1024,
    uniform_eps=0.0,
    prior_clip_max=1.0,
    vscale=0.9,
):
    """
    Returns fwd((enc_np, legal_np)) -> (probs_np, val_np).

    uniform_eps mixes uniform mass over legal moves:
        probs = (1 - eps) * probs + eps * uniform_legal

    prior_clip_max caps peaks after mixing (no floor clipping):
        probs = min(probs, clip_max), then renorm
    """

    big_neg = tf.constant(-1e9, dtype=tf.float32)
    eps_small = tf.constant(1e-12, dtype=tf.float32)

    ue = tf.constant(float(uniform_eps), dtype=tf.float32)
    ue = tf.clip_by_value(ue, 0.0, 1.0)

    pcm = tf.constant(float(prior_clip_max), dtype=tf.float32)
    value_scale_c = tf.constant(float(vscale), dtype=tf.float32)

    @tf.function(input_signature=[
        tf.TensorSpec([None, 64], tf.int32),
        tf.TensorSpec([None, 4288], tf.int32),
    ], experimental_compile=True)
    def graph(enc, legal):
        logits, value = model(enc, training=False)
        logits = tf.reshape(logits, [tf.shape(logits)[0], -1])

        mask = tf.cast(tf.reshape(legal, [tf.shape(logits)[0], -1]), tf.float32)
        logits32 = tf.cast(logits, tf.float32)

        masked_logits32 = tf.where(mask > 0.5, logits32, big_neg)

        row_max = tf.reduce_max(masked_logits32, axis=1, keepdims=True)
        exp = tf.exp(masked_logits32 - row_max) * mask
        sumexp = tf.reduce_sum(exp, axis=1, keepdims=True)

        has_any = sumexp > 0.0
        probs = tf.where(has_any, exp / (sumexp + eps_small), tf.zeros_like(exp))

        if uniform_eps != 0.0:
            legal_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
            uni = tf.where(
                legal_sum > 0.0,
                mask / (legal_sum + eps_small),
                tf.zeros_like(mask),
            )
            probs = (1.0 - ue) * probs + ue * uni

        if prior_clip_max < 1.0:
            probs = tf.minimum(probs, pcm) * mask

        s = tf.reduce_sum(probs, axis=1, keepdims=True)
        probs = tf.where(s > eps_small, probs / (s + eps_small),
                         tf.zeros_like(probs))

        value_f = tf.cast(value, tf.float32) * value_scale_c
        return probs, value_f

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
                parts[1] = np.concatenate([parts[1], p_val], axis=0)
            i = j

        return parts

    return fwd


def squeeze_excitation(x, channels, reduction=4, name=None):
    nm = "" if name is None else name + "_"
    se = layers.GlobalAveragePooling2D(name=nm + "gap")(x)
    se = layers.Dense(channels // reduction, activation="relu",
                      kernel_initializer=HE,
                      name=nm + "fc1")(se)
                      
    # make last dense start as near-identity
    # zero weights, bias ~3 -> sigmoid(3)=0.95
    se = layers.Dense(channels, activation="sigmoid",
                      kernel_initializer=Zeros(),
                      bias_initializer=Constant(3.0),
                      name=nm + "fc2")(se)

    se = layers.Reshape((1, 1, channels), name=nm + "reshape")(se)
    return layers.Multiply(name=nm + "scale")([x, se])


def res_block_layernormSE(x, channels, reduction, leak=0.01, name=None):
    """
    Residual block reworked to use LayerNorm and a small SE module.
    Keeps the same public name for easy replacement.
    """
    nm = "" if name is None else name + "_"

    conv1 = layers.Conv2D(
        channels, 3, padding="same", use_bias=False,
        kernel_initializer=HE, name=nm + "conv1"
    )(x)

    ln1 = layers.LayerNormalization(axis=-1, name=nm + "ln1")(conv1)
    act1 = layers.LeakyReLU(alpha=leak, name=nm + "lrelu1")(ln1)

    conv2 = layers.Conv2D(
        channels, 3, padding="same", use_bias=False,
        kernel_initializer=HE, name=nm + "conv2"
    )(act1)

    ln2 = layers.LayerNormalization(axis=-1, name=nm + "ln2")(conv2)

    if reduction > 0:
        se_out = squeeze_excitation(ln2, channels, reduction=reduction,
                                    name=nm + "se")
        out = layers.Add(name=nm + "add")([x, se_out])
    else:
        out = ln2

    out = layers.LeakyReLU(alpha=leak, name=nm + "lrelu_out")(out)
    return out


def build_conv_flat_64x67SE(
    name,
    d_model_embed=128,
    vocab_size=21,
    filters=256,
    n_blocks=9,
    reduction=16
):
    """
    Conv-heavy model using LayerNorm in the trunk and SE inside each block.
    """
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")

    de, fl, vs = d_model_embed, filters, vocab_size

    enc_in = Input(shape=(64,), dtype="int32", name="enc_in")
    token_emb = layers.Embedding(input_dim=vs, output_dim=de, name="token_emb")(enc_in)
    x = layers.Reshape((8, 8, de), name="to_2d")(token_emb)

    x = layers.Conv2D(
        fl, 3, padding="same", use_bias=False,
        kernel_initializer=HE, name="conv_init"
    )(x)

    x = layers.LayerNormalization(axis=-1, name="ln_init")(x)
    x = layers.LeakyReLU(alpha=0.01, name="lrelu_init")(x)

    for i in range(n_blocks):
        x = res_block_layernormSE(x, fl, reduction=reduction, leak=0.01, name=f"res{i}")

    x = layers.Conv2D(fl, 1, padding="same", use_bias=False,
                      name="conv_mix_1x1")(x)
    x = layers.LayerNormalization(axis=-1, name="ln_mix")(x)
    x = layers.LeakyReLU(alpha=0.01, name="lrelu_mix")(x)

    policy_map = layers.Conv2D(
        67, 1, padding="same", activation=None, name="policy_conv_67"
    )(x)

    feat_64_67 = layers.Reshape((64, 67), name="to_64_67")(policy_map)
    policy_logits = layers.Reshape((64 * 67,), name="policy_logits")(
        feat_64_67
    )

    v_pool = layers.GlobalAveragePooling2D(name="v_gap")(x)
    v = layers.Dense(128, activation="relu", name="v_fc1")(v_pool)
    value_out = layers.Dense(1, activation="tanh", dtype="float32", name="value_out")(v)

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


def build_conformer_64x67(
    name,
    d_model_embed=128,
    vocab_size=21,
    conv_filters=256,
    conv_blocks=4,
    transformer_layers=4,
    num_heads=8,
    ff_dim=1024,
    dropout=0.05
):
    """
    Conformer-like model: conv frontend -> transformer encoder stack ->
    64x67 policy head + value head.
    """
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")

    de, vs = d_model_embed, vocab_size

    # internal: residual conv block using LayerNorm (no BatchNorm)
    def res_block_layernorm(x_in, channels, leak=0.01, name=None):
        nm = "" if name is None else name + "_"

        conv1 = layers.Conv2D(
            channels, 3, padding="same", use_bias=False,
            kernel_initializer=HE, name=nm + "conv1"
        )(x_in)

        act1 = layers.LeakyReLU(alpha=leak, name=nm + "lrelu1")(conv1)

        conv2 = layers.Conv2D(
            channels, 3, padding="same", use_bias=False,
            kernel_initializer=HE, name=nm + "conv2"
        )(act1)

        ln2 = layers.LayerNormalization(epsilon=1e-5, name=nm + "ln2")(conv2)

        out = layers.Add(name=nm + "add")([x_in, ln2])
        out = layers.LeakyReLU(alpha=leak, name=nm + "lrelu_out")(out)
        return out

    # inputs / token embedding -> 2D conv feature map
    enc_in = Input(shape=(64,), dtype="int32", name="enc_in")
    token_emb = layers.Embedding(input_dim=vs, output_dim=de, name="token_emb")(enc_in)
    x = layers.Reshape((8, 8, de), name="to_2d")(token_emb)

    # small conv frontend stack using residual blocks
    # project if embed depth != conv_filters in first block
    if de != conv_filters:
        x = layers.Conv2D(
            conv_filters, 1, padding="same", use_bias=False,
            kernel_initializer=HE, name="conv_proj")(x)
        
        x = layers.LayerNormalization(axis=-1, name="ln_proj")(x)
        x = layers.LeakyReLU(alpha=0.01, name="lrelf_proj")(x)

    for i in range(conv_blocks):
        x = res_block_layernorm(x, conv_filters, leak=0.01, name=f"conv{i}")

    # flatten spatial to sequence for transformer: (batch, 64, C)
    seq = layers.Reshape((64, conv_filters), name="to_seq")(x)

    pos = layers.Embedding(64, conv_filters, name="pos_emb")(
        layers.Lambda(lambda _: tf.range(64), name="pos_idx")(enc_in)
    )
    pos = layers.Lambda(lambda z: tf.expand_dims(z, 0), name="pos_bcast")(pos)
    seq = layers.Add(name="add_pos")([seq, pos])

    # transformer encoder stack
    def transformer_encoder_layer(src, layer_idx):
        nm = f"t{layer_idx}_"
        ln1 = layers.LayerNormalization(axis=-1, name=nm + "ln1")(src)
        
        attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=conv_filters // num_heads,
            name=nm + "mha")(ln1, ln1)

        attn = layers.Dropout(dropout, name=nm + "attn_drop")(attn)
        attn_out = layers.Add(name=nm + "attn_add")([src, attn])

        ln2 = layers.LayerNormalization(axis=-1, name=nm + "ln2")(attn_out)
        ff = layers.Dense(ff_dim, activation="gelu",
                          kernel_initializer=HE, name=nm + "ff1")(ln2)

        ff = layers.Dropout(dropout, name=nm + "ff_drop")(ff)
        ff = layers.Dense(conv_filters, name=nm + "ff2")(ff)
        ff_out = layers.Add(name=nm + "ff_add")([attn_out, ff])
        return ff_out

    t = seq
    for i in range(transformer_layers):
        t = transformer_encoder_layer(t, i)

    # back to spatial map
    x = layers.Reshape((8, 8, conv_filters), name="from_seq")(t)

    # small conv mixer before heads
    x = layers.Conv2D(
        conv_filters, 1, padding="same", use_bias=False, name="conv_mix_1x1")(x)

    x = layers.LayerNormalization(axis=-1, name="ln_mix")(x)
    x = layers.LeakyReLU(alpha=0.01, name="lrelu_mix")(x)

    # policy head -> 64 x 67
    policy_map = layers.Conv2D(
        67, 1, padding="same", activation=None, name="policy_conv_67")(x)

    feat_64_67 = layers.Reshape((64, 67), name="to_64_67")(policy_map)
    policy_logits = layers.Reshape((64 * 67,), name="policy_logits")(feat_64_67)

    # value head
    v = layers.Conv2D(
        64, 3, padding="same", use_bias=False,
        kernel_initializer=HE, name="v_conv1")(x)

    v = layers.LayerNormalization(epsilon=1e-5, name="v_ln1")(v)
    v = layers.LeakyReLU(alpha=0.01, name="v_lrelu1")(v)

    # global descriptor
    v = layers.GlobalAveragePooling2D(name="v_gap")(v)

    # deeper MLP for value (extra dense layer for more capacity)
    v = layers.Dense(256, activation="relu", name="v_fc1")(v)
    v = layers.Dropout(dropout, name="v_fc_drop")(v)
    v = layers.Dense(128, activation="relu", name="v_fc2")(v)

    # final value output (float32 for numerical stability)
    value_out = layers.Dense(
        1, activation="tanh", dtype="float32", name="value_out")(v)

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