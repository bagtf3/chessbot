"""legacy_models.py — TF model builders and inference utilities.

Preserved for xerces.py / xerces_cli.py which still use TF inference.
Not used by the live selfplay loop (pt_eager / ort_trt backends).
"""
from __future__ import annotations

from chessbot.model import VOCAB_SIZE, SEQ_LEN


# ---------------------------------------------------------------------------
# TF shared helpers
# ---------------------------------------------------------------------------

def tf_attn_pool_value_head(x, C, layers, activation="gelu"):
    """Attention-pool value head for TF functional API."""
    if len(x.shape) == 4:
        x_seq = layers.Reshape((64, C), name="v_seq")(x)
    else:
        x_seq = x
    x_seq   = layers.LayerNormalization(axis=-1, name="v_ln")(x_seq)
    attn_w  = layers.Dense(1, name="attn_w")(x_seq)
    attn_w  = layers.Softmax(axis=1, name="attn_softmax")(attn_w)
    attn_wT = layers.Permute((2, 1), name="attn_w_T")(attn_w)
    v = layers.Dot(axes=[2, 1], name="v_pool")([attn_wT, x_seq])
    v = layers.Reshape((C,), name="v_squeeze")(v)
    v = layers.Dense(256, activation=activation, name="v_fc1")(v)
    v = layers.Dense(128, activation=activation, name="v_fc2")(v)
    return layers.Dense(3, dtype="float32", name="value_out")(v)


def tf_relational_policy_head(x, C, layers):
    """Bilinear from/to policy head for TF functional API."""
    import math
    if len(x.shape) == 3:
        x_seq = x
        x_2d  = layers.Reshape((8, 8, C), name="pol_x_to_2d")(x)
    else:
        x_seq = layers.Reshape((64, C), name="pol_to_seq")(x)
        x_2d  = x
    x_seq    = layers.LayerNormalization(name="pol_prenorm")(x_seq)
    from_h   = layers.Dense(256, activation="gelu", name="pol_from_fc1")(x_seq)
    from_vec = layers.Dense(256, name="pol_from_fc2")(from_h)
    to_h     = layers.Dense(256, activation="gelu", name="pol_to_fc1")(x_seq)
    to_vec   = layers.Dense(256, name="pol_to_fc2")(to_h)
    norm = layers.Dot(axes=[2, 2], name="pol_qk_dot")([from_vec, to_vec])
    norm = layers.Lambda(lambda t: t * (1.0 / math.sqrt(256)), name="pol_scale")(norm)
    norm = layers.Reshape((64 * 64,), name="pol_normal_flat")(norm)
    p    = layers.Conv2D(64, 1, use_bias=False, padding="same", name="pol_promo_mix")(x_2d)
    p    = layers.LayerNormalization(axis=-1, name="pol_promo_mix_ln")(p)
    p    = layers.LeakyReLU(0.02, name="pol_promo_mix_act")(p)
    skip = p
    p    = layers.Conv2D(64, 3, use_bias=False, padding="same", name="pol_promo_c1")(p)
    p    = layers.LayerNormalization(axis=-1, name="pol_promo_ln")(p)
    p    = layers.LeakyReLU(0.02, name="pol_promo_act")(p)
    p    = layers.Add(name="pol_promo_skip")([p, skip])
    promo = layers.Conv2D(3, 1, padding="same", name="pol_promo_conv")(p)
    promo = layers.Reshape((8 * 8 * 3,), name="pol_promo_flat")(promo)
    return layers.Concatenate(axis=1, name="policy_logits")([norm, promo])


def tf_compile(model):
    """Compile a TF model with Adam+LossScale and CE losses, set default attrs."""
    import tensorflow as tf
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out":     tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    }
    model.compile(optimizer=opt, loss=loss_dict,
                  loss_weights={"policy_logits": 1.0, "value_out": 1.0})
    model._default_opt       = opt
    model._default_loss_dict = loss_dict
    return model


# ---------------------------------------------------------------------------
# TF builders
# ---------------------------------------------------------------------------

def build_tf_transformer(cfg: dict):
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    de = cfg["d_embed"]
    tl = cfg["transformer_layers"]
    nh = cfg["num_heads"]
    fd = cfg["ff_dim"]
    dr = cfg["dropout"]

    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x   = layers.Embedding(VOCAB_SIZE, de, name="token_emb")(inp)
    pos = layers.Embedding(64, de, name="pos_emb")(
        tf.keras.backend.arange(0, 64, dtype="int32")
    )
    pos = layers.Lambda(lambda t: tf.expand_dims(t, axis=0), name="pos_expand")(pos)
    x   = x + pos

    for i in range(tl):
        r = x
        x = layers.LayerNormalization(axis=-1, name=f"t{i}_ln1")(x)
        x = layers.MultiHeadAttention(
            num_heads=nh, key_dim=de // nh, dropout=0.0, name=f"t{i}_mha"
        )(x, x)
        x = layers.Dropout(dr, name=f"t{i}_drop1")(x)
        x = x + r
        r = x
        x = layers.LayerNormalization(axis=-1, name=f"t{i}_ln2")(x)
        x = layers.Dense(fd, activation="gelu", name=f"t{i}_ff1")(x)
        x = layers.Dropout(dr, name=f"t{i}_drop2")(x)
        x = layers.Dense(de, name=f"t{i}_ff2")(x)
        x = x + r

    x_2d = layers.Reshape((8, 8, de), name="x_to_2d")(x)
    v    = layers.Conv2D(de, 1, use_bias=False, padding="same", name="value_mix")(x_2d)
    v    = layers.LayerNormalization(axis=-1, name="value_mix_ln")(v)
    v    = layers.LeakyReLU(0.02, name="value_mix_act")(v)

    value  = tf_attn_pool_value_head(v, de, layers)
    policy = tf_relational_policy_head(x_2d, de, layers)

    model = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


def build_tf_conformer_interweaved(cfg: dict):
    """Interweaved conformer: cb x [prenorm-MHA -> ConvRes block]."""
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    de = cfg["d_embed"]
    cf = cfg["conv_filters"]
    cb = cfg["conv_blocks"]
    nh = cfg["num_heads"]
    dr = cfg["dropout"]

    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x   = layers.Embedding(VOCAB_SIZE, de, name="token_emb")(inp)
    x   = layers.Reshape((8, 8, de), name="to_2d")(x)

    if de != cf:
        x = layers.Conv2D(cf, 1, use_bias=False, padding="same", name="conv_proj")(x)
        x = layers.LayerNormalization(axis=-1, name="ln_proj")(x)
        x = layers.LeakyReLU(0.02, name="lrelu_proj")(x)

    pos = layers.Embedding(64, cf, name="pos_emb")(
        tf.keras.backend.arange(0, 64, dtype="int32")
    )
    pos = layers.Lambda(lambda t: tf.expand_dims(t, axis=0), name="pos_expand")(pos)

    for i in range(cb):
        s = layers.Reshape((64, cf), name=f"b{i}_to_seq")(x)
        if i == 0:
            s = s + pos
        r = s
        s = layers.LayerNormalization(axis=-1, name=f"b{i}_ln1")(s)
        s = layers.MultiHeadAttention(
            num_heads=nh, key_dim=cf // nh, dropout=0.0, name=f"b{i}_mha"
        )(s, s)
        s = layers.Dropout(dr, name=f"b{i}_drop1")(s)
        s = s + r
        x = layers.Reshape((8, 8, cf), name=f"b{i}_to_2d")(s)
        r = x
        h = layers.LayerNormalization(axis=-1, name=f"b{i}_ln2")(x)
        h = layers.Conv2D(cf, 3, use_bias=False, padding="same", name=f"b{i}_c1")(h)
        h = layers.LeakyReLU(0.02, name=f"b{i}_lr1")(h)
        h = layers.Conv2D(cf, 3, use_bias=False, padding="same", name=f"b{i}_c2")(h)
        x = r + h

    v    = layers.Conv2D(cf, 1, use_bias=False, padding="same", name="value_mix")(x)
    v    = layers.LayerNormalization(axis=-1, name="value_mix_ln")(v)
    v    = layers.LeakyReLU(0.02, name="value_mix_act")(v)
    value  = tf_attn_pool_value_head(v, cf, layers)
    policy = tf_relational_policy_head(x, cf, layers)

    model = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


def build_tf_transformer_branched(cfg: dict):
    """Shared trunk splits into 3 independent branch MHAs for value/policy-from/policy-to."""
    import math
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    de = cfg["d_embed"]
    tl = cfg["transformer_layers"]
    nh = cfg["num_heads"]
    fd = cfg["ff_dim"]
    dr = cfg["dropout"]

    def prenorm_mha(x, tag):
        r = x
        x = layers.LayerNormalization(axis=-1, name=f"{tag}_ln")(x)
        x = layers.MultiHeadAttention(
            num_heads=nh, key_dim=de // nh, dropout=0.0, name=f"{tag}_mha"
        )(x, x)
        x = layers.Dropout(dr, name=f"{tag}_drop")(x)
        return x + r

    def prenorm_ff(x, out_dim, tag):
        r = x
        x = layers.LayerNormalization(axis=-1, name=f"{tag}_ln2")(x)
        x = layers.Dense(fd, activation="gelu", name=f"{tag}_ff1")(x)
        x = layers.Dropout(dr, name=f"{tag}_drop2")(x)
        x = layers.Dense(out_dim, name=f"{tag}_ff2")(x)
        return x + r if out_dim == de else x

    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x   = layers.Embedding(VOCAB_SIZE, de, name="token_emb")(inp)
    pos = layers.Embedding(64, de, name="pos_emb")(
        tf.keras.backend.arange(0, 64, dtype="int32")
    )
    pos = layers.Lambda(lambda t: tf.expand_dims(t, axis=0), name="pos_expand")(pos)
    x   = x + pos

    for i in range(tl):
        x = prenorm_mha(x, f"t{i}")
        x = prenorm_ff(x, de, f"t{i}")

    v_seq  = prenorm_mha(x, "bv");  v_seq  = prenorm_ff(v_seq, de, "bv")
    pf_seq = prenorm_mha(x, "bf");  pf_seq = prenorm_ff(pf_seq, de, "bf")
    pt_seq = prenorm_mha(x, "bt");  pt_seq = prenorm_ff(pt_seq, de, "bt")

    v_2d  = layers.Reshape((8, 8, de), name="v_to_2d")(v_seq)
    value = tf_attn_pool_value_head(v_2d, de, layers)

    pf_n     = layers.LayerNormalization(name="pol_from_prenorm")(pf_seq)
    from_h   = layers.Dense(256, activation="gelu", name="pol_from_fc1")(pf_n)
    from_vec = layers.Dense(256, name="pol_from_fc2")(from_h)
    pt_n     = layers.LayerNormalization(name="pol_to_prenorm")(pt_seq)
    to_h     = layers.Dense(256, activation="gelu", name="pol_to_fc1")(pt_n)
    to_vec   = layers.Dense(256, name="pol_to_fc2")(to_h)

    norm  = layers.Dot(axes=[2, 2], name="pol_qk_dot")([from_vec, to_vec])
    norm  = layers.Lambda(lambda t: t * (1.0 / math.sqrt(256)), name="pol_scale")(norm)
    norm  = layers.Reshape((64 * 64,), name="pol_normal_flat")(norm)

    x_2d  = layers.Reshape((8, 8, de), name="x_to_2d")(x)
    p     = layers.Conv2D(64, 1, use_bias=False, padding="same", name="pol_promo_mix")(x_2d)
    p     = layers.LayerNormalization(axis=-1, name="pol_promo_mix_ln")(p)
    p     = layers.LeakyReLU(0.02, name="pol_promo_mix_act")(p)
    skip  = p
    p     = layers.Conv2D(64, 3, use_bias=False, padding="same", name="pol_promo_c1")(p)
    p     = layers.LayerNormalization(axis=-1, name="pol_promo_ln")(p)
    p     = layers.LeakyReLU(0.02, name="pol_promo_act")(p)
    p     = layers.Add(name="pol_promo_skip")([p, skip])
    promo = layers.Conv2D(3, 1, padding="same", name="pol_promo_conv")(p)
    promo = layers.Reshape((8 * 8 * 3,), name="pol_promo_flat")(promo)

    policy = layers.Concatenate(axis=1, name="policy_logits")([norm, promo])

    model = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


TF_BUILDERS: dict[str, object] = {
    "16m-transformer":           build_tf_transformer,
    "16m-conformer-interweaved": build_tf_conformer_interweaved,
    "16m-transformer-branched":  build_tf_transformer_branched,
}


# ---------------------------------------------------------------------------
# Active legacy production model (TF conformer, superseded by PT .ts)
# ---------------------------------------------------------------------------

def build_conformer_64x67(
    name,
    d_model_embed=128,
    vocab_size=21,
    conv_filters=256,
    conv_blocks=4,
    transformer_layers=4,
    num_heads=8,
    ff_dim=1024,
    dropout=0.05,
):
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")

    de, vs = d_model_embed, vocab_size
    HE = "he_normal"

    def res_block_ln(x_in, channels, leak=0.01, nm=""):
        h = layers.Conv2D(channels, 3, padding="same", use_bias=False,
                          kernel_initializer=HE, name=nm + "conv1")(x_in)
        h = layers.LeakyReLU(alpha=leak, name=nm + "lrelu1")(h)
        h = layers.Conv2D(channels, 3, padding="same", use_bias=False,
                          kernel_initializer=HE, name=nm + "conv2")(h)
        h = layers.LayerNormalization(epsilon=1e-5, name=nm + "ln2")(h)
        out = layers.Add(name=nm + "add")([x_in, h])
        return layers.LeakyReLU(alpha=leak, name=nm + "lrelu_out")(out)

    enc_in    = Input(shape=(64,), dtype="int32", name="enc_in")
    token_emb = layers.Embedding(input_dim=vs, output_dim=de, name="token_emb")(enc_in)
    x         = layers.Reshape((8, 8, de), name="to_2d")(token_emb)

    if de != conv_filters:
        x = layers.Conv2D(conv_filters, 1, padding="same", use_bias=False,
                          kernel_initializer=HE, name="conv_proj")(x)
        x = layers.LayerNormalization(axis=-1, name="ln_proj")(x)
        x = layers.LeakyReLU(alpha=0.01, name="lrelf_proj")(x)

    for i in range(conv_blocks):
        x = res_block_ln(x, conv_filters, leak=0.01, nm=f"conv{i}_")

    seq = layers.Reshape((64, conv_filters), name="to_seq")(x)
    pos = layers.Embedding(64, conv_filters, name="pos_emb")(
        layers.Lambda(lambda _: tf.range(64), name="pos_idx")(enc_in)
    )
    pos = layers.Lambda(lambda z: tf.expand_dims(z, 0), name="pos_bcast")(pos)
    seq = layers.Add(name="add_pos")([seq, pos])

    for i in range(transformer_layers):
        nm   = f"t{i}_"
        ln1  = layers.LayerNormalization(axis=-1, name=nm + "ln1")(seq)
        attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=conv_filters // num_heads, name=nm + "mha"
        )(ln1, ln1)
        attn = layers.Dropout(dropout, name=nm + "attn_drop")(attn)
        seq  = layers.Add(name=nm + "attn_add")([seq, attn])
        ln2  = layers.LayerNormalization(axis=-1, name=nm + "ln2")(seq)
        ff   = layers.Dense(ff_dim, activation="gelu", kernel_initializer=HE,
                            name=nm + "ff1")(ln2)
        ff   = layers.Dropout(dropout, name=nm + "ff_drop")(ff)
        ff   = layers.Dense(conv_filters, name=nm + "ff2")(ff)
        seq  = layers.Add(name=nm + "ff_add")([seq, ff])

    x = layers.Reshape((8, 8, conv_filters), name="from_seq")(seq)
    x = layers.Conv2D(conv_filters, 1, padding="same", use_bias=False,
                      name="conv_mix_1x1")(x)
    x = layers.LayerNormalization(axis=-1, name="ln_mix")(x)
    x = layers.LeakyReLU(alpha=0.01, name="lrelu_mix")(x)

    policy_map    = layers.Conv2D(67, 1, padding="same", activation=None,
                                  name="policy_conv_67")(x)
    policy_logits = layers.Reshape((64 * 67,), name="policy_logits")(
        layers.Reshape((64, 67), name="to_64_67")(policy_map)
    )

    v = layers.Conv2D(64, 3, padding="same", use_bias=False,
                      kernel_initializer=HE, name="v_conv1")(x)
    v = layers.LayerNormalization(epsilon=1e-5, name="v_ln1")(v)
    v = layers.LeakyReLU(alpha=0.01, name="v_lrelu1")(v)
    v = layers.GlobalAveragePooling2D(name="v_gap")(v)
    v = layers.Dense(256, activation="relu", name="v_fc1")(v)
    v = layers.Dropout(dropout, name="v_fc_drop")(v)
    v = layers.Dense(128, activation="relu", name="v_fc2")(v)
    value_out = layers.Dense(3, dtype="float32", name="value_out")(v)

    model = Model(inputs=[enc_in], outputs=[policy_logits, value_out], name=name)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    opt = mixed_precision.LossScaleOptimizer(opt)
    loss_dict    = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out":     tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    }
    loss_weights = {"policy_logits": 1.0, "value_out": 1.0}
    model.compile(optimizer=opt, loss=loss_dict, loss_weights=loss_weights)
    model._default_opt       = opt
    model._default_loss_dict = loss_dict
    return model, opt, loss_weights, loss_dict


# ---------------------------------------------------------------------------
# TF utility functions
# ---------------------------------------------------------------------------

def load_model(model_loc: str):
    import tensorflow as tf
    return tf.keras.models.load_model(model_loc, compile=False)


def save_model(model, model_loc: str) -> None:
    model.save(model_loc, save_format="h5")


def set_loss_weights(model, loss_weights: dict, steps_per_execution: int = 1,
                     jit: bool = False):
    import tensorflow as tf
    opt_candidate = model._default_opt
    if isinstance(opt_candidate, tf.keras.mixed_precision.LossScaleOptimizer):
        base_opt = opt_candidate.inner_optimizer
    else:
        base_opt = opt_candidate
    opt_cfg = tf.keras.optimizers.serialize(base_opt)
    opt     = tf.keras.optimizers.deserialize(opt_cfg)
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    model.compile(
        optimizer=opt,
        loss=model._default_loss_dict,
        loss_weights=loss_weights,
        steps_per_execution=steps_per_execution,
        jit_compile=jit,
    )
    model.loss_weights  = loss_weights
    model._default_opt  = opt
    return model


def make_conv_infer(model, max_bs=1024, vscale=None):
    """XLA-compiled TF inference wrapper."""
    import numpy as np
    import tensorflow as tf

    @tf.function(
        input_signature=[tf.TensorSpec([None, 64], tf.int32)],
        experimental_compile=True,
    )
    def graph(enc):
        logits, value = model(enc, training=False)
        logits = tf.cast(tf.reshape(logits, [tf.shape(logits)[0], -1]), tf.float32)
        wdl    = tf.nn.softmax(tf.cast(value, tf.float32), axis=-1)
        return logits, wdl

    def base_fwd(pair):
        if not isinstance(pair, (list, tuple)):
            raise ValueError("pass (enc_np, ...) tuple")
        logits_tf, val_tf = graph(tf.convert_to_tensor(pair[0], dtype=tf.int32))
        return logits_tf.numpy(), val_tf.numpy()

    if max_bs is None:
        return base_fwd

    def fwd(pair):
        enc_np = pair[0]
        B = int(enc_np.shape[0])
        if B <= max_bs:
            return base_fwd(pair)
        parts = None
        i = 0
        while i < B:
            j = min(i + max_bs, B)
            p_logits, p_val = base_fwd((enc_np[i:j],))
            if parts is None:
                parts = [p_logits, p_val]
            else:
                parts[0] = np.concatenate([parts[0], p_logits], axis=0)
                parts[1] = np.concatenate([parts[1], p_val], axis=0)
            i = j
        return parts

    return fwd


def reclaim_vram(mb):
    import tensorflow as tf
    bytes_target = mb * 1024 * 1024
    n = max(1, bytes_target // 4)
    with tf.device("/GPU:0"):
        x = tf.ones([n], dtype=tf.float32)
        y = tf.reduce_sum(x)
    _ = y.numpy()
    return True
