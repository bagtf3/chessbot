"""
transfer_weights_fixedres.py

Transfers weights from cfiw_wdl_run3_model.h5 (broken residual architecture)
into the fixed pre-norm residual architecture, saving as cfiw_wdl_run3_model_fixedres.h5.

Changes between old and new conv blocks:
  OLD: r=x; h=c1(x); h=lr1(h); h=c2(h); h=ln2(h); x=LeakyReLU(r+h)  [b{i}_out]
  NEW: r=x; h=ln2(x); h=c1(h); h=lr1(h); h=c2(h); x=r+h

Layer weight mapping:
  b{i}_ln2  gamma/beta: same name, same shape (pre-norm now, was post-norm)
  b{i}_c1   kernel: same name, same shape
  b{i}_c2   kernel: same name, same shape
  b{i}_out  LeakyReLU: no weights, layer removed
  All other layers: identical names and shapes
"""

import os
import sys
import numpy as np

SRC_MODEL = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\cfiw_wdl_run3\cfiw_wdl_run3_model.h5"
RUN_DIR = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\val_test"
DST_MODEL = os.path.join(RUN_DIR, "val_test_model.h5")

sys.path.insert(0, r"C:\Users\Bryan\repos\chessbot\src")
sys.path.insert(0, r"C:\Users\Bryan\repos\chessbot\scripts")


def build_fixed_model():
    import tensorflow as tf
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    from tensorflow.keras import layers, Input, Model

    SEQ_LEN = 64
    VOCAB_SIZE = 21
    de = 256
    cf = 256
    cb = 10
    nh = 8
    dr = 0.025

    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x = layers.Embedding(VOCAB_SIZE, de, name="token_emb")(inp)
    x = layers.Reshape((8, 8, de), name="to_2d")(x)

    if de != cf:
        x = layers.Conv2D(cf, 1, use_bias=False, padding="same", name="conv_proj")(x)
        x = layers.LayerNormalization(axis=-1, name="ln_proj")(x)
        x = layers.LeakyReLU(0.01, name="lrelu_proj")(x)

    pos = layers.Embedding(64, cf, name="pos_emb")(
        tf.keras.backend.arange(0, 64, dtype="int32")
    )
    pos = layers.Lambda(
        lambda t: tf.expand_dims(t, axis=0),
        name="pos_expand",
    )(pos)

    for i in range(cb):
        s = layers.Reshape((64, cf), name=f"b{i}_to_seq")(x)
        if i == 0:
            s = s + pos
        if i == 2:
            s = s + 0.1 * pos
        if i == 4:
            s = s + 0.05 * pos
        if i == 6:
            s = s + 0.025 * pos

        r = s
        s = layers.LayerNormalization(axis=-1, name=f"b{i}_ln1")(s)
        s = layers.MultiHeadAttention(
            num_heads=nh,
            key_dim=cf // nh,
            dropout=0.0,
            name=f"b{i}_mha",
        )(s, s)
        s = layers.Dropout(dr, name=f"b{i}_drop1")(s)
        s = s + r

        x = layers.Reshape((8, 8, cf), name=f"b{i}_to_2d")(s)
        r = x
        h = layers.LayerNormalization(axis=-1, name=f"b{i}_ln2")(x)
        h = layers.Conv2D(cf, 3, use_bias=False, padding="same", name=f"b{i}_c1")(h)
        h = layers.LeakyReLU(0.01, name=f"b{i}_lr1")(h)
        h = layers.Conv2D(cf, 3, use_bias=False, padding="same", name=f"b{i}_c2")(h)
        x = r + h

    # value head
    v = layers.Conv2D(cf, 1, use_bias=False, padding="same", name="value_mix")(x)
    v = layers.LayerNormalization(axis=-1, name="value_mix_ln")(v)
    v = layers.LeakyReLU(0.01, name="value_mix_act")(v)

    # attn pool value head
    v_seq = layers.Reshape((64, cf), name="v_seq")(v)
    v_seq = layers.LayerNormalization(axis=-1, name="v_ln")(v_seq)
    attn_w = layers.Dense(1, name="attn_w")(v_seq)
    attn_w = layers.Softmax(axis=1, name="attn_softmax")(attn_w)
    attn_wT = layers.Permute((2, 1), name="attn_w_T")(attn_w)
    vp = layers.Dot(axes=[2, 1], name="v_pool")([attn_wT, v_seq])
    vp = layers.Reshape((cf,), name="v_squeeze")(vp)
    vp = layers.Dense(256, activation="relu", name="v_fc1")(vp)
    vp = layers.Dense(128, activation="relu", name="v_fc2")(vp)
    value = layers.Dense(3, dtype="float32", name="value_out")(vp)

    x_seq = layers.Reshape((64, cf), name="pol_to_seq")(x)

    from_h = layers.Dense(384, activation="gelu", name="pol_from_fc1")(x_seq)
    from_h = layers.Dense(256, activation="gelu", name="pol_from_fc2")(from_h)
    from_vec = layers.Dense(128, name="pol_from_proj")(from_h)

    to_h = layers.Dense(384, activation="gelu", name="pol_to_fc1")(x_seq)
    to_h = layers.Dense(256, activation="gelu", name="pol_to_fc2")(to_h)
    to_vec = layers.Dense(128, name="pol_to_proj")(to_h)

    normal_logits = layers.Dot(axes=[2, 2], name="pol_qk_dot")([from_vec, to_vec])
    normal_logits = layers.Lambda(
        lambda t: t / (128 ** 0.5), name="pol_qk_scale")(normal_logits)
    normal_logits = layers.Reshape((64 * 64,), name="pol_normal_flat")(normal_logits)

    p = layers.Conv2D(64, 1, use_bias=False, padding="same", name="pol_promo_mix")(x)
    p = layers.LayerNormalization(axis=-1, name="pol_promo_mix_ln")(p)
    p = layers.LeakyReLU(0.01, name="pol_promo_mix_act")(p)
    p = layers.Conv2D(64, 3, use_bias=False, padding="same", name="pol_promo_c1")(p)
    p = layers.LayerNormalization(axis=-1, name="pol_promo_ln")(p)
    p = layers.LeakyReLU(0.01, name="pol_promo_act")(p)
    promo_logits = layers.Conv2D(3, 1, padding="same", name="pol_promo_conv")(p)
    promo_logits = layers.Reshape((8 * 8 * 3,), name="pol_promo_flat")(promo_logits)

    policy = layers.Concatenate(axis=1, name="policy_logits")([normal_logits, promo_logits])

    model = Model(inp, [policy, value])
    print(f"  fixed model params: {model.count_params():,}")
    return model


def transfer_weights(src_model, dst_model):
    src_by_name = {layer.name: layer for layer in src_model.layers}
    dst_by_name = {layer.name: layer for layer in dst_model.layers}

    # b{i}_out layers exist in src but not dst (LeakyReLU after add - no weights anyway)
    skipped_no_dst = []
    skipped_no_src = []
    skipped_shape = []
    transferred = []

    for name, dst_layer in dst_by_name.items():
        dst_weights = dst_layer.get_weights()
        if not dst_weights:
            continue

        if name not in src_by_name:
            skipped_no_src.append(name)
            continue

        src_layer = src_by_name[name]
        src_weights = src_layer.get_weights()

        if len(src_weights) != len(dst_weights):
            skipped_shape.append((name, "weight count mismatch", len(src_weights), len(dst_weights)))
            continue

        shape_ok = all(s.shape == d.shape for s, d in zip(src_weights, dst_weights))
        if not shape_ok:
            skipped_shape.append((name, "shape mismatch",
                                   [w.shape for w in src_weights],
                                   [w.shape for w in dst_weights]))
            continue

        dst_layer.set_weights(src_weights)
        transferred.append(name)

    for name, src_layer in src_by_name.items():
        if name not in dst_by_name and src_layer.get_weights():
            skipped_no_dst.append(name)

    print(f"\n  transferred: {len(transferred)} layers")
    if skipped_no_src:
        print(f"  skipped (no src match): {skipped_no_src}")
    if skipped_no_dst:
        print(f"  src layers with weights not in dst: {skipped_no_dst}")
    if skipped_shape:
        for entry in skipped_shape:
            print(f"  shape mismatch: {entry}")


def main():
    import tensorflow as tf
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    print(f"loading source model: {SRC_MODEL}")
    src_model = tf.keras.models.load_model(SRC_MODEL, compile=False)
    print(f"  source params: {src_model.count_params():,}")

    print("building fixed architecture...")
    dst_model = build_fixed_model()

    print("transferring weights...")
    transfer_weights(src_model, dst_model)

    print(f"\nsaving -> {DST_MODEL}")
    dst_model.save(DST_MODEL)
    print("saved")

    import shutil
    ckpts_dir = os.path.join(RUN_DIR, "train_ckpts")
    if os.path.isdir(ckpts_dir):
        shutil.rmtree(ckpts_dir)
        print(f"deleted {ckpts_dir}")
    else:
        print("no train_ckpts dir found")
    print("done")


if __name__ == "__main__":
    main()
