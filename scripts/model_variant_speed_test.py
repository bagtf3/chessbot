#!/usr/bin/env python3
"""model_variant_speed_test.py

Speed test for chess model architecture variants.

Variants:
  16m-pure-conv              Pure conv ResNet, no context (baseline)
  16m-conformer              Conv front end + MHA back end (current production)
  16m-conformer-interweaved  Per-block MHA → ConvRes (MHA + conv interleaved each block)
  16m-film                   2×bare-MHA context encoder → FiLM scale/shift per conv block
  16m-concat-fusion          Global context concatenated mid-block, mixed by 3×3 conv (leaky relu on g_x before cat)
  16m-gated-ctx              Conv resblocks + x-gated additive context residual
  16m-transformer            Pure transformer, d_embed=384, 8 layers, 8 heads (~16M params)

Backends tested:
  TF + XLA     (fp16)
  PT eager     (fp16)
  PT compiled  (fp16, inductor → cudagraphs fallback)
  ORT + TRT    (fp16, engine cached per architecture)

Flags:
  --dry-run      Build all models, print param counts, skip speed tests and saving
  --no-save      Run full speed test but do not save TF models or CSV
  --skip-trt     Skip ORT+TensorRT (fastest way to get TF+PT numbers)
  --batch-sizes  Space-separated list (default: 1 4 8 16 32 64 128 256)

Usage:
  python model_variant_speed_test.py
  python model_variant_speed_test.py --dry-run
  python model_variant_speed_test.py --skip-trt
  python model_variant_speed_test.py --no-save
  python model_variant_speed_test.py --batch-sizes 1 32 64 256
"""

import os
import gc
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# XLA libdevice fix — must precede TF import
# ---------------------------------------------------------------------------

def fix_xla_libdevice():
    import shutil, tempfile
    cuda_dir = os.getenv("CUDA_DIR", "")
    if not cuda_dir:
        return
    src = os.path.join(cuda_dir, "nvvm", "libdevice", "libdevice.10.bc")
    if not os.path.exists(src):
        return
    dst_dir = os.path.join(tempfile.gettempdir(), "xla_cuda", "nvvm", "libdevice")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "libdevice.10.bc")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    os.environ["XLA_FLAGS"] = (
        f"--xla_gpu_cuda_data_dir="
        f"{os.path.join(tempfile.gettempdir(), 'xla_cuda')}"
    )

fix_xla_libdevice()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

VOCAB_SIZE  = 21
SEQ_LEN     = 64
N_WARMUP    = 50
N_ITERS     = 100
MODEL_DIR   = os.getenv("MODEL_DIR", "")
TRT_CACHE   = os.getenv("TRT_ENGINE_CACHE_DIR",
                        os.path.join(os.path.expanduser("~"), ".cache", "xerces_variant_trt"))

DEFAULT_BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256]

# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS = {
    "16m-pure-conv": dict(
        pure_conv=True,
        d_embed=288, conv_filters=288, conv_blocks=11, dropout=0.05,
    ),
    "16m-conformer": dict(
        conformer=True,
        conv_filters=256, conv_blocks=10,
        transformer_layers=5, num_heads=8, ff_dim=1024, dropout=0.05,
    ),
    "16m-film": dict(
        film=True,
        d_embed=256, conv_filters=256, conv_blocks=12,
        mha_heads=8, mha_layers=2, dropout=0.05,
    ),
    "16m-concat-fusion": dict(
        concat_fusion=True,
        d_embed=256, conv_filters=256, conv_blocks=12,
        mha_heads=8, mha_layers=2, dropout=0.05,
    ),
    "16m-gated-ctx": dict(
        gated_ctx=True,
        d_embed=256, conv_filters=256, conv_blocks=12,
        mha_heads=8, mha_layers=2, dropout=0.05,
    ),
    "16m-conformer-interweaved": dict(
        conformer_interweaved=True,
        d_embed=256,
        conv_filters=256,
        conv_blocks=10,
        num_heads=8,
        ff_dim=1024,
        dropout=0.05,
    ),
    "16m-transformer": dict(
        transformer=True,
        d_embed=512,
        transformer_layers=7,
        num_heads=16,
        ff_dim=1024,
        dropout=0.05,
    ),
    "16m-transformer-branched": dict(
        transformer_branched=True,
        d_embed=512,
        transformer_layers=4,
        num_heads=16,
        ff_dim=1024,
        dropout=0.05,
    ),
}

VARIANT_ORDER = list(VARIANTS.keys())

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run",    action="store_true",
                   help="Build models + print param counts only, no speed test, no saving")
    p.add_argument("--no-save",    action="store_true",
                   help="Run full speed test but skip saving TF models and CSV")
    p.add_argument("--skip-trt",   action="store_true",
                   help="Skip ORT+TensorRT benchmark (TF+XLA and PT still run)")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES,
                   metavar="B", help="Batch sizes to benchmark")
    p.add_argument("--model-dir",  default=MODEL_DIR,
                   help="Directory to save TF .h5 models and CSV results")
    p.add_argument("--variants",   nargs="+", choices=VARIANT_ORDER, default=None,
                   help="Run only specific variants (default: all 7)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_tokens(batch_size):
    return np.random.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), dtype=np.int32)


def describe_variant(name, cfg):
    C = cfg.get("conv_filters", cfg.get("d_embed", 256))
    print(f"\n  {'─'*60}")
    print(f"  Variant      : {name}")
    if cfg.get("pure_conv"):
        print(f"  Architecture : pure conv ResNet — no global context")
        print(f"  blocks       : {cfg['conv_blocks']} × [2×Conv({C},3,no bias) + LN]")
    elif cfg.get("conformer"):
        print(f"  Architecture : conformer — conv front end, MHA backend")
        print(f"  conv_blocks  : {cfg['conv_blocks']}  trans_layers: {cfg['transformer_layers']}")
        print(f"  num_heads    : {cfg['num_heads']}  ff_dim: {cfg['ff_dim']}")
    elif cfg.get("conformer_interweaved"):
        print("  Architecture : interweaved conformer — per-block MHA + ConvRes")
        print(f"  blocks       : {cfg['conv_blocks']} × [MHA(heads={cfg['num_heads']})"
            f" + Conv({C},3) + Conv({C},3)]")
    elif cfg.get("transformer"):
        print(f"  Architecture : pure transformer")
        print(f"  layers       : {cfg['transformer_layers']}  heads: {cfg['num_heads']}  ff_dim: {cfg['ff_dim']}")
    elif cfg.get("film"):
        print(f"  Architecture : FiLM — static context → affine scale/shift per block")
        print(f"  ctx encoder  : {cfg['mha_layers']} × bare-MHA(heads={cfg['mha_heads']}, d={C})")
        print(f"  conv blocks  : {cfg['conv_blocks']} × [Conv({C},3)×2 + LN + FiLM(γ,β)]")
    elif cfg.get("concat_fusion"):
        print(f"  Architecture : Concat-Fusion — g injected mid-block via concat + 3×3")
        print(f"  ctx encoder  : {cfg['mha_layers']} × bare-MHA(heads={cfg['mha_heads']}, d={C})")
        print(f"  conv blocks  : {cfg['conv_blocks']} × [Conv({C},3) → cat(h,gx:64ch) → Conv({C},3 on 320)]")
    elif cfg.get("gated_ctx"):
        print(f"  Architecture : Gated-Ctx — x-sigmoid gate controls context absorption")
        print(f"  ctx encoder  : {cfg['mha_layers']} × bare-MHA(heads={cfg['mha_heads']}, d={C})")
        print(f"  conv blocks  : {cfg['conv_blocks']} × [ConvResBlock + sigmoid(x)·ctx_proj(g)]")
    print(f"  value head   : attention pooling -> Dense(256) -> Dense(128) -> WDL [3] (raw logits)")
    print(f"  {'─'*60}")


def make_cache_key(name, cfg):
    if cfg.get("pure_conv"):     return f"pure-conv-cb{cfg['conv_blocks']}"
    if cfg.get("conformer"):     return f"conformer-cb{cfg['conv_blocks']}-tl{cfg['transformer_layers']}"
    if cfg.get("transformer"):   return f"transformer-tl{cfg['transformer_layers']}-nh{cfg['num_heads']}"
    if cfg.get("film"):          return f"film-cb{cfg['conv_blocks']}-nh{cfg['mha_heads']}"
    if cfg.get("concat_fusion"): return f"concat-fusion-cb{cfg['conv_blocks']}"
    if cfg.get("gated_ctx"):     return f"gated-ctx-cb{cfg['conv_blocks']}"
    if cfg.get("conformer_interweaved"):
        return f"conformer-interweaved-cb{cfg['conv_blocks']}-nh{cfg['num_heads']}"
    return name.replace(" ", "-")


def save_csv_incremental(csv_rows, model_dir):
    """Save accumulated CSV rows to disk (called after each variant)."""
    try:
        import pandas as pd
        os.makedirs(model_dir, exist_ok=True)
        csv_path = os.path.join(model_dir, "variant_speed_results.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"  CSV saved → {csv_path}")
    except Exception as e:
        print(f"  [WARN] CSV save failed: {e}")

# ---------------------------------------------------------------------------
# Shared TF helpers
# ---------------------------------------------------------------------------

def tf_compile(model):
    import tensorflow as tf
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    }
    model.compile(optimizer=opt, loss=loss_dict,
                  loss_weights={"policy_logits": 1.0, "value_out": 1.0})
    model._default_opt       = opt
    model._default_loss_dict = loss_dict
    return model


def tf_attn_pool_value_head(x, C, layers, activation="gelu"):
    """Attention pooling value head. x: (B,8,8,C) or (B,64,C)."""
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
    """Relational policy head. x: (B,8,8,C) or (B,64,C).
    Returns policy_logits (B,4288): 4096 normal moves + 192 underpromotions."""
    if len(x.shape) == 3:
        x_seq = x
        x_2d  = layers.Reshape((8, 8, C), name="pol_x_to_2d")(x)
    else:
        x_seq = layers.Reshape((64, C), name="pol_to_seq")(x)
        x_2d  = x

    from_h   = layers.Dense(256, activation="gelu", name="pol_from_fc1")(x_seq)
    from_h   = layers.LayerNormalization(name="pol_from_ln1")(from_h)
    from_vec = layers.Dense(256, activation="gelu", name="pol_from_fc2")(from_h)
    from_vec = layers.LayerNormalization(name="pol_from_ln2")(from_vec)

    to_h     = layers.Dense(256, activation="gelu", name="pol_to_fc1")(x_seq)
    to_h     = layers.LayerNormalization(name="pol_to_ln1")(to_h)
    to_vec   = layers.Dense(256)(to_h)
    to_vec   = layers.LayerNormalization(name="pol_to_ln2")(to_vec)

    normal_logits = layers.Dot(axes=[2, 2], name="pol_qk_dot")([from_vec, to_vec])
    normal_logits = layers.Reshape((64 * 64,), name="pol_normal_flat")(normal_logits)

    p      = layers.Conv2D(64, 1, use_bias=False, padding="same", name="pol_promo_mix")(x_2d)
    p      = layers.LayerNormalization(axis=-1, name="pol_promo_mix_ln")(p)
    p      = layers.LeakyReLU(0.02, name="pol_promo_mix_act")(p)
    p_skip = p

    p      = layers.Conv2D(64, 3, use_bias=False, padding="same", name="pol_promo_c1")(p)
    p      = layers.LayerNormalization(axis=-1, name="pol_promo_ln")(p)
    p      = layers.LeakyReLU(0.02, name="pol_promo_act")(p)
    p      = layers.Add(name="pol_promo_skip")([p, p_skip])
    
    promo_logits = layers.Conv2D(3, 1, padding="same", name="pol_promo_conv")(p)
    promo_logits = layers.Reshape((8 * 8 * 3,), name="pol_promo_flat")(promo_logits)

    return layers.Concatenate(axis=1, name="policy_logits")([normal_logits, promo_logits])


def tf_context_encoder(proj, C, nh, nl, layers, tf):
    """Shared context encoder: positional embedding + nl × prenorm MHA+FF.
    Returns g_2d (B,8,8,C). The conv stack starts from proj (no pos)."""
    pos = layers.Embedding(64, C, name="pos_emb")(
        tf.keras.backend.arange(0, 64, dtype="int32")
    )
    g = proj + pos
    for i in range(nl):
        r = g
        g = layers.LayerNormalization(axis=-1, name=f"ctx{i}_ln")(g)
        g = layers.MultiHeadAttention(
            num_heads=nh, key_dim=C // nh, dropout=0.0, name=f"ctx{i}_mha"
        )(g, g)
        g = g + r
        r = g
        g = layers.LayerNormalization(axis=-1, name=f"ctx{i}_ff_ln")(g)
        g = layers.Dense(C, activation="gelu", name=f"ctx{i}_ff")(g)
        g = g + r
    g = layers.LayerNormalization(axis=-1, name="ctx_out_ln")(g)
    return layers.Reshape((8, 8, C), name="g_to_2d")(g)

# ---------------------------------------------------------------------------
# TF builders
# ---------------------------------------------------------------------------

def build_tf_pure_conv(cfg):
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    C = cfg["conv_filters"]; cb = cfg["conv_blocks"]
    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x   = layers.Embedding(VOCAB_SIZE, C, name="token_emb")(inp)
    x   = layers.Reshape((8, 8, C), name="x_to_2d")(x)

    for i in range(cb):
        r = x
        h = layers.Conv2D(C, 3, use_bias=False, padding="same", name=f"cv{i}_c1")(x)
        h = layers.LeakyReLU(0.01, name=f"cv{i}_lr1")(h)
        h = layers.Conv2D(C, 3, use_bias=False, padding="same", name=f"cv{i}_c2")(h)
        h = layers.LayerNormalization(axis=-1, name=f"cv{i}_ln")(h)
        x = layers.LeakyReLU(0.01, name=f"cv{i}_out")(r + h)

    policy = tf_relational_policy_head(x, C, layers)
    value  = tf_attn_pool_value_head(x, C, layers)
    model  = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


def build_tf_conformer(cfg):
    """cb ConvResBlocks → tl Transformer layers → 1×1 conv mix.
    Single cf dimension throughout. One pos embedding, injected before
    conv blocks and again before the transformer."""
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    cf = cfg["conv_filters"]
    cb = cfg["conv_blocks"]
    tl = cfg["transformer_layers"]
    nh = cfg["num_heads"]
    fd = cfg["ff_dim"]
    dr = cfg["dropout"]

    sq = tf.keras.backend.arange(0, 64, dtype="int32")

    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x   = layers.Embedding(VOCAB_SIZE, cf, name="token_emb")(inp)

    # single pos embedding — reused at both injection points
    pos_emb = layers.Embedding(64, cf, name="pos_emb")
    pos  = layers.Lambda(lambda t: tf.expand_dims(t, axis=0), name="pos_expand")(pos_emb(sq))

    x = x + pos                                            # inject before conv
    x = layers.Reshape((8, 8, cf), name="to_2d")(x)

    for i in range(cb):
        h = layers.Conv2D(cf, 3, use_bias=False, padding="same", name=f"c{i}_c1")(x)
        h = layers.LeakyReLU(0.01, name=f"c{i}_lr1")(h)
        h = layers.Conv2D(cf, 3, use_bias=False, padding="same", name=f"c{i}_c2")(h)
        h = layers.LayerNormalization(axis=-1, name=f"c{i}_ln")(h)
        x = layers.LeakyReLU(0.01, name=f"c{i}_out")(x + h)

    x = layers.Reshape((64, cf), name="to_seq")(x)
    x = x + pos                                            # inject before transformer

    for i in range(tl):
        r = x
        x = layers.LayerNormalization(axis=-1, name=f"t{i}_ln1")(x)
        x = layers.MultiHeadAttention(
            num_heads=nh,
            key_dim=cf // nh,
            dropout=0.0,
            name=f"t{i}_mha",
        )(x, x)
        x = layers.Dropout(dr, name=f"t{i}_drop1")(x)
        x = x + r

        r = x
        x = layers.LayerNormalization(axis=-1, name=f"t{i}_ln2")(x)
        x = layers.Dense(fd, activation="gelu", name=f"t{i}_ff1")(x)
        x = layers.Dropout(dr, name=f"t{i}_drop2")(x)
        x = layers.Dense(cf, name=f"t{i}_ff2")(x)
        x = x + r

    x = layers.Reshape((8, 8, cf), name="from_seq")(x)
    x = layers.Conv2D(
        cf,
        1,
        use_bias=False,
        padding="same",
        name="conv_mix",
    )(x)
    x = layers.LayerNormalization(axis=-1, name="ln_mix")(x)
    x = layers.LeakyReLU(0.01, name="lrelu_mix")(x)

    policy = tf_relational_policy_head(x, cf, layers)
    value  = tf_attn_pool_value_head(x, cf, layers)

    model = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


def build_tf_conformer_interweaved(cfg):
    """Interweaved conformer: 10 x [MHA -> Conv -> Conv]."""
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    de = cfg["d_embed"]
    cf = cfg["conv_filters"]
    cb = cfg["conv_blocks"]
    nh = cfg["num_heads"]
    dr = cfg["dropout"]

    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x = layers.Embedding(VOCAB_SIZE, de, name="token_emb")(inp)
    x = layers.Reshape((8, 8, de), name="to_2d")(x)

    if de != cf:
        x = layers.Conv2D(cf, 1, use_bias=False, padding="same", name="conv_proj")(x)
        x = layers.LayerNormalization(axis=-1, name="ln_proj")(x)
        x = layers.LeakyReLU(0.02, name="lrelu_proj")(x)

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
        h = layers.LeakyReLU(0.02, name=f"b{i}_lr1")(h)
        h = layers.Conv2D(cf, 3, use_bias=False, padding="same", name=f"b{i}_c2")(h)
        x = r + h
    
    # VALUE HEAD
    v = layers.Conv2D(cf, 1, use_bias=False, padding="same", name="value_mix")(x)
    v = layers.LayerNormalization(axis=-1, name="value_mix_ln")(v)
    v = layers.LeakyReLU(0.02, name="value_mix_act")(v)

    value = tf_attn_pool_value_head(v, cf, layers)

    policy = tf_relational_policy_head(x, cf, layers)

    model = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)

def build_tf_2(cfg):
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    de = cfg["d_embed"]
    tl = cfg["transformer_layers"]
    nh = cfg["num_heads"]
    fd = cfg["ff_dim"]
    dr = cfg["dropout"]

    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x = layers.Embedding(VOCAB_SIZE, de, name="token_emb")(inp)

    pos = layers.Embedding(64, de, name="pos_emb")(
        tf.keras.backend.arange(0, 64, dtype="int32")
    )
    pos = layers.Lambda(
        lambda t: tf.expand_dims(t, axis=0),
        name="pos_expand",
    )(pos)

    x = x + pos

    for i in range(tl):
        r = x
        x = layers.LayerNormalization(axis=-1, name=f"t{i}_ln1")(x)
        x = layers.MultiHeadAttention(
            num_heads=nh,
            key_dim=de // nh,
            dropout=0.0,
            name=f"t{i}_mha",
        )(x, x)
        x = layers.Dropout(dr, name=f"t{i}_drop1")(x)
        x = x + r

        r = x
        x = layers.LayerNormalization(axis=-1, name=f"t{i}_ln2")(x)
        x = layers.Dense(fd, activation="gelu", name=f"t{i}_ff1")(x)
        x = layers.Dropout(dr, name=f"t{i}_drop2")(x)
        x = layers.Dense(de, name=f"t{i}_ff2")(x)
        x = x + r

    # x is (B, 64, de) — reshape once, shared by both heads
    x_2d = layers.Reshape((8, 8, de), name="x_to_2d")(x)

    # value head (matching conformer-interweaved)
    v = layers.Conv2D(de, 1, use_bias=False, padding="same", name="value_mix")(x_2d)
    v = layers.LayerNormalization(axis=-1, name="value_mix_ln")(v)
    v = layers.LeakyReLU(0.02, name="value_mix_act")(v)
    value = tf_attn_pool_value_head(v, de, layers)

    policy = tf_relational_policy_head(x_2d, de, layers)

    model = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


def build_tf_film(cfg):
    """2×MHA context encoder → FiLM gammas/betas computed upfront → conv stack.
    Note: ff_dim key in preset is vestigial — context encoder has no FF layers."""
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    C = cfg["conv_filters"]; cb = cfg["conv_blocks"]
    nh = cfg["mha_heads"]; nl = cfg["mha_layers"]
    inp  = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    proj = layers.Embedding(VOCAB_SIZE, C, name="token_emb")(inp)
    g_2d = tf_context_encoder(proj, C, nh, nl, layers, tf)
    # all gammas and betas computed upfront from static g_2d
    gammas = [layers.Conv2D(C, 1, padding="same", name=f"film{i}_gamma",
                            kernel_initializer="zeros", bias_initializer="ones")(g_2d) for i in range(cb)]
    betas  = [layers.Conv2D(C, 1, padding="same", name=f"film{i}_beta",
                            kernel_initializer="zeros")(g_2d) for i in range(cb)]
    # conv stack starts from raw token proj (no pos, no ctx)
    x = layers.Reshape((8, 8, C), name="x_to_2d")(proj)
    for i in range(cb):
        r = x
        h = layers.Conv2D(C, 3, use_bias=False, padding="same", name=f"cv{i}_c1")(x)
        h = layers.LeakyReLU(0.01, name=f"cv{i}_lr1")(h)
        h = layers.Conv2D(C, 3, use_bias=False, padding="same", name=f"cv{i}_c2")(h)
        h = layers.LayerNormalization(axis=-1, name=f"cv{i}_ln")(h)
        h = h * gammas[i] + betas[i]   # FiLM: gamma unrestricted (not sigmoid)
        x = layers.LeakyReLU(0.01, name=f"cv{i}_out")(r + h)
    policy = tf_relational_policy_head(x, C, layers)
    value  = tf_attn_pool_value_head(x, C, layers)
    model  = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


def build_tf_concat_fusion(cfg):
    """Per-block: Conv(C,3) → concat with g_x(64ch) → Conv(C,3,bias=True) → LN."""
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    C = cfg["conv_filters"]; G = 64; cb = cfg["conv_blocks"]
    nh = cfg["mha_heads"]; nl = cfg["mha_layers"]
    inp  = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    proj = layers.Embedding(VOCAB_SIZE, C, name="token_emb")(inp)
    g_2d = tf_context_encoder(proj, C, nh, nl, layers, tf)
    x    = layers.Reshape((8, 8, C), name="x_to_2d")(proj)
    for i in range(cb):
        r   = x
        h   = layers.Conv2D(C, 3, use_bias=False, padding="same", name=f"cv{i}_c1")(x)
        h   = layers.LeakyReLU(0.01, name=f"cv{i}_lr1")(h)
        g_x = layers.Conv2D(G, 1, padding="same", name=f"cv{i}_gx")(g_2d)
        g_x = layers.LeakyReLU(0.01, name=f"cv{i}_gx_lr")(g_x)
        h   = layers.Concatenate(axis=-1, name=f"cv{i}_cat")([h, g_x])  # (B,8,8,C+G=320)
        h   = layers.Conv2D(C, 3, padding="same", name=f"cv{i}_c2")(h)  # bias=True (mixing conv)
        h   = layers.LayerNormalization(axis=-1, name=f"cv{i}_ln")(h)
        x   = layers.LeakyReLU(0.01, name=f"cv{i}_out")(r + h)
    policy = tf_relational_policy_head(x, C, layers)
    value  = tf_attn_pool_value_head(x, C, layers)
    model  = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


def build_tf_gated_ctx(cfg):
    """Standard ConvResBlock then sigmoid-gated context injection per block."""
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    C = cfg["conv_filters"]; cb = cfg["conv_blocks"]
    nh = cfg["mha_heads"]; nl = cfg["mha_layers"]
    inp  = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    proj = layers.Embedding(VOCAB_SIZE, C, name="token_emb")(inp)
    g_2d = tf_context_encoder(proj, C, nh, nl, layers, tf)
    x    = layers.Reshape((8, 8, C), name="x_to_2d")(proj)
    for i in range(cb):
        # standard conv resblock
        r = x
        h = layers.Conv2D(C, 3, use_bias=False, padding="same", name=f"cv{i}_c1")(x)
        h = layers.LeakyReLU(0.01, name=f"cv{i}_lr1")(h)
        h = layers.Conv2D(C, 3, use_bias=False, padding="same", name=f"cv{i}_c2")(h)
        h = layers.LayerNormalization(axis=-1, name=f"cv{i}_ln")(h)
        x = layers.LeakyReLU(0.01, name=f"cv{i}_out")(r + h)
        # gated context injection: gate depends on x (post-resblock)
        gate     = layers.Conv2D(C, 1, padding="same", name=f"cv{i}_gate")(x)
        gate     = layers.Activation("sigmoid", name=f"cv{i}_sig")(gate)
        ctx_proj = layers.Conv2D(C, 1, padding="same", name=f"cv{i}_ctx",
                                kernel_initializer="zeros")(g_2d)
        x        = layers.Add(name=f"cv{i}_gadd")([x, layers.Multiply(
                       name=f"cv{i}_gmul")([gate, ctx_proj])])
    policy = tf_relational_policy_head(x, C, layers)
    value  = tf_attn_pool_value_head(x, C, layers)
    model  = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


def build_tf_transformer(cfg):
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    de = cfg["d_embed"]
    tl = cfg["transformer_layers"]
    nh = cfg["num_heads"]
    fd = cfg["ff_dim"]
    dr = cfg["dropout"]

    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x = layers.Embedding(VOCAB_SIZE, de, name="token_emb")(inp)
    pos = layers.Embedding(64, de, name="pos_emb")(
        tf.keras.backend.arange(0, 64, dtype="int32")
    )
    pos = layers.Lambda(
        lambda t: tf.expand_dims(t, axis=0), name="pos_expand"
    )(pos)
    x = x + pos

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
    v = layers.Conv2D(de, 1, use_bias=False, padding="same", name="value_mix")(x_2d)
    v = layers.LayerNormalization(axis=-1, name="value_mix_ln")(v)
    v = layers.LeakyReLU(0.02, name="value_mix_act")(v)
    value  = tf_attn_pool_value_head(v, de, layers)
    policy = tf_relational_policy_head(x_2d, de, layers)

    model = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


def build_tf_transformer_branched(cfg):
    """7-layer shared trunk splits into 3 independent branch MHAs:
      - value branch:   1 MHA -> attn-pool value head
      - policy-from:    1 MHA -> de->384->256 projection -> from_vec (B,64,256)
      - policy-to:      1 MHA -> de->384->256 projection -> to_vec (B,64,256)
    Cross-bilinear dot(from_vec, to_vec) -> (B,4096) + promo conv -> (B,4288).
    """
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

    def prenorm_dense(x, final_dense, tag):
        r = x
        x = layers.LayerNormalization(axis=-1, name=f"{tag}_ln2")(x)
        x = layers.Dense(fd, activation="gelu", name=f"{tag}_ff1")(x)
        x = layers.Dropout(dr, name=f"{tag}_drop2")(x)
        x = layers.Dense(final_dense, name=f"{tag}_ff2")(x)
        return x + r if final_dense == de else x

    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x = layers.Embedding(VOCAB_SIZE, de, name="token_emb")(inp)
    pos = layers.Embedding(64, de, name="pos_emb")(
        tf.keras.backend.arange(0, 64, dtype="int32")
    )
    pos = layers.Lambda(lambda t: tf.expand_dims(t, axis=0), name="pos_expand")(pos)
    x = x + pos

    for i in range(tl):
        x = prenorm_mha(x, f"t{i}")
        x = prenorm_dense(x, de, f"t{i}")

    # 3 independent branch MHAs
    v_seq  = prenorm_mha(x, "bv")
    v_seq  = prenorm_dense(v_seq, de, "bv")

    pf_seq = x
    pt_seq = x
    for j in range(1):
        pf_seq = prenorm_mha(pf_seq, f"bf{j}")
        pf_seq = prenorm_dense(pf_seq, de, f"bf{j}")

        pt_seq = prenorm_mha(pt_seq, f"bt{j}")
        pt_seq = prenorm_dense(pt_seq, de, f"bt{j}")

    # value branch
    v_2d  = layers.Reshape((8, 8, de), name="v_to_2d")(v_seq)
    value = tf_attn_pool_value_head(v_2d, de, layers)

    # cross-bilinear policy: from-branch and to-branch project independently
    from_vec = layers.Dense(256, activation="gelu", name="pol_from_fc2")(pf_seq)
    from_vec = layers.LayerNormalization(name="pol_from_ln2")(from_vec)

    to_vec = layers.Dense(256, activation="gelu", name="pol_to_fc2")(pt_seq)
    to_vec = layers.LayerNormalization(name="pol_to_ln2")(to_vec)

    normal_logits = layers.Dot(axes=[2, 2], name="pol_qk_dot")([from_vec, to_vec])
    normal_logits = layers.Reshape((64 * 64,), name="pol_normal_flat")(normal_logits)

    # promo from shared trunk output
    x_2d   = layers.Reshape((8, 8, de), name="x_to_2d")(x)
    p      = layers.Conv2D(64, 1, use_bias=False, padding="same", name="pol_promo_mix")(x_2d)
    p      = layers.LayerNormalization(axis=-1, name="pol_promo_mix_ln")(p)
    p      = layers.LeakyReLU(0.02, name="pol_promo_mix_act")(p)
    p_skip = p
    p      = layers.Conv2D(64, 3, use_bias=False, padding="same", name="pol_promo_c1")(p)
    p      = layers.LayerNormalization(axis=-1, name="pol_promo_ln")(p)
    p      = layers.LeakyReLU(0.02, name="pol_promo_act")(p)
    p      = layers.Add(name="pol_promo_skip")([p, p_skip])
    promo_logits = layers.Conv2D(3, 1, padding="same", name="pol_promo_conv")(p)
    promo_logits = layers.Reshape((8 * 8 * 3,), name="pol_promo_flat")(promo_logits)

    policy = layers.Concatenate(axis=1, name="policy_logits")([normal_logits, promo_logits])

    model = Model(inp, [policy, value])
    print(f"  TF params: {model.count_params():,}")
    return tf_compile(model)


TF_BUILDERS = {
    "16m-pure-conv":               build_tf_pure_conv,
    "16m-conformer":               build_tf_conformer,
    "16m-conformer-interweaved":   build_tf_conformer_interweaved,
    "16m-film":                    build_tf_film,
    "16m-concat-fusion":           build_tf_concat_fusion,
    "16m-gated-ctx":               build_tf_gated_ctx,
    "16m-transformer":             build_tf_transformer,
    "16m-transformer-branched":    build_tf_transformer_branched,
}

# ---------------------------------------------------------------------------
# PyTorch shared helpers
# ---------------------------------------------------------------------------

def make_ln2d(ch):
    import torch, torch.nn as nn
    class Ln2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(ch))
            self.b = nn.Parameter(torch.zeros(ch))
        def forward(self, x):
            c = x - x.mean(1, keepdim=True)
            return c * torch.rsqrt((c*c).mean(1, keepdim=True) + 1e-5) \
                   * self.w.view(1,-1,1,1) + self.b.view(1,-1,1,1)
    return Ln2d()


def make_bare_attn(d, heads):
    import torch.nn as nn
    class BareAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln   = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        def forward(self, x):
            n = self.ln(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            return x + h
    return BareAttn()


def pt_attn_pool(x_seq, linear):
    import torch
    w = torch.softmax(linear(x_seq), dim=1)
    return (x_seq * w).sum(dim=1)


def make_pt_attn_pool_value_head(C):
    """Attn-pool value head. Accepts (B,C,8,8) or (B,64,C). Returns value_out (B,3)."""
    import torch.nn as nn, torch.nn.functional as F
    class AttnPoolValueHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln  = nn.LayerNorm(C)
            self.ap  = nn.Linear(C, 1)
            self.fc1 = nn.Linear(C, 256)
            self.fc2 = nn.Linear(256, 128)
            self.out = nn.Linear(128, 3)
        def forward(self, x):
            import torch
            if x.dim() == 4:
                B = x.shape[0]
                x = x.permute(0, 2, 3, 1).reshape(B, 64, C)
            x = self.ln(x)
            w = torch.softmax(self.ap(x), dim=1)
            v = (x * w).sum(dim=1)
            v = F.relu(self.fc1(v))
            v = F.relu(self.fc2(v))
            return self.out(v)
    return AttnPoolValueHead()


def make_pt_relational_policy_head(C):
    """Relational policy head. Accepts (B,C,8,8). Returns policy_logits (B,4288)."""
    import torch.nn as nn, torch.nn.functional as F
    class RelationalPolicyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.from_fc   = nn.Linear(C, 256)
            self.from_ln1  = nn.LayerNorm(256)
            self.from_fc2  = nn.Linear(256, 128)
            self.from_ln2  = nn.LayerNorm(128)
            self.to_fc     = nn.Linear(C, 256)
            self.to_ln1    = nn.LayerNorm(256)
            self.to_fc2    = nn.Linear(256, 128)
            self.to_ln2    = nn.LayerNorm(128)
            self.promo_mix = nn.Conv2d(C, 64, 1, bias=False)
            self.promo_mln = make_ln2d(64)
            self.promo_c1  = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.promo_ln  = make_ln2d(64)
            self.promo_out = nn.Conv2d(64, 3, 1)
        def forward(self, x):
            import torch
            B = x.shape[0]
            s        = x.permute(0, 2, 3, 1).reshape(B, 64, C)
            from_vec = self.from_ln2(F.gelu(self.from_fc2(self.from_ln1(F.gelu(self.from_fc(s))))))
            to_vec   = self.to_ln2(F.gelu(self.to_fc2(self.to_ln1(F.gelu(self.to_fc(s))))))
            normal   = torch.bmm(from_vec, to_vec.transpose(1, 2)).reshape(B, 64 * 64)
            p        = F.leaky_relu(self.promo_mln(self.promo_mix(x)), 0.02)
            p_skip   = p
            p        = F.leaky_relu(self.promo_ln(self.promo_c1(p)), 0.02)
            promo    = self.promo_out(p + p_skip).permute(0, 2, 3, 1).reshape(B, 8 * 8 * 3)
            return torch.cat([normal, promo], dim=1)
    return RelationalPolicyHead()

# ---------------------------------------------------------------------------
# PyTorch builders
# ---------------------------------------------------------------------------

def build_pt_pure_conv(cfg):
    import torch, torch.nn as nn, torch.nn.functional as F
    C = cfg["conv_filters"]; cb = cfg["conv_blocks"]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb          = nn.Embedding(VOCAB_SIZE, C)
            self.blks         = nn.ModuleList([self._blk() for _ in range(cb)])
            self.policy_head  = make_pt_relational_policy_head(C)
            self.value_head   = make_pt_attn_pool_value_head(C)

        def _blk(self):
            return nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1, bias=False), nn.LeakyReLU(0.01, True),
                nn.Conv2d(C, C, 3, padding=1, bias=False), make_ln2d(C),
            )

        def forward(self, t):
            B = t.shape[0]
            x = self.emb(t).reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
            for blk in self.blks:
                r = x; h = blk(x); x = F.leaky_relu(r + h, 0.01)
            return self.policy_head(x), self.value_head(x)

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_conformer(cfg):
    """cb ConvResBlocks → tl Transformer layers → 1×1 conv mix.
    Single cf dimension throughout. One pos embedding, injected before
    conv blocks and again before the transformer."""
    import torch, torch.nn as nn, torch.nn.functional as F
    cf = cfg["conv_filters"]; cb = cfg["conv_blocks"]
    tl = cfg["transformer_layers"]; nh = cfg["num_heads"]
    fd = cfg["ff_dim"]; dr = cfg["dropout"]

    class TxLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1   = nn.LayerNorm(cf)
            self.attn  = nn.MultiheadAttention(cf, nh, dropout=0.0, batch_first=True)
            self.drop1 = nn.Dropout(dr)
            self.ln2   = nn.LayerNorm(cf)
            self.ff1   = nn.Linear(cf, fd)
            self.drop2 = nn.Dropout(dr)
            self.ff2   = nn.Linear(fd, cf)

        def forward(self, x):
            n = self.ln1(x); h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop1(h)
            return x + self.ff2(self.drop2(F.gelu(self.ff1(self.ln2(x)))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb         = nn.Embedding(VOCAB_SIZE, cf)
            self.pos         = nn.Embedding(64, cf)
            self.cblks       = nn.ModuleList([self._cblk() for _ in range(cb)])
            self.txs         = nn.ModuleList([TxLayer() for _ in range(tl)])
            self.mix         = nn.Conv2d(cf, cf, 1, bias=False)
            self.lnm         = make_ln2d(cf)
            self.policy_head = make_pt_relational_policy_head(cf)
            self.value_head  = make_pt_attn_pool_value_head(cf)

        def _cblk(self):
            return nn.Sequential(
                nn.Conv2d(cf, cf, 3, padding=1, bias=False), nn.LeakyReLU(0.01, True),
                nn.Conv2d(cf, cf, 3, padding=1, bias=False), make_ln2d(cf),
            )

        def forward(self, t):
            B   = t.shape[0]
            pos = self.pos(torch.arange(64, device=t.device)).unsqueeze(0)  # (1,64,cf)
            x   = self.emb(t) + pos                                          # inject before conv
            x   = x.reshape(B, 8, 8, cf).permute(0, 3, 1, 2).contiguous()
            for blk in self.cblks:
                r = x; x = F.leaky_relu(r + blk(x), 0.01)
            x = x.permute(0, 2, 3, 1).reshape(B, 64, cf) + pos             # inject before transformer
            for tx in self.txs:
                x = tx(x)
            x = x.reshape(B, 8, 8, cf).permute(0, 3, 1, 2).contiguous()
            x = F.leaky_relu(self.lnm(self.mix(x)), 0.01)
            return self.policy_head(x), self.value_head(x)

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_conformer_interweaved(cfg):
    """Interweaved conformer: 10 x [MHA -> Conv -> Conv].
    Op-for-op identical to build_tf_conformer_interweaved:
      - graduated pos injection: block 0=1.0, 2=0.1, 4=0.05, 6=0.025, others=none
      - LayerNorm before attention pooling in value head
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    de = cfg["d_embed"]
    cf = cfg["conv_filters"]
    cb = cfg["conv_blocks"]
    nh = cfg["num_heads"]
    dr = cfg["dropout"]

    # Graduated pos injection scales matching TF: {block_idx: scale}
    _POS_SCALES = {0: 1.0, 2: 0.1, 4: 0.05, 6: 0.025}

    class Block(nn.Module):
        def __init__(self, pos_scale):
            super().__init__()
            self.pos_scale = pos_scale
            self.ln1 = nn.LayerNorm(cf)
            self.attn = nn.MultiheadAttention(
                cf,
                nh,
                dropout=0.0,
                batch_first=True,
            )
            self.drop1 = nn.Dropout(dr)
            self.c1 = nn.Conv2d(cf, cf, 3, padding=1, bias=False)
            self.lr1 = nn.LeakyReLU(0.01, True)
            self.c2 = nn.Conv2d(cf, cf, 3, padding=1, bias=False)
            self.ln2 = make_ln2d(cf)

        def forward(self, x, pos):
            b = x.shape[0]

            s = x.permute(0, 2, 3, 1).reshape(b, 64, cf)
            if self.pos_scale != 0.0:
                s = s + self.pos_scale * pos

            r = s
            n = self.ln1(s)
            h, _ = self.attn(n, n, n, need_weights=False)
            s = r + self.drop1(h)

            x = s.reshape(b, 8, 8, cf).permute(0, 3, 1, 2).contiguous()

            r = x
            h = self.lr1(self.c1(x))
            h = self.ln2(self.c2(h))
            x = F.leaky_relu(r + h, 0.01)

            return x

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB_SIZE, de)
            self.proj = nn.Conv2d(de, cf, 1, bias=False) if de != cf else None
            self.lnp = make_ln2d(cf) if de != cf else None
            self.pos = nn.Embedding(64, cf)
            self.blocks = nn.ModuleList(
                [Block(_POS_SCALES.get(i, 0.0)) for i in range(cb)]
            )
            self.mix         = nn.Conv2d(cf, cf, 1, bias=False)
            self.lnm         = make_ln2d(cf)
            self.policy_head = make_pt_relational_policy_head(cf)
            self.value_head  = make_pt_attn_pool_value_head(cf)

        def forward(self, t):
            b = t.shape[0]

            x = self.emb(t).reshape(b, 8, 8, de).permute(0, 3, 1, 2)
            x = x.contiguous()

            if self.proj is not None:
                x = F.leaky_relu(self.lnp(self.proj(x)), 0.01)

            pos = self.pos(torch.arange(64, device=t.device)).unsqueeze(0)

            for block in self.blocks:
                x = block(x, pos)

            x = F.leaky_relu(self.lnm(self.mix(x)), 0.01)
            return self.policy_head(x), self.value_head(x)

    m = M()
    print(f"  PT params: {sum([p.numel() for p in m.parameters()]):,}")
    return m


def build_pt_transformer_16m(cfg):
    """Pure transformer with flat pos reinjection (0.1 per layer after first).
    Heads match TF: relational bilinear policy (4288), value_mix + attn pool value.
    Default config: d_embed=384, 8 layers, 8 heads, ff_dim=1536 (~16M params).
    """
    import torch, torch.nn as nn, torch.nn.functional as F

    de = cfg["d_embed"]
    tl = cfg["transformer_layers"]
    nh = cfg["num_heads"]
    fd = cfg["ff_dim"]
    dr = cfg["dropout"]

    class TxLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1   = nn.LayerNorm(de)
            self.attn  = nn.MultiheadAttention(de, nh, dropout=0.0, batch_first=True)
            self.drop1 = nn.Dropout(dr)
            self.ln2   = nn.LayerNorm(de)
            self.ff1   = nn.Linear(de, fd)
            self.drop2 = nn.Dropout(dr)
            self.ff2   = nn.Linear(fd, de)

        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop1(h)
            return x + self.ff2(self.drop2(F.gelu(self.ff1(self.ln2(x)))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(VOCAB_SIZE, de)
            self.pos  = nn.Embedding(64, de)
            self.txs         = nn.ModuleList([TxLayer() for _ in range(tl)])
            self.val_mix     = nn.Conv2d(de, de, 1, bias=False)
            self.val_mix_ln  = make_ln2d(de)
            self.policy_head = make_pt_relational_policy_head(de)
            self.value_head  = make_pt_attn_pool_value_head(de)

        def forward(self, t):
            B   = t.shape[0]
            pos = self.pos(torch.arange(64, device=t.device)).unsqueeze(0)
            x   = self.emb(t) + pos
            for i, tx in enumerate(self.txs):
                if i > 0:
                    x = x + 0.1 * pos
                x = tx(x)
            # x: (B, 64, de)

            x_2d = x.reshape(B, 8, 8, de).permute(0, 3, 1, 2).contiguous()
            v_2d = F.leaky_relu(self.val_mix_ln(self.val_mix(x_2d)), 0.02)
            return self.policy_head(x_2d), self.value_head(v_2d)

    m = M()
    print(f"  PT params: {sum([p.numel() for p in m.parameters()]):,}")
    return m


def build_pt_film(cfg):
    """2×MHA context encoder → all gammas/betas from g upfront → FiLM conv stack."""
    import torch, torch.nn as nn, torch.nn.functional as F
    C = cfg["conv_filters"]; cb = cfg["conv_blocks"]
    nh = cfg["mha_heads"]; nl = cfg["mha_layers"]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb   = nn.Embedding(VOCAB_SIZE, C)
            self.pos   = nn.Embedding(64, C)
            self.ctx   = nn.ModuleList([make_bare_attn(C, nh) for _ in range(nl)])
            self.gamma = nn.ModuleList([nn.Conv2d(C, C, 1) for _ in range(cb)])
            self.beta  = nn.ModuleList([nn.Conv2d(C, C, 1) for _ in range(cb)])
            self.blks        = nn.ModuleList([self._blk() for _ in range(cb)])
            self.policy_head = make_pt_relational_policy_head(C)
            self.value_head  = make_pt_attn_pool_value_head(C)

        def _blk(self):
            # Returns a ModuleList so we can apply each op separately
            return nn.ModuleList([
                nn.Conv2d(C, C, 3, padding=1, bias=False),
                nn.LeakyReLU(0.01, True),
                nn.Conv2d(C, C, 3, padding=1, bias=False),
                make_ln2d(C),
            ])

        def forward(self, t):
            B = t.shape[0]
            proj = self.emb(t)          # (B, 64, C)
            g = proj + self.pos(torch.arange(64, device=t.device)).unsqueeze(0)
            for layer in self.ctx:
                g = layer(g)
            g2 = g.reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()  # (B,C,8,8)
            gs = [self.gamma[i](g2) for i in range(cb)]
            bs = [self.beta[i](g2)  for i in range(cb)]
            x = proj.reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
            for i, (c1, lr1, c2, ln) in enumerate(self.blks):
                r = x; h = lr1(c1(x)); h = ln(c2(h))
                h = h * gs[i] + bs[i]
                x = F.leaky_relu(r + h, 0.01)
            return self.policy_head(x), self.value_head(x)

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_concat_fusion(cfg):
    """Per-block: Conv(C,3) → concat with g_x(64ch projected from g) → Conv(C+G→C,3)."""
    import torch, torch.nn as nn, torch.nn.functional as F
    C = cfg["conv_filters"]; G = 64; cb = cfg["conv_blocks"]
    nh = cfg["mha_heads"]; nl = cfg["mha_layers"]

    class CFBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1   = nn.Conv2d(C, C, 3, padding=1, bias=False)
            self.lr1  = nn.LeakyReLU(0.01, True)
            self.gx   = nn.Conv2d(C, G, 1)
            self.gxlr = nn.LeakyReLU(0.01, True)
            self.c2   = nn.Conv2d(C + G, C, 3, padding=1)  # bias=True (mixing conv)
            self.ln   = make_ln2d(C)
            self.lr   = nn.LeakyReLU(0.01, True)

        def forward(self, x, g2):
            r  = x
            h  = self.lr1(self.c1(x))
            gx = self.gxlr(self.gx(g2))
            h  = torch.cat([h, gx], dim=1)   # (B, C+G, 8, 8)
            h  = self.ln(self.c2(h))
            return self.lr(r + h)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(VOCAB_SIZE, C)
            self.pos  = nn.Embedding(64, C)
            self.ctx  = nn.ModuleList([make_bare_attn(C, nh) for _ in range(nl)])
            self.blks        = nn.ModuleList([CFBlock() for _ in range(cb)])
            self.policy_head = make_pt_relational_policy_head(C)
            self.value_head  = make_pt_attn_pool_value_head(C)

        def forward(self, t):
            B = t.shape[0]
            proj = self.emb(t)
            g = proj + self.pos(torch.arange(64, device=t.device)).unsqueeze(0)
            for layer in self.ctx:
                g = layer(g)
            g2 = g.reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
            x  = proj.reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
            for blk in self.blks:
                x = blk(x, g2)
            return self.policy_head(x), self.value_head(x)

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_gated_ctx(cfg):
    """Standard ConvResBlock then sigmoid-gated context injection per block."""
    import torch, torch.nn as nn, torch.nn.functional as F
    C = cfg["conv_filters"]; cb = cfg["conv_blocks"]
    nh = cfg["mha_heads"]; nl = cfg["mha_layers"]

    class GCBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1   = nn.Conv2d(C, C, 3, padding=1, bias=False)
            self.lr1  = nn.LeakyReLU(0.01, True)
            self.c2   = nn.Conv2d(C, C, 3, padding=1, bias=False)
            self.ln   = make_ln2d(C)
            self.lr   = nn.LeakyReLU(0.01, True)
            self.gate = nn.Conv2d(C, C, 1)   # gate from post-resblock x
            self.ctx  = nn.Conv2d(C, C, 1)   # context projection from g_2d

        def forward(self, x, g2):
            r = x; h = self.lr1(self.c1(x)); h = self.ln(self.c2(h))
            x = self.lr(r + h)
            return x + torch.sigmoid(self.gate(x)) * self.ctx(g2)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(VOCAB_SIZE, C)
            self.pos  = nn.Embedding(64, C)
            self.ctx  = nn.ModuleList([make_bare_attn(C, nh) for _ in range(nl)])
            self.blks        = nn.ModuleList([GCBlock() for _ in range(cb)])
            self.policy_head = make_pt_relational_policy_head(C)
            self.value_head  = make_pt_attn_pool_value_head(C)

        def forward(self, t):
            B = t.shape[0]
            proj = self.emb(t)
            g = proj + self.pos(torch.arange(64, device=t.device)).unsqueeze(0)
            for layer in self.ctx:
                g = layer(g)
            g2 = g.reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
            x  = proj.reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
            for blk in self.blks:
                x = blk(x, g2)
            return self.policy_head(x), self.value_head(x)

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


PT_BUILDERS = {
    "16m-pure-conv":             build_pt_pure_conv,
    "16m-conformer":             build_pt_conformer,
    "16m-film":                  build_pt_film,
    "16m-concat-fusion":         build_pt_concat_fusion,
    "16m-gated-ctx":             build_pt_gated_ctx,
    "16m-transformer":           build_pt_transformer_16m,
    "16m-conformer-interweaved": build_pt_conformer_interweaved,
}

# ---------------------------------------------------------------------------
# Inference wrappers
# ---------------------------------------------------------------------------

def make_tf_infer(name, cfg):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    model = TF_BUILDERS[name](cfg)

    @tf.function(
        input_signature=[tf.TensorSpec([None, SEQ_LEN], tf.int32)],
        jit_compile=True,
    )
    def _fwd(x):
        return model(x, training=False)

    _fwd(tf.constant(random_tokens(4)))   # trigger trace / XLA compile

    def infer(tokens_np):
        out = _fwd(tf.constant(tokens_np))
        return out[0].numpy(), out[1].numpy()

    return infer, model


def make_pt_eager_infer(name, cfg, device):
    import torch
    model = PT_BUILDERS[name](cfg).half().to(device).eval()

    def infer(tokens_np):
        with torch.no_grad():
            x = torch.from_numpy(tokens_np).long().to(device)
            p, v = model(x)
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    return infer


def make_pt_compiled_infer(name, cfg, device):
    import torch
    model   = PT_BUILDERS[name](cfg).half().to(device).eval()
    backend = "inductor"
    compiled = torch.compile(model, dynamic=True, backend=backend)
    try:
        with torch.no_grad():
            compiled(torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device))
        print(f"  torch.compile backend: {backend}")
    except Exception as e:
        print(f"  inductor failed ({type(e).__name__}), falling back to cudagraphs")
        backend = "cudagraphs"
        torch._dynamo.config.cache_size_limit = 32
        compiled = torch.compile(model, dynamic=False, backend=backend)

    def infer(tokens_np):
        with torch.no_grad():
            x = torch.from_numpy(tokens_np).long().to(device)
            p, v = compiled(x)
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    return infer, backend


def export_to_onnx(name, cfg, device, onnx_path, opset=14):
    import torch
    print(f"  Exporting {name} to ONNX (opset {opset}) ...")
    model = PT_BUILDERS[name](cfg).half().to(device).eval()
    dummy = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    try:
        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=opset,
            input_names=["enc_in"],
            output_names=["policy_logits", "value_out"],
            dynamic_axes={"enc_in": {0: "batch"},
                          "policy_logits": {0: "batch"},
                          "value_out": {0: "batch"}},
        )
    except Exception as e:
        if opset < 16:
            print(f"  opset {opset} failed ({type(e).__name__}), retrying with opset 16 ...")
            torch.onnx.export(
                model, dummy, onnx_path,
                opset_version=16,
                input_names=["enc_in"],
                output_names=["policy_logits", "value_out"],
                dynamic_axes={"enc_in": {0: "batch"},
                              "policy_logits": {0: "batch"},
                              "value_out": {0: "batch"}},
            )
        else:
            raise
    del model
    print("  ONNX export done.")


def run_shape_inference(onnx_path):
    try:
        import onnx
        from onnx import shape_inference
        model = onnx.load(onnx_path)
        onnx.save(shape_inference.infer_shapes(model), onnx_path)
    except Exception as e:
        print(f"  [WARN] shape inference skipped: {e}")


def make_trt_infer(onnx_path, cache_dir):
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3
    avail = ort.get_available_providers()
    providers = []
    if "TensorrtExecutionProvider" in avail:
        providers.append(("TensorrtExecutionProvider", {
            "trt_fp16_enable":         True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path":   cache_dir,
            "trt_max_workspace_size":  4 * 1024 * 1024 * 1024,
            "trt_profile_min_shapes":  "enc_in:1x64",
            "trt_profile_opt_shapes":  "enc_in:64x64",
            "trt_profile_max_shapes":  "enc_in:256x64",
        }))
    else:
        print("  WARNING: TensorrtExecutionProvider not available — falling back to CUDA EP")
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    print(f"  TRT session providers: {session.get_providers()}")
    iname = session.get_inputs()[0].name
    itype = session.get_inputs()[0].type

    def infer(tokens_np):
        feed = tokens_np.astype(np.int64) if "int64" in itype else tokens_np
        policy, value = session.run(None, {iname: feed})
        return policy, value

    return infer, session

# ---------------------------------------------------------------------------
# Speed test
# ---------------------------------------------------------------------------

def speed_test(label, infer_fn, batch_sizes):
    print(f"\n  {'─'*56}")
    print(f"  {label}")
    print(f"  warmup={N_WARMUP}  iters={N_ITERS}")
    print(f"  {'batch':<7} {'latency_ms':>12} {'samples/s':>13}")
    print(f"  {'-'*7} {'-'*12} {'-'*13}")
    results = {}
    for b in batch_sizes:
        inp = random_tokens(b)
        for _ in range(N_WARMUP):
            infer_fn(inp)
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            infer_fn(inp)
        t1 = time.perf_counter()
        avg_s = (t1 - t0) / N_ITERS
        tput  = b / avg_s
        print(f"  {b:<7} {avg_s*1000:>12.2f} {tput:>13.0f}")
        results[b] = {"latency_ms": round(avg_s * 1000, 3), "throughput": round(tput, 1)}
    return results

# ---------------------------------------------------------------------------
# TF GPU init helper (called once per variant to avoid repeated setup)
# ---------------------------------------------------------------------------

tf_gpu_initialized = False

def init_tf_gpu():
    global tf_gpu_initialized
    if tf_gpu_initialized:
        return
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    tf_gpu_initialized = True

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    variants = args.variants or VARIANT_ORDER
    bs       = args.batch_sizes
    save     = not args.no_save and not args.dry_run

    if save:
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(TRT_CACHE, exist_ok=True)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PyTorch device: {device}")
    except ImportError:
        device = None
        print("WARNING: torch not available")

    all_results = {}
    csv_rows    = []

    for name in variants:
        cfg = dict(VARIANTS[name])
        describe_variant(name, cfg)

        res = {}

        # ── dry-run: just build + count ──────────────────────────────────
        if args.dry_run:
            if name in TF_BUILDERS:
                print(f"  Building TF model ...")
                try:
                    init_tf_gpu()
                    import tensorflow as tf
                    tf_model = TF_BUILDERS[name](cfg)
                    del tf_model
                    gc.collect()
                    tf.keras.backend.clear_session()
                except Exception as e:
                    print(f"  [ERROR] TF build failed: {e}")
            else:
                print(f"  Skipping TF build (PT-only variant)")

            if device is not None and name in PT_BUILDERS:
                print(f"  Building PT model ...")
                try:
                    pt_model = PT_BUILDERS[name](cfg)
                    del pt_model
                    gc.collect()
                except Exception as e:
                    print(f"  [ERROR] PT build failed: {e}")
            continue

        # ── TF + XLA ──────────────────────────────────────────────────────
        if name in TF_BUILDERS:
            try:
                init_tf_gpu()
                import tensorflow as tf
                print(f"\n  Building TF+XLA for {name} ...")
                tf_infer, tf_model = make_tf_infer(name, cfg)
                res["TF+XLA"] = speed_test(f"{name}  TF+XLA [fp16]", tf_infer, bs)

                if save:
                    path = os.path.join(args.model_dir, f"{name}_model.h5")
                    tf_model.save(path)
                    print(f"  Saved TF model → {path}")

                del tf_infer, tf_model
                gc.collect()
                tf.keras.backend.clear_session()
            except Exception as e:
                print(f"  [ERROR] TF+XLA for {name}: {e}")
        else:
            print(f"\n  Skipping TF+XLA for {name} (PT-only variant)")

        # ── PT eager ──────────────────────────────────────────────────────
        if device is not None and name in PT_BUILDERS:
            try:
                import torch
                print(f"\n  Building PT eager for {name} ...")
                eager = make_pt_eager_infer(name, cfg, device)
                res["PT eager"] = speed_test(f"{name}  PT eager [fp16]", eager, bs)

                if save and ("transformer" in name or "conformer-interweaved" in name):
                    pt_model = PT_BUILDERS[name](cfg).half().to(device)
                    pt_path  = os.path.join(args.model_dir, f"{name}_pt_model.pt")
                    torch.save({"model": pt_model.state_dict()}, pt_path)
                    print(f"  Saved PT model → {pt_path}")
                    del pt_model

                del eager
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [ERROR] PT eager for {name}: {e}")

            # ── PT compiled ──────────────────────────────────────────────
            # try:
            #     import torch
            #     print(f"\n  Building PT compiled for {name} ...")
            #     compiled, be = make_pt_compiled_infer(name, cfg, device)
            #     res[f"PT {be}"] = speed_test(f"{name}  PT {be} [fp16]", compiled, bs)
            #     del compiled
            #     gc.collect()
            #     torch.cuda.empty_cache()
            # except Exception as e:
            #     print(f"  [ERROR] PT compiled for {name}: {e}")

        # ── ORT + TRT ─────────────────────────────────────────────────────
        if not args.skip_trt and device is not None:
            try:
                import torch
                key       = make_cache_key(name, cfg)
                onnx_path = os.path.join(TRT_CACHE, f"{key}.onnx")

                if not os.path.exists(onnx_path):
                    export_to_onnx(name, cfg, device, onnx_path)
                    run_shape_inference(onnx_path)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_path}")

                print(f"\n  Building TRT session for {name} ...")
                trt_infer, trt_sess = make_trt_infer(onnx_path, TRT_CACHE)
                res["ORT TRT"] = speed_test(f"{name}  ORT TRT [fp16]", trt_infer, bs)
                del trt_sess
                gc.collect()
            except Exception as e:
                print(f"  [ERROR] ORT TRT for {name}: {e}")

        all_results[name] = res

        # accumulate CSV rows
        for backend, batch_res in res.items():
            for batch_size, metrics in batch_res.items():
                csv_rows.append({
                    "variant":    name,
                    "backend":    backend,
                    "batch_size": batch_size,
                    "latency_ms": metrics["latency_ms"],
                    "throughput": metrics["throughput"],
                })

        # save CSV incrementally after each variant
        if save and csv_rows:
            save_csv_incremental(csv_rows, args.model_dir)

    # ── dry-run exit ───────────────────────────────────────────────────────
    if args.dry_run:
        print("\n  Dry run complete.")
        return

    # ── Final CSV save ─────────────────────────────────────────────────────
    if save and csv_rows:
        save_csv_incremental(csv_rows, args.model_dir)

    # ── Summary table ──────────────────────────────────────────────────────
    if all_results:
        key_bs = [b for b in [1, 32, 64, 256] if b in bs]

        # Collect all backends that appeared in results
        seen_backends = []
        for name in variants:
            for be in all_results.get(name, {}).keys():
                if be not in seen_backends:
                    seen_backends.append(be)

        # Canonical ordering: TF+XLA first, then PT eager, PT compiled variants, ORT TRT
        ordered_backends = []
        for prio in ["TF+XLA", "PT eager"]:
            if prio in seen_backends:
                ordered_backends.append(prio)
        for be in seen_backends:
            if be.startswith("PT ") and be not in ordered_backends:
                ordered_backends.append(be)
        if "ORT TRT" in seen_backends:
            ordered_backends.append("ORT TRT")

        DISPLAY_NAMES = {
            "16m-pure-conv":             "16m-pure-conv",
            "16m-conformer":             "16m-conformer",
            "16m-conformer-interweaved": "16m-conf-intwv",
            "16m-film":                  "16m-film",
            "16m-concat-fusion":         "16m-concat-fus",
            "16m-gated-ctx":             "16m-gated-ctx",
            "16m-transformer":           "16m-transformer",
        }

        cw      = 12
        nw      = 22
        total_w = 2 + nw + cw * len(key_bs)
        for backend in ordered_backends:
            has = any(backend in all_results.get(n, {}) for n in variants)
            if not has:
                continue
            print(f"\n{'═'*total_w}")
            print(f"  {backend} throughput (samples/s)")
            print(f"{'═'*total_w}")
            print(f"  {'variant':<{nw}}" + "".join(f"{b:>{cw}}" for b in key_bs))
            print("  " + "─" * (nw + cw * len(key_bs)))
            for n in variants:
                label = DISPLAY_NAMES.get(n, n[:nw])
                row   = f"  {label:<{nw}}"
                bres  = all_results.get(n, {}).get(backend, {})
                for b in key_bs:
                    tp = bres.get(b, {}).get("throughput")
                    row += f"{tp:>{cw}.0f}" if tp is not None else f"{'N/A':>{cw}}"
                print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
