"""model.py — canonical model definitions for Xerces chess engine.

All builders import their framework lazily so this module is safe to
import from any context (PT subprocess, TF training, selfplay loop).

Exports:
    VOCAB_SIZE, SEQ_LEN
    VARIANTS            — shared cfg dict used by speed test and bootstrap
    TF_BUILDERS         — {name: fn(cfg) -> compiled TF model}
    PT_BUILDERS         — {name: fn(cfg) -> PT nn.Module}
    make_pt_relational_policy_head, make_pt_attn_pool_value_head, make_ln2d
    tf_relational_policy_head, tf_attn_pool_value_head, tf_compile
    build_conformer_64x67   — active production selfplay model
    make_conv_infer         — XLA-compiled TF inference wrapper
    set_loss_weights        — recompile TF model with updated loss weights
    load_model, save_model  — TF .h5 helpers
"""
from __future__ import annotations

VOCAB_SIZE = 21
SEQ_LEN    = 64

VARIANTS: dict[str, dict] = {
    "16m-transformer": dict(
        transformer=True,
        d_embed=512, transformer_layers=7, num_heads=16,
        ff_dim=1024, dropout=0.05,
    ),
    "16m-conformer-interweaved": dict(
        conformer_interweaved=True,
        d_embed=256, conv_filters=256, conv_blocks=10,
        num_heads=8, ff_dim=1024, dropout=0.05,
    ),
    "16m-transformer-branched": dict(
        transformer_branched=True,
        d_embed=512, transformer_layers=4, num_heads=16,
        ff_dim=1024, dropout=0.05,
    ),
    "13m-precond-conformer": dict(
        precond_conformer=True,
        conv_filters=256, num_heads=8, dropout=0.05,
        pre_blocks=4, mha_blocks=4,
    ),
}


# ---------------------------------------------------------------------------
# PyTorch shared helpers
# ---------------------------------------------------------------------------

def make_ln2d(ch: int):
    """Channel-wise LayerNorm for (B, C, H, W) tensors."""
    import torch
    import torch.nn as nn

    class Ln2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(ch))
            self.b = nn.Parameter(torch.zeros(ch))

        def forward(self, x):
            c = x - x.mean(1, keepdim=True)
            return (
                c * torch.rsqrt((c * c).mean(1, keepdim=True) + 1e-5)
                * self.w.view(1, -1, 1, 1) + self.b.view(1, -1, 1, 1)
            )

    return Ln2d()


def make_pt_attn_pool_value_head(C: int):
    """Attention-pool value head. Accepts (B, C, 8, 8) or (B, 64, C)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class AttnPoolValueHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln    = nn.LayerNorm(C)
            self.proj  = nn.Linear(C, 256)
            self.mha   = nn.MultiheadAttention(256, 4, dropout=0.0, batch_first=True)
            self.query = nn.Parameter(torch.randn(1, 1, 256))
            self.out   = nn.Linear(256, 3)

        def forward(self, x):
            if x.dim() == 4:
                x = x.permute(0, 2, 3, 1).reshape(-1, 64, C)
            x = F.gelu(self.proj(self.ln(x)))           # (B, 64, 256)
            q = self.query.expand(x.shape[0], -1, -1)   # (B, 1, 256)
            h, _ = self.mha(q, x, x, need_weights=False)
            return self.out(h.squeeze(1))

    return AttnPoolValueHead()


def make_pt_mha_policy_head(C: int):
    """MHA-refined bilinear policy head. Accepts (B, C, 8, 8).
    Each from/to branch gets LN -> proj(256) -> GELU -> MHA -> dense(256),
    then BMM dot product scaled by 1/sqrt(256) for move logits.
    Returns (B, 4288): 4096 normal + 192 underpromotion logits."""
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    D = 256

    class MHAPolicyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln         = nn.LayerNorm(C)
            self.scale      = 1.0 / math.sqrt(D)

            self.from_proj  = nn.Linear(C, D)
            self.from_mha   = nn.MultiheadAttention(D, 4, dropout=0.0, batch_first=True)
            self.from_ln    = nn.LayerNorm(D)
            self.from_out   = nn.Linear(D, D)

            self.to_proj    = nn.Linear(C, D)
            self.to_mha     = nn.MultiheadAttention(D, 4, dropout=0.0, batch_first=True)
            self.to_ln      = nn.LayerNorm(D)
            self.to_out     = nn.Linear(D, D)

            self.promo_mix  = nn.Conv2d(C, 64, 1, bias=False)
            self.promo_mln  = make_ln2d(64)
            self.promo_c1   = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.promo_ln   = make_ln2d(64)
            self.promo_out  = nn.Conv2d(64, 3, 1)

            mask = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool()
            self.register_buffer("sometimes_legal", mask)

        def forward(self, x):
            s  = self.ln(x.permute(0, 2, 3, 1).reshape(-1, 64, C))

            f  = F.gelu(self.from_proj(s))
            fn = self.from_ln(f)
            fh, _ = self.from_mha(fn, fn, fn, need_weights=False)
            fv = self.from_out(f + fh)

            t  = F.gelu(self.to_proj(s))
            tn = self.to_ln(t)
            th, _ = self.to_mha(tn, tn, tn, need_weights=False)
            tv = self.to_out(t + th)

            dots   = torch.bmm(fv.float(), tv.float().transpose(1, 2)).mul(self.scale).reshape(-1, 64 * 64)

            p      = F.leaky_relu(self.promo_mln(self.promo_mix(x)), 0.02)
            sk     = p
            p      = F.leaky_relu(self.promo_ln(self.promo_c1(p)), 0.02)
            promo  = self.promo_out(p + sk).permute(0, 2, 3, 1).reshape(-1, 8 * 8 * 3).float()
            
            logits = torch.cat([dots, promo], dim=1)
            logits = logits.masked_fill(~self.sometimes_legal, -1e9)
            return logits.to(x.dtype)

    return MHAPolicyHead()


def make_pt_relational_policy_head(C: int):
    """Bilinear from/to policy head. Accepts (B, C, 8, 8).
    Returns (B, 4288): 4096 normal + 192 underpromotion logits.
    Never-legal indices are masked to -1e9 so they carry no gradient."""
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    class RelationalPolicyHead(nn.Module):
        def __init__(self):
            super().__init__()
            inner_dim = 512
            self.ln        = nn.LayerNorm(C)
            self.from_fc1  = nn.Linear(C, inner_dim)
            self.from_fc2  = nn.Linear(inner_dim, inner_dim)

            self.to_fc1    = nn.Linear(C, inner_dim)
            self.to_fc2    = nn.Linear(inner_dim, inner_dim)
            self.scale     = 1.0 / math.sqrt(inner_dim)

            self.promo_mix = nn.Conv2d(C, 64, 1, bias=False)
            self.promo_mln = make_ln2d(64)
            self.promo_c1  = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.promo_ln  = make_ln2d(64)
            self.promo_out = nn.Conv2d(64, 3, 1)

            mask = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool()
            self.register_buffer("sometimes_legal", mask)

        def forward(self, x):
            s     = self.ln(x.permute(0, 2, 3, 1).reshape(-1, 64, C))
            fv    = self.from_fc2(F.gelu(self.from_fc1(s)))
            tv    = self.to_fc2(F.gelu(self.to_fc1(s)))
            dots  = torch.bmm(fv.float(), tv.float().transpose(1, 2)).mul(self.scale).reshape(-1, 64 * 64)
            p     = F.leaky_relu(self.promo_mln(self.promo_mix(x)), 0.02)
            sk    = p
            p     = F.leaky_relu(self.promo_ln(self.promo_c1(p)), 0.02)
            promo = self.promo_out(p + sk).permute(0, 2, 3, 1).reshape(-1, 8 * 8 * 3).float()
            logits = torch.cat([dots, promo], dim=1)
            logits = logits.masked_fill(~self.sometimes_legal, -1e9)
            return logits.to(x.dtype)

    return RelationalPolicyHead()


# ---------------------------------------------------------------------------
# TF shared helpers
# ---------------------------------------------------------------------------

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


def tf_attn_pool_value_head(x, C, layers, activation="gelu"):
    """Attention-pool value head for TF functional API.
    x: (B, 8, 8, C) or (B, 64, C). Returns value_out (B, 3) float32."""
    
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
    """Bilinear from/to policy head for TF functional API.
    x: (B, 8, 8, C) or (B, 64, C). Returns policy_logits (B, 4288)."""
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

    to_h   = layers.Dense(256, activation="gelu", name="pol_to_fc1")(x_seq)
    to_vec = layers.Dense(256, name="pol_to_fc2")(to_h)

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
    """Shared trunk of tl layers splits into 3 independent branch MHAs:
    value branch, policy-from branch, policy-to branch.
    Cross-bilinear dot(from_vec, to_vec) -> (B,4096) + promo -> (B,4288)."""
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

    import math
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
    "16m-transformer":          build_tf_transformer,
    "16m-conformer-interweaved": build_tf_conformer_interweaved,
    "16m-transformer-branched": build_tf_transformer_branched,
}


# ---------------------------------------------------------------------------
# PyTorch builders
# ---------------------------------------------------------------------------

def build_pt_transformer_16m(cfg: dict):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
            self.emb         = nn.Embedding(VOCAB_SIZE, de)
            self.pos         = nn.Embedding(SEQ_LEN, de)
            self.txs         = nn.ModuleList([TxLayer() for _ in range(tl)])
            self.val_mix     = nn.Conv2d(de, de, 1, bias=False)
            self.val_mix_ln  = make_ln2d(de)
            self.policy_head = make_pt_relational_policy_head(de)
            self.value_head  = make_pt_attn_pool_value_head(de)

        def forward(self, t):
            pos = self.pos(torch.arange(SEQ_LEN, device=t.device)).unsqueeze(0)
            x   = self.emb(t) + pos
            for tx in self.txs:
                x = tx(x)
            x_2d = x.reshape(-1, 8, 8, de).permute(0, 3, 1, 2).contiguous()
            v_2d = F.leaky_relu(self.val_mix_ln(self.val_mix(x_2d)), 0.02)
            return self.policy_head(x_2d), self.value_head(v_2d)

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_conformer_interweaved(cfg: dict):
    """Interweaved conformer: cb x [prenorm-MHA -> ConvRes].
    Graduated pos injection: block 0=1.0, 2=0.1, 4=0.05, 6=0.025, others=none."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    de = cfg["d_embed"]
    cf = cfg["conv_filters"]
    cb = cfg["conv_blocks"]
    nh = cfg["num_heads"]
    dr = cfg["dropout"]

    POS_SCALES = {0: 1.0, 2: 0.1, 4: 0.05, 6: 0.025}

    class Block(nn.Module):
        def __init__(self, pos_scale):
            super().__init__()
            self.pos_scale = pos_scale
            self.ln1   = nn.LayerNorm(cf)
            self.attn  = nn.MultiheadAttention(cf, nh, dropout=0.0, batch_first=True)
            self.drop1 = nn.Dropout(dr)
            self.c1    = nn.Conv2d(cf, cf, 3, padding=1, bias=False)
            self.lr1   = nn.LeakyReLU(0.01, True)
            self.c2    = nn.Conv2d(cf, cf, 3, padding=1, bias=False)
            self.ln2   = make_ln2d(cf)

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
            return F.leaky_relu(r + h, 0.01)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb         = nn.Embedding(VOCAB_SIZE, de)
            self.proj        = nn.Conv2d(de, cf, 1, bias=False) if de != cf else None
            self.lnp         = make_ln2d(cf) if de != cf else None
            self.pos         = nn.Embedding(SEQ_LEN, cf)
            self.blocks      = nn.ModuleList(
                [Block(POS_SCALES.get(i, 0.0)) for i in range(cb)]
            )
            self.mix         = nn.Conv2d(cf, cf, 1, bias=False)
            self.lnm         = make_ln2d(cf)
            self.policy_head = make_pt_relational_policy_head(cf)
            self.value_head  = make_pt_attn_pool_value_head(cf)

        def forward(self, t):
            b = t.shape[0]
            x = self.emb(t).reshape(b, 8, 8, de).permute(0, 3, 1, 2).contiguous()
            if self.proj is not None:
                x = F.leaky_relu(self.lnp(self.proj(x)), 0.01)
            pos = self.pos(torch.arange(SEQ_LEN, device=t.device)).unsqueeze(0)
            for block in self.blocks:
                x = block(x, pos)
            x = F.leaky_relu(self.lnm(self.mix(x)), 0.01)
            return self.policy_head(x), self.value_head(x)

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_precond_conformer(cfg: dict):
    """4x Conv(256) preconditioner -> concat pos(256) -> 512 -> 4x [prenorm-MHA -> FF(512->512->512)] -> heads."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    CF = cfg["conv_filters"]   # 256
    D  = CF * 2                # 512 after concat
    nh = cfg["num_heads"]
    dr = cfg["dropout"]
    pb = cfg["pre_blocks"]
    mb = cfg["mha_blocks"]

    class ConvBlock(nn.Module):
        def __init__(self, prenorm=True):
            super().__init__()
            self.ln = make_ln2d(CF) if prenorm else None
            self.c1 = nn.Conv2d(CF, CF, 3, padding=1, bias=False)
            self.c2 = nn.Conv2d(CF, CF, 3, padding=1, bias=False)

        def forward(self, x):
            r = x
            h = self.ln(x) if self.ln is not None else x
            h = F.leaky_relu(self.c1(h), 0.01)
            h = F.leaky_relu(self.c2(h), 0.01)
            return r + h

    class TxBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1  = nn.LayerNorm(D)
            self.attn = nn.MultiheadAttention(D, nh, dropout=0.0, batch_first=True)
            self.drop = nn.Dropout(dr)
            self.ln2  = nn.LayerNorm(D)
            self.ff1  = nn.Linear(D, D)
            self.ff2  = nn.Linear(D, D)

        def forward(self, x):
            r = x
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = r + self.drop(h)
            r = x
            return r + self.ff2(F.gelu(self.ff1(self.ln2(x))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb         = nn.Embedding(VOCAB_SIZE, CF)
            self.pre         = nn.ModuleList([ConvBlock(prenorm=(i > 0)) for i in range(pb)])
            self.pos         = nn.Embedding(SEQ_LEN, CF)
            self.blocks      = nn.ModuleList([TxBlock() for _ in range(mb)])
            self.mix         = nn.Conv2d(D, D, 1, bias=False)
            self.lnm         = make_ln2d(D)
            self.policy_head = make_pt_mha_policy_head(D)
            self.value_head  = make_pt_attn_pool_value_head(D)

        def forward(self, t):
            B = t.shape[0]
            x = self.emb(t).reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()
            for blk in self.pre:
                x = blk(x)
            conv_seq = x.permute(0, 2, 3, 1).reshape(B, 64, CF)
            pos = self.pos(torch.arange(SEQ_LEN, device=t.device)).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([conv_seq, pos], dim=-1)
            for blk in self.blocks:
                x = blk(x)
            x = x.reshape(B, 8, 8, D).permute(0, 3, 1, 2).contiguous()
            x = F.leaky_relu(self.lnm(self.mix(x)), 0.01)
            return self.policy_head(x), self.value_head(x)

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


PT_BUILDERS: dict[str, object] = {
    "16m-transformer":          build_pt_transformer_16m,
    "16m-conformer-interweaved": build_pt_conformer_interweaved,
    "13m-precond-conformer":    build_pt_precond_conformer,
}


# ---------------------------------------------------------------------------
# Active production selfplay model
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
    """Conv frontend -> transformer stack -> 64x67 policy + value head.
    Active production model used during selfplay."""
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
    """XLA-compiled TF inference wrapper.
    Returns fwd((enc_np,)) -> (logits_np, wdl_np).
    logits_np: (B, 4288) float32 raw policy logits.
    wdl_np:    (B, 3)    float32 STM-POV softmax probs."""
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
