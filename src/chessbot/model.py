"""model.py — PyTorch model definitions for Xerces chess engine.

Exports:
    VOCAB_SIZE, SEQ_LEN
    VARIANTS     — shared cfg dict used by speed test and bootstrap
    PT_BUILDERS  — {name: fn(cfg) -> PT nn.Module}
    make_pt_relational_policy_head, make_pt_attn_pool_value_head, make_ln2d
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
    "16m-precond-smartgate": dict(
        conv_filters=256, num_heads=8, dropout=0.03,
        pre_blocks=4,
    ),
    "16m-precond-smartgate-lc0": dict(
        conv_filters=256, num_heads=8, dropout=0.03,
        pre_blocks=4, lc0_input=True,
    ),
    "16m-precond-smartgate-xc0h": dict(
        conv_filters=256, num_heads=8, dropout=0.03,
        pre_blocks=4, xc0h_K=6,
    ),
    "18m-precond-smartgate-xc0h-6c4t": dict(
        conv_filters=256, num_heads=8, dropout=0.03,
        pre_blocks=6, tx_blocks=4, xc0h_K=6,
    ),
    "hybrid-conv-attn": dict(
        hybrid_conv_attn=True,
        n_trunk_blocks=6, trunk_dim=1024, dropout=0.02,
    ),
    "conv-gemm": dict(
        n_trunk_blocks=6, dropout=0.02,
    ),
    "conv-gemm-smartgate": dict(
        n_trunk_blocks=6, dropout=0.02,
    ),
    "conv-mha-gemm-smartgate": dict(
        n_trunk_blocks=6, dropout=0.02,
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


def make_pt_attn_pool_value_head(C: int, n_heads: int = 4):
    """Attention-pool value head. Accepts (B, C, 8, 8) or (B, N, C)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class AttnPoolValueHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln    = nn.LayerNorm(C)
            self.proj  = nn.Linear(C, 256)
            self.mha   = nn.MultiheadAttention(256, n_heads, dropout=0.0, batch_first=True)
            self.query = nn.Parameter(torch.randn(1, 1, 256))
            self.out   = nn.Linear(256, 3)

        def forward(self, x):
            x = F.gelu(self.proj(self.ln(x)))
            q = self.query.expand(x.shape[0], -1, -1)   # (B, 1, 256)
            h, _ = self.mha(q, x, x, need_weights=False)
            return self.out(h.squeeze(1))

    return AttnPoolValueHead()


def make_pt_mha_policy_head(C: int, n_heads: int = 4):
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
            self.from_mha   = nn.MultiheadAttention(D, n_heads, dropout=0.0, batch_first=True)
            self.from_ln    = nn.LayerNorm(D)
            self.from_out   = nn.Linear(D, D)

            self.to_proj    = nn.Linear(C, D)
            self.to_mha     = nn.MultiheadAttention(D, n_heads, dropout=0.0, batch_first=True)
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
            B  = x.shape[0]
            s  = self.ln(x)                              # (B, 68, C)

            f   = F.gelu(self.from_proj(s))              # (B, 68, 256)
            fn  = self.from_ln(f)
            fh, _ = self.from_mha(fn[:, :64, :], fn, fn, need_weights=False)
            fv  = self.from_out(f[:, :64, :] + fh)      # (B, 64, 256)

            t   = F.gelu(self.to_proj(s))                # (B, 68, 256)
            tn  = self.to_ln(t)
            th, _ = self.to_mha(tn[:, :64, :], tn, tn, need_weights=False)
            tv  = self.to_out(t[:, :64, :] + th)        # (B, 64, 256)

            dots  = torch.bmm(fv.float(), tv.float().transpose(1, 2)).mul(self.scale).reshape(-1, 64 * 64)

            xp    = x[:, :64, :].reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
            p     = F.leaky_relu(self.promo_mln(self.promo_mix(xp)), 0.02)
            sk    = p
            p     = F.leaky_relu(self.promo_ln(self.promo_c1(p)), 0.02)
            promo = self.promo_out(p + sk).permute(0, 2, 3, 1).reshape(-1, 8 * 8 * 3).float()

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
# PyTorch builders
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PyTorch builders
# ---------------------------------------------------------------------------

def build_pt_transformer_16m(cfg: dict, log_params: bool = False):
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
            return self.policy_head(x_2d), self.value_head(v_2d.permute(0, 2, 3, 1).reshape(-1, 64, de))

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_conformer_interweaved(cfg: dict, log_params: bool = False):
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
            return self.policy_head(x), self.value_head(x.permute(0, 2, 3, 1).reshape(-1, 64, cf))

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_precond_conformer(cfg: dict, log_params: bool = False):
    """
    4x Conv(256) preconditioner -> concat pos(256) -> 512 ->
    4x [prenorm-MHA -> FF(512->512->512)] -> heads.
    """
    
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
        def __init__(self, ff_dim=D):
            super().__init__()
            self.ln1  = nn.LayerNorm(D)
            self.attn = nn.MultiheadAttention(D, nh, dropout=0.0, batch_first=True)
            self.drop = nn.Dropout(dr)
            self.ln2  = nn.LayerNorm(D)
            self.ff1  = nn.Linear(D, ff_dim)
            self.ff2  = nn.Linear(ff_dim, D)

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
            self.emb           = nn.Embedding(VOCAB_SIZE, CF)
            self.pre           = nn.ModuleList([ConvBlock(prenorm=(i > 0)) for i in range(pb)])
            self.pos           = nn.Embedding(SEQ_LEN, CF)
            self.conv_ln       = nn.LayerNorm(CF)
            self.global_tokens = nn.Parameter(torch.randn(1, 4, D))

            ff_dims = [512, 768, 768, 512]
            self.blocks        = nn.ModuleList([TxBlock(ff_dim=ffd) for ffd in ff_dims])
            self.policy_head   = make_pt_mha_policy_head(D, n_heads=8)
            self.value_head    = make_pt_attn_pool_value_head(D, n_heads=8)

        def forward(self, t):
            B = t.shape[0]
            x = self.emb(t).reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()
            for blk in self.pre:
                x = blk(x)

            conv_seq = self.conv_ln(x.permute(0, 2, 3, 1).reshape(B, 64, CF))
            pos = self.pos(torch.arange(SEQ_LEN, device=t.device)).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([conv_seq, pos], dim=-1)

            gt = self.global_tokens.expand(B, -1, -1)
            x = torch.cat([x, gt], dim=1)

            for blk in self.blocks:
                x = blk(x)

            return self.policy_head(x), self.value_head(x)

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_hybrid_conv_attn(cfg: dict, log_params: bool = False):
    """Conv frontend + pos-cat -> 256-d -> self-attn(68) -> cross-attn(globals->68).

    Encoder: Embedding(21,128) -> 2xConv2D(128) -> cat([x, pos], dim=-1) -> [B,64,256]
      -> cat 4 global tokens -> [B,68,256]
      -> MHABlock self-attn -> CrossAttnBlock(Q=globals[4], KV=68) -> [B,4,256]
      -> reshape [B,1024]
    Trunk: n_trunk_blocks x SwiGLU @ trunk_dim
    Heads: policy Linear(D->1858->scatter 4288) + WDL value (3)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    ENC_DIM     = 128
    ENC_DIM_MHA = 256
    N_GLOBAL    = 4
    ENC_HEADS   = 8
    ENC_FF      = 1024
    POLICY_DIM  = 4288
    N_LEGAL     = 1858

    n_trunk = cfg["n_trunk_blocks"]
    D       = cfg["trunk_dim"]
    dr      = cfg["dropout"]
    assert N_GLOBAL * ENC_DIM_MHA == D

    sl_mask = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool()
    sl_idx  = sl_mask.nonzero(as_tuple=True)[0]  # [1858]

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))
            self.eps   = eps

        def forward(self, x):
            return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale

    class ConvBlock2D(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.c1 = nn.Conv2d(d, d, 3, padding=1, bias=False)
            self.c2 = nn.Conv2d(d, d, 3, padding=1, bias=False)

        def forward(self, x):
            B, _, d = x.shape
            h = self.ln(x).reshape(B, 8, 8, d).permute(0, 3, 1, 2)
            h = F.gelu(self.c1(h))
            h = self.c2(h)
            return x + h.permute(0, 2, 3, 1).reshape(B, 64, d)

    class MHABlock(nn.Module):
        def __init__(self, d, n_heads, ff_dim):
            super().__init__()
            self.ln1  = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.ln2  = nn.LayerNorm(d)
            self.ff1  = nn.Linear(d, ff_dim)
            self.ff2  = nn.Linear(ff_dim, d)

        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + h
            return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

    class CrossAttnBlock(nn.Module):
        def __init__(self, d, n_heads, ff_dim):
            super().__init__()
            self.ln_q  = nn.LayerNorm(d)
            self.ln_kv = nn.LayerNorm(d)
            self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.ln2   = nn.LayerNorm(d)
            self.ff1   = nn.Linear(d, ff_dim)
            self.ff2   = nn.Linear(ff_dim, d)

        def forward(self, q, kv):
            kv_n = self.ln_kv(kv)
            h, _ = self.attn(self.ln_q(q), kv_n, kv_n, need_weights=False)
            q = q + h
            return q + self.ff2(F.gelu(self.ff1(self.ln2(q))))

    class SwiGLUBlock(nn.Module):
        def __init__(self, d, hidden):
            super().__init__()
            self.norm   = RMSNorm(d)
            self.w_gate = nn.Linear(d, hidden, bias=False)
            self.w_up   = nn.Linear(d, hidden, bias=False)
            self.w_down = nn.Linear(hidden, d, bias=False)
            self.drop   = nn.Dropout(dr)
            nn.init.zeros_(self.w_down.weight)

        def forward(self, x):
            h = self.norm(x)
            return x + self.drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb           = nn.Embedding(VOCAB_SIZE, ENC_DIM)
            self.conv          = nn.ModuleList([ConvBlock2D(ENC_DIM), ConvBlock2D(ENC_DIM)])
            self.pos           = nn.Embedding(SEQ_LEN, ENC_DIM)
            self.ln_x          = nn.LayerNorm(ENC_DIM)
            self.ln_pos        = nn.LayerNorm(ENC_DIM)
            self.global_tokens = nn.Parameter(torch.randn(1, N_GLOBAL, ENC_DIM_MHA) * 0.02)
            self.self_attn     = MHABlock(ENC_DIM_MHA, ENC_HEADS, ENC_FF)
            self.cross_attn    = CrossAttnBlock(ENC_DIM_MHA, ENC_HEADS, ENC_FF)

            hidden = D * 3 // 2
            self.trunk = nn.ModuleList([SwiGLUBlock(D, hidden) for _ in range(n_trunk)])

            self.pol_norm = RMSNorm(D)
            self.pol_w1   = nn.Linear(D, D, bias=False)
            self.pol_w2   = nn.Linear(D, N_LEGAL, bias=False)
            self.register_buffer("sl_idx", sl_idx)

            self.val_norm = RMSNorm(D)
            self.val_w1   = nn.Linear(D, D // 2, bias=False)
            self.val_w2   = nn.Linear(D // 2, 3, bias=False)

        def forward(self, tokens):
            B = tokens.shape[0]
            x   = self.emb(tokens)                                               # [B, 64, 128]
            for blk in self.conv:
                x = blk(x)
            pos = self.pos(torch.arange(SEQ_LEN, device=tokens.device))         # [64, 128]
            x   = torch.cat([self.ln_x(x), self.ln_pos(pos).unsqueeze(0).expand(B, -1, -1)], dim=-1)  # [B, 64, 256]
            gt  = self.global_tokens.expand(B, -1, -1)                          # [B, 4, 256]
            seq = torch.cat([x, gt], dim=1)                                     # [B, 68, 256]
            seq = self.self_attn(seq)                                            # [B, 68, 256]
            g   = self.cross_attn(seq[:, -N_GLOBAL:], seq)                      # [B, 4, 256]
            x   = g.reshape(B, -1)                                              # [B, 1024]
            for blk in self.trunk:
                x = blk(x)
            legal_logits = self.pol_w2(F.silu(self.pol_w1(self.pol_norm(x))))
            pol = torch.full((B, POLICY_DIM), -3e4, device=legal_logits.device, dtype=torch.float32)
            pol.scatter_(1, self.sl_idx.unsqueeze(0).expand(B, -1), legal_logits.float())
            val = self.val_w2(F.silu(self.val_w1(self.val_norm(x))))
            return pol, val

    m = M()
    total = sum(p.numel() for p in m.parameters())
    print(f"  hybrid-conv-attn architecture:")
    print(f"    encoder : Embedding({VOCAB_SIZE},{ENC_DIM}) -> 2xConv2D({ENC_DIM}) -> cat(pos) -> [{ENC_DIM_MHA}]")
    print(f"    attn    : self-attn over [B,68,{ENC_DIM_MHA}] -> cross-attn(Q=globals[{N_GLOBAL}], KV=68)")
    print(f"    bridge  : [B,{N_GLOBAL},{ENC_DIM_MHA}] -> [B,{D}]")
    print(f"    trunk   : {n_trunk}x SwiGLUBlock(d={D}, hidden={D * 3 // 2}, dropout={dr})")
    print(f"    heads   : policy Linear({D}->{D}->{N_LEGAL}->scatter {POLICY_DIM}) | value Linear({D}->{D//2}->3)")
    print(f"    total params: {total:,}")
    return m


def build_pt_conv_gemm(cfg: dict, log_params: bool = False):
    """4x GatedConvBlock2D(128, channels-first) -> GatedPoolCompressor(12 pools)
    -> [B, 1536] -> 6x alternating GELU/SwiGLU trunk
    -> WDL head after block 4, policy Linear(1536->1858) scatter to 4288 after block 6.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    ENC_DIM    = 128
    N_POOLS    = 12
    TRUNK_DIM  = N_POOLS * ENC_DIM   # 1536
    HIDDEN     = TRUNK_DIM * 3 // 2  # 2304
    POLICY_DIM = 4288

    n_trunk = cfg["n_trunk_blocks"]
    dr      = cfg["dropout"]

    sl_mask = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool()
    sl_idx  = sl_mask.nonzero(as_tuple=True)[0]  # [1858]

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))
            self.eps   = eps
        def forward(self, x):
            return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale

    class GatedConvBlock2D(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln     = nn.LayerNorm([d, 8, 8])
            self.c_gate = nn.Conv2d(d, d, 3, padding=1, bias=False)
            self.c_up   = nn.Conv2d(d, d, 3, padding=1, bias=False)
            self.c_out  = nn.Conv2d(d, d, 3, padding=1, bias=False)
        def forward(self, x):
            h = self.ln(x)
            return x + self.c_out(F.silu(self.c_gate(h)) * self.c_up(h))

    class GatedPoolCompressor(nn.Module):
        def __init__(self, channels, n_pools):
            super().__init__()
            self.channels   = channels
            self.n_pools    = n_pools
            self.norm       = nn.LayerNorm(channels)
            self.value_proj = nn.Linear(channels, channels * n_pools, bias=False)
            self.gate_proj  = nn.Linear(channels, n_pools, bias=False)
        def forward(self, x):          # [B, channels, 8, 8] channels-first
            B = x.shape[0]
            x = x.reshape(B, self.channels, 64).permute(0, 2, 1)  # [B, 64, C]
            h = self.norm(x)
            values = self.value_proj(h).reshape(B, 64, self.n_pools, self.channels).permute(0, 2, 1, 3)
            gates  = self.gate_proj(h).permute(0, 2, 1).softmax(dim=-1)
            pooled = (values * gates[:, :, :, None]).sum(dim=2)
            return pooled.reshape(B, self.n_pools * self.channels)

    class SwiGLUBlock(nn.Module):
        def __init__(self, d, hidden):
            super().__init__()
            self.norm   = RMSNorm(d)
            self.w_gate = nn.Linear(d, hidden, bias=False)
            self.w_up   = nn.Linear(d, hidden, bias=False)
            self.w_down = nn.Linear(hidden, d, bias=False)
            self.drop   = nn.Dropout(dr)
            nn.init.zeros_(self.w_down.weight)
        def forward(self, x):
            h = self.norm(x)
            return x + self.drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))

    class GELUMlpBlock(nn.Module):
        def __init__(self, d, hidden):
            super().__init__()
            self.norm = RMSNorm(d)
            self.w1   = nn.Linear(d, hidden, bias=False)
            self.w2   = nn.Linear(hidden, d, bias=False)
            self.drop = nn.Dropout(dr)
            nn.init.zeros_(self.w2.weight)
        def forward(self, x):
            return x + self.drop(self.w2(F.gelu(self.w1(self.norm(x)))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb      = nn.Embedding(VOCAB_SIZE, ENC_DIM)
            self.conv     = nn.ModuleList([GatedConvBlock2D(ENC_DIM) for _ in range(4)])
            self.compress = GatedPoolCompressor(ENC_DIM, N_POOLS)

            def make_block(i):
                return GELUMlpBlock(TRUNK_DIM, HIDDEN) if i % 2 == 0 else SwiGLUBlock(TRUNK_DIM, HIDDEN)
            self.trunk = nn.ModuleList([make_block(i) for i in range(n_trunk)])

            self.val_norm = RMSNorm(TRUNK_DIM)
            self.val_w1   = nn.Linear(TRUNK_DIM, TRUNK_DIM // 2, bias=False)
            self.val_w2   = nn.Linear(TRUNK_DIM // 2, 3, bias=False)

            self.pol_norm = RMSNorm(TRUNK_DIM)
            self.pol_out  = nn.Linear(TRUNK_DIM, 1858, bias=False)
            self.register_buffer("sl_idx", sl_idx)

        def forward(self, tokens):
            B = tokens.shape[0]
            x = self.emb(tokens)                                             # [B, 64, 128]
            x = x.reshape(B, 8, 8, ENC_DIM).permute(0, 3, 1, 2)           # [B, 128, 8, 8]
            for blk in self.conv:
                x = blk(x)
            x = self.compress(x)                                             # [B, 1536]
            for blk in self.trunk[:4]:
                x = blk(x)
            val = self.val_w2(F.silu(self.val_w1(self.val_norm(x))))
            for blk in self.trunk[4:]:
                x = blk(x)
            legal_logits = self.pol_out(self.pol_norm(x))                   # [B, 1858]
            pol = torch.full((B, POLICY_DIM), -3e4, device=legal_logits.device, dtype=torch.float32)
            pol.scatter_(1, self.sl_idx.unsqueeze(0).expand(B, -1), legal_logits.float())
            return pol, val

    m = M()
    total = sum(p.numel() for p in m.parameters())
    print(f"  conv-gemm architecture:")
    print(f"    encoder : Embedding({VOCAB_SIZE},{ENC_DIM}) -> 4x GatedConvBlock2D({ENC_DIM}) channels-first")
    print(f"    compress: GatedPoolCompressor({N_POOLS} pools) -> [{TRUNK_DIM}]")
    print(f"    trunk   : {n_trunk}x alt GELU/SwiGLU(d={TRUNK_DIM}, hidden={HIDDEN}), WDL branch @4")
    print(f"    heads   : policy Linear({TRUNK_DIM}->1858->scatter 4288) | value Linear({TRUNK_DIM}->{TRUNK_DIM//2}->3)")
    print(f"    total params: {total:,}")
    return m


def build_pt_conv_gemm_smartgate(cfg: dict, log_params: bool = False):
    """Same conv encoder as conv-gemm but with a SplitGatedPoolCompressor (18 pools).
    Main arm (1536-d): identical 6x trunk + WDL + quality policy logits.
    Gate arm (768-d):  1x SwiGLU -> Linear(768, 1858, bias=True).
    Combined: legal_logits = quality_logits + F.logsigmoid(gate_raw).
    logsigmoid is always <= 0, so the gate is suppress-only.
    Gate bias init=4.0 -> logsigmoid(4.0) ~= -0.018 (near no-op at init).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    ENC_DIM    = 128
    N_MAIN     = 12
    N_GATE     = 6
    N_POOLS    = N_MAIN + N_GATE
    TRUNK_DIM  = N_MAIN * ENC_DIM    # 1536
    D_GATE     = N_GATE * ENC_DIM    # 768
    HIDDEN     = TRUNK_DIM * 3 // 2  # 2304
    H_GATE     = D_GATE * 3 // 2     # 1152
    POLICY_DIM = 4288

    n_trunk = cfg["n_trunk_blocks"]
    dr      = cfg["dropout"]

    sl_mask = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool()
    sl_idx  = sl_mask.nonzero(as_tuple=True)[0]  # [1858]

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))
            self.eps   = eps
        def forward(self, x):
            return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale

    class GatedConvBlock2D(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln     = nn.LayerNorm([d, 8, 8])
            self.c_gate = nn.Conv2d(d, d, 3, padding=1, bias=False)
            self.c_up   = nn.Conv2d(d, d, 3, padding=1, bias=False)
            self.c_out  = nn.Conv2d(d, d, 3, padding=1, bias=False)
        def forward(self, x):
            h = self.ln(x)
            return x + self.c_out(F.silu(self.c_gate(h)) * self.c_up(h))

    class SplitGatedPoolCompressor(nn.Module):
        def __init__(self, channels, n_main, n_gate):
            super().__init__()
            n_pools          = n_main + n_gate
            self.channels    = channels
            self.n_pools     = n_pools
            self.n_main      = n_main
            self.n_gate      = n_gate
            self.norm        = nn.LayerNorm(channels)
            self.value_proj  = nn.Linear(channels, channels * n_pools, bias=False)
            self.gate_proj   = nn.Linear(channels, n_pools, bias=False)
        def forward(self, x):          # [B, C, 8, 8]
            B = x.shape[0]
            x = x.reshape(B, self.channels, 64).permute(0, 2, 1)  # [B, 64, C]
            h = self.norm(x)
            values = self.value_proj(h).reshape(B, 64, self.n_pools, self.channels).permute(0, 2, 1, 3)
            gates  = self.gate_proj(h).permute(0, 2, 1).softmax(dim=-1)
            pooled = (values * gates[:, :, :, None]).sum(dim=2)
            main_x = pooled[:, :self.n_main, :].reshape(B, self.n_main * self.channels)
            gate_x = pooled[:, self.n_main:, :].reshape(B, self.n_gate * self.channels)
            return main_x, gate_x

    class SwiGLUBlock(nn.Module):
        def __init__(self, d, hidden):
            super().__init__()
            self.norm   = RMSNorm(d)
            self.w_gate = nn.Linear(d, hidden, bias=False)
            self.w_up   = nn.Linear(d, hidden, bias=False)
            self.w_down = nn.Linear(hidden, d, bias=False)
            self.drop   = nn.Dropout(dr)
            nn.init.zeros_(self.w_down.weight)
        def forward(self, x):
            h = self.norm(x)
            return x + self.drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))

    class GELUMlpBlock(nn.Module):
        def __init__(self, d, hidden):
            super().__init__()
            self.norm = RMSNorm(d)
            self.w1   = nn.Linear(d, hidden, bias=False)
            self.w2   = nn.Linear(hidden, d, bias=False)
            self.drop = nn.Dropout(dr)
            nn.init.zeros_(self.w2.weight)
        def forward(self, x):
            return x + self.drop(self.w2(F.gelu(self.w1(self.norm(x)))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb      = nn.Embedding(VOCAB_SIZE, ENC_DIM)
            self.conv     = nn.ModuleList([GatedConvBlock2D(ENC_DIM) for _ in range(4)])
            self.compress = SplitGatedPoolCompressor(ENC_DIM, N_MAIN, N_GATE)

            def make_block(i):
                return GELUMlpBlock(TRUNK_DIM, HIDDEN) if i % 2 == 0 else SwiGLUBlock(TRUNK_DIM, HIDDEN)
            self.trunk = nn.ModuleList([make_block(i) for i in range(n_trunk)])

            self.val_norm = RMSNorm(TRUNK_DIM)
            self.val_w1   = nn.Linear(TRUNK_DIM, TRUNK_DIM // 2, bias=False)
            self.val_w2   = nn.Linear(TRUNK_DIM // 2, 3, bias=False)

            self.pol_norm  = RMSNorm(TRUNK_DIM)
            self.pol_out   = nn.Linear(TRUNK_DIM, 1858, bias=False)

            self.gate_blk  = SwiGLUBlock(D_GATE, H_GATE)
            self.gate_norm = RMSNorm(D_GATE)
            self.gate_out  = nn.Linear(D_GATE, 1858, bias=True)
            nn.init.constant_(self.gate_out.bias, 4.0)

            self.register_buffer("sl_idx", sl_idx)

        def forward(self, tokens):
            B = tokens.shape[0]
            x = self.emb(tokens)
            x = x.reshape(B, 8, 8, ENC_DIM).permute(0, 3, 1, 2)           # [B, 128, 8, 8]
            for blk in self.conv:
                x = blk(x)
            main_x, gate_x = self.compress(x)                               # [B,1536], [B,768]
            for blk in self.trunk[:4]:
                main_x = blk(main_x)
            val = self.val_w2(F.silu(self.val_w1(self.val_norm(main_x))))
            for blk in self.trunk[4:]:
                main_x = blk(main_x)
            quality_logits = self.pol_out(self.pol_norm(main_x))            # [B, 1858]
            gate_x   = self.gate_blk(gate_x)
            gate_raw = self.gate_out(self.gate_norm(gate_x))               # [B, 1858]
            legal_logits = quality_logits + F.logsigmoid(gate_raw)
            pol = torch.full((B, POLICY_DIM), -3e4, device=legal_logits.device, dtype=torch.float32)
            pol.scatter_(1, self.sl_idx.unsqueeze(0).expand(B, -1), legal_logits.float())
            return pol, val

    m = M()
    total = sum(p.numel() for p in m.parameters())
    print(f"  conv-gemm-smartgate architecture:")
    print(f"    encoder : Embedding({VOCAB_SIZE},{ENC_DIM}) -> 4x GatedConvBlock2D({ENC_DIM}) channels-first")
    print(f"    compress: SplitGatedPoolCompressor({N_POOLS} pools: {N_MAIN} main + {N_GATE} gate)")
    print(f"    trunk   : {n_trunk}x alt GELU/SwiGLU(d={TRUNK_DIM}, hidden={HIDDEN}), WDL branch @4")
    print(f"    gate    : SwiGLUBlock({D_GATE},{H_GATE}) -> Linear({D_GATE}->1858, bias=4.0)")
    print(f"    heads   : policy quality+logsigmoid(gate) -> scatter 4288 | value -> 3")
    print(f"    total params: {total:,}")
    return m


def build_pt_conv_mha_gemm_smartgate(cfg: dict, log_params: bool = False):
    """Conv encoder -> pos-cat [B,64,256] -> append 8 accum tokens -> [B,72,256]
    -> 1x MHABlock self-attn over 72 tokens
    -> 1x CrossAttnBlock(q=accum[8], kv=72) -> [B,8,256]
    -> split: first 6 -> [B,1536] GEMM belly, last 2 -> [B,512] smartgate
    -> 1536-d GEMM trunk (alt GELU/SwiGLU), WDL@4
    -> quality policy Linear(1536->1858) + logsigmoid(gate Linear(512->1858)) -> scatter 4288
    Gate is suppress-only; bias init=4.0 -> near no-op at init.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    ENC_DIM     = 128
    ENC_DIM_MHA = 256
    ENC_HEADS   = 8
    ENC_FF      = 1024
    TRUNK_DIM   = 1536
    HIDDEN      = TRUNK_DIM * 3 // 2   # 2304
    D_GATE      = 512
    H_GATE      = D_GATE * 3 // 2      # 768
    POLICY_DIM  = 4288

    n_trunk = cfg["n_trunk_blocks"]
    dr      = cfg["dropout"]

    sl_mask = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool()
    sl_idx  = sl_mask.nonzero(as_tuple=True)[0]  # [1858]

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))
            self.eps   = eps
        def forward(self, x):
            return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale

    class ConvBlock2D(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln = nn.LayerNorm([d, 8, 8])
            self.c1 = nn.Conv2d(d, d, 3, padding=1, bias=False)
            self.c2 = nn.Conv2d(d, d, 3, padding=1, bias=False)
        def forward(self, x):
            h = F.gelu(self.c1(self.ln(x)))
            return x + self.c2(h)

    class MHABlock(nn.Module):
        def __init__(self, d, n_heads, ff_dim):
            super().__init__()
            self.ln1  = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.ln2  = nn.LayerNorm(d)
            self.ff1  = nn.Linear(d, ff_dim)
            self.ff2  = nn.Linear(ff_dim, d)
        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + h
            return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

    class CrossAttnBlock(nn.Module):
        def __init__(self, d, n_heads, ff_dim):
            super().__init__()
            self.ln_q  = nn.LayerNorm(d)
            self.ln_kv = nn.LayerNorm(d)
            self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.ln2   = nn.LayerNorm(d)
            self.ff1   = nn.Linear(d, ff_dim)
            self.ff2   = nn.Linear(ff_dim, d)
        def forward(self, q, kv):
            kv_n = self.ln_kv(kv)
            h, _ = self.attn(self.ln_q(q), kv_n, kv_n, need_weights=False)
            q = q + h
            return q + self.ff2(F.gelu(self.ff1(self.ln2(q))))

    class SwiGLUBlock(nn.Module):
        def __init__(self, d, hidden):
            super().__init__()
            self.norm   = RMSNorm(d)
            self.w_gate = nn.Linear(d, hidden, bias=False)
            self.w_up   = nn.Linear(d, hidden, bias=False)
            self.w_down = nn.Linear(hidden, d, bias=False)
            self.drop   = nn.Dropout(dr)
            nn.init.zeros_(self.w_down.weight)
        def forward(self, x):
            h = self.norm(x)
            return x + self.drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))

    class GELUMlpBlock(nn.Module):
        def __init__(self, d, hidden):
            super().__init__()
            self.norm = RMSNorm(d)
            self.w1   = nn.Linear(d, hidden, bias=False)
            self.w2   = nn.Linear(hidden, d, bias=False)
            self.drop = nn.Dropout(dr)
            nn.init.zeros_(self.w2.weight)
        def forward(self, x):
            return x + self.drop(self.w2(F.gelu(self.w1(self.norm(x)))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb    = nn.Embedding(VOCAB_SIZE, ENC_DIM)
            self.conv   = nn.ModuleList([ConvBlock2D(ENC_DIM), ConvBlock2D(ENC_DIM)])
            self.pos    = nn.Embedding(SEQ_LEN, ENC_DIM)
            self.ln_x   = nn.LayerNorm(ENC_DIM)
            self.ln_pos = nn.LayerNorm(ENC_DIM)

            self.accum_tokens = nn.Parameter(torch.randn(1, 8, ENC_DIM_MHA) * 0.02)
            self.self_attn    = MHABlock(ENC_DIM_MHA, ENC_HEADS, ENC_FF)
            self.cross_attn   = CrossAttnBlock(ENC_DIM_MHA, ENC_HEADS, ENC_FF)

            def make_block(i):
                return GELUMlpBlock(TRUNK_DIM, HIDDEN) if i % 2 == 0 else SwiGLUBlock(TRUNK_DIM, HIDDEN)
            self.trunk = nn.ModuleList([make_block(i) for i in range(n_trunk)])

            self.val_norm = RMSNorm(TRUNK_DIM)
            self.val_w1   = nn.Linear(TRUNK_DIM, TRUNK_DIM // 2, bias=False)
            self.val_w2   = nn.Linear(TRUNK_DIM // 2, 3, bias=False)

            self.pol_norm  = RMSNorm(TRUNK_DIM)
            self.pol_out   = nn.Linear(TRUNK_DIM, 1858, bias=False)

            self.gate_blk  = SwiGLUBlock(D_GATE, H_GATE)
            self.gate_norm = RMSNorm(D_GATE)
            self.gate_out  = nn.Linear(D_GATE, 1858, bias=True)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.constant_(self.gate_out.bias, 4.0)

            self.register_buffer("sl_idx", sl_idx)

        def forward(self, tokens):
            B = tokens.shape[0]
            x = self.emb(tokens)                                                       # [B, 64, 128]
            x = x.reshape(B, 8, 8, ENC_DIM).permute(0, 3, 1, 2)                     # [B, 128, 8, 8]
            for blk in self.conv:
                x = blk(x)
            x   = x.reshape(B, ENC_DIM, SEQ_LEN).permute(0, 2, 1)                    # [B, 64, 128]
            pos = self.pos(torch.arange(SEQ_LEN, device=tokens.device))
            x   = torch.cat([self.ln_x(x), self.ln_pos(pos).unsqueeze(0).expand(B, -1, -1)], dim=-1)  # [B, 64, 256]
            seq = torch.cat([x, self.accum_tokens.expand(B, -1, -1)], dim=1)          # [B, 72, 256]
            seq = self.self_attn(seq)                                                   # [B, 72, 256]
            g   = self.cross_attn(seq[:, -8:], seq)                                    # [B, 8, 256]
            main_x = g[:, :6, :].reshape(B, -1)                                        # [B, 1536]
            gate_x = g[:, 6:, :].reshape(B, -1)                                        # [B, 512]
            for blk in self.trunk[:4]:
                main_x = blk(main_x)
            val = self.val_w2(F.silu(self.val_w1(self.val_norm(main_x))))
            for blk in self.trunk[4:]:
                main_x = blk(main_x)
            quality_logits = self.pol_out(self.pol_norm(main_x))                       # [B, 1858]
            gate_x   = self.gate_blk(gate_x)
            gate_raw = self.gate_out(self.gate_norm(gate_x))                           # [B, 1858]
            legal_logits = quality_logits + F.logsigmoid(gate_raw)
            pol = torch.full((B, POLICY_DIM), -3e4, device=legal_logits.device, dtype=torch.float32)
            pol.scatter_(1, self.sl_idx.unsqueeze(0).expand(B, -1), legal_logits.float())
            return pol, val

    m = M()
    total = sum(p.numel() for p in m.parameters())
    print(f"  conv-mha-gemm-smartgate architecture:")
    print(f"    encoder : Embedding({VOCAB_SIZE},{ENC_DIM}) -> 2xConvBlock2D({ENC_DIM}) channels-first -> cat(pos) -> [{ENC_DIM_MHA}]")
    print(f"    attn    : append 8 accum tokens -> MHABlock(72) -> CrossAttnBlock(q=8, kv=72)")
    print(f"    split   : first 6x256 -> [{TRUNK_DIM}] | last 2x256 -> [{D_GATE}]")
    print(f"    trunk   : {n_trunk}x alt GELU/SwiGLU(d={TRUNK_DIM}, hidden={HIDDEN}), WDL branch @4")
    print(f"    gate    : SwiGLUBlock({D_GATE},{H_GATE}) -> Linear({D_GATE}->1858, bias=4.0)")
    print(f"    heads   : policy quality+logsigmoid(gate) -> scatter 4288 | value -> 3")
    print(f"    total params: {total:,}")
    return m


def build_pt_precond_smartgate(cfg: dict, log_params: bool = False):
    """4x Conv(256) preconditioner -> concat pos(256) -> 512-d
    -> append 8 specialized global accumulators -> 4x transformer blocks
    -> shared trunk_ln -> WDL(acc 0) + SmartGate(acc 1) + from/to MHA policy(acc 2-7).
    Gate is suppress-only (logsigmoid <= 0); bias init=4.0 -> near no-op at init.

    If cfg["lc0_input"] is set, the token embedding stem is replaced by a 1x1
    conv over (B, 112, 8, 8) lc0 planes; everything downstream is identical.
    Used for the lc0-vs-xc0 encoding bake-off.

    If cfg["xc0h_K"] is set (xc0h = xc0 with history), the stem instead consumes
    the flat compact history-token encoding from pyfastchess's
    board.history_tokens(K) / MCTSForest.get_all_history_tokens(K), K = cfg["xc0h_K"]:
    K frames of 64 slim tokens + K repetition flags + castling/stm/hmc, flattened
    to (B, K*64+K+3). A shared per-square token embedding (+ learned per-frame-age
    bias) plus a learned castling plane and scaled stm/hmc/rep channels are
    concatenated per square and projected to CF; everything downstream is
    identical to the xc0/lc0 stems. See PLANS.md section 4.
    """
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    CF      = cfg["conv_filters"]   # 256
    add_pos = bool(cfg.get("add_pos", False))
    D       = CF if add_pos else CF * 2   # 256 if pos is added, else 512 (pos concatenated)
    nh  = cfg["num_heads"]      # 8
    dr  = cfg["dropout"]
    interleave_n = cfg.get("interleave_n")   # if set: N x (1 MHA block, 2 conv blocks)
    pb  = 0 if interleave_n else cfg["pre_blocks"]     # 4
    PDH = 256
    lc0_input = bool(cfg.get("lc0_input", False))
    XC0H_K    = cfg.get("xc0h_K")      # None -> not xc0h; presence of K IS the flag
    xc0h_input = XC0H_K is not None
    XC0H_VOCAB = 15   # 0=empty, 1-6 us P/N/B/R/Q/K, 7-12 them P/N/B/R/Q/K, 13=EP, 14=PAD
    XC0H_DEMB  = 18

    sl_idx = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0]

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))
            self.eps   = eps
        def forward(self, x):
            xf = x.float()
            n  = xf / xf.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
            return (n * self.scale.float()).to(x.dtype)

    class ConvBlock(nn.Module):
        def __init__(self, prenorm=True):
            super().__init__()
            self.ln = make_ln2d(CF) if prenorm else None
            self.c1 = nn.Conv2d(CF, CF, 3, padding=1, bias=False)
            self.c2 = nn.Conv2d(CF, CF, 3, padding=1, bias=False)
        def forward(self, x):
            h = self.ln(x) if self.ln is not None else x
            return x + F.leaky_relu(self.c2(F.leaky_relu(self.c1(h), 0.01)), 0.01)

    class TxBlock(nn.Module):
        def __init__(self, ff_dim):
            super().__init__()
            self.ln1  = nn.LayerNorm(D)
            self.attn = nn.MultiheadAttention(D, nh, dropout=0.0, batch_first=True)
            self.drop = nn.Dropout(dr)
            self.ln2  = nn.LayerNorm(D)
            self.ff1  = nn.Linear(D, ff_dim)
            self.ff2  = nn.Linear(ff_dim, D)
        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop(h)
            return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            if lc0_input:
                self.stem    = nn.Conv2d(112, CF, 1)
                self.stem_ln = make_ln2d(CF)
            elif xc0h_input:
                xc0h_in_ch = XC0H_K * XC0H_DEMB + XC0H_K + 3   # frames + rep + castle + stm + hmc
                self.xc0h_tok_emb  = nn.Embedding(XC0H_VOCAB, XC0H_DEMB)
                with torch.no_grad():
                    d = XC0H_DEMB
                    empty    = torch.randn(d); knight   = torch.randn(d)
                    pawn     = torch.randn(d); diagonal = torch.randn(d)
                    us       = torch.randn(d); orthogon = torch.randn(d)
                    them     = torch.randn(d); king_prim= torch.randn(d)
                    pad      = torch.randn(d)
                    rows = [
                        empty,                          # 0  empty
                        us + pawn,                      # 1  us_P
                        us + knight,                    # 2  us_N
                        us + diagonal,                  # 3  us_B
                        us + orthogon,                  # 4  us_R
                        us + diagonal + orthogon,       # 5  us_Q
                        us + king_prim,                 # 6  us_K
                        them + pawn,                    # 7  them_P
                        them + knight,                  # 8  them_N
                        them + diagonal,                # 9  them_B
                        them + orthogon,                # 10 them_R
                        them + diagonal + orthogon,     # 11 them_Q
                        them + king_prim,               # 12 them_K
                        empty + them,                   # 13 EP
                        pad,                            # 14 PAD
                    ]
                    E = torch.stack(rows)
                    E = E / (E.norm(dim=1, keepdim=True) + 1e-6)
                    E = E + torch.randn_like(E) * 0.02
                    self.xc0h_tok_emb.weight.copy_(E)
                self.xc0h_cast_emb = nn.Embedding(16, 64)
                # project to CF-4; 4 spatial planes (ones, checkerboard, rank, file)
                # appended after the norm so LN can't destroy their constant/gradient
                # values. Each gets its own learnable scale so convs can attenuate any
                # plane toward zero if the geometry isn't useful.
                self.xc0h_proj    = nn.Linear(xc0h_in_ch, CF - 4)
                self.xc0h_proj_ln = nn.LayerNorm(CF - 4)

                # free learnable scales (init 1.0): 3 for rep/stm/hmc, 4 for spatial planes
                self.xc0h_rep_scale     = nn.Parameter(torch.tensor(1.0))
                self.xc0h_stm_scale     = nn.Parameter(torch.tensor(1.0))
                self.xc0h_hmc_scale     = nn.Parameter(torch.tensor(1.0))
                self.xc0h_ones_scale    = nn.Parameter(torch.tensor(1.0))
                self.xc0h_checker_scale = nn.Parameter(torch.tensor(1.0))
                self.xc0h_rank_scale    = nn.Parameter(torch.tensor(1.0))
                self.xc0h_file_scale    = nn.Parameter(torch.tensor(1.0))

                sqs = torch.arange(64).float()
                ranks_sq = sqs // 8
                files_sq = sqs % 8
                self.register_buffer("xc0h_spatial", torch.stack([
                    torch.ones(64),
                    ((ranks_sq + files_sq) % 2) * 2 - 1,
                    ranks_sq / 7.0 * 2 - 1,
                    files_sq / 7.0 * 2 - 1,
                ], dim=-1).unsqueeze(0))                                        # [1, 64, 4]
            else:
                self.emb     = nn.Embedding(VOCAB_SIZE, CF)
            self.pre     = nn.ModuleList([ConvBlock(prenorm=(i > 0)) for i in range(pb)])
            self.pos     = nn.Embedding(SEQ_LEN, CF)
            self.conv_ln = nn.LayerNorm(CF)

            self.global_tokens = nn.Parameter(torch.randn(1, 8, D) * 0.02)

            if interleave_n:
                ff_dims = [768] + [1024] * (interleave_n - 2) + [768] \
                    if interleave_n >= 2 else [768] * interleave_n
                self.interleave_tx   = nn.ModuleList([TxBlock(ffd) for ffd in ff_dims])
                self.interleave_conv = nn.ModuleList(
                    [ConvBlock(prenorm=True) for _ in range(2 * interleave_n)])
            else:
                tx_blocks = cfg.get("tx_blocks", 4)
                ff_dims   = [768] + [1024] * (tx_blocks - 2) + [768]
                self.blocks = nn.ModuleList([TxBlock(ffd) for ffd in ff_dims])
            self.trunk_ln = nn.LayerNorm(D)

            self.wdl_w1 = nn.Linear(D, D // 2, bias=False)
            self.wdl_w2 = nn.Linear(D // 2, 3, bias=False)

            self.gate_w_gate = nn.Linear(D, D * 3 // 2, bias=False)
            self.gate_w_up   = nn.Linear(D, D * 3 // 2, bias=False)
            self.gate_w_down = nn.Linear(D * 3 // 2, D, bias=False)
            self.gate_drop   = nn.Dropout(dr)
            nn.init.zeros_(self.gate_w_down.weight)
            self.gate_norm = RMSNorm(D)
            self.gate_out  = nn.Linear(D, 1858, bias=True)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.constant_(self.gate_out.bias, 4.0)

            self.from_proj = nn.Linear(D, PDH)
            self.from_ln   = nn.LayerNorm(PDH)
            self.from_mha  = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
            self.from_out  = nn.Linear(PDH, PDH)

            self.to_proj   = nn.Linear(D, PDH)
            self.to_ln     = nn.LayerNorm(PDH)
            self.to_mha    = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
            self.to_out    = nn.Linear(PDH, PDH)

            self.scale      = 1.0 / math.sqrt(PDH)
            self.promo_from = nn.Linear(PDH, 3, bias=False)
            self.promo_to   = nn.Linear(PDH, 3, bias=False)

            self.register_buffer("sl_idx", sl_idx)

        def forward(self, x_in):
            B = x_in.shape[0]
            if lc0_input:
                x = self.stem_ln(self.stem(x_in))                             # [B, CF, 8, 8]
            elif xc0h_input:
                K = XC0H_K
                # model's running dtype (fp16 under half() inference, fp32 in training) --
                # rep/stm/hmc are cast here and run natively in that dtype throughout,
                # same as everywhere else in this model past the embed.
                dtype = self.xc0h_tok_emb.weight.dtype
                tok  = x_in[:, :K * 64].reshape(B, K, 64).long()
                rep  = x_in[:, K * 64:K * 64 + K].to(dtype)
                cast = x_in[:, K * 64 + K].long()
                stm  = x_in[:, K * 64 + K + 1].to(dtype)
                hmc  = x_in[:, K * 64 + K + 2].to(dtype)

                tok_emb = self.xc0h_tok_emb(tok)                                    # [B,K,64,d]
                tok_emb = tok_emb.permute(0, 2, 1, 3).reshape(B, 64, K * XC0H_DEMB)  # [B,64,K*d]

                # rep/stm/hmc are raw 0/1 (or unbounded for hmc) -- rescale to roughly
                # the same [-1,1] order of magnitude as the (~unit-variance) embeddings
                # so none of them get washed out or dominate at the projection below.
                # Each also gets its own free learnable scale (init 1.0).
                rep_planes = (rep * 2.0 - 1.0).unsqueeze(1).expand(B, 64, K) * self.xc0h_rep_scale
                cast_plane = self.xc0h_cast_emb(cast).unsqueeze(-1)                 # [B,64,1]
                stm_plane  = (stm * 2.0 - 1.0).view(B, 1, 1).expand(B, 64, 1) * self.xc0h_stm_scale
                hmc_plane  = (hmc / 99.0 * 2.0 - 1.0).view(B, 1, 1).expand(B, 64, 1) * self.xc0h_hmc_scale

                fused = torch.cat(
                    [tok_emb, rep_planes, cast_plane, stm_plane, hmc_plane], dim=-1)  # [B,64,in_ch]
                projected = self.xc0h_proj_ln(self.xc0h_proj(fused))                # [B, 64, CF-4]
                scales  = torch.stack([self.xc0h_ones_scale, self.xc0h_checker_scale,
                                       self.xc0h_rank_scale, self.xc0h_file_scale])
                spatial = self.xc0h_spatial.to(projected.dtype).expand(B, -1, -1) * scales  # [B, 64, 4]
                x = torch.cat([projected, spatial], dim=-1)                          # [B, 64, CF]
                x = x.reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()         # [B, CF, 8, 8]
            else:
                x = self.emb(x_in).reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()
            for blk in self.pre:
                x = blk(x)
            seq = self.conv_ln(x.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, CF))
            pos = self.pos(torch.arange(SEQ_LEN, device=x_in.device)).unsqueeze(0).expand(B, -1, -1)
            x = (seq + pos) if add_pos else torch.cat([seq, pos], dim=-1)          # [B, 64, D]
            x = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)    # [B, 72, D]
            if interleave_n:
                for i, tx in enumerate(self.interleave_tx):
                    x = tx(x)
                    board, glob = x[:, :SEQ_LEN], x[:, SEQ_LEN:]
                    board = board.reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()
                    board = self.interleave_conv[2 * i](board)
                    board = self.interleave_conv[2 * i + 1](board)
                    board = board.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, CF)
                    x = torch.cat([board, glob], dim=1)
            else:
                for blk in self.blocks:
                    x = blk(x)
            x = self.trunk_ln(x)                                                # [B, 72, D]

            wdl = self.wdl_w2(F.gelu(self.wdl_w1(x[:, 64, :])))               # [B, 3]

            gi       = x[:, 65, :]
            h        = F.silu(self.gate_w_gate(gi)) * self.gate_w_up(gi)
            g        = gi + self.gate_drop(self.gate_w_down(h))
            gate_raw = self.gate_out(self.gate_norm(g))                         # [B, 1858]

            from_set = torch.cat([x[:, :64, :], x[:, 66:68, :], x[:, 70:72, :]], dim=1)  # [B, 68, D]
            to_set   = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:72, :]], dim=1)  # [B, 68, D]

            f_proj = F.gelu(self.from_proj(from_set))
            fn     = self.from_ln(f_proj)
            fh, _  = self.from_mha(fn[:, :64, :], fn, fn, need_weights=False)
            fv     = self.from_out(f_proj[:, :64, :] + fh)                     # [B, 64, PDH]

            t_proj = F.gelu(self.to_proj(to_set))
            tn     = self.to_ln(t_proj)
            th, _  = self.to_mha(tn[:, :64, :], tn, tn, need_weights=False)
            tv     = self.to_out(t_proj[:, :64, :] + th)                       # [B, 64, PDH]

            dots_full = torch.bmm(fv, tv.transpose(1, 2)).mul(self.scale)          # [B, 64, 64]
            dots      = dots_full.reshape(B, 64 * 64)

            dots_sub = dots_full[:, 48:56, 56:64]                                   # [B, 8, 8]
            pf       = self.promo_from(fv[:, 48:56, :])                             # [B, 8, 3]
            pt       = self.promo_to(tv[:, 56:64, :])                               # [B, 8, 3]
            promo    = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]).permute(0, 3, 2, 1).reshape(B, 192).float()

            raw_4288 = torch.cat([dots, promo], dim=1)
            q_sl     = raw_4288[:, self.sl_idx]
            combined = q_sl + F.logsigmoid(gate_raw.float())
            return combined.to(x.dtype), wdl

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_conv_shallow_mha(cfg: dict, log_params: bool = False):
    """2x Conv(128) -> LN(x) -> concat(x, x, pos(128)) -> 384-d
    -> append 8 global accumulators -> 8x TxBlock(d=384, ff_dim=1024)
    -> trunk_ln -> WDL(acc 0) + SmartGate(acc 1) + from/to MHA policy(acc 2-7) at 384-d.
    SmartGate intermediate dim stays at 768 (same as 16m-precond-smartgate).
    """
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    CF     = cfg["conv_filters"]   # 128
    D      = CF * 3                # 384  (LN(x) + LN(x) + pos(128))
    nh     = cfg["num_heads"]      # 8
    dr     = cfg["dropout"]
    mb     = cfg["mha_blocks"]     # 8
    PDH    = D                     # 384, no downstep in policy heads
    GATE_H = 768

    sl_idx = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0]

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))
            self.eps   = eps
        def forward(self, x):
            return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale

    class ConvBlock(nn.Module):
        def __init__(self, prenorm=True):
            super().__init__()
            self.ln = make_ln2d(CF) if prenorm else None
            self.c1 = nn.Conv2d(CF, CF, 3, padding=1, bias=False)
            self.c2 = nn.Conv2d(CF, CF, 3, padding=1, bias=False)
        def forward(self, x):
            h = self.ln(x) if self.ln is not None else x
            return x + F.leaky_relu(self.c2(F.leaky_relu(self.c1(h), 0.01)), 0.01)

    class TxBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1  = nn.LayerNorm(D)
            self.attn = nn.MultiheadAttention(D, nh, dropout=0.0, batch_first=True)
            self.drop = nn.Dropout(dr)
            self.ln2  = nn.LayerNorm(D)
            self.ff1  = nn.Linear(D, 1024)
            self.ff2  = nn.Linear(1024, D)
        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop(h)
            return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb     = nn.Embedding(VOCAB_SIZE, CF)
            self.pre     = nn.ModuleList([ConvBlock(prenorm=(i > 0)) for i in range(2)])
            self.pos     = nn.Embedding(SEQ_LEN, CF)
            self.conv_ln = nn.LayerNorm(CF)

            self.global_tokens = nn.Parameter(torch.randn(1, 8, D) * 0.02)

            self.blocks   = nn.ModuleList([TxBlock() for _ in range(mb)])
            self.trunk_ln = nn.LayerNorm(D)

            self.wdl_w1 = nn.Linear(D, D // 2, bias=False)
            self.wdl_w2 = nn.Linear(D // 2, 3, bias=False)

            self.gate_w_gate = nn.Linear(D, GATE_H, bias=False)
            self.gate_w_up   = nn.Linear(D, GATE_H, bias=False)
            self.gate_w_down = nn.Linear(GATE_H, D, bias=False)
            self.gate_drop   = nn.Dropout(dr)
            nn.init.zeros_(self.gate_w_down.weight)
            self.gate_norm = RMSNorm(D)
            self.gate_out  = nn.Linear(D, 1858, bias=True)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.constant_(self.gate_out.bias, 4.0)

            self.from_proj = nn.Linear(D, PDH)
            self.from_ln   = nn.LayerNorm(PDH)
            self.from_mha  = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
            self.from_out  = nn.Linear(PDH, PDH)

            self.to_proj   = nn.Linear(D, PDH)
            self.to_ln     = nn.LayerNorm(PDH)
            self.to_mha    = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
            self.to_out    = nn.Linear(PDH, PDH)

            self.scale      = 1.0 / math.sqrt(PDH)
            self.promo_from = nn.Linear(PDH, 3, bias=False)
            self.promo_to   = nn.Linear(PDH, 3, bias=False)

            self.register_buffer("sl_idx", sl_idx)

        def forward(self, tokens):
            B = tokens.shape[0]
            x = self.emb(tokens).reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()
            for blk in self.pre:
                x = blk(x)
            seq = self.conv_ln(x.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, CF))
            pos = self.pos(torch.arange(SEQ_LEN, device=tokens.device)).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([seq, seq, pos], dim=-1)                             # [B, 64, 384]
            x = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)   # [B, 72, 384]
            for blk in self.blocks:
                x = blk(x)
            x = self.trunk_ln(x)                                               # [B, 72, 384]

            wdl = self.wdl_w2(F.gelu(self.wdl_w1(x[:, 64, :])))               # [B, 3]

            gi       = x[:, 65, :]
            h        = F.silu(self.gate_w_gate(gi)) * self.gate_w_up(gi)
            g        = gi + self.gate_drop(self.gate_w_down(h))
            gate_raw = self.gate_out(self.gate_norm(g))                         # [B, 1858]

            from_set = torch.cat([x[:, :64, :], x[:, 66:68, :], x[:, 70:72, :]], dim=1)  # [B, 68, 384]
            to_set   = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:72, :]], dim=1)  # [B, 68, 384]

            f_proj = F.gelu(self.from_proj(from_set))
            fn     = self.from_ln(f_proj)
            fh, _  = self.from_mha(fn[:, :64, :], fn, fn, need_weights=False)
            fv     = self.from_out(f_proj[:, :64, :] + fh)                     # [B, 64, 256]

            t_proj = F.gelu(self.to_proj(to_set))
            tn     = self.to_ln(t_proj)
            th, _  = self.to_mha(tn[:, :64, :], tn, tn, need_weights=False)
            tv     = self.to_out(t_proj[:, :64, :] + th)                       # [B, 64, 256]

            dots_full = torch.bmm(fv, tv.transpose(1, 2)).mul(self.scale)      # [B, 64, 64]
            dots      = dots_full.reshape(B, 64 * 64)

            dots_sub = dots_full[:, 48:56, 56:64]                              # [B, 8, 8]
            pf       = self.promo_from(fv[:, 48:56, :])                        # [B, 8, 3]
            pt       = self.promo_to(tv[:, 56:64, :])                          # [B, 8, 3]
            promo    = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]).permute(0, 3, 2, 1).reshape(B, 192).float()

            raw_4288 = torch.cat([dots, promo], dim=1)
            q_sl     = raw_4288[:, self.sl_idx]
            combined = q_sl + F.logsigmoid(gate_raw.float())
            return combined.to(x.dtype), wdl

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_conv_pure(cfg: dict, log_params: bool = False):
    """N x ConvBlock(conv_filters) channels-first (NCHW) backbone -> trunk_ln,
    N = cfg["num_blocks"] (currently 16 in model_variant_speed_test_pt.py's CONV_PURE_CFG).
    WDL:       Conv(256,8) -> leaky_relu -> reshape [B,512] -> GELU -> Linear(512,3).
    SmartGate: Conv(256,16,3x3) -> leaky_relu -> reshape [B,1024] -> Linear(1024,512)
               -> GELU -> LayerNorm(512) -> SwiGLU(512,768,512) -> RMSNorm -> Linear(512,1858),
               same D=512 SwiGLU/RMSNorm/gate_out shape and zero-init/bias=4.0 convention
               as the precond-smartgate family's gate.
    Policy:    board tokens [B,64,256] -> from/to MHA at PDH=256 -> bilinear -> smartgate -> 1858.
    """
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    D       = cfg["conv_filters"]   # 128
    nb      = cfg["num_blocks"]
    dr      = cfg["dropout"]
    PDH     = D                     # 128
    GATE_D  = 512
    GATE_H  = GATE_D * 3 // 2

    sl_idx = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0]

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))
            self.eps   = eps
        def forward(self, x):
            return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale

    class ConvBlock(nn.Module):
        def __init__(self, prenorm=True):
            super().__init__()
            self.ln = make_ln2d(D) if prenorm else None
            self.c1 = nn.Conv2d(D, D, 3, padding=1, bias=False)
            self.c2 = nn.Conv2d(D, D, 3, padding=1, bias=False)
        def forward(self, x):
            h = self.ln(x) if self.ln is not None else x
            return x + F.leaky_relu(self.c2(F.leaky_relu(self.c1(h), 0.01)), 0.01)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb      = nn.Embedding(VOCAB_SIZE, D)
            self.blocks   = nn.ModuleList([ConvBlock(prenorm=(i > 0)) for i in range(nb)])
            self.trunk_ln = make_ln2d(D)

            self.wdl_conv = nn.Conv2d(D, 8, 1, bias=False)
            self.wdl_out  = nn.Linear(512, 3, bias=False)

            self.gate_conv    = nn.Conv2d(D, 16, 3, padding=1, bias=False)   # [B,16,8,8]
            self.gate_proj    = nn.Linear(16 * 64, GATE_D)                    # 1024 -> 512
            self.gate_proj_ln = nn.LayerNorm(GATE_D)

            self.gate_w_gate = nn.Linear(GATE_D, GATE_H, bias=False)
            self.gate_w_up   = nn.Linear(GATE_D, GATE_H, bias=False)
            self.gate_w_down = nn.Linear(GATE_H, GATE_D, bias=False)
            self.gate_drop   = nn.Dropout(dr)
            nn.init.zeros_(self.gate_w_down.weight)
            self.gate_norm = RMSNorm(GATE_D)
            self.gate_out  = nn.Linear(GATE_D, 1858, bias=True)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.constant_(self.gate_out.bias, 4.0)

            self.from_proj = nn.Linear(D, PDH)
            self.from_ln   = nn.LayerNorm(PDH)
            self.from_mha  = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
            self.from_out  = nn.Linear(PDH, PDH)

            self.to_proj   = nn.Linear(D, PDH)
            self.to_ln     = nn.LayerNorm(PDH)
            self.to_mha    = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
            self.to_out    = nn.Linear(PDH, PDH)

            self.scale      = 1.0 / math.sqrt(PDH)
            self.promo_from = nn.Linear(PDH, 3, bias=False)
            self.promo_to   = nn.Linear(PDH, 3, bias=False)

            self.register_buffer("sl_idx", sl_idx)

        def forward(self, tokens):
            B = tokens.shape[0]
            x = self.emb(tokens).reshape(B, 8, 8, D).permute(0, 3, 1, 2).contiguous()
            for blk in self.blocks:
                x = blk(x)
            trunk = self.trunk_ln(x)                                            # [B, D, 8, 8]

            wdl_h = F.leaky_relu(self.wdl_conv(trunk), 0.01).reshape(B, 512)
            wdl   = self.wdl_out(F.gelu(wdl_h))                                # [B, 3]

            gc = F.leaky_relu(self.gate_conv(trunk), 0.01).reshape(B, 16 * 64)  # [B, 1024]
            gi = self.gate_proj_ln(F.gelu(self.gate_proj(gc)))                  # [B, 512]
            h  = F.silu(self.gate_w_gate(gi)) * self.gate_w_up(gi)
            g        = gi + self.gate_drop(self.gate_w_down(h))
            gate_raw = self.gate_out(self.gate_norm(g))                         # [B, 1858]

            board  = trunk.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, D)         # [B, 64, 256]

            f_proj = F.gelu(self.from_proj(board))
            fn     = self.from_ln(f_proj)
            fh, _  = self.from_mha(fn, fn, fn, need_weights=False)
            fv     = self.from_out(f_proj + fh)                                # [B, 64, 128]

            t_proj = F.gelu(self.to_proj(board))
            tn     = self.to_ln(t_proj)
            th, _  = self.to_mha(tn, tn, tn, need_weights=False)
            tv     = self.to_out(t_proj + th)                                  # [B, 64, 128]

            dots_full = torch.bmm(fv, tv.transpose(1, 2)).mul(self.scale)      # [B, 64, 64]
            dots      = dots_full.reshape(B, 64 * 64)

            dots_sub = dots_full[:, 48:56, 56:64]
            pf       = self.promo_from(fv[:, 48:56, :])
            pt       = self.promo_to(tv[:, 56:64, :])
            promo    = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]).permute(0, 3, 2, 1).reshape(B, 192).float()

            raw_4288 = torch.cat([dots.float(), promo], dim=1)
            q_sl     = raw_4288[:, self.sl_idx]
            combined = q_sl + F.logsigmoid(gate_raw.float())
            return combined.to(trunk.dtype), wdl

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_full_mha_smartgate(cfg: dict, log_params: bool = False):
    """No conv. Embedding(256) + pos(128) -> cat -> 384-d [B,64,384]
    -> append 8 global accum tokens -> 7x TxBlock(d=384) -> Linear(384->512) expander
    -> 1x TxBlock(d=512) -> trunk_ln(512)
    -> same WDL/SmartGate/from-to-policy heads as 16m-precond-smartgate at 512-d.
    """
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    D_SMALL = 384   # 256 tok + 128 pos
    D_LARGE = 512   # after expander, used by last block and all heads
    nh  = cfg["num_heads"]   # 8
    dr  = cfg["dropout"]
    PDH = 256

    sl_idx = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0]

    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))
            self.eps   = eps
        def forward(self, x):
            return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale

    class TxBlock(nn.Module):
        def __init__(self, d, ff_dim):
            super().__init__()
            self.ln1  = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, nh, dropout=0.0, batch_first=True)
            self.drop = nn.Dropout(dr)
            self.ln2  = nn.LayerNorm(d)
            self.ff1  = nn.Linear(d, ff_dim)
            self.ff2  = nn.Linear(ff_dim, d)
        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop(h)
            return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, 256)
            self.pos     = nn.Embedding(SEQ_LEN, 128)
            self.tok_ln  = nn.LayerNorm(256)
            self.pos_ln  = nn.LayerNorm(128)

            self.global_tokens = nn.Parameter(torch.randn(1, 8, D_SMALL) * 0.02)

            small_ff_dims = [512, 768, 768, 768, 768, 768, 768]
            self.small_blocks = nn.ModuleList([TxBlock(D_SMALL, ffd) for ffd in small_ff_dims])
            self.expander     = nn.Linear(D_SMALL, D_LARGE, bias=False)
            self.expand_ln    = nn.LayerNorm(D_LARGE)
            self.large_block  = TxBlock(D_LARGE, 512)
            self.trunk_ln     = nn.LayerNorm(D_LARGE)

            self.wdl_w1 = nn.Linear(D_LARGE, D_LARGE // 2, bias=False)
            self.wdl_w2 = nn.Linear(D_LARGE // 2, 3, bias=False)

            self.gate_w_gate = nn.Linear(D_LARGE, D_LARGE * 3 // 2, bias=False)
            self.gate_w_up   = nn.Linear(D_LARGE, D_LARGE * 3 // 2, bias=False)
            self.gate_w_down = nn.Linear(D_LARGE * 3 // 2, D_LARGE, bias=False)
            self.gate_drop   = nn.Dropout(dr)
            nn.init.zeros_(self.gate_w_down.weight)
            self.gate_norm = RMSNorm(D_LARGE)
            self.gate_out  = nn.Linear(D_LARGE, 1858, bias=True)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.constant_(self.gate_out.bias, 4.0)

            self.from_proj = nn.Linear(D_LARGE, PDH)
            self.from_ln   = nn.LayerNorm(PDH)
            self.from_mha  = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
            self.from_out  = nn.Linear(PDH, PDH)

            self.to_proj   = nn.Linear(D_LARGE, PDH)
            self.to_ln     = nn.LayerNorm(PDH)
            self.to_mha    = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
            self.to_out    = nn.Linear(PDH, PDH)

            self.scale      = 1.0 / math.sqrt(PDH)
            self.promo_from = nn.Linear(PDH, 3, bias=False)
            self.promo_to   = nn.Linear(PDH, 3, bias=False)

            self.register_buffer("sl_idx", sl_idx)

        def forward(self, tokens):
            B = tokens.shape[0]
            pos = self.pos(torch.arange(SEQ_LEN, device=tokens.device)).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([self.tok_ln(self.tok_emb(tokens)), self.pos_ln(pos)], dim=-1)  # [B, 64, 384]
            x = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)               # [B, 72, 384]
            for blk in self.small_blocks:
                x = blk(x)
            x = self.expand_ln(self.expander(x))                                           # [B, 72, 512]
            x = self.large_block(x)
            x = self.trunk_ln(x)

            wdl = self.wdl_w2(F.gelu(self.wdl_w1(x[:, 64, :])))

            gi       = x[:, 65, :]
            h        = F.silu(self.gate_w_gate(gi)) * self.gate_w_up(gi)
            g        = gi + self.gate_drop(self.gate_w_down(h))
            gate_raw = self.gate_out(self.gate_norm(g))

            from_set = torch.cat([x[:, :64, :], x[:, 66:68, :], x[:, 70:72, :]], dim=1)  # [B, 68, 512]
            to_set   = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:72, :]], dim=1)  # [B, 68, 512]

            f_proj = F.gelu(self.from_proj(from_set))
            fn     = self.from_ln(f_proj)
            fh, _  = self.from_mha(fn[:, :64, :], fn, fn, need_weights=False)
            fv     = self.from_out(f_proj[:, :64, :] + fh)

            t_proj = F.gelu(self.to_proj(to_set))
            tn     = self.to_ln(t_proj)
            th, _  = self.to_mha(tn[:, :64, :], tn, tn, need_weights=False)
            tv     = self.to_out(t_proj[:, :64, :] + th)

            dots_full = torch.bmm(fv, tv.transpose(1, 2)).mul(self.scale)
            dots      = dots_full.reshape(B, 64 * 64)

            dots_sub = dots_full[:, 48:56, 56:64]
            pf       = self.promo_from(fv[:, 48:56, :])
            pt       = self.promo_to(tv[:, 56:64, :])
            promo    = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]).permute(0, 3, 2, 1).reshape(B, 192).float()

            raw_4288 = torch.cat([dots, promo], dim=1)
            q_sl     = raw_4288[:, self.sl_idx]
            combined = q_sl + F.logsigmoid(gate_raw.float())
            return combined.to(x.dtype), wdl

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


PT_BUILDERS: dict[str, object] = {
    "16m-transformer":           build_pt_transformer_16m,
    "16m-conformer-interweaved": build_pt_conformer_interweaved,
    "13m-precond-conformer":     build_pt_precond_conformer,
    "16m-precond-smartgate":                build_pt_precond_smartgate,
    "16m-precond-smartgate-lc0":            build_pt_precond_smartgate,
    "16m-precond-smartgate-xc0h":           build_pt_precond_smartgate,
    "18m-precond-smartgate-xc0h-6c4t":      build_pt_precond_smartgate,
    "conv-shallow-mha":                      build_pt_conv_shallow_mha,
    "full-mha-smartgate":                   build_pt_full_mha_smartgate,
    "hybrid-conv-attn":          build_pt_hybrid_conv_attn,
    "conv-gemm":                 build_pt_conv_gemm,
    "conv-gemm-smartgate":       build_pt_conv_gemm_smartgate,
    "conv-mha-gemm-smartgate":   build_pt_conv_mha_gemm_smartgate,
}
