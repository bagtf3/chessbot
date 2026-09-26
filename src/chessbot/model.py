"""PyTorch models for Xerces. VARIANTS holds each model's cfg, PT_BUILDERS maps
the same names to fn(cfg) -> nn.Module."""
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
    "precond-signed-4g6t-d512": dict(
        conv_filters=256, pos_dim=256, dropout=0.03, pre_blocks=4, tx_blocks=6,
        ff_dim=1024, pre_block="globalizer", xc0h_K=6,
    ),
    "precond-mha-4g6t-d512": dict(
        conv_filters=256, pos_dim=256, dropout=0.03, pre_blocks=4, tx_blocks=6,
        ff_dim=1024, pre_block="globalizer", xc0h_K=6,
        attn="mha", num_heads=8,
    ),
    "18m-precond-smartgate-xc0h-6c4t": dict(
        conv_filters=256, num_heads=8, dropout=0.03,
        pre_blocks=6, tx_blocks=4, xc0h_K=6,
    ),
    "precond-mixer-6m4t-d512": dict(
        conv_filters=256, pos_dim=256, dropout=0.03, mixer_blocks=6, tx_blocks=4,
        tok_expansion=4, chan_expansion=4, num_heads=8, xc0h_K=6,
    ),
    "precond-mha-8c4t-d384": dict(
        conv_filters=256, pos_dim=128, dropout=0.03, pre_blocks=8, tx_blocks=4,
        ff_dim=1024, pre_block="conv", attn="mha", num_heads=8, xc0h_K=6,
    ),
    "precond-mha-8c6t-d256": dict(
        conv_filters=256, dropout=0.03, pre_blocks=8, tx_blocks=6,
        ff_dim=1024, num_heads=8, xc0h_K=6,
    ),
    "precond-mha-10c6t-d256": dict(
        conv_filters=256, dropout=0.01, pre_blocks=10, tx_blocks=6,
        ff_dim=1024, num_heads=8, xc0h_K=6,
    ),
    "precond-mha-10c6t-d256-wdl3": dict(
        conv_filters=256, dropout=0.02, pre_blocks=10, tx_blocks=6,
        ff_dim=1024, num_heads=8, xc0h_K=6, wdl_split=True, attn_impl="sdpa",
    ),
    "precond-mha-8c8t-d256": dict(
        conv_filters=256, dropout=0.03, pre_blocks=8, tx_blocks=8,
        ff_dim=1024, num_heads=8, xc0h_K=6,
    ),
    "conv-globalizer-2c6g": dict(
        conv_filters=256, pos_dim=128, num_blocks=6, stem_blocks=2,
        dropout=0.03, xc0h_K=6,
    ),
    "conv-pure": dict(
        conv_filters=256, num_blocks=16, dropout=0.03, xc0h_K=6, value_tap=4,
    ),
    "m13-7-8c8t-d256": dict(
        conv_filters=256, num_heads=8, ff_dim=1024, xc0h_K=6,
        n_plain=2, n_gated=6, n_ff=4, n_swiglu=4, attn_gate=True, prelu=True,
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


def make_rms2d(ch: int, eps: float = 1e-6):
    """Channel-wise RMSNorm for (B, C, H, W): per square over C, scale only."""
    import torch
    import torch.nn as nn

    class Rms2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(ch))

        def forward(self, x):
            x = x.clamp(-250.0, 250.0)
            xf = x.float()
            r = torch.rsqrt(xf.pow(2).mean(1, keepdim=True) + eps)
            return (xf * r * self.w.float().view(1, -1, 1, 1)).to(x.dtype)

    return Rms2d()


def make_rms1d(d: int, eps: float = 1e-6):
    """RMSNorm over the last dim. Param is named `scale` to stay key-compatible
    with the inline RMSNorm the smartgate heads have always used."""
    import torch
    import torch.nn as nn

    class Rms1d(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))

        def forward(self, x):
            x = x.clamp(-250.0, 250.0)
            xf = x.float()
            y = xf / xf.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
            return (y * self.scale.float()).to(x.dtype)

    return Rms1d()


def make_l1norm(d: int, eps: float = 1e-6):
    """Mean-abs norm over the last dim, learned per-channel scale and bias. The
    +/-250 clamp caps the residual stream so it cannot grow past fp16 range."""
    import torch
    import torch.nn as nn

    class L1Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))
            self.b = nn.Parameter(torch.zeros(d))

        def forward(self, x):
            x = x.clamp(-250.0, 250.0)
            y = x / (x.abs().mean(-1, keepdim=True) + eps)
            return y * self.scale.to(x.dtype) + self.b.to(x.dtype)

    return L1Norm()


def make_l1p5norm(d: int, eps: float = 1e-6):
    """x / mean(|x|^1.5)^(2/3) over the last dim, per-channel scale + bias, in
    the input dtype. The custom Function saves only x and per-token stats."""
    import torch
    import torch.nn as nn

    def stats(x):
        a = x.abs()
        m = (a * (a + eps).sqrt()).mean(-1, keepdim=True) + eps
        with torch.autocast("cuda", enabled=False):
            denom = m.pow(2.0 / 3.0)
        return m, denom

    class L1point5Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale, b):
            m, denom = stats(x)
            ctx.save_for_backward(x, scale, m, denom)
            return x / denom * scale + b

        @staticmethod
        def backward(ctx, g):
            x, scale, m, denom = ctx.saved_tensors
            y = x / denom
            gs = g * scale
            c = (gs * y).sum(-1, keepdim=True) * (2.0 / (3.0 * x.shape[-1])) / m
            s = (x.abs() + eps).sqrt()
            dx = gs / denom - c * (x.sign() * s + x / (2.0 * s))
            dims = tuple(range(g.dim() - 1))
            return dx, (g * y).sum(dims), g.sum(dims)

    class L1point5Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(d))
            self.b = nn.Parameter(torch.zeros(d))

        def forward(self, x):
            scale, b = self.scale.to(x.dtype), self.b.to(x.dtype)
            if torch.is_grad_enabled() and x.requires_grad:
                return L1point5Fn.apply(x, scale, b)
            return x / stats(x)[1] * scale + b

    return L1point5Norm()


def make_rms_conv_block(D: int):
    """conv-pure residual block: rms -> conv3x3 -> leaky -> conv3x3 -> leaky -> add.
    c1 has a bias (RMSNorm has none to absorb the offset), c2 does not."""
    import torch.nn as nn
    import torch.nn.functional as F

    class RmsConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = make_rms2d(D)
            self.c1 = nn.Conv2d(D, D, 3, padding=1, bias=True)
            self.c2 = nn.Conv2d(D, D, 3, padding=1, bias=False)

        def forward(self, x):
            h = self.norm(x)
            h = F.leaky_relu(self.c1(h), 0.01)
            h = F.leaky_relu(self.c2(h), 0.01)
            return x + h

    return RmsConvBlock()


def make_globalizer_block(D: int = 256, local_ch: int = None, glob_ch: int = 16):
    """Split local/global residual block: one 3x3 conv, its last glob_ch channels
    flattened through a SwiGLU bottleneck and concatenated back, so every square
    sees whole-board state without attention. One RMSNorm."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if local_ch is None:
        local_ch = D - glob_ch        # 240 at D=256

    gdim = glob_ch * SEQ_LEN          # 1024
    hidden = gdim // 2                # 512

    class GlobalizerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = make_rms2d(D)
            self.c1 = nn.Conv2d(D, D, 3, padding=1, bias=True)
            self.lin_in = nn.Linear(gdim, gdim, bias=True)     # emits gate|value
            self.lin_out = nn.Linear(hidden, gdim, bias=True)
            self.c2 = nn.Conv2d(D, D, 3, padding=1, bias=False)

        def forward(self, x):
            B = x.shape[0]
            h = F.leaky_relu(self.c1(self.norm(x)), 0.01)      # [B, D, 8, 8]
            local, g = h[:, :local_ch], h[:, local_ch:]

            g = g.reshape(B, gdim)                             # [B, 1024]
            gate, value = self.lin_in(g).chunk(2, dim=-1)      # [B, 512] each
            g = F.silu(gate) * value
            g = self.lin_out(g).reshape(B, glob_ch, 8, 8)      # [B, 16, 8, 8]

            h = torch.cat([local, g], dim=1)                   # [B, D, 8, 8]
            h = F.leaky_relu(self.c2(h), 0.01)
            return x + h

    return GlobalizerBlock()


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
    """MHA-refined bilinear from/to policy head on (B, C, 8, 8); returns
    (B, 4288) = 4096 from/to + 192 underpromotion logits."""
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
    """4x Conv(256) preconditioner -> concat pos(256) -> 512 -> 4x prenorm
    MHA/FF blocks -> heads."""
    
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


def build_pt_precond_smartgate(cfg: dict, log_params: bool = False):
    """4x Conv preconditioner -> concat pos -> 8 global accumulators -> 4x tx ->
    WDL(acc0) + suppress-only SmartGate(acc1) + from/to MHA policy(acc2-7).
    Stem: token embed, or cfg lc0_input (112 lc0 planes), or cfg xc0h_K (history)."""
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
                attach_xc0h_stem(self, CF, XC0H_K)
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
                x = xc0h_stem_forward(self, x_in, CF, XC0H_K)                 # [B, CF, 8, 8]
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


def build_pt_precond_signed(cfg: dict, log_params: bool = False):
    """Precond skeleton with a swappable attention sublayer: cfg attn "signed"
    (tanh-bounded bilinear over 72 tokens, no softmax) or "mha". RMSNorm, flat
    FFN, free pos_dim. Name kept for checkpoint compat."""
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    CF        = cfg["conv_filters"]
    POS       = cfg.get("pos_dim", CF)
    D         = CF + POS
    dr        = cfg["dropout"]
    pb        = cfg["pre_blocks"]
    tx_blocks = cfg.get("tx_blocks", 6)
    ff_dim    = cfg.get("ff_dim", 1024)
    pre_block = cfg.get("pre_block", "globalizer")
    attn      = cfg.get("attn", "signed")
    nh        = cfg.get("num_heads", 8)
    PDH       = 256
    XC0H_K    = cfg.get("xc0h_K")
    xc0h_input = XC0H_K is not None
    signed_scale = math.sqrt(D)

    if attn not in ("signed", "mha"):
        raise ValueError(f"attn must be 'signed' or 'mha', got {attn!r}")

    sl_idx = torch.from_numpy(
        pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0]

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
            # c1 reads straight off RMS (no re-centering, so needs its own
            # bias); c2 reads a leaky_relu, no bias needed
            self.ln = make_rms2d(CF) if prenorm else None
            self.c1 = nn.Conv2d(CF, CF, 3, padding=1, bias=True)
            self.c2 = nn.Conv2d(CF, CF, 3, padding=1, bias=False)
        def forward(self, x):
            h = self.ln(x) if self.ln is not None else x
            return x + F.leaky_relu(self.c2(F.leaky_relu(self.c1(h), 0.01)), 0.01)

    class SignedTxBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = RMSNorm(D)
            # no separate key projection: W_q W_k^T collapses to one matrix,
            # and at a single head with head_dim == D the collapsed form is
            # both cheaper and higher rank than the factored one
            self.interaction = nn.Linear(D, D, bias=True)
            self.value       = nn.Linear(D, D, bias=True)
            self.drop = nn.Dropout(dr)
            self.ln2  = RMSNorm(D)
            self.ff1  = nn.Linear(D, ff_dim)
            self.ff2  = nn.Linear(ff_dim, D)

        def forward(self, x):
            n = self.ln1(x)
            q = self.interaction(n)
            v = F.silu(self.value(n))
            w = torch.tanh(torch.matmul(q, n.transpose(-1, -2)) / signed_scale)
            w = w / w.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
            x = x + self.drop(torch.matmul(w, v))
            return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

    class MhaTxBlock(nn.Module):
        """Same skeleton as SignedTxBlock with softmax MHA in the sublayer."""
        def __init__(self):
            super().__init__()
            self.ln1  = RMSNorm(D)
            self.attn = nn.MultiheadAttention(D, nh, dropout=0.0,
                                              batch_first=True, bias=True)
            self.drop = nn.Dropout(dr)
            self.ln2  = RMSNorm(D)
            self.ff1  = nn.Linear(D, ff_dim, bias=True)
            self.ff2  = nn.Linear(ff_dim, D, bias=False)

        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop(h)
            return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

    tx_block_cls = MhaTxBlock if attn == "mha" else SignedTxBlock

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            if xc0h_input:
                attach_xc0h_stem(self, CF, XC0H_K, norm_type="rms")
            else:
                self.emb = nn.Embedding(VOCAB_SIZE, CF)

            if pre_block == "globalizer":
                self.pre = nn.ModuleList(
                    [make_globalizer_block(CF) for _ in range(pb)])
            else:
                self.pre = nn.ModuleList(
                    [ConvBlock(prenorm=(i > 0)) for i in range(pb)])

            self.pos     = nn.Embedding(SEQ_LEN, POS)
            self.conv_ln = RMSNorm(CF)

            self.global_tokens = nn.Parameter(torch.randn(1, 8, D) * 0.02)
            self.blocks   = nn.ModuleList([tx_block_cls() for _ in range(tx_blocks)])
            self.trunk_ln = RMSNorm(D)

            # wdl_w1 / gate_w_gate / gate_w_up read trunk_ln directly, so they
            # take a bias now that it is RMS instead of LayerNorm. wdl_w2 and
            # gate_w_down sit after an activation, not a norm, so they keep the
            # prod head's bias=False.
            self.wdl_w1 = nn.Linear(D, D // 2, bias=True)
            self.wdl_w2 = nn.Linear(D // 2, 3, bias=False)

            self.gate_w_gate = nn.Linear(D, D * 3 // 2, bias=True)
            self.gate_w_up   = nn.Linear(D, D * 3 // 2, bias=True)
            self.gate_w_down = nn.Linear(D * 3 // 2, D, bias=False)
            self.gate_drop   = nn.Dropout(dr)
            nn.init.zeros_(self.gate_w_down.weight)
            self.gate_norm = RMSNorm(D)
            self.gate_out  = nn.Linear(D, 1858, bias=True)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.constant_(self.gate_out.bias, 4.0)

            self.from_proj = nn.Linear(D, PDH, bias=True)
            self.from_ln   = RMSNorm(PDH)
            self.from_mha  = nn.MultiheadAttention(PDH, 4, dropout=0.0,
                                                   batch_first=True, bias=True)
            self.from_out  = nn.Linear(PDH, PDH, bias=False)

            self.to_proj   = nn.Linear(D, PDH, bias=True)
            self.to_ln     = RMSNorm(PDH)
            self.to_mha    = nn.MultiheadAttention(PDH, 4, dropout=0.0,
                                                   batch_first=True, bias=True)
            self.to_out    = nn.Linear(PDH, PDH, bias=False)

            self.scale      = 1.0 / math.sqrt(PDH)
            self.promo_from = nn.Linear(PDH, 3, bias=False)
            self.promo_to   = nn.Linear(PDH, 3, bias=False)

            self.register_buffer("sl_idx", sl_idx)

        def forward(self, x_in):
            B = x_in.shape[0]
            if xc0h_input:
                x = xc0h_stem_forward(self, x_in, CF, XC0H_K)                 # [B, CF, 8, 8]
            else:
                x = self.emb(x_in).reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()
            for blk in self.pre:
                x = blk(x)
            seq = self.conv_ln(x.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, CF))
            pos = self.pos(torch.arange(SEQ_LEN, device=x_in.device)).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([seq, pos], dim=-1)                                  # [B, 64, D]
            x = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)    # [B, 72, D]
            for blk in self.blocks:
                x = blk(x)
            x = self.trunk_ln(x)                                               # [B, 72, D]

            wdl = self.wdl_w2(F.gelu(self.wdl_w1(x[:, 64, :])))                # [B, 3]

            gi       = x[:, 65, :]
            h        = F.silu(self.gate_w_gate(gi)) * self.gate_w_up(gi)
            g        = gi + self.gate_drop(self.gate_w_down(h))
            gate_raw = self.gate_out(self.gate_norm(g))                        # [B, 1858]

            from_set = torch.cat([x[:, :64, :], x[:, 66:68, :], x[:, 70:72, :]], dim=1)
            to_set   = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:72, :]], dim=1)

            f_proj = F.gelu(self.from_proj(from_set))
            fn     = self.from_ln(f_proj)
            fh, _  = self.from_mha(fn[:, :64, :], fn, fn, need_weights=False)
            fv     = self.from_out(f_proj[:, :64, :] + fh)                     # [B, 64, PDH]

            t_proj = F.gelu(self.to_proj(to_set))
            tn     = self.to_ln(t_proj)
            th, _  = self.to_mha(tn[:, :64, :], tn, tn, need_weights=False)
            tv     = self.to_out(t_proj[:, :64, :] + th)                       # [B, 64, PDH]

            dots_full = torch.bmm(fv, tv.transpose(1, 2)).mul(self.scale)      # [B, 64, 64]
            dots      = dots_full.reshape(B, 64 * 64)

            dots_sub = dots_full[:, 48:56, 56:64]                              # [B, 8, 8]
            pf       = self.promo_from(fv[:, 48:56, :])                        # [B, 8, 3]
            pt       = self.promo_to(tv[:, 56:64, :])                          # [B, 8, 3]
            promo    = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]
                        ).permute(0, 3, 2, 1).reshape(B, 192).float()

            raw_4288 = torch.cat([dots, promo], dim=1)
            q_sl     = raw_4288[:, self.sl_idx]
            combined = q_sl + F.logsigmoid(gate_raw.float())
            return combined.to(x.dtype), wdl

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_precond_mixer(cfg: dict, log_params: bool = False):
    """MLP-Mixer preconditioner (fixed learned token-mix over the 64 squares) ->
    concat pos -> 6c4t-style MHA trunk and heads, RMSNorm throughout. Linears
    reading straight off an RMSNorm carry a bias; the rest do not."""
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    CF        = cfg["conv_filters"]           # 256
    POS       = cfg.get("pos_dim", 256)
    D         = CF + POS                      # 512
    dr        = cfg["dropout"]
    mixer_blocks = cfg.get("mixer_blocks", 6)
    tx_blocks    = cfg.get("tx_blocks", 4)
    tok_mult  = cfg.get("tok_expansion", 4)
    chan_mult = cfg.get("chan_expansion", 4)
    nh        = cfg.get("num_heads", 8)
    PDH       = 256
    XC0H_K    = cfg.get("xc0h_K")
    xc0h_input = XC0H_K is not None

    tok_hidden  = SEQ_LEN * tok_mult          # 64 * 4 = 256
    chan_hidden = CF * chan_mult              # 256 * 4 = 1024

    sl_idx = torch.from_numpy(
        pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0]

    class MixerBlock(nn.Module):
        """One shared RMSNorm feeding token-mix and channel-mix in parallel,
        each added back to the residual (GPT-J-style parallel block)."""
        def __init__(self):
            super().__init__()
            self.norm      = make_rms1d(CF)
            self.tok_up    = nn.Linear(SEQ_LEN, tok_hidden, bias=True)
            self.tok_down  = nn.Linear(tok_hidden, SEQ_LEN, bias=False)
            self.chan_up   = nn.Linear(CF, chan_hidden, bias=True)
            self.chan_down = nn.Linear(chan_hidden, CF, bias=False)

        def forward(self, x):
            n = self.norm(x)                           # [B, 64, CF]

            t = n.transpose(1, 2)                      # [B, CF, 64]
            t = self.tok_down(F.gelu(self.tok_up(t)))
            t = t.transpose(1, 2)                      # [B, 64, CF]

            c = self.chan_down(F.gelu(self.chan_up(n)))

            return x + t + c

    class MhaTxBlock(nn.Module):
        def __init__(self, ff_dim):
            super().__init__()
            self.ln1  = make_rms1d(D)
            self.attn = nn.MultiheadAttention(D, nh, dropout=0.0,
                                              batch_first=True, bias=True)
            self.drop = nn.Dropout(dr)
            self.ln2  = make_rms1d(D)
            self.ff1  = nn.Linear(D, ff_dim, bias=True)
            self.ff2  = nn.Linear(ff_dim, D, bias=False)

        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop(h)
            n2 = self.ln2(x)
            return x + self.ff2(F.gelu(self.ff1(n2)))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            if xc0h_input:
                attach_xc0h_stem(self, CF, XC0H_K, norm_type="rms")
            else:
                self.emb = nn.Embedding(VOCAB_SIZE, CF)

            self.mixer = nn.ModuleList(
                [MixerBlock() for _ in range(mixer_blocks)])
            self.mixer_out_norm = make_rms1d(CF)
            self.pos = nn.Embedding(SEQ_LEN, POS)

            self.global_tokens = nn.Parameter(torch.randn(1, 8, D))
            ff_dims = [768] + [1024] * (tx_blocks - 2) + [768]
            self.blocks   = nn.ModuleList([MhaTxBlock(ffd) for ffd in ff_dims])
            self.trunk_ln = make_rms1d(D)

            self.wdl_w1 = nn.Linear(D, D // 2, bias=True)
            self.wdl_w2 = nn.Linear(D // 2, 3, bias=False)

            self.gate_w_gate = nn.Linear(D, D * 3 // 2, bias=True)
            self.gate_w_up   = nn.Linear(D, D * 3 // 2, bias=True)
            self.gate_w_down = nn.Linear(D * 3 // 2, D, bias=False)
            self.gate_drop   = nn.Dropout(dr)
            nn.init.zeros_(self.gate_w_down.weight)
            self.gate_norm = make_rms1d(D)
            self.gate_out  = nn.Linear(D, 1858, bias=True)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.constant_(self.gate_out.bias, 4.0)

            self.from_proj = nn.Linear(D, PDH, bias=True)
            self.from_ln   = make_rms1d(PDH)
            self.from_mha  = nn.MultiheadAttention(PDH, 4, dropout=0.0,
                                                   batch_first=True, bias=True)
            self.from_out  = nn.Linear(PDH, PDH, bias=False)

            self.to_proj   = nn.Linear(D, PDH, bias=True)
            self.to_ln     = make_rms1d(PDH)
            self.to_mha    = nn.MultiheadAttention(PDH, 4, dropout=0.0,
                                                   batch_first=True, bias=True)
            self.to_out    = nn.Linear(PDH, PDH, bias=False)

            self.scale      = 1.0 / math.sqrt(PDH)
            self.promo_from = nn.Linear(PDH, 3, bias=False)
            self.promo_to   = nn.Linear(PDH, 3, bias=False)

            self.register_buffer("sl_idx", sl_idx)

        def forward(self, x_in):
            B = x_in.shape[0]
            if xc0h_input:
                x = xc0h_stem_forward(self, x_in, CF, XC0H_K)          # [B,CF,8,8]
            else:
                x = self.emb(x_in).reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()
            x = x.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, CF)          # [B,64,CF]

            for blk in self.mixer:
                x = blk(x)

            pos = self.pos(torch.arange(SEQ_LEN, device=x_in.device))
            pos = pos.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([self.mixer_out_norm(x), pos], dim=-1)       # [B,64,D]
            x = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)  # [B,72,D]

            for blk in self.blocks:
                x = blk(x)
            x = self.trunk_ln(x)                                       # [B,72,D]

            wdl = self.wdl_w2(F.gelu(self.wdl_w1(x[:, 64, :])))        # [B,3]

            gi       = x[:, 65, :]
            h        = F.silu(self.gate_w_gate(gi)) * self.gate_w_up(gi)
            g        = gi + self.gate_drop(self.gate_w_down(h))
            gate_raw = self.gate_out(self.gate_norm(g))                 # [B,1858]

            from_set = torch.cat([x[:, :64, :], x[:, 66:68, :], x[:, 70:72, :]], dim=1)
            to_set   = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:72, :]], dim=1)

            f_proj = F.gelu(self.from_proj(from_set))
            fn     = self.from_ln(f_proj)
            fh, _  = self.from_mha(fn[:, :64, :], fn, fn, need_weights=False)
            fv     = self.from_out(f_proj[:, :64, :] + fh)                     # [B, 64, PDH]

            t_proj = F.gelu(self.to_proj(to_set))
            tn     = self.to_ln(t_proj)
            th, _  = self.to_mha(tn[:, :64, :], tn, tn, need_weights=False)
            tv     = self.to_out(t_proj[:, :64, :] + th)                       # [B, 64, PDH]

            dots_full = torch.bmm(fv, tv.transpose(1, 2)).mul(self.scale)      # [B, 64, 64]
            dots      = dots_full.reshape(B, 64 * 64)

            dots_sub = dots_full[:, 48:56, 56:64]                              # [B, 8, 8]
            pf       = self.promo_from(fv[:, 48:56, :])                        # [B, 8, 3]
            pt       = self.promo_to(tv[:, 56:64, :])                          # [B, 8, 3]
            promo    = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]
                        ).permute(0, 3, 2, 1).reshape(B, 192).float()

            raw_4288 = torch.cat([dots.float(), promo], dim=1)
            q_sl     = raw_4288[:, self.sl_idx]
            combined = q_sl + F.logsigmoid(gate_raw.float())
            return combined.to(x.dtype), wdl

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_precond_addpos(cfg: dict, log_params: bool = False):
    """Precond that never widens: D = conv_filters, pos added via a learned
    pos_scale (no norm), 8 accumulators, MHA trunk, RMSNorm. SmartGate reads
    acc1 cat acc6 so it stays 2*D wide."""
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    D         = cfg["conv_filters"]           # 256, no widening
    dr        = cfg["dropout"]
    pb        = cfg["pre_blocks"]
    tx_blocks = cfg.get("tx_blocks", 6)
    ff_dim    = cfg.get("ff_dim", 1024)
    nh        = cfg.get("num_heads", 8)
    PDH       = 256
    XC0H_K    = cfg.get("xc0h_K")
    xc0h_input = XC0H_K is not None
    wdl_split = cfg.get("wdl_split", False)   # 3 independent W/D/L heads off acc0
    attn_impl = cfg.get("attn_impl", "mha")   # "mha" (nn.MultiheadAttention) or "sdpa"
    norm_impl = cfg.get("norm_impl", "rms")   # "rms" or "ln" (nn.LayerNorm, fusable)
    mk1d = nn.LayerNorm if norm_impl == "ln" else make_rms1d
    mk2d = make_ln2d    if norm_impl == "ln" else make_rms2d
    stem_norm = "layernorm" if norm_impl == "ln" else "rms"

    sl_idx = torch.from_numpy(
        pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0]

    class ConvBlock(nn.Module):
        def __init__(self, prenorm=True):
            super().__init__()
            self.ln = mk2d(D) if prenorm else None
            self.c1 = nn.Conv2d(D, D, 3, padding=1, bias=True)
            self.c2 = nn.Conv2d(D, D, 3, padding=1, bias=False)

        def forward(self, x):
            h = self.ln(x) if self.ln is not None else x
            return x + F.leaky_relu(self.c2(F.leaky_relu(self.c1(h), 0.01)), 0.01)

    class SdpaMHA(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.h  = heads
            self.dh = dim // heads
            self.q  = nn.Linear(dim, dim, bias=True)
            self.k  = nn.Linear(dim, dim, bias=True)
            self.v  = nn.Linear(dim, dim, bias=True)
            self.o  = nn.Linear(dim, dim, bias=True)

        def forward(self, q_in, kv_in):
            B, Sq, dim = q_in.shape
            Sk = kv_in.shape[1]
            q = self.q(q_in).view(B, Sq, self.h, self.dh).transpose(1, 2)
            k = self.k(kv_in).view(B, Sk, self.h, self.dh).transpose(1, 2)
            v = self.v(kv_in).view(B, Sk, self.h, self.dh).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v)
            return self.o(a.transpose(1, 2).reshape(B, Sq, dim))

    class MhaTxBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1  = mk1d(D)
            if attn_impl == "sdpa":
                self.attn = SdpaMHA(D, nh)
            else:
                self.attn = nn.MultiheadAttention(D, nh, dropout=0.0,
                                                  batch_first=True, bias=True)
            self.drop = nn.Dropout(dr)
            self.ln2  = mk1d(D)
            self.ff1  = nn.Linear(D, ff_dim, bias=True)
            self.ff2  = nn.Linear(ff_dim, D, bias=False)

        def forward(self, x):
            n = self.ln1(x)
            if attn_impl == "sdpa":
                h = self.attn(n, n)
            else:
                h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop(h)
            return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            if xc0h_input:
                attach_xc0h_stem(self, D, XC0H_K, norm_type=stem_norm)
            else:
                self.emb = nn.Embedding(VOCAB_SIZE, D)

            self.pre = nn.ModuleList(
                [ConvBlock(prenorm=(i > 0)) for i in range(pb)])

            self.pos       = nn.Embedding(SEQ_LEN, D)
            self.pos_scale = nn.Parameter(torch.tensor(1.0))

            self.global_tokens = nn.Parameter(torch.randn(1, 8, D) * 0.02)
            self.blocks   = nn.ModuleList([MhaTxBlock() for _ in range(tx_blocks)])
            self.trunk_ln = mk1d(D)

            if wdl_split:
                vh = 64   # per-head readout width; gives each W/D/L head real capacity
                self.wdl_hw = nn.Linear(D, vh, bias=True)
                self.wdl_hd = nn.Linear(D, vh, bias=True)
                self.wdl_hl = nn.Linear(D, vh, bias=True)
                self.wdl_ow = nn.Linear(vh, 1, bias=False)
                self.wdl_od = nn.Linear(vh, 1, bias=False)
                self.wdl_ol = nn.Linear(vh, 1, bias=False)
            else:
                self.wdl_w1 = nn.Linear(D, D // 2, bias=True)
                self.wdl_w2 = nn.Linear(D // 2, 3, bias=False)

            # SmartGate reads acc(1) concat acc(6), so it works at 2*D
            # regardless of how narrow the trunk itself runs
            GD = D * 2
            self.gate_w_gate = nn.Linear(GD, GD * 3 // 2, bias=True)
            self.gate_w_up   = nn.Linear(GD, GD * 3 // 2, bias=True)
            self.gate_w_down = nn.Linear(GD * 3 // 2, GD, bias=False)
            self.gate_drop   = nn.Dropout(dr)
            nn.init.zeros_(self.gate_w_down.weight)
            self.gate_norm = mk1d(GD)
            self.gate_out  = nn.Linear(GD, 1858, bias=True)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.constant_(self.gate_out.bias, 4.0)

            self.from_proj = nn.Linear(D, PDH, bias=True)
            self.from_ln   = mk1d(PDH)
            if attn_impl == "sdpa":
                self.from_mha = SdpaMHA(PDH, 4)
            else:
                self.from_mha = nn.MultiheadAttention(PDH, 4, dropout=0.0,
                                                      batch_first=True, bias=True)
            self.from_out  = nn.Linear(PDH, PDH, bias=False)

            self.to_proj   = nn.Linear(D, PDH, bias=True)
            self.to_ln     = mk1d(PDH)
            if attn_impl == "sdpa":
                self.to_mha = SdpaMHA(PDH, 4)
            else:
                self.to_mha = nn.MultiheadAttention(PDH, 4, dropout=0.0,
                                                    batch_first=True, bias=True)
            self.to_out    = nn.Linear(PDH, PDH, bias=False)

            self.scale      = 1.0 / math.sqrt(PDH)
            self.promo_from = nn.Linear(PDH, 3, bias=False)
            self.promo_to   = nn.Linear(PDH, 3, bias=False)

            self.register_buffer("sl_idx", sl_idx)

        def forward(self, x_in):
            B = x_in.shape[0]
            if xc0h_input:
                x = xc0h_stem_forward(self, x_in, D, XC0H_K)              # [B, D, 8, 8]
            else:
                x = self.emb(x_in).reshape(B, 8, 8, D).permute(0, 3, 1, 2).contiguous()
            for blk in self.pre:
                x = blk(x)

            board = x.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, D)          # [B, 64, D], unnormed
            pos   = self.pos(torch.arange(SEQ_LEN, device=x_in.device))
            pos   = pos.unsqueeze(0).expand(B, -1, -1)
            x = board + self.pos_scale * pos                              # [B, 64, D]
            x = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)  # [B, 72, D]

            for blk in self.blocks:
                x = blk(x)
            x = self.trunk_ln(x)                                           # [B, 72, D]

            if wdl_split:
                w = self.wdl_ow(F.gelu(self.wdl_hw(x[:, 64, :])))          # acc0
                d = self.wdl_od(F.gelu(self.wdl_hd(x[:, 65, :])))          # acc1
                l = self.wdl_ol(F.gelu(self.wdl_hl(x[:, 66, :])))          # acc2
                wdl = torch.cat([w, d, l], dim=-1)                         # [B, 3]
            else:
                wdl = self.wdl_w2(F.gelu(self.wdl_w1(x[:, 64, :])))         # [B, 3]

            if wdl_split:
                gi       = torch.cat([x[:, 67, :], x[:, 68, :]], dim=-1)      # [B, 2D]
                from_set = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:71, :]], dim=1)
                to_set   = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 71:72, :]], dim=1)
            else:
                gi       = torch.cat([x[:, 65, :], x[:, 70, :]], dim=-1)      # [B, 2D]
                from_set = torch.cat([x[:, :64, :], x[:, 66:68, :], x[:, 70:72, :]], dim=1)
                to_set   = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:72, :]], dim=1)
            h        = F.silu(self.gate_w_gate(gi)) * self.gate_w_up(gi)
            gd       = self.gate_w_down(h)
            g        = gi + self.gate_drop(gd)
            g        = g.clamp(-250.0, 250.0)
            gate_raw = self.gate_out(self.gate_norm(g))                     # [B, 1858]

            # from_proj/to_proj and from_out/to_out are residual-wrapped so a
            # gradient path survives regardless of what those weights do --
            # relies on D == PDH (true for every config using this builder
            # today); would need a projection on the skip if that ever
            # stops holding.
            f_proj = from_set + F.gelu(self.from_proj(from_set))
            fn     = self.from_ln(f_proj)
            if attn_impl == "sdpa":
                fh = self.from_mha(fn[:, :64, :], fn)
            else:
                fh, _ = self.from_mha(fn[:, :64, :], fn, fn, need_weights=False)
            f_base = f_proj[:, :64, :] + fh
            fv     = f_base + self.from_out(f_base)                        # [B, 64, PDH]

            t_proj = to_set + F.gelu(self.to_proj(to_set))
            tn     = self.to_ln(t_proj)
            if attn_impl == "sdpa":
                th = self.to_mha(tn[:, :64, :], tn)
            else:
                th, _ = self.to_mha(tn[:, :64, :], tn, tn, need_weights=False)
            t_base = t_proj[:, :64, :] + th
            tv     = t_base + self.to_out(t_base)                          # [B, 64, PDH]

            dots_full = torch.bmm(fv, tv.transpose(1, 2)).mul(self.scale)  # [B, 64, 64]
            dots      = dots_full.reshape(B, 64 * 64)

            dots_sub = dots_full[:, 48:56, 56:64]                          # [B, 8, 8]
            pf       = self.promo_from(fv[:, 48:56, :])                    # [B, 8, 3]
            pt       = self.promo_to(tv[:, 56:64, :])                      # [B, 8, 3]
            promo    = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]
                        ).permute(0, 3, 2, 1).reshape(B, 192)

            raw_4288 = torch.cat([dots, promo], dim=1)
            q_sl     = raw_4288[:, self.sl_idx]
            combined = q_sl + F.logsigmoid(gate_raw)
            return combined, wdl

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_conv_shallow_mha(cfg: dict, log_params: bool = False):
    """2x Conv(128) -> concat(x, x, pos) -> 384-d, 8 accumulators, 8x tx ->
    WDL / SmartGate / from-to heads at 384-d (gate hidden 768)."""
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


XC0H_VOCAB = 15   # 0=empty, 1-6 us P/N/B/R/Q/K, 7-12 them P/N/B/R/Q/K, 13=EP, 14=PAD
XC0H_DEMB  = 18


def attach_xc0h_stem(m, CF, K, norm_type="layernorm"):
    """Attach the xc0h history-token stem (input (B, K*64+K+3)) onto `m`; pair
    with xc0h_stem_forward. norm_type ("layernorm", "rms", "l1", "l1p5")
    changes xc0h_proj_ln's state dict keys."""
    import torch
    import torch.nn as nn

    norms = {"layernorm": nn.LayerNorm, "rms": make_rms1d, "l1": make_l1norm,
             "l1p5": make_l1p5norm}
    if norm_type not in norms:
        raise ValueError(f"norm_type must be one of {sorted(norms)}, got {norm_type!r}")

    in_ch = K * XC0H_DEMB + K + 3      # frames + rep + castle + stm + hmc
    m.xc0h_tok_emb = nn.Embedding(XC0H_VOCAB, XC0H_DEMB)
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
        m.xc0h_tok_emb.weight.copy_(E)
    m.xc0h_cast_emb = nn.Embedding(16, 64)
    # project to CF-4; 4 spatial planes (ones, checkerboard, rank, file)
    # appended after the norm so LN can't destroy their constant/gradient
    # values. Each gets its own learnable scale so convs can attenuate any
    # plane toward zero if the geometry isn't useful.
    m.xc0h_proj    = nn.Linear(in_ch, CF - 4)
    m.xc0h_proj_ln = norms[norm_type](CF - 4)

    # free learnable scales (init 1.0): 3 for rep/stm/hmc, 4 for spatial planes
    m.xc0h_rep_scale     = nn.Parameter(torch.tensor(1.0))
    m.xc0h_stm_scale     = nn.Parameter(torch.tensor(1.0))
    m.xc0h_hmc_scale     = nn.Parameter(torch.tensor(1.0))
    m.xc0h_ones_scale    = nn.Parameter(torch.tensor(1.0))
    m.xc0h_checker_scale = nn.Parameter(torch.tensor(1.0))
    m.xc0h_rank_scale    = nn.Parameter(torch.tensor(1.0))
    m.xc0h_file_scale    = nn.Parameter(torch.tensor(1.0))

    sqs = torch.arange(64).float()
    ranks_sq = sqs // 8
    files_sq = sqs % 8
    m.register_buffer("xc0h_spatial", torch.stack([
        torch.ones(64),
        ((ranks_sq + files_sq) % 2) * 2 - 1,
        ranks_sq / 7.0 * 2 - 1,
        files_sq / 7.0 * 2 - 1,
    ], dim=-1).unsqueeze(0))                                        # [1, 64, 4]


def xc0h_stem_forward(m, x_in, CF, K):
    """Run the stem attached by attach_xc0h_stem. Returns (B, CF, 8, 8)."""
    import torch

    B = x_in.shape[0]
    # model's running dtype (fp16 under half() inference, fp32 in training) --
    # rep/stm/hmc are cast here and run natively in that dtype throughout,
    # same as everywhere else in this model past the embed.
    dtype = m.xc0h_tok_emb.weight.dtype
    tok  = x_in[:, :K * 64].reshape(B, K, 64).long()
    rep  = x_in[:, K * 64:K * 64 + K].to(dtype)
    cast = x_in[:, K * 64 + K].long()
    stm  = x_in[:, K * 64 + K + 1].to(dtype)
    hmc  = x_in[:, K * 64 + K + 2].to(dtype)

    tok_emb = m.xc0h_tok_emb(tok)                                       # [B,K,64,d]
    tok_emb = tok_emb.permute(0, 2, 1, 3).reshape(B, 64, K * XC0H_DEMB)  # [B,64,K*d]

    # rep/stm/hmc are raw 0/1 (or unbounded for hmc) -- rescale to roughly
    # the same [-1,1] order of magnitude as the (~unit-variance) embeddings
    # so none of them get washed out or dominate at the projection below.
    # Each also gets its own free learnable scale (init 1.0).
    rep_planes = (rep * 2.0 - 1.0).unsqueeze(1).expand(B, 64, K) * m.xc0h_rep_scale
    cast_plane = m.xc0h_cast_emb(cast).unsqueeze(-1)                    # [B,64,1]
    stm_plane  = (stm * 2.0 - 1.0).view(B, 1, 1).expand(B, 64, 1) * m.xc0h_stm_scale
    hmc_plane  = (hmc / 99.0 * 2.0 - 1.0).view(B, 1, 1).expand(B, 64, 1) * m.xc0h_hmc_scale

    fused = torch.cat(
        [tok_emb, rep_planes, cast_plane, stm_plane, hmc_plane], dim=-1)  # [B,64,in_ch]
    projected = m.xc0h_proj_ln(m.xc0h_proj(fused))                      # [B, 64, CF-4]
    scales  = torch.stack([m.xc0h_ones_scale, m.xc0h_checker_scale,
                           m.xc0h_rank_scale, m.xc0h_file_scale])
    spatial = m.xc0h_spatial.to(projected.dtype).expand(B, -1, -1) * scales  # [B, 64, 4]
    x = torch.cat([projected, spatial], dim=-1)                          # [B, 64, CF]
    return x.reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()       # [B, CF, 8, 8]


def attach_value_branch(m, D, block_fn, vd=256, heads=8, ff_mult=2):
    """Value branch tapped before the trunk's last blocks: one block_fn block,
    then an accumulator token reads the squares via self- then cross-attn (with
    a residual so the first read survives). vp_proj is skipped when D == vd."""
    import torch
    import torch.nn as nn

    # no norm on the block output: vp_norm below is an RMS over the same
    # channel axis, and the permute between them is a pure reshape
    m.value_block = block_fn()

    m.vp_norm  = make_rms1d(D)
    m.vp_proj  = nn.Linear(D, vd, bias=True) if D != vd else None
    m.vp_query = nn.Parameter(torch.randn(1, 1, vd))

    m.vp_mha = nn.MultiheadAttention(vd, heads, dropout=0.0, batch_first=True,
                                     bias=True)
    m.vp_ff_up   = nn.Linear(vd, vd * ff_mult, bias=True)
    m.vp_ff_down = nn.Linear(vd * ff_mult, vd, bias=True)
    m.vp_norm2   = make_rms1d(vd)

    m.wdl_mha = nn.MultiheadAttention(vd, heads, dropout=0.0, batch_first=True,
                                      bias=True)
    m.wdl_w1  = nn.Linear(vd, 64, bias=True)
    m.wdl_out = nn.Linear(64, 3, bias=True)


def value_branch_forward(m, x, B):
    """Run the branch attached by attach_value_branch. x is the tapped trunk
    activation, [B, D, 8, 8]. Returns WDL logits [B, 3]."""
    import torch
    import torch.nn.functional as F

    h = m.value_block(x)                                     # [B, D, 8, 8]
    board = h.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, -1)    # [B, 64, D]

    bp = m.vp_norm(board)
    if m.vp_proj is not None:
        bp = m.vp_proj(bp)                                   # [B, 64, vd]
    q  = m.vp_query.expand(B, -1, -1).to(bp.dtype)
    h  = torch.cat([bp, q], dim=1)                           # [B, 65, vd]

    r, _ = m.vp_mha(h, h, h, need_weights=False)
    r = m.vp_ff_down(F.silu(m.vp_ff_up(r)))
    h = m.vp_norm2(h + r)                                    # [B, 65, vd]

    w      = h[:, SEQ_LEN:]                                  # [B, 1, vd]
    squares = h[:, :SEQ_LEN]                                 # [B, 64, vd]
    v, _ = m.wdl_mha(w, squares, squares, need_weights=False)
    v = w + F.silu(v)                                        # [B, 1, vd]

    v = F.silu(m.wdl_w1(v.squeeze(1)))
    return m.wdl_out(v)


def attach_conv_pure_heads(m, D, PDH, GATE_D, GATE_H, dr, sl_idx):
    """Attach the conv-* family's SmartGate + from/to policy heads onto `m`.
    WDL lives on its own tap (attach_value_branch)."""
    import math
    import torch.nn as nn

    # trunk is post-RMSNorm, which rescales without re-centering, so gate_conv
    # carries its own bias -- it feeds a leaky_relu whose kink sits at zero and
    # there is nothing upstream to position it against.
    m.gate_conv = nn.Conv2d(D, 16, 3, padding=1, bias=True)       # [B,16,8,8]
    m.gate_proj = nn.Linear(16 * SEQ_LEN, GATE_D)                 # 1024 -> 512
    m.gate_proj_ln = make_rms1d(GATE_D)

    # both read gate_proj_ln, which is RMS and does not re-center, so they take
    # a bias. gate_w_down follows the SwiGLU product rather than a norm, so it
    # stays unbiased and zero-init.
    m.gate_w_gate = nn.Linear(GATE_D, GATE_H, bias=True)
    m.gate_w_up = nn.Linear(GATE_D, GATE_H, bias=True)
    m.gate_w_down = nn.Linear(GATE_H, GATE_D, bias=False)
    m.gate_drop = nn.Dropout(dr)
    nn.init.zeros_(m.gate_w_down.weight)
    m.gate_norm = make_rms1d(GATE_D)
    m.gate_out = nn.Linear(GATE_D, 1858, bias=True)
    nn.init.zeros_(m.gate_out.weight)
    nn.init.constant_(m.gate_out.bias, 4.0)

    m.from_proj = nn.Linear(D, PDH)
    m.from_ln = make_rms1d(PDH)
    m.from_mha = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
    m.from_out = nn.Linear(PDH, PDH)

    m.to_proj = nn.Linear(D, PDH)
    m.to_ln = make_rms1d(PDH)
    m.to_mha = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
    m.to_out = nn.Linear(PDH, PDH)

    m.scale = 1.0 / math.sqrt(PDH)
    m.promo_from = nn.Linear(PDH, 3, bias=False)
    m.promo_to = nn.Linear(PDH, 3, bias=False)

    m.register_buffer("sl_idx", sl_idx)


def conv_pure_heads_forward(m, trunk, B, D):
    """attach_conv_pure_heads heads on post-norm [B, D, 8, 8]; returns
    policy_1858 only (WDL comes from value_branch_forward)."""
    import torch
    import torch.nn.functional as F

    gc = F.leaky_relu(m.gate_conv(trunk), 0.01).reshape(B, 16 * SEQ_LEN)
    gi = m.gate_proj_ln(F.gelu(m.gate_proj(gc)))                      # [B, 512]
    h = F.silu(m.gate_w_gate(gi)) * m.gate_w_up(gi)
    g = gi + m.gate_drop(m.gate_w_down(h))
    gate_raw = m.gate_out(m.gate_norm(g))                             # [B, 1858]

    board = trunk.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, D)          # [B, 64, D]

    f_proj = F.gelu(m.from_proj(board))
    fn = m.from_ln(f_proj)
    fh, _ = m.from_mha(fn, fn, fn, need_weights=False)
    fv = m.from_out(f_proj + fh)

    t_proj = F.gelu(m.to_proj(board))
    tn = m.to_ln(t_proj)
    th, _ = m.to_mha(tn, tn, tn, need_weights=False)
    tv = m.to_out(t_proj + th)

    dots_full = torch.bmm(fv, tv.transpose(1, 2)).mul(m.scale)        # [B, 64, 64]
    dots = dots_full.reshape(B, SEQ_LEN * SEQ_LEN)

    dots_sub = dots_full[:, 48:56, 56:64]
    pf = m.promo_from(fv[:, 48:56, :])
    pt = m.promo_to(tv[:, 56:64, :])
    promo = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]
             ).permute(0, 3, 2, 1).reshape(B, 192).float()

    raw_4288 = torch.cat([dots.float(), promo], dim=1)
    q_sl = raw_4288[:, m.sl_idx]
    combined = q_sl + F.logsigmoid(gate_raw.float())
    return combined.to(trunk.dtype)


def build_conv_rms_backbone(cfg: dict, block_fn, n_blocks, log_params=False,
                            name=""):
    """RMSNorm conv family skeleton: stem -> n_stem RMS conv blocks -> optional
    pos-concat widen (cfg pos_dim) -> n_blocks x block_fn() -> conv-pure heads.
    cfg xc0h_K picks the history stem."""
    import torch
    import torch.nn as nn
    import pyfastchess

    CF = cfg["conv_filters"]
    POS = cfg.get("pos_dim", 0)
    D = CF + POS
    dr = cfg["dropout"]
    XC0H_K = cfg.get("xc0h_K")
    xc0h_input = XC0H_K is not None
    # policy head width is pinned, not tied to the trunk: from_proj/to_proj are
    # Linear(D, PDH) and rescale either way. Matches conv-pure and the precond
    # models, so widening the trunk is a body experiment and not a head one.
    PDH = 256
    GATE_D = 512
    GATE_H = GATE_D * 3 // 2
    n_stem = cfg.get("stem_blocks", 2)
    # value branches off this many blocks before the end, so the tail blocks
    # get policy gradient only and can specialize for it
    n_tap = min(cfg.get("value_tap", 2), n_blocks)
    n_body = n_blocks - n_tap
    value_d = cfg.get("value_dim", 256)

    sl_idx = torch.from_numpy(
        pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            if xc0h_input:
                attach_xc0h_stem(self, CF, XC0H_K, norm_type="rms")
            else:
                self.emb = nn.Embedding(VOCAB_SIZE, CF)
            self.stem = nn.ModuleList(
                [make_rms_conv_block(CF) for _ in range(n_stem)])
            if POS:
                self.conv_ln = make_rms2d(CF)
                self.pos = nn.Embedding(SEQ_LEN, POS)
            self.blocks = nn.ModuleList([block_fn() for _ in range(n_body)])
            self.tail   = nn.ModuleList([block_fn() for _ in range(n_tap)])
            attach_value_branch(self, D, block_fn, value_d)
            self.trunk_ln = make_rms2d(D)
            attach_conv_pure_heads(self, D, PDH, GATE_D, GATE_H, dr, sl_idx)

        def forward(self, tokens):
            B = tokens.shape[0]
            if xc0h_input:
                x = xc0h_stem_forward(self, tokens, CF, XC0H_K)
            else:
                x = self.emb(tokens).reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()
            for blk in self.stem:
                x = blk(x)
            if POS:
                p = self.pos(torch.arange(SEQ_LEN, device=tokens.device))
                p = p.reshape(1, 8, 8, POS).permute(0, 3, 1, 2).expand(B, -1, -1, -1)
                x = torch.cat([self.conv_ln(x), p.to(x.dtype)], dim=1)   # [B, D, 8, 8]
            for blk in self.blocks:
                x = blk(x)
            wdl = value_branch_forward(self, x, B)                       # [B, 3]
            for blk in self.tail:
                x = blk(x)
            trunk = self.trunk_ln(x)
            return conv_pure_heads_forward(self, trunk, B, D), wdl

    m = M()
    if log_params:
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name} PT params: {n:,}")
    return m


def build_pt_conv_globalizer(cfg: dict, log_params: bool = False):
    """conv-globalizer: RMS conv stem, optional pos widen, then globalizer blocks
    (SwiGLU global branch, no attention)."""
    D  = cfg["conv_filters"] + cfg.get("pos_dim", 0)
    nb = cfg.get("num_blocks", 8)
    ns = cfg.get("stem_blocks", 2)
    return build_conv_rms_backbone(
        cfg,
        block_fn=lambda: make_globalizer_block(D),
        n_blocks=nb,
        log_params=log_params,
        name=f"conv-globalizer-{ns}c{nb}g",
    )


def build_pt_conv_pure(cfg: dict, log_params: bool = False):
    """cfg num_blocks x RMS conv blocks (NCHW) -> trunk RMSNorm -> conv-pure
    WDL / SmartGate / from-to heads."""
    import torch
    import torch.nn as nn
    import pyfastchess

    D      = cfg["conv_filters"]
    nb     = cfg["num_blocks"]
    dr     = cfg["dropout"]
    PDH    = 256
    GATE_D = 512
    GATE_H = GATE_D * 3 // 2
    XC0H_K = cfg.get("xc0h_K")
    xc0h_input = XC0H_K is not None
    n_tap  = min(cfg.get("value_tap", 2), nb)
    n_body = nb - n_tap
    value_d = cfg.get("value_dim", 256)

    sl_idx = torch.from_numpy(
        pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            if xc0h_input:
                attach_xc0h_stem(self, D, XC0H_K, norm_type="rms")
            else:
                self.emb  = nn.Embedding(VOCAB_SIZE, D)
            self.blocks   = nn.ModuleList(
                [make_rms_conv_block(D) for _ in range(n_body)])
            self.tail     = nn.ModuleList(
                [make_rms_conv_block(D) for _ in range(n_tap)])
            attach_value_branch(self, D, lambda: make_rms_conv_block(D), value_d)
            self.trunk_ln = make_rms2d(D)
            attach_conv_pure_heads(self, D, PDH, GATE_D, GATE_H, dr, sl_idx)

        def forward(self, tokens):
            B = tokens.shape[0]
            if xc0h_input:
                x = xc0h_stem_forward(self, tokens, D, XC0H_K)
            else:
                x = self.emb(tokens).reshape(B, 8, 8, D).permute(0, 3, 1, 2).contiguous()
            for blk in self.blocks:
                x = blk(x)
            wdl = value_branch_forward(self, x, B)                       # [B, 3]
            for blk in self.tail:
                x = blk(x)
            trunk = self.trunk_ln(x)
            return conv_pure_heads_forward(self, trunk, B, D), wdl

    m = M()
    if log_params:
        print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_full_mha_smartgate(cfg: dict, log_params: bool = False):
    """No conv: embed + pos -> 384-d, 8 accumulators, 7x tx, widen to 512, 1x tx
    -> 16m-precond-smartgate heads at 512-d."""
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


def build_pt_expandy_swiglu(cfg: dict, log_params: bool = False):
    """m13-7: xc0h stem -> plain/expandy/gated-expandy convs -> FF then SwiGLU tx
    (8 accumulators) -> wdl3 + dot policy, minus scale * sigmoid(gate_raw)
    (shut-logit, trained by the legality BCE). cfg norm: layernorm | l1p5."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyfastchess

    import random

    D         = cfg["conv_filters"]
    nh        = cfg.get("num_heads", 8)
    ff_dim    = cfg.get("ff_dim", 1024)
    n_plain   = cfg.get("n_plain", 0)
    n_expandy = cfg.get("n_expandy", 0)
    n_gated   = cfg.get("n_gated", 0)
    n_ff      = cfg.get("n_ff", 0)
    n_swiglu  = cfg.get("n_swiglu", 0)
    groups    = cfg.get("expandy_groups", 4)
    dr        = cfg.get("dropout", 0.05)
    norm      = cfg.get("norm", "layernorm")
    norms     = {"layernorm": nn.LayerNorm, "l1p5": make_l1p5norm}
    if norm not in norms:
        raise ValueError(f"norm must be one of {sorted(norms)}, got {norm!r}")
    make_norm = norms[norm]
    XC0H_K    = cfg["xc0h_K"]
    PDH, POLICY_HEADS = 256, 4
    GATE_D, SMARTGATE_H = 2 * D, 1024
    GATE_MAX = 12.0          # gate_scale held in [0, GATE_MAX]
    GATE_SCALE_INIT = 1.0    # start weak; the gate earns scale once it knows legality
    GATE_BIAS_MAX = 5.0      # per-logit shut bias held in [0, GATE_BIAS_MAX]
    GATE_BIAS_INIT = 4.0     # sigmoid(4) = 0.98: starts mostly shut
    GATE_EMA_BETA = 1.0 - 1.0 / 500.0
    LEAK = 0.03
    attn_gate = cfg.get("attn_gate", False)   # SwiGLU-block SDPA out * (1 + tanh(g(xq)))
    prelu     = cfg.get("prelu", False)       # conv leaky_relus -> per-channel PReLU
    PRELU_MIN, PRELU_MAX = -0.5, 0.5

    sl_idx = torch.from_numpy(
        pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0]
    n_main = int((sl_idx < 4096).sum())
    # promo grid is type*64 + to_file*8 + from_file; gather it in pyfastchess's
    # packed tail order: valid file pairs (from outer, to inner) x type (q, r, b)
    promo_idx = torch.tensor([
        t * 64 + tf * 8 + ff
        for ff in range(8) for tf in range(max(0, ff - 1), min(7, ff + 1) + 1)
        for t in range(3)])

    def norm_nchw(norm, x):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    class GateOut(nn.Module):
        """1858 shut-logits (per-logit bias) + learned scale; project() clamps
        both. A checkpoint without `shut_logit` is open-logit: negated on load."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(1858, GATE_D) * 10 ** -1.5)
            self.bias = nn.Parameter(torch.full((1858,), GATE_BIAS_INIT))
            self.scale = nn.Parameter(torch.tensor(GATE_SCALE_INIT))
            self.register_buffer("shut_logit", torch.ones(()))
            self.converted = False

        def forward(self, g):
            return F.linear(g, self.weight, self.bias)

        def project(self):
            with torch.no_grad():
                self.bias.clamp_(0.0, GATE_BIAS_MAX)
                self.scale.clamp_(0.0, GATE_MAX)

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            if prefix + "weight" in state_dict and prefix + "shut_logit" not in state_dict:
                state_dict[prefix + "weight"] = -state_dict[prefix + "weight"]
                b = -state_dict[prefix + "bias"]
                state_dict[prefix + "bias"] = b.expand(1858).clone()
                state_dict[prefix + "scale"] = torch.tensor(GATE_SCALE_INIT)
                state_dict[prefix + "shut_logit"] = torch.ones(())
                self.converted = True
            super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def make_act(ch):
        # PReLU at init == leaky_relu(LEAK), so a graft onto a leaky checkpoint is exact
        return nn.PReLU(ch, init=LEAK) if prelu else nn.LeakyReLU(LEAK)

    class Attention(nn.Module):
        """gate=True scales the SDPA output by 1 + tanh(g(xq)); g is zero-init, so
        the gate is exactly 1 at init (identity graft onto an ungated checkpoint).
        Gate mean/std/range per (token, channel) tracked as 500-step ag_* EMAs
        while `ema_on` (set by M.set_hotspot_scan)."""

        def __init__(self, dim, heads, gate=False):
            super().__init__()
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim, bias=False)   # softmax is shift-invariant per query
            self.v = nn.Linear(dim, dim)
            self.o = nn.Linear(dim, dim)
            self.g = nn.Linear(dim, dim) if gate else None
            if gate:
                nn.init.zeros_(self.g.weight)
                nn.init.zeros_(self.g.bias)
                n_tok = SEQ_LEN + 8
                self.register_buffer("ag_mean_ema", torch.ones(n_tok, dim))
                self.register_buffer("ag_std_ema", torch.zeros(n_tok, dim))
                self.register_buffer("ag_range_ema", torch.zeros(n_tok, dim))
            self.heads = heads
            self.head_dim = dim // heads
            self.ema_on = False

        def forward(self, xq, xkv):
            bq, nq, dim = xq.shape
            nk = xkv.shape[1]
            q = self.q(xq).view(bq, nq, self.heads, self.head_dim).transpose(1, 2)
            k = self.k(xkv).view(bq, nk, self.heads, self.head_dim).transpose(1, 2)
            v = self.v(xkv).view(bq, nk, self.heads, self.head_dim).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bq, nq, dim)
            if self.g is not None:
                gate = 1 + torch.tanh(self.g(xq))
                if self.ema_on and self.training and torch.is_grad_enabled():
                    with torch.no_grad():
                        gh = gate.detach().half()
                        w = 1 - GATE_EMA_BETA
                        self.ag_mean_ema.lerp_(gh.mean(0).float(), w)
                        self.ag_std_ema.lerp_(gh.std(0).float(), w)
                        self.ag_range_ema.lerp_((gh.amax(0) - gh.amin(0)).float(), w)
                a = a * gate
            return self.o(a)

    class ConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = make_norm(D)
            self.c1 = nn.Conv2d(D, D, 3, padding=1)
            self.c2 = nn.Conv2d(D, D, 3, padding=1, bias=False)
            self.a1 = make_act(D)
            self.a2 = make_act(D)

        def forward(self, x):
            h = self.a1(self.c1(norm_nchw(self.norm, x)))
            return x + self.a2(self.c2(h))

    class ExpandyConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = make_norm(D)
            self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
            self.p1 = nn.Conv2d(2 * D, D, 1, bias=False)
            self.c2 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
            self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)
            self.a1 = make_act(2 * D)
            self.a2 = make_act(2 * D)

        def forward(self, x):
            h = self.p1(self.a1(self.c1(norm_nchw(self.n1, x))))
            return x + self.p2(self.a2(self.c2(h)))

    class GatedExpandyConvBlock(nn.Module):
        """Gate mean/std/range per (channel, square) tracked as 500-step EMAs
        while `ema_on` (set by M.set_hotspot_scan) for the gate-health readout."""

        def __init__(self):
            super().__init__()
            self.n1 = make_norm(D)
            self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
            self.p1 = nn.Conv2d(2 * D, D, 1, bias=False)
            self.c_gate = nn.Conv2d(D, D, 1)
            self.c2 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
            self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)
            self.a1 = make_act(2 * D)
            self.a2 = make_act(2 * D)
            self.register_buffer("gate_mean_ema", torch.full((D, 8, 8), 0.5))
            self.register_buffer("gate_std_ema", torch.full((D, 8, 8), 0.16))
            self.register_buffer("gate_range_ema", torch.zeros(D, 8, 8))
            self.ema_on = False

        def forward(self, x):
            h = self.p1(self.a1(self.c1(norm_nchw(self.n1, x))))
            gate = torch.sigmoid(self.c_gate(h))
            if self.ema_on and self.training and torch.is_grad_enabled():
                with torch.no_grad():
                    gh = gate.detach().half()
                    w = 1 - GATE_EMA_BETA
                    self.gate_mean_ema.lerp_(gh.mean(0).float(), w)
                    self.gate_std_ema.lerp_(gh.std(0).float(), w)
                    self.gate_range_ema.lerp_((gh.amax(0) - gh.amin(0)).float(), w)
            return x + gate * self.p2(self.a2(self.c2(h)))

    class MaxMonitor(nn.Module):
        """Training-only hotspot stats (max |x| and its (token, channel), per-step
        max, mean) in device buffers while `active`; read_reset() is the one host
        sync. (T, C) are the last two dims; 2-D inputs report (C,)."""

        def __init__(self):
            super().__init__()
            self.active = False
            self.has_t = True
            self.steps = 0
            self.n_act = 0
            for name in ("cur", "peak", "maxsum", "actsum"):
                self.register_buffer(name, torch.zeros(()), persistent=False)
            for name in ("cur_tc", "peak_tc"):
                self.register_buffer(
                    name, torch.zeros(2, dtype=torch.long), persistent=False)

        def forward(self, x):
            if self.active and self.training and torch.is_grad_enabled():
                with torch.no_grad():
                    xh = x.detach().half()
                    a = xh.abs().reshape(-1)
                    i = a.argmax()
                    v = a[i].float()
                    self.has_t = x.dim() >= 3
                    C = x.shape[-1]
                    T = x.shape[-2] if self.has_t else 1
                    tc = torch.stack([i // C % T, i % C])
                    self.cur_tc.copy_(torch.where(v > self.cur, tc, self.cur_tc))
                    self.cur.copy_(torch.maximum(self.cur, v))
                    self.actsum.add_(xh.mean(dtype=torch.float32))
                    self.n_act += 1

        def end_step(self):
            if self.active:
                self.peak_tc.copy_(
                    torch.where(self.cur > self.peak, self.cur_tc, self.peak_tc))
                self.peak.copy_(torch.maximum(self.peak, self.cur))
                self.maxsum.add_(self.cur)
                self.cur.zero_()
                self.steps += 1

        def read_reset(self):
            peak, maxsum, actsum, t, c = torch.cat([
                torch.stack([self.peak, self.maxsum, self.actsum]),
                self.peak_tc.float()]).tolist()
            out = (peak, maxsum / max(self.steps, 1), actsum / max(self.n_act, 1),
                   (int(t), int(c)) if self.has_t else (int(c),))
            for buf in (self.cur, self.peak, self.maxsum, self.actsum,
                        self.cur_tc, self.peak_tc):
                buf.zero_()
            self.steps = self.n_act = 0
            return out

    class TxBlockFF(nn.Module):
        def __init__(self, down_mon):
            super().__init__()
            self.n1 = make_norm(D)
            self.attn = Attention(D, nh)
            self.n2 = make_norm(D)
            self.ff1 = nn.Linear(D, ff_dim)
            self.ff2 = nn.Linear(ff_dim, D, bias=False)
            self.down_mon = down_mon
            self.attn_drop = nn.Dropout(dr)
            self.ff_drop = nn.Dropout(dr)

        def forward(self, x):
            n = self.n1(x)
            x = x + self.attn_drop(self.attn(n, n))
            down_out = self.ff2(F.gelu(self.ff1(self.n2(x))))
            self.down_mon(down_out)
            return x + self.ff_drop(down_out)

    class TxBlockSwiGLU(nn.Module):
        def __init__(self, down_mon):
            super().__init__()
            self.n1 = make_norm(D)
            self.attn = Attention(D, nh, gate=attn_gate)
            self.n2 = make_norm(D)
            self.gate = nn.Linear(D, 4 * D)
            self.up = nn.Linear(D, 4 * D)
            self.down = nn.Linear(4 * D, D, bias=False)
            self.down_mon = down_mon
            self.attn_drop = nn.Dropout(dr)
            self.ff_drop = nn.Dropout(dr)

        def forward(self, x):
            n = self.n1(x)
            x = x + self.attn_drop(self.attn(n, n))
            n = self.n2(x)
            down_out = self.down(F.silu(self.gate(n)) * self.up(n))
            self.down_mon(down_out)
            return x + self.ff_drop(down_out)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            attach_xc0h_stem(self, D, XC0H_K, norm_type=norm)
            self.convs = nn.ModuleList(
                [ConvBlock() for _ in range(n_plain)]
                + [ExpandyConvBlock() for _ in range(n_expandy)]
                + [GatedExpandyConvBlock() for _ in range(n_gated)])

            self.pos = nn.Parameter(torch.randn(SEQ_LEN, D) * 0.02)
            self.pos_scale = nn.Parameter(torch.tensor(1.0))
            self.global_tokens = nn.Parameter(torch.randn(1, 8, D) * 0.02)
            self.ff_down_mon = MaxMonitor()
            self.swiglu_down_mon = MaxMonitor()
            self.res_mon = MaxMonitor()
            self.blocks = nn.ModuleList(
                [TxBlockFF(self.ff_down_mon) for _ in range(n_ff)]
                + [TxBlockSwiGLU(self.swiglu_down_mon) for _ in range(n_swiglu)])
            self.trunk = make_norm(D)

            # wdl3: acc0/acc1/acc2 each feed an independent W/D/L head
            vh = 64   # per-head readout width; gives each W/D/L head real capacity
            self.wdl_hw = nn.Linear(D, vh, bias=True)
            self.wdl_hd = nn.Linear(D, vh, bias=True)
            self.wdl_hl = nn.Linear(D, vh, bias=True)
            self.wdl_ow = nn.Linear(vh, 1, bias=False)
            self.wdl_od = nn.Linear(vh, 1, bias=False)
            self.wdl_ol = nn.Linear(vh, 1, bias=False)

            self.from_proj = nn.Linear(D, PDH)
            self.from_norm = make_norm(PDH)
            self.from_mha = Attention(PDH, POLICY_HEADS)
            self.from_out = nn.Linear(PDH, PDH, bias=False)
            self.to_proj = nn.Linear(D, PDH)
            self.to_norm = make_norm(PDH)
            self.to_mha = Attention(PDH, POLICY_HEADS)
            self.to_out = nn.Linear(PDH, PDH, bias=False)
            self.from_dot_norm = make_norm(PDH)
            self.to_dot_norm = make_norm(PDH)
            self.promo_from = nn.Linear(PDH, 3, bias=False)
            self.promo_to = nn.Linear(PDH, 3, bias=False)

            self.gate_ff1 = nn.Linear(GATE_D, SMARTGATE_H)
            self.gate_ff2 = nn.Linear(SMARTGATE_H, GATE_D, bias=False)
            self.gate_norm = make_norm(GATE_D)
            self.gate_out = GateOut()
            with torch.no_grad():
                self.gate_ff2.weight.normal_(std=10 ** -1.5)
            self.gate_raw = None
            self.dot_raw_mon = MaxMonitor()
            self.register_buffer("sl_idx", sl_idx)
            self.register_buffer("promo_idx", promo_idx, persistent=False)

        def forward(self, x_in):
            B = x_in.shape[0]
            x = xc0h_stem_forward(self, x_in, D, XC0H_K)
            for blk in self.convs:
                x = blk(x)
            board = x.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, D)
            x = board + self.pos_scale * self.pos.unsqueeze(0)
            x = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)
            for blk in self.blocks:
                x = blk(x)
            self.res_mon(x)
            x = self.trunk(x)
            w = self.wdl_ow(F.gelu(self.wdl_hw(x[:, 64, :])))   # acc0
            d = self.wdl_od(F.gelu(self.wdl_hd(x[:, 65, :])))   # acc1
            l = self.wdl_ol(F.gelu(self.wdl_hl(x[:, 66, :])))   # acc2
            wdl = torch.cat([w, d, l], dim=-1)

            from_set = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:71, :]], dim=1)
            to_set   = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 71:72, :]], dim=1)
            f_proj = from_set + F.gelu(self.from_proj(from_set))
            fn     = self.from_norm(f_proj)
            f_base = f_proj[:, :64, :] + self.from_mha(fn[:, :64, :], fn)
            fv     = f_base + self.from_out(f_base)
            t_proj = to_set + F.gelu(self.to_proj(to_set))
            tn     = self.to_norm(t_proj)
            t_base = t_proj[:, :64, :] + self.to_mha(tn[:, :64, :], tn)
            tv     = t_base + self.to_out(t_base)
            dots_raw = torch.bmm(self.from_dot_norm(fv), self.to_dot_norm(tv).transpose(1, 2))
            self.dot_raw_mon(dots_raw)
            dots_full = dots_raw * (PDH ** -0.5)
            dots      = dots_full.reshape(B, 64 * 64)
            dots_sub  = dots_full[:, 48:56, 56:64]
            pf = self.promo_from(fv[:, 48:56, :])
            pt = self.promo_to(tv[:, 56:64, :])
            promo = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]
                     ).permute(0, 3, 2, 1).reshape(B, 192)
            policy = torch.cat(
                [dots[:, self.sl_idx[:n_main]], promo[:, self.promo_idx]], dim=-1)

            gi   = torch.cat([x[:, 67, :], x[:, 68, :]], dim=-1)
            down = self.gate_ff2(F.gelu(self.gate_ff1(gi)))
            g    = self.gate_norm(gi + down)
            self.gate_raw = self.gate_out(g)
            gate = self.gate_out.scale * torch.sigmoid(self.gate_raw)
            return policy - gate, wdl

        def hotspot_monitors(self):
            return {
                "policy_dot": self.dot_raw_mon,
                "ff_down": self.ff_down_mon,
                "swiglu_down": self.swiglu_down_mon,
                "residual": self.res_mon,
            }

        def set_dropout(self, duty):
            """Each TxBlock dropout site runs with chance `duty`, flipped
            independently; off = eval mode, no kernel launched."""
            for blk in self.blocks:
                blk.attn_drop.train(random.random() < duty)
                blk.ff_drop.train(random.random() < duty)

        def project_params(self):
            """Clamp constrained params back into range; call after each opt step."""
            self.gate_out.project()
            if not hasattr(self, "prelu_list"):
                self.prelu_list = [m for m in self.modules() if isinstance(m, nn.PReLU)]
            with torch.no_grad():
                for m in self.prelu_list:
                    m.weight.clamp_(PRELU_MIN, PRELU_MAX)

        def gate_param_stats(self):
            """(gate bias min, mean, max, gate scale). Host sync, print cadence only."""
            b = self.gate_out.bias.detach().float()
            return tuple(torch.stack(
                [b.min(), b.mean(), b.max(), self.gate_out.scale.detach().float()]).tolist())

        def set_hotspot_scan(self, on):
            for mon in self.hotspot_monitors().values():
                mon.active = on
            for m in self.modules():
                if hasattr(m, "ema_on"):
                    m.ema_on = on

        def hotspot_end_step(self):
            """Close one training step on every monitor. No host sync."""
            for mon in self.hotspot_monitors().values():
                mon.end_step()

        def hotspot_stats(self):
            """Reads and resets every monitor (host sync, print cadence only).
            Returns name -> (abs max, avg per-step max, signed act mean,
            (T, C) or (C,) of abs max)."""
            return {k: mon.read_reset() for k, mon in self.hotspot_monitors().items()}

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
    "conv-globalizer-2c6g":                 build_pt_conv_globalizer,
    "precond-signed-4g6t-d512":             build_pt_precond_signed,
    "precond-mha-4g6t-d512":                build_pt_precond_signed,
    "precond-mixer-6m4t-d512":              build_pt_precond_mixer,
    "precond-mha-8c4t-d384":                build_pt_precond_signed,
    "precond-mha-8c6t-d256":                build_pt_precond_addpos,
    "precond-mha-10c6t-d256":               build_pt_precond_addpos,
    "precond-mha-10c6t-d256-wdl3":          build_pt_precond_addpos,
    "precond-mha-8c8t-d256":                build_pt_precond_addpos,
    "conv-pure":                            build_pt_conv_pure,
    "conv-shallow-mha":                      build_pt_conv_shallow_mha,
    "full-mha-smartgate":                   build_pt_full_mha_smartgate,
    "m13-7-8c8t-d256":                      build_pt_expandy_swiglu,
}
