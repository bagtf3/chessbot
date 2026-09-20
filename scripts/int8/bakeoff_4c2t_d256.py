"""Architecture bake-off: 4c2t d256 trunk, float trunk-architecture study.

Plain float training (fp16 AMP + GradScaler), no int8: no weight/activation
quantization, no QAT, no ONNX/TRT export.  The trunk block is the only variable;
stem, pos, policy heads and value head match the prod module layout so prod
weights warm-start directly.

Layers: L1Norm, plain ConvBlock, ExpandyConvBlock (grouped 3x3 expand D->2D,
leaky, 1x1 contract, twice), GatedExpandyConvBlock (expandy with a 1x1 sigmoid
gate off stage 1 on stage 2), TxBlockFF (gelu FF) and TxBlockSwiGLU.

    m1    4 plain conv                    2 plain-FF tx      baseline
    m7    4 plain conv                    2 SwiGLU tx        gated FF
    m11   2 plain + 2 expandy conv        2 plain-FF tx      separable convs
    m13   2 plain + 2 gated-expandy conv  2 plain-FF tx      gated separable convs
    m11-7, m13-7-*, m11-13-*, ...         combinations of the above, deeper/wider

Value head is the d256 single-head W/D/L readout off acc0.

Range bangers leash the fp16-hot real quantities: the raw policy-gate logits, the
logsigmoid suppression term, the policy dot product, and the SwiGLU gate*up
product.  All per-report-step metrics for every model go to one CSV; each model
is saved.  Every MODEL_SPECS entry runs in order unless --only narrows it.

    python scripts/int8/bakeoff_4c2t_d256.py [--only m11,m13]
"""
import argparse
import csv
import datetime
import gc
import glob
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import poc_4c2t_wdl3_d256_maxnet as poc
from poc_4c2t_wdl3_d256_maxnet import (
    D, HEADS, FF, PDH, POLICY_HEADS, GATE_D, BOARD, TOKENS,
    XC0H_VOCAB, XC0H_DEMB, XC0H_IN, XC0H_LEN, POLICY_DIM, EPS, FP16_CLIP,
    SparseShardBatches, ramp, load_param, Tee,
)

import pyfastchess

SMARTGATE_H = 1024  # smartgate SwiGLU hidden dim
SOFTCLAMP_M = 100.0  # tanh soft-clamp ceiling for smartgate activations
SOFTCLAMP_C = 80.0  # knee: identity within +/-C, tanh soft-bound to +/-M beyond
GATE_EMA_BETA = 1.0 - 1.0 / 500.0  # 500-step EMA for gate mean/std/range tracking

GATE_LOGIT_CEIL = 127.0  # range banger on the raw policy-gate logits
GACT_CEIL = 16.0  # range banger on the logsigmoid suppression term
SWIGLU_PROD_CEIL = 96.0  # range banger on the SwiGLU gate*up product

def soft_clamp(x, m=SOFTCLAMP_M, c=SOFTCLAMP_C):
    """Identity within +/-c, tanh soft-bound to +/-m beyond; slope 1 (C1) at the
    knee, output strictly in (-m, m)."""
    a = x.abs()
    mag = a.clamp(max=c) + (m - c) * torch.tanh(F.relu(a - c) / (m - c))
    return torch.sign(x) * mag


class L1Norm(nn.Module):
    """Mean-abs normalization island: x / mean(|x|), learned per-channel scale and
    bias.  The +/-250 clamp mirrors prod RMSNorm, capping the residual stream so it
    cannot grow past the fp16 range between blocks."""

    def __init__(self, d, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((d,), float(scale)))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        x = x.clamp(-250.0, 250.0)
        y = x / (x.abs().mean(-1, keepdim=True) + EPS)
        return y * self.scale.to(x.dtype) + self.b.to(x.dtype)


class Attention(nn.Module):
    """SDPA-equivalent MHA, mirroring prod q/k/v/o.  Cross-attention: query and
    key/value streams are separate arguments (self-attention passes the same tensor
    twice).  Scores/softmax run in an fp32 island so autocast cannot overflow."""

    def __init__(self, dim=D, heads=HEADS):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

    def forward(self, xq, xkv):
        bq, nq, _ = xq.shape
        nk = xkv.shape[1]
        q = self.q(xq).view(bq, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(xkv).view(bq, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(xkv).view(bq, nk, self.heads, self.head_dim).transpose(1, 2)
        a = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        a = a.softmax(dim=-1)
        h = (a @ v).transpose(1, 2).reshape(bq, nq, self.dim)
        return self.o(h)


class ConvBlock(nn.Module):
    """Prenorm, c1(bias)->leaky, c2(no bias)->leaky, residual."""

    def __init__(self):
        super().__init__()
        self.norm = L1Norm(D)
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        self.c2 = nn.Conv2d(D, D, 3, padding=1, bias=False)

    def forward(self, x):
        h = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.leaky_relu(self.c1(h), 0.01)
        h = self.c2(h)
        return x + F.leaky_relu(h, 0.01)


class ExpandyConvBlock(nn.Module):
    """Separable/inverted-bottleneck conv block: one L1Norm, then two grouped-3x3
    expand (D->2D) -> leaky at 2D -> 1x1 contract (2D->D) stages, residual.  ~0.72x
    a full ConvBlock's params at groups=4; TRT fuses the leaky into the grouped conv."""

    def __init__(self, groups=4):
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(2 * D, D, 1, bias=False)
        self.c2 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))
        h = self.p2(F.leaky_relu(self.c2(h), 0.05))
        return x + h


class GatedExpandyConvBlock(nn.Module):
    """ExpandyConvBlock with a gate: a 1x1 sigmoid off the stage-1 output gates the
    stage-2 output.  x + gate*stage2; gate mean/std/range EMA."""

    def __init__(self, groups=4):
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(2 * D, D, 1, bias=False)
        self.c_gate = nn.Conv2d(D, D, 1)
        self.c2 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)
        self.register_buffer("gate_mean_ema", torch.full((D, BOARD, BOARD), 0.5))
        self.register_buffer("gate_std_ema", torch.full((D, BOARD, BOARD), 0.16))
        self.register_buffer("gate_range_ema", torch.zeros(D, BOARD, BOARD))

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))
        gate = torch.sigmoid(self.c_gate(h))
        h = self.p2(F.leaky_relu(self.c2(h), 0.05))
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                deb = GATE_EMA_BETA
                gf = gate.detach().float()
                self.gate_mean_ema.mul_(deb).add_(gf.mean(0), alpha=1 - deb)
                self.gate_std_ema.mul_(deb).add_(gf.std(0), alpha=1 - deb)
                gr = gf.amax(0) - gf.amin(0)
                self.gate_range_ema.mul_(deb).add_(gr, alpha=1 - deb)
        return x + gate * h


class TxBlockFF(nn.Module):
    """Plain FF transformer block: norm -> attn -> residual -> norm -> gelu FF."""

    def __init__(self):
        super().__init__()
        self.n1 = L1Norm(D)
        self.attn = Attention(D, HEADS)
        self.n2 = L1Norm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n)
        h = F.gelu(self.ff1(self.n2(x)))
        return x + self.ff2(h)


class TxBlockSwiGLU(nn.Module):
    """4x-SwiGLU replacement: 256 -> 1024 gate/up, silu(gate)*up, 1024 -> 256.  The
    gate*up product is the hottest activation in the block; range-banged to +/-96."""

    def __init__(self):
        super().__init__()
        self.n1 = L1Norm(D)
        self.attn = Attention(D, HEADS)
        self.n2 = L1Norm(D)
        self.gate = nn.Linear(D, 4 * D)
        self.up = nn.Linear(D, 4 * D)
        self.down = nn.Linear(4 * D, D, bias=False)
        self.last_product = None

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n)
        n = self.n2(x)
        product = F.silu(self.gate(n)) * self.up(n)
        self.last_product = product.detach()
        return x + self.down(product)


class BakeoffModel(nn.Module):
    """Configurable 4c2t d256 trunk.  conv_mode picks the conv-trunk variant;
    tx_mode picks the transformer variant.  Value head is the d256 single-head
    W/D/L readout off acc0.  Stem, pos, policy heads and gate match the prod layout
    so prod weights load directly."""

    def __init__(self, conv_mode="vanilla", tx_mode="vanilla"):
        super().__init__()
        self.conv_mode = conv_mode
        self.tx_mode = tx_mode

        self.tok_emb = nn.Embedding(XC0H_VOCAB, XC0H_DEMB)
        self.cast_emb = nn.Embedding(16, 64)
        self.stem_proj = nn.Linear(XC0H_IN, D - 4)
        self.stem_norm = L1Norm(D - 4)
        for name in ("rep", "stm", "hmc", "ones", "checker", "rank", "file"):
            setattr(self, f"{name}_scale", nn.Parameter(torch.tensor(1.0)))
        sqs = torch.arange(64).float()
        ranks_sq, files_sq = sqs // 8, sqs % 8
        self.register_buffer("xc0h_spatial", torch.stack([
            torch.ones(64), ((ranks_sq + files_sq) % 2) * 2 - 1,
            ranks_sq / 7.0 * 2 - 1, files_sq / 7.0 * 2 - 1], dim=-1).unsqueeze(0))

        # conv_mode -> (n plain, n expandy, n gated expandy), applied in that order
        conv_counts = {
            "vanilla": (4, 0, 0),
            "expandy": (2, 2, 0),
            "expandy_1_gated_expandy_2": (0, 1, 2),
            "gated_expandy": (2, 0, 2),
            "gated_expandy_1plain_3": (1, 0, 3),
        }
        if conv_mode not in conv_counts:
            raise ValueError(f"conv_mode={conv_mode}")
        n_plain, n_exp, n_gexp = conv_counts[conv_mode]
        self.convs = nn.ModuleList([ConvBlock() for _ in range(n_plain)])
        self.expandy_convs = nn.ModuleList([ExpandyConvBlock() for _ in range(n_exp)])
        self.gated_expandy_convs = nn.ModuleList(
            [GatedExpandyConvBlock() for _ in range(n_gexp)])

        self.pos = nn.Parameter(torch.randn(TOKENS, D) * 0.02)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.global_tokens = nn.Parameter(torch.randn(1, 8, D) * 0.02)

        # tx_mode -> (n plain FF, n SwiGLU), FF blocks first
        tx_counts = {
            "vanilla": (2, 0),
            "swiglu": (0, 2),
            "ff2_swiglu2": (2, 2),
            "ff3_swiglu2": (3, 2),
        }
        if tx_mode not in tx_counts:
            raise ValueError(f"tx_mode={tx_mode}")
        n_ff, n_sw = tx_counts[tx_mode]
        self.blocks = nn.ModuleList(
            [TxBlockFF() for _ in range(n_ff)] + [TxBlockSwiGLU() for _ in range(n_sw)])
        self.trunk = L1Norm(D)

        # d256 value head: single shared readout off acc0, three-logit W/D/L.
        self.wdl_w1 = nn.Linear(D, D)
        self.wdl_norm = nn.RMSNorm(D)
        self.wdl_w2 = nn.Linear(D, 3, bias=True)

        self.from_proj = nn.Linear(D, PDH)
        self.from_norm = L1Norm(PDH)
        self.from_mha = Attention(PDH, POLICY_HEADS)
        self.from_out = nn.Linear(PDH, PDH, bias=False)
        self.to_proj = nn.Linear(D, PDH)
        self.to_norm = L1Norm(PDH)
        self.to_mha = Attention(PDH, POLICY_HEADS)
        self.to_out = nn.Linear(PDH, PDH, bias=False)
        self.from_dot_norm = L1Norm(PDH)
        self.to_dot_norm = L1Norm(PDH)
        self.promo_from = nn.Linear(PDH, 3, bias=False)
        self.promo_to = nn.Linear(PDH, 3, bias=False)

        self.gate_w_gate = nn.Linear(GATE_D, SMARTGATE_H)
        self.gate_w_up = nn.Linear(GATE_D, SMARTGATE_H)
        self.gate_w_down = nn.Linear(SMARTGATE_H, GATE_D, bias=False)
        self.gate_norm = L1Norm(GATE_D)
        self.gate_out = nn.Linear(GATE_D, POLICY_DIM, bias=True)
        with torch.no_grad():
            self.gate_w_down.weight.normal_(std=10 ** -1.5)
            self.gate_out.weight.normal_(std=10 ** -1.5)
            self.gate_out.bias.fill_(4.0)
        self.last_gate_raw = None
        self.last_gate_act = None
        self.last_dot_raw = None
        self.last_wdl_hidden = None
        self.last_wdl_logits = None
        self.register_buffer("sl_idx", torch.from_numpy(
            pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0])

    def stem(self, x):
        b = x.shape[0]
        K = poc.XC0H_K
        dtype = self.tok_emb.weight.dtype
        tok = x[:, :K * 64].long().view(b, K, TOKENS)
        rep = x[:, K * 64:K * 64 + K].to(dtype)
        cast = x[:, K * 64 + K].long()
        stm = x[:, K * 64 + K + 1].to(dtype)
        hmc = x[:, K * 64 + K + 2].to(dtype)
        te = self.tok_emb(tok).permute(0, 2, 1, 3).reshape(b, TOKENS, K * XC0H_DEMB)
        rep_planes = (rep * 2.0 - 1.0).unsqueeze(1).expand(b, TOKENS, K) * self.rep_scale
        cast_plane = self.cast_emb(cast).unsqueeze(-1)
        stm_plane = (stm * 2.0 - 1.0).view(b, 1, 1).expand(b, TOKENS, 1) * self.stm_scale
        hmc_plane = (hmc / 99.0 * 2.0 - 1.0).view(b, 1, 1)
        hmc_plane = hmc_plane.expand(b, TOKENS, 1) * self.hmc_scale
        fused = torch.cat([te, rep_planes, cast_plane, stm_plane, hmc_plane], dim=-1)
        proj = self.stem_norm(self.stem_proj(fused))
        scales = torch.stack([self.ones_scale, self.checker_scale,
                              self.rank_scale, self.file_scale])
        spatial = self.xc0h_spatial.to(proj.dtype).expand(b, -1, -1) * scales
        x = torch.cat([proj, spatial], dim=-1)
        return x.reshape(b, BOARD, BOARD, D).permute(0, 3, 1, 2)

    def forward(self, x):
        if x.shape[1] != XC0H_LEN:
            raise ValueError(f"expected xc0h [B,{XC0H_LEN}], got {tuple(x.shape)}")
        b = x.shape[0]
        x = self.stem(x)
        for block in self.convs:
            x = block(x)
        for block in self.expandy_convs:
            x = block(x)
        for block in self.gated_expandy_convs:
            x = block(x)
        board = x.permute(0, 2, 3, 1).reshape(b, TOKENS, D)
        x = board + self.pos_scale * self.pos.unsqueeze(0)
        x = torch.cat([x, self.global_tokens.expand(b, -1, -1)], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.trunk(x)
        wdl_hidden = self.wdl_norm(F.gelu(self.wdl_w1(x[:, TOKENS, :])))
        self.last_wdl_hidden = wdl_hidden
        wdl = self.wdl_w2(wdl_hidden)
        self.last_wdl_logits = wdl

        from_set = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:71, :]], dim=1)
        to_set = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 71:72, :]], dim=1)
        f_proj = from_set + F.gelu(self.from_proj(from_set))
        fn = self.from_norm(f_proj)
        f_base = f_proj[:, :64, :] + self.from_mha(fn[:, :64, :], fn)
        f_vec = f_base + self.from_out(f_base)
        t_proj = to_set + F.gelu(self.to_proj(to_set))
        tn = self.to_norm(t_proj)
        t_base = t_proj[:, :64, :] + self.to_mha(tn[:, :64, :], tn)
        t_vec = t_base + self.to_out(t_base)
        fq = self.from_dot_norm(f_vec)
        tq = self.to_dot_norm(t_vec)
        dots_raw = torch.bmm(fq, tq.transpose(1, 2)).clamp(-FP16_CLIP, FP16_CLIP)
        self.last_dot_raw = dots_raw
        dots_full = dots_raw * (PDH ** -0.5)
        dots = dots_full.reshape(b, TOKENS * TOKENS)
        dots_sub = dots_full[:, 48:56, 56:64]
        pf = self.promo_from(f_vec[:, 48:56, :])
        pt = self.promo_to(t_vec[:, 56:64, :])
        promo = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]
                 ).permute(0, 3, 2, 1).reshape(b, 192)
        policy = torch.cat([dots, promo], dim=-1)[:, self.sl_idx]

        gi = torch.cat([x[:, 67, :], x[:, 68, :]], dim=-1)
        gate = soft_clamp(F.silu(self.gate_w_gate(gi)))
        up = soft_clamp(self.gate_w_up(gi))
        h = soft_clamp(gate * up)
        down = soft_clamp(self.gate_w_down(h))
        g = (gi + down).clamp(-250.0, 250.0)
        g = self.gate_norm(g)
        gate_raw = self.gate_out(g)
        self.last_gate_raw = gate_raw

        gate_act = F.logsigmoid(gate_raw.clamp(-250.0, 250.0))
        self.last_gate_act = gate_act
        return policy + gate_act, wdl


def rfmt(v):
    if v >= 1e3:
        return f"{v:.1f}"
    return f"{v:.2f}" if v >= 10 else f"{v:.3f}"


def range_penalty(model):
    """Range banger on the fp16-hot real activations: the raw policy-gate logits
    and the SwiGLU gate*up product.  Sum-relu over the overage (not mean, so
    outliers aren't diluted); pair with a small coefficient.  Returns (penalty,
    n_sites_over)."""
    total = 0.0
    over = 0
    raw = model.last_gate_raw
    if raw is not None and raw.detach().abs().max() > GATE_LOGIT_CEIL:
        total = total + F.relu(raw.abs() - GATE_LOGIT_CEIL).sum()
        over += 1
    for block in model.blocks:
        p = getattr(block, "last_product", None)
        if p is not None and p.detach().abs().max() > SWIGLU_PROD_CEIL:
            total = total + F.relu(p.abs() - SWIGLU_PROD_CEIL).sum()
            over += 1
    return total, over


def range_offenders(model, topn=3):
    """Top-N range-banger offenders this step, (name, offense sum, max single coord
    magnitude, banger target): the raw gate logits, the logsigmoid suppression
    term, and any SwiGLU gate*up product."""
    offs = []

    def add(name, absval, target):
        over = F.relu(absval - target)
        if over.numel() and float(over.max()) > 0:
            offs.append((name, float(over.sum()), float(absval.max()), round(target)))

    raw = model.last_gate_raw
    if raw is not None:
        add("gate_out.raw", raw.detach().abs(), GATE_LOGIT_CEIL)
    act = model.last_gate_act
    if act is not None:
        add("gate_logsigmoid", act.detach().abs(), GACT_CEIL)
    for i, block in enumerate(model.blocks):
        p = getattr(block, "last_product", None)
        if p is not None:
            add(f"blocks.{i}.swiglu_prod", p.detach().abs(), SWIGLU_PROD_CEIL)
    offs.sort(key=lambda t: t[1], reverse=True)
    return offs[:topn]


def ema_summary(ema):
    """Six-number summary (min, max, mean, p25, p50, p75) of an EMA tensor."""
    e = ema.detach().float().flatten()
    qs = torch.tensor([0.25, 0.5, 0.75], device=e.device)
    p25, p50, p75 = torch.quantile(e, qs).tolist()
    return (float(e.min()), float(e.max()), float(e.mean()), p25, p50, p75)


def conv_gate_ema(blocks):
    """Per-block mean/std/range EMA summaries of a conv gate."""
    return [
        (
            i,
            ema_summary(b.gate_mean_ema),
            ema_summary(b.gate_std_ema),
            ema_summary(b.gate_range_ema),
        )
        for i, b in enumerate(blocks)
    ]


def print_gate_ema(rows, tag):
    """Print mean/std/range six-number gate-EMA readouts, columns aligned.

    Label is padded to the width of the widest ("range"); every stat is a fixed
    5-wide 2-decimal field so mean/std/range rows stay column-locked."""
    for label, which in (("mean", 1), ("std", 2), ("range", 3)):
        head = f"  {tag} gate-EMA {label:<5}"
        pad = " " * len(head)
        for j, r in enumerate(rows):
            s = r[which]
            prefix = head if j == 0 else pad
            print(f"{prefix}  b{r[0]}  min {s[0]:5.2f}  max {s[1]:5.2f}"
                  f"  mean {s[2]:5.2f}  p25 {s[3]:5.2f}  p50 {s[4]:5.2f}"
                  f"  p75 {s[5]:5.2f}", flush=True)


def load_weight(param, W, blend):
    """Copy a prod weight into a plain parameter, blending in `blend` matched-scale
    noise.  Works for Linear and Conv2d weights and biases alike."""
    with torch.no_grad():
        Wf = W.float()
        if blend > 0.0 and Wf.ndim >= 2:
            Wf = (1.0 - blend) * Wf + blend * torch.randn_like(Wf) * Wf.std()
        param.copy_(Wf.to(param.dtype).reshape(param.shape))


def load_linear(mod, sd, key, blend):
    """Load a Linear or Conv2d from `{key}.weight` (+ `{key}.bias` if present)."""
    load_weight(mod.weight, sd[f"{key}.weight"], blend)
    if mod.bias is not None and f"{key}.bias" in sd:
        with torch.no_grad():
            mod.bias.copy_(sd[f"{key}.bias"].float().to(mod.bias.dtype))


def try_load_linear(mod, sd, key, blend):
    """load_linear, but leave the layer fresh when the checkpoint shape doesn't
    match the current layer (e.g. a widened smartgate)."""
    w = sd[f"{key}.weight"]
    if tuple(w.shape) != tuple(mod.weight.shape):
        print(f"[init] skip {key}: ckpt {tuple(w.shape)} !="
              f" model {tuple(mod.weight.shape)}, left fresh", flush=True)
        return
    load_linear(mod, sd, key, blend)


def warm_start(model, ckpt_path, blend, conv_src=1):
    """Steal every structurally-shared tensor from a prod wdl3 checkpoint: stem,
    pos, plain convs, transformer attn (+ FF only for plain-FF blocks), policy
    heads, gate.  Fresh: L1Norm scales/biases, expandy convs, SwiGLU FF, value head."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    load_param(model.tok_emb.weight, sd["xc0h_tok_emb.weight"])
    load_param(model.cast_emb.weight, sd["xc0h_cast_emb.weight"])
    load_linear(model.stem_proj, sd, "xc0h_proj", blend)
    for name in ("rep", "stm", "hmc", "ones", "checker", "rank", "file"):
        load_param(getattr(model, f"{name}_scale"), sd[f"xc0h_{name}_scale"])
    load_param(model.pos, sd["pos.weight"])
    load_param(model.pos_scale, sd["pos_scale"])
    load_param(model.global_tokens, sd["global_tokens"])

    for i, block in enumerate(model.convs):
        c = f"pre.{conv_src + i}"
        load_linear(block.c1, sd, f"{c}.c1", blend)
        load_linear(block.c2, sd, f"{c}.c2", blend)

    for i, block in enumerate(model.blocks):
        t = f"blocks.{i}"
        for w in ("q", "k", "v", "o"):
            load_linear(getattr(block.attn, w), sd, f"{t}.attn.{w}", blend)
        if isinstance(block, TxBlockFF):
            load_linear(block.ff1, sd, f"{t}.ff1", blend)
            load_linear(block.ff2, sd, f"{t}.ff2", blend)

    for side in ("from", "to"):
        load_linear(getattr(model, f"{side}_proj"), sd, f"{side}_proj", blend)
        for w in ("q", "k", "v", "o"):
            load_linear(getattr(getattr(model, f"{side}_mha"), w),
                        sd, f"{side}_mha.{w}", blend)
        load_linear(getattr(model, f"{side}_out"), sd, f"{side}_out", blend)
    load_linear(model.promo_from, sd, "promo_from", blend)
    load_linear(model.promo_to, sd, "promo_to", blend)

    try_load_linear(model.gate_w_gate, sd, "gate_w_gate", blend)
    try_load_linear(model.gate_w_up, sd, "gate_w_up", blend)
    try_load_linear(model.gate_w_down, sd, "gate_w_down", blend)
    load_linear(model.gate_out, sd, "gate_out", blend)
    print(f"[init] warm-started {model.conv_mode}/{model.tx_mode} from "
          f"{Path(ckpt_path).name} (value head + expandy/swiglu fresh)", flush=True)


def resume_from(model, ckpt_path):
    """Reload a bake-off checkpoint to continue training. Loads non-strict so
    that gate/DC *_ema tracking buffers absent from older checkpoints are added
    fresh (zeros) rather than blocking the load. Any missing non-ema tensor or
    unexpected key is a real mismatch and raises. Returns the checkpoint's step
    count so cumulative progress can be logged."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    res = model.load_state_dict(ck["model"], strict=False)
    bad_missing = [k for k in res.missing_keys if not k.endswith("_ema")]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(
            f"resume mismatch loading {Path(ckpt_path).name}: "
            f"missing={bad_missing} unexpected={list(res.unexpected_keys)}")
    added = [k for k in res.missing_keys if k.endswith("_ema")]
    prev_steps = int(ck.get("steps", 0))
    print(f"[resume] {model.conv_mode}/{model.tx_mode} from "
          f"{Path(ckpt_path).name} (prev steps {prev_steps})", flush=True)
    if added:
        print(f"[resume] added {len(added)} fresh ema buffers: "
              f"{', '.join(added)}", flush=True)
    return prev_steps
# Architecture catalogue.  Every entry runs in order unless --only narrows it.
MODEL_SPECS = {
    "m1": dict(name="m1_conv-van_tx-van", conv="vanilla", tx="vanilla"),
    "m7": dict(name="m7_conv-van_tx-swiglu", conv="vanilla", tx="swiglu"),
    "m11": dict(name="m11_conv-expandy_tx-van", conv="expandy", tx="vanilla"),
    "m13": dict(name="m13_conv-gated-expandy_tx-van", conv="gated_expandy", tx="vanilla"),
    "m11-7": dict(name="m11-7_conv-expandy_tx-swiglu", conv="expandy", tx="swiglu"),
    "m11-13-1-7-3c5t": dict(
        name="m11-13-1-7-3c5t",
        conv="expandy_1_gated_expandy_2", tx="ff3_swiglu2"
    ),
    "m13-7-4c4t": dict(
        name="m13-7-4c4t",
        conv="gated_expandy_1plain_3", tx="ff2_swiglu2"
    ),
}

SCALER_MAX_SCALE = 2.0 ** 16
FP16_MIN = 5.960464477539063e-8
FP16_MAX = 65504.0


def gfmt(v):
    return f"{v:.6f}" if v == 0 or abs(v) >= 1e-5 else f"{v:.2e}"


CSV_FIELDS = [
    "model", "conv_mode", "tx_mode", "step",
    "pol_ce", "wdl_ce", "mse", "corr", "grad_scale",
    "range_pen", "range_over", "gact_pen", "dot_peak", "dot_cap", "dot_pen",
    "grad_global", "grad_mean", "grad_median",
    "max_grad_l2", "max_grad_l2_name", "min_grad_l2", "min_grad_l2_name",
    "max_grad", "max_grad_name", "min_grad", "min_grad_name", "n_fp16_min",
    "wdl_hidden_abs_mean", "wdl_hidden_rms", "wdl_hidden_abs_max",
    "wdl_logits_abs_mean", "wdl_logits_rms", "wdl_logits_abs_max",
    "wdl_w2_grad_l2", "wdl_w2_grad_abs_mean", "wdl_w2_grad_scaled_max",
    "clip",
    "cg0_rema_mean", "cg0_rema_p50", "cg1_rema_mean", "cg1_rema_p50",
    "cg2_rema_mean", "cg2_rema_p50",
]


def train_one(spec, args, train_data, val_files, writer, fh, out_dir):
    val_data = SparseShardBatches(args.shard_dir, seed=args.split_seed + 1,
                                  files=val_files,
                                  batch_size=args.val_batch, pool_shards=2,
                                  ready_batches=4)
    model = BakeoffModel(conv_mode=spec["conv"], tx_mode=spec["tx"]).to(args.device)
    start_step = 0
    prev_steps = 0
    if args.resume:
        src = out_dir / f"{spec['name']}{args.resume_tag}.pt"
        prev_steps = resume_from(model, src)
    elif args.init_from:
        warm_start(model, args.init_from, blend=args.init_rand_blend,
                   conv_src=args.init_conv_src)
    else:
        print(f"[init] {spec['name']}: random init (no warm-start)", flush=True)
    model.to(args.device)

    betas = (1.0 - 1.0 / args.beta1_span, 1.0 - 1.0 / args.beta2_span)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=betas)
    amp = not args.no_amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=2**12)
    clip_warmup = args.grad_clip_warmup
    total_steps = args.steps
    has_gate = bool(model.gated_expandy_convs)
    print(f"\n=== {spec['name']}  conv={spec['conv']} tx={spec['tx']}"
          f"  steps={total_steps} ===", flush=True)
    print(f"[opt] WD={args.weight_decay} on {len(decay)} tensors;"
          f" {len(no_decay)} WD-free  lr {args.lr}->{args.lr_end} decay"
          f" {args.lr_decay_start}-{args.lr_decay_end}", flush=True)

    for step in range(start_step, total_steps):
        dot_cap = args.dot_cap_start + (args.dot_cap_end - args.dot_cap_start) * \
            ramp(step, args.dot_cap_hold, args.dot_cap_steps - args.dot_cap_hold)
        clip = args.grad_clip_start + (args.grad_clip_end - args.grad_clip_start) * \
            ramp(step, 0, clip_warmup)
        lr = args.lr + (args.lr_end - args.lr) * ramp(
            step, args.lr_decay_start, args.lr_decay_end - args.lr_decay_start)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, policy_t, wdl_t = train_data.batch(args.batch, args.device)
        report = step % 250 == 0 or step == total_steps - 1
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            policy, wdl = model(x)
            policy_ce = F.cross_entropy(policy, policy_t)
            wdl_ce = F.cross_entropy(wdl, wdl_t)
            loss = policy_ce + args.value_weight * wdl_ce
            range_pen, n_over = range_penalty(model)
            gact_pen = F.relu(model.last_gate_act.abs() - GACT_CEIL).sum()
            loss = loss + args.bounds_weight * (range_pen + gact_pen)
            dot_pen = F.relu(model.last_dot_raw.abs() - dot_cap).sum()
            loss = loss + args.dot_weight * dot_pen
            dot_peak = float(model.last_dot_raw.detach().abs().max())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step} ({spec['name']})")
        opt.zero_grad(set_to_none=True)
        backward_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if report:
            names, l2s = [], []
            min_grad, min_grad_name = float("inf"), ""
            max_grad, max_grad_name = 0.0, ""
            n_fp16_min = 0
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach().float()
                if p.ndim >= 2:
                    names.append(name)
                    l2s.append(g.norm())
                ga = g.abs() * backward_scale
                n_fp16_min += int((ga == FP16_MIN).sum())
                gmax = float(ga.max())
                if gmax > max_grad:
                    max_grad, max_grad_name = gmax, name
                nz = ga[ga > 0]
                if nz.numel():
                    gmin = float(nz.min())
                    if gmin < min_grad:
                        min_grad, min_grad_name = gmin, name
            pgn = torch.stack(l2s)
            min_idx, max_idx = int(pgn.argmin()), int(pgn.argmax())
            min_grad_l2, min_grad_l2_name = float(pgn[min_idx]), names[min_idx]
            max_grad_l2, max_grad_l2_name = float(pgn[max_idx]), names[max_idx]
            min_grad_risk = min_grad < FP16_MIN
            max_grad_risk = max_grad > FP16_MAX
            wh = model.last_wdl_hidden.detach().float()
            wdl_hidden_abs_mean = float(wh.abs().mean())
            wdl_hidden_rms = float(wh.pow(2).mean().sqrt())
            wdl_hidden_abs_max = float(wh.abs().max())
            wl = model.last_wdl_logits.detach().float()
            wdl_logits_abs_mean = float(wl.abs().mean())
            wdl_logits_rms = float(wl.pow(2).mean().sqrt())
            wdl_logits_abs_max = float(wl.abs().max())
            wg =model.wdl_w2.weight.grad.detach().float()
            wdl_w2_grad_l2 = float(wg.norm())
            wdl_w2_grad_abs_mean = float(wg.abs().mean())
            wdl_w2_grad_scaled_max = float(wg.abs().max()) * backward_scale
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() > SCALER_MAX_SCALE:
            scaler.update(new_scale=SCALER_MAX_SCALE)

        if not report:
            continue
        offenders = range_offenders(model)  # train-batch state, before val forward
        val_enc, val_pol, val_wdl = val_data.batch(args.val_batch, args.device)
        with torch.no_grad():
            pol, wdlv = model(val_enc)
            probs = wdlv.float().softmax(-1)
            target_q = val_wdl[:, 0] - val_wdl[:, 2]
            pred_q = probs[:, 0] - probs[:, 2]
            mse = F.mse_loss(pred_q, target_q)
            corr = torch.corrcoef(torch.stack([pred_q, target_q]))[0, 1]
            pol_ce = F.cross_entropy(pol, val_pol)
            wdl_ce_v = F.cross_entropy(wdlv, val_wdl)
        row = {k: "" for k in CSV_FIELDS}
        row.update(
            model=spec["name"], conv_mode=spec["conv"], tx_mode=spec["tx"],
            step=step,
            pol_ce=f"{pol_ce:.4f}", wdl_ce=f"{wdl_ce_v:.4f}",
            mse=f"{mse:.4f}", corr=f"{corr:.4f}",
            grad_scale=f"{scaler.get_scale():.1f}",
            range_pen=f"{float(range_pen):.2f}", range_over=n_over,
            gact_pen=f"{float(gact_pen):.2f}",
            dot_peak=f"{dot_peak:.1f}", dot_cap=f"{dot_cap:.1f}",
            dot_pen=f"{float(dot_pen):.2f}",
            grad_global=f"{float(grad_norm):.4f}", grad_mean=f"{pgn.mean():.4f}",
            grad_median=f"{pgn.median():.4f}",
            max_grad_l2=gfmt(max_grad_l2), max_grad_l2_name=max_grad_l2_name,
            min_grad_l2=gfmt(min_grad_l2), min_grad_l2_name=min_grad_l2_name,
            max_grad=gfmt(max_grad), max_grad_name=max_grad_name,
            min_grad=gfmt(min_grad), min_grad_name=min_grad_name,
            n_fp16_min=n_fp16_min,
            wdl_hidden_abs_mean=f"{wdl_hidden_abs_mean:.4f}",
            wdl_hidden_rms=f"{wdl_hidden_rms:.4f}",
            wdl_hidden_abs_max=f"{wdl_hidden_abs_max:.4f}",
            wdl_logits_abs_mean=f"{wdl_logits_abs_mean:.4f}",
            wdl_logits_rms=f"{wdl_logits_rms:.4f}",
            wdl_logits_abs_max=f"{wdl_logits_abs_max:.4f}",
            wdl_w2_grad_l2=gfmt(wdl_w2_grad_l2),
            wdl_w2_grad_abs_mean=gfmt(wdl_w2_grad_abs_mean),
            wdl_w2_grad_scaled_max=gfmt(wdl_w2_grad_scaled_max),
            clip=f"{clip:.2f}")
        for i, b in enumerate(model.gated_expandy_convs):
            r = ema_summary(b.gate_range_ema)
            row.update({f"cg{i}_rema_mean": f"{r[2]:.3f}",
                        f"cg{i}_rema_p50": f"{r[4]:.3f}"})
        writer.writerow(row)
        tag = f"[{spec['name']} {step:5d}]"
        print(f"{tag} pol_ce {pol_ce:6.3f}  wdl_ce {wdl_ce_v:5.3f}"
              f"  mse {mse:5.3f}  corr {corr:+6.3f}", flush=True)
        print(f"{'':2}range {float(range_pen):7.2f}/{n_over:<3d}"
              f"  gact {float(gact_pen):8.2f}"
              f"  dot {dot_peak:6.0f}/{dot_cap:<5.0f}  grad {float(grad_norm):7.2f}"
              f"  scale {scaler.get_scale():.0f}", flush=True)
        min_box = "[x]" if min_grad_risk else "[ ]"
        max_box = "[x]" if max_grad_risk else "[ ]"
        nw = max(len(min_grad_l2_name), len(min_grad_name))
        vw = max(len(gfmt(max_grad_l2)), len(gfmt(max_grad)))
        print(f"{'':2}grad L2         {gfmt(min_grad_l2)} {min_grad_l2_name:<{nw}}"
              f"  ->     {gfmt(max_grad_l2):<{vw}}  {max_grad_l2_name}", flush=True)
        print(f"{'':2}scaled grad {min_box} {gfmt(min_grad)} {min_grad_name:<{nw}}"
              f"  -> {max_box} {gfmt(max_grad):<{vw}}  {max_grad_name}"
              f"  n@fp16_min {n_fp16_min}", flush=True)
        print(f"{'':2}wdl head  h mean {wdl_hidden_abs_mean:.4f}"
              f"  rms {wdl_hidden_rms:.4f}  max {wdl_hidden_abs_max:.4f}"
              f"   logits mean {wdl_logits_abs_mean:.4f}"
              f"  rms {wdl_logits_rms:.4f}  max {wdl_logits_abs_max:.4f}",
              flush=True)
        print(f"{'':12}grad L2 {gfmt(wdl_w2_grad_l2)}"
              f"  mean {gfmt(wdl_w2_grad_abs_mean)}"
              f"  scaled max {gfmt(wdl_w2_grad_scaled_max)}", flush=True)
        if offenders:
            top = "  ".join(f"{name} {rfmt(s)}/{rfmt(mx)} ({tgt})"
                            for name, s, mx, tgt in offenders)
            print(f"{'':2}offenders  {top}", flush=True)
        if has_gate:
            print_gate_ema(conv_gate_ema(model.gated_expandy_convs), "GE")
        print(flush=True)
        fh.flush()

    ckpt = out_dir / f"{spec['name']}{args.tag}.pt"
    torch.save({"model": model.state_dict(), "conv_mode": spec["conv"],
                "tx_mode": spec["tx"], "steps": prev_steps + total_steps}, ckpt)
    print(f"[save] {ckpt}", flush=True)
    del model, opt, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--lr-end", type=float, default=4e-4)
    ap.add_argument("--lr-decay-start", type=int, default=15_000,
                    help="hold lr flat until here, then linearly decay to lr-end")
    
    ap.add_argument("--lr-decay-end", type=int, default=20000,
                    help="reach lr-end here; held flat for any steps beyond")
    
    ap.add_argument("--weight-decay", type=float, default=0.005)
    ap.add_argument("--beta1-span", type=float, default=10.0)
    ap.add_argument("--beta2-span", type=float, default=500.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--grad-clip-start", type=float, default=1.0)
    ap.add_argument("--grad-clip-end", type=float, default=3.0)
    ap.add_argument("--grad-clip-warmup", type=int, default=500)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--bounds-weight", type=float, default=1e-6)

    ap.add_argument("--dot-cap-start", type=float, default=20480)

    ap.add_argument("--dot-cap-end", type=float, default=1024.0)
    ap.add_argument("--dot-cap-hold", type=int, default=25,
                    help="hold dot cap at start until this step, then ramp")
    
    ap.add_argument("--dot-cap-steps", type=int, default=2000,
                    help="ramp dot cap start->end over hold..this, flat after")
    ap.add_argument("--dot-weight", type=float, default=1e-6)
    ap.add_argument("--init-from",
                    default=(r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs"
                             r"\20m_wdl3_pretrain\20m_wdl3_pretrain_model_player.pt"))
    
    ap.add_argument("--random", action="store_true",
                    help="fresh random init for all models (skip warm-start)")
    
    ap.add_argument("--tag", default="",
                    help="suffix for saved checkpoints so runs don't clobber each "
                         "other; defaults to '_rand' when --random is set")
    
    ap.add_argument("--resume", action="store_true",
                    help="continue training each model from its own saved "
                         "checkpoint (out-dir / name+resume-tag .pt) instead of "
                         "warm-starting; adds any missing ema tracking buffers")
    ap.add_argument("--resume-tag", default="_rand",
                    help="tag of the checkpoints to resume from")

    ap.add_argument("--init-rand-blend", type=float, default=0.5)
    
    ap.add_argument("--init-conv-src", type=int, default=1)
    ap.add_argument("--shard-dir",
                    default=(r"C:\Users\Bryan\Data\chessbot_data"
                             r"\training_data\pretrain_shards"))
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="fraction of shard files held out for validation")
    ap.add_argument("--split-seed", type=int, default=1,
                    help="seed for the train/val shard file split")
    ap.add_argument("--out-dir",
                    default=(r"C:\Users\Bryan\Data\chessbot\_data\models"
                             r"\bakeoff_4c2t_d256"))
    ap.add_argument("--only", default=None,
                    help="comma-separated model ids or names to run "
                         "(default: every MODEL_SPECS entry in order)")
    args = ap.parse_args()
    if args.random:
        args.init_from = ""
        if not args.tag:
            args.tag = "_rand"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = Tee(out_dir / f"bakeoff_{stamp}.log", sys.stdout)
    sys.stderr = Tee(out_dir / f"bakeoff_{stamp}.err", sys.stderr)

    torch.manual_seed(0)
    files = sorted(glob.glob(str(Path(args.shard_dir) / "*.pkl.gz")))
    random.Random(args.split_seed).shuffle(files)
    n_val = max(1, int(len(files) * args.val_frac))
    val_files, train_files = files[:n_val], files[n_val:]
    print(f"[data] {len(train_files)} train shards / {len(val_files)} val shards"
          f" (split_seed={args.split_seed})", flush=True)
    train_data = SparseShardBatches(args.shard_dir, seed=args.split_seed,
                                    files=train_files,
                                    batch_size=args.batch, pool_shards=6,
                                    ready_batches=32)

    todo = list(MODEL_SPECS.values())
    if args.only:
        by_name = {spec["name"]: spec for spec in MODEL_SPECS.values()}
        todo = []
        for key in args.only.split(","):
            if key in MODEL_SPECS:
                todo.append(MODEL_SPECS[key])
            elif key in by_name:
                todo.append(by_name[key])
            else:
                raise ValueError(f"--only: unknown model {key!r}")

    csv_path = out_dir / f"bakeoff_metrics_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        print(f"[csv] {csv_path}", flush=True)
        for spec in todo:
            train_one(spec, args, train_data, val_files, writer, fh, out_dir)
            fh.flush()
    print(f"\n[done] {len(todo)} models trained -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
