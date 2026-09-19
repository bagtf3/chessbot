"""Architecture bake-off: 4c2t d256 trunk, float trunk-architecture study.

Plain float training (fp16 AMP + GradScaler), no int8: no weight/activation
quantization, no QAT, no ONNX/TRT export.  The trunk block is the only variable;
stem, pos, policy heads and value head match the prod module layout so prod
weights warm-start directly.

    m1   4 plain conv                    2 plain-FF tx      baseline
    m2   2 plain + 2 maxnet .1/.9        2 plain-FF tx      routing convs
    m4   4 plain conv                    2 DeciderCombiner  routing FF (d256)
    m5   4 plain conv                    2 DeciderCombiner  routing FF (d384)
    m6   4 plain conv                    2 DeciderCombiner  routing FF (d512)
    m7   4 plain conv                    2 SwiGLU-768 tx    gated FF
    m8   1 plain + 3 gated conv          2 plain-FF tx      GLU convs
    m9   2 plain + 2 decider conv        2 plain-FF tx      dual-branch convs
    m11  2 plain + 2 expandy conv        2 plain-FF tx      separable convs
    m11-fast  2 plain + 2 expandy-fast   2 plain-FF tx      expandy, no mid norm
    m11-fat   2 plain + 2 expandy-fat    2 plain-FF tx      expandy-fast, 2D middle
    m14  2 plain + 2 SE-expandy conv     2 plain-FF tx      expandy-fast, D/16 gate
    m12  2 plain + 2 DC-softmax conv     2 plain-FF tx      softmax-routed convs
    m13  2 plain + 2 gated-expandy conv  2 plain-FF tx      gated separable convs
    m15  4 reverse-conv blocks            2 plain-FF tx      board squares as channels
    m16  4 plain conv                     2 MHA-no-out tx    attention without output proj

Value head is the d256 single-head W/D/L readout off acc0.

Routing blocks (maxnet, DeciderCombiner, decider/gated conv) are lightly pressured
to decide rather than blend by one grid penalty on the live gate output:
  span gridpen    sum g*(1-g) only for gates in (lo,hi), lifts gates off the dead
                  0.5 without forcing a hard rail
A high fixed routing sharpness (ROUTING_K) provides additional separation.

Range bangers leash the fp16-hot real quantities: the raw policy-gate logits, the
logsigmoid suppression term, the policy dot product, and the SwiGLU gate*up
product.  All per-report-step metrics for every model go to one CSV; each model
is saved.

    python scripts/int8/bakeoff_4c2t_d256.py
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
DC_SEQ = TOKENS + 8  # transformer sequence length seen by DC gates (board + globals)
DC_EMA_BETA = 1.0 - 1.0 / 500.0  # 500-step EMA for gate mean/std/range tracking
SPAN_GRID_START, SPAN_GRID_END, SPAN_GRID_RAMP = 205_500, 208_500, 1000  # gate span-gridpen window
SPAN_GRID_LO, SPAN_GRID_HI = 0.2, 0.8  # only penalize gates inside this band (light)
SPAN_GRID_WEIGHT = 1e-6  # gate span gridpen coefficient
ROUTING_K = 2.0  # fixed routing sharpness for maxnet and DC gates
DCSM_AB_GROUPS = 4  # grouped-conv groups for the DCSoftMax content/gate branches
DCSM_SM_GROUPS = (64, 64)  # per-block softmax groups: block0=64, block1=64

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

    def __init__(self, dim=D, heads=HEADS, output_proj=True):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim) if output_proj else None
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
        return self.o(h) if self.o is not None else h


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


class ReverseConvBlock(nn.Module):
    """ReverseConv on [B, 64 board-square channels, 16, 16 features]."""

    def __init__(self):
        super().__init__()
        self.norm = L1Norm(D)
        self.c1 = nn.Conv2d(TOKENS, 2 * TOKENS, 3, padding=1)
        self.c2 = nn.Conv2d(2 * TOKENS, TOKENS, 3, padding=1, bias=False)

    def forward(self, x):
        b = x.shape[0]
        h = self.norm(x.reshape(b, BOARD, BOARD, D)).reshape(b, TOKENS, 16, 16)
        h = F.leaky_relu(self.c1(h), 0.01)
        h = self.c2(h)
        return x + F.leaky_relu(h, 0.01)


class MaxNetConvBlock(nn.Module):
    """Residual conv block with two competing second-conv hypotheses.  A sigmoid
    router sigmoid(k*(a-b)) lets both branches learn; hard mode swaps to an exact
    compare/select.  A ReLU on the blend keeps the block nonlinear (a soft blend of
    two linear convs is otherwise linear).  Per-channel EMAs track routing."""

    def __init__(self):
        super().__init__()
        self.norm = L1Norm(D)
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        self.c2a = nn.Conv2d(D, D, 3, padding=1, bias=False)
        self.c2b = nn.Conv2d(D, D, 3, padding=1, bias=False)
        self.k = ROUTING_K
        self.hard = False
        self.register_buffer("gate_mean_ema", torch.full((D, BOARD, BOARD), 0.5))
        self.register_buffer("gate_std_ema", torch.full((D, BOARD, BOARD), 0.16))
        self.register_buffer("gate_range_ema", torch.zeros(D, BOARD, BOARD))
        self.last_gate_live = None

    def forward(self, x):
        h = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.leaky_relu(self.c1(h), 0.01)
        a = self.c2a(h)
        b = self.c2b(h)
        soft_gate = torch.sigmoid(self.k.to(a.dtype) * (a - b))
        self.last_gate_live = soft_gate
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                deb = DC_EMA_BETA
                gf = soft_gate.detach().float()
                self.gate_mean_ema.mul_(deb).add_(gf.mean(0), alpha=1 - deb)
                self.gate_std_ema.mul_(deb).add_(gf.std(0), alpha=1 - deb)
                
                gr = gf.amax(0) - gf.amin(0)
                self.gate_range_ema.mul_(deb).add_(gr, alpha=1 - deb)

        if self.hard:
            h = torch.where(a > b, a, b)
        else:
            h = soft_gate * a + (1.0 - soft_gate) * b
        return x + F.relu(h)


class GatedConvBlock(nn.Module):
    """Gated conv (GLU-conv, Dauphin 2017): a 3x3 value conv gated per (channel,
    square) by a 1x1 sigmoid decider.  c2 is the LINEAR value branch; the sigmoid
    gate is the block's nonlinearity, mirroring SwiGLU's linear up-branch."""

    def __init__(self):
        super().__init__()
        self.norm = L1Norm(D)
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        self.cgate = nn.Conv2d(D, D, 1)
        self.c2 = nn.Conv2d(D, D, 3, padding=1, bias=False)
        self.register_buffer("gate_mean_ema", torch.full((D, BOARD, BOARD), 0.5))
        self.register_buffer("gate_std_ema", torch.full((D, BOARD, BOARD), 0.16))
        self.register_buffer("gate_range_ema", torch.zeros(D, BOARD, BOARD))
        self.last_gate_live = None

    def forward(self, x):
        h = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.leaky_relu(self.c1(h), 0.01)
        gate = torch.sigmoid(self.cgate(h))
        val = self.c2(h)
        self.last_gate_live = gate
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                gf = gate.detach().float()
                self.gate_mean_ema.mul_(DC_EMA_BETA).add_(
                    gf.mean(0), alpha=1 - DC_EMA_BETA)
                self.gate_std_ema.mul_(DC_EMA_BETA).add_(gf.std(0), alpha=1 - DC_EMA_BETA)
                gr = gf.amax(0) - gf.amin(0)
                self.gate_range_ema.mul_(DC_EMA_BETA).add_(
                    gr, alpha=1 - DC_EMA_BETA)
        return x + gate * val


class DeciderConvBlock(nn.Module):
    """Conv-trunk DeciderCombiner: two competing second convs (c2a/c2b, leaky)
    routed by an independent 1x1 sigmoid valve.  valve*a + (1-valve)*b, residual."""

    def __init__(self):
        super().__init__()
        self.norm = L1Norm(D)
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        self.valve = nn.Conv2d(D, D, 1)
        self.c2a = nn.Conv2d(D, D, 3, padding=1, bias=False)
        self.c2b = nn.Conv2d(D, D, 3, padding=1, bias=False)
        self.register_buffer("gate_mean_ema", torch.full((D, BOARD, BOARD), 0.5))
        self.register_buffer("gate_std_ema", torch.full((D, BOARD, BOARD), 0.16))
        self.register_buffer("gate_range_ema", torch.zeros(D, BOARD, BOARD))
        self.last_gate_live = None

    def forward(self, x):
        n = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.leaky_relu(self.c1(n), 0.05)
        valve = torch.sigmoid(self.valve(h))
        a = F.leaky_relu(self.c2a(h), 0.02)
        b = F.leaky_relu(self.c2b(h), 0.02)
        self.last_gate_live = valve
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                gf = valve.detach().float()
                self.gate_mean_ema.mul_(DC_EMA_BETA).add_(
                    gf.mean(0), alpha=1 - DC_EMA_BETA)
                self.gate_std_ema.mul_(DC_EMA_BETA).add_(gf.std(0), alpha=1 - DC_EMA_BETA)
                gr = gf.amax(0) - gf.amin(0)
                self.gate_range_ema.mul_(DC_EMA_BETA).add_(
                    gr, alpha=1 - DC_EMA_BETA)
        return x + valve * a + (1.0 - valve) * b


class ExpandyConvBlock(nn.Module):
    """Separable/inverted-bottleneck conv block: grouped 3x3 expand (D->2D) + 1x1
    contract (2D->D), leaky, twice, residual.  ~0.72x a full ConvBlock's params at
    groups=4, with the leaky activating in the wider 2D space before projecting."""

    def __init__(self, groups=4):
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(2 * D, D, 1, bias=False)
        self.n2 = L1Norm(D)
        self.c2 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))
        h = self.n2(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p2(F.leaky_relu(self.c2(h), 0.05))
        return x + h


class ExpandyFastConvBlock(nn.Module):
    """ExpandyConvBlock with the mid-block L1Norm removed: one norm, then two
    grouped-3x3 expand -> leaky at 2D -> 1x1 contract stages.  TRT fuses the
    leaky into the grouped conv, so this benches ~13% faster than ExpandyConvBlock."""

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


class ExpandyFatConvBlock(nn.Module):
    """ExpandyFastConvBlock that stays at 2D through the middle: grouped 3x3
    D->2D, leaky, 1x1 2D->2D, grouped 3x3 2D->2D, leaky, 1x1 2D->D."""

    def __init__(self, groups=4):
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(2 * D, 2 * D, 1, bias=False)
        self.c2 = nn.Conv2d(2 * D, 2 * D, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))
        h = self.p2(F.leaky_relu(self.c2(h), 0.05))
        return x + h


class GatedExpandedConvBlock(nn.Module):
    """ExpandyFastConvBlock with an m8-style gate: a 1x1 sigmoid off the stage-1
    output gates the stage-2 output.  x + gate*stage2; gate mean/std/range EMA."""

    def __init__(self, groups=4):
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, 2*D, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(2*D, D, 1, bias=False)

        self.c_gate = nn.Conv2d(D, D, 1)

        self.c2 = nn.Conv2d(D, 2*D, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(2*D, D, 1, bias=False)

        self.register_buffer("gate_mean_ema", torch.full((D, BOARD, BOARD), 0.5))
        self.register_buffer("gate_std_ema", torch.full((D, BOARD, BOARD), 0.16))
        self.register_buffer("gate_range_ema", torch.zeros(D, BOARD, BOARD))
        self.last_gate_live = None

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))

        gate = torch.sigmoid(self.c_gate(h))
        self.last_gate_live = gate

        h = self.p2(F.leaky_relu(self.c2(h), 0.05))

        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                deb = DC_EMA_BETA
                gf = gate.detach().float()
                self.gate_mean_ema.mul_(deb).add_(gf.mean(0), alpha=1 - deb)
                self.gate_std_ema.mul_(deb).add_(gf.std(0), alpha=1 - deb)
                gr = gf.amax(0) - gf.amin(0)
                self.gate_range_ema.mul_(deb).add_(gr, alpha=1 - deb)
        
        return x + gate * h


class SEExpandedConvBlock(nn.Module):
    """ExpandyFastConvBlock with an SE-shaped spatial gate: 1x1 D->D/16 -> leaky ->
    1x1 D/16->D -> sigmoid off the stage-1 output, no global pool.  x + gate*stage2."""

    def __init__(self, groups=4):
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, 2*D, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(2*D, D, 1, bias=False)

        self.c_gate_down = nn.Conv2d(D, D//16, 1)
        self.c_gate_up = nn.Conv2d(D//16, D, 1)

        self.c2 = nn.Conv2d(D, 2*D, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(2*D, D, 1, bias=False)

        self.register_buffer("gate_mean_ema", torch.full((D, BOARD, BOARD), 0.5))
        self.register_buffer("gate_std_ema", torch.full((D, BOARD, BOARD), 0.16))
        self.register_buffer("gate_range_ema", torch.zeros(D, BOARD, BOARD))
        self.last_gate_live = None

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))

        gd = F.leaky_relu(self.c_gate_down(h), 0.02)
        gate = torch.sigmoid(self.c_gate_up(gd))
        self.last_gate_live = gate

        h = self.p2(F.leaky_relu(self.c2(h), 0.05))

        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                deb = DC_EMA_BETA
                gf = gate.detach().float()
                self.gate_mean_ema.mul_(deb).add_(gf.mean(0), alpha=1 - deb)
                self.gate_std_ema.mul_(deb).add_(gf.std(0), alpha=1 - deb)
                gr = gf.amax(0) - gf.amin(0)
                self.gate_range_ema.mul_(deb).add_(gr, alpha=1 - deb)
        
        return x + gate * h


class DCSoftMaxConvBlock(nn.Module):
    """DeciderCombiner conv with a softmax competition gate.  A full 3x3 c1, then
    two grouped expand->contract content branches (xa, xb) and a grouped gate branch
    (xg).  xg is split into sm_groups groups of D/sm_groups channels; a per-(group,
    token) softmax over the channels-in-group makes the blend weight h a competition
    (each token's h sums to sm_groups over D, so sm_groups sets the xa/xb mass split).
    Blend h*xa + (1-h)*xb, residual.  h is tracked with the same mean/std gate EMA as
    the sigmoid blocks and participates in the span gridpen."""

    def __init__(self, ab_groups=DCSM_AB_GROUPS, sm_groups=64):
        super().__init__()
        self.sm_groups = sm_groups
        self.norm = L1Norm(D)
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        
        self.dw_a = nn.Conv2d(D, D, 3, padding=1, groups=ab_groups)
        self.p_a = nn.Conv2d(D, D, 1, bias=False)
        self.dw_b = nn.Conv2d(D, D, 3, padding=1, groups=ab_groups)
        self.p_b = nn.Conv2d(D, D, 1, bias=False)
        
        self.register_buffer("gate_mean_ema", torch.full((D, BOARD, BOARD), 0.5))
        self.register_buffer("gate_std_ema", torch.full((D, BOARD, BOARD), 0.16))
        self.register_buffer("gate_range_ema", torch.zeros(D, BOARD, BOARD))
        self.last_gate_live = None

    def forward(self, x):
        n = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.leaky_relu(self.c1(n), 0.025)

        xa = self.p_a(F.leaky_relu(self.dw_a(h), 0.05))
        xb = self.p_b(F.leaky_relu(self.dw_b(h), 0.05))

        xg = xa
        B, C, H, W = xg.shape
        g = xg.view(B, self.sm_groups, C // self.sm_groups, H * W).softmax(dim=2)
        g = g.reshape(B, C, H, W)
        self.last_gate_live = g
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                gf = g.detach().float()
                self.gate_mean_ema.mul_(DC_EMA_BETA).add_(
                    gf.mean(0), alpha=1 - DC_EMA_BETA)
                self.gate_std_ema.mul_(DC_EMA_BETA).add_(gf.std(0), alpha=1 - DC_EMA_BETA)
                gr = gf.amax(0) - gf.amin(0)
                self.gate_range_ema.mul_(DC_EMA_BETA).add_(
                    gr, alpha=1 - DC_EMA_BETA)
        return x + g * xa + (1.0 - g) * xb


class DeciderCombiner(nn.Module):
    """Learned per-element router replacing the 4x FF.  Three d->hidden
    projections: two content branches a, b and a decider h.  Train blends with
    sigmoid(k*h); hard mode selects with where(h>0) under a soft straight-through
    gradient.  hidden>d up-projects, decides wide, steps back to d.  LeakyReLU(0.05)
    on the selected output adds pointwise nonlinearity, signed for the residual."""

    def __init__(self, d=D, hidden=None):
        super().__init__()
        hidden = hidden or d
        self.A = nn.Linear(d, hidden, bias=False)
        self.B = nn.Linear(d, hidden, bias=False)
        self.H = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False) if hidden != d else None
        self.hard = False
        self.k = nn.Parameter(torch.tensor(1.0))  # learnable routing sharpness (abs)
        self.last_gate = None
        self.last_gate_live = None
        self.register_buffer("mean_ema", torch.full((DC_SEQ, hidden), 0.5))
        self.register_buffer("std_ema", torch.full((DC_SEQ, hidden), 0.16))
        self.register_buffer("range_ema", torch.zeros(DC_SEQ, hidden))

    def forward(self, x):
        a = self.A(x)
        b = self.B(x)
        h = self.H(x)
        g = torch.sigmoid(self.k.abs().to(a.dtype) * h)
        self.last_gate = g.detach()
        self.last_gate_live = g
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                gf = g.detach().float()
                self.mean_ema.mul_(DC_EMA_BETA).add_(gf.mean(0), alpha=1 - DC_EMA_BETA)
                self.std_ema.mul_(DC_EMA_BETA).add_(gf.std(0), alpha=1 - DC_EMA_BETA)
                gr = gf.amax(0) - gf.amin(0)
                self.range_ema.mul_(DC_EMA_BETA).add_(
                    gr, alpha=1 - DC_EMA_BETA)
        if self.hard:
            hard = torch.where(h > 0, a, b)
            soft = g * a + (1.0 - g) * b
            sel = soft + (hard - soft).detach()
        else:
            sel = g * a + (1.0 - g) * b
        sel = F.leaky_relu(sel, 0.05)
        return self.down(sel) if self.down is not None else sel


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


class TxBlockFFNoOut(nn.Module):
    """Plain FF transformer whose attention result has no output projection."""

    def __init__(self):
        super().__init__()
        self.n1 = L1Norm(D)
        self.attn = Attention(D, HEADS, output_proj=False)
        self.n2 = L1Norm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n)
        h = F.gelu(self.ff1(self.n2(x)))
        return x + self.ff2(h)


class TxBlockDC(nn.Module):
    """Transformer block whose FF is replaced by a DeciderCombiner."""

    def __init__(self, dc_hidden=None):
        super().__init__()
        self.n1 = L1Norm(D)
        self.attn = Attention(D, HEADS)
        self.n2 = L1Norm(D)
        self.dc = DeciderCombiner(D, hidden=dc_hidden)

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n)
        return x + self.dc(self.n2(x))


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

    def __init__(self, conv_mode="vanilla", tx_mode="vanilla", dc_hidden=None):
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

        self.max_convs = nn.ModuleList([])
        self.gated_convs = nn.ModuleList([])
        self.dc_convs = nn.ModuleList([])
        self.expandy_convs = nn.ModuleList([])
        self.gated_expandy_convs = nn.ModuleList([])
        self.sm_convs = nn.ModuleList([])
        self.reverse_convs = nn.ModuleList([])
        if conv_mode == "vanilla":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(4)])
        elif conv_mode == "reverse":
            self.reverse_stem_layout = True
            self.convs = nn.ModuleList([])
            self.reverse_convs = nn.ModuleList([ReverseConvBlock() for _ in range(4)])
        elif conv_mode == "rainbow":
            self.convs = nn.ModuleList([ConvBlock()])
            self.expandy_convs = nn.ModuleList([ExpandyFastConvBlock()])
            self.gated_expandy_convs = nn.ModuleList([GatedExpandedConvBlock()])
            self.sm_convs = nn.ModuleList([
                DCSoftMaxConvBlock(ab_groups=DCSM_AB_GROUPS,
                                   sm_groups=DCSM_SM_GROUPS[0])])
        elif conv_mode == "expandy":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(2)])
            self.expandy_convs = nn.ModuleList([ExpandyConvBlock() for _ in range(2)])
        elif conv_mode == "expandy_fast":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(2)])
            self.expandy_convs = nn.ModuleList(
                [ExpandyFastConvBlock() for _ in range(2)])
        elif conv_mode == "expandy_fat":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(2)])
            self.expandy_convs = nn.ModuleList(
                [ExpandyFatConvBlock() for _ in range(2)])
        elif conv_mode == "gated_expandy":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(2)])
            self.gated_expandy_convs = nn.ModuleList(
                [GatedExpandedConvBlock() for _ in range(2)])
        elif conv_mode == "se_expandy":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(2)])
            self.gated_expandy_convs = nn.ModuleList(
                [SEExpandedConvBlock() for _ in range(2)])
        elif conv_mode == "dcsoftmax":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(2)])
            self.sm_convs = nn.ModuleList([
                DCSoftMaxConvBlock(ab_groups=DCSM_AB_GROUPS, sm_groups=sg)
                for sg in DCSM_SM_GROUPS])
        elif conv_mode == "maxnet":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(2)])
            self.max_convs = nn.ModuleList([MaxNetConvBlock() for _ in range(2)])
        elif conv_mode == "dcconv":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(2)])
            self.dc_convs = nn.ModuleList([DeciderConvBlock() for _ in range(2)])
        elif conv_mode == "gated":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(1)])
            self.gated_convs = nn.ModuleList([GatedConvBlock() for _ in range(3)])
        else:
            raise ValueError(f"conv_mode={conv_mode}")

        self.pos = nn.Parameter(torch.randn(TOKENS, D) * 0.02)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.global_tokens = nn.Parameter(torch.randn(1, 8, D) * 0.02)

        if tx_mode == "vanilla":
            self.blocks = nn.ModuleList([TxBlockFF() for _ in range(2)])
        elif tx_mode == "dc":
            self.blocks = nn.ModuleList([TxBlockDC(dc_hidden) for _ in range(2)])
        elif tx_mode == "swiglu":
            self.blocks = nn.ModuleList([TxBlockSwiGLU() for _ in range(2)])
        elif tx_mode == "noout":
            self.blocks = nn.ModuleList([TxBlockFFNoOut() for _ in range(2)])
        elif tx_mode == "ff_swiglu":
            self.blocks = nn.ModuleList([TxBlockFF(), TxBlockSwiGLU()])
        else:
            raise ValueError(f"tx_mode={tx_mode}")
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

    def set_k(self, k):
        for block in self.max_convs:
            block.k = k

    def dc_ks(self):
        return [float(b.dc.k.abs()) for b in self.blocks if hasattr(b, "dc")]

    def set_hard(self, hard):
        for block in self.max_convs:
            block.hard = hard
        for block in self.blocks:
            if hasattr(block, "dc"):
                block.dc.hard = hard

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
        if getattr(self, "reverse_stem_layout", False):
            return x.reshape(b, TOKENS, 16, 16)
        return x.reshape(b, BOARD, BOARD, D).permute(0, 3, 1, 2)

    def forward(self, x):
        if x.shape[1] != XC0H_LEN:
            raise ValueError(f"expected xc0h [B,{XC0H_LEN}], got {tuple(x.shape)}")
        b = x.shape[0]
        x = self.stem(x)
        for block in self.convs:
            x = block(x)
        for block in self.max_convs:
            x = block(x)
        for block in self.dc_convs:
            x = block(x)
        for block in self.expandy_convs:
            x = block(x)
        for block in self.gated_expandy_convs:
            x = block(x)
        for block in self.sm_convs:
            x = block(x)
        for block in self.gated_convs:
            x = block(x)
        for block in self.reverse_convs:
            x = block(x)
        if self.conv_mode == "reverse":
            x = x.reshape(b, BOARD, BOARD, D).permute(0, 3, 1, 2)
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


def dc_winrate_pcts(model):
    """Per-block p50/p80/p95 of DC branch-A winrate across channels."""
    qs = torch.tensor([0.5, 0.8, 0.95])
    rows = []
    for i, block in enumerate(model.blocks):
        dc = getattr(block, "dc", None)
        if dc is None or dc.last_gate is None:
            continue
        g = dc.last_gate.float().mean(dim=(0, 1))
        p50, p80, p95 = torch.quantile(g, qs.to(g.device)).tolist()
        rows.append((i, p50, p80, p95))
    return rows


def ema_summary(ema):
    """Six-number summary (min, max, mean, p25, p50, p75) of an EMA tensor."""
    e = ema.detach().float().flatten()
    qs = torch.tensor([0.25, 0.5, 0.75], device=e.device)
    p25, p50, p75 = torch.quantile(e, qs).tolist()
    return (float(e.min()), float(e.max()), float(e.mean()), p25, p50, p75)


def dc_gate_ema(model):
    """Per-block mean/std/range EMA summaries of the DC gate."""
    rows = []
    for i, block in enumerate(model.blocks):
        dc = getattr(block, "dc", None)
        if dc is None:
            continue
        rows.append((
            i,
            ema_summary(dc.mean_ema),
            ema_summary(dc.std_ema),
            ema_summary(dc.range_ema),
        ))
    return rows


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


def conv_gate_blocks(model):
    """The populated conv-gate block list for the current conv mode (at most one is
    non-empty), for CSV range-EMA logging."""
    return (list(model.max_convs) or list(model.dc_convs)
            or list(model.gated_convs) or list(model.gated_expandy_convs)
            or list(model.sm_convs))


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


def dc_gate_health(model, std_eps=0.05, rail=0.05):
    """Per-block DC channel health, (i, vary, stuck0, stuck1, stuckmid, total)."""
    rows = []
    for i, block in enumerate(model.blocks):
        dc = getattr(block, "dc", None)
        if dc is None or dc.last_gate is None:
            continue
        g = dc.last_gate.float()
        mean_c = g.mean(dim=(0, 1))
        std_c = g.std(dim=(0, 1))
        stuck = std_c <= std_eps
        s0 = int((stuck & (mean_c < rail)).sum())
        s1 = int((stuck & (mean_c > 1 - rail)).sum())
        smid = int((stuck & (mean_c >= rail) & (mean_c <= 1 - rail)).sum())
        rows.append((i, int((~stuck).sum()), s0, s1, smid, mean_c.numel()))
    return rows


def dc_stats(model):
    """Per-block DeciderCombiner gate summary (mean sigmoid(h) per channel)."""
    rows = []
    for i, block in enumerate(model.blocks):
        dc = getattr(block, "dc", None)
        if dc is None or dc.last_gate is None:
            continue
        g = dc.last_gate.float().mean(dim=(0, 1))
        rows.append((i, float(g.mean()), float(g.min()), float(g.max())))
    return rows


def span_gridpen(model, lo=SPAN_GRID_LO, hi=SPAN_GRID_HI):
    """Light span gridpen over every routing gate (maxnet, dc-conv, gated-conv,
    transformer-DC).  Sum of g*(1-g) over in-band elements only: gates past a rail
    get no penalty, so they aren't forced hard 0/1, but dead-middle gates are pushed
    toward decisive."""
    gates = [b.last_gate_live for b in model.max_convs] \
        + [b.last_gate_live for b in model.dc_convs] \
        + [b.last_gate_live for b in model.gated_convs] \
        + [b.last_gate_live for b in model.gated_expandy_convs] \
        + [b.last_gate_live for b in model.sm_convs] \
        + [b.dc.last_gate_live for b in model.blocks if hasattr(b, "dc")]
    total = 0.0
    for g in gates:
        if g is None:
            continue
        gm = g * (1.0 - g)
        total = total + gm[(g > lo) & (g < hi)].sum()
    return total


def has_routing_gates(model):
    return bool(model.max_convs) or bool(model.dc_convs) or bool(model.gated_convs) \
        or bool(model.gated_expandy_convs) or bool(model.sm_convs) \
        or any(hasattr(b, "dc") for b in model.blocks)


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
    pos, plain convs, maxnet convs (c1/c2a from one source conv, c2b from a distinct
    one so the branches differ), transformer attn (+ FF only for plain-FF blocks),
    policy heads, gate.  L1Norm scales/biases are intentionally fresh.  Also fresh:
    expandy convs, DeciderCombiner projections, and the d256 value head."""
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

    n_conv, n_max = len(model.convs), len(model.max_convs)
    for i, block in enumerate(model.max_convs):
        c = f"pre.{conv_src + n_conv + i}"            # own conv: c1, c2a branch
        cb = f"pre.{conv_src + n_conv + n_max + i}"   # a distinct conv for the c2b branch
        load_linear(block.c1, sd, f"{c}.c1", blend)
        load_linear(block.c2a, sd, f"{c}.c2", blend)
        load_linear(block.c2b, sd, f"{cb}.c2", blend)

    n_dcc = len(model.dc_convs)
    for i, block in enumerate(model.dc_convs):
        c = f"pre.{conv_src + n_conv + i}"            # own conv: c1, c2a branch
        cb = f"pre.{conv_src + n_conv + n_dcc + i}"   # a distinct conv for the c2b branch
        load_linear(block.c1, sd, f"{c}.c1", blend)
        load_linear(block.c2a, sd, f"{c}.c2", blend)
        load_linear(block.c2b, sd, f"{cb}.c2", blend)   # valve stays fresh

    for i, block in enumerate(model.gated_convs):
        c = f"pre.{conv_src + n_conv + i}"            # c1 + c2 value; cgate stays fresh
        load_linear(block.c1, sd, f"{c}.c1", blend)
        load_linear(block.c2, sd, f"{c}.c2", blend)

    for i, block in enumerate(model.sm_convs):
        c = f"pre.{conv_src + n_conv + i}"            # full c1; branches stay fresh
        load_linear(block.c1, sd, f"{c}.c1", blend)

    for i, block in enumerate(model.blocks):
        t = f"blocks.{i}"
        for w in ("q", "k", "v"):
            load_linear(getattr(block.attn, w), sd, f"{t}.attn.{w}", blend)
        if block.attn.o is not None:
            load_linear(block.attn.o, sd, f"{t}.attn.o", blend)
        if isinstance(block, (TxBlockFF, TxBlockFFNoOut)):
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
          f"{Path(ckpt_path).name} (value head + routing branches fresh)", flush=True)


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


# Architecture catalogue.  Keep every bake-off candidate registered here; changing
# RUN_ORDER below is the only thing needed to choose which candidates run by default.
MODEL_SPECS = {
    "m1": dict(name="m1_conv-van_tx-van", conv="vanilla", tx="vanilla"),
    "m2": dict(name="m2_conv-maxnet_tx-van", conv="maxnet", tx="vanilla"),
    "m6": dict(
        name="m6_conv-van_tx-dc1024", conv="vanilla",
        tx="dc", dc_hidden=1024
    ),

    "m7": dict(name="m7_conv-van_tx-swiglu", conv="vanilla", tx="swiglu"),
    "m8": dict(name="m8_conv-gated_tx-van", conv="gated", tx="vanilla"),
    "m11": dict(name="m11_conv-expandy_tx-van", conv="expandy_fast", tx="vanilla"),
    
    "m11-fast": dict(
        name="m11-fast_conv-expandyfast_tx-van",
        conv="expandy_fast",tx="vanilla"
    ),

    "m12": dict(name="m12_conv-dcsoftmax_tx-van", conv="dcsoftmax", tx="vanilla"),
    "m13": dict(name="m13_conv-gated-expandy_tx-van", conv="gated_expandy", tx="vanilla"),
    "m14": dict(name="m14_conv-se-expandy_tx-van", conv="se_expandy", tx="vanilla"),
    "m15": dict(name="m15_conv-reverse_tx-van", conv="reverse", tx="vanilla"),
    "m16": dict(name="m16_conv-van_tx-noout", conv="vanilla", tx="noout"),
    "m69": dict(name="m69-rainbow-variant", conv="rainbow", tx="ff_swiglu"),

    "m2-6": dict(
        name="m2-6_conv-maxnet_tx-dc1024", conv="maxnet",
        tx="dc", dc_hidden=1024
    ),

    "m2-7": dict(name="m2-7_conv-maxnet_tx-swiglu", conv="maxnet", tx="swiglu"),

    "m11-6": dict(
        name="m11-6_conv-expandy_tx-dc1024", conv="expandy_fast",
        tx="dc", dc_hidden=1024
    ),

    "m11-7": dict(name="m11-7_conv-expandy_tx-swiglu", conv="expandy_fast", tx="swiglu"),
}

# Default execution order.  m10 (the MaxNet + DC combination) is intentionally
# not registered as a bake-off candidate.
#RUN_ORDER = ("m12", "m11", "m6", "m1", "m2", "m7", "m8")
RUN_ORDER = ("m2-6", "m2-7", "m11-6", "m11-7")
MODELS = [MODEL_SPECS[model_id] for model_id in RUN_ORDER]

SCALER_MAX_SCALE = 2.0 ** 16
FP16_MIN = 5.960464477539063e-8
FP16_MAX = 65504.0


def gfmt(v):
    return f"{v:.6f}" if v == 0 or abs(v) >= 1e-5 else f"{v:.2e}"


CSV_FIELDS = [
    "model", "conv_mode", "tx_mode", "step", "phase",
    "pol_ce", "wdl_ce", "mse", "corr", "grad_scale",
    "range_pen", "range_over", "gact_pen", "dot_peak", "dot_cap", "dot_pen",
    "grad_global", "grad_mean", "grad_median",
    "max_grad_l2", "max_grad_l2_name", "min_grad_l2", "min_grad_l2_name",
    "max_grad", "max_grad_name", "min_grad", "min_grad_name", "n_fp16_min",
    "wdl_hidden_abs_mean", "wdl_hidden_rms", "wdl_hidden_abs_max",
    "wdl_logits_abs_mean", "wdl_logits_rms", "wdl_logits_abs_max",
    "wdl_w2_grad_l2", "wdl_w2_grad_abs_mean", "wdl_w2_grad_scaled_max",
    "clip",
    "k", "grid_pen",
    "dc0_gmean", "dc0_gmin", "dc0_gmax", "dc0_p50", "dc0_p80", "dc0_p95",
    "dc0_vary", "dc0_stuck0", "dc0_stuck1", "dc0_stuckmid",
    "dc0_mema_min", "dc0_mema_max", "dc0_sema_min", "dc0_sema_p50",
    "dc0_rema_mean", "dc0_rema_p50",
    "dc1_gmean", "dc1_gmax", "dc1_gmin", "dc1_p50", "dc1_p80", "dc1_p95",
    "dc1_vary", "dc1_stuck0", "dc1_stuck1", "dc1_stuckmid",
    "dc1_mema_min", "dc1_mema_max", "dc1_sema_min", "dc1_sema_p50",
    "dc1_rema_mean", "dc1_rema_p50",
    "cg0_rema_mean", "cg0_rema_p50", "cg1_rema_mean", "cg1_rema_p50",
    "cg2_rema_mean", "cg2_rema_p50",
]


def train_one(spec, args, train_data, val_files, writer, fh, out_dir):
    val_data = SparseShardBatches(args.shard_dir, seed=args.split_seed + 1,
                                  files=val_files,
                                  batch_size=args.val_batch, pool_shards=2,
                                  ready_batches=4)
    model = BakeoffModel(conv_mode=spec["conv"], tx_mode=spec["tx"],
                         dc_hidden=spec.get("dc_hidden")).to(args.device)
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
    has_maxnet = spec["conv"] == "maxnet"
    has_gated = spec["conv"] == "gated"
    has_gated_expandy = spec["conv"] in ("gated_expandy", "se_expandy")
    has_dcconv = spec["conv"] == "dcconv"
    has_sm = spec["conv"] == "dcsoftmax"
    has_dc = spec["tx"] == "dc"
    has_span = has_routing_gates(model)
    has_routing = has_span
    print(f"\n=== {spec['name']}  conv={spec['conv']} tx={spec['tx']}"
          f"  steps={total_steps} ===", flush=True)
    print(f"[opt] WD={args.weight_decay} on {len(decay)} tensors;"
          f" {len(no_decay)} WD-free  lr {args.lr}->{args.lr_end} decay"
          f" {args.lr_decay_start}-{args.lr_decay_end}", flush=True)
    if has_span:
        print(f"[span-gridpen] band [{SPAN_GRID_LO},{SPAN_GRID_HI}] steps "
              f"{SPAN_GRID_START}-{SPAN_GRID_END} ramp {SPAN_GRID_RAMP} "
              f"(w={SPAN_GRID_WEIGHT})", flush=True)

    for step in range(start_step, total_steps):
        if has_maxnet:
            model.set_k(ROUTING_K)
        gridpen_on = has_span and SPAN_GRID_START <= step < SPAN_GRID_END
        phase = "span" if gridpen_on else "train"
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
            grid_pen = 0.0
            if gridpen_on:
                grid_pen = span_gridpen(model)
                w = SPAN_GRID_WEIGHT * ramp(step, SPAN_GRID_START, SPAN_GRID_RAMP)
                loss = loss + w * grid_pen
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
        ks = model.dc_ks() if has_dc else ([ROUTING_K] if has_maxnet else [])
        kstr = "/".join(f"{v:.2f}" for v in ks) if ks else "-"
        row = {k: "" for k in CSV_FIELDS}
        row.update(
            model=spec["name"], conv_mode=spec["conv"], tx_mode=spec["tx"],
            step=step, phase=phase,
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
            clip=f"{clip:.2f}",
            k=kstr if has_routing else "",
            grid_pen=f"{float(grid_pen):.2f}" if has_span else "")
        for i, gm, gmin, gmax in dc_stats(model):
            row.update({f"dc{i}_gmean": f"{gm:.3f}", f"dc{i}_gmin": f"{gmin:.3f}",
                        f"dc{i}_gmax": f"{gmax:.3f}"})
        for i, p50, p80, p95 in dc_winrate_pcts(model):
            row.update({f"dc{i}_p50": f"{p50:.3f}", f"dc{i}_p80": f"{p80:.3f}",
                        f"dc{i}_p95": f"{p95:.3f}"})
        for i, v, s0, s1, sm, tot in dc_gate_health(model):
            row.update({f"dc{i}_vary": v, f"dc{i}_stuck0": s0,
                        f"dc{i}_stuck1": s1, f"dc{i}_stuckmid": sm})
        for i, m, s, r in dc_gate_ema(model):
            row.update({
                f"dc{i}_mema_min": f"{m[0]:.3f}",
                f"dc{i}_mema_max": f"{m[1]:.3f}",
                f"dc{i}_sema_min": f"{s[0]:.3f}",
                f"dc{i}_sema_p50": f"{s[4]:.3f}",
                f"dc{i}_rema_mean": f"{r[2]:.3f}",
                f"dc{i}_rema_p50": f"{r[4]:.3f}",
            })
        for i, b in enumerate(conv_gate_blocks(model)):
            r = ema_summary(b.gate_range_ema)
            row.update({f"cg{i}_rema_mean": f"{r[2]:.3f}",
                        f"cg{i}_rema_p50": f"{r[4]:.3f}"})
        writer.writerow(row)
        tag = f"[{spec['name']} {step:5d}] {phase:<6}"
        print(f"{tag} pol_ce {pol_ce:6.3f}  wdl_ce {wdl_ce_v:5.3f}"
              f"  mse {mse:5.3f}  corr {corr:+6.3f}", flush=True)
        print(f"{'':2}range {float(range_pen):7.2f}/{n_over:<3d}"
              f"  gact {float(gact_pen):8.2f}"
              f"  dot {dot_peak:6.0f}/{dot_cap:<5.0f}  grad {float(grad_norm):7.2f}"
              f"  k {kstr}  grid {float(grid_pen):6.2f}"
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
        if has_maxnet:
            print_gate_ema(conv_gate_ema(model.max_convs), "MX")
        if has_dcconv:
            print_gate_ema(conv_gate_ema(model.dc_convs), "DV")
        if has_gated:
            print_gate_ema(conv_gate_ema(model.gated_convs), "GC")
        if has_gated_expandy:
            print_gate_ema(conv_gate_ema(model.gated_expandy_convs), "GE")
        if has_sm:
            print_gate_ema(conv_gate_ema(model.sm_convs), "SM")
        if has_dc:
            print_gate_ema(dc_gate_ema(model), "DC")
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
                    help="comma-separated model ids or names to run, drawn from "
                         "the full MODEL_SPECS catalogue (default: RUN_ORDER)")
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

    todo = MODELS
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
