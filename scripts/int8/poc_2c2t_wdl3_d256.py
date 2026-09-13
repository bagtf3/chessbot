"""Train a 2-conv / 2-transformer / policy+WDL model toward explicit INT8.

This POC deliberately uses *two* kinds of scale:

    real weight       = q_weight     * weight_scale[output_channel]
    real activation   = q_activation * activation_scale[tensor]

``q_weight`` is a learned float during the early phase and becomes an int8
weight at QAT/export.  ``q_activation`` is introduced by ActQuant at each
residual-safe boundary.  In normal mode ActQuant is an identity and supplies
range/grid losses; in QAT mode it uses round+clamp in the forward with an STE.

Data comes from the real sparse bootstrap shards: ``(xc0h, sparse_policy, wdl,
source)``.  This is intentionally a small random-shard loader for the POC, not
the production async pool.

    python scripts/int8/poc_2c2t_wdl3_d256.py --device cuda
"""
import argparse
import datetime
import gc
import glob
import gzip
import math
from pathlib import Path
import pickle
import random
import subprocess
import sys
import queue
import threading

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyfastchess


D, HEADS, FF = 256, 8, 1024
PDH, POLICY_HEADS = 256, 4
GATE_D, GATE_H = 2 * D, 3 * D  # SmartGate reads two accumulators, SwiGLU at 768
BOARD, TOKENS = 8, 64
XC0H_K, XC0H_VOCAB, XC0H_DEMB = 6, 15, 18
XC0H_IN = XC0H_K * XC0H_DEMB + XC0H_K + 3  # stem projection input width
XC0H_LEN = XC0H_K * 64 + XC0H_K + 3
POLICY_DIM = 1858
QMAX = 127.0
# Range banger aims here, below the QMAX clamp, so coords settle with headroom and
# never hit the int8 edge.  Without the buffer, QAT's clamp is a one-way ratchet:
# the STE passes gradient through a clamped value, so the task shoves coords past
# 127 for free and only the banger resists (range_pen spikes at cutover).
RANGE_TARGET = 100.0
EPS = 1e-6
ACT_SCALE_FLOOR = 1e-3
FP16_CLIP = 60000.0  # hard fp16 safety bound (< 65504) for the policy dot output
GRID_FRAC = (0.6, 1.0)  # random weightset coverage ramped over the grid phase
SETTLE_STEPS = 1000  # final QAT steps that decay LR to settle onto int values
# Conditional-collection range banger (ported from fp16_harden_finetune): when
# RANGE_ON, each ActQuant checks its peak cheaply and only stashes its coordinate
# into RANGE_ACTS when it actually exceeds int8 range, so in-range activations pay
# nothing (no penalty math, no extra backward graph).  Cleared each training step.
RANGE_ON = False
RANGE_ACTS = []


def positive(raw):
    return raw.square().add(EPS)


def raw_for(value):
    return math.sqrt(max(value - EPS, 0.0))


class _RoundSTE(torch.autograd.Function):
    """Hard round to the int8 grid; straight-through gradient (identity)."""
    @staticmethod
    def forward(ctx, x):
        return x.round().clamp(-QMAX, QMAX)

    @staticmethod
    def backward(ctx, grad):
        return grad


def round_clamp_ste(x):
    return _RoundSTE.apply(x)


class Tee:
    """Mirror stdout/stderr to a line-buffered log file so a bare run always
    leaves a full trace -- including the final traceback if it crashes."""

    def __init__(self, path, stream):
        self.f = open(path, "w", buffering=1, encoding="utf-8")
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()


def start_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(log_dir) / f"poc_run_{stamp}.log"
    sys.stdout = Tee(path, sys.stdout)
    sys.stderr = Tee(path, sys.stderr)
    print(f"[log] mirroring stdout/stderr to {path}", flush=True)


def export_constant(x):
    """Turn a learned scale into an ONNX initializer, not a runtime scale op."""
    return torch.tensor(x.detach().cpu().numpy(), dtype=x.dtype, device=x.device)


class _ActivationQDQ(torch.autograd.Function):
    """QAT identity in PyTorch; explicit ONNX Q/DQ pair during export."""
    @staticmethod
    def forward(ctx, x, scale):
        return round_clamp_ste(x / scale) * scale

    @staticmethod
    def backward(ctx, grad):
        # This path is only selected while torch.onnx.export is tracing.
        return grad, None

    @staticmethod
    def symbolic(g, x, scale):
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
        q = g.op("QuantizeLinear", x, scale, zero)
        return g.op("DequantizeLinear", q, scale, zero)


class _WeightQDQ(torch.autograd.Function):
    """Emit q=round(Q) with scale 1, followed by per-output-channel DQ(s)."""
    @staticmethod
    def forward(ctx, q_coordinate, scale):
        view = (-1,) + (1,) * (q_coordinate.ndim - 1)
        return round_clamp_ste(q_coordinate) * scale.view(view)

    @staticmethod
    def backward(ctx, grad):
        return grad, None

    @staticmethod
    def symbolic(g, q_coordinate, scale):
        one = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int8))
        q = g.op("QuantizeLinear", q_coordinate, one, zero)
        # Q is [out,...], so scale axis 0 is the output channel/row.  Per-axis DQ
        # needs an explicit [out_channels] int8 zero-point vector, and it must be a
        # static initializer: ORT tolerates a computed one but the raw TRT parser
        # rejects it (validateQDQInputs).  scale is a baked constant, so its length
        # is known at trace time.
        n = scale.type().sizes()[0]
        zp_vec = g.op("Constant", value_t=torch.zeros(n, dtype=torch.int8))
        return g.op("DequantizeLinear", q, scale, zp_vec, axis_i=0)


class QWeight(nn.Module):
    """Shared Q/s machinery for a Conv or Linear weight tensor."""

    def __init__(self, shape):
        super().__init__()
        # Keep FP32 master Q parameters for Adam/STE stability.  CUDA autocast
        # still executes the Conv/Linear matmuls in FP16; export hard-casts the
        # converged coordinates to INT8.
        self.Q = nn.Parameter(torch.empty(shape))
        self.sprime = nn.Parameter(torch.empty(shape[0]))
        # Range banger ceiling: below the QMAX clamp for headroom (see RANGE_TARGET).
        self.range_max = RANGE_TARGET
        self.reset_parameters()

    def reset_parameters(self):
        # Spread each channel's peak to RANGE_TARGET so the full int8 range is used
        # from the start (init at 64 left coords maxing at ~64 -> effectively int7,
        # 2x the quant step).  The banger holds them here; 127 stays as headroom.
        w = torch.empty_like(self.Q)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        c = RANGE_TARGET / w.abs().flatten(1).amax(1, keepdim=True)
        with torch.no_grad():
            self.Q.copy_(w * c.view(-1, *([1] * (w.ndim - 1))))
            self.sprime.copy_((1.0 / c.squeeze(1) - EPS).clamp_min(0).sqrt())

    def scale(self):
        return positive(self.sprime)

    def weight(self, qat):
        if qat and torch.onnx.is_in_onnx_export():
            return _WeightQDQ.apply(self.Q, export_constant(self.scale()))
        q = round_clamp_ste(self.Q) if qat else self.Q
        return q * self.scale().view(-1, *([1] * (q.ndim - 1)))


class QConv2d(nn.Module):
    def __init__(self, cin, cout, kernel=3, bias=True):
        super().__init__()
        self.w = QWeight((cout, cin, kernel, kernel))
        self.b = nn.Parameter(torch.zeros(cout)) if bias else None
        self.padding = kernel // 2

    def forward(self, x, qat):
        return F.conv2d(x, self.w.weight(qat), self.b, padding=self.padding)


class QLinear(nn.Module):
    def __init__(self, din, dout, bias=True):
        super().__init__()
        self.w = QWeight((dout, din))
        self.b = nn.Parameter(torch.zeros(dout)) if bias else None

    def forward(self, x, qat):
        return F.linear(x, self.w.weight(qat), self.b)


class ActQuant(nn.Module):
    """A learned per-tensor activation scale at one INT8 handoff.

    The tensor remains real-valued in normal mode.  QAT mode returns the exact
    represented value ``round(x / s_a) * s_a`` (with an STE gradient).
    """

    def __init__(self, scale=0.25, learn=True):
        super().__init__()
        self.sprime = nn.Parameter(torch.tensor(raw_for(scale)), requires_grad=learn)
        self.range_max = RANGE_TARGET

    def scale(self):
        return positive(self.sprime).clamp_min(ACT_SCALE_FLOOR)

    def forward(self, x, qat):
        if qat and torch.onnx.is_in_onnx_export():
            return _ActivationQDQ.apply(x, export_constant(self.scale()))
        s = self.scale()
        # Conditional collection: only stash (coordinate, ceiling) for the range
        # penalty when the coordinate actually breaks this site's range.  The peak
        # check is one cheap reduction; in-range activations build no penalty.
        if RANGE_ON and torch.is_grad_enabled():
            if x.detach().abs().max() > self.range_max * s:
                RANGE_ACTS.append((x / s, self.range_max))
        if qat:
            return round_clamp_ste(x / s) * s
        return x


class L1Norm(nn.Module):
    """Float normalization island followed by a quantized activation boundary."""

    def __init__(self, d, activation_scale=0.25):
        super().__init__()
        # Init scale 1.0, matching prod RMSNorm; a fresh norm should not start
        # 20x hot and pump oversized activations into the next quantized layer.
        self.sprime = nn.Parameter(torch.full((d,), raw_for(1.0)))
        self.b = nn.Parameter(torch.zeros(d))
        self.q = ActQuant(activation_scale)

    def forward(self, x, qat):
        # Clamp mirrors prod RMSNorm: caps the residual stream so it cannot grow
        # past the fp16 range between blocks (inf clamps back to a finite bound).
        xf = x.float().clamp(-250.0, 250.0)
        y = xf / (xf.abs().mean(-1, keepdim=True) + EPS)
        y = y * positive(self.sprime) + self.b
        return self.q(y.to(x.dtype), qat)


class ConvBlock(nn.Module):
    """Mirrors the prod ConvBlock: prenorm, c1(bias)->leaky, c2(no bias)->leaky,
    residual.  L1Norm stands in for prod's RMSNorm; ActQuant marks int8 handoffs.
    """

    def __init__(self):
        super().__init__()
        self.norm = L1Norm(D, activation_scale=0.25)
        self.c1 = QConv2d(D, D)
        self.a1 = ActQuant(0.25)
        self.c2 = QConv2d(D, D, bias=False)
        self.out = ActQuant(0.25)

    def forward(self, x, qat):
        # NCHW -> normalize across channels.  The transpose only changes view.
        h = self.norm(x.permute(0, 2, 3, 1), qat).permute(0, 3, 1, 2)
        h = self.a1(F.leaky_relu(self.c1(h, qat), 0.01), qat)
        h = self.c2(h, qat)
        return self.out(x + F.leaky_relu(h, 0.01), qat)


class Attention(nn.Module):
    """SDPA-equivalent MHA, mirroring prod SdpaMHA q/k/v/o.  Cross-attention:
    query stream and key/value stream are separate arguments (self-attention
    passes the same tensor twice)."""

    def __init__(self, dim=D, heads=HEADS):
        super().__init__()
        self.q = QLinear(dim, dim)
        self.k = QLinear(dim, dim)
        self.v = QLinear(dim, dim)
        self.o = QLinear(dim, dim)
        self.q_out = ActQuant(0.25)
        self.k_out = ActQuant(0.25)
        self.v_out = ActQuant(0.25)
        self.o_out = ActQuant(0.25)
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

    def forward(self, xq, xkv, qat):
        bq, nq, _ = xq.shape
        nk = xkv.shape[1]
        q = self.q_out(self.q(xq, qat), qat).view(
            bq, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_out(self.k(xkv, qat), qat).view(
            bq, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_out(self.v(xkv, qat), qat).view(
            bq, nk, self.heads, self.head_dim).transpose(1, 2)
        # Scores and softmax are the intentional FP32 attention island; autocast
        # would otherwise recast the matmuls to fp16 and overflow.
        with torch.autocast("cuda", enabled=False):
            a = (q.float() @ k.float().transpose(-1, -2)) * (self.head_dim ** -0.5)
            a = a.softmax(dim=-1)
            h = (a @ v.float()).transpose(1, 2).reshape(bq, nq, self.dim)
        return self.o_out(self.o(h.to(xq.dtype), qat), qat)


class TxBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = L1Norm(D, activation_scale=0.25)
        self.attn = Attention(D, HEADS)
        self.after_attn = ActQuant(0.25)
        self.n2 = L1Norm(D, activation_scale=0.25)
        self.ff1 = QLinear(D, FF)
        self.ff_mid = ActQuant(0.25)
        self.ff2 = QLinear(FF, D, bias=False)
        self.out = ActQuant(0.25)

    def forward(self, x, qat):
        n = self.n1(x, qat)
        x = self.after_attn(x + self.attn(n, n, qat), qat)
        h = self.ff_mid(F.gelu(self.ff1(self.n2(x, qat), qat)), qat)
        return self.out(x + self.ff2(h, qat), qat)


class WDL3Head(nn.Module):
    """Independent W/D/L reads from accumulator tokens 0..2, one small readout
    MLP each.  Mirrors prod wdl_h{w,d,l} / wdl_o{w,d,l}."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.ModuleList([QLinear(D, 64) for _ in range(3)])
        self.mid_q = nn.ModuleList([ActQuant(0.25) for _ in range(3)])
        self.out = nn.ModuleList([QLinear(64, 1, bias=False) for _ in range(3)])

    def forward(self, tokens, qat):
        logits = []
        for i in range(3):
            h = self.mid_q[i](F.gelu(self.hidden[i](tokens[:, TOKENS + i, :], qat)), qat)
            logits.append(self.out[i](h, qat))
        return torch.cat(logits, dim=-1)


class Int8Ready2C2TWDL(nn.Module):
    """INT8-ready trunk that mirrors precond-mha-10c6t-d256-wdl3 module-for-module
    (stem, conv preconditioner, additive pos, 8 accumulators, MHA transformer,
    WDL3 head, SwiGLU SmartGate at 2D, MHA from/to policy), so prod weights load
    directly.  It runs 2 convs and 2 transformers instead of 10 and 6, and every
    RMSNorm is an L1Norm island with an int8 handoff instead."""

    def __init__(self, pre_blocks=2, tx_blocks=2, dot_quant=True):
        super().__init__()
        self.dot_quant = dot_quant
        # xc0h stem: categorical piece tokens + rep/cast/stm/hmc planes, then a
        # quantized projection and four constant spatial planes appended after
        # the norm (so it cannot wash out their constant/gradient values).
        self.tok_emb = nn.Embedding(XC0H_VOCAB, XC0H_DEMB)
        self.cast_emb = nn.Embedding(16, 64)
        self.stem_proj = QLinear(XC0H_IN, D - 4)
        self.stem_norm = L1Norm(D - 4, activation_scale=0.25)
        self.stem_out = ActQuant(0.25)
        for name in ("rep", "stm", "hmc", "ones", "checker", "rank", "file"):
            setattr(self, f"{name}_scale", nn.Parameter(torch.tensor(1.0)))
        sqs = torch.arange(64).float()
        ranks_sq, files_sq = sqs // 8, sqs % 8
        self.register_buffer("xc0h_spatial", torch.stack([
            torch.ones(64), ((ranks_sq + files_sq) % 2) * 2 - 1,
            ranks_sq / 7.0 * 2 - 1, files_sq / 7.0 * 2 - 1], dim=-1).unsqueeze(0))

        self.convs = nn.ModuleList([ConvBlock() for _ in range(pre_blocks)])
        self.pos = nn.Parameter(torch.randn(TOKENS, D) * 0.02)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.global_tokens = nn.Parameter(torch.randn(1, 8, D) * 0.02)
        self.tokens_out = ActQuant(0.25)
        self.blocks = nn.ModuleList([TxBlock() for _ in range(tx_blocks)])
        self.trunk = L1Norm(D, activation_scale=0.25)
        self.value_head = WDL3Head()

        self.from_proj = QLinear(D, PDH)
        self.from_proj_q = ActQuant(0.25)
        self.from_norm = L1Norm(PDH, activation_scale=0.25)
        self.from_mha = Attention(PDH, POLICY_HEADS)
        self.from_out = QLinear(PDH, PDH, bias=False)
        self.from_out_q = ActQuant(0.25)
        self.to_proj = QLinear(D, PDH)
        self.to_proj_q = ActQuant(0.25)
        self.to_norm = L1Norm(PDH, activation_scale=0.25)
        self.to_mha = Attention(PDH, POLICY_HEADS)
        self.to_out = QLinear(PDH, PDH, bias=False)
        self.to_out_q = ActQuant(0.25)
        # Condition + quantize the dot inputs so the policy bmm is a plain int8
        # matmul (int32 accumulate, no overflow) like ff1 and the smartgate.
        self.from_dot_norm = L1Norm(PDH, activation_scale=0.25)
        self.from_dot_q = ActQuant(0.25)
        self.to_dot_norm = L1Norm(PDH, activation_scale=0.25)
        self.to_dot_q = ActQuant(0.25)
        self.promo_from = QLinear(PDH, 3, bias=False)
        self.promo_to = QLinear(PDH, 3, bias=False)

        self.gate_w_gate = QLinear(GATE_D, GATE_H)
        self.gate_gate_q = ActQuant(0.25)
        self.gate_w_up = QLinear(GATE_D, GATE_H)
        self.gate_up_q = ActQuant(0.25)
        self.gate_h_q = ActQuant(0.25)
        self.gate_w_down = QLinear(GATE_H, GATE_D, bias=False)
        self.gate_down_q = ActQuant(0.25)
        self.gate_norm = L1Norm(GATE_D, activation_scale=0.25)
        self.gate_out = QLinear(GATE_D, POLICY_DIM, bias=True)
        with torch.no_grad():
            self.gate_w_down.w.Q.zero_()
            self.gate_out.w.Q.zero_()
            self.gate_out.b.fill_(4.0)
        self.last_gate_raw = None
        self.last_dot_raw = None
        self.register_buffer("sl_idx", torch.from_numpy(
            pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0])

    def stem(self, x, qat):
        b = x.shape[0]
        K = XC0H_K
        tok = x[:, :K * 64].long().view(b, K, TOKENS)
        rep = x[:, K * 64:K * 64 + K].float()
        cast = x[:, K * 64 + K].long()
        stm = x[:, K * 64 + K + 1].float()
        hmc = x[:, K * 64 + K + 2].float()
        te = self.tok_emb(tok).permute(0, 2, 1, 3).reshape(b, TOKENS, K * XC0H_DEMB)
        rep_planes = (rep * 2.0 - 1.0).unsqueeze(1).expand(b, TOKENS, K) * self.rep_scale
        cast_plane = self.cast_emb(cast).unsqueeze(-1)
        stm_plane = (stm * 2.0 - 1.0).view(b, 1, 1).expand(b, TOKENS, 1) * self.stm_scale
        hmc_plane = (hmc / 99.0 * 2.0 - 1.0).view(b, 1, 1).expand(b, TOKENS, 1) * self.hmc_scale
        fused = torch.cat([te, rep_planes, cast_plane, stm_plane, hmc_plane], dim=-1)
        proj = self.stem_norm(self.stem_proj(fused, qat), qat)
        scales = torch.stack([self.ones_scale, self.checker_scale,
                              self.rank_scale, self.file_scale])
        spatial = self.xc0h_spatial.to(proj.dtype).expand(b, -1, -1) * scales
        x = self.stem_out(torch.cat([proj, spatial], dim=-1), qat)
        return x.reshape(b, BOARD, BOARD, D).permute(0, 3, 1, 2)

    def forward(self, x, qat=False):
        if x.shape[1] != XC0H_LEN:
            raise ValueError(f"expected xc0h [B,{XC0H_LEN}], got {tuple(x.shape)}")
        b = x.shape[0]
        x = self.stem(x, qat)
        for block in self.convs:
            x = block(x, qat)
        board = x.permute(0, 2, 3, 1).reshape(b, TOKENS, D)
        x = board + self.pos_scale * self.pos.unsqueeze(0)
        x = torch.cat([x, self.global_tokens.expand(b, -1, -1)], dim=1)
        x = self.tokens_out(x, qat)
        for block in self.blocks:
            x = block(x, qat)
        x = self.trunk(x, qat)
        wdl = self.value_head(x, qat)

        from_set = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:71, :]], dim=1)
        to_set = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 71:72, :]], dim=1)
        f_proj = from_set + self.from_proj_q(F.gelu(self.from_proj(from_set, qat)), qat)
        fn = self.from_norm(f_proj, qat)
        f_base = f_proj[:, :64, :] + self.from_mha(fn[:, :64, :], fn, qat)
        f_vec = f_base + self.from_out_q(self.from_out(f_base, qat), qat)
        t_proj = to_set + self.to_proj_q(F.gelu(self.to_proj(to_set, qat)), qat)
        tn = self.to_norm(t_proj, qat)
        t_base = t_proj[:, :64, :] + self.to_mha(tn[:, :64, :], tn, qat)
        t_vec = t_base + self.to_out_q(self.to_out(t_base, qat), qat)
        # Normalize + quantize both dot inputs, then let the bmm be a plain int8
        # matmul like every other one: int8 x int8 accumulates in int32, which
        # cannot overflow for a 256-wide dot (127*127*256 fits int32 easily) -- no
        # fp16 island, no clamps.  The L1Norm conditions each vector so the int8
        # grid captures it tightly; the ActQuant emits the Q/DQ that make TRT pick
        # an int8 GEMM.
        if self.dot_quant:
            fq = self.from_dot_q(self.from_dot_norm(f_vec, qat), qat)
            tq = self.to_dot_q(self.to_dot_norm(t_vec, qat), qat)
        else:
            # Legacy INT8 path: retain the quantized trunk/head handoffs, but
            # feed the BMM directly from those vectors (no dedicated dot
            # normalization/quantization stage).
            fq, tq = f_vec, t_vec
        # Stay in fp16; hard-clip the dot at the fp16 boundary so an overflow is
        # caught (inf clamps back to finite, grad 0 there) instead of NaN-ing.  The
        # int8 engine accumulates in int32 and never reaches this; it is a pure
        # fp16-training guard.  The soft dot banger keeps it far below this anyway.
        dots_raw = torch.bmm(fq, tq.transpose(1, 2)).clamp(-FP16_CLIP, FP16_CLIP)
        self.last_dot_raw = dots_raw
        dots_full = dots_raw * (PDH ** -0.5)
        dots = dots_full.reshape(b, TOKENS * TOKENS)
        dots_sub = dots_full[:, 48:56, 56:64]
        pf = self.promo_from(f_vec[:, 48:56, :], qat)
        pt = self.promo_to(t_vec[:, 56:64, :], qat)
        promo = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]
                 ).permute(0, 3, 2, 1).reshape(b, 192)
        policy = torch.cat([dots, promo], dim=-1)[:, self.sl_idx]

        gi = torch.cat([x[:, 67, :], x[:, 68, :]], dim=-1)
        h = self.gate_h_q(self.gate_gate_q(F.silu(self.gate_w_gate(gi, qat)), qat)
                          * self.gate_up_q(self.gate_w_up(gi, qat), qat), qat)
        g = gi + self.gate_down_q(self.gate_w_down(h, qat), qat)
        g = g.clamp(-250.0, 250.0)
        gate_raw = self.gate_out(self.gate_norm(g, qat), qat)
        self.last_gate_raw = gate_raw
        # Clamp before logsigmoid so a runaway gate logit can't reach -inf and
        # nan the whole policy; the range banger still pulls it toward [-127,127].
        return policy + F.logsigmoid(gate_raw.float().clamp(-250.0, 250.0)), wdl


def range_penalty(model):
    """Linear range banger, conditional collection, live from step 0.

    Each site has its own ceiling (`range_max`): int8 (127) for most, tightened
    toward an fp16-safe bound for the policy-dot key layers.  Only sites over their
    own ceiling contribute a linear sum-relu push; in-range sites cost nothing.
    Sum (not mean) so outliers are not diluted -- pair it with a small coefficient.
    The raw gate logits feed logsigmoid, so they ride along here (gate_out has no
    ActQuant of its own).  Returns (penalty, n_sites_over) for reporting.
    """
    total = 0.0
    over = 0
    for m in model.modules():
        if isinstance(m, QWeight):
            if m.Q.detach().abs().max() > m.range_max:
                total = total + F.relu(m.Q.abs() - m.range_max).sum()
                over += 1
    for coord, ceil in RANGE_ACTS:
        total = total + F.relu(coord.abs() - ceil).sum()
    over += len(RANGE_ACTS)
    raw = getattr(model, "last_gate_raw", None)
    if raw is not None and raw.detach().abs().max() > QMAX:
        total = total + F.relu(raw.abs() - QMAX).sum()
        over += 1
    return total, over


def grid_penalty(model, frac, sin2_alpha=0.0):
    """Nearest-int grid penalty on a random fraction of weightsets each step.

    No threshold: every chosen set is pulled fully onto the integer grid.  `frac`
    of the weightsets are picked at random per step (ramped 0.6 -> 1.0 over the
    grid phase), so coverage is stochastic early and complete by the end -- cheap,
    with no mean-based gate hiding stragglers.

    Per element the penalty blends two scale-matched shapes:
      sin2_alpha * (0.25 * sin^2(pi*q))  +  (1 - sin2_alpha) * (q - round(q))^2
    sin^2 is soft at the half-integer (no cusp, gradient -> 0 there) so it snaps
    near-int weights while letting ~0.5 stragglers flutter; residual^2 is the firm
    final snap.  0.25 peak-matches sin^2 to residual^2 so the weight means the same.
    When sin2_alpha == 0 the sin^2 term is not computed.
    Returns (penalty, n_sets, mean_offset_over_sample)."""
    qweights = [m for m in model.modules() if isinstance(m, QWeight)]
    k = max(1, round(frac * len(qweights)))
    chosen = random.sample(qweights, min(k, len(qweights)))
    total = 0.0
    offsets = []
    for m in chosen:
        q = m.Q
        target = q.detach().round().clamp(-QMAX, QMAX)
        offsets.append(float((q.detach() - target).abs().mean()))
        res2 = (q - target).square()
        if sin2_alpha > 0.0:
            sin2 = 0.25 * torch.sin(math.pi * q).square()
            term = sin2_alpha * sin2 + (1.0 - sin2_alpha) * res2
        else:
            term = res2
        total = total + term.sum()
    mean_off = sum(offsets) / max(len(offsets), 1)
    return total, len(chosen), mean_off


def grid_mean_offset(model):
    """Cheap no-grad mean |Q - round(Q)| over all weightsets, for logging when the
    grid penalty itself is off (e.g. during QAT)."""
    with torch.no_grad():
        offs = [float((m.Q - m.Q.round().clamp(-QMAX, QMAX)).abs().mean())
                for m in model.modules() if isinstance(m, QWeight)]
    return sum(offs) / max(len(offs), 1)


def load_qweight(qw, W, blend):
    """Represent a dense prod weight as Q * per-output-channel s, exactly, then
    nudge the integer coordinate by a `blend` fraction of matched-scale noise."""
    with torch.no_grad():
        Wf = W.float()
        amax = Wf.abs().flatten(1).amax(1).clamp_min(1e-12)
        c = RANGE_TARGET / amax  # spread stolen weights across the full int8 range
        Q = Wf * c.view(-1, *([1] * (Wf.ndim - 1)))
        if blend > 0.0:
            Q = (1.0 - blend) * Q + blend * torch.randn_like(Q) * Q.std()
        qw.Q.copy_(Q.to(qw.Q.dtype))
        qw.sprime.copy_((1.0 / c - EPS).clamp_min(0.0).sqrt().to(qw.sprime.dtype))


def load_linear(qlin, sd, key, blend):
    load_qweight(qlin.w, sd[f"{key}.weight"], blend)
    if qlin.b is not None and f"{key}.bias" in sd:
        with torch.no_grad():
            qlin.b.copy_(sd[f"{key}.bias"].float().to(qlin.b.dtype))


def load_l1norm(l1, rms_scale):
    """Seed an L1Norm from a prod RMSNorm scale.  For gaussian activations
    mean|x| ~ 0.7979 * rms, so a matched L1 output uses 0.7979 * the rms scale."""
    with torch.no_grad():
        tgt = (0.7979 * rms_scale.float()).clamp_min(0.0)
        l1.sprime.copy_((tgt - EPS).clamp_min(0.0).sqrt().to(l1.sprime.dtype))
        l1.b.zero_()


def load_param(dst, src):
    with torch.no_grad():
        dst.copy_(src.to(dst.dtype).reshape(dst.shape))


def init_from_prod(model, ckpt_path, blend=0.1, conv_src=1, tx_src=0):
    """Warm-start from a precond-mha-10c6t-d256-wdl3 checkpoint: steal the stem,
    one conv block, pos, one transformer, and the whole value/policy/gate heads;
    leave the extra conv and transformer fresh.  L1Norm islands are seeded from
    the prod RMS scales.  Structural, not functional: this is a better basin, not
    a working model."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]

    load_param(model.tok_emb.weight, sd["xc0h_tok_emb.weight"])
    load_param(model.cast_emb.weight, sd["xc0h_cast_emb.weight"])
    load_linear(model.stem_proj, sd, "xc0h_proj", blend)
    load_l1norm(model.stem_norm, sd["xc0h_proj_ln.scale"])
    for name in ("rep", "stm", "hmc", "ones", "checker", "rank", "file"):
        load_param(getattr(model, f"{name}_scale"), sd[f"xc0h_{name}_scale"])
    load_param(model.pos, sd["pos.weight"])
    load_param(model.pos_scale, sd["pos_scale"])
    load_param(model.global_tokens, sd["global_tokens"])

    c = f"pre.{conv_src}"
    load_l1norm(model.convs[0].norm, sd[f"{c}.ln.w"])
    load_linear(model.convs[0].c1, sd, f"{c}.c1", blend)
    load_linear(model.convs[0].c2, sd, f"{c}.c2", blend)

    t = f"blocks.{tx_src}"
    load_l1norm(model.blocks[0].n1, sd[f"{t}.ln1.scale"])
    for w in ("q", "k", "v", "o"):
        load_linear(getattr(model.blocks[0].attn, w), sd, f"{t}.attn.{w}", blend)
    load_l1norm(model.blocks[0].n2, sd[f"{t}.ln2.scale"])
    load_linear(model.blocks[0].ff1, sd, f"{t}.ff1", blend)
    load_linear(model.blocks[0].ff2, sd, f"{t}.ff2", blend)

    load_l1norm(model.trunk, sd["trunk_ln.scale"])
    for i, tag in enumerate(("w", "d", "l")):
        load_linear(model.value_head.hidden[i], sd, f"wdl_h{tag}", blend)
        load_linear(model.value_head.out[i], sd, f"wdl_o{tag}", blend)

    for side in ("from", "to"):
        load_linear(getattr(model, f"{side}_proj"), sd, f"{side}_proj", blend)
        load_l1norm(getattr(model, f"{side}_norm"), sd[f"{side}_ln.scale"])
        for w in ("q", "k", "v", "o"):
            load_linear(getattr(getattr(model, f"{side}_mha"), w),
                        sd, f"{side}_mha.{w}", blend)
        load_linear(getattr(model, f"{side}_out"), sd, f"{side}_out", blend)
    load_linear(model.promo_from, sd, "promo_from", blend)
    load_linear(model.promo_to, sd, "promo_to", blend)

    load_linear(model.gate_w_gate, sd, "gate_w_gate", blend)
    load_linear(model.gate_w_up, sd, "gate_w_up", blend)
    load_linear(model.gate_w_down, sd, "gate_w_down", blend)
    load_l1norm(model.gate_norm, sd["gate_norm.scale"])
    load_linear(model.gate_out, sd, "gate_out", blend)
    print(f"[init] warm-started from {Path(ckpt_path).name}"
          f" (conv pre.{conv_src}, tx blocks.{tx_src}, rand_blend={blend})", flush=True)


def densify_policy(sparse):
    idx, values = sparse
    out = np.zeros(POLICY_DIM, dtype=np.float32)
    out[np.asarray(idx, dtype=np.int32)] = np.asarray(values, dtype=np.float32)
    return out


class SparseShardBatches:
    """Two-shard background prefetch pool for the real-data POC.

    The worker handles gzip, pickle, policy densification, and NumPy stacking
    while CUDA trains on the preceding shard.  This is intentionally much
    smaller than the production async pool, but removes synchronous disk/Python
    stalls from the hot path.
    """
    def __init__(self, shard_dir, seed=0):
        self.files = sorted(glob.glob(str(Path(shard_dir) / "*.pkl.gz")))
        if not self.files:
            raise FileNotFoundError(f"no .pkl.gz sparse shards in {shard_dir}")
        self.rng = random.Random(seed)
        self.ready = queue.Queue(maxsize=2)
        self.current = None
        self.offset = 0
        self.worker = threading.Thread(target=self._loader, daemon=True,
                                       name="int8-poc-shard-prefetch")
        self.worker.start()

    def _loader(self):
        while True:
            path = self.rng.choice(self.files)
            try:
                with gzip.open(path, "rb") as fh:
                    records = pickle.load(fh)
                self.rng.shuffle(records)
                chunk = (
                    np.stack([r[0] for r in records]).astype(np.int64),
                    np.stack([densify_policy(r[1]) for r in records]),
                    np.stack([r[2] for r in records]).astype(np.float32),
                )
                self.ready.put(chunk)
            except Exception as exc:
                print(f"[poc shard] skipped {Path(path).name}: {exc}", flush=True)

    def batch(self, n, device):
        pieces, left = [], n
        while left:
            if self.current is None or self.offset == self.current[0].shape[0]:
                self.current = self.ready.get()
                self.offset = 0
            take = min(left, self.current[0].shape[0] - self.offset)
            pieces.append(tuple(a[self.offset:self.offset + take] for a in self.current))
            self.offset += take
            left -= take
        enc_np = np.concatenate([p[0] for p in pieces])
        pol_np = np.concatenate([p[1] for p in pieces])
        wdl_np = np.concatenate([p[2] for p in pieces])
        enc = torch.as_tensor(enc_np, dtype=torch.long, device=device)
        pol = torch.as_tensor(pol_np, dtype=torch.float32, device=device)
        wdl = torch.as_tensor(wdl_np, dtype=torch.float32, device=device)
        return enc, pol, wdl


class QATExportWrapper(nn.Module):
    """Bake a fixed qat flag into the exported inference graph: qat=True emits the
    explicit Q/DQ (int8), qat=False emits the plain float graph (no Q/DQ)."""

    def __init__(self, model, qat=True):
        super().__init__()
        self.model = model
        self.qat = qat

    def forward(self, board):
        return self.model(board, qat=self.qat)


def export_onnx(wrapper, dummy, onnx_path):
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(onnx_path), opset_version=18,
            input_names=["xc0h"], output_names=["policy_logits", "wdl_logits"],
            dynamic_axes={"xc0h": {0: "batch"}, "policy_logits": {0: "batch"},
                          "wdl_logits": {0: "batch"}},
            do_constant_folding=False,
        )


def build_trt_session(onnx_path, out_dir, batch, int8):
    import tensorrt  # noqa: F401: registers TRT DLLs before ORT loads its EP
    import onnxruntime as ort
    opts = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(out_dir),
        "trt_int8_enable": int8,
        "trt_fp16_enable": False,
        "trt_profile_min_shapes": f"xc0h:1x{XC0H_LEN}",
        "trt_profile_opt_shapes": f"xc0h:{batch}x{XC0H_LEN}",
        "trt_profile_max_shapes": f"xc0h:{batch}x{XC0H_LEN}",
    }
    session = ort.InferenceSession(
        str(onnx_path), providers=[("TensorrtExecutionProvider", opts)])
    if session.get_providers()[0] != "TensorrtExecutionProvider":
        raise RuntimeError(f"TRT EP was not selected: {session.get_providers()}")
    return session


def speed_test(model, out_dir, batch, probe, iters=200, warmup=30):
    """Benchmark plain FP32 (no Q/DQ) vs INT8+FP32 (fp16 off) TRT engines at a
    fixed batch.  Weights are irrelevant to timing, so this runs on whatever
    model is passed (warm-started, no training needed)."""
    import time
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = out_dir / "poc_2c2t_wdl3_fp32.onnx"
    int8_onnx = out_dir / "poc_2c2t_wdl3_int8_qdq.onnx"
    dummy = probe[:1]
    export_onnx(QATExportWrapper(model, qat=False).eval(), dummy, fp32_onnx)
    export_onnx(QATExportWrapper(model, qat=True).eval(), dummy, int8_onnx)

    x = probe
    if x.shape[0] < batch:
        x = x.repeat((batch + x.shape[0] - 1) // x.shape[0], 1)
    x_np = x[:batch].cpu().numpy()

    def run(session):
        for _ in range(warmup):
            session.run(None, {"xc0h": x_np})
        t = time.perf_counter()
        for _ in range(iters):
            session.run(None, {"xc0h": x_np})
        return (time.perf_counter() - t) / iters

    fp32_ms = run(build_trt_session(fp32_onnx, out_dir, batch, int8=False)) * 1e3
    int8_ms = run(build_trt_session(int8_onnx, out_dir, batch, int8=True)) * 1e3
    print(f"[speed] batch={batch}  fp32={fp32_ms:.3f} ms/infer"
          f" ({batch / fp32_ms * 1e3:.0f} pos/s)", flush=True)
    print(f"[speed] batch={batch}  int8+fp32={int8_ms:.3f} ms/infer"
          f" ({batch / int8_ms * 1e3:.0f} pos/s)", flush=True)
    print(f"[speed] int8/fp32 speedup = {fp32_ms / int8_ms:.2f}x", flush=True)


def export_and_try_trt(model, device, out_dir, batch, probe, run_name="poc_2c2t_wdl3"):
    """Freeze-QAT export, strict explicit-Q/DQ check, then TRT INT8 build."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{run_name}_qat.pt"
    onnx_path = out_dir / "poc_2c2t_wdl3_int8_qdq.onnx"
    torch.save({"model": model.state_dict(), "qat": True}, checkpoint)

    wrapper = QATExportWrapper(model).eval()
    dummy = probe[:1]
    print(f"[export] checkpoint: {checkpoint}", flush=True)
    print(f"[export] explicit-Q/DQ ONNX graph: {onnx_path}", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(onnx_path), opset_version=18,
            input_names=["xc0h"], output_names=["policy_logits", "wdl_logits"],
            dynamic_axes={"xc0h": {0: "batch"}, "policy_logits": {0: "batch"},
                          "wdl_logits": {0: "batch"}},
            do_constant_folding=False,
        )

    import onnx
    graph = onnx.load(str(onnx_path)).graph
    qdq = {op: sum(node.op_type == op for node in graph.node)
           for op in ("QuantizeLinear", "DequantizeLinear")}
    if not qdq["QuantizeLinear"] or qdq["QuantizeLinear"] != qdq["DequantizeLinear"]:
        raise RuntimeError(f"explicit-Q/DQ export failed: {qdq}")
    print(f"[onnx] Q nodes={qdq['QuantizeLinear']}  DQ nodes={qdq['DequantizeLinear']}", flush=True)

    # Compute the reference while PyTorch is still alive, then tear it down
    # before TensorRT asks CUDA for its builder/workspace allocations.
    probe8 = probe[:min(batch, 8)]
    with torch.no_grad():
        expected_pol, expected_wdl = wrapper(probe8)
    expected_pol = expected_pol.float().cpu()
    expected_wdl = expected_wdl.float().cpu()
    probe8_np = probe8.detach().cpu().numpy()
    model.cpu()
    del wrapper, probe8
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[export] released PyTorch model/CUDA cache before TRT build", flush=True)

    # int8 + fp32 hybrid: int8 comes from the Q/DQ, everything else is fp32.
    # fp16 is left OFF on purpose -- the 256-wide policy dot product overflows
    # fp16 (int8-range inputs give ~127*127*256/16 >> 65504), which drove MAE to
    # inf.  trt_detailed_build_log makes ORT build the engine at DETAILED profiling
    # verbosity so the inspector can read each layer's precision afterward.  ORT
    # partitions unsupported ops (int64 embedding gather, etc.) to CPU; a direct
    # monolithic TRT build of the whole graph segfaults, so we let ORT build.
    import tensorrt  # noqa: F401: registers TRT DLLs before ORT loads its EP
    import onnxruntime as ort
    for stale in out_dir.glob("*.engine"):
        stale.unlink()
    providers = [("TensorrtExecutionProvider", {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(out_dir),
        "trt_int8_enable": True,
        "trt_fp16_enable": False,
        "trt_detailed_build_log": True,
        "trt_profile_min_shapes": f"xc0h:1x{XC0H_LEN}",
        "trt_profile_opt_shapes": f"xc0h:{batch}x{XC0H_LEN}",
        "trt_profile_max_shapes": f"xc0h:{batch}x{XC0H_LEN}",
    })]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    if session.get_providers()[0] != "TensorrtExecutionProvider":
        raise RuntimeError(f"TRT EP was not selected: {session.get_providers()}")
    actual_pol, actual_wdl = session.run(["policy_logits", "wdl_logits"],
                                         {"xc0h": probe8_np})
    mae = ((expected_pol - torch.from_numpy(actual_pol)).abs().mean()
           + (expected_wdl - torch.from_numpy(actual_wdl)).abs().mean()).item() / 2
    if not math.isfinite(mae):
        raise RuntimeError(f"TRT output non-finite (MAE={mae}); int8+fp32 run failed")
    print(f"[trt] ORT int8+fp32 run OK  QAT-vs-TRT MAE={mae:.6f}", flush=True)
    confirm_int8_fp32(out_dir, onnx_path, batch)


def confirm_int8_fp32(out_dir, onnx_path, batch):
    """Confirm the engine is INT8 + FP32 with no FP16.

    Primary proof: ORT tags its engine-cache filename with the enabled build
    precisions, so int8 present + fp16 absent is a hard read of the build config;
    the finite MAE above is the behavioral proof (fp16 in the policy dot product
    overflowed to inf).  We also try a literal per-layer precision dump in a clean
    subprocess (trt_confirm.py) -- but a whole-graph DETAILED build segfaults TRT
    on the int64 embedding that ORT offloads to CPU, so that dump is best-effort.
    """
    out_dir = Path(out_dir)
    engines = sorted(out_dir.glob("*.engine"))
    if not engines:
        raise RuntimeError("TRT provider ran but did not write a cached engine")
    name = max(engines, key=lambda p: p.stat().st_mtime).name.lower()
    has_int8, has_fp16 = "int8" in name, "fp16" in name
    print(f"[confirm] ORT engine precision tag: int8={has_int8} fp16={has_fp16}"
          f"  ({name})", flush=True)
    if not has_int8 or has_fp16:
        raise RuntimeError(f"engine is not int8-without-fp16 as required: {name}")

    helper = Path(__file__).with_name("trt_confirm.py")
    result = subprocess.run(
        [sys.executable, str(helper), "--onnx", str(onnx_path),
         "--out-engine", str(out_dir / "poc_2c2t_wdl3_int8_fp32.engine"),
         "--out-json", str(out_dir / "poc_2c2t_wdl3_int8_fp32_layers.json"),
         "--min-shape", f"1x{XC0H_LEN}", "--opt-shape", f"{batch}x{XC0H_LEN}"],
        capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stdout.write(result.stderr)
    if result.returncode == 0:
        print("[confirm] HARD CONFIRMED int8+fp32 via per-layer precision dump", flush=True)
    else:
        print(f"[confirm] per-layer dump unavailable (rc={result.returncode}); "
              f"confirmed int8+fp32 by engine tag + finite MAE instead", flush=True)


def ramp(step, start, length):
    if length <= 0:
        return 1.0 if step >= start else 0.0
    return min(1.0, max(0.0, (step - start) / length))


def qat_start_step(args):
    return args.float_steps + args.grid_blend_steps + args.grid_steps


def schedule(step, args):
    """Four-phase schedule: float, grid blend, main grid, then hard QAT.

    Range bangers run in every phase.  The grid penalty is enabled only from
    the end of the float phase through the end of the main-grid phase.  At the
    QAT boundary represented-value rounding turns on hard.

    LR stays at full through the QAT body: under rounding a sub-integer nudge
    lands on the same int, so a reduced LR would freeze the model (no motion
    across integer boundaries).  Only the last SETTLE_STEPS decay the LR to
    qat_lr_mult, to settle onto the final int values cleanly.

    Returns (qat, lr_mult, phase).
    """
    qat_start = qat_start_step(args)
    if step < qat_start:
        return False, 1.0, "train"
    total = qat_start + args.qat_steps
    settle_start = total - SETTLE_STEPS
    if step < settle_start:
        return True, 1.0, "qat"
    lr_mult = 1.0 - (1.0 - args.qat_lr_mult) * ramp(step, settle_start, SETTLE_STEPS)
    return True, lr_mult, "settle"


def main():
    global RANGE_ON
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=0.005)
    ap.add_argument("--beta1-span", type=float, default=10.0,
                    help="AdamW beta1 as an EMA span; beta1 = 1 - 1/span")
    ap.add_argument("--beta2-span", type=float, default=500.0,
                    help="AdamW beta2 as an EMA span; beta2 = 1 - 1/span")
    ap.add_argument("--value-weight", type=float, default=2.0,
                    help="relative WDL cross-entropy weight; production warmup uses policy + 2 * WDL")
    ap.add_argument("--qat-lr-mult", type=float, default=0.1,
                    help="LR multiplier reached at fully hard QAT")
    ap.add_argument("--grad-clip-start", type=float, default=1.0)
    ap.add_argument("--grad-clip-end", type=float, default=3.0)
    ap.add_argument("--grad-clip-warmup", type=int, default=500,
                    help="steps to ramp clip start->end; 0 uses --float-steps")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable fp16 autocast (debug only; ~4.5x slower)")
    ap.add_argument("--float-steps", type=int, default=3_000,
                    help="float learning steps; range banger remains active")
    ap.add_argument("--grid-blend-steps", type=int, default=2_000,
                    help="steps blending sin^2 grid loss toward residual^2")
    ap.add_argument("--grid-steps", type=int, default=2_000,
                    help="steps of full residual^2 grid penalty before hard QAT")
    ap.add_argument("--qat-steps", type=int, default=5_000,
                    help="hard-QAT steps with range bangers and no grid penalty")
    
    ap.add_argument("--bounds-weight", type=float, default=1e-6,
                    help="range banger coefficient (conditional sum-relu, from step 0)")
    ap.add_argument("--grid-weight", type=float, default=5e-6,
                    help="grid penalty coefficient (conditional sum-squared per weightset)")
    
    ap.add_argument("--grid-warmup-scale", type=float, default=0.1,
                    help="grid weight multiplier at the start of grid blending")
    ap.add_argument("--dot-cap-start", type=float, default=10240.0,
                    help="soft ceiling on |dots_raw| at step 0 (logits = cap/16)")
    ap.add_argument("--dot-cap-end", type=float, default=1024.0,
                    help="soft ceiling on |dots_raw| reached by qat_start")
    ap.add_argument("--dot-weight", type=float, default=1e-6,
                    help="coefficient for the dot-magnitude banger (sum-relu)")
    ap.add_argument("--run-name", default="poc_2c2t_wdl3_int8dot",
                    help="checkpoint/log basename; keep distinct so old runs are not overwritten")
    ap.add_argument("--init-from",
                    default=r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\20m_wdl3_pretrain\20m_wdl3_pretrain_model_player.pt",
                    help="warm-start from a precond-mha-10c6t-d256-wdl3 checkpoint; \"\" disables")
    ap.add_argument("--init-rand-blend", type=float, default=0.1,
                    help="fraction of matched-scale noise blended into stolen weights")
    ap.add_argument("--init-conv-src", type=int, default=1,
                    help="prod pre.<i> conv block to steal into convs[0]")
    ap.add_argument("--init-tx-src", type=int, default=0,
                    help="prod blocks.<i> transformer to steal into blocks[0]")
    ap.add_argument("--shard-dir", default=r"C:\Users\Bryan\Data\chessbot_data\training_data\pretrain_shards")
    ap.add_argument("--val-batch", type=int, default=256)
    ap.add_argument("--speed-test", action="store_true",
                    help="benchmark fp32 vs int8+fp32 TRT engines and exit (no training)")
    ap.add_argument("--export-only", default=None,
                    help="load a completed POC checkpoint and run only ONNX/TRT export")
    ap.add_argument("--export-dir", default=r"C:\Users\Bryan\Data\chessbot\_data\models\int8_poc_2c2t",
                    help="write QAT checkpoint/ONNX and try ORT TensorRT here; empty disables")
    ap.add_argument("--log-dir", default=None,
                    help="directory for the timestamped run log; defaults to export-dir")
    args = ap.parse_args()

    start_logging(args.log_dir or args.export_dir or ".")
    torch.manual_seed(0)
    train_data = SparseShardBatches(args.shard_dir, seed=0)
    val_data = SparseShardBatches(args.shard_dir, seed=1)
    val_enc, val_pol, val_wdl = val_data.batch(args.val_batch, args.device)
    model = Int8Ready2C2TWDL().to(args.device)
    if args.init_from:
        init_from_prod(model, args.init_from, blend=args.init_rand_blend,
                       conv_src=args.init_conv_src, tx_src=args.init_tx_src)
        model.to(args.device)
    if args.speed_test:
        model.eval()
        speed_test(model, args.export_dir, args.batch, val_enc)
        return
    if args.export_only:
        state = torch.load(args.export_only, map_location=args.device)
        model.load_state_dict(state["model"])
        model.eval()
        export_and_try_trt(model, args.device, args.export_dir, args.batch, val_enc)
        return
    betas = (1.0 - 1.0 / args.beta1_span, 1.0 - 1.0 / args.beta2_span)
    # Decay only ndim>=2 params (weight matrices / conv kernels).  Per-channel
    # scales (sprime), biases, norm gains and scalars are ndim<=1 and left
    # WD-free: decaying a per-channel scale drives it to the EPS floor and kills
    # the channel (12.7% dead when they were decayed).
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=betas)
    print(f"[opt] WD={args.weight_decay} on {len(decay)} ndim>=2 tensors;"
          f" {len(no_decay)} 1-D tensors WD-free", flush=True)
    amp = not args.no_amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    qat_start = qat_start_step(args)
    total_steps = qat_start + args.qat_steps
    print(f"[plan] amp={amp} total_steps={total_steps}"
          f" phases=float:{args.float_steps} grid_blend:{args.grid_blend_steps}"
          f" grid:{args.grid_steps} qat:{args.qat_steps}"
          f" qat_start={qat_start} (hard cutover)", flush=True)
    print(f"[plan] AdamW lr={args.lr} wd={args.weight_decay}"
          f" betas=({betas[0]:.4f},{betas[1]:.4f})"
          f" clip={args.grad_clip_start}->{args.grad_clip_end}"
          f" range_w={args.bounds_weight} grid_w={args.grid_weight}"
          f" grid_frac={GRID_FRAC[0]}->{GRID_FRAC[1]}"
          f"  policy dot = int8 (norm+quant, int32 accum)", flush=True)
    print("[step ...] metrics: validation quality (qat outputs once past cutover)", flush=True)
    print("[step ...] bangers: range + grid penalty state", flush=True)
    print("[step ...] dot: int8 policy-dot magnitude, its cap, and int8 gap (pol_KL)", flush=True)
    print("[step ...] grads: per-parameter L2 norm distribution before clipping", flush=True)

    RANGE_ON = True
    clip_warmup = args.grad_clip_warmup or args.float_steps
    ckpt = Path(args.export_dir) / f"{args.run_name}_qat.pt"
    pre_qat_ckpt = Path(args.export_dir) / f"{args.run_name}_pre_qat.pt"
    Path(args.export_dir).mkdir(parents=True, exist_ok=True)
    for step in range(total_steps):
        if step == qat_start:
            torch.save({"model": model.state_dict(), "step": step}, pre_qat_ckpt)
            print(f"[ckpt] saved pre-QAT model: {pre_qat_ckpt}", flush=True)
        qat, lr_mult, phase = schedule(step, args)
        # Grid runs only up to qat_start, and its threshold reaches its min there,
        # so weights are on-grid BEFORE QAT.  During QAT the round STE enforces the
        # grid itself -- running grid pen too would be double gradients on the same
        # offset, so it is off (not even computed; offset still tracked for the log).
        grid_on = args.float_steps <= step < qat_start
        # Coverage ramps 0.6 -> 1.0 across the whole grid phase (float_steps..qat).
        grid_frac = GRID_FRAC[0] + (GRID_FRAC[1] - GRID_FRAC[0]) * \
            ramp(step, args.float_steps, qat_start - args.float_steps)
        if step < args.float_steps + args.grid_blend_steps:
            # Grid blend: weight ramps up while sin^2 transitions to residual^2.
            frac = ramp(step, args.float_steps, args.grid_blend_steps)
            grid_scale = args.grid_warmup_scale + (1.0 - args.grid_warmup_scale) * frac
            grid_alpha = 1.0 - frac
        else:
            grid_scale, grid_alpha = 1.0, 0.0
        dot_cap = args.dot_cap_start + (args.dot_cap_end - args.dot_cap_start) * \
            ramp(step, 0, qat_start)
        clip = args.grad_clip_start + (args.grad_clip_end - args.grad_clip_start) * \
            ramp(step, 0, clip_warmup)
        for group in opt.param_groups:
            group["lr"] = args.lr * lr_mult
        x, policy_t, wdl_t = train_data.batch(args.batch, args.device)
        report = step % 250 == 0 or step == total_steps - 1
        RANGE_ACTS.clear()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            policy, wdl = model(x, qat=qat)
            policy_ce = F.cross_entropy(policy, policy_t)
            wdl_ce = F.cross_entropy(wdl, wdl_t)
            loss = policy_ce + args.value_weight * wdl_ce
            range_pen, n_over = range_penalty(model)
            loss = loss + args.bounds_weight * range_pen
            dot_pen = F.relu(model.last_dot_raw.abs() - dot_cap).sum()
            loss = loss + args.dot_weight * dot_pen
            dot_peak = float(model.last_dot_raw.detach().abs().max())
            grid = 0.0
            n_grid = 0
            grid_off = 0.0
            if grid_on:
                grid, n_grid, grid_off = grid_penalty(model, grid_frac, grid_alpha)
                loss = loss + args.grid_weight * grid_scale * grid
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step}; qat={qat} "
                f"policy_finite={bool(torch.isfinite(policy).all())} "
                f"wdl_finite={bool(torch.isfinite(wdl).all())} "
                f"policy_ce={policy_ce.float().item()} wdl_ce={wdl_ce.float().item()} "
                f"grid={float(grid)} range={float(range_pen)}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if report:
            parameter_grad_norms = torch.stack([
                p.grad.detach().float().norm()
                for p in model.parameters() if p.grad is not None
            ])
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Under AMP a non-finite grad means the scaler overshot its loss scale;
        # scaler.step skips the update and scaler.update backs the scale off.
        if not amp and not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}; qat={qat}")
        scaler.step(opt)
        scaler.update()

        if report:
            if step and step % 1_000 == 0:
                print(flush=True)
            with torch.no_grad():
                qat_pol, qat_wdl = model(val_enc, qat=True)
                if qat:
                    # In QAT the model IS the int8 model; skip the float forward.
                    active_pol, active_wdl = qat_pol, qat_wdl
                    pol_kl = float("nan")
                else:
                    float_pol, float_wdl = model(val_enc, qat=False)
                    active_pol, active_wdl = float_pol, float_wdl
                    # int8 gap: how far the hard-rounded outputs sit from float.
                    pol_kl = F.kl_div(qat_pol.float().log_softmax(-1),
                                      float_pol.float().softmax(-1), reduction="batchmean")
                probs = active_wdl.float().softmax(-1)
                target_q = val_wdl[:, 0] - val_wdl[:, 2]
                pred_q = probs[:, 0] - probs[:, 2]
                mse = F.mse_loss(pred_q, target_q)
                corr = torch.corrcoef(torch.stack([pred_q, target_q]))[0, 1]
                if not grid_on:  # grid pen off (pre-warmup or QAT); still show offset
                    grid_off = grid_mean_offset(model)
                print(f"[step {step:5d}] metrics ({phase})"
                      f" pol_ce={F.cross_entropy(active_pol, val_pol):.3f}"
                      f" wdl_ce={F.cross_entropy(active_wdl, val_wdl):.3f}"
                      f" mse={mse:.3f} corr={corr:.3f}", flush=True)
                print(f"[step {step:5d}] bangers  range_pen={float(range_pen):.2f}"
                      f" over={n_over}  grid_pen={float(grid):.2f} grid_sets={n_grid}"
                      f" grid_off={grid_off:.3f} frac={grid_frac:.2f}"
                      f" a={grid_alpha:.2f} gsc={grid_scale:.2f}", flush=True)
                print(f"[step {step:5d}] dot      peak={dot_peak:.0f} cap={dot_cap:.0f}"
                      f" pen={float(dot_pen):.2f}  pol_KL={pol_kl:.3f}", flush=True)
                print(f"[step {step:5d}] grads    global={float(grad_norm):.3f}"
                      f" clip={clip:.2f}"
                      f" mean={parameter_grad_norms.mean():.3f}"
                      f" median={parameter_grad_norms.median():.3f}"
                      f" min={parameter_grad_norms.min():.3f}"
                      f" max={parameter_grad_norms.max():.3f}", flush=True)
                print(flush=True)

    if args.export_dir:
        export_and_try_trt(model, args.device, args.export_dir, args.batch,
                           val_enc, run_name=args.run_name)


if __name__ == "__main__":
    main()
