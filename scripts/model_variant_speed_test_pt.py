#!/usr/bin/env python3
"""model_variant_speed_test_pt.py

Conv-backbone speed tests for chess policy+value models.

Models:
  Conv+MHA+GEMM+SmartGate -- channels-first conv encoder -> pos-cat -> 2x self-attn over 64
                              -> cross-attn accumulators (main 6x256->1536, gate 3x256->768)
                              -> 1536-d GEMM trunk (alt GELU/SwiGLU), WDL@4,
                                 quality policy + smartgate logsigmoid, 1858->scatter 4288
  Conv+GEMM               -- 4x gated conv (128-d, channels-first) -> GatedPoolCompressor
                              -> 1536-d alternating GELU/SwiGLU trunk, WDL branch at 4,
                                 policy head 1858 -> scatter to 4288
  Conv+GEMM+SmartGate     -- same encoder -> SplitGatedPoolCompressor(18 pools)
                              -> main[1536] + gate[768]; policy = quality + logsigmoid(gate)

Flags:
  --dry-run              Build models, print param counts, skip speed test
  --skip-trt             Skip ORT+TensorRT
  --skip-hybrid          Skip Conv+MHA+GEMM+SmartGate model
  --skip-convgemm        Skip Conv+GEMM model
  --skip-convgemm-smartgate  Skip Conv+GEMM+SmartGate model
  --batch-sizes    Space-separated list (default: 1 2 4 8 16 32 64 128 256 512 1024)
  --trunk-blocks   GEMM trunk blocks for Conv+MHA (default: 6)
  --gemm-blocks    MLP trunk blocks for Conv+GEMM (default: 6)
  --dropout        Dropout (default: 0.02)
"""

import os
import gc
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN    = 64
ENC_VOCAB  = 21
POLICY_DIM = 4288

N_WARMUP = 50
N_ITERS  = 100

TRT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "conv_chess_trt")

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _load_never_legal():
    import pyfastchess
    return ~torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool()


def _load_sometimes_legal_idx():
    import pyfastchess
    mask = torch.from_numpy(pyfastchess.build_sometimes_legal_mask()).bool()
    return mask.nonzero(as_tuple=True)[0]  # [1858]


# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps   = eps

    def forward(self, x):
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale


class SwiGLUBlock(nn.Module):
    def __init__(self, d, hidden, dropout=0.0):
        super().__init__()
        self.norm   = RMSNorm(d)
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up   = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
        self.drop   = nn.Dropout(dropout)
        nn.init.zeros_(self.w_down.weight)

    def forward(self, x):
        h = self.norm(x)
        return x + self.drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))


class GELUMlpBlock(nn.Module):
    def __init__(self, d, hidden, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d)
        self.w1   = nn.Linear(d, hidden, bias=False)
        self.w2   = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.w2.weight)

    def forward(self, x):
        return x + self.drop(self.w2(F.gelu(self.w1(self.norm(x)))))


# ---------------------------------------------------------------------------
# Conv+MHA+GEMM+SmartGate
# ---------------------------------------------------------------------------

ENC_DIM     = 128
ENC_DIM_MHA = 256   # after cat([x, pos], dim=-1)
ENC_HEADS   = 8
ENC_FF_DIM  = 1024


class ConvBlock2D(nn.Module):
    """Channels-first prenorm conv block. x: [B, d, 8, 8]"""
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


class ConvAttnMlp(nn.Module):
    """Conv encoder -> pos-cat [B,64,256] -> append 8 accum tokens -> [B,72,256]
    -> 1x MHABlock over 72 tokens
    -> 1x CrossAttnBlock(q=accum[8], kv=seq[72]) -> [B,8,256]
    -> split: first 6 -> [B,1536] GEMM belly, last 2 -> [B,512] smartgate
    -> 1536-d GEMM trunk (alt GELU/SwiGLU), WDL@4
    -> quality policy [B,1858] + smartgate logsigmoid(gate) -> scatter [B,4288]
    Gate is suppress-only; bias init=4.0 -> near no-op at init.
    """
    def __init__(self, n_trunk_blocks=6, dropout=0.02):
        super().__init__()
        D      = 1536
        H      = D * 3 // 2       # 2304
        D_gate = 512
        H_gate = D_gate * 3 // 2  # 768

        self.emb    = nn.Embedding(ENC_VOCAB, ENC_DIM)
        self.conv   = nn.ModuleList([ConvBlock2D(ENC_DIM), ConvBlock2D(ENC_DIM)])
        self.pos    = nn.Embedding(SEQ_LEN, ENC_DIM)
        self.ln_x   = nn.LayerNorm(ENC_DIM)
        self.ln_pos = nn.LayerNorm(ENC_DIM)

        self.accum_tokens = nn.Parameter(torch.randn(1, 8, ENC_DIM_MHA) * 0.02)
        self.self_attn    = MHABlock(ENC_DIM_MHA, ENC_HEADS, ENC_FF_DIM)
        self.cross_attn   = CrossAttnBlock(ENC_DIM_MHA, ENC_HEADS, ENC_FF_DIM)

        def make_block(i):
            return GELUMlpBlock(D, H, dropout) if i % 2 == 0 else SwiGLUBlock(D, H, dropout)
        self.trunk = nn.ModuleList([make_block(i) for i in range(n_trunk_blocks)])

        self.val_norm = RMSNorm(D)
        self.val_w1   = nn.Linear(D, D // 2, bias=False)
        self.val_w2   = nn.Linear(D // 2, 3, bias=False)

        self.pol_norm  = RMSNorm(D)
        self.pol_out   = nn.Linear(D, 1858, bias=False)

        self.gate_blk  = SwiGLUBlock(D_gate, H_gate, dropout)
        self.gate_norm = RMSNorm(D_gate)
        self.gate_out  = nn.Linear(D_gate, 1858, bias=True)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.constant_(self.gate_out.bias, 4.0)

        self.register_buffer("sl_idx", _load_sometimes_legal_idx())

    def forward(self, tokens):
        B = tokens.shape[0]
        x = self.emb(tokens)                                                           # [B, 64, 128]
        x = x.reshape(B, 8, 8, ENC_DIM).permute(0, 3, 1, 2)                         # [B, 128, 8, 8]
        for blk in self.conv:
            x = blk(x)
        x   = x.reshape(B, ENC_DIM, SEQ_LEN).permute(0, 2, 1)                        # [B, 64, 128]
        pos = self.pos(torch.arange(SEQ_LEN, device=tokens.device))
        x   = torch.cat([self.ln_x(x), self.ln_pos(pos).unsqueeze(0).expand(B, -1, -1)], dim=-1)  # [B, 64, 256]

        seq = torch.cat([x, self.accum_tokens.expand(B, -1, -1)], dim=1)              # [B, 72, 256]
        seq = self.self_attn(seq)                                                       # [B, 72, 256]
        g   = self.cross_attn(seq[:, -8:], seq)                                        # [B, 8, 256]

        main_x = g[:, :6, :].reshape(B, -1)                                            # [B, 1536]
        gate_x = g[:, 6:, :].reshape(B, -1)                                            # [B, 512]

        for blk in self.trunk[:4]:
            main_x = blk(main_x)
        val = self.val_w2(F.silu(self.val_w1(self.val_norm(main_x))))

        for blk in self.trunk[4:]:
            main_x = blk(main_x)

        quality_logits = self.pol_out(self.pol_norm(main_x))                           # [B, 1858]

        gate_x   = self.gate_blk(gate_x)
        gate_raw = self.gate_out(self.gate_norm(gate_x))                               # [B, 1858]

        legal_logits = quality_logits + F.logsigmoid(gate_raw)
        pol = torch.full((B, POLICY_DIM), -3e4, device=legal_logits.device, dtype=legal_logits.dtype)
        pol.scatter_(1, self.sl_idx.unsqueeze(0).expand(B, -1), legal_logits)
        return pol, val


def build_conv_attn_model(n_trunk_blocks=6, dropout=0.02):
    m = ConvAttnMlp(n_trunk_blocks=n_trunk_blocks, dropout=dropout)
    print(f"  params: {sum(p.numel() for p in m.parameters()):,}")
    return m


# ---------------------------------------------------------------------------
# Conv+GEMM
# ---------------------------------------------------------------------------

GEMM_ENC_DIM   = 128
GEMM_N_POOLS   = 12
GEMM_TRUNK_DIM = GEMM_N_POOLS * GEMM_ENC_DIM  # 1536


class GatedConvBlock2D(nn.Module):
    """Channels-first gated conv block. x: [B, d, 8, 8]"""
    def __init__(self, d=128):
        super().__init__()
        self.ln     = nn.LayerNorm([d, 8, 8])
        self.c_gate = nn.Conv2d(d, d, 3, padding=1, bias=False)
        self.c_up   = nn.Conv2d(d, d, 3, padding=1, bias=False)
        self.c_out  = nn.Conv2d(d, d, 3, padding=1, bias=False)

    def forward(self, x):                                      # [B, d, 8, 8]
        h = self.ln(x)
        return x + self.c_out(F.silu(self.c_gate(h)) * self.c_up(h))


class GatedPoolCompressor(nn.Module):
    """N softmax-attention pools over 64 squares -> [B, N*channels]."""
    def __init__(self, channels=128, n_pools=12):
        super().__init__()
        self.channels   = channels
        self.n_pools    = n_pools
        self.norm       = nn.LayerNorm(channels)
        self.value_proj = nn.Linear(channels, channels * n_pools, bias=False)
        self.gate_proj  = nn.Linear(channels, n_pools, bias=False)

    def forward(self, x):                                      # [B, channels, 8, 8] channels-first
        B      = x.shape[0]
        x      = x.reshape(B, self.channels, 64).permute(0, 2, 1)  # [B, 64, C] — 2 free views
        h      = self.norm(x)
        values = self.value_proj(h).reshape(B, 64, self.n_pools, self.channels)
        values = values.permute(0, 2, 1, 3)                   # [B, P, 64, C]
        gates  = self.gate_proj(h).permute(0, 2, 1).softmax(dim=-1)  # [B, P, 64]
        pooled = (values * gates[:, :, :, None]).sum(dim=2)   # [B, P, C]
        return pooled.reshape(B, self.n_pools * self.channels) # [B, 1536]


class ConvGEMMMlp(nn.Module):
    """6x GatedConvBlock2D(128, channels-first) -> GatedPoolCompressor -> [B, 1536]
    -> 4x alternating GELU/SwiGLU -> WDL head
    -> 2x alternating GELU/SwiGLU -> policy Linear(1536->1858) -> scatter to 4288
    """
    def __init__(self, n_trunk_blocks=6, dropout=0.02):
        super().__init__()
        D = GEMM_TRUNK_DIM       # 1536
        H = D * 3 // 2          # 2304

        self.emb      = nn.Embedding(ENC_VOCAB, GEMM_ENC_DIM)
        self.conv     = nn.ModuleList([GatedConvBlock2D(GEMM_ENC_DIM) for _ in range(4)])
        self.compress = GatedPoolCompressor(GEMM_ENC_DIM, GEMM_N_POOLS)

        def make_block(i):
            return GELUMlpBlock(D, H, dropout) if i % 2 == 0 else SwiGLUBlock(D, H, dropout)
        self.trunk = nn.ModuleList([make_block(i) for i in range(n_trunk_blocks)])

        self.val_norm = RMSNorm(D)
        self.val_w1   = nn.Linear(D, D // 2, bias=False)
        self.val_w2   = nn.Linear(D // 2, 3, bias=False)

        self.pol_norm = RMSNorm(D)
        self.pol_out  = nn.Linear(D, 1858, bias=False)
        self.register_buffer("sl_idx", _load_sometimes_legal_idx())  # [1858] -> 4288

    def forward(self, tokens):
        B = tokens.shape[0]
        x = self.emb(tokens)                                           # [B, 64, 128] contiguous
        x = x.reshape(B, 8, 8, GEMM_ENC_DIM).permute(0, 3, 1, 2)    # [B, 128, 8, 8] — 2 free views
        for blk in self.conv:
            x = blk(x)
        x = self.compress(x)                                           # [B, 1536]

        for blk in self.trunk[:4]:
            x = blk(x)

        val = self.val_w2(F.silu(self.val_w1(self.val_norm(x))))      # [B, 3]

        for blk in self.trunk[4:]:
            x = blk(x)

        legal_logits = self.pol_out(self.pol_norm(x))                  # [B, 1858]
        pol = torch.full((B, POLICY_DIM), -3e4, device=x.device, dtype=x.dtype)
        pol.scatter_(1, self.sl_idx.unsqueeze(0).expand(B, -1), legal_logits)

        return pol, val


def build_conv_gemm_model(n_trunk_blocks=6, dropout=0.02):
    m = ConvGEMMMlp(n_trunk_blocks=n_trunk_blocks, dropout=dropout)
    print(f"  params: {sum(p.numel() for p in m.parameters()):,}")
    return m


# ---------------------------------------------------------------------------
# Conv+GEMM+SmartGate
# ---------------------------------------------------------------------------

class SplitGatedPoolCompressor(nn.Module):
    """18-pool gated compressor split into main (12 pools) + gate (6 pools)."""
    def __init__(self, channels=128, n_main=12, n_gate=6):
        super().__init__()
        n_pools          = n_main + n_gate
        self.channels    = channels
        self.n_pools     = n_pools
        self.n_main      = n_main
        self.n_gate      = n_gate
        self.norm        = nn.LayerNorm(channels)
        self.value_proj  = nn.Linear(channels, channels * n_pools, bias=False)
        self.gate_proj   = nn.Linear(channels, n_pools, bias=False)

    def forward(self, x):                                       # [B, C, 8, 8]
        B      = x.shape[0]
        x      = x.reshape(B, self.channels, 64).permute(0, 2, 1)   # [B, 64, C]
        h      = self.norm(x)
        values = self.value_proj(h).reshape(B, 64, self.n_pools, self.channels)
        values = values.permute(0, 2, 1, 3)                          # [B, P, 64, C]
        gates  = self.gate_proj(h).permute(0, 2, 1).softmax(dim=-1) # [B, P, 64]
        pooled = (values * gates[:, :, :, None]).sum(dim=2)          # [B, P, C]
        main_x = pooled[:, :self.n_main, :].reshape(B, self.n_main * self.channels)
        gate_x = pooled[:, self.n_main:, :].reshape(B, self.n_gate * self.channels)
        return main_x, gate_x


class ConvGEMMSmartGateMlp(nn.Module):
    """Same conv encoder as Conv+GEMM, but with a split compressor.
    Main arm (1536-d): identical 6x trunk + WDL + quality policy logits.
    Gate arm (768-d):  1x SwiGLU -> Linear(768, 1858, bias=True).
    Combined: legal_logits = quality_logits + F.logsigmoid(gate_raw).
    logsigmoid is always <= 0, so the gate is suppress-only.
    Gate bias init=4.0 -> logsigmoid(4.0) ~= -0.018 (near no-op at init).
    """
    def __init__(self, n_trunk_blocks=6, dropout=0.02):
        super().__init__()
        D      = GEMM_TRUNK_DIM        # 1536
        H      = D * 3 // 2            # 2304
        D_gate = 768
        H_gate = D_gate * 3 // 2       # 1152

        self.emb      = nn.Embedding(ENC_VOCAB, GEMM_ENC_DIM)
        self.conv     = nn.ModuleList([GatedConvBlock2D(GEMM_ENC_DIM) for _ in range(4)])
        self.compress = SplitGatedPoolCompressor(GEMM_ENC_DIM, n_main=12, n_gate=6)

        def make_block(i):
            return GELUMlpBlock(D, H, dropout) if i % 2 == 0 else SwiGLUBlock(D, H, dropout)
        self.trunk = nn.ModuleList([make_block(i) for i in range(n_trunk_blocks)])

        self.val_norm = RMSNorm(D)
        self.val_w1   = nn.Linear(D, D // 2, bias=False)
        self.val_w2   = nn.Linear(D // 2, 3, bias=False)

        self.pol_norm = RMSNorm(D)
        self.pol_out  = nn.Linear(D, 1858, bias=False)

        self.gate_blk  = SwiGLUBlock(D_gate, H_gate, dropout)
        self.gate_norm = RMSNorm(D_gate)
        self.gate_out  = nn.Linear(D_gate, 1858, bias=True)
        nn.init.constant_(self.gate_out.bias, 4.0)

        self.register_buffer("sl_idx", _load_sometimes_legal_idx())

    def forward(self, tokens):
        B = tokens.shape[0]
        x = self.emb(tokens)
        x = x.reshape(B, 8, 8, GEMM_ENC_DIM).permute(0, 3, 1, 2)   # [B, 128, 8, 8]
        for blk in self.conv:
            x = blk(x)
        main_x, gate_x = self.compress(x)                             # [B,1536], [B,768]

        for blk in self.trunk[:4]:
            main_x = blk(main_x)
        val = self.val_w2(F.silu(self.val_w1(self.val_norm(main_x))))

        for blk in self.trunk[4:]:
            main_x = blk(main_x)
        quality_logits = self.pol_out(self.pol_norm(main_x))          # [B, 1858]

        gate_x   = self.gate_blk(gate_x)
        gate_raw = self.gate_out(self.gate_norm(gate_x))              # [B, 1858]

        legal_logits = quality_logits + F.logsigmoid(gate_raw)
        pol = torch.full((B, POLICY_DIM), -3e4, device=legal_logits.device, dtype=legal_logits.dtype)
        pol.scatter_(1, self.sl_idx.unsqueeze(0).expand(B, -1), legal_logits)
        return pol, val


def build_conv_gemm_smartgate_model(n_trunk_blocks=6, dropout=0.02):
    m = ConvGEMMSmartGateMlp(n_trunk_blocks=n_trunk_blocks, dropout=dropout)
    print(f"  params: {sum(p.numel() for p in m.parameters()):,}")
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--skip-trt",       action="store_true")
    p.add_argument("--skip-hybrid",    action="store_true")
    p.add_argument("--skip-convgemm",           action="store_true")
    p.add_argument("--skip-convgemm-smartgate", action="store_true")
    p.add_argument("--batch-sizes",    nargs="+", type=int, default=DEFAULT_BATCH_SIZES, metavar="B")
    p.add_argument("--trunk-blocks",   type=int,   default=6,    help="GEMM trunk blocks for Conv+MHA")
    p.add_argument("--gemm-blocks",    type=int,   default=6,    help="MLP trunk blocks for Conv+GEMM")
    p.add_argument("--dropout",        type=float, default=0.02)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_tokens(batch_size, vocab=ENC_VOCAB):
    return np.random.randint(0, vocab, (batch_size, SEQ_LEN), dtype=np.int32)


# ---------------------------------------------------------------------------
# Inference wrappers
# ---------------------------------------------------------------------------

def make_pt_eager_infer(model, device):
    model = model.half().to(device).eval()

    def infer(tokens_np):
        with torch.no_grad():
            x = torch.from_numpy(tokens_np).long().to(device)
            p, v = model(x)
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    return infer


def export_to_onnx(model, device, onnx_path, opset=14):
    print(f"  Exporting to ONNX (opset {opset}) ...")
    model = model.half().to(device).eval()
    dummy = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    axes  = {"tokens": {0: "batch"}, "policy_logits": {0: "batch"}, "value_out": {0: "batch"}}
    kwargs = dict(
        opset_version=opset,
        input_names=["tokens"],
        output_names=["policy_logits", "value_out"],
        dynamic_axes=axes,
    )
    try:
        torch.onnx.export(model, dummy, onnx_path, **kwargs)
    except Exception as e:
        if opset < 16:
            print(f"  opset {opset} failed ({type(e).__name__}), retrying opset 16 ...")
            torch.onnx.export(model, dummy, onnx_path, **{**kwargs, "opset_version": 16})
        else:
            raise
    print("  ONNX export done.")


def make_trt_infer(onnx_path, cache_dir, max_bs=512):
    import hashlib
    import tensorrt
    import onnxruntime as ort

    h = hashlib.sha256()
    with open(onnx_path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    prefix = f"conv_chess_{h.hexdigest()[:12]}"

    ort.set_default_logger_severity(3)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3

    trt_opts = {
        "trt_engine_cache_enable":  True,
        "trt_engine_cache_path":    cache_dir,
        "trt_engine_cache_prefix":  prefix,
        "trt_fp16_enable":          True,
        "trt_force_timing_cache":   True,
        "trt_max_workspace_size":   4 * 1024 * 1024 * 1024,
        "trt_profile_min_shapes":   "tokens:1x64",
        "trt_profile_opt_shapes":   f"tokens:{max_bs}x64",
        "trt_profile_max_shapes":   f"tokens:{max_bs}x64",
        "trt_timing_cache_enable":  True,
        "trt_timing_cache_path":    cache_dir,
    }

    engines = [f for f in os.listdir(cache_dir) if f.startswith(prefix) and f.endswith(".engine")]
    cached  = bool(engines)
    print(f"  [trt] {'loading cached engine' if cached else 'no cache, compiling...'}")

    providers = [("TensorrtExecutionProvider", trt_opts)]
    t0      = time.perf_counter()
    session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    elapsed = time.perf_counter() - t0
    active  = session.get_providers()[0]
    if active != "TensorrtExecutionProvider":
        raise RuntimeError(f"[trt] expected TensorrtExecutionProvider but got {active}")
    print(f"  [trt] ready in {elapsed:.1f}s")

    iname = session.get_inputs()[0].name

    def infer(tokens_np):
        return session.run(None, {iname: tokens_np.astype(np.int64)})

    return infer, session


# ---------------------------------------------------------------------------
# Speed test
# ---------------------------------------------------------------------------

def speed_test(label, infer_fn, batch_sizes):
    print(f"\n  {'-'*56}")
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
        elapsed = time.perf_counter() - t0
        avg_s   = elapsed / N_ITERS
        tput    = b / avg_s
        print(f"  {b:<7} {avg_s*1000:>12.2f} {tput:>13.0f}")
        results[b] = {"latency_ms": round(avg_s * 1000, 3), "throughput": round(tput, 1)}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    bs     = args.batch_sizes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch device: {device}")

    # Conv+MHA+GEMM+SmartGate
    if not args.skip_hybrid:
        print(f"\n{'='*60}")
        print(f"  Conv+MHA+GEMM+SmartGate  (2xConvBlock2D({ENC_DIM}) -> pos-cat -> [{ENC_DIM_MHA}]"
              f" -> append 8 accum tokens -> 1x self-attn(72)"
              f" -> 1x cross-attn(q=accum[8], kv=72) -> split first6/last2 -> [1536]/[512]"
              f" -> {args.trunk_blocks}x alt GELU/SwiGLU, WDL@4,"
              f" quality+logsigmoid(gate) 1858->4288)")
        build_conv_attn_model(args.trunk_blocks, args.dropout)

        if not args.dry_run:
            try:
                print(f"\n  Building Conv+MHA+GEMM+SmartGate PT eager ...")
                eager_h = make_pt_eager_infer(build_conv_attn_model(args.trunk_blocks, args.dropout), device)
                res = speed_test("Conv+MHA+GEMM+SmartGate  PT eager [fp16]", eager_h, bs)
                del eager_h; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [ERROR] Conv+MHA+GEMM+SmartGate PT eager: {e}")

        if not args.skip_trt:
            try:
                os.makedirs(TRT_CACHE, exist_ok=True)
                onnx_h = os.path.join(TRT_CACHE, f"conv_mha_t{args.trunk_blocks}.onnx")
                if not os.path.exists(onnx_h):
                    export_to_onnx(build_conv_attn_model(args.trunk_blocks, args.dropout), device, onnx_h)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_h}")
                trt_h, trt_sess_h = make_trt_infer(onnx_h, TRT_CACHE, max_bs=max(bs))
                res = speed_test("Conv+MHA+GEMM+SmartGate  ORT TRT [fp16]", trt_h, bs)
                del trt_sess_h; gc.collect()
            except Exception as e:
                print(f"  [ERROR] Conv+MHA+GEMM+SmartGate TRT: {e}")

    # Conv+GEMM
    if not args.skip_convgemm:
        print(f"\n{'='*60}")
        print(f"  Conv+GEMM  (4x GatedConvBlock2D({GEMM_ENC_DIM}) channels-first"
              f" -> GatedPoolCompressor({GEMM_N_POOLS} pools) -> [{GEMM_TRUNK_DIM}]"
              f" -> {args.gemm_blocks}x alt GELU/SwiGLU, WDL@4, policy 1858->4288)")
        build_conv_gemm_model(args.gemm_blocks, args.dropout)

        if not args.dry_run:
            try:
                print(f"\n  Building Conv+GEMM PT eager ...")
                eager_g = make_pt_eager_infer(
                    build_conv_gemm_model(args.gemm_blocks, args.dropout), device)
                res = speed_test("Conv+GEMM  PT eager [fp16]", eager_g, bs)
                del eager_g; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [ERROR] Conv+GEMM PT eager: {e}")

        if not args.skip_trt:
            try:
                os.makedirs(TRT_CACHE, exist_ok=True)
                onnx_g = os.path.join(TRT_CACHE, f"conv_gemm_t{args.gemm_blocks}.onnx")
                if not os.path.exists(onnx_g):
                    export_to_onnx(
                        build_conv_gemm_model(args.gemm_blocks, args.dropout), device, onnx_g)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_g}")
                trt_g, trt_sess_g = make_trt_infer(onnx_g, TRT_CACHE, max_bs=max(bs))
                res = speed_test("Conv+GEMM  ORT TRT [fp16]", trt_g, bs)
                del trt_sess_g; gc.collect()
            except Exception as e:
                print(f"  [ERROR] Conv+GEMM TRT: {e}")

    # Conv+GEMM+SmartGate
    if not args.skip_convgemm_smartgate:
        print(f"\n{'='*60}")
        print(f"  Conv+GEMM+SmartGate  (same encoder -> SplitGatedPoolCompressor(18 pools)"
              f" -> main[1536] + gate[768]"
              f" -> {args.gemm_blocks}x alt GELU/SwiGLU + gate SwiGLU,"
              f" policy quality+logsigmoid(gate) 1858->4288)")
        build_conv_gemm_smartgate_model(args.gemm_blocks, args.dropout)

        if not args.dry_run:
            try:
                print(f"\n  Building Conv+GEMM+SmartGate PT eager ...")
                eager_sg = make_pt_eager_infer(
                    build_conv_gemm_smartgate_model(args.gemm_blocks, args.dropout), device)
                res = speed_test("Conv+GEMM+SmartGate  PT eager [fp16]", eager_sg, bs)
                del eager_sg; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [ERROR] Conv+GEMM+SmartGate PT eager: {e}")

        if not args.skip_trt:
            try:
                os.makedirs(TRT_CACHE, exist_ok=True)
                onnx_sg = os.path.join(TRT_CACHE, f"conv_gemm_smartgate_t{args.gemm_blocks}.onnx")
                if not os.path.exists(onnx_sg):
                    export_to_onnx(
                        build_conv_gemm_smartgate_model(args.gemm_blocks, args.dropout), device, onnx_sg)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_sg}")
                trt_sg, trt_sess_sg = make_trt_infer(onnx_sg, TRT_CACHE, max_bs=max(bs))
                res = speed_test("Conv+GEMM+SmartGate  ORT TRT [fp16]", trt_sg, bs)
                del trt_sess_sg; gc.collect()
            except Exception as e:
                print(f"  [ERROR] Conv+GEMM+SmartGate TRT: {e}")

    if args.dry_run:
        print("\n  Dry run complete.")
        return

    print("\nDone.")


if __name__ == "__main__":
    main()
