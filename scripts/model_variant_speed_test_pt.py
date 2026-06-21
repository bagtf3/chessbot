#!/usr/bin/env python3
"""model_variant_speed_test_pt.py

Conv-backbone speed tests for chess policy+value models.

Models:
  Conv+MHA+GEMM+SmartGate -- channels-first conv encoder -> pos-cat -> 2x self-attn over 64
                              -> cross-attn accumulators (main 6x256->1536, gate 2x256->512)
                              -> 1536-d GEMM trunk (alt GELU/SwiGLU), WDL@4,
                                 quality policy + smartgate logsigmoid, 1858->scatter 4288
  13m-precond-conformer    -- 4x Conv(256) preconditioner -> concat pos(256) -> 512-d
                              -> 4x [prenorm-MHA(heads=8) -> FF(512)] with varying ff_dims
                              -> MHA policy + attn-pool value heads
  13m-precond-conformer-v2 -- same preconditioner; 8 specialized global accumulators
                              (WDL/gate/from-spec/to-spec/shared) -> 4x transformer blocks
                              -> shared trunk_ln -> WDL(0) + SmartGate(1) + from/to policy(2-7)

Flags:
  --dry-run              Build models, print param counts, skip speed test
  --skip-trt             Skip ORT+TensorRT
  --skip-hybrid          Skip Conv+MHA+GEMM+SmartGate model
  --skip-precond         Skip 13m-precond-conformer model
  --skip-precond-v2      Skip 13m-precond-conformer-v2 model
  --batch-sizes    Space-separated list (default: 1 2 4 8 16 32 64 128 256 512 1024)
  --trunk-blocks   GEMM trunk blocks for Conv+MHA (default: 6)
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

import math
from chessbot.model import build_pt_precond_conformer, make_ln2d

SEQ_LEN    = 64
ENC_VOCAB  = 21
POLICY_DIM = 4288

N_WARMUP = 50
N_ITERS  = 100

TRT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "conv_chess_trt")

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

PRECOND_CFG = dict(
    precond_conformer=True,
    conv_filters=256, num_heads=8, dropout=0.05,
    pre_blocks=4, mha_blocks=4,
)


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
# 13m-precond-conformer v2: 8 specialized global accumulators
#   0 = WDL            1 = smartgate
#   2,3 = from-specials  4,5 = to-specials  6,7 = shared context
# Post-trunk shared LN; policy head has no starting LN.
# Gate is suppress-only (logsigmoid <= 0); composed with quality before scatter.
# ---------------------------------------------------------------------------

def build_precond_v2_model(cfg=None):
    cfg = cfg or PRECOND_CFG

    CF  = cfg["conv_filters"]   # 256
    D   = CF * 2                # 512
    nh  = cfg["num_heads"]      # 8
    dr  = cfg["dropout"]        # 0.05
    pb  = cfg["pre_blocks"]     # 4
    PDH = 256                   # policy inner dim

    sl_idx = _load_sometimes_legal_idx()   # [1858]

    class ConvBlock(nn.Module):
        def __init__(self, prenorm=True):
            super().__init__()
            self.ln = make_ln2d(CF) if prenorm else None
            self.c1 = nn.Conv2d(CF, CF, 3, padding=1, bias=False)
            self.c2 = nn.Conv2d(CF, CF, 3, padding=1, bias=False)
        def forward(self, x):
            r = x
            h = self.ln(x) if self.ln is not None else x
            return r + F.leaky_relu(self.c2(F.leaky_relu(self.c1(h), 0.01)), 0.01)

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
            r = x; n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = r + self.drop(h)
            return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb     = nn.Embedding(ENC_VOCAB, CF)
            self.pre     = nn.ModuleList([ConvBlock(prenorm=(i > 0)) for i in range(pb)])
            self.pos     = nn.Embedding(SEQ_LEN, CF)
            self.conv_ln = nn.LayerNorm(CF)

            self.global_tokens = nn.Parameter(torch.randn(1, 8, D) * 0.02)

            ff_dims = [768, 1024, 1024, 768]
            self.blocks   = nn.ModuleList([TxBlock(ffd) for ffd in ff_dims])
            self.trunk_ln = nn.LayerNorm(D)

            # WDL: accumulator 0 (position 64 in 72-token sequence)
            self.wdl_w1 = nn.Linear(D, D // 2, bias=False)
            self.wdl_w2 = nn.Linear(D // 2, 3, bias=False)

            # SmartGate: accumulator 1 (position 65)
            # No prenorm — input is already trunk_ln'd
            self.gate_w_gate = nn.Linear(D, D * 3 // 2, bias=False)
            self.gate_w_up   = nn.Linear(D, D * 3 // 2, bias=False)
            self.gate_w_down = nn.Linear(D * 3 // 2, D, bias=False)
            self.gate_drop   = nn.Dropout(dr)
            nn.init.zeros_(self.gate_w_down.weight)
            self.gate_norm = RMSNorm(D)
            self.gate_out  = nn.Linear(D, 1858, bias=True)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.constant_(self.gate_out.bias, 4.0)

            # Policy head: accumulators 2-7, no starting LN (trunk_ln covers it)
            # from_set = board(0-63) + from-specials(66,67) + shared(70,71) -> 68 tokens
            # to_set   = board(0-63) + to-specials(68,69)   + shared(70,71) -> 68 tokens
            self.from_proj = nn.Linear(D, PDH)
            self.from_ln   = nn.LayerNorm(PDH)
            self.from_mha  = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
            self.from_out  = nn.Linear(PDH, PDH)

            self.to_proj   = nn.Linear(D, PDH)
            self.to_ln     = nn.LayerNorm(PDH)
            self.to_mha    = nn.MultiheadAttention(PDH, 4, dropout=0.0, batch_first=True)
            self.to_out    = nn.Linear(PDH, PDH)

            self.scale = 1.0 / math.sqrt(PDH)

            # Promo head (conv over board features)
            self.promo_mix = nn.Conv2d(D, 64, 1, bias=False)
            self.promo_mln = make_ln2d(64)
            self.promo_c1  = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.promo_ln  = make_ln2d(64)
            self.promo_out = nn.Conv2d(64, 3, 1)

            self.register_buffer("sl_idx", sl_idx)

        def forward(self, tokens):
            B = tokens.shape[0]

            # Conv preconditioner
            x = self.emb(tokens).reshape(B, 8, 8, CF).permute(0, 3, 1, 2).contiguous()
            for blk in self.pre:
                x = blk(x)
            seq = self.conv_ln(x.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, CF))
            pos = self.pos(torch.arange(SEQ_LEN, device=tokens.device)).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([seq, pos], dim=-1)                                   # [B, 64, D]

            # Append 8 global accumulators -> [B, 72, D]
            x = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)

            for blk in self.blocks:
                x = blk(x)

            x = self.trunk_ln(x)                                                # [B, 72, D]

            # WDL: accumulator 0
            wdl = self.wdl_w2(F.gelu(self.wdl_w1(x[:, 64, :])))               # [B, 3]

            # SmartGate: accumulator 1 (no prenorm — already trunk_ln'd)
            gi       = x[:, 65, :]
            h        = F.silu(self.gate_w_gate(gi)) * self.gate_w_up(gi)
            g        = gi + self.gate_drop(self.gate_w_down(h))
            gate_raw = self.gate_out(self.gate_norm(g))                         # [B, 1858]

            # Policy: construct from/to sets via contiguous slices + cat
            from_set = torch.cat([x[:, :64, :], x[:, 66:68, :], x[:, 70:72, :]], dim=1)  # [B, 68, D]
            to_set   = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:72, :]], dim=1)  # [B, 68, D]

            # From branch: board positions query the from-specialist set
            f_proj = F.gelu(self.from_proj(from_set))                          # [B, 68, PDH]
            fn     = self.from_ln(f_proj)
            fh, _  = self.from_mha(fn[:, :64, :], fn, fn, need_weights=False)  # [B, 64, PDH]
            fv     = self.from_out(f_proj[:, :64, :] + fh)                     # [B, 64, PDH]

            # To branch: board positions query the to-specialist set
            t_proj = F.gelu(self.to_proj(to_set))                              # [B, 68, PDH]
            tn     = self.to_ln(t_proj)
            th, _  = self.to_mha(tn[:, :64, :], tn, tn, need_weights=False)   # [B, 64, PDH]
            tv     = self.to_out(t_proj[:, :64, :] + th)                       # [B, 64, PDH]

            dots  = torch.bmm(fv.float(), tv.float().transpose(1, 2)).mul(self.scale).reshape(B, 64 * 64)

            # Promo head
            xb    = x[:, :64, :].reshape(B, 8, 8, D).permute(0, 3, 1, 2).contiguous()
            p     = F.leaky_relu(self.promo_mln(self.promo_mix(xb)), 0.02)
            sk    = p
            p     = F.leaky_relu(self.promo_ln(self.promo_c1(p)), 0.02)
            promo = self.promo_out(p + sk).permute(0, 2, 3, 1).reshape(B, 8 * 8 * 3).float()

            # Compose gate with quality: gather 1858 sometimes-legal, add logsigmoid, scatter
            raw_4288 = torch.cat([dots, promo], dim=1)                          # [B, 4288]
            q_sl     = raw_4288[:, self.sl_idx]                                 # [B, 1858]
            combined = q_sl + F.logsigmoid(gate_raw.float())                   # [B, 1858]
            pol = torch.full((B, POLICY_DIM), -3e4, device=tokens.device, dtype=combined.dtype)
            pol.scatter_(1, self.sl_idx.unsqueeze(0).expand(B, -1), combined)
            return pol.to(x.dtype), wdl

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run",           action="store_true")
    p.add_argument("--skip-trt",          action="store_true")
    p.add_argument("--skip-hybrid",       action="store_true")
    p.add_argument("--skip-precond",      action="store_true")
    p.add_argument("--skip-precond-v2",   action="store_true")
    p.add_argument("--batch-sizes",       nargs="+", type=int, default=DEFAULT_BATCH_SIZES, metavar="B")
    p.add_argument("--trunk-blocks",      type=int,   default=6,    help="GEMM trunk blocks for Conv+MHA")
    p.add_argument("--dropout",           type=float, default=0.02)
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
                speed_test("Conv+MHA+GEMM+SmartGate  PT eager [fp16]", eager_h, bs)
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
                speed_test("Conv+MHA+GEMM+SmartGate  ORT TRT [fp16]", trt_h, bs)
                del trt_sess_h; gc.collect()
            except Exception as e:
                print(f"  [ERROR] Conv+MHA+GEMM+SmartGate TRT: {e}")

    # 13m-precond-conformer
    if not args.skip_precond:
        print(f"\n{'='*60}")
        print(f"  13m-precond-conformer  (4xConv(256) -> cat pos(256) -> 512-d"
              f" -> 4x [prenorm-MHA(heads=8) -> FF] -> MHA policy + attn-pool value)")
        build_pt_precond_conformer(PRECOND_CFG)

        if not args.dry_run:
            try:
                print(f"\n  Building 13m-precond-conformer PT eager ...")
                eager_p = make_pt_eager_infer(build_pt_precond_conformer(PRECOND_CFG), device)
                speed_test("13m-precond-conformer  PT eager [fp16]", eager_p, bs)
                del eager_p; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [ERROR] 13m-precond-conformer PT eager: {e}")

        if not args.skip_trt:
            try:
                os.makedirs(TRT_CACHE, exist_ok=True)
                onnx_p = os.path.join(TRT_CACHE, "precond_conformer_pb4_mb4.onnx")
                if not os.path.exists(onnx_p):
                    export_to_onnx(build_pt_precond_conformer(PRECOND_CFG), device, onnx_p)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_p}")
                trt_p, trt_sess_p = make_trt_infer(onnx_p, TRT_CACHE, max_bs=max(bs))
                speed_test("13m-precond-conformer  ORT TRT [fp16]", trt_p, bs)
                del trt_sess_p; gc.collect()
            except Exception as e:
                print(f"  [ERROR] 13m-precond-conformer TRT: {e}")

    # 13m-precond-conformer v2
    if not args.skip_precond_v2:
        print(f"\n{'='*60}")
        print(f"  13m-precond-conformer-v2  (8 global accumulators:"
              f" WDL/gate/from-spec/to-spec/shared -> 4x [prenorm-MHA(heads=8)]"
              f" -> trunk_ln -> specialized heads with SmartGate policy)")
        build_precond_v2_model()

        if not args.dry_run:
            try:
                print(f"\n  Building 13m-precond-conformer-v2 PT eager ...")
                eager_v2 = make_pt_eager_infer(build_precond_v2_model(), device)
                speed_test("13m-precond-conformer-v2  PT eager [fp16]", eager_v2, bs)
                del eager_v2; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [ERROR] 13m-precond-conformer-v2 PT eager: {e}")

        if not args.skip_trt:
            try:
                os.makedirs(TRT_CACHE, exist_ok=True)
                onnx_v2 = os.path.join(TRT_CACHE, "precond_conformer_v2_pb4_mb4.onnx")
                if not os.path.exists(onnx_v2):
                    export_to_onnx(build_precond_v2_model(), device, onnx_v2)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_v2}")
                trt_v2, trt_sess_v2 = make_trt_infer(onnx_v2, TRT_CACHE, max_bs=max(bs))
                speed_test("13m-precond-conformer-v2  ORT TRT [fp16]", trt_v2, bs)
                del trt_sess_v2; gc.collect()
            except Exception as e:
                print(f"  [ERROR] 13m-precond-conformer-v2 TRT: {e}")

    if args.dry_run:
        print("\n  Dry run complete.")
        return

    print("\nDone.")


if __name__ == "__main__":
    main()
