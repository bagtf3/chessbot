#!/usr/bin/env python3
"""model_variant_speed_test_pt.py

PyTorch-only speed test for NNUE-style MLP chess model.

Architecture:
  - 400-vocab embedding table (512-d per side)
  - 64 token inputs -> us/them 512-d accumulators -> concat(us, -them) -> 1024-d
  - N residual MLP blocks: x = x + w2(gelu(w1(ln(x))))
  - Policy head (independent): 1024 -> 1024 -> 4288, never-legal masked
  - Value head  (independent): 1024 -> 512  -> 3

Backends:
  PT eager    (fp16)
  PT compiled (fp16, inductor -> cudagraphs fallback)
  ORT + TRT   (fp16, engine cached)

Flags:
  --dry-run      Build model, print param count, skip speed test
  --skip-trt     Skip ORT+TensorRT
  --batch-sizes  Space-separated list (default: 1 4 8 16 32 64 128 256)
  --blocks       Number of residual blocks (default: 10)
  --d-model      Trunk dimension (default: 1024)
"""

import os
import gc
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 400
SEQ_LEN    = 64
EMB_DIM    = 512      # per-side accumulator; trunk = 2 * EMB_DIM after concat
POLICY_DIM = 4288

N_WARMUP   = 50
N_ITERS    = 100

TRT_CACHE  = os.path.join(os.path.expanduser("~"), ".cache", "nnue_mlp_trt")

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _load_never_legal() -> torch.Tensor:
    import pyfastchess
    sometimes = pyfastchess.build_sometimes_legal_mask().astype(bool)
    return torch.from_numpy(~sometimes)   # [4288] bool, True = never legal


# ---------------------------------------------------------------------------
# Standard model
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, d)
        self.w2 = nn.Linear(d, d)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(self.ln(x))))


class NNUEMlp(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, d_model=1024, n_blocks=10):
        super().__init__()
        assert d_model == emb_dim * 2, f"d_model must be 2 * emb_dim"
        self.emb   = nn.Embedding(vocab_size, emb_dim)
        self.trunk = nn.ModuleList([ResBlock(d_model) for _ in range(n_blocks)])

        # independent policy head
        self.pol_ln = nn.LayerNorm(d_model)
        self.pol_w1 = nn.Linear(d_model, d_model)
        self.pol_w2 = nn.Linear(d_model, POLICY_DIM)
        self.register_buffer("never_legal", _load_never_legal())

        # independent value head
        self.val_ln = nn.LayerNorm(d_model)
        self.val_w1 = nn.Linear(d_model, d_model // 2)
        self.val_w2 = nn.Linear(d_model // 2, 3)

    def forward(self, tokens):
        e    = self.emb(tokens)
        us   = e[:, :32].sum(dim=1)
        them = e[:, 32:].sum(dim=1)
        x    = torch.cat([us, -them], dim=1)
        for blk in self.trunk:
            x = blk(x)
        pol = self.pol_w2(F.gelu(self.pol_w1(self.pol_ln(x))))
        pol = pol.masked_fill(self.never_legal, float('-inf'))
        val = self.val_w2(F.gelu(self.val_w1(self.val_ln(x))))
        return pol, val


# ---------------------------------------------------------------------------
# Best-practices model: RMSNorm + SwiGLU + no-bias linears + zero-init down
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps   = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


class SwiGLUBlock(nn.Module):
    def __init__(self, d, hidden, dropout=0.1):
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


class NNUEMlpBP(nn.Module):
    """Best-practices variant: RMSNorm, SwiGLU, no-bias, 3/2x hidden, dropout."""
    def __init__(self, vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, d_model=1024,
                 n_blocks=10, dropout=0.1):
        super().__init__()
        assert d_model == emb_dim * 2, f"d_model must be 2 * emb_dim"
        hidden = d_model * 3 // 2
        self.emb   = nn.Embedding(vocab_size, emb_dim)
        self.trunk = nn.ModuleList([SwiGLUBlock(d_model, hidden, dropout) for _ in range(n_blocks)])

        # independent policy head
        self.pol_norm = RMSNorm(d_model)
        self.pol_w1   = nn.Linear(d_model, d_model, bias=False)
        self.pol_w2   = nn.Linear(d_model, POLICY_DIM, bias=False)
        self.register_buffer("never_legal", _load_never_legal())

        # independent value head
        self.val_norm = RMSNorm(d_model)
        self.val_w1   = nn.Linear(d_model, d_model // 2, bias=False)
        self.val_w2   = nn.Linear(d_model // 2, 3, bias=False)

    def forward(self, tokens):
        e    = self.emb(tokens)
        us   = e[:, :32].sum(dim=1)
        them = e[:, 32:].sum(dim=1)
        x    = torch.cat([us, -them], dim=1)
        for blk in self.trunk:
            x = blk(x)
        pol = self.pol_w2(F.silu(self.pol_w1(self.pol_norm(x))))
        pol = pol.masked_fill(self.never_legal, float('-inf'))
        val = self.val_w2(F.silu(self.val_w1(self.val_norm(x))))
        return pol, val


# ---------------------------------------------------------------------------
# Hybrid: Conv frontend -> pos enc -> MHA w/ global tokens -> SwiGLU trunk
#
# Encoder (128-d):
#   Embedding(21, 128) -> 2x ConvBlock -> pos_enc -> cat 8 global tokens
#   -> 2x MHA block -> extract global tokens [B,8,128] -> reshape [B,1024]
# Trunk: 6x SwiGLU @ 1024
# Heads: independent policy (masked 4288) + value (3)
# ---------------------------------------------------------------------------

ENC_DIM     = 128      # embedding + conv dim
ENC_DIM_MHA = 256      # after cat([x, pos], dim=-1); N_GLOBAL * ENC_DIM_MHA = trunk_dim
ENC_VOCAB   = 21
N_GLOBAL    = 4        # 4 * 256 = 1024
ENC_HEADS   = 8
ENC_FF_DIM  = 1024     # 4x ENC_DIM_MHA


class ConvBlock2D(nn.Module):
    """Prenorm 2-layer conv block operating on [B, 64, d] via [B, d, 8, 8] reshape."""
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.c1 = nn.Conv2d(d, d, 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(d, d, 3, padding=1, bias=False)

    def forward(self, x):           # x: [B, 64, d]
        B, _, d = x.shape
        h = self.ln(x).reshape(B, 8, 8, d).permute(0, 3, 1, 2)   # [B, d, 8, 8]
        h = F.gelu(self.c1(h))
        h = self.c2(h)
        return x + h.permute(0, 2, 3, 1).reshape(B, 64, d)


class MHABlock(nn.Module):
    """Standard prenorm self-attention + FF block."""
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
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class CrossAttnBlock(nn.Module):
    """Prenorm cross-attention + FF block. Q=globals, KV=full sequence."""
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
        q = q + self.ff2(F.gelu(self.ff1(self.ln2(q))))
        return q


class ConvAttnMlp(nn.Module):
    """Conv encoder -> self-attn over 68 -> cross-attn globals->68 -> SwiGLU trunk.

    Embed(21,128) -> 2xConv2D(128) -> cat([x, pos_enc], dim=-1) -> [B,64,256]
    -> cat 4 global tokens -> [B,68,256]
    -> MHABlock self-attn -> CrossAttnBlock(Q=globals[4], KV=68) -> [B,4,256]
    -> reshape [B,1024] -> 6x SwiGLU trunk.
    """
    def __init__(self, n_trunk_blocks=6, trunk_dim=1024, dropout=0.02):
        super().__init__()
        assert N_GLOBAL * ENC_DIM_MHA == trunk_dim, \
            f"N_GLOBAL({N_GLOBAL}) * ENC_DIM_MHA({ENC_DIM_MHA}) must equal trunk_dim({trunk_dim})"

        self.emb            = nn.Embedding(ENC_VOCAB, ENC_DIM)
        self.conv           = nn.ModuleList([ConvBlock2D(ENC_DIM), ConvBlock2D(ENC_DIM)])
        self.pos            = nn.Embedding(SEQ_LEN, ENC_DIM)
        self.ln_x           = nn.LayerNorm(ENC_DIM)
        self.ln_pos         = nn.LayerNorm(ENC_DIM)
        self.global_tokens  = nn.Parameter(torch.randn(1, N_GLOBAL, ENC_DIM_MHA) * 0.02)
        self.self_attn      = MHABlock(ENC_DIM_MHA, ENC_HEADS, ENC_FF_DIM)
        self.cross_attn     = CrossAttnBlock(ENC_DIM_MHA, ENC_HEADS, ENC_FF_DIM)

        hidden = trunk_dim * 3 // 2
        self.trunk = nn.ModuleList([SwiGLUBlock(trunk_dim, hidden, dropout)
                                    for _ in range(n_trunk_blocks)])

        self.pol_norm = RMSNorm(trunk_dim)
        self.pol_w1   = nn.Linear(trunk_dim, trunk_dim, bias=False)
        self.pol_w2   = nn.Linear(trunk_dim, POLICY_DIM, bias=False)
        self.register_buffer("never_legal", _load_never_legal())

        self.val_norm = RMSNorm(trunk_dim)
        self.val_w1   = nn.Linear(trunk_dim, trunk_dim // 2, bias=False)
        self.val_w2   = nn.Linear(trunk_dim // 2, 3, bias=False)

    def forward(self, tokens):
        B = tokens.shape[0]
        x = self.emb(tokens)                                              # [B, 64, 128]
        for blk in self.conv:
            x = blk(x)
        pos = self.pos(torch.arange(SEQ_LEN, device=tokens.device))      # [64, 128]
        x = torch.cat([self.ln_x(x), self.ln_pos(pos).unsqueeze(0).expand(B, -1, -1)], dim=-1)  # [B, 64, 256]
        gt = self.global_tokens.expand(B, -1, -1)                        # [B, 4, 256]
        seq = torch.cat([x, gt], dim=1)                                  # [B, 68, 256]
        seq = self.self_attn(seq)                                         # [B, 68, 256]
        g = self.cross_attn(seq[:, -N_GLOBAL:], seq)                     # [B, 4, 256]
        x = g.reshape(B, -1)                                             # [B, 1024]
        for blk in self.trunk:
            x = blk(x)
        pol = self.pol_w2(F.silu(self.pol_w1(self.pol_norm(x))))
        pol = pol.masked_fill(self.never_legal, float('-inf'))
        val = self.val_w2(F.silu(self.val_w1(self.val_norm(x))))
        return pol, val


def build_conv_attn_model(n_trunk_blocks=6, dropout=0.02):
    m = ConvAttnMlp(n_trunk_blocks=n_trunk_blocks, dropout=dropout)
    n = sum(p.numel() for p in m.parameters())
    print(f"  params: {n:,}")
    return m


# ---------------------------------------------------------------------------
# Gated softmax pool: learns N attention maps over 64 squares, each extracts
# a 256-d summary. N=4 pools -> flatten to [B, 1024].
# ---------------------------------------------------------------------------

class GatedSoftmaxCompressor(nn.Module):
    """N learned softmax pools over 64 squares. Each pool: softmax-weighted
    sum of per-square values -> [B, N, channels] -> flatten [B, N*channels]."""
    def __init__(self, channels=256, n_pools=4):
        super().__init__()
        self.channels = channels
        self.n_pools  = n_pools
        self.norm  = nn.LayerNorm(channels)
        self.value = nn.Linear(channels, channels * n_pools, bias=False)
        self.gate  = nn.Linear(channels, n_pools, bias=False)

    def forward(self, x):               # x: [B, 64, channels] or [B, 8, 8, channels]
        B = x.shape[0]
        x = x.reshape(B, 64, self.channels)
        h = self.norm(x)
        values = self.value(h).reshape(B, 64, self.n_pools, self.channels)
        values = values.permute(0, 2, 1, 3)                 # [B, n_pools, 64, channels]
        gates  = self.gate(h).permute(0, 2, 1).softmax(-1)  # [B, n_pools, 64]
        pooled = (values * gates[:, :, :, None]).sum(dim=2)  # [B, n_pools, channels]
        return pooled.reshape(B, self.n_pools * self.channels)  # [B, 1024]


POOL_CONV_DIM = ENC_DIM_MHA   # 256
N_POOLS       = 4              # 4 * 256 = 1024


class ConvGatedPoolMlp(nn.Module):
    """Embed(21,128) -> cat(pos) -> [B,64,256] -> 2xConvBlock2D(256)
    -> GatedSoftmaxCompressor -> [B,1024] -> SwiGLU trunk -> heads."""
    def __init__(self, n_trunk_blocks=6, trunk_dim=1024, dropout=0.02):
        super().__init__()
        assert N_POOLS * POOL_CONV_DIM == trunk_dim

        self.emb      = nn.Embedding(ENC_VOCAB, ENC_DIM)
        self.pos      = nn.Embedding(SEQ_LEN, ENC_DIM)
        self.conv     = nn.ModuleList([ConvBlock2D(POOL_CONV_DIM), ConvBlock2D(POOL_CONV_DIM)])
        self.compress = GatedSoftmaxCompressor(POOL_CONV_DIM, N_POOLS)

        hidden = trunk_dim * 3 // 2
        self.trunk = nn.ModuleList([SwiGLUBlock(trunk_dim, hidden, dropout)
                                    for _ in range(n_trunk_blocks)])

        self.pol_norm = RMSNorm(trunk_dim)
        self.pol_w1   = nn.Linear(trunk_dim, trunk_dim, bias=False)
        self.pol_w2   = nn.Linear(trunk_dim, POLICY_DIM, bias=False)
        self.register_buffer("never_legal", _load_never_legal())

        self.val_norm = RMSNorm(trunk_dim)
        self.val_w1   = nn.Linear(trunk_dim, trunk_dim // 2, bias=False)
        self.val_w2   = nn.Linear(trunk_dim // 2, 3, bias=False)

    def forward(self, tokens):
        B = tokens.shape[0]
        x   = self.emb(tokens)                                               # [B, 64, 128]
        pos = self.pos(torch.arange(SEQ_LEN, device=tokens.device))         # [64, 128]
        x   = torch.cat([x, pos.unsqueeze(0).expand(B, -1, -1)], dim=-1)   # [B, 64, 256]
        for blk in self.conv:
            x = blk(x)                                                       # [B, 64, 256]
        x = self.compress(x)                                                 # [B, 1024]
        for blk in self.trunk:
            x = blk(x)
        pol = self.pol_w2(F.silu(self.pol_w1(self.pol_norm(x))))
        pol = pol.masked_fill(self.never_legal, float('-inf'))
        val = self.val_w2(F.silu(self.val_w1(self.val_norm(x))))
        return pol, val


def build_conv_pool_model(n_trunk_blocks=6, dropout=0.02):
    m = ConvGatedPoolMlp(n_trunk_blocks=n_trunk_blocks, dropout=dropout)
    n = sum(p.numel() for p in m.parameters())
    print(f"  params: {n:,}")
    return m


def build_model(n_blocks=10, d_model=1024):
    m = NNUEMlp(emb_dim=d_model // 2, d_model=d_model, n_blocks=n_blocks)
    n = sum(p.numel() for p in m.parameters())
    print(f"  params: {n:,}")
    return m


def build_model_bp(n_blocks=10, d_model=1024):
    m = NNUEMlpBP(emb_dim=d_model // 2, d_model=d_model, n_blocks=n_blocks)
    n = sum(p.numel() for p in m.parameters())
    print(f"  params: {n:,}")
    return m

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run",     action="store_true",
                   help="Build model + print param count only, no speed test")
    p.add_argument("--skip-trt",    action="store_true",
                   help="Skip ORT+TensorRT benchmark")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES,
                   metavar="B")
    p.add_argument("--blocks",        type=int, default=6,    help="Residual block count (NNUE models)")
    p.add_argument("--d-model",       type=int, default=1024, help="Trunk dimension (NNUE models)")
    p.add_argument("--trunk-blocks",  type=int,   default=6,    help="SwiGLU trunk blocks (hybrid model)")
    p.add_argument("--dropout",       type=float, default=0.02, help="Dropout for hybrid model")
    p.add_argument("--skip-hybrid",   action="store_true",      help="Skip Conv+MHA hybrid model")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_tokens(batch_size, vocab=VOCAB_SIZE):
    return np.random.randint(0, vocab, (batch_size, SEQ_LEN), dtype=np.int32)


def save_csv(csv_rows, path="nnue_mlp_speed_results.csv"):
    try:
        import pandas as pd
        pd.DataFrame(csv_rows).to_csv(path, index=False)
        print(f"  CSV saved -> {path}")
    except Exception as e:
        print(f"  [WARN] CSV save failed: {e}")

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


def make_pt_compiled_infer(model, device, mode="default"):
    model    = model.half().to(device).eval()
    backend  = "inductor"
    compiled = torch.compile(model, dynamic=True, backend=backend, mode=mode)
    try:
        with torch.no_grad():
            compiled(torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device))
        print(f"  torch.compile backend={backend} mode={mode}")
    except Exception as e:
        print(f"  inductor failed ({type(e).__name__}), falling back to cudagraphs")
        backend  = "cudagraphs"
        torch._dynamo.config.cache_size_limit = 32
        compiled = torch.compile(model, dynamic=False, backend=backend)

    def infer(tokens_np):
        with torch.no_grad():
            x = torch.from_numpy(tokens_np).long().to(device)
            p, v = compiled(x)
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    return infer, backend, mode


def make_pt_jit_infer(model, device):
    model  = model.half().to(device).eval()
    dummy  = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    traced = torch.jit.trace(model, dummy)
    traced = torch.jit.optimize_for_inference(traced)
    print("  jit.trace + optimize_for_inference done")

    def infer(tokens_np):
        with torch.no_grad():
            x = torch.from_numpy(tokens_np).long().to(device)
            p, v = traced(x)
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
            print(f"  opset {opset} failed ({type(e).__name__}), retrying with opset 16 ...")
            torch.onnx.export(model, dummy, onnx_path, **{**kwargs, "opset_version": 16})
        else:
            raise
    print("  ONNX export done.")


def make_trt_infer(onnx_path, cache_dir, max_bs=512):
    import hashlib
    import tensorrt  # registers TRT DLLs with Windows before ORT loads its TRT provider
    import onnxruntime as ort

    h = hashlib.sha256()
    with open(onnx_path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    prefix = f"nnue_mlp_{h.hexdigest()[:12]}"

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
    if cached:
        print(f"  [trt] loading cached engine: {engines[0]}")
    else:
        print(f"  [trt] no cached engine for {prefix!r}, compiling...")

    providers = [("TensorrtExecutionProvider", trt_opts)]
    t0        = time.perf_counter()
    session   = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    elapsed   = time.perf_counter() - t0
    active    = session.get_providers()[0]
    if active != "TensorrtExecutionProvider":
        raise RuntimeError(f"[trt] expected TensorrtExecutionProvider but got {active}")
    label = "cache hit" if cached else "compiled"
    print(f"  [trt] {label} in {elapsed:.1f}s  provider: TensorrtExecutionProvider")

    iname = session.get_inputs()[0].name

    def infer(tokens_np):
        return session.run(None, {iname: tokens_np.astype(np.int64)})

    return infer, session

# ---------------------------------------------------------------------------
# Speed test
# ---------------------------------------------------------------------------

def speed_test(label, infer_fn, batch_sizes, vocab=VOCAB_SIZE):
    print(f"\n  {'-'*56}")
    print(f"  {label}")
    print(f"  warmup={N_WARMUP}  iters={N_ITERS}")
    print(f"  {'batch':<7} {'latency_ms':>12} {'samples/s':>13}")
    print(f"  {'-'*7} {'-'*12} {'-'*13}")
    results = {}
    for b in batch_sizes:
        inp = random_tokens(b, vocab)
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

    print(f"PyTorch device : {device}")
    print(f"blocks={args.blocks}  d_model={args.d_model}  emb_dim={args.d_model // 2}  vocab={VOCAB_SIZE}")
    print(f"\n  {'-'*60}")
    print(f"  NNUE-MLP  ({args.blocks} blocks, d_model={args.d_model})")

    build_model(args.blocks, args.d_model)   # print param count

    csv_rows = []

    # PT eager
    if not args.dry_run:
        try:
            print(f"\n  Building PT eager ...")
            eager = make_pt_eager_infer(build_model(args.blocks, args.d_model), device)
            res   = speed_test("NNUE-MLP  PT eager [fp16]", eager, bs)
            for b, m in res.items():
                csv_rows.append({"backend": "PT eager", "batch_size": b, **m})
            del eager
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] PT eager: {e}")

    # ORT + TRT
    if not args.skip_trt:
        try:
            os.makedirs(TRT_CACHE, exist_ok=True)
            key       = f"nnue_mlp_b{args.blocks}_d{args.d_model}"
            onnx_path = os.path.join(TRT_CACHE, f"{key}.onnx")
            if not os.path.exists(onnx_path):
                export_to_onnx(build_model(args.blocks, args.d_model), device, onnx_path)
            else:
                print(f"  [TRT] ONNX cached: {onnx_path}")
            trt_infer, trt_sess = make_trt_infer(onnx_path, TRT_CACHE, max_bs=max(bs))
            res = speed_test("NNUE-MLP  ORT TRT [fp16]", trt_infer, bs)
            for b, m in res.items():
                csv_rows.append({"backend": "ORT TRT", "batch_size": b, **m})
            del trt_sess
            gc.collect()
        except Exception as e:
            print(f"  [ERROR] ORT TRT: {e}")

    # Best-practices model (RMSNorm + SwiGLU)----------------------------
    print(f"\n{'='*60}")
    hidden_bp = args.d_model * 3 // 2
    print(f"  NNUE-MLP-BP  ({args.blocks} blocks, d_model={args.d_model}, hidden={hidden_bp})")
    build_model_bp(args.blocks, args.d_model)

    if not args.dry_run:
        try:
            print(f"\n  Building BP PT eager ...")
            eager_bp = make_pt_eager_infer(build_model_bp(args.blocks, args.d_model), device)
            res = speed_test("NNUE-MLP-BP  PT eager [fp16]", eager_bp, bs)
            for b, m in res.items():
                csv_rows.append({"backend": "BP PT eager", "batch_size": b, **m})
            del eager_bp
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] BP PT eager: {e}")

    if not args.skip_trt:
        try:
            os.makedirs(TRT_CACHE, exist_ok=True)
            key_bp    = f"nnue_mlp_bp_b{args.blocks}_d{args.d_model}"
            onnx_bp   = os.path.join(TRT_CACHE, f"{key_bp}.onnx")
            if not os.path.exists(onnx_bp):
                export_to_onnx(build_model_bp(args.blocks, args.d_model), device, onnx_bp)
            else:
                print(f"  [TRT] ONNX cached: {onnx_bp}")
            trt_bp, trt_sess_bp = make_trt_infer(onnx_bp, TRT_CACHE, max_bs=max(bs))
            res = speed_test("NNUE-MLP-BP  ORT TRT [fp16]", trt_bp, bs)
            for b, m in res.items():
                csv_rows.append({"backend": "BP ORT TRT", "batch_size": b, **m})
            del trt_sess_bp
            gc.collect()
        except Exception as e:
            print(f"  [ERROR] BP ORT TRT: {e}")

    # Conv+MHA hybrid
    if not args.skip_hybrid:
        print(f"\n{'='*60}")
        print(f"  Conv+MHA Hybrid  (enc={ENC_DIM}->{ENC_DIM_MHA} via pos-cat, 2xConv+2xMHA, {N_GLOBAL} global tokens -> {N_GLOBAL*ENC_DIM_MHA}-d trunk, {args.trunk_blocks} SwiGLU blocks)")
        build_conv_attn_model(args.trunk_blocks, args.dropout)

        if not args.dry_run:
            try:
                print(f"\n  Building Hybrid PT eager ...")
                eager_h = make_pt_eager_infer(build_conv_attn_model(args.trunk_blocks, args.dropout), device)
                res = speed_test("Conv+MHA Hybrid  PT eager [fp16]", eager_h, bs, vocab=ENC_VOCAB)
                for b, mres in res.items():
                    csv_rows.append({"backend": "Hybrid PT eager", "batch_size": b, **mres})
                del eager_h
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [ERROR] Hybrid PT eager: {e}")

        if not args.skip_trt:
            try:
                os.makedirs(TRT_CACHE, exist_ok=True)
                key_h    = f"conv_attn_t{args.trunk_blocks}"
                onnx_h   = os.path.join(TRT_CACHE, f"{key_h}.onnx")
                if not os.path.exists(onnx_h):
                    export_to_onnx(build_conv_attn_model(args.trunk_blocks, args.dropout), device, onnx_h)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_h}")
                trt_h, trt_sess_h = make_trt_infer(onnx_h, TRT_CACHE, max_bs=max(bs))
                res = speed_test("Conv+MHA Hybrid  ORT TRT [fp16]", trt_h, bs, vocab=ENC_VOCAB)
                for b, mres in res.items():
                    csv_rows.append({"backend": "Hybrid ORT TRT", "batch_size": b, **mres})
                del trt_sess_h
                gc.collect()
            except Exception as e:
                print(f"  [ERROR] Hybrid ORT TRT: {e}")

    # Conv + gated softmax pool
    print(f"\n{'='*60}")
    print(f"  Conv+GatedPool  (enc={ENC_DIM}->{POOL_CONV_DIM} via pos-cat, 2xConv2D({POOL_CONV_DIM}), {N_POOLS} softmax pools -> {N_POOLS*POOL_CONV_DIM}-d trunk, {args.trunk_blocks} SwiGLU blocks)")
    build_conv_pool_model(args.trunk_blocks, args.dropout)

    if args.dry_run:
        print("\n  Dry run complete.")
        return

    try:
        print(f"\n  Building ConvPool PT eager ...")
        eager_p = make_pt_eager_infer(build_conv_pool_model(args.trunk_blocks, args.dropout), device)
        res = speed_test("Conv+GatedPool  PT eager [fp16]", eager_p, bs, vocab=ENC_VOCAB)
        for b, mres in res.items():
            csv_rows.append({"backend": "ConvPool PT eager", "batch_size": b, **mres})
        del eager_p
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [ERROR] ConvPool PT eager: {e}")

    if csv_rows:
        save_csv(csv_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
