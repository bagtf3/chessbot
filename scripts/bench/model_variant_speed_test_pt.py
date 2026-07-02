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
  full-mha-smartgate        -- no conv; emb(256)+pos(256)->512 -> append 8 accum tokens
                              -> 8x full MHA transformer blocks -> same WDL/SmartGate/
                                 from-to-policy heads as 16m-precond-smartgate

Flags:
  --dry-run              Build models, print param counts, skip speed test
  --skip-trt             Skip ORT+TensorRT
  --skip-hybrid          Skip Conv+MHA+GEMM+SmartGate model
  --batch-sizes    Space-separated list (default: 1 2 4 8 16 32 64 128 256 512)
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
from chessbot.model import (build_pt_precond_smartgate, build_pt_conv_shallow_mha,
                            build_pt_conv_pure)

SEQ_LEN    = 64
ENC_VOCAB  = 21
POLICY_DIM = 4288

N_WARMUP = 50
N_ITERS  = 100

TRT_CACHE    = os.path.join(os.path.expanduser("~"), ".cache", "conv_chess_trt", "speed_test_cache")
CSV_RESULTS  = os.path.join(TRT_CACHE, "speed_results.csv")

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

PRECOND_CFG = dict(
    precond_conformer=True,
    conv_filters=256, num_heads=8, dropout=0.05,
    pre_blocks=4, mha_blocks=4,
)

CONV_SHALLOW_CFG = dict(
    conv_filters=128, num_heads=8, dropout=0.03, mha_blocks=8,
)

CONV_PURE_CFG = dict(
    conv_filters=256, num_blocks=12, dropout=0.03,
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


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run",       action="store_true")
    p.add_argument("--clear-cache",   nargs="?", const="soft", default=None,
                   metavar="hard", help="Clear engine/ONNX cache (bare) or everything incl. CSV (hard)")
    p.add_argument("--skip-trt",      action="store_true")
    p.add_argument("--skip-hybrid",   action="store_true")
    p.add_argument("--batch-sizes",   nargs="+", type=int, default=DEFAULT_BATCH_SIZES, metavar="B")
    p.add_argument("--trunk-blocks",      type=int,   default=6,    help="GEMM trunk blocks for Conv+MHA")
    p.add_argument("--dropout",           type=float, default=0.02)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_tokens(batch_size, vocab=ENC_VOCAB):
    return np.random.randint(0, vocab, (batch_size, SEQ_LEN), dtype=np.int32)


def load_results_csv():
    """Returns {(label, params): {batch_size: {latency_ms, throughput}}}"""
    import csv
    if not os.path.exists(CSV_RESULTS):
        return {}
    data = {}
    with open(CSV_RESULTS, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["label"], int(row["params"]))
            b   = int(row["batch_size"])
            data.setdefault(key, {})[b] = {
                "latency_ms": float(row["latency_ms"]),
                "throughput": float(row["throughput"]),
            }
    return data


def save_results_csv(label, params, results):
    """Append rows for (label, params) to CSV; writes header if file is new."""
    import csv
    write_header = not os.path.exists(CSV_RESULTS)
    with open(CSV_RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "params", "batch_size", "latency_ms", "throughput"])
        if write_header:
            w.writeheader()
        for b, v in sorted(results.items()):
            w.writerow({"label": label, "params": params, "batch_size": b,
                        "latency_ms": v["latency_ms"], "throughput": v["throughput"]})


def display_cached_result(label, cached, batch_sizes):
    """Print speed_test-style table from cached data; return result dict."""
    print(f"\n  {'-'*56}")
    print(f"  {label}  (cached)")
    print(f"  {'batch':<7} {'latency_ms':>12} {'samples/s':>13}")
    print(f"  {'-'*7} {'-'*12} {'-'*13}")
    out = {}
    for b in batch_sizes:
        if b in cached:
            v = cached[b]
            print(f"  {b:<7} {v['latency_ms']:>12.2f} {v['throughput']:>13.0f}")
            out[b] = v
    return out


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


def export_to_onnx(model, device, onnx_path):
    import warnings
    import onnx
    from onnx import shape_inference as onnx_si
    print(f"  Exporting to ONNX (opset 18) ...")
    model = model.half().to(device).eval()
    dummy = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    with torch.no_grad():
        model(dummy)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Constant folding in symbolic shape inference')
        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=18,
            input_names=["enc_in"],
            output_names=["policy_logits", "value_out"],
            dynamic_axes={"enc_in": {0: "batch"}, "policy_logits": {0: "batch"}, "value_out": {0: "batch"}},
            do_constant_folding=False,
        )
    proto = onnx.load(onnx_path)
    proto = onnx_si.infer_shapes(proto)
    onnx.save(proto, onnx_path)
    sz_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ONNX export done ({sz_mb:.1f} MB)")


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
    prefix = f"bench_{h.hexdigest()[:12]}"

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
        "trt_profile_min_shapes":   "enc_in:1x64",
        "trt_profile_opt_shapes":   f"enc_in:{max_bs}x64",
        "trt_profile_max_shapes":   f"enc_in:{max_bs}x64",
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

    def infer(tokens_np):
        return session.run(None, {"enc_in": tokens_np.astype(np.int64)})

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

def print_summary_table(all_results, batch_sizes):
    TABLE_BS = [b for b in [64, 128, 256, 512] if b in batch_sizes]
    if not TABLE_BS or not all_results:
        return
    col_w = 12
    label_w = max(len(k) for k in all_results) + 2
    header = f"  {'model/backend':<{label_w}}" + "".join(f"{'bs='+str(b):>{col_w}}" for b in TABLE_BS)
    print(f"\n{'='*60}")
    print(f"  samples/s summary")
    print(f"  {'-'*(label_w + col_w * len(TABLE_BS))}")
    print(header)
    print(f"  {'-'*(label_w + col_w * len(TABLE_BS))}")
    for label, res in all_results.items():
        row = f"  {label:<{label_w}}"
        for b in TABLE_BS:
            val = res.get(b, {}).get("throughput")
            row += f"{int(val):>{col_w},}" if val is not None else f"{'--':>{col_w}}"
        print(row)
    print(f"  {'-'*(label_w + col_w * len(TABLE_BS))}")


def main():
    import shutil
    args   = parse_args()
    bs     = args.batch_sizes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.clear_cache:
        if os.path.exists(TRT_CACHE):
            if args.clear_cache == "hard":
                shutil.rmtree(TRT_CACHE)
            else:
                for fname in os.listdir(TRT_CACHE):
                    if fname != os.path.basename(CSV_RESULTS):
                        fpath = os.path.join(TRT_CACHE, fname)
                        if os.path.isfile(fpath):
                            os.remove(fpath)
            print(f"Cleared cache ({args.clear_cache}): {TRT_CACHE}")
    os.makedirs(TRT_CACHE, exist_ok=True)

    print(f"PyTorch device: {device}")
    print(f"TRT cache:      {TRT_CACHE}")

    csv_data    = load_results_csv()
    all_results = {}

    def run_or_cached(lbl, infer_fn, params):
        hit = csv_data.get((lbl, params))
        if hit and all(b in hit for b in bs):
            return display_cached_result(lbl + " [fp16]", hit, bs)
        result = speed_test(lbl + " [fp16]", infer_fn, bs)
        save_results_csv(lbl, params, result)
        return result

    # Conv+MHA+GEMM+SmartGate
    if not args.skip_hybrid:
        print(f"\n{'='*60}")
        print(f"  Conv+MHA+GEMM+SmartGate  (2xConvBlock2D({ENC_DIM}) -> pos-cat -> [{ENC_DIM_MHA}]"
              f" -> append 8 accum tokens -> 1x self-attn(72)"
              f" -> 1x cross-attn(q=accum[8], kv=72) -> split first6/last2 -> [1536]/[512]"
              f" -> {args.trunk_blocks}x alt GELU/SwiGLU, WDL@4,"
              f" quality+logsigmoid(gate) 1858->4288)")
        model_h  = build_conv_attn_model(args.trunk_blocks, args.dropout)
        params_h = sum(p.numel() for p in model_h.parameters())

        if not args.dry_run:
            try:
                print(f"\n  Building Conv+MHA+GEMM+SmartGate PT eager ...")
                eager_h = make_pt_eager_infer(model_h, device)
                lbl = "Conv+MHA+GEMM+SG  PT eager"
                all_results[lbl] = run_or_cached(lbl, eager_h, params_h)
                del eager_h; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [ERROR] Conv+MHA+GEMM+SmartGate PT eager: {e}")

            if not args.skip_trt:
                try:
                    lbl    = "Conv+MHA+GEMM+SG  ORT TRT"
                    onnx_h = os.path.join(TRT_CACHE, f"conv_mha_t{args.trunk_blocks}_{params_h}.onnx")
                    if not os.path.exists(onnx_h):
                        export_to_onnx(build_conv_attn_model(args.trunk_blocks, args.dropout), device, onnx_h)
                    else:
                        print(f"  [TRT] ONNX cached: {onnx_h}")
                    trt_h, trt_sess_h = make_trt_infer(onnx_h, TRT_CACHE, max_bs=max(bs))
                    all_results[lbl] = run_or_cached(lbl, trt_h, params_h)
                    del trt_sess_h; gc.collect()
                except Exception as e:
                    print(f"  [ERROR] Conv+MHA+GEMM+SmartGate TRT: {e}")

    # 16m-precond-smartgate
    print(f"\n{'='*60}")
    print(f"  16m-precond-smartgate  (bilinear subblock underpromo: dots_sub + 2x Linear(PDH,3))")
    model_sg  = build_pt_precond_smartgate(PRECOND_CFG)
    params_sg = sum(p.numel() for p in model_sg.parameters())

    if not args.dry_run:
        try:
            print(f"\n  Building 16m-precond-smartgate PT eager ...")
            eager_sg = make_pt_eager_infer(model_sg, device)
            lbl = "16m-precond-smartgate  PT eager"
            all_results[lbl] = run_or_cached(lbl, eager_sg, params_sg)
            del eager_sg; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] 16m-precond-smartgate PT eager: {e}")

    # conv-shallow-mha
    print(f"\n{'='*60}")
    print(f"  conv-shallow-mha  (2xConv(128) -> cat LN(conv)+pos(128) -> 256-d"
          f" -> 8 global accum tokens -> {CONV_SHALLOW_CFG['mha_blocks']}x MHA(ff=1024)"
          f" -> trunk_ln -> WDL/SmartGate/from-to-policy heads)")
    model_cs  = build_pt_conv_shallow_mha(CONV_SHALLOW_CFG)
    params_cs = sum(p.numel() for p in model_cs.parameters())

    if not args.dry_run:
        try:
            print(f"\n  Building conv-shallow-mha PT eager ...")
            eager_cs = make_pt_eager_infer(model_cs, device)
            lbl = "conv-shallow-mha  PT eager"
            all_results[lbl] = run_or_cached(lbl, eager_cs, params_cs)
            del eager_cs; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] conv-shallow-mha PT eager: {e}")

        if not args.skip_trt:
            try:
                lbl     = "conv-shallow-mha  ORT TRT"
                onnx_cs = os.path.join(TRT_CACHE, f"conv_shallow_mha_{params_cs}.onnx")
                if not os.path.exists(onnx_cs):
                    export_to_onnx(build_pt_conv_shallow_mha(CONV_SHALLOW_CFG), device, onnx_cs)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_cs}")
                trt_cs, trt_sess_cs = make_trt_infer(onnx_cs, TRT_CACHE, max_bs=max(bs))
                all_results[lbl] = run_or_cached(lbl, trt_cs, params_cs)
                del trt_sess_cs; gc.collect()
            except Exception as e:
                print(f"  [ERROR] conv-shallow-mha TRT: {e}")

    # conv-pure
    print(f"\n{'='*60}")
    print(f"  conv-pure  (12x ConvBlock(128) NCHW -> trunk_ln"
          f" -> WDL conv(8)/reshape/GELU, SmartGate pool/16-d, policy 2xLinear->256->MHA(128))")
    model_cp  = build_pt_conv_pure(CONV_PURE_CFG)
    params_cp = sum(p.numel() for p in model_cp.parameters())

    if not args.dry_run:
        try:
            print(f"\n  Building conv-pure PT eager ...")
            eager_cp = make_pt_eager_infer(model_cp, device)
            lbl = "conv-pure  PT eager"
            all_results[lbl] = run_or_cached(lbl, eager_cp, params_cp)
            del eager_cp; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] conv-pure PT eager: {e}")

        if not args.skip_trt:
            try:
                lbl     = "conv-pure  ORT TRT"
                onnx_cp = os.path.join(TRT_CACHE, f"conv_pure_{params_cp}.onnx")
                if not os.path.exists(onnx_cp):
                    export_to_onnx(build_pt_conv_pure(CONV_PURE_CFG), device, onnx_cp)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_cp}")
                trt_cp, trt_sess_cp = make_trt_infer(onnx_cp, TRT_CACHE, max_bs=max(bs))
                all_results[lbl] = run_or_cached(lbl, trt_cp, params_cp)
                del trt_sess_cp; gc.collect()
            except Exception as e:
                print(f"  [ERROR] conv-pure TRT: {e}")

    if args.dry_run:
        print("\n  Dry run complete.")
        return

    print_summary_table(all_results, bs)
    print("\nDone.")


if __name__ == "__main__":
    main()
