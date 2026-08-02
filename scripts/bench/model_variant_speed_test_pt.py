#!/usr/bin/env python3
"""model_variant_speed_test_pt.py

Conv-backbone speed tests for chess policy+value models.

Models:
  18m-precond-smartgate-xc0h-K6-6c4t -- 6 conv preconditioner blocks, 4 transformer
                                        blocks, pos concatenated (CF=256 -> D=512).
                                        Keeps its existing LayerNorm; untouched here.
  conv-pure                          -- 16x RMS conv block (256) NCHW -> trunk RMSNorm
                                        -> conv-pure WDL / SmartGate / from-to heads
  conv-globalizer-2c6g               -- 2 RMS conv blocks + 6 globalizer blocks (384):
                                        368ch local 3x3 + 16ch 1x1 global branch through
                                        a 1024->512->1024 SwiGLU, concat back to 384
  conv-signed-attn-2c10a             -- 2 RMS conv blocks + 10 signed-attention blocks (256):
                                        tanh-bounded L1-normalized signed board attention
                                        instead of softmax

The three conv models share the identical conv-pure heads and input stem, and all
use RMSNorm. The 18m preconditioned model deliberately keeps LayerNorm.

Flags:
  --dry-run              Build models, print param counts + shape checks, skip speed test
  --skip-trt             Skip ORT+TensorRT
  --skip-convpure        Skip all three conv models
  --batch-sizes    Space-separated list (default: 1 2 4 8 16 32 64 128 256 512)
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
from chessbot.model import (
    build_pt_precond_smartgate, build_pt_conv_pure,
    build_pt_conv_globalizer, build_pt_conv_signed_attn,
)

SEQ_LEN    = 64
ENC_VOCAB  = 21
POLICY_DIM = 4288

N_WARMUP = 50
N_ITERS  = 100

TRT_CACHE    = os.path.join(os.path.expanduser("~"), ".cache", "conv_chess_trt", "speed_test_cache")
CSV_RESULTS  = os.path.join(TRT_CACHE, "speed_results.csv")

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

XC0H_CFG = dict(conv_filters=256, num_heads=8, dropout=0.05, pre_blocks=4, xc0h_K=6)

XC0H_6C4T_CFG = dict(
    conv_filters=256, num_heads=8, dropout=0.05,
    pre_blocks=6, tx_blocks=4, xc0h_K=6,
)

# flat width of board.history_tokens(K): K*64 slim tokens + K rep flags
# + castling + stm + hmc. Matches random_xc0h_tokens.
XC0H_IN_LEN = 6 * 64 + 6 + 3

CONV_PURE_CFG = dict(
    conv_filters=256, num_blocks=16, dropout=0.03,
)

# 2 standard RMS conv blocks + 6 backbone blocks, d_model 384
CONV_GLOBALIZER_CFG = dict(
    conv_filters=384, num_blocks=6, stem_blocks=2, dropout=0.03,
)

# 2 standard RMS conv blocks + 10 backbone blocks, d_model 256
CONV_SIGNED_ATTN_CFG = dict(
    conv_filters=256, num_blocks=10, stem_blocks=2, dropout=0.03,
)

# (label, builder, cfg, onnx stem) -- the three conv models are identical in
# stem, heads and IO, so they run through one code path.
CONV_MODELS = [
    ("conv-pure",            build_pt_conv_pure,        CONV_PURE_CFG,         "conv_pure"),
    ("conv-globalizer-2c6g",  build_pt_conv_globalizer,  CONV_GLOBALIZER_CFG,   "conv_globalizer_2c6g"),
    ("conv-signed-attn-2c10a", build_pt_conv_signed_attn, CONV_SIGNED_ATTN_CFG, "conv_signed_attn_2c10a"),
]

# rough targets from the design spec; reported alongside the real count
EXPECTED_PARAMS = {
    "conv-globalizer-2c6g": 34_297_000,
    "conv-signed-attn-2c10a": 19_275_000,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run",       action="store_true")
    p.add_argument("--clear-cache",   nargs="?", const="soft", default=None,
                   metavar="hard", help="Clear engine/ONNX cache (bare) or everything incl. CSV (hard)")
    p.add_argument("--skip-trt",      action="store_true")
    p.add_argument("--skip-convpure", action="store_true", default=False)
    p.add_argument("--batch-sizes",   nargs="+", type=int, default=DEFAULT_BATCH_SIZES, metavar="B")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_tokens(batch_size, vocab=ENC_VOCAB):
    return np.random.randint(0, vocab, (batch_size, SEQ_LEN), dtype=np.int32)


def random_lc0_features(batch_size):
    """(B, 112, 8, 8) uint8, matching board_to_lc0_features's raw output (rule50
    unscaled). Preprocessing (float cast + rule50/99) happens at the model boundary,
    same as production (see make_pt_infer's is_lc0 branch)."""
    return np.random.randint(0, 2, (batch_size, 112, 8, 8), dtype=np.uint8)


def lc0_trt_preprocess(enc_np):
    """Raw uint8 lc0 planes -> float32, rule50 (plane 109) /99, cast to float16 --
    the ONNX graph's input is already-preprocessed float16, matching
    train_pytorch.export_ts_to_onnx's lc0 dummy convention."""
    x = enc_np.astype(np.float32)
    x[:, 109] /= 99.0
    return x.astype(np.float16)


def random_xc0h_tokens(batch_size, K):
    """(B, K*64+K+3) int32, matching board.history_tokens(K)'s flat layout:
    K frames of 64 slim tokens (vocab 0..14) + K repetition flags (0/1)
    + castling (0..15) + stm (0/1) + hmc (0..99ish)."""
    n = K * 64 + K + 3
    out = np.zeros((batch_size, n), dtype=np.int32)
    out[:, :K * 64]          = np.random.randint(0, 15, (batch_size, K * 64))
    out[:, K * 64:K * 64 + K] = np.random.randint(0, 2, (batch_size, K))
    out[:, K * 64 + K]        = np.random.randint(0, 16, batch_size)
    out[:, K * 64 + K + 1]    = np.random.randint(0, 2, batch_size)
    out[:, K * 64 + K + 2]    = np.random.randint(0, 100, batch_size)
    return out


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

def make_pt_eager_infer(model, device, is_lc0=False):
    model = model.half().to(device).eval()

    def infer(enc_np):
        with torch.no_grad():
            if is_lc0:
                # raw uint8 lc0 planes -> float, rule50 (plane 109) /99 at the
                # model-input boundary, matching train_pytorch.make_pt_infer
                x = torch.from_numpy(enc_np).float().to(device)
                x[:, 109] /= 99.0
                x = x.half()
            else:
                x = torch.from_numpy(enc_np).long().to(device)
            p, v = model(x)
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    return infer


POLICY_1858 = 1858


def verify_outputs(model, device, label, batch=4, input_fn=None):
    """Forward a small batch and check the head shapes.

    Returns (ok, policy_np, value_np, enc_np) so the caller can reuse the same
    inputs for the PT-vs-export comparison.
    """
    input_fn = input_fn or random_tokens
    enc_np = input_fn(batch)
    m = model.half().to(device).eval()
    with torch.no_grad():
        p, v = m(torch.from_numpy(enc_np).long().to(device))
    p_np = p.float().cpu().numpy()
    v_np = v.float().cpu().numpy()

    ok = True
    if p_np.shape != (batch, POLICY_1858):
        print(f"  [SHAPE] {label}: policy {p_np.shape}, expected {(batch, POLICY_1858)}")
        ok = False
    if v_np.shape != (batch, 3):
        print(f"  [SHAPE] {label}: value {v_np.shape}, expected {(batch, 3)}")
        ok = False
    if not (np.isfinite(p_np).all() and np.isfinite(v_np).all()):
        print(f"  [NAN] {label}: non-finite outputs")
        ok = False
    if ok:
        print(f"  shapes ok: policy {p_np.shape}  value {v_np.shape}  (finite)")
    return ok, p_np, v_np, enc_np


def compare_pt_vs_export(pt_policy, pt_value, infer_fn, enc_np, label):
    """Max/mean absolute difference between the eager and exported paths."""
    try:
        ep, ev = infer_fn(enc_np)
    except Exception as e:
        print(f"  [ERROR] {label} export comparison: {e}")
        return None
    dp = np.abs(ep - pt_policy)
    dv = np.abs(ev - pt_value)
    print(f"  PT vs export  policy: max {dp.max():.4e}  mean {dp.mean():.4e}")
    print(f"                value : max {dv.max():.4e}  mean {dv.mean():.4e}")
    return {"policy_max": float(dp.max()), "policy_mean": float(dp.mean()),
            "value_max": float(dv.max()), "value_mean": float(dv.mean())}


def export_to_onnx(model, device, onnx_path, dummy=None):
    import warnings
    import onnx
    from onnx import shape_inference as onnx_si
    print(f"  Exporting to ONNX (opset 18) ...")
    model = model.half().to(device).eval()
    if dummy is None:
        dummy = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    else:
        dummy = dummy.to(device)
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


def make_trt_infer(onnx_path, cache_dir, max_bs=512, input_shape="64", preprocess_fn=None):
    """input_shape: profile shape suffix after the batch dim, e.g. "64",
    "112x8x8", or "393" (xc0h-K6's flat width). preprocess_fn(enc_np) -> np.ndarray
    ready for session.run; defaults to a plain int64 cast (xc0/xc0h token inputs).
    lc0 needs float32->rule50/99->float16, matching train_pytorch.make_pt_infer."""
    import hashlib
    import tensorrt
    import onnxruntime as ort

    if preprocess_fn is None:
        preprocess_fn = lambda x: x.astype(np.int64)

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
        "trt_profile_min_shapes":   f"enc_in:1x{input_shape}",
        "trt_profile_opt_shapes":   f"enc_in:{max_bs}x{input_shape}",
        "trt_profile_max_shapes":   f"enc_in:{max_bs}x{input_shape}",
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

    def infer(enc_np):
        return session.run(None, {"enc_in": preprocess_fn(enc_np)})

    return infer, session


# ---------------------------------------------------------------------------
# Speed test
# ---------------------------------------------------------------------------

def speed_test(label, infer_fn, batch_sizes, input_fn=random_tokens):
    print(f"\n  {'-'*56}")
    print(f"  {label}")
    print(f"  warmup={N_WARMUP}  iters={N_ITERS}")
    print(f"  {'batch':<7} {'latency_ms':>12} {'samples/s':>13}")
    print(f"  {'-'*7} {'-'*12} {'-'*13}")
    results = {}
    for b in batch_sizes:
        inp = input_fn(b)
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
    # sort by bs=256 throughput, fastest first; fall back to the largest batch
    # actually run when 256 isn't in the sweep. Rows missing it sink to the end.
    sort_bs = 256 if 256 in TABLE_BS else TABLE_BS[-1]

    def sort_key(item):
        val = item[1].get(sort_bs, {}).get("throughput")
        return (val is None, -(val or 0))

    for label, res in sorted(all_results.items(), key=sort_key):
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

    def run_or_cached(lbl, infer_fn, params, input_fn=random_tokens):
        hit = csv_data.get((lbl, params))
        if hit and all(b in hit for b in bs):
            return display_cached_result(lbl + " [fp16]", hit, bs)
        result = speed_test(lbl + " [fp16]", infer_fn, bs, input_fn=input_fn)
        save_results_csv(lbl, params, result)
        return result

    # 18m-precond-smartgate-xc0h-K6-6c4t
    print(f"\n{'='*60}")
    print(f"  18m-precond-smartgate-xc0h-K6-6c4t  (same stem as xc0h-K6;"
          f" 6 conv preconditioner blocks, 4 transformer blocks (unchanged),"
          f" pos concatenated so CF=256 -> D=512, same as prod)")
    model_6c4t  = build_pt_precond_smartgate(XC0H_6C4T_CFG)
    params_6c4t = sum(p.numel() for p in model_6c4t.parameters())

    if not args.dry_run:
        try:
            print(f"\n  Building 18m-precond-smartgate-xc0h-K6-6c4t PT eager ...")
            eager_6c4t = make_pt_eager_infer(model_6c4t, device)
            lbl = "18m-precond-smartgate-xc0h-K6-6c4t  PT eager"
            all_results[lbl] = run_or_cached(
                lbl, eager_6c4t, params_6c4t,
                input_fn=lambda b: random_xc0h_tokens(b, 6))
            del eager_6c4t; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] 18m-precond-smartgate-xc0h-K6-6c4t PT eager: {e}")

        if not args.skip_trt:
            try:
                lbl        = "18m-precond-smartgate-xc0h-K6-6c4t  ORT TRT"
                onnx_6c4t  = os.path.join(TRT_CACHE, f"precond_xc0h_k6_6c4t_{params_6c4t}.onnx")
                if not os.path.exists(onnx_6c4t):
                    dummy = torch.zeros(
                        1, XC0H_IN_LEN, dtype=torch.long, device=device)
                    export_to_onnx(build_pt_precond_smartgate(XC0H_6C4T_CFG), device, onnx_6c4t, dummy=dummy)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_6c4t}")
                trt_6c4t, trt_sess_6c4t = make_trt_infer(
                    onnx_6c4t, TRT_CACHE, max_bs=max(bs),
                    input_shape=str(XC0H_IN_LEN))
                all_results[lbl] = run_or_cached(
                    lbl, trt_6c4t, params_6c4t,
                    input_fn=lambda b: random_xc0h_tokens(b, 6))
                del trt_sess_6c4t; gc.collect()
            except Exception as e:
                print(f"  [ERROR] 18m-precond-smartgate-xc0h-K6-6c4t TRT: {e}")

    # conv family: conv-pure, conv-globalizer-2c6g, conv-signed-attn-2c10a.
    # Same stem, same heads, same IO, so one loop covers all three.
    if not args.skip_convpure:
        for label, builder, cfg, stem in CONV_MODELS:
            print(f"\n{'='*60}")
            print(f"  {label}")
            model = builder(cfg)
            params = sum(q.numel() for q in model.parameters())
            trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
            print(f"  params: {params:,}  (trainable {trainable:,})")
            exp = EXPECTED_PARAMS.get(label)
            if exp:
                print(f"  spec estimate: ~{exp:,}  "
                      f"(delta {params - exp:+,}, {100.0 * (params - exp) / exp:+.2f}%)")

            ok, pt_p, pt_v, enc_np = verify_outputs(model, device, label)
            if not ok:
                print(f"  [SKIP] {label}: output verification failed")
                del model; gc.collect(); torch.cuda.empty_cache()
                continue

            if args.dry_run:
                del model; gc.collect(); torch.cuda.empty_cache()
                continue

            try:
                print(f"\n  Building {label} PT eager ...")
                eager = make_pt_eager_infer(model, device)
                lbl = f"{label}  PT eager"
                all_results[lbl] = run_or_cached(lbl, eager, params)
                del eager
            except Exception as e:
                print(f"  [ERROR] {label} PT eager: {e}")
            del model; gc.collect(); torch.cuda.empty_cache()

            if args.skip_trt:
                continue
            try:
                lbl = f"{label}  ORT TRT"
                onnx_path = os.path.join(TRT_CACHE, f"{stem}_{params}.onnx")
                if not os.path.exists(onnx_path):
                    export_to_onnx(builder(cfg), device, onnx_path)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_path}")
                trt_fn, trt_sess = make_trt_infer(onnx_path, TRT_CACHE, max_bs=max(bs))
                compare_pt_vs_export(pt_p, pt_v, trt_fn, enc_np, label)
                all_results[lbl] = run_or_cached(lbl, trt_fn, params)
                del trt_sess; gc.collect()
            except Exception as e:
                print(f"  [ERROR] {label} TRT: {e}")

    if args.dry_run:
        print("\n  Dry run complete.")
        return

    print_summary_table(all_results, bs)
    print("\nDone.")


if __name__ == "__main__":
    main()
