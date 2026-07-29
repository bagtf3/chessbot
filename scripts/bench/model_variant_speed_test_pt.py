#!/usr/bin/env python3
"""model_variant_speed_test_pt.py

Conv-backbone speed tests for chess policy+value models.

Models:
  16m-precond-smartgate-xc0h-K6      -- prod: 4x Conv(256) preconditioner -> concat pos(256)
                                         -> 512-d -> 8 global accumulators -> 4x transformer
                                         blocks -> shared trunk_ln -> WDL(0) + SmartGate(1)
                                         + from/to policy(2-7)
  18m-precond-smartgate-xc0h-K6-6c4t -- same stem; 6 conv preconditioner blocks, 4
                                        transformer blocks (unchanged), pos concatenated
                                        (CF=256 -> D=512, same as prod)
  conv-pure                          -- 16x ConvBlock(256) NCHW -> trunk_ln -> WDL conv(8)
                                         -> SmartGate conv(16,3x3)->1024->512 SwiGLU/RMSNorm
                                         -> policy 2xLinear->256->MHA(256)

Flags:
  --dry-run              Build models, print param counts, skip speed test
  --skip-trt             Skip ORT+TensorRT
  --skip-convpure        Skip conv-pure model
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
from chessbot.model import build_pt_precond_smartgate, build_pt_conv_pure

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

CONV_PURE_CFG = dict(
    conv_filters=256, num_blocks=16, dropout=0.03,
)


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
    "112x8x8", or "391" (xc0h-K6's flat width). preprocess_fn(enc_np) -> np.ndarray
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

    def run_or_cached(lbl, infer_fn, params, input_fn=random_tokens):
        hit = csv_data.get((lbl, params))
        if hit and all(b in hit for b in bs):
            return display_cached_result(lbl + " [fp16]", hit, bs)
        result = speed_test(lbl + " [fp16]", infer_fn, bs, input_fn=input_fn)
        save_results_csv(lbl, params, result)
        return result

    # 16m-precond-smartgate-xc0h-K6
    print(f"\n{'='*60}")
    print(f"  16m-precond-smartgate-xc0h-K6  (compact history-token stem, K=6;"
          f" everything downstream identical to 16m-precond-smartgate)")
    model_xh  = build_pt_precond_smartgate(XC0H_CFG)
    params_xh = sum(p.numel() for p in model_xh.parameters())
    in_len_xh = 6 * 64 + 6 + 3

    if not args.dry_run:
        try:
            print(f"\n  Building 16m-precond-smartgate-xc0h-K6 PT eager ...")
            eager_xh = make_pt_eager_infer(model_xh, device)
            lbl = "16m-precond-smartgate-xc0h-K6  PT eager"
            all_results[lbl] = run_or_cached(
                lbl, eager_xh, params_xh,
                input_fn=lambda b: random_xc0h_tokens(b, 6))
            del eager_xh; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] 16m-precond-smartgate-xc0h-K6 PT eager: {e}")

        if not args.skip_trt:
            try:
                lbl     = "16m-precond-smartgate-xc0h-K6  ORT TRT"
                onnx_xh = os.path.join(TRT_CACHE, f"precond_xc0h_k6_{params_xh}.onnx")
                if not os.path.exists(onnx_xh):
                    dummy = torch.zeros(1, in_len_xh, dtype=torch.long, device=device)
                    export_to_onnx(build_pt_precond_smartgate(XC0H_CFG), device, onnx_xh, dummy=dummy)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_xh}")
                trt_xh, trt_sess_xh = make_trt_infer(
                    onnx_xh, TRT_CACHE, max_bs=max(bs), input_shape=str(in_len_xh))
                all_results[lbl] = run_or_cached(
                    lbl, trt_xh, params_xh,
                    input_fn=lambda b: random_xc0h_tokens(b, 6))
                del trt_sess_xh; gc.collect()
            except Exception as e:
                print(f"  [ERROR] 16m-precond-smartgate-xc0h-K6 TRT: {e}")

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
                    dummy = torch.zeros(1, in_len_xh, dtype=torch.long, device=device)
                    export_to_onnx(build_pt_precond_smartgate(XC0H_6C4T_CFG), device, onnx_6c4t, dummy=dummy)
                else:
                    print(f"  [TRT] ONNX cached: {onnx_6c4t}")
                trt_6c4t, trt_sess_6c4t = make_trt_infer(
                    onnx_6c4t, TRT_CACHE, max_bs=max(bs), input_shape=str(in_len_xh))
                all_results[lbl] = run_or_cached(
                    lbl, trt_6c4t, params_6c4t,
                    input_fn=lambda b: random_xc0h_tokens(b, 6))
                del trt_sess_6c4t; gc.collect()
            except Exception as e:
                print(f"  [ERROR] 18m-precond-smartgate-xc0h-K6-6c4t TRT: {e}")

    # conv-pure
    if not args.skip_convpure:
        print(f"\n{'='*60}")
        print(f"  conv-pure  (16x ConvBlock(256) NCHW -> trunk_ln"
              f" -> WDL conv(8)/reshape/GELU, SmartGate conv(16)->1024->512 SwiGLU,"
              f" policy 2xLinear->256->MHA(256))")
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
