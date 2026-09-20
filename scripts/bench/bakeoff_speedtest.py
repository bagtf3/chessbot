#!/usr/bin/env python3
"""bakeoff_speedtest.py

Speed test for a fixed set of bake-off checkpoints: PT eager (fp16) vs
ORT+TensorRT (fp16), across a handful of batch sizes.  Follows the TRT recipe
in scripts/bench/model_variant_speed_test_pt.py (ORT TensorrtExecutionProvider,
fp16 engine + timing cache, opset-18 ONNX with a dynamic batch axis).

Models (from bakeoff_4c2t_d256.MODEL_SPECS): m1 m7 m11 m13 by default.  Each is
loaded from out-dir / <name><tag>.pt.  ema tracking buffers absent from a
checkpoint are tolerated (they do not affect inference).

Input is the flat xc0h K=6 encoding (B, 393) int; heads are 1858 policy / 3 WDL.
"""

import os
import gc
import sys
import time
import argparse

import numpy as np
import torch

INT8_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "int8"))
if INT8_DIR not in sys.path:
    sys.path.insert(0, INT8_DIR)

from bakeoff_4c2t_d256 import BakeoffModel, MODEL_SPECS
from poc_4c2t_wdl3_d256_maxnet import XC0H_LEN, XC0H_K, XC0H_VOCAB, POLICY_DIM

VALUE_DIM = 3
N_WARMUP = 50
N_ITERS = 100

MODELS_TO_TEST = ["m1", "m7", "m11", "m13"]
DEFAULT_BATCH_SIZES = [32, 64, 128, 256]

TRT_CACHE = os.path.join(os.path.expanduser("~"), ".cache",
                         "conv_chess_trt", "bakeoff_speedtest_cache")
DEFAULT_OUT_DIR = (r"C:\Users\Bryan\Data\chessbot\_data\models"
                   r"\bakeoff_4c2t_d256")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="dir holding the bake-off checkpoints")
    p.add_argument("--tag", default="_rand",
                   help="checkpoint tag suffix: <name><tag>.pt")
    p.add_argument("--models", nargs="+", default=MODELS_TO_TEST, metavar="ID")
    p.add_argument("--batch-sizes", nargs="+", type=int,
                   default=DEFAULT_BATCH_SIZES, metavar="B")
    p.add_argument("--skip-trt", action="store_true")
    p.add_argument("--clear-cache", action="store_true",
                   help="wipe the ONNX/engine cache before running")
    return p.parse_args()


def random_bakeoff_tokens(batch_size):
    """(B, XC0H_LEN) int32 matching board.history_tokens(6)'s flat layout:
    K frames of 64 slim tokens (0..XC0H_VOCAB-1) + K repetition flags + castling
    (0..15) + stm (0/1) + hmc (0..99)."""
    K = XC0H_K
    out = np.zeros((batch_size, XC0H_LEN), dtype=np.int32)
    out[:, :K * 64] = np.random.randint(0, XC0H_VOCAB, (batch_size, K * 64))
    out[:, K * 64:K * 64 + K] = np.random.randint(0, 2, (batch_size, K))
    out[:, K * 64 + K] = np.random.randint(0, 16, batch_size)
    out[:, K * 64 + K + 1] = np.random.randint(0, 2, batch_size)
    out[:, K * 64 + K + 2] = np.random.randint(0, 100, batch_size)
    return out


def load_model(mid, tag, out_dir):
    """Build the spec's BakeoffModel and load its checkpoint. Missing *_ema
    buffers are tolerated; any other missing/unexpected key raises."""
    spec = MODEL_SPECS[mid]
    model = BakeoffModel(conv_mode=spec["conv"], tx_mode=spec["tx"])
    ckpt = os.path.join(out_dir, f"{spec['name']}{tag}.pt")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if not k.endswith("_ema")]
    if bad or res.unexpected_keys:
        raise RuntimeError(
            f"load mismatch {os.path.basename(ckpt)}: "
            f"missing={bad} unexpected={list(res.unexpected_keys)}")
    return spec, model, ckpt


def make_pt_eager_infer(model, device):
    model = model.half().to(device).eval()

    def infer(enc_np):
        with torch.no_grad():
            x = torch.from_numpy(enc_np).long().to(device)
            p, v = model(x)
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    return infer


def verify_outputs(model, device, label, batch=4):
    enc_np = random_bakeoff_tokens(batch)
    m = model.half().to(device).eval()
    with torch.no_grad():
        p, v = m(torch.from_numpy(enc_np).long().to(device))
    p_np = p.float().cpu().numpy()
    v_np = v.float().cpu().numpy()
    ok = True
    if p_np.shape != (batch, POLICY_DIM):
        print(f"  [SHAPE] {label}: policy {p_np.shape}, want {(batch, POLICY_DIM)}")
        ok = False
    if v_np.shape != (batch, VALUE_DIM):
        print(f"  [SHAPE] {label}: value {v_np.shape}, want {(batch, VALUE_DIM)}")
        ok = False
    if not (np.isfinite(p_np).all() and np.isfinite(v_np).all()):
        print(f"  [NAN] {label}: non-finite outputs")
        ok = False
    if ok:
        print(f"  shapes ok: policy {p_np.shape}  value {v_np.shape}  (finite)")
    return ok, p_np, v_np, enc_np


def compare_pt_vs_export(pt_p, pt_v, infer_fn, enc_np, label):
    try:
        ep, ev = infer_fn(enc_np)
    except Exception as e:
        print(f"  [ERROR] {label} export comparison: {e}")
        return
    dp = np.abs(ep - pt_p)
    dv = np.abs(ev - pt_v)
    print(f"  PT vs export  policy: max {dp.max():.4e}  mean {dp.mean():.4e}")
    print(f"                value : max {dv.max():.4e}  mean {dv.mean():.4e}")


def export_to_onnx(model, device, onnx_path):
    import warnings
    import onnx
    from onnx import shape_inference as onnx_si
    print(f"  Exporting to ONNX (opset 18) ...")
    model = model.half().to(device).eval()
    dummy = torch.zeros(1, XC0H_LEN, dtype=torch.long, device=device)
    with torch.no_grad():
        model(dummy)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Constant folding in symbolic shape inference")
        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=18,
            input_names=["enc_in"],
            output_names=["policy_logits", "value_out"],
            dynamic_axes={"enc_in": {0: "batch"},
                          "policy_logits": {0: "batch"},
                          "value_out": {0: "batch"}},
            do_constant_folding=False)
    proto = onnx.load(onnx_path)
    proto = onnx_si.infer_shapes(proto)
    onnx.save(proto, onnx_path)
    sz_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ONNX export done ({sz_mb:.1f} MB)")


def make_trt_infer(onnx_path, cache_dir, max_bs):
    import hashlib
    import tensorrt  # noqa: F401  (import registers the EP)
    import onnxruntime as ort

    h = hashlib.sha256()
    with open(onnx_path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    prefix = f"bake_{h.hexdigest()[:12]}"

    ort.set_default_logger_severity(3)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3

    shape = str(XC0H_LEN)
    trt_opts = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_engine_cache_prefix": prefix,
        "trt_fp16_enable": True,
        "trt_force_timing_cache": True,
        "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
        "trt_profile_min_shapes": f"enc_in:1x{shape}",
        "trt_profile_opt_shapes": f"enc_in:{max_bs}x{shape}",
        "trt_profile_max_shapes": f"enc_in:{max_bs}x{shape}",
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache_dir,
    }

    engines = [f for f in os.listdir(cache_dir)
               if f.startswith(prefix) and f.endswith(".engine")]
    print(f"  [trt] {'loading cached engine' if engines else 'no cache, compiling...'}")

    providers = [("TensorrtExecutionProvider", trt_opts)]
    t0 = time.perf_counter()
    session = ort.InferenceSession(
        onnx_path, sess_options=sess_opts, providers=providers)
    active = session.get_providers()[0]
    if active != "TensorrtExecutionProvider":
        raise RuntimeError(f"[trt] expected TensorrtExecutionProvider, got {active}")
    print(f"  [trt] ready in {time.perf_counter() - t0:.1f}s")

    def infer(enc_np):
        return session.run(None, {"enc_in": enc_np.astype(np.int64)})

    return infer, session


def speed_test(label, infer_fn, batch_sizes):
    print(f"\n  {'-'*56}")
    print(f"  {label}")
    print(f"  warmup={N_WARMUP}  iters={N_ITERS}")
    print(f"  {'batch':<7} {'latency_ms':>12} {'samples/s':>13}")
    print(f"  {'-'*7} {'-'*12} {'-'*13}")
    results = {}
    for b in batch_sizes:
        inp = random_bakeoff_tokens(b)
        for _ in range(N_WARMUP):
            infer_fn(inp)
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            infer_fn(inp)
        avg_s = (time.perf_counter() - t0) / N_ITERS
        tput = b / avg_s
        print(f"  {b:<7} {avg_s*1000:>12.2f} {tput:>13.0f}")
        results[b] = {"latency_ms": round(avg_s * 1000, 3),
                      "throughput": round(tput, 1)}
    return results


def print_summary_table(all_results, batch_sizes):
    if not all_results:
        return
    col_w = 12
    label_w = max(len(k) for k in all_results) + 2
    header = f"  {'model/backend':<{label_w}}" + \
        "".join(f"{'bs='+str(b):>{col_w}}" for b in batch_sizes)
    line = "-" * (label_w + col_w * len(batch_sizes))
    print(f"\n{'='*60}")
    print(f"  samples/s summary")
    print(f"  {line}")
    print(header)
    print(f"  {line}")
    sort_bs = 256 if 256 in batch_sizes else batch_sizes[-1]

    def sort_key(item):
        val = item[1].get(sort_bs, {}).get("throughput")
        return (val is None, -(val or 0))

    for label, res in sorted(all_results.items(), key=sort_key):
        row = f"  {label:<{label_w}}"
        for b in batch_sizes:
            val = res.get(b, {}).get("throughput")
            row += f"{int(val):>{col_w},}" if val is not None else f"{'--':>{col_w}}"
        print(row)
    print(f"  {line}")


def main():
    import shutil
    args = parse_args()
    bs = args.batch_sizes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.clear_cache and os.path.exists(TRT_CACHE):
        shutil.rmtree(TRT_CACHE)
        print(f"Cleared cache: {TRT_CACHE}")
    os.makedirs(TRT_CACHE, exist_ok=True)

    print(f"PyTorch device: {device}")
    print(f"TRT cache:      {TRT_CACHE}")
    print(f"checkpoints:    {args.out_dir}  (tag '{args.tag}')")

    all_results = {}
    for mid in args.models:
        print(f"\n{'='*60}")
        spec, model, ckpt = load_model(mid, args.tag, args.out_dir)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {mid}: {spec['name']}  conv={spec['conv']} tx={spec['tx']}")
        print(f"  ckpt {os.path.basename(ckpt)}  params {params:,}")

        ok, pt_p, pt_v, enc_np = verify_outputs(model, device, mid)
        if not ok:
            print(f"  [SKIP] {mid}: output verification failed")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        try:
            print(f"\n  Building {mid} PT eager ...")
            eager = make_pt_eager_infer(model, device)
            lbl = f"{mid}  PT eager"
            all_results[lbl] = speed_test(lbl + " [fp16]", eager, bs)
            del eager
        except Exception as e:
            print(f"  [ERROR] {mid} PT eager: {e}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if args.skip_trt:
            continue
        try:
            lbl = f"{mid}  ORT TRT"
            onnx_path = os.path.join(TRT_CACHE, f"{spec['name']}_{params}.onnx")
            if not os.path.exists(onnx_path):
                _, exp_model, _ = load_model(mid, args.tag, args.out_dir)
                export_to_onnx(exp_model, device, onnx_path)
                del exp_model
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print(f"  [TRT] ONNX cached: {onnx_path}")
            trt_fn, trt_sess = make_trt_infer(onnx_path, TRT_CACHE, max_bs=max(bs))
            compare_pt_vs_export(pt_p, pt_v, trt_fn, enc_np, mid)
            all_results[lbl] = speed_test(lbl + " [fp16]", trt_fn, bs)
            del trt_sess
            gc.collect()
        except Exception as e:
            print(f"  [ERROR] {mid} TRT: {e}")

    print_summary_table(all_results, bs)
    print("\nDone.")


if __name__ == "__main__":
    main()
