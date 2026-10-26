#!/usr/bin/env python3
"""attn_head_speed_test.py

Attention-head A/B for the addpos precond family. Tests precond-mha-10c6t-d256
and its wdl3 head variant, each with nn.MultiheadAttention vs
F.scaled_dot_product_attention, on PT eager and ORT+TensorRT. Speed only --
random weights, weight values are irrelevant to throughput.

Flags:
  --skip-trt          Skip the ORT+TensorRT runs (PT eager only)
  --batch-sizes       Space-separated list (default: 1 8 32 64 128 256 384 512)
  --clear-cache       Remove this test's ONNX/engine cache before running
"""

import os
import gc
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model_variant_speed_test_pt import (
    make_pt_eager_infer, export_to_onnx, make_trt_infer, speed_test,
    verify_outputs, random_xc0h_tokens, XC0H_IN_LEN,
)
from chessbot.model import build_pt_precond_addpos

CACHE = os.path.join(os.path.expanduser("~"), ".cache",
                     "conv_chess_trt", "attn_head_cache")

DEFAULT_BATCH_SIZES = [1, 8, 32, 64, 128, 256, 384, 512]

BASE = dict(conv_filters=256, dropout=0.02, pre_blocks=10, tx_blocks=6,
            ff_dim=1024, num_heads=8, xc0h_K=6)


def cfg(wdl_split, attn_impl, norm_impl):
    c = dict(BASE)
    c["wdl_split"] = wdl_split
    c["attn_impl"] = attn_impl
    c["norm_impl"] = norm_impl
    return c


# (label, cfg, onnx stem)
MODELS = [
    ("10c6t_wdl3  ln mha",  cfg(True, "mha",  "ln"), "precond_10c6t_wdl3_ln_mha"),
    ("10c6t_wdl3  ln sdpa", cfg(True, "sdpa", "ln"), "precond_10c6t_wdl3_ln_sdpa"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--skip-trt",    action="store_true")
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--batch-sizes", nargs="+", type=int,
                   default=DEFAULT_BATCH_SIZES, metavar="B")
    return p.parse_args()


def summary_table(all_results, batch_sizes):
    cols = [b for b in [64, 128, 256, 384, 512] if b in batch_sizes]
    if not cols or not all_results:
        return
    cw = 12
    lw = max(len(k) for k in all_results) + 2
    print(f"\n{'='*72}")
    print(f"  samples/s summary")
    print(f"  {'-'*(lw + cw*len(cols))}")
    print(f"  {'model/backend':<{lw}}" + "".join(f"{'bs='+str(b):>{cw}}" for b in cols))
    print(f"  {'-'*(lw + cw*len(cols))}")
    for label in all_results:
        row = f"  {label:<{lw}}"
        for b in cols:
            v = all_results[label].get(b, {}).get("throughput")
            row += f"{int(v):>{cw},}" if v is not None else f"{'--':>{cw}}"
        print(row)
    print(f"  {'-'*(lw + cw*len(cols))}", flush=True)


def main():
    import shutil
    args   = parse_args()
    bs     = args.batch_sizes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.clear_cache and os.path.exists(CACHE):
        shutil.rmtree(CACHE)
        print(f"Cleared cache: {CACHE}")
    os.makedirs(CACHE, exist_ok=True)

    print(f"PyTorch device: {device}", flush=True)
    print(f"cache:          {CACHE}", flush=True)

    input_fn = lambda b: random_xc0h_tokens(b, BASE["xc0h_K"])
    all_results = {}

    for label, model_cfg, stem in MODELS:
        print(f"\n{'='*72}")
        print(f"  {label}", flush=True)
        model  = build_pt_precond_addpos(model_cfg)
        params = sum(p.numel() for p in model.parameters())
        print(f"  params: {params:,}", flush=True)

        ok, _, _, _ = verify_outputs(model, device, label, input_fn=input_fn)
        if not ok:
            print(f"  [SKIP] {label}: output verification failed", flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
            continue

        try:
            eager = make_pt_eager_infer(model, device)
            all_results[f"{label}  PT eager"] = speed_test(
                f"{label}  PT eager [fp16]", eager, bs, input_fn=input_fn)
            del eager
        except Exception as e:
            print(f"  [ERROR] {label} PT eager: {e}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()

        if args.skip_trt:
            continue
        try:
            onnx_path = os.path.join(CACHE, f"{stem}_{params}.onnx")
            if not os.path.exists(onnx_path):
                dummy = torch.zeros(1, XC0H_IN_LEN, dtype=torch.long, device=device)
                export_to_onnx(build_pt_precond_addpos(model_cfg), device,
                               onnx_path, dummy=dummy)
            else:
                print(f"  [TRT] ONNX cached: {onnx_path}", flush=True)
            trt_fn, trt_sess = make_trt_infer(
                onnx_path, CACHE, max_bs=max(bs), input_shape=str(XC0H_IN_LEN))
            all_results[f"{label}  ORT TRT"] = speed_test(
                f"{label}  ORT TRT [fp16]", trt_fn, bs, input_fn=input_fn)
            del trt_sess; gc.collect()
        except Exception as e:
            print(f"  [ERROR] {label} TRT: {e}", flush=True)

    summary_table(all_results, bs)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
