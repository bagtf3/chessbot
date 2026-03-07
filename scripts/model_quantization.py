#!/usr/bin/env python3
"""model_quantization.py

Optimizes a Xerces conformer model with TF-TRT (FP16 and INT8) and benchmarks
against the TF+XLA baseline. Uses TensorFlow's built-in TRT integration —
no extra packages required.

TF-TRT vs ONNX+TRT:
    TF-TRT partitions the graph into TRT-supported subgraphs and compiles them
    into native TRT engines, while leaving unsupported ops (e.g. the Embedding
    layer, some custom activations) in TF. The result is faster than plain XLA
    but stays in the TF ecosystem.

Usage:
    python model_quantization.py --model /path/to/model.h5
    python model_quantization.py --model /path/to/model.h5 --skip-tf --skip-int8
"""

import argparse
import gc
import os
import sys
import time

# Must be set before TF is imported so XLA can find libdevice for JIT compilation.
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
)

import numpy as np
import tensorflow as tf

VOCAB_SIZE = 21
SEQ_LEN    = 64

N_WARMUP_TF  = 20
N_WARMUP_TRT = 40   # TRT may build its engine on first calls per shape
N_ITERS      = 100


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Path to trained Xerces .h5 model")
    p.add_argument("--trt-dir", default=None,
                   help="Where to save TRT SavedModels (default: alongside .h5)")
    p.add_argument("--skip-tf",   action="store_true", help="Skip TF+XLA baseline")
    p.add_argument("--skip-int8", action="store_true", help="Skip INT8 session")
    p.add_argument("--batch-sizes", nargs="+", type=int,
                   default=[1, 4, 8, 16, 32, 64, 128, 256, 512])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_tokens(batch_size: int) -> np.ndarray:
    return np.random.randint(0, VOCAB_SIZE, size=(batch_size, SEQ_LEN), dtype=np.int32)


def print_gpu_info():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TF sees {len(gpus)} GPU(s): {[g.name for g in gpus]}")


# ---------------------------------------------------------------------------
# TF + XLA baseline
# ---------------------------------------------------------------------------

def make_tf_infer(model_path: str):
    model = tf.keras.models.load_model(model_path)

    @tf.function(
        input_signature=[tf.TensorSpec([None, SEQ_LEN], tf.int32)],
        jit_compile=True,
    )
    def _fwd(x):
        return model(x, training=False)

    # trigger XLA trace before timing
    _ = _fwd(tf.constant(random_tokens(64)))

    def infer(tokens_np: np.ndarray):
        out = _fwd(tf.constant(tokens_np))
        policy = out[0].numpy()
        value  = out[1].numpy()
        return policy, value

    return infer, model


# ---------------------------------------------------------------------------
# TF-TRT conversion
# ---------------------------------------------------------------------------

def convert_tftrt(saved_model_dir: str, output_dir: str,
                  precision: str, max_workspace_mb: int = 4096):
    """
    Convert a SavedModel to a TF-TRT optimized SavedModel.
    precision: "FP16" or "INT8"
    """
    from tensorflow.python.compiler.tensorrt import trt_convert as trt

    params = trt.TrtConversionParams(
        precision_mode=precision,
        max_workspace_size_bytes=max_workspace_mb * 1024 * 1024,
        minimum_segment_size=3,        # min TRT-eligible ops to form a subgraph
        max_batch_size=512,
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_dir,
        conversion_params=params,
    )

    def calibration_input_fn():
        for b in [1, 8, 32, 64, 128, 256, 512]:
            yield (tf.constant(random_tokens(b)),)

    if precision == "INT8":
        print("  Converting [INT8] with calibration ...")
        converter.convert(calibration_input_fn=calibration_input_fn)
    else:
        print(f"  Converting [{precision}] ...")
        converter.convert()

    print(f"  Building engines and saving to {output_dir} ...")
    converter.build(input_fn=calibration_input_fn)
    converter.save(output_dir)
    print(f"  Done.")


def make_tftrt_infer(trt_saved_model_dir: str):
    """Load a TF-TRT SavedModel and return an inference callable."""
    loaded = tf.saved_model.load(trt_saved_model_dir)
    infer_fn = loaded.signatures["serving_default"]

    # discover input/output key names
    input_keys  = list(infer_fn.structured_input_signature[1].keys())
    output_keys = list(infer_fn.structured_outputs.keys())
    print(f"  Input keys:  {input_keys}")
    print(f"  Output keys: {output_keys}")

    # policy logits will be the 4288-dim output, value will be dim-1
    policy_key = next(k for k in output_keys
                      if infer_fn.structured_outputs[k].shape[-1] != 1)
    value_key  = next(k for k in output_keys
                      if infer_fn.structured_outputs[k].shape[-1] == 1)

    def infer(tokens_np: np.ndarray):
        inp = tf.constant(tokens_np)
        out = infer_fn(**{input_keys[0]: inp})
        policy = out[policy_key].numpy()
        value  = out[value_key].numpy()
        return policy, value

    return infer


# ---------------------------------------------------------------------------
# Speed test
# ---------------------------------------------------------------------------

def speed_test(label: str, infer_fn, batch_sizes,
               n_warmup: int = N_WARMUP_TF, n_iters: int = N_ITERS):
    banner = f"  {label}  "
    print(f"\n{'#' * 62}")
    print(f"{banner:^62}")
    print(f"{'#' * 62}")
    print(f"  warmup={n_warmup}  iters={n_iters}\n")
    print(f"  {'batch':<7} {'latency_ms':>12} {'samples/s':>13}")
    print(f"  {'-'*7} {'-'*12} {'-'*13}")

    results = {}
    for b in batch_sizes:
        inp = random_tokens(b)

        for _ in range(n_warmup):
            policy, value = infer_fn(inp)
        _ = policy[0]  # materialise last warmup result

        t0 = time.perf_counter()
        for _ in range(n_iters):
            policy, value = infer_fn(inp)
        _ = policy[0]  # sync final result
        t1 = time.perf_counter()

        avg_s  = (t1 - t0) / n_iters
        avg_ms = avg_s * 1000.0
        tput   = b / avg_s

        print(f"  {b:<7} {avg_ms:>12.2f} {tput:>13.0f}")
        results[b] = {"latency_ms": avg_ms, "throughput": tput}

    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(all_results: dict, batch_sizes):
    names = list(all_results.keys())
    col_w = 20

    print(f"\n{'=' * 62}")
    print("  THROUGHPUT COMPARISON  (samples/s)")
    print(f"  baseline = {names[0]}")
    print(f"{'=' * 62}")

    hdr = f"  {'batch':<7}" + "".join(f"{n:>{col_w}}" for n in names)
    print(hdr)
    print("  " + "-" * (7 + col_w * len(names)))

    baseline_name = names[0]
    for b in batch_sizes:
        row = f"  {b:<7}"
        base_tp = all_results[baseline_name].get(b, {}).get("throughput", 0)
        for i, name in enumerate(names):
            tp = all_results[name].get(b, {}).get("throughput", 0)
            if i == 0:
                row += f"{tp:>{col_w}.0f}"
            else:
                pct  = (tp / base_tp - 1.0) * 100 if base_tp > 0 else 0
                cell = f"{tp:.0f} ({pct:+.0f}%)"
                row += f"{cell:>{col_w}}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    model_dir  = os.path.dirname(os.path.abspath(args.model))
    stem       = os.path.splitext(os.path.basename(args.model))[0]
    trt_root   = args.trt_dir or os.path.join(model_dir, stem + "_trt")
    sm_dir     = os.path.join(trt_root, "saved_model")   # base SavedModel
    fp16_dir   = os.path.join(trt_root, "trt_fp16")
    int8_dir   = os.path.join(trt_root, "trt_int8")

    print_gpu_info()
    batch_sizes = args.batch_sizes
    all_results = {}

    # ------------------------------------------------------------------
    # 1. TF + XLA baseline
    # ------------------------------------------------------------------
    if not args.skip_tf:
        print("\n" + "=" * 62)
        print("  Loading model for TF+XLA baseline ...")
        tf_infer, tf_model = make_tf_infer(args.model)

        all_results["TF+XLA (fp16)"] = speed_test(
            "TF + XLA  [fp16]", tf_infer, batch_sizes,
            n_warmup=N_WARMUP_TF,
        )

        # Export SavedModel for TRT conversion (do it while model is loaded)
        if not os.path.exists(sm_dir):
            print(f"\nSaving SavedModel → {sm_dir}")
            os.makedirs(sm_dir, exist_ok=True)
            tf.saved_model.save(tf_model, sm_dir)
            print("  SavedModel saved.")

        del tf_infer, tf_model
        gc.collect()
        tf.keras.backend.clear_session()

    else:
        # If skipping TF baseline we still need the SavedModel for TRT
        if not os.path.exists(sm_dir):
            print(f"\nLoading model to export SavedModel → {sm_dir}")
            m = tf.keras.models.load_model(args.model)
            os.makedirs(sm_dir, exist_ok=True)
            tf.saved_model.save(m, sm_dir)
            del m
            gc.collect()
            tf.keras.backend.clear_session()

    # ------------------------------------------------------------------
    # 2. TF-TRT FP16
    # ------------------------------------------------------------------
    if not os.path.exists(fp16_dir):
        print("\nConverting TF-TRT FP16 ...")
        convert_tftrt(sm_dir, fp16_dir, precision="FP16")
    else:
        print(f"\nTF-TRT FP16 already built: {fp16_dir}")

    print("\nLoading TF-TRT FP16 session ...")
    trt_fp16_infer = make_tftrt_infer(fp16_dir)

    all_results["TF-TRT (fp16)"] = speed_test(
        "TF-TRT  [FP16]", trt_fp16_infer, batch_sizes,
        n_warmup=N_WARMUP_TRT,
    )
    del trt_fp16_infer
    gc.collect()

    # ------------------------------------------------------------------
    # 3. TF-TRT INT8
    # ------------------------------------------------------------------
    if not args.skip_int8:
        if not os.path.exists(int8_dir):
            print("\nConverting TF-TRT INT8 ...")
            convert_tftrt(sm_dir, int8_dir, precision="INT8")
        else:
            print(f"\nTF-TRT INT8 already built: {int8_dir}")

        print("\nLoading TF-TRT INT8 session ...")
        trt_int8_infer = make_tftrt_infer(int8_dir)

        all_results["TF-TRT (int8)"] = speed_test(
            "TF-TRT  [INT8]", trt_int8_infer, batch_sizes,
            n_warmup=N_WARMUP_TRT,
        )
        del trt_int8_infer
        gc.collect()

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print_comparison(all_results, batch_sizes)


if __name__ == "__main__":
    main()
