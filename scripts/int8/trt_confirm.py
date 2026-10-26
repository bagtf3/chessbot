"""Build a strongly-typed INT8+FP32 TRT engine from an explicit-Q/DQ ONNX at
DETAILED verbosity and print a per-layer precision histogram.

Run in its own process (only TensorRT imported) so the builder owns a clean CUDA
context -- building alongside torch/onnxruntime in one process segfaults.  Exit 0
only when at least one layer is INT8 and none are FP16.
"""
import argparse
import json
import sys
from pathlib import Path

import tensorrt as trt


def shape(s):
    return tuple(int(x) for x in s.split("x"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out-engine", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--input", default="xc0h")
    ap.add_argument("--min-shape", required=True)
    ap.add_argument("--opt-shape", required=True)
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(args.onnx).read_bytes()):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        print(f"[confirm] parse failed: {errs}", flush=True)
        sys.exit(2)

    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape(args.input, shape(args.min_shape),
                      shape(args.opt_shape), shape(args.opt_shape))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[confirm] build returned no engine", flush=True)
        sys.exit(3)
    Path(args.out_engine).write_bytes(serialized)
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    info = engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON)
    Path(args.out_json).write_text(info, encoding="utf-8")

    doc = json.loads(info)
    layers = doc["Layers"] if isinstance(doc, dict) else doc
    tally = {}
    fp16 = []
    for lyr in layers:
        prec = lyr.get("Precision", "UNKNOWN") if isinstance(lyr, dict) else "UNKNOWN"
        tally[prec] = tally.get(prec, 0) + 1
        if "HALF" in prec.upper() or "FP16" in prec.upper():
            fp16.append(lyr.get("Name", "?"))
    order = sorted(tally.items(), key=lambda kv: -kv[1])
    print(f"[confirm] layers={len(layers)}  histogram: "
          + "  ".join(f"{k}={v}" for k, v in order), flush=True)
    if fp16:
        print(f"[confirm] FAIL: {len(fp16)} FP16 layers: {fp16[:8]}", flush=True)
        sys.exit(4)
    n_int8 = sum(v for k, v in tally.items() if "INT8" in k.upper())
    print(f"[confirm] INT8={n_int8}  FP16=0  engine={args.out_engine}", flush=True)
    sys.exit(0 if n_int8 > 0 else 5)


if __name__ == "__main__":
    main()
