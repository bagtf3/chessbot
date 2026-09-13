"""Train the toy, freeze, export explicit Q/DQ ONNX, and check onnxruntime
reproduces the frozen torch reference bit for bit.

    python scripts/int8/poc_onnx_qdq.py [--cpu] [--steps 2000] [--out file.onnx]
"""
import argparse
import os
from collections import Counter

import numpy as np
import torch

from chessbot.int8 import freeze_int8
from chessbot.int8.onnx_qdq import export_frozen_sequential_qdq
from poc_layer_demo import train_toy

DEFAULT_OUT = r"C:\Users\Bryan\Data\chessbot_data\int8_poc\toy_qdq.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    model, batch = train_toy(dev, args.steps)
    frozen = freeze_int8(model)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    proto = export_frozen_sequential_qdq(frozen, args.out)
    ops = Counter(n.op_type for n in proto.graph.node)
    print(f"\n[onnx] wrote {args.out}", flush=True)
    print(f"[onnx] {len(proto.graph.node)} nodes: "
          + "  ".join(f"{k}={v}" for k, v in sorted(ops.items())), flush=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    x, _ = batch()
    x_np = x.cpu().numpy().astype(np.float32)
    ort_out = sess.run(["output"], {"input": x_np})[0]
    with torch.no_grad():
        ref = frozen(x.to(torch.int8)).cpu().numpy()

    diff = np.abs(ort_out.astype(np.int32) - ref.astype(np.int32))
    print(f"[ort]  output dtype={ort_out.dtype}  shape={ort_out.shape}", flush=True)
    print(f"[ort]  vs frozen torch: exact={np.mean(diff == 0):.4f}"
          f"  within_1={np.mean(diff <= 1):.4f}  max={diff.max()}", flush=True)


if __name__ == "__main__":
    main()
