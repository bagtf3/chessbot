"""Sustained fp32 vs int8+fp32 inference speed + VRAM test on real data.

Loads N shards into ONE in-memory int64 buffer up front, so the timed pass does
zero disk/Python I/O -- pure engine throughput.  Runs each engine in its OWN
subprocess so GPU memory is isolated and you can watch it cleanly per engine.

    python scripts/int8/speed_sustained.py            # runs both, back to back
    python scripts/int8/speed_sustained.py --engine int8   # one engine

Everything prints as it happens so you know when to watch nvidia-smi.
"""
import argparse
import glob
import gzip
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poc_2c2t_wdl3_d256 as poc  # noqa: E402

SHARD_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\pretrain_shards"
OUT_DIR = r"C:\Users\Bryan\Data\chessbot\_data\models\int8_poc_2c2t"
CKPT = os.path.join(OUT_DIR, "poc_2c2t_wdl3_int8dot_qat.pt")


def gpu_mem_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        return out.stdout.strip().splitlines()[0].strip() + " MiB"
    except Exception:
        return "?"


def load_buffer(shard_dir, n_shards, per_shard):
    files = sorted(glob.glob(str(Path(shard_dir) / "*.pkl.gz")))
    random.Random(0).shuffle(files)
    encs = []
    for path in files[:n_shards]:
        with gzip.open(path, "rb") as fh:
            records = pickle.load(fh)
        random.Random(1).shuffle(records)
        for r in records[:per_shard]:
            encs.append(np.asarray(r[0], dtype=np.int64))
        print(f"[buffer] loaded {Path(path).name}: {min(per_shard, len(records))} rows"
              f"  (running total {len(encs)})", flush=True)
    buf = np.stack(encs)
    print(f"[buffer] ONE buffer ready: shape={buf.shape} dtype={buf.dtype}"
          f"  ~{buf.nbytes / 2**20:.0f} MiB in RAM (no I/O during timing)", flush=True)
    return buf


def export_onnxes(out_dir, checkpoint):
    import torch
    out_dir = Path(out_dir)
    model = poc.Int8Ready2C2TWDL(dot_quant=True)
    legacy_model = poc.Int8Ready2C2TWDL(dot_quant=False)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    legacy_model.load_state_dict(state["model"])
    model.eval()
    legacy_model.eval()
    fp32 = out_dir / "poc_2c2t_wdl3_fp32.onnx"
    int8 = out_dir / "poc_2c2t_wdl3_int8dot_qdq.onnx"
    legacy = out_dir / "poc_2c2t_wdl3_int8_qdq.onnx"
    dummy = torch.zeros(1, poc.XC0H_LEN, dtype=torch.long)
    poc.QAT_BLEND = 0.0
    poc.export_onnx(poc.QATExportWrapper(model, qat=False).eval(), dummy, fp32)
    poc.QAT_BLEND = 1.0
    poc.export_onnx(poc.QATExportWrapper(legacy_model, qat=True).eval(), dummy, legacy)
    poc.export_onnx(poc.QATExportWrapper(model, qat=True).eval(), dummy, int8)
    print(f"[export] wrote {fp32.name}, {legacy.name}, and {int8.name}", flush=True)
    return str(fp32), str(legacy), str(int8)


def run_one(label, onnx_path, out_dir, batch, repeats, warmup):
    int8 = label != "fp32"
    print(f"\n========== {label.upper()} ENGINE ==========", flush=True)
    print(f"[{label}] GPU mem before build: {gpu_mem_mb()}", flush=True)
    print(f"[{label}] building/compiling TRT engine (int8={int8}, fp16=False) ...", flush=True)
    t0 = time.perf_counter()
    session = poc.build_trt_session(onnx_path, out_dir, batch, int8=int8)
    print(f"[{label}] engine ready in {time.perf_counter() - t0:.1f}s"
          f"   GPU mem after build: {gpu_mem_mb()}", flush=True)

    buf = load_buffer(SHARD_DIR, ARGS.shards, ARGS.per_shard)
    n = buf.shape[0] - (buf.shape[0] % batch)
    print(f"[{label}] warming up {warmup} batches ...", flush=True)
    for s in range(0, warmup * batch, batch):
        session.run(None, {"xc0h": buf[s:s + batch]})

    print(f"\n>>> {label.upper()} SUSTAINED PASS STARTING -- WATCH VRAM NOW <<<", flush=True)
    print(f"[{label}] {repeats} passes over {n} positions at batch {batch}"
          f"  (GPU mem: {gpu_mem_mb()})", flush=True)
    total_pos = 0
    t0 = time.perf_counter()
    for rep in range(repeats):
        for s in range(0, n, batch):
            session.run(None, {"xc0h": buf[s:s + batch]})
        total_pos += n
        print(f"[{label}] pass {rep + 1}/{repeats} done"
              f"  elapsed={time.perf_counter() - t0:.1f}s  GPU mem={gpu_mem_mb()}", flush=True)
    dt = time.perf_counter() - t0

    pos_s = total_pos / dt
    ms_batch = dt / (total_pos / batch) * 1e3
    print(f"\n[{label}] DONE: {total_pos} positions in {dt:.2f}s"
          f"  ->  {pos_s:.0f} pos/s   {ms_batch:.3f} ms/batch"
          f"   final GPU mem: {gpu_mem_mb()}", flush=True)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fp32", "int8", "int8dot", "all", "both"], default="all",
                    help="run one engine or all three (fp32, legacy int8, new int8dot)")
    ap.add_argument("--onnx", default=None, help="internal: prebuilt onnx for a child run")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--per-shard", type=int, default=10240)
    ap.add_argument("--repeats", type=int, default=20,
                    help="passes over the whole buffer (longer = easier to watch VRAM)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--checkpoint", default=CKPT,
                    help="trained POC checkpoint; defaults to the current int8dot artifact")
    ARGS = ap.parse_args()

    if ARGS.engine not in ("all", "both"):
        run_one(ARGS.engine, ARGS.onnx, OUT_DIR, ARGS.batch, ARGS.repeats, ARGS.warmup)
        return

    print("[main] exporting fresh ONNX from checkpoint ...", flush=True)
    fp32_onnx, legacy_onnx, int8dot_onnx = export_onnxes(OUT_DIR, ARGS.checkpoint)
    runs = [("fp32", fp32_onnx), ("int8", legacy_onnx), ("int8dot", int8dot_onnx)]
    for label, onnx_path in runs:
        print(f"\n[main] launching {label} in a clean subprocess (isolated VRAM) ...", flush=True)
        rc = subprocess.run(
            [sys.executable, "-u", os.path.abspath(__file__),
             "--engine", label, "--onnx", onnx_path,
             "--batch", str(ARGS.batch), "--shards", str(ARGS.shards),
             "--per-shard", str(ARGS.per_shard), "--repeats", str(ARGS.repeats),
             "--warmup", str(ARGS.warmup)]).returncode
        if rc != 0:
            print(f"[main] {label} run failed (rc={rc})", flush=True)
    print("\n[main] requested engine comparisons done -- compare the pos/s and GPU-mem lines above.", flush=True)


if __name__ == "__main__":
    main()
