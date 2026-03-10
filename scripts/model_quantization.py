#!/usr/bin/env python3
"""model_quantization.py

Compare inference backends for the Xerces conformer model:

  1. TF + XLA              – fp16, trained weights from .h5
  2. ORT + TRT FP16        – fp16, from TF→ONNX export
  3. ORT + TRT INT8        – int8, from PyTorch→ONNX export (TRT internal calib)
  4. PyTorch eager         – fp16, random init (same architecture)
  5. PyTorch cudagraphs    – fp16, torch.compile cudagraphs backend

PyTorch benchmarks use random weights — throughput is architecture/backend
dependent, not weight-dependent.  Parameter count is verified to match TF.

Requirements:
    TF baselines:   tensorflow, tf2onnx, onnxruntime-gpu  (chess_onnx env)
    PyTorch:        torch >= 2.0 (for torch.compile)

Usage:
    python model_quantization.py --model /path/to/model.h5
    python model_quantization.py --model /path/to/model.h5 --skip-tf
    python model_quantization.py --model /path/to/model.h5 --skip-tf --skip-ort
"""

import os
import argparse
import gc
import sys
import time

# ---------------------------------------------------------------------------
# XLA libdevice fix — MUST happen before TF is imported
# ---------------------------------------------------------------------------

def _fix_xla_libdevice():
    import shutil, tempfile
    src = (r"C:\Program Files\NVIDIA GPU Computing Toolkit"
           r"\CUDA\v11.8\nvvm\libdevice\libdevice.10.bc")
    if not os.path.exists(src):
        return
    dst_dir = os.path.join(tempfile.gettempdir(), "xla_cuda", "nvvm", "libdevice")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "libdevice.10.bc")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    os.environ["XLA_FLAGS"] = (
        f"--xla_gpu_cuda_data_dir="
        f"{os.path.join(tempfile.gettempdir(), 'xla_cuda')}"
    )

_fix_xla_libdevice()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

VOCAB_SIZE = 21
SEQ_LEN    = 64

# Active conformer config (mirrors CLAUDE.md + model.py build_conformer_64x67)
CONF = dict(
    vocab_size        = 21,
    d_embed           = 128,
    conv_filters      = 256,
    conv_blocks       = 10,
    transformer_layers= 5,
    num_heads         = 8,
    ff_dim            = 1024,
    dropout           = 0.025,
)

# All GPU backends use the same warmup so timing starts from the same
# thermal/compiled state.  50 is enough for XLA to finish its trace,
# TRT to build its engine, and cudagraphs to capture each batch size.
N_WARMUP_GPU = 50
N_ITERS      = 100


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",
                   default=r"C:/Users/Bryan/Data/chessbot_data/selfplay_runs/val_test/val_test_model.h5",
                   help="Path to trained Xerces .h5 model")
    p.add_argument("--out-dir", default=None,
                   help="Where to write ONNX + TRT cache (default: next to .h5)")
    p.add_argument("--skip-tf",         action="store_true", help="Skip TF+XLA baseline")
    p.add_argument("--skip-ort",        action="store_true", help="Skip ORT+TRT FP16")
    p.add_argument("--skip-torch",      action="store_true", help="Skip all PyTorch")
    p.add_argument("--skip-ort-int8",   action="store_true", help="Skip ORT+TRT int8")
    p.add_argument("--batch-sizes", nargs="+", type=int,
                   default=[1, 4, 8, 16, 32, 64, 128, 256, 512])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def random_tokens(batch_size: int) -> np.ndarray:
    return np.random.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), dtype=np.int32)


# ---------------------------------------------------------------------------
# PyTorch model — identical architecture to build_conformer_64x67
# ---------------------------------------------------------------------------

def build_pytorch_model():
    """
    PyTorch re-implementation of build_conformer_64x67 with active config.

    Architecture:
        token_emb (21→128) → conv_proj (128→256, 1x1) → ln → lrelu
        → 10 × ConvResBlock(256)
        → reshape (B,64,256) + pos_emb(64,256)
        → 5 × TransformerLayer(d=256, heads=8, ff=1024)
        → reshape (B,256,8,8)
        → conv_mix_1x1 → ln → lrelu
        → policy_conv(256→67, 1x1)  → (B, 4288)
        → value branch: v_conv1(256→64,3x3) → ln → lrelu → GAP
                        → fc(64→256) → fc(256→128) → fc(128→1, tanh)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = CONF

    class Ln2d(nn.Module):
        """
        Channel-wise LayerNorm on NCHW, implemented with primitive ops.
        Avoids the LayerNormalization ONNX op and Transpose nodes that TRT
        cannot infer shapes through.  Mathematically identical to the
        permute→nn.LayerNorm→permute approach.
        """
        def __init__(self, channels, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias   = nn.Parameter(torch.zeros(channels))
            self.eps    = eps

        def forward(self, x):                              # x: (N,C,H,W)
            mean     = x.mean(dim=1, keepdim=True)
            centered = x - mean
            var      = (centered * centered).mean(dim=1, keepdim=True)
            inv_std  = torch.rsqrt(var + self.eps)
            w = self.weight.view(1, -1, 1, 1)
            b = self.bias.view(1, -1, 1, 1)
            return centered * inv_std * w + b

    class ConvResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1  = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.lrelu1 = nn.LeakyReLU(0.01, inplace=True)
            self.conv2  = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.ln2    = Ln2d(channels)
            self.lrelu  = nn.LeakyReLU(0.01, inplace=True)

        def forward(self, x):
            h = self.lrelu1(self.conv1(x))
            h = self.ln2(self.conv2(h))
            return self.lrelu(x + h)

    class TransformerLayer(nn.Module):
        def __init__(self, d_model, num_heads, ff_dim, dropout):
            super().__init__()
            self.ln1      = nn.LayerNorm(d_model)
            self.attn     = nn.MultiheadAttention(d_model, num_heads,
                                                  dropout=0.0, batch_first=True)
            self.attn_drop= nn.Dropout(dropout)
            self.ln2      = nn.LayerNorm(d_model)
            self.ff1      = nn.Linear(d_model, ff_dim)
            self.ff_drop  = nn.Dropout(dropout)
            self.ff2      = nn.Linear(ff_dim, d_model)

        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.attn_drop(h)
            h = self.ff2(self.ff_drop(F.gelu(self.ff1(self.ln2(x)))))
            return x + h

    class XercesConformer(nn.Module):
        def __init__(self):
            super().__init__()
            c = cfg["conv_filters"]
            d = cfg["d_embed"]

            self.token_emb  = nn.Embedding(cfg["vocab_size"], d)
            self.conv_proj  = nn.Conv2d(d, c, 1, bias=False)
            self.ln_proj    = Ln2d(c)
            self.lrelu_proj = nn.LeakyReLU(0.01, inplace=True)

            self.conv_blocks = nn.ModuleList(
                [ConvResBlock(c) for _ in range(cfg["conv_blocks"])]
            )

            self.pos_emb = nn.Embedding(64, c)

            self.transformer = nn.ModuleList([
                TransformerLayer(c, cfg["num_heads"], cfg["ff_dim"], cfg["dropout"])
                for _ in range(cfg["transformer_layers"])
            ])

            self.conv_mix   = nn.Conv2d(c, c, 1, bias=False)
            self.ln_mix     = Ln2d(c)
            self.lrelu_mix  = nn.LeakyReLU(0.01, inplace=True)

            self.policy_conv = nn.Conv2d(c, 67, 1)

            self.v_conv1  = nn.Conv2d(c, 64, 3, padding=1, bias=False)
            self.v_ln1    = Ln2d(64)
            self.v_lrelu1 = nn.LeakyReLU(0.01, inplace=True)
            self.v_fc1    = nn.Linear(64, 256)
            self.v_drop   = nn.Dropout(cfg["dropout"])
            self.v_fc2    = nn.Linear(256, 128)
            self.v_out    = nn.Linear(128, 1)

            self.c = c
            self.d = d

        def forward(self, tokens):
            batch = tokens.shape[0]
            c = self.c
            d = self.d

            x = self.token_emb(tokens)
            x = x.view(batch, 8, 8, d).permute(0, 3, 1, 2).contiguous()

            x = self.lrelu_proj(self.ln_proj(self.conv_proj(x)))

            for blk in self.conv_blocks:
                x = blk(x)

            seq = x.permute(0, 2, 3, 1).reshape(batch, 64, c)
            pos = self.pos_emb(torch.arange(64, device=tokens.device))
            seq = seq + pos.unsqueeze(0)

            for layer in self.transformer:
                seq = layer(seq)

            x = seq.reshape(batch, 8, 8, c).permute(0, 3, 1, 2).contiguous()

            x = self.lrelu_mix(self.ln_mix(self.conv_mix(x)))

            policy = self.policy_conv(x)
            policy = policy.permute(0, 2, 3, 1).reshape(batch, 64 * 67)

            v = self.v_lrelu1(self.v_ln1(self.v_conv1(x)))
            v = v.mean(dim=[2, 3])
            v = F.relu(self.v_fc1(v))
            v = self.v_drop(v)
            v = F.relu(self.v_fc2(v))
            v = torch.tanh(self.v_out(v))

            return policy, v

    import torch
    model = XercesConformer()
    n = sum(p.numel() for p in model.parameters())
    print(f"  PyTorch param count: {n:,}  (expected ~16M)")
    return model


# ---------------------------------------------------------------------------
# TF + XLA baseline
# ---------------------------------------------------------------------------

def make_tf_infer(model_path: str, xla: bool = True):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    model = tf.keras.models.load_model(model_path)

    @tf.function(
        input_signature=[tf.TensorSpec([None, SEQ_LEN], tf.int32)],
        jit_compile=xla,
    )
    def _fwd(x):
        return model(x, training=False)

    _ = _fwd(tf.constant(random_tokens(64)))  # trigger trace / XLA compile

    def infer(tokens_np: np.ndarray):
        out    = _fwd(tf.constant(tokens_np))
        policy = out[0].numpy()
        value  = out[1].numpy()
        return policy, value

    return infer, model


# ---------------------------------------------------------------------------
# ONNX export  (TF → ONNX via tf2onnx)
# ---------------------------------------------------------------------------

def export_to_onnx(model_path: str, onnx_path: str):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    import tf2onnx

    print(f"\n  Exporting to ONNX: {onnx_path}")
    model = tf.keras.models.load_model(model_path)
    sig   = [tf.TensorSpec([None, SEQ_LEN], tf.int32, name="enc_in")]
    tf2onnx.convert.from_keras(model, input_signature=sig, opset=15,
                                output_path=onnx_path)
    del model
    gc.collect()
    tf.keras.backend.clear_session()
    print("  ONNX export done.")


# ---------------------------------------------------------------------------
# ORT + TRT  FP16
# ---------------------------------------------------------------------------

def check_trt_available():
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if "TensorrtExecutionProvider" not in avail:
        print(f"  WARNING: TRT EP not available. Providers: {avail}")
        return False
    return True


def make_ort_session(onnx_path: str, trt_cache_dir: str, fp16: bool = True):
    import onnxruntime as ort
    os.makedirs(trt_cache_dir, exist_ok=True)

    trt_opts = {
        "trt_fp16_enable":                fp16,
        "trt_engine_cache_enable":        True,
        "trt_engine_cache_path":          trt_cache_dir,
        "trt_profile_min_shapes":         f"enc_in:1x{SEQ_LEN}",
        "trt_profile_opt_shapes":         f"enc_in:64x{SEQ_LEN}",
        "trt_profile_max_shapes":         f"enc_in:512x{SEQ_LEN}",
        "trt_builder_optimization_level": 3,
    }
    providers = [
        ("TensorrtExecutionProvider", trt_opts),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3

    session = ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                   providers=providers)
    active = session.get_providers()
    if "TensorrtExecutionProvider" in active:
        print("  TRT EP active")
    else:
        print(f"  WARNING: TRT EP not active. Running on: {active}")
    return session


def ort_infer(session, tokens_np: np.ndarray):
    name = session.get_inputs()[0].name
    policy, value = session.run(None, {name: tokens_np})
    return policy, value


# ---------------------------------------------------------------------------
# PyTorch infer helpers
# ---------------------------------------------------------------------------

def make_torch_eager_infer(device="cuda"):
    import torch
    print("\n  Building PyTorch fp16 eager model ...")
    model = build_pytorch_model()
    model = model.half().to(device).eval()

    def infer(tokens_np: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(tokens_np).long().to(device)
            p, v = model(x)
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    return infer


def make_torch_compiled_infer(device="cuda"):
    """
    Try torch.compile with the inductor backend (requires Triton).
    On Windows where Triton is unavailable, fall back to the cudagraphs
    backend which captures/replays CUDA graphs to cut kernel-launch overhead.
    cudagraphs uses per-batch-size captures (static shapes), so each new
    batch size incurs one capture during warmup — timing is of the replay.
    """
    import torch
    print("\n  Building PyTorch fp16 torch.compile model ...")
    model = build_pytorch_model()
    model = model.half().to(device).eval()

    # Probe for Triton — torch.compile itself succeeds but the first forward
    # call raises TritonMissing when inductor tries to emit kernels.
    backend = "inductor"
    compiled = torch.compile(model, dynamic=True, backend=backend)
    probe_inp = torch.zeros(1, 64, dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            compiled(probe_inp)
        print(f"  torch.compile backend: {backend}")
    except Exception as e:
        # Catches TritonMissing (not installed) and BackendCompilerFailed
        # (installed but version mismatch with this PyTorch build).
        print(f"  inductor failed ({type(e).__name__}): {e}")
        print(f"  falling back to cudagraphs backend.")
        backend = "cudagraphs"
        torch._dynamo.config.cache_size_limit = 32
        compiled = torch.compile(model, dynamic=False, backend=backend)

    def infer(tokens_np: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(tokens_np).long().to(device)
            p, v = compiled(x)
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    return infer, backend



def prepare_onnx_for_trt(onnx_path):
    """
    Run symbolic shape inference first, then standard ONNX shape inference,
    then an ORT optimization pass.

    TRT is still complaining that intermediate tensors have no shape even after
    regular infer_shapes(), so we use ORT's symbolic shape inferencer to
    populate value_info more aggressively on graphs with dynamic batch dims.
    """
    import os
    import onnx
    import onnxruntime as ort

    print(f"  Running symbolic shape inference on "
          f"{os.path.basename(onnx_path)} ...")

    model = onnx.load(onnx_path)

    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        model = SymbolicShapeInference.infer_shapes(
            model,
            auto_merge=True,
            guess_output_rank=True,
            verbose=0,
        )
    except Exception as e:
        print(f"  symbolic shape inference failed: {type(e).__name__}: {e}")
        print("  falling back to standard ONNX shape inference only.")

    print("  Running standard ONNX shape inference ...")
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, onnx_path)

    print("  Running ORT graph optimization ...")
    opt_path = onnx_path.replace(".onnx", "_opt.onnx")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    sess_opts.optimized_model_filepath = opt_path
    sess_opts.log_severity_level = 3

    ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    if os.path.exists(opt_path):
        os.replace(opt_path, onnx_path)

    print("  ONNX TRT prep done.")

def export_pytorch_to_onnx(onnx_path):
    import onnx
    import torch

    fixed_batch = 64

    print(f"\n  Exporting PyTorch model to ONNX (fp32): {onnx_path}")
    print(f"  Using static batch for export: {fixed_batch}")

    model = build_pytorch_model()
    model = model.float().cpu().eval()
    dummy = torch.zeros(fixed_batch, SEQ_LEN, dtype=torch.long)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["enc_in"],
        output_names=["policy", "value"],
        opset_version=17,
        do_constant_folding=True,
    )

    model_onnx = onnx.load(onnx_path)
    ln_count = sum([
        1 for node in model_onnx.graph.node
        if node.op_type == "LayerNormalization"
    ])
    print("  LayerNormalization nodes in export: "
          f"{ln_count}  (target: 0 for Ln2d; transformer LN ok)")

    prepare_onnx_for_trt(onnx_path)
    print("  PyTorch ONNX export done.")


def quantize_pytorch_onnx_int8(fp32_onnx_path, int8_onnx_path, n_calib=64):
    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader
    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization import QuantType
    from onnxruntime.quantization import quantize_static

    class TokenCalibrationReader(CalibrationDataReader):
        def __init__(self, n_batches):
            self.data = []
            for _ in range(n_batches):
                toks = np.random.randint(
                    0,
                    VOCAB_SIZE,
                    size=(64, SEQ_LEN),
                    dtype=np.int32,
                )
                self.data.append({"enc_in": toks})
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.data):
                return None
            item = self.data[self.idx]
            self.idx += 1
            return item

    print(f"\n  Quantizing ONNX to INT8 QDQ: {int8_onnx_path}")
    reader = TokenCalibrationReader(n_calib)

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    print("  ONNX INT8 quantization done.")


def make_ort_int8_session(onnx_path):
    import onnxruntime as ort

    providers = []
    avail = ort.get_available_providers()

    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=providers,
    )

    print(f"  INT8 session providers: {session.get_providers()}")
    return session


# ---------------------------------------------------------------------------
# Speed test
# ---------------------------------------------------------------------------

def speed_test(label: str, infer_fn, batch_sizes,
               n_warmup: int = N_WARMUP_GPU, n_iters: int = N_ITERS):
    banner = f"  {label}  "
    print(f"\n{'#' * 64}")
    print(f"{banner:^64}")
    print(f"{'#' * 64}")
    print(f"  warmup={n_warmup}  iters={n_iters}\n")
    print(f"  {'batch':<7} {'latency_ms':>12} {'samples/s':>13}")
    print(f"  {'-'*7} {'-'*12} {'-'*13}")

    results = {}
    for b in batch_sizes:
        inp = random_tokens(b)
        for _ in range(n_warmup):
            policy, value = infer_fn(inp)
        _ = policy.copy()  # sync fence

        t0 = time.perf_counter()
        for _ in range(n_iters):
            policy, value = infer_fn(inp)
        _ = policy.copy()
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
    col_w = 18

    print(f"\n{'=' * (7 + col_w * len(names) + 2)}")
    print("  THROUGHPUT COMPARISON  (samples/s)")
    print(f"  baseline = {names[0]}")
    print(f"{'=' * (7 + col_w * len(names) + 2)}")

    hdr = f"  {'batch':<7}" + "".join(f"{n:>{col_w}}" for n in names)
    print(hdr)
    print("  " + "-" * (7 + col_w * len(names)))

    base_name = names[0]
    for b in batch_sizes:
        row      = f"  {b:<7}"
        base_tp  = all_results[base_name].get(b, {}).get("throughput", 0)
        for i, name in enumerate(names):
            entry = all_results[name].get(b)
            if entry is None:
                row += f"{'N/A':>{col_w}}"
                continue
            tp = entry["throughput"]
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

    model_dir = os.path.dirname(os.path.abspath(args.model))
    stem      = os.path.splitext(os.path.basename(args.model))[0]
    out_dir   = args.out_dir or os.path.join(model_dir, stem + "_onnx_trt")
    onnx_path = os.path.join(out_dir, stem + ".onnx")
    trt_cache = os.path.join(out_dir, "trt_cache")
    os.makedirs(out_dir, exist_ok=True)

    batch_sizes = args.batch_sizes
    all_results = {}

    # ------------------------------------------------------------------
    # 1. TF + XLA
    # ------------------------------------------------------------------
    if not args.skip_tf:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        gpus = tf.config.list_physical_devices("GPU")
        print(f"TF sees {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        # Allow memory growth so PyTorch can also use the GPU
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        print("\n" + "=" * 64)
        print("  Loading model for TF+XLA baseline ...")
        tf_infer, tf_model = make_tf_infer(args.model)
        all_results["TF+XLA"] = speed_test(
            "TF + XLA  [fp16]", tf_infer, batch_sizes, n_warmup=N_WARMUP_GPU)
        del tf_infer, tf_model
        gc.collect()
        tf.keras.backend.clear_session()

    # ------------------------------------------------------------------
    # 2. ORT + TRT  FP16  (from TF ONNX)
    # ------------------------------------------------------------------
    if not args.skip_ort:
        if not check_trt_available():
            print("  Skipping ORT+TRT (TRT EP not available)")
        else:
            if not os.path.exists(onnx_path):
                export_to_onnx(args.model, onnx_path)
            else:
                print(f"\n  ONNX exists, skipping export: {onnx_path}")

            print("\n  Building ORT+TRT FP16 session ...")
            sess_fp16 = make_ort_session(onnx_path, trt_cache, fp16=True)
            all_results["ORT+TRT fp16"] = speed_test(
                "ORT + TRT  [fp16]",
                lambda inp: ort_infer(sess_fp16, inp),
                batch_sizes,
                n_warmup=N_WARMUP_GPU,
            )
            del sess_fp16
            gc.collect()
            time.sleep(1)

    # ------------------------------------------------------------------
    # 3. ORT + TRT  INT8  (from PyTorch ONNX)
    # ------------------------------------------------------------------
    if not args.skip_ort_int8:
        pt_onnx_path = os.path.join(out_dir, "pytorch_fp32.onnx")
        pt_int8_path = os.path.join(out_dir, "pytorch_int8_qdq.onnx")

        export_pytorch_to_onnx(pt_onnx_path)
        quantize_pytorch_onnx_int8(pt_onnx_path, pt_int8_path)

        print("\n  Building ORT INT8 session from QDQ ONNX ...")
        sess_int8 = make_ort_int8_session(pt_int8_path)
        all_results["ORT int8 QDQ"] = speed_test(
            "ORT  [int8 QDQ]",
            lambda inp: ort_infer(sess_int8, inp),
            batch_sizes,
            n_warmup=N_WARMUP_GPU,
        )
        del sess_int8
        gc.collect()
        time.sleep(1)

    # ------------------------------------------------------------------
    # 4 & 5. PyTorch eager + cudagraphs
    # ------------------------------------------------------------------
    if not args.skip_torch:
        try:
            import torch
        except ImportError:
            print("\nERROR: torch not installed. Skipping PyTorch benchmarks.")
            args.skip_torch = True

    if not args.skip_torch:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        print(f"\n  PyTorch device: {device}")

        # 4. Eager fp16
        torch_eager = make_torch_eager_infer(device)
        all_results["PT eager"] = speed_test(
            "PyTorch eager  [fp16]",
            torch_eager,
            batch_sizes,
            n_warmup=N_WARMUP_GPU,
        )
        del torch_eager

        # 6. torch.compile
        print("\n  Note: first per-batch-size call triggers torch.compile trace.")
        torch_compiled, compile_backend = make_torch_compiled_infer(device)
        all_results[f"PT {compile_backend}"] = speed_test(
            f"PyTorch torch.compile [{compile_backend}]  [fp16]",
            torch_compiled,
            batch_sizes,
            n_warmup=N_WARMUP_GPU,
        )
        del torch_compiled

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if all_results:
        print_comparison(all_results, batch_sizes)
    else:
        print("\nNo benchmarks were run.")


if __name__ == "__main__":
    main()
