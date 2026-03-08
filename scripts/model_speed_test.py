#!/usr/bin/env python3
"""model_speed_test.py

Unified throughput benchmark for Xerces-style architectures.

Backends:
  1. TF + XLA     (fp16, random weights)
  2. PT eager     (fp16, random weights)
  3. PT compiled  (fp16, inductor → cudagraphs fallback on Windows)

Architecture is controlled by --preset plus optional overrides.
Three families are supported, selected automatically by config:
  conv-only      conv_blocks > 0, transformer_layers = 0
  conformer      conv_blocks > 0, transformer_layers > 0
  transformer    conv_blocks = 0, transformer_layers > 0

Named presets:
  2m-conv         ~2M   embedding → 6×ConvResBlock(128) → heads
  16m-conformer   ~16M  active model architecture
  16m-transformer ~16.6M  d_model=384, 12 layers, 6 heads

Usage:
  python model_speed_test.py --preset 16m-conformer
  python model_speed_test.py --preset 2m-conv --batch-sizes 1 8 64 256 512
  python model_speed_test.py --conv-filters 320 --conv-blocks 12 --transformer-layers 0
  python model_speed_test.py --preset 16m-conformer --save /path/to/model.h5
"""

import os
import gc
import time
import argparse

# ---------------------------------------------------------------------------
# XLA libdevice fix — must precede TF import
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
N_WARMUP   = 50
N_ITERS    = 100

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "2m-conv": dict(
        d_embed=64, conv_filters=128, conv_blocks=6,
        transformer_layers=0, num_heads=0, ff_dim=0, dropout=0.0,
    ),
    "16m-conformer": dict(
        d_embed=128, conv_filters=256, conv_blocks=10,
        transformer_layers=5, num_heads=8, ff_dim=1024, dropout=0.025,
    ),
    "16m-transformer": dict(
        d_embed=384, conv_filters=0, conv_blocks=0,
        transformer_layers=12, num_heads=6, ff_dim=1024, dropout=0.1,
    ),
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="Named architecture preset (can be overridden by individual flags)")
    p.add_argument("--d-embed",            type=int,   default=None)
    p.add_argument("--conv-filters",       type=int,   default=None)
    p.add_argument("--conv-blocks",        type=int,   default=None)
    p.add_argument("--transformer-layers", type=int,   default=None)
    p.add_argument("--num-heads",          type=int,   default=None)
    p.add_argument("--ff-dim",             type=int,   default=None)
    p.add_argument("--dropout",            type=float, default=None)
    p.add_argument("--skip-tf",    action="store_true", help="Skip TF+XLA")
    p.add_argument("--skip-torch", action="store_true", help="Skip PT eager + compiled")
    p.add_argument("--with-ort",   action="store_true",
                   help="Also benchmark ORT + CUDA EP with built-in graph optimizer")
    p.add_argument("--save", default=None, metavar="PATH",
                   help="Save compiled TF model to this .h5 path")
    p.add_argument("--batch-sizes", nargs="+", type=int,
                   default=[1, 4, 8, 16, 32, 64, 128, 256])
    return p.parse_args()


def resolve_cfg(args):
    base = dict(PRESETS[args.preset]) if args.preset else dict(PRESETS["16m-conformer"])
    overrides = {
        "d_embed":            args.d_embed,
        "conv_filters":       args.conv_filters,
        "conv_blocks":        args.conv_blocks,
        "transformer_layers": args.transformer_layers,
        "num_heads":          args.num_heads,
        "ff_dim":             args.ff_dim,
        "dropout":            args.dropout,
    }
    for k, v in overrides.items():
        if v is not None:
            base[k] = v
    base["vocab_size"] = VOCAB_SIZE
    return base


def describe_cfg(cfg):
    cb = cfg["conv_blocks"]
    tl = cfg["transformer_layers"]
    W  = cfg["conv_filters"] if cb > 0 else cfg["d_embed"]
    if   cb > 0 and tl > 0: arch = "conformer"
    elif cb > 0:             arch = "conv-only"
    else:                    arch = "pure-transformer"
    print(f"\n  Architecture : {arch}")
    print(f"  d_embed={cfg['d_embed']}  conv_filters={cfg['conv_filters']}  conv_blocks={cb}")
    if tl > 0:
        hd = W // cfg["num_heads"] if cfg["num_heads"] > 0 else "N/A"
        print(f"  transformer_layers={tl}  num_heads={cfg['num_heads']}  "
              f"ff_dim={cfg['ff_dim']}  head_dim={hd}")
    print(f"  dropout={cfg['dropout']}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def random_tokens(batch_size):
    return np.random.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), dtype=np.int32)


# ---------------------------------------------------------------------------
# TF model — unified across all three architecture families
# ---------------------------------------------------------------------------

def build_tf_model(cfg):
    import tensorflow as tf
    from tensorflow.keras import layers, Input, Model
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    de = cfg["d_embed"]
    cf = cfg["conv_filters"]
    cb = cfg["conv_blocks"]
    tl = cfg["transformer_layers"]
    nh = cfg["num_heads"]
    fd = cfg["ff_dim"]
    dr = cfg["dropout"]
    W  = cf if cb > 0 else de   # working width through transformer / heads

    inp = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="enc_in")
    x = layers.Embedding(VOCAB_SIZE, de, name="token_emb")(inp)   # (B, 64, de)

    # ---- Conv frontend (optional) ----
    if cb > 0:
        x = layers.Reshape((8, 8, de), name="to_2d")(x)
        if de != cf:
            x = layers.Conv2D(cf, 1, use_bias=False, padding="same", name="conv_proj")(x)
            x = layers.LayerNormalization(axis=-1, name="ln_proj")(x)
            x = layers.LeakyReLU(0.01, name="lrelu_proj")(x)
        for i in range(cb):
            h = layers.Conv2D(cf, 3, use_bias=False, padding="same", name=f"c{i}_conv1")(x)
            h = layers.LeakyReLU(0.01, name=f"c{i}_lrelu1")(h)
            h = layers.Conv2D(cf, 3, use_bias=False, padding="same", name=f"c{i}_conv2")(h)
            h = layers.LayerNormalization(axis=-1, name=f"c{i}_ln")(h)
            x = layers.LeakyReLU(0.01, name=f"c{i}_out")(x + h)
        if tl > 0:
            x = layers.Reshape((64, cf), name="to_seq")(x)   # (B, 64, cf)

    # ---- Transformer (optional) ----
    if tl > 0:
        pos = layers.Embedding(64, W, name="pos_emb")(
            tf.keras.backend.arange(0, 64, dtype="int32")
        )
        x = x + pos
        for i in range(tl):
            r = x
            x = layers.LayerNormalization(axis=-1, name=f"t{i}_ln1")(x)
            x = layers.MultiHeadAttention(
                num_heads=nh, key_dim=W // nh, dropout=0.0, name=f"t{i}_mha"
            )(x, x)
            x = layers.Dropout(dr, name=f"t{i}_attn_drop")(x)
            x = x + r
            r = x
            x = layers.LayerNormalization(axis=-1, name=f"t{i}_ln2")(x)
            x = layers.Dense(fd, activation="gelu", name=f"t{i}_ff1")(x)
            x = layers.Dropout(dr, name=f"t{i}_ff_drop")(x)
            x = layers.Dense(W, name=f"t{i}_ff2")(x)
            x = x + r
        if cb > 0:
            x = layers.Reshape((8, 8, W), name="from_seq")(x)   # back to spatial

    # ---- Conv mix (spatial path only) ----
    if cb > 0:
        x = layers.Conv2D(cf, 1, use_bias=False, padding="same", name="conv_mix")(x)
        x = layers.LayerNormalization(axis=-1, name="ln_mix")(x)
        x = layers.LeakyReLU(0.01, name="lrelu_mix")(x)

    # ---- Policy head ----
    if cb > 0:
        policy = layers.Conv2D(67, 1, padding="same", name="policy_conv")(x)
        policy = layers.Reshape((SEQ_LEN * 67,), name="policy_logits")(policy)
    else:
        policy = layers.Dense(67, name="policy_dense")(x)          # (B, 64, 67)
        policy = layers.Reshape((SEQ_LEN * 67,), name="policy_logits")(policy)

    # ---- Value head ----
    if cb > 0:
        v = layers.Conv2D(64, 3, use_bias=False, padding="same", name="v_conv")(x)
        v = layers.LayerNormalization(axis=-1, name="v_ln")(v)
        v = layers.LeakyReLU(0.01, name="v_lrelu")(v)
        v = layers.GlobalAveragePooling2D(name="v_gap")(v)
    else:
        v = layers.GlobalAveragePooling1D(name="v_gap")(x)

    v = layers.Dense(256, activation="relu", name="v_fc1")(v)
    v = layers.Dense(128, activation="relu", name="v_fc2")(v)
    value = layers.Dense(1, activation="tanh", dtype="float32", name="value_out")(v)

    model = Model(inp, [policy, value])
    print(f"  TF param count: {model.count_params():,}")

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out": "mse",
    }
    loss_weights = {"policy_logits": 1.0, "value_out": 1.0}
    model.compile(optimizer=opt, loss=loss_dict, loss_weights=loss_weights)
    model._default_opt       = opt
    model._default_loss_dict = loss_dict
    return model


def make_tf_infer(cfg, xla=True):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    model = build_tf_model(cfg)

    @tf.function(
        input_signature=[tf.TensorSpec([None, SEQ_LEN], tf.int32)],
        jit_compile=xla,
    )
    def _fwd(x):
        return model(x, training=False)

    _ = _fwd(tf.constant(random_tokens(64)))   # trigger trace / XLA compile

    def infer(tokens_np):
        out = _fwd(tf.constant(tokens_np))
        return out[0].numpy(), out[1].numpy()

    return infer, model


# ---------------------------------------------------------------------------
# PyTorch model — unified across all three architecture families
# ---------------------------------------------------------------------------

def build_pytorch_model(cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    de = cfg["d_embed"]
    cf = cfg["conv_filters"]
    cb = cfg["conv_blocks"]
    tl = cfg["transformer_layers"]
    nh = cfg["num_heads"]
    fd = cfg["ff_dim"]
    dr = cfg["dropout"]
    W  = cf if cb > 0 else de

    class Ln2d(nn.Module):
        def __init__(self, ch, eps=1e-5):
            super().__init__()
            self.w   = nn.Parameter(torch.ones(ch))
            self.b   = nn.Parameter(torch.zeros(ch))
            self.eps = eps
        def forward(self, x):
            c = x - x.mean(1, keepdim=True)
            return c * torch.rsqrt((c*c).mean(1, keepdim=True) + self.eps) \
                   * self.w.view(1,-1,1,1) + self.b.view(1,-1,1,1)

    class ConvResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1  = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.lrelu1 = nn.LeakyReLU(0.01, inplace=True)
            self.conv2  = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.ln2    = Ln2d(ch)
            self.lrelu  = nn.LeakyReLU(0.01, inplace=True)
        def forward(self, x):
            h = self.lrelu1(self.conv1(x))
            return self.lrelu(x + self.ln2(self.conv2(h)))

    class TransformerLayer(nn.Module):
        def __init__(self, d, heads, ffd, drop):
            super().__init__()
            self.ln1   = nn.LayerNorm(d)
            self.attn  = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
            self.drop1 = nn.Dropout(drop)
            self.ln2   = nn.LayerNorm(d)
            self.ff1   = nn.Linear(d, ffd)
            self.drop2 = nn.Dropout(drop)
            self.ff2   = nn.Linear(ffd, d)
        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop1(h)
            h = self.ff2(self.drop2(F.gelu(self.ff1(self.ln2(x)))))
            return x + h

    class UnifiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._cb = cb
            self._tl = tl
            self._de = de
            self._cf = cf
            self._W  = W

            self.token_emb = nn.Embedding(VOCAB_SIZE, de)

            if cb > 0:
                self._has_proj = (de != cf)
                if de != cf:
                    self.conv_proj  = nn.Conv2d(de, cf, 1, bias=False)
                    self.ln_proj    = Ln2d(cf)
                    self.lrelu_proj = nn.LeakyReLU(0.01, inplace=True)
                self.conv_blocks = nn.ModuleList([ConvResBlock(cf) for _ in range(cb)])
                self.conv_mix    = nn.Conv2d(cf, cf, 1, bias=False)
                self.ln_mix      = Ln2d(cf)
                self.lrelu_mix   = nn.LeakyReLU(0.01, inplace=True)
                self.policy_conv = nn.Conv2d(cf, 67, 1)
                self.v_conv      = nn.Conv2d(cf, 64, 3, padding=1, bias=False)
                self.v_ln        = Ln2d(64)
                self.v_lrelu     = nn.LeakyReLU(0.01, inplace=True)
                v_in = 64
            else:
                self._has_proj   = False
                self.policy_proj = nn.Linear(W, 67)
                v_in = W

            if tl > 0:
                self.pos_emb     = nn.Embedding(64, W)
                self.transformer = nn.ModuleList(
                    [TransformerLayer(W, nh, fd, dr) for _ in range(tl)]
                )

            self.v_fc1 = nn.Linear(v_in, 256)
            self.v_fc2 = nn.Linear(256, 128)
            self.v_out = nn.Linear(128, 1)

        def forward(self, tokens):
            B = tokens.shape[0]
            x = self.token_emb(tokens)                         # (B, 64, de)

            if self._cb > 0:
                x = x.view(B, 8, 8, self._de).permute(0, 3, 1, 2).contiguous()
                if self._has_proj:
                    x = self.lrelu_proj(self.ln_proj(self.conv_proj(x)))
                for blk in self.conv_blocks:
                    x = blk(x)
                if self._tl > 0:
                    x = x.permute(0, 2, 3, 1).reshape(B, 64, self._cf)

            if self._tl > 0:
                pos = self.pos_emb(torch.arange(64, device=tokens.device))
                x = x + pos.unsqueeze(0)
                for layer in self.transformer:
                    x = layer(x)
                if self._cb > 0:
                    x = x.reshape(B, 8, 8, self._cf).permute(0, 3, 1, 2).contiguous()

            if self._cb > 0:
                x      = self.lrelu_mix(self.ln_mix(self.conv_mix(x)))
                policy = self.policy_conv(x).permute(0, 2, 3, 1).reshape(B, SEQ_LEN * 67)
                v      = self.v_lrelu(self.v_ln(self.v_conv(x))).mean(dim=[2, 3])
            else:
                policy = self.policy_proj(x).reshape(B, SEQ_LEN * 67)
                v      = x.mean(dim=1)

            v = F.relu(self.v_fc1(v))
            v = F.relu(self.v_fc2(v))
            v = torch.tanh(self.v_out(v))
            return policy, v

    import torch
    model = UnifiedModel()
    print(f"  PyTorch param count: {sum(p.numel() for p in model.parameters()):,}")
    return model


def make_torch_eager_infer(cfg, device):
    import torch
    print("\n  Building PyTorch fp16 eager model ...")
    model = build_pytorch_model(cfg).half().to(device).eval()

    def infer(tokens_np):
        with torch.no_grad():
            x = torch.from_numpy(tokens_np).long().to(device)
            p, v = model(x)
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    return infer


def make_torch_compiled_infer(cfg, device):
    import torch
    print("\n  Building PyTorch fp16 torch.compile model ...")
    model = build_pytorch_model(cfg).half().to(device).eval()

    backend = "inductor"
    compiled = torch.compile(model, dynamic=True, backend=backend)
    try:
        with torch.no_grad():
            compiled(torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device))
        print(f"  torch.compile backend: {backend}")
    except Exception as e:
        print(f"  inductor failed ({type(e).__name__}): falling back to cudagraphs")
        backend = "cudagraphs"
        torch._dynamo.config.cache_size_limit = 32
        compiled = torch.compile(model, dynamic=False, backend=backend)

    def infer(tokens_np):
        with torch.no_grad():
            x = torch.from_numpy(tokens_np).long().to(device)
            p, v = compiled(x)
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    return infer, backend


# ---------------------------------------------------------------------------
# ORT + CUDA EP  (ORT built-in graph optimizer, no TRT)
# ---------------------------------------------------------------------------

def export_tf_to_onnx(tf_model, onnx_path):
    import tensorflow as tf
    import tf2onnx

    print(f"\n  Exporting TF model to ONNX: {onnx_path}")
    sig = [tf.TensorSpec([None, SEQ_LEN], tf.int32, name="enc_in")]
    tf2onnx.convert.from_keras(tf_model, input_signature=sig, opset=15,
                                output_path=onnx_path)
    print("  ONNX export done.")


def make_ort_cuda_session(onnx_path):
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3

    avail = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                   providers=providers)
    print(f"  ORT session providers: {session.get_providers()}")
    return session


def ort_infer(session, tokens_np):
    name = session.get_inputs()[0].name
    policy, value = session.run(None, {name: tokens_np})
    return policy, value


# ---------------------------------------------------------------------------
# Speed test
# ---------------------------------------------------------------------------

def speed_test(label, infer_fn, batch_sizes, n_warmup=N_WARMUP, n_iters=N_ITERS):
    print(f"\n{'#' * 64}")
    print(f"  {label}  ".center(64))
    print(f"{'#' * 64}")
    print(f"  warmup={n_warmup}  iters={n_iters}\n")
    print(f"  {'batch':<7} {'latency_ms':>12} {'samples/s':>13}")
    print(f"  {'-'*7} {'-'*12} {'-'*13}")

    results = {}
    for b in batch_sizes:
        inp = random_tokens(b)
        for _ in range(n_warmup):
            p, v = infer_fn(inp)
        _ = p.copy()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            p, v = infer_fn(inp)
        _ = p.copy()
        t1 = time.perf_counter()

        avg_s = (t1 - t0) / n_iters
        tput  = b / avg_s
        print(f"  {b:<7} {avg_s*1000:>12.2f} {tput:>13.0f}")
        results[b] = {"latency_ms": avg_s * 1000, "throughput": tput}

    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(all_results):
    names = list(all_results.keys())
    col_w = 20
    all_bs = sorted(set(b for r in all_results.values() for b in r))

    print(f"\n{'=' * (9 + col_w * len(names))}")
    print("  THROUGHPUT COMPARISON  (samples/s)")
    print(f"  baseline = {names[0]}")
    print(f"{'=' * (9 + col_w * len(names))}")
    print(f"  {'batch':<7}" + "".join(f"{n:>{col_w}}" for n in names))
    print("  " + "-" * (7 + col_w * len(names)))

    base = names[0]
    for b in all_bs:
        row     = f"  {b:<7}"
        base_tp = all_results[base].get(b, {}).get("throughput", 0)
        for i, name in enumerate(names):
            entry = all_results[name].get(b)
            if entry is None:
                row += f"{'N/A':>{col_w}}"
                continue
            tp = entry["throughput"]
            if i == 0:
                row += f"{tp:>{col_w}.0f}"
            else:
                pct  = (tp / base_tp - 1) * 100 if base_tp > 0 else 0
                row += f"{f'{tp:.0f} ({pct:+.0f}%)':>{col_w}}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    cfg     = resolve_cfg(args)
    describe_cfg(cfg)

    all_results = {}

    # ------------------------------------------------------------------
    # 1. TF + XLA
    # ------------------------------------------------------------------
    if not args.skip_tf:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        gpus = tf.config.list_physical_devices("GPU")
        print(f"\nTF sees {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        print("  Building TF model ...")
        tf_infer, tf_model = make_tf_infer(cfg)

        if args.save:
            save_path = args.save
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            tf_model.save(save_path)
            print(f"  Saved → {save_path}")

        onnx_path = None
        if args.with_ort:
            import tempfile
            onnx_path = os.path.join(tempfile.gettempdir(), "xerces_bench.onnx")
            export_tf_to_onnx(tf_model, onnx_path)

        all_results["TF+XLA"] = speed_test("TF + XLA  [fp16]", tf_infer, args.batch_sizes)
        del tf_infer, tf_model
        gc.collect()
        tf.keras.backend.clear_session()

        if args.with_ort and onnx_path:
            print("\n  Building ORT + CUDA EP session ...")
            sess_ort = make_ort_cuda_session(onnx_path)
            all_results["ORT CUDA"] = speed_test(
                "ORT + CUDA EP  [fp32, ORT optimizer]",
                lambda inp: ort_infer(sess_ort, inp),
                args.batch_sizes,
            )
            del sess_ort
            gc.collect()

    # ------------------------------------------------------------------
    # 2 & 3. PyTorch eager + compiled
    # ------------------------------------------------------------------
    if not args.skip_torch:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\n  PyTorch device: {device}")
        except ImportError:
            print("\nERROR: torch not installed. Skipping PyTorch benchmarks.")
            args.skip_torch = True

    if not args.skip_torch:
        eager = make_torch_eager_infer(cfg, device)
        all_results["PT eager"] = speed_test(
            "PyTorch eager  [fp16]", eager, args.batch_sizes
        )
        del eager

        compiled, be = make_torch_compiled_infer(cfg, device)
        all_results[f"PT {be}"] = speed_test(
            f"PyTorch {be}  [fp16]", compiled, args.batch_sizes
        )
        del compiled

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if all_results:
        print_comparison(all_results)
    else:
        print("\nNo benchmarks were run.")


if __name__ == "__main__":
    main()
