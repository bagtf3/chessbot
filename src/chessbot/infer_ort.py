import numpy as np
import onnxruntime as ort


def make_ort_infer(onnx_path, engine_cache_dir, fp16, max_bs, vscale):
    """
    ORT/TRT inference wrapper. Returns (session, fwd) where fwd matches the
    make_conv_infer interface: fwd((enc_np, _)) -> (logits_np, val_np).
    Softmax, uniform_eps, and prior_clip_max are applied in C++ build_priors.
    """
    import os

    if engine_cache_dir:
        os.makedirs(engine_cache_dir, exist_ok=True)
        engines_before = set(f for f in os.listdir(engine_cache_dir) if f.endswith(".engine"))
        if engines_before:
            print(f"[ort] {len(engines_before)} cached engine(s) found — expecting cache hit (~20s)")
        else:
            print(f"[ort] no cached engines — first build will take ~10-15 min")
        print(f"[ort] onnx: {onnx_path}")
        print(f"[ort] cache dir: {engine_cache_dir}")
    else:
        engines_before = set()
        print("[ort] WARNING: no TRT engine cache dir — engine rebuilt every session")
        print(f"[ort] onnx: {onnx_path}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": engine_cache_dir,
                "trt_fp16_enable": fp16,
                "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
                "trt_profile_min_shapes": "enc_in:1x64",
                "trt_profile_opt_shapes": "enc_in:64x64",
                "trt_profile_max_shapes": f"enc_in:{max_bs}x64",
            },
        ),
        "CUDAExecutionProvider",
    ]

    session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)

    if engine_cache_dir:
        engines_after = set(f for f in os.listdir(engine_cache_dir) if f.endswith(".engine"))
        new_engines = engines_after - engines_before
        if new_engines:
            print(f"[ort] engine compiled and cached: {next(iter(new_engines))}")
        else:
            print(f"[ort] cache hit confirmed — no new engine files written")

    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type
    np_dtype = np.int64 if "int64" in input_type else np.int32

    def base_fwd(pair):
        enc_np = pair[0].astype(np_dtype)
        outs   = session.run(None, {input_name: enc_np})
        logits = outs[0].astype(np.float32)   # [B, 4288]
        value  = np.clip(outs[1].astype(np.float32) * vscale, -1.0, 1.0)
        return logits, value

    if max_bs is None:
        return session, base_fwd

    def fwd(pair):
        enc_np = pair[0]
        B = int(enc_np.shape[0])
        if B <= max_bs:
            return base_fwd(pair)

        logits_parts, val_parts = [], []
        for i in range(0, B, max_bs):
            j = min(i + max_bs, B)
            lg, v = base_fwd((enc_np[i:j],))
            logits_parts.append(lg)
            val_parts.append(v)

        return np.concatenate(logits_parts, axis=0), np.concatenate(val_parts, axis=0)

    return session, fwd
