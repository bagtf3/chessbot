import numpy as np
import onnxruntime as ort


def make_ort_infer(onnx_path, engine_cache_dir, fp16, max_bs, uniform_eps, prior_clip_max, vscale):
    """
    ORT/TRT inference wrapper. Returns (session, fwd) where fwd matches the
    make_conv_infer interface: fwd((enc_np, mask_np)) -> (probs_np, val_np).

    Post-processing (masked softmax, uniform eps, prior clip) is done in numpy,
    mirroring what make_conv_infer does in XLA.
    """
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": engine_cache_dir,
                "trt_fp16_enable": fp16,
            },
        ),
        "CUDAExecutionProvider",
    ]

    session = ort.InferenceSession(onnx_path, providers=providers)

    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type
    np_dtype = np.int64 if "int64" in input_type else np.int32

    eps_small = 1e-12

    def base_fwd(pair):
        enc_np, mask_np = pair
        enc_np = enc_np.astype(np_dtype)

        outs   = session.run(None, {input_name: enc_np})
        logits = outs[0].astype(np.float32)   # [B, 4288]
        value  = outs[1].astype(np.float32)   # [B, 1]

        mask   = mask_np.astype(np.float32)
        logits = np.where(mask > 0.5, logits, -1e9)
        row_max = logits.max(axis=1, keepdims=True)
        exp     = np.exp(logits - row_max) * mask
        sum_exp = exp.sum(axis=1, keepdims=True)
        probs   = np.where(sum_exp > 0, exp / (sum_exp + eps_small), np.zeros_like(exp))

        if uniform_eps > 0.0:
            legal_sum = mask.sum(axis=1, keepdims=True)
            uni   = np.where(legal_sum > 0, mask / (legal_sum + eps_small), np.zeros_like(mask))
            probs = (1.0 - uniform_eps) * probs + uniform_eps * uni

        if prior_clip_max < 1.0:
            probs = np.minimum(probs, prior_clip_max) * mask

        s     = probs.sum(axis=1, keepdims=True)
        probs = np.where(s > eps_small, probs / (s + eps_small), np.zeros_like(probs))
        value = np.clip(value * vscale, -1.0, 1.0)

        return probs, value

    if max_bs is None:
        return session, base_fwd

    def fwd(pair):
        enc_np, mask_np = pair
        B = int(enc_np.shape[0])
        if B <= max_bs:
            return base_fwd(pair)

        probs_parts, val_parts = [], []
        for i in range(0, B, max_bs):
            j = min(i + max_bs, B)
            p, v = base_fwd((enc_np[i:j], mask_np[i:j]))
            probs_parts.append(p)
            val_parts.append(v)

        return np.concatenate(probs_parts, axis=0), np.concatenate(val_parts, axis=0)

    return session, fwd
