import os
import numpy as np
import torch
import torch.nn.functional as F


def enforce_pytorch_gpu_or_die(max_tries=5, sleep_s=1.0):
    import time
    for _ in range(max_tries):
        if torch.cuda.is_available():
            print(f"[pytorch] GPU: {torch.cuda.get_device_name(0)}")
            return
        time.sleep(sleep_s)
    raise RuntimeError("PyTorch sees no CUDA GPU. Refusing to run on CPU.")


_FALLBACK_ARCH = "16m-conformer-interweaved"


def load_pt_model(path):
    """Returns (model, arch_name)."""
    from chessbot.model import PT_BUILDERS
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        state_dict = obj["model"]
        if isinstance(state_dict, torch.nn.Module):
            return state_dict, _FALLBACK_ARCH
        arch = obj.get("arch", _FALLBACK_ARCH)
        model = PT_BUILDERS[arch]()
        model.load_state_dict(state_dict)
        return model, arch
    if isinstance(obj, torch.nn.Module):
        return obj, _FALLBACK_ARCH
    raise ValueError(
        f"[pytorch] unrecognized checkpoint format in {path}. "
        "Expected a dict with key 'model'."
    )


def save_pt_model(model, path, arch):
    torch.save({"model": model.state_dict(), "arch": arch}, path)


def train_pt_model(model, X, M, P, Y_value, vwht, pwht, cfg, args):
    device = torch.device("cuda")
    model  = model.to(device).train()
    opt    = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    n          = len(X)
    X_t      = torch.from_numpy(X).long().to(device)
    P_t      = torch.from_numpy(P).float().to(device)
    Y_t      = torch.from_numpy(Y_value).float().unsqueeze(1).to(device)
    vwht_t   = torch.from_numpy(vwht).float().to(device)
    pwht_t   = torch.from_numpy(pwht).float().to(device)

    for epoch in range(args.epochs):
        idx        = torch.randperm(n, device=device)
        total_loss = 0.0
        steps      = 0

        for start in range(0, n, args.batch_size):
            batch_idx = idx[start:start + args.batch_size]
            xb  = X_t[batch_idx]
            pb  = P_t[batch_idx]
            yb  = Y_t[batch_idx]
            vwb = vwht_t[batch_idx]
            pwb = pwht_t[batch_idx]

            policy_logits, value_out = model(xb)

            log_probs   = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(pb * log_probs).sum(dim=-1)   # [B]
            policy_loss = (policy_loss * pwb).mean()

            value_loss  = F.mse_loss(value_out, yb, reduction="none").squeeze(1)
            value_loss  = (value_loss * vwb).mean()

            loss = policy_loss + value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            steps += 1

        print(f"[pytorch] epoch {epoch}  loss={total_loss / max(1, steps):.4f}")


def make_pt_infer(model, max_bs, uniform_eps, prior_clip_max, vscale):
    """
    Returns (model, fwd) where fwd((enc_np, mask_np)) -> (probs_np, val_np).
    Matches the make_conv_infer / make_ort_infer interface.
    """
    device = torch.device("cuda")
    model = model.to(device).half().eval()
    eps_small = 1e-12

    ue      = torch.tensor(uniform_eps,  device=device, dtype=torch.float32)
    pcm     = torch.tensor(prior_clip_max, device=device, dtype=torch.float32)
    vscale_t = torch.tensor(vscale,       device=device, dtype=torch.float32)
    eps_t   = torch.tensor(eps_small,     device=device, dtype=torch.float32)

    def base_fwd(pair):
        enc_np, mask_np = pair
        mask = torch.from_numpy(mask_np.astype(np.float32)).to(device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            enc_t = torch.from_numpy(enc_np).long().to(device)
            logits, value = model(enc_t)

        logits = logits.float()
        logits = torch.where(mask > 0.5, logits, torch.tensor(-1e9, device=device))
        row_max = logits.max(dim=1, keepdim=True).values
        exp     = torch.exp(logits - row_max) * mask
        sum_exp = exp.sum(dim=1, keepdim=True)
        probs   = exp / (sum_exp + eps_t)

        if uniform_eps > 0.0:
            legal_sum = mask.sum(dim=1, keepdim=True)
            uni   = mask / (legal_sum + eps_t)
            probs = (1.0 - ue) * probs + ue * uni

        if prior_clip_max < 1.0:
            probs = torch.minimum(probs, pcm) * mask
            s     = probs.sum(dim=1, keepdim=True)
            probs = probs / (s + eps_t)

        value = (value.float() * vscale_t).clamp(-1.0, 1.0)
        return probs.cpu().numpy(), value.cpu().numpy()

    if max_bs is None:
        return model, base_fwd

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

    return model, fwd


def export_pt_to_onnx(model, onnx_path):
    import onnx
    from onnx import shape_inference

    device = next(model.parameters()).device
    model.eval()

    # Export with int32 input so TRT doesn't warn about INT64 cast.
    # The wrapper casts to int64 internally for the embedding layer.
    class _Int32Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, t):
            return self.m(t.long())

    export_model = _Int32Wrapper(model)
    dummy = torch.zeros(1, 64, dtype=torch.int32, device=device)

    for opset in (14, 16):
        try:
            torch.onnx.export(
                export_model,
                dummy,
                onnx_path,
                input_names=["enc_in"],
                output_names=["policy_logits", "value_out"],
                dynamic_axes={
                    "enc_in":        {0: "batch"},
                    "policy_logits": {0: "batch"},
                    "value_out":     {0: "batch"},
                },
                opset_version=opset,
            )
            print(f"[pytorch] ONNX export → {onnx_path}  (opset {opset})")
            break
        except Exception as e:
            print(f"[pytorch] opset {opset} failed ({type(e).__name__}), retrying...")

    # TRT requires shape info on all intermediate tensors
    proto = onnx.load(onnx_path)
    proto = shape_inference.infer_shapes(proto)
    onnx.save(proto, onnx_path)
    print(f"[pytorch] ONNX shape inference complete")
