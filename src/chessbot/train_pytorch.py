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


def load_pt_model(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    raise ValueError(
        f"[pytorch] unrecognized checkpoint format in {path}. "
        "Expected a saved nn.Module or a dict with key 'model'."
    )


def save_pt_model(model, path):
    torch.save(model, path)


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


def export_pt_to_onnx(model, onnx_path):
    device = next(model.parameters()).device
    model.eval()
    dummy = torch.zeros(1, 64, dtype=torch.long, device=device)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["enc_in"],
        output_names=["policy_logits", "value_out"],
        dynamic_axes={
            "enc_in":        {0: "batch"},
            "policy_logits": {0: "batch"},
            "value_out":     {0: "batch"},
        },
        opset_version=17,
    )
    print(f"[pytorch] ONNX export → {onnx_path}")
