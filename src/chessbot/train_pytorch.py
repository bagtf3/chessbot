import os
import time
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


FALLBACK_ARCH = "13m-precond-conformer"


def shorten_path(p):
    parts = p.replace("\\", "/").split("/")
    idx = next((i for i, x in enumerate(parts) if x == "selfplay_runs"), None)
    return "/".join(parts[idx:]) if idx is not None else p


def log_policy_mask_status(model):
    ph = getattr(model, "policy_head", None)
    buf = getattr(ph, "sometimes_legal", None) if ph is not None else None
    if buf is None:
        # newer architectures register never_legal directly on the model
        never = getattr(model, "never_legal", None)
        if never is not None:
            n = int((~never).sum().item())
            if n != 1858:
                print(f"[load_pt_model] never_legal: WARN expected 1858 sometimes-legal got {n}")
            return
        sl_idx = getattr(model, "sl_idx", None)
        if sl_idx is not None:
            n = int(sl_idx.shape[0])
            if n != 1858:
                print(f"[load_pt_model] sl_idx: WARN expected 1858 scatter indices got {n}")
            return
        print("[load_pt_model] policy mask: NOT FOUND (no policy_head.sometimes_legal, never_legal, or sl_idx)")
        return
    n = int(buf.sum().item())
    if n != 1858:
        print(f"[load_pt_model] policy_head.sometimes_legal: WARN expected 1858 got {n}")


def companion_pt(ts_path: str) -> str:
    """Return the state_dict companion path for a .ts file."""
    return ts_path[:-3] + ".pt"


def load_pt_model(path):
    """Returns (model, arch_or_None).
    .ts path: loads ScriptModule directly (no architecture dependency).
              arch read from companion .pt metadata if present.
    .pt path: loads state_dict checkpoint and rebuilds from VARIANTS.
    """
    if path.endswith(".ts"):
        pt_path = companion_pt(path)
        if os.path.exists(pt_path):
            meta = torch.load(pt_path, map_location="cpu")
            if isinstance(meta, dict) and "model" in meta:
                from chessbot.model import PT_BUILDERS, VARIANTS
                arch = meta.get("arch", FALLBACK_ARCH)
                model = PT_BUILDERS[arch](VARIANTS[arch])
                model.load_state_dict(meta["model"], strict=False)
                log_policy_mask_status(model)
                return model, arch
        # fallback: ScriptModule (inference only, no backprop)
        return torch.jit.load(path, map_location="cpu"), None
    from chessbot.model import PT_BUILDERS, VARIANTS
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        state_dict = obj["model"]
        if isinstance(state_dict, torch.nn.Module):
            return state_dict, FALLBACK_ARCH
        arch = obj.get("arch", FALLBACK_ARCH)
        model = PT_BUILDERS[arch](VARIANTS[arch])
        model.load_state_dict(state_dict, strict=False)
        log_policy_mask_status(model)
        return model, arch
    if isinstance(obj, torch.nn.Module):
        return obj, FALLBACK_ARCH
    raise ValueError(f"[pytorch] unrecognized checkpoint format in {path}")


def save_pt_model(model, path, arch=None, opt=None):
    """Save model.
    .ts path: TorchScript trace for inference + companion .pt state_dict for training.
    .pt path: state_dict dict only.
    """
    if path.endswith(".ts"):
        trace_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = trace_device.type == "cuda"
        trace_model = model.to(trace_device).half().eval() if use_fp16 else model.to(trace_device).eval()
        if isinstance(trace_model, torch.jit.ScriptModule):
            torch.jit.save(trace_model, path)
        else:
            from chessbot.model import VARIANTS
            cfg = VARIANTS.get(arch, {})
            if cfg.get("lc0_input", False):
                dtype = torch.float16 if use_fp16 else torch.float32
                dummy = torch.zeros(1, 112, 8, 8, dtype=dtype, device=trace_device)
            elif cfg.get("xc0h_K") is not None:
                K = cfg["xc0h_K"]
                dummy = torch.zeros(1, K * 64 + K + 3, dtype=torch.long, device=trace_device)
            else:
                dummy = torch.zeros(1, 64, dtype=torch.long, device=trace_device)
            with torch.no_grad():
                traced = torch.jit.trace(trace_model, dummy)
            torch.jit.save(traced, path)
        print(f"[pytorch] TorchScript saved -> {shorten_path(path)}")
        pt_path = companion_pt(path)
        payload = {"model": model.state_dict(), "arch": arch}
        torch.save(payload, pt_path)
        print(f"[pytorch] companion weights saved -> {shorten_path(pt_path)}")
        return
    payload = {"model": model.state_dict(), "arch": arch}
    if opt is not None:
        payload["opt"] = opt.state_dict()
    torch.save(payload, path)
    print(f"[pytorch] checkpoint saved -> {path}")



RETRAIN_CLIP_NORM = 20.0


def pt_opt_state_path(model_path: str, run_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(run_dir, "train_ckpts", stem + "_opt_state.pt")


def print_pt_fit_history(epoch_losses, epoch, label=""):
    """epoch_losses: list of dicts with keys 'loss','policy','value' (one per epoch)."""
    if len(epoch_losses) < 2:
        return
    metrics = [("total", "loss"), ("policy", "policy"), ("value", "value")]
    rows = []
    for name, key in metrics:
        start = epoch_losses[0][key]
        end   = epoch_losses[-1][key]
        delta = start - end
        mark  = "*" if delta < 0 else "+"
        rows.append((name, start, end, delta, mark))

    name_w = max(len(r[0]) for r in rows)
    num_w  = 8
    etag   = f"[epoch {epoch:4d}]{(' ' + label) if label else ''}"
    fmt = (
        f"{etag} [model fit] "
        f"{{name:<{name_w}}} : value: {{start:{num_w}.4f}} -> "
        f"{{end:{num_w}.4f}}  delta: {{delta:{num_w}.4f}} {{mark}}"
    )
    for name, start, end, delta, mark in rows:
        print(fmt.format(name=name, start=start, end=end, delta=delta, mark=mark))


def print_pt_grad_stats(epoch_grad_stats, epoch, label=""):
    if not epoch_grad_stats:
        return
    etag    = f"[epoch {epoch:4d}]{(' ' + label) if label else ''}"
    clips   = [gs['gn_clips'] for gs in epoch_grad_stats]
    steps   = [gs['gn_steps'] for gs in epoch_grad_stats]
    means   = [gs['gn_mean']   for gs in epoch_grad_stats]
    medians = [gs['gn_median'] for gs in epoch_grad_stats]
    mins    = [gs['gn_min']    for gs in epoch_grad_stats]
    maxs    = [gs['gn_max']    for gs in epoch_grad_stats]
    breakdown = "  ".join(f"ep{i}: {c}" for i, c in enumerate(clips))
    print(
        f"{etag} [grad stats] "
        f"mean={np.mean(means):.2f}  median={np.mean(medians):.2f}  "
        f"min={min(mins):.2f}  max={max(maxs):.2f}  "
        f"clips={sum(clips)}/{sum(steps)} ({breakdown})"
    )


def retrain_pt(model_path, X, P, Y_wdl, vwht, pwht, cfg, epoch, args,
               label="", timings=None):
    """Load, train 2 epochs at LR/1.5, save. Mirrors retrain_one_model."""
    if timings is None:
        timings = {}
    tag = f"[retrain{(' ' + label) if label else ''}]"
    short_model = os.path.join(
        os.path.basename(os.path.dirname(model_path)),
        os.path.basename(model_path))
    print(f"{tag} loading {short_model}")

    t0 = time.time()
    model, arch = load_pt_model(model_path)
    device = torch.device("cuda")
    model  = model.to(device).train()
    timings['load_model'] = timings.get('load_model', 0.0) + (time.time() - t0)

    n      = len(X)
    if cfg.encoding_type == "lc0":
        # raw uint8 lc0 planes -> float, rule50 /99 at the model-input boundary
        X_t = torch.from_numpy(X).float().to(device)
        X_t[:, 109] /= 99.0
    else:
        X_t = torch.from_numpy(X).long().to(device)
    P_t    = torch.from_numpy(P).float().to(device)
    Y_t    = torch.from_numpy(Y_wdl).float().to(device)
    vwht_t = torch.from_numpy(vwht).float().to(device)
    pwht_t = torch.from_numpy(pwht).float().to(device)

    lr = cfg.learning_rate
    opt = torch.optim.Adam(
        model.parameters(), lr=lr,
        betas=(0.9, cfg.adam_beta2),
        weight_decay=1e-6,
    )
    opt_state_path = pt_opt_state_path(model_path, cfg.run_dir)
    if os.path.exists(opt_state_path):
        try:
            state = torch.load(opt_state_path, map_location="cpu")
            opt.load_state_dict(state)
            for pg in opt.param_groups:
                pg['lr'] = lr
                pg['betas'] = (0.9, cfg.adam_beta2)
                pg['weight_decay'] = 1e-6
            for param_state in opt.state.values():
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        param_state[k] = v.to(device)
            print(f"{tag} restored Adam state")
        except Exception as e:
            print(f"{tag} failed to load Adam state ({e}) - starting fresh")
    else:
        print(f"{tag} no prior Adam state - starting fresh")

    epoch_losses     = []
    epoch_grad_stats = []
    t0 = time.time()
    for ep_idx in range(1):
        idx          = torch.randperm(n, device=device)
        total        = value_total = policy_total = 0.0
        steps        = 0
        grad_norms   = []
        clip_count   = 0

        for start in range(0, n, args.batch_size):
            batch_idx = idx[start:start + args.batch_size]
            xb  = X_t[batch_idx]
            pb  = P_t[batch_idx]
            yb  = Y_t[batch_idx]
            vwb = vwht_t[batch_idx]
            pwb = pwht_t[batch_idx]

            policy_logits, value_out = model(xb)

            log_probs   = F.log_softmax(policy_logits, dim=-1)
            policy_loss = (-(pb * log_probs).sum(dim=-1) * pwb).mean()

            log_wdl    = F.log_softmax(value_out, dim=-1)
            value_loss = (-(yb * log_wdl).sum(dim=-1) * vwb).mean()

            loss = cfg.policy_loss_weight * policy_loss + cfg.value_loss_weight * value_loss
            opt.zero_grad()
            loss.backward()
            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), RETRAIN_CLIP_NORM).item()
            grad_norms.append(raw_norm)
            if raw_norm > RETRAIN_CLIP_NORM:
                clip_count += 1
            opt.step()

            total        += loss.item()
            policy_total += policy_loss.item()
            value_total  += value_loss.item()
            steps        += 1

        gn = np.array(grad_norms)
        epoch_losses.append({
            'loss':   total        / max(1, steps),
            'policy': policy_total / max(1, steps),
            'value':  value_total  / max(1, steps),
        })
        epoch_grad_stats.append({
            'gn_mean':   float(gn.mean()),
            'gn_median': float(np.median(gn)),
            'gn_min':    float(gn.min()),
            'gn_max':    float(gn.max()),
            'gn_clips':  clip_count,
            'gn_steps':  steps,
        })

    timings['fit'] = timings.get('fit', 0.0) + (time.time() - t0)
    print_pt_fit_history(epoch_losses, epoch, label=label)
    print_pt_grad_stats(epoch_grad_stats, epoch, label=label)

    os.makedirs(os.path.dirname(opt_state_path), exist_ok=True)
    torch.save(opt.state_dict(), opt_state_path)
    print(f"{tag} saved Adam state")

    t0 = time.time()
    if model_path.endswith(".ts"):
        bak_path = model_path[:-3] + "_backup.ts"
    else:
        bak_path = model_path[:-3] + "_backup.pt"
    if os.path.exists(model_path):
        try:
            os.replace(model_path, bak_path)
            print(f"{tag} backed up existing model")
        except Exception as e:
            print(f"{tag} failed to backup existing model:", e)

    save_pt_model(model, model_path, arch)
    print(f"{tag} retraining complete for epoch {epoch}")
    timings['save'] = timings.get('save', 0.0) + (time.time() - t0)


def make_pt_infer(model, max_bs, encoding_type="xc0"):
    """
    Returns (model, fwd) where fwd(enc_np) -> (logits_np, wdl_np).
    logits_np: (B, 1858) raw policy logits
    wdl_np:    (B, 3) softmax WDL probabilities
    Softmax on policy, uniform_eps, prior_clip_max applied in C++ build_priors.
    """
    device = torch.device("cuda")
    model = model.to(device).half().eval()
    model = torch.jit.optimize_for_inference(model)
    is_lc0 = encoding_type == "lc0"

    def base_fwd(pair):
        enc_np = pair[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if is_lc0:
                # drained lc0 planes are raw uint8 (rule50 unscaled)
                enc_t = torch.from_numpy(enc_np).float().to(device)
                enc_t[:, 109] /= 99.0
                enc_t = enc_t.half()
            else:
                enc_t = torch.from_numpy(enc_np).long().to(device)
            logits, value = model(enc_t)
        logits = logits.float()
        wdl = F.softmax(value.float(), dim=-1)
        return logits.cpu().numpy(), wdl.cpu().numpy()

    if max_bs is None:
        return model, base_fwd

    def fwd(pair):
        enc_np = pair[0]
        B = int(enc_np.shape[0])
        if B <= max_bs:
            return base_fwd(pair)
        logits_parts, wdl_parts = [], []
        for i in range(0, B, max_bs):
            j = min(i + max_bs, B)
            lg, w = base_fwd((enc_np[i:j],))
            logits_parts.append(lg)
            wdl_parts.append(w)
        return np.concatenate(logits_parts, axis=0), np.concatenate(wdl_parts, axis=0)

    return model, fwd


def export_ts_to_onnx(ts_path, onnx_path, encoding_type="xc0"):
    """
    Export model to ONNX with dynamic batch axis.
    Loads from companion .pt as an eager half-precision model — TorchScript export
    fails because aten::_native_multi_head_attention is not ONNX-exportable.
    Tries dynamo first; falls back to legacy opset-18.
    """
    import onnx
    from onnx import shape_inference as onnx_si
    from chessbot.model import PT_BUILDERS, VARIANTS

    pt_path = companion_pt(ts_path)
    ckpt = torch.load(pt_path, map_location='cuda')
    arch = ckpt.get('arch', FALLBACK_ARCH)
    model = PT_BUILDERS[arch](VARIANTS[arch]).half().cuda().eval()
    model.load_state_dict(ckpt['model'])

    if encoding_type == "lc0":
        dummy = torch.zeros(1, 112, 8, 8, dtype=torch.float16, device='cuda')
    elif encoding_type == "xc0h":
        dummy = torch.zeros(1, 393, dtype=torch.long, device='cuda')
    else:
        dummy = torch.zeros(1, 64, dtype=torch.long, device='cuda')
    with torch.no_grad():
        model(dummy)  # verify forward pass before export

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Constant folding in symbolic shape inference')
        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=18,
            input_names=['enc_in'],
            output_names=['policy_logits', 'value_out'],
            dynamic_axes={
                'enc_in':        {0: 'batch'},
                'policy_logits': {0: 'batch'},
                'value_out':     {0: 'batch'},
            },
            do_constant_folding=False,
        )
    print(f'[export] ONNX -> {onnx_path}')

    proto = onnx.load(onnx_path)
    proto = onnx_si.infer_shapes(proto)
    onnx.save(proto, onnx_path)
    del model
    sz_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f'[export] shape inference complete  ({sz_mb:.1f} MB)')
