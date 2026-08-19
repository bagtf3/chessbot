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

# TEMPORARY cap on how many records retrain_pt puts on the GPU at once.
# Remove once retrain moves to the tfrec + async EpochBuffer streamer.
VRAM_CHUNK = 5120

# Dynamic LW: mirrors the pretrain LW-twiddle recipe (bootstrap_model_async_
# tfrec_pt.py). Fixed 1:2 policy:value shape, rescaled every retrain cycle by
# an EMA of the raw per-cycle CEs so the weighted total matches
# RETRAIN_LW_TARGET_POLICY_MULT*policy_ce_hat + RETRAIN_LW_TARGET_VALUE_MULT*
# value_ce_hat. Each retrain cycle already spans a full pass over a large,
# variable-sized batch of records (not a fixed 20-step epoch like pretrain),
# so instead of a fixed per-call alpha, the EMA weight is derived from step
# count: pretrain's span is LW_EMA_SPAN_EPOCHS(20)*PT_STEPS_PER_EPOCH(20) =
# 400 steps at batch=512 (matches cfg.retrain_batch_size), so a retrain cycle
# that runs `steps` steps carries alpha = steps / RETRAIN_LW_EMA_SPAN_STEPS,
# capped at 1 -- same steps-of-memory invariant, regardless of cycle size.
RETRAIN_LW_BASE_POLICY        = 1.0
RETRAIN_LW_BASE_VALUE         = 2.0
RETRAIN_LW_TARGET_POLICY_MULT = 1.0
RETRAIN_LW_TARGET_VALUE_MULT  = 4.0
RETRAIN_LW_EMA_SPAN_STEPS     = 400


def retrain_lw_state_path(model_path: str, run_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(run_dir, "train_ckpts", stem + "_lw_state.pt")


def retrain_lw_ema_update(ema, raw, steps: int):
    if ema is None:
        return raw
    alpha = min(1.0, steps / RETRAIN_LW_EMA_SPAN_STEPS)
    return (1.0 - alpha) * ema + alpha * raw


def retrain_lw_from_state(lw_state: dict):
    policy_ce_hat = lw_state.get("policy_ce_hat")
    value_ce_hat  = lw_state.get("value_ce_hat")
    if policy_ce_hat is None or value_ce_hat is None:
        return RETRAIN_LW_BASE_POLICY, RETRAIN_LW_BASE_VALUE
    current_total = RETRAIN_LW_BASE_POLICY * policy_ce_hat + RETRAIN_LW_BASE_VALUE * value_ce_hat
    target_total = (
        RETRAIN_LW_TARGET_POLICY_MULT * policy_ce_hat
        + RETRAIN_LW_TARGET_VALUE_MULT * value_ce_hat
    )
    scale = target_total / current_total
    return scale * RETRAIN_LW_BASE_POLICY, scale * RETRAIN_LW_BASE_VALUE


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


def load_pt_model(path, log_params=False):
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
                model = PT_BUILDERS[arch](VARIANTS[arch], log_params=log_params)
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
        model = PT_BUILDERS[arch](VARIANTS[arch], log_params=log_params)
        model.load_state_dict(state_dict, strict=False)
        log_policy_mask_status(model)
        return model, arch
    if isinstance(obj, torch.nn.Module):
        return obj, FALLBACK_ARCH
    raise ValueError(f"[pytorch] unrecognized checkpoint format in {path}")


def save_pt_model(model, path, arch=None, opt=None, trace=True):
    """Save model.
    .ts path: TorchScript trace for inference + companion .pt state_dict for training.
    .pt path: state_dict dict only.

    trace=False writes only the companion. Nothing reads the trace on the
    selfplay path: pt_eager inference is retired and export_ts_to_onnx loads
    companion_pt() instead, because TorchScript cannot export
    aten::_native_multi_head_attention. The model is still moved and halved
    either way, so the companion is byte-identical to the traced case.
    """
    if path.endswith(".ts"):
        trace_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = trace_device.type == "cuda"
        trace_model = model.to(trace_device).half().eval() if use_fp16 else model.to(trace_device).eval()
        if not trace:
            pt_path = companion_pt(path)
            torch.save({"model": model.state_dict(), "arch": arch}, pt_path)
            print(f"[pytorch] weights saved -> {os.path.basename(pt_path)}")
            return
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
        print(f"[pytorch] TorchScript saved -> {os.path.basename(path)}")
        pt_path = companion_pt(path)
        payload = {"model": model.state_dict(), "arch": arch}
        torch.save(payload, pt_path)
        print(f"[pytorch] companion weights saved -> {os.path.basename(pt_path)}")
        return
    payload = {"model": model.state_dict(), "arch": arch}
    if opt is not None:
        payload["opt"] = opt.state_dict()
    torch.save(payload, path)
    print(f"[pytorch] checkpoint saved -> {path}")



def pt_opt_state_path(model_path: str, run_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(run_dir, "train_ckpts", stem + "_opt_state.pt")


PLAYER_SPAN = 7


def player_state_path(model_path: str, run_dir: str) -> str:
    """float32 EMA accumulator for the played model."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(run_dir, "train_ckpts", stem + "_player_state.pt")


def player_model_path(model_path: str) -> str:
    """{run}_model.ts (trainer) -> {run}_model_player.ts (played)"""
    root, ext = os.path.splitext(model_path)
    return root + "_player" + ext


def update_player_ema(model_path, run_dir, model, arch, seed_state=None,
                      span=PLAYER_SPAN):
    """Fold the freshly trained weights into the played model's EMA.

    Two lineages. The trainer is model_path: retrain always resumes from it
    and the optimizer state stays paired with it, so nothing here feeds back
    into training. This writes a separate _player file that selfplay plays.

    Kept in float32 on disk. The model is fp16 and repeated 2/3:1/3 blending
    at half precision drifts, so the accumulator carries the extra bits and
    only the exported model is cast back down.
    """
    if arch is None:
        print("[swa] no arch on the loaded model, skipping EMA export")
        return None

    alpha = 2.0 / (span + 1.0)
    new = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    path = player_state_path(model_path, run_dir)
    if os.path.exists(path):
        ema = torch.load(path, map_location="cpu")
        src = "stored"
    else:
        base = seed_state if seed_state is not None else new
        ema = {k: v.detach().cpu().float() for k, v in base.items()}
        src = "seeded from pre-retrain weights"

    out = {}
    for k, v in new.items():
        # index buffers such as sl_idx are not weights -- carry, never average
        if v.dtype.is_floating_point and k in ema:
            out[k] = (1.0 - alpha) * ema[k] + alpha * v.float()
        else:
            out[k] = v.clone()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)

    # Only the companion .pt is written. export_ts_to_onnx never opens the
    # trace -- it loads companion_pt(ts_path) and rebuilds an eager fp16 model,
    # because TorchScript export chokes on aten::_native_multi_head_attention.
    # So tracing a _player.ts would cost a CUDA trace and 39 MB for nothing.
    played = {k: (v.half() if v.dtype.is_floating_point else v)
              for k, v in out.items()}
    out_path = companion_pt(player_model_path(model_path))
    torch.save({"model": played, "arch": arch}, out_path)
    print(f"[swa] EMA{span} alpha={alpha:.3f} ({src}) -> "
          f"{os.path.basename(out_path)}")
    return out_path


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
    tag     = f"[retrain{(' ' + label) if label else ''}]"
    clips   = [gs['gn_clips'] for gs in epoch_grad_stats]
    steps   = [gs['gn_steps'] for gs in epoch_grad_stats]
    means   = [gs['gn_mean']   for gs in epoch_grad_stats]
    medians = [gs['gn_median'] for gs in epoch_grad_stats]
    mins    = [gs['gn_min']    for gs in epoch_grad_stats]
    maxs    = [gs['gn_max']    for gs in epoch_grad_stats]
    print(
        f"{tag} grad stats for epoch {epoch}: "
        f"mean={np.mean(means):.2f}  median={np.mean(medians):.2f}  "
        f"min={min(mins):.2f}  max={max(maxs):.2f}  "
        f"clips={sum(clips)}/{sum(steps)}"
    )


def print_pt_sample_weight_stats(vwht, pwht, epoch, label=""):
    tag = f"[retrain{(' ' + label) if label else ''}]"
    for name, w in (("pwht", np.asarray(pwht)), ("vwht", np.asarray(vwht))):
        if not w.size:
            continue
        print(
            f"{tag} {name} stats for epoch {epoch}: "
            f"mean={w.mean():.2f}  median={np.median(w):.2f}  "
            f"min={w.min():.2f}  max={w.max():.2f}  n={w.size}"
        )


def retrain_pt(model_path, X, P, Y_wdl, vwht, pwht, cfg, epoch, args,
               label="", timings=None, model=None, arch=None):
    """Load, train one epoch, save."""
    if timings is None:
        timings = {}
    tag = f"[retrain{(' ' + label) if label else ''}]"
    short_model = os.path.join(
        os.path.basename(os.path.dirname(model_path)),
        os.path.basename(model_path))

    t0 = time.time()
    if model is None:
        print(f"{tag} loading {short_model}")
        model, arch = load_pt_model(model_path)
    device = torch.device("cuda")
    model  = model.to(device).train()
    timings['load_model'] = timings.get('load_model', 0.0) + (time.time() - t0)

    n = len(X)

    lr = cfg.learning_rate
    clip_norm = cfg.retrain_clip_norm
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

    lw_state_path = retrain_lw_state_path(model_path, cfg.run_dir)
    if os.path.exists(lw_state_path):
        lw_state = torch.load(lw_state_path, map_location="cpu")
        print(f"{tag} restored LW state: "
              f"policy_ce_hat={lw_state.get('policy_ce_hat'):.3f}  "
              f"value_ce_hat={lw_state.get('value_ce_hat'):.3f}")
    else:
        lw_state = {"policy_ce_hat": None, "value_ce_hat": None}
        print(f"{tag} no prior LW state - starting fresh")
    policy_lw, value_lw = retrain_lw_from_state(lw_state)
    print(f"{tag} dynamic LW: policy={policy_lw:.3f}  value={value_lw:.3f}")

    from torch.amp import autocast, GradScaler
    scaler = GradScaler("cuda")

    epoch_losses     = []
    epoch_grad_stats = []
    nan_skips        = 0
    t0 = time.time()
    for ep_idx in range(1):
        # TEMPORARY: chunk the dataset into <=VRAM_CHUNK-record groups so a
        # single retrain never pushes more than that onto the GPU at once --
        # loading the whole array upfront was blowing past VRAM on large
        # retrains and falling back to slow system-RAM spillover. Remove
        # once retrain moves to the tfrec + async EpochBuffer streamer.
        perm         = np.random.permutation(n)
        total        = value_total = policy_total = 0.0
        steps        = 0
        grad_norms   = []
        clip_count   = 0

        for chunk_start in range(0, n, VRAM_CHUNK):
            chunk_idx = perm[chunk_start:chunk_start + VRAM_CHUNK]

            if cfg.encoding_type == "lc0":
                X_t = torch.from_numpy(X[chunk_idx]).float().to(device)
                X_t[:, 109] /= 99.0
            else:
                X_t = torch.from_numpy(X[chunk_idx]).long().to(device)
            P_t    = torch.from_numpy(P[chunk_idx]).float().to(device)
            Y_t    = torch.from_numpy(Y_wdl[chunk_idx]).float().to(device)
            vwht_t = torch.from_numpy(vwht[chunk_idx]).float().to(device)
            pwht_t = torch.from_numpy(pwht[chunk_idx]).float().to(device)

            for start in range(0, len(chunk_idx), args.batch_size):
                end = start + args.batch_size
                xb  = X_t[start:end]
                pb  = P_t[start:end]
                yb  = Y_t[start:end]
                vwb = vwht_t[start:end]
                pwb = pwht_t[start:end]

                opt.zero_grad()
                with autocast("cuda"):
                    policy_logits, value_out = model(xb)

                    log_probs   = F.log_softmax(policy_logits, dim=-1)
                    policy_loss = (-(pb * log_probs).sum(dim=-1) * pwb).mean()

                    log_wdl    = F.log_softmax(value_out, dim=-1)
                    value_loss = (-(yb * log_wdl).sum(dim=-1) * vwb).mean()

                    loss = policy_lw * policy_loss + value_lw * value_loss

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm).item()

                if not np.isfinite(raw_norm):
                    nan_skips += 1
                    dump_path = os.path.join(
                        os.path.dirname(model_path),
                        f"nan_batch_{label or 'retrain'}_ep{epoch}_{int(time.time())}.pkl")
                    try:
                        import pickle
                        with open(dump_path, "wb") as f:
                            pickle.dump({
                                'batch_idx': chunk_idx[start:end],
                                'X':    xb.detach().cpu().numpy(),
                                'P':    pb.detach().cpu().numpy(),
                                'Y':    yb.detach().cpu().numpy(),
                                'vwht': vwb.detach().cpu().numpy(),
                                'pwht': pwb.detach().cpu().numpy(),
                                'loss': loss.item() if torch.isfinite(loss) else float('nan'),
                                'policy_loss': policy_loss.item() if torch.isfinite(policy_loss) else float('nan'),
                                'value_loss':  value_loss.item() if torch.isfinite(value_loss) else float('nan'),
                            }, f, protocol=pickle.HIGHEST_PROTOCOL)
                        print(f"{tag} WARNING: non-finite grad norm ({raw_norm}) at step {steps}, "
                              f"epoch {epoch} -- skipping this batch's update, dumped -> {dump_path}")
                    except Exception as e:
                        print(f"{tag} WARNING: non-finite grad norm ({raw_norm}), "
                              f"failed to dump offending batch: {e}")
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    continue

                grad_norms.append(raw_norm)
                if raw_norm > clip_norm:
                    clip_count += 1
                scaler.step(opt)
                scaler.update()

                total        += loss.item()
                policy_total += policy_loss.item()
                value_total  += value_loss.item()
                steps        += 1

        if nan_skips:
            print(f"{tag} {nan_skips} batch(es) skipped this epoch due to non-finite grad norm")

        gn = np.array(grad_norms)
        epoch_losses.append({
            'loss':   total        / max(1, steps),
            'policy': policy_total / max(1, steps),
            'value':  value_total  / max(1, steps),
        })
        epoch_grad_stats.append({
            'gn_mean':   float(gn.mean())   if gn.size else float('nan'),
            'gn_median': float(np.median(gn)) if gn.size else float('nan'),
            'gn_min':    float(gn.min())    if gn.size else float('nan'),
            'gn_max':    float(gn.max())    if gn.size else float('nan'),
            'gn_clips':  clip_count,
            'gn_steps':  steps,
        })

    timings['fit'] = timings.get('fit', 0.0) + (time.time() - t0)
    print_pt_fit_history(epoch_losses, epoch, label=label)

    os.makedirs(os.path.dirname(opt_state_path), exist_ok=True)
    torch.save(opt.state_dict(), opt_state_path)

    cycle_steps = epoch_grad_stats[-1]['gn_steps']
    lw_state["policy_ce_hat"] = retrain_lw_ema_update(
        lw_state["policy_ce_hat"], epoch_losses[-1]['policy'], cycle_steps)
    lw_state["value_ce_hat"] = retrain_lw_ema_update(
        lw_state["value_ce_hat"], epoch_losses[-1]['value'], cycle_steps)
    torch.save(lw_state, lw_state_path)
    print(f"{tag} LW state updated: policy_ce_hat={lw_state['policy_ce_hat']:.3f}  "
          f"value_ce_hat={lw_state['value_ce_hat']:.3f}  (cycle_steps={cycle_steps})")

    t0 = time.time()
    # Roll aside the weights, not the trace. The trace is no longer written,
    # and a _backup.ts had no companion anyway, so it could only ever load as
    # an inference-only ScriptModule -- useless for resuming training.
    weights_path = companion_pt(model_path) if model_path.endswith(".ts") \
        else model_path
    bak_path = weights_path[:-3] + "_backup.pt"
    if os.path.exists(weights_path):
        try:
            os.replace(weights_path, bak_path)
        except Exception as e:
            print(f"{tag} failed to backup existing model:", e)

    save_pt_model(model, model_path, arch, trace=False)
    print_pt_sample_weight_stats(vwht, pwht, epoch, label=label)
    print_pt_grad_stats(epoch_grad_stats, epoch, label=label)
    print(f"{tag} retraining complete for epoch {epoch}")
    timings['save'] = timings.get('save', 0.0) + (time.time() - t0)

    # same two fields pretraining records alongside each validation row
    return {
        'train_loss': epoch_losses[-1]['loss'],
        'gn_mean': epoch_grad_stats[-1]['gn_mean'],
    }


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

    proto = onnx.load(onnx_path)
    proto = onnx_si.infer_shapes(proto)
    onnx.save(proto, onnx_path)
    del model
