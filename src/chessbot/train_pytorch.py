import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from chessbot.replay_buffer import stack_policies


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

# Dynamic LW, the pretrain LW-twiddle recipe (bootstrap_model_async_tfrec_pt.py).
# RETRAIN_LW_BASE_POLICY / RETRAIN_LW_BASE_VALUE set the policy:value RATIO.
# Both weights are then multiplied by one shared scale so the weighted total
# matches RETRAIN_LW_TARGET_POLICY_MULT * policy_ce_hat +
# RETRAIN_LW_TARGET_VALUE_MULT * value_ce_hat: the TARGET mults pin the overall
# loss LEVEL while the BASE ratio is moved (e.g. annealed toward even).
# When the TARGET mults equal the BASE values the scale is 1 and this is a
# plain fixed-weight loss. Under Adam only the ratio changes the update; the
# shared scale mostly shifts how often clip_norm bites.
# *_ce_hat are EMAs of the raw per-cycle CEs. Retrain cycles run a variable
# number of steps, so the EMA alpha is steps / RETRAIN_LW_EMA_SPAN_STEPS,
# capped at 1, keeping the pretrain span measured in steps.
RETRAIN_LW_BASE_POLICY        = 1.0
RETRAIN_LW_BASE_VALUE         = 1.0
RETRAIN_LW_TARGET_POLICY_MULT = 1.0
RETRAIN_LW_TARGET_VALUE_MULT  = 1.0
RETRAIN_LW_EMA_SPAN_STEPS     = 400

# GradScaler start and ceiling: one notch below this model's known overflow
# point, so the scaler's growth never probes into an overflow-skipped step
MAX_GRAD_SCALE = 2 ** 15

# Same recipe as the pretrainer. Smartgate illegality BCE on model.gate_raw
# (legal = policy target > 0; every legal move carries a 1-visit floor), pinned
# to GATE_SHARE * (weighted policy + value) while above GATE_W_FLOOR.
GATE_LEGAL_POS_W = 5.0
GATE_SHARE       = 0.05
GATE_W_FLOOR     = 5.0

# Patience on the held-out shard's training loss, scored at the start weights
# (logged only) and every EVAL_EVERY batches (skipped batches included). The
# first eval after the start sets the baseline. PATIENCE fails drop the LR, at
# most LR_DROPS times; once the drops are spent, PATIENCE fails stop training,
# but never before the end of epoch 1. Through epoch 1 a loss counts as better
# only when the paired z of the per-position gain over the best eval exceeds
# PAIRED_Z; the IMPROVE_REL gate is logged beside it. Past epoch 1 an eval is a
# best when its loss is at or below the midpoint of the start loss and the
# best loss, else a fail; a loss below the best becomes the best, tightening
# the midpoint. The pick is the latest best past epoch 1, else the end of
# epoch 1. The LR rises linearly from its floor (LR_DROP_FACTOR ** LR_DROPS of
# the LR) to the LR over the first WARMUP_BATCHES batches. The in-sample
# tracked sets are scored with the held-out shard, logged only. The check-in
# shard is scored once, on the picked weights, into CHECKIN_CSV.
PATIENCE             = 3
WARMUP_BATCHES       = 120
LR_DROPS             = 2
LR_DROP_FACTOR       = 10 ** -0.5
MAX_EPOCHS           = 2
IMPROVE_REL          = 0.0005
PAIRED_Z             = 2.0
EVAL_EVERY           = 120
PATIENCE_CSV         = "patience_tracking.csv"
CHECKIN_CSV          = "checkin_tracking.csv"
CHECKIN_MEANS        = (3, 5)


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


def save_pt_model(model, path, arch=None, opt=None, trace=True, quiet=False):
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
            if not quiet:
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
    if not quiet:
        print(f"[pytorch] checkpoint saved -> {path}")



def pt_opt_state_path(model_path: str, run_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(run_dir, "train_ckpts", stem + "_opt_state.pt")


def pt_scaler_state_path(model_path: str, run_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(run_dir, "train_ckpts", stem + "_scaler_state.pt")


def decay_groups(model, wd):
    """Decoupled WD on 2D+ weights only; biases, norm gains, PReLU slopes exempt."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def flat_opt_state(model, opt):
    """Per-param optimizer state in the single-group layout kept on disk."""
    flat = torch.optim.Adam(model.parameters())
    for p in model.parameters():
        if p in opt.state:
            flat.state[p] = opt.state[p]
    return flat.state_dict()


def cpu_opt_state(model, opt):
    """A CPU copy of flat_opt_state, kept beside a weights snapshot."""
    sd = flat_opt_state(model, opt)
    state = {k: {kk: (v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v)
                 for kk, v in ps.items()} for k, ps in sd["state"].items()}
    return {"state": state, "param_groups": [dict(g) for g in sd["param_groups"]]}


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


def print_patience_row(tag, nb, ep, means, best, z, thresh, action, lr):
    """One check-in row of the patience table; None prints blank."""
    def cell(v, w, spec):
        return ("" if v is None else format(v, spec)).rjust(w)
    print(f"{tag} {nb:>4}  {ep:>4.2f}  {cell(means.get('replay_loss'), 6, '.4f')}  "
          f"{cell(means.get('primary_loss'), 6, '.4f')}  {means['loss']:>6.4f}  "
          f"{cell(best, 6, '.4f')}  {cell(z, 5, '+.2f')}  {cell(thresh, 6, '.4f')}  "
          f"{action:<6}  {lr:.1e}")


def print_pt_grad_stats(epoch_grad_stats, gate, skips, label=""):
    if not epoch_grad_stats:
        return
    tag     = f"[retrain{(' ' + label) if label else ''}]"
    clips   = [gs['gn_clips'] for gs in epoch_grad_stats]
    steps   = [gs['gn_steps'] for gs in epoch_grad_stats]
    means   = [gs['gn_mean']   for gs in epoch_grad_stats]
    medians = [gs['gn_median'] for gs in epoch_grad_stats]
    maxs    = [gs['gn_max']    for gs in epoch_grad_stats]
    print(f"{tag} grad mean {np.mean(means):.2f} med {np.mean(medians):.2f} "
          f"max {max(maxs):.2f} clips {sum(clips)}/{sum(steps)}  "
          f"gate {gate:.3f}  skips {skips}")


def normalize_bounded_mean(x, lo=0.05, hi=3.0, target=1.0, iters=100, tol=1e-9):
    """Pin a weight array to mean == target while holding hard bounds [lo, hi].

    Iterative clamp-then-rescale: clamp to [lo, hi], scale toward the target mean,
    repeat.  Values pinned at a bound stay there; the interior carries the mean.
    Bounds are never violated (max cannot exceed hi, min cannot fall below lo); to
    keep the mean exact the realized extremes may sit inside the bounds.  Always
    feasible here since lo < target < hi.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
    for _ in range(iters):
        m = x.mean()
        if m <= 0 or abs(m - target) <= tol:
            break
        x = np.clip(x * (target / m), lo, hi)
    return x


def epoch_batch_count(n, batch_size):
    """Batches in one pass: VRAM_CHUNK chunks, each split into batches."""
    return sum(-(-min(VRAM_CHUNK, n - s) // batch_size)
               for s in range(0, n, VRAM_CHUNK))


def patience_eval_batches(epoch_batches):
    """Batch counts at which the held-out shard is scored: every EVAL_EVERY,
    plus the end of epoch 1."""
    total = epoch_batches * MAX_EPOCHS
    return sorted(set(range(EVAL_EVERY, total + 1, EVAL_EVERY)) | {epoch_batches})


def write_csv_row(cfg, name, row):
    import csv
    path = os.path.join(cfg.run_dir, name)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def prep_eval_set(arrays):
    """Dense policies and bounded-mean record weights for eval_set_losses."""
    X, P, Y, vw, pw = arrays
    return (X, stack_policies(P, index=None).astype(np.float16), Y,
            normalize_bounded_mean(vw).astype(np.float32),
            normalize_bounded_mean(pw).astype(np.float32))


def eval_set_losses(model, s, batch_size, policy_lw, value_lw, encoding_type):
    """Per-position weighted training loss on a prepped set, and the mean
    record-weighted policy and value CE. Leaves the model in eval mode."""
    from torch.amp import autocast
    X, P, Y, vw, pw = s
    device = torch.device("cuda")
    model.eval()
    lp = np.empty(len(X), dtype=np.float64)
    lv = np.empty(len(X), dtype=np.float64)
    with torch.no_grad(), autocast("cuda"):
        for i in range(0, len(X), batch_size):
            j = min(i + batch_size, len(X))
            if encoding_type == "lc0":
                xb = torch.from_numpy(X[i:j]).float().to(device)
                xb[:, 109] /= 99.0
            else:
                xb = torch.from_numpy(X[i:j]).long().to(device)
            pol, val = model(xb)
            pb = torch.from_numpy(P[i:j]).float().to(device)
            yb = torch.from_numpy(Y[i:j]).float().to(device)
            lp[i:j] = (-(pb * F.log_softmax(pol.float(), dim=-1)).sum(dim=-1)
                       ).double().cpu().numpy()
            lv[i:j] = (-(yb * F.log_softmax(val.float(), dim=-1)).sum(dim=-1)
                       ).double().cpu().numpy()
    lp *= pw
    lv *= vw
    return policy_lw * lp + value_lw * lv, lp.mean(), lv.mean()


def score_checkin_shard(cfg, epoch, model, arrays, batch_size, policy_lw, value_lw,
                        tag="[retrain]"):
    """Score the check-in shard, print it beside the means of the last
    CHECKIN_MEANS rows that exist, and append its row to CHECKIN_CSV."""
    import csv
    vec, lp, lv = eval_set_losses(model, prep_eval_set(arrays), batch_size,
                                  policy_lw, value_lw, cfg.encoding_type)
    path = os.path.join(cfg.run_dir, CHECKIN_CSV)
    prior = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            prior = [float(r["loss"]) for r in csv.DictReader(f)]
    lasts = "".join(f"  last{k} {np.mean(prior[-k:]):.4f}"
                    for k in CHECKIN_MEANS if len(prior) >= k)
    print(f"{tag} checkin shard {vec.mean():.4f}  "
          f"(policy {lp:.4f}  value {lv:.4f}){lasts}")
    write_csv_row(cfg, CHECKIN_CSV, {
        "run_tag": cfg.run_tag, "epoch": epoch, "loss": round(vec.mean(), 6),
        "policy": round(lp, 6), "value": round(lv, 6)})


def cpu_state(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def retrain_pt(model_path, X, P, Y_wdl, vwht, pwht, cfg, epoch, args,
               label="", timings=None, model=None, arch=None, holdout=None,
               tracked=None, checkin=None):
    """Load, train, save. Without holdout: one epoch. With holdout
    (X, P, Y, vwht, pwht of the held-out shard): up to MAX_EPOCHS under
    patience, saving the picked weights. tracked maps a name to in-sample
    arrays scored with the held-out shard; checkin is the check-in shard's
    arrays."""
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
    for m in model.modules():   # selfplay retrains never use dropout
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    timings['load_model'] = timings.get('load_model', 0.0) + (time.time() - t0)

    n = len(X)

    # Pin the assembled batch weights to mean 1.0 within [0.05, 3.0], all told
    # (every source, lc0 shards included).  The KL boost and opening reweight only
    # push mass around; this is where the final invariant is enforced.
    vwht = normalize_bounded_mean(vwht).astype(np.float32)
    pwht = normalize_bounded_mean(pwht).astype(np.float32)

    lr = cfg.learning_rate
    lr_floor = lr * LR_DROP_FACTOR ** LR_DROPS
    clip_norm = cfg.retrain_clip_norm
    opt = torch.optim.AdamW(
        decay_groups(model, cfg.retrain_weight_decay), lr=lr,
        betas=(0.9, cfg.adam_beta2),
        fused=True,
    )
    restored = []
    opt_state_path = pt_opt_state_path(model_path, cfg.run_dir)
    if os.path.exists(opt_state_path):
        try:
            flat = torch.optim.Adam(model.parameters())
            flat.load_state_dict(torch.load(opt_state_path, map_location="cpu"))
            for p, param_state in flat.state.items():
                opt.state[p] = param_state
            for param_state in opt.state.values():
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        param_state[k] = v.to(device)
                if 'step' in param_state:   # fused Adam wants step as fp32
                    param_state['step'] = param_state['step'].float()
            restored.append("adam")
        except Exception as e:
            print(f"{tag} failed to load Adam state ({e}) - starting fresh")
    else:
        print(f"{tag} no prior Adam state - starting fresh")

    lw_state_path = retrain_lw_state_path(model_path, cfg.run_dir)
    if os.path.exists(lw_state_path):
        lw_state = torch.load(lw_state_path, map_location="cpu")
        restored.append("lw")
    else:
        lw_state = {"policy_ce_hat": None, "value_ce_hat": None}
        print(f"{tag} no prior LW state - starting fresh")
    policy_lw, value_lw = retrain_lw_from_state(lw_state)

    from torch.amp import autocast, GradScaler
    # State is persisted across retrains (same as the AdamW state above), so
    # only the very first retrain ever cold-starts, at MAX_GRAD_SCALE.
    scaler = GradScaler("cuda", init_scale=MAX_GRAD_SCALE)
    scaler_state_path = pt_scaler_state_path(model_path, cfg.run_dir)
    if os.path.exists(scaler_state_path):
        try:
            sd = torch.load(scaler_state_path, map_location="cpu")
            sd["scale"] = min(float(sd["scale"]), float(MAX_GRAD_SCALE))
            scaler.load_state_dict(sd)
            restored.append("scaler")
        except Exception as e:
            print(f"{tag} failed to load GradScaler state ({e}) - starting fresh")
    else:
        print(f"{tag} no prior GradScaler state - starting fresh")
    print(f"{tag} {epoch} restored {'/'.join(restored) or 'nothing'}  "
          f"LW p {policy_lw:.3f} v {value_lw:.3f}  scale {scaler.get_scale():.0f}  "
          f"warmup {lr_floor:.1e}->{lr:.1e} over {WARMUP_BATCHES}")

    has_gate = hasattr(model, "gate_raw")
    has_proj = hasattr(model, "project_params")
    g_losses = []

    def set_train_mode():
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()

    epoch_batches = epoch_batch_count(n, args.batch_size)
    pat = None
    if holdout is not None:
        eval_sets = {"held_out": holdout, **(tracked or {})}
        eval_sets = {k: prep_eval_set(v) for k, v in eval_sets.items()}

        def score_sets():
            """Per-position losses of every eval set, held-out first, and
            their means keyed by csv column."""
            vecs = {k: eval_set_losses(model, s, args.batch_size, policy_lw,
                                       value_lw, cfg.encoding_type)[0]
                    for k, s in eval_sets.items()}
            set_train_mode()
            means = {("loss" if k == "held_out" else f"{k}_loss"): round(v.mean(), 6)
                     for k, v in vecs.items()}
            return vecs, means

        vecs, means = score_sets()
        pat = {"eval_at": set(patience_eval_batches(epoch_batches)),
               "best": None, "best_vec": None, "best_batch": 0,
               "fails": 0, "drops": 0, "lr": lr, "stop": False,
               "best_state": None, "epoch1_state": None,
               "best_opt": None, "epoch1_opt": None,
               "start": vecs["held_out"].mean(), "means": {0: means}}
        print(f"{tag} {'b':>4}  {'ep':>4}  {'repl':>6}  {'prim':>6}  {'held':>6}  "
              f"{'best':>6}  {'z':>5}  {'thresh':>6}  {'action':<6}  lr")
        print_patience_row(tag, 0, 0.0, means, None, None, None, "start", lr_floor)
        write_csv_row(cfg, PATIENCE_CSV, {
            "epoch": epoch, "batch": 0, "epoch_frac": 0.0, "lr": lr_floor, **means,
            "best_loss": "", "thresh": "", "rel_gain": "", "gate_a": "", "z": "",
            "gate_b": "", "fails": 0, "drops": 0, "action": "start"})

    def patience_eval(nb):
        vecs, means = score_sets()
        pat["means"][nb] = means
        vec = vecs["held_out"]
        loss_now = vec.mean()
        first = pat["best"] is None
        prev = loss_now if first else pat["best"]
        d = (vec if first else pat["best_vec"]) - vec
        z = 0.0 if first else d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        gate_a = loss_now < prev * (1.0 - IMPROVE_REL)
        gate_b = first or z > PAIRED_Z
        thresh = None
        if nb > epoch_batches:
            thresh = (pat["start"] + pat["best"]) / 2
            gate_b = loss_now <= thresh
        if nb == epoch_batches:
            pat["epoch1_state"] = cpu_state(model)
            pat["epoch1_opt"] = cpu_opt_state(model, opt)
        if gate_b and nb > epoch_batches:
            pat.update(best_batch=nb, fails=0)
            pat["best_state"] = cpu_state(model)
            pat["best_opt"] = cpu_opt_state(model, opt)
            action = "best"
        elif gate_b:
            pat.update(best=loss_now, best_vec=vec, best_batch=nb, fails=0)
            action = "best"
        else:
            pat["fails"] += 1
            action = "fail"
            if pat["fails"] >= PATIENCE:
                if pat["drops"] < LR_DROPS:
                    pat["drops"] += 1
                    pat["fails"] = 0
                    pat["lr"] *= LR_DROP_FACTOR
                    for g in opt.param_groups:
                        g["lr"] = pat["lr"]
                    action = "drop"
                elif nb >= epoch_batches:
                    pat["stop"] = True
                    action = "stop"
        if thresh is not None and loss_now < pat["best"]:
            pat.update(best=loss_now, best_vec=vec)
        print_patience_row(tag, nb, nb / epoch_batches, means,
                           None if first else prev, None if first else z,
                           thresh, action, pat["lr"])
        write_csv_row(cfg, PATIENCE_CSV, {
            "epoch": epoch, "batch": nb, "epoch_frac": round(nb / epoch_batches, 4),
            "lr": pat["lr"], **means, "best_loss": round(prev, 6),
            "thresh": "" if thresh is None else round(thresh, 6),
            "rel_gain": round(1 - loss_now / prev, 6), "gate_a": int(gate_a),
            "z": round(z, 3), "gate_b": int(gate_b), "fails": pat["fails"],
            "drops": pat["drops"], "action": action})

    epoch_losses     = []
    epoch_grad_stats = []
    nan_skips        = 0
    nb               = 0
    seen             = 0
    t0 = time.time()
    for ep_idx in range(MAX_EPOCHS if pat else 1):
        if pat and pat["stop"]:
            break
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
            P_t    = torch.from_numpy(
                stack_policies([P[i] for i in chunk_idx], index=None)).to(device)
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

                if nb < WARMUP_BATCHES:
                    for g in opt.param_groups:
                        g["lr"] = lr_floor + (lr - lr_floor) * (nb + 1) / WARMUP_BATCHES
                opt.zero_grad()
                with autocast("cuda"):
                    policy_logits, value_out = model(xb)

                    log_probs   = F.log_softmax(policy_logits, dim=-1)
                    policy_loss = (-(pb * log_probs).sum(dim=-1) * pwb).mean()

                    log_wdl    = F.log_softmax(value_out, dim=-1)
                    value_loss = (-(yb * log_wdl).sum(dim=-1) * vwb).mean()

                    loss = policy_lw * policy_loss + value_lw * value_loss
                    if has_gate:
                        legal = pb > 0
                        g_w = torch.where(legal, GATE_LEGAL_POS_W, 1.0)
                        g_w = g_w * (g_w.shape[-1] / g_w.sum(-1, keepdim=True))
                        g_loss = F.binary_cross_entropy_with_logits(
                            model.gate_raw.float(),
                            (~legal).float(),
                            weight=g_w,
                            reduction="none",
                        ).sum(-1).mean()
                        gate_w = GATE_SHARE * loss.detach() / g_loss.detach().clamp(
                            min=GATE_W_FLOOR)
                        loss = loss + gate_w * g_loss
                        g_losses.append(g_loss.detach())

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm).item()

                if not np.isfinite(raw_norm):
                    # GradScaler overflow probing; scaler.step skips the update
                    nan_skips += 1
                    print(f"{tag} non-finite grad norm ({raw_norm}) at step {steps}, "
                          f"epoch {epoch}, loss {loss.item():.4f}, "
                          f"scale {scaler.get_scale():.0f} -- update skipped")
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                else:
                    grad_norms.append(raw_norm)
                    if raw_norm > clip_norm:
                        clip_count += 1
                    scaler.step(opt)
                    scaler.update()
                    if scaler.get_scale() > MAX_GRAD_SCALE:
                        scaler.update(new_scale=float(MAX_GRAD_SCALE))
                    if has_proj:
                        model.project_params()

                    total        += loss.item()
                    policy_total += policy_loss.item()
                    value_total  += value_loss.item()
                    steps        += 1

                nb += 1
                seen += len(xb)
                if pat and nb in pat["eval_at"]:
                    patience_eval(nb)
                    if pat["stop"]:
                        break
            if pat and pat["stop"]:
                break

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
    timings['samples'] = timings.get('samples', 0) + seen

    # the optimizer state saved is the one taken with the picked weights
    opt_sd = None
    if pat:
        if pat["best_batch"] > epoch_batches:
            picked = pat["best_batch"]
            model.load_state_dict(pat["best_state"])
            opt_sd = pat["best_opt"]
        else:
            picked = epoch_batches
            if nb > epoch_batches:
                model.load_state_dict(pat["epoch1_state"])
                opt_sd = pat["epoch1_opt"]
        pm = pat["means"][picked]
        print(f"{tag} pick b={picked} ({picked / epoch_batches:.2f})  "
              f"repl {pm.get('replay_loss', float('nan')):.4f}  "
              f"prim {pm.get('primary_loss', float('nan')):.4f}  "
              f"held {pm['loss']:.4f}  trained {nb} batches")
        write_csv_row(cfg, PATIENCE_CSV, {
            "epoch": epoch, "batch": picked, "epoch_frac": round(picked / epoch_batches, 4),
            "lr": pat["lr"], **pat["means"][picked], "best_loss": round(pat["best"], 6),
            "thresh": "", "rel_gain": "", "gate_a": "", "z": "", "gate_b": "",
            "fails": pat["fails"], "drops": pat["drops"], "action": "pick"})
        if checkin is not None:
            score_checkin_shard(cfg, epoch, model, checkin, args.batch_size,
                                policy_lw, value_lw, tag=tag)

    os.makedirs(os.path.dirname(opt_state_path), exist_ok=True)
    torch.save(opt_sd or flat_opt_state(model, opt), opt_state_path)
    torch.save(scaler.state_dict(), scaler_state_path)

    cycle_steps = epoch_grad_stats[-1]['gn_steps']
    lw_state["policy_ce_hat"] = retrain_lw_ema_update(
        lw_state["policy_ce_hat"], epoch_losses[-1]['policy'], cycle_steps)
    lw_state["value_ce_hat"] = retrain_lw_ema_update(
        lw_state["value_ce_hat"], epoch_losses[-1]['value'], cycle_steps)
    torch.save(lw_state, lw_state_path)

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

    save_pt_model(model, model_path, arch, trace=False, quiet=True)
    gate = torch.stack(g_losses).mean().item() if g_losses else float("nan")
    print()
    print_pt_grad_stats(epoch_grad_stats, gate, nan_skips, label=label)
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
    Loads from companion .pt as an eager fp32 model — TorchScript export fails
    because aten::_native_multi_head_attention is not ONNX-exportable. fp32 is
    required: fp16 graph weights make TRT run the wide policy bmm in fp16, which
    overflows and collapses the policy. Precision is chosen at engine build.
    Tries dynamo first; falls back to legacy opset-18.
    """
    import onnx
    from onnx import shape_inference as onnx_si
    from chessbot.model import PT_BUILDERS, VARIANTS

    pt_path = companion_pt(ts_path)
    ckpt = torch.load(pt_path, map_location='cuda')
    arch = ckpt.get('arch', FALLBACK_ARCH)
    model = PT_BUILDERS[arch](VARIANTS[arch]).cuda().eval()
    model.load_state_dict(ckpt['model'])

    if encoding_type == "lc0":
        dummy = torch.zeros(1, 112, 8, 8, dtype=torch.float32, device='cuda')
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
