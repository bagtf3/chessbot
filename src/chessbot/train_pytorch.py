import os
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F

from chessbot.replay_buffer import POLICY_DIM, stack_policies


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
# a non-finite grad norm with a finite loss halves the scale and reruns the
# step, up to this many times; past that, or with a non-finite loss, the
# update is skipped
MAX_STEP_RETRIES = 4
# a batch with a non-finite loss is saved here (in the run dir) and skipped
INF_LOSS_DIR = "inf_loss_batches"

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
# PAIRED_Z; the IMPROVE_REL gate is logged beside it. Hb is the lowest
# held-out loss so far, the start included. Past epoch 1 an eval is a best
# when every in-sample tracked set beats its lowest loss so far by FIT_MIN_GAIN
# (relative) and the held-out loss is within HELD_TOL (relative) of Hb or
# passes the paired z gate; else a fail. The pick is the latest best past
# epoch 1, else the end of epoch 1. The LR rises linearly from its floor
# (LR_DROP_FACTOR ** LR_DROPS of the LR) to the LR over the first
# WARMUP_BATCHES batches. An eval at FORCE_DROP_BATCH with drops left sets the
# LR to its floor and spends the drops. The check-in shard is scored once, on
# the picked weights, into CHECKIN_CSV.
PATIENCE             = 3
FORCE_DROP_BATCH     = 1200
WARMUP_BATCHES       = 120
LR_DROPS             = 2
LR_DROP_FACTOR       = 10 ** -0.5
MAX_EPOCHS           = 2
IMPROVE_REL          = 0.0005
PAIRED_Z             = 2.0
HELD_TOL             = 0.001
FIT_MIN_GAIN         = 0.00025
EVAL_EVERY           = 120
PATIENCE_CSV         = "patience_tracking.csv"
CHECKIN_CSV          = "checkin_tracking.csv"
CHECKIN_MEANS        = (3, 5)
# start vs picked weights on a training-set sample, by start-pass miss tier
TIER_CSV             = "tier_tracking.csv"
TIER_CHUNK           = 2048


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


def print_patience_header(tag):
    print(f"{tag} {'b':>4}  {'ep':>4}  {'repl':>6}   {'prim':>6}   {'held':>6}   "
          f"{'thresh':>6}  {'best':>6}  {'z':>5}  {'action':<6}  lr")


def print_patience_row(tag, nb, ep, means, best, z, thresh, action, lr, marks=None):
    """One check-in row of the patience table; None prints blank and an
    unscored tracked set prints -. marks maps a means key to the + or - shown
    after its loss."""
    marks = marks or {}

    def cell(v, w, spec):
        return ("" if v is None else format(v, spec)).rjust(w)

    def loss(k, spec):
        v = means.get(k)
        return ("-" if v is None else format(v, spec)).rjust(6) + marks.get(k, " ")
    print(f"{tag} {nb:>4}  {ep:>4.2f}  {loss('replay_loss', '.3f')}  "
          f"{loss('primary_loss', '.3f')}  {loss('loss', '.4f')}  "
          f"{cell(thresh, 6, '.4f')}  {cell(best, 6, '.3f')}  {cell(z, 5, '+.2f')}  "
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
    """Append row; columns the file lacks are added to its header, older rows
    left blank in them."""
    import csv
    path = os.path.join(cfg.run_dir, name)
    new = not os.path.exists(path)
    fields = list(row)
    if not new:
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            old = list(r.fieldnames or [])
            rows = list(r) if set(fields) - set(old) else None
        fields = old + [k for k in fields if k not in old]
        if rows is not None:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def prep_eval_set(arrays):
    """Dense policies and bounded-mean record weights for eval_set_losses."""
    X, P, Y, vw, pw = arrays
    return (X, stack_policies(P, index=None).astype(np.float16), Y,
            normalize_bounded_mean(vw).astype(np.float32),
            normalize_bounded_mean(pw).astype(np.float32))


def x_to_device(arr, encoding_type, device):
    if encoding_type == "lc0":
        x = torch.from_numpy(arr).float().to(device)
        x[:, 109] /= 99.0
        return x
    return torch.from_numpy(arr).long().to(device)


def start_pass(model, X, P, Y, batch_size, encoding_type, keep=()):
    """One no-grad pass over a training set with the start weights. Returns
    per-record policy and value CE ("lp", "lv") and their targets' entropies
    ("hp", "hv"), so CE minus entropy is the KL miss; for the sorted record
    indices in keep, also their WDL ("wdl") and policy logits ("logits").
    Leaves the model in eval mode."""
    from torch.amp import autocast
    device = torch.device("cuda")
    model.to(device).eval()
    n = len(X)
    out ={k: np.empty(n, dtype=np.float64) for k in ("lp", "lv", "hp", "hv")}
    keep = np.asarray(keep, dtype=np.int64)
    out["wdl"] = np.zeros((len(keep), 3), dtype=np.float32)
    out["logits"] = np.zeros((len(keep), POLICY_DIM), dtype=np.float32)
    with torch.no_grad(), autocast("cuda"):
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            pol, val = model(x_to_device(X[i:j], encoding_type, device))
            pol, val = pol.float(), val.float()
            # fp16 targets, as prep_eval_set stores them for the check-ins
            pb = torch.from_numpy(stack_policies(P[i:j], index=None)
                                  .astype(np.float16)).float().to(device)
            yb = torch.from_numpy(Y[i:j]).float().to(device)
            for k, t, logit in (("p", pb, pol), ("v", yb, val)):
                out["l" + k][i:j] = (-(t * F.log_softmax(logit, dim=-1)).sum(dim=-1)
                                     ).double().cpu().numpy()
                out["h" + k][i:j] = (-(t * torch.log(t.clamp_min(1e-12))).sum(dim=-1)
                                     ).double().cpu().numpy()
            a, b = np.searchsorted(keep, i), np.searchsorted(keep, j)
            if b > a:
                rows = torch.from_numpy(keep[a:b] - i).to(device)
                out["wdl"][a:b] = F.softmax(val[rows], dim=-1).cpu().numpy()
                out["logits"][a:b] = pol[rows].cpu().numpy()
    return out


def eval_set_losses(model, s, batch_size, policy_lw, value_lw, encoding_type):
    """Per-position weighted training loss on a prepped set, and the mean
    record-weighted policy and value CE. Leaves the model in eval mode."""
    lp, lv = eval_set_ce(model, s, batch_size, encoding_type)
    lp *= s[4]
    lv *= s[3]
    return policy_lw * lp + value_lw * lv, lp.mean(), lv.mean()


def eval_set_ce(model, s, batch_size, encoding_type):
    """Per-position unweighted policy and value CE on a prepped set. Leaves
    the model in eval mode."""
    from torch.amp import autocast
    X, P, Y = s[:3]
    device = torch.device("cuda")
    model.eval()
    lp = np.empty(len(X), dtype=np.float64)
    lv = np.empty(len(X), dtype=np.float64)
    with torch.no_grad(), autocast("cuda"):
        for i in range(0, len(X), batch_size):
            j = min(i + batch_size, len(X))
            pol, val = model(x_to_device(X[i:j], encoding_type, device))
            pb = torch.from_numpy(P[i:j]).float().to(device)
            yb = torch.from_numpy(Y[i:j]).float().to(device)
            lp[i:j] = (-(pb * F.log_softmax(pol.float(), dim=-1)).sum(dim=-1)
                       ).double().cpu().numpy()
            lv[i:j] = (-(yb * F.log_softmax(val.float(), dim=-1)).sum(dim=-1)
                       ).double().cpu().numpy()
    return lp, lv


def score_checkin_shard(cfg, epoch, model, arrays, batch_size, policy_lw, value_lw,
                        tag="[retrain]"):
    """Score the check-in shard, print it beside the means of the last
    CHECKIN_MEANS rows that exist, and append its row to CHECKIN_CSV: the
    record-weighted losses and the unweighted (unw_*) ones."""
    import csv
    s = prep_eval_set(arrays)
    up, uv = eval_set_ce(model, s, batch_size, cfg.encoding_type)
    vec = policy_lw * up * s[4] + value_lw * uv * s[3]
    lp, lv = (up * s[4]).mean(), (uv * s[3]).mean()
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
        "policy": round(lp, 6), "value": round(lv, 6),
        "unw_loss": round((policy_lw * up + value_lw * uv).mean(), 6),
        "unw_policy": round(up.mean(), 6), "unw_value": round(uv.mean(), 6)})


def tier_policy_rows(logits, tgt):
    """Per record, over legal moves: KL(target || net), the net's prob on the
    target's top move and on its top 3 (top1_mass / top3_mass in the
    validation csvs), and whether the net's top move is the target's."""
    legal = tgt > 0
    z = np.where(legal, logits.astype(np.float64), -np.inf)
    z -= z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    t1 = tgt.argmax(axis=1)
    rows = np.arange(len(tgt))
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(legal, tgt * (np.log(np.maximum(tgt, 1e-12)) - logp), 0).sum(1)
    top3 = np.argpartition(-tgt, 2, axis=1)[:, :3]
    return (kl, np.exp(logp[rows, t1]),
            np.exp(np.take_along_axis(logp, top3, axis=1)).sum(axis=1),
            logp.argmax(axis=1) == t1)


def score_tiers(cfg, epoch, model, ts, batch_size, tag="[retrain]"):
    """Start vs picked weights on the tier sample ts, grouped by each head's
    start-pass miss tier: one TIER_CSV row per head and tier, and a console
    line per head for the top tier and the middle tier."""
    from torch.amp import autocast
    device = torch.device("cuda")
    model.eval()
    n = len(ts["X"])
    wdl1 = np.zeros((n, 3), dtype=np.float64)
    pol_keys = ("kl", "top1_mass", "top3_mass", "top1_exact")
    pol = {k: np.zeros((2, n)) for k in pol_keys}
    with torch.no_grad(), autocast("cuda"):
        for i in range(0, n, TIER_CHUNK):
            j = min(i + TIER_CHUNK, n)
            logits1 = np.zeros((j - i, POLICY_DIM), dtype=np.float32)
            for a in range(i, j, batch_size):
                b = min(a + batch_size, j)
                p, v = model(x_to_device(ts["X"][a:b], cfg.encoding_type, device))
                logits1[a - i:b - i] = p.float().cpu().numpy()
                wdl1[a:b] = F.softmax(v.float(), dim=-1).cpu().numpy()
            tgt = ts["P"][i:j].astype(np.float64)
            for s, logits in enumerate((ts["logits0"][i:j], logits1)):
                for k, v in zip(pol_keys, tier_policy_rows(logits, tgt)):
                    pol[k][s, i:j] = v

    qt = ts["Y"][:, 0] - ts["Y"][:, 2]
    gap = [np.abs(w[:, 0] - w[:, 2] - qt) for w in (ts["wdl0"], wdl1)]

    def verdict(q):
        return np.sign(np.where(np.abs(q) > 0.3, q, 0))
    vmatch = [verdict(w[:, 0] - w[:, 2]) == verdict(qt) for w in (ts["wdl0"], wdl1)]

    out = {}
    for head, tier in (("policy", ts["tier_p"]), ("value", ts["tier_v"])):
        for t, name in enumerate(ts["names"]):
            m = tier == t
            row = {"epoch": epoch, "head": head, "tier": name, "n": int(m.sum())}
            if head == "policy":
                for k, f in (("kl", np.median), ("top1_mass", np.median),
                             ("top3_mass", np.median), ("top1_exact", np.mean)):
                    row[k + "_before"] = round(float(f(pol[k][0, m])), 6)
                    row[k + "_after"] = round(float(f(pol[k][1, m])), 6)
            else:
                row.update(gap_before=round(float(np.median(gap[0][m])), 6),
                           gap_after=round(float(np.median(gap[1][m])), 6),
                           toward=round(float(np.mean(gap[1][m] < gap[0][m])), 6),
                           verdict_before=round(float(np.mean(vmatch[0][m])), 6),
                           verdict_after=round(float(np.mean(vmatch[1][m])), 6))
            out[head, name] = row
    cols = ["epoch", "head", "tier", "n", "kl_before", "kl_after",
            "top1_mass_before", "top1_mass_after", "top3_mass_before",
            "top3_mass_after", "top1_exact_before", "top1_exact_after",
            "gap_before", "gap_after",
            "toward", "verdict_before", "verdict_after"]
    for row in out.values():
        write_csv_row(cfg, TIER_CSV, {c: row.get(c, "") for c in cols})

    top, mid = ts["names"][-1], ts["names"][1]

    def pol_str(r):
        return (f"top3 {r['top3_mass_before']:.3f}->{r['top3_mass_after']:.3f}  "
                f"top1 exact {r['top1_exact_before']:.1%}->"
                f"{r['top1_exact_after']:.1%}")

    def val_str(r):
        return (f"gap {r['gap_before']:.3f}->{r['gap_after']:.3f}  "
                f"toward {r['toward']:.0%}")
    for head, fmt in (("policy", pol_str), ("value", val_str)):
        print(f"{tag} tier {head:<6} {top} {fmt(out[head, top])}  |  "
              f"{mid} {fmt(out[head, mid])}")


def cpu_state(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def retrain_pt(model_path, X, P, Y_wdl, vwht, pwht, cfg, epoch, args,
               label="", timings=None, model=None, arch=None, holdout=None,
               tracked=None, checkin=None, tiers=None):
    """Load, train, save. Without holdout: one epoch. With holdout
    (X, P, Y, vwht, pwht of the held-out shard): up to MAX_EPOCHS under
    patience, saving the picked weights. tracked maps a name to an in-sample
    set's (arrays, start policy CE, start value CE), scored with the held-out
    shard from the start CE on; checkin is the check-in shard's arrays; tiers
    is the score_tiers sample, scored on the picked weights."""
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
        tracked = tracked or {}
        eval_sets = {"held_out": holdout, **{k: v[0] for k, v in tracked.items()}}
        eval_sets = {k: prep_eval_set(v) for k, v in eval_sets.items()}

        def score_sets(names=None):
            """Per-position losses of the eval sets (all, or names), held-out
            first, and their means keyed by csv column."""
            vecs = {k: eval_set_losses(model, s, args.batch_size, policy_lw,
                                       value_lw, cfg.encoding_type)[0]
                    for k, s in eval_sets.items() if names is None or k in names}
            set_train_mode()
            return vecs, set_means(vecs)

        def set_means(vecs):
            return {("loss" if k == "held_out" else f"{k}_loss"): round(v.mean(), 6)
                    for k, v in vecs.items()}

        # the tracked sets' start loss comes from the worker's start pass
        vecs, _ = score_sets(["held_out"])
        for k, (_, lp0, lv0) in tracked.items():
            s = eval_sets[k]
            vecs[k] = policy_lw * lp0 * s[4] + value_lw * lv0 * s[3]
        means = set_means(vecs)
        pat = {"eval_at": set(patience_eval_batches(epoch_batches)),
               "best": None, "best_vec": None, "best_batch": 0,
               "fails": 0, "drops": 0, "lr": lr, "stop": False,
               "best_state": None, "epoch1_state": None,
               "best_opt": None, "epoch1_opt": None,
               "hb": vecs["held_out"].mean(), "means": {0: means},
               "fit_low": {k: v for k, v in means.items() if k != "loss"}}
        print_patience_header(tag)
        print_patience_row(tag, 0, 0.0, means, None, None, None, "start", lr_floor)
        write_csv_row(cfg, PATIENCE_CSV, {
            "epoch": epoch, "batch": 0, "epoch_frac": 0.0, "lr": lr_floor, **means,
            "best_loss": "", "thresh": "", "rel_gain": "", "gate_a": "", "z": "",
            "gate_b": "", "fails": 0, "drops": 0, "action": "start"})

    def patience_eval(nb):
        # the tracked sets are scored from the end of epoch 1, the first eval
        # the fit check reads
        vecs, means = score_sets(None if nb >= epoch_batches else ["held_out"])
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
        marks = None
        if nb > epoch_batches:
            thresh = pat["hb"] * (1.0 + HELD_TOL)
            fit = {k: means[k] <= low * (1.0 - FIT_MIN_GAIN)
                   for k, low in pat["fit_low"].items()}
            held_ok = loss_now <= thresh or z > PAIRED_Z
            gate_b = all(fit.values()) and held_ok
            marks = {k: "+" if ok else "-" for k, ok in {**fit, "loss": held_ok}.items()}
        if nb == epoch_batches:
            pat["epoch1_state"] = cpu_state(model)
            pat["epoch1_opt"] = cpu_opt_state(model, opt)
        if gate_b and nb > epoch_batches:
            pat.update(best_batch=nb, fails=0)
            if z > PAIRED_Z:
                pat.update(best=loss_now, best_vec=vec)
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
        if nb == FORCE_DROP_BATCH and pat["drops"] < LR_DROPS:
            pat.update(drops=LR_DROPS, fails=0, lr=lr_floor)
            for g in opt.param_groups:
                g["lr"] = lr_floor
            action += "+drop"
        pat["hb"] = min(pat["hb"], loss_now)
        for k in pat["fit_low"]:
            if k in means:
                pat["fit_low"][k] = min(pat["fit_low"][k], means[k])
        print_patience_row(tag, nb, nb / epoch_batches, means,
                           None if first else prev, None if first else z,
                           thresh, action, pat["lr"], marks)
        write_csv_row(cfg, PATIENCE_CSV, {
            "epoch": epoch, "batch": nb, "epoch_frac": round(nb / epoch_batches, 4),
            "lr": pat["lr"], **means, "best_loss": round(prev, 6),
            "thresh": "" if thresh is None else round(thresh, 6),
            "rel_gain": round(1 - loss_now / prev, 6), "gate_a": int(gate_a),
            "z": round(z, 3), "gate_b": int(gate_b), "fails": pat["fails"],
            "drops": pat["drops"], "action": action})

    def forward_loss(xb, pb, yb, vwb, pwb):
        with autocast("cuda"):
            policy_logits, value_out = model(xb)

            log_probs   = F.log_softmax(policy_logits, dim=-1)
            policy_loss = (-(pb * log_probs).sum(dim=-1) * pwb).mean()

            log_wdl    = F.log_softmax(value_out, dim=-1)
            value_loss = (-(yb * log_wdl).sum(dim=-1) * vwb).mean()

            loss = policy_lw * policy_loss + value_lw * value_loss
            g_loss = None
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
        return loss, policy_loss, value_loss, g_loss

    epoch_losses     = []
    epoch_grad_stats = []
    nan_skips        = 0
    overflows        = []   # (batch, retries, loss finite, skipped)
    scale_start      = scaler.get_scale()
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
                retries = 0
                while True:
                    opt.zero_grad()
                    loss, policy_loss, value_loss, g_loss = forward_loss(
                        xb, pb, yb, vwb, pwb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    raw_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_norm).item()
                    loss_ok = np.isfinite(loss.item())
                    if (np.isfinite(raw_norm) or not loss_ok
                            or retries == MAX_STEP_RETRIES):
                        break
                    retries += 1
                    scaler.update(new_scale=scaler.get_scale() / 2)

                skipped = not (np.isfinite(raw_norm) and loss_ok)
                if retries or skipped:
                    overflows.append((nb, retries, loss_ok, skipped))
                if not loss_ok:
                    d = os.path.join(cfg.run_dir, INF_LOSS_DIR)
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, f"{epoch}_b{nb}.pkl"), "wb") as f:
                        pickle.dump({k: t.cpu().numpy() for k, t in (
                            ("X", xb), ("P", pb), ("Y", yb), ("vw", vwb),
                            ("pw", pwb))}, f)
                if skipped:
                    nan_skips += 1
                    if np.isfinite(raw_norm):
                        # finite grads would step: only reset the scaler
                        scaler.update(new_scale=scaler.get_scale())
                    else:
                        # scaler.step skips the update
                        scaler.step(opt)
                        scaler.update()
                    opt.zero_grad()
                else:
                    if g_loss is not None:
                        g_losses.append(g_loss.detach())
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
        if tiers is not None:
            score_tiers(cfg, epoch, model, tiers, args.batch_size, tag=tag)

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
    if overflows:
        inf = [b for b, _, ok, _ in overflows if not ok]
        print(f"{tag} overflow {len(overflows)}: "
              f"{sum(not s for *_, s in overflows)} recovered, {nan_skips} skipped, "
              f"{sum(r for _, r, _, _ in overflows)} retries  "
              f"scale {scale_start:.0f} -> {scaler.get_scale():.0f}")
        if inf:
            print(f"{tag}   inf loss at b{', b'.join(map(str, inf))} -> "
                  f"{INF_LOSS_DIR}/")
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
