"""
analyze_pt_weights.py — PT equivalent of analyze_optimizer_slots.py + grad_tape_experiment.py

Sections:
  1. Weight health  — dead/frozen/healthy params per layer group
  2. SGD+Nesterov   — momentum buffer analysis: velocity, alignment, sign consistency
  3. Weight delta   — vs pretrain epoch-2000 checkpoint (same arch required)
  4. Gradient flow  — policy vs value head contribution, per-group |grad|

Usage:
  python scripts/analyze_pt_weights.py [--model PATH] [--pretrain PATH] [--sgd PATH]

Defaults to val_test paths.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

VAL_TEST_DIR  = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\val_test"
MULTI_DIR     = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\val_test_multi"

DEFAULT_MODEL    = os.path.join(VAL_TEST_DIR, "val_test_model.pt")
DEFAULT_PRETRAIN = os.path.join(MULTI_DIR, "16m-transformer_pt_ckpt2000.pt")
DEFAULT_OPT      = os.path.join(VAL_TEST_DIR, "train_ckpts", "val_test_model_opt_state.pt")
DEFAULT_SNAPSHOT = os.path.join(VAL_TEST_DIR, "grad_analysis_snapshot.pkl")

DEAD_THRESH   = 1e-8
FROZEN_THRESH = 1e-5
BATCH         = 512
VOCAB_SIZE    = 21
SEQ_LEN       = 64

# --- param grouping ---

def is_ln(name):
    if name.endswith(".w") or name.endswith(".b"):
        return True
    if re.search(r"\.(ln1|ln2|val_mix_ln|lnm|lnp"
                 r"|from_ln\d*|to_ln\d*"
                 r"|promo_mln|promo_ln)\.(weight|bias)$", name):
        return True
    if re.search(r"(policy_head|value_head)\.ln\.(weight|bias)$", name):
        return True
    return False


def param_group(name, skip_ln=True):
    """Semantic group for a named parameter. Returns None to skip."""
    if skip_ln and is_ln(name):
        return None
    if name in ("emb.weight", "pos.weight"):
        return "embed"
    m = re.match(r"pre\.(\d+)\.(c1|c2)", name)
    if m:
        return f"pre{int(m.group(1)):02d}_conv"
    m = re.match(r"blocks\.(\d+)\.(attn|ff1|ff2)", name)
    if m:
        blk = int(m.group(1))
        kind = m.group(2)
        return f"tx{blk:02d}_attn" if kind == "attn" else f"tx{blk:02d}_ff"
    # legacy txs.* naming (older archs)
    m = re.match(r"txs\.(\d+)\.(attn|ff1|ff2)", name)
    if m:
        blk = int(m.group(1))
        kind = m.group(2)
        return f"tx{blk:02d}_attn" if kind == "attn" else f"tx{blk:02d}_ff"
    if name.startswith("mix."):
        return "mix"
    if name.startswith("val_mix."):
        return "mix"
    if name.startswith("policy_head."):
        return "head_policy"
    if name.startswith("value_head."):
        return "head_value"
    return "other"


def depth_key(g):
    m = re.match(r"pre(\d+)_conv", g)
    if m:
        return (int(m.group(1)), 0)
    m = re.match(r"tx(\d+)_(attn|ff)", g)
    if m:
        blk = int(m.group(1))
        return (20 + blk, 0 if m.group(2) == "attn" else 1)
    order = {"embed": (-1, 0), "mix": (90, 0),
             "head_policy": (99, 1), "head_value": (99, 2), "other": (99, 9)}
    return order.get(g, (50, g))


# --- weight health ---

def analyze_weights(sd):
    groups = defaultdict(lambda: {
        "n": 0, "dead": 0, "frozen": 0,
        "abs_sum": 0.0, "abs_max": 0.0, "sq_sum": 0.0,
    })
    for name, w in sd.items():
        grp = param_group(name)
        if grp is None:
            continue
        w = w.float().numpy()
        g = groups[grp]
        n = w.size
        g["n"]        += n
        g["dead"]     += int(np.sum(np.abs(w) < DEAD_THRESH))
        g["frozen"]   += int(np.sum(np.abs(w) < FROZEN_THRESH))
        g["abs_sum"]  += float(np.sum(np.abs(w)))
        g["abs_max"]   = max(g["abs_max"], float(np.max(np.abs(w))))
        g["sq_sum"]   += float(np.sum(w * w))
    return groups


def print_weight_health(groups):
    hdr = (f"{'Group':<20} {'Params':>10} {'Dead%':>7} {'Frz%':>7}"
           f" {'|w|_mean':>10} {'|w|_max':>10} {'w_rms':>10}")
    print(hdr)
    print("-" * len(hdr))
    tot_n = tot_dead = tot_frz = 0
    for grp in sorted(groups, key=depth_key):
        g = groups[grp]
        n = g["n"]
        dp  = 100 * g["dead"]   / n
        fp  = 100 * g["frozen"] / n
        wm  = g["abs_sum"] / n
        wM  = g["abs_max"]
        rms = (g["sq_sum"] / n) ** 0.5
        tot_n    += n
        tot_dead += g["dead"]
        tot_frz  += g["frozen"]
        print(f"{grp:<20} {n:>10,} {dp:>6.1f}% {fp:>6.1f}%"
              f" {wm:>10.3e} {wM:>10.3e} {rms:>10.3e}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<20} {tot_n:>10,} {100*tot_dead/tot_n:>6.1f}% {100*tot_frz/tot_n:>6.1f}%")
    print(f"dead<{DEAD_THRESH:.0e}  frozen<{FROZEN_THRESH:.0e}")


# --- Adam optimizer state ---

def analyze_adam(sd, opt_state):
    """opt_state: raw state dict from torch.load (Adam).
    sd: model state_dict. param index order matches state_dict key order."""
    param_names = list(sd.keys())
    state = opt_state["state"]
    pg  = opt_state["param_groups"][0]
    lr  = pg["lr"]
    b1, b2 = pg["betas"]
    eps = pg["eps"]
    wd  = pg["weight_decay"]

    groups = defaultdict(lambda: {
        "n": 0,
        "m1_abs_sum": 0.0,   # sum |exp_avg|
        "v_sum": 0.0,        # sum exp_avg_sq
        "snr_sum": 0.0,      # sum |m1| / sqrt(v + eps)
        "w_abs_sum": 0.0,
    })

    for idx, name in enumerate(param_names):
        grp = param_group(name)
        if grp is None:
            continue
        if idx not in state:
            continue
        s    = state[idx]
        if "exp_avg" not in s:
            continue
        m1 = s["exp_avg"].float().numpy()
        v  = s["exp_avg_sq"].float().numpy()
        w  = sd[name].float().numpy()
        g  = groups[grp]
        n  = m1.size
        snr = np.abs(m1) / (np.sqrt(v) + eps)
        g["n"]          += n
        g["m1_abs_sum"] += float(np.sum(np.abs(m1)))
        g["v_sum"]      += float(np.sum(v))
        g["snr_sum"]    += float(np.sum(snr))
        g["w_abs_sum"]  += float(np.sum(np.abs(w)))

    return groups, lr, b1, b2, eps, wd


def print_adam(groups, lr, b1, b2, eps, wd):
    hdr = (f"{'Group':<20} {'Params':>10} {'|m1|_mean':>12} {'sqrt(v)_mean':>13}"
           f" {'snr_mean':>10} {'eff_step/w':>12}")
    print(f"lr={lr:.4g}  betas=({b1},{b2})  eps={eps:.1e}  weight_decay={wd}")
    print(hdr)
    print("-" * len(hdr))
    for grp in sorted(groups, key=depth_key):
        g = groups[grp]
        if g["n"] == 0:
            continue
        n    = g["n"]
        m1m  = g["m1_abs_sum"] / n
        sqv  = (g["v_sum"] / n) ** 0.5
        snr  = g["snr_sum"] / n
        wm   = g["w_abs_sum"] / n
        eff  = lr * snr / (wm + 1e-12)
        print(f"{grp:<20} {n:>10,} {m1m:>12.3e} {sqv:>13.3e}"
              f" {snr:>10.4f} {eff:>12.4f}")
    print()
    print("snr_mean: |m1| / sqrt(v+eps) per param — near 1 = consistent gradient")
    print("  direction, near 0 = noisy/conflicting gradients")
    print("eff_step/w: lr * snr / |w|_mean — effective relative update per step")


# --- weight delta vs pretrain ---

def analyze_delta(sd_cur, sd_pre):
    groups = defaultdict(lambda: {
        "n": 0, "delta_sum": 0.0, "delta_max": 0.0,
        "pre_sum": 0.0, "cur_sum": 0.0,
    })
    missing = []
    for name in sd_cur:
        grp = param_group(name)
        if grp is None:
            continue
        if name not in sd_pre:
            missing.append(name)
            continue
        w_cur = sd_cur[name].float().numpy()
        w_pre = sd_pre[name].float().numpy()
        if w_cur.shape != w_pre.shape:
            missing.append(f"{name}(shape mismatch)")
            continue
        d   = np.abs(w_cur - w_pre)
        g   = groups[grp]
        n   = d.size
        g["n"]         += n
        g["delta_sum"] += float(np.sum(d))
        g["delta_max"]  = max(g["delta_max"], float(np.max(d)))
        g["pre_sum"]   += float(np.sum(np.abs(w_pre)))
        g["cur_sum"]   += float(np.sum(np.abs(w_cur)))
    if missing:
        print(f"  [delta] skipped {len(missing)} params (missing/shape mismatch): {missing[:5]}")
    return groups


def print_delta(groups):
    hdr = (f"{'Group':<20} {'Params':>10} {'|delta|_mean':>13} {'delta_rel%':>11}"
           f" {'|delta|_max':>12}")
    print(hdr)
    print("-" * len(hdr))
    for grp in sorted(groups, key=depth_key):
        g = groups[grp]
        if g["n"] == 0:
            continue
        dm  = g["delta_sum"] / g["n"]
        pre = g["pre_sum"]   / g["n"]
        rel = 100.0 * dm / (pre + 1e-12)
        dM  = g["delta_max"]
        print(f"{grp:<20} {g['n']:>10,} {dm:>13.3e} {rel:>10.2f}% {dM:>12.3e}")
    print()
    print("delta_rel = |delta|_mean / |w_pretrain|_mean  (% drift from pretrain init)")


# --- gradient flow ---

def load_snapshot(path, n=BATCH):
    """Load pkl snapshot and return (X, P, Y, vwht, pwht) as np arrays."""
    import pickle
    with open(path, "rb") as fh:
        records = pickle.load(fh)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(records))[:n]
    X_list, P_list, Y_list, vw_list, pw_list = [], [], [], [], []
    for i in idx:
        x, _mask, policy, value, vw, pw = records[i]
        X_list.append(x)
        P_list.append(policy)
        Y_list.append(value)
        vw_list.append(vw)
        pw_list.append(pw)
    return (
        np.stack(X_list).astype(np.int64),
        np.stack(P_list).astype(np.float32),
        np.stack(Y_list).astype(np.float32),
        np.array(vw_list, dtype=np.float32),
        np.array(pw_list, dtype=np.float32),
    )


def analyze_gradients(model, snapshot_path=None, pol_lw=1.0, val_lw=1.0):
    """Run forward+backward on GPU, collect per-group grad stats.
    Uses real chess data from snapshot if available, else synthetic.
    pol_lw / val_lw should match cfg.policy_loss_weight / cfg.value_loss_weight."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    if snapshot_path and os.path.exists(snapshot_path):
        X_np, pol_np, wdl_np, vwht, pwht = load_snapshot(snapshot_path, n=BATCH)
        data_label = f"real chess data (B={BATCH}, from grad_analysis_snapshot.pkl)"
    else:
        rng    = np.random.default_rng(42)
        X_np   = rng.integers(0, VOCAB_SIZE, size=(BATCH, SEQ_LEN)).astype(np.int64)
        pol_np = rng.standard_normal((BATCH, 4288)).astype(np.float32)
        pol_np = np.exp(pol_np - pol_np.max(axis=1, keepdims=True))
        pol_np /= pol_np.sum(axis=1, keepdims=True)
        wdl_np = rng.dirichlet([1.5, 2.0, 1.5], size=BATCH).astype(np.float32)
        vwht   = np.ones(BATCH, dtype=np.float32)
        pwht   = np.ones(BATCH, dtype=np.float32)
        data_label = f"synthetic data (B={BATCH}) — run build_grad_snapshot.py for real data"
    print(f"  data: {data_label}")
    print(f"  loss weights: policy={pol_lw}  value={val_lw}")

    X_t  = torch.from_numpy(X_np).to(device)
    P_t  = torch.from_numpy(pol_np).to(device)
    Y_t  = torch.from_numpy(wdl_np).to(device)
    pw_t = torch.from_numpy(pwht).to(device)
    vw_t = torch.from_numpy(vwht).to(device)

    def run_pass(loss_fn):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        pol_logits, val_out = model(X_t)
        loss = loss_fn(pol_logits.float(), val_out.float(), P_t, Y_t, pw_t, vw_t)
        loss.backward()
        grads = {name: (p.grad.detach().cpu() if p.grad is not None else None)
                 for name, p in model.named_parameters()}
        return float(loss), grads

    def combined_loss(pol, val, P, Y, pw, vw):
        lp = (-(P * F.log_softmax(pol, dim=-1)).sum(-1) * pw).mean()
        lv = (-(Y * F.log_softmax(val, dim=-1)).sum(-1) * vw).mean()
        return pol_lw * lp + val_lw * lv

    def pol_only_loss(pol, val, P, Y, pw, vw):
        return pol_lw * (-(P * F.log_softmax(pol, dim=-1)).sum(-1) * pw).mean()

    def val_only_loss(pol, val, P, Y, pw, vw):
        return val_lw * (-(Y * F.log_softmax(val, dim=-1)).sum(-1) * vw).mean()

    loss_c, g_c = run_pass(combined_loss)
    loss_p, g_p = run_pass(pol_only_loss)
    loss_v, g_v = run_pass(val_only_loss)

    def collect(grads):
        out = defaultdict(lambda: {"n": 0, "abs_sum": 0.0, "w_abs_sum": 0.0})
        for name, p in model.named_parameters():
            grp = param_group(name)
            if grp is None or grads.get(name) is None:
                continue
            g_arr = grads[name].float().cpu().numpy()
            w_arr = p.detach().float().cpu().numpy()
            n = g_arr.size
            out[grp]["n"]         += n
            out[grp]["abs_sum"]   += float(np.sum(np.abs(g_arr)))
            out[grp]["w_abs_sum"] += float(np.sum(np.abs(w_arr)))
        return out

    gc = collect(g_c)
    gp = collect(g_p)
    gv = collect(g_v)

    return gc, gp, gv, loss_c, loss_p, loss_v


def print_gradients(gc, gp, gv, loss_c, loss_p, loss_v):
    print(f"loss_pol={loss_p:.4f}  loss_val={loss_v:.4f}  total={loss_c:.4f}")
    print()
    hdr = (f"{'Group':<20} {'|g|_comb':>12} {'|g|_pol':>12} {'|g|_val':>12}"
           f"  {'pol%':>6}  {'val%':>6}  {'g/w':>8}")
    print(hdr)
    print("-" * len(hdr))
    ref_groups = [k for k in gc if re.match(r"tx\d+_ff", k)]
    ref_c = np.mean([gc[k]["abs_sum"] / gc[k]["n"] for k in ref_groups]) if ref_groups else 1.0

    for grp in sorted(gc, key=depth_key):
        n  = gc[grp]["n"]
        gm_c = gc[grp]["abs_sum"] / n
        gm_p = gp[grp]["abs_sum"] / gp[grp]["n"] if grp in gp else 0.0
        gm_v = gv[grp]["abs_sum"] / gv[grp]["n"] if grp in gv else 0.0
        tot  = gm_p + gm_v
        pp   = 100 * gm_p / tot if tot > 0 else 0.0
        vp   = 100 * gm_v / tot if tot > 0 else 0.0
        wm   = gc[grp]["w_abs_sum"] / n
        gw   = gm_c / wm if wm > 0 else float("inf")
        print(f"{grp:<20} {gm_c:>12.3e} {gm_p:>12.3e} {gm_v:>12.3e}"
              f"  {pp:>5.1f}%  {vp:>5.1f}%  {gw:>8.4f}")
    print()
    print(f"backbone ff ref (combined): {ref_c:.3e}")
    print("g/w: gradient-to-weight ratio (< 0.001 suggests stale / not learning)")


# --- combined forward-pass analysis (sections 5 / 6) ---

def analyze_forward_sections(model, snapshot_path, chunk=256):
    """Single chunked forward pass capturing attention weights and tx00 input scales.
    Returns (tx_attn, head_attn, scale_x)
      tx_attn:   {block_idx: np.ndarray (H, Nq, Nk)} mean over all positions
      head_attn: {'value','from','to': np.ndarray (H, Nq, Nk)}
      scale_x:   first-chunk raw input to tx00, shape (b, 68, D)
    """
    import pickle
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = model.to(device).eval()
    n_blocks = len(model.blocks)

    with open(snapshot_path, "rb") as fh:
        records = pickle.load(fh)
    encs = np.stack([r[0] for r in records])

    cap_tx_attn = {}
    cap_head    = {}
    scale_x     = {}
    tx_attn_sum = [None] * n_blocks
    head_sum    = {}
    n_total     = 0

    def make_patched_tx(idx, blk):
        def patched_forward(x):
            if idx == 0 and not scale_x:
                scale_x['x'] = x.detach().cpu()
            r = x
            n = blk.ln1(x)
            h, aw = blk.attn(n, n, n, need_weights=True, average_attn_weights=False)
            cap_tx_attn[idx] = aw.detach().cpu()
            x   = r + blk.drop(h)
            return x + blk.ff2(F.gelu(blk.ff1(blk.ln2(x))))
        return patched_forward

    orig_tx = {}
    for i, blk in enumerate(model.blocks):
        orig_tx[i] = blk.forward
        blk.forward = make_patched_tx(i, blk)

    vh = model.value_head
    orig_vh = vh.forward
    def patched_vh(x):
        xp = F.gelu(vh.proj(vh.ln(x)))
        q  = vh.query.expand(xp.shape[0], -1, -1)
        h, aw = vh.mha(q, xp, xp, need_weights=True, average_attn_weights=False)
        cap_head['value'] = aw.detach().cpu()
        return vh.out(h.squeeze(1))
    vh.forward = patched_vh

    ph = model.policy_head
    orig_ph = ph.forward
    def patched_ph(x):
        B, N, C = x.shape
        s  = ph.ln(x)
        f  = F.gelu(ph.from_proj(s))
        fn = ph.from_ln(f)
        fh, aw_f = ph.from_mha(fn[:, :64], fn, fn, need_weights=True, average_attn_weights=False)
        cap_head['from'] = aw_f.detach().cpu()
        fv = ph.from_out(f[:, :64] + fh)
        t  = F.gelu(ph.to_proj(s))
        tn = ph.to_ln(t)
        th, aw_t = ph.to_mha(tn[:, :64], tn, tn, need_weights=True, average_attn_weights=False)
        cap_head['to'] = aw_t.detach().cpu()
        tv    = ph.to_out(t[:, :64] + th)
        dots  = torch.bmm(fv.float(), tv.float().transpose(1, 2)).mul(ph.scale).reshape(-1, 64 * 64)
        xp    = x[:, :64].reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
        p     = F.leaky_relu(ph.promo_mln(ph.promo_mix(xp)), 0.02)
        sk    = p
        p     = F.leaky_relu(ph.promo_ln(ph.promo_c1(p)), 0.02)
        promo = ph.promo_out(p + sk).permute(0, 2, 3, 1).reshape(-1, 192).float()
        logits = torch.cat([dots, promo], dim=1)
        logits = logits.masked_fill(~ph.sometimes_legal, -1e9)
        return logits.to(x.dtype)
    ph.forward = patched_ph

    with torch.no_grad():
        for start in range(0, len(encs), chunk):
            batch = torch.from_numpy(encs[start:start+chunk].astype(np.int64)).to(device)
            cap_tx_attn.clear(); cap_head.clear()
            model(batch)
            for i in range(n_blocks):
                s = cap_tx_attn[i].float().sum(dim=0)
                tx_attn_sum[i] = s if tx_attn_sum[i] is None else tx_attn_sum[i] + s
            for key in ('value', 'from', 'to'):
                s = cap_head[key].float().sum(dim=0)
                head_sum[key] = s if head_sum.get(key) is None else head_sum[key] + s
            n_total += batch.shape[0]

    for i, blk in enumerate(model.blocks):
        blk.forward = orig_tx[i]
    vh.forward = orig_vh
    ph.forward = orig_ph

    tx_attn   = {i: (tx_attn_sum[i] / n_total).numpy() for i in range(n_blocks)}
    head_attn = {k: (head_sum[k]    / n_total).numpy()  for k in head_sum}
    return tx_attn, head_attn, scale_x.get('x')


def print_global_tokens(captured, head_captured, n_board=64, n_global=4, save_dir=None):
    """captured: {block_idx: np.ndarray (H, N_q, N_k)} mean over batch
       head_captured: {'value','from','to': np.ndarray (H, N_q, N_k)}"""
    uniform = 1.0 / (n_board + n_global)

    hdr = (f"{'Block':<8} {'heads':>5}  "
           f"{'board->global':>14}  {'global->board':>14}  {'global->global':>15}  {'board->board':>13}")
    print(hdr)
    print("-" * len(hdr))
    for idx in sorted(captured):
        w       = captured[idx]           # (H, Nq, Nk)
        board_q = w[:, :n_board, :]
        glob_q  = w[:, n_board:, :]
        print(f"tx{idx:02d}     {w.shape[0]:>5}  "
              f"{board_q[:,:,n_board:].mean():>13.3%}  "
              f"{glob_q[:,:,:n_board].mean():>14.3%}  "
              f"{glob_q[:,:,n_board:].mean():>15.3%}  "
              f"{board_q[:,:,:n_board].mean():>13.3%}")
    print()
    print("board->global : how much board squares use global tokens as context")
    print("global->board : how much global tokens pull from board squares")
    print("global->global: self-attention among the 4 global tokens")
    print()

    def key_ratios(aw_np):
        """(H, Nq, Nk) -> (Nk,) ratio vs uniform."""
        return aw_np.mean(axis=(0, 1)) / uniform

    # build per-key ratio vectors for all 7 attention maps
    tx_ratios   = {idx: key_ratios(captured[idx])    for idx in sorted(captured)}
    val_ratios  = key_ratios(head_captured['value'])
    from_ratios = key_ratios(head_captured['from'])
    to_ratios   = key_ratios(head_captured['to'])
    avg_ratios  = np.mean(list(tx_ratios.values()) + [val_ratios, from_ratios, to_ratios], axis=0)

    def ratio_to_display(ratios_68):
        """(68,) ratio array -> (9,8) display grid with globals row."""
        disp = np.full((9, 8), np.nan)
        disp[0, 2] = ratios_68[n_board + 0]
        disp[0, 3] = ratios_68[n_board + 1]
        disp[0, 4] = ratios_68[n_board + 2]
        disp[0, 5] = ratios_68[n_board + 3]
        grid = ratios_68[:n_board].reshape(8, 8)
        for r in range(8):
            disp[1 + (7 - r), :] = grid[r, :]
        return disp

    print("Mean attention received per square (key), averaged over queries/heads/batch:")
    print("(ratio vs uniform 1/68  |  STM-as-white, rank 1=bottom)")
    print()
    all_panels = (
        [(f"tx{idx:02d}", tx_ratios[idx]) for idx in sorted(tx_ratios)]
        + [("value", val_ratios), ("pol from", from_ratios),
           ("pol to",  to_ratios), ("avg all 7", avg_ratios)]
    )
    for label, ratios in all_panels:
        grid = ratios[:n_board].reshape(8, 8)
        print(f"  {label}  (ratio vs uniform)")
        for rank in range(7, -1, -1):
            row = "  ".join(f"{grid[rank, f]:4.2f}" for f in range(8))
            print(f"  rank{rank+1}  {row}")
        print()

    if save_dir:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n_cols   = 4
            files    = list("abcdefgh")
            ylabels  = ["global"] + [str(r) for r in range(8, 0, -1)]
            titles   = [f"tx{idx:02d}" for idx in sorted(tx_ratios)]
            row2_titles = ["value head", "pol from-branch", "pol to-branch", "avg all 7"]
            all_titles  = titles + row2_titles
            all_ratios  = [tx_ratios[idx] for idx in sorted(tx_ratios)] + \
                          [val_ratios, from_ratios, to_ratios, avg_ratios]

            fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 9))

            for panel_i, (ax, title, ratios) in enumerate(zip(axes.flat, all_titles, all_ratios)):
                disp = ratio_to_display(ratios)
                im   = ax.imshow(disp, vmin=0.5, vmax=2.0, cmap="RdYlGn", aspect="equal")
                ax.set_title(f"{title}\nattn received (ratio vs uniform)", fontsize=9)
                ax.set_xticks(range(8)); ax.set_xticklabels(files, fontsize=7)
                ax.set_yticks(range(9)); ax.set_yticklabels(ylabels, fontsize=7)
                for row in range(1, 9):
                    for f in range(8):
                        v = disp[row, f]
                        if not np.isnan(v):
                            ax.text(f, row, f"{v:.2f}", ha="center", va="center",
                                    fontsize=6, color="black")
                for gi, col in enumerate([2, 3, 4, 5]):
                    v = disp[0, col]
                    if not np.isnan(v):
                        ax.text(col, 0, f"g{gi}\n{v:.2f}", ha="center", va="center",
                                fontsize=6, color="black")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.suptitle("Attention received per position (ratio vs uniform 1/68 baseline)\n"
                         "top: transformer blocks  |  bottom: heads + overall average", fontsize=11)
            plt.tight_layout()
            out_path = os.path.join(save_dir, "global_token_attn_heatmap.png")
            plt.savefig(out_path, dpi=120)
            plt.close()
            print(f"  heatmap saved -> {out_path}")
        except Exception as e:
            print(f"  [heatmap] could not save: {e}")


def print_scale_diagnostic(scale_x):
    """Print tx00 input scale stats from first-chunk raw activations (B, 68, D)."""
    x = scale_x.float()
    conv_slice   = x[:, :64, :256]
    pos_slice    = x[:, :64, 256:]
    global_slice = x[:, 64:, :]

    def stats(t, label):
        rms  = t.pow(2).mean().sqrt().item()
        mean = t.abs().mean().item()
        mx   = t.abs().max().item()
        print(f"  {label:<20}  shape={tuple(t.shape)}  rms={rms:.4f}  |mean|={mean:.4f}  max={mx:.4f}")

    print(f"  (B={x.shape[0]} positions, raw activations before tx00 ln1)")
    stats(conv_slice,   "conv_seq  [:64, :256]")
    stats(pos_slice,    "pos_emb   [:64, 256:]")
    stats(global_slice, "globals   [64:,  :]  ")
    print()
    conv_rms   = conv_slice.pow(2).mean().sqrt().item()
    pos_rms    = pos_slice.pow(2).mean().sqrt().item()
    global_rms = global_slice.pow(2).mean().sqrt().item()
    print(f"  scale ratios:  global/conv={global_rms/conv_rms:.2f}x  "
          f"global/pos={global_rms/pos_rms:.2f}x  "
          f"pos/conv={pos_rms/conv_rms:.2f}x")


# --- per-position pol-from attention ---

def analyze_per_position(model, snapshot_path, save_dir=None):
    import pickle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    with open(snapshot_path, "rb") as fh:
        records = pickle.load(fh)

    PIECE_GLYPHS = {0: ".", 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
                    7: "K", 8: "K", 9: "K",
                    10: "p", 11: "n", 12: "b", 13: "r", 14: "q", 15: "k",
                    16: "k", 17: "k", 18: "k", 19: "e"}

    results = []  # list of (enc, from_attn_per_key_68)

    ph = model.policy_head
    orig_ph = ph.forward

    for idx in range(len(records)):
        rec = records[int(idx)]
        enc = rec[0]  # (64,) int tokens
        X_t = torch.from_numpy(enc.astype(np.int64)).unsqueeze(0).to(device)

        captured_from = {}

        def patched_ph(x):
            B, N, C = x.shape
            s  = ph.ln(x)
            f  = F.gelu(ph.from_proj(s))
            fn = ph.from_ln(f)
            fh, aw_f = ph.from_mha(fn[:, :64], fn, fn,
                                    need_weights=True, average_attn_weights=False)
            captured_from['w'] = aw_f.detach().cpu()   # (1, H, 64, 68)
            fv = ph.from_out(f[:, :64] + fh)
            t  = F.gelu(ph.to_proj(s))
            tn = ph.to_ln(t)
            th, _ = ph.to_mha(tn[:, :64], tn, tn, need_weights=False)
            tv    = ph.to_out(t[:, :64] + th)
            dots  = torch.bmm(fv.float(), tv.float().transpose(1, 2)).mul(ph.scale).reshape(-1, 64 * 64)
            xp    = x[:, :64].reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
            p     = F.leaky_relu(ph.promo_mln(ph.promo_mix(xp)), 0.02)
            sk    = p
            p     = F.leaky_relu(ph.promo_ln(ph.promo_c1(p)), 0.02)
            promo = ph.promo_out(p + sk).permute(0, 2, 3, 1).reshape(-1, 192).float()
            logits = torch.cat([dots, promo], dim=1)
            logits = logits.masked_fill(~ph.sometimes_legal, -1e9)
            return logits.to(x.dtype)

        ph.forward = patched_ph
        with torch.no_grad():
            model(X_t)
        ph.forward = orig_ph

        aw = captured_from['w'].float().numpy()   # (1, H, 64, 68)
        attn_per_key = aw.mean(axis=(0, 1, 2))   # (68,) mean over batch/heads/queries
        results.append((enc, attn_per_key))

    if save_dir:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            uniform  = 1.0 / 68
            files    = list("abcdefgh")
            ylabels  = ["global"] + [str(r) for r in range(8, 0, -1)]
            ncols    = 4
            nrows    = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 5 * nrows))
            axes = np.array(axes).reshape(-1)

            for ax, (enc, attn_per_key) in zip(axes, results):
                ratios = attn_per_key / uniform
                disp   = np.full((9, 8), np.nan)
                disp[0, 2] = ratios[64]; disp[0, 3] = ratios[65]
                disp[0, 4] = ratios[66]; disp[0, 5] = ratios[67]
                grid = ratios[:64].reshape(8, 8)
                for r in range(8):
                    disp[1 + (7 - r), :] = grid[r, :]

                im = ax.imshow(disp, vmin=0.5, vmax=2.0, cmap="RdYlGn", aspect="equal")

                ax.set_xticks(range(8)); ax.set_xticklabels(files, fontsize=7)
                ax.set_yticks(range(9)); ax.set_yticklabels(ylabels, fontsize=7)

                enc_grid = enc.reshape(8, 8)
                for r in range(8):
                    for f in range(8):
                        token   = int(enc_grid[r, f])
                        glyph   = PIECE_GLYPHS.get(token, "?")
                        row_idx = 1 + (7 - r)
                        v       = disp[row_idx, f]
                        occupied = token != 0
                        label = f"{glyph}\n{v:.2f}" if occupied else f"{v:.2f}"
                        color = "blue" if occupied else "black"
                        ax.text(f, row_idx, label, ha="center", va="center",
                                fontsize=6, color=color, fontweight="bold" if occupied else "normal")
                for gi, col in enumerate([2, 3, 4, 5]):
                    v = disp[0, col]
                    if not np.isnan(v):
                        ax.text(col, 0, f"g{gi}\n{v:.2f}", ha="center", va="center",
                                fontsize=6, color="black")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title("pol from-branch attn received\n(ratio vs uniform)", fontsize=8)

            for ax in axes[n:]:
                ax.axis("off")

            fig.suptitle("Per-position policy from-branch attention\n"
                         "blue=occupied square | ratio vs uniform 1/68", fontsize=11)
            plt.tight_layout()
            out_path = os.path.join(save_dir, "per_position_pol_from_attn.png")
            plt.savefig(out_path, dpi=120)
            plt.close()
            print(f"  saved -> {out_path}")
        except Exception as e:
            print(f"  [per-position] could not save: {e}")


# --- LN scalar health (separate check) ---

def check_ln_params(sd):
    dead_ln = []
    for name, w in sd.items():
        if not is_ln(name):
            continue
        w = w.float().numpy()
        abs_mean = float(np.mean(np.abs(w)))
        if abs_mean < 1e-4:
            dead_ln.append((name, abs_mean))
    if dead_ln:
        print(f"  WARNING: {len(dead_ln)} LN param tensors near-zero (possible dead LN):")
        for nm, v in dead_ln[:10]:
            print(f"    {nm}  mean={v:.2e}")
    else:
        print(f"  LN params OK (all mean |w| > 1e-4)")




# --- main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help="path to val_test PT checkpoint (.pt)")
    parser.add_argument("--pretrain", default=DEFAULT_PRETRAIN,
                        help="path to pretrain reference checkpoint (.pt)")
    parser.add_argument("--opt",      default=DEFAULT_OPT,
                        help="path to optimizer state (.pt)")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT,
                        help="path to grad_analysis_snapshot.pkl (built by build_grad_snapshot.py)")
    parser.add_argument("--pol-lw",  type=float, default=None,
                        help="policy loss weight (default: read from config.yaml)")
    parser.add_argument("--val-lw",  type=float, default=None,
                        help="value loss weight (default: read from config.yaml)")
    parser.add_argument("--per-position", type=int, default=0,
                        help="if >0, run per-position pol-from analysis on N random positions")
    args = parser.parse_args()

    print(f"model:    {args.model}")
    print(f"pretrain: {args.pretrain}")
    print(f"opt:      {args.opt}")
    print()

    # load model
    ckpt = torch.load(args.model, map_location="cpu")
    sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    arch = ckpt.get("arch", "?") if isinstance(ckpt, dict) else "?"
    print(f"arch={arch}  params={len(sd)}")
    print()

    # ---- 1. weight health ----
    print("=" * 72)
    print("1. WEIGHT HEALTH")
    print("=" * 72)
    wh = analyze_weights(sd)
    print_weight_health(wh)
    print()
    print("LN param check:")
    check_ln_params(sd)
    print()

    # ---- 2. Adam optimizer state ----
    if os.path.exists(args.opt):
        print("=" * 72)
        print("2. ADAM OPTIMIZER STATE")
        print("=" * 72)
        opt_state = torch.load(args.opt, map_location="cpu")
        mg, lr, b1, b2, eps, wd = analyze_adam(sd, opt_state)
        print_adam(mg, lr, b1, b2, eps, wd)
        print()
    else:
        print(f"[opt] state not found at {args.opt} — skipping section 2")
        print()

    # ---- 3. weight delta vs pretrain ----
    if os.path.exists(args.pretrain):
        print("=" * 72)
        print("3. WEIGHT DELTA vs PRETRAIN")
        print("=" * 72)
        pre_ckpt = torch.load(args.pretrain, map_location="cpu")
        sd_pre   = pre_ckpt["model"] if isinstance(pre_ckpt, dict) and "model" in pre_ckpt else pre_ckpt
        pre_arch = pre_ckpt.get("arch", "?") if isinstance(pre_ckpt, dict) else "?"
        pre_ep   = pre_ckpt.get("epoch", "?") if isinstance(pre_ckpt, dict) else "?"
        print(f"reference: arch={pre_arch}  epoch={pre_ep}  params={len(sd_pre)}")
        dg = analyze_delta(sd, sd_pre)
        print_delta(dg)
    else:
        print(f"[delta] pretrain ckpt not found at {args.pretrain} — skipping section 3")
    print()

    # ---- 4. gradient flow ----
    print("=" * 72)
    print("4. GRADIENT FLOW")
    print("=" * 72)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from chessbot.model import PT_BUILDERS, VARIANTS

    pol_lw = args.pol_lw
    val_lw = args.val_lw
    if pol_lw is None or val_lw is None:
        cfg_yaml = os.path.join(VAL_TEST_DIR, "config.yaml")
        try:
            import yaml
            with open(cfg_yaml) as fh:
                cfg_raw = yaml.safe_load(fh)
            pol_lw = pol_lw if pol_lw is not None else float(cfg_raw.get("policy_loss_weight", 1.0))
            val_lw = val_lw if val_lw is not None else float(cfg_raw.get("value_loss_weight", 1.0))
        except Exception as e:
            print(f"  [grad] could not read config.yaml ({e}), defaulting to 1.0/1.0")
            pol_lw = pol_lw or 1.0
            val_lw = val_lw or 1.0

    if arch not in PT_BUILDERS:
        print(f"[grad] arch '{arch}' not in PT_BUILDERS — skipping section 4")
        return
    model = PT_BUILDERS[arch](VARIANTS[arch])
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=True)
    if missing_keys:
        print(f"  [grad] missing keys: {missing_keys[:5]}")
    gc, gp, gv, lc, lp, lv = analyze_gradients(
        model, snapshot_path=args.snapshot, pol_lw=pol_lw, val_lw=val_lw)
    print_gradients(gc, gp, gv, lc, lp, lv)

    # ---- 5 / 6 / 7. single forward pass ----
    if hasattr(model, "global_tokens") and args.snapshot and os.path.exists(args.snapshot):
        model_fwd = PT_BUILDERS[arch](VARIANTS[arch])
        model_fwd.load_state_dict(sd, strict=True)
        tx_attn, head_attn, scale_x = analyze_forward_sections(model_fwd, args.snapshot)

        print("=" * 72)
        print("5. GLOBAL TOKEN ATTENTION")
        print("=" * 72)
        print_global_tokens(tx_attn, head_attn, save_dir=os.path.dirname(args.model))

        print("=" * 72)
        print("6. TX00 INPUT SCALE DIAGNOSTIC")
        print("=" * 72)
        if scale_x is not None:
            print_scale_diagnostic(scale_x)

    # ---- 8. per-position pol-from ----

    if args.per_position > 0 and hasattr(model, "global_tokens") and \
            args.snapshot and os.path.exists(args.snapshot):
        print("=" * 72)
        print("8. PER-POSITION POL-FROM ATTENTION")
        print("=" * 72)
        model3 = PT_BUILDERS[arch](VARIANTS[arch])
        model3.load_state_dict(sd, strict=True)
        analyze_per_position(model3, args.snapshot, save_dir=VAL_TEST_DIR)


if __name__ == "__main__":
    main()
