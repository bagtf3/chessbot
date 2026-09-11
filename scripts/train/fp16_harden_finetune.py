"""fp16 hardening refit: drive every RMS norm input below tau via a
per-element sum-ReLU penalty on |x|, so TRT fp16 conversion is safe.

tau starts at tau_start (500) and drops 5/10/5/10... each epoch until it
reaches the target (225). lam stays flat at lam0 for lam_warmup epochs, then
log-ramps to lam1 over the remainder.

    python scripts/train/fp16_harden_finetune.py
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.amp import autocast
from dotenv import load_dotenv

load_dotenv(r"C:\Users\Bryan\repos\chessbot\.env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bootstrap_model_async_sparse_pkl as B
from chessbot.pretrain import EPOCH_SIZE, PLOT_EVERY, split_train_val

NAME       = "precond-mha-10c6t-d256-wdl3"
RUN_DIR    = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\20m_wdl3_gatereg_exp"
START_CKPT = os.path.join(RUN_DIR, f"{NAME}_pt_ckpt28900.pt")
START_EP   = 28901


def tau_for_epoch(n, tau_start, tau_target, flat=10):
    """Flat at tau_start for `flat` epochs, then drop 5/10 alternating, floor at tau_target."""
    if n < flat:
        return tau_start
    drops = n - flat
    cum = (drops // 2) * 15 + (5 if drops % 2 == 1 else 0)
    return max(tau_target, tau_start - cum)


def lam_for_epoch(n, lam0, lam1, warmup):
    """Flat at lam0 for warmup epochs, then log-ramp to lam1."""
    if n < warmup:
        return lam0
    frac = (n - warmup) / max(1, (1000 - warmup) - 1)
    return lam0 * (lam1 / lam0) ** frac


def install_hooks(model, cap):
    site_names = []

    def make_hook(name):
        def hook(m, inp):
            x = inp[0]
            with torch.no_grad():
                v = float(x.detach().abs().max())
                if v > cap["site_max"].get(name, 0.0):
                    cap["site_max"][name] = v
            if v > cap["tau"]:
                cap["acts"].append(x)
        return hook

    for name, mod in model.named_modules():
        if mod.__class__.__name__ in ("Rms1d", "Rms2d"):
            mod.register_forward_pre_hook(make_hook(name))
            site_names.append(name)

    print(f"[harden] hooked {len(site_names)} pow(2) sites", flush=True)
    return site_names


def epoch_penalty(acts, tau, device):
    if not acts:
        return torch.zeros((), dtype=torch.float32, device=device), 0.0
    pen = acts[0].new_zeros((), dtype=torch.float32)
    peak = 0.0
    for x in acts:
        pen = pen + torch.relu(x.abs() - tau).float().sum()
        with torch.no_grad():
            peak = max(peak, float(x.detach().abs().max()))
    return pen, peak


def fit_epoch(model, opt, scaler, bundle, lw, cap, lam, tau, device):
    enc      = torch.as_tensor(bundle["enc_in"], dtype=torch.long, device=device)
    policy_t = torch.as_tensor(bundle["policy_logits"], dtype=torch.float32, device=device)
    value_t  = torch.as_tensor(bundle["value_out"], dtype=torch.float32, device=device)
    w        = torch.as_tensor(bundle["weight"], dtype=torch.float32, device=device)

    perm = torch.randperm(enc.shape[0], device=device)
    model.train()
    tp = tv = tt = tpen = 0.0
    peak_ep = 0.0
    n_batches = 0
    cap["site_max"].clear()
    cap["gg_max"] = cap["gu_max"] = cap["h_max"] = 0.0
    cap["tau"] = tau

    for start in range(0, enc.shape[0], B.PT_BATCH_SIZE):
        idx = perm[start:start + B.PT_BATCH_SIZE]
        opt.zero_grad(set_to_none=True)
        cap["acts"] = []

        with autocast("cuda"):
            pol, val = model(enc[idx])
            p_loss   = F.cross_entropy(pol, policy_t[idx]) * lw["policy_logits"]
            v_loss   = (F.cross_entropy(val, value_t[idx], reduction="none")
                        * w[idx]).mean() * lw["value_out"]
            pen, peak = epoch_penalty(cap["acts"], tau, enc.device)
            loss = p_loss + v_loss + lam * pen

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()

        tp += p_loss.item()
        tv += v_loss.item()
        tt += loss.item()
        tpen += float(pen)
        peak_ep = max(peak_ep, peak)
        n_batches += 1

    nb = max(n_batches, 1)
    return tp / nb, tv / nb, tt / nb, tpen / nb, peak_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",      type=int,   default=1000)
    ap.add_argument("--tau",         type=float, default=225.0)
    ap.add_argument("--tau_start",   type=float, default=225.0)
    ap.add_argument("--lam0",        type=float, default=3e-7)
    ap.add_argument("--lam1",        type=float, default=1e-3)
    ap.add_argument("--lam_warmup",  type=int,   default=0)
    args = ap.parse_args()

    end_ep = START_EP + args.epochs
    device = torch.device("cuda")

    print(f"[harden] loading {START_CKPT}", flush=True)
    model, opt, scaler, _ = B.load_pt_model(START_CKPT, NAME, device, lr=B.LR_MIN)
    for pg in opt.param_groups:
        pg["lr"] = B.LR_MIN

    print(
        f"[harden] lr={B.LR_MIN:.2e} (flat)"
        f"  tau {args.tau_start:.0f} -> {args.tau:.0f} (alternating 5/10 drops)"
        f"  lam {args.lam0:.1e} flat {args.lam_warmup}ep then log-ramp -> {args.lam1:.1e}"
        f"  epochs {START_EP}..{end_ep-1}",
        flush=True)

    cap = {"acts": [], "site_max": {}, "tau": args.tau_start,
           "gg_max": 0.0, "gu_max": 0.0, "h_max": 0.0}
    site_names = install_hooks(model, cap)

    def out_hook(key):
        def hook(m, inp, out):
            with torch.no_grad():
                v = float(out.detach().abs().max())
                if v > cap[key]:
                    cap[key] = v
        return hook

    def pre_hook(key):
        def hook(m, inp):
            with torch.no_grad():
                v = float(inp[0].detach().abs().max())
                if v > cap[key]:
                    cap[key] = v
        return hook

    H_TAU = 128.0

    model.gate_w_gate.register_forward_hook(out_hook("gg_max"))
    model.gate_w_up.register_forward_hook(out_hook("gu_max"))

    def h_hook(m, inp):
        h = inp[0]
        with torch.no_grad():
            v = float(h.detach().abs().max())
            if v > cap["h_max"]:
                cap["h_max"] = v

    model.gate_w_down.register_forward_pre_hook(h_hook)

    all_files = sorted(glob.glob(os.path.join(B.DEFAULT_SHARD_DIR, "*.pkl.gz")))
    train_files, val_files = split_train_val(all_files)
    print(f"[harden] shards train={len(train_files)} val={len(val_files)}", flush=True)

    train_pool = B.AsyncShardPool(
        train_files, high=B.BUFFER_CAP, low=B.BUFFER_CAP - B.REFILL_BAND,
        epoch_size=EPOCH_SIZE, drain=True, n_ready=2, name="train")
    val_pool = B.AsyncShardPool(
        val_files, high=B.VAL_BUFFER_CAP, low=B.VAL_BUFFER_CAP - B.REFILL_BAND,
        epoch_size=EPOCH_SIZE, drain=True, n_ready=1, name="val")
    train_pool.start()
    val_pool.start()

    progress_file = os.path.join(RUN_DIR, "harden_eval_progress.csv")
    plot_file     = os.path.join(RUN_DIR, f"{NAME}_harden_plot.png")
    eval_df = None
    policy_ce_hat = value_ce_hat = None
    active_lw = {"policy_lw": B.LW_WARMUP_POLICY, "value_lw": B.LW_WARMUP_VALUE,
                 "scale": None}

    for ep in range(START_EP, end_ep):
        n   = ep - START_EP
        tau = tau_for_epoch(n, args.tau_start, args.tau)
        lam = lam_for_epoch(n, args.lam0, args.lam1, args.lam_warmup)

        if policy_ce_hat is None:
            policy_lw, value_lw = B.LW_WARMUP_POLICY, B.LW_WARMUP_VALUE
        else:
            policy_lw, value_lw, _, _ = B.lw_for_epoch(
                ep, policy_ce_hat, value_ce_hat, active_lw)
        active_lw = {"policy_lw": policy_lw, "value_lw": value_lw, "scale": None}
        lw = {"policy_logits": policy_lw, "value_out": value_lw}

        t0 = time.time()
        bundle = train_pool.get()
        p, v, tot, pen, peak = fit_epoch(
            model, opt, scaler, bundle, lw, cap, lam, tau, device)
        policy_ce_hat = B.lw_ema_update(policy_ce_hat, p / policy_lw)
        value_ce_hat  = B.lw_ema_update(value_ce_hat,  v / value_lw)

        site_maxes = cap["site_max"]
        top5     = sorted(site_maxes.items(), key=lambda kv: -kv[1])[:5]
        top5_str = "  ".join(f"{k}={mv:.0f}" for k, mv in top5)
        over_tau = sum(1 for mv in site_maxes.values() if mv > tau)
        over256  = sum(1 for mv in site_maxes.values() if mv > 256.0)
        over127  = sum(1 for mv in site_maxes.values() if mv > 127.0)
        gi_max   = getattr(model, "_mon_gi_max",  float("nan"))
        fv_max   = getattr(model, "_mon_fv_max",  float("nan"))
        tv_max   = getattr(model, "_mon_tv_max",  float("nan"))
        bmm_max  = getattr(model, "_mon_bmm_max", float("nan"))
        elapsed  = time.time() - t0

        print(flush=True)
        print(f"  ep {ep}  ({elapsed:.1f}s)", flush=True)
        print(f"  chess   loss={tot:.4f}  p={p:.4f}  v={v:.4f}", flush=True)
        print(f"  harden  pen={pen:.1f}  lam={lam:.2e}  tau={tau:.0f}"
              f"  peak={peak:.0f}  over_tau={over_tau}  over256={over256}"
              f"  over127={over127}/{len(site_names)}", flush=True)
        gd_max = getattr(model, "_mon_gd_max", float("nan"))
        print(f"  gate    gi={gi_max:.0f}  gg={cap['gg_max']:.0f}"
              f"  gu={cap['gu_max']:.0f}  h={cap['h_max']:.0f}  gd={gd_max:.0f}", flush=True)
        print(f"  bmm     fv={fv_max:.0f}  tv={tv_max:.0f}  out={bmm_max:.0f}", flush=True)
        print(f"  top5    {top5_str}", flush=True)

        if ep % PLOT_EVERY == 0 or ep == end_ep - 1:
            vb = val_pool.get()
            eval_df, _, _ = B.do_eval(
                model, NAME + "_harden", ep, vb, eval_df, progress_file,
                plot_file, device, train_loss=tot, gn_mean=None)

        if ep % 100 == 0 or ep == end_ep - 1:
            out = os.path.join(RUN_DIR, f"{NAME}_pt_ckpt{ep:05d}.pt")
            B.save_pt_ckpt(model, opt, scaler, ep, NAME, out)
            print(f"  [ckpt]  {out}", flush=True)

    train_pool.stop()
    val_pool.stop()
    print("\n[harden] done", flush=True)


if __name__ == "__main__":
    main()
