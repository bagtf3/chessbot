"""Side experiment: tame smart-gate activations on ckpt28000 without a full
re-pretrain.

Loads a COPY of ckpt28000, resumes pretrain data at a flat LR floor, and adds a
reverse-ReLU (hinge) penalty on the gate FFN activations to drive their
magnitude down below a fp16-safe ceiling. AdamW decay is raised to 0.01 on the
2D+ weight group (the pretrain ndim split is preserved) to lean on the upstream
weights. Loss-weight shape and LR schedule shape follow the pretrain script.
Runs 1000 epochs and logs gate magnitude + penalty so lambda can be
recalibrated.

    python scripts/train/gatereg_finetune_exp.py [--epochs 1000] [--tau 64]
        [--lam0 1e-5] [--lam1 1e-3] [--wd 0.01]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Bryan\repos\chessbot\.env")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap_model_async_sparse_pkl as B
from chessbot.pretrain import EPOCH_SIZE, PLOT_EVERY, lr_for_epoch, split_train_val

NAME       = "precond-mha-10c6t-d256-wdl3"
RUN_DIR    = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\20m_wdl3_gatereg_exp"
START_CKPT = os.path.join(RUN_DIR, f"{NAME}_pt_ckpt28100.pt")
START_EP   = 28100
SHARD_DIR  = B.DEFAULT_SHARD_DIR
LR_FLAT    = B.LR_MIN
GATE_MODS  = ("gate_w_gate", "gate_w_up", "gate_w_down")


def gate_penalty(acts, tau):
    """Per-activation reverse-ReLU hinge: each element over tau is penalized by
    its own excess, the rest contribute nothing. Summed so every violator feels
    the same push (lam), undiluted by the elements that are already in range."""
    import torch
    pen = acts[0].new_zeros((), dtype=torch.float32)
    peak = 0.0
    for a in acts:
        af = a.float()
        pen = pen + torch.relu(af.abs() - tau).sum()
        peak = max(peak, float(af.abs().max()))
    return pen, peak


def fit_epoch(model, opt, scaler, bundle, lw, max_norm, device, lam, tau, cap):
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    enc      = torch.as_tensor(bundle["enc_in"], dtype=torch.long).to(device)
    policy_t = torch.as_tensor(bundle["policy_logits"], dtype=torch.float32).to(device)
    value_t  = torch.as_tensor(bundle["value_out"], dtype=torch.float32).to(device)
    w        = torch.as_tensor(bundle["weight"], dtype=torch.float32).to(device)

    n    = enc.shape[0]
    perm = torch.randperm(n, device=device)
    model.train()
    tp = tv = tt = tpen = 0.0
    peak_epoch = 0.0
    n_batches = 0

    for start in range(0, n, B.PT_BATCH_SIZE):
        idx = perm[start:start + B.PT_BATCH_SIZE]
        opt.zero_grad(set_to_none=True)
        cap["acts"] = []
        with autocast("cuda"):
            pol, val = model(enc[idx])
            p_loss = F.cross_entropy(pol, policy_t[idx]) * lw["policy_logits"]
            v_loss = (F.cross_entropy(val, value_t[idx], reduction="none")
                      * w[idx]).mean() * lw["value_out"]
            pen, peak = gate_penalty(cap["acts"], tau)
            loss = p_loss + v_loss + lam * pen
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(opt)
        scaler.update()
        tp += p_loss.item(); tv += v_loss.item(); tt += loss.item()
        tpen += float(pen); peak_epoch = max(peak_epoch, peak)
        n_batches += 1

    return (tp / n_batches, tv / n_batches, tt / n_batches,
            tpen / n_batches, peak_epoch)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=64.0)
    ap.add_argument("--lam0", type=float, default=1e-7)
    ap.add_argument("--lam1", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=0.01)
    args = ap.parse_args()

    end_ep = START_EP + args.epochs
    device = torch.device("cuda")
    print(f"[exp] loading {START_CKPT}", flush=True)
    model, opt, scaler, _ = B.load_pt_model(START_CKPT, NAME, device, lr=LR_FLAT)

    # bump decay on the 2D+ group only (ndim split preserved), flat LR floor
    for pg in opt.param_groups:
        if pg["weight_decay"] > 0:
            pg["weight_decay"] = args.wd
        pg["lr"] = LR_FLAT
    print(f"[exp] AdamW wd={args.wd} (2D+ group)  lr={LR_FLAT:.2e} (flat)  "
          f"tau={args.tau}  lam {args.lam0:.1e}->{args.lam1:.1e}  "
          f"epochs {START_EP}..{end_ep - 1}", flush=True)

    cap = {"acts": []}
    def out_hook(_m, _i, o):
        cap["acts"].append(o)
    def in_hook(_m, i):
        cap["acts"].append(i[0])
    for gn in GATE_MODS:
        getattr(model, gn).register_forward_hook(out_hook)
    # exact overflow tensor: gate_norm's input g (post-residual, post-dropout)
    model.gate_norm.register_forward_pre_hook(in_hook)

    all_files = sorted(glob.glob(os.path.join(SHARD_DIR, "*.pkl.gz")))
    train_files, val_files = split_train_val(all_files)
    print(f"[exp] shards train={len(train_files)} val={len(val_files)}", flush=True)

    train_pool = B.AsyncShardPool(
        train_files, high=B.BUFFER_CAP, low=B.BUFFER_CAP - B.REFILL_BAND,
        epoch_size=EPOCH_SIZE, drain=True, n_ready=2, name="train")
    val_pool = B.AsyncShardPool(
        val_files, high=B.VAL_BUFFER_CAP, low=B.VAL_BUFFER_CAP - B.REFILL_BAND,
        epoch_size=EPOCH_SIZE, drain=True, n_ready=1, name="val")
    train_pool.start(); val_pool.start()

    progress_file = os.path.join(RUN_DIR, "gatereg_eval_progress.csv")
    plot_file = os.path.join(RUN_DIR, f"{NAME}_gatereg_plot.png")
    eval_df = None
    policy_ce_hat = value_ce_hat = None
    active_lw = {"policy_lw": B.LW_WARMUP_POLICY, "value_lw": B.LW_WARMUP_VALUE,
                 "scale": None}

    for ep in range(START_EP, end_ep):
        frac = (ep - START_EP) / max(1, args.epochs - 1)
        lam = args.lam0 + (args.lam1 - args.lam0) * frac

        if policy_ce_hat is None:   # no lw_state in ckpt; seed from warmup LW
            policy_lw, value_lw, scale = B.LW_WARMUP_POLICY, B.LW_WARMUP_VALUE, None
        else:
            policy_lw, value_lw, scale, _ = B.lw_for_epoch(
                ep, policy_ce_hat, value_ce_hat, active_lw)
        active_lw = {"policy_lw": policy_lw, "value_lw": value_lw, "scale": scale}
        lw = {"policy_logits": policy_lw, "value_out": value_lw}

        t0 = time.time()
        bundle = train_pool.get()
        p, v, tot, pen, peak = fit_epoch(
            model, opt, scaler, bundle, lw, max_norm=10.0, device=device,
            lam=lam, tau=args.tau, cap=cap)
        policy_ce_hat = B.lw_ema_update(policy_ce_hat, p / policy_lw)
        value_ce_hat  = B.lw_ema_update(value_ce_hat, v / value_lw)

        print(f"[ep {ep:5d}] loss={tot:.4f} p={p:.4f} v={v:.4f} "
              f"pen={pen:.2f} lam={lam:.2e} gate_peak={peak:8.1f} "
              f"({time.time() - t0:.1f}s)", flush=True)

        if ep % PLOT_EVERY == 0 or ep == end_ep - 1:
            vb = val_pool.get()
            eval_df, _, _ = B.do_eval(
                model, NAME + "_gatereg", ep, vb, eval_df, progress_file,
                plot_file, device, train_loss=tot, gn_mean=None)

        if ep % 100 == 0 or ep == end_ep - 1:
            out = os.path.join(RUN_DIR, f"{NAME}_pt_ckpt{ep:04d}.pt")
            B.save_pt_ckpt(model, opt, scaler, ep, NAME, out)

    train_pool.stop(); val_pool.stop()
    print("[exp] done", flush=True)


if __name__ == "__main__":
    main()
