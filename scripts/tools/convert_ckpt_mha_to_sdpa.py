#!/usr/bin/env python3
"""convert_ckpt_mha_to_sdpa.py

Rewrite an addpos-family checkpoint (model + AdamW optimizer state) from the
nn.MultiheadAttention layout to the F.scaled_dot_product_attention layout, in
place, preserving the exact bundle structure and filename. Lossless: the packed
in_proj (weight+bias) is chunk-split into q/k/v, out_proj becomes o, and the
matching Adam moments are split the same way, so training resumes with identical
weights AND optimizer state.

Verifies mha-vs-sdpa output equivalence before writing; aborts if they diverge.

  python scripts/tools/convert_ckpt_mha_to_sdpa.py <ckpt.pt> --variant precond-mha-10c6t-d256-wdl3
  add --dry-run to build/remap/verify without writing.
"""

import os
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn

from chessbot.model import VARIANTS, PT_BUILDERS

PT_ADAM_BETA2   = 0.95
PT_WEIGHT_DECAY = 0.008
TOL             = 1e-3   # max abs output diff allowed (fp32 reduction-order)


def make_adam(model, lr):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay,    "weight_decay": PT_WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, PT_ADAM_BETA2))


def attn_prefixes(model):
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.MultiheadAttention)]


def remap_model_sd(old_sd, prefixes):
    """Packed in_proj -> q/k/v, out_proj -> o; copy everything else."""
    new_sd, consumed = {}, set()
    for p in prefixes:
        E = old_sd[f"{p}.in_proj_weight"].shape[1]
        W = old_sd[f"{p}.in_proj_weight"]
        b = old_sd[f"{p}.in_proj_bias"]
        new_sd[f"{p}.q.weight"] = W[:E].clone()
        new_sd[f"{p}.k.weight"] = W[E:2 * E].clone()
        new_sd[f"{p}.v.weight"] = W[2 * E:3 * E].clone()
        new_sd[f"{p}.q.bias"]   = b[:E].clone()
        new_sd[f"{p}.k.bias"]   = b[E:2 * E].clone()
        new_sd[f"{p}.v.bias"]   = b[2 * E:3 * E].clone()
        new_sd[f"{p}.o.weight"] = old_sd[f"{p}.out_proj.weight"].clone()
        new_sd[f"{p}.o.bias"]   = old_sd[f"{p}.out_proj.bias"].clone()
        for s in ("in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"):
            consumed.add(f"{p}.{s}")
    for k, v in old_sd.items():
        if k not in consumed:
            new_sd[k] = v
    return new_sd


def slice_moment(st, sl):
    out = {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))
           for k, v in st.items()}
    out["exp_avg"]    = st["exp_avg"][sl].clone()
    out["exp_avg_sq"] = st["exp_avg_sq"][sl].clone()
    return out


def clone_moment(st):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in st.items()}


def remap_optimizer(old_model, old_opt, new_model, new_opt, prefixes):
    old_named = dict(old_model.named_parameters())
    new_named = dict(new_model.named_parameters())
    name_state = {n: old_opt.state[p] for n, p in old_named.items()
                  if p in old_opt.state}

    consumed = set()
    for p in prefixes:
        E = old_named[f"{p}.in_proj_weight"].shape[1]
        iw, ib = name_state[f"{p}.in_proj_weight"], name_state[f"{p}.in_proj_bias"]
        ow, ob = name_state[f"{p}.out_proj.weight"], name_state[f"{p}.out_proj.bias"]
        new_opt.state[new_named[f"{p}.q.weight"]] = slice_moment(iw, slice(0, E))
        new_opt.state[new_named[f"{p}.k.weight"]] = slice_moment(iw, slice(E, 2 * E))
        new_opt.state[new_named[f"{p}.v.weight"]] = slice_moment(iw, slice(2 * E, 3 * E))
        new_opt.state[new_named[f"{p}.q.bias"]]   = slice_moment(ib, slice(0, E))
        new_opt.state[new_named[f"{p}.k.bias"]]   = slice_moment(ib, slice(E, 2 * E))
        new_opt.state[new_named[f"{p}.v.bias"]]   = slice_moment(ib, slice(2 * E, 3 * E))
        new_opt.state[new_named[f"{p}.o.weight"]] = clone_moment(ow)
        new_opt.state[new_named[f"{p}.o.bias"]]   = clone_moment(ob)
        for s in ("in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"):
            consumed.add(f"{p}.{s}")
    for n, p in new_named.items():
        if n in consumed or n not in name_state:
            continue
        new_opt.state[p] = clone_moment(name_state[n])


def random_xc0h(batch, K=6):
    n = K * 64 + K + 3
    out = np.zeros((batch, n), dtype=np.int64)
    out[:, :K * 64]           = np.random.randint(0, 15, (batch, K * 64))
    out[:, K * 64:K * 64 + K] = np.random.randint(0, 2, (batch, K))
    out[:, K * 64 + K]        = np.random.randint(0, 16, batch)
    out[:, K * 64 + K + 1]    = np.random.randint(0, 2, batch)
    out[:, K * 64 + K + 2]    = np.random.randint(0, 100, batch)
    return torch.from_numpy(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[load] {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    base_cfg = dict(VARIANTS[args.variant])
    builder  = PT_BUILDERS[args.variant]

    old_cfg = dict(base_cfg); old_cfg["attn_impl"] = "mha"
    new_cfg = dict(base_cfg); new_cfg["attn_impl"] = "sdpa"

    old_model = builder(old_cfg).eval()
    new_model = builder(new_cfg).eval()
    old_model.load_state_dict(ckpt["model"])
    prefixes = attn_prefixes(old_model)
    print(f"[attn] {len(prefixes)} attention modules: {prefixes}", flush=True)

    new_sd = remap_model_sd(ckpt["model"], prefixes)
    new_model.load_state_dict(new_sd)   # strict: verifies key/shape coverage
    print("[model] remapped state loaded strict-ok", flush=True)

    # numerical equivalence, fp32 CPU
    with torch.no_grad():
        x = random_xc0h(4)
        po, vo = old_model(x)
        pn, vn = new_model(x)
    dp = (po - pn).abs().max().item()
    dv = (vo - vn).abs().max().item()
    print(f"[verify] max|dpolicy|={dp:.3e}  max|dvalue|={dv:.3e}  (tol {TOL})", flush=True)
    if dp > TOL or dv > TOL:
        print("[ABORT] outputs diverge beyond tolerance; NOT writing.", flush=True)
        return

    lr0    = ckpt["optimizer"]["param_groups"][0].get("lr", 1e-4)
    old_opt = make_adam(old_model, lr0)
    new_opt = make_adam(new_model, lr0)
    old_opt.load_state_dict(ckpt["optimizer"])
    remap_optimizer(old_model, old_opt, new_model, new_opt, prefixes)
    new_opt_sd = new_opt.state_dict()

    # round-trip: a fresh optimizer on the sdpa model must accept it
    make_adam(new_model, lr0).load_state_dict(new_opt_sd)
    print("[opt] remapped optimizer state round-trips clean", flush=True)

    out = {
        "model": new_model.state_dict(),
        "optimizer": new_opt_sd,
        "scaler": ckpt.get("scaler"),
        "epoch": ckpt.get("epoch"),
        "arch": ckpt.get("arch"),
        "lw_state": ckpt.get("lw_state"),
    }
    if args.dry_run:
        print("[dry-run] verified; not writing.", flush=True)
        return

    tmp = args.ckpt + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, args.ckpt)
    print(f"[write] in-place -> {args.ckpt}  (epoch={out['epoch']}, arch={out['arch']})",
          flush=True)


if __name__ == "__main__":
    main()
