"""
analyze_pt_weights.py — Adam SNR probe for current model checkpoint.

Usage:
  python scripts/bench/analyze_pt_weights.py --model PATH --opt PATH
"""
import argparse
import re
import numpy as np
import torch
from collections import defaultdict


RUN_DIR  = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\16m_xc0hK6_run3"
DEFAULT_MODEL = RUN_DIR + r"\16m_xc0hK6_run3_model.pt"
DEFAULT_OPT   = RUN_DIR + r"\train_ckpts\16m_xc0hK6_run3_model_opt_state.pt"


def param_group(name):
    if name.startswith("emb.") or name.startswith("pos."):
        return "embed"
    if name.startswith("xc0h_") or name.startswith("xc0h"):
        return "xc0h_stem"
    if re.match(r"pre\.\d+\.", name):
        idx = int(re.search(r"pre\.(\d+)\.", name).group(1))
        return f"pre{idx:02d}"
    if re.match(r"blocks?\.\d+\.", name):
        idx = int(re.search(r"blocks?\.(\d+)\.", name).group(1))
        kind = "attn" if "attn" in name else "ff"
        return f"tx{idx:02d}_{kind}"
    if "accum" in name or name.startswith("global_tokens"):
        return "accum"
    if "gate" in name:
        return "gate"
    if name.startswith("wdl_"):
        return "head_value"
    if name.startswith("from_") or name.startswith("to_") or name.startswith("promo_"):
        return "head_policy"
    if "val_" in name or "value" in name:
        return "head_value"
    if "pol_" in name or "policy" in name:
        return "head_policy"
    if "mix" in name:
        return "mix"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--opt",   default=DEFAULT_OPT)
    args = parser.parse_args()

    print(f"model: {args.model}")
    print(f"opt:   {args.opt}")
    print()

    ckpt = torch.load(args.model, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    arch = ckpt.get("arch", "?") if isinstance(ckpt, dict) else "?"
    param_names = list(sd.keys())
    print(f"arch={arch}  params={len(param_names)}")
    print()

    opt = torch.load(args.opt, map_location="cpu")
    state = opt["state"]
    pg0 = opt["param_groups"][0]
    b1, b2 = pg0["betas"]
    lr  = pg0["lr"]
    eps = pg0["eps"]
    print(f"lr={lr}  betas=({b1},{b2})  eps={eps}")
    print()

    # accumulate adaptive denominator stats per group
    groups = defaultdict(lambda: {"sqrt_v_sum": 0.0, "n": 0})

    for i, name in enumerate(param_names):
        if i not in state:
            continue
        s = state[i]
        if "exp_avg_sq" not in s:
            continue

        step = s.get("step", 1)
        if isinstance(step, torch.Tensor):
            step = int(step.item())

        v = s["exp_avg_sq"].float().numpy()
        v_hat = v / (1.0 - b2 ** step)
        sqrt_v = np.sqrt(v_hat)

        g = groups[param_group(name)]
        g["sqrt_v_sum"] += float(np.sum(sqrt_v))
        g["n"]          += sqrt_v.size

    print(f"{'group':<18} {'mean_sqrt(v)':>14} {'eff_lr (lr/sqrtv)':>18} {'n_params':>10}")
    print("-" * 64)
    total_sv = 0.0
    total_n  = 0
    for gname in sorted(groups):
        g = groups[gname]
        n = g["n"]
        if n == 0:
            continue
        sv   = g["sqrt_v_sum"] / n
        elr  = lr / (sv + eps)
        print(f"{gname:<18} {sv:>14.6f} {elr:>18.6f} {n:>10,}")
        total_sv += g["sqrt_v_sum"]
        total_n  += n

    print("-" * 64)
    overall = total_sv / total_n if total_n else 0.0
    print(f"{'OVERALL':<18} {overall:>14.6f} {lr/(overall+eps):>18.6f}")
    print()
    print("mean_sqrt(v): adaptive denominator — higher = Adam squashes updates more")
    print("eff_lr: lr / mean_sqrt(v) — actual per-param learning rate after squashing")


if __name__ == "__main__":
    main()
