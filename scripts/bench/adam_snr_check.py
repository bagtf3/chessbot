"""
adam_snr_check.py -- Adam optimizer state SNR/effective-step probe.

Reads only the saved Adam state (exp_avg, exp_avg_sq, step) -- never touches
inference weights. Per param group, reports:
  snr        = |m_hat| / (sqrt(v_hat) + eps)   -- consistent-direction signal
               vs step-to-step noise. Adam's own bias-corrected estimate of
               "is the gradient pointing somewhere real, or just jittering."
  eff_step/w = |lr * m_hat / (sqrt(v_hat)+eps)| / |w|  -- actual update size
               this step, relative to the weight's own scale.

Usage:
  python scripts/bench/adam_snr_check.py --model PATH --opt PATH [--arch NAME]
"""
import argparse
import re
from collections import defaultdict

import numpy as np
import torch


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
    parser.add_argument("--model", required=True)
    parser.add_argument("--opt", required=True)
    parser.add_argument("--arch", default=None,
                         help="override arch name (default: read from --model ckpt)")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    arch = args.arch or (ckpt.get("arch", "?") if isinstance(ckpt, dict) else "?")

    from chessbot.model import VARIANTS, PT_BUILDERS
    cfg = VARIANTS[arch]
    model = PT_BUILDERS[arch](cfg)
    model.load_state_dict(sd)
    param_names = [n for n, _ in model.named_parameters()]

    opt = torch.load(args.opt, map_location="cpu")
    state = opt["state"]
    pg0 = opt["param_groups"][0]
    b1, b2 = pg0["betas"]
    lr, eps = pg0["lr"], pg0["eps"]

    steps = sorted({
        int(s["step"].item()) if isinstance(s["step"], torch.Tensor) else s["step"]
        for s in state.values() if "step" in s
    })
    print(f"arch={arch}  lr={lr}  betas=({b1},{b2})  eps={eps}")
    print(f"optimizer steps: min={steps[0]}  max={steps[-1]}"
          + ("  (uniform)" if len(steps) == 1 else "  (mixed -- some params updated less)"))
    print()

    groups = defaultdict(lambda: {
        "snr_sum": 0.0, "eff_ratio_sum": 0.0, "n": 0,
    })

    named = dict(zip(param_names, model.parameters()))

    for i, name in enumerate(param_names):
        if i not in state:
            continue
        s = state[i]
        if "exp_avg" not in s or "exp_avg_sq" not in s:
            continue

        step = s["step"]
        step = int(step.item()) if isinstance(step, torch.Tensor) else step
        if step == 0:
            continue

        m = s["exp_avg"].float().numpy()
        v = s["exp_avg_sq"].float().numpy()
        m_hat = m / (1.0 - b1 ** step)
        v_hat = v / (1.0 - b2 ** step)
        sqrt_v = np.sqrt(v_hat)

        snr = np.abs(m_hat) / (sqrt_v + eps)
        update = lr * m_hat / (sqrt_v + eps)

        w = named[name].detach().float().numpy()
        w_abs = np.abs(w)
        eff_ratio = np.abs(update) / (w_abs + 1e-8)

        g = groups[param_group(name)]
        g["snr_sum"] += float(np.sum(snr))
        g["eff_ratio_sum"] += float(np.sum(eff_ratio))
        g["n"] += snr.size

    print(f"{'group':<14} {'snr_mean':>10} {'eff_step/w':>12} {'n_params':>10}")
    print("-" * 50)
    total_snr = total_eff = 0.0
    total_n = 0
    for gname in sorted(groups):
        g = groups[gname]
        n = g["n"]
        if n == 0:
            continue
        snr_mean = g["snr_sum"] / n
        eff_mean = g["eff_ratio_sum"] / n
        print(f"{gname:<14} {snr_mean:>10.4f} {eff_mean:>12.6f} {n:>10,}")
        total_snr += g["snr_sum"]
        total_eff += g["eff_ratio_sum"]
        total_n += n

    print("-" * 50)
    print(f"{'OVERALL':<14} {total_snr/total_n:>10.4f} {total_eff/total_n:>12.6f} {total_n:>10,}")
    print()
    print("snr: |m_hat| / (sqrt(v_hat)+eps) -- ~0 means gradient direction is")
    print("     flipping step to step (noise-dominated); higher = consistent signal")
    print("eff_step/w: |actual update this step| / |weight| -- how big a step Adam")
    print("     is currently taking relative to the parameter's own scale")


if __name__ == "__main__":
    main()
