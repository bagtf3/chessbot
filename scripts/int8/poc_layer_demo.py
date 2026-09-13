"""Int8-native POC on a toy problem.

Trains IntLinear -> L1Norm -> IntLinear to mimic a fixed teacher, following the
plan's phases: task loss + range bangers from step 0, sin^2 blended in, leaky
relu annealed to hard relu, then freeze to int8 and compare forwards.

    python scripts/int8/poc_layer_demo.py [--cpu] [--steps 4000]
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from chessbot.int8 import (
    IntLinear, L1Norm, ActivationRange, integer_penalty, weight_range_penalty,
    integer_stats, freeze_int8,
)
from chessbot.int8 import layers

DIN, DH, DOUT = 64, 128, 32
BATCH = 512
LR = 3e-3
LAM_RANGE = 1e-4
MU_MAX = 1e-3


def ramp(step, start, length):
    return min(1.0, max(0.0, (step - start) / length))


def make_teacher(dev):
    teacher = nn.Sequential(
        nn.Linear(DIN, DH), nn.Tanh(), nn.Linear(DH, DOUT)).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    def batch():
        x = torch.randint(-127, 128, (BATCH, DIN), device=dev).float()
        return x, teacher(x / 127.0) * 100.0
    return batch


def train_toy(dev, steps, log=True):
    """Returns (trained float model, batch fn). Phase boundaries scale with steps."""
    torch.manual_seed(0)
    batch = make_teacher(dev)
    model = nn.Sequential(
        IntLinear(DIN, DH, act=True), L1Norm(DH), IntLinear(DH, DOUT)).to(dev)
    acts = ActivationRange(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mu_start, mu_ramp = steps // 4, steps * 3 // 8
    slope_start, slope_ramp = steps * 5 // 8, steps // 4

    if log:
        print(f"device={dev}  steps={steps}", flush=True)
        print("step   mse     mu       slope  Q_on_grid  Q_gap   Q_max  act_peak", flush=True)
    for step in range(steps):
        mu = MU_MAX * ramp(step, mu_start, mu_ramp)
        layers.LEAKY_SLOPE = 0.01 * (1.0 - ramp(step, slope_start, slope_ramp))

        x, y = batch()
        acts.reset()
        mse = F.mse_loss(model(x), y)
        pen_range = acts.penalty + weight_range_penalty(model)
        loss = mse / 100.0 + LAM_RANGE * pen_range + mu * integer_penalty(model)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if log and (step % 250 == 0 or step == steps - 1):
            st = integer_stats(model)
            print(f"{step:5d}  {mse.item():6.2f}  {mu:.1e}  {layers.LEAKY_SLOPE:.3f}"
                  f"  {st['on_grid']:9.3f}  {st['mean_gap']:.3f}  {st['max_q']:5.0f}"
                  f"  {acts.peak():8.1f}", flush=True)

    return model.eval(), batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    model, batch = train_toy(dev, args.steps)
    frozen = freeze_int8(model)
    x, y = batch()
    with torch.no_grad():
        pf = model(x)
        pq = frozen(x.to(torch.int8)).float()
    d = (pf - pq).abs()

    print(flush=True)
    for name, m in frozen.named_modules():
        if hasattr(m, "Q"):
            print(f"{name}: Q int8 {tuple(m.Q.shape)}  s [{m.s.min():.2e}, {m.s.max():.2e}]",
                  flush=True)
    print(flush=True)
    print(f"float vs int8   MAE={d.mean():.3f}  max={d.max():.1f}"
          f"  within_1LSB={(d <= 1.0).float().mean():.3f}", flush=True)
    print(f"mse vs teacher  float={F.mse_loss(pf, y):.2f}  int8={F.mse_loss(pq, y):.2f}",
          flush=True)


if __name__ == "__main__":
    main()
