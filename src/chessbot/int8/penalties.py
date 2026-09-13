"""Training pressures that make the int8 swap lossless.

  integer_penalty       sin^2(pi * Q)         Q onto the integer grid
  weight_range_penalty  relu(|Q| - 127)       Q inside int8
  ActivationRange       relu(|a| - 127)       every layer output inside int8
"""
import math

import torch

from chessbot.int8.layers import IntLinear, L1Norm, QMAX


def int_linears(model):
    return [m for m in model.modules() if isinstance(m, IntLinear)]


def integer_penalty(model):
    pen = 0.0
    for m in int_linears(model):
        pen = pen + torch.sin(math.pi * m.Q).pow(2).sum()
    return pen


def weight_range_penalty(model, qmax=QMAX):
    pen = 0.0
    for m in int_linears(model):
        pen = pen + torch.relu(m.Q.abs() - qmax).sum()
    return pen


def integer_stats(model, tol=0.05):
    gaps, max_q = [], 0.0
    for m in int_linears(model):
        q = m.Q.detach()
        gaps.append((q - q.round()).abs().flatten())
        max_q = max(max_q, float(q.abs().max()))
    g = torch.cat(gaps)
    return {"on_grid": float((g < tol).float().mean()),
            "mean_gap": float(g.mean()),
            "max_q": max_q}


class ActivationRange:
    """Hooks every IntLinear / L1Norm output. Call reset() before each forward;
    afterwards .penalty is the summed range violation and .peaks maps
    site name -> max|a| seen since the last reset."""

    def __init__(self, model, tau=QMAX):
        self.tau = tau
        self.reset()
        for name, m in model.named_modules():
            if isinstance(m, (IntLinear, L1Norm)):
                m.register_forward_hook(self.make_hook(name))

    def reset(self):
        self.penalty = 0.0
        self.peaks = {}

    def make_hook(self, name):
        def hook(mod, inp, out):
            self.penalty = self.penalty + torch.relu(out.abs() - self.tau).sum()
            v = float(out.detach().abs().max())
            if v > self.peaks.get(name, 0.0):
                self.peaks[name] = v
        return hook

    def peak(self):
        return max(self.peaks.values()) if self.peaks else 0.0
