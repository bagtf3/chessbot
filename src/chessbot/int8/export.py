"""Freeze a trained IntLinear / L1Norm model into its int8 twin.

The int8 modules here are reference implementations of what the TRT kernel
does: int8 x int8 -> int32 accumulate, fp32 requant multiply, bias, clamp,
round, write int8. The int32 accumulator is emulated in float64, which is
exact for the magnitudes involved (< 2**53).
"""
import copy

import torch
import torch.nn as nn

from chessbot.int8.layers import IntLinear, L1Norm, EPS, QMAX


def to_int8(y):
    return y.round().clamp(-QMAX, QMAX).to(torch.int8)


class Int8Linear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        with torch.no_grad():
            self.register_buffer("Q", to_int8(lin.Q))
            self.register_buffer("s", lin.scale().float().clone())
            self.register_buffer("b", lin.b.float().clone())
        self.act = lin.act

    def forward(self, x):
        acc = x.double() @ self.Q.double().t()
        y = acc.float() * self.s + self.b
        if self.act:
            y = y.clamp_min(0.0)
        return to_int8(y)


class Int8L1Norm(nn.Module):
    def __init__(self, ln):
        super().__init__()
        with torch.no_grad():
            self.register_buffer("s", ln.scale().float().clone())
            self.register_buffer("b", ln.b.float().clone())

    def forward(self, x):
        xf = x.float()
        m = xf.abs().mean(-1, keepdim=True) + EPS
        return to_int8((xf / m) * self.s + self.b)


def swap_children(mod):
    for name, child in mod.named_children():
        if isinstance(child, IntLinear):
            setattr(mod, name, Int8Linear(child))
        elif isinstance(child, L1Norm):
            setattr(mod, name, Int8L1Norm(child))
        else:
            swap_children(child)


def freeze_int8(model):
    """Deep copy with every IntLinear / L1Norm replaced by its int8 twin.
    Input to the frozen model must already be int8 on the scale-1 grid."""
    frozen = copy.deepcopy(model).eval()
    swap_children(frozen)
    return frozen
