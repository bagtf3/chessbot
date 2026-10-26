"""Int8-native training layers. See docs/int8_native_model_plan.md.

Conventions
  scale-1 activations : the float value IS the int8 value, just not rounded yet.
                        Range bangers hold every activation inside [-127, 127].
  W = Q * s           : Q is the weight parameter, trained onto the integer grid
                        inside [-127, 127]. s = EPS + sprime**2 is the per-channel
                        requant scale (always > 0). W is derived, never stored.
  b                   : plain learnable bias, kept separate from s.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6
QMAX = 127.0
Q_INIT_MAX = 64.0
LN_SCALE_INIT = 20.0
LEAKY_SLOPE = 0.01


def scale_from(sprime):
    return EPS + sprime ** 2


def sprime_for(s):
    return math.sqrt(max(s - EPS, 0.0))


class IntLinear(nn.Module):
    """y = x @ (Q * s).T + b, optionally followed by leaky relu.

    At export: Q -> round(Q) as int8, s -> fp32 requant multiplier on the
    int32 accumulator, leaky relu -> hard clamp at 0.
    """

    def __init__(self, din, dout, act=False):
        super().__init__()
        self.Q = nn.Parameter(torch.empty(dout, din))
        self.sprime = nn.Parameter(torch.empty(dout))
        self.b = nn.Parameter(torch.zeros(dout))
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        w = torch.empty_like(self.Q)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        c = Q_INIT_MAX / w.abs().amax(dim=1, keepdim=True)
        with torch.no_grad():
            self.Q.copy_(w * c)
            self.sprime.copy_((1.0 / c.squeeze(1) - EPS).clamp_min(0.0).sqrt())

    def scale(self):
        return scale_from(self.sprime)

    def weight(self):
        return self.Q * self.scale().unsqueeze(1)

    def forward(self, x):
        y = F.linear(x, self.weight().to(x.dtype), self.b.to(x.dtype))
        if self.act:
            y = F.leaky_relu(y, LEAKY_SLOPE)
        return y


class L1Norm(nn.Module):
    """y = x / mean|x| * s + b.  No pow(2), no sqrt.

    s = EPS + sprime**2 is the output scale that lands y back on the int8 grid.
    The mean runs in fp32; everything else is integer at export.
    """

    def __init__(self, d):
        super().__init__()
        self.sprime = nn.Parameter(torch.full((d,), sprime_for(LN_SCALE_INIT)))
        self.b = nn.Parameter(torch.zeros(d))

    def scale(self):
        return scale_from(self.sprime)

    def forward(self, x):
        xf = x.float()
        m = xf.abs().mean(-1, keepdim=True) + EPS
        y = (xf / m) * self.scale() + self.b
        return y.to(x.dtype)
