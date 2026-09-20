#!/usr/bin/env python3
"""bakeoff_speedtest_10c6t.py

Speed test for the bake-off block variants scaled to a 10c6t trunk, random
weights.  PT eager (fp16) vs ORT+TensorRT (fp16) across batch sizes, reusing the
harness in bakeoff_speedtest.py.

    m1     4 plain + 6 plain conv     6x MHA + FF 4x
    m7     4 plain + 6 plain conv     6x MHA + SwiGLU 4x
    m8     4 plain + 6 gated conv     6x MHA + FF 4x
    m11    10 expandy-fast conv       6x MHA + FF 4x
    m11f   10 expandy-fatty conv      6x MHA + FF 4x   (2D through the middle)
    m12    4 plain + 6 dcsoftmax conv 6x MHA + FF 4x   (no mid norm)
    m13    4 plain + 6 gated-expandy  6x MHA + FF 4x
    m14    4 plain + 6 SE-expandy     6x MHA + FF 4x
    m11-6  4 plain + 6 expandy conv   6x MHA + DC768
    m11-7  4 plain + 6 expandy conv   6x MHA + SwiGLU 4x
    m13-7  4 plain + 6 gated-expandy  3x MHA + FF 4x, then 3x MHA + SwiGLU 4x
    m13-1-7-5c10t  5 gated-expandy    5x MHA + FF 4x, then 5x MHA + SwiGLU 4x
    m11-1-6c12t-d256  6 expandy-fast  12x MHA + FF 4x, d256
    m13-1-7-6c12t-d256  6 gated-expandy  8x MHA + FF 4x, then 4x MHA + SwiGLU 4x
    m69    4 plain + 4 expandy + 2 gated conv   3x MHA + FF 4x, then 3x MHA + SwiGLU 4x
    m13-1-8c4t-d256-512  8 gated-expandy d256 convs, concatenate d256 pos -> d512,
                         then 4x MHA + FF 4x; value d512, policy d256

Gate/DC blocks carry no EMA tracking buffers.  Results also go to a CSV in
--out-dir.
"""
import os
import sys
import csv
import gc
import glob
import hashlib
import time
import argparse
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from bakeoff_speedtest import (
    TRT_CACHE, DEFAULT_OUT_DIR, verify_outputs, make_pt_eager_infer,
    export_to_onnx, make_trt_infer, compare_pt_vs_export, speed_test,
    print_summary_table,
)
from bakeoff_4c2t_d256 import (
    BakeoffModel, L1Norm, Attention, ConvBlock, TxBlockFF, TxBlockFFNoOut,
    D, HEADS, EPS,
)
import bakeoff_4c2t_d256 as b4

N_PLAIN = 4
N_VARIANT = 6
N_TX = 6
SWIGLU_H = 4 * D
DC_H = 1024

MODELS_TO_TEST = ["m69", "m15", "m16"]
DEFAULT_BATCH_SIZES = [32, 64, 128, 256]

MASTER_CSV = "speedtest_10c6t_master.csv"
CSV_FIELDS = ["model", "params", "backend", "batch", "latency_ms",
              "samples_per_s", "run"]
SYNC_TIMEOUT = 360
YES_WORDS = {"y", "yes", "ye", "yeah", "yep", "yup", "sure", "ok", "okay", "do it"}
# param counts for run CSVs written before the params column existed
PARAMS_FALLBACK = {
    "m1": 19980155, "m7": 21559163, "m8": 20374907, "m11": 16716155,
    "m11-6": 19588481, "m11-7": 19600763,
}


class RMSNormExport(nn.Module):
    """nn.RMSNorm equivalent written in exportable ops (aten::rms_norm has no
    ONNX symbolic)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(x.dtype)


class ReverseLayoutExit(nn.Module):
    def forward(self, x):
        b = x.shape[0]
        return x.reshape(b, 8, 8, D).permute(0, 3, 1, 2)


class ReverseConvBlockNative(nn.Module):
    """ReverseConv on [B, 64 board-square channels, 16, 16 features]."""

    def __init__(self):
        super().__init__()
        self.norm = L1Norm(D)
        self.c1 = nn.Conv2d(64, 128, 3, padding=1)
        self.c2 = nn.Conv2d(128, 64, 3, padding=1, bias=False)

    def forward(self, x):
        b = x.shape[0]
        h = self.norm(x.reshape(b, 8, 8, D)).reshape(b, 64, 16, 16)
        h = F.leaky_relu(self.c1(h), 0.01)
        h = self.c2(h)
        return x + F.leaky_relu(h, 0.01)


class GlobalL1Norm(nn.Module):
    """One L1 magnitude scale for every sample, regardless of tensor layout."""

    def forward(self, x):
        x = x.clamp(-250.0, 250.0)
        dims = tuple(range(1, x.ndim))
        return x / (x.abs().mean(dim=dims, keepdim=True) + EPS)


def replace_l1_norms(module):
    for name, child in module.named_children():
        if isinstance(child, L1Norm):
            setattr(module, name, GlobalL1Norm())
        else:
            replace_l1_norms(child)


class ConvBlockGlobalL1(nn.Module):
    """Plain ConvBlock with global L1 normalization in its native NCHW layout."""

    def __init__(self):
        super().__init__()
        self.norm = GlobalL1Norm()
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        self.c2 = nn.Conv2d(D, D, 3, padding=1, bias=False)

    def forward(self, x):
        h = F.leaky_relu(self.c1(self.norm(x)), 0.01)
        h = self.c2(h)
        return x + F.leaky_relu(h, 0.01)


class ExpandyConvBlockMaybeFaster(nn.Module):
    """ExpandyConvBlock with a single norm; leaky at 2D between each grouped-3x3
    and its 1x1 contraction."""

    def __init__(self, groups=4):
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(2 * D, D, 1, bias=False)
        self.c2 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))
        h = self.p2(F.leaky_relu(self.c2(h), 0.05))
        return x + h


class ExpandyConvBlockFasterFatty(nn.Module):
    """ExpandyConvBlock with a single norm; leaky at 2D between each grouped-3x3
    and its 1x1 contraction."""

    def __init__(self, groups=4):
        D2 = 2*D
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, D2, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(D2, D2, 1, bias=False)
        self.c2 = nn.Conv2d(D2, D2, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(D2, D, 1, bias=False)

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))
        h = self.p2(F.leaky_relu(self.c2(h), 0.05))
        return x + h

    
class DCSoftMaxConvBlockLean(nn.Module):
    """DCSoftMaxConvBlock without the mid-block L1Norm (normb) or EMA buffers."""

    def __init__(self, ab_groups=4, sm_groups=64):
        super().__init__()
        self.sm_groups = sm_groups
        self.norm = L1Norm(D)
        self.c1 = nn.Conv2d(D, D, 3, padding=1)

        self.dw_a = nn.Conv2d(D, D, 3, padding=1, groups=ab_groups)
        self.p_a = nn.Conv2d(D, D, 1, bias=False)

        self.dw_b = nn.Conv2d(D, D, 3, padding=1, groups=ab_groups)
        self.p_b = nn.Conv2d(D, D, 1, bias=False)

        #self.dw_c = nn.Conv2d(D, D, 3, padding=1, groups=ab_groups)
        #self.p_c = nn.Conv2d(D, D, 1)

    def forward(self, x):
        n = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.leaky_relu(self.c1(n), 0.02)

        xa = self.p_a(F.leaky_relu(self.dw_a(h), 0.05))
        xb = self.p_b(F.leaky_relu(self.dw_b(h), 0.05))

        #g = F.sigmoid(xa)
        #return x + g*xa + (1.0-g)*xb
        #return x + g * xa + (1.0 - g) * xb
        #g = F.sigmoid(self.p_c(self.dw_c(h)))
        #return x + g * xa + (1.0 - g) * xb
        xg = xa
        B, C, H, W = xg.shape
        g = xg.view(B, self.sm_groups, C // self.sm_groups, H * W).softmax(dim=2)
        g = g.reshape(B, C, H, W)
        return x + g * xa + (1.0 - g) * xb
        

class GatedExpandedConvBlockLean(nn.Module):
    """GatedExpandedConvBlock (m13) without EMA buffers."""

    def __init__(self, groups=4):
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(2 * D, D, 1, bias=False)
        self.c_gate = nn.Conv2d(D, D, 1)
        self.c2 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))
        gate = torch.sigmoid(self.c_gate(h))
        h = self.p2(F.leaky_relu(self.c2(h), 0.05))
        return x + gate * h


class SEExpandedConvBlockLean(nn.Module):
    """SEExpandedConvBlock (m14) without EMA buffers."""

    def __init__(self, groups=4):
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(2 * D, D, 1, bias=False)
        self.c_gate_down = nn.Conv2d(D, D // 16, 1)
        self.c_gate_up = nn.Conv2d(D // 16, D, 1)
        self.c2 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))
        gd = F.leaky_relu(self.c_gate_down(h), 0.02)
        gate = torch.sigmoid(self.c_gate_up(gd))
        h = self.p2(F.leaky_relu(self.c2(h), 0.05))
        return x + gate * h


class GatedConvBlockLean(nn.Module):
    """GLU conv: 3x3 value conv gated per (channel, square) by a 1x1 sigmoid."""

    def __init__(self):
        super().__init__()
        self.norm = L1Norm(D)
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        self.cgate = nn.Conv2d(D, D, 1)
        self.c2 = nn.Conv2d(D, D, 3, padding=1, bias=False)

    def forward(self, x):
        h = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.leaky_relu(self.c1(h), 0.01)
        gate = torch.sigmoid(self.cgate(h).float()).to(h.dtype)
        return x + gate * self.c2(h)


class DeciderCombinerLean(nn.Module):
    """Soft-routed FF: sigmoid(k*H(x)) blends A(x)/B(x), leaky, down-project."""

    def __init__(self, hidden):
        super().__init__()
        self.A = nn.Linear(D, hidden, bias=False)
        self.B = nn.Linear(D, hidden, bias=False)
        self.H = nn.Linear(D, hidden, bias=False)
        self.down = nn.Linear(hidden, D, bias=False)
        self.k = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        a, b, h = self.A(x), self.B(x), self.H(x)
        g = torch.sigmoid(self.k.abs() * h.float()).to(a.dtype)
        sel = F.leaky_relu(g * a + (1.0 - g) * b, 0.05)
        return self.down(sel)


class MaxNetConvBlockLean(nn.Module):
    """Lean Version of MaxNet"""

    def __init__(self):
        super().__init__()
        self.norm = L1Norm(D)
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        self.c2a = nn.Conv2d(D, D, 3, padding=1, bias=False)
        self.c2b = nn.Conv2d(D, D, 3, padding=1, bias=False)
        self.k = 2.0

    def forward(self, x):
        h = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = F.leaky_relu(self.c1(h), 0.01)
        a = self.c2a(h)
        b = self.c2b(h)
        soft_gate = torch.sigmoid(self.k * (a - b))
        h = soft_gate * a + (1.0 - soft_gate) * b
        return x + F.relu(h)


class TxBlockDCLean(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.n1 = L1Norm(D)
        self.attn = Attention(D, HEADS)
        self.n2 = L1Norm(D)
        self.dc = DeciderCombinerLean(hidden)

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n)
        return x + self.dc(self.n2(x))


class TxBlockSwiGLULean(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.n1 = L1Norm(D)
        self.attn = Attention(D, HEADS)
        self.n2 = L1Norm(D)
        self.gate = nn.Linear(D, hidden)
        self.up = nn.Linear(D, hidden)
        self.down = nn.Linear(hidden, D, bias=False)

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n)
        n = self.n2(x)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))


class TxBlockSwiGLUNoOut(nn.Module):
    """SwiGLU TX block with Q/K/V attention and no output projection."""

    def __init__(self, hidden):
        super().__init__()
        self.n1 = L1Norm(D)
        self.attn = Attention(D, HEADS, output_proj=False)
        self.n2 = L1Norm(D)
        self.gate = nn.Linear(D, hidden)
        self.up = nn.Linear(D, hidden)
        self.down = nn.Linear(hidden, D, bias=False)

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n)
        n = self.n2(x)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))


MODEL_SPECS = {
    "m1": dict(name="m1_10c6t_conv-van_tx-van", conv="vanilla", tx="vanilla"),
    "m2": dict(name="m2_conv-maxnet_tx-van", conv="maxnet", tx="vanilla"),
    "m6": dict(name="m6_10c6t_conv-van_tx-dc1024", conv="vanilla", tx="dc"),
    "m7": dict(name="m7_10c6t_conv-van_tx-swiglu4x", conv="vanilla", tx="swiglu"),
    "m8": dict(name="m8_10c6t_conv-gated_tx-van", conv="gated", tx="vanilla"),
    "m11": dict(
        name="m11_10c6t_conv-expandywide10_tx-van", conv="expandy_fast_all", tx="vanilla"
    ),

    "m12": dict(
        name="m12_conv-dcsoftmax_tx-van", conv="dcsoftmax", tx="vanilla"
    ),

    "m13": dict(
        name="m13_10c6t_conv-gated-expandy_tx-van", conv="gated_expandy", tx="vanilla"
    ),
    "m13-7": dict(
        name="m13-7_10c6t_conv-gated-expandy_tx-3ff3swiglu4x",
        conv="gated_expandy", tx="ff_swiglu"
    ),
    "m13-1-7-5c10t": dict(
        name="m13-1-7-5c10t_conv-5gated-expandy_tx-5ff5swiglu4x",
        conv="gated_expandy_5", tx="ff_swiglu_5"
    ),
    "m11-1-6c12t-d256": dict(
        name="m11-1-6c12t-d256_conv-6expandy_tx-12ff4x",
        conv="expandy_fast_6", tx="vanilla_12"
    ),
    "m13-1-7-6c12t-d256": dict(
        name="m13-1-7-6c12t-d256_conv-6gated-expandy_tx-8ff4swiglu4x",
        conv="gated_expandy_6", tx="ff8_swiglu4_noout"
    ),
    "m1-13-1-7-8c8t-d256": dict(
        name="m1-13-1-7-8c8t-d256_conv-4plain4gated-expandy_tx-4ff4swiglu4x",
        conv="gated_expandy_4", tx="ff_swiglu_4"
    ),
    "m13-7-8c8t": dict(
        name="m13-7-8c8t",
        conv="gated_expandy_2plain_6", tx="ff4_swiglu3_4"
    ),
    "m13-1-8c4t-d256-512": dict(
        name="m13-1-8c4t-d256-512_conv-8gated-expandy_tx-4ff4x",
        builder="m13_1_8c4t_d256_512"
    ),
    "m11-1-8c4t-d256-d512": dict(
        name="m11-1-8c4t-d256-d512_conv-8expandy_tx-4ff3x",
        builder="m11_1_8c4t_d256_d512"
    ),
    "m11-1-8c4t-d256-512": dict(
        name="m11-1-8c4t-d256-512_conv-8expandy_tx-4ff3x",
        builder="m11_1_8c4t_d256_d512"
    ),
    "m11-1-8c10t": dict(
        name="m11-1-8c10t_conv-8expandy_tx-10ff4x",
        conv="expandy_fast_8", tx="vanilla_10"
    ),
    "m11-1-7-8c8t": dict(
        name="m11-1-7-8c8t_conv-8expandy_tx-4ff4swiglu4x",
        conv="expandy_fast_8", tx="ff_swiglu_4"
    ),
    "m11-13-1-7-6c10t-d256": dict(
        name="m11-13-1-7-6c10t-d256_conv-2expandy4gated-expandy_tx-6ff4swiglu4x",
        conv="expandy_gated_expandy_2_4", tx="ff_swiglu_6_4"
    ),
    "m14": dict(name="m14_10c6t_conv-se-expandy_tx-van", conv="se_expandy", tx="vanilla"),
    "m15": dict(name="m15_10c6t_conv-reverse_tx-van", conv="reverse", tx="vanilla"),
    "m16": dict(name="m16_10c6t_conv-van_tx-noout", conv="vanilla", tx="noout"),
    "m17": dict(name="m17_10c6t_conv-van-globall1_tx-van", conv="global_l1", tx="vanilla"),
    "m11-6": dict(
        name="m11-6_10c6t_conv-expandy_tx-dc1024", conv="expandy_fast_all", tx="dc"
    ),

    "m11-7": dict(
        name="m11-7_10c6t_conv-expandy_tx-swiglu4x", conv="expandy_fast_all",tx="swiglu"
    ),
        "m69": dict(
        name="m69_10c6t_conv-4plain4expandy2gated_tx-3ff3swiglu4x",
        conv="expandy_gated", tx="ff_swiglu"
    ),
}


class Bakeoff10c6t(BakeoffModel):
    """BakeoffModel stem/heads with the trunk swapped for 4 plain + 6 variant convs
    and 6 tx blocks."""

    def __init__(self, conv_mode, tx_mode):
        super().__init__(conv_mode="vanilla", tx_mode="vanilla")
        self.conv_mode = conv_mode
        self.tx_mode = tx_mode
        self.wdl_norm = RMSNormExport(D)
        self.convs = nn.ModuleList([ConvBlock() for _ in range(N_PLAIN)])
        if conv_mode == "vanilla":
            self.convs.extend([ConvBlock() for _ in range(N_VARIANT)])

        elif conv_mode == "global_l1":
            self.convs = nn.ModuleList(
                [ConvBlockGlobalL1() for _ in range(N_PLAIN + N_VARIANT)])

        elif conv_mode == "reverse":
            self.reverse_stem_layout = True
            self.convs = nn.ModuleList([])
            self.reverse_convs = nn.ModuleList(
                [ReverseConvBlockNative() for _ in range(N_PLAIN + N_VARIANT)]
                + [ReverseLayoutExit()])

        elif conv_mode == "maxnet":
            self.max_convs = nn.ModuleList(
                [MaxNetConvBlockLean() for _ in range(N_VARIANT)])
        elif conv_mode == "gated":
            self.gated_convs = nn.ModuleList(
                [GatedConvBlockLean() for _ in range(N_VARIANT)])
        elif conv_mode == "expandy":
            self.expandy_convs = nn.ModuleList(
                [ExpandyConvBlockMaybeFaster() for _ in range(N_VARIANT)])
        elif conv_mode == "expandy_gated":
            self.expandy_convs = nn.ModuleList(
                [ExpandyConvBlockMaybeFaster() for _ in range(4)])
            self.gated_convs = nn.ModuleList(
                [GatedConvBlockLean() for _ in range(2)])
        elif conv_mode == "expandy_gated_expandy_2_4":
            self.convs = nn.ModuleList([])
            self.expandy_convs = nn.ModuleList(
                [ExpandyConvBlockMaybeFaster() for _ in range(2)])
            self.gated_expandy_convs = nn.ModuleList(
                [GatedExpandedConvBlockLean() for _ in range(4)])
        elif conv_mode == "expandy_fast_all":
            self.convs = nn.ModuleList([])
            self.expandy_convs = nn.ModuleList(
                [ExpandyConvBlockMaybeFaster() for _ in range(N_PLAIN + N_VARIANT)])
        elif conv_mode == "expandy_fast_6":
            self.convs = nn.ModuleList([])
            self.expandy_convs = nn.ModuleList(
                [ExpandyConvBlockMaybeFaster() for _ in range(6)])
        elif conv_mode == "expandy_fast_8":
            self.convs = nn.ModuleList([])
            self.expandy_convs = nn.ModuleList(
                [ExpandyConvBlockMaybeFaster() for _ in range(8)])
        elif conv_mode == "gated_expandy":
            self.gated_expandy_convs = nn.ModuleList(
                [GatedExpandedConvBlockLean() for _ in range(N_VARIANT)])
        elif conv_mode == "gated_expandy_5":
            self.convs = nn.ModuleList([])
            self.gated_expandy_convs = nn.ModuleList(
                [GatedExpandedConvBlockLean() for _ in range(5)])
        elif conv_mode == "gated_expandy_4":
            self.gated_expandy_convs = nn.ModuleList(
                [GatedExpandedConvBlockLean() for _ in range(4)])
        elif conv_mode == "gated_expandy_6":
            self.convs = nn.ModuleList([])
            self.gated_expandy_convs = nn.ModuleList(
                [GatedExpandedConvBlockLean() for _ in range(6)])
        elif conv_mode == "gated_expandy_2plain_6":
            self.convs = nn.ModuleList([ConvBlock() for _ in range(2)])
            self.gated_expandy_convs = nn.ModuleList(
                [GatedExpandedConvBlockLean() for _ in range(6)])
        elif conv_mode == "se_expandy":
            self.gated_expandy_convs = nn.ModuleList(
                [SEExpandedConvBlockLean() for _ in range(N_VARIANT)])
        elif conv_mode == "dcsoftmax":
            self.sm_convs = nn.ModuleList(
                [DCSoftMaxConvBlockLean() for _ in range(N_VARIANT)])
        elif conv_mode == "expandy_fatty_all":
            self.convs = nn.ModuleList([])
            self.expandy_convs = nn.ModuleList(
                [ExpandyConvBlockFasterFatty() for _ in range(N_PLAIN + N_VARIANT)])
        else:
            raise ValueError(f"conv_mode={conv_mode}")

        if tx_mode == "vanilla":
            self.blocks = nn.ModuleList([TxBlockFF() for _ in range(N_TX)])
        elif tx_mode == "vanilla_8":
            self.blocks = nn.ModuleList([TxBlockFF() for _ in range(8)])
        elif tx_mode == "vanilla_10":
            self.blocks = nn.ModuleList([TxBlockFF() for _ in range(10)])
        elif tx_mode == "vanilla_12":
            self.blocks = nn.ModuleList([TxBlockFF() for _ in range(12)])
        elif tx_mode == "vanilla_12_noout":
            self.blocks = nn.ModuleList([TxBlockFFNoOut() for _ in range(12)])
        elif tx_mode == "noout":
            self.blocks = nn.ModuleList([TxBlockFFNoOut() for _ in range(N_TX)])
        elif tx_mode == "dc":
            self.blocks = nn.ModuleList([TxBlockDCLean(DC_H) for _ in range(N_TX)])
        elif tx_mode == "swiglu":
            self.blocks = nn.ModuleList(
                [TxBlockSwiGLULean(SWIGLU_H) for _ in range(N_TX)])
        elif tx_mode == "ff_swiglu":
            self.blocks = nn.ModuleList(
                [TxBlockFF() for _ in range(3)] +
                [TxBlockSwiGLULean(SWIGLU_H) for _ in range(3)])
        elif tx_mode == "ff_swiglu_5":
            self.blocks = nn.ModuleList(
                [TxBlockFF() for _ in range(5)] +
                [TxBlockSwiGLULean(SWIGLU_H) for _ in range(5)])
        elif tx_mode == "ff_swiglu_4":
            self.blocks = nn.ModuleList(
                [TxBlockFF() for _ in range(4)] +
                [TxBlockSwiGLULean(SWIGLU_H) for _ in range(4)])
        elif tx_mode == "ff4_swiglu3_4":
            self.blocks = nn.ModuleList(
                [TxBlockFF() for _ in range(4)] +
                [TxBlockSwiGLULean(3 * D) for _ in range(4)])
        elif tx_mode == "ff_swiglu_6_4":
            self.blocks = nn.ModuleList(
                [TxBlockFF() for _ in range(6)] +
                [TxBlockSwiGLULean(SWIGLU_H) for _ in range(4)])
        elif tx_mode == "ff8_swiglu4":
            self.blocks = nn.ModuleList(
                [TxBlockFF() for _ in range(8)] +
                [TxBlockSwiGLULean(SWIGLU_H) for _ in range(4)])
        elif tx_mode == "ff8_swiglu4_noout":
            self.blocks = nn.ModuleList(
                [TxBlockFFNoOut() for _ in range(8)] +
                [TxBlockSwiGLUNoOut(SWIGLU_H) for _ in range(4)])
        else:
            raise ValueError(f"tx_mode={tx_mode}")

        if conv_mode == "global_l1":
            replace_l1_norms(self)


class TxBlockFFD(nn.Module):
    """Plain FF transformer block at an arbitrary trunk width."""

    def __init__(self, dim, heads, ff_mult=4):
        super().__init__()
        self.n1 = L1Norm(dim)
        self.attn = Attention(dim, heads)
        self.n2 = L1Norm(dim)
        self.ff1 = nn.Linear(dim, ff_mult * dim)
        self.ff2 = nn.Linear(ff_mult * dim, dim, bias=False)

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n)
        return x + self.ff2(F.gelu(self.ff1(self.n2(x))))


class BakeoffM13_1_8c4tD256_512(nn.Module):
    """Eight d256 Gated-Expandy convs, then position-concatenated d512 tokens."""

    stem_dim = D
    dim = 512
    heads = 8

    def __init__(self, conv_kind="gated_expandy"):
        super().__init__()
        dim = self.dim
        self.tok_emb = nn.Embedding(b4.XC0H_VOCAB, b4.XC0H_DEMB)
        self.cast_emb = nn.Embedding(16, 64)
        self.stem_proj = nn.Linear(b4.XC0H_IN, self.stem_dim - 4)
        self.stem_norm = L1Norm(self.stem_dim - 4)
        for name in ("rep", "stm", "hmc", "ones", "checker", "rank", "file"):
            setattr(self, f"{name}_scale", nn.Parameter(torch.tensor(1.0)))
        sqs = torch.arange(b4.TOKENS).float()
        ranks_sq, files_sq = sqs // 8, sqs % 8
        self.register_buffer("xc0h_spatial", torch.stack([
            torch.ones(b4.TOKENS), ((ranks_sq + files_sq) % 2) * 2 - 1,
            ranks_sq / 7.0 * 2 - 1, files_sq / 7.0 * 2 - 1,
        ], dim=-1).unsqueeze(0))

        if conv_kind == "gated_expandy":
            conv_block = GatedExpandedConvBlockLean
        elif conv_kind == "expandy":
            conv_block = ExpandyConvBlockMaybeFaster
        else:
            raise ValueError(f"conv_kind={conv_kind}")
        self.convs = nn.ModuleList([conv_block() for _ in range(8)])
        self.pos = nn.Parameter(torch.randn(b4.TOKENS, self.stem_dim) * 0.02)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.global_tokens = nn.Parameter(torch.randn(1, 8, dim) * 0.02)
        self.blocks = nn.ModuleList([TxBlockFFD(dim, self.heads, ff_mult=3) for _ in range(4)])
        self.trunk = L1Norm(dim)

        self.wdl_w1 = nn.Linear(dim, dim)
        self.wdl_norm = RMSNormExport(dim)
        self.wdl_w2 = nn.Linear(dim, 3)

        self.from_proj = nn.Linear(dim, b4.PDH)
        self.from_skip = nn.Linear(dim, b4.PDH, bias=False)
        self.from_norm = L1Norm(b4.PDH)
        self.from_mha = Attention(b4.PDH, b4.POLICY_HEADS)
        self.from_out = nn.Linear(b4.PDH, b4.PDH, bias=False)
        self.to_proj = nn.Linear(dim, b4.PDH)
        self.to_skip = nn.Linear(dim, b4.PDH, bias=False)
        self.to_norm = L1Norm(b4.PDH)
        self.to_mha = Attention(b4.PDH, b4.POLICY_HEADS)
        self.to_out = nn.Linear(b4.PDH, b4.PDH, bias=False)
        self.from_dot_norm = L1Norm(b4.PDH)
        self.to_dot_norm = L1Norm(b4.PDH)
        self.promo_from = nn.Linear(b4.PDH, 3, bias=False)
        self.promo_to = nn.Linear(b4.PDH, 3, bias=False)

        gate_d = 2 * dim
        self.gate_w_gate = nn.Linear(gate_d, b4.SMARTGATE_H)
        self.gate_w_up = nn.Linear(gate_d, b4.SMARTGATE_H)
        self.gate_w_down = nn.Linear(b4.SMARTGATE_H, gate_d, bias=False)
        self.gate_norm = L1Norm(gate_d)
        self.gate_out = nn.Linear(gate_d, b4.POLICY_DIM)
        with torch.no_grad():
            self.gate_w_down.weight.normal_(std=10 ** -1.5)
            self.gate_out.weight.normal_(std=10 ** -1.5)
            self.gate_out.bias.fill_(4.0)
        self.register_buffer("sl_idx", torch.from_numpy(
            b4.pyfastchess.build_sometimes_legal_mask()).bool().nonzero(as_tuple=True)[0])

    def stem(self, x):
        b = x.shape[0]
        k = b4.poc.XC0H_K
        dtype = self.tok_emb.weight.dtype
        tok = x[:, :k * 64].long().view(b, k, b4.TOKENS)
        rep = x[:, k * 64:k * 64 + k].to(dtype)
        cast = x[:, k * 64 + k].long()
        stm = x[:, k * 64 + k + 1].to(dtype)
        hmc = x[:, k * 64 + k + 2].to(dtype)
        te = self.tok_emb(tok).permute(0, 2, 1, 3).reshape(b, b4.TOKENS, k * b4.XC0H_DEMB)
        rep_planes = (rep * 2.0 - 1.0).unsqueeze(1).expand(b, b4.TOKENS, k) * self.rep_scale
        cast_plane = self.cast_emb(cast).unsqueeze(-1)
        stm_plane = (stm * 2.0 - 1.0).view(b, 1, 1).expand(b, b4.TOKENS, 1) * self.stm_scale
        hmc_plane = (hmc / 99.0 * 2.0 - 1.0).view(b, 1, 1).expand(b, b4.TOKENS, 1) * self.hmc_scale
        fused = torch.cat([te, rep_planes, cast_plane, stm_plane, hmc_plane], dim=-1)
        proj = self.stem_norm(self.stem_proj(fused))
        scales = torch.stack([self.ones_scale, self.checker_scale, self.rank_scale, self.file_scale])
        spatial = self.xc0h_spatial.to(proj.dtype).expand(b, -1, -1) * scales
        return torch.cat([proj, spatial], dim=-1).reshape(b, b4.BOARD, b4.BOARD, self.stem_dim).permute(0, 3, 1, 2)

    def forward(self, x):
        if x.shape[1] != b4.XC0H_LEN:
            raise ValueError(f"expected xc0h [B,{b4.XC0H_LEN}], got {tuple(x.shape)}")
        b = x.shape[0]
        x = self.stem(x)
        for block in self.convs:
            x = block(x)
        x = x.permute(0, 2, 3, 1).reshape(b, b4.TOKENS, self.stem_dim)
        x = torch.cat([x, self.pos_scale * self.pos.unsqueeze(0).expand(b, -1, -1)], dim=-1)
        x = torch.cat([x, self.global_tokens.expand(b, -1, -1)], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.trunk(x)
        wdl = self.wdl_w2(self.wdl_norm(F.gelu(self.wdl_w1(x[:, b4.TOKENS, :]))))

        from_set = torch.cat([x[:, :64], x[:, 68:70], x[:, 70:71]], dim=1)
        to_set = torch.cat([x[:, :64], x[:, 68:70], x[:, 71:72]], dim=1)
        f_proj = self.from_skip(from_set) + F.gelu(self.from_proj(from_set))
        f_base = f_proj[:, :64] + self.from_mha(self.from_norm(f_proj)[:, :64], self.from_norm(f_proj))
        t_proj = self.to_skip(to_set) + F.gelu(self.to_proj(to_set))
        t_base = t_proj[:, :64] + self.to_mha(self.to_norm(t_proj)[:, :64], self.to_norm(t_proj))
        f_vec, t_vec = f_base + self.from_out(f_base), t_base + self.to_out(t_base)
        dots_full = torch.bmm(self.from_dot_norm(f_vec), self.to_dot_norm(t_vec).transpose(1, 2)).clamp(-b4.FP16_CLIP, b4.FP16_CLIP) * (b4.PDH ** -0.5)
        dots = dots_full.reshape(b, b4.TOKENS * b4.TOKENS)
        promo = (dots_full[:, 48:56, 56:64, None] + self.promo_from(f_vec[:, 48:56])[:, :, None] + self.promo_to(t_vec[:, 56:64])[:, None, :]).permute(0, 3, 2, 1).reshape(b, 192)
        policy = torch.cat([dots, promo], dim=-1)[:, self.sl_idx]

        gi = torch.cat([x[:, 67], x[:, 68]], dim=-1)
        gate = b4.soft_clamp(F.silu(self.gate_w_gate(gi)))
        down = b4.soft_clamp(self.gate_w_down(b4.soft_clamp(gate * b4.soft_clamp(self.gate_w_up(gi)))))
        gate_raw = self.gate_out(self.gate_norm((gi + down).clamp(-250.0, 250.0)))
        return policy + F.logsigmoid(gate_raw.clamp(-250.0, 250.0)), wdl


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="dir the results CSV is written to")
    p.add_argument("--models", nargs="+", default=MODELS_TO_TEST, metavar="ID")
    p.add_argument("--batch-sizes", nargs="+", type=int,
                   default=DEFAULT_BATCH_SIZES, metavar="B")
    p.add_argument("--skip-trt", action="store_true")
    p.add_argument("--fresh", action="store_true",
                   help="before each TRT run, remove that model's cached ONNX "
                        "export and compiled TensorRT engine")
    p.add_argument("--sync", action="store_true",
                   help="after the run, offer to write this run's rows into "
                        "the master CSV (overwrites those models)")
    p.add_argument("--build-master", action="store_true",
                   help="rebuild the master CSV from every speedtest_10c6t_*.csv "
                        "in --out-dir (latest run per model wins) and exit")
    return p.parse_args()


def write_csv(all_results, param_counts, batch_sizes, path, run):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_FIELDS)
        for label, res in all_results.items():
            model, backend = label.split("  ", 1)
            for b in batch_sizes:
                r = res.get(b)
                if r is not None:
                    w.writerow([model, param_counts[model], backend, b,
                                r["latency_ms"], r["throughput"], run])
    print(f"\nresults -> {path}")


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def run_stamp(path):
    return os.path.basename(path)[len("speedtest_10c6t_"):-len(".csv")]


def sync_master(run_csv, master_csv):
    """Replace every model present in run_csv with run_csv's rows in master_csv."""
    new = read_rows(run_csv)
    models = {r["model"] for r in new}
    old = read_rows(master_csv) if os.path.exists(master_csv) else []
    kept = [r for r in old if r["model"] not in models]
    write_rows(kept + new, master_csv)
    print(f"master <- {sorted(models)}  ({master_csv})")


def build_master(out_dir, master_csv):
    """Compile every run CSV in out_dir; per model, the newest run's rows win.
    Runs that predate the params column get it from the known counts."""
    paths = sorted(glob.glob(os.path.join(out_dir, "speedtest_10c6t_*.csv")))
    latest = {}
    for p in paths:
        stamp = run_stamp(p)
        for r in read_rows(p):
            r.setdefault("params", PARAMS_FALLBACK.get(r["model"], ""))
            r.setdefault("run", stamp)
            latest.setdefault(r["model"], {})
            if stamp >= latest[r["model"]].get("run", ""):
                if latest[r["model"]].get("run") != stamp:
                    latest[r["model"]] = {"run": stamp, "rows": []}
                latest[r["model"]]["rows"].append(r)
    rows = [r for m in sorted(latest) for r in latest[m]["rows"]]
    write_rows(rows, master_csv)
    for m in sorted(latest):
        print(f"  {m:<8} run {latest[m]['run']}  rows {len(latest[m]['rows'])}")
    print(f"master -> {master_csv}")


def timed_input(prompt, timeout):
    """input() that returns None if nothing is entered within timeout seconds."""
    box = []
    t = threading.Thread(target=lambda: box.append(input(prompt)), daemon=True)
    t.start()
    t.join(timeout)
    return box[0] if box else None


def wants_sync(answer):
    return answer is not None and answer.strip().lower() in YES_WORDS


def clear_model_trt_cache(onnx_path):
    """Remove an ONNX export and the engine derived from its current contents."""
    if not os.path.exists(onnx_path):
        return

    digest = hashlib.sha256()
    with open(onnx_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    prefix = f"bake_{digest.hexdigest()[:12]}"
    for engine_path in glob.glob(os.path.join(TRT_CACHE, f"{prefix}_*.engine")):
        os.remove(engine_path)
        print(f"  [fresh] removed engine: {engine_path}", flush=True)

    os.remove(onnx_path)
    print(f"  [fresh] removed ONNX:   {onnx_path}", flush=True)


def main():
    args = parse_args()
    master_csv = os.path.join(args.out_dir, MASTER_CSV)
    if args.build_master:
        build_master(args.out_dir, master_csv)
        return

    bs = args.batch_sizes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(TRT_CACHE, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"speedtest_10c6t_{stamp}.csv")

    print(f"PyTorch device: {device}", flush=True)
    print(f"TRT cache:      {TRT_CACHE}", flush=True)

    all_results = {}
    param_counts = {}
    for mid in args.models:
        spec = MODEL_SPECS[mid]
        print(f"\n{'='*60}", flush=True)
        if spec.get("builder") == "m13_1_8c4t_d256_512":
            model = BakeoffM13_1_8c4tD256_512()
        elif spec.get("builder") == "m11_1_8c4t_d256_d512":
            model = BakeoffM13_1_8c4tD256_512(conv_kind="expandy")
        else:
            model = Bakeoff10c6t(spec["conv"], spec["tx"])
        params = sum(p.numel() for p in model.parameters())
        param_counts[mid] = params
        print(f"  {mid}: {spec['name']}  params {params:,}", flush=True)

        ok, pt_p, pt_v, enc_np = verify_outputs(model, device, mid)
        if not ok:
            print(f"  [SKIP] {mid}: output verification failed", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        print(f"\n  Building {mid} PT eager ...", flush=True)
        eager = make_pt_eager_infer(model, device)
        lbl = f"{mid}  PT eager"
        all_results[lbl] = speed_test(lbl + " [fp16]", eager, bs)
        del eager

        if not args.skip_trt:
            lbl = f"{mid}  ORT TRT"
            onnx_path = os.path.join(TRT_CACHE, f"{spec['name']}_{params}.onnx")
            if args.fresh:
                clear_model_trt_cache(onnx_path)
            if not os.path.exists(onnx_path):
                export_to_onnx(model, device, onnx_path)
            else:
                print(f"  [TRT] ONNX cached: {onnx_path}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            trt_fn, trt_sess = make_trt_infer(onnx_path, TRT_CACHE, max_bs=max(bs))
            compare_pt_vs_export(pt_p, pt_v, trt_fn, enc_np, mid)
            all_results[lbl] = speed_test(lbl + " [fp16]", trt_fn, bs)
            del trt_sess
        else:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        write_csv(all_results, param_counts, bs, csv_path, stamp)

    print_summary_table(all_results, bs)
    print("\nDone.", flush=True)

    if args.sync and all_results:
        ans = timed_input(
            f"\nDo you want to sync CSV results into {MASTER_CSV}? "
            f"[y/N, {SYNC_TIMEOUT}s] ", SYNC_TIMEOUT)
        if wants_sync(ans):
            sync_master(csv_path, master_csv)
        else:
            print("not synced" if ans is not None else "\nno input, not synced")


if __name__ == "__main__":
    main()
