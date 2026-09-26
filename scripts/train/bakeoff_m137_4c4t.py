"""m13-7 4c4t d256 bake-off: growth-resistant activations on the prod m13-7 build.

The model mirrors build_pt_expandy_swiglu (model.py) op for op with LayerNorm
everywhere, except every attention layer uses a fused q|k|v weight with tiny
random q/v biases (k bias fixed 0); the training step, gate BCE, hotspot checks, grad report and data
pools are the prod trainer's own (imported). Train and val pools are built once
and shared, so every model draws from the same shard split but sees different
samples in a different order.

`act` replaces every conv/FF/head activation and bounds each SwiGLU as
silu(gate) * act(up); sigmoid gates and softmaxes are untouched. act=None is prod.

    symlog          sign(x) * log(1 + |x|)
    residual_decay  every trunk residual add is x = 0.9 * x + layer(x)
    gated           sigmoid gate on trunk SDPA output (query side), before o
    vanilla         prod m13-7, 1 plain + 3 gated-expandy conv, 2 FF + 2 SwiGLU tx

    python scripts/train/bakeoff_m137_4c4t.py [--only symlog,residual_decay]
"""
import argparse
import csv
import datetime
import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

import pyfastchess

import bootstrap_model_async_sparse_pkl as prod
from chessbot.model import attach_xc0h_stem, xc0h_stem_forward
from chessbot.pretrain import split_train_val
from chessbot.utils import batch_policy_metrics, format_time, print_validation

D, NH, FF_DIM, XC0H_K = 256, 8, 1024, 6
N_PLAIN, N_GATED, N_FF, N_SWIGLU = 1, 3, 2, 2
GROUPS = 4
DROPOUT = 0.05
LEAK = 0.03
SEQ_LEN = 64
N_TOKENS = SEQ_LEN + 8
PDH, POLICY_HEADS = 256, 4
GATE_D, SMARTGATE_H = 2 * D, 1024
GATE_MAX = 12.0
GATE_SCALE_INIT = 1.0
GATE_BIAS_MAX = 5.0
GATE_EMA_BETA = 1.0 - 1.0 / 500.0
BIAS_INIT_STD = 1e-3

LR_START, LR_END = 8e-4, 1e-4
LR_DECAY_FRAC = 0.8
STEPS_PER_EPOCH = 20
REPORT_EVERY = 10   # epochs: validation, gate EMA, hotspot checks, grad report
LR_SHOW_EVERY = 5
VAL_SIZE = 4096

MODEL_SPECS = {
    "symlog": dict(act="symlog"),
    "residual_decay": dict(res_decay=0.9),
    "gated": dict(attn_gate=True),
    "vanilla": dict(),
}

CSV_FIELDS = [
    "model", "epoch", "step", "lr", "value_mse", "value_corr", "value_ce",
    "policy_ce", "uniform_ce", "ce_gain", "top1_exact", "top1_mass", "top3_mass",
    "top5_mass", "top5_mass_target", "mass_on_legal", "avg_top_prob",
    "train_loss", "gate_loss", "gn_mean", "sec_per_epoch",
]

sl_idx_np = pyfastchess.build_sometimes_legal_mask()
SL_IDX = torch.from_numpy(sl_idx_np).bool().nonzero(as_tuple=True)[0]
N_MAIN = int((SL_IDX < 4096).sum())
# promo grid is type*64 + to_file*8 + from_file; gathered in pyfastchess's packed
# tail order: valid file pairs (from outer, to inner) x type (q, r, b)
PROMO_IDX = torch.tensor([
    t * 64 + tf * 8 + ff
    for ff in range(8) for tf in range(max(0, ff - 1), min(7, ff + 1) + 1)
    for t in range(3)])


class Tee:
    def __init__(self, path, stream):
        self.f = open(path, "w", buffering=1, encoding="utf-8")
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()


def lr_at(step, steps):
    start = int(LR_DECAY_FRAC * steps)
    if step < start:
        return LR_START
    return LR_START + (LR_END - LR_START) * (step - start) / max(1, steps - start)


def norm_nchw(norm, x):
    return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SymLog(nn.Module):
    """sign(x) * log(1 + |x|); autocast off so log1p stays fp16."""

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return torch.sign(x) * torch.log1p(x.abs())


ACTS = {
    "symlog": lambda ch: SymLog(),
}


def make_act(ch, act, default):
    """`act` for `ch` channels, or the prod `default` module when act is None."""
    return ACTS[act](ch) if act else default


def decayed(x, decay):
    """Residual stream carried into an add; decay 1.0 is a plain passthrough."""
    return x if decay == 1.0 else decay * x


def update_ema(mod, prefix, g):
    """lerp g's batch mean/std/range into mod.<prefix>_{mean,std,range}_ema."""
    with torch.no_grad():
        gf = g.detach().float()
        a = 1 - GATE_EMA_BETA
        getattr(mod, f"{prefix}_mean_ema").lerp_(gf.mean(0), a)
        getattr(mod, f"{prefix}_std_ema").lerp_(gf.std(0), a)
        getattr(mod, f"{prefix}_range_ema").lerp_(gf.amax(0) - gf.amin(0), a)


def register_ema(mod, prefix, shape):
    mod.register_buffer(f"{prefix}_mean_ema", torch.full(shape, 0.5))
    mod.register_buffer(f"{prefix}_std_ema", torch.full(shape, 0.16))
    mod.register_buffer(f"{prefix}_range_ema", torch.zeros(shape))


class GateOut(nn.Module):
    """1858 shut-logits (per-logit bias) + learned scale; project() clamps both."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1858, GATE_D) * 10 ** -1.5)
        self.bias = nn.Parameter(torch.randn(1858) * BIAS_INIT_STD)
        self.scale = nn.Parameter(torch.tensor(GATE_SCALE_INIT))

    def forward(self, g):
        return F.linear(g, self.weight, self.bias)

    def project(self):
        with torch.no_grad():
            self.bias.clamp_(0.0, GATE_BIAS_MAX)
            self.scale.clamp_(0.0, GATE_MAX)


class AttnGate(nn.Module):
    """sigmoid(Linear(xq)) on the SDPA output; ag_* EMAs per (token, channel)."""

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Linear(dim, dim)
        register_ema(self, "ag", (N_TOKENS, dim))

    def forward(self, a, xq):
        gate = torch.sigmoid(self.g(xq))
        if self.training and torch.is_grad_enabled():
            update_ema(self, "ag", gate)
        return a * gate


class Attention(nn.Module):
    """Fused q|k|v weight; k bias is structurally 0 (softmax is shift-invariant
    per query), q/v biases learned. self_attn runs one GEMM; cross-attn runs q off
    xq and k|v off xkv from the same weight. gate adds a sigmoid output gate."""

    def __init__(self, dim, heads, self_attn, gate=False):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.bq = nn.Parameter(torch.randn(dim) * BIAS_INIT_STD)
        self.bv = nn.Parameter(torch.randn(dim) * BIAS_INIT_STD)
        self.o = nn.Linear(dim, dim)
        self.gate = AttnGate(dim) if gate else None
        self.self_attn = self_attn
        self.heads = heads
        self.head_dim = dim // heads

    def split_heads(self, t):
        B, n, _ = t.shape
        return t.reshape(B, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, xq, xkv):
        B, nq, dim = xq.shape
        b = torch.cat([self.bq, self.bq.new_zeros(dim), self.bv])
        if self.self_attn:
            h = self.qkv(xq)
            h = h + b.to(h.dtype)
            q, k, v = h.chunk(3, dim=-1)
        else:
            W = self.qkv.weight
            q = F.linear(xq, W[:dim], b[:dim])
            k, v = F.linear(xkv, W[dim:], b[dim:]).chunk(2, dim=-1)
        q, k, v = (self.split_heads(t) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, nq, dim)
        if self.gate is not None:
            a = self.gate(a, xq)
        return self.o(a)


class ConvBlock(nn.Module):
    def __init__(self, act, decay):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.c1 = nn.Conv2d(D, D, 3, padding=1)
        self.c2 = nn.Conv2d(D, D, 3, padding=1, bias=False)
        self.a1 = make_act(D, act, nn.LeakyReLU(LEAK))
        self.a2 = make_act(D, act, nn.LeakyReLU(LEAK))
        self.decay = decay

    def forward(self, x):
        h = self.a1(self.c1(norm_nchw(self.norm, x)))
        return decayed(x, self.decay) + self.a2(self.c2(h))


class GatedExpandyConvBlock(nn.Module):
    """Gate mean/std/range per (channel, square) tracked as 500-step EMAs."""

    def __init__(self, act, decay):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=GROUPS)
        self.p1 = nn.Conv2d(2 * D, D, 1, bias=False)
        self.c_gate = nn.Conv2d(D, D, 1)
        self.c2 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=GROUPS)
        self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)
        self.a1 = make_act(2 * D, act, nn.LeakyReLU(LEAK))
        self.a2 = make_act(2 * D, act, nn.LeakyReLU(LEAK))
        self.decay = decay
        register_ema(self, "gate", (D, 8, 8))

    def forward(self, x):
        h1 = self.p1(self.a1(self.c1(norm_nchw(self.n1, x))))
        gate = torch.sigmoid(self.c_gate(h1))
        if self.training and torch.is_grad_enabled():
            update_ema(self, "gate", gate)
        return decayed(x, self.decay) + gate * self.p2(self.a2(self.c2(h1)))


class MaxMonitor(nn.Module):
    """Training-only hotspot stats (max |x| and its (token, channel), per-step
    max, mean) in device buffers while `active`; read_reset() is the one host
    sync. (T, C) are the last two dims; 2-D inputs report (C,)."""

    def __init__(self):
        super().__init__()
        self.active = False
        self.has_t = True
        self.steps = 0
        self.n_act = 0
        for name in ("cur", "peak", "maxsum", "actsum"):
            self.register_buffer(name, torch.zeros(()), persistent=False)
        for name in ("cur_tc", "peak_tc"):
            self.register_buffer(
                name, torch.zeros(2, dtype=torch.long), persistent=False)

    def forward(self, x):
        if self.active and self.training and torch.is_grad_enabled():
            with torch.no_grad():
                a = x.abs().reshape(-1)
                i = a.argmax()
                v = a[i].float()
                self.has_t = x.dim() >= 3
                C = x.shape[-1]
                T = x.shape[-2] if self.has_t else 1
                tc = torch.stack([i // C % T, i % C])
                self.cur_tc.copy_(torch.where(v > self.cur, tc, self.cur_tc))
                self.cur.copy_(torch.maximum(self.cur, v))
                self.actsum.add_(x.mean(dtype=torch.float32))
                self.n_act += 1

    def end_step(self):
        if self.active:
            self.peak_tc.copy_(
                torch.where(self.cur > self.peak, self.cur_tc, self.peak_tc))
            self.peak.copy_(torch.maximum(self.peak, self.cur))
            self.maxsum.add_(self.cur)
            self.cur.zero_()
            self.steps += 1

    def read_reset(self):
        peak, maxsum, actsum, t, c = torch.cat([
            torch.stack([self.peak, self.maxsum, self.actsum]),
            self.peak_tc.float()]).tolist()
        out = (peak, maxsum / max(self.steps, 1), actsum / max(self.n_act, 1),
               (int(t), int(c)) if self.has_t else (int(c),))
        for buf in (self.cur, self.peak, self.maxsum, self.actsum,
                    self.cur_tc, self.peak_tc):
            buf.zero_()
        self.steps = self.n_act = 0
        return out


class TxBlockFF(nn.Module):
    def __init__(self, down_mon, act, gate, decay):
        super().__init__()
        self.decay = decay
        self.n1 = nn.LayerNorm(D)
        self.attn = Attention(D, NH, self_attn=True, gate=gate)
        self.n2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF_DIM)
        self.ff_act = make_act(FF_DIM, act, nn.GELU())
        self.ff2 = nn.Linear(FF_DIM, D, bias=False)
        self.down_mon = down_mon
        self.attn_drop = nn.Dropout(DROPOUT)
        self.ff_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        n = self.n1(x)
        x = decayed(x, self.decay) + self.attn_drop(self.attn(n, n))
        down_out = self.ff2(self.ff_act(self.ff1(self.n2(x))))
        self.down_mon(down_out)
        return decayed(x, self.decay) + self.ff_drop(down_out)


class TxBlockSwiGLU(nn.Module):
    def __init__(self, down_mon, prod_mon, act, gate, decay):
        super().__init__()
        self.decay = decay
        self.n1 = nn.LayerNorm(D)
        self.attn = Attention(D, NH, self_attn=True, gate=gate)
        self.n2 = nn.LayerNorm(D)
        self.gate = nn.Linear(D, 4 * D)
        self.up = nn.Linear(D, 4 * D)
        self.up_act = make_act(4 * D, act, nn.Identity())
        self.down = nn.Linear(4 * D, D, bias=False)
        self.down_mon = down_mon
        self.prod_mon = prod_mon
        self.attn_drop = nn.Dropout(DROPOUT)
        self.ff_drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        n = self.n1(x)
        x = decayed(x, self.decay) + self.attn_drop(self.attn(n, n))
        n = self.n2(x)
        product = F.silu(self.gate(n)) * self.up_act(self.up(n))
        self.prod_mon(product)
        down_out = self.down(product)
        self.down_mon(down_out)
        return decayed(x, self.decay) + self.ff_drop(down_out)


class M137(nn.Module):
    """m13-7 4c4t. Routing: 64-66 W/D/L, 67+68 gate, from = 68,69,70,
    to = 68,69,71."""

    def __init__(self, act=None, attn_gate=False, res_decay=1.0):
        super().__init__()
        attach_xc0h_stem(self, D, XC0H_K, norm_type="layernorm")
        self.convs = nn.ModuleList(
            [ConvBlock(act, res_decay) for _ in range(N_PLAIN)]
            + [GatedExpandyConvBlock(act, res_decay) for _ in range(N_GATED)])

        self.pos = nn.Parameter(torch.randn(SEQ_LEN, D) * 0.02)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.global_tokens = nn.Parameter(torch.randn(1, 8, D) * 0.02)
        self.ff_down_mon = MaxMonitor()
        self.swiglu_down_mon = MaxMonitor()
        self.swiglu_prod_mon = MaxMonitor()
        self.res_mon = MaxMonitor()
        self.blocks = nn.ModuleList(
            [TxBlockFF(self.ff_down_mon, act, attn_gate, res_decay)
             for _ in range(N_FF)]
            + [TxBlockSwiGLU(self.swiglu_down_mon, self.swiglu_prod_mon,
                             act, attn_gate, res_decay) for _ in range(N_SWIGLU)])
        self.trunk = nn.LayerNorm(D)

        vh = 64
        self.wdl_hw = nn.Linear(D, vh, bias=True)
        self.wdl_hd = nn.Linear(D, vh, bias=True)
        self.wdl_hl = nn.Linear(D, vh, bias=True)
        self.wdl_aw = make_act(vh, act, nn.GELU())
        self.wdl_ad = make_act(vh, act, nn.GELU())
        self.wdl_al = make_act(vh, act, nn.GELU())
        self.wdl_ow = nn.Linear(vh, 1, bias=False)
        self.wdl_od = nn.Linear(vh, 1, bias=False)
        self.wdl_ol = nn.Linear(vh, 1, bias=False)

        self.from_proj = nn.Linear(D, PDH)
        self.from_act = make_act(PDH, act, nn.GELU())
        self.from_norm = nn.LayerNorm(PDH)
        self.from_mha = Attention(PDH, POLICY_HEADS, self_attn=False)
        self.from_out = nn.Linear(PDH, PDH, bias=False)
        self.to_proj = nn.Linear(D, PDH)
        self.to_act = make_act(PDH, act, nn.GELU())
        self.to_norm = nn.LayerNorm(PDH)
        self.to_mha = Attention(PDH, POLICY_HEADS, self_attn=False)
        self.to_out = nn.Linear(PDH, PDH, bias=False)
        self.from_dot_norm = nn.LayerNorm(PDH)
        self.to_dot_norm = nn.LayerNorm(PDH)
        self.promo_from = nn.Linear(PDH, 3, bias=False)
        self.promo_to = nn.Linear(PDH, 3, bias=False)

        self.gate_ff1 = nn.Linear(GATE_D, SMARTGATE_H)
        self.gate_act = make_act(SMARTGATE_H, act, nn.GELU())
        self.gate_ff2 = nn.Linear(SMARTGATE_H, GATE_D, bias=False)
        self.gate_norm = nn.LayerNorm(GATE_D)
        self.gate_out = GateOut()
        with torch.no_grad():
            self.gate_ff2.weight.normal_(std=10 ** -1.5)
        self.gate_act_mon = MaxMonitor()
        self.gate_raw = None
        self.dot_raw_mon = MaxMonitor()
        self.register_buffer("sl_idx", SL_IDX.clone())
        self.register_buffer("promo_idx", PROMO_IDX.clone(), persistent=False)

    def forward(self, x_in):
        B = x_in.shape[0]
        x = xc0h_stem_forward(self, x_in, D, XC0H_K)
        for blk in self.convs:
            x = blk(x)
        board = x.permute(0, 2, 3, 1).reshape(B, SEQ_LEN, D)
        x = board + self.pos_scale * self.pos.unsqueeze(0)
        x = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        self.res_mon(x)
        x = self.trunk(x)
        w = self.wdl_ow(self.wdl_aw(self.wdl_hw(x[:, 64, :])))
        d = self.wdl_od(self.wdl_ad(self.wdl_hd(x[:, 65, :])))
        l = self.wdl_ol(self.wdl_al(self.wdl_hl(x[:, 66, :])))
        wdl = torch.cat([w, d, l], dim=-1)

        from_set = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:71, :]], dim=1)
        to_set = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 71:72, :]], dim=1)
        f_proj = from_set + self.from_act(self.from_proj(from_set))
        fn = self.from_norm(f_proj)
        f_base = f_proj[:, :64, :] + self.from_mha(fn[:, :64, :], fn)
        fv = f_base + self.from_out(f_base)
        t_proj = to_set + self.to_act(self.to_proj(to_set))
        tn = self.to_norm(t_proj)
        t_base = t_proj[:, :64, :] + self.to_mha(tn[:, :64, :], tn)
        tv = t_base + self.to_out(t_base)
        dots_raw = torch.bmm(self.from_dot_norm(fv), self.to_dot_norm(tv).transpose(1, 2))
        self.dot_raw_mon(dots_raw)
        dots_full = dots_raw * (PDH ** -0.5)
        dots = dots_full.reshape(B, 64 * 64)
        dots_sub = dots_full[:, 48:56, 56:64]
        pf = self.promo_from(fv[:, 48:56, :])
        pt = self.promo_to(tv[:, 56:64, :])
        promo = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]
                 ).permute(0, 3, 2, 1).reshape(B, 192)
        policy = torch.cat(
            [dots[:, self.sl_idx[:N_MAIN]], promo[:, self.promo_idx]], dim=-1)

        gi = torch.cat([x[:, 67, :], x[:, 68, :]], dim=-1)
        down = self.gate_ff2(self.gate_act(self.gate_ff1(gi)))
        g = self.gate_norm(gi + down)
        self.gate_raw = self.gate_out(g)
        gate = self.gate_out.scale * torch.sigmoid(self.gate_raw)
        self.gate_act_mon(gate)
        return policy - gate, wdl

    def hotspot_monitors(self):
        return {
            "smartgate": self.gate_act_mon,
            "policy_dot": self.dot_raw_mon,
            "ff_down": self.ff_down_mon,
            "swiglu_prod": self.swiglu_prod_mon,
            "swiglu_down": self.swiglu_down_mon,
            "residual": self.res_mon,
        }

    def set_dropout(self, duty):
        """Each TxBlock dropout site runs with chance `duty`, flipped
        independently; off = eval mode, no kernel launched."""
        for blk in self.blocks:
            blk.attn_drop.train(random.random() < duty)
            blk.ff_drop.train(random.random() < duty)

    def project_params(self):
        """Clamp constrained params back into range; call after each opt step."""
        self.gate_out.project()

    def gate_param_stats(self):
        b = self.gate_out.bias.detach().float()
        return tuple(torch.stack(
            [b.min(), b.mean(), b.max(), self.gate_out.scale.detach().float()]).tolist())

    def set_hotspot_scan(self, on):
        for mon in self.hotspot_monitors().values():
            mon.active = on

    def hotspot_end_step(self):
        for mon in self.hotspot_monitors().values():
            mon.end_step()

    def hotspot_stats(self):
        return {k: mon.read_reset() for k, mon in self.hotspot_monitors().items()}


def evaluate(model, bundle, epoch, batch_size, device):
    enc = torch.as_tensor(bundle["enc_in"], dtype=torch.long).to(device)
    model.eval()
    all_pol, all_val = [], []
    with torch.no_grad(), autocast("cuda"):
        for start in range(0, enc.shape[0], batch_size):
            pol, val = model(enc[start:start + batch_size])
            all_pol.append(pol.float().cpu().numpy())
            all_val.append(val.float().cpu().numpy())
    pol_preds = np.concatenate(all_pol, axis=0)
    val_logits = np.concatenate(all_val, axis=0)
    target_wdl = bundle["value_out"]
    pstack = bundle["policy_logits"]
    mstack = (pstack > 0).astype(np.int32)

    val_wdl = scipy.special.softmax(val_logits, axis=1)
    val_q = val_wdl[:, 0] - val_wdl[:, 2]
    tgt_q = target_wdl[:, 0] - target_wdl[:, 2]
    stats = {
        "value_mse": np.mean((val_q - tgt_q) ** 2),
        "value_corr": np.corrcoef(val_q, tgt_q)[0, 1],
        "value_ce": -np.mean(
            np.sum(target_wdl * np.log(np.clip(val_wdl, 1e-9, None)), axis=1)),
        **batch_policy_metrics(pol_preds, pstack, mstack),
    }
    print_validation(epoch, stats)
    return stats


def train_one(key, spec, args, train_pool, val_pool, writer, fh, out_dir):
    name = f"m13-7-4c4t-{key}"
    device = args.device
    model = M137(**spec).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    epochs = args.steps // STEPS_PER_EPOCH
    print(f"\n=== {name}  {spec or 'prod'}  params {n_params:,}"
          f"  steps {args.steps}  epochs {epochs} ===", flush=True)

    opt = prod.make_adam(model, LR_START)
    scaler = GradScaler("cuda", init_scale=2 ** 15)
    clip_state = prod.new_clip_state()
    begin = time.time()
    epoch_times = []
    t_fetch = t_fit = t_eval = 0.0
    total_samples = window_samples = 0
    win_loss = win_gate = win_gn = 0.0
    win_n = win_gate_n = win_gn_n = 0

    for ep in range(epochs):
        lr = lr_at(ep * STEPS_PER_EPOCH, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        epoch_start = time.time()
        print("-" * 89)
        report = ep % REPORT_EVERY == 0 or ep == epochs - 1

        if report:
            t0 = time.time()
            stats = evaluate(model, val_pool.get(), ep, args.batch_size, device)
            row ={k: stats.get(k, "") for k in CSV_FIELDS}
            row.update(
                model=name, epoch=ep, step=ep * STEPS_PER_EPOCH, lr=f"{lr:.4e}",
                train_loss=win_loss / win_n if win_n else "",
                gate_loss=win_gate / win_gate_n if win_gate_n else "",
                gn_mean=win_gn / win_gn_n if win_gn_n else "",
                sec_per_epoch=np.mean(epoch_times) if epoch_times else "")
            writer.writerow(row)
            fh.flush()
            win_loss = win_gate = win_gn = 0.0
            win_n = win_gate_n = win_gn_n = 0
            t_eval += time.time() - t0

        t0 = time.time()
        bundle = train_pool.get()
        t_fetch += time.time() - t0

        t0 = time.time()
        model.set_hotspot_scan(report)
        p_loss, v_loss, t_loss, gns = prod.fit_epoch(
            model, opt, scaler, bundle, prod.pt_clipnorm_for_epoch(ep), device,
            clip_state, report=report)
        t_fit += time.time() - t0

        n_this = int(bundle["enc_in"].shape[0])
        total_samples += n_this
        window_samples += n_this
        win_loss += t_loss
        win_n += 1
        if gns["g_loss"] is not None:
            win_gate += gns["g_loss"]
            win_gate_n += 1
        if gns["scanned"]:
            win_gn += gns["gn_mean"]
            win_gn_n += 1

        g_str = f"gate_loss: {gns['g_loss']:.2f}  " if gns["g_loss"] is not None else ""
        lr_str = f"  lr: {lr:.2e}" if ep % LR_SHOW_EVERY == 0 else ""
        print(f"[epoch {ep:4d}] policy_loss: {p_loss:.2f}  "
              f"value_loss: {v_loss:.2f}  {g_str}total: {t_loss:.2f}  "
              f"samples: {total_samples:,}{lr_str}", flush=True)
        prod.epoch_diagnostics(model, scaler, gns, report)

        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        print(f"[time check] last: {format_time(elapsed)}  "
              f"avg: {format_time(np.mean(epoch_times))}  "
              f"total: {format_time(time.time() - begin)}")
        if report and ep > 0:
            print(f"[timing/{REPORT_EVERY}ep] fetch: {t_fetch:.2f}s  fit: {t_fit:.2f}s  "
                  f"eval: {t_eval:.2f}s  samples_window: {window_samples:,}  "
                  f"samples_total: {total_samples:,}")
            t_fetch = t_fit = t_eval = 0.0
            window_samples = 0
        print(flush=True)

    ckpt = out_dir / f"{name}.pt"
    torch.save({"model": model.state_dict(), "spec": spec, "steps": args.steps}, ckpt)
    print(f"[save] {ckpt}", flush=True)
    del model, opt, scaler
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--shard-dir", default=prod.DEFAULT_SHARD_DIR)
    ap.add_argument("--out-dir",
                    default=(r"C:\Users\Bryan\Data\chessbot\_data\models"
                             r"\bakeoff_m137_4c4t"))
    ap.add_argument("--only", default=None,
                    help="comma-separated MODEL_SPECS keys (default: all, in order)")
    args = ap.parse_args()

    todo = list(MODEL_SPECS) if not args.only else args.only.split(",")
    for key in todo:
        if key not in MODEL_SPECS:
            raise ValueError(f"--only: unknown model {key!r}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = Tee(out_dir / f"bakeoff_{stamp}.log", sys.stdout)
    sys.stderr = Tee(out_dir / f"bakeoff_{stamp}.err", sys.stderr)

    prod.PT_BATCH_SIZE = args.batch_size
    epoch_size = args.batch_size * STEPS_PER_EPOCH
    train_files, val_files = split_train_val(prod.list_shards(args.shard_dir))
    print(f"[data] {len(train_files)} train shards / {len(val_files)} val shards"
          f"  epoch {epoch_size:,}  models {', '.join(todo)}", flush=True)
    train_pool = prod.AsyncShardPool(
        train_files, high=prod.BUFFER_CAP, low=prod.BUFFER_CAP - prod.REFILL_BAND,
        epoch_size=epoch_size, drain=True, n_ready=2, name="train")
    val_pool = prod.AsyncShardPool(
        val_files, high=prod.VAL_BUFFER_CAP, low=prod.VAL_BUFFER_CAP - prod.REFILL_BAND,
        epoch_size=VAL_SIZE, drain=True, n_ready=1, name="val", subsample=False)
    train_pool.start()
    val_pool.start()

    csv_path = out_dir / f"bakeoff_metrics_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        print(f"[csv] {csv_path}", flush=True)
        for key in todo:
            train_one(key, MODEL_SPECS[key], args, train_pool, val_pool,
                      writer, fh, out_dir)
    train_pool.stop()
    val_pool.stop()
    print(f"\n[done] {len(todo)} models -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
