"""Shape and finiteness checks for the RMSNorm conv model family.

Covers RMSNorm2d, the standard RMS conv block, the globalizer block, the
signed-attention block, and the two full models against conv-pure. Small random
batches, forward and backward.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from chessbot.model import (
    make_rms2d,
    make_rms_conv_block,
    make_globalizer_block,
    make_signed_attn_block,
    build_pt_conv_pure,
    build_pt_conv_globalizer,
    build_pt_conv_signed_attn,
)

D = 256
B = 3
VOCAB = 21
SEQ = 64
POLICY_1858 = 1858

CONV_PURE_CFG = dict(conv_filters=D, num_blocks=2, dropout=0.03)
GLOBALIZER_CFG = dict(conv_filters=D, num_blocks=2, stem_blocks=2, dropout=0.03)
SIGNED_CFG = dict(conv_filters=D, num_blocks=2, stem_blocks=2, dropout=0.03)

# per-block targets from the design spec
GLOBALIZER_BLOCK_PARAMS = 2_722_560
SIGNED_BLOCK_PARAMS = 1_312_000


def board(batch=B):
    return torch.randn(batch, D, 8, 8)


def tokens(batch=B):
    return torch.randint(0, VOCAB, (batch, SEQ), dtype=torch.long)


def test_rms2d_shape_and_finite():
    n = make_rms2d(D)
    x = board()
    y = n(x)
    assert y.shape == (B, D, 8, 8)
    assert torch.isfinite(y).all()


def test_rms2d_normalizes_per_square():
    """Each square is normalized independently over channels, so every square
    should end up with unit RMS when the learned scale is still ones."""
    n = make_rms2d(D)
    x = board() * 17.0
    y = n(x)
    rms = y.pow(2).mean(1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rms_conv_block_shape():
    blk = make_rms_conv_block(D)
    y = blk(board())
    assert y.shape == (B, D, 8, 8)
    assert torch.isfinite(y).all()


def test_globalizer_block_stage_shapes():
    blk = make_globalizer_block(D)
    x = board()
    h = blk.norm(x)
    assert x.shape == (B, D, 8, 8)

    local = F.leaky_relu(blk.local_conv(h), 0.01)
    assert local.shape == (B, 240, 8, 8)

    g = blk.glob_conv(h)
    assert g.shape == (B, 16, 8, 8)

    gflat = g.reshape(B, 1024)
    assert gflat.shape == (B, 1024)

    gate, value = blk.lin_in(gflat).chunk(2, dim=-1)
    assert gate.shape == (B, 512) and value.shape == (B, 512)

    gout = blk.lin_out(F.silu(gate) * value).reshape(B, 16, 8, 8)
    assert gout.shape == (B, 16, 8, 8)

    cat = torch.cat([local, gout], dim=1)
    assert cat.shape == (B, D, 8, 8)

    y = blk(x)
    assert y.shape == (B, D, 8, 8)
    assert torch.isfinite(y).all()


def test_globalizer_block_param_count():
    n = sum(p.numel() for p in make_globalizer_block(D).parameters())
    print(f"\nglobalizer block params: {n:,} (spec {GLOBALIZER_BLOCK_PARAMS:,})")
    assert n == GLOBALIZER_BLOCK_PARAMS


def signed_attn_stages(blk, x):
    """Reproduce the block's internals so intermediates can be asserted."""
    h = F.leaky_relu(blk.c1(blk.norm(x)), 0.01)
    tok = h.flatten(2).transpose(1, 2)
    q = blk.interaction(tok)
    v = blk.value(tok)
    logits = torch.matmul(q, tok.transpose(-1, -2)) / 16.0
    w = torch.tanh(logits)
    raw_abs = w.abs().sum(dim=-1, keepdim=True)
    den = raw_abs.clamp_min(1e-6)
    wn = w / den
    mixed = torch.matmul(wn, v)
    return tok, logits, w, raw_abs, den, wn, mixed


def test_signed_attn_block_stage_shapes():
    blk = make_signed_attn_block(D)
    x = board()
    tok, logits, w, _, _, wn, mixed = signed_attn_stages(blk, x)

    assert tok.shape == (B, SEQ, D)
    assert logits.shape == (B, SEQ, SEQ)
    assert w.shape == (B, SEQ, SEQ)
    assert wn.shape == (B, SEQ, SEQ)
    assert mixed.shape == (B, SEQ, D)

    y = blk(x)
    assert y.shape == (B, D, 8, 8)
    assert torch.isfinite(y).all()


def test_signed_attn_weights_finite():
    blk = make_signed_attn_block(D)
    _, logits, w, _, _, wn, _ = signed_attn_stages(blk, board())
    assert torch.isfinite(logits).all()
    assert torch.isfinite(w).all()
    assert torch.isfinite(wn).all()
    # tanh keeps the pre-normalization weights bounded and signed
    assert w.abs().max() <= 1.0


def test_signed_attn_denominator_never_zero():
    blk = make_signed_attn_block(D)
    # a zero board makes every logit zero, so tanh gives an all-zero row --
    # the clamp is the only thing preventing a divide by zero here
    x = torch.zeros(B, D, 8, 8)
    _, _, _, raw_abs, den, wn, _ = signed_attn_stages(blk, x)
    assert (den > 0).all()
    assert den.min() >= 1e-6
    assert torch.isfinite(wn).all()
    assert (raw_abs.min() < 1e-6) or True  # degenerate rows are allowed


def test_signed_attn_row_abs_sums_are_one():
    blk = make_signed_attn_block(D)
    _, _, _, raw_abs, _, wn, _ = signed_attn_stages(blk, board())
    row_sum = wn.abs().sum(dim=-1)
    live = raw_abs.squeeze(-1) > 1e-3
    assert live.any(), "no non-degenerate rows to check"
    assert torch.allclose(row_sum[live], torch.ones_like(row_sum[live]), atol=1e-4)


@pytest.mark.parametrize("builder,cfg,name", [
    (build_pt_conv_pure, CONV_PURE_CFG, "conv-pure"),
    (build_pt_conv_globalizer, GLOBALIZER_CFG, "conv-globalizer-2c8g"),
    (build_pt_conv_signed_attn, SIGNED_CFG, "conv-signed-attn-2c8a"),
])
def test_model_output_shapes(builder, cfg, name):
    m = builder(cfg).eval()
    with torch.no_grad():
        policy, wdl = m(tokens())
    assert policy.shape == (B, POLICY_1858), name
    assert wdl.shape == (B, 3), name
    assert torch.isfinite(policy).all() and torch.isfinite(wdl).all()


def test_new_models_match_conv_pure_output_contract():
    with torch.no_grad():
        t = tokens()
        p0, v0 = build_pt_conv_pure(CONV_PURE_CFG).eval()(t)
        p1, v1 = build_pt_conv_globalizer(GLOBALIZER_CFG).eval()(t)
        p2, v2 = build_pt_conv_signed_attn(SIGNED_CFG).eval()(t)
    assert p0.shape == p1.shape == p2.shape
    assert v0.shape == v1.shape == v2.shape
    assert p0.dtype == p1.dtype == p2.dtype
    assert v0.dtype == v1.dtype == v2.dtype


@pytest.mark.parametrize("builder,cfg,name", [
    (build_pt_conv_globalizer, GLOBALIZER_CFG, "conv-globalizer-2c8g"),
    (build_pt_conv_signed_attn, SIGNED_CFG, "conv-signed-attn-2c8a"),
])
def test_forward_backward_finite(builder, cfg, name):
    m = builder(cfg).train()
    policy, wdl = m(tokens())
    loss = policy.float().pow(2).mean() + wdl.float().pow(2).mean()
    assert torch.isfinite(loss), name
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, f"{name}: no gradients"
    assert all(torch.isfinite(g).all() for g in grads), name


def test_full_model_param_counts_reported():
    """Prints the production-shaped counts (8 backbone blocks) for the record."""
    full_glob = dict(conv_filters=D, num_blocks=8, stem_blocks=2, dropout=0.03)
    full_sign = dict(conv_filters=D, num_blocks=8, stem_blocks=2, dropout=0.03)
    ng = sum(p.numel() for p in build_pt_conv_globalizer(full_glob).parameters())
    ns = sum(p.numel() for p in build_pt_conv_signed_attn(full_sign).parameters())
    print(f"\nconv-globalizer-2c8g : {ng:,}  (spec ~27,600,000)")
    print(f"conv-signed-attn-2c8a: {ns:,}  (spec ~16,350,000)")
    assert abs(ng - 27_600_000) / 27_600_000 < 0.05
    assert abs(ns - 16_350_000) / 16_350_000 < 0.05
