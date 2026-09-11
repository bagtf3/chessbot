# Int8-Native Chess Model — Design Plan

## Vision

A model that is int8 everywhere except softmax. Not post-hoc quantized — *built* int8.
Training happens in fp16/32 with quantization constraints baked in so TRT conversion is
just fusing passes on an already-integer-ready graph. No calibration. No surprises.
Residual stream: int8 throughout.

---

## POC Spec

| Field | Value |
|---|---|
| Architecture | 2 conv blocks + 2 transformer blocks |
| Heads | policy, value, smartgate |
| Input encoding | xc0hK6 (64 tokens, K=6 history plies) |
| d_model | 256 |
| Parameters | ~2-4M (POC testpiece) |
| Training dtype | fp16/bf16 with fp32 accumulation |
| Inference dtype | int8 everywhere, fp32 softmax only |

---

## Quantization Scheme

### Weights: `W = s * Q`

- `W` — normal fp32/fp16 learned weight during training
- `s` — fp32 per-channel scale, shape `[out_channels]` (e.g. `[256]` for d_model=256 layers)
- `Q` — int8 weight extracted at export: `Q = clamp(round(W / s.unsqueeze(1)), -127, 127)`

During training `W` and `s` are just two learned parameters. No fake quantization, no
rounding. The sin²(πW/s) penalty steers `W/s` toward integers so that by export time
`round(W/s)` is nearly lossless.

Per-channel `s` means each output neuron gets its own scale tuned to its own weight
distribution — this is the standard for weight quantization in TRT and gives much better
accuracy than per-tensor.

### Optional QAT Phase

After chess loss + range bangers + sin² have settled into a stable pattern, optionally
introduce fake quantization:

```python
Q_fake = clamp(round(W / s.unsqueeze(1)), -127, 127)
W_eff  = Q_fake * s.unsqueeze(1)   # forward uses quantized weights
# backward: STE — gradient passes through round as identity
```

This is added late, not from the start. By then `W/s` is already near integers so the
rounding noise is negligible and the model barely notices. Standard QAT introduces this
early to force the model to compensate for rounding noise — our approach eliminates the
noise first via sin², making QAT optional verification rather than a load-bearing step.

### Scale Regularization

`s` must stay positive and nonzero:

```python
s = F.softplus(s_raw) + eps
```

---

## Matmul: int8 × int8 → int32

CUDA tensor cores do `int8 x int8 -> int32` accumulation natively. After accumulation:

```
int32_out → requant_scale → activation → clamp(-127, 127) → round → int8
```

The activation (LeakyReLU during training, hard clamp at inference) is **folded into
the requant step** — no separate op. Just: scale, clamp, round, write int8.

---

## Activation Strategy

- **Training**: `LeakyReLU(alpha=0.01)` — keeps gradients alive through the clamp region.
  Bounded by relu range bangers at tau=100 from epoch 0.
- **Inference / export**: swap to `clamp(0, 127)`, folded into requant. Zero cost.

---

## L1 Norm

Standard RMSNorm uses `x / sqrt(mean(x²))` — pow(2) is the fp16/int8 killer.

Replace with:

```python
def l1_norm(x, scale, eps=1e-6):
    xf = x.float()                              # fp32 for the mean only
    norm = xf.abs().mean(-1, keepdim=True).add(eps)
    y = (xf / norm) * scale                     # scale: [d_model], broadcasts
    return clamp_round(y, -127, 127).to(torch.int8)
```

`scale` is shape `[d_model]` (e.g. `[256]`), learned fp32. It does three jobs at once:
1. **Quantizer** — maps unit-L1 output into int8 range
2. **Bias absorber** — absorbs any systematic residual bias
3. **Residual rescaler** — keeps the int8 residual stream properly scaled after each add

The norm itself (`abs().mean()`) stays fp32 — two ops, not a bottleneck. Everything
else is integer.

---

## Residual Stream

```
int8_in
  → int8 matmul → int32 → requant (scale + leakyrelu + clamp + round) → int8
  → int8 residual add (int16 accumulate, clamp back to int8) → int8
  → L1 norm (fp32 mean only) → mul(scale[256]) → clamp_round → int8
  → repeat
```

Residual add overflows int8 range if both operands are large. Two options:
- Accumulate as int16, clamp back: `clamp(a.int16() + b.int16(), -127, 127).to(int8)`
- Let the following L1 norm absorb it — it renormalizes anyway

Either way, no fp32 spill in the residual stream.

---

## Training Phases

### Phase 1 — Chess loss + range bangers (from epoch 0)

Normal pretraining. `W` and `s` learn freely. Range bangers keep activations inside
int8 window from the start. sin² penalty weight = 0 (wired, off).

```python
penalty = lam * relu(|x| - tau).sum()   # tau=100, lam ramps from 1e-7
```

Applied at every L1 norm input site.

### Phase 2 — Blend in sin² slowly

Once chess loss has converged and activations are stable, ramp `mu` from 0:

```python
penalty += mu * sin(pi * W / s.unsqueeze(1)).pow(2).sum()
```

Weights drift toward integers in Q-space. Range bangers keep them from going wild.
By convergence `round(W/s)` is essentially lossless.

### Phase 3 — Optional QAT

Once chess + range bangers + sin² have a stable equilibrium, optionally add fake
quantization (see QAT phase above). Verifies that rounding is truly lossless before
export. Skip if validation shows no accuracy gap between W and round(W/s)*s.

### Phase 4 — Export

```python
Q = clamp(round(W / s.unsqueeze(1)), -127, 127).to(torch.int8)
# rebuild model with Q (int8) + s (fp32) instead of W
# emit Q/DQ nodes, run TRT fusing passes
```

---

## Softmax — The Only fp32 Island

Policy head softmax stays fp32. One justified island, TRT handles it cleanly.

---

## TRT Export

The graph TRT sees:

- Q/DQ nodes around every weight (dequant for matmul input, requant after output)
- L1 norm: `abs → mean → div → mul(scale[256]) → clamp_round` — scale fuses into preceding Q/DQ
- Residual adds as int8/int16
- Softmax in fp32, isolated

No calibration. Scales are model weights — they were learned. TRT just fuses.

---

## Build Sequence

1. Define `Int8Linear(in, out)` — `W` + per-channel `s[out]`, optional Q_fake
2. Define `L1Norm(d_model)` — `abs().mean()` fp32, learned `scale[d_model]`, `clamp_round`
3. Build 2c2t model using these primitives
4. Training loop: chess loss + range bangers from epoch 0
5. Blend in sin² penalty once chess has settled
6. Validate: accuracy vs fp32 baseline at each checkpoint
7. Optional: add QAT once sin² equilibrium is stable
8. Export: freeze Q, emit Q/DQ nodes, run TRT fusing passes
9. Benchmark: latency vs fp16 TRT baseline

---

## Why This Works

Standard QAT introduces rounding noise early and forces the model to compensate.
This approach eliminates the source of the problem first — sin² steers weights to the
integer grid, range bangers keep activations in range — so by export time quantization
is nearly free. The model was built to be int8. It doesn't need to be approximated.

The result: `int8` is not an approximation of the fp32 model. It is the model.
