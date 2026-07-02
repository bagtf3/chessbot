# SmartGate

SmartGate is a suppress-only policy gate. Xerces computes its move priors with a relational from/to head — two attention towers whose dot products score every source-destination pair. That head is good at geometry (which moves are plausible given piece placement) but it has no cheap way to say "this specific move is a blunder *here*." SmartGate is a second, parallel policy pathway whose only job is to veto moves the from/to head over-weights, without ever being able to invent a move the from/to head didn't already like.

It is live in the active `16m-precond-smartgate` model, but the mechanism is novel enough to be worth calling out on its own.

## Its Own Accumulator

The trunk carries 8 global accumulator tokens appended to the 64 board squares. SmartGate reads from a single dedicated token (accumulator 1), separate from the WDL token and the from/to policy tokens. This gives the gate a private read-out register: it can attend over the whole board and learn a representation tuned specifically for spotting mistakes, rather than sharing capacity with the value or policy heads.

## The Gate

The accumulator vector is refined by a SwiGLU-style gated feed-forward, then projected to one logit per move over the 1858 "sometimes-legal" move domain:

```python
gi       = x[:, 65, :]                                  # gate accumulator token
h        = F.silu(gate_w_gate(gi)) * gate_w_up(gi)      # SwiGLU
g        = gi + gate_drop(gate_w_down(h))               # residual
gate_raw = gate_out(gate_norm(g))                       # [B, 1858]
```

The raw from/to logits (the flat 4288-slot move space) are indexed down to the same 1858 legal slots, then combined:

```python
combined = q_sl + F.logsigmoid(gate_raw)
```

Because `logsigmoid(x) <= 0` for all `x`, the gate can only ever *subtract* from a move's logit. It has no additive branch. A move that the from/to head scores low stays low; the gate cannot rescue it. All the gate can do is pull down a move the from/to head liked too much.

## Near No-Op at Init

`gate_out` initializes with zero weights and a bias of **4.0**. Since `logsigmoid(4) ≈ -0.018`, every gate starts at essentially zero suppression, so the model begins training as if the gate weren't there and the from/to head's priors pass through untouched. The gate only earns its suppression as training discovers which moves are systematically mis-scored. This keeps early training stable — the gate can't destabilize a head that hasn't learned anything yet — and makes the gate's contribution interpretable: any deviation from 0 is a learned veto.

## Why Suppress-Only

An additive gate could quietly become a second, redundant policy head, competing with the from/to towers and muddying which pathway is responsible for a prediction. Constraining it to subtraction forces a clean division of labor: the from/to head proposes, the gate disposes. It also matches the actual failure mode it targets — the from/to head's errors are far more often "confidently plays a move that loses material here" than "fails to consider a good move at all," and suppression is the direct remedy for the former.
