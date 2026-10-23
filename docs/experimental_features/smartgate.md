# SmartGate

SmartGate is a suppress-only policy gate. Xerces scores moves with a relational from/to head: two attention towers whose dot products score every source-destination pair. That head is good at geometry (which moves are plausible given piece placement), but it has no cheap way to say "this specific move is wrong *here*", whether that means illegal in this exact position or a known bad idea. SmartGate is a second, parallel pathway whose only job is to shut moves the from/to head over-weights. It can never invent a move the from/to head didn't already like.

It is live in the active m137_8c8t model. See the [neural network page](../architecture/neural_network.md) for where it sits in the full architecture.

## Its Own Accumulators

The trunk carries 8 global accumulator tokens appended to the 64 board squares. SmartGate reads two of them concatenated: a private token (67) that nothing else reads, and a token (68) it shares with both policy towers. The private register lets it learn a representation tuned for vetoing moves. The shared one ties it to what the policy head is attending to.

## The Gate

The 512-d accumulator pair goes through a GELU feed-forward with a residual and norm, then a linear layer to one **shut-logit** per move over the 1858 "sometimes-legal" move domain:

```python
gi       = cat(x[:, 67], x[:, 68])                     # 2 accumulators, 512-d
g        = gate_norm(gi + gate_ff2(gelu(gate_ff1(gi))))
gate_raw = gate_out(g)                                 # [B, 1858] shut-logits
policy   = move_logits - scale * sigmoid(gate_raw)
```

Because `scale * sigmoid(.) >= 0`, the gate can only ever *subtract* from a move's logit. It has no additive branch: a move the from/to head scores low stays low, and the gate cannot rescue it.

Two learnable knobs bound how hard it can push. Both are clamped after every optimizer step:

- **Per-move bias**, held in `[0, 5]`: each of the 1858 slots has its own baseline openness.
- **Global scale**, held in `[0, 12]`: the maximum logit penalty the gate can apply.

## Weak at Init, Earns Its Scale

The per-move bias starts at 4.0 (`sigmoid(4) ≈ 0.98`, mostly shut) and the scale starts at 1.0. At init the gate therefore subtracts a near-uniform ~1 from every move, which softmax ignores, and the from/to head's priors pass through effectively untouched. The scale only grows as the gate learns to tell moves apart, so it can't destabilize a policy head that hasn't learned anything yet.

## Supervised on Legality

SmartGate isn't left to discover its job from the policy loss alone. It gets its own objective: a binary cross-entropy on `gate_raw` predicting whether each of the 1858 moves is **illegal** in the current position. Any move the policy target gives visits to counts as legal, since every legal move carries a one-visit floor. Legal moves are up-weighted so the rare "keep open" class isn't drowned out. The gate loss is held at a small fixed share of the main policy + value loss, and its share shrinks once it is well trained, so a converged gate stops competing with the main objective.

The result is a learned, position-specific legality mask that keeps policy mass off impossible moves. On top of that, the policy loss still flows through the gate, so it remains free to shut legal moves the from/to head systematically over-rates.

## Why Suppress-Only

An additive gate could quietly become a second, redundant policy head, competing with the from/to towers and muddying which pathway is responsible for a prediction. Constraining it to subtraction forces a clean division of labor: the from/to head proposes, the gate disposes. It also matches the failure modes it targets. Illegal moves and confidently wrong moves both call for suppression; neither needs a second voice proposing new moves.
