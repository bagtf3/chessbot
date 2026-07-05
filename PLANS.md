# end_of_round Sprint Plans

## 1. Drain training buffer at end of round with one large retrain [DONE]

At the end of a round, instead of leaving leftover games in the training
buffer queue for the next round's normal-sized retrains, do one large retrain
that consumes the largest integer multiple of `retrain_size` that fits in the
current buffer, sampled randomly from the buffer.

Example: `retrain_size=10240`, buffer has 77000 games -> end-of-round retrain
uses `floor(77000 / 10240) * 10240 = 71680` games, chosen randomly from the
77000.

All other (non-end-of-round) retrains keep the current approach unchanged.

## 2. Update bootstrap data pipeline for 1858 policy output

Update the `build_bootstrap_data` suite of scripts, `shuffle_xco`,
`lco_to_xco`, and the other tools inside `/repos/xerces_training`, along with
`bootstrap_model_async_tfrec_pt` in chessbot, so the whole bootstrap data
pipeline works with the new 1858-slot policy output instead of the old
4288-flat/legacy target shape.

This spans two repos:
- `chessbot/scripts/train/bootstrap_model_async_tfrec_pt.py` (and its sibling
  `bootstrap_model_async_tfrec.py`) — consumer side, needs to read/feed
  1858-dim policy targets into the PT model instead of whatever shape it
  currently expects.
- `xerces_training` (`chunkparser.py`, `parse_utils.py`, `uci_to_idx.py`,
  `shufflebuffer.py`, and the `shuffle_xco` / `lco_to_xco` conversion tools) —
  producer side, needs the LC0 chunk -> Xerces record conversion and shuffle
  pipeline to emit/consume 1858-dim policy consistently end to end.

Audit each script for hardcoded policy dims (4288, legacy flat indices, etc.)
and reconcile them against the 1858 legal-move domain used by
`build_pt_precond_smartgate`.

## 3. Factored / semantic token embeddings (structured but learnable)

Replace the plain learnable token `Embedding` with a hand-designed **factored
embedding** whose dimensions are meaningful bit-fields, so related tokens share
structure by construction instead of the net discovering it from scratch. Init
with the structured codes to inject the inductive bias, but keep it **learnable**
so the model is free to refine/warp the geometry (not frozen).

Idea: each token's vector is a concatenation of semantic bits, e.g.
`[is_pawn, is_knight, is_bishop, is_rook, is_queen, is_king, is_opp_side,
castle_K, castle_Q, is_ep, ...]`. Then STM-pawn (token 1) and OPP-pawn (token 10)
share the whole piece subvector and differ only in `is_opp_side` — the existing
`+9` STM/OPP offset becomes an explicit axis, and piece-type similarity is baked
in rather than learned. The conv stem gets a factored, semantically-aligned input.

Design notes:
- Implement as a learnable `nn.Embedding` whose `.weight` is *initialized* to the
  factored code matrix (not `freeze=True`). Optionally freeze early / unfreeze
  later, or train at reduced LR, so the bias survives early training.
- Plain one-hot buys nothing here (a trainable first layer absorbs it -> identical
  to a learned embedding). The value is entirely in the *shared-bit* structure.
- Dovetails with the compact history-token encoder (see below): the same bit-field
  scheme can pack piece/side/castle/EP/repetition/hmc into one small fixed-width
  code, so history frames tokenize cheaply with structure intact.
- Fixed geometry can be wrong and (if frozen) uncorrectable -> init-then-learn is
  the hedge.
- Does NOT address inference throughput (gather cost is identical); this is an
  accuracy / sample-efficiency / inductive-bias play, not a speed one.

Related: lightweight token-embedded history encoder (~384 tokens = K frames x 64,
channel-concat per square + rep/hmc meta tokens) to recover lc0's history/rep/
stm/hmc info at ~xc0 cost. Let the bakeoff arms (curr-only vs 4hist vs lc0,
nohmc/norep/nostm) define the minimal sufficient token budget before building.
