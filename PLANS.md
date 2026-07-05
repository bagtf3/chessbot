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

## 3. Compositional (move-geometry) embedding init -- init only, learnable

Keep the plain integer token vocab and a plain learnable `nn.Embedding` at
encode/embed time -- tokens stay `us_pawn`, `them_pawn`, `empty`, `us_queen`,
`ep`, etc. The idea touches **only the initialization** of `.weight`: seed it so
related tokens already share structure, then train normally so the model can
warp/refine or discard it. If the prior is trash the net just moves away from it;
a learnable first layer means it cannot hurt asymptotically.

The clever bit: init from a small set of **movement-primitive** basis vectors and
compose each token as a *sum* of the primitives that describe how it moves --
not just categorical one-hots. This bakes in real chess move geometry.

Basis (random iid vectors, one per primitive):
`EMPTY, PAWN, KNIGHT, DIAG (diagonal slider), ORTHO (up/down/left/right slider),
KING, EP, US, THEM`.

Compose token inits as sums:
```
empty      = EMPTY
us_pawn    = US   + PAWN
them_pawn  = THEM + PAWN
us_knight  = US   + KNIGHT
us_bishop  = US   + DIAG
us_rook    = US   + ORTHO
them_queen = THEM + DIAG + ORTHO      # queen = union of bishop + rook rays
us_king    = US   + KING
ep         = EMPTY + EP               # ep square is an empty square, flavored
```

Why this beats the earlier one-hot bit-field sketch: `queen = DIAG + ORTHO` is
literally true about attack geometry, so a stem filter that learns "diagonal ray
influence" fires for bishop AND queen for free (they share the DIAG component),
and the orthogonal-ray filter fires for rook AND queen. The `US`/`THEM` split
becomes an explicit linear ownership axis, so ownership-symmetric patterns
(attack/defend) get a direction to exploit from step 1. One-hot categories throw
all of this away.

Formulation: it's a factor model `E_init = C @ B`, where `B` is the primitive
basis (n_primitives x d) and `C` is a hand-written token x primitive incidence
matrix. Compute `E_init`, drop into `nn.Embedding.weight`, train.

Design notes:
- Keep it **learnable** (not frozen). Optionally reduced-LR or freeze-early /
  unfreeze-later so the bias survives the first steps.
- Normalize the composed rows: summing k iid primitives scales variance ~k, so
  `them_queen` (3) would start with a bigger norm than `us_pawn` (2) than `empty`
  (1). Either scale the basis small (sigma ~ 1/sqrt(max_k)) or L2-normalize each
  row of `E_init` so no token dominates at init.
- Add small per-token jitter `E_init = C @ B + eps * randn` so each token has a
  sliver of private capacity to specialize away from its shared structure.
- Zero runtime cost: encode still emits the same integer tokens, embed is the
  same gather. This is an accuracy / sample-efficiency / inductive-bias play,
  NOT a speed one; benefit is largest early, in low-d history frames, and when
  data-limited, and likely narrows toward convergence.
- Validate as a bake-off arm against plain-random init, identical everything
  else -- want the training curve, not a vibe.

## 4. Compact token-based history encoder (lc0 richness at ~xc0 cost)

Goal: keep xc0's lightweight "pass a few ints" transport, but carry lc0's
history / repetition / hmc / stm signal. Everything is folded into a per-square
channel stack at the **stem only**; the trunk is untouched and still enters at
`(B, CF, 8, 8)` exactly like the current `build_pt_precond_smartgate` stem
(mirror the existing `lc0_input` if/else). Selected by a new `encoding_type`
value (e.g. `"txc0"`).

### Token vocab (slim, ~14 types)
`empty, us_{P,N,B,R,Q,K}, them_{P,N,B,R,Q,K}, ep` -- bare king (no castling in the
token), EP kept as an in-band token on its square. Embedding width is a free
hyperparameter (decoupled from vocab); plan `d_embed = 18`. Init with the
compositional scheme from section 3.

### Frames (current + history)
K frames total (current + K-1 history), each a full `(64,)` xc0-style token plane
-> `Embedding(vocab, 18)` -> 18 channels per square. Plan K=6 -> `6 x 18 = 108`
channels. Leave history frames as full xc0 tokens (not stripped): EP in history is
genuinely informative (a double-push just happened), and the redundant bits are
cheap for the projection to ignore.

Canonicalization (critical): fix orientation and ownership to the **current** STM
across ALL frames -- a single rank flip `sq ^ 56` (rank-only, never a file flip or
180 rotation, which would swap king/queen-side and corrupt castling), and `us`/
`them` frozen by current STM. Do NOT recompute STM per frame. In C++, mirror the
`unmake()` walk in `board_to_lc0_features` but emit int16 tokens per frame instead
of bitboard planes.

### Meta channels (stacked onto the end -- order doesn't matter, the stem
projection fully mixes them)
- **Castling (16-dim, reshaped):** all 16 us/them x {none,K,Q,both} states are
  common in normal play, so encode the joint state as a single 16-way token ->
  `Embedding(16, 64)` -> reshape `(8, 8)` -> concat as **1 learned plane**. A
  single 16-way embedding captures the *joint* us/them state (vs 4 independent
  binary planes = only marginals), and the learned 8x8 can concentrate signal on
  the actual king/rook squares -- recovering the spatial co-location lost by
  going to a bare king token. Compute the 16-state index in us/them terms so it
  stays aligned with the flipped board. Current frame only.
- **STM:** 1 broadcast plane (0 white / 1 black). Tokens are us/them-relative so
  the net can't otherwise tell true color; this restores the intrinsic
  white-to-move bias.
- **hmc:** 1 broadcast plane, `halfmove_clock / 99.0` (rule50-style).
- **Repetition:** 1 scalar channel **per frame** (K total). Value 0/1/2, kept as
  an ordinal scalar (not a 3-way embedding) so "closer to a draw" is monotone for
  free. Semantics = live 3-fold count: occurrences of that frame's exact position
  **within the current irreversible block** (since the last pawn move or capture),
  capped at 2; any frame predating the last irreversible move is dead -> 0. hmc is
  exactly "plies since last irreversible move," so it is the block boundary --
  that's why "hmc reset since then -> 0" is exact, not a heuristic. From-now
  semantics (evaluate every frame against the current block) is the intended
  version; confirm vs at-the-time before building. Cheap to compute in the same
  `unmake()` walk.

### Channel budget (K=6, d_embed=18)
`6*18 (pieces) + 6 (per-frame rep) + 1 (castling) + 1 (stm) + 1 (hmc) = 117`
channels -> `(B, 117, 8, 8)` -> project (1x1 conv / Linear) to `CF=256` -> trunk
unchanged. Lands right next to lc0's 112 with no one-hot bloat (~652 bytes/pos vs
lc0's ~7 KB).

### Combine method
Concat-then-project (Scheme A): embed each frame small, concat all frames + meta
channels, one projection to CF. Faithful analog of lc0's 1x1 conv over stacked
planes, but over compact learned embeddings. Additive-sum (Scheme B) is a lighter
ablation arm.

### Plumbing checklist
- pyfastchess: `board_to_history_tokens(b, K)` reusing the `unmake()` walk
  (`backend.cpp` ~942-956), emit int16 tokens + per-frame rep; binding + a
  `MCTSForest` batch variant paralleling `get_all_lc0_features`.
- `config.py`: add the `encoding_type` value; dispatch in `rescore.py::encode_board`
  and `looper.py`.
- `infer_ort_trt.py`: `ENC_SHAPE` + `enc_dummy` branch for the new fixed int shape.
- `model.py`: third stem branch in `build_pt_precond_smartgate` behind a cfg flag;
  trunk untouched.
- Retrain int path already keys on `encoding_type`; fixed length keeps `np.stack`
  happy as long as all samples share K.

Let the existing bake-off arms (curr-only vs 4hist vs lc0; nohmc / norep / nostm)
define the minimal sufficient token budget (K, which meta channels) before
committing.
