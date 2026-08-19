# Neural Network

## Board Encoding

The base building block is a single position encoded as 64 integer tokens, one per square, always from the perspective of the side to move (STM-as-white). Black positions are flipped via `sq ^ 56` before encoding so the model always sees its own pieces as the "white" side.

Single-frame token values:

| Range | Meaning |
|---|---|
| 0 | Empty square |
| 1–5 | STM pieces: pawn, knight, bishop, rook, queen |
| 6–9 | STM king with castling rights: none, kingside, queenside, both |
| 10–18 | Opponent pieces (same structure, +9 offset) |
| 19 | En passant target square |

Castling rights are folded into the king token in this single-frame encoding — no separate planes are needed. The en passant square gets its own token on the target square, giving the model direct spatial context about which file is eligible. Vocabulary size is 21. This encoding is defined in C++ (pyfastchess) and is fixed across the entire stack.

The **primary model encoding** (`xc0h`) stacks 6 of these frames — the current position plus 5 prior positions, oldest padded if the game is younger than that — instead of just the current one. Each frame uses a simplified per-square token (pieces 1-6/7-12 by ownership, no castling folded in, EP token, pad token for missing history), all frozen to the same STM-relative orientation so frames stay spatially aligned across time. The stack is followed by a per-frame repetition flag and 3 metadata scalars (combined castling rights, side to move, halfmove clock), for a flat length of `K*64 + K + 3` — 393 tokens at `history_K=6`. This was adopted as the primary architecture after testing showed it outperforming the single-frame, no-history version (see the FAQ: "Do you use history planes?").

## Preconditioner-SmartGate Architecture

The active model family is `precond-smartgate` (`build_pt_precond_smartgate` in `src/chessbot/model.py`), currently run against the `xc0h` history encoding above. It is a PyTorch model trained in float16. The design pairs a convolutional preconditioner with a transformer trunk, and attaches every output head to a dedicated global accumulator token rather than pooling over the board.

**Input and embedding.** The token sequence goes through a token embedding (`vocab_size=21 → conv_filters=256`) and is reshaped to an 8×8 spatial grid (the history/metadata tail is handled separately from the per-square grid).

**Convolutional preconditioner.** 4 residual conv blocks (`pre_blocks=4`) run over the 8×8 grid. Each block is `LayerNorm → Conv2D(3×3) → LeakyReLU → Conv2D(3×3) → LeakyReLU`, added back as a residual (the first block skips the prenorm). This lets local spatial structure resolve before any attention runs.

**Positional concat.** The preconditioned grid is flattened to a (64, 256) sequence and LayerNorm'd, then a learned positional embedding (`64 → 256`) is **concatenated** (not added), producing a 512-d (`D = conv_filters * 2`) representation per square. Positional information rides in its own channels instead of being summed into the content.

**Global accumulator tokens.** 8 learned global tokens are appended to the sequence, giving `[B, 72, D]`. These are the model's read-out registers: after the trunk, each output head reads from its own token instead of pooling the 64 board squares.

**Transformer trunk.** 4 prenorm transformer blocks (`num_heads=8`) with GELU feed-forwards of width `[768, 1024, 1024, 768]`. Each block is `LayerNorm → MHA → residual`, then `LayerNorm → Linear → GELU → Linear → residual`. A shared `trunk_ln` LayerNorm closes the trunk. No BatchNorm anywhere — small, variable selfplay batches make it unstable.

**Value head (accumulator 0).** `Linear(D → D/2) → GELU → Linear(D/2 → 3)` reads token 64, producing raw WDL logits. Trained as a 3-class [win, draw, loss] softmax, STM-POV.

**Policy head — from/to attention (accumulators 2–7).** Two projection towers (`Linear(D → 256)`) build a 256-d vector per square, each refined by its own 4-head self-attention (`from_mha`, `to_mha`) that also attends over policy-specific accumulator tokens. Normal-move logits are the scaled dot products `(from · to^T) / sqrt(256)` → a (64, 64) matrix flattened to 4096. Underpromotion logits (192, from-file × to-file × piece) come from small `promo_from`/`promo_to` projections added to the relevant dot-product sub-block. Concatenated, these give the 4288-slot dense move space.

**SmartGate (accumulator 1) and the 1858 domain.** In parallel, a gated FFN (`SiLU` gate × up-projection → down-projection residual → RMSNorm → `Linear(D → 1858)`) reads token 65 and produces one gate logit per **legal-in-some-position** move. The raw 4288 logits are indexed down to the 1858 "sometimes-legal" move slots (`build_sometimes_legal_mask`), and the gate is applied as `logits + logsigmoid(gate)`. Because `logsigmoid ≤ 0`, the gate is **suppress-only** — it can dampen a move but never invent one. Its output bias initializes to 4.0 (`logsigmoid(4) ≈ 0`), so at init the gate is a near no-op and only learns to veto moves the from/to head over-weights. The model's policy output is these 1858 legal logits; C++ scatters them back into the 4288 flat space.

The value used elsewhere (`nn_value`) is derived in C++ as `win - loss`. Policy post-processing — softmax over legal moves, `uniform_eps` blending, `prior_clip_max` clamping — is handled in C++ (`build_priors`) after raw logits are inserted into the raw cache.

## ONNX Runtime + TensorRT Inference

Inference runs through ONNX Runtime with the TensorRT execution provider (`src/chessbot/infer_ort_trt.py`), the `ort_trt` backend. TensorFlow has been removed from the active path (old Keras/TF architectures are quarantined in `legacy_models.py`).

After each retrain, `prepare_trt()` exports the current TorchScript model to ONNX (`export_ts_to_onnx`, input `enc_in: [B, 64]` int64) and stages it in the run's TRT cache. Workers then call `make_ort_trt_infer()`, which builds an ORT `InferenceSession` with:

- `TensorrtExecutionProvider` — fp16 enabled, 4 GB workspace, fixed shape profile (`enc_in:1x64` min, `{max_bs}x64` opt/max). Compiled engines are cached to disk keyed by the sha256[:12] of the ONNX bytes, so subsequent sessions load the engine instead of paying the multi-minute compile. TRT is required, not a fallback — the session raises if it doesn't come up on TRT.

`infer(pair)` casts the board batch to int64, runs the session for `policy_logits` and `value_out`, and softmaxes the WDL logits on the CPU, returning `(logits [B, 1858], wdl [B, 3])`. `pt_eager` (raw PyTorch) remains available for debugging and as the retrain backend; both expose the same `(logits, wdl)` interface, so the selfplay loop is inference-backend-agnostic.

A third backend, `lc0_trt`, swaps in a distilled Leela network over the same TRT machinery — see the [rescoring pipeline](evaluation_pipeline.md) for how LC0 is used as a distillation teacher.
