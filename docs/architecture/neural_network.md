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

## m137_8c8t Architecture

The active model is **m137_8c8t** (registry key `m13-7-8c8t-d256`, built by `build_pt_expandy_swiglu` in `src/chessbot/model.py`): 8 conv blocks followed by 8 transformer blocks at width `D = 256`, ~18.4M parameters, reading the `xc0h` history encoding above. It is a PyTorch model trained under fp16 autocast. LayerNorm is used throughout (an experimental L1.5 norm is available via `cfg["norm"]`); there is no BatchNorm, since small, variable selfplay batches make it unstable.

**xc0h stem.** Each of the 6 history frames is embedded per square with an 18-d token embedding (15-token vocab). The embedding table is initialized compositionally rather than randomly: every piece row is `side + movement primitive` (`us`/`them` plus `pawn`, `knight`, `diagonal`, `orthogonal`, `king`), so a queen starts as `diagonal + orthogonal`, and the EP token starts as `empty + them`. The 6 frames are concatenated per square (108 channels) alongside the per-frame repetition flags, a castling-rights embedding (16 states → one learned value per square), side to move and the halfmove clock, each scalar with its own learnable scale. A linear projection + LayerNorm maps this to `D - 4 = 252` channels, and 4 fixed spatial planes (ones, checkerboard, rank, file) are appended **after** the norm so it cannot flatten their geometry. Each plane has a learnable scale so the convs can mute any that aren't useful. The result is a `[B, 256, 8, 8]` grid.

**Conv stack (8 blocks).** All conv blocks are prenorm residual blocks over the 8×8 grid, so local spatial structure resolves before any attention runs.

- **2 plain blocks**: `LN → Conv3×3 → act → Conv3×3 → act`, added back to the input.
- **6 gated expandy blocks**: each half of the block is a depthwise-separable-style factorization: a **grouped 3×3 conv (4 groups) that expands `D → 2D`**, then a **1×1 pointwise conv back down to `D`**. This buys a wide 512-channel intermediate at a fraction of a dense 3×3's cost. Between the two halves, a 1×1 conv + sigmoid produces a **per-square, per-channel gate** that multiplies the block's output before the residual add, so each block can choose, square by square, how much it writes into the stream.

**Learnable PReLUs.** Every conv activation is a per-channel `PReLU`. Slopes initialize at 0.03, which makes them exactly the `LeakyReLU(0.03)` they replaced, so grafting onto an older leaky checkpoint is lossless. They are clamped to `[-0.5, 0.5]` after every optimizer step (`project_params`), which keeps any channel from drifting into a linear or sign-flipping regime.

**Positions and global accumulators.** The grid is flattened to 64 square tokens and a learned positional embedding is **added**, scaled by a learnable `pos_scale`. 8 learned global tokens are then appended, giving a `[B, 72, 256]` sequence. These are the model's read-out registers: after the trunk, every output head reads from dedicated accumulator tokens instead of pooling the board.

| Token | Read by |
|---|---|
| 64, 65, 66 | Win, draw and loss heads |
| 67 | SmartGate |
| 68 | SmartGate and both policy towers (shared) |
| 69 | Both policy towers |
| 70 | From tower only |
| 71 | To tower only |

**Transformer trunk (8 blocks).** All blocks are prenorm: `LN → MHA (8 heads) → residual`, then `LN → FFN → residual`. The K projection has no bias, since softmax is shift-invariant per query and a K bias would be dead weight.

- **Blocks 1–4 (FF)**: GELU feed-forward, `256 → 1024 → 256`.
- **Blocks 5–8 (SwiGLU)**: `down(SiLU(gate(x)) * up(x))` with a 1024-wide hidden layer, plus **tanh-gated attention**. The attention output is multiplied by `1 + tanh(g(x))`, where `g` is a per-token, per-channel linear map. The gate ranges over `(0, 2)`, so it can both suppress and **boost** what each token takes from attention. `g` is zero-initialized, so the gate is exactly 1 at init and the block starts as plain attention.

Dropout in the trunk is stochastic per site: each attention/FFN dropout site is switched on independently with a set duty probability (`set_dropout`) rather than always running. A final LayerNorm closes the trunk.

**Value head (tokens 64–66).** Win, draw and loss each get an independent head on their own accumulator: `Linear(256 → 64) → GELU → Linear(64 → 1)`. The three outputs form the WDL logits, trained as a 3-class softmax, STM-POV.

**Policy head: from/to attention.** Two towers build a 256-d "from" and "to" vector per square. Each tower takes the 64 squares plus its accumulators (68, 69, and 70 or 71), applies a GELU projection residual, then runs a 4-head attention in which the squares query over the squares **and** the accumulators, followed by an output residual. Move logits are the dot products of the normalized from/to vectors, scaled by `1/sqrt(256)`, giving a 64×64 from-to matrix. Queen, rook and bishop promotions get their own logits: the rank-7 → rank-8 sub-block of that matrix plus small per-piece `promo_from`/`promo_to` projections. A knight promotion uses the plain from → to slot. Only the 1858 **sometimes-legal** move slots (`build_sometimes_legal_mask`) are emitted; C++ scatters them back into the 4288 flat space.

**SmartGate: a learned legality veto.** In parallel, the gate reads tokens 67 and 68 concatenated (512-d) through a GELU FFN (`512 → 1024 → 512`) with a residual and LayerNorm, then a linear layer to **one shut-logit per move slot** (1858). The policy output is

```
policy = move_logits - scale * sigmoid(gate_raw)
```

Because the subtracted term is always ≥ 0, the gate is **suppress-only**: it can push a move down but never invent one. Both knobs are learnable and clamped after every step. The per-move bias lives in `[0, 5]` and starts at 4.0 (`sigmoid(4) ≈ 0.98`, mostly shut). The global `scale` lives in `[0, 12]` and starts at 1.0. At init the gate therefore applies a near-uniform shift that softmax ignores, and it earns its scale as it learns legality.

**Gate loss.** The gate has its own supervised objective: a BCE on `gate_raw` predicting **illegality**, with any move the policy target gives visits to counted as legal (every legal move carries a 1-visit floor). Legal moves are up-weighted 5× and the weights are renormalized per position. The gate loss is held at **5% of the main (policy + value) loss**. Once it falls below 5.0 its share shrinks proportionally, so a well-trained gate stops competing with the main objective.

**Diagnostics.** In training, the model can track hotspot statistics (peak activations and where they occur) at the policy dot products, the FFN and SwiGLU down-projections, and the residual stream. It also keeps 500-step EMAs of gate mean, spread and range for both the conv gates and the attention gates. These feed the gate-health readout in the retrain logs and add no host syncs outside print cadence.

The value used elsewhere (`nn_value`) is derived in C++ as `win - loss`. Policy post-processing — softmax over legal moves, `uniform_eps` blending, `prior_clip_max` clamping — is handled in C++ (`build_priors`) after raw logits are inserted into the raw cache.

## ONNX Runtime + TensorRT Inference

Inference runs through ONNX Runtime with the TensorRT execution provider (`src/chessbot/infer_ort_trt.py`), the `ort_trt` backend. TensorFlow has been removed from the active path (old Keras/TF architectures are quarantined in `legacy_models.py`).

After each retrain, `prepare_trt()` exports the current TorchScript model to ONNX (`export_ts_to_onnx`, input `enc_in: [B, 393]` int64 for xc0h) and stages it in the run's TRT cache. Workers then call `make_ort_trt_infer()`, which builds an ORT `InferenceSession` with:

- `TensorrtExecutionProvider` — fp16 enabled, 4 GB workspace, fixed shape profile (`enc_in:1x393` min, `{max_bs}x393` opt/max). Compiled engines are cached to disk keyed by the sha256[:12] of the ONNX bytes, so subsequent sessions load the engine instead of paying the multi-minute compile. TRT is required, not a fallback — the session raises if it doesn't come up on TRT.

`infer(pair)` casts the board batch to int64, runs the session for `policy_logits` and `value_out`, and softmaxes the WDL logits on the CPU, returning `(logits [B, 1858], wdl [B, 3])`. `pt_eager` (raw PyTorch) remains available for debugging and as the retrain backend; both expose the same `(logits, wdl)` interface, so the selfplay loop is inference-backend-agnostic.

A third backend, `lc0_trt`, swaps in a distilled Leela network over the same TRT machinery — see the [rescoring pipeline](evaluation_pipeline.md) for how LC0 is used as a distillation teacher.
