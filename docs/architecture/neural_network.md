# Neural Network

## Board Encoding

The neural network receives a board as 64 integer tokens, one per square, always from the perspective of the side to move (STM-as-white). Black positions are flipped via `sq ^ 56` before encoding so the model always sees its own pieces as the "white" side.

Token values:

| Range | Meaning |
|---|---|
| 0 | Empty square |
| 1–5 | STM pieces: pawn, knight, bishop, rook, queen |
| 6–9 | STM king with castling rights: none, kingside, queenside, both |
| 10–18 | Opponent pieces (same structure, +9 offset) |
| 19 | En passant target square |

Castling rights are folded into the king token — no separate planes are needed. The en passant square gets its own token on the target square, giving the model direct spatial context about which file is eligible. There are no stacked history planes; the model sees only the current position.

Vocabulary size is 21. This encoding is defined in C++ (`backend.hpp`) and is fixed across the entire stack.

## Conformer Architecture

The active model is `16m-conformer-interweaved`, defined in `scripts/model_variant_speed_test.py`. It is a hybrid architecture where each of the 10 blocks interweaves a transformer attention step with a convolutional residual step, rather than stacking all conv blocks followed by all transformer layers.

**Input and embedding.** The 64-token sequence is passed through a token embedding (`vocab_size=21 → d_embed=256`) and immediately reshaped to an 8×8 spatial grid.

**Positional embedding — graduated drip.** A positional embedding (`64 → cf=256`) is injected at block entry, in sequence form, with decreasing strength at fixed intervals: full strength at block 0, then ×0.1 at block 2, ×0.05 at block 4, and ×0.025 at block 6. This graduated drip lets the model absorb positional structure early and gradually de-emphasizes it as the representation matures.

**10 interweaved blocks.** Each block processes the representation in two stages:

1. *Attention stage* — reshape to sequence (64, cf), prenorm MHA: `LayerNorm → MHA(num_heads=8, key_dim=32) → Dropout → residual`.
2. *Conv stage* — reshape back to (8, 8, cf), conv residual: `Conv2D(3×3) → LeakyReLU → Conv2D(3×3) → LayerNorm → residual + LeakyReLU`.

No BatchNorm anywhere — small and variable batch sizes during selfplay make it unstable.

**Policy head — relational bilinear.** The spatial output is flattened to a (64, cf) sequence. Two independent projection towers compute a 128-dim vector per square:

- *From-tower*: `Dense(384, gelu) → Dense(256, gelu) → Dense(128)` — one vector per source square.
- *To-tower*: `Dense(384, gelu) → Dense(256, gelu) → Dense(128)` — one vector per destination square.

Normal move logits are computed as scaled dot products between from- and to-vectors: `(from · to^T) / sqrt(128)`, producing a (64, 64) logit matrix flattened to 4096 values. Underpromotion logits (192 values, encoding from-file × to-file × piece type) are produced by a separate small conv head and concatenated, giving 4288 total policy logits.

**Value head.** A 1×1 conv mix + LayerNorm + LeakyReLU branch feeds into attention pooling (learned per-square weights), then `Dense(256) → Dense(128) → Dense(3, float32)` producing raw WDL logits. The model is trained with `CategoricalCrossentropy(from_logits=True)` against a 3-component [win, draw, loss] target.

The model is trained with mixed float16 precision (`mixed_float16` global policy set at import time).

## TensorFlow + XLA Inference

The primary inference path uses TensorFlow with XLA compilation. `make_conv_infer()` in `looper.py` wraps the loaded Keras model in a `@tf.function(experimental_compile=True)` callable and returns a `fwd((boards_np,))` function that accepts a batch of 64-token board arrays and returns `(policy_logits [B, 4288], value [B, 1])`.

XLA traces the function at the first call for each batch size. To avoid repeated retracing, the selfplay loop pads batches to the next power of two or to a fixed set of candidate sizes. Throughput on an RTX 2080 is roughly 12k board evaluations per second at typical selfplay batch sizes.

`make_conv_infer` applies softmax to the value logits inside the XLA graph, returning `(policy_logits [B, 4288], wdl [B, 3])` where `wdl` is `[win, draw, loss]` probabilities, STM-POV. The scalar value used elsewhere (`nn_value`) is derived in C++ as `win - loss`. Policy post-processing — softmax over legal moves, `uniform_eps` blending, `prior_clip_max` clamping — is handled in C++ (`build_priors`) after raw logits are inserted into the raw cache.

## ONNX / TensorRT

An alternative inference backend is available via `infer_ort.py`, using ONNX Runtime with the TensorRT execution provider.

After retraining, `retrain_worker.py` can optionally export the Keras model to ONNX using `tf2onnx` (`export_tf_to_onnx()`), targeting opset 17. The exported model takes `enc_in: [B, 64]` int32 and produces policy logits and value.

`make_ort_infer()` creates an ORT `InferenceSession` with:

- `TensorrtExecutionProvider` — fp16 enabled, dynamic shape profiles from batch 1 to `max_bs`, engine file cached to disk so subsequent sessions skip the ~10-15 minute TRT compile step.
- `CUDAExecutionProvider` as fallback.

The session returns the same `(logits, value)` interface as `make_conv_infer`, so the selfplay loop is inference-backend-agnostic. ONNX/TRT is primarily used for benchmarking and when TRT's kernel fusion provides a throughput advantage over TF+XLA for a given batch size profile.
