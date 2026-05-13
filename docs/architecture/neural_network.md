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

The active model is defined in `scripts/model_variant_speed_test.py` as a hybrid conv-transformer network with roughly 16M parameters.

**Input and embedding.** The 64-token sequence is passed through an embedding layer (`vocab_size=21`, `d_model_embed=128`) and then reshaped to an 8×8 spatial grid.

**Convolutional frontend.** 10 residual convolutional blocks (`conv_filters=256`) with LayerNorm (not BatchNorm — small and variable batch sizes make BatchNorm unstable during selfplay inference). This stage builds local spatial features.

**Transformer encoder.** The spatial map is flattened to a 64-token sequence with positional embeddings added, then passed through 5 transformer encoder layers (`num_heads=8`, `ff_dim=1024`, `dropout=0.025`, LayerNorm-before-attention). The transformer stage builds global, cross-square relationships.

**Policy head.** The sequence is reshaped back to 8×8, then a convolutional head produces a 64×67 logit map, flattened to 4288 values. The 67 slots per source square encode 64 destination squares plus 3 underpromotion slots (knight, bishop, rook; queen promotion falls on the destination square). At inference, illegal moves are masked to -inf before softmax.

**Value head.** Attention pooling over the spatial sequence (learned per-square weights, summed) feeds into two dense layers and a `Dense(3, float32)` output producing raw WDL logits. The model is trained with `CategoricalCrossentropy(from_logits=True)` against a 3-component [win, draw, loss] target.

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
