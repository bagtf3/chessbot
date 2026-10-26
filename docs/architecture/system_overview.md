# Stack Overview

## Two-Repo Structure

Xerces is built across two repositories that are developed and versioned independently.

`pyfastchess` is a C++ Python extension (`.pyd` on Windows, `.so` on Linux). It is installed into the Python environment and imported as `import pyfastchess`. It owns everything performance-critical: board state, legal move generation, Zobrist hashing, the MCTS tree, and position caching. None of this crosses the Python/C++ boundary at inference time — the C++ tree drives selection internally and surfaces only the leaf nodes that need NN evaluation.

`chessbot` is the Python package (`src/chessbot/` in src layout). It imports `pyfastchess` and owns the rest: neural network inference, training orchestration, selfplay loop, rescoring, game review, and configuration. The split is intentional — the Python side is meant to stay thin so that a future full-C++ inference path requires minimal rework.

## The Chess-Library Backend

The C++ board layer wraps [Disservin's chess-library](https://github.com/Disservin/chess-library), a header-only C++20 library for board representation and move generation. Rather than exposing it directly to Python, `pyfastchess` presents it through a `Backend` facade defined in `backend.hpp`.

The facade exposes the operations Xerces actually uses:

- **`board_to_64_tokens()`** — encodes the position as 64 int16 token IDs, one per square, in side-to-move-as-white orientation. This is the primary input to the neural network.
- **`legal_move_mask()`** — returns a 4288-element uint8 mask and a (uci, flat_index) map. The 4288 slots correspond to the 64×67 policy logit layout (64 source squares × 67 destination slots, where the extra 3 slots per source handle underpromotions).
- **`moves_to_indices()`** — converts a list of UCI strings to flat policy indices; used when inserting cached priors.
- **`push_uci()` / `unmake()`** — make and unmake moves with full history for tree reuse.
- **`hash()`** — incremental Zobrist key, used as the cache lookup key.
- Terminal detection, check detection, and material counting are also exposed for adjudication.

The 64-token board encoding and the 4288-slot move space are defined entirely in C++ and are fixed across the whole stack. The Python side never encodes boards or computes move indices itself.

## Python/C++ Interface

`pyfastchess` is built with [pybind11](https://github.com/pybind/pybind11). The binding layer in `binding.cpp` exposes `Board`, `MCTSTree`, `Cache`, and supporting structs (`WDL`, `ChildDetail`, `PVItem`, `CollectResults`) as Python objects.

The critical hot-path interface is:

1. Python calls `tree.collect_many_leaves(n_new, n_fastpath)` — C++ runs PUCT descent and returns new leaf nodes that need NN evaluation, plus counts of cache hits and terminals.
2. Python runs batched NN inference on the leaf boards.
3. Python calls `tree.raw_cache_bulk_insert_np(zobrists, values, logits)` to populate the raw policy cache, then `tree.apply_result(node, priors, wdl)` per leaf.

Everything between steps 1 and 3 — PUCT scoring, virtual loss, backpropagation, cache lookup — happens entirely in C++ with no Python overhead.

## Configuration

Both repos read from a shared `Config` object (`src/chessbot/config.py`) loaded from a YAML file at run start. Unknown keys raise `AttributeError` immediately rather than silently being ignored. The C++ tree receives its configuration through explicit setter calls (`set_cpuct()`, `set_dirichlet()`, `set_vscale()`, etc.) after the tree is constructed.
