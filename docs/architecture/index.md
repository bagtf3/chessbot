# Architecture

Xerces is split across two repositories. `pyfastchess` is a C++ Python extension that owns board representation, the MCTS tree, and position caching. `chessbot` is the Python package that owns neural network inference, training orchestration, rescoring, and tooling.

```
chessbot (Python)
  run_selfplay.py       orchestrator — workers, retrain loop, validation
  GameLooper            batched NN inference, concurrent game management
  Rescorer              SF analysis, blunder detection, training data prep
  retrain_worker.py     model training subprocess
  GameViewer            game replay and review tools
  model.py              network architecture definitions
  infer_ort_trt.py      ONNX Runtime + TensorRT inference backend

pyfastchess (C++ extension)
  MCTSTree              PUCT selection, backprop, early stopping hooks
  MCTSNode              per-node state, visit counts, WDL accumulators
  Backend (Board)       board state, move gen, 64-token encoding
  Cache                 LRU priors cache, raw policy cache
```

Pages in this section:

- **[Stack Overview](system_overview.md)** — how Python and C++ are layered; the chess-library backend stub
- **[Search](search.md)** — MCTS in C++: node structure, PUCT selection, caches
- **[Neural Network](neural_network.md)** — precond-smartgate model, ONNX Runtime + TensorRT inference
- **[Selfplay Suite](training_pipeline.md)** — GameLooper, StockfishThread, game curriculum
- **[Rescoring & Retraining](evaluation_pipeline.md)** — SF analysis pipeline, training data prep, retrain subprocess
- **[Review Tools](runtime_and_inference.md)** — GameViewer, RecordKeeper
