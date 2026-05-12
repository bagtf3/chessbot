# What is Xerces?

Xerces is an open-source, neural-network powered chess engine, trained on multiple data sources: published Lc0 game data, curated Xerces self-play data, and ongoing self-play with Stockfish rescoring.

- Monte Carlo Tree Search with PUCT selection
- Neural network evaluation (multiple architectures — currently a [conformer](https://arxiv.org/abs/2005.08100) ~16M params)
- Stockfish post-hoc rescoring for blunder detection and visit redistribution
- Async retrain loop running between self-play rounds

The [Python package](https://github.com/bagtf3/chessbot) handles training, search orchestration, and inference using TensorFlow and ONNX Runtime. It sits on top of a high-performance [C++ backend](https://github.com/bagtf3/pyfastchess) built on [Disservin's chess-library](https://github.com/Disservin/chess-library), which provides the core board representation, move generation, and MCTS tree.
