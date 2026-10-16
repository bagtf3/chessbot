# What is Xerces?

Xerces is an open-source, neural-network powered chess engine, trained on multiple data sources: published Lc0 game data, curated Xerces self-play data, and ongoing self-play with Stockfish rescoring.

- Monte Carlo Tree Search with PUCT selection
- Neural network evaluation (a preconditioner + transformer "precond-smartgate" model, ~15.8M params)
- Stockfish post-hoc rescoring for blunder detection and visit redistribution
- Async retrain loop running between self-play rounds

The [Python package](https://github.com/bagtf3/chessbot) handles training, search orchestration, and inference in PyTorch, served through ONNX Runtime with TensorRT. It sits on top of a high-performance [C++ backend](https://github.com/bagtf3/pyfastchess) built on [Disservin's chess-library](https://github.com/Disservin/chess-library), which provides the core board representation, move generation, and MCTS tree.

## Design Philosophy

Xerces is built around the idea that chess strength emerges from the interaction between search and learning, not from neural scale alone.

Rather than pursuing the largest possible networks, Xerces explores whether moderately-sized, information-dense models paired with aggressive, adaptive search can produce strong and interesting play under real compute constraints.

The project emphasizes:

- search as a first-class intelligence system
- efficient information flow through priors
- empirical refinement during search
- runtime adaptability
- systems-level optimization over benchmark chasing

Xerces treats the neural network as a powerful guide, not an oracle. The search process itself is viewed as an active computational partner capable of reshaping, amplifying, and stress-testing learned beliefs in real time.

Part of the appeal is the constraint itself. Xerces is developed and trained entirely on a single consumer GPU (RTX 2080, 8GB VRAM, i9-9700K) as a non-commercial project. Running on cloud compute or a multi-GPU cluster is a deliberate non-goal. The interesting engineering challenge is doing more with less, and the design decisions throughout the stack reflect that.
