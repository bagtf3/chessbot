# Engine Landscape

Chess engines have gone through a few distinct eras. Xerces sits in the most recent one.

## Deep Blue

[Deep Blue](https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)) was IBM's purpose-built chess supercomputer that defeated Garry Kasparov in 1997. It represented the apex of handcrafted, hardware-accelerated alpha-beta search — evaluation functions written by grandmasters, tuned by engineers, running on custom VLSI chips. Deep Blue was a landmark achievement, but it was also a dead end: the knowledge was bespoke, brittle, and non-transferable.

## Stockfish

[Stockfish](https://stockfishchess.org/) is the dominant open-source classical engine and the gold standard for raw playing strength in the alpha-beta lineage. Classical engines like Stockfish work by searching a game tree with alpha-beta pruning — exhaustively evaluating positions to a fixed depth, using handcrafted (and later, NNUE-learned) evaluation to score leaf nodes. Stockfish is extraordinarily strong, but its search is fundamentally bounded by depth: it knows what it can see, and [no further](https://en.wikipedia.org/wiki/Horizon_effect).

## AlphaZero

DeepMind's [AlphaZero paper (2017)](https://arxiv.org/abs/1712.01815) changed the picture entirely. AlphaZero replaced alpha-beta search with [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) guided by a neural network trained purely through self-play — no handcrafted evaluation, no opening books, no endgame tablebases. It learned chess from scratch in hours and surpassed Stockfish. The key insight was that a strong policy and value network can focus search on the most promising lines, making deep alpha-beta exhaustiveness unnecessary.

## Leela Chess Zero

[Leela Chess Zero (Lc0)](https://lczero.org/) is the open-source community implementation of the AlphaZero approach. It proved the method generalizes beyond DeepMind's compute budget — with enough self-play games and a community of contributors, an open project could reach superhuman strength using the same MCTS + neural network paradigm. Lc0 has beaten Stockfish in top computer chess competition, winning [TCEC Season 15](https://en.wikipedia.org/wiki/TCEC_Season_15) and [Season 17](https://en.wikipedia.org/wiki/TCEC_Season_17). Lc0 is Xerces's closest conceptual ancestor and an ongoing reference point for comparison.
