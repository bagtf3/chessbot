# FAQ

## What is Xerces?

Xerces is an open-source chess engine built around Monte Carlo Tree Search and a custom-trained neural network. It is a personal project, built and trained on a single consumer GPU (RTX 2080, i9-9700K) by one person (so far).

---

## Is Xerces an Lc0 clone or fork?

No, I wrote the C++ and python myself from scratch. Interestingly though, the `chess-library` backend was provided by Disservin, the lead developer of Stockfish.

---

## Why build this instead of just using Stockfish or LC0?

I've been interested in chess and chess engines for a long time and the best way for me to see how
things work is to build it myself. It's also fun to be able to experiment with new ideas on my own project.

---

## How strong is it?

I estimate around 3050 Elo (human scale) based on validation runs against Stockfish at fixed depth (12, 15, 20 plies, etc). In competition mode it runs 8-10 centipawn loss on average.

---

## Has it ever beaten Stockfish?

Yes, once.

---

## Leela?

Not even close.

---

## Why train on a single GPU instead of cloud compute?

That's intentional. Part of the appeal of the project is the constraint — figuring out how much strength you can get out of a limited hardware budget is more interesting than throwing resources at the problem. The design decisions throughout the stack (efficient board encoding, aggressive caching, adaptive early stopping, prior fudging) all exist because compute is tight.

---

## Do you use history planes?

Yes, as of the current primary architecture. Xerces started from the opposite hypothesis — that AlphaZero-style history planes are largely cargo cult, with the model learning to ignore most of them. That was tested head-to-head against a history-aware encoding, and stacking a short window of prior positions turned out to earn its keep, so it's now the primary encoding rather than a side experiment. The current scheme stacks the current position plus 5 prior frames (6 total), each in the same token-per-square style as the base encoding, alongside a per-frame repetition flag and a few metadata scalars (castling rights, side to move, halfmove clock). It's a lighter touch than the classic 8-ply AZ history stack — full board planes per frame — while still giving the model direct access to recent piece trajectories and repetitions instead of relying solely on the search tree for that signal.

---

## What's different about the board encoding?

Rather than binary piece-plane stacks, Xerces's base encoding represents a position as 64 integer tokens — one per square — always from the side-to-move's perspective. The primary model stacks 6 of these frames (current position + 5 prior) rather than just one — see "Do you use history planes?" above. A few design choices worth noting:

- Castling rights are folded into the king token (9 possible values) in the single-frame encoding, rather than using separate planes.
- The en passant target square receives a dedicated token directly on that square, giving the model spatial context about which file is eligible.
- The board is always flipped so the side to move appears as "white," so the model never has to learn two perspectives of the same position — history frames stay frozen to that same orientation too, so they remain spatially aligned with the current frame.

---

## How fast does it run?

On the RTX 2080, the ~15.8M parameter precond-smartgate network is served in fp16 through ONNX Runtime + TensorRT. With the `MCTSForest` batching leaf encoding across all active games, selfplay sustains around 40,000 leaves per second, of which roughly 21,000 per second actually reach the network. The gap is absorbed before inference: the C++ priors cache resolves repeated positions, terminal nodes need no evaluation, and matelock's must-visit leaves are forced without a fresh prediction. That works out to roughly 750 games per hour of training; validation vs Stockfish is about 100 games per hour.

---

## Can I play against Xerces?

---

## Why the name Xerces?

Mostly because I think its cool!

---

## What's next for the project?
More selfplay. I think the `better data -> better search -> better model -> better data` loop is still progressing (slowly). Beyond that, the iterative deepening MCTS experiment is in progress, and the rescoring pipeline continues to be tuned as the model improves.

---

## What's the biggest thing you've learned building this?

1. Confidently wrong priors are much much worse than noisy but useful priors.
2. Managing side-to-move (STM) perspective flipping is the hardest part of the whole project.
3. Stockfish and Leela are hard to beat.

---
