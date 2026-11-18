# Search Controller

The search controller decides when a search has seen enough. It is a small causal temporal model that watches the MCTS search at a check-in every 100 root visits and judges whether the current leader is settled and trustworthy. It runs inside the C++ search through ONNX Runtime on the CPU, so selfplay never stops to ask Python mid-search.

It is live in selfplay, validation and both blunder-replay probes, and it is the only stop rule.

## What It Sees

- **Root context**, built once per search: the network's WDL and policy shape, material, mobility, king safety, pawn structure and game state.
- **Check-in features**, built every check-in: the top four root children's visit shares, recent visit flow, Q, Q trends and priors, plus leader stability and the shape of the visit distribution.

It never sees the sim budget, so it generalizes beyond whatever sims schedule is in use.

## Two Heads

- **Stability**: will the leader still be the leader at the end of a full search?
- **Quality**: is the leader a good move, judged against Stockfish?

## How a Search Ends

- **`ctrl`**: both heads clear their thresholds on consecutive check-ins with the same leader. The search stops early with confidence.
- **`lock`**: the leader's visit lead can no longer be overtaken in the sims that remain.
- **Extension**: a lock with low quality, in a position that isn't a dead draw, is rejected. The search gets more sims, and pruning is relaxed until the next move. The quality bar loosens as the position gets more decided.

The controller is a redistributor. Easy moves stop early, and the sims they save go to the hard positions it flags.

## What It Bought Us

From the val_test A/B against the old tiered early stop, and the first 1,800 games of production selfplay:

| | controller | old rules |
|---|---|---|
| TSL per move (val_test A/B) | -5.5% | baseline |
| sims per move (val_test A/B) | +4% | baseline |
| TSL per move (selfplay) | 7.04 | 7.42 |
| blunder rate (selfplay) | 1.02% | 1.19% |
| sims per move (selfplay) | 1,286 | 1,347 |

Inside the controller (production selfplay):

| | |
|---|---|
| moves ending in a confident `ctrl` stop | 43% |
| sims per `ctrl` stop | 522 |
| TSL on `ctrl` stops | 2.15 |
| sims saved vs. the ceiling | 41% |
| extended moves where the move changed | 60% |
| TSL gained per changed move from extending | +19 |
| cost per move | ~6 ms |

## Why a Model

Hand-tuned stop rules have to decide when to stop from a fixed recipe. The old tiered rule watched the visit distribution settle, which is easy to fool: a move can take a large visit lead early and coast on it. The controller learns from the search's own trajectory what a settled search looks like, and it separately asks whether the move is any good. That second question is what lets it spend more on positions where a confident search is confidently wrong.
