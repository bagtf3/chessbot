# Search

## MCTS Overview

Xerces uses Monte Carlo Tree Search with PUCT selection, implemented entirely in C++ inside `pyfastchess`. The Python side submits leaf nodes for NN evaluation and calls back with results; everything else — descent, virtual loss, backpropagation, caching — is handled in C++.

## Node Structure

Each `MCTSNode` holds:

- **Visit count** (`N`) — atomic, incremented during virtual loss and resolved on backprop.
- **WDL accumulators** (`p_win`, `p_draw`, `p_loss`) — cumulative sums of backpropped WDL values (white-POV). `Q = (p_win - p_loss) / N`.
- **Qema** — exponential moving average of Q. Span is configurable (`qema_span`, default 40). Used in robust selection to smooth noisy early estimates.
- **Qdelta_sign** — EMA of the sign of Q deltas relative to the parent's STM POV; positive means Q has been trending in favor of the side to move. Span is configurable (`qdelta_span`, default 100). Used to detect Q convergence in early stopping.
- **Children** (`ordered_children`) — vector of `ChildEntry` structs with `uci`, `prior` (post-fudge), `raw_prior`, and a lazily-allocated `unique_ptr<MCTSNode>`. Children are only allocated when first selected.
- **State flags** — `is_expanded`, `is_pending`, `is_inflight`, `is_terminal`.
- **Visit share** — EMA tracking how often this child is selected relative to its siblings.

## PUCT Selection

The selection formula is standard PUCT:

```
U = c_puct * prior * sqrt(parent_N) / (1 + child_N)
score = Q_eff + U
```

`Q_eff` applies contempt adjustment on top of the raw Q value. Unvisited children initialize with FPU reduction: `Q_parent - fpu_reduction`, discouraging immediate exploration of all children.

Selection is pruned based on the remaining simulation budget: the tree skips children whose prior falls below a threshold determined by `pruning_factor`, concentrating visits on the most promising moves.

If a forced mate is detected anywhere in the tree, the node sets a `must_visit_uci` flag via an atomic state machine, overriding PUCT to steer all remaining visits toward the mating line.

## Leaf Collection

`collect_many_leaves(n_new, n_fastpath)` is the main interface Python calls each inference batch. It runs PUCT descent up to `n_new + n_fastpath` times and categorizes the results:

- **New leaf** — node not yet expanded; board position and Zobrist key returned to Python for NN evaluation.
- **Cache hit** — node found in the priors cache; value applied immediately without Python involvement, counted toward `n_fastpath`.
- **Terminal** — checkmate, stalemate, 50-move rule, or repetition draw; resolved with a hard WDL without NN.

Python only sees new leaves. Cache hits and terminals are handled entirely in C++.

## Backpropagation

After `apply_result(node, priors, wdl)` is called, WDL values are backpropped along the path from the expanded leaf to the root. Each ancestor accumulates the WDL and recomputes Q. Virtual loss is removed at the same time.

## Caches

### Priors Cache

The priors cache is a 1-million-entry LRU map keyed on the 64-bit Zobrist hash. Each entry stores:

- The post-softmax policy priors (uci, prior) for all legal moves at that position
- The original NN WDL (white-POV)
- The scalar value (`win - loss`)
- The visit count at the time the entry was written

On a cache hit during descent, the tree expands the node using the stored priors and WDL immediately, then backprops — no NN call needed. The hit rate is tracked and surfaced in telemetry.

### Raw Policy Cache

A secondary cache stores raw (Zobrist, value, policy_logits) tuples, inserted in bulk by Python after each NN inference batch via `raw_cache_bulk_insert_np()`. This feeds the priors cache on the C++ side: when the tree needs priors for a node, it checks the raw cache first, applies softmax, `uniform_eps`, and `prior_clip_max` in C++ (`build_priors`), and stores the result in the priors cache.

The separation allows Python to insert raw logits once per batch and let C++ handle post-processing, keeping the interface thin.

## Tree Reuse

When a move is played, `advance_root(move_uci)` promotes the chosen child to be the new root. The subtree rooted at that child is preserved — all its accumulated visits, WDL, and cached priors carry over into the next search. Subtrees for other moves are discarded. Tree reuse is configurable and is on by default.

## Python-Side Extensions

`MCTSTree` in `mcts_utils.py` extends the C++ `fasttree` with Python-side logic:

- **Simulation scheduling** — sim budget is set per-ply from a dict schedule (e.g., ramp from 1000 at ply 0 to 2800 at ply 45).
- **Early stopping** — checks every `es_check_every` sims using two criteria: robust selection confidence (RSC) and Jensen-Shannon divergence stability of the visit distribution. When either is met, the search stops before the ceiling.
- **Move sampling** — after search completes, the move to play is sampled from the top-5 candidates using temperature that decays linearly from ply 0 to ply 30.
- **Dirichlet noise** — added to root priors once per move to encourage exploration.
