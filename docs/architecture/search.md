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

### Sharp Move Boost

Captures and checks are where a policy net most often under-rates a move. The sharp boost guarantees them a look without touching the priors. While a node's visit count is inside `sharp_window` (`[100, 250]`), each **sharp** child (one that is a capture, a check, or both) gets an extra exploration term:

```
U += u_scale * sharp_boost * sharp * max(0, 1 - N/sharp_M) / (1 + N)
```

`sharp` is `capture + check` (0–2), so a capturing check gets double. The bonus decays linearly to zero by `sharp_M` visits (10), so it only lifts moves the search has barely explored and then gets out of the way. Live settings are `sharp_boost: 0.05` and `sharp_M: 10.0`.

- **Priors run unencumbered first.** Below 100 visits the node searches on plain PUCT. The boost only checks, once the node is established, that sharp moves got a few visits.
- **Additive, not multiplicative.** A multiplier on the prior can't be tuned across the prior range (3× of 0.001 is still nothing), so the bonus is added to the prior term.
- **Applies at any depth**, keyed on each node's own visit count. Sharp flags are tagged lazily, once per node, the first time it enters the window. They are not stored in the cache.
- **Boosted visits are kept out of training.** Each child counts the visits the boost decided (`boosted_N`), and these are subtracted before the visit distribution becomes a policy target. The net is never trained to imitate the boost itself.

Telemetry reports how often the boost changed a selection (`sharp=` on the PUCT stats line), a per-move `sharp_by_depth` histogram, and per-candidate `boosted` counts in game records. In a live sweep (~1,700 games per arm), `sharp_boost` 0.05 cut selfplay CPL from 7.07 to 6.54 versus no boost. A follow-up sweep over `sharp_M` (5, 10, 20) was flat.

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

A secondary cache stores raw (Zobrist, value, policy_logits) tuples, inserted in bulk by Python after each NN inference batch via `raw_cache_bulk_insert_np()`. This feeds the priors cache on the C++ side: when the tree needs priors for a node, it checks the raw cache first, applies softmax, `uniform_eps`, and `prior_clip_max` in C++ (`build_priors`), and stores the result in the priors cache. Optionally, priors that are still too spread out after that are sharpened further — see [Prior Temperature Scaling](../experimental_features/tempscale.md).

The separation allows Python to insert raw logits once per batch and let C++ handle post-processing, keeping the interface thin.

## Tree Reuse

When a move is played, `advance_root(move_uci)` promotes the chosen child to be the new root. The subtree rooted at that child is preserved — all its accumulated visits, WDL, and cached priors carry over into the next search. Subtrees for other moves are discarded. Tree reuse is configurable and is on by default.

## Python-Side Extensions

`MCTSTree` in `mcts_utils.py` extends the C++ `fasttree` with Python-side logic:

- **Simulation scheduling** — sim budget is set per-ply from a dict schedule (e.g., ramp from 1000 at ply 0 to 2800 at ply 45).
- **Effort scaling** — on top of the ply schedule, each move's sims floor and ceiling are multiplied by a scale from an **effort table**. The table was built offline from ~200k training games by measuring how often an outcome-changing mistake happens in each bin of game phase × |parent Q| × parent draw probability. Bins where Xerces plays worse get more sims; quiet, low-stakes bins get fewer. Scales are clamped to `[0.5×, 2×]`, so the budget moves to where mistakes actually happen rather than growing overall. `effort_alpha` sets how strongly the measured mistake rates are turned into scales (0 = flat).
- **Early stopping** — checks every `es_check_every` sims using two criteria: robust selection confidence (RSC) and Jensen-Shannon divergence stability of the visit distribution. When either is met, the search stops before the ceiling.
- **Move sampling** — after search completes, the move to play is sampled from the top-5 candidates using temperature that decays linearly from ply 0 to ply 30.
- **Dirichlet noise** — added to root priors once per move to encourage exploration.
