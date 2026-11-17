# Roadmap: Search

Planned and candidate work on MCTS, the search controller and the C++
search. `pyfastchess/perf_review.md` is a separate backlog of C++
performance items.

## Search controller

Current design: `SEARCH_CONTROLLER.md`.

- **Retrain the controller on live self-play (top priority).** Required, not
  optional: it learned on one network's searches, and its stability and
  quality reads drift as the network changes. Retrain on `ctrl_record` data
  regularly so it stays in sync.

Incremental tuning:
- **Extension A/B.** Sample `ctrl_ext_max` 0 vs 2 on the same scenarios to
  measure what extensions are worth directly, not only through the
  changed-vs-rejected pair.
- **`ctrl_ext_q0` sweep.** Most uncaught blunders are plain locks. Try 0.7 /
  0.8 / 0.9 as a sampleable to see how many a higher bar catches, and at
  what sims cost.
- **Is the quality head blind to lock blunders?** Use `ctrl_qual` and
  `ctrl_rej` from the lean logs: if blunder locks sit just above the bar,
  raising it helps; if they read 0.95+, only retraining does.

## Cache as a transposition table: update cached W/D/L (and maybe priors)

- During backprop, every ~100 visits, fold the node's backed-up W/D/L into
  its cache entry with an EMA (span 4-5).
- Track how many visits inform each entry, so an n=100 update never
  overwrites an n=800 one.
- Priors: a slower schedule (every 200, 400, 800, ...) and EMA (span 8-10),
  possibly only when a KL divergence is detected.
- Updating priors was tried before and was worse, but that was before the
  rep-aware salted cache, and W/D/L was never tested.

## MateLock++: mate scout + MateLock + sticky bounds

| Part | Job |
|---|---|
| Scout | finds forcing mates past the tree's edge, from new leaves |
| MateLock (`must_visit` + `performance_penalty`, exists) | stress-tests a mate: can we force the game there from root, or can the opponent avoid it |
| Sticky bounds | lock in proven results as MateLock peppers the PV |

**Scout**
- Fires on newly expanded leaves once |Q| passes a threshold (~0.75).
- No-NN search over checks (maybe checks + captures) with a node budget.
- On a forced mate: the leaf becomes a pseudo-terminal with distance = mate
  depth, stores the mating move, and triggers MateLock like a real terminal.
- Each node is scouted at most once.

**Sticky bounds**
- `lo` / `hi` white-POV bounds in {-1, 0, +1} plus `end_plies`; proven when
  `lo == hi`.
- Backup: a proven win among the children proves the parent (min plies + 1);
  every child instantiated and lost proves a loss (max plies + 1); otherwise
  tighten.
- A proven node is an exact leaf. Rep results are safe inside our tree
  (rep history per node); never store draws in a shared hash (GHI).

**Move time**
- Proven win: play the shortest. Proven loss: play the longest. Root proven:
  stop early.
- `naive_loser`: when lost, play normal MCTS until the mate is under a
  threshold distance, to avoid sac-everything survival lines.

**Risks**
- A pseudo-terminal that becomes the root must expand normally or play its
  stored move.
- Scout must respect rep history and the halfmove clock.
- Selection and stopping must handle children with N = 0 or that are exact.
- Training targets for proven plies: value 1/0/0 and a point-mass policy,
  tagged so the pipeline accepts a missing visit distribution.
- CPU: gate on |Q| and checks, budget per scout and per move, log scout time.

**Build order:** sticky bounds on real terminals; scout + MateLock trigger;
filters and budgets; training targets; `naive_loser`; perpetual-check draws.
Possibly fold in Syzygy if tablebase reads are fast enough.

## Async leaf prefetch

- A side thread plucks leaves into a side queue to warm the cache.
- Most useful when running one game at a time (TCEC) and batches would go
  out short.

## Slimmer priors cache

- `PriorEntry` is 12 bytes per legal move. Store `prior` / `raw_prior` as
  fp16 and drop `move_idx` (only the miss path reads it): 6 bytes per move.
- Bigger win: container overhead. A flat open-addressing table with CLOCK
  eviction and inline priors instead of `unordered_map` + LRU list.
- Same idea for the tree: fp16 priors in `ChildEntry`; accumulators stay
  fp32.
- Needs the C++ speed harness (`scripts/bench/speedtest/`) first, so each
  change is measured on fixed positions.
