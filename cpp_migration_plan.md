# Moving selfplay hot-path work out of Python

## Context

The packed-move change (pyfastchess `75c9e43`) took the C++ search from 46k to
118k sims/s and flipped the rig from CPU-bound to GPU-bound: duty 66.8% ->
77.5%, gap 0.025s -> 0.017s. Further C++ search optimisation has little left
to give.

The remaining Python overhead sits in the per-game-per-iteration and per-ply
code under `scripts/run_selfplay.py`. This plan removes it.

**The finding that shapes the plan:** the single largest item needs no C++ at
all. `stop_simulating` runs a movegen and builds ~35 throwaway Python strings
128x per outer iteration to answer a question that only changes once per ply.
Fixing that is ~6% of the worker's Python time. By contrast each genuine C++
port is worth ~1%. So the order is: bank the cheap Python wins, re-measure,
then port early stopping to C++.

Rough per-worker cost model (20 outer iters/s x 128 games = 2560 game-steps/s,
~17 moves/s, ~75 heavy early-stop checks/s):

| site | calls/s | est. share of one core |
|---|---|---|
| `stop_simulating` -> `legal_moves()` | 2560 | ~6% |
| `collect_tree_search_data` `rnd()` | ~5600 | ~1.1% |
| `robust_selection_criteria` marshalling | ~100 | ~0.7% |
| `process_results` list build | 2560 | ~0.4% |
| `record_es_check` + JSD | 75 | ~0.2% |
| syzygy FEN round-trip | <=17 | ~0.1% |

Total recoverable is ~8-12% of one core per worker. Stages 1-4 capture most of
it with no rebuild. There is one unquantified item (GC pressure from retained
`tree_data`) that could exceed all of the above; stage 2 measures it.

## Stage 0: give the bench a loop mode (prerequisite)

`scripts/bench/speedtest/bench.py:104` (`search()`) pins sim count and never
calls `stop_simulating`, `best()`, `collect_tree_search_data`, or
`check_for_terminal`; `bench_config()` disables sampling and noise. **The
harness cannot measure anything in this plan.** Every stage below needs this
first.

Add alongside `search()` (do not modify `search` -- it is the existing
equivalence oracle and must stay byte-stable):

- `run_loop_bench(...)` shaped like `looper.py:296-347`: N `ChessGame` objects,
  a `MCTSForest`, `stub_infer`, driving the real `stop_simulating` ->
  `make_move_from_tree` -> `check_for_terminal` path. Report
  `outer_iters_per_s`, `py_us_per_game_step`, and `perf_counter` buckets for
  `stop_sim_us` / `move_end_us`.
- GC instrumentation: gen0/1/2 counts and `gc.callbacks` timing.
- An `--equiv` mode dumping per game: stop ply per move, stop reason, and the
  root visit vector. This JSON is the oracle for every "behaviour-preserving"
  claim below.

## Stage 1: cache the legal-move count (pure Python)

`src/chessbot/mcts_utils.py:412`. The count only changes in `advance()`
(`mcts_utils.py:202`), the sole place the board is pushed.

- In `MCTSTree.__init__` and at the end of `advance()`, set
  `self.n_legal = len(board.legal_moves())`.
- `stop_simulating` reads `self.n_legal < 2`.

Behaviour-preserving: same predicate, same value, evaluated where `self.board`
is identical. No rebuild. **~6%, and ~90k string allocations/s removed.**

## Stage 2: measure GC, then fix `tree_data` retention (pure Python)

`collect_tree_search_data` (`mcts_utils.py:534-620`) retains
`self.tree_data[ply]` for the whole game: ~35 candidate dicts x 12 keys plus PV
dicts, x ~200 plies x 128 games ~= 900k live dicts per worker. Every gen-2
collection walks all of them.

Measure first with the stage-0 instrumentation, then pick the cheapest fix the
numbers justify:

- If gen-2 time is significant: `gc.freeze()` after worker startup plus raised
  `gc.set_threshold` for gen1/gen2. Zero structural change.
- If that is not enough: serialise each ply's `data` at creation and reassemble
  in `finalize_game_data` (`looper.py:486`). Consumers are `rescore.py:536-580`
  and `review.py`, both of which index by ply.

## Stage 3: `rnd` and the `counts` churn (pure Python)

One commit, two independent one-liners.

- `src/chessbot/utils.py:134` is `rnd = np.round`, a full numpy dispatch on a
  Python scalar, ~380 calls per ply. Make it `round(x, n)`. Both use banker's
  rounding, so values agree; the log dtype changes from `np.float64` to `float`,
  which is better for the pickle path. Check `rescore.py:536-580` and
  `review.py` for anything assuming numpy scalars.
- `looper.py:367-395`: replace the unbounded `counts` list-of-lists with 13
  running integer accumulators plus an `n_groups` counter, read directly by
  `maybe_push_telemetry` (`looper.py:506`). Removes ~2560 list allocations/s
  and the periodic `np.array` over ~115k rows.

**~1.5% combined.**

## Stage 4: syzygy board reuse (pure Python)

`mcts_utils.py:730-746` does `chess.Board(self.board.fen())` every ply once
<=5 pieces, inside a bare `except:`.

- `ChessGame` already maintains `self.python_chess_board` for Stockfish games
  (`mcts_utils.py:469`, pushed at `:526`). Generalise it: construct lazily the
  first time `piece_count() <= 5`, push every subsequent move, and probe against
  the live board.
- Narrow the `except` to the real cases and log once per game. The bare except
  currently also swallows `get_tablebase()` returning `None`, so a bad
  `ENDGAME_LOC` fails silently and permanently.

Does not move to C++ -- `probe_wdl` is python-chess. ~0.1%, but it removes a FEN
serialize plus parse from every endgame ply and surfaces a real config bug.

## Checkpoint: re-measure

Run the prod rig and compare `duty`, `gap`, `actual preds/s` against the
pre-1.2 baseline and the current round-3 log. If the gap has closed, stop here
-- stage 5 is ~1% and a real diff.

## Stage 5: move early stopping into C++

Only if Python is still the gap driver. The natural boundary is a
`should_stop()` on the tree that owns the JSD ring buffer, **not** cheap
accessors -- `es_checks` never escapes `mcts_utils.py`, and accessors would
still marshal `ChildDetail` vectors to Python just to compute a bool.

New in `pyfastchess/src/mcts.hpp` / `mcts.cpp` / `binding.cpp`:

```cpp
struct EarlyStopParams {
    int sims_floor, sims_ceiling, check_every;
    int min_top_visits, min_delta;
    bool use_robust;
    double jsd_thresh; int jsd_n_stable, jsd_min_delta, jsd_min_sims;
    double rsc_visit_share_floor;  // 0.15, hardcoded at mcts_utils.py:317
    double rsc_margin;             // 0.05, hardcoded at mcts_utils.py:321
    int rsc_top_n, rsc_min_visits; // 5, 100
};
void set_early_stop_params(const EarlyStopParams& p);  // per-instance
void reset_early_stop();
struct StopDecision { bool stop; std::string reason; };  // "", "full", "rsc", "jsd"
StopDecision should_stop(int sims_done);
```

`should_stop` reuses the existing `robust_selection_criteria()`
(`mcts.cpp:1329`) and `root_child_visits()` (`mcts.cpp:1160`) as plain C++
calls with no pybind conversion -- that is the entire win. Reuse those
functions rather than reimplementing their sorts, so tie-ordering is unchanged.
Store the ring buffer's `visit_dist` keyed by packed move, not UCI.

Python side: `stop_simulating` becomes the `n_legal` check plus one
`super().should_stop(...)` call. Delete `maybe_early_stop`, `record_es_check`,
`compute_visit_dist`, `js_divergence`, `jsd_convergence_stop`,
`rsc_performance_stop`, `ESCheck`, and the `es_*` state (~165 lines).

Stays in Python deliberately:
- `sims_completed_this_move` -- two writers (`looper.py:347`, the bench). Pass
  it as an argument rather than creating a second source of truth.
- The `sims_ceiling` schedule dict (`mcts_utils.py:46-53, 219`). C++ gets a
  scalar.
- `sim_stop_reason` -- `apply_stockfish_result` writes `"sf"` into it
  (`mcts_utils.py:634`) and `collect_tree_search_data` reads it (`:566`).

Equivalence bar is "same in intent" (agreed): `std::log` may differ from
`np.log` in the last ULP, which flips a stop only if a JSD lands within ~1e-15
of `jsd_thresh`. Verify by comparing the stop-ply distribution and stop-reason
counts across 128 games against the pre-change build via `--equiv`; do not
demand exact agreement.

## Stage 6: fused search snapshot (only if already rebuilding)

`collect_tree_search_data` makes three tree traversals (`depth_stats`,
`robust_selection_criteria`, `principal_variation`). A single
`search_snapshot()` returning all three saves one RSC marshal and two walks,
~0.12%. Not worth a rebuild on its own.

Do **not** build the telemetry dicts in C++ -- they must become Python objects
for the game log anyway, and it would hardcode the log schema into the
extension.

## Not moving, and why

| item | reason |
|---|---|
| `check_for_terminal` game rules (resign counters, material streak, eval-draw window) | Operates on Python-side `recents`, runs 17x/s, already backed by cheap C++ predicates. Pure risk, ~0 win. |
| Syzygy probe | python-chess dependency; the fix is board reuse. |
| `best()` / `select_using_robust` (`mcts_utils.py:100-200`) | ~1.7 calls/s. `robust_only_above=2400` > `sims_ceiling=800`, so `select_using_robust` never runs in training. ~0.007% of core. Touching the sampling path also changes the RNG stream. |
| `process_results` | Pure integer arithmetic; fix is Python-side (stage 3). |
| `scripts/misc/*` | Deprecated. If a change would require updating a file there, that file is a deletion candidate instead. |

## Risks

1. **Per-game config randomisation** (`cfg.sampleable`). Anything crossing into
   C++ must be a per-instance setter called from `MCTSTree.__init__`. Do not put
   `EarlyStopParams` in `singleton_registry.hpp` alongside the process-global
   caches. Note `ChessGame.__init__` (`mcts_utils.py:473`) builds the tree from a
   *copy* (`tree_cfg`) for `piece_training`, not `self.config`.
2. **`es_jsd_thresh` may be a list** (`config.py:44` comment, resolved by
   `utils.maybe_random_from_list`). `mcts_utils.py:341` compares it directly
   today. Confirm resolution happens before the tree is constructed.
3. **`sim_budget_` vs a new sims-ceiling field.** `advance()` calls
   `set_sim_budget(self.sims_ceiling)` (`mcts_utils.py:217`), but `sim_budget_`
   drives *pruning*, not stopping -- they only happen to be equal today. Have
   `should_stop` read `sim_budget()` rather than duplicating the field, or a
   schedule change will silently alter pruning.
4. **`root_child_details` is non-const** and calls `update_visit_share`
   (`mcts.cpp:1216`). The decay composes, so adding/removing calls is
   behaviour-preserving for `visit_share` -- but `cd.last_visit` is captured
   before the update, so telemetry does shift. Add a test in `pyfastchess/tests/`
   pinning that invariant before relying on it.
5. **Stage 0 is load-bearing.** Landing any stage without it produces a bench run
   that measures nothing.

## Verification

- Per stage: one commit, one bench run via `run_loop_bench`, compared against
  the previous accepted run. Stored runs live in
  `chessbot_data/speed_testing/runs/`; the current anchor is
  `packed_move_rerun_20260801_192904`.
- Stages 1-4 are behaviour-preserving: `--equiv` output (stop plies, stop
  reasons, root visit vectors) must be identical to the previous build.
- Stage 5 is behaviour-changing at the ULP level: compare stop-ply and
  stop-reason *distributions*, not exact equality.
- `python -m pytest tests/ -q` in pyfastchess for any C++ change (currently 50
  tests).
- Do not rebuild pyfastchess while the prod rig is running -- workers pick up a
  different build mid-test.
- Final check is the prod round-3 log: `duty`, `gap`, `actual preds/s`, `mps`,
  `lps` against the pre-1.2 baseline.
