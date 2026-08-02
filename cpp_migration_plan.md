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
it with no rebuild.

## Status

Branch `speedups` off `dev`. **Stages 0, 1 and 3 landed** (`c5760c6`).
**Stage 2 dropped** -- measured, the hypothesis was wrong.

First `--loop` run, 32 games x 150 iters (under CPU contention with the prod
rig, so absolutes are inflated; the ratios are the signal):

```
steps/s             10536
stop_simulating      5.97 us/step
collect             27.85 us/step
make_move         1623.05 us/move
gc                  0.000 s  (0.02%)   gen0=9 gen1=1 gen2=0
```

Two things that change the plan:

- **GC is a non-issue.** Stage 2 is dropped, see below.
- **`make_move` dominates**, at ~270x the per-step cost of `stop_simulating`.
  That is `collect_tree_search_data` + `best()` + `check_for_terminal`. It
  moves stage 6 (fused snapshot) ahead of stage 5 on pure speed grounds --
  though stage 5 is still worth doing for the capability reasons below.

No baseline run exists for stages 1/3: the loop mode did not exist before them.
Measuring their effect means reverting `src/chessbot/` temporarily, which is
unsafe while the rig is live (see Risks).

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

## Stage 2: GC pressure -- DROPPED, measured at 0.02%

The hypothesis was that `tree_data` retention (~900k live dicts per worker)
would make gen-2 collections expensive. Measured with the stage-0
instrumentation: **gc 0.000s (0.02% of wall), gen0=9 gen1=1 gen2=0**. Gen-2
never ran at all.

The hypothesis was wrong. `tree_data` is large but it is almost entirely dicts
of floats and strings, and the *net* allocation rate -- which is what actually
triggers collections -- is dominated by short-lived objects, most of which
stages 1 and 3 just removed. No work here.

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

The natural boundary is a `should_stop()` on the tree that owns the JSD ring
buffer, **not** cheap accessors -- `es_checks` never escapes `mcts_utils.py`,
and accessors would still marshal `ChildDetail` vectors to Python just to
compute a bool.

**The case for this is capability, not just speed.** Living next to the tree
lets the stop condition see things Python cannot cheaply reach:

- **Hard stop when only one root child is still live.** When every child but
  one has been pruned, the search is just re-visiting a decided position.
  Python cannot see this without marshalling the whole child list; the root's
  PUCT scan has it for free.

  Note `cc.count_pruned` is **not** the signal -- it is an event tally summed
  over every descent in a `collect_many_leaves` call, and it is incremented
  both per-child and in bulk (`cc->count_pruned += cap_sz - i` on the early
  break). What is needed is a per-scan count of survivors, overwritten rather
  than accumulated: in `select_child_lazy_ptr`, when `parent == nullptr`, count
  the iterations that reach the scoring branch and store it as e.g.
  `root_live_children`. One live child means decided.

  Two conditions have to hold for that count to mean anything: `cap_sz ==
  n_child` (already required for `do_prune`, so the `2 + parent_visits` skip
  window is not truncating), and the scan must have run to completion rather
  than hitting the `unseen_visits` break -- on that break the remainder is
  unexamined, not proven dead.
- **Stop exactly on the ceiling.** Today the ceiling is checked in Python
  *before* `collect_many_leaves`, and that call then adds up to `micro_batch`
  new leaves **plus up to `n_fastpath` (1024) cached/terminal ones** --
  `looper.py:339` passes `max_fastpath=1024`. So `sims_completed_this_move` can
  overshoot `sims_ceiling` by far more than the 4-sim micro-batch suggests,
  and the overshoot grows with cache hit rate (currently ~65%). A C++ check
  inside the collect loop stops dead on the ceiling and makes the sim budget
  mean what it says.
- **No Python frame per micro-batch.** `stop_simulating` currently costs ~6
  us/step measured, at ~2500 steps/s per worker.

These are behaviour *changes*, and good ones -- but they must land as separate,
individually benched steps after the port itself is proven behaviour-neutral.
Do not bundle them with the move.

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

## Stage 5b: terminal checking

`check_for_terminal` (`mcts_utils.py:683-761`) runs once per ply. Most of it is
already backed by C++ predicates, so this is not a wholesale port -- three
specific pieces are worth moving, the rest should stay in Python.

**Worth moving:**

1. **`is_game_over()` returns two `std::string`s** (`backend.cpp:313`), built
   from enums by `reason_to_string` / `result_to_string`, then compared against
   string literals back in Python. Add an enum-returning `game_result()` and
   keep the string pair for Python display only. `terminal_or_legal_moves()`
   (`backend.cpp:691`) already computes exactly this without strings and is
   **not currently bound** -- bind it, or a thin wrapper over it.

2. **`material_count()` scans 64 squares** (`backend.cpp:342-360`) calling
   `board_.at(sq)` per square. It is pure bitboards underneath: sum
   `popcount(pieces(pt, WHITE)) * value` minus the black side, 10 popcounts
   instead of 64 lookups plus branches.

3. **`piece_count()` does the same 64-square scan** (`backend.cpp:362-371`) to
   answer what is one `popcount(occ())`. It is called per ply from the syzygy
   gate *and* from `best()`'s sampling gate (`mcts_utils.py:106`), so it runs
   more often than the material check.

Both (2) and (3) are internal rewrites -- same signature, same result, no API
or Python change, verifiable by a unit test over random FENs.

**Staying in Python:**

- Resignation counters, the material-diff streak, the eval-draw window. These
  read `self.recents` (per-game Python history) and fire ~17x/s. Moving them
  means marshalling that history into C++ and duplicating config for no win.
- The syzygy probe -- `probe_wdl` is python-chess. Stage 4 fixes the cost by
  reusing a live board instead of a FEN round-trip.

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
6. **`spawn_workers` sits inside the per-round loop** (`run_selfplay.py:440` ->
   `:525`), so every new selfplay round spawns fresh worker processes that
   re-import `chessbot` from disk. **Uncommitted edits to `src/chessbot/` are
   picked up by a live rig at the next round boundary.** Do not leave the
   working tree half-changed while the rig is up, and do not revert files to
   take a baseline measurement mid-run.

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
