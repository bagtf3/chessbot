# Roundless SelfPlayRunner

## Context

`scripts/run_selfplay.py` `main()` is 584 lines with a 310-line inner loop. Every
helper is a free function taking the state it needs as parameters —
`top_up_queues` has 8. `blunder_pool` is a bare dict referenced at 18 sites.
Blunder-replay scheduling state is split across three owners (Rescorer,
GameGenerator, run_selfplay).

This has cost real bugs. Three in one session: the initial `top_up_queues` was
handed the *previous* round's `n_retrains` for the BRP age gate; the
`is_validation` guard was added to one of two dump sites and not the other; and
the trickle ratio had to be chased through five comment sites across three files.

Rounds exist for two reasons that are both obsolete: separating selfplay from
validation, and a historical inability to retrain mid-round. Retrain now happens
mid-round via pause/reload, and validation can be a queue instead of a mode.

**Removing rounds is what makes a single owner object coherent.** Roughly half
of `main()`'s state is per-round and half per-run; with rounds, one object would
have ~half its attributes stale at any moment, and you'd need a second `Round`
object to manage that. Delete rounds and there is one lifetime, one owner, and
no boundary for anything to go stale across.

Outcome: continuous selfplay that never stops for a boundary, a `SelfPlayRunner`
owning all long-lived state, and helpers that read `self` instead of taking 8
parameters.

## Design

**`SelfPlayRunner`** in `scripts/run_selfplay.py` owns everything currently
living as `main()` locals:

```
self.rescorer, self.recorder, self.game_gen, self.blunder_pool
self.game_q, self.val_q, self.procs, self.sf_rescore_threads
self.current_epoch, self.brp_credits_owed, self.total_games
self.val_pending, self.val_rows, self.val_worker_idx
self.retrain_worker, self.cfg
```

Methods replace the free functions: `enqueue_blunder_replays()`,
`top_up_queues()`, `drain_results()`, `tick_rescorer()`, `check_retrain()`,
`maybe_start_validation()`, `finish_validation_batch()`. No parameters — they
read `self`.

**Validation becomes a queue, not a mode.** `play_vs_sf_prob` is `0.0` in the
selfplay config, so `assign_sf` (`game_utils.py:574`) short-circuits and
**`sf_queue` is permanently empty today**. `val_q` takes that slot: `spawn_workers`
(`run_selfplay.py:153-156`) already wires one queue to one worker, and
`pull_from_queue` (`looper.py:195-222`) already drains it at priority over
`game_queue`. No worker-side change needed.

**Batch completion by counting.** Runner sets `self.val_pending = batch_size`
when it enqueues. The `recent_q` drain counts arriving `paired_validation` metas
down to zero, then calls `finish_validation_batch()`. Worker keeps living; a
different worker is chosen next cycle.

## Files and changes

**`scripts/run_selfplay.py`** — the bulk. `main()` becomes thin: build
`SelfPlayRunner`, call `.run()`. Keep the old `main()` in place until cutover is
verified, then delete.

**`src/chessbot/game_utils.py`**
- `GameGenerator.__init__` (`:501-509`): drop `sf_cap` (dead — `play_vs_sf_prob`
  is 0.0 and `assign_sf` returns early). Bound `used_fens`.
- `used_fens` (`:503, 616, 623`): UHO reads a fixed PGN corpus and `pre_opened`
  fixed pkl path lists, both finite. Unbounded roundless, the 20-attempt retry
  at `:616` silently burns 20 board rebuilds per game and returns a duplicate
  anyway. Cap it (bounded set + FIFO eviction, or clear every N games).
- `validation_games()` (`:666`): take an explicit batch size instead of
  `cfg.n_games // 2`.

**`src/chessbot/validation.py`**
- `build_validation_summary` (`:217-304`): take the runner's `val_rows` list
  plus the validation cfg, instead of a `RecordKeeper` with a bolted-on
  `.config` (`run_selfplay.py:963`).
- Fix the dead key while here: it reads
  `g.get("stockfish_is_white", g.get("stockfish_color", False))` but the looper
  only ever writes `stockfish_color` (`looper.py:473`). Works purely on the
  fallback today.
- `paired_validation_games` (`:163-199`) is dead code — orchestrator uses
  `GameGenerator.validation_games`. Delete.
- Keep `create_validation_config`, `find_last_history_entry_for_run`,
  `continue_depth_from_previous_cfg` as-is — ladder state is already
  file-backed via `validation_history.jsonl` and survives roundless untouched.

**`src/chessbot/review.py` — RecordKeeper**
- Currently rebuilt per round (`run_selfplay.py:512`), so every attribute is
  round-scoped by construction. Make it run-scoped.
- Logging shows only `recent_games[-1500:]` (`maybe_log_results(window=1500)`,
  then `print_recent_summary(window=2000)` slices again). **Requested: show all
  games.** Roundless, retaining every meta is unbounded memory — so accumulate
  per-`(scenario, bucket)` counters incrementally in `ingest_recents` instead of
  slicing a retained list. "All games" becomes O(1) memory and free.
- Drop `run_num` from the log header.

**`src/chessbot/utils.py`**
- `summarize_recent_games` (`:~1190`): `rows` sorts alphabetically by scenario.
  **Requested: `paired_validation` last** — add a sort key that sinks it.
- `print_recent_summary`: take precomputed counters rather than a games list.

**`src/chessbot/config.py`**
- Remove `n_rounds`, `validation_every`.
- `n_games` → total games budget, **counting validation and selfplay together**.
- Add `validate_every_n_retrains = 5`, `validation_games_per_batch = 256`.
- Note `training_queue_buffer` and `retrain_size` are not `Config` attributes at
  all and nothing reads them — inert keys accepted by `update_via`. CLAUDE.md
  claims `training_queue_buffer` drives the retrain trigger; the real trigger is
  `n_files >= PRIMARY_TRIGGER_SHARDS`. Fix the doc.

## Work order

1. `SelfPlayRunner` skeleton — move `main()` state to attributes, phases to
   zero-arg methods, rounds already gone. Old `main()` untouched alongside.
2. `val_q` + designated worker + `val_pending` counting + `finish_validation_batch`.
3. RecordKeeper/`print_recent_summary` to incremental counters; all-games
   logging; `paired_validation` sorted last.
4. Config key changes; `used_fens` bound; `sf_cap` removal.
5. Cut `main()` over to the runner. Delete old `main()` and dead
   `paired_validation_games`.

## Invariants that must not break

- **Ladder fires once per complete batch.** `build_validation_summary` applies a
  two-consecutive-runs-over-50% rule (`consec_over_50`) — it counts *runs*, not
  games. A rolling window would double-count overlapping games and advance SF
  depth roughly twice as fast. Batches must be discrete and non-overlapping.
- **Partial batches must never bump depth.** The only gate today is
  `min_games = 30`; a crash mid-batch leaves 30+ colour-imbalanced games that
  would bias `score` and move the ladder. Needs an explicit complete/incomplete
  marker.
- Validation games keep `skip_training` (`rescore.py:743-744`, keyed on
  `'validation' in scenario`) — already correct, do not regress.
- The validating worker pauses and reloads through a retrain like every other
  worker. `pause_wait_and_reload` (`looper.py:157`) tears down the TRT session so
  the retrain gets the GPU; exempting it would hold VRAM the retrain needs. A
  batch straddling one reload is fine — `PLAYER_SPAN=7` means adjacent epochs
  differ by 25% of one training step.
- Stamp the batch's **starting** epoch on every row in that batch.

## Verification

1. **Before cutover**, capture a baseline: current `probe_table.py` /
   `pairwise.py` output, last `validation_history.jsonl` entry, and a
   `[rescore]` stats block.
2. Smoke run on a scratch run tag (`init_new_run.py <tag> --clone
   18m_10c6t_SWA_selfplay2`) with a small `validate_every_n_retrains` so
   validation triggers quickly. Confirm:
   - selfplay games keep flowing while one worker drains `val_q`
   - `val_pending` reaches 0 and `finish_validation_batch` fires exactly once
   - one new line in `validation_history.jsonl`, schema unchanged, `depth` /
     `consec_over_50` / `bumped` consistent with the pre-change record
   - the log shows all games and lists `paired_validation` last
3. **Ladder regression check**: run two consecutive validation batches with a
   forced >50% score and confirm depth bumps exactly once, not twice.
4. Kill the process mid-validation-batch, restart, and confirm the partial batch
   does **not** bump depth.
5. Confirm a retrain fires and completes with the validating worker paused, and
   that `probe_table.py` still parses the new run's `blunder_probe/`.
6. Only then point it at the real run.

## Settled decisions

| question | answer |
|---|---|
| sequencing | one pass, roundless from the start |
| termination | total games budget, validation + selfplay counted together |
| batch completion signal | rotate worker, no respawn, count completions off `recent_q` |
| cadence | config param, start at `validate_every_n_retrains = 5` |
