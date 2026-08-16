# Blunder-replay probe: pool refactor + piggyback pipeline

Status snapshot as of 2026-08-15. Two efforts, sequenced back to back:

1. **Pool schema refactor** (DONE) — `short_fen`-keyed live pool, eviction
   instead of retirement, live ingestion from regular selfplay.
2. **Piggyback pipeline** (IN PROGRESS) — replace the bespoke probe
   subprocess + bespoke SF thread with: blunder-replay positions flow
   through the *same* game queue, loopers, and SF-rescoring thread pool as
   every other game. This doc tracks part 2.

## North star

Delete the standalone probe-worker subprocess (`spawn_probe_worker`,
`child_probe_looper`, `run_probe`) and the bespoke deep-SF pass
(`analyse_probe_file`, `run_deep_sf`). Blunder-replay positions should be
just another `GameSpec` flooding the normal `game_queue`, played by the
normal `GameLooper`, scored by the normal `SFRescoreThread` pool, collected
by the normal `Rescorer` — tagged so they skip games/hour and training-data
generation, but otherwise indistinguishable from a real game to the
telemetry/pause/reload/reaper machinery.

Trigger cadence stays retrain-count-based (same idea as today's
`PROBE_EVERY_RETRAINS`), but instead of spawning a worker process, it pulls
positions from the pool and floods `game_queue` with them.

## Part 1 — pool refactor (DONE)

- `src/chessbot/blunder_replay.py` (new module): pool primitives
  (`short_fen`, `load_probe_pool`, `save_probe_pool`, `evict_position`,
  `evicted_path`), the shared `is_blunder_candidate(v, p, cpl, z_stm)` rule
  function (single-pass, returns `(is_candidate, label)`), and the old
  `run_probe`/`analyse_probe_file`/`run_deep_sf` subprocess code (the last
  one commented out, not deleted, to avoid recreating the
  `rescore.py <-> validation.py` circular import while it's still needed as
  reference).
- `validation.py` trimmed to just the paired-validation-vs-Stockfish ladder.
- Pool keyed by `short_fen` (board+turn+castling+ep, no move counters) as a
  literal dict key, stored at rest as a dict (no more list+recompute).
  Eviction replaces retirement: immediate on `same_move AND found_equiv`, or
  2 consecutive `found_equiv` hits. Evicted records go to
  `<pool>_evicted.jsonl`, enough to rebuild/replay later.
- Two pools stay separate files (`blunder_positions_raw.pkl.gz` validation,
  `blunder_positions_selfplay.pkl.gz` selfplay) — schema is merge-compatible
  if that's ever wanted, but no merge is happening.
- `rescore.py`'s `finalize_game` live-ingests new candidates
  (`maybe_queue_blunder_candidate`, buffered, merged into the pool every 128
  via `flush_blunder_buffer`) using the same A/B/G3/F rules the old
  `scan_blunder_positions.py` scan applied post-hoc — no separate scan pass
  needed for *discovering* new candidates any more.
- One-time migration (`scripts/tools/migrate_probe_pool.py`) converted both
  existing pools to the new schema; old-format backups preserved
  (`*.pre_migration.pkl.gz`), not deleted. `scripts/tools/pairwise.py`
  re-sourced from the permanent per-epoch `probe_e*.csv` files (not the live
  pool, which loses history on eviction) and required backfilling those
  CSVs with a `short_fen` column from the preserved backups.
- Added `queue_blunder_eviction(short_fen_key, reason, epoch)` on
  `Rescorer` — sits unused right now, it's the intended destination for
  probe-replay eviction decisions once part 2 lands.

## Part 2 — piggyback pipeline (IN PROGRESS)

### Done

- **`looper.py`** (`finalize_game_data`): `probe_mode` concept fully
  removed. Branches on `game.meta.get("scenario") == "blunder_replay_probe"`
  instead. That branch builds a `brp_result_info` dict and pushes it
  straight onto `recent_games_q` — **no pkl.gz written**, no
  `games_finished` bump, no `pkl_file` key. Everything else (telemetry:
  mps/lps/batch fill/cache hit rate) already flows independently of
  `finalize_game_data`, so it's correct for free.
- **`run_selfplay.py`**: `pulled_games` drain now splits on that scenario
  tag into a separate `finished_blunder_replays` deque instead of
  `finished_games`, so `recorder.ingest_recents`/`update_game_index` (and
  therefore games/hour) never see them. Both deques get pushed into the
  rescorer in the same pass (`rescorer.submit(...)` /
  `rescorer.submit_blunder_replay(...)`) before one shared `rescorer.tick()`.
- **`rescore.py`**: `blunder_replay_intake` deque + `submit_blunder_replay`
  added. `tick()` prioritizes draining `blunder_replay_intake` before the
  normal `intake`, both now capped at `len(self.pending) < 40` (raised from
  20, since brp "games" are one ply and clear fast — no reason to starve the
  SF threads). `start_blunder_replay()` added, mirroring `start_game()`'s
  SF-submission shape but stripped of the candidate-moves/visits/policy
  machinery that only matters for training-example construction.
- **`game_utils.py`**: new `reconcile_game_boards(start_fen, history_uci,
  moves_played)` — found and fixed a real, separate bug in the process:
  every board reconstruction from a game log
  (`chess.Board(start_fen)`/`Board(start_fen)` with no history replay) was
  losing lc0 history-plane correctness and repetition state whenever
  `history_uci` and `moves_played` diverge (UHO/pre_opened scenarios, where
  `moves_played` only covers the model's own moves). Centralized the
  correct version (verified against `build_bootstrap_records.py`'s existing
  fen-checked implementation) and wired it into every live call site:
  `rescore.py` `start_game`, `start_blunder_replay`, `finalize_game`'s lc0
  waypoint board; `review.py` `GameViewer.reset`/`goto`,
  `analyze_with_sf_core`; `build_bootstrap_records.py`'s `iter_xc0_game`.
  Non-silent on genuine failure (prints a warning) but silent on the normal
  no-history-to-replay case. `scripts/misc/mine_search.py` (same bug, dead
  file) deleted instead of fixed.

### Not done yet — next up

1. **`brp_data` payload shape** (next task). `looper.py`'s
   `brp_result_info` needs an audit: adequate for what `start_blunder_replay`
   /`finalize_game` need downstream, without carrying dead weight. Known
   issue right now: it sets `"xerces_uci"` (a single move string), but
   `start_blunder_replay` reads `brp_data.get('moves_played', [])` — a key
   that doesn't exist in `brp_result_info` at all, so `to_sf_positions` ends
   up empty and nothing gets submitted. This is the immediate next fix.
2. **`finalize_game` branch for blunder replays.** It currently reads
   `game_state['game_data']` and `game_state['cfg']` unconditionally — a brp
   `game_state` has `'brp_data'` and no `'cfg'` key, so it'll KeyError as
   soon as a replay reaches it. Needs either a branch at the top of
   `finalize_game` or a dedicated `finalize_blunder_replay` that: compares
   the replay's move/CPL against the pool record's `orig_move`/`orig_cpl`,
   appends a `probes[]` entry (`found_equiv`, `same_move`), and calls
   `queue_blunder_eviction` on a same-equiv or 2-streak hit (function
   already exists, unused).
3. **Per-job movetime for `SFRescoreThread`.** Right now `movetime_ms` is a
   thread-level attribute shared by every game that thread processes
   (toggled base/throttled only by backlog). Blunder replays want the
   pool's deeper `sf_ms`/ratchet budget, not the normal `rescore_movetime_ms`
   default. Proposed mechanism (not yet built): extend the queue item to
   `(gid, positions, movetime_ms)`, build `pass1_limit`/`pass2_limit` per
   popped item in `SFRescoreThread.run` instead of once per batch.
4. **Collector + analysis.** Once replays finalize, need to accumulate
   results and periodically emit something like today's `probe_e{epoch}.csv`
   — but replays trickle continuously now instead of arriving in one
   epoch-stamped batch from a worker run, so the "one CSV per probe run"
   convention doesn't map cleanly. Open question, not yet decided: single
   running CSV/JSONL appended to per-replay (epoch column still stamped from
   `model_epoch`), or something else.
5. **The flood-the-queue injector.** Still need a function that, on the
   retrain-count cadence, pulls a batch of positions from the pool and
   pushes `GameSpec`s (scenario `"blunder_replay_probe"`) into `game_queue`
   at a metered rate — this replaces `spawn_probe_worker` /
   `PROBE_EVERY_RETRAINS`'s current subprocess-spawn behavior. Not started.
   `probe_specs()` in `blunder_replay.py` is the closest existing code (and
   now scenario-aligned) but is still shaped for the old bespoke-worker path
   — doesn't thread `uci_path`/`history_uci` into meta, which
   `reconcile_game_boards` needs to warm replay boards (right now it
   silently falls back to `start_fen`-only for replays, which is the
   "normal" no-history case, not wrong, just not preserving history planes).
6. **Deletion pass.** Once 1-5 are live and verified end-to-end:
   `spawn_probe_worker`, `child_probe_looper`, `collect_probe_worker`,
   `PROBE_EVERY_RETRAINS` cadence code in `run_selfplay.py`; `run_probe`,
   `analyse_probe_file`, the commented-out `run_deep_sf` in
   `blunder_replay.py`.

## Known stale/non-issues (don't re-flag)

- `run_selfplay.py:734`'s comment about "blunder replays trigger a top-up"
  refers to an older design iteration, not live code — ignore it, it'll get
  cleaned up in the deletion pass.
- `brp_result_info`'s `"stockfish_color"` key duplicating
  `game.meta`'s `"stockfish_is_white"` looked like a bug on first read but
  isn't — same two-key pattern the normal-game `mem_summary` already uses,
  `game.meta`'s spread supplies the real key downstream.
