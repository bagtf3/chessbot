# Roadmap: Infrastructure

Workers, orchestration, rescore, retrain, data layout and housekeeping.

## Rescore rewrite (top priority, badly needed)

- Clean up and simplify the whole module; it has grown by accretion.
- Keep the cliff-walker protection and the deblundering.
- Add Syzygy to the Z-correction bundle for games played without
  tablebases.
- Firm assumptions:
  1. Validation plies are never trained on directly. Validation blunders
     still feed lc0 correction.
  2. vs_stockfish (and later vs_lc0) games are always trained on, SF plies
     included. `train_on_stockfish` goes away.
- Paired validation becomes a per-ply routing decision like any other mode:
  a validation xc0 ply gets the full routing (blunder -> lc0 3-ply, teacher
  stream -> lc0 1-ply, stats); only its own sample stays out of
  `live_buffer`.
- One cp scale with a mate band (`CP_MAX` / `CP_MATE`, shorter mates rank
  higher), so slow and lost mates get a magnitude.
- Record SF mates per analyzed ply (flag and distance, best and played).
  These are the MDH labels.

## Config rework (top priority, alongside the rescore rewrite)

Today a game's params pass through many layers (`config.py` defaults,
`config.yaml`, TRT setup, `validation_config.yaml`, code-level scenario and
stop-rule overrides, per-move scaling, the extension swap), each overwriting
the last. Two parts:

### One config with override sections, and a clean resolver
- One `config.yaml`: the base (selfplay), then override sections that list
  only what differs:

  ```yaml
  c_puct: ...
  pruning_factor: 1.5
  extension_overrides:
    pruning_factor: 1.0

  validation_overrides:
    sims_floor: 5000
    extension_overrides:
      c_puct: 2.0
  ```

- New override types follow the same pattern. BRP and lc0 keep their own
  files.
- Resolution is deterministic and happens once, at game creation, the same
  way sampled params are drawn: base -> the context's overrides -> sampled
  params. The game spec carries the final config, and nothing downstream
  rewrites it.
- The context's extension overrides go to the tree once. C++ swaps them in
  at an extension and back when the search ends. This replaces one-off
  params like `ctrl_ext_pruning`.

### Config holds tunables only
- `Config.init_paths()` derives run_dir, buffer dirs, game logs, index and
  csv paths and `model_path` from `selfplay_dir` + `run_tag`, and creates
  directories. It reruns on load, `update_via` and `copy(init=True)`, which
  can clobber `model_path` after TRT setup (the BRP and lc0 builders restore
  it by hand).
- Move all of that to a helper that runs once in the runner before workers
  spawn (a no-op after the first run beyond string joins). Workers get the
  computed paths and never derive them.
- The model / TRT location is set once by the TRT setup.
- Other entry points that read path fields (retrain worker, scripts, review
  tools) call the same helper. Size this with a grep first.

## SFServer

One process that owns every Stockfish engine, in place of SF threads spread
across the loopers (vs_stockfish moves, validation) and rescore (the
`SFRescoreThread` pool).
- Clients (loopers, rescore, BRP) send requests and get results back over
  pipes, or shared memory if pipes are too slow.
- Request kinds: play a move (latency-critical), analyze a game (throughput,
  one engine per game for TT locality), probe a position.
- Priorities: play moves before validation analysis before selfplay
  rescore, so a backlog never stalls games.
- One place for engine count, threads, hash, the rescore time tiers and
  engine restarts on a crash.
- Separates concerns: loopers play games, rescore scores them, and neither
  manages engines.
- The same pattern later serves an Lc0Server (`ROADMAP_IDEAS.md`).

## Rescore tuning

### SF time by position type
- Depth analysis (sp10): depth tracks material far more than anything else
  (about 15 at full material, 21 at 10 or less). Endgames sit near the
  depth cap.
- Lead: scale the movetime with material instead of only the decided /
  busy tiers.

## Retrain

### Multi-epoch retraining with patience
- Lead from a side experiment: get more out of each retrain's data when
  the holdout shard says it is still paying off.
- Use the holdout shard (already read only, never trained, left in
  primary) as the patience set: score it every S steps.
- No improvement within the patience window -> LR / sqrt(10), at most
  twice.
- Always run at least 1 epoch; stop when patience runs out after that,
  hard cap 3 epochs.
- The before/after fit check stays on the same shard.

### Robust worker -> runner messaging
- Today the worker reports once, in `retrain_done`; a hard death (OOM
  kill, CUDA segfault, terminate) skips the crash wrapper and the runner
  learns nothing about what got done.
- Send incremental messages as each step lands (`saved` once the model
  is written, `rotated` once the shards are moved, then `done`) so the
  runner always knows the true state: new weights or not, data spent or
  not.
- The runner acts on the last message received: reload TRT and release
  with new weights after `saved`; warn when shards were not rotated
  (they will be trained again) or when the csv rows are missing.
- Save the model via `.tmp` + `os.replace` so a crash mid-save cannot
  leave a half-written model.

## Resumable games (soft shutdown)

- A soft shutdown mode: each looper worker records, for every game in
  progress, everything needed to resume it exactly where it stopped.
- On the next start the games pick up mid-game. Downstream consumers
  (`rescore.py`, `review.py`, game logs, index) cannot tell a resumed game
  from one played straight through.
- Main win: no lost mid-game progress on a restart when there is no time
  to wait out `no new games`.
- Also enables moving games between workers, e.g. gathering every
  remaining validation game onto one worker, or rebalancing a lopsided
  fleet.
- The snag: the pyfastchess board, `MCTSTree` and the forest are not
  picklable. The saved state has to rebuild the game move by move: the
  start position and pre-moved opening moves, every played move, and the
  per-ply search data already logged (policy, visits, values, stop
  reasons, sampled params, scenario meta).
- Each game finishes its current search before it is saved, so every
  saved ply is complete. The tree is not saved: at most one search tree
  of progress per game is lost, and the next move searches from a fresh
  tree.
- Hard requirement: a resumed game's training data must be identical to
  what an uninterrupted game would have produced. If that cannot be
  guaranteed 100% of the time, the feature is not worth building.

## Runner command audit

- Review the operator commands in `selfplay_runner.py` (stdin parser and
  `handle_commands`): stop modes, pause / unpause kinds, respawn, end
  validation, full clear, status, help.
- Clean up the logic where it has grown by accretion, check the commands
  interact sanely (e.g. holds during a drain or a retrain), and fill
  gaps or extend where useful.

## Housekeeping

### RAM hygiene
- Processes should import only what they use. Measured 2026-11-09:
  `chessbot.review` adds ~164 MB, `chessbot.mcts_utils` ~76 MB (pandas,
  `IPython.display`, pyplot), roughly 1 GB across a run.
- Leads: `run_selfplay.py` imports `selfplay_runner` outside the `__main__`
  guard; `child_looper` lives in `selfplay_runner.py`; `looper.py` and
  `rescore.py` import `review` for one helper each; `utils.py` imports
  pyplot, `IPython.display`, `chess.svg` and pandas at the top.

### Data layout: one folder per lineage
- Replace one top-level folder per run with one per model lineage,
  organized by data type (configs, models, csvs, pkls, buffers, gamelogs
  per run).
- Follow-ups: `Config.init_paths`, `init_new_run.py --iterate`,
  `previous_run_tag` lookups, progress-csv carryover, `game_index`,
  playxerces' `XERCES_RUN_CONFIG`.

### Faster, lighter selfplay analysis
- The analysis needs `analyze_results_combined.pkl` rebuilt from
  `analysis_staging/` first: slow, hard on RAM, won't scale.
- First find what still reads the combined pickle; then read only the
  columns and games needed, incrementally.
- Plan: parquet under `chessbot_data/scored_games/<run_tag>/<scenario>/`,
  plain folder names (no hive `=`), `scenario` also stored as a column.
  - `per_ply`: per-ply SF analysis only (CPL, TSL, KL, ...). No policy or
    visits; those stay in `game_logs` for the rare review.
  - `per_game_summary`: one row per game, index fields plus the per-game
    stats `explore.py` uses. Small; this is what most analysis reads.
  - Each game writes its own file to staging; every N games an async merge
    writes one part (`.tmp` + `os.replace`, then deletes the singles).
    Readers read parts plus staging and dedupe on `game_id`.
  - No combined file. Readers glob many parts and pyarrow builds one df.
  - No shard-membership pointers. Find a game by scenario folder plus a
    `game_id` filter.
  - One reader helper that unifies schemas across runs (missing columns ->
    NaN).
  - Run deletion (val_test script) must also remove the run's
    `scored_games` folder.
  - Migrate frozen runs one at a time: split `df_all` by scenario, one
    merged file per scenario.
- Training shards stay pkl.gz: reads are async during training, so a
  columnar format (npz / parquet, fp16) would only save disk. Revisit for
  the ~100M pretraining corpus.

## Decided against (for now)

- **playxerces auto-following the newest run.** Users will pick the model.
- **playxerces ponder switch.** Not needed until the full UCI harness.
- **C++ budget for `sf_move_sims`.** Riding fastpaths while SF searches is
  basically free, and the deeper search helps SF learning.
