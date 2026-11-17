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

## Decided against (for now)

- **playxerces auto-following the newest run.** Users will pick the model.
- **playxerces ponder switch.** Not needed until the full UCI harness.
- **C++ budget for `sf_move_sims`.** Riding fastpaths while SF searches is
  basically free, and the deeper search helps SF learning.
