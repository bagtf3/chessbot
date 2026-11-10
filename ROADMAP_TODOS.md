# Xerces Roadmap and TODOs

Ideas and planned work for the next phase. `pyfastchess/perf_review.md` is a
separate backlog of C++ performance items and is not repeated here.

## Search and MCTS

### 1. `SearchParams` refactor (mcts.cpp / mcts.hpp)
- One `SearchParams` struct that replaces the pile of setters and getters.
- Maps to and from a Python dict, has robust defaults, and is easy to
  configure and monitor.
- Centralizes everything search-related: c_puct, early-stop params, sims
  floor/ceiling/budget, randomness (noise, sampling), and the rest of the
  C++ knobs.
- Exposes params that are hard-coded today, e.g. the root noise ratio and its
  ply splits.
- Goals: zero performance loss, and one obvious place to find a param, update
  it, and see where the code reads it from.

### 2. Remove the descent cap when early-stop params are set
- `try_break` (1000 descents per `collect_many_leaves` call) is not needed
  for stopping correctness: the ceiling and `es_check_every` are checked on
  every non-blocked descent, fastpaths included.
- Remove it only when `es_params_set_`; keep it as a hang guard otherwise
  (scripts and benches that build trees without early-stop params).
- Update `COLLECT_DESCENT_CAP` / `f_stop` telemetry in `looper.py` alongside.
- Test before landing: warm-cache positions, capped vs uncapped, compare best
  move, visit distributions and matelock outcomes. Uncapped streaks run
  longer before pending leaves are backed up, which can shift visits and
  therefore policy targets.

### 3. Dynamic mbs ramp (after the `dynamic_mbs_min` sweep)
- Remains from the QMM era.
- Test a 2-4-8-16 ramp instead of a single 2x step.

### 4. Gap-block leader tracking
- Remains from the QMM era.
- Use `gap_block_sims` / `gap_block_leader` to measure how often the gap
  block changes the move vs. only extends search wastefully.
- Compare `gap_block_leader` against `xc0_move`, not `move_played`, so
  sampling isn't miscounted.
- Stretch: viewer flags the gap-block leader when it differs from the final
  move.

### 5. Cache as a transposition table: update cached W/D/L (and maybe priors)
- During backprop, every ~100 visits, fold the node's backed-up W/D/L into its
  cache entry with an EMA (span 4-5).
- Track how many visits inform each cache entry, so an n=100 update never
  blows away an n=800 one.
- Priors: revisit with a slower schedule (every 200, 400, 800, 1600, ...) and
  a slower EMA (span 8-10), possibly applied only when a KL divergence is
  detected.
- History: updating priors was tried before and was worse than not updating,
  but that was before the rep-aware salted cache, and W/D/L was never tested.
- Goal: the more a node is searched, the better its cached estimate. A refuted
  line updates its nodes' W/D/L, and a refuted prior gets lowered.

### 6. Sticky mates and an async alpha-beta solver
- When Q > 0.9, a side thread takes the tree and runs alpha-beta on it, with
  no NN, looking for forced mates or forced draws.
- Labels existing nodes with what it proves.
- Possibly fold in Syzygy, if tablebase reads are fast enough.

### 7. Async leaf prefetch
- A side thread plucks leaves into a side queue to warm the cache.
- Most useful when running one game at a time (e.g. TCEC) and batches would
  otherwise go out short.

## Workers and orchestration

### 8. Merge the lc0 worker into a regular worker
- Put lc0 work on worker 1, or the next worker after the SF worker, instead of
  a bespoke worker with its own CUDA/TRT/RAM partition.
- Run lc0 and xc0 games together and route leaves to the right batcher: the
  lc0 batcher stays local, xc0 uses the inference server.
- Kill the lc0 session when validation starts and let the lc0 queue back up,
  as now.

### 9. vs_lc0 game mode (unlocked by #8)
- Play vs lc0 like vs_stockfish, for expert learning.
- Needs two trees and two caches, so it's clear whose NN and search made each
  move and the game stays pure. Needs thought, but looks possible.
- Adds vs_lc0 expert learning to the curriculum while keeping lc0 blunder
  correction.

### 10. BRP on a single worker (maybe)
- Today every worker goes to BRP during a round, so selfplay stalls.
- Option: send all BRP to the third (non-SF, non-lc0) worker and let the others
  keep playing games.
- Open question: whether BRP is worth the compute at all. It reads the same
  every run, so it's closer to monitoring than learning.

### 11. BRP redesign
- About 50% from a static set of positions, which tracks overall progress.
- About 50% from new blunders, each tried once after it's 5+ epochs old.
- The first-try pass rate is the main stat worth tracking.

## Training data and curriculum

### 12. Rescore rewrite
- Keep the cliff-walker protection and the deblundering.
- Add Syzygy to the Z-correction bundle for games played without tablebases.
- Clean up and simplify the whole module.

### 13. Dynamic game generation from recent games
- Look at recently completed games (e.g. the last 10k) and generate games
  that maximize information content or learning.
- Target lines that are interesting but under-explored: contested lines, and
  lines where policy or value was miscalibrated.
- Find positions the model needs to improve on and spend more training time
  there.

### 14. More unbalanced openings
- Current UHO book is roughly +99 to +149; look for more unbalanced opening
  books.
- Thesis: curriculum can be the difference maker. Startpos + sampling is
  elegant, but compute should go where it matters most.

## Models

### 15. Next-gen model
- Pretrain and deploy the 25-30M next-gen model with smolgen, masked policy
  CE instead of SmartGate, and the other features planned for it.

## Housekeeping

### 16. RAM hygiene
- General goal: processes should only import what they use. Top-of-file
  imports of notebook and plotting libraries add up across spawned processes.
- Measured 2026-11-09 in a fresh process: `chessbot.review` adds ~164 MB,
  `chessbot.mcts_utils` ~76 MB (via `utils`). The main contributors are
  pandas (+pyarrow), `IPython.display` (+jedi, prompt_toolkit, ...) and
  `matplotlib.pyplot`. Roughly 1 GB across a selfplay run.
- Places to look (leads, not required fixes):
  - `scripts/run_selfplay.py` imports `selfplay_runner` outside the
    `__main__` guard, so every spawned child (workers, inference server,
    retrain worker) re-imports the full runner graph.
  - `child_looper` lives in `selfplay_runner.py`, so workers import the whole
    runner module to reach it.
  - `looper.py` imports `review` for one helper
    (`score_to_value_stm_pov_tanh`); `rescore.py` imports it for
    `load_game_index` / `ANALYZE_PKL`.
  - `utils.py` imports pyplot, `IPython.display`, `chess.svg` and pandas at
    the top; `mcts_utils` pulls all of that in for `rnd`.
  - `review.py` imports `IPython.display` and `PIL` at the top for notebook
    viewing.
  - playxerces: `engine_client.start()` imports `xerces_engine` in the web
    server process just to pass `main` as the spawn target.

### 17. Data layout: one folder per lineage
- Replace one top-level folder per run (`m137_8c8t_selfplay1`,
  `..._selfplay2`, ...) with one folder per model lineage, organized by data
  type. Rough shape, not final:

  ```
  m137_8c8t/
      configs/
      models/
      csvs/
      pkls/
      primary_buffer/
      replay_buffer/
      gamelogs/
          selfplay0/
          selfplay1/
  ```

- Migrate selected historic data into the new layout, then make sure
  everything going forward writes to it. No new folder under `SP_DIR` per
  run.
- Things that assume one run dir per tag and will need to follow:
  `Config.init_paths`, `init_new_run.py --iterate`, `previous_run_tag`
  lookups (e.g. BRP history), progress-csv carryover, `game_index`, and
  playxerces' `XERCES_RUN_CONFIG`.

### 18. Faster, lighter selfplay analysis (`repos/scratch/explore.py`)
- Today the analysis needs a rebuild of the combined pickle first
  (`analysis_staging/` chunks merged into `analyze_results_combined.pkl`, ~500
  MB on sp8). It's slow, hard on RAM, and won't scale to large game counts.
- First step: find out what still reads `analyze_results_combined.pkl` and
  whether it serves any purpose beyond feeding the analysis.
- Goal: eliminate the rebuild if possible, and refactor the analysis path for
  performance, e.g. read only the columns/games needed, incrementally, from a
  format that doesn't have to be fully loaded.

## Metrics

### 19. Next-gen game-quality stats beyond CPL and BMR
- Best-vs-played CPL is too noisy in one-sided positions: 30 CPL in a tight
  game is very different from 30 CPL a rook up. Scoring, weighting and
  experiment decisions are sometimes made off it, so it needs a better
  replacement.
- Candidates:
  - Self-sabotage damage: how much a move hurts the mover's own outlook, as
    the playxerces review measures it.
  - Improved CPL: loss measured in a way that accounts for how decided the
    position already is, e.g. expected-score (W/D/L) loss rather than raw
    centipawns.

## Decided against (for now)
- **playxerces auto-following the newest run.** Users will soon be able to
  pick the model, and the site will likely run a smaller, still superhuman
  model while the TCEC-trim xc0 grows toward the 30M model.
- **playxerces ponder on/off switch.** Not needed until the full UCI /
  cutechess-cli harness, which is a few months out.
- **C++ budget for `sf_move_sims`.** Riding fastpaths while SF searches is
  basically free, and the deeper xc0 search on SF moves helps SF learning and
  makes validation wins and losses easier to interpret.
