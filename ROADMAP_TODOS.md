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
- First read (2026-11-11, sp9, 1000 sampled games, ~110k xc0 plies; SF
  depth 16 on the switches; qmm params stay locked for now):
  - The block fires on 17% of plies; about 37% of a blocked ply's sims come
    after the block (small logged sample).
  - 7.8% of blocked plies switch away from the leader. The final pick is
    better 57%, equal 9%, worse 34%. Blunder-size swings (ESL >= 0.12):
    5.0% saved vs 1.4% caused. Mean ESL 0.039 (leader) vs 0.023 (pick).
    Switches after a later tier1/tier2 stop gain about twice what
    switches at the ceiling gain.
  - Of the 92% that keep the leader, 84% had it best or equiv already and
    2% kept a 100+ cp loss. Those extra sims are the cost to trim.

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

### 6. MateLock++: mate scout + MateLock + sticky bounds
Three parts, each covering a blind spot of the others:

| Part | Job |
|---|---|
| Scout | finds forcing mates past the tree's edge, from new leaves |
| MateLock (`must_visit` + `performance_penalty`, exists) | stress-tests a mate: can we force the game there from root, or can the opponent avoid it |
| Sticky bounds | lock in proven results as MateLock peppers the PV, so they stop needing search |

**Scout**
- Fires on newly expanded leaves once |Q| passes a threshold (start around
  0.75).
- No-NN search over check-only moves (maybe checks + captures), with a node
  budget.
- On a forced mate: mark the leaf as a pseudo-terminal with distance = mate
  depth, store the first move of the mating line, and trigger MateLock
  exactly as a real terminal would. The leaf needs no NN eval.
- Each node is scouted at most once.
- Later: perpetual-check search for forced draws.

**Sticky bounds (node fields)**
- `lo` / `hi` white-POV bounds in {-1, 0, +1}, plus `end_plies`. Proven when
  `lo == hi`.
- Propagate in backup:
  - any child is a proven win for the mover: the parent is a proven win,
    min plies + 1;
  - every child is instantiated and a proven loss: the parent is a proven
    loss, max plies + 1;
  - otherwise tighten the bounds.
- A proven node is an exact leaf: visits return its value at full power,
  with no expansion below it.
- Our tree carries rep history per node, so repetition results are safe to
  propagate inside it. Never store draw results in a shared hash table (GHI).

**Move time**
- Proven win: play the shortest. Proven loss: play the longest.
- Root proven: stop searching early.
- `naive_loser` setting: when lost, play normal MCTS until the forced mate
  is under a threshold distance, then extend. This avoids sac-everything
  survival lines (e.g. M31 against itself in rook-odds).
- Expose `lo` / `hi` / `end_plies` in ChildDetail for `best()` and early stop.

**Risks to design around**
- Playing out the mate without stalling or crashing:
  - A pseudo-terminal that becomes the root (tree reuse) must drop its
    terminal state and expand normally, or play its stored mating move.
  - Opponent replies outside the proven line just re-enter normal search.
  - Selection, early stop and RSC must handle roots or children with
    N = 0 or that are exact.
- Scout correctness: it must respect the node's rep history and the
  halfmove clock, so a "mate" line can't run through a draw by rule.
- Training targets: proven plies get synthetic targets: value 1/0/0 for the
  side to move, and a point-mass policy on the mating move. Tag the records
  so the pipeline doesn't treat a missing visit distribution as broken.
  This is also hard-target data for won positions.
- CPU cost:
  - gate on |Q| and on checks being available;
  - per-scout node budget and a per-move cap on scouts;
  - log scout time per move and compare sims/sec with it on vs off.

**Log from day one**
- Per move: leaves scouted, leaves proven, and whether the root got proven.
- How often a proof changed the final move.
- Scout CPU time per move.

**Build order**
1. Sticky bounds on real terminals only.
2. Scout wiring: pseudo-terminal + MateLock trigger.
3. Qualifying filters and budgets.
4. Training targets.
5. `naive_loser`.
6. Perpetual-check draws.

Possibly fold in Syzygy if tablebase reads are fast enough.

### 7. Async leaf prefetch
- A side thread plucks leaves into a side queue to warm the cache.
- Most useful when running one game at a time (e.g. TCEC) and batches would
  otherwise go out short.

### 20. RSC as the one selector; best_* and root_* stats
- Goal: always select on RSC. It matches most-visited nearly always, and the
  rare disagreements are when the most-visited move is suspect (bad Qema and
  Qdelta_sign). Harden the RSC rules first so it can carry every move.
- Today the selector depends on how the search stopped:
  - tier1 / tier2 stops already need the RSC argmax to agree with the leader
    (tier1) or the top two (tier2).
  - `full` stops (ceiling, including gap-block extensions) use
    `select_using_robust` in `mcts_utils.py`, with its own hardcoded
    `(5, 60)` and visit floor instead of the es params.
  - Everything else falls back to C++ `best()`, which is plain most-visited.
- `best_wdl`, `best_q` / `Q_stm` (and the resignation Q) come from
  `details[0]`, the most-visited child, not the move chosen. On a refuted
  leader the value target gets the refuted line's numbers. All `best_*`
  should come from the chosen (RSC) move. Decide whether that is its mean
  W/D/L or its Qmm.
- Save `root_*` as well (`root.p_win` / `p_draw` / `p_loss`, `root.Q`).
  Measure root vs best to see if the gap carries signal (sharp positions,
  late leader changes, policy misses).

### 21. Slimmer priors cache (stretch)
- Goal: more cache entries per GB, so more NN evals are skipped on hits.
- `PriorEntry` today is packed_move + move_idx (uint16 each) + prior +
  raw_prior (fp32 each), 12 bytes per legal move.
  - Store `prior` and `raw_prior` as fp16. They come out of fp16 inference,
    so nothing is lost.
  - Drop `move_idx` from the cache. It is only read when NN logits are
    scattered onto children, which never happens for a node expanded from a
    cache hit; the miss path can derive it from `packed_move` + STM.
  - Together: 12 -> 6 bytes per move.
- Bigger win: container overhead. `unordered_map` + `std::list` LRU + a
  separately allocated priors vector are 3-4 heap allocations and ~180 bytes
  of pointers per entry. A flat open-addressing table with CLOCK eviction and
  inline or arena-stored priors removes most of that and speeds up lookups.
- Same idea for the tree: fp16 priors in `ChildEntry`; accumulators (W,
  W/D/L sums, Q, EMAs) stay fp32.
- Speed work like this needs the C++ speed-testing harness dusted off first
  (`scripts/bench/speedtest/`), so each change is measured on fixed positions
  rather than judged from live selfplay throughput.

### 22. Board-less nodes (later; tournament play)
- Not for the next push. Nodes stop holding a `backend::Board` copy; the
  tree keeps the root position and each descent pushes moves onto a walk
  board as it goes (copy the root once per descent, or unmake back up), as
  lc0 does.
- Gains: far less RAM (nodes shrink from hundreds of bytes to tens), better
  cache locality in selection, cheaper `advance` (freeing a subtree of small
  nodes). Raw CPU per descent is likely close to a wash: one board copy per
  new node becomes one `makeMove` per ply walked.
- Won't speed up multi-worker selfplay, which is GPU-bound. It matters for
  single-game TCEC-style play, where one search has to feed the GPU as fast
  as possible and CPU-side efficiencies like this and #21 show up directly.
- Touches everything that reads `node->board`: zobrist keys, terminal and
  repetition checks, sharp tagging, the encoders (leaves collected for a
  batch must be encoded while their walk board exists, or keep their path),
  `root_child_details` and PV output.
- One option (not final): a pool of walk boards per tree, sized to the
  most leaves that can be in flight. A descent takes a board; a terminal or
  cache-hit leaf hands it straight back, while a leaf waiting on the NN keeps
  it camped there so the batch encoder reads it as it reads `node->board`
  today. Backup unmakes one move per level on its way to the root and
  returns the board. `advance` with evals pending tears the pool down and
  runs a rectifying pass from the new root to rebuild any camped boards.
  Steady state allocates nothing per descent.
- Measure with the speed harness (#21) before and after.

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
- Firm assumptions for the rewrite:
  1. Validation plies are never trained on directly. Learning validation
     games ply for ply would teach the model SF's moves from the validation
     openings and wreck the validation signal. Validation blunders are still
     learned from, through the same lc0 correction pipeline as selfplay.
  2. vs_stockfish (and later vs_lc0) games are always trained on, SF plies
     included. Playing an expert is a deliberate config choice: playing one
     without training on it is validation, and not wanting it at all means
     leaving the mode out of `game_probs`. `train_on_stockfish` goes away,
     along with its `game_cfg_keys` snapshot and the combined
     `train_on_stockfish` / `skip_all_training` test in `start_game`.
  - Together these turn routing into two rules: an SF ply trains unless the
    game is validation (then it's skipped and never analyzed), and an xc0 ply
    always runs the full routing, with validation only blocking its own
    `live_buffer` sample.
- Treat paired validation like any other game mode: one per-ply routing
  decision for every game, instead of a game-level switch.
  - Today `is_validation_game` sets `skip_all_training` once per game
    (`start_game`). SF plies are dropped before analysis, and in
    `finalize_game` loop 1 every xc0 ply stops at `if ply['skip_training']`.
  - What a validation xc0 ply still gets today: SF analysis, CPL/ESL in the
    game rows, and `maybe_add_blunder_candidate` (validation is a BRP
    scenario), which is its only route to lc0. That route only fires for a new
    key while the pool has room.
  - What it misses: blunder stats, the `to_lc0` fallback for blunders the pool
    didn't take, and the 1-ply teacher stream.
  - Target: a validation xc0 ply goes through the same routing as a selfplay
    ply (blunder -> lc0 3-ply, teacher stream -> lc0 1-ply, stats). The only
    difference is that its own sample never reaches `live_buffer`. A blunder
    is a blunder and can be learned from through lc0; training on the
    validation game itself would taint the validation signal.
  - SF-played plies in validation are never SF-analyzed or trained, as
    today.
  - Decide: whether validation blunders feed the main blunder stats block or
    get their own rows, and whether lc0 samples from validation positions need
    a cap given the validation openings recur.
- One cp scale with a mate band: `CP_MAX` / `CP_MATE`.
  - Replaces the current +/-1200 cp scale, even though it changes the
    familiar numbers.
  - Mapping:

    ```
    no mate:   v = clamp(cp, -CP_MAX, CP_MAX)
    mate:      v = sign * (CP_MATE - k * d)    # d = plies to mate
    ```

  - `CP_MATE - k * d_max` sits above `CP_MAX`: every mate outranks every cp
    score, and shorter mates rank higher.
  - CPL is a plain subtraction everywhere. A slower mate costs `k` per ply.
    Losing a mate drops out of the band (a big loss). Anything above
    `CP_MAX` counts as equally won.
  - Fixes ESL's blindness in won positions: gives a precision blunder type
    for "slow mate" and "lost mate", with a magnitude (`missed_mate` today
    has none).
  - ESL stays as-is for everything else.
- Record SF mates explicitly per analyzed ply:
  - mate flag and mate distance in plies, for the best move and for the move
    played, not just a cp equivalent;
  - these are the label source for an MDH head and the precision metrics
    (see #6).
- Closeness `c = 1 - d/H` (H = horizon in plies) is the bounded form of the
  same distance, for a head or a search tie-break. For rescore, H is SF's
  real mate distance, so no short cap is needed.

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

### 24. Mate distance head (MDH)
- A radar plus distance: 0 when quiet, rising as a forced mate gets close.
  Preferred over MLH: our games end early (resign, eval draw, syzygy stop,
  max length), so game-length labels are cut short, while mate labels come
  from the position itself.
- First read on t1-big's MLH (2026-11-11, ~9.4k endgame positions with
  saturated Q):
  - MLH spreads ~11-13 plies across near-tied siblings.
  - It ranks SF mate distance with Spearman 0.78.
  - As a radar it gets AUC 0.87.
  - It adds only ~1 pt over the prior at picking SF's best move, and all of
    that within ~10 plies of mate. MateLock++ (#6) covers that range with
    proofs.
- Labels: SF mate distances from rescore (#12) and MateLock++ proofs.
  Syzygy gives win/loss without a distance.
- Shape: softmax buckets for "we mate in" / "we get mated in" plus "none",
  with a horizon H of about 20 plies.
  - Read out radar = `1 - p_none`.
  - Read out `E[d | mate] = sum(p_b * mean_d_b) / (1 - p_none)`.
  - Read out closeness `E[c]`, with `c = 1 - d/H` and bucket weights
    `1 - mean_d_b / H` from the label means.
- Uses:
  - radar (P(short mate)) triggers the MateLock++ scout;
  - closeness is a tie-break only when Q is saturated;
  - for+against sum as a sharpness signal for search allocation.
  - Q stays pure.
- Backup: scalar `c_for` / `c_against` accumulators averaged like Q, sides
  swapping and losing `1/H` per ply. Kept out of W. Sticky bounds override.
- Build after MateLock++, once labels have accumulated.

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

### 23. Is the teacher working? lc0 likeness and blunder rates over time
- Two signals that the lc0 streams are paying off:
  - The net gets more lc0-like: lc0-slice validation (`eval_progress_lc0.csv`)
    improves.
  - xc0 blunders less: ESL inaccuracy and blunder rates per ply (the rescore
    blunder-types table) fall over time.
- The lc0 slice is the epoch's fresh lc0 records, so its level moves with
  what the streams send. ESL selection and one-ply teacher sends made it a
  harder slice: read trends within a stream regime, not across a change.
  Blunder rates don't depend on what gets sent, so they are the steadier
  read.
- Track blunder rates per epoch so they plot next to the lc0 metrics: inaccuracy,
  blunder, blunder-not-won rates, and the share of each that reached lc0. Split by
  scenario so curriculum shifts don't pass for improvement.

## Decided against (for now)
- **playxerces auto-following the newest run.** Users will soon be able to
  pick the model, and the site will likely run a smaller, still superhuman
  model while the TCEC-trim xc0 grows toward the 30M model.
- **playxerces ponder on/off switch.** Not needed until the full UCI /
  cutechess-cli harness, which is a few months out.
- **C++ budget for `sf_move_sims`.** Riding fastpaths while SF searches is
  basically free, and the deeper xc0 search on SF moves helps SF learning and
  makes validation wins and losses easier to interpret.
