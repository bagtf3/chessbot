# Experimentation

Offline tools around xc0 search: a position corpus mined from selfplay, the
search-controller data collector and trainer, and a retrain simulator. They
share the xc0 inference server with selfplay's launch code; each worker owns
its own NN caches.

## Modules

```
experimentation/
  corpus.py              mine, stratify, weight, save/load
  collector.py           search-controller training data (trajectories)
  train_controller.py    fit the controller to one or more collections
  export_controller.py   export a checkpoint for pyfastchess's C++ runtime
  predict_controller.py  run a checkpoint over collections, fire/gate tables
  retrain_sim.py         read-only retrain simulation
  runner.py              SF-label helpers (SfEval, is_equiv) and start_server
```

The controller itself is documented in `SEARCH_CONTROLLER.md`.

## SearchParams (pyfastchess)

One flat struct holds every search parameter. Per-move state stays outside
it (the stop state, the controller tracking, the contempt latch,
`noise_added_`).

Python API:
- `MCTSTree(board, params)`: params is a dict or `SearchParams`; missing
  fields take the struct defaults.
- `tree.set_params(params=None, **kw)`: merges into the current params, kw
  winning. Unknown names raise. Recomputes derived values (EMA alphas).
- `tree.get_params()`: dict of every field, defaults included, plus the
  derived flags. `tree.get_params("c_puct", "sims_max")`: just those;
  unknown names map to None.

Derived flags: `set_params` decides once whether a value turns its feature
off and stores a read-only flag (trailing underscore; setting one raises):

| flag | on when |
|---|---|
| noise_on_ | dirichlet_eps > 0 and dirichlet_alpha > 0 |
| sharp_on_ | sharp_boost > 0 |
| contempt_on_ | contempt_fight_c > 0 |
| pruning_on_ | pruning_factor > 0 |
| descent_cap_on_ | tree_descent_cap > 0 |

Tempscale keeps its value checks (entropy target 0 = off, trigger_q as is).
Move sampling is Python-only (`best()`): `move_sample_temp_plies = 0` turns it
off. `add_root_noise` is dropped for `dirichlet_eps = 0`.

Python sets params once per move (in `advance` / `reset_for_new_move`), never
per leaf; the hot path reads `params_` members directly.

Config side (Python): `sims_floor` and `sims_ceiling` (scalar) plus an optional
`sims_schedule` (`{ply: ceiling}`). Each move Python evaluates the schedule at
the current ply, applies the effort scale, and sets `sims_min` / `sims_max`.
c_puct keeps its scalar-or-dict form.

## Cache policy

The NN caches (`raw_policy_cache`, `priors_cache`) are per worker process.
Trees are always fresh per position (`reuse_tree` off).

## Corpus

Records hold `fen`, `uci_path` (full history for xc0h), SF evals and best
move, plus source fields (run tag, game id, ply, model epoch, scenario) and
stored stats (CPL, KL, CE, played move).

Mined from `game_index.json` + `analyze_results_combined.pkl` (rebuilt first),
opening only sampled game logs. Stratified:
- high KL, high CE, high CPL, critical positions, random baseline
- each record keeps its stratum and sampling weight, so results can be
  reweighted to representative while hard positions are overrepresented

Built once per corpus, saved, reused.

## Ideas

- Tree purity: search to N, then rebuild a bare root to the same N on the
  warm cache; compare cost (cold vs warm) and the result (pick agreement,
  root Q / Qmm delta, visit KL, PV agreement).
- Refutation -> cache correction -> rebuild: label refutation events offline
  (a PV move that led, lost the lead, and SF says was bad), overwrite the
  cache WDL of the refuted line's nodes, rebuild and compare to the
  uncorrected rebuild. Writes go through `raw_cache_lookup` +
  `raw_cache_bulk_insert`; `priors_cache` entries for corrected keys must be
  invalidated or updated to match.
- Refutation signals: refuted leaders show Qema and Qdelta_sign separating
  leader from pick while Q and Qmm barely do; the open question is lead time.
- policy blind vs value blind misses; forced-visit probes into the SF-good
  move; lc0 on the same corpus as a reference; a standing benchmark corpus
  across model checkpoints; wall time and nps per setting.

Every run: one pinned model checkpoint (hash recorded), TRT fp16 as in
selfplay, no root noise, no move sampling, selfplay paused while it owns the
GPU.
