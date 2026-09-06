# Xc0 Data + Retrain Overhaul TODOs (2026-09-05)

Corpus snapshot: 13,081 pretrain shards = ~133.95M records (13,081 x 10,240)
feeding `18m_gen2_pretrain`.

DONE: #1 (rescorer tscaler strip), #3 (looper lean {header,plies} pkl + FATTY
collector + emit_header), #4 (unified prior_clip/uniform_eps blend, C++
mcts.cpp:997-1013 and Py utils.blend_to_uniform), #6 (buffer 12/12/4), #7
(4-tuple native read via normalize_record_arity; only bootstrap rewrites).
pyfastchess rebuilt (nn_wdl_stm / min_sims / sim_budget getters live).

## 2. Rescorer: mirror the migration script  [NOT STARTED -- the only real build left]
- Rescore becomes the live-migrator. Read `<game_id>.pkl` ({header, plies}) from
  game_logs, then per ply: run SF and add `cpl` (+ `cpl_depth`), build the
  blended `policy` target from `raw_visits` (item #5 -- call blend_to_uniform),
  compute `z_wdl` / `wdl_target`.
- Decide `reviewable` (paired_validation OR random < review_rate); if not, trim
  the FATTY extras to LEAN.
- Write final `<game_id>.pkl.gz`, update official `game_index.json`
  (+ mean_cpl / mean_bmr), then delete the looper's `.pkl`.
- Touches rescore's current reader (analyze_and_rescore ~651) + the brp reader.

## 5. Blend on retrain policy targets  [folds into #2]
- `blend_to_uniform` exists (utils.py) with no callers yet. Wire it into #2's
  policy-target build so retrain doesn't sharpen. `raw_visits` stays un-blended.

## 8. Historic -> cold storage drain (HDD)  [UNDECIDED -- may not do]
- After 4 historic shards sampled, move them to HDD cold storage.
- Draining historic = signal it's time for a new re-pretrain.
- Overturns the "NEVER delete/modify historic" rule in CLAUDE.md + memory --
  update those if this lands.
- Budget: 13,081 / 4 = ~3,270 retrains before depletion. Riskiest; defer.

## 9. Auto-archive the pretrain eval CSV  [DARKHORSE -- do before selfplay restarts]
- `eval_progress.csv` is UNIFIED: once selfplay starts in the same run_dir the
  looper appends its rows to the very same file (next_model_epoch reads it), so
  the pure-pretrain curve gets mixed in and is lost unless snapshotted first.
- Neither bootstrap_model_async_sparse_pkl.py nor retrain_worker.py renames or
  copies it -- the `_pretrain_continued.csv` in the d256 dir was a manual rename.
- Fix: on the pretrain->selfplay handoff, freeze a copy (e.g.
  eval_progress.csv -> eval_progress_pretrain.csv) so the pretrain-only history
  survives. Small local edit; land it BEFORE the selfplay gauntlet reboots or
  the gen2 pretrain curve gets clobbered.

## Possible follow-up (not on original list)
- tempscale entropy path (mcts.cpp:1016-1079) still uses the old two-step
  blend+clamp; could share the unified blend if desired. Separate from #4.
