# Xc0 Data + Retrain Overhaul TODOs (2026-09-05)

Corpus snapshot: 13,081 pretrain shards = ~133.95M records (13,081 x 10,240)
feeding `18m_gen2_pretrain`.

DONE: #1 (rescorer tscaler strip), #3 (looper lean {header,plies} pkl + FATTY
collector + emit_header), #4 (unified prior_clip/uniform_eps blend, C++
mcts.cpp:997-1013 and Py utils.blend_to_uniform), #6 (buffer 12/12/4), #7
(4-tuple native read via normalize_record_arity; only bootstrap rewrites).
pyfastchess rebuilt (nn_wdl_stm / min_sims / sim_budget getters live).

## 2. Rescorer: become the live-migrator  [IN PROGRESS -- the only real build left]
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

## 8. Historic -> cold storage drain (HDD)  [DONE]
- `rb.cold_store_files` (shutil.move, cross-drive) retires the 4 consumed
  historic shards after a successful retrain, in poll_retrain's `if ok:` block
  alongside the existing primary->replay rotation. Skipped when
  `historic_cold_dir` is empty, so the drain is opt-in via env.
- Env: `BOOTSTRAP_PKL_GZ` (was BOOTSTRAP_TFREC_DIR -- historic is pkl.gz now)
  -> training_data/pretrain_shards; `HISTORIC_COLD_DIR` ->
  D:/chessbot_cold/training_data/18m_gen_pretraining_shards.
- Draining historic = signal it's time for a new re-pretrain. Documented in
  CLAUDE.md's Training Loop section. (The "NEVER delete/modify historic" rule
  this item claimed to overturn was never actually in CLAUDE.md or memory.)
- Budget: 13,081 / 4 = ~3,270 retrains before depletion. `sample_files` raises
  once fewer than 4 remain, which is the intended loud stop.

## 9. Auto-archive the pretrain eval CSV  [DONE]
- `rescore.migrate_pretrain_progress(run_dir)` freezes the whole pretrain
  `eval_progress.csv` into `eval_progress_pretrain_continued.csv` and empties
  eval_progress.csv (schema kept) so selfplay's counter restarts at 0. No-op
  once the continued file exists, so it runs exactly once per run.
- Wired at selfplay startup: `selfplay_runner.py` calls it right after
  get_unprocessed, before next_model_epoch reads the counter. Sole entry point
  (`run_selfplay.py` -> `run_roundless`) always hits it.
- New runs also carry the continued file forward via init_new_run.py.

## Possible follow-up (not on original list)
- tempscale entropy path (mcts.cpp:1016-1079) still uses the old two-step
  blend+clamp; could share the unified blend if desired. Separate from #4.
