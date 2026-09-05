# Xc0 Proposed TODOs (early scoping) - 2026-09-05

Light early scoping only. The original TODO list (`Xc0_05_SEP_TODOs.md`) is left
un-scoped on purpose - we do that together.

---

## P1. STM-POV: lock in testing, prune redundant flips

WHY: value POV correctness is load-bearing and mostly right, but there are real
flip-then-unflip seams and untested cross-boundary spots.

STATE (found):
- Rescore-side STM-POV IS well covered: `tests/test_rescore_math.py`
  (blend/equiv/blend_wdl all STM-POV via colour-mirror), `tests/test_move_scoring.py`.
- C++ orientation covered: pyfastchess `tests/test_history_tokens.py`,
  `test_terminal_fused.py`.
- GAPS (no test): raw-cache STM<->white swap in `mcts.cpp:356-362` and `:929-935`;
  SF-override Q derivation `mcts_utils.py:465-467`; Python-side cross-boundary
  "encode(black_pos)+flipped WDL == mirrored white pos".

REDUNDANCY SUSPECTS:
- A. `mcts_utils.py:462-463,477` - `Q_white` == `best_q` verbatim, yet both plus
  `Q_stm` carried in the `recents` tuple. Straight duplication.
- B. `mcts.cpp:356-362` / `:929-935` - raw cache materializes white-POV AND
  STM-POV from one value in the same block.
- C. Central round trip: Python emits STM-POV WDL -> C++ flips to white-POV for
  the engine -> every Python target consumer flips back to STM-POV. Persisted
  target is STM-POV at both ends; white-POV lives only inside the engine.
- D. `rescore.py:709-715` re-derives `Q_stm` from `best_wdl` with a flip as a
  fallback, though `Q_stm` was already stored STM-POV at `mcts_utils.py:470`.

PROPOSE (light):
- Add the 3 missing tests (raw-cache round trip in pyfastchess; SF-override Q;
  Python cross-boundary mirror). Cheap, high-value guard rails.
- Collapse suspect A (carry one Q_white, derive STM-POV at use). Suspects B/C/D
  are "document the single source of truth, don't churn the engine" unless we
  find an actual double-application bug. Low risk to leave, low reward to move.
- EFFORT: tests small; A trivial; B/C/D audit-only.

## P2. BRP vs lc0 distillation - couple, don't cut

WHY: user asks if BRP still earns its keep or if lc0 blunder-fix covers it.

STATE (found): they are COUPLED, not substituted.
- Both fire off the same `is_blunder_candidate` stream.
- lc0 (when `n_lc0_workers>0`) gates BRP pool admission (`rescore.py:438-444`),
  absorbs high-CPL positions BRP won't train (`brp_enqueue_max_cpl=90`), catches
  escalated still-failing probes.
- BRP's unique remaining value: (a) found_equiv self-correction TRAINING examples
  (`rescore.py:1058`), (b) validation/self-correction telemetry
  (`blunder_probe_history.jsonl`, reviewables).
- The memory note "BRP replaced by lc0 distillation" is NOT documented in-repo;
  agent flagged it unverified. `TEMPORARY` markers around the lc0 audit path
  (`rescore.py:320,1286,1302`) mean the boundary is still in flux.

PROPOSE (light):
- Do NOT rip BRP out. Reframe it: lc0 = training-target teacher, BRP = self-
  correction tracker + cheap teacher-free target when the model already fixes
  itself.
- Small decision to make together: with lc0 always-on in the new regime, do we
  keep BRP emitting training examples at all, or demote it to pure telemetry?
  That's the real question, not "delete BRP." Needs your call.
- Resolve the `TEMPORARY` markers before we lean on this boundary.

## P3. ~500k low-quality cold-storage games - mine or bury

WHY: user weighing whether to mine them.

STATE (found): NOT represented anywhere in code - no path, tag, format, or
count. Almost certainly off-repo on the HDD, most likely the same fat
`game_logs/<id>_log.pkl` logs (or legacy tfrec.gz). Cannot confirm from code.

PROPOSE (light):
- Before any mining decision we need a cheap census: point me at the HDD path and
  I'll sample N games, run them through the same prod-validation harness
  (`validate_prod_on_sources.py`) to get CE/polCE/MSE/corr vs the current sources.
- Decision rule: mine only the slice that clears a quality bar (e.g. corr(Q) and
  polCE within the prod band we already measured), and only after CPL<=30 +
  importance filtering (see P5). Bulk-bury the rest.
- Likely outcome: not worth mining wholesale; worth mining a filtered high-signal
  tail. But this is data-dependent - census first.

## P4. SP_DIR reorg -> family/run/ nesting

WHY: 50 flat run_tag dirs; naming already encodes `<family>_<run>`.

STATE (found):
- Canonical join is ONE line: `config.py:194` `init_paths` (all Config-derived
  subpaths hang off `run_dir`). Fixing that covers the happy path.
- BUT ~20 scripts bypass Config and re-hardcode `os.path.join(SP_DIR, run_tag)`.
  Highest risk: sites that `os.listdir(SP_DIR)` and prefix-filter run_tags
  (`scripts/bench/speedtest/fixtures.py:19`, `probe_table.py`, several data
  scripts) - a nested scheme breaks these unless they become a 2-level walk.
- Cross-run continuity sites need family-aware resolution:
  `blunder_replay.py:355`, `validation.py:114` (previous_run_tag).

PROPOSE (light):
- Introduce ONE resolver helper (e.g. `run_dir_for(run_tag)` in `__init__.py`
  or `config.py`) that maps a run_tag to its nested `<family>/<run>` path, with a
  parse rule (split on known family prefix, or a manifest). Route every
  hardcoded join and every listdir through it. Single choke point = manageable
  blast radius.
- Migration: a one-shot mover that walks current flat dirs, derives family, and
  `move`s each into `<family>/<run>/`. Reversible, dry-run first.
- Decide the family-parse rule together (prefix heuristic vs explicit manifest) -
  `18m_10c6t_SWA_selfplay6` -> family `18m_10c6t_SWA`? or `18m_10c6t`? Naming is
  ambiguous; your call.
- Do this AFTER the gen2 pretrain finishes (it writes into
  `18m_gen2_pretrain/`); don't move a live run's dir.

## P5. Detect + upsample high-importance positions

WHY: 134M records swamp critical positions; user hypothesis = CPL 0 + high KL.

STATE (found) - the signals ALREADY EXIST:
- `cpl` stored per ply (`migrate_game_logs_mp.py:518`; 0 = best move).
- `kl` stored per ply (`migrate_game_logs_mp.py:497`, visits||priors divergence) -
  exactly the "target diverges from prior" signal.
- KL-boost upsampling ALREADY implemented on the SELFPLAY retrain path: `pwht`
  boosted by `kl_boost_median_mult=1.5` / `kl_boost_p80_mult=3.0`
  (`rescore.py:1443-1447`), plus opening-repetition reweight.
- THE GAP is the PRETRAIN path only: `build_pretrain_shards.py` filters on
  CPL<=30 then DROPS cpl+kl into a weightless 4-tuple; trainer samples uniformly
  with `w=ones` (`bootstrap_model_async_sparse_pkl.py:344`).

REFERENCE (lc0, my own-ideas dig): lczero-training `tf/chunkparser.py:443-458`
implements "diff_focus" - rejection-sampling that keeps a record with prob tied
to `total = (q_weight*|best_q-orig_q| + pol_kld)/(q_weight+pol_scale)`. High
value-error OR high policy-KL = kept more often. This is the user's instinct,
already battle-tested in lc0. Note it needs model-vs-target Q diff + policy KLD
per record.

PROPOSE (light) - two flavors, pick later:
- (a) BUILD-TIME importance weight: carry a 5th field (importance) through the
  pretrain tuple, computed from stored cpl (0-bonus) + kl (high-bonus) at shard
  build. Trainer biases `np.random.choice` by it, OR feeds it as `pwht`/`vwht`.
  Cleanest reuse of existing selfplay weighting machinery.
- (b) LOAD-TIME diff_focus: port lc0's rejection sampler into `AsyncShardPool`
  fill - keep positions in proportion to importance, no schema change. But needs
  the model's live prediction (q diff) so it's a per-batch cost; cpl+kl-only
  variant avoids that.
- RECOMMEND starting with (a) using the already-stored cpl+kl (no model in the
  loop) - smallest change, reuses `pwht`. Escalate to diff_focus-style q-error
  only if cpl+kl alone underweights.
- CAUTION: this interacts with TODO #6 (prior_clip_max on retrain targets) and
  the opening-accumulator dedup - don't double-count opening-frequency reweight
  and importance weight.

---

## Own-ideas (beyond the 5 asked)

- O1. Importance field is a natural home for the cold-storage census (P3) too -
  one scoring function ranks any source, mined or historic.
- O2. When SP_DIR goes nested (P4), the historic->cold-storage drain (orig TODO
  #7) gets cleaner: cold storage becomes another top-level sibling of the run
  families, drained shards keep their family/run provenance.
- O3. `xerces_training` (tfrecord/TF path) can be fully retired once orig TODO #1
  (looper writes lean pkl) + the new sparse trainer land - it's the last TF
  tether. Flag for deletion, not now.

## Notes on original TODO list (brief, non-scoping)

- #1 (looper->lean pkl): fat logs at `<run_dir>/game_logs/<id>_log.pkl` are what
  the migration scripts consume; replacing them at write-time removes the whole
  migration step. Interacts with P5 - decide if the lean format keeps cpl+kl for
  importance (it should).
- #5/#6 (prior_clip_max<->uniform_eps): `prior_clip_max=0.75` in memory,
  `uniform_eps=0.25`. Unifying at MCTS prior-blend AND on stored retrain targets
  is coherent; just make sure P5 importance-weight is applied AFTER the clip so a
  clipped target isn't also sharpened.
- #2 (12/12/4 sizing): these are sample counts; reconcile with replay_buffer
  constants (144/144, live 204,800) which are caps, not draws.
- #8 (native 4-tuple in sp-retrain): P5 flavor (a) adds a 5th importance field -
  keep the retrain reader tolerant of both 4- and 5-tuple, or bump both together.
