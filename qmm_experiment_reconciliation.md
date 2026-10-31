# Qmm Experiment Reconciliation

## Context
The val_test qmm study (round 1: 18 arms, round 2: 6 arms + dynamic mbs, about
32k self-play and 3k validation games) showed:
- The gap block helps.
- Force helps, but only together with the gap block.
- Gate is neutral.

Several companion changes landed on `qmm_revisited` in the same period: the
cliff-walking value fix, the KL fix, dynamic mbs, the sampling and noise
changes, and the logging upgrades.

This document scopes what stays, what is backed out, and the order of
operations for merging `qmm_revisited` back to `dev` in both repos (chessbot,
pyfastchess). The aim is to keep the wins and drop the curiosity-only
instrumentation, which carries a real performance cost.

**Locked settings:**
- Self-play: `qmm_params: [true, 3, 0.05]`.
- Validation: `qmm_params: [true, 5, 0.03]`.

## Sequence
1. Finish validation 495. Run the final full splits and confirm the picks hold.
2. Do a final pass on the probe data, then delete the four probe jsonls in
   val_test (about 8.6 GB): `qmm_probe*.jsonl`, `qmm_probe_val*.jsonl`.
3. Stop the run. Apply everything below in both repos and rebuild pyfastchess.
4. Update the val_test `config.yaml` and `validation_config.yaml` (see Configs).
   Restart only when every yaml and config is correct.
5. Run a few epochs in val_test on the reconciled code and confirm nothing
   moved.
6. Merge to `dev`, then take the setup to sp7.

## Keep

### pyfastchess
- **Qmm core:**
  - `n_done`;
  - `Qmm`, `Qmm_second`, `Qmm_best_packed`, `Qmm_is_set`;
  - `qmm_cand`, `qmm_rescan`, `qmm_update`;
  - `kQmmMinVisits`;
  - the backprop hook in `back_up_along_path_nolock`.
- **Params:** `QmmParams` + `set_qmm_params(force, gate, gap_block)`.
- **Force:**
  - the `qmm_force_packed` slot;
  - the force branch in `select_child_lazy_ptr`;
  - mate must-visit priority over a pending force.
- **`on_qmm_new_best`:** trimmed to set the force only, with no stats.
- **`forced_N`** on the node and in `ChildDetail`. Forced visits must stay
  out of the policy target.
- **`ChildDetail.Qmm`:** the gap block reads `details[0].Qmm`, and the viewer
  shows a Qmm column.
- **Gap-block veto** in `evaluate_early_stop`.
- **`raw_prior`** on `ChildDetail` + its binding.
- **`effective_mbs`** + its binding.
- **Root noise schedule:** 1.5x eps for plies 0-12, full 13-30, half 31-60,
  none 61+. Keep it hard-coded. Add a comment that the ratio and ply splits
  may be parameterized later.

### chessbot
- `qmm_params` config + the `set_qmm_params` call (`config.py`,
  `mcts_utils.py`).
- Policy target uses `cd.N - cd.boosted_N - cd.forced_N` (`mcts_utils.py`).
- `P_raw` in candidate dicts (`mcts_utils.py`).
- **Sampling** (`mcts_utils.py` `best()`):
  - `SAMPLE_TOP_K = 7`;
  - the leader is always a candidate;
  - other candidates need `N >= 1`.
- **Cliff-walking fix:** `deviation_outcomes`, `blend_wdl_stm(outcome_wdl,
  best_wdl)`, `value_bootstrap_at_sample` (`rescore.py`, `config.py`).
- **KL fix:** both rescorer KLs on `P_raw` and both KL(visits || prior)
  (`rescore.py`). The viewer KL and the `kl > x` seek match (`review.py`).
- **Logging:**
  - the new loop block layout;
  - `LOG_WIDTH = 79` dividers;
  - `avg plies` and `draw=%`;
  - `calls/s`, `short`, `2x calls`, `n_preds`;
  - the `mbs_used` window fix;
  - `COLLECT_DESCENT_CAP` (`looper.py`, `review.py`, `utils.py`).
- **Viewer:** `P_raw` column (`review.py`).
- **Dynamic mbs:** `dynamic_mbs` + the exact-fill loop (`config.py`,
  `looper.py`). Fixed `true` in the yamls; the `config.py` default stays
  False.
- **Other branch items:**
  - the `use_blunder_replay` flag;
  - `init_new_run.py --keep-source`;
  - the selfplay_runner fix that never re-validates an epoch already in
    history.

## Back out

### pyfastchess (curiosity only, real performance cost)
- `QmmStats` and its binding. Replaced by the gap-block counter below.
- `root_leader_`, `reset_root_leader()` and the leader-flip tracking in
  backprop. This runs on every root backup.
- `forced_by_depth_`, `kForcedDepths`, the per-descent bucket and the
  `forced_by_depth` binding.
- `CollectCounts.count_qmm_forced` / `count_qmm_overridden`.
- Node `qmm_new_bests` and its reset in `seed_new_root_plus_one`.
- `ChildDetail` `qmm_new_bests`, `qmm_gap12`, `qmm_depth` and
  `qmm_line_stats()`. This rescans grandchildren and walks the Qmm chain on
  every `root_child_details()` call, including every early-stop check-in.

### chessbot
- The qmm probe collector: `write_qmm_probe`, the probe counters,
  `QMM_PROBE_GAMES`, `QMM_PROBE_FILENAME`, `QMM_PROBE_VAL_FILENAME`
  (`rescore.py`).
- The per-ply `qmm_stats` dict (`mcts_utils.py`).
- Candidate-dict `qmm_new_bests`, `qmm_gap12`, `qmm_depth`, `forced`
  (`mcts_utils.py`). Keep `Qmm`.
- `played_pv_len`. Replaced by the PV upgrade below.
- The `qmm_params` memlog split (`looper.py`) and the `sampleable_cols` entry
  (`explore.py`). qmm is no longer sampled.
- `validation_freeze_depth` (`config.py`, `validation.py`).
- `SAMPLE_MAX_Q_DELTA = 0.2` -> **0.4**. This is a loosening rather than a
  removal: it still blocks outright hangs while allowing exploration.
  Revisit tightening once opening diversity recovers, using the 1.e4 share,
  distinct replies to 1.d4 and the startpos draw %.

## Add

### Gap-block counter (per move, real instances only)
- **C++:** a small per-move record, reset in `advance_root`.
  - `gap_block_sims`: root visits at the **first** veto, 0 if none.
  - `gap_block_leader`: the most-visited root move at that veto.
  - Written once, at the first tier1/tier2 stop vetoed by the gap block.
    Ceiling stops are never vetoed.
  - Exposed through one binding.
- **Python:** written per ply, as `data["gap_block_sims"]` and
  `data["gap_block_leader"]`.
- **Lean record:** added to the per-ply keys in `write_lean_record`, so every
  game carries them.
- **Analysis use:**

  | Question | Read |
  |---|---|
  | Did a block happen? | `gap_block_sims > 0` |
  | How long did it extend search? | `sims - gap_block_sims` |
  | Did it change the move? | `gap_block_leader != xc0_move`. Compare with `xc0_move`, not `move_played`, so sampling isn't miscounted. |

- **Naming:** `gap_block_*`, kept distinct from the console `[puct]
  blocked=`, which counts in-flight descent collisions.

### PV upgrade (`mcts_utils.py`, `rescore.py`, `review.py`)
- **Cap:** both PVs extend from 24 to 64 plies. Validation lines regularly go
  past 24.
- **`pv`:** the most-visited chain from Xerces's choice (`xc0_move`, else the
  played move). Every step has uci / visits / P / Q, as today.
- **`pv_played`:**
  - When `move_played != xc0_move`, the chain under the move actually
    played, in the same format.
  - The reason can be sampling, an SF move on SF's turn, or anything else
    that makes them differ. SF turns follow the same rule.
  - When the two moves are equal, there is only `pv`, and `pv_played` is
    absent or None.
- **Storage:**
  - Full lists for reviewable games: all validation games plus the review
    sample. `pv` is already in the reviewable extras; add `pv_played`.
  - Every game also stores `pv_len` and `pv_played_len` (None when equal)
    in the per-ply lean keys.
- **Viewer `show_pv`:**
  - Prints `pv` as today.
  - When `pv_played` is present, prints it as a second, labeled block (e.g.
    "played line").
  - The played-move star marking works the same way in both blocks.
- **Cost:** two PV walks per move only when the moves differ. It gets turned
  off for TCEC.

### Viewer: any flagged move outside the top 5 (`review.py` `show_moves`)
- **Today:** the Xc0 move and the SF best get the `...` break row when they
  rank outside the top 5. The sampled (played) move does not, and sampling
  now allows up to rank 7.
- **Change:** generalize. Any flagged move outside the top 5 gets the
  existing `...` treatment:
  - the SF best;
  - the sampled/played move;
  - the Xc0 move.
- Rows are deduplicated and ordered by visit rank.
- `need_rank` also covers the sampled move.
- The flags keep their current text: `<-Xc0`, `<-sampled`, `<-SF,samp`,
  `<-Xc0, SF`.

## Configs (val_test first, then sp7)
- **`config.yaml`:**
  - `qmm_params: [true, 3, 0.05]` as a fixed value, removed from
    `sampleable`.
  - `dynamic_mbs: true` as a fixed value, removed from `sampleable`.
  - Keep `value_bootstrap_at_sample: true` and
    `move_sample_temp_range: [0.3, 2.25]`.
- **`validation_config.yaml`:**
  - `qmm_params: [true, 5, 0.03]`.
  - `dynamic_mbs: true`.
  - Drop `validation_freeze_depth`.
- **Moving to sp7:** don't carry the val_test-only settings.
  - `use_blunder_replay: false`;
  - `n_lc0_workers: 0`.

## LATER (not in this reconciliation)
- Parameterize `dynamic_mbs` (threshold, multiplier, a 2-4-8-16 ramp) and
  sweep-test it.
- Parameterize the root noise ratio and ply splits.
- Revisit `SAMPLE_MAX_Q_DELTA` once opening diversity recovers.
- Stretch: the viewer flags the gap-block leader (`gap_block_leader`) when it
  differs from the final move.

## Verification (during step 5)
- The pyfastchess build succeeds, and `set_qmm_params` plus the gap-block
  binding are present. `qmm_stats` and `forced_by_depth` are gone.
- Console: the loop block renders with the 79-wide layout, and `2x calls`
  is above 0.
- No `qmm_probe*.jsonl` files are written.
- A fresh lean record has, per ply:
  - `gap_block_sims`, `gap_block_leader`, `pv_len`, `pv_played_len`;
  - in a validation game, `pv` up to 64 long and `pv_played` on sampled or
    SF-differs plies.
- In the viewer, on a game where the sampled move ranks 6 or 7, the break
  row shows it. `pv` prints both lines.
- The splits/analysis from the game logs show blocked %, extra sims and the
  move-changed rate, with numbers in the same range as the study.
