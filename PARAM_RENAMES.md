# Search parameter renames (SearchParams cutover, 2026-11-13)

pyfastchess search parameters moved into one flat `SearchParams` struct
(`pyfastchess/src/mcts.hpp`, `SEARCH_PARAMS`). Config keys were renamed to
match the struct field names, so a config key that names a search param goes
straight into the tree. Runs from m137_8c8t_selfplay10 onward use the new
names; earlier runs, their configs and their game logs use the old ones.

## Config keys

| old | new | notes |
|---|---|---|
| uniform_eps | prior_uniform_eps | |
| reuse_tree | tree_reuse | |
| sharp_window [lo, hi] | sharp_lo, sharp_hi | |
| qmm_params [force, gate, gap_block] | qmm_force, qmm_gate, qmm_gap_block | |
| sims_ceiling as a dict | sims_schedule | sims_ceiling is always a scalar now |
| add_root_noise | removed | dirichlet_eps = 0 turns noise off |
| sample_moves | removed | move_sample_temp_plies = 0 turns sampling off |

Unchanged: c_puct (scalar or {ply: value}), sims_floor, sims_ceiling (scalar),
fpu_reduction, pruning_factor, vscale, prior_clip_max, dirichlet_eps,
dirichlet_alpha, sharp_M, sharp_boost, qema_span, qdelta_span, contempt_*,
tempscale_*, es_check_every, es_tier1_consec, es_tier1_jsd_thresh,
es_tier2_consec, es_tier2_jsd_thresh.

New config keys (previously hard-coded in C++, or new): es_tier1_dq12_floor,
es_tier1_qema_veto, es_observe_only, es_record_all, rsc_top_n,
rsc_min_visits, rsc_visit_share_floor, rsc_margin, recent_visits_span, qmm_min_visits, contempt_min_completed, tree_descent_cap,
tree_resort_thresholds.

## Per-move tree values

Python resolves these each move and sets them on the tree; they are not
config keys.

| old | new |
|---|---|
| sim budget (`set_sim_budget`, `sim_budget()`) | sims_max |
| early-stop min_sims (`min_sims()`) | sims_min |

## pyfastchess API

| old | new |
|---|---|
| `MCTSTree(board, c_puct, sim_budget, pruning_factor, uniform_eps, prior_clip_max)` | `MCTSTree(board, params=None, **kw)` |
| `set_cpuct`, `set_sim_budget`, `set_dirichlet`, `set_reuse_tree`, `set_vscale`, `set_contempt`, `set_tempscale_*`, `set_uniform_eps`, `set_prior_clip_max`, `set_fpu_reduction`, `set_sharp_boost`, `set_qmm_params`, `set_qema_span`, `set_qdelta_span`, `set_early_stop_params` | `set_params(params=None, **kw)` |
| `cpuct()`, `sim_budget()`, `min_sims()`, `dirichlet_eps()`, `reuse_tree()`, ... | `get_params(*names)` |
| `robust_selection_criteria(5, 60)` | `robust_selection_criteria()` (rsc_top_n, rsc_min_visits) |

`get_params()` also returns the derived read-only flags: noise_on_, es_on_,
sharp_on_, contempt_on_, qmm_gap_block_on_, pruning_on_, descent_cap_on_.

## Logged data

| where | old | new |
|---|---|---|
| game header / game_index | uniform_eps | prior_uniform_eps (sims_schedule added) |
| game_index / memlog sampled params | uniform_eps, reuse_tree, qmm_params ("force,gate,gap_block"), sharp_window ("lo,hi") | the new config keys above |

## Behaviour changes

- Early stop reads RSC with rsc_min_visits = 60 (it used 100; Python's RSC
  calls already used 60).
- The "full" stop at sims_max always applies. It used to apply only once
  early-stop params had been set, which selfplay always did.
