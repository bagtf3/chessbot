# Item #2 Design: Native lean data writer (looper + rescore)

Goal: selfplay writes the lean per-game sparse format directly to `game_logs/`
and `game_index.json` -- the shape `migrate_game_logs_mp.py` currently produces
from fat logs -- so no post-hoc migration is ever needed again.

## Output artifacts (final names, no _tmp / _migrated)
- `<run_dir>/game_logs/<game_id>.pkl.gz` -- lean per-game dict (below).
- `<run_dir>/game_index.json` -- per-game index rows (as migration writes today).
- Fat log retained ONLY for reviewables: `is_review = (scenario ==
  'paired_validation') or (random.random() < review_rate)`. Reviewable games keep
  the full fat pkl in addition to the lean file; non-reviewable games get lean
  only.

## Who writes what
The migration is a pure transform of the fat log (no inference); split it across
the two places that already hold the inputs live:

- looper.py (has live board + per-ply `tree_data` from `collect_tree_search_data`)
  writes the search-derived fields at game end:
  `xc0h` (via `board.history_tokens(history_K)` captured per ply before push),
  sparse `policy` (unified-blended target), `raw_visits` (pristine sparse),
  `best_wdl` (STM-POV), `root_wdl_nn` (STM-POV, NEW), `z_wdl`, `kl`, `sel_method`,
  `xc0_move`, `stop_reason`, `stm`, plus review-only extras
  (`sims,time,avg_depth,max_depth,children_visited,total_children,candidate_moves,pv`)
  gated by `is_review`.
- rescore.py (post-hoc SF) adds `cpl` and computes
  `wdl_target = blend_wdl(z_stm, best_wdl)`, plus the lc0-distill enqueue decision
  -- the two things that need SF / final result.

## Per-ply schema (lean)
Same as migrated format, plus `root_wdl_nn`:
`stm, move_played, sel_method, xc0_move, stop_reason, best_wdl(STM),
root_wdl_nn(STM), z_wdl, wdl_target, xc0h, policy(sparse, blended),
raw_visits(sparse, pristine), kl, cpl`.
Keep both `policy` and `raw_visits` (sparse int16/fp16, cheap).

## root_wdl_nn (NEW plumbing)
- Already stored in C++: `node->value = wdl_white_pov` set once at expansion
  (`mcts.cpp:525`), a raw NN WDL 3-tuple, never accumulated, NOT vscaled.
- Surface with two lean POV getters on the node in pyfastchess bindings, no MCTS
  or storage change:
  - `nn_wdl_white` -> `node.value` as-is (white-POV)
  - `nn_wdl_stm`   -> `node.value` with win/loss swapped iff `get_stm_pov() < 0`
- Capture in `collect_tree_search_data`: `root_wdl_nn = self.tree.root().nn_wdl_stm`.
  Root is guaranteed expanded there (function early-returns otherwise).
- POV convention: draw is invariant; STM<->white is only a win/loss swap
  (`mcts.cpp:358/931`). `best_wdl` stays a Python flip keyed on known `turn` for
  now (getters later if desired).
- vscale isolation confirmed: `v_scalar = (win-loss)*vscale_` feeds only W/Q/Qema
  (`mcts.cpp:564`); never `p_win/p_draw/p_loss` or `node.value`. So `best_wdl`
  (= p_win/N) and `root_wdl_nn` (= node.value) are vscale-free.

## Unified uniform_eps + prior_clip_max blend (item #4, prerequisite here)
One algorithm, two implementations (C++ and Python, not shared code):
```
blend_to_uniform(p, uniform_eps, prior_clip_max):
    n = len(p)
    u = 1 / n
    eps = uniform_eps
    if prior_clip_max < 1 and n >= 5:
        top = max(p)
        if top > prior_clip_max:
            eps = max(eps, (top - prior_clip_max) / (top - u))   # exact clip
    return (1 - eps) * p + eps * u        # single pass; sum stays 1, no renorm
```
- C++: replace the two-step uniform-mix + hard-clamp + renorm (`mcts.cpp:999-1031`)
  with this single blend on the priors "when we get the priors."
- Python: same math on the normalized visit-count POLICY TARGET before writing to
  `game_logs` (retrain-ready). `raw_visits` is NOT blended.
- `n >= 5` gate everywhere.

## Sequencing
The unified blend is shared by #2's policy-target build and #4. Recommend landing
`blend_to_uniform` (C++ + Python) first, then #2 consumes it -- avoids writing a
soon-dead clip path.

## Open / deferred
- pyfastchess C++ getters need a rebuild of the .pyd (user builds).
- ChildDetail POV getters (for best_wdl) deferred; Python flip for now.
