# Search controller in C++ (xc0 / pyfastchess)

## Context
The K5L6 search controller (production model: `controller_M_two_head_1024.pt`,
two sigmoid heads, stability and quality) decides at every 100-root-visit
check-in whether a search should stop, continue or extend. Today it only
exists in PyTorch and in offline replays. Self-play must never pause for
Python mid-search, so the controller has to run inside C++ search: at each
check-in, build the 64 check-in features (already in C++,
`build_check_in_features`), feed them plus the cached 48 context features
to the model through the ONNX Runtime C++ API (CPU, cached causal step
graph), and act on the result before the next descent.

Outcome: one `collect_many_leaves` loop that stops, extends or continues
by itself; Python only reads `es_tripped` / `es_stop_reason` as it does
today. Correctness is tested; speed is judged by GPH/MPS when self-play
restarts.

Decisions taken: ORT C++ API (CPU EP), not hand-written kernels. Self-play
aligns qema/qdelta spans (160/160) and fpu (0.2) with training and keeps
noise and tree reuse.

vscale applies only at PUCT selection; every stored and exported
statistic is raw (see "Selection-only vscale" in the build notes).
Self-play runs at `vscale: 1.0`, matching the training data.

## Repos and key files
- pyfastchess (C++): `src/mcts.hpp`, `src/mcts.cpp`, `src/binding.cpp`,
  `CMakeLists.txt`, `pyproject.toml`, `src/pyfastchess/__init__.py`, `tests/`
- chessbot: `src/chessbot/config.py`, `src/chessbot/mcts_utils.py`,
  `src/chessbot/looper.py`, `src/chessbot/rescore.py` (lean log pass-through),
  new `src/chessbot/experimentation/export_controller.py`

## 1. ONNX Runtime C++ in the build (pyfastchess)
- Fetch the official `onnxruntime-win-x64-1.18.0` CPU release zip (matches the
  installed onnxruntime-gpu 1.18.0) into `third_party/onnxruntime/`
  (headers, `onnxruntime.lib`, `onnxruntime.dll`). Git-ignored; a short README
  line says where it comes from.
- `CMakeLists.txt`: `ORT_ROOT` cache var defaulting to that path; include dir,
  link `onnxruntime.lib`, `install(FILES onnxruntime.dll DESTINATION
  pyfastchess)` so it sits next to `_core.pyd`.
- `src/pyfastchess/__init__.py`: `os.add_dll_directory(<package dir>)` before
  importing `_core`, so Windows loads this onnxruntime.dll, not another copy on
  PATH. The in-process onnxruntime-gpu Python module statically links its own
  core, so the two do not clash.

## 2. Export (chessbot) - `experimentation/export_controller.py`
`python -m chessbot.experimentation.export_controller <ckpt.pt>` writes
`<ckpt stem>.ctrl/` with:
- `init.onnx`: context (1,48) -> p (1,128), FiLM scale/beta (6,2,128).
- `step.onnx`: inputs check-in (1,64), p, FiLM, per-block conv caches,
  running sum, count -> outputs logits (1,2), new caches, new sum/count. Built
  per `search_controller_model.md` "cached causal step graph": `v @ Wt`, 2-D
  tensors so MatMul+Add fuse into Gemm, FiLM precomputed in init.
- InputScale folded into the first Linear of each input path.
- `meta.json`: spec, width, k, dilations, mixers, cache shapes, heads
  (`two_head` / `quality_only`), `feature_version` (from the checkpoint's
  collection metadata, `context48-v1/checkin64-v2`), check-in interval 100,
  context and check-in feature names.
- Built-in parity check: run the exported init+step through Python
  onnxruntime (CPU EP) over 200 real test trajectories and assert max |logit
  difference| vs the PyTorch full causal forward < 1e-4; also write a small
  `parity.npz` fixture (context, check-ins, expected logits) for the C++ test.

## 3. Controller runtime in C++ (pyfastchess)
New `src/controller.hpp/.cpp`:
- `ControllerModel`: one process-wide `Ort::Env` and two `Ort::Session`s
  (CPU EP, intra/inter-op 1 thread, sequential, `ORT_ENABLE_ALL`). Loaded
  once via `pyfastchess.load_search_controller(dir)`; validates meta against
  compiled constants (`NUM_CHECK_IN_FEATURES`, `NUM_SEARCH_CONTEXT_FEATURES`,
  feature names, `CTRL_FEATURE_VERSION`, interval 100) and fails loudly on any
  mismatch. Held in the existing singleton registry.
- `ControllerState` per tree: p, FiLM, ping-pong conv caches, sum, count
  (about 28.5 KB). `init(context)` runs the init graph; `step(checkin) ->
  (stab, qual)` runs the step graph with OrtValues created over the state
  buffers (zero copy) and swaps the ping-pong buffers. Reset per search.

## 4. Check-in hook and stop rules (pyfastchess `collect_many_leaves`)
When `ctrl_enable` is on, the per-descent block at `mcts.cpp:609-619` becomes:
1. Check-in boundary: root N reaches the next multiple of 100 (absolute, as
   in training on fresh trees). With an inherited root, the first boundary is
   the next multiple above the inherited N, and `ctrl_.checkins` starts at
   `floor(inherited N / 100)` so the check-in-count feature agrees with root N.
2. At a boundary: build features with the existing `build_check_in_features`
   plus a lock-free `ctrl_` commit (factored out of
   `search_check_in_features`, which keeps its locking wrapper for Python).
   First boundary: `ControllerState::init(root context)`. Then `step()`,
   apply the rules below, record if `ctrl_record`.
3. Ceiling: `sims_max` is a whole number of check-ins, so the check-in at
   sims_max makes the stop-or-extend call; a budget stop (`full`) is only a
   backstop.

Rules (SEARCH_PARAMS, see the ctrl comment in `mcts.hpp`):
- No stop below `ctrl_min_stop`. `sims_max` is rounded up to a multiple of
  100 and never set below `ctrl_min_stop`.
- Leader: one ranking everywhere (`ranked_children`): visits, then STM Q
  (W / n_done), then the visit-share EMA ticked to the current root visit,
  then prior, then list order. `best()`, `root_child_details`, the PV, the
  check-in top-4 slots and the lock all use it. The move played is always
  this leader.
- Agreement: stability >= `ctrl_s_thresh` and quality >= `ctrl_q_thresh` on
  `ctrl_agree_consec` straight check-ins with the same leader -> `ctrl_agree`.
  The streak counts from the first check-in.
- Lock: leader N - runner-up N > sims_max - root N (at sims_max every
  non-tied lead counts; a tie is a stop point too). Accepted (`lock`) unless
  D < `ctrl_ext_d_max` and quality < T(|Q|), T = `ctrl_ext_q0` for |Q| <=
  `ctrl_ext_absq0`, then falling by `ctrl_ext_slope` per unit of |Q| with no
  cut-off. |Q| and D come from the root's raw WDL sums / n_done (no vscale,
  no virtual loss).
- Extension: sims_max += `ctrl_ext_sims`; the leader is the rejected move;
  streaks reset. After it, agreement stops (`ctrl_agree_ext`) after
  `ctrl_agree_consec` on a different leader, or `ctrl_rejected_consec` on the
  rejected one.
- Re-lock: a different leader stops (`relock_new`); the rejected move with
  higher quality than at rejection stops (`relock_better`); a lead of at least
  `ctrl_big_lead` stops (`relock_lead`); otherwise one emergency extension of
  max(`ctrl_ext_sims`, 2 x lead), rounded up to 100. The next lock always
  stops (`final_lock`).
- Off on Stockfish's turns in vs-SF games (`ChessGame.sync_controller`), with
  no recording.
- `ctrl_enable` with `es_check_every > 0` is a validation error (one stop
  system at a time); `ctrl_enable` without a loaded model is an error.
- `checkin_break` (default false): `collect_many_leaves` returns right after
  any check-in boundary, decided or not, for callers that act at exact
  boundaries. Self-play leaves it off and keeps rolling.
- Reasons surface through the existing `es_tripped` / `es_stop_reason`.

Recording (`ctrl_record`, per tree, default false): per check-in, the 64
features, stab, qual, root N, leader (packed move), root Q and a rule
code; the 48 context features once; plus the stop reason. `tree.get_ctrl_record()` returns a dict of numpy arrays (features as
float16). Cleared by `reset_early_stop`.

## 5. Python integration (chessbot)
- `config.py`: `ctrl_model` (path to the `.ctrl` dir, "" = off) plus defaults
  for every new SearchParams field (`ctrl_*`, `checkin_break`). They reach
  C++ through the existing `SEARCH_FIELDS` pass-through in
  `mcts_utils.py:20`.
- Worker start (`looper.py`): if `ctrl_model` is set,
  `pf.load_search_controller(path)` once per process.
- `mcts_utils.stop_simulating` is unchanged: reasons arrive as
  `es_stop_reason`. With the controller on, RSC selection is skipped; the
  most visited move is played and sampling works downstream as before.
- Logging: `ChessGame.collect_tree_search_data` adds `ply["ctrl"] =
  tree.get_ctrl_record()` when `ctrl_record` is on; `rescore.py`'s lean log
  conversion keeps the `ctrl` key.
- Self-play config (user applies): `ctrl_model`, `ctrl_enable: true`,
  `es_check_every: 0`, `qema_span: 160`, `qdelta_span: 160`,
  `fpu_reduction: 0.2`, `vscale: 1.0`, chosen ctrl thresholds.

## Verification (correctness only)
Before any change, record golden outputs from the current build: a few
deterministic stub-eval searches (pattern of `tests/test_early_stop.py`), per
child N and Q.

New `tests/test_search_controller.py`, stub eval, no GPU:
- Controller off: per-child N and Q identical to golden (the hook adds no
  behaviour change when disabled).
- Model parity: load the exported `.ctrl` in C++ and replay `parity.npz`
  through a debug binding; logits match within 1e-4.
- Feature parity: two identical deterministic searches, one driven by the
  old Python boundary path (`search_check_in_features` at exact boundaries),
  one recording through the C++ hook; the 64 features match exactly.
- Rules: `ctrl_q_thresh` 0 stops at the first eligible check-in (`ctrl_q`);
  1.01 never stops; consecutive count respected; the lock agrees with the
  collector's `dominance()` on the same visit counts; extension fires once,
  raises the ceiling by `ctrl_ext_max`, ends on `ext_q` or `ext_full`;
  `checkin_break` returns at every boundary; inherited-root boundaries and
  check-in counts are as specified.
- Validation errors: `ctrl_enable` with `es_check_every > 0`; without a model;
  a model whose meta mismatches the compiled features.
- Existing pyfastchess tests still pass; chessbot imports and one short
  self-play smoke game run with the controller on.
- Speed is not tested here; it shows in GPH / MPS when self-play restarts.

## Build notes
Running log of what was done, decisions made during the build, and anything
to review together. Newest last.

- Golden fixture (step 0): `pyfastchess/tests/search_harness.py` (stub eval
  keyed by zobrist, forest loop) and `tests/make_golden_search.py` recorded
  `tests/data/golden_search.json` from the pre-change build (commit 0aa5dbb,
  clean tree): 5 searches (startpos, a middlegame, KPK; tiers on/off; vscale
  0.9), per-child N / n_done / Q. Two runs matched exactly before writing.
- ORT SDK: official `onnxruntime-win-x64-1.18.0` (CPU) unpacked to
  `pyfastchess/third_party/onnxruntime` (git-ignored, fetch steps in
  pyfastchess README). CMake: `ORT_ROOT` cache var, fatal error if missing,
  links `onnxruntime.lib`, installs `onnxruntime.dll` next to `_core`.
  `pyfastchess/__init__.py` adds `_core`'s directory (site-packages in the
  editable install, not the source dir) to the DLL search path.
- Export: `chessbot/experimentation/export_controller.py`. M2H1024 exported to
  `controller_M_two_head_1024.ctrl/` (init.onnx 1.2 MB, step.onnx 2.6 MB,
  meta.json, parity.npz). InputScale folded into the first Linear of both
  input paths. Parity vs the PyTorch full causal forward over 200 held-out
  test trajectories: max |logit diff| 1.05e-05 (tolerance 1e-4).
  Opset 17, so LayerNorm is one node.
- Review item: step.onnx has 204 nodes, 95 of them Constants (slice
  indices, folded by ORT at load). Python `session.run` measures ~244 us per
  step including Python overhead; the design doc's tuned C++-style figure was
  ~180 us. Possible later trim: emit FiLM per block as separate init outputs
  and avoid strided slices, if the per-check-in cost matters in GPH/MPS.
- C++ runtime: `pyfastchess/src/controller.hpp/.cpp`. One process-wide model
  (a module-level `unique_ptr` in controller.cpp rather than the singleton
  registry: it's one object, and keeping it out of the cache registry keeps
  that file about caches). `load_controller` checks the graphs' I/O names and
  shapes against the export layout. `ControllerState` double-buffers the
  step state and wraps it in OrtValues with no copies.
- meta.json is validated in Python (`pyfastchess.load_search_controller`):
  feature version, check-in interval, context and check-in feature names,
  heads. Keeps a JSON parser out of the C++.
- C++ vs PyTorch parity on the exported fixture: max |prob diff| 2.4e-07.
  Step cost through the C++ path: ~179 us per check-in (matches the design
  doc's ~180 us).
- Controller off: all 5 golden searches identical (per-child N, n_done, Q,
  stop reason) to the pre-change build.
- Hook: in controller mode the check-in at a boundary runs before the
  sims_max check, so a stop at sims_max uses that check-in's quality. The
  old (non-controller) path is unchanged.
- Plan change (review): the extension raises `params_.sims_max` on the tree
  instead of tracking a separate effective ceiling. Root pruning, the budget
  stop and `effective_mbs` all read sims_max, so a separate ceiling would
  have pruned nearly everything during the extra search. Python already sets
  sims_max every move (`advance()`), so an extension never leaks into the
  next move. `get_params("sims_max")` therefore shows the extended budget.
- Rule order at a check-in (from `ctrl_min_sims`): while extended, quality
  >= `ctrl_ext_resume` stops (`ext_q`); otherwise the quality streak
  (`ctrl_q`), then the decided-position stability rule (`ctrl_s`). The lock
  is checked after those and goes through the same extension decision as
  sims_max. The quality streak only counts check-ins at or past
  `ctrl_min_sims`, as in the replays.
- Recording: `tree.get_ctrl_record()` -> context float32[48], features
  float16[T,64], probs float32[T,2], root_n, root_q, rule code (names in
  `pyfastchess.CTRL_RULES`), leader uci, stop_reason.
- Validation errors: ctrl_enable without a loaded model; ctrl_enable with
  es_check_every > 0; ctrl_decided_q > 0 with a quality-only model;
  ctrl_q_consec < 1 or ctrl_ext_max < 0.
- chessbot: `config.py` gets `ctrl_model` plus defaults for all 13 new
  SearchParams fields (they flow to C++ through `SEARCH_FIELDS` like every
  other search param). `GameLooper.__init__` loads `ctrl_model` once per
  worker process. `collect_tree_search_data` stores `ply["ctrl"] =
  tree.get_ctrl_record()` when `ctrl_record` is on; the lean log writer
  (`rescore.write_lean_record`) keeps `ctrl` when present.
- Note: the per-ply `sims_max` in game logs reads the tree's sims_max, so on
  an extended search it shows the extended budget.
- Tests (`pyfastchess/tests/test_search_controller.py`, 14 tests, all pass):
  controller-off golden match; C++ vs PyTorch parity; check-in features from
  the C++ hook identical to the Python boundary path (`search_check_in_features`
  at exact multiples of 100), descent for descent; quality rule (first
  eligible check-in, consecutive count); unreachable threshold runs to
  sims_max with one check-in per 100; lock is exact dominance and only below
  sims_max; a check-in landing on sims_max is `full`, not `lock`; extension
  raises sims_max once and ends on `ext_full` or `ext_q`; `checkin_break`
  never runs past a boundary; inherited roots check in at the next multiple
  of 100 with a matching check-in number; validation errors (es_check_every,
  ctrl_q_consec, meta mismatch, no model in a fresh process).
- Full pyfastchess suite: 68 pass, 1 fails:
  `test_early_stop.py::test_tier1_fires_on_a_mega_runaway_with_degenerate_rsc`.
  It fails identically on the original code (checked by stashing the changes,
  rebuilding and rerunning), so it predates this build. Not touched.
- Bug found by the smoke game and fixed: a check-in landing exactly on
  sims_max made the lock test trivially true, so budget stops were reported
  as `lock`. The lock now only counts below sims_max.
- chessbot smoke game (stub eval, selfplay10 config with the controller
  settings below, 12 plies): config reaches C++, stops, extension, `best()`,
  `advance()` and `ply["ctrl"]` records all work end to end. With a random
  stub eval quality stays low, so every move extended then locked; real
  stop behaviour needs the real network.
- Tree reuse: `m137_8c8t_selfplay10/config.yaml` has `tree_reuse: false`, so
  self-play searches start fresh each move, exactly like the training data.
  The inherited-root handling only matters if reuse is turned back on.

- Rule set replaced (2026-11-16) with the spec in section 4. Old params
  (`ctrl_min_sims`, `ctrl_q_consec`, `ctrl_decided_q`, `ctrl_lock`,
  `ctrl_ext_max`, `ctrl_doubt*`, `ctrl_ext_resume`) are gone; a two-head
  model is now required. Rule codes: none, ctrl_agree, lock, extend,
  ctrl_agree_ext, relock_new, relock_better, relock_lead, emergency,
  final_lock.
- Replay of the new rules on M2H1024's 20480 test trajectories
  (`experiments/replay_new_stop_rules_M2H1024.csv`):
  - sims 1309 vs xc0 1736 (24.6% saved); TSL 14.40 vs 14.64; SF-good 86.3%
    vs 85.9%; blunders vs xc0 0.03%.
  - endings: q_stop 43.8% (540 sims), lock_accept 54.5%, extensions 1.7%
    (1.4% one, 0.3% two).
  - extended positions resolved in the data: TSL 56.8 vs xc0 87.4, +2391
    sims (data past xc0's ceiling only exists where SF rejected the move).
  - saved / dTSL by stratum: random 27.8% / -0.14, high_cpl 9.3% / -1.06,
    mate_found 60.3% / -0.05, refutation 4.7% / -0.46.
- Tie-break: `ranked_children(node, k)` is read-only and does not reorder
  `ordered_children`. Its visit-share key matches the ticked EMA from
  `update_visit_share` exactly (checked against `root_child_details`).
- Tests after the rule change: 69 pass, the same 1 pre-existing failure.
  `test_tier2_fires_on_a_two_horse_race` now also accepts tier1, since exact
  visit ties no longer flip the leader.

- No tree reuse with the controller: `advance_root` keeps no subtree while
  `ctrl_enable` is on, including the forced matelock reuse (|Q| > 0.9). The
  controller only knows searches from a fresh root; a reused root has
  check-ins it never saw (leader history, visits past the ceiling).
  `ChessGame.push_move` sets `ctrl_enable` for the side about to move before
  the advance, so Stockfish's turns keep the old reuse and their sims budget
  is not rounded to check-ins.
- Controller load order: `GameLooper.__init__` loads `ctrl_model` before the
  first `pull_from_queue`, once per process, so a non-empty startup queue no
  longer builds controller trees with no model. It is no longer reloaded on
  every model reload.

- Telemetry: C++ keeps a per-search `CtrlSummary` (`tree.get_ctrl_summary()`:
  check-ins, ms inside check-ins, sims_max before extensions, ext, rejected
  move, latest stab/qual, tied check-ins), with or without `ctrl_record`.
  `collect_tree_search_data` puts it on the in-memory ply as `ctrl_sum`; the
  lean log does not keep it. Rescore prints a controller section every
  `CTRL_WINDOW_GAMES` games (stop mix with sims / TSL / CPL / blunder rate per
  ending, sims vs schedule, extensions, ms and check-ins per move, stab/qual
  at the stop, tie rate) and then drops the window. With the controller on,
  the tier1/tier2/full table prints only its KL lines.

- Selection-only vscale: backups add the full WDL to `p_win/p_draw/p_loss`,
  raw win-loss to `W`, and `Q = W / n_done`; `Qema`, `Qdelta_sign`, `Qmm`,
  `Q_eff`, ChildDetail, logs and controller features are all raw. Each node
  also splits its win/loss sums by the evaluated leaf (network `*_nt`,
  terminal `*_t`, classified by the leaf for every node on the path),
  internal only. PUCT reads `MCTSNode::select_q(vscale)` =
  (vscale x (win_nt - loss_nt) + win_t - loss_t) / n_done, minus the node's
  contempt shift (Q - Q_eff); the parent's FPU base uses the same value. So
  the order is: raw value -> vscale on the network part -> contempt ->
  FPU / PUCT. Contempt latches on raw root Q. At vscale 1 `select_q` returns
  `Q_eff` exactly, so default searches are bit-identical. Changing vscale
  changes selection at once and no stored value. Tests:
  `pyfastchess/tests/test_vscale_selection.py` (8); the golden vscale 0.9
  search keeps identical visits, with Q now raw (golden Q / 0.9).
- vscale gate: `vscale_gate_q` (0 = vscale always on). Defaults (C++ and
  config.py): vscale 0.9, gate 0.9. When set, each
  descent uses `tree.active_vscale()`: vscale while the root's raw |Q| is
  above the gate, else 1. Since stored stats are raw, the switch can flip
  mid-search safely. val_test runs `vscale: 0.9`, `vscale_gate_q: 0.9` in
  selfplay, validation, BRP and lc0.

- Extensions widen the root: the first extension and the emergency both set
  `pruning_factor` to `ctrl_ext_pruning` (when pruning is on and higher);
  `reset_early_stop` restores the saved value, so the next move runs at its
  configured factor. Tests cover the restore through advance + set_params.
- val_test `sims_ceiling: 2000`: effort scale gives 1000-4000 per move.

### Review items
1. Cost per check-in is ~179 us of CPU on the worker's main thread (the
   collect loop is synchronous). At ~30 check-ins per move that is ~5 ms per
   move per game. Whether the saved sims outweigh it shows in GPH / MPS.
   If it doesn't, the step graph can be trimmed (see the export note) or the
   check-in could move off the hot path.
2. RSC is off with the controller; every stop plays the leader.
3. val_test config (set 2026-11-16): `ctrl_model`, `ctrl_enable: true`,
   `es_check_every: 0`, `use_robust: false`, `qmm_gap_block` off; rule
   params at their defaults. `ctrl_record: true` on some or all games to
   collect retraining data.
4. Nothing is committed yet, in either repo.
5. Selection-only vscale changes forced matelock reuse slightly: its
   |Q| > 0.9 test now reads raw Q. That reuse is off with the controller on.
6. The export only covers models with the M/L layout from
   train_controller.py; a quality-only model exports the same way and sets
   `heads: quality_only` (the stability rule then refuses to enable).

### Files
- pyfastchess: `CMakeLists.txt`, `README.md`, `.gitignore`,
  `src/controller.hpp`, `src/controller.cpp` (new), `src/mcts.hpp`,
  `src/mcts.cpp`, `src/binding.cpp`, `src/pyfastchess/__init__.py`,
  `tests/search_harness.py`, `tests/make_golden_search.py`,
  `tests/data/golden_search.json`, `tests/test_search_controller.py` (new).
- chessbot: `src/chessbot/experimentation/export_controller.py` (new),
  `src/chessbot/config.py`, `src/chessbot/looper.py`,
  `src/chessbot/mcts_utils.py`, `src/chessbot/rescore.py`, this file.
- Exported model: `D:/Data/chessbot_data/experiments/collections/
  ctrl_collect_150k_20261115_104609/controller_M_two_head_1024.ctrl/`.

## Future thoughts (not planned work; review before acting)

### Is the synchronous check-in a throughput problem?
Today every check-in runs inline in `collect_many_leaves`: ~180 us per step,
and the worker loop waits on it.
- Average cost is small: ~1 check-in per 100 leaves, so at ~6k leaves/s per
  worker that is ~60 check-ins/s, ~11 ms per second, ~1%.
- Bursts are the risk: a cache-hit run at ~120k leaves/s means ~1,200
  check-ins/s, ~20% of the loop for the length of the burst.
- First step if it matters: a per-worker counter of time spent in controller
  steps vs wall time, measured in val_test. Under ~3% overall, nothing more
  is needed.
- Cheapest fix if it does matter: trim the step graph (~200 nodes at ~1.7 us
  of ORT overhead each; the math itself is ~650k multiply-adds). Could roughly
  halve the step.

### Controller as an oracle, with non-blocking check-ins
The controller owns each search's causal state, keyed by a per-tree ID.
Trees send each check-in's features as a request and read the head outputs
back when they need them.
- Blocking rule: a check-in blocks only if its eval could stop the search.
  - Below `ctrl_min_sims`: never blocks; the evals still have to run, in
    order, to build the causal state for later check-ins.
  - Quality streak needing `ctrl_q_consec` approvals: check-in k can only
    stop if the previous k-1 approvals are in. If an earlier one is still
    pending, k has to wait for it anyway (results apply in causal order).
    So after the floor the non-blocking window is mostly the check-in right
    after a known reject.
  - Lock and sims_max are known without the model; the extension decision
    only blocks at the check-in where one of them is reached.
- Non-blocking: queue the request and keep searching; poll for the result
  later (before the next check-in that needs it).
- Blocking: queue the request, stop collecting and return from
  `collect_many_leaves`, so the rest of the forest keeps collecting; pick the
  tree back up on a later pass once its results are in.
- Overfill `games_at_once` so blocked trees don't leave the batch short.
  Per-game speed may dip slightly, offset by the sims the controller saves;
  overall throughput is held.
- The big lever: run the controller on its own C++ thread (no GIL) so the
  queue drains while the worker waits on GPU inference each pass. Blocking
  evals are then usually done by the next pass, so the stall is near zero.
- Costs: request queue, per-tree FIFO with in-order application, a worker
  thread and clean shutdown, cancelling pending requests when a game ends
  or a search resets. Moderate complexity.
- The current single model (`g_model` in controller.cpp) is not safe to
  reload while another thread is mid-step; any async version should hold the
  model through a `shared_ptr` per search.

### Batching across trees: not pursuing
Running many trees' check-ins as one batched ORT call would cut ORT's
per-call overhead, but it isn't worth it unless absolutely needed:
- CPU batching gains far less than GPU batching.
- Two check-ins from the same trajectory can never share a batch (causal
  order), so it needs many different trajectories waiting at a check-in at
  the same moment.
- It needs a batched step graph plus gather/scatter of each tree's conv
  caches: complicated, for small gains.

Suggested order if throughput suffers: (1) timing counter in val_test,
(2) trim the step graph, (3) oracle + background thread with the blocking
rule above, without batching.
