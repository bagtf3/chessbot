# Search Controller

A small causal temporal model that watches an MCTS search at fixed check-ins
and decides when to stop or extend. It runs inside pyfastchess's C++ search
(ONNX Runtime, CPU), so self-play never pauses for Python mid-search.
Python only reads `es_tripped` / `es_stop_reason`.

The rule params are documented at the `ctrl_*` params in
`pyfastchess/src/mcts.hpp`; config defaults live in `config.py`.

## Stop rules

Check-ins run every 100 root visits. No stop below `ctrl_min_stop`;
`sims_max` is rounded up to a check-in and held at `ctrl_min_stop` or more.

- **`ctrl`**: stability >= `ctrl_s_thresh` and quality >= `ctrl_q_thresh` on
  `ctrl_agree_consec` straight check-ins with the same leader.
- **`lock`**: exact visit dominance, leader N - runner-up N > sims_max -
  root N (at sims_max every non-tied lead counts; a tie goes to the
  tie-break).
- **Extension**: with an extension left (`ctrl_ext_max` per move), a lock is
  rejected when D < `ctrl_ext_d_max` and quality < `ctrl_ext_q0` -
  max(0, |Q| - `ctrl_ext_absq0`) * `ctrl_ext_slope` (|Q|, D from the root's
  raw WDL). Then sims_max rises by `ctrl_ext_sims`, the `ctrl` streak
  restarts, and the root's `pruning_factor` drops to `ctrl_ext_pruning` until
  the next move. With no extension left, the lock plays.
- Leader everywhere (`best()`, details, PV, check-in slots, lock): visits,
  then STM Q, then the visit-share EMA, then prior, then list order.
- Off on Stockfish's turns in vs-SF games. No tree reuse while it is on.
- `ctrl_enable` needs `es_check_every: 0` (one stop system at a time) and a
  loaded model. RSC selection is off; every stop plays the leader.
- `stop_rules` (`ctrl` / `old`, `game_utils.STOP_RULE_ARMS`) switches every
  stop setting together, so the old tiered early stop can be sampled per
  game.

## Telemetry

- `tree.get_ctrl_summary()` per search: check-ins, ms inside check-ins,
  sims_max before extensions, `ext` (extensions used), first rejected
  leader, latest stab/qual, tied check-ins, and per rejected lock (leader,
  stab, qual, root STM Q, D). Rescore prints its controller table from this.
- Lean game logs keep `ctrl_stab`, `ctrl_qual`, `ctrl_ext`, `ctrl_rej`.
- `ctrl_record` (off by default) keeps every check-in (features, heads,
  root N, leader, root Q, rule code) as `ply["ctrl"]`, for retraining data.

## Model: K5L6

- Width 128. Six causal dilated Conv1D blocks, kernel 5, dilations
  `[1, 1, 2, 2, 4, 4]`, receptive field 57 check-ins.
- Conv block: `x + silu(LayerNorm(conv(x)) * (1 + gamma) + beta)`, with
  FiLM gamma/beta from the root context (zero-init, starts as identity).
- Context encoder: `Linear(48 -> 128)` -> SiLU -> residual SwiGLU (h256).
- Step input: `Linear(64 -> 128)`.
- Pooling: latest token, running mean of all tokens, and the context.
- Head: `Linear(384 -> 128)` -> SiLU -> two sigmoid heads, stability and
  quality.
- Sizes differ only in per-check-in SwiGLU mixers: **M** (production) has
  h128 after blocks 2 and 4 (955k params); L has h256 after 2, 4 and 6.
- Inputs use fixed empirical scales (typical values about 0..1), then a
  learnable per-feature scale and bias folded into the first Linear at
  export. Board features clip at `[0, 2]`; check-in features at `[-2, 2]`,
  except check-in number and leader run, which clip at 8.

Production model: `controller_M_two_head_1024`, trained by
`experimentation/train_controller.py`, exported by
`experimentation/export_controller.py` to a `.ctrl` dir (`init.onnx`,
`step.onnx`, `meta.json`, `parity.npz`).

## Inference

- **Init graph**, once per search at the first check-in: context -> p and
  each block's FiLM scale and beta.
- **Step graph**, once per check-in, per block: `window = cat(cache, h)`,
  `y = window[::d].reshape(1, -1) @ W`, new cache `window[1:]`, then the
  residual, then the mixer if present. About 28.5 KB of state per search.
- Keep tensors 2-D `(1, 128)` and use `v @ Wt` (a 1-D matvec is ~95 us per
  layer in ORT). ORT costs about 1.7 us per node.
- C++ runtime (`pyfastchess/src/controller.*`): one process-wide model,
  loaded by `pyfastchess.load_search_controller(dir)`, which validates
  `meta.json` (feature version, interval, feature names, heads). Per-tree
  `ControllerState` double-buffers the step state as zero-copy OrtValues.
- Cost: ~180 us per check-in on the worker thread, about 3-7 ms per move.

## Root context vector (48)

Built once per search from the expanded root. STM-POV.

| # | group | features |
|---|---|---|
| 1-4 | value | NN W, D, L; settled = 1 - WDL entropy / ln 3 |
| 5-12 | policy | effective move count, normalized entropy, top-5 raw priors, tactical prior mass (captures + checks) |
| 13-16 | game state | half-move clock, repetitions, full move number, last move a capture |
| 17-22 | mobility | legal moves, captures, checks, pawn pushes, king moves, in check |
| 23-32 | material | our and their P, N, B, R, Q |
| 33-41 | control | squares attacked by us / them / both, hanging pieces, attacked and defended material |
| 42-43 | king safety | share of each king ring attacked |
| 44-48 | pawns | average pawn depth, passed pawns, pawn tension |

Board features come from `Board.board_features()` (raw) and
`MCTSNode.search_context_features()` (scaled, reusing the root's move
counts and sharp tags). FEN-only starts give last capture and repetitions
0; HMC and full move come from the FEN counters.

## Check-in vector (64)

Built every check-in. STM-POV. Slots are the top 4 root children by visits.

- Progress: check-in number.
- Root value: backed-up W, D, L.
- Top-4 slots (slots 3-4 add an `exists` flag): visit share, recency share,
  visits since the last check-in, Q (W / n_done), Qema, Qmm, Qdelta_sign,
  raw prior, prior rank, is check, is capture.
- Leader stats: leader run, Q gap to slot 2, Q gap to best Q, Q - Qema,
  Q - Qmm, Qema gap to best Qema, visit gap, leader changes, unique leaders.
- Visit distribution: entropy, effective visited moves, visited / legal,
  top-3 share, KL(visits || raw priors).

No budget features, so the controller generalizes beyond the sims schedule.
Linear gaps (Q1 - Q2, visit gap, Q - Qema) are kept on purpose.

## Selection-only vscale

Every stored and exported statistic is raw. PUCT reads `select_q(vscale)`,
which scales only the network part of a node's value before contempt and
FPU. `vscale_gate_q` turns vscale on only while the root's raw |Q| is above
the gate.

## Open questions

- **Synchronous check-ins.** Average cost is ~1% of the loop, but cache-hit
  bursts can push it toward 20% while they last. First step if it matters:
  a per-worker timer. Cheapest fix: trim the step graph (~200 nodes, about
  half of them constants).
- **Batching check-ins across trees** is not worth it: CPU batching gains
  little, and check-ins from one trajectory can never share a batch.
