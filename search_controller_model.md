# Search Controller Model

A small causal temporal model that watches an MCTS search at fixed check-ins
and decides when to stop. It replaces the hand-written early-stop rules
(tier1/tier2, RSC, qmm gap block) and the floor/ceiling sims gates.

Each search gets:
- one **root vector** (48 features), read once at the first check-in and fed
  through an init graph that conditions the whole network (FiLM);
- one **check-in vector** (64 features) per check-in (every 100 root visits),
  fed through a cached step graph that costs the same at every check-in.

## Architecture: K5L6

Shared by both sizes:

| part | spec |
|---|---|
| width | 128 throughout |
| temporal trunk | 6 causal dilated Conv1D blocks, kernel 5, dilations `[1, 1, 2, 2, 4, 4]` |
| receptive field | 57 check-ins (`1 + sum((k - 1) * d)`) |
| conv block | `x + silu(LayerNorm(conv(x)) * (1 + gamma) + beta)` |
| step input | `Linear(64 -> 128)` on the check-in vector |
| context encoder | `Linear(48 -> 128)` -> SiLU -> residual SwiGLU block (h256) -> `p` |
| FiLM | `Linear(128 -> 6 * 2 * 128)` from `p`, gamma and beta per conv block, zero-init (starts as identity) |
| pooling | latest token, running mean of all tokens, and `p` |
| head | `Linear(3 * 128 -> 128)` -> SiLU -> two sigmoid heads: stability, quality |

SwiGLU block (used by the mixers and the context encoder):
`x + down(silu(up(LayerNorm(x))) * gate(LayerNorm(x)))`

### M and L

The sizes differ only in the per-check-in SwiGLU mixers. A mixer runs right
after its conv block's residual output and before the next conv block reads
the activation. Mixers act on one check-in's 128 features (no time mixing),
so they need no cache. The next layer's cache stores the post-mixer
activation.

| | mixers | params |
|---|---|---|
| **M** | h128 after conv blocks 2 and 4 | 954,882 |
| **L** | h256 after conv blocks 2, 4 and 6 (the last one runs before pooling) | 1,152,898 |

**M is the starting model.**

### Input scaling

- Both input vectors use the fixed empirical scales listed below, which put
  typical values in about 0..1.
- Board features are also clipped to `[0, 2]`, so no rare position can blow
  up the input.
- On top of that, each vector gets a learnable per-feature scale (init 1.0)
  and bias (init 0), with no weight decay. They fold into the first `Linear`
  at export, so inference cost is zero.
- No pre-norm on the inputs: they are already scaled and bounded.

## Inference: cached causal step graph

The model only ever runs cached. The full causal graph is used for
verification only.

- **Init graph** (once per search): root vector -> `p`, plus each conv
  block's FiLM `scale = 1 + gamma` and `beta`.
- **Step graph** (once per check-in), per conv block:
  - `window = cat(cache, h)`
  - `y = window[::d].reshape(1, -1) @ W`, where
    `W = conv.weight.permute(2, 1, 0).reshape(-1, 128)`
  - the new cache is `window[1:]` (the `(k - 1) * d` prior inputs)
  - `h = h + silu(LayerNorm(y + bias) * scale + beta)`, then the mixer if present
- **Running state:** the sum and count for the running mean, plus per-layer
  caches. That's 28.5 KB of conv cache for K5L6.
- **Runtime:** ORT CPUExecutionProvider, FP32, intra/inter-op 1 thread,
  `ORT_ENABLE_ALL`, sequential, persistent sessions.
- **I/O binding** with ping-pong state buffers: numpy arrays wrapped as
  OrtValues, zero-copy. This is the production-realistic method.
- The cached step matches the full causal model to within 1.2e-07. I/O
  binding is bit-identical to `session.run`.

### What made the step fast

- A 1-D `W @ v` matvec is slow in ORT (about 95 us per layer). Use `v @ Wt`.
- Keep everything 2-D `(1, 128)` so MatMul + Add fuse into Gemm. That took
  K3 4L from 62 to 48 nodes.
- Precompute FiLM in the init graph rather than the step.
- ORT costs about 1.7 us per node. An empty graph costs 9.8 us with I/O
  binding and 21 us with `session.run`.

### Speed

CPU only (GPU hidden with `CUDA_VISIBLE_DEVICES=-1`), cached step with I/O
binding, 400 sequences x 128 steps. Step cost is flat across check-ins
1..128.

| model | init us | step median us | step p95 us | us per search (init + 40 check-ins) |
|---|---|---|---|---|
| K5L6 base (no mixers) | 106 | 153.5 | 158.5 | 6,246 |
| K5L6-S (one h64 mixer) | 105 | 164.3 | 169.0 | 6,677 |
| K5L6-M, old 32-wide context | 104 | 177.4 | 183.2 | 7,200 |
| K5L6-L, old 32-wide context | 108 | 205.1 | 211.6 | 8,312 |
| **K5L6-M, 48 + SwiGLU context** | **133** | **179.7** | 194.6 | **7,321** |
| **K5L6-L, 48 + SwiGLU context** | **134** | **207.4** | 233.4 | **8,430** |

- The SwiGLU context block adds about 30 us to init (once per search) and
  nothing measurable to the step (+2 us, inside the ~5 us run-to-run noise).
- M is under the 200 us step target. L is just over it.
- At 4000 sims per search, M costs about 1.8 us per sim.

Earlier bake-off (cached, I/O binding, no mixers): K5 4L `[1,2,4,8]` 112 us
with RF 61; K3 6L `[1,1,2,2,4,4]` 130 us with RF 29; K5 6L
`[1,1,2,2,4,4]` 149 us with RF 57.

## Root vector (48)

Built once per search, at the first check-in, so the root is always expanded
(priors and sharp tags exist).
- WDL is the 0-visit NN eval (`node.value`), never refreshed.
- The init graph runs once, so FiLM and the conv caches never go out of sync.
- Live root WDL updates go in through the check-in vector.

All values are STM-POV. The "scale" column is the divisor applied before the
clip.

**Value (1-4)**

| # | feature | scale |
|---|---|---|
| 1-3 | W, D, L (NN eval) | 1 |
| 4 | settled: 1 - (WDL entropy / ln 3) | 1 |

**Policy (5-12)**

| # | feature | scale |
|---|---|---|
| 5 | effective move count, 1 / sum(p^2) | 64, clipped at 2 |
| 6 | policy entropy / ln(n legal) | 1 |
| 7-11 | top-5 raw priors, sorted, 0 if fewer than 5 moves | 1 |
| 12 | tactical prior mass: raw prior on captures + checks | 1 |

**Game state (13-16)**

| # | feature | scale |
|---|---|---|
| 13 | half-move clock | 50 |
| 14 | repetition count | 1 |
| 15 | full move number | 100 |
| 16 | last move was a capture | 1 |

**Mobility (17-22)**

| # | feature | scale |
|---|---|---|
| 17 | legal moves | 64 |
| 18 | captures | 8 |
| 19 | checks | 8 |
| 20 | pawn pushes, including promotions | 16 |
| 21 | king moves, including castling | 8 |
| 22 | in check | 1 |

**Material (23-32):** our P, N, B, R, Q, then their P, N, B, R, Q. Scales:
P 8; N, B, R 2; Q 1.

**Control (33-41)**

| # | feature | scale |
|---|---|---|
| 33-34 | squares attacked by us, by them | 48 |
| 35 | squares attacked by both | 32 |
| 36-37 | hanging pieces (attacked, undefended, not king), ours and theirs | 4 |
| 38-39 | attacked material value (P1 N3 B3 R5 Q9), ours and theirs | 20 |
| 40-41 | defended material value, ours and theirs | 39 |

**King safety (42-43):** share of each king's ring (adjacent squares) attacked
by the enemy, ours and theirs. Scale 1.

**Pawns (44-48)**

| # | feature | scale |
|---|---|---|
| 44-45 | average pawn depth (ranks advanced from the start rank), ours and theirs | 4 |
| 46-47 | passed pawns, ours and theirs | 4 |
| 48 | pawn tension: our-pawn-attacks-their-pawn pairs (symmetric, one count), plus legal en passant captures counted from the move list | 4 |

### Feature definitions and edge cases

- **Last capture:**
  - HMC > 0 means no capture.
  - If HMC == 0, read the board's last move:
    - a non-pawn move is a capture;
    - a pawn move is a capture only if it changed file (covers en passant and
      capture-promotions);
    - castling never resets the clock.
- **Repetitions:** chess.hpp's repetition scan over stored past positions.
- **FEN-only starts (no history):**
  - last capture and repetitions are 0;
  - HMC and full move come from the FEN counters;
  - missing or junk counters (`""`, `-`, `- -`, `x y`) parse to 0 and 1
    without error.
- **Pre-opened games and reused-tree roots:** carry full history (at least 9
  plies), so every feature is exact.
- **Checks:** `sharp - capture`, reusing the root's sharp tags (capture +
  check, 0..2) instead of a second `givesCheck` pass.

### Scale tuning (board features)

- Sampled 100k positions from sp7-sp10 (25k per run, 2,500 games per run, at
  most 10 positions per game).
- Stratified by ply bucket (edges 20, 40, 60, 90, 130) and root WDL bucket
  (drawish when D >= 0.6, otherwise losing, balanced or winning by W - L).
- Each game was replayed from `init_fen` through `history_uci`. The replay was
  checked against the logged `xc0h` tokens first (105/105 plies matched).
- **First pass:** nothing blew up (worst scaled value 3.5, from 7 rooks), but
  about 10 features were squashed near 0. Examples of scaled p99: HMC 0.27,
  tension 0.25, hanging 0.25.
- **Retune:** scales chosen so each feature's scaled p99 is about 0.5..1,
  plus the `[0, 2]` clip.
- **Resample** (100k new positions with a fixed seed; game IDs recorded):
  - the C++ scaled output matches Python's `min(raw / scale, 2)` exactly;
  - every board feature has scaled p99 between 0.5 and 1 and std between 0.11
    and 0.48;
  - no feature exceeds 1 in more than 0.45% of positions;
  - the clip fires only on rooks (0.10%), knights (0.05-0.06%), checks
    (0.007%) and queens (0.003%).
- HMC, reps and tension stay narrow (std 0.11-0.17) because they are mostly 0
  by nature.

## Check-in vector (64)

Built at every check-in, every 100 root visits. All values are STM-POV. Slots
are the top 4 root children by visits.

**Progress (1)**
1. Check-in number (scale about 20, to be tuned)

**Root value (3)**
2-4. Root W, D, L, backed up (`p_win / N`, `p_draw / N`, `p_loss / N`)

**Top-4 bundle (46)**
- Slots 1-2 have 11 features each. A root with 0 or 1 legal moves is never
  searched, so these always exist.
- Slots 3-4 have 12 each: an `exists` flag plus the same 11.
- A missing move's slot is all 0.

Per slot:
- [slots 3-4 only] exists
- visit share, N / root N
- recency visit share (`visit_share`, the EMA, decayed to the current tick as
  `root_child_details()` does)
- visits since the last check-in, divided by the root's visit gain since the
  last check-in, from a snapshot of root child visits taken at each check-in
  (equal to / 100 at exact check-ins; the top 4 sum to at most 1)
- Q = W / n_done, the mean of completed backups (the tree's own Q divides by
  N, in-flight included, and is only refreshed at that child's backups)
- Qema
- Qmm, with `qmm_cand` semantics: the child's Qmm once set, else the plain
  mean, same as `root_child_details()`
- Qdelta_sign (already stored in the root's STM POV by backup)

Visit counts (shares, visits since the last check-in) use N, in-flight
included: a descent PUCT committed to is a visit. Only the value uses
completed backups.
- raw prior
- prior rank (rank in the raw priors)
- is check
- is capture

**Leader stats (9).** The leader is slot 1, the most-visited child.
- consecutive check-ins as leader
- leader Q - slot 2 Q
- leader Q - best Q among the top 4 (0 if the leader has the best Q)
- leader Q - leader Qema
- leader Q - leader Qmm
- leader Qema - best Qema among the top 4 (0 if the leader has the best Qema)
- visit gap: slot 1 share - slot 2 share
- leader changes, counted at check-ins, this search
- unique leaders seen at check-ins this search

**Visit distribution (5)**
- visit entropy, normalized
- effective visited moves, 1 / sum(share^2)
- visited children / legal moves
- top-3 share
- KL(visits || raw priors)

### Design notes

- **No budget features.** Sims floor and ceiling are left out so the
  controller generalizes beyond the current schedule.
- **Linear combinations are kept on purpose.** Gaps such as `Q1 - Q2`, the
  visit gap and `Q - Qema` hand the model likely-important terms one layer
  sooner, so products like `(Q1 - Q2) * (vs1 - vs2)` are reachable earlier in
  a 6-layer net.
- **No squaring.** It loses the sign and crushes small gaps. Scale is handled
  by empirical scales plus the learned per-feature scale.
- **Leader changes are counted at check-ins, not per sim.** That's cheap and
  matches what the controller sees. A per-sim flicker count can be added later
  if needed.
- **Tree reuse.** The per-search sequence starts fresh: the check-in number,
  leader tracking and visit snapshot all reset in `reset_early_stop()`, which
  Python calls right after `advance_root`. The first check-in's "visits since
  last check-in" counts only this search's visits. Tree-state features (visit
  share, Q, Qema, root W/D/L) include everything inherited.
- **Check-ins must fire at exact multiples.** The early-stop hook in
  `collect_many_leaves` runs after every descent, including cache hits and
  mate-lock terminal runs, so it lands on exact sim counts. The controller
  hooks into the same spot, from the first 100 visits.

### Check-in scales

Values not listed are used as-is: shares, recency share, visits since last
check-in, root W/D/L, Q, Qema, Qmm, Qdelta_sign, raw prior, flags, visit gap,
entropy, visited / legal and top-3 share.

| feature | transform |
|---|---|
| check-in number | / 40 |
| prior rank | / 10 |
| leader run (consecutive check-ins the current leader has led, counting this one) | / 25 |
| leader changes | / 5 |
| unique leaders | / 4 |
| effective visited moves | / 8 |
| KL(visits \|\| raw priors) | / 3 |
| leader Q - best Q, leader Q - Qema, leader Q - Qmm, leader Qema - best Qema | x 4 |
| leader Q - slot 2 Q | x 2 |

- Every check-in feature is then clipped to `[-2, 2]` (`FEATURE_CLIP`),
  except check-in number and leader run, which clip at 8 (`CK_COUNT_CLIP`)
  so both keep a full range out to 20k sims.
- Leader run uses / 25 rather than the data-fit / 40: in selfplay the leader
  is expected to flip more than in these searches.
- The learnable per-feature input scale starts at 1.0 on top of these.

**Tuning data.** The rpo_c26k base arm (`results.pkl`): 25,895 real-NN
searches (floor 800, ceiling 4000), 475,885 check-ins on the 100-sim grid,
rebuilt in Python.
- Exact from the pickle: everything except the items below.
- Approximate:
  - slot Q (rebuilt as `Q * N / n_done`, which overshoots when N far exceeds
    n_done; C++ uses the exact `W / n_done`);
  - prior rank, entropy, effective visited moves and KL (only the top-8 kept
    children are in the pickle).
- Is check and is capture are binary and were not rebuilt.

Findings with the first guessed scales:
- check-in number and leader run overflowed (p99 2.0 and 3.9);
- unique leaders, effective visited moves and KL ran a little high;
- the leader Q gaps were squashed (std 0.02-0.13, p99 at most 0.63).
Everything else already had p99 of about 1 or less.

## C++ (pyfastchess, branch `bitboard_featurtes`)

### Backend (`backend.hpp` / `backend.cpp`)

- `enum BoardFeature`: the 36 board-feature slots, in the order above.
- `struct MoveCounts`: legal moves, captures, checks, pawn pushes, king moves.
- `board_features(const Board&, const MoveCounts&)`: raw features.
- `board_features(const Board&)`: generates moves and counts them itself.
- `scale_board_features(BoardFeatures&)`: divides by `BOARD_SCALE` and clips
  at `FEATURE_CLIP` (shared with the context vector's effective move count).
- `Board::last_move()`: last pushed move, empty if there's no history.
- Attack maps are built per color from chess.hpp's attack tables:
  - pawn shifts, plus knight/bishop/rook/king attacks;
  - queens counted as both bishop and rook.
- Passed-pawn masks are precomputed per color and square.

### MCTS (`mcts.hpp` / `mcts.cpp`)

- `MCTSNode::search_context_features()`:
  - builds the 48-vector from the node's `value` and `ordered_children`
    (packed move, raw prior and sharp flag together, so there's no
    ordering to keep in sync);
  - caches it in `search_context_cache`;
  - `advance_root` clears that cache when a reused child becomes the root.
- `make_expanded_node(...)` and `bench_search_context_features(...)`: test and benchmark
  helpers.
- `MCTSTree::search_check_in_features()`:
  - builds the 64-vector in one pass over `root->ordered_children` (top 4 by
    visits, totals), plus O(n) scans for prior rank and the snapshot lookup;
  - then advances `ctrl_` (`ControllerTrack`: check-in count, last root
    visits, leader, run, changes, unique leaders, child visit snapshot).
- `build_check_in_features(LeaderStep&) const` does the read-only build, so
  `bench_search_check_in_features(reps)` can time it without changing the
  tracking.
- `reset_early_stop()` resets `ctrl_` and snapshots the root children.
- Guessed scales for the non-0..1 check-in features are named constants
  (`CK_*`) in `mcts.cpp`.

### Python

- `Board.board_features(scaled=False)`
- `MCTSNode.search_context_features()`
- `pyfastchess.search_context_features_for(board, wdl_white_pov, {uci: raw_prior})`
- `pyfastchess.bench_search_context_features(boards, reps)`
- `MCTSTree.search_check_in_features()`
- `MCTSTree.bench_search_check_in_features(reps)`
- `BOARD_FEATURE_NAMES`, `NN_FEATURE_NAMES`, `CHECK_IN_FEATURE_NAMES`

`Board.board_features()` is the raw 36 board-side features (game state,
mobility, material, control, king safety, pawns). It generates its own legal
moves; `scaled=True` returns the divided and clipped form. It's used for
sampling and tests; the search uses `search_context_features()`, which reads
the move counts off the root's children instead.

### Verification

- **All 48 against an independent Python reference:** 300 random roots with
  random WDL and priors; max error 2.4e-07.
- **Pawn tension:** checked against a Python brute force on the same boards.
- **Hand cases:** start position, captures and hanging pieces after
  1.e4 d5 2.exd5 Qxd5 3.Nc3, passed pawns, in check, promotions, pawn tension
  with en passant.

### Speed

Median ns per root, over 2,000 random-playout positions.

| stage | ns |
|---|---|
| movegen alone | 253 |
| all board features with their own movegen | 930 |
| `search_context_features()`, sharp tags already set | 870 |
| `search_context_features()`, sharp tags from scratch | 1,380 |
| cached read | 7 |

- The bitboard features cost about 100 ns.
- The main cost is `givesCheck` per move. That's why checks reuse the sharp
  tags.
- The whole root vector is under 1% of the init graph, once per search.

Check-in vector: 30 searches x 2000 sims through the production collect,
raw-cache and resolve path, with a fake NN (random logits and WDL), 582
check-ins. Build plus snapshot: median 700 ns, p95 900, max 1000. That's
about 0.4% of the model's step.

## Files

### Model

The controller model exists only in the benchmark script so far:
`C:\Users\bryan\AppData\Local\Temp\claude\C--Users-bryan-repos-chessbot\fdeffdd9-1b46-479b-8af9-d94f9e248922\scratchpad\ctrl_bench.py`.
- `TemporalBlock`, `SwiGLUMixer` and `Controller` define the model; `SPECS`
  selects the M and L variants.
- `Init` and `Step` are the cached init and step graphs.
- The steppers run the step graph with I/O binding.
- The exported ONNX graphs go to `ctrl_bench\` next to it.

### Scripts

All in `C:\Users\bryan\AppData\Local\Temp\claude\C--Users-bryan-repos-chessbot\fdeffdd9-1b46-479b-8af9-d94f9e248922\scratchpad`. That's a session scratchpad, so
copy anything worth keeping into the repo.

| script | what it does |
|---|---|
| `ctrl_bench.py` | model definitions; exports full, init and step ONNX graphs; verifies the cached step against the full causal model (ORT and PyTorch); times the step with I/O binding and `session.run`. Logs: `ctrl_bench.log` (base/S/M/L), `ctrl_bench_ctx.log` (M/L with the 48 + SwiGLU context) |
| `profile_step.py` | ORT per-op and per-node profile of the step graph (found the 1-D matvec cost) |
| `ort_floor.py` | ORT empty-graph floor (I/O binding vs run) and optimized-graph node counts |
| `cached_quick.py` | quick cached-vs-full check and timing for K3/K5 4L |
| `controller_speed.py` | first CPU speed test (eager fp32/fp16, ORT, frozen torch); superseded by `ctrl_bench.py` |
| `wdl_onesided.py` | \|W - L\| vs 1 - normed WDL entropy over WDL sweeps |
| `board_feats_check.py` | first board-feature hand checks; uses APIs that have since been removed, so it's stale |
| `root_feats_check.py` | validates all 48 context features against an independent Python reference (300 random roots), pawn tension hand cases, and times `bench_search_context_features` |
| `feat_sample.py` | 100k-position board-feature sampler over sp7-sp10 (4 workers, ply and WDL stratified, replay verified against `xc0h`). Data: `feat_sample\` (first pass, raw) and `feat_sample2\` (resample: raw, scaled, game IDs) |
| `checkin_scale_rpo.py` | rebuilds the 64 check-in features from the rpo_c26k base arm's `results.pkl` (streamed) and reports their ranges; output `checkin_scale_rpo.npz` |
| `checkin_bench.py` | runs fake-NN searches through the production collect/resolve path, times `bench_search_check_in_features`, prints check-in feature ranges, and checks the tree-reuse snapshot |

## Open items

1. Hook the check-in build into `collect_many_leaves` at exact multiples, and
   log the root vector and check-in sequence per search.
2. Define the labels for the stability and quality heads.
3. Add the model to `model.py`, then the training script and the ORT export
   (init and step graphs, folding the learned input scales).
