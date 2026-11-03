# Xerces Chess Engine - Claude Instructions

## Two-Repo Structure
This project spans two repositories:
- `chessbot` (this repo) — Python package: selfplay loop, MCTS wrapper, model, training, review
- `pyfastchess` (C++ repo) — Python extension providing `MCTSTree` base class, board, caches

The C++ repo builds a `.pyd` extension installed into the Python environment.
Start Claude Code from the parent directory (`C:\Users\Bryan\repos\`) to access both repos.

## Project Identity
- Engine name: **Xerces**
- Python package: `chessbot` (src layout: `src/chessbot/`)
- Entry point: `scripts/run_selfplay.py <run_tag>`
- Run configs live at: `D:\Data\chessbot_data\selfplay_runs\<run_tag>\config.yaml`
- Active run tag: check the current run dir under `selfplay_dir` — don't assume; this drifts

## Key Files
- `src/chessbot/mcts_utils.py` — `MCTSTree` (wraps C++) and `ChessGame`
- `src/chessbot/looper.py` — `GameLooper`, batched inference, `StockfishThread`
- `src/chessbot/rescore.py` — post-hoc SF analysis, training data prep, rescoring
- `src/chessbot/review.py` — `GameViewer`, `RecordKeeper`, lc0 review tools
- `src/chessbot/config.py` — `Config` class (yaml/json/dict loadable)
- `src/chessbot/model.py` — all model architectures
- `src/chessbot/utils.py` — `GameGenerator`, scoring utils, SF helpers
- `scripts/run_selfplay.py` — main orchestrator (multi-process workers, retrain loop)
- `scripts/retrain_worker.py` — subprocess target for model retraining

## Board Encoding (the sauce)
64 integer tokens, one per square, always in STM-as-white orientation (flip via `sq ^ 56`).

| Token | Meaning |
|---|---|
| 0 | Empty |
| 1-5 | STM pawn, knight, bishop, rook, queen |
| 6 | STM king, no castling rights |
| 7 | STM king, kingside only |
| 8 | STM king, queenside only |
| 9 | STM king, both sides (BASE=9) |
| 10-18 | OPP pieces (same structure, +9 offset) |
| 19 | En passant square |

- Castling rights encoded into the king token — no separate planes needed
- EP encoded as a token ON the ep square — model gets spatial/file context for free
- History planes ARE used: `encoding_type=xc0h` with `history_K` past-ply boards stacked
  into the position embedding. The "AZ history planes are cargo cult" hypothesis was
  tested head-to-head against a no-history encoding and rejected decisively — history
  helps. Do not resurrect the no-history variant as a default.
- vocab_size=21 in all models

## Active Model Architecture
m137_8c8t: `build_pt_expandy_swiglu` (registry key `m13-7-8c8t-d256`):
- xc0h stem -> plain + gated expandy convs -> add pos -> FF + SwiGLU tx blocks,
  LayerNorm throughout
- Gated expandy: grouped 3x3 + 1x1 pointwise, sigmoid gate per square/channel
- Per-channel PReLU; tanh-gated attention on SwiGLU blocks
- Global accumulator tokens for W/D/L, SmartGate and from/to policy
- SmartGate: policy - scale * sigmoid(gate_raw), suppress-only, legality BCE gate loss
- PyTorch fp16 autocast
- Policy: 1858 legal logits scattered to 4288 flat
- Value: WDL 3-class softmax, STM-POV
- Inference: ORT+TRT fp16 via `infer_ort_trt.py` / `prepare_trt`
- Dims, block counts and init values live in the builder in `model.py` (a path
  for when a task needs them, not an instruction to go look)

## MCTS
- `MCTSTree` extends C++ `fasttree`, adds Python-side early stopping, noise, sampling
- `best()` returns 3-tuple: `(move_uci, method, xerces_top_if_sampled_diff)`
- Early stopping: floor→ceiling sims budget, `robust_selection_criteria`, extensions
- Sims schedule: dict keyed by ply (e.g. ramps up through middlegame, backs off endgame)
- c_puct sampled from list per game; dirichlet noise per move

## Training Loop
- Workers play N games concurrently, batch NN inference centrally
- Finished games → `Rescorer.analyze_and_rescore()` (post-hoc SF analysis per ply)
- Blunder detection → visit redistribution → flat 4288-dim policy targets
- `training_queue_buffer` games accumulated → pause workers → retrain subprocess → unpause
- VRAM reclaimed via spawn process before retrain
- Each retrain draws primary + replay + historic shards; the counts are the
  `RETRAIN_*_SHARDS` constants in `replay_buffer.py`. Historic can be set to 0
  (draw, validation slice and its csv row all switch off). After a successful
  retrain: primary → replay buffer, the drawn replay shards are discarded, and
  historic shards stay in place (historic is read-only and can be redrawn; a
  full copy lives on E:).
- Historic corpus lives at `BOOTSTRAP_PKL_GZ` (pkl.gz shards, not tfrecords —
  the `BOOTSTRAP_TFREC_DIR` name survives only in legacy tfrec-era scripts).
  Mostly older xc0 selfplay plus some lc0, mixed within shards.

## Game Curriculum
- Scenarios: startpos, UHO, pre_opened, pre_opened_mini, vs_stockfish,
  random_init, piece_odds, piece_training
- UHO = Unbalanced Human Openings (UHO_XXL_2022_+100_+129.pgn)
- The mix is `game_probs` in the run's `config.yaml` (a path for when a task
  needs it, not an instruction to go look)
- Bootstrapped by pretraining on the historic corpus (older xc0 + lc0)

## Validation Ladder
- Paired games from UHO positions (each played as both colors)
- Score > 0.5 vs Stockfish fixed depth → increment depth
## Coding Conventions
- All Q values are white-POV in C++; Python flips with `sign = 1 if white_to_move else -1`
- STM-POV value = `Q_white * sign`
- Config loaded from YAML via `Config.from_yaml()`; unknown keys raise `AttributeError`
- Games saved as pkl to `<run_dir>/game_logs/<game_id>_log.pkl`
- Analysis chunks staged to `<run_dir>/analysis_staging/`, merged to `analyze_results_combined.pkl`

## Style Preferences
- Lines under 89 chars where possible (not strict)
- No `_leading_underscores` in names
- No comment dividers with dashes or box-drawing characters
- Minimal comments and code — avoid verbosity, prefer clean readable code over explanation
- Comments/docstrings describe the current state of the code only — never reference what it used to do, what changed, or "now"/"anymore" framing. A reader with zero history should never hit a comment that only makes sense if they know the prior version. Changelog-type info goes in the commit message or chat, not inline.
- No Unicode characters outside ASCII in print statements or log output — Windows subprocesses use cp1252 and will crash on characters like arrows or other non-ASCII symbols. Use ASCII equivalents (e.g. `->` not `->`).

## Working Rules
- Manual mode: no autonomy. Every action (reads, searches, tool calls, edits) goes
  through the user's approval. Never switch modes (plan mode, worktrees, anything)
  without explicit permission. A question like "do you want plan mode?" is not
  permission.
- Wait for orders: do exactly what is asked. For any concern or scope change, state it
  with a proposed fix and ask first. Never scout or prep unasked.

## Chat Response Format
User is dyslexic. During iterative status-update style responses (reporting what changed, asking something, flagging a concern, saying what's next), structure with visual headers, each with terse bullet points:
- `=== DONE ===`: summary of what was done.
- `=== FLAGGED ===`: use sparingly, only for real problems: something is wrong or an issue is expected.
- `=== OPEN ===`: strictly open questions or decisions about items in DONE or FLAGGED. Don't stretch to fill it.
- `=== NEXT ===`: next steps. Leave it out unless the user asks; they will ask in a follow-up when they want next steps.

Omit a section entirely if empty rather than writing "none". Don't force this onto short direct answers or already-tabular output.
