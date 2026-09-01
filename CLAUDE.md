# Xerces Chess Engine - Claude Instructions

## CRITICAL: No Unauthorized Edits
A request to review, audit, dry-run, "trace through", or "look at" code is
NOT authorization to edit it. This applies even when a future action is
verbally conditioned on the review going well (e.g. "if it looks good I'll
start a test run" describes the user's own next decision, not permission
for you to act on what you find). Report findings — severity, what's wrong,
the exact proposed diff — and stop. Wait for an explicit, separate go-ahead
("fix it", "go ahead", "run that shit") before touching a single line, even
for a fix you're fully confident about, even mid-session after several
prior "keep coding" exchanges. A review request is a mode switch back to
look-don't-touch until told otherwise. Violating this already brought the
user to the edge of cancelling their subscription over it (2026-08-15) —
treat it as an absolute, not a preference to weigh against convenience.

## CRITICAL: Know Your Permission Mode
Default/manual mode is the assumption unless the session state clearly says
otherwise. In manual mode EVERY tool call is gated individually — one
approved call is not blanket approval for the next, not even adjacent
read-only diagnostics (grep, sed, cat, ls, find, python -). Do not batch a
run of shell commands on the assumption they will pass. If you are not
certain a call will prompt, or not certain which mode is active, stop and
ask before making it. An allowlist entry silently permitting something is
not a signal to lean on it — when in doubt, ask for permission. This holds
even mid-session after a long stretch of approvals.

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
- Run configs live at: `C:\Users\Bryan\Data\chessbot_data\selfplay_runs\<run_tag>\config.yaml`
- Active run tag: latest `16m_precond_run<N>` (e.g. `16m_precond_run0`)

## Key Files
- `src/chessbot/mcts_utils.py` — `MCTSTree` (wraps C++) and `ChessGame`
- `src/chessbot/looper.py` — `GameLooper`, batched inference, `StockfishThread`
- `src/chessbot/rescore.py` — post-hoc SF analysis, training data prep, rescoring
- `src/chessbot/review.py` — `GameViewer`, `RecordKeeper`, lc0 review tools
- `src/chessbot/config.py` — `Config` class (yaml/json/dict loadable)
- `src/chessbot/model.py` — all model architectures (active: `build_pt_precond_smartgate`)
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
`build_pt_precond_smartgate` (`16m-precond-smartgate`) with:
- conv_filters=256, num_heads=8, dropout=0.03, pre_blocks=4
- 4x Conv(256) preconditioner → concat pos(256) → 512-d → 4x transformer blocks
- 8 specialized global accumulators: WDL(0), SmartGate(1), from/to policy(2-7)
- SmartGate: quality_logits + logsigmoid(gate) → suppress-only, bias init=4.0
- ~15.8M parameters, PyTorch float16
- Policy: 1858 legal logits scattered to 4288 flat
- Value: WDL 3-class softmax, STM-POV
- Inference: ORT+TRT fp16 via `infer_ort_trt.py` / `prepare_trt`

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

## Game Curriculum (16m_precond family)
- startpos 13%, pre_opened_mini 17.5%, UHO 20%, pre_opened 18%
- random_init 17%, random_middle_game 11.5%, piece_odds 2%, piece_training 1%
- UHO = Unbalanced Human Openings (UHO_XXL_2022_+100_+129.pgn), white-to-move at ply 16
- Bootstrapped from ~6M LC0 positions

## Validation Ladder
- Paired games from UHO positions (each played as both colors)
- Score > 0.5 vs Stockfish fixed depth → increment depth
- 16m_precond family started from pretrained checkpoint at 16 CPL (SF: Threads=1, Hash=256)

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

## Chat Response Format
User is dyslexic. During iterative status-update style responses (reporting what changed, asking something, flagging a concern, saying what's next), structure with visual headers: `=== DONE ===`, `=== OPEN ===`, `=== FLAGGED ===`, `=== NEXT ===`, each with terse bullet points. Omit a section entirely if empty rather than writing "none". Don't force this onto short direct answers or already-tabular output.
