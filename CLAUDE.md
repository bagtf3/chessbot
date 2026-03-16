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
- Run configs live at: `C:\Users\Bryan\Data\chessbot_data\selfplay_runs\<run_tag>\config.yaml`
- Active run tag: `val_test`

## Key Files
- `src/chessbot/mcts_utils.py` — `MCTSTree` (wraps C++) and `ChessGame`
- `src/chessbot/looper.py` — `GameLooper`, batched inference, `StockfishThread`
- `src/chessbot/rescore.py` — post-hoc SF analysis, training data prep, rescoring
- `src/chessbot/review.py` — `GameViewer`, `RecordKeeper`, lc0 review tools
- `src/chessbot/config.py` — `Config` class (yaml/json/dict loadable)
- `src/chessbot/model.py` — all model architectures (active: `build_conformer_64x67`)
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
- No stacked history planes (intentional design choice — hypothesis: AZ history planes are cargo cult)
- vocab_size=21 in all models

## Active Model Architecture
`build_conformer_64x67` with:
- d_model_embed=128, vocab_size=21, conv_filters=256, conv_blocks=10
- transformer_layers=5, num_heads=8, ff_dim=1024, dropout=0.025
- ~16M parameters, mixed float16
- Policy: 64x67 logits (4288 flat, masked softmax at inference)
- Value: tanh scalar, STM-POV
- Inference wrapper `make_conv_infer`: XLA-compiled, applies uniform_eps + prior_clip_max in graph
- ~12k preds/sec, ~40k nodes/sec with cache

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

## Game Curriculum (val_test)
- startpos 13%, pre_opened_mini 17.5%, UHO 20%, pre_opened 18%
- random_init 17%, random_middle_game 11.5%, piece_odds 2%, piece_training 1%
- UHO = Unbalanced Human Openings (UHO_XXL_2022_+100_+129.pgn), white-to-move at ply 16
- Bootstrapped from ~6M LC0 positions

## Validation Ladder
- Paired games from UHO positions (each played as both colors)
- Score > 0.5 vs Stockfish fixed depth → increment depth
- Currently plateaued ~depth 11 (SF: Threads=1, Hash=256)

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
- No Unicode characters outside ASCII in print statements or log output — Windows subprocesses use cp1252 and will crash on characters like arrows or other non-ASCII symbols. Use ASCII equivalents (e.g. `->` not `->`).
