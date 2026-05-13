# Selfplay Suite

## Orchestration

`scripts/run_selfplay.py` is the entry point for a training run. It spawns N worker processes and runs a persistent outer loop that manages rescoring, retraining, and validation. Each worker runs `child_looper()`, which initializes a `GameLooper` and calls `run()`.

Workers communicate with the main process through several multiprocessing queues:

- `recent_q` — game results and per-game metadata (W/D/L, sims used, CPL after rescoring)
- `telemetry_q` — per-worker performance metrics (MPS, LPS, batch fill rate, cache hit rate)
- `pause_q` / `unpause_q` — synchronization signals sent during model reload between retrain epochs

The main process also maintains the retrain trigger: when the `Rescorer` has accumulated enough training examples (`retrain_size`), it pauses all workers, launches the retrain subprocess, waits for completion, broadcasts the new model path, then unpauses workers.

![Selfplay telemetry](../images/selfplay_telemetry.JPG)

The live telemetry view aggregates across all workers and updates each reporting interval. Key metrics: moves/sec (MPS) and leaf evaluations/sec (LPS) reflect game and NN throughput respectively; games/hour gives the effective data generation rate; cache hit rate shows what fraction of leaf collections were resolved by the C++ priors cache without NN inference; PUCT stats and per-scenario W/D/L breakdowns make it easy to spot if one game type is producing systematically lopsided results.

## GameLooper

`GameLooper` manages a pool of `active_games` (up to `games_at_once` concurrent `ChessGame` instances) and drives the inference loop.

Each iteration:

1. `collect_many_leaves(this_mbs, max_fastpath)` — called across all active games simultaneously, gathering new leaf nodes that need NN evaluation. Cache hits and terminals are resolved in C++ and don't surface here.
2. Leaf boards are stacked into a batch. The batch is padded to the next valid size for XLA trace reuse.
3. `fwd((boards_np,))` — NN inference. Returns `(policy_logits [B, 4288], value [B, 1])`.
4. `raw_cache_bulk_insert_np(zobrists, values, logits)` — inserts results into the C++ raw cache.
5. Per-leaf: `tree.apply_result(node, priors, wdl)` — expands the node and triggers backprop.
6. Games that have reached a terminal state are finalized, serialized to pickle, and replaced with fresh games.

The inference step is intentionally kept as the sole GPU operation. There is no per-game GPU work; everything else is CPU-side C++ or Python orchestration.

**Pause and reload.** When the main process sends a pause signal, `GameLooper` drains any in-flight leaves, calls `tf.keras.backend.clear_session()` (or `torch.cuda.empty_cache()` for the PyTorch backend) to release VRAM, and blocks on the unpause queue. On resume it reloads the model from the path provided in the unpause message and recompiles the inference function. VRAM is released before the retrain subprocess starts so both fit on the same GPU.

## StockfishThread

`StockfishThread` runs a Stockfish engine on a background daemon thread, handling positions that need engine evaluation during selfplay — primarily for games played against Stockfish and for the vs-SF validation rounds.

Games submit a (game_id, board_fen) pair via `submit()`. The thread drains its input queue, runs Stockfish to the configured depth, and pushes (game_id, result_tuple) to a response queue. The result tuple contains `(eval_cp_stm_pov, best_move_uci, wdl_tuple)`. Games poll the response queue at the start of each move.

## Game Curriculum

`GameGenerator` selects starting positions according to a weighted scenario mix configured per run. The active `val_test` curriculum is:

| Scenario | Weight | Description |
|---|---|---|
| `startpos` | 13% | Standard starting position |
| `pre_opened_mini` | 17.5% | Short opening lines (<= 2 moves) |
| `UHO` | 20% | Unbalanced Human Openings, white to move at ply 16 |
| `pre_opened` | 18% | Longer opening lines (> 2 moves) |
| `random_init` | 17% | Random legal position from the opening phase |
| `random_middle_game` | 11.5% | Random init advanced to ply 20-31 |
| `piece_odds` | 2% | Material-unbalanced positions |
| `piece_training` | 1% | Focused endgame piece training positions |

UHO positions come from `UHO_XXL_2022_+100_+129.pgn`, a corpus of unbalanced human games. All UHO positions are white-to-move at ply 16, ensuring the engine sees non-trivial positions with varied pawn structures early in training.

## Game Adjudication

Games terminate on checkmate or draw by the rules, but several early-exit conditions are also active:

- **Eval collar** — if one side's evaluation stays above a threshold for `collar_n_consec` consecutive moves, the game is adjudicated as a win for that side.
- **Eval draw** — if all evaluations within a rolling window stay near zero, the game is adjudicated as a draw.
- **Max game length** — hard cap to prevent runaway games.

Syzygy tablebase adjudication and material-diff cutoffs are configurable but disabled in the current `val_test` run.
