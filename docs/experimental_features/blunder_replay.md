# Blunder Replay

When the rescoring pipeline detects a blunder, rather than just adjusting the training label for that position, Xerces queues a new selfplay game starting from the blundered position. The idea is to generate fresh data at exactly the positions where the model is currently weakest, rather than waiting for those positions to appear organically in future games.

## How It Works

During rescoring, `collar_z_map()` scans the SF evaluation trace for positions where the played move lost significant centipawns relative to the SF best move. Each detected blunder produces a replay spec — the starting FEN plus the move sequence up to the blundered ply:

```python
self.blunder_replay_specs.append({
    'fen': game_data['start_fen'],
    'moves': game_data['moves_played'][:blunder_ply],
    'stockfish_is_white': actual_stm_is_white,
    'source_game_id': gid
})
```

The orchestrator drains `blunder_replay_specs` between rescoring cycles and injects these as new game specs into the worker pool. The replay game plays out from the blundered position under normal selfplay rules — full MCTS, normal sim budget, normal adjudication.

Blunders are classified into two types:

- **winning** — the blundering side was already ahead when the mistake happened. These are always replayed; a model that throws away won positions is a high-priority training target.
- **neutral** — the position was approximately equal. These are replayed at 25% probability to avoid flooding the queue with noisy positions from equal-ish games.

Blunders from Stockfish (in vs-SF games) are skipped — there's no value in replaying a position where the engine played well and SF made the mistake.

## Effect

Blunder replay creates a feedback loop between rescoring and data generation. The model makes a mistake, SF catches it, and the next training batch includes more games from that position. Over time this concentrates data on the model's current blind spots rather than uniformly sampling positions it already handles well.
