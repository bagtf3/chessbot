# Model Training

## Pretraining

Before self-play begins, the model is pretrained on static data. This gives it basic chess structure before it has to generate its own training data.

Pretraining can include positions analyzed by Stockfish, published Lc0 data, existing Xc0 game data, or
other supervised targets. The goal is not to make a perfect engine immediately.
The goal is to reduce the amount of self-play training required.

A randomly initialized model gives weak priors and noisy values. Search can still
run, but it wastes a lot of effort. A pretrained model gives the search better
first guesses, which makes self-play refinement much more efficient.

![Pretraining metrics](../images/pretraining_metrics.png)

## Self-Play Refinement

After pretraining, Xerces improves through self-play.

In self-play, Xerces plays games against itself. For every move, it stores the
position, the final search visit distribution, and eventually the game result.
Those records become training examples.

A simplified training row looks like this:

```text
position: board state before the move
policy target: MCTS visit distribution
value target: game outcome or refined value target
```

The policy target teaches the model to imitate the search. The value target
teaches the model which positions are actually converting into wins, losses, or
draws.

After enough games are collected, the model is retrained. Then the updated model
plays more self-play games, generating a stronger batch of data. The cycle
repeats.

![Self-play model metrics](../images/selfplay_model_metrics.png)

![Self-play CPL and BMR](../images/selfplay_cpl_bmr.png)

## Training Mode vs Competition Mode

Xerces behaves differently depending on whether it is generating training data or
trying to play the strongest possible move.

### Training Mode

In training mode, the goal is not only to win the current game. The goal is to
produce useful data.

Training mode may use more randomness, exploration, sampling, or noise. This
helps the engine see a wider range of positions instead of repeating the same
lines forever.

The engine records search distributions, results, metadata, and later rescoring
information. These outputs are more important than a single game's result,
because they feed the next training cycle.

Training mode is designed to be curious.

### Competition Mode

In competition mode, the goal is simple: play the strongest move available.

Competition mode reduces randomness and uses the search result more directly.
The engine usually chooses the most robust move from the root search, often the
move with the strongest visit count or best final selection score.

Exploration still exists inside the search through PUCT, but the final move
choice is much more deterministic. The engine is no longer trying to create broad
training data. It is trying to convert the current position.

Competition mode is designed to be ruthless.
