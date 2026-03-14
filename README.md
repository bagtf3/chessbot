# Xerces (Xc0)

MCTS chess engine with a neural network policy/value head, trained entirely through self-play against Stockfish rescoring.

- Monte Carlo Tree Search with PUCT selection
- Neural network evaluation (multiple architectures. Currently: [MHA->CONV->CONV]x10 conformer , ~16M params)
- Stockfish post-hoc rescoring for blunder detection and visit redistribution
- Async retrain loop running between self-play rounds

---

# Telemetry

## Selfplay

<p align="center">
  <img src="docs/images/selfplay_telemetry.jpg" width="750">
</p>

Live stats during self-play: moves/sec, leaf evals/sec, games/hour, cache hit rate, PUCT stats, per-scenario results.

---

## Retrain Loop

<p align="center">
  <img src="docs/images/retrain_loop.jpg" width="750">
</p>

Retraining runs as a subprocess between self-play rounds. Tracks value MSE, value correlation, policy cross-entropy, top-1 accuracy, and CE gain vs uniform.

---

## MCTS Move Analysis

<p align="center">
  <img src="docs/images/gameviewer.jpg" width="750">
</p>

Post-hoc analysis via review.py::GameViewer. Per-move breakdown: visit counts, policy prior vs search distribution, Q/Qe values, PUCT contributions, entropy, KL divergence, and Stockfish comparison.

---

## Lc0 Comparison

<p align="center">
  <img src="docs/images/lc0_compare.jpg" width="750">
</p>

Compare Xerces policy outputs against Leela Chess Zero to look at distribution differences and KL divergence.
Summon during game review with `Lc0 n nodes`

---

## Deeper Stockfish Evaluation

<p align="center">
  <img src="docs/images/stockfish_eval.jpg" width="750">
</p>

Re-run Stockfish on any position during game review via `sf <depth>` to search `depth` plies (sf 24 shown).

---

# Setup

    pip install -r requirements.txt
    export PYTHONPATH=./src


# Quick Start

## Run Selfplay

    python scripts/run_selfplay.py <run_tag>

## Bootstrap / Pretrain

    python scripts/bootstrap_model_async_tfrec.py

## Architecture Benchmarks

    python scripts/model_variant_speed_test.py
