# Robust Selection Criteria

Robust Selection Criteria (RSC) is an alternative to pure visit-count move selection. Rather than always playing the most-visited move, RSC scores each candidate using a weighted ensemble of five signals and selects the move with the strongest combined case. It is used both for final move selection and as a component of early stopping.

## The Five Signals

RSC considers the top-N candidates (up to 5, minimum 100 visits each). For each, it collects:

- **Visit count** — raw simulation visits
- **Q** — mean action value (white-POV, flipped for black)
- **Qema** — exponential moving average of Q, smoothing out noisy early estimates
- **Visit share** — EMA of how frequently this child was selected relative to siblings during search
- **Qdelta_sign** — sign of the recent Q delta trend; positive means Q has been improving

Each signal is min-max normalized to [0, 1] across the candidate set, then converted to a probability distribution. The final RSC score is an equal-weight sum:

```cpp
const float w = 0.2f;
float score = w*p_vis[i] + w*p_q[i] + w*p_qe[i] + w*p_vs[i] + w*p_ds[i];
```

## Move Selection

RSC is activated above a configurable sim threshold (`robust_only_above`). Below that threshold, Xerces falls back to most-visited. When RSC is active, only candidates with at least `min(most_visits * 0.7, 2000)` visits are eligible — moves that haven't been explored enough to form a reliable estimate are excluded regardless of their score.

```python
visit_threshold = min(most_visits * 0.7, 2000)
best_uci = max(
    (d.uci for d in details if d.N >= visit_threshold and d.uci in rsc),
    key=lambda u: rsc[u]
)
```

## Why Not Just Most-Visited

Pure visit count can be gamed by PUCT: a move with high uncertainty gets a large U bonus and accumulates visits early, even if its Q ultimately converges to something mediocre. RSC requires multiple independent signals to agree. A move wins the RSC vote only if it's well-visited *and* has a good value *and* that value has been stable — not just because it looked exciting to explore early in the search.
