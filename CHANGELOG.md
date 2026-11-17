# Changelog

Brief notes on finished features, tracked from 2026-11-17. An entry is added
once a feature is done developing and tuning, not while it is in progress.

## 2026-11-17

- **Search controller (C++).** A small two-head temporal model (stability,
  quality) runs at every 100-root-visit check-in inside the search and
  decides when to stop. Two stop types: `ctrl` (the heads agree the leader
  is good enough) and `lock` (visit counts rule out every other move). A
  lock whose quality falls under a |Q|-sloped bar is extended instead, up to
  `ctrl_ext_max` times. Extensions that change the move gain roughly 20 TSL
  over the rejected move.
- **Rescore SF budgets.** Time-based search again (no depth floors).
  Decided positions get 0.8x the movetime; busy positions (more than 30
  legal moves or 60 material, from ply 10) get +20 ms.
- **Extension scoring.** On an extended move that changed its leader, the
  rejected and played moves are scored together in one multipv-2 search.
- **Controller logging.** Lean game logs keep the heads at the stop and at
  each rejected lock.
- **Logging (look and feel).** The epoch log's scenario table gains D%,
  CPL, TSL, CE and KL columns (from rescore), drops the SF-colour column,
  splits validation by xc0's colour, and adds selfplay / validation totals
  and a BRP row. Rescore prints every 100 games instead of every batch:
  game-weighted quality trends at 100 / 500 / 2000 games, then throughput,
  SF and targets, replacing the batch / last 10 / overall block.

  Before:
  ```
  scenario             sf          N      W      D      L    avg_plies
  vs_stockfish         white     161     72     81      8       110.9
  [rescore]  stats      |   CPL  |   TSL  |   CE  |   KL  | CPL>30 |
  [rescore]  last 10    |  6.05  |  7.10  | 0.51  | 0.35  | 0.062  |
  ```
  After:
  ```
  scenario             N     W     D     L    D%    CPL    TSL    CE    KL  plies
  vs_stockfish       307    80   158    69  51.5   7.90   9.85  0.52  0.36  112.5
  [rescore] trend (game EMA)   CPL    TSL    CE    KL   TSL>50  TSL>120
  [rescore] ema 500           6.03   7.08  0.50  0.34     2.6%     0.7%
  ```
