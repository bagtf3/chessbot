# Roadmap: Model and Training Data

Planned and candidate work on the network, its heads, and what it trains
on.

## Next-gen model

- Pretrain and deploy the 25-30M model with smolgen, masked policy CE
  instead of SmartGate, and the other features planned for it.

## Mate distance head (MDH)

- A radar plus distance: 0 when quiet, rising as a forced mate gets close.
  Preferred over MLH: our games end early (resign, eval draw, syzygy, max
  length), so game-length labels are cut short.
- t1-big's MLH (2026-11-11, ~9.4k endgame positions): ranks SF mate
  distance with Spearman 0.78, radar AUC 0.87, but adds only ~1 pt over the
  prior at picking SF's best move, all within ~10 plies of mate.
- Labels: SF mate distances from rescore and MateLock++ proofs.
- Shape: softmax buckets for "we mate in" / "we get mated in" plus "none",
  horizon ~20 plies. Radar = `1 - p_none`; closeness from bucket means.
- Uses: radar triggers the MateLock++ scout; closeness breaks ties only when
  Q is saturated; for + against as a sharpness signal. Q stays pure.
- Build after MateLock++ (`ROADMAP_SEARCH.md`), once labels accumulate.

## More unbalanced openings

- The UHO book is roughly +99 to +149; look for more unbalanced books.
- Curriculum can be the difference maker: compute should go where it
  matters most.

## Is the lc0 teacher working?

- Two signals: lc0-slice validation (`eval_progress_lc0.csv`) improves, and
  xc0's ESL inaccuracy and blunder rates fall.
- The lc0 slice moves with what the streams send, so read trends within a
  stream regime. Blunder rates are the steadier read.
- Track blunder rates per epoch next to the lc0 metrics, split by scenario,
  so curriculum shifts don't pass for improvement.
