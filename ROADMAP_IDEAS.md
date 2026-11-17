# Roadmap: Ideas

Less formed thoughts that are interesting but not on the agenda.

## Board-less nodes

- Nodes stop holding a board copy; each descent pushes moves onto a walk
  board, as lc0 does.
- Gains: far less RAM, better cache locality, cheaper `advance`. Raw CPU per
  descent is likely a wash.
- Matters for single-game TCEC-style play, not GPU-bound selfplay.
- Touches everything that reads `node->board`: zobrist keys, terminal and
  repetition checks, sharp tagging, the encoders, details and PV output.
- One option: a pool of walk boards per tree, sized to the most in-flight
  leaves; leaves waiting on the NN keep their board.

## Controller as an oracle, with non-blocking check-ins

- The controller owns each search's causal state; trees send check-in
  features as requests and read results when they need them.
- A check-in blocks only if its result could stop the search; otherwise the
  tree keeps searching and polls later.
- Run the controller on its own C++ thread so the queue drains while the
  worker waits on GPU inference.
- Costs: request queue, per-tree in-order application, a worker thread,
  cancelling on game end; the model would need per-search `shared_ptr`
  ownership to be reload-safe.

## Lc0Server and a vs_leela game mode

- Same pattern as the SFServer (`ROADMAP_INFRA.md`): one process serving
  lc0 positions, in place of today's bespoke lc0 worker with its own
  CUDA/TRT/RAM partition.
- Runs an lc0 network inside the xc0 search harness, for speed.
- Unlocks vs_leela: play lc0 like vs_stockfish, for expert learning. Needs
  two trees and two caches so each move's NN and search are clear.

## BRP (back burner)

- Today every worker goes to BRP during a round, so selfplay stalls.
  Option: send all BRP to one worker.
- Redesign: about half from a static position set (tracks progress), half
  from new blunders tried once after they are 5+ epochs old. First-try pass
  rate is the main stat.
- Open question: whether BRP is worth the compute at all.

## Dynamic game generation from recent games

- Look at recent games (e.g. the last 10k) and generate games that maximize
  learning: contested lines, under-explored lines, and lines where policy or
  value was miscalibrated.
