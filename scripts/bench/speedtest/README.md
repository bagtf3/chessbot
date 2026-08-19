# MCTS speed testing

Barebones fixed-work bench for the pyfastchess C++ search. Built to support
the optimization plan in `pyfastchess/perf_review.md`.

**YAGNI. Start small, add gear when a decision needs it.** An earlier version
of this rig had three stub policy modes, a play-through mode, and a multiproc
selfplay bench. All of it got deleted before it measured anything useful. If
you want to add a mode here, first name the decision it unblocks.

## The one idea

Pin the work, measure the time. Every position runs an exact sim count with
early stopping, dirichlet noise and move sampling switched off, so a C++
change moves the wall clock and nothing else.

Rate-over-a-mixed-workload was tried first and thrown out: adaptive early
stopping makes the sim count a function of float results, so any change that
perturbs floats also changes how much work happens. That confounds exactly
the A/B you are trying to run.

## Use

```
python -m scripts.bench.speedtest.fixtures --name default
python -m scripts.bench.speedtest.bench --fixtures <path> --n-sims 2000
python -m scripts.bench.speedtest.bench --fixtures <path> --rebuild
python -m scripts.bench.speedtest.compare <run_dir_a> <run_dir_b>
```

Everything is a plain function too, so it imports fine into Spyder.

Results land in `chessbot_data/speed_testing/runs/<tag>_<timestamp>/`, stamped
with both repos' git SHA + dirty flag. Rows are flushed after each position so
a crash mid-run does not lose the whole thing.

## What it reports

`sims/sec`, `us/sim`, `puct/sim`, `nodes/sec`, peak RSS -- overall and
**bucketed by ply**. The ply curve is the point: it is what settles whether
per-node board copying (perf_review Part 3) is worth touching.

`--rebuild` additionally advances the root on a cache-warm tree and re-times
reaching the same sim count. Under `reuse_tree: false` the rebuild path is
the hot path, so that number is the one Phases 1-2 actually move.

## Determinism

`stub_infer` derives a fake eval purely from the position zobrist, so the same
position gets the same eval in any process. With noise off, root visit counts
are reproducible; `rows[].visits` is saved for exactly that comparison. A
behaviour-preserving refactor must reproduce them.

Use `--backend real` to run the actual model instead (ort_trt / pt_eager).
That measures the full production stack; the stub isolates C++ search.

## Known crasher

`tests/repro_segfault_endgame.py` in the pyfastchess repo segfaults in pure
pyfastchess (no chessbot code) on an endgame FEN at ~50+ sims. Fixture sets
drawn from game logs can contain positions that hit it.
