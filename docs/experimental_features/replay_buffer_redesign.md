# Retrain Buffer / Replay Buffer Redesign (Proposed)

**Status: design locked, implementation deferred.** Simply raising `retrain_size`
and `training_queue_buffer` to the larger values used in `18m_6c4t_pretrain_run0`
appears to be resolving the CPL-drift problem this redesign was meant to fix,
across the board. Build this only if that stops holding up over time. Numbers
below are ballpark, not final.

## Motivation

The current retrain flow (`training_queue_buffer` flat list in RAM, drained by
`retrain_size` on each retrain) was producing degenerate drift shortly after
handoff from pretraining into selfplay — small, highly-correlated retrain
batches sampled from a buffer that was only ~2x the retrain size. Two things
were tried first and both helped: (1) bumping `retrain_size`/
`training_queue_buffer` to much larger multiples, (2) matching LR/batch_size/
adam_beta2 to what pretraining actually converged with instead of the selfplay
defaults that had drifted away from them.

This document describes a more structural fix — a disk-backed FIFO replay
buffer with a frozen anchor (pretrain) set for regularization — in case the
buffer-size bump alone doesn't hold up over longer runs.

## Base Constants (example numbers, not final)

| name | value | derivation |
|---|---|---|
| `retrain_batch_size` | 512 | base unit everything else is a multiple of |
| `shard_size` | 10,240 | 20 x batch_size — every tfrec file is exactly this many records, always, no exceptions |
| `live_buffer_size` | 102,400 | 10 shards — RAM-resident mixing pool before a shard gets written to disk |
| `retrain_primary_shards` | 4 | files pulled from `primary_retrain_buffer/` per retrain = 40,960 records |
| `retrain_replay_shards` | 4 | files pulled from `replay_buffer/` per retrain = 40,960 records |
| `retrain_historic_shards` | 2 | files pulled from `historic/` (pristine pretrain data) = 20,480 records |
| `replay_seed_shards` | 16 | 4x `retrain_replay_shards` — initial replay pool size, held constant thereafter |

Total records per retrain = 40,960 + 40,960 + 20,480 = **102,400 = exactly 200
batches of 512**. That's the entire single pass through the retrain dataset.

Naming rule for the eventual config: **don't** reintroduce a single
`retrain_size` scalar split via `//2` math to derive both a trigger threshold
and a total training-set size — that overload is what caused the primary-buffer
unbounded-growth bug found during design (see below). Use explicit separate
knobs (`shard_size`, `retrain_primary_shards`, `retrain_replay_shards`,
`retrain_historic_shards`) instead.

## Flow

1. Selfplay/rescore continuously appends samples (each tagged with `source`,
   identifying real selfplay/game provenance) to `live_buffer` in RAM.
2. Whenever `len(live_buffer) >= live_buffer_size`: sample `shard_size`
   records **without replacement, removing them from `live_buffer`**, write
   one tfrec.gz shard into `primary_retrain_buffer/`. This must be
   sample-then-remove — sampling without removal leaves the buffer still over
   threshold and the trigger fires again immediately on the same pool.
3. Repeat step 2 as new samples keep arriving. Net ~`shard_size` fresh samples
   are needed between successive shard writes (since removing one shard's
   worth drops the buffer just under threshold each time).
4. When `primary_retrain_buffer` file count >= `retrain_primary_shards`,
   trigger a retrain and **drain it completely to zero**. The trigger
   threshold must equal the drain amount exactly — using a bigger trigger and
   draining only half (an earlier version of this plan) is what causes
   unbounded disk growth over time.
5. Sample `retrain_replay_shards` files at random from `replay_buffer/` and
   remove them (discard) — they've served their purpose.
6. Sample `retrain_historic_shards` files at random from `historic/`,
   **read-only** — nothing removed, nothing added. Historic data is pristine
   and permanent.
7. Build a dict grouping the ~10 selected files by origin
   (`primary`/`replay`/`historic`) and hand it to the retrain worker(s).
   `make_dataset(files, shuffle_buffer, repeat=False)` (already exists in
   `src/chessbot/pretrain.py:96`, ported from the pretrain pipeline) shuffles
   file order, interleaves reads across all files concurrently via
   `cycle_length=AUTOTUNE`, shuffles records, batches — a single pass, then
   `StopIteration`. No resampling, no reopening files, exactly one epoch
   through the composed set.
8. Worker0's eval/progress pass restricts itself to the `primary`-origin
   files only (guaranteed never-before-trained-on real selfplay data),
   applying the existing `source`-presence check within those as a
   second-layer sanity filter. `historic` records never carry `source` and
   must be skipped by that check; `replay` records do carry `source` (they
   were primary once) but are excluded from eval by virtue of not being in
   the `primary` group for this retrain.
9. After training finishes: the files that came from `primary` in step 4 get
   **moved** (`os.rename`/`shutil.move`, not re-encoded) into
   `replay_buffer/`. The `replay` files sampled in step 5 are already gone
   (discarded). Net change to replay pool size: `-retrain_replay_shards
   +retrain_primary_shards` — zero if the two constants are equal, which
   keeps the replay pool size fixed at `replay_seed_shards` long-term.
10. Loop back to step 1.

**Steady-state replay dwell time**: with seed=16 and 4 replaced per cycle,
each replay sample survives ~4 retrain cycles on average before rotating out.
That ratio (`replay_seed_shards / retrain_replay_shards`, recommended 2-4x)
is the knob to tune if CPL stability responds to more/less replay depth.

**Cadence**: retrain frequency is purely a function of selfplay throughput —
however long it takes to accumulate `retrain_primary_shards * shard_size`
fresh samples.

## Storage: real directories, not an abstraction

`primary_retrain_buffer/`, `replay_buffer/`, and `historic/` are plain
directories on disk, not a virtual/manifest-tracked buffer. This works because
the flow above is strictly 1:1 bounded (drain-to-zero on trigger, replace-in
replace-out on replay) — no unbounded growth is possible by construction.
Benefits: crash recovery is free (directory contents *are* the state, derive
counts via glob rather than tracking counters that can desync), `mv` between
primary->replay is instant (same filesystem, no re-encode), and the buffers
stay directly inspectable (`ls`, spot-check a shard) without tooling.

**Hard rule: never delete from or add to `historic/`.** It is read-only for
the lifetime of the run. Keep it physically separate from
`primary_retrain_buffer/`/`replay_buffer/` (not a subdirectory of either) as
insurance against a future refactor accidentally routing a move/delete call
into it. `replay_buffer/` is seeded at run start by **copying** (never
moving) `replay_seed_shards` files out of `historic/`.

## File format: tfrec.gz (gzip), keep compression

Considered dropping gzip for latency, decided against it:

- Record size: `xc0h_board` int16[393] (786B) + `policy` float32[1858]
  (7,432B) + `wdl` float32[3] (~12B) ~= 8.3KB/record uncompressed. A
  10,240-record shard is ~85MB uncompressed; a full 10-shard retrain set is
  ~850MB uncompressed.
- `policy` is a legal-move visit distribution over 1858 slots with only ~35
  legal moves on average per chess position — >98% exact-zero floats, which
  gzip/DEFLATE compresses extremely well (long identical-byte runs). Expect
  shards closer to 10-15MB gzipped.
- Reads are effectively free regardless: the ported async
  read-ahead/shuffle-buffer pipeline (`EpochBufferThread` + `tf.data`
  prefetch, from `bootstrap_model_async_tfrec_pt.py`) overlaps I/O and
  decompression with the training step, so neither format choice is on the
  critical path.
- Given that, gzip's smaller footprint wins on write/seed-copy/replay-move
  cost and steady-state disk usage with no real downside. Keep gzip.

## Open / Not Yet Decided

- Module boundaries: what code moves out of `bootstrap_model_async_tfrec_pt.py`
  into shared modules vs. what's new for the retrain-side consumer.
  `make_dataset(..., repeat=False)` already supports everything needed
  (single-pass, multi-file interleave, shuffle buffer) — confirmed by
  reading `src/chessbot/pretrain.py:96`, should be a lift-and-shift rather
  than a rebuild.
- Exact `live_buffer_size` (100k vs 200k+) — leaning toward keeping it at
  ~10 shards (102,400) since its only job is smoothing temporal/game-order
  correlation before chunking, and replay+historic already carry the
  diversity load. Not confirmed by benchmarking.
- Disk growth safety valve for `primary_retrain_buffer/` in the event
  selfplay production outpaces retrain consumption for a stretch (deep-sim
  validation rounds, retrain bottleneck, etc.) — not addressed by the
  trigger-equals-drain fix alone if production genuinely races ahead
  indefinitely. Needs at least a backlog metric, possibly a cap.
