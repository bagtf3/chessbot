# Retrain Buffer / Replay Buffer Redesign (v2)

**Status: live, first real test running as of 2026-07-29.** v1 of this design
was built and torn out (`git clean`) after selfplay testing showed it wasn't
holding up. v2 is now fully coded and running against a live selfplay run
(`18m_6c4t_pretrain_run0`). Not claimed flawless yet -- this is the first time
the full cycle has executed against real CUDA/GPU. Numbers below are still
module-level constants, to be promoted to config params once the shape is
validated against real runs.

## Why this exists

Two failure modes have been fought repeatedly on this project and this
redesign targets both at once:

1. **Buffer too small** -> retrain batches are small and highly correlated
   (drawn from a narrow recent window of selfplay), producing degenerate CPL
   drift shortly after handoff from pretraining into selfplay.
2. **Buffer too large / naive fix** -> pause/predict/load/train juggling
   becomes inefficient: workers sit paused too long, or GPU contention between
   validation prediction and training spikes badly.

The goal is a buffer large enough to properly decorrelate retrain batches
(replay + historic anchor data mixed with fresh primary data) while keeping
the pause window during a retrain cycle as short as the actual training step,
via a headstart-loading worker architecture instead of a blocking one.

## What to watch for / success criteria

- **CPL drift stability** post-handoff from pretrain into selfplay -- the
  original symptom this whole redesign chain (buffer-size bump, then this)
  was built to fix.
- **Pause duration per retrain cycle** should be close to just the actual
  training step (`retrain_pt`'s one epoch over 204,800 records), not
  training + validation + data loading, since the 31/32 headstart moves
  replay/historic loading (and validation) before the pause starts.
- **`primary_buffer/` file count** should oscillate in a bounded range
  (24-32) rather than growing over the course of a run -- if it grows
  unbounded, retrain throughput isn't keeping pace with selfplay production
  (see Open Risks).
- **No stale-weight windows**: selfplay workers should always resume on the
  freshly retrained model, not a cached/stale TRT engine.
- **`replay_buffer/` size** should stay fixed at `replay_seed_shards` (64)
  long-term, not drift.

## Base Constants (`src/chessbot/replay_buffer.py`)

| name | value | derivation |
|---|---|---|
| `retrain_batch_size` (cfg) | 512 | base unit everything else is a multiple of |
| `SHARD_SIZE` | 10,240 | 20 x batch_size -- every shard file is exactly this many records |
| `LIVE_BUFFER_SIZE` | 102,400 | 10 shards -- RAM-resident pool, replaces `rescorer.training_data` |
| `PRIMARY_TRIGGER_SHARDS` | 32 | files in `primary_buffer/` needed to fire a retrain = 327,680 records |
| `RETRAIN_PRIMARY_SHARDS` | 8 | files sampled from `primary_buffer/` per retrain = 81,920 records |
| `RETRAIN_REPLAY_SHARDS` | 8 | files sampled from `replay_buffer/` per retrain = 81,920 records |
| `RETRAIN_HISTORIC_SHARDS` | 4 | files sampled from `historic/` per retrain = 40,960 records |
| `REPLAY_SEED_SHARDS` | 64 | initial + steady-state `replay_buffer/` size |

Total per retrain = 81,920 + 81,920 + 40,960 = **204,800 records = 400 batches
of 512**, one pass (`retrain_pt` runs exactly 1 epoch).

`primary_buffer/` trigger (32 files) is intentionally larger than the sampled
amount (8): after a retrain samples 8 of 32 and moves them out, 24 remain, so
file count oscillates 24-32 rather than draining to zero -- gives the random
sample more of a temporally-mixed pool to draw from. This is bounded *only if*
retrain throughput keeps pace with selfplay production (see Open Risks).

## Worker Architecture (`src/chessbot/retrain_worker.py`)

The retrain worker is a real `ctx.Process(target=run_retrain_worker, ...)`
with a dedicated `msg_q`/`result_q` pair -- spawned the same way selfplay
workers are, not a `subprocess.Popen` script polled for exit (the old
`scripts/retrain_worker.py` CLI script has been deleted). It is **spawned
early, at 31/32 primary shards** (not at process start, not at 32/32) to get
a headstart on replay/historic loading, and is **torn down and joined/
terminated after every retrain cycle** -- a deliberate VRAM-reclaim policy.

Message protocol:

- main -> worker, sent immediately on spawn (31/32):
  `{"cmd": "preload", "replay": [8 paths], "historic": [4 paths]}`
- main -> worker, sent once primary hits 32/32:
  `{"cmd": "start", "primary": [8 paths]}`
- worker -> main: `{"cmd": "retrain_ready"}` -- validation is fully done and
  `predictions_latest.pkl` is already on disk by this point.
- worker -> main: `{"cmd": "retrain_done", "ok": bool, "error": str|None}`

### Worker-side flow (`retrain_worker_body`)

1. On spawn: import torch/model code, load `cfg`, block on `msg_q.get()` for
   `preload` (arrives near-instantly since main sends it right after spawn).
2. Read the 8 `replay` + 4 `historic` files into RAM -- this is the headstart
   work, overlapping with primary_buffer's last shard filling up. Historic
   reads lazily import TF here (env vars for log suppression are set via
   `os.environ.setdefault` *before* this first `import tensorflow` anywhere
   in the process, and `tf.get_logger().setLevel("ERROR")` -- verified clean,
   no warning spam).
3. Block on `msg_q.get()` for `start` (32/32 signal).
4. Load latest model, fp16 eval mode (`.half().eval()` + `no_grad()`).
   Read the 8 `primary` files, run prediction over all 81,920 records.
5. Write `predictions_latest.pkl` atomically (temp + `os.replace`) as
   `{"samples": [...], "preds": [...]}` -- self-contained, so the main
   process never needs to re-read the primary shards from disk to know what
   the true targets were.
6. Drop the eval model, `torch.cuda.empty_cache()`.
7. `shuffle_rewrite_primary`: shuffle the already-loaded 8-file primary
   record pool **across file boundaries** (not a per-file permutation --
   verified empirically that record IDs from file 3 can land in file 1 after
   rewrite) and rewrite back into the same filenames, atomically. No
   re-read from disk (fixed a redundant double-read during implementation).
8. Combine primary + replay + historic (204,800 records), shuffle again,
   build `X, P, Y, vwht, pwht`.
9. Send `retrain_ready`, sleep 10s (stopgap, see Open Risks), load a *fresh*
   model instance (separate from the tossed-out eval copy), train one epoch
   via the existing `retrain_pt` (`autocast("cuda")` + `GradScaler`, fp32
   master weights / fp16 compute -- unchanged from pre-redesign code, never
   raw `.half()`-the-model training), save.
10. Send `retrain_done`, hard-exit via `os._exit(0)` after flushing stdout/
    stderr (TF hangs on normal interpreter exit otherwise -- same pattern as
    `shuffle_xc0.py`/`bootstrap_model_async_tfrec_pt.py`).

### Main-process flow (`scripts/run_selfplay.py`)

- On `retrain_ready`: pause selfplay workers. Nothing else -- metrics wait
  until after retrain_done.
- On `retrain_done`, **in this exact order** (hard rule, confirmed with the
  user after a real ordering bug was caught and fixed during implementation):
  1. join/terminate the worker process
  2. reload config from yaml, sync into `rescorer`/SF threads
  3. if ok and `inference_backend == ort_trt`: rebuild the TRT engine for the
     new weights (`prepare_trt`) -- **must** happen before unpause, since
     selfplay workers only reload their model/engine right after receiving
     `unpause`; unpausing first would risk them grabbing a stale engine or
     each redundantly recompiling their own
  4. **unpause** -- first real action after the TRT rebuild prerequisite
  5. move the 8 (now reshuffled) primary files into `replay_buffer/`,
     discard the 8 replay files sampled at preload time
  6. `aggregate_metrics` -- console print, both CSVs, plots, unchanged from
     before the redesign except it now reads the self-contained samples+preds
     payload instead of a separately-tracked `pending_pred_samples`
  7. back to the rescoring loop

## Live Buffer (`rescorer.live_buffer`, replaces `rescorer.training_data`)

`LiveBuffer` holds up to 102,400 records in RAM. `Rescorer.tick()` calls
`flush_all_ready()` every tick, writing a `.pkl.gz` shard to `primary_buffer/`
whenever the pool is at capacity. On startup, `remaining_untrained.pkl` (if
present) loads straight into `live_buffer.records`, then gets flushed the
same way -- for the very first live test, that file held 326,568 leftover
records, immediately producing 31 shards (one short of the 32/32 trigger).

## Storage: real directories

`primary_buffer/`, `replay_buffer/`, `historic/` are plain directories, state
is directory contents. `historic/` (`BOOTSTRAP_TFREC_DIR` in `.env`, already
existed -- no new env var was needed) is never written to, moved from, or
deleted from. `replay_buffer/` is seeded once at run start by **copying**
64 files at random from `historic/`; net change per retrain cycle is
+8 (from primary) / -8 (sampled and discarded), keeping it fixed long-term.

## File Formats -- verified mixed-format reading

`historic/` and freshly-seeded `replay_buffer/` entries are `.tfrecord.gz`
(TF `tf.train.Example`, fields `xc0h_board`/`policy`/`wdl`); everything
`LiveBuffer` or the retrain worker writes is `.pkl.gz` (7-tuple records:
`x, mask, policy, Y, vwht, pwht, source`). `replay_buffer/` therefore holds a
genuine mix of both formats simultaneously once cycles start rotating primary
files in -- `read_shard()` dispatches per-file on extension and both
`list_shard_files`/`sample_files` are extension-agnostic. Verified directly:
seeded 3 real historic `.tfrecord.gz` files into a test replay dir alongside
a synthetic `.pkl.gz` shard, loaded and consolidated into training arrays
with no errors, correct shapes (`(N, 393)` boards matching this run's
`encoding_type: xc0h, history_K: 6`).

## Naming / Style

No `_leading_underscore` names (an old `import random as _random` alias was
also cleaned up along the way). Sizes/ratios stay as module constants until
the shape is validated, then get promoted to config params.

## Open Risks

- **Primary buffer unbounded growth**: the 32-trigger/8-sample asymmetry is
  bounded only if retrain throughput keeps pace with selfplay production. If
  selfplay persistently outpaces retrain (long deep-sim/validation stretches,
  retrain backlog), `primary_buffer/` grows without bound. No safety valve
  yet -- carried over unresolved from v1, watch file count over long runs.
- **`sleep(10)` before training**: worker sends `retrain_ready` then sleeps a
  fixed 10s rather than waiting for an explicit pause-confirmed ack from
  main. If pausing every selfplay worker ever takes longer than that, training
  could start while a worker is still mid-move on the GPU. Fine as a first
  pass; may need a real ack message later if it causes problems in practice.
