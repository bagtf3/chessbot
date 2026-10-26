#!/usr/bin/env python3
"""
Long-run, low-verbosity threading prototype.

Purpose: run for RUN_SECONDS with multiple worker threads producing
C-level work (NumPy BLAS) so you can watch CPU core utilization.
Only prints a small summary at the end.

Usage:
  python threading_prototype_long.py
"""
import os
# Make individual NumPy BLAS calls single-threaded so multiple threads can use different cores.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import threading
import queue
import time
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

# -----------------------
# Config (tweak these)
# -----------------------
NUM_WORKERS = 6         # number of native worker threads (set close to #cores)
BATCH_TARGET = 32       # how many encodings batcher tries to drain per model call
ENC_DIM = (512, 512)    # encoding generator dims (bigger dims => more CPU per np.dot)
WORKER_THINK_ITER = 8   # number of np.dot iterations per produced encoding (increase to load more)
GPU_SIM_RUNTIME = 0.010 # simulated "model" time (seconds)
RUN_SECONDS = 60        # total run time
QUEUE_POLL_SLEEP = 0.005  # small sleep when no work to avoid busy-waiting

# -----------------------
# Shared structures
# -----------------------
encoding_queue = queue.Queue()
cache = {}
cache_lock = threading.Lock()
pred_events = {}
pred_events_lock = threading.Lock()

# Instrumentation
compute_intervals = []           # list of (start, end, worker_id)
compute_intervals_lock = threading.Lock()
batches_processed = 0
batches_processed_lock = threading.Lock()
total_consumed = 0
total_consumed_lock = threading.Lock()

# -----------------------
# Helpers
# -----------------------
def make_static_matrices(seed, dim=ENC_DIM):
    rng = np.random.RandomState(seed)
    A = rng.randn(dim[0], dim[1]).astype(np.float32)
    B = rng.randn(dim[1], dim[0]).astype(np.float32)
    return A, B

# -----------------------
# Worker
# -----------------------
def worker_loop(worker_id, stop_event):
    """
    Worker repeatedly produces encodings until stop_event is set.
    Each produced encoding is enqueued for the batcher.
    """
    gid = 0
    # have per-worker matrices to avoid re-seeding cost
    while not stop_event.is_set():
        # produce a synthetic encoding (heavy C-level work)
        A, B = make_static_matrices((worker_id << 16) ^ gid)
        t0_compute = time.time()
        enc = None
        for _ in range(WORKER_THINK_ITER):
            tmp = A.dot(B)                 # BLAS in C
            enc = np.mean(tmp, axis=1).astype(np.float32)
        t1_compute = time.time()
        # record compute interval
        with compute_intervals_lock:
            compute_intervals.append((t0_compute, t1_compute, worker_id))

        encoding_id = f"{worker_id}:{gid}:{time.time_ns()}"
        send_ts = time.time()

        # place a simple event so the worker could wait (not used here to minimize overhead)
        ev = threading.Event()
        with pred_events_lock:
            pred_events[encoding_id] = ev

        # push to shared queue
        encoding_queue.put((gid, worker_id, send_ts, encoding_id, enc))

        # don't wait here — this worker produces continuously for CPU load
        gid += 1

# -----------------------
# Batcher
# -----------------------
def process_one_batch_nonblocking():
    """
    Pull up to BATCH_TARGET items from encoding_queue without blocking long.
    Simulate a batch forward, commit preds to cache, signal events.
    Returns: consumed_count, model_time_s
    """
    batch = []
    try:
        item = encoding_queue.get_nowait()
        batch.append(item)
    except queue.Empty:
        return 0, 0.0

    while len(batch) < BATCH_TARGET:
        try:
            itm = encoding_queue.get_nowait()
            batch.append(itm)
        except queue.Empty:
            break

    X = np.stack([it[4] for it in batch], axis=0)
    t_model_start = time.time()
    # fake model work
    _ = np.sum(X, axis=1)  # cheap op, heavy-ish already from stacking
    time.sleep(GPU_SIM_RUNTIME)
    t_model_end = time.time()
    model_time = t_model_end - t_model_start

    # commit preds and signal
    commit_ts = time.time()
    for i, it in enumerate(batch):
        gid, worker_id, send_ts, encoding_id, enc = it
        pred_val = float(i)  # dummy
        with cache_lock:
            cache[encoding_id] = {"pred": pred_val, "send_ts": send_ts, "commit_ts": commit_ts}
        with pred_events_lock:
            ev = pred_events.get(encoding_id)
        if ev is not None:
            ev.set()

    # update global counters
    with batches_processed_lock:
        global batches_processed
        batches_processed += 1
    with total_consumed_lock:
        global total_consumed
        total_consumed += len(batch)

    return len(batch), model_time

def batcher_loop(stop_event):
    """
    Continuously drain the queue until stop_event and the queue is empty.
    """
    while not stop_event.is_set() or not encoding_queue.empty():
        consumed, model_time = process_one_batch_nonblocking()
        if consumed == 0:
            time.sleep(QUEUE_POLL_SLEEP)
            continue
    # done

# -----------------------
# Main runner
# -----------------------
def run_long():
    stop_event = threading.Event()
    # start batcher thread
    batcher = threading.Thread(target=batcher_loop, args=(stop_event,), daemon=True)
    batcher.start()

    # start workers
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = []
        for wid in range(NUM_WORKERS):
            futures.append(ex.submit(worker_loop, wid, stop_event))

        # run for RUN_SECONDS with minimal printing
        t_start = time.time()
        try:
            while time.time() - t_start < RUN_SECONDS:
                # sleep quietly; user monitors CPU externally
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Interrupted by user; stopping early.")
        # signal stop
        stop_event.set()
        # wait for workers to end (they will exit when they check stop_event)
        # futures will return eventually; we won't call result() to avoid blocking too long
        # but give a short grace period for threads to finish producing and batcher to drain
        time.sleep(1.0)
        # ensure batcher drains remaining items
        stop_event.set()
        batcher.join(timeout=5.0)

    # final summary (minimal)
    print("\n--- run summary ---")
    print(f"RUN_SECONDS: {RUN_SECONDS}")
    print(f"NUM_WORKERS: {NUM_WORKERS}, WORKER_THINK_ITER: {WORKER_THINK_ITER}, ENC_DIM: {ENC_DIM}")
    print(f"Total batches processed: {batches_processed}")
    print(f"Total encodings consumed: {total_consumed}")
    # compute concurrency max
    with compute_intervals_lock:
        intervals = list(compute_intervals)
    events = []
    for (s, e, w) in intervals:
        events.append((s, 1)); events.append((e, -1))
    events.sort()
    cur = 0; max_conc = 0
    for t, d in events:
        cur += d
        if cur > max_conc:
            max_conc = cur
    print("Recorded compute intervals:", len(intervals))
    print("Max concurrent compute intervals observed:", max_conc)
    # sample stats to give a feel (not verbose)
    if total_consumed > 0:
        print(f"Avg batch size (approx): {total_consumed / batches_processed:.2f}")
    print("Done.")

if __name__ == "__main__":
    print("Starting long-run threading prototype. Minimal output; monitor CPU with htop/Task Manager.")
    run_long()
