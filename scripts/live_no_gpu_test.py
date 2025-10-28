#!/usr/bin/env python3
# live_no_gpu_test.py
# Purpose: launch many MCTS trees, farm collection to worker threads,
# main thread periodically polls pending_encoded(5) and commits random noise
# to the C++ raw cache, then resolves pending leaves on each tree.
#
# Requires: your project env where `looper.Config`, `chessbot.mcts_utils.MCTSTree`,
# and `pyfastchess.raw_cache_bulk_insert` are importable.

import threading
import time
import random
import numpy as np
import chess

# import local classes from your project (as in looper.py)
from chessbot.looper import Config
from chessbot.utils import random_init
from chessbot.mcts_utils import MCTSTree
from pyfastchess import raw_cache_bulk_insert, raw_cache_clear, priors_cache_clear, priors_cache_stats

# ---- params you can tweak ----
NUM_TREES = 12                # number of trees (main creates them)
WORKERS = 6                   # worker threads
TREES_PER_WORKER = NUM_TREES // WORKERS
COLLECT_MBS = 32              # call collect_many_leaves(n_new=COLLECT_MBS)
COLLECT_FASTPATH = 16
DURATION_SEC = 120            # how long the test runs
BATCH_TARGET = 128            # how many cache inserts per raw_cache_bulk_insert call
POLL_INTERVAL = 0.02          # main loop poll interval (s)
PRINT_EVERY = 10.0            # seconds between status prints
# -------------------------------

def make_random_raw_heads(factorized_bins):
    # Return (value_scalar, p_from, p_to, p_piece, p_promo)
    F, T, Kp, Kpr = factorized_bins
    # random logits -> softmax
    def rand_soft(n):
        a = np.random.rand(n).astype(np.float32)
        a /= a.sum()
        return a
    value = float(np.random.uniform(-1.0, 1.0))
    return (value, rand_soft(F), rand_soft(T), rand_soft(Kp), rand_soft(Kpr))

def worker_loop(worker_id, trees, stop_ev):
    # Each worker repeatedly calls collect_many_leaves on its trees to fill pending_nodes_
    while not stop_ev.is_set():
        for t in trees:
            # collect_many_leaves is a C++ call (fast)
            nn, nt, nc = t.collect_many_leaves(COLLECT_MBS, COLLECT_FASTPATH)
            # deliberately no extra Python-side work here — want C++ to run on CPU
            # tiny yield to allow scheduler, but keep load high
        # optional: tiny sleep to avoid 100% busy spin on tiny machines
        # time.sleep(0.0001)

def main():
    cfg = Config()
    cfg.factorized_bins = getattr(cfg, "factorized_bins", (64, 64, 6, 4))

    # create trees in main thread
    trees = []
    for i in range(NUM_TREES):
        board = random_init(5)
        t = MCTSTree(board, cfg)
        trees.append(t)

    # start worker threads and assign trees round-robin
    stop_ev = threading.Event()
    worker_threads = []
    for w in range(WORKERS):
        start = w * TREES_PER_WORKER
        end = start + TREES_PER_WORKER
        th = threading.Thread(target=worker_loop, args=(w, trees[start:end], stop_ev), daemon=True)
        th.start()
        worker_threads.append(th)

    print(f"Started {WORKERS} workers; {NUM_TREES} trees total. Running target {DURATION_SEC}s...")
    t_start = time.time()
    end_time = t_start + DURATION_SEC
    next_print = t_start + PRINT_EVERY

    # clear caches
    try:
        raw_cache_clear = getattr(__import__('pyfastchess'), 'raw_cache_clear', None)
        if raw_cache_clear:
            raw_cache_clear()
    except Exception:
        pass
    try:
        priors_cache_clear()
    except Exception:
        pass

    inserted_total = 0
    rounds = 0
    to_commit = []

    # safety knobs
    MAX_PER_TREE_PENDING_PROCESSED = 32   # don't process more than this many pending entries per tree per loop
    MAX_COMMIT_SECONDS = 5.0              # warn if cache bulk insert takes longer than this

    try:
        while time.time() < end_time:
            rounds += 1
            now = time.time()

            # Collect pending encoded from trees:
            for t in trees:
                # break early if we've reached the end_time
                if time.time() >= end_time:
                    break

                pending = t.pending_encoded(5)  # list
                if not pending:
                    continue

                # limit per-tree processing so one busy tree can't hog the loop
                num_proc = 0
                for (zob, planes) in pending:
                    if num_proc >= MAX_PER_TREE_PENDING_PROCESSED or time.time() >= end_time:
                        break
                    num_proc += 1

                    # Build fake random heads
                    value, pf, pt, pc, pr = make_random_raw_heads(cfg.factorized_bins)
                    to_commit.append((int(zob), float(value),
                                      np.asarray(pf, dtype=np.float32),
                                      np.asarray(pt, dtype=np.float32),
                                      np.asarray(pc, dtype=np.float32),
                                      np.asarray(pr, dtype=np.float32)))

                    # flush in chunked batches to avoid huge single insert
                    if len(to_commit) >= BATCH_TARGET:
                        commit_start = time.time()
                        try:
                            raw_cache_bulk_insert(to_commit)
                        except Exception as e:
                            print("raw_cache_bulk_insert exception:", e)
                        commit_time = time.time() - commit_start
                        if commit_time > MAX_COMMIT_SECONDS:
                            print(f"WARNING: raw_cache_bulk_insert took {commit_time:.2f}s")
                        inserted_total += len(to_commit)
                        to_commit = []

            # final small flush (but bounded so we don't block too long)
            if to_commit:
                commit_start = time.time()
                try:
                    raw_cache_bulk_insert(to_commit)
                except Exception as e:
                    print("raw_cache_bulk_insert exception:", e)
                commit_time = time.time() - commit_start
                if commit_time > MAX_COMMIT_SECONDS:
                    print(f"WARNING: raw_cache_bulk_insert took {commit_time:.2f}s")
                inserted_total += len(to_commit)
                to_commit = []

            # Resolve pending in trees, but cap per-tree resolve time so we don't stall
            for t in trees:
                if time.time() >= end_time:
                    break
                # quick resolve call (should be fast). If your resolve_pending can be long,
                # consider adding a small guard or a non-blocking variant on the C++ side.
                t.resolve_pending()

            # periodic status print
            now = time.time()
            if now >= next_print:
                next_print = now + PRINT_EVERY
                try:
                    pc = priors_cache_stats()
                except Exception:
                    pc = None
                elapsed = now - t_start
                print(f"[{int(elapsed)}s] rounds={rounds} total_inserted={inserted_total} priors_cache={pc}")
            # sleep a small amount to avoid tight busy loop
            time.sleep(POLL_INTERVAL)

    finally:
        # signal workers to stop and join them
        stop_ev.set()
        stop_join_deadline = time.time() + 2.0
        for th in worker_threads:
            remaining = max(0.0, stop_join_deadline - time.time())
            th.join(timeout=remaining)

        # final short drain/resolve pass (bounded)
        final_deadline = time.time() + 2.0
        # flush remaining commits once more
        if to_commit:
            try:
                raw_cache_bulk_insert(to_commit)
                inserted_total += len(to_commit)
            except Exception as e:
                print("raw_cache_bulk_insert exception at shutdown:", e)
            to_commit = []

        for t in trees:
            if time.time() >= final_deadline:
                break
            t.resolve_pending()

    print("Done. inserted_total:", inserted_total, "rounds:", rounds, "elapsed:", time.time() - t_start)


if __name__ == "__main__":
    main()
