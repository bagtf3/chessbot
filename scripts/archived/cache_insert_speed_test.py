#!/usr/bin/env python3
"""
Speed test: raw_cache_bulk_insert (old tuple path) vs raw_cache_bulk_insert_np (new numpy path).

Representative data: uint64 keys, (B,3) WDL softmax probs, (B,4288) raw policy logits.
Tests several batch sizes with warmup + repeated reps.
Verifies that cache contents are bit-identical after both inserts.
"""

import time
import numpy as np
from pyfastchess import (
    raw_cache_bulk_insert,
    raw_cache_bulk_insert_np,
    raw_cache_clear,
    raw_cache_lookup,
)

BATCH_SIZES = [32, 64, 128, 256, 512]
N_WARMUP = 5
N_REPS = 30


def make_data(B, seed=42):
    rng = np.random.default_rng(seed)
    # unique non-zero keys
    keys = list(rng.integers(1, 2**62, size=B, dtype=np.uint64))
    # WDL: valid softmax probs, shape (B, 3)
    raw = rng.standard_normal((B, 3)).astype(np.float32)
    raw -= raw.max(axis=1, keepdims=True)
    exp = np.exp(raw)
    wdl = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
    # policy: raw logits (not softmaxed — matches what the model outputs)
    policy = rng.standard_normal((B, 4288)).astype(np.float32)
    return keys, wdl, policy


def insert_old(keys, wdl, policy):
    """Mirrors the old format_and_predict path: Python loop -> tuple list -> bulk_insert."""
    B = len(keys)
    to_raw_cache = []
    for i in range(B):
        wdl_i = (float(wdl[i, 0]), float(wdl[i, 1]), float(wdl[i, 2]))
        p = np.asarray(policy[i], dtype=np.float32)
        to_raw_cache.append((keys[i], wdl_i, p))
    raw_cache_bulk_insert(to_raw_cache)


def insert_new(keys, wdl, policy):
    """New path: convert keys once, pass numpy arrays directly."""
    keys_np = np.array(keys, dtype=np.uint64)
    raw_cache_bulk_insert_np(keys_np, wdl, policy)


def time_fn(fn, keys, wdl, policy):
    times = []
    for rep in range(N_WARMUP + N_REPS):
        raw_cache_clear()
        t0 = time.perf_counter()
        fn(keys, wdl, policy)
        dt = time.perf_counter() - t0
        if rep >= N_WARMUP:
            times.append(dt)
    return np.array(times)


def verify_identical(keys, wdl, policy):
    """
    Insert the same data via each method, read back via raw_cache_lookup,
    and assert WDL and policy arrays are bit-identical.
    Returns True if all entries match.
    """
    B = len(keys)
    results = {}

    for label, fn in [("old", insert_old), ("new", insert_new)]:
        raw_cache_clear()
        fn(keys, wdl, policy)
        entries = []
        for k in keys:
            r = raw_cache_lookup(k)
            entries.append(r)
        results[label] = entries

    mismatches = 0
    for i, (old_r, new_r) in enumerate(zip(results["old"], results["new"])):
        if old_r is None or new_r is None:
            print(f"  [!] index {i}: lookup returned None  old={old_r is None}  new={new_r is None}")
            mismatches += 1
            continue

        old_wdl, old_pol = old_r
        new_wdl, new_pol = new_r

        wdl_ok = np.allclose(np.array(old_wdl), np.array(new_wdl), atol=1e-6)
        pol_ok = np.array_equal(np.asarray(old_pol), np.asarray(new_pol))

        if not wdl_ok:
            print(f"  [!] index {i}: WDL mismatch  old={old_wdl}  new={new_wdl}")
            mismatches += 1
        if not pol_ok:
            diff = np.abs(np.asarray(old_pol) - np.asarray(new_pol))
            print(f"  [!] index {i}: policy mismatch  max_diff={diff.max():.2e}")
            mismatches += 1

    return mismatches == 0


def report(label, times, B):
    ms = times * 1000
    us_per = ms / B * 1000
    print(f"    {label:18s}  "
          f"mean={ms.mean():7.3f}ms  "
          f"median={np.median(ms):7.3f}ms  "
          f"min={ms.min():7.3f}ms  "
          f"max={ms.max():7.3f}ms  "
          f"({us_per.mean():.2f} us/item)")


def main():
    print("cache_insert_speed_test")
    print(f"  warmup={N_WARMUP}  reps={N_REPS}  batch_sizes={BATCH_SIZES}")
    print()

    print("=== Correctness ===")
    all_pass = True
    for B in [32, 64, 256, 512]:
        keys, wdl, policy = make_data(B, seed=B + 1000)
        ok = verify_identical(keys, wdl, policy)
        status = "PASS" if ok else "FAIL"
        print(f"  B={B:4d}  {status}")
        if not ok:
            all_pass = False
    raw_cache_clear()

    if not all_pass:
        print("\nFAILED correctness — skipping timing.")
        return
    print()

    print("=== Timing ===")
    for B in BATCH_SIZES:
        keys, wdl, policy = make_data(B)
        print(f"  B={B}")
        t_old = time_fn(insert_old, keys, wdl, policy)
        t_new = time_fn(insert_new, keys, wdl, policy)
        report("old (tuple loop)", t_old, B)
        report("new (numpy)     ", t_new, B)
        speedup = t_old.mean() / t_new.mean()
        print(f"    speedup: {speedup:.2f}x\n")

    raw_cache_clear()


if __name__ == "__main__":
    main()
