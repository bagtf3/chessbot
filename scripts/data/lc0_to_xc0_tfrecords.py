"""lc0_to_xc0_tfrecords.py -- Convert Lc0 binary training data to xc0-format
.tfrecord.gz shards.

N worker processes each run ChunkParser on a chunk subset (workers=0 mode),
convert to Xerces format (clip+renorm policy, no uniform blend, draw_drop=0.25),
and push serialized records into a shared queue. Main process Fisher-Yates
shuffles and writes 10240-record shards to --out-dir.

Usage:
    python scripts/lc0_to_xc0_tfrecords.py [--lc0-dir PATH] [--out-dir PATH] [--workers N]
"""
import argparse
import multiprocessing as mp
import os
import queue
import random
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from xerces_training.chunkparser import ChunkParser

RECORDS_PER_FILE = 10_240
BUFFER_SIZE      = 300_000
POLICY_MAX_CLIP  = 0.6
CHUNK_BATCH_SIZE = 512
DRAW_DROP_RATE   = 0.75
Z_BLEND          = 0.5

DEFAULT_LC0_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\lc0"
DEFAULT_OUT_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\staging"

OPTIONS = tf.io.TFRecordOptions(compression_type="GZIP")


def scan_chunks(lc0_dir):
    chunks = []
    run_dirs = sorted(
        d for d in os.listdir(lc0_dir)
        if os.path.isdir(os.path.join(lc0_dir, d))
        and d != "040126"
    )
    for d in run_dirs:
        full = os.path.join(lc0_dir, d)
        gz_files = sorted(f for f in os.listdir(full) if f.endswith(".gz") and not f.endswith(".tfrecord.gz"))
        print(f"[scan]  {d}: {len(gz_files):,} files")
        chunks.extend(os.path.join(full, f) for f in gz_files)
    print(f"[scan]  total: {len(chunks):,} chunk files")
    return chunks


def make_example(enc_in, mask, policy, value: np.ndarray):
    enc_bytes = tf.io.serialize_tensor(tf.constant(enc_in.astype(np.int16))).numpy()
    mask_bytes = tf.io.serialize_tensor(tf.constant(mask.astype(np.int32))).numpy()
    pol_bytes  = tf.io.serialize_tensor(tf.constant(policy.astype(np.float32))).numpy()
    ex = tf.train.Example(features=tf.train.Features(feature={
        "enc_in":        tf.train.Feature(bytes_list=tf.train.BytesList(value=[enc_bytes])),
        "mask":          tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_bytes])),
        "policy_logits": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pol_bytes])),
        "value_out":     tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist())),
        "weight":        tf.train.Feature(float_list=tf.train.FloatList(value=[1.0])),
    }))
    return ex.SerializeToString()


def conversion_worker(chunk_subset, out_queue, worker_id):
    """Spawned process: parse chunk subset, convert, push serialized records to queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    random.seed(worker_id * 31337 + os.getpid())

    parser = ChunkParser(
        chunk_subset,
        shuffle_size=1,
        sample=1,
        batch_size=CHUNK_BATCH_SIZE,
        diff_focus_min=1.0,
        diff_focus_slope=0.0,
        diff_focus_q_weight=0.0,
        diff_focus_pol_scale=1.0,
        draw_drop_rate=DRAW_DROP_RATE,
        workers=0,
    )

    n_sent = 0
    for tokens_b, masks_b, policies_b, result_wdl_b, search_wdl_b in parser.sequential():
        clipped = np.minimum(policies_b, POLICY_MAX_CLIP)
        sums = clipped.sum(axis=1, keepdims=True)
        sums = np.where(sums > 0, sums, 1.0)
        clipped /= sums

        wdl_targets = Z_BLEND * result_wdl_b + (1 - Z_BLEND) * search_wdl_b
        records = [
            make_example(tokens_b[i], masks_b[i], clipped[i], wdl_targets[i])
            for i in range(len(result_wdl_b))
        ]
        out_queue.put(records)
        n_sent += len(records)

    out_queue.put(None)  # sentinel: this worker is done
    print(f"[worker {worker_id}] done, sent {n_sent:,} records", flush=True)


def writer_thread(write_q, out_dir, begin):
    total_out = 0
    while True:
        item = write_q.get()
        if item is None:
            break
        shard_id, records, total_in = item
        path = os.path.join(out_dir, f"lc0_{shard_id:05d}.tfrecord.gz")
        with tf.io.TFRecordWriter(path, OPTIONS) as w:
            for rec in records:
                w.write(rec)
        total_out += len(records)
        elapsed = time.time() - begin
        print(
            f"[shard {shard_id:05d}]  "
            f"in={total_in:,}  out={total_out:,}  "
            f"elapsed={elapsed:.0f}s  "
            f"rate={total_in / max(elapsed, 1e-9):,.0f} rec/s"
        )
        write_q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc0-dir",    default=DEFAULT_LC0_DIR)
    ap.add_argument("--out-dir",    default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers",    type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--target-pos", type=int, default=0,
                    help="stop after this many records (0 = no limit)")
    args = ap.parse_args()

    chunks = scan_chunks(args.lc0_dir)
    if not chunks:
        raise SystemExit("[error] no chunk files found")

    os.makedirs(args.out_dir, exist_ok=True)
    random.shuffle(chunks)

    n_workers     = args.workers
    chunk_subsets = [chunks[i::n_workers] for i in range(n_workers)]

    target_pos = args.target_pos or float("inf")

    print(f"[config] workers={n_workers}  buffer={BUFFER_SIZE:,}  records_per_file={RECORDS_PER_FILE:,}")
    print(f"[config] policy_max_clip={POLICY_MAX_CLIP}  draw_drop={DRAW_DROP_RATE}")
    print(f"[config] target_pos={args.target_pos or 'unlimited'}  out_dir={args.out_dir}")
    for i, sub in enumerate(chunk_subsets):
        print(f"[config] worker {i}: {len(sub):,} chunks")

    ctx       = mp.get_context("spawn")
    out_queue = ctx.Queue(maxsize=n_workers * 8)

    processes = []
    for wid, subset in enumerate(chunk_subsets):
        p = ctx.Process(target=conversion_worker, args=(subset, out_queue, wid), daemon=False)
        p.start()
        processes.append(p)
    print(f"[main] spawned {n_workers} worker processes")

    begin   = time.time()
    write_q = queue.Queue(maxsize=3)
    wt      = threading.Thread(target=writer_thread, args=(write_q, args.out_dir, begin), daemon=True)
    wt.start()

    buffer       = []
    output_batch = []
    shard_id     = 0
    total_in     = 0
    active       = n_workers

    def flush():
        nonlocal output_batch, shard_id
        while len(output_batch) >= RECORDS_PER_FILE:
            chunk        = output_batch[:RECORDS_PER_FILE]
            output_batch = output_batch[RECORDS_PER_FILE:]
            write_q.put((shard_id, chunk, total_in))
            shard_id += 1

    while active > 0:
        try:
            item = out_queue.get(timeout=60)
        except queue.Empty:
            alive = sum(p.is_alive() for p in processes)
            print(f"[main] queue timeout, workers alive={alive}/{n_workers}")
            if alive == 0:
                break
            continue

        if item is None:
            active -= 1
            print(f"[main] worker done, {active} remaining")
            continue

        hit_target = False
        for rec in item:
            total_in += 1
            if len(buffer) < BUFFER_SIZE:
                buffer.append(rec)
                if total_in % 100_000 == 0:
                    elapsed = time.time() - begin
                    print(
                        f"[fill]  buffered={len(buffer):,}/{BUFFER_SIZE:,}"
                        f"  total={total_in:,}  elapsed={elapsed:.0f}s"
                        f"  rate={total_in / max(elapsed, 1e-9):,.0f} rec/s"
                    )
            else:
                idx = random.randrange(BUFFER_SIZE)
                output_batch.append(buffer[idx])
                buffer[idx] = rec
                flush()
            if total_in >= target_pos:
                hit_target = True
                break
        if hit_target:
            print(f"[main] target_pos={args.target_pos:,} reached, stopping")
            for p in processes:
                p.terminate()
            break

    print(f"[main] all workers done ({total_in:,} records), draining buffer ({len(buffer):,})...")
    random.shuffle(buffer)
    output_batch.extend(buffer)
    buffer = []
    flush()

    if output_batch:
        write_q.put((shard_id, output_batch, total_in))
        shard_id += 1

    write_q.put(None)
    wt.join()

    for p in processes:
        p.join(timeout=5)

    elapsed = time.time() - begin
    total_M = total_in / 1_000_000
    print()
    print(f"[done]  shards={shard_id:,}  total={total_in:,} ({total_M:.1f}M)  elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
