"""lc0_to_xc0_tfrecords.py -- Convert lc0 V6 binary training data to xc0h-K6
.tfrecord.gz shards.

Decodes directly from V6 packed bit planes to xc0h tokens (no float32
intermediate). N worker processes each drain a disjoint chunk subset and push
serialised records into a shared queue. Main process Fisher-Yates shuffles and
writes 10240-record shards to --out-dir.

Usage:
    python scripts/data/lc0_to_xc0_tfrecords.py [--lc0-dir PATH] [--out-dir PATH]
"""
import argparse
import multiprocessing as mp
import os
import queue
import random
import struct
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pyfastchess as pf

from xerces_training.chunkparser import ChunkParser, V6_STRUCT_STRING
from xerces_training.parse_utils import (
    v6_planes_to_xc0h, get_policy_vector, XC0_POLICY_SIZE,
)

RECORDS_PER_FILE = 10_240
BUFFER_SIZE      = 300_000
POLICY_MAX_CLIP  = 0.6
DRAW_DROP_RATE   = 0.75
Z_BLEND          = 0.5

DEFAULT_LC0_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\lc0"
DEFAULT_OUT_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\staging"

OPTIONS  = tf.io.TFRecordOptions(compression_type="GZIP")
V6_STRUC = struct.Struct(V6_STRUCT_STRING)

SL_IDX = np.nonzero(pf.build_sometimes_legal_mask().astype(bool))[0]  # (1858,)
N_1858  = len(SL_IDX)


def scan_chunks(lc0_dir):
    chunks = []
    for d in sorted(os.listdir(lc0_dir)):
        if d == "040126":
            continue
        full = os.path.join(lc0_dir, d)
        if not os.path.isdir(full):
            continue
        gz = sorted(f for f in os.listdir(full)
                    if f.endswith(".gz") and not f.endswith(".tfrecord.gz"))
        print(f"[scan]  {d}: {len(gz):,} files")
        chunks.extend(os.path.join(full, f) for f in gz)
    print(f"[scan]  total: {len(chunks):,} chunk files")
    return chunks


def bytes_feature(b):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))


def make_example(fmt, xc0h_board, lc0_board, policy, legal, wdl):
    feat = {
        "policy": bytes_feature(policy.astype(np.float32).tobytes()),
        "legal":  bytes_feature(legal.astype(np.uint8).tobytes()),
        "wdl":    tf.train.Feature(float_list=tf.train.FloatList(value=wdl.tolist())),
    }
    if fmt in ("xc0h", "both"):
        feat["xc0h_board"] = bytes_feature(xc0h_board.astype(np.int16).tobytes())
    if fmt in ("lc0", "both"):
        feat["lc0_board"] = bytes_feature(lc0_board.astype(np.float16).tobytes())
    return tf.train.Example(
        features=tf.train.Features(feature=feat)).SerializeToString()


def convert_record(rec, fmt, dummy_tokens):
    """Unpack one V6 record -> (xc0h_board, lc0_board, policy, legal, wdl) or None."""
    (ver, input_format, probs_bytes, planes_bytes,
     us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count,
     invariance_info, dep_result,
     root_q, best_q, root_d, best_d, root_m, best_m, plies_left,
     result_q, result_d, played_q, played_d, played_m,
     orig_q, orig_d, orig_m,
     visits, played_idx, best_idx,
     _r1, _r2, _r3, _r4) = V6_STRUC.unpack(rec)

    probs = np.frombuffer(probs_bytes, dtype=np.float32)
    pol4288, mask4288 = get_policy_vector(probs, stm, us_ooo, us_oo, dummy_tokens)

    policy = pol4288[SL_IDX]
    legal  = mask4288[SL_IDX].astype(np.uint8)

    s = policy.sum()
    if s <= 0:
        return None
    clipped = np.minimum(policy, POLICY_MAX_CLIP)
    cs = clipped.sum()
    policy = clipped / cs if cs > 0 else clipped

    result_wdl = np.array([
        0.5 * (1.0 - result_d + result_q),
        result_d,
        0.5 * (1.0 - result_d - result_q),
    ], dtype=np.float32)
    search_wdl = np.array([
        0.5 * (1.0 - best_d + best_q),
        best_d,
        0.5 * (1.0 - best_d - best_q),
    ], dtype=np.float32)
    wdl = Z_BLEND * result_wdl + (1.0 - Z_BLEND) * search_wdl

    xc0h_board = None
    if fmt in ("xc0h", "both"):
        xc0h_board = v6_planes_to_xc0h(planes_bytes, us_ooo, us_oo, them_ooo, them_oo,
                                        stm, rule50_count)
    return xc0h_board, policy, legal, wdl


def conversion_worker(chunk_subset, out_queue, worker_id, fmt):  # noqa: C901
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    random.seed(worker_id * 31337 + os.getpid())

    parser = ChunkParser(
        chunk_subset,
        shuffle_size=1,
        sample=1,
        batch_size=1,
        diff_focus_min=1.0,
        diff_focus_slope=0.0,
        diff_focus_q_weight=0.0,
        diff_focus_pol_scale=1.0,
        draw_drop_rate=DRAW_DROP_RATE,
        workers=0,
    )

    dummy_tokens = np.zeros(64, dtype=np.int64)
    need_lc0 = fmt in ("lc0", "both")
    n_sent = 0
    records = []

    for rec in parser.inner.sequential_gen():
        lc0_board = None
        if need_lc0:
            tup = parser.inner.convert_v6_to_tuple(rec, emit_lc0=True)
            lc0_board = np.asarray(tup[5], dtype=np.float16)  # (112,8,8) already /99
        result = convert_record(rec, fmt, dummy_tokens)
        if result is None:
            continue
        xc0h_board, policy, legal, wdl = result
        records.append(make_example(fmt, xc0h_board, lc0_board, policy, legal, wdl))
        if len(records) >= 512:
            out_queue.put(records)
            n_sent += len(records)
            records = []

    if records:
        out_queue.put(records)
        n_sent += len(records)

    out_queue.put(None)
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
                    help="stop after this many records (0=no limit)")
    ap.add_argument("--format",     default="xc0h", choices=("xc0h", "lc0", "both"),
                    help="board encoding(s) to write per record")
    args = ap.parse_args()

    chunks = scan_chunks(args.lc0_dir)
    if not chunks:
        raise SystemExit("[error] no chunk files found")

    os.makedirs(args.out_dir, exist_ok=True)
    random.shuffle(chunks)

    n_workers     = args.workers
    chunk_subsets = [chunks[i::n_workers] for i in range(n_workers)]
    target_pos    = args.target_pos or float("inf")

    print(f"[config] workers={n_workers}  buffer={BUFFER_SIZE:,}  "
          f"records_per_file={RECORDS_PER_FILE:,}")
    print(f"[config] policy_max_clip={POLICY_MAX_CLIP}  draw_drop={DRAW_DROP_RATE}  "
          f"format={args.format}")
    print(f"[config] target_pos={args.target_pos or 'unlimited'}  out={args.out_dir}")

    ctx       = mp.get_context("spawn")
    out_queue = ctx.Queue(maxsize=n_workers * 8)

    processes = []
    for wid, subset in enumerate(chunk_subsets):
        p = ctx.Process(target=conversion_worker, args=(subset, out_queue, wid, args.format),
                        daemon=False)
        p.start()
        processes.append(p)
    print(f"[main] spawned {n_workers} worker processes")

    begin   = time.time()
    write_q = queue.Queue(maxsize=3)
    wt = threading.Thread(target=writer_thread, args=(write_q, args.out_dir, begin),
                          daemon=True)
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

    print(f"[main] draining buffer ({len(buffer):,})...")
    random.shuffle(buffer)
    output_batch.extend(buffer)
    buffer = []
    flush()

    if output_batch:
        write_q.put((shard_id, output_batch, total_in))

    write_q.put(None)
    wt.join()
    for p in processes:
        p.join(timeout=5)

    elapsed = time.time() - begin
    print(f"\n[done]  shards={shard_id:,}  total={total_in:,} ({total_in/1e6:.1f}M)"
          f"  elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
