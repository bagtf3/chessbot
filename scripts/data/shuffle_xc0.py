"""shuffle_xc0.py — Read sharded tfrecord.gz files through a Fisher-Yates
shuffle buffer and write re-randomized tfrecord.gz files of RECORDS_PER_FILE
positions each. One pass through all input — stops when exhausted.

With --workers N, input files are split into N random chunks processed in
parallel. Each worker writes its own shards named xc0_shuf_w{id}_{shard:05d}.tfrecord.gz
so there are no write conflicts. All logging is aggregated by the main process
and printed as combined totals.

Usage:
    python scripts/shuffle_xc0.py [--in-dir PATH] [--out-dir PATH] [--workers N]
"""
import argparse
import gzip
import multiprocessing as mp
import os
import queue
import random
import struct
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


BUFFER_SIZE      = 384_000
RECORDS_PER_FILE = 10_240

OPTIONS = tf.io.TFRecordOptions(compression_type="GZIP")


def iter_records(file_list):
    for path in file_list:
        with gzip.open(path, "rb") as f:
            while True:
                hdr = f.read(12)
                if len(hdr) < 12:
                    break
                length = struct.unpack("<Q", hdr[:8])[0]
                data = f.read(length)
                if len(data) < length:
                    break
                f.read(4)
                yield data


def writer_thread(write_q, out_dir, worker_id, log_q):
    total_out = 0
    total_in  = 0
    while True:
        item = write_q.get()
        if item is None:
            break
        shard_id, records, total_in = item
        path = os.path.join(out_dir, f"xc0_shuf_w{worker_id}_{shard_id:05d}.tfrecord.gz")
        with tf.io.TFRecordWriter(path, OPTIONS) as w:
            for rec in records:
                w.write(rec)
        total_out += len(records)
        log_q.put({"kind": "shard", "worker_id": worker_id,
                   "total_in": total_in, "total_out": total_out})
        write_q.task_done()


def run_worker(worker_id, in_files, out_dir, buffer_size, records_per_file, log_q):
    prefix = f"xc0_shuf_w{worker_id}_"
    existing = [f for f in os.listdir(out_dir)
                if f.startswith(prefix) and f.endswith(".tfrecord.gz")]
    shard_start = 0
    if existing:
        nums = []
        for name in existing:
            stem = name.replace(".tfrecord.gz", "")
            try:
                nums.append(int(stem.split("_")[-1]))
            except ValueError:
                pass
        if nums:
            shard_start = max(nums) + 1

    random.shuffle(in_files)

    write_q = queue.Queue(maxsize=3)
    writer  = threading.Thread(
        target=writer_thread,
        args=(write_q, out_dir, worker_id, log_q),
        daemon=True,
    )
    writer.start()

    buffer       = []
    output_batch = []
    shard_id     = shard_start
    total_in     = 0

    def flush():
        nonlocal output_batch, shard_id
        while len(output_batch) >= records_per_file:
            chunk        = output_batch[:records_per_file]
            output_batch = output_batch[records_per_file:]
            write_q.put((shard_id, chunk, total_in))
            shard_id += 1

    for raw in iter_records(in_files):
        total_in += 1
        if len(buffer) < buffer_size:
            buffer.append(raw)
            if total_in % 100_000 == 0:
                log_q.put({"kind": "fill", "worker_id": worker_id,
                           "buffered": len(buffer), "total_in": total_in})
        else:
            idx = random.randrange(buffer_size)
            output_batch.append(buffer[idx])
            buffer[idx] = raw
            flush()

    log_q.put({"kind": "drain", "worker_id": worker_id,
               "total_in": total_in, "buffer_len": len(buffer)})
    random.shuffle(buffer)
    write_q.maxsize = 0
    i = 0
    while i + records_per_file <= len(buffer):
        write_q.put((shard_id, buffer[i:i + records_per_file], total_in))
        shard_id += 1
        i += records_per_file
    remainder = output_batch + buffer[i:]
    if remainder:
        write_q.put((shard_id, remainder, total_in))
        shard_id += 1

    write_q.put(None)
    writer.join()

    log_q.put({"kind": "done", "worker_id": worker_id,
               "shards": shard_id - shard_start, "total_in": total_in})
    log_q.put(None)
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


def partition(items, n):
    return [items[i::n] for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir",  required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    in_files = sorted(
        os.path.join(args.in_dir, f)
        for f in os.listdir(args.in_dir)
        if f.endswith(".tfrecord.gz")
    )
    if not in_files:
        parser.error(f"no .tfrecord.gz files found in {args.in_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    random.shuffle(in_files)

    print(f"[shuffle] input files  : {len(in_files):,}")
    print(f"[shuffle] workers      : {args.workers}")
    print(f"[shuffle] buffer/worker: {BUFFER_SIZE:,}")
    print(f"[shuffle] records/file : {RECORDS_PER_FILE:,}")
    print(f"[shuffle] output dir   : {args.out_dir}")

    chunks = partition(in_files, args.workers)
    ctx    = mp.get_context("spawn")
    log_q  = ctx.Queue()

    procs = []
    for i, chunk in enumerate(chunks):
        p = ctx.Process(target=run_worker,
                        args=(i, chunk, args.out_dir, BUFFER_SIZE, RECORDS_PER_FILE, log_q))
        p.start()
        procs.append(p)

    # per-worker state for aggregation
    w_in     = [0] * args.workers  # total records read
    w_out    = [0] * args.workers  # total records written
    w_buf    = [0] * args.workers  # current buffer fill
    w_drain  = [False] * args.workers
    begin    = time.time()
    done     = 0

    while done < args.workers:
        try:
            msg = log_q.get(timeout=2.0)
        except Exception:
            for p in procs:
                if not p.is_alive():
                    done += 1
            continue

        if msg is None:
            done += 1
            continue

        kind = msg["kind"]
        wid  = msg["worker_id"]

        if kind == "fill":
            w_in[wid]  = msg["total_in"]
            w_buf[wid] = msg["buffered"]
            total_in  = sum(w_in)
            total_buf = sum(w_buf)
            elapsed   = time.time() - begin
            print(f"[fill]  read={total_in:,}  buffered={total_buf:,}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

        elif kind == "shard":
            w_in[wid]  = msg["total_in"]
            w_out[wid] = msg["total_out"]
            total_in  = sum(w_in)
            total_out = sum(w_out)
            elapsed   = time.time() - begin
            rate      = total_out / max(elapsed, 1e-9)
            print(f"[shard]  read={total_in:,}  written={total_out:,}  "
                  f"rate={rate:,.0f} rec/s  elapsed={elapsed:.0f}s", flush=True)

        elif kind == "drain":
            w_in[wid]   = msg["total_in"]
            w_drain[wid] = True
            total_in    = sum(w_in)
            n_draining  = sum(w_drain)
            print(f"[drain]  {n_draining}/{args.workers} workers draining  "
                  f"total_read={total_in:,}", flush=True)

        elif kind == "done":
            w_in[wid]  = msg["total_in"]
            total_in   = sum(w_in)
            total_out  = sum(w_out)
            elapsed    = time.time() - begin
            print(f"[worker done]  workers_finished={done+1}/{args.workers}  "
                  f"read={total_in:,}  written={total_out:,}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    elapsed   = time.time() - begin
    total_out = sum(w_out)
    total_in  = sum(w_in)
    print(f"\n[done] elapsed={elapsed:.0f}s  total_read={total_in:,}  "
          f"total_written={total_out:,}  rate={total_out/max(elapsed,1e-9):,.0f} rec/s",
          flush=True)

    for p in procs:
        p.join(timeout=10)


if __name__ == "__main__":
    main()
