"""shuffle_xc0.py — Read sharded tfrecord.gz files through a Fisher-Yates
shuffle buffer and write re-randomized tfrecord.gz files of RECORDS_PER_FILE
positions each. One pass through all input — stops when exhausted.

Writing happens in a background thread so the main loop keeps reading/swapping
without blocking on gzip compression.

Multiple passes can be chained:
  pass 1: --in-dir bootstrap_tfrecords  --out-dir xc0/pass1
  pass 2: --in-dir xc0/pass1           --out-dir xc0/pass2
  ...

Usage:
    python scripts/shuffle_xc0.py [--in-dir PATH] [--out-dir PATH]
"""
import argparse
import gzip
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
DEFAULT_OUT_DIR  = r"C:\Users\Bryan\Data\chessbot_data\training_data\wdl_30m_records_062126"
DEFAULT_IN_DIR   = os.getenv("BOOTSTRAP_TFREC_DIR", "")

OPTIONS = tf.io.TFRecordOptions(compression_type="GZIP")


def iter_records(file_list):
    """Yield raw serialized Example bytes from all files."""
    for path in file_list:
        with gzip.open(path, "rb") as f:
            while True:
                hdr = f.read(12)  # uint64 length + uint32 crc
                if len(hdr) < 12:
                    break
                length = struct.unpack("<Q", hdr[:8])[0]
                data = f.read(length)
                if len(data) < length:
                    break
                f.read(4)  # skip data crc
                yield data


def writer_thread(write_q, out_dir, begin):
    """Background thread: pull (shard_id, records, total_in) from queue and write."""
    total_out = 0
    while True:
        item = write_q.get()
        if item is None:
            break
        shard_id, records, total_in = item
        path = os.path.join(out_dir, f"xc0_{shard_id:05d}.tfrecord.gz")
        with tf.io.TFRecordWriter(path, OPTIONS) as w:
            for rec in records:
                w.write(rec)
        total_out += len(records)
        elapsed = time.time() - begin
        print(
            f"[shard {shard_id:05d}]  "
            f"total_in={total_in:,}  total_out={total_out:,}  "
            f"elapsed={elapsed:.0f}s  "
            f"rate={total_out / max(elapsed, 1e-9):,.0f} rec/s"
        )
        write_q.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir",  default=DEFAULT_IN_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.in_dir:
        parser.error("--in-dir is required (or set $BOOTSTRAP_TFREC_DIR)")

    in_files = sorted(
        os.path.join(args.in_dir, f)
        for f in os.listdir(args.in_dir)
        if f.endswith(".tfrecord.gz")
    )
    if not in_files:
        parser.error(f"no .tfrecord.gz files found in {args.in_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    random.shuffle(in_files)

    existing = [
        f for f in os.listdir(args.out_dir)
        if f.endswith(".tfrecord.gz")
    ]
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

    print(f"[shuffle] input files  : {len(in_files):,}")
    print(f"[shuffle] buffer_size  : {BUFFER_SIZE:,}")
    print(f"[shuffle] records/file : {RECORDS_PER_FILE:,}")
    print(f"[shuffle] output dir   : {args.out_dir}")
    print(f"[shuffle] shard_start  : {shard_start:,}")

    begin     = time.time()
    write_q   = queue.Queue(maxsize=3)  # cap in-flight shards to ~3
    writer    = threading.Thread(
        target=writer_thread, args=(write_q, args.out_dir, begin), daemon=True
    )
    writer.start()

    buffer       = []
    output_batch = []
    shard_id     = shard_start
    total_in     = 0

    def flush():
        nonlocal output_batch, shard_id
        while len(output_batch) >= RECORDS_PER_FILE:
            chunk        = output_batch[:RECORDS_PER_FILE]
            output_batch = output_batch[RECORDS_PER_FILE:]
            write_q.put((shard_id, chunk, total_in))
            shard_id += 1

    for raw in iter_records(in_files):
        total_in += 1
        if len(buffer) < BUFFER_SIZE:
            buffer.append(raw)
            if total_in % 100_000 == 0:
                elapsed = time.time() - begin
                print(
                    f"[fill]  buffered={len(buffer):,}/{BUFFER_SIZE:,}"
                    f"  elapsed={elapsed:.0f}s"
                )
        else:
            idx = random.randrange(BUFFER_SIZE)
            output_batch.append(buffer[idx])
            buffer[idx] = raw
            flush()

    print(
        f"[shuffle] input exhausted ({total_in:,} records),"
        f" draining buffer ({len(buffer):,})..."
    )
    random.shuffle(buffer)
    write_q.maxsize = 0  # unbounded for drain — let writer run uncontested
    i = 0
    while i + RECORDS_PER_FILE <= len(buffer):
        write_q.put((shard_id, buffer[i:i + RECORDS_PER_FILE], total_in))
        shard_id += 1
        i += RECORDS_PER_FILE
    remainder = output_batch + buffer[i:]
    if remainder:
        write_q.put((shard_id, remainder, total_in))
        shard_id += 1

    write_q.put(None)
    writer.join()

    elapsed = time.time() - begin
    print(f"\n[done] shards={shard_id:,}  total_in={total_in:,}  elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
