"""Build shuffled, pretrain-ready 10240 shards from every data source.

Sources: the 19 run_tag per-game pkls (game_logs_tmp else game_logs), lc0_pkl
per-game pkls, and the lc0_distilled_positions / lc0_BRP shard files. The parent
gathers every file across sources, shuffles the mega-list, and round-robins it to
4 workers (each gets a random cross-section of all sources). Each worker streams
its files -- read, CPL<=30 filter, opening accumulator, per-file shuffle, scatter
across 16 buffers -- and flushes a buffer only once it passes 40960, drawing 10240
at random for hyper-mixing. Reads and writes are async within each worker.

Close-out: each worker drains its buffers to full shards and writes the sub-10240
remainder to a _tail_<wid>.pkl.gz; the parent merges the tails into a final shard.

Usage:
    python scripts/data/build_pretrain_shards.py [--out-dir DIR] [--workers 4]
        [--npz PATH] [--cpl-max 30]
"""
import argparse
import glob
import gzip
import multiprocessing as mp
import os
import pickle
import queue
import random
import threading
import time

import numpy as np

from chessbot import SP_DIR
from chessbot.opening_counts import slot_from_x
from chessbot.replay_buffer import (
    POLICY_DIM, densify_policy, sparsify_policy,
    new_shard_path, write_pkl_gz_shard,
)

TD          = r"C:\Users\Bryan\Data\chessbot_data\training_data"
SHARD_SIZE  = 10240
BUFFER_CAP  = 40960
N_BUFFERS   = 16
CPL_MAX     = 30
SRC         = "pretraining"
DEFAULT_NPZ = os.path.join(SP_DIR, "18m_10c6t_SWA_selfplay6", "opening_counts.npz")

MERGE_BANDS = [(2000, 128), (501, 64), (201, 32), (61, 16), (31, 4), (10, 2)]

RUN_TAGS = [
    "16m_precond_run0", "16m_precond_run1", "16m_precond_run2",
    "16m_xc0hK6_run1", "16m_xc0hK6_run2", "16m_xc0hK6_run3", "16m_xc0hK6_run4",
    "18m_10c6t_d256_pretrain", "18m_10c6t_d256_selfplay0", "18m_10c6t_d256_selfplay1",
    "18m_10c6t_SWA_selfplay0", "18m_10c6t_SWA_selfplay1", "18m_10c6t_SWA_selfplay2",
    "18m_10c6t_SWA_selfplay3", "18m_10c6t_SWA_selfplay4", "18m_10c6t_SWA_selfplay5",
    "18m_10c6t_SWA_selfplay6",
    "18m_6c4t_pretrain_run0", "18m_6c4t_selfplay_run0",
]


class Consolidator:
    """Per-worker opening accumulator: merges frequent opening positions."""

    def __init__(self, npz_path):
        self.merge_map = {}
        if npz_path and os.path.exists(npz_path):
            with np.load(npz_path) as z:
                idx  = z['idx'].astype(np.int64)
                vals = np.floor(z['vals'])
            for slot, cnt in zip(idx, vals):
                for lo, m in MERGE_BANDS:
                    if cnt >= lo:
                        self.merge_map[int(slot)] = m
                        break
        self.buffers = {}
        self.n_merged = 0   # composites emitted (accumulated positions -> buffers)

    def submit(self, rec):
        s = slot_from_x(rec[0])
        m = self.merge_map.get(s)
        if m is None:
            return (rec,)
        buf = self.buffers.setdefault(s, [])
        buf.append(rec)
        if len(buf) >= m:
            comp = self._agg(buf)
            self.buffers[s] = []
            return (comp,)
        return ()

    def flush_all(self):
        out = [self._agg(b) for b in self.buffers.values() if b]
        self.buffers.clear()
        return out

    def _agg(self, recs):
        self.n_merged += 1
        pd = np.zeros(POLICY_DIM, dtype=np.float32)
        for r in recs:
            pd += densify_policy(r[1])
        pd /= len(recs)
        Y = np.mean(np.stack([r[2] for r in recs]), axis=0).astype(np.float32)
        return (recs[0][0], sparsify_policy(pd), Y, SRC)


def policy_to_sparse(policy):
    """Normalize any stored policy to (int16 idx, float32 vals)."""
    if policy is None:
        return (np.array([], dtype=np.int16), np.array([], dtype=np.float32))
    if isinstance(policy, tuple):
        return policy
    if not policy:
        return (np.array([], dtype=np.int16), np.array([], dtype=np.float32))
    idx  = np.array([i for i, _ in policy], dtype=np.int16)
    vals = np.array([v for _, v in policy], dtype=np.float32)
    return (idx, vals)


def extract_unified(path, typ, cpl_max):
    with gzip.open(path, 'rb') as f:
        obj = pickle.load(f)
    if typ == 'shard':
        # incoming 7-tuple (xc0h, None, policy, wdl, 1, 1, src) -> 4-tuple
        return [(r[0], r[2], r[3], SRC) for r in obj]

    out = []
    for ply in obj.get('plies', []):
        cpl = ply.get('cpl')
        if cpl is not None and cpl > cpl_max:
            continue
        x = ply.get('xc0h')
        if x is None or 'wdl_target' not in ply:
            continue
        out.append((
            np.asarray(x, dtype=np.int16),
            policy_to_sparse(ply.get('policy')),
            np.asarray(ply['wdl_target'], dtype=np.float32),
            SRC,
        ))
    return out


def worker(wid, files, out_dir, npz_path, cpl_max, stats_q):
    cons    = Consolidator(npz_path)
    buffers = [[] for _ in range(N_BUFFERS)]
    read_q  = queue.Queue(maxsize=6)
    write_q = queue.Queue(maxsize=6)
    n_written = [0]

    def reader():
        for path, typ in files:
            try:
                recs = extract_unified(path, typ, cpl_max)
            except Exception as e:
                print(f"[w{wid}] read fail {os.path.basename(path)}: {e}", flush=True)
                recs = []
            read_q.put((typ, recs))
        read_q.put(None)

    def writer():
        while True:
            shard = write_q.get()
            if shard is None:
                break
            write_pkl_gz_shard(shard, new_shard_path(out_dir))
            n_written[0] += len(shard)

    rt = threading.Thread(target=reader, daemon=True); rt.start()
    wt = threading.Thread(target=writer, daemon=True); wt.start()

    n_in = n_files = 0
    t0 = time.time()
    last_push = t0

    def push(done=False):
        stats_q.put({'wid': wid, 'n_files': n_files, 'n_in': n_in,
                     'n_written': n_written[0], 'n_merged': cons.n_merged,
                     't': time.time() - t0, 'done': done})

    while True:
        item = read_q.get()
        if item is None:
            break
        typ, recs = item
        n_files += 1
        n_in += len(recs)
        if typ == 'shard':
            pending = list(recs)   # pre-built shards: skip the opening accumulator
        else:
            pending = []
            for r in recs:
                pending.extend(cons.submit(r))
        random.shuffle(pending)
        for r in pending:
            buffers[random.randrange(N_BUFFERS)].append(r)
        for b in buffers:
            if len(b) >= BUFFER_CAP:
                pick = set(random.sample(range(len(b)), SHARD_SIZE))
                shard = [b[i] for i in pick]
                b[:]  = [b[i] for i in range(len(b)) if i not in pick]
                write_q.put(shard)
        now = time.time()
        if n_files % 100 == 0 or now - last_push >= 10:
            last_push = now
            push()

    for out in cons.flush_all():
        buffers[random.randrange(N_BUFFERS)].append(out)
    pool = [r for b in buffers for r in b]
    random.shuffle(pool)
    while len(pool) >= SHARD_SIZE:
        write_q.put(pool[:SHARD_SIZE])
        pool = pool[SHARD_SIZE:]

    write_q.put(None)
    wt.join()

    if pool:
        with gzip.open(os.path.join(out_dir, f'_tail_{wid}.pkl.gz'),
                       'wb', compresslevel=6) as f:
            pickle.dump(pool, f, protocol=4)

    push(done=True)
    stats_q.close()
    stats_q.join_thread()   # flush the final stat before os._exit drops it
    os._exit(0)


def run_tag_dir(tag):
    rd  = os.path.join(SP_DIR, tag)
    tmp = os.path.join(rd, 'game_logs_tmp')
    if os.path.isdir(tmp) and any(f.endswith('.pkl.gz') for f in os.listdir(tmp)):
        return tmp
    return os.path.join(rd, 'game_logs')


def gather_files():
    files = []
    for tag in RUN_TAGS:
        d = run_tag_dir(tag)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.pkl.gz'):
                    files.append((os.path.join(d, f), 'pergame'))

    lc0_pkl = os.path.join(TD, 'lc0_pkl')
    if os.path.isdir(lc0_pkl):
        for sub in os.listdir(lc0_pkl):
            dd = os.path.join(lc0_pkl, sub)
            if os.path.isdir(dd):
                for f in os.listdir(dd):
                    if f.endswith('.pkl.gz'):
                        files.append((os.path.join(dd, f), 'pergame'))

    for name in ('lc0_distilled_positions', 'lc0_BRP'):
        d = os.path.join(TD, name)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.pkl.gz'):
                    files.append((os.path.join(d, f), 'shard'))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(TD, 'pretrain_shards'),
                    dest='out_dir')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--npz', default=DEFAULT_NPZ)
    ap.add_argument('--cpl-max', type=int, default=CPL_MAX, dest='cpl_max')
    ap.add_argument('--limit', type=int, default=0,
                    help="cap total input files (0 = all); for a quick real run")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    print("[main] gathering files across all sources...", flush=True)
    files = gather_files()
    n_shard = sum(1 for f in files if f[1] == 'shard')
    print(f"[main] {len(files):,} files ({n_shard:,} shard-files, "
          f"{len(files)-n_shard:,} per-game)", flush=True)

    random.shuffle(files)
    if args.limit and args.limit < len(files):
        files = files[:args.limit]
        print(f"[main] --limit: using {len(files):,} files", flush=True)
    parts = [files[i::args.workers] for i in range(args.workers)]
    for i, p in enumerate(parts):
        ns = sum(1 for f in p if f[1] == 'shard')
        print(f"  worker {i}: {len(p):,} files ({ns:,} shard)", flush=True)

    stats_q = mp.Queue()
    procs = []
    for wid, part in enumerate(parts):
        pr = mp.Process(target=worker,
                        args=(wid, part, args.out_dir, args.npz, args.cpl_max, stats_q))
        pr.start()
        procs.append(pr)

    per = {}
    n_workers = len(procs)
    t0 = time.time()
    last = 0.0

    def drain():
        while True:
            try:
                s = stats_q.get_nowait()
            except queue.Empty:
                break
            per[s['wid']] = s

    def report():
        el = time.time() - t0
        h, m, s = int(el // 3600), int(el % 3600 // 60), int(el % 60)
        es = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        n_done = sum(1 for v in per.values() if v.get('done'))
        col = "  {:<8} {:>13} {:>13} {:>8} {:>9} {:>8} {:>11}"
        print(f"\n=== Pretrain Shards [{es}] | {n_done}/{n_workers} workers done ===\n",
              flush=True)
        print(col.format("worker", "scanned", "written", "shards", "accum", "up(s)", "rec/s"),
              flush=True)
        tot_scan = tot_wr = tot_sh = tot_mg = 0
        ups = []
        for wid in range(n_workers):
            v = per.get(wid)
            if v is None:
                print(col.format(f"w{wid}", "-", "-", "-", "-", "-", "-"), flush=True)
                continue
            scan, wr, up = v['n_in'], v['n_written'], max(v['t'], 1e-6)
            mg = v.get('n_merged', 0)
            sh = wr // SHARD_SIZE
            tag = f"w{wid}" + ("*" if v.get('done') else "")
            print(col.format(tag, f"{scan:,}", f"{wr:,}", f"{sh:,}",
                             f"{mg:,}", f"{up:.0f}", f"{scan/up:,.0f}"), flush=True)
            tot_scan += scan; tot_wr += wr; tot_sh += sh; tot_mg += mg; ups.append(up)
        el2 = max(el, 1e-6)
        print("  " + "-" * 73, flush=True)
        print(col.format("TOTAL", f"{tot_scan:,}", f"{tot_wr:,}", f"{tot_sh:,}",
                         f"{tot_mg:,}", "", f"{tot_scan/el2:,.0f}"), flush=True)
        n = len(ups) or 1
        print(col.format("AVG", f"{tot_scan//n:,}", f"{tot_wr//n:,}", f"{tot_sh//n:,}",
                         f"{tot_mg//n:,}", f"{sum(ups)/n:.0f}" if ups else "-",
                         f"{tot_scan/el2/n:,.0f}"), flush=True)

    while any(p.is_alive() for p in procs):
        drain()
        now = time.time()
        if now - last >= 30:
            last = now
            report()
        time.sleep(0.5)
    drain()
    report()
    for p in procs:
        p.join(timeout=5)

    # Merge worker tails into final shard(s).
    tails = sorted(glob.glob(os.path.join(args.out_dir, '_tail_*.pkl.gz')))
    pool = []
    for t in tails:
        with gzip.open(t, 'rb') as f:
            pool.extend(pickle.load(f))
        os.remove(t)
    tail_records = len(pool)
    random.shuffle(pool)
    n_final = 0
    while len(pool) >= SHARD_SIZE:
        write_pkl_gz_shard(pool[:SHARD_SIZE], new_shard_path(args.out_dir))
        pool = pool[SHARD_SIZE:]
        n_final += 1
    if pool:
        write_pkl_gz_shard(pool, new_shard_path(args.out_dir))
        n_final += 1

    n_wr = sum(v['n_written'] for v in per.values())
    total_shards = n_wr // SHARD_SIZE + n_final
    total_records = n_wr + tail_records
    el = time.time() - t0
    print(f"\n[done] {total_shards:,} shards, {total_records:,} records "
          f"written in {el:.0f}s -> {args.out_dir}", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
