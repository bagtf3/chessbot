"""Convert on-disk Lc0 V6 training games to per-game sparse pkl.gz.

Each source .gz (one lc0 self-play game) becomes one .pkl.gz with an xc0hK6
board (393 tokens) per ply, sparse policy, and a 50/50 result+search WDL target.
No CPL, no Stockfish -- these are lc0's own high-quality targets, so cpl=0 on
every ply. Output mirrors the source batch folders under lc0_pkl/, each carrying
a copy of the source LICENSE plus a conversion note.

Usage:
    python scripts/data/migrate_lc0_games.py [--batch FOLDER|all]
        [--max-workers 6] [--z-blend 0.5]
"""
import argparse
import gzip
import multiprocessing as mp
import os
import pickle
import queue
import struct
import time

import numpy as np
import pyfastchess as pf

from xerces_training.chunkparser import V6_STRUCT_STRING
from xerces_training.parse_utils import v6_planes_to_xc0h, get_policy_vector

LC0_SRC_ROOT = r"C:\Users\Bryan\Data\chessbot_data\training_data\lc0"
OUT_ROOT     = r"C:\Users\Bryan\Data\chessbot_data\training_data\lc0_pkl"

SL_IDX  = np.nonzero(pf.build_sometimes_legal_mask().astype(bool))[0]
V6      = struct.Struct(V6_STRUCT_STRING)
DUMMY64 = np.zeros(64, dtype=np.int64)
K       = 6
INT16_MAX = 32767

LICENSE_ADDENDUM = """
--- Xerces format-conversion note ---
The .pkl.gz files in this folder are a 1:1 format conversion of the adjacent
Leela Chess Zero V6 training chunks (training.*.gz). No data was added or
filtered: each game's positions, policy (MCTS visit distribution), and game
result were decoded into xc0h board tokens, sparse policy, and 50/50
result+search WDL targets for use by the Xerces chess engine. The underlying
data remains under ODbL 1.0 / DBCL 1.0 as stated above.
Attribution: Leela Chess Zero (lczero.org).
"""


def wdl_from_qd(q, d):
    """STM-POV WDL from lc0 (q, d) pair. lc0 q/d are already side-to-move POV."""
    return np.array([0.5 * (1 - d + q), d, 0.5 * (1 - d - q)], dtype=np.float32)


def dense_to_policy_list(policy_1858):
    nz = np.nonzero(policy_1858)[0]
    return [(int(i), np.float16(policy_1858[i])) for i in nz]


def raw_visits_sparse(pol_1858_prenorm, visits):
    raw = np.round(pol_1858_prenorm * float(visits)).astype(np.int32)
    nz = np.nonzero(raw)[0]
    counts = np.clip(raw[nz], 0, INT16_MAX).astype(np.int16)
    return (nz.astype(np.int16), counts)


def decode_game(path, z_blend):
    with gzip.open(path, 'rb') as f:
        raw = f.read()
    n = len(raw) // V6.size
    plies = []
    result_white = None
    for i in range(n):
        t = V6.unpack(raw[i * V6.size:(i + 1) * V6.size])
        probs_bytes, planes_bytes = t[2], t[3]
        us_ooo, us_oo, them_ooo, them_oo, stm, rule50 = t[4], t[5], t[6], t[7], t[8], t[9]
        best_q, best_d = t[13], t[15]
        result_q, result_d = t[19], t[20]
        visits = t[27]

        xc0h = v6_planes_to_xc0h(
            planes_bytes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50, K,
        )

        probs = np.frombuffer(probs_bytes, dtype=np.float32)
        pol4288, _ = get_policy_vector(probs, stm, us_ooo, us_oo, DUMMY64)
        pol_1858 = pol4288[SL_IDX].astype(np.float32)
        s = pol_1858.sum()
        if s > 0:
            policy = dense_to_policy_list(pol_1858 / s)
            raw_visits = raw_visits_sparse(pol_1858, visits)
        else:
            policy = []
            raw_visits = (np.array([], dtype=np.int16), np.array([], dtype=np.int16))

        best_wdl = wdl_from_qd(best_q, best_d)
        z_wdl    = wdl_from_qd(result_q, result_d)
        wdl_target = (z_blend * z_wdl + (1.0 - z_blend) * best_wdl).astype(np.float32)

        if result_white is None:
            wp = result_q if stm == 0 else -result_q
            result_white = int(round(wp))

        plies.append({
            'xc0h':       np.asarray(xc0h, dtype=np.int16),
            'policy':     policy,
            'raw_visits': raw_visits,
            'best_wdl':   best_wdl,
            'z_wdl':      z_wdl,
            'wdl_target': wdl_target,
            'cpl':        0,
        })
    return plies, result_white


def stem_of(path):
    b = os.path.basename(path)
    return b[:-3] if b.endswith('.gz') else b


def convert_one(src_path, out_dir, batch, z_blend):
    stem = stem_of(src_path)
    out_path = os.path.join(out_dir, stem + '.pkl.gz')
    if os.path.exists(out_path):
        return 'skip', 0

    plies, result_white = decode_game(src_path, z_blend)
    if not plies:
        return 'empty', 0

    record = {
        'header': {
            'game_id': f'{batch}__{stem}',
            'source':  'lc0',
            'k':       K,
            'result':  result_white,
            'n_plies': len(plies),
        },
        'plies': plies,
    }
    tmp = out_path + '.tmp'
    with gzip.open(tmp, 'wb', compresslevel=6) as f:
        pickle.dump(record, f, protocol=4)
    os.replace(tmp, out_path)
    return 'ok', len(plies)


def worker(worker_id, files, out_dir, batch, z_blend, stats_q):
    n_ok = n_skip = n_ply = 0
    for i, src in enumerate(files):
        try:
            status, np_plies = convert_one(src, out_dir, batch, z_blend)
        except Exception as e:
            print(f"[w{worker_id}] skip {os.path.basename(src)}: {e}", flush=True)
            continue
        if status == 'ok':
            n_ok  += 1
            n_ply += np_plies
        elif status == 'skip':
            n_skip += 1
        if (i + 1) % 500 == 0:
            stats_q.put({'wid': worker_id, 'n_ok': n_ok, 'n_skip': n_skip,
                         'n_ply': n_ply, 'i': i + 1, 'total': len(files)})
    stats_q.put({'wid': worker_id, 'n_ok': n_ok, 'n_skip': n_skip,
                 'n_ply': n_ply, 'i': len(files), 'total': len(files), 'done': True})
    os._exit(0)


def setup_license(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_lic = os.path.join(out_dir, 'LICENSE')
    if os.path.exists(out_lic):
        return
    src_lic = os.path.join(src_dir, 'LICENSE')
    text = ''
    if os.path.exists(src_lic):
        with open(src_lic, encoding='utf-8') as f:
            text = f.read()
    with open(out_lic, 'w', encoding='utf-8') as f:
        f.write(text.rstrip() + '\n' + LICENSE_ADDENDUM.strip() + '\n')


def partition(items, n):
    return [items[i::n] for i in range(n)]


def process_batch(batch, max_workers, z_blend):
    src_dir = os.path.join(LC0_SRC_ROOT, batch)
    out_dir = os.path.join(OUT_ROOT, batch)
    setup_license(src_dir, out_dir)

    files = sorted(
        os.path.join(src_dir, f) for f in os.listdir(src_dir)
        if f.endswith('.gz')
    )
    print(f"\n=== {batch}: {len(files):,} source games -> {out_dir} ===", flush=True)
    if not files:
        return

    parts = [p for p in partition(files, max_workers) if p]
    stats_q = mp.Queue()
    procs = []
    for wid, part in enumerate(parts):
        p = mp.Process(target=worker,
                       args=(wid, part, out_dir, batch, z_blend, stats_q))
        p.start()
        procs.append(p)

    per = {}
    t0 = time.time()
    last_print = 0.0

    def drain():
        while True:
            try:
                s = stats_q.get_nowait()
            except queue.Empty:
                break
            per[s['wid']] = s

    def report(tag):
        tot_ok   = sum(v['n_ok']   for v in per.values())
        tot_skip = sum(v['n_skip'] for v in per.values())
        tot_ply  = sum(v['n_ply']  for v in per.values())
        tot_i    = sum(v['i']      for v in per.values())
        el       = max(time.time() - t0, 1e-6)
        print(f"  [{batch}] {tag} {tot_i:,}/{len(files):,}  ok={tot_ok:,}  "
              f"skip={tot_skip:,}  plies={tot_ply:,}  "
              f"{tot_ok/el:.0f} games/s  {tot_ply/el:.0f} plies/s", flush=True)

    # Wait on process liveness, not on a final queue message: workers os._exit
    # right after their last put, which can drop that message and hang a
    # message-counting wait forever.
    while any(p.is_alive() for p in procs):
        drain()
        now = time.time()
        if now - last_print >= 15:
            last_print = now
            report('...')
        time.sleep(0.5)

    drain()
    for p in procs:
        p.join(timeout=5)
    report('DONE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', default='all',
                    help="source batch folder name, or 'all'")
    ap.add_argument('--max-workers', type=int, default=6, dest='max_workers')
    ap.add_argument('--z-blend', type=float, default=0.5, dest='z_blend',
                    help="weight on game-result WDL vs search WDL (0.5 = 50/50)")
    args = ap.parse_args()

    if args.batch == 'all':
        batches = sorted(
            d for d in os.listdir(LC0_SRC_ROOT)
            if os.path.isdir(os.path.join(LC0_SRC_ROOT, d))
        )
    else:
        batches = [args.batch]

    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"[main] {len(batches)} batch folders  workers={args.max_workers}  "
          f"z_blend={args.z_blend}", flush=True)
    for batch in batches:
        process_batch(batch, args.max_workers, args.z_blend)
    print("\n[main] all batches done", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
