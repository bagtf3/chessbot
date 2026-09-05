"""Salvage busted lc0 distillation shards.

The busted shards have full 1858-dim policies (softmax over all moves, including
illegal). This script first merges lc0_distilled_positions_batch1 into
lc0_distilled_positions, then runs xc0 inference on each position, uses
CDF(0.999) to identify legal move indices, masks the stored lc0 policy to those
indices, renormalizes, applies prior_clip_max, and rewrites each shard in place
(atomic tmp-then-replace).

Index spaces (XC0 vs LC0) differ only for castling; small error is acceptable
for a salvage pass.

Usage:
    python scripts/data/fix_lc0_shards.py [--model-tag TAG] [--cdf 0.999]
        [--prior-clip-max 0.75] [--workers 4] [--batch 256]
"""
import argparse
import gzip
import os
import pickle
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from chessbot import SP_DIR
from chessbot.train_pytorch import load_pt_model

LC0_DIR       = r'C:\Users\Bryan\Data\chessbot_data\training_data\lc0_distilled_positions'
LC0_BATCH1_DIR = r'C:\Users\Bryan\Data\chessbot_data\training_data\lc0_distilled_positions_batch1'
POLICY_DIM    = 1858


def merge_batch1():
    """Move all .pkl.gz files from batch1 dir into LC0_DIR, renaming on conflict."""
    if not os.path.isdir(LC0_BATCH1_DIR):
        print('batch1 dir not found, skipping merge', flush=True)
        return
    files = [f for f in os.listdir(LC0_BATCH1_DIR) if f.endswith('.pkl.gz')]
    if not files:
        print('batch1 dir empty, skipping merge', flush=True)
        return
    moved = 0
    renamed = 0
    for fname in files:
        src  = os.path.join(LC0_BATCH1_DIR, fname)
        dst  = os.path.join(LC0_DIR, fname)
        if os.path.exists(dst):
            base, ext2 = fname[:-7], '.pkl.gz'
            n = 0
            while os.path.exists(dst):
                n += 1
                dst = os.path.join(LC0_DIR, f'{base}_b1_{n}{ext2}')
            renamed += 1
        shutil.move(src, dst)
        moved += 1
    print(f'merged batch1: {moved} files moved ({renamed} renamed)', flush=True)


def sparse_to_dense(sparse_policy):
    idxs, vals = sparse_policy
    p = np.zeros(POLICY_DIM, dtype=np.float32)
    if len(idxs):
        p[idxs.astype(np.int32)] = vals.astype(np.float32)
    return p


def cdf_legal_masks(logits_np, cdf_cutoff):
    """Return list of bool[1858] arrays marking legal-move indices via CDF cutoff."""
    e = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    masks = []
    for row in p:
        order = np.argsort(-row)
        cdf   = np.cumsum(row[order])
        n     = int(np.searchsorted(cdf, cdf_cutoff)) + 1
        mask  = np.zeros(POLICY_DIM, dtype=bool)
        mask[order[:n]] = True
        masks.append(mask)
    return masks


def filter_policy(dense, mask, prior_clip_max):
    """Mask dense lc0 policy to legal indices, renormalize, apply clip_max."""
    p     = dense * mask
    total = p.sum()
    if total <= 0:
        p     = mask.astype(np.float32)
        total = p.sum()
    p /= total
    k = int(mask.sum())
    if prior_clip_max is not None and k >= 5:
        top = float(p.max())
        if top > prior_clip_max:
            uniform = 1.0 / k
            alpha   = (prior_clip_max - top) / (uniform - top)
            p       = alpha * uniform + (1.0 - alpha) * p
    nz = np.nonzero(mask)[0]
    return (nz.astype(np.int16), p[nz].astype(np.float32))


def process_shard(path, model, device, lock, batch_size, cdf_cutoff, prior_clip_max):
    with gzip.open(path, 'rb') as f:
        records = pickle.load(f)

    xc0h_all  = [np.asarray(r[0], dtype=np.int64) for r in records]
    dense_all = [sparse_to_dense(r[2]) for r in records]

    masks = []
    N = len(records)
    for i in range(0, N, batch_size):
        batch = np.stack(xc0h_all[i:i + batch_size])
        with lock:
            with torch.no_grad():
                xb        = torch.from_numpy(batch).to(device)
                logits, _ = model(xb)
                logits_np = logits.float().cpu().numpy()
        masks.extend(cdf_legal_masks(logits_np, cdf_cutoff))

    new_records = []
    for rec, dense, mask in zip(records, dense_all, masks):
        xc0h, _, _, wdl, w1, w2, src = rec
        sparse = filter_policy(dense, mask, prior_clip_max)
        new_records.append((xc0h, None, sparse, wdl, w1, w2, src))

    tmp = path + '.tmp'
    with gzip.open(tmp, 'wb') as f:
        pickle.dump(new_records, f, protocol=4)
    os.replace(tmp, path)

    sz_mb = os.path.getsize(path) / 1e6
    return len(records), sz_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-tag',      default='18m_10c6t_SWA_selfplay6')
    ap.add_argument('--cdf',            type=float, default=0.999)
    ap.add_argument('--prior-clip-max', type=float, default=0.75)
    ap.add_argument('--workers',        type=int,   default=4)
    ap.add_argument('--batch',          type=int,   default=256)
    args = ap.parse_args()

    merge_batch1()

    model_pt = os.path.join(SP_DIR, args.model_tag, f'{args.model_tag}_model.pt')
    print(f'loading model {model_pt}', flush=True)
    model, arch = load_pt_model(model_pt)
    device = torch.device('cuda')
    model  = model.to(device).half().eval()
    print(f'arch={arch}  device={device}', flush=True)

    shards = sorted(
        os.path.join(LC0_DIR, f) for f in os.listdir(LC0_DIR)
        if f.endswith('.pkl.gz')
    )
    print(f'{len(shards):,} shards to process  cdf={args.cdf}  '
          f'clip_max={args.prior_clip_max}  workers={args.workers}', flush=True)

    lock      = threading.Lock()
    t0        = time.time()
    done      = 0
    total_pos = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                process_shard, path, model, device, lock,
                args.batch, args.cdf, args.prior_clip_max
            ): path
            for path in shards
        }
        for fut in as_completed(futs):
            path = futs[fut]
            try:
                n_pos, sz_mb = fut.result()
            except Exception as e:
                print(f'  ERROR {os.path.basename(path)}: {e}', flush=True)
                continue
            done      += 1
            total_pos += n_pos
            elapsed    = time.time() - t0
            pos_s      = total_pos / max(elapsed, 1e-6)
            pct        = 100 * done / len(shards)
            print(f'  [{done:3d}/{len(shards)}  {pct:.1f}%]  '
                  f'{os.path.basename(path)}  {n_pos} pos  {sz_mb:.2f} MB  '
                  f'{pos_s:.0f} pos/s', flush=True)

    elapsed = time.time() - t0
    print(f'\ndone: {done}/{len(shards)} shards  {total_pos:,} pos  '
          f'{elapsed:.1f}s  {total_pos/max(elapsed,1e-6):.0f} pos/s', flush=True)


if __name__ == '__main__':
    main()
