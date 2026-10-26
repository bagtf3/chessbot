"""Eval migrated shard data against latest prod model.

Loads smoketest shards, runs inference in pt eager mode on GPU,
reports MSE(Q), CE(WDL), policy CE, corr(Q).

Usage:
    python scripts/data/eval_migrated_data.py --run_tag 16m_precond_run2 --model_tag 18m_10c6t_SWA_selfplay6
"""
import argparse
import glob
import gzip
import os
import pickle
import random

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F

from chessbot import SP_DIR
from chessbot.replay_buffer import POLICY_DIM
from chessbot.train_pytorch import load_pt_model

BATCH = 256


def load_shards(shard_paths):
    xc0h_list, policy_list, wdl_list = [], [], []
    for p in shard_paths:
        with gzip.open(p, 'rb') as f:
            records = pickle.load(f)
        for rec in records:
            xc0h, _, sparse_policy, wdl_target, _, _, _ = rec
            xc0h_list.append(xc0h)
            policy_list.append(sparse_policy)
            wdl_list.append(wdl_target)
    return xc0h_list, policy_list, wdl_list


def sparse_to_dense(sparse_policy, dim=POLICY_DIM):
    idxs, vals = sparse_policy
    p = np.zeros(dim, dtype=np.float32)
    if len(idxs):
        p[idxs.astype(np.int32)] = vals.astype(np.float32)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_tag', required=True)
    ap.add_argument('--model_tag', required=True)
    args = ap.parse_args()

    run_dir   = os.path.join(SP_DIR, args.run_tag)
    model_dir = os.path.join(SP_DIR, args.model_tag)
    model_pt  = os.path.join(model_dir, f'{args.model_tag}_model.pt')

    shard_paths = sorted(glob.glob(os.path.join(run_dir, 'smoketest_shard_*.pkl.gz')))
    if not shard_paths:
        print(f"no smoketest shards found in {run_dir}")
        return
    print(f"loading {len(shard_paths)} shards from {args.run_tag}", flush=True)
    xc0h_list, policy_list, wdl_list = load_shards(shard_paths)

    idx = list(range(len(xc0h_list)))
    random.shuffle(idx)
    xc0h_list   = [xc0h_list[i]   for i in idx]
    policy_list = [policy_list[i] for i in idx]
    wdl_list    = [wdl_list[i]    for i in idx]
    N = len(xc0h_list)
    print(f"{N:,} records shuffled", flush=True)

    print(f"loading model {model_pt}", flush=True)
    model, arch = load_pt_model(model_pt)
    device = torch.device('cuda')
    model = model.to(device).half().eval()
    print(f"arch={arch}  device={device}", flush=True)

    X = np.stack(xc0h_list).astype(np.int64)
    Y_wdl = np.stack(wdl_list).astype(np.float32)
    P_dense = np.stack([sparse_to_dense(p) for p in policy_list])

    all_logits, all_wdl_pred = [], []
    with torch.no_grad():
        for i in range(0, N, BATCH):
            xb = torch.from_numpy(X[i:i+BATCH]).to(device)
            logits, value = model(xb)
            all_logits.append(logits.float().cpu().numpy())
            all_wdl_pred.append(F.softmax(value.float(), dim=-1).cpu().numpy())
            if (i // BATCH) % 20 == 0:
                print(f"  {i+BATCH:,}/{N:,}", flush=True)

    logits   = np.concatenate(all_logits)
    wdl_pred = np.concatenate(all_wdl_pred)

    Q_pred   = wdl_pred[:, 0] - wdl_pred[:, 2]
    Q_target = Y_wdl[:, 0]    - Y_wdl[:, 2]
    mse_q    = float(np.mean((Q_pred - Q_target) ** 2))
    corr_q   = float(scipy.stats.pearsonr(Q_pred, Q_target)[0])

    log_wdl_pred = np.log(np.clip(wdl_pred, 1e-9, 1.0))
    ce_wdl = float(-np.mean(np.sum(Y_wdl * log_wdl_pred, axis=-1)))

    max_l = logits.max(-1, keepdims=True)
    log_p = logits - max_l - np.log(np.sum(np.exp(logits - max_l), axis=-1, keepdims=True))
    has_policy = P_dense.sum(-1) > 0
    policy_ce = float(-np.mean(np.sum(P_dense[has_policy] * log_p[has_policy], axis=-1)))

    print(f"\n{'='*50}", flush=True)
    print(f"  N            {N:,}", flush=True)
    print(f"  MSE(Q)       {mse_q:.5f}", flush=True)
    print(f"  corr(Q)      {corr_q:.4f}", flush=True)
    print(f"  CE(WDL)      {ce_wdl:.5f}", flush=True)
    print(f"  policy CE    {policy_ce:.5f}  (on {has_policy.sum():,} non-empty)", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == '__main__':
    main()
