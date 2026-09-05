"""Validate the xc0 prod model across every data source.

Samples records from each tap by sampling FILENAMES first, then loads only the
chosen files (never loads a whole run_tag to sample). Runs the prod model in pt
eager fp16 (batch 256) and reports, per source: CE(WDL), policy CE, MSE(Q) and
corr(Q), where Q = p_win - p_loss (prod prediction vs the source's stored target).

Usage:
    python scripts/data/validate_prod_on_sources.py
        [--model-tag 18m_10c6t_SWA_selfplay6] [--games 100]
        [--distill-shards 10] [--brp-shards 5] [--lc0pkl-games 100] [--seed 0]
"""
import argparse
import gzip
import os
import pickle
import random

import numpy as np
import scipy.stats
import torch

from chessbot import SP_DIR
from chessbot.replay_buffer import POLICY_DIM
from chessbot.train_pytorch import load_pt_model

TD    = r"C:\Users\Bryan\Data\chessbot_data\training_data"
BATCH = 256

RUN_TAGS = [
    "16m_precond_run0", "16m_precond_run1", "16m_precond_run2",
    "16m_xc0hK6_run1", "16m_xc0hK6_run2", "16m_xc0hK6_run3", "16m_xc0hK6_run4",
    "18m_10c6t_d256_pretrain", "18m_10c6t_d256_selfplay0", "18m_10c6t_d256_selfplay1",
    "18m_10c6t_SWA_selfplay0", "18m_10c6t_SWA_selfplay1", "18m_10c6t_SWA_selfplay2",
    "18m_10c6t_SWA_selfplay3", "18m_10c6t_SWA_selfplay4", "18m_10c6t_SWA_selfplay5",
    "18m_10c6t_SWA_selfplay6",
    "18m_6c4t_pretrain_run0", "18m_6c4t_selfplay_run0",
]


def policy_to_dense(policy):
    """Scatter a sparse policy to dense[POLICY_DIM].

    Handles shard form (idx_array, vals_array) and per-game form
    list-of-(int, float16). Missing/empty -> all zeros.
    """
    p = np.zeros(POLICY_DIM, dtype=np.float32)
    if not policy:
        return p
    if isinstance(policy, tuple):
        idx, vals = policy
        if len(idx):
            p[np.asarray(idx, dtype=np.int32)] = np.asarray(vals, dtype=np.float32)
    else:
        for i, v in policy:
            p[int(i)] = float(v)
    return p


def sample_names(d, n):
    names = [f for f in os.listdir(d) if f.endswith('.pkl.gz')]
    return random.sample(names, min(n, len(names)))


def shard_source(d, n_shards):
    """Each chosen shard is a list of 7-tuples (xc0h, None, policy, wdl, ...)."""
    recs = []
    for name in sample_names(d, n_shards):
        with gzip.open(os.path.join(d, name), 'rb') as f:
            shard = pickle.load(f)
        for r in shard:
            recs.append((
                np.asarray(r[0], dtype=np.int64),
                policy_to_dense(r[2]),
                np.asarray(r[3], dtype=np.float32),
            ))
    return recs


def plies_from_game(path):
    with gzip.open(path, 'rb') as f:
        g = pickle.load(f)
    out = []
    for ply in g.get('plies', []):
        xc0h = ply.get('xc0h')
        if xc0h is None or 'wdl_target' not in ply:
            continue
        out.append((
            np.asarray(xc0h, dtype=np.int64),
            policy_to_dense(ply.get('policy')),
            np.asarray(ply['wdl_target'], dtype=np.float32),
        ))
    return out


def pergame_source(d, n_games):
    recs = []
    for name in sample_names(d, n_games):
        recs.extend(plies_from_game(os.path.join(d, name)))
    return recs


def run_tag_dir(tag):
    rd  = os.path.join(SP_DIR, tag)
    tmp = os.path.join(rd, 'game_logs_tmp')
    if os.path.isdir(tmp) and any(f.endswith('.pkl.gz') for f in os.listdir(tmp)):
        return tmp
    return os.path.join(rd, 'game_logs')


def lc0pkl_source(n_games):
    root  = os.path.join(TD, 'lc0_pkl')
    files = []
    for sub in os.listdir(root):
        dd = os.path.join(root, sub)
        if os.path.isdir(dd):
            files += [os.path.join(dd, f) for f in os.listdir(dd)
                      if f.endswith('.pkl.gz')]
    recs = []
    for p in random.sample(files, min(n_games, len(files))):
        recs.extend(plies_from_game(p))
    return recs


def evaluate(recs, model, device):
    """Stream through batches; never materialize full logits/policy arrays."""
    n = len(recs)
    Qp = np.empty(n, dtype=np.float64)
    Qt = np.empty(n, dtype=np.float64)
    ce_sum = 0.0
    pce_sum = 0.0
    pce_n = 0
    with torch.no_grad():
        for i in range(0, n, BATCH):
            batch = recs[i:i + BATCH]
            xb = torch.from_numpy(
                np.stack([b[0] for b in batch]).astype(np.int64)).to(device)
            logits, value = model(xb)
            logits = logits.float().cpu().numpy()
            wdlp   = torch.softmax(value.float(), dim=-1).cpu().numpy()

            Yt = np.stack([b[2] for b in batch])
            Pd = np.stack([b[1] for b in batch])

            Qp[i:i + len(batch)] = wdlp[:, 0] - wdlp[:, 2]
            Qt[i:i + len(batch)] = Yt[:, 0] - Yt[:, 2]

            ce_sum += -np.sum(Yt * np.log(np.clip(wdlp, 1e-9, 1.0)))

            maxl  = logits.max(-1, keepdims=True)
            logsm = logits - maxl - np.log(
                np.sum(np.exp(logits - maxl), axis=-1, keepdims=True))
            has = Pd.sum(-1) > 0
            if has.any():
                pce_sum += -np.sum(Pd[has] * logsm[has])
                pce_n   += int(has.sum())

    ce   = ce_sum / n
    pce  = pce_sum / pce_n if pce_n else float('nan')
    mse  = float(np.mean((Qp - Qt) ** 2))
    corr = float(scipy.stats.pearsonr(Qp, Qt)[0])
    return n, ce, pce, mse, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-tag', default='18m_10c6t_SWA_selfplay6')
    ap.add_argument('--games',          type=int, default=100)
    ap.add_argument('--distill-shards', type=int, default=10, dest='distill_shards')
    ap.add_argument('--brp-shards',     type=int, default=5,  dest='brp_shards')
    ap.add_argument('--lc0pkl-games',   type=int, default=100, dest='lc0pkl_games')
    ap.add_argument('--seed',           type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    model_pt = os.path.join(SP_DIR, args.model_tag, f'{args.model_tag}_model_player.pt')
    print(f"loading prod model {model_pt}", flush=True)
    model, arch = load_pt_model(model_pt)
    device = torch.device('cuda')
    model = model.to(device).half().eval()
    print(f"arch={arch}  device={device}\n", flush=True)

    sources = [
        ('lc0_distilled_positions',
         lambda: shard_source(os.path.join(TD, 'lc0_distilled_positions'),
                              args.distill_shards)),
        ('lc0_BRP',
         lambda: shard_source(os.path.join(TD, 'lc0_BRP'), args.brp_shards)),
        ('lc0_pkl',
         lambda: lc0pkl_source(args.lc0pkl_games)),
    ]
    for tag in RUN_TAGS:
        sources.append((tag, lambda t=tag: pergame_source(run_tag_dir(t), args.games)))

    hdr = f"  {'source':<28} {'N':>9} {'CE(WDL)':>9} {'polCE':>8} {'MSE(Q)':>9} {'corr(Q)':>8}"
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    rows = []
    for name, loader in sources:
        try:
            recs = loader()
        except Exception as e:
            print(f"  {name:<28}  ERROR: {e}", flush=True)
            continue
        if not recs:
            print(f"  {name:<28} {'0':>9}  (no records)", flush=True)
            continue
        n, ce, pce, mse, corr = evaluate(recs, model, device)
        rows.append((name, n, ce, pce, mse, corr))
        print(f"  {name:<28} {n:>9,} {ce:>9.4f} {pce:>8.3f} "
              f"{mse:>9.5f} {corr:>8.4f}", flush=True)

    print("\ndone.", flush=True)


if __name__ == '__main__':
    main()
