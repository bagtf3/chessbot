"""Multi-process migration: N worker processes, one per run_tag.

Workers lazily iterate game logs, write lean per-game pkl.gz files, and
forward (lc0_feats, xc0h) bundles to the parent via a shared queue. The
parent owns the bt4 Lc0Thread and is the sole writer of lc0 distilled shards.

Usage:
    python scripts/data/migrate_game_logs_mp.py \
        --run_tags 18m_10c6t_SWA_selfplay3 18m_10c6t_SWA_selfplay4 \
        --prior-max-clip 0.65 \
        --lc0-distill-cpl 30 \
        [--lc0_model bt4] [--batch 128] [--review_rate 0.005]
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import pickle
import queue
import random
import sys
import threading
import time

import numpy as np

from chessbot import SP_DIR
from chessbot.replay_buffer import sparsify_policy, new_shard_path, write_pkl_gz_shard

LC0_SHARD_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\lc0_distilled_positions"
LC0_BATCH_SIZE = 128
CPL_LC0_THRESHOLD = 30

SENTINEL = None


def stable_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def align_sf(moves, sf_rows):
    sf_rows = sf_rows.copy()
    sf_rows['move_num_int'] = sf_rows['move_num'].astype(int)
    sf_rows = sf_rows.sort_values('move_num_int', kind='stable').reset_index(drop=True)
    by_ply = {}
    j = 0
    for ridx, r in sf_rows.iterrows():
        uci = r.get('played_move', '')
        if not uci:
            continue
        for k in range(j, len(moves)):
            if moves[k] == uci:
                by_ply[k] = ridx
                j = k + 1
                break
    return sf_rows, by_ply


def kl_divergence(visits, priors):
    v = np.array(visits, dtype=np.float64)
    p = np.array(priors, dtype=np.float64)
    v_sum, p_sum = v.sum(), p.sum()
    if v_sum <= 0 or p_sum <= 0:
        return 0.0
    v, p = v / v_sum, p / p_sum
    mask = (v > 0) & (p > 0)
    if not mask.any():
        return 0.0
    return float(np.sum(v[mask] * np.log(v[mask] / p[mask])))


def build_policy_sparse(board, candidate_moves, N_1858, prior_clip_max=None):
    visits_map = {c['uci']: c['visits'] for c in candidate_moves}
    legal = board.legal_moves()
    for u in legal:
        visits_map[u] = max(visits_map.get(u, 0), 1)
    ucis = list(visits_map.keys())
    counts = np.array([visits_map[u] for u in ucis], dtype=np.float32)
    counts /= counts.sum()

    k = len(legal)
    if prior_clip_max is not None and k >= 5:
        top = counts.max()
        if top > prior_clip_max:
            uniform = 1.0 / k
            alpha = (prior_clip_max - top) / (uniform - top)
            counts = alpha * uniform + (1.0 - alpha) * counts

    counts = counts.astype(np.float16)
    indices = board.moves_to_indices(ucis)
    policy = np.zeros(N_1858, dtype=np.float16)
    for ci, prob in zip(indices, counts):
        policy[ci] += prob
    nonzero = np.nonzero(policy)[0]
    return list(zip(nonzero.tolist(), policy[nonzero].tolist()))


def build_raw_visits_sparse(board, candidate_moves, N_1858):
    """Raw tree visit counts, no floor/normalization. (int16 indices, int16 counts)."""
    with_visits = [(c['uci'], c['visits']) for c in candidate_moves if c.get('visits', 0) > 0]
    if not with_visits:
        return (np.array([], dtype=np.int16), np.array([], dtype=np.int16))
    ucis, counts = zip(*with_visits)
    raw = np.zeros(N_1858, dtype=np.int32)
    for ci, v in zip(board.moves_to_indices(list(ucis)), counts):
        raw[ci] += v
    nz = np.nonzero(raw)[0]
    return (nz.astype(np.int16), raw[nz].astype(np.int16))


def load_game_index(run_tag):
    path = os.path.join(SP_DIR, run_tag, 'game_index.json')
    if not os.path.exists(path):
        return {}
    idx = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gid = row.get('game_id')
            if gid:
                idx[gid] = row
    return idx


def load_sf_index(run_tag, needed_ids=None):
    import pandas as pd
    run_dir = os.path.join(SP_DIR, run_tag)

    combined = os.path.join(run_dir, 'analyze_results_combined.pkl')
    if os.path.exists(combined):
        with open(combined, 'rb') as f:
            data = pickle.load(f)
        df = data['df_all']
    else:
        parquet = os.path.join(run_dir, 'df_all.parquet')
        if not os.path.exists(parquet):
            return {}
        df = pd.read_parquet(parquet)
        if 'loss' not in df.columns and 'delta' in df.columns:
            df = df.rename(columns={'delta': 'loss'})

    if needed_ids is not None:
        df = df[df['game_id'].isin(needed_ids)]

    return {gid: sub for gid, sub in df.groupby('game_id', sort=False)}


def write_index_row(fh, index_row, record):
    plies = record['plies']
    cpl_vals = [p['cpl'] for p in plies if 'cpl' in p]
    mean_cpl = round(sum(cpl_vals) / len(cpl_vals), 4) if cpl_vals else None
    mean_bmr = round(sum(1 for c in cpl_vals if c == 0) / len(cpl_vals), 4) if cpl_vals else None

    game_id = record['header']['game_id']
    pkl_file = index_row.get('pkl_file', '')
    new_pkl = os.path.join(os.path.dirname(os.path.dirname(pkl_file)),
                           'game_logs', f'{game_id}.pkl.gz')

    row = dict(index_row)
    row['pkl_file'] = new_pkl
    row['reviewable'] = record['header']['reviewable']
    row['mean_cpl'] = mean_cpl
    row['mean_bmr'] = mean_bmr
    fh.write(json.dumps(row) + '\n')
    fh.flush()


class Lc0Thread(threading.Thread):
    """Owns bt4 inference. Drains (lc0_feats, xc0h) from a queue.Queue,
    fires batches, writes 10240-record shards to LC0_SHARD_DIR."""

    def __init__(self, infer_fn, model_name='lc0', batch_size=LC0_BATCH_SIZE,
                 shard_dir=LC0_SHARD_DIR):
        super().__init__(daemon=True)
        self.infer_fn = infer_fn
        self.model_name = model_name
        self.batch_size = batch_size
        self.shard_dir = shard_dir
        self.q = queue.Queue(maxsize=batch_size * 32)
        self.buffer = []
        self.n_written = 0
        self.n_shards = 0
        self.n_sentinels = 0
        self.n_workers = 0
        self.start_time = None

    def run(self):
        os.makedirs(self.shard_dir, exist_ok=True)
        self.start_time = time.time()
        batch = []
        while True:
            try:
                item = self.q.get(timeout=1.0)
            except queue.Empty:
                if batch:
                    self.fire(batch)
                    batch = []
                continue

            if item is SENTINEL:
                self.n_sentinels += 1
                if self.n_sentinels >= self.n_workers:
                    if batch:
                        self.fire(batch)
                    self.flush()
                    break
                continue

            batch.append(item)
            if len(batch) >= self.batch_size:
                self.fire(batch)
                batch = []

    def fire(self, batch):
        feats = np.stack([b[0] for b in batch], axis=0)
        logits, wdl = self.infer_fn((feats,))
        for i, (_, xc0h, lc0_idx, xc0_idx, clip_max) in enumerate(batch):
            raw = logits[i][lc0_idx.astype(np.int32)]
            raw = raw - raw.max()
            p = np.exp(raw)
            p /= p.sum()
            k = len(xc0_idx)
            if clip_max is not None and k >= 5:
                top = p.max()
                if top > clip_max:
                    uniform = 1.0 / k
                    alpha = (clip_max - top) / (uniform - top)
                    p = alpha * uniform + (1.0 - alpha) * p
            rec = (
                xc0h,
                None,
                (xc0_idx, p.astype(np.float32)),
                wdl[i].astype(np.float32),
                1.0,
                1.0,
                'lc0_distilled',
            )
            self.buffer.append(rec)
        while len(self.buffer) >= 10240:
            shard, self.buffer = self.buffer[:10240], self.buffer[10240:]
            path = new_shard_path(self.shard_dir, suffix=f'-{self.model_name}.pkl.gz')
            write_pkl_gz_shard(shard, path)
            self.n_written += 10240
            self.n_shards += 1

    def flush(self):
        if self.buffer:
            path = new_shard_path(self.shard_dir, suffix=f'-{self.model_name}.pkl.gz')
            write_pkl_gz_shard(self.buffer, path)
            self.n_written += len(self.buffer)
            self.n_shards += 1
            self.buffer = []


def worker(run_tag, lc0_mp_q, stats_q, review_rate, prior_clip_override,
           lc0_distill_cpl, worker_id):
    """Migrates all game logs for one run_tag. Writes lean pkl.gz to
    game_logs_tmp and forwards lc0 bundles to lc0_mp_q."""
    import pyfastchess as pf
    from chessbot.game_utils import reconcile_game_boards, STARTPOS_FEN
    from chessbot.rescore import blend_wdl, z_to_wdl

    SL_IDX = np.nonzero(pf.build_sometimes_legal_mask().astype(bool))[0]
    N_1858 = len(SL_IDX)

    run_dir = os.path.join(SP_DIR, run_tag)
    log_dir = os.path.join(run_dir, 'game_logs')
    out_dir = os.path.join(run_dir, 'game_logs_tmp')
    os.makedirs(out_dir, exist_ok=True)

    idx_out = os.path.join(run_dir, 'game_index_migrated.jsonl')

    # Resume dedup (cheap): a game is done only with BOTH a jsonl row AND a tmp
    # file. Clean up partial state (orphan file or orphan row) before resuming.
    jsonl_rows = {}
    if os.path.exists(idx_out):
        with open(idx_out, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    gid = row.get('game_id')
                    if gid:
                        jsonl_rows[gid] = line
                except Exception:
                    pass

    tmp_files = {
        os.path.splitext(os.path.splitext(f)[0])[0]  # strip .pkl.gz
        for f in os.listdir(out_dir) if f.endswith('.pkl.gz')
    }

    done_ids = set(jsonl_rows) & tmp_files

    for gid in tmp_files - done_ids:
        try:
            os.remove(os.path.join(out_dir, f'{gid}.pkl.gz'))
        except OSError:
            pass

    with open(idx_out, 'w', encoding='utf-8') as fh:
        for gid in done_ids:
            fh.write(jsonl_rows[gid] + '\n')

    # game_index is a cheap json read; use it to decide remaining work BEFORE any
    # heavy load. Exclude already-done and blunder_replay (skipped outright).
    game_index = load_game_index(run_tag)

    all_files = [
        os.path.join(log_dir, f) for f in os.listdir(log_dir)
        if f.endswith('.pkl.gz') or f.endswith('.pkl')
    ]
    n_total = len(all_files)

    work_files = []
    for path in all_files:
        fid = os.path.basename(path).replace('.pkl.gz', '').replace('.pkl', '')
        if fid in done_ids:
            continue
        if game_index.get(fid, {}).get('scenario') == 'blunder_replay':
            continue
        work_files.append(path)
    random.shuffle(work_files)

    n_ok = len(done_ids)
    n_lc0 = 0
    n_plies = 0
    start_time = time.time()

    def push_stats(done=False):
        elapsed = time.time() - start_time
        stats_q.put({
            'run_tag': run_tag,
            'n_done': n_ok,
            'n_total': n_total,
            'n_lc0': n_lc0,
            'n_plies': n_plies,
            'elapsed': elapsed,
            'finished': done,
        })

    # No-op fast path: nothing to migrate, skip the expensive sf_index load.
    if not work_files:
        push_stats(done=True)
        lc0_mp_q.put(SENTINEL)
        return

    # Heavy load only for the games we will actually migrate.
    needed_ids = {
        os.path.basename(p).replace('.pkl.gz', '').replace('.pkl', '')
        for p in work_files
    }
    sf_index = load_sf_index(run_tag, needed_ids=needed_ids)

    with open(idx_out, 'a', encoding='utf-8') as idx_fh:
        for i, path in enumerate(work_files):
            opener = gzip.open if path.endswith('.gz') else open
            try:
                with opener(path, 'rb') as f:
                    log = pickle.load(f)
            except Exception:
                continue

            sc = log.get('scenario', '')
            if sc == 'blunder_replay':
                continue
            is_review = (sc == 'paired_validation') or (random.random() < review_rate)
            game_id = log.get('game_id', os.path.basename(path).replace('.pkl.gz', '').replace('.pkl', ''))

            if game_id in done_ids:
                continue

            try:
                record, lc0_count = migrate_game_worker(
                    log, is_review, sf_index, lc0_mp_q,
                    lc0_distill_cpl, prior_clip_override,
                    pf, reconcile_game_boards, blend_wdl, z_to_wdl,
                    N_1858,
                )
            except Exception as e:
                print(f"[w{worker_id}] skip {game_id}: {e}", flush=True)
                continue

            n_lc0 += lc0_count
            n_plies += len(record['plies'])
            out_path = os.path.join(out_dir, f"{game_id}.pkl.gz")
            with gzip.open(out_path, 'wb', compresslevel=6) as f:
                pickle.dump(record, f, protocol=4)

            if game_id in game_index:
                write_index_row(idx_fh, game_index[game_id], record)

            n_ok += 1
            if n_ok % 200 == 0:
                push_stats()

    push_stats(done=True)
    lc0_mp_q.put(SENTINEL)


def migrate_game_worker(log, is_review, sf_index, lc0_mp_q,
                        lc0_distill_cpl, prior_clip_override,
                        pf, reconcile_game_boards, blend_wdl, z_to_wdl,
                        N_1858):
    """Returns (record, n_lc0_enqueued)."""
    from chessbot.game_utils import STARTPOS_FEN
    n_lc0_enqueued = 0
    history_K = log.get('history_K', 6)
    moves = log.get('moves_played', [])
    start_fen = log.get('start_fen') or STARTPOS_FEN
    scenario = log.get('scenario', '')
    tree_data = {int(k): v for k, v in log.get('tree_search_data', {}).items()}

    no_warm = scenario in ('piece_odds', 'piece_training', 'random_init',
                           'random_middle_game', 'random_endgame')
    if no_warm:
        board = pf.Board(start_fen)
    else:
        _, board = reconcile_game_boards(start_fen, log.get('history_uci'), moves)

    prior_clip_max = prior_clip_override if prior_clip_override is not None else log.get('prior_clip_max')
    result = log.get('result', 0)

    sf_by_ply, sf_rows = {}, None
    if sf_index is not None:
        sf_df = sf_index.get(log.get('game_id'))
        if sf_df is not None and len(sf_df):
            _, sf_by_ply = align_sf(moves, sf_df)
            sf_rows = sf_df.sort_values('move_num').reset_index(drop=True)

    header = {
        'game_id':            log.get('game_id'),
        'run_tag':            log.get('run_tag'),
        'model_epoch':        log.get('model_epoch'),
        'scenario':           scenario,
        'result':             log.get('result', 0),
        'end_reason':         log.get('end_reason'),
        'start_fen':          start_fen,
        'history_uci':        log.get('history_uci'),
        'vs_stockfish':       log.get('vs_stockfish', False),
        'stockfish_color':    log.get('stockfish_color'),
        'reviewable':         is_review,
        'c_puct':             log.get('c_puct'),
        'sims_floor':         log.get('sims_floor'),
        'sims_ceiling':       log.get('sims_ceiling'),
        'dirichlet_alpha':    log.get('dirichlet_alpha'),
        'dirichlet_eps':      log.get('dirichlet_eps'),
        'uniform_eps':        log.get('uniform_eps'),
        'fpu_reduction':      log.get('fpu_reduction'),
        'move_sample_temp_range': log.get('move_sample_temp_range'),
        'move_sample_temp_plies': log.get('move_sample_temp_plies'),
        'history_K':          history_K,
    }

    plies = []
    for ply, move_played in enumerate(moves):
        node = tree_data.get(ply)
        xc0h = np.asarray(board.history_tokens(history_K), dtype=np.int16)

        is_white = board.white_to_move()
        wdl_white_pov = node.get('best_wdl') if node else None
        if wdl_white_pov is not None and not is_white:
            best_wdl_stm = (wdl_white_pov[2], wdl_white_pov[1], wdl_white_pov[0])
        else:
            best_wdl_stm = wdl_white_pov

        z_stm = result if is_white else -result
        z_wdl = z_to_wdl(z_stm)
        wdl_target = blend_wdl(z_stm, wdl_white_pov, is_white)

        entry = {
            'stm':         is_white,
            'move_played': move_played,
            'sel_method':  node.get('selection_method') if node else None,
            'xc0_move':    node.get('xc0_move') if node else None,
            'stop_reason': node.get('stop_reason') if node else None,
            'best_wdl':    best_wdl_stm,
            'z_wdl':       z_wdl,
            'wdl_target':  wdl_target,
            'xc0h':        xc0h,
        }

        cms = node.get('candidate_moves', []) if node else []
        if cms:
            entry['policy']     = build_policy_sparse(board, cms, N_1858, prior_clip_max)
            entry['raw_visits'] = build_raw_visits_sparse(board, cms, N_1858)
            visits = [c['visits'] for c in cms]
            priors = [c['P'] for c in cms]
            entry['kl'] = round(kl_divergence(visits, priors), 5)
        else:
            entry['policy']     = []
            entry['raw_visits'] = (np.array([], dtype=np.int16), np.array([], dtype=np.int16))
            entry['kl'] = 0.0

        if is_review and node:
            entry['sims'] = node.get('sims')
            entry['time'] = node.get('time')
            entry['avg_depth'] = node.get('avg_depth')
            entry['max_depth'] = node.get('max_depth')
            entry['children_visited'] = node.get('children_visited')
            entry['total_children'] = node.get('total_children')
            entry['candidate_moves'] = cms
            entry['pv'] = node.get('pv', [])

        if sf_rows is not None:
            idx = sf_by_ply.get(ply)
            if idx is not None:
                loss = sf_rows.iloc[idx]['loss']
                if loss == loss:
                    entry['cpl'] = int(loss)
                elif entry.get('sel_method') == 'stockfish':
                    if entry.get('xc0_move') == move_played:
                        entry['cpl'] = 0

        queued = False
        if lc0_mp_q is not None:
            cpl = entry.get('cpl')
            sel = entry.get('sel_method')
            sf_disagree = sel == 'stockfish' and 'cpl' not in entry
            high_cpl = cpl is not None and cpl > lc0_distill_cpl
            if sf_disagree or high_cpl:
                from chessbot.lc0_utils import lc0_table_index
                from xerces_training.uci_to_idx import uci_to_idx as UCI_TO_IDX
                lc0_feats  = board.lc0_features()
                legal_ucis = board.legal_moves()
                tbl        = UCI_TO_IDX[lc0_table_index(lc0_feats)]
                lc0_idx    = np.array([tbl[u.rstrip('n')] for u in legal_ucis], dtype=np.int16)
                xc0_idx    = np.array(board.moves_to_indices(legal_ucis), dtype=np.int16)
                lc0_mp_q.put((lc0_feats, xc0h, lc0_idx, xc0_idx, prior_clip_max))
                queued = True

        plies.append(entry)
        board.push_uci(move_played)
        if queued:
            n_lc0_enqueued += 1

    return {'header': header, 'plies': plies}, n_lc0_enqueued


def count_games(run_tag):
    path = os.path.join(SP_DIR, run_tag, 'game_index.json')
    if not os.path.exists(path):
        return 0
    with open(path, encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_tags', nargs='+', required=True)
    ap.add_argument('--max-workers', type=int, default=0, dest='max_workers',
                    help='max concurrent worker processes (default: one per run_tag)')
    ap.add_argument('--review_rate', type=float, default=0.005)
    ap.add_argument('--prior-max-clip', type=float, default=None, dest='prior_max_clip')
    ap.add_argument('--lc0_model', default='',
                    help='lc0 model name (default: LC0_DISTILL_MODEL env var)')
    ap.add_argument('--lc0-distill-cpl', type=int, default=CPL_LC0_THRESHOLD,
                    dest='lc0_distill_cpl')
    ap.add_argument('--batch', type=int, default=LC0_BATCH_SIZE,
                    help='lc0 inference batch size')
    ap.add_argument('--skip-lc0', action='store_true')
    args = ap.parse_args()

    max_workers = args.max_workers or len(args.run_tags)

    print("[main] counting games per run_tag...", flush=True)
    game_counts = {tag: count_games(tag) for tag in args.run_tags}
    sorted_tags = sorted(args.run_tags, key=lambda t: game_counts[t], reverse=True)
    for tag in sorted_tags:
        print(f"  {tag}: {game_counts[tag]:,} games", flush=True)

    lc0_thread = None
    lc0_mp_q = None

    if not args.skip_lc0:
        lc0_model = args.lc0_model or os.environ.get('LC0_DISTILL_MODEL', '')
        lc0_cache = os.environ.get('LC0_DISTILL_TRT_CACHE', '')
        if not lc0_model or not lc0_cache:
            raise RuntimeError(
                "lc0 requires LC0_DISTILL_MODEL and LC0_DISTILL_TRT_CACHE env vars"
            )
        from chessbot.lc0_utils import make_lc0_trt_session, make_lc0_infer
        lc0_onnx = os.path.join(lc0_cache, f'{lc0_model}.onnx')
        sess = make_lc0_trt_session(lc0_onnx, lc0_model, lc0_cache, args.batch, args.batch * 2)
        infer_fn = make_lc0_infer(sess)

        lc0_mp_q = mp.Queue(maxsize=args.batch * 64)
        lc0_thread = Lc0Thread(infer_fn, model_name=lc0_model, batch_size=args.batch)
        lc0_thread.n_workers = len(sorted_tags)
        bridge_stop = threading.Event()

        def bridge():
            while not bridge_stop.is_set():
                try:
                    item = lc0_mp_q.get(timeout=0.5)
                    lc0_thread.q.put(item)
                except Exception:
                    continue

        bridge_thread = threading.Thread(target=bridge, daemon=True)
        bridge_thread.start()
        lc0_thread.start()
        print(f"[lc0] thread started ({lc0_model}, CPL > {args.lc0_distill_cpl}, "
              f"batch={args.batch}, total_workers={len(sorted_tags)})", flush=True)

    stats_q = mp.Queue()
    run_start = time.time()
    worker_stats = {tag: None for tag in sorted_tags}

    # shared state for reporter (written by main thread, read by reporter thread)
    pool_state = {
        'queued':  list(sorted_tags[max_workers:]),
        'active':  set(sorted_tags[:max_workers]),
        'done':    [],
    }

    def reporter():
        while True:
            try:
                while True:
                    s = stats_q.get_nowait()
                    worker_stats[s['run_tag']] = s
            except Exception:
                pass

            elapsed = time.time() - run_start
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            sec = int(elapsed % 60)
            elapsed_str = f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

            n_done = len(pool_state['done'])
            n_tags = len(sorted_tags)
            print(f"\n=== Migration [{elapsed_str}] | {n_done}/{n_tags} tags done ===\n", flush=True)

            col = "  {:<32} {:>14}   {:>5}   {:>7}   {:>6}   {:>6}"
            print(col.format("tag", "done/total", "f/s", "ply/s", "lc0/s", "ETA"), flush=True)

            active = list(pool_state['active'])
            tot_files = tot_plies = tot_lc0 = 0
            for tag in active:
                s = worker_stats.get(tag)
                if s is None:
                    print(f"  {tag:<32}  starting...", flush=True)
                    continue
                n_ok  = s['n_done']
                n_tot = s['n_total']
                n_lc0 = s['n_lc0']
                n_pl  = s.get('n_plies', 0)
                el    = max(s['elapsed'], 1.0)
                fps   = n_ok / el
                plps  = n_pl / el
                lc0ps = n_lc0 / el
                remaining = n_tot - n_ok
                eta_s = remaining / fps if fps > 0 else 0
                eta_str = f"~{int(eta_s//60)}m" if eta_s > 60 else f"~{int(eta_s)}s"
                print(col.format(tag, f"{n_ok:,}/{n_tot:,}",
                                 f"{fps:.1f}", f"{plps:.0f}", f"{lc0ps:.1f}", eta_str), flush=True)
                tot_files += n_ok
                tot_plies += n_pl
                tot_lc0   += n_lc0

            if active:
                el_total = max(time.time() - run_start, 1.0)
                print(col.format("TOTAL", f"{tot_files:,}",
                                 f"{tot_files/el_total:.1f}",
                                 f"{tot_plies/el_total:.0f}",
                                 f"{tot_lc0/el_total:.1f}", ""), flush=True)

            if lc0_thread is not None:
                lc0_el = max(time.time() - (lc0_thread.start_time or run_start), 1.0)
                lc0ps = lc0_thread.n_written / lc0_el
                print(f"\n  lc0/{lc0_thread.model_name}   {lc0_thread.n_written:,} pos"
                      f"  {lc0_thread.n_shards} shards  {lc0ps:.1f} pos/s", flush=True)

            queued = pool_state['queued']
            if queued:
                print("", flush=True)
                for tag in queued:
                    print(f"  {tag:<32}  queued ({game_counts[tag]:,} games)", flush=True)

            done = pool_state['done']
            if done:
                print("", flush=True)
                for tag in done:
                    s = worker_stats[tag]
                    n_ok  = s['n_done']
                    n_pl  = s.get('n_plies', 0)
                    n_lc0 = s['n_lc0']
                    el    = max(s['elapsed'], 1.0)
                    print(f"  {tag:<32}  DONE ({n_ok:,} files, {n_pl:,} plies, "
                          f"{n_ok/el:.1f} f/s avg, {n_lc0:,} lc0)", flush=True)

            if n_done >= n_tags:
                break
            time.sleep(30)

    reporter_thread = threading.Thread(target=reporter, daemon=True)
    reporter_thread.start()

    # pool: launch up to max_workers, refill as slots open
    tag_queue = list(sorted_tags)
    active_procs = {}  # tag -> Process
    wid = 0

    for tag in tag_queue[:max_workers]:
        p = mp.Process(
            target=worker,
            args=(tag, lc0_mp_q, stats_q, args.review_rate, args.prior_max_clip,
                  args.lc0_distill_cpl, wid),
            daemon=False,
        )
        p.start()
        active_procs[tag] = p
        print(f"[main] started worker {wid} for {tag} ({game_counts[tag]:,} games)", flush=True)
        wid += 1

    pending = tag_queue[max_workers:]

    while active_procs or pending:
        for tag in list(active_procs):
            if not active_procs[tag].is_alive():
                active_procs[tag].join()
                del active_procs[tag]
                pool_state['active'].discard(tag)
                pool_state['done'].append(tag)

                if pending:
                    next_tag = pending.pop(0)
                    pool_state['queued'].remove(next_tag)
                    pool_state['active'].add(next_tag)
                    p = mp.Process(
                        target=worker,
                        args=(next_tag, lc0_mp_q, stats_q, args.review_rate,
                              args.prior_max_clip, args.lc0_distill_cpl, wid),
                        daemon=False,
                    )
                    p.start()
                    active_procs[next_tag] = p
                    print(f"[main] started worker {wid} for {next_tag} "
                          f"({game_counts[next_tag]:,} games)", flush=True)
                    wid += 1
        time.sleep(1)

    reporter_thread.join(timeout=5)

    if lc0_thread is not None:
        print("[lc0] all workers done, waiting for thread to flush...", flush=True)
        lc0_thread.join()
        bridge_stop.set()
        print(f"[lc0] done, {lc0_thread.n_written:,} records written to {LC0_SHARD_DIR}",
              flush=True)

    print("[main] all done", flush=True)
    os._exit(0)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
