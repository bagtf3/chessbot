"""Migrate old fat game logs to the new lean per-game pkl format.

Usage:
    python scripts/data/migrate_game_logs.py --run_tag 18m_10c6t_SWA_selfplay1 \
        --out_dir /path/to/game_logs_test \
        --n_regular 15 --n_validation 4 --n_review 1
"""
import argparse
import gzip
import math
import os
import pickle
import random

import numpy as np
import pyfastchess as pf

from chessbot import SP_DIR
from chessbot.game_utils import reconcile_game_boards
from chessbot.replay_buffer import sparsify_policy, POLICY_DIM

SL_IDX = np.nonzero(pf.build_sometimes_legal_mask().astype(bool))[0]
N_1858 = len(SL_IDX)


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
    v_sum = v.sum()
    p_sum = p.sum()
    if v_sum <= 0 or p_sum <= 0:
        return 0.0
    v = v / v_sum
    p = p / p_sum
    mask = (v > 0) & (p > 0)
    if not mask.any():
        return 0.0
    return float(np.sum(v[mask] * np.log(v[mask] / p[mask])))


def load_log(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        return pickle.load(f)


def build_policy_sparse(board, candidate_moves, prior_clip_max=None):
    visits_map = {c['uci']: c['visits'] for c in candidate_moves}
    legal = board.legal_moves()
    for u in legal:
        if u not in visits_map:
            visits_map[u] = 1
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


def load_sf_index(run_tag):
    path = os.path.join(SP_DIR, run_tag, 'analyze_results_combined.pkl')
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        data = pickle.load(f)
    df = data['df_all']
    return {gid: sub for gid, sub in df.groupby('game_id', sort=False)}


def migrate_game(log, is_review, sf_index=None):
    history_K = log.get('history_K', 6)
    moves = log.get('moves_played', [])
    start_fen = log.get('start_fen')
    scenario = log.get('scenario', '')
    tree_data = {int(k): v for k, v in log.get('tree_search_data', {}).items()}

    no_warm = scenario in ('piece_odds', 'piece_training', 'random_init',
                           'random_middle_game', 'random_endgame')
    if no_warm:
        _, board = None, pf.Board(start_fen)
    else:
        _, board = reconcile_game_boards(
            start_fen, log.get('history_uci'), moves
        )
    prior_clip_max = log.get('prior_clip_max')

    sf_by_ply = {}
    if sf_index is not None:
        sf_df = sf_index.get(log.get('game_id'))
        if sf_df is not None and len(sf_df):
            _, sf_by_ply = align_sf(moves, sf_df)
            sf_rows = sf_df.sort_values('move_num').reset_index(drop=True)
        else:
            sf_rows = None
    else:
        sf_rows = None

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
        raw_wdl = node.get('best_wdl') if node else None
        if raw_wdl is not None and not is_white:
            raw_wdl = (raw_wdl[2], raw_wdl[1], raw_wdl[0])

        entry = {
            'stm':         is_white,
            'move_played': move_played,
            'sel_method':  node.get('selection_method') if node else None,
            'xc0_move':    node.get('xc0_move') if node else None,
            'stop_reason': node.get('stop_reason') if node else None,
            'best_wdl':    raw_wdl,
            'xc0h':        xc0h,
        }

        cms = node.get('candidate_moves', []) if node else []
        if cms:
            entry['policy'] = build_policy_sparse(board, cms, prior_clip_max)
            visits = [c['visits'] for c in cms]
            priors = [c['P'] for c in cms]
            entry['kl'] = round(kl_divergence(visits, priors), 5)
        else:
            entry['policy'] = []
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
                    entry['cpl'] = 0

        plies.append(entry)
        board.push_uci(move_played)

    return {'header': header, 'plies': plies}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_tag', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_regular', type=int, default=15)
    ap.add_argument('--n_validation', type=int, default=4)
    ap.add_argument('--n_review', type=int, default=1)
    args = ap.parse_args()

    log_dir = os.path.join(SP_DIR, args.run_tag, 'game_logs')
    os.makedirs(args.out_dir, exist_ok=True)

    sf_index = load_sf_index(args.run_tag)
    if sf_index:
        print(f"[sf] loaded {len(sf_index):,} game SF records", flush=True)
    else:
        print("[sf] no per-ply SF data found, cpl will be absent", flush=True)

    all_files = [
        os.path.join(log_dir, f) for f in os.listdir(log_dir)
        if f.endswith('.pkl.gz') or f.endswith('.pkl')
    ]
    random.shuffle(all_files)

    val_files = []
    reg_files = []
    for path in all_files:
        try:
            log = load_log(path)
        except Exception:
            continue
        sc = log.get('scenario', '')
        if sc == 'paired_validation' and len(val_files) < args.n_validation:
            val_files.append((path, log))
        elif sc != 'paired_validation' and len(reg_files) < (args.n_regular + args.n_review):
            reg_files.append((path, log))
        if len(val_files) >= args.n_validation and len(reg_files) >= (args.n_regular + args.n_review):
            break

    review_idx = set(random.sample(range(len(reg_files)), min(args.n_review, len(reg_files))))

    candidates = (
        [(path, log, True) for path, log in val_files] +
        [(path, log, i in review_idx) for i, (path, log) in enumerate(reg_files)]
    )

    for path, log, is_review in candidates:
        game_id = log.get('game_id', os.path.basename(path).replace('.pkl.gz', '').replace('.pkl', ''))
        out_path = os.path.join(args.out_dir, f"{game_id}.pkl.gz")
        try:
            record = migrate_game(log, is_review, sf_index=sf_index if sf_index else None)
        except Exception as e:
            print(f"[skip] {game_id}: {e}", flush=True)
            continue
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(record, f, protocol=4)
        scenario = record['header']['scenario']
        n_plies = len(record['plies'])
        print(f"[ok] {game_id}  scenario={scenario}  plies={n_plies}  review={is_review}", flush=True)

    print(f"\n[done] {len(candidates)} games -> {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()
