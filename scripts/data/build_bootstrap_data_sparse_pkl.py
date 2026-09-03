"""Build sparse pkl bootstrap shards from xc0 selfplay games and lc0 V6 chunks.

Usage:
    python scripts/data/build_bootstrap_data_sparse_pkl.py --config config.yaml
"""
import argparse
import gzip
import multiprocessing as mp
import os
import pickle
import queue
import random
import struct
import time
import types

import numpy as np
import yaml
import pyfastchess as pf

from xerces_training.chunkparser import ChunkParser, V6_STRUCT_STRING
from xerces_training.parse_utils import v6_planes_to_xc0h, get_policy_vector

from xerces_training.uci_to_idx import uci_to_idx as UCI_TO_IDX

from chessbot import SP_DIR
from chessbot.lc0_utils import lc0_logits_to_xc0_batch, make_lc0_trt_session
from chessbot.opening_counts import slot_from_x
from chessbot.replay_buffer import (
    LiveBuffer, sparsify_policy, densify_policy,
    new_shard_path, write_pkl_gz_shard, POLICY_DIM,
)
from chessbot.review import load_game_index, ANALYZE_PKL
from chessbot.utils import scalar_to_wdl as to_wdl
from chessbot.game_utils import reconcile_game_boards

SL_IDX = np.nonzero(pf.build_sometimes_legal_mask().astype(bool))[0]
N_1858 = len(SL_IDX)
V6_STRUC = struct.Struct(V6_STRUCT_STRING)

# decayed-count thresholds -> merge_N, highest first
MERGE_BANDS = [
    (2000, 128),
    (501,   64),
    (201,   32),
    (61,    16),
    (31,     4),
    (10,     2),
]


def load_config(path):
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    cfg = types.SimpleNamespace(**raw)
    cfg.skip_scenarios = set(cfg.skip_scenarios)
    cfg.no_warm_scenarios = set(cfg.no_warm_scenarios)
    cfg.lc0_full_scenarios = set(cfg.lc0_full_scenarios)
    return cfg


# ---------------------------------------------------------------------------
# opening consolidator
# ---------------------------------------------------------------------------

class OpeningConsolidator:
    """Defer-accumulate-aggregate for frequent opening positions.

    Loaded from opening_counts.npz once at parent startup; long-lived across
    all run_tags. Positions not in merge_map pass straight through.
    """

    def __init__(self, npz_path):
        with np.load(npz_path) as z:
            idx  = z['idx'].astype(np.int64)
            vals = np.floor(z['vals'])
        self.merge_map = {}
        for slot, cnt in zip(idx, vals):
            for lo, merge_n in MERGE_BANDS:
                if cnt >= lo:
                    self.merge_map[int(slot)] = merge_n
                    break
        self.buffers = {}
        print(f"[consolidator] {len(self.merge_map):,} slots tracked", flush=True)

    def submit(self, record):
        """Returns list of records to feed into the live buffer (0 or 1 items)."""
        s = slot_from_x(record[0])
        if s not in self.merge_map:
            return [record]
        buf = self.buffers.setdefault(s, [])
        buf.append(record)
        if len(buf) >= self.merge_map[s]:
            composite = self.aggregate(buf)
            self.buffers[s] = []
            return [composite]
        return []

    def flush_all(self):
        """Drain partial buffers at end of full data pass."""
        out = []
        for buf in self.buffers.values():
            if buf:
                out.append(self.aggregate(buf))
        self.buffers.clear()
        return out

    def aggregate(self, records):
        x = records[0][0]
        policy_dense = np.zeros(POLICY_DIM, dtype=np.float32)
        for r in records:
            policy_dense += densify_policy(r[2])
        policy_dense /= len(records)
        Y = np.mean(np.stack([r[3] for r in records]), axis=0).astype(np.float32)
        return (x, None, sparsify_policy(policy_dense), Y, 1.0, 1.0, records[0][6])


# ---------------------------------------------------------------------------
# lc0 V6 chunk source
# ---------------------------------------------------------------------------

def lc0_stream(chunks, cfg, stop_evt):
    parser = ChunkParser(
        chunks,
        workers=0,
        batch_size=1,
        draw_drop_rate=cfg.lc0_draw_drop,
        diff_focus_min=1.0,
        diff_focus_slope=0.0,
    )
    dummy    = np.zeros(64, dtype=np.int64)
    prev_pl  = -1
    game_ply = 0
    for rec in parser.inner.sequential_gen():
        if stop_evt.is_set():
            return
        (ver, ifmt, probs_bytes, planes_bytes,
         us_ooo, us_oo, them_ooo, them_oo, stm, rule50,
         inv, dep,
         root_q, best_q, root_d, best_d, root_m, best_m, pl,
         result_q, result_d, pq, pd, pm, oq, od, om,
         visits, pidx, bidx, r1, r2, r3, r4) = V6_STRUC.unpack(rec)

        if pl > prev_pl:
            game_ply = 0
        else:
            game_ply += 1
        prev_pl = pl

        if random.random() >= min(1.0, 0.05 + 0.95 * game_ply / 15.0):
            continue

        probs = np.frombuffer(probs_bytes, dtype=np.float32)
        pol4288, _ = get_policy_vector(probs, stm, us_ooo, us_oo, dummy)
        policy_1858 = pol4288[SL_IDX]
        s = policy_1858.sum()
        if s <= 0:
            continue
        policy_1858 = policy_1858 / s

        result_wdl = np.array([
            0.5 * (1 - result_d + result_q),
            result_d,
            0.5 * (1 - result_d - result_q),
        ], dtype=np.float32)
        
        search_wdl = np.array([
            0.5 * (1 - best_d + best_q),
            best_d,
            0.5 * (1 - best_d - best_q),
        ], dtype=np.float32)

        Y = (cfg.z_blend * result_wdl + (1.0 - cfg.z_blend) * search_wdl).astype(np.float32)

        xc0h = v6_planes_to_xc0h(
            planes_bytes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50, cfg.k,
        )
        yield (np.asarray(xc0h, dtype=np.int16), None,
               sparsify_policy(policy_1858), Y, 1.0, 1.0, 'lc0')


def lc0_chunk_worker(worker_id, chunks, cfg, record_q, log_q, stop_evt):
    random.shuffle(chunks)
    n = 0
    for record in lc0_stream(chunks, cfg, stop_evt):
        if stop_evt.is_set():
            break
        record_q.put(('record', record))
        n += 1
        if n % 50_000 == 0:
            log_q.put({'name': f'lc0-{worker_id}', 'kind': 'progress', 'n': n})
    log_q.put({'name': f'lc0-{worker_id}', 'kind': 'done', 'n': n})
    os._exit(0)


# ---------------------------------------------------------------------------
# xc0 source helpers
# ---------------------------------------------------------------------------

def load_log(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        return pickle.load(f)


def load_xc0_games(run_tag):
    rd = os.path.join(SP_DIR, run_tag)
    analyze_path = os.path.join(rd, ANALYZE_PKL)
    if not os.path.exists(analyze_path):
        print(f"[xc0] {run_tag}: no analyze pkl, skipping", flush=True)
        return [], {}
    games = [g for g in load_game_index(rd) if isinstance(g.get('pkl_file'), str)]
    with open(analyze_path, 'rb') as fh:
        prev = pickle.load(fh)
    sf_by_gid = {
        gid: sub
        for gid, sub in prev['df_all'].groupby('game_id', sort=False)
    }
    random.shuffle(games)
    print(f"[xc0] {run_tag}: {len(games):,} games", flush=True)
    return games, sf_by_gid


def detect_has_wdl(games):
    for g in games:
        if not os.path.exists(g['pkl_file']):
            continue
        log = load_log(g['pkl_file'])
        tree_data = log.get('tree_search_data', {})
        if tree_data:
            return any(n.get('best_wdl') is not None for n in tree_data.values())
    return False


def game_cpl_mean(sf_df):
    if sf_df is None or not len(sf_df):
        return None
    return float(np.clip(sf_df['loss'], -1000, 1000).mean())


def collar_is_noisy(sf_df, result, cfg):
    if sf_df is None or not len(sf_df) or 'best_absolute' not in sf_df.columns:
        return False
    evals = sf_df.sort_values('move_num')['best_absolute'].to_numpy()
    holder = None
    holder_start = None
    run_sign = 0
    run_count = 0
    for i, ev in enumerate(evals):
        if ev > cfg.collar_threshold_cp:
            sign = 1
        elif ev < -cfg.collar_threshold_cp:
            sign = -1
        else:
            sign = 0
        if sign != 0 and sign == run_sign:
            run_count += 1
        elif sign != 0:
            run_sign, run_count = sign, 1
        else:
            run_sign, run_count = 0, 0
        if run_count >= cfg.collar_n_consec:
            holder = 'white' if run_sign == 1 else 'black'
            holder_start = i
            break
    if holder is None:
        return False
    if float(result) != (1.0 if holder == 'white' else -1.0):
        return True
    below = 0
    for ev in evals[holder_start:]:
        pov = ev if holder == 'white' else -ev
        if pov < 0:
            below += 1
            if below >= cfg.collar_reversal_n_consec:
                return True
        else:
            below = 0
    return False


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


def make_xc0_record(board, node, cms, is_white, result, cfg, has_wdl):
    """Build a sparse 7-tuple from a node's candidate move list. Returns None
    if the node is missing WDL on a WDL-head run."""
    visits_map = {c['uci']: c['visits'] for c in cms}
    for u in board.legal_moves():
        if u not in visits_map:
            visits_map[u] = 1
    ucis   = list(visits_map.keys())
    counts = np.array([visits_map[u] for u in ucis], dtype=np.float32)
    s      = counts.sum()
    if s <= 0:
        return None
    pi = counts / s

    indices = board.moves_to_indices(ucis)
    policy  = np.zeros(N_1858, dtype=np.float32)
    for ci, prob in zip(indices, pi):
        policy[ci] += prob

    wdl_node = node.get('best_wdl')
    if wdl_node is not None:
        search_wdl = np.array(wdl_node, dtype=np.float32)[
            [0, 1, 2] if is_white else [2, 1, 0]]
    elif has_wdl:
        return None
    else:
        search_wdl = to_wdl(node.get('Q_stm', 0.0))

    z = (1 if is_white else -1) if result > 0 else (
        (-1 if is_white else 1) if result < 0 else 0)
    Y = (cfg.z_blend * to_wdl(z) + (1.0 - cfg.z_blend) * search_wdl).astype(np.float32)
    xc0h = np.asarray(board.history_tokens(cfg.k), dtype=np.int16)
    return (xc0h, None, sparsify_policy(policy), Y, 1.0, 1.0, 'xc0')


def iter_xc0_game(game, sf_df, run_tag, has_wdl, cfg):
    """Yields ('record', 7-tuple) or ('sim_req', dict) per eligible ply."""
    if not os.path.exists(game['pkl_file']):
        return
    log      = load_log(game['pkl_file'])
    scenario = log.get('scenario') or game.get('scenario', '')
    if scenario in cfg.skip_scenarios:
        return

    moves = log.get('moves_played', [])
    if len(moves) < 10:
        return

    start_fen = log.get('start_fen')
    tree_data = {int(k): v for k, v in log.get('tree_search_data', {}).items()}
    result    = log.get('result', 0)
    vs_sf     = log.get('vs_stockfish', False)
    sf_color  = log.get('stockfish_color', None)

    sf_rows, sf_by_ply = None, {}
    if sf_df is not None and len(sf_df):
        sf_rows, sf_by_ply = align_sf(moves, sf_df)

    if scenario in cfg.no_warm_scenarios:
        board = pf.Board(start_fen)
    else:
        _, board = reconcile_game_boards(start_fen, log.get('history_uci'), moves)

    uci_path = []

    for ply, move_played in enumerate(moves):
        is_white  = board.white_to_move()
        is_sf_ply = vs_sf and sf_color is not None and (
            (is_white and sf_color) or (not is_white and not sf_color))

        node = tree_data.get(ply)
        cms  = node.get('candidate_moves') if node else None

        if is_sf_ply:
            if cms:
                top_xc0 = max(cms, key=lambda c: c['visits'])['uci']
                if top_xc0 == move_played:
                    rec = make_xc0_record(
                        board, node, cms, is_white, result, cfg, has_wdl
                    )
                    
                    if rec is not None:
                        yield ('record', rec)
                else:
                    xc0h = np.asarray(board.history_tokens(cfg.k), dtype=np.int16)
                    z    = (1 if is_white else -1) if result > 0 else (
                           (-1 if is_white else 1) if result < 0 else 0)
                    yield ('sim_req', {
                        'uci_path': list(uci_path),
                        'xc0h': xc0h,
                        'z': z,
                        'source': 'lc0d_sf',
                    })
        elif cms:
            idx  = sf_by_ply.get(ply)
            loss = sf_rows.iloc[idx]['loss'] if idx is not None else None
            if loss is not None and loss <= cfg.move_cpl_max:
                rec = make_xc0_record(board, node, cms, is_white, result, cfg, has_wdl)
                if rec is not None:
                    yield ('record', rec)
            else:
                xc0h = np.asarray(board.history_tokens(cfg.k), dtype=np.int16)
                z    = (1 if is_white else -1) if result > 0 else (
                       (-1 if is_white else 1) if result < 0 else 0)
                yield ('sim_req', {
                    'uci_path': list(uci_path),
                    'xc0h': xc0h,
                    'z': z,
                    'source': 'lc0d',
                })

        board.push_uci(move_played)
        uci_path.append(move_played)


def xc0_worker(worker_id, run_tags, cfg, record_q, sim_req_q, log_q, stop_evt):
    n_rec = n_sim = 0
    for run_tag in run_tags:
        if stop_evt.is_set():
            break
        games, sf_by_gid = load_xc0_games(run_tag)
        has_wdl = detect_has_wdl(games)
        print(f"[xc0-{worker_id}] {run_tag}: "
              f"{'WDL' if has_wdl else 'scalar-Q'}", flush=True)

        for game in games:
            if stop_evt.is_set():
                break
            if game.get('scenario') in cfg.skip_scenarios:
                continue
            sf_df    = sf_by_gid.get(game['game_id'])
            game_cpl = game_cpl_mean(sf_df)
            if game_cpl is not None and game_cpl > cfg.game_cpl_max:
                continue
            if collar_is_noisy(sf_df, game.get('result', 0), cfg):
                continue

            for tag, payload in iter_xc0_game(game, sf_df, run_tag, has_wdl, cfg):
                if tag == 'record':
                    record_q.put(('record', payload))
                    n_rec += 1
                else:
                    try:
                        sim_req_q.put_nowait(payload)
                        n_sim += 1
                    except queue.Full:
                        pass

            if (n_rec + n_sim) % 50_000 < 200:
                log_q.put({'name': f'xc0-{worker_id}', 'kind': 'progress',
                           'n_rec': n_rec, 'n_sim': n_sim})

    log_q.put({'name': f'xc0-{worker_id}', 'kind': 'done',
               'n_rec': n_rec, 'n_sim': n_sim})
    os._exit(0)


# ---------------------------------------------------------------------------
# lc0 sim worker
# ---------------------------------------------------------------------------

def lc0_sim_worker(worker_id, cfg, sim_req_q, sim_result_q, log_q):
    """Pull position requests, run lc0 MCTS, push result records.

    Uses lc0 TRT engine + MCTSTree at sims_floor/ceiling from cfg.
    No effort scaling. Root visit distribution only (no PV chain).
    """
    lc0_model     = os.getenv('LC0_DISTILL_MODEL', '')
    lc0_trt_cache = os.getenv('LC0_DISTILL_TRT_CACHE', '')
    if not lc0_model or not lc0_trt_cache:
        raise RuntimeError(f"[lc0-sim-{worker_id}] LC0_DISTILL_MODEL / "
                           f"LC0_DISTILL_TRT_CACHE not set")

    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    lc0_onnx = os.path.join(lc0_trt_cache, f'{lc0_model}.onnx')
    batch    = getattr(cfg, 'lc0_distill_batch', 64)
    sess     = make_lc0_trt_session(
        lc0_onnx, lc0_model, lc0_trt_cache,
        opt_batch=batch,
        max_batch=batch * 2,
    )
    if sess.get_providers()[0] != 'TensorrtExecutionProvider':
        raise RuntimeError(f"[lc0-sim-{worker_id}] TRT failed")

    # TODO: wire MCTSTree with lc0 session at sims_floor/ceiling.
    # Placeholder: single-inference fallback using Lc0Batcher interface.
    from chessbot.lc0_utils import Lc0Board
    import chess

    input_name   = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    n = 0
    while True:
        try:
            req = sim_req_q.get(timeout=2.0)
        except queue.Empty:
            continue
        if req is None:
            break

        board = chess.Board()
        for uci in req['uci_path']:
            board.push_uci(uci)

        lb      = Lc0Board(board)
        feats   = lb.features().astype(np.float32)[np.newaxis]
        feats[0, 109] /= 99.0
        outs    = sess.run(output_names, {input_name: feats})
        logits  = outs[0][0]
        wdl_raw = outs[1][0]

        ti      = int(feats[0, 104, 0, 0]) | int(feats[0, 105, 0, 0])
        ti     += 2 * int(feats[0, 108, 0, 0])
        ucis    = [m.uci() for m in board.legal_moves]
        lc0_idx = np.array([UCI_TO_IDX[ti][u.rstrip('n')] for u in ucis], dtype=np.int32)
        xc0_idx = np.array(
            pf.Board(board.fen()).moves_to_indices(ucis), dtype=np.int32
        )
        policies = lc0_logits_to_xc0_batch([logits], [lc0_idx], [xc0_idx])
        policy   = policies[0]

        is_white   = board.turn
        search_wdl = np.array(wdl_raw, dtype=np.float32)[
            [0, 1, 2] if is_white else [2, 1, 0]]
        z = req['z']
        Y = (cfg.z_blend * to_wdl(z) + (1.0 - cfg.z_blend) * search_wdl).astype(np.float32)

        record = (req['xc0h'], None, sparsify_policy(policy), Y, 1.0, 1.0, req['source'])
        sim_result_q.put(record)
        n += 1
        if n % 10_000 == 0:
            log_q.put({'name': f'lc0sim-{worker_id}', 'kind': 'progress', 'n': n})

    log_q.put({'name': f'lc0sim-{worker_id}', 'kind': 'done', 'n': n})
    os._exit(0)


# ---------------------------------------------------------------------------
# parent coordinator
# ---------------------------------------------------------------------------

def resume_shard_id(out_dir, prefix):
    marker   = prefix + '_'
    existing = [f for f in os.listdir(out_dir)
                if f.startswith(marker) and f.endswith('.pkl.gz')]
    if not existing:
        return 0
    nums = []
    for name in existing:
        stem = name[:-len('.pkl.gz')]
        try:
            nums.append(int(stem.rsplit('_', 1)[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 0


def partition(items, n):
    return [items[i::n] for i in range(n)]


def run_coordinator(cfg, record_q, sim_req_q, sim_result_q, log_q,
                    n_data_workers, n_sim_workers):
    os.makedirs(cfg.out_dir, exist_ok=True)
    consolidator = OpeningConsolidator(cfg.npz_path)
    live_buf     = LiveBuffer(
        primary_dir=cfg.out_dir,
        capacity=cfg.live_buffer_capacity,
        shard_size=cfg.records_per_shard,
        counts=None,
    )

    def feed(record):
        for r in consolidator.submit(record):
            live_buf.append(r)
            live_buf.flush_all_ready()

    done_data = 0
    done_sim  = 0
    n_rec     = 0
    n_sim_in  = 0
    begin     = time.time()

    while done_data < n_data_workers or done_sim < n_sim_workers:
        # drain log queue first (non-blocking)
        while True:
            try:
                msg = log_q.get_nowait()
            except queue.Empty:
                break
            if msg['kind'] == 'done':
                name = msg['name']
                if name.startswith('lc0sim-'):
                    done_sim += 1
                    print(f"[{name}] done  n={msg['n']:,}", flush=True)
                else:
                    done_data += 1
                    print(f"[{name}] done  "
                          f"n_rec={msg.get('n_rec', msg.get('n', 0)):,}  "
                          f"n_sim={msg.get('n_sim', 0):,}", flush=True)
                    if done_data == n_data_workers:
                        for _ in range(n_sim_workers):
                            sim_req_q.put(None)
            elif msg['kind'] == 'progress':
                elapsed = time.time() - begin
                rate    = (n_rec + n_sim_in) / max(elapsed, 1e-9)
                print(f"[coord] n_rec={n_rec:,}  n_sim_in={n_sim_in:,}  "
                      f"buf={len(live_buf):,}  {rate:,.0f}/s", flush=True)

        # drain record queue
        drained = 0
        while drained < 2000:
            try:
                tag, payload = record_q.get_nowait()
                feed(payload)
                n_rec += 1
                drained += 1
            except queue.Empty:
                break

        # drain sim results
        drained = 0
        while drained < 500:
            try:
                record = sim_result_q.get_nowait()
                feed(record)
                n_sim_in += 1
                drained += 1
            except queue.Empty:
                break

        if drained == 0 and done_data == n_data_workers:
            # all data workers done; wait on sim results
            try:
                record = sim_result_q.get(timeout=1.0)
                feed(record)
                n_sim_in += 1
            except queue.Empty:
                pass

    # flush remaining partial consolidation buffers
    for r in consolidator.flush_all():
        live_buf.append(r)
    # write final partial shard regardless of capacity
    if live_buf.records:
        path = new_shard_path(cfg.out_dir)
        write_pkl_gz_shard(live_buf.records, path)
        live_buf.records.clear()

    elapsed = time.time() - begin
    print(f"\n[done]  n_rec={n_rec:,}  n_sim_in={n_sim_in:,}  "
          f"elapsed={elapsed:.0f}s  out={cfg.out_dir}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # pre-compile lc0 TRT engine in supervisor so workers never race
    lc0_model     = os.getenv('LC0_DISTILL_MODEL', '')
    lc0_trt_cache = os.getenv('LC0_DISTILL_TRT_CACHE', '')
    if lc0_model and lc0_trt_cache:
        from chessbot.lc0_utils import make_lc0_trt_session
        lc0_onnx = os.path.join(lc0_trt_cache, f'{lc0_model}.onnx')
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        print(f"[main] pre-compiling lc0 TRT engine ({lc0_model})...", flush=True)
        batch = getattr(cfg, 'lc0_distill_batch', 64)
        sess  = make_lc0_trt_session(
            lc0_onnx, lc0_model, lc0_trt_cache,
            opt_batch=batch,
            max_batch=batch * 2,
        )
        del sess
        print(f"[main] lc0 TRT engine ready", flush=True)

    # gather lc0 V6 chunks
    lc0_chunks = []
    lc0_dir    = getattr(cfg, 'lc0_dir', '')
    if lc0_dir and os.path.isdir(lc0_dir):
        for d in sorted(os.listdir(lc0_dir)):
            full = os.path.join(lc0_dir, d)
            if not os.path.isdir(full):
                continue
            lc0_chunks.extend(
                os.path.join(full, f) for f in os.listdir(full)
                if f.endswith('.gz') and not f.endswith('.tfrecord.gz')
            )
        print(f"[scan] {len(lc0_chunks):,} lc0 chunk files", flush=True)

    run_tags = list(cfg.run_tags)
    print(f"[scan] {len(run_tags)} xc0 run tags: {run_tags}", flush=True)

    ctx           = mp.get_context('spawn')
    record_q      = ctx.Queue(maxsize=8_000)
    sim_req_q     = ctx.Queue(maxsize=cfg.sim_req_q_maxsize)
    sim_result_q  = ctx.Queue(maxsize=4_000)
    log_q         = ctx.Queue()
    stop_evt      = ctx.Event()

    workers = []

    # xc0 workers: partition run_tags across max_concurrent_xc0 workers
    xc0_parts = partition(run_tags, cfg.max_concurrent_xc0)
    for i, part in enumerate(xc0_parts):
        if not part:
            continue
        p = ctx.Process(
            target=xc0_worker,
            args=(i, part, cfg, record_q, sim_req_q, log_q, stop_evt),
        )
        p.start()
        workers.append(('xc0', p))

    # lc0 chunk workers
    if lc0_chunks:
        n_lc0_chunk = getattr(cfg, 'n_lc0_chunk_workers', 3)
        for i, chunk_part in enumerate(partition(lc0_chunks, n_lc0_chunk)):
            if not chunk_part:
                continue
            p = ctx.Process(
                target=lc0_chunk_worker,
                args=(i, chunk_part, cfg, record_q, log_q, stop_evt),
            )
            p.start()
            workers.append(('lc0', p))

    # lc0 sim workers
    for i in range(cfg.n_lc0_sim_workers):
        p = ctx.Process(
            target=lc0_sim_worker,
            args=(i, cfg, sim_req_q, sim_result_q, log_q),
        )
        p.start()
        workers.append(('lc0sim', p))

    n_data_workers = sum(1 for kind, _ in workers if kind != 'lc0sim')
    n_sim_workers  = sum(1 for kind, _ in workers if kind == 'lc0sim')

    print(f"[main] {len(workers)} workers  "
          f"xc0={len(xc0_parts)}  "
          f"lc0_chunk={len(lc0_chunks)>0 and n_lc0_chunk or 0}  "
          f"lc0sim={n_sim_workers}", flush=True)

    run_coordinator(cfg, record_q, sim_req_q, sim_result_q, log_q,
                    n_data_workers, n_sim_workers)

    for _, p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    os._exit(0)


if __name__ == '__main__':
    mp.freeze_support()
    main()
