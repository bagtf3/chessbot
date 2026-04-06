import os, json, gzip, pathlib, time
from types import SimpleNamespace
from pathlib import Path
import uuid
import sys
import random
import subprocess
import threading
from collections import deque, defaultdict
from queue import Queue, Empty

import pickle
import chess, chess.engine

import pandas as pd
import numpy as np

from pyfastchess import Board

from chessbot import SF_LOC
from chessbot.utils import (
    score_cp_stm_pov, score_cp_white_pov, rnd, kl_divergence,
    batch_policy_metrics_from_priors, print_validation,
)

RS = "[rescore]"
ANALYZE_PKL = "analyze_results_combined.pkl"
EVICTION_WINDOW = 5000
EVICTION_MAX_SIZE = 80000


class SFCache:
    """
    Position-keyed cache for Stockfish analysis results.
    Key: short FEN (position + turn + castling + ep, no move clocks).
    Each entry stores the best move found and any other moves analyzed
    at that position, so repeated positions skip SF calls entirely.
    """

    def __init__(self, eviction_window=EVICTION_WINDOW, max_size=EVICTION_MAX_SIZE):
        self.data = {}
        self.eviction_window = eviction_window
        self.max_size = max_size

    def get(self, key):
        return self.data.get(key)

    def touch(self, key, game_num):
        entry = self.data.get(key)
        if entry is not None:
            entry['hits'] += 1
            entry['last_seen'] = game_num

    def set_best(self, key, uci, cp, abs_cp, wdl, game_num):
        if key not in self.data:
            self.data[key] = {
                'best': None, 'others': {}, 'hits': 0, 'last_seen': game_num
            }
        entry = self.data[key]
        if wdl is None and entry['best'] is not None:
            wdl = entry['best'][3]
        entry['best'] = (uci, cp, abs_cp, wdl)
        entry['hits'] += 1
        entry['last_seen'] = game_num

    def set_move(self, key, uci, cp, abs_cp, game_num):
        entry = self.data.get(key)
        if entry is not None:
            entry['others'][uci] = (cp, abs_cp)
            entry['last_seen'] = game_num

    def maybe_evict(self, game_num):
        if len(self.data) < self.max_size:
            return 0
        threshold = game_num - self.eviction_window
        stale = [k for k, v in self.data.items() if v['last_seen'] < threshold]
        for k in stale:
            del self.data[k]
        return len(stale)

    def stats(self):
        return {
            'size': len(self.data),
            'total_hits': sum(v['hits'] for v in self.data.values()),
        }


class SFRescoreThread:
    """
    Pure SF worker. Pulls (req_id, board, move) from req_q, runs analysis,
    pushes (req_id, result) to res_q. No cache, no game logic.
    move=None -> best-move analysis, returns type='best' result.
    move=Move -> single-move analysis, returns type='move' result.
    Multiple instances can share the same req_q/res_q for parallelism.
    """

    def __init__(self, req_q, res_q, cfg):
        self.req_q = req_q
        self.res_q = res_q
        self.depth = cfg.rescore_depth
        self.sf_config = {'Hash': 256, 'UCI_ShowWDL': True}
        self.stop_ev = threading.Event()
        self.t = None
        self.eng = None

    def start(self):
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def close(self):
        self.stop_ev.set()
        self.req_q.put(None)
        if self.t is not None:
            self.t.join()
        if self.eng is not None:
            self.eng.quit()
            self.eng = None

    def run(self):
        self.eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
        self.eng.configure(self.sf_config)
        limit = chess.engine.Limit(depth=self.depth)

        while not self.stop_ev.is_set():
            try:
                item = self.req_q.get(timeout=0.01)
            except Empty:
                continue
            if item is None:
                break
            req_id, board, move = item
            try:
                t0 = time.time()
                if move is None:
                    info = self.eng.analyse(board, limit, info=chess.engine.INFO_ALL)
                    elapsed = time.time() - t0
                    best = info['pv'][0]
                    wdl = info.get('wdl')
                    wdl_val = (
                        (wdl.relative.wins - wdl.relative.losses) / 1000.0
                        if wdl is not None else None
                    )
                    self.res_q.put((req_id, {
                        'type': 'best',
                        'best_uci': str(best),
                        'best_cp': score_cp_stm_pov(info['score']),
                        'best_abs': score_cp_white_pov(info['score'], clipped=False),
                        'wdl': wdl_val,
                        'elapsed': elapsed,
                    }))
                else:
                    info = self.eng.analyse(
                        board, limit,
                        root_moves=[move], info=chess.engine.INFO_ALL
                    )
                    elapsed = time.time() - t0
                    self.res_q.put((req_id, {
                        'type': 'move',
                        'move_uci': str(move),
                        'move_cp': score_cp_stm_pov(info['score']),
                        'move_abs': score_cp_white_pov(info['score'], clipped=False),
                        'elapsed': elapsed,
                    }))
            except Exception as e:
                self.res_q.put(('__error__', e))
                self.stop_ev.set()
                return


class Rescorer(object):

    def __init__(self, cfg, req_q, res_q, cache):
        self.config = cfg
        self.req_q = req_q
        self.res_q = res_q
        self.cache = cache

        self.training_data = []
        self.analyzed_results = []
        self.pending_metrics = []


        self.start_time = time.time()
        self.games_seen = set()
        self.games_processed = 0
        self.written_total = 0
        self.written_this_round = 0
        self.n_saved = 0
        self.tcpl = 0
        self.tbmr = 0

        self.n_sf_submitted = 0
        self.n_cache_hits = 0
        self.sf_call_time = 0.0
        self.sf_call_count = 0
        self.sf_compute_time = 0.0
        self.sf_compute_count = 0
        self.last_10_cpls = []
        self.last_10_bmrs = []

        zero_collar = lambda: {
            'games': 0, 'triggers': 0, 'positions': 0,
            'total_pos': 0, 'seen': 0,
        }
        self.collar_total = zero_collar()
        self.collar_window = zero_collar()
        self.collar_history = []

        zero_stop = lambda: {'n': 0, 'cpl': 0.0, 'bmr': 0.0, 'sims': 0.0}
        self.total_stop = {st: zero_stop() for st in ("full", "rsc", "jsd")}
        self.window_stop = {st: zero_stop() for st in ("full", "rsc", "jsd")}

        self.intake = deque()
        self.pending = {}
        self.req_map = {}
        self.req_counter = 0

        self.init_analyzer()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @property
    def training_data_size(self):
        return len(self.training_data)

    def next_req_id(self):
        self.req_counter += 1
        return self.req_counter

    def maybe_seed_cache(self, b_fast, mv, Q_stm, turn, repetitions):
        short_fen = b_fast.fen(include_counters=False)
        reps = 3 if repetitions[short_fen] >= 3 else 0
        hmc = b_fast.halfmove_clock()
        halfmoves = 0 if hmc < 45 else hmc
        cache_key = (short_fen, reps, halfmoves)
        if self.cache.get(cache_key) is None:
            best_cp = int(Q_stm * 1000)
            best_abs = best_cp if turn else -best_cp
            self.cache.set_best(cache_key, mv, best_cp, best_abs, None, self.games_processed)

    def submit(self, pkl_file):
        self.intake.append(pkl_file)

    def tick(self):
        while True:
            try:
                req_id, result = self.res_q.get_nowait()
            except Empty:
                break
            if req_id == '__error__':
                raise result
            self.handle_sf_result(req_id, result)

        if len(self.intake) > len(self.pending):
            intake_list = list(self.intake)
            random.shuffle(intake_list)
            self.intake = deque(intake_list)
        while self.intake and len(self.pending) < 20:
            self.start_game(self.intake.popleft())

        done = [gid for gid, g in self.pending.items() if not g['waiting']]
        for gid in done:
            self.finalize_game(self.pending.pop(gid))

        if self.games_processed > 0 and self.games_processed % 100 == 0:
            self.cache.maybe_evict(self.games_processed)

    def init_analyzer(self):
        run_dir = self.config.run_dir
        idx_path = os.path.join(run_dir, "game_index.json")

        # first merge any existing analysis
        _ = combine_analysis_staging(run_dir)

        # seen games from existing analyze pkl
        self.games_seen = set()
        analyze_pkl_path = os.path.join(run_dir, ANALYZE_PKL)

        if os.path.exists(analyze_pkl_path):
            with open(analyze_pkl_path, "rb") as f:
                combined = pickle.load(f)
            if "df_means" in combined and combined["df_means"] is not None:
                self.games_seen = set(
                    combined["df_means"]["game_id"].astype(str).tolist()
                )

    def get_unprocessed(self):
        run_dir = self.config.run_dir
        idx_path = os.path.join(run_dir, "game_index.json")
        idx = load_game_index(idx_path)
        entries = [idx] if isinstance(idx, dict) else idx

        unprocessed = deque()
        for rec in entries:
            gid = str(rec.get("game_id"))
            pkl_path = rec.get("pkl_file")
            if not pkl_path or not os.path.exists(pkl_path):
                continue
            if gid in self.games_seen:
                continue
            unprocessed.append(rec)
        
        if len(unprocessed):
            n = len(unprocessed)
            print(f"[rescore] {n} unprocessed game(s) currently in queue")
        
        return unprocessed

    def reset_writer(self):
        self.written_this_round = 0
    
    def make_policy_example(self, board, ucis, visits):
        indices = board.moves_to_indices(ucis)
        policy = np.zeros(64 * 67, dtype=np.float32)
        s = sum(visits)
        pi = np.array([v / s for v in visits], dtype=np.float32)
        eps = self.config.uniform_eps
        uniform_mass = 1 / len(ucis) if len(ucis) else 0.0
        pi = eps * uniform_mass + (1.0 - eps) * pi
        pi = np.clip(pi, 0.0, self.config.prior_clip_max)
        pi = pi / pi.sum()
        for idx, p in zip(indices, pi):
            policy[idx] += p
        x = board.encode_64_tokens()
        mask = board.legal_move_mask()
        return x, mask, policy

    def append_flat_policy_example(self, board, ucis, visits, Y, vwht, pwht):
        x, mask, policy = self.make_policy_example(board, ucis, visits)
        self.training_data.append((x, mask, policy, Y, vwht, pwht))
    
    def write_training_data_pkl(self, size=None, randomize=True):
        cfg = self.config
        out_dir = pathlib.Path(cfg.pending_training_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # clear stale pkl shards from any previous failed retrain
        deleted = []
        for p in out_dir.iterdir():
            if p.is_file() and p.suffix == ".pkl":
                p.unlink()
                deleted.append(p.name)

        if deleted:
            print(f"{RS} deleted {len(deleted)} stale pkl files in {out_dir}")

        if randomize:
            random.shuffle(self.training_data)

        if size is None:
            size = cfg.get_retrain_size

        chunk = self.training_data[:size]
        remainder = self.training_data[size:]

        filename = f"{int(time.time())}-{uuid.uuid4().hex}.pkl"
        out_path = out_dir / filename

        with open(out_path, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.training_data = remainder
        self.written_this_round += len(chunk)
        self.written_total += len(chunk)

    def training_data_from_sf(self, board, mv, cm, Y, is_draw):
        cfg = self.config

        rows = [(c["uci"], c["visits"]) for c in cm]
        total_visits = sum([n for _, n in rows]) if rows else 0

        if rows and total_visits > 10:
            top_uci = max(rows, key=lambda x: x[1])[0]
            pairs = encourage_best_move(rows, top_uci, mv, board.legal_moves())
            ucis = [v[0] for v in pairs]
            visits = [v[1] for v in pairs]
        else:
            raw = make_fake_visits(mv, board.legal_moves(), ratio_best=60)
            ucis = [x[0] for x in raw]
            visits = [int(x[1]) for x in raw]

        priors_map = {c["uci"]: c["P"] for c in cm}
        priors = [priors_map.get(u, 0.0) for u in ucis]
        kl = kl_divergence(priors, visits)

        vwht = value_weight_for_game(cfg, is_draw)
        pwht = 1.0
        if kl > cfg.KL_boost_threshold:
            pwht *= cfg.KL_weight_boost

        self.append_flat_policy_example(board, ucis, visits, Y, vwht, pwht)

    def start_game(self, pkl_file):
        path = str(pkl_file)
        if path.lower().endswith('.pkl.gz'):
            with gzip.open(path, 'rb') as f:
                game_data = pickle.load(f)
        elif path.lower().endswith('.pkl'):
            with open(path, 'rb') as f:
                game_data = pickle.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)

        gid = str(game_data.get('game_id'))
        if gid in self.games_seen:
            return

        cfg = self.config
        board_ch = chess.Board(game_data['start_fen'])
        b_fast = Board(game_data['start_fen'])

        tree_data = game_data.get('tree_search_data', {})
        vs_stockfish = game_data.get('vs_stockfish', False)
        sf_color = game_data.get('stockfish_is_white')
        game_sf_depth = game_data.get('sf_depth')
        result = game_data['result']
        is_draw = (result == 0) or (result == 0.0)

        skip_all_training = False
        if not cfg.train_on_validation:
            if 'validation' in game_data.get('scenario', '').lower():
                skip_all_training = True

        game_cfg_keys = (
            'train_on_stockfish', 'z_mix',
            'KL_weight_boost', 'KL_boost_threshold',
            'rescore_equiv_range', 'rescore_blunder_cp_loser',
            'rescore_blunder_cp_winner', 'rescore_inaccuracy_cp',
            'uniform_eps', 'prior_clip_max',
            'collar_threshold_cp', 'collar_n_consec', 'collar_reset_cp',
            'collar_rescore_dry_run', 'rescore_analyze_batch',
            'draw_value_scale',
        )
        game_state = {
            'gid': gid,
            'game_data': game_data,
            'cfg': SimpleNamespace(**{k: getattr(cfg, k) for k in game_cfg_keys}),
            'ply_states': [],
            'waiting': set(),
        }

        repetitions = defaultdict(int)
        repetitions[b_fast.fen(include_counters=False)] += 1
        for i, mv in enumerate(game_data.get('moves_played', [])):
            move_ch = chess.Move.from_uci(mv)
            is_sf_move = vs_stockfish and (board_ch.turn == sf_color)
            turn = board_ch.turn

            if is_sf_move and ((not cfg.train_on_stockfish) or skip_all_training):
                if not skip_all_training and game_sf_depth and game_sf_depth >= cfg.rescore_depth:
                    tr_sf = tree_data.get(i, tree_data.get(str(i), {}))
                    Q_stm = tr_sf.get('Q_stm')
                    if Q_stm is not None:
                        self.maybe_seed_cache(b_fast, mv, Q_stm, turn, repetitions)
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                repetitions[b_fast.fen(include_counters=False)] += 1
                continue

            tr = tree_data.get(i, tree_data.get(str(i), {}))
            cm = tr.get('candidate_moves', [])
            if not cm:
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                repetitions[b_fast.fen(include_counters=False)] += 1
                continue

            Z_stm = result if turn else -result
            if 'Q_stm' in tr:
                Q = tr['Q_stm']
            else:
                this_q = tr.get('best_Q', tr.get('visit_weighted_Q'))
                Q = this_q if turn else -this_q

            Y_init = np.clip(cfg.z_mix * Z_stm + (1.0 - cfg.z_mix) * Q, -1.0, 1.0)
            if Y_init != Y_init:
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                repetitions[b_fast.fen(include_counters=False)] += 1
                continue

            if is_sf_move:
                if game_sf_depth and game_sf_depth >= cfg.rescore_depth:
                    self.maybe_seed_cache(b_fast, mv, Q, turn, repetitions)
                self.training_data_from_sf(b_fast, mv, cm, Y_init, is_draw)
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                repetitions[b_fast.fen(include_counters=False)] += 1
                continue

            visits = [(c['uci'], max(1, c['visits'])) for c in cm]
            visits = sorted(visits, key=lambda x: x[1], reverse=True)
            if not visits:
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                repetitions[b_fast.fen(include_counters=False)] += 1
                continue

            sel_method = tr.get('selection_method')
            xc0_move = tr.get('xc0_move')
            xerces_uci = mv
            if xc0_move and xc0_move != mv:
                xerces_uci = xc0_move
            elif not sel_method:
                xerces_uci = visits[0][0]

            top_uci = visits[0][0]
            if xerces_uci != top_uci:
                vmap = {u: n for u, n in visits}
                top_n = vmap.get(top_uci, 1)
                xc0_n = vmap.get(xerces_uci, 1)
                vmap[top_uci] = max(1, xc0_n)
                vmap[xerces_uci] = max(1, top_n)
                visits = sorted(vmap.items(), key=lambda x: x[1], reverse=True)

            lms = b_fast.legal_moves()
            idx_map = dict(zip(lms, b_fast.moves_to_indices(lms)))
            x = b_fast.encode_64_tokens()
            mask = b_fast.legal_move_mask()
            short_fen = b_fast.fen(include_counters=False)
            reps = 3 if repetitions[short_fen] >= 3 else 0
            hmc = b_fast.halfmove_clock()
            halfmoves = 0 if hmc < 45 else hmc
            cache_key = (short_fen, reps, halfmoves)

            ply = {
                'ply_idx': i,
                'mv': mv,
                'turn': bool(turn),
                'tr': tr,
                'cm': cm,
                'Z_stm': Z_stm,
                'Q': Q,
                'visits': visits,
                'lms': lms,
                'x': x,
                'mask': mask,
                'idx_map': idx_map,
                'skip_training': skip_all_training,
                'xerces_uci': xerces_uci,
                'board_ch': board_ch.copy(),
                'cache_key': cache_key,
                'best_uci': None,
                'best_cp': None,
                'best_abs': None,
                'sf_wdl': None,
                'played_cp': None,
                'played_abs': None,
                'resolved': False,
            }

            entry = self.cache.get(cache_key)
            if entry and entry['best'] is not None:
                best_uci, best_cp, best_abs, *_wdl = entry['best']
                best_wdl = _wdl[0] if _wdl else None
                ply['best_uci'] = best_uci
                ply['best_cp'] = best_cp
                ply['best_abs'] = best_abs
                ply['sf_wdl'] = best_wdl
                self.cache.touch(cache_key, self.games_processed)
                self.n_cache_hits += 1

                if xerces_uci == best_uci:
                    ply['played_cp'] = best_cp
                    ply['played_abs'] = best_abs
                    ply['resolved'] = True
                elif xerces_uci in entry['others']:
                    played_cp, played_abs = entry['others'][xerces_uci]
                    ply['played_cp'] = played_cp
                    ply['played_abs'] = played_abs
                    ply['resolved'] = True
                else:
                    rid = self.next_req_id()
                    self.req_map[rid] = (gid, i, 'move', time.time())
                    game_state['waiting'].add(rid)
                    cmove = chess.Move.from_uci(xerces_uci)
                    self.req_q.put((rid, board_ch.copy(), cmove))
                    self.n_sf_submitted += 1
            else:
                rid = self.next_req_id()
                self.req_map[rid] = (gid, i, 'best', time.time())
                game_state['waiting'].add(rid)
                self.req_q.put((rid, board_ch.copy(), None))
                self.n_sf_submitted += 1

            game_state['ply_states'].append(ply)
            board_ch.push(move_ch)
            b_fast.push_uci(mv)
            repetitions[b_fast.fen(include_counters=False)] += 1

        self.pending[gid] = game_state
        if not game_state['waiting']:
            self.finalize_game(self.pending.pop(gid))

    def handle_sf_result(self, req_id, result):
        if req_id not in self.req_map:
            return
        
        gid, ply_idx, req_type, t_sent = self.req_map.pop(req_id)
        self.sf_call_time += time.time() - t_sent
        self.sf_call_count += 1
        elapsed = result.get('elapsed', 0.0)
        self.sf_compute_time += elapsed
        self.sf_compute_count += 1
        if gid not in self.pending:
            return

        game = self.pending[gid]
        game['waiting'].discard(req_id)
        ply = next((p for p in game['ply_states'] if p['ply_idx'] == ply_idx), None)
        if ply is None:
            return

        key = ply['cache_key']

        if req_type == 'best':
            best_uci = result['best_uci']
            best_cp = result['best_cp']
            best_abs = result['best_abs']
            wdl = result['wdl']
            self.cache.set_best(
                key, best_uci, best_cp, best_abs, wdl,
                self.games_processed
            )

            ply['best_uci'] = best_uci
            ply['best_cp'] = best_cp
            ply['best_abs'] = best_abs
            ply['sf_wdl'] = wdl

            xerces_uci = ply['xerces_uci']
            entry = self.cache.get(key)
            if xerces_uci == best_uci:
                ply['played_cp'] = best_cp
                ply['played_abs'] = best_abs
                ply['resolved'] = True
            elif entry and xerces_uci in entry['others']:
                played_cp, played_abs = entry['others'][xerces_uci]
                ply['played_cp'] = played_cp
                ply['played_abs'] = played_abs
                ply['resolved'] = True
            else:
                rid = self.next_req_id()
                self.req_map[rid] = (gid, ply_idx, 'move', time.time())
                game['waiting'].add(rid)
                self.req_q.put(
                    (rid, ply['board_ch'], chess.Move.from_uci(xerces_uci))
                )
                self.n_sf_submitted += 1

        elif req_type == 'move':
            played_cp = result['move_cp']
            played_abs = result['move_abs']
            played_uci = result['move_uci']

            entry = self.cache.get(key)
            if entry and entry['best'] is not None:
                old_best_uci, old_best_cp, old_best_abs, *_wdl = entry['best']
                old_best_wdl = _wdl[0] if _wdl else None
                if played_cp > old_best_cp:
                    # xerces move is actually stronger — promote it to best
                    self.cache.set_best(
                        key, played_uci, played_cp, played_abs,
                        old_best_wdl, self.games_processed
                    )
                    self.cache.set_move(
                        key, old_best_uci, old_best_cp, old_best_abs,
                        self.games_processed
                    )
                    ply['best_uci'] = played_uci
                    ply['best_cp'] = played_cp
                    ply['best_abs'] = played_abs
                else:
                    self.cache.set_move(
                        key, played_uci, played_cp, played_abs,
                        self.games_processed
                    )
            else:
                self.cache.set_move(
                    key, played_uci, played_cp, played_abs,
                    self.games_processed
                )

            ply['played_cp'] = played_cp
            ply['played_abs'] = played_abs
            ply['resolved'] = True

    def finalize_game(self, game_state):
        cfg = game_state['cfg']
        game_data = game_state['game_data']
        gid = game_state['gid']
        result = game_data['result']
        is_draw = (result == 0) or (result == 0.0)

        KL_coef = cfg.KL_weight_boost
        do_KL_boost = (KL_coef > 0) and (KL_coef != 1.0)
        EQUIV = cfg.rescore_equiv_range

        cpl_s = cpl_w = cpl_b = 0.0
        nw = nb = 0
        rows = []
        eval_trace = []
        pending = []
        pending_meta = []

        for ply in sorted(game_state['ply_states'], key=lambda p: p['ply_idx']):
            i = ply['ply_idx']
            mv = ply['mv']
            turn = ply['turn']
            tr = ply['tr']
            cm = ply['cm']
            Z_stm = ply['Z_stm']
            Q = ply['Q']
            xerces_uci = ply['xerces_uci']
            best_uci_raw = ply['best_uci']
            best_cp = ply['best_cp']
            best_abs = ply['best_abs']
            sf_wdl = ply['sf_wdl']
            played_cp = ply['played_cp']
            played_abs = ply.get('played_abs', best_abs)
            visits = list(ply['visits'])

            best_uci = best_uci_raw
            delta = best_cp - played_cp
            if abs(delta) <= EQUIV:
                delta = 0
            
            elif delta <= -EQUIV:
                # xerces found a notably better move than SF
                best_uci = xerces_uci
                best_cp = played_cp
                best_abs = played_abs

            loss_this = delta
            cpl_s += loss_this
            if turn:
                cpl_w += loss_this
                nw += 1
            else:
                cpl_b += loss_this
                nb += 1

            missed_mate = (best_cp >= 1200) and (played_cp >= 500)
            if missed_mate:
                loss_this = min(200, loss_this)

            eval_trace.append((i, best_abs))
            rows.append([
                i, mv, xerces_uci, best_uci,
                best_cp, loss_this, played_cp,
                best_abs, played_abs,
                turn, loss_this,
                tr.get('stop_reason', ''), tr.get('sims', 0),
            ])

            if ply['skip_training']:
                continue

            lms = ply['lms']
            blunder_cp = cfg.rescore_blunder_cp_loser
            if Z_stm > 0.0:
                blunder_cp = cfg.rescore_blunder_cp_winner

            kl_eligible = False
            xc0_uci = visits[0][0]
            xc0_n = visits[0][1]
            
            # ensure all legal moves have at least 1 visit
            vmap = {u: max(1, int(v)) for u, v in visits}
            for m in lms:
                if m not in vmap:
                    vmap[m] = 1
            visits = sorted(vmap.items(), key=lambda x: x[1], reverse=True)

            is_true_blunder = not (played_cp > 350 and Z_stm > 0)

            # visit correction based on move quality
            if loss_this <= EQUIV:
                kl_eligible = True

                # respect SF best move, set a modest floor
                vmap[best_uci] = max(vmap.get(best_uci, 1), max(1, xc0_n // 4))
            
            elif loss_this <= cfg.rescore_inaccuracy_cp or missed_mate:
                vmap[best_uci] = max(vmap.get(best_uci, 1), max(1, xc0_n // 2))

            elif loss_this < blunder_cp:
                vmap[best_uci] = max(vmap.get(best_uci, 1), xc0_n)
                if is_true_blunder:
                    vmap[xc0_uci] = max(1, xc0_n // 2)

            else:
                vmap[best_uci] = xc0_n
                if is_true_blunder:
                    vmap[xc0_uci] = max(1, xc0_n // 4)

            # re-sort after any adjustments
            visits = sorted(vmap.items(), key=lambda x: x[1], reverse=True)
            if not visits or sum(v[1] for v in visits) <= 0:
                print(f"[rescore] visits invalid; skipping move_idx={i} played={mv}")
                continue

            mvs = [v[0] for v in visits]
            vis = [v[1] for v in visits]
            priors_map = {c['uci']: c['P'] for c in cm}
            priors = [priors_map.get(u, 0.0) for u in mvs]

            vwht = value_weight_for_game(cfg, is_draw)
            pwht = 1.0
            if kl_eligible and do_KL_boost:
                kl = kl_divergence(priors, vis)
                if kl >= cfg.KL_boost_threshold:
                    pwht *= KL_coef

            # build policy from precomputed board state
            idx_map = ply['idx_map']
            indices = [idx_map[u] for u in mvs]
            policy = np.zeros(64 * 67, dtype=np.float32)
            s = sum(vis)
            pi = np.array([v / s for v in vis], dtype=np.float32)
            eps = cfg.uniform_eps
            uni = 1.0 / len(mvs) if mvs else 0.0
            pi = eps * uni + (1.0 - eps) * pi
            pi = np.clip(pi, 0.0, cfg.prior_clip_max)
            pi = pi / pi.sum()
            for idx, p in zip(indices, pi):
                policy[idx] += p

            pending.append((ply['x'], ply['mask'], policy, Q, turn, i, vwht, pwht))
            pending_meta.append({
                'stm': turn,
                'nn_value': tr.get('nn_value'),
                'nn_raw_priors': tr.get('nn_raw_priors', []),
                'mass_on_legal': tr.get('nn_mass_on_legal'),
                'sf_cp': best_cp,
                'sf_wdl': sf_wdl,
                'candidate_visits': list(zip(mvs, vis)),
                'result_z_stm': Z_stm,
            })

        eff_z_by_ply, n_triggers = collar_z_map(
            eval_trace, result,
            cfg.collar_threshold_cp, cfg.collar_n_consec, cfg.collar_reset_cp,
        )
        n_diff = 0
        for (x, mask, policy, Q, is_white, ply_i, vwht, pwht), meta in zip(
            pending, pending_meta
        ):
            z_orig = result if is_white else -result
            eff_z_white = eff_z_by_ply.get(ply_i, result)
            z_eff = eff_z_white if is_white else -eff_z_white
            if eff_z_white != result:
                n_diff += 1

            z = z_orig if cfg.collar_rescore_dry_run else z_eff
            Y = np.clip(cfg.z_mix * z + (1.0 - cfg.z_mix) * Q, -1.0, 1.0)
            self.training_data.append((x, mask, policy, Y, vwht, pwht))
            self.pending_metrics.append({**meta, 'target_y': float(Y)})
        self.accumulate_collar_stats(n_triggers, n_diff, len(pending))

        cols = [
            'move_num', 'played_move', 'most_visited_move', 'best_move',
            'best_cp', 'delta', 'played_cp',
            'best_absolute', 'played_absolute', 'stm', 'loss', 'stop_reason', 'sims',
        ]
        out_df = pd.DataFrame(rows, columns=cols)
        out_df['played_best_move'] = out_df['delta'] <= 0

        mask_w = out_df['stm'] == True
        mask_b = out_df['stm'] == False
        overall_bmr = out_df['played_best_move'].mean() if len(out_df) else np.nan
        white_bmr = out_df.loc[mask_w, 'played_best_move'].mean() if mask_w.any() else np.nan
        black_bmr = out_df.loc[mask_b, 'played_best_move'].mean() if mask_b.any() else np.nan

        out = {
            'plies': nw + nb,
            'overall_cpl': rnd(cpl_s / (nw + nb), 3) if (nw + nb) else np.nan,
            'white_cpl': rnd(cpl_w / nw, 3) if nw else np.nan,
            'black_cpl': rnd(cpl_b / nb, 3) if nb else np.nan,
            'overall_best_move_rate': overall_bmr,
            'best_move_rate_white': white_bmr,
            'best_move_rate_black': black_bmr,
        }
        for key in ['game_id', 'scenario', 'stockfish_color', 'ts']:
            val = game_data.get(key)
            if key == 'ts':
                val = int(val)
            out[key] = val
            out_df[key] = val

        stop_stats = {}
        for st in ('full', 'rsc', 'jsd'):
            mask_st = out_df['stop_reason'] == st
            n_st = mask_st.sum()
            stop_stats[st] = {
                'n': int(n_st),
                'cpl': rnd(out_df.loc[mask_st, 'delta'].mean(), 3) if n_st else np.nan,
                'bmr': (
                    out_df.loc[mask_st, 'played_best_move'].mean() if n_st else np.nan
                ),
                'sims': (
                    float(out_df.loc[mask_st, 'sims'].mean()) if n_st else float('nan')
                ),
            }
        out['stop_stats'] = stop_stats
        out['df'] = out_df

        self.analyzed_results.append(out)
        self.games_processed += 1
        self.games_seen.add(gid)
        self.accumulate_stop_stats(stop_stats)
        if self.games_processed % 100 == 0:
            self.print_stop_stats()
        if len(self.analyzed_results) >= cfg.rescore_analyze_batch:
            self.push_analyzed(report=True)


    def accumulate_collar_stats(self, n_triggers, n_diff, n_total):
        has_collar = n_triggers > 0
        for acc in (self.collar_total, self.collar_window):
            acc['seen'] += 1
            acc['triggers'] += n_triggers
            acc['games'] += int(has_collar)
            acc['positions'] += n_diff
            acc['total_pos'] += n_total

    def accumulate_stop_stats(self, stop_stats):
        for st, s in stop_stats.items():
            n = s['n']
            if not n:
                continue
            for acc in (self.total_stop, self.window_stop):
                acc[st]['n'] += n
                acc[st]['cpl'] += s['cpl'] * n
                acc[st]['bmr'] += s['bmr'] * n
                if not np.isnan(s.get('sims', float('nan'))):
                    acc[st]['sims'] += s['sims'] * n

    def print_stop_stats(self):
        stops = ("rsc", "jsd", "full")

        def pct(acc, st):
            total_n = sum(acc[s]['n'] for s in stops)
            n = acc[st]['n']
            return n / total_n if total_n else 0.0

        def cpl(acc, st):
            n = acc[st]['n']
            return acc[st]['cpl'] / n if n else float('nan')

        def bmr(acc, st):
            n = acc[st]['n']
            return acc[st]['bmr'] / n if n else float('nan')

        def avg_sims(acc, st):
            n = acc[st]['n']
            return round(acc[st]['sims'] / n) if n else 0

        def rows(label, acc):
            hdr  = f"{RS}  {'':<12} |" + "".join(f"  {st:<4}({pct(acc,st):.0%})  |" for st in stops)
            crow = f"{RS}  {label:<12} |" + "".join(f"  CPL {cpl(acc,st):5.2f}  |" for st in stops)
            brow = f"{RS}  {label:<12} |" + "".join(f"  BMR {bmr(acc,st):.3f}  |" for st in stops)
            return hdr, crow, brow

        wh, wc, wb = rows("last 100", self.window_stop)
        th, tc, tb = rows("overall",  self.total_stop)
        tph = (f"{RS}  {'':<12} |"
               + "".join(f"  {st:<4}({pct(self.total_stop,st):.0%})  |" for st in stops))
        srow = (f"{RS}  {'avg sims':<12} |"
                + "".join(f"    {avg_sims(self.window_stop,st):5d}    |" for st in stops))
        print(wh)
        print(srow)
        print(wc)
        print(wb)
        print()
        print(tph)
        print(tc)
        print(tb)

        for st in stops:
            self.window_stop[st] = {'n': 0, 'cpl': 0.0, 'bmr': 0.0, 'sims': 0.0}

    def print_collar_stats(self):
        dry = " [dry]" if self.config.collar_rescore_dry_run else ""

        def fmt_row(label, s):
            pos_pct = (s['positions'] / s['total_pos'] * 100) if s['total_pos'] else 0.0
            return (
                f"{RS} Collar{dry}  {label:<12}"
                f"  trigs {s['triggers']:>4}"
                f"  collar_games {s['games']:>3}/{s['seen']:<3}"
                f"  pos {s['positions']:>5} ({pos_pct:.1f}%)"
            )

        w = self.collar_window
        print(fmt_row(f"batch {w['seen']:>3}:", w))

        if len(self.collar_history) >= 10:
            combined = {'games': 0, 'triggers': 0, 'positions': 0, 'total_pos': 0, 'seen': 0}
            for h in self.collar_history[-10:]:
                for k in combined:
                    combined[k] += h[k]
            print(fmt_row("last 300:", combined))

    def aggregate_metrics(self, epoch, vscale, progress_csv_path):
        if not self.pending_metrics:
            return

        chunk_size = self.config.retrain_size
        chunk = self.pending_metrics[:chunk_size]
        self.pending_metrics = self.pending_metrics[chunk_size:]

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        nn_vals_stm, target_ys, sf_cps, sf_wdls, result_zs = [], [], [], [], []
        mol_vals, policy_samples = [], []

        for m in chunk:
            nn_v = m.get('nn_value')
            if nn_v is None:
                continue
            stm_sign = 1.0 if m['stm'] else -1.0
            nn_stm = float(np.clip(nn_v * stm_sign / vscale, -1.0, 1.0))
            nn_vals_stm.append(nn_stm)
            target_ys.append(m['target_y'])
            sf_cps.append(m['sf_cp'])
            wdl = m.get('sf_wdl')
            sf_wdls.append(wdl if wdl is not None else float('nan'))
            result_zs.append(m['result_z_stm'])
            mol = m.get('mass_on_legal')
            mol_vals.append(mol if mol is not None else float('nan'))
            priors_map = {u: p for u, p in m.get('nn_raw_priors', [])}
            policy_samples.append((priors_map, m.get('candidate_visits', [])))

        nn_vals_stm = np.array(nn_vals_stm, dtype=np.float32)
        target_ys   = np.array(target_ys,   dtype=np.float32)
        sf_cps      = np.array(sf_cps,      dtype=np.float32)
        sf_wdls     = np.array(sf_wdls,     dtype=np.float32)
        result_zs   = np.array(result_zs,   dtype=np.float32)
        mol_vals    = np.array(mol_vals,     dtype=np.float32)

        valid_v = ~np.isnan(nn_vals_stm) & ~np.isnan(target_ys)
        n = int(valid_v.sum())
        val_mse = np.mean(
            (nn_vals_stm[valid_v] - target_ys[valid_v]) ** 2
        ) if n else float('nan')
        val_corr = (
            np.corrcoef(nn_vals_stm[valid_v], target_ys[valid_v])[0, 1]
        ) if n > 1 else float('nan')

        pol_stats = batch_policy_metrics_from_priors(
            policy_samples, self.config.uniform_eps, self.config.prior_clip_max)

        valid_m = ~np.isnan(mol_vals)
        mol_est = float(np.mean(mol_vals[valid_m])) if valid_m.any() else float('nan')
        mol_cov = float(valid_m.mean()) if len(valid_m) else 0.0

        stats = {'value_mse': val_mse, 'value_corr': val_corr}
        stats.update(pol_stats)
        stats['mass_on_legal'] = mol_est
        print_validation(epoch, stats, mass_on_legal=mol_est, mol_coverage=mol_cov)

        row = {**stats, 'model_epoch': epoch, 'n_samples': n,
               'mol_coverage_pct': round(mol_cov * 100, 1)}
        row_df = pd.DataFrame([row])
        if os.path.exists(progress_csv_path):
            all_df = pd.concat(
                [pd.read_csv(progress_csv_path), row_df], ignore_index=True)
        else:
            all_df = row_df
        all_df.round(5).to_csv(progress_csv_path, index=False)

        MAX_SC = 5000
        n_pts = len(nn_vals_stm)
        idx = (np.random.choice(n_pts, MAX_SC, replace=False)
               if n_pts > MAX_SC
               else np.arange(n_pts))

        xv   = nn_vals_stm[idx]
        ytgt = target_ys[idx]
        ycp  = sf_cps[idx]
        ywdl = sf_wdls[idx]
        zc   = result_zs[idx]

        z_palette = {1.0: "#3cb371", 0.0: "#ffd700", -1.0: "#ff8c00"}
        c_all = [z_palette.get(round(float(z)), "#999999") for z in zc]

        import matplotlib.patches as mpatches
        legend = [
            mpatches.Patch(color="#3cb371", label="win"),
            mpatches.Patch(color="#ffd700", label="draw"),
            mpatches.Patch(color="#ff8c00", label="loss"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].scatter(ytgt, xv, s=4, alpha=0.3, c='steelblue')
        axes[0].set_xlabel("target Y")
        axes[0].set_ylabel("NN value (STM-POV)")
        axes[0].set_title(f"Epoch {epoch}: NN value vs target  MSE={val_mse:.2f} r={val_corr:.2f}")

        valid2 = ~np.isnan(ycp)
        xv2, ycp2 = xv[valid2], ycp[valid2]
        ycp2_tanh = np.tanh(ycp2 * (np.arctanh(0.8) / 500.0))
        mse2 = np.mean((xv2 - ycp2_tanh) ** 2) if len(xv2) > 1 else float('nan')
        corr2 = np.corrcoef(xv2, ycp2)[0, 1] if len(xv2) > 1 else float('nan')
        c2 = [c_all[i] for i in range(len(xv)) if valid2[i]]
        axes[1].scatter(ycp2, xv2, s=4, alpha=0.3, c=c2)
        axes[1].set_xlabel("SF CP (STM-POV)")
        axes[1].set_ylabel("NN value (STM-POV)")
        axes[1].set_title(f"NN value vs SF centipawns  MSE={mse2:.2f} r={corr2:.2f}")
        axes[1].legend(handles=legend, fontsize=8)

        valid3 = ~np.isnan(ywdl)
        xv3_raw, ywdl3_raw = xv[valid3], ywdl[valid3]
        mse3 = np.mean((xv3_raw - ywdl3_raw) ** 2) if len(xv3_raw) > 1 else float('nan')
        corr3 = np.corrcoef(xv3_raw, ywdl3_raw)[0, 1] if len(xv3_raw) > 1 else float('nan')
        c3 = [c_all[i] for i in range(len(xv)) if valid3[i]]

        def boundary_jitter(arr, scale=0.05):
            j = np.random.uniform(-scale, scale, size=arr.shape).astype(arr.dtype)
            j = np.where(arr + j > 1.0, -np.abs(j), np.where(arr + j < -1.0, np.abs(j), j))
            return arr + j

        xv3   = boundary_jitter(xv3_raw)
        ywdl3 = boundary_jitter(ywdl3_raw)
        axes[2].scatter(ywdl3, xv3, s=4, alpha=0.3, c=c3)
        axes[2].set_xlabel("SF WDL score (STM-POV)")
        axes[2].set_ylabel("NN value (STM-POV)")
        axes[2].set_title(f"NN value vs SF WDL  MSE={mse3:.2f} r={corr3:.2f}")
        axes[2].legend(handles=legend, fontsize=8)

        fig.tight_layout()
        plot_path = os.path.join(os.path.dirname(progress_csv_path), "validation_latest.png")
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[metrics] plot saved")

    def push_analyzed(self, report=True):
        # safeguard here
        if not len(self.analyzed_results):
            return
        
        run_dir = self.config.run_dir
        outp, c, b = save_analysis_chunk_simple(run_dir, self.analyzed_results)
        self.n_saved += 1    
        self.tcpl += c; self.tbmr += b

        # update the rolling windows
        self.last_10_cpls.append(c)
        self.last_10_cpls = self.last_10_cpls[-10:]
        self.last_10_bmrs.append(b)
        self.last_10_bmrs = self.last_10_bmrs[-10:]

        if report:
            if len(self.last_10_cpls) >= 10:
                last_10_avg_c = np.mean(self.last_10_cpls)
                last_10_avg_b = np.mean(self.last_10_bmrs)
                print(
                    f"{RS} {'Last 10 avg:':<16} CPL {last_10_avg_c:.3f}",
                    f"BMR {last_10_avg_b:.3f}"
                )

            if self.n_saved >= 2:
                cpl_mean = self.tcpl/self.n_saved
                tmbr_mean = self.tbmr/self.n_saved 
                print(
                    f"{RS} {'Overall stats:':<16} CPL {cpl_mean:.3f}",
                    f"BMR {tmbr_mean:.3f}"
                )

            n_tot = self.games_processed
            if n_tot > self.config.rescore_analyze_batch:
                rate = n_tot / (time.time() - self.start_time)
                print(f"{RS} Total Games: {n_tot} ({rate:.3f} games/sec)")
            
            cache_stats = self.cache.stats()
            avg_total_ms = (
                1000 * self.sf_call_time / self.sf_call_count
                if self.sf_call_count else 0.0
            )
            avg_compute_ms = (
                1000 * self.sf_compute_time / self.sf_compute_count
                if self.sf_compute_count else 0.0
            )
            print(
                f"{RS} SF requests: {self.n_sf_submitted} submitted,"
                f" {self.n_cache_hits} cache hits,"
                f" compute {avg_compute_ms:.0f}ms  total {avg_total_ms:.0f}ms,"
                f" req_q {self.req_q.qsize()}  pending {len(self.pending)},"
                f" cache size {cache_stats['size']}"
            )
            self.sf_call_time = 0.0
            self.sf_call_count = 0
            self.sf_compute_time = 0.0
            self.sf_compute_count = 0
            print()

            self.print_collar_stats()

            w_this = self.written_this_round
            wtot = self.written_total
            print(f"{RS} Training samples this round: {w_this} | total: {wtot}")
            print()

        self.collar_history.append(dict(self.collar_window))
        self.collar_history = self.collar_history[-10:]
        zero_collar = {'games': 0, 'triggers': 0, 'positions': 0, 'total_pos': 0, 'seen': 0}
        self.collar_window = dict(zero_collar)

        self.analyzed_results = []

# helpers
def collar_z_map(eval_trace, game_result, threshold, n_consec, reset_cp):
    """
    Walk eval_trace (list of (ply, white_pov_cp)) and return
    (eff_z_by_ply, n_triggers).

    eff_z_by_ply: dict[ply -> white-pov effective result]
      - positions inside a collar segment: +1.0 or -1.0 (collar winner)
      - positions outside any collar: float(game_result)
    n_triggers: number of collar resets (blunder-induced segment splits)

    Collar activates after n_consec consecutive plies above threshold (white)
    or below -threshold (black), retroactively marking those plies.
    Collar resets when the eval crosses back within reset_cp of zero.
    The reset ply itself starts the new segment (gets game_result).
    """
    if not eval_trace:
        return {}, 0

    plies = [p for p, _ in eval_trace]
    evals = [e for _, e in eval_trace]
    n = len(evals)

    collar_state = [None] * n
    collar = None
    count = 0
    segment_start = 0
    n_triggers = 0

    for i, ev in enumerate(evals):
        if collar is None:
            if ev > threshold:
                if count <= 0:
                    count = 0
                count += 1
            elif ev < -threshold:
                if count >= 0:
                    count = 0
                count -= 1
            else:
                count = 0

            if count >= n_consec:
                for j in range(segment_start, i + 1):
                    collar_state[j] = 'white'
                collar = 'white'
                count = 0
            elif count <= -n_consec:
                for j in range(segment_start, i + 1):
                    collar_state[j] = 'black'
                collar = 'black'
                count = 0
        else:
            broken = (
                (collar == 'white' and ev < reset_cp) or
                (collar == 'black' and ev > -reset_cp)
            )
            if broken:
                segment_start = i
                collar = None
                count = 0
                n_triggers += 1
            else:
                collar_state[i] = collar

    eff_z = {}

    # if we detected 0 flips, no updates.
    if not n_triggers:
        return eff_z, n_triggers
    
    for i, ply in enumerate(plies):
        if collar_state[i] == 'white':
            eff_z[ply] = 1.0
        elif collar_state[i] == 'black':
            eff_z[ply] = -1.0
        else:
            eff_z[ply] = float(game_result)

    return eff_z, n_triggers


def value_weight_for_game(cfg, is_draw):
    return cfg.draw_value_scale if is_draw else 1.0


def save_pickle_atomic(obj, path, tries=0):
    try:
        tmp = str(path) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, str(path))
    except Exception as e:
        if tries <= 5:
            time.sleep(0.5)
            save_pickle_atomic(obj, path, tries=tries+1)
        else:
            raise e


def safe_mean(arr):
    a = np.asarray([x for x in arr if x is not None and not np.isnan(x)])
    return np.nanmean(a) if a.size else float("nan")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # First, try regular JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: parse line-by-line (JSONL / concatenated objects)
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items
    

def load_game_index(path=None):
    if not path.endswith("game_index.json"):
        path = os.path.join(path, "game_index.json")
    if not os.path.exists(path):
        return []
    return load_json(path)


def dedupe_results(results):
    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset='game_id', keep='first')
    # dont want the raw key, its just extra data
    if 'raw' in df.columns:
        df = df.drop(columns=['raw'])
    
    return df.to_dict(orient='records')


def lightweight_summary(results):
    """
    Build the small summary dict you use in combined chunk files.
    'results' is a list of per-game dicts (the merged_results or results).
    """
    wm = [r.get("white_cpl", np.nan) for r in results]
    bm = [r.get("black_cpl", np.nan) for r in results]
    om = [r.get("overall_cpl", np.nan) for r in results]
    wb = [r.get("best_move_rate_white", np.nan) for r in results]
    bb = [r.get("best_move_rate_black", np.nan) for r in results]
    ob = [r.get("overall_best_move_rate", np.nan) for r in results]

    return {
        "games": len(results),
        "avg_white_mean_cpl": round(safe_mean(wm), 3),
        "avg_black_mean_cpl": round(safe_mean(bm), 3),
        "avg_overall_mean_cpl": round(safe_mean(om), 3),
        "avg_best_move_rate_white": round(safe_mean(wb), 3),
        "avg_best_move_rate_black": round(safe_mean(bb), 3),
        "avg_overall_best_move_rate": round(safe_mean(ob), 3)
    }


def combine_analysis_staging(run_dir):
    """
    Combine all chunked pickles in run_dir/analysis_staging into the single
    ANALYZE_PKL file at run_dir/ANALYZE_PKL. After a successful combine,
    delete the chunk files.

    Behavior:
      - If ANALYZE_PKL exists, new chunks are appended to it.
      - If ANALYZE_PKL does not exist, a new combined file is created.
      - Chunk files are removed only after the combined pkl is written.
    """
    staging = os.path.join(run_dir, "analysis_staging")
    out_pkl = os.path.join(run_dir, ANALYZE_PKL)

    # nothing to do if staging doesn't exist
    if not os.path.isdir(staging):
        print(f"[combine] no /analysis_staging dir")
        return None

    # list chunk files
    fns = sorted([fn for fn in os.listdir(staging) if fn.endswith(".pkl")])
    if not fns:
        print(f"[combine] no chunk files in /analysis_staging dir")
        return None

    # load existing combined (if any)
    if os.path.exists(out_pkl):
        with open(out_pkl, "rb") as f:
            combined = pickle.load(f)
        prev_results = list(combined.get("results", []))
        prev_df_all = combined.get("df_all", None)
        prev_df_means = combined.get("df_means", None)
    else:
        prev_results = []
        prev_df_all = None
        prev_df_means = None

    # accumulate chunk content
    chunk_results = []
    chunk_dfs = []
    chunk_means = []

    for fn in fns:
        path = os.path.join(staging, fn)
        # load each chunk (let exceptions propagate)
        with open(path, "rb") as f:
            chunk = pickle.load(f)

        # expect chunk structure {summary, results, df_all, df_means}
        cres = chunk.get("results", [])
        if cres:
            chunk_results.extend(cres)

        cdf = chunk.get("df_all", None)
        if cdf is not None:
            chunk_dfs.append(cdf)

        cmeans = chunk.get("df_means", None)
        if cmeans is not None:
            chunk_means.append(cmeans)

    # merge results lists
    merged_results = prev_results + chunk_results

    # merge df_all
    if prev_df_all is None:
        if chunk_dfs:
            df_all = pd.concat(chunk_dfs, ignore_index=True)
        else:
            df_all = None
    else:
        if chunk_dfs:
            df_all = pd.concat([prev_df_all] + chunk_dfs, ignore_index=True)
        else:
            df_all = prev_df_all

    # merge df_means
    if prev_df_means is None:
        if chunk_means:
            df_means = pd.concat(chunk_means, ignore_index=True)
        else:
            df_means = None
    else:
        if chunk_means:
            df_means = pd.concat([prev_df_means] + chunk_means, ignore_index=True)
        else:
            df_means = prev_df_means

    # de dupe
    df_all = df_all.drop_duplicates(['game_id', 'move_num']).sort_values("ts")
    df_means = df_all.drop_duplicates(['game_id']).sort_values("ts")
    merged_results = dedupe_results(merged_results)

    # recompute lightweight summary from merged_results
    summary = lightweight_summary(merged_results)

    combined_new = {
        "summary": summary,
        "results": merged_results,
        "df_all": df_all,
        "df_means": df_means
    }

    # persist atomically using existing helper
    save_pickle_atomic(combined_new, out_pkl)
    print(f"[combine] wrote combined ANALYZE_PKL -> {Path(out_pkl).name} "
          f"({len(merged_results)} games)")

    # delete the chunk files that we just combined
    for fn in fns:
        path = os.path.join(staging, fn)
        os.remove(path)
    print(f"[combine] removed {len(fns)} chunk files from {Path(staging).name}")

    return combined_new


def save_analysis_chunk_simple(run_dir, batch):
    """
    Build a combined-style object for `batch` (list of analysis_out dicts)
    and write it as one pickle into run_dir/analysis_staging/ with a unique
    filename. No tmp file, no exceptions swallowed.
    """
    staging = os.path.join(run_dir, "analysis_staging")
    os.makedirs(staging, exist_ok=True)

    # results list (one row per game)
    results = []
    all_dfs = []
    for analysis_out in batch:
        row = {
            "game_id": analysis_out.get("game_id"),
            "ts": analysis_out.get("ts"),
            "white_cpl": analysis_out.get("white_cpl"),
            "black_cpl": analysis_out.get("black_cpl"),
            "overall_cpl": analysis_out.get("overall_cpl"),
            "best_move_rate_white": analysis_out.get("best_move_rate_white"),
            "best_move_rate_black": analysis_out.get("best_move_rate_black"),
            "overall_best_move_rate": analysis_out.get("overall_best_move_rate")
        }
        results.append(row)
        df = analysis_out.get("df")
        if df is not None:
            all_dfs.append(df)

    # df_all is concat of per-game dfs (or None)
    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else None
    df_means = pd.DataFrame.from_records(results) if results else None
    summary = lightweight_summary(results)

    chunk_obj = {
        "summary": summary,
        "results": results,
        "df_all": df_all,
        "df_means": df_means
    }

    cpl = df_all.delta.mean()
    bmr = df_all.played_best_move.mean()
    
    print(f"{RS} Saving {len(batch)} analyzed games")
    print(f"{RS} {'Batch stats:':<16} CPL {cpl:.3f} BMR {bmr:.3f}")

    fname = f"{int(time.time())}_{uuid.uuid4().hex}.pkl"
    outp = os.path.join(staging, fname)

    with open(outp, "wb") as f:
        pickle.dump(chunk_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return outp, cpl, bmr


def make_fake_visits(mv, lms, ratio_best=60):
    visits = [[mv, int(ratio_best)]]
    
    # may only be 1 legal move
    if len(lms) < 2:
        return visits
    
    sub_optimal = 100 - ratio_best
    bad_visits = 1 + min(5, int(sub_optimal / len(lms)))
    visits += [[m, bad_visits] for m in lms if m != mv]
    return visits


def encourage_best_move(visits, played_mv, best_mv, lms):
    """
    visits: list of (uci, count) from tree
    Sets best_mv visits equal to played_mv visits without lowering played_mv.
    Ensures all legal moves have at least 1 visit.
    Returns list of [uci, int_visits] sorted desc.
    """
    d = {u: max(1, int(v)) for u, v in visits if u}
    for m in lms:
        if m not in d:
            d[m] = 1
    d[best_mv] = max(d.get(best_mv, 1), d.get(played_mv, 1))
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return [[u, int(v)] for u, v in items]


def launch_retrain_async(run_tag, rt_script, working_cfg, epoch=None):
    cmd = [sys.executable, rt_script, "--run-dir", working_cfg.run_dir]
    cmd += ["--batch-size", str(working_cfg.retrain_batch_size)]
    if epoch is not None:
        cmd += ["--epoch", str(epoch)]

    print(f"[retrain] launching worker")

    start_new_session = False
    creationflags = 0
    if os.name == "posix":
        start_new_session = True
    else:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=start_new_session,
            creationflags=creationflags,
            text=True,
            bufsize=1,   # line buffered
        )
    
    except Exception as e:
        raise RuntimeError(f"[retrain] failed to start retrain worker: {e}")

    return {
        "p": p,
        "cmd": cmd,
        "stdout_buf": deque(),
        "stderr_buf": deque(),
        "done": False,
        "rc": None,
    }


def drain_pipe_lines(pipe, buf, max_lines=200):
    n = 0
    if pipe is None:
        return 0

    # Use readline() in a bounded loop; it may block if no newline is available.
    # To keep this non-blocking, only call it when poll() indicates process ended,
    # OR keep max_lines small and accept that it can block if the worker writes
    # partial lines without '\n'. Most scripts print lines, so this is usually fine.
    while n < max_lines:
        ln = pipe.readline()
        if not ln:
            break
        buf.append(ln.rstrip("\n"))
        n += 1
    return n


def poll_retrain(handle, print_output=True):
    """
    Call frequently from your main loop.
    Returns: (done: bool, rc: int | None)
    """
    p = handle["p"]

    rc = p.poll()
    handle["rc"] = rc

    # If you want "live" output while running, you need non-blocking IO (selectors)
    # or a reader thread. The simple safe option: only drain once it's done.
    if rc is None:
        return False, None

    # Process ended: drain remaining output fully
    drain_pipe_lines(p.stdout, handle["stdout_buf"], max_lines=10_000)
    drain_pipe_lines(p.stderr, handle["stderr_buf"], max_lines=10_000)

    if print_output:
        if handle["stdout_buf"]:
            print("[retrain] STDOUT:")
            while handle["stdout_buf"]:
                print(handle["stdout_buf"].popleft())

        if handle["stderr_buf"]:
            print("[retrain] STDERR:")
            while handle["stderr_buf"]:
                print(handle["stderr_buf"].popleft())

    handle["done"] = True
    print(f"[retrain] worker finished exit_code={rc}")

    # Close pipes to release resources
    if p.stdout is not None:
        p.stdout.close()
    if p.stderr is not None:
        p.stderr.close()

    if rc != 0:
        raise RuntimeError(f"[retrain] worker failed; exit_code={rc}")

    return True, rc


def reclaim_vram(mb):
    import tensorflow as tf

    bytes_target = mb * 1024 * 1024
    n = max(1, bytes_target // 4)

    with tf.device("/GPU:0"):
        x = tf.ones([n], dtype=tf.float32)
        y = tf.reduce_sum(x)

    _ = y.numpy()
    return True
