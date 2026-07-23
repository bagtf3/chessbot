import os, json, gzip, pathlib, time
from types import SimpleNamespace
from pathlib import Path
import uuid
import sys
import random
import subprocess
import threading
from collections import deque, defaultdict
from queue import Empty

import pickle
import chess, chess.engine

import pandas as pd
import numpy as np

from pyfastchess import Board

from chessbot import SF_LOC
from chessbot.utils import (
    score_cp_stm_pov, score_cp_white_pov, rnd, kl_divergence,
    batch_policy_metrics_from_priors, print_validation, calc_entropy,
)
from chessbot.lc0_utils import lc0_logits_to_xc0_batch, lc0_table_index
from xerces_training.uci_to_idx import uci_to_idx as UCI_TO_IDX

RS = "[rescore]"
ANALYZE_PKL = "analyze_results_combined.pkl"
EVICTION_WINDOW = 10000
EVICTION_MAX_SIZE = 250000
C = 0.9699
D = d = np.arctanh(0.5)


class Lc0Thread:
    """
    Synchronous batcher for LC0 distillation inference.

    Accumulates (112,8,8) uint8 feature arrays until batch_size is reached,
    then runs a single ORT forward pass and converts results to XC0 training
    samples. No real thread — flushes inline to share the caller's GPU context
    rather than spawning a competing CUDA stream.

    batch_size is the primary GPU-contention lever: larger = less frequent
    inference interruptions to the selfplay workers.
    """

    def __init__(self, ort_session, batch_size=64):
        self.sess       = ort_session
        self.batch_size = batch_size
        self.pending    = []   # (features_uint8, table_index, x, mask, vwht, pwht)
        self.results    = []   # completed (x, mask, xc0_policy, lc0_wdl, vwht, pwht)
        self.n_inferences = 0
        self.n_samples    = 0

        # discover input/output names from session metadata once
        inputs  = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.input_name   = inputs[0].name
        self.output_names = [o.name for o in outputs]

    def submit(self, features_uint8, board, x, mask, vwht, pwht):
        ti   = lc0_table_index(features_uint8)
        ucis = board.legal_moves()
        lc0_idx = np.array([UCI_TO_IDX[ti][u.rstrip('n')] for u in ucis], dtype=np.int32)
        xc0_idx = np.array(board.moves_to_indices(ucis), dtype=np.int32)
        self.pending.append((features_uint8, lc0_idx, xc0_idx, x, mask, vwht, pwht))
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        batch        = self.pending
        self.pending = []

        feats = np.stack([item[0] for item in batch]).astype(np.float32)
        feats[:, 109] /= 99.0  # rule50 plane: raw halfmove clock -> [0,1]

        outs   = self.sess.run(self.output_names, {self.input_name: feats})
        logits = outs[0]  # (N, 1858) policy logits
        wdl    = outs[1]  # (N, 3) WDL probs, STM-POV

        lc0_idx_list = [item[1] for item in batch]
        xc0_idx_list = [item[2] for item in batch]
        xc0_policies = lc0_logits_to_xc0_batch(logits, lc0_idx_list, xc0_idx_list)

        for i, item in enumerate(batch):
            _, _, _, x, mask, vwht, pwht = item
            policy = xc0_policies[i] * mask
            s = policy.sum()
            if s > 0:
                policy /= s
            self.results.append((
                x, mask,
                policy,
                np.array(wdl[i], dtype=np.float32),
                vwht, pwht,
            ))

        self.n_inferences += 1
        self.n_samples    += len(batch)

    def drain(self):
        out          = self.results
        self.results = []
        return out

    def stats(self):
        return {
            'inferences': self.n_inferences,
            'samples':    self.n_samples,
            'pending':    len(self.pending),
        }


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

    def set_best(self, key, uci, cp, abs_cp, game_num, depth=0):
        if key not in self.data:
            self.data[key] = {
                'best': None, 'others': {}, 'hits': 0, 'last_seen': game_num, 'depth': 0
            }
        entry = self.data[key]
        if depth < entry.get('depth', 0):
            return False
        entry['best'] = (uci, cp, abs_cp)
        entry['depth'] = depth
        entry['hits'] += 1
        entry['last_seen'] = game_num
        return True

    def set_move(self, key, uci, cp, abs_cp, game_num):
        entry = self.data.get(key)
        if entry is not None:
            entry['others'][uci] = (cp, abs_cp)
            entry['last_seen'] = game_num

    def load_seed(self, path):
        """Merge a pre-computed SFCache pickle (gzipped) into this cache.
        Existing entries are not overwritten — the seeded depth-20 values win
        only for positions not yet seen in the live run."""
        import gzip as gz
        with gz.open(path, "rb") as f:
            saved = pickle.load(f)
        seeded = saved.get("data", {})
        depth = saved.get("depth", "?")
        added = 0
        for key, entry in seeded.items():
            if key not in self.data:
                entry["last_seen"] = 10 ** 9  # pin: never evict
                self.data[key] = entry
                added += 1
        print(f"[SFCache] loaded seed: {added:,} entries (depth={depth}) from {path}")
        return added

    def save(self, path):
        """Save cache to disk as gzipped pickle, atomically."""
        tmp = str(path) + ".tmp"
        with gzip.open(tmp, "wb") as f:
            pickle.dump({"data": self.data}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, str(path))

    def roll_merge(self, path):
        """Load a previously saved cache and merge into current.
        For each key: higher-depth best wins; others dicts are unioned."""
        with gzip.open(path, "rb") as f:
            saved = pickle.load(f)
        saved_data = saved.get("data", {})
        added = updated = others_added = 0
        for key, se in saved_data.items():
            if key not in self.data:
                se["last_seen"] = 0
                self.data[key] = se
                added += 1
            else:
                cur = self.data[key]
                if (se.get("depth", 0) > cur.get("depth", 0)
                        and se.get("best") is not None):
                    cur["best"] = se["best"]
                    cur["depth"] = se["depth"]
                    updated += 1
                for uci, val in se.get("others", {}).items():
                    if uci not in cur["others"]:
                        cur["others"][uci] = val
                        others_added += 1
        print(
            f"[SFCache] roll_merge: {added:,} added, {updated:,} best upgraded,"
            f" {others_added:,} other moves merged from {os.path.basename(path)}"
        )
        return added, updated

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
    SF worker that processes one game at a time for TT locality.
    submit_game() enqueues a whole game; the thread processes positions
    sequentially in ply order and returns a batch result.

    positions: [(ply_idx, board_copy, xerces_uci, known_best_uci), ...]
      known_best_uci=None  -> full analysis needed (cache miss)
      known_best_uci=<uci> -> best already known, only analyse xerces move
    """

    def __init__(self, sf_game_q, sf_res_q, cfg):
        self.game_q = sf_game_q
        self.res_q  = sf_res_q

        # config floor; depth may throttle below this
        self.base_depth = cfg.rescore_depth  
        self.depth = self.base_depth
        self.sf_config = {'Hash': 256}
        self.stop_ev = threading.Event()
        self.t = None
        self.eng = None

    def submit_game(self, gid, positions):
        self.game_q.put((gid, positions))

    def update_config(self, cfg):
        self.base_depth = cfg.rescore_depth
        if self.depth > self.base_depth:
            self.depth = self.base_depth

    def start(self):
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def close(self):
        # each thread checks its own stop_ev; no shared-queue sentinel needed
        self.stop_ev.set()  
        if self.t is not None:
            self.t.join()
        if self.eng is not None:
            self.eng.quit()
            self.eng = None

    def run(self):
        self.eng = chess.engine.SimpleEngine.popen_uci(SF_LOC, timeout=180)
        self.eng.configure(self.sf_config)

        while not self.stop_ev.is_set():
            try:
                item = self.game_q.get(timeout=0.01)
            except Empty:
                continue
            if item is None:
                break
            gid, positions = item
            results = []
            try:
                for ply_idx, board, xerces_uci, known_best_uci in positions:
                    limit = chess.engine.Limit(depth=self.depth)
                    elapsed = 0.0

                    # pass 1: full best-move analysis (skipped if caller already has best)
                    if known_best_uci is None:
                        t0 = time.time()
                        info = self.eng.analyse(board, limit, info=chess.engine.INFO_ALL)
                        elapsed += time.time() - t0
                        best_uci = str(info['pv'][0])
                        pv_ucis  = [str(m) for m in info.get('pv', [])[:4]]
                        best_cp  = score_cp_stm_pov(info['score'])
                        best_abs = score_cp_white_pov(info['score'], clipped=False)
                    else:
                        # partial cache hit: best known
                        # cp filled in by handle_game_results
                        best_uci = known_best_uci
                        best_cp  = None
                        best_abs = None
                        pv_ucis  = []

                    # pass 2: score xerces's move only if it differs from best
                    if xerces_uci != best_uci:
                        t0 = time.time()
                        info2 = self.eng.analyse(
                            board, limit,
                            root_moves=[chess.Move.from_uci(xerces_uci)],
                            info=chess.engine.INFO_ALL,
                        )
                        elapsed  += time.time() - t0
                        played_cp  = score_cp_stm_pov(info2['score'])
                        played_abs = score_cp_white_pov(info2['score'], clipped=False)
                    else:
                        played_cp  = best_cp
                        played_abs = best_abs

                    results.append({
                        'ply_idx':    ply_idx,
                        'best_uci':   best_uci,
                        'pv_ucis':    pv_ucis,
                        'best_cp':    best_cp,
                        'best_abs':   best_abs,
                        'played_cp':  played_cp,
                        'played_abs': played_abs,
                        'elapsed':    elapsed,
                        'depth':      self.depth,
                    })
            
            except Exception as e:
                self.res_q.put(('__error__', e))
                self.stop_ev.set()
                return

            # whole game done — push batch so Rescorer can finalize in one shot
            self.res_q.put((gid, results))


class Rescorer(object):

    def __init__(self, cfg, sf_game_q, sf_res_q, cache):
        self.config   = cfg
        self.game_q   = sf_game_q
        self.res_q    = sf_res_q
        self.cache    = cache

        self.training_data = []
        self.analyzed_results = []
        self.pending_metrics = []


        self.start_time = None  # set on first game to exclude idle startup time
        self.games_seen = set()
        self.games_processed = 0
        self.written_total = 0
        self.written_this_round = 0
        self.n_saved = 0
        self.total_cpl_plies = 0.0
        self.total_bmr_plies = 0.0
        self.total_plies = 0

        self.n_sf_submitted = 0
        self.n_cache_hits = 0
        self.sf_compute_time = 0.0
        self.sf_compute_count = 0
        self.n_sf_threads = cfg.rescore_n_sf_threads
        self.current_depth = cfg.rescore_depth
        self.last_10_cpls = []
        self.last_10_bmrs = []
        self.last_10_plies = []

        zero_collar = lambda: {
            'games': 0, 'triggers': 0, 'positions': 0,
            'total_pos': 0, 'seen': 0}

        self.collar_total = zero_collar()
        self.collar_window = zero_collar()
        self.collar_history = []

        zero_stop = lambda: {'n': 0, 'cpl': 0.0, 'bmr': 0.0, 'sims': 0.0}
        self.total_stop = {st: zero_stop() for st in ("full", "rsc", "jsd")}
        self.window_stop = {st: zero_stop() for st in ("full", "rsc", "jsd")}

        zero_sc = lambda: {
            'total': 0, 'accepted': 0, 'kl': 0, 'ce': 0, 'cpl': 0, 'rng': 0,
            'lc0_blunder': 0, 'lc0_pv': 0, 'lc0_enrich': 0}

        self.sample_counts = zero_sc()
        self.sample_counts_window = zero_sc()

        self.tscale_n = 0
        self.tscale_eligible = 0
        self.tscale_iters = 0
        self.tscale_ne_before = 0.0
        self.tscale_ne_after = 0.0
        self.tscale_best_T = 0.0
        self.tscale_time = 0.0

        self.intake = deque()
        self.pending = {}
        self.pending_lc0_metrics = []

        lc0_model     = cfg.lc0_distill_model_name or os.getenv('LC0_DISTILL_MODEL', '')
        lc0_trt_cache = os.getenv('LC0_DISTILL_TRT_CACHE', '')
        self.lc0_thread = None
        if lc0_model and lc0_trt_cache:
            from chessbot.lc0_utils import make_lc0_trt_session
            batch_size = cfg.lc0_distill_batch_size
            lc0_onnx = os.path.join(lc0_trt_cache, f'{lc0_model}.onnx')
            sess = make_lc0_trt_session(
                lc0_onnx, lc0_model, lc0_trt_cache,
                opt_batch=batch_size, max_batch=batch_size * 2)
            
            # Hard-fail if TRT didn't load
            # silent CPU fallback would silently bottleneck training.
            active = sess.get_providers()[0]
            if active != 'TensorrtExecutionProvider':
                msg = f"{RS} Lc0Thread TRT failed, got {active}."
                msg += " Check CUDA/TRT DLLs in PATH."
                raise RuntimeError(msg)
            
            self.lc0_thread = Lc0Thread(sess, batch_size=batch_size)
            print(f"{RS} Lc0Thread: TRT batch_size={batch_size}")

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

    def maybe_seed_cache(self, b_fast, mv, Q_stm, turn, repetitions, depth=0):
        short_fen = b_fast.fen(include_counters=False)
        reps = 2 if repetitions[short_fen] >= 2 else 0
        hmc = b_fast.halfmove_clock()
        halfmoves = 0 if hmc < 45 else hmc
        cache_key = (short_fen, reps, halfmoves)
        best_cp = int(np.arctanh(np.clip(Q_stm, -C, C)) * 100.0 / D)
        best_abs = best_cp if turn else -best_cp
        self.cache.set_best(
            cache_key, mv, best_cp, best_abs, self.games_processed, depth=depth
        )

    def submit(self, pkl_file):
        self.intake.append(pkl_file)

    def tick(self):
        if self.lc0_thread is not None:
            for sample in self.lc0_thread.drain():
                self.training_data.append(sample)

        # drain completed game batches from SF threads
        while True:
            try:
                gid, results = self.res_q.get_nowait()
            except Empty:
                break
            if gid == '__error__':
                raise results
            self.handle_game_results(gid, results)

        if len(self.intake) > len(self.pending):
            intake_list = list(self.intake)
            random.shuffle(intake_list)
            self.intake = deque(intake_list)
        
        while self.intake and len(self.pending) < 20:
            self.start_game(self.intake.popleft())

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
    
    def encode_board(self, board):
        """Board -> model input, per cfg.encoding_type. lc0 planes are raw uint8
        (rule50 unscaled); the /99 + float conversion happens at the model-input
        boundary (retrain / infer). board must carry full move history for lc0."""
        if self.config.encoding_type == "lc0":
            return board.lc0_features()
        if self.config.encoding_type == "xc0h":
            return board.history_tokens(self.config.history_K)
        return board.encode_64_tokens()

    def make_policy_example(self, board, ucis, visits):
        indices = board.moves_to_indices(ucis)
        policy = np.zeros(1858, dtype=np.float32)
        s = sum(visits)
        pi = np.array([v / s for v in visits], dtype=np.float32)
        pi = np.clip(pi, 0.0, self.config.prior_clip_max)
        pi = pi / pi.sum()
        for idx, p in zip(indices, pi):
            policy[idx] += p
        x = self.encode_board(board)
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
            size = cfg.retrain_size

        chunk = self.training_data[:size]
        remainder = self.training_data[size:]

        filename = f"{int(time.time())}-{uuid.uuid4().hex}.pkl"
        out_path = out_dir / filename

        with open(out_path, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.training_data = remainder
        self.written_this_round += len(chunk)
        self.written_total += len(chunk)

    def training_data_from_sf(self, board, mv, cm, Y, is_draw, policy_weight=1.0):
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
        self.append_flat_policy_example(board, ucis, visits, Y, vwht, policy_weight)

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

        is_validation_game = 'validation' in game_data.get('scenario', '').lower()
        skip_all_training = is_validation_game

        # snapshot config fields so hot-reloads don't affect an in-flight game
        game_cfg_keys = (
            'train_on_stockfish', 'z_mix',
            'KL_weight_boost', 'KL_boost_threshold',
            'rescore_equiv_range', 'rescore_blunder_cp_loser',
            'rescore_blunder_cp_winner', 'rescore_inaccuracy_cp',
            'uniform_eps', 'prior_clip_max',
            'collar_threshold_cp', 'collar_n_consec', 'collar_reset_cp',
            'use_collar_rescoring', 'rescore_analyze_batch',
            'draw_value_scale',
            'rescore_kl_threshold', 'rescore_ce_threshold', 'rescore_sample_floor',
            'rescore_target_acceptance',
            'vscale', 'lc0_enrich_frac', 'lc0_enrich_weight'
        )
        game_state = {
            'gid': gid,
            'game_data': game_data,
            'cfg': SimpleNamespace(**{k: getattr(cfg, k) for k in game_cfg_keys}),
            'ply_states': [],
            'waiting': False,  # True once a batch has been submitted to a SF thread
        }

        # (ply_idx, board_copy, xerces_uci, known_best_uci) for cache misses
        sf_positions = []
        repetitions = defaultdict(int)
        repetitions[b_fast.fen(include_counters=False)] += 1
        # walk each move: route SF moves, build ply state, check cache
        for i, mv in enumerate(game_data.get('moves_played', [])):
            move_ch = chess.Move.from_uci(mv)
            is_sf_move = vs_stockfish and (board_ch.turn == sf_color)
            turn = board_ch.turn

            if is_sf_move and ((not cfg.train_on_stockfish) or skip_all_training):
                deep_enough = game_sf_depth and game_sf_depth >= cfg.rescore_depth
                if not skip_all_training and deep_enough:
                    tr_sf = tree_data.get(i, tree_data.get(str(i), {}))
                    Q_stm = tr_sf.get('Q_stm')
                    if Q_stm is not None:
                        self.maybe_seed_cache(
                            b_fast, mv, Q_stm, turn, repetitions, depth=game_sf_depth
                        )
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
                wdl = tr.get('best_wdl')
                this_q = (wdl[0] - wdl[2]) if wdl is not None else 0.0
                Q = this_q if turn else -this_q

            sf_wdl_tr = tr.get('sf_wdl')
            if sf_wdl_tr is not None and len(sf_wdl_tr) == 3:
                Y_init = (0.5*z_to_wdl(Z_stm) + 0.5*np.array(sf_wdl_tr, dtype=np.float32))
            else:
                Y_init = z_to_wdl(Z_stm)

            if is_sf_move:
                if game_sf_depth and game_sf_depth >= cfg.rescore_depth:
                    self.maybe_seed_cache(
                        b_fast, mv, Q, turn, repetitions, depth=game_sf_depth
                    )

                sf_pwht = 0.0 if is_validation_game else 1.0
                self.training_data_from_sf(
                    b_fast, mv, cm, Y_init, is_draw, policy_weight=sf_pwht
                )

                self.pending_metrics.append({
                    'stm': turn,
                    'nn_value': tr.get('nn_value'),
                    'nn_raw_priors': tr.get('nn_raw_priors', []),
                    'mass_on_legal': tr.get('nn_mass_on_legal'),
                    'sf_cp': int(np.arctanh(np.clip(Q, -C, C)) * 100.0 / D),
                    'candidate_visits': [(c['uci'], c['visits']) for c in cm],
                    'result_z_stm': Z_stm,
                    'target_y': Y_init[0] - Y_init[2],
                    'policy_eligible': sf_pwht > 0
                })

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
            x = self.encode_board(b_fast)
            mask = b_fast.legal_move_mask()
            short_fen = b_fast.fen(include_counters=False)
            reps = 2 if repetitions[short_fen] >= 2 else 0
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
                'cache_key': cache_key,
                'best_uci': None,
                'best_cp': None,
                'best_abs': None,
                'pv_ucis': [],
                'played_cp': None,
                'played_abs': None,
                'resolved': False,
            }

            # cache only checked for opening positions — mid/late game rarely repeats
            if i < 30:
                entry = self.cache.get(cache_key)
            else:
                entry = None

            if entry and entry['best'] is not None:
                best_uci, best_cp, best_abs, *_ = entry['best']
                ply['best_uci'] = best_uci
                ply['best_cp'] = best_cp
                ply['best_abs'] = best_abs
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
                    # best known but xerces move unscored
                    # pass known_best_uci to skip pass 1
                    sf_positions.append((i, board_ch.copy(), xerces_uci, best_uci))
            else:
                # full cache miss — thread runs both passes
                sf_positions.append((i, board_ch.copy(), xerces_uci, None))

            game_state['ply_states'].append(ply)
            board_ch.push(move_ch)
            b_fast.push_uci(mv)
            repetitions[b_fast.fen(include_counters=False)] += 1

        self.pending[gid] = game_state
        # submit to SF thread or finalize immediately if cache covered everything
        if sf_positions:
            self.n_sf_submitted += len(sf_positions)
            game_state['waiting'] = True
            self.game_q.put((gid, sf_positions))
        else:
            self.finalize_game(self.pending.pop(gid))

    def handle_game_results(self, gid, results):
        if gid not in self.pending:
            return

        game = self.pending[gid]
        ply_map = {p['ply_idx']: p for p in game['ply_states']}

        for r in results:
            ply_idx    = r['ply_idx']
            best_uci   = r['best_uci']
            best_cp    = r['best_cp']
            best_abs   = r['best_abs']
            played_cp  = r['played_cp']
            played_abs = r['played_abs']

            self.sf_compute_time  += r['elapsed']
            self.sf_compute_count += 1

            ply = ply_map.get(ply_idx)
            if ply is None:
                continue

            key        = ply['cache_key']
            xerces_uci = ply['xerces_uci']
            depth      = r['depth']
            cache_eligible = ply_idx < 30  # mirror the GET gate in start_game

            if best_cp is not None:
                # full analysis result — write best to cache
                if cache_eligible:
                    self.cache.set_best(
                        key, best_uci, best_cp, best_abs,
                        self.games_processed, depth=depth
                    )
                ply['best_uci'] = best_uci
                ply['best_cp']  = best_cp
                ply['best_abs'] = best_abs
            # else: partial hit — best_* already populated from cache in start_game

            if xerces_uci != best_uci:
                # xerces deviated — check if it actually found something better
                entry = self.cache.get(key) if cache_eligible else None
                if entry and entry['best'] is not None:
                    old_best_uci, old_best_cp, old_best_abs, *_ = entry['best']
                    if played_cp > old_best_cp:
                        # xerces move is stronger — promote it to best in cache
                        promoted = self.cache.set_best(
                            key, xerces_uci, played_cp, played_abs,
                            self.games_processed, depth=depth
                        )
                        self.cache.set_move(
                            key, old_best_uci, old_best_cp, old_best_abs,
                            self.games_processed
                        )
                        if promoted:
                            ply['best_uci'] = xerces_uci
                            ply['best_cp']  = played_cp
                            ply['best_abs'] = played_abs
                    else:
                        self.cache.set_move(
                            key, xerces_uci, played_cp, played_abs,
                            self.games_processed
                        )
                elif not cache_eligible:
                    pass  # ply >= 30: skip cache writes entirely
                else:
                    self.cache.set_move(
                        key, xerces_uci, played_cp, played_abs,
                        self.games_processed
                    )

            ply['pv_ucis']    = r.get('pv_ucis', [])
            ply['played_cp']  = played_cp
            ply['played_abs'] = played_abs
            ply['resolved']   = True

        game['waiting'] = False
        self.finalize_game(self.pending.pop(gid))

    def finalize_game(self, game_state):
        if self.start_time is None:
            self.start_time = time.time()
        cfg = game_state['cfg']
        game_data = game_state['game_data']
        gid = game_state['gid']
        result = game_data['result']
        is_draw = (result == 0) or (result == 0.0)
        vs_stockfish = game_data.get('vs_stockfish', False)
        sf_color = game_data.get('stockfish_is_white')

        KL_coef = cfg.KL_weight_boost
        do_KL_boost = (KL_coef > 0) and (KL_coef != 1.0)
        EQUIV = cfg.rescore_equiv_range

        # acceptance thresholds
        kl_t = cfg.rescore_kl_threshold
        ce_t = cfg.rescore_ce_threshold
        cpl_t = cfg.rescore_inaccuracy_cp

        cpl_s = 0.0
        n_plies = 0
        rows = []
        eval_trace = []
        turn_at = {}
        pending = []
        pending_aux = []
        kl_map = {}
        ce_map = {}
        mse_map = {}

        # loop 1: per-ply stats, visit redistribution, policy build, accumulate pending
        for ply in sorted(game_state['ply_states'], key=lambda p: p['ply_idx']):
            i = ply['ply_idx']
            mv = ply['mv']
            turn = ply['turn']
            turn_at[i] = turn
            tr = ply['tr']
            cm = ply['cm']
            Z_stm = ply['Z_stm']
            Q = ply['Q']
            xerces_uci = ply['xerces_uci']
            best_uci_raw = ply['best_uci']
            best_cp = ply['best_cp']
            best_abs = ply['best_abs']
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
            missed_mate = (best_cp >= 1200) and (played_cp >= 500)
            if missed_mate:
                loss_this = min(200, loss_this)
            cpl_s += loss_this
            n_plies += 1

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
            # ensure all legal moves have at least 1 visit
            vmap = {u: max(1, int(v)) for u, v in visits}
            for m in lms:
                if m not in vmap:
                    vmap[m] = 1
            
            visits = sorted(vmap.items(), key=lambda x: x[1], reverse=True)

            kl_eligible = loss_this <= EQUIV

            # variable blunder thresholds based on outcome
            blunder_cp = cfg.rescore_blunder_cp_winner
            if Z_stm <= 0.0:
                blunder_cp = cfg.rescore_blunder_cp_loser

            missed_mate_still_won = missed_mate and Z_stm > 0

            is_blunder = loss_this >= blunder_cp
            true_blunder = not (played_cp > 350 and Z_stm > 0) and is_blunder

            # mild and big blunders: drop xc0 record, queue lc0 instead
            skip_xc0 = not missed_mate_still_won and (loss_this >= blunder_cp)
            if missed_mate_still_won:
                lc0_mode = None
            elif true_blunder:
                lc0_mode = 'pos_sf_xc0_pov'
            elif is_blunder:
                lc0_mode = 'pos_sf_pov'
            elif loss_this >= cfg.rescore_inaccuracy_cp:
                lc0_mode = 'pos_only'
            else:
                lc0_mode = None

            # just FYI if we do any adjustments we need to re-sort.

            mvs = [v[0] for v in visits]
            vis = [v[1] for v in visits]

            if Z_stm >= 0 and loss_this <= EQUIV and len(mvs) >= 5:
                self.tscale_eligible += 1
                vis_arr = np.array(vis, dtype=np.float64)
                vis_arr /= vis_arr.sum()
                _, ne = calc_entropy(vis_arr)
                target_e = 0.65
                if ne > target_e:
                    t0 = time.perf_counter()
                    lo, hi, best, best_dist = 0.3, 1.0, 0.75, float('inf')
                    iters = 0
                    for _ in range(5):
                        mid = (lo + hi) / 2
                        scaled = vis_arr ** (1.0 / mid)
                        scaled /= scaled.sum()
                        _, ne_mid = calc_entropy(scaled)
                        iters += 1
                        if ne_mid < ne:
                            dist = abs(ne_mid - target_e)
                            if dist < best_dist:
                                best, best_dist = mid, dist
                            if dist < 0.05:
                                break
                        if ne_mid < target_e:
                            lo = mid
                        else:
                            hi = mid
                    vis_arr = vis_arr ** (1.0 / best)
                    vis = list(vis_arr / vis_arr.sum())
                    _, ne_after = calc_entropy(np.array(vis, dtype=np.float64))
                    self.tscale_n += 1
                    self.tscale_iters += iters
                    self.tscale_ne_before += ne
                    self.tscale_ne_after += ne_after
                    self.tscale_best_T += best
                    self.tscale_time += time.perf_counter() - t0

            priors_map = {c['uci']: c['P'] for c in cm}
            priors = [priors_map.get(u, 0.0) for u in mvs]

            vwht = value_weight_for_game(cfg, is_draw)
            pwht = 1.0
            kl = kl_divergence(priors, vis)
            kl_map[i] = kl
            if kl_eligible and do_KL_boost:
                if kl >= cfg.KL_boost_threshold:
                    pwht *= KL_coef

            # build policy from precomputed board state
            idx_map = ply['idx_map']
            indices = [idx_map[u] for u in mvs]
            policy = np.zeros(1858, dtype=np.float32)
            s = sum(vis)
            pi = np.array([v / s for v in vis], dtype=np.float32)
            pi = np.clip(pi, 0.0, cfg.prior_clip_max)
            pi = pi / pi.sum()
            for idx, p in zip(indices, pi):
                policy[idx] += p

            pending.append((ply['x'], ply['mask'], policy, Q, turn, i, vwht, pwht))
            sf_pv = ply.get('pv_ucis', [])
            if lc0_mode == 'pos_sf_xc0_pov':
                xc0_pv = [e['uci'] for e in ply['tr'].get('pv', [])[:4]]
                pv_seqs = [sf_pv, xc0_pv]
            elif lc0_mode == 'pos_sf_pov':
                pv_seqs = [sf_pv[:1]]
            else:
                pv_seqs = []

            pending_aux.append({
                'sfen':            ply['cache_key'][0],
                'pv_seqs':         pv_seqs,
                'skip_xc0':        skip_xc0,
                'lc0_mode':        lc0_mode,
                'is_blunder':      is_blunder,
                'ply_i':           i,
                'stm':             turn,
                'nn_value':        tr.get('nn_value'),
                'nn_wdl':          tr.get('nn_wdl'),
                'nn_raw_priors':   tr.get('nn_raw_priors', []),
                'mass_on_legal':   tr.get('nn_mass_on_legal'),
                'best_wdl':        tr.get('best_wdl'),
                'sf_cp':           best_cp,
                'candidate_visits': list(zip(mvs, vis)),
                'result_z_stm':    Z_stm,
                'kl':              kl,
                'cpl':             loss_this,
            })

        # collar-adjusted result targets
        eff_z_by_ply, n_triggers = collar_z_map(eval_trace, result, cfg, gid)

        n_diff = 0
        sc = self.sample_counts
        scw = self.sample_counts_window

        start_fen    = game_data['start_fen']
        moves_played = game_data['moves_played']

        # loop 2: compute WDL targets, add to training, build lc0 waypoints
        lc0_waypoints = {}
        for tup, aux in zip(pending, pending_aux):
            x, mask, policy, Q, is_white, ply_i, vwht, pwht = tup
            z_orig = result if is_white else -result
            eff_z_white = eff_z_by_ply.get(ply_i, result)
            z_eff = eff_z_white if is_white else -eff_z_white
            if eff_z_white != result:
                n_diff += 1

            z = z_eff if cfg.use_collar_rescoring else z_orig
            Y = blend_wdl(z, aux.get('best_wdl'), is_white)

            nn_wdl = aux['nn_wdl']
            nn_value = aux['nn_value']
            if nn_wdl is not None:
                eps = 1e-7
                nw = nn_wdl if is_white else (nn_wdl[2], nn_wdl[1], nn_wdl[0])
                p = np.clip(nw, eps, 1 - eps)
                p = p / p.sum()
                ce = -float(np.dot(Y, np.log(p)))
            else:
                ce = 0.0
            if nn_value is not None:
                sign = 1 if is_white else -1
                nn_value_stm = np.clip(nn_value * sign, -1.0, 1.0)
                mse_map[ply_i] = (nn_value_stm - (Y[0] - Y[2])) ** 2
            ce_map[ply_i] = ce
            hit_kl  = aux['kl'] > kl_t
            hit_ce  = ce > ce_t
            hit_cpl = aux['cpl'] >= cpl_t

            sc['total'] += 1
            scw['total'] += 1

            entry = (x, mask, policy, Y, vwht, pwht)

            if not aux.get('skip_xc0'):
                self.training_data.append(entry)
                sc['accepted'] += 1
                scw['accepted'] += 1
                if hit_kl:
                    sc['kl'] += 1
                    scw['kl'] += 1
                if hit_ce:
                    sc['ce'] += 1
                    scw['ce'] += 1
                if hit_cpl:
                    sc['cpl'] += 1
                    scw['cpl'] += 1

            if self.lc0_thread is not None:
                if aux.get('lc0_mode'):
                    lc0_waypoints[aux['ply_i']] = (entry, aux, Y, is_white)
                    sc['lc0_blunder'] += 1
                    scw['lc0_blunder'] += 1
                elif random.random() < cfg.lc0_enrich_frac:
                    lc0_waypoints[aux['ply_i']] = (entry, aux, Y, is_white)
                    sc['lc0_enrich'] += 1
                    scw['lc0_enrich'] += 1

            self.pending_metrics.append({
                **aux,
                'target_y': Y[0] - Y[2],
                'target_wdl': Y,
            })

        # soft pool: sample up to target acceptance rate
        # n_hard = ...
        # n_soft = len(soft_pool)
        # n_total = n_hard + n_soft
        # target_n = int(cfg.rescore_target_acceptance * n_total)
        # n_need = max(0, target_n - n_hard)
        # n_sample = min(n_soft, max(int(cfg.rescore_sample_floor * n_soft), n_need))
        # if n_sample > 0 and n_soft > 0:
        #     chosen = np.random.choice(n_soft, size=n_sample, replace=False)
        #     for i in chosen:
        #         entry, aux, Y, is_white = soft_pool[i]
        #         self.training_data.append(entry)
        #         sc['accepted'] += 1
        #         scw['accepted'] += 1
        #         sc['rng'] += 1
        #         scw['rng'] += 1
        #         if self.lc0_thread is not None and random.random() < cfg.lc0_enrich_frac:
        #             lc0_waypoints[aux['ply_i']] = (entry, aux, Y, is_white)
        #             sc['lc0_enrich'] += 1
        #             scw['lc0_enrich'] += 1

        if lc0_waypoints and self.lc0_thread is not None:
            lc0_meta = []
            b_lc0 = Board(start_fen)
            for ply_i, mv in enumerate(moves_played):
                if ply_i in lc0_waypoints:
                    entry, aux, Y, is_white = lc0_waypoints[ply_i]
                    got_fen = b_lc0.fen(include_counters=False)
                    if got_fen != aux['sfen']:
                        msg = f"[rescore] FEN mismatch at ply {ply_i}: {got_fen} "
                        msg += f"!= {aux['sfen']}"
                        print(msg)

                    x, mask, _, _, vwht, _ = entry
                    ucis_now = b_lc0.legal_moves()
                    self.lc0_thread.submit(b_lc0.lc0_features(), b_lc0, x, mask, vwht, 1.0)
                    xc0_idxs = b_lc0.moves_to_indices(ucis_now)
                    lc0_meta.append(('main', aux, Y, is_white, list(zip(ucis_now, xc0_idxs))))

                    n_pv = 0
                    seen_fens = set()
                    for pv_seq in aux.get('pv_seqs', []):
                        b_pv = b_lc0.clone()
                        for pv_mv in pv_seq:
                            if b_pv.is_terminal():
                                break
                            b_pv.push_uci(pv_mv)
                            if b_pv.is_terminal():
                                break
                            fen = b_pv.fen(include_counters=False)
                            if fen in seen_fens:
                                continue
                            seen_fens.add(fen)
                            self.lc0_thread.submit(
                                b_pv.lc0_features(),
                                b_pv,
                                self.encode_board(b_pv),
                                b_pv.legal_move_mask(),
                                vwht, 1.0,
                            )
                            lc0_meta.append(('pv', None, None, None, []))
                            n_pv += 1
                    sc['lc0_pv'] += n_pv
                    scw['lc0_pv'] += n_pv
                b_lc0.push_uci(mv)

            self.lc0_thread.flush()
            w = cfg.lc0_enrich_weight
            for (kind, aux, Y, is_white, uci_flat), result in zip(lc0_meta, self.lc0_thread.drain()):
                x, mask, policy, lc0_wdl, vwht, pwht = result
                self.training_data.append((x, mask, policy, lc0_wdl, vwht * w, pwht * w))
                if kind == 'main' and aux is not None:
                    lc0_pol = policy
                    lc0_policy_dict = {u: float(lc0_pol[i]) for u, i in uci_flat}
                    lc0_wdl_stm = lc0_wdl
                    nn_wdl = aux.get('nn_wdl')
                    if nn_wdl is not None:
                        nn_wdl_stm = np.array(nn_wdl if is_white else
                            [nn_wdl[2], nn_wdl[1], nn_wdl[0]], dtype=np.float32)
                    else:
                        nn_wdl_stm = None
                    stm_sign = 1.0 if is_white else -1.0
                    nn_v = aux.get('nn_value')
                    self.pending_lc0_metrics.append({
                        'lc0_wdl': lc0_wdl_stm,
                        'lc0_policy_dict': lc0_policy_dict,
                        'target_wdl': Y,
                        'nn_wdl': nn_wdl_stm,
                        'nn_raw_priors': aux.get('nn_raw_priors', []),
                        'candidate_visits': aux.get('candidate_visits', []),
                        'nn_value_stm': float(np.clip(nn_v * stm_sign, -1.0, 1.0)) if nn_v is not None else None,
                        'sf_cp': aux.get('sf_cp'),
                        'result_z_stm': aux.get('result_z_stm'),
                    })
        
        self.accumulate_collar_stats(n_triggers, n_diff, len(pending))

        cols = [
            'move_num', 'played_move', 'most_visited_move', 'best_move',
            'best_cp', 'delta', 'played_cp',
            'best_absolute', 'played_absolute', 'stm', 'loss', 'stop_reason', 'sims',
        ]

        out_df = pd.DataFrame(rows, columns=cols)
        out_df['played_best_move'] = out_df['delta'] <= 0
        out_df['rescore_kl']  = out_df['move_num'].map(kl_map)
        out_df['rescore_ce']  = out_df['move_num'].map(ce_map)
        out_df['rescore_mse'] = out_df['move_num'].map(mse_map)

        overall_bmr = out_df['played_best_move'].mean() if len(out_df) else np.nan

        out = {
            'plies': n_plies,
            'overall_cpl': rnd(cpl_s / n_plies, 3) if n_plies else np.nan,
            'overall_best_move_rate': overall_bmr,
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
                'bmr': out_df.loc[mask_st, 'played_best_move'].mean() if n_st else np.nan,
                'sims': out_df.loc[mask_st, 'sims'].mean() if n_st else np.nan
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
        W = 12

        def col(val):
            return f"  {val:>{W}}  |"

        def pct(acc, st):
            total_n = sum(acc[s]['n'] for s in stops)
            return acc[st]['n'] / total_n if total_n else 0.0

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
            hdr  = f"{RS}  {'':<12} |"
            crow = f"{RS}  {label:<12} |"
            brow = f"{RS}  {label:<12} |"
            srow = f"{RS}  {'avg sims':<12} |"
            for st in stops:
                hdr  += col(f"{st} ({pct(acc, st):.0%})")
                crow += col(f"CPL {cpl(acc, st):5.2f}")
                brow += col(f"BMR {bmr(acc, st):.3f}")
                srow += col(avg_sims(acc, st))
            return hdr, srow, crow, brow

        wh, ws, wc, wb = rows("last 100", self.window_stop)
        th, _,  tc, tb = rows("overall",  self.total_stop)
        print(wh)
        print(ws)
        print(wc)
        print(wb)
        print()
        print(th)
        print(tc)
        print(tb)

        for st in stops:
            self.window_stop[st] = {'n': 0, 'cpl': 0.0, 'bmr': 0.0, 'sims': 0.0}

    def print_collar_stats(self):
        dry = "" if self.config.use_collar_rescoring else " [dry]"
        tag = f"Collar{dry}"
        W = 10

        def col(val):
            return f"  {val:>{W}}  |"

        def row(label, s):
            pos_pct = s['positions'] / s['total_pos'] if s['total_pos'] else 0.0
            games_str = f"{s['games']}/{s['seen']}"
            pos_str = f"{s['positions']} ({pos_pct:.1%})"
            return (f"{RS}  {label:<12} |"
                    + col(s['triggers'])
                    + col(games_str)
                    + col(pos_str))

        hdr = (f"{RS}  {tag:<12} |"
               + col("trigs")
               + col("games")
               + col("pos (%)"))
        w = self.collar_window
        print(hdr)
        print(row(f"batch {w['seen']:>3}", w))

        if len(self.collar_history) >= 10:
            combined = {
                'games': 0, 'triggers': 0, 'positions': 0,
                'total_pos': 0, 'seen': 0
            }
            for h in self.collar_history[-10:]:
                for k in combined:
                    combined[k] += h[k]
            print(row("last 300", combined))

    def print_sample_stats(self):
        cats = ("cpl", "kl", "ce", "rng")
        labels = ("CPL", " KL", " CE", "rng")
        W = 10

        def col(val):
            return f"  {val:>{W}}  |"

        def row(label, sc):
            acc = sc['accepted']
            ratio = acc / sc['total'] if sc['total'] else 0.0
            lbl = f"{label} {ratio:.0%}"
            out = f"{RS}  {lbl:<12} |"
            for c in cats:
                val = f"{sc[c] / acc:.1%}" if acc else "--"
                out += col(val)
            return out

        hdr = f"{RS}  {'':<12} |" + "".join(col(lbl) for lbl in labels)
        print(hdr)
        print(row("batch", self.sample_counts_window))
        print(row("total", self.sample_counts))
        print()

        ehdr = (f"{RS}  {'enrich':<12} |"
                + "".join(col(lbl) for lbl in ("blunder", "pv", "enrich", "lc0-%")))
        print(ehdr)

        def enrich_row(label, sc):
            lc0_total  = sc['lc0_blunder'] + sc['lc0_pv'] + sc['lc0_enrich']
            total_tr   = sc['accepted'] + sc['lc0_pv'] + sc['lc0_enrich']
            enrich_pct = f"{lc0_total / total_tr * 100:.1f}%" if total_tr else "--"
            return (f"{RS}  {label:<12} |"
                    + col(sc['lc0_blunder'])
                    + col(sc['lc0_pv'])
                    + col(sc['lc0_enrich'])
                    + col(enrich_pct))

        print(enrich_row("batch", self.sample_counts_window))
        print(enrich_row("total", self.sample_counts))

        # LC0 output: how many positions have flushed into training_data
        if self.lc0_thread is not None:
            lst = self.lc0_thread.stats()
            print(f"{RS}  lc0 output: {lst['samples']} flushed"
                  f"  {lst['inferences']} batches"
                  f"  {lst['pending']} pending")

        zero_sc = lambda: {
            'total': 0, 'accepted': 0, 'kl': 0, 'ce': 0, 'cpl': 0, 'rng': 0,
            'lc0_blunder': 0, 'lc0_pv': 0, 'lc0_enrich': 0,
        }
        self.sample_counts_window = zero_sc()

    def aggregate_metrics(self, epoch, vscale, progress_csv_path, size=None):
        if len(self.pending_metrics) < 0.8 * self.config.retrain_size:
            return

        chunk_size = size if size is not None else len(self.pending_metrics)
        chunk = self.pending_metrics[:chunk_size]
        self.pending_metrics = self.pending_metrics[chunk_size:]

        nn_vals_stm, target_ys, sf_cps, result_zs = [], [], [], []
        mol_vals, policy_samples = [], []
        nn_wdls, target_wdls = [], []

        for m in chunk:
            nn_v = m.get('nn_value')
            if nn_v is None:
                continue
            stm_sign = 1.0 if m['stm'] else -1.0
            nn_stm = np.clip(nn_v * stm_sign, -1.0, 1.0)
            nn_vals_stm.append(nn_stm)
            target_ys.append(m['target_y'])
            sf_cps.append(m['sf_cp'])
            result_zs.append(m['result_z_stm'])
            mol = m.get('mass_on_legal')
            mol_vals.append(mol if mol is not None else float('nan'))
            priors_map = {u: p for u, p in m.get('nn_raw_priors', [])}
            if m.get('policy_eligible', True):
                policy_samples.append((priors_map, m.get('candidate_visits', [])))
            nn_wdl = m.get('nn_wdl')
            tgt_wdl = m.get('target_wdl')
            if nn_wdl is not None and tgt_wdl is not None:
                if not m['stm']:  # white-POV -> STM-POV for black: swap win/loss
                    nn_wdl = (nn_wdl[2], nn_wdl[1], nn_wdl[0])
                nn_wdls.append(nn_wdl)
                target_wdls.append(tgt_wdl)

        nn_vals_stm = np.array(nn_vals_stm, dtype=np.float32)
        target_ys   = np.array(target_ys,   dtype=np.float32)
        sf_cps      = np.array(sf_cps,      dtype=np.float32)
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

        if nn_wdls:
            nn_wdl_arr  = np.array(nn_wdls,     dtype=np.float64)
            tgt_wdl_arr = np.array(target_wdls, dtype=np.float64)
            eps = 1e-7
            nn_wdl_arr = np.clip(nn_wdl_arr, eps, 1.0 - eps)
            nn_wdl_arr /= nn_wdl_arr.sum(axis=1, keepdims=True)
            val_ce = -np.mean(np.sum(tgt_wdl_arr * np.log(nn_wdl_arr), axis=1))
            wdl_bias = np.mean(nn_wdl_arr - tgt_wdl_arr, axis=0)
            ce_components = -np.mean(tgt_wdl_arr * np.log(nn_wdl_arr), axis=0)
        else:
            val_ce = float('nan')
            wdl_bias = np.array([float('nan')] * 3)
            ce_components = np.array([float('nan')] * 3)

        stats = {'value_mse': val_mse, 'value_corr': val_corr, 'value_ce': val_ce}
        stats.update(pol_stats)
        stats['mass_on_legal'] = mol_est
        stats['wdl_bias_w'] = float(wdl_bias[0])
        stats['wdl_bias_d'] = float(wdl_bias[1])
        stats['wdl_bias_l'] = float(wdl_bias[2])
        stats['ce_w'] = float(ce_components[0])
        stats['ce_d'] = float(ce_components[1])
        stats['ce_l'] = float(ce_components[2])
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

        from chessbot.plot_utils import plot_validation
        plot_path = os.path.join(os.path.dirname(progress_csv_path), "validation_latest.png")
        plot_validation(
            epoch=epoch,
            plot_path=plot_path,
            nn_vals_stm=nn_vals_stm,
            target_ys=target_ys,
            sf_cps=sf_cps,
            result_zs=result_zs,
            val_mse=val_mse,
            val_corr=val_corr,
            val_ce=val_ce,
            nn_wdl_arr=nn_wdl_arr if nn_wdls else None,
            tgt_wdl_arr=tgt_wdl_arr if nn_wdls else None,
            wdl_bias=wdl_bias,
            ce_components=ce_components,
        )
        print(f"[metrics] plot saved")

        if self.pending_lc0_metrics and self.config.lc0_enrich_frac > 0:
            lc0_csv  = progress_csv_path.replace('eval_progress.csv', 'lc0_metrics.csv')
            lc0_plot = os.path.join(os.path.dirname(progress_csv_path), 'lc0_validation_latest.png')
            self.aggregate_lc0_metrics(epoch, lc0_csv, lc0_plot)

    def aggregate_lc0_metrics(self, epoch, lc0_csv_path, lc0_plot_path):
        chunk = self.pending_lc0_metrics
        self.pending_lc0_metrics = []

        lc0_wdls, nn_wdls, tgt_wdls = [], [], []
        lc0_val_stms, nn_vals_stm, tgt_ys_scalar = [], [], []
        sf_cps, result_zs = [], []
        policy_samples_lc0 = []
        policy_align_samples = []   # (lc0_pol_dict, xc0_priors_dict, xc0_visits_dict)

        for m in chunk:
            if m['nn_wdl'] is None:
                continue
            lc0_wdl = m['lc0_wdl']
            tgt_wdl = m['target_wdl']
            lc0_wdls.append(lc0_wdl)
            nn_wdls.append(m['nn_wdl'])
            tgt_wdls.append(tgt_wdl)
            lc0_val_stms.append(float(lc0_wdl[0] - lc0_wdl[2]))
            tgt_ys_scalar.append(float(tgt_wdl[0] - tgt_wdl[2]))
            nn_vals_stm.append(m['nn_value_stm'])
            sf_cps.append(m['sf_cp'])
            result_zs.append(m['result_z_stm'])
            pol_d = m.get('lc0_policy_dict')
            cv    = m.get('candidate_visits')
            nr    = m.get('nn_raw_priors', [])
            if pol_d and cv:
                policy_samples_lc0.append((pol_d, cv))
            if pol_d and nr:
                xc0_priors = {u: p for u, p in nr}
                xc0_visits = {u: float(c) for u, c in cv} if cv else {}
                policy_align_samples.append((pol_d, xc0_priors, xc0_visits))

        if not lc0_wdls:
            return

        eps = 1e-7
        lc0_arr = np.clip(np.array(lc0_wdls, dtype=np.float64), eps, 1 - eps)
        lc0_arr /= lc0_arr.sum(axis=1, keepdims=True)
        nn_arr  = np.clip(np.array(nn_wdls,  dtype=np.float64), eps, 1 - eps)
        nn_arr  /= nn_arr.sum(axis=1, keepdims=True)
        tgt_arr = np.array(tgt_wdls, dtype=np.float64)

        ce_lc0_true = float(-np.mean(np.sum(tgt_arr * np.log(lc0_arr), axis=1)))
        ce_xc0_lc0  = float(-np.mean(np.sum(lc0_arr * np.log(nn_arr),  axis=1)))
        bias_lc0_true = np.mean(lc0_arr - tgt_arr, axis=0)
        bias_xc0_lc0  = np.mean(nn_arr  - lc0_arr, axis=0)
        ce_comp_lc0_true = -np.mean(tgt_arr * np.log(lc0_arr), axis=0)
        ce_comp_xc0_lc0  = -np.mean(lc0_arr * np.log(nn_arr),  axis=0)

        lc0_val_stms  = np.array(lc0_val_stms,  dtype=np.float32)
        tgt_ys_scalar = np.array(tgt_ys_scalar, dtype=np.float32)
        nn_vals_stm   = np.array([v if v is not None else float('nan') for v in nn_vals_stm], dtype=np.float32)
        sf_cps        = np.array([v if v is not None else float('nan') for v in sf_cps],      dtype=np.float32)
        result_zs     = np.array([v if v is not None else float('nan') for v in result_zs],   dtype=np.float32)

        valid_v = ~np.isnan(lc0_val_stms) & ~np.isnan(tgt_ys_scalar)
        lc0_value_mse  = float(np.mean((lc0_val_stms[valid_v] - tgt_ys_scalar[valid_v]) ** 2)) if valid_v.any() else float('nan')
        lc0_value_corr = float(np.corrcoef(lc0_val_stms[valid_v], tgt_ys_scalar[valid_v])[0, 1]) if valid_v.sum() > 1 else float('nan')

        lc0_pol_stats = batch_policy_metrics_from_priors(
            policy_samples_lc0, self.config.uniform_eps, self.config.prior_clip_max)

        align_stats = self.compute_policy_align_metrics(policy_align_samples)

        row = {
            'model_epoch': epoch, 'n_samples': len(lc0_wdls),
            'lc0_value_mse':      round(lc0_value_mse,  5),
            'lc0_value_corr':     round(lc0_value_corr, 5),
            'lc0_value_ce':       round(ce_lc0_true,    5),
            'lc0_vs_true_bias_w': round(float(bias_lc0_true[0]), 5),
            'lc0_vs_true_bias_d': round(float(bias_lc0_true[1]), 5),
            'lc0_vs_true_bias_l': round(float(bias_lc0_true[2]), 5),
            'lc0_policy_ce':      round(lc0_pol_stats.get('policy_ce',   float('nan')), 5),
            'lc0_uniform_ce':     round(lc0_pol_stats.get('uniform_ce',  float('nan')), 5),
            'lc0_ce_gain':        round(lc0_pol_stats.get('ce_gain',     float('nan')), 5),
            'lc0_top1_exact':     round(lc0_pol_stats.get('top1_exact',  float('nan')), 5),
            'lc0_avg_top_prob':   round(lc0_pol_stats.get('avg_top_prob',float('nan')), 5),
            'lc0_top1_mass':      round(lc0_pol_stats.get('top1_mass',   float('nan')), 5),
            'lc0_top3_mass':      round(lc0_pol_stats.get('top3_mass',   float('nan')), 5),
            'lc0_top5_mass':      round(lc0_pol_stats.get('top5_mass',   float('nan')), 5),
            'xc0_vs_lc0_ce':     round(ce_xc0_lc0, 5),
            'xc0_vs_lc0_bias_w': round(float(bias_xc0_lc0[0]), 5),
            'xc0_vs_lc0_bias_d': round(float(bias_xc0_lc0[1]), 5),
            'xc0_vs_lc0_bias_l': round(float(bias_xc0_lc0[2]), 5),
            # policy alignment: lc0 priors vs xc0 priors/visits
            'kl_lc0_xc0_priors':      round(align_stats['kl_lc0_xc0_priors'],      5),
            'ce_lc0gt_xc0_visits':    round(align_stats['ce_lc0gt_xc0_visits'],    5),
            'prior_top1_agree':       round(align_stats['prior_top1_agree'],        5),
            'prior_top3_agree':       round(align_stats['prior_top3_agree'],        5),
            'prior_top5_agree':       round(align_stats['prior_top5_agree'],        5),
        }
        row_df = pd.DataFrame([row])
        if os.path.exists(lc0_csv_path):
            all_df = pd.concat([pd.read_csv(lc0_csv_path), row_df], ignore_index=True)
        else:
            all_df = row_df
        all_df.round(5).to_csv(lc0_csv_path, index=False)

        from chessbot.plot_utils import plot_lc0_validation
        plot_lc0_validation(
            epoch=epoch,
            plot_path=lc0_plot_path,
            lc0_arr=lc0_arr,
            nn_arr=nn_arr,
            tgt_arr=tgt_arr,
            lc0_vals_stm=lc0_val_stms,
            nn_vals_stm=nn_vals_stm,
            tgt_ys=tgt_ys_scalar,
            sf_cps=sf_cps,
            result_zs=result_zs,
            ce_lc0_true=ce_lc0_true,
            ce_xc0_lc0=ce_xc0_lc0,
            bias_lc0_true=bias_lc0_true,
            bias_xc0_lc0=bias_xc0_lc0,
            ce_comp_lc0_true=ce_comp_lc0_true,
            ce_comp_xc0_lc0=ce_comp_xc0_lc0,
            lc0_value_mse=lc0_value_mse,
            lc0_value_corr=lc0_value_corr,
        )

    @staticmethod
    def compute_policy_align_metrics(samples):
        eps = 1e-12
        nan = float('nan')
        kl_vals, ce_vis_vals = [], []
        top1, top3, top5 = [], [], []

        for lc0_pol, xc0_priors, xc0_visits in samples:
            ucis = list(lc0_pol.keys())
            if not ucis:
                continue

            lc0_p = np.array([lc0_pol.get(u, 0.0) for u in ucis], dtype=np.float64)
            lc0_s = lc0_p.sum()
            if lc0_s <= 0:
                continue
            lc0_p /= lc0_s

            xc0_p = np.array([xc0_priors.get(u, 0.0) for u in ucis], dtype=np.float64)
            xc0_s = xc0_p.sum()
            if xc0_s > 0:
                xc0_p /= xc0_s
                # KL(lc0 || xc0_priors)
                kl_vals.append(float(np.sum(lc0_p * np.log((lc0_p + eps) / (xc0_p + eps)))))
                # top-k agreement: lc0 top-1 in xc0 top-k
                lc0_top1_idx = int(np.argmax(lc0_p))
                xc0_sorted   = np.argsort(-xc0_p)
                top1.append(float(xc0_sorted[0] == lc0_top1_idx))
                top3.append(float(lc0_top1_idx in xc0_sorted[:min(3, len(xc0_sorted))]))
                top5.append(float(lc0_top1_idx in xc0_sorted[:min(5, len(xc0_sorted))]))

            # CE(lc0 as GT vs xc0 visits)
            xc0_v = np.array([xc0_visits.get(u, 0.0) for u in ucis], dtype=np.float64)
            xc0_vs = xc0_v.sum()
            if xc0_vs > 0:
                xc0_v /= xc0_vs
                ce_vis_vals.append(float(-np.sum(lc0_p * np.log(xc0_v + eps))))

        def mean_or_nan(lst):
            return float(np.mean(lst)) if lst else nan

        return {
            'kl_lc0_xc0_priors':   mean_or_nan(kl_vals),
            'ce_lc0gt_xc0_visits': mean_or_nan(ce_vis_vals),
            'prior_top1_agree':    mean_or_nan(top1),
            'prior_top3_agree':    mean_or_nan(top3),
            'prior_top5_agree':    mean_or_nan(top5),
        }

    def push_analyzed(self, report=True):
        # safeguard here
        if not len(self.analyzed_results):
            return
        
        run_dir = self.config.run_dir
        outp, c, b, plies = save_analysis_chunk_simple(run_dir, self.analyzed_results)
        self.n_saved += 1
        self.total_cpl_plies += c * plies
        self.total_bmr_plies += b * plies
        self.total_plies += plies

        self.last_10_cpls.append(c)
        self.last_10_cpls = self.last_10_cpls[-10:]
        self.last_10_bmrs.append(b)
        self.last_10_bmrs = self.last_10_bmrs[-10:]
        self.last_10_plies.append(plies)
        self.last_10_plies = self.last_10_plies[-10:]

        if report:
            if len(self.last_10_cpls) >= 10:
                w = np.array(self.last_10_plies, dtype=np.float64)
                last_10_avg_c = np.dot(self.last_10_cpls, w) / w.sum()
                last_10_avg_b = np.dot(self.last_10_bmrs, w) / w.sum()
                print(
                    f"{RS} {'Last 10 avg:':<16} CPL {last_10_avg_c:.3f}",
                    f"BMR {last_10_avg_b:.3f}"
                )

            if self.n_saved >= 2:
                cpl_mean = self.total_cpl_plies / self.total_plies
                bmr_mean = self.total_bmr_plies / self.total_plies
                print(
                    f"{RS} {'Overall stats:':<16} CPL {cpl_mean:.3f}",
                    f"BMR {bmr_mean:.3f}"
                )

            n_tot = self.games_processed
            if n_tot > self.config.rescore_analyze_batch and self.start_time is not None:
                elapsed = time.time() - self.start_time
                games_hr = n_tot / elapsed * 3600
                moves_sec = self.total_plies / elapsed
                print(f"{RS} Total Games: {n_tot}"
                      f" ({games_hr:.1f} games/hr, {moves_sec:.2f} moves/sec)"
                      f" ({self.n_sf_threads} workers, depth={self.current_depth})")
            
            cache_stats = self.cache.stats()
            avg_compute_ms = (
                1000 * self.sf_compute_time / self.sf_compute_count
                if self.sf_compute_count else 0.0
            )
            sz = cache_stats['size']
            backlog = len(self.intake) + len(self.pending)
            compute_str = f"{avg_compute_ms:.0f}ms"
            W = 10

            def col(val):
                return f"  {val:>{W}}  |"

            sf_cols = [
                ("positions", str(self.n_sf_submitted)),
                ("hits",      str(self.n_cache_hits)),
                ("cache sz",  str(sz)),
                ("compute",   compute_str),
                ("backlog",   str(backlog)),
            ]
            sf_hdr = (f"{RS}  {'SF':<12} |"
                      + "".join(col(h) for h, _ in sf_cols))
            sf_row = (f"{RS}  {'':<12} |"
                      + "".join(col(v) for _, v in sf_cols))
            print(sf_hdr)
            print(sf_row)
            self.sf_compute_time = 0.0
            self.sf_compute_count = 0
            print()

            self.print_collar_stats()
            print()
            self.print_sample_stats()

            n, el = self.tscale_n, self.tscale_eligible
            if el > 0:
                frac = n / el
                avg_iters = self.tscale_iters / n if n else 0
                avg_ne_b  = self.tscale_ne_before / n if n else 0
                avg_ne_a  = self.tscale_ne_after / n if n else 0
                avg_T     = self.tscale_best_T / n if n else 0
                avg_us    = self.tscale_time / n * 1e6 if n else 0
                print(f"{RS}  tscale: {n}/{el} ({frac:.1%})  ne {avg_ne_b:.3f}->{avg_ne_a:.3f}  delta={avg_ne_a-avg_ne_b:+.3f}  avg_T={avg_T:.3f}")
                print(f"{RS}  tscale: avg_iters={avg_iters:.1f}  avg={avg_us:.1f}us  total={self.tscale_time*1000:.1f}ms")
            self.tscale_n = self.tscale_eligible = self.tscale_iters = 0
            self.tscale_ne_before = self.tscale_ne_after = self.tscale_best_T = self.tscale_time = 0.0

            w_this = self.written_this_round
            wtot = self.written_total
            print(f"{RS} Training samples this round: {w_this} | total: {wtot}")
            print()

        self.collar_history.append(dict(self.collar_window))
        self.collar_history = self.collar_history[-10:]
        zero_collar = {
            'games': 0, 'triggers': 0, 'positions': 0,
            'total_pos': 0, 'seen': 0
        }
        
        self.collar_window = dict(zero_collar)

        self.analyzed_results = []

# helpers
def collar_z_map(eval_trace, game_result, cfg, game_id=None):
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

    threshold = cfg.collar_threshold_cp
    n_consec = max(1, cfg.collar_n_consec)
    reset_cp = cfg.collar_reset_cp

    if not eval_trace:
        return {}, 0

    plies = [p for p, _ in eval_trace]
    evals = [e for _, e in eval_trace]
    n = len(evals)

    collar_state = [None] * n
    collar = None
    count = 0
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
                for j in range(i - count + 1, i + 1):
                    collar_state[j] = 'white'
                collar = 'white'
                count = 0
            elif count <= -n_consec:
                for j in range(i + count + 1, i + 1):
                    collar_state[j] = 'black'
                collar = 'black'
                count = 0
        else:
            broken = (
                (collar == 'white' and ev < reset_cp) or
                (collar == 'black' and ev > -reset_cp)
            )

            if broken:
                collar = None
                count = 0
                n_triggers += 1
            else:
                collar_state[i] = collar

    eff_z = {}

    for i, ply in enumerate(plies):
        if collar_state[i] == 'white':
            eff_z[ply] = 1.0
        elif collar_state[i] == 'black':
            eff_z[ply] = -1.0
        else:
            eff_z[ply] = float(game_result)

    if n_triggers == 0 or all(v == float(game_result) for v in eff_z.values()):
        return {}, 0

    return eff_z, n_triggers


def value_weight_for_game(cfg, is_draw):
    return cfg.draw_value_scale if is_draw else 1.0


def z_to_wdl(z_stm):
    if z_stm > 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif z_stm < 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def blend_wdl(z_stm, best_wdl_white_pov, is_white):
    """50% game result (one-hot), 50% Xerces search WDL."""
    z_wdl = z_to_wdl(z_stm)

    if best_wdl_white_pov is not None:
        bw = np.array(best_wdl_white_pov, dtype=np.float32)
        if not is_white:
            bw = bw[[2, 1, 0]]
    else:
        bw = z_wdl

    return (0.5 * z_wdl + 0.5 * bw).astype(np.float32)


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
    om = [r.get("overall_cpl", np.nan) for r in results]
    ob = [r.get("overall_best_move_rate", np.nan) for r in results]

    return {
        "games": len(results),
        "avg_overall_mean_cpl": round(safe_mean(om), 3),
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
            "overall_cpl": analysis_out.get("overall_cpl"),
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
    plies = len(df_all)

    print(f"{RS} Saving {len(batch)} analyzed games")
    print(f"{RS} {'Batch stats:':<16} CPL {cpl:.3f} BMR {bmr:.3f}")

    fname = f"{int(time.time())}_{uuid.uuid4().hex}.pkl"
    outp = os.path.join(staging, fname)

    with open(outp, "wb") as f:
        pickle.dump(chunk_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return outp, cpl, bmr, plies


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


