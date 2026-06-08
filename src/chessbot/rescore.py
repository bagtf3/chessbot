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
from chessbot.lc0_utils import lc0_logits_to_xc0_batch

RS = "[rescore]"
ANALYZE_PKL = "analyze_results_combined.pkl"
EVICTION_WINDOW = 5000
EVICTION_MAX_SIZE = 80000
C = 0.9699
D = d = np.arctanh(0.5)

def lc0_table_index(features_uint8):
    """Derive LC0->XC0 table index (0-3) from suffix planes of lc0_features() output."""
    us_ooo = int(features_uint8[104, 0, 0])
    us_oo  = int(features_uint8[105, 0, 0])
    stm    = int(features_uint8[108, 0, 0])  # 0=white, 1=black
    return (us_ooo | us_oo) + 2 * stm


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

    def submit(self, features_uint8, x, mask, vwht, pwht):
        ti = lc0_table_index(features_uint8)
        self.pending.append((features_uint8, ti, x, mask, vwht, pwht))
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

        tis          = np.array([item[1] for item in batch])
        xc0_policies = lc0_logits_to_xc0_batch(logits, tis)

        for i, item in enumerate(batch):
            _, _, x, mask, vwht, pwht = item
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
            f" {others_added:,} other moves merged from {path}"
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
        self.eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
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

        self.blunder_replay_specs = []
        self.blunder_replay_counts = {'winning': 0, 'neutral': 0, 'mismatch': 0}

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

        zero_sc = lambda: {
            'total': 0, 'accepted': 0, 'kl': 0, 'ce': 0, 'cpl': 0, 'rng': 0,
            'lc0_blunder': 0, 'lc0_pv': 0, 'book': 0,
        }
        self.sample_counts = zero_sc()
        self.sample_counts_window = zero_sc()

        self.intake = deque()
        self.pending = {}

        book_path = os.getenv('ENRICHMENT_BOOK_PATH', '')
        self.enrichment_book = None
        if book_path and os.path.exists(book_path):
            with open(book_path, 'rb') as f:
                self.enrichment_book = pickle.load(f)
            print(
                (f"{RS} Enrichment book: {len(self.enrichment_book)} "
                f"positions from {book_path}"))

        lc0_model     = cfg.lc0_distill_model_name or os.getenv('LC0_DISTILL_MODEL', '')
        lc0_trt_cache = os.getenv('LC0_DISTILL_TRT_CACHE', '')
        self.lc0_thread = None
        if lc0_model and lc0_trt_cache:
            from chessbot.lc0_utils import make_lc0_trt_session
            batch_size = cfg.lc0_distill_batch_size
            lc0_onnx = os.path.join(lc0_trt_cache, f'{lc0_model}.onnx')
            sess = make_lc0_trt_session(lc0_onnx, lc0_model, lc0_trt_cache,
                                        opt_batch=batch_size, max_batch=batch_size * 2)
            # Hard-fail if TRT didn't load — silent CPU fallback would silently bottleneck training.
            active = sess.get_providers()[0]
            if active != 'TensorrtExecutionProvider':
                raise RuntimeError(f"{RS} Lc0Thread TRT failed, got {active}. Check CUDA/TRT DLLs in PATH.")
            self.lc0_thread = Lc0Thread(sess, batch_size=batch_size)
            print(f"{RS} Lc0Thread: TRT batch_size={batch_size}")

        self.init_analyzer()

    def maybe_enrich(self, entry, sfen):
        """50% chance to replace policy+Y with enrichment book data for book positions"""
        if self.enrichment_book is None:
            return entry
        book_entry = self.enrichment_book.get(sfen)
        if book_entry is None or random.random() >= 0.5:
            return entry
        x, mask, policy, Y, vwht, pwht = entry
        book_xc0, book_wdl = book_entry
        if not book_xc0.any():
            return entry
        return (x, mask, book_xc0, np.array(book_wdl, dtype=np.float32), vwht, pwht)

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
        reps = 3 if repetitions[short_fen] >= 3 else 0
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
            'vscale'
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
            cpl_s += loss_this
            n_plies += 1

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

            is_blunder = loss_this >= blunder_cp and is_true_blunder

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
            kl = kl_divergence(priors, vis)
            kl_map[i] = kl
            if kl_eligible and do_KL_boost:
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
            pending_aux.append({
                'sfen':            ply['cache_key'][0],
                'pv_ucis':         ply.get('pv_ucis', []) if is_blunder else [],
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

        # collar-adjusted result targets and blunder replay candidates
        eff_z_by_ply, n_triggers, replayable_blunders = collar_z_map(
            eval_trace, result, cfg, self.config.blunder_replay_min_ply, gid
        )

        # look for and handle blunder replays
        for blunder_ply, blunder_type, blunderer_is_white in replayable_blunders:
            if vs_stockfish and blunderer_is_white == sf_color:
                # SF blundered, skip
                self.blunder_replay_counts['mismatch'] += 1
                continue  

            if blunder_type == 'neutral' and random.random() > 0.25:
                continue

            actual_stm_is_white = turn_at.get(blunder_ply)
            if actual_stm_is_white is None:
                self.blunder_replay_counts['mismatch'] += 1
                continue

            self.blunder_replay_specs.append({
                'fen': game_data['start_fen'],
                'moves': game_data['moves_played'][:blunder_ply],
                'stockfish_is_white': actual_stm_is_white,
                'source_game_id': gid
            })

            self.blunder_replay_counts[blunder_type] += 1
            total = sum(self.blunder_replay_counts.values())
            if total % 50 == 0:
                c = self.blunder_replay_counts
                print(
                    f"[blunder replay] tally at {total}:"
                    f" winning={c['winning']} neutral={c['neutral']}"
                    f" mismatch={c['mismatch']}"
                )
        
        n_diff = 0
        hard_accepted = []
        soft_pool = []
        sc = self.sample_counts
        scw = self.sample_counts_window

        # loop 2: compute WDL targets and CE, classify into hard/soft
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
            flags = (hit_kl, hit_ce, hit_cpl)
            if hit_kl or hit_ce or hit_cpl:
                hard_accepted.append((entry, flags, aux))
            else:
                soft_pool.append((entry, aux))

            self.pending_metrics.append({
                **aux,
                'target_y': Y[0] - Y[2],
                'target_wdl': Y,
            })

        start_fen    = game_data['start_fen']
        moves_played = game_data['moves_played']

        # hard-accepted: classify each entry, collect LC0 waypoints for singlepass replay
        lc0_waypoints = {}
        for entry, (hit_kl, hit_ce, hit_cpl), aux in hard_accepted:
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

            if aux['is_blunder'] and self.lc0_thread is not None:
                self.training_data.append(entry)
                lc0_waypoints[aux['ply_i']] = (entry, aux)
                sc['lc0_blunder'] += 1
                scw['lc0_blunder'] += 1
            else:
                enriched = self.maybe_enrich(entry, aux['sfen'])
                self.training_data.append(enriched)
                if enriched is not entry:
                    sc['book'] += 1
                    scw['book'] += 1

        # Single forward pass: replay game from start_fen to collect lc0_features() at each
        # blunder ply. Must use push_uci replay (not Board(fen)) so lc0_features() has full
        # move history for all 8 history planes.
        if lc0_waypoints:
            b_lc0 = Board(start_fen)
            for ply_i, mv in enumerate(moves_played):
                if ply_i in lc0_waypoints:
                    entry, aux = lc0_waypoints[ply_i]
                    got_fen = b_lc0.fen(include_counters=False)
                    if got_fen != aux['sfen']:
                        msg = f"[rescore] FEN mismatch at ply {ply_i}: {got_fen} "
                        msg += f"!= {aux['sfen']}"
                        print(msg)

                    x, mask, _, _, vwht, _ = entry
                    self.lc0_thread.submit(b_lc0.lc0_features(), x, mask, vwht, 1.0)
                    n_pv = 0
                    if aux['pv_ucis']:
                        # pv_ucis = pv[0:4]: pv[0] is SF's best move from
                        # the blunder position, subsequent entries are the continuation.
                        # Must start from pv[0] — skipping it would push a move
                        # for the wrong side and corrupt the board.
                        b_pv = b_lc0.clone()
                        for pv_mv in aux['pv_ucis']:
                            if b_pv.is_terminal():
                                break
                            b_pv.push_uci(pv_mv)
                            if b_pv.is_terminal():
                                break
                            self.lc0_thread.submit(
                                b_pv.lc0_features(),
                                b_pv.encode_64_tokens(),
                                b_pv.legal_move_mask(),
                                vwht, 1.0,
                            )
                            n_pv += 1
                    sc['lc0_pv'] += n_pv
                    scw['lc0_pv'] += n_pv
                b_lc0.push_uci(mv)

        # soft pool: sample up to target acceptance rate
        n_hard = len(hard_accepted)
        n_soft = len(soft_pool)
        n_total = n_hard + n_soft
        target_n = int(cfg.rescore_target_acceptance * n_total)
        n_need = max(0, target_n - n_hard)
        n_sample = min(n_soft, max(int(cfg.rescore_sample_floor * n_soft), n_need))
        if n_sample > 0 and n_soft > 0:
            chosen = np.random.choice(n_soft, size=n_sample, replace=False)
            for i in chosen:
                entry, aux = soft_pool[i]
                enriched = self.maybe_enrich(entry, aux['sfen'])
                self.training_data.append(enriched)
                if enriched is not entry:
                    sc['book'] += 1
                    scw['book'] += 1
                sc['accepted'] += 1
                scw['accepted'] += 1
                sc['rng'] += 1
                scw['rng'] += 1
        
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
                + "".join(col(lbl) for lbl in ("blunder", "pv", "book", "lc0-%")))
        print(ehdr)

        def enrich_row(label, sc):
            blur_pv    = sc['lc0_blunder'] + sc['lc0_pv']
            total_tr   = sc['accepted'] + sc['lc0_pv']
            enrich_n   = sc['book'] + blur_pv
            enrich_pct = f"{enrich_n / total_tr * 100:.1f}%" if total_tr else "--"
            return (f"{RS}  {label:<12} |"
                    + col(sc['lc0_blunder'])
                    + col(sc['lc0_pv'])
                    + col(sc['book'])
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
            'lc0_blunder': 0, 'lc0_pv': 0, 'book': 0,
        }
        self.sample_counts_window = zero_sc()

    def aggregate_metrics(self, epoch, vscale, progress_csv_path, size=None):
        if len(self.pending_metrics) < self.config.retrain_size // 2:
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
def collar_z_map(eval_trace, game_result, cfg, replay_min_ply, game_id=None):
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
        return {}, 0, []

    plies = [p for p, _ in eval_trace]
    evals = [e for _, e in eval_trace]
    n = len(evals)

    collar_state = [None] * n
    collar = None
    count = 0
    segment_start = 0
    n_triggers = 0
    replayable_blunders = []
    neutral_blunders = []
    collar_set_by_non_winner = False
    neutral_position = False
    neutral_count = 0

    for i, ev in enumerate(evals):
        # detect a neutral -> losing deviation
        if neutral_position and abs(ev) >= reset_cp:
            if plies[i] >= replay_min_ply:
                # if the delta is large, assume blunder
                if abs(ev - evals[i-1]) > abs(reset_cp - threshold):
                    # and the current STM loses the game
                    if game_result > 0 if ev > 0 else game_result < 0:
                        neutral_blunders.append((plies[i-1], 'neutral', ev < 0))

        neutral_count = neutral_count + 1 if abs(ev) < reset_cp else 0
        neutral_position = neutral_count >= n_consec

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
                collar_set_by_non_winner = game_result < 1.0
                count = 0
            elif count <= -n_consec:
                for j in range(segment_start, i + 1):
                    collar_state[j] = 'black'
                collar = 'black'
                collar_set_by_non_winner = game_result > -1.0
                count = 0
        else:
            broken = (
                (collar == 'white' and ev < reset_cp) or
                (collar == 'black' and ev > -reset_cp)
            )

            if broken:
                # first, check if replayable based on collar holder and prev eval
                if plies[i] >= replay_min_ply:
                    if collar_set_by_non_winner and abs(evals[i-1]) > threshold:
                        replayable_blunders.append(
                            (plies[i-1], 'winning', collar == 'white')
                        )

                # then continue collar logic
                segment_start = i
                collar = None
                count = 0
                n_triggers += 1

            else:
                collar_state[i] = collar

    eff_z = {}

    # if we detected 0 flips, no updates.
    if not n_triggers:
        return eff_z, n_triggers, neutral_blunders

    for i, ply in enumerate(plies):
        if collar_state[i] == 'white':
            eff_z[ply] = 1.0
        elif collar_state[i] == 'black':
            eff_z[ply] = -1.0
        else:
            eff_z[ply] = float(game_result)

    # sanity check: if all eff_z values collapsed back to game_result, no real flips
    if all(v == float(game_result) for v in eff_z.values()):
        return {}, 0, neutral_blunders

    return eff_z, n_triggers, replayable_blunders + neutral_blunders


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


def reclaim_vram(mb):
    import tensorflow as tf

    bytes_target = mb * 1024 * 1024
    n = max(1, bytes_target // 4)

    with tf.device("/GPU:0"):
        x = tf.ones([n], dtype=tf.float32)
        y = tf.reduce_sum(x)

    _ = y.numpy()
    return True
