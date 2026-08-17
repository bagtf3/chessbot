import os, json, gzip, glob, time
from types import SimpleNamespace
from pathlib import Path
import uuid
import random
import threading
from collections import deque
from queue import Empty

import pickle
import chess, chess.engine

import pandas as pd
import numpy as np

from pyfastchess import Board

from chessbot import SF_LOC
from chessbot.utils import (
    score_cp_stm_pov, score_cp_white_pov, rnd, kl_divergence, cross_entropy,
    calc_entropy, batch_policy_metrics, cp_to_value_tanh, scalar_to_wdl,
)
from chessbot.lc0_utils import lc0_logits_to_xc0_batch, lc0_table_index
from chessbot.replay_buffer import (
    LiveBuffer, SEED_SOURCE, sparsify_policy, stack_policies,
)
from chessbot.opening_counts import (
    COUNTS_FILE, REWEIGHT_INELIGIBLE, OpeningCounts,
)

from chessbot.blunder_replay import (
    BLUNDER_POOL_ENV, BLUNDER_SCENARIOS, is_blunder_candidate,
    load_probe_pool, save_probe_pool, evicted_path, evict_position,
    BLUNDER_REPLAY_EQUIV_CPL, BLUNDER_REPLAY_HISTORY_FILENAME,
    blunder_replay_dir, BLUNDER_POOL_MAX_SIZE,
    BLUNDER_REPLAY_MIN_CACHE_DEPTH, ratchet_store, cached_deep_score,
    BLUNDER_REPLAY_START_MS, BLUNDER_REPLAY_MS_STEP, BLUNDER_REPLAY_MS_MAX_MULT,
)

from chessbot.game_utils import reconcile_game_boards, short_fen

from xerces_training.uci_to_idx import uci_to_idx as UCI_TO_IDX

RS = "[rescore]"
BLUNDER_MERGE_EVERY = 256
# blunder-replay analysis fires purely on collected-row count, not a retrain
# cadence -- this is how many rows accumulate in blunder_replay_probe.jsonl
# before a summary is computed and the file resets to a fresh window
BLUNDER_REPLAY_ANALYSIS_EVERY = 4096
BLUNDER_STAT_KEYS = (
    "total", "missed_mate", "A", "B", "G3", "F", "U",
    "true_blunder", "false_blunder", "inaccuracy", "all_blunders",
)
ANALYZE_PKL = "analyze_results_combined.pkl"
PRETRAIN_CONTINUED_CSV = "eval_progress_pretrain_continued.csv"
# pretraining validated every 20 epochs; the continued file keeps that step
PRETRAIN_EPOCH_STEP = 20
# a ply losing more than this many centipawns counts as a blunder for the
# L<n> rate in the rescore stats line
BLUNDER_CP = 30
# rolling stat window, in push_analyzed chunks (30 games each)
RECENT_WINDOW = 100
# SF rescore search caps. cfg.rescore_movetime_ms is the tunable budget; the
# catch-up trim and the depth backstop stay fixed so there is one knob.

C = 0.9699
D = d = np.arctanh(0.5)

# Validation provenance buckets. seaborn "deep" blue/orange/green, keyed by
# name so a source that is absent this round leaves the others' colors alone.
SOURCE_COLORS = {'xc0': '#4C72B0', 'lc0': '#DD8452', 'historic': '#55A868'}
SOURCE_ORDER = ('xc0', 'lc0', 'historic')


def count_jsonl_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum([1 for _ in f])


def plot_source(source):
    """Collapse a record's provenance tag to a plot bucket. SEED_SOURCE and
    the untagged tfrec.gz records are both pretrain-distribution data, so both
    read as historic; the lc0_pv/lc0_blunder/lc0_enrich split survives only in
    the print_sample_stats counters."""
    if source is None or source == SEED_SOURCE:
        return 'historic'
    if source.startswith('lc0'):
        return 'lc0'
    return 'xc0'


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
        self.pending    = []   # (features_uint8, lc0_idx, xc0_idx, x, vwht, pwht)
        self.results    = []   # completed (x, xc0_policy, lc0_wdl, vwht, pwht)
        self.n_inferences = 0
        self.n_samples    = 0

        # discover input/output names from session metadata once
        inputs  = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.input_name   = inputs[0].name
        self.output_names = [o.name for o in outputs]

    def submit(self, features_uint8, board, x, vwht, pwht):
        ti   = lc0_table_index(features_uint8)
        ucis = board.legal_moves()
        lc0_idx = np.array([UCI_TO_IDX[ti][u.rstrip('n')] for u in ucis], dtype=np.int32)
        xc0_idx = np.array(board.moves_to_indices(ucis), dtype=np.int32)
        self.pending.append((features_uint8, lc0_idx, xc0_idx, x, vwht, pwht))
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
            _, _, _, x, vwht, pwht = item
            # No legal-move mask: lc0_logits_to_xc0_batch scatters into a zeroed
            # (n, 1858) at exactly this board's legal indices, so the row is
            # already zero outside the legal set. The old `* mask` was a no-op
            # that also promoted float32 -> float64, since the mask arrived as a
            # python list. The renormalise stays -- it still matters when a
            # position has no legal moves, and when two ucis collide on one
            # xc0 index.
            policy = xc0_policies[i].copy()
            s = policy.sum()
            if s > 0:
                policy /= s
            self.results.append((
                x,
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


class SFRescoreThread:
    """
    SF worker that processes one game at a time for TT locality.
    submit_game() enqueues a whole game; the thread processes positions
    sequentially in ply order and returns a batch result.

    positions: [(ply_idx, board_copy, xerces_uci), ...]
    """

    def __init__(self, sf_game_q, sf_res_q, cfg):
        self.game_q = sf_game_q
        self.res_q  = sf_res_q

        # SF stops on whichever binds first: the time budget or MAX_DEPTH.
        # MAX_DEPTH is a backstop, not the primary limit -- a depth-bound call
        # returns immediately instead of spending the rest of the budget, which
        # mostly catches warm-TT positions (endgames, the single-root-move
        # rerun) that would otherwise run away.
        self.max_depth = 24
        self.base_ms = cfg.rescore_movetime_ms
        self.throttled_ms = max(10, self.base_ms - 15)
        self.movetime_ms = self.base_ms
        self.sf_config = {'Hash': 256}
        self.stop_ev = threading.Event()
        self.t = None
        self.eng = None

    def submit_game(self, gid, positions, movetime_ms=None):
        info = {"movetime_ms": movetime_ms} if movetime_ms is not None else {}
        self.game_q.put((gid, positions, info))

    def update_config(self, cfg):
        self.base_ms = cfg.rescore_movetime_ms
        self.throttled_ms = max(10, self.base_ms - 15)
        if self.movetime_ms > self.base_ms:
            self.movetime_ms = self.base_ms

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
            gid, positions, info = item
            results = []

            movetime_ms = info.get("movetime_ms", self.movetime_ms)

            pass1_limit = chess.engine.Limit(
                time= movetime_ms/ 1000.0, depth=self.max_depth)

            # longer search for pass2 because Xc0 move may be better
            pass2_limit = chess.engine.Limit(
                time=(movetime_ms + 20)/ 1000.0, depth=self.max_depth)
            
            try:
                for ply_idx, board, xerces_uci in positions:
                    elapsed = 0.0
                    depths = []

                    # pass 1: full best-move analysis
                    t0 = time.time()
                    info = self.eng.analyse(
                        board, pass1_limit, info=chess.engine.INFO_ALL)
                    
                    elapsed += time.time() - t0
                    depth = info.get('depth', 0)
                    best_uci = str(info['pv'][0])
                    pv_ucis  = [str(m) for m in info.get('pv', [])[:3]]
                    best_cp  = score_cp_stm_pov(info['score'])
                    best_abs = score_cp_white_pov(info['score'], clipped=False)

                    # pass 2: score xerces's move only if it differs from best
                    rerun = xerces_uci != best_uci
                    if rerun:
                        t0 = time.time()
                        info2 = self.eng.analyse(
                            board, pass2_limit,
                            root_moves=[chess.Move.from_uci(xerces_uci)],
                            info=chess.engine.INFO_ALL,
                        )
                        elapsed  += time.time() - t0
                        depths.append(info2.get('depth', 0))
                        played_cp  = score_cp_stm_pov(info2['score'])
                        played_abs = score_cp_white_pov(info2['score'], clipped=False)
                    else:
                        played_cp  = best_cp
                        played_abs = best_abs
                        depths.append(depth)

                    results.append({
                        'ply_idx':    ply_idx,
                        'best_uci':   best_uci,
                        'pv_ucis':    pv_ucis,
                        'best_cp':    best_cp,
                        'best_abs':   best_abs,
                        'played_cp':  played_cp,
                        'played_abs': played_abs,
                        'elapsed':    elapsed,
                        'best_depth': depth,      # pass 1, the free search
                        'depth':      depths[0],  # pass 2, the played move
                        'depths':     depths,
                        'rerun':      rerun,
                    })
            
            except Exception as e:
                self.res_q.put(('__error__', e))
                self.stop_ev.set()
                return

            # whole game done — push batch so Rescorer can finalize in one shot
            self.res_q.put((gid, results))


class Rescorer(object):
    def __init__(self, cfg, sf_game_q, sf_res_q):
        self.config   = cfg
        self.game_q   = sf_game_q
        self.res_q    = sf_res_q

        self.opening_counts = OpeningCounts(os.path.join(cfg.run_dir, COUNTS_FILE))
        loaded = self.opening_counts.load()
        n = self.opening_counts.occupancy()
        print(f"{RS} opening counts: {'loaded' if loaded else 'cold start'}, "
              f"{n} slots occupied")

        self.live_buffer = LiveBuffer(cfg.primary_buffer_dir, counts=self.opening_counts)
        self.analyzed_results = []
        self.closed = False

        # pool itself is never cached here -- it's passed into tick() each
        # call from the one floating copy in run_selfplay.py's main(), since
        # this class only ever mutates it (new candidates, replay evictions),
        # never samples from it. blunder_pool_path is just a string, and
        # n_pool_changes is this class's own save-cadence counter, so both
        # are safe to hold as real instance state.
        self.blunder_pool_path = os.environ.get(BLUNDER_POOL_ENV) or None
        self.n_pool_changes = 0
        self.pool_save_times = []

        self.start_time = None  # set on first game to exclude idle startup time
        self.games_seen = set()
        self.games_processed = 0
        self.written_total = 0
        self.written_this_round = 0
        self.n_saved = 0
        self.total_cpl_plies = 0.0
        self.total_bmr_plies = 0.0
        self.total_blunder_plies = 0.0
        self.total_kl_plies = 0.0
        self.total_ce_plies = 0.0
        self.total_plies = 0
        self.total_kl_valid_plies = 0.0
        self.total_ce_valid_plies = 0.0

        self.n_sf_submitted = 0
        self.sf_compute_time = 0.0
        self.sf_compute_count = 0
        self.sf_call_count = 0
        self.sf_rerun_count = 0
        self.n_sf_threads = cfg.rescore_n_sf_threads
        self.sf_depth_sum = 0
        self.sf_depth_count = 0
        self.current_movetime_ms = 0
        self.recent_cpls = []
        self.recent_bmrs = []
        self.recent_blunders = []
        self.recent_kls = []
        self.recent_ces = []
        self.recent_plies = []
        self.recent_kl_plies = []
        self.recent_ce_plies = []

        zero_stop = lambda: {'n': 0, 'cpl': 0.0, 'bmr': 0.0, 'sims': 0.0}
        self.total_stop = {st: zero_stop() for st in ("full", "rsc", "jsd")}
        self.window_stop = {st: zero_stop() for st in ("full", "rsc", "jsd")}

        zero_blunder = lambda: {'n': 0, 'cpl': 0.0}
        self.total_blunder = {st: zero_blunder() for st in BLUNDER_STAT_KEYS}
        self.window_blunder = {st: zero_blunder() for st in BLUNDER_STAT_KEYS}

        zero_sc = lambda: {
            'total': 0, 'accepted': 0,
            'lc0_blunder': 0, 'lc0_inacc': 0, 'lc0_pv': 0, 'lc0_enrich': 0}

        self.sample_counts = zero_sc()
        self.sample_counts_window = zero_sc()

        self.tscale_n = 0
        self.tscale_eligible = 0
        self.tscale_iters = 0
        self.tscale_ne_before = 0.0
        self.tscale_ne_after = 0.0
        self.tscale_best_T = 0.0
        self.tscale_time = 0.0

        self.kl_q50 = 0.345  # online-tracked EMA median of eligible-ply KL
        self.kl_q80 = 0.657  # online-tracked EMA 80th percentile of eligible-ply KL

        self.intake = deque()
        self.blunder_replay_intake = deque()
        self.pending = {}

        self.blunder_replay_jsonl = os.path.join(
            blunder_replay_dir(cfg), "blunder_replay_probe.jsonl")
        self.n_replay_rows = count_jsonl_lines(self.blunder_replay_jsonl)
        self.n_replay_finalized = 0
        # 2:1 replay trickle meter -- drained by top_up_queues
        self.brp_credits = 0

        lc0_model = cfg.lc0_distill_model_name or os.getenv('LC0_DISTILL_MODEL', '')
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

    def maybe_add_blunder_candidate(self, pool, game_data, gid, ply_idx,
                                    played_uci, best_uci, best_cp,
                                    played_cp, cpl, z_stm,
                                    best_depth=None, played_depth=None):
        """
        Same A/B/G3/F rules the old scan_blunder_positions.py scan applied
        post-hoc, checked live against the values finalize_game already
        computed for this ply. Mutates pool directly (dedup by short_fen);
        maybe_save_pool decides when that hits disk. Only STARTPOS-traceable
        scenarios qualify: piece_odds, piece_training and random_init seed a
        custom FEN directly, so history_uci for those never traces back to
        STARTPOS. If the discovering game's own SF pass already reached
        BLUNDER_REPLAY_MIN_CACHE_DEPTH+, seed the deep cache immediately so
        the first replay probe can potentially skip SF entirely.
        """
        scenario = game_data.get('scenario')
        if scenario not in BLUNDER_SCENARIOS:
            return
        is_candidate, label = is_blunder_candidate(best_cp, played_cp, cpl, z_stm)
        if not is_candidate:
            return

        history = game_data.get('history_uci')
        played = game_data.get('moves_played') or []
        if not history:
            return

        offset = len(history) - len(played)
        uci_path = list(history[:offset + ply_idx])
        board = chess.Board()
        for uci in uci_path:
            board.push_uci(uci)

        fen = board.fen()
        key = short_fen(fen)
        # meter counts every eligible blunder, including the ones that end up
        # as a dedup bump or get dropped against a full pool
        self.brp_credits += 1

        if key in pool:
            pool[key]["times_seen"] = pool[key].get("times_seen", 1) + 1
        elif len(pool) >= BLUNDER_POOL_MAX_SIZE:
            # pool is full -- drop the candidate, no eviction pressure yet
            return
        else:
            rec = {
                "fen": fen,
                "uci_path": uci_path,
                "start_ply": offset + ply_idx,
                "run_tag": self.config.run_tag,
                "game_id": gid,
                "move_num": ply_idx,
                "model_epoch": game_data.get("model_epoch"),
                "scenario": scenario,
                "z_stm": z_stm,
                "played_move": played_uci,
                "sf_best_move": best_uci,
                "best_cp": best_cp,
                "played_cp": played_cp,
                "cpl": cpl,
                "times_seen": 1,
            }
            if (best_depth or 0) >= BLUNDER_REPLAY_MIN_CACHE_DEPTH:
                ratchet_store(rec, best_uci, best_cp, best_depth)
            if (played_depth or 0) >= BLUNDER_REPLAY_MIN_CACHE_DEPTH:
                ratchet_store(rec, played_uci, played_cp, played_depth)
            if rec.get("deep_evals", {}).get(best_uci) is not None:
                rec["deep_best_move"] = best_uci
                rec["deep_best_cp"] = best_cp
            pool[key] = rec

        self.n_pool_changes += 1
        self.maybe_save_pool(pool)

    def maybe_save_pool(self, pool):
        """
        Atomic tmp-write + swap every BLUNDER_MERGE_EVERY mutations, rate
        limited on top to at most 2 saves per 30s -- at high replay
        throughput the count trigger alone could fire many times a second.
        Changes just stay pending (nothing is lost) until the window opens
        back up.
        """
        if pool is None or self.n_pool_changes < BLUNDER_MERGE_EVERY:
            return
        now = time.time()
        self.pool_save_times = [
            t for t in self.pool_save_times if now - t < 30.0]
        if len(self.pool_save_times) >= 2:
            return
        save_probe_pool(pool, self.blunder_pool_path)
        self.pool_save_times.append(now)
        self.n_pool_changes = 0
        print(f"{RS} blunder pool saved: {len(pool)} live")

    def close(self, pool=None):
        """Idempotent -- run_selfplay closes on the normal path and again in
        cleanup, and the counts are unchanged between the two."""
        if self.closed:
            return
        self.closed = True
        if pool is not None and self.n_pool_changes > 0:
            save_probe_pool(pool, self.blunder_pool_path)
            self.n_pool_changes = 0
        path = self.opening_counts.save()
        print(f"{RS} opening counts saved to {os.path.basename(path)} "
              f"({self.opening_counts.occupancy()} slots)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @property
    def training_data_size(self):
        return len(self.live_buffer)

    def submit(self, pkl_file):
        self.intake.append(pkl_file)

    def submit_blunder_replay(self, brp_data):
        self.blunder_replay_intake.append(brp_data)

    def take_brp_credits(self):
        """Read-and-clear the eligible-blunder count since the last call."""
        n = self.brp_credits
        self.brp_credits = 0
        return n

    def tick(self, pool=None):
        if self.lc0_thread is not None:
            # Lc0Thread yields (x, policy, wdl, vwht, pwht); normalise to the
            # 7-slot record schema rather than appending it raw. Unreachable in
            # practice -- the game path flushes and drains before this runs.
            for x, policy, wdl, vwht, pwht in self.lc0_thread.drain():
                self.live_buffer.append(
                    (x, None, sparsify_policy(policy), wdl, vwht, pwht,
                     'lc0_enrich'), REWEIGHT_INELIGIBLE)

        # drain completed game batches from SF threads
        while True:
            try:
                gid, results = self.res_q.get_nowait()
            except Empty:
                break
            if gid == '__error__':
                raise results
            self.handle_game_results(gid, results, pool)

        flushed = self.live_buffer.flush_all_ready()
        if flushed:
            n = len(flushed) * self.live_buffer.shard_size
            self.written_this_round += n
            self.written_total += n

        # prefer blunder_replays to clear those
        # larger pending for these because theyre just single moves
        while self.blunder_replay_intake and len(self.pending) < 40:
            brp_data = self.blunder_replay_intake.popleft()
            self.start_blunder_replay(brp_data=brp_data, pool=pool)

        if len(self.intake) > len(self.pending):
            intake_list = list(self.intake)
            random.shuffle(intake_list)
            self.intake = deque(intake_list)
        
        while self.intake and len(self.pending) < 40:
            self.start_game(self.intake.popleft(), pool)

    def init_analyzer(self):
        run_dir = self.config.run_dir

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
        """
        Returns (unprocessed_games, unprocessed_blunder_replays). The second
        is always empty: replays never write a pkl_file/game_index entry (by
        design -- see finalize_game_data's blunder_replay_probe branch), so
        one that died mid-flight leaves no recoverable trace and just gets
        resampled later.
        """
        run_dir = self.config.run_dir
        idx_path = os.path.join(run_dir, "game_index.json")
        idx = load_game_index(idx_path)
        entries = [idx] if isinstance(idx, dict) else idx

        unprocessed_games = deque()
        for rec in entries:
            gid = str(rec.get("game_id"))
            pkl_path = rec.get("pkl_file")
            if not pkl_path or not os.path.exists(pkl_path):
                continue
            if gid in self.games_seen:
                continue
            unprocessed_games.append(rec)

        if len(unprocessed_games):
            n = len(unprocessed_games)
            print(f"[rescore] {n} unprocessed game(s) currently in queue")

        unprocessed_blunder_replays = deque()
        return unprocessed_games, unprocessed_blunder_replays

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
        return x, policy

    def append_flat_policy_example(self, board, ucis, visits, Y, vwht, pwht):
        x, policy = self.make_policy_example(board, ucis, visits)
        okey = self.opening_counts.key(board, x)
        self.opening_counts.bump(okey)
        self.live_buffer.append(
            (x, None, sparsify_policy(policy), Y, vwht, pwht, 'xc0'), okey)

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

        vwht = value_weight_for_game(cfg, is_draw)
        self.append_flat_policy_example(board, ucis, visits, Y, vwht, policy_weight)

    def start_game(self, pkl_file, pool):
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
        board_ch, b_fast = reconcile_game_boards(
            game_data['start_fen'], game_data.get('history_uci'),
            game_data.get('moves_played'),
        )

        tree_data = game_data.get('tree_search_data', {})
        vs_stockfish = game_data.get('vs_stockfish', False)
        sf_color = game_data.get('stockfish_is_white')
        result = game_data['result']
        is_draw = (result == 0) or (result == 0.0)

        is_validation_game = 'validation' in game_data.get('scenario', '').lower()
        skip_all_training = is_validation_game

        # snapshot config fields so hot-reloads don't affect an in-flight game
        game_cfg_keys = (
            'train_on_stockfish',
            'kl_boost_median_mult', 'kl_boost_p80_mult', 'kl_quantile_lr',
            'rescore_equiv_range', 'rescore_blunder_cp_loser',
            'rescore_blunder_cp_winner', 'rescore_inaccuracy_cp',
            'inaccuracy_downweight',
            'uniform_eps', 'prior_clip_max',
            'rescore_analyze_batch',
            'draw_value_scale',
            'vscale', 'lc0_enrich_frac', 'lc0_enrich_weight'
        )

        game_state = {
            'gid': gid,
            'game_data': game_data,
            'cfg': SimpleNamespace(**{k: getattr(cfg, k) for k in game_cfg_keys}),
            'ply_states': [],
            'waiting': False,  # True once a batch has been submitted to a SF thread
        }

        # (ply_idx, board_copy, xerces_uci) queued for SF analysis
        to_sf_positions = []
        # walk each move: route SF moves, build ply state
        for i, mv in enumerate(game_data.get('moves_played', [])):
            move_ch = chess.Move.from_uci(mv)
            is_sf_move = vs_stockfish and (board_ch.turn == sf_color)
            turn = board_ch.turn

            if is_sf_move and ((not cfg.train_on_stockfish) or skip_all_training):
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            tr = tree_data.get(i, tree_data.get(str(i), {}))
            cm = tr.get('candidate_moves', [])
            if not cm:
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            Z_stm = result if turn else -result
            if 'Q_stm' in tr:
                Q = tr['Q_stm']
            else:
                wdl = tr.get('best_wdl')
                this_q = (wdl[0] - wdl[2]) if wdl is not None else 0.0
                Q = this_q if turn else -this_q

            sf_wdl = tr.get('sf_wdl')
            if sf_wdl is not None and len(sf_wdl) == 3:
                Y_init = (0.5*z_to_wdl(Z_stm) + 0.5*np.array(sf_wdl, dtype=np.float32))
            else:
                Y_init = z_to_wdl(Z_stm)

            if is_sf_move:
                sf_pwht = 0.0 if is_validation_game else 1.0
                self.training_data_from_sf(
                    b_fast, mv, cm, Y_init, is_draw, policy_weight=sf_pwht
                )

                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            visits = [(c['uci'], max(1, c['visits'])) for c in cm]
            visits = sorted(visits, key=lambda x: x[1], reverse=True)
            if not visits:
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
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
            # short_fen still needed for the lc0 waypoint FEN check
            short_fen = b_fast.fen(include_counters=False)

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
                'idx_map': idx_map,
                'okey': self.opening_counts.key(b_fast, x),
                'skip_training': skip_all_training,
                'xerces_uci': xerces_uci,
                'sfen': short_fen,
                'best_uci': None,
                'best_cp': None,
                'best_abs': None,
                'pv_ucis': [],
                'played_cp': None,
                'played_abs': None,
                'resolved': False,
            }

            to_sf_positions.append((i, board_ch.copy(), xerces_uci))

            game_state['ply_states'].append(ply)
            board_ch.push(move_ch)
            b_fast.push_uci(mv)

        self.pending[gid] = game_state
        # submit to SF thread, or finalize immediately if every move was SF's own
        if to_sf_positions:
            self.n_sf_submitted += len(to_sf_positions)
            game_state['waiting'] = True
            self.game_q.put((gid, to_sf_positions, {}))
        else:
            self.finalize_game(self.pending.pop(gid), pool)

    def start_blunder_replay(self, brp_data, pool):
        gid = str(brp_data.get('game_id'))
        if gid in self.games_seen:
            return

        short_fen_key = brp_data.get('short_fen')
        xerces_uci = brp_data.get('xerces_uci')

        # both the best move and the move actually played here already
        # cached at BLUNDER_REPLAY_MIN_CACHE_DEPTH+ from an earlier probe or
        # from initial pool-add seeding -- skip SF entirely
        if pool is not None and short_fen_key in pool:
            hit = cached_deep_score(
                pool[short_fen_key], xerces_uci, BLUNDER_REPLAY_MIN_CACHE_DEPTH)
            if hit is not None:
                self.games_seen.add(gid)
                self.finalize_replay_result(
                    brp_data, pool,
                    probe_move=xerces_uci,
                    best_uci=hit['deep_best_move'],
                    best_cp=hit['deep_best_cp'],
                    played_cp=hit['deep_played_cp'],
                    best_depth=hit['deep_best_depth'],
                    played_depth=hit['deep_depth'],
                    sims=None, stop_reason=None,
                    from_cache=True,
                )
                return

        board_ch, b_fast = reconcile_game_boards(
            brp_data['start_fen'], brp_data.get('history_uci'),
            brp_data.get('moves_played'),
        )
        tree_data = brp_data.get('tree_search_data', {})

        game_state = {
            'gid': gid,
            'brp_data': brp_data,
            'ply_states': [],
            'waiting': False,  # True once a batch has been submitted to a SF thread
        }

        to_sf_positions = []
        for i, mv in enumerate(brp_data.get('moves_played', [])):
            move_ch = chess.Move.from_uci(mv)
            turn = board_ch.turn

            tr = tree_data.get(i, tree_data.get(str(i), {}))
            xerces_uci = brp_data.get('xerces_uci')

            # cloned pre-move so a qualifying probe can build a training
            # example at finalize time without backtracking from history_uci
            ply = {
                'ply_idx': i, 'mv': mv,
                'turn': bool(turn),
                'tr': tr,
                'xerces_uci': xerces_uci,
                'board': b_fast.clone(),
                'resolved': False,
            }

            to_sf_positions.append((i, board_ch.copy(), xerces_uci))

            game_state['ply_states'].append(ply)
            board_ch.push(move_ch)
            b_fast.push_uci(mv)

        self.pending[gid] = game_state
        # submit to SF thread, or finalize immediately if every move was SF's own
        if to_sf_positions:
            self.n_sf_submitted += len(to_sf_positions)
            game_state['waiting'] = True
            sf_ms = BLUNDER_REPLAY_START_MS
            if pool is not None and short_fen_key in pool:
                src = pool[short_fen_key]
                if not src.get('sf_ms'):
                    src['sf_ms'] = BLUNDER_REPLAY_START_MS
                    self.n_pool_changes += 1
                    self.maybe_save_pool(pool)
                sf_ms = src['sf_ms']
            self.game_q.put((gid, to_sf_positions, {"movetime_ms": sf_ms}))
        else:
            # should never get here (every replay has exactly one move), but
            # if it does, pop instead of leaking a permanent pending slot
            self.pending.pop(gid, None)

    def handle_game_results(self, gid, results, pool):
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
            ds = r.get('depths') or []
            self.sf_depth_sum   += sum(ds)
            self.sf_depth_count += len(ds)
            if r.get('rerun'):
                self.sf_call_count += 2
                self.sf_rerun_count += 1
            else:
                self.sf_call_count += 1

            ply = ply_map.get(ply_idx)
            if ply is None:
                continue

            ply['best_uci']   = best_uci
            ply['best_cp']    = best_cp
            ply['best_abs']   = best_abs
            ply['pv_ucis']    = r.get('pv_ucis', [])
            ply['played_cp']  = played_cp
            ply['played_abs'] = played_abs
            ply['best_depth'] = r.get('best_depth')
            ply['depth']      = r.get('depth')
            ply['resolved']   = True

        game['waiting'] = False
        game_state = self.pending.pop(gid)
        if 'brp_data' in game_state:
            self.finalize_blunder_replay(game_state, pool)
        else:
            self.finalize_game(game_state, pool)

    def finalize_blunder_replay(self, game_state, pool):
        """One SF result per replay. Evicts on the first found_equiv, full
        stop -- with replays this frequent a streak requirement is
        pointless."""
        gid = game_state['gid']
        brp_data = game_state['brp_data']
        ply_states = game_state['ply_states']
        if not ply_states or not ply_states[0].get('resolved'):
            self.games_seen.add(gid)
            return
        ply = ply_states[0]
        tr = ply.get('tr') or {}

        self.games_seen.add(gid)
        self.finalize_replay_result(
            brp_data, pool,
            probe_move=ply.get('xerces_uci'),
            best_uci=ply.get('best_uci'),
            best_cp=ply.get('best_cp'),
            played_cp=ply.get('played_cp'),
            best_depth=ply.get('best_depth'),
            played_depth=ply.get('depth'),
            sims=tr.get('sims'), stop_reason=tr.get('stop_reason'),
            from_cache=False,
            ply=ply,
        )

    def finalize_replay_result(self, brp_data, pool, probe_move, best_uci,
                                best_cp, played_cp, best_depth, played_depth,
                                sims, stop_reason, from_cache, ply=None):
        """
        Shared by the normal SF-completion path (finalize_blunder_replay)
        and start_blunder_replay's cache-hit shortcut. Ratchet-stores the
        best/played move eval+depth onto the pool record on every outcome,
        not just eviction, so a later probe of the same still-live position
        can skip SF via cached_deep_score.
        """
        short_fen_key = brp_data.get('short_fen')
        orig_move = brp_data.get('orig_move')
        orig_cpl = brp_data.get('orig_cpl')
        epoch = brp_data.get('model_epoch')

        probe_cpl = best_cp - played_cp
        found_best = probe_move == best_uci
        found_equiv = probe_cpl <= BLUNDER_REPLAY_EQUIV_CPL
        same_move = probe_move == orig_move
        added_to_buffer = False
        added_epoch = None

        if short_fen_key and pool is not None and short_fen_key in pool:
            src = pool[short_fen_key]
            added_epoch = src.get('model_epoch')
            if not from_cache:
                if (best_depth or 0) >= BLUNDER_REPLAY_MIN_CACHE_DEPTH:
                    ratchet_store(src, best_uci, best_cp, best_depth)
                if (played_depth or 0) >= BLUNDER_REPLAY_MIN_CACHE_DEPTH:
                    ratchet_store(src, probe_move, played_cp, played_depth)
                depths_map = src.get('deep_depths') or {}
                prev_depth = depths_map.get(src.get('deep_best_move')) or 0
                if (best_depth or 0) >= prev_depth:
                    src['deep_best_move'] = best_uci
                    src['deep_best_cp'] = best_cp

                # neither resolved (found_equiv) nor locked a deep-enough
                # best move this probe -- buy more depth next time
                if not found_equiv and (best_depth or 0) < BLUNDER_REPLAY_MIN_CACHE_DEPTH:
                    current_ms = src.get('sf_ms') or BLUNDER_REPLAY_START_MS
                    src['sf_ms'] = min(
                        current_ms + BLUNDER_REPLAY_MS_STEP,
                        BLUNDER_REPLAY_START_MS * BLUNDER_REPLAY_MS_MAX_MULT,
                    )

            if found_equiv:
                evict_position(pool, short_fen_key, "found_equiv", epoch,
                               evicted_path(self.blunder_pool_path))
            elif ply is not None and probe_cpl > 60:
                age_ok = (added_epoch is None
                          or (epoch is not None and epoch - added_epoch > 5))
                if age_ok:
                    added_to_buffer = self.add_blunder_replay_training_example(
                        ply, epoch, best_uci, best_cp, probe_cpl)

            self.n_pool_changes += 1
            self.maybe_save_pool(pool)

        row = {
            "short_fen": short_fen_key,
            "src_run_tag": brp_data.get('src_run_tag'),
            "model_epoch": epoch,
            "added_epoch": added_epoch,
            "orig_move": orig_move,
            "orig_cpl": orig_cpl,
            "probe_move": probe_move,
            "probe_cpl": probe_cpl,
            "deep_best": best_uci,
            "best_depth": best_depth,
            "same_move": same_move,
            "found_best": found_best,
            "found_equiv": found_equiv,
            "evicted": found_equiv,
            "added_to_buffer": added_to_buffer,
            "sims": sims,
            "stop_reason": stop_reason,
            "from_cache": from_cache,
        }
        with open(self.blunder_replay_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")

        self.n_replay_rows += 1
        self.n_replay_finalized += 1
        if self.n_replay_rows >= BLUNDER_REPLAY_ANALYSIS_EVERY:
            self.analyse_brp_file()

    def add_blunder_replay_training_example(self, ply, epoch, best_uci,
                                             best_cp, probe_cpl):
        """
        Still-live, still-bad, stale-discovery replay probes never earn a
        normal training example on their own -- this is the only path that
        turns one into a live_buffer sample. Blends the replaying model's
        own search (visits + WDL) with the SF probe result: 1 visit on every
        legal move plus the rest piled on SF's best move, and SF's cp turned
        into a WDL via the tanh value curve. alpha (SF weight) scales 0.25 at
        probe_cpl=60 up to 0.75 at probe_cpl>=150 -- the worse xc0's move,
        the more the sample leans on SF.
        """
        board = ply['board']
        tr = ply.get('tr') or {}

        cm = tr.get('candidate_moves', [])
        xc0_pairs = [(c['uci'], max(1, c['visits'])) for c in cm]
        if not xc0_pairs:
            return False

        legal_ucis = board.legal_moves()
        if best_uci not in legal_ucis:
            return False

        total_visits = sum(v for _, v in xc0_pairs)
        sf_visits = {u: 1 for u in legal_ucis}
        sf_visits[best_uci] += max(total_visits - len(legal_ucis), 0)

        xc0_probs = {u: v / total_visits for u, v in xc0_pairs}
        sf_total = sum(sf_visits.values())
        sf_probs = {u: v / sf_total for u, v in sf_visits.items()}

        alpha = 0.25 + (min(max(probe_cpl, 60), 150) - 60) / 90.0 * 0.5

        ucis = sorted(set(xc0_probs) | set(sf_probs))
        blended = np.array([
            alpha * sf_probs.get(u, 0.0) + (1 - alpha) * xc0_probs.get(u, 0.0)
            for u in ucis
        ], dtype=np.float32)
        blended = blended / blended.sum()

        indices = board.moves_to_indices(ucis)
        policy = np.zeros(1858, dtype=np.float32)
        for idx, p in zip(indices, blended):
            policy[idx] += p

        wdl_xc0 = tr.get('best_wdl')
        wdl_xc0 = (np.array(wdl_xc0, dtype=np.float32) if wdl_xc0 is not None
                   else scalar_to_wdl(tr.get('Q_stm', 0.0)))
        wdl_sf = scalar_to_wdl(cp_to_value_tanh(best_cp, mid_cp=200.0))
        Y = alpha * wdl_sf + (1.0 - alpha) * wdl_xc0

        x = self.encode_board(board)
        okey = self.opening_counts.key(board, x)
        self.opening_counts.bump(okey)
        # distinct source tag so retrain_worker.py can exclude these blended
        # synthetic-target samples from the validation draw
        self.live_buffer.append(
            (x, None, sparsify_policy(policy), Y, 1.0, 1.0, 'xc0_replay_blend'),
            okey)
        return True

    def analyse_brp_file(self):
        """
        Replaces the old analyse_probe_file: no SF pass here, that already
        happened per-replay in finalize_blunder_replay, which also evicts
        immediately on found_equiv. This just summarizes the collected jsonl
        into a CSV + history line, as if one full probe run just finished,
        stamped with whatever epoch is current -- not necessarily on any
        retrain cadence. Fires purely on row count.
        """
        rows = []
        with open(self.blunder_replay_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if not rows:
            self.n_replay_rows = 0
            return

        new_df = pd.DataFrame(rows)
        # max, not rows[-1] -- a backlogged batch of stale-epoch rows can get
        # processed well after the run has moved on, and rows[-1] would then
        # silently attribute the whole window to that old epoch. TODO: use
        # the live current epoch directly instead of inferring it from the
        # window's own rows.
        epoch = max([r.get("model_epoch") for r in rows])

        # two 2048-row windows can land on the same epoch if replay
        # throughput outpaces the retrain cadence -- append onto that
        # epoch's existing CSV instead of splitting across multiple files.
        # probe_e*.csv naming matches the old convention -- probe_table.py,
        # pairwise.py, and backfill_probe_csv_keys.py all glob for it.
        existing = glob.glob(os.path.join(
            blunder_replay_dir(self.config), f"probe_e{epoch}_*.csv"))
        if existing:
            csv_path = existing[0]
            df = pd.concat([pd.read_csv(csv_path), new_df], ignore_index=True)
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(
                blunder_replay_dir(self.config), f"probe_e{epoch}_{stamp}.csv")
            df = new_df
        df.to_csv(csv_path, index=False)

        summary = {
            "ts": int(time.time()),
            "n_retrains": epoch,
            "n_positions": len(df),
            "orig_cpl": df["orig_cpl"].mean(),
            "probe_cpl": df["probe_cpl"].mean(),
            "improved": (df["probe_cpl"] < df["orig_cpl"]).mean(),
            "worsened": (df["probe_cpl"] > df["orig_cpl"]).mean(),
            "same_move": df["same_move"].mean(),
            "found_best": df["found_best"].mean(),
            "found_equiv": df["found_equiv"].mean(),
            "n_evicted": int(df["evicted"].sum()),
            "n_added_to_buffer": int(df["added_to_buffer"].sum()),
            "cache_hit": df["from_cache"].mean(),
            "file": os.path.basename(csv_path),
        }
        hist_path = os.path.join(
            self.config.run_dir, BLUNDER_REPLAY_HISTORY_FILENAME)
        with open(hist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, default=float) + "\n")

        print(f"{RS} blunder replay analysis @ epoch {epoch}: "
              f"n={summary['n_positions']} "
              f"cpl {summary['orig_cpl']:.1f} -> {summary['probe_cpl']:.1f}  "
              f"same_move {summary['same_move']:.1%}  "
              f"found_equiv {summary['found_equiv']:.1%}  "
              f"evicted {summary['n_evicted']}  "
              f"added_to_buffer {summary['n_added_to_buffer']}")

        open(self.blunder_replay_jsonl, "w", encoding="utf-8").close()
        self.n_replay_rows = 0

    def finalize_game(self, game_state, pool):
        if self.start_time is None:
            self.start_time = time.time()
        cfg = game_state['cfg']
        game_data = game_state['game_data']
        gid = game_state['gid']
        result = game_data['result']
        is_draw = (result == 0) or (result == 0.0)
        vs_stockfish = game_data.get('vs_stockfish', False)
        sf_color = game_data.get('stockfish_is_white')

        EQUIV = cfg.rescore_equiv_range

        cpl_s = 0.0
        n_plies = 0
        rows = []
        eval_trace = []
        turn_at = {}
        pending = []
        pending_aux = []

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

            # variable blunder threshold based on outcome -- computed here
            # (not just below, where it used to live) so missed_mate can
            # gate on "did this actually clear the blunder bar", not just
            # "are both scores huge". A small gap between two huge scores
            # (e.g. best=1500, played=1450) is not a missed mate. This also
            # rules out the delta<=-EQUIV case above on its own: delta is
            # negative there and blunder_cp is always positive, so
            # delta >= blunder_cp can never be true when xerces outplayed
            # SF's own recorded best.
            blunder_cp = cfg.rescore_blunder_cp_winner
            if Z_stm <= 0.0:
                blunder_cp = cfg.rescore_blunder_cp_loser

            missed_mate = (
                best_cp >= 1200 and played_cp >= 500 and Z_stm > 0
                and delta >= blunder_cp
            )
            # missed mates are not really critical and have an oversized CPL penalty.
            if missed_mate:
                loss_this = min(100, loss_this)
            cpl_s += loss_this
            n_plies += 1

            # consider adding this position to the blunder replay pool
            if pool is not None and not missed_mate:
                self.maybe_add_blunder_candidate(
                    pool, game_data, gid, i, xerces_uci, best_uci, best_cp,
                    played_cp, loss_this, Z_stm,
                    best_depth=ply.get('best_depth'), played_depth=ply.get('depth'),
                )

            eval_trace.append((i, best_abs))
            rows.append([
                i, mv, xerces_uci, best_uci,
                best_cp, loss_this, played_cp,
                best_abs, played_abs,
                turn, loss_this,
                tr.get('stop_reason', ''), tr.get('sims', 0),
                np.nan, np.nan,  # kl, ce -- filled in below if this ply trains
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

            is_blunder = loss_this >= blunder_cp
            true_blunder = not (played_cp > 350 and Z_stm > 0) and is_blunder
            is_inaccuracy = (
                not missed_mate and not is_blunder
                and loss_this >= cfg.rescore_inaccuracy_cp
            )

            self.accumulate_blunder_stats(
                missed_mate, is_blunder, true_blunder, is_inaccuracy,
                best_cp, played_cp, loss_this, Z_stm,
            )

            # mild and big blunders: drop xc0 record, queue lc0 instead
            skip_xc0 = not missed_mate and (loss_this >= blunder_cp)
            if missed_mate:
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
                ne = calc_entropy(vis_arr, normed_only=True)
                target_e = 0.65
                if ne > target_e:
                    t0 = time.perf_counter()
                    lo, hi, best, best_dist = 0.3, 1.0, 0.75, float('inf')
                    iters = 0
                    for _ in range(5):
                        mid = (lo + hi) / 2
                        scaled = vis_arr ** (1.0 / mid)
                        scaled /= scaled.sum()
                        ne_mid = calc_entropy(scaled, normed_only=True)
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
                    ne_after = calc_entropy(
                        np.array(vis, dtype=np.float64), normed_only=True)
                    
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
            rows[-1][-2] = kl
            self.kl_q50 = update_ema_quantile(self.kl_q50, kl, 0.5, cfg.kl_quantile_lr)
            self.kl_q80 = update_ema_quantile(self.kl_q80, kl, 0.8, cfg.kl_quantile_lr)
            if kl_eligible:
                if kl >= self.kl_q80:
                    pwht *= cfg.kl_boost_p80_mult
                elif kl >= self.kl_q50:
                    pwht *= cfg.kl_boost_median_mult

            # the inaccuracy downweight applies to the xc0 record only. The lc0
            # record that accompanies it is a second independent observation
            # and carries its own weight, so it submits vwht_base -- inheriting
            # the halved vwht would compose 0.5 xc0 + 0.25 lc0 on the value
            # side against 0.5 + 0.5 on the policy side.
            vwht_base = vwht
            if lc0_mode == 'pos_only':
                pwht *= cfg.inaccuracy_downweight
                vwht *= cfg.inaccuracy_downweight

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

            pending.append((ply['x'], policy, Q, turn, i, vwht, pwht, ply['okey']))
            sf_pv = ply.get('pv_ucis', [])
            if lc0_mode == 'pos_sf_xc0_pov':
                xc0_pv = [e['uci'] for e in ply['tr'].get('pv', [])[:3]]
                pv_seqs = [sf_pv, xc0_pv]
            elif lc0_mode == 'pos_sf_pov':
                pv_seqs = [sf_pv[:2]]
            else:
                pv_seqs = []

            pending_aux.append({
                'sfen':            ply['sfen'],
                'pv_seqs':         pv_seqs,
                'skip_xc0':        skip_xc0,
                'lc0_mode':        lc0_mode,
                'vwht_base':       vwht_base,
                'is_blunder':      is_blunder,
                'ply_i':           i,
                'best_wdl':        tr.get('best_wdl'),
            })

        sc = self.sample_counts
        scw = self.sample_counts_window

        start_fen    = game_data['start_fen']
        moves_played = game_data['moves_played']
        history_uci  = game_data.get('history_uci')
        row_by_ply   = {r[0]: r for r in rows}

        # loop 2: compute WDL targets, add to training, build lc0 waypoints
        lc0_waypoints = {}
        for tup, aux in zip(pending, pending_aux):
            x, policy, Q, is_white, ply_i, vwht, pwht, okey = tup
            z = result if is_white else -result
            Y = blend_wdl(z, aux.get('best_wdl'), is_white)

            wdl_node = aux.get('best_wdl')
            if wdl_node is not None:
                model_wdl = np.array(wdl_node, dtype=np.float32)
                if not is_white:
                    model_wdl = model_wdl[[2, 1, 0]]
                row = row_by_ply.get(ply_i)
                if row is not None:
                    # logged CE only -- scored against the raw game result, not
                    # Y. Y is half model_wdl, so CE(Y, model_wdl) is partly
                    # self-referential and barely moves. Y itself is unchanged.
                    row[-1] = cross_entropy(z_to_wdl(z), model_wdl)

            sc['total'] += 1
            scw['total'] += 1

            entry = (x, None, sparsify_policy(policy), Y, vwht, pwht, 'xc0')

            if not aux.get('skip_xc0'):
                # an inaccuracy ply is already suppressed to 0.5 to make room
                # for its lc0 half, so the opening band must not take a second
                # bite. The bump still happens: the position was played.
                xc0_key = (REWEIGHT_INELIGIBLE
                           if aux.get('lc0_mode') == 'pos_only' else okey)
                self.opening_counts.bump(okey)
                self.live_buffer.append(entry, xc0_key)
                sc['accepted'] += 1
                scw['accepted'] += 1

            if self.lc0_thread is not None:
                mode = aux.get('lc0_mode')
                if mode in ('pos_sf_xc0_pov', 'pos_sf_pov'):
                    lc0_waypoints[aux['ply_i']] = (entry, aux, Y, is_white)
                    sc['lc0_blunder'] += 1
                    scw['lc0_blunder'] += 1
                elif mode == 'pos_only':
                    lc0_waypoints[aux['ply_i']] = (entry, aux, Y, is_white)
                    sc['lc0_inacc'] += 1
                    scw['lc0_inacc'] += 1
                elif random.random() < cfg.lc0_enrich_frac:
                    # sampled on frac alone. This used to be gated on the
                    # position being past MAX_TRACKED_PLY, to avoid spending
                    # inference on an opening the band would suppress anyway --
                    # but lc0 records are never reweighted now, so inclusion no
                    # longer has anything to do with weighting. (The gate was
                    # dead regardless: it tested `okey is None`, and key()
                    # returns REWEIGHT_INELIGIBLE past the ply window, never
                    # None, so no enrich record has ever been produced.)
                    lc0_waypoints[aux['ply_i']] = (entry, aux, Y, is_white)
                    sc['lc0_enrich'] += 1
                    scw['lc0_enrich'] += 1

        if lc0_waypoints and self.lc0_thread is not None:
            lc0_meta = []
            _, b_lc0 = reconcile_game_boards(start_fen, history_uci, moves_played)
            for ply_i, mv in enumerate(moves_played):
                if ply_i in lc0_waypoints:
                    entry, aux, Y, is_white = lc0_waypoints[ply_i]
                    got_fen = b_lc0.fen(include_counters=False)
                    if got_fen != aux['sfen']:
                        msg = f"[rescore] FEN mismatch at ply {ply_i}: {got_fen} "
                        msg += f"!= {aux['sfen']}"
                        print(msg)

                    x, _, _, _, _, _, _ = entry
                    vwht = aux['vwht_base']
                    ucis_now = b_lc0.legal_moves()
                    self.lc0_thread.submit(b_lc0.lc0_features(), b_lc0, x, vwht, 1.0)
                    xc0_idxs = b_lc0.moves_to_indices(ucis_now)
                    lc0_meta.append(
                        ('main', aux, Y, is_white, list(zip(ucis_now, xc0_idxs)))
                    )

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
                            x_pv = self.encode_board(b_pv)
                            self.lc0_thread.submit(
                                b_pv.lc0_features(),
                                b_pv,
                                x_pv,
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
                x, policy, lc0_wdl, vwht, pwht = result
                if kind == 'pv':
                    source = 'lc0_pv'  # PV nodes only ever come from blunder/inacc waypoints
                elif aux.get('lc0_mode'):
                    source = 'lc0_blunder'  # main position from a blunder/inacc waypoint
                else:
                    source = 'lc0_enrich'  # main position from random enrich sampling
                # no bump and no key: only xerces-played moves count toward the
                # opening table, and no lc0 record is ever reweighted by it
                self.live_buffer.append(
                    (x, None, sparsify_policy(policy), lc0_wdl,
                     vwht * w, pwht * w, source), REWEIGHT_INELIGIBLE)
        
        cols = [
            'move_num', 'played_move', 'most_visited_move', 'best_move',
            'best_cp', 'delta', 'played_cp',
            'best_absolute', 'played_absolute', 'stm', 'loss', 'stop_reason', 'sims',
            'kl', 'ce',
        ]

        out_df = pd.DataFrame(rows, columns=cols)
        out_df['played_best_move'] = out_df['delta'] <= 0

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
            self.print_blunder_stats()
        if len(self.analyzed_results) >= cfg.rescore_analyze_batch:
            self.push_analyzed(report=True)


    def accumulate_blunder_stats(self, missed_mate, is_blunder, true_blunder,
                                 is_inaccuracy, best_cp, played_cp, cpl, z_stm):
        def bump(key, value):
            for acc in (self.total_blunder, self.window_blunder):
                acc[key]['n'] += 1
                acc[key]['cpl'] += value

        bump('total', cpl)

        if missed_mate:
            bump('missed_mate', cpl)
        else:
            _, label = is_blunder_candidate(best_cp, played_cp, cpl, z_stm)
            if label is not None:
                bump(label, cpl)
            elif is_blunder:
                bump('U', cpl)

        if true_blunder:
            bump('true_blunder', cpl)
        elif is_blunder:
            bump('false_blunder', cpl)

        if is_inaccuracy:
            bump('inaccuracy', cpl)

        if missed_mate or is_blunder:
            bump('all_blunders', cpl)

    def print_blunder_stats(self):
        acc = self.window_blunder
        primary = ("A", "B", "G3", "U")
        pool_eligible = ("A", "B", "G3", "F")

        def avg(k):
            n = acc[k]['n']
            return acc[k]['cpl'] / n if n else float('nan')

        def row(label, k):
            if acc[k]['n'] == 0:
                return
            print(f"{RS}  {label:<24} n={acc[k]['n']:>6}  cpl={avg(k):6.2f}")

        print(f"{RS} blunder types (last 100):")
        row("total", "total")
        row("inaccuracies", "inaccuracy")
        for k in primary:
            row(k, k)
        row("true_blunder", "true_blunder")
        row("all blunders", "all_blunders")

        # same A/B/G3/F rule maybe_add_blunder_candidate uses -- fraction of
        # all trainable plies this window that would qualify for the pool
        pool_n = sum(acc[k]['n'] for k in pool_eligible)
        total_n = acc['total']['n']
        if pool_n:
            pool_pct = pool_n / total_n if total_n else 0.0
            print(f"{RS}  {'pool-eligible (A/B/G3/F)':<24} "
                  f"n={pool_n:>6}  of {total_n} moves ({pool_pct:.2%})")

        for k in BLUNDER_STAT_KEYS:
            self.window_blunder[k] = {'n': 0, 'cpl': 0.0}

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
        print(f"{RS} KL running medians: "
              f"q50={self.kl_q50:.3f}  q80={self.kl_q80:.3f}")

        for st in stops:
            self.window_stop[st] = {'n': 0, 'cpl': 0.0, 'bmr': 0.0, 'sims': 0.0}

    def print_sample_stats(self):
        W = 10

        def col(val):
            return f"  {val:>{W}}  |"

        def enrich_row(label, sc):
            blunder = sc['lc0_blunder']
            inacc   = sc.get('lc0_inacc', 0)  # absent in state exported before this field existed
            lc0_total = blunder + inacc + sc['lc0_pv'] + sc['lc0_enrich']
            total_tr  = sc['accepted'] + sc['lc0_pv'] + sc['lc0_enrich']
            enrich_pct = f"{lc0_total / total_tr * 100:.1f}%" if total_tr else "--"
            return (f"{RS}  {label:<12} |"
                    + col(blunder)
                    + col(inacc)
                    + col(sc['lc0_pv'])
                    + col(sc['lc0_enrich'])
                    + col(enrich_pct))

        ehdr = (f"{RS}  {'enrich':<12} |"
                + "".join(col(lbl) for lbl in ("blunder", "inacc", "pv", "enrich", "lc0-%")))
        print(ehdr)
        print(enrich_row("batch", self.sample_counts_window))
        print(enrich_row("total", self.sample_counts))

        if self.lc0_thread is not None:
            lst = self.lc0_thread.stats()
            print(f"{RS}  lc0 output: {lst['samples']} flushed"
                  f"  {lst['inferences']} batches"
                  f"  {lst['pending']} pending")

        self.sample_counts_window = {
            'total': 0, 'accepted': 0,
            'lc0_blunder': 0, 'lc0_inacc': 0, 'lc0_pv': 0, 'lc0_enrich': 0,
        }

    def aggregate_metrics(self, epoch, progress_csv_path, pred_pkl_path,
                          train_stats=None):
        """train_stats: {'train_loss': ..., 'gn_mean': ...} from the retrain
        that just finished. Validation is measured on the model going INTO
        that retrain, matching how pretraining pairs an eval with the loss of
        the epoch it belongs to."""
        if not os.path.exists(pred_pkl_path):
            print(f"[metrics] pred pkl not found, skipping: {pred_pkl_path}")
            return

        with open(pred_pkl_path, 'rb') as f:
            payload = pickle.load(f)

        eps = 1e-7

        def build_group(stream):
            g = {'wdl_pred': [], 'wdl_true': [], 'pol_pred': [], 'source': []}
            samples = stream["samples"]
            preds = stream["preds"]
            if len(samples) != len(preds):
                raise RuntimeError(
                    f"sample/pred count mismatch: {len(samples)} vs {len(preds)}")
            for (pred_wdl, pred_pol), sample in zip(preds, samples):
                g['wdl_pred'].append(pred_wdl)
                g['wdl_true'].append(np.array(sample[3], dtype=np.float64))
                g['pol_pred'].append(pred_pol)
                g['source'].append(sample[6])
            # one shot, and it handles sparse and dense records alike
            g['pol_true'] = stack_policies(samples)
            return g

        primary = build_group(payload["primary"])
        historic = build_group(payload["historic"])

        if not primary['wdl_pred'] and not historic['wdl_pred']:
            print("[metrics] no samples in pred pkl, skipping")
            return

        def metrics_block(g):
            if not g['wdl_pred']:
                return None
            pp = np.array(g['wdl_pred'], dtype=np.float64)
            pt = np.array(g['wdl_true'], dtype=np.float64)
            pp = np.clip(pp, eps, 1 - eps)
            pp /= pp.sum(axis=1, keepdims=True)

            # scalar value for correlation: expected score from WDL
            pred_v = pp[:, 0] - pp[:, 2]
            true_v = pt[:, 0] - pt[:, 2]
            ce = -np.mean(np.sum(pt * np.log(pp), axis=1))
            mse = np.mean((pred_v - true_v) ** 2)
            corr = np.corrcoef(pred_v, true_v)[0, 1] if len(pred_v) > 1 else 0.0

            logits = np.array(g['pol_pred'], dtype=np.float32)
            true_p = g['pol_true']

            # Same function pretraining validates with, on the same
            # target-support mask it uses (bootstrap's mstack = pstack > 0),
            # so every policy column lines up across the pretrain/selfplay
            # seam instead of being a lookalike reimplementation.
            support = (true_p > 0).astype(np.int32)
            out = batch_policy_metrics(logits, true_p, support)
            out.update({'ce': ce, 'mse': mse, 'corr': corr,
                        'n': len(g['wdl_pred'])})
            # mass on the #1 entry of the TARGET distribution -- whichever
            # source produced the record (xc0 visits, lc0 output, historic)
            out['avg_top_prob_target'] = np.mean(true_p.max(axis=1))

            return out

        pb = metrics_block(primary)
        hb = metrics_block(historic)

        pfx = f"[epoch {epoch:4d}] [metrics]"
        LBL_W = 60

        def wdl_line(b):
            return (f"CE={b['ce']:.3f}  MSE={b['mse']:.3f}  "
                    f"corr={b['corr']:.2f}  CE_gain={b['ce_gain']:.2f}  n={b['n']}")

        def pol_line(b):
            return (f"top1_exact={b['top1_exact']:.2f}  "
                    f"mass: top1={b['top1_mass']:.2f}  top3={b['top3_mass']:.2f}  top5={b['top5_mass']:.2f}")

        def print_slice(b, label, line_fn):
            if not b:
                return
            print(f"{pfx} {line_fn(b):<{LBL_W}}({label})")

        slices = [(pb, 'primary'), (hb, 'historic')]
        for b, label in slices:
            print_slice(b, label, wdl_line)
        for b, label in slices:
            print_slice(b, label, pol_line)

        # internal key -> csv column. Names match the pretraining schema
        # (bootstrap's UNIFIED_COLS) so both files stack on the same axes.
        col_map = {
            'mse': 'value_mse', 'corr': 'value_corr', 'ce': 'value_ce',
            'policy_ce': 'policy_ce', 'uniform_ce': 'uniform_ce', 'ce_gain': 'ce_gain',
            'top1_exact': 'top1_exact',
            'top1_mass': 'top1_mass', 'top3_mass': 'top3_mass', 'top5_mass': 'top5_mass',
            'n': 'n_samples',
            'mass_on_legal': 'mass_on_legal',
            'avg_top_prob': 'avg_top_prob',
            'avg_top_prob_target': 'avg_top_prob_target',
            'exp_prob_model': 'exp_prob_model',
            'exp_prob_uniform': 'exp_prob_uniform',
            'prob_on_others': 'prob_on_others',
        }

        def save_csv(path, b, row_epoch):
            if not b:
                return
            row = {'model_epoch': row_epoch}
            for k, col in col_map.items():
                if k in b:
                    row[col] = round(b[k], 4)
            if train_stats:
                row.update(train_stats)

            if not os.path.exists(path):
                pd.DataFrame([row]).round(4).to_csv(path, index=False)
                return

            # positional append rather than concat: a one-row frame whose
            # pretrain-only columns are all-NA is exactly the case pandas
            # deprecated the dtype inference for, and it warned every retrain.
            all_df = pd.read_csv(path)
            cols = list(dict.fromkeys(list(all_df.columns) + list(row)))
            all_df = all_df.reindex(columns=cols)
            all_df.loc[len(all_df)] = [row.get(c, np.nan) for c in cols]
            all_df.round(4).to_csv(path, index=False)

        run_dir = os.path.dirname(progress_csv_path)
        historic_csv = os.path.join(run_dir, PRETRAIN_CONTINUED_CSV)
        save_csv(progress_csv_path, pb, epoch)
        # the historic stream continues pretraining's own epoch axis, which
        # advanced 20 at a time -- keep stepping by 20 rather than switching
        # to the retrain counter, so the two halves of the curve share a scale
        save_csv(historic_csv, hb, next_pretrain_epoch(historic_csv))
        print(f"{pfx} saved to {os.path.basename(progress_csv_path)} "
              f"+ {PRETRAIN_CONTINUED_CSV}")

        # every value is a list except pol_true, which stack_policies builds as
        # an (N, 1858) array -- `+` on that would add element-wise, not append
        def cat(a, b):
            if isinstance(a, np.ndarray):
                return np.concatenate([a, b])
            return a + b

        combined = {k: cat(primary[k], historic[k]) for k in primary}
        self.save_validation_plots(
            epoch, os.path.join(run_dir, 'validation_latest.png'), combined)

    def save_validation_plots(self, epoch, out_path, g):
        """Single 5-panel figure over the combined primary + historic
        validation draw, split by provenance wherever a split is meaningful."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not g['wdl_pred']:
            return

        sqrt3_2 = np.sqrt(3) / 2
        max_scatter = 5000
        eps = 1e-9

        pp = np.clip(np.array(g['wdl_pred'], dtype=np.float64), eps, 1.0)
        pp /= pp.sum(axis=1, keepdims=True)
        pt = np.array(g['wdl_true'], dtype=np.float64)
        src = np.array([plot_source(s) for s in g['source']])

        pred_v = pp[:, 0] - pp[:, 2]
        true_v = pt[:, 0] - pt[:, 2]
        per_ce = -np.sum(pt * np.log(pp), axis=1)
        mean_ce = np.mean(per_ce)
        mse = np.mean((pred_v - true_v) ** 2)
        corr = np.corrcoef(pred_v, true_v)[0, 1] if len(pred_v) > 1 else 0.0

        logits = np.array(g['pol_pred'], dtype=np.float32)
        true_p = np.array(g['pol_true'], dtype=np.float32)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        pred_probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        def topk_mass(k, sel):
            tp, pr = true_p[sel], pred_probs[sel]
            ki = np.argsort(tp, axis=1)[:, -k:]
            return np.mean(pr[np.arange(len(pr))[:, None], ki].sum(axis=1))

        present = [s for s in SOURCE_ORDER if (src == s).any()]
        n = len(pred_v)
        width = 0.8 / max(len(present), 1)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Epoch {epoch}: validation  n={n}", fontsize=12)

        # one shared source legend -- per-axes legends collide with the
        # rotated bar value labels on the grouped panels
        fig.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=SOURCE_COLORS[s], label=s)
                     for s in present],
            loc='upper right', ncol=len(present), fontsize=9, framealpha=0.9,
        )

        def grouped(ax, labels, per_source, fmt, title, ylabel, ylim=None):
            xpos = np.arange(len(labels))
            span = max(abs(v) for vals in per_source.values() for v in vals) or 1.0
            pad = span * 0.03
            for i, s in enumerate(present):
                off = (i - (len(present) - 1) / 2) * width
                ax.bar(xpos + off, per_source[s], width=width * 0.9,
                       color=SOURCE_COLORS[s], label=s)
                for x, v in zip(xpos + off, per_source[s]):
                    ax.text(x, v + (pad if v >= 0 else -pad), fmt.format(v),
                            ha='center', va='bottom' if v >= 0 else 'top',
                            fontsize=7, rotation=90)
            ax.set_xticks(xpos)
            ax.set_xticklabels(labels)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if ylim is None:
                ax.margins(y=0.30)
            else:
                ax.set_ylim(*ylim)

        # largest source first so it does not paint over the smaller ones
        per_plot = max_scatter // max(len(present), 1)
        counts = {s: int((src == s).sum()) for s in present}
        for s in sorted(present, key=lambda s: -counts[s]):
            m = np.flatnonzero(src == s)
            take = np.random.choice(m, min(per_plot, len(m)), replace=False)
            axes[0, 0].scatter(true_v[take], pred_v[take], s=4, alpha=0.18,
                               c=SOURCE_COLORS[s])
        axes[0, 0].set_xlabel("target value (W-L)")
        axes[0, 0].set_ylabel("pred value (W-L)")
        axes[0, 0].set_title(f"value scatter  MSE={mse:.3f}  r={corr:.3f}")
        # drawn largest-first so big sources do not bury small ones, but
        # legended in canonical order with opaque proxy markers -- the live
        # handles inherit the scatter alpha and are unreadable at this size
        axes[0, 0].legend(
            handles=[plt.Line2D([], [], marker='o', linestyle='', markersize=6,
                                color=SOURCE_COLORS[s], label=f"{s} (n={counts[s]})")
                     for s in present],
            loc='upper left', fontsize=8, framealpha=0.9)

        masses = {s: [topk_mass(k, src == s) for k in (1, 3, 5)] for s in present}
        grouped(axes[0, 1], ['top1', 'top3', 'top5'], masses, '{:.3f}',
                'policy top-k mass by source', 'mass', ylim=(0, 1.0))

        idx = np.random.choice(n, min(max_scatter, n), replace=False)
        tx = 0.5 * pp[idx, 0] + pp[idx, 1]
        ty = sqrt3_2 * pp[idx, 0]
        axes[0, 2].plot([0.5, 1.0, 0.0, 0.5], [sqrt3_2, 0.0, 0.0, sqrt3_2], 'k-', lw=0.8)
        sc = axes[0, 2].scatter(
            tx, ty, s=4, alpha=0.4,
            c=per_ce[idx], cmap='RdYlGn_r', vmin=0, vmax=2.0,
        )
        cbar = fig.colorbar(sc, ax=axes[0, 2], shrink=0.8)
        thresholds = np.arange(0, 2.01, 0.25)
        cdf_vals = [np.mean(per_ce <= t) for t in thresholds]
        cbar.set_ticks(thresholds)
        cbar.set_ticklabels([f'{t:.2f}  ({v:.2f})' for t, v in zip(thresholds, cdf_vals)])
        axes[0, 2].text(0.5, sqrt3_2 + 0.03, 'W', ha='center', va='bottom', fontsize=9)
        axes[0, 2].text(1.03, -0.03, 'D', ha='left', va='top', fontsize=9)
        axes[0, 2].text(-0.03, -0.03, 'L', ha='right', va='top', fontsize=9)
        axes[0, 2].set_aspect('equal')
        axes[0, 2].axis('off')
        axes[0, 2].set_title(f'WDL ternary  mean CE={mean_ce:.3f}', pad=18)

        wdl_labels = ['W', 'D', 'L']
        bias = {s: np.mean(pp[src == s] - pt[src == s], axis=0) for s in present}
        ce_comp = {
            s: -np.mean(pt[src == s] * np.log(pp[src == s]), axis=0)
            for s in present
        }

        grouped(axes[1, 0], wdl_labels, bias, '{:+.4f}',
                'WDL bias by source', 'mean(pred - target)')
        axes[1, 0].axhline(0, color='black', lw=0.8)

        grouped(axes[1, 1], wdl_labels, ce_comp, '{:.4f}',
                f'CE by component  total={mean_ce:.3f}',
                'mean CE contribution (nats)')

        axes[1, 2].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    def push_analyzed(self, report=True):
        # safeguard here
        if not len(self.analyzed_results):
            return
        
        run_dir = self.config.run_dir
        outp, c, b, kl, ce, plies, kl_n, ce_n, bl = save_analysis_chunk_simple(
            run_dir, self.analyzed_results)
        self.n_saved += 1
        self.total_cpl_plies += c * plies
        self.total_bmr_plies += b * plies
        self.total_blunder_plies += bl * plies
        self.total_plies += plies
        # kl/ce are undefined (NaN) for batches with no trainable plies
        # (e.g. all-validation rounds) -- weight/accumulate by valid plies
        # only, so they don't poison the running totals with NaN.
        if kl_n:
            self.total_kl_plies += kl * kl_n
            self.total_kl_valid_plies += kl_n
        if ce_n:
            self.total_ce_plies += ce * ce_n
            self.total_ce_valid_plies += ce_n

        self.recent_cpls.append(c)
        self.recent_cpls = self.recent_cpls[-RECENT_WINDOW:]
        self.recent_bmrs.append(b)
        self.recent_bmrs = self.recent_bmrs[-RECENT_WINDOW:]
        self.recent_blunders.append(bl)
        self.recent_blunders = self.recent_blunders[-RECENT_WINDOW:]
        self.recent_plies.append(plies)
        self.recent_plies = self.recent_plies[-RECENT_WINDOW:]
        if kl_n:
            self.recent_kls.append(kl)
            self.recent_kls = self.recent_kls[-RECENT_WINDOW:]
            self.recent_kl_plies.append(kl_n)
            self.recent_kl_plies = self.recent_kl_plies[-RECENT_WINDOW:]
        if ce_n:
            self.recent_ces.append(ce)
            self.recent_ces = self.recent_ces[-RECENT_WINDOW:]
            self.recent_ce_plies.append(ce_n)
            self.recent_ce_plies = self.recent_ce_plies[-RECENT_WINDOW:]

        if report:
            for n in (10, RECENT_WINDOW):
                if len(self.recent_cpls) < n:
                    continue
                print(fmt_rescore_stats(
                    f"Last {n} avg",
                    window_avg(self.recent_cpls, self.recent_plies, n),
                    window_avg(self.recent_bmrs, self.recent_plies, n),
                    window_avg(self.recent_ces, self.recent_ce_plies, n),
                    window_avg(self.recent_kls, self.recent_kl_plies, n),
                    window_avg(self.recent_blunders, self.recent_plies, n)))

            if self.n_saved >= 2:
                cpl_mean = self.total_cpl_plies / self.total_plies
                bmr_mean = self.total_bmr_plies / self.total_plies
                blunder_mean = self.total_blunder_plies / self.total_plies
                kl_mean = (self.total_kl_plies / self.total_kl_valid_plies
                           if self.total_kl_valid_plies else float('nan'))
                ce_mean = (self.total_ce_plies / self.total_ce_valid_plies
                           if self.total_ce_valid_plies else float('nan'))
                print(fmt_rescore_stats("Overall stats", cpl_mean, bmr_mean,
                                        ce_mean, kl_mean, blunder_mean))

            W = 10

            def col(val):
                return f"  {val:>{W}}  |"

            # true search depth reached under the time budget, averaged over
            # every SF call this window (both passes), not seldepth
            avg_depth = (self.sf_depth_sum / self.sf_depth_count
                         if self.sf_depth_count else 0.0)

            n_tot = self.games_processed
            if n_tot > self.config.rescore_analyze_batch and self.start_time is not None:
                elapsed = time.time() - self.start_time
                games_hr = n_tot / elapsed * 3600
                moves_sec = self.total_plies / elapsed
                tg_cols = [
                    ("games",      str(n_tot)),
                    ("games/hr",   f"{games_hr:.1f}"),
                    ("moves/sec",  f"{moves_sec:.2f}"),
                    ("replays",    str(self.n_replay_finalized)),
                    ("depth",      f"{avg_depth:.1f}"),
                ]
                tg_hdr = (f"{RS}  {'Total':<12} |"
                          + "".join(col(h) for h, _ in tg_cols))
                tg_row = (f"{RS}  {'':<12} |"
                          + "".join(col(v) for _, v in tg_cols))
                print(tg_hdr)
                print(tg_row)

            avg_ms_per_pos = (
                1000 * self.sf_compute_time / self.sf_call_count
                if self.sf_call_count else 0.0
            )
            avg_ms_per_ply = (
                1000 * self.sf_compute_time / self.sf_compute_count
                if self.sf_compute_count else 0.0
            )
            # share of plies where xerces already played SF's best move, so
            # the second pass was skipped
            first_run_pct = (
                100 * (self.sf_compute_count - self.sf_rerun_count)
                / self.sf_compute_count
                if self.sf_compute_count else 0.0
            )
            backlog = len(self.intake) + len(self.pending)

            sf_cols = [
                ("positions", str(self.n_sf_submitted)),
                ("ms/pos",    f"{avg_ms_per_pos:.0f}ms"),
                ("ms/ply",    f"{avg_ms_per_ply:.0f}ms"),
                ("1st run %", f"{first_run_pct:.1f}%"),
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
            self.sf_call_count = 0
            self.sf_rerun_count = 0
            self.sf_depth_sum = 0
            self.sf_depth_count = 0
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

        self.analyzed_results = []

# helpers
def value_weight_for_game(cfg, is_draw):
    return cfg.draw_value_scale if is_draw else 1.0


def update_ema_quantile(estimate, x, target_q, lr):
    """Online stochastic-approximation quantile tracker (Robbins-Monro).
    Self-corrects toward the point where P(X < estimate) == target_q."""
    if x < estimate:
        return estimate + lr * (target_q - 1.0)
    return estimate + lr * target_q


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


def fmt_rescore_stats(label, cpl, bmr, ce, kl, blunder):
    # CPL swings between 1 and 2+ digits -- fixed width keeps columns aligned
    return (f"{RS} {label:<14}:  CPL {cpl:>5.2f}  BMR {bmr:>4.2f}  "
            f"CE {ce:>4.2f}  KL {kl:>4.2f}  CPL>{BLUNDER_CP} {blunder:>5.3f}")


def migrate_pretrain_progress(run_dir):
    """First-run split of the unified progress csv.

    Pretraining writes its validation rows straight into eval_progress.csv
    (bootstrap's unified mode). Selfplay wants that file to hold only its own
    rows, counting from 0, with the pretraining axis carried on separately in
    PRETRAIN_CONTINUED_CSV -- which the historic validation stream then keeps
    extending by PRETRAIN_EPOCH_STEP.

    Call once at startup, before next_model_epoch reads the counter. No-op
    once PRETRAIN_CONTINUED_CSV exists, so it runs exactly one time per run.
    Everything in eval_progress.csv at that moment is pretraining output --
    selfplay has not written a row yet -- so the whole file moves.
    """
    progress = os.path.join(run_dir, "eval_progress.csv")
    historic = os.path.join(run_dir, PRETRAIN_CONTINUED_CSV)

    if os.path.exists(historic) or not os.path.exists(progress):
        return

    df = pd.read_csv(progress)
    if not len(df):
        return

    df.to_csv(historic, index=False)
    # keep the schema, drop the rows: next_model_epoch then returns 0 and the
    # selfplay counter starts clean
    df.iloc[0:0].to_csv(progress, index=False)
    lo, hi = int(df.model_epoch.min()), int(df.model_epoch.max())
    print(f"{RS} moved {len(df)} pretrain rows (epoch {lo}..{hi}) -> "
          f"{PRETRAIN_CONTINUED_CSV}; eval_progress.csv restarts at 0")


def next_pretrain_epoch(path, step=PRETRAIN_EPOCH_STEP):
    """Next model_epoch for the pretrain-continued file: one step past the
    last row. Pretraining validated every `step` epochs and this file just
    keeps that axis going."""
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path)
    if not len(df) or 'model_epoch' not in df.columns:
        return 0
    return int(df['model_epoch'].max()) + step


def window_avg(vals, wts, n):
    """Ply-weighted mean over the last n entries. vals and wts are the same
    window (kl/ce keep their own, since chunks with no trainable plies never
    append to them)."""
    w = np.array(wts[-n:], dtype=np.float64)
    if not len(w) or not w.sum():
        return float('nan')
    return np.dot(vals[-n:], w) / w.sum()


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
    # rate of plies dropping more than BLUNDER_CP -- CPL is a mean and a few
    # huge losses move it as much as a broad decline, so track the tail too
    blunder = (df_all.delta > BLUNDER_CP).mean()
    kl_n = int(df_all.kl.notna().sum())
    ce_n = int(df_all.ce.notna().sum())
    kl  = df_all.kl.mean()
    ce  = df_all.ce.mean()
    plies = len(df_all)

    print(f"{RS} Saving {len(batch)} analyzed games")
    print(fmt_rescore_stats("Batch stats", cpl, bmr, ce, kl, blunder))

    fname = f"{int(time.time())}_{uuid.uuid4().hex}.pkl"
    outp = os.path.join(staging, fname)

    with open(outp, "wb") as f:
        pickle.dump(chunk_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return outp, cpl, bmr, kl, ce, plies, kl_n, ce_n, blunder


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


