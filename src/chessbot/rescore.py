import os, json, gzip, time
from types import SimpleNamespace
from pathlib import Path
import uuid
import random
import threading
from collections import deque, defaultdict
from queue import Empty

import pickle
import chess, chess.engine

import pandas as pd
import numpy as np

from pyfastchess import Board

from chessbot import SF_LOC
from chessbot.config import XC0H_K
from chessbot.utils import (
    score_cp_stm_pov, score_cp_white_pov, rnd, kl_divergence, cross_entropy,
    scalar_to_wdl, cp_to_value_tanh, expected_score_loss,
    ema_step, load_json, save_pickle_atomic, blend_to_uniform,
    QUALITY, ReportRow, report_row_name, cell,
)
from chessbot.replay_buffer import (
    LiveBuffer, SF_SOURCE, sparsify_policy,
    new_shard_path, write_pkl_gz_shard, flush_correction_shards, is_lc0_record,
)
from chessbot.opening_counts import (
    COUNTS_FILE, REWEIGHT_INELIGIBLE, OpeningCounts, band_weights,
)

from chessbot.lc0_replay import (
    lc0_replay_spec,
)

from chessbot.game_utils import reconcile_game_boards

from chessbot.move_scoring import make_scorer

from chessbot.review import load_game_index, ANALYZE_PKL

RS = "[rescore]"
LC0_BRP_SHARD_DIR = os.environ.get("LC0_BRP_SHARD_PATH", "")
# lc0 audit smoothing, counted in scored positions
AUDIT_EMA_SPAN = 2000
# blunder rule: a rescore_esl_blunder slip in a game the mover lost or drew,
# an ESL_ANY_MIN slip in any game, or F (a won position thrown away)
ESL_ANY_MIN = 0.20
F_MIN_V = 1200
F_MAX_P = 1000
# teacher stream: plies the blunder path did not take,
# sent to lc0 for one ply when ESL clears rescore_esl_inaccuracy and xc0's
# search disagrees with SF's move (see policy_value_miss). The xc0 sample
# still trains, at LC0_SENT_XC0_WEIGHT.
# weight multiplier on an xc0 sample whose position also went to lc0
LC0_SENT_XC0_WEIGHT = 0.5
# admit: every live_buffer sample is accepted on these rules. A move SF played
# counts as TSL 0. KLm = max(KLv / KLv_p80, KLp / KLp_p80), clipped to
# [ACCEPT_KL_FLOOR, 1]; a right move (TSL <= ACCEPT_TSL_GOOD) that was extended,
# had a lock rejected or has KLm 1 is always taken, the rest are drawn at
# max(ACCEPT_FLOOR, draw * opening band weight * KLm)
ACCEPT_TSL_GOOD = 50
ACCEPT_KL_FLOOR = 0.4
ACCEPT_FLOOR = 0.05
ACCEPT_DRAW_P = 0.8
# a forced move teaches nothing; a few stay so the model keeps seeing them
ACCEPT_ONE_LEGAL_P = 0.2
LC0_TEACHER_TOP_K = 5
LC0_TEACHER_PV_K = 2
BLUNDER_STAT_KEYS = (
    "total", "missed_mate", "inaccuracy", "blunder", "blunder_lost",
    "blunder_any", "F",
)


def is_esl_candidate(v, p, esl, z_stm, esl_blunder):
    """Scalar ESL rule: E for the ESL thresholds, F for a thrown-away won
    position, F first. Returns (is_candidate, label)."""
    if z_stm <= 0 and v >= F_MIN_V and p < F_MAX_P:
        return True, "F"
    if esl >= ESL_ANY_MIN or (z_stm <= 0 and esl >= esl_blunder):
        return True, "E"
    return False, None


def policy_value_miss(cm, sf_uci):
    """SF's move outside xc0's top K by visits or by prior, or top PV_K by
    prior yet outside the top PV_K by visits -- policy and value disagree."""
    by_visits = [c['uci'] for c in
                 sorted(cm, key=lambda c: c['visits'], reverse=True)]
    by_prior = [c['uci'] for c in sorted(cm, key=lambda c: c['P'], reverse=True)]
    k, pv_k = LC0_TEACHER_TOP_K, LC0_TEACHER_PV_K
    return (sf_uci not in by_visits[:k]
            or sf_uci not in by_prior[:k]
            or (sf_uci in by_prior[:pv_k] and sf_uci not in by_visits[:pv_k]))


# the rescore report (scenario table, trends, throughput, targets) prints
# every REPORT_EVERY analyzed games
REPORT_EVERY = 100
# game-weighted EMA spans for the trend lines. The shortest is a running
# mean for its first span of games; each longer one starts at the previous
# span's game count, seeded with that EMA, and shows once its own span is in.
TREND_SPANS = (100, 500, 2000)
# per-ply TSL thresholds for the mistake / blunder rates in the trend lines
TSL_MISTAKE = 50
TSL_BLUNDER = 120
# per-ply EMA span for the trend table's acc row (admitted xc0 plies)
ACC_EMA_SPAN = 20000
# stop/blunder stat blocks print every N analyzed games; their window
# accumulators reset on print, so this is the cadence and the window both
STATS_WINDOW_GAMES = 200
CTRL_WINDOW_GAMES = 100
# SF rescore search caps. cfg.rescore_movetime_ms is the tunable budget; the
# catch-up trim and the depth backstop stay fixed so there is one knob.

C = 0.9699
D = d = np.arctanh(0.5)

# Rescore shortcuts. A move that follows an earlier search's PV is not
# searched: CPL 0, scored at the bookend's eval, once the bookend (the
# position where the game leaves the PV, searched anyway) confirms the PV's
# eval within PV_CONFIRM_ES expected score; otherwise every skipped move is
# searched. Positions with search |Q| or D above DECIDED_QD get
# DECIDED_TIME_FRAC of the movetime; other positions from BUSY_MIN_PLY on
# (game ply from startpos, by the FEN's move counters) with more than
# BUSY_LEGAL legal moves or BUSY_MATERIAL material get BUSY_EXTRA_MS more.
# An extended controller search that changed its leader has both moves
# scored in one multipv search at PAIR_TIME_MULT of the movetime.
PV_CONFIRM_ES = 0.02
DECIDED_QD = 0.9
DECIDED_TIME_FRAC = 0.8
BUSY_MIN_PLY = 10
BUSY_LEGAL = 30
BUSY_MATERIAL = 60
BUSY_EXTRA_MS = 20
PAIR_TIME_MULT = 2.25
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
                chess.QUEEN: 9}


def ctrl_shaky(tr):
    """The controller extended this move or rejected one of its locks."""
    cs = tr.get('ctrl_sum') or {}
    return bool(cs.get('ext') or cs.get('rejections'))


def material(board):
    """Both sides' material in pawns."""
    return sum(v * len(board.pieces(pt, c))
               for pt, v in PIECE_VALUES.items() for c in chess.COLORS)


class CtrlStats:
    """Running sums of the search controller's per-move outcomes. Each group
    is [moves, sims, TSL, CPL, blunders]."""
    GROUPS = ("ctrl", "lock", "ext ctrl", "ext lock", "other")

    def __init__(self):
        self.groups = {g: [0, 0.0, 0.0, 0.0, 0] for g in self.GROUPS}
        self.all = [0, 0.0, 0.0, 0.0, 0]
        self.old = [0, 0.0, 0.0, 0.0, 0]
        self.n = 0
        self.sims = self.sched = 0          # moves the controller ran
        self.old_sims = self.old_sched = 0
        self.n_ext = 0
        self.n_multi = 0                    # moves extended more than once
        self.ext_extra = self.ext_changed = 0.0
        # extended moves: [moves, TSL] for a kept leader; [moves, TSL,
        # rejected TSL] where the leader changed, both from the side-by-side
        # pair search
        self.ext_same = [0, 0.0]
        self.ext_chg = [0, 0.0, 0.0]
        self.ms = self.checkins = self.stab = self.qual = self.ties = 0.0

    @staticmethod
    def bump(g, sims, cpl, esl, blunder):
        g[0] += 1
        g[1] += sims
        g[2] += 1000 * esl
        g[3] += cpl
        g[4] += esl >= blunder

    def add_ctrl(self, c, reason, sims, cpl, esl, played, blunder, pair_esl=None):
        group = ("ext " if c['ext'] else "") + reason
        if group not in self.GROUPS:
            group = "other"
        if c['ext']:
            self.n_ext += 1
            self.ext_extra += sims - c['sims_max_start']
            self.ext_changed += played != c['rejected']
            self.n_multi += c['ext'] >= 2
            if played == c['rejected']:
                self.ext_same[0] += 1
                self.ext_same[1] += 1000 * esl
            elif pair_esl is not None:
                self.ext_chg[0] += 1
                self.ext_chg[1] += 1000 * pair_esl[0]
                self.ext_chg[2] += 1000 * pair_esl[1]
        self.bump(self.groups[group], sims, cpl, esl, blunder)
        self.bump(self.all, sims, cpl, esl, blunder)
        self.n += 1
        if c['checkins']:
            self.sims += sims
            self.sched += c['sims_max_start']
        self.ms += c['ms']
        self.checkins += c['checkins']
        self.stab += c['stab']
        self.qual += c['qual']
        self.ties += c['ties']

    def add_old(self, sims, cpl, esl, sims_max, blunder):
        self.bump(self.old, sims, cpl, esl, blunder)
        self.old_sims += sims
        self.old_sched += sims_max or 0


class Trend:
    """Game-weighted EMAs of one metric at TREND_SPANS (see there)."""

    def __init__(self):
        self.n = 0
        self.vals = [None] * len(TREND_SPANS)

    def add(self, x):
        self.n += 1
        for i, span in enumerate(TREND_SPANS):
            v = self.vals[i]
            if v is None:
                if i == 0:
                    self.vals[0] = x
                continue
            a = 1.0 / self.n if i == 0 and self.n <= span else 2.0 / (span + 1)
            self.vals[i] = v + a * (x - v)
        for i in range(1, len(TREND_SPANS)):
            if self.vals[i] is None and self.n == TREND_SPANS[i - 1]:
                self.vals[i] = self.vals[i - 1]

    def value(self, i):
        """The i-th span's value once it has seen that many games, else None."""
        return self.vals[i] if self.n >= TREND_SPANS[i] else None


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
        self.movetime_ms = cfg.rescore_movetime_ms
        self.sf_config = {'Threads': 1, 'Hash': 256}
        self.stop_ev = threading.Event()
        self.t = None
        self.eng = None
        # assigned by the runner, which owns the fleet and its numbering
        self.id = None

    def submit_game(self, gid, positions, movetime_ms=None):
        info = {"movetime_ms": movetime_ms} if movetime_ms is not None else {}
        self.game_q.put((gid, positions, info))

    def update_config(self, cfg):
        self.movetime_ms = cfg.rescore_movetime_ms

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

    def search(self, ply_idx, board, xerces_uci, limits, rejected=None):
        """Best move and xc0's move under limits, a (pass1, pass2) pair. When
        xc0 played something other than the controller's rejected leader, both
        moves are also scored side by side at PAIR_TIME_MULT of the movetime."""
        pass1_limit, pass2_limit = limits
        elapsed = 0.0
        depths = []
        t0 = time.time()
        info = self.eng.analyse(board, pass1_limit, info=chess.engine.INFO_ALL)
        elapsed += time.time() - t0
        depth = info.get('depth', 0)
        pv = [str(m) for m in info.get('pv', [])]
        best_uci = pv[0]
        best_cp = score_cp_stm_pov(info['score'])
        best_abs = score_cp_white_pov(info['score'])

        # pass 2: score xerces's move only if it differs from best
        rerun = xerces_uci != best_uci
        if rerun:
            t0 = time.time()
            info2 = self.eng.analyse(board, pass2_limit, info=chess.engine.INFO_ALL,
                                     root_moves=[chess.Move.from_uci(xerces_uci)])
            elapsed += time.time() - t0
            depths.append(info2.get('depth', 0))
            played_cp = score_cp_stm_pov(info2['score'])
            played_abs = score_cp_white_pov(info2['score'])
        else:
            played_cp, played_abs = best_cp, best_abs
            depths.append(depth)

        pair_cp = None
        if rejected and rejected != xerces_uci:
            t0 = time.time()
            infos = self.eng.analyse(
                board, chess.engine.Limit(time=self.movetime_ms * PAIR_TIME_MULT / 1000.0,
                                          depth=self.max_depth),
                multipv=2,
                root_moves=[chess.Move.from_uci(m) for m in (xerces_uci, rejected)])
            elapsed += time.time() - t0
            cps = {str(i['pv'][0]): score_cp_stm_pov(i['score'])
                   for i in infos if i.get('pv')}
            if xerces_uci in cps and rejected in cps:
                pair_cp = (cps[xerces_uci], cps[rejected])

        return {
            'pair_cp':    pair_cp,    # (xc0's move, rejected leader)
            'ply_idx':    ply_idx,
            'best_uci':   best_uci,
            'pv_ucis':    pv[:3],
            'pv':         pv,
            'best_cp':    best_cp,
            'best_abs':   best_abs,
            'played_cp':  played_cp,
            'played_abs': played_abs,
            'elapsed':    elapsed,
            'best_depth': depth,      # pass 1, the free search
            'depth':      depths[0],  # pass 2, the played move
            'depths':     depths,
            'rerun':      rerun,
        }

    @staticmethod
    def unsearched(ply_idx, board, xerces_uci, abs_cp, depth, pv, why):
        """A result scored without a search: xc0's move is the best move, at
        a white-POV eval taken from elsewhere."""
        cp = abs_cp if board.turn == chess.WHITE else -abs_cp
        return {
            'ply_idx': ply_idx, 'best_uci': xerces_uci, 'pv_ucis': pv[:3], 'pv': pv,
            'best_cp': cp, 'best_abs': abs_cp, 'played_cp': cp, 'played_abs': abs_cp,
            'elapsed': 0.0, 'best_depth': depth, 'depth': depth, 'depths': [],
            'rerun': False, 'skipped': why,
        }

    def rescore_game(self, positions, moves, tiers):
        """One game's positions in ply order. Forced moves are not searched.
        A move on the PV of the last searched position is not searched
        either. The bookend is the next searched position still on the PV's
        line (where xc0 left it: SF's best and the move played, as for any
        position); if its eval matches the PV's, the skipped moves get CPL 0
        at the bookend's eval, else they are searched. With no such position
        (the game ends, or a move off the PV came first) the last skipped
        move is searched as the bookend."""
        out = {}
        origin = None   # (ply, result) whose PV the game is following
        chain = []      # positions on origin's PV, not searched yet
        last_abs = 0.0

        def on_line(k):
            # the game reached k along origin's PV
            o_ply, o_res = origin
            off = k - o_ply
            return 0 < off <= len(o_res['pv']) and moves[o_ply:k] == o_res['pv'][:off]

        def on_pv(k, xerces_uci):
            off = k - origin[0]
            pv = origin[1]['pv']
            return on_line(k) and off < len(pv) and xerces_uci == pv[off]

        def settle(k=None, book=None):
            if not chain:
                return
            if book is None or not on_line(k):
                last = chain.pop()
                book = self.search(last[0], last[1], last[2], tiers[last[3]])
                out[last[0]] = book
            o_res = origin[1]
            # expected-score gap either way, white POV
            gap = max(expected_score_loss(o_res['best_abs'], book['best_abs']),
                      expected_score_loss(book['best_abs'], o_res['best_abs']))
            for c, board, xerces_uci, tier, _ in chain:
                off = c - origin[0]
                out[c] = (self.unsearched(c, board, xerces_uci, book['best_abs'],
                                          book['best_depth'], o_res['pv'][off:], "pv")
                          if gap <= PV_CONFIRM_ES
                          else self.search(c, board, xerces_uci, tiers[tier]))
            chain.clear()

        for pos in positions:
            k, board, xerces_uci, tier, rejected = pos
            if board.legal_moves.count() == 1:
                out[k] = self.unsearched(k, board, xerces_uci, last_abs, 0, [], "forced")
                continue
            # a rejected leader needs its own score, so it is never skipped
            if origin is not None and not rejected and on_pv(k, xerces_uci):
                chain.append(pos)
                continue
            r = self.search(k, board, xerces_uci, tiers[tier], rejected)
            settle(k, r)
            out[k] = r
            last_abs = r['best_abs']
            origin = (k, r) if r['pv'] else None
        settle()
        return [out[k] for k in sorted(out)]

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

            def timed(ms):
                # longer search for pass2 because Xc0 move may be better
                return (chess.engine.Limit(time=ms / 1000.0, depth=self.max_depth),
                        chess.engine.Limit(time=(ms + 20) / 1000.0, depth=self.max_depth))

            tiers = {None: timed(movetime_ms),
                     "decided": timed(movetime_ms * DECIDED_TIME_FRAC),
                     "busy": timed(movetime_ms + BUSY_EXTRA_MS)}

            try:
                if info.get("moves") is None:
                    results = [self.search(pos[0], pos[1], pos[2], tiers[None])
                               for pos in positions]
                else:
                    results = self.rescore_game(positions, info["moves"], tiers)
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
        # the fast track: lc0 blunder corrections, written as their own
        # shards. Never flushed on its own; flush_correction_shards drains it.
        self.correction_buffer = LiveBuffer(cfg.primary_buffer_dir)
        self.analyzed_results = []
        self.closed = False

        self.start_time = None  # set on first game to exclude idle startup time
        self.games_seen = set()
        self.games_processed = 0
        self.written_total = 0
        self.written_this_round = 0
        self.total_plies = 0
        # since start: rows by scenario and selfplay quality trends
        # (validation games stay out of the trends)
        self.report_rows = defaultdict(ReportRow)
        self.trends = {m: Trend() for m in (*QUALITY, "tsl_mistake", "tsl_blunder")}
        # seeded at observed admitted-ply levels
        self.acc_ema = {'cpl': 4.81, 'tsl': 6.03, 'klv': 0.115, 'klp': 0.47,
                        'tsl_mistake': 0.045, 'tsl_blunder': 0.002}

        self.n_sf_submitted = 0
        self.sf_compute_time = 0.0
        self.sf_compute_count = 0
        self.sf_call_count = 0
        self.sf_rerun_count = 0
        self.sf_skipped = 0
        self.n_sf_workers = cfg.rescore_n_sf_workers
        self.sf_depth_sum = 0
        self.sf_depth_count = 0
        self.current_movetime_ms = 0

        zero_stop = lambda: {'n': 0, 'cpl': 0.0, 'bmr': 0.0, 'sims': 0.0}
        self.total_stop = {st: zero_stop() for st in ("full", "tier1", "tier2")}
        self.window_stop = {st: zero_stop() for st in ("full", "tier1", "tier2")}

        # search controller, xc0's own moves only; printed and dropped every
        # CTRL_WINDOW_GAMES games
        # running sums since this process started; printed every
        # CTRL_WINDOW_GAMES games
        self.ctrl_acc = CtrlStats()
        self.ctrl_games = 0

        zero_blunder = lambda: {'n': 0, 'cpl': 0.0, 'lc0': 0}
        self.total_blunder = {st: zero_blunder() for st in BLUNDER_STAT_KEYS}
        self.window_blunder = {st: zero_blunder() for st in BLUNDER_STAT_KEYS}

        self.sample_counts = zero_sample_counts()
        self.sample_counts_window = zero_sample_counts()

        self.scorer = make_scorer('rescore', cfg)
        self.lc0_scorer = make_scorer('lc0', cfg)

        # TEMPORARY lc0 audit: teacher output sent through SF to score it.
        # Remove this and its call sites once lc0 quality is trusted.
        self.lc0_audit = {}
        self.lc0_audit_n = 0
        # EMA rather than a lifetime mean: a mid-run change to the sims budget
        # or the scoring rules should show up rather than be diluted forever.
        # ply0 lc0 and xc0 only ever step together, so uplift always reconciles.
        # seeded at the observed teacher/xerces loss so one lucky first
        # probe does not own the number for thousands of samples
        self.lc0_audit_ema = {'lc0': 9.4, 'xc0': 97.6, 'lc0_tsl': 15.0, 'xc0_tsl': 120.0}
        # ply0 audits, and those lc0 answered within ACCEPT_TSL_GOOD
        self.lc0_confirm_n = 0
        self.lc0_confirm_ok = 0

        self.kl_q50 = 0.21  # online-tracked EMA median of eligible-ply KL
        self.kl_q80 = 0.55  # online-tracked EMA 80th percentile of eligible-ply KL
        self.kl_value_q50 = 0.024  # EMA median of eligible-ply KL(target, root_wdl_nn)
        self.kl_value_q80 = 0.124  # EMA 80th percentile of the same

        self.intake = deque()
        self.pending = {}

        # teacher probes staged here; the runner chains them through the lc0
        # server. lc0_cfg stays None when the run has no lc0 server, which
        # turns queue_lc0_replay into a no-op.
        self.lc0_pending = []
        self.lc0_cfg = None

        # fills from lc0 replays whose ply0 clears the SF audit; flushed to a
        # shard at 10240, for the next pretraining run, not selfplay training
        self.lc0_pretrain_buffer = []

        self.init_analyzer()

    def close(self):
        """Idempotent -- the runner closes on the normal path and again in
        cleanup, and the counts are unchanged between the two."""
        if self.closed:
            return
        self.closed = True
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
        return len(self.live_buffer) + len(self.correction_buffer)

    def buffered_records(self):
        """The live buffer, the on-disk live_buffer.pkl form."""
        return self.live_buffer.records

    def correction_records(self):
        """The correction buffer, the on-disk correction_buffer.pkl form."""
        return self.correction_buffer.records

    def load_buffered_records(self, records, correction=None):
        """Load saved buffers back and write any shards they already fill.
        With no correction file the run predates it, and its live buffer
        file holds both, told apart by source (lc0 then meant correction)."""
        if correction is None:
            correction = [r for r in records if is_lc0_record(r)]
            records = [r for r in records if not is_lc0_record(r)]
        self.live_buffer.load_records(records)
        self.correction_buffer.load_records(correction, REWEIGHT_INELIGIBLE)
        return self.flush_buffers()

    def flush_buffers(self):
        return (self.live_buffer.flush_all_ready()
                + flush_correction_shards(self.correction_buffer))

    def submit(self, pkl_file):
        self.intake.append(pkl_file)

    def tick(self):
        # drain completed game batches from SF threads
        while True:
            try:
                gid, results = self.res_q.get_nowait()
            except Empty:
                break
            if gid == '__error__':
                raise results
            self.handle_game_results(gid, results)

        flushed = self.flush_buffers()
        if flushed:
            n = len(flushed) * self.live_buffer.shard_size
            self.written_this_round += n
            self.written_total += n

        if len(self.intake) > len(self.pending):
            intake_list = list(self.intake)
            random.shuffle(intake_list)
            self.intake = deque(intake_list)

        while self.intake and len(self.pending) < 40:
            self.start_game(self.intake.popleft())

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
            df_all = combined.get("df_all")
            if df_all is not None:
                self.games_seen = set(df_all["game_id"].astype(str).unique())

    def get_unprocessed(self):
        """Staged games not yet processed.

        write_lean_record writes <gid>.pkl.gz, appends the final game_index.json
        row, then deletes the looper .pkl, in that order. So a game is done once
        it has a final row (which guarantees its gz), and a surviving .pkl marks
        an unprocessed game. Resume off the staging inbox: reprocess any staged
        game not yet in the final index whose .pkl is still on disk, clearing a
        partial gz left by a crash between the gz write and the row append.
        """
        game_dir = self.config.game_dir
        final = load_game_index(self.config.game_index_file)
        final = [final] if isinstance(final, dict) else final
        done = {str(r.get("game_id")) for r in final}

        staging_path = self.config.game_index_staging_file
        staging = load_json(staging_path) if os.path.exists(staging_path) else []
        staging = [staging] if isinstance(staging, dict) else staging

        unprocessed_games = deque()
        for rec in staging:
            gid = str(rec.get("game_id"))
            if gid in done or gid in self.games_seen:
                continue
            raw_pkl = os.path.join(game_dir, gid + ".pkl")
            if not os.path.exists(raw_pkl):
                continue  # no source to reprocess from; skip (rare, accepted)
            gz = os.path.join(game_dir, gid + ".pkl.gz")
            if os.path.exists(gz):
                os.remove(gz)  # partial gz from a pre-row crash; redo clean
            rec["pkl_file"] = raw_pkl
            unprocessed_games.append(rec)

        if len(unprocessed_games):
            print(f"[rescore] {len(unprocessed_games)} unprocessed game(s) currently in queue")

        return unprocessed_games

    def reset_writer(self):
        self.written_this_round = 0
    
    def encode_board(self, board):
        """Board -> model input, per cfg.encoding_type. lc0 planes are raw uint8
        (rule50 unscaled); the /99 + float conversion happens at the model-input
        boundary (retrain / infer). board must carry full move history for lc0."""
        if self.config.encoding_type == "lc0":
            return board.lc0_features()
        if self.config.encoding_type == "xc0h":
            return board.history_tokens(XC0H_K)
        return board.encode_64_tokens()

    def track_accepted(self, cpl, tsl, klv, klp):
        a = 2.0 / (ACC_EMA_SPAN + 1.0)
        for k, v in (('cpl', cpl), ('tsl', tsl), ('klv', klv), ('klp', klp),
                     ('tsl_mistake', tsl > TSL_MISTAKE),
                     ('tsl_blunder', tsl > TSL_BLUNDER)):
            self.acc_ema[k] += a * (float(v) - self.acc_ema[k])

    @staticmethod
    def target_kl(Y, node):
        nn = node.get('root_wdl_nn')
        return float(kl_divergence(Y, nn)) if nn is not None else 0.0

    def admit(self, entry, okey, tsl, klv, klp, shaky, is_draw, n_legal):
        """The one way into live_buffer. Returns True if the sample was taken."""
        if n_legal <= 1:
            ok = random.random() < ACCEPT_ONE_LEGAL_P
        else:
            klm = max(klv / max(self.kl_value_q80, 1e-6),
                      klp / max(self.kl_q80, 1e-6))
            klm = min(max(klm, ACCEPT_KL_FLOOR), 1.0)
            if tsl <= ACCEPT_TSL_GOOD and (shaky or klm >= 1.0):
                ok = True
            else:
                freq = (1.0 if okey in (None, REWEIGHT_INELIGIBLE)
                        else float(band_weights(self.opening_counts.counts[okey])))
                p = (ACCEPT_DRAW_P if is_draw else 1.0) * freq * klm
                ok = random.random() < max(ACCEPT_FLOOR, p)
        if ok:
            self.live_buffer.append(entry, okey)
        return ok

    def count_offer(self, kind, ok):
        """Offered / accepted counts for an expert's plies (sf or vsl)."""
        for c in (self.sample_counts, self.sample_counts_window):
            c[f'{kind}_offered'] += 1
            c[f'{kind}_accepted'] += int(ok)

    def make_policy_example(self, board, ucis, visits):
        indices = board.moves_to_indices(ucis)
        policy = np.zeros(1858, dtype=np.float32)
        s = sum(visits)
        pi = np.array([v / s for v in visits], dtype=np.float32)
        pi = blend_to_uniform(pi, 1e-6, self.config.prior_clip_max)
        for idx, p in zip(indices, pi):
            policy[idx] += p
        x = self.encode_board(board)
        return x, policy

    def append_flat_policy_example(self, board, ucis, visits, Y, vwht, pwht,
                                   source='xc0'):
        x, policy = self.make_policy_example(board, ucis, visits)
        okey = self.opening_counts.key(board, x)
        self.opening_counts.bump(okey)
        self.live_buffer.append(
            (x, None, sparsify_policy(policy), Y, vwht, pwht, source), okey)
        return policy

    def training_data_from_sf(self, board, mv, tr, sf_won):
        """One SF ply. Policy: Xerces's search when SF played the search's
        top move; otherwise sf_move_target on SF's move blended with the
        search, SF weight SF_POLICY_W_WON if SF won the game
        else SF_POLICY_W_OTHER. The clip is applied after the blend. The KL
        boost matches a Xerces ply. Returns (x, dense policy, okey, pwht, kl,
        n_legal); finalize_game attaches the value target and admits it."""
        cfg = self.config
        cm = tr['candidate_moves']
        lms = board.legal_moves()
        idx_uci = dict(zip(board.moves_to_indices(lms), lms))
        vmap = {u: 1.0 for u in lms}
        for idx, v in zip(*tr['raw_visits']):
            u = idx_uci.get(int(idx))
            if u is not None:
                vmap[u] = max(1.0, float(v))
        ucis = list(vmap)
        xc0 = np.array([vmap[u] for u in ucis], dtype=np.float64)

        pi = xc0 / xc0.sum()
        if mv != max(vmap, key=vmap.get):
            w = SF_POLICY_W_WON if sf_won else SF_POLICY_W_OTHER
            pi = w * sf_move_target(ucis, mv) + (1 - w) * pi

        pwht = 1.0
        priors_map = {c['uci']: c.get('P_raw', c['P']) for c in cm}
        kl = kl_divergence(list(xc0), [priors_map.get(u, 0.0) for u in ucis])
        if kl >= self.kl_q80:
            pwht *= cfg.kl_boost_p80_mult

        x, policy = self.make_policy_example(board, ucis, list(pi))
        return (x, policy, self.opening_counts.key(board, x), pwht, kl,
                len(lms))

    def training_data_from_lc0(self, board, tr):
        """One lc0 ply of a vs_leela game, built as an xc0 ply is: lc0's
        visits are the policy target and the ply's NN fields are xc0's, so the
        KL is lc0's target against xc0's priors. Returns (x, dense policy,
        okey, pwht, kl, n_legal); finalize_game attaches the value target and
        admits it."""
        cm = tr['candidate_moves']
        lms = board.legal_moves()
        vmap = {c['uci']: max(1, int(c['visits'])) for c in cm}
        for u in lms:
            vmap.setdefault(u, 1)
        priors_map = {c['uci']: c.get('P_raw', c['P']) for c in cm}
        kl = kl_divergence(list(vmap.values()),
                           [priors_map.get(u, 0.0) for u in vmap])
        pwht = self.config.kl_boost_p80_mult if kl >= self.kl_q80 else 1.0
        x = tr['xc0h']
        policy = np.zeros(1858, dtype=np.float32)
        for idx, prob in tr['policy']:
            policy[idx] = prob
        return (x, policy, self.opening_counts.key(board, x), pwht, kl, len(lms))

    def start_game(self, pkl_file):
        path = str(pkl_file)
        opener = gzip.open if path.lower().endswith('.gz') else open
        with opener(path, 'rb') as f:
            record = pickle.load(f)

        header = record['header']
        plies = record['plies']
        gid = str(header.get('game_id'))
        if gid in self.games_seen:
            return

        cfg = self.config
        moves_played = [p['move_played'] for p in plies]
        board_ch, b_fast = reconcile_game_boards(
            header['start_fen'], header.get('history_uci'), moves_played,
        )

        # working game object: the FATTY header carried forward plus the
        # derived move list the lc0-replay helpers consume
        game_data = {**header, 'moves_played': moves_played}
        tree_data = {i: p for i, p in enumerate(plies)}
        vs_stockfish = header.get('vs_stockfish', False)
        sf_color = header.get('stockfish_color')
        result = header['result']

        is_validation_game = 'validation' in header.get('scenario', '').lower()
        skip_all_training = is_validation_game

        # snapshot config fields so hot-reloads don't affect an in-flight game.
        # prior_clip_max and the KL boosts are read live instead: a reload of
        # those applies to every target built from then on.
        game_cfg = SimpleNamespace(
            train_on_stockfish=cfg.train_on_stockfish,
            rescore_equiv_min=cfg.rescore_equiv_min,
            rescore_equiv_max=cfg.rescore_equiv_max,
            rescore_esl_inaccuracy=cfg.rescore_esl_inaccuracy,
            rescore_esl_blunder=cfg.rescore_esl_blunder,
            rescore_analyze_batch=cfg.rescore_analyze_batch,
            vscale=cfg.vscale,
        )

        game_state = {
            'gid': gid,
            'game_data': game_data,
            'header': header,
            'plies': plies,
            'result': result,
            'pkl_path': path,
            'cfg': game_cfg,
            'ply_states': [],
            # (ply_i, stm, outcome wdl, example parts) per trained SF ply
            'sf_examples': [],
            'lc0_examples': [],
            'waiting': False,  # True once a batch has been submitted to a SF thread
        }

        outcomes = deviation_outcomes(
            plies, board_ch.turn, result, cfg.value_bootstrap_at_sample)

        # (ply_idx, board_copy, xerces_uci) queued for SF analysis
        to_sf_positions = []
        # walk each move: route SF moves, build ply state
        for i, mv in enumerate(game_data.get('moves_played', [])):
            move_ch = chess.Move.from_uci(mv)
            is_sf_move = vs_stockfish and (board_ch.turn == sf_color)
            turn = board_ch.turn
            tr = plies[i]
            tr['stm'] = bool(turn)
            tr['z_wdl'] = z_to_wdl(result if turn else -result)
            outcome = outcomes[i] if turn else outcomes[i][::-1].copy()
            tr['wdl_target'] = blend_wdl_stm(outcome, tr.get('best_wdl'))
            cm = tr.get('candidate_moves', [])
            if cm:
                tr['policy'] = policy_from_raw_visits(
                    tr['raw_visits'], cfg.prior_clip_max)
                tr['kl'] = float(kl_divergence(
                    [c['visits'] for c in cm],
                    [c.get('P_raw', c['P']) for c in cm]))
            else:
                tr['policy'] = []
                tr['kl'] = 0.0
            wdl, nn_wdl = tr.get('best_wdl'), tr.get('root_wdl_nn')
            tr['kl_value'] = (float(kl_divergence(wdl, nn_wdl))
                              if wdl is not None and nn_wdl is not None else 0.0)

            if is_sf_move and ((not cfg.train_on_stockfish) or skip_all_training):
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            if not cm:
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            # an lc0 ply (vs_leela) trains on lc0's search but is never scored:
            # it stays out of SF analysis, blunder routing and every stat
            if tr.get('sel_method') == 'lc0':
                if not skip_all_training:
                    game_state['lc0_examples'].append(
                        (i, turn, self.training_data_from_lc0(b_fast, tr)))
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            Z_stm = result if turn else -result
            if 'Q_stm' in tr:
                Q = tr['Q_stm']
            else:
                wdl = tr.get('best_wdl')
                this_q = (wdl[0] - wdl[2]) if wdl is not None else 0.0
                Q = this_q

            if is_sf_move:
                # value target waits for finalize_game, which knows the
                # blunder cuts
                sf_won = result != 0 and (result > 0) == bool(sf_color)
                ex = self.training_data_from_sf(b_fast, mv, tr, sf_won)
                tr['policy'] = [(int(j), float(ex[1][j]))
                                for j in np.flatnonzero(ex[1])]
                game_state['sf_examples'].append((i, turn, outcome, ex))

                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            visits = [(c['uci'], max(1, c['visits'])) for c in cm]
            visits = sorted(visits, key=lambda x: x[1], reverse=True)
            if not visits:
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            sel_method = tr.get('sel_method')
            xc0_move = tr.get('xc0_move')
            xerces_uci = mv
            if xc0_move and xc0_move != mv:
                xerces_uci = xc0_move
            elif not sel_method:
                xerces_uci = visits[0][0]

            lms = b_fast.legal_moves()
            idx_map = dict(zip(lms, b_fast.moves_to_indices(lms)))
            x = tr['xc0h']
            # short_fen still needed for the lc0 waypoint FEN check
            short_fen = b_fast.fen(include_counters=False)

            parentQ_white = np.nan
            parent_p_draw = np.nan
            if i > 0:
                ptr = tree_data.get(i - 1, tree_data.get(str(i - 1), {}))
                pmv = game_data['moves_played'][i - 1]
                for c in ptr.get('candidate_moves', []):
                    if c['uci'] == pmv:
                        parentQ_white = c['Q']
                        parent_p_draw = c['Q_draw']
                        break

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
                'parentQ_white': parentQ_white,
                'parent_p_draw': parent_p_draw,
                'best_uci': None,
                'best_cp': None,
                'best_abs': None,
                'pv_ucis': [],
                'played_cp': None,
                'played_abs': None,
                'resolved': False,
            }

            # SF time tier: decided by the search's own WDL (STM), else busy
            # by legal moves or material
            sw = tr.get('best_wdl') or (0.0, 0.0, 0.0)
            if abs(sw[0] - sw[2]) > DECIDED_QD or sw[1] > DECIDED_QD:
                tier = "decided"
            elif board_ch.ply() >= BUSY_MIN_PLY and (
                    board_ch.legal_moves.count() > BUSY_LEGAL
                    or material(board_ch) > BUSY_MATERIAL):
                tier = "busy"
            else:
                tier = None
            rejected = (tr.get('ctrl_sum') or {}).get('rejected') or None
            to_sf_positions.append((i, board_ch.copy(), xerces_uci, tier, rejected))

            game_state['ply_states'].append(ply)
            board_ch.push(move_ch)
            b_fast.push_uci(mv)

        self.pending[gid] = game_state
        # submit to SF thread, or finalize immediately if every move was SF's own
        if to_sf_positions:
            self.n_sf_submitted += len(to_sf_positions)
            game_state['waiting'] = True
            # validation games are the cleanest quality signal there is --
            # high sims, no training use -- so they buy a deeper SF pass than
            # selfplay.
            # the moves let the SF thread follow PVs across plies
            info = {"moves": list(game_data.get('moves_played', []))}
            if is_validation_game and cfg.rescore_movetime_validation_ms:
                info["movetime_ms"] = cfg.rescore_movetime_validation_ms
            self.game_q.put((gid, to_sf_positions, info))
        else:
            self.finalize_game(self.pending.pop(gid))

    def handle_game_results(self, gid, results):
        # TEMPORARY lc0 audit, never enters self.pending
        if gid in self.lc0_audit:
            self.record_lc0_audit(gid, results)
            return

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

            if r.get('skipped'):
                self.sf_skipped += 1
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
            ply['pair_cp']    = r.get('pair_cp')
            ply['resolved']   = True

        game['waiting'] = False
        self.finalize_game(self.pending.pop(gid))

    def queue_lc0_replay(self, game_data, aux, meta_extra=None):
        """Hand one blunder position to the teacher fleet.

        Only stages the spec; the runner drains lc0_pending onto the queue.
        Nothing is held for the reply -- a probe that never returns just
        means this position contributes no sample.
        """
        if self.lc0_cfg is None:
            return None

        # xc0_move is xc0's answer at the position; lc0 matching it ends the
        # confirmation chain after one ply
        meta = {
            'src_game_id': game_data.get('game_id'),
            'orig_cpl': aux.get('orig_cpl'),
            'xc0_move': aux.get('xerces_uci'),
        }
        meta.update(meta_extra or {})
        spec = lc0_replay_spec(
            self.lc0_cfg, game_data, aux['ply_i'], aux['sfen'], meta)
        if spec is not None:
            self.lc0_pending.append(spec)
        return spec

    def take_lc0_specs(self):
        """Read-and-clear the staged teacher probes."""
        out = self.lc0_pending
        self.lc0_pending = []
        return out

    def ingest_lc0_replay(self, meta):
        """Turn a returned teacher probe into correction_buffer samples.

        One per ply actually played -- syzygy or a terminal can end the probe
        short of LC0_BRP_PLIES. Straight visits and WDL, weight 1.0, no
        opening-table bump: these are not xerces-played positions.
        """
        # rebuilt from the path, not start_fen: a bare FEN has no history and
        # the run's encoding may stack it
        board = Board(meta['replay_fen'])
        for uci in meta['replay_moves']:
            board.push_uci(uci)

        tree_data = meta.get('tree_search_data') or {}
        n = 0
        audit = []
        records = []
        for k, mv in enumerate(meta.get('moves_played') or []):
            # collect_tree_search_data runs before plies is bumped, so the
            # search at moves_played[k] is stored under k, not k + 1
            tr = tree_data.get(k, tree_data.get(str(k)))
            if tr:
                rec = self.add_lc0_replay_training_example(board, tr)
                if rec:
                    n += 1
                    records.append(rec)
                # only ply 0 is audited; the continuations go in untested
                cm = tr.get('candidate_moves') or [] if k == 0 else []
                if cm:
                    top = max(cm, key=lambda c: int(c['visits']))['uci']
                    audit.append((k, chess.Board(board.fen()), top))
            board.push_uci(mv)

        self.submit_lc0_audit(
            meta.get('game_id'), audit, meta.get('xc0_cpl'), records,
            meta.get('xc0_tsl'), meta.get('stream') != 'teacher')

        self.sample_counts['lc0_samples'] += n
        self.sample_counts_window['lc0_samples'] += n
        return n

    def submit_lc0_audit(self, game_id, positions, xc0_cpl=None, records=None,
                         xc0_tsl=None, blunder=False):
        """TEMPORARY. Score the teacher's own moves with SF, same budget and
        throttle as normal rescoring. Prefixed gid so it cannot collide with
        a real game in self.pending."""
        if not positions:
            return
        gid = f"lc0aud_{game_id}"
        self.lc0_audit[gid] = {
            'xc0_cpl': xc0_cpl,
            'xc0_tsl': xc0_tsl,
            'ucis': {k: uci for k, _, uci in positions},
            'records': records or [],
            'blunder': blunder,
        }
        self.game_q.put((gid, positions, {}))

    def record_lc0_audit(self, gid, results):
        """TEMPORARY. Only ply 0 is submitted: it is the escalated blunder and
        the only ply xc0 also answered. Scored under normal rescoring rules."""
        info = self.lc0_audit.pop(gid, None) or {}
        ucis = info.get('ucis') or {}
        xc0_cpl = info.get('xc0_cpl')
        xc0_tsl = info.get('xc0_tsl')

        for r in results:
            k = r['ply_idx']
            ms = self.lc0_scorer.score(
                r['best_uci'], r['best_cp'], ucis.get(k, ''), r['played_cp'])
            lc0_tsl = 1000 * expected_score_loss(ms.best_cp, ms.best_cp - ms.loss)

            # each pair steps together, so the printed subtraction is one population
            E = self.lc0_audit_ema
            if xc0_cpl is not None:
                self.lc0_audit_n += 1
                E['lc0'] = ema_step(E.get('lc0'), ms.loss, AUDIT_EMA_SPAN)
                E['xc0'] = ema_step(E.get('xc0'), xc0_cpl, AUDIT_EMA_SPAN)
            if xc0_tsl is not None:
                E['lc0_tsl'] = ema_step(E.get('lc0_tsl'), lc0_tsl, AUDIT_EMA_SPAN)
                E['xc0_tsl'] = ema_step(E.get('xc0_tsl'), xc0_tsl, AUDIT_EMA_SPAN)

            if k == 0:
                ok = lc0_tsl <= ACCEPT_TSL_GOOD
                if info.get('blunder'):
                    self.lc0_confirm_n += 1
                    self.lc0_confirm_ok += ok
                if ok:
                    self.add_to_lc0_pretrain_buffer(info.get('records'))

    def add_to_lc0_pretrain_buffer(self, records):
        """Ply0-confirmed lc0 replays, set aside for the next deep pretrain --
        not mixed into live_buffer or any other buffer, not trained on now."""
        if not records or not LC0_BRP_SHARD_DIR:
            return
        self.lc0_pretrain_buffer.extend(records)
        while len(self.lc0_pretrain_buffer) >= 10240:
            shard, self.lc0_pretrain_buffer = (
                self.lc0_pretrain_buffer[:10240], self.lc0_pretrain_buffer[10240:])
            os.makedirs(LC0_BRP_SHARD_DIR, exist_ok=True)
            write_pkl_gz_shard(shard, new_shard_path(LC0_BRP_SHARD_DIR))

    def add_lc0_replay_training_example(self, board, tr):
        """Teacher visits to policy: 1-visit floor on every legal move, then the
        unified blend_to_uniform(prior_clip_max), same as every other target."""
        cm = tr.get('candidate_moves') or []
        wdl = tr.get('best_wdl')
        if not cm or wdl is None:
            return False

        legal = board.legal_moves()
        if len(legal) <= 1 and random.random() >= ACCEPT_ONE_LEGAL_P:
            return False

        vmap = {c['uci']: max(1, int(c['visits'])) for c in cm}
        for u in legal:
            vmap.setdefault(u, 1)

        ucis = list(vmap)
        counts = np.array([vmap[u] for u in ucis], dtype=np.float32)
        counts /= counts.sum()
        counts = blend_to_uniform(counts, 1e-6, self.config.prior_clip_max)
        policy = np.zeros(1858, dtype=np.float32)
        for idx, p in zip(board.moves_to_indices(ucis), counts):
            policy[idx] += p

        Y = np.array(wdl, dtype=np.float32)
        record = (self.encode_board(board), None, sparsify_policy(policy),
                  Y, 1.0, 1.0, 'lc0')
        self.correction_buffer.append(record, REWEIGHT_INELIGIBLE)
        return record

    def finalize_game(self, game_state):
        if self.start_time is None:
            self.start_time = time.time()
        cfg = game_state['cfg']
        game_data = game_state['game_data']
        gid = game_state['gid']
        result = game_data['result']
        is_draw = (result == 0) or (result == 0.0)

        cpl_s = 0.0
        n_plies = 0
        cpl_by_ply = {}
        rows = []
        eval_trace = []
        turn_at = {}
        # white-POV SF value of the position at each blunder ply
        cut_white = {}
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

            ms = self.scorer.score(
                best_uci_raw, best_cp, xerces_uci, played_cp, z_stm=Z_stm)

            best_uci, best_cp = ms.best_uci, ms.best_cp
            if ms.swapped:
                best_abs = played_abs
            EQUIV = ms.equiv
            missed_mate = ms.missed_mate
            loss_this = ms.loss
            cpl_s += loss_this
            n_plies += 1
            esl = expected_score_loss(best_cp, best_cp - loss_this)
            cpl_by_ply[i] = (int(round(loss_this)), ply.get('best_depth'), rnd(esl, 4))
            is_blunder = not missed_mate and is_esl_candidate(
                best_cp, played_cp, esl, Z_stm, cfg.rescore_esl_blunder)[0]

            if is_blunder:
                v = cp_to_value_tanh(best_cp)
                cut_white[i] = v if turn else -v

            eval_trace.append((i, best_abs))
            if 'ctrl_sum' in tr:
                pair = ply.get('pair_cp')
                pair_esl = None
                if pair is not None:
                    ref = max(best_cp, *pair)
                    pair_esl = tuple(expected_score_loss(ref, cp) for cp in pair)
                self.ctrl_acc.add_ctrl(tr['ctrl_sum'], tr.get('stop_reason', ''),
                                       tr.get('sims', 0), loss_this, esl, xerces_uci,
                                       self.config.rescore_esl_blunder, pair_esl)
            elif tr.get('sel_method') != 'stockfish':
                # xc0 moves searched without the controller (stop_rules old)
                self.ctrl_acc.add_old(tr.get('sims', 0), loss_this, esl,
                                      tr.get('sims_max', 0),
                                      self.config.rescore_esl_blunder)
            rows.append([
                i, mv, xerces_uci, best_uci,
                best_cp, loss_this, played_cp,
                best_abs, played_abs,
                turn, loss_this,
                tr.get('stop_reason', ''), tr.get('sims', 0),
                ply.get('parentQ_white', np.nan), ply.get('parent_p_draw', np.nan),
                esl,
                np.nan, np.nan,  # kl, ce -- filled in below; ce only if this ply trains
            ])

            lms = ply['lms']
            # ensure all legal moves have at least 1 visit
            vmap = {u: max(1, int(v)) for u, v in visits}
            for m in lms:
                if m not in vmap:
                    vmap[m] = 1

            visits = sorted(vmap.items(), key=lambda x: x[1], reverse=True)
            mvs = [v[0] for v in visits]
            vis = [v[1] for v in visits]

            priors_map = {c['uci']: c.get('P_raw', c['P']) for c in cm}
            priors = [priors_map.get(u, 0.0) for u in mvs]

            # KLp is logged for every scored ply, validation included; only
            # training plies move the quantile trackers below
            kl = kl_divergence(vis, priors)
            rows[-1][-2] = kl

            if ply['skip_training']:
                continue

            kl_eligible = loss_this <= EQUIV

            stat_keys = self.accumulate_blunder_stats(
                missed_mate, best_cp, played_cp, loss_this, Z_stm, esl, cfg)

            # a blunder is handed to the teacher instead of trained on
            # directly; everything else trains on its raw visit policy
            to_lc0 = is_blunder

            to_teacher = (not to_lc0 and not missed_mate
                          and esl >= cfg.rescore_esl_inaccuracy
                          and policy_value_miss(cm, best_uci))

            vwht = pwht = 1.0
            self.kl_q50 = update_ema_quantile(self.kl_q50, kl, 0.5, self.config.kl_quantile_lr)
            self.kl_q80 = update_ema_quantile(self.kl_q80, kl, 0.8, self.config.kl_quantile_lr)
            if kl_eligible and kl >= self.kl_q80:
                pwht *= self.config.kl_boost_p80_mult

            # build policy from precomputed board state
            idx_map = ply['idx_map']
            policy = np.zeros(1858, dtype=np.float32)
            for idx, prob in tr['policy']:
                policy[idx] = prob

            pending.append((ply['x'], policy, Q, turn, i, vwht, pwht, ply['okey']))

            pending_aux.append({
                'sfen':            ply['sfen'],
                'xerces_uci':      ply['xerces_uci'],
                'to_lc0':          to_lc0,
                'to_teacher':      to_teacher,
                'loss':            loss_this,
                'orig_cpl':        loss_this,
                'equiv':           EQUIV,
                'best_uci':        best_uci,
                'best_cp':         best_cp,
                'lms':             lms,
                'idx_map':         idx_map,
                'is_blunder':      is_blunder,
                'stat_keys':       stat_keys,
                'esl':             esl,
                'kl':              kl,
                'kl_eligible':     kl_eligible,
                'ply_i':           i,
                'best_wdl':        tr.get('best_wdl'),
            })

        sc = self.sample_counts
        scw = self.sample_counts_window

        row_by_ply = {r[0]: r for r in rows}
        cut_outcome = blunder_cut_outcomes(
            game_state['plies'], cut_white, self.config.value_bootstrap_at_sample)

        # SF plies: cliff-walker-corrected outcome, half SF's own tanh(cp)
        for ply_i, stm, outcome, (x, policy, okey, pwht, kl, n_legal) in (
                game_state.get('sf_examples', ())):
            cut = cut_outcome.get(ply_i)
            if cut is not None:
                outcome = scalar_to_wdl(cut if stm else -cut)
            node = game_state['plies'][ply_i]
            sf_v = node.get('sf_value')
            Y = outcome if sf_v is None else (
                0.5 * outcome + 0.5 * scalar_to_wdl(sf_v)).astype(np.float32)
            node['wdl_target'] = Y
            klv = self.target_kl(Y, node)
            vwht = (self.config.kl_boost_p80_mult
                    if klv >= max(0.03, self.kl_value_q80) else 1.0)
            self.opening_counts.bump(okey)
            self.count_offer('sf', self.admit(
                (x, None, sparsify_policy(policy), Y, vwht, pwht, SF_SOURCE),
                okey, 0.0, klv, kl, ctrl_shaky(node), is_draw, n_legal))

        # lc0 plies: valued as an xc0 ply is, cliff-walker cut included, and
        # offered to admit with TSL taken as 0
        for ply_i, stm, (x, policy, okey, pwht, kl, n_legal) in (
                game_state.get('lc0_examples', ())):
            node = game_state['plies'][ply_i]
            Y = node['wdl_target']
            cut = cut_outcome.get(ply_i)
            if cut is not None:
                Y = blend_wdl_stm(scalar_to_wdl(cut if stm else -cut),
                                  node.get('best_wdl'))
                node['wdl_target'] = Y
            klv = self.target_kl(Y, node)
            vwht = (self.config.kl_boost_p80_mult
                    if klv >= max(0.03, self.kl_value_q80) else 1.0)
            self.opening_counts.bump(okey)
            self.count_offer('vsl', self.admit(
                (x, None, sparsify_policy(policy), Y, vwht, pwht, 'lc0'),
                okey, 0.0, klv, kl, ctrl_shaky(node), is_draw, n_legal))

        # loop 2: compute WDL targets and add to training
        for tup, aux in zip(pending, pending_aux):
            x, policy, Q, is_white, ply_i, vwht, pwht, okey = tup
            z = result if is_white else -result
            node = game_state['plies'][ply_i]
            Y = node['wdl_target']
            # a later blunder decided the result, so the outcome half of the
            # target is SF's value before that blunder instead
            cut = cut_outcome.get(ply_i)
            if cut is not None:
                outcome = scalar_to_wdl(cut if is_white else -cut)
                Y = blend_wdl_stm(outcome, node.get('best_wdl'))
                node['wdl_target'] = Y

            klv = self.target_kl(Y, node)
            self.kl_value_q50 = update_ema_quantile(
                self.kl_value_q50, klv, 0.5, self.config.kl_quantile_lr)
            self.kl_value_q80 = update_ema_quantile(
                self.kl_value_q80, klv, 0.8, self.config.kl_quantile_lr)
            # floored so a zeroed tracker doesn't boost every ply
            if aux['kl_eligible'] and klv >= max(0.03, self.kl_value_q80):
                vwht *= self.config.kl_boost_p80_mult

            wdl_node = aux.get('best_wdl')
            if wdl_node is not None:
                model_wdl = np.array(wdl_node, dtype=np.float32)
                row = row_by_ply.get(ply_i)
                if row is not None:
                    # logged CE only -- scored against the raw game result, not
                    # Y. Y is half model_wdl, so CE(Y, model_wdl) is partly
                    # self-referential and barely moves. Y itself is unchanged.
                    row[-1] = cross_entropy(z_to_wdl(z), model_wdl)


            # the position was played either way, so the opening table is
            # bumped even when the sample itself goes to the teacher
            self.opening_counts.bump(okey)

            if aux['to_lc0']:
                spec = self.queue_lc0_replay(
                    game_data, aux,
                    {'xc0_cpl': aux['loss'], 'xc0_tsl': 1000 * aux['esl']})
                if spec is not None:
                    sc['lc0_queued'] += 1
                    scw['lc0_queued'] += 1
                    self.count_lc0_send(aux['stat_keys'])
                    continue
                print(f"{RS} lc0 queue refused blunder {gid} ply {ply_i}, "
                      f"left to the training path")

            sent = False
            if aux['to_teacher']:
                spec = self.queue_lc0_replay(
                    game_data, aux, {'xc0_cpl': aux['loss'], 'xc0_tsl': 1000 * aux['esl'],
                                     'stream': 'teacher'})
                if spec is not None:
                    sc['lc0_queued'] += 1
                    scw['lc0_queued'] += 1
                    sent = True
            if sent:
                self.count_lc0_send(aux['stat_keys'])
            # the teacher's samples for this position train at full weight
            if sent:
                vwht *= LC0_SENT_XC0_WEIGHT
                pwht *= LC0_SENT_XC0_WEIGHT

            entry = (x, None, sparsify_policy(policy), Y, vwht, pwht, 'xc0')
            sc['offered'] += 1
            scw['offered'] += 1
            if self.admit(entry, okey, 1000 * aux['esl'], klv, aux['kl'],
                          ctrl_shaky(node), is_draw, len(aux['lms'])):
                sc['accepted'] += 1
                scw['accepted'] += 1
                self.track_accepted(aux['loss'], 1000 * aux['esl'], klv, aux['kl'])

        cols = [
            'move_num', 'played_move', 'most_visited_move', 'best_move',
            'best_cp', 'delta', 'played_cp',
            'best_absolute', 'played_absolute', 'stm', 'loss', 'stop_reason', 'sims',
            'parentQ_white', 'parent_p_draw',
            'esl',
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
        for st in ('full', 'tier1', 'tier2'):
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
        self.total_plies += n_plies
        self.record_report_game(game_data, len(game_state['plies']), out_df,
                                game_state['ply_states'])
        self.write_lean_record(game_state, cpl_by_ply)
        self.games_seen.add(gid)
        self.accumulate_stop_stats(stop_stats)
        # cfg here is the per-game snapshot; the controller flag is run-wide
        if self.config.ctrl_enable:
            self.ctrl_games += 1
            if self.ctrl_games % CTRL_WINDOW_GAMES == 0:
                self.print_ctrl_stats()
        if self.games_processed % STATS_WINDOW_GAMES == 0:
            self.print_stop_stats()
            self.print_blunder_stats()
        if len(self.analyzed_results) >= cfg.rescore_analyze_batch:
            self.push_analyzed()
        if self.games_processed % REPORT_EVERY == 0:
            self.print_report()

    def record_report_game(self, game_data, game_plies, out_df, ply_states):
        """One finished game into its scenario row and, for selfplay, the
        quality trends. Quality is the game's mean over its scored xc0
        plies. KLp and KLv are the ones admit samples on: KL(visits with a
        1-visit floor on every legal move || priors) and KL(value target || NN
        WDL); the value target is final here (cliff-walker cut applied)."""
        n = len(out_df)
        tsl = 1000 * out_df['esl'] if n else None
        trs = [p['tr'] for p in ply_states if p.get('tr')]
        klps = out_df['kl'].dropna().tolist() if n else []
        klvs = [kl_divergence(t['wdl_target'], t['root_wdl_nn']) for t in trs
                if t.get('wdl_target') is not None
                and t.get('root_wdl_nn') is not None]
        quality = {
            'cpl': out_df['delta'].mean() if n else None,
            'tsl': tsl.mean() if n else None,
            'klv': float(np.mean(klvs)) if klvs else None,
            'klp': float(np.mean(klps)) if klps else None,
        }
        scenario = game_data.get('scenario', 'unknown')
        name = report_row_name(scenario, game_data.get('stockfish_color'))
        self.report_rows[name].add_game(game_data['result'], game_plies, quality)
        if scenario == 'paired_validation' or not n:
            return
        quality['tsl_mistake'] = float((tsl > TSL_MISTAKE).mean())
        quality['tsl_blunder'] = float((tsl > TSL_BLUNDER).mean())
        for m, v in quality.items():
            if v is not None and np.isfinite(v):
                self.trends[m].add(v)


    def decide_reviewable(self, scenario):
        """Keep the fat record for review. Later this may fold in KL/CPL/RMS(Q)
        signal; for now paired validation plus a flat sample rate."""
        return (scenario == 'paired_validation') or (random.random() < REVIEW_RATE)

    def write_lean_record(self, game_state, cpl_by_ply):
        """Transform the fat {header, plies} into the on-disk retrain-ready
        record: STM-POV lean plies (+ fat extras when reviewable), write
        <game_id>.pkl.gz, append the final game_index row, drop the looper .pkl."""
        header = dict(game_state['header'])
        plies_in = game_state['plies']
        reviewable = self.decide_reviewable(header.get('scenario', ''))

        out_plies = []
        for i, node in enumerate(plies_in):
            entry = {k: node[k] for k in (
                'stm', 'move_played', 'z_wdl', 'wdl_target', 'policy',
                'kl', 'kl_value')}
            for k in ('abs_ply', 'sel_method', 'xc0_move', 'stop_reason',
                      'best_wdl', 'root_wdl_nn', 'xc0h', 'raw_visits',
                      'gap_block_sims', 'gap_block_leader', 'pv_len',
                      'pv_played_len', 'sf_value'):
                entry[k] = node.get(k)
            # search-controller check-ins, only on games run with ctrl_record
            if 'ctrl' in node:
                entry['ctrl'] = node['ctrl']
            # the controller's heads at the stop, and per rejected lock
            # (leader uci, stab, qual, root STM Q, D)
            cs = node.get('ctrl_sum')
            if cs:
                entry['ctrl_stab'] = round(cs['stab'], 4)
                entry['ctrl_qual'] = round(cs['qual'], 4)
                entry['ctrl_ext'] = cs['ext']
                entry['ctrl_rej'] = [(m, *(round(x, 4) for x in v))
                                     for m, *v in cs.get('rejections', [])]

            hit = cpl_by_ply.get(i)
            if hit is not None:
                entry['cpl'], entry['cpl_depth'], entry['esl'] = hit
            elif (node.get('sel_method') == 'stockfish'
                  and node.get('xc0_move') == node.get('move_played')):
                entry['cpl'] = 0
                entry['cpl_depth'] = None
                entry['esl'] = 0.0

            if reviewable:
                for k in ('sims', 'time', 'avg_depth', 'max_depth',
                          'children_visited', 'total_children', 'pv',
                          'pv_played', 'sharp_by_depth'):
                    if k in node:
                        entry[k] = node[k]
                entry['candidate_moves'] = node.get('candidate_moves', [])

            out_plies.append(entry)

        cpl_vals = [p['cpl'] for p in out_plies if 'cpl' in p]
        n = len(cpl_vals)
        header['reviewable'] = reviewable
        header['mean_cpl'] = round(sum(cpl_vals) / n, 4) if n else None
        header['mean_bmr'] = round(sum([1 for c in cpl_vals if c == 0]) / n, 4) if n else None
        esl_vals = [p['esl'] for p in out_plies if 'esl' in p]
        header['mean_esl'] = round(sum(esl_vals) / len(esl_vals), 4) if esl_vals else None

        record = {'header': header, 'plies': out_plies}
        out_path = os.path.join(self.config.game_dir, header['game_id'] + '.pkl.gz')
        with gzip.open(out_path + '.tmp', 'wb', compresslevel=6) as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(out_path + '.tmp', out_path)

        self.append_final_index(header, out_path)

        old_pkl = game_state.get('pkl_path')
        if old_pkl and old_pkl.lower().endswith('.pkl') and os.path.exists(old_pkl):
            os.remove(old_pkl)
        return record

    def append_final_index(self, header, pkl_gz_path):
        """Append the closed-out game_index.json row: the header plus the final
        pkl path, beat_sf, and analysis stats. This row is the done marker."""
        beat_sf = False
        if header.get('vs_stockfish'):
            result = header.get('result', 0)
            sf_white = header.get('stockfish_color')
            beat_sf = bool((result > 0 and not sf_white) or (result < 0 and sf_white))

        row = {**header, 'pkl_file': pkl_gz_path, 'beat_sf': beat_sf}
        with open(self.config.game_index_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, default=str) + '\n')
            f.flush()

    def accumulate_blunder_stats(self, missed_mate, best_cp, played_cp, cpl,
                                 z_stm, esl, cfg):
        """Bump the blunder-type rows this ply falls in; returns their keys."""
        keys = []

        def bump(key, value):
            keys.append(key)
            for acc in (self.total_blunder, self.window_blunder):
                acc[key]['n'] += 1
                acc[key]['cpl'] += value

        bump('total', cpl)
        if missed_mate:
            bump('missed_mate', cpl)
            return keys

        if esl >= cfg.rescore_esl_blunder:
            bump('blunder', cpl)
            if z_stm <= 0:
                bump('blunder_lost', cpl)
        elif esl >= cfg.rescore_esl_inaccuracy:
            bump('inaccuracy', cpl)
        if esl >= ESL_ANY_MIN:
            bump('blunder_any', cpl)

        _, label = is_esl_candidate(
            best_cp, played_cp, esl, z_stm, cfg.rescore_esl_blunder)
        if label == "F":
            bump('F', cpl)
        return keys

    def count_lc0_send(self, keys):
        for acc in (self.total_blunder, self.window_blunder):
            for key in keys:
                acc[key]['lc0'] += 1

    def print_blunder_stats(self):
        acc = self.window_blunder
        cfg = self.config

        def avg(k):
            n = acc[k]['n']
            return acc[k]['cpl'] / n if n else float('nan')

        total_n = acc['total']['n']

        def row(label, k):
            n = acc[k]['n']
            if n == 0:
                return
            pct = n / total_n if total_n else 0.0
            lc0 = acc[k]['lc0']
            print(f"{RS}  {label:<24} n={n:>6}  {pct:>7.2%}  cpl={avg(k):>8.2f}"
                  f"  lc0={lc0:>5} ({lc0 / n:>6.1%})")

        print(f"{RS} blunder types (last {STATS_WINDOW_GAMES}):")
        row("total", "total")
        row(f"ESL {cfg.rescore_esl_inaccuracy:.2f}"
            f"-{cfg.rescore_esl_blunder:.2f}", "inaccuracy")
        row(f"ESL >= {cfg.rescore_esl_blunder:.2f}", "blunder")
        row(f"ESL >= {cfg.rescore_esl_blunder:.2f}, not won", "blunder_lost")
        row(f"ESL >= {ESL_ANY_MIN:.2f}", "blunder_any")
        row("F", "F")

        for k in BLUNDER_STAT_KEYS:
            self.window_blunder[k] = {'n': 0, 'cpl': 0.0, 'lc0': 0}

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

    def print_ctrl_stats(self):
        """Search controller since this process started: how each search
        ended and the quality of the move it played."""
        a = self.ctrl_acc
        n = a.n
        if not n:
            return

        def line(label, g, share):
            k, sims, tsl, cpl, bl = g
            if not k:
                return f"{RS}  {label:<9}{'-':>7}{'-':>7}{'-':>7}{'-':>7}{'-':>8}"
            return (f"{RS}  {label:<9}{share:7.1%}{sims / k:7.0f}{tsl / k:7.2f}"
                    f"{cpl / k:7.2f}{bl / k:8.2%}")

        old_n = f" / {a.old[0]} old xc0" if a.old[0] else ""
        print(f"{RS} controller, all {self.ctrl_games} games since start "
              f"({n} ctrl{old_n} moves)")
        print(f"{RS}  {'stop':<9}{'share':>7}{'sims':>7}{'TSL':>7}{'CPL':>7}{'blund':>8}")
        for g in CtrlStats.GROUPS[:-1]:
            print(line(g, a.groups[g], a.groups[g][0] / n))
        print(line("all", a.all, 1.0))
        if a.old[0]:
            # old rules: share is of all xc0 moves
            print(line("old", a.old, a.old[0] / (n + a.old[0])))

        saved = 1 - a.sims / a.sched if a.sched else float("nan")
        old_txt = (f"   old {a.old_sims / 1e6:.2f}M/{a.old_sched / 1e6:.2f}M "
                   f"{1 - a.old_sims / a.old_sched:.1%}" if a.old_sched else "")
        n_ext = a.n_ext
        extra = a.ext_extra / n_ext if n_ext else 0.0
        changed = a.ext_changed / n_ext if n_ext else 0.0
        print(f"{RS}  sims     {a.sims / 1e6:.2f}M/{a.sched / 1e6:.2f}M "
              f"saved {saved:.1%}{old_txt}")
        print(f"{RS}  extend   {n_ext} ({n_ext / n:.1%})   2+ {a.n_multi} "
              f"({a.n_multi / n:.1%})   {extra:+.0f} sims   changed {changed:.0%}")
        sk, st = a.ext_same
        ck, ct, cr = a.ext_chg
        same = st / sk if sk else 0.0
        chg = ct / ck if ck else 0.0
        rej = cr / ck if ck else 0.0
        print(f"{RS}  ext TSL  same {same:.2f} | changed {chg:.2f} | "
              f"rejected {rej:.2f} | gain {rej - chg:+.2f} ({ck})")
        print(f"{RS}  ctrl     {a.ms / n:.1f} ms/mv   {a.checkins / n:.1f} ci/mv"
              f"   stab {a.stab / n:.3f}   qual {a.qual / n:.3f}"
              f"   ties {a.ties / max(1, a.checkins):.2%}")

    def print_stop_stats(self):
        stops = ("tier1", "tier2", "full")
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

        wh, ws, wc, wb = rows(f"last {STATS_WINDOW_GAMES}", self.window_stop)
        th, _,  tc, tb = rows("overall",  self.total_stop)
        if self.config.ctrl_enable:
            print()
            return
        print(wh)
        print(ws)
        print(wc)
        print(wb)
        print()
        print(th)
        print(tc)
        print(tb)
        print()

        for st in stops:
            self.window_stop[st] = {'n': 0, 'cpl': 0.0, 'bmr': 0.0, 'sims': 0.0}

    def push_analyzed(self):
        """Write the analyzed games out as one staging chunk."""
        if self.analyzed_results:
            save_analysis_chunk_simple(self.config.run_dir, self.analyzed_results)
        self.analyzed_results = []

    def print_report(self):
        """Every REPORT_EVERY games: selfplay quality trends, throughput, SF
        and training targets. The SF and recent-target windows reset here.
        The per-scenario table is the epoch log's (it reads report_rows)."""
        W = 78

        def out(s=""):
            print(f"{RS} {s}")

        def cols(label, cells, w=10):
            return f"{label:<16}" + "".join(f"{c:>{w}}" for c in cells)

        print()
        out("-" * W)
        out(f"{'trend (game EMA)':<16}{'CPL':>7}{'TSL':>7}{'KLv':>7}{'KLp':>6}"
            f"{f'TSL>{TSL_MISTAKE}':>9}{f'TSL>{TSL_BLUNDER}':>9}")
        t = self.trends
        for i, span in enumerate(TREND_SPANS):
            if t['cpl'].value(i) is None:
                continue
            mistake = t['tsl_mistake'].value(i)
            blunder = t['tsl_blunder'].value(i)
            out(f"{f'ema {span}':<16}" + cell(t['cpl'].value(i), 7, ".2f")
                + cell(t['tsl'].value(i), 7, ".2f") + cell(t['klv'].value(i), 7, ".3f")
                + cell(t['klp'].value(i), 6, ".2f")
                + cell(None if mistake is None else 100 * mistake, 8, ".1f") + "%"
                + cell(None if blunder is None else 100 * blunder, 8, ".1f") + "%")
        for label, klv, klp in (("p50", self.kl_value_q50, self.kl_q50),
                                ("p80", self.kl_value_q80, self.kl_q80)):
            out(f"{label:<16}" + cell(None, 7, "") + cell(None, 7, "")
                + cell(klv, 7, ".3f") + cell(klp, 6, ".2f")
                + cell(None, 9, "") + cell(None, 9, ""))
        a = self.acc_ema
        if a:
            out(f"{'acc':<16}" + cell(a['cpl'], 7, ".2f") + cell(a['tsl'], 7, ".2f")
                + cell(a['klv'], 7, ".3f") + cell(a['klp'], 6, ".2f")
                + cell(100 * a['tsl_mistake'], 8, ".1f") + "%"
                + cell(100 * a['tsl_blunder'], 8, ".1f") + "%")
        print()
        out("-" * W)

        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            avg_depth = (self.sf_depth_sum / self.sf_depth_count
                         if self.sf_depth_count else 0.0)
            out(cols("throughput", ["games", "games/hr", "moves/s", "depth"]))
            out(cols("", [self.games_processed,
                          f"{self.games_processed / elapsed * 3600:.1f}",
                          f"{self.total_plies / elapsed:.2f}",
                          f"{avg_depth:.1f}"]))

        ms_pos = (1000 * self.sf_compute_time / self.sf_call_count
                  if self.sf_call_count else 0.0)
        ms_ply = (1000 * self.sf_compute_time / self.sf_compute_count
                  if self.sf_compute_count else 0.0)
        # share of plies where xerces already played SF's best move, so the
        # second pass was skipped
        first_run = (100 * (self.sf_compute_count - self.sf_rerun_count)
                     / self.sf_compute_count if self.sf_compute_count else 0.0)
        out(cols("SF", ["positions", "ms/pos", "ms/ply", "1st run", "backlog"]))
        out(cols("", [self.n_sf_submitted, f"{ms_pos:.0f}ms", f"{ms_ply:.0f}ms",
                      f"{first_run:.1f}%", len(self.intake) + len(self.pending)]))

        def kept(sc, kind):
            off = sc[f'{kind}offered']
            return f"{sc[f'{kind}accepted'] / off * 100:.1f}%" if off else "-"

        def admitted_cells(sc):
            return [sc['accepted'], kept(sc, ''), sc['sf_accepted'], kept(sc, 'sf_'),
                    sc['vsl_accepted'], kept(sc, 'vsl_')]

        def correction_cells(sc):
            # sent and back do not balance within a window: a chain is
            # answered a few seconds after it is sent, often after this print
            total = (sc['accepted'] + sc['sf_accepted'] + sc['vsl_accepted']
                     + sc['lc0_samples'])
            pct = f"{sc['lc0_samples'] / total * 100:.1f}%" if total else "-"
            return [sc['lc0_queued'], sc['lc0_samples'], pct]

        out(cols("admitted", ["xc0", "xc0-%", "sf", "sf-%", "lc0", "lc0-%"]))
        out(cols(f"last {REPORT_EVERY}", admitted_cells(self.sample_counts_window)))
        out(cols("total", admitted_cells(self.sample_counts)))
        out(cols("corrections", ["sent", "back", "share"]))
        out(cols(f"last {REPORT_EVERY}", correction_cells(self.sample_counts_window)))
        out(cols("total", correction_cells(self.sample_counts)))
        out("-" * W)
        out(f"training samples this round: {self.written_this_round} | "
            f"total: {self.written_total}")
        print()

        self.sf_compute_time = 0.0
        self.sf_compute_count = 0
        self.sf_call_count = 0
        self.sf_rerun_count = 0
        self.sf_skipped = 0
        self.sf_depth_sum = 0
        self.sf_depth_count = 0
        self.sample_counts_window = zero_sample_counts()

# helpers
def zero_sample_counts():
    """xc0 offered / accepted (no prefix), the same for SF's and lc0's plies
    in their match games, and the correction chains sent and samples back."""
    return {'offered': 0, 'accepted': 0, 'sf_offered': 0, 'sf_accepted': 0,
            'vsl_offered': 0, 'vsl_accepted': 0,
            'lc0_queued': 0, 'lc0_samples': 0}


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


def blend_wdl_stm(outcome_wdl_stm, best_wdl_stm):
    """Blend an outcome WDL and the search WDL, both in STM perspective."""
    if best_wdl_stm is None:
        return outcome_wdl_stm
    return (0.5 * outcome_wdl_stm + 0.5 * np.asarray(best_wdl_stm, dtype=np.float32))


def sampled_off_pick(p):
    """A sampled move that differs from xc0's own pick. Older logs leave
    xc0_move None when the two match."""
    xc0 = p.get('xc0_move')
    return (p.get('sel_method') == 'sampled' and bool(xc0)
            and xc0 != p.get('move_played'))


def deviation_outcomes(plies, white_first, result, bootstrap):
    """White-POV outcome WDL per ply. With bootstrap, a sampled move that
    differed from Xerces's choice ends a segment: that ply and every ply
    before it, back to the previous such move, take the search WDL from just
    before the sampled move instead of the game result."""
    carry = z_to_wdl(result)
    out = [None] * len(plies)
    for i in range(len(plies) - 1, -1, -1):
        p = plies[i] or {}
        bw = p.get('best_wdl')
        if (bootstrap and bw is not None and sampled_off_pick(p)):
            bw = np.asarray(bw, dtype=np.float32)
            white = white_first if i % 2 == 0 else not white_first
            carry = bw if white else bw[::-1].copy()
        out[i] = carry
    return out


def blunder_cut_outcomes(plies, cut_white, bootstrap):
    """White-POV value per ply from the nearest SF-flagged blunder at or after
    it. A sampled deviation that deviation_outcomes already cuts at wins when
    it is nearer; plies with neither ahead of them are left out."""
    out = {}
    carry = None
    for i in range(len(plies) - 1, -1, -1):
        p = plies[i] or {}
        if i in cut_white:
            carry = cut_white[i]
        elif (bootstrap and p.get('best_wdl') is not None
                and sampled_off_pick(p)):
            carry = None
        if carry is not None:
            out[i] = carry
    return out


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


REVIEW_RATE = 0.005
# SF-ply policy: SF's share of the blend when SF's move is not the search's
# top move, by game outcome from SF's side
SF_POLICY_W_WON = 0.8
SF_POLICY_W_OTHER = 0.5
# SF's move as a policy target: this share on SF's move, the rest flat
SF_MOVE_SHARE = 0.8


def sf_move_target(ucis, mv):
    """SF_MOVE_SHARE on mv, the rest spread evenly over the other moves in
    ucis; 1.0 when mv is the only move. Aligned to ucis."""
    n = len(ucis)
    if n == 1:
        return np.ones(1)
    p = np.full(n, (1.0 - SF_MOVE_SHARE) / (n - 1))
    p[ucis.index(mv)] = SF_MOVE_SHARE
    return p


def policy_from_raw_visits(raw_visits, prior_clip_max):
    indices, visits = raw_visits
    counts = np.maximum(np.asarray(visits, dtype=np.float32), 1)
    counts /= counts.sum()
    probs = blend_to_uniform(counts, 1e-6, prior_clip_max)
    return list(zip(np.asarray(indices).tolist(), probs.tolist()))


def build_policy_sparse(board, candidate_moves, prior_clip_max):
    """Blended sparse training policy: floor every legal move to 1 visit,
    normalize, blend toward uniform (clip-only via 1e-6 eps), keep legal entries.
    Returns [(index, probability)] with float32 precision."""
    vmap = {c['uci']: c['visits'] for c in candidate_moves}
    for u in board.legal_moves():
        vmap[u] = max(vmap.get(u, 0), 1)
    ucis = list(vmap.keys())
    counts = np.array([vmap[u] for u in ucis], dtype=np.float32)
    counts /= counts.sum()
    # 1e-6 eps: clip-only, no uniform floor unless prior_clip_max forces it
    counts = blend_to_uniform(counts, 1e-6, prior_clip_max)
    policy = np.zeros(1858, dtype=np.float32)
    for ci, prob in zip(board.moves_to_indices(ucis), counts):
        policy[ci] += prob
    nz = np.nonzero(policy)[0]
    return list(zip(nz.tolist(), policy[nz].tolist()))


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
        prev_df_all = combined.get("df_all", None)
    else:
        prev_df_all = None

    # accumulate chunk content
    chunk_dfs = []

    for fn in fns:
        path = os.path.join(staging, fn)
        # load each chunk (let exceptions propagate)
        with open(path, "rb") as f:
            chunk = pickle.load(f)

        cdf = chunk.get("df_all", None)
        if cdf is not None:
            chunk_dfs.append(cdf)

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

    # columns dropped from the written file are recoverable: ts/scenario/
    # stockfish_color from game_index on game_id; best_cp/played_cp as
    # clip(<absolute> * stm_sign); loss is an exact duplicate of delta
    drop_cols = ['ts', 'scenario', 'stockfish_color',
                 'best_cp', 'played_cp', 'loss']
    df_all = (df_all.drop_duplicates(['game_id', 'move_num'])
              .sort_values(['game_id', 'move_num']))
    df_all = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns])

    combined_new = {"df_all": df_all}

    # persist atomically using existing helper
    save_pickle_atomic(combined_new, out_pkl)
    print(f"[combine] wrote combined ANALYZE_PKL -> {Path(out_pkl).name} "
          f"({df_all['game_id'].nunique()} games)")

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

    all_dfs = [analysis_out.get("df") for analysis_out in batch]
    all_dfs = [df for df in all_dfs if df is not None]

    # df_all is concat of per-game dfs (or None)
    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else None

    chunk_obj = {"df_all": df_all}

    fname = f"{int(time.time())}_{uuid.uuid4().hex}.pkl"
    outp = os.path.join(staging, fname)

    with open(outp, "wb") as f:
        pickle.dump(chunk_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return outp




