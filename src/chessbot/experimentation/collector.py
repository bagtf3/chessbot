"""Search-controller training-data collector.

One corpus position is one trajectory: one tree, searched without noise
under the standard search params, with every check-in recorded from the
first one (see search_controller_model.md for the feature vectors).

Each budget stage gets an xc0 reference move:
- locked at the first check-in past min_decision where the leader's visit
  lead over the runner-up exceeds the visits left to the stage ceiling
  (no other move can catch it), else
- the most-visited move at the stage ceiling.

The search pauses at that check-in until the SF teacher has judged the
reference. Approved: one continuation draw (stop here, run to the ceiling,
or stop at a random intermediate check-in), then the trajectory ends.
Disapproved: the stage is frozen, the search runs to its ceiling, the
ceiling doubles (clamped to the cap: global_cap, or its stratum's cap in strata)
and the same tree carries on.

After the search ends the tree is released; the trajectory waits off the
GPU for the teacher evals of every check-in's most-visited move, then
becomes one record with two per-check-in targets:
- stability: the move is in its stage's xc0 equivalence set;
- quality: the move is SF-equivalent to the teacher's best.
"""
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import pickle
import queue
import random
import subprocess
import sys
import threading
import time
import zlib
from collections import Counter

import chess
import chess.engine
import numpy as np
import pandas as pd
import yaml
from pyfastchess import (
    BOARD_FEATURE_NAMES, CHECK_IN_FEATURE_NAMES, NN_FEATURE_NAMES,
    MCTSForest, raw_cache_bulk_insert_np)

from chessbot import SF_LOC
from chessbot.config import XC0H_K, Config
from chessbot.effort_scale import bin_key
from chessbot.experimentation.corpus import CORPUS_DIR
from chessbot.experimentation.runner import (
    SfEval, Unpickler, is_equiv, position_boards, sf_eval_from_score,
    start_server)
from chessbot.mcts_utils import MCTSTree, micro_batch_for
from chessbot.utils import compact_count, compact_span

SCHEMA_VERSION = 1
FEATURE_VERSION = "context48-v1/checkin64-v2"
CONTINUATION_POLICY = "v1"
XC0_EQUIV_RULE = "v1"
EXPERT_RULE = "v1"
COLLECTIONS_DIR = r"D:\Data\chessbot_data\experiments\collections"
# deepest SF eval per (full FEN, move) across every collection
SF_CACHE = os.path.join(COLLECTIONS_DIR, "sf_cache.pkl")
POS_FIELDS = ["init_fen", "uci_path", "fen", "short_fen", "best_move",
              "best_cp", "played_move", "played_cp", "cpl_depth", "game_id",
              "abs_ply", "parentQ_white", "parent_p_draw", "run_tag", "stratum"]
WORKER_TELEMETRY_S = 5.0
STATUS_EVERY = 100

DEFAULTS = {
    "check_every": 100,
    "initial_budget": None,        # None: the ply's scheduled sims ceiling
    # this share of trajectories, drawn per trajectory, starts at short_budget
    # instead; stages past it double as usual
    "short_budget": None,
    "short_budget_frac": 0.0,
    # corpus stratum -> how its positions search: budget (starting ceiling,
    # over the two above) and cap (over global_cap); any other key is a search
    # param set on its trees, over the run config's
    "strata": {},
    "min_decision_visits": None,   # None: the run config's sims_floor
    "global_cap": 16000,
    # SF depth per worker, steered by its eval backlog with hysteresis:
    # over high -> min; over mid -> at most min + 1; under mid -> at least
    # min + 1; under low -> max (min + 2). New trajectories take the depth
    # current when they start.
    "teacher_depth": 17,
    "teacher_depth_min": 16,
    "teacher_depth_max": 18,
    "teacher_backlog_low": 10,
    "teacher_backlog_mid": 50,
    "teacher_backlog_high": 100,
    "sf_engines": 2,               # SF engines per worker
    "sf_hash": 512,                # MB per engine, over the run config's
    # corpus rows' own SF evals at this depth or deeper seed the teacher
    "corpus_eval_min_depth": 17,
    "continuation": {"full": 0.25, "stop": 0.5, "random": 0.25},
    "xc0_equiv_visits": 0.05,      # share of the winner's visits
    "xc0_equiv_q": 0.02,
    "xc0_equiv_window": 3,         # check-ins whose leaders are candidates
    "shard_size": 5000,
    "max_live": 256,               # searching trajectories per worker
    "max_trees": None,             # searching + judging; None: 2 * max_live
    "chunk_size": 64,
    "n_workers": 3,
    "server_threads": 2,
    "seed": 1,
    # search params set over the run config's
    "vscale": 0.9,
    "vscale_gate_q": 0.9,
    "fpu_reduction": 0.2,
    "qema_span": 160,
    "qdelta_span": 160,
}
SEARCH_OVERRIDES = (
    "vscale", "vscale_gate_q", "fpu_reduction", "qema_span", "qdelta_span")


def dominance(visits, root_n, ceiling, min_decision):
    """Exact visit-dominance test over every legal root child.

    visits: N per child in deterministic order, most visited first, ties in
    the tree's child order. Locked when root_n > min_decision and the leader
    leads the runner-up by more than the visits left to the ceiling: each
    later descent adds one visit to one child, so nothing can catch it."""
    leader_n = visits[0]
    runner_up_n = visits[1] if len(visits) > 1 else 0
    gap = leader_n - runner_up_n
    remaining = ceiling - root_n
    eligible = root_n > min_decision
    return {"eligible": eligible, "locked": eligible and gap > remaining,
            "root_n": root_n, "ceiling": ceiling, "leader_n": leader_n,
            "runner_up_n": runner_up_n, "gap": gap, "remaining": remaining}


def xc0_equiv_set(winner, recent_leaders, stats, visits_frac, q_tol):
    """The winner plus any recent leader within visits_frac of the winner's
    visits and q_tol STM-POV Q of it. stats: uci -> (N, n_done, q_stm)."""
    out = [winner]
    wn, wd, wq = stats[winner]
    for m in dict.fromkeys(recent_leaders):
        if m == winner or m not in stats:
            continue
        n, d, q = stats[m]
        if (d > 0 and wd > 0 and abs(n - wn) <= visits_frac * wn
                and abs(q - wq) <= q_tol):
            out.append(m)
    return out


def next_teacher_depth(depth, backlog, coll):
    """The worker's teacher depth after seeing backlog (see DEFAULTS)."""
    lo, hi = coll["teacher_depth_min"], coll["teacher_depth_max"]
    mid = lo + 1
    if backlog > coll["teacher_backlog_high"]:
        return lo
    if backlog > coll["teacher_backlog_mid"]:
        return min(depth, mid)
    if backlog < coll["teacher_backlog_low"]:
        return hi
    if backlog < coll["teacher_backlog_mid"]:
        return max(depth, mid)
    return depth


def trajectory_rng(seed, trajectory_id):
    """Per-trajectory RNG from the collection seed and the trajectory ID."""
    digest = hashlib.blake2b(f"{seed}:{trajectory_id}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def sample_continuation(rng, intervals_left, probs):
    """(drawn, effective, intervals): drawn is the branch the draw picked,
    effective the one carried out, intervals how many check-in intervals
    past the approval check-in the search runs.

    random picks uniformly among strictly intermediate endpoints, 1 to
    intervals_left - 1. With none (intervals_left 1) it falls to stop or
    full with equal odds; at the ceiling (intervals_left 0) every branch
    stops there. Both numbers are always drawn, so the draw sequence never
    depends on the branch."""
    u, v = rng.random(), rng.random()
    drawn = ("full" if u < probs["full"]
             else "stop" if u < probs["full"] + probs["stop"] else "random")
    if intervals_left == 0:
        return drawn, "stop", 0
    effective = drawn
    if drawn == "random":
        if intervals_left >= 2:
            return drawn, "random", 1 + int(v * (intervals_left - 1))
        effective = "stop" if v < 0.5 else "full"
    return drawn, effective, (0 if effective == "stop" else intervals_left)


def expert_set(baseline, evals):
    """Moves SF-equivalent to the teacher's best. baseline: (uci, SfEval)
    from an unrestricted search; evals: uci -> SfEval, each searched
    restricted to that move. The baseline move itself always counts."""
    best_uci, best_ev = baseline
    return {m for m, ev in evals.items() if m == best_uci or is_equiv(best_ev, ev)}


def build_targets(moves, stage_ids, stage_sets, expert, judged):
    """targets[T, 2] and target_valid[T, 2]: column 0 stability (the move
    is in its stage's xc0 set), column 1 quality (the move is in the expert
    set). Quality is valid only for moves the teacher judged."""
    t = len(moves)
    targets = np.zeros((t, 2), np.float32)
    valid = np.zeros((t, 2), bool)
    for i, (m, s) in enumerate(zip(moves, stage_ids)):
        if stage_sets[s] is not None:
            targets[i, 0] = m in stage_sets[s]
            valid[i, 0] = True
        if m in judged:
            targets[i, 1] = m in expert
            valid[i, 1] = True
    return targets, valid


def eval_depth(val):
    """Depth of a cached eval: an SfEval, or a baseline (uci, SfEval)."""
    return (val[1] if isinstance(val, tuple) and not isinstance(val, SfEval) else val).depth


def keep_deepest(cache, key, val):
    """Store val under key unless the cache already holds one as deep.
    Returns True when it was stored."""
    have = cache.get(key)
    if have is not None and eval_depth(have) >= eval_depth(val):
        return False
    cache[key] = val
    return True


def read_eval_log(path, cache):
    """Fold an eval log ((fen, uci or None), value) stream into cache,
    deepest kept. A partial record left by a crash ends the read."""
    with open(path, "rb") as f:
        while True:
            try:
                key, val = Unpickler(f).load()
            except (EOFError, pickle.UnpicklingError):
                return
            keep_deepest(cache, key, val)


def load_sf_cache():
    if not os.path.exists(SF_CACHE):
        return {}
    with open(SF_CACHE, "rb") as f:
        return Unpickler(f).load()


def sync_sf_cache():
    """Fold every collection's per-worker eval logs into the shared SF
    cache, deepest eval per (full FEN, move) kept, and save it. Returns its
    size."""
    cache = load_sf_cache()
    for run in os.listdir(COLLECTIONS_DIR):
        run_dir = os.path.join(COLLECTIONS_DIR, run)
        if os.path.isdir(run_dir):
            for f in os.listdir(run_dir):
                if f.startswith("sf_evals_") and f.endswith(".pkl"):
                    read_eval_log(os.path.join(run_dir, f), cache)
    tmp = SF_CACHE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, SF_CACHE)
    return len(cache)


class Teacher:
    """SF teacher for one worker: n engines on threads. Each position is
    pinned to one engine, so all of its searches share that engine's TT.

    Urgent requests (baselines, stage approvals) are singletons on their
    engine's queue. Other requests (check-in leaders) are stowed in a bundle
    per position; an engine with no urgent work takes one of its bundles and
    scores all its moves in one MultiPV search restricted to them, preferring
    the position it just searched (warm TT), then flushed bundles (search
    over, labels awaited), then the oldest.

    evals holds the deepest eval per (full FEN, move), seeded from the
    shared cache; a lookup takes any eval at the asked depth or deeper.
    baseline(key, depth): (best uci, SfEval) from an unrestricted search.
    move(key, uci, depth): SfEval of the search restricted to that move.
    Every value is side-to-move POV, as SfEval documents. New evals are
    appended to log_path as they land."""

    def __init__(self, sf_config, n_engines, seed=None, log_path=None, hash_mb=None):
        self.queues = [queue.Queue() for _ in range(n_engines)]
        self.res_q = queue.Queue()
        self.stop_ev = threading.Event()
        self.evals = dict(seed or {})   # (key, uci or None) -> SfEval or (uci, SfEval)
        self.pending = {}               # (key, uci, depth) -> urgent
        self.bundles = {}               # (key, depth) -> stowed non-urgent moves
        self.lock = threading.Lock()
        self.n_evals = 0
        self.err = None
        self.log = open(log_path, "ab") if log_path else None
        self.seq = itertools.count()
        config = dict(sf_config, **({"Hash": hash_mb} if hash_mb else {}))
        self.threads = [threading.Thread(target=self.run, args=(config, i), daemon=True)
                        for i in range(n_engines)]
        for t in self.threads:
            t.start()

    def engine_of(self, key):
        return zlib.crc32(key.encode()) % len(self.queues)

    def backlog(self):
        return len(self.pending)

    def lookup(self, key, uci, depth):
        val = self.evals.get((key, uci))
        return val if val is not None and eval_depth(val) >= depth else None

    def seed_corpus(self, key, pos, min_depth):
        """The corpus row's own SF evals, when cpl_depth reaches min_depth:
        its best move as the baseline and its played move. Kept only where
        nothing as deep is held, and never logged to the shared cache."""
        d = pos.get("cpl_depth")
        if d is None or not np.isfinite(d) or d < min_depth:
            return
        d = int(d)
        if isinstance(pos.get("best_move"), str) and np.isfinite(pos.get("best_cp", np.nan)):
            keep_deepest(self.evals, (key, None),
                         (pos["best_move"], SfEval(cp_stm=int(pos["best_cp"]), mate_stm=None,
                                                   depth=d, source="corpus")))
        if (isinstance(pos.get("played_move"), str)
                and np.isfinite(pos.get("played_cp", np.nan))):
            keep_deepest(self.evals, (key, pos["played_move"]),
                         SfEval(cp_stm=int(pos["played_cp"]), mate_stm=None, depth=d,
                                source="corpus"))

    def request(self, key, board, uci, depth, urgent=False):
        k = (key, uci, depth)
        if (self.lookup(key, uci, depth) is not None or self.pending.get(k, False)
                or (k in self.pending and not urgent)):
            return
        self.pending[k] = urgent
        if urgent:
            self.queues[self.engine_of(key)].put((k, board.copy(stack=True)))
            return
        with self.lock:
            b = self.bundles.get((key, depth))
            if b is None:
                b = self.bundles[(key, depth)] = {
                    "board": board.copy(stack=True), "moves": set(), "flushed": False,
                    "seq": next(self.seq)}
            b["moves"].add(uci)

    def flush(self, key, depth):
        """The search is over: its labels are now waited on."""
        with self.lock:
            b = self.bundles.get((key, depth))
            if b is not None:
                b["flushed"] = True

    def take_bundle(self, engine, last_key):
        with self.lock:
            mine = [bk for bk in self.bundles if self.engine_of(bk[0]) == engine]
            if not mine:
                return None
            bk = min(mine, key=lambda bk: (bk[0] != last_key,
                                           not self.bundles[bk]["flushed"],
                                           self.bundles[bk]["seq"]))
            return bk, self.bundles.pop(bk)

    def baseline(self, key, depth):
        return self.lookup(key, None, depth)

    def move(self, key, uci, depth):
        return self.lookup(key, uci, depth)

    def drain(self):
        while True:
            try:
                k, val = self.res_q.get_nowait()
            except queue.Empty:
                return
            if k == "__error__":
                raise RuntimeError("[collect] SF teacher failed") from self.err
            key, uci, _ = k
            if keep_deepest(self.evals, (key, uci), val):
                self.n_evals += 1
                if self.log is not None:
                    pickle.dump(((key, uci), val), self.log,
                                protocol=pickle.HIGHEST_PROTOCOL)
                    self.log.flush()
            self.pending.pop(k, None)

    def close(self):
        self.stop_ev.set()
        for t in self.threads:
            t.join()
        if self.log is not None:
            self.log.close()

    def run(self, sf_config, engine):
        eng = chess.engine.SimpleEngine.popen_uci(SF_LOC, timeout=60)
        eng.configure(sf_config)
        last_key = None
        try:
            while not self.stop_ev.is_set():
                try:
                    k, board = self.queues[engine].get(timeout=0.01)
                except queue.Empty:
                    job = self.take_bundle(engine, last_key)
                    if job is None:
                        continue
                    (last_key, depth), b = job
                    self.run_bundle(eng, last_key, depth, b)
                    continue
                last_key = k[0]
                self.res_q.put((k, self.search(eng, board, k[1], k[2])))
        except Exception as e:
            self.err = e
            self.res_q.put(("__error__", None))
        finally:
            eng.quit()

    @staticmethod
    def search(eng, board, uci, depth):
        """Values are stored at the asked depth: SF reports less when it stops
        early on a forced line, and that must still satisfy the ask."""
        if uci is None:
            info = eng.analyse(board, chess.engine.Limit(depth=depth))
            return info["pv"][0].uci(), sf_eval_from_score(info["score"], depth)
        info = eng.analyse(board, chess.engine.Limit(depth=depth),
                           root_moves=[chess.Move.from_uci(uci)])
        return sf_eval_from_score(info["score"], depth)

    def run_bundle(self, eng, key, depth, b):
        """A baseline in the bundle runs first; its moves then share one
        MultiPV search restricted to them, which at a fixed depth scores
        every line exactly. A move MultiPV does not report runs alone."""
        board, moves = b["board"], b["moves"]
        if None in moves:
            self.res_q.put(((key, None, depth), self.search(eng, board, None, depth)))
        ucis = sorted(m for m in moves
                      if m is not None and self.lookup(key, m, depth) is None)
        if len(ucis) > 1:
            infos = eng.analyse(board, chess.engine.Limit(depth=depth),
                                multipv=len(ucis),
                                root_moves=[chess.Move.from_uci(m) for m in ucis])
            for info in infos:
                uci = info["pv"][0].uci() if info.get("pv") else None
                if uci in ucis and "score" in info:
                    self.res_q.put(((key, uci, depth),
                                    sf_eval_from_score(info["score"], depth)))
                    ucis.remove(uci)
        for uci in ucis:
            self.res_q.put(((key, uci, depth), self.search(eng, board, uci, depth)))


class Trajectory:
    """One position's search, from its first check-in to its end, and the
    teacher evals that label it. state: searching -> (judging <-> searching)
    -> labelling -> finished."""

    def __init__(self, spec, cfg, coll, teacher, forest, depth=None):
        self.spec = spec
        self.coll = coll
        self.cfg = cfg
        self.teacher = teacher
        self.forest = forest
        self.pos = pos = spec["pos"]
        self.tid = spec["trajectory_id"]
        # teacher evals key on the full FEN: counters change SF's answer
        self.key = pos["fen"]
        self.interval = coll["check_every"]
        self.t0 = time.time()
        self.fb, self.cb = position_boards(pos)

        self.tree = MCTSTree(self.fb.clone(), cfg)
        mp_ = self.tree.move_params()
        ceiling = coll["initial_budget"] or mp_["sims_max"]
        # its own draw, so the continuation draws stay as they were
        if (coll["short_budget"] and trajectory_rng(f"{coll['seed']}:budget", self.tid)
                .random() < coll["short_budget_frac"]):
            ceiling = coll["short_budget"]
        stratum = dict(coll["strata"].get(pos.get("stratum"), {}))
        ceiling = stratum.pop("budget", ceiling)
        ceiling = -(-int(ceiling) // self.interval) * self.interval
        self.cap = stratum.pop("cap", coll["global_cap"])
        self.ceiling = min(ceiling, self.cap)
        self.min_decision = coll["min_decision_visits"] or mp_["sims_min"]
        self.c_puct = mp_["c_puct"]
        self.search_params = stratum
        self.tree.set_params(sims_min=int(self.min_decision), sims_max=self.ceiling,
                             **self.search_params)
        self.tree.reset_early_stop()
        forest.add_tree(self.tree)

        # the asked depth; any eval at it or deeper satisfies it, so a deep
        # cached or corpus baseline is used as is
        self.depth = depth or coll["teacher_depth"]
        teacher.seed_corpus(self.key, pos, coll["corpus_eval_min_depth"])
        self.rng = trajectory_rng(coll["seed"], self.tid)
        self.moves = sorted(self.fb.legal_moves())
        self.col = {m: i for i, m in enumerate(self.moves)}

        self.state = "searching"
        self.boundary = self.interval
        self.context = None
        self.rows = []
        self.stages = [self.new_stage(0)]
        self.continuation = None
        self.end_reason = None
        teacher.request(self.key, self.cb, None, self.depth, urgent=True)

    def new_stage(self, start):
        return {"stage_id": len(getattr(self, "stages", [])), "start_sims": start + self.interval,
                "ceiling": self.ceiling, "reference": None, "approved": None,
                "expert": None}

    def root_n(self):
        return int(self.tree.root().N)

    def step(self, max_new):
        """One bounded collect: never past the next check-in boundary, so
        cache-hit and terminal runs stop there too. Returns CollectResults,
        or None when this trajectory is not searching."""
        if self.state != "searching":
            return None
        left = self.boundary - self.root_n()
        a = max(1, min(max_new, left))
        res = self.tree.collect_many_leaves(a, left - a + 1)
        n = self.root_n()
        if n > self.boundary:
            raise RuntimeError(f"[collect] {self.tid}: root N {n} passed check-in "
                               f"{self.boundary}")
        if n == self.boundary:
            self.on_checkin(n)
        return res

    def on_checkin(self, root_n):
        stage = self.stages[-1]
        feats = np.asarray(self.tree.search_check_in_features(), np.float32)
        root = self.tree.root()
        if self.context is None:
            self.context = np.asarray(root.search_context_features(), np.float32)
        details = self.tree.root_child_details()
        flip = 1.0 if self.fb.white_to_move() else -1.0
        n_arr = np.zeros(len(self.moves), np.int32)
        d_arr = np.zeros(len(self.moves), np.int32)
        q_arr = np.zeros(len(self.moves), np.float32)
        for cd in details:
            i = self.col[cd.uci]
            n_arr[i], d_arr[i], q_arr[i] = cd.N, cd.n_done, flip * cd.Q
        leader = details[0].uci
        self.rows.append({"features": feats, "root_n": root_n,
                          "n_done": int(root.n_done), "move": leader,
                          "stage_id": stage["stage_id"], "child_n": n_arr,
                          "child_n_done": d_arr, "child_q": q_arr})
        self.teacher.request(self.key, self.cb, leader, self.depth)
        self.boundary += self.interval

        if stage["reference"] is None:
            proof = dominance([cd.N for cd in details], root_n, stage["ceiling"],
                              self.min_decision)
            if proof["locked"] or root_n >= stage["ceiling"]:
                self.decide(stage, details, proof, root_n)
                return
        elif stage["approved"]:
            if root_n >= self.continuation["endpoint"]:
                self.end_search("approved")
        elif root_n >= stage["ceiling"]:
            self.extend()

    def decide(self, stage, details, proof, root_n):
        winner = details[0].uci
        window = self.coll["xc0_equiv_window"]
        recent = [r["move"] for r in self.rows[-window:]]
        flip = 1.0 if self.fb.white_to_move() else -1.0
        stats = {cd.uci: (cd.N, cd.n_done, flip * cd.Q) for cd in details}
        stage["reference"] = {
            "checkin": len(self.rows) - 1, "sims": root_n,
            "how": "lock" if proof["locked"] else "ceiling", "winner": winner,
            "proof": proof, "recent_leaders": recent,
            "children": {m: stats[m] for m in self.moves if m in stats},
            "xc0_set": xc0_equiv_set(winner, recent, stats,
                                     self.coll["xc0_equiv_visits"],
                                     self.coll["xc0_equiv_q"]),
        }
        self.teacher.request(self.key, self.cb, winner, self.depth, urgent=True)
        self.state = "judging"

    def poll(self):
        """Advance on teacher evals that have landed."""
        if self.state == "judging":
            self.judge()
        if self.state == "labelling" and self.labels_ready():
            self.state = "finished"

    def judge(self):
        stage = self.stages[-1]
        winner = stage["reference"]["winner"]
        base = self.teacher.baseline(self.key, self.depth)
        ev = self.teacher.move(self.key, winner, self.depth)
        if base is None or ev is None:
            return
        approved = winner in expert_set(base, {winner: ev})
        stage["approved"] = approved
        stage["expert"] = {"depth": self.depth, "baseline_move": base[0],
                           "baseline": base[1], "winner_eval": ev}
        root_n = self.root_n()
        if approved:
            left = (stage["ceiling"] - root_n) // self.interval
            drawn, effective, k = sample_continuation(self.rng, left,
                                                      self.coll["continuation"])
            self.continuation = {
                "policy": CONTINUATION_POLICY, "stage_id": stage["stage_id"],
                "approval_checkin": len(self.rows) - 1, "approval_sims": root_n,
                "intervals_left": left, "drawn": drawn, "branch": effective,
                "endpoint": root_n + k * self.interval}
            if k == 0:
                self.end_search("approved")
            else:
                self.state = "searching"
        elif root_n >= stage["ceiling"]:
            self.extend()
        else:
            self.state = "searching"

    def extend(self):
        """Double the ceiling on the same tree, or end at the global cap."""
        if self.ceiling >= self.cap:
            self.end_search("global_cap")
            return
        self.ceiling = min(2 * self.ceiling, self.cap)
        self.tree.set_params(sims_max=self.ceiling)
        self.stages.append(self.new_stage(self.root_n()))
        self.state = "searching"

    def end_search(self, reason):
        self.end_reason = reason
        self.teacher.flush(self.key, self.depth)
        self.final_sims = self.root_n()
        self.forest.pop_tree(self.tree)
        self.tree = None
        self.state = "labelling"

    def labels_ready(self):
        if self.teacher.baseline(self.key, self.depth) is None:
            return False
        return all(self.teacher.move(self.key, m, self.depth) is not None
                   for m in {r["move"] for r in self.rows})

    def record(self):
        base = self.teacher.baseline(self.key, self.depth)
        cands = sorted({r["move"] for r in self.rows})
        evals = {m: self.teacher.move(self.key, m, self.depth) for m in cands}
        evals = {m: ev for m, ev in evals.items() if ev is not None}
        expert = expert_set(base, evals) if base is not None else set()
        moves = [r["move"] for r in self.rows]
        stage_ids = np.array([r["stage_id"] for r in self.rows], np.int32)
        stage_sets = [s["reference"]["xc0_set"] if s["reference"] else None
                      for s in self.stages]
        targets, valid = build_targets(moves, stage_ids, stage_sets, expert,
                                       set(evals) if base is not None else set())
        pos = self.pos
        return {
            "schema_version": SCHEMA_VERSION,
            "trajectory_id": self.tid,
            "position_group_id": self.spec["position_group_id"],
            "effort_bin": self.spec["effort_bin"],
            "position_selection_id": self.spec["position_selection_id"],
            "position": {k: pos[k] for k in POS_FIELDS},
            "moves": self.moves,
            "context": self.context,
            "checkins": np.stack([r["features"] for r in self.rows]),
            "root_visits": np.array([r["root_n"] for r in self.rows], np.int32),
            "completed_backups": np.array([r["n_done"] for r in self.rows], np.int32),
            "checkpoint_move": moves,
            "stage_id": stage_ids,
            "child_n": np.stack([r["child_n"] for r in self.rows]),
            "child_n_done": np.stack([r["child_n_done"] for r in self.rows]),
            "child_q": np.stack([r["child_q"] for r in self.rows]),
            "targets": targets,
            "target_valid": valid,
            "stages": self.stages,
            "continuation": self.continuation,
            "end_reason": self.end_reason,
            "final_sims": self.final_sims,
            "teacher": {"depth": self.depth, "rule": EXPERT_RULE,
                        "baseline_move": base[0] if base else None,
                        "baseline": base[1] if base else None, "evals": evals,
                        "expert_set": sorted(expert)},
            "xc0_equiv_rule": XC0_EQUIV_RULE,
            "search": {"c_puct": self.c_puct, "min_decision": int(self.min_decision),
                       "initial_ceiling": self.stages[0]["ceiling"],
                       "stratum_params": self.search_params},
            "wall_s": time.time() - self.t0,
        }


def worker_main(wid, cfg, coll, remote, job_q, res_q, out_dir):
    """Process target. Searching trajectories count toward max_live; ones
    waiting only on teacher labels have released their trees and do not,
    so the GPU stays fed while SF catches up. Records go straight to this
    worker's shards; the main process only gets their summaries."""
    from chessbot.infer_server import make_remote_infer
    infer = make_remote_infer(remote)
    out = Output(out_dir, coll["shard_size"], f"w{wid}")
    macro = cfg.macro_batch
    forest = MCTSForest()
    teacher = Teacher(cfg.sf_config, coll["sf_engines"], seed=load_sf_cache(),
                      log_path=os.path.join(out_dir, f"sf_evals_w{wid}.pkl"),
                      hash_mb=coll["sf_hash"])
    live, labelling = [], []
    depth = coll["teacher_depth"]
    no_more = False
    tel = Counter()
    tel_t0 = time.time()
    try:
        while True:
            # refill on searching count, so trajectories paused on SF never
            # take search slots; max_trees bounds the trees kept alive
            searching = sum(t.state == "searching" for t in live)
            while (not no_more
                   and searching + coll["chunk_size"] <= coll["max_live"]
                   and len(live) + coll["chunk_size"] <= coll["max_trees"]):
                try:
                    chunk = job_q.get_nowait()
                except queue.Empty:
                    break
                if chunk is None:
                    no_more = True
                    break
                live += [Trajectory(spec, cfg, coll, teacher, forest, depth)
                         for spec in chunk]
                searching += len(chunk)
            if not live and not labelling:
                if no_more:
                    break
                time.sleep(0.05)
                continue

            teacher.drain()
            depth = next_teacher_depth(depth, teacher.backlog(), coll)
            for tr in live + labelling:
                tr.poll()

            forest.resolve_all_inflight()
            filled = 0
            for tr in live:
                if tr.state != "searching":
                    continue
                res = tr.step(min(micro_batch_for(tr.tree, cfg), macro - filled))
                if res is None:
                    continue
                filled += res.count_new
                tel["new"] += res.count_new
                tel["cached"] += res.count_cached
                tel["terminal"] += res.count_terminal
                if filled >= macro:
                    break

            keys_np, enc_np = forest.get_all_history_tokens(XC0H_K)
            for i in range(0, len(keys_np), macro):
                pol, wdl = infer((enc_np[i:i + macro],))
                raw_cache_bulk_insert_np(keys_np[i:i + macro], wdl, pol)
                tel["batches"] += 1
                tel["batch_n"] += len(keys_np[i:i + macro])

            for tr in [t for t in live if t.state in ("labelling", "finished")]:
                live.remove(tr)
                labelling.append(tr)
            for tr in [t for t in labelling if t.state == "finished"]:
                labelling.remove(tr)
                rec = tr.record()
                out.write(rec)
                res_q.put({"worker": wid, "summary": summary_of(rec)})

            now = time.time()
            if now - tel_t0 >= WORKER_TELEMETRY_S:
                leaves = tel["new"] + tel["cached"] + tel["terminal"]
                res_q.put({"worker": wid, "telemetry": {
                    "lps": leaves / (now - tel_t0), "new": tel["new"],
                    "cached": tel["cached"], "terminal": tel["terminal"],
                    "batch": tel["batch_n"] / max(tel["batches"], 1),
                    "searching": sum(t.state == "searching" for t in live),
                    "judging": sum(t.state == "judging" for t in live),
                    "labelling": len(labelling),
                    "sf_backlog": teacher.backlog(), "sf_evals": teacher.n_evals,
                    "sf_depth": depth}})
                tel, tel_t0 = Counter(), now
    finally:
        teacher.close()
        res_q.put({"worker": wid, "worker_done": True})


def build_specs(coll, corpus_name):
    """Trajectory specs, with positions under two legal moves set aside.
    Returns (specs, excluded).

    position_group_id is the source game and ply ("<game_id>:<abs_ply>"),
    effort_bin the looper's (phase, |Q|, p_draw) bin, position_selection_id
    how the corpus picked its positions ("close" or "random"); a corpus
    "random" stratum row is always "random"."""
    from pyfastchess import Board
    selection = coll.get("position_selection")
    if selection is None:
        if "close" not in corpus_name:
            raise ValueError(f"[collect] set position_selection for corpus {corpus_name}")
        selection = "close"
    corpus = pd.read_pickle(os.path.join(CORPUS_DIR, f"{corpus_name}.pkl.gz"))
    if coll.get("limit"):
        corpus = corpus.sample(n=min(coll["limit"], len(corpus)),
                               random_state=coll["seed"])
    specs, excluded = [], []
    for pos_id, row in corpus.iterrows():
        pos = {f: row[f] for f in POS_FIELDS}
        tid = f"{corpus_name}:{int(pos_id)}"
        if len(Board(pos["fen"]).legal_moves()) < 2:
            excluded.append({"trajectory_id": tid, "reason": "under_two_legal_moves"})
            continue
        specs.append({"trajectory_id": tid, "pos_id": int(pos_id), "pos": pos,
                      "position_group_id": f"{pos['game_id']}:{int(pos['abs_ply'])}",
                      "effort_bin": bin_key(int(pos["abs_ply"]),
                                            float(pos["parentQ_white"]),
                                            float(pos["parent_p_draw"])),
                      "position_selection_id": ("random" if pos["stratum"] == "random"
                                                else selection)})
    return specs, excluded


def check_settings(coll):
    k = coll["check_every"]
    for name in ("initial_budget", "short_budget", "global_cap"):
        v = coll[name]
        if v is not None and v % k:
            raise ValueError(f"[collect] {name} {v} is not a multiple of check_every {k}")
    for stratum, s in coll["strata"].items():
        for name in ("budget", "cap"):
            if name in s and s[name] % k:
                raise ValueError(f"[collect] strata {stratum} {name} {s[name]} is not a "
                                 f"multiple of check_every {k}")
    p = coll["continuation"]
    if abs(p["full"] + p["stop"] + p["random"] - 1.0) > 1e-9:
        raise ValueError(f"[collect] continuation odds must sum to 1: {p}")


def git_rev(path):
    try:
        rev = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def shard_paths(out_dir, prefix=""):
    head = f"trajectories_{prefix}"
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                  if f.startswith(head) and f.endswith(".pkl"))


def read_shard_ids(path):
    """Trajectory IDs in one shard. A partial record left by a crash is cut
    off the end so later appends stay readable."""
    ids, good = [], 0
    with open(path, "rb") as f:
        while True:
            try:
                rec = Unpickler(f).load()
            except (EOFError, pickle.UnpicklingError):
                break
            ids.append(rec["trajectory_id"])
            good = f.tell()
    if good < os.path.getsize(path):
        with open(path, "r+b") as f:
            f.truncate(good)
    return ids


def done_trajectory_ids(out_dir):
    return {tid for path in shard_paths(out_dir) for tid in read_shard_ids(path)}


class Output:
    """One worker's records, appended to its own sharded pickle streams
    (trajectories_w<id>_<shard>.pkl), so records never pass through the
    main process."""

    def __init__(self, out_dir, shard_size, prefix):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.prefix = prefix
        self.shard, self.in_shard = 0, 0
        paths = shard_paths(out_dir, prefix + "_")
        if paths:
            self.shard = int(paths[-1][-9:-4])
            self.in_shard = len(read_shard_ids(paths[-1]))

    def write(self, rec):
        if self.in_shard >= self.shard_size:
            self.shard, self.in_shard = self.shard + 1, 0
        path = os.path.join(self.out_dir,
                            f"trajectories_{self.prefix}_{self.shard:05d}.pkl")
        with open(path, "ab") as f:
            pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.in_shard += 1


def iter_shard(path):
    """One shard's trajectory records, one at a time."""
    with open(path, "rb") as f:
        while True:
            try:
                yield Unpickler(f).load()
            except EOFError:
                return


def iter_records(out_dir):
    """Every trajectory record in a collection dir, one at a time, shard by
    shard, so a reader never holds more than one full record."""
    for path in shard_paths(out_dir):
        yield from iter_shard(path)


def load_records(out_dir):
    """Every trajectory record in a collection dir."""
    return list(iter_records(out_dir))


def summary_of(rec):
    """The few numbers the status line needs."""
    return {"end_reason": rec["end_reason"], "checkins": len(rec["root_visits"]),
            "sims": rec["final_sims"], "stages": len(rec["stages"]),
            "quality_valid": float(rec["target_valid"][:, 1].mean()),
            "depth": rec["teacher"]["depth"]}


class Status:
    def __init__(self, n_total):
        self.n_total = n_total
        self.n = 0
        self.ends = Counter()
        self.checkins = []
        self.sims = []
        self.stages = []
        self.valid_q = []
        self.depths = []
        self.workers = {}

    def add(self, summary):
        self.n += 1
        self.ends[summary["end_reason"]] += 1
        self.checkins.append(summary["checkins"])
        self.sims.append(summary["sims"])
        self.stages.append(summary["stages"])
        self.valid_q.append(summary["quality_valid"])
        self.depths.append(summary["depth"])

    def line(self, elapsed):
        w = list(self.workers.values())
        rate = self.n / max(elapsed, 1e-9)
        eta = compact_span((self.n_total - self.n) / rate) if rate > 0 else "--"
        ends = "  ".join(f"{k} {v}" for k, v in sorted(self.ends.items()))
        return "\n".join([
            f"[collect] {self.n}/{self.n_total} trajectories   eta {eta}",
            f"      speed     {compact_count(sum(t['lps'] for t in w))} lps   "
            f"batch {np.mean([t['batch'] for t in w]) if w else 0:.0f}   "
            f"{rate * 60:.0f} traj/m",
            f"      live      searching {sum(t['searching'] for t in w)}   "
            f"judging {sum(t['judging'] for t in w)}   "
            f"labelling {sum(t['labelling'] for t in w)}",
            f"      sf        evals {sum(t['sf_evals'] for t in w)}   "
            f"backlog {sum(t['sf_backlog'] for t in w)}   "
            f"depth now {'/'.join(str(t['sf_depth']) for t in w) or '-'}   "
            f"avg {np.mean(self.depths) if self.depths else 0:.2f}",
            f"      ends      {ends}",
            f"      per traj  check-ins {np.mean(self.checkins) if self.checkins else 0:.1f}   "
            f"sims {np.mean(self.sims) if self.sims else 0:.0f}   "
            f"stages {np.mean(self.stages) if self.stages else 0:.2f}   "
            f"quality valid {100 * np.mean(self.valid_q) if self.valid_q else 0:.1f}%",
        ])


def collection_meta(coll, cfg, n_specs, excluded):
    from chessbot.infer_ort_trt import selfplay_trt_paths
    _, model_name, onnx_path = selfplay_trt_paths(cfg)
    here = os.path.dirname(os.path.abspath(__file__))
    import pyfastchess
    return {
        "schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
        "context_feature_names": list(NN_FEATURE_NAMES) + list(BOARD_FEATURE_NAMES),
        "checkin_feature_names": list(CHECK_IN_FEATURE_NAMES),
        "rules": {"continuation": CONTINUATION_POLICY, "xc0_equiv": XC0_EQUIV_RULE,
                  "expert": EXPERT_RULE},
        "collection": coll, "n_trajectories": n_specs, "excluded": excluded,
        "nn": {"model_name": model_name, "onnx": onnx_path},
        "code": {"chessbot": git_rev(here),
                 "pyfastchess": git_rev(os.path.dirname(os.path.dirname(
                     os.path.dirname(pyfastchess.__file__))))},
        "search_overrides": {"dirichlet_eps": 0.0, "es_check_every": 0,
                             **{k: coll[k] for k in SEARCH_OVERRIDES}},
    }


def run_collection(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    coll = {**DEFAULTS, **{k: v for k, v in raw.items()
                           if k not in ("name", "run_config", "corpus", "resume")}}
    check_settings(coll)
    if coll["max_trees"] is None:
        coll["max_trees"] = 2 * coll["max_live"]
    cfg = Config.from_yaml(raw["run_config"])
    cfg.sampleable = None
    # no root noise; the old tier/RSC/gap-block stop rules and their snapshots off
    cfg.update({"dirichlet_eps": 0.0, "es_check_every": 0, "es_record_all": False,
                **{k: coll[k] for k in SEARCH_OVERRIDES}})

    if raw.get("resume"):
        out_dir = os.path.join(COLLECTIONS_DIR, raw["resume"])
    else:
        out_dir = os.path.join(COLLECTIONS_DIR,
                               f"{raw['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir)
    done_ids = done_trajectory_ids(out_dir)
    print(f"[collect] SF cache: {sync_sf_cache()} evals", flush=True)
    specs, excluded = build_specs(coll, raw["corpus"])
    specs = [s for s in specs if s["trajectory_id"] not in done_ids]
    random.Random(coll["seed"]).shuffle(specs)
    with open(os.path.join(out_dir, "collection.yaml"), "w") as f:
        yaml.safe_dump(raw, f)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(collection_meta(coll, cfg, len(specs), excluded), f, indent=1,
                  default=str)

    ctx = mp.get_context("spawn")
    n_workers = coll["n_workers"]
    server, ctrl, slots = start_server(ctx, cfg, n_workers, coll["server_threads"])
    job_q, res_q = ctx.Queue(), ctx.Queue()
    job_q.cancel_join_thread()
    for i in range(0, len(specs), coll["chunk_size"]):
        job_q.put(specs[i:i + coll["chunk_size"]])
    for _ in range(n_workers):
        job_q.put(None)
    workers = [ctx.Process(target=worker_main,
                           args=(w, cfg, coll, slots[w], job_q, res_q, out_dir),
                           daemon=True)
               for w in range(n_workers)]
    for p in workers:
        p.start()
    print(f"[collect] {len(specs)} trajectories ({len(done_ids)} done, "
          f"{len(excluded)} excluded) -> {n_workers} workers, output {out_dir}",
          flush=True)

    t0 = time.time()
    status = Status(len(specs))
    n_done = 0
    try:
        while n_done < n_workers:
            try:
                msg = res_q.get(timeout=1.0)
            except queue.Empty:
                if not any(p.is_alive() for p in workers):
                    raise RuntimeError("[collect] every worker died")
                continue
            if msg.get("worker_done"):
                n_done += 1
            elif "telemetry" in msg:
                status.workers[msg["worker"]] = msg["telemetry"]
            else:
                status.add(msg["summary"])
                if status.n % STATUS_EVERY == 0:
                    print("\n" + status.line(time.time() - t0), flush=True)
        print("\n" + status.line(time.time() - t0), flush=True)
    finally:
        ctrl.send({"cmd": "stop"})
        server.join(timeout=30)
        for p in workers:
            p.join(timeout=10)
        print(f"[collect] SF cache: {sync_sf_cache()} evals", flush=True)
    print(f"[collect] done: {status.n} trajectories in {time.time() - t0:.0f}s", flush=True)
    return out_dir


if __name__ == "__main__":
    from chessbot.experimentation import collector
    code = 1
    try:
        collector.run_collection(sys.argv[1])
        code = 0
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
