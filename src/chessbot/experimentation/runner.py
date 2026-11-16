"""Experiment runner: search corpus positions under arms and record every
check-in.

A job is one corpus position under one arm. Its tree runs with
es_record_all + es_observe_only, so the C++ side snapshots every check-in and
never stops on its own; the RPO rule here decides when the job is done:

- stop once xc0 would stop AND its pick is SF-equivalent to the SF best
  (is_equiv: the same move, within GOOD_CP, or within GOOD_ESL unless the
  best is a forced mate; evals at MIN_LABEL_DEPTH+);
- a pick with no label queues an async SF eval and the search keeps going;
- at sims_max (the C++ "full" stop) stop if the robust pick is good, else
  double the budget, up to SIMS_CAP (or the experiment's sims_cap), after
  which the job ends as never-found. The same tree carries on by default; an arm with rebuild:
  true starts a fresh one (the NN cache stays warm).

Every would-stop, label request and label arrival is logged with the sim
count it happened at; post-processing reads what happened from there.
"""
import multiprocessing as mp
import os
import pickle
import queue
import random
import sys
import threading
import time
from typing import NamedTuple, Optional

import chess
import chess.engine
import numpy as np
import pandas as pd
import yaml
from pyfastchess import Board as fastboard
from pyfastchess import MCTSForest, raw_cache_bulk_insert_np

from chessbot import SF_LOC
from chessbot.config import XC0H_K, Config
from chessbot.experimentation.corpus import CORPUS_DIR, CP_MAX
from chessbot.mcts_utils import MCTSTree, micro_batch_for
from chessbot.utils import (
    compact_count, compact_span, cp_to_value_tanh)

EXPERIMENTS_DIR = r"D:\Data\chessbot_data\experiments\runs"
SERVER_WAIT_S = 300.0
WORKER_TELEMETRY_S = 5.0
STATUS_EVERY_JOBS = 100
SUMMARY_EVERY_JOBS = 500
# master corpus and labels are rewritten after this many merged results
SYNC_EVERY = 200
POS_FIELDS = ["init_fen", "uci_path", "fen", "short_fen", "best_move",
              "best_cp", "played_move", "played_cp", "cpl_depth"]

GOOD_CP = 10
# a pick also counts as SF-equivalent within this expected score loss of the
# best, unless the best is a forced mate for the side to move
GOOD_ESL = 0.02
MIN_LABEL_DEPTH = 16
SIMS_CAP = 16000
# check-ins whose xc0 pick is SF-labelled for the CPL-by-sims report; the
# top ones line up with sims_max doubling to SIMS_CAP
MILESTONE_SIMS = (200, 500, 1000, 2000, 4000, 8000, 16000)
# past THIN_ABOVE_SIMS a snapshot is kept only once per THIN_EVERY sims,
# unless it would stop, vetoes on the gap, or changes the leader, pick or
# stop reason
THIN_ABOVE_SIMS = 6000
THIN_EVERY = 200
# snapshots keep only the moves that were in the top TOP_CHILDREN by N at any
# check-in of the tree, each one at every check-in
TOP_CHILDREN = 8
# report rows: share of jobs whose search went past each count
REACH_SIMS = (500, 1000, 2000, 4000, 8000, 16000)
# own-move ESL bars for the summary: rescore's inaccuracy and blunder
ESL_BARS = (0.05, 0.12)


class SfEval(NamedTuple):
    """SF's eval of one move played from one position.

    Every value is from the point of view of the side to move in that
    position -- the player choosing the move -- so higher is always better
    for them and the loss of a pick is best.cp_stm - pick.cp_stm.
    """
    cp_stm: int                # clipped to +-CP_MAX; a forced mate is +-CP_MAX
    mate_stm: Optional[int]    # None, or moves to mate: +N they mate, -N they get mated
    depth: Optional[int]
    source: str                # "corpus" (rescore data) or "sf" (evaluated here)


def mate_sign(ev):
    """+1 / -1 when ev is a forced mate for / against the side to move, else
    0. Corpus evals record no mates, so a cp at +-CP_MAX counts as one."""
    if ev.mate_stm is not None:
        return 1 if ev.mate_stm > 0 else -1
    if abs(ev.cp_stm) >= CP_MAX:
        return 1 if ev.cp_stm > 0 else -1
    return 0


def sf_value(ev):
    """ev as a -1..1 value: a forced mate is exactly +-1, anything else goes
    through cp_to_value_tanh."""
    m = mate_sign(ev)
    return float(m) if m else float(cp_to_value_tanh(ev.cp_stm))


def is_equiv(best, ev):
    """Whether pick eval ev is as good as best for the RPO rule: within
    GOOD_CP of it (or better), or within GOOD_ESL expected score loss on
    sf_value. A forced mate for the side to move only accepts the cp rule,
    so missing a mate is never equivalent."""
    if best.cp_stm - ev.cp_stm <= GOOD_CP:
        return True
    if mate_sign(best) > 0:
        return False
    return 0.5 * max(0.0, sf_value(best) - sf_value(ev)) <= GOOD_ESL


def sf_eval_from_score(score, depth):
    """SfEval from a python-chess PovScore of the position's side to move."""
    rel = score.relative
    mate = rel.mate()
    if mate is not None:
        cp = CP_MAX if mate > 0 else -CP_MAX
    else:
        cp = int(np.clip(rel.score(), -CP_MAX, CP_MAX))
    return SfEval(cp_stm=cp, mate_stm=mate, depth=depth, source="sf")

SNAP_FIELDS = [
    "sims", "n_done", "ns", "tot_new", "tot_cached", "tot_terminal",
    "tot_blocked", "inflight", "jsd", "delta_12", "dQ12", "tier1_fail",
    "tier2_fail", "would_stop", "reason", "pick", "gap_veto", "rsc_argmax", "root_Q",
    "root_Qmm", "root_win", "root_draw", "root_loss", "avg_depth", "max_depth",
    "pv1_len", "pv2_len",
]
CHILD_FIELDS = [
    "uci", "N", "n_done", "Q", "Q_eff", "Qema", "Qmm", "Qdelta_sign",
    "prior", "raw_prior", "visit_share", "last_visit", "boosted_N", "forced_N",
    "win", "draw", "loss", "is_terminal",
]


def snapshot_dict(s):
    """CheckinSnapshot as plain python, so it pickles and outlives the tree."""
    d = {f: getattr(s, f) for f in SNAP_FIELDS}
    d["children"] = [dict({f: getattr(c, f) for f in CHILD_FIELDS}, rcs=r)
                     for c, r in zip(s.children, s.rcs)]
    return d


def leader(s):
    return s["children"][0]["uci"] if s["children"] else None


def keep_top_children(snaps):
    keep = set()
    for s in snaps:
        top = sorted(s["children"], key=lambda c: -c["N"])[:TOP_CHILDREN]
        keep.update(c["uci"] for c in top)
    for s in snaps:
        s["children"] = [c for c in s["children"] if c["uci"] in keep]
    return snaps


def thin_snapshots(snaps):
    """snaps with the quiet check-ins past THIN_ABOVE_SIMS dropped; the last
    one is always kept."""
    out, prev, kept_bucket = [], None, -1
    for i, s in enumerate(snaps):
        bucket = s["sims"] // THIN_EVERY
        keep = (s["sims"] <= THIN_ABOVE_SIMS or bucket > kept_bucket
                or s["would_stop"] or s["gap_veto"] or i == len(snaps) - 1
                or (prev is not None and (leader(s) != leader(prev)
                                          or s["pick"] != prev["pick"]
                                          or s["reason"] != prev["reason"])))
        if keep:
            out.append(s)
            kept_bucket = bucket
        prev = s
    return out


def position_boards(pos):
    """The position as a pyfastchess board (history for xc0h) and a
    python-chess board (repetitions for SF)."""
    fb = fastboard(pos["init_fen"])
    cb = chess.Board(pos["init_fen"])
    for uci in pos["uci_path"]:
        fb.push_uci(uci)
        cb.push_uci(uci)
    return fb, cb


class LabelThread:
    """One SF engine on its own thread: evaluates a single move of a position
    at a fixed depth. Requests and results go through queues."""

    def __init__(self, sf_config):
        self.sf_config = sf_config
        self.req_q = queue.Queue()
        self.res_q = queue.Queue()
        self.stop_ev = threading.Event()
        self.err = None
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def submit(self, key, board, uci, depth):
        self.req_q.put((key, board.copy(stack=True), uci, depth))

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.res_q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self.stop_ev.set()
        self.t.join()
        if self.err is not None:
            raise self.err

    def run(self):
        eng = chess.engine.SimpleEngine.popen_uci(SF_LOC, timeout=60)
        eng.configure(self.sf_config)
        try:
            while not self.stop_ev.is_set():
                try:
                    key, board, uci, depth = self.req_q.get(timeout=0.01)
                except queue.Empty:
                    continue
                info = eng.analyse(board, chess.engine.Limit(depth=depth),
                                   root_moves=[chess.Move.from_uci(uci)])
                ev = sf_eval_from_score(info["score"], info.get("depth", depth))
                self.res_q.put((key, uci, depth, ev))
        except Exception as e:
            self.err = e
            self.res_q.put(("__error__", None, None, None))
        finally:
            eng.quit()


class LabelBook:
    """SF move evals: labels[short_fen][uci] = SfEval for that move played
    from that position. Seeded from the corpus; an eval missing or shallower
    than the depth a job asks for is run once by the label thread and kept.
    A deeper eval replaces a shallower one."""

    def __init__(self, sf_config):
        self.labels = {}
        self.pending = set()
        self.n_evals = 0
        self.thread = LabelThread(sf_config)

    def store(self, key, uci, ev):
        have = self.labels.setdefault(key, {}).get(uci)
        if have is None or (ev.depth or 0) > (have.depth or 0):
            self.labels[key][uci] = ev

    def seed_corpus(self, key, uci, cp_stm, depth):
        """Corpus cps are side-to-move POV already and record no mates."""
        if uci is None or cp_stm is None or np.isnan(cp_stm):
            return
        self.store(key, uci, SfEval(cp_stm=int(cp_stm), mate_stm=None,
                                    depth=depth, source="corpus"))

    def get(self, key, uci, depth=MIN_LABEL_DEPTH):
        """The SfEval at depth or deeper, or None."""
        ev = self.labels.get(key, {}).get(uci)
        if ev is None or (ev.depth or 0) < depth:
            return None
        return ev

    def request(self, key, board, uci, depth=MIN_LABEL_DEPTH):
        """Queue an eval at depth unless one that deep is known or already
        queued. Returns True when a new request went out."""
        if self.get(key, uci, depth) is not None or (key, uci, depth) in self.pending:
            return False
        self.pending.add((key, uci, depth))
        self.thread.submit(key, board, uci, depth)
        return True

    def drain(self):
        """Store finished evals; returns the (key, uci) pairs that landed."""
        landed = []
        for key, uci, depth, ev in self.thread.drain():
            if key == "__error__":
                self.thread.close()
            self.pending.discard((key, uci, depth))
            self.store(key, uci, ev)
            self.n_evals += 1
            landed.append((key, uci))
        return landed

    def close(self):
        self.thread.close()


class Job:
    """One corpus position under one arm, searched until the RPO rule ends
    it.

    At sims_max without an SF-good pick the budget doubles: by default the
    same tree carries on with sims_max raised; an arm with rebuild: true
    starts a fresh tree instead (the NN cache stays warm), and the fresh
    tree's data replaces the old. SF judges at self.depth, MIN_LABEL_DEPTH
    plus one per doubling.

    native is the move xc0 itself would play: the pick at the first check-in
    its stop rule fires, else its robust pick at the first sims_max.
    milestones holds xc0's pick at each MILESTONE_SIMS check-in.
    """

    def __init__(self, spec, base_cfg, book, forest, sims_cap=SIMS_CAP):
        self.spec = spec
        self.sims_cap = sims_cap
        self.pos = pos = spec["pos"]
        self.key = pos["short_fen"]
        self.corpus_best = pos["best_move"]
        self.book = book
        self.forest = forest
        self.rebuild = bool(spec["arm"].get("rebuild", False))

        cfg = base_cfg.copy()
        cfg.update(spec["arm"].get("overrides", {}))
        self.cfg = cfg
        self.check_every = cfg.es_check_every

        self.tree = None
        self.depth = MIN_LABEL_DEPTH
        self.doublings = 0
        self.total_sims = 0      # sims spent, every tree included
        self.attempt = None      # the final tree
        self.prior = []          # each doubling: sims_max, depth, pick, cpl
        self.milestones = {}     # check-in sims -> xc0 pick
        self.native = None       # {"sims", "pick", "how"}
        self.events = []
        self.outcome = None
        self.waiting_on = None   # pick awaiting a label after a full stop
        self.t0 = time.time()

        for uci, ev in spec.get("labels", {}).items():
            book.store(self.key, uci, ev)
        book.seed_corpus(self.key, pos["best_move"], pos["best_cp"], pos["cpl_depth"])
        book.seed_corpus(self.key, pos["played_move"], pos["played_cp"], pos["cpl_depth"])
        self.fb, self.cb = position_boards(pos)
        self.request(self.corpus_best)
        self.new_tree(cfg.sims_ceiling)

    def new_tree(self, sims_max):
        self.tree = MCTSTree(self.fb.clone(), self.cfg)
        self.tree.set_params(sims_max=int(sims_max), es_record_all=True,
                             es_observe_only=True)
        self.forest.add_tree(self.tree)
        self.sims_max = int(sims_max)
        self.seen_bucket = 0
        self.milestones = {}

    def end_tree(self, end_reason):
        snaps = thin_snapshots(keep_top_children(
            [snapshot_dict(s) for s in self.tree.gather_stop_stats()]))
        self.total_sims += self.sims()
        self.attempt = {"sims_max": self.sims_max, "depth": self.depth,
                        "end": end_reason, "snapshots": snaps}
        self.forest.pop_tree(self.tree)
        self.tree = None

    def sims(self):
        return int(self.tree.root().N) if self.tree is not None else 0

    def log(self, kind, **kw):
        self.events.append({"kind": kind, "doubling": self.doublings,
                            "sims": self.sims(), **kw})

    def request(self, uci):
        if self.book.request(self.key, self.cb, uci, self.depth):
            self.log("label_request", move=uci, depth=self.depth)

    def best(self):
        """(uci, SfEval) of the best move among the evals at self.depth or
        deeper, or (corpus best, None) before any exist."""
        known = [(u, ev) for u, ev in self.book.labels.get(self.key, {}).items()
                 if (ev.depth or 0) >= self.depth]
        if not known:
            return self.corpus_best, None
        return max(known, key=lambda kv: kv[1].cp_stm)

    def judge(self, pick):
        """(good, cpl) for a pick. cpl = best.cp_stm - pick.cp_stm, negative
        when the pick beats SF's best; good = is_equiv(best, pick).

        Nothing is judged until the corpus best has an eval at self.depth;
        until then the corpus best itself counts as good (cpl None) and
        every other pick waits: (None, None).
        """
        if self.book.get(self.key, self.corpus_best, self.depth) is None:
            return (True, None) if pick == self.corpus_best else (None, None)
        ev = self.book.get(self.key, pick, self.depth)
        if ev is None:
            return None, None
        _, best = self.best()
        return is_equiv(best, ev), best.cp_stm - ev.cp_stm

    def equiv_of(self, pick):
        """is_equiv of pick against the best at self.depth, None if unlabelled."""
        _, best = self.best()
        ev = self.book.get(self.key, pick, self.depth)
        return is_equiv(best, ev) if best is not None and ev is not None else None

    def cpl_of(self, pick):
        """cpl of pick against the best at self.depth, None if unlabelled."""
        _, best = self.best()
        ev = self.book.get(self.key, pick, self.depth)
        return best.cp_stm - ev.cp_stm if best is not None and ev is not None else None

    def esl_of(self, pick):
        """Expected score loss of pick against the best at self.depth, 0..1,
        on sf_value (forced mates are +-1), None if unlabelled."""
        _, best = self.best()
        ev = self.book.get(self.key, pick, self.depth)
        if best is None or ev is None:
            return None
        return 0.5 * max(0.0, sf_value(best) - sf_value(ev))

    def on_label(self, uci):
        ev = self.book.get(self.key, uci, 0)
        self.log("label_arrived", move=uci,
                 cp_stm=ev.cp_stm if ev else None,
                 mate_stm=ev.mate_stm if ev else None,
                 depth=ev.depth if ev else None)
        if self.waiting_on is not None:
            self.decide_full(self.waiting_on)

    def mbs(self):
        """Leaves this job asks for per collect, from its own arm config."""
        if self.tree is None:
            return self.cfg.micro_batch
        return micro_batch_for(self.tree, self.cfg)

    def step(self, mbs):
        """One collect on a live tree. Returns its CollectResults, or None
        when the job had nothing to collect."""
        if self.outcome is not None or self.tree is None or self.waiting_on:
            return None
        if self.tree.es_tripped:
            self.on_full()
            return None
        res = self.tree.collect_many_leaves(mbs)
        n = res.count_new + res.count_cached + res.count_terminal
        self.tree.sims_completed_this_move += n
        bucket = self.sims() // self.check_every
        if bucket > self.seen_bucket:
            self.seen_bucket = bucket
            snap = self.tree.latest_checkin()
            if snap is not None:
                self.on_checkin(snap)
        return res

    def mark_milestone(self, sims, pick):
        if sims in MILESTONE_SIMS and sims not in self.milestones:
            self.milestones[sims] = pick
            self.request(pick)

    def on_checkin(self, snap):
        if not snap.children:
            return
        self.mark_milestone(snap.sims, snap.children[0].uci)
        if not snap.would_stop:
            return
        pick = snap.pick or snap.children[0].uci
        if self.native is None:
            self.native = {"sims": snap.sims, "pick": pick, "how": "stop"}
            self.request(pick)
        good, cpl = self.judge(pick)
        self.log("would_stop", move=pick, good=good, cpl=cpl, reason=snap.reason)
        if good is None:
            self.request(pick)
        elif good:
            self.finish("found", pick, "would_stop", cpl)

    def on_full(self):
        """The tree hit sims_max. The full stop fires before that sim's
        check-in, so the milestone pick (most visited) is taken here; the
        move xc0 plays at a full stop is its robust pick."""
        sims = self.sims()
        self.tree.sim_stop_reason = ""
        top, _ = self.tree.select_xerces_top_move()
        self.mark_milestone(sims, top)
        self.tree.sim_stop_reason = "full"
        pick, _ = self.tree.select_xerces_top_move()
        if self.native is None:
            self.native = {"sims": sims, "pick": pick, "how": "full"}
            self.request(pick)
        self.log("full", move=pick)
        self.decide_full(pick)

    def decide_full(self, pick):
        good, cpl = self.judge(pick)
        if good is None:
            self.waiting_on = pick
            self.request(pick)
            self.request(self.best()[0])
            return
        self.waiting_on = None
        self.log("full_judged", move=pick, good=good, cpl=cpl)
        if good:
            self.finish("found", pick, "full", cpl)
        elif self.sims_max >= self.sims_cap:
            self.finish("never", pick, "cap", cpl)
        else:
            self.double(pick, cpl)

    def double(self, pick, cpl):
        """Double sims_max and judge one ply deeper from here on."""
        self.prior.append({"sims_max": self.sims_max, "depth": self.depth,
                           "pick": pick, "cpl": cpl})
        prev_best = self.best()[0]
        self.doublings += 1
        self.depth += 1
        self.request(self.corpus_best)
        self.request(prev_best)
        sims_max = min(2 * self.sims_max, self.sims_cap)
        if self.rebuild:
            self.end_tree("rebuild")
            self.new_tree(sims_max)
        else:
            self.tree.sim_stop_reason = ""
            self.tree.set_params(sims_max=sims_max)
            self.sims_max = sims_max

    def finish(self, status, pick, how, cpl):
        if self.tree is not None:
            self.end_tree(how)
        self.request(self.corpus_best)
        self.outcome = {"status": status, "pick": pick, "how": how, "cpl": cpl,
                        "corpus_best": self.corpus_best,
                        "doublings": self.doublings, "depth": self.depth,
                        "total_sims": self.total_sims,
                        "wall_s": time.time() - self.t0}

    def done(self):
        """Finished, and the corpus best, every milestone pick and the native
        pick have evals at self.depth (the tree is gone by then, so waiting
        is free)."""
        if self.outcome is None:
            return False
        need = [self.corpus_best, *self.milestones.values()]
        if self.native is not None:
            need.append(self.native["pick"])
        missing = [u for u in need if self.book.get(self.key, u, self.depth) is None]
        for u in missing:
            self.request(u)
        return not missing

    def result(self):
        best_uci, _ = self.best()
        milestones = {sims: {"pick": pick, "cpl": self.cpl_of(pick),
                             "equiv": self.equiv_of(pick)}
                      for sims, pick in self.milestones.items()}
        native = None
        if self.native is not None:
            pick = self.native["pick"]
            native = dict(self.native, cpl=self.cpl_of(pick), esl=self.esl_of(pick),
                          equiv=self.equiv_of(pick))
        snaps = self.attempt["snapshots"] if self.attempt else []
        return {"job_id": self.spec["job_id"], "arm": self.spec["arm"]["name"],
                "pos_id": self.spec["pos_id"],
                "outcome": dict(self.outcome, best=best_uci),
                "max_sims": snaps[-1]["sims"] if snaps else 0,
                "native": native, "milestones": milestones,
                "events": self.events, "attempt": self.attempt,
                "prior": self.prior, "labels_key": self.key,
                "labels": dict(self.book.labels.get(self.key, {}))}


def worker_main(wid, base_cfg, remote, job_q, res_q, max_live, chunk_jobs,
                sims_cap):
    """Process target. Pulls job chunks (see build_chunks) from job_q until
    a None, starting a chunk whenever another chunk_jobs still fits under
    max_live, so live jobs stay near max_live and chunks overlap. Leaves of
    all live jobs are batched through the xc0 server slot; each finished
    job's result goes on res_q."""
    from chessbot.infer_server import make_remote_infer
    infer = make_remote_infer(remote)
    macro = base_cfg.macro_batch
    forest = MCTSForest()
    book = LabelBook(base_cfg.sf_config)
    live, by_key = [], {}
    no_more = False
    tel_t0, tel_batches, tel_batch_n = time.time(), 0, 0
    tel_new = tel_cached = tel_terminal = 0

    try:
        while True:
            while not no_more and len(live) + chunk_jobs <= max_live:
                try:
                    chunk = job_q.get_nowait()
                except queue.Empty:
                    break
                if chunk is None:
                    no_more = True
                    break
                for spec in chunk:
                    job = Job(spec, base_cfg, book, forest, sims_cap)
                    live.append(job)
                    by_key.setdefault(job.key, []).append(job)
            if not live:
                if no_more:
                    break
                time.sleep(0.05)
                continue

            for key, uci in book.drain():
                for job in by_key.get(key, []):
                    job.on_label(uci)

            # oldest jobs first, stopping once one NN batch is full, as the
            # selfplay looper does
            forest.resolve_all_inflight()
            filled = 0
            for job in live:
                res = job.step(min(job.mbs(), macro - filled))
                if res is None:
                    continue
                filled += res.count_new
                tel_new += res.count_new
                tel_cached += res.count_cached
                tel_terminal += res.count_terminal
                if filled >= macro:
                    break

            keys_np, enc_np = forest.get_all_history_tokens(XC0H_K)
            for i in range(0, len(keys_np), macro):
                pol, wdl = infer((enc_np[i:i + macro],))
                raw_cache_bulk_insert_np(keys_np[i:i + macro], wdl, pol)
                tel_batches += 1
                tel_batch_n += len(keys_np[i:i + macro])

            for job in [j for j in live if j.done()]:
                res_q.put({"worker": wid, **job.result()})
                live.remove(job)
                by_key[job.key].remove(job)

            now = time.time()
            if now - tel_t0 >= WORKER_TELEMETRY_S:
                leaves = tel_new + tel_cached + tel_terminal
                res_q.put({"worker": wid, "telemetry": {
                    "lps": leaves / (now - tel_t0),
                    "new": tel_new, "cached": tel_cached,
                    "terminal": tel_terminal,
                    "batch": tel_batch_n / max(tel_batches, 1),
                    "live": len(live), "sf_pending": len(book.pending),
                    "sf_evals": book.n_evals}})
                tel_t0, tel_batches, tel_batch_n = now, 0, 0
                tel_new = tel_cached = tel_terminal = 0
    finally:
        book.close()
        res_q.put({"worker": wid, "worker_done": True})


class Master:
    """The corpus and its SF labels as they stand across runs.

    labels[short_fen][uci] = SfEval, kept beside the corpus as
    <corpus>_labels.pkl. merge() folds a result's labels in (a deeper eval
    wins) and moves the corpus row's best_move / best_cp / best_depth to the
    best deep eval when SF's corpus best turns out not to be best."""

    def __init__(self, corpus_name):
        self.corpus_path = os.path.join(CORPUS_DIR, f"{corpus_name}.pkl.gz")
        self.labels_path = os.path.join(CORPUS_DIR, f"{corpus_name}_labels.pkl")
        self.corpus = pd.read_pickle(self.corpus_path)
        if "best_depth" not in self.corpus.columns:
            self.corpus["best_depth"] = self.corpus["cpl_depth"]
        self.labels = {}
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "rb") as f:
                self.labels = Unpickler(f).load()
        self.row_of = {k: i for i, k in enumerate(self.corpus.short_fen)}
        self.n_best_moved = 0

    def merge(self, key, labels):
        mine = self.labels.setdefault(key, {})
        for uci, ev in labels.items():
            have = mine.get(uci)
            if have is None or (ev.depth or 0) > (have.depth or 0):
                mine[uci] = ev
        deep = [(u, ev) for u, ev in mine.items()
                if (ev.depth or 0) >= MIN_LABEL_DEPTH]
        if not deep:
            return
        uci, ev = max(deep, key=lambda kv: kv[1].cp_stm)
        i = self.row_of[key]
        if uci != self.corpus.at[i, "best_move"]:
            self.n_best_moved += 1
        self.corpus.at[i, "best_move"] = uci
        self.corpus.at[i, "best_cp"] = ev.cp_stm
        self.corpus.at[i, "best_depth"] = ev.depth

    def save(self):
        tmp = self.corpus_path + ".tmp"
        self.corpus.to_pickle(tmp, compression="gzip")
        os.replace(tmp, self.corpus_path)
        with open(self.labels_path + ".tmp", "wb") as f:
            pickle.dump(self.labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(self.labels_path + ".tmp", self.labels_path)


def report_cpl(cpl, equiv):
    """cpl as the reports count it: a pick the RPO rule calls SF-equivalent
    counts as 0."""
    return 0 if equiv else cpl


def equiv_flag(rec):
    """The SF-equivalence a result recorded for a pick. Results from before
    is_equiv carry only cpl, judged then by the cp rule alone."""
    return rec["equiv"] if "equiv" in rec else rec["cpl"] <= GOOD_CP


class ArmStats:
    """Totals for one arm, or for the whole run. Every cpl kept here has
    been through report_cpl; results.pkl keeps the raw values."""

    def __init__(self):
        self.n = 0
        # each job is exactly one of: own (xc0's own stop was SF-equiv),
        # later (found only by searching past it), never (not even at the cap)
        self.own = self.later = self.never = 0
        self.doublings = 0
        self.total_sims = 0
        self.stop_sims = []
        self.max_sims = []
        self.cpl = []
        self.milestone_cpl = {s: [] for s in MILESTONE_SIMS}
        self.native_sims = []
        self.native_cpl = []
        self.native_esl = []

    def add(self, res):
        o = res["outcome"]
        self.n += 1
        self.doublings += o["doublings"]
        self.total_sims += o["total_sims"]
        self.max_sims.append(res["max_sims"])
        if o["cpl"] is not None:
            # a found job ended on an SF-equivalent move; a never job did not
            self.cpl.append(report_cpl(o["cpl"], o["status"] == "found"))
        for sims, m in res["milestones"].items():
            if m["cpl"] is not None:
                self.milestone_cpl[sims].append(report_cpl(m["cpl"], equiv_flag(m)))
        own_good = False
        nat = res["native"]
        if nat is not None:
            self.native_sims.append(nat["sims"])
            if nat["cpl"] is not None:
                own_good = equiv_flag(nat)
                self.native_cpl.append(report_cpl(nat["cpl"], own_good))
                self.native_esl.append(0.0 if own_good else nat["esl"])
        if o["status"] == "never":
            self.never += 1
            return
        self.stop_sims.append(res["max_sims"])
        if own_good:
            self.own += 1
        else:
            self.later += 1

    def sims_per_job(self):
        return self.total_sims / max(self.n, 1)

    def median_stop(self):
        return int(np.median(self.stop_sims)) if self.stop_sims else 0


class Status:
    """Running totals for the launcher's reports."""

    def __init__(self, n_jobs, arm_names):
        self.n_jobs = n_jobs
        self.all = ArmStats()
        self.arms = {a: ArmStats() for a in arm_names}
        self.arm_w = max(len("arm"), *(len(a) for a in arm_names)) + 2
        self.workers = {}

    def add(self, res):
        self.all.add(res)
        self.arms[res["arm"]].add(res)

    def line(self, elapsed, best_moved):
        w = list(self.workers.values())
        lps = sum(t["lps"] for t in w)
        live = sum(t["live"] for t in w)
        batch = np.mean([t["batch"] for t in w]) if w else 0.0
        leaves = sum(t["new"] + t["cached"] + t["terminal"] for t in w) or 1
        hit = 100.0 * sum(t["cached"] for t in w) / leaves
        new = 100.0 * sum(t["new"] for t in w) / leaves
        term = 100.0 * sum(t["terminal"] for t in w) / leaves
        sf_evals = sum(t["sf_evals"] for t in w)
        sf_pending = sum(t["sf_pending"] for t in w)
        a = self.all
        rate = a.n / max(elapsed, 1e-9)
        eta = compact_span((self.n_jobs - a.n) / rate) if rate > 0 else "--"
        pad = " " * 6

        def row(label, *cells):
            return (pad + f"{label:<10}"
                    + "".join(f"{c:<16}" for c in cells)).rstrip()

        return "\n".join([
            f"[exp] {a.n}/{self.n_jobs} jobs   eta {eta}",
            row("speed", f"{compact_count(lps)} lps", f"live {live}",
                f"batch {batch:.0f}", f"{rate * 3600:.0f} jobs/h"),
            row("cache", f"hit {hit:.1f}%", f"new {new:.1f}%",
                f"terminal {term:.1f}%"),
            row("outcome", f"own {a.own}", f"later {a.later}",
                f"never {a.never}"),
            row("search", f"sims@stop {a.median_stop()}",
                f"sims/job {a.sims_per_job():.0f}", f"doublings {a.doublings}"),
            row("sf", f"evals {sf_evals}", f"pending {sf_pending}",
                f"best moved {best_moved}"),
        ])

    def table(self, title, header, rows):
        """Arm-per-row table; each column is as wide as its widest entry
        plus two spaces."""
        w = self.arm_w
        widths = [max(len(h), *(len(cells[i]) for _, cells in rows)) + 2
                  for i, h in enumerate(header)]
        out = [title, "  " + f"{'arm':<{w}}"
               + "".join(f"{h:>{cw}}" for h, cw in zip(header, widths))]
        for name, cells in rows:
            out.append("  " + f"{name:<{w}}"
                       + "".join(f"{c:>{cw}}" for c, cw in zip(cells, widths)))
        return "\n".join(out)

    def tables(self):
        cpl_rows, reach_rows = [], []
        for name, s in self.arms.items():
            native = f"{np.mean(s.native_cpl):.1f}" if s.native_cpl else "-"
            cpl_rows.append((name, [native] + [
                f"{np.mean(s.milestone_cpl[m]):.1f}" if s.milestone_cpl[m] else "-"
                for m in MILESTONE_SIMS]))
            reach = [f"{100.0 * np.mean(np.array(s.max_sims) > r):.0f}%"
                     if s.max_sims else "-" for r in REACH_SIMS]
            reach_rows.append((name, reach + [f"{s.sims_per_job():.0f}"]))
        return "\n\n".join([
            self.table("CPL by sims   (xc0's pick vs SF best, mean cp, within equiv = 0; "
                       "own = where its own stop rule fired; - = no job got there)",
                       ["own"] + [f"@{m}" for m in MILESTONE_SIMS], cpl_rows),
            self.table("searched past   (% of jobs)",
                       [f">{r}" for r in REACH_SIMS] + ["sims/job"], reach_rows),
        ])

    def summary_order(self):
        """base first, then by mean own esl, then mean own sims, low to high;
        arms with no own data go last."""
        def key(name):
            s = self.arms[name]
            if not s.native_esl:
                return (1, 0.0, 0.0)
            return (0, np.mean(s.native_esl), np.mean(s.native_sims))
        rest = sorted((a for a in self.arms if a != "base"), key=key)
        return (["base"] if "base" in self.arms else []) + rest

    def summary(self):
        rows = []
        for name in self.summary_order():
            s = self.arms[name]
            n = max(s.n, 1)
            rows.append((name, [
                str(s.n), str(s.own), str(s.later), str(s.never),
                str(s.doublings), f"{s.sims_per_job():.0f}",
                str(s.median_stop()),
                f"{np.mean(s.native_sims):.0f}" if s.native_sims else "-",
                f"{1000 * np.mean(s.native_esl):.1f}" if s.native_esl else "-",
                *(f"{100.0 * np.mean(np.array(s.native_esl) > t):.1f}%"
                  if s.native_esl else "-" for t in ESL_BARS),
                f"{np.mean(s.native_cpl):.1f}" if s.native_cpl else "-",
                f"{np.mean(s.cpl):.1f}" if s.cpl else "-"]))
        return self.table(
            "arm summary",
            ["jobs", "own", "later", "never", "2x", "sims/job",
             "stop sims", "own sims", "own esl",
             *(f"esl>{t}" for t in ESL_BARS), "own cpl", "final cpl"], rows)


def start_server(ctx, cfg, n_slots, n_threads):
    """One xc0 inference server on the run's current engine, with n_slots
    worker slots. Mirrors SelfplayRunner.start_xc0_server."""
    from chessbot.infer_ort_trt import selfplay_trt_paths
    from chessbot.infer_server import server_main

    trt_dir, model_name, onnx_path = selfplay_trt_paths(cfg)
    load = {"cmd": "load", "onnx": onnx_path, "model_name": model_name,
            "trt_dir": trt_dir}
    reqs = [ctx.Pipe(duplex=False) for _ in range(n_slots)]
    reps = [ctx.Pipe(duplex=False) for _ in range(n_slots)]
    ctrl_r, ctrl_w = ctx.Pipe(duplex=False)
    ready_q = ctx.Queue()
    ready_ev, release_ev = ctx.Event(), ctx.Event()
    spec = {"max_bs": cfg.macro_batch, "encoding": cfg.encoding_type,
            "n_threads": n_threads}
    p = ctx.Process(target=server_main,
                    args=(spec, load, [r for r, _ in reqs], [w for _, w in reps],
                          ctrl_r, ready_q, ready_ev, release_ev),
                    daemon=True)
    p.start()
    for r, _ in reqs:
        r.close()
    for _, w in reps:
        w.close()
    ctrl_r.close()

    descs = ready_q.get(timeout=SERVER_WAIT_S)
    slots = [{**d, "req": rq[1], "rep": rp[0]}
             for d, rq, rp in zip(descs, reqs, reps)]
    print(f"[exp] xc0 server up: {n_threads} sessions, {n_slots} slots, "
          f"{os.path.basename(onnx_path)}", flush=True)
    return p, ctrl_w, slots


def build_chunks(exp, master, done=()):
    """Job specs in chunks, one chunk per queue item. Every arm of a position
    stays in one chunk, so one worker runs them and they share that
    position's NN cache; whole positions are packed until a chunk reaches
    chunk_jobs, and the jobs inside a chunk are shuffled so a position's arms
    start staggered among other positions. Arms marked first: true get their
    own chunks, queued ahead of the rest, so they mostly meet a cold cache.
    (arm name, pos_id) pairs in done are skipped."""
    corpus = master.corpus
    if exp.get("strata"):
        corpus = corpus[corpus.stratum.isin(exp["strata"])]
    if exp.get("limit"):
        corpus = corpus.sample(n=min(exp["limit"], len(corpus)),
                               random_state=exp.get("seed", 0))
    rng = random.Random(exp.get("seed", 0))
    first = [a for a in exp["arms"] if a.get("first")]
    rest = [a for a in exp["arms"] if not a.get("first")]
    chunks = []
    n = 0
    for arms in (first, rest):
        if not arms:
            continue
        chunk = []
        for pos_id, row in corpus.iterrows():
            pos = {f: row[f] for f in POS_FIELDS}
            labels = master.labels.get(pos["short_fen"], {})
            for arm in arms:
                if (arm["name"], int(pos_id)) in done:
                    continue
                chunk.append({"job_id": n, "pos_id": int(pos_id), "pos": pos,
                              "arm": arm, "labels": dict(labels)})
                n += 1
            if len(chunk) >= exp["chunk_jobs"]:
                rng.shuffle(chunk)
                chunks.append(chunk)
                chunk = []
        if chunk:
            rng.shuffle(chunk)
            chunks.append(chunk)
    return chunks


def run_experiment(exp_path):
    """Launch an experiment from its yaml: one xc0 server, n_workers worker
    processes fed from one job queue. Each result is appended to
    results.pkl as it arrives (read back with load_results) and its labels
    are merged into the master corpus copy. With resume: <run dir name>,
    results append to that run's results.pkl and its finished
    (arm, pos_id) jobs are skipped."""
    with open(exp_path) as f:
        exp = yaml.safe_load(f)
    cfg = Config.from_yaml(exp["run_config"])
    # experiments never draw selfplay's per-game sampled params: every job
    # uses the config's fixed values plus base_overrides and its arm
    cfg.sampleable = None
    # experiment-wide settings over the run config, under every arm
    cfg.update(exp.get("base_overrides", {}))
    master = Master(exp["corpus"])
    if exp.get("resume"):
        out_dir = os.path.join(EXPERIMENTS_DIR, exp["resume"])
        done = completed_jobs(os.path.join(out_dir, "results.pkl"))
        print(f"[exp] resuming {out_dir}: {len(done)} jobs already done",
              flush=True)
    else:
        out_dir = os.path.join(EXPERIMENTS_DIR,
                               f"{exp['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir)
        done = set()
    chunks = build_chunks(exp, master, done)
    n_jobs = sum(len(c) for c in chunks)

    with open(os.path.join(out_dir, "experiment.yaml"), "w") as f:
        yaml.safe_dump(exp, f)
    results_path = os.path.join(out_dir, "results.pkl")

    ctx = mp.get_context("spawn")
    n_workers = exp["n_workers"]
    server, ctrl, slots = start_server(ctx, cfg, n_workers, exp["server_threads"])
    job_q, res_q = ctx.Queue(), ctx.Queue()
    # jobs left behind by dead workers must not block interpreter exit
    job_q.cancel_join_thread()
    for chunk in chunks:
        job_q.put(chunk)
    for _ in range(n_workers):
        job_q.put(None)

    # far more jobs live than one NN batch needs: collection stops at a full
    # batch, and the spare jobs cover ones waiting on SF labels
    workers = [ctx.Process(target=worker_main,
                           args=(w, cfg, slots[w], job_q, res_q,
                                 exp["max_live"], exp["chunk_jobs"],
                                 exp.get("sims_cap", SIMS_CAP)),
                           daemon=True)
               for w in range(n_workers)]
    for p in workers:
        p.start()
    print(f"[exp] {n_jobs} jobs in {len(chunks)} chunks -> "
          f"{n_workers} workers, results in {out_dir}",
          flush=True)

    t0 = time.time()
    n_done = n_workers_done = 0
    status = Status(n_jobs, [a["name"] for a in exp["arms"]])
    try:
        with open(results_path, "ab") as out:
            while n_workers_done < n_workers:
                try:
                    res = res_q.get(timeout=1.0)
                except queue.Empty:
                    if not any(p.is_alive() for p in workers):
                        raise RuntimeError("[exp] every worker died")
                    res = None
                if res is None:
                    pass
                elif res.get("worker_done"):
                    n_workers_done += 1
                elif "telemetry" in res:
                    status.workers[res["worker"]] = res["telemetry"]
                else:
                    pickle.dump(res, out, protocol=pickle.HIGHEST_PROTOCOL)
                    out.flush()
                    master.merge(res["labels_key"], res["labels"])
                    status.add(res)
                    n_done += 1
                    if n_done % SYNC_EVERY == 0:
                        master.save()
                    if n_done % STATUS_EVERY_JOBS == 0:
                        print("\n" + status.line(time.time() - t0,
                                                 master.n_best_moved), flush=True)
                    if n_done % SUMMARY_EVERY_JOBS == 0:
                        print("\n" + status.summary(), flush=True)
        print("\n" + status.line(time.time() - t0, master.n_best_moved),
              flush=True)
        print("\n" + status.summary(), flush=True)
    finally:
        master.save()
        ctrl.send({"cmd": "stop"})
        server.join(timeout=30)
        for p in workers:
            p.join(timeout=10)
    print(f"[exp] done: {n_done} jobs in {time.time() - t0:.0f}s, "
          f"best moved on {master.n_best_moved} positions", flush=True)
    return out_dir


class Unpickler(pickle.Unpickler):
    """Reads SfEvals pickled by a launcher that ran as __main__."""

    def find_class(self, module, name):
        if name == "SfEval":
            return SfEval
        return super().find_class(module, name)


def completed_jobs(path):
    """(arm, pos_id) of every result in a results.pkl. A partial record left
    by a crash mid-write is cut off the end so later appends stay readable."""
    done, good = set(), 0
    with open(path, "rb") as f:
        while True:
            try:
                res = Unpickler(f).load()
            except (EOFError, pickle.UnpicklingError):
                break
            done.add((res["arm"], res["pos_id"]))
            good = f.tell()
    if good < os.path.getsize(path):
        with open(path, "r+b") as f:
            f.truncate(good)
    return done


def load_results(path):
    """Every result appended to a results.pkl."""
    out = []
    with open(path, "rb") as f:
        while True:
            try:
                out.append(Unpickler(f).load())
            except EOFError:
                return out


if __name__ == "__main__":
    # os._exit: a crashed worker leaves queue feeder threads that would
    # otherwise keep the launcher alive after the run ends
    # run through the importable module, not __main__, so pickled SfEvals
    # name chessbot.experimentation.runner and load anywhere
    from chessbot.experimentation import runner
    code = 1
    try:
        runner.run_experiment(sys.argv[1])
        code = 0
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
