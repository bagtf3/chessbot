"""Search-controller collector: decision rules, continuation draws, targets,
output, and exact check-ins on real trees driven by a fake NN and teacher."""
import hashlib
import os
import pickle

import numpy as np
import pytest
import pyfastchess
from pyfastchess import MCTSForest, raw_cache_bulk_insert_np

from chessbot.config import XC0H_K, Config
from chessbot.experimentation import collector as C
from chessbot.experimentation.runner import SfEval

PROBS = {"full": 0.25, "stop": 0.25, "random": 0.5}
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"


class FixedRng:
    def __init__(self, *values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0)


def test_dominance_strict_boundary():
    eq = C.dominance([600, 300, 0], 901, 1201, 800)
    assert eq["gap"] == eq["remaining"] == 300 and not eq["locked"]
    assert C.dominance([601, 300, 0], 902, 1201, 800)["locked"]


def test_dominance_needs_more_than_min_decision():
    assert not C.dominance([800, 0], 800, 801, 800)["locked"]
    assert C.dominance([801, 0], 801, 801, 800)["locked"]


def test_dominance_unvisited_runner_up():
    p = C.dominance([500, 0, 0], 501, 1000, 400)
    assert p["runner_up_n"] == 0 and p["gap"] == 500 and p["locked"]


def test_dominance_single_visited_child_of_many():
    assert not C.dominance([300, 0, 0, 0], 301, 700, 200)["locked"]


def test_continuation_branches():
    assert C.sample_continuation(FixedRng(0.1, 0.9), 5, PROBS) == ("full", "full", 5)
    assert C.sample_continuation(FixedRng(0.3, 0.9), 5, PROBS) == ("stop", "stop", 0)
    drawn, eff, k = C.sample_continuation(FixedRng(0.7, 0.0), 5, PROBS)
    assert (drawn, eff, k) == ("random", "random", 1)
    assert C.sample_continuation(FixedRng(0.7, 0.999), 5, PROBS)[2] == 4


def test_continuation_short_remainders():
    assert C.sample_continuation(FixedRng(0.1, 0.5), 0, PROBS) == ("full", "stop", 0)
    assert C.sample_continuation(FixedRng(0.7, 0.2), 1, PROBS) == ("random", "stop", 0)
    assert C.sample_continuation(FixedRng(0.7, 0.8), 1, PROBS) == ("random", "full", 1)
    assert C.sample_continuation(FixedRng(0.7, 0.99), 2, PROBS) == ("random", "random", 1)


def test_random_endpoints_strictly_intermediate():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(2000):
        drawn, eff, k = C.sample_continuation(rng, 6, PROBS)
        if eff == "random":
            seen.add(k)
    assert seen == {1, 2, 3, 4, 5}


def test_trajectory_rng_reproducible():
    a = C.trajectory_rng(1, "c26k_close:42").random(4)
    b = C.trajectory_rng(1, "c26k_close:42").random(4)
    c = C.trajectory_rng(1, "c26k_close:43").random(4)
    d = C.trajectory_rng(2, "c26k_close:42").random(4)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, d)


def test_xc0_equiv_set():
    stats = {"a": (500, 498, 0.30), "b": (490, 489, 0.31), "c": (495, 490, 0.20),
             "d": (495, 0, 0.30), "e": (499, 499, 0.30), "f": (489, 489, 0.30)}
    out = C.xc0_equiv_set("a", ["b", "c", "d", "f", "a"], stats, 0.02, 0.02)
    # b: 10 visits = 2% of the winner's 500; c: Q too far; d: no completed
    # backups; e: not a recent leader; f: 11 visits off, past 2%
    assert out == ["a", "b"]


def ev(cp):
    return SfEval(cp_stm=cp, mate_stm=None, depth=17, source="sf")


def test_expert_set_and_stable_but_bad():
    base = ("e2e4", ev(30))
    expert = C.expert_set(base, {"e2e4": ev(25), "d2d4": ev(28), "a2a3": ev(-80)})
    assert expert == {"e2e4", "d2d4"}
    targets, valid = C.build_targets(["a2a3", "d2d4"], [0, 0], [["a2a3"]], expert,
                                     {"a2a3", "d2d4"})
    assert targets.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert valid.all()


def test_missing_teacher_labels_are_masked_not_negative():
    targets, valid = C.build_targets(["a2a3", "b2b3"], [0, 0], [["a2a3"]], set(),
                                     {"a2a3"})
    assert valid[:, 0].all()
    assert valid[:, 1].tolist() == [True, False]
    assert targets[1, 1] == 0.0


def test_output_shards_and_resume(tmp_path):
    out = C.Output(str(tmp_path), 2, "w0")
    for i in range(5):
        out.write({"trajectory_id": f"t{i}", "x": np.arange(i)})
    paths = C.shard_paths(str(tmp_path))
    assert len(paths) == 3
    with open(paths[-1], "ab") as f:
        f.write(b"\x80\x05partial")
    assert C.done_trajectory_ids(str(tmp_path)) == {f"t{i}" for i in range(5)}
    again = C.Output(str(tmp_path), 2, "w0")
    again.write({"trajectory_id": "t5", "x": np.arange(1)})
    recs = C.load_records(str(tmp_path))
    assert [r["trajectory_id"] for r in recs] == [f"t{i}" for i in range(6)]


class FakeTeacher:
    """Evaluates after `delay` drains: the baseline is a move no candidate
    plays at cp 0, so a candidate is approved exactly when good(uci)."""

    def __init__(self, good=lambda uci: True, delay=0):
        self.good = good
        self.delay = delay
        self.tick = 0
        self.pending = {}
        self.evals = {}
        self.n_evals = 0

    def backlog(self):
        return len(self.pending)

    def cached_depth(self, key):
        return None

    def request(self, key, board, uci, depth, urgent=False):
        k = (key, depth, uci)
        if k not in self.evals and k not in self.pending:
            self.pending[k] = self.tick + self.delay

    def flush(self, key, depth):
        pass

    def seed_corpus(self, key, pos, min_depth):
        pass

    def drain(self):
        self.tick += 1
        for k, at in list(self.pending.items()):
            if at < self.tick:
                key, depth, uci = k
                self.evals[k] = (("a1a1", ev(0)) if uci is None
                                 else ev(0 if self.good(uci) else -300))
                del self.pending[k]

    def baseline(self, key, depth):
        return self.evals.get((key, depth, None))

    def move(self, key, uci, depth):
        return self.evals.get((key, depth, uci))


def fake_nn(forest):
    """Deterministic per position: the same leaf always gets the same output."""
    keys, _ = forest.get_all_history_tokens(XC0H_K)
    if not len(keys):
        return
    wdl = np.zeros((len(keys), 3), np.float32)
    pol = np.zeros((len(keys), 1858), np.float32)
    for i, k in enumerate(keys):
        r = np.random.default_rng(int(k))
        wdl[i] = r.dirichlet([2, 2, 2])
        pol[i] = r.standard_normal(1858) * 1.5
    raw_cache_bulk_insert_np(keys, wdl, pol)


def coll(**kw):
    out = dict(C.DEFAULTS)
    out.update({"initial_budget": 1000, "min_decision_visits": 400, "global_cap": 4000})
    out.update(kw)
    return out


def spec(fen, tid="t0"):
    pos = {f: None for f in C.POS_FIELDS}
    pos.update({"init_fen": fen, "uci_path": [], "fen": fen,
                "short_fen": " ".join(fen.split()[:4])})
    return {"trajectory_id": tid, "pos": pos, "position_group_id": "g:0",
            "effort_bin": "0|[0.0,0.1)|[0.0,0.3)", "position_selection_id": "close"}


def run(trajs, teacher, forest, mbs=16, loops=200000):
    counts = {"cached": 0, "terminal": 0}
    for _ in range(loops):
        if all(t.state == "finished" for t in trajs):
            return counts
        teacher.drain()
        for t in trajs:
            t.poll()
        forest.resolve_all_inflight()
        for t in trajs:
            res = t.step(mbs)
            if res is not None:
                counts["cached"] += res.count_cached
                counts["terminal"] += res.count_terminal
        fake_nn(forest)
    raise AssertionError("trajectories did not finish")


@pytest.fixture
def cfg():
    pyfastchess.raw_cache_clear()
    pyfastchess.priors_cache_clear()
    c = Config()
    c.update({"dirichlet_eps": 0.0})
    return c


def check_rows(rec, interval=100):
    n = rec["root_visits"]
    assert np.array_equal(n, interval * np.arange(1, len(n) + 1))
    t = len(n)
    for name in ("checkins", "completed_backups", "stage_id", "child_n",
                 "child_n_done", "child_q", "targets", "target_valid"):
        assert len(rec[name]) == t, name
    assert rec["checkins"].shape == (t, 64) and rec["context"].shape == (48,)
    assert np.allclose(rec["checkins"][:, 0] * 40, np.arange(1, t + 1))


def test_exact_checkins_through_terminal_runs(cfg):
    teacher, forest = FakeTeacher(), MCTSForest()
    tr = C.Trajectory(spec(MATE_IN_ONE), cfg, coll(), teacher, forest)
    counts = run([tr], teacher, forest)
    rec = tr.record()
    assert counts["terminal"] > 0
    check_rows(rec)


def test_exact_checkins_through_cache_hits(cfg):
    teacher, forest = FakeTeacher(), MCTSForest()
    first = C.Trajectory(spec(START, "a"), cfg, coll(), teacher, forest)
    run([first], teacher, forest)
    second = C.Trajectory(spec(START, "b"), cfg, coll(), teacher, forest)
    counts = run([second], teacher, forest)
    assert counts["cached"] > 0
    check_rows(second.record())


def test_leader_matches_tree_best(cfg):
    teacher, forest = FakeTeacher(), MCTSForest()
    tr = C.Trajectory(spec(START), cfg, coll(), teacher, forest)
    leaders = []
    orig = tr.on_checkin

    def spy(root_n):
        details = tr.tree.root_child_details()
        ns = [d.N for d in details]
        assert ns == sorted(ns, reverse=True)
        # the C++ most-visited pick; the Python wrapper's best() samples
        leaders.append((details[0].uci, pyfastchess.MCTSTree.best(tr.tree)[0]))
        orig(root_n)
    tr.on_checkin = spy
    run([tr], teacher, forest)
    assert leaders and all(a == b for a, b in leaders)


def test_disapproval_extends_stages_to_global_cap(cfg):
    teacher, forest = FakeTeacher(good=lambda uci: False), MCTSForest()
    tr = C.Trajectory(spec(START), cfg,
                      coll(initial_budget=300, min_decision_visits=100, global_cap=1200),
                      teacher, forest)
    run([tr], teacher, forest)
    rec = tr.record()
    check_rows(rec)
    assert rec["end_reason"] == "global_cap" and rec["final_sims"] == 1200
    assert [s["ceiling"] for s in rec["stages"]] == [300, 600, 1200]
    assert [s["start_sims"] for s in rec["stages"]] == [100, 400, 700]
    for s in rec["stages"]:
        ref = s["reference"]
        assert s["approved"] is False and ref["sims"] <= s["ceiling"]
        assert rec["checkpoint_move"][ref["checkin"]] == ref["winner"]
        assert rec["root_visits"][ref["checkin"]] == ref["sims"]
    assert list(rec["root_visits"]).count(300) == 1
    assert list(rec["root_visits"]).count(600) == 1
    for sid, s in enumerate(rec["stages"]):
        rows = rec["stage_id"] == sid
        assert rec["root_visits"][rows].min() == s["start_sims"]
        assert rec["root_visits"][rows].max() == s["ceiling"]
    assert rec["continuation"] is None


def test_teacher_latency_does_not_change_the_draw(cfg):
    ends = []
    for delay in (0, 40):
        pyfastchess.raw_cache_clear()
        pyfastchess.priors_cache_clear()
        teacher, forest = FakeTeacher(delay=delay), MCTSForest()
        tr = C.Trajectory(spec(START, "same"), cfg, coll(), teacher, forest)
        run([tr], teacher, forest)
        rec = tr.record()
        ends.append((rec["continuation"], rec["final_sims"], list(rec["root_visits"])))
    assert ends[0] == ends[1]


def test_record_round_trips(cfg, tmp_path):
    teacher, forest = FakeTeacher(good=lambda uci: uci.startswith("e")), MCTSForest()
    tr = C.Trajectory(spec(START), cfg, coll(), teacher, forest)
    run([tr], teacher, forest)
    out = C.Output(str(tmp_path), 10, "w0")
    out.write(tr.record())
    rec = C.load_records(str(tmp_path))[0]
    check_rows(rec)
    assert rec["target_valid"].all()
    q = rec["targets"][:, 1]
    assert np.array_equal(q, [float(m.startswith("e")) for m in rec["checkpoint_move"]])


def test_teacher_depth_hysteresis():
    c = dict(C.DEFAULTS)
    step = C.next_teacher_depth
    assert step(17, 5, c) == 18
    assert step(18, 30, c) == 18
    assert step(18, 60, c) == 17
    assert step(17, 60, c) == 17
    assert step(16, 60, c) == 16
    assert step(17, 150, c) == 16
    assert step(16, 30, c) == 17
    assert step(16, 50, c) == 16
    assert step(17, 5, c) == 18
    assert step(16, 5, c) == 18


def test_sf_cache_keeps_deepest(tmp_path):
    cache = {}
    k = ("fen", "e2e4")
    assert C.keep_deepest(cache, k, SfEval(30, None, 17, "sf"))
    assert not C.keep_deepest(cache, k, SfEval(10, None, 16, "sf"))
    assert not C.keep_deepest(cache, k, SfEval(10, None, 17, "sf"))
    assert C.keep_deepest(cache, k, SfEval(25, None, 18, "sf"))
    base = ("e2e4", SfEval(40, None, 17, "sf"))
    assert C.keep_deepest(cache, ("fen", None), base) and C.eval_depth(base) == 17
    log = tmp_path / "sf_evals_w0.pkl"
    with open(log, "wb") as f:
        pickle.dump((k, SfEval(5, None, 19, "sf")), f)
        pickle.dump((k, SfEval(6, None, 16, "sf")), f)
        f.write(b"\x80\x05partial")
    C.read_eval_log(str(log), cache)
    assert cache[k].depth == 19 and cache[k].cp_stm == 5
    assert cache[("fen", None)] == base
