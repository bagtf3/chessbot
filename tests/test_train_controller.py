import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

# During review this file is next to the staged trainer; in the repository
# import the installed package under test.
if (Path(__file__).parent / "train_controller.py").exists():
    spec = importlib.util.spec_from_file_location("review_trainer", Path(__file__).parent / "train_controller.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
else:
    from chessbot.experimentation import train_controller as T


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def item(n, label=0, tid="t"):
    return {"trajectory_id": tid, "split_group": tid,
            "context": np.zeros(48, np.float32), "checkins": np.zeros((n, 64), np.float32),
            "targets": np.full((n, 2), label, np.float32), "valid": np.ones((n, 2), bool),
            "root_visits": np.arange(1, n+1)*100, "checkpoint_move": ["e2e4"]*n,
            "min_decision": 0, "teacher": {}, "losses": [None] * n, "xc0_cap": 400,
            "xc0_k": n - 1, "xc0_stop": 400, "xc0_locked": False,
            "xc0_set": {"e2e4"}, "sf_set": {"e2e4"}}


def test_loss_equal_trajectories_and_padding():
    logits = torch.tensor([[[0., 0.]]*4, [[2., 2.]]*4], requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[0, 1:] = False
    expected = (F.softplus(torch.tensor(0.)) + F.softplus(torch.tensor(2.)))/2
    loss = T.masked_bce(logits, torch.zeros_like(logits), valid).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0
    long = logits.detach().clone()
    long[0, 1:] = -100
    torch.testing.assert_close(T.masked_bce(long, torch.zeros_like(long), valid).sum(), expected)


def test_missing_heads_and_unlabelled_trajectory():
    logits = torch.tensor([[[0., 50.]], [[2., 2.]], [[-20., -20.]]])
    valid = torch.tensor([[[True, False]], [[True, True]], [[False, False]]])
    expected = (F.softplus(torch.tensor(0.)) + F.softplus(torch.tensor(2.)))/2
    torch.testing.assert_close(T.masked_bce(logits, torch.zeros_like(logits), valid).sum(), expected)


def test_prior_is_trajectory_weighted():
    assert T.pos_rate([item(1, 0), item(20, 1)]) == [.5, .5]


class Constant(torch.nn.Module):
    def forward(self, ctx, ck):
        return torch.zeros((*ck.shape[:2], 2), device=ck.device)


def test_evaluation_matches_loss_and_batch_size():
    data = [item(1, 0), item(13, 1), item(4, 0)]
    a = T.evaluate(Constant(), data, 1, "cpu", [.5, .5])
    b = T.evaluate(Constant(), data, 3, "cpu", [.5, .5])
    assert a["loss"] == pytest.approx(np.log(2))
    assert a["loss"] == pytest.approx(a["baseline_loss"])
    assert a["loss"] == pytest.approx(b["loss"])
    assert a["heads"]["quality"]["trajectory_accuracy"] == pytest.approx(2/3)


def test_causal_prefix_and_right_padding():
    torch.manual_seed(1)
    m = T.Controller(**T.SPECS["M"]).eval()
    with torch.no_grad():
        m.film_proj.weight.normal_(std=.03)
        ctx, ck = torch.randn(2, 48), torch.randn(2, 10, 64)
        full = m(ctx, ck)
        prefix = m(ctx, ck[:, :4])
        changed = ck.clone(); changed[:, 4:] += 100
        torch.testing.assert_close(full[:, :4], prefix, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(full[:, :4], m(ctx, changed)[:, :4], atol=2e-6, rtol=2e-5)
        padded = F.pad(ck[:, :4], (0, 0, 0, 6))
        torch.testing.assert_close(prefix, m(ctx, padded)[:, :4], atol=2e-6, rtol=2e-5)


def test_buckets_use_every_trajectory_once():
    data = [item(n) for n in range(1, 42)]
    groups = T.batch_indices(data, 4, np.random.default_rng(0))
    assert sorted(j for group in groups for j in group) == list(range(len(data)))
    assert all(len(group) <= 4 for group in groups)
    assert sum(max(len(data[j]["checkins"]) for j in g)*len(g) for g in groups) < 41*41


def test_fire_replay_against_xc0_move():
    from chessbot.experimentation.runner import SfEval
    ev = lambda cp: SfEval(cp_stm=cp, mate_stm=None, depth=16, source="sf")
    a, b = item(4), item(3)
    a["checkpoint_move"] = ["a", "b", "c", "d"]
    a["losses"] = T.checkin_losses({"baseline": ev(50), "evals": {"b": ev(0), "d": ev(50)}},
                                   a["checkpoint_move"])
    a["xc0_set"], a["sf_set"] = {"d"}, {"b", "d"}
    preds = [(a, np.array([[.1, .1], [.95, .95], [.5, .5], [1, 1]])), (b, np.zeros((3, 2)))]
    early = T.fire_replay(preds, .9)
    tsl = 1000 * .5 * (T.sf_value(ev(50)) - T.sf_value(ev(0)))
    assert early["stop"] == .5 and early["sims"] == 200 and early["saved"] == 200
    assert early["ext"] == 0 and early["total"] == 300
    assert early["tsl_tot"] == pytest.approx(tsl) and early["cpl_tot"] == 50
    assert early["unfire"] == 1 and early["xc0"] == 0 and early["sf"] == 1
    assert early["tsl"] == pytest.approx(tsl) and early["dtsl"] == pytest.approx(tsl)
    assert early["cpl"] == 50 and early["dcpl"] == 50
    assert early["d50"] == float(tsl > 50)
    late = T.fire_replay(preds, .99)
    assert late["sims"] == 400 and late["saved"] == 0
    assert late["unfire"] == 0 and late["xc0"] == 1 and late["dtsl"] == 0
    assert late["total"] == 400 and late["tsl_tot"] == 0
    ref = T.xc0_reference(preds)
    assert ref["stop"] == 0 and ref["sims"] is None and ref["total"] == 400


def record(tid, game):
    d = item(3, tid=tid)
    d["schema_version"] = T.SCHEMA_VERSION
    d["target_valid"] = d.pop("valid")
    d["position"] = {"run_tag": "run", "game_id": game}
    d["position_group_id"] = tid
    d["search"] = {"min_decision": 0, "initial_ceiling": 400}
    d["stages"] = [{"reference": {"checkin": 2, "how": "ceiling", "xc0_set": ["e2e4"]}}]
    return d


def test_load_collection_and_validation(monkeypatch):
    rows = [record(f"g{i}", f"g{i}") for i in range(6)]
    monkeypatch.setattr(T, "iter_records", lambda _: iter(rows))
    data = T.load_collection("coll", metadata={})
    assert len(data) == 6 and all(d["collection"] == "coll" for d in data)
    assert "teacher" not in data[0] and len(data[0]["losses"]) == 3
    rows[0]["checkins"][0,0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        T.load_collection("coll", metadata={})


def test_incompatible_metadata(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"schema_version": -1}))
    with pytest.raises(ValueError, match="schema_version"):
        T.load_metadata(tmp_path)


def test_hold_out_draws_from_whole_shards(tmp_path):
    import pickle
    coll = tmp_path / "coll"
    coll.mkdir()
    files = []
    for s in range(4):
        path = coll / f"trajectories_w0_{s:05d}.pkl"
        with open(path, "wb") as f:
            for i in range(10):
                pickle.dump(record(f"s{s}:{i}", f"s{s}"), f)
        files.append(str(path))
    (val, test), spare, held, per_traj = T.hold_out(files, (8, 6), 2,
                                                    np.random.default_rng(0), 1)
    assert len(val) == 8 and len(test) == 6 and len(held) == 2 and len(spare) == 4
    ids = [d["trajectory_id"] for d in val + test + spare]
    assert len(set(ids)) == 20
    assert {i.split(":")[0] for i in ids} == {f"s{int(p[-9:-4])}" for p in held}
    assert per_traj > 0
