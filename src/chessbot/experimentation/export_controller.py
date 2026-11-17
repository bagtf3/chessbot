"""Export a trained search controller for pyfastchess's C++ runtime.

Writes <checkpoint stem>.ctrl/ next to the checkpoint:
  init.onnx   root context (1,48) -> p, FiLM scale and beta per conv block
  step.onnx   one check-in (1,64) plus the running state -> head logits and
              the next state (the cached causal step graph of
              search_controller_model.md)
  meta.json   shapes, input / output names, heads, feature version and
              feature names, which the C++ loader validates
  parity.npz  a few test trajectories with the PyTorch logits, for the C++
              parity test

The export checks itself: init + step through onnxruntime must match the
PyTorch full causal forward on PARITY_TRAJ test trajectories.

python -m chessbot.experimentation.export_controller <checkpoint.pt>
"""
import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chessbot.experimentation import train_controller as T
from chessbot.experimentation.collector import COLLECTIONS_DIR

OPSET = 17
CHECK_EVERY = 100
PARITY_TRAJ = 200
FIXTURE_TRAJ = 20
PARITY_TOL = 1e-4


def folded_linear(lin, scale):
    """lin(x * s + b) as one Linear: W' = W * s, c' = W @ b + c."""
    out = nn.Linear(lin.in_features, lin.out_features)
    with torch.no_grad():
        out.weight.copy_(lin.weight * scale.scale.unsqueeze(0))
        out.bias.copy_(lin.weight @ scale.bias + lin.bias)
    return out


class InitGraph(nn.Module):
    """context (1,48) -> p (1,128), film_scale (6,128), film_beta (6,128)."""

    def __init__(self, m):
        super().__init__()
        self.first = folded_linear(m.position_proj[0], m.context_scale)
        self.mixer = m.position_proj[2]
        self.film = m.film_proj
        self.n_blocks = len(m.blocks)

    def forward(self, context):
        p = self.mixer(F.silu(self.first(context)))
        film = self.film(p).view(self.n_blocks, 2, T.WIDTH)
        return p, 1.0 + film[:, 0], film[:, 1]


class StepGraph(nn.Module):
    """One check-in through the causal blocks. Each block's cache holds its
    last (k-1)*d inputs, oldest first; zeros stand in for the left padding
    of the full model."""

    def __init__(self, m):
        super().__init__()
        self.inp = folded_linear(m.input_proj, m.checkin_scale)
        self.k = m.blocks[0].k
        self.dil = [b.d for b in m.blocks]
        # conv weight (out, in, k) -> (k*in, out): row j*in + i is tap j, input i
        self.conv_w = nn.ParameterList(
            [nn.Parameter(b.conv.weight.detach().permute(2, 1, 0).reshape(-1, T.WIDTH).clone())
             for b in m.blocks])
        self.conv_b = nn.ParameterList([nn.Parameter(b.conv.bias.detach().clone())
                                        for b in m.blocks])
        self.norms = nn.ModuleList([b.norm for b in m.blocks])
        self.mixers = m.mixers
        self.output_proj, self.stab, self.qual = m.output_proj, m.stability_head, m.quality_head

    def forward(self, checkin, p, film_scale, film_beta, run_sum, count, *caches):
        h = self.inp(checkin)
        new_caches = []
        for i, d in enumerate(self.dil):
            window = torch.cat([caches[i], h], 0)
            taps = window[::d].reshape(1, -1)
            y = taps @ self.conv_w[i] + self.conv_b[i]
            h = h + F.silu(self.norms[i](y) * film_scale[i:i + 1] + film_beta[i:i + 1])
            new_caches.append(window[1:])
            if str(i) in self.mixers:
                h = self.mixers[str(i)](h)
        run_sum = run_sum + h
        count = count + 1.0
        out = F.silu(self.output_proj(torch.cat([h, run_sum / count, p], 1)))
        logits = torch.cat([self.stab(out), self.qual(out)], 1)
        return (logits, run_sum, count, *new_caches)


def export(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model = T.Controller(**ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    init_g, step_g = InitGraph(model).eval(), StepGraph(model).eval()
    k, dil = step_g.k, step_g.dil
    cache_rows = [(k - 1) * d for d in dil]
    out_dir = os.path.splitext(ckpt_path)[0] + ".ctrl"
    os.makedirs(out_dir, exist_ok=True)

    cache_in = [f"cache{i}" for i in range(len(dil))]
    cache_out = [f"cache{i}_out" for i in range(len(dil))]
    step_inputs = ["checkin", "p", "film_scale", "film_beta", "sum", "count"] + cache_in
    step_outputs = ["logits", "sum_out", "count_out"] + cache_out
    with torch.no_grad():
        torch.onnx.export(init_g, (torch.zeros(1, T.CONTEXT_DIM),),
                          os.path.join(out_dir, "init.onnx"), opset_version=OPSET,
                          input_names=["context"], output_names=["p", "film_scale", "film_beta"],
                          dynamo=False)
        p, fs, fb = init_g(torch.zeros(1, T.CONTEXT_DIM))
        example = (torch.zeros(1, T.CHECKIN_DIM), p, fs, fb, torch.zeros(1, T.WIDTH),
                   torch.zeros(1, 1), *[torch.zeros(r, T.WIDTH) for r in cache_rows])
        torch.onnx.export(step_g, example, os.path.join(out_dir, "step.onnx"),
                          opset_version=OPSET, input_names=step_inputs,
                          output_names=step_outputs, dynamo=False)

    meta_coll = ckpt["collection_metadata"][0]
    heads = ckpt.get("training_config", {}).get("heads", "both")
    meta = {"source_checkpoint": os.path.abspath(ckpt_path), "spec": ckpt["spec"],
            "width": T.WIDTH, "k": k, "dilations": dil, "cache_rows": cache_rows,
            "n_blocks": len(dil), "heads": "two_head" if heads == "both" else "quality_only",
            "feature_version": meta_coll["feature_version"], "check_every": CHECK_EVERY,
            "context_dim": T.CONTEXT_DIM, "checkin_dim": T.CHECKIN_DIM,
            "context_names": meta_coll["context_feature_names"],
            "checkin_names": meta_coll["checkin_feature_names"],
            "init_inputs": ["context"], "init_outputs": ["p", "film_scale", "film_beta"],
            "step_inputs": step_inputs, "step_outputs": step_outputs}

    diff, fixture = parity(model, out_dir, ckpt, cache_rows)
    meta["parity_max_abs_diff"] = diff
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    np.savez(os.path.join(out_dir, "parity.npz"), **fixture)
    print(f"[export] {out_dir}: parity max |logit diff| {diff:.2e} over {PARITY_TRAJ} test "
          f"trajectories (tolerance {PARITY_TOL:g})")
    if not diff < PARITY_TOL:
        raise SystemExit(f"[export] parity failed: {diff:.2e} >= {PARITY_TOL:g}")
    return out_dir


def run_cached(init_sess, step_sess, context, checkins, cache_rows):
    """Logits per check-in from the exported graphs, stepping like C++."""
    p, fs, fb = init_sess.run(None, {"context": context[None].astype(np.float32)})
    state = {"sum": np.zeros((1, T.WIDTH), np.float32), "count": np.zeros((1, 1), np.float32),
             **{f"cache{i}": np.zeros((r, T.WIDTH), np.float32) for i, r in enumerate(cache_rows)}}
    out = []
    for ck in checkins:
        res = step_sess.run(None, {"checkin": ck[None].astype(np.float32), "p": p,
                                   "film_scale": fs, "film_beta": fb, **state})
        out.append(res[0][0])
        state["sum"], state["count"] = res[1], res[2]
        for i in range(len(cache_rows)):
            state[f"cache{i}"] = res[3 + i]
    return np.array(out)


def parity(model, out_dir, ckpt, cache_rows):
    """Max |logit diff| between the exported cached graphs and the PyTorch
    full causal forward on test trajectories, plus a small fixture."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    init_sess = ort.InferenceSession(os.path.join(out_dir, "init.onnx"), so,
                                     providers=["CPUExecutionProvider"])
    step_sess = ort.InferenceSession(os.path.join(out_dir, "step.onnx"), so,
                                     providers=["CPUExecutionProvider"])
    split = ckpt["split"]
    test_ids = set(split["test"])
    data = []
    for rel in split["held_out_shards"]:
        data += [d for d in T.read_shard(os.path.join(COLLECTIONS_DIR, rel))
                 if d["trajectory_id"] in test_ids]
        if len(data) >= PARITY_TRAJ:
            break
    data = data[:PARITY_TRAJ]
    worst, ctxs, cks, logits = 0.0, [], [], []
    with torch.no_grad():
        for d in data:
            ref = model(torch.from_numpy(d["context"])[None],
                        torch.from_numpy(d["checkins"])[None])[0].numpy()
            got = run_cached(init_sess, step_sess, d["context"], d["checkins"], cache_rows)
            worst = max(worst, float(np.abs(got - ref).max()))
            if len(ctxs) < FIXTURE_TRAJ:
                ctxs.append(d["context"]); cks.append(d["checkins"]); logits.append(ref)
    n = max(len(c) for c in cks)
    fixture = {"context": np.stack(ctxs).astype(np.float32),
               "lengths": np.array([len(c) for c in cks], np.int32),
               "checkins": np.stack([np.pad(c, ((0, n - len(c)), (0, 0))) for c in cks]).astype(np.float32),
               "logits": np.stack([np.pad(l, ((0, n - len(l)), (0, 0))) for l in logits]).astype(np.float32)}
    return worst, fixture


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    export(ap.parse_args().checkpoint)
