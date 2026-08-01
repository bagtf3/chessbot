"""Fixed-work MCTS bench: pin the sims, measure the time.

Deliberately barebones. Every position runs an exact number of sims with
early stopping, noise and move sampling disabled, so a code change moves the
wall clock and nothing else. Rate-over-a-mixed-workload benches were tried
first and thrown out: adaptive early stopping makes sim count a function of
float results, which confounds exactly the A/B you want.

YAGNI. Add measurement gear when a decision actually needs it, not before.
"""

import argparse
import time

import numpy as np
import psutil

from pyfastchess import Board as fastboard
from pyfastchess import MCTSForest, raw_cache_bulk_insert_np
from pyfastchess import priors_cache_clear, raw_cache_clear

from chessbot.config import Config
from chessbot.mcts_utils import MCTSTree

from . import paths
from .fixtures import load_fixture_set

POLICY_DIM = 1858
U64 = np.uint64
COL_STRIDE = U64(0x9E3779B97F4A7C15)


def hash_mix(x):
    x = x ^ (x >> U64(33))
    x = x * U64(0xFF51AFD7ED558CCD)
    x = x ^ (x >> U64(33))
    x = x * U64(0xC4CEB9FE1A85EC53)
    x = x ^ (x >> U64(33))
    return x


def hash_floats(keys_np, n_cols, salt=0):
    keys = (keys_np.astype(U64) ^ U64(salt))[:, None]
    cols = (np.arange(n_cols, dtype=U64) * COL_STRIDE)[None, :]
    h = hash_mix(keys ^ cols)
    return (h >> U64(11)).astype(np.float64) / float(1 << 53)


POOL_SIZE = 4096
POOL_MASK = U64(POOL_SIZE - 1)


def build_pool(seed=0):
    """Precomputed eval pool, indexed by zobrist.

    Hashing a full 1858-wide policy per key cost ~650us/call and ate ~44% of
    bench wall time -- it swamped the C++ search we are trying to measure.
    A pool lookup is a gather instead, ~30x cheaper, and still keyed purely
    off the zobrist so evals stay deterministic per position.
    """
    rng = np.random.default_rng(seed)
    policy = rng.uniform(-3.0, 3.0, size=(POOL_SIZE, POLICY_DIM)).astype(np.float32)
    wdl = rng.uniform(0.0, 1.0, size=(POOL_SIZE, 3)).astype(np.float32) + 1e-6
    wdl /= wdl.sum(axis=1, keepdims=True)
    return policy, wdl


POOL_POLICY, POOL_WDL = build_pool()


def stub_infer(keys_np, enc_np):
    """Deterministic fake eval derived only from the position zobrist."""
    idx = (keys_np.astype(U64) & POOL_MASK).astype(np.intp)
    return POOL_POLICY[idx], POOL_WDL[idx]


def build_real_infer(cfg):
    if cfg.inference_backend == "ort_trt":
        from chessbot.infer_ort_trt import make_ort_trt_infer
        _, infer = make_ort_trt_infer(
            cfg.model_path, cfg.trt_model_name, cfg.trt_cache,
            cfg.macro_batch, encoding_type=cfg.encoding_type)
    elif cfg.inference_backend == "pt_eager":
        import torch
        from chessbot.train_pytorch import make_pt_infer
        model = torch.jit.load(cfg.model_path, map_location="cpu")
        _, infer = make_pt_infer(
            model, max_bs=cfg.macro_batch, encoding_type=cfg.encoding_type)
    else:
        raise ValueError(f"unsupported backend: {cfg.inference_backend!r}")

    def infer_fn(keys_np, enc_np):
        return infer((enc_np,))
    return infer_fn


def bench_config(base=None, reuse_tree=False):
    """Config with everything nondeterministic switched off."""
    cfg = base or Config()
    cfg.add_root_noise = False
    cfg.dirichlet_eps = 0.0
    cfg.sample_moves = False
    cfg.reuse_tree = reuse_tree
    return cfg


def search(tree, forest, cfg, infer_fn, n_sims, batch, max_fastpath=1024):
    """Run exactly n_sims (or until the tree stops producing). Returns stats."""
    sims = 0
    puct = 0
    new_nodes = 0
    pruned = 0
    t0 = time.perf_counter()

    while sims < n_sims:
        forest.resolve_all_inflight()
        res = tree.collect_many_leaves(batch, max_fastpath)

        got = res.count_new + res.count_cached + res.count_terminal
        if got == 0:
            break

        sims += got
        puct += res.total_puct
        new_nodes += res.count_new
        pruned += res.total_pruned
        tree.sims_completed_this_move += got

        keys_np, enc_np = forest.get_all_history_tokens(cfg.history_K)
        for i in range(0, len(keys_np), cfg.macro_batch):
            k = keys_np[i:i + cfg.macro_batch]
            policy, wdl = infer_fn(k, enc_np[i:i + cfg.macro_batch])
            raw_cache_bulk_insert_np(k, wdl, policy)

    return {
        "wall_s": time.perf_counter() - t0,
        "sims": sims,
        "puct": puct,
        "new_nodes": new_nodes,
        "pruned": pruned,
    }


def run_position(cfg, fen, n_sims, infer_fn, batch, rebuild=False):
    priors_cache_clear()
    raw_cache_clear()

    board = fastboard(fen)
    tree = MCTSTree(board, cfg)
    forest = MCTSForest()
    forest.add_tree(tree)

    out = search(tree, forest, cfg, infer_fn, n_sims, batch)
    out["visits"] = tree.root_child_visits()

    if rebuild:
        # caches are warm now; advance and time re-reaching the same sim count.
        # this is the number the C++ rebuild-path work actually moves.
        mv, _method, _alt = tree.best()
        if mv is not None:
            tree.advance(board, mv)
            tree.reset_for_new_move()
            tree.sims_completed_this_move = 0
            warm = search(tree, forest, cfg, infer_fn, n_sims, batch)
            out["rebuild_wall_s"] = warm["wall_s"]
            out["rebuild_sims"] = warm["sims"]
            out["rebuild_puct"] = warm["puct"]
            out["rebuild_new_nodes"] = warm["new_nodes"]

    forest.pop_tree(tree)
    return out


def ply_bucket(ply):
    for edge in (10, 30, 60, 100):
        if ply < edge:
            return f"<{edge}"
    return ">=100"


def summarize(rows):
    """Aggregate overall and per ply bucket. The ply curve is the point."""
    def agg(subset):
        wall = sum(r["wall_s"] for r in subset)
        sims = sum(r["sims"] for r in subset)
        puct = sum(r["puct"] for r in subset)
        nodes = sum(r["new_nodes"] for r in subset)
        pruned = sum(r.get("pruned", 0) for r in subset)
        return {
            "n_positions": len(subset),
            "wall_s": wall,
            "sims": sims,
            "sims_per_sec": sims / wall if wall else 0.0,
            "nodes_per_sec": nodes / wall if wall else 0.0,
            "puct_per_sim": puct / sims if sims else 0.0,
            "pruned_per_sim": pruned / sims if sims else 0.0,
            "us_per_sim": (wall / sims * 1e6) if sims else 0.0,
        }

    buckets = {}
    for r in rows:
        buckets.setdefault(ply_bucket(r["ply"]), []).append(r)

    return {
        "overall": agg(rows),
        "by_ply": {k: agg(v) for k, v in sorted(buckets.items())},
    }


def rebuild_agg(rows):
    rb = [r for r in rows if "rebuild_wall_s" in r]
    wall = sum(r["rebuild_wall_s"] for r in rb)
    sims = sum(r["rebuild_sims"] for r in rb)
    nodes = sum(r.get("rebuild_new_nodes", 0) for r in rb)
    return {
        "n_positions": len(rb),
        "wall_s": wall,
        "sims": sims,
        "sims_per_sec": sims / wall if wall else 0.0,
        "nodes_per_sec": nodes / wall if wall else 0.0,
        "us_per_sim": (wall / sims * 1e6) if sims else 0.0,
    }


def mean_sd(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, var ** 0.5


def sweep_stats(rows, rebuild):
    """Per-sweep scalars, the units the mean/sd are computed over."""
    o = summarize(rows)["overall"]
    out = {k: o[k] for k in
           ("wall_s", "sims", "sims_per_sec", "nodes_per_sec",
            "us_per_sim", "puct_per_sim", "pruned_per_sim")}
    if rebuild:
        rb = rebuild_agg(rows)
        out["rebuild_us_per_sim"] = rb["us_per_sim"]
        out["rebuild_sims_per_sec"] = rb["sims_per_sec"]
    return out


def run_bench(fixtures, n_sims=2000, batch=32, backend="stub", cfg=None,
              reuse_tree=False, rebuild=False, sweeps=6, tag="bench",
              save=True, quiet=False):
    """Run the fixture set `sweeps` times in one process.

    Repeats live inside the run so a single invocation yields mean/sd and the
    A/B is decidable without eyeballing a shell loop. Same-build spread was
    ~2%, so one sweep cannot resolve the ~2% changes we are chasing.
    """
    cfg = bench_config(cfg, reuse_tree=reuse_tree)
    infer_fn = stub_infer if backend == "stub" else build_real_infer(cfg)
    proc = psutil.Process()

    # rows are flushed as we go: a C++ segfault on one fixture takes the whole
    # process down, and losing the run with it is annoying.
    run_dir = paths.new_run_dir(tag) if save else None

    all_rows = []
    per_sweep = []
    for s in range(sweeps):
        rows = []
        for fx in fixtures:
            r = run_position(cfg, fx["fen"], n_sims, infer_fn, batch,
                             rebuild=rebuild)
            r["ply"] = fx.get("ply", 0)
            r["source"] = fx.get("source", "")
            r["fen"] = fx["fen"]
            r["sweep"] = s
            rows.append(r)
        all_rows += rows

        st = sweep_stats(rows, rebuild)
        per_sweep.append(st)
        if not quiet:
            print(f"sweep {s + 1}/{sweeps}: {st['sims_per_sec']:.0f} sims/s  "
                  f"{st['us_per_sim']:.2f} us/sim")
        if run_dir is not None:
            paths.save_json(run_dir / "rows.json", all_rows)

    keys = sorted(per_sweep[0].keys())
    stats = {}
    for k in keys:
        m, sd = mean_sd([p[k] for p in per_sweep])
        stats[k] = m
        stats[k + "_sd"] = sd

    result = {
        "n_sims": n_sims,
        "batch": batch,
        "backend": backend,
        "reuse_tree": reuse_tree,
        "rebuild": rebuild,
        "sweeps": sweeps,
        "n_fixtures": len(fixtures),
        "peak_rss_mb": proc.memory_info().rss / (1024 * 1024),
        "repo_stamps": paths.repo_stamps(),
        "stats": stats,
        "per_sweep": per_sweep,
        "summary": summarize(all_rows),
        "rows": all_rows,
    }
    if rebuild:
        result["summary"]["rebuild"] = rebuild_agg(all_rows)

    if run_dir is not None:
        paths.save_json(run_dir / "result.json", result)
        result["run_dir"] = str(run_dir)
        print(f"\nwrote {run_dir}")

    print_summary(result)
    return result


def print_summary(result):
    s = result["summary"]
    o = s["overall"]
    st = result.get("stats", {})

    print(f"\n{result['sweeps']} sweeps x {result['n_fixtures']} pos "
          f"@ {result['n_sims']} sims")
    print(f"  sims/s   {st.get('sims_per_sec', 0):>9.0f} "
          f"+/- {st.get('sims_per_sec_sd', 0):.0f}")
    print(f"  nodes/s  {st.get('nodes_per_sec', 0):>9.0f} "
          f"+/- {st.get('nodes_per_sec_sd', 0):.0f}")
    print(f"  us/sim   {st.get('us_per_sim', 0):>9.2f} "
          f"+/- {st.get('us_per_sim_sd', 0):.2f}")
    print(f"  puct/sim {o['puct_per_sim']:>9.2f}   "
          f"pruned/sim {o['pruned_per_sim']:.3f}")
    if "rebuild_us_per_sim" in st:
        print(f"  rebuild  {st['rebuild_us_per_sim']:>9.2f} "
              f"+/- {st.get('rebuild_us_per_sim_sd', 0):.2f} us/sim")

    print(f"\n{'ply':<8}{'pos':>5}{'sims/s':>10}{'nodes/s':>10}"
          f"{'us/sim':>10}{'puct/sim':>11}")
    for k, v in s["by_ply"].items():
        print(f"{k:<8}{v['n_positions']:>5}{v['sims_per_sec']:>10.0f}"
              f"{v['nodes_per_sec']:>10.0f}"
              f"{v['us_per_sim']:>10.2f}{v['puct_per_sim']:>11.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--backend", choices=["stub", "real"], default="stub")
    ap.add_argument("--config", default=None)
    ap.add_argument("--reuse-tree", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sweeps", type=int, default=6,
                    help="repeat the fixture set N times in-process")
    ap.add_argument("--tag", default="bench")
    args = ap.parse_args()

    fixtures = load_fixture_set(args.fixtures)
    if args.limit:
        fixtures = fixtures[:args.limit]

    cfg = Config.from_yaml(args.config) if args.config else None
    run_bench(fixtures, n_sims=args.n_sims, batch=args.batch,
              backend=args.backend, cfg=cfg, reuse_tree=args.reuse_tree,
              rebuild=args.rebuild, sweeps=args.sweeps, tag=args.tag)


if __name__ == "__main__":
    main()
