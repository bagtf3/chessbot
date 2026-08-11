"""Offline harness for root_chunk_size. Mirrors the looper's forest loop but
fakes the NN, so it runs with no GPU and no model and can be pointed at a
live-ish workload safely.

Verifies, per move and across advance_root:
  - N conservation: root.N == descents that actually resolved
  - sum(child N) never exceeds root N
  - the search still produces a legal move and the game terminates
and measures leaves/sec, PUCT per leaf, blocked rate, depth.

The fake NN draws its policy from a Dirichlet so the priors have realistic
concentration (a uniform-random policy makes every child look equally good and
hides the selection behaviour we care about). Values are keyed off the zobrist
so a run is reproducible; pass --stochastic for fresh draws each time.

  python scripts/tools/chunk_harness.py --chunks 1,4 --games 24 --micro 8
"""
import argparse
import time

import numpy as np
import pyfastchess as pf

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FASTPATH = 1024


class FakeNN:
    """Dirichlet policies, precomputed into a pool and indexed by zobrist.

    Drawing a fresh 1858-dim Dirichlet per key costs far more than the tree
    work being measured and would swamp the timings. A pool keeps the priors
    realistically peaky while making lookup a gather.
    """

    def __init__(self, alpha, pool=512, seed=0):
        r = np.random.default_rng(seed)
        d = r.dirichlet(np.full(1858, alpha), size=pool).astype(np.float32)
        self.pol = np.log(d + 1e-9)
        w = r.dirichlet([1.0, 1.4, 1.0], size=pool).astype(np.float32)
        self.wdl = w
        self.pool = pool

    def __call__(self, keys):
        idx = (keys % self.pool).astype(np.int64)
        return self.wdl[idx], self.pol[idx]


class Game:
    """One tree, played to a fixed ply budget, advancing the root each move."""

    def __init__(self, sims_per_move, max_plies, cfg):
        self.board = pf.Board(START)
        self.tree = pf.MCTSTree(self.board, cfg["c_puct"], float(cfg["sims"]),
                                cfg["pruning"], cfg["uniform_eps"],
                                cfg["prior_clip_max"])
        self.tree.set_dirichlet(0.0, 0.3)
        self.tree.set_reuse_tree(cfg["reuse_tree"])
        self.tree.set_root_chunk_size(cfg["chunk"])
        self.tree.set_chunk_warmup_visits(cfg["warmup"])
        self.sims_per_move = sims_per_move
        self.max_plies = max_plies
        self.sims = 0          # descents resolved for the current move
        self.plies = 0
        self.done = False
        self.move_checks = []  # (root_N, sims, sum_kids) at each move boundary

    def note(self, res):
        self.sims += res.count_new + res.count_cached + res.count_terminal

    def maybe_move(self):
        if self.done or self.sims < self.sims_per_move:
            return
        root = self.tree.root()
        kids = self.tree.root_child_visits() or []
        self.move_checks.append((root.N, self.sims, sum(n for _, n in kids)))

        mv, _ = self.tree.best()
        if not mv or self.plies >= self.max_plies:
            self.done = True
            return
        if not self.tree.advance_root(mv):
            self.done = True
            return
        self.board.push_uci(mv)
        self.plies += 1
        self.sims = 0


def run(chunk, args):
    pf.raw_cache_clear()
    pf.priors_cache_clear()
    nn = FakeNN(args.alpha, seed=0)
    cfg = {
        "chunk": chunk, "c_puct": 2.0, "sims": args.sims,
        "pruning": 1.33, "uniform_eps": 0.001, "prior_clip_max": 0.8,
        "reuse_tree": args.reuse_tree, "warmup": args.warmup,
    }
    forest = pf.MCTSForest()
    games = [Game(args.sims, args.plies, cfg) for _ in range(args.games)]
    for g in games:
        forest.add_tree(g.tree)

    leaves = puct = blocked = depth = 0
    t_tree = t_nn = 0.0
    t_resolve = t_collect = t_encode = 0.0
    t0 = time.time()
    for _ in range(args.iters):
        live = [g for g in games if not g.done]
        if not live:
            break
        # split the tree work: only `collect` is touched by chunking, so if it
        # is a small share of the total then chunking's speed ceiling is small
        # no matter how much PUCT it removes
        ts = time.perf_counter()
        forest.resolve_all_inflight()
        t_resolve += time.perf_counter() - ts

        ts = time.perf_counter()
        for g in live:
            res = g.tree.collect_many_leaves(args.micro, FASTPATH)
            g.note(res)
            leaves += res.count_new + res.count_cached + res.count_terminal
            puct += res.total_puct
            blocked += res.total_blocked
            depth += res.total_depth
        t_collect += time.perf_counter() - ts

        ts = time.perf_counter()
        keys, _enc = forest.get_all_encoded()
        t_encode += time.perf_counter() - ts
        t_tree = t_resolve + t_collect + t_encode

        # stand-in for the GPU; excluded from the tree timing
        ts = time.perf_counter()
        if len(keys):
            wdl, pol = nn(keys)
            pf.raw_cache_bulk_insert_np(keys, wdl, pol)
        t_nn += time.perf_counter() - ts

        for g in live:
            g.maybe_move()
    elapsed = time.time() - t0

    bad_n = bad_kids = 0
    moves = 0
    for g in games:
        for root_n, sims, kids in g.move_checks:
            moves += 1
            if root_n != sims:
                bad_n += 1
            if kids > root_n:
                bad_kids += 1
    return {
        "chunk": chunk, "leaves": leaves, "moves": moves,
        "plies": sum(g.plies for g in games),
        "sec": elapsed, "t_tree": t_tree, "t_nn": t_nn,
        "t_resolve": t_resolve, "t_collect": t_collect, "t_encode": t_encode,
        "collect_pct": 100 * t_collect / max(1e-9, t_tree),
        "leaves_sec": leaves / max(1e-9, t_tree),
        "puct_leaf": puct / max(1, leaves),
        "blk_leaf": blocked / max(1, leaves),
        "depth_leaf": depth / max(1, leaves),
        "bad_n": bad_n, "bad_kids": bad_kids,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="1,2,4,8,16")
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--plies", type=int, default=12)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Dirichlet concentration for the fake policy; small = peaky")
    p.add_argument("--reuse-tree", action="store_true")
    p.add_argument("--warmup", type=int, default=100,
                   help="root visits before chunking engages; 0 = immediate")
    args = p.parse_args()

    print(f"games={args.games} micro={args.micro} sims/move={args.sims} "
          f"plies<={args.plies} iters={args.iters} alpha={args.alpha} "
          f"reuse_tree={args.reuse_tree} warmup={args.warmup}")
    hdr = (f"{'chunk':>6}{'leaves':>9}{'moves':>7}{'tree_s':>8}{'coll_s':>8}{'coll%':>7}{'leaves/s':>10}"
           f"{'puct/leaf':>11}{'blk/leaf':>10}{'depth/leaf':>12}"
           f"{'badN':>6}{'badKid':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for c in [int(x) for x in args.chunks.split(",")]:
        r = run(c, args)
        rows.append(r)
        print(f"{r['chunk']:>6}{r['leaves']:>9}{r['moves']:>7}{r['t_tree']:>8.1f}{r['t_collect']:>8.1f}{r['collect_pct']:>7.0f}"
              f"{r['leaves_sec']:>10.0f}{r['puct_leaf']:>11.2f}"
              f"{r['blk_leaf']:>10.3f}{r['depth_leaf']:>12.2f}"
              f"{r['bad_n']:>6}{r['bad_kids']:>8}")

    print()
    viol = sum(r["bad_n"] + r["bad_kids"] for r in rows)
    if viol:
        print(f"FAIL: {viol} invariant violation(s)")
    else:
        print(f"PASS: N conserved at every move boundary "
              f"({sum(r['moves'] for r in rows)} checks)")
    base = next((r for r in rows if r["chunk"] == 1), None)
    if base:
        print()
        print(f"{'chunk':>6}{'puct saved':>12}{'speed':>9}")
        for r in rows:
            print(f"{r['chunk']:>6}"
                  f"{100*(1-r['puct_leaf']/base['puct_leaf']):>11.1f}%"
                  f"{r['leaves_sec']/base['leaves_sec']:>8.2f}x")


if __name__ == "__main__":
    main()
