"""
Benchmark: MCTSForest.get_all_encoded() vs Python loop + pending_encoded_64_tokens()

128 trees x N leaves per tree, N in [1, 2, 3, 4], 10 reps each.
Both approaches are destructive (pop_pending_to_inflight), so trees are recreated per rep.
"""

import time
import numpy as np
import pyfastchess as pf

N_TREES = 128
REPS = 10
LEAVES_LIST = [1, 2, 3, 4]


def make_tree():
    board = pf.Board()
    t = pf.MCTSTree(board, 1.5, 800, 1.0, 0.25, 0.75)
    t.set_reuse_tree(False)
    return t, board


def expand_root(tree, board):
    """Feed a dummy NN result to the root so children can be collected."""
    tree.collect_many_leaves(1, 0)
    items = tree.pending_encoded_64_tokens()  # pops root to inflight
    if not items:
        return
    legal = board.legal_moves()
    if not legal:
        return
    uniform = 1.0 / len(legal)
    priors = [(mv, uniform) for mv in legal]
    root_node = tree.root()
    tree.apply_result(root_node, priors, (0.33, 0.34, 0.33))


def setup_trees(n_leaves):
    """
    Create N_TREES trees each with up to n_leaves pending NN leaves.
    For n_leaves > 1 the root is pre-expanded so children can be queued.
    Returns (trees, actual_pending_count).
    """
    trees = []
    total_pending = 0
    for _ in range(N_TREES):
        t, board = make_tree()
        if n_leaves > 1:
            expand_root(t, board)
        t.collect_many_leaves(n_leaves, 0)
        p = t.count_pending()
        total_pending += p
        trees.append(t)
    return trees, total_pending


def python_encode(trees):
    """Current looper.py path: per-tree pending_encoded_64_tokens + stack."""
    keys = []
    boards = []
    for t in trees:
        for z, arr in t.pending_encoded_64_tokens():
            keys.append(z)
            boards.append(np.asarray(arr, dtype=np.int32))
    if not boards:
        return None, None
    keys_np = np.array(keys, dtype=np.uint64)
    boards_np = np.stack(boards, axis=0)
    return keys_np, boards_np


def forest_encode(trees):
    """MCTSForest path: single C++ call returns pre-stacked arrays."""
    forest = pf.MCTSForest()
    for t in trees:
        forest.add_tree(t)
    keys_np, boards_np = forest.get_all_encoded()
    boards_np = boards_np.astype(np.int32)
    return keys_np, boards_np


def run_bench(fn, n_leaves):
    times = []
    actual_N = None
    for _ in range(REPS):
        trees, total_pending = setup_trees(n_leaves)
        if actual_N is None:
            actual_N = total_pending
        t0 = time.perf_counter()
        keys_np, boards_np = fn(trees)
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, actual_N


def main():
    print(f"N_TREES={N_TREES}, REPS={REPS}")
    print()
    print(f"{'leaves':>7} {'actual_N':>9} {'python_ms':>11} {'forest_ms':>11} {'speedup':>9}")
    print("-" * 52)

    for n_leaves in LEAVES_LIST:
        py_ms, actual_N = run_bench(python_encode, n_leaves)
        fo_ms, _        = run_bench(forest_encode,  n_leaves)
        speedup = py_ms / fo_ms if fo_ms > 0 else float("inf")
        print(f"{n_leaves:>7} {actual_N:>9} {py_ms:>11.3f} {fo_ms:>11.3f} {speedup:>9.2f}x")

    print()


if __name__ == "__main__":
    main()
