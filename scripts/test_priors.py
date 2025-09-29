# tools/test_priors.py
import time
import math
import random
from collections import defaultdict

import numpy as np

import pyfastchess as pf  # your extension
from pyfastchess import PriorConfig, PriorEngine

# stable softmax
def softmax_logits(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    xm = x.max()
    ex = np.exp(x - xm)
    s = ex.sum()
    return (ex / (s + 1e-30)).astype(np.float32)

# Old python prior munger (faithful to original snippet)
def old_python_prior_munger(board,
                            legal,
                            logits_from,
                            logits_to,
                            logits_piece,
                            logits_promo,
                            cfg,
                            root_stm,
                            stm_leaf):
    # Softmax the factorized heads
    pf = softmax_logits(logits_from)
    pt = softmax_logits(logits_to)
    pp = softmax_logits(logits_piece)
    pr = softmax_logits(logits_promo)

    # sizes are ensured by caller to be >= max indices
    fr, to, pc, pr_idx = board.moves_to_labels(legal)

    # compute base priors (product of head probs)
    n = len(legal)
    pri = []
    s = 0.0
    for i, mv in enumerate(legal):
        fi, ti, pci, pri_i = fr[i], to[i], pc[i], pr_idx[i]
        v = float(max(0.0, float(pf[fi]) * float(pt[ti]) *
                       float(pp[pci]) * float(pr[pri_i])))
        pri.append((mv, v))
        s += v

    # normalize
    if s > 0.0:
        pri = [(m, p / s) for (m, p) in pri]
    else:
        u = 1.0 / float(n) if n > 0 else 0.0
        pri = [(m, u) for (m, _) in pri]

    # blend with uniform
    is_endgame = board.piece_count() <= 14
    if root_stm != stm_leaf:
        mix = cfg['opponent_uniform_mix']
    elif is_endgame:
        mix = cfg['endgame_uniform_mix']
    else:
        mix = cfg['anytime_uniform_mix']

    if mix > 0.0:
        u = 1.0 / float(n) if n > 0 else 0.0
        pri = [(m, (1.0 - mix) * p + mix * u) for (m, p) in pri]

    # Post-opening boosts (old behavior: multiplicative repetition penalty)
    if cfg['use_prior_boosts'] and board.history_size() > 10:
        g_chk = cfg['anytime_prior_adjustments'].get("gives_check", 0.0)
        repp = cfg['anytime_prior_adjustments'].get("repetition_penalty", 1.0)

        # endgame overrides
        egc, egpp = 0.0, 0.0
        if is_endgame:
            eg_adj = cfg['endgame_prior_adjustments']
            repp = eg_adj.get("repetition_penalty", repp)
            egc = eg_adj.get("capture", 0.0)
            egpp = eg_adj.get("pawn_push", 0.0)

        adjusted = []
        for (m, p) in pri:
            ppv = p
            if g_chk and board.gives_check(m):
                ppv += g_chk
            if repp != 1.0 and board.would_be_repetition(m, 1):
                ppv *= repp
            if is_endgame:
                if egpp and board.is_pawn_move(m):
                    ppv += egpp
                if egc and board.is_capture(m):
                    ppv += egc
            adjusted.append((m, ppv))
        pri = adjusted
        # note: original python didn't re-normalize here — we keep that behavior
    return pri

# Build PriorConfig for PriorEngine similar to MCTSTree.__init__ mapping
def make_prior_config_from_cfg(cfg):
    pcfg = PriorConfig()
    pcfg.anytime_uniform_mix = float(cfg['anytime_uniform_mix'])
    pcfg.endgame_uniform_mix = float(cfg['endgame_uniform_mix'])
    pcfg.opponent_uniform_mix = float(cfg['opponent_uniform_mix'])
    pcfg.use_prior_boosts = bool(cfg['use_prior_boosts'])
    pcfg.anytime_gives_check = float(
        cfg['anytime_prior_adjustments'].get("gives_check", 0.0))
    # new engine expects subtractive rep param: we map multiplicative penalty p -> (1-p)
    pcfg.anytime_repetition_sub = float(
        1.0 - cfg['anytime_prior_adjustments'].get("repetition_penalty", 1.0))
    eg = cfg['endgame_prior_adjustments']
    pcfg.endgame_pawn_push = float(eg.get("pawn_push", 0.0))
    pcfg.endgame_capture = float(eg.get("capture", 0.0))
    pcfg.endgame_repetition_sub = float(
        1.0 - eg.get("repetition_penalty", 1.0))
    # clipping defaults (expose options in cfg if desired)
    pcfg.clip_enabled = True
    pcfg.clip_min = float(cfg.get("prior_clip_min", 1e-6))
    pcfg.clip_max = float(cfg.get("prior_clip_max", 1.0))
    return pcfg

# metrics
def l1_distance(dict_a, dict_b):
    all_keys = set(dict_a) | set(dict_b)
    return sum([abs(dict_a.get(k, 0.0) - dict_b.get(k, 0.0))
               for k in all_keys])

def kl_divergence(p_dict, q_dict, eps=1e-12):
    # KL(p||q)
    all_keys = set(p_dict) | set(q_dict)
    kl = 0.0
    for k in all_keys:
        p = p_dict.get(k, 0.0) + eps
        q = q_dict.get(k, 0.0) + eps
        kl += p * math.log(p / q)
    return kl

def test_against_positions(num_positions=200, seed=1234):
    random.seed(seed)
    np.random.seed(seed)

    # example config (tweak to match your real cfg)
    cfg = {
        "anytime_uniform_mix": 0.25,
        "endgame_uniform_mix": 0.5,
        "opponent_uniform_mix": 0.9,
        "use_prior_boosts": True,
        "anytime_prior_adjustments": {
            "gives_check": 0.12,
            "repetition_penalty": 0.9,
        },
        "endgame_prior_adjustments": {
            "capture": 0.08,
            "pawn_push": 0.06,
            "repetition_penalty": 0.7,
        },
        # optional clip overrides
        "prior_clip_min": 1e-6,
        "prior_clip_max": 1.0,
    }

    pcfg = make_prior_config_from_cfg(cfg)
    pengine = PriorEngine(pcfg)

    # positions: generate by random playouts from start
    boards = []
    from chessbot.utils import random_init
    for i in range(num_positions):
        depth = random.randint(0, 40)
        b = random_init(depth)
        boards.append(b)

    stats = {"old_time": 0.0, "new_time": 0.0,
             "l1": [], "kl": [], "n_moves": []}

    for i, b in enumerate(boards):
        legal = b.legal_moves()
        if not legal:
            continue

        # build factorized logits sized to label indices
        fr, to, pc, pr_idx = b.moves_to_labels(legal)
        # ensure sizes
        size_from = int(max(fr) + 1) if fr else 1
        size_to   = int(max(to) + 1) if to else 1
        size_piece = int(max(pc) + 1) if pc else 1
        size_promo = int(max(pr_idx) + 1) if pr_idx else 1

        # random logits as test input (use same for both mungers)
        logits_from = np.random.randn(size_from).astype(np.float32)
        logits_to   = np.random.randn(size_to).astype(np.float32)
        logits_piece= np.random.randn(size_piece).astype(np.float32)
        logits_promo= np.random.randn(size_promo).astype(np.float32)

        # old python
        t0 = time.perf_counter()
        pri_old = old_python_prior_munger(
            b, legal,
            logits_from, logits_to, logits_piece, logits_promo,
            cfg,  # cfg dict
            # root_stm -- use white for half, black for half
            "w" if (i % 2 == 0) else "b",
            b.side_to_move())
        t1 = time.perf_counter()

        # new C++ PriorEngine (expects softmax arrays or logits? binding expects floats arrays)
        # pass softmax outputs as 1D np.array float32 (like mcts_utils)
        sf_from = softmax_logits(logits_from)
        sf_to   = softmax_logits(logits_to)
        sf_piece= softmax_logits(logits_piece)
        sf_promo= softmax_logits(logits_promo)

        t2 = time.perf_counter()
        pri_new = pengine.build(
            b, legal, sf_from, sf_to, sf_piece, sf_promo,
            "w" if (i % 2 == 0) else "b", b.side_to_move())
        t3 = time.perf_counter()

        stats['old_time'] += (t1 - t0)
        stats['new_time'] += (t3 - t2)
        # convert to dicts (move -> prob). Old pri may not be normalized; normalize for fair metric

        sum_old = sum([max(0.0, p) for (_m, p) in pri_old])
        if sum_old > 0:
            old_dict = {m: max(0.0, p) / sum_old for (m, p) in pri_old}
        else:
            old_dict = {m: 1.0/len(legal) for m in legal}

        # new pri is already normalized by engine (but still coerce dict)
        new_dict = {m: float(p) for (m, p) in pri_new}
        # ensure sums
        # fixed
        s_new = 0.0
        for v in new_dict.values():
            s_new += float(v)
        if s_new == 0.0:
            s_new = 1.0
        for k, v in list(new_dict.items()):
            new_dict[k] = float(v) / s_new

        l1 = l1_distance(old_dict, new_dict)
        kl = kl_divergence(old_dict, new_dict)

        stats['l1'].append(l1)
        stats['kl'].append(kl)
        stats['n_moves'].append(len(legal))

    # summary
    n = len(stats['l1'])
    print("Positions tested:", n)
    print("Avg legal moves:", sum(stats['n_moves'])/n if n else 0)
    print("Total old python time: {:.6f}s".format(stats['old_time']))
    print("Total new  C++  time: {:.6f}s".format(stats['new_time']))
    print("Avg time / pos: old = {:.6f} ms, new = {:.6f} ms".format(
        (stats['old_time']/n)*1000.0, (stats['new_time']/n)*1000.0))
    ratio = stats['old_time'] / stats['new_time'] if stats['new_time'] > 0 else float("inf")
    print("Speedup (old / new): {:.2f}x".format(ratio))

    print("Avg L1 between priors: {:.6f}".format(
        sum(stats['l1'])/n if n else 0.0))
    print("Avg KL(p_old||p_new): {:.6f}".format(
        sum(stats['kl'])/n if n else 0.0))

    return stats

if __name__ == "__main__":
    # tweak num_positions down if you want a fast smoke-test
    stats = test_against_positions(num_positions=5000)
