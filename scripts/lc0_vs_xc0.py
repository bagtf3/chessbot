import os
import re
import math
import time
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_rows", None)

import chess
import chess.engine

from chessbot import LC0_LOC


EPS = 1e-12


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size < 2 or b.size < 2:
        return 0.0

    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0.0 or sb <= 0.0:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def rank_of_move(ucis, probs, target_uci):
    if not target_uci:
        return 0

    pairs = list(zip(ucis, probs))
    pairs.sort(key=lambda x: x[1], reverse=True)

    for i, (u, _) in enumerate(pairs, start=1):
        if u == target_uci:
            return i

    return len(pairs) + 1


def parse_int_safe(s, default=None):
    try:
        return int(s)
    except Exception:
        return default


def parse_float_safe(s, default=None):
    if s is None:
        return default

    t = s.strip()
    if not t:
        return default

    if "-" in t and t.replace("-", "") in [".----", "----", ".---", "---"]:
        return default

    if t in ["-.----", "--.---", "--.----", "nan", "-nan"]:
        return default

    try:
        return float(t)
    except Exception:
        return default
    

def cp_to_value_tanh(cp, mid_cp=200.0):
    k = math.atanh(0.5) / mid_cp
    return np.clip(math.tanh(k * cp), -0.97, 0.97)


def score_to_value_stm(info_score, board, mate_cp=2500):
    if info_score is None:
        return 0.0

    rel = info_score.relative
    cp = rel.score(mate_score=mate_cp)
    if cp is None:
        cp = 0
    return cp_to_value_tanh(cp)


def calc_entropy(weights):
    a = np.asarray(weights, dtype=float)
    if a.size == 0:
        return 0.0, 0.0

    a = np.where(a > 0.0, a, 0.0)
    total = a.sum()
    if total <= 0.0:
        return 0.0, 0.0

    p = a / total
    mask = p > 0.0
    pm = p[mask]
    ent = - (pm * np.log2(pm)).sum()

    n = p.size
    norm = ent / np.log2(n) if n > 1 else 0.0
    return ent, norm


def kl_divergence_bits(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if p.sum() <= 0.0:
        p = np.ones_like(p, dtype=np.float64) / p.size
    else:
        p = p / p.sum()

    if q.sum() <= 0.0:
        q = np.ones_like(q, dtype=np.float64) / q.size
    else:
        q = q / q.sum()

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    return np.sum(p * np.log2(p / q))


def normalize_prob(x):
    a = np.asarray(x, dtype=float)
    s = a.sum()
    if s <= 0.0:
        return np.ones_like(a, dtype=float) / max(1, a.size)
    return a / s


def ensure_all_legal_have_visits(legal_ucis, visit_map):
    out = []
    for u in legal_ucis:
        v = visit_map.get(u, 0)
        if v <= 0:
            v = 1
        out.append(v)
    return out


def policy_metrics_from_probs(model_probs, label_probs):
    mp = normalize_prob(model_probs)
    lp = normalize_prob(label_probs)

    ce = -np.sum(lp * np.log(mp + EPS))
    n = mp.size
    uniform = np.ones(n, dtype=float) / max(1, n)
    uniform_ce = -np.sum(lp * np.log(uniform + EPS))

    model_best = int(mp.argmax())
    true_best = int(lp.argmax())

    top1_exact = 1.0 if model_best == true_best else 0.0
    avg_top_prob = float(mp[model_best])

    top1_mass = float(mp[true_best])

    k3 = min(3, n)
    k5 = min(5, n)

    top3_idx = np.argpartition(-lp, k3 - 1)[:k3]
    top5_idx = np.argpartition(-lp, k5 - 1)[:k5]

    top3_mass = float(mp[top3_idx].sum())
    top5_mass = float(mp[top5_idx].sum())

    support = lp > 0
    support[true_best] = False
    prob_on_others = float((mp * support).sum())

    return {
        "policy_ce": float(ce),
        "uniform_ce": float(uniform_ce),
        "ce_gain": float(uniform_ce - ce),
        "top1_exact": float(top1_exact),
        "avg_top_prob": float(avg_top_prob),
        "top1_mass": float(top1_mass),
        "top3_mass": float(top3_mass),
        "top5_mass": float(top5_mass),
        "prob_on_others": float(prob_on_others),
        "mass_on_legal": 1.0,
    }


def compute_target_y(game_result, stm_is_white, tr):
    z_stm = game_result if stm_is_white else -1 * game_result

    if "Q_stm" in tr:
        q_stm = tr["Q_stm"]
    else:
        q = tr.get("best_Q", tr.get("visit_weighted_Q", 0.0))
        q_stm = q if stm_is_white else -1 * q

    y = 0.5 * z_stm + 0.5 * q_stm
    return float(np.clip(y, -1.0, 1.0))


class Lc0VerboseParser:
    def __init__(self):
        self.pat_move = re.compile(r"^([a-h][1-8][a-h][1-8][qrbn]?)\s")
        self.pat_n = re.compile(r"\bN:\s*([0-9]+)")
        self.pat_p = re.compile(r"\(P:\s*([0-9.]+)%\)")
        self.pat_q = re.compile(r"\(Q:\s*([\-0-9.]+)\)")
        self.pat_u = re.compile(r"\(U:\s*([\-0-9.]+)\)")
        self.pat_s = re.compile(r"\(S:\s*([\-0-9.]+)\)")
        self.pat_v = re.compile(r"\(V:\s*([\-0-9.]+)\)")
        self.pat_wl = re.compile(r"\(WL:\s*([\-0-9.]+)\)")
        self.pat_d = re.compile(r"\(D:\s*([0-9.]+)\)")
        self.pat_m = re.compile(r"\(M:\s*([0-9.]+)\)")

    def parse_stream_info_strings(self, info):
        s = info.get("string")
        if not s:
            return []

        if not isinstance(s, list):
            s = [s]

        rows = []
        for line in s:
            m = self.pat_move.search(line)
            if not m:
                continue

            mv = m.group(1)
            if mv == "node":
                continue

            row = {"move": mv, "raw": line}

            m2 = self.pat_n.search(line)
            if m2:
                n = parse_int_safe(m2.group(1), default=None)
                if n is not None:
                    row["N"] = n

            m2 = self.pat_p.search(line)
            if m2:
                p = parse_float_safe(m2.group(1), default=None)
                if p is not None:
                    row["P_pct"] = p

            for key, pat in [
                ("Q", self.pat_q),
                ("U", self.pat_u),
                ("S", self.pat_s),
                ("V", self.pat_v),
                ("WL", self.pat_wl),
                ("D", self.pat_d),
                ("M", self.pat_m),
            ]:
                m2 = pat.search(line)
                if not m2:
                    continue

                v = parse_float_safe(m2.group(1), default=None)
                if v is not None:
                    row[key] = v

            rows.append(row)

        return rows


thread_local = threading.local()


def get_thread_lc0_engine(lc0_loc, verbose=True, minibatch_size=64):
    eng = getattr(thread_local, "lc0_engine", None)
    if eng is not None:
        return eng

    eng = chess.engine.SimpleEngine.popen_uci(lc0_loc)

    cfg = {"MinibatchSize": minibatch_size}
    if verbose:
        cfg["VerboseMoveStats"] = True

    eng.configure(cfg)

    thread_local.lc0_engine = eng
    thread_local.lc0_parser = Lc0VerboseParser()
    return eng


def close_thread_lc0_engine():
    eng = getattr(thread_local, "lc0_engine", None)
    if eng is not None:
        try:
            eng.quit()
        except Exception:
            pass
    thread_local.lc0_engine = None
    thread_local.lc0_parser = None


def lc0_analyze_position(board, nodes=60000):
    eng = thread_local.lc0_engine
    parser = thread_local.lc0_parser

    limit = chess.engine.Limit(nodes=nodes)

    last_by_move = {}
    root_score = None
    root_pv0 = None

    with eng.analysis(board, limit, info=chess.engine.INFO_ALL) as an:
        for info in an:
            if "score" in info and root_score is None:
                root_score = info.get("score")

            pv = info.get("pv")
            if pv and root_pv0 is None:
                try:
                    root_pv0 = pv[0].uci()
                except Exception:
                    root_pv0 = None

            rows = parser.parse_stream_info_strings(info)
            for r in rows:
                last_by_move[r["move"]] = r

    rows = list(last_by_move.values())
    rows = sorted(rows, key=lambda r: r.get("N", 0), reverse=True)

    return {
        "rows": rows,
        "root_score": root_score,
        "root_pv0": root_pv0,
    }


def load_game_pkl(path):
    with open(path, "rb") as f:
        g = pickle.load(f)
    return g


def iter_positions_from_game(game):
    board = chess.Board(game.get("start_fen"))
    tsd = game.get("tree_search_data", {})
    moves = game.get("moves_played", [])
    gid = str(game.get("game_id"))

    for ply_i, mv_uci in enumerate(moves):
        if board.is_game_over():
            break

        tr = tsd.get(ply_i, tsd.get(str(ply_i), None))
        if tr is None:
            mv = chess.Move.from_uci(mv_uci)
            if mv in board.legal_moves:
                board.push(mv)
            else:
                break
            continue

        cm = tr.get("candidate_moves", [])
        if not cm:
            mv = chess.Move.from_uci(mv_uci)
            if mv in board.legal_moves:
                board.push(mv)
            else:
                break
            continue

        yield {
            "game_id": gid,
            "ply_i": ply_i,
            "fen": board.fen(),
            "board": board.copy(stack=False),
            "tr": tr,
            "played_uci": mv_uci,
            "result": game.get("result", 0.0),
        }

        mv = chess.Move.from_uci(mv_uci)
        if mv in board.legal_moves:
            board.push(mv)
        else:
            break


def summarize_one_position(pos, nodes):
    board = pos["board"]
    tr = pos["tr"]
    cm = tr.get("candidate_moves", [])

    legal_ucis = [m.uci() for m in board.legal_moves]

    xc0_visit_map = {c["uci"]: int(c.get("visits", 0)) for c in cm}
    xc0_prior_map = {c["uci"]: float(c.get("P", 0.0)) for c in cm}
    xc0_q_map = {c["uci"]: float(c.get("Q", 0.0)) for c in cm}

    xc0_visits = ensure_all_legal_have_visits(legal_ucis, xc0_visit_map)
    xc0_priors = [xc0_prior_map.get(u, 0.0) for u in legal_ucis]

    xc0_visit_p = normalize_prob(xc0_visits)
    xc0_prior_p = normalize_prob(xc0_priors)

    xc0_ent_prior_norm = float(calc_entropy(xc0_prior_p)[1])
    xc0_ent_visit_norm = float(calc_entropy(xc0_visit_p)[1])
    xc0_kl_vis_prior_bits = float(kl_divergence_bits(xc0_visit_p, xc0_prior_p))

    lc0 = lc0_analyze_position(board, nodes=nodes)
    lc0_rows = lc0["rows"]

    lc0_visit_map = {r["move"]: int(r.get("N", 0)) for r in lc0_rows}
    lc0_prior_map = {r["move"]: float(r.get("P_pct", 0.0)) for r in lc0_rows}
    lc0_qs_map = {r["move"]: float(r.get("Q", 0.0)) for r in lc0_rows}

    lc0_visits = [lc0_visit_map.get(u, 0) for u in legal_ucis]
    lc0_priors_pct = [lc0_prior_map.get(u, 0.0) for u in legal_ucis]

    lc0_visit_p = normalize_prob(lc0_visits)
    lc0_prior_p = normalize_prob(lc0_priors_pct)

    lc0_ent_prior_norm = float(calc_entropy(lc0_prior_p)[1])
    lc0_ent_visit_norm = float(calc_entropy(lc0_visit_p)[1])
    lc0_kl_vis_prior_bits = float(kl_divergence_bits(lc0_visit_p, lc0_prior_p))

    kl_xc0_vis_lc0_vis_bits = float(kl_divergence_bits(xc0_visit_p, lc0_visit_p))
    kl_lc0_vis_xc0_vis_bits = float(kl_divergence_bits(lc0_visit_p, xc0_visit_p))
    kl_xc0_pr_lc0_pr_bits = float(kl_divergence_bits(xc0_prior_p, lc0_prior_p))

    policy_corr = safe_corr(lc0_prior_p, xc0_prior_p)

    lc0_policy_metrics = policy_metrics_from_probs(lc0_prior_p, xc0_visit_p)

    y = compute_target_y(pos["result"], bool(board.turn), tr)
    v_pred = score_to_value_stm(lc0.get("root_score"), board)

    lc0_top = legal_ucis[int(lc0_prior_p.argmax())] if legal_ucis else None
    xc0_top = legal_ucis[int(xc0_visit_p.argmax())] if legal_ucis else None

    rank_xc0_top_in_lc0_prior = rank_of_move(legal_ucis, lc0_prior_p, xc0_top)
    rank_lc0_top_in_xc0_prior = rank_of_move(legal_ucis, xc0_prior_p, lc0_top)

    out = {
        "game_id": pos["game_id"],
        "ply_i": pos["ply_i"],
        "stm_white": bool(board.turn),
        "fen": pos["fen"],
        "played_uci": pos["played_uci"],
        "xc0_move": tr.get("xc0_move"),
        "sel_method": tr.get("selection_method"),
        "lc0_pv0": lc0.get("root_pv0"),
        "xc0_ent_prior_norm": xc0_ent_prior_norm,
        "xc0_ent_visit_norm": xc0_ent_visit_norm,
        "xc0_kl_vis_prior_bits": xc0_kl_vis_prior_bits,
        "lc0_ent_prior_norm": lc0_ent_prior_norm,
        "lc0_ent_visit_norm": lc0_ent_visit_norm,
        "lc0_kl_vis_prior_bits": lc0_kl_vis_prior_bits,
        "kl_xc0_vis_lc0_vis_bits": kl_xc0_vis_lc0_vis_bits,
        "kl_lc0_vis_xc0_vis_bits": kl_lc0_vis_xc0_vis_bits,
        "kl_xc0_pr_lc0_pr_bits": kl_xc0_pr_lc0_pr_bits,
        "policy_corr": float(policy_corr),
        "y_target": float(y),
        "v_pred": float(v_pred),
        "lc0_top_prior_move": lc0_top,
        "xc0_top_visit_move": xc0_top,
        "top_move_match": 1.0 if lc0_top == xc0_top else 0.0,
        "rank_xc0_top_in_lc0_prior": int(rank_xc0_top_in_lc0_prior),
        "rank_lc0_top_in_xc0_prior": int(rank_lc0_top_in_xc0_prior),
    }
    out.update(lc0_policy_metrics)

    if lc0_top is not None and xc0_top is not None:
        out["q_lc0_top"] = float(lc0_qs_map.get(lc0_top, 0.0))
        out["q_xc0_top"] = float(xc0_q_map.get(xc0_top, 0.0))
    else:
        out["q_lc0_top"] = 0.0
        out["q_xc0_top"] = 0.0

    return out


def aggregate_chunk_metrics(pos_rows):
    if not pos_rows:
        return None

    df = pd.DataFrame(pos_rows)

    y = df["y_target"].to_numpy(dtype=float)
    v = df["v_pred"].to_numpy(dtype=float)

    mse = float(np.mean((v - y) ** 2))

    corr = 0.0
    if y.size >= 2 and np.std(y) > 0 and np.std(v) > 0:
        corr = float(np.corrcoef(v, y)[0, 1])

    keys = [
        "policy_ce",
        "uniform_ce",
        "ce_gain",
        "top1_exact",
        "avg_top_prob",
        "top1_mass",
        "top3_mass",
        "top5_mass",
        "prob_on_others",
        "mass_on_legal",
    ]
    out = {"value_mse": mse, "value_corr": corr}
    for k in keys:
        out[k] = float(df[k].mean())

    out["n_samples"] = int(df.shape[0])
    return out


def run_lc0_vs_xc0(
    pkl_paths,
    lc0_loc=LC0_LOC,
    nodes=10000,
    n_threads=4,
    break_every=5000,
    save_dir=None,
):
    all_pos_rows = []
    chunk_rows = []

    epoch_rows = []
    chunk_buf = []

    t0 = time.time()
    n_pos = 0

    def worker_init():
        get_thread_lc0_engine(lc0_loc, verbose=True, minibatch_size=64)

    def worker_done():
        close_thread_lc0_engine()

    def worker_task(pos):
        return summarize_one_position(pos, nodes=nodes)

    with ThreadPoolExecutor(max_workers=n_threads, initializer=worker_init) as ex:
        futs = []

        for path in pkl_paths:
            g = load_game_pkl(path)
            for pos in iter_positions_from_game(g):
                futs.append(ex.submit(worker_task, pos))

        for i, fut in enumerate(as_completed(futs), start=1):
            row = fut.result()
            all_pos_rows.append(row)
            chunk_buf.append(row)
            n_pos += 1

            if (break_every is not None) and (n_pos % break_every == 0):
                stats = aggregate_chunk_metrics(chunk_buf)
                if stats is not None:
                    stats["model_epoch"] = len(epoch_rows)
                    epoch_rows.append(stats)

                dt = time.time() - t0
                r = n_pos / max(EPS, dt)
                print(
                    "[lc0_eval] positions", n_pos,
                    "rate", round(r, 2), "/s",
                    "last_chunk_n", len(chunk_buf),
                )
                chunk_buf = []

    if chunk_buf:
        stats = aggregate_chunk_metrics(chunk_buf)
        if stats is not None:
            stats["model_epoch"] = len(epoch_rows)
            epoch_rows.append(stats)

    df_pos = pd.DataFrame(all_pos_rows)
    df_epoch = pd.DataFrame(epoch_rows)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        pos_path = os.path.join(save_dir, "lc0_vs_xc0_positions.pkl")
        ep_path = os.path.join(save_dir, "lc0_validation_like_epochs.pkl")

        with open(pos_path, "wb") as f:
            pickle.dump(df_pos, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(ep_path, "wb") as f:
            pickle.dump(df_epoch, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("[lc0_eval] wrote:", pos_path)
        print("[lc0_eval] wrote:", ep_path)

    return df_pos, df_epoch

#%%
game_logs = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/cf_10x256x5_deep_sims1/game_logs"
all_games = os.listdir(game_logs)
import random
random.shuffle(all_games)
some_games = all_games[:1000]
pkl_paths = [os.path.join(game_logs, sg) for sg in some_games]
# Example usage:
# pkl_paths = [
#     r"C:\...\some_game.pkl",
#     r"C:\...\some_other_game.pkl",
# ]
#
df_pos, df_epoch = run_lc0_vs_xc0(
    pkl_paths,
    nodes=5000,
    n_threads=8,
    break_every=1000,
    save_dir=None
)

df_pos.save("df_pos.csv", index=False)
df_epoch.save("df_epoch.csv", index=False)
#%%
