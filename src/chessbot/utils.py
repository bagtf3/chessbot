import os
import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import math, random, time, pickle
from pathlib import Path
from time import time as _now
from pathlib import Path

import matplotlib.pyplot as plt
import chess
import chess.engine
import chess.svg
import chess.pgn
from IPython.display import SVG, display, clear_output

from pyfastchess import Board as fastboard

from chessbot import SF_LOC
from chessbot import features as ft

from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import re
import io


MATE_CP = 2500
CLIP_MAX = 1200
EPS = 1e-12

def cp_to_value(cp):
    #check for mates
    if cp.score() is None:
        return np.clip(cp.score(mate_score=16), -10, 10) / 10
        
    else:
        return np.clip(cp.score() / 1000, -0.95, 0.95)


def cp_to_value_tanh(cp, mid_cp=200.0):
    # scale so tanh(k * mid_cp) == 0.5  ->  k = atanh(0.5) / mid_cp
    k = math.atanh(0.5) / mid_cp

    # clip so checkmates still look much better
    return np.clip(math.tanh(k * cp), -0.97, 0.97)


def score_to_value_white(board_score):
    # always look from whites perspective
    from_white = board_score.white()
    return cp_to_value(from_white)


def score_to_value_stm_pov(board_score):
    # always look from whites perspective
    rel_score = board_score.relative
    return cp_to_value(rel_score)

# another method of converting scores
def score_clipped(x, clip_max=CLIP_MAX):
    return np.clip(x.score(mate_score=MATE_CP), -clip_max, clip_max)


def score_cp_white_pov(pov_score, clipped=True, mate_cp=MATE_CP):
    scr = pov_score.white()
    return score_clipped(scr) if clipped else scr.score(mate_score=mate_cp)


def score_cp_stm_pov(pov_score, clipped=True, mate_cp=MATE_CP):
    scr = pov_score.relative
    return score_clipped(scr) if clipped else scr.score(mate_score=mate_cp)


def sf_eval(b, score_fn=score_to_value_stm_pov, depth=12, time_lim=None, engine=None):
    if not isinstance(b, chess.Board):
        b = chess.Board(b.fen())
        
    if engine is None:
        new_eng = True
        engine = chess.engine.SimpleEngine.popen_uci(SF_LOC)
        engine.configure({"Threads": 1, "Hash": 128})
    else:
        new_eng = False

    # dynamic limit for depth, time limit or both
    if depth is None:
        limit = chess.engine.Limit(time=time_lim)
    
    elif time_lim is None:
        limit = chess.engine.Limit(depth=depth)

    else:
        limit = chess.engine.Limit(depth=depth, time=time_lim)

    try:
        info = engine.analyse(b, limit=limit, info=chess.engine.INFO_ALL)

        val = score_fn(info['score'])
        best_move = info.get("pv", [""])[0]
        search_depth = info['depth']
    except Exception as e:
        print(e)

    finally:
        if new_eng:
            engine.quit()
    
    # if time given, return search depth
    if time_lim is None:
        return val, str(best_move)
    else:
        return val, str(best_move), search_depth


with open(os.getenv("PRE_OPENED_UCI_PATHS", ""), "rb") as f:
    PATHS = pickle.load(f)

with open(os.getenv("PRE_OPENED_UCI_PATHS_MINI", ""), "rb") as f:
    MINI_PATHS = pickle.load(f)

with open(os.getenv("UHO_PGN_PATH", ""), "r", encoding="utf-8", errors="replace") as f:
    PGN_TEXT = f.read()


def rnd(x, n):
    return np.round(x, n)


def maybe_random_from_list(item):
    if isinstance(item, (list, set)):
        return np.random.choice(item)
    return item


def show_board(board, flipped=False, sleep=0.1):
    clear_output(wait=True)
    display(SVG(chess.svg.board(board=board, flipped=flipped)))
    time.sleep(sleep)
    
    
def mirror_board(board):
    return board.mirror()


def mirror_move(mv: chess.Move) -> chess.Move:
    """Mirror a move through the board center (promotion piece is unchanged)."""
    return chess.Move(
        chess.square_mirror(mv.from_square),
        chess.square_mirror(mv.to_square),
        promotion=mv.promotion
    )


def make_random_move(board):
    if isinstance(board, chess.Board):
        moves = list(board.legal_moves)
        board.push(random.choice(moves))
        return board
    
    if isinstance(board, fastboard):
        moves = list(board.legal_moves())
        board.push_uci(random.choice(moves))
        return board    


def get_all_board_features(board):
    all_feats = ft.all_king_exposure_features(board)
    all_feats.update(ft.all_piece_features(board))
    
    wt, blk = ft.get_piece_value_sum(board)
    all_feats['material'] = [wt, blk]
    
    return all_feats


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    y = np.exp(x)
    s = np.sum(y)
    return y / s if s > 0 else np.full_like(y, 1.0 / len(y))


def calc_entropy(visits):
    # visits: sequence or ndarray of nonneg weights/probs
    if isinstance(visits, list):
        a = np.array(visits, dtype=float, copy=False)
    else:
        a = visits

    if a.size == 0:
        return 0.0, 0.0

    # force negatives to zero (defensive)
    a = np.where(a > 0.0, a, 0.0)

    total = a.sum()
    if total <= 0.0:
        return 0.0, 0.0

    p = a / total
    mask = p > 0.0
    p_mask = p[mask]

    # compute entropy only on positive probs to avoid log2(0)
    ent = - (p_mask * np.log2(p_mask)).sum()

    n = p.size
    norm = ent / np.log2(n) if n > 1 else 0.0

    return ent, norm


def kl_divergence(p_list, q_list):
    # p and q must be same length lists or arrays
    eps = 1e-12
    p = np.asarray(p_list, dtype=float) + eps
    q = np.asarray(q_list, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def kl_divergence_bits(p, q, eps=1e-12):
    """KL(p || q) in bits."""
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


def ensure_df(df_or_dicts):
    if isinstance(df_or_dicts, pd.DataFrame):
        return df_or_dicts.copy()
    return pd.DataFrame(df_or_dicts).copy()


def plot_sf_simple(df):
    """
    Line plot over epochs showing:
      - wins (total)
      - draws (total)
      - draws as White
      - draws as Black
      - average game length (plies) on a secondary axis
    """
    df = ensure_df(df).sort_values("epoch").reset_index(drop=True)

    ep = df["epoch"].astype(int).values
    wins_total   = df.get("wins",         0).fillna(0).astype(int).values
    draws_total  = df.get("draws",        0).fillna(0).astype(int).values
    white_draws  = df.get("white_draws",  0).fillna(0).astype(int).values
    black_draws  = df.get("black_draws",  0).fillna(0).astype(int).values
    avg_ply      = df.get("avg_ply",    np.nan).values / 5

    fig, ax1 = plt.subplots(figsize=(10,4))

    # counts (left axis)
    ax1.plot(ep, wins_total,  lw=1.5, label="Wins (total)")
    ax1.plot(ep, draws_total, lw=1.5, label="Draws (total)")
    ax1.plot(ep, white_draws, lw=1.2, alpha=0.9, label="Draws as White")
    ax1.plot(ep, black_draws, lw=1.2, alpha=0.9, label="Draws as Black")
    ax1.plot(ep, avg_ply,     lw=1.5, color='gray', label="Avg plies / 5")
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Counts")
    ax1.grid(True, alpha=0.3)

    # combined legend
    l1, lab1 = ax1.get_legend_handles_labels()
    ax1.legend(l1, lab1, loc="upper left")

    plt.title("SF600 exhibitions: wins/draws and avg game length")
    plt.tight_layout()
    plt.show()


def stable_softmax(logits):
    # logits: (B,4096) float32
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=1, keepdims=True)
    return e / (s + 1e-9)


def batch_policy_metrics(logits, labels, mask):
    eps = 1e-12
    big_neg = -1e6

    # mask logits, compute stable softmax
    masked_logits = np.where(mask > 0.5, logits, big_neg)
    probs = stable_softmax(masked_logits)

    # CE vs labels and vs uniform on legal set
    per_ce = -np.sum(labels * np.log(probs + eps), axis=1)
    policy_ce = per_ce.mean()

    legal_counts = mask.sum(axis=1, keepdims=True)
    uniform = mask / (legal_counts + eps)
    per_ce_uniform = -np.sum(labels * np.log(uniform + eps), axis=1)
    uniform_ce = per_ce_uniform.mean()

    exp_prob_model = np.sum(labels * probs, axis=1).mean()
    exp_prob_uniform = np.sum(labels * uniform, axis=1).mean()

    # best indices
    true_best = labels.argmax(axis=1)
    model_best = probs.argmax(axis=1)

    # top1 exact (model picks same argmax as labels)
    top1_exact = (model_best == true_best).mean()

    # avg prob of model's chosen best (keeps backward compat)
    top_probs = probs[np.arange(probs.shape[0]), model_best]
    avg_top_prob = top_probs.mean()

    # mass on label-defined top-1 / top-3 / top-5 (per-sample then batch mean)
    top1_per = probs[np.arange(probs.shape[0]), true_best]

    # get top-k indices by label (unsorted within the k)
    top3_idx = np.argpartition(-labels, 3 - 1, axis=1)[:, :3]
    top5_idx = np.argpartition(-labels, 5 - 1, axis=1)[:, :5]

    top3_per = np.take_along_axis(probs, top3_idx, axis=1).sum(axis=1)
    top5_per = np.take_along_axis(probs, top5_idx, axis=1).sum(axis=1)

    top1_mass = top1_per.mean()
    top3_mass = top3_per.mean()
    top5_mass = top5_per.mean()

    # other support / distribution metrics
    support_mask = labels > 0
    support_mask[np.arange(support_mask.shape[0]), true_best] = False
    prob_on_others_per = (probs * support_mask).sum(axis=1)
    prob_on_others = prob_on_others_per.mean()

    # fraction of raw (unmasked) softmax mass that lie on legal moves
    raw_probs = stable_softmax(logits)       # do not apply mask here
    mass_on_legal_per = np.sum(raw_probs * mask, axis=1)
    mass_on_legal = mass_on_legal_per.mean()

    return {
        "policy_ce": policy_ce,
        "uniform_ce": uniform_ce,
        "ce_gain": (uniform_ce - policy_ce),
        "exp_prob_model": exp_prob_model,
        "exp_prob_uniform": exp_prob_uniform,
        "top1_exact": top1_exact,
        "avg_top_prob": avg_top_prob,
        "top1_mass": top1_mass,
        "top3_mass": top3_mass,
        "top5_mass": top5_mass,
        "prob_on_others": prob_on_others,
        "mass_on_legal": mass_on_legal
    }


def print_validation(epoch, stats):
    eps = 1e-12

    keys = [
        "val_mse", "val_corr",
        "policy_ce", "uniform_ce", "ce_gain",
        "top1_exact", "avg_top_prob",
        "top1_mass", "prob_on_others"
    ]
    name_w = max(len(k) for k in keys)
    num_w = 8
    fmt_num = f"{{value:{num_w}.4f}}"
    def pair(k, v):
        return f"{k:<{name_w}}: {fmt_num.format(value=v)}"
    ratio = stats.get("top1_mass", 0.0) / (stats.get("prob_on_others", 0.0) + eps)

    print(f"[epoch {epoch:4d}] [validation] {pair('value_mse', stats['value_mse'])}  "
          f"{pair('value_corr', stats['value_corr'])}")
    print(f"[epoch {epoch:4d}] [validation] {pair('policy_ce', stats['policy_ce'])}  "
          f"{pair('uniform_ce', stats['uniform_ce'])}  ce_gain: {stats['ce_gain']:.4f}")
    print(f"[epoch {epoch:4d}] [validation] {pair('top1_exact', stats['top1_exact'])}  "
          f"{pair('avg_top_prob', stats['avg_top_prob'])}")
    print(f"[epoch {epoch:4d}] [validation] {pair('top1_mass', stats['top1_mass'])}  "
          f"{pair('true ratio', ratio)}")


def score_game_data(model, X, M, Y, epoch, save_path=None, preds=None):
    """ Run model.predict -> plot -> metrics -> return a single-row """
    if preds is None:
        preds = model.predict(X, verbose=0, batch_size=128)
    value_preds = preds[1].ravel()  # (B,) numpy from model.predict

    y_value = Y['value_out']
    targets = y_value.ravel()

    value_mse  = np.mean((value_preds - targets) ** 2)
    value_corr = np.corrcoef(value_preds, targets)[0, 1]

    plt.scatter(targets, value_preds, s=6)
    plt.plot([-1, 1], [-1, 1], linestyle="--", color="red", alpha=0.6)
    plt.xlim(-1, 1); plt.ylim(-1, 1); plt.gca()
    plt.xlabel("target"); plt.ylabel("pred"); plt.title("pred vs target")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

    policy_logits = preds[0]  # (B,4096)
    policy_true   = Y['policy_logits']
    policy_stats = batch_policy_metrics(policy_logits, policy_true, M)

    # build print dict and call the printer
    print_metrics = {"value_mse": value_mse, "value_corr": value_corr}
    print_metrics.update(policy_stats)
    print_validation(epoch, print_metrics)

    # create DF and return
    eval_df = pd.DataFrame([print_metrics])
    eval_df['model_epoch'] = epoch
    eval_df['n_samples'] = int(np.asarray(targets).shape[0])

    return eval_df


def moving_average_pd(arr, window=15):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().values


def plot_training_progress(metrics_history, epoch=None, save_path=None):
    """
    metrics_history: pd.DataFrame or dict-like with columns used below.
    If epoch is None, try to infer from metrics_history['model_epoch'].max(),
    otherwise use number of rows.
    Produces two figures and either shows them or writes:
      save_path           -> primary (2x2)
      save_path + "_extra"-> extra (1x3)
    """

    df = metrics_history.copy()

    # helper to safely extract column arrays (or empty list)
    def col_vals(name):
        if name in df.columns:
            return df[name].tolist()
        return []

    # infer epoch if needed
    if epoch is None:
        if "model_epoch" in df.columns and len(df):
            epoch = int(max(df["model_epoch"]))
        else:
            epoch = len(df)

    hide_first = 10
    if epoch < 12:
        return

    ma_window = min(30, max(3, int(epoch * 0.2)))
    if ma_window % 2 == 0:
        ma_window += 1

    # prepare x-axis baseline
    N = len(df)
    x_full = np.arange(N)
    start = hide_first
    xs = x_full[start:]

    # first figure (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # policy CE
    ax = axes[0, 0]
    raw = np.array(col_vals("policy_ce") or [])
    ma = moving_average_pd(raw, window=ma_window)[start:] if raw.size else np.array([])
    raw_seg = raw[start:] if raw.size else np.array([])
    if raw_seg.size:
        ax.plot(xs, raw_seg, label="policy_ce", alpha=0.6, lw=1)
    if ma.size:
        ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
    ax.set_title("policy CE (nats)")
    ax.legend()

    # CE gain vs uniform
    ax = axes[0, 1]
    raw = np.array(col_vals("ce_gain") or [])
    ma = moving_average_pd(raw, window=ma_window)[start:] if raw.size else np.array([])
    raw_seg = raw[start:] if raw.size else np.array([])
    if raw_seg.size:
        ax.plot(xs, raw_seg, label="ce_gain", alpha=0.6, lw=1)
    if ma.size:
        ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
    ax.set_title("CE gain vs uniform")
    ax.legend()

    # value MSE
    ax = axes[1, 0]
    raw = np.array(col_vals("value_mse") or [])
    ma = moving_average_pd(raw, window=ma_window)[start:] if raw.size else np.array([])
    raw_seg = raw[start:] if raw.size else np.array([])
    if raw_seg.size:
        ax.plot(xs, raw_seg, label="mse", alpha=0.6, lw=1)
    if ma.size:
        ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
    ax.set_title("value MSE")
    ax.legend()

    # value corr
    ax = axes[1, 1]
    raw = np.array(col_vals("value_corr") or [])
    ma = moving_average_pd(raw, window=ma_window)[start:] if raw.size else np.array([])
    raw_seg = raw[start:] if raw.size else np.array([])
    if raw_seg.size:
        ax.plot(xs, raw_seg, label="corr", alpha=0.6, lw=1)
    if ma.size:
        ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
    ax.set_title("value corr")
    ax.legend()
    plt.tight_layout()

    # Second figure (1x3)
    fig2, axs = plt.subplots(1, 3, figsize=(15, 4))

    # (0) left: top1 / top3 / top5 - show only MA
    ax = axs[0]
    t1 = col_vals("top1_mass")
    t3 = col_vals("top3_mass")
    t5 = col_vals("top5_mass")
    any_top = any(len(arr) for arr in (t1, t3, t5))
    if not any_top:
        ax.text(0.5, 0.5, "no top-k data", ha="center", va="center")
        ax.set_axis_off()
    else:
        l1, l3, l5 = f"top1 MA{ma_window}", f"top3 MA{ma_window}", f"top5 MA{ma_window}"
        if len(t1):
            ma_t1 = moving_average_pd(np.array(t1), window=ma_window)
            ax.plot(np.arange(len(ma_t1)), ma_t1, label=l1, linewidth=2)
        if len(t3):
            ma_t3 = moving_average_pd(np.array(t3), window=ma_window)
            ax.plot(np.arange(len(ma_t3)), ma_t3, label=l3, linewidth=2)
        if len(t5):
            ma_t5 = moving_average_pd(np.array(t5), window=ma_window)
            ax.plot(np.arange(len(ma_t5)), ma_t5, label=l5, linewidth=2)
        ax.set_title("mean top-k mass (MA shown)")
        ax.legend(fontsize=8)

    # (1) middle: mass_on_legal (raw + MA)
    ax = axs[1]
    mol = col_vals("mass_on_legal")
    if not len(mol):
        ax.text(0.5, 0.5, "missing: mass_on_legal", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(mol))
        ax.plot(x, mol, label="mass_on_legal", alpha=0.6, lw=1)
        ma_mol = moving_average_pd(np.array(mol), window=ma_window)
        ax.plot(x, ma_mol, lw=2, alpha=0.7, label=f"MA{ma_window}")
        ax.set_title("mass_on_legal")
        ax.legend(fontsize=8)

    # (2) right: avg_top_prob & top1_exact
    ax = axs[2]
    avg_tp = col_vals("avg_top_prob")
    t1_exact = col_vals("top1_exact")
    have_any = len(avg_tp) or len(t1_exact)
    if not have_any:
        ax.text(0.5, 0.5, "missing: avg_top_prob / top1_exact", ha="center", va="center")
        ax.set_axis_off()
    else:
        plotted = False
        if len(avg_tp):
            ma_avg = moving_average_pd(np.array(avg_tp), window=ma_window)
            l = f"avg_max_prob MA{ma_window}"
            ax.plot(np.arange(len(ma_avg)), ma_avg, label=l, alpha=0.9, lw=2)
            plotted = True
        if len(t1_exact):
            ma_t1ex = moving_average_pd(np.array(t1_exact), window=ma_window)
            l = f"top1_exact MA{ma_window}"
            ax.plot(np.arange(len(ma_t1ex)), ma_t1ex, label=l, alpha=0.9, lw=2)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "no data after smoothing", ha="center", va="center")
            ax.set_axis_off()
        else:
            ax.set_title("avg_max_prob & top1_exact (MA shown)")
            ax.legend(fontsize=8)
    plt.tight_layout()

    # save (primary and _extra)
    if save_path is not None:
        root, ext = os.path.splitext(save_path)
        if ext == "":
            ext = ".png"
            root = save_path
        out1 = root + ext
        out2 = root + "_extra" + ext
        fig.savefig(out1, dpi=150)
        plt.close(fig)
        fig2.savefig(out2, dpi=150)
        plt.close(fig2)
    else:
        fig.show()
        fig2.show()


def log_and_plot_sf(intra_training_summaries, show=True, save_path=None):
    """
    Pretty-print the latest Stockfish summary and plot simple trends across retrains.

    Args:
        intra_training_summaries: {step: {"summary": {...}, "games": [...]}, ...}
        show: if True, plt.show() the figures (ignored if save_path is given)
        save_path: if set, save figures as f"{save_path}_cpl.png", f"{save_path}_bmr.png"

    Prints:
        Latest step's games, CPL (overall/white/black), and best-move rates.
    Plots (if >=2 steps exist):
        1) Overall mean CPL vs step
        2) Overall best-move rate (%) vs step
    """
    its = intra_training_summaries
    
    if not its:
        print("No SF analysis yet.")
        return

    steps = sorted(its.keys())
    latest_step = steps[-1]
    s = its[latest_step]["summary"]

    # Pretty print latest
    print("\n=== Stockfish Analysis (latest) ===")
    print(f"Retrain step: {latest_step}")
    print(f"Games analyzed: {int(s['games'])}")
    print(f"Overall mean CPL: {s['avg_overall_mean_cpl']:.3f}")
    print(f"  - White mean CPL: {s['avg_white_mean_cpl']:.3f}")
    print(f"  - Black mean CPL: {s['avg_black_mean_cpl']:.3f}")
    print(f"Overall best-move rate: {s['avg_overall_best_move_rate']*100:.1f}%")
    print(f"  - White best-move rate: {s['avg_best_move_rate_white']*100:.1f}%")
    print(f"  - Black best-move rate: {s['avg_best_move_rate_black']*100:.1f}%")
    print("===================================\n")

    # Need >=2 points to plot a trend
    if len(steps) < 2:
        return

    # Build series
    overall_cpl = [its[k]["summary"]["avg_overall_mean_cpl"] for k in steps]
    overall_bmr_pct = [
        its[k]["summary"]["avg_overall_best_move_rate"] * 100.0 for k in steps
    ]

    # 1) Overall mean CPL trend
    fig1 = plt.figure(figsize=(6.4, 3.6))
    plt.plot(steps, overall_cpl, marker="o")
    plt.title("Overall Mean CPL vs Retrain Step")
    plt.xlabel("Retrain step")
    plt.ylabel("Mean CPL (lower is better)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        fig1.savefig(f"{save_path}_cpl.png", dpi=150)
    
    if show:
        plt.show()
    plt.close(fig1)

    # 2) Overall best-move rate (%) trend
    fig2 = plt.figure(figsize=(6.4, 3.6))
    plt.plot(steps, overall_bmr_pct, marker="o")
    plt.title("Overall Best-Move Rate vs Retrain Step")
    plt.xlabel("Retrain step")
    plt.ylabel("Best-move rate (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        fig2.savefig(f"{save_path}_bmr.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig2)


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.3f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.2f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.2f}s"


def make_jsonable(obj):
    """Recursively convert numpy types to built-in Python types so json.dump works.
    - ndarray -> list (obj.tolist())
    - numpy scalar -> int/float/bool via .item()
    - dict/list/tuple -> recurse
    Leaves normal Python objects untouched.
    """
    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # numpy scalar (int32, float64, bool_, ...)
    if isinstance(obj, (np.generic,)):
        try:
            return obj.item()
        except Exception:
            # fallback: cast with Python builtins
            if np.issubdtype(obj.dtype, np.integer):
                return int(obj)
            if np.issubdtype(obj.dtype, np.floating):
                return float(obj)
            if np.issubdtype(obj.dtype, np.bool_):
                return bool(obj)
            return str(obj)

    # dict: recurse
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}

    # list/tuple: recurse and keep type as list
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]

    # other objects: leave as-is (json.dump will fail if it's unsupported)
    return obj


def find_script(filename, start_file=None):
    """
    Locate filename in the same folder as start_file (or this file),
    then fallback to the current working directory.
    Returns the absolute path as a string.
    Raises FileNotFoundError if not found.
    """
    if start_file:
        base = Path(start_file).resolve().parent
    else:
        base = Path(__file__).resolve().parent

    candidate = base / filename
    if candidate.exists():
        return str(candidate)

    candidate = Path.cwd() / filename
    if candidate.exists():
        return str(candidate)

    raise FileNotFoundError(
        f"{filename} not found in {base} or cwd {Path.cwd()}"
    )


def random_init(plies=5, python_chess=False):
    b = fastboard()
    p = 0
    counter = 0
    limit = int(3*plies)
    while p < plies:
        # safety in case it gets caught in some weird corner
        counter += 1
        if counter > limit:
            # really bad luck, just try again
            return random_init(plies=plies)
        
        moves = b.legal_moves()
        while not len(moves):
            b.unmake()
            moves = b.legal_moves()
            p -= 1
            if p == 0:
                # more bad luck, startover
                return random_init(plies=plies)
        
        b.push_uci(random.choice(moves))
        p += 1
    
    if python_chess:
        b = chess.Board(b.fen())
    
    return b


def greedy_sf_tree_paths(n_pos=5000, multipv=4, thresh=90, margin=120):
    """
    Return a list of UCI move lists (paths) from STARTPOS via greedy BFS.
    Includes STARTPOS as [] and all intermediate paths until n_positions reached.

    Args:
        n_pos : target number of positions (approx; includes STARTPOS)
        multipv     : how many top moves to expand per node
        thresh : only expand moves within (best_cp - cp) <= eval_thresh
        margin : only add paths with abs(cb) within this margin

    Returns:
        paths : list[list[str]] such as:
            [
              ['e2e4'], ['d2d4'], ['c2c4'], ['g1f3'],
              ['e2e4','d7d5'], ... ]
    """
    
    def short_fen(fen):
        return " ".join(fen.split(" ")[:4])
    
    start = chess.Board()
    seed_sans = ["e4", "d4", "c4", "Nf3", "g3", "c3", "f4"]
    replies_sans = ["e5", "c5", "c6", "e6", "Nf6", "d5", "d6", "g6"]

    paths = []
    seen = set()
    q = deque()

    for san in seed_sans:
        mv = start.parse_san(san)
        b2 = start.copy(); b2.push(mv)
        q.append((b2, [mv.uci()]))
        for rep in replies_sans:
            mv_rep = b2.parse_san(rep)
            b3 = b2.copy(); b3.push(mv_rep)
            q.append((b3, [mv.uci(), mv_rep.uci()]))

    eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
    eng.configure({"Threads": 2, "Hash": 256})
    limit = chess.engine.Limit(depth=20, time=0.075)
    start_time = time.time()
    last_check_in = start_time
    try:
        while q and len(paths) < n_pos:
            now = time.time()
            if now - last_check_in > 20:
                rt = format_time(now - start_time)
                nstr = f"{len(paths)}/{n_pos} completed paths"
                pps = len(paths) / (now - start_time)
                print("[check in]", nstr, f"{pps:.3f} paths/sec. tot run time:", rt)
                last_check_in = now

            board, path = q.popleft()
            key = short_fen(board.fen())
            if key in seen:
                continue

            # For terminal boards, accept and don't expand
            if board.is_game_over():
                seen.add(key)
                paths.append(path)
                continue

            # analyse once, use result both for eval_margin check and expansion
            info = eng.analyse(board, limit=limit, multipv=multipv)
            if not info:
                continue

            # evaluation from White's POV (absolute position eval)
            score = info[0].get("score")
            eval_white = None
            if score is not None:
                eval_white = score.pov(chess.WHITE).score(mate_score=1500)

            # skip positions with mate or undefined score when using margin
            if margin is not None:
                if eval_white is None:
                    continue
                if abs(eval_white) > margin:
                    continue

            # passed margin (or margin disabled) -> accept path
            seen.add(key)
            paths.append(path)

            # now expand selected multipv moves (filtered by thresh)
            best_cp = info[0]["score"].pov(board.turn).score(mate_score=1500)
            if best_cp is None:
                best_cp = 0

            for d in info:
                sc = d["score"].pov(board.turn).score(mate_score=1500)
                if sc is None:
                    continue
                if best_cp - sc > thresh:
                    continue
                if "pv" not in d or not d["pv"]:
                    continue
                mv = d["pv"][0]
                b2 = board.copy()
                b2.push(mv)
                q.append((b2, path + [mv.uci()]))
    finally:
        eng.quit()

    return paths

    
def get_pre_opened_game(index=None, mini=False):
    b = fastboard()
    path_list = MINI_PATHS if mini else PATHS
    if index is None:
        moves_to_play = random.choice(path_list)
    else:
        try:
            moves_to_play = path_list[index]
        except:
            print(
                f"No premove path found for {index}!",
                f"please choose 0 - {len(path_list)-1}.",
                "Selecting random premove path"
            )
            moves_to_play = random.choice(path_list)

    for mtp in moves_to_play:
        b.push_uci(mtp)
    return b


def create_UHO_PGN_game():
    """
    Sample a game from PGN_TEXT (already loaded in memory), push mainline moves
    onto a fastboard(), and return the resulting board.

    index: 0-based game index, or None for random
    max_plies: cap number of half-moves played (None = all)
    """
    UHO_EVENT_START_RE = re.compile(r'(?m)^\[Event "')

    starts = [m.start() for m in UHO_EVENT_START_RE.finditer(PGN_TEXT)]
    if not starts:
        raise ValueError("No games found (no [Event at line start).")

    k = random.randrange(len(starts))
    a = starts[k]
    b = starts[k + 1] if k + 1 < len(starts) else len(PGN_TEXT)
    chunk = PGN_TEXT[a:b]

    blank = chunk.find("\n\n")
    moves_blob = chunk if blank == -1 else chunk[blank + 2:]

    game = chess.pgn.read_game(io.StringIO(moves_blob))
    if game is None:
        return fastboard()

    fb = fastboard()
    n = 0
    for mv in game.mainline_moves():
        fb.push_uci(mv.uci())
        n += 1

    return fb


def random_board_setup(pieces, wk=None, bk=None, queens=True, pyfast=True):
    """
    Make a legal endgame-like position with exactly `pieces` total pieces.
    Pieces are drawn from a bag proportional to a real starting set:
      per color: 8P, 2N, 2B, 2R, 1Q (plus the king already placed)
    Constraints:
      - pawns never on 1st/8th rank for their color
      - never exceed real caps per piece type per color
      - board is valid and not immediately game-over
      - if ensure_move, side to move has at least one legal move
    """
    if pieces < 6:
        raise ValueError("pieces must be >= 6")
    if pieces > 32:
        raise ValueError("pieces must be <= 32")

    # Max counts per color, mirroring real chess
    cap = {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
    }
    
    if not queens:
        cap[chess.QUEEN] = 0
    
    pool_piece_types = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN
    ]
    colors = [chess.WHITE, chess.BLACK]

    # Squares where pawns are allowed
    pawn_ok_squares = set([i for i in range(8, 56)])

    # Weighted draw from the remaining bag
    def draw_piece_type_and_color(rem_white, rem_black):
        bag = []
        # fill bag with counts so probability ∝ remaining allowed
        for pt in pool_piece_types:
            for _ in range(rem_white[pt]):
                bag.append((pt, chess.WHITE))
            for _ in range(rem_black[pt]):
                bag.append((pt, chess.BLACK))
        if not bag:
            return None
        return random.choice(bag)

    # Place a non-pawn on any empty square; pawn on allowed rank squares
    def place_piece(board, piece_type, color):
        empties = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]
        if not empties:
            return False

        if piece_type == chess.PAWN:
            candidates = [sq for sq in empties if sq in pawn_ok_squares]
        else:
            candidates = empties

        if not candidates:
            return False

        sq = random.choice(candidates)
        board.set_piece_at(sq, chess.Piece(piece_type, color))
        return True

    # Try until we get a valid, non-terminal position
    for _ in range(1000):
        board = chess.Board(None)

        # Kings first (any distinct squares is fine; validity checked later)
        
        wk = random.choice(chess.SQUARES) if wk is None else wk
        while True:
            bk = random.choice(chess.SQUARES) if bk is None else bk
            if bk != wk:
                break
            # prevents spinning forever
            bk= None

        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

        # Remaining counts per color
        rem_white = dict(cap)
        rem_black = dict(cap)

        # How many more to place
        need = pieces - 2
        ok_build = True

        while need > 0 and ok_build:
            choice = draw_piece_type_and_color(rem_white, rem_black)
            if choice is None:
                ok_build = False
                break

            pt, color = choice
            placed = place_piece(board, pt, color)
            if not placed:
                # If we failed to place this kind, remove this option once
                # by temporarily decrementing and continue; if it hits zero
                # it won't be drawn again.
                if color == chess.WHITE:
                    if rem_white[pt] > 0:
                        rem_white[pt] -= 1
                else:
                    if rem_black[pt] > 0:
                        rem_black[pt] -= 1
                continue

            # Successful placement consumes from that color's pool
            if color == chess.WHITE:
                rem_white[pt] -= 1
            else:
                rem_black[pt] -= 1
            need -= 1

        if not ok_build or need != 0:
            continue

        # Randomize side to move
        board.turn = random.choice(colors)

        # Final validity checks
        if not board.is_valid():
            # try to flip the turn
            board.turn = not board.turn
            if not board.is_valid():
                continue
        
        if board.is_game_over():
            continue
        if board.legal_moves.count() == 0:
            continue
        
        return fastboard(board.fen()) if pyfast else board
    
    #if we made it here, it couldnt work
    print("Unable to find valid board. Loosening requirements...")
    next_pieces = max(6, pieces-1)
    return random_board_setup(next_pieces, wk=None, bk=None, queens=False, pyfast=pyfast)


def make_piece_odds_board():
    b = chess.Board()
    meta = {"scenario": "piece_odds", "removed": {"white": [], "black": []}}

    remove_from = random.choice([chess.WHITE, chess.BLACK])
    removable = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    to_remove = random.choice(removable)
    need_to_remove = np.random.randint(1, 4) if to_remove == chess.PAWN else 1

    color_str = "white" if remove_from == chess.WHITE else "black"

    for _ in range(need_to_remove):
        # recompute candidates fresh each time (so we never hit an empty square)
        pool = [sq for sq, pc in b.piece_map().items()
                if pc.color == remove_from and pc.piece_type == to_remove]
        if not pool:
            break  # nothing left of this type/color. bail

        sq = random.choice(pool)
        pc = b.remove_piece_at(sq)
        meta["removed"][color_str].append(pc.symbol())

    return fastboard(b.fen()), meta


def random_backrow_fen():
    pieces = ["K", "Q", "R", "R", "B", "B", "N", "N"]

    # shuffle to get a random white backrank
    random.shuffle(pieces)
    white_back = "".join(pieces)

    random.shuffle(pieces)
    black_back = "".join(pieces).lower()

    # pawns and empty ranks standard; castling field '-' disables castling
    fen = f"{black_back}/pppppppp/8/8/8/8/PPPPPPPP/{white_back} w - - 0 1"
    return fen


def make_piece_training_board():
    fens = {
        "rooks": "rrrrkrrr/pppppppp/8/8/8/8/PPPPPPPP/RRRRKRRR w - - 0 1",
        "bishops": "bbbbkbbb/pppppppp/8/8/8/8/PPPPPPPP/BBBBKBBB w - - 0 1",
        "knights": "nnnnknnn/pppppppp/8/8/8/8/PPPPPPPP/NNNNKNNN w - - 0 1",
        "b_vs_k":"bbbbkbbb/pppppppp/8/8/8/8/PPPPPPPP/NNNNKNNN w - - 0 1",
        "extra_queen": 'qnb1kbnq/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1',
        "random_backrow": random_backrow_fen()
    }
    
    pick = random.choice(list(fens.keys()))
    board = chess.Board(fens[pick])
    if np.random.uniform() < 0.5:
        board = board.mirror()
    return fastboard(board.fen()), {"scenario": f"pt_{pick}"}
        

class UhoPgnSampler:
    def __init__(self, pgn_text):
        self.pgn_text = pgn_text
        self.uho_event_start_re = re.compile(r'(?m)^\[Event "')
        self.starts = [
            m.start()
            for m in self.uho_event_start_re.finditer(pgn_text)
        ]
        if not self.starts:
            raise ValueError("No games found (no [Event at line start).")

    def sample_fastboard(self):
        k = random.randrange(len(self.starts))
        a = self.starts[k]
        b = self.starts[k + 1] if k + 1 < len(self.starts) else len(self.pgn_text)
        chunk = self.pgn_text[a:b]

        blank = chunk.find("\n\n")
        moves_blob = chunk if blank == -1 else chunk[blank + 2:]

        game = chess.pgn.read_game(io.StringIO(moves_blob))
        if game is None:
            return fastboard()

        fb = fastboard()
        for mv in game.mainline_moves():
            fb.push_uci(mv.uci())

        return fb


class GameGenerator(object):
    """ Curriculum game generator """
    def __init__(self, cfg):
        self.config = cfg
        self.game_types = list(self.config.game_probs.keys())

        self.uho_sampler = None
        if "UHO" in self.game_types:
            self.uho_sampler = UhoPgnSampler(PGN_TEXT)

    def new_board(self, game_type=None):
        if game_type is not None and game_type not in self.game_types:
            raise ValueError(
                f"Unknown game type {game_type}. "
                f"Pick one of {self.game_types}"
            )

        if game_type is None:
            types, probs = zip(*self.config.game_probs.items())
            game_type = random.choices(types, weights=probs, k=1)[0]

        if game_type == "pre_opened":
            board = get_pre_opened_game()
            meta = {"scenario": "pre_opened"}

        elif game_type == "pre_opened_mini":
            board = get_pre_opened_game(mini=True)
            meta = {"scenario": "pre_opened_mini"}

        elif game_type == "UHO":
            board = self.uho_sampler.sample_fastboard()
            meta = {"scenario": "UHO"}

        elif game_type == "random_init":
            plies = 2 * np.random.randint(1, 5)
            board = random_init(plies)
            meta = {"scenario": "random_init", "start_plies": plies}

        elif game_type == "random_middle_game":
            plies = np.random.randint(20, 31)
            board = random_init(plies)
            meta = {"scenario": "random_middle_game", "start_plies": plies}

        elif game_type == "random_endgame":
            pieces = np.random.randint(8, 14)
            wk = np.random.randint(0, 33)
            bk = np.random.randint(33, 64)
            board = random_board_setup(pieces, wk, bk, queens=False)
            meta = {"scenario": "random_endgame", "pieces": pieces}

        elif game_type == "piece_odds":
            board, meta = make_piece_odds_board()

        elif game_type == "piece_training":
            board, meta = make_piece_training_board()
            meta["scenario"] = "piece_training"

        elif game_type == "startpos":
            board = fastboard()
            meta = {"scenario": "startpos"}

        else:
            raise ValueError(f"unhandled game_type {game_type}")

        if board.legal_moves():
            return board, meta

        return self.new_board(game_type=game_type)


def evaluate_game_sf(moves_uci, start_fen=None, depth=8, mate_cp=1500):
    def _score_pov_cp(pov_score, white_to_move, mate_cp):
        s = pov_score.white() if white_to_move else pov_score.black()
        return float(s.score(mate_score=mate_cp))

    board = chess.Board() if start_fen is None else chess.Board(start_fen)
    white_cpl, black_cpl = [], []
    best_moves = [0, 0]

    # diag counters
    drops = {
        "no_best_score": 0,
        "mv_illegal": 0,
        "no_played_score": 0,
        "engine_err": 0,
    }

    limit = chess.engine.Limit(depth=int(depth))
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
        for idx, u in enumerate(moves_uci):
            if board.is_game_over():
                break

            mv = None
            try:
                mv = chess.Move.from_uci(u)
            except Exception:
                mv = None

            try:
                root_for_best = board.copy(stack=False)
                root_for_play = board.copy(stack=False)
                white_to_move = root_for_best.turn

                bm = eng.play(root_for_best, limit=limit)
                best_uci = bm.move.uci() if (bm and bm.move) else None

                best_info = eng.analyse(
                    root_for_best, limit=limit,
                    info=chess.engine.INFO_SCORE
                )
                best_sc = best_info.get("score")
                best_cp = _score_pov_cp(best_sc, white_to_move, mate_cp)

                played_cp = None
                if mv is not None and mv in root_for_play.legal_moves:
                    played_info = eng.analyse(
                        root_for_play, limit=limit,
                        root_moves=[mv], info=chess.engine.INFO_SCORE
                    )
                    played_sc = played_info.get("score")
                    played_cp = _score_pov_cp(played_sc, white_to_move, mate_cp)
                else:
                    drops["mv_illegal"] += 1

                if best_uci is not None and best_uci == u:
                    best_moves[1 if white_to_move else 0] += 1

                if best_cp is None:
                    drops["no_best_score"] += 1
                if played_cp is None:
                    drops["no_played_score"] += 1

                if (best_cp is not None) and (played_cp is not None):
                    cpl = max(0.0, float(best_cp) - float(played_cp))
                    (white_cpl if white_to_move else black_cpl).append(cpl)

            except (chess.engine.EngineError, ValueError):
                drops["engine_err"] += 1

            # only push legal parsed moves
            if mv is not None and mv in board.legal_moves:
                board.push(mv)
            else:
                # stop if input game is desynced
                break

    # summaries (unchanged)
    def summary(arr):
        if not arr:
            return {"mean_cpl": 0.0, "max_cpl": 0.0, "n": 0}
        n = len(arr)
        return {
            "mean_cpl": round(sum(arr) / n, 3),
            "max_cpl": round(max(arr), 3),
            "n": n,
        }

    white = summary(white_cpl)
    black = summary(black_cpl)
    all_cpl = white_cpl + black_cpl
    overall = round(sum(all_cpl) / len(all_cpl), 3) if all_cpl else 0.0

    total_white_plies = max(1, white["n"])
    total_black_plies = max(1, black["n"])
    bm_rate_white = round(best_moves[1] / total_white_plies, 3)
    bm_rate_black = round(best_moves[0] / total_black_plies, 3)
    bm_overall = round(
        (best_moves[1] + best_moves[0]) /
        (total_white_plies + total_black_plies), 3
    )

    out = {
        "plies": white["n"] + black["n"],
        "white": white,
        "black": black,
        "overall_mean_cpl": overall,
        "best_move_rate_white": bm_rate_white,
        "best_move_rate_black": bm_rate_black,
        "overall_best_move_rate": bm_overall,
        "drops": drops,  # <-- diagnostics
    }
    return out


def evaluate_many_games(games, depth=12, workers=4, mate_cp=1500):
    def worker(moves, fen):
        return evaluate_game_sf(
            moves_uci=list(moves), start_fen=fen, depth=depth, mate_cp=mate_cp)

    N = len(games)
    results = [None] * N

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = {}
        for i, g in enumerate(games):
            futs[ex.submit(worker, g["moves"], g["start_fen"])] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = None

    # keep successful in-order
    results = [r for r in results if r is not None]

    # results is a list of per-game dicts from evaluate_game_sf
    if not results:
        return {"summary": {
            "games": 0,
            "avg_white_mean_cpl": 0.0,
            "avg_black_mean_cpl": 0.0,
            "avg_overall_mean_cpl": 0.0,
            "avg_best_move_rate_white": 0.0,
            "avg_best_move_rate_black": 0.0,
            "avg_overall_best_move_rate": 0.0,
        }, "games": []}
    
    w_means = [r["white"]["mean_cpl"] for r in results]
    b_means = [r["black"]["mean_cpl"] for r in results]
    o_means = [r["overall_mean_cpl"] for r in results]
    
    w_best = [r["best_move_rate_white"] for r in results]
    b_best = [r["best_move_rate_black"] for r in results]
    o_best = [r["overall_best_move_rate"] for r in results]
    
    def avg(xs): 
        return round(sum(xs) / len(xs), 3) if xs else 0.0
    
    summary = {
        "games": len(results),
        "avg_white_mean_cpl": avg(w_means),
        "avg_black_mean_cpl": avg(b_means),
        "avg_overall_mean_cpl": avg(o_means),
        "avg_best_move_rate_white": avg(w_best),
        "avg_best_move_rate_black": avg(b_best),
        "avg_overall_best_move_rate": avg(o_best),
    }
    
    return {"summary": summary, "games": results}


class RateMeter(object):
    def __init__(self, name, interval_s=120.0):
        self.name = name
        self.interval_s = interval_s
        self.t0 = _now()
        self.t_last = self.t0
        self.total = 0
        self.last_total = 0

    def tick(self, n=1):
        self.total += int(n)

    def rate(self):
        # instantaneous rate since last maybe_report (or since start if never called)
        t = _now()
        dtime = t - self.t_last
        dtotal = self.total - self.last_total

        # update for next time
        self.t_last = t
        self.last_total = self.total

        if dtime <= 0:
            return 0.0
        return dtotal / dtime

    def maybe_report(self, extra=""):
        t = _now()
        dt = t - self.t_last
        if dt < self.interval_s:
            return None
        
        rate = self.rate()
        msg = f"{self.name}: {rate:.1f}/s  (total {self.total})"
        if extra:
            msg += f"  {extra}"
        print(msg)
        return rate


def summarize_recent_games(recent, result_is_bot_pov=True):
    """
    Per-(scenario,sf_bucket) stats (unchanged) + bot-vs-SF W/D/L totals.
    For the SF totals we assume `result` is WHITE-POV:
      r > 0 => white won, r < 0 => black won, r == 0 => draw
    """
    def sf_bucket(vs_sf, sf_flag):
        if not vs_sf:
            return "none"
        return "white" if sf_flag else "black"

    stats = defaultdict(lambda: {"N": 0, "W": 0, "L": 0, "D": 0, "plies_sum": 0})

    # bot vs Stockfish totals only
    sf_overall = {"N": 0, "W": 0, "L": 0, "D": 0, "plies_sum": 0}

    for g in recent:
        scenario = g.get("scenario", "unknown")
        bucket = sf_bucket(g.get("vs_stockfish", False), g.get("stockfish_color"))
        key = (scenario, bucket)

        r = g.get("result", 0.0)

        # scenario table: keep your existing semantics (bot-POV by default)
        r_for_table = r
        if not result_is_bot_pov:
            sf_is_white = bool(g.get("stockfish_color"))
            bot_is_white = not sf_is_white
            r_for_table = r if bot_is_white else -r

        s = stats[key]
        s["N"] += 1
        plies = int(g.get("plies", 0))
        s["plies_sum"] += plies
        if r_for_table > 0:
            s["W"] += 1
        elif r_for_table < 0:
            s["L"] += 1
        else:
            s["D"] += 1

        # bot vs SF: count bot wins by checking if SF lost (WHITE-POV logic)
        if g.get("vs_stockfish", False):
            sf_overall["N"] += 1
            sf_overall["plies_sum"] += plies

            sf_is_white = bool(g.get("stockfish_color"))
            if r == 0:
                sf_overall["D"] += 1
            else:
                bot_won = (r < 0) if sf_is_white else (r > 0)
                if bot_won:
                    sf_overall["W"] += 1
                else:
                    sf_overall["L"] += 1

    bucket_order = {"white": 0, "black": 1, "none": 2}
    rows = sorted(
        stats.items(), key=lambda kv: (kv[0][0], bucket_order.get(kv[0][1], 99))
    )

    return stats, rows, sf_overall


def print_recent_summary(recent, window=2000, result_is_bot_pov=True):
    """
    Pretty-print the scenario table and the bot-vs-Stockfish W/D/L line,
    plus a wins-by-scenario breakdown (SF games only).
    """
    recent = recent[-window:]
    stats, rows, sf_overall = summarize_recent_games(
        recent, result_is_bot_pov=result_is_bot_pov
    )

    # scenario table (unchanged formatting)
    print(
        f"{'scenario':<20} {'sf':<6} {'N':>4} "
        f"{'W':>4} {'D':>4} {'L':>4}   {'avg_plies':>10}"
    )
    print("-" * 60)
    for (scenario, bucket), s in rows:
        avg = (s["plies_sum"] / s["N"]) if s["N"] else 0.0
        print(
            f"{scenario:<20} {bucket:<6} {s['N']:>4} "
            f"{s['W']:>4} {s['D']:>4} {s['L']:>4}  "
            f"{avg:>10.1f}"
        )
    print("-" * 60)

    # bot vs Stockfish summary (score: W=1, D=0.5, L=0)
    total = sf_overall["N"]
    w, d, l = sf_overall["W"], sf_overall["D"], sf_overall["L"]
    avg = (sf_overall["plies_sum"] / total) if total else 0.0

    score_pts = w + 0.5 * d
    score = (score_pts / total) if total else 0.0

    print(
        f"Total SF games: {total:>4}  W/D/L={w}/{d}/{l}  ",
        f"score={score:.3f}  avg_plies={avg:.1f}"
    )

    # compute counts: scenario -> wins (combine colors)
    wins_by_scenario = defaultdict(int)
    for g in recent:
        if not g.get("vs_stockfish", False):
            continue
        r = g.get("result", 0.0)
        if r == 0:
            continue
        sf_is_white = g.get("stockfish_color")
        bot_won = (r < 0) if sf_is_white else (r > 0)
        if bot_won:
            scenario = g.get("scenario", "unknown")
            wins_by_scenario[scenario] += 1

    if wins_by_scenario:
        print("\nWins by scenario (SF games only):")
        for s, c in sorted(
            wins_by_scenario.items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            print(f"  {s:<30} {c}")
    print("~" * 60)
