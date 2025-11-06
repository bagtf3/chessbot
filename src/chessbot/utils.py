import os
import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import math, random, time, pickle
from time import time as _now

import matplotlib.pyplot as plt
import seaborn as sns
import chess
import chess.engine
import chess.svg
from IPython.display import SVG, display, clear_output

from pyfastchess import Board as fastboard

from chessbot import SF_LOC
from chessbot import features as ft

from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


MATE_CP = 2500
CLIP_MAX = 1200
EPS = 1e-12

def cp_to_value(cp):
    #check for mates
    if cp.score() is None:
        return np.clip(cp.score(mate_score=16), -10, 10) / 10
        
    else:
        return np.clip(cp.score() / 1000, -0.95, 0.95)


def cp_to_value_tanh(cp, mid_cp=400.0):
    # scale so tanh(k * mid_cp) == 0.5  =>  k = atanh(0.5) / mid_cp
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


def score_cp_relative(pov_score, clipped=True, mate_cp=MATE_CP):
    scr = pov_score.relative
    return score_clipped(scr) if clipped else scr.score(mate_score=mate_cp)


uci_path_path =  r"C:/Users/Bryan/Data/chessbot_data/uci_paths3000.pkl"
with open(uci_path_path, "rb") as f:
    PATHS = pickle.load(f)
    

def rnd(x, n):
    return np.round(x, n)


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


def plot_training_progress(all_evals, max_cols=4, save_path=None):
    """
    Two figures:
      1) 1 row x 3 cols: total_loss, policy_logits, value_out (MA15)
      2) 2 rows x 3 cols: 6 plots:
         - topk (mean_top1/3/10 together, no MA)
         - legal_mass_mean (MA15)
         - mean_kl (MA15)
         - mean_pred_entropy (MA15)
         - value_corr (MA15)
         - value_bias (MA15)
    Saves to save_path and save_path_extra (same dir) or shows if save_path is None.
    """
    if "model_epoch" in all_evals.columns:
        all_evals = all_evals.drop(columns=["model_epoch"])

    # helper: safe get column values, returns None if missing
    def col_vals(name):
        return all_evals[name].values if name in all_evals.columns else None

    # First figure: 1x3 (MA15) — raw lines one color, MA lines another color
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 4))

    # choose stable colors: blue for raw, orange for MA (matches your attached plot)
    palette = sns.color_palette("tab10")
    RAW_COL = palette[0]
    MA_COL  = palette[1]   
    MA_ALPHA = 0.6         

    for ax, name in zip(axs1, cols1):
        y = col_vals(name)
        if y is None:
            ax.text(0.5, 0.5, f"missing: {name}", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = np.arange(len(y))
        # raw series (same color for all panels)
        ax.plot(x, y, label=name, color=RAW_COL)
        # MA15 (same MA color for all panels, slightly transparent)
        ma = pd.Series(y).rolling(15, min_periods=1).mean().values
        ax.plot(x, ma, lw=2, alpha=MA_ALPHA, label=f"{name} (MA15)", color=MA_COL)
        ax.set_title(name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    figs.append((fig1, axs1))

    # Second figure: 2x3
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 8))
    axs2 = axs2.flatten()

    # (1) topk together no MA
    ax = axs2[0]
    t1 = col_vals("mean_top1_mass")
    t3 = col_vals("mean_top3_mass")
    t10 = col_vals("mean_top10_mass")
    if t1 is None and t3 is None and t10 is None:
        ax.text(0.5, 0.5, "no top-k data", ha="center", va="center")
        ax.set_axis_off()
    else:
        # safe lengths (avoid evaluating numpy array truthiness)
        l1 = len(t1) if t1 is not None else 0
        l3 = len(t3) if t3 is not None else 0
        l10 = len(t10) if t10 is not None else 0
        maxlen = max(l1, l3, l10)
        x = np.arange(maxlen)

        if t1 is not None:
            ax.plot(np.arange(len(t1)), t1, label="top1", color=palette[0])
        if t3 is not None:
            ax.plot(np.arange(len(t3)), t3, label="top3", color=palette[1])
        if t10 is not None:
            ax.plot(np.arange(len(t10)), t10, label="top10", color=palette[2])

        # autoscale: use log if all tiny and >0
        maxv = 0.0
        for arr in (t1, t3, t10):
            if arr is not None and np.size(arr):
                try:
                    maxv = max(maxv, float(np.nanmax(arr)))
                except ValueError:
                    pass
        if 0 < maxv < 1e-3:
            ax.set_yscale("log")
            ax.set_ylim(max(1e-6, maxv * 1e-4), max(1e-3, maxv * 10.0))
        else:
            ax.set_ylim(0, max(1e-3, maxv * 1.2))
        ax.set_title("mean top-k mass (no MA)")
        ax.legend()

    # (2) legal_mass_mean MA15
    ax = axs2[1]
    y = col_vals("legal_mass_mean")
    if y is None:
        ax.text(0.5, 0.5, "missing: legal_mass_mean", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(y))
        ax.plot(x, y, label="legal_mass_mean")
        ma = pd.Series(y).rolling(15, min_periods=1).mean().values
        ax.plot(x, ma, lw=2, alpha=0.7, label="legal_mass_mean (MA15)")
        ax.set_title("legal_mass_mean")
        ax.legend(fontsize=8)

    # (3) mean_kl MA15
    ax = axs2[2]
    y = col_vals("mean_kl")
    if y is None:
        ax.text(0.5, 0.5, "missing: mean_kl", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(y))
        ax.plot(x, y, label="mean_kl")
        ma = pd.Series(y).rolling(15, min_periods=1).mean().values
        ax.plot(x, ma, lw=2, alpha=0.7, label="mean_kl (MA15)")
        ax.set_title("mean_kl")
        ax.legend(fontsize=8)

    # (4) mean_pred_entropy MA15
    ax = axs2[3]
    y = col_vals("mean_pred_entropy")
    if y is None:
        ax.text(0.5, 0.5, "missing: mean_pred_entropy",
                ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(y))
        ax.plot(x, y, label="mean_pred_entropy")
        ma = pd.Series(y).rolling(15, min_periods=1).mean().values
        ax.plot(x, ma, lw=2, alpha=0.7,
                label="mean_pred_entropy (MA15)")
        ax.set_title("mean_pred_entropy")
        ax.legend(fontsize=8)

    # (5) value_corr MA15
    ax = axs2[4]
    y = col_vals("value_corr")
    if y is None:
        ax.text(0.5, 0.5, "missing: value_corr", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(y))
        ax.plot(x, y, label="value_corr")
        ma = pd.Series(y).rolling(15, min_periods=1).mean().values
        ax.plot(x, ma, lw=2, alpha=0.7, label="value_corr (MA15)")
        ax.set_title("value_corr")
        ax.legend(fontsize=8)

    # (6) value_bias MA15
    ax = axs2[5]
    y = col_vals("value_bias")
    if y is None:
        ax.text(0.5, 0.5, "missing: value_bias", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(y))
        ax.plot(x, y, label="value_bias")
        ma = pd.Series(y).rolling(15, min_periods=1).mean().values
        ax.plot(x, ma, lw=2, alpha=0.7, label="value_bias (MA15)")
        ax.set_title("value_bias")
        ax.legend(fontsize=8)

    plt.tight_layout()
    figs.append((fig2, axs2))

    # Save or show
    if save_path is not None:
        # primary save
        root, ext = os.path.splitext(save_path or "")
        if ext == "":
            ext = ".png"
            root = save_path or "training_progress"
        out1 = root + ext
        figs[0][0].savefig(out1, dpi=150)
        plt.close(figs[0][0])
        # extra save
        out2 = root + "_extra" + ext
        figs[1][0].savefig(out2, dpi=150)
        plt.close(figs[1][0])
    else:
        # interactive show both
        for f, _ in figs:
            f.show()


def plot_pred_vs_true_grid(model, preds, X, y_true_dict, save_path=None):
    """
    Single figure: value (hexbin) + policy diagnostics for flat 4096 head.
    Returns a dict of numeric metrics (so caller can write them into df).
    Assumes exact keys:
      preds['policy_logits'] (N,4096), preds['value_out'] (N,1)
      y_true_dict['policy'] (N,4096), y_true_dict['value'] (N,)
    """
    
    EPS = 1e-12
    # value arrays (explicit)
    yp_v = np.asarray(preds['value_out'], dtype=np.float32).reshape(-1)
    yt_v = np.asarray(y_true_dict['value'], dtype=np.float32).reshape(-1)

    # policy logits -> softmax probs (explicit stable softmax)
    # X is the model input list/tuple: [boards, legal_mask]
    legal_mask = np.asarray(X[1])
    mask = (legal_mask > 0.5)
    
    logits = np.asarray(preds['policy_logits'], dtype=np.float64)  # (N,4096)
    masked = np.where(mask, logits, -np.inf)                       # illegal -> -inf
    row_max = np.max(masked, axis=1, keepdims=True)                # -inf if fully masked
    finite = np.isfinite(row_max).flatten()
    P_pred = np.zeros_like(masked, dtype=np.float64)
    
    # masked softmax leveraging np.exp(-np.inf) = 0
    if finite.any():
        shifted = masked[finite] - row_max[finite]                # stable shift
        exp_shift = np.exp(shifted)
        P_pred[finite] = exp_shift / (exp_shift.sum(axis=1, keepdims=True) + 1e-300)

    # true policy probs (explicit normalize)
    P_true = np.asarray(y_true_dict['policy'], dtype=np.float32)
    P_true = P_true / (P_true.sum(axis=1, keepdims=True) + EPS)

    # plotting canvas (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax_v = axs[0, 0]
    ax_scatter = axs[0, 1]
    ax_bar = axs[1, 0]
    ax_heat = axs[1, 1]

    # value hexbin
    yp = yp_v.reshape(-1)
    yt = yt_v.reshape(-1)
    n = min(len(yp), len(yt))
    if n > 0:
        yp, yt = yp[:n], yt[:n]
        hb = ax_v.hexbin(yt, yp, gridsize=80, mincnt=1, cmap="magma")
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax_v.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax_v.set_title("value: pred vs true (hexbin)")
        ax_v.set_xlabel("True")
        ax_v.set_ylabel("Pred")
        fig.colorbar(hb, ax=ax_v, fraction=0.046, pad=0.04)
    else:
        ax_v.text(0.5, 0.5, "no value data", ha="center", va="center")
        ax_v.set_axis_off()

    # ensure policy arrays exist (they do by contract)
    if P_pred.size == 0 or P_true.size == 0:
        ax_scatter.text(0.5, 0.5, "no policy head found", ha="center",
                        va="center")
        ax_bar.set_visible(False)
        ax_heat.set_visible(False)
        plt.tight_layout()
        if save_path:
            root, ext = os.path.splitext(save_path or "")
            out = root + "_value" + ext
            plt.savefig(out, dpi=150)
            plt.close(fig)
        else:
            plt.show(block=False)
        # return minimal metrics (N=0) to caller
        return {
            "N": 0, "value_mse": float("nan"), "value_bias": float("nan"),
            "value_corr": float("nan")
        }

    # policy diagnostics numbers
    N = P_pred.shape[0]
    idx_true_top1 = P_true.argmax(axis=1)
    true_top1_prob = P_true[np.arange(N), idx_true_top1]
    pred_on_true_top1 = P_pred[np.arange(N), idx_true_top1]
    pred_top1_prob = P_pred.max(axis=1)

    # scatter hexbin: pred_on_true vs true_top1_prob
    ax_scatter.hexbin(true_top1_prob, pred_on_true_top1, gridsize=80, mincnt=1)
    ax_scatter.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax_scatter.set_xlabel("true top1 prob")
    ax_scatter.set_ylabel("model prob on true top1")
    ax_scatter.set_title(
        "model prob on true top1 vs true top1 (hexbin)")

    # mean mass on target top-k (explicit)
    ranks_true = np.argsort(P_true, axis=1)  # ascending indices
    def mean_mass(k):
        idx = ranks_true[:, -k:]
        return np.mean([P_pred[i, idx[i]].sum() for i in range(N)])
    m1 = mean_mass(1)
    m3 = mean_mass(3)
    m10 = mean_mass(10)

    # draw bar with numeric labels (keeps simple)
    bars = ax_bar.bar(["top1", "top3", "top10"], [m1, m3, m10])
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_title("mean model mass on target top-k")
    for rect, v in zip(bars, (m1, m3, m10)):
        label = f"{v:.3e}" if v < 1e-3 else f"{v:.3f}"
        ax_bar.text(rect.get_x() + rect.get_width() / 2, v * 1.05,
                    label, ha="center", va="bottom", fontsize=8)

    # heatmap: mean 'from' map true || pred (8x8 each)
    P2_true = P_true.reshape((N, 64, 64))
    P2_pred = P_pred.reshape((N, 64, 64))
    
    mean_from_true = P2_true.sum(axis=2).mean(axis=0).reshape(8, 8)
    mean_from_pred = P2_pred.sum(axis=2).mean(axis=0).reshape(8, 8)

    combined = np.hstack([mean_from_true, mean_from_pred])
    im = ax_heat.imshow(
        combined, interpolation="nearest", aspect="auto", origin="lower"
    )

    h, w = combined.shape
    ax_heat.set_xticks(np.arange(w))
    ax_heat.set_yticks(np.arange(h))

    files = list("abcdefgh")
    xticks = files + [""] * 8   # left block labeled, right block blank
    ax_heat.set_xticklabels(xticks, rotation=90, fontsize=8)
    ax_heat.set_yticklabels([str(r) for r in range(1, 9)], fontsize=8)

    # optional light grid between squares
    ax_heat.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax_heat.grid(which="minor", color="w", linewidth=0.5, alpha=0.6)

    ax_heat.set_title("mean 'from' map: [true || pred]")
    ax_heat.axvline(7.5, color="w", linewidth=1)
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        root, ext = os.path.splitext(save_path or "")
        out = root + "_vp" + ext
        plt.savefig(out, dpi=150)
        plt.close(fig)
    else:
        plt.show(block=False)

    # ----- numeric metrics to return -----
    # KL (mean over batch, nats): mean sum P_true * (log P_true - log P_pred)
    kl_per = (P_true * (np.log(P_true + EPS) - np.log(P_pred + EPS))).sum(axis=1)
    mean_kl = np.mean(kl_per)

    # entropy of predictions (mean)
    ent_per = (-(P_pred * np.log(P_pred + EPS))).sum(axis=1)
    mean_entropy = np.mean(ent_per)
    
    ent_true_per = (-(P_true * np.log(P_true + EPS))).sum(axis=1)  # true entropy (nats)
    mean_true_entropy = np.mean(ent_true_per)

    # legal_mass_mean: use P_true>0 as proxy for legal moves
    legal_mask = (P_true > 0).astype(np.float32)
    legal_mass = (logits * legal_mask).sum(axis=1)
    legal_mass_mean = np.mean(legal_mass)

    # mean pred_on_true and mean pred_top1
    mean_highest_prob = np.mean(pred_top1_prob)

    # value metrics computed here (so caller doesn't need to)
    # align lengths
    nval = min(len(yp_v), len(yt_v))
    yp_val = yp_v[:nval]
    yt_val = yt_v[:nval]
    value_mse = np.mean((yt_val - yp_val) ** 2) if nval > 0 else float("nan")
    value_bias = np.mean(yp_val - yt_val) if nval > 0 else float("nan")
    if nval > 1 and yt_val.std() > 0 and yp_val.std() > 0:
        value_corr = np.corrcoef(yt_val, yp_val)[0, 1]
    else:
        value_corr = float("nan")

    metrics = {
        "N": int(N),
        "mean_kl": mean_kl,
        "mean_pred_entropy": mean_entropy,
        "mean_true_entropy": mean_true_entropy,
        "legal_mass_mean": legal_mass_mean,
        "mean_highest_prob": mean_highest_prob,
        "mean_top1_mass": m1,
        "mean_top3_mass": m3,
        "mean_top10_mass": m10,
        "value_mse": value_mse,
        "value_bias": value_bias,
        "value_corr": value_corr
    }

    return metrics


def score_game_data(model, X, Y_batch, save_path=None):
    """
    Run model.predict -> plot -> model.evaluate -> return a single-row
    DataFrame with Keras numeric outputs + our custom metrics.
    """
    raw_preds = model.predict(X, batch_size=512, verbose=0)
    preds = {name: raw_preds[i] for i, name in enumerate(model.output_names)}

    # Plot overview grid and get metrics (plot function returns metrics dict)
    metrics = plot_pred_vs_true_grid(model, preds, X, Y_batch, save_path=save_path)

    # Keras evaluate -> dataframe row (keep original columns)
    cols = ['total_loss'] + model.output_names
    eval_vals = model.evaluate(X, Y_batch, verbose=0)
    eval_df = pd.DataFrame(eval_vals, index=cols).T

    # Add our metrics as columns (explicit)
    for k, v in metrics.items():
        # if metric exists already in eval_df, overwrite; else create column
        eval_df[k] = v

    # Also print some chosen metrics for quick console feedback
    print("\n=== Extra Metrics ===")
    print(
        f"value MSE: {metrics.get('value_mse'):.5f} ",
        f"value corr: {metrics.get('value_corr'):.3f}"
    )
    
    print(
        f"N: {metrics.get('N')}  mean KL: {metrics.get('mean_kl'):.5f} ",
        f"mean top1 mass: {metrics.get('mean_top1_mass'):.5f}\n"
    )
    return eval_df


def log_and_plot_sf(intra_training_summaries, show=True, save_path=None):
    """
    Pretty-print the latest Stockfish summary and plot simple trends across retrains.

    Args:
        intra_training_summaries: {step: {"summary": {...}, "games": [...]}, ...}
        show: if True, plt.show() the figures (ignored if save_path is given)
        save_path: if set, save figures as f"{save_path}_cpl.png" and f"{save_path}_bmr.png"

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
    print(f"Overall mean CPL: {float(s['avg_overall_mean_cpl']):.3f}")
    print(f"  - White mean CPL: {float(s['avg_white_mean_cpl']):.3f}")
    print(f"  - Black mean CPL: {float(s['avg_black_mean_cpl']):.3f}")
    print(f"Overall best-move rate: {float(s['avg_overall_best_move_rate'])*100:.1f}%")
    print(f"  - White best-move rate: {float(s['avg_best_move_rate_white'])*100:.1f}%")
    print(f"  - Black best-move rate: {float(s['avg_best_move_rate_black'])*100:.1f}%")
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


def greedy_sf_tree_paths(n_positions=2000, multipv=4, max_depth=7, eval_thresh=150):
    """
    Return a list of UCI move lists (paths) from STARTPOS via greedy BFS.
    Includes STARTPOS as [] and all intermediate paths until n_positions reached.

    Args:
        n_positions : target number of positions (approx; includes STARTPOS)
        multipv     : how many top moves to expand per node
        max_depth   : Stockfish search depth for analysis
        eval_thresh : only expand moves within (best_cp - cp) <= eval_thresh
        sf_path     : path to stockfish binary; if None, uses chess.engine default

    Returns:
        paths : list[list[str]] such as:
            [ [],
              ['e2e4'], ['d2d4'], ['c2c4'], ['g1f3'],
              ['e2e4','d7d5'], ... ]
    """
    
    def short_fen(fen):
        return " ".join(fen.split(" ")[:4])
    
    # Seed: STARTPOS plus four common first moves
    start = chess.Board()
    seed_sans = ["e4", "d4", "Nf3", "c4"]

    paths = []                 # output paths
    seen = set()               # short-FEN dedup
    q = deque()                # queue of (board, path)

    # Add STARTPOS
    paths.append([])                       # []
    seen.add(short_fen(start.fen()))

    # Enqueue seeds
    for san in seed_sans:
        mv = start.parse_san(san)
        b2 = start.copy(); b2.push(mv)
        q.append( (b2, [mv.uci()]) )

    eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)

    try:
        while q and len(paths) < n_positions:
            board, path = q.popleft()
            key = short_fen(board.fen())
            if key in seen:
                continue

            seen.add(key)
            paths.append(path)

            if board.is_game_over():
                continue

            info = eng.analyse(
                board, chess.engine.Limit(depth=max_depth), multipv=multipv
            )
            if not info:
                continue
            
            best_cp = info[0]["score"].pov(board.turn).score(mate_score=1500)
            if best_cp is None:
                best_cp = 0

            for d in info:
                sc = d["score"].pov(board.turn).score(mate_score=1500)
                if sc is None:
                    continue
                if best_cp - sc > eval_thresh:
                    continue
                if "pv" not in d or not d["pv"]:
                    continue
                mv = d["pv"][0]
                b2 = board.copy()
                b2.push(mv)
                q.append( (b2, path + [mv.uci()]) )
    finally:
        eng.quit()

    return paths

    
def get_pre_opened_game(index=None):
    b = fastboard()

    if index is None:
        moves_to_play = random.choice(PATHS)
    else:
        try:
            moves_to_play = PATHS[index]
        except:
            print(
                f"No premove path found for {index}!",
                f"please choose 0 - {len(PATHS)-1}.",
                "Selecting random premove path"
            )
            moves_to_play = random.choice(PATHS)

    for mtp in moves_to_play:
        b.push_uci(mtp)
    return b


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


def make_piece_training_board():
    fens = {
        "rooks": "rrrrkrrr/pppppppp/8/8/8/8/PPPPPPPP/RRRRKRRR w - - 0 1",
        "bishops": "bbbbkbbb/pppppppp/8/8/8/8/PPPPPPPP/BBBBKBBB w - - 0 1",
        "knights": "nnnnknnn/pppppppp/8/8/8/8/PPPPPPPP/NNNNKNNN w - - 0 1",
        "only_pawns": "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
        "b_vs_k":"bbbbkbbb/pppppppp/8/8/8/8/PPPPPPPP/NNNNKNNN w - - 0 1",
        "extra_queen": 'qnb1kbnq/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1'
    }
    
    pick = random.choice(list(fens.keys()))
    board = chess.Board(fens[pick])
    if np.random.uniform() < 0.5:
        board = board.mirror()
    return fastboard(board.fen()), {"scenario": pick}
        

class GameGenerator(object):
    """ Curriculum game generator """
    def __init__(self, cfg):
        self.config = cfg
        self.game_types = list(self.config.game_probs.keys())

    def new_board(self, game_type=None):
        # sanity check
        if game_type is not None and game_type not in self.game_types:
            raise ValueError(
                f"Unknown game type {game_type}. "
                f"Pick one of {self.game_types}")

        # sample if not given
        if game_type is None:
            types, probs = zip(*self.config.game_probs.items())
            game_type = random.choices(types, weights=probs, k=1)[0]

        # dispatch
        if game_type == "pre_opened":
            board = get_pre_opened_game()
            meta = {"scenario": "pre_opened"}
            
        elif game_type == "random_init":
            plies = 2*np.random.randint(0, 4)
            board = random_init(plies)
            meta = {"scenario": "random_init", "start_plies": plies}
            
        elif game_type == "random_middle_game":
            # just more plies of random_init to land mid-game
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
            
        else:
            raise ValueError(f"unhandled game_type {game_type}")
        
        # quick check to make sure there are legal moves
        if board.legal_moves():
            return board, meta
        # otherwise look for a new board
        else:
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
    Per-(scenario,sf_bucket) stats (unchanged) + bot-vs-SF W/L/D totals.
    For the SF totals we assume `result` is WHITE-POV:
      r > 0 => white won, r < 0 => black won, r == 0 => draw
    This matches your own description for detecting bot wins vs SF.
    """
    def sf_bucket(vs_sf, sf_flag):
        if not vs_sf:
            return "none"
        return "white" if sf_flag else "black"

    stats = defaultdict(lambda: {"N": 0, "W": 0, "L": 0, "D": 0, "plies_sum": 0})

    # bot vs Stockfish totals only (what you care about)
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
    rows = sorted(stats.items(),
                  key=lambda kv: (kv[0][0], bucket_order.get(kv[0][1], 99)))

    return stats, rows, sf_overall


def print_recent_summary(recent, window=500, result_is_bot_pov=True):
    """
    Pretty-print the scenario table and the bot-vs-Stockfish W/L/D line,
    plus a wins-by-scenario breakdown (SF games only).
    """
    recent = recent[-window:]
    stats, rows, sf_overall = summarize_recent_games(
        recent, result_is_bot_pov=result_is_bot_pov
    )

    # scenario table (unchanged formatting)
    print(
        f"{'scenario':<20} {'sf':<6} {'N':>4} "
        f"{'W':>4} {'L':>4} {'D':>4}   {'avg_plies':>10}"
    )
    print("-" * 60)
    for (scenario, bucket), s in rows:
        avg = (s["plies_sum"] / s["N"]) if s["N"] else 0.0
        print(
            f"{scenario:<20} {bucket:<6} {s['N']:>4} "
            f"{s['W']:>4} {s['L']:>4} {s['D']:>4}  "
            f"{avg:>10.1f}"
        )
    print("-" * 60)

    # bot vs Stockfish summary (90-char lines)
    def pct(n, d): return (n / d) if d else 0.0
    total = sf_overall["N"]
    w, l, d = sf_overall["W"], sf_overall["L"], sf_overall["D"]
    win = pct(w, total)
    avg = (sf_overall["plies_sum"] / total) if total else 0.0
    print(
        f"Total SF games: {total:>4}  W/L/D={w}/{l}/{d}  ",
        f"win_rate={win:.1%}  avg_plies={avg:.1f}"
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
        # sort by descending wins, then alphabetically
        for s, c in sorted(wins_by_scenario.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {s:<30} {c}")
    print("~" * 60)