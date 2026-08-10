"""Analysis helpers for selfplay runs.

Extracted from scratch/explore.py so the scratch script stays a thin driver.
Plot helpers render to the active matplotlib backend (Spyder's Plots pane
when run there); they do not save files.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from chessbot.utils import rnd


def weight_moving_avg(df, col, N, weight_col="total_plies"):
    w = df[weight_col]
    vw = df[col] * w

    c = vw.rolling(N, center=True, min_periods=1).sum() / \
        w.rolling(N, center=True, min_periods=1).sum()

    t = vw.rolling(N, min_periods=1).sum() / \
        w.rolling(N, min_periods=1).sum()

    k = int(1.5 * N)

    out = c.copy()
    out.iloc[k:] = t.iloc[k:]

    # simple linear blend over last N/2 before cutoff
    b0 = max(k - N // 2, 0)
    idx = np.arange(b0, k)
    a = (idx - b0) / (k - b0)

    out.iloc[b0:k] = (1 - a) * c.iloc[b0:k].to_numpy() + \
                     a * t.iloc[b0:k].to_numpy()

    return out


def weighted_mean(df, col, weight_col):
    return np.sum(df[col] * df[weight_col]) / df[weight_col].sum()


# pale hues for the SF-depth background bands. Light enough to sit under the
# data, and none of them read as the blue / orange / green the CPL / BMR /
# ELO series use. Indexed with modulo -- depth only ever steps by 1, so
# consecutive bands can never collide; a repeat is 6 depths away.
DEPTH_BAND_COLORS = [
    "#e4ebf5",  # blue
    "#ece4f5",  # lavender
    "#e4f5ec",  # mint
    "#f5f0e0",  # sand
    "#f5e4e8",  # rose
    "#e0f2f5",  # cyan
]


def depth_runs(idxs, depths, n_end):
    """Merge consecutive validations at the same depth into one span, so a
    depth held for five checkpoints is one band rather than five."""
    runs, start_i = [], 0
    for i in range(1, len(depths) + 1):
        if i == len(depths) or depths[i] != depths[start_i]:
            x0 = idxs[start_i] if start_i else 0
            x1 = idxs[i] if i < len(idxs) else n_end
            runs.append((x0, x1, depths[start_i]))
            start_i = i
    return runs


def draw_depth_bands(ax, idxs, depths, n_end):
    runs = depth_runs(list(idxs), list(depths), n_end)
    uniq = sorted({r[2] for r in runs})
    for x0, x1, dp in runs:
        ax.axvspan(
            x0, x1, lw=0, zorder=0,
            color=DEPTH_BAND_COLORS[uniq.index(dp) % len(DEPTH_BAND_COLORS)],
        )
        if x0 > 0:
            ax.axvline(x0, color="0.55", lw=0.8, ls="--", alpha=0.7, zorder=1)
        ax.annotate(f"d{int(dp)}", ((x0 + x1) / 2, 0.995),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=9, color="0.35", fontweight="bold")


def sampleable_cols(rd, fname="config.yaml"):
    """Column names the memlog carries for a run's sampleable params.
    looper.py writes the sampled scalar under the param name, except
    move_sample_temp_range which it splits into _min/_max."""
    path = os.path.join(rd, fname)
    if not os.path.exists(path):
        return []

    samp = (yaml.safe_load(open(path)) or {}).get("sampleable") or {}
    cols = []
    for param in samp:
        if param == "move_sample_temp_range":
            cols += ["move_sample_temp_min", "move_sample_temp_max"]
        else:
            cols.append(param)
    return cols


def usable_split_cols(df, cols):
    """Trim to columns that are actually present and actually vary. Params
    added part-way through a run are NaN on older games, and a sampleable
    with one option left produces a single-value split worth nothing."""
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].dropna().nunique() < 2:
            continue
        out.append(c)
    return out


# counts print as whole numbers; rates and CPL want decimals
SPLIT_ROUNDING = {"sims_per_ply": 0, "sims_done": 0, "plies": 0}


def split_report(df, cols, stat_cols, weight_col="plies"):
    """Weighted split of stat_cols by each sampleable, N per bucket."""
    rounding = {c: SPLIT_ROUNDING.get(c, 3) for c in stat_cols}
    for col in cols:
        sub = df[df[col].notna()]
        if not len(sub):
            continue
        print(f"  {col}  ".center(72, "#"))
        stats = sub.groupby(col)[stat_cols + [weight_col]].apply(
            lambda d: pd.Series(
                {c: weighted_mean(d, c, weight_col) for c in stat_cols})
        ).round(rounding)
        stats = stats.join(sub.groupby(col).size().rename("N"))
        print(stats.sort_values(stat_cols[0]))
        print()


def plot_cpl_and_bmr(df, window=30, title=None):
    """
    Plot overall_cpl (left y, blue) and overall_best_move_rate (right y, orange)
    vs game index (sorted by ts). Shows raw (faint) + smoothed (bold).
    Renders to Spyder's Plots pane.
    """
    d = df.sort_values("ts").reset_index(drop=True).copy()
    x = range(len(d))
    
    #cpl_s = d["overall_cpl"].rolling(window, center=True, min_periods=1).mean()
    cpl_s = weight_moving_avg(d, "overall_cpl", window, "total_plies")
    # bmr_s = d["overall_best_move_rate"].rolling(
    #     window, center=True, min_periods=1).mean()
    bmr_s = weight_moving_avg(d, "overall_best_move_rate", window, "total_plies")

    fig, ax_l = plt.subplots(figsize=(12, 6))

    # CPL (left axis, BLUE)
    ax_l.plot(x, d["overall_cpl"].values, color="tab:blue", alpha=0.25, linewidth=1)
    l1, = ax_l.plot(
        x, cpl_s.values, color="tab:blue", linewidth=2, label="Overall CPL (smoothed)")
    
    ax_l.set_xlabel("Game # (sorted by time)")
    ax_l.set_ylabel("Overall CPL", color="tab:blue")
    ax_l.tick_params(axis="y", labelcolor="tab:blue")
    ax_l.grid(True, alpha=0.3)
    
    if title is not None:
        ax_l.set_title(title)

    # BMR (right axis, ORANGE)
    ax_r = ax_l.twinx()
    ax_r.plot(
        x, d["overall_best_move_rate"].values, color="tab:orange",
        alpha=0.25, linewidth=1)
    
    l2, = ax_r.plot(
        x, bmr_s.values, color="tab:orange",
        linewidth=2, label="Best Move Rate (smoothed)")
    
    ax_r.set_ylabel("Overall Best Move Rate", color="tab:orange")
    ax_r.tick_params(axis="y", labelcolor="tab:orange")

    fig.legend(handles=[l1, l2], loc="upper right")
    plt.tight_layout()
    plt.show()


def trend_check(df, window=100):
    """
    Sort by ts, smooth (centered rolling mean), then report:
    - linear slope per 1000 games and per window (with block-permutation p)
    - Spearman rank correlation (monotonic trend)
    """
    d = df.sort_values("ts").reset_index(drop=True).copy()
    n = len(d)
    x = np.arange(n)

    cpl = weight_moving_avg(d, "overall_cpl", window).values
    bmr = weight_moving_avg(d, "overall_best_move_rate", window).values

    x0 = x - x.mean()
    denom = float(x0 @ x0)

    def slope_per_game(y):
        # returns slope per one game (index unit)
        return float((x0 @ (y - y.mean())) / denom)

    # observed slopes (per game)
    cpl_s_pg = slope_per_game(cpl)
    bmr_s_pg = slope_per_game(bmr)

    # convert to requested units
    cpl_s_per_1000 = float(1000 * cpl_s_pg)
    cpl_s_per_window = float(window * cpl_s_pg)
    bmr_s_per_1000 = float(1000 * bmr_s_pg)
    bmr_s_per_window = float(window * bmr_s_pg)

    # permutation p-values via block shuffling (block size = window)
    nperm = 3000
    cnt_c = 0
    cnt_b = 0

    # build block slices
    blk = []
    if window <= 1 or window >= n:
        # degenerate: single block = whole series
        blk = [slice(0, n)]
    else:
        i = 0
        while i < n:
            blk.append(slice(i, min(i + window, n)))
            i += window

    for _ in range(nperm):
        # shuffle blocks to preserve short-range dependence
        order = np.arange(len(blk))
        np.random.shuffle(order)

        seq = np.empty_like(cpl)
        pos = 0
        for j in order:
            s = blk[j]
            ln = s.stop - s.start
            seq[pos:pos + ln] = cpl[s]
            pos += ln
        perm_cpl = seq

        order2 = np.arange(len(blk))
        np.random.shuffle(order2)
        seq2 = np.empty_like(bmr)
        pos = 0
        for j in order2:
            s = blk[j]
            ln = s.stop - s.start
            seq2[pos:pos + ln] = bmr[s]
            pos += ln
        perm_bmr = seq2

        # slope per game for permuted series, then scale to per1000
        bp_c = slope_per_game(perm_cpl) * 1000.0
        bp_b = slope_per_game(perm_bmr) * 1000.0

        if abs(bp_c) >= abs(cpl_s_per_1000):
            cnt_c += 1
        if abs(bp_b) >= abs(bmr_s_per_1000):
            cnt_b += 1

    p_cpl = (cnt_c + 1) / (nperm + 1)
    p_bmr = (cnt_b + 1) / (nperm + 1)

    out = {
        "cpl_slope_per_1000": rnd(cpl_s_per_1000, 5),
        "cpl_slope_per_window": rnd(cpl_s_per_window, 5),
        "cpl_slope_p": rnd(p_cpl, 5),
        "cpl_spearman": rnd(spearmanr(x, cpl, nan_policy="omit").statistic, 5),
        "bmr_slope_per_1000": rnd(bmr_s_per_1000, 5),
        "bmr_slope_per_window": rnd(bmr_s_per_window, 5),
        "bmr_slope_p": rnd(p_bmr, 5),
        "bmr_spearman": rnd(spearmanr(x, bmr, nan_policy="omit").statistic, 5),
    }
    return out


def report_cpl_and_bmr(df_trim, window):
    # compute aggregates
    overall_cpl = weighted_mean(df_trim, 'overall_cpl', 'total_plies')
    overall_bmr = weighted_mean(df_trim, 'overall_best_move_rate', 'total_plies')

    first = df_trim.head(window)
    last = df_trim.tail(window)
    
    first_cpl = weighted_mean(first, 'overall_cpl', 'total_plies')
    first_bmr = weighted_mean(first, 'overall_best_move_rate', 'total_plies')
    
    last_cpl = weighted_mean(last, 'overall_cpl', 'total_plies')
    last_bmr = weighted_mean(last, 'overall_best_move_rate', 'total_plies')

    # pretty print
    label_w = 18
    val_w = 10

    print("#" * 60)
    print(f"{'':<{label_w}}{'CPL':>{val_w}}{'BMR':>{val_w}}")
    print(f"{'-'*label_w}{'-'*val_w}{'-'*val_w}{'-'*val_w}")
    print(f"{'First ' + str(window):<{label_w}}{first_cpl:>{val_w}.3f}"
          f"{first_bmr:>{val_w}.3f}")
    print(f"{'Last ' + str(window):<{label_w}}{last_cpl:>{val_w}.3f}"
          f"{last_bmr:>{val_w}.3f}")
    print(f"{'Overall':<{label_w}}{overall_cpl:>{val_w}.3f}"
          f"{overall_bmr:>{val_w}.3f}")
    print()
    

def plot_and_report(df_trim, window):
    plot_cpl_and_bmr(df_trim, window=window)
    report_cpl_and_bmr(df_trim, window)


def plot_validation_with_elo(df_val, val_df, VAL_WINDOW, min_depth=8):
    """
    CPL (left), BMR (mid-right), Model ELO (far-right green).
    No interpolation of model_elo. X axis is game index sorted by timestamp.
    Fails loudly if assumptions are violated.
    """
    
    # first filter via min depth
    val_df_sorted = val_df.sort_values("ts").reset_index(drop=True).copy()
    val_df_sorted = val_df_sorted.query("depth >= @min_depth")
    
    # sort games by timestamp and use integer index for x-axis
    d = df_val.sort_values("ts").reset_index(drop=True).copy()
    x = np.arange(len(d))

    cpl_s = weight_moving_avg(d, "overall_cpl", VAL_WINDOW)
    bmr_s = weight_moving_avg(d, "overall_best_move_rate", VAL_WINDOW)

    # smooth model_elo across checkpoints (moving average on val_df)
    elo_window = min(50, int(len(val_df) * 0.2) + 1)
    val_df_sorted["elo_rm"] = val_df_sorted["model_elo"].rolling(
        elo_window, center=True, min_periods=1
    ).mean()

    # map each checkpoint to nearest game index (first game >= checkpoint ts)
    idxs = np.searchsorted(d["ts"].values, val_df_sorted["ts"].values, side="left")
    idxs = np.clip(idxs, 0, len(d) - 1)

    # figure
    fig, ax_l = plt.subplots(figsize=(12, 6))

    # SF depth as background bands. Drawn on ax_l at zorder 0 -- twin axes
    # paint in creation order, so putting these on ax3 would cover CPL/BMR.
    draw_depth_bands(ax_l, idxs, val_df_sorted["depth"].values, len(d))

    # CPL raw (faint) + smoothed (bold)
    ax_l.plot(x, d["overall_cpl"].values, color="tab:blue", alpha=0.25,
              linewidth=1, zorder=2)
    l1, = ax_l.plot(x, cpl_s.values, color="tab:blue", linewidth=2, zorder=3,
                    label="Overall CPL (smoothed)")
    ax_l.set_xlabel("Game # (sorted by time)")
    ax_l.set_ylabel("Overall CPL", color="tab:blue")
    ax_l.tick_params(axis="y", labelcolor="tab:blue")
    ax_l.grid(True, alpha=0.3)
    ax_l.set_xlim(0, len(d))

    # BMR raw (faint) + smoothed (bold) on twin axis
    ax_r = ax_l.twinx()
    ax_r.plot(x, d["overall_best_move_rate"].values, color="tab:orange",
              alpha=0.25, linewidth=1)
    l2, = ax_r.plot(x, bmr_s.values, color="tab:orange", linewidth=2,
                    label="Best Move Rate (smoothed)")
    ax_r.set_ylabel("Overall Best Move Rate", color="tab:orange")
    ax_r.tick_params(axis="y", labelcolor="tab:orange")

    # Model ELO plotted only at checkpoint indices (no value interpolation)
    ax3 = ax_l.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.plot(idxs, val_df_sorted["model_elo"].values, color="green",
             alpha=0.35, linewidth=1, marker="o", markersize=3)

    # score on each raw checkpoint. Depth is already carried by the bands, so
    # it stays out of the label; offsets alternate so neighbouring points at
    # the same elo do not print on top of each other.
    for i, (xi, yi, sc) in enumerate(zip(
        idxs, val_df_sorted["model_elo"].values, val_df_sorted["score"].values
    )):
        ax3.annotate(
            f"{sc:.3f}", (xi, yi), textcoords="offset points",
            xytext=(0, 8 if i % 2 == 0 else -13), ha="center",
            fontsize=7.5, color="green",
        )
    l3, = ax3.plot(idxs, val_df_sorted["elo_rm"].values, color="green",
                   linewidth=2, marker="o", markersize=4,
                   label=f"Model ELO (rm{elo_window})")
    ax3.set_ylabel("Model ELO", color="green")
    ax3.tick_params(axis="y", labelcolor="green")

    # combined legend
    lines = [l1, l2, l3]
    labels = ["Overall CPL (smoothed)", "Best Move Rate (smoothed)",
              f"Model ELO (ma{elo_window})"]
    fig.legend(lines, labels, loc="upper left")

    plt.title("CPL (left), BMR (middle-right), Model ELO (far-right)")
    plt.tight_layout()
