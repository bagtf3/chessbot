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


def col_trend(df, col, window=100, weight_col="total_plies"):
    """Sort by ts, smooth (weighted if weight_col is present, else plain
    centered rolling mean), then report for one column vs row index:
    - linear slope per 10% of the series (rounded to the nearest row) and
      per window (with block-permutation p)
    - Spearman rank correlation (monotonic trend)
    Single-metric core shared by trend_check and print_trend. Scaling the
    slope to 10% of whatever n is, rather than a fixed row count, keeps it
    readable across wildly different series lengths (selfplay games vs
    validation checkpoints) without per-caller tuning.
    """
    d = df.sort_values("ts").reset_index(drop=True).copy()
    n = len(d)
    x = np.arange(n)
    per = max(1, round(n * 0.1))

    if weight_col in d.columns:
        y = weight_moving_avg(d, col, window, weight_col).values
    else:
        y = d[col].rolling(window, center=True, min_periods=1).mean().values

    x0 = x - x.mean()
    denom = float(x0 @ x0)

    def slope_per_row(v):
        # returns slope per one row (index unit)
        return float((x0 @ (v - v.mean())) / denom)

    s_pg = slope_per_row(y)
    s_per = float(per * s_pg)
    s_per_window = float(window * s_pg)

    # permutation p-value via block shuffling (block size = window)
    nperm = 3000
    cnt = 0

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
        order = np.arange(len(blk))
        np.random.shuffle(order)
        seq = np.empty_like(y)
        pos = 0
        for j in order:
            s = blk[j]
            ln = s.stop - s.start
            seq[pos:pos + ln] = y[s]
            pos += ln
        bp = slope_per_row(seq) * per
        if abs(bp) >= abs(s_per):
            cnt += 1

    p = (cnt + 1) / (nperm + 1)

    return {
        "slope_per": rnd(s_per, 5),
        "per": per,
        "slope_per_window": rnd(s_per_window, 5),
        "slope_p": rnd(p, 5),
        "spearman": rnd(spearmanr(x, y, nan_policy="omit").statistic, 5),
    }


def print_trend(label, df, col, window=100, weight_col="total_plies"):
    """One line: direction and significance of col's movement vs row index."""
    t = col_trend(df, col, window, weight_col)
    arrow = ("up" if t["slope_per"] > 0
             else "down" if t["slope_per"] < 0 else "flat")
    print(f"  {label:<24} slope/{t['per']}={t['slope_per']:>9.4f}  "
          f"p={t['slope_p']:.3f}  spearman={t['spearman']:+.3f}  ({arrow})")


def trend_check(df, window=100):
    """cpl/bmr trend vs game index -- see col_trend for the single-metric
    core this wraps."""
    cpl = col_trend(df, "overall_cpl", window)
    bmr = col_trend(df, "overall_best_move_rate", window)
    return {
        "cpl_slope_per": cpl["slope_per"],
        "cpl_per": cpl["per"],
        "cpl_slope_per_window": cpl["slope_per_window"],
        "cpl_slope_p": cpl["slope_p"],
        "cpl_spearman": cpl["spearman"],
        "bmr_slope_per": bmr["slope_per"],
        "bmr_per": bmr["per"],
        "bmr_slope_per_window": bmr["slope_per_window"],
        "bmr_slope_p": bmr["slope_p"],
        "bmr_spearman": bmr["spearman"],
    }


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
