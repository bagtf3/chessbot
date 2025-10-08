import json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import math
from scipy.stats import spearmanr
import pickle
from chessbot.utils import rnd


RUN_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_selfplay_phase2"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # First, try regular JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: parse line-by-line (JSONL / concatenated objects)
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items
    

def load_game_index(path=None):
    if path is None:
        path = os.path.join(RUN_DIR, "game_index.json")
        
    return load_json(path)


def plot_cpl_and_bmr(df, window=30, title=None):
    """
    Plot overall_cpl (left y, blue) and overall_best_move_rate (right y, orange)
    vs game index (sorted by ts). Shows raw (faint) + smoothed (bold).
    Renders to Spyder's Plots pane.
    """
    d = df.sort_values("ts").reset_index(drop=True).copy()
    x = range(len(d))

    cpl_s = d["overall_cpl"].rolling(window, center=True, min_periods=1).mean()
    bmr_s = d["overall_best_move_rate"].rolling(
        window, center=True, min_periods=1).mean()

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
    - linear slope per 100 games
    - start→end change (first/last decile means)
    - Spearman rank correlation (monotonic trend)
    """
    d = df.sort_values("ts").reset_index(drop=True).copy()
    x = np.arange(len(d))  # game index

    cpl = d["overall_cpl"].rolling(window, center=True, min_periods=1).mean().values
    bmr = d["overall_best_move_rate"].rolling(
        window, center=True, min_periods=1).mean().values

    def lin_slope_per100(y):
        # simple least-squares slope scaled per 100 games
        x0 = x - x.mean()
        b = (x0 @ (y - y.mean())) / (x0 @ x0)
        return rnd(100 * b, 5)

    def start_end_change(y, frac=0.1):
        k = max(1, int(len(y) * frac))
        return rnd(np.nanmean(y[-k:]) - np.nanmean(y[:k]), 5)

    # metrics
    out = {
        f"cpl_slope_per_{window}": lin_slope_per100(cpl),
        "cpl_start_to_end": start_end_change(cpl),
        "cpl_spearman": rnd(spearmanr(x, cpl, nan_policy="omit").statistic, 5),
        f"bmr_slope_per_{window}": lin_slope_per100(bmr),
        "bmr_start_to_end": start_end_change(bmr),
        "bmr_spearman": rnd(spearmanr(x, bmr, nan_policy="omit").statistic, 5)
    }
    return out


def add_points_vs_sf(df):
    """
    Adds two columns:
      - bot_result_vs_sf: {-1,0,1} from the BOT'S pov (NaN if not vs SF)
      - points_vs_sf: {0,0.5,1} from the BOT'S pov (NaN if not vs SF)
    Assumes df has: ['result', 'vs_stockfish', 'stockfish_color'].
    'result' is white-POV: -1 loss, 0 draw, +1 win.
    """
    d = df.copy()
    vs_sf = d["vs_stockfish"] == True


    sign = np.where(d["stockfish_color"] == True, -1.0, 1.0)
    bot_res = np.where(vs_sf, sign * d["result"].astype(float), np.nan)

    d["bot_result_vs_sf"] = bot_res
    d["points_vs_sf"] = (bot_res + 1.0) / 2.0  # -1/0/+1 -> 0/0.5/1
    return d


def rolling_points_vs_sf(df, window=200):
    """
    Returns a new DataFrame sorted by time with a rolling mean of points_vs_sf.
    NaNs (non-SF games) are ignored inside the window.
    """
    d = add_points_vs_sf(df).sort_values("ts").reset_index(drop=True)
    d["rolling_points_vs_sf"] = (d["points_vs_sf"].rolling(window, min_periods=1).mean())
    return d


def wilson(p, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    halfw  = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5) / denom
    return center - halfw, center + halfw


def plot_rolling_rates_with_ci(df, window=200):
    d = df[df.vs_stockfish].sort_values("ts").reset_index(drop=True).copy()
    sign = np.where(d.stockfish_color == True, -1.0, 1.0)
    res = sign * d.result.astype(float)
    win = (res == 1).astype(float)
    draw = (res == 0).astype(float)
    pts = (res + 1.0) / 2.0

    # trailing rolling stats
    p_win = win.rolling(window, center=True, min_periods=1).mean()
    p_draw = draw.rolling(window, center=True, min_periods=1).mean()
    m_pts = pts.rolling(window, center=True, min_periods=1).mean()
    s_pts = pts.rolling(window, center=True, min_periods=2).std()
    n_pts = pts.rolling(window, center=True, min_periods=1).count()

    # 95% normal CI for mean of points (bounded but fine as a quick approx)
    z = 1.96
    se = s_pts / np.sqrt(n_pts.replace(0, np.nan))
    lo = m_pts - z * se
    hi = m_pts + z * se

    x = range(len(d))
    plt.figure(figsize=(11, 5))
    plt.plot(x, p_win, label="win rate", color="tab:orange", lw=2)
    plt.plot(x, p_draw, label="draw rate", color="tab:blue", lw=2)
    plt.plot(x, m_pts, label="points (mean)", color="tab:green", lw=2)
    plt.fill_between(x, lo, hi, color="tab:green", alpha=0.15, linewidth=0)
    
    # auto-fit y with padding, clipped to [0,1]
    y_min = np.nanmin([p_win.min(), p_draw.min(), m_pts.min()])
    y_max = np.nanmax([p_win.max(), p_draw.max(), m_pts.max()])
    pad = 0.05 * (y_max - y_min if np.isfinite(y_max - y_min) and y_max > y_min else 1.0)
    plt.ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    
    plt.xlabel(f"Game # vs SF (trailing window={window})")
    plt.ylabel("Rate / Points")
    plt.title("Trailing win/draw; points with 95% CI")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()


def tail_vs_prev(df, window=200):
    def z_two_prop(p1, p2, n1, n2):
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        return (p2 - p1) / se if se else float("nan")

    def z_to_p(z):
        return math.erfc(abs(z) / math.sqrt(2.0))

    def z_two_mean(m1, m2, s1, s2, n1, n2):
        # Welch z (normal approx)
        se = math.sqrt((s1 * s1) / n1 + (s2 * s2) / n2)
        return (m2 - m1) / se if se else float("nan")
    
    d = df[df.vs_stockfish].sort_values("ts").reset_index(drop=True)
    d = d.copy()
    sign = np.where(d.stockfish_color == True, -1.0, 1.0)
    res = sign * d.result.astype(float)
    pts = (res + 1.0) / 2.0
    wins = (res == 1).astype(float)
    draws = (res == 0).astype(float)

    tail = slice(-window, None)
    prev = slice(-2 * window, -window)

    def mean_n(x):
        arr = np.asarray(x, dtype=float)
        return float(np.nanmean(arr)), int(np.sum(~np.isnan(arr)))

    def var_n(x):
        arr = np.asarray(x, dtype=float)
        return float(np.nanvar(arr, ddof=1)), int(np.sum(~np.isnan(arr)))

    rows = {}

    # points: use Welch z on 0/0.5/1
    m1, n1 = mean_n(pts[prev]); m2, n2 = mean_n(pts[tail])
    s1, _ = var_n(pts[prev]);   s2, _ = var_n(pts[tail])
    z = z_two_mean(m1, m2, math.sqrt(s1), math.sqrt(s2), n1, n2)
    rows["points"] = {
        "prev_mean": m1, "prev_n": n1,
        "tail_mean": m2, "tail_n": n2,
        "delta_mean": m2 - m1,
        "z": z, "p_two_sided": z_to_p(z),
    }

    # wins/draws: two-proportion z-tests
    for name, arr in (("win_rate", wins), ("draw_rate", draws)):
        p1, n1 = mean_n(arr[prev]); p2, n2 = mean_n(arr[tail])
        z = z_two_prop(p1, p2, n1, n2)
        rows[name] = {
            "prev_mean": p1, "prev_n": n1,
            "tail_mean": p2, "tail_n": n2,
            "delta_mean": p2 - p1,
        }

    return pd.DataFrame(rows).T.round(6)
#%%
from pprint import pprint
#%matplotlib inline
all_games = load_game_index()

df_games = pd.DataFrame.from_records(all_games)
df_games["ts"] = df_games["ts"].round().astype("int64")
df_games = rolling_points_vs_sf(df_games)

plot_rolling_rates_with_ci(df_games, window=250)
tail_vs_prev(df_games, window=250)

pkl_files = [f for f in os.listdir(RUN_DIR) if "analyze_results_combined.pkl" in f]
if pkl_files:
    outfile = os.path.join(RUN_DIR, pkl_files[0])
    with open(outfile, "rb") as fp:
        prev_run = pickle.load(fp)
        df_means = prev_run['df_means']

df_all = prev_run['df_all']
df_all['played_best_move'] = df_all.loss <= 10
bmr = df_all.groupby("game_id")['played_best_move'].mean()
df_all['clipped_loss'] = np.clip(df_all['loss'], 0, 600)
clipped_cpl = df_all.groupby("game_id")['clipped_loss'].mean()

df_trim = df_means.query("overall_cpl > 0").query("overall_cpl < 500")
df_trim['overall_best_move_rate'] = df_trim.game_id.map(bmr)
df_trim['overall_cpl'] = df_trim.game_id.map(clipped_cpl)

plot_cpl_and_bmr(df_trim, window=500)
print("Overall CPL", prev_run['summary']['avg_overall_mean_cpl'])
pprint(trend_check(df_trim, window=500))


#%%
d = prev_run['df_all']
from chessbot.review import GameViewer
all_games = load_game_index()
view = [g for g in all_games if (g['beat_sf']) and (g['scenario'] == 'b_vs_k')]
gv = GameViewer(view[-1]['json_file'], sf_df=d); gv.replay()


#%%

