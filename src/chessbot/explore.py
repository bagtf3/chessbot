import json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import math
from scipy.stats import spearmanr
import pickle
from chessbot.utils import rnd
from chessbot.review import GameViewer, load_json, load_game_index

RUN_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_selfplay_phase3"


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
    - linear slope per 100 games (with permutation p-value)
    - Spearman rank correlation (monotonic trend)
    """
    d = df.sort_values("ts").reset_index(drop=True).copy()
    x = np.arange(len(d))  # game index

    cpl = d["overall_cpl"].rolling(window, center=True, min_periods=1).mean().values
    bmr = d["overall_best_move_rate"].rolling(
        window, center=True, min_periods=1).mean().values

    def lin_slope_per100(y):
        x0 = x - x.mean()
        b = (x0 @ (y - y.mean())) / (x0 @ x0)
        return float(100 * b)

    # slopes
    cpl_s = lin_slope_per100(cpl)
    bmr_s = lin_slope_per100(bmr)

    # permutation p-values for slope (3000 perms, no seed)
    nperm = 3000
    abs_cpl = abs(cpl_s)
    abs_bmr = abs(bmr_s)
    cnt_c = 0
    cnt_b = 0
    x0 = x - x.mean()
    for _ in range(nperm):
        yp = np.random.permutation(cpl)
        bp = (x0 @ (yp - yp.mean())) / (x0 @ x0) * 100
        if abs(bp) >= abs_cpl: cnt_c += 1
        yp2 = np.random.permutation(bmr)
        bp2 = (x0 @ (yp2 - yp2.mean())) / (x0 @ x0) * 100
        if abs(bp2) >= abs_bmr: cnt_b += 1
    p_cpl = (cnt_c + 1) / (nperm + 1)
    p_bmr = (cnt_b + 1) / (nperm + 1)

    out = {
        "cpl_slope_per_100": rnd(cpl_s, 5),
        "cpl_slope_p": rnd(p_cpl, 5),
        "cpl_spearman": rnd(spearmanr(x, cpl, nan_policy="omit").statistic, 5),
        "bmr_slope_per_100": rnd(bmr_s, 5),
        "bmr_slope_p": rnd(p_bmr, 5),
        "bmr_spearman": rnd(spearmanr(x, bmr, nan_policy="omit").statistic, 5),
    }
    return out

df_list = []
#%%
from pprint import pprint
from chessbot.review import combine_analysis_staging, ANALYZE_PKL
#%matplotlib inline
run_dir = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_test"

all_games = load_game_index(run_dir)

CLIP_UB = 500
WINDOW = 500

_ = combine_analysis_staging(run_dir)
pkl = os.path.join(run_dir, ANALYZE_PKL)
with open(pkl, "rb") as f:
    prev_run = pickle.load(f)

df_all = prev_run['df_all']
df_means = prev_run['df_means']


# tidy up CPL
df_all['clipped_loss'] = np.clip(df_all['loss'], -1000, 1000)
clipped_cpl = df_all.groupby("game_id")['clipped_loss'].mean()

# BMR
bmr = df_all.groupby("game_id")['played_best_move'].mean()

#Top3
if 'in_top3' not in df_all.columns:
    df_all['in_top3'] = False
    
df_all['in_top3'] = df_all.in_top3 | df_all.played_best_move
t3r = df_all.groupby("game_id")["in_top3"].mean()

df_trim = df_means.copy()
df_trim['overall_best_move_rate'] = df_trim.game_id.map(bmr)
df_trim['overall_cpl'] = df_trim.game_id.map(clipped_cpl)
df_trim['overall_top3_rate'] = df_trim.game_id.map(t3r)

df_trim.overall_cpl = np.clip(df_trim.overall_cpl, 0, CLIP_UB)
df_trim = df_trim.sort_values('ts')

#df_list.append(df_trim)
#df_trim = pd.concat(df_list).drop_duplicates().sort_values("ts")

plot_cpl_and_bmr(df_trim, window=WINDOW)

# compute aggregates
overall_cpl = df_trim['overall_cpl'].mean()
overall_bmr = df_trim['overall_best_move_rate'].mean()
overall_top3 = df_trim['overall_top3_rate'].mean()

first = df_trim.head(WINDOW)
last = df_trim.tail(WINDOW)

first_cpl = first['overall_cpl'].mean()
first_bmr = first['overall_best_move_rate'].mean()
first_top3 = first['overall_top3_rate'].mean()

last_cpl = last['overall_cpl'].mean()
last_bmr = last['overall_best_move_rate'].mean()
last_top3 = last['overall_top3_rate'].mean()

# pretty print
label_w = 18
val_w = 10

print("#" * 60)
print(f"{'':<{label_w}}{'CPL':>{val_w}}{'BMR':>{val_w}}{'Top3':>{val_w}}")
print(f"{'-'*label_w}{'-'*val_w}{'-'*val_w}{'-'*val_w}")
print(f"{'First ' + str(WINDOW):<{label_w}}{first_cpl:>{val_w}.3f}"
      f"{first_bmr:>{val_w}.3f}{first_top3:>{val_w}.3f}")
print(f"{'Last ' + str(WINDOW):<{label_w}}{last_cpl:>{val_w}.3f}"
      f"{last_bmr:>{val_w}.3f}{last_top3:>{val_w}.3f}")
print(f"{'Overall':<{label_w}}{overall_cpl:>{val_w}.3f}"
      f"{overall_bmr:>{val_w}.3f}{overall_top3:>{val_w}.3f}")
print()

pprint(trend_check(df_trim, window=WINDOW))
#%%

d = prev_run['df_all']
worst = '7d6eaf91-e370-4d97-9ee3-8aa5fd08bd8e'
worst  = [g for g in all_games if g['game_id'] == worst]

wins = [g for g in all_games if (g['beat_sf'])]
wins = [g for g in wins if (g['scenario'] == 'piece_odds')]

scored_games = set(d.game_id.unique())
scored = [g for g in all_games if g['game_id'] in scored_games]
pre_opened = [g for g in scored if g['scenario'] == 'pre_opened']

gv = GameViewer(pre_opened[-1]['json_file'], sf_df=d); gv.replay()



#%%
