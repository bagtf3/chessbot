import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from scipy.stats import spearmanr
import pickle
from chessbot.utils import rnd
from chessbot.review import GameViewer, load_game_index

from pprint import pprint
from chessbot.review import combine_analysis_staging, ANALYZE_PKL


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
    - linear slope per 1000 games and per window (with block-permutation p)
    - Spearman rank correlation (monotonic trend)
    """
    d = df.sort_values("ts").reset_index(drop=True).copy()
    n = len(d)
    x = np.arange(n)

    cpl = d["overall_cpl"].rolling(window, center=True,
                                   min_periods=1).mean().values
    bmr = d["overall_best_move_rate"].rolling(
        window, center=True, min_periods=1).mean().values

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



def plot_and_report(df_trim, window):
    plot_cpl_and_bmr(df_trim, window=window)

    # compute aggregates
    overall_cpl = df_trim['overall_cpl'].mean()
    overall_bmr = df_trim['overall_best_move_rate'].mean()
    overall_top3 = df_trim['overall_top3_rate'].mean()

    first = df_trim.head(window)
    last = df_trim.tail(window)

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
    print(f"{'First ' + str(window):<{label_w}}{first_cpl:>{val_w}.3f}"
          f"{first_bmr:>{val_w}.3f}{first_top3:>{val_w}.3f}")
    print(f"{'Last ' + str(window):<{label_w}}{last_cpl:>{val_w}.3f}"
          f"{last_bmr:>{val_w}.3f}{last_top3:>{val_w}.3f}")
    print(f"{'Overall':<{label_w}}{overall_cpl:>{val_w}.3f}"
          f"{overall_bmr:>{val_w}.3f}{overall_top3:>{val_w}.3f}")
    print()
    
#%%
## Plot everything so far
CLIP_UB = 500

df_list = []
progress_list = []
epoch_counter = 0

root = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_selfplay"
suffixes = ["", "_phase2", "_phase3", "_phase4"]

for s in suffixes:
    rd = root + s
    _ = combine_analysis_staging(rd)
    all_games = load_game_index(rd)
    
    eval_progress = os.path.join(rd, "eval_progress.csv")
    progress_df = pd.read_csv(eval_progress)
    progress_df['model_epoch'] += 1 + epoch_counter
    epoch_counter = progress_df['model_epoch'].max()
    progress_list.append(progress_df)
    
    pkl = os.path.join(rd, ANALYZE_PKL)
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
    df_list.append(df_trim)
    del df_all
    del df_means
    del df_trim


df_trim = pd.concat(df_list).drop_duplicates(['game_id']).sort_values("ts")
WINDOW = int(len(df_trim) * 0.15 // 10 * 10)
print(len(df_trim), f"games completed. Using window size {WINDOW}")
print()

plot_and_report(df_trim, WINDOW)
pprint(trend_check(df_trim, window=WINDOW))

# all_evals = pd.concat(progress_list)
# from chessbot.utils import plot_training_progress
# plot_training_progress(all_evals, max_cols=4, save_path=None)
#%%
# plot single phase
run_dir = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_better_stopping_test"

all_games = load_game_index(run_dir)
CLIP_UB = 500

_ = combine_analysis_staging(run_dir)
pkl = os.path.join(run_dir, ANALYZE_PKL)
with open(pkl, "rb") as f:
    prev_run = pickle.load(f)

df_all = prev_run['df_all']
df_means = prev_run['df_means']
WINDOW = int(len(df_means) * 0.2 // 10 * 10)
print(len(df_means), f"games completed. Using window size {WINDOW}")
print()

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

plot_and_report(df_trim, WINDOW)
pprint(trend_check(df_trim, window=WINDOW))
#%%

d = prev_run['df_all']
scored_games = set(d.game_id.unique())
scored = [g for g in all_games if g['game_id'] in scored_games]
pre_opened = [g for g in scored if g['scenario'] == 'pre_opened']

gv = GameViewer(pre_opened[-1]['json_file'], sf_df=d); gv.replay()

g = pre_opened[-2]
gid = g['game_id']

gdf = d.query("game_id == @gid")
gdf['move_num']
gdf['move_num'] = gdf['move_num'].astype(int)
gdf.sort_values("move_num")


#%%








