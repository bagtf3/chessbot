import os
import pandas as pd
import numpy as np
import matplotlib

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


def plot_validation_with_elo(df_val, val_df, VAL_WINDOW):
    """
    CPL (left), BMR (mid-right), Model ELO (far-right green).
    No interpolation of model_elo. X axis is game index sorted by timestamp.
    Fails loudly if assumptions are violated.
    """
    # sort games by timestamp and use integer index for x-axis
    d = df_val.sort_values("ts").reset_index(drop=True).copy()
    x = np.arange(len(d))

    # smoothed CPL and BMR (centered moving average)
    cpl_s = d["overall_cpl"].rolling(
        VAL_WINDOW, center=True, min_periods=1
    ).mean()
    bmr_s = d["overall_best_move_rate"].rolling(
        VAL_WINDOW, center=True, min_periods=1
    ).mean()

    # smooth model_elo across checkpoints (moving average on val_df)
    elo_window = min(50, int(len(val_df) * 0.2) + 1)
    val_df_sorted = val_df.sort_values("ts").reset_index(drop=True).copy()
    val_df_sorted["elo_rm"] = val_df_sorted["model_elo"].rolling(
        elo_window, center=True, min_periods=1
    ).mean()

    # map each checkpoint to nearest game index (first game >= checkpoint ts)
    idxs = np.searchsorted(d["ts"].values, val_df_sorted["ts"].values, side="left")
    idxs = np.clip(idxs, 0, len(d) - 1)

    # figure
    fig, ax_l = plt.subplots(figsize=(12, 6))

    # CPL raw (faint) + smoothed (bold)
    ax_l.plot(x, d["overall_cpl"].values, color="tab:blue", alpha=0.25, linewidth=1)
    l1, = ax_l.plot(x, cpl_s.values, color="tab:blue", linewidth=2,
                    label="Overall CPL (smoothed)")
    ax_l.set_xlabel("Game # (sorted by time)")
    ax_l.set_ylabel("Overall CPL", color="tab:blue")
    ax_l.tick_params(axis="y", labelcolor="tab:blue")
    ax_l.grid(True, alpha=0.3)

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


#%%
# plot single phase
run_dir = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_net_flat_run1"

all_games = load_game_index(run_dir)
CLIP_UB = 500

_ = combine_analysis_staging(run_dir)
pkl = os.path.join(run_dir, ANALYZE_PKL)
with open(pkl, "rb") as f:
    prev_run = pickle.load(f)

df_all = prev_run['df_all']
df_means = prev_run['df_means']

df_all = df_all.query("scenario != 'paired_validation'").copy()
df_means = df_means.query("scenario != 'paired_validation'").copy()

WINDOW = min(3000, max(5, int(len(df_means) * 0.2 // 10 * 10)))
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


df_all['blunder300'] = df_all.delta > 300
b300 = df_all.groupby("game_id")['blunder300'].mean()
df_all['blunder100'] = df_all.delta > 100
b100 = df_all.groupby("game_id")['blunder100'].mean()

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
scored = [g for g in all_games if not g['vs_stockfish'] and g['result'] != 0]
games = [g for g in scored if g['scenario'] == 'pre_opened']
gv = GameViewer(games[-5]['json_file'], sf_df=d); gv.replay()

#%%
## Plot everything so far
CLIP_UB = 500

df_list = []
progress_list = []
epoch_counter = 0

#root = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_selfplay"
#suffixes = ["", "_phase2", "_phase3", "_phase4"]
#root = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/new_conv_net_run"
#suffixes = ["0", "1"]
root = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_net_flat_run"
suffixes = ["0", "1"]
val_dfs = []
for s in suffixes:
    rd = root + s
    jsonl = os.path.join(rd, "validation_history.jsonl")
    if os.path.exists(jsonl):
        val_dfs.append(pd.read_json(jsonl, lines=True))
        
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
df_trim = df_trim.query("scenario != 'random_endgame'").copy()

df_val = df_trim.query("scenario == 'paired_validation'").copy()
df_trim = df_trim.query("scenario != 'paired_validation'").copy()

WINDOW = min(3500, int(len(df_trim) * 0.15 // 10 * 10))
print(len(df_trim), f"games completed. Using window size {WINDOW}")
print()

plot_and_report(df_trim, WINDOW)
pprint(trend_check(df_trim, window=WINDOW))

all_evals = pd.concat(progress_list)
from chessbot.utils import plot_training_progress
plot_training_progress(all_evals, save_path=None)

val_df = pd.concat(val_dfs)
VAL_WINDOW = min(3500, int(len(df_val) * 0.15 // 10 * 10))
plot_validation_with_elo(df_val, val_df, VAL_WINDOW)
#%%

d = prev_run['df_all']
scored_games = set(d.game_id.unique())
scored = [g for g in all_games if g['game_id'] in scored_games]
games = [g for g in scored if g['beat_sf']]
gv = GameViewer(games[-1]['json_file'], sf_df=d); gv.replay()
#%%
import chess
import chess.pgn


def moves_to_san_list(move_uci_list):
    """
    Given a list of UCI move strings, return the corresponding SAN list.
    Assumes the moves start from the standard initial position and are
    legal when played in sequence. Fails loudly otherwise.
    """
    board = chess.Board()
    san_list = []
    for uci in move_uci_list:
        mv = chess.Move.from_uci(uci)
        san = board.san(mv)
        san_list.append(san)
        board.push(mv)
    return san_list


def moves_san_list_to_pgn_text(san_list, headers=None, result="*"):
    """
    Format a list of SAN moves into a PGN string with optional headers.
    `headers` is a dict of header key->value (e.g., Event, White, Black).
    `result` is the game result string (\"1-0\", \"0-1\", \"1/2-1/2\" or \"*\").
    """
    if headers is None:
        headers = {}
    lines = []
    # write headers
    for k, v in headers.items():
        lines.append(f'[{k} "{v}"]')
    # ensure Result header present
    lines.append(f'[Result "{result}"]')
    lines.append("")  # blank line before moves

    # assemble move text with move numbers
    parts = []
    for i in range(0, len(san_list), 2):
        move_no = i // 2 + 1
        white = san_list[i]
        if i + 1 < len(san_list):
            black = san_list[i + 1]
            parts.append(f"{move_no}. {white} {black}")
        else:
            parts.append(f"{move_no}. {white}")
    moves_text = " ".join(parts)
    moves_text = moves_text.strip()
    if result:
        moves_text = moves_text + " " + result
    lines.append(moves_text)
    lines.append("")  # final newline
    return "\n".join(lines)


def board_history_uci_to_pgn(move_uci_list, headers=None, result="*"):
    """
    Convenience: uci-list -> PGN text. Uses a fresh board to compute SAN.
    """
    san_list = moves_to_san_list(move_uci_list)
    return moves_san_list_to_pgn_text(san_list, headers=headers, result=result)

# assume gv.log['history_uci'] is a list of UCI moves for one game
moves = gv.log["history_uci"]
hdrs = {"Event": "XecresValidation", "White": "SF14d3", "Black": "Xerces"}
pgn_text = board_history_uci_to_pgn(moves, headers=hdrs, result="0-1")
print(pgn_text)



#%%
cols = ['overall_best_move_rate', 'overall_cpl', 'overall_top3_rate']
b = df_trim.iloc[:-WINDOW, ]
before = b.groupby(['scenario'])[cols].mean()

a = df_trim.iloc[-WINDOW:, ]
after = a.groupby(['scenario'])[cols].mean()
print(f"Breakdown by scenario last {WINDOW} games")
print(after.sort_values("overall_cpl").round(3))

delta = (after-before).sort_values("overall_cpl", ascending=False).round(3)
print()
print(f"Delta by scenario last {WINDOW} games")
print(delta)
print()

from chessbot.utils import PATHS

first_move = ['e2e4', 'd2d4', 'c2c4', 'g1f3', 'g2g3']
second_move = ['e7e5', 'd7d5', 'c7c5', 'g8f6', 'c7c6', 'e7e6', 'd7d6']
l = []
l.append([])
for i in first_move:
    l.append([i])
    for j in second_move:
        l.append([i, j])


from pyfastchess import Board
for moves in l:
    b = Board()
    for move in moves:
        b.push_uci(move)
    
    if moves not in PATHS:
        print(moves)
        PATHS.append(moves)

uci_path_path =  r"C:/Users/Bryan/Data/chessbot_data/uci_paths3000_plus.pkl"
with open(uci_path_path, "wb") as f:
    pickle.dump(PATHS, f, protocol=pickle.HIGHEST_PROTOCOL)
