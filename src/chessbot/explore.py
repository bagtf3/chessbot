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
#%matplotlib inline
run_dir = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_selfplay_phase3"
all_games = load_game_index(run_dir)

CLIP_UB = 400
WINDOW = 2000

pkl_files = [f for f in os.listdir(run_dir) if "analyze_results_combined.pkl" in f]
if pkl_files:
    outfile = os.path.join(run_dir, pkl_files[0])
    with open(outfile, "rb") as fp:
        prev_run = pickle.load(fp)
        df_means = prev_run['df_means']

df_all = prev_run['df_all']

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

df_list.append(df_trim)

df_trim = pd.concat(df_list).drop_duplicates().sort_values("ts")
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
run_dir = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_test"
all_games = load_game_index(run_dir)

from chessbot.review import process_game_and_update_pkl
from pyfastchess import Board

def stm(b):
    return b.side_to_move() == 'w'


def cp_to_value(cp, mid_cp=400.0):
    # scale so tanh(k * mid_cp) == 0.5  =>  k = atanh(0.5) / mid_cp
    k = math.atanh(0.5) / mid_cp
    return math.tanh(k * cp)


def make_fake_visits(b, mv, lms, ratio_best=50):
    visits = [[mv, int(ratio_best)]]
    
    # may only be 1 legal move
    if len(lms) == 1:
        return visits
    
    ratio_not_best = 100-ratio_best
    sup_optimal = min(1, int(ratio_not_best / (len(lms) - 1)))
    visits += [[m, int(sup_optimal)] for m in lms if m != mv]
    return visits


def make_training_sample(b, v, visits):
    # turn counts into move probabilites
    ucis   = [x[0] for x in visits]
    counts = np.array([x[1] for x in visits], dtype=np.float32)
    s = counts.sum()
    pi = (counts / s) if s > 0.0 else None
    
    # labels per-legal
    fr, to, piece, promo = b.moves_to_labels(ucis=ucis)

    # allocate heads
    from_m = np.zeros(64, dtype=np.float32)
    to_m   = np.zeros(64, dtype=np.float32)
    pc_m   = np.zeros(6, dtype=np.float32)
    pr_m   = np.zeros(4, dtype=np.float32)

    # accumulate probs
    for i, p in enumerate(pi):
        from_m[fr[i]] += p
        to_m[to[i]]   += p
        pc_m[piece[i]]+= p
        pr_m[promo[i]]+= p

    # snapshot inputs and push example
    x = b.stacked_planes(5)
    
    return (x, {"from": from_m, "to": to_m, "piece": pc_m, "promo": pr_m, "v": v})
    
import chess
from chessbot import SF_LOC
from chessbot.utils import score_cp_white_pov
def sf_eval(b, engine=None):
    if not isinstance(b, chess.Board):
        b = chess.Board(b.fen())
        
    if engine is None:
        new_eng = True
        engine = chess.engine.SimpleEngine.popen_uci(SF_LOC)
        engine.configure({"Threads": 1, "Hash": 128})

    else:
        new_eng = False

    try:
        info = engine.analyse(
            b, limit=chess.engine.Limit(depth=DEPTH), info=chess.engine.INFO_ALL
        )
        
        score = score_cp_white_pov(info['score'])
        val = cp_to_value(score)
        best_move = info['pv'][0]

    except Exception as e:
        print(e)

    finally:
        if new_eng:
            engine.quit()
    return val, str(best_move)




DEPTH = 13
BLUNDER_THRESHOLD = 60

game_json_path = all_games[-2]['json_file']

game_data, analysis_out, combined = process_game_and_update_pkl(
    game_json_path, run_dir, engine=None, depth=DEPTH
)

def create_additional_training_data(analysis_out, game_data, engine=None):
    df = analysis_out['df']
    tree_data = game_data['tree_search_data']
    
    
    training_data = []
    vs_stockfish = game_data['vs_stockfish']
    sf_color = game_data['stockfish_is_white']
    b = Board(game_data['start_fen'])
    
    
    for i, mv in enumerate(game_data['moves_played']):
        if vs_stockfish and (sf_color == stm(b)):
            # we already have these, dont duplicate
            b.push_uci(mv)
            continue
        
        lms = b.legal_moves()
        # terminals are handled elsewhere
        if not lms:
            break
        
        tr = tree_data.get(i, tree_data.get(str(i), {}))
        cm = tr.get('candidate_moves', [])
        
        if cm:
            visits = [[c['uci'], c['visits']] for c in cm]
            visits = sorted(visits, key=lambda x: x[1], reverse=True)
        
        else:
            # make up fake visits if we dont have any
            visits = make_fake_visits(b, mv, lms, ratio_best=50)
        
        # find the df row associated to this move
        row = df.query("played_move == @mv")
        if len(row) > 1:
            row = df.query("played_move == @mv").query("move_num == @i")
        if len(row) == 0:
            stri = str(i)
            row = df.query("played_move == @mv").query("move_num == @stri")
        if len(row) == 0:
            print(f"no row found for {i}, {mv}")
        
        # happy path, not a blunder
        if row['delta'].item() < BLUNDER_THRESHOLD:
            v = cp_to_value(row['played_cp'].item())
            ts = make_training_sample(b, v, visits)
            training_data.append(ts)
            b.push_uci(mv)
        
        # if a blunder, dont use actual visits (theyre wrong)
        else:
            best_v = cp_to_value(row['best_cp'].item())
            best_mv = row['best_move'].item()
            best_visits = make_fake_visits(b, best_mv, lms)
            ts_best = make_training_sample(b, best_v, best_visits)
            training_data.append(ts_best)
            
            # furthermore, show the best continuation
            b2 = b.clone()
            b2.push_uci(best_mv)        
            cont_moves = 1
            ct = [best_mv]
            while cont_moves < 3 :
                lms2 = b2.legal_moves()
                if not lms2:
                    break
                cont_val, cont_best = sf_eval(b2)
                ct.append(cont_best)
                cont_visits = make_fake_visits(b2, cont_best, lms2)
                cont_ts = make_training_sample(b2, cont_val, cont_visits)
                training_data.append(cont_ts)
                b2.push_uci(cont_best)        
                cont_moves += 1
            
            # and the values of its PV to learn those positions are bad
            pv = tr.get('pv', [])
            # if no PV, just move on
            if not pv:
                b.push_uci(mv)
                continue
            
            # run the 3 PV moves
            b2 = b.clone()
            for m in pv[:3]:
                b2.push_uci(m['uci'])
                lms2 = b2.legal_moves()
                if not lms2:
                    break
                
                pv_val, pv_best = sf_eval(b2)
                pv_visits = make_fake_visits(b2, pv_best, lms2)
                pv_ts = make_training_sample(b2, pv_val, pv_visits)
                training_data.append(pv_ts)
            
            # push to move and let the loop roll over
            b.push_uci(mv)



        
    

        
    
    
        







