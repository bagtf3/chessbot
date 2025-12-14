import os, random
import pandas as pd
import numpy as np
import pickle
from chessbot.review import GameViewer, load_game_index
from chessbot.review import ANALYZE_PKL
from collections import defaultdict

SP_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs"
run_tags = [
    'conv_1000_selfplay_phase3','conv_1000_selfplay_phase4',
    "conv_net_flat_12blocks_run0", "conv_12x512SE",
    "conv_net_flat_run1", "conv_net_flat_run2"
]

all_games = []
df_list = []
for rt in run_tags:
    rd = os.path.join(SP_DIR, rt)
    all_games += load_game_index(rd)

    pkl = os.path.join(rd, ANALYZE_PKL)
    with open(pkl, "rb") as f:
        prev_run = pickle.load(f)
    
    df_all = prev_run['df_all']
    df_means = prev_run['df_means']
    
    # tidy up CPL
    df_all['clipped_loss'] = np.clip(df_all['loss'], -1000, 1000)
    clipped_cpl = df_all.groupby("game_id")['clipped_loss'].mean()

    df_means['overall_cpl'] = df_means.game_id.map(clipped_cpl)
    df_means['run_tag'] = rt
    df_list.append(df_means)
    del df_all
    del df_means

df_trim = pd.concat(df_list).drop_duplicates(['game_id']).sort_values("ts")
df_trim = df_trim.query("scenario != 'random_endgame'").copy()
df_trim = df_trim.query("scenario != 'paired_validation'").copy()


meta = pd.DataFrame(all_games)
meta = meta.query("plies >= 10")
exclude = ['random_endgame', 'paired_validation']
meta = meta.query("scenario != @exclude")

meta = meta.merge(df_trim[['game_id', 'run_tag', 'overall_cpl']], on='game_id')
meta = meta.query("overall_cpl <= 50")

training_games = meta['json_file'].to_list()

random.shuffle(training_games)
buffer = defaultdict(list)

thresh = 10000
idx = 0
draw_rate = 0.5
while len(buffer['X']) < thresh: 
    game = training_games[idx]
    idx += 1
    gv = GameViewer(game, sf_df=None)
        
    if gv.result == 0:
        if np.random.random() < draw_rate:
            continue
    if len(gv.moves_uci) < 10:
       continue
            
    X, M, P, Z, V, R = gv.generate_training_data(sf_skip=False)
    if X:
        buffer['X'] += X
        buffer['M'] += M
        buffer['P'] += P
        buffer['Z'] += Z
        buffer['V'] += V
        buffer['R'] += R
