import os
import pandas as pd
import numpy as np
import pickle
from chessbot.review import load_game_index
from chessbot.review import ANALYZE_PKL
import json


CPL_MIN = 60
screened_dir = "C:/Users/Bryan/Data/chessbot_data/training_data/screened_games"
SP_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs"

skip = ['cf_10x256x5_lc0_training', 'cf_10x256x5_low_sims']
run_tags = [rt for rt in os.listdir(SP_DIR) if rt not in skip]

total_deleted = 0
for rt in run_tags:
    print("="*60)
    print(f"[clean up] starting JSON trimming for {rt}")
    print("="*60)
    rd = os.path.join(SP_DIR, rt)
    all_games = load_game_index(rd)
    
    pkl = os.path.join(rd, ANALYZE_PKL)
    with open(pkl, "rb") as f:
        prev_run = pickle.load(f)
    
    df_all = prev_run['df_all']
    df_means = prev_run['df_means']
        
    # tidy up CPL
    df_all['clipped_loss'] = np.clip(df_all['loss'], -1000, 1000)
    clipped_cpl = df_all.groupby("game_id")['clipped_loss'].mean()
    df_means['overall_cpl'] = df_means.game_id.map(clipped_cpl)
    
    if 'plies' in df_means.columns:
        to_delete = df_means.query("not ((plies >= 10) and (overall_cpl <= @CPL_MIN))")
    else:
        to_delete = df_means.query("overall_cpl > @CPL_MIN")
        
    to_delete_set = set(to_delete['game_id'])
    
    all_games_df = pd.DataFrame(all_games)
    all_games_df['overall_cpl'] = all_games_df['game_id'].map(clipped_cpl)
    all_games = all_games_df.to_dict(orient="records")
    
    n_deleted = 0
    missing_json = 0
    for a in all_games:
        j = a.get('json_file', None)
        if j is None:
            continue
        
        if not os.path.exists(j):
            a['json_file'] = None
            print(f"[not found] {j}")
            missing_json += 1
        
        elif a['game_id'] in to_delete_set:
            os.remove(j)
            n_deleted += 1
            total_deleted += 1
            a['json_file'] = None
    
    print(f"found {missing_json} missing reconds from {rt}")
    print(f"deleted {n_deleted} records from {rt}")
    print()
    
    # backup original and write new JSONL atomically
    index_path = os.path.join(rd, "game_index.json")
    bak = index_path + ".bak"
    os.replace(index_path, bak)
    tmp = index_path + ".tmp"
    
    with open(tmp, "w", encoding="utf-8") as out:
        for a in all_games:
            out.write(json.dumps(a, ensure_ascii=False) + "\n")
            
    os.replace(tmp, index_path)
    print(f"wrote {len(all_games)} entries back to {index_path}; original -> {bak}")
    print("-"*60)
    print()
    
print(f"total deleted {total_deleted} records")
    
    

