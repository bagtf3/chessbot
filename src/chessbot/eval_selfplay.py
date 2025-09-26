import json, os, time
import pandas as pd
import numpy as np

import pickle
import chess
import chess.engine
from chessbot import SF_LOC
from chessbot.utils import rnd, format_time
from concurrent.futures import ProcessPoolExecutor, as_completed

RUN_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_selfplay"

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


def score_cp(pov_score, white_to_move, mate_cp):
    return pov_score.white().score(mate_score=mate_cp)


def analyze_with_sf_core(game_data, eng, depth=10):
    limit = chess.engine.Limit(depth=depth)
    board = chess.Board(game_data['start_fen'])
    sf_color = game_data['stockfish_is_white']

    cpl_s = cpl_w = cpl_b = 0.0
    nw = nb = 0
    rows = []

    for i, mv in enumerate(game_data['moves_played']):
        move = chess.Move.from_uci(mv)

        # skip SF plies
        if board.turn == sf_color:
            board.push(move)
            continue

        # best line (white-POV cp)
        sf_best = eng.play(board, limit=limit, info=chess.engine.INFO_ALL)
        best_move = sf_best.move
        best_cp = score_cp(sf_best.info['score'], board.turn, mate_cp=1500)
        best_relative = sf_best.info['score'].relative.score(mate_score=1500)

        # played move cp
        played_info = eng.analyse(
            board, limit=limit, root_moves=[move], info=chess.engine.INFO_SCORE
        )
        played_cp = score_cp(played_info["score"], board.turn, mate_cp=1500)

        delta_signed = best_cp - played_cp
        if board.turn:  # White to move
            loss_this = max(0, delta_signed); cpl_w += loss_this; nw += 1
        else:          # Black to move
            loss_this = max(0, -delta_signed); cpl_b += loss_this; nb += 1
        cpl_s += loss_this

        rows.append([
            i, mv, str(best_move), best_cp,
            delta_signed, played_cp, best_relative, board.turn, loss_this
        ])
        board.push(move)

    cols = [
        'move_num','played_move','best_move','best_cp',
        'delta','played_cp','relative_score','stm','loss'
    ]
    out_df = pd.DataFrame(rows, columns=cols)
    out_df["played_best_move"] = out_df["played_move"] == out_df["best_move"]
    
    # replace the three rate lines with this robust version
    mask_w = out_df["stm"] == True
    mask_b = out_df["stm"] == False
    
    overall_bmr = out_df["played_best_move"].mean() if len(out_df) else np.nan
    white_bmr = out_df.loc[mask_w, "played_best_move"].mean() if mask_w.any() else np.nan
    black_bmr = out_df.loc[mask_b, "played_best_move"].mean() if mask_b.any() else np.nan
    
    out = {
        "plies": nw + nb,
        "overall_cpl": rnd(cpl_s/(nw+nb), 3) if (nw+nb) else np.nan,
        "white_cpl":   rnd(cpl_w/nw, 3)      if nw else np.nan,
        "black_cpl":   rnd(cpl_b/nb, 3)      if nb else np.nan,
        "overall_best_move_rate": overall_bmr,
        "best_move_rate_white":    white_bmr,
        "best_move_rate_black":    black_bmr,
    }
    
    for key in ['game_id','scenario','stockfish_color','ts']:
        out[key] = game_data.get(key)
        out_df[key] = game_data.get(key)
    out['df'] = out_df
    return out


def analyze_with_sf(file_loc, depth=10):
    game_data = load_json(file_loc)
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
        try:
            # optional: avoid CPU oversubscription
            eng.configure({"Threads": 1})
        except Exception:
            pass
        return analyze_with_sf_core(game_data, eng, depth=depth)


def worker_shard(shard, depth):
    # one engine for the whole shard
    out = []
    eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
    try:
        try:
            eng.configure({"Threads": 1})
        except Exception:
            pass
        for g in shard:
            jd = load_json(g['json_file'])
            out.append(analyze_with_sf_core(jd, eng, depth=depth))
    finally:
        try: eng.close()
        except Exception: pass
    return out

    
def analyze_many_games(games, workers=4, depth=10):
    start = time.time()
    # shard games evenly across workers
    shards = [games[i::workers] for i in range(max(1,int(workers)))]
    results = []
    with ProcessPoolExecutor(max_workers=len(shards)) as ex:
        futs = [ex.submit(worker_shard, sh, depth) for sh in shards if sh]
        for fut in as_completed(futs):
            results.extend(fut.result())

    # same aggregation as your original
    w_means = [r["white_cpl"] for r in results]
    b_means = [r["black_cpl"] for r in results]
    o_means = [r["overall_cpl"] for r in results]
    w_best  = [r["best_move_rate_white"] for r in results]
    b_best  = [r["best_move_rate_black"] for r in results]
    o_best  = [r["overall_best_move_rate"] for r in results]

    summary = {
        "games": len(results),
        "avg_white_mean_cpl": rnd(np.nanmean(w_means), 3),
        "avg_black_mean_cpl": rnd(np.nanmean(b_means), 3),
        "avg_overall_mean_cpl": rnd(np.nanmean(o_means), 3),
        "avg_best_move_rate_white": rnd(np.nanmean(w_best), 3),
        "avg_best_move_rate_black": rnd(np.nanmean(b_best), 3),
        "avg_overall_best_move_rate": rnd(np.nanmean(o_best), 3),
    }
    
    df_all  = pd.concat([r.pop("df") for r in results], ignore_index=True)
    df_means = pd.DataFrame.from_records(results)
    if "ts" in df_means:
        df_means["ts"] = df_means["ts"].round().astype("Int64")
        
    stop = time.time()
    total = format_time(stop-start)
    print(f"Analyzed {len(games)} games in {total} seconds.")
    return summary, results, df_all, df_means


def combine_run_stats(previous, summary, results, df_all, df_means):
    new_results = previous['results'] + results
    new_df_all = pd.concat([previous['df_all'], df_all])
    new_df_means = pd.concat([previous['df_means'], df_means])
    
    w_means = [r["white_cpl"] for r in new_results]
    b_means = [r["black_cpl"] for r in new_results]
    o_means = [r["overall_cpl"] for r in new_results]
    
    w_best = [r["best_move_rate_white"] for r in new_results]
    b_best = [r["best_move_rate_black"] for r in new_results]
    o_best = [r["overall_best_move_rate"] for r in new_results]
    
    new_summary = {
        "games": len(new_results),
        "avg_white_mean_cpl": rnd(np.nanmean(w_means), 3),
        "avg_black_mean_cpl": rnd(np.nanmean(b_means), 3),
        "avg_overall_mean_cpl": rnd(np.nanmean(o_means), 3),
        "avg_best_move_rate_white": rnd(np.nanmean(w_best), 3),
        "avg_best_move_rate_black": rnd(np.nanmean(b_best), 3),
        "avg_overall_best_move_rate": rnd(np.nanmean(o_best), 3),
    }
    
    return {
        "summary": new_summary, "results": new_results,
        "df_all": new_df_all, "df_means": new_df_means
    }


if __name__ == '__main__':
    all_games = load_game_index()
    pkl_file = os.path.join(RUN_DIR, "analyze_results_combined.pkl")
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as fp:
            prev_run = pickle.load(fp)
            old_df_means = prev_run['df_means']
            games = [g for g in all_games if g['game_id'] not in set(old_df_means.game_id)]
            print(f"Found {len(games)} new games")
    else:
        prev_run = None
        games = all_games
        print(f"No pkl found. Running all {len(games)} games")

    summary, results, df_all, df_means = analyze_many_games(games, workers=6)
    
    if prev_run is not None:
        new_pkl = combine_run_stats(prev_run, summary, results, df_all, df_means)
    else:
        new_pkl = {
            "summary": summary, "results": results,
            "df_all":df_all, "df_means":df_means
        }
    
    with open(pkl_file, "wb") as f:
        pickle.dump(new_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
