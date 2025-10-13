import json, os, time
import pandas as pd
import numpy as np

import pickle
from pprint import pprint
import chess
import chess.engine
from chessbot import SF_LOC
from chessbot.utils import rnd, format_time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm


RUN_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_selfplay_deep_priors"

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


def score_cp(pov_score, mate_cp):
    return pov_score.white().score(mate_score=mate_cp)


def analyze_with_deeper_look(move, board, limit, eng, max_depth=16, cpl_trigger=50):
    # best line (white-POV cp)
    sf_best = eng.play(board, limit=limit, info=chess.engine.INFO_ALL)
    best_move = sf_best.move
    res = {'best_move': best_move}
    if move == best_move:
        res['best_cp'] = score_cp(sf_best.info['score'], mate_cp=1500)
        res['played_cp'] = res['best_cp']
        res['best_relative'] = sf_best.info['score'].relative.score(mate_score=1500)
        res['delta_signed'] = 0
        return res

    # analyze the 2 moves together
    root_moves = [move, best_move]
    info = eng.analyse(
        board, limit=limit, root_moves=root_moves,
        info=chess.engine.INFO_ALL, multipv=2
    )

    played_info = [i for i in info if i['pv'][0] == move][0]
    played_cp = score_cp(played_info["score"], mate_cp=1500)

    best_info = [i for i in info if i['pv'][0] == best_move][0]
    best_cp = score_cp(best_info["score"], mate_cp=1500)

    delta_signed = best_cp - played_cp
    if delta_signed >= cpl_trigger:
        new_limit = chess.engine.Limit(depth=max_depth)
        # recurse with infinite cpl_trigger to make sure it returns
        return analyze_with_deeper_look(move, board, new_limit, eng, max_depth, np.inf)
    
    else:
        res['best_cp'] = best_cp
        res['played_cp'] = played_cp
        res['best_relative'] = best_info['score'].relative.score(mate_score=1500)
        res['delta_signed'] = delta_signed
        return res


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

        res = analyze_with_deeper_look(
            move, board, limit, eng, max_depth=18, cpl_trigger=50
        )

        delta_signed = res['delta_signed']
        if board.turn:
            loss_this = max(0, delta_signed); cpl_w += loss_this; nw += 1
        else:
            loss_this = max(0, -delta_signed); cpl_b += loss_this; nb += 1
        cpl_s += loss_this

        rows.append([
            i, mv, str(res['best_move']), res['best_cp'], delta_signed,
            res['played_cp'], res['best_relative'], board.turn, loss_this
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


def worker_shard(games, depth, progress_q, result_q):
    """Each worker runs its own engine, sends results + heartbeats."""
    eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
    eng.configure({"Threads": 1, "Hash": 64})

    out = []
    for g in games:
        try:
            jd = load_json(g["json_file"])
            out.append(analyze_with_sf_core(jd, eng, depth=depth))
            progress_q.put(1)
        except Exception as e:
            print(f"error in worker: {e}")
    eng.close()
    result_q.put(out)


def analyze_many_games(games, workers=6, depth=10):
    start = time.time()
    shards = [games[i::workers] for i in range(workers)]
    total = len(games)

    progress_q = mp.Queue()
    result_q   = mp.Queue()

    procs = [
        mp.Process(target=worker_shard, args=(shards[i], depth, progress_q, result_q))
        for i in range(workers)
    ]
    for p in procs: p.start()

    results = []
    done = 0
    start = time.perf_counter()

    with tqdm(total=total, unit="game", desc="Analyzing games") as pbar:
        active = len(procs)
        while active > 0:
            # drain any progress heartbeats
            while not progress_q.empty():
                progress_q.get_nowait()
                done += 1
                pbar.update(1)

            # collect finished result batches (non-blocking)
            while not result_q.empty():
                part = result_q.get_nowait()
                results.extend(part)
                active -= 1

            time.sleep(0.5)

    for p in procs: p.join()

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
    if "ts" in df_means.columns:
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
    CHUNK_THRESHOLD = 2400
    CHUNK_SIZE = 1200

    all_games = load_game_index()
    pkl_file = os.path.join(RUN_DIR, "analyze_results_combined.pkl")

    # Load prior combined run and filter out already-processed games
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as fp:
            prev_run = pickle.load(fp)
            old_df_means = prev_run['df_means']
            done_ids = set(old_df_means.game_id)
            games = [g for g in all_games if g['game_id'] not in done_ids]
            print(f"Found {len(games)} new games")
    else:
        prev_run = None
        games = all_games
        print(f"No pkl found. Running all {len(games)} games")

    total_games = len(games)
    if total_games == 0:
        print("Nothing to do.")
        # still print the overall summary if a prev_run exists
        if prev_run is not None:
            print("\n=== Overall Summary ===")
            pprint(prev_run['summary'])
        raise SystemExit(0)

    # Decide chunking
    if total_games > CHUNK_THRESHOLD:
        chunks = [games[i:i + CHUNK_SIZE] for i in range(0, total_games, CHUNK_SIZE)]
    else:
        chunks = [games]

    total_chunks = len(chunks)
    combined = prev_run if prev_run is not None else None

    t0 = time.time()
    for ci, chunk in enumerate(chunks, 1):
        print()
        print(f"=== Chunk {ci}/{total_chunks} : {len(chunk)} games ===")

        summary, results, df_all, df_means = analyze_many_games(chunk, workers=6)

        print("=== New Game Summary ===")
        pprint(summary)

        # combine with prior results and/or earlier chunks
        if combined is not None:
            combined = combine_run_stats(combined, summary, results, df_all, df_means)
        else:
            combined = {
                "summary": summary,
                "results": results,
                "df_all": df_all,
                "df_means": df_means,
            }

        # persist after each chunk so progress is saved
        with open(pkl_file, "wb") as f:
            pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elapsed = time.time() - t0
        done_chunks = ci
        remain_chunks = total_chunks - ci
        print(f"Chunks completed: {done_chunks}/{total_chunks}  "
              f"Remaining: {remain_chunks}  "
              f"Elapsed: {format_time(elapsed)}")

        if remain_chunks > 0:
            avg_chunk_s = elapsed / float(done_chunks)
            eta = avg_chunk_s * remain_chunks
            print(f"Estimated remaining: {format_time(eta)}")

    print()
    print("=== Overall Summary ===")
    pprint(combined['summary'])