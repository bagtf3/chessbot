import json, os, time
import pandas as pd
import numpy as np

import pickle
from pprint import pprint
import chess
import chess.engine
from chessbot import SF_LOC
from chessbot.utils import rnd, format_time
from chessbot.review import load_json, load_game_index, analyze_with_sf_core

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm


RUN_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_test"
N_WORKERS = 6


def worker_shard(games, progress_q, result_q):
    """Each worker runs its own engine, sends results + heartbeats."""
    eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
    eng.configure({"Threads": 1, "Hash": 128})

    out = []
    for g in games:
        try:
            jd = load_json(g["json_file"])
            out.append(analyze_with_sf_core(jd, eng))
            progress_q.put(1)
        except Exception as e:
            print(f"error in worker: {e}")
    eng.close()
    result_q.put(out)


def analyze_many_games(games):
    start = time.time()
    shards = [games[i::N_WORKERS] for i in range(N_WORKERS)]
    total = len(games)

    progress_q = mp.Queue()
    result_q   = mp.Queue()

    procs = [
        mp.Process(target=worker_shard, args=(shards[i], progress_q, result_q))
        for i in range(N_WORKERS)
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

            time.sleep(0.1)

    for p in procs: p.join()

    # aggregate
    w_means = [r["white_cpl"] for r in results]
    b_means = [r["black_cpl"] for r in results]
    o_means = [r["overall_cpl"] for r in results]
    w_best  = [r["best_move_rate_white"] for r in results]
    b_best  = [r["best_move_rate_black"] for r in results]
    o_best  = [r["overall_best_move_rate"] for r in results]

    top3_rates = [r.get("plays_in_top3_rate", np.nan) for r in results]

    summary = {
        "games": len(results),
        "avg_white_mean_cpl": rnd(np.nanmean(w_means), 3),
        "avg_black_mean_cpl": rnd(np.nanmean(b_means), 3),
        "avg_overall_mean_cpl": rnd(np.nanmean(o_means), 3),
        "avg_best_move_rate_white": rnd(np.nanmean(w_best), 3),
        "avg_best_move_rate_black": rnd(np.nanmean(b_best), 3),
        "avg_overall_best_move_rate": rnd(np.nanmean(o_best), 3),
        "avg_played_in_top3_rate": rnd(np.nanmean(top3_rates), 3),
    }

    
    df_all  = pd.concat([r.pop("df") for r in results], ignore_index=True)
    df_means = pd.DataFrame.from_records(results)
    if "ts" in df_means.columns:
        df_means["ts"] = df_means["ts"].round().astype("Int64")
        
    stop = time.time()
    total = format_time(stop-start)
    print(f"Analyzed {len(games)} games in {total}")
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

    top3_rates = [r.get("plays_in_top3_rate", np.nan) for r in new_results]
    
    new_summary = {
        "games": len(new_results),
        "avg_white_mean_cpl": rnd(np.nanmean(w_means), 3),
        "avg_black_mean_cpl": rnd(np.nanmean(b_means), 3),
        "avg_overall_mean_cpl": rnd(np.nanmean(o_means), 3),
        "avg_best_move_rate_white": rnd(np.nanmean(w_best), 3),
        "avg_best_move_rate_black": rnd(np.nanmean(b_best), 3),
        "avg_overall_best_move_rate": rnd(np.nanmean(o_best), 3),
        "avg_played_in_top3_rate": rnd(np.nanmean(top3_rates), 3)
    }

    return {
        "summary": new_summary, "results": new_results,
        "df_all": new_df_all, "df_means": new_df_means
    }


if __name__ == '__main__':
    CHUNK_THRESHOLD = 599
    CHUNK_SIZE = 300

    all_games = load_game_index(RUN_DIR)
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

        summary, results, df_all, df_means = analyze_many_games(chunk)

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

