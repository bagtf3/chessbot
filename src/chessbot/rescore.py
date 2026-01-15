import os, json, pathlib, time
import psutil
import uuid

import pickle

import chess, chess.svg
from IPython.display import SVG, display, clear_output
import chess.engine

import signal
from multiprocessing import Process

import pandas as pd
import numpy as np

from pyfastchess import Board

from chessbot import SF_LOC
from chessbot.config import Config
from chessbot.utils import (
    score_cp_stm_pov, score_cp_white_pov, score_to_value_stm_pov, rnd,
    calc_entropy, cp_to_value_tanh, sf_eval
)

TRAINING_PKL = "additional_training_data.pkl"
ANALYZE_PKL = "analyze_results_combined.pkl"

# stops the post hoc server
POST_HOC_STOP = False
PH = "[post hoc]"

# default analysis params
POLL_INTERVAL = 20
BLUNDER_CP = 150
ANALYZE_BATCH = 30
DEPTH = 12
EQUIV_RANGE = 30

def post_hoc_signal_handler(signum, frame):
    global POST_HOC_STOP
    POST_HOC_STOP = True


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
    if not path.endswith("game_index.json"):
        path = os.path.join(path, "game_index.json")
    return load_json(path)


def analyze_with_rank(move, board, limit, eng):
    # Get Stockfish root best at this depth (white-POV score included in info)
    top3 = eng.analyse(board, limit=limit, info=chess.engine.INFO_ALL, multipv=3)
    best = [t for t in top3 if t['multipv'] == 1][0]
    
    best_move = best['pv'][0]
    best_cp   = score_cp_stm_pov(best["score"])
    best_abs  = score_cp_white_pov(best["score"], clipped=False)
    
    # default
    res = {}
    res['best_move'] = best_move
    res['best_cp'] = best_cp
    res['best_absolute'] = best_abs
    
    if move == best_move:
        res['played_cp'] = best_cp
        res['played_absolute'] = best_abs
        res['delta_signed'] = 0
        res['sf_rank'] = 1
        res['in_top3'] = True
        return res
    
    played = [t for t in top3 if t['pv'][0] == move]
    # if played move not in top3, need to check again
    if not played:
        in_top3 = False
        played = eng.analyse(
            board, limit=limit, root_moves=[move], info=chess.engine.INFO_ALL
        )
        
    else:
        played = played[0]
        in_top3 = True
    
    played_cp = score_cp_stm_pov(played['score'])
    played_abs = score_cp_white_pov(played["score"], clipped=False)
    delta = best_cp - played_cp

    # within equivalence range -> wash: treat as equal, loss=0 and mark both as best
    if abs(delta) <= EQUIV_RANGE:
        res['best_move'] = move # our move is also best
        res['played_cp'] = best_cp
        res['played_absolute'] = best_abs
        res['delta_signed'] = 0
        res['in_top3'] = True
        return res

    # If the played move appears better (delta negative beyond EQUIV_RANGE),
    # treat the played move as the best move (but keep delta_signed negative).
    if delta <= -EQUIV_RANGE:
        res['best_move'] = move
        res['best_cp'] = played_cp
        res['played_cp'] = played_cp
        res['best_absolute'] = played_abs
        res['played_absolute'] = played_abs
        res['delta_signed'] = delta
        res['in_top3'] = True
        return res
    
    # otherwise, our move is worse
    res['played_cp'] = played_cp
    res['played_absolute'] = played_abs
    res['delta_signed'] = delta
    res['in_top3'] = in_top3
    return res


def analyze_with_sf_core(game_data, eng, depth=None):
    if depth is None:
        depth = DEPTH
    limit = chess.engine.Limit(depth=depth)
    board = chess.Board(game_data['start_fen'])

    vs_stockfish = game_data['vs_stockfish']
    sf_color = game_data['stockfish_is_white']

    cpl_s = cpl_w = cpl_b = 0.0
    nw = nb = 0
    rows = []

    for i, mv in enumerate(game_data['moves_played']):
        move = chess.Move.from_uci(mv)

        # skip SF plies
        if vs_stockfish and (board.turn == sf_color):
            board.push(move)
            continue

        res = analyze_with_rank(move, board, limit, eng)

        loss_this = res['delta_signed']
        cpl_s += loss_this
        if board.turn:
            cpl_w += loss_this; nw += 1
        else:
            cpl_b += loss_this; nb += 1

        # append in_top3 flag into the row so it propagates into df
        rows.append([
            i, mv, str(res['best_move']), res['best_cp'], loss_this,
            res['played_cp'],
            res.get('in_top3', False),
            res['best_absolute'], res['played_absolute'],
            board.turn, loss_this
        ])
        board.push(move)

    cols = [
        'move_num','played_move','best_move','best_cp',
        'delta','played_cp','in_top3',
        'best_absolute','played_absolute','stm','loss'
    ]
    out_df = pd.DataFrame(rows, columns=cols)
    out_df["played_best_move"] = out_df["played_move"] == out_df["best_move"]
    
    # replace the three rate lines with this robust version
    mask_w = out_df["stm"] == True
    mask_b = out_df["stm"] == False
    
    overall_bmr = out_df["played_best_move"].mean() if len(out_df) else np.nan
    white_bmr = out_df.loc[mask_w, "played_best_move"].mean() if mask_w.any() else np.nan
    black_bmr = out_df.loc[mask_b, "played_best_move"].mean() if mask_b.any() else np.nan

    # compute in_top3 aggregate counts and rate for this game
    top3_cnt = int(out_df["in_top3"].sum())
    total_plies = len(out_df)
    not_in_top3_cnt = int(total_plies - top3_cnt)
    top3_rate = out_df["in_top3"].mean() if total_plies else np.nan
    
    out = {
        "plies": nw + nb,
        "overall_cpl": rnd(cpl_s/(nw+nb), 3) if (nw+nb) else np.nan,
        "white_cpl":   rnd(cpl_w/nw, 3)      if nw else np.nan,
        "black_cpl":   rnd(cpl_b/nb, 3)      if nb else np.nan,
        "overall_best_move_rate": overall_bmr,
        "best_move_rate_white":    white_bmr,
        "best_move_rate_black":    black_bmr,
        "plays_in_top3_cnt": top3_cnt,
        "not_in_top3_cnt": not_in_top3_cnt,
        "plays_in_top3_rate": top3_rate,
    }
    
    for key in ['game_id','scenario','stockfish_color','ts']:
        out[key] = game_data.get(key)
        out_df[key] = game_data.get(key)
    out['df'] = out_df
    return out

## post hoc server
def save_pickle_atomic(obj, path, tries=0):
    try:
        tmp = str(path) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, str(path))
    except Exception as e:
        if tries <= 5:
            time.sleep(0.5)
            save_pickle_atomic(obj, path, tries=tries+1)
        else:
            raise e


def safe_mean(arr):
    a = np.asarray([x for x in arr if x is not None and not np.isnan(x)])
    return np.nanmean(a) if a.size else float("nan")


def lightweight_summary(results):
    """
    Build the small summary dict you use in combined chunk files.
    'results' is a list of per-game dicts (the merged_results or results).
    """
    wm = [r.get("white_cpl", np.nan) for r in results]
    bm = [r.get("black_cpl", np.nan) for r in results]
    om = [r.get("overall_cpl", np.nan) for r in results]
    wb = [r.get("best_move_rate_white", np.nan) for r in results]
    bb = [r.get("best_move_rate_black", np.nan) for r in results]
    ob = [r.get("overall_best_move_rate", np.nan) for r in results]
    t3 = [r.get("plays_in_top3_rate", np.nan) for r in results]

    return {
        "games": len(results),
        "avg_white_mean_cpl": round(safe_mean(wm), 3),
        "avg_black_mean_cpl": round(safe_mean(bm), 3),
        "avg_overall_mean_cpl": round(safe_mean(om), 3),
        "avg_best_move_rate_white": round(safe_mean(wb), 3),
        "avg_best_move_rate_black": round(safe_mean(bb), 3),
        "avg_overall_best_move_rate": round(safe_mean(ob), 3),
        "avg_played_in_top3_rate": round(safe_mean(t3), 3),
    }


def dedupe_results(results):
    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset='game_id', keep='first')
    # dont want the raw key, its just extra data
    if 'raw' in df.columns:
        df = df.drop(columns=['raw'])
    
    return df.to_dict(orient='records')


def combine_analysis_staging(run_dir):
    """
    Combine all chunked pickles in run_dir/analysis_staging into the single
    ANALYZE_PKL file at run_dir/ANALYZE_PKL. After a successful combine,
    delete the chunk files.

    Behavior:
      - If ANALYZE_PKL exists, new chunks are appended to it.
      - If ANALYZE_PKL does not exist, a new combined file is created.
      - Chunk files are removed only after the combined pkl is written.
    """
    staging = os.path.join(run_dir, "analysis_staging")
    out_pkl = os.path.join(run_dir, ANALYZE_PKL)

    # nothing to do if staging doesn't exist
    if not os.path.isdir(staging):
        print(f"[combine] no /analysis_staging dir")
        return None

    # list chunk files
    fns = sorted([fn for fn in os.listdir(staging) if fn.endswith(".pkl")])
    if not fns:
        print(f"[combine] no chunk files in /analysis_staging dir")
        return None

    # load existing combined (if any)
    if os.path.exists(out_pkl):
        with open(out_pkl, "rb") as f:
            combined = pickle.load(f)
        prev_results = list(combined.get("results", []))
        prev_df_all = combined.get("df_all", None)
        prev_df_means = combined.get("df_means", None)
    else:
        prev_results = []
        prev_df_all = None
        prev_df_means = None

    # accumulate chunk content
    chunk_results = []
    chunk_dfs = []
    chunk_means = []

    for fn in fns:
        path = os.path.join(staging, fn)
        # load each chunk (let exceptions propagate)
        with open(path, "rb") as f:
            chunk = pickle.load(f)

        # expect chunk structure {summary, results, df_all, df_means}
        cres = chunk.get("results", [])
        if cres:
            chunk_results.extend(cres)

        cdf = chunk.get("df_all", None)
        if cdf is not None:
            chunk_dfs.append(cdf)

        cmeans = chunk.get("df_means", None)
        if cmeans is not None:
            chunk_means.append(cmeans)

    # merge results lists
    merged_results = prev_results + chunk_results

    # merge df_all
    if prev_df_all is None:
        if chunk_dfs:
            df_all = pd.concat(chunk_dfs, ignore_index=True)
        else:
            df_all = None
    else:
        if chunk_dfs:
            df_all = pd.concat([prev_df_all] + chunk_dfs, ignore_index=True)
        else:
            df_all = prev_df_all

    # merge df_means
    if prev_df_means is None:
        if chunk_means:
            df_means = pd.concat(chunk_means, ignore_index=True)
        else:
            df_means = None
    else:
        if chunk_means:
            df_means = pd.concat([prev_df_means] + chunk_means, ignore_index=True)
        else:
            df_means = prev_df_means

    # de dupe
    df_all = df_all.drop_duplicates(['game_id', 'move_num']).sort_values("ts")
    df_means = df_all.drop_duplicates(['game_id']).sort_values("ts")
    merged_results = dedupe_results(merged_results)

    # recompute lightweight summary from merged_results
    summary = lightweight_summary(merged_results)

    combined_new = {
        "summary": summary,
        "results": merged_results,
        "df_all": df_all,
        "df_means": df_means
    }

    # persist atomically using existing helper
    save_pickle_atomic(combined_new, out_pkl)
    print(f"[combine] wrote combined ANALYZE_PKL -> {out_pkl} "
          f"({len(merged_results)} games)")

    # delete the chunk files that we just combined
    for fn in fns:
        path = os.path.join(staging, fn)
        os.remove(path)
    print(f"[combine] removed {len(fns)} chunk files from {staging}")

    return combined_new


def save_analysis_chunk_simple(run_dir, batch):
    """
    Build a combined-style object for `batch` (list of analysis_out dicts)
    and write it as one pickle into run_dir/analysis_staging/ with a unique
    filename. No tmp file, no exceptions swallowed.
    """
    staging = os.path.join(run_dir, "analysis_staging")
    os.makedirs(staging, exist_ok=True)

    # results list (one row per game)
    results = []
    all_dfs = []
    for analysis_out in batch:
        row = {
            "game_id": analysis_out.get("game_id"),
            "ts": analysis_out.get("ts"),
            "white_cpl": analysis_out.get("white_cpl"),
            "black_cpl": analysis_out.get("black_cpl"),
            "overall_cpl": analysis_out.get("overall_cpl"),
            "best_move_rate_white": analysis_out.get("best_move_rate_white"),
            "best_move_rate_black": analysis_out.get("best_move_rate_black"),
            "overall_best_move_rate": analysis_out.get("overall_best_move_rate"),
            "plays_in_top3_rate": analysis_out.get("plays_in_top3_rate")
        }
        results.append(row)
        df = analysis_out.get("df")
        if df is not None:
            all_dfs.append(df)

    # df_all is concat of per-game dfs (or None)
    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else None
    df_means = pd.DataFrame.from_records(results) if results else None
    summary = lightweight_summary(results)

    chunk_obj = {
        "summary": summary,
        "results": results,
        "df_all": df_all,
        "df_means": df_means
    }

    cpl = df_all.delta.mean()
    bmr = df_all.played_best_move.mean()
    top3 = df_all.in_top3.mean()

    print(f"{PH} Saving {len(batch)} analyzed games")
    print(f"{PH} {'Batch stats:':<16} CPL {cpl:.3f} BMR {bmr:.3f} TOP3 {top3:.3f}")

    fname = f"{int(time.time())}_{uuid.uuid4().hex}.pkl"
    outp = os.path.join(staging, fname)

    with open(outp, "wb") as f:
        pickle.dump(chunk_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return outp, cpl, bmr, top3


def make_fake_visits(mv, lms, ratio_best=60):
    visits = [[mv, int(ratio_best)]]
    
    # may only be 1 legal move
    if len(lms) < 2:
        return visits
    
    sub_optimal = 100 - ratio_best
    bad_visits = 1 + min(5, int(sub_optimal / len(lms)))
    visits += [[m, bad_visits] for m in lms if m != mv]
    return visits


def adjust_visits_from_cm(cm, played_mv, best_mv, lms):
    """
    cm: list of {'uci': ..., 'visits': ...}
    Ensure every legal move in lms appears (min 1) and swap visits
    for best and played with 30% bump. This stabilizes training.
    Return list of [uci, int_visits] sorted desc.
    """
    # build dict of existing counts (min 1)
    d = {}
    for c in cm:
        u = c.get('uci')
        v = int(c.get('visits', 1))
        if u:
            d[u] = max(1, v)

    # ensure all legal moves exist with min 1
    for m in lms:
        if m not in d:
            d[m] = 1

    # compute old max
    old_max = max(d.values()) if d else 1

    # adjust by swapping visits between played and best with 30% bump
    d[played_mv] = int(np.ceil(0.7*d[best_mv]))
    d[best_mv] = int(np.ceil(1.3*old_max))

    # build sorted list
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return [[u, int(v)] for u, v in items]


def make_training_sample(b, v, visits):
    # turn counts into move probabilites
    ucis   = [x[0] for x in visits]
    counts = np.array([x[1] for x in visits], dtype=np.float32)
    s = counts.sum()
    pi = (counts / s) if s > 0.0 else np.zeros_like(counts)
    
    # get indices from C++
    indices = b.moves_to_indices(ucis)  # list of ints (0..4288)
    policy = np.zeros(64 * 67, dtype=np.float32)

    # accumulate probs into flattened policy
    for idx, p in zip(indices, pi):
        policy[idx] += p

    x = b.encode_64_tokens()
    mask = b.legal_move_mask()

    # needs to match looper's training_queue
    # (x, mask, policy, z_stm, vwq, z_tapered)
    tup = (x, mask, policy, 0, v, 0)
    return tup


def sfe(b, engine):
    """ Quick SF eval wrapper """
    return sf_eval(b, score_fn=score_to_value_stm_pov, depth=DEPTH, engine=engine)


def mine_additional_training_data(analysis_out, game_data, engine=None):
    def stm(b):
        return b.side_to_move() == 'w'

    df = analysis_out['df']

    tree_data = game_data['tree_search_data']
    vs_stockfish = game_data['vs_stockfish']
    sf_color = game_data['stockfish_is_white']

    b = Board(game_data['start_fen'])
    training_data = []
    for i, mv in enumerate(game_data['moves_played']):
        if vs_stockfish and (sf_color == stm(b)):
            # we already have these, dont duplicate
            b.push_uci(mv)
            continue

        lms = b.legal_moves()
        # terminals are handled elsewhere
        if not lms:
            break

        # find the df row associated to this move
        row = df.query("played_move == @mv")
        if len(row) > 1:
            row = df.query("played_move == @mv").query("move_num == @i")
        if len(row) == 0:
            row = df.loc[df.move_num == str(i)].query("played_move == @mv")
        if len(row) == 0:
            b.push_uci(mv)
            continue

        # happy path, not a blunder, no action
        if row['delta'].item() < BLUNDER_CP:
            b.push_uci(mv)
            continue

        # if we are here, the move was a blunder
        best_v = cp_to_value_tanh(row['best_cp'].item())
        best_mv = row['best_move'].item()

        tr = tree_data.get(i, tree_data.get(str(i), {}))
        cm = tr.get('candidate_moves', [])

        # if a blunder, dont use actual visits (theyre wrong)
        if cm:
            best_visits = adjust_visits_from_cm(cm, mv, best_mv, lms)
        else:
            best_visits = make_fake_visits(best_mv, lms)

        # sanity check: best_visits must exist and sum to > 0
        if not best_visits or sum([v[1] for v in best_visits]) <= 0:
            print("[blunder mining] visits invalid or sum <= 0; skipping example",
                  "move_idx=", i, "played=", mv, "best=", best_mv)
            b.push_uci(mv)
            continue

        # make training sample and append
        ts_best = make_training_sample(b, best_v, best_visits)
        training_data.append(ts_best)

        # push to move and let the loop roll over
        b.push_uci(mv)

    return training_data


def report_unprocessed(entries, seen_games):
    unproc = sum([1 for e in entries if e.get('game_id') not in seen_games])
    if unproc:
        print(f"{PH} {unproc} unprocessed game(s) currently in queue")


def post_hoc_worker(run_cfg, batch_games=10, batch_secs=90):
    run_dir = run_cfg.run_dir
    bonus_data = run_cfg.mine_bonus_data

    ## trying to deprioritize the server so it doesnt slow down main training looper
    p = psutil.Process()

    if os.name == "posix":
        # lower CPU priority (nice). 10 is a polite background value.
        p.nice(10)
        # set IO priority to idle if available (Linux)
        if hasattr(p, "ionice"):
            p.ionice(psutil.IOPRIO_CLASS_IDLE)

    elif os.name == "nt":
        # Windows: lower process priority class
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    # pin worker to a small set of cores (use last 1-2 physical cores as example)
    # adjust the slice to pick different cores if you want
    phy = psutil.cpu_count(logical=False) or psutil.cpu_count()
    if phy is not None:
        use_count = 1 if phy <= 2 else 2
        start = max(0, phy - use_count)
        cores = list(range(start, start + use_count))
        # cpu_affinity expects a list of logical/core ids; ok on Linux & Windows
        p.cpu_affinity(cores)

    # apply overrides
    POLL_INTERVAL = run_cfg.post_hoc_poll_interval
    BLUNDER_CP = run_cfg.post_hoc_blunder_cp
    ANALYZE_BATCH = run_cfg.post_hoc_analyze_batch
    DEPTH = run_cfg.post_hoc_depth
    EQUIV_RANGE = run_cfg.post_hoc_equiv_range

    # also set them in globals so other functions (defined above) see them
    globals().update({
        "POLL_INTERVAL": POLL_INTERVAL,
        "BLUNDER_CP": BLUNDER_CP,
        "ANALYZE_BATCH": ANALYZE_BATCH,
        "DEPTH": DEPTH,
        "EQUIV_RANGE": EQUIV_RANGE
    })

    # post hoc logic starts here
    idx_path = os.path.join(run_dir, "game_index.json")
    pkl_path = os.path.join(run_dir, TRAINING_PKL)

    # first merge any existing analysis
    _ = combine_analysis_staging(run_dir)

    # seen games from existing analyze pkl
    seen_games = set()
    analyze_pkl_path = os.path.join(run_dir, ANALYZE_PKL)
    if os.path.exists(analyze_pkl_path):
        with open(analyze_pkl_path, "rb") as f:
            combined = pickle.load(f)
        if "df_means" in combined and combined["df_means"] is not None:
            seen_games = set(
                combined["df_means"]["game_id"].astype(str).tolist()
            )

    # start fresh with training data
    running_list = []
    batch_samples = []
    analyzed_batch = []

    # analysis stats
    n_saved = 0
    tcpl, tbmr, ttop3 = 0, 0, 0

    games_since_flush = 0
    last_flush = time.time()

    # install signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, post_hoc_signal_handler)
    signal.signal(signal.SIGTERM, post_hoc_signal_handler)

    # single engine reused across loop
    eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
    eng.configure({"Threads": 1, "Hash": 128})

    unproc_report = True
    try:
        # see if there is a starting pkl file
        last_seen_pkl = os.path.exists(pkl_path)
        next_threshold = 500
        while not POST_HOC_STOP:
            if not os.path.exists(idx_path):
                # check shutdown every poll_interval
                time.sleep(POLL_INTERVAL)
                continue

            idx = load_game_index(idx_path)
            entries = [idx] if isinstance(idx, dict) else idx
            if unproc_report:
                report_unprocessed(entries, seen_games)
                unproc_report = False

            for rec in entries:
                if POST_HOC_STOP:
                    break

                gid = str(rec.get("game_id"))
                json_path = rec.get("json_file")
                if not json_path or not os.path.exists(json_path):
                    continue
                if gid in seen_games:
                    continue
                
                # run analysis in-memory (do NOT persist per-game here)
                with open(json_path, "r", encoding="utf-8") as gf:
                    game_data = json.load(gf)
                
                analysis_out = analyze_with_sf_core(game_data, eng=eng)
                analysis_out["game_id"] = game_data.get("game_id")
                analysis_out["ts"] = game_data.get("ts")

                # accumulate for bundled saving later
                analyzed_batch.append(analysis_out)
                # create training tuples for this game
                if bonus_data:
                    samples = mine_additional_training_data(
                        analysis_out, game_data, engine=eng
                    )

                    if samples:
                        batch_samples.extend(samples)

                seen_games.add(gid)
                games_since_flush += 1

                # write chunk files of ANALYZE_BATCH games into analysis_staging/
                if len(analyzed_batch) >= ANALYZE_BATCH:
                    n_saved += 1
                    outp, c, b, t = save_analysis_chunk_simple(run_dir, analyzed_batch)
                    tcpl += c; tbmr += b; ttop3 += t
                    if n_saved >= 2:
                        print(
                            f"{PH} {'Overall stats:':<16} CPL {tcpl/n_saved:.3f}",
                            f"BMR {tbmr/n_saved:.3f} TOP3 {ttop3/n_saved:.3f}"
                        )
                    
                    analyzed_batch.clear()
                    unproc_report = True

                now = time.time()
                if games_since_flush >= batch_games or (now - last_flush) >= batch_secs:
                    if batch_samples:
                        # check if the pkl has been cleared
                        curr_exists = os.path.exists(pkl_path)
                        if last_seen_pkl and not curr_exists:
                            running_list = []
                            next_threshold = 500
                        last_seen_pkl = curr_exists

                        running_list.extend(batch_samples)
                        save_pickle_atomic(running_list, pkl_path)
                        # reflect exact on-disk state
                        last_seen_pkl = True
                        lrl = len(running_list)
                        if lrl >= next_threshold:
                            next_threshold += 500
                            print(f"{PH} pushed {lrl} training samples")
                    
                    batch_samples = []
                    games_since_flush = 0
                    last_flush = now

            # sleep but wake quickly if shutdown requested
            for _ in range(max(1, POLL_INTERVAL)):
                if POST_HOC_STOP:
                    break
                time.sleep(1)

    finally:
        # final flush: write any partially-accumulated analysis batch
        try:
            if analyzed_batch:
                print(f"{PH} final flush: saving {len(analyzed_batch)} analyzed games")
                # reuse existing chunk writer (writes into analysis_staging/)
                save_analysis_chunk_simple(run_dir, analyzed_batch)
                analyzed_batch.clear()
        except Exception as e:
            print(f"{PH} final flush (analyzed_batch) failed: {e}")
        
        # ensure engine is cleanly quit
        try:
            eng.quit()
        except Exception:
            pass

        print("[post_hoc] post_hoc_worker exiting cleanly")
    

def start_post_hoc_server(cfg):
    """Start post-hoc worker process and forward the working Config object."""
    p = Process(target=post_hoc_worker, args=(cfg,), daemon=False)
    p.start()

    parts = os.path.normpath(cfg.run_dir).split(os.path.sep)
    tail = os.path.sep.join(parts[-2:])
    print(f"[post hoc] started server pid={p.pid} run_dir={tail} "
        f"mine_bonus={cfg.mine_bonus_data}")
    return p


def stop_post_hoc_server(p, timeout=10):
    if p is None:
        return

    # prefer polite SIGINT on POSIX so worker can flush and quitEngine
    if os.name == "posix":
        os.kill(p.pid, signal.SIGINT)
    else:
        p.terminate()

    start = time.time()
    p.join(timeout)
    if p.is_alive():
        # escalate to terminate/kill
        p.terminate()
        p.join(3)

    if p.is_alive():
        print("post_hoc_server did not exit cleanly; process still alive")
    else:
        print("post_hoc_server stopped")
        del p
