import json, sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import spearmanr

import chess
import chess.engine
from chessbot import SF_LOC
from chessbot.utils import show_board, rnd
from concurrent.futures import ThreadPoolExecutor, as_completed


RUN_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_200_selfplay"

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


def summarize_games(games):
    rows = []
    for g in games:
        vs_sf = bool(g.get("started_vs_stockfish"))
        sf_col = g.get("stockfish_color")
        sf_color = "white" if sf_col is True else ("black" if sf_col is False else None)

        rows.append({
            "scenario": g.get("scenario"),
            "result": g.get("result"),
            "plies": g.get("plies"),
            "vs_sf": vs_sf,
            "beat_sf": g.get("beat_sf") if vs_sf else None,
            "sf_color": sf_color,
        })

    df = pd.DataFrame(rows)

    grp = df.groupby(["scenario", "sf_color"], dropna=False)

    out = grp.agg(
        n_games=("result", "size"),
        avg_result=("result", "mean"),
        avg_plies=("plies", "mean"),
        n_vs_sf=("vs_sf", "sum"),
        n_selfplay=("vs_sf", lambda s: int((~s).sum())),
        n_beat_sf=("beat_sf", lambda s: int((s == True).sum())),
    ).reset_index()

    out["beat_sf_rate"] = out.apply(
        lambda r: (r["n_beat_sf"] / r["n_vs_sf"]) if r["n_vs_sf"] else None, axis=1
    )

    return out


def step_game(board, moves_uci, eng, flipped=False, depth=15, mate_cp=1500):
    """
    Controls:
      <enter>  next move
      b        back one move
      s        eval (always White POV)
      g <n>    jump to ply n (0-based)
      q        quit
    """
    def who_moved(ply_idx, sf_color):
        if sf_color is None:
            return "Move"
        is_white = (ply_idx % 2 == 0)
        sf_turn = (sf_color and is_white) or ((not sf_color) and (not is_white))
        return "SF" if sf_turn else "Bot"

    def eval_white_pov():
        eng_lim = chess.engine.Limit(depth=depth)
        eng_inf = info=chess.engine.INFO_SCORE
        info = eng.analyse(board, eng_lim, eng_inf)
        sc = info["score"].pov(chess.WHITE)
        if sc.is_mate():
            return f"mate {sc.mate()}"
        return f"{sc.score() if sc.score() is not None else 0} cp"

    def redraw():
        show_board(board, flipped=flipped)
        print(f"ply={len(board.move_stack)} | eval(White POV)={eval_white_pov()}")

    ply = 0
    redraw()

    while True:
        cmd = input("[enter]=next, b=back, s=eval, g <n>=goto, q=quit > ").strip()
        if cmd == "":
            if ply >= len(moves_uci):
                print("end of game")
                continue
            u = moves_uci[ply]
            mv = chess.Move.from_uci(u)
            label = who_moved(ply, flipped)  # flipped == stockfish_color
            print(f"{label} plays {board.san(mv)}")
            board.push(mv)
            ply += 1
            redraw()
            continue

        if cmd == "b":
            if board.move_stack:
                board.pop()
                ply -= 1
            redraw()
            continue

        if cmd == "s":
            print(f"eval(White POV)={eval_white_pov()}")
            continue

        if cmd.startswith("g "):
            n = int(cmd.split()[1])
            n = max(0, min(n, len(moves_uci)))
            while board.move_stack:
                board.pop()
            for i in range(n):
                board.push_uci(moves_uci[i])
            ply = n
            redraw()
            continue

        if cmd == "q":
            break

        print("unknown command")


def score_cp(pov_score, white_to_move, mate_cp):    
    return pov_score.white().score(mate_score=mate_cp)


def analyze_with_sf(file_loc):
    game_data = load_json(file_loc)
    limit = chess.engine.Limit(depth=10)
    board = chess.Board(game_data['start_fen'])
    sf_color = game_data['stockfish_is_white']
        
    cpl_s, cpl_w, cpl_b =  0, 0, 0
    nw, nb = 0, 0
    out = []
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
        for i, mv in enumerate(game_data['moves_played']):
            move = chess.Move.from_uci(mv)
            # dont care about stockfish moves
            if board.turn == sf_color:
                board.push(move)
                continue
            else:
                sf_best = eng.play(board, limit=limit, info=chess.engine.INFO_ALL)
                best_move = sf_best.move
                best_cp = score_cp(sf_best.info['score'], board.turn, mate_cp=1500)
                best_relative = sf_best.info['score'].relative.score(mate_score=1500)
                
                played_info = eng.analyse(
                    board, limit=limit, root_moves=[move], info=chess.engine.INFO_SCORE
                )
                played_sc = played_info.get("score")
                played_cp = score_cp(played_sc, board.turn, mate_cp=1500)
                delta = best_cp - played_cp
                
                if board.turn:
                    cpl_s += delta
                    cpl_w += delta
                    nw +=1
                else:
                    cpl_s += -delta
                    cpl_b += -delta
                    nb+=1
                    
                l = [
                    i, mv, str(best_move), best_cp, played_cp,
                    np.abs(delta), best_relative, board.turn
                ]
                out.append(l)
                board.push(move)
    
    cols = [
        'move_num', "played_move", "best_move", "best_cp",
        "delta", "played_cp", "relative_score", "stm"
    ]
    out_df = pd.DataFrame(out, columns=cols)
    out_df['played_best_move'] = out_df['played_move'] == out_df["best_move"]
    
    out = {
        "plies": nw+nb,
        "overall_cpl": rnd(cpl_s/(nw+nb), 3) if nw+nb else np.nan,
        "white_cpl": rnd(cpl_w/(nw), 3) if nw else np.nan,
        "black_cpl": rnd(cpl_b/(nb), 3) if nb else np.nan,
        "overall_best_move_rate": out_df['played_best_move'].mean(),
        "best_move_rate_white": out_df.query("stm")['played_best_move'].mean(),
        "best_move_rate_black": out_df.query("not stm")['played_best_move'].mean()
    }
    
    for key in ['game_id', 'scenario', 'stockfish_color', 'ts']:
        out[key] = game_data[key]
        out_df[key] = game_data[key]
    
    out['df'] = out_df
    return out


def analyze_many_games(games, workers=8):
    def worker(file_loc):
        return analyze_with_sf(file_loc)

    N = len(games)
    results = [None] * N

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = {}
        for i, g in enumerate(games):
            futs[ex.submit(worker, g['json_file'])] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(e)
                results[i] = None
    
    results = [r for r in results if r is not None]
    
    w_means = [r["white_cpl"] for r in results]
    b_means = [r["black_cpl"] for r in results]
    o_means = [r["overall_cpl"] for r in results]
    
    w_best = [r["best_move_rate_white"] for r in results]
    b_best = [r["best_move_rate_black"] for r in results]
    o_best = [r["overall_best_move_rate"] for r in results]
    
    summary = {
        "games": len(results),
        "avg_white_mean_cpl": rnd(np.nanmean(w_means), 3),
        "avg_black_mean_cpl": rnd(np.nanmean(b_means), 3),
        "avg_overall_mean_cpl": rnd(np.nanmean(o_means), 3),
        "avg_best_move_rate_white": rnd(np.nanmean(w_best), 3),
        "avg_best_move_rate_black": rnd(np.nanmean(b_best), 3),
        "avg_overall_best_move_rate": rnd(np.nanmean(o_best), 3),
    }
    
    df_all = pd.concat([r.pop("df") for r in results])
    df_means = pd.DataFrame.from_records(results)
    df_means["ts"] = df_means["ts"].round().astype("int64")
    
    return summary, results, df_all, df_means


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
    - linear slope per 100 games
    - start→end change (first/last decile means)
    - Spearman rank correlation (monotonic trend)
    """
    d = df.sort_values("ts").reset_index(drop=True).copy()
    x = np.arange(len(d))  # game index

    cpl = d["overall_cpl"].rolling(window, center=True, min_periods=1).mean().values
    bmr = d["overall_best_move_rate"].rolling(
        window, center=True, min_periods=1).mean().values

    def lin_slope_per100(y):
        # simple least-squares slope scaled per 100 games
        x0 = x - x.mean()
        b = (x0 @ (y - y.mean())) / (x0 @ x0)
        return rnd(100 * b, 5)

    def start_end_change(y, frac=0.1):
        k = max(1, int(len(y) * frac))
        return rnd(np.nanmean(y[-k:]) - np.nanmean(y[:k]), 5)

    # metrics
    out = {
        f"cpl_slope_per_{window}": lin_slope_per100(cpl),
        "cpl_start_to_end": start_end_change(cpl),
        "cpl_spearman": rnd(spearmanr(x, cpl, nan_policy="omit").statistic, 5),
        f"bmr_slope_per_{window}": lin_slope_per100(bmr),
        "bmr_start_to_end": start_end_change(bmr),
        "bmr_spearman": rnd(spearmanr(x, bmr, nan_policy="omit").statistic, 5)
    }
    return out


def add_points_vs_sf(df):
    """
    Adds two columns:
      - bot_result_vs_sf: {-1,0,1} from the BOT'S pov (NaN if not vs SF)
      - points_vs_sf: {0,0.5,1} from the BOT'S pov (NaN if not vs SF)
    Assumes df has: ['result', 'started_vs_stockfish', 'stockfish_color'].
    'result' is white-POV: -1 loss, 0 draw, +1 win.
    """
    d = df.copy()
    vs_sf = d["started_vs_stockfish"] == True

    # bot POV: if SF is white, flip sign; if SF is black, keep sign
    sign = np.where(d["stockfish_color"] == True, -1.0, 1.0)
    bot_res = np.where(vs_sf, sign * d["result"].astype(float), np.nan)

    d["bot_result_vs_sf"] = bot_res
    d["points_vs_sf"] = (bot_res + 1.0) / 2.0  # -1/0/+1 -> 0/0.5/1
    return d


def rolling_points_vs_sf(df, window=200):
    """
    Returns a new DataFrame sorted by time with a rolling mean of points_vs_sf.
    NaNs (non-SF games) are ignored inside the window.
    """
    d = add_points_vs_sf(df).sort_values("ts").reset_index(drop=True)
    d["rolling_points_vs_sf"] = (d["points_vs_sf"].rolling(window, min_periods=1).mean())
    return d


def wilson(p, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    halfw  = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5) / denom
    return center - halfw, center + halfw


def plot_rolling_rates_with_ci(df, window=200):
    d = df[df.started_vs_stockfish].sort_values("ts").reset_index(drop=True).copy()
    sign = np.where(d.stockfish_color == True, -1.0, 1.0)
    res = sign * d.result.astype(float)
    win = (res == 1).astype(float)
    draw = (res == 0).astype(float)
    pts = (res + 1.0) / 2.0

    # trailing rolling stats
    p_win = win.rolling(window, center=True, min_periods=1).mean()
    p_draw = draw.rolling(window, center=True, min_periods=1).mean()
    m_pts = pts.rolling(window, center=True, min_periods=1).mean()
    s_pts = pts.rolling(window, center=True, min_periods=2).std()
    n_pts = pts.rolling(window, center=True, min_periods=1).count()

    # 95% normal CI for mean of points (bounded but fine as a quick approx)
    z = 1.96
    se = s_pts / np.sqrt(n_pts.replace(0, np.nan))
    lo = m_pts - z * se
    hi = m_pts + z * se

    x = range(len(d))
    plt.figure(figsize=(11, 5))
    plt.plot(x, p_win, label="win rate", color="tab:orange", lw=2)
    plt.plot(x, p_draw, label="draw rate", color="tab:blue", lw=2)
    plt.plot(x, m_pts, label="points (mean)", color="tab:green", lw=2)
    plt.fill_between(x, lo, hi, color="tab:green", alpha=0.15, linewidth=0)
    
    # auto-fit y with padding, clipped to [0,1]
    y_min = np.nanmin([p_win.min(), p_draw.min(), m_pts.min()])
    y_max = np.nanmax([p_win.max(), p_draw.max(), m_pts.max()])
    pad = 0.05 * (y_max - y_min if np.isfinite(y_max - y_min) and y_max > y_min else 1.0)
    plt.ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    
    plt.xlabel(f"Game # vs SF (trailing window={window})")
    plt.ylabel("Rate / Points")
    plt.title("Trailing win/draw; points with 95% CI")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()


def tail_vs_prev(df, window=200):
    def z_two_prop(p1, p2, n1, n2):
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        return (p2 - p1) / se if se else float("nan")

    def z_to_p(z):
        return math.erfc(abs(z) / math.sqrt(2.0))

    def z_two_mean(m1, m2, s1, s2, n1, n2):
        # Welch z (normal approx)
        se = math.sqrt((s1 * s1) / n1 + (s2 * s2) / n2)
        return (m2 - m1) / se if se else float("nan")
    
    d = df[df.started_vs_stockfish].sort_values("ts").reset_index(drop=True)
    d = d.copy()
    sign = np.where(d.stockfish_color == True, -1.0, 1.0)
    res = sign * d.result.astype(float)
    pts = (res + 1.0) / 2.0
    wins = (res == 1).astype(float)
    draws = (res == 0).astype(float)

    tail = slice(-window, None)
    prev = slice(-2 * window, -window)

    def mean_n(x):
        arr = np.asarray(x, dtype=float)
        return float(np.nanmean(arr)), int(np.sum(~np.isnan(arr)))

    def var_n(x):
        arr = np.asarray(x, dtype=float)
        return float(np.nanvar(arr, ddof=1)), int(np.sum(~np.isnan(arr)))

    rows = {}

    # points: use Welch z on 0/0.5/1
    m1, n1 = mean_n(pts[prev]); m2, n2 = mean_n(pts[tail])
    s1, _ = var_n(pts[prev]);   s2, _ = var_n(pts[tail])
    z = z_two_mean(m1, m2, math.sqrt(s1), math.sqrt(s2), n1, n2)
    rows["points"] = {
        "prev_mean": m1, "prev_n": n1,
        "tail_mean": m2, "tail_n": n2,
        "delta_mean": m2 - m1,
        "z": z, "p_two_sided": z_to_p(z),
    }

    # wins/draws: two-proportion z-tests
    for name, arr in (("win_rate", wins), ("draw_rate", draws)):
        p1, n1 = mean_n(arr[prev]); p2, n2 = mean_n(arr[tail])
        z = z_two_prop(p1, p2, n1, n2)
        rows[name] = {
            "prev_mean": p1, "prev_n": n1,
            "tail_mean": p2, "tail_n": n2,
            "delta_mean": p2 - p1,
            "z": z, "p_two_sided": z_to_p(z),
        }

    return pd.DataFrame(rows).T.round(6)

games = load_game_index()
summary, results, df_all, df_means = analyze_many_games(games, workers=4)

# f = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_selfplay/analyze_results_2470_games.pkl"
# import pickle
# with open(f, "rb") as fp:
#     run1000 = pickle.load(fp)
# df_means = run1000['df_means']
    
df_trim = df_means.query("overall_cpl > 0").query("overall_cpl < 500")
plot_cpl_and_bmr(df_trim, window=50)
trend_check(df_trim, window=50)

df_games = pd.DataFrame.from_records(games)
df_games["ts"] = df_games["ts"].round().astype("int64")
df_games = rolling_points_vs_sf(df_games)

plot_rolling_rates_with_ci(df_games, window=30)
tail_vs_prev(df_games, window=30)
