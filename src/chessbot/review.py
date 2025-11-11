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
from chessbot.utils import (
    score_cp_stm_pov, score_cp_white_pov, score_to_value_stm_pov, rnd,
    calc_entropy, cp_to_value_tanh, sf_eval
)


POLL_INTERVAL = 7

BLUNDER_CP = 60
TRAINING_PKL = "additional_training_data.pkl"
ANALYZE_PKL = "analyze_results_combined.pkl"
ANALYZE_BATCH = 30

# default analysis params
DEPTH = 12
EQUIV_RANGE = 10

# stops the post hoc server
POST_HOC_STOP = False

PH = "[post hoc]"

def post_hoc_signal_handler(signum, frame):
    global POST_HOC_STOP
    POST_HOC_STOP = True


class GameViewer:
    def __init__(self, log_path, sf_df=None):
        self.path = pathlib.Path(log_path)
        with open(self.path, "r", encoding="utf-8") as f:
            self.log = json.load(f)
    
        self.start_fen = self.log.get("start_fen")
        self.moves_uci = self.log.get("moves_played", [])
        self.tree_data = self.log.get("tree_search_data", {})
        self.result = self.log.get("result")
        self.game_id = self.log.get("game_id")
    
        self.sf_rows = None
        self._sf_by_ply = {}  # maps ply index -> row index into self.sf_rows
    
        if sf_df is not None and self.game_id is not None:
            try:
                g = sf_df[sf_df["game_id"] == self.game_id].copy()
                if not g.empty:
                    g['move_num_int'] = g['move_num'].astype(int)
                    g = g.sort_values('move_num_int', kind="stable")
                    g = g.reset_index(drop=True)
                    self.sf_rows = g
                    self.align_sf_rows_to_json()
            except Exception:
                self.sf_rows = None
                self._sf_by_ply = {}
    
        self.reset()

    def reset(self):
        self.board = chess.Board(self.start_fen)
        self.ply = 0  # 0 = before first move

    def goto(self, ply):
        ply = max(0, min(ply, len(self.moves_uci)))
        self.board = chess.Board(self.start_fen)
        for u in self.moves_uci[:ply]:
            self.board.push_uci(u)
        self.ply = ply
        return self
    
    def sf_row_for_ply(self, ply):
        if (not self._sf_by_ply) or (self.sf_rows is None):
            return None
        idx = self._sf_by_ply.get(ply)
        if idx is None:
            return None
        return self.sf_rows.iloc[idx]

    def run_stockfish_topk(self, depth=16, k=3):
        """
        Return list[(uci, san, cp_white_pov, pv)] for top-k moves.
        Let exceptions propagate so we see stack traces if SF fails.
        """
        out = []
        limit = chess.engine.Limit(depth=depth, time=10.0)
        with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
            infos = eng.analyse(self.board, limit=limit, multipv=k)
            if not isinstance(infos, list):
                infos = [infos]
            for info in infos:
                pv = info.get("pv", [])
                move_uci = pv[0].uci() if pv else None
                san = None
                if move_uci:
                    san = self.board.san(chess.Move.from_uci(move_uci))
                score_obj = info.get("score")
                cp = None
                if score_obj is not None:
                    cp = score_obj.white().score(mate_score=1500)
                out.append((move_uci, san, cp, pv))
        return out
    
    def show_sf_overlay(self, token="sf"):
        """
        Parse token like 'sf' or 'sf20' to run stockfish.
          - 'sf' -> depth 16
          - 'sfNN' -> depth NN (e.g. sf20 -> depth 20)
        Prints SF top-3 moves (white-pov cp at depth) and if the chosen move
        isn't in top-3, prints its cp as well (forced evaluation).
        """
        # parse token
        if token == "sf":
            depth = 16
        else:
            try:
                depth = int(token[2:]) if token.startswith("sf") else 16
            except Exception:
                depth = 16
    
        print(f"Running Stockfish (depth={depth}, time cap=10s, multipv=3)...")
        topk = self.run_stockfish_topk(depth=depth, k=3)
        if not topk:
            print("No SF info (maybe engine failed).")
            return
    
        # print top-3
        print("SF top moves (white-POV cp):")
        top_ucis = set()
        for i, (uci, san, cp, pv) in enumerate(topk, start=1):
            cp_str = ("mate" if cp is None else str(int(cp)))
            print(f"  #{i}. {san:<6}  cp={cp_str}")
            if uci:
                top_ucis.add(uci)
    
        # check the move about to be played (selected from moves_uci[self.ply])
        if self.ply < len(self.moves_uci):
            upcoming_uci = self.moves_uci[self.ply]
            upcoming_san = self.board.san(chess.Move.from_uci(upcoming_uci))
            if upcoming_uci not in top_ucis:
                # get forced eval for the played move
                print(f"\nPlayed move {upcoming_san} not in SF top-3; computing cp...")
                limit = chess.engine.Limit(depth=depth, time=10.0)
                with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
                    info = eng.analyse(
                        self.board, limit=limit,
                        root_moves=[chess.Move.from_uci(upcoming_uci)]
                    )
                    
                    score_obj = info.get("score")
                    if score_obj is not None:
                        cp = score_obj.white().score(mate_score=1500)
                        print(f"  cp(played) = {int(cp)} (white-POV)")
                    else:
                        print("  played move eval not available")
            else:
                print(f"\nPlayed move {upcoming_san} is in SF top-3.")
    
    def align_sf_rows_to_json(self):
        self._sf_by_ply = {}
        if self.sf_rows is None:
            return
    
        seq = self.moves_uci or []
        j = 0  # cursor in JSON move list
        for ridx, r in self.sf_rows.iterrows():
            uci = r.get("played_move", "")
            if not uci:
                continue
    
            found = -1
            # requires the rows to be sorted by move_num correctly
            for k in range(j, len(seq)):
                if seq[k] == uci:
                    found = k
                    break
    
            if found >= 0:
                self._sf_by_ply[found] = ridx
                j = found + 1  # advance so next DF row maps to a later ply
            # else: no match; leave it unmapped (overlay will skip)
        
    def next(self):
        if self.ply < len(self.moves_uci):
            self.board.push_uci(self.moves_uci[self.ply])
            self.ply += 1

    def prev(self):
        if self.ply > 0:
            self.ply -= 1
            self.board.pop()
    
    def who_moved(self):
        mover = "White" if self.board.turn == chess.WHITE else "Black"
        
        vs_stockfish = self.log.get("vs_stockfish", False)
        if not vs_stockfish:
            return f"{mover} (MCTS)"
        
        sf_color = self.log.get("stockfish_color", None)
        if sf_color is None:
            return f"{mover} (MCTS)"
        
        if (self.board.turn and sf_color) or \
           (not self.board.turn and not sf_color):
            return f"{mover} (stockfish)"
        return f"{mover} (MCTS)"

    def show_board(self, flipped=False):
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=self.board, flipped=flipped)))

    def show_moves(self, top_n=5):
        if self.ply >= len(self.moves_uci):
            print("End of game.")
            return

        who = self.who_moved()
        chosen = self.moves_uci[self.ply]
        try:
            chosen_san = self.board.san(chess.Move.from_uci(chosen))
        except Exception:
            chosen_san = "?"

        print()
        print(f"Ply {self.ply+1}: {who} about to play {chosen_san}")
        print("=" * 60)

        node = self.tree_data.get(chosen) or {}
        if not node:
            node = self.tree_data.get(str(self.ply))

        if node is None:
            print("  (no candidate_moves in log)")
            return

        cands = node.get("candidate_moves") or []
        sims = node.get("sims", 0)
        t = node.get("time", 0.0)
        avg_d = node.get("avg_depth", 0.0)
        max_d = node.get("max_depth", 0)
        cv = node.get("children_visited", 0)
        tc = node.get("total_children", 0)
        
        uniq = node.get("unique_sims", None)
        line = (f"  sims={sims}  time={t:.2f}s  avg_depth={avg_d:.2f}  "
                f"max_depth={max_d} children visited={cv}/{tc}")
        if uniq is not None and sims:
            frac = uniq / max(1, sims)
            line += f"  unique={uniq} ({frac:.0%})"
        print(line)

        # entropy 
        visits_list = [c.get("visits", 0) for c in cands]
        ent, norm = calc_entropy(visits_list)
        print(f"  entropy: raw={ent:.3f} bits  norm={norm:.3f}")

        cands_sorted = sorted(
            cands, key=lambda x: x.get("visits", 0), reverse=True
        )
        is_sf_turn = ("stockfish" in str(who).lower())

        # index of SF's actual move among candidates (or None)
        sf_idx = None
        for i, c in enumerate(cands_sorted):
            if c.get("uci") == chosen:
                sf_idx = i
                break

        def print_row(c, mark=False, show_rank=False, rank_val=None):
            san = self.board.san(chess.Move.from_uci(c.get("uci", "")))
            marker = "  <- SF" if mark else ""
            rank_str = f"  (rank #{rank_val})" if (show_rank and rank_val) else ""
            print(
                f"   {san:<6} visits={c.get('visits',0):<5} "
                f"Q={c.get('Q',0):+.3f} P={c.get('P',0):.3f} U={c.get('U',0):+.3f}"
                f"{marker}{rank_str}"
            )

        shown_ucis = set()
        # top-N: never show ranks; just mark if SF move is in top-N
        for i, c in enumerate(cands_sorted[:top_n]):
            print_row(c, mark=is_sf_turn and (c.get("uci") == chosen))
            shown_ucis.add(c.get("uci"))

        # if SF's move exists but wasn't in top-N, show ellipsis + row WITH rank
        if is_sf_turn and sf_idx is not None:
            if cands_sorted[sf_idx].get("uci") not in shown_ucis:
                print("   ...")
                print_row(
                    cands_sorted[sf_idx], mark=True,
                    show_rank=True, rank_val=sf_idx + 1,
                )

        vwq = node.get("visit_weighted_Q")
        if vwq is not None:
            print(f"\nvisit-weighted Q={vwq}")

        # SF overlay (optional)
        r = self.sf_row_for_ply(self.ply)
        if r is not None:
            stm_white = (self.board.turn == chess.WHITE)

            best_uci   = str(r.get("best_move", "") or "")
            played_uci = str(r.get("played_move", "") or "")

            best_cp_raw   = r.get("best_cp", None)
            played_cp_raw = r.get("played_cp", None)

            def pov(cp):
                if cp is None or (isinstance(cp, float) and np.isnan(cp)):
                    return None
                return int(cp if stm_white else -cp)

            best_cp_pov   = pov(best_cp_raw)
            played_cp_pov = pov(played_cp_raw)

            loss = r.get("clipped_loss", r.get("loss", None))
            if loss is None and best_cp_pov is not None and played_cp_pov is not None:
                loss = max(0, best_cp_pov - played_cp_pov)

            # recompute match flag from UCIs to avoid DF drift
            matched = (best_uci == played_uci) if best_uci and played_uci else False

            def to_san(uci):
                return self.board.san(chess.Move.from_uci(uci))

            best_san   = to_san(best_uci) if best_uci else "?"
            played_san = to_san(played_uci) if played_uci else "?"

            parts = []
            if loss is not None:
                parts.append(f"CPL={int(loss)}")
            if played_san != "?":
                parts.append(f"played={played_san} ({'✓' if matched else '×'})")
            if best_san != "?":
                parts.append(f"SF best={best_san}")
            if (best_cp_pov is not None) and (played_cp_pov is not None):
                parts.append(f"cp(best/played)={best_cp_pov}/{played_cp_pov}")

            if parts:
                print("SF:", "  ".join(parts))
        print("=" * 60)

    def show_options(self):
        # concise CLI help for replay mode commands
        print("Commands:")
        print("  [Enter] / Space      forward one move")
        print("  b, back              previous move")
        print("  q, quit, exit        quit replay")
        print("  o, options, help     show this help text")
        print("  sf                   stockfish overlay (uses default depth)")
        print("  sf<D>                stockfish eval to depth D, e.g. sf12")
        print("  pv                   show principal variation (min_vis=1)")
        print("  pv<N>                show principal variation filtered by")
        print("                       minimum visits, e.g. pv8")

    def replay(self):
        print(f"Replaying {self.log['scenario']}. Result {self.log['result']}")
        # flip the board if sf plays white
        sf_color = self.log.get("stockfish_color", None)
        flipped = sf_color if sf_color else False
        print("Controls: Enter/Space=forward, b=back, q=quit, o=options")
        shown = False
        while True:
            if not shown:
                self.show_board(flipped=flipped)
                self.show_moves()
                shown = True
            cmd = input("[Enter]=fwd, b=back, q=quit, sf=stockfish eval, o=options > ")
            cmd = cmd.strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd in ("o", "options", "help", "h", "?"):
                self.show_options()
            elif cmd in ("b", "back"):
                shown = False
                self.prev()
            # elif cmd.startswith("pv"):
            #     # pv or pv8
            #     if cmd == "pv":
            #         self.show_pv(min_vis=1)
            #     else:
            #         try:
            #             n = int(cmd[2:])  # e.g. pv8
            #             self.show_pv(min_vis=n)
            #         except Exception:
            #             self.show_pv(min_vis=1)
            
            elif cmd.startswith("sf"):
                self.show_sf_overlay(cmd)  # cmd parsed for depth inside method
            else:
                # default: forward one move
                shown = False
                self.next()


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


def analyze_with_sf_core(game_data, eng, depth=DEPTH):
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


def make_fake_visits(mv, lms, ratio_best=50):
    visits = [[mv, int(ratio_best)]]
    
    # may only be 1 legal move
    if len(lms) == 1:
        return visits
    
    ratio_not_best = 100-ratio_best
    sup_optimal = max(1, int(ratio_not_best / (len(lms) - 1)))
    visits += [[m, int(sup_optimal)] for m in lms if m != mv]
    return visits


def make_training_sample(b, v, visits):
    # turn counts into move probabilites
    ucis   = [x[0] for x in visits]
    counts = np.array([x[1] for x in visits], dtype=np.float32)
    s = counts.sum()
    pi = (counts / s) if s > 0.0 else np.zeros_like(counts)
    
    # get indices from C++
    indices = b.moves_to_indices(ucis)  # list of ints (0..4095)
    policy = np.zeros(64 * 64, dtype=np.float32)

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
        
        tr = tree_data.get(i, tree_data.get(str(i), {}))
        cm = tr.get('candidate_moves', [])
        
        if cm:
            visits = [[c['uci'], c['visits']] for c in cm]
            visits = sorted(visits, key=lambda x: x[1], reverse=True)
        
        else:
            # make up fake visits if we dont have any
            visits = make_fake_visits(mv, lms, ratio_best=50)
        
        # if visits dont look right, skip
        if sum([v[1] for v in visits]) <= 0:
            b.push_uci(mv)
            continue
            
        # find the df row associated to this move
        row = df.query("played_move == @mv")
        if len(row) > 1:
            row = df.query("played_move == @mv").query("move_num == @i")
        if len(row) == 0:
            row = df.loc[df.move_num == str(i)].query("played_move == @mv")
        if len(row) == 0:
            continue
        
        # happy path, not a blunder, take stm pov sf eval
        if row['delta'].item() < BLUNDER_CP:
            #v = cp_to_value_tanh(row['played_cp'].item())
            #ts = make_training_sample(b, v, visits)
            #training_data.append(ts)
            b.push_uci(mv)
        
        # if a blunder, dont use actual visits (theyre wrong)
        else:
            best_v = cp_to_value_tanh(row['best_cp'].item())
            best_mv = row['best_move'].item()
            best_visits = make_fake_visits(best_mv, lms)
            ts_best = make_training_sample(b, best_v, best_visits)
            training_data.append(ts_best)
            
            # # furthermore, show the best continuation
            b2 = b.clone()
            b2.push_uci(best_mv)
            cont_moves = 1
            while cont_moves < 2 :
                lms2 = b2.legal_moves()
                if not lms2:
                    break
                cont_val, cont_best = sfe(b2, engine=engine)
                cont_visits = make_fake_visits(cont_best, lms2)
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
            
            # run the 2 next PV moves
            b2 = b.clone()
            for m in pv[:3]:
                b2.push_uci(m['uci'])
                lms2 = b2.legal_moves()
                if not lms2:
                    break
               
                pv_val, pv_best = sfe(b2, engine=engine)
                pv_visits = make_fake_visits(pv_best, lms2)
                pv_ts = make_training_sample(b2, pv_val, pv_visits)
                training_data.append(pv_ts)
            
            # push to move and let the loop roll over
            b.push_uci(mv)

    return training_data


def report_unprocessed(entries, seen_games):
    unproc = sum([1 for e in entries if e.get('game_id') not in seen_games])
    if unproc:
        print(f"{PH} {unproc} unprocessed game(s) currently in queue")


def post_hoc_worker(run_dir, bonus_data=True, batch_games=10, batch_secs=90):
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
    eng.configure({"Threads": 1, "Hash": 64})

    unproc_report = True
    try:
        # see if there is a starting pkl file
        last_seen_pkl = os.path.exists(pkl_path)
        next_threshold = 1000
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

                # create training tuples for this game (unchanged)
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
                            next_threshold = 1000
                        last_seen_pkl = curr_exists

                        running_list.extend(batch_samples)
                        save_pickle_atomic(running_list, pkl_path)
                        # reflect exact on-disk state
                        last_seen_pkl = True
                        lrl = len(running_list)
                        if lrl >= next_threshold:
                            next_threshold += 1000
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
        # ensure engine is cleanly quit
        eng.quit()
        print("[post_hoc] post_hoc_worker exiting cleanly")


def start_post_hoc_server(run_dir, bonus_data=True):
    """Start post-hoc worker process and forward bonus_data toggle."""
    p = Process(target=post_hoc_worker, args=(run_dir, bonus_data), daemon=False)
    p.start()
    print(f"[post hoc] started server pid={p.pid} bonus_data={bonus_data}")
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
