import os, json, pathlib, time
from pathlib import Path
import uuid
import sys
import random
import subprocess
from collections import deque

import pickle
import chess, chess.engine

import pandas as pd
import numpy as np

from pyfastchess import Board

from chessbot import SF_LOC
from chessbot.utils import (
    score_cp_stm_pov, score_cp_white_pov, rnd, cp_to_value_tanh, kl_divergence
)

RS = "[rescore]"
ANALYZE_PKL = "analyze_results_combined.pkl"

class Rescorer(object):
    eng = None

    def __init__(self, cfg):
        self.config = cfg
        self.training_data = []
        self.analyzed_results = []

        self.train_on_stockfish = cfg.train_on_stockfish
        self.train_on_validation = cfg.train_on_validation

        self.eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
        self.eng.configure(cfg.sf_config)

        self.start_time = time.time()
        self.games_seen = set()
        self.games_processed = 0
        self.written_total = 0
        self.written_this_round = 0
        self.n_saved = 0
        self.tcpl = 0
        self.tbmr = 0

        # stockfish time keeping
        self.n_sf_best = 0
        self.sf_best_time = 0

        self.n_sf_played = 0
        self.sf_played_time = 0
        # sf max will store the 10 longest sf search times, to reduce outliers
        self.sf_max = [0.0]*10
        self.last_10_cpls = []
        self.last_10_bmrs = []

        self.init_analyzer()

    def close(self):
        if self.eng is not None:
            self.eng.quit()
            self.eng = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def init_analyzer(self):
        run_dir = self.config.run_dir
        idx_path = os.path.join(run_dir, "game_index.json")

        # first merge any existing analysis
        _ = combine_analysis_staging(run_dir)

        # seen games from existing analyze pkl
        self.games_seen = set()
        analyze_pkl_path = os.path.join(run_dir, ANALYZE_PKL)

        if os.path.exists(analyze_pkl_path):
            with open(analyze_pkl_path, "rb") as f:
                combined = pickle.load(f)
            if "df_means" in combined and combined["df_means"] is not None:
                self.games_seen = set(
                    combined["df_means"]["game_id"].astype(str).tolist()
                )

    def get_unprocessed(self):
        run_dir = self.config.run_dir
        idx_path = os.path.join(run_dir, "game_index.json")
        idx = load_game_index(idx_path)
        entries = [idx] if isinstance(idx, dict) else idx

        unprocessed = deque()
        for rec in entries:
            gid = str(rec.get("game_id"))
            pkl_path = rec.get("pkl_file")
            if not pkl_path or not os.path.exists(pkl_path):
                continue
            if gid in self.games_seen:
                continue
            unprocessed.append(rec)
        
        if len(unprocessed):
            n = len(unprocessed)
            print(f"[rescore] {n} unprocessed game(s) currently in queue")
        
        return unprocessed

    def reset_writer(self):
        self.written_this_round = 0
        self.sf_best_time = 0
        self.n_sf_best = 0
        self.sf_played_time = 0
        self.n_sf_played = 0
        self.sf_max = [0.0]*10
    
    def append_flat_policy_example(self, board, ucis, visits, Y, vwht, pwht):
        """
        Snapshot inputs and a flat 4288-length policy vector for training.
        - ucis: list[str] legal moves (same order as probs)
        - pi:  list/array of probs (sum ~= 1)
        - Y: target for value head
        """
        cfg = self.config
        # get indices from C++
        indices = board.moves_to_indices(ucis)  # list of int (0..4288)
        policy = np.zeros(64 * 67, dtype=np.float32)

        # normalize visits -> pi
        s = sum(visits)
        pi = np.array([v / s for v in visits], dtype=np.float32)
        pi = np.clip(pi, cfg.prior_clip_min, cfg.prior_clip_max)
        pi = pi / pi.sum()

        # accumulate probs into flattened policy
        for idx, p in zip(indices, pi):
            policy[idx] += p

        # snapshot inputs and push example
        x = board.encode_64_tokens()
        mask = board.legal_move_mask()

        self.training_data.append((x, mask, policy, Y, vwht, pwht))
    
    def write_training_data_pkl(self, size=None, randomize=True):
        cfg = self.config
        out_dir = pathlib.Path(cfg.pending_training_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # we need to clear out any existing pkls because theyre probably corrupted
        deleted = []
        for p in out_dir.iterdir():
            if not p.is_file():
                continue

            suf = p.suffix.lower()
            if suf in [".pkl", ".pickle"]:
                p.unlink()
                deleted.append(p.name)

        if deleted:
            print(f"{RS} deleted {len(deleted)} stale pkls in {out_dir}")

        if randomize:
            random.shuffle(self.training_data)

        if size is None:
            size = cfg.get_retrain_size

        chunk = self.training_data[:size]
        remainder = self.training_data[size:]

        filename = f"{int(time.time())}-{uuid.uuid4().hex}.pkl"
        out_path = out_dir / filename

        with open(out_path, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.training_data = remainder
        self.written_this_round += len(chunk)
        self.written_total += len(chunk)

    def training_data_from_sf(self, board, mv, cm, Y):
        cfg = self.config

        rows = [(c['uci'], c['visits']) for c in cm]
        total_visits = sum([n for _, n in rows]) if rows else 0

        visit_map = None
        use_tree_visits = False

        # make sure we have at least 10 visits for stability
        if rows and total_visits > 10:
            visit_map = {u: n for u, n in rows}
            most_visited_uci, max_visits = max(rows, key=lambda x: x[1])

            if most_visited_uci == mv:
                use_tree_visits = True
            else:
                # boost top visited with SF move
                top_count = visit_map[most_visited_uci]
                visit_map[mv] = 1 + int(top_count*1.25)
                use_tree_visits = True

        if use_tree_visits and visit_map is not None:
            ucis = board.legal_moves()
            visits = [max(1, visit_map.get(u, 1)) for u in ucis]
        else:
            raw = make_fake_visits(mv, board.legal_moves(), ratio_best=60)
            ucis = [x[0] for x in raw]
            visits = [int(x[1]) for x in raw]

        # calc KL divergence after adjustments
        priors_map = {c['uci']: c['P'] for c in cm}
        priors = [priors_map.get(u, 0.0) for u in ucis]
        kl = kl_divergence(priors, visits)

        vwht = cfg.value_loss_weight
        pwht = cfg.policy_loss_weight
        if kl > cfg.KL_boost_threshold:
            pwht *= cfg.KL_weight_boost

        self.append_flat_policy_example(board, ucis, visits, Y, vwht, pwht)
    
    def analyze_with_rank(self, move, board):
        # Get Stockfish root best at this depth (white-POV score included in info)
        cfg = self.config
        eng = self.eng
        limit = chess.engine.Limit(depth=cfg.post_hoc_depth)
        info_all = chess.engine.INFO_ALL

        t0 = time.perf_counter()
        top1 = eng.analyse(board, limit=limit, info=info_all)
        sf_time = time.perf_counter() - t0
        self.sf_best_time += sf_time
        self.n_sf_best += 1

        # collect the max time 
        if sf_time > self.sf_max[0]:
            self.sf_max.append(sf_time)
            self.sf_max = sorted(self.sf_max)[-10:]

        best_move = top1['pv'][0]
        best_cp   = score_cp_stm_pov(top1["score"])
        best_abs  = score_cp_white_pov(top1["score"], clipped=False)
        
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
            return res

        t0 = time.perf_counter()
        played = eng.analyse(board, limit=limit, root_moves=[move], info=info_all)
        
        sf_time = time.perf_counter() - t0
        self.sf_played_time += sf_time
        self.n_sf_played += 1

        # collect the max time 
        if sf_time > self.sf_max[0]:
            self.sf_max.append(sf_time)
            self.sf_max = sorted(self.sf_max)[-10:]

        played_cp = score_cp_stm_pov(played['score'])
        played_abs = score_cp_white_pov(played["score"], clipped=False)
        delta = best_cp - played_cp
        
        # within equivalence range -> wash: treat as equal, loss=0 and mark both as best
        EQUIV_RANGE = cfg.post_hoc_equiv_range
        if abs(delta) <= EQUIV_RANGE:
            res['best_move'] = move # our move is also best
            res['played_cp'] = best_cp
            res['played_absolute'] = best_abs
            res['delta_signed'] = 0
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
            return res
        
        # otherwise, our move is worse
        res['played_cp'] = played_cp
        res['played_absolute'] = played_abs
        res['delta_signed'] = delta
        return res

    def analyze_and_rescore(self, game_data):
        """
        Single-pass analyze + rescore training-data maker.

        - train_on_stockfish: include plies played by SF in training if True.
        - base_weight: default sample weight used as 'weight' in returned sample.
        """
        cfg = self.config

        # allow passing a filepath (str or Path) to a pkl/json game record
        if isinstance(game_data, (str, pathlib.Path)):
            path = str(game_data)
            if path.lower().endswith(('.pkl', '.json')) and os.path.exists(path):
                if path.lower().endswith('.pkl'):
                    with open(path, 'rb') as f:
                        game_data = pickle.load(f)
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        game_data = json.load(f)

        # must be a dict from here on
        if not isinstance(game_data, dict):
            raise TypeError("game_data must be a dict or path to .pkl/.json")

        gid = game_data.get("game_id")
        if gid in self.games_seen:
            return
        
        board_ch = chess.Board(game_data['start_fen'])
        b_fast = Board(game_data['start_fen'])

        tree_data = game_data.get('tree_search_data', {})
        vs_stockfish = game_data.get('vs_stockfish', False)
        sf_color = game_data.get('stockfish_is_white')
        result = game_data['result']

        cpl_s = cpl_w = cpl_b = 0.0
        nw = nb = 0
        rows = []
        
        # set a flag in case we want to skip all training data
        skip_all_training = False
        if not self.train_on_validation:
            if 'validation' in game_data['scenario'].lower():
                skip_all_training = True
        
        KL_coef = self.config.KL_weight_boost
        do_KL_boost = (KL_coef > 0) and (KL_coef != 1.0)
        
        for i, mv in enumerate(game_data.get('moves_played', [])):
            move_ch = chess.Move.from_uci(mv)
            is_sf_move = vs_stockfish and (board_ch.turn == sf_color)
            turn = board_ch.turn
            
            # handle sf-played plies
            if is_sf_move and ((not self.train_on_stockfish) or (skip_all_training)):
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            # tree data for this move
            tr = tree_data.get(i, tree_data.get(str(i), {}))
            cm = tr.get('candidate_moves', [])
            if not cm:
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            # calculate Y value
            Z_stm = result if turn else -1*result
            if 'Q_stm' in tr.keys():
                Q = tr['Q_stm']
            else:
                this_q = tr.get("best_Q", tr.get("visit_weighted_Q"))
                Q = this_q if turn else -1*this_q
            

            Y = np.clip(0.5*Z_stm + 0.5*Q, -1.0, 1.0)

            if Y != Y:
                print("[rescore] nan value detected for Y")
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            # if sf_move, gather training data, push moves, continue
            if is_sf_move:
                self.training_data_from_sf(b_fast, mv, cm, Y)
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            # if here, its MCTS move
            # move_played may not be most visited due to temp sampling
            # so inspect visits
            visits = [(c['uci'], max(1, c['visits'])) for c in cm]
            visits = sorted(visits, key=lambda x: x[1], reverse=True)

            most_visited_uci = visits[0][0]
            most_visited_ch = chess.Move.from_uci(most_visited_uci)
            res = self.analyze_with_rank(most_visited_ch, board_ch)

            loss_this = res['delta_signed']
            cpl_s += loss_this
            if board_ch.turn:
                cpl_w += loss_this
                nw += 1
            else:
                cpl_b += loss_this
                nb += 1
            
            # we penalize missed-mate-but-still-winning less harshly
            missed_mate = (res.get('best_cp', 0) >= 1200) and (res['played_cp'] >= 500)
            if missed_mate:
                # cap loss_this at 300, we are still winning here
                loss_this = min(300, loss_this)
            
            rows.append([
                i, mv, most_visited_uci, str(res['best_move']),
                res['best_cp'], loss_this, res['played_cp'],
                res['best_absolute'], res['played_absolute'],
                board_ch.turn, loss_this
            ])

            if skip_all_training:
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            # screen training data, adjust if needed and append
            lms = b_fast.legal_moves()

            # determine correct cp threshold
            blunder_cp = cfg.post_hoc_blunder_cp_loser
            if Z_stm > 0.0:
                blunder_cp = cfg.post_hoc_blunder_cp_winner

            # these moves are fine, no changes            
            if loss_this <= min(60, blunder_cp):
                best_mv = mv
                visits = ensure_all_legal_moves_have_visits(visits, lms)
            
            # for mild blunders or missed-mate-but-still-winning, adjust visits
            elif (loss_this < blunder_cp) or missed_mate:
                best_mv = str(res.get('best_move'))
                visits = adjust_visits_from_cm(cm, mv, best_mv, lms, was_blunder=False)

            # everything else is a blunder
            else:
                best_mv = str(res.get('best_move'))
                visits = adjust_visits_from_cm(cm, mv, best_mv, lms, was_blunder=True)
                # use the SF value here since its a blunder
                best_cp = res.get('best_cp')
                Y = cp_to_value_tanh(best_cp) if best_cp is not None else 0.0

            if not visits or sum([v[1] for v in visits]) <= 0:
                print("[rescorer] visits invalid or sum <= 0; skipping sample",
                    "move_idx=", i, "played=", mv, "best=", best_mv)
                board_ch.push(move_ch)
                b_fast.push_uci(mv)
                continue

            mvs = [v[0] for v in visits]
            vis = [v[1] for v in visits]

            priors_map = {c['uci']: c['P'] for c in cm}
            priors = [priors_map.get(u, 0.0) for u in mvs]

            vwht = cfg.value_loss_weight
            pwht = cfg.policy_loss_weight

            # adjust training weights based on KL
            if do_KL_boost:
                kl = kl_divergence(priors, vis)
                if kl >= cfg.KL_boost_threshold:
                    pwht *= KL_coef
            
            self.append_flat_policy_example(b_fast, mvs, vis, Y, vwht, pwht)
            
            board_ch.push(move_ch)
            b_fast.push_uci(mv)

        # assemble df and summary
        cols = [
            'move_num', 'played_move', 'most_visited_move', 'best_move',
            'best_cp', 'delta', 'played_cp',
            'best_absolute', 'played_absolute', 'stm', 'loss'
        ]

        out_df = pd.DataFrame(rows, columns=cols)
        out_df['played_best_move'] = out_df['most_visited_move'] == out_df['best_move']

        mask_w = out_df['stm'] == True
        mask_b = out_df['stm'] == False

        overall_bmr = out_df['played_best_move'].mean() if len(out_df) else np.nan
        white_bmr = out_df.loc[mask_w, 'played_best_move'].mean() if mask_w.any() else np.nan
        black_bmr = out_df.loc[mask_b, 'played_best_move'].mean() if mask_b.any() else np.nan

        out = {
            'plies': nw + nb,
            'overall_cpl': rnd(cpl_s / (nw + nb), 3) if (nw + nb) else np.nan,
            'white_cpl': rnd(cpl_w / nw, 3) if nw else np.nan,
            'black_cpl': rnd(cpl_b / nb, 3) if nb else np.nan,
            'overall_best_move_rate': overall_bmr,
            'best_move_rate_white': white_bmr,
            'best_move_rate_black': black_bmr
        }

        for key in ['game_id', 'scenario', 'stockfish_color', 'ts']:
            val = game_data.get(key)
            if key == 'ts':
                val = int(val)
                
            out[key] = val
            out_df[key] = val

        out['df'] = out_df
        self.analyzed_results.append(out)
        self.games_processed += 1
        self.games_seen.add(gid)
        if len(self.analyzed_results) >= self.config.post_hoc_analyze_batch:
            self.push_analyzed(report=True)
        
    def push_analyzed(self, report=True):
        # safeguard here
        if not len(self.analyzed_results):
            return
        
        run_dir = self.config.run_dir
        outp, c, b = save_analysis_chunk_simple(run_dir, self.analyzed_results)
        self.n_saved += 1    
        self.tcpl += c; self.tbmr += b

        # update the rolling windows
        self.last_10_cpls.append(c)
        self.last_10_cpls = self.last_10_cpls[-10:]
        self.last_10_bmrs.append(b)
        self.last_10_bmrs = self.last_10_bmrs[-10:]

        if report:
            if len(self.last_10_cpls) >= 10:
                last_10_avg_c = np.mean(self.last_10_cpls)
                last_10_avg_b = np.mean(self.last_10_bmrs)
                print(
                    f"{RS} {'Last 10 avg:':<16} CPL {last_10_avg_c:.3f}",
                    f"BMR {last_10_avg_b:.3f}"
                )

            if self.n_saved >= 2:
                cpl_mean = self.tcpl/self.n_saved
                tmbr_mean = self.tbmr/self.n_saved 
                print(
                    f"{RS} {'Overall stats:':<16} CPL {cpl_mean:.3f}",
                    f"BMR {tmbr_mean:.3f}"
                )

            n_tot = self.games_processed
            if n_tot > self.config.post_hoc_analyze_batch:
                rate = n_tot / (time.time() - self.start_time)
                print(f"{RS} Total Games: {n_tot} ({rate:.3f} games/sec)")
            
            # SF timing summary
            if self.n_sf_best > 0:
                avg_first = self.sf_best_time / self.n_sf_best
                if self.n_sf_played > 0:
                    avg_rerun = self.sf_played_time / self.n_sf_played
                else:
                    avg_rerun = 0.0

                total = int(self.n_sf_played + self.n_sf_best)
                rerun_rate = (self.n_sf_played / self.n_sf_best)
                expected = avg_first + rerun_rate * avg_rerun
                
                print(f"{RS} SF timing: Total {total} | rerun rate {rerun_rate:.3f}")
                print(
                    f"{RS} SF timing: avg_first {avg_first:.3f} "
                    f"avg_rerun {avg_rerun:.3f} exp {expected:.3f} "
                    f"max {self.sf_max[-1]:.3f}"
                )
            
            w_this = self.written_this_round
            wtot = self.written_total
            print(f"{RS} Training samples this round: {w_this} | total: {wtot}")
        
        self.analyzed_results = []

# helpers
def ensure_all_legal_moves_have_visits(visit_pairs, lms):
    """
    visit_pairs: list of (uci, visits) or [uci, visits]
    lms: list of legal move UCIs
    Ensures every legal move appears with visits >= 1.
    Returns list of [uci, int_visits] sorted desc.
    """
    d = {}
    for u, v in visit_pairs:
        if not u:
            continue
        d[u] = max(1, int(v))

    for m in lms:
        if m not in d:
            d[m] = 1

    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return [[u, int(v)] for u, v in items]


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
    if not os.path.exists(path):
        return []
    return load_json(path)


def dedupe_results(results):
    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset='game_id', keep='first')
    # dont want the raw key, its just extra data
    if 'raw' in df.columns:
        df = df.drop(columns=['raw'])
    
    return df.to_dict(orient='records')


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

    return {
        "games": len(results),
        "avg_white_mean_cpl": round(safe_mean(wm), 3),
        "avg_black_mean_cpl": round(safe_mean(bm), 3),
        "avg_overall_mean_cpl": round(safe_mean(om), 3),
        "avg_best_move_rate_white": round(safe_mean(wb), 3),
        "avg_best_move_rate_black": round(safe_mean(bb), 3),
        "avg_overall_best_move_rate": round(safe_mean(ob), 3)
    }


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
    print(f"[combine] wrote combined ANALYZE_PKL -> {Path(out_pkl).name} "
          f"({len(merged_results)} games)")

    # delete the chunk files that we just combined
    for fn in fns:
        path = os.path.join(staging, fn)
        os.remove(path)
    print(f"[combine] removed {len(fns)} chunk files from {Path(staging).name}")

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
            "overall_best_move_rate": analysis_out.get("overall_best_move_rate")
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
    
    print(f"{RS} Saving {len(batch)} analyzed games")
    print(f"{RS} {'Batch stats:':<16} CPL {cpl:.3f} BMR {bmr:.3f}")

    fname = f"{int(time.time())}_{uuid.uuid4().hex}.pkl"
    outp = os.path.join(staging, fname)

    with open(outp, "wb") as f:
        pickle.dump(chunk_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return outp, cpl, bmr


def make_fake_visits(mv, lms, ratio_best=60):
    visits = [[mv, int(ratio_best)]]
    
    # may only be 1 legal move
    if len(lms) < 2:
        return visits
    
    sub_optimal = 100 - ratio_best
    bad_visits = 1 + min(5, int(sub_optimal / len(lms)))
    visits += [[m, bad_visits] for m in lms if m != mv]
    return visits


def adjust_visits_from_cm(cm, played_mv, best_mv, lms, was_blunder=False):
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
    old_min = min(d.values()) if d else 1

    # adjust by swapping visits between played and best with 30% bump
    d[played_mv] = int(np.ceil(0.7*d[best_mv]))
    if was_blunder:
        d[best_mv] = old_min
    else:
        d[best_mv] = int(np.ceil(1.3*old_max))

    # build sorted list
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return [[u, int(v)] for u, v in items]


def launch_retrain_async(run_tag, rt_script, working_cfg):
    cmd = [sys.executable, rt_script, "--run-dir", working_cfg.run_dir]
    cmd += ["--batch-size", str(working_cfg.retrain_batch_size)]

    print(f"[retrain] launching worker")

    start_new_session = False
    creationflags = 0
    if os.name == "posix":
        start_new_session = True
    else:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=start_new_session,
            creationflags=creationflags,
            text=True,
            bufsize=1,   # line buffered
        )
    
    except Exception as e:
        raise RuntimeError(f"[retrain] failed to start retrain worker: {e}")

    return {
        "p": p,
        "cmd": cmd,
        "stdout_buf": deque(),
        "stderr_buf": deque(),
        "done": False,
        "rc": None,
    }


def drain_pipe_lines(pipe, buf, max_lines=200):
    n = 0
    if pipe is None:
        return 0

    # Use readline() in a bounded loop; it may block if no newline is available.
    # To keep this non-blocking, only call it when poll() indicates process ended,
    # OR keep max_lines small and accept that it can block if the worker writes
    # partial lines without '\n'. Most scripts print lines, so this is usually fine.
    while n < max_lines:
        ln = pipe.readline()
        if not ln:
            break
        buf.append(ln.rstrip("\n"))
        n += 1
    return n


def poll_retrain(handle, print_output=True):
    """
    Call frequently from your main loop.
    Returns: (done: bool, rc: int | None)
    """
    p = handle["p"]

    rc = p.poll()
    handle["rc"] = rc

    # If you want "live" output while running, you need non-blocking IO (selectors)
    # or a reader thread. The simple safe option: only drain once it's done.
    if rc is None:
        return False, None

    # Process ended: drain remaining output fully
    drain_pipe_lines(p.stdout, handle["stdout_buf"], max_lines=10_000)
    drain_pipe_lines(p.stderr, handle["stderr_buf"], max_lines=10_000)

    if print_output:
        if handle["stdout_buf"]:
            print("[retrain] STDOUT:")
            while handle["stdout_buf"]:
                print(handle["stdout_buf"].popleft())

        if handle["stderr_buf"]:
            print("[retrain] STDERR:")
            while handle["stderr_buf"]:
                print(handle["stderr_buf"].popleft())

    handle["done"] = True
    print(f"[retrain] worker finished exit_code={rc}")

    # Close pipes to release resources
    if p.stdout is not None:
        p.stdout.close()
    if p.stderr is not None:
        p.stderr.close()

    if rc != 0:
        raise RuntimeError(f"[retrain] worker failed; exit_code={rc}")

    return True, rc


def reclaim_vram(mb):
    import tensorflow as tf

    bytes_target = mb * 1024 * 1024
    n = max(1, bytes_target // 4)

    with tf.device("/GPU:0"):
        x = tf.ones([n], dtype=tf.float32)
        y = tf.reduce_sum(x)

    _ = y.numpy()
    return True
