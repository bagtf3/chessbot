import os, json, pathlib, time
import psutil
import uuid

import pickle
from collections import defaultdict

import chess, chess.svg
from IPython.display import SVG, display, clear_output
import chess.engine

import signal
from multiprocessing import Process

import pandas as pd
import numpy as np

from pyfastchess import Board

from chessbot import SF_LOC
from chessbot.utils import print_recent_summary, format_time
from chessbot.utils import (
    score_cp_stm_pov, score_cp_white_pov, score_to_value_stm_pov, rnd,
    calc_entropy, cp_to_value_tanh, sf_eval, kl_divergence_bits
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


class GameViewer:
    def __init__(self, log_path, sf_df=None):
        self.path = pathlib.Path(log_path)
        if not self.path.exists():
            raise FileNotFoundError(f"log file not found: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix == ".json":
            with open(self.path, "r", encoding="utf-8") as f:
                self.log = json.load(f)
        elif suffix in (".pkl", ".pickle"):
            with open(self.path, "rb") as f:
                self.log = pickle.load(f)
        else:
            raise Exception("log_path must be json or pickle")
        
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
        self.board = Board(self.start_fen)
        self.ply = 0  # 0 = before first move

    def goto(self, ply):
        ply = max(0, min(ply, len(self.moves_uci)))
        self.board = Board(self.start_fen)
        for u in self.moves_uci[:ply]:
            self.board.push_uci(u)
        self.ply = ply
        return self
    
    def turn(self):
        return self.board.side_to_move() == 'w'
    
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
        """
        out = []
        limit = chess.engine.Limit(depth=depth, time=10.0)
        with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
            sf_board = chess.Board(self.board.fen())
            infos = eng.analyse(sf_board, limit=limit, multipv=k)
            if not isinstance(infos, list):
                infos = [infos]
            for info in infos:
                pv = info.get("pv", [])
                move_uci = pv[0].uci() if pv else None
                san = None
                if move_uci:
                    san = self.board.san(move_uci)
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
        isn't in top-3, prints its cp as well
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
            upcoming_san = self.board.san(upcoming_uci)
            if upcoming_uci not in top_ucis:
                # get forced eval for the played move
                print(f"\nPlayed move {upcoming_san} not in SF top-3; computing cp...")
                limit = chess.engine.Limit(depth=depth, time=10.0)
                with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
                    sf_board = chess.Board(self.board.fen())
                    info = eng.analyse(
                        sf_board, limit=limit,
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
            self.board.unmake()
    
    def who_moved(self):
        mover = "White" if self.turn() else "Black"
        
        vs_stockfish = self.log.get("vs_stockfish", False)
        if not vs_stockfish:
            return f"{mover} (MCTS)"
        
        sf_color = self.log.get("stockfish_color", None)
        if sf_color is None:
            return f"{mover} (MCTS)"
        
        if (self.turn() and sf_color) or \
           (not self.turn() and not sf_color):
            return f"{mover} (stockfish)"
        return f"{mover} (MCTS)"

    def show_board(self, flipped=False):
        clear_output(wait=True)
        chess_board = chess.Board(self.board.fen())
        display(SVG(chess.svg.board(board=chess_board, flipped=flipped)))

    def node_who_chosen(self):
        """ conveniently return node, who move, what move was chosen """
        if self.ply >= len(self.moves_uci):
            return None, None, None

        chosen = self.moves_uci[self.ply]

        # very old games use the uci to index
        node = self.tree_data.get(chosen) or {}
        if not node:
            # try string
            node = self.tree_data.get(str(self.ply))
            
            # then int
        if not node:
            node = self.tree_data.get(self.ply)

        who = self.who_moved()

        return node, who, chosen        

    def show_pv(self, min_vis=1):
        """
        Print principal variation recorded in the tree data for the current ply.
        Print all PV moves (subject to min_vis), marking with ' *' those moves
        that were actually played in the game at the corresponding ply.
        """
        if self.ply >= len(self.moves_uci):
            print("End of game.")
            return

        node, who, chosen = self.node_who_chosen()

        if not node:
            print("No PV available for this node.")
            return

        pv = node.get("pv") or []
        if not pv:
            print("No PV recorded.")
            return

        board_tmp = chess.Board(self.board.fen())
        rows = []
        for i, step in enumerate(pv):
            visits = int(step.get("visits", 0))
            if visits < min_vis:
                break
            uci = step.get("uci")
            if not uci:
                break

            idx = self.ply + i
            played = (idx < len(self.moves_uci) and self.moves_uci[idx] == uci)

            san = board_tmp.san(chess.Move.from_uci(uci))
            p = step.get("P", 0.0)
            q = step.get("Q", 0.0)

            rows.append((i + 1, san, uci, visits, float(p), float(q), played))
            board_tmp.push_uci(uci)

        if not rows:
            print(f"No PV moves meet min_vis={min_vis}")
            return

        san_w = max(8, max(len(r[1]) for r in rows))
        uci_w = max(6, max(len(r[2]) for r in rows))

        for no, san, uci, visits, p, q, played in rows:
            star = " *" if played else ""
            line = (f"{no:3d}. {san:<{san_w}}  uci={uci:<{uci_w}}  "
                    f"visits={visits:5d}  P={p:0.4f}  Q={q:0.4f}{star}")
            print(line)
    
    def print_row(self, c, mark=False, show_rank=False, rank_val=None):
        san = self.board.san(c.get("uci", ""))
        marker = "  <- SF" if mark else ""
        rank_str = f"  (rank #{rank_val})" if (show_rank and rank_val) else ""
        Q = c.get('Q',0)
        P = c.get('P',0)
        U = c.get('U',0)
        cPUCT = self.log.get('c_puct', 1.5)
        Qrel = Q if self.board.side_to_move() == 'w' else -1*Q
        PUCT = Qrel + cPUCT*U
        print(
            f"   {san:<6} visits={c.get('visits',0):<5} "
            f"Q: {Q:+.3f}  P: {P:.3f}  PUCT: {PUCT:+.3f}"
            f"{marker}{rank_str}"
        )
    
    def show_moves(self, top_n=5):
        if self.ply >= len(self.moves_uci):
            print("End of game.")
            return

        node, who, chosen = self.node_who_chosen()

        try:
            chosen_san = self.board.san(chosen)
        except Exception:
            chosen_san = "?"

        print()
        print(f"Ply {self.ply+1}: {who} about to play {chosen_san}")
        print("=" * 60)

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

        # entropy: show normed entropy for visits and priors + KL/JS
        visits_list = [c.get("visits", 0) for c in cands]
        priors_list = [c.get("P", 0.0) for c in cands]

        norm_vis, p_vis = self.compute_norm_entropy(visits_list)
        norm_pri, p_pri = self.compute_norm_entropy(priors_list)
        kl = kl_divergence_bits(p_vis, p_pri)

        print(
            f"  entropy (norm'd): visits={norm_vis:.3f} "
            f"priors={norm_pri:.3f} KL(vis||pr)={kl:.3f} bits"
        )

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

        shown_ucis = set()
        # top-N: never show ranks; just mark if SF move is in top-N
        for i, c in enumerate(cands_sorted[:top_n]):
            self.print_row(c, mark=is_sf_turn and (c.get("uci") == chosen))
            shown_ucis.add(c.get("uci"))

        # if SF's move exists but wasn't in top-N, show ellipsis + row WITH rank
        if is_sf_turn and sf_idx is not None:
            if cands_sorted[sf_idx].get("uci") not in shown_ucis:
                print("   ...")
                self.print_row(
                    cands_sorted[sf_idx], mark=True,
                    show_rank=True, rank_val=sf_idx + 1,
                )

        this_q = node.get("best_Q")
        if this_q is None:
            this_q = node.get("visit_weighted_Q")
            if this_q is not None:
                print(f"\nvisit-weighted Q={this_q}")
        else:
            print(f"\nmost-visted Q={this_q}")

        # SF overlay (optional)
        r = self.sf_row_for_ply(self.ply)
        if r is not None:
            stm_white = self.turn()

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
                return self.board.san(uci)

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

    def show_visits(self, uci_or_san):
        node, who, chosen = self.node_who_chosen()
        if node is None:
            print(f"No visit info available for {uci_or_san}")
            return

        cands = node.get("candidate_moves") or []
        scands = sorted(cands, key=lambda x: x['visits'], reverse=True)
        move = {}
        rank = 0
        for c in scands:
            rank += 1
            if uci_or_san in [c['uci'], self.board.san(c['uci'])]:
                move = c
                break
        
        if not move:
            print(f"No visit info available for {uci_or_san}")
            return
        
        print(f"  === Showing visit info for {uci_or_san} ===")
        self.print_row(move)
        print(f"   Rank: {rank}\tShare: {100*move['visits']/node['sims']:.3f}%")

    def show_options(self):
        # concise CLI help for replay mode commands
        print("Commands:")
        print("  [Enter] / Space      forward one move")
        print("  b<N>                 go back N moves, e.g. b5 goes back 5 moves")
        print("  b, back              previous move (same as b1)")
        print("  q, quit, exit        quit replay")
        print("  o, options, help     show this help text")
        print("  fen                  print FEN for current position")
        print("  sf                   stockfish overlay (uses default depth)")
        print("  sf<D>                stockfish eval to depth D, e.g. sf12")
        print("  visits <move>        show MCTS visits for specific move")
        print("  visits <N>           show visit info for N top moves")
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
            cmd = cmd.strip()
            cmd_cased = cmd
            cmd = cmd.lower()
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd in ("o", "options", "help", "h", "?"):
                self.show_options()
            elif cmd == "fen":
                print(self.board.fen())
            elif cmd.startswith("pv"):
                # pv or pvN (e.g. pv8)
                if cmd == "pv":
                    n = 1
                else:
                    s = cmd[2:]
                    n = int(s) if s.isdigit() else 1
                self.show_pv(min_vis=n)
            elif cmd.startswith("sf"):
                self.show_sf_overlay(cmd)  # cmd parsed for depth inside method
            elif cmd.startswith("b"):
                # b, back, b5, b 5 all supported
                s = cmd[1:].strip()
                if not s:
                    n = 1
                elif s.isdigit():
                    n = int(s)
                else:
                    # fallback to single step back
                    n = 1
                # use goto to rebuild board safely and clamp bounds
                target = max(0, self.ply - n)
                shown = False
                self.goto(target)
            elif cmd.startswith("visits"):
                split = [c.strip() for c in cmd.split(" ") if c.strip()]
                # check to see if a int was given to show deeper visit info
                if split[-1].isdigit():
                    n = int(split[-1])
                    self.show_moves(top_n=n)
                else:
                    move_str = cmd_cased.replace(" ", "").replace("visits", "")
                    self.show_visits(move_str)
            else:
                # default: forward one move
                shown = False
                self.next()

    def compute_norm_entropy(self, arr):
        """Return (normalized_entropy_bits, prob_vector)."""
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return 0.0, np.array([], dtype=np.float64)
        s = a.sum()
        if s <= 0.0:
            p = np.ones(a.size, dtype=np.float64) / a.size
        else:
            p = a / s
        
        ent, norm = calc_entropy(p)
        return norm, p

    def generate_training_data(self, sf_skip=False, **kwargs):
        """
        Walk the game using self.next() and produce Xerces training examples.
        If sf_skip is True, plies played by Stockfish (per who_moved()) are
        skipped.
        """
        # X, Mask, Pi, result (Z), this_Q, moves Remaining
        X, M, P, Z, V, R = [], [], [], [], [], []
        result = self.result

        # start from initial position
        self.reset()

        total_plies = len(self.moves_uci)
        while self.ply < total_plies:
            # detect whether the side to move is Stockfish
            move_played = self.moves_uci[self.ply]
            mover = self.who_moved().lower()
            is_white_move = "white" in mover
            is_sf_move = "stockfish" in mover

            if sf_skip and is_sf_move:
                # advance and skip this ply
                self.next()
                continue

            rb = self.board
            lms = rb.legal_moves()
            
            node = self.tree_data.get(str(self.ply), {})
            if not node:
                self.next()
                continue
            
            cms = node.get("candidate_moves", {})
            if cms:
                visits = [[x['uci'], x['visits']] for x in cms]
                visited = set([x[0] for x in visits])
                
                # add in all legal moves if missing
                for move in [l for l in lms if l not in visited]:
                    visited.append([[move, 1]])
                    
                visits = sorted(visits, key=lambda x: x[1], reverse=True)
                
                if is_sf_move and visits[0][0] != move_played:
                    if sum([x[1] for x in visits]) > 100:
                        most_visited = visits[0][0]
                        for v in visits:
                            if v[0] == move_played:
                                v[0] = most_visited
                                break
                        visits[0][0] = move_played
                    else:
                        visits = make_fake_visits(move_played, lms, ratio_best=51)
            
            else:
                # legal moves and synthetic visits
                visits = make_fake_visits(move_played, lms, ratio_best=51)

            # check_boost = kwargs.get("check_boost", 0)
            # capture_boost = kwargs.get("capture_boost", 0)
            # if check_boost or capture_boost:
            #     for i, (move, v) in enumerate(visits):
            #         if rb.gives_check(move):
            #             visits[i][1] += check_boost
            #         if rb.is_capture(move):
            #             visits[i][1] += capture_boost

            counts = np.array([x[1] for x in visits], dtype=np.float32)
            s = counts.sum()
            if s > 0.0:
                pi = counts / s
            else:
                # uniform fallback over legal moves
                n_l = len(lms)
                if n_l == 0:
                    self.next()
                    continue
                pi = np.ones(n_l, dtype=np.float32) / float(n_l)
            
            # map legal moves -> flat xerces indices and build policy
            
            visited = [x[0] for x in visits]
            indices = rb.moves_to_indices(visited)
            policy = np.zeros(64 * 67, dtype=np.float32)
            for idx, prob in zip(indices, pi):
                policy[idx] += prob

            # inputs and mask
            x = rb.encode_64_tokens()
            mask = rb.legal_move_mask()

            # value target 0.5*Z + 0.5*best_q (q of most visited child)
            this_q = node.get("best_Q", node.get("visit_weighted_Q"))
            this_q_stm = this_q if is_white_move else -this_q
            if result > 0:
                z = 1 if is_white_move else -1
            elif result < 0:
                z = -1 if is_white_move else 1
            else:
                z = 0
            
            X.append(x)
            M.append(mask)
            P.append(policy)
            Z.append(z)
            V.append(this_q_stm)
            R.append(len(self.moves_uci) - int(self.ply))

            # advance to next ply using existing helper
            self.next()

        return X, M, P, Z, V, R


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
    # (x, mask, policy, z_stm, best_q, z_tapered)
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


class RecordKeeper(object):    
    def __init__(self, n_retrains, run_num=None, every_sec=60):
        self.n_retrains = n_retrains
        self.run_num = run_num
        self.every_sec = every_sec
        self._last_stats_log = time.time()
        self._run_start = time.time()

        self.sims_done_total = 0
        self.total_plies = 0
        self.games_finished = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.training_queue = 0

        self.recent_games = []
        self.telemetry = {}

    def ingest_recents(self, recent):
        looper_id = recent['looper_id']
        meta = recent['meta']

        self.games_finished += 1
        self.total_plies += meta['plies']
        self.sims_done_total += meta['sims_done_total']

        if meta['result'] > 0:
            self.white_wins += 1
        elif meta['result'] < 0:
            self.black_wins += 1
        else:
            self.draws += 1
        
        self.recent_games.append(meta)

    def ingest_telemetry(self, telemetry):
        # store the latest from each only.
        looper_id = telemetry['looper_id']
        info = telemetry['telemetry']

        # update if existing, else create entry
        if looper_id in self.telemetry.keys():
            self.telemetry[looper_id].update(info)
        else:
            self.telemetry[looper_id] = info

        self.maybe_log_results()

    def get_agg_metrics(self):
        time_delta = time.time() - 120

        to_sum = [
            "mps", "lps", "n_active", "n_groups", "s_collected", "s_fast",
            "s_terminals", "s_cached", "s_fast_stops", "s_collect_stops",
            "s_priorless", "s_puct", "preds_per_second",
            # priors cache (per-worker) telemetry, aggregate across workers
            "cache_size", "cache_capacity", "cache_queries", "cache_hits"
        ]

        summed = defaultdict(float)
        sum_seen = set()

        to_avg = ["mbs", "fwd_target", "apl", "pred_wait", "avg_ply"]
        avged = defaultdict(list)
        avg_seen = set()

        for looper_id, info in self.telemetry.items():
            # if telemetry is timed out, skip it
            if info.get("ts", 0) < time_delta:
                continue

            for k in to_sum:
                summed[k] += info.get(k, 0)
                sum_seen.add(k)

            for k in to_avg:
                avged[k].append(info.get(k, 0))
                avg_seen.add(k)

        summed_out = {k: summed[k] for k in sorted(sum_seen)}
        avg_out = {k: np.mean(avged[k]) for k in sorted(avg_seen)}
        return summed_out, avg_out

    def maybe_log_results(self, window=1500, force=False):
        now = time.time()
        if not force and (now - self._last_stats_log < self.every_sec):
            return

        self._last_stats_log = now

        # pull stats
        summed, avged = self.get_agg_metrics()
        avg_moves = (self.total_plies / max(1, self.games_finished))
        gph =  3600 * self.games_finished / (now - self._run_start)
        
        print()
        print(f" Round {self.run_num} Logging ".center(72, "~"))
        
        mps, lps = summed.get("mps", 0), summed.get("lps", 0)
        print(f"[speed stats] mps={mps:.1f}  lps={lps:.1f}  gph={gph:.2f}")
        
        print(
            f"[game stats]  finished={self.games_finished}  "
            f"W/D/L={self.white_wins}/{self.draws}/{self.black_wins}  "
            f"avg_len={avg_moves:.1f} moves")
        print("-" * 72)

        recent = self.recent_games[-500:]
        if not recent:
            print("(no recent games to break down)")
            print("~" * 72)
            self.log_loop_stats(summed, avged)
            return

        # pretty printer
        print_recent_summary(recent, window=window)
        print(
            f"Length of training queue: {self.training_queue} ",
            f"Current retrain number: {self.n_retrains}\n"
        )
        # chain log_loop_stats here as well
        self.log_loop_stats(summed, avged)
        return

    def log_loop_stats(self, summed, avged):
        n_groups = summed.get("n_groups", 0)
        if n_groups == 0:
            return
        
        s_collected     = summed.get("s_collected", 0)
        s_terminals     = summed.get("s_terminals", 0)
        s_cached        = summed.get("s_cached", 0)
        s_fast_stops    = summed.get("s_fast_stops", 0)
        s_collect_stops = summed.get("s_collect_stops", 0)
        s_priorless     = summed.get("s_priorless", 0)
        s_puct          = summed.get("s_puct", 0)

        avg_new = s_collected / n_groups

        total_overall = s_collected + s_terminals + s_cached
        pct_cached_overall = 100.0 * s_cached / max(1, total_overall)
        pct_term_overall = 100.0 * s_terminals / max(1, total_overall)

        f_stops_pct = 100.0 * s_fast_stops / max(1, n_groups)
        collect_stops_pct = 100.0 * s_collect_stops / max(1, n_groups)

        mbs = avged['mbs']
        print("-"*72)
        left1 = f"[loop stats] groups={n_groups:.0f}  mbs={mbs:.0f}"
        right1 = f"new: collected={s_collected:.0f} avg={avg_new:.2f}"

        left2 = f"[stop stats] fastpath_breaks={s_fast_stops:.0f} ({f_stops_pct:.2f}%)"
        right2 = f"collect_breaks={s_collect_stops:.0f} ({collect_stops_pct:.2f}%)"

        # preds / active / finished runtime pre-compute
        apl = avged['apl']
        fwd_target = avged['fwd_target']
        fill_pct = 100.0 * apl / max(1.0, fwd_target)

        pred_wait = avged['pred_wait']
        preds_per_sec = summed['preds_per_second']

        left3 = f"[pred stats] fill={apl:.1f}/{fwd_target} ({fill_pct:.1f}%)"
        right3 = f"wait={pred_wait:.03f}s preds/s={preds_per_sec:.1f}"

        with_priors = total_overall - s_priorless
        puct_avg = s_puct / with_priors if with_priors else 0.0
        priorless_pct = 100.0 * s_priorless / max(1, total_overall)
        left4 = f"[leaf stats] priorless={s_priorless:.0f} ({priorless_pct:.2f}%)"
        right4 = f"puct={s_puct:.0f}  puct/leaf={puct_avg:.1f}"

        left5 = f"[cache hits] cached={s_cached:.0f} ({pct_cached_overall:.3f}%)"
        right5 = f"terminals={s_terminals:.0f} ({pct_term_overall:.3f}%)"

        sims = self.sims_done_total
        moves = self.total_plies
        sims_per_move = sims / moves if moves > 0 else 0.0

        n_active = summed['n_active']
        avg_ply = avged['avg_ply']
        left6 = f"[game stats] n={n_active:.0f} avg ply={avg_ply:.2f}"
        right6 = f"sims per move={sims_per_move:.2f}"

        col_width = 40
        print(f"{left1:<{col_width}} | {right1}")
        print(f"{left2:<{col_width}} | {right2}")
        print(f"{left3:<{col_width}} | {right3}")
        print(f"{left4:<{col_width}} | {right4}")
        print(f"{left5:<{col_width}} | {right5}")
        print(f"{left6:<{col_width}} | {right6}")

        # show game duration if its available
        last50 = self.recent_games[-50:]
        durations = [g.get("duration", 0.0) for g in last50]
        avg_runtime = None
        if sum(durations) > 0:
            avg_runtime = format_time(np.mean(durations))
            if avg_runtime:
                print(f"[game stats] last 50 runtime: {avg_runtime}")
        print("-"*72)