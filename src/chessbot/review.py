import json, pathlib
import chess, chess.svg
import chess.engine
from chessbot import SF_LOC
import numpy as np
from IPython.display import SVG, display, clear_output


class GameViewer:
    # in GameViewer.__init__
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
                    # deterministic order: prefer timestamp, else move_num if present
                    order = [c for c in ["ts", "move_num"] if c in g.columns]
                    if order:
                        g = g.sort_values(order, kind="stable")
                    g = g.reset_index(drop=True)
                    self.sf_rows = g
                    self._align_sf_rows_to_json()
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
    
    def _align_sf_rows_to_json(self):
        self._sf_by_ply = {}
        if self.sf_rows is None:
            return
    
        seq = list(self.moves_uci or [])
        j = 0  # cursor in JSON move list
    
        for ridx, r in self.sf_rows.iterrows():
            uci = str(r.get("played_move", "") or "")
            if not uci:
                continue
    
            found = -1
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
        sf_color = self.log.get("stockfish_color", None)
        if sf_color is None:
            return f"{mover} (bot)"
        
        if (self.board.turn and sf_color) or \
           (not self.board.turn and not sf_color):
            return f"{mover} (stockfish)"
        return f"{mover} (bot)"

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
        # new
        uniq = node.get("unique_sims", None)
        line = (f"  sims={sims}  time={t:.2f}s  avg_depth={avg_d:.2f}  max_depth={max_d} "
                f"children visited={cv}/{tc}")
        if uniq is not None and sims:
            frac = uniq / max(1, sims)
            line += f"  unique={uniq} ({frac:.0%})"
        print(line)
    
        cands_sorted = sorted(cands, key=lambda x: x.get("visits", 0), reverse=True)
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
                f"P={c.get('P',0):.3f} U={c.get('U',0):+.3f} Q={c.get('Q',0):+.3f}"
                f"{marker}{rank_str}"
            )
    
        shown_ucis = set()
        # top-N: never show ranks; just mark if SF move happens to be in top-N
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
            if best_san != "?":
                parts.append(f"SF best={best_san}")
            if played_san != "?":
                parts.append(f"played={played_san} ({'✓' if matched else '×'})")
            if (best_cp_pov is not None) and (played_cp_pov is not None):
                parts.append(f"cp(best/played)={best_cp_pov}/{played_cp_pov}")
        
            if parts:
                print("SF:", "  ".join(parts))
        print("=" * 60)

    def replay(self):
        print(f"Replaying {self.log['scenario']}. Result {self.log['result']}")
        # flip the board if sf plays white
        sf_color = self.log.get("stockfish_color", None)
        flipped = sf_color if sf_color else False
        print("Controls: Enter/Space=forward, b=back, q=quit")
        shown = False
        while True:
            if not shown:
                self.show_board(flipped=flipped)
                self.show_moves()
                shown = True
            cmd = input("[Enter]=fwd, b=back, q=quit, sf=stockfish eval > ")
            cmd = cmd.strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
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
