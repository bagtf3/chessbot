import json, pathlib
import chess, chess.svg
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
    
    def _sf_row_for_ply(self, ply):
        # uses the alignment map built in _align_sf_rows_to_json()
        if not self._sf_by_ply or self.sf_rows is None:
            return None
        ridx = self._sf_by_ply.get(ply)
        if ridx is None:
            return None
        try:
            return self.sf_rows.iloc[ridx]  # or .loc[ridx] if you prefer
        except Exception:
            return None


    # add inside GameViewer
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
    
        def _print_row(c, mark=False, show_rank=False, rank_val=None):
            try:
                san = self.board.san(chess.Move.from_uci(c.get("uci", "")))
            except Exception:
                san = "?"
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
            _print_row(c, mark=is_sf_turn and (c.get("uci") == chosen))
            shown_ucis.add(c.get("uci"))
    
        # if SF's move exists but wasn't in top-N, show ellipsis + row WITH rank
        if is_sf_turn and sf_idx is not None:
            if cands_sorted[sf_idx].get("uci") not in shown_ucis:
                print("   ...")
                _print_row(
                    cands_sorted[sf_idx],
                    mark=True,
                    show_rank=True,
                    rank_val=sf_idx + 1,
                )
        
        vwq = node.get("visit_weighted_Q")
        if vwq is not None:
            print(f"\nvisit-weighted Q={vwq}")
        
        # SF overlay (optional)
        r = self._sf_row_for_ply(self.ply)
        if r is not None:
            stm_white = (self.board.turn == chess.WHITE)
        
            best_uci   = str(r.get("best_move", "") or "")
            played_uci = str(r.get("played_move", "") or "")
        
            best_cp_raw   = r.get("best_cp", None)
            played_cp_raw = r.get("played_cp", None)
        
            def _pov(cp):
                if cp is None or (isinstance(cp, float) and np.isnan(cp)):
                    return None
                return int(cp if stm_white else -cp)
        
            best_cp_pov   = _pov(best_cp_raw)
            played_cp_pov = _pov(played_cp_raw)
        
            loss = r.get("clipped_loss", r.get("loss", None))
            if loss is None and best_cp_pov is not None and played_cp_pov is not None:
                loss = max(0, best_cp_pov - played_cp_pov)
        
            # recompute match flag from UCIs to avoid DF drift
            matched = (best_uci == played_uci) if best_uci and played_uci else False
        
            def _to_san(uci):
                try:
                    return self.board.san(chess.Move.from_uci(uci))
                except Exception:
                    return "?"
        
            best_san   = _to_san(best_uci) if best_uci else "?"
            played_san = _to_san(played_uci) if played_uci else "?"
        
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
        while True:
            self.show_board(flipped=flipped)
            self.show_moves()
            cmd = input("[Enter]=fwd, b=back, q=quit > ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd in ("b", "back"):
                self.prev()
            else:
                self.next()
