import json, pathlib, sys
import chess, chess.svg
from IPython.display import SVG, display, clear_output

class GameViewer:
    def __init__(self, log_path):
        self.path = pathlib.Path(log_path)
        with open(self.path, "r", encoding="utf-8") as f:
            self.log = json.load(f)
                    
        self.start_fen = self.log["start_fen"]
        self.moves_uci = self.log["moves_played"]
        self.tree_data = self.log.get("tree_search_data", {})
        self.result = self.log.get("result", None)
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
