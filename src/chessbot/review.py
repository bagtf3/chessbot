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
        chosen = self.moves_uci[self.ply]
        chosen_san = self.board.san(chess.Move.from_uci(chosen))
        print(f"Ply {self.ply+1}: {self.who_moved()} about to play {chosen_san}")


        node = self.tree_data.get(chosen) or {}
        cands = node.get("candidate_moves") or []
        sims = node.get("sims", 0)
        t = node.get("time", 0.0)
        avg_d = node.get("avg_depth", 0.0)
        max_d = node.get("max_depth", 0)
        print(f"  sims={sims} time={t:.2f}s avg_depth={avg_d:.2f} max_depth={max_d}")
        if not cands:
            print("  (no candidate_moves in log)")
            return
        cands = sorted(cands, key=lambda x: x.get("visits", 0), reverse=True)
        for c in cands[:top_n]:
            try:
                san = self.board.san(chess.Move.from_uci(c['uci']))
            except:
                san = "?"
            print(f"   {san:<6} visits={c.get('visits',0):<5} "
                  f"P={c.get('P',0):.3f} Q={c.get('Q',0):+.3f} "
                  f"U={c.get('U',0):+.3f}")

    def replay(self):
        print(f"Replaying {self.log['scenario']}")
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


#po = [g for g in games if g['stockfish_color'] is None]
#GameViewer(po[4]['json_file']).replay()


