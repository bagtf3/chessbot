import os, json, gzip, pathlib, time, random
import pickle

# the retained list would otherwise grow for the life of the process; the
# counters carry the whole-run view, this is only for the last-50 line
RECENT_GAMES_CAP = 20000

# per-window counts, as opposed to the rates and gauges alongside them in the
# telemetry dict. Only these get scaled onto the print's span.
COUNTER_KEYS = (
    "n_groups", "s_collected", "s_fast", "s_terminals", "s_cached",
    "s_fast_stops", "s_collect_stops", "s_blocked", "s_puct", "s_must_visit",
    "s_skipped", "s_pruned", "s_penalty", "s_depth",
)
from collections import defaultdict, deque

import io

import chess, chess.svg
from IPython.display import SVG, display, clear_output
import chess.engine
import cairosvg
from PIL import Image

import pandas as pd
import numpy as np

from chessbot import SF_LOC
from chessbot.engines import lc0_analyze
from chessbot.game_utils import reconcile_game_boards
from chessbot.utils import (
    print_summary_from_stats, accumulate_game_stats,
    format_time, compact_count,
)
from chessbot.utils import (
    score_cp_stm_pov, score_cp_white_pov, score_to_value_stm_pov,
    score_to_value_stm_pov_tanh, rnd,
    calc_entropy, kl_divergence, load_json, save_pickle_atomic,
)


ANALYZE_PKL = "analyze_results_combined.pkl"
VISUALS_DIR = r"C:\Users\Bryan\Data\chessbot_data\visuals"

# default analysis params
DEPTH = 12
EQUIV_RANGE = 30


class GameViewer:
    def __init__(self, log_path, sf_df=None):
        self.path = pathlib.Path(log_path)
        if not self.path.exists():
            raise FileNotFoundError(f"log file not found: {self.path}")

        suffix = self.path.suffix.lower()
        name = self.path.name.lower()
        if suffix == ".json":
            with open(self.path, "r", encoding="utf-8") as f:
                self.log = json.load(f)
        elif name.endswith(".pkl.gz"):
            with gzip.open(self.path, "rb") as f:
                self.log = pickle.load(f)
        elif suffix in (".pkl", ".pickle"):
            with open(self.path, "rb") as f:
                self.log = pickle.load(f)
        else:
            raise Exception("log_path must be json or pickle")

        self.init_from_log(self.log, sf_df)

    @classmethod
    def from_record(cls, row, sf_df=None):
        """Viewer over an already-loaded log dict, e.g. one brp_reviewables
        row. No file behind it, so self.path is None."""
        v = cls.__new__(cls)
        v.path = None
        v.log = row
        v.init_from_log(row, sf_df)
        return v

    @classmethod
    def from_batch(cls, jsonl_path, sf_df=None):
        """Every row of a brp_reviewables jsonl, steppable one probe at a
        time. See BrpReviewBatch."""
        return BrpReviewBatch(jsonl_path, sf_df=sf_df)

    def init_from_log(self, log, sf_df=None):
        self.start_fen = log.get("start_fen")
        self.moves_uci = log.get("moves_played", [])
        self.tree_data = {int(k): v for k, v in log.get("tree_search_data", {}).items()}
        self.result = log.get("result")
        self.game_id = log.get("game_id")

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

    def warmed_boards(self):
        return reconcile_game_boards(
            self.start_fen, self.log.get('history_uci'), self.moves_uci,
        )

    def reset(self):
        self.board_ch, self.board = self.warmed_boards()
        self.ply = 0  # 0 = before first move

    def goto(self, ply):
        ply = max(0, min(ply, len(self.moves_uci)))
        self.board_ch, self.board = self.warmed_boards()
        for u in self.moves_uci[:ply]:
            self.board.push_uci(u)
            self.board_ch.push(chess.Move.from_uci(u))
        self.ply = ply
        return self
    
    def turn(self):
        return self.board.white_to_move()
    
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
            sf_board = self.board_ch.copy()
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

    def pv_to_san(self, pv, max_moves=5):
        """Convert a list of chess.Move to a readable SAN string (up to max_moves plies)."""
        board = chess.Board(self.board.fen())
        parts = []
        for i, move in enumerate(pv[:max_moves]):
            if board.turn == chess.WHITE:
                parts.append(f"{board.fullmove_number}.")
            elif i == 0:
                parts.append(f"{board.fullmove_number}...")
            parts.append(board.san(move))
            board.push(move)
        return " ".join(parts)
    
    def show_sf_overlay(self, token="sf"):
        """
        Modes:
          sf / sf<N>          - top-3 at depth N (default 16)
          sf <move> <depth>   - analyze specific move (UCI or SAN) at given depth
        """
        parts = token.split()
        depth = 16
        move_str = None

        # extract depth from "sfNN" prefix or from a numeric part
        try:
            if len(parts[0]) > 2:
                depth = int(parts[0][2:])
        except (ValueError, IndexError):
            pass
        for part in parts[1:]:
            try:
                depth = int(part)
            except ValueError:
                move_str = part

        if move_str is not None:
            # resolve move_str as UCI or SAN against current legal moves
            sf_board = self.board_ch.copy()
            root_move = None
            try:
                m = chess.Move.from_uci(move_str)
                if m in sf_board.legal_moves:
                    root_move = m
            except ValueError:
                pass
            if root_move is None:
                try:
                    root_move = sf_board.parse_san(move_str)
                except ValueError:
                    pass
            if root_move is None:
                print(f"Not a valid move in this position: {move_str}")
                return

            san = sf_board.san(root_move)
            sign = 1 if self.turn() else -1
            print(f"Running Stockfish (depth={depth}, time cap=10s, move={san})...")
            limit = chess.engine.Limit(depth=depth, time=10.0)
            with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
                info = eng.analyse(sf_board, limit=limit,
                                   root_moves=[root_move],
                                   info=chess.engine.INFO_ALL)
            score_obj = info.get("score")
            cp_stm = None
            if score_obj is not None:
                cp_w = score_obj.white().score(mate_score=1500)
                cp_stm = None if cp_w is None else int(cp_w * sign)
            cp_str = "mate" if cp_stm is None else str(cp_stm)
            pv = info.get("pv", [])
            print(f"  {san:<6}  cp={cp_str}")
            if pv:
                print(f"  {self.pv_to_san(pv)}")
            return

        print(f"Running Stockfish (depth={depth}, time cap=10s, multipv=3)...")
        topk = self.run_stockfish_topk(depth=depth, k=3)
        if not topk:
            print("No SF info (maybe engine failed).")
            return

        sign = 1 if self.turn() else -1
        print("SF top moves (STM-POV cp):")
        top_ucis = set()
        for i, (uci, san, cp, pv) in enumerate(topk, start=1):
            cp_stm = None if cp is None else int(cp * sign)
            cp_str = "mate" if cp_stm is None else str(cp_stm)
            print(f"  #{i}. {san:<6}  cp={cp_str}")
            if pv:
                print(f"      {self.pv_to_san(pv)}")
            if uci:
                top_ucis.add(uci)

        if self.ply < len(self.moves_uci):
            upcoming_uci = self.moves_uci[self.ply]
            upcoming_san = self.board.san(upcoming_uci)
            if upcoming_uci not in top_ucis:
                print(f"\nPlayed move {upcoming_san} not in SF top-3; computing cp...")
                limit = chess.engine.Limit(depth=depth, time=10.0)
                with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
                    sf_board = self.board_ch.copy()
                    info = eng.analyse(
                        sf_board, limit=limit,
                        root_moves=[chess.Move.from_uci(upcoming_uci)],
                        info=chess.engine.INFO_ALL
                    )
                score_obj = info.get("score")
                if score_obj is None:
                    print("  played move eval not available")
                else:
                    cp = score_obj.white().score(mate_score=1500)
                    print(f"  {upcoming_san:<6}  cp={int(cp * sign)}")
                pv = info.get("pv", [])
                if pv:
                    print(f"      {self.pv_to_san(pv)}")
            else:
                print(f"\nPlayed move {upcoming_san} is in SF top-3.")

    def run_lc0_topk(self, nodes=4000, k=5, engine_cfg=None):
        """Return top-k rows from lc0 (sorted by N) for the current position."""
        sf_board = self.board_ch.copy()
        rows = lc0_analyze(sf_board, nodes=nodes, engine_cfg=engine_cfg)
        return rows[:k]

    def show_lc0_overlay(self, token="lc0"):
        """Parse 'lc0' or 'lc0 N' and print lc0 top-5 vs xc0 comparison."""
        nodes = 4000
        parts = token.split()
        if len(parts) >= 2:
            try:
                nodes = int(parts[1])
            except ValueError:
                pass

        print(f"Running lc0 ({nodes} nodes)...")
        sf_board = self.board_ch.copy()
        all_rows = lc0_analyze(sf_board, nodes=nodes)
        if not all_rows:
            print("No lc0 data returned.")
            return

        # lc0 reports castling in Chess960 UCI (e.g. e8h8 = king captures rook).
        # Normalize to standard king-destination UCI so san() and xc0 lookup work.
        castle_norm = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}
        for r in all_rows:
            r["move"] = castle_norm.get(r["move"], r["move"])

        # xc0 candidate lookup: uci -> candidate dict
        node, _, _ = self.node_who_chosen()
        cands = (node.get("candidate_moves") or []) if node else []
        xc0_map = {c["uci"]: c for c in cands}

        white_to_move = self.board.white_to_move()
        sign = 1 if white_to_move else -1

        # stats over all rows using the union of lc0 and xc0 moves
        all_ucis = list({r["move"] for r in all_rows} | set(xc0_map))
        lc0_visit_map = {r["move"]: r.get("N", 0) for r in all_rows}
        lc0_prior_map = {r["move"]: r.get("P_pct", 0.0) / 100.0 for r in all_rows}
        xc0_visit_map = {c["uci"]: c.get("visits", 0) for c in cands}
        xc0_prior_map = {c["uci"]: c.get("P", 0.0) for c in cands}

        lc0_vis_vec  = np.array([lc0_visit_map.get(u, 0)   for u in all_ucis],
                                dtype=np.float64)
        lc0_pri_vec  = np.array([lc0_prior_map.get(u, 0.0) for u in all_ucis],
                                dtype=np.float64)
        xc0_vis_vec  = np.array([xc0_visit_map.get(u, 0)   for u in all_ucis],
                                dtype=np.float64)
        xc0_pri_vec  = np.array([xc0_prior_map.get(u, 0.0) for u in all_ucis],
                                dtype=np.float64)

        lc0_ent_vis, lc0_pvis = self.compute_norm_entropy(lc0_vis_vec)
        lc0_ent_pri, lc0_ppri = self.compute_norm_entropy(lc0_pri_vec)
        xc0_ent_vis, xc0_pvis = self.compute_norm_entropy(xc0_vis_vec)
        xc0_ent_pri, xc0_ppri = self.compute_norm_entropy(xc0_pri_vec)
        lc0_kl_vp    = kl_divergence(lc0_vis_vec, lc0_pri_vec)
        kl_xpri_lpri = kl_divergence(xc0_pri_vec, lc0_pri_vec)
        kl_xvis_lvis = kl_divergence(xc0_vis_vec, lc0_vis_vec)

        hdr = (
            f"  {'SAN':<7}  {'N':>6}  {'P(lc0)':>7}  {'P(xc0)':>7}"
            f"  |  {'Q(lc0)':>7}  {'Q(xc0)':>7}  |  {'D':>5}"
        )
        sep = "  " + "-" * (len(hdr) - 2)
        to_move = "White" if white_to_move else "Black"
        print(f"\n  lc0 top moves ({nodes} nodes, {to_move} to move, Q=STM-pov):")
        print(hdr)
        print(sep)

        top_ucis = set()
        for r in all_rows[:5]:
            mv  = r["move"]
            san = self.board.san(mv)
            n     = r.get("N", 0)
            p_lc0 = r.get("P_pct", 0.0) / 100.0
            q_lc0 = r.get("Q", 0.0)
            d     = r.get("D", 0.0)

            xc0   = xc0_map.get(mv)
            p_xc0 = xc0["P"] if xc0 else float("nan")
            q_xc0 = xc0["Q"] * sign if xc0 else float("nan")

            print(
                f"  {san:<7}  {n:>6}  {p_lc0:>7.3f}  {p_xc0:>7.3f}"
                f"  |  {q_lc0:>+7.3f}  {q_xc0:>+7.3f}  |  {d:>5.3f}"
            )
            top_ucis.add(mv)

        print(sep)
        print(
            f"  {'ent (norm):':<16}  {'vis':>5}  {'pri':>5}  KL(vis||pri)"
        )
        print(
            f"  {'lc0':<16}  {lc0_ent_vis:>5.3f}  {lc0_ent_pri:>5.3f}"
            f"  {lc0_kl_vp:.3f}"
        )
        print(
            f"  {'cross KL:':<16}"
            f"  xc0_p||lc0_p={kl_xpri_lpri:.3f}"
            f"   xc0_v||lc0_v={kl_xvis_lvis:.3f}"
        )

        if self.ply < len(self.moves_uci):
            upcoming_uci = self.moves_uci[self.ply]
            upcoming_san = self.board.san(upcoming_uci)
            if upcoming_uci not in top_ucis:
                match = next((r for r in all_rows if r["move"] == upcoming_uci), None)
                if match:
                    q_lc0 = match.get("Q", 0.0)
                    n     = match.get("N", 0)
                    print(
                        f"\nPlayed {upcoming_san} not in top-5:"
                        f"  N={n}  Q(lc0)={q_lc0:+.3f}"
                    )
                else:
                    print(f"\nPlayed {upcoming_san} not visited by lc0.")
            else:
                print(f"\nPlayed {upcoming_san} is in lc0 top-5.")

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
            mv = self.moves_uci[self.ply]
            self.board.push_uci(mv)
            self.board_ch.push(chess.Move.from_uci(mv))
            self.ply += 1

    def prev(self):
        if self.ply > 0:
            self.ply -= 1
            self.board.unmake()
            self.board_ch.pop()
    
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

    def to_gif(self, refresh=0.3, save_to=None, flipped=False):
        """Render the full game as an animated gif, one frame per ply."""
        if save_to is None:
            game_id = self.game_id or self.path.stem
            save_to = pathlib.Path(VISUALS_DIR) / f"{game_id}.gif"
        else:
            save_to = pathlib.Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)

        board = chess.Board(self.start_fen) if self.start_fen else chess.Board()
        frames = []
        lastmove = None
        for i in range(len(self.moves_uci) + 1):
            svg_data = chess.svg.board(
                board=board, flipped=flipped, lastmove=lastmove,
            )
            png_bytes = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
            frames.append(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
            if i < len(self.moves_uci):
                move = chess.Move.from_uci(self.moves_uci[i])
                board.push(move)
                lastmove = move

        frames[0].save(
            save_to, format="GIF", save_all=True, append_images=frames[1:],
            duration=int(refresh * 1000), loop=0,
        )
        return save_to

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

    def compute_rsc_top5(self, children):
        if not children:
            return {}

        items = list(children)
        items.sort(key=lambda x: x.get("visits", 0), reverse=True)

        k = min(5, len(items))
        top = items[:k]

        if k == 1:
            uci = top[0].get("uci", "")
            return {uci: 1.0}

        flip = 1.0 if self.board.white_to_move() else -1.0

        def minmax(vals):
            lo = min(vals)
            hi = max(vals)
            if hi <= lo:
                return [0.5 for _ in vals]
            return [(v - lo) / (hi - lo) for v in vals]

        def to_prob(vals01):
            s = sum(vals01)
            if s <= 0.0:
                kk = len(vals01)
                return [1.0 / kk for _ in vals01]
            return [v / s for v in vals01]

        visits = [c.get("visits", 0) for c in top]
        vs = [c.get("visit_share", 0.0) for c in top]
        q = [flip * c.get("Q", 0.0) for c in top]
        qe = [flip * c.get("Qema", 0.0) for c in top]
        ds = [c.get("Qdelta_sign", 0.0) for c in top]

        p_vis = to_prob(minmax(visits))
        p_vs = to_prob(minmax(vs))
        p_q = to_prob(minmax(q))
        p_qe = to_prob(minmax(qe))
        p_ds = to_prob(minmax(ds))

        w = 0.2
        out = {}
        for i in range(k):
            score = (
                w * p_vis[i]
                + w * p_vs[i]
                + w * p_q[i]
                + w * p_qe[i]
                + w * p_ds[i]
            )
            out[top[i].get("uci", "")] = score

        return out

    def print_header(self, show_rank=False, show_rsc=False):
        cols = [
            ("SAN", 7), ("N", 6), ("vs", 7),
            ("|", 1),
            ("Q", 7), ("Qe", 7), ("dS", 7), ("D", 6),
            ("|", 1),
            ("P", 7), ("U", 7), ("PUCT", 8),
            ("|", 1),
            ("flags", 9),
        ]

        if show_rsc:
            cols.append(("rsc", 7))

        if show_rank:
            cols.append(("rank", 7))

        hdr_body = " ".join([f"{n:^{w}}" for n, w in cols])
        sep_body = " ".join(["-" * w if w > 1 else "|" for _, w in cols])

        indent = "  "
        hdr = indent + hdr_body
        sep = indent + sep_body

        full_width = len(hdr)

        print(indent + "-" * (full_width - len(indent)))
        print(hdr)
        print(sep)

    def print_row(
        self,
        c,
        mark=False,
        show_rank=False,
        rank_val=None,
        show_rsc=False,
        rsc_val=None,
        extra_flags=None,
    ):
        uci = c.get("uci", "")
        san = self.board.san(uci) if uci else "?"

        visits = c.get("visits", 0)
        visit_share = c.get("visit_share", 0.0)

        sign = 1 if self.board.white_to_move() else -1
        q = c.get("Q", 0.0) * sign
        qema = c.get("Qema", 0.0) * sign
        ds = c.get("Qdelta_sign", 0.0)

        p = c.get("P", 0.0)
        u = c.get("U", 0.0) * (1.0 + 0.5 * np.clip(ds, -0.5, 0.5))
        puct = q + u

        flags = []
        if mark:
            flags.append("<-SF")
        if extra_flags:
            flags += list(extra_flags)
        if c.get("is_terminal", False):
            flags.append("T")

        flags_txt = ", ".join([f for f in flags if f])

        q_draw = c.get("Q_draw", float("nan"))
        parts = [
            f"{san:<7}", f"{visits:^6}", f"{visit_share:^7.3f}",
            "|",
            f"{q:^+7.3f}", f"{qema:^+7.3f}", f"{ds:^+7.3f}", f"{q_draw:^6.3f}",
            "|",
            f"{p:^7.3f}", f"{u:^+7.3f}", f"{puct:^+8.3f}",
            "|",
            f"{flags_txt:^9}",
        ]

        if show_rsc:
            rv = "" if rsc_val is None else f"{rsc_val:0.3f}"
            parts.append(f"{rv:^7}")

        if show_rank:
            r = rank_val if rank_val is not None else ""
            parts.append(f"{r!s:<7}")

        print("  " + " ".join(parts))

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
        stop = node.get("stop_reason") or "?"

        line = (
            f"  sims={sims}  stop={stop}  time={t:.2f}s  avg_depth={avg_d:.2f}  "
            f"max_depth={max_d} children visited={cv}/{tc}"
        )
        print(line)

        visits_list = [c.get("visits", 0) for c in cands]
        priors_list = [c.get("P", 0.0) for c in cands]

        norm_vis, p_vis = self.compute_norm_entropy(visits_list)
        norm_pri, p_pri = self.compute_norm_entropy(priors_list)
        kl = kl_divergence(visits_list, priors_list)

        r_sf = self.sf_row_for_ply(self.ply)
        nn_wdl    = node.get('nn_wdl')
        nn_value  = node.get('nn_value')
        best_wdl  = node.get('best_wdl')
        sf_wdl_tr = node.get('sf_wdl')
        is_white  = self.turn()
        sign      = 1 if is_white else -1

        is_sf_turn = ("stockfish" in str(who).lower())

        nn_wdl_stm = None
        if nn_wdl is not None:
            nw = np.array(nn_wdl if is_white else (nn_wdl[2], nn_wdl[1], nn_wdl[0]), dtype=np.float64)
            nn_wdl_stm = nw / nw.sum() if nw.sum() > 0 else nw

        # reconstruct Y matching rescore.py formulas exactly:
        #   SF move:     0.5*z + 0.5*sf_wdl_tr  (fallback: pure z)
        #   Xerces move: 0.5*z + 0.5*best_wdl   (fallback: pure z)
        Y = None
        if nn_value is not None:
            z_stm = self.result * sign
            if z_stm > 0:
                z_wdl = np.array([1.0, 0.0, 0.0])
            elif z_stm < 0:
                z_wdl = np.array([0.0, 0.0, 1.0])
            else:
                z_wdl = np.array([0.0, 1.0, 0.0])

            if is_sf_turn:
                if sf_wdl_tr is not None and len(sf_wdl_tr) == 3:
                    Y = 0.5 * z_wdl + 0.5 * np.array(sf_wdl_tr, dtype=np.float64)
                else:
                    Y = z_wdl.copy()
            else:
                if best_wdl is not None:
                    bw = np.array(best_wdl, dtype=np.float64)
                    if not is_white:
                        bw = bw[[2, 1, 0]]
                    Y = 0.5 * z_wdl + 0.5 * bw
                else:
                    Y = z_wdl.copy()

        # source KL/CE/MSE from sf_df if available, else compute and mark with *
        if r_sf is not None and 'rescore_kl' in r_sf and not np.isnan(r_sf['rescore_kl']):
            kl_str = f"{r_sf['rescore_kl']:.3f} ({kl:.3f}*)"
        else:
            kl_str = f"{kl:.3f}*"

        if r_sf is not None and 'rescore_ce' in r_sf and not np.isnan(r_sf['rescore_ce']):
            ce_str = f"{r_sf['rescore_ce']:.3f}"
        elif Y is not None and nn_wdl_stm is not None:
            eps = 1e-7
            p_ce = np.clip(nn_wdl_stm, eps, 1 - eps)
            p_ce = p_ce / p_ce.sum()
            ce_str = f"{-float(np.dot(Y, np.log(p_ce))):.3f}*"
        else:
            ce_str = "n/a"

        if r_sf is not None and 'rescore_mse' in r_sf and not np.isnan(r_sf['rescore_mse']):
            mse_str = f"{r_sf['rescore_mse']:.3f}"
        elif Y is not None and nn_value is not None:
            nn_value_stm = np.clip(nn_value * sign, -1.0, 1.0)
            mse_str = f"{(nn_value_stm - (Y[0] - Y[2])) ** 2:.3f}*"
        else:
            mse_str = "n/a"

        print(
            f"  entropy (norm'd): visits = {norm_vis:.3f} "
            f" priors = {norm_pri:.3f}  KL = {kl_str}"
            f"  CE = {ce_str}  MSE = {mse_str}"
        )
        if nn_wdl_stm is not None or Y is not None:
            nn_wdl_str = (f"W = {nn_wdl_stm[0]:.3f}  D = {nn_wdl_stm[1]:.3f}  L = {nn_wdl_stm[2]:.3f}"
                          if nn_wdl_stm is not None else "n/a")
            tgt_wdl_str = (f"W = {Y[0]:.3f}  D = {Y[1]:.3f}  L = {Y[2]:.3f}"
                           if Y is not None else "n/a")
            print(f"  nn WDL (STM): {nn_wdl_str}  |  tgt WDL: {tgt_wdl_str}")

        cands_sorted = sorted(
            cands, key=lambda x: x.get("visits", 0), reverse=True
        )

        sf_best_uci = None
        if r_sf is not None:
            sf_best_uci = str(r_sf.get("best_move", "") or "")

        raw_xc0 = node.get("xc0_move")
        if (not is_sf_turn) and (not raw_xc0):
            xc0_move = chosen
        else:
            xc0_move = raw_xc0

        sampled = bool((not is_sf_turn) and raw_xc0 and (raw_xc0 != chosen))

        xc0_idx = None
        if xc0_move:
            for i, c in enumerate(cands_sorted):
                if c.get("uci") == xc0_move:
                    xc0_idx = i
                    break

        sf_best_idx = None
        if sf_best_uci:
            for i, c in enumerate(cands_sorted):
                if c.get("uci") == sf_best_uci:
                    sf_best_idx = i
                    break

        chosen_idx = None
        for i, c in enumerate(cands_sorted):
            if c.get("uci") == chosen:
                chosen_idx = i
                break

        need_rank = False
        if is_sf_turn and chosen_idx is not None:
            need_rank = (chosen_idx >= top_n)

        if (not is_sf_turn) and sf_best_idx is not None:
            need_rank = need_rank or (sf_best_idx >= top_n)

        if (not is_sf_turn) and xc0_idx is not None:
            need_rank = need_rank or (xc0_idx >= top_n)

        self.print_header(show_rank=need_rank, show_rsc=True)
        rsc_map = self.compute_rsc_top5(cands_sorted[:top_n])

        shown_ucis = set()

        for i, c in enumerate(cands_sorted[:top_n]):
            uci = c.get("uci")
            extra = []

            if not is_sf_turn:
                if xc0_move and (uci == xc0_move):
                    if sf_best_uci and (uci == sf_best_uci):
                        extra.append("<-Xc0, SF")
                    else:
                        extra.append("<-Xc0")

                if sampled and (uci == chosen) and (chosen != xc0_move):
                    if sf_best_uci and (chosen == sf_best_uci):
                        extra.append("<-SF,samp")
                    else:
                        extra.append("<-sampled")

            mark = False
            if is_sf_turn:
                mark = (uci == chosen)
            else:
                if sf_best_uci and (uci == sf_best_uci):
                    is_xc0_sf = bool(xc0_move and (uci == xc0_move))
                    is_samp_sf = bool(
                        sampled and (chosen == sf_best_uci) and (uci == chosen)
                        and (chosen != xc0_move)
                    )
                    if (not is_xc0_sf) and (not is_samp_sf):
                        mark = True

            self.print_row(
                c,
                mark=mark,
                show_rank=need_rank,
                rank_val=(i + 1) if need_rank else None,
                show_rsc=True,
                rsc_val=rsc_map.get(uci),
                extra_flags=extra,
            )
            shown_ucis.add(uci)

        if (not is_sf_turn) and xc0_move and (xc0_move not in shown_ucis):
            if xc0_idx is not None:
                print("   ...")
                extra = []
                if sf_best_uci and (xc0_move == sf_best_uci):
                    extra.append("<-Xc0, SF")
                else:
                    extra.append("<-Xc0")

                self.print_row(
                    cands_sorted[xc0_idx],
                    mark=False,
                    show_rank=True,
                    rank_val=xc0_idx + 1,
                    show_rsc=True,
                    rsc_val=None,
                    extra_flags=extra
                )
                shown_ucis.add(xc0_move)

        if (not is_sf_turn) and sf_best_uci and (sf_best_uci not in shown_ucis):
            if sf_best_idx is not None:
                print("   ...")
                extra = []
                mark = True

                if sampled and (sf_best_uci == chosen) and (chosen != xc0_move):
                    extra.append("<-SF,samp")
                    mark = False

                self.print_row(
                    cands_sorted[sf_best_idx],
                    mark=mark,
                    show_rank=True,
                    rank_val=sf_best_idx + 1,
                    show_rsc=True,
                    rsc_val=None,
                    extra_flags=extra
                )
        if is_sf_turn and chosen and (chosen not in shown_ucis):
            if chosen_idx is not None:
                print("   ...")
                self.print_row(
                    cands_sorted[chosen_idx],
                    mark=True,
                    show_rank=True,
                    rank_val=chosen_idx + 1,
                    show_rsc=True,
                    rsc_val=None,
                    extra_flags=None,
                )
                shown_ucis.add(chosen)

        wdl = node.get("best_wdl")
        this_q = (wdl[0] - wdl[2]) if wdl is not None else None
        if this_q is not None:
            print(f"\nBest Q={this_q:+.4f}")

        r = self.sf_row_for_ply(self.ply)
        if r is not None:
            stm_white = self.turn()

            best_uci = str(r.get("best_move", "") or "")
            played_uci = str(r.get("played_move", "") or "")

            best_cp_raw = r.get("best_cp", None)
            played_cp_raw = r.get("played_cp", None)

            def pov(cp):
                if cp is None or (isinstance(cp, float) and np.isnan(cp)):
                    return None
                return int(cp if stm_white else -cp)

            best_cp_pov = pov(best_cp_raw)
            played_cp_pov = pov(played_cp_raw)

            loss = r.get("clipped_loss", r.get("loss", None))
            if loss is None and best_cp_pov is not None and played_cp_pov is not None:
                loss = max(0, best_cp_pov - played_cp_pov)

            xc0_uci = node.get("xc0_move") if node else None
            cmp_uci = xc0_uci if (xc0_uci and xc0_uci != chosen) else chosen
            matched = ((best_uci == cmp_uci) if best_uci and cmp_uci else False)

            def to_san(uci):
                return self.board.san(uci)

            best_san = to_san(best_uci) if best_uci else "?"
            played_san = to_san(played_uci) if played_uci else "?"
            xc0_san = to_san(xc0_uci) if xc0_uci else "?"

            parts = []
            if loss is not None:
                parts.append(f"CPL={int(loss)}")

            if played_san != "?":
                if xc0_uci and xc0_san != "?":
                    parts.append(f"played={played_san}")
                    xc0_cp = f" ({played_cp_pov})" if played_cp_pov is not None else ""
                    parts.append(f"Xc0={xc0_san}{xc0_cp}")
                else:
                    played_cp = f" ({played_cp_pov})" if played_cp_pov is not None else ""
                    parts.append(f"played={played_san}{played_cp}")

            if best_san != "?":
                best_cp = f" ({best_cp_pov})" if best_cp_pov is not None else ""
                parts.append(f"SF best={best_san}{best_cp}")

            parts.append(f"played best: {'✓' if matched else '×'}")

            if parts:
                print("SF:", "  ".join(parts))

        print("=" * 60)

    def show_visits(self, uci_or_san):
        node, who, chosen = self.node_who_chosen()
        if node is None:
            print(f"No visit info available for {uci_or_san}")
            return

        cands = node.get("candidate_moves") or []
        scands = sorted(cands, key=lambda x: x["visits"], reverse=True)

        move = None
        rank = 0
        for c in scands:
            rank += 1
            if uci_or_san in [c["uci"], self.board.san(c["uci"])]:
                move = c
                break

        if move is None:
            print(f"No visit info available for {uci_or_san}")
            return

        is_sf_turn = ("stockfish" in str(who).lower())
        xc0_move = node.get("xc0_move") if node else None

        r = self.sf_row_for_ply(self.ply)
        sf_best_uci = None
        if r is not None:
            sf_best_uci = str(r.get("best_move", "") or "")

        uci = move.get("uci")

        sampled = bool(xc0_move and (xc0_move != chosen) and (not is_sf_turn))

        extra = []
        mark = False

        if is_sf_turn:
            if uci == chosen:
                mark = True
        else:
            if xc0_move and uci == xc0_move:
                if sf_best_uci and (uci == sf_best_uci):
                    extra.append("<-Xc0, SF")
                else:
                    extra.append("<-Xc0")

            if sampled and (uci == chosen) and (chosen != xc0_move):
                if sf_best_uci and (chosen == sf_best_uci):
                    extra.append("<-SF,samp")
                else:
                    extra.append("<-sampled")

            if sf_best_uci and (uci == sf_best_uci):
                mark = True

        print(f"  === Showing visit info for {uci_or_san} ===")
        self.print_header(show_rank=True, show_rsc=False)
        self.print_row(
            move,
            mark=mark,
            show_rank=True,
            rank_val=rank,
            show_rsc=False,
            extra_flags=extra,
        )
        print(f"   Rank: {rank}\tShare: {100 * move['visits'] / node['sims']:.3f}%")

    def seek_cpl(self, threshold):
        if self.sf_rows is None:
            print("No SF data loaded for this game.")
            return
        for ply in range(self.ply + 1, len(self.moves_uci)):
            r = self.sf_row_for_ply(ply)
            if r is None:
                continue
            loss = r.get("clipped_loss", r.get("loss", None))
            if loss is not None and loss >= threshold:
                self.goto(ply)
                print(f"Found CPL={int(loss)} at ply {ply + 1}")
                return
        print(f"No move with CPL >= {threshold} found after ply {self.ply + 1}.")

    def seek_kl(self, threshold):
        for ply in range(self.ply + 1, len(self.moves_uci)):
            node = self.tree_data.get(ply) or {}
            cands = node.get("candidate_moves") or []
            if not cands:
                continue
            visits_list = [c.get("visits", 0) for c in cands]
            priors_list = [c.get("P", 0.0) for c in cands]
            _, p_vis = self.compute_norm_entropy(visits_list)
            _, p_pri = self.compute_norm_entropy(priors_list)
            kl = kl_divergence(visits_list, priors_list)
            if kl >= threshold:
                self.goto(ply)
                print(f"Found KL={kl:.3f} at ply {ply + 1}")
                return
        print(f"No move with KL >= {threshold} found after ply {self.ply + 1}.")

    SAC_MATERIAL = 2
    SAC_Q = 0.3
    SAC_HOLD_PLIES = 5

    def find_sac_ply(self, start_ply=0):
        """First ply at or after start_ply that gives up >= SAC_MATERIAL pts
        of material and still shows Q > SAC_Q for the sacrificer through
        SAC_HOLD_PLIES plies after it (the opponent's three replies plus the
        sacrificer's two follow-ups) -- long enough that a quick intermezzo
        exchange that gives the material right back doesn't qualify. Returns
        the ply index, or None.

        Read-only: does not touch self.ply/self.board. Shared by seek_sac()
        (jumps the viewer there) and contains_sac() (just wants the bool).
        """
        values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9,
        }

        def material_white(board):
            total = 0
            for pt, val in values.items():
                total += val * len(board.pieces(pt, chess.WHITE))
                total -= val * len(board.pieces(pt, chess.BLACK))
            return total

        def sac_q(ply, sign):
            node = self.tree_data.get(ply) or self.tree_data.get(str(ply))
            wdl = node.get("best_wdl") if node else None
            if not wdl:
                return None
            return (wdl[0] - wdl[2]) * sign

        def nbrq_count(board, color):
            return sum(
                len(board.pieces(pt, color))
                for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
            )

        def stockfish_moved(is_white):
            if not self.log.get("vs_stockfish", False):
                return False
            sf_color = self.log.get("stockfish_color")
            if sf_color is None:
                return False
            return is_white == bool(sf_color)

        board, _ = self.warmed_boards()
        for u in self.moves_uci[:start_ply]:
            board.push(chess.Move.from_uci(u))

        n = len(self.moves_uci)
        for i in range(start_ply, n - self.SAC_HOLD_PLIES):
            sac_white = board.turn == chess.WHITE
            sign = 1 if sac_white else -1
            sac_color = chess.WHITE if sac_white else chess.BLACK

            if stockfish_moved(sac_white):
                board.push(chess.Move.from_uci(self.moves_uci[i]))
                continue

            # material must be roughly even (or better) going INTO this move --
            # otherwise i is just some later move played while already down,
            # not the move that actually creates the deficit
            pre_mat = material_white(board) * sign
            pre_nbrq = nbrq_count(board, sac_color)
            board.push(chess.Move.from_uci(self.moves_uci[i]))

            if pre_mat <= -self.SAC_MATERIAL:
                continue

            # material can't drop on your own move -- only the opponent's
            # reply actually removes a piece from the board, whether i put a
            # piece en prise or just left an existing one hanging. So the
            # deficit check starts at i+1 (the opponent's capture), not
            # right after i itself.
            ok = True
            probe = board.copy()
            for k in range(i + 1, i + 1 + self.SAC_HOLD_PLIES):
                probe.push(chess.Move.from_uci(self.moves_uci[k]))
                mat = material_white(probe) * sign
                q = sac_q(k + 1, sign)
                if mat > -self.SAC_MATERIAL or q is None or q <= self.SAC_Q:
                    ok = False
                    break
            # net material was lost, but only counts as a sac if a piece
            # (not just pawns) actually left the board for it
            if ok and nbrq_count(probe, sac_color) < pre_nbrq:
                return i

        return None

    def seek_sac(self):
        ply = self.find_sac_ply(self.ply)
        if ply is None:
            print(f"No qualifying sacrifices found after ply {self.ply + 1}.")
            return
        self.goto(ply)

    def contains_sac(self):
        """True if this game has any qualifying sacrifice, anywhere. For
        scanning a bundle of games: gv.contains_sac() to find showcase
        material for review, without moving the viewer."""
        return self.find_sac_ply(0) is not None

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
        print("  lc0                  lc0 overlay (default 4000 nodes)")
        print("  lc0 <N>              lc0 eval with N nodes, e.g. lc0 8000")
        print("  visits <move>        show MCTS visits for specific move")
        print("  visits <N>           show visit info for N top moves")
        print("  pv                   show principal variation (min_vis=1)")
        print("  pv<N>                show principal variation filtered by")
        print("                       minimum visits, e.g. pv8")
        print("  cpl > <N>            seek next move with CPL >= N, e.g. cpl > 50")
        print("  kl > <N>             seek next move with KL >= N, e.g. kl > 1.5")
        print("  next sac             seek next sacrifice (>=2pt material, Q>0.3")
        print("                       held 5 plies)")
        print("  temp <T>             show priors temperature-scaled by T (read-only)")

    def replay(self):
        print(f"Replaying {self.log['scenario']}. Result {self.log['result']}")
        sf_color = self.log.get("stockfish_color", None)
        vs_stockfish = self.log.get("vs_stockfish", False)
        if vs_stockfish:
            flipped = sf_color if sf_color else False
        else:
            result = self.log.get("result", 0)
            flipped = result < 0
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
                self.show_sf_overlay(cmd_cased)
            elif cmd.startswith("lc0"):
                self.show_lc0_overlay(cmd)
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
            elif cmd.startswith("cpl"):
                parts = cmd.replace(">", " ").split()
                try:
                    threshold = float(parts[-1])
                    shown = False
                    self.seek_cpl(threshold)
                except (ValueError, IndexError):
                    print("Usage: cpl > <number>")
            elif cmd.startswith("kl"):
                parts = cmd.replace(">", " ").split()
                try:
                    threshold = float(parts[-1])
                    shown = False
                    self.seek_kl(threshold)
                except (ValueError, IndexError):
                    print("Usage: kl > <number>")
            elif cmd == "next sac":
                shown = False
                self.seek_sac()
            elif cmd.startswith("temp"):
                parts = cmd.split()
                try:
                    arg = parts[1]
                    if arg.startswith("e="):
                        self.show_temp(None, target_entropy=float(arg[2:]))
                    else:
                        self.show_temp(float(arg))
                except (IndexError, ValueError):
                    print("Usage: temp <T>  or  temp e=<0-1>")
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

    def temp_for_entropy(self, priors, target_norm_entropy, tol=1e-4):
        p = np.array(priors, dtype=np.float64)
        p = p / p.sum()

        def norm_entropy_at(t):
            scaled = p ** (1.0 / t)
            scaled /= scaled.sum()
            return calc_entropy(scaled, normed_only=True)

        lo, hi = 1e-6, 10.0
        steps = 0
        best = (lo + hi) / 2
        for _ in range(60):
            mid = (lo + hi) / 2
            steps += 1
            val = norm_entropy_at(mid)
            best = mid
            if val < target_norm_entropy:
                lo = mid
            else:
                hi = mid
            if abs(val - target_norm_entropy) < 0.05:
                break
        self._last_temp_steps = steps
        return best

    def show_temp(self, temp, target_entropy=None):
        node, who, chosen = self.node_who_chosen()
        if node is None:
            print("  (no candidate_moves in log)")
            return
        cands = node.get("candidate_moves") or []
        if not cands:
            print("  (no candidates)")
            return

        priors = np.array([c.get("P", 0.0) for c in cands], dtype=np.float64)
        ucis   = [c.get("uci", "?") for c in cands]

        if target_entropy is not None:
            temp = self.temp_for_entropy(priors, target_entropy)

        p_new = priors ** (1.0 / temp)
        s = p_new.sum()
        if s <= 0:
            print("  (priors are zero)")
            return
        p_new = p_new / s

        norm_orig, _ = self.compute_norm_entropy(priors)
        norm_temp, _ = self.compute_norm_entropy(p_new)

        steps_str = f"  ({self._last_temp_steps} steps)" if target_entropy is not None else ""
        print(f"\n  T={temp:.4f}  entropy (norm'd): original = {norm_orig:.3f}  temp-scaled = {norm_temp:.3f}{steps_str}")
        print(f"  {'move':<8}  {'prior':>8}  {'temp':>8}  {'delta':>8}")
        order = np.argsort(p_new)[::-1]
        for i in order:
            delta = p_new[i] - priors[i]
            print(f"  {ucis[i]:<8}  {priors[i]:>8.4f}  {p_new[i]:>8.4f}  {delta:>+8.4f}")

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

    def generate_training_data(self, min_visits=200, cpl_threshold=60, skip_sf_moves=False, min_ply=0,
                               emit_lc0=False, policy_clip=0.6, **kwargs):
        """
        Walk the game and produce Xerces training examples.
        - Plies below min_ply are skipped.
        - If skip_sf_moves, all plies where stockfish moved are skipped.
        - Plies with fewer than min_visits total MCTS visits are skipped.
        - If sf_df was passed to __init__ and cpl_threshold is set, plies where
          the played move's CPL (loss column) exceeds cpl_threshold are skipped.
          Plies with no matching SF row are also skipped when SF data is present.
        - policy_clip caps per-move visit prob (renormalized); None disables it.
        - emit_lc0 also returns per-ply lc0 (112,8,8) features and the ply index,
          appended as two extra lists (X, M, P, Z, V, WDL, R, L, PLY).
        """
        X, M, P, Z, V, WDL, R, L, PLY = [], [], [], [], [], [], [], [], []
        result = self.result
        self.reset()

        use_cpl_filter = (self.sf_rows is not None) and (cpl_threshold is not None)

        total_plies = len(self.moves_uci)
        while self.ply < total_plies:
            move_played = self.moves_uci[self.ply]
            mover = self.who_moved().lower()
            is_white_move = "white" in mover
            is_sf_move = "stockfish" in mover

            if self.ply < min_ply:
                self.next()
                continue

            if skip_sf_moves and is_sf_move:
                self.next()
                continue

            if use_cpl_filter:
                sf_row = self.sf_row_for_ply(self.ply)
                if sf_row is None or sf_row['loss'] > cpl_threshold:
                    self.next()
                    continue

            rb = self.board
            lms = rb.legal_moves()

            node = self.tree_data.get(self.ply, {})
            if not node:
                self.next()
                continue

            cms = node.get("candidate_moves", {})
            if not cms:
                self.next()
                continue

            visits = [[x['uci'], x['visits']] for x in cms]

            if sum(v[1] for v in visits) < min_visits:
                self.next()
                continue

            visited = set(x[0] for x in visits)
            for move in [l for l in lms if l not in visited]:
                visited.add(move)
                visits.append([move, 1])

            visits = sorted(visits, key=lambda x: x[1], reverse=True)

            counts = np.array([x[1] for x in visits], dtype=np.float32)
            s = counts.sum()
            if s > 0.0:
                pi = counts / s
            else:
                n_l = len(lms)
                if n_l == 0:
                    self.next()
                    continue
                pi = np.ones(n_l, dtype=np.float32) / float(n_l)
            if policy_clip is not None:
                pi = np.minimum(pi, policy_clip)
                pi /= pi.sum()

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
            wdl = node.get("best_wdl")
            this_q = (wdl[0] - wdl[2]) if wdl is not None else 0.0
            this_q_stm = this_q if is_white_move else -this_q
            if result > 0:
                z = 1 if is_white_move else -1
            elif result < 0:
                z = -1 if is_white_move else 1
            else:
                z = 0
            
            this_wdl = node.get("best_wdl")
            X.append(x)
            M.append(mask)
            P.append(policy)
            Z.append(z)
            V.append(this_q_stm)
            WDL.append(this_wdl)
            R.append(len(self.moves_uci) - int(self.ply))
            if emit_lc0:
                L.append(rb.lc0_features())
                PLY.append(int(self.ply))

            # advance to next ply using existing helper
            self.next()

        if emit_lc0:
            return X, M, P, Z, V, WDL, R, L, PLY
        return X, M, P, Z, V, WDL, R


def load_game_index(path=None):
    if not path.endswith("game_index.json"):
        path = os.path.join(path, "game_index.json")
    if not os.path.exists(path):
        return []
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
    board, _ = reconcile_game_boards(
        game_data['start_fen'], game_data.get('history_uci'),
        game_data['moves_played'],
    )

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
        prev_df_all = combined.get("df_all", None)
    else:
        prev_df_all = None

    # accumulate chunk content
    chunk_dfs = []

    for fn in fns:
        path = os.path.join(staging, fn)
        # load each chunk (let exceptions propagate)
        with open(path, "rb") as f:
            chunk = pickle.load(f)

        cdf = chunk.get("df_all", None)
        if cdf is not None:
            chunk_dfs.append(cdf)

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

    # de dupe, then rebuild the one-row-per-game skeleton from it directly
    df_all = df_all.drop_duplicates(['game_id', 'move_num']).sort_values("ts")
    df_means = df_all.drop_duplicates(['game_id']).sort_values("ts")

    combined_new = {
        "df_all": df_all,
        "df_means": df_means
    }

    # persist atomically using existing helper
    save_pickle_atomic(combined_new, out_pkl)
    print(f"[combine] wrote combined ANALYZE_PKL -> {out_pkl} "
          f"({len(df_means)} games)")

    # delete the chunk files that we just combined
    for fn in fns:
        path = os.path.join(staging, fn)
        os.remove(path)
    print(f"[combine] removed {len(fns)} chunk files from {staging}")

    return combined_new



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



class BrpReviewBatch:
    """Step through brp_reviewables.jsonl. Every row is a single probe move,
    so next/prev walk between positions, not plies."""

    def __init__(self, jsonl_path, sf_df=None):
        self.path = pathlib.Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"reviewables not found: {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(ln) for ln in f if ln.strip()]
        if not self.rows:
            raise ValueError(f"no rows in {self.path}")
        self.sf_df = sf_df
        self.i = 0
        self.viewer = GameViewer.from_record(self.rows[0], sf_df)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return GameViewer.from_record(self.rows[i], self.sf_df)

    def seek(self, i):
        """Move to row i without printing anything."""
        self.i = max(0, min(i, len(self.rows) - 1))
        self.viewer = GameViewer.from_record(self.rows[self.i], self.sf_df)
        return self.viewer

    def goto(self, i):
        self.seek(i)
        return self.show()

    def next(self):
        return self.goto(self.i + 1)

    def prev(self):
        return self.goto(self.i - 1)

    def row(self):
        return self.rows[self.i]

    def summary(self):
        """Every row minus the tree, for scanning/filtering in pandas."""
        drop = {"tree_search_data", "moves_played", "history_uci"}
        return pd.DataFrame([
            {k: v for k, v in r.items() if k not in drop} for r in self.rows
        ])

    def fails_index(self, min_fails):
        """Index of the next row at or past min_fails, wrapping once."""
        n = len(self.rows)
        for step in range(1, n + 1):
            j = (self.i + step) % n
            if (self.rows[j].get("n_fails") or 0) >= min_fails:
                return j
        return None

    def seek_fails(self, min_fails):
        j = self.fails_index(min_fails)
        if j is None:
            print(f"[brp] no row with n_fails >= {min_fails}")
            return self.viewer
        return self.goto(j)

    def cpl_index(self, min_cpl):
        """Index of the next row at or past min_cpl probe_cpl, wrapping once."""
        n = len(self.rows)
        for step in range(1, n + 1):
            j = (self.i + step) % n
            cpl = self.rows[j].get("probe_cpl")
            if cpl is not None and cpl >= min_cpl:
                return j
        return None

    def seek_cpl(self, min_cpl):
        j = self.cpl_index(min_cpl)
        if j is None:
            print(f"[brp] no row with probe_cpl >= {min_cpl}")
            return self.viewer
        return self.goto(j)

    def show(self, top_n=5):
        self.viewer.show_board()
        self.show_header()
        self.viewer.show_moves(top_n=top_n)
        return self.viewer

    def san(self, uci):
        """SAN from the probe position; all header moves are legal there."""
        return self.viewer.board.san(uci) if uci else "?"

    def show_header(self):
        r = self.row()
        print(f"[{self.i + 1}/{len(self.rows)}] {r.get('short_fen')}  "
              f"epoch {r.get('model_epoch')}  fails {r.get('n_fails')}")
        print(f"  orig  {self.san(r.get('orig_move'))} "
              f"(cpl {r.get('orig_cpl')})  "
              f"orig_best {self.san(r.get('orig_best'))}")
        print(f"  probe {self.san(r.get('probe_move'))} "
              f"(cpl {r.get('probe_cpl')})  "
              f"sf_best {self.san(r.get('sf_best'))} "
              f"@d{r.get('sf_best_depth')} cp {r.get('sf_best_cp')}")
        print(f"  from  {r.get('src_run_tag')} / {r.get('src_game_id')} "
              f"move {r.get('src_move_num')}")

    def show_options(self):
        print("Commands:")
        print("  [Enter] / n          next probe")
        print("  b, b<N>              back N probes, e.g. b5")
        print("  g<N>                 go to probe N, e.g. g120")
        print("  f<N>                 seek next probe with n_fails >= N")
        print("  c<N>                 seek next probe with probe_cpl >= N")
        print("  q, quit, exit        quit review")
        print("  o, options, help     show this help text")
        print("  fen                  print FEN for this position")
        print("  hdr                  reprint the probe header")
        print("  sf                   stockfish overlay (default depth)")
        print("  sf<D>                stockfish eval to depth D, e.g. sf20")
        print("  lc0                  lc0 overlay (default 4000 nodes)")
        print("  lc0 <N>              lc0 eval with N nodes, e.g. lc0 8000")
        print("  pv                   xc0 principal variation (min_vis=1)")
        print("  pv<N>                xc0 pv filtered by min visits, e.g. pv8")
        print("  visits <move>        show MCTS visits for a specific move")
        print("  visits <N>           show visit info for N top moves")
        print("  temp <T>             show priors temperature-scaled by T")

    def replay(self, top_n=5):
        print(f"Reviewing {len(self.rows)} probes from {self.path.name}")
        print("Controls: Enter=next, b=back, g<N>=goto, f<N>=fails, o=options")
        shown = False
        while True:
            if not shown:
                self.show(top_n=top_n)
                shown = True
            v = self.viewer
            cmd = input("[Enter]=next, b=back, q=quit, lc0/sf/pv, o=options > ")
            cmd = cmd.strip()
            cmd_cased = cmd
            cmd = cmd.lower()
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd in ("o", "options", "help", "h", "?"):
                self.show_options()
            elif cmd == "fen":
                print(v.board.fen())
            elif cmd == "hdr":
                self.show_header()
            elif cmd.startswith("pv"):
                s = cmd[2:].strip()
                v.show_pv(min_vis=int(s) if s.isdigit() else 1)
            elif cmd.startswith("sf"):
                v.show_sf_overlay(cmd_cased)
            elif cmd.startswith("lc0"):
                v.show_lc0_overlay(cmd)
            elif cmd.startswith("visits"):
                split = [c.strip() for c in cmd.split(" ") if c.strip()]
                if split[-1].isdigit():
                    v.show_moves(top_n=int(split[-1]))
                else:
                    move_str = cmd_cased.replace(" ", "").replace("visits", "")
                    v.show_visits(move_str)
            elif cmd.startswith("temp"):
                parts = cmd.split()
                try:
                    arg = parts[1]
                    if arg.startswith("e="):
                        v.show_temp(None, target_entropy=float(arg[2:]))
                    else:
                        v.show_temp(float(arg))
                except (IndexError, ValueError):
                    print("Usage: temp <T>  or  temp e=<0-1>")
            elif cmd.startswith("g"):
                s = cmd[1:].strip()
                if s.isdigit():
                    shown = False
                    self.seek(int(s))
                else:
                    print("Usage: g<N>, e.g. g120")
            elif cmd.startswith("f"):
                s = cmd[1:].strip()
                if s.isdigit():
                    j = self.fails_index(int(s))
                    if j is None:
                        print(f"[brp] no row with n_fails >= {s}")
                    else:
                        shown = False
                        self.seek(j)
                else:
                    print("Usage: f<N>, e.g. f3")
            elif cmd.startswith("c"):
                s = cmd[1:].strip()
                if s.isdigit():
                    j = self.cpl_index(int(s))
                    if j is None:
                        print(f"[brp] no row with probe_cpl >= {s}")
                    else:
                        shown = False
                        self.seek(j)
                else:
                    print("Usage: c<N>, e.g. c200")
            elif cmd.startswith("b"):
                s = cmd[1:].strip()
                n = int(s) if s.isdigit() else 1
                shown = False
                self.seek(self.i - n)
            else:
                shown = False
                self.seek(self.i + 1)


class RecordKeeper(object):    
    def __init__(self, n_retrains, every_sec=60):
        self.n_retrains = n_retrains
        self.every_sec = every_sec
        self._last_stats_log = time.time()
        self._run_start = time.time()

        self.sims_done_total = 0
        self.total_plies = 0
        self.mcts_sims_total = 0
        self.mcts_plies_total = 0
        self.games_finished = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.training_queue = 0
        self.primary_buffer_dir = None
        self.primary_buffer_trigger = 0

        # every stat the summary reports is a counter (N/W/L/D/plies_sum),
        # so the whole-run view accumulates in O(1) instead of re-scanning a
        # retained list. recent_games stays for the last-50 runtime line, but
        # is bounded: it would otherwise grow for the life of the process.
        self.recent_games = deque(maxlen=RECENT_GAMES_CAP)
        self.scenario_stats = defaultdict(
            lambda: {"N": 0, "W": 0, "L": 0, "D": 0, "plies_sum": 0})
        self.sf_overall = {"N": 0, "W": 0, "L": 0, "D": 0, "plies_sum": 0}
        self.wins_by_scenario = defaultdict(int)
        self.telemetry = {}
        self.lc0_telemetry = {}
        # set by the runner: it owns the queue and sees the replies
        self.lc0_backlog = 0
        self.lc0_probes_done = 0
        # set by the runner from the rescorer's own audit state each pass
        self.lc0_audit_ema = {}
        self.lc0_audit_n = 0
        self.brp_admit_n = 0
        self.brp_admit_ok = 0

    def ingest_recents(self, recent):
        meta = recent['meta']

        self.games_finished += 1
        self.total_plies += meta['plies']
        self.sims_done_total += meta['sims_done_total']
        self.mcts_sims_total += meta['mcts_sims_total']
        self.mcts_plies_total += meta['mcts_plies']

        if meta['result'] > 0:
            self.white_wins += 1
        elif meta['result'] < 0:
            self.black_wins += 1
        else:
            self.draws += 1
        
        self.recent_games.append(meta)
        accumulate_game_stats(meta, self.scenario_stats, self.sf_overall,
                              self.wins_by_scenario)

    def ingest_telemetry(self, telemetry):
        # store the latest from each only. lc0 is kept apart so its throughput
        # never lands in the selfplay fleet's mps/lps.
        looper_id = telemetry['looper_id']
        info = telemetry['telemetry']

        # partial pushes carry partial_ts, not ts, so merging one cannot make
        # the rates underneath it look fresh to get_agg_metrics

        store = self.telemetry
        if telemetry.get('kind') == 'lc0':
            store = self.lc0_telemetry

        # update if existing, else create entry
        if looper_id in store.keys():
            store[looper_id].update(info)
        else:
            store[looper_id] = info

        self.maybe_log_results()

    def get_agg_metrics(self, print_span=None):
        # three push windows of slack: at a 60s cadence a tighter filter drops
        # a worker that is merely a second late and under-reports mps/lps
        time_delta = time.time() - 180

        to_sum = [
            "mps", "lps", "n_active", "n_groups", "s_collected", "s_fast",
            "s_terminals", "s_cached", "s_fast_stops", "s_collect_stops",
            "s_blocked", "s_puct", "preds_per_second",

            "s_must_visit",
            "s_skipped", "s_pruned", "s_penalty", "s_depth",

            # priors cache (per-worker) telemetry, aggregate across workers
            "cache_size", "cache_capacity", "cache_queries", "cache_hits"
        ]

        summed = defaultdict(float)
        sum_seen = set()

        to_avg = ["mbs", "batch_target", "apl", "infer_gap", "pred_wait",
                  "avg_ply", "spm", "cache_hit_ema"]
        avged = defaultdict(list)
        avg_seen = set()

        for _, info in self.telemetry.items():
            if info.get("ts", 0) < time_delta:
                continue

            # a child's counts cover its own push window, which is not the
            # span since the last print. Scale each child onto the print's
            # span so a long or stale window reads at the right size.
            window = info.get("window_s") or 0.0
            scale = print_span / window if (print_span and window > 0) else 1.0

            for k in to_sum:
                v = info.get(k, 0)
                summed[k] += v * scale if k in COUNTER_KEYS else v
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

        # span this print covers, read before the stamp moves
        print_span = now - self._last_stats_log
        self._last_stats_log = now

        # pull stats
        summed, avged = self.get_agg_metrics(print_span)
        avg_moves = (self.total_plies / max(1, self.games_finished))
        gph =  3600 * self.games_finished / (now - self._run_start)
        
        print()
        print(f" Epoch {self.n_retrains} Logging ".center(72, "~"))
        
        mps, lps = summed.get("mps", 0), summed.get("lps", 0)
        print(f"[speed stats] mps={mps:.1f}  lps={lps:.1f}  gph={gph:.2f}")
        
        print(
            f"[game stats]  finished={self.games_finished}  "
            f"W/D/L={self.white_wins}/{self.draws}/{self.black_wins}  "
            f"avg_len={avg_moves:.1f} moves")
        print("-" * 72)

        if not self.games_finished:
            print("(no games to break down)")
            print("~" * 72)
            self.log_lc0_stats()
            self.log_loop_stats(summed, avged)
            return

        # counters cover the whole run, not the last `window` games
        print_summary_from_stats(self.scenario_stats, self.sf_overall,
                                 self.wins_by_scenario)
        primary_count = (len(os.listdir(self.primary_buffer_dir))
                         if self.primary_buffer_dir else 0)
        print(
            f"live buffer: {self.training_queue}  "
            f"primary_buffer: {primary_count}/{self.primary_buffer_trigger}  "
            f"retrain number: {self.n_retrains}\n"
        )
        # chain log_loop_stats here as well
        self.log_lc0_stats()
        self.log_loop_stats(summed, avged)
        return

    def log_lc0_stats(self):
        """Throughput of the lc0 teacher fleet, kept out of [speed stats] so
        the selfplay numbers stay comparable across runs that do and do not
        have an lc0 worker."""
        fresh = [i for i in self.lc0_telemetry.values()
                 if i.get("ts", 0) >= time.time() - 180]
        if not fresh:
            return

        mps = sum(i.get("mps", 0.0) for i in fresh)
        lps = sum(i.get("lps", 0.0) for i in fresh)
        pps = sum(i.get("preds_per_second", 0.0) for i in fresh)
        active = sum(i.get("n_active", 0) for i in fresh)

        print(f"[lc0 stats] mps={mps:.2f}  lps={lps:.1f}  "
              f"preds/s={pps:.1f}  moves/hr={mps * 3600:.0f}  "
              f"active={active}")

        spm = np.mean([i.get("spm", 0.0) for i in fresh])
        batch = np.mean([i.get("apl", 0.0) for i in fresh])
        rates = [i["cache_hit_ema"] for i in fresh if "cache_hit_ema" in i]
        hit = f"{100.0 * np.mean(rates):.1f}%" if rates else "--"

        print(f"[lc0 queue] backlog={self.lc0_backlog}  "
              f"done={self.lc0_probes_done}  sims/move={spm:.0f}  "
              f"batch={batch:.0f}  cache_hit={hit}")

        E = self.lc0_audit_ema
        if 'lc0' in E:
            if self.brp_admit_n:
                pct = 100.0 * self.brp_admit_ok / self.brp_admit_n
                print(f"[lc0 audit] candidates={self.brp_admit_n}  "
                      f"confirmed blunders={self.brp_admit_ok} ({pct:.1f}%)")
            for tag, lk, xk in (("ema500 ", 'lc0', 'xc0'),
                                ("ema10k ", 'lc0_l', 'xc0_l')):
                lc0, xc0 = E[lk], E[xk]
                print(f"[lc0 audit] ply0 {tag} n={self.lc0_audit_n:<5}"
                      f" lc0={lc0:<6.1f}vs  xc0={xc0:<7.1f}"
                      f"uplift={xc0 - lc0:+.1f}")

    def log_loop_stats(self, summed, avged):
        n_groups = summed.get("n_groups", 0)
        if n_groups == 0:
            return

        s_collected = summed.get("s_collected", 0)
        s_terminals = summed.get("s_terminals", 0)
        s_cached = summed.get("s_cached", 0)
        s_fast_stops = summed.get("s_fast_stops", 0)
        s_collect_stops = summed.get("s_collect_stops", 0)
        s_blocked = summed.get("s_blocked", 0)
        s_puct = summed.get("s_puct", 0)
        s_depth = summed.get("s_depth", 0)

        s_must_visit = summed.get("s_must_visit", 0)

        s_skipped = summed.get("s_skipped", 0)
        s_pruned = summed.get("s_pruned", 0)

        avg_new = s_collected / n_groups

        total_overall = s_collected + s_terminals + s_cached
        pct_cached_overall = 100.0 * s_cached / max(1, total_overall)
        pct_term_overall = 100.0 * s_terminals / max(1, total_overall)

        f_stops_pct = 100.0 * s_fast_stops / max(1, n_groups)
        collect_stops_pct = 100.0 * s_collect_stops / max(1, n_groups)

        mbs = avged["mbs"]
        print("-" * 72)
        left1 = f"[loop stats] groups={n_groups:.0f}  mbs={mbs:.1f}"
        right1 = f"new: collected={s_collected:.0f} avg={avg_new:.2f}"

        left2 = f"[stop stats] fastpath_stops={s_fast_stops:.0f} ({f_stops_pct:.2f}%)"
        right2 = f"collect_stops={s_collect_stops:.0f} ({collect_stops_pct:.2f}%)"

        apl = avged["apl"]
        infer_gap = avged.get("infer_gap", 0.0)

        pred_wait = avged["pred_wait"]
        preds_per_sec = summed["preds_per_second"]

        duty_cycle = 0.0
        cycle_time = pred_wait + infer_gap
        if cycle_time:
            duty_cycle = pred_wait / cycle_time
        realized_preds_per_sec = duty_cycle * preds_per_sec

        left3 = f"[pred stats] batch={apl:.1f}  duty={duty_cycle * 100:.1f}%"
        right3 = f"gap={infer_gap:.3f}s wait={pred_wait:.3f}s"

        left3b = f"[pred stats] actual preds/s={realized_preds_per_sec:.1f}"
        right3b = f"ideal preds/s={preds_per_sec:.1f}"

        puct_per_leaf = s_puct / total_overall if total_overall else 0.0
        avg_depth = s_depth / total_overall if total_overall else 0.0
        blocked_pct = 100.0 * s_blocked / (s_blocked + s_collected) if (s_blocked + s_collected) > 0 else 0.0
        left4 = f"[puct stats] blocked={s_blocked:.0f} ({blocked_pct:.3f}%)"
        right4 = (
            f"evals={compact_count(s_puct)}  evals/leaf={puct_per_leaf:.1f}"
            f"  avg_depth={avg_depth:.1f}"
        )

        tot_skip = s_skipped + s_pruned
        avoided_r = tot_skip / s_puct if s_puct else 0.0
        skip_to_prune = s_skipped / s_pruned if s_pruned else 0.0
        avoid_per_leaf = tot_skip / total_overall if total_overall else 0.0
        left5 = f"[puct stats] avoidance={avoided_r:.3f}  s/p={skip_to_prune:.1f}"
        right5 = f"avoid/leaf={avoid_per_leaf:.1f}  must_visit={s_must_visit:.0f}"

        left6 = f"[cache hits] cached={s_cached:.0f} ({pct_cached_overall:.3f}%)"
        right6 = f"terminals={s_terminals:.0f} ({pct_term_overall:.3f}%)"

        # EMA of the live rate, not the run average: this should move when
        # validation or a hard probe changes what the search is doing
        sims_per_move = avged.get("spm", 0.0)

        n_active = summed["n_active"]
        avg_ply = avged["avg_ply"]
        left7 = f"[game stats] n={n_active:.0f}  avg ply={avg_ply:.2f}"
        right7 = f"sims per move={sims_per_move:.2f}"

        col_width = 40
        print(f"{left1:<{col_width}} | {right1}")
        print(f"{left2:<{col_width}} | {right2}")
        print(f"{left3:<{col_width}} | {right3}")
        print(f"{left3b:<{col_width}} | {right3b}")
        print(f"{left4:<{col_width}} | {right4}")
        print(f"{left5:<{col_width}} | {right5}")
        print(f"{left6:<{col_width}} | {right6}")
        print(f"{left7:<{col_width}} | {right7}")

        last50 = list(self.recent_games)[-50:]
        durations = [g.get("duration", 0.0) for g in last50]
        avg_runtime = None
        if sum(durations) > 0:
            avg_runtime = format_time(np.mean(durations))
            if avg_runtime:
                print(f"[game stats] last 50 runtime: {avg_runtime}")
        print("-" * 72)
