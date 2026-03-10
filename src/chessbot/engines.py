import re
import subprocess
import chess, chess.engine

from chessbot import LC0_LOC


def parse_int_safe(s, default=None):
    try:
        return int(s)
    except Exception:
        return default


def parse_float_safe(s, default=None):
    if s is None:
        return default
    t = s.strip()
    if not t:
        return default
    try:
        return float(t)
    except Exception:
        return default


class Lc0Parser:
    """Parse verbose move stats from lc0 info strings."""

    def __init__(self):
        self.pat_move = re.compile(r"^([a-h][1-8][a-h][1-8][qrbn]?)\s")
        self.pat_n    = re.compile(r"\bN:\s*([0-9]+)")
        self.pat_p    = re.compile(r"\(P:\s*([0-9.]+)%\)")
        self.pat_q    = re.compile(r"\(Q:\s*([\-0-9.]+)\)")
        self.pat_wl   = re.compile(r"\(WL:\s*([\-0-9.]+)\)")
        self.pat_d    = re.compile(r"\(D:\s*([0-9.]+)\)")
        self.pat_m    = re.compile(r"\(M:\s*([0-9.]+)\)")

    def parse_info(self, info):
        """Return list of move-stat dicts from one UCI info dict."""
        s = info.get("string")
        if not s:
            return []
        if not isinstance(s, list):
            s = [s]

        rows = []
        for line in s:
            m = self.pat_move.search(line)
            if not m:
                continue
            mv = m.group(1)
            if mv == "node":
                continue

            row = {"move": mv}

            m2 = self.pat_n.search(line)
            if m2:
                n = parse_int_safe(m2.group(1))
                if n is not None:
                    row["N"] = n

            m2 = self.pat_p.search(line)
            if m2:
                p = parse_float_safe(m2.group(1))
                if p is not None:
                    row["P_pct"] = p

            for key, pat in [
                ("Q",  self.pat_q),
                ("WL", self.pat_wl),
                ("D",  self.pat_d),
                ("M",  self.pat_m),
            ]:
                m2 = pat.search(line)
                if m2:
                    v = parse_float_safe(m2.group(1))
                    if v is not None:
                        row[key] = v

            rows.append(row)
        return rows


def lc0_analyze(board, nodes=4000, lc0_loc=None, engine_cfg=None):
    """Run lc0 on board for the given node budget. Returns sorted rows.

    engine_cfg: dict of UCI options merged on top of the defaults.
    """
    loc = lc0_loc or LC0_LOC
    parser = Lc0Parser()
    limit = chess.engine.Limit(nodes=nodes)
    last_by_move = {}

    cfg = {"VerboseMoveStats": True, "MinibatchSize": 64}
    if engine_cfg:
        cfg.update(engine_cfg)

    with chess.engine.SimpleEngine.popen_uci(loc, stderr=subprocess.DEVNULL) as eng:
        eng.configure(cfg)
        with eng.analysis(board, limit, info=chess.engine.INFO_ALL) as an:
            for info in an:
                rows = parser.parse_info(info)
                for r in rows:
                    last_by_move[r["move"]] = r

    rows = sorted(last_by_move.values(), key=lambda r: r.get("N", 0), reverse=True)
    return rows
