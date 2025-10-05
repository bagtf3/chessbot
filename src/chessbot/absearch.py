import time
from chessbot.search_utils import TranspositionTable, bound_flag
from chessbot.search_utils import VALUE_MATE
from pyfastchess import terminal_value_white_pov


class GAUnit:
    """
    Per-contestant search unit. each has its own TT and evaluator.
    Iterative deepening via search(), inner absearch and qsearch.
    """

    def __init__(self, evaluator=None, **limits):
        self.evaluator = evaluator
        self.tt = TranspositionTable()

        # limits
        self.dl = limits.get("depth", 5)
        self.tl = limits.get("time", None)
        self.nl = limits.get("node", None)
        # qsearch specific
        self.max_qply = limits.get("max_qply", 6)
        self.max_qcaptures = limits.get("max_qcaptures", 12)
        self.qdelta = limits.get("qdelta", None)
        
        # some stats
        self.nodes = 0
        self.qnodes = 0
        
        # qsearch specific stats
        self.use_fast_q = False
        self.max_qply_seen = 0
        self.qdepth_hist = {}
        self.qcaptures_considered = 0
        self.qtime_ms = 0

    def evaluate_board(self, board):
        if self.evaluator:
            return self.evaluator.evaluate(board)
        return board.material_count()

    def terminal_value(self, board, ply):
        """Use pyfastchess helper and map to mate-distance values."""
        res = terminal_value_white_pov(board)
        # None -> ongoing, 0 -> draw
        if not res:
            return res
        return VALUE_MATE - ply if res == 1 else -VALUE_MATE + ply

    def extract_pv(self, board, max_len=32):
        pv = []
        bb = board.clone()
        for _ in range(max_len):
            e = self.tt.probe_entry(bb.hash())
            if not e:
                break
            mv = e.get("move_uci")
            if not mv:
                break
            if not bb.push_uci(mv):
                break
            pv.append(mv)
        return pv
    
    def qsearch(self, board, alpha, beta, ply, start_time=None):
        """
        Fast quiescence that delegates to the C++ board.qsearch(...) wrapper.
    
        Updates same counters as Python qsearch:
          - self.max_qply_seen
          - self.qdepth_hist[ply]
          - self.qnodes (adds nodes returned by C++)
        Returns: score (int/float)
        """
        # bookkeeping
        if ply > self.max_qply_seen:
            self.max_qply_seen = ply
        self.qdepth_hist[ply] = self.qdepth_hist.get(ply, 0) + 1
        
        if self.tl and start_time and (time.time() - start_time) > self.tl:
            return self.evaluate_board(board)
        if self.nl and self.nodes >= self.nl:
            return self.evaluate_board(board)
    
        # prepare qopts dict to pass to C++
        qopts = {
            "max_qply": getattr(self, "max_qply", None),
            "max_qcaptures": getattr(self, "max_qcaptures", None),
            "qdelta": getattr(self, "qdelta", None),
            "time_limit_ms": None if not self.tl else int(self.tl * 1000),
            "node_limit": self.nl
        }
        # remove keys with None because the wrapper expects only provided fields
        qopts = {k: v for k, v in qopts.items() if v is not None}
    
        # call into C++ qsearch: returns (score_cp, qstats_dict)
        score, qstats = board.qsearch(alpha, beta, self.evaluator, qopts)
    
        # update counters from returned qstats
        # qstats keys: "qnodes", "max_qply_seen", "captures_considered", "time_used_ms"
        self.qnodes += qstats.get("qnodes", 0)
    
        max_qply_local = qstats.get("max_qply_seen", 0)
        if max_qply_local > self.max_qply_seen:
            self.max_qply_seen = max_qply_local
    
        # keep captures_considered/time_used
        self.qcaptures_considered = qstats.get("captures_considered", 0)
        self.qtime_ms = qstats.get("time_used_ms", 0)
    
        return score

    def absearch(self, board, depth, alpha, beta, ply, start_time=None):
        """Negamax alpha-beta with TT. cutoff => eval fallback"""
        # time/node cutoff as base-case -> return evaluation
        if self.tl and start_time and (time.time() - start_time) > self.tl:
            return self.evaluate_board(board)
        if self.nl and self.nodes >= self.nl:
            return self.evaluate_board(board)

        self.nodes += 1

        key = board.hash()
        tt_ent = self.tt.probe_entry(key)
        if tt_ent and tt_ent.get("key") == key and tt_ent.get("depth", -1) >= depth:
            s = tt_ent.get("score")
            flag = tt_ent.get("flag")
            val = self.tt.score_from_TT(s, ply)
            if flag == bound_flag.EXACTBOUND:
                return val
            if flag == bound_flag.LOWERBOUND and val >= beta:
                return val
            if flag == bound_flag.UPPERBOUND and val <= alpha:
                return val

        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply, start_time)

        tt_best = tt_ent["move_uci"] if tt_ent else None

        scores_moves = board.ordered_moves(tt_best)
        if not scores_moves:
            if board.in_check():
                return -VALUE_MATE + ply
            return 0

        moves = [m for (s, m) in scores_moves]

        # canonical setup: remember old alpha
        old_alpha = alpha
        best_score = -VALUE_MATE
        best_move = None

        # main negamax loop
        moves_made = 0
        for mv in moves:
            moves_made += 1
            board.push_uci(mv)
            score = -self.absearch(board, depth - 1, -beta, -alpha, ply + 1, start_time)
            board.unmake()

            if score > best_score:
                best_score = score
                best_move = mv

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break

        # baked in terminal check for free
        if moves_made == 0:
            return -VALUE_MATE + ply if board.in_check() else 0

        # canonical bound classification AFTER the loop (matches Disservin)
        if best_score >= beta:
            flag = bound_flag.LOWERBOUND
        elif alpha != old_alpha:
            flag = bound_flag.EXACTBOUND
        else:
            flag = bound_flag.UPPERBOUND

        self.tt.store_entry(key, depth, flag, best_score, best_move or "0000", ply)
        return best_score

    def absearch_root(self, board, depth, alpha, beta, start_time=None):
        key = board.hash()
        tt_entry = self.tt.probe_entry(key)
        tt_best = tt_entry["move_uci"] if tt_entry else None
        
        scores, moves = [], []
        for s, m in board.ordered_moves(tt_best):
            scores.append(s)
            moves.append(m)
        
        if not moves:
            return self.qsearch(board, alpha, beta, 0, start_time), []
        
        best_score = -VALUE_MATE
        best_move = None
        best_pv = []

        for mv in moves:
            board.push_uci(mv)
            score = -self.absearch(board, depth - 1, -beta, -alpha, 1, start_time)
            child_pv = self.extract_pv(board, max_len=depth-1)
            board.unmake()

            if score > best_score:
                best_score = score
                best_move = mv
                best_pv = [mv] + child_pv

            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        
        if best_move:
            self.tt.store_entry(
                key, depth, bound_flag.EXACTBOUND, best_score, best_move, 0
            )
        
        return best_score, best_pv

    def search(self, board, max_depth=None, verbose=False):
        self.nodes = 0
        self.qnodes = 0
        self.max_qply_seen = 0
        self.qdepth_hist = {}
        
        max_depth = max_depth or self.dl
        start = time.time()
        best_move = None
        best_score = 0
        pv = []
        info = {"depths": [], "n_nodes": 0, "qnodes": 0, "time": 0.0, "best_pv": []}
        aspiration = 100
    
        for depth in range(1, max_depth + 1):
            # time check up-front
            if self.tl and (time.time() - start) > self.tl:
                break
    
            alpha = -VALUE_MATE
            beta = VALUE_MATE
            if best_move is not None:
                alpha = best_score - aspiration
                beta = best_score + aspiration
    
            score, pv = self.absearch_root(board, depth, alpha, beta, start_time=start)
            # if aspiration fail, rerun wide window (no exceptions)
            if score <= alpha + 1 or score >= beta - 1:
                score, pv = self.absearch_root(
                    board, depth, -VALUE_MATE, VALUE_MATE, start_time=start
                )
    
            best_score = score
            best_move = pv[0] if pv else None
            elapsed = time.time() - start
    
            # store pv in per-depth record and keep current best pv
            info["depths"].append(
                (depth, best_move, best_score, self.nodes, self.qnodes,
                 elapsed, pv)
            )
            info["best_pv"] = pv
    
            if verbose:
                print(
                    f"depth {depth}: best={best_move} score={best_score} "
                    f"nodes={self.nodes} qnodes={self.qnodes} time={elapsed:.2f}s "
                    f"pv={' '.join(pv) if pv else '[]'}"
                )
    
            if self.tl and elapsed > self.tl:
                break
    
        info["n_nodes"] = self.nodes
        info["qnodes"] = self.qnodes
        info["last_pv"] = pv if pv else []
        info["time"] = time.time() - start
        return best_move, best_score, info
    
    def print_qstats(self):
        print("qnodes:", self.qnodes, "max_qply_seen:", self.max_qply_seen)
        # print top 8 ply counts (sorted by ply)
        items = sorted(self.qdepth_hist.items())
        print("qdepth histogram (ply:count):")
        for ply, cnt in items[:8]:
            print(f"  {ply:2d}: {cnt}")
        if len(items) > 8:
            print(f"  ... ({len(items)} ply levels total)")
