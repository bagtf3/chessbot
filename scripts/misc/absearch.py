import time
from chessbot.search_utils import TranspositionTable, bound_flag
from chessbot.search_utils import VALUE_MATE
from pyfastchess import terminal_value_white_pov


class GAUnit:
    """
    Per-contestant search unit. White-POV minimax alpha-beta.
    White always maximizes; Black always minimizes.
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

        # stats
        self.nodes = 0
        self.qnodes = 0

        #qsearch specific
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
            return int(res) if res is not None else res
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
        """Call into the C++ qsearch wrapper and update qstats."""
        if ply > self.max_qply_seen:
            self.max_qply_seen = ply
        self.qdepth_hist[ply] = self.qdepth_hist.get(ply, 0) + 1

        if self.tl and start_time and (time.time() - start_time) > self.tl:
            return self.evaluate_board(board)
        if self.nl and self.nodes >= self.nl:
            return self.evaluate_board(board)

        qopts = {
            "max_qply": getattr(self, "max_qply", None),
            "max_qcaptures": getattr(self, "max_qcaptures", None),
            "qdelta": getattr(self, "qdelta", None),
            "time_limit_ms": None if not self.tl else int(self.tl * 1000),
            "node_limit": self.nl
        }
        qopts = {k: v for k, v in qopts.items() if v is not None}

        # board.qsearch returns (score, qstats) per your binding
        score, qstats = board.qsearch(alpha, beta, self.evaluator, qopts)

        self.qnodes += qstats.get("qnodes", 0)
        max_qply_local = qstats.get("max_qply_seen", 0)
        if max_qply_local > self.max_qply_seen:
            self.max_qply_seen = max_qply_local

        self.qcaptures_considered = qstats.get("captures_considered", 0)
        self.qtime_ms = qstats.get("time_used_ms", 0)

        # qsearch already returns white-POV value (per convention)
        return score

    def absearch(self, board, depth, alpha, beta, ply, white_to_move, start_time=None):
        """
        Minimax alpha-beta with TT (white-POV).
        alpha/beta are bounds in white-POV space.
        If board.side_to_move() == "w": node is maximizing.
        Else: node is minimizing.
        """
        # time / node cutoffs
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
            # TT flags interpreted in white-POV alpha/beta space
            if flag == bound_flag.EXACTBOUND:
                return val
            if flag == bound_flag.LOWERBOUND and val >= beta:
                return val
            if flag == bound_flag.UPPERBOUND and val <= alpha:
                return val
        
        # terminal check
        tv = self.terminal_value(board, ply)
        if tv is not None:
            return tv
        
        # leaf: use qsearch
        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply, start_time)
        
        # ordering: recommend TT best first
        tt_best = tt_ent["move_uci"] if tt_ent else None
        scores_moves = board.ordered_moves(tt_best)
        moves = [m for (s, m) in scores_moves]
        old_alpha, old_beta = alpha, beta

        # maximizing
        if white_to_move:
            best_score = -VALUE_MATE
            best_move = None
            # search children, maximize
            for mv in moves:
                board.push_uci(mv)
                score = self.absearch(
                    board, depth - 1, alpha, beta, ply + 1,
                    not white_to_move, start_time)
                board.unmake()
                if score > best_score:
                    best_score = score
                    best_move = mv
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    # cutoff: store as LOWERBOUND (value >= beta)
                    break

            # determine bound for TT (white-POV space)
            if best_score >= beta:
                flag = bound_flag.LOWERBOUND
            elif alpha != old_alpha:
                flag = bound_flag.EXACTBOUND
            else:
                flag = bound_flag.UPPERBOUND

        else:
            # minimizing node (black to move)
            best_score = VALUE_MATE
            best_move = None
            for mv in moves:
                board.push_uci(mv)
                score = self.absearch(
                    board, depth - 1, alpha, beta, ply + 1,
                     not white_to_move, start_time)
                board.unmake()
                if score < best_score:
                    best_score = score
                    best_move = mv
                if score < beta:
                    beta = score
                if alpha >= beta:
                    # cutoff: store as UPPERBOUND (value <= alpha)
                    break

            # determine bound for TT at minimize node
            if best_score <= alpha:
                flag = bound_flag.UPPERBOUND
            elif beta != old_beta:
                flag = bound_flag.EXACTBOUND
            else:
                flag = bound_flag.LOWERBOUND

        # store in TT (white-POV score)
        self.tt.store_entry(key, depth, flag, best_score, best_move or "0000", ply)
        return best_score

    def absearch_root(self, board, depth, alpha, beta, start_time=None):
        """Root wrapper: enumerates moves, returns best_score (white-POV)
        and principal variation as a list of UCI moves. Root respects white-POV:
        if black to move, we pick the minimal child score.
        """
        legal = board.legal_moves()
        if not legal:
            # if no legal moves, delegate to qsearch at root and return pv=[]
            qv = self.qsearch(board, alpha, beta, 0, start_time)
            return qv, []

        key = board.hash()
        tt_entry = self.tt.probe_entry(key)
        tt_best = tt_entry["move_uci"] if tt_entry else None

        scores_moves = board.ordered_moves(tt_best)
        moves = [m for (s, m) in scores_moves]

        maximizing = True if board.side_to_move() == "w" else False

        if maximizing:
            best_score = -VALUE_MATE
        else:
            best_score = VALUE_MATE

        best_move = None
        best_pv = []

        old_alpha, old_beta = alpha, beta

        for mv in moves:
            board.push_uci(mv)
            child_score = self.absearch(
                board, depth - 1, alpha, beta, 1, not maximizing, start_time
            )
            child_pv = self.extract_pv(board, max_len=depth-1)
            board.unmake()

            # choose according to side to move at root
            better = child_score > best_score if maximizing else child_score < best_score
            if better:
                best_score = child_score
                best_move = mv
                best_pv = [mv] + child_pv

            # update alpha/beta for root side
            if maximizing:
                if child_score > alpha:
                    alpha = child_score
            else:
                if child_score < beta:
                    beta = child_score

            if alpha >= beta:
                break

        if best_move:
            # store exact root result in TT
            self.tt.store_entry(
                key, depth, bound_flag.EXACTBOUND, best_score, best_move, 0
            )

        return best_score, best_pv

    def search(self, board, max_depth=None, verbose=False):
        """Iterative deepening entry point (keeps previous API)."""
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
        aspiration = 150

        for depth in range(1, max_depth + 1):
            if self.tl and (time.time() - start) > self.tl:
                break

            alpha = -VALUE_MATE
            beta = VALUE_MATE
            if best_move is not None:
                alpha = best_score - aspiration
                beta = best_score + aspiration

            score, pv = self.absearch_root(board, depth, alpha, beta, start_time=start)
            # aspiration fail -> full window
            if score <= alpha + 1 or score >= beta - 1:
                score, pv = self.absearch_root(
                    board, depth, -VALUE_MATE, VALUE_MATE, start_time=start
                )

            best_score = score
            best_move = pv[0] if pv else None
            elapsed = time.time() - start

            info["depths"].append(
                (depth, best_move, best_score, self.nodes, self.qnodes, elapsed, pv))

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
