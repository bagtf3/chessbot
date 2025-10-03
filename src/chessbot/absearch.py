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

        self.nodes = 0
        self.qnodes = 0

    def evaluate_board(self, board):
        if self.evaluator:
            return self.evaluator.evaluate(board)
        return board.material_count()

    def terminal_value(self, board, ply):
        """Use pyfastchess helper and map to mate-distance values."""
        res = terminal_value_white_pov(board)
        # None => ongoing, 0 => draw
        if not res:
            return res
        return VALUE_MATE - ply if res > 0 else -VALUE_MATE + ply

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

    def move_order_key(self, board, mv, tt_best):
        if tt_best and mv == tt_best:
            return 100000
        if board.is_capture(mv):
            return 1000 + board.mvvlva(mv)
        if board.gives_check(mv):
            return 500
        return 0

    def qsearch(self, board, alpha, beta, ply, start_time=None):
        """Quiescence: captures only. low-allocation, hot-loop friendly."""
        # time/node cutoff -> return static evaluation
        if self.tl and start_time and (time.time() - start_time) > self.tl:
            return self.evaluate_board(board)
        if self.nl and self.nodes >= self.nl:
            return self.evaluate_board(board)
    
        self.qnodes += 1
    
        # local refs for speed
        eval_fn = self.evaluate_board
        is_capture = board.is_capture
        mvvlva = board.mvvlva
        legal_moves = board.legal_moves()
    
        stand = eval_fn(board)
        if stand >= beta:
            return stand
        if alpha < stand:
            alpha = stand
    
        # collect captures with precomputed mvvlva scores
        scored = []
        append = scored.append
        for m in legal_moves:
            if is_capture(m):
                append((mvvlva(m), m))
    
        if not scored:
            return stand
    
        scored.sort(reverse=True)  # sort by mvvlva score (first tuple item)
    
        best = stand
        for _, mv in scored:
            if not board.push_uci(mv):
                continue
            sc = -self.qsearch(board, -beta, -alpha, ply + 1, start_time)
            board.unmake()
            if sc >= beta:
                return sc
            if sc > best:
                best = sc
                if sc > alpha:
                    alpha = sc
        return best


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

        tv = self.terminal_value(board, ply)
        if tv is not None:
            return tv

        tt_best = tt_ent["move_uci"] if tt_ent else None

        legal = board.legal_moves()
        if not legal:
            if board.in_check():
                return -VALUE_MATE + ply
            return 0

        moves = sorted(legal, key=lambda m: self.move_order_key(board, m, tt_best),
                       reverse=True)

        best_score = -VALUE_MATE
        best_move = None
        flag = bound_flag.UPPERBOUND

        for mv in moves:
            if not board.push_uci(mv):
                continue
            score = -self.absearch(board, depth - 1, -beta, -alpha, ply + 1, start_time)
            board.unmake()

            if score > best_score:
                best_score = score
                best_move = mv
            if score > alpha:
                alpha = score
                flag = bound_flag.EXACTBOUND
            if alpha >= beta:
                flag = bound_flag.LOWERBOUND
                break

        self.tt.store_entry(key, depth, flag, best_score, best_move or "0000", ply)
        return best_score

    def absearch_root(self, board, depth, alpha, beta, start_time=None):
        legal = board.legal_moves()
        if not legal:
            return self.qsearch(board, alpha, beta, 0, start_time), []

        key = board.hash()
        tt_entry = self.tt.probe_entry(key)
        tt_best = tt_entry["move_uci"] if tt_entry else None

        moves = sorted(
            legal, key=lambda m: self.move_order_key(board, m, tt_best),
            reverse=True)

        best_score = -VALUE_MATE
        best_move = None
        best_pv = []

        for mv in moves:
            if not board.push_uci(mv):
                continue
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
            self.tt.store_entry(key, depth, bound_flag.EXACTBOUND,
                                best_score, best_move, 0)
        return best_score, best_pv

    def search(self, board, max_depth=None, verbose=False):
        self.nodes = 0
        self.qnodes = 0
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


if __name__ == '__main__':
    from pyfastchess import Board as fastboard
    from chessbot.utils import random_init
    
    u = GAUnit(evaluator=None, depth=5, time=10, node=None, verbose=True)
    b = random_init(10)
    #best, score, info = u.search(b, verbose=True)

    # show top 50 lines sorted by cumulative time
    %prun -l 50 u.search(b, verbose=True)
