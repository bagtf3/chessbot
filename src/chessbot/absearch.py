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
        
        # qsearch depth tracking
        self.max_qply_seen = 0
        self.qdepth_hist = {}

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
        # record ply stats
        if ply > self.max_qply_seen:
            self.max_qply_seen = ply
        self.qdepth_hist[ply] = self.qdepth_hist.get(ply, 0) + 1

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

        # **BUG FIX**: return the best found quiescent score (was a bare `return`)
        return best
    
    def qsearch_fast(self, board, alpha, beta, ply, start_time=None):
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
    
        # same cutoff logic as python qsearch (preserve behavior)
        if self.tl and start_time and (time.time() - start_time) > self.tl:
            return self.evaluate_board(board)
        if self.nl and self.nodes >= self.nl:
            return self.evaluate_board(board)
    
        # prepare qopts dict to pass to C++
        qopts = {
            "max_qply": getattr(self, "max_qply", None),
            "max_qcaptures": getattr(self, "max_qcaptures", None),
            "qdelta": getattr(self, "qdelta", None),
            "time_limit_ms": None if not getattr(self, "tl", None) else int(self.tl * 1000),
            "node_limit": getattr(self, "nl", None),
        }
        # remove keys with None because the wrapper expects only provided fields
        qopts = {k: v for k, v in qopts.items() if v is not None}
    
        # call into C++ qsearch: returns (score_cp, qstats_dict)
        score, qstats = board.qsearch(alpha, beta, self.evaluator, qopts)
    
        # update counters from returned qstats
        # qstats keys: "qnodes", "max_qply_seen", "captures_considered", "time_used_ms"
        qnodes_local = int(qstats.get("qnodes", 0))
        self.qnodes += qnodes_local
    
        max_qply_local = int(qstats.get("max_qply_seen", 0))
        if max_qply_local > self.max_qply_seen:
            self.max_qply_seen = max_qply_local
    
        # keep captures_considered/time_used if you track them; example:
        if hasattr(self, "captures_considered"):
            self.captures_considered = getattr(self, "captures_considered", 0)
            self.captures_considered += int(qstats.get("captures_considered", 0))
        if hasattr(self, "qtime_ms"):
            self.qtime_ms = getattr(self, "qtime_ms", 0)
            self.qtime_ms += int(qstats.get("time_used_ms", 0))
    
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



if __name__ == '__main__':
    from pyfastchess import Board as fastboard
    from chessbot.utils import random_init

    # assume these imports are available in your project
    import numpy as np
    import chess                      # used by the psqt tables
    from pyfastchess import Evaluator # C++ evaluator binding

    from chessbot.psqt import psqt_values

    # helper: convert the psqt_values dict to a (4,6,64) int32 array
    def build_psqt_buckets(psqt_values):
        # piece order expected by Evaluator: P,N,B,R,Q,K -> indices 0..5
        piece_order = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN, chess.KING]
    
        # base 6x64 from psqt_values
        base = np.zeros((6, 64), dtype=np.int32)
        for pi, piece in enumerate(piece_order):
            tbl = psqt_values[piece]  # list of 64 ints
            base[pi, :] = np.array(tbl, dtype=np.int32)
    
        # bucket0..bucket2: reuse base (you can tweak later via GA)
        buckets = np.zeros((4, 6, 64), dtype=np.int32)
        buckets[0] = base.copy()   # 0..19 plies
        buckets[1] = base.copy()   # 20..39 plies
        buckets[2] = base.copy()   # 40..59 plies
    
        # bucket3 = endgame tweak: reward advanced pawns and king centrality
        eg = base.copy()
    
        # increase pawn rewards on advanced ranks (white-pov):
        # indices are 0..63 with 0==a1, 8==a2, ... 56==a8
        # we add modest positive bonuses to ranks 5-7 (indices 32..55), bigger on 7th rank
        pawn_idx = 0  # pawn is piece index 0
        for sq in range(64):
            rank = (sq // 8) + 1  # rank 1..8
            # reward advanced pawns (white-perspective)
            if rank >= 6:
                eg[pawn_idx, sq] += 30   # strongly encourage passed/advanced pawns
            elif rank == 5:
                eg[pawn_idx, sq] += 12
            elif rank == 4:
                eg[pawn_idx, sq] += 6
    
        # king: encourage centralization in endgame (reduce "safety" penalty)
        king_idx = 5
        # add +40 to central squares (d4,e4,d5,e5 indices)
        central_squares = []
        files = [3, 4]   # d,e (0=a)
        ranks = [3, 4]   # 4th,5th rank (0-based rank indices 3,4)
        for r in ranks:
            for f in files:
                central_squares.append(r*8 + f)
        for sq in central_squares:
            eg[king_idx, sq] += 40
    
        # Slightly reduce piece-square values for heavy pieces in endgame (encourage activity)
        for pidx in (3, 4):  # rook and queen
            eg[pidx, :] += 6
    
        buckets[3] = eg
    
        # Flatten to shape (4, 384) or (1536,) — evaluator accepts multiple shapes
        return buckets.reshape((4, 6*64)).astype(np.int32)
    
    psqt_arr = build_psqt_buckets(psqt_values)


    # mobility weights (pawn..king) in centipawns per reachable empty square
    mobility_weights = [0, 4, 6, 2, 1, 0]  # ints, length 6
    
    # tactical weights: 3 features per piece (attacked_by_lower, defended, hanging)
    # ordering: [pawn_att_low, pawn_def, pawn_hang, knight_att_low, ...]
    tactical_weights = [
        # pawn
        -10,  4,  -40,
        # knight
        -35, 10, -120,
        # bishop
        -30, 10, -110,
        # rook
        -45, 10, -150,
        # queen
        -90, 20, -350,
        # king
        0,   20, 0
    ]

    king_weights = [0, 0, 0]   # placeholder (unused for now)
    stm_bias = 30               # no side-to-move bias by default
    global_scale = 100         # 100 == 1.00
    
    # Assemble config dict for Evaluator.configure()
    weights_dict = {
        "psqt": psqt_arr,                    # shape (4,384) int32
        "mobility_weights": mobility_weights,
        "tactical_weights": tactical_weights,
        "king_weights": king_weights,
        "stm_bias": stm_bias,
        "global_scale": global_scale
    }

    # instantiate and configure the C++ evaluator
    ev = Evaluator()
    ev.configure(weights_dict)

    # quick test
    b = fastboard()   # startpos
    print("eval:", ev.evaluate(b))
    print("itemized:", ev.evaluate_itemized(b))

    # plug into GAUnit (replace your import/module path as needed)
    u = GAUnit(evaluator=ev, depth=5, time=10, node=None, verbose=True)
    u.max_qply = 5
    u.max_qcaptures = 8
    u.qdelta = 0

    # now search as before:
    best, score, info = u.search(random_init(1), max_depth=5, verbose=True)
    u.print_qstats()
    
    
    # quick smoke test
b = random_init(1)
u = GAUnit(evaluator=ev, depth=3, time=1)
res = u.qsearch(b, -10000, 10000, 0)
print("qsearch returned:", res, type(res))
