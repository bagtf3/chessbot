import json
import os
import uuid
from time import time as _now

import chess, chess.syzygy
import numpy as np

from pyfastchess import MCTSTree as fasttree
from pyfastchess import Board as fastboard
from pyfastchess import terminal_value_white_pov

from pyfastchess import SearchParams

from chessbot import ENDGAME_LOC
from chessbot.config import XC0H_K
from chessbot.utils import rnd

# C++ search param names; the trailing-underscore entries are derived flags
SEARCH_FIELDS = [k for k in SearchParams().to_dict() if not k.endswith("_")]
# set by MCTSTree.move_params each move rather than read from the config
MOVE_FIELDS = ("c_puct", "sims_min", "sims_max")

POLICY_DIM = 1858
# max_length adjudication, given at least N plies: white-POV mean resign_q over
# the last N; at or past THRESH the side ahead wins, else a draw
MAX_LENGTH_ADJ_N = 6
MAX_LENGTH_ADJ_THRESH = 0.5
TABLEBASE = None
PV_MAX = 64


def build_raw_visits_sparse(board, ucis, visits):
    """Every legal move as sparse (int16 idx, int16 count), index-sorted. Legal
    but unvisited moves are kept with count 0 so rescore sees the full move set
    and derives the policy target from this."""
    vmap = {u: v for u, v in zip(ucis, visits)}
    legal = board.legal_moves()
    idxs = np.asarray(board.moves_to_indices(legal), dtype=np.int16)
    counts = np.array([vmap.get(u, 0) for u in legal], dtype=np.int16)
    order = np.argsort(idxs, kind="stable")
    return (idxs[order], counts[order])


def get_tablebase():
    """Open the syzygy tablebase once per process and reuse the handle.

    It used to be opened and closed inside the per-ply terminal check, which
    reopened the tablebase files on every probe.
    """
    global TABLEBASE
    if TABLEBASE is None and ENDGAME_LOC and os.path.isdir(ENDGAME_LOC):
        TABLEBASE = chess.syzygy.open_tablebase(ENDGAME_LOC)
    return TABLEBASE

def scheduled(schedule, ply, default):
    """Value of the {start_ply: value} entry with the largest start ply at or
    below ply (the first entry before that); default with no schedule."""
    if not schedule:
        return default
    items = sorted(schedule.items())
    val = items[0][1]
    for start, v in items:
        if ply >= start:
            val = v
    return val


class MCTSTree(fasttree):
    def __init__(self, board, cfg):
        self.config = cfg

        # c_puct can also be a dict {start_ply: c_puct} on absolute game ply
        self.c_puct_schedule = cfg.c_puct if isinstance(cfg.c_puct, dict) else None

        # bookkeeping
        self.board = board
        self.only_one_move = len(board.legal_moves()) < 2
        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()

        self._move_started_at = _now()
        self.sims_completed_this_move = 0

        self.n_moves_played = 0

        # every C++ search param comes from the config field of the same name;
        # c_puct and sims_min / sims_max are resolved per move
        params = {k: getattr(cfg, k) for k in SEARCH_FIELDS if k not in MOVE_FIELDS}
        params.update(self.move_params())
        super().__init__(board, params)

        # sim_stop_reason stays Python-owned: apply_stockfish_result writes
        # "sf" into it directly, which C++ has no way to produce.
        self.sim_stop_reason = ""

        self.effort_lut = {}
        self.effort_q_edges = None
        self.effort_d_edges = None
        self.effort_ply_edges = None
        if cfg.effort_scale_table:
            from chessbot.effort_scale import Q_EDGES, D_EDGES, PLY_EDGES, bin_key
            lut_data = json.load(open(cfg.effort_scale_table))
            self.effort_lut = lut_data.get(str(cfg.effort_alpha), {})
            edges = lut_data.get("edges", {})
            self.effort_q_edges   = tuple(edges.get("Q",   Q_EDGES))
            self.effort_d_edges   = tuple(edges.get("D",   D_EDGES))
            self.effort_ply_edges = tuple(edges.get("PLY", PLY_EDGES))
            if self.effort_lut:
                test_key = bin_key(
                    20, 0.15, 0.45,
                    self.effort_q_edges, self.effort_d_edges, self.effort_ply_edges,
                )

                if test_key not in self.effort_lut:
                    sample = next(iter(self.effort_lut))
                    raise RuntimeError(
                        f"effort LUT key mismatch: {test_key!r} not in table "
                        f"(sample key: {sample!r})")
        self.saved_parent_q = 0.0
        self.saved_parent_pdraw = 0.0

    def best(self):
        """
        Returns a 3-tuple:
        (move_uci_to_play, selection_method, xerces_top_move_if_diff_else_none)

        xerces_top_move_if_diff_else_none is set only when we sampled a different
        move than Xerces would have played deterministically.
        """
        xerces_move, xerces_method = self.select_xerces_top_move()

        # determine if we need to sample or not
        if self.board.piece_count() <= 20:
            sample = False

        # at/after the convergence ply, just use C++/base behavior;
        # move_sample_temp_plies = 0 never samples
        elif (self.n_plies >= self.config.move_sample_temp_plies):
            sample = False

        else:
            sample = True

        if not sample:
            return xerces_move, xerces_method, None

        # if here we are sampling
        details = self.root_child_details()
        if not details:
            return xerces_move, xerces_method, None

        sign = 1.0 if self.board.white_to_move() else -1.0
        lead = details[0]
        q_floor = sign * lead.Q - self.config.sample_max_q_delta
        cands = [lead] + [
            cd for cd in details[1:self.config.sample_top_k]
            if cd.N >= 1 and sign * cd.Q >= q_floor
        ]

        # trivial cases
        if len(cands) == 1 or lead.N <= 0:
            move = lead.uci
            other = xerces_move if move != xerces_move else None
            return move, "most_visited", other

        top_ucis = [cd.uci for cd in cands]
        top_visits = np.array([cd.N for cd in cands], dtype=np.float64)

        temp_min = self.config.move_sample_temp_range[0]
        temp_max = self.config.move_sample_temp_range[1]
        temp_plies = self.config.move_sample_temp_plies

        frac = max(0.0, min(1.0, (temp_plies - self.n_plies) / temp_plies))
        temp = temp_min + (temp_max - temp_min) * frac

        # build stable logits from visits: log(visits) keeps scale sane
        logits = np.log(top_visits + 1e-12) / max(1e-12, temp)
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        probs = exps / exps.sum()

        idx = np.random.choice(len(top_ucis), p=probs)
        move = top_ucis[idx]
        other = xerces_move if move != xerces_move else None
        return move, "sampled", other

    def select_xerces_top_move(self):
        """
        Deterministic Xerces choice (no sampling), returning:
        (uci, method) where method is "robust" or "most_visited".
        """
        if self.sim_stop_reason == "full" and self.config.use_robust:
            uci = self.select_using_robust()
            return uci, "robust"

        uci, _ = super().best()
        return uci, "most_visited"

    def select_using_robust(self):
        """
        Deterministic robust selector.
        Returns just the uci (so callers can label it however they want).
        """
        rsc, details = self.robust_selection_criteria()
        if (not rsc) or (not details) or (len(details) == 1):
            uci, _ = super().best()
            return uci

        most_visits = details[0].N
        # anything > 2k visits is well-searched
        visit_threshold = min(most_visits * 0.7, 2000) 

        best_rsc = -np.inf
        best_uci = None
        for d in details:
            if (d.uci in rsc) and (d.N >= visit_threshold):
                this_rsc = rsc[d.uci]
                if this_rsc > best_rsc:
                    best_rsc = this_rsc
                    best_uci = d.uci

        if best_uci is None:
            uci, _ = super().best()
            return uci

        return best_uci
        
    def advance(self, board, move_uci):
        """
        Safe root advance: mutate the C++ tree, then sync Python-side bookeeping.
        Drops any queued (stale) leaf pointers.
        """
        # Capture parent Q/pdraw before the tree resets on advance.
        r = self.root()
        if r is not None and r.N > 0:
            self.saved_parent_q = r.Q
            self.saved_parent_pdraw = r.p_draw / r.N

        # Never try to assign self.root in Python; C++ owns the root.
        self.advance_root(move_uci)

        # Keep external board & counters in sync for your caller's logic
        board.push_uci(move_uci)
        self.board = board
        self.only_one_move = len(board.legal_moves()) < 2

        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()

        self.set_params(self.move_params(self.effort_scale()))
        self.reset_early_stop()

    def move_params(self, scale=1.0):
        """c_puct, sims_min and sims_max for the current ply."""
        cfg = self.config
        ceiling = scheduled(cfg.sims_schedule, self.n_plies, cfg.sims_ceiling)
        return {
            "c_puct": float(scheduled(self.c_puct_schedule, self.n_plies, cfg.c_puct)),
            "sims_min": int(round(cfg.sims_floor * scale)),
            "sims_max": int(round(ceiling * scale)),
        }

    def effort_scale(self):
        if not self.effort_lut:
            return 1.0
        from chessbot.effort_scale import bin_key
        key = bin_key(
            self.n_plies, self.saved_parent_q, self.saved_parent_pdraw,
            self.effort_q_edges, self.effort_d_edges, self.effort_ply_edges,
        )
        return self.effort_lut.get(key, 1.0)

    def get_sim_decision_probs(self):
        # not implemented right now
        return None

    def stop_simulating(self):
        # do at least 1 sims to stabilize the tree
        if self.sims_completed_this_move < 1:
            return False

        if self.only_one_move:
            self.sim_stop_reason = "Only 1 legal move"
            return True

        if self.es_tripped:
            self.sim_stop_reason = self.es_stop_reason
            return True

        return False

    def reset_for_new_move(self):
        """
        Resets everything, counts visits from previous trees.
        """

        self.n_moves_played += 1

        existing = 0
        try:
            r = self.root()
            # Prefer the root's own N if available
            n_root = int(r.N)
            if n_root > 0:
                existing = n_root
            else:
                # sum child visits if root.N isnt populated
                rows = self.root_child_visits() or []
                existing = int(sum([n for (_u, n) in rows]))
        except Exception as e:
            print(f"error encountered calculating sims: {e}")
            existing = 0
        
        # carry the existing visit count forward
        self.sims_completed_this_move = existing

        # standard housekeeping
        self._move_started_at = _now()
        self.sim_stop_reason = ""


class ChessGame(object):
    def __init__(self, spec):
        self.game_id = str(uuid.uuid4())
        self.started_at = _now()

        self.config = spec.cfg
        self.init_fen = spec.fen
        self.preopen_moves_uci = spec.moves or []

        # build the board
        board = fastboard(spec.fen)
        for mv in spec.moves:
            board.push_uci(mv)

        self.start_fen = board.fen()
        self.board = board

        meta = spec.meta
        self.meta = meta
        
        self.vs_stockfish = meta['vs_stockfish']

        # if vs stockfish, we need to maintain a python-chess board as well
        self.python_chess_board = None
        if self.vs_stockfish:
            self.python_chess_board = chess.Board(self.board.fen())
        
        self.stockfish_is_white = meta['stockfish_is_white']
        self.sf_search_depth = []

        # scenario overrides are already applied to the spec's cfg
        self.tree = MCTSTree(self.board, self.config)
        self.tree_data = {}
        self.moves_played = []
        self.recents = []
        
        self.resign_counter_white = 0
        self.resign_counter_black = 0
        self.outcome = None
        self.end_reason = ""
        self.plies = 0
        # set by the end_validation command: adjudicate as at max_game_length
        self.wrap_up = False
        self.mcts_sims_total = 0
        self.mcts_plies = 0
        self.sf_eval = None
        self.sf_wdl = None

        self.sf_pending = False
        self.sf_ready = False
        self.sf_res_tup = None
        self.tf_last_resolved = 0

        # set when to start checking for draw by agreement
        if self.config.use_eval_draw:
            self.next_eval_draw_check = self.config.eval_draw_min_plies
        else:
            self.next_eval_draw_check = 999

    def emit_header(self, model_epoch):
        """Two dicts: mem_summary (light index/telemetry essentials) and rest
        (seed, config, and replay fields). Merged they are the on-disk header;
        mem_summary alone is the index row. Called at game end, so outcome,
        history, and timing are final. model_epoch and game_end come from looper."""
        cfg = self.config
        game_end = _now()
        
        mem_summary = {
            "ts": self.started_at,
            "game_start": self.started_at,
            "game_end": game_end,
            "duration": game_end - self.started_at,
            "model_epoch": model_epoch,
            "game_id": self.game_id,
            "scenario": self.meta.get("scenario", ""),
            "plies": self.plies,
            "result": int(self.outcome or 0),
            "end_reason": self.end_reason,
            "vs_stockfish": self.vs_stockfish,
            "stockfish_color": self.stockfish_is_white,
            "sf_depth": self.meta.get("sf_depth"),
            "start_type": self.meta.get("start_type"),
            "mcts_sims_total": self.mcts_sims_total,
            "mcts_plies": self.mcts_plies,
            "start_fen": self.start_fen,
        }
        rest = {
            "run_tag": cfg.run_tag,
            "init_fen": self.init_fen,
            "start_ply": len(self.preopen_moves_uci),
            "preopen_moves_uci": self.preopen_moves_uci,
            "history_uci": self.board.history_uci(),
            "encoding": cfg.encoding_type,
            "c_puct": cfg.c_puct,
            "sims_floor": cfg.sims_floor,
            "sims_ceiling": cfg.sims_ceiling,
            "sims_schedule": cfg.sims_schedule,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "dirichlet_eps": cfg.dirichlet_eps,
            "prior_uniform_eps": cfg.prior_uniform_eps,
            "fpu_reduction": cfg.fpu_reduction,
            "move_sample_temp_range": cfg.move_sample_temp_range,
            "move_sample_temp_plies": cfg.move_sample_temp_plies,
        }
        return mem_summary, rest

    def turn(self, return_bool=True):
        if return_bool:
            return self.board.white_to_move()
        return self.board.side_to_move()
    
    def is_stockfish_turn(self):
        return self.vs_stockfish and (self.stockfish_is_white == self.turn())
    
    def push_move(self, mv, method, xc0_move):
        # collect search data then push and update
        self.collect_tree_search_data(mv, method, xc0_move)

        if method != "stockfish":
            self.mcts_sims_total += self.tree.sims_completed_this_move
            self.mcts_plies += 1

        # advance tree (pushes move) and reset
        self.tree.advance(self.board, mv)

        # keep python chess board sync'd too
        if self.python_chess_board is not None:
            self.python_chess_board.push(chess.Move.from_uci(mv))
                                             
        self.tree.reset_for_new_move()
        self.moves_played.append(mv)
        self.plies += 1
        return self.check_for_terminal()
    
    def collect_tree_search_data(self, mv, method, xc0_move):
        root = self.tree.root()
        if not root.is_expanded:
            return
        
        sp = self.tree.get_params("c_puct", "sims_min", "sims_max")
        c_puct = sp["c_puct"]

        # timing
        start = self.tree._move_started_at
        if start is None:
            start = _now()
            self.tree._move_started_at = start
        elapsed = _now() - start
    
        # C++ summaries
        avg_depth, max_depth = self.tree.depth_stats()
        rsc, details = self.tree.robust_selection_criteria()
        rsc = {} if not rsc else rsc
        if details is None:
            return
        
        sims = int(self.tree.sims_completed_this_move)
        total_children = len(details)
        visited_children = sum([1 for cd in details if cd.N > 0])

        turn = self.turn()  # STM, True = white

        # IMPORTANT: read node stats and encode the board BEFORE the move is
        # pushed, otherwise the values change.
        best_d = details[0]
        if turn:
            best_wdl = (rnd(best_d.win, 4), rnd(best_d.draw, 4), rnd(best_d.loss, 4))
        else:
            best_wdl = (rnd(best_d.loss, 4), rnd(best_d.draw, 4), rnd(best_d.win, 4))
        rw = root.nn_wdl_stm
        root_wdl_nn = (rnd(rw[0], 4), rnd(rw[1], 4), rnd(rw[2], 4))

        best_q = rnd(best_d.Q, 4)
        Q_white = best_q
        Q_stm = Q_white if turn else -Q_white
        # resignation reads the unscaled search WDL, not the vscaled Q
        resign_q = best_wdl[0] - best_wdl[2]
        if self.is_stockfish_turn():
            Q_stm = self.sf_eval
            Q_white = Q_stm if turn else -Q_stm
            resign_q = Q_stm

        # keep a small list of items for gameplay checking
        self.recents.append((mv, Q_stm, Q_white, best_q, turn, resign_q))

        # boost- and Qmm-force-decided visits are left out of the policy target
        ucis = [cd.uci for cd in details]
        visits = [cd.N - cd.boosted_N - cd.forced_N for cd in details]
        gb = self.tree.gap_block()

        data = {
            "abs_ply": self.board.history_size(),
            "stm": turn,
            "move_played": mv,
            "xc0_move": xc0_move,
            "sel_method": method,
            "stop_reason": self.tree.sim_stop_reason,
            "time": rnd(elapsed, 3),
            "sims": sims,
            "sims_min": int(sp["sims_min"]),
            "sims_max": int(sp["sims_max"]),
            "best_wdl": best_wdl,
            "root_wdl_nn": root_wdl_nn,
            "xc0h": np.asarray(self.board.history_tokens(XC0H_K), dtype=np.int16),
            "raw_visits": build_raw_visits_sparse(self.board, ucis, visits),
            "avg_depth": rnd(avg_depth, 2), "max_depth": max_depth,
            "sharp_by_depth": list(self.tree.sharp_by_depth()),
            "gap_block_sims": gb.sims,
            "gap_block_leader": gb.leader or None,
            "children_visited": visited_children,
            "total_children": total_children,
        }
        if self.is_stockfish_turn():
            # SF's tanh(cp) value at play time, STM-POV
            data["sf_value"] = rnd(self.sf_eval, 4)

        # sumN for U term
        sumN = max(1, root.N)

        candidate_moves = []
        for cd in details:
            U = c_puct * cd.prior * (sumN ** 0.5) / (1 + cd.N)
            cm = {
                "uci": cd.uci, "visits": cd.N,
                "visit_share": cd.visit_share, "last_visit": cd.last_visit,
                "Q": rnd(cd.Q, 4), "P": rnd(cd.prior, 4),
                "P_raw": rnd(cd.raw_prior, 4), "U": rnd(U, 4),
                "Qema": rnd(cd.Qema, 4), "Qdelta_sign": rnd(cd.Qdelta_sign, 4),
                "Qdema": rnd(cd.Qdema, 4),
                "boosted": cd.boosted_N,
                "Qmm": rnd(cd.Qmm, 4),
                "Q_draw": rnd(cd.draw, 4) if cd.N > 0 else float("nan"),
                "is_terminal": cd.is_terminal, "rcs": rnd(rsc.get(cd.uci, 0.0), 4)
            }
            
            candidate_moves.append(cm)
        data["candidate_moves"] = candidate_moves

        # pv traces Xerces's choice (most-visited on SF turns); pv_played traces
        # the move actually played when it differs (sampled or SF)
        if method == "stockfish":
            pv_start, xerces_pick = "", (details[0].uci if details else None)
        else:
            pv_start = xerces_pick = xc0_move or mv
        pv = self.pv_dicts(pv_start)
        pv_played = self.pv_dicts(mv) if mv != xerces_pick else None
        data["pv"] = pv
        data["pv_played"] = pv_played
        data["pv_len"] = len(pv)
        data["pv_played_len"] = None if pv_played is None else len(pv_played)

        self.tree_data[self.plies] = data

    def pv_dicts(self, start):
        return [{"uci": x.uci, "visits": int(x.visits), "P": rnd(x.P, 4),
                 "Q": rnd(x.Q, 4)}
                for x in self.tree.principal_variation(PV_MAX, start)]

    def set_stockfish_result(self, res_tup):
        self.sf_res_tup = res_tup
        self.sf_ready = True

    def apply_stockfish_result(self, res_tup):
        sf_v, best_move = res_tup[0], res_tup[1]
        self.sf_wdl = res_tup[2] if len(res_tup) > 2 else None

        self.sf_eval = sf_v
        self.sf_res_tup = None
        self.sf_ready = False
        self.sf_pending = False
        self.tree.sim_stop_reason = "sf"
        return self.push_move(best_move, "stockfish", None)

    def make_move_from_tree(self):
        """ Play best-by-visits  """
        root = self.tree.root()
        if not root.is_expanded:
            return False

        mv, method, xc0_mv = self.tree.best()
        if mv is None:
            return False
        
        return self.push_move(mv, method, xc0_mv)

    def check_for_eval_draw(self, cfg):
        n_last = cfg.eval_draw_span
        thresh_d = cfg.eval_draw_thresh
            
        # if we don't yet have a full window, schedule when we will
        if len(self.recents) < n_last:
            needed = n_last - len(self.recents)
            self.next_eval_draw_check = self.plies + needed
            return False
        
        # check and see if we've passed the test, if not when to check again
        violated = False
        violating_index = None
        for i, r in enumerate(reversed(self.recents[-n_last:])):
            # r is a tuple: (mv, Q_stm, Q_white, best_q, turn, resign_q)
            if abs(r[1]) > thresh_d:
                violated = True
                violating_index = i
                break
        
        # dont test again until its possible to have an eval_draw
        if violated:            
            needed = n_last - violating_index
            self.next_eval_draw_check = self.plies + needed
            return False

        # no eval draws with queen(s) on the board.
        if self.board.any_queens():
            return False
        
        # if here, agree to draw
        else:
            return True
    
    def check_for_terminal(self):
        cfg = self.config
        # look for conventional terminal states
        reason, result = self.board.is_game_over()
        if reason != 'none':
            self.outcome = terminal_value_white_pov(self.board)
            self.end_reason = reason
            return True

        # resignation: hits count from the first ply, so a resignation can
        # land on resign_min_plies itself
        if cfg.allow_resignation and self.recents:
            _, _, _, _, turn, resign_q = self.recents[-1]
            if turn:
                if resign_q < -cfg.resign_threshold:
                    self.resign_counter_white += 1
                else:
                    self.resign_counter_white = 0
            else:
                if resign_q < -cfg.resign_threshold:
                    self.resign_counter_black += 1
                else:
                    self.resign_counter_black = 0

            if self.plies >= cfg.resign_min_plies:
                if self.resign_counter_white >= cfg.resign_consecutive:
                    self.outcome = -1.0
                    self.end_reason = "resign"
                    return True
                if self.resign_counter_black >= cfg.resign_consecutive:
                    self.outcome = 1.0
                    self.end_reason = "resign"
                    return True

        # Syzygy probe if few pieces
        # flip syzygy on if about to end due to length
        if cfg.use_syzygy or (self.plies >= cfg.max_game_length):
            if self.board.piece_count() <= 5:
                # may not work so just go as normal
                try:
                    outcomes = {-2: -1, -1:-1, 0:0, 1:1, 2:1}
                    # gotta flip back to python chess here
                    chess_board = chess.Board(self.board.fen())
                    table_res = get_tablebase().probe_wdl(chess_board)

                    # -1 and 1 are not guaranteed winners
                    if table_res in [-2, 0, 2]:
                        table_res = table_res if chess_board.turn else -1*table_res
                        self.outcome = outcomes[table_res]
                        self.end_reason = "syzygy"
                        return True
                except:
                    pass

        # check for draws based on eval or move limit
        if self.plies >= self.next_eval_draw_check:
            if self.check_for_eval_draw(cfg):
                self.outcome = 0.0
                self.end_reason = "eval_draw"
                return True

        # check for overall game_length limit
        if self.plies > cfg.max_game_length or self.wrap_up:
            self.outcome = 0.0
            self.end_reason = "max_length"
            if len(self.recents) >= MAX_LENGTH_ADJ_N:
                last = self.recents[-MAX_LENGTH_ADJ_N:]
                q = np.mean([r[5] if r[4] else -r[5] for r in last])
                if abs(q) >= MAX_LENGTH_ADJ_THRESH:
                    self.outcome = float(np.sign(q))
                    self.end_reason = "resign (max_length)"
                else:
                    self.end_reason = "eval_draw (max_length)"
            return True
        # if we made it here the game is active
        return False
