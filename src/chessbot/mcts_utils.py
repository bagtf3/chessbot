import json
import os
import random
import uuid
from time import time as _now

import chess, chess.syzygy
import numpy as np

from pyfastchess import MCTSTree as fasttree
from pyfastchess import terminal_value_white_pov

from chessbot import ENDGAME_LOC
from chessbot.utils import kl_divergence, rnd

TABLEBASE = None


def get_tablebase():
    """Open the syzygy tablebase once per process and reuse the handle.

    It used to be opened and closed inside the per-ply terminal check, which
    reopened the tablebase files on every probe.
    """
    global TABLEBASE
    if TABLEBASE is None and ENDGAME_LOC and os.path.isdir(ENDGAME_LOC):
        TABLEBASE = chess.syzygy.open_tablebase(ENDGAME_LOC)
    return TABLEBASE

class MCTSTree(fasttree):
    def __init__(self, board, cfg):
        self.config = cfg

        self.c_puct = float(cfg.c_puct)

        # set floor and ceiling 
        self.sims_floor = cfg.sims_floor
        self.sims_ceiling_schedule = {}

        # can also be a dict, interpreted as a sims schedule
        if isinstance(cfg.sims_ceiling, dict):
            # start with the lowest, update at each interval
            sc = cfg.sims_ceiling.copy()
            lowest = min(sc.keys())
            self.sims_ceiling = sc.pop(lowest)
            self.sims_ceiling_schedule = sc
        else:
            self.sims_ceiling = cfg.sims_ceiling

        self.pruning_factor = cfg.pruning_factor
        super().__init__(board, self.c_puct, self.sims_ceiling, self.pruning_factor,
                         cfg.uniform_eps, cfg.prior_clip_max)

        # bookkeeping
        self.board = board
        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()

        # make sure we're starting at the correct sim schedule entry
        if self.sims_ceiling_schedule:
            for i in range(self.n_plies):
                if i in self.sims_ceiling_schedule:
                    self.set_new_sims_ceiling(n_plies=i)

        self._move_started_at = _now()
        self.sims_completed_this_move = 0

        self.n_moves_played = 0
        self.sims_done_total = 0

        # configure C++ tree
        # explicit casts required — these are passed directly to C++
        if cfg.add_root_noise:
            self.set_dirichlet(float(cfg.dirichlet_eps), float(cfg.dirichlet_alpha))
        self.set_reuse_tree(bool(cfg.reuse_tree))
        self.set_vscale(float(cfg.vscale))
        self.set_fpu_reduction(float(cfg.fpu_reduction))
        self.set_qema_span(float(cfg.qema_span))
        self.set_qdelta_span(float(cfg.qdelta_span))
        self.set_contempt(
            float(cfg.contempt_zero_q),
            float(cfg.contempt_full_q),
            float(cfg.contempt_fight_c),
        )
        self.set_tempscale_entropy_target(float(cfg.tempscale_entropy_target))
        self.set_tempscale_trigger_q(float(cfg.tempscale_trigger_q))

        # sim_stop_reason stays Python-owned: apply_stockfish_result writes
        # "sf" into it directly, which C++ has no way to produce.
        self.sim_stop_reason = ""

        # tiered early-stop rule now lives in C++, evaluated inside
        # collect_many_leaves's own loop so checkins land on exact sim
        # counts (see pyfastchess mcts.cpp evaluate_early_stop).
        self.es_params = dict(
            es_check_every=int(cfg.es_check_every),
            tier1_consec=int(cfg.es_tier1_consec),
            tier1_jsd_thresh=float(cfg.es_tier1_jsd_thresh),
            tier2_consec=int(cfg.es_tier2_consec),
            tier2_jsd_thresh=float(cfg.es_tier2_jsd_thresh),
        )
        self.set_early_stop_params(min_sims=int(cfg.sims_floor), **self.es_params)

        self.effort_lut = {}
        self.effort_q_edges = None
        self.effort_d_edges = None
        self.effort_ply_edges = None
        if getattr(cfg, "effort_scale_table", ""):
            import json
            from chessbot.effort_scale import Q_EDGES, D_EDGES, PLY_EDGES, bin_key
            lut_data = json.load(open(cfg.effort_scale_table))
            self.effort_lut = lut_data.get(str(getattr(cfg, "effort_alpha", 0.0)), {})
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
        if self.config.sample_moves == False:
            sample = False

        elif self.board.piece_count() <= 20:
            sample = False
        
        # at/after the convergence ply, just use C++/base behavior
        elif (self.n_plies >= self.config.move_sample_temp_plies):
            sample = False

        else:
            sample = True

        if not sample:
            return xerces_move, xerces_method, None

        # if here we are sampling
        rows = self.root_child_visits()
        if not rows:
            return xerces_move, xerces_method, None

        ucis = [u for u, _ in rows]
        visits = np.array([n for _, n in rows], dtype=np.float64)

        # trivial cases
        if len(ucis) == 1 or visits.sum() <= 0.0:
            move = ucis[0]
            other = xerces_move if move != xerces_move else None
            return move, "most_visited", other

        # force sampling only from the top 5 moves
        top_k = min(5, len(ucis))
        top_ucis = ucis[:top_k]
        top_visits = visits[:top_k]

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
        rsc, details = self.robust_selection_criteria(5, 60)
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

        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()

        self.set_sim_budget(float(self.sims_ceiling))

        if self.sims_ceiling_schedule:
            if self.n_plies in self.sims_ceiling_schedule.keys():
                self.set_new_sims_ceiling()

        if self.effort_lut:
            self.set_early_stop_params(
                min_sims=int(self.sims_floor), **self.es_params)
            self.apply_effort_scale()

        self.reset_early_stop()

    def apply_effort_scale(self):
        from chessbot.effort_scale import bin_key
        key = bin_key(
            self.n_plies, self.saved_parent_q, self.saved_parent_pdraw,
            self.effort_q_edges, self.effort_d_edges, self.effort_ply_edges,
        )
        s = self.effort_lut.get(key, 1.0)
        if s != 1.0:
            self.set_sim_budget(float(round(self.sims_ceiling * s)))
            self.set_early_stop_params(
                min_sims=int(round(self.sims_floor * s)), **self.es_params)

    def set_new_sims_ceiling(self, n_plies=None):
        if n_plies is None:
            n_plies = self.n_plies

        new_ceiling = self.sims_ceiling_schedule.get(n_plies, None)
        if new_ceiling is None:
            return
        
        self.sims_ceiling = new_ceiling
        self.set_sim_budget(float(new_ceiling))

    def get_sim_decision_probs(self):
        # not implemented right now
        return None

    def stop_simulating(self):
        # do at least 1 sims to stabilize the tree
        if self.sims_completed_this_move < 1:
            return False

        # check for only 1 move
        if len(self.board.legal_moves()) < 2:
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
        self.sims_done_total += self.sims_completed_this_move
        
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
    def __init__(self, board, meta, cfg):
        self.game_id = str(uuid.uuid4())
        self.started_at = _now()

        self.config = cfg

        self.board = board
        self.starting_fen = self.board.fen()
        self.meta = meta
        
        self.vs_stockfish = meta['vs_stockfish']

        # if vs stockfish, we need to maintain a python-chess boardas well
        self.python_chess_board = None
        if self.vs_stockfish:
            self.python_chess_board = chess.Board(self.board.fen())
        
        self.stockfish_is_white = meta['stockfish_is_white']
        self.sf_search_depth = []
        tree_cfg = self.config
        scenario = meta.get('scenario', '')
        if scenario == 'piece_training' and not self.config.is_validation_run:
            tree_cfg = self.config.copy()
            tree_cfg.move_sample_temp_range = [0.2, 0.2]
        
        self.tree = MCTSTree(self.board, tree_cfg)
        self.tree_data = {}
        self.moves_played = []
        self.recents = []
        
        self.mat_adv_counter = 0
        self.resign_counter_white = 0
        self.resign_counter_black = 0
        self.outcome = None
        self.end_reason = ""
        self.plies = 0
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
        
        c_puct = self.tree.c_puct
        
        # timing
        start = self.tree._move_started_at
        if start is None:
            start = _now()
            self.tree._move_started_at = start
        elapsed = _now() - start
    
        # C++ summaries
        avg_depth, max_depth = self.tree.depth_stats()
        rsc, details = self.tree.robust_selection_criteria(5, 60)
        rsc = {} if not rsc else rsc
        if details is None:
            return
        
        sims = int(self.tree.sims_completed_this_move)
        total_children = len(details)
        visited_children = sum([1 for cd in details if cd.N > 0])

        turn = self.turn() # STM
        data = {
            "move_played": mv, "selection_method":method, "xc0_move": xc0_move,
            "sims": sims, "time": rnd(elapsed, 3),
            "avg_depth": rnd(avg_depth, 2), "max_depth": max_depth,
            "children_visited": visited_children,
            "total_children": total_children,
            "stop_reason": self.tree.sim_stop_reason, "stm": turn
        }
        
        # IMPORTANT, this MUST happen before the move is pushed otherwise the vals change
        best_d = details[0]
        best_wdl = (rnd(best_d.win, 4), rnd(best_d.draw, 4), rnd(best_d.loss, 4))
        best_q = rnd(best_d.Q, 4)
        Q_white = best_q
        Q_stm = Q_white if turn else -Q_white
        if self.is_stockfish_turn():
            Q_stm = self.sf_eval
            Q_white = Q_stm if turn else -Q_stm

        data['best_wdl'] = best_wdl
        data['Q_stm'] = Q_stm
        data['Q_white'] = Q_white

        if self.is_stockfish_turn() and self.sf_wdl is not None:
            data['sf_wdl'] = self.sf_wdl
        
        # keep a small list of items for gameplay checking
        self.recents.append((mv, Q_stm, Q_white, best_q, turn))

        # sumN for U term
        sumN = max(1, root.N)

        candidate_moves = []
        for cd in details:
            U = c_puct * cd.prior * (sumN ** 0.5) / (1 + cd.N)
            cm = {
                "uci": cd.uci, "visits": cd.N,
                "visit_share": cd.visit_share, "last_visit": cd.last_visit,
                "Q": rnd(cd.Q, 4), "P": rnd(cd.prior, 4), "U": rnd(U, 4),
                "Qema": rnd(cd.Qema, 4), "Qdelta_sign": rnd(cd.Qdelta_sign, 4),
                "Q_draw": rnd(cd.draw, 4) if cd.N > 0 else float("nan"),
                "is_terminal": cd.is_terminal, "rcs": rnd(rsc.get(cd.uci, 0.0), 4)
            }
            
            candidate_moves.append(cm)
        data["candidate_moves"] = candidate_moves

        # fast PV via C++; for sampled moves trace from xc0 choice, not the sampled move
        pv_start = "" if method == "stockfish" else (xc0_move or mv)
        pv = []
        pv_items = self.tree.principal_variation(24, pv_start)
        for x in pv_items:
            pv.append({
                "uci": x.uci, "visits": int(x.visits),
                "P": rnd(x.P, 4), "Q": rnd(x.Q, 4)
            })
        
        # attach PV snapshot (may be empty if no deeper visited chain exists)
        data["pv"] = pv

        self.tree_data[self.plies] = data

        es_rows = self.tree.es_debug_rows()
        if es_rows:
            visits_list = [cd.N for cd in details]
            priors_list = [cd.prior for cd in details]
            kl = kl_divergence(visits_list, priors_list)
            if random.random() < 0.10 or kl > 0.75:
                self.dump_jsd_debug(turn, sims, self.tree.sim_stop_reason, es_rows)

    def dump_jsd_debug(self, turn, final_sims, final_stop_reason, es_rows):
        out_dir = os.path.join(self.config.run_dir, "jsd_debug")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.game_id}.jsonl")

        with open(out_path, "a") as f:
            for row in es_rows:
                line = {
                    "game_id": self.game_id, "ply": self.plies, "stm": turn,
                    "final_sims": final_sims, "final_stop_reason": final_stop_reason,
                    "sims": row.sims, "jsd": rnd(row.jsd, 6),
                    "top_uci": row.top_uci, "second_uci": row.second_uci,
                    "delta_12": row.delta_12, "dQ12": rnd(row.dQ12, 4),
                    "Q1": rnd(row.Q1, 4), "Qema1": rnd(row.Qema1, 4),
                }
                f.write(json.dumps(line) + "\n")
    
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
            # r is a tuple: (mv, Q_stm, Q_white, best_q, turn)
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

        # raw material difference
        if cfg.use_material_diff:
            mat_diff = self.board.material_count()
            if abs(mat_diff) >= cfg.material_diff_cutoff:
                self.mat_adv_counter += 1
            else:
                self.mat_adv_counter = 0

            if self.mat_adv_counter >= cfg.material_diff_cutoff_span:
                self.outcome = 1.0 if mat_diff > 0 else -1.0
                self.end_reason = "material_diff"
                return True

        # resignation
        if cfg.allow_resignation and self.plies >= cfg.resign_min_plies and self.recents:
            _, q_stm, _, _, turn = self.recents[-1]
            if turn:
                if q_stm < -cfg.resign_threshold:
                    self.resign_counter_white += 1
                else:
                    self.resign_counter_white = 0
            else:
                if q_stm < -cfg.resign_threshold:
                    self.resign_counter_black += 1
                else:
                    self.resign_counter_black = 0

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
        if self.plies > cfg.max_game_length:
            self.outcome = 0.0
            self.end_reason = "max_length"
            return True
        # if we made it here the game is active
        return False
