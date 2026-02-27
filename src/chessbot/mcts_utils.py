import uuid
from time import time as _now

import chess, chess.syzygy
import tl2cgen as tl2
import numpy as np

from pyfastchess import MCTSTree as fasttree
from pyfastchess import terminal_value_white_pov

from chessbot import ENDGAME_LOC
from chessbot.review import score_to_value_stm_pov
from chessbot.utils import rnd
import chessbot.utils as cbu
from collections import defaultdict


class MCTSTree(fasttree):
    def __init__(self, board, cfg):
        self.config = cfg

        # if c_puct is given as a list, pick an option randomly
        # must be float for c++
        self.c_puct = float(cbu.maybe_random_from_list(cfg.c_puct))

        # set floor and ceiling 
        self.sims_floor = cfg.sims_floor
        self.sims_ceiling_schedule = {}
        
        # ceiling can be a list for varied gameplay
        self.sims_ceiling = cbu.maybe_random_from_list(cfg.sims_ceiling)

        # can also be a dict, interpreted as a sims schedule
        if isinstance(cfg.sims_ceiling, dict):
            # start with the lowest, update at each interval
            sc = cfg.sims_ceiling.copy()
            lowest = min(sc.keys())
            self.sims_ceiling = sc.pop(lowest)
            self.sims_ceiling_schedule = sc
        else:
            self.sims_ceiling = cfg.sims_ceiling

        # naive early stop/bonus sims based on 1-2 move visit delta
        # interpret as ratio
        if cfg.target_delta < 1:
            self.target_delta = np.floor(cfg.target_delta*self.sims_ceiling)

        # otherwise use number per se
        else:
            self.target_delta = int(cfg.target_delta)

        self.pruning_factor = cfg.pruning_factor
        super().__init__(board, self.c_puct, self.sims_ceiling, self.pruning_factor)

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
        
        # root noise
        self.add_root_noise = cfg.add_root_noise
        self.root_noise_added = False

        # epsilon may be sampled
        if self.add_root_noise:
            self.dirichlet_eps = float(cbu.maybe_random_from_list(cfg.dirichlet_eps))
        else:
            self.dirichlet_eps = 0.0

        # early-stop rolling state
        self._es_last_checked_at = 0
        self._es_tripped = False
        self.sim_stop_reason = ""

        # for early stop
        self.most_visited = []
        self.runner_up = []
        self.visit_delta = []

        self.es_fails = defaultdict(int)
        self.extension_reasons = defaultdict(int)
        self.es_times = []

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

        # at/after the convergence ply, just use C++/base behavior
        elif (self.n_plies >= 30) or (self.board.piece_count() <= 20):
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

        # temperature schedule: linear decay from temp_max (ply 0) to temp_min (ply 30)
        temp_min = self.config.move_sample_temp_range[0]
        temp_max = self.config.move_sample_temp_range[1]

        frac = max(0.0, min(1.0, (30.0 - self.n_plies) / 30.0))
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
        if self.sims_completed_this_move >= self.config.robust_only_above:
            uci = self.select_using_robust()
            return uci, "robust"

        uci, _ = super().best()
        return uci, "most_visited"

    def select_using_robust(self):
        """
        Deterministic robust selector.
        Returns just the uci (so callers can label it however they want).
        """
        rsc, details = self.robust_selection_criteria(5, 100)
        if (not rsc) or (not details) or (len(details) == 1):
            uci, _ = super().best()
            return uci

        most_visits = details[0].N
        sims_done = self.sims_completed_this_move
        floor = 500 + 0.1 * sims_done
        visit_threshold = min(max(200, most_visits * 0.7), floor)

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
        # Never try to assign self.root in Python; C++ owns the root.
        self.advance_root(move_uci)

        # Keep external board & counters in sync for your caller's logic
        board.push_uci(move_uci)
        self.board = board
            
        # add noise to the root for exploration
        self.root_noise_added = False
        self.add_root_dirichlet_noise()

        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()

        # sims budget may have been adjusted with bonuses, so reset
        self.set_sim_budget(float(self.sims_ceiling))

        # if a ratio is used, update that
        if self.config.target_delta < 1:
            self.target_delta = np.floor(self.config.target_delta*self.sims_ceiling)
        
        # check the sims schedule, update if applicapable
        if self.sims_ceiling_schedule:
            if self.n_plies in self.sims_ceiling_schedule.keys():
                self.set_new_sims_ceiling()
        
        self.most_visited.clear()
        self.runner_up.clear()
        self.visit_delta.clear()

    def set_new_sims_ceiling(self, n_plies=None):
        if n_plies is None:
            n_plies = self.n_plies

        new_ceiling = self.sims_ceiling_schedule.get(n_plies, None)
        if new_ceiling is None:
            return
        
        self.sims_ceiling = new_ceiling
        # this updates the c++ tree so e.g. smart pruning knows the new budget
        self.set_sim_budget(float(new_ceiling))

        # if a ratio is used, update that
        if self.config.target_delta < 1:
            self.target_delta = np.floor(self.config.target_delta*new_ceiling)

    def needs_root_noise(self, check_sims=False):
        check = self.add_root_noise and not self.root_noise_added
        if not check:
            return False

        if not check_sims:
            return check
        return check and (self.sims_completed_this_move > 0)

    def add_root_dirichlet_noise(self):
        if not self.needs_root_noise():
            return

        super().add_root_dirichlet_noise(
            eps=self.dirichlet_eps, alpha=self.config.dirichlet_alpha)
        
        self.root_noise_added = True

    def get_sim_decision_probs(self):
        # not implemented right now
        return None

    def maybe_early_stop(self):
        if self._es_tripped:
            return True

        # upper limit check
        cfg = self.config
        sims_done = self.sims_completed_this_move

        # if not at the floor, stop here
        if sims_done < cfg.sims_floor:
            return False
        
        # upper limit check
        if sims_done >= cfg.sims_absolute_ceiling:
            self._es_tripped = True
            return True
        
        # everything else is gated
        if sims_done - self._es_last_checked_at < cfg.es_check_every:
            return False
        
        # register this check
        self._es_last_checked_at = sims_done
        
        es_time_start = _now()

        # evenly weighted visits, vs, Q, Qema, dS
        rsc, details = self.robust_selection_criteria(5, 100)

        # would only see this if the tree isnt very far along or is in transition
        if not details or len(details) < 2:
            return False

        # some book keeping
        d0, d1 = details[0], details[1]
        visits0, visits1 = d0.N, d1.N
        visit_delta = visits0 - visits1

        self.most_visited.append(d0.uci)
        self.runner_up.append(d1.uci)
        self.visit_delta.append(visit_delta)
        
        # early stop: if min_delta and most-visited also has best rsc
        if visit_delta >= cfg.min_delta and cfg.use_robust:
            if visits0 >= cfg.min_top_visits:
                if self.should_early_stop(rsc, details):
                    self._es_tripped = True
                    self.es_times.append(_now() - es_time_start)
                    return True
            else:
                self.es_fails['min_top_visits_not_met'] += 1

        else:
            self.es_fails['visit_delta_too_small'] += 1
        
        # next check: visit gap criteria
        curr_budget = self.sim_budget()
        remaining = max(0, curr_budget - sims_done)
        stop_condition = min(remaining, self.target_delta)

        # if stopping condition reached, stop unless we have reason to extend
        stop_triggered = False
        if visit_delta >= stop_condition:
            stop_triggered = True

            # check for extensions only if not maxed
            if curr_budget < cfg.sims_absolute_ceiling:
                if self.should_extend_sims(rsc, details):
                    stop_triggered = False
                    new_ceiling = curr_budget + cfg.es_check_every
                    new_ceiling = min(cfg.sims_absolute_ceiling, new_ceiling)
                    self.set_sim_budget(new_ceiling)
                    self.es_times.append(_now() - es_time_start)
                    return False
                
        # stop here if budget reached and no reason to extend
        if stop_triggered:
            self._es_tripped = True
            self.es_times.append(_now() - es_time_start)
            return True
        else:
            self.extension_reasons['stop_condition_not_met'] += 1
        
        # everything else is a no stop
        self.es_times.append(_now() - es_time_start)
        return False

    def should_early_stop(self, rsc, details):
        # if only one move with > 100 visits...
        if not rsc or len(rsc) < 2:
            self.es_fails['es_granted'] += 1
            return True
            
        d0 = details[0]
        best_rsc_uci = max(rsc, key=rsc.get)
        # best rsc 
        if best_rsc_uci != d0.uci:
            self.es_fails['not_best_rsc'] += 1
            return False
        
        # 15% visit_share
        if d0.visit_share < 0.15:
            self.es_fails['vs_under_15'] += 1
            return False
        
        # make sure rsc delta and dS are sufficient
        vals = sorted(rsc.values(), reverse=True)
        rsc_delta = vals[0] - vals[1]
        if rsc_delta < 0.05:
            self.es_fails['rsc_delta_under_05'] += 1
            return False
        
        dS_dict = {d.uci: d.Qdelta_sign for d in details if d.uci in rsc}
        if d0.Qdelta_sign < 0.0 and len(dS_dict) >= 3:
            worst_dS_uci = sorted(dS_dict, key=dS_dict.get)  # ascending
            if d0.uci in worst_dS_uci[:2]:
                self.es_fails['dS_in_bottom_2'] += 1
                return False
        
        # if any move has better visit_share and dS, it might be able to catch or pass
        vs_dict = {d.uci: d.visit_share for d in details if d.uci in rsc}
        for uci, visit_share in vs_dict.items():
            if uci == d0.uci:
                continue
            if visit_share > d0.visit_share:
                if dS_dict[uci] > d0.Qdelta_sign:
                    self.es_fails['move_with_better_vs_and_dS'] += 1
                    return False
        
        # if here, all checks passed
        self.es_fails['es_granted'] += 1
        return True

    def should_extend_sims(self, rsc, details):
        d0 = details[0]
        if d0.N < self.config.min_top_visits:
            self.extension_reasons['min_top_visits_not_met'] += 1
            return True

        # if the same #2 move has been improving
        if len(self.runner_up) >= 3:
            if len(set(self.runner_up[-3:])) == 1:
                a, b, c = self.visit_delta[-3:]
                if (c < b) and (b < a):
                    self.extension_reasons['runner_up_move_trending'] += 1
                    return True
        
        # if most visited not in top 2 rsc
        top2_uci = sorted(rsc, key=rsc.get, reverse=True)[:2]
        if d0.uci not in top2_uci:
            self.extension_reasons['not_in_top2_rsc'] += 1
            return True
            
        # if low visit share and another move has better dS    
        if d0.visit_share < 0.2:
            for d in details:
                if d.uci not in rsc or d.uci == d0.uci:
                    continue
                
                if d.Qdelta_sign > d0.Qdelta_sign:
                    self.extension_reasons['low_vs_and_not_top_dS'] += 1
                    return True
        
        # dont want dS in bottom 
        d1 = details[1]
        visit_delta = d0.N - d1.N
        if d0.Qdelta_sign < 0.0 and visit_delta < 500:
            dS_dict = {d.uci: d.Qdelta_sign for d in details if d.uci in rsc}
            if len(dS_dict) >= 3:
                worst_dS_uci = sorted(dS_dict, key=dS_dict.get)  # ascending
                worst_k = 2
                if d0.uci in worst_dS_uci[:worst_k]:
                    self.extension_reasons['dS_in_bottom_2'] += 1
                    return False
        
        self.extension_reasons['no_extension'] += 1
        return False

    def stop_simulating(self):
        # do at least 1 sims to stabilize the tree
        if self.sims_completed_this_move < 1:
            return False

        # check for only 1 move
        if len(self.board.legal_moves()) < 2:
            self.sim_stop_reason = "Only 1 legal move"
            return True

        return self.maybe_early_stop()

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

        # early-stop state
        self._es_last_checked_at = 0
        self._es_tripped = False
        self.sim_stop_reason = ""


class ChessGame(object):
    def __init__(self, board, meta, cfg):
        self.game_id = str(uuid.uuid4())
        self.started_at = _now()

        # we may sample adjudicators to vary gameplay
        if cfg.sample_adjudicators:
            cfg = cfg.copy()
            cfg.use_material_diff = bool(np.random.random() > 0.5)
            cfg.use_syzygy = bool(np.random.random() > 0.5)
            cfg.use_eval_draw = bool(np.random.random() > 0.5)
            cfg.use_eval_collar = bool(np.random.random() > 0.5)
            
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
        self.tree = MCTSTree(self.board, self.config)
        self.tree_data = {}
        self.moves_played = []
        self.recents = []
        
        self.mat_adv_counter = 0
        self.outcome = None
        self.plies = 0
        self.sf_eval = None
        
        self.sf_pending = False
        self.sf_ready = False
        self.sf_res_tup = None
        self.tf_last_resolved = 0

        # eval collar and eval draw to shorten selfplay games
        self.collar_stop_set = False
        self.collar_stop_eventual_outcome = None
        if self.config.use_eval_collar:
            self.next_collar_stop_check = cfg.eval_collar_min_plies
        else:
            self.next_collar_stop_check = 999

        # set when to start checking for draw by agreement
        if self.config.use_eval_draw:
            self.next_eval_draw_check = self.config.eval_draw_min_plies
        else:
            self.next_eval_draw_check = 999
    
    def turn(self, return_bool=True):
        stm = self.board.side_to_move()
        return stm == 'w' if return_bool else stm
    
    def is_stockfish_turn(self):
        return self.vs_stockfish and (self.stockfish_is_white == self.turn())
    
    def push_move(self, mv, method, xc0_move):
        # collect search data then push and update
        self.collect_tree_search_data(mv, method, xc0_move)

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
        rsc, details = self.tree.robust_selection_criteria(5, 100)
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
        
        # IMPORTANT, this MUST happen before the move is pushed, otherwise the values change
        vwq = rnd(self.tree.visit_weighted_Q(), 4)
        best_q = rnd(details[0].Q, 4)
        Q_white = best_q
        Q_stm = Q_white if turn else -Q_white
        if self.is_stockfish_turn():
            Q_stm = self.sf_eval
            Q_white = Q_stm if turn else -Q_stm

        data["visit_weighted_Q"] = vwq
        data['best_Q'] = best_q
        data['Q_stm'] = Q_stm
        data['Q_white'] = Q_white

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
                "Qema": rnd(cd.Qema, 4),"Qdelta_sign": rnd(cd.Qdelta_sign, 4),
                "is_terminal": cd.is_terminal, "rcs": rnd(rsc.get(cd.uci, 0.0), 4)
            }
            
            candidate_moves.append(cm)
        data["candidate_moves"] = candidate_moves

        # fast PV via C++
        pv = []
        pv_items = self.tree.principal_variation(24)
        for x in pv_items:
            pv.append({
                "uci": x.uci, "visits": int(x.visits),
                "P": rnd(x.P, 4), "Q": rnd(x.Q, 4)
            })
        
        # attach PV snapshot (may be empty if no deeper visited chain exists)
        data["pv"] = pv
        self.tree_data[self.plies] = data
    
    def make_move_with_stockfish(self, eng):
        """ Stockfish plays one move. Record depth if not using depth limit """
        legal = self.board.legal_moves()
        if not legal:
            return self.check_for_terminal()

        # get SF move + signed eval (white POV)
        res_tup = cbu.sf_eval(
            self.python_chess_board, score_fn=score_to_value_stm_pov,
            depth=self.config.sf_depth, engine=eng
        )

        if len(res_tup) == 2:
            sf_v, best_move = res_tup
        else:
            sf_v, best_move, searched = res_tup
            self.sf_search_depth.append(searched)

        self.sf_eval = sf_v
        return self.push_move(best_move, "stockfish", None)
    
    def set_stockfish_result(self, res_tup):
        self.sf_res_tup = res_tup
        self.sf_ready = True

    def apply_stockfish_result(self, res_tup):
        if len(res_tup) == 2:
            sf_v, best_move = res_tup
        else:
            sf_v, best_move, searched = res_tup
            self.sf_search_depth.append(searched)

        self.sf_eval = sf_v
        self.sf_res_tup = None
        self.sf_ready = False
        self.sf_pending = False
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
        piece_part = self.board.fen().split(" ")[0]
        if "q" in piece_part.lower():
            return False
        
        # if here, agree to draw
        else:
            return True
    
    def check_for_collar_stop(self, cfg):
        n_last = cfg.eval_collar_span
        thresh = cfg.eval_collar_thresh

        # first-time detection of a collar stop candidate
        if not self.collar_stop_set:
            if self.plies < self.next_collar_stop_check:
                return False, None

            # scan newest->oldest. rev_i 0 == newest
            sign_check = None
            for rev_i, ex in enumerate(reversed(self.recents[-n_last:])):
                # ex is a tuple: (mv, Q_stm, Q_white, best_q, turn)
                best_q = ex[3]
                # magnitude must meet the lock threshold
                if abs(best_q) < thresh:
                    needed = n_last - rev_i
                    self.next_collar_stop_check = self.plies + needed
                    return False, None

                # use white-POV signed value here
                q_white = ex[2]
                sign = 1*(q_white > 0) - 1*(q_white < 0)

                if sign_check is None:
                    sign_check = sign
                elif sign != sign_check:
                    # mixed signs -> wait until this newest violator is evicted
                    needed = n_last - rev_i
                    self.next_collar_stop_check = self.plies + needed
                    return False, None

            # all checks passed. set collar stop
            self.collar_stop_set = True
            self.collar_stop_eventual_outcome = sign_check  # -1 or +1
            # keep trigger as a positive magnitude for release checks
            self.collar_stop_trigger = cfg.eval_collar_trigger
            return False, None

        # already locked, check to see if max game length is approaching
        elif self.plies >= cfg.max_game_length:
            return True, self.collar_stop_eventual_outcome
        
        # otherwise check to see if a blunder has lowered the score
        else:
            check_depth = 3
            tail = self.recents[-check_depth:]

            for ex in tail:
                # ex is a tuple: (mv, Q_stm, Q_white, best_q, turn)
                q_white = ex[2]
                # positive when leader still ahead
                sign_stability = self.collar_stop_eventual_outcome * q_white

                # if sign_stability falls below the trigger, we stop and award win
                if sign_stability < self.collar_stop_trigger:
                    return True, self.collar_stop_eventual_outcome
        
        # no stop detected
        return False, None

    def check_for_terminal(self):
        cfg = self.config
        # check to see if a collar stop has been set
        if cfg.use_eval_collar:
            if self.plies >= cfg.eval_collar_min_plies:
                collar_stop, outcome = self.check_for_collar_stop(cfg)
                if collar_stop:
                    self.outcome = outcome
                    return True
        
        # otherwise look for conventional terminal states
        reason, result = self.board.is_game_over()
        if reason != 'none':
            self.outcome = terminal_value_white_pov(self.board)
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
                return True
        
        # Syzygy probe if few pieces
        # flip syzygy on if about to end due to length
        if cfg.use_syzygy or (self.plies >= cfg.max_game_length):
            if self.board.piece_count() <= 5:
                # may not work so just go as normal
                try:
                    outcomes = {-2: -1, -1:-1, 0:0, 1:1, 2:1}
                    with chess.syzygy.open_tablebase(ENDGAME_LOC) as tablebase:
                        # gotta flip back to python chess here
                        chess_board = chess.Board(self.board.fen())
                        table_res = tablebase.probe_wdl(chess_board)

                    # -1 and 1 are not guaranteed winners
                    if table_res in [-2, 0, 2]:
                        table_res = table_res if chess_board.turn else -1*table_res
                        self.outcome = outcomes[table_res]
                        return True
                except:
                    pass

        # check for draws based on eval or move limit
        if self.plies >= self.next_eval_draw_check:
            if self.check_for_eval_draw(cfg):
                self.outcome = 0.0
                return True

        # check for overal game_length limit
        if self.plies > cfg.max_game_length:
            self.outcome = 0.0
            if self.collar_stop_set:
                self.outcome = self.collar_stop_eventual_outcome
            return True
        # if we made it here the game is active
        return False
