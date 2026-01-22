import uuid
from time import time as _now

import chess, chess.syzygy
import tl2cgen as tl2
import numpy as np

from pyfastchess import MCTSTree as fasttree
from pyfastchess import create_prior_engine, configure_prior_engine, prior_engine_build
from pyfastchess import terminal_value_white_pov

from chessbot import ENDGAME_LOC
from chessbot.review import score_to_value_stm_pov, make_fake_visits
from chessbot.utils import calc_entropy, rnd
import chessbot.utils as cbu


class MCTSTree(fasttree):
    def __init__(self, board, cfg):
        self.config = cfg

        # if c_puct is given as a list, pick an option randomly
        # otherwise assume its a float or int        
        c = cfg.c_puct
        if isinstance(c, (int, float)):
            self.c_puct = float(c)
        elif isinstance(c, (list, set)):
            self.c_puct = float(np.random.choice(c))
        else:
            self.c_puct = c
        
        self.ema_span = cfg.ema_span

        # set floor and ceiling 
        self.sims_floor = cfg.sims_floor
        self.sims_ceiling_schedule = {}
        #ceiling can be a list for varied gameplay
        if isinstance(cfg.sims_ceiling, (list, set)):
            self.sims_ceiling = np.random.choice(cfg.sims_ceiling)
        # can also be a dict, interpreted as a sims schedule
        elif isinstance(cfg.sims_ceiling, dict):
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
        super().__init__(
            board, self.c_puct, self.ema_span,
            self.sims_ceiling, self.pruning_factor
        )

        # bookkeeping
        self.board = board
        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()
        #self.piece_count = board.piece_count()

        self._move_started_at = _now()
        self.sims_completed_this_move = 0

        self.n_moves_played = 0
        self.sims_done_total = 0
        
        # root noise
        self.add_root_noise = cfg.add_root_noise
        self.root_noise_added = False

        # early-stop rolling state
        self._es_last_checked_at = 0
        self._es_tripped = False
        self.sim_stop_reason = ""
        
        self.sim_decision_model = None
        if self.config.use_sim_decision_model:
            self.sim_decision_model = tl2.Predictor(
                self.config.sim_decision_model_path, nthread=1
            )

    def best(self):
        """
        Before ply 20: sample from the top 5 moves by visits using a
        temperature schedule that decays to near-deterministic by ply 20.
        At/after ply 20: delegate to the base implementation.
        Returns (uci, None) like the original.
        """
        if self.config.sample_moves == False:
            return super().best()

        # if at/after the convergence ply, just use C++/base behavior
        if (self.n_plies >= 20) or (self.board.piece_count() <= 20):
            return super().best()

        # gather root visits (desc sorted list of (uci, N))
        rows = self.root_child_visits()
        if not rows:
            return super().best()

        ucis = [u for u, _ in rows]
        visits = np.array([n for _, n in rows], dtype=np.float64)

        # trivial cases
        if len(ucis) == 1 or visits.sum() <= 0.0:
            return ucis[0], None

        # force sampling only from the top 5 moves
        top_k = min(5, len(ucis))
        top_ucis = ucis[:top_k]
        top_visits = visits[:top_k]

        # temperature schedule: linear decay from temp_max (ply 0) to
        # temp_min (ply 20). Small temp_min makes softmax -> argmax.
        temp_min = self.config.move_sample_temp_range[0]
        temp_max = self.config.move_sample_temp_range[1]
        
        frac = max(0.0, min(1.0, (20.0 - self.n_plies) / 20.0))
        temp = temp_min + (temp_max - temp_min) * frac

        # build stable logits from visits: use log(visits) so scale is sane
        logits = np.log(top_visits + 1e-12) / max(1e-12, temp)
        logits = logits - np.max(logits)
        exps = np.exp(logits)   
        probs = exps / exps.sum()

        idx = np.random.choice(len(top_ucis), p=probs)
        return top_ucis[idx], None

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

        # check the sims schedule, update if applicapable
        if self.sims_ceiling_schedule:
            if self.n_plies in self.sims_ceiling_schedule.keys():
                new_ceiling = self.sims_ceiling_schedule[self.n_plies]
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
            eps=self.config.dirichlet_eps, alpha=self.config.dirichlet_alpha
        )
        self.root_noise_added = True

    def get_sim_decision_probs(self):
        if self.sim_decision_model is None:
            return None
        
        vec = []
        root = self.root()
        deets = self.root_child_details()
        rows = self.root_child_visits()

        # n_cands, top share, visit_margin, visit norm
        vec.append(len(root.legal_moves))
        visits = [v[1] for v in rows]

        # not sure how this happens, but return None
        if len(visits) < 2:
            print("len(visists) < 2 somehow")
            return None
        
        vec.append(visits[0] / max(1.0, root.N))
        vec.append(visits[0]-visits[1])

        _, norm_vis = calc_entropy(visits)
        vec.append(norm_vis)

        # P norm,  Q norm, # U norm
        c_puct = self.c_puct
        S = root.N
        Ps, Qs, Us = [], [], []
        mult = 1 if root.board.side_to_move() == 'w' else -1
        for c in deets:
            Ps.append(c.prior)
            Qs.append(mult * c.Q)
            Us.append(c_puct * c.prior * (S ** 0.5) / (1 + c.N))

        # prefer selected Q to be the best Q at stop
        # i.e. do not stop unless the top move is best Q, independent of everything else
        q_margin = Qs[0] - max(Qs)
        if self.config.prefer_top_q:
            if q_margin < 0:
                return np.array([1.0, 0.0])

        _, norm_p = calc_entropy(Ps)
        _, norm_q = calc_entropy(Qs)
        _, norm_u = calc_entropy(Us)

        vec.append(norm_p)
        vec.append(norm_q)
        vec.append(norm_u)

        # top_p, q_margin, q_delta
        vec.append(Ps[0])
        vec.append(Qs[0] - Qs[1])
        vec.append(q_margin)

        # is_middlegame, is endgame
        vec.append(1*((self.n_plies < 20) and (self.piece_count > 12)))
        vec.append(1*((self.n_plies > 60) or (self.piece_count < 12)))

        dmat = tl2.DMatrix(np.array(vec, dtype=np.float32, ndmin=2))
        probs = self.sim_decision_model.predict(dmat)
        return probs.ravel()

    def maybe_early_stop(self):
        if self._es_tripped:
            return True
                    
        sims_done = self.sims_completed_this_move
        if sims_done < self.config.sims_floor:
            return False

        if sims_done >= self.sims_ceiling:
            self.sim_stop_reason = f"Sim ceiling {self.sims_ceiling} reached"
            return True

        if sims_done - self._es_last_checked_at < self.config.es_check_every:
            return False

        # if here, determine visit delta and test for early stop
        self._es_last_checked_at = sims_done
        rows = self.root_child_visits()
        visits = [v[1] for v in rows]
        visit_delta = visits[0]-visits[1]
        if visit_delta >= self.target_delta:
            self.sim_stop_reason = f"Visit delta {visit_delta} >= {self.target_delta}"
            return True
        return False
        
        # # if not using the dec model, just hit the target
        # if not self.config.use_sim_decision_model:
        #     return sims_done >= sims_target

        # if sims_done < self.config.sims_floor:
        #     return False

        # if sims_done - self._es_last_checked_at < self.config.es_check_every:
        #     return False

        # # if here, run ES check
        # self._es_last_checked_at = sims_done
        # probs = self.get_sim_decision_probs()
        # if probs is None:
        #     return False

        # # best move prob
        # bmp = np.ravel(probs)[-1]
        # string = f"best move prob {bmp:.3f} sims_done {sims_done}"
        # # need to be above the es (early stop) threshold to stop here
        # if sims_done >= self.sims_ceiling:
        #     self.sim_stop_reason = f"Sim Limit reached: {string}"
        #     return True

        # if sims_done < sims_target:
        #     if bmp > self.config.es_best_move_threshold:
        #         self._es_tripped = True
        #         self.sim_stop_reason = f"ES triggered: {string}"
        #         return True

        # if sims_done >= sims_target:
        #     # need to be above the bs (bonus sims) threshold to stop here
        #     if bmp > self.config.bs_best_move_threshold:
        #         self.sim_stop_reason = f"Sufficient: {string}"
        #         return True
        
        # otherwise keep searching
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
    
    def push_move(self, mv):
        # collect search data then push and update
        self.collect_tree_search_data(mv)

        # advance tree (pushes move) and reset
        self.tree.advance(self.board, mv)
        self.tree.reset_for_new_move()
        self.moves_played.append(mv)
        self.plies += 1
        return self.check_for_terminal()
    
    def collect_tree_search_data(self, mv):
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
        details = self.tree.root_child_details()
        if details is None:
            return
        
        sims = int(self.tree.sims_completed_this_move)
        total_children = len(details)
        visited_children = sum([1 for cd in details if cd.N > 0])

        turn = self.turn() # STM
        data = {
            "sims": sims, "time": rnd(elapsed, 3),
            "avg_depth": rnd(avg_depth, 2), "max_depth": max_depth,
            "children_visited": visited_children,
            "total_children": total_children,
            "stop_reason": self.tree.sim_stop_reason, "stm": turn
        }
        
        # IMPORTANT, this MUST happen before the move is pushed, otherwise the values change
        vwq = rnd(self.tree.visit_weighted_Q(), 4)
        Q_white = vwq
        Q_stm = Q_white if turn else -Q_white
        if self.is_stockfish_turn():
            Q_stm = self.sf_eval
            Q_white = Q_stm if turn else -Q_stm

        data["visit_weighted_Q"] = vwq
        data['Q_stm'] = Q_stm
        data['Q_white'] = Q_white

        # keep a small list of items for gameplay checking
        self.recents.append((mv, Q_stm, Q_white, vwq, turn))

        # sumN for U term
        sumN = max(1, root.N)

        candidate_moves = []
        for cd in details:
            U = c_puct * cd.prior * (sumN ** 0.5) / (1 + cd.N)
            cm = {
                "uci": cd.uci, "visits": cd.N,
                "P": rnd(cd.prior, 4), "Q": rnd(cd.Q, 4), "U": rnd(U, 4),
                "Q_ema": rnd(cd.Q_ema, 4),
                "is_terminal": cd.is_terminal
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
        tl = 0.25  # time limit
        res_tup = cbu.sf_eval(
            self.board, score_fn=score_to_value_stm_pov,
            depth=self.config.sf_depth, time_lim=tl, engine=eng
        )

        if len(res_tup) == 2:
            sf_v, best_move = res_tup
        else:
            sf_v, best_move, searched = res_tup
            self.sf_search_depth.append(searched)

        self.sf_eval = sf_v
        return self.push_move(best_move)

    def make_move_from_tree(self):
        """ Play best-by-visits  """
        root = self.tree.root()
        if not root.is_expanded:
            return False

        mv, _ = self.tree.best()
        if mv is None:
            return False
        
        return self.push_move(mv)

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
            # r is a tuple: (mv, Q_stm, Q_white, vwq, turn)
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
        # prefer eval check above first to push next check back
        elif 'q' in self.board.fen().lower():
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
                # ex is a tuple: (mv, Q_stm, Q_white, vwq, turn)
                vwq = ex[3]
                # magnitude must meet the lock threshold
                if abs(vwq) < thresh:
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

        # already locked: check for release (blunder) in a short tail
        else:
            check_depth = 3
            tail = self.recents[-check_depth:]

            for ex in tail:
                # ex is a tuple: (mv, Q_stm, Q_white, vwq, turn)
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
        if cfg.use_syzygy:
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
