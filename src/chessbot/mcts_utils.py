from time import time as _now
from chessbot.psqt import build_weights
from pyfastchess import Evaluator, MCTSTree as fasttree
from pyfastchess import create_prior_engine, configure_prior_engine, prior_engine_build
from collections import OrderedDict
from chessbot.utils import calc_entropy
import tl2cgen as tl2
import numpy as np


class MCTSTree(fasttree):
    ev = Evaluator()
    ev.configure(build_weights(zeros=False))

    def __init__(self, board, cfg):
        self.config = cfg
        self.c_puct = float(cfg.c_puct)
        super().__init__(board, self.c_puct, MCTSTree.ev)

        # bookkeeping
        self.board = board
        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()
        self.piece_count = board.piece_count()

        self._move_started_at = _now()
        self.sims_completed_this_move = 0

        self.moves_played = 0
        self.sims_done_total = 0

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
        # If not configured, delegate straight to the C++/base implementation.
        if not self.config.use_q_override:
            return super().best()
    
        details = self.root_child_details()
    
        # build candidate list
        cands = []
        white_to_move = self.root().board.side_to_move() == 'w'
        mult = 1 if white_to_move else -1
        for d in details:
            cands.append({"uci": d.uci, "visits": d.N, "Q": mult * d.Q, "P": d.prior})
    
        # sort by visits descending
        c_sorted = sorted(cands, key=lambda x: x["visits"], reverse=True)
        top = c_sorted[0]
        top_vis = top["visits"]
        top_q = top["Q"]
    
        # read thresholds from Config
        vis_ratio = self.config.q_override_vis_ratio
        q_margin = self.config.q_override_q_margin
        min_vis_cfg = self.config.q_override_min_vis
        top_k = self.config.q_override_top_k
    
        # Compute absolute minimum visits required
        vis_min = max(min_vis_cfg, top_vis * vis_ratio)
    
        # Eligible among top_k
        eligible = [c for c in c_sorted[:top_k] if c["visits"] >= vis_min]
    
        if not eligible:
            return top["uci"], None
    
        # Pick the eligible one with highest Q
        best_q_c = max(eligible, key=lambda x: x["Q"])
    
        if best_q_c["Q"] >= top_q + q_margin and best_q_c["uci"] != top["uci"]:
            
            print(f"[choose move] Playing non-top-Q  white to move: {white_to_move}")
            print(f"[choose move] {best_q_c} vs {top}")
            return best_q_c["uci"], None
        else:
            return top["uci"], None

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
        if self.config.add_root_noise:
            self.add_root_dirichlet_noise(
                eps=self.config.dirichlet_eps,
                alpha=self.config.dirichlet_alpha
            )

        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()
        self.piece_count = board.piece_count()

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
        sims_target = self.config.sims_target

        # if not using the dec model, just hit the target
        if not self.config.use_sim_decision_model:
            return sims_done >= sims_target

        if sims_done < self.config.sims_floor:
            return False

        if sims_done - self._es_last_checked_at < self.config.es_check_every:
            return False

        # if here, run ES check
        self._es_last_checked_at = sims_done
        probs = self.get_sim_decision_probs()
        if probs is None:
            return False

        # best move prob
        bmp = np.ravel(probs)[-1]
        string = f"best move prob {bmp:.3f} sims_done {sims_done}"
        # need to be above the es (early stop) threshold to stop here
        if sims_done >= self.config.sims_ceiling:
            self.sim_stop_reason = f"Sim Limit reached: {string}"
            return True

        if sims_done < sims_target:
            if bmp > self.config.es_best_move_threshold:
                self._es_tripped = True
                self.sim_stop_reason = f"ES triggered: {string}"
                return True

        if sims_done >= sims_target:
            # need to be above the bs (bonus sims) threshold to stop here
            if bmp > self.config.bs_best_move_threshold:
                self.sim_stop_reason = f"Sufficient: {string}"
                return True
        
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

        self.moves_played += 1
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

