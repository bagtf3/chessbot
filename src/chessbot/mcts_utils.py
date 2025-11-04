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
        self.configure_prior_engine()

        # bookkeeping
        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()
        self.piece_count = board.piece_count()

        self._move_started_at = _now()
        self.sims_completed_this_move = 0
        self.awaiting_predictions = []

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

    def configure_prior_engine(self):
        """Creates and configures singleton prior engine in c++ """
        create_prior_engine()
        conf_dict = self.config.to_dict()
        configure_prior_engine(conf_dict)

    def collect_one_leaf(self, lru=None):
        """
        Walk PUCT to a leaf (in C++) and return an inference request dict,
        or a terminal-hit dict that should NOT be batched.
        """
        leaf = super().collect_one_leaf()
        cache_key = leaf.board.hash()

        req = {
            "leaf": leaf, "cache_key": cache_key,
            "terminal": False, "already_applied": False
        }
    
        # terminal handled in C++: count it here
        if leaf.is_terminal:
            self.sims_completed_this_move += 1
            req['terminal'] = True
            req['already_applied'] = True
            return req

        # if not terminal, will need this
        req["stm_leaf"] = leaf.board.side_to_move()

        # check the LRU cache
        if lru is not None:
            cached = lru.get(cache_key, None, touch=True)
            lru.bonus_queries += 1
            if cached is not None:
                lru.bonus_hits += 1
                self.sims_completed_this_move += 1
                req['already_applied'] = True
                self.apply_cached(req, cached)
                return req
        
        # if not terminal or cached, send standard request
        req['enc'] = leaf.board.stacked_planes(5)
        self.awaiting_predictions.append(req)
        return req

    def resolve_awaiting(self, cache):
        if not self.awaiting_predictions:
            return 0

        keep = []
        for req in self.awaiting_predictions:
            cached = cache[req["cache_key"]]
            self.apply_cached(req, cached)
            self.sims_completed_this_move += 1

        self.awaiting_predictions = keep
        return len(self.awaiting_predictions)

    def apply_cached(self, req, cached):
        """
        Compute priors and delegate to C++ apply_result.
        'cached' must have keys: value, from, to, piece, promo (factorized heads).
        """

        # retrieve factorized softmaxes
        p_from  = cached["from"]
        p_to    = cached["to"]
        p_piece = cached["piece"]
        p_promo = cached["promo"]
    
        # let the C++ PriorEngine handle mixing, boosts, clipping and renorm
        pri = prior_engine_build(
            leaf.board, leaf.legal_moves, p_from, p_to, p_piece, p_promo
        )
    
        # expansion + backup (C++ apply_result expects pri as (move, prob) pairs)
        self.apply_result(leaf, pri, cached['value'])

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

        # Invalidate any queued leaves/tokens from prior epoch
        self.awaiting_predictions.clear()

        # Keep external board & counters in sync for your caller's logic
        board.push_uci(move_uci)
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
        # check for only 1 move
        if len(self.root().legal_moves) < 2:
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
        self.awaiting_predictions.clear()
        self._move_started_at = _now()

        # early-stop state
        self._es_last_checked_at = 0
        self._es_tripped = False
        self.sim_stop_reason = ""


class LRUCache:
    """
    Minimal LRU cache backed by collections.OrderedDict.
    """

    def __init__(self, maxsize=100_000):
        self.maxsize = int(maxsize)
        self._od = OrderedDict()
        self.bonus_queries = 0
        self.bonus_hits = 0
        self.evictions = 0

    def __len__(self):
        return len(self._od)

    def __contains__(self, key):
        return key in self._od

    def put(self, key, value, touch=False):
        """Insert or update value and mark key as most-recent. Evict if needed."""
        self._od[key] = value
        # eviction if over capacity
        if touch:
            self._od.move_to_end(key, last=True)
        
        if len(self._od) > self.maxsize:
            # pop least-recent (first item)
            self._od.popitem(last=False)
            self.evictions += 1

    def get(self, key, default=None, touch=False):
        if key not in self._od:
            return default
        if touch:
            self._od.move_to_end(key, last=True)
        return self._od[key]

    def pop(self, key, default=None):
        return self._od.pop(key, default)

    def touch(self, key):
        """Move entry to the most recent side"""
        if key in self._od:
            # replace value and mark as most-recent
            self._od.move_to_end(key, last=True)
    
    def clear(self):
        """Empty the cache."""
        self._od.clear()
        self.bonus_hits = 0
        self.bonus_queries = 0
        self.evictions = 0

    def keys(self):
        """Return keys from least-recent to most-recent."""
        return self._od.keys()

    def items(self):
        """Return (key, value) pairs from least-recent to most-recent."""
        return self._od.items()

    def most_recent_items(self, n):
        """Iterate up to `n` most-recent items (most-recent first)."""
        if n <= 0:
            return iter([])
        # OrderedDict is from least->most; take last n and reverse
        it = list(self._od.items())[-n:][::-1]
        return iter(it)

    def stats(self):
        print("LRUCache Stats")
        print(f"size={len(self._od)}/{self.maxsize}, evictions={self.evictions}")
        bh = self.bonus_hits
        bq = self.bonus_queries
        bhr = (bh / float(bq)) if (bq != 0) else 0.0
        print(f"bonus_hits={bh}, bonus_queries={bq}, bonus_hit_rate={bhr:.3f}")
        print("="*60)

    # dict-style conveniences
    def __setitem__(self, key, value):
        self.put(key, value)

    def __getitem__(self, key):
        return self._od[key]

    def __repr__(self):
        return "LRUCache(maxsize={}, len={}, evictions={})".format(
            self.maxsize, len(self), self.evictions)
