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

        # bookkeeping mirroring old interface
        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()
        self.piece_count = board.piece_count()
        self.sims_target = None

        self._move_started_at = _now()
        self.sims_completed_this_move = 0
        self.awaiting_predictions = []

        # early-stop rolling state
        self._es_history = []
        self._es_last_checked_at = 0
        self._es_tripped = False
        self.sim_stop_reason = ""

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
        for d in details:
            cands.append({"uci": d.uci, "visits": d.N, "Q": d.Q, "P": d.prior})
    
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
        vis_min = max(min_vis_cfg, int(top_vis * vis_ratio))
    
        # Eligible among top_k
        eligible = [c for c in c_sorted[:top_k] if c["visits"] >= vis_min]
    
        if not eligible:
            return top["uci"], None
    
        # Pick the eligible one with highest Q
        best_q_c = max(eligible, key=lambda x: x["Q"])
    
        if best_q_c["Q"] >= top_q + q_margin and best_q_c["uci"] != top["uci"]:
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

        # n_cands
        vec.append(len(root.legal_moves))

        # top share, visit_margin, visit norm
        visits = [v[1] for v in rows]
        vec.append(visits[0]/root.N)
        vec.append(visits[0]-visits[1])

        ent_vis, norm_vis = calc_entropy(visits)
        vec.append(norm_vis)

        # P norm
        priors = [c.prior for c in deets]
        ent_p, norm_p = calc_entropy(priors)
        vec.append(norm_p)

        # Q norm
        qs = [c.Q for c in deets]
        ent_q, norm_q = calc_entropy(qs)
        vec.append(norm_q)

        # U norm
        cp = self.c_puct
        S = root.N
        Us = [cp * cd.prior * (S ** 0.5) / (1 + cd.N) for cd in deets]
        ent_u, norm_u = calc_entropy(Us)
        vec.append(norm_u)

        # top_p, q_margin, q_delta
        vec.append(deets[0].prior)
        vec.append(qs[0] - qs[1])
        vec.append(qs[0] - max(qs))

        # is_middlegame, is endgame
        vec.append(1*((self.n_plies < 20) and (self.piece_count < 12)))
        vec.append(1*((self.n_plies > 60) or (self.piece_count < 12)))

        dmat = tl2.DMatrix(np.array(vec, dtype=np.float32, ndmin=2))
        probs = self.sim_decision_model.predict(dmat)
        return probs[0][0]

    def maybe_early_stop(self):
        if self._es_tripped:
            return True
        
        sims_done = self.sims_completed_this_move
        if sims_done - self._es_last_checked_at < self.config.es_check_every:
            return False

        # immediate checks that apply regardless of es_min_sims
        sims_target = self.sims_target
        rows = self.root_child_visits()
        if rows:
            # if the top node already has >= es_top_node_frac * sims_target, stop
            top_vis = rows[0][1]
            if top_vis >= self.config.es_top_node_frac * sims_target:
                self._es_tripped = True
                self._es_after_sims = sims_done
                thresh = self.config.es_top_node_frac * sims_target
                self._es_reason = f"top_node_frac top_vis={top_vis} thresh={thresh:.1f}"
                return True
    
        # from here on enforce min sims and periodic checking as before
        if sims_done < self.config.es_min_sims:
            return False
        
        self._es_last_checked_at = sims_done
        rows = self.root_child_visits()
        if len(rows) < 2:
            return False
    
        n1, n2 = rows[0][1], rows[1][1]
        gap = n1 - n2
        remaining = max(0, sims_target - sims_done)
    
        if gap > self.config.es_gap_frac * remaining:
            self._es_tripped = True
            self._es_reason = (
                f"gap_vs_remaining n1={n1} n2={n2} gap={gap} "
                f"remaining={remaining} thresh={self.config.es_gap_frac * remaining:.1f}"
            )
            self._es_after_sims = sims_done
            return True
        return False

    def stop_simulating(self):
        if self.sims_target is None:
            b = self.root().board
            mvs = b.legal_moves()
            # if there is only 1 move, make it
            if len(mvs) == 1:
                # set this ultra low just incase
                self.sims_target = 1
                return True
            
            # use the sims target, or 600*num_moves to speed up forced positions
            self.sims_target = min(self.config.sims_target, 600*len(mvs))
        if self.sims_completed_this_move >= self.sims_target:
            return True
        return self.maybe_early_stop()

    def reset_for_new_move(self):
        """
        Resets everything, counts visits from previous trees.
        """
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
        self.sims_target = None

        # early-stop state
        self._es_history.clear()
        self._es_last_checked_at = 0
        self._es_tripped = False
        self._es_reason = ""
        self._es_after_sims = 0


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
