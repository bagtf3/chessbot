from time import time as _now
from chessbot.psqt import build_weights
from pyfastchess import MCTSTree as fasttree
from pyfastchess import PriorConfig, PriorEngine, Evaluator
from collections import OrderedDict


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
        self.sims_target = None

        self._move_started_at = _now()
        self.sims_completed_this_move = 0
        self.awaiting_predictions = []

        # early-stop rolling state
        self._es_history = []
        self._es_last_checked_at = 0
        self._es_tripped = False
        self._es_reason = ""
        self._es_after_sims = 0
    
    def configure_prior_engine(self):
        cfg = self.config
    
        def _get(name, default=None):
            if hasattr(cfg, name):
                return getattr(cfg, name)
            return default
    
        # top-level mixes / flags
        pcfg = PriorConfig()
        pcfg.anytime_uniform_mix = float(_get("anytime_uniform_mix", 0.15))
        pcfg.endgame_uniform_mix = float(_get("endgame_uniform_mix", 0.25))
        pcfg.opponent_uniform_mix = float(_get("opponent_uniform_mix", 0.5))
        pcfg.use_prior_boosts = bool(_get("use_prior_boosts", True))
    
        # prior adjustments (these are expected to be dict-like)
        anytime_adj = _get("anytime_prior_adjustments", {}) or {}
        eg_adj = _get("endgame_prior_adjustments", {}) or {}
    
        pcfg.anytime_gives_check = float(anytime_adj.get("gives_check", 0.0))
        pcfg.anytime_repetition_sub = float(anytime_adj.get("repetition_penalty", 0.0))
        pcfg.endgame_pawn_push = float(eg_adj.get("pawn_push", 0.0))
        pcfg.endgame_capture = float(eg_adj.get("capture", 0.0))
        pcfg.endgame_repetition_sub = float(eg_adj.get("repetition_penalty", 0.0))
    
        # optional clipping overrides (if present on cfg)
        if _get("prior_clip_min", None) is not None:
            pcfg.clip_min = float(_get("prior_clip_min"))
        if _get("prior_clip_max", None) is not None:
            pcfg.clip_max = float(_get("prior_clip_max"))
    
        # attach both the config and engine to the instance for later inspection
        self._prior_cfg = pcfg
        pcfg.clip_enabled = True 
        self._prior_engine = PriorEngine(pcfg)
        return pcfg

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
        # if lru is not None:
        #     cached = lru.get(cache_key, None, touch=True)
        #     lru.bonus_queries += 1
        #     if cached is not None:
        #         lru.bonus_hits += 1
        #         self.sims_completed_this_move += 1
        #         req['already_applied'] = True
        #         self.apply_cached(req, cached)
        #         return req
        
        # # if not terminal or cached, send standard request
        # req['enc'] = leaf.board.stacked_planes(5)
        # self.awaiting_predictions.append(req)

        # need to increase sims count here since were skipping everyhing else
        self.sims_completed_this_move += 1
        return req

    def resolve_awaiting(self, cache):
        if not self.awaiting_predictions:
            return 0

        keep = []
        for req in self.awaiting_predictions:
            cached = cache[req["cache_key"]]
            # this shouldnt be possible
            if cached is None:
                keep.append(req)
                continue

            self.apply_cached(req, cached)
            self.sims_completed_this_move += 1

        self.awaiting_predictions = keep
        return len(self.awaiting_predictions)

    def apply_cached(self, req, cached):
        """
        Compute priors and delegate to C++ apply_result.
        'cached' must have keys: value, from, to, piece, promo (factorized heads).
        """
        leaf = req["leaf"]
        legal = leaf.board.legal_moves()

        # retrieve factorized softmaxes
        p_from  = cached["from"]
        p_to    = cached["to"]
        p_piece = cached["piece"]
        p_promo = cached["promo"]
    
        # let the C++ PriorEngine handle mixing, boosts, clipping and renorm
        pri = self._prior_engine.build(
            leaf.board, legal, p_from, p_to, p_piece, p_promo,
            self.root_stm, req["stm_leaf"]
        )
    
        # expansion + backup (C++ apply_result expects pri as (move, prob) pairs)
        self.apply_result(leaf, pri, cached["value"])

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
