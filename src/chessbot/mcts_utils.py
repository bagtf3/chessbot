from time import time as _now
from chessbot.utils import softmax
from pyfastchess import MCTSTree as fasttree
from pyfastchess import PriorConfig, PriorEngine


class MCTSTree(fasttree):
    def __init__(self, board, cfg):
        self.config = cfg
        self.c_puct = float(cfg.c_puct)
        super().__init__(board, self.c_puct)
        self.configure_prior_engine()

        # bookkeeping mirroring old interface
        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()

        self._move_started_at = _now()
        self.sims_completed_this_move = 0
        self.awaiting_predictions = []

        # unique-sims tracking (per move)
        self._uniq_keys = set()
        self.unique_sims_this_move = 0

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

    def collect_one_leaf(self, board):
        """
        Walk PUCT to a leaf (in C++) and return an inference request dict,
        or a terminal-hit dict that should NOT be batched.
        """
        leaf = super().collect_one_leaf()
    
        # cheap key: (zobrist, last1 UCI)
        tail = leaf.board.history_uci()
        last1 = tail[-1] if tail else ""
        cache_key = (leaf.board.hash(), last1)
    
        # terminal handled in C++: count it here if you track stats, don't batch
        if leaf.is_terminal:
            self.sims_completed_this_move += 1
            return {
                "leaf": leaf,
                "cache_key": cache_key,
                "terminal": True,
                "already_applied": True,
                "terminal_value": leaf.value,
            }
    
        req = {
            "leaf": leaf,
            "cache_key": cache_key,
            "terminal": False,
            "already_applied": False,
            "stm_leaf": leaf.board.side_to_move(),
            "enc": leaf.board.stacked_planes(5),
        }
        self.awaiting_predictions.append(req)
        return req


    def resolve_awaiting(self, board, quick_cache):
        if not self.awaiting_predictions:
            return 0

        keep = []
        for req in self.awaiting_predictions:
            cached = quick_cache.get(req["cache_key"])
            if cached is None:
                keep.append(req)
                continue

            self.apply_cached(req, cached)
            self.sims_completed_this_move += 1

            # bump uniques on first application of this key
            key = req["cache_key"]
            if key not in self._uniq_keys:
                self._uniq_keys.add(key)
                self.unique_sims_this_move += 1

        self.awaiting_predictions = keep
        return len(self.awaiting_predictions)

    def apply_cached(self, req, cached):
        """
        Compute priors (softmax in Python) and delegate to C++ apply_result.
        'cached' must have keys: value, from, to, piece, promo (factorized heads).
        """
        leaf = req["leaf"]
        legal = leaf.board.legal_moves()
    
        # compute factorized softmaxes in python (numpy)
        sf_from  = softmax(cached["from"])
        sf_to    = softmax(cached["to"])
        sf_piece = softmax(cached["piece"])
        sf_promo = softmax(cached["promo"])
    
        # let the C++ PriorEngine handle mixing, boosts, clipping and renorm
        pri = self._prior_engine.build(
            leaf.board, legal,
            sf_from, sf_to, sf_piece, sf_promo,
            self.root_stm, req["stm_leaf"]
        )
    
        # expansion + backup (C++ apply_result expects pri as (move, prob) pairs)
        self.apply_result(leaf, pri, cached["value"])

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

    def maybe_early_stop(self, sims_target):
        sims_done = self.sims_completed_this_move
        if self._es_tripped:
            return True
        if sims_done < self.config.es_min_sims:
            return False
        if sims_done - self._es_last_checked_at < self.config.es_check_every:
            return False

        self._es_last_checked_at = sims_done
        rows = self.root_child_visits()
        if len(rows) < 2:
            return False

        n1, n2 = rows[0][1], rows[1][1]
        gap = n1 - n2
        remaining = max(0, sims_target - sims_done)

        if gap > self.config.es_gap_frac * float(remaining):
            self._es_tripped = True
            self._es_reason = (
                f"gap_vs_remaining n1={n1} n2={n2} gap={gap} "
                f"remaining={remaining} thresh={self.config.es_gap_frac * remaining:.1f}"
            )
            self._es_after_sims = sims_done
            return True
        return False

    def stop_simulating(self):
        sims_target = self.config.sims_target
        if self.n_plies > 60:
            sims_target = self.config.sims_target_endgame
        if self.sims_completed_this_move >= sims_target:
            return True
        return self.maybe_early_stop(sims_target)

    def reset_for_new_move(self):
        self.sims_completed_this_move = 0
        self.awaiting_predictions.clear()
        self._move_started_at = _now()

        # early-stop state
        self._es_history.clear()
        self._es_last_checked_at = 0
        self._es_tripped = False
        self._es_reason = ""
        self._es_after_sims = 0

        # reset uniques
        self._uniq_keys.clear()
        self.unique_sims_this_move = 0
