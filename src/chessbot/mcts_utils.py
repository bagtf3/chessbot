from time import time as _now
from chessbot.utils import softmax
from pyfastchess import priors_from_heads
from pyfastchess import MCTSTree as fasttree


class MCTSTree(fasttree):
    """
    Python ergonomics around the fast C++ tree.

    Key changes vs the old pure-Python class:
      - Never reassign self.root in Python; we call advance_root() in C++.
      - We keep at most one outstanding leaf (collect → apply) to avoid
        multi-path vloss bookkeeping. If you queue, we epoch-guard & clear on advance.
      - apply_result() computes priors (mix by ply) then delegates to C++ apply_result.
    """
    def __init__(self, board, cfg):
        self.config = cfg
        self.c_puct = float(cfg.c_puct)
        super().__init__(board, self.c_puct)

        # bookkeeping mirroring old interface
        self.root_board_fen = board.fen()
        self.n_plies = board.history_size()

        self._move_started_at = _now()
        self.sims_completed_this_move = 0
        self.awaiting_predictions = []

        # early-stop rolling state
        self._es_history = []
        self._es_last_checked_at = 0
        self._es_tripped = False
        self._es_reason = ""
        self._es_after_sims = 0
    
    def collect_one_leaf(self, board):
        """
        Walk PUCT+virtual to a leaf (in C++) and return an inference request dict,
        or None if a cache hit allowed us to apply immediately.

        NOTE: We *do not* mutate the Python 'board' here; the C++ tree holds
        its own board per node.
        """
        leaf = super().collect_one_leaf()
        legal = leaf.board.legal_moves()

        # Build a cache key off the leaf's board state/history (no pushes needed).
        short_fen = leaf.board.fen(include_counters=False)
        move_tail = "|".join(leaf.board.history_uci()[-5:])
        cache_key = f"{short_fen}|{move_tail}"

        req = {
            "leaf": leaf,
            "legal": legal,
            "enc": leaf.board.stacked_planes(5),
            "stm_white": (leaf.board.side_to_move() == 'w'),
            "cache_key": cache_key,
            "n_plies": self.n_plies,
            "epoch": self.epoch
        }

        self.awaiting_predictions.append(req)
        return req

    def resolve_awaiting(self, board, quick_cache):
        """
        Apply any results now available in quick_cache; drop stale by epoch.
        Returns how many were applied.
        """
        if not self.awaiting_predictions:
            return 0

        applied, keep = 0, []
        for req in self.awaiting_predictions:
            cached = quick_cache.get(req["cache_key"])
            if cached is None or req["epoch"] != self.epoch:
                # keep if no cache yet and still current epoch
                if cached is None and req["epoch"] == self.epoch:
                    keep.append(req)
                continue

            self._apply_cached(req, cached)
            self.sims_completed_this_move += 1
            applied += 1

        self.awaiting_predictions = keep
        return applied

    def _apply_cached(self, req, cached):
        """
        Compute priors (with ply-based mix) and delegate to C++ apply_result.
        'cached' must have keys: value, from, to, piece, promo (factorized heads).
        """
        leaf = req["leaf"]
        legal = req["legal"] or leaf.board.legal_moves()

        # Pick uniform mix by game phase
        mix = self.config.anytime_uniform_mix
        is_endgame = leaf.board.piece_count() <= 14 or self.n_plies >= 70
        mix = self.config.endgame_uniform_mix if is_endgame else mix

        pri = priors_from_heads(
            leaf.board, legal,
            softmax(cached["from"]).tolist(), softmax(cached["to"]).tolist(),
            softmax(cached["piece"]).tolist(), softmax(cached["promo"]).tolist(),
            mix=mix)
        
        # check for bumps        
        if is_endgame and self.config.use_prior_boosts:
            eg_adj = self.config.endgame_prior_adjustments
            repp = eg_adj.get("repetition_penalty", 0.0)
            egc = eg_adj.get("capture", 0.0)
            egpp = eg_adj.get("pawn_push", 0.0)
            egchk = eg_adj.get("gives_check", 0.0)

            adjusted_pri = []        
            for m, p in pri:
                if repp:
                    # down scale to prevent going negative
                    if leaf.board.would_be_repetition(m, 1): p *= repp
                if egpp:
                    if leaf.board.is_pawn_move(m): p += egpp
                if egc:
                    if leaf.board.is_capture(m): p += egc
                if egchk:
                    if leaf.board.gives_check(m): p += egchk
                
                adjusted_pri.append((m, p))
            pri = adjusted_pri
        
        # C++ expansion + backup (also pops vloss along the last selected path)
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
