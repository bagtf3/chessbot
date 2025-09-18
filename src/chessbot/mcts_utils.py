import math
from collections import OrderedDict
from time import time as _now
from chessbot.utils import softmax
from pyfastchess import MCTSNode, priors_from_heads
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
    
    def collect_one_leaf(self, board, reuse_cache):
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
        move_hist = "|".join(leaf.board.history_uci())
        move_tail = "|".join(leaf.board.history_uci()[-5:])
        cache_key = f"{short_fen}|{move_tail}"

        req = {
            "leaf": leaf,
            "legal": legal,
            "enc": leaf.board.stacked_planes(5),
            "stm_white": (leaf.board.side_to_move() == 'w'),
            "cache_key": cache_key,
            "n_plies": self.n_plies,
            "move_history": move_hist if self.n_plies < 30 else None,
            "epoch": self.epoch
        }

        # Opportunistic cache hit
        if self.n_plies <= self.config.write_ply_max + 5:
            cached = reuse_cache.get(cache_key)
            if cached is not None and req["epoch"] == self.epoch:
                self._apply_cached(req, cached)
                self.sims_completed_this_move += 1
                return None

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
        if self.n_plies < 20:
            mix = self.config.uniform_mix_opening
        elif self.n_plies < 50:
            mix = self.config.uniform_mix_later
        else:
            mix = self.config.uniform_mix_endgame

        pri = priors_from_heads(
            leaf.board, legal,
            softmax(cached["from"]), softmax(cached["to"]),
            softmax(cached["piece"]), softmax(cached["promo"]),
            mix=mix
        )
        
        # C++ expansion + backup (also pops vloss along the last selected path)
        self.apply_result(leaf, pri, float(cached["value"]))
    
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


# class MCTSTree:
#     def __init__(self, board, cfg):
#         self.root = MCTSNode(board.side_to_move())
#         self.root_board_fen = board.fen()
#         self.n_plies = board.history_size()
#         self.config = cfg

#         self.c_puct = cfg.c_puct
        
#         self._move_started_at = _now()
#         self.sims_completed_this_move = 0
#         self.awaiting_predictions = []

#         # early-stop rolling state
#         self._es_history = []
#         self._es_last_checked_at = 0
#         self._es_tripped = False
#         self._es_reason = ""
#         self._es_after_sims = 0

#     def select_child(self, node):
#         sumN = max(1, node.N + sum([c.vloss for c in node.children.values()]))
#         best, best_score = None, -1e9
#         for mv, child in node.children.items():
#             p = node.P.get(mv, 0.0)
#             u = self.c_puct * p * math.sqrt(sumN) / (1 + child.N + child.vloss)
#             q = child.Q
#             if node.stm == 'b':       # flip only here
#                 q = -q
#             score = q + u
#             if score > best_score:
#                 best_score = score
#                 best = (mv, child)
#         return best


#     def backup(self, path, leaf_value):
#         v = float(leaf_value)         # white-POV scalar
#         for n in reversed(path):
#             n.N += 1
#             n.W += v                  # no sign flip
#             n.Q = n.W / n.N

#     def collect_one_leaf(self, board, reuse_cache):
#         """
#         Walk PUCT+virtual to a leaf. If cache hits, apply immediately and count
#         a sim. Otherwise enqueue the request in awaiting_predictions and return it.
#         """
#         path = [self.root]
#         while path[-1].is_expanded and path[-1].children:
#             mv, ch = self.select_child(path[-1])
#             path.append(ch)

#         for n in path:
#             n.vloss += 1

#         for i in range(1, len(path)):
#             board.push_uci(path[i].uci)

#         short_fen = board.fen(include_counters=False)
#         n_plies = self.n_plies
#         move_str = "|".join(board.history_uci()[-5:])
#         move_history = "|".join(board.history_uci())
#         cache_key = short_fen + "|" + move_str

#         enc = board.stacked_planes(5)
#         legal = board.legal_moves()

#         # pop back
#         for _ in range(len(path) - 1):
#             board.unmake()

#         leaf = path[-1]
#         req = {
#             "path": path, "leaf": leaf, "enc": enc, "legal": legal,
#             "stm_white": (leaf.stm == 'w'), "cache_key": cache_key,
#             "n_plies": n_plies
#         }
        
#         # only check the cache if were close to the cutoff
#         if n_plies <= self.config.write_ply_max + 5:
#             cached = reuse_cache.get(cache_key)
#             if cached is not None:
#                 # cached is a dict with keys: value, from, to, piece, promo
#                 self.apply_result(
#                     board, req,
#                     cached["value"], cached["from"], cached["to"],
#                     cached["piece"], cached["promo"]
#                 )
#                 self.sims_completed_this_move += 1
#                 return None  # handled by cache now

#         if n_plies < 30:
#             req['move_history'] = move_history

#         self.awaiting_predictions.append(req)
#         return req

#     def resolve_awaiting(self, board, quick_cache):
#         """
#         Scan awaiting_predictions for cache hits, apply them, and remove from queue.
#         Returns how many were applied.
#         """
#         if not self.awaiting_predictions:
#             return 0

#         applied = 0
#         keep = []
#         for req in self.awaiting_predictions:
#             cached = quick_cache.get(req["cache_key"])
#             if cached is None:
#                 keep.append(req)
#                 continue
#             self.apply_result(
#                 board, req,
#                 cached["value"], cached["from"], cached["to"],
#                 cached["piece"], cached["promo"]
#             )
#             self.sims_completed_this_move += 1
#             applied += 1

#         self.awaiting_predictions = keep
#         return applied

#     def apply_result(self, board, req, value_w, p_from, p_to, p_piece, p_promo):
#         """
#         Expand leaf using model outputs and back up value. All board pushes/
#         pops happen here. No model calls.
#         """
#         path = req["path"]
#         leaf = req["leaf"]

#         for i in range(1, len(path)):
#             board.push_uci(path[i].uci)

#         tv = terminal_value_white_pov(board)
#         if tv is not None:
#             leaf.value = tv
#             leaf.legal = []
#             leaf.is_expanded = True
#         else:
#             leaf.value = float(value_w)
#             legal = req["legal"] if req["legal"] else board.legal_moves()
#             leaf.legal = legal
#             if legal:
#                 if self.n_plies < 20:
#                     mix = self.config.uniform_mix_opening
#                 elif self.n_plies < 50:
#                     mix = self.config.uniform_mix_later
#                 else:
#                     mix = self.config.uniform_mix_endgame

#                 pri = priors_from_heads(
#                     board, legal, softmax(p_from), softmax(p_to),
#                     softmax(p_piece), softmax(p_promo), mix=mix
#                 )
#                 leaf.P = pri
#                 for mv in legal:
#                     board.push_uci(mv)
#                     leaf.children[mv] = MCTSNode(
#                         board.side_to_move(), uci=mv, parent=leaf
#                     )
#                     board.unmake()
#             leaf.is_expanded = True

#         for _ in range(len(path) - 1):
#             board.unmake()

#         for n in path:
#             n.vloss -= 1
#         self.backup(path, leaf.value)

#     def advance(self, board, move_uci):
#         if move_uci in self.root.children:
#             new_root = self.root.children[move_uci]
#             new_root.parent = None
#             self.root = new_root
#             board.push_uci(move_uci)
#             self.root_board_fen = board.fen()
#             self.n_plies = board.history_size()
#         else:
#             board.push_uci(move_uci)
#             self.root = MCTSNode(board.side_to_move())
#             self.root_board_fen = board.fen()
#             self.n_plies = board.history_size()
    
#     def root_child_visits(self):
#         node = self.root
#         if not node.children:
#             return []
#         rows = []
#         for mv, ch in node.children.items():
#             rows.append((mv, ch.N))
#         rows.sort(key=lambda t: t[1], reverse=True)
#         return rows
    
#     def visit_weighted_Q(self):
#         node = self.root
#         if not node.children:
#             return 0.0
#         wts = [ch.N for ch in node.children.values() if ch.N > 0]
#         q_vals  = [ch.Q for ch in node.children.values() if ch.N > 0]
#         total = sum(wts)
#         return sum([w * q for w, q in zip(wts, q_vals)]) / total if total > 0 else 0.0

#     def maybe_early_stop(self, sims_target):
#         sims_done = self.sims_completed_this_move
#         if self._es_tripped:
#             return True

#         if sims_done < self.config.es_min_sims:
#             return False
#         if sims_done - self._es_last_checked_at < self.config.es_check_every:
#             return False
#         self._es_last_checked_at = sims_done

#         rows = self.root_child_visits()
#         if len(rows) < 2:
#             return False

#         n1 = rows[0][1]
#         n2 = rows[1][1]
#         gap = n1 - n2
#         remaining = max(0, sims_target - sims_done)

#         if gap > self.config.es_gap_frac * float(remaining):
#             self._es_tripped = True
#             self._es_reason = (
#                 "gap_vs_remaining "
#                 "n1=%d n2=%d gap=%d remaining=%d thresh=%.1f"
#                 % (n1, n2, gap, remaining, self.config.es_gap_frac * remaining)
#             )
#             self._es_after_sims = sims_done
#             return True
#         return False

#     def stop_simulating(self):
#         sims_target = self.config.sims_target
#         if self.n_plies > 50:
#             sims_target = self.config.sims_target_endgame
        
#         # hard budget
#         if self.sims_completed_this_move >= sims_target:
#             return True
        
#         # simple early-stop
#         return self.maybe_early_stop(sims_target)

#     def reset_for_new_move(self):
#         self.sims_completed_this_move = 0
#         self.awaiting_predictions = []
#         self._move_started_at = _now()

#         # early-stop state
#         self._es_history = []
#         self._es_last_checked_at = 0
#         self._es_tripped = False
#         self._es_reason = ""
#         self._es_after_sims = 0

#         # drop any lingering virtual losses anywhere in the tree
#         stack = [self.root]
#         while stack:
#             n = stack.pop()
#             n.vloss = 0
#             for ch in n.children.values():
#                 stack.append(ch)

#     def best(self):
#         if not self.root.children:
#             return None, self.root.value
#         items = [(m, c.N, c.Q) for m, c in self.root.children.items()]
#         m, _, q = max(items, key=lambda x: x[1])
#         return m, q


class ReuseCache:
    def __init__(self, capacity=300_000):
        self.capacity = int(capacity)
        self._store = OrderedDict()  # key -> value
        self.checks = 0
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.evictions = 0

    def get(self, key):
        self.checks += 1
        try:
            val = self._store[key]
            self._store.move_to_end(key, last=True)  # LRU touch
            self.hits += 1
            return val
        except KeyError:
            self.misses += 1
            return None

    def put(self, key, value):
        self.puts += 1
        if key in self._store:
            self._store.move_to_end(key, last=True)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # evict LRU
            self.evictions += 1

    # optional dict-like sugar
    def __len__(self): return len(self._store)
    def __getitem__(self, k): 
        v = self.get(k)
        if v is None: raise KeyError(k)
        return v
    def __setitem__(self, k, v): self.put(k, v)

    def clear(self):
        self._store.clear()
        self.checks = self.hits = self.misses = self.puts = self.evictions = 0

    def stats(self):
        hr = (self.hits / self.checks) if self.checks else 0.0
        return {
            "size": len(self._store),
            "capacity": self.capacity,
            "checks": self.checks,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hr,
            "puts": self.puts,
            "evictions": self.evictions,
        }
