import math
from chessbot.utils import softmax

class MCTSNode:
    def __init__(self, stm, uci=None, parent=None):
        self.parent = parent
        self.uci = uci
        self.stm = stm
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = {}
        self.is_expanded = False
        self.value = None
        self.legal = None
        self.vloss = 0

def terminal_value_white_pov(board):
    reason, result = board.is_game_over()
    if reason == 'none':
        return None
    if reason == 'checkmate':
        winner = 'w' if board.side_to_move() == 'b' else 'b'
        return 1.0 if winner == 'w' else -1.0
    return 0.0


def priors_from_heads(board, legal, p_from, p_to, p_piece, p_promo, mix=0.5):
    """
    Returns dict move -> prior. Mix adds uniform mass:
      final = (1 - mix) * priors + mix * uniform
    """
    if not legal:
        return {}

    fr, to, piece, promo = board.moves_to_labels(ucis=legal)

    pri = []
    for i in range(len(legal)):
        pf = p_from[fr[i]]
        pt = p_to[to[i]]
        pp = p_piece[piece[i]]
        pr = p_promo[promo[i]]  # 0=Q/none, 1=N, 2=B, 3=R
        pri.append(pf * pt * pp * pr)

    s = float(sum(pri))
    n = len(legal)
    if s > 0.0:
        pri = [p / s for p in pri]
    else:
        pri = [1.0 / n] * n

    m = max(0.0, min(1.0, float(mix)))
    if m > 0.0:
        u = 1.0 / n
        pri = [(1.0 - m) * p + m * u for p in pri]
        t = float(sum(pri))
        if t > 0.0:
            inv = 1.0 / t
            pri = [p * inv for p in pri]
        else:
            pri = [u] * n

    return {mv: p for mv, p in zip(legal, pri)}


class MCTSTree:
    def __init__(self, board, cfg):
        self.root = MCTSNode(board.side_to_move())
        self.root_board_fen = board.fen()
        self.config = cfg

        self.c_puct = cfg.c_puct
        self.dir_eps = cfg.dirichlet_eps
        self.dir_alpha = cfg.dirichlet_alpha

        self.sims_completed_this_move = 0
        self.awaiting_predictions = []

        # early-stop rolling state
        self._es_history = []
        self._es_last_checked_at = 0
        self._es_tripped = False
        self._es_reason = ""
        self._es_after_sims = 0

    def select_child(self, node):
        sumN = max(1, node.N + sum([c.vloss for c in node.children.values()]))
        best = None
        best_score = -1e9
        for mv, child in node.children.items():
            p = node.P.get(mv, 0.0)
            u = self.c_puct * p * math.sqrt(sumN) / (1 + child.N + child.vloss)
            score = child.Q + u if node.stm == 'w' else -child.Q + u
            if score > best_score:
                best_score = score
                best = (mv, child)
        return best

    def backup(self, path, leaf_value):
        v = leaf_value
        last_stm = path[-1].stm
        for n in reversed(path):
            n.N += 1
            n.W += v if n.stm == last_stm else -v
            n.Q = n.W / n.N

    def collect_one_leaf(self, board, reuse_cache):
        """
        Walk PUCT+virtual to a leaf. If cache hits, apply immediately and count
        a sim. Otherwise enqueue the request in awaiting_predictions and return it.
        """
        path = [self.root]
        while path[-1].is_expanded and path[-1].children:
            mv, ch = self.select_child(path[-1])
            path.append(ch)

        for n in path:
            n.vloss += 1

        for i in range(1, len(path)):
            board.push_uci(path[i].uci)

        short_fen = board.fen(include_counters=False)
        n_plies = board.history_size()
        move_str = "|".join(board.history_uci()[-5:])
        cache_key = short_fen + "|" + move_str

        enc = board.stacked_planes(5)
        legal = board.legal_moves()

        # pop back
        for _ in range(len(path) - 1):
            board.unmake()

        leaf = path[-1]
        req = {
            "path": path, "leaf": leaf, "enc": enc, "legal": legal,
            "stm_white": (leaf.stm == 'w'), "cache_key": cache_key,
            "n_plies": n_plies
        }

        cached = reuse_cache.get(cache_key)
        if cached is not None:
            # cached is a dict with keys: value, from, to, piece, promo
            self.apply_result(
                board, req,
                cached["value"], cached["from"], cached["to"],
                cached["piece"], cached["promo"]
            )
            self.sims_completed_this_move += 1
            return None  # handled by cache now

        if n_plies < 20:
            req['move_history'] = "|".join(board.history_uci())

        self.awaiting_predictions.append(req)
        return req

    def resolve_awaiting(self, board, quick_cache):
        """
        Scan awaiting_predictions for cache hits, apply them, and remove from queue.
        Returns how many were applied.
        """
        if not self.awaiting_predictions:
            return 0

        applied = 0
        keep = []
        for req in self.awaiting_predictions:
            cached = quick_cache.get(req["cache_key"])
            if cached is None:
                keep.append(req)
                continue
            self.apply_result(
                board, req,
                cached["value"], cached["from"], cached["to"],
                cached["piece"], cached["promo"]
            )
            self.sims_completed_this_move += 1
            applied += 1

        self.awaiting_predictions = keep
        return applied

    def apply_result(self, board, req, value_w, p_from, p_to, p_piece, p_promo):
        """
        Expand leaf using model outputs and back up value. All board pushes/
        pops happen here. No model calls.
        """
        path = req["path"]
        leaf = req["leaf"]

        for i in range(1, len(path)):
            board.push_uci(path[i].uci)

        tv = terminal_value_white_pov(board)
        if tv is not None:
            leaf.value = tv if leaf.stm == 'w' else -tv
            leaf.legal = []
            leaf.is_expanded = True
        else:
            v_w = float(value_w)
            leaf.value = v_w if leaf.stm == 'w' else -v_w
            legal = req["legal"] if req["legal"] else board.legal_moves()
            leaf.legal = legal
            if legal:
                pri = priors_from_heads(
                    board, legal,
                    softmax(p_from), softmax(p_to),
                    softmax(p_piece), softmax(p_promo),
                    mix=self.config.uniform_mix
                )
                leaf.P = pri
                for mv in legal:
                    board.push_uci(mv)
                    leaf.children[mv] = MCTSNode(
                        board.side_to_move(), uci=mv, parent=leaf
                    )
                    board.unmake()
            leaf.is_expanded = True

        for _ in range(len(path) - 1):
            board.unmake()

        for n in path:
            n.vloss -= 1
        self.backup(path, leaf.value)

    def advance(self, board, move_uci):
        if move_uci in self.root.children:
            new_root = self.root.children[move_uci]
            new_root.parent = None
            self.root = new_root
            board.push_uci(move_uci)
            self.root_board_fen = board.fen()
        else:
            board.push_uci(move_uci)
            self.root = MCTSNode(board.side_to_move())
            self.root_board_fen = board.fen()
    
    def root_child_visits(self):
        node = self.root
        if not node.children:
            return []
        rows = []
        for mv, ch in node.children.items():
            rows.append((mv, ch.N))
        rows.sort(key=lambda t: t[1], reverse=True)
        return rows

    def maybe_early_stop(self):
        sims_done = self.sims_completed_this_move
        sims_target = self.config.sims_target

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

        n1 = rows[0][1]
        n2 = rows[1][1]
        gap = n1 - n2
        remaining = max(0, sims_target - sims_done)

        if gap > self.config.es_gap_frac * float(remaining):
            self._es_tripped = True
            self._es_reason = (
                "gap_vs_remaining "
                "n1=%d n2=%d gap=%d remaining=%d thresh=%.1f"
                % (n1, n2, gap, remaining, self.config.es_gap_frac * remaining)
            )
            self._es_after_sims = sims_done
            return True
        return False

    def stop_simulating(self):
        # hard budget
        if self.sims_completed_this_move >= self.config.sims_target:
            return True
        # simple early-stop
        return self.maybe_early_stop()

    def reset_for_new_move(self):
        self.sims_completed_this_move = 0
        self.awaiting_predictions = []

        # early-stop state
        self._es_history = []
        self._es_last_checked_at = 0
        self._es_tripped = False
        self._es_reason = ""
        self._es_after_sims = 0

        # drop any lingering virtual losses anywhere in the tree
        stack = [self.root]
        while stack:
            n = stack.pop()
            n.vloss = 0
            for ch in n.children.values():
                stack.append(ch)

    def best(self):
        if not self.root.children:
            return None, self.root.value
        items = [(m, c.N, c.Q) for m, c in self.root.children.items()]
        m, _, q = max(items, key=lambda x: x[1])
        return m, q