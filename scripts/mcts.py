import os, json
import chess
import math

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
import random, time

import chessbot.utils as cbu
from chessbot.utils import get_pre_opened_game, softmax

from tensorflow import keras

from chessbot import SF_LOC, ENDGAME_LOC
import chess.syzygy

from pyfastchess import Board as fastboard
from stockfish import Stockfish

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"

def load_model(model_loc):
    model = keras.models.load_model(model_loc)
    return model

model = load_model(MODEL_DIR + "/conv_model_big_v1000.h5")

# Simple early-stop params
ES_MIN_SIMS = 128        # do not check before this many sims
ES_CHECK_EVERY = 4       # check cadence
ES_GAP_FRAC = 0.80       # stop if (N1 - N2) > ES_GAP_FRAC * remaining


def show_top_moves(root, board, top_n=4, c_puct=2.0, show_san=True):
    """Pretty-print the top candidate root moves by visit count, with Q/U/P.

    root: MCTSNode at the current root.
    board: fastboard positioned at root (used only for SAN pretty names).
    top_n: how many moves to display.
    c_puct: exploration constant for the U term display.
    show_san: if True, attempt to print SAN next to each UCI.
    """
    if not root.children:
        print("\nTop candidate moves:\n  (no children)")
        return None

    # Sort by visits descending
    sorted_children = sorted(
        root.children.items(),
        key=lambda kv: kv[1].N,
        reverse=True
    )

    # Compute U scores against current root stats
    sumN = max(1, root.N + sum([c.vloss for _, c in root.children.items()]))

    print("\nTop candidate moves:")
    for move_uci, node in sorted_children[:top_n]:
        p = root.P.get(move_uci, 0.0)
        u = c_puct * p * (sumN ** 0.5) / (1 + node.N + node.vloss)

        san = "?"
        if show_san:
            san = board.san(move_uci)

        line = (
            f"{move_uci:<6} "
            f"{'('+san+')':<12}  "
            f"visits={node.N:<5} "
            f"P={p:>.3f}  "
            f"Q={node.Q:+.3f}  "
            f"U={u:+.3f}"
        )
        print(line)

    # Chosen move by max visits
    best_move_uci, best_node = sorted_children[0]
    best_san = best_move_uci
    
    if show_san:
        best_san = board.san(best_move_uci)

    print(
        f"\nChosen move: {best_move_uci} ({best_san})  "
        f"(visits={best_node.N}, Q={best_node.Q:+.3f})"
    )

    return best_move_uci


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


def model_eval(board, model):
    X = np.asarray([board.stacked_planes(5)], dtype=np.float32)
    out = model.predict(X, batch_size=1, verbose=0)
    if isinstance(out, list):
        names = list(getattr(model, 'output_names', []))
        if names:
            v = out[names.index('value')]
            pf = out[names.index('best_from')]
            pt = out[names.index('best_to')]
            ppiece = out[names.index('best_piece')]
            ppromo = out[names.index('best_promo')]
        else:
            v, pf, pt, ppiece, ppromo = out
    elif isinstance(out, dict):
        v = out['value']; pf = out['best_from']; pt = out['best_to']
        ppiece = out['best_piece']; ppromo = out['best_promo']
    else:
        raise RuntimeError('unexpected model outputs')
    return (
        float(np.asarray(v, np.float32).reshape(-1)[0]),
        softmax(pf.reshape(-1)),
        softmax(pt.reshape(-1)),
        softmax(ppiece.reshape(-1)),
        softmax(ppromo.reshape(-1)),
    )


class MCTSTree:
    def __init__(self, board, model, c_puct=1.5, eps=0.25, alpha=0.03):
        self.model = model
        self.c_puct = c_puct
        self.dir_eps = eps
        self.dir_alpha = alpha
        self.root = MCTSNode(board.side_to_move())
        self.root_board_fen = board.fen()

        # early-stop rolling state
        self._es_history = []
        self._es_last_checked_at = 0
        self._es_tripped = False
        self._es_reason = ""
        self._es_after_sims = 0

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
    
    def _root_child_visits(self):
        node = self.root
        if not node.children:
            return []
        rows = []
        for mv, ch in node.children.items():
            rows.append((mv, ch.N))
        rows.sort(key=lambda t: t[1], reverse=True)
        return rows

    def maybe_early_stop(self, sims_done, sims_target):
        if self._es_tripped:
            return True

        if sims_done < ES_MIN_SIMS:
            return False
        if sims_done - self._es_last_checked_at < ES_CHECK_EVERY:
            return False
        self._es_last_checked_at = sims_done

        rows = self._root_child_visits()
        if len(rows) < 2:
            return False

        n1 = rows[0][1]
        n2 = rows[1][1]
        gap = n1 - n2
        remaining = max(0, sims_target - sims_done)

        if gap > ES_GAP_FRAC * float(remaining):
            self._es_tripped = True
            self._es_reason = (
                "gap_vs_remaining "
                "n1=%d n2=%d gap=%d remaining=%d thresh=%.1f"
                % (n1, n2, gap, remaining, ES_GAP_FRAC * remaining)
            )
            self._es_after_sims = sims_done
            return True

        return False


    def _record_top(self, move):
        self._es_history.append(move)
        if len(self._es_history) > self._es_hist_k:
            self._es_history = self._es_history[-self._es_hist_k:]

    def select_child(self, node):
        sumN = max(1, node.N + sum([c.vloss for c in node.children.values()]))
        best, best_score = None, -1e9
        for mv, child in node.children.items():
            p = node.P.get(mv, 0.0)
            u = self.c_puct * p * math.sqrt(sumN) / (1 + child.N + child.vloss)
            q = child.Q
            if node.stm == 'b':       # flip only here
                q = -q
            score = q + u
            if score > best_score:
                best_score = score
                best = (mv, child)
        return best

    def expand(self, board, node):
        tv = terminal_value_white_pov(board)
        if tv is not None:
            node.is_expanded = True
            node.value = tv if node.stm == 'w' else -tv
            node.legal = []
            return node.value
        print("Expansion with model eval")
        v_w, pf, pt, ppiece, ppromo = model_eval(board, self.model)
        node.value = v_w if node.stm == 'w' else -v_w
        legal = board.legal_moves()
        node.legal = legal
        if not legal:
            node.is_expanded = True
            return node.value

        node.P = priors_from_heads(board, legal, pf, pt, ppiece, ppromo)
        for m in legal:
            board.push_uci(m)
            node.children[m] = MCTSNode(board.side_to_move(), uci=m, parent=node)
            board.unmake()
        node.is_expanded = True
        return node.value

    def backup(self, path, leaf_value):
        v = float(leaf_value)         # white-POV scalar
        for n in reversed(path):
            n.N += 1
            n.W += v                  # no sign flip
            n.Q = n.W / n.N

    def simulate(self, board, total_sims=144, batch_schedule=(2, 4, 6, 8)):
        start = time.time()
        sims_done = 0
        depth_sum = 0
        depth_max = 0
        sched = list(batch_schedule) if batch_schedule else [4]
        s_idx = 0

        # ensure these are reset at the start of simulate
        self._es_last_checked_at = 0
        self._es_tripped = False
        self._es_reason = ""
        self._es_after_sims = 0

        while sims_done < total_sims:
            want = sched[s_idx % len(sched)]
            s_idx += 1
            want = min(want, total_sims - sims_done)
    
            paths = []
            leaves = []
            for _ in range(want):
                p = [self.root]
                while p[-1].is_expanded and p[-1].children:
                    mv, child = self.select_child(p[-1])
                    p.append(child)
                for n in p:
                    n.vloss += 1
                paths.append(p)
                leaves.append(p[-1])
    
                d = len(p) - 1
                depth_sum += d
                if d > depth_max:
                    depth_max = d
    
            X = []
            for p in paths:
                for i in range(1, len(p)):
                    board.push_uci(p[i].uci)
                X.append(board.stacked_planes(5))
                for _ in range(len(p) - 1):
                    board.unmake()
    
            if X:
                X = np.asarray(X, dtype=np.float32)
                out = self.model.predict(X, batch_size=len(X), verbose=0)
                if isinstance(out, list):
                    names = list(getattr(self.model, 'output_names', []))
                    v = out[names.index('value')]
                    pf = out[names.index('best_from')]
                    pt = out[names.index('best_to')]
                    ppiece = out[names.index('best_piece')]
                    ppromo = out[names.index('best_promo')]
                else:
                    v = out['value']; pf = out['best_from']; pt = out['best_to']
                    ppiece = out['best_piece']; ppromo = out['best_promo']
    
                v = np.asarray(v, np.float32).reshape(-1)
                pf = pf.reshape(len(leaves), -1)
                pt = pt.reshape(len(leaves), -1)
                ppiece = ppiece.reshape(len(leaves), -1)
                ppromo = ppromo.reshape(len(leaves), -1)
    
                for i, p in enumerate(paths):
                    leaf = p[-1]
                    for j in range(1, len(p)):
                        board.push_uci(p[j].uci)
    
                    tv = terminal_value_white_pov(board)
                    if tv is not None:
                        leaf.value = tv
                        leaf.legal = []
                        leaf.is_expanded = True
                    else:
                        leaf.value = float(v[i])
                        legal = board.legal_moves()
                        leaf.legal = legal
                        if legal:
                            p_from = softmax(pf[i])
                            p_to = softmax(pt[i])
                            p_piece = softmax(ppiece[i])
                            p_promo = softmax(ppromo[i])
                            P = priors_from_heads(
                                board, legal, p_from, p_to, p_piece, p_promo
                            )
                            leaf.P = P
                            for mv in legal:
                                board.push_uci(mv)
                                leaf.children[mv] = MCTSNode(
                                    board.side_to_move(), uci=mv, parent=leaf
                                )
                                board.unmake()
                        leaf.is_expanded = True
    
                    for _ in range(len(p) - 1):
                        board.unmake()
    
            for p in paths:
                for n in p:
                    n.vloss -= 1
                self.backup(p, p[-1].value)
    
            sims_done += want

            # >>> EARLY-STOP HOOK <<<
            if self.maybe_early_stop(sims_done, total_sims):
                break

        return dict(
            time_s=time.time() - start,
            sims=sims_done,
            avg_depth=(depth_sum / sims_done if sims_done else 0.0),
            max_depth=depth_max,
            early_stop=self._es_tripped,
            es_reason=self._es_reason,
            es_after_sims=self._es_after_sims,
        )

    def best(self):
        if not self.root.children:
            return None, self.root.value
        items = [(m, c.N, c.Q) for m, c in self.root.children.items()]
        m, _, q = max(items, key=lambda x: x[1])
        return m, q


def choose_move_mcts(board, model, simulations=144, c_puct=1.5,
                     reuse_tree=None, batch_schedule=(2, 4, 6, 8),
                     batch_size=None, verbose=True):
    t0 = time.time()
    tree = reuse_tree
    if tree is None or tree.root_board_fen != board.fen():
        tree = MCTSTree(board.clone(), model, c_puct=c_puct)

    b = board.clone()
    if batch_size is not None:
        stats = tree.simulate(b, total_sims=simulations,
                              batch_schedule=(batch_size,))
    else:
        stats = tree.simulate(b, total_sims=simulations,
                              batch_schedule=batch_schedule)

    mv, val = tree.best()
    info = dict(
        sims=stats.get('sims', simulations),
        time_s=time.time() - t0,
        root_children=len(tree.root.children),
        visited_root_children=sum(
            [1 for _, c in tree.root.children.items() if c.N > 0]
        ),
        batch_schedule=(batch_schedule if batch_size is None else (batch_size,)),
        avg_depth=stats.get('avg_depth', 0.0),
        max_depth=stats.get('max_depth', 0),
        early_stop=stats.get('early_stop', False),
        es_reason=stats.get('es_reason', ""),
        es_after_sims=stats.get('es_after_sims', 0),
    )

    if verbose:
        sched = info['batch_schedule']
        # in choose_move_mcts(...) when printing
        print(
            f"[mcts] stm={tree.root.stm} best={mv} Q={val:+.3f} "
            f"sims={info['sims']} time={info['time_s']:.3f}s "
            f"visited_children={info['visited_root_children']}/"
            f"{info['root_children']} schedule={sched} "
            f"avg_depth={info['avg_depth']:.2f} max_depth={info['max_depth']}"
        )

    return mv, val, info, tree


def pick_best_move(board, model, simulations=144, **kwargs):
    """Clone the board and return the chosen move, its value, and info."""
    b = board.clone()
    mv, val, info, tree = choose_move_mcts(b, model, simulations=simulations, **kwargs)
    return mv, val, info, tree
#%%
stockfish_eval = Stockfish(path=SF_LOC)
stockfish_eval.set_depth(10)

stockfish_play = Stockfish(path=SF_LOC)
curr_elo_challenge = 800
stockfish_play.set_elo_rating(curr_elo_challenge)
stockfish_play.set_depth(1)
stockfish_play.update_engine_parameters({"Skill Level": 5})

move_time_min = 0.5
for game_num in range(5):
    game1 = get_pre_opened_game()
    game2 = game1.clone()
    
    for i, board in enumerate([game1, game2]):
        tree = MCTSTree(board.clone(), model, c_puct=2.0, eps=0.05, alpha=0.03)
        sf_plays = 'w' if i else 'b'
        cbu.show_board(chess.Board(board.fen()), flipped=sf_plays=='w')
        reason, result = board.is_game_over()
        while reason == 'none':
            if board.history_size() > 100:
                print("Move limit reached!")
                break
            
            this_fen = board.fen()
            stockfish_eval.set_fen_position(this_fen)
            ev = stockfish_eval.get_evaluation()
            print(f"Current SF Eval: {ev['value']} {ev['type'].upper()}\n")
            
            start = time.time()
            if board.side_to_move() == sf_plays:
                time.sleep(move_time_min)
                stockfish_play.set_fen_position(this_fen)
                stockfish_play.set_elo_rating(curr_elo_challenge)
                stockfish_play.set_depth(1)
                move_uci = stockfish_play.get_best_move()
                san = board.san(move_uci)
                print(f"{san} played by SF")
            
            else:
                mv, val, info, tree = choose_move_mcts(
                    board, model, simulations=1600, c_puct=1.5,
                    reuse_tree=tree, batch_schedule=(16,), verbose=True
                )

                move_uci = show_top_moves(tree.root, board, top_n=4)
            
            board.push_uci(move_uci)
            cbu.show_board(chess.Board(board.fen()), flipped=sf_plays=='w')
            tree.advance(board.clone(), move_uci)
            
            print("\n")
            reason, result = board.is_game_over()
            
            stop = time.time()
            if stop-start < move_time_min:
                time.sleep(move_time_min - (stop-start))



