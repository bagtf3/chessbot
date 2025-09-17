import numpy as np
import pandas as pd
import uuid, os
from time import time as _now
from collections import defaultdict

import chess
import chess.syzygy

from chessbot import ENDGAME_LOC
from chessbot.model import load_model
from chessbot.mcts_utils import MCTSTree, ReuseCache

from chessbot.utils import score_game_data, plot_training_progress, log_and_plot_sf
from chessbot.utils import (
    get_pre_opened_game, evaluate_many_games, show_board, random_init
)    

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"
VIS_DIR = "C:/Users/Bryan/Data/chessbot_data/visuals"

class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """
    # MCTS
    c_puct = 1.5
    virtual_loss = 1.0
    uniform_mix_opening = 0.25
    uniform_mix_later = 0.25

    # Simulation schedule
    sims_target = 400
    micro_batch_size = 16
    
    # Game stuff
    move_limit = 150
    material_diff_cutoff = 12
    material_diff_cutoff_span = 15
    n_training_games = 1000
    restart_after_result = True
    random_init_blend = 0.5 # how many random_init vs pre-opened
    random_init_plies = 8 # number of random init'd moves
    
    # early stop
    es_min_sims = 120
    es_check_every = 4
    es_gap_frac = 0.75

    # Cache
    write_ply_max = 20

    # TF
    training_queue_min = 2048
    vwq_blend = 0.5
    target_mean = 0.3
    draw_frac = 0.3
    factorized_bins = (64, 64, 6, 4)
    model_save_pattern = "conv_super_bootstrapped"
    sf_plot_path = os.path.join(VIS_DIR, "conv_super_bootstrapped_sf_eval")
    progress_plot_path = os.path.join(VIS_DIR, "conv_super_bootstrapped_progress.png")

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in dir(self)
            if not k.startswith("_") and not callable(getattr(self, k))
        }


class ChessGame(object):
    def __init__(self, board=None, cfg=None):
        self.game_id = str(uuid.uuid4())
        self.config = cfg or Config()
        
        if board is None:
            if np.random.uniform() <= self.config.random_init_blend:
                board = random_init(self.config.random_init_plies)
            else:
                board = get_pre_opened_game()
                
        self.board = board
        self.tree = MCTSTree(self.board, self.config)

        self.mat_adv_counter = 0
        self.outcome = None
        self.examples = [] 
        self.plies = 0
        self.vwq = None

    def make_move_from_tree(self):
        """
        Snapshot policy targets from root visits, then play best-by-visits.
        """
        root = self.tree.root
        if not root.children:
            return False

        ucis = list(root.children.keys())
        visits = np.array([root.children[m].N for m in ucis], dtype=np.float32)
        s = float(visits.sum())
        pi = (visits / s) if s > 0.0 else None

        vwq = self.tree.visit_weighted_Q()  # grab visit-weighted Q
        self.vwq = vwq

        if pi is not None:
            fr, to, piece, promo = self.board.moves_to_labels(ucis=ucis)
            
            # these are infereed automatically 
            F, T, Kp, Kpr = self.config.factorized_bins
            from_m = np.zeros(F, dtype=np.float32)
            to_m = np.zeros(T, dtype=np.float32)
            pc_m = np.zeros(Kp, dtype=np.float32)
            pr_m = np.zeros(Kpr, dtype=np.float32)

            for i, p in enumerate(pi):
                fi, ti, pi_idx, pr_idx = fr[i], to[i], piece[i], promo[i]
                from_m[fi]   += p
                to_m[ti]     += p
                pc_m[pi_idx] += p
                pr_m[pr_idx] += p

            x = self.board.stacked_planes(5)
            self.examples.append(
                (x,
                {
                    "from": from_m, "to": to_m, "piece": pc_m,
                    "promo": pr_m, "vwq": float(vwq)
                })
            )

        mv, _ = self.tree.best()
        if mv is None:
            return False

        self.tree.advance(self.board, mv)
        self.tree.reset_for_new_move()
        self.plies += 1
        return True

    def check_for_terminal(self):
        reason, result = self.board.is_game_over()
        if reason != 'none':
            print(self.game_id, "Game over detected:", reason, self.board.fen())
            self.outcome = self.evaluate_terminal(reason, result)
            return True
    
        mat_diff = self.board.material_count()
        if abs(mat_diff) >= self.config.material_diff_cutoff:
            self.mat_adv_counter += 1
        else:
            self.mat_adv_counter = 0
    
        if self.mat_adv_counter >= self.config.material_diff_cutoff_span:
            print(self.game_id, "Material cutoff:", mat_diff)
            self.outcome = 1.0 if mat_diff > 0 else -1.0
            return True
        
        # Syzygy probe if few pieces
        if self.board.piece_count() <= 5:
            # may not work so just go as normal
            try:
                outcomes = {-2: -1, -1:-1, 0:0, 1:1, 2:1}
                with chess.syzygy.open_tablebase(ENDGAME_LOC) as tablebase:
                    # gotta flip back to python chess here
                    chess_board = chess.Board(self.board.fen())
                    table_res = tablebase.probe_wdl(chess_board)
                    
                table_res = table_res if chess_board.turn else -1*table_res
                self.outcome = outcomes[table_res]
                print(self.game_id, "Endgame Reached:", self.outcome, self.board.fen())
                return True
            except:
                pass
           
        hs = self.board.history_size()
        if hs > self.config.move_limit:
            print(self.game_id, "Move limit reached:", hs)
            self.outcome = 0.0
            return True
        # if we made it here the game is active
        return False
    
    def evaluate_terminal(self, reason, result):
        if reason == "none":
            raise Exception("Non terminal board found!")

        if reason == "checkmate":
            loser = self.board.side_to_move()
            return -1 if loser == "w" else 1
        return 0
        
    def turn(self, return_bool=True):
        stm = self.board.side_to_move()
        if return_bool:
            return stm == 'w'
        else:
            return stm
        
    def show_board(self, flipped=False, sleep=0.0):
        sb = chess.Board(self.board.fen())
        show_board(sb, flipped=flipped, sleep=sleep)

    def show_top_moves(self, top_n=4, show_san=True):
        root = self.tree.root
        c_puct = self.config.c_puct
        if not root.children:
            print("\nTop candidate moves:\n  (no children)")
            return None
    
        # search summary: sims, time, avg/max depth, children visited
        start = getattr(self.tree, "_move_started_at", None)
        if start is None:
            start = _now()
            self.tree._move_started_at = start
        elapsed = _now() - start
    
        def _depth_stats(r):
            total_v, sum_vd, dmax = 0, 0.0, 0
            stack = [(r, 0)]
            while stack:
                n, d = stack.pop()
                if n is not r and n.N > 0:
                    total_v += n.N
                    sum_vd += d * n.N
                    if d > dmax:
                        dmax = d
                for ch in n.children.values():
                    stack.append((ch, d + 1))
            avg_d = (sum_vd / total_v) if total_v > 0 else 0.0
            return avg_d, dmax
    
        avg_depth, max_depth = _depth_stats(root)
        sims = int(getattr(self.tree, "sims_completed_this_move", 0))
    
        total_children = len(root.children)
        visited_children = sum([1 for ch in root.children.values() if ch.N > 0])
    
        print(
            f"\nSearch summary: sims={sims}  time={elapsed:.2f}s  "
            f"avg_depth={avg_depth:.2f}  max_depth={max_depth}  "
            f"children_visited={visited_children}/{total_children}"
        )
    
        # top-N printout
        sorted_children = sorted(
            root.children.items(), key=lambda kv: kv[1].N, reverse=True
        )
        sumN = max(1, root.N + sum([c.vloss for _, c in root.children.items()]))
    
        print("Top candidate moves:")
        for move_uci, node in sorted_children[:top_n]:
            p = root.P.get(move_uci, 0.0)
            u = c_puct * p * (sumN ** 0.5) / (1 + node.N + node.vloss)
            san = self.board.san(move_uci) if show_san else "?"
            line = (
                f"{move_uci:<6} "
                f"{'('+san+')':<12}  "
                f"visits={node.N:<5} "
                f"P={p:>.3f}  "
                f"Q={node.Q:+.3f}  "
                f"U={u:+.3f}"
            )
            print(line)
    
        # chosen move
        best_move_uci, best_node = sorted_children[0]
        best_san = self.board.san(best_move_uci) if show_san else best_move_uci
    
        # visit-weighted root Q
        v_mcts = best_node.Q if self.vwq is None else self.vwq
    
        print(
            f"\nChosen move: {best_move_uci} ({best_san})  "
            f"(visits={best_node.N}, Q={best_node.Q:+.3f}, visit-weighted Q={v_mcts:+.3f})"
        )
    
    def print_pv(self, max_len=4):
        b = self.board.clone()
        node = self.tree.root
        moves, qs = [], []
        for _ in range(max_len):
            if not node.children:
                break
            m, child = max(node.children.items(), key=lambda kv: kv[1].N)
            moves.append(b.san(m))
            qs.append(child.Q)
            b.push_uci(m)
            node = child
        pv = " ".join(moves) if moves else "(none)"
        qtrail = " -> ".join([f"{q:+.3f}" for q in qs])
        print(f"PV: {pv}")
        if qs:
            print(f"Q trail: {qtrail}  (final {qs[-1]:+.3f})")


class GameLooper(object):
    """
    Orchestrates N games concurrently, central batching, caches, and training.
    """
    def __init__(self, games, model, cfg):
        self.config = cfg or Config()
        
        if isinstance(games, int):
            self.active_games = [ChessGame() for _ in range(games)]
        else:
            # assumes these are pre-loaded Game objects in a list
            self.active_games = games
        
        self.model = model
        self.training_queue = []
        self.reuse_cache = ReuseCache()
        self.quick_cache = {}
        self.pos_counter = defaultdict(int)

        self.games_finished = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_plies = 0
        self.all_evals = pd.DataFrame()
        self.pre_game_data = []
        self.game_data = []
        self.n_retrains = 0
        self.intra_training_summaries = {}
    
    def run(self):
        """
        Main loop. Each round: collect, predict, route, apply, maybe play.
        """
        completed_games = 0
        while completed_games < self.config.n_training_games:
            if not self.active_games:
                break
            
            preds_batch = []
            finished = []
            for game in self.active_games:
                # see if its time to make a move
                game.tree.resolve_awaiting(game.board, self.quick_cache)

                if game.tree.stop_simulating():
                    # show the first game to monitor progress
                    display_game = game.game_id == self.active_games[0].game_id
                    if display_game:
                        game.show_top_moves()
                        
                    game.make_move_from_tree()
                    
                    if display_game:
                        game.show_board()

                    if game.check_for_terminal():
                        completed_games += 1
                        # populate training queue, maybe retrain
                        self.finalize_game_data(game)
                        self.log_results()
                        finished.append(game.game_id)
                        continue

                n_collected, tries = 0, 0
                while n_collected < self.config.micro_batch_size:
                    leaf_req = game.tree.collect_one_leaf(game.board, self.reuse_cache)
                    if leaf_req is None:
                        tries += 1
                        if tries > 4: break
                    else:
                        n_collected += 1
                        preds_batch.append(leaf_req)

            # this will also update cache(s)
            self.format_and_predict(preds_batch)

            # resolve predictions back to the games/trees
            applied = 0
            for game in self.active_games:
                applied += game.tree.resolve_awaiting(game.board, self.quick_cache)
            
            # if all 0, we can clear the quick cache, keep it light
            if applied == 0:
                self.quick_cache = {}

            # remove finished games and init new ones
            self.active_games = [
                g for g in self.active_games if g.game_id not in finished
            ]
            
            if self.config.restart_after_result:
                self.active_games += [ChessGame() for _ in range(len(finished))]
        
        # maybe return some logs or something some day
        return

    def format_and_predict(self, preds_batch):
        """
        Take leaf requests from all games, run one model call, and write
        results into quick_cache (and reuse_cache).
        Each req must have: 'enc', 'cache_key'.
        """
        if not preds_batch:
            return

        X = np.asarray([r["enc"] for r in preds_batch], dtype=np.float32)

        out = self.model.predict(X, batch_size=768, verbose=0)

        if isinstance(out, list):
            names = list(getattr(self.model, 'output_names', []))
            v = out[names.index('value')]
            pf = out[names.index('best_from')]
            pt = out[names.index('best_to')]
            ppc = out[names.index('best_piece')]
            ppr = out[names.index('best_promo')]
        else:
            v   = out['value']
            pf  = out['best_from']
            pt  = out['best_to']
            ppc = out['best_piece']
            ppr = out['best_promo']

        # write to caches keyed by the req cache_key
        for i, req in enumerate(preds_batch):
            key = req["cache_key"]
            out_i = {
                "value": float(v[i].item()),
                "from": pf[i],
                "to": pt[i],
                "piece": ppc[i],
                "promo": ppr[i],
            }
            self.quick_cache[key] = out_i

            # optional long-lived reuse
            if req.get("n_plies", 999) <= self.config.write_ply_max:
                self.reuse_cache[key] = out_i
            
            # store common positions to warm cache with after training
            mh = req.get("move_history", False)
            if mh:
                self.pos_counter[mh] += 1

    def finalize_game_data(self, game):
        """
        Attach the final scalar outcome to every per-move example and enqueue.
        Outcome is already white-POV (-1/0/+1) and does not need flipping.
        """
        # aggregate stats
        self.games_finished += 1
        self.total_plies += game.plies
        if game.outcome > 0:
            self.white_wins += 1
        elif game.outcome < 0:
            self.black_wins += 1
        else:
            self.draws += 1

        res = {
            "ts": _now(),
            "game_id": game.game_id,
            "result": float(game.outcome if game.outcome is not None else 0.0),
            "plies": int(game.plies), "history_uci": game.board.history_uci()
        }
        
        res.update(self.config.to_dict())
        self.pre_game_data.append(res)

        z = float(game.outcome if game.outcome is not None else 0.0)
        for x, heads in game.examples:
            vwq = heads.get("vwq", 0.0)
            self.training_queue.append((x, heads, z, vwq))
        game.examples = []

        thresh = getattr(self.config, "training_queue_min", 2048)
        if len(self.training_queue) >= thresh:
            self.trigger_retrain()
        
    def trigger_retrain(self):
        """
        Build training tensors from self.training_queue and do a quick fit.
        Blends game outcomes with visit-weighted Q for the value head.
        Balances wins vs losses on the value head with per-sample weights.
        """
        if not self.training_queue:
            return
    
        # unpack examples
        X, Y_from, Y_to, Y_piece, Y_promo, Z, Vwq = [], [], [], [], [], [], []
        for x, heads, z, vwq in self.training_queue:
            X.append(x)
            Y_from.append(heads["from"])
            Y_to.append(heads["to"])
            Y_piece.append(heads["piece"])
            Y_promo.append(heads["promo"])
            Z.append(z)
            Vwq.append(vwq)
    
        # blend outcome with visit-weighted Q
        alpha = self.config.vwq_blend
        Z = np.asarray(Z, dtype=np.float32)
        Vwq = np.asarray(Vwq, dtype=np.float32)
        Y_value = (1 - alpha) * Z + alpha * Vwq
    
        # assemble training dict
        X = np.asarray(X, dtype=np.float32)
        Y = {
            "value": Y_value.astype(np.float32),
            "best_from": np.asarray(Y_from, dtype=np.float32),
            "best_to": np.asarray(Y_to, dtype=np.float32),
            "best_piece": np.asarray(Y_piece, dtype=np.float32),
            "best_promo": np.asarray(Y_promo, dtype=np.float32),
        }
    
        # per-sample weights (based on raw outcomes)
        target_mean = self.config.target_mean
        draw_frac = self.config.draw_frac
    
        pos = float(np.sum(Z > 0))
        neg = float(np.sum(Z < 0))
        nz  = pos + neg
    
        if nz > 0 and pos > 0 and neg > 0:
            w_pos = 0.5 * nz / pos
            w_neg = 0.5 * nz / neg
            w_draw = draw_frac * 0.5 * (w_pos + w_neg)
            w = np.where(Z > 0, w_pos, np.where(Z < 0, w_neg, w_draw)).astype(np.float32)
        else:
            w = np.ones_like(Z, dtype=np.float32)
    
        # scale weights to target mean
        mean_w = float(w.mean()) if w.size else 1.0
        if mean_w > 0:
            w *= (target_mean / mean_w)
        else:
            w[:] = target_mean
    
        # evaluation and logging
        eval_df = score_game_data(self.model, X, Y)
        self.all_evals = pd.concat([self.all_evals, eval_df])
        if len(self.all_evals) and len(self.all_evals) % 4 == 0:
            plot_training_progress(
                self.all_evals, save_path=self.config.progress_plot_path
            )
    
        move_lists = [g['history_uci'] for g in self.pre_game_data]
        print(f"Analyzing last {len(move_lists)} games with Stockfish(d=8)...")
        sf_analysis = evaluate_many_games(move_lists, depth=8, workers=10, mate_cp=1500)
        self.intra_training_summaries[self.n_retrains] = sf_analysis
        log_and_plot_sf(self.intra_training_summaries, save_path=self.config.sf_plot_path)
    
        # archive pre-game data
        self.game_data += self.pre_game_data
        self.pre_game_data = []   # <- small typo fix from pre_data_data
    
        # fit (only value head weighted)
        self.model.fit(
            X, Y, epochs=1, batch_size=512, verbose=0, sample_weight={"value": w}
        )
        
        # clear training queue and bump retrain counter
        self.training_queue = []
        self.reuse_cache = ReuseCache()
        self.n_retrains += 1
        if self.n_retrains % 10 == 0:
            outfile = self.config.model_save_pattern + "_latest.h5"
            model_out = os.path.join(MODEL_DIR, outfile)
            model.save(model_out)
            
        if self.n_retrains % 50 == 0:
            outfile = self.config.model_save_pattern + f"_v{self.n_retrains}.h5"
            model_out = os.path.join(MODEL_DIR, outfile)
            model.save(model_out)

    def log_results(self):
        avg_moves = (self.total_plies / max(1, self.games_finished)) / 2.0
        print("~"*50)
        print(
            f"[stats] finished={self.games_finished}  "
            f"W/L/D={self.white_wins}/{self.black_wins}/{self.draws}  "
            f"avg_len={avg_moves:.1f} moves"
        )
        print("~"*50)


if __name__ == '__main__':
    try_latest = os.path.join(MODEL_DIR, "conv_super_bootstrapped_latest.h5")
    if os.path.exists(try_latest):
        model = load_model(try_latest)
    
    # fallback
    else:
        model = load_model(MODEL_DIR + "/conv_model_big_v1000.h5")
        
    looper = GameLooper(games=48, model=model, cfg=Config())
    looper.run()
