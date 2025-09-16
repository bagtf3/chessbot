import numpy as np
import pandas as pd
import uuid
from time import time as _now
from collections import defaultdict

import chess
import chess.syzygy

from chessbot import ENDGAME_LOC
from chessbot.model import load_model
from chessbot.mcts_utils import MCTSTree, ReuseCache
from chessbot.utils import (
    get_pre_opened_game, show_board, score_game_data, plot_training_progress
)

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"


class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """
    n_training_games = 1000
    
    # MCTS
    c_puct = 1.5
    virtual_loss = 1.0
    dirichlet_alpha = 0.3
    dirichlet_eps = 0.25
    uniform_mix_opening = 0.15
    uniform_mix_later = 0.25

    # Simulation schedule
    sims_target = 400
    
    # Game stuff
    micro_batch_size = 16
    move_limit = 150
    material_diff_cutoff = 12
    material_diff_cutoff_span = 13
    
    # early stop
    es_min_sims = 200
    es_check_every = 4
    es_gap_frac = 0.7

    # Cache
    write_ply_max = 20

    # TF
    training_queue_min = 2048

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
        self.board = board or get_pre_opened_game()
        self.tree = MCTSTree(self.board, self.config)

        self.mat_adv_counter = 0
        self.outcome = None
        self.examples = [] 
        self.plies = 0

    def make_move_from_tree(self):
        """
        Snapshot policy targets from root visits, then play best-by-visits.
        """
        root = self.tree.root
        if not root.children:
            return False
    
        # visits -> π over legal UCIs at the root
        ucis = list(root.children.keys())
        visits = np.array([root.children[m].N for m in ucis], dtype=np.float32)
        s = float(visits.sum())
        pi = (visits / s) if s > 0.0 else None
    
        # factorize π into fixed-size marginals for heads
        if pi is not None:
            fr, to, piece, promo = self.board.moves_to_labels(ucis=ucis)
    
            F   = getattr(self.config, "from_bins", 64)
            T   = getattr(self.config, "to_bins", 64)
            Kp  = getattr(self.config, "piece_bins", 6)
            Kpr = getattr(self.config, "promo_bins", 4)
    
            from_m = np.zeros(F,   dtype=np.float32)
            to_m   = np.zeros(T,   dtype=np.float32)
            pc_m   = np.zeros(Kp,  dtype=np.float32)
            pr_m   = np.zeros(Kpr, dtype=np.float32)
    
            for i, p in enumerate(pi):
                fi, ti, pi_idx, pr_idx = fr[i], to[i], piece[i], promo[i]
                if 0 <= fi < F:      from_m[fi]   += p
                if 0 <= ti < T:      to_m[ti]     += p
                if 0 <= pi_idx < Kp: pc_m[pi_idx] += p
                if 0 <= pr_idx < Kpr: pr_m[pr_idx] += p
    
            planes_k = getattr(self.config, "planes_k", 5)
            x = self.board.stacked_planes(planes_k)
            self.examples.append(
                (x, {"from": from_m, "to": to_m, "piece": pc_m, "promo": pr_m})
            )
    
        # choose move by visits and advance
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
    
        if c_puct is None:
            c_puct = getattr(self.config, "c_puct", 2.0)
    
        sorted_children = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
        sumN = max(1, root.N + sum([c.vloss for _, c in root.children.items()]))
    
        print("\nTop candidate moves:")
        for move_uci, node in sorted_children[:top_n]:
            p = root.P.get(move_uci, 0.0)
            u = c_puct * p * (sumN ** 0.5) / (1 + node.N + node.vloss)
            san = "?"
            if show_san:
                san = self.board.san(move_uci)
            line = (
                f"{move_uci:<6} "
                f"{'('+san+')':<12}  "
                f"visits={node.N:<5} "
                f"P={p:>.3f}  "
                f"Q={node.Q:+.3f}  "
                f"U={u:+.3f}"
            )
            print(line)
    
        best_move_uci, best_node = sorted_children[0]
        best_san = best_move_uci
        if show_san:
            best_san = self.board.san(best_move_uci)
        print(
            f"\nChosen move: {best_move_uci} ({best_san})  "
            f"(visits={best_node.N}, Q={best_node.Q:+.3f})"
        )


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
        
        self._infer_head_dims()
        self.n_retrains = 0
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
        self.game_data = []
    
    def run(self):
        """
        Main loop. Each round: collect, predict, route, apply, maybe play.
        """
        completed_games = 0
        while completed_games < self.config.n_training_games:
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
            "plies": int(game.plies),
        }
        
        res.update(self.config.to_dict())
        self.game_data.append(res)

        z = float(game.outcome if game.outcome is not None else 0.0)
        for x, heads in game.examples:
            self.training_queue.append((x, heads, z))
        game.examples = []  # free memory

        thresh = getattr(self.config, "training_queue_min", 2048)
        if len(self.training_queue) >= thresh:
            self.prep_and_retrain()
        
    def prep_and_retrain(self):
        """
        Build training tensors from self.training_queue and do a quick fit.
        Balances wins vs losses on the value head with per-sample weights.
        """
        if not self.training_queue:
            return

        # unpack
        X, Y_from, Y_to, Y_piece, Y_promo, Z = [], [], [], [], [], []
        for x, heads, z in self.training_queue:
            X.append(x)
            Y_from.append(heads["from"])
            Y_to.append(heads["to"])
            Y_piece.append(heads["piece"])
            Y_promo.append(heads["promo"])
            Z.append(z)

        X = np.asarray(X, dtype=np.float32)
        Y = {
            "value": np.asarray(Z, dtype=np.float32),
            "best_from": np.asarray(Y_from, dtype=np.float32),
            "best_to": np.asarray(Y_to, dtype=np.float32),
            "best_piece": np.asarray(Y_piece, dtype=np.float32),
            "best_promo": np.asarray(Y_promo, dtype=np.float32),
        }

        # per-sample weights to balance wins vs losses (white-POV)
        z = Y["value"]
        pos = float(np.sum(z > 0))
        neg = float(np.sum(z < 0))
        nz = pos + neg
        # if only one side present, fall back to ones
        if nz > 0 and pos > 0 and neg > 0:
            w_pos = 0.5 * nz / pos
            w_neg = 0.5 * nz / neg
            w = np.where(z > 0, w_pos, np.where(z < 0, w_neg, 0.25 * (w_pos + w_neg)))
        else:
            w = np.ones_like(z, dtype=np.float32)

        eval_df = score_game_data(self.model, X, Y)
        self.all_evals = pd.concat([self.all_evals, eval_df])
        if len(self.all_evals) and len(self.all_evals) % 4 == 0:
            plot_training_progress(self.all_evals)

        # train (only the value head gets sample weighting)
        self.model.fit(
            X, Y, epochs=1, batch_size=512, verbose=0, sample_weight={"value": w}
        )

        # clear queue after training
        self.training_queue = []
        self.reuse_cache = ReuseCache()
        self.n_retrains += 1
    
    def log_results(self):
        avg_moves = (self.total_plies / max(1, self.games_finished)) / 2.0
        print("~"*50)
        print(
            f"[stats] finished={self.games_finished}  "
            f"W/L/D={self.white_wins}/{self.black_wins}/{self.draws}  "
            f"avg_len={avg_moves:.1f} moves"
        )
        print("~"*50)
    
    def _normalize_out(self, out):
        # Ensure dict with expected keys regardless of model return type
        if isinstance(out, dict):
            return out
        names = list(getattr(self.model, "output_names", [])) or \
                ["value", "best_from", "best_to", "best_piece", "best_promo"]
        if not isinstance(out, (list, tuple)):
            out = [out]
        return {k: v for k, v in zip(names, out)}

    def _infer_head_dims(self):
        # use first game's encoder to make a sample input
        sample_enc = self.active_games[0].board.stacked_planes(
            getattr(self.config, "planes_k", 5)
        )
        X = np.asarray([sample_enc], dtype=np.float32)
        raw = self.model.predict(X, batch_size=1, verbose=0)
        out = self._normalize_out(raw)
    
        # write bins onto config (so collection uses fixed sizes)
        self.config.from_bins  = int(out["best_from"].shape[-1])
        self.config.to_bins    = int(out["best_to"].shape[-1])
        self.config.piece_bins = int(out["best_piece"].shape[-1])
        self.config.promo_bins = int(out["best_promo"].shape[-1])

 
if __name__ == '__main__':
    model = load_model(MODEL_DIR + "/conv_model_big_v1000.h5")
    looper = GameLooper(games=32, model=model, cfg=Config())
    looper.run()


games_moves_uci = [g['history_uci'] for g in looper.game_data]
    