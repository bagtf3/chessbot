import numpy as np
import pandas as pd
import uuid, os
from time import time as _now

import random
import chess
import chess.syzygy

from pyfastchess import terminal_value_white_pov

from chessbot import ENDGAME_LOC, SF_LOC
from chessbot.model import load_model
from chessbot.mcts_utils import MCTSTree

import chessbot.utils as cbu
from chessbot.utils import score_game_data, plot_training_progress, log_and_plot_sf
from chessbot.utils import evaluate_many_games, show_board, RateMeter

from chessbot.encoding import sf_top_moves_to_values
from stockfish import Stockfish

stockfish = Stockfish(path=SF_LOC)
stockfish.update_engine_parameters({"Threads": 4, "Contempt": 30})
stockfish.set_depth(5)

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"
VIS_DIR = "C:/Users/Bryan/Data/chessbot_data/visuals"


class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """
    # MCTS
    c_puct = 1.5
    virtual_loss = 1.0
    anytime_uniform_mix = 0.15
    endgame_uniform_mix = 0.15

    # Simulation schedule
    sims_target = 400
    sims_target_endgame = 600
    micro_batch_size = 8
    
    # Game stuff
    move_limit = 200
    material_diff_cutoff = 50
    material_diff_cutoff_span = 1000
    n_training_games = 3000
    restart_after_result = True
    play_vs_sf_prob = 2.0
    sf_depth = 5
    
    game_probs = {
        "pre_opened": 0.2, "random_init": 0.2,
        "random_middle_game": 0.2, "random_endgame": 0.2,
        "piece_odds": 0.1, "piece_training": 0.1}
    
    # boosts/penalize
    use_prior_boosts = True
    endgame_prior_adjustments = {
        "pawn_push":0.15, "capture":0.15,
        "repetition_penalty": 0.75, "gives_check": 0.1}
    
    # early stop
    es_min_sims = 128
    es_check_every = 4
    es_gap_frac = 0.80

    # TF
    training_queue_min = 2048
    vwq_blend = 0.5
    target_mean = 0.3
    draw_frac = 0.3
    factorized_bins = (64, 64, 6, 4)
    
    # files
    model_save_pattern = "conv_super_bootstrapped"
    sf_plot_path = os.path.join(VIS_DIR, "conv_super_bootstrapped_sf_eval")
    progress_plot_path = os.path.join(VIS_DIR, "conv_super_bootstrapped_progress.png")

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in dir(self)
            if not k.startswith("_") and not callable(getattr(self, k))
        }


class GameGenerator(object):
    """
    Curriculum game generator.
    Chooses among: pre-opened, random-init, middle-game, endgame slices.
    """
    def __init__(self, cfg=None):
        self.config = cfg or Config()
        self.game_types = list(self.config.game_probs.keys())

    def new_board(self, game_type=None):
        # sanity check
        if game_type is not None and game_type not in self.game_types:
            raise ValueError(
                f"Unknown game type {game_type}. "
                f"Pick one of {self.game_types}")

        # sample if not given
        if game_type is None:
            types, probs = zip(*self.config.game_probs.items())
            game_type = random.choices(types, weights=probs, k=1)[0]

        # dispatch
        if game_type == "pre_opened":
            board = cbu.get_pre_opened_game()
            meta = {"scenario": "pre_opened"}
            
        elif game_type == "random_init":
            plies = np.random.randint(0, 4)
            board = cbu.random_init(plies)
            meta = {"scenario": "random_init", "start_plies": plies}
            
        elif game_type == "random_middle_game":
            # just more plies of random_init to land mid-game
            plies = np.random.randint(20, 31)
            board = cbu.random_init(plies)
            meta = {"scenario": "random_middle_game", "start_plies": plies}
            
        elif game_type == "random_endgame":
            # you wrote random_board_setup / random_endgame_init
            pieces = np.random.randint(8, 14)
            wk = np.random.randint(0, 33)
            bk = np.random.randint(33, 64)
            board = cbu.random_board_setup(pieces, wk, bk, queens=False)
            meta = {"scenario": "random_endgame", "pieces": pieces}
            
        elif game_type == "piece_odds":
            board, meta = cbu.make_piece_odds_board()
            
        elif game_type == "piece_training":
            board, meta = cbu.make_piece_training_board()
            
        else:
            raise ValueError(f"unhandled game_type {game_type}")
            
        return board, meta


class ChessGame(object):
    def __init__(self, board=None, cfg=None):
        self.game_id = str(uuid.uuid4())
        self.config = cfg or Config()
        
        if board is None:
            gg = GameGenerator(cfg=self.config)
            board, meta = gg.new_board("piece_odds")
        else:
            meta = {"scenario": "user provided board"}
        
        self.board = board
        self.starting_fen = self.board.fen()
        self.moves_played = []
        self.meta = meta
        
        # determine if vs stockfish or full self-play
        self.vs_stockfish = False
        self.stockfish_is_white = None
        if np.random.uniform() < self.config.play_vs_sf_prob:
            self.vs_stockfish = True
            #self.stockfish_is_white = np.random.uniform() < 0.5
            self.stockfish_is_white = True
            if self.board.material_count() > 0:
                self.stockfish_is_white = False
        
        self.tree = MCTSTree(self.board, self.config)

        self.mat_adv_counter = 0
        self.outcome = None
        self.examples = [] 
        self.plies = 0
        self.vwq = None
        self.sf_eval = None
    
    def turn(self, return_bool=True):
        stm = self.board.side_to_move()
        if return_bool:
            return stm == 'w'
        else:
            return stm
    
    def is_stockfish_turn(self):
        return self.vs_stockfish and self.stockfish_is_white == self.turn()
    
    def get_stockfish_move(self):
        try:
            stockfish.set_fen_position(self.starting_fen)
            if self.moves_played:
                stockfish.make_moves_from_current_position(self.moves_played)
        except Exception as e:
            print(e)
            print("Setting stockfish from current position")
            stockfish.set_position(self.board.fen())
        
        # raw values for SF-returned moves
        sf_moves = stockfish.get_top_moves(1)    
        move = sf_moves[0]['Move']
        
        val_sf = sf_top_moves_to_values(sf_moves)[0]
        
        return move, val_sf
    
    def make_move_with_stockfish(self):
        """
        Stockfish plays one move. We record a supervised-style target where
        95% prob is on SF's chosen move, 5% is spread uniformly over the rest.
        The value target is SF's signed eval for the side to move (white-POV).
        """
        # no legal moves -> let terminal handler decide
        legal = self.board.legal_moves()
        if not legal:
            return self.check_for_terminal()
    
        # choose SF move + signed eval (white POV)
        mv, sf_v = self.get_stockfish_move()
        self.sf_eval = sf_v
        
        # build policy over ALL legal moves: 95% on mv, 5% over others
        n = len(legal)
        probs = np.zeros(n, dtype=np.float32)
        sel = legal.index(mv)
        if n == 1:
            probs[0] = 1.0
        else:
            probs[sel] = 0.95
            spill = 0.05 / float(n - 1)
            for i in range(n):
                if i != sel:
                    probs[i] = spill
    
        # create the example just like a MCTS move and send to queue
        self.append_factorized_example(ucis=legal, pi=probs, vwq=sf_v)
        
        # advance tree (pushes move) and reset
        self.tree.advance(self.board, mv)
        self.tree.reset_for_new_move()
        self.moves_played.append(mv)
        self.plies += 1
        return self.check_for_terminal()
    
    def make_move_from_tree(self):
        """
        Snapshot policy targets from root visits, then play best-by-visits.
        """
        root = self.tree.root()
        if not root.is_expanded:
            return False
    
        rows = self.tree.root_child_visits()  # [(uci, N)] sorted desc
        if not rows:
            return False
    
        ucis   = [u for u, _ in rows]
        visits = np.array([n for _, n in rows], dtype=np.float32)
        s = float(visits.sum())
        pi = (visits / s) if s > 0.0 else None
    
        vwq = self.tree.visit_weighted_Q()
        self.vwq = vwq
    
        if pi is not None:
            self.append_factorized_example(ucis=ucis, pi=pi, vwq=vwq)
    
        mv, _ = self.tree.best()
        if mv is None:
            return False
    
        self.tree.advance(self.board, mv)
        self.tree.reset_for_new_move()
        self.moves_played.append(mv)
        self.plies += 1
        return self.check_for_terminal()

    def append_factorized_example(self, ucis, pi, vwq):
        """
        Snapshot inputs and factorized policy targets for the given move dist.
        - ucis: list[str] legal moves (same order as probs)
        - probs: list/array of probabilities summing ~1
        - vwq: scalar value target (white-POV)
        """
        # labels per-legal
        fr, to, piece, promo = self.board.moves_to_labels(ucis=ucis)

        # allocate heads
        F, T, Kp, Kpr = self.config.factorized_bins
        from_m = np.zeros(F, dtype=np.float32)
        to_m   = np.zeros(T, dtype=np.float32)
        pc_m   = np.zeros(Kp, dtype=np.float32)
        pr_m   = np.zeros(Kpr, dtype=np.float32)

        # accumulate probs
        for i, p in enumerate(pi):
            from_m[fr[i]] += p
            to_m[to[i]]   += p
            pc_m[piece[i]]+= p
            pr_m[promo[i]]+= p

        # snapshot inputs and push example
        x = self.board.stacked_planes(5)
        self.examples.append(
            (x, {"from": from_m, "to": to_m, "piece": pc_m,
                 "promo": pr_m, "vwq": float(vwq)})
        )

    def check_for_terminal(self):
        reason, result = self.board.is_game_over()
        if reason != 'none':
            self.outcome = terminal_value_white_pov(self.board)
            return True
    
        mat_diff = self.board.material_count()
        if abs(mat_diff) >= self.config.material_diff_cutoff:
            self.mat_adv_counter += 1
        else:
            self.mat_adv_counter = 0
    
        if self.mat_adv_counter >= self.config.material_diff_cutoff_span:
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
                return True
            except:
                pass
           
        hs = self.board.history_size()
        if hs > self.config.move_limit:
            self.outcome = 0.0
            return True
        # if we made it here the game is active
        return False
    
    def show_board(self, flipped=False, sleep=0.0):
        sb = chess.Board(self.board.fen())
        show_board(sb, flipped=flipped, sleep=sleep)

    def show_top_moves(self, top_n=4, show_san=True):
        root = self.tree.root()
        c_puct = self.config.c_puct
    
        if not root.is_expanded:
            print("\nTop candidate moves:\n  (no children)")
            return None
    
        # timing
        start = getattr(self.tree, "_move_started_at", None)
        if start is None:
            start = _now()
            self.tree._move_started_at = start
        elapsed = _now() - start
    
        # C++ summaries
        avg_depth, max_depth = self.tree.depth_stats()
        details = self.tree.root_child_details()
    
        if not details:
            print("\nTop candidate moves:\n  (no children)")
            return None
    
        sims = int(getattr(self.tree, "sims_completed_this_move", 0))
        total_children = len(details)
        visited_children = sum([1 for cd in details if cd.N > 0])
    
        print(
            f"\nSearch summary: sims={sims}  time={elapsed:.2f}s  "
            f"avg_depth={avg_depth:.2f}  max_depth={max_depth}  "
            f"children_visited={visited_children}/{total_children}"
        )
    
        # sumN for U term
        sum_vloss = sum([cd.vloss for cd in details])
        sumN = max(1, root.N + sum_vloss)
    
        print("Top candidate moves:")
        for cd in details[:top_n]:
            U = c_puct * cd.prior * (sumN ** 0.5) / (1 + cd.N + cd.vloss)
            san = self.board.san(cd.uci) if show_san else "?"
            print(f"{cd.uci:<6} {('('+san+')'):<12}  visits={cd.N:<5}  "
                  f"P={cd.prior:>.3f}  Q={cd.Q:+.3f}  U={U:+.3f}")
    
        # chosen move + visit-weighted Q
        best_move_uci, _ = self.tree.best()
        if best_move_uci is None:
            return None
    
        best_san = self.board.san(best_move_uci) if show_san else best_move_uci
        # If vwq already computed elsewhere, prefer it; otherwise use the Q
        chosen = next((cd for cd in details if cd.uci == best_move_uci), None)
        v_mcts = self.vwq if self.vwq is not None else chosen.Q
        
        print(
            f"\nChosen move: {best_move_uci} ({best_san})  "
            f"(visits={(chosen.N if chosen else 0)}, "
            f"Q={(chosen.Q if chosen else 0.0):+.3f}, "
            f"visit-weighted Q={v_mcts:+.3f})"
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
        self.training_queue = []
        self.quick_cache = {}

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
        
        # update global stockfish
        stockfish.set_depth(self.config.sf_depth)
        
        self.mps = RateMeter("moves")
        self.lps = RateMeter("leafs")
        self._last_stats_log = 0.0

    def run(self):
        """
        Main loop. Each round: for each game either let SF move (if applicable)
        or run MCTS step (collect/predict/apply). Keeps central batching.
        """
        completed_games = 0
        
        while completed_games < self.config.n_training_games:
            if not self.active_games:
                break
    
            preds_batch = []
            finished = []
    
            for game in list(self.active_games):
                # optional UI on first game for visibility
                display_game = (game.game_id == self.active_games[0].game_id)
                
                # if its stockfish turn, let SF move and skip MCTS this ply
                if game.is_stockfish_turn():
                    sf_terminal = game.make_move_with_stockfish()
                    self.mps.tick(1)
                    
                    if display_game:
                        game.show_board()
                        b = game.board.clone()
                        last_move = b.history_uci()[-1]
                        b.unmake()
                        last_san = b.san(last_move)
                        print(f"Stockfish plays {last_san} ",
                              f"Current stockfish eval {np.round(game.sf_eval, 3)}")
                        
                    if sf_terminal:
                        completed_games += 1
                        self.finalize_game_data(game)
                        self.maybe_log_results()
                        finished.append(game.game_id)
                    # just pass until the next turn
                    continue
    
                # resolve any pending predictions (from last batch) for this game
                game.tree.resolve_awaiting(game.board, self.quick_cache)
    
                # if this game has reached its local sim budget, make the move
                if game.tree.stop_simulating():
                    if display_game:
                        game.show_top_moves()
    
                    # bot plays from tree
                    mcts_terminal = game.make_move_from_tree()
                    self.mps.tick(1)
                    
                    if display_game:
                        game.show_board()
                    
                    # terminal after bot move?
                    if mcts_terminal:
                        completed_games += 1
                        self.finalize_game_data(game)
                        self.maybe_log_results()
                        finished.append(game.game_id)
                        continue
    
                # otherwise, collect up to micro_batch_size leaves for this game
                n_collected, tries = 0, 0
                while n_collected < self.config.micro_batch_size:
                    leaf_req = game.tree.collect_one_leaf(game.board)
                    if leaf_req is None:
                        tries += 1
                        if tries > 4:
                            break
                    else:
                        n_collected += 1
                        self.lps.tick(1)
                        preds_batch.append(leaf_req)
            
            # predict on the central batch (also updates caches)
            if preds_batch:
                self.format_and_predict(preds_batch)
    
            # resolve fresh predictions back into each game tree
            applied = 0
            for game in self.active_games:
                applied += game.tree.resolve_awaiting(game.board, self.quick_cache)
    
            # if nothing was applied, clear the quick cache to keep it light
            if applied == 0:
                self.quick_cache = {}
    
            # remove finished games and respawn if configured
            if finished:
                self.active_games = [
                    g for g in self.active_games if g.game_id not in finished]
                
                if self.config.restart_after_result:
                    self.active_games += [ChessGame() for _ in range(len(finished))]
        return

    def format_and_predict(self, preds_batch):
        """
        Take leaf requests from all games, run one model call, and write
        results into quick_cache
        Each req must have: 'enc', 'cache_key'.
        """
        if not preds_batch:
            return

        X = np.asarray([r["enc"] for r in preds_batch], dtype=np.float32)

        out = self.model.predict(X, batch_size=1024, verbose=0)

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
            "ts": _now(), "game_id": game.game_id, "start_fen": game.starting_fen,
            "result": game.outcome if game.outcome is not None else 0.0,
            "plies": int(game.plies), "history_uci": game.board.history_uci(),
            "moves_played": game.moves_played, "vs_stockfish": game.vs_stockfish,
            "stockfish_color": game.stockfish_is_white}
        
        res.update(self.config.to_dict())
        res.update(game.meta)
        
        self.pre_game_data.append(res)

        z = float(game.outcome if game.outcome is not None else 0.0)
        for x, heads in game.examples:
            vwq = heads.get("vwq", 0.0)
            self.training_queue.append((x, heads, z, vwq))
        game.examples = []

        if len(self.training_queue) >= self.config.training_queue_min:
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
            "best_promo": np.asarray(Y_promo, dtype=np.float32)}
    
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
    
        d = []
        for g in self.pre_game_data:
            # dont eval on sf moves
            if g['vs_stockfish']:
                continue
            
            d.append({'moves': g['moves_played'], "start_fen": g['start_fen']})
        
        if d:
            print(f"Analyzing last {len(d)} games with Stockfish(d=8)...")
            sf_analysis = evaluate_many_games(d, depth=8, workers=10, mate_cp=1500)
            self.intra_training_summaries[self.n_retrains] = sf_analysis
            log_and_plot_sf(
                self.intra_training_summaries, save_path=self.config.sf_plot_path
            )
        
        # archive pre-game data
        self.game_data += self.pre_game_data
        self.pre_game_data = []   # <- small typo fix from pre_data_data
    
        # fit (only value head weighted)
        self.model.fit(
            X, Y, epochs=1, batch_size=512, verbose=0, sample_weight={"value": w}
        )
        
        # save new model
        outfile = self.config.model_save_pattern + "_latest.h5"
        model_out = os.path.join(MODEL_DIR, outfile)
        self.model.save(model_out)
        
        # clear training queue and bump retrain counter
        self.training_queue = []
        
        self.n_retrains += 1
        if self.n_retrains % 10 == 0:
            outfile = self.config.model_save_pattern + f"_v{self.n_retrains}.h5"
            model_out = os.path.join(MODEL_DIR, outfile)
            self.model.save(model_out)  
    
    def maybe_log_results(self, every_sec=120.0, window=500):
        """
        Periodically print overall stats PLUS a breakdown by (scenario, sf_color).
        - window: only aggregate last N finished games to keep it cheap.
        """
        def sf_bucket(vs_sf, sf_is_white):
            if not vs_sf:
                return "none"
            return "white" if sf_is_white else "black"
    
        now = _now()
        if now - self._last_stats_log < every_sec:
            return
        self._last_stats_log = now
    
        # overall
        avg_moves = (self.total_plies / max(1, self.games_finished))
        print("~" * 60)
        print(
            f"[stats] finished={self.games_finished}  "
            f"W/L/D={self.white_wins}/{self.black_wins}/{self.draws}  "
            f"avg_len={avg_moves:.1f} moves  |  "
            f"mps={self.mps.rate():.1f}  lps={self.lps.rate():.1f}")
        print("-" * 60)
        
        # recent finished games (use archived + pending pre_game_data)
        recent = (self.game_data + self.pre_game_data)[-window:]
        if not recent:
            print("(no recent games to break down)")
            print("~" * 60)
            return
    
        # nice printout
        cbu.print_recent_summary(recent, window=window)
        print(
            f"Length of training queue: {len(self.training_queue)} ",
            f"Number of retrains: {self.n_retrains}")

#%%
if __name__ == '__main__':
    try_latest = os.path.join(MODEL_DIR, "conv_super_bootstrapped_latest.h5")
    fallback = os.path.join(MODEL_DIR + "/conv_model_big_v1000.h5")
    model_file = try_latest if os.path.exists(try_latest) else fallback
    
    print(f"Loading model from {model_file}")
    model = load_model(model_file)
    
    looper = GameLooper(games=32, model=model, cfg=Config())
    looper.run()
