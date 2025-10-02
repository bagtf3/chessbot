import uuid, os
import pathlib, json
from time import time as _now

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from collections import deque 
import random
import chess
import chess.syzygy

from pyfastchess import terminal_value_white_pov

from chessbot import ENDGAME_LOC, SF_LOC
from chessbot.model import load_model
from chessbot.mcts_utils import MCTSTree, LRUCache

import chessbot.utils as cbu
from chessbot.utils import (
    rnd, score_game_data, plot_training_progress, show_board, RateMeter
)

from chessbot.encoding import score_to_cp_white


class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """

    # files
    run_tag = "conv_1000_selfplay_phase2"
    selfplay_dir =  "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/"
    init_model = "C:/Users/Bryan\Data/chessbot_data/selfplay_runs/conv_1000_selfplay/conv_1000_selfplay_model.h5"
    # MCTS
    c_puct = 1.25
    anytime_uniform_mix = 0.35
    endgame_uniform_mix = 0.35
    opponent_uniform_mix = 0.5

    # Simulation schedule
    sims_target = 1200
    micro_batch_size = 12

    # early stop
    es_min_sims = 600
    es_check_every = 24
    es_gap_frac = 0.7
    es_top_node_frac = 0.75
    
    # Q-override selection
    use_q_override = True
    q_override_vis_ratio = 0.80
    q_override_q_margin = 0.1
    q_override_min_vis = 100
    q_override_top_k = 3
    
    # Game stuff
    games_at_once = 100
    n_training_games = 500
    lru_cache_size = 500_000
    
    move_limit = 160
    material_diff_cutoff = 15
    material_diff_cutoff_span = 20

    play_vs_sf_prob = 0.5
    sf_depth = 6
    
    game_probs = {
        "pre_opened": 0.2, "random_init": 0.2,
        "random_middle_game": 0.2, "random_endgame": 0.1,
        "piece_odds": 0.2, "piece_training": 0.1
    }
    
    # boosts/penalize
    use_prior_boosts = True
    prior_clip_max = 0.35
    prior_clip_min = 0.05
    endgame_prior_adjustments = {
        "pawn_push":0.15, "capture":0.15, "repetition_penalty": 0.05
    }
    
    anytime_prior_adjustments = {"gives_check": 0.1, "repetition_penalty": 0.05}

    # TF
    training_queue_min = 4096
    vwq_blend = 0.5
    use_vwq_alpha_taper = True
    target_mean = 0.5
    draw_frac = 0.5
    factorized_bins = (64, 64, 6, 4)

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in dir(self)
            if not k.startswith("_") and not callable(getattr(self, k))
        }
    
    def update(self, mapping=None, **kwargs):
        if mapping is not None:
            try:
                items = mapping.items()
            except AttributeError:
                items = mapping
            for k, v in items:
                if not hasattr(self, k):
                    raise AttributeError(f"Unknown config key: {k}")
                setattr(self, k, v)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown config key: {k}")
            setattr(self, k, v)


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
            board, meta = gg.new_board()
        else:
            meta = {"scenario": "user provided board"}
        
        self.board = board
        self.starting_fen = self.board.fen()
        self.moves_played = []
        self.tree_data = {}
        self.meta = meta
        
        # determine if vs stockfish or full self-play
        self.vs_stockfish = False
        self.stockfish_is_white = None
        if np.random.uniform() < self.config.play_vs_sf_prob:
            self.vs_stockfish = True
            self.stockfish_is_white = np.random.uniform() < 0.5
        
        self.meta['stockfish_is_white'] = self.stockfish_is_white
        
        self.tree = MCTSTree(self.board, self.config)
        self.mat_adv_counter = 0
        self.vwq_adv_counter = 0
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
        return self.vs_stockfish and (self.stockfish_is_white == self.turn())
    
    def get_stockfish_move(self, eng):
        b = chess.Board(self.board.fen())
        info = eng.analyse(b, chess.engine.Limit(depth=6), info=chess.engine.INFO_ALL)
        val_sf = score_to_cp_white(info['score'])
        move = str(info['pv'][0])
        
        return move, val_sf
    
    def push_move(self, mv):
        # collect search data then push and update
        
        try:
            self.collect_tree_search_data(mv)
        except Exception as e:
            print(e)
        
        # advance tree (pushes move) and reset
        self.tree.advance(self.board, mv)
        self.tree.reset_for_new_move()
        self.moves_played.append(mv)
        self.plies += 1
        return self.check_for_terminal()
    
    def collect_tree_search_data(self, mv):
        root = self.tree.root()
        if not root.is_expanded:
            return
        
        c_puct = self.config.c_puct
        
        # timing
        start = getattr(self.tree, "_move_started_at", None)
        if start is None:
            start = _now()
            self.tree._move_started_at = start
        elapsed = _now() - start
    
        # C++ summaries
        avg_depth, max_depth = self.tree.depth_stats()
        details = self.tree.root_child_details()
        if details is None:
            return
    
        sims = int(getattr(self.tree, "sims_completed_this_move", 0))
        total_children = len(details)
        visited_children = sum([1 for cd in details if cd.N > 0])
    
        data = {
            "sims": sims, "time": rnd(elapsed, 3),
            "avg_depth": rnd(avg_depth, 2), "max_depth": max_depth,
            "children_visited": visited_children,
            "total_children": total_children,
            "visit_weighted_Q": rnd(self.tree.visit_weighted_Q(), 4)
        }
    
        # sumN for U term (no vloss in the new flow)
        sumN = max(1, root.N)

        candidate_moves = []
        for cd in details:
            U = c_puct * cd.prior * (sumN ** 0.5) / (1 + cd.N)
            cm = {
                "uci": cd.uci, "san": self.board.san(cd.uci),
                "visits": cd.N, "P": rnd(cd.prior, 4), "Q": rnd(cd.Q, 4),
                "U": rnd(U, 4), "is_terminal": cd.is_terminal,
                "vprime_visits": cd.vprime_visits
            }
            
            candidate_moves.append(cm)
        data["candidate_moves"] = candidate_moves

        self.tree_data[self.plies] = data

        
    def make_move_with_stockfish(self, eng):
        """
        Stockfish plays one move. We record a supervised-style target where
        60% prob is on SF's chosen move, 50% is spread uniformly over the rest.
        The value target is SF's signed eval for the side to move (white-POV).
        """
        # no legal moves -> let terminal handler decide
        legal = self.board.legal_moves()
        if not legal:
            return self.check_for_terminal()
    
        # choose SF move + signed eval (white POV)
        mv, sf_v = self.get_stockfish_move(eng)
        self.sf_eval = sf_v
        
        # build policy over ALL legal moves: 60% on mv, 40% over others
        n = len(legal)
        probs = np.zeros(n, dtype=np.float32)
        sel = legal.index(mv)
        if n == 1:
            probs[0] = 1.0
        else:
            probs[sel] = 0.60
            spill = 0.40 / float(n - 1)
            for i in range(n):
                if i != sel:
                    probs[i] = spill
    
        # create the example just like a MCTS move and send to queue
        self.append_factorized_example(ucis=legal, pi=probs, vwq=sf_v)
        return self.push_move(mv)
    
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
        
        return self.push_move(mv)
        
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
                 "promo": pr_m, "vwq": vwq, "ply": self.plies}))

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


class GameLooper(object):
    """
    Orchestrates N games concurrently, central batching, caches, and training.
    """
    def __init__(self, games, model, cfg):
        self.config = cfg or Config()
        self.active_games = []

        if isinstance(games, int):
            # generate unique starting games    
            starting_fens = set()
            while len(self.active_games) < games:
                cg = ChessGame()
                if cg.starting_fen in starting_fens:
                    continue

                starting_fens.add(cg.starting_fen)
                self.active_games.append(cg)
        else:
            # assumes these are pre-loaded Game objects in a list
            self.active_games = games
        
        self.model = model
        self.training_queue = []
        self.stats_window = self.config.n_training_games
        self.recent_games = deque(maxlen=self.stats_window)

        self.games_finished = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_plies = 0
        self.n_retrains = 0
        self.all_evals = pd.DataFrame()
        self.clear_lru_cache = False
        
        self.mps = RateMeter("moves")
        self.lps = RateMeter("leafs")
        self._last_stats_log = 0.0

    def run(self):
        """
        Main loop. Each round: for each game either let SF move (if applicable)
        or run MCTS step (collect/predict/apply). Keeps central batching.
        """
        completed_games = 0
        mbs = self.config.micro_batch_size
        try_break = int(2.5 * mbs)
        lpb = []
        mps, lps = self.mps, self.lps
        lru = LRUCache(self.config.lru_cache_size)
        counts = []
        with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
            while completed_games < self.config.n_training_games:
                if not self.active_games:
                    break
        
                preds_batch = []
                finished = []
                for game in list(self.active_games):
                    # if its stockfish turn, let SF move and skip MCTS this ply
                    if game.is_stockfish_turn():
                        sf_terminal = game.make_move_with_stockfish(eng)
                        mps.tick(1)
                            
                        if sf_terminal:
                            completed_games += 1
                            self.finalize_game_data(game)
                            self.maybe_log_results()
                            finished.append(game.game_id)
                            # pass until the next turn
                            continue
        
                    # if this game has reached its local sim budget, make the move
                    if game.tree.stop_simulating():
                        # bot plays from tree
                        mcts_terminal = game.make_move_from_tree()
                        mps.tick(1)
                        
                        # terminal after bot move?
                        if mcts_terminal:
                            completed_games += 1
                            self.finalize_game_data(game)
                            self.maybe_log_results()
                            finished.append(game.game_id)
                            continue
                        
                        # if its stockfish turn, dont do any sims
                        if game.is_stockfish_turn():
                            continue
        
                    # otherwise, collect up to micro_batch_size leaves for this game
                    n_collected, tries = 0, 0
                    bonus_hit, terminal = 0, 0
                    try_stop, collect_stop = 0, 0
                    while True: #n_collected < mbs:
                        leaf_req = game.tree.collect_one_leaf(lru)
                        if leaf_req is None:
                            # this shouldnt really happen
                            tries += 1
                        elif leaf_req["already_applied"]:
                            lps.tick(1)
                            tries += 1
                            if leaf_req['terminal']:
                                terminal += 1
                            else:
                                bonus_hit += 1
                        else:    
                            n_collected += 1 
                            lps.tick(1)
                            preds_batch.append(leaf_req)
                        # so we dont spin forever
                        if tries >= try_break:
                            try_stop += 1
                            break
                        if n_collected >= mbs:
                            collect_stop += 1
                            break
    
                    counts.append([n_collected, tries, terminal, bonus_hit, try_stop, collect_stop])
                # predict on the central batch (also updates caches)
                logged = self.maybe_log_results()
                if logged:
                    lru.stats()
                    # --- quick loop stats (temporary) ---
                    if counts:
                        n_groups = len(counts)
                        s_collected   = sum([r[0] for r in counts])
                        s_tries       = sum([r[1] for r in counts])
                        s_terminals   = sum([r[2] for r in counts])
                        s_bonus_hits  = sum([r[3] for r in counts])
                        s_try_stops   = sum([r[4] for r in counts])
                        s_collect_stops = sum([r[5] for r in counts])
    
                        avg_collected = s_collected / n_groups
                        avg_tries     = s_tries / n_groups
                        fill_ratio    = s_collected / max(1, n_groups * mbs)
                        hit_ratio       = s_bonus_hits / max(1, s_tries)
                        terminal_ratio  = s_terminals  / max(1, s_tries)
                        print("------------------------------------------------------------")
                        print("Loop stats"
                            f" | groups={n_groups}  mbs={mbs}"
                            f" | avg_collected={avg_collected:.2f}  avg_tries={avg_tries:.2f}"
                            f" | fill_ratio={fill_ratio:.3f} ")
                        print(f"Hits: bonus={s_bonus_hits} ({hit_ratio:.3%})"
                            f" | terminals={s_terminals} ({terminal_ratio:.3%})")
                        print(f"Stops: try_breaks={s_try_stops}  collect_breaks={s_collect_stops}")
                        if lpb:
                            print(f"Avg len of preds_batch {np.mean(lpb):.3f}")
                        avgply = np.mean([g.plies for g in self.active_games])
                        print(f"Avg ply of active_games {avgply:<.3f}")
                        print("------------------------------------------------------------")
                    counts = []
                    lpb = []
                if preds_batch:
                    self.format_and_predict(preds_batch, lru)
                    lpb.append(len(preds_batch))
        
                # resolve fresh predictions back into each game tree
                still_waiting = 0
                for game in self.active_games:
                    still_waiting += game.tree.resolve_awaiting(lru)
                
                if self.clear_lru_cache:
                    if not logged:
                        lru.stats()
                    lru.clear()
                    self.clear_lru_cache = False
    
                # remove any finished games
                self.active_games = [
                    g for g in self.active_games if g.game_id not in finished
                ]
                    
                if completed_games + len(self.active_games) < self.config.n_training_games:
                    while len(self.active_games) < self.config.games_at_once:
                        self.active_games.append(ChessGame())
        
        # final report
        self.maybe_log_results(force=True)
        return

    def format_and_predict(self, preds_batch, lru):
        """
        Take leaf requests from all games, run one model call, and write
        results into cache
        Each req must have: 'enc', 'cache_key'.
        """
        if not preds_batch:
            return

        X = np.asarray([r["enc"] for r in preds_batch], dtype=np.float32)
        out = self.model.predict(X, batch_size=1200, verbose=0)

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
            lru[key] = out_i

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
        
        mem_summary = {
            "ts": _now(),
            "game_id": game.game_id,
            "scenario": game.meta.get("scenario", ""),
            "plies": int(game.plies),
            "result": game.outcome or 0.0,
            "vs_stockfish": game.vs_stockfish,
            "stockfish_color": game.stockfish_is_white
        }
        # small in-memory record for recent prints only
        self.recent_games.append(mem_summary)

        # on-disk record (full)
        res = {
            "start_fen": game.starting_fen,
            "history_uci": game.board.history_uci(),
            "moves_played": game.moves_played,
            "model_epoch": self.n_retrains,
            "n_retrains": self.n_retrains
        }
        res.update(mem_summary)
        res.update(self.config.to_dict())
        res.update(game.meta)

        # attach tree search data to disk record
        res["tree_search_data"] = game.tree_data

        # save per-game JSON
        out_file = os.path.join(self.config.game_dir, game.game_id + "_log.json")
        out_path = pathlib.Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        # update JSONL index (small)
        keep = [
            "game_id", "ts", "n_retrains", "scenario", "plies",
            "vs_stockfish", "stockfish_color", "result"
        ]
        new_idx = {k: res[k] for k in keep}
        new_idx["json_file"] = out_file
        
        new_idx['beat_sf'] = False
        if new_idx['vs_stockfish']:
            game_result = new_idx['result']
            if game_result > 0 and not game.stockfish_is_white:
                new_idx['beat_sf'] = True
            elif game_result < 0 and game.stockfish_is_white:
                new_idx['beat_sf'] = True
            else:
                new_idx['beat_sf'] = False
        
        # append to JSONL index (create parent dirs if needed)
        idx_file = self.config.game_index_file
        os.makedirs(os.path.dirname(idx_file), exist_ok=True)
        with open(idx_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_idx, ensure_ascii=False) + "\n")
        
        z = game.outcome if game.outcome is not None else 0.0
        gl = game.plies # game length
        for x, heads in game.examples:
            vwq = heads.get("vwq", 0.0)
            ply = heads.get("ply", 0.0)
            taper = ply / gl if gl else 0.0
            self.training_queue.append((x, heads, z, vwq, taper))
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
        X, Y_from, Y_to, Y_piece, Y_promo = [], [], [], [], []
        Z, Vwq, taper_list = [], [], []
        for x, heads, z, vwq, taper in self.training_queue:
            X.append(x)
            Y_from.append(heads["from"])
            Y_to.append(heads["to"])
            Y_piece.append(heads["piece"])
            Y_promo.append(heads["promo"])
            Z.append(z)
            Vwq.append(vwq)
            taper_list.append(taper)

        Z = np.asarray(Z_list, dtype=np.float32)
        Vwq = np.asarray(Vwq_list, dtype=np.float32)
        taper_arr = np.asarray(taper_list, dtype=np.float32)

        # blend outcome with visit-weighted Q
        alpha = getattr(self.config, "vwq_blend", 0.0)
        use_taper = getattr(self.config, "use_vwq_alpha_taper", False)
        alpha_eff = alpha * taper_arr if use_taper else alpha
        Y_value = (1.0 - alpha_eff) * Z + alpha_eff * Vwq
        
        # assemble training dict
        X = np.asarray(X, dtype=np.float32)
        Y = {
            "value": Y_value.astype(np.float32),
            "best_from": np.asarray(Y_from, dtype=np.float32),
            "best_to": np.asarray(Y_to, dtype=np.float32),
            "best_piece": np.asarray(Y_piece, dtype=np.float32),
            "best_promo": np.asarray(Y_promo, dtype=np.float32)
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
            w = np.where(Z>0, w_pos, np.where(Z<0, w_neg, w_draw)).astype(np.float32)
        else:
            w = np.ones_like(Z, dtype=np.float32)
    
        # scale weights to target mean
        mean_w = float(w.mean()) if w.size else 1.0
        if mean_w > 0:
            w *= (target_mean / mean_w)
        else:
            w[:] = target_mean
    
        # evaluation and logging
        plt_file = os.path.join(self.config.run_dir, "true_vs_pred_plot_latest.png")
        eval_df = score_game_data(self.model, X, Y, save_path=plt_file)
        eval_df['model_epoch'] = self.n_retrains
        self.all_evals = pd.concat([self.all_evals, eval_df])
        self.all_evals.round(3).to_csv(self.config.progress_csv_path, index=False)
        
        if len(self.all_evals) and len(self.all_evals) % 4 == 0:
            plt_file = self.config.progress_plot_path
            plot_training_progress(self.all_evals, save_path=plt_file)

        # build sample weights for each head
        even_weights = np.ones_like(w)
        s_wts = {
            'value': w, "best_from": even_weights, "best_to": even_weights,
            "best_piece": even_weights, 'best_promo': even_weights
        }

        # fit and  save new model
        self.model.fit(X, Y, epochs=1, batch_size=512, verbose=0, sample_weight=s_wts)        
        self.model.save(self.config.model_path)
        
        # clear training queue, clear LRU and bump retrain counter
        self.training_queue = []
        self.n_retrains += 1
        self.clear_lru_cache = True
    
    def maybe_log_results(self, every_sec=60.0, window=500, force=False):
        def sf_bucket(vs_sf, sf_is_white):
            if not vs_sf:
                return "none"
            return "white" if sf_is_white else "black"

        now = _now()
        if not force and (now - self._last_stats_log < every_sec):
            return False
        self._last_stats_log = now

        avg_moves = (self.total_plies / max(1, self.games_finished))
        print("~" * 60)
        print(
            f"[stats] finished={self.games_finished}  "
            f"W/L/D={self.white_wins}/{self.black_wins}/{self.draws}  "
            f"avg_len={avg_moves:.1f} moves  |  "
            f"mps={self.mps.rate():.1f}  lps={self.lps.rate():.1f}"
        )
        print("-" * 60)

        recent = list(self.recent_games)[-min(window, len(self.recent_games)):]
        if not recent:
            print("(no recent games to break down)")
            print("~" * 60)
            return True

        # re-use your existing pretty printer
        cbu.print_recent_summary(recent, window=window)
        print(
            f"Length of training queue: {len(self.training_queue)} ",
            f"Number of retrains: {self.n_retrains}\n"
        )
        return True


def init_selfplay():
    # file structure first
    config = Config()
    config.selfplay_dir
    run_dir = os.path.join(config.selfplay_dir, config.run_tag)
    Config.run_dir = run_dir
    
    game_dir = os.path.join(run_dir, "game_logs")
    Config.game_dir = game_dir
    
    Config.game_index_file = os.path.join(run_dir, "game_index.json")
    Config.progress_csv_path = os.path.join(run_dir, "eval_progress.csv")
    Config.progress_plot_path = os.path.join(run_dir, "eval_progress.png")
    
    for d in [run_dir, game_dir]:
        os.makedirs(d, exist_ok=True)
    config = Config()
    
    # deal with model file structure first
    model_name = config.run_tag + "_model.h5"
    model_path = os.path.join(run_dir, model_name)
    Config.model_path = model_path
    
    if os.path.exists(model_path):
        print(f"Loading {model_name}")
        model = load_model(model_path)
    else:
        print(f"Loading {config.init_model}")
        model = load_model(config.init_model)
        model.save(model_path)
    config = Config()
    
    return model, config


def main():
    model, config = init_selfplay()
    looper = GameLooper(games=config.games_at_once, model=model, cfg=Config())
    
    # infer the number of trainings already done from existing files
    if os.path.exists(config.progress_csv_path):
        try:
            progress_df = pd.read_csv(config.progress_csv_path)
            n_retrains = len(progress_df)
            looper.n_retrains = n_retrains
            looper.all_evals = progress_df
        except Exception as e:
            print(e)
    
    looper.run()

if __name__ == '__main__':
    main()
    main()
    main()
    main()
