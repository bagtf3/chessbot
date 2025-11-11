import uuid, os, pickle
import pathlib, json
import time
_now = time.time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import chess, chess.syzygy

from pyfastchess import terminal_value_white_pov, raw_cache_bulk_insert
from pyfastchess import raw_cache_clear, priors_cache_clear, priors_cache_stats

from chessbot import ENDGAME_LOC, SF_LOC
from chessbot.model import load_model, save_model, make_conv_infer
from chessbot.mcts_utils import MCTSTree

from chessbot.config import Config

import chessbot.utils as cbu
from chessbot.utils import rnd, RateMeter, softmax, GameGenerator, score_to_value_stm_pov
from chessbot.review import start_post_hoc_server, stop_post_hoc_server, make_fake_visits


class ChessGame(object):
    def __init__(self, board, meta, cfg):
        self.game_id = str(uuid.uuid4())
        self.config = cfg
        self.started_at = _now()

        self.board = board
        self.starting_fen = self.board.fen()
        self.meta = meta
        
        self.vs_stockfish = meta['vs_stockfish']
        self.stockfish_is_white = meta['stockfish_is_white']
        self.tree = MCTSTree(self.board, self.config)
        self.tree_data = {}
        self.moves_played = []
        
        self.mat_adv_counter = 0
        self.vwq_adv_counter = 0
        self.outcome = None
        self.examples = []
        self.plies = 0
        self.vwq = None
        self.sf_eval = None
    
    def turn(self, return_bool=True):
        stm = self.tree.root().board.side_to_move()
        return stm == 'w' if return_bool else stm
    
    def is_stockfish_turn(self):
        return self.vs_stockfish and (self.stockfish_is_white == self.turn())
    
    def get_stockfish_move(self, eng):
        val_sf, best_move = cbu.sf_eval(
            self.tree.root().board, score_fn=score_to_value_stm_pov,
            depth=self.config.sf_depth, engine=eng
        )

        return best_move, val_sf
    
    def push_move(self, mv):
        # collect search data then push and update
        self.collect_tree_search_data(mv)

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
        start = self.tree._move_started_at
        if start is None:
            start = _now()
            self.tree._move_started_at = start
        elapsed = _now() - start
    
        # C++ summaries
        avg_depth, max_depth = self.tree.depth_stats()
        details = self.tree.root_child_details()
        if details is None:
            return
        
        sims = int(self.tree.sims_completed_this_move)
        total_children = len(details)
        visited_children = sum([1 for cd in details if cd.N > 0])

        data = {
            "sims": sims, "time": rnd(elapsed, 3),
            "avg_depth": rnd(avg_depth, 2), "max_depth": max_depth,
            "children_visited": visited_children,
            "total_children": total_children,
            "visit_weighted_Q": rnd(self.tree.visit_weighted_Q(), 4),
            "stop_reason": self.tree.sim_stop_reason
        }
        # sumN for U term
        sumN = max(1, root.N)

        candidate_moves = []
        for cd in details:
            U = c_puct * cd.prior * (sumN ** 0.5) / (1 + cd.N)
            cm = {
                "uci": cd.uci, "visits": cd.N,
                "P": rnd(cd.prior, 4), "Q": rnd(cd.Q, 4), "U": rnd(U, 4),
                "is_terminal": cd.is_terminal, "vprime_visits": cd.vprime_visits
            }
            
            candidate_moves.append(cm)
        data["candidate_moves"] = candidate_moves

        # fast PV via C++
        pv = []
        pv_items = self.tree.principal_variation(24)
        for x in pv_items:
            pv.append({
                "uci": x.uci, "visits": int(x.visits),
                "P": rnd(x.P, 4), "Q": rnd(x.Q, 4),
            })
        
        # attach PV snapshot (may be empty if no deeper visited chain exists)
        data["pv"] = pv
        self.tree_data[self.plies] = data
    
    def make_move_with_stockfish(self, eng):
        """
        Stockfish plays one move. When the MCTS root has enough sims and the
        tree agrees (or nearly agrees) with Stockfish, use the tree's visit
        counts as the policy target (optionally bumping SF's visits so it's #1).
        Otherwise fall back to the old 60/40 supervised target.
        """
        
        legal = self.board.legal_moves()
        if not legal:
            return self.check_for_terminal()

        # get SF move + signed eval (white POV)
        mv, sf_v = self.get_stockfish_move(eng)
        
        self.sf_eval = sf_v

        # gather root visit info from the tree (list sorted desc by visits)
        rows = self.tree.root_child_visits()  # [(uci, N)] sorted desc
        total_visits = sum([n for _, n in rows]) if rows else 0

        use_tree_visits = False
        visit_map = None

        if rows and total_visits >= 20:
            # map uci -> visits for quick lookup
            visit_map = {u: n for u, n in rows}
            most_visited_uci, max_visits = rows[0]
            sf_visits = visit_map.get(mv, 0.0)

            # case A: tree already picks SF move as top choice
            if most_visited_uci == mv:
                use_tree_visits = True

            # case B: SF move is close to top -> use tree visits but nudge SF to top
            elif sf_visits >= 0.5 * max_visits:
                use_tree_visits = True
                # make SF strictly first by setting its visits > max_visits
                visit_map[mv] = max_visits + 1

        if use_tree_visits and visit_map is not None:
            visits = [max(1, visit_map.get(u, 1.0)) for u in legal]
            ucis = legal

        else:
            # make_fake_visits returns [[uci,count], ...]
            raw = make_fake_visits(mv, legal, ratio_best=60)
            ucis   = [x[0] for x in raw]
            visits = [x[1] for x in raw]
        
        s = sum(visits)
        pi = np.array([v / s for v in visits], dtype=np.float32)
        self.append_flat_policy_example(ucis=ucis, pi=pi, vwq=sf_v, turn=self.turn())

        # finally play SF's move on the board and advance tree
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
        s = visits.sum()
        pi = (visits / s) if s > 0.0 else np.zeros_like(visits)
        vwq = self.tree.visit_weighted_Q()

        # tree is white POV, we want STM-POV so we flip here
        vwq = vwq if self.turn() else -vwq
    
        if pi is not None:
            self.append_flat_policy_example(ucis=ucis, pi=pi, vwq=vwq, turn=self.turn())

        mv, _ = self.tree.best()
        if mv is None:
            return False
        return self.push_move(mv)
    
    def append_flat_policy_example(self, ucis, pi, vwq, turn):
        """
        Snapshot inputs and a flat 4096-length policy vector for training.
        - ucis: list[str] legal moves (same order as probs)
        - pi:  list/array of probs (sum ~= 1)
        - vwq: scalar value target (white-POV)
        """
        
        # get indices from C++
        indices = self.tree.root().board.moves_to_indices(ucis)  # list of int (0..4095)
        policy = np.zeros(64 * 64, dtype=np.float32)
        
        # accumulate probs into flattened policy
        for idx, p in zip(indices, pi):
            policy[idx] += p

        # snapshot inputs and push example
        x = self.board.encode_64_tokens()
        mask = self.board.legal_move_mask()
        self.examples.append((x, mask, policy, vwq, self.plies, turn))

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


class GameLooper(object):
    """
    Orchestrates N games concurrently, central batching, caches, and training.
    """
    def __init__(self, model, cfg):
        self.config = cfg or Config()
        self.game_gen = GameGenerator(self.config)
        self.games_finished = 0
        self.active_games = []
        self.sf_count = 0
        self.fill_active_games()

        self.model = model

        cfg = self.config
        self.infer = make_conv_infer(
            self.model, max_bs=cfg.fwd_batch,
            min_p=cfg.prior_clip_min, max_p=cfg.prior_clip_max, temp=1
        )

        self.training_queue = []
        self.recent_games = []

        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_plies = 0
        self.n_retrains = 0
        self.all_evals = pd.DataFrame()
        self.clear_cache = False
        
        self._run_start = _now()
        self.mps = RateMeter("moves")
        self.lps = RateMeter("leafs")
        self._last_stats_log = 0.0

        self.moves_played = 0
        self.sims_done_total = 0

    def fill_active_games(self):
        fens_seen = set()
        cfg = self.config
        needed = cfg.n_training_games - self.games_finished - len(self.active_games)
        if needed <= 0:
            return

        # add games w.r.t. num needed and games_at_once
        for _ in range(needed):
            if len(self.active_games) >= cfg.games_at_once:
                return

            # this has game probs from the config
            board, meta = self.game_gen.new_board()
            fen = board.fen()
            while fen in fens_seen:
                board, meta = self.game_gen.new_board()
                fen = board.fen()

            # add new unique fen
            fens_seen.add(fen)

            meta['vs_stockfish'] = False
            meta['stockfish_is_white'] = False
            if np.random.uniform() <= cfg.play_vs_sf_prob:
                self.sf_count += 1
                meta['vs_stockfish'] = True
            
                # alternate sf color to balance white and black
                meta['stockfish_is_white'] = bool(self.sf_count % 2)
            
            cg = ChessGame(board=board, meta=meta, cfg=self.config)
            self.active_games.append(cg)
    
    def run(self):
        """
        Main loop. Each round: for each game either let SF move (if applicable)
        or run MCTS step (collect/predict/apply).
        """
        # reset
        self.games_finished = 0

        cfg = self.config
        mbs = cfg.micro_batch_size
        max_fastpath = max(200, int(2.5 * mbs))
        lpb, counts = [], []
        mps, lps = self.mps, self.lps
        with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
            eng.configure({"Threads": 2, "Hash": 256})
            while self.games_finished < cfg.n_training_games:
                if not self.active_games:
                    break
        
                preds_batch = []
                finished = []
                for game in self.active_games:
                    # if its stockfish turn, let SF move and skip MCTS this ply
                    if game.is_stockfish_turn():
                        sf_terminal = game.make_move_with_stockfish(eng)
                        mps.tick(1)
                            
                        if sf_terminal:
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
                            self.finalize_game_data(game)
                            self.maybe_log_results()
                            finished.append(game.game_id)
                            continue
                        
                        # if its stockfish turn, dont do any sims
                        if game.is_stockfish_turn():
                            continue

                    # otherwise, collect up to micro_batch_size leaves for this game
                    # new, terminal, cached
                    nn, nt, nc = game.tree.collect_many_leaves(mbs, max_fastpath)
                    lps.tick(nn + nc)

                    # update sim count here
                    game.tree.sims_completed_this_move += nn + nt + nc

                    if nn:
                        preds_batch += game.tree.pending_encoded_64_tokens()

                    fastpaths = nt + nc
                    fast_stop, collect_stop = 0, 1
                    if nn < mbs:
                        fast_stop, collect_stop = 1, 0
    
                    counts.append([nn, fastpaths, nt, nc, fast_stop, collect_stop])
                
                # run predictions if we have any. sends results to c++ raw cache
                if preds_batch:
                    self.format_and_predict(preds_batch)
                    lpb.append(len(preds_batch))

                if self.maybe_log_results():
                    self.log_loop_stats(counts, mbs, lpb)
                    counts = []
                    lpb = []
                
                # resolve fresh predictions back into each game tree
                for game in self.active_games:
                   game.tree.resolve_pending()
                
                # clear caches after model training and the last round of moves
                if self.clear_cache:
                    pc = priors_cache_stats()
                    p_size = pc.get("size", 0)
                    p_cap  = pc.get("capacity", 1)
                    p_ev   = pc.get("evictions", 0)
                    p_q    = pc.get("queries", 0)
                    p_h    = pc.get("hits", 0)
                    p_hit  = (100.0 * p_h / p_q) if p_q else 0.0
                    p_evr = (100.0 * p_ev / p_cap) if p_cap else 0.0

                    print("Clearing caches after training")
                    cs = "[cache stats]"
                    print(f"{cs} size: {p_size}/{p_cap} evictions: {p_ev} queries: {p_q}")
                    print(
                        f"{cs} hits: {p_h} hit_rate: {p_hit:.2f}% evict_rate: {p_evr:.2f}%"
                    )

                    # finally clear them
                    raw_cache_clear()
                    priors_cache_clear()
                    self.clear_cache = False

                # remove any finished games
                self.active_games = [
                    g for g in self.active_games if g.game_id not in finished
                ]
                
                # add in more games if needed
                self.fill_active_games()
                    
        # train with whatever we got and final report (> 2000)
        if len(self.training_queue) >= 2000:
            self.trigger_retrain()

        self.maybe_log_results(force=True)
        return

    def format_and_predict(self, preds_batch):
        """
        preds_batch: list of items produced by tree.pending_encoded_stm_pov(...)
        expected shape:
            - (zobrist, board_np, legal_np)

        This calls self.infer on the GPU (masked softmax + clip + renorm).
        It writes into the raw policy cache as (zobrist, {"value": v, "policy": probs}).
        """
        if not preds_batch:
            return

        # collect arrays + keys
        boards = []
        legals = []
        keys = []

        for item in preds_batch:
            keys.append(item[0])
            board_np = item[1]
            legal_np = item[2]

            boards.append(np.asarray(board_np, dtype=np.int32))
            legals.append(np.asarray(legal_np, dtype=np.int32))

        # stack to batch
        boards_np = np.stack(boards, axis=0)   # (B,8,8,29)
        legals_np = np.stack(legals, axis=0)   # (B,4096)

        # pad up to fwd_batch if needed (helps XLA/static-trace shapes)
        target_bs = self.config.fwd_batch
        B = boards_np.shape[0]
        if B < target_bs:
            pad = target_bs - B
            pad_boards = np.zeros((pad, ) + boards_np.shape[1:], dtype=boards_np.dtype)
            pad_legals = np.zeros((pad, legals_np.shape[1]), dtype=legals_np.dtype)
            boards_np_p = np.concatenate([boards_np, pad_boards], axis=0)
            legals_np_p = np.concatenate([legals_np, pad_legals], axis=0)
            probs_np_p, vals_np_p = self.infer((boards_np_p, legals_np_p))
            probs_np = probs_np_p[:B]
            vals_np = vals_np_p[:B]
        else:
            probs_np, vals_np = self.infer((boards_np, legals_np))

        # build raw_cache rows: (zobrist, {"value": v, "policy": probs})
        to_raw_cache = []
        for i, k in enumerate(keys):
            v = np.asarray(vals_np[i]).reshape(())  # scalar
            p = np.asarray(probs_np[i], dtype=np.float32)  # (4096,)
            to_raw_cache.append((k, v, p))

        # bulk insert (C++ must be updated to accept this format)
        raw_cache_bulk_insert(to_raw_cache)

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
        
        sims_total = game.tree.sims_done_total 
        moves = game.tree.moves_played
        avg_sims = sims_total/moves if moves > 0 else 0

        # store for logging too
        self.moves_played += moves
        self.sims_done_total += sims_total
        
        # cast types for JSON 
        mem_summary = {
            "ts": _now(),
            "game_id": game.game_id,
            "scenario": game.meta.get("scenario", ""),
            "plies": game.plies,
            "result": game.outcome or 0.0,
            "vs_stockfish": game.vs_stockfish,
            "stockfish_color": game.stockfish_is_white,
            "duration": _now() - game.started_at,
            "sims_per_move": round(avg_sims, 3)
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

        # make json safe
        res = cbu.make_jsonable(res)
        
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
        
        # game outcome in white POV (1 = white win, -1 = black win)
        z = game.outcome if game.outcome is not None else 0.0
        g = game.plies or 0  # game length

        # use g-1 as denominator only when there are >= 2 plies; otherwise taper=0
        denom = max(1, g-1)

        for x, mask, policy, vwq, ply, turn in game.examples:
            # linear taper of outcome in [0, 1]
            taper = 0.0 if z == 0 else ply / denom
            z_stm = z if turn else -z # flip based on stm pov
            z_tapered = taper * z_stm
            self.training_queue.append((x, mask, policy, z_stm, vwq, z_tapered))
        game.examples = []

        if len(self.training_queue) >= self.config.training_queue_min:
            self.trigger_retrain()


    def trigger_retrain(self):
        """
        Build training tensors from self.training_queue and fit.
        Supports new flattened 'policy' head (4096) stored in heads["policy"].
        If any legacy factorized examples are present (heads without 'policy'),
        they are skipped and a warning is printed. This keeps logic simple and
        avoids brittle reconstruction of legal masks from factorized heads.
        """
        if not self.training_queue:
            return

        # check for additional data pkl and load+remove if present
        add_pkl = os.path.join(self.config.run_dir, "additional_training_data.pkl")
        additional = []

        if self.config.mine_bonus_data:
            if os.path.exists(add_pkl):
                # may hit an unlucky access deny if during a write.
                tries = 0
                while tries < 3:
                    try:
                        with open(add_pkl, "rb") as f:
                            additional = pickle.load(f)
                        # remove immediately so nothing is re-read later
                        os.remove(add_pkl)
                        n_add = len(additional)
                        print(f"[retrain] found {n_add} additional training samples")
                        break
                    except:
                        # if we get an error, wait a bit and try again
                        time.sleep(0.5)
                        tries += 1
            
            if not additional:
                print("No additional data found at", add_pkl)
        
        # combined list: existing queue first, additional appended
        combined = list(self.training_queue) + list(additional)
        n_main = len(self.training_queue)

        # Unpack examples, but only accept examples that include 'policy'
        X_list = []
        P_list = []        # flattened 4096 policy vectors
        mask_list = []     # derived legal-mask (0/1)
        Z_list = []
        Z_taper_list = []
        Vwq_list = []
        is_add_flag = []

        for i, (x, mask, policy, z_stm, vwq, z_tapered) in enumerate(combined):
            X_list.append(x)
            P_list.append(policy)
            mask_list.append((policy > 0).astype(np.int32))
            Z_list.append(z_stm)
            Vwq_list.append(vwq)
            Z_taper_list.append(z_tapered)
            is_add_flag.append(i >= n_main)  # True for additional samples

        # stack arrays
        X = np.asarray(X_list, dtype=np.float32)             # (N, 8,8,29) expected
        P = np.stack(P_list, axis=0).astype(np.float32)      # (N, 4096)
        M = np.stack(mask_list, axis=0).astype(np.int32)     # (N, 4096)
        Z = np.asarray(Z_list, dtype=np.float32)
        Vwq = np.asarray(Vwq_list, dtype=np.float32)
        Z_taper_arr = np.asarray(Z_taper_list, dtype=np.float32)
        is_add = np.asarray(is_add_flag[:len(X_list)], dtype=np.bool_)

        # optionally blend (possibly tapered) outcome with visit-weighted Q
        use_taper = self.config.use_z_taper
        alpha = self.config.z_blend
        Zstar = Z_taper_arr if use_taper else Z
        Y_value = np.clip((1.0-alpha)*Vwq + alpha*Zstar, -1.0, 1.0)
        # ensure additional samples are not blended
        if is_add.any():
            Y_value[is_add] = Vwq[is_add]

        # initial per-sample weights (ones)
        weights = np.ones_like(Y_value, dtype=np.float32)
        lw = self.config.target_loss_weights

        # assemble final Y dict and sample_weight mapping for Keras fit
        Y = {"value_out": Y_value.astype(np.float32), "policy_logits": P}
        s_wts = {k: weights*lw.get(k, 1.0) for k in Y.keys()}

        # downweight draws so value head doesnt collapse
        v_wts = s_wts['value_out']
        wts_before = v_wts.sum()
        draw_mask = Z == 0
        v_wts[draw_mask] *= max(self.config.draw_weight, 0.001)
        wts_after = v_wts.sum()

        wt_lost = wts_before - wts_after
        n_non_draws = (~draw_mask).sum()
        if wt_lost > 0 and n_non_draws > 0:
            v_wts[~draw_mask] += wt_lost/n_non_draws
        
        s_wts['value_out'] = v_wts

        # evaluation and logging (reuse existing helpers) - unchanged
        plt_file = os.path.join(self.config.run_dir, "true_vs_pred_plot_latest.png")
        epoch = self.n_retrains
        eval_df = cbu.score_game_data(self.model, X, M, Y, epoch, save_path=plt_file)
        self.all_evals = pd.concat([self.all_evals, eval_df])
        self.all_evals.round(5).to_csv(self.config.progress_csv_path, index=False)

        if len(self.all_evals) and len(self.all_evals) % 4 == 0:
            prog_plt_file = self.config.progress_plot_path
            cbu.plot_training_progress(
                self.all_evals, epoch=epoch, save_path=prog_plt_file
            )

        # fit and save new model: X is a list/tuple matching model inputs (planes, mask)
        history = self.model.fit(
            X, Y, epochs=3, batch_size=128, verbose=0,
            sample_weight=s_wts, shuffle=True
        )

        rows = []
        for m, v in history.history.items():
            name = "total" if m == "loss" else m.replace("_loss", "")
            start = v[0]; end = v[-1]
            delta = start - end
            mark = "*" if delta < 0 else "+"
            rows.append((name, start, end, delta, mark))
        
        name_w = max(len(r[0]) for r in rows)
        num_w = 8   # width for numbers (including decimal point)
        fmt = (f"[epoch {epoch:4d}] [model fit]  "
            f"{{name:<{name_w}}} : value: {{start:{num_w}.4f}} -> "
            f"{{end:{num_w}.4f}}  delta: {{delta:{num_w}.4f}} {{mark}}")

        for name, start, end, delta, mark in rows:
            print(fmt.format(name=name, start=start, end=end, delta=delta, mark=mark))

        self.model.save(self.config.model_path)

        # update the fwd helper and clear queues/caches
        cfg = self.config
        self.infer = make_conv_infer(
            self.model, max_bs=cfg.fwd_batch,
            min_p=cfg.prior_clip_min, max_p=cfg.prior_clip_max, temp=1
        )

        # clear training queue (we consumed the in-memory queue)
        self.training_queue = []
        self.n_retrains += 1
        self.clear_cache = True

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
        gph =  3600 * self.games_finished / (now - self._run_start)

        print()
        print("~" * 72)
        print(
            f"[speed stats] mps={self.mps.rate():.1f}  "
            f"lps={self.lps.rate():.1f}  gph={gph:.2f}")
        
        print(
            f"[game stats]  finished={self.games_finished}  "
            f"W/L/D={self.white_wins}/{self.black_wins}/{self.draws}  "
            f"avg_len={avg_moves:.1f} moves")
        print("-" * 72)

        recent = list(self.recent_games)[-min(window, len(self.recent_games)):]
        if not recent:
            print("(no recent games to break down)")
            print("~" * 72)
            return True

        # pretty printer
        cbu.print_recent_summary(recent, window=window)
        print(
            f"Length of training queue: {len(self.training_queue)} ",
            f"Number of retrains: {self.n_retrains}\n"
        )
        return True

    def log_loop_stats(self, counts, mbs, lpb):
        if not counts:
            return

        # aggregate
        n_groups = len(counts)
        s_collected  = sum([r[0] for r in counts])
        s_fast       = sum([r[1] for r in counts])
        s_terminals  = sum([r[2] for r in counts])
        s_cached     = sum([r[3] for r in counts])
        s_fast_stops = sum([r[4] for r in counts])
        s_collect_stops = sum([r[5] for r in counts])

        avg_new = s_collected / n_groups

        total_overall = s_collected + s_terminals + s_cached
        term_to_cached = s_terminals / s_cached if s_cached > 0 else 0.0
        pct_cached_overall = 100.0 * s_cached / max(1, total_overall)
        pct_term_overall = 100.0 * s_terminals / max(1, total_overall)

        fast_stops_pct = 100.0 * s_fast_stops / max(1, n_groups)
        collect_stops_pct = 100.0 * s_collect_stops / max(1, n_groups)

        print("-"*72)
        left1 = f"[loop stats] groups={n_groups}  mbs={mbs}"
        right1 = f"new: collected={s_collected}, avg={avg_new:.2f}"

        left2 = f"[stop stats] fastpath_breaks={s_fast_stops} ({fast_stops_pct:.2f}%)"
        right2 = f"collect_breaks={s_collect_stops} ({collect_stops_pct:.2f}%)"

        left3 = f"[cache hits] cached={s_cached} ({pct_cached_overall:.3f}%)"
        right3 = f"terminals={s_terminals} ({pct_term_overall:.3f}%)"

        col_width = 40
        print(f"{left1:<{col_width}} | {right1}")
        print(f"{left2:<{col_width}} | {right2}")
        print(f"{left3:<{col_width}} | {right3}")

        if lpb:
            print(f"Avg len of preds_batch {np.mean(lpb):.3f}")

        avg_ply = np.mean([g.plies for g in self.active_games])
        print(f"Avg ply of active_games {avg_ply:.3f}")
        print(f"Number of active_games {len(self.active_games)}")

        # avg duration of last 50 completed games
        last50 = self.recent_games[-50:]
        durations = [g.get("duration", 0.0) for g in last50]
        if sum(durations) > 0:
            avg_len_str = cbu.format_time(np.mean(durations))
            print(f"Avg game runtime (last {len(last50)}) {avg_len_str}")
        
        sims = self.sims_done_total
        moves = self.moves_played
        sims_per_move = sims/moves if moves > 0 else 0
        if sims_per_move:
            print(f"Avg sims per move {sims_per_move:.3f}")
        print("-"*72)


def init_selfplay():
    # file structure first
    config = Config()
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
        print(f"Loading {model_path}")
        model = load_model(model_path)
    else:
        print(f"Loading {config.init_model}")
        model = load_model(config.init_model)
        save_model(model, model_path)
    config = Config()
    
    return model, config


def main():
    model, config = init_selfplay()
    looper = GameLooper(model=model, cfg=Config())

    # infer the number of trainings already done from existing files
    if os.path.exists(config.progress_csv_path):
        try:
            progress_df = pd.read_csv(config.progress_csv_path)
            n_retrains = len(progress_df)
            looper.n_retrains = n_retrains
            looper.all_evals = progress_df
        except Exception as e:
            print(e)
    
    # start the analysis server
    if config.run_post_hoc:
        phs = start_post_hoc_server(
            looper.config.run_dir,
            bonus_data=config.mine_bonus_data
        )
        try:
            looper.run()
        except Exception as e:
            print("Error encountered", e)
        finally:
            stop_post_hoc_server(phs, timeout=10)
    
    else:
        looper.run()


if __name__ == '__main__':
    main()