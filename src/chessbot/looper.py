import os, pickle, random
import pathlib, json
import time, gc
import sys, subprocess

_now = time.time

import numpy as np
import pandas as pd
import tensorflow as tf

import chess

from pyfastchess import raw_cache_bulk_insert
from pyfastchess import raw_cache_clear, priors_cache_clear, priors_cache_stats

from chessbot import SF_LOC

from chessbot.model import load_model, save_model, make_conv_infer
from chessbot.validation import paired_validation_games
from chessbot.mcts_utils import ChessGame

import chessbot.utils as cbu
from chessbot.utils import RateMeter, GameGenerator


class GameLooper(object):
    """
    Orchestrates N games concurrently, central batching, caches, and training.
    """
    def __init__(self, model, cfg, recent_games_q, telemetry_q):
        self.config = cfg
        self.recent_games_q = recent_games_q
        self.telemetry_q = telemetry_q
        self.game_gen = GameGenerator(self.config)
        self.games_finished = 0
        self.active_games = []
        self.sf_count = 0

        # different game logic for training vs validation
        if self.config.is_validation_run:
            self.active_games = paired_validation_games(self.config)
            self.collect_training_data = False
        else:
            self.fill_active_games()
            self.collect_training_data = True
        
        self.model = model

        self.eng = None
        if cfg.play_vs_sf_prob > 0.0:
            self.eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
            self.eng.configure(cfg.sf_config)

        cfg = self.config
        self.infer = make_conv_infer(
            self.model, max_bs=cfg.fwd_batch,
            min_p=cfg.prior_clip_min, max_p=cfg.prior_clip_max,
            vscale=cfg.vscale
        )

        self.infer_is_warm = False
        self.batch_candidates = self.create_batch_candidates(cfg)

        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_plies = 0
        self.n_retrains = 0
        self.retrain_procs = []
        self.clear_cache = False
        
        self._run_start = _now()
        self.mps = RateMeter("moves")
        self.lps = RateMeter("leafs")
        self._last_stats_log = 0.0
        self.prediction_times = []
        self.sf_search_depths = []

        self.moves_played = 0
        self.sims_done_total = 0
    
    def close(self):
        if self.eng is not None:
            self.eng.quit()
            self.eng = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def warmup_infer(self):    
        if not self.infer_is_warm:
            print(f"[model warmup] warming GPU for {self.batch_candidates}")
            for bs in self.batch_candidates:
                for _ in range(3):
                    rep_mask = (np.random.rand(target_bs, 4288) < 0.02).astype(np.int32)
                    rep_enc = (np.random.rand(target_bs, 64) < 0.32).astype(np.int32)
                    self.infer((rep_enc, rep_mask))
            self.infer_is_warm = True
            print("[model warmup] warm up complete")

    def restart_run(self, run_num=None):
        cfg = self.config
        self.infer = make_conv_infer(
            self.model, max_bs=cfg.fwd_batch,
            min_p=cfg.prior_clip_min, max_p=cfg.prior_clip_max,
            vscale=cfg.vscale
        )

        self.infer_is_warm = False
        return self.run(run_num=run_num)

    def create_batch_candidates(self, cfg):
        # create sizes to warm up
        batch_candidates = set()
        batch_candidates.add(cfg.fwd_batch)
        bs = 16
        while bs <= cfg.fwd_batch:
            batch_candidates.add(bs)
            bs *= 2
        
        # split difference between last 2
        if len(batch_candidates) >= 2:
            sbc = sorted(batch_candidates)
            sbc.append(int(sbc[-1] - sbc[-2] / 2))
            return sorted(set(sbc))
        return sorted(set(batch_candidates))

    def fill_active_games(self):
        cfg = self.config
        needed = cfg.n_games - self.games_finished - len(self.active_games)
        if needed <= 0:
            return

        # add games w.r.t. num needed and games_at_once
        for _ in range(needed):
            if len(self.active_games) >= cfg.games_at_once:
                return

            # this has game probs from the config
            board, meta = self.game_gen.new_board()

            meta['vs_stockfish'] = False
            meta['stockfish_is_white'] = False
            meta['vs_stockfish'] = np.random.uniform() <= cfg.play_vs_sf_prob

            # stockfish only plays certain scenarios
            if meta['scenario'] in cfg.sf_exclude:
                meta['vs_stockfish'] = False

            if meta['vs_stockfish']:
                self.sf_count += 1
                meta['vs_stockfish'] = True
            
                # alternate sf color to balance white and black
                meta['stockfish_is_white'] = bool(self.sf_count % 2)
            
            cg = ChessGame(board=board, meta=meta, cfg=self.config)
            self.active_games.append(cg)
    
    def run(self, run_num=None):
        """
        Main loop. Each round: for each game either let SF move (if applicable)
        or run MCTS step (collect/predict/apply).
        """

        cfg = self.config
        eng = self.eng
        self.warmup_infer()
        mbs = cfg.micro_batch
        max_fastpath = max(200, int(2.5 * mbs))
        lpb, counts = [], []
        mps, lps = self.mps, self.lps
        while self.games_finished < cfg.n_games:
            if not self.active_games:
                break
            
            preds_batch = []
            finished = []
            for game in self.active_games[:cfg.games_at_once]:
                if game.tree.needs_root_noise(check_sims=True):
                    game.tree.add_root_dirichlet_noise()

                # if its stockfish turn, let SF move and skip MCTS this ply
                if game.is_stockfish_turn():
                    if game.tree.sims_completed_this_move > cfg.sf_move_sims:
                        sf_terminal = game.make_move_with_stockfish(eng)
                        mps.tick(1)
                            
                        if sf_terminal:
                            self.finalize_game_data(game)
                            finished.append(game.game_id)
                            # pass until the next turn
                            continue
    
                # if this game has reached its local sim budget, make the move
                elif game.tree.stop_simulating():
                    # bot plays from tree
                    mcts_terminal = game.make_move_from_tree()
                    mps.tick(1)
                    
                    # terminal after bot move?
                    if mcts_terminal:
                        self.finalize_game_data(game)
                        finished.append(game.game_id)
                        continue

                # collect up to micro_batch leaves for this game
                # CollectResults object from C++
                res = game.tree.collect_many_leaves(mbs, max_fastpath)
                
                nn = res.count_new
                nt = res.count_terminal
                nc = res.count_cached
                pl = res.total_priorless
                pu = res.total_puct

                # new + cached + terminal
                n_leafs = nn + nc + nt 
                lps.tick(n_leafs)

                # update sim count
                game.tree.sims_completed_this_move += n_leafs

                if nn:
                    preds_batch += game.tree.pending_encoded_64_tokens()

                fastpaths = nt + nc
                f_stop, c_stop = 0, 1
                if nn < mbs:
                    f_stop, c_stop = 1, 0

                counts.append([nn, fastpaths, nt, nc, f_stop, c_stop, pl, pu])
            
            # run predictions if we have any. sends results to c++ raw cache
            if preds_batch:
                self.format_and_predict(preds_batch)
                lpb.append(len(preds_batch))

            if self.maybe_log_results(run_num=run_num):
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
                p_evr  = (100.0 * p_ev / p_cap) if p_cap else 0.0

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
        
        self.maybe_log_results(force=True, run_num=run_num)
        return True

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
        boards_np = np.stack(boards, axis=0)   # (B,64)
        legals_np = np.stack(legals, axis=0)   # (B,4288)
        # pad up to fwd_batch or a smaller power of 2 if needed
        # (helps XLA/static-trace shapes)

        target_bs = self.config.fwd_batch
        B = boards_np.shape[0]
        if B < target_bs:
            # choose padding target safely
            for new_target in self.batch_candidates:
                if new_target >= B:
                    break
            pad = max(0, int(new_target - B))
            if pad:
                pad_boards = np.zeros((pad,)+boards_np.shape[1:], dtype=boards_np.dtype)
                pad_legals = np.zeros((pad, legals_np.shape[1]), dtype=legals_np.dtype)
                boards_np_p = np.concatenate([boards_np, pad_boards], axis=0)
                legals_np_p = np.concatenate([legals_np, pad_legals], axis=0)
                start = _now()
                probs_np_p, vals_np_p = self.infer((boards_np_p, legals_np_p))
                probs_np = probs_np_p[:B]
                vals_np = vals_np_p[:B]
            
            else:
                start = _now()
                probs_np, vals_np = self.infer((boards_np, legals_np))
        else:
            start = _now()
            probs_np, vals_np = self.infer((boards_np, legals_np))
        
        # build raw_cache rows: (zobrist, {"value": v, "policy": probs})
        to_raw_cache = []
        for i, k in enumerate(keys):
            v = np.asarray(vals_np[i]).reshape(())  # scalar
            p = np.asarray(probs_np[i], dtype=np.float32)  # (4288,)
            to_raw_cache.append((k, v, p))

        # bulk insert
        raw_cache_bulk_insert(to_raw_cache)
        stop = _now()
        self.prediction_times.append(stop-start)

    def finalize_game_data(self, game):
        """
        Attach the final scalar outcome to every per-move example and enqueue.
        Outcome is already white-POV (-1/0/+1) and does not need flipping.
        """
        # do not count extremely short games
        if game.plies <= self.config.min_game_length:
            game.examples = []
            return

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
        self.sf_search_depths += game.sf_search_depth

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
        
        out_file = os.path.join(self.config.game_dir, game.game_id + "_log.pkl")
        out_path = pathlib.Path(out_file)

        mem_summary['pkl_file'] = outpath
        # this is small, push it to parent process via Queue instead on appending
        if self.game
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
        res['c_puct'] = game.tree.c_puct

        # attach tree search data to disk record
        res["tree_search_data"] = game.tree_data

        # make json/pickle safe
        res = cbu.make_jsonable(res)
        
        out_file = os.path.join(self.config.game_dir, game.game_id + "_log.pkl")
        out_path = pathlib.Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # clear the game recents to prevent mem leaks
        game.recents.clear()
        return

    def maybe_log_results(self, every_sec=60.0, window=500, force=False, run_num=None):
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
        if run_num is None:
            print("~"*72)
        else:
            print(f" Round {run_num} Logging ".center(72, "~"))
        
        print(
            f"[speed stats] mps={self.mps.rate():.1f}  "
            f"lps={self.lps.rate():.1f}  gph={gph:.2f}")
        
        print(
            f"[game stats]  finished={self.games_finished}  "
            f"W/D/L={self.white_wins}/{self.draws}/{self.black_wins}  "
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
            f"Current retrain number: {self.n_retrains}\n"
        )
        return True

    def log_loop_stats(self, counts, mbs, lpb):
        if not counts:
            return

        # aggregate
        n_groups = len(counts)
        s_collected     = sum([r[0] for r in counts])
        s_fast          = sum([r[1] for r in counts])
        s_terminals     = sum([r[2] for r in counts])
        s_cached        = sum([r[3] for r in counts])
        s_fast_stops    = sum([r[4] for r in counts])
        s_collect_stops = sum([r[5] for r in counts])
        s_priorless     = sum([r[6] for r in counts])
        s_puct          = sum([r[7] for r in counts])

        avg_new = s_collected / n_groups

        total_overall = s_collected + s_terminals + s_cached
        term_to_cached = s_terminals / s_cached if s_cached > 0 else 0.0
        pct_cached_overall = 100.0 * s_cached / max(1, total_overall)
        pct_term_overall = 100.0 * s_terminals / max(1, total_overall)

        fast_stops_pct = 100.0 * s_fast_stops / max(1, n_groups)
        collect_stops_pct = 100.0 * s_collect_stops / max(1, n_groups)

        print("-"*72)
        left1 = f"[loop stats] groups={n_groups}  mbs={mbs}"
        right1 = f"new: collected={s_collected} avg={avg_new:.2f}"

        left2 = f"[stop stats] fastpath_breaks={s_fast_stops} ({fast_stops_pct:.2f}%)"
        right2 = f"collect_breaks={s_collect_stops} ({collect_stops_pct:.2f}%)"

        # preds / active / finished runtime pre-compute
        apl = np.mean(lpb) if lpb else 0.0
        fwd_target = self.config.fwd_batch
        fill_pct = 100.0 * apl / max(1.0, fwd_target)

        pred_wait = np.mean(self.prediction_times) if self.prediction_times else 0.0
        self.prediction_times.clear()
        preds_per_sec = apl / pred_wait if pred_wait > 0 else 0.0

        left3 = f"[pred stats] fill={apl:.1f}/{fwd_target} ({fill_pct:.1f}%)"
        right3 = f"wait={pred_wait:.03f}s preds/s={preds_per_sec:.1f}"

        with_priors = total_overall - s_priorless
        puct_avg = s_puct / with_priors if with_priors else 0.0
        priorless_pct = 100.0 * s_priorless / max(1, total_overall)
        left4 = f"[leaf stats] priorless={s_priorless} ({priorless_pct:.2f}%)"
        right4 = f"puct={int(s_puct)}  puct/leaf={puct_avg:.1f}"

        left5 = f"[cache hits] cached={s_cached} ({pct_cached_overall:.3f}%)"
        right5 = f"terminals={s_terminals} ({pct_term_overall:.3f}%)"

        sims = self.sims_done_total
        moves = self.moves_played
        sims_per_move = sims / moves if moves > 0 else 0.0

        n_active = len(self.active_games)
        avg_ply = np.mean([g.plies for g in self.active_games]) if n_active else 0.0
        left6 = f"[game stats] n={n_active} avg ply={avg_ply:.2f}"
        right6 = f"sims per move={sims_per_move:.2f}"

        col_width = 40
        print(f"{left1:<{col_width}} | {right1}")
        print(f"{left2:<{col_width}} | {right2}")
        print(f"{left3:<{col_width}} | {right3}")
        print(f"{left4:<{col_width}} | {right4}")
        print(f"{left5:<{col_width}} | {right5}")
        print(f"{left6:<{col_width}} | {right6}")

        # show game duration if its available
        last50 = self.recent_games[-50:]
        durations = [g.get("duration", 0.0) for g in last50]
        avg_runtime = None
        if sum(durations) > 0:
            avg_runtime = cbu.format_time(np.mean(durations))
            if avg_runtime:
                print(f"[game stats] last 50 runtime: {avg_runtime}")
        print("-"*72)

    def spawn_next_looper(self, clear_caches=True):
        """
        Create a fresh GameLooper using the model at config.model_path, transfer
        runtime internals into it, and return the new looper.
        """
        # make sure we free TF/Keras state before loading the new model
        print(
            f"[spawn] reloading model from "
            f"{os.path.basename(self.config.model_path)}"
        )
        tf.keras.backend.clear_session()

        # load model and create fresh looper (fresh infer, fresh tf funcs)
        new_model = load_model(self.config.model_path)
        new_looper = GameLooper(model=new_model, cfg=self.config.copy())

        # fields to transfer (lists and counters)
        transfer_fields = [
            "active_games", "training_queue", "recent_games",
            "games_finished", "white_wins", "black_wins", "draws",
            "total_plies", "n_retrains", "sf_count",
            "moves_played", "sims_done_total", "sf_search_depths",
        ]

        for fld in transfer_fields:
            setattr(new_looper, fld, getattr(self, fld))

        # transfer meters and timing state (keep same objects so history is retained)
        new_looper.mps = self.mps
        new_looper.lps = self.lps
        new_looper.prediction_times = self.prediction_times
        new_looper._run_start = getattr(self, "_run_start", _now())

        # ensure new looper starts with a fresh infer warm state
        new_looper.infer_is_warm = False

        # optionally clear shared caches (helps avoid stale C++/raw cache issues)
        if clear_caches:
            print("[spawn] clearing raw/prior caches to avoid leaks")
            raw_cache_clear()
            new_looper.clear_cache = False

        # free old model object and try to force collection
        del self.model
        del self.infer
        gc.collect()

        print("[spawn] new looper ready; continuing with transferred games")
        return new_looper


def init_selfplay(config, gameplay_q, telemetry_q):
    # pre-built config (from yaml)
    model_name = config.run_tag + "_model.h5"

    if not getattr(config, "model_path", False):
        model_path = os.path.join(config.run_dir, model_name)
        config.model_path = model_path
    else:
        model_path = config.model_path

    if os.path.exists(model_path):
        print(f"[init] Loading {model_name}")
        model = load_model(model_path)
    else:
        print(f"[init] Loading {config.init_model}")
        model = load_model(config.init_model)
        save_model(model, model_path)

    looper = GameLooper(model=model, cfg=config.copy(), gameplay_q, telemetry_q)

    # infer number of retrains already done from existing progress csv
    if os.path.exists(config.progress_csv_path):
        try:
            progress_df = pd.read_csv(config.progress_csv_path)
            n_retrains = len(progress_df)
            looper.n_retrains = n_retrains
        except Exception as e:
            print("[init_selfplay] failed reading progress csv:", e)

    return looper
