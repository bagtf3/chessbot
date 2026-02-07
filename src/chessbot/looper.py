import os, pickle, random
import pathlib, json
import time, gc
import sys, subprocess

from queue import Empty

_now = time.time

import numpy as np
import pandas as pd
import tensorflow as tf

import chess

from pyfastchess import raw_cache_bulk_insert, raw_cache_clear, priors_cache_clear

from chessbot import SF_LOC

from chessbot.model import load_model, save_model, make_conv_infer
from chessbot.validation import paired_validation_games
from chessbot.mcts_utils import ChessGame
from chessbot.utils import RateMeter, GameGenerator


class GameLooper(object):
    """
    Orchestrates N games concurrently, central batching, caches, and training.
    """
    def __init__(self, model, cfg, recent_games_q, telemetry_q, msg_q):
        self.id = cfg.id
        self.config = cfg
        self.recent_games_q = recent_games_q
        self.telemetry_q = telemetry_q
        self.msg_q = msg_q
        self.unpause_queued = False
        self.game_gen = GameGenerator(self.config)
        self.games_finished = 0
        self.active_games = []
        self.sf_count = 0

        # different game logic for training vs validation
        if self.config.is_validation_run:
            self.active_games = paired_validation_games(self.config)
        else:
            self.fill_active_games()
        
        self.model = model

        self.eng = None
        if cfg.play_vs_sf_prob > 0.0:
            self.eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
            self.eng.configure(cfg.sf_config)
        
        # pulls in model from config and XLA compilers inferencer
        self.load_reload_model()

        self.infer_is_warm = False
        self.batch_candidates = self.create_batch_candidates(self.config)
        self.n_retrains = 0
        
        self._run_start = _now()
        self.mps = RateMeter("moves")
        self.lps = RateMeter("leafs")
        self._last_stats_log = _now()
        self.prediction_times = []
        self.sf_search_depths = []
    
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
        if self.infer_is_warm:
            return
        print(f"[model warmup] warming GPU for {self.batch_candidates}")
        for bs in self.batch_candidates:
            for _ in range(3):
                rep_mask = (np.random.rand(bs, 4288) < 0.02).astype(np.int32)
                rep_enc = (np.random.rand(bs, 64) < 0.32).astype(np.int32)
                self.infer((rep_enc, rep_mask))

        self.infer_is_warm = True
        print("[model warmup] warm up complete")

    def load_reload_model(self):
        cfg = self.config
        self.model = load_model(cfg.model_path)
        self.infer = make_conv_infer(
            self.model, max_bs=cfg.fwd_batch,
            min_p=cfg.prior_clip_min, max_p=cfg.prior_clip_max,
            vscale=cfg.vscale)
    
    def check_for_pause(self):
        """
        Drain msg_q (non-blocking). Return True if a pause was requested.

        Important: do NOT discard "unpause" if it arrives early. Buffer it so a later
        pause_wait_and_reload() will not deadlock waiting for an unpause that was
        already consumed.
        """
        while True:
            try:
                msg = self.msg_q.get_nowait()
            except Empty:
                return False

            cmd = msg.get("cmd") if isinstance(msg, dict) else msg

            if cmd == "pause":
                return True

            if cmd == "unpause":
                self.unpause_queued = True
                continue

            # ignore everything else
            continue

    def pause_wait_and_reload(self):
        """
        Pause hard: clear caches, tear down GPU objects, then wait until "unpause".
        After "unpause", reload model + infer and return.

        If "unpause" arrived early and was buffered, do not block.
        """
        del self.infer
        self.infer = None

        del self.model
        self.model = None

        tf.keras.backend.clear_session()
        gc.collect()

        raw_cache_clear()
        priors_cache_clear()

        if self.unpause_queued:
            self.unpause_queued = False
        else:
            while True:
                msg = self.msg_q.get()
                cmd = msg.get("cmd") if isinstance(msg, dict) else msg

                if cmd == "unpause":
                    break

                if cmd == "pause":
                    continue

                # ignore everything else
                continue

        self.load_reload_model()
        self.n_retrains += 1

    def create_batch_candidates(self, cfg):
        # create sizes to warm up
        batch_candidates = set()
        batch_candidates.add(cfg.fwd_batch)
        bs = 8
        while bs <= cfg.fwd_batch:
            batch_candidates.add(bs)
            bs *= 2
        
        # split difference between last 2 if large
        if len(batch_candidates) >= 2 and cfg.fwd_batch >= 128:
            sbc = sorted(batch_candidates)
            lo = sbc[-2]
            hi = sbc[-1]
            mid = lo + (hi - lo) // 2
            sbc.append(mid)
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
    
    def run(self, stop_event=None):
        """
        Main loop. Each round: for each game either let SF move (if applicable)
        or run MCTS step (collect/predict/apply).
        """

        cfg = self.config
        eng = self.eng
        mbs = cfg.micro_batch
        max_fastpath = max(200, int(2.5 * mbs))
        lpb, counts = [], []
        mps, lps = self.mps, self.lps
        while self.games_finished < cfg.n_games:
            # check a few stopping conditions 
            if stop_event is not None:
                if stop_event.is_set():
                    return
            
            if not self.active_games:
                break
            
            if self.check_for_pause():
                self.pause_wait_and_reload()

            # selfplay loop starts here
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
                    # xerces plays from tree
                    mcts_terminal = game.make_move_from_tree()
                    mps.tick(1)
                    
                    # terminal after xerces move?
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

            if self.maybe_push_telemetry(counts, lpb, force=False):
                self.prediction_times.clear()
                counts = []
                lpb = []
            
            # resolve fresh predictions back into each game tree
            for game in self.active_games:
                game.tree.resolve_pending()

            # remove any finished games
            self.active_games = [
                g for g in self.active_games if g.game_id not in finished
            ]
            
            # add in more games if needed
            self.fill_active_games()
        
        self.maybe_push_telemetry(counts, lpb, force=True)
        return 0

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
        
        sims_total = game.tree.sims_done_total 
        moves = game.tree.n_moves_played
        avg_sims = sims_total/moves if moves > 0 else 0

        # read adjudicator flags (direct attrs; they always exist)
        mat = self.config.use_material_diff
        tb = self.config.use_syzygy
        dr = self.config.use_eval_draw
        cl = self.config.use_eval_collar

        # encode into 0..15 index: bit0=material, bit1=syzygy, bit2=eval_draw,
        # bit3=eval_collar
        adjudication_index = (mat) | (tb << 1) | (dr << 2) | (cl << 3)

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
            "sims_done_total": sims_total,
            "start_fen": game.starting_fen,
            "adjudication_index": adjudication_index,
            # this is the specific c_puct, not the list of options
            "c_puct": game.tree.c_puct
        }

        # on-disk record (full)
        res = {
            "history_uci": game.board.history_uci(),
            "moves_played": game.moves_played,
            "model_epoch": self.n_retrains
        }

        res.update(mem_summary)
        res.update(self.config.to_dict())
        res.update(game.meta)

        # attach tree search data to disk record
        res["tree_search_data"] = game.tree_data
        res['c_puct'] = game.tree.c_puct
        out_file = os.path.join(self.config.game_dir, game.game_id + "_log.pkl")
        out_path = pathlib.Path(out_file)
        mem_summary['pkl_file'] = str(out_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # this is small, push it to parent process via Queue instead on appending
        self.recent_games_q.put({"looper_id": self.id, "meta": mem_summary})

        # this keeps games and plies in sync for logging
        self.update_partial_telemetry()
        
        # clear the game recents to prevent mem leaks
        game.recents.clear()
        return
    
    def maybe_push_telemetry(self, counts, lpb, every_sec=45.0, force=False):
        now = _now()
        if not force:
            if now - self._last_stats_log < every_sec:
                return False

        self._last_stats_log = now

        telemetry = {
            "ts": now,
            "mps": self.mps.rate(),
            "lps": self.lps.rate(),
            "mbs": self.config.micro_batch,
            "apl": np.mean(lpb) if lpb else 0.0,
            "pred_wait": 0.0,
            "preds_per_second": 0.0,
            "fwd_target": self.config.fwd_batch,
            "n_active": len(self.active_games),
            "avg_ply": 0.0,
            "n_groups": len(counts),
            "s_collected": sum([r[0] for r in counts]),
            "s_fast": sum([r[1] for r in counts]),
            "s_terminals": sum([r[2] for r in counts]),
            "s_cached": sum([r[3] for r in counts]),
            "s_fast_stops": sum([r[4] for r in counts]),
            "s_collect_stops": sum([r[5] for r in counts]),
            "s_priorless": sum([r[6] for r in counts]),
            "s_puct": sum([r[7] for r in counts]),
        }

        if self.active_games:
            telemetry['avg_ply'] = np.mean([g.plies for g in self.active_games])

        if self.prediction_times:
            telemetry['pred_wait'] = np.mean(self.prediction_times)
            telemetry['preds_per_second'] = telemetry['apl'] / telemetry['pred_wait']
        
        # put telemetry on the queue and return True to clear counts and lpb
        self.telemetry_q.put({"looper_id": self.id, "telemetry": telemetry})
        return True
    
    def update_partial_telemetry(self):
        """send a partial update to the telemetry for more time sensitive metrics"""
        partial_telem = {
            "ts": time.time(),
            "n_active": len(self.active_games),
            "avg_ply": 0.0
        }

        if self.active_games:
            partial_telem['avg_ply'] = np.mean([g.plies for g in self.active_games])

        self.telemetry_q.put({"looper_id": self.id, "telemetry": partial_telem})


def init_selfplay(config, recent_games_q, telemetry_q, msg_q):
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

    looper = GameLooper(
        model=model, cfg=config.copy(),
        recent_games_q=recent_games_q, telemetry_q=telemetry_q, msg_q=msg_q
    )

    # infer number of retrains already done from existing progress csv
    if os.path.exists(config.progress_csv_path):
        progress_df = pd.read_csv(config.progress_csv_path)
        n_retrains = len(progress_df)
        looper.n_retrains = n_retrains

    return looper
