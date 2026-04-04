import os, pickle, gzip, random
import pathlib, json
import time, gc
import sys, subprocess

import threading
from queue import Queue, Empty

import chess
import chess.engine

from chessbot.review import score_to_value_stm_pov

_now = time.time

import numpy as np
import pandas as pd
import tensorflow as tf

from pyfastchess import raw_cache_bulk_insert, priors_cache_clear, priors_cache_stats

from chessbot import SF_LOC

from chessbot.model import load_model, save_model, make_conv_infer
from chessbot.mcts_utils import ChessGame
from chessbot.utils import RateMeter, sf_eval
from chessbot.game_utils import GameSpec
#from chessbot.tf_thread import Batcher, TensorFlowThread


class GameLooper(object):
    """
    Orchestrates N games concurrently, central batching, caches, and training.
    """
    def __init__(self, model, cfg, recent_q, telem_q, msg_q, game_queue=None, sf_queue=None):
        self.id = cfg.id
        self.config = cfg
        self.recent_games_q = recent_q
        self.telemetry_q = telem_q
        self.msg_q = msg_q
        self.game_queue = game_queue
        self.sf_queue = sf_queue
        self.unpause_queued = False
        self.games_finished = 0
        self.active_games = []
        self.sf_games = {}
        self.sf_thread = None  # lazy-initialized on first vs_stockfish game

        self.pull_from_queue()

        self.model = model
        
        # pulls in model from config and XLA compilers inferencer
        self.load_reload_model()
        
        self.batch_candidates = self.create_batch_candidates(self.config)
        self.n_retrains = 0
        
        self._run_start = _now()
        self.mps = RateMeter("moves")
        self.lps = RateMeter("leafs")
        self._last_stats_log = _now()
        self.prediction_times = []

        # self.batcher = Batcher(cfg, self.batch_candidates)
        # self.tf_thread = TensorFlowThread(
        #     cfg, self.infer, max_inflight=cfg.max_tf_inflight)
        
        # self.tf_thread.start()
    
    def close(self):
        if self.sf_thread is not None:
            self.sf_thread.close()
            self.sf_thread = None
        
        # if self.tf_thread is not None:
        #     self.tf_thread.close()
        #     self.tf_thread = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def load_reload_model(self):
        cfg = self.config
        if cfg.inference_backend == "pytorch":
            from chessbot.train_pytorch import load_pt_model, make_pt_infer
            model, _ = load_pt_model(cfg.pytorch_model_path)
            self.model, self.infer = make_pt_infer(
                model,
                max_bs=cfg.fwd_batch,
                vscale=cfg.vscale,
            )
        else:
            self.model = load_model(cfg.model_path)
            self.infer = make_conv_infer(
                self.model,
                max_bs=cfg.fwd_batch,
                vscale=cfg.vscale,
            )
    
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
        Pause hard: stop TF preds, tear down GPU objects, then wait until "unpause".
        After "unpause", reload model + infer and return.

        If "unpause" arrived early and was buffered, do not block.
        """
        # if self.tf_thread is not None:
        #     self.tf_thread.pause(blocking=True)
        #     with self.tf_thread.infer_lock:
        #         self.tf_thread.infer = None
        
        del self.infer
        self.infer = None

        del self.model
        self.model = None

        if self.config.inference_backend == "pytorch":
            import torch
            torch.cuda.empty_cache()
        else:
            tf.keras.backend.clear_session()
        gc.collect()

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

                continue

        self.load_reload_model()

        # if self.tf_thread is not None:
        #     with self.tf_thread.infer_lock:
        #         self.tf_thread.infer = self.infer
        #     self.tf_thread.unpause()

        self.n_retrains += 1

    def create_batch_candidates(self, cfg):
        # create sizes
        batch_candidates = set([cfg.min_batch, cfg.fwd_batch])
        bs = cfg.min_batch
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

    def pull_from_queue(self):
        from pyfastchess import Board as fastboard
        coast_limit = self.config.n_games + 3
        while (len(self.active_games) < self.config.games_at_once and
               len(self.active_games) + self.games_finished < coast_limit):
            spec = None
            if self.sf_queue is not None:
                try:
                    spec = self.sf_queue.get_nowait()
                except Exception:
                    pass
            if spec is None:
                try:
                    spec = self.game_queue.get_nowait()
                except Exception:
                    break
            if spec.meta.get("vs_stockfish") and self.sf_thread is None:
                self.sf_thread = StockfishThread(
                    self.config.sf_config, self.config.sf_depth
                )
                self.sf_thread.start()
            board = fastboard(spec.fen)
            for mv in spec.moves:
                board.push_uci(mv)
            cg = ChessGame(board=board, meta=spec.meta, cfg=spec.cfg)
            self.active_games.append(cg)

    def run(self, stop_event=None):
        """
        Main loop. Each round: for each game either let SF move (if applicable)
        or run MCTS step (collect/predict/apply).
        """

        cfg = self.config
        #batcher = self.batcher
        mbs = cfg.micro_batch
        fwd = cfg.fwd_batch
        max_fastpath = max(512, int(2.5 * mbs))
        mps = self.mps

        #(batch size, target), counts returned
        pred_fill, counts = [], []
        finished_ids = set()
        #passes_since_trigger = 0
        while self.games_finished < cfg.n_games:
            # check a few stopping conditions 
            if stop_event is not None:
                if stop_event.is_set():
                    return
            
            # remove any finished games
            if finished_ids:
                self.active_games = [
                    g for g in self.active_games if g.game_id not in finished_ids
                ]
            
            # add in more games if needed
            self.pull_from_queue()

            if not self.active_games:
                break
            
            if self.check_for_pause():
                self.pause_wait_and_reload()

            # drain sf thread queue
            if self.sf_thread is not None:
                sf_results = self.sf_thread.drain()
                if sf_results:
                    for tup in sf_results:
                        if tup[0] == "__error__":
                            raise tup[1]

                        game_id, res_tup = tup
                        g = self.sf_games[game_id]
                        g.set_stockfish_result(res_tup)

            # usually we will have perfectly filled batches, but some games
            # may fill less than mbs, leaving room to pick extra leaves from other games
            likely_fill = min(fwd, mbs * min(len(self.active_games), cfg.games_at_once))
            batch_target = fwd
            for candidate in self.batch_candidates:
                if candidate >= likely_fill:
                    batch_target = candidate
                    break
            
            bonus = max(0, batch_target - likely_fill)
            finished = []
            preds_batch = []
            # selfplay loop starts here
            for game in self.active_games[:cfg.games_at_once]:
                sf_terminal, mcts_terminal = False, False
                # first resolve any recent preds
                game.tree.resolve_inflight()

                # if its stockfish turn, check if the move is ready
                # otherwise do not block and move on
                if game.is_stockfish_turn():
                    if not game.sf_pending:
                        # submit move to sf queue here
                        self.sf_thread.submit(game.game_id, game.python_chess_board)

                        # stash the game here for easy reference later
                        self.sf_games[game.game_id] = game
                        game.sf_pending = True
                    
                    if game.tree.sims_completed_this_move >= cfg.sf_move_sims:
                        if game.sf_pending and game.sf_ready:
                            # sets pending, ready, res_tup to False, False, None
                            sf_terminal = game.apply_stockfish_result(game.sf_res_tup)
                            mps.tick(1)
                        else:
                            bonus += mbs
                            continue
                
                # if this game has reached its local sim budget, make the move
                elif game.tree.stop_simulating():
                    # xerces plays from tree
                    mcts_terminal = game.make_move_from_tree()
                    mps.tick(1)

                # terminal check. do not sim on a finished game
                if sf_terminal or mcts_terminal:
                    self.finalize_game_data(game)
                    finished.append(game.game_id)
                    if game.game_id in self.sf_games:
                        del self.sf_games[game.game_id]
                    bonus += mbs
                    continue

                # if bonus available we can bump our microbatch up to 2x
                this_mbs = mbs + min(bonus, mbs) if bonus > 0 else mbs
                res = game.tree.collect_many_leaves(this_mbs, max_fastpath)
                nn, n_leafs = self.process_results(res, counts, this_mbs)
                
                # update bonus, could go up or down
                bonus += mbs - nn
                
                # update sim count
                game.tree.sims_completed_this_move += n_leafs
                if nn:
                    preds_batch += game.tree.pending_encoded_64_tokens()

            if preds_batch:
                batch_target, batch_size = self.format_and_predict(preds_batch)
                pred_fill.append((batch_size, batch_target))

            if self.maybe_push_telemetry(counts, pred_fill, force=False):
                self.prediction_times.clear()
                counts.clear(); pred_fill.clear()
            
            if finished:
                finished_ids.update(finished)
            
        self.maybe_push_telemetry(counts, pred_fill, force=True)
        return 0

    def process_results(self, res, counts, mbs):
        nn = res.count_new
        nt = res.count_terminal
        nc = res.count_cached

        pl = res.total_priorless
        pu = res.total_puct

        mv = res.total_must_visit
        wp = res.total_with_priors

        sk = res.total_skipped
        pr = res.total_pruned
        pen = res.total_penalty

        n_leafs = nn + nc + nt
        self.lps.tick(n_leafs)

        fastpaths = nt + nc
        f_stop, c_stop = 0, 1
        if nn < mbs:
            f_stop, c_stop = 1, 0

        counts.append([
            nn, fastpaths, nt, nc, f_stop, c_stop,
            pl, pu,
            mv, wp,
            sk, pr, pen
        ])
        return nn, n_leafs

    def format_and_predict(self, preds_batch):
        """
        preds_batch: list of items produced by tree.pending_encoded_64_tokens(...)
        expected shape:
            - (zobrist, board_np)

        Runs inference and writes into the raw policy cache as (zobrist, value, probs).
        Masking/softmax is applied in C++ build_priors.
        """

        if not preds_batch:
            return

        # collect arrays + keys
        boards = []
        keys = []

        for item in preds_batch:
            keys.append(item[0])
            boards.append(np.asarray(item[1], dtype=np.int32))

        # stack to batch
        boards_np = np.stack(boards, axis=0)   # (B,64)
        # pad up to max_batch or a smaller power of 2 if needed
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
                boards_np_p = np.concatenate([boards_np, pad_boards], axis=0)
                start = _now()
                probs_np_p, vals_np_p = self.infer((boards_np_p,))
                probs_np = probs_np_p[:B]
                vals_np = vals_np_p[:B]

            else:
                start = _now()
                probs_np, vals_np = self.infer((boards_np,))
        else:
            new_target = target_bs
            start = _now()
            probs_np, vals_np = self.infer((boards_np,))
        
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
        return new_target, B

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
        cfg = game.config

        # read adjudicator flags (direct attrs; they always exist)
        mat = cfg.use_material_diff
        tb = cfg.use_syzygy
        dr = cfg.use_eval_draw
        cl = cfg.use_eval_collar

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
            "c_puct": game.tree.c_puct,
            "dirichlet_eps": game.tree.dirichlet_eps(),
            "es_jsd_thresh": game.tree.es_jsd_thresh,
            "uniform_eps": cfg.uniform_eps,
            "prior_clip_max": cfg.prior_clip_max,
            "reuse_tree": cfg.reuse_tree
        }

        # on-disk record (full)
        res = {
            "history_uci": game.board.history_uci(),
            "moves_played": game.moves_played,
            "model_epoch": self.n_retrains
        }
        
        res.update(cfg.to_dict())
        res.update(mem_summary)
        res.update(game.meta)

        # unscale Q values from inference vscale before saving
        vs = cfg.vscale
        if vs and vs != 1.0:
            q_keys = ("Q_stm", "Q_white", "best_Q", "visit_weighted_Q")
            for td in game.tree_data.values():
                for k in q_keys:
                    if k in td:
                        td[k] = np.clip(td[k] / vs, -1.0, 1.0)

        # attach tree search data to disk record
        res["tree_search_data"] = game.tree_data
        res['c_puct'] = game.tree.c_puct
        res["dirichlet_eps"] = game.tree.dirichlet_eps()
        res["es_jsd_thresh"] = game.tree.es_jsd_thresh

        out_file = os.path.join(cfg.game_dir, game.game_id + "_log.pkl.gz")
        out_path = pathlib.Path(out_file)
        mem_summary['pkl_file'] = str(out_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # this is small, push it to parent process via Queue instead on appending
        self.recent_games_q.put({"looper_id": self.id, "meta": mem_summary})

        # this keeps games and plies in sync for logging
        self.update_partial_telemetry()
        
        # clear the game recents to prevent mem leaks
        game.recents.clear()
        return
    
    def maybe_push_telemetry(self, counts, pred_fill, every_sec=45.0, force=False):
        now = _now()
        if not force:
            if now - self._last_stats_log < every_sec:
                return False

        self._last_stats_log = now

        lpb = [p[0] for p in pred_fill]
        target = [p[1] for p in pred_fill]

        s_collected = sum([r[0] for r in counts])
        s_fast = sum([r[1] for r in counts])
        s_terminals = sum([r[2] for r in counts])
        s_cached = sum([r[3] for r in counts])
        s_fast_stops = sum([r[4] for r in counts])
        s_collect_stops = sum([r[5] for r in counts])

        s_priorless = sum([r[6] for r in counts])
        s_puct = sum([r[7] for r in counts])

        s_must_visit = sum([r[8] for r in counts])
        s_with_priors = sum([r[9] for r in counts])

        s_skipped = sum([r[10] for r in counts])
        s_pruned = sum([r[11] for r in counts])
        s_penalty = sum([r[12] for r in counts])

        telemetry = {
            "ts": now,
            "mps": self.mps.rate(),
            "lps": self.lps.rate(),
            "mbs": self.config.micro_batch,
            "apl": np.mean(lpb) if lpb else 0.0,
            "pred_wait": 0.0,
            "preds_per_second": 0.0,
            "batch_target": np.mean(target) if target else 0.0,
            "n_active": len(self.active_games),
            "avg_ply": 0.0,
            "n_groups": len(counts),

            "s_collected": s_collected,
            "s_fast": s_fast,
            "s_terminals": s_terminals,
            "s_cached": s_cached,
            "s_fast_stops": s_fast_stops,
            "s_collect_stops": s_collect_stops,

            "s_priorless": s_priorless,
            "s_puct": s_puct,

            "s_must_visit": s_must_visit,
            "s_with_priors": s_with_priors,

            "s_skipped": s_skipped,
            "s_pruned": s_pruned,
            "s_penalty": s_penalty,
        }

        if self.active_games:
            telemetry["avg_ply"] = np.mean([g.plies for g in self.active_games])

        # if self.tf_thread:
        #     tf_stats = self.tf_thread.stats()
        #     telemetry["pred_wait"] = tf_stats["mean_pred_s"]
        #     telemetry["preds_per_second"] = telemetry["apl"] / telemetry["pred_wait"]
        telemetry["pred_wait"] = np.mean(self.prediction_times) if self.prediction_times else 0.0
        telemetry["preds_per_second"] = (telemetry["apl"] / telemetry["pred_wait"]
                                         if telemetry["pred_wait"] else 0.0)

        pcs = priors_cache_stats()
        for k, v in pcs.items():
            telemetry[f"cache_{k}"] = v

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


def init_selfplay(config, recent_games_q, telemetry_q, msg_q, game_queue=None, sf_queue=None):
    # pre-built config (from yaml)
    model_name = config.run_tag + "_model.h5"

    if not getattr(config, "model_path", False):
        model_path = os.path.join(config.run_dir, model_name)
        config.model_path = model_path
    else:
        model_path = config.model_path

    if config.inference_backend == "pytorch":
        model = None
    elif os.path.exists(model_path):
        print(f"[init] Loading {model_name}")
        model = load_model(model_path)
    else:
        print(f"[init] Loading {config.init_model}")
        model = load_model(config.init_model)
        save_model(model, model_path)

    looper = GameLooper(
        model=model, cfg=config.copy(),
        recent_q=recent_games_q, telem_q=telemetry_q, msg_q=msg_q,
        game_queue=game_queue, sf_queue=sf_queue,
    )

    # infer number of retrains already done from existing progress csv
    if os.path.exists(config.progress_csv_path):
        progress_df = pd.read_csv(config.progress_csv_path)
        n_retrains = len(progress_df)
        looper.n_retrains = n_retrains

    return looper


class StockfishThread(object):
    def __init__(self, sf_config, depth):
        self.sf_loc = SF_LOC
        self.sf_config = sf_config
        self.depth = depth

        self.req_q = Queue()
        self.res_q = Queue()
        self.stop_ev = threading.Event()
        self.t = None

        self.eng = None
        self.err = None

    def start(self):
        if self.t is not None:
            return
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def close(self):
        self.stop_ev.set()
        if self.t is not None:
            self.t.join()
        if self.eng is not None:
            self.eng.quit()
        self.t = None
        self.eng = None

        if self.err is not None:
            raise self.err

    def submit(self, game_id, board):
        b = board.copy(stack=True)
        self.req_q.put((game_id, b))

    def drain(self, max_items=9999):
        out = []
        for _ in range(max_items):
            try:
                out.append(self.res_q.get_nowait())
            except Empty:
                break
        return out

    def run(self):
        self.eng = chess.engine.SimpleEngine.popen_uci(self.sf_loc)
        self.eng.configure(self.sf_config)

        while not self.stop_ev.is_set():
            try:
                game_id, board = self.req_q.get(timeout=0.01)
            except Empty:
                continue

            try:
                res_tup = sf_eval(
                    board, score_fn=score_to_value_stm_pov,
                    depth=self.depth, engine=self.eng
                )

                self.res_q.put((game_id, res_tup))
            except Exception as e:
                self.err = e
                self.res_q.put(("__error__", e))
                self.stop_ev.set()
                return
