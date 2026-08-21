import os, pickle, gzip
import pathlib
import time, gc

import threading
from queue import Queue, Empty

import chess
import chess.engine

from chessbot.review import score_to_value_stm_pov_tanh

now = time.time

import numpy as np
from pyfastchess import (
    raw_cache_bulk_insert_np, priors_cache_clear, priors_cache_stats, MCTSForest
)

from chessbot import SF_LOC

from chessbot.infer_ort_trt import make_ort_trt_infer

from chessbot.mcts_utils import ChessGame
from chessbot.utils import RateMeter, ema_step, sf_eval, next_model_epoch

# telemetry smoothing, counted in pushes rather than seconds. sims/move is
# deliberately faster: it should visibly react to validation and hard probes.
RATE_EMA_SPAN = 8
SPM_EMA_SPAN = 3

# priors-cache fields that accumulate in C++ rather than reading as a gauge
CACHE_COUNTERS = ("queries", "hits", "evictions")


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
        self.stop_at_empty = False
        self.games_finished = 0
        self.active_games = []
        self.sf_games = {}
        self.sf_thread = None  # lazy-initialized on first vs_stockfish game
        self.sf_teardown_pending = False
        self.forest = MCTSForest()

        self.pull_from_queue()

        self.model = model
        
        # pulls in model from config and XLA compilers inferencer
        self.load_reload_model()
        
        self.n_retrains = 0
        
        self.run_start = now()
        self.mps = RateMeter("moves")
        self.lps = RateMeter("leafs")
        self.mps_ema = None
        self.lps_ema = None
        self.spm_ema = None
        self.cache_hit_ema = None
        self.cache_prev = {}
        self.last_stats_log = now()
        self.prediction_times = []
        self.infer_gaps = []
        self.last_infer_end = None


    def tear_down_sf(self):
        """Drop the Stockfish engine. Safe to call when there is none.

        The engine holds the depth it was built with, so dropping it between
        validation batches is what lets a bumped ladder take effect.
        """
        if self.sf_thread is not None:
            self.sf_thread.close()
            self.sf_thread = None

    def close(self):
        self.tear_down_sf()



    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def load_reload_model(self):
        cfg = self.config
        encoding = cfg.encoding_type
        if encoding not in ("xc0", "lc0", "xc0h"):
            raise RuntimeError(f"unknown encoding_type {encoding!r}")

        # the encoder is chosen SOLELY by encoding_type -- never by the backend.
        if encoding == "lc0":
            self.batch_encoder = self.forest.get_all_lc0_features
        elif encoding == "xc0h":
            K = cfg.history_K
            self.batch_encoder = lambda: self.forest.get_all_history_tokens(K)
        else:
            self.batch_encoder = self.forest.get_all_encoded

        if cfg.inference_backend == "ort_trt":
            self.model, self.infer = make_ort_trt_infer(
                cfg.model_path,
                cfg.trt_model_name,
                cfg.trt_cache,
                cfg.macro_batch,
                encoding_type=encoding,
            )

        elif cfg.inference_backend == "lc0_trt":
            if encoding != "lc0":
                raise RuntimeError(
                    "lc0_trt inference requires encoding_type='lc0' (real Leela "
                    "reads 112x8x8 planes); got encoding_type=%r" % encoding)
            
            from chessbot.lc0_utils import make_lc0_trt_session, make_lc0_infer
            lc0_model = cfg.lc0_distill_model_name or os.getenv('LC0_DISTILL_MODEL', '')
            lc0_cache = os.getenv('LC0_DISTILL_TRT_CACHE', '')
            if not lc0_model or not lc0_cache:
                raise RuntimeError(
                    "lc0_trt requires lc0_distill_model_name and LC0_DISTILL_TRT_CACHE"
                )

            lc0_onnx = os.path.join(lc0_cache, f'{lc0_model}.onnx')
            sess = make_lc0_trt_session(
                lc0_onnx, lc0_model, lc0_cache, cfg.macro_batch,
            )
            self.model = sess
            self.infer = make_lc0_infer(sess)
    
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

            if cmd == "drain_and_stop":
                self.stop_at_empty = True
                self.active_games = self.active_games[:self.config.games_at_once]
                continue

            if cmd == "tear_down_sf":
                # never mid-game: an active vs_stockfish game still needs the
                # engine, so defer and let the run loop retry
                if any(g.vs_stockfish for g in self.active_games):
                    self.sf_teardown_pending = True
                else:
                    self.tear_down_sf()
                continue

            continue

    def pause_wait_and_reload(self):
        """
        Pause hard: stop TF preds, tear down GPU objects, then wait until "unpause".
        After "unpause", reload model + infer and return.

        If "unpause" arrived early and was buffered, do not block.
        """
        del self.infer
        self.infer = None

        del self.model
        self.model = None

        # ort_trt and lc0_trt manage their own device memory, so there is
        # nothing to reclaim here beyond the python side
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

        self.n_retrains += 1

    def pull_from_queue(self):
        from pyfastchess import Board as fastboard
        games_at_once = self.config.games_at_once
        # keep draining until full or both queues empty
        while len(self.active_games) < games_at_once:
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
                # depth is fixed for this engine's life; a moved ladder only
                # takes effect after a tear_down_sf rebuilds it
                self.sf_thread = StockfishThread(
                    spec.cfg.sf_config, spec.cfg.sf_validation_depth
                )
                self.sf_thread.start()
            game_cfg = spec.cfg
            board = fastboard(spec.fen)
            for mv in spec.moves:
                board.push_uci(mv)
            cg = ChessGame(board=board, meta=spec.meta, cfg=game_cfg)
            self.active_games.append(cg)
            self.forest.add_tree(cg.tree)

    def run(self, stop_event=None):
        """
        Main loop. Each round: for each game either let SF move (if applicable)
        or run MCTS step (collect/predict/apply).
        """

        cfg = self.config
        max_fastpath = 1024
        mps = self.mps

        # nn macro batch
        macro = cfg.macro_batch
        
        #(batch size, target), counts returned
        pred_fill, counts, mbs_used = [], [], []
        finished_ids = set()
        #passes_since_trigger = 0
        while True:
            # check a few stopping conditions 
            if stop_event is not None:
                if stop_event.is_set():
                    return
            
            if finished_ids:
                self.active_games = [
                    g for g in self.active_games
                    if g.game_id not in finished_ids
                ]
            
            # add in more games if needed
            self.pull_from_queue()

            if not self.active_games:
                if self.stop_at_empty:
                    break
                # queue transiently empty — wait up to 3min for parent to refill
                for _ in range(360):
                    time.sleep(0.5)
                    self.pull_from_queue()
                    if self.active_games:
                        break
                if not self.active_games:
                    break
            
            if self.check_for_pause():
                self.pause_wait_and_reload()

            # a teardown that arrived mid-game retries here until the last
            # vs_stockfish game has drained out
            if self.sf_teardown_pending:
                if not any(g.vs_stockfish for g in self.active_games):
                    self.tear_down_sf()
                    self.sf_teardown_pending = False

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

            bonus = 0
            finished = []
            mbs_used = []
            self.forest.resolve_all_inflight()
            for game in self.active_games[:cfg.games_at_once]:
                g_mbs = game.config.micro_batch
                sf_terminal, mcts_terminal = False, False

                # if its stockfish turn, check if the move is ready
                # otherwise do not block and move on
                if game.is_stockfish_turn():
                    if not game.sf_pending:
                        # submit move to sf queue here
                        self.sf_thread.submit(game.game_id, game.python_chess_board)

                        # stash the game here for easy reference later
                        self.sf_games[game.game_id] = game
                        game.sf_pending = True

                    if (game.tree.sims_completed_this_move
                            >= game.config.sf_move_sims):
                        if game.sf_pending and game.sf_ready:
                            # sets pending, ready, res_tup to False, False, None
                            sf_terminal = game.apply_stockfish_result(game.sf_res_tup)
                            mps.tick(1)
                        else:
                            bonus += g_mbs
                            continue

                # if this game has reached its local sim budget, make the move
                elif game.tree.stop_simulating():
                    # xerces plays from tree
                    mcts_terminal = game.make_move_from_tree()
                    mps.tick(1)

                # terminal check. do not sim on a finished game
                if sf_terminal or mcts_terminal:
                    self.forest.pop_tree(game.tree)
                    self.finalize_game_data(game)
                    finished.append(game.game_id)
                    if game.game_id in self.sf_games:
                        del self.sf_games[game.game_id]
                    bonus += g_mbs
                    continue

                # if bonus available we can bump our microbatch up by at most 2
                this_mbs = g_mbs + max(0, min(bonus, 2))
                res = game.tree.collect_many_leaves(this_mbs, max_fastpath)
                nn, n_leafs = self.process_results(res, counts, this_mbs)
                mbs_used.append(this_mbs)

                # update bonus, could go up or down
                bonus += g_mbs - nn

                # update sim count
                game.tree.sims_completed_this_move += n_leafs

            keys_np, enc_np = self.batch_encoder()
            if len(keys_np):
                for i in range(0, len(keys_np), macro):
                    pred_fill.append(
                        self.format_and_predict(keys_np[i:i+macro], enc_np[i:i+macro])
                    )

            if self.maybe_push_telemetry(counts, pred_fill, mbs_used, force=False):
                self.prediction_times.clear()
                self.infer_gaps.clear()
                counts.clear(); pred_fill.clear()
            
            if finished:
                finished_ids.update(finished)
            
        self.maybe_push_telemetry(counts, pred_fill, mbs_used, force=True)
        return 0

    def process_results(self, res, counts, mbs):
        nn = res.count_new
        nt = res.count_terminal
        nc = res.count_cached

        bl = res.total_blocked
        pu = res.total_puct

        mv = res.total_must_visit

        sk = res.total_skipped
        pr = res.total_pruned
        pen = res.total_penalty
        dp = res.total_depth

        n_leafs = nn + nc + nt
        self.lps.tick(n_leafs)

        fastpaths = nt + nc
        f_stop, c_stop = 0, 1
        if nn < mbs:
            f_stop, c_stop = 1, 0

        counts.append([
            nn, fastpaths, nt, nc, f_stop, c_stop,
            bl, pu, mv, sk, pr, pen, dp
        ])

        return nn, n_leafs

    def format_and_predict(self, keys_np, boards_np):
        """
        keys_np: uint64[B] zobrist keys
        boards_np: encoded boards from batch_encoder -- xc0 tokens (B, 64) int16
        or lc0 planes (B, 112, 8, 8) uint8, per cfg.encoding_type. self.infer owns
        the dtype/rule50 conversion for the active encoding.

        Runs inference and writes into the raw policy cache as (zobrist, value, probs).
        Masking/softmax is applied in C++ build_priors.
        """
        B = len(keys_np)
        if B == 0:
            return
        start = now()
        if self.last_infer_end is not None:
            self.infer_gaps.append(start - self.last_infer_end)

        probs_np, vals_np = self.infer((boards_np,))
        raw_cache_bulk_insert_np(keys_np, vals_np, probs_np)
        self.last_infer_end = now()
        self.prediction_times.append(self.last_infer_end - start)
        
        return B

    def finalize_game_data(self, game):
        """
        Attach the final scalar outcome to every per-move example and enqueue.
        Outcome is already white-POV (-1/0/+1) and does not need flipping.
        """

        scenario = game.meta.get("scenario", "")

        # replay probes are a handful of moves per position: accumulate in
        # memory and let the driver route them, instead of a pkl per position
        if scenario in ("blunder_replay_probe", "lc0_replay_probe"):
            brp_result_info = {
                "ts": now(),
                "game_id": game.game_id,
                "scenario": scenario,
                "vs_stockfish": game.vs_stockfish,
                "stockfish_color": game.stockfish_is_white,
                "duration": now() - game.started_at,
                "mcts_sims_total": game.mcts_sims_total,
                "mcts_plies": game.mcts_plies,
                "start_fen": game.starting_fen,
                "model_epoch": self.n_retrains,
                "moves_played": game.moves_played,
                "xerces_uci": game.moves_played[0] if game.moves_played else None,
                "tree_search_data": game.tree_data,
                "sims_done_total": game.tree.sims_done_total,
            }

            brp_result_info.update({**game.meta})

            # parent process will handle this based on the scenario key 
            self.recent_games_q.put({"looper_id": self.id, "meta": brp_result_info})

            game.examples = []
            game.recents.clear()
            return

        # game.config, not self.config: validation pins min_game_length to 1 so
        # an adjudicated game still reports and the batch can complete
        if game.plies <= game.config.min_game_length:
            game.examples = []
            return

        # aggregate stats
        self.games_finished += 1
        sims_total = game.tree.sims_done_total
        cfg = game.config

        # encode adjudicator flags: bit0=material, bit1=syzygy, bit2=eval_draw
        mat = cfg.use_material_diff
        tb = cfg.use_syzygy
        dr = cfg.use_eval_draw
        adjudication_index = (mat) | (tb << 1) | (dr << 2)

        # cast types for JSON 
        mem_summary = {
            "ts": now(),
            "game_id": game.game_id,
            "scenario": scenario,
            "plies": game.plies,
            "result": game.outcome or 0.0,
            "end_reason": game.end_reason,
            "vs_stockfish": game.vs_stockfish,
            "stockfish_color": game.stockfish_is_white,
            "duration": now() - game.started_at,
            "sims_done_total": sims_total,
            "mcts_sims_total": game.mcts_sims_total,
            "mcts_plies": game.mcts_plies,
            "start_fen": game.starting_fen,
            "adjudication_index": adjudication_index,
            "c_puct": cfg.c_puct,
            "dirichlet_eps": cfg.dirichlet_eps,
            "uniform_eps": cfg.uniform_eps,
        }

        # add any per-game sampled param values (specific value used, not the options)
        for param in cfg.sampleable:
            val = getattr(cfg, param)
            if param == 'move_sample_temp_range' and isinstance(val, list):
                mem_summary['move_sample_temp_min'] = val[0]
                mem_summary['move_sample_temp_max'] = val[1]
            elif isinstance(val, list):
                mem_summary[param] = ','.join(str(v) for v in val)
            else:
                mem_summary[param] = val

        # on-disk record (full)
        res = {
            "history_uci": game.board.history_uci(),
            "moves_played": game.moves_played,
            "model_epoch": self.n_retrains
        }
        
        res.update(cfg.to_dict())
        res.update(mem_summary)
        res.update(game.meta)

        # attach tree search data to disk record
        res["tree_search_data"] = game.tree_data

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
    
    def maybe_push_telemetry(self, counts, pred_fill, mbs_used, every_sec=60.0, force=False):
        ts_now = now()
        if not force:
            if ts_now - self.last_stats_log < every_sec:
                return False

        # the counters below cover exactly this span; the parent needs it to
        # normalise, since push and print cadences do not divide evenly
        window_s = ts_now - self.last_stats_log
        self.last_stats_log = ts_now

        lpb = pred_fill
        target = self.config.macro_batch

        # every column of counts is a plain sum; one pass instead of 13
        if counts:
            col = np.array(counts, dtype=np.int64).sum(axis=0)
        else:
            col = np.zeros(13, dtype=np.int64)

        (s_collected, s_fast, s_terminals, s_cached, s_fast_stops,
         s_collect_stops, s_blocked, s_puct, s_must_visit, s_skipped,
         s_pruned, s_penalty, s_depth) = (int(v) for v in col)

        # rate() is destructive, so each meter is read exactly once here
        raw_mps = self.mps.rate()
        raw_lps = self.lps.rate()
        self.mps_ema = ema_step(self.mps_ema, raw_mps, RATE_EMA_SPAN)
        self.lps_ema = ema_step(self.lps_ema, raw_lps, RATE_EMA_SPAN)
        raw_spm = raw_lps / raw_mps if raw_mps else 0.0
        self.spm_ema = ema_step(self.spm_ema, raw_spm, SPM_EMA_SPAN)

        telemetry = {
            "ts": ts_now,
            "window_s": window_s,
            "mps": self.mps_ema,
            "lps": self.lps_ema,
            "spm": self.spm_ema,
            "mbs": np.mean(mbs_used) if mbs_used else 0.0,
            "apl": np.mean(lpb) if lpb else 0.0,
            "infer_gap": np.mean(self.infer_gaps) if self.infer_gaps else 0.0,
            "pred_wait": 0.0,
            "preds_per_second": 0.0,
            "batch_target": target,
            "n_active": len(self.active_games),
            "avg_ply": 0.0,
            "n_groups": len(counts),

            "s_collected": s_collected,
            "s_fast": s_fast,
            "s_terminals": s_terminals,
            "s_cached": s_cached,
            "s_fast_stops": s_fast_stops,
            "s_collect_stops": s_collect_stops,

            "s_blocked": s_blocked,
            "s_puct": s_puct,

            "s_must_visit": s_must_visit,

            "s_skipped": s_skipped,
            "s_pruned": s_pruned,
            "s_penalty": s_penalty,
            "s_depth": s_depth,
        }
        tm = telemetry

        if self.active_games:
            tm["avg_ply"] = np.mean([g.plies for g in self.active_games])

        tm["pred_wait"] = np.mean(self.prediction_times) if self.prediction_times else 0.0

        tm["preds_per_second"] = 0.0
        if tm["pred_wait"]:
            tm["preds_per_second"] = tm["apl"] / tm["pred_wait"]

        # size and capacity are gauges; queries, hits and evictions are C++
        # counters that only reset at retrain, so they ship as per-window
        # deltas. A counter that went backwards means that reset happened.
        pcs = priors_cache_stats()
        for k, v in pcs.items():
            if k in CACHE_COUNTERS:
                prev = self.cache_prev.get(k, 0)
                tm[f"cache_{k}"] = v - prev if v >= prev else v
                self.cache_prev[k] = v
            else:
                tm[f"cache_{k}"] = v

        # smoothed on the ratio rather than the counts, so it stays meaningful
        # whatever the window length was
        d_q = tm.get("cache_queries", 0)
        if d_q:
            self.cache_hit_ema = ema_step(
                self.cache_hit_ema, tm.get("cache_hits", 0) / d_q,
                RATE_EMA_SPAN)
        if self.cache_hit_ema is not None:
            tm["cache_hit_ema"] = self.cache_hit_ema

        self.telemetry_q.put({
            "looper_id": self.id, "telemetry": tm,
            "kind": self.config.telemetry_kind,
        })

        return True
    
    def update_partial_telemetry(self):
        """Refresh the time-sensitive fields between full pushes.

        Deliberately carries no 'ts': the parent merges these into the last
        full push, so stamping ts here would make stale rates look fresh.
        """
        partial_telem = {
            "partial_ts": time.time(),
            "n_active": len(self.active_games),
            "avg_ply": 0.0
        }

        if self.active_games:
            partial_telem['avg_ply'] = np.mean([g.plies for g in self.active_games])

        self.telemetry_q.put({
            "looper_id": self.id, "telemetry": partial_telem,
            "kind": self.config.telemetry_kind,
        })


def init_selfplay(config, recent_games_q, telemetry_q, msg_q, game_queue=None, sf_queue=None):
    model = None  # loaded per-worker in load_reload_model

    looper = GameLooper(
        model=model, cfg=config.copy(),
        recent_q=recent_games_q, telem_q=telemetry_q, msg_q=msg_q,
        game_queue=game_queue, sf_queue=sf_queue,
    )

    # infer the current epoch from the progress csv. Row count is wrong once the
    # csv carries pretrain rows (written every Nth epoch), so use the same
    # max+1 rule the runner uses and stay in agreement with it.
    looper.n_retrains = next_model_epoch(config.progress_csv_path)

    return looper


class StockfishThread(object):
    def __init__(self, sf_config, depth):
        self.sf_loc = SF_LOC
        self.sf_config = {**sf_config, "UCI_ShowWDL": True}
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
        self.eng = chess.engine.SimpleEngine.popen_uci(self.sf_loc, timeout=60)
        self.eng.configure(self.sf_config)

        while not self.stop_ev.is_set():
            try:
                game_id, board = self.req_q.get(timeout=0.01)
            except Empty:
                continue

            try:
                res_tup = sf_eval(
                    board, score_fn=lambda s: score_to_value_stm_pov_tanh(s, mid_cp=150.0),
                    depth=self.depth, engine=self.eng
                )

                self.res_q.put((game_id, res_tup))
            except Exception as e:
                self.err = e
                self.res_q.put(("__error__", e))
                self.stop_ev.set()
                return
