import os, pickle
import pathlib, json
import time, gc
import sys, subprocess

_now = time.time

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")

import chess

from pyfastchess import terminal_value_white_pov, raw_cache_bulk_insert
from pyfastchess import raw_cache_clear, priors_cache_clear, priors_cache_stats

from chessbot import ENDGAME_LOC, SF_LOC
from chessbot.model import load_model, save_model, make_conv_infer
from chessbot.mcts_utils import MCTSTree, ChessGame
from chessbot.config import Config
from chessbot.validation import ValidationConfig, paired_validation_games
from chessbot.validation import build_validation_summary, append_validation_summary

import chessbot.utils as cbu
from chessbot.utils import rnd, RateMeter, softmax, GameGenerator
from chessbot.review import start_post_hoc_server, stop_post_hoc_server


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

        # different game logic for training vs validation
        if self.config.is_validation_run:
            self.active_games = paired_validation_games(self.config)
            self.collect_training_data = False
        else:
            self.fill_active_games()
            self.collect_training_data = True
        
        self.model = model

        cfg = self.config
        self.infer = make_conv_infer(
            self.model, max_bs=cfg.fwd_batch,
            min_p=cfg.prior_clip_min, max_p=cfg.prior_clip_max,
            vscale=cfg.vscale
        )

        self.infer_is_warm = False
        self.batch_candidates = self.create_batch_candidates(cfg)
        self.training_queue = []
        self.recent_games = []

        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_plies = 0
        self.n_retrains = 0
        self.clear_cache = False
        
        self._run_start = _now()
        self.mps = RateMeter("moves")
        self.lps = RateMeter("leafs")
        self._last_stats_log = 0.0
        self.prediction_times = []
        self.sf_search_depths = []

        self.moves_played = 0
        self.sims_done_total = 0

    def create_batch_candidates(self, cfg):
        # create sizes to warm up
        batch_candidates = set()
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
        sf_games = ['startpos', 'pre_opened', 'pre_opened_mini', 'random_init', 'paired_validation']

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
            if meta['scenario'] not in sf_games:
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
        mbs = cfg.micro_batch
        max_fastpath = max(200, int(2.5 * mbs))
        lpb, counts = [], []
        mps, lps = self.mps, self.lps
        with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
            eng.configure(cfg.sf_config)
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
                        
                        # if its stockfish turn, dont do any sims
                        #if game.is_stockfish_turn():
                        #    continue

                    # otherwise, collect up to micro_batch leaves for this game
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
                    game.tree.sims_completed_this_move += (n_leafs)

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

        # train with whatever we got and final report
        if self.collect_training_data:
            self.trigger_retrain()

        self.maybe_log_results(force=True, run_num=run_num)
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
        boards_np = np.stack(boards, axis=0)   # (B,64)
        legals_np = np.stack(legals, axis=0)   # (B,4288)
        # pad up to fwd_batch or a smaller power of 2 if needed
        # (helps XLA/static-trace shapes)

        target_bs = self.config.fwd_batch
        # need to make sure the tf.function is warm
        if not self.infer_is_warm:
            print(f"[model warmup] warming GPU for {self.batch_candidates}")
            for bs in self.batch_candidates:
                for _ in range(3):
                    rep_mask = (np.random.rand(target_bs, 4288) < 0.02).astype(np.int32)
                    rep_enc = (np.random.rand(target_bs, 64) < 0.32).astype(np.int32)
                    self.infer((rep_enc, rep_mask))
            self.infer_is_warm = True
            print("[model warmup] warm up complete")

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

        # we do not train on validation games to prevent leakage        
        if not self.collect_training_data:
            game.examples = []
            return
        
        # game outcome in white POV (1 = white win, -1 = black win)
        z = game.outcome if game.outcome is not None else 0.0
        g = game.plies or 0  # game length

        # use g-1 as denominator only when there are >= 2 plies; otherwise taper=0
        denom = max(1, g-1)

        vscale = self.config.vscale
        vscale = vscale if vscale > 0 else 1.0
        for x, mask, policy, vwq, ply, turn in game.examples:
            # linear taper of outcome in [0, 1]
            taper = 0.0 if z == 0 else ply / denom
            z_stm = z if turn else -z # flip based on stm pov
            z_tapered = taper * z_stm
            vwq = vwq/vscale # unscale back to [-1, 1]
            self.training_queue.append((x, mask, policy, z_stm, vwq, z_tapered))
        game.examples = []

    def trigger_retrain(self):
        """
        Pickles training queue and currnet config, starts retrain_worker in
        a subprocess to avoid memory leak and slowdowns.
        reloads model and infer on success.
        """

        if not self.training_queue:
            return
        
        if not self.collect_training_data:
            return
        
        # write training queue (list) to pkl
        run_dir = self.config.run_dir
        p_pending = os.path.join(run_dir, "pending_retrain.pkl")
        tmp_pending = p_pending + ".tmp"
        with open(tmp_pending, "wb") as f:
            pickle.dump(self.training_queue, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_pending, p_pending)

        # pickle config dict (cfg may have been updated at runtime)
        cfg_dict = self.config.to_dict()
        cfg_dict['n_retrains'] = self.n_retrains
        p_cfg = os.path.join(run_dir, "config.pkl")
        tmp_cfg = p_cfg + ".tmp"

        with open(tmp_cfg, "wb") as f:
            pickle.dump(cfg_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_cfg, p_cfg)

        # find retrain_worker.py
        retrain_script = cbu.find_script("retrain_worker.py", start_file=__file__)

        # delete the model to free up GPU RAM
        del self.model, self.infer
        gc.collect()

        # spawn worker and fail fast if it fails
        cmd = [sys.executable, retrain_script, "--run-dir", run_dir]
        print(f"[retrain] launching worker with {len(self.training_queue)} samples")
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        rc = proc.returncode
        if rc != 0:
            # print to stderr
            sys.stderr.write("[retrain] worker failed, aborting looper\n")
            sys.stderr.write(f"[retrain] exit_code={rc}\n")
            sys.stderr.write("=== WORKER STDOUT ===\n")
            sys.stderr.write((proc.stdout or "") + "\n")
            sys.stderr.write("=== WORKER STDERR ===\n")
            sys.stderr.write((proc.stderr or "") + "\n")
            # fail fast and loud
            raise RuntimeError("retrain worker failed; aborting looper")
        else:
            sys.stdout.write(proc.stdout or "")
        
        # if here, retrain was a success. load new model and make new infer
        # make sure memory is clean to prevent slowdowns
        #cfg = self.config
        #del self.model, self.infer
        #gc.collect()

        self.infer_is_warm = False

        # clear training queue
        self.training_queue = []
        self.n_retrains += 1
        self.clear_cache = True

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


def init_selfplay(validation=False):
    # choose config type and ensure paths are initialized
    config = Config() if not validation else ValidationConfig()

    model_name = config.run_tag + "_model.h5"
    model_path = os.path.join(config.run_dir, model_name)
    config.model_path = model_path
    Config.model_path = model_path

    if os.path.exists(model_path):
        print(f"[init] Loading {model_name}")
        model = load_model(model_path)
    else:
        print(f"[init] Loading {config.init_model}")
        model = load_model(config.init_model)
        save_model(model, model_path)

    looper = GameLooper(model=model, cfg=config)

    # infer number of retrains already done from existing progress csv
    if os.path.exists(config.progress_csv_path):
        try:
            progress_df = pd.read_csv(config.progress_csv_path)
            n_retrains = len(progress_df)
            looper.n_retrains = n_retrains
        except Exception as e:
            print("[init_selfplay] failed reading progress csv:", e)

    return looper, config

#%%
if __name__ == '__main__':
    cfg = Config()
    phs = None
    start = _now()
    n_games, run_num = 0, 1
    try:
        while run_num <= cfg.n_rounds:
            if n_games:
                elapsed = _now() - start
                rt = cbu.format_time(elapsed)
                gph = 3600 * n_games / elapsed
                print(
                    f"[main loop] Time: {rt} ",
                    f"Games: {n_games} ({gph:.1f}/hr)"
                )

            is_validation = run_num % cfg.validation_every == 0
            looper, cfg = init_selfplay(is_validation)
            
            # use endgame tables on alternating runs to balance speed and learning
            if not is_validation:
                cfg.use_syzygy = bool(run_num % 2)

            # look to start post hoc server on the first loop if its not yet running
            if phs is None and cfg.run_post_hoc and cfg.run_dir:
                phs = start_post_hoc_server(cfg.run_dir, bonus_data=cfg.mine_bonus_data)

            looper.run(run_num)
            run_num += 1
            if is_validation:
                summary = build_validation_summary(looper)
                append_validation_summary(looper.config.run_dir, summary)
            
            n_games += looper.games_finished

    finally:
        if phs is not None:
            stop_post_hoc_server(phs, timeout=10)
