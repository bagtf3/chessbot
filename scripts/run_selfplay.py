# run_selfplay.py
import os
import time
import queue as py_queue
import multiprocessing as mp
import sys
import json
from collections import defaultdict, deque
import pathlib

import numpy as np
from chessbot import SF_LOC, SP_DIR
from chessbot.looper import GameLooper, init_selfplay
from chessbot.rescore import Rescorer
from chessbot.config import Config
from chessbot.utils import print_recent_summary, summarize_recent_games, format_time

_now = time.time

PRINT_EVERY = 60.0
PROCESS_TIME = 10.0
ANALYSIS_BATCH = 30


def make_parent_queues():
    """
    Create mp queues in parent and return them.
    Create them from default ctx (platform default).
    """
    ctx = mp.get_context()
    recent_q = ctx.Queue()
    telemetry_q = ctx.Queue()
    return recent_q, telemetry_q


def spawn_workers(cfg, recent_q, telemetry_q):
    procs = []
    ctx = mp.get_context()
    n_workers = max(1, int(cfg.n_workers))
    for i in range(cfg.n_workers):
        c = cfg.copy()
        c.id = f"w{i}"
        p = ctx.Process(target=child_looper, args=(c, recent_q, telemetry_q))
        p.start()
        procs.append((p, c.id))
    return procs


def child_looper(cfg, recent_games_q, telemetry_q):
    with init_selfplay(cfg, recent_games_q, telemetry_q) as looper:
        looper.run()


def update_game_index(game, base_cfg):
    # update JSONL index (small)

    game['beat_sf'] = False
    if game['vs_stockfish']:
        game_result = game['result']
        if game_result > 0 and not game['stockfish_color']:
            game['beat_sf'] = True
        elif game_result < 0 and game['stockfish_color']:
            game['beat_sf'] = True
        else:
            game['beat_sf'] = False
    
    # append to JSONL index (create parent dirs if needed)
    idx_file = base_cfg.game_index_file
    os.makedirs(os.path.dirname(idx_file), exist_ok=True)
    with open(idx_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(game, ensure_ascii=False) + "\n")


class RecordKeeper(object):    
    def __init__(self, n_retrains, every_sec=60):
        self.n_retrains = n_retrains
        self.every_sec = every_sec
        self._last_stats_log = _now()
        self._run_start = _now()

        self.sims_done_total = 0
        self.moves_played = 0
        self.total_plies = 0
        self.games_finished = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.training_queue = 0

        self.recent_games = []
        self.telemetry = {}

    def ingest_recents(self, recent):
        looper_id = recent['looper_id']
        meta = recent['meta']

        self.games_finished += 1
        self.total_plies += meta['plies']
        self.sims_done_total += meta['sims_done_total']
        self.moves_played += meta['moves_played']

        if meta['result'] > 0:
            self.white_wins += 1
        elif meta['result'] < 0:
            self.black_wins += 1
        else:
            self.draws += 1
        
        self.recent_games.append(meta)

    def ingest_telemetry(self, telemetry):
        # store the latest from each only.
        looper_id = telemetry['looper_id']
        info = telemetry['telemetry']
        self.telemetry[looper_id] = info
        self.maybe_log_results()

    def get_agg_metrics(self):
        time_delta = _now() - 120
        to_sum = [
            "mps", "lps", "n_active", "n_groups", "s_collected", "s_fast",
            "s_terminals", "s_cached", "s_fast_stops", "s_collect_stops",
            "s_priorless", "s_puct", "preds_per_second"
        ]

        summed = defaultdict(float)
        sum_seen = set()

        to_avg = ['mbs', 'fwd_target', 'apl', "pred_wait", 'avg_ply']
        avged = defaultdict(list)
        avg_seen = set()
        for looper_id, info in self.telemetry.items():
            # if telemetry is timed out, skip it
            if info['ts'] < time_delta:
                continue
            for tosum in to_sum:
                summed[tosum] += info[tosum]
                sum_seen.add(tosum)

            for ta in to_avg:
                avged[ta].append(info[ta])
                avg_seen.add(ta)

        summed_out = {k: summed[k] for k in sorted(sum_seen)}
        avg_out = {k: np.mean(avged[k]) for k in sorted(avg_seen)}
        return summed_out, avg_out

    def maybe_log_results(self, window=500, force=False, run_num=None):
        now = _now()
        if not force and (now - self._last_stats_log < self.every_sec):
            return False
        self._last_stats_log = now

        avg_moves = (self.total_plies / max(1, self.games_finished))
        gph =  3600 * self.games_finished / (now - self._run_start)

        summed, avged = self.get_agg_metrics()
        print()
        if run_num is None:
            print("~"*72)
        else:
            print(f" Round {run_num} Logging ".center(72, "~"))
        
        mps, lps = summed.get("mps", 0), summed.get("lps", 0)
        print(f"[speed stats] mps={mps:.1f}  lps={lps:.1f}  gph={gph:.2f}")
        
        print(
            f"[game stats]  finished={self.games_finished}  "
            f"W/D/L={self.white_wins}/{self.draws}/{self.black_wins}  "
            f"avg_len={avg_moves:.1f} moves")
        print("-" * 72)

        recent = self.recent_games[-500:]
        if not recent:
            print("(no recent games to break down)")
            print("~" * 72)
            return True

        # pretty printer
        print_recent_summary(recent, window=window)
        print(
            f"Length of training queue: {self.training_queue} ",
            f"Current retrain number: {self.n_retrains}\n"
        )
        # chain log_loop_stats here as well
        self.log_loop_stats(sm)
        return

    def log_loop_stats(self, summed, avged):
        n_groups = summed.get("n_groups", 0)
        if n_groups == 0:
            return
        
        s_collected     = summed.get("s_collected", 0)
        s_fast          = summed.get("s_fast", 0)
        s_terminals     = summed.get("s_terminals", 0)
        s_cached        = summed.get("s_cached", 0)
        s_fast_stops    = summed.get("s_fast_stops", 0)
        s_collect_stops = summed.get("s_collect_stops", 0)
        s_priorless     = summed.get("s_priorless", 0)
        s_puct          = summed.get("s_puct", 0)

        avg_new = s_collected / n_groups

        total_overall = s_collected + s_terminals + s_cached
        term_to_cached = s_terminals / s_cached if s_cached > 0 else 0.0
        pct_cached_overall = 100.0 * s_cached / max(1, total_overall)
        pct_term_overall = 100.0 * s_terminals / max(1, total_overall)

        fast_stops_pct = 100.0 * s_fast_stops / max(1, n_groups)
        collect_stops_pct = 100.0 * s_collect_stops / max(1, n_groups)

        mbs = avged['mbs']
        print("-"*72)
        left1 = f"[loop stats] groups={n_groups}  mbs={mbs}"
        right1 = f"new: collected={s_collected} avg={avg_new:.2f}"

        left2 = f"[stop stats] fastpath_breaks={s_fast_stops} ({fast_stops_pct:.2f}%)"
        right2 = f"collect_breaks={s_collect_stops} ({collect_stops_pct:.2f}%)"

        # preds / active / finished runtime pre-compute
        apl = avged['apl']
        fwd_target = avged['fwd_target']
        fill_pct = 100.0 * apl / max(1.0, fwd_target)

        pred_wait = avged['pred_wait']
        preds_per_sec = summed['preds_per_second']

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

        n_active = summed['n_active']
        avg_ply = avged['avg_ply']
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python looper.py <run_tag>")
        sys.exit(1)

    run_tag = sys.argv[1]
    run_dir = os.path.join(SP_DIR, run_tag)

    if not os.path.isdir(run_dir):
        print(f"[error] run dir not found: {run_dir}")
        sys.exit(1)

    # find run config yaml
    yaml_path = None
    for nm in ("config.yaml", "config.yml"):
        p = os.path.join(run_dir, nm)
        if os.path.exists(p):
            yaml_path = p
            break

    if yaml_path is None:
        print(f"[error] no config.yaml or config.yml found in {run_dir}")
        sys.exit(1)

    # load base config and validation configs
    base_cfg = Config.from_yaml(yaml_path, init=True)

    # validation yaml path
    val_yaml_path = os.path.join(base_cfg.run_dir, "validation_config.yaml")

    # init the rescorer
    rescorer = Rescorer(base_cfg)

    start = time.time()
    n_games, run_num = 0, 1

    finished_games = []
    analyzed_games = []
    for selfplay_round in range(base_cfg.n_rounds):
        is_validation = False
        if selfplay_round > 0 & selfplay_round % base_cfg.validation_every == 0:
            is_validation = True

        if is_validation:
            working_cfg = Config.from_yaml(val_yaml_path, init=True)
        else:
            working_cfg = Config.from_yaml(yaml_path, init=True)
        
        recent_q, telemetry_q = make_parent_queues()
        procs = spawn_workers(working_cfg, recent_q, telemetry_q)

        # infer n_retrains
        if os.path.exists(working_cfg.progress_csv_path):
            progress_df = pd.read_csv(working_cfg.progress_csv_path)
            n_retrains = len(progress_df)
        else:
            n_retrains = 0
        
        recorder = RecordKeeper(n_retrains=n_retrains, every_sec=60.0)
        def alive(procs):
            return any([p.is_alive() for (p, cid) in procs])
        
        def keep_running(procs, training_samples):
            need_more = training_samples < working_cfg.training_queue_thresh
            return alive(procs) or need_more

        training_samples = 0
        while keep_running(procs, training_samples):
            # break here if no workers and no finished games
            if not alive(procs) and len(finished_games) == 0:
                break

            # check telemetry
            try:
                msg = telemetry_q.get_nowait()
                recorder.ingest_telemetry(msg)
            except py_queue.Empty:
                pass

            # drain recent_q into batch (non-blocking)
            while True:
                try:
                    game = recent_q.get_nowait()
                    recorder.ingest_recents(game)
                    update_game_index(game['meta'], base_cfg)
                    finished_games.append(game)
                except py_queue.Empty:
                    break
            
            recorder.maybe_log_results()

            process_start = time.time()
            # process games for a little bit then keep checking
            while time.time() < process_start + PROCESS_TIME:
                if not len(finished_games):
                    time.sleep(2.0)
                    break
                
                to_process = finished_games.pop(0)
                pkl_file = to_process['meta']['pkl_file']
                out = rescorer.analyze_and_rescore(pkl_file)
                analyzed_games.append(out)
                recorder.training_queue = rescorer.written_so_far
                if len(analyzed_games) >= 10:
                    print(f"[main loop] {training_samples} training samples saved so far")
                    print(f"[main loop] {len(finished_games)} games remaining to be processed")
                    analyzed_games = []
                    # need to write this out to pkl
        
        # when done
        recorder.maybe_log_results(force=True)

