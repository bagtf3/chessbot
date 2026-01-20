# run_selfplay.py
import os
import time
import queue as py_queue
import multiprocessing as mp
import sys
import json
from collections import deque
import pathlib

import numpy as np
import pandas as pd

from chessbot import SF_LOC, SP_DIR
from chessbot.looper import GameLooper, init_selfplay
from chessbot.rescore import Rescorer
from chessbot.review import RecordKeeper
from chessbot.config import Config
from chessbot.utils import make_jsonable, format_time
from chessbot.validation import build_validation_summary, create_validation_config


import signal
import threading

STOP_REQUESTED = threading.Event()


def request_stop(signum=None, frame=None):
    STOP_REQUESTED.set()


_now = time.time

PRINT_EVERY = 60.0
PROCESS_TIME = 20.0
ANALYSIS_BATCH = 30


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

    # just to be safe
    game = make_jsonable(game)

    # append to JSONL index (create parent dirs if needed)
    idx_file = base_cfg.game_index_file
    os.makedirs(os.path.dirname(idx_file), exist_ok=True)
    with open(idx_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(game, ensure_ascii=False) + "\n")


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
    ctx = mp.get_context()
    procs = []

    n_workers = max(1, cfg.n_workers)
    for i in range(n_workers):
        c = cfg.copy()
        c.id = f"w{i}"

        if (not cfg.is_validation_run) and (i > 0):
            c.play_vs_sf_prob = 0.0

        stop_ev = ctx.Event()
        p = ctx.Process(
            target=child_looper,
            args=(c, stop_ev, recent_q, telemetry_q),
        )
        p.start()

        procs.append({
            "id": c.id,
            "p": p,
            "stop_ev": stop_ev,
            "stop_sent_at": None,
            "term_sent_at": None,
            "kill_sent_at": None,
        })

    return procs


def child_looper(cfg, stop_ev, recent_games_q, telemetry_q):
    with init_selfplay(cfg, recent_games_q, telemetry_q) as looper:
        looper.run(stop_ev)


def check_and_reap_procs(procs, request_stop=False, grace_s=5.0, term_s=2.0):
    """
    - If a proc exited: join + close + remove from list.
    - If request_stop: set stop_ev once.
    - If still alive after grace_s: terminate.
    - If still alive after term_s more: kill (where supported).
    """
    now = time.monotonic()
    kept = []

    for w in procs:
        p = w["p"]

        if not p.is_alive():
            p.join(timeout=0)
            p.close()
            continue

        if request_stop and w["stop_sent_at"] is None:
            w["stop_ev"].set()
            w["stop_sent_at"] = now

        if w["stop_sent_at"] is not None:
            if w["term_sent_at"] is None:
                if (now - w["stop_sent_at"]) >= grace_s:
                    p.terminate()
                    w["term_sent_at"] = now

            elif w["kill_sent_at"] is None:
                if (now - w["term_sent_at"]) >= term_s:
                    if hasattr(p, "kill"):
                        p.kill()
                    else:
                        p.terminate()
                    w["kill_sent_at"] = now

        kept.append(w)

    return kept


def drain_queue(q):
    out = []
    if q is None:
        return out

    while True:
        try:
            out.append(q.get_nowait())
        except py_queue.Empty:
            break
    return out


def shutdown_round(procs, recent_q, telemetry_q, max_wait_s=10.0):
    start = time.monotonic()

    if procs:
        procs = check_and_reap_procs(procs, request_stop=True)

    while procs and (time.monotonic() - start) < max_wait_s:
        procs = check_and_reap_procs(procs, request_stop=True)
        time.sleep(0.05)

    # Even if some are still alive, close queues after we’ve tried; but ideally procs empty.
    if recent_q is not None:
        recent_q.close()
        recent_q.join_thread()

    if telemetry_q is not None:
        telemetry_q.close()
        telemetry_q.join_thread()


def parse_paths(run_tag):
    run_dir = os.path.join(SP_DIR, run_tag)

    if not os.path.isdir(run_dir):
        raise Exception(f"[error] run dir not found: {run_dir}")

    # find run config yaml
    yaml_path = None
    for nm in ("config.yaml", "config.yml"):
        p = os.path.join(run_dir, nm)
        if os.path.exists(p):
            yaml_path = p
            break

    if yaml_path is None:
        raise Exception(f"[error] no config.yaml or config.yml found in {run_dir}")

    # load base config and validation configs
    base_cfg = Config.from_yaml(yaml_path, init=True)

    # validation yaml path
    val_yaml_path = os.path.join(base_cfg.run_dir, "validation_config.yaml")
    return base_cfg, yaml_path, val_yaml_path


def main(run_tag):
    # build base config and work out the yaml paths
    base_cfg, yaml_path, val_yaml_path = parse_paths(run_tag)

    # init the rescorer
    rescorer = Rescorer(base_cfg)

    start = time.time()
    n_games, run_num = 0, 1

    finished_games = deque()
    analyzed_games = []
    procs = []
    cpl_list = []
    recent_q = None
    telemetry_q = None

    try:
        for selfplay_round in range(base_cfg.n_rounds):
            n_processed = 0
            if STOP_REQUESTED.is_set():
                break
            
            run_num = 1 + selfplay_round
            if run_num % base_cfg.validation_every == 0:
                is_validation = True
                working_cfg = create_validation_config(base_cfg, val_yaml_path)

            else:
                is_validation = False
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

            procs = check_and_reap_procs(procs)
            needed = working_cfg.training_queue_thresh
            while len(procs) or (recorder.training_queue < needed):
                if STOP_REQUESTED.is_set():
                    procs = check_and_reap_procs(procs, request_stop=True)
                    break
                
                # check for finished procs
                procs = check_and_reap_procs(procs)

                # break here if no workers and no finished games
                if not procs and len(finished_games) == 0:
                    break

                # check telemetry
                msgs = drain_queue(telemetry_q)
                for msg in msgs:
                    recorder.ingest_telemetry(msg)
                
                # drain recent_q into batch (non-blocking)
                pulled_games = drain_queue(recent_q)
                for game in pulled_games:
                    recorder.ingest_recents(game)
                    update_game_index(game['meta'], base_cfg)
                    if game['meta']['plies'] != game['meta']['n_moves_played']:
                        print("[meta mismatch]", game['meta']['plies'], game['meta']['n_moves_played'])
                    finished_games.append(game)
                
                recorder.maybe_log_results(run_num=run_num)

                process_start = time.time()
                # process games for a little bit then keep checking
                while time.time() < process_start + PROCESS_TIME:
                    if not len(finished_games):
                        time.sleep(2.0)
                        break
                    
                    to_process = finished_games.popleft()
                    pkl_file = to_process['meta']['pkl_file']
                    out = rescorer.analyze_and_rescore(pkl_file)
                    n_processed += 1
                    cpl_list.append(out['overall_cpl'])
                    analyzed_games.append(out)
                    recorder.training_queue = rescorer.written_so_far
                    # need to write this out to pkl

                if len(cpl_list) and (n_processed % 10 == 0):
                    print(f"[rescorer] avg CPL so far ({len(cpl_list)} games): {np.mean(cpl_list):.3f}")
                    print(
                        f"[rescorer] {n_processed} games processed this round | "
                        f"{len(finished_games)} waiting in queue"
                    )

                    print(f"[main loop] n procs: {len(procs)}: needed: {needed}, have: {recorder.training_queue}")
            
            # when done, close the queues
            shutdown_round(procs, recent_q, telemetry_q)
            procs = []
            recent_q = None
            telemetry_q = None
            recorder.training_queue = 0
            rescorer.written_so_far = 0

            # selfplay round report
            recorder.maybe_log_results(force=True, run_num=run_num)

            if is_validation:
                recorder.config = working_cfg
                build_validation_summary(recorder)
        
        # capture to return situation
        return 0 if not STOP_REQUESTED.is_set() else 1
    
    except KeyboardInterrupt:
        return 1
    
    finally:
        # if Ctrl+C happens mid-round, we land here and still attempt cleanup
        shutdown_round(procs, recent_q, telemetry_q)


if __name__ == "__main__":
    # build in gracefully exits
    signal.signal(signal.SIGINT, request_stop)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    # handle args
    if len(sys.argv) < 2:
        print("Usage: python looper.py <run_tag>")
        sys.exit(1)

    run_tag = sys.argv[1]
    main(run_tag)
    