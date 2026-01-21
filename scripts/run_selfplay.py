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
from chessbot.rescore import Rescorer, launch_retrain_async, poll_retrain
from chessbot.review import RecordKeeper
from chessbot.config import Config
from chessbot.utils import make_jsonable, format_time, find_script
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

        # customize workers
        if not cfg.is_validation_run:
            # only SF on first worker (saves CPU)
            if i > 0:
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


def shutdown_round(procs, recent_q, telemetry_q, max_wait_s=15.0):
    deadline = time.monotonic() + max_wait_s

    # Ask nicely first and start escalation timers
    procs = check_and_reap_procs(procs, request_stop=True)

    while procs and time.monotonic() < deadline:
        procs = check_and_reap_procs(procs, request_stop=True)
        time.sleep(0.05)

    # Hard guarantee: if anything survived, kill it and reap it now.
    if procs:
        for w in procs:
            p = w["p"]
            if p.is_alive():
                p.terminate()

        # Give terminate a moment, then kill anything still alive.
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            procs = check_and_reap_procs(procs, request_stop=False)
            if not procs:
                break
            time.sleep(0.05)

        for w in procs:
            p = w["p"]
            if p.is_alive():
                if hasattr(p, "kill"):
                    p.kill()
                else:
                    p.terminate()

        # Final reap pass (blocking join is OK here; they're supposed to be dead)
        for w in list(procs):
            p = w["p"]
            p.join(timeout=2.0)
            if not p.is_alive():
                p.close()

        procs = [w for w in procs if w["p"].is_alive()]

    # Close queues at the very end (after children are gone)
    if recent_q is not None:
        recent_q.close()
        recent_q.join_thread()

    if telemetry_q is not None:
        telemetry_q.close()
        telemetry_q.join_thread()

    return procs


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


def launch_retrain(run_tag, working_cfg, n_samples):
    rt_script = find_script("retrain_worker.py", start_file=__file__)
    if not rt_script:
        raise RuntimeError("retrain_worker.py not found")

    return launch_retrain_async(run_tag, rt_script, working_cfg, n_samples)


def main(run_tag):
    # build base config and work out the yaml paths
    base_cfg, yaml_path, val_yaml_path = parse_paths(run_tag)

    # init the rescorer
    rescorer = Rescorer(base_cfg)
    finished_games = rescorer.get_unprocessed()

    start = time.time()
    procs = []
    cpl_list = []
    recent_q = None
    telemetry_q = None
    retrain = None
    total_games = 0
    try:    
        for selfplay_round in range(base_cfg.n_rounds):
            run_num = 1 + selfplay_round

            n_processed = 0
            next_print = 10
            if STOP_REQUESTED.is_set():
                break

            print("#"*72)
            print(f"   Starting Round {run_num}   ".center(72, "#"))
            print("#"*72)
            if run_num > 1:
                run_time = time.time() - start
                gph = 3600 * total_games / run_time
                print(f"[main loop] Run Time: {format_time(run_time)}")
                print(f"[main loop] Games Completed {total_games} ({gph:.2f} per hour)")
                print(f"[main loop] {len(finished_games)} unprocessed games in the queue")
            
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
            
            recorder = RecordKeeper(n_retrains, run_num, every_sec=45.0)

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
                    finished_games.append(game)
                
                recorder.maybe_log_results()

                process_start = time.time()
                # process games for a little bit then keep checking
                while time.time() < process_start + PROCESS_TIME:
                    if not len(finished_games):
                        time.sleep(2.0)
                        break
                    
                    to_process = finished_games.popleft()
                    # might be nested or flat depending on where it came from
                    if 'meta' in to_process.keys():
                        pkl_file = to_process['meta']['pkl_file']
                    else:
                        pkl_file = to_process['pkl_file']
                    rescorer.analyze_and_rescore(pkl_file)
                    recorder.training_queue = rescorer.written_this_round
                    # need to write this out to pkl
            
            # when done, close the queues
            procs = shutdown_round(procs, recent_q, telemetry_q)
            if procs:
                print(f"[warn] {len(procs)} workers still alive after shutdown")
            
            procs = []
            recent_q = None
            telemetry_q = None

            # selfplay round report
            recorder.maybe_log_results(force=True)
            total_games += recorder.games_finished

            if is_validation:
                recorder.config = working_cfg
                build_validation_summary(recorder)
                # we do not train after validation currently
                continue
            
            # check if we have enough to run retraining
            n_samples = rescorer.written_this_round
            if n_samples >= needed:
                if retrain is None:
                    retrain = launch_retrain(run_tag, working_cfg, n_samples)
                    rescorer.reset_writer()
                while retrain is not None:
                    done, rc = poll_retrain(retrain, print_output=True)
                    if done:
                        retrain = None
                    
                    if len(finished_games):
                        to_process = finished_games.popleft()
                        # might be nested or flat depending on where it came from
                        if 'meta' in to_process.keys():
                            pkl_file = to_process['meta']['pkl_file']
                        else:
                            pkl_file = to_process['pkl_file']
                        rescorer.analyze_and_rescore(pkl_file)
                    
                    else:
                        time.sleep(0.05)

        # capture the return situation
        rescorer.push_analyzed(report=True)
        rescorer.close()
        alive = mp.active_children()
        if alive:
            print("[warn] active children at end:", [p.pid for p in alive])

        return 0 if not STOP_REQUESTED.is_set() else 1
    
    except KeyboardInterrupt:
        return 1
    
    finally:
        # if Ctrl+C happens mid-round, we land here and still attempt cleanup
        rescorer.push_analyzed(report=True)
        rescorer.close()
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
    