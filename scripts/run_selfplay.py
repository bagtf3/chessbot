# run_selfplay.py
import os
import time
import queue as py_queue
import multiprocessing as mp
import sys
import json
from collections import defaultdict, deque
import pathlib

from chessbot import SF_LOC, SP_DIR
from chessbot.looper import GameLooper, init_selfplay
from chessbot.rescore import Rescorer
from chessbot.config import Config

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
    n_workers = min(1, int(cfg.n_workers))
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
                print("telemetry:", msg)
            except py_queue.Empty:
                pass

            # drain recent_q into batch (non-blocking)
            while True:
                try:
                    game = recent_q.get_nowait()
                    update_game_index(game['meta'], base_cfg)
                    finished_games.append(game)
                except py_queue.Empty:
                    break
            
            process_start = time.time()
            # process games for a little bit then keep checking
            while time.time() < process_start + PROCESS_TIME:
                if not len(finished_games):
                    time.sleep(2.0)
                    break
                
                to_process = finished_games.pop(0)
                pkl_file = to_process['meta']['pkl_file']
                print(f"[rescorer] processing {pkl_file}")
                out = rescorer.analyze_and_rescore(pkl_file)
                analyzed_games.append(out)
                training_samples = rescorer.written_so_far
                if len(analyzed_games) >= 10:
                    print(f"[main loop] {training_samples} training samples saved so far")
                    print(f"[main loop] {len(finished_games)} games remaining to be processed")
                    analyzed_games = []
                    # need to write this out to pkl

