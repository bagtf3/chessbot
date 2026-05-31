# run_selfplay.py
import os
import time
import queue as py_queue
import multiprocessing as mp
import sys
import json

import numpy as np
import pandas as pd

from chessbot import SP_DIR, SF_SEED_CACHE
from chessbot.looper import init_selfplay
from chessbot.rescore import (
    Rescorer, SFRescoreThread, SFCache,
    launch_retrain_async, poll_retrain, reclaim_vram,
)
from chessbot.review import RecordKeeper
from chessbot.config import Config
from chessbot.utils import make_jsonable, format_time, find_script
from chessbot.validation import build_validation_summary, create_validation_config
from chessbot.game_utils import GameGenerator, GameSpec, resolve_cfg

import pickle
import signal
import threading
import queue

STOP_REQUESTED = threading.Event()

MAX_BACKLOG = 250
GAME_QUEUE_MIN = 36  # top up when central queue drops below this


def request_stop(signum=None, frame=None):
    STOP_REQUESTED.set()


def update_game_index(game, base_cfg):
    # update JSONL index
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
    ctx = mp.get_context()
    recent_q = ctx.Queue()
    telemetry_q = ctx.Queue()
    return recent_q, telemetry_q


def spawn_workers(cfg, recent_q, telemetry_q, game_queue, sf_queue=None):
    ctx = mp.get_context()
    procs = []

    n_workers = max(1, cfg.n_workers)
    for i in range(n_workers):
        c = cfg.copy()
        c.id = f"w{i}"

        stop_ev = ctx.Event()
        msg_q = ctx.Queue()

        # worker 0 gets the sf_queue; all others do pure selfplay
        worker_sf_q = None
        if i == 0:
            worker_sf_q = sf_queue

        p = ctx.Process(
            target=child_looper,
            args=(c, stop_ev, recent_q, telemetry_q, msg_q, game_queue, worker_sf_q),
            daemon=True,
        )
        p.start()

        procs.append({
            "id": c.id,
            "p": p,
            "stop_ev": stop_ev,
            "msg_q": msg_q,
            "stop_sent_at": None,
            "term_sent_at": None,
            "kill_sent_at": None,
        })

    return procs


def child_looper(
    cfg,
    stop_ev,
    recent_games_q,
    telemetry_q,
    msg_q,
    game_queue,
    sf_queue=None,
):
    if cfg.inference_backend == "pt_eager":
        import torch
        torch.backends.cudnn.benchmark = True
    with init_selfplay(
        cfg, recent_games_q, telemetry_q, msg_q,
        game_queue=game_queue, sf_queue=sf_queue,
    ) as looper:
        looper.run(stop_ev)


def top_up_queues(game_queue, sf_queue, game_gen, rscr=None, budget=None, target=None):
    added = 0
    if target is None:
        target = GAME_QUEUE_MIN * 2

    sf_target = target // 2

    # drain pending blunder replays onto sf_queue before filling with regular games
    if rscr is not None and sf_queue is not None:
        while rscr.blunder_replay_specs:
            if budget is not None and added >= budget:
                break

            spec_data = rscr.blunder_replay_specs.pop(0)
            meta = {
                'scenario': 'blunder_replay',
                'vs_stockfish': True,
                'stockfish_is_white': spec_data['stockfish_is_white'],
            }

            game_spec_cfg = resolve_cfg(game_gen.config)

            # update config on blunder replay for deeper search
            game_spec_cfg.sample_moves = False
            game_spec_cfg.sf_move_sims = max(300, game_spec_cfg.sf_move_sims)
            
            sf_queue.put(GameSpec(
                fen=spec_data['fen'], moves=spec_data['moves'],
                meta=meta, cfg=game_spec_cfg,
            ))
            
            added += 1

    while budget is None or added < budget:
        game_ok = game_queue.qsize() >= target
        sf_ok = sf_queue is None or sf_queue.qsize() >= sf_target
        if game_ok or sf_ok:
            break
        spec = game_gen.next_game()
        if spec.meta.get("vs_stockfish") and sf_queue is not None:
            sf_queue.put(spec)
        else:
            game_queue.put(spec)
        added += 1

    return added


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


def reclaim_vram_worker(mb):
    reclaim_vram(mb)


def launch_retrain(run_tag, working_cfg, epoch=0):
    rt_script = find_script("retrain_worker.py", start_file=__file__)
    if not rt_script:
        raise RuntimeError("retrain_worker.py not found")
    return launch_retrain_async(run_tag, rt_script, working_cfg, epoch=epoch)


def pull_pkl(to_process):
    # might be nested or flat depending on where it came from
    if 'meta' in to_process.keys():
        return to_process['meta']['pkl_file']
    else:
        return to_process['pkl_file']


def main(run_tag):
    # build base config and work out the yaml paths
    base_cfg, yaml_path, val_yaml_path = parse_paths(run_tag)


    # init the async SF worker(s) and rescorer
    sf_game_q = queue.Queue()
    sf_res_q  = queue.Queue()
    sf_rescore_threads = [
        SFRescoreThread(sf_game_q, sf_res_q, base_cfg)
        for _ in range(max(1, base_cfg.rescore_n_sf_threads))
    ]

    for t in sf_rescore_threads:
        t.start()

    cache = SFCache(
        eviction_window=base_cfg.rescore_eviction_window,
        max_size=base_cfg.rescore_cache_size,
    )
    cache_path = os.path.join(base_cfg.run_dir, "sf_cache.pkl.gz")

    if SF_SEED_CACHE and os.path.exists(SF_SEED_CACHE):
        cache.load_seed(SF_SEED_CACHE)
    if os.path.exists(cache_path):
        cache.roll_merge(cache_path)
    rescorer = Rescorer(base_cfg, sf_game_q, sf_res_q, cache)
    finished_games = rescorer.get_unprocessed()

    # load any previously saved untrained samples
    remaining_pkl = os.path.join(base_cfg.run_dir, "remaining_untrained.pkl")
    if os.path.exists(remaining_pkl):
        with open(remaining_pkl, "rb") as f:
            rescorer.training_data = pickle.load(f)
        os.remove(remaining_pkl)
        n_loaded = len(rescorer.training_data)
        print(f"[main] loaded {n_loaded} samples from remaining_untrained.pkl")

    # infer n_retrains
    if os.path.exists(base_cfg.progress_csv_path):
        progress_df = pd.read_csv(base_cfg.progress_csv_path)
        n_retrains = len(progress_df)
    else:
        n_retrains = 0
    
    recorder = RecordKeeper(n_retrains, run_num=0, every_sec=45.0)

    start = time.time()
    procs = []
    recent_q = None
    telemetry_q = None
    game_queue = None
    sf_queue = None
    game_gen = None
    retrain = None
    total_games = 0
    try:    
        for selfplay_round in range(base_cfg.n_rounds):
            run_num = 1 + selfplay_round
            recorder = RecordKeeper(n_retrains, run_num=run_num, every_sec=45.0)
            recorder.training_queue = len(rescorer.training_data)

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
                rescorer_pending = len(rescorer.intake) + len(rescorer.pending)
                print(f"[main loop] {len(finished_games)} unprocessed + {rescorer_pending} in rescorer queue")

            # fresh reload each pass
            working_cfg = Config.from_yaml(yaml_path, init=True)
            is_validation = False

            if run_num % base_cfg.validation_every == 0:
                is_validation = True
                working_cfg = create_validation_config(working_cfg, val_yaml_path)
            
            # update the rescorer config
            rescorer.config = working_cfg
            recent_q, telemetry_q = make_parent_queues()

            ctx = mp.get_context()
            game_gen = GameGenerator(working_cfg)

            if is_validation:
                game_queue = ctx.Queue()
                sf_queue = None
                for spec in game_gen.validation_games():
                    game_queue.put(spec)
                procs = spawn_workers(working_cfg, recent_q, telemetry_q, game_queue)
                for w in procs:
                    w["msg_q"].put("drain_and_stop")
            else:
                game_queue = ctx.Queue()
                sf_queue = ctx.Queue()
                n_workers = max(1, working_cfg.n_workers)
                n_games = working_cfg.n_games
                initial_target = n_workers * working_cfg.games_at_once + GAME_QUEUE_MIN
                total_queued = top_up_queues(
                    game_queue, sf_queue, game_gen, rescorer,
                    budget=n_games, target=initial_target,
                )

                procs = spawn_workers(
                    working_cfg, recent_q, telemetry_q, game_queue, sf_queue
                )

            # validation is pre-filled and already signaled; training tracks budget below
            stop_signal_sent = is_validation

            # infer n_retrains
            if os.path.exists(working_cfg.progress_csv_path):
                progress_df = pd.read_csv(working_cfg.progress_csv_path)
                n_retrains = len(progress_df)
            else:
                n_retrains = 0

            procs = check_and_reap_procs(procs)
            needed_to_retrain = working_cfg.training_queue_buffer
            sf_throttled = False
            while len(procs) or (recorder.training_queue < needed_to_retrain):
                if STOP_REQUESTED.is_set():
                    procs = check_and_reap_procs(procs, request_stop=True)
                    break
                
                # check for finished procs
                procs = check_and_reap_procs(procs)

                # break if no workers and backlog is small enough to carry into next round
                if not procs and len(finished_games) < MAX_BACKLOG:
                    break

                # refill queues; blunder replays trigger a top-up even if queues aren't low
                if not stop_signal_sent:
                    has_blunders = bool(rescorer.blunder_replay_specs)
                    sf_low = sf_queue is not None and sf_queue.qsize() < GAME_QUEUE_MIN // 2
                    game_low = game_queue.qsize() < GAME_QUEUE_MIN
                    if has_blunders or sf_low or game_low:
                        added = top_up_queues(
                            game_queue, sf_queue, game_gen, rescorer,
                            budget=n_games - total_queued,
                        )
                        total_queued += added
                        # n_games reached — tell all workers to drain and exit
                        if total_queued >= n_games:
                            for w in procs:
                                w["msg_q"].put("drain_and_stop")
                            stop_signal_sent = True

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

                # submit all pending games and tick the rescorer pipeline
                while finished_games:
                    to_process = finished_games.popleft()
                    rescorer.submit(pull_pkl(to_process))
                rescorer.tick()

                sf_backlog = len(rescorer.intake) + len(rescorer.pending)
                if sf_backlog > 100 and not sf_throttled:
                    for t in sf_rescore_threads:
                        t.depth = max(1, t.base_depth - 1)
                    rescorer.current_depth = sf_rescore_threads[0].depth
                    sf_throttled = True
                    print(f"[rescore] backlog {sf_backlog}, depth -> {rescorer.current_depth}")
                elif sf_backlog < 10 and sf_throttled:
                    for t in sf_rescore_threads:
                        t.depth = t.base_depth
                    rescorer.current_depth = sf_rescore_threads[0].depth
                    sf_throttled = False
                    print(f"[rescore] backlog cleared, depth -> {rescorer.current_depth}")

                recorder.training_queue = rescorer.training_data_size

                # check for a retrain
                if recorder.training_queue >= needed_to_retrain:
                    # drain so write gets accurate data; workers keep playing meanwhile
                    k = max(1, recorder.training_queue // working_cfg.retrain_size)
                    rescorer.write_training_data_pkl(
                        size=k * working_cfg.retrain_size, randomize=True)
                    recorder.training_queue = rescorer.training_data_size

                    # pause workers before reclaim + launch
                    for p in procs:
                        p["msg_q"].put("pause")

                    if retrain is None:
                        retrain = launch_retrain(run_tag, working_cfg, epoch=n_retrains)
                        rescorer.reset_writer()

                    while retrain is not None:
                        done, rc = poll_retrain(retrain, print_output=True)
                        if done:
                            retrain = None
                            recorder.n_retrains += 1
                            working_cfg = Config.from_yaml(yaml_path, init=True)
                            if is_validation:
                                working_cfg = create_validation_config(working_cfg, val_yaml_path)
                            rescorer.config = working_cfg
                            for t in sf_rescore_threads:
                                t.update_config(working_cfg)
                            sf_throttled = False
                            rescorer.aggregate_metrics(
                                n_retrains, working_cfg.vscale,
                                working_cfg.progress_csv_path)
                            n_retrains += 1

                        # keep submitting games while waiting on retrain
                        pulled_games = drain_queue(recent_q)
                        for game in pulled_games:
                            recorder.ingest_recents(game)
                            update_game_index(game['meta'], base_cfg)
                            finished_games.append(game)
                        while finished_games:
                            to_process = finished_games.popleft()
                            rescorer.submit(pull_pkl(to_process))
                        rescorer.tick()

                        time.sleep(0.05)

                    # unpause workers
                    for p in procs:
                        p["msg_q"].put("unpause")

                else:
                    time.sleep(0.5)
            
            # when done, close the queues
            procs = shutdown_round(procs, recent_q, telemetry_q)
            if procs:
                print(f"[warn] {len(procs)} workers still alive after shutdown")

            if sf_queue is not None:
                while True:
                    try:
                        spec = sf_queue.get_nowait()
                        if spec.meta.get('scenario') == 'blunder_replay':
                            rescorer.blunder_replay_specs.append({
                                'fen': spec.fen,
                                'moves': spec.moves,
                                'stockfish_is_white': spec.meta['stockfish_is_white']
                            })
                    
                    except Exception:
                        break

            for q in (game_queue, sf_queue):
                if q is not None:
                    q.cancel_join_thread()
                    q.close()

            procs = []
            recent_q = None
            telemetry_q = None
            game_queue = None
            sf_queue = None
            game_gen = None

            # selfplay round report
            recorder.maybe_log_results(force=True)
            total_games += recorder.games_finished
            cache.save(cache_path)

            if is_validation:
                recorder.config = working_cfg
                build_validation_summary(recorder)
                # we do not train after validation currently
                continue

        # drain any remaining games before exit
        remaining = len(finished_games) + len(rescorer.intake) + len(rescorer.pending)
        if remaining and not STOP_REQUESTED.is_set():
            print(f"[main] {remaining} games remaining after all rounds — continuing post-process")
            while (finished_games or rescorer.intake or rescorer.pending) and not STOP_REQUESTED.is_set():
                while finished_games:
                    to_process = finished_games.popleft()
                    rescorer.submit(pull_pkl(to_process))
                rescorer.tick()
                recorder.training_queue = rescorer.training_data_size
                if recorder.training_queue >= needed_to_retrain:
                    k = max(1, recorder.training_queue // working_cfg.retrain_size)
                    rescorer.write_training_data_pkl(size=k * working_cfg.retrain_size, randomize=True)
                    recorder.training_queue = rescorer.training_data_size
                    if retrain is None:
                        retrain = launch_retrain(run_tag, working_cfg, epoch=n_retrains)
                        rescorer.reset_writer()
                    while retrain is not None:
                        done, rc = poll_retrain(retrain, print_output=True)
                        if done:
                            retrain = None
                            recorder.n_retrains += 1
                            working_cfg = Config.from_yaml(yaml_path, init=True)
                            rescorer.config = working_cfg
                            rescorer.aggregate_metrics(
                                n_retrains, working_cfg.vscale, working_cfg.progress_csv_path)
                            n_retrains += 1
                        while finished_games:
                            to_process = finished_games.popleft()
                            rescorer.submit(pull_pkl(to_process))
                        rescorer.tick()
                        time.sleep(0.05)
                else:
                    time.sleep(0.1)

        # capture the return situation
        rescorer.tick()
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
        rescorer.tick()
        rescorer.push_analyzed(report=True)
        cache.save(cache_path)
        for t in sf_rescore_threads:
            t.close()
        if rescorer.training_data:
            remaining_pkl = os.path.join(
                base_cfg.run_dir, "remaining_untrained.pkl"
            )
            with open(remaining_pkl, "wb") as f:
                pickle.dump(rescorer.training_data, f)
            n_saved = len(rescorer.training_data)
            print(f"[main] saved {n_saved} samples to remaining_untrained.pkl")
        rescorer.close()
        for q in (game_queue, sf_queue):
            if q is not None:
                try:
                    q.cancel_join_thread()
                    q.close()
                except Exception:
                    pass
        shutdown_round(procs, recent_q, telemetry_q)


if __name__ == "__main__":
    # build in graceful exits
    signal.signal(signal.SIGINT, request_stop)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    # handle args
    if len(sys.argv) < 2:
        print("Usage: python looper.py <run_tag>")
        sys.exit(1)

    run_tag = sys.argv[1]
    main(run_tag)
    