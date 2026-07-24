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
    launch_retrain_async, poll_retrain,
)
from chessbot.review import RecordKeeper
from chessbot.config import Config
from chessbot.utils import make_jsonable, format_time, find_script
from chessbot.validation import build_validation_summary, create_validation_config
from chessbot.infer_ort_trt import prepare_trt, selfplay_trt_paths
from chessbot.game_utils import GameGenerator, GameSpec, resolve_cfg

import pickle
import signal
import threading
import queue

STOP_REQUESTED = threading.Event()
STOP_AFTER_ROUND_REQUESTED = threading.Event()
NEXT_ROUND_REQUESTED = threading.Event()
RELOAD_REQUESTED = threading.Event()
PAUSE_REQUESTED = threading.Event()
UNPAUSE_REQUESTED = threading.Event()
SAVE_TRAINING_DATA_REQUESTED = threading.Event()
SAVE_TRAINING_DATA_N = None

MAX_BACKLOG = 250
GAME_QUEUE_MIN = 36  # top up when central queue drops below this


def request_stop(signum=None, frame=None):
    STOP_REQUESTED.set()


def stdin_listener():
    for line in sys.stdin:
        cmd = line.strip().lower()
        if not cmd:
            continue
        parts = cmd.split()
        if cmd == 'stop':
            print("[cmd] stopping -- saving data and exiting")
            request_stop()
        elif cmd == 'stop now':
            print("[cmd] hard stop")
            os._exit(1)
        elif cmd == 'stop after this round':
            print("[cmd] will stop after this round completes")
            STOP_AFTER_ROUND_REQUESTED.set()
        elif cmd == 'next round':
            print("[cmd] next round requested")
            NEXT_ROUND_REQUESTED.set()
        elif cmd == 'reload':
            print("[cmd] reload queued -- applies at next round start")
            RELOAD_REQUESTED.set()
        elif parts[0] == 'pause':
            dur = int(parts[1]) if len(parts) > 1 else None
            print(f"[cmd] pausing workers{f' for {dur}s' if dur else ''}")
            PAUSE_REQUESTED.set()
            if dur:
                def auto_unpause(d=dur):
                    time.sleep(d)
                    print("[cmd] auto-unpause")
                    UNPAUSE_REQUESTED.set()
                threading.Thread(target=auto_unpause, daemon=True).start()
        elif cmd == 'unpause':
            print("[cmd] unpausing workers")
            UNPAUSE_REQUESTED.set()
        elif parts[0] == 'save' and ' '.join(parts[1:3]) == 'training data':
            global SAVE_TRAINING_DATA_N
            SAVE_TRAINING_DATA_N = int(parts[3]) if len(parts) > 3 else None
            SAVE_TRAINING_DATA_REQUESTED.set()
            print(f"[cmd] save training data requested{f' (n={SAVE_TRAINING_DATA_N})' if SAVE_TRAINING_DATA_N else ''}")
        else:
            print(f"[cmd] unknown command: {cmd!r}")


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


def top_up_queues(game_queue, sf_queue, game_gen, budget=None, target=None):
    added = 0
    if target is None:
        target = GAME_QUEUE_MIN * 2

    sf_target = target // 2

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

    threading.Thread(target=stdin_listener, daemon=True).start()

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
            cmd_paused = False
            recorder = RecordKeeper(n_retrains, run_num=run_num, every_sec=45.0)
            recorder.training_queue = len(rescorer.training_data)

            if STOP_REQUESTED.is_set() or STOP_AFTER_ROUND_REQUESTED.is_set():
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

            if RELOAD_REQUESTED.is_set():
                RELOAD_REQUESTED.clear()
                import importlib
                import chessbot.rescore
                import chessbot.review
                importlib.reload(chessbot.rescore)
                importlib.reload(chessbot.review)
                state = rescorer.export_state()
                rescorer = chessbot.rescore.Rescorer(base_cfg, sf_game_q, sf_res_q, cache)
                rescorer.import_state(state)
                recorder = chessbot.review.RecordKeeper(n_retrains, run_num=run_num, every_sec=45.0)
                recorder.training_queue = len(rescorer.training_data)
                print("[cmd] reloaded rescore/review, rescorer state migrated")

            if run_num % base_cfg.validation_every == 0:
                is_validation = True
                working_cfg = create_validation_config(working_cfg, val_yaml_path)

            if working_cfg.inference_backend == 'ort_trt':
                if is_validation:
                    trt_dir = os.path.join(working_cfg.run_dir, 'val_trt')
                    model_name = f'{working_cfg.run_tag}_val'
                else:
                    trt_dir, model_name, _ = selfplay_trt_paths(working_cfg)
                working_cfg = prepare_trt(working_cfg, trt_dir, model_name)
            
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
                    game_queue, sf_queue, game_gen,
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
            while len(procs) or (recorder.training_queue < needed_to_retrain):
                if STOP_REQUESTED.is_set():
                    procs = check_and_reap_procs(procs, request_stop=True)
                    break

                if NEXT_ROUND_REQUESTED.is_set():
                    NEXT_ROUND_REQUESTED.clear()
                    if not stop_signal_sent:
                        for w in procs:
                            w["msg_q"].put("drain_and_stop")
                        stop_signal_sent = True
                    print("[cmd] draining workers for next round")
                    break

                if PAUSE_REQUESTED.is_set():
                    PAUSE_REQUESTED.clear()
                    if not cmd_paused:
                        for p in procs:
                            p["msg_q"].put("pause")
                        cmd_paused = True
                        print("[cmd] workers paused")

                if cmd_paused and UNPAUSE_REQUESTED.is_set():
                    UNPAUSE_REQUESTED.clear()
                    for p in procs:
                        p["msg_q"].put("unpause")
                    cmd_paused = False
                    print("[cmd] workers unpaused")

                if SAVE_TRAINING_DATA_REQUESTED.is_set():
                    SAVE_TRAINING_DATA_REQUESTED.clear()
                    data = rescorer.training_data
                    n = SAVE_TRAINING_DATA_N
                    if n and n < len(data):
                        import random as _random
                        data = _random.sample(data, n)
                    ts = int(time.time())
                    snap_path = os.path.join(base_cfg.run_dir, f"training_snapshot_{ts}.pkl")
                    with open(snap_path, "wb") as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"[cmd] saved {len(data)} samples to training_snapshot_{ts}.pkl")

                # check for finished procs
                procs = check_and_reap_procs(procs)

                # break if no workers and backlog is small enough to carry into next round
                if not procs and len(finished_games) < MAX_BACKLOG:
                    break

                # refill queues; blunder replays trigger a top-up even if queues aren't low
                if not stop_signal_sent:
                    sf_low = sf_queue is not None and sf_queue.qsize() < GAME_QUEUE_MIN // 2
                    game_low = game_queue.qsize() < GAME_QUEUE_MIN
                    if sf_low or game_low:
                        added = top_up_queues(
                            game_queue, sf_queue, game_gen,
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
                is_throttled = sf_rescore_threads[0].depth < sf_rescore_threads[0].base_depth
                if sf_backlog > 100 and not is_throttled:
                    for t in sf_rescore_threads:
                        t.depth = max(1, t.base_depth - 1)
                    rescorer.current_depth = sf_rescore_threads[0].depth
                    print(f"[rescore] backlog {sf_backlog}, depth -> {rescorer.current_depth}")
                elif sf_backlog < 10 and is_throttled:
                    for t in sf_rescore_threads:
                        t.depth = t.base_depth
                    rescorer.current_depth = sf_rescore_threads[0].depth
                    print(f"[rescore] backlog cleared, depth -> {rescorer.current_depth}")

                recorder.training_queue = rescorer.training_data_size

                # check for a retrain
                if recorder.training_queue >= needed_to_retrain:
                    # drain so write gets accurate data; workers keep playing meanwhile
                    rescorer.write_training_data_pkl(
                        size=working_cfg.retrain_size, randomize=True)
                    recorder.training_queue = rescorer.training_data_size

                    # worker 0 predicts training batch then pauses; rest just pause
                    pred_pkl_path = os.path.join(
                        working_cfg.pending_training_dir, "predictions_latest.pkl")
                    if os.path.exists(pred_pkl_path):
                        os.remove(pred_pkl_path)
                    if procs:
                        procs[0]["msg_q"].put({
                            "cmd": "predict_then_pause",
                            "training_dir": working_cfg.pending_training_dir,
                            "pred_pkl_path": pred_pkl_path,
                        })
                    for p in procs[1:]:
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
                                # preserve throttle state; cap to new base if config lowered depth
                                current_depth = t.depth
                                t.update_config(working_cfg)
                                t.depth = min(current_depth, t.base_depth)
                            rescorer.current_depth = sf_rescore_threads[0].depth
                            rescorer.aggregate_metrics(
                                n_retrains,
                                working_cfg.progress_csv_path,
                                pred_pkl_path)
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

                    if working_cfg.inference_backend == 'ort_trt' and not is_validation:
                        trt_dir, model_name, _ = selfplay_trt_paths(working_cfg)
                        working_cfg = prepare_trt(working_cfg, trt_dir, model_name)

                    # unpause workers
                    for p in procs:
                        p["msg_q"].put("unpause")

                else:
                    time.sleep(0.5)
            
            # when done, close the queues
            procs = shutdown_round(procs, recent_q, telemetry_q)
            if procs:
                print(f"[warn] {len(procs)} workers still alive after shutdown")

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

            # end of round: drain the largest multiple of retrain_size sitting
            # in the buffer (randomized), rather than carrying it all forward
            # into the next round's normal single-batch retrains. On the last
            # round there's no next round to carry stragglers into, so keep
            # looping until the SF-rescore pipeline (intake/pending) is fully
            # drained instead of returning after a single pass.
            is_last_round = run_num >= base_cfg.n_rounds
            while True:
                recorder.training_queue = rescorer.training_data_size
                k = recorder.training_queue // working_cfg.retrain_size
                if k > 0:
                    drain_size = k * working_cfg.retrain_size
                    print(
                        f"[main] end of round: draining {drain_size} of "
                        f"{recorder.training_queue} queued samples"
                    )
                    rescorer.write_training_data_pkl(size=drain_size, randomize=True)
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
                            for t in sf_rescore_threads:
                                # preserve throttle state; cap to new base if config lowered depth
                                current_depth = t.depth
                                t.update_config(working_cfg)
                                t.depth = min(current_depth, t.base_depth)
                            rescorer.current_depth = sf_rescore_threads[0].depth
                            eor_pred_pkl = os.path.join(
                                working_cfg.pending_training_dir, "predictions_latest.pkl")
                            rescorer.aggregate_metrics(
                                n_retrains,
                                working_cfg.progress_csv_path,
                                eor_pred_pkl)
                            n_retrains += 1

                        # workers for this round are already stopped; just keep
                        # ticking the rescorer so SF-pending games can finish
                        while finished_games:
                            to_process = finished_games.popleft()
                            rescorer.submit(pull_pkl(to_process))
                        rescorer.tick()

                        time.sleep(0.05)

                if not is_last_round:
                    break

                # last round: wait for any still-in-flight SF rescoring to
                # land, then loop back and check for another full multiple.
                while finished_games:
                    to_process = finished_games.popleft()
                    rescorer.submit(pull_pkl(to_process))
                rescorer.tick()

                pending_left = finished_games or rescorer.intake or rescorer.pending
                if not pending_left or STOP_REQUESTED.is_set():
                    break

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
    