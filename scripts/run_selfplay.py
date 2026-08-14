# run_selfplay.py
import os
import time
import multiprocessing as mp
import sys
import json
import queue
import random

import numpy as np
import pandas as pd

from chessbot import SP_DIR
from chessbot.looper import init_selfplay
from chessbot.rescore import Rescorer, SFRescoreThread, migrate_pretrain_progress
from chessbot.review import RecordKeeper
from chessbot.config import Config
from chessbot.utils import make_jsonable, format_time, next_model_epoch
from chessbot.validation import (
    build_validation_summary, create_validation_config, create_probe_config,
    probe_out_path, run_probe, analyse_probe_file, last_probe_epoch,
)
from chessbot.infer_ort_trt import prepare_trt, selfplay_trt_paths
from chessbot.game_utils import GameGenerator, GameSpec, resolve_cfg
from chessbot.retrain_worker import run_retrain_worker
import chessbot.replay_buffer as rb

import pickle
import signal
import threading

STOP_REQUESTED = threading.Event()
STOP_AFTER_ROUND_REQUESTED = threading.Event()
NEXT_ROUND_REQUESTED = threading.Event()
NO_NEW_GAMES_REQUESTED = threading.Event()
ROUND_LIVE = threading.Event()
PAUSE_REQUESTED = threading.Event()
UNPAUSE_REQUESTED = threading.Event()
SAVE_TRAINING_DATA_REQUESTED = threading.Event()
SAVE_TRAINING_DATA_N = None
WORKER_PROCS = []

MAX_BACKLOG = 250
GAME_QUEUE_MIN = 36  # top up when central queue drops below this

# blunder-replay probe: one extra worker every Nth retrain, replaying a random
# sample of scanned blunder positions. Seeded on the epoch, so each probe draws
# a fresh sample and any one of them can be reproduced from its history row.
PROBE_EVERY_RETRAINS = 5
PROBE_N_POSITIONS = 2048


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
            for w in WORKER_PROCS:
                try:
                    w.kill()
                except Exception:
                    pass
            os._exit(1)
        elif cmd == 'stop after this round':
            print("[cmd] will stop after this round completes")
            STOP_AFTER_ROUND_REQUESTED.set()
        elif cmd == 'next round':
            if not ROUND_LIVE.is_set():
                print("[cmd] next round: no round running, ignored")
            else:
                print("[cmd] next round requested")
                NEXT_ROUND_REQUESTED.set()
        elif cmd == 'no new games':
            if not ROUND_LIVE.is_set():
                print("[cmd] no new games: no round running, ignored")
            else:
                print("[cmd] no new games -- draining current games, "
                      "round ends normally")
                NO_NEW_GAMES_REQUESTED.set()
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
        WORKER_PROCS.append(p)

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
    with init_selfplay(
        cfg, recent_games_q, telemetry_q, msg_q,
        game_queue=game_queue, sf_queue=sf_queue,
    ) as looper:
        looper.run(stop_ev)


def child_probe_looper(cfg, stop_ev, msg_q, n_positions, seed, out_path):
    """Blunder-replay probe worker: builds its own games from the scanned
    position pool, never touches the shared game queue."""
    run_probe(cfg, n_positions, seed, out_path, stop_ev=stop_ev, msg_q=msg_q)


def spawn_probe_worker(cfg, n_retrains, n_positions=PROBE_N_POSITIONS):
    ctx = mp.get_context()
    c = create_probe_config(cfg)
    c.id = "probe"
    out_path = probe_out_path(c, n_retrains)

    stop_ev = ctx.Event()
    msg_q = ctx.Queue()
    p = ctx.Process(
        target=child_probe_looper,
        args=(c, stop_ev, msg_q, n_positions, n_retrains, out_path),
        daemon=True,
    )
    p.start()
    WORKER_PROCS.append(p)

    print(f"[probe] worker up at epoch {n_retrains}: {n_positions} positions "
          f"-> {os.path.basename(out_path)}")

    return {
        "id": c.id,
        "p": p,
        "stop_ev": stop_ev,
        "msg_q": msg_q,
        "out_path": out_path,
        "n_retrains": n_retrains,
        "stop_sent_at": None,
        "term_sent_at": None,
        "kill_sent_at": None,
    }


def collect_probe_worker(probe_worker, procs):
    """
    Once the probe worker is gone from procs the reaper has seen it exit, so
    its file is final. Hand it to a short-lived process for the deep SF pass;
    that is minutes of pure Stockfish time and must not block the main loop.

    Returns the probe_worker to keep holding (None once collected).
    """
    if probe_worker is None:
        return None

    if any(w is probe_worker for w in procs):
        return probe_worker

    out_path = probe_worker["out_path"]
    if not os.path.exists(out_path):
        print("[probe] worker exited without writing a file")
        return None

    p = mp.get_context().Process(
        target=analyse_probe_file,
        args=(out_path,),
        kwargs={"n_retrains": probe_worker["n_retrains"]},
        daemon=True,
    )
    p.start()
    print(f"[probe] deep SF analysis launched on {os.path.basename(out_path)}")
    return None


class ValidationSpecSource:
    """Wraps a pre-generated list of validation GameSpecs so top_up_queues
    can pull from it the same way it pulls from GameGenerator.next_game()."""

    def __init__(self, specs):
        self.specs = specs
        self.idx = 0

    def next_game(self):
        if self.idx >= len(self.specs):
            return None
        spec = self.specs[self.idx]
        self.idx += 1
        return spec


def top_up_queues(game_queue, sf_queue, game_gen, budget=None, target=None):
    added = 0
    if target is None:
        target = GAME_QUEUE_MIN * 2

    sf_target = target // 2

    while budget is None or added < budget:
        game_ok = game_queue.qsize() >= target
        sf_ok = sf_queue is not None and sf_queue.qsize() >= sf_target
        if game_ok or sf_ok:
            break
        spec = game_gen.next_game()
        if spec is None:
            break
        if spec.meta.get("vs_stockfish") and sf_queue is not None:
            sf_queue.put(spec)
        else:
            game_queue.put(spec)
        added += 1

    return added


def close_proc(p):
    p.close()
    try:
        WORKER_PROCS.remove(p)
    except ValueError:
        pass


def log_worker_death(w, crash_log_path=None):
    """Unexpected exit (we never asked this worker to stop): log id, exit
    code, and a timestamp. On Windows a segfault/access-violation shows up
    as a large negative exitcode (e.g. -1073741819 == 0xC0000005) rather
    than a Python traceback, since a native crash never reaches the
    interpreter -- this is the only signal available for that case."""
    p = w["p"]
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] worker {w['id']} died unexpectedly, exitcode={p.exitcode}"
    print(f"[worker-death] {line}")
    if crash_log_path:
        try:
            with open(crash_log_path, "a") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[worker-death] failed to write crash log: {e}")


def check_and_reap_procs(procs, request_stop=False, grace_s=5.0, term_s=2.0,
                          crash_log_path=None):
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
            # exitcode 0 means the worker's main returned normally, which only
            # happens on a graceful path -- drain_and_stop at end of round is
            # the common one, and it does not set stop_sent_at. A real crash
            # exits non-zero (or negative for a signal).
            if w["stop_sent_at"] is None and p.exitcode != 0:
                log_worker_death(w, crash_log_path)
            close_proc(p)
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
        except queue.Empty:
            break
    return out


def shutdown_round(procs, recent_q, telemetry_q, max_wait_s=15.0):
    # max_wait_s=None waits indefinitely for workers to finish their
    # in-flight games naturally, with no forced kill.
    deadline = None if max_wait_s is None else time.monotonic() + max_wait_s

    # Ask nicely first and start escalation timers
    procs = check_and_reap_procs(procs, request_stop=True)

    while procs and (deadline is None or time.monotonic() < deadline):
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
                close_proc(p)

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


def pull_pkl(to_process):
    # might be nested or flat depending on where it came from
    if 'meta' in to_process.keys():
        return to_process['meta']['pkl_file']
    else:
        return to_process['pkl_file']


def main(run_tag):
    global Config
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

    rescorer = Rescorer(base_cfg, sf_game_q, sf_res_q)
    finished_games = rescorer.get_unprocessed()

    # load any previously saved untrained samples
    remaining_pkl = rb.find_remaining_untrained(base_cfg.run_dir)
    if remaining_pkl:
        rescorer.live_buffer.load_records(
            rb.read_remaining_untrained(remaining_pkl))
        os.remove(remaining_pkl)
        n_loaded = len(rescorer.live_buffer)
        print(f"[main] loaded {n_loaded} samples from "
              f"{os.path.basename(remaining_pkl)}")
        rescorer.live_buffer.flush_all_ready()

    rb.seed_replay_buffer(base_cfg.historic_dir, base_cfg.replay_buffer_dir)

    # split pretraining's rows out of eval_progress.csv before the counter is
    # read -- no-op after the first startup of a run
    migrate_pretrain_progress(base_cfg.run_dir)

    n_retrains = next_model_epoch(base_cfg.progress_csv_path)

    recorder = RecordKeeper(n_retrains, run_num=0, every_sec=45.0)

    threading.Thread(target=stdin_listener, daemon=True).start()

    start = time.time()
    procs = []
    recent_q = None
    telemetry_q = None
    game_queue = None
    sf_queue = None
    game_gen = None
    retrain_worker = None
    probe_worker = None
    # with nothing on record, probe on the first round; otherwise pick the
    # cadence back up where the previous process left it
    last_probe = last_probe_epoch(
        base_cfg.run_dir,
        selfplay_dir=base_cfg.selfplay_dir,
        previous_run_tag=base_cfg.previous_run_tag,
    )
    if last_probe is None:
        last_probe = -PROBE_EVERY_RETRAINS
        print("[probe] no probe history -- probing on the first round")
    else:
        print(f"[probe] last probe at epoch {last_probe}, next at "
              f"{last_probe + PROBE_EVERY_RETRAINS}")
    total_games = 0

    PTS = rb.PRIMARY_TRIGGER_SHARDS
    try:    
        for selfplay_round in range(base_cfg.n_rounds):
            run_num = 1 + selfplay_round
            cmd_paused = False
            recorder = RecordKeeper(n_retrains, run_num=run_num, every_sec=45.0)
            recorder.training_queue = len(rescorer.live_buffer)
            recorder.primary_buffer_dir = base_cfg.primary_buffer_dir
            recorder.primary_buffer_trigger = PTS

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

            # config is re-read from yaml each round, so edits to the run's
            # config.yaml take effect at the next round boundary
            working_cfg = Config.from_yaml(yaml_path, init=True)
            is_validation = False

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

            n_workers = max(1, working_cfg.n_workers)
            initial_target = n_workers * working_cfg.games_at_once + GAME_QUEUE_MIN

            game_queue = ctx.Queue()
            n_games = working_cfg.n_games

            if is_validation:
                sf_queue = None
                spec_source = ValidationSpecSource(game_gen.validation_games())
            else:
                sf_queue = ctx.Queue()
                spec_source = game_gen

            total_queued = top_up_queues(
                game_queue, sf_queue, spec_source,
                budget=n_games, target=initial_target,
            )

            procs = spawn_workers(
                working_cfg, recent_q, telemetry_q, game_queue, sf_queue
            )

            # Validation and normal rounds now share identical queue fill/
            # top-up logic via top_up_queues + spec_source. Validation used
            # to dump its whole remaining spec backlog into game_queue in one
            # shot on the first top-up, which pushed total_queued to n_games
            # before most of those games were ever pulled, triggering
            # drain_and_stop early and stranding queued-but-unplayed games.
            stop_signal_sent = False

            # 'no new games' / 'next round' only mean anything while a round's
            # worker loop is running. Typed during the gap between rounds
            # (drain, rescore tick, TRT rebuild) the event stays set and the
            # next round consumes it on its first pass -- draining the queues
            # and freezing top-up right after the initial fill.
            NO_NEW_GAMES_REQUESTED.clear()
            NEXT_ROUND_REQUESTED.clear()
            ROUND_LIVE.set()

            n_retrains = next_model_epoch(working_cfg.progress_csv_path)

            procs = check_and_reap_procs(
                procs,
                crash_log_path=os.path.join(working_cfg.run_dir, "worker_crash_log.txt"),
            )
            waiting_on_retrain = False
            # keep spinning while a retrain is in flight even after every
            # worker has exited -- see the break below
            while len(procs) or retrain_worker is not None:
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

                if NO_NEW_GAMES_REQUESTED.is_set():
                    NO_NEW_GAMES_REQUESTED.clear()
                    if not stop_signal_sent:
                        for w in procs:
                            w["msg_q"].put("drain_and_stop")
                        stop_signal_sent = True
                        dropped = len(drain_queue(game_queue))
                        if sf_queue is not None:
                            dropped += len(drain_queue(sf_queue))
                        print(f"[cmd] no new games -- workers draining, "
                              f"dropped {dropped} queued-but-unplayed games, "
                              f"queues frozen for rest of round")
                    else:
                        print("[cmd] no new games: queues already frozen, no-op")

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
                    data = rescorer.live_buffer.records
                    n = SAVE_TRAINING_DATA_N
                    if n and n < len(data):
                        data = random.sample(data, n)
                    ts = int(time.time())
                    snap_path = os.path.join(base_cfg.run_dir, f"training_snapshot_{ts}.pkl")
                    with open(snap_path, "wb") as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"[cmd] saved {len(data)} samples to training_snapshot_{ts}.pkl")

                # check for finished procs
                procs = check_and_reap_procs(
                    procs,
                    crash_log_path=os.path.join(working_cfg.run_dir, "worker_crash_log.txt"),
                )
                probe_worker = collect_probe_worker(probe_worker, procs)

                # The cadence is measured in retrains, so this is checked every
                # pass rather than at round start: a round spans many retrains,
                # and a round-start check fires the probe late and at round
                # cadence instead. One extra worker on top of the selfplay
                # ones -- it lives in procs like any other, so pause/unpause,
                # drain_and_stop and the reaper all apply to it for free.
                probe_due = (not is_validation and probe_worker is None
                             and not stop_signal_sent
                             and n_retrains >= last_probe + PROBE_EVERY_RETRAINS)
                if probe_due:
                    last_probe = n_retrains
                    probe_worker = spawn_probe_worker(working_cfg, n_retrains)
                    procs.append(probe_worker)

                # break if no workers and backlog is small enough to carry into next round
                if not procs and len(finished_games) < MAX_BACKLOG:
                    if retrain_worker is None:
                        break
                    # Round is done but a retrain is still running. Hold here
                    # instead of tearing down: round N+1 would spawn workers
                    # that build TRT engines and run inference on the same GPU
                    # this retrain is training on, and its retrain_done would
                    # never be consumed -- no config reload, no TRT rebuild,
                    # no primary->replay rotation, no metrics, no epoch bump.
                    if not waiting_on_retrain:
                        print("[round] games finished, holding round open for "
                              "the in-flight retrain before starting the next")
                        waiting_on_retrain = True

                # refill queues; blunder replays trigger a top-up even if queues aren't low
                if not stop_signal_sent:
                    sf_low = sf_queue is not None and sf_queue.qsize() < GAME_QUEUE_MIN // 2
                    game_low = game_queue.qsize() < GAME_QUEUE_MIN
                    if sf_low or game_low:
                        added = top_up_queues(
                            game_queue, sf_queue, spec_source,
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
                # sync every pass, not just inside the throttle arms, so the
                # telemetry budget column reflects the threads' real state
                rescorer.current_movetime_ms = sf_rescore_threads[0].movetime_ms
                is_throttled = (sf_rescore_threads[0].movetime_ms
                                < sf_rescore_threads[0].base_ms)
                if sf_backlog > 100 and not is_throttled:
                    for t in sf_rescore_threads:
                        t.movetime_ms = t.throttled_ms
                    rescorer.current_movetime_ms = sf_rescore_threads[0].movetime_ms
                    print(f"[rescore] backlog {sf_backlog}, movetime -> "
                          f"{rescorer.current_movetime_ms}ms")
                elif sf_backlog < 10 and is_throttled:
                    for t in sf_rescore_threads:
                        t.movetime_ms = t.base_ms
                    rescorer.current_movetime_ms = sf_rescore_threads[0].movetime_ms
                    print(f"[rescore] backlog cleared, movetime -> "
                          f"{rescorer.current_movetime_ms}ms")

                recorder.training_queue = rescorer.training_data_size

                # spawn the retrain worker once primary_buffer is actually full.
                # It used to launch a shard early (31/32) to overlap replay +
                # historic loading with the last fill, but the main loop just sat
                # waiting for 32/32 anyway, so there was nothing to overlap with.
                primary_files = rb.list_shard_files(working_cfg.primary_buffer_dir)
                n_files = len(primary_files)
                # `and procs`: once the workers are gone the round is closing,
                # so don't start a fresh retrain that would hold it open again
                launch_retrain = (retrain_worker is None and not is_validation
                                  and bool(procs))
                if launch_retrain and n_files >= PTS:
                    replay_files = rb.sample_files(
                        working_cfg.replay_buffer_dir, rb.RETRAIN_REPLAY_SHARDS)

                    historic_files = rb.sample_files(
                        working_cfg.historic_dir, rb.RETRAIN_HISTORIC_SHARDS)

                    primary_sample = random.sample(
                        primary_files, rb.RETRAIN_PRIMARY_SHARDS)

                    retrain_msg_q = ctx.Queue()
                    retrain_result_q = ctx.Queue()
                    retrain_p = ctx.Process(
                        target=run_retrain_worker,
                        args=(working_cfg.run_dir, retrain_msg_q, retrain_result_q, n_retrains),
                        daemon=True,
                    )
                    retrain_p.start()
                    # both messages up front; the worker still consumes them in
                    # order and does the same preload -> validate -> train run
                    retrain_msg_q.put({
                        "cmd": "preload", "replay": replay_files, "historic": historic_files,
                    })
                    retrain_msg_q.put({"cmd": "start", "primary": primary_sample})
                    retrain_worker = {
                        "p": retrain_p, "msg_q": retrain_msg_q, "result_q": retrain_result_q,
                        "replay_files": replay_files, "primary_files": primary_sample,
                    }
                    print(f"[retrain] worker launched at {n_files}/{PTS} "
                          f"-- loading {rb.RETRAIN_REPLAY_SHARDS} replay + "
                          f"{rb.RETRAIN_HISTORIC_SHARDS} historic shards, "
                          f"then validating + building retrain set")
                    rescorer.reset_writer()

                # check on the retrain worker once per pass -- no nested loop,
                # we just come back around every outer-loop iteration anyway
                if retrain_worker is not None:
                    try:
                        result = retrain_worker["result_q"].get_nowait()
                    except queue.Empty:
                        result = None

                    if result is not None and result["cmd"] == "retrain_ready":
                        # validation is already done and predictions_latest.pkl
                        # is on disk (self-contained samples+preds) by the time
                        # this arrives -- just pause here. Metrics run after
                        # retrain_done, once workers are back up.
                        for p in procs:
                            p["msg_q"].put("pause")
                        print("[retrain] validation done, workers paused, training now")

                    elif result is not None and result["cmd"] == "retrain_done":
                        print(f"[retrain] worker done ok={result.get('ok')}")
                        retrain_worker["p"].join(timeout=15.0)
                        if retrain_worker["p"].is_alive():
                            retrain_worker["p"].terminate()
                            print("[retrain] worker process did not exit, terminated")
                        else:
                            print("[retrain] worker process exited")

                        recorder.n_retrains += 1
                        working_cfg = Config.from_yaml(yaml_path, init=True)
                        if is_validation:
                            working_cfg = create_validation_config(working_cfg, val_yaml_path)
                        rescorer.config = working_cfg
                        for t in sf_rescore_threads:
                            # budget is hardcoded, so throttle state survives
                            t.update_config(working_cfg)
                        rescorer.current_movetime_ms = sf_rescore_threads[0].movetime_ms

                        if result.get("ok") and not is_validation:
                            if working_cfg.inference_backend == 'ort_trt':
                                # workers only reload their model/engine after
                                # unpause, so the TRT engine must be rebuilt for
                                # the new weights first -- otherwise workers would
                                # either grab a stale engine or each redundantly
                                # recompile their own.
                                trt_dir, model_name, _ = selfplay_trt_paths(working_cfg)
                                working_cfg = prepare_trt(working_cfg, trt_dir, model_name)

                        # hard rule: unpause is the first thing that happens
                        # once retrain has actually finished (plus the TRT
                        # rebuild above, an unavoidable prerequisite for it).
                        for p in procs:
                            p["msg_q"].put("unpause")
                        print("[retrain] workers unpaused")

                        if result.get("ok"):
                            rb.move_files(
                                retrain_worker["primary_files"],
                                working_cfg.replay_buffer_dir
                            )

                            rb.discard_files(retrain_worker["replay_files"])
                            print("[retrain] primary -> replay rotated, cycle complete")
                        else:
                            print(f"[retrain] worker failed: {result.get('error')}")

                        pred_pkl_path = os.path.join(
                            working_cfg.run_dir, "predictions_latest.pkl"
                        )

                        rescorer.aggregate_metrics(
                            n_retrains, working_cfg.progress_csv_path, pred_pkl_path,
                            train_stats=result.get("train_stats"))

                        # aggregation is the only consumer and it has the
                        # numbers now, so ~880 MB of samples+preds can go. It
                        # is rewritten from scratch by the next retrain, and a
                        # raise above leaves it in place for inspection.
                        if os.path.exists(pred_pkl_path):
                            os.remove(pred_pkl_path)
                            print("[retrain] removed predictions_latest.pkl")

                        n_retrains += 1
                        retrain_worker = None

                time.sleep(0.05)

            ROUND_LIVE.clear()

            # when done, close the queues. Workers already got drain_and_stop,
            # so they'll exit on their own once in-flight games finish; only
            # force a hard kill deadline on the truly last round, so the
            # process is guaranteed to terminate.
            is_final_round = run_num >= base_cfg.n_rounds
            shutdown_wait_s = 15.0 if is_final_round else None
            procs = shutdown_round(
                procs, recent_q, telemetry_q,
                max_wait_s=shutdown_wait_s
            )

            if procs:
                print(f"[warn] {len(procs)} workers still alive after shutdown")

            probe_worker = collect_probe_worker(probe_worker, procs)

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

            if is_validation:
                recorder.config = working_cfg
                build_validation_summary(recorder)
                # we do not train after validation currently
                continue

            # No round-end drain/force-retrain needed: primary_buffer/,
            # replay_buffer/, and historic/ are persistent directories, not
            # ephemeral per-round state, so any leftover shards just carry
            # over to the next round/run untouched. On the truly last round,
            # just wait for any still-in-flight SF rescoring to land so
            # nothing gets dropped.
            while is_final_round:
                if STOP_REQUESTED.is_set():
                    break
                while finished_games:
                    to_process = finished_games.popleft()
                    rescorer.submit(pull_pkl(to_process))
                rescorer.tick()

                pending_left = finished_games or rescorer.intake or rescorer.pending
                if not pending_left:
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
        if retrain_worker is not None and retrain_worker["p"].is_alive():
            retrain_worker["p"].terminate()
        rescorer.tick()
        rescorer.push_analyzed(report=True)
        for t in sf_rescore_threads:
            t.close()
        if rescorer.live_buffer.records:
            remaining_pkl = rb.write_remaining_untrained(
                base_cfg.run_dir, rescorer.live_buffer.records
            )
            n_saved = len(rescorer.live_buffer.records)
            print(f"[main] saved {n_saved} samples to "
                  f"{os.path.basename(remaining_pkl)}")
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
    