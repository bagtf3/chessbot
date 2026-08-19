"""
Roundless selfplay orchestrator.

Replaces run_selfplay.main()'s round loop with one continuous lifetime.
Workers spawn once and never come down for a boundary; validation is a batch
of specs dropped into the same queue everything else uses, not a mode the
whole fleet switches into.

Lives here rather than in scripts/ so it can be imported and exercised --
run_selfplay.py cannot be imported at all -- and so the mp children import a
real module instead of re-importing the entry script under spawn.

Nothing here is wired up yet: run_selfplay.main() is untouched and still owns
the live path until cutover.
"""
import multiprocessing as mp
import os
import pickle
import queue
import random
import signal
import sys
import threading
import json
import time
from collections import deque

from chessbot import SP_DIR
from chessbot.blunder_replay import (
    BRP_HISTORY_FILENAME, load_probe_pool,
    last_epoch_in_blunder_replay_history,
)
from chessbot.config import Config
from chessbot.infer_ort_trt import prepare_trt, selfplay_trt_paths
from chessbot.game_utils import GameGenerator
from chessbot.looper import init_selfplay
from chessbot.rescore import (
    BRP_ANALYSIS_EVERY, Rescorer, SFRescoreThread, migrate_pretrain_progress,
)
from chessbot.retrain_worker import run_retrain_worker
from chessbot.review import RecordKeeper
from chessbot.utils import format_time, make_jsonable, next_model_epoch
from chessbot.validation import (
    build_validation_summary_from_rows, create_validation_config,
)
import chessbot.replay_buffer as rb

GAME_QUEUE_MIN = 36
# validation batches ride the same queue as everything else, which is safe
# only because that queue is kept shallow -- raise the target much above this
# and batches start waiting behind selfplay, and the ladder cadence drifts
GAME_QUEUE_TARGET = GAME_QUEUE_MIN * 2
# a batch stuck this long is closed out on whatever came back rather than
# left to wedge the cadence forever. Generous on purpose: it should only ever
# fire on a genuinely stuck batch, not a slow one
VAL_BATCH_TIMEOUT_S = 7200.0


def child_looper(cfg, stop_ev, recent_games_q, telemetry_q, msg_q, game_queue):
    """mp.Process target. One queue now -- selfplay, validation and blunder
    replays all arrive on it and are told apart by spec.meta."""
    with init_selfplay(
        cfg, recent_games_q, telemetry_q, 
        msg_q, game_queue=game_queue
    ) as looper:
        looper.run(stop_ev)


def drain_queue(q):
    out = []
    if q is None:
        return out
    while True:
        try:
            out.append(q.get_nowait())
        except Exception:
            return out


def pull_pkl(to_process):
    """Metas arrive nested from recent_q and flat from the unprocessed
    backlog the rescorer recovers off disk."""
    if 'meta' in to_process:
        return to_process['meta']['pkl_file']
    return to_process['pkl_file']


def close_proc(p, registry):
    p.close()
    try:
        registry.remove(p)
    except ValueError:
        pass


def log_worker_death(w, crash_log_path=None):
    """Unexpected exit. On Windows a native crash surfaces only as a large
    negative exitcode (e.g. -1073741819 == 0xC0000005), never a traceback,
    so this is the sole signal for that case."""
    p = w["p"]
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] worker {w['id']} died unexpectedly, exitcode={p.exitcode}"
    print(f"[worker-death] {line}", flush=True)
    if crash_log_path:
        with open(crash_log_path, "a") as f:
            f.write(line + '\n')


def check_and_reap_procs(procs, registry, request_stop=False, grace_s=5.0,
                         term_s=2.0, crash_log_path=None):
    """Join exited procs; escalate stop -> terminate -> kill for the rest."""
    now = time.monotonic()
    kept = []
    for w in procs:
        p = w["p"]
        if not p.is_alive():
            p.join(timeout=0)
            # exitcode 0 only happens on a graceful path (drain_and_stop),
            # which never sets stop_sent_at
            if w["stop_sent_at"] is None and p.exitcode != 0:
                log_worker_death(w, crash_log_path)
            close_proc(p, registry)
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


def stdin_listener(controls, worker_procs):
    """Operator console. Takes the Controls it drives rather than reaching
    for module globals, so there is no ambiguity about which set is live
    while the old script still exists alongside.

    'next round' and 'stop after this round' are gone -- there are no rounds.
    """
    for line in sys.stdin:
        cmd = line.strip().lower()
        if not cmd:
            continue
        parts = cmd.split()

        if cmd == 'stop':
            print("[cmd] stopping -- saving data and exiting", flush=True)
            controls.stop.set()

        elif cmd == 'stop now':
            print("[cmd] hard stop", flush=True)
            for w in worker_procs:
                try:
                    w.kill()
                except Exception:
                    pass
            os._exit(1)

        elif cmd == 'no new games':
            print("[cmd] no new games -- draining what is in flight",
                  flush=True)
            controls.no_new_games.set()

        elif parts[0] == 'pause':
            dur = int(parts[1]) if len(parts) > 1 else None
            suffix = f" for {dur}s" if dur else ""
            print(f"[cmd] pausing workers{suffix}", flush=True)
            controls.pause.set()
            if dur:
                def auto_unpause(d=dur):
                    time.sleep(d)
                    print("[cmd] auto-unpause", flush=True)
                    controls.unpause.set()
                threading.Thread(target=auto_unpause, daemon=True).start()

        elif cmd == 'unpause':
            print("[cmd] unpausing workers", flush=True)
            controls.unpause.set()

        elif parts[0] == 'save' and ' '.join(parts[1:3]) == 'training data':
            n = int(parts[3]) if len(parts) > 3 else None
            controls.save_training_data_n = n
            controls.save_training_data.set()
            print(f"[cmd] save training data requested"
                  f"{f' (n={n})' if n else ''}", flush=True)

        elif cmd == 'status':
            controls.status.set()

        else:
            print(f"[cmd] unknown command: {cmd!r}", flush=True)


class Controls:
    """Operator flags, set by the stdin listener and read by the runner.

    An object rather than module globals so the listener and the runner
    provably share one set -- during the side-by-side dev period the old
    script still owns its own copies.
    """

    def __init__(self):
        self.stop = threading.Event()
        self.pause = threading.Event()
        self.unpause = threading.Event()
        self.no_new_games = threading.Event()
        self.save_training_data = threading.Event()
        self.save_training_data_n = None
        self.status = threading.Event()


class SelfPlayRunner:
    """
    Owns every piece of long-lived state that used to be a main() local.
    Phase methods take no arguments and read self, which is the point: the
    old free functions took up to 8 parameters and the bugs came from
    passing the wrong round's value.
    """

    def __init__(self, run_tag, base_cfg, yaml_path, val_yaml_path, controls=None):
        self.run_tag = run_tag
        self.base_cfg = base_cfg
        self.yaml_path = yaml_path
        self.val_yaml_path = val_yaml_path
        self.controls = controls or Controls()

        self.cfg = None            # live working config, reloaded at retrain
        self.val_cfg = None        # ladder config; owns sf_depth

        self.rescorer = None
        self.recorder = None
        self.game_gen = None
        self.blunder_pool = None
        self.sf_rescore_threads = []

        self.ctx = mp.get_context()
        self.game_q = None
        self.recent_q = None
        self.telemetry_q = None
        self.procs = []
        self.worker_procs = []

        self.current_epoch = 0
        self.games_at_start = 0
        self.total_games = 0
        self.total_queued = 0
        self.finished_games = deque()
        self.finished_brp = deque()

        # validation batch state. val_pending counts paired_validation metas
        # still owed; the deadline is the backstop for one that never lands.
        self.val_pending = 0
        self.val_rows = []
        self.val_batch_started_at = None
        self.val_batch_epoch = None
        self.last_validated_epoch = None

        self.retrain_worker = None
        self.cmd_paused = False
        self.stop_signal_sent = False
        self.started_at = None

    # ---- setup -------------------------------------------------------

    def setup(self):
        """Everything main() did once before its round loop."""
        cfg = self.base_cfg

        sf_game_q = queue.Queue()
        sf_res_q = queue.Queue()
        self.sf_rescore_threads = [
            SFRescoreThread(sf_game_q, sf_res_q, cfg)
            for _ in range(max(1, cfg.rescore_n_sf_workers))
        ]

        for t in self.sf_rescore_threads:
            t.start()

        # one live in-RAM copy; the Rescorer mutates it and the replay
        # sampler reads it. Never crosses the process boundary. None when no
        # pool is configured, which every replay path already handles.
        self.blunder_pool = load_probe_pool()

        self.rescorer = Rescorer(cfg, sf_game_q, sf_res_q)
        finished, finished_brp = self.rescorer.get_unprocessed()
        self.finished_games = finished
        self.finished_brp = finished_brp

        # three unrelated startup chores: adopt the previous process's
        # untrained samples, top the replay buffer up from historic, and split
        # pretraining's rows out of eval_progress.csv before the epoch counter
        # below reads it
        self.load_live_buffer()
        rb.seed_replay_buffer(cfg.historic_dir, cfg.replay_buffer_dir)
        migrate_pretrain_progress(cfg.run_dir)

        self.current_epoch = next_model_epoch(cfg.progress_csv_path)
        self.recorder = RecordKeeper(self.current_epoch, every_sec=45.0)
        self.recorder.training_queue = len(self.rescorer.live_buffer)
        self.recorder.primary_buffer_dir = cfg.primary_buffer_dir
        self.recorder.primary_buffer_trigger = rb.PRIMARY_TRIGGER_SHARDS

        self.reload_config()
        self.val_cfg = create_validation_config(self.cfg, self.val_yaml_path)
        self.crash_log_path = os.path.join(self.cfg.run_dir, "worker_crash_log.txt")

        # credit games already on disk against the budget. total_queued is
        # seeded too: anything queued but unplayed when the last process died
        # went with it, so what is owed is measured from completions.
        self.games_at_start = self.games_already_completed()
        self.total_games = self.games_at_start
        self.total_queued = self.games_at_start
        if self.games_at_start:
            left = max(0, self.cfg.n_games - self.games_at_start)
            print(f"[runner] resuming: {self.games_at_start:,} games already "
                  f"on disk, {left:,} of {self.cfg.n_games:,} left",
                  flush=True)

        self.game_gen = GameGenerator(self.cfg)
        self.started_at = time.time()

    def games_already_completed(self):
        """Games this run has already finished, counted off the on-disk index.

        The budget is a property of the run, not of one process, so a restart
        resumes against it instead of starting the count over. Blunder-replay
        probes never reach the index -- they return before update_game_index
        -- so they correctly do not consume budget.
        """
        path = self.base_cfg.game_index_file
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def load_live_buffer(self):
        """Adopt the untrained samples the previous process left behind."""
        cfg = self.base_cfg
        path = rb.find_live_buffer(cfg.run_dir)
        if not path:
            return

        self.rescorer.live_buffer.load_records(rb.read_live_buffer(path))
        print(f"[runner] loaded {len(self.rescorer.live_buffer)} samples "
              f"from {os.path.basename(path)}", flush=True)
        self.rescorer.live_buffer.flush_all_ready()
        # re-sync to canonical names immediately, so there is never a window
        # with no valid on-disk copy
        rb.sync_live_buffer(cfg.run_dir, self.rescorer.live_buffer.records)
        canonical = os.path.join(cfg.run_dir, rb.LIVE_BUFFER_PKL)
        if os.path.abspath(path) != os.path.abspath(canonical):
            os.remove(path)

    def reload_config(self):
        """Re-read the run yaml and rebuild the TRT engine for it.

        Roundless there is one engine for everything: validation shares the
        selfplay batch shapes exactly, and its search params travel per-game
        on the spec's own cfg, so the old separate val_trt build is gone.
        """
        self.cfg = Config.from_yaml(self.yaml_path, init=True)
        if self.cfg.inference_backend == 'ort_trt':
            trt_dir, model_name, _ = selfplay_trt_paths(self.cfg)
            self.cfg = prepare_trt(self.cfg, trt_dir, model_name)
        if self.rescorer is not None:
            self.rescorer.config = self.cfg
        # the generator caches its config; rounds used to hide this by
        # rebuilding it each round, roundless it must be re-pointed
        if self.game_gen is not None:
            self.game_gen.config = self.cfg

    # ---- phases ------------------------------------------------------

    def spawn_workers(self):
        """One queue to every worker. There is no designated validator:
        whoever has capacity picks up whatever spec is next, and validation
        is told apart by spec.meta, not by which process pulled it."""
        self.recent_q = self.ctx.Queue()
        self.telemetry_q = self.ctx.Queue()
        self.game_q = self.ctx.Queue()

        for i in range(max(1, self.cfg.n_workers)):
            c = self.cfg.copy()
            c.id = f"w{i}"
            stop_ev = self.ctx.Event()
            msg_q = self.ctx.Queue()
            p = self.ctx.Process(
                target=child_looper,
                args=(c, stop_ev, self.recent_q, self.telemetry_q, msg_q,
                      self.game_q),
                daemon=True,
            )
            p.start()
            self.worker_procs.append(p)
            self.procs.append({
                "id": c.id, "p": p, "stop_ev": stop_ev, "msg_q": msg_q,
                "stop_sent_at": None, "term_sent_at": None,
                "kill_sent_at": None,
            })
        print(f"[runner] spawned {len(self.procs)} workers", flush=True)

    def broadcast(self, cmd):
        for w in self.procs:
            w["msg_q"].put(cmd)

    # ---- queue feeding -----------------------------------------------

    def top_up_queue(self):
        """Keep the queue shallow. Depth is what keeps validation batches
        from queueing behind selfplay -- see GAME_QUEUE_TARGET."""
        if self.stop_signal_sent or self.controls.no_new_games.is_set():
            return 0

        budget = self.cfg.n_games - self.total_queued
        added = 0
        while added < budget and self.game_q.qsize() < GAME_QUEUE_TARGET:
            spec = self.game_gen.next_game()
            if spec is None:
                break
            self.game_q.put(spec)
            added += 1

        self.total_queued += added
        # blunder replays ride the same queue and do not count against the
        # games budget -- they are probes, not selfplay
        self.drain_brp_credits()

        if self.total_queued >= self.cfg.n_games:
            print(f"[runner] games budget {self.cfg.n_games} queued, draining",
                  flush=True)
            self.broadcast("drain_and_stop")
            self.stop_signal_sent = True
        return added

    def drain_brp_credits(self):
        """1:1 trickle: one replay per eligible blunder the rescorer found
        since the last pass. Credits accrue while the queue is full."""
        if self.blunder_pool is None:
            return 0
        n = self.rescorer.take_brp_credits()
        if n <= 0:
            return 0
        return self.enqueue_blunder_replays(n, "trickle")

    def enqueue_blunder_replays(self, n, reason):
        if self.blunder_pool is None or n <= 0:
            return 0
        specs = self.game_gen.next_blunder_replays(
            self.blunder_pool, n, self.rescorer.brp_until_analysis(),
            current_epoch=self.current_epoch,
        )
        for spec in specs:
            self.game_q.put(spec)
        if specs:
            print(f"[blunder_replay] queued {len(specs)} ({reason})",
                  flush=True)
        return len(specs)

    # ---- validation ---------------------------------------------------

    def maybe_start_validation(self):
        """Fire on the retrain cadence. A batch is a discrete, non-overlapping
        set: the ladder counts runs, not games, so overlapping windows would
        advance depth roughly twice as fast."""
        if self.val_pending:
            return
        every = self.cfg.validate_every_n_retrains
        if not every or self.current_epoch % every:
            return
        if self.last_validated_epoch == self.current_epoch:
            return

        n_pairs = max(1, self.cfg.validation_games_per_batch // 2)
        specs = self.game_gen.validation_games(
            cfg=self.val_cfg, n_pairs=n_pairs)
        if not specs:
            print("[validation] no specs generated, skipping", flush=True)
            return

        for spec in specs:
            self.game_q.put(spec)

        # validation counts against the games budget the same as selfplay,
        # so the two sides of total_queued/total_games agree
        self.total_queued += len(specs)
        self.val_pending = len(specs)
        self.val_rows = []
        self.val_batch_started_at = time.monotonic()
        # a retrain can land mid-batch, so the stamp is taken here, not at
        # finish, and every row in the batch belongs to this epoch
        self.val_batch_epoch = self.current_epoch
        self.last_validated_epoch = self.current_epoch
        print(f"[validation] batch of {len(specs)} queued at epoch "
              f"{self.current_epoch}, sf_depth {self.val_cfg.sf_depth}",
              flush=True)

    def finish_validation_batch(self):
        """Complete batch only. Writes one history row, then rebuilds the
        validation config so a bump actually reaches the next batch -- the
        bump is recorded to history, never onto the live cfg."""
        rows, self.val_rows = self.val_rows, []
        self.val_pending = 0
        self.val_batch_started_at = None

        build_validation_summary_from_rows(
            rows, self.val_cfg, model_epoch=self.val_batch_epoch)
        # engines are built per-spec-config and cached for the worker's life,
        # so they must be dropped or the next batch replays the old depth
        self.broadcast("tear_down_sf")
        self.val_cfg = create_validation_config(self.cfg, self.val_yaml_path)

    def close_partial_validation_batch(self, why):
        """Summarise whatever came back. The batch is colour-imbalanced, so
        score is biased and consec_over_50 may be reset by a truncated
        result -- accepted rather than losing the reading entirely. Nothing
        came back at all means there is nothing to summarise."""
        print(f"[validation] closing partial batch ({why}), "
              f"{self.val_pending} never returned, "
              f"summarising {len(self.val_rows)}", flush=True)
        if not self.val_rows:
            self.val_pending = 0
            self.val_batch_started_at = None
            self.broadcast("tear_down_sf")
            return
        self.finish_validation_batch()

    def check_validation_timeout(self):
        if not self.val_pending or self.val_batch_started_at is None:
            return
        if time.monotonic() - self.val_batch_started_at > VAL_BATCH_TIMEOUT_S:
            self.close_partial_validation_batch("timeout")

    # ---- results ------------------------------------------------------

    def drain_results(self):
        """Route finished games. Validation metas are counted here -- with
        the min_game_length fix in the looper they can no longer vanish
        through the short-game filter, so a plain counter is honest."""
        for msg in drain_queue(self.telemetry_q):
            self.recorder.ingest_telemetry(msg)

        for game in drain_queue(self.recent_q):
            meta = game["meta"]
            scenario = meta.get("scenario")

            if scenario == "blunder_replay_probe":
                self.finished_brp.append(game)
                continue

            self.recorder.ingest_recents(game)
            self.update_game_index(meta)
            self.finished_games.append(game)
            self.total_games += 1

            # only collect while a batch is actually live, so a straggler
            # arriving after one closed cannot land in the next one's buffer
            if scenario == "paired_validation" and self.val_pending:
                self.val_rows.append(meta)
                self.val_pending -= 1
                if self.val_pending == 0:
                    self.finish_validation_batch()

        self.recorder.maybe_log_results()

    def update_game_index(self, game):
        game["beat_sf"] = False
        if game.get("vs_stockfish"):
            result = game["result"]
            sf_white = game["stockfish_color"]
            game["beat_sf"] = bool(
                (result > 0 and not sf_white) or (result < 0 and sf_white))

        game = make_jsonable(game)
        idx_file = self.base_cfg.game_index_file
        os.makedirs(os.path.dirname(idx_file), exist_ok=True)
        with open(idx_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(game, ensure_ascii=False) + '\n')

    def tick_rescorer(self):
        while self.finished_games:
            # CLAUDE its saying pull_pkl here is not defined
            self.rescorer.submit(pull_pkl(self.finished_games.popleft()))
        while self.finished_brp:
            self.rescorer.submit_blunder_replay(
                self.finished_brp.popleft()["meta"])
        self.rescorer.tick(self.blunder_pool)
        self.throttle_sf()
        self.recorder.training_queue = self.rescorer.training_data_size

    def throttle_sf(self):
        """Back the SF budget off while the rescore backlog is deep."""
        backlog = len(self.rescorer.intake) + len(self.rescorer.pending)
        sf0 = self.sf_rescore_threads[0]
        # sync every pass so telemetry shows the threads' real state
        self.rescorer.current_movetime_ms = sf0.movetime_ms
        throttled = sf0.movetime_ms < sf0.base_ms

        if backlog > 100 and not throttled:
            for t in self.sf_rescore_threads:
                t.movetime_ms = t.throttled_ms
        elif backlog < 10 and throttled:
            for t in self.sf_rescore_threads:
                t.movetime_ms = t.base_ms
        else:
            return

        self.rescorer.current_movetime_ms = sf0.movetime_ms
        print(f"[rescore] backlog {backlog}, movetime -> "
              f"{self.rescorer.current_movetime_ms}ms", flush=True)

    # ---- retrain ------------------------------------------------------

    def check_retrain(self):
        self.maybe_launch_retrain()
        self.poll_retrain()

    def maybe_launch_retrain(self):
        if self.retrain_worker is not None or not self.procs:
            return
        primary = rb.list_shard_files(self.cfg.primary_buffer_dir)
        if len(primary) < rb.PRIMARY_TRIGGER_SHARDS:
            return

        replay_files = rb.sample_files(
            self.cfg.replay_buffer_dir, rb.RETRAIN_REPLAY_SHARDS)
        historic_files = rb.sample_files(
            self.cfg.historic_dir, rb.RETRAIN_HISTORIC_SHARDS)
        primary_sample = random.sample(primary, rb.RETRAIN_PRIMARY_SHARDS)

        msg_q = self.ctx.Queue()
        result_q = self.ctx.Queue()
        p = self.ctx.Process(
            target=run_retrain_worker,
            args=(self.cfg.run_dir, msg_q, result_q, self.current_epoch),
            daemon=True,
        )
        p.start()
        # both up front; the worker consumes them in order
        msg_q.put({"cmd": "preload", "replay": replay_files,
                   "historic": historic_files})
        msg_q.put({"cmd": "start", "primary": primary_sample})
        self.retrain_worker = {
            "p": p, "msg_q": msg_q, "result_q": result_q,
            "replay_files": replay_files, "primary_files": primary_sample,
        }
        print(f"[retrain] launched at {len(primary)}/"
              f"{rb.PRIMARY_TRIGGER_SHARDS}", flush=True)
        self.rescorer.reset_writer()

    def poll_retrain(self):
        if self.retrain_worker is None:
            return
        try:
            result = self.retrain_worker["result_q"].get_nowait()
        except queue.Empty:
            return
        if result is None:
            return

        if result["cmd"] == "retrain_ready":
            # worker-side validation is done and predictions are on disk;
            # pause so the retrain gets the GPU
            self.broadcast("pause")
            self.cmd_paused = True
            print("[retrain] workers paused, training now", flush=True)
            return

        if result["cmd"] != "retrain_done":
            return

        ok = result.get("ok")
        print(f"[retrain] done ok={ok}", flush=True)
        p = self.retrain_worker["p"]
        p.join(timeout=15.0)
        if p.is_alive():
            p.terminate()
            print("[retrain] worker did not exit, terminated", flush=True)

        self.recorder.n_retrains += 1
        self.reload_config()
        for t in self.sf_rescore_threads:
            t.update_config(self.cfg)
        self.rescorer.current_movetime_ms = self.sf_rescore_threads[0].movetime_ms

        # hard rule: unpause as soon as training is done, the TRT rebuild in
        # reload_config being the one unavoidable prerequisite. This overrides
        # an operator pause -- holding it would stall the run forever -- so
        # cmd_paused is cleared to match, rather than left claiming a pause
        # that no longer exists.
        self.broadcast("unpause")
        if self.cmd_paused:
            self.cmd_paused = False
            print("[retrain] operator pause released by retrain completion",
                  flush=True)
        print("[retrain] workers unpaused", flush=True)

        if ok:
            rb.move_files(self.retrain_worker["primary_files"],
                          self.cfg.replay_buffer_dir)
            rb.discard_files(self.retrain_worker["replay_files"])
            print("[retrain] primary -> replay rotated", flush=True)
        else:
            print(f"[retrain] failed: {result.get('error')}", flush=True)

        pred_pkl = os.path.join(self.cfg.run_dir, "predictions_latest.pkl")
        self.rescorer.aggregate_metrics(
            self.current_epoch, self.cfg.progress_csv_path, pred_pkl,
            train_stats=result.get("train_stats"))
        # aggregation is the only consumer and it has the numbers now
        if os.path.exists(pred_pkl):
            os.remove(pred_pkl)

        # re-read rather than increment: aggregate_metrics runs even when the
        # retrain failed, so a local counter can drift from the csv with
        # nothing to resync it. Disk is the one source of truth.
        self.current_epoch = next_model_epoch(self.cfg.progress_csv_path)
        self.retrain_worker = None

        # fresh weights: probe the new epoch in bulk so most of an analysis
        # window lands on one model instead of smearing across epochs
        if not self.stop_signal_sent:
            self.enqueue_blunder_replays(
                BRP_ANALYSIS_EVERY // 2,
                f"post-retrain, epoch {self.current_epoch}")

        rb.sync_live_buffer(self.base_cfg.run_dir,
                            self.rescorer.live_buffer.records)
        self.maybe_start_validation()

    # ---- commands -----------------------------------------------------

    def handle_commands(self):
        c = self.controls
        if c.no_new_games.is_set():
            c.no_new_games.clear()
            if not self.stop_signal_sent:
                self.broadcast("drain_and_stop")
                self.stop_signal_sent = True
                dropped = len(drain_queue(self.game_q))
                print(f"[cmd] no new games -- dropped {dropped} queued, "
                      f"queue frozen", flush=True)

        if c.pause.is_set():
            c.pause.clear()
            if not self.cmd_paused:
                self.broadcast("pause")
                self.cmd_paused = True
                print("[cmd] workers paused", flush=True)

        # cleared whether or not anything was paused: left set it would latch
        # and cancel the next pause on the following pass
        if c.unpause.is_set():
            c.unpause.clear()
            if self.cmd_paused:
                self.broadcast("unpause")
                self.cmd_paused = False
                print("[cmd] workers unpaused", flush=True)
            else:
                print("[cmd] unpause: workers are not paused, no-op",
                      flush=True)

        if c.save_training_data.is_set():
            c.save_training_data.clear()
            self.save_training_snapshot()

        if c.status.is_set():
            c.status.clear()
            self.report()

    def save_training_snapshot(self):
        data = self.rescorer.live_buffer.records
        n = self.controls.save_training_data_n
        if n and n < len(data):
            data = random.sample(data, n)
        name = f"training_snapshot_{int(time.time())}.pkl"
        with open(os.path.join(self.base_cfg.run_dir, name), "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[cmd] saved {len(data)} samples to {name}", flush=True)

    # ---- main loop ----------------------------------------------------

    def run(self):
        self.setup()
        self.spawn_workers()
        self.initial_fill()
        try:
            # the loop ends when the workers are gone and nothing is training.
            # Draining the rescore backlog is shutdown's job, not a condition
            # here -- MAX_BACKLOG used to stall between rounds for that.
            while self.procs or self.retrain_worker is not None:
                if self.controls.stop.is_set():
                    self.procs = check_and_reap_procs(
                        self.procs, self.worker_procs, request_stop=True)
                    break

                self.handle_commands()
                self.procs = check_and_reap_procs(
                    self.procs, self.worker_procs,
                    crash_log_path=self.crash_log_path)

                self.top_up_queue()
                self.drain_results()
                self.tick_rescorer()
                self.check_retrain()
                self.check_validation_timeout()
                time.sleep(0.05)
            return 0 if not self.controls.stop.is_set() else 1
        finally:
            self.shutdown()

    def initial_fill(self):
        """Seed the queue in priority order: probes, validation, selfplay.

        The queue is FIFO, so this order is the only thing deciding what the
        workers see first. Probes lead because they are single-move -- workers
        churn them fast, so the SF rescore threads have work from the first
        seconds instead of idling until full games land. Validation follows,
        which also staggers the Stockfish engines: workers reach the first
        validation game at slightly different times rather than all spinning
        one up at once.
        """
        if self.blunder_pool:
            last = last_epoch_in_blunder_replay_history(
                os.path.join(self.cfg.run_dir, BRP_HISTORY_FILENAME)
            )

            if last is None or self.current_epoch > last:
                self.enqueue_blunder_replays(
                    BRP_ANALYSIS_EVERY // 2,
                    f"startup, epoch {self.current_epoch}"
                )

        self.maybe_start_validation()
        self.top_up_queue()

    def report(self):
        elapsed = time.time() - self.started_at
        # rate is this process's own games; total_games is seeded from disk,
        # so using it would divide a whole run's output by one session's clock
        this_run = self.total_games - self.games_at_start
        gph = 3600 * this_run / elapsed if elapsed else 0.0
        print(f"[runner] {format_time(elapsed)} | {this_run:,} games this "
              f"session ({gph:.1f}/hr) | {self.total_games:,} of "
              f"{self.cfg.n_games:,} budget", flush=True)

    def shutdown(self):
        if self.retrain_worker is not None and self.retrain_worker["p"].is_alive():
            self.retrain_worker["p"].terminate()
        if self.val_pending:
            self.close_partial_validation_batch("shutdown")

        self.drain_rescorer()
        self.rescorer.push_analyzed(report=True)
        # after the drain, never before: pending only empties as these threads
        # return results
        for t in self.sf_rescore_threads:
            t.close()
        if self.rescorer.live_buffer.records:
            path = rb.sync_live_buffer(
                self.base_cfg.run_dir, self.rescorer.live_buffer.records)
            print(f"[runner] saved {len(self.rescorer.live_buffer.records)} "
                  f"samples to {os.path.basename(path)}", flush=True)
        self.rescorer.close(self.blunder_pool)
        self.stop_workers()
        self.report()

    def drain_rescorer(self, timeout_s=300.0):
        """Finish the SF work already in flight before exiting.

        The rescorer never signals "done" -- it is drained when the four
        collections below are empty. pending only empties as the SF threads
        return results, so a dead thread would spin here forever: hence the
        deadline, which reports what it gave up on rather than hanging.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            self.tick_rescorer()
            outstanding = (
                len(self.finished_games) + len(self.finished_brp)
                + len(self.rescorer.intake)
                + len(self.rescorer.blunder_replay_intake)
                + len(self.rescorer.pending))
            if not outstanding:
                return True
            if time.monotonic() > deadline:
                print(f"[runner] rescore drain timed out, {outstanding} "
                      f"games unfinished", flush=True)
                return False
            time.sleep(0.1)

    def stop_workers(self, max_wait_s=15.0):
        deadline = None if max_wait_s is None else time.monotonic() + max_wait_s
        self.procs = check_and_reap_procs(
            self.procs, self.worker_procs, request_stop=True)
        while self.procs and (deadline is None or time.monotonic() < deadline):
            self.procs = check_and_reap_procs(
                self.procs, self.worker_procs, request_stop=True)
            time.sleep(0.05)

        for w in self.procs:
            if w["p"].is_alive():
                w["p"].terminate()
        for w in list(self.procs):
            w["p"].join(timeout=2.0)
            if not w["p"].is_alive():
                close_proc(w["p"], self.worker_procs)
        self.procs = [w for w in self.procs if w["p"].is_alive()]

        for q in (self.recent_q, self.telemetry_q, self.game_q):
            if q is not None:
                q.cancel_join_thread()
                q.close()


def parse_paths(run_tag):
    run_dir = os.path.join(SP_DIR, run_tag)
    if not os.path.isdir(run_dir):
        raise Exception(f"[error] run dir not found: {run_dir}")

    yaml_path = None
    for nm in ("config.yaml", "config.yml"):
        candidate = os.path.join(run_dir, nm)
        if os.path.exists(candidate):
            yaml_path = candidate
            break
    if yaml_path is None:
        raise Exception(f"[error] no config.yaml or config.yml in {run_dir}")

    base_cfg = Config.from_yaml(yaml_path, init=True)
    val_yaml_path = os.path.join(base_cfg.run_dir, "validation_config.yaml")
    return base_cfg, yaml_path, val_yaml_path


def run_roundless(run_tag):
    """Entry point. run_selfplay.py calls this once cutover happens; until
    then it is reachable directly for smoke runs."""
    base_cfg, yaml_path, val_yaml_path = parse_paths(run_tag)
    controls = Controls()
    runner = SelfPlayRunner(
        run_tag, base_cfg, yaml_path, val_yaml_path, controls=controls)

    # run_selfplay's __main__ points these at its own STOP_REQUESTED, which
    # this runner never reads -- and an installed handler stops SIGINT from
    # raising KeyboardInterrupt, so without this Ctrl-C does nothing at all
    def on_signal(signum=None, frame=None):
        print("[signal] stop requested", flush=True)
        controls.stop.set()

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)
    threading.Thread(
        target=stdin_listener,
        args=(controls, runner.worker_procs),
        daemon=True,
    ).start()
    return runner.run()
