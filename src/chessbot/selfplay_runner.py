"""
Roundless selfplay orchestrator.

The run has one continuous lifetime instead of a sequence of rounds. Workers
spawn once and only come down at the end or to be replaced after a crash;
validation is a batch of specs dropped into the same queue everything else
uses, not a mode the whole fleet switches into.

Lives here rather than in scripts/ so it can be imported and exercised, and
so the mp children import a real module instead of re-importing the entry
script under spawn. scripts/run_selfplay.py is a shim over run_roundless.
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

from chessbot.lc0_replay import (
    LC0_BRP_SCENARIO, create_lc0_replay_config,
)

from chessbot.rescore import (
    BRP_ANALYSIS_EVERY, Rescorer, SFRescoreThread, migrate_pretrain_progress,
)
from chessbot.retrain_worker import run_retrain_worker
from chessbot.review import RecordKeeper
from chessbot.utils import format_time, make_jsonable, next_model_epoch
from chessbot.validation import (
    build_validation_summary_from_rows, create_validation_config,
    print_validation_info,
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
# the retrain worker sleeps this long after retrain_ready before it touches
# the GPU; a pause issued outside that handshake gets the same settle window
WORKER_PAUSE_SETTLE_S = 10.0
# a worker that dies on startup would otherwise be respawned every loop pass
RESPAWN_COOLDOWN_S = 30.0
# brief mop-up for SF work already in flight when an operator stops. Kept
# short on purpose: anything unfinished is on disk and resumes next run.
STOP_DRAIN_S = 5.0
# the two worker fleets. Pause, respawn and status all address them by kind.
WORKER_KINDS = ("xc0", "lc0")


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


HELP_TEXT = """[cmd] commands:
  stop                  finish up, save data, exit
  stop now              hard kill
  no new games          drain what is in flight
  pause [kind] [secs]   kind is xc0 or lc0; bare pause holds both
  unpause [kind]        releases the operator hold on that fleet
  add sf worker         one more SF rescore thread
  kill sf worker        retire the oldest SF thread after its current game
  respawn lc0           pause, tear down and rebuild the lc0 fleet from yaml
  full clear            drop all pauses, sf back to the config count
  save training data N  snapshot N live-buffer samples
  status                fleets, sf threads, pause state, backlog
  help                  this"""


def parse_kinds_and_dur(args):
    """(kinds, duration) for a pause/unpause argument list.

    A bare command addresses every fleet. Returns (None, None) on an
    unrecognised fleet name so the caller can complain rather than silently
    pausing everything.
    """
    kinds, dur = set(), None
    for a in args:
        if a.isdigit():
            dur = int(a)
        elif a in WORKER_KINDS:
            kinds.add(a)
        else:
            return None, None
    return (kinds or set(WORKER_KINDS)), dur


def stdin_listener(controls, worker_procs):
    """Operator console. Takes the Controls it drives rather than reaching
    for module globals, so the listener and the runner provably share one
    set of flags.
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
            kinds, dur = parse_kinds_and_dur(parts[1:])
            if kinds is None:
                print(f"[cmd] unknown fleet: {' '.join(parts[1:])!r}",
                      flush=True)
                continue
            suffix = f" for {dur}s" if dur else ""
            print(f"[cmd] pausing {'+'.join(sorted(kinds))}{suffix}",
                  flush=True)
            controls.pause_kinds |= kinds
            controls.pause.set()
            if dur:
                def auto_unpause(d=dur, k=set(kinds)):
                    time.sleep(d)
                    print("[cmd] auto-unpause", flush=True)
                    controls.unpause_kinds |= k
                    controls.unpause.set()
                threading.Thread(target=auto_unpause, daemon=True).start()

        elif parts[0] == 'unpause':
            kinds, _ = parse_kinds_and_dur(parts[1:])
            if kinds is None:
                print(f"[cmd] unknown fleet: {' '.join(parts[1:])!r}",
                      flush=True)
                continue
            print(f"[cmd] unpausing {'+'.join(sorted(kinds))}", flush=True)
            controls.unpause_kinds |= kinds
            controls.unpause.set()

        elif ' '.join(parts[:3]) == 'add sf worker':
            controls.add_sf_n += 1
            print("[cmd] add sf worker requested", flush=True)

        elif ' '.join(parts[:3]) == 'kill sf worker':
            controls.kill_sf_n += 1
            print("[cmd] kill sf worker requested", flush=True)

        elif cmd == 'respawn lc0':
            controls.respawn_lc0.set()
            print("[cmd] respawn lc0 requested", flush=True)

        elif cmd == 'full clear':
            print("[cmd] full clear -- dropping pauses, sf back to config",
                  flush=True)
            controls.full_clear.set()

        elif cmd in ('help', '?'):
            print(HELP_TEXT, flush=True)

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
    """Operator flags, set by the stdin listener and read by the runner."""

    def __init__(self):
        self.stop = threading.Event()
        self.pause = threading.Event()
        self.unpause = threading.Event()
        self.no_new_games = threading.Event()
        self.save_training_data = threading.Event()
        self.save_training_data_n = None
        self.status = threading.Event()
        self.full_clear = threading.Event()
        self.respawn_lc0 = threading.Event()

        # kinds accumulate rather than overwrite, so two commands typed inside
        # one loop pass do not lose the first one
        self.pause_kinds = set()
        self.unpause_kinds = set()

        # counts, not flags: repeated commands stack
        self.add_sf_n = 0
        self.kill_sf_n = 0


class SelfPlayRunner:
    """
    Owns every piece of long-lived state for the run. Phase methods take no
    arguments and read self, which is the point: the free functions this
    replaced took up to 8 parameters and the bugs came from passing the
    wrong value.
    """

    def __init__(self, run_tag, base_cfg, yaml_path, val_yaml_path, controls=None):
        self.run_tag = run_tag
        self.base_cfg = base_cfg
        self.yaml_path = yaml_path
        self.val_yaml_path = val_yaml_path
        self.controls = controls or Controls()

        self.cfg = None            # live working config, reloaded at retrain
        self.val_cfg = None        # ladder config; owns sf_validation_depth

        self.rescorer = None
        self.recorder = None
        self.game_gen = None
        self.blunder_pool = None
        self.sf_rescore_threads = []

        self.ctx = mp.get_context()
        self.game_q = None
        self.lc0_q = None
        self.recent_q = None
        self.telemetry_q = None
        self.procs = []
        self.worker_procs = []
        self.lc0_cfg = None

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
        # kinds actually paused right now, and the subset the operator is
        # holding. Retrain may only release what it is not holding.
        self.paused_kinds = set()
        self.operator_paused = set()
        self.stop_signal_sent = False
        self.started_at = None
        self.next_worker_id = 0
        self.last_respawn_at = 0.0

        self.sf_game_q = None
        self.sf_res_q = None
        self.next_sf_id = 0
        # retired threads, still finishing the game they were mid-way through
        self.sf_retiring = []

    def setup(self):
        """One-time startup: SF threads, rescorer, buffers, epoch counter."""
        cfg = self.base_cfg

        self.sf_game_q = sf_game_q = queue.Queue()
        self.sf_res_q = sf_res_q = queue.Queue()
        for _ in range(max(1, cfg.rescore_n_sf_workers)):
            self.add_sf_thread()

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
        self.recorder = RecordKeeper(self.current_epoch, every_sec=60.0)
        self.recorder.training_queue = len(self.rescorer.live_buffer)
        self.recorder.primary_buffer_dir = cfg.primary_buffer_dir
        self.recorder.primary_buffer_trigger = rb.PRIMARY_TRIGGER_SHARDS

        self.reload_config()
        self.val_cfg = create_validation_config(
            self.cfg, self.val_yaml_path, verbose=False)
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

        One engine covers everything: validation shares the selfplay batch
        shapes exactly, and its search params travel per-game on the spec's
        own cfg, so there is no separate val_trt build.
        """
        self.cfg = Config.from_yaml(self.yaml_path, init=True)
        if self.cfg.inference_backend == 'ort_trt':
            trt_dir, model_name, _ = selfplay_trt_paths(self.cfg)
            self.cfg = prepare_trt(self.cfg, trt_dir, model_name)
        if self.rescorer is not None:
            self.rescorer.config = self.cfg
        # the generator holds a config reference and reads it live, so
        # re-pointing it here is what makes a yaml edit take effect
        if self.game_gen is not None:
            self.game_gen.config = self.cfg
        # search params ride each spec's own cfg, so rebuilding this is enough
        # for a live lc0 worker to pick them up. Batch sizes, encoding and the
        # net are read at spawn and still need a restart.
        if self.lc0_cfg is not None:
            self.lc0_cfg = create_lc0_replay_config(self.cfg)
            self.rescorer.lc0_cfg = self.lc0_cfg

    def spawn_worker(self):
        """Start one worker off the current config, so a replacement picks
        up the latest TRT engine without any extra plumbing. Ids come from a
        monotonic counter rather than the fleet index: a respawn must not
        reuse the dead worker's name or the crash log stops making sense."""
        c = self.cfg.copy()
        c.id = f"w{self.next_worker_id}"
        self.next_worker_id += 1
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
            "kind": "xc0", "stop_sent_at": None, "term_sent_at": None,
            "kill_sent_at": None,
        })
        return c.id

    def spawn_lc0_worker(self):
        """Start one lc0 teacher probe worker.

        Lives in self.procs so it inherits pause, reap and respawn, but never
        reads game_q: the two fleets would steal each other's specs.
        """
        c = self.lc0_cfg.copy()
        c.id = f"lc0w{self.next_worker_id}"
        self.next_worker_id += 1
        stop_ev = self.ctx.Event()
        msg_q = self.ctx.Queue()
        p = self.ctx.Process(
            target=child_looper,
            args=(c, stop_ev, self.recent_q, self.telemetry_q, msg_q,
                  self.lc0_q),
            daemon=True,
        )
        p.start()
        self.worker_procs.append(p)
        self.procs.append({
            "id": c.id, "p": p, "stop_ev": stop_ev, "msg_q": msg_q,
            "kind": "lc0", "stop_sent_at": None, "term_sent_at": None,
            "kill_sent_at": None,
        })
        return c.id

    def procs_of(self, kind):
        return [w for w in self.procs if w["kind"] == kind]

    def lc0_wanted(self):
        return self.cfg.n_lc0_workers if self.lc0_cfg is not None else 0

    def spawn_workers(self):
        """One queue to every selfplay worker. There is no designated
        validator: whoever has capacity picks up whatever spec is next, and
        validation is told apart by spec.meta, not by which process pulled it.
        """
        self.recent_q = self.ctx.Queue()
        self.telemetry_q = self.ctx.Queue()
        self.game_q = self.ctx.Queue()

        for _ in range(max(1, self.cfg.n_workers)):
            self.spawn_worker()
        print(f"[runner] spawned {len(self.procs)} workers", flush=True)

        if self.cfg.n_lc0_workers:
            self.lc0_q = self.ctx.Queue()
            self.lc0_cfg = create_lc0_replay_config(self.cfg)
            # handing the rescorer the config is what switches blunders from
            # the blend path onto the teacher path
            self.rescorer.lc0_cfg = self.lc0_cfg
            for _ in range(self.cfg.n_lc0_workers):
                self.spawn_lc0_worker()
            print(f"[runner] spawned {self.cfg.n_lc0_workers} lc0 workers "
                  f"({self.lc0_cfg.lc0_distill_model_name})", flush=True)

    def drain_lc0_specs(self):
        """Move staged teacher probes from the rescorer onto the lc0 queue."""
        if self.lc0_q is None:
            return 0
        specs = self.rescorer.take_lc0_specs()
        for spec in specs:
            self.lc0_q.put(spec)
        self.recorder.lc0_backlog = self.lc0_q.qsize()
        return len(specs)

    def fleet_wanted(self):
        """True while the run still wants workers running.

        The main loop needs this as well as self.procs: between the last
        worker dying and its replacement starting the fleet is empty for up
        to RESPAWN_COOLDOWN_S, and self.procs alone would end the run there.
        """
        if (self.stop_signal_sent
                or self.controls.stop.is_set()
                or self.controls.no_new_games.is_set()):
            return False
        return self.total_queued < self.cfg.n_games

    def maybe_respawn_workers(self):
        """Backfill the fleet after a crash.

        Rounds used to hide worker deaths by rebuilding the fleet at every
        boundary. Roundless nothing does, so without this a 50k-game run
        degrades 3 -> 2 -> 1 and then exits with budget left.
        """
        if not self.fleet_wanted():
            return 0

        now = time.monotonic()
        if now - self.last_respawn_at < RESPAWN_COOLDOWN_S:
            return 0

        # one per pass, so a fleet that lost several workers refills over
        # several cooldowns rather than storming the GPU at once
        for kind, target, spawn in (
            ("xc0", max(1, self.cfg.n_workers), self.spawn_worker),
            ("lc0", self.lc0_wanted(), self.spawn_lc0_worker),
        ):
            # a fresh worker starts unpaused and would contend for the GPU
            # with whatever the pause was protecting; wait for the unpause
            if kind in self.paused_kinds:
                continue
            if len(self.procs_of(kind)) >= target:
                continue
            self.last_respawn_at = now
            wid = spawn()
            print(f"[runner] respawned {kind} worker {wid} "
                  f"(fleet {len(self.procs_of(kind))}/{target})", flush=True)
            return 1
        return 0

    def broadcast(self, cmd, kinds=None):
        for w in self.procs:
            if kinds is None or w["kind"] in kinds:
                w["msg_q"].put(cmd)

    def pause_workers(self, why, kinds=None):
        """Pause the named fleets, idempotently.

        Only kinds that are actually running are sent anything. An unpause
        delivered to a running worker is not a no-op -- the looper buffers it
        as unpause_queued and skips the *next* pause, playing straight through
        a retrain -- so state is tracked here and never re-sent.
        """
        kinds = set(kinds) if kinds else set(WORKER_KINDS)
        todo = kinds - self.paused_kinds
        if not todo:
            return False
        self.broadcast("pause", todo)
        self.paused_kinds |= todo
        print(f"[runner] paused {'+'.join(sorted(todo))} ({why})", flush=True)
        return True

    def unpause_workers(self, why, kinds=None):
        """Release the named fleets, minus anything the operator is holding.

        This filter is what stops a finishing retrain from silently undoing an
        operator pause issued while it was in flight.
        """
        kinds = set(kinds) if kinds else set(WORKER_KINDS)
        todo = (kinds & self.paused_kinds) - self.operator_paused
        if not todo:
            return False
        self.broadcast("unpause", todo)
        self.paused_kinds -= todo
        print(f"[runner] unpaused {'+'.join(sorted(todo))} ({why})", flush=True)
        return True

    def respawn_lc0_workers(self, why="operator"):
        """Pause, tear down and rebuild the lc0 fleet from a fresh read of
        lc0_replay_config.yaml -- the only way spawn-time params like
        games_at_once take effect without a full run restart. xc0 has no
        equivalent command; its workers already reload every retrain."""
        if self.lc0_cfg is None:
            print("[runner] respawn lc0: no lc0 fleet configured", flush=True)
            return
        self.pause_workers(why, kinds={"lc0"})
        self.operator_paused.add("lc0")
        for w in self.procs_of("lc0"):
            w["stop_ev"].set()
            try:
                w["p"].terminate()
            except Exception:
                pass
        self.procs = [w for w in self.procs if w["kind"] != "lc0"]
        self.lc0_cfg = create_lc0_replay_config(self.cfg)
        self.rescorer.lc0_cfg = self.lc0_cfg
        self.paused_kinds.discard("lc0")
        self.operator_paused.discard("lc0")
        for _ in range(self.cfg.n_lc0_workers):
            self.spawn_lc0_worker()
        print(f"[runner] respawned lc0 fleet ({why})", flush=True)

    def top_up_queue(self, target=None):
        """Keep the queue shallow. Depth is what keeps validation batches
        from queueing behind selfplay -- see GAME_QUEUE_TARGET. The startup
        fill passes a deeper target so the fleet reaches capacity at once
        instead of over several loop passes."""
        if self.stop_signal_sent or self.controls.no_new_games.is_set():
            return 0

        target = GAME_QUEUE_TARGET if target is None else target
        budget = self.cfg.n_games - self.total_queued
        added = 0
        while added < budget and self.game_q.qsize() < target:
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
        return self.enqueue_blunder_replays(n)

    def enqueue_blunder_replays(self, n):
        """Queue n replay probes, from either the trickle or a bulk dump.
        Progress shows up in the rescore table's replays column."""
        if self.blunder_pool is None or n <= 0:
            return 0
        specs = self.game_gen.next_blunder_replays(
            self.blunder_pool, n, current_epoch=self.current_epoch,
        )
        for spec in specs:
            self.game_q.put(spec)
        return len(specs)

    def maybe_start_validation(self):
        """Fire on the retrain cadence. A batch is a discrete, non-overlapping
        set: the ladder counts runs, not games, so overlapping windows would
        advance depth roughly twice as fast."""
        if self.val_pending or self.stop_signal_sent:
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
              f"{self.current_epoch}, sf_depth "
              f"{self.val_cfg.sf_validation_depth}",
              flush=True)
        print_validation_info(self.val_cfg)

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
        self.val_cfg = create_validation_config(
            self.cfg, self.val_yaml_path, verbose=False)

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

    def drain_results(self):
        """Route finished games. Validation metas are counted here: the
        validation config pins min_game_length to 1, so they cannot vanish
        through the looper's short-game filter and a plain counter is
        honest."""
        for msg in drain_queue(self.telemetry_q):
            self.recorder.ingest_telemetry(msg)

        for game in drain_queue(self.recent_q):
            meta = game["meta"]
            scenario = meta.get("scenario")

            if scenario == "blunder_replay_probe":
                self.finished_brp.append(game)
                continue

            if scenario == LC0_BRP_SCENARIO:
                self.rescorer.ingest_lc0_replay(meta)
                self.recorder.lc0_probes_done += 1
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

        self.recorder.lc0_audit_ema = self.rescorer.lc0_audit_ema
        self.recorder.lc0_audit_n = self.rescorer.lc0_audit_n
        self.recorder.brp_admit_n = self.rescorer.brp_admit_n
        self.recorder.brp_admit_ok = self.rescorer.brp_admit_ok
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
        self.scale_sf()
        self.recorder.training_queue = self.rescorer.training_data_size

    def sf_target(self):
        cfg = self.cfg or self.base_cfg
        return max(1, cfg.rescore_n_sf_workers)

    def add_sf_thread(self, why="config"):
        """Start one SF rescore thread and append it to the tail."""
        cfg = self.cfg or self.base_cfg
        t = SFRescoreThread(self.sf_game_q, self.sf_res_q, cfg)
        t.id = f"sf{self.next_sf_id}"
        self.next_sf_id += 1
        t.start()
        self.sf_rescore_threads.append(t)
        print(f"[sf] added {t.id} ({why}), now "
              f"{len(self.sf_rescore_threads)} threads", flush=True)
        return t.id

    def retire_sf_thread(self, why="operator"):
        """Retire the oldest thread. Asynchronous by design: the thread checks
        its stop flag once per game, so joining here would block the run loop
        for the length of a full analysis."""
        if len(self.sf_rescore_threads) <= 1:
            print("[sf] refusing to retire the last thread", flush=True)
            return None
        t = self.sf_rescore_threads.pop(0)
        t.stop_ev.set()
        self.sf_retiring.append(t)
        print(f"[sf] retiring {t.id} ({why}), finishing its current game; "
              f"{len(self.sf_rescore_threads)} threads left", flush=True)
        return t.id

    def reap_sf_threads(self):
        """Quit the engine of any retired thread that has exited its loop.

        run() does not quit the engine on the way out -- close() did that after
        the join -- so the Stockfish process is only released here.
        """
        still = []
        for t in self.sf_retiring:
            if t.t is not None and t.t.is_alive():
                still.append(t)
                continue
            if t.eng is not None:
                t.eng.quit()
                t.eng = None
            print(f"[sf] {t.id} finished and gone", flush=True)
        self.sf_retiring = still

    def scale_sf(self):
        """Run one extra SF thread while the rescore backlog is deep.

        Bounded at target + 1, which is what makes it self-limiting: the add
        condition goes false as soon as the extra exists, so no cooldown or
        counter is needed. Dropping under the low mark returns it to config.
        """
        self.reap_sf_threads()
        backlog = len(self.rescorer.intake) + len(self.rescorer.pending)
        target = self.sf_target()
        n = len(self.sf_rescore_threads)
        self.rescorer.current_movetime_ms = self.sf_rescore_threads[0].movetime_ms

        if backlog > 100 and n < target + 1:
            self.add_sf_thread("backlog")
        elif backlog < 10 and n > target:
            self.retire_sf_thread("backlog cleared")

    def check_retrain(self):
        self.maybe_launch_retrain()
        self.poll_retrain()

    def maybe_launch_retrain(self):
        if self.retrain_worker is not None:
            return
        # an operator pause blocks retraining, and with it validation and the
        # bulk BRP enqueue, both of which hang off poll_retrain
        if self.operator_paused:
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
            self.pause_workers("retrain")
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

        # reload_config deletes and rebuilds the .engine file, so the workers
        # must have dropped their sessions first. retrain_ready already
        # paused them on the normal path; a worker that died before sending
        # it did not, so pause here and allow the same settle window the
        # retrain worker gives itself after the handshake.
        if self.pause_workers("retrain teardown"):
            time.sleep(WORKER_PAUSE_SETTLE_S)

        self.recorder.n_retrains += 1
        self.reload_config()
        for t in self.sf_rescore_threads:
            t.update_config(self.cfg)
        self.rescorer.current_movetime_ms = self.sf_rescore_threads[0].movetime_ms

        # hard rule: unpause as soon as training is done, the TRT rebuild in
        # reload_config being the one unavoidable prerequisite. A pause the
        # operator issued mid-retrain survives -- unpause_workers filters it
        # out -- and blocks the next retrain until they clear it.
        self.unpause_workers("retrain complete")

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
            self.enqueue_blunder_replays(BRP_ANALYSIS_EVERY // 2)

        rb.sync_live_buffer(self.base_cfg.run_dir,
                            self.rescorer.live_buffer.records)
        self.maybe_start_validation()

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
            kinds, c.pause_kinds = c.pause_kinds, set()
            self.operator_paused |= kinds
            if not self.pause_workers("operator", kinds):
                print("[cmd] pause: already paused, hold recorded", flush=True)

        # cleared whether or not anything was paused: left set it would latch
        # and cancel the next pause on the following pass
        if c.unpause.is_set():
            c.unpause.clear()
            kinds, c.unpause_kinds = c.unpause_kinds, set()
            # drop the hold first, or unpause_workers filters out the very
            # kinds the operator is asking for
            self.operator_paused -= kinds
            if not self.unpause_workers("operator", kinds):
                print("[cmd] unpause: nothing paused there, no-op", flush=True)

        if c.respawn_lc0.is_set():
            c.respawn_lc0.clear()
            self.respawn_lc0_workers("operator")

        if c.full_clear.is_set():
            c.full_clear.clear()
            self.full_clear()

        while c.add_sf_n > 0:
            c.add_sf_n -= 1
            self.add_sf_thread("operator")

        while c.kill_sf_n > 0:
            c.kill_sf_n -= 1
            self.retire_sf_thread("operator")

        if c.save_training_data.is_set():
            c.save_training_data.clear()
            self.save_training_snapshot()

        if c.status.is_set():
            c.status.clear()
            self.report()

    def full_clear(self):
        """Drop every operator hold and return the SF fleet to the config.

        A retrain in flight owns its own pause; releasing that would put the
        workers back on the GPU underneath it, so the hold is dropped and the
        unpause is left to the retrain's own completion path.
        """
        self.operator_paused = set()
        if self.retrain_worker is not None:
            print("[cmd] full clear: holds dropped, but a retrain is in "
                  "flight -- it will unpause when it finishes", flush=True)
        else:
            self.unpause_workers("full clear")

        while len(self.sf_rescore_threads) > self.sf_target():
            if self.retire_sf_thread("full clear") is None:
                break
        while len(self.sf_rescore_threads) < self.sf_target():
            self.add_sf_thread("full clear")

    def save_training_snapshot(self):
        data = self.rescorer.live_buffer.records
        n = self.controls.save_training_data_n
        if n and n < len(data):
            data = random.sample(data, n)
        name = f"training_snapshot_{int(time.time())}.pkl"
        with open(os.path.join(self.base_cfg.run_dir, name), "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[cmd] saved {len(data)} samples to {name}", flush=True)

    def run(self):
        self.setup()
        self.spawn_workers()
        self.initial_fill()
        try:
            # the run is over only once the budget is spent, every game has
            # been played, the rescorer has nothing left, and no retrain is in
            # flight. Staying in the loop for the backlog is what lets a
            # retrain that comes due during the final drain still fire; a dead
            # SF thread surfaces as a raise out of tick, not a hang.
            while (self.procs
                   or self.fleet_wanted()
                   or self.retrain_worker is not None
                   or self.rescore_outstanding()):
                if self.controls.stop.is_set():
                    self.procs = check_and_reap_procs(
                        self.procs, self.worker_procs, request_stop=True)
                    break

                self.handle_commands()
                self.procs = check_and_reap_procs(
                    self.procs, self.worker_procs,
                    crash_log_path=self.crash_log_path)
                self.maybe_respawn_workers()

                self.top_up_queue()
                self.drain_lc0_specs()
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

        Selfplay fills deeper than the steady-state target: every worker slot
        is empty right now, so the shallow target would take several loop
        passes to bring the fleet up to capacity.
        """
        if self.blunder_pool:
            last = last_epoch_in_blunder_replay_history(
                os.path.join(self.cfg.run_dir, BRP_HISTORY_FILENAME)
            )

            if last is None or self.current_epoch > last:
                self.enqueue_blunder_replays(BRP_ANALYSIS_EVERY // 2)

        self.maybe_start_validation()
        self.top_up_queue(
            target=self.cfg.n_workers * self.cfg.games_at_once
            + GAME_QUEUE_TARGET
        )

    def report(self):
        elapsed = time.time() - self.started_at
        # rate is this process's own games; total_games is seeded from disk,
        # so using it would divide a whole run's output by one session's clock
        this_run = self.total_games - self.games_at_start
        gph = 3600 * this_run / elapsed if elapsed else 0.0
        print(f"[runner] {format_time(elapsed)} | {this_run:,} games this "
              f"session ({gph:.1f}/hr) | {self.total_games:,} of "
              f"{self.cfg.n_games:,} budget", flush=True)

        fleets = []
        for kind, target in (("xc0", max(1, self.cfg.n_workers)),
                             ("lc0", self.lc0_wanted())):
            state = "paused" if kind in self.paused_kinds else "running"
            if kind in self.operator_paused:
                state = "paused(operator)"
            fleets.append(f"{kind} {len(self.procs_of(kind))}/{target} {state}")
        print(f"[status]  {' | '.join(fleets)}", flush=True)

        retiring = f" +{len(self.sf_retiring)} retiring" if self.sf_retiring \
            else ""
        ids = ",".join(t.id for t in self.sf_rescore_threads)
        backlog = len(self.rescorer.intake) + len(self.rescorer.pending)
        print(f"[status]  sf {len(self.sf_rescore_threads)}/"
              f"{self.sf_target()}{retiring} [{ids}] at "
              f"{self.rescorer.current_movetime_ms}ms | backlog {backlog}",
              flush=True)

        if self.retrain_worker is not None:
            retrain = "in flight"
        elif self.operator_paused:
            retrain = "blocked by operator pause"
        else:
            retrain = "idle"
        print(f"[status]  retrain {retrain} | epoch {self.current_epoch}",
              flush=True)

    def shutdown(self):
        if self.retrain_worker is not None and self.retrain_worker["p"].is_alive():
            self.retrain_worker["p"].terminate()
        if self.val_pending:
            self.close_partial_validation_batch("shutdown")

        # Only on a stop. The run loop will not exit on its own until
        # rescore_outstanding() is zero, so draining the natural path again
        # here would always be a no-op.
        if self.controls.stop.is_set():
            self.drain_rescorer(STOP_DRAIN_S)

        self.rescorer.push_analyzed(report=True)
        # after the drain, never before: pending only empties as these threads
        # return results
        for t in self.sf_rescore_threads + self.sf_retiring:
            t.close()
        self.sf_retiring = []
        if self.rescorer.live_buffer.records:
            path = rb.sync_live_buffer(
                self.base_cfg.run_dir, self.rescorer.live_buffer.records)
            print(f"[runner] saved {len(self.rescorer.live_buffer.records)} "
                  f"samples to {os.path.basename(path)}", flush=True)
        self.rescorer.close(self.blunder_pool)
        self.stop_workers()
        self.report()

    def rescore_outstanding(self):
        """Games still owed a Stockfish pass, anywhere in the pipeline."""
        return (len(self.finished_games) + len(self.finished_brp)
                + len(self.rescorer.intake)
                + len(self.rescorer.blunder_replay_intake)
                + len(self.rescorer.pending))

    def drain_rescorer(self, timeout_s=STOP_DRAIN_S):
        """Finish the SF work already in flight before exiting.

        The rescorer never signals "done" -- it is drained when the four
        collections below are empty. pending only empties as the SF threads
        return results, so a dead thread would spin here forever: hence the
        deadline, which reports what it gave up on rather than hanging.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            self.tick_rescorer()
            outstanding = self.rescore_outstanding()
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

        for q in (self.recent_q, self.telemetry_q, self.game_q,
                  self.lc0_q):
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
    """Entry point. scripts/run_selfplay.py is a shim over this."""
    base_cfg, yaml_path, val_yaml_path = parse_paths(run_tag)
    controls = Controls()
    runner = SelfPlayRunner(
        run_tag, base_cfg, yaml_path, val_yaml_path, controls=controls)

    # an installed handler stops SIGINT from raising KeyboardInterrupt, so
    # these are what make Ctrl-C reach the runner at all
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
