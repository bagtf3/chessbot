"""
Unattended selfplay: run a tag to completion, --iterate into the next tag,
convert the finished run's analyze pkl to parquet, start the next tag, repeat.
Usage: python run_selfplay_loop.py <run_tag> [--loops N] [options]

A run is complete only when run_selfplay exits 0 AND the on-disk game count has
reached the config's n_games. Anything else relaunches the same tag (the runner
resumes from disk). Iteration is idempotent: if the next tag already has a
config.yaml it is verified and adopted instead of iterated again, so restarting
this script can never chain --iterate forward over a run.

Self-healing:
  - crash / nonzero exit        -> relaunch same tag, backoff grows per failure
  - no output for --silent-min  -> kill tree, relaunch
  - no new game for --stall-min -> kill tree, relaunch (only while budget remains)
  - orphaned workers/stockfish  -> killed before every relaunch
Halts (safe stop, loud log) on: --max-failures consecutive no-progress
failures, free disk under --min-free-gb, or a failed/incomplete iterate.

Remote stop: create <selfplay_dir>/STOP_SELFPLAY_LOOP. The current run
finishes; nothing new starts.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import psutil
import yaml

from chessbot import SP_DIR

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SELFPLAY_SCRIPT = os.path.join(SCRIPTS_DIR, "run_selfplay.py")
INIT_SCRIPT = os.path.join(SCRIPTS_DIR, "init_new_run.py")
PARQUET_SCRIPT = os.path.join(SCRIPTS_DIR, "tools", "convert_analyze_to_parquet.py")
STOP_FILE = os.path.join(SP_DIR, "STOP_SELFPLAY_LOOP")
POLL_S = 10


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, fh=None):
    line = f"[loop] {ts()} {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def run_dir(tag):
    return os.path.join(SP_DIR, tag)


def next_tag(tag):
    m = re.match(r"^(.*?)(\d+)$", tag)
    if not m:
        raise ValueError(f"no trailing integer in '{tag}', cannot iterate")
    return f"{m.group(1)}{int(m.group(2)) + 1}"


def n_games_budget(tag):
    with open(os.path.join(run_dir(tag), "config.yaml"), "r", encoding="utf-8") as fh:
        return int((yaml.safe_load(fh) or {}).get("n_games", 0))


def games_done(tag):
    """Same count the runner resumes against: lines in the staging index (one
    small jsonl, never the game logs). Called at launch/exit and on a suspected
    stall only; the per-poll watchdog uses the file's mtime."""
    path = os.path.join(run_dir(tag), "game_index_staging.json")
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def last_game_time(tag):
    path = os.path.join(run_dir(tag), "game_index_staging.json")
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def run_is_ready(tag):
    """What a freshly iterated run must hold before it can start."""
    d = run_dir(tag)
    need = ["config.yaml", "validation_config.yaml", f"{tag}_model.pt"]
    missing = [n for n in need if not os.path.exists(os.path.join(d, n))]
    if not os.path.isdir(os.path.join(d, "replay_buffer")):
        missing.append("replay_buffer/")
    return missing


def kill_pids(procs, fh):
    for p in procs:
        try:
            if p.is_running():
                p.kill()
        except psutil.NoSuchProcess:
            continue
        except psutil.Error as e:
            log(f"warning: could not kill pid {p.pid}: {e}", fh)
    gone, alive = psutil.wait_procs(procs, timeout=15)
    for p in alive:
        log(f"warning: pid {p.pid} survived kill", fh)


def kill_tree(proc, known, fh):
    """taskkill the live tree, then anything we saw spawn that outlived it."""
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    kill_pids(list(known.values()), fh)


def snapshot_children(pid, known):
    """Remember every descendant by (pid, create_time) so orphans can be found
    after the parent dies and Windows forgets the tree."""
    try:
        for c in psutil.Process(pid).children(recursive=True):
            known[(c.pid, c.create_time())] = c
    except psutil.NoSuchProcess:
        pass
    except psutil.Error:
        pass


def reap_orphans(known, fh):
    alive = []
    for (pid, ctime), p in known.items():
        try:
            if p.is_running() and p.create_time() == ctime:
                alive.append(p)
        except psutil.NoSuchProcess:
            continue
    if alive:
        log(f"killing {len(alive)} orphaned child process(es): "
            f"{', '.join(f'{p.pid}:{p.name()}' for p in alive)}", fh)
        kill_pids(alive, fh)
    known.clear()


def pump_output(stream, fh, beat):
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
        beat[0] = time.time()


def run_selfplay_once(tag, args, fh):
    """One launch of run_selfplay. Returns (rc, reason)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, "-u", SELFPLAY_SCRIPT, tag],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    beat = [time.time()]
    pump = threading.Thread(target=pump_output, args=(proc.stdout, fh, beat),
                            daemon=True)
    pump.start()
    known = {}
    budget = n_games_budget(tag)
    started = time.time()
    log(f"launched run_selfplay {tag} (pid {proc.pid})", fh)

    try:
        while proc.poll() is None:
            time.sleep(POLL_S)
            snapshot_children(proc.pid, known)
            now = time.time()
            silent_min = (now - beat[0]) / 60
            if silent_min > args.silent_min:
                log(f"no output for {silent_min:.0f} min -- killing tree", fh)
                kill_tree(proc, known, fh)
                proc.wait()
                return None, "silent"
            last_game = max(last_game_time(tag), started)
            stall_min = (now - last_game) / 60
            if stall_min > args.stall_min and games_done(tag) < budget:
                log(f"no new game for {stall_min:.0f} min with budget left "
                    f"-- killing tree", fh)
                kill_tree(proc, known, fh)
                proc.wait()
                return None, "stalled"
    except KeyboardInterrupt:
        log("Ctrl-C: waiting up to 10 min for run_selfplay to stop cleanly", fh)
        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            kill_tree(proc, known, fh)
        raise
    finally:
        pump.join(timeout=5)
        reap_orphans(known, fh)

    return proc.returncode, "exited"


def iterate(tag, fh):
    """--iterate tag -> next tag, exactly once. Returns the new tag or None."""
    new = next_tag(tag)
    if os.path.exists(os.path.join(run_dir(new), "config.yaml")):
        log(f"{new} already exists -- adopting it instead of iterating again", fh)
    else:
        log(f"iterating {tag} -> {new}", fh)
        res = subprocess.run(
            [sys.executable, "-u", INIT_SCRIPT, "--iterate", tag],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=SCRIPTS_DIR)
        out = res.stdout.decode("utf-8", errors="replace")
        print(out, flush=True)
        fh.write(out)
        fh.flush()
        if res.returncode != 0:
            log(f"HALT: init_new_run --iterate {tag} exited {res.returncode}", fh)
            return None
    missing = run_is_ready(new)
    if missing:
        log(f"HALT: {new} is incomplete, missing {missing}. Fix by hand; not "
            f"retrying because --iterate deletes the source buffers", fh)
        return None
    return new


def convert_to_parquet(tag, fh):
    """Convert a finished, iterated-away (cold) run's analyze pkl. A failure
    keeps the pkl intact, so it is logged loudly and the loop carries on."""
    log(f"converting {tag} analyze pkl -> parquet", fh)
    res = subprocess.run(
        [sys.executable, "-u", PARQUET_SCRIPT, tag, "--base", SP_DIR],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = res.stdout.decode("utf-8", errors="replace")
    print(out, flush=True)
    fh.write(out)
    fh.flush()
    if res.returncode != 0:
        log(f"WARNING: parquet conversion of {tag} exited {res.returncode}; "
            f"pkl kept, continuing", fh)


def run_to_completion(tag, args, fh):
    """Relaunch `tag` until it completes. True on completion, False on halt."""
    failures = 0
    while True:
        budget = n_games_budget(tag)
        before = games_done(tag)
        if free_gb(SP_DIR) < args.min_free_gb:
            log(f"HALT: {free_gb(SP_DIR):.0f} GB free < {args.min_free_gb} GB", fh)
            return False
        log(f"{tag}: {before:,} / {budget:,} games", fh)

        rc, reason = run_selfplay_once(tag, args, fh)
        after = games_done(tag)
        if rc == 0 and after >= budget:
            log(f"{tag} complete: {after:,} games", fh)
            return True

        progressed = after > before
        failures = 0 if progressed else failures + 1
        log(f"{tag} {reason} rc={rc}, games {before:,} -> {after:,}, "
            f"consecutive no-progress failures {failures}/{args.max_failures}", fh)
        if failures >= args.max_failures:
            log(f"HALT: {tag} failed {failures}x without progress", fh)
            return False
        delay = args.restart_delay * min(max(failures, 1), 6)
        log(f"relaunching {tag} in {delay:.0f}s", fh)
        time.sleep(delay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_tag")
    p.add_argument("--loops", type=int, default=15,
                   help="runs to complete, counting the starting tag")
    p.add_argument("--restart-delay", type=float, default=30.0,
                   help="base seconds between relaunches; grows with failures")
    p.add_argument("--max-failures", type=int, default=5,
                   help="consecutive relaunches with no new games before halting")
    p.add_argument("--silent-min", type=float, default=30.0,
                   help="kill + relaunch after this many minutes with no output")
    p.add_argument("--stall-min", type=float, default=90.0,
                   help="kill + relaunch after this many minutes with no new game")
    p.add_argument("--min-free-gb", type=float, default=5.0)
    args = p.parse_args()

    tag = args.run_tag
    if not os.path.exists(os.path.join(run_dir(tag), "config.yaml")):
        sys.exit(f"[loop] no config.yaml for {tag} in {SP_DIR}")
    next_tag(tag)   # fail now, not after a multi-day run, if tag can't iterate

    log_path = os.path.join(SP_DIR, f"selfplay_loop_{datetime.now():%Y%m%d_%H%M%S}.log")
    with open(log_path, "a", encoding="utf-8") as fh:
        log(f"starting at {tag}, {args.loops} run(s), log -> {log_path}", fh)
        try:
            for i in range(1, args.loops + 1):
                if os.path.exists(STOP_FILE):
                    log(f"stop file present ({STOP_FILE}) -- stopping", fh)
                    break
                log(f"=== run {i}/{args.loops}: {tag} ===", fh)
                if not run_to_completion(tag, args, fh):
                    sys.exit(1)
                if i == args.loops:
                    break
                if os.path.exists(STOP_FILE):
                    log(f"stop file present ({STOP_FILE}) -- not iterating", fh)
                    break
                new = iterate(tag, fh)
                if new is None:
                    sys.exit(1)
                convert_to_parquet(tag, fh)
                tag = new
            log(f"done. last run: {tag}", fh)
        except KeyboardInterrupt:
            log(f"stopped by Ctrl-C at {tag}", fh)
            sys.exit(130)


if __name__ == "__main__":
    main()
