"""
Outer wrapper that runs run_selfplay.py in a self-healing loop.
Usage: python run_selfplay_loop.py <run_tag> [--loops N] [--restart-delay SECS] [--timeout SECS]

--timeout: if the run has not exited within this many seconds, kill the whole
           process tree and start fresh. Default: no timeout.
"""
import subprocess
import sys
import time
import argparse
from datetime import datetime


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def kill_tree(proc):
    """Kill process and all its children (Windows-safe)."""
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[loop] {ts()} warning: taskkill failed: {e}")
        try:
            proc.kill()
        except Exception:
            pass


def run_one(script, run_tag, timeout):
    """
    Launch run_selfplay as a subprocess. If timeout is set and the process
    hasn't exited by then, kill the whole tree and return 'timeout'.
    Returns the exit code, or the string 'timeout'.
    """
    proc = subprocess.Popen([sys.executable, script, run_tag])
    try:
        proc.wait(timeout=timeout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print(f"[loop] {ts()} timeout ({timeout}s) reached -- killing process tree")
        kill_tree(proc)
        proc.wait()
        return "timeout"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_tag")
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--restart-delay", type=float, default=10.0,
                        help="Seconds to wait between restarts")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Kill and restart if run hangs longer than this many seconds")
    args = parser.parse_args()

    script = __file__.replace("run_selfplay_loop.py", "run_selfplay.py")
    timeout_str = f"{args.timeout}s" if args.timeout else "none"
    print(f"[loop] {ts()} starting {args.loops} loop(s) of '{args.run_tag}' "
          f"(timeout={timeout_str})")

    for i in range(1, args.loops + 1):
        print(f"[loop] {ts()} --- loop {i}/{args.loops} ---")
        try:
            rc = run_one(script, args.run_tag, args.timeout)
            if rc == "timeout":
                print(f"[loop] {ts()} loop {i} killed after timeout -- restarting")
            elif rc == 0:
                print(f"[loop] {ts()} loop {i} finished cleanly (rc=0)")
            else:
                print(f"[loop] {ts()} loop {i} exited with rc={rc} -- restarting")
        except Exception as e:
            print(f"[loop] {ts()} loop {i} raised exception: {e} -- restarting")

        if i < args.loops:
            print(f"[loop] {ts()} waiting {args.restart_delay}s before restart...")
            time.sleep(args.restart_delay)

    print(f"[loop] {ts()} all {args.loops} loops done")


if __name__ == "__main__":
    main()
