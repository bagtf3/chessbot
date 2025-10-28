from __future__ import annotations

import multiprocessing as mp
import subprocess, sys, time, queue, os, signal
from dataclasses import dataclass

# Defaults
_DEFAULT_EVAL_SCRIPT = os.path.abspath("/mnt/data/eval_selfplay.py")
_DEFAULT_DEBOUNCE_SEC = 1.0
_DEFAULT_PYTHON = sys.executable


@dataclass
class _Msg:
    kind: str  # "run", "stop", "ping"
    payload: str | None = None


class EvalServer:
    def __init__(self, eval_script: str | None = None, debounce_sec: float = _DEFAULT_DEBOUNCE_SEC):
        self._eval_script = os.path.abspath(eval_script or _DEFAULT_EVAL_SCRIPT)
        if not os.path.exists(self._eval_script):
            raise FileNotFoundError(f"eval script not found: {self._eval_script}")

        self._debounce_sec = float(debounce_sec)

        self._q: mp.Queue[_Msg] = mp.Queue()
        self._proc = mp.Process(
            target=self._worker,
            args=(self._q, self._eval_script, self._debounce_sec),
            name="EvalServerWorker",
            daemon=True,
        )
        self._proc.start()

    # ---------------- Public API ----------------

    def submit(self, file_path: str) -> None:
        """
        Fire-and-forget signal. Returns immediately.

        The worker coalesces multiple submits and launches a single eval run.
        """
        try:
            self._q.put_nowait(_Msg("run", file_path))
        except Exception:
            # Queue might be full/broken; ignore to keep non-blocking contract
            pass

    def stop(self, timeout: float = 10.0) -> None:
        """
        Ask the worker to stop after finishing any active eval.
        """
        try:
            self._q.put_nowait(_Msg("stop"))
        except Exception:
            pass

        if self._proc.is_alive():
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                # Hard terminate as a last resort
                self._proc.kill()

    def is_alive(self) -> bool:
        return self._proc.is_alive()

    # --------------- Worker & helpers ---------------

    @staticmethod
    def _build_cmd(eval_script: str) -> list[str]:
        """
        Command to invoke the eval script. Adjust here if you later add args.
        """
        return [_DEFAULT_PYTHON, "-u", eval_script]

    @staticmethod
    def _launch_eval(eval_script: str) -> int:
        """
        Launch eval_selfplay.py and wait for it to finish.
        Returns process returncode.
        """
        cmd = EvalServer._build_cmd(eval_script)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
            # Stream output to parent stdout so you can see progress in logs
            for line in proc.stdout:
                print(f"[eval] {line.rstrip()}")
            proc.wait()
            return int(proc.returncode or 0)
        except Exception as e:
            print(f"[eval] failed to launch: {e}")
            return -1

    @staticmethod
    def _worker(q: mp.Queue, eval_script: str, debounce_sec: float) -> None:
        """
        Worker loop:
        - Listens for "run" messages.
        - Debounces bursts, then runs the eval script once.
        - Ignores individual file paths; relies on eval script to discover new work.
        """
        pending = False
        last_trigger = 0.0
        running = True

        while running:
            try:
                msg: _Msg = q.get(timeout=0.25)
            except queue.Empty:
                msg = None

            now = time.time()
            if msg:
                if msg.kind == "run":
                    pending = True
                    last_trigger = now
                elif msg.kind == "stop":
                    running = False
                elif msg.kind == "ping":
                    pass

            # If there is pending work and we've passed the debounce window,
            # start one eval job.
            if pending and (now - last_trigger) >= debounce_sec:
                pending = False
                rc = EvalServer._launch_eval(eval_script)
                # Log return code: negative return codes are failures
                if rc != 0:
                    print(f"[eval] process exited with code {rc}")

        # graceful exit
        return


# Convenience constructor
def start(eval_script: str | None = None, debounce_sec: float = _DEFAULT_DEBOUNCE_SEC) -> EvalServer:
    """
    Start the eval server worker and return the controller.
    """
    return EvalServer(eval_script=eval_script, debounce_sec=debounce_sec)
