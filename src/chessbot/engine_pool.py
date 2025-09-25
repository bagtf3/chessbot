# engine_pool.py
import atexit, threading
import chess.engine

class _EnginePool:
    def __init__(self):
        self._lock = threading.RLock()
        self._eng = None
        self._path = None
        self._depth = 12
        self._users = 0

    def set_path(self, path, depth=12):
        with self._lock:
            self._path = str(path)
            self._depth = int(depth)

    def acquire(self):
        with self._lock:
            if self._eng is None:
                if not self._path:
                    raise RuntimeError("Engine path not set. Call set_path().")
                self._eng = chess.engine.SimpleEngine.popen_uci(self._path)
            self._users += 1
            return self._eng, self._depth

    def release(self):
        with self._lock:
            self._users = max(0, self._users - 1)

    def shutdown(self):
        with self._lock:
            if self._eng is not None:
                try:
                    self._eng.close()
                except Exception:
                    pass
            self._eng = None
            self._users = 0

_POOL = _EnginePool()
atexit.register(_POOL.shutdown)

def set_engine_path(path, depth=12):
    _POOL.set_path(path, depth)

def acquire_engine():
    return _POOL.acquire()

def release_engine():
    _POOL.release()

def shutdown_engine():
    _POOL.shutdown()
