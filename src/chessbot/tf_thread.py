import time
import threading
from queue import Queue, Empty
from collections import deque

import numpy as np
from pyfastchess import raw_cache_bulk_insert

KEY_PAD = -9999
class Batcher(object):
    def __init__(self, cfg, batch_candidates):
        self.cfg = cfg
        self.batch_candidates = batch_candidates

        self.currently_batched_keys = set()
        self.keys = []
        self.boards = []
        self.legals = []

        self.total_seen = 0
        self.duplicate_removed = 0

    def __len__(self):
        return len(self.boards)
    
    def submit(self, micro_batch):
        """ Take a micro batch from MCTS, check for uniqueness, add to batch lists"""
        for item in micro_batch:
            if not item:
                continue

            #self.total_seen += 1
            # key (hash), board, legal moves
            k, b, l = item
            
            # no dupes allowed
            if k in self.currently_batched_keys:
                #self.duplicate_removed += 1
                continue

            self.currently_batched_keys.add(k)
            self.keys.append(k)
            self.boards.append(np.asarray(b, dtype=np.int32))
            self.legals.append(np.asarray(l, dtype=np.int32))

    def pop_batch(self, max_size=None, always_pad=False, never_pad=False):
        """Pop a batch and return (keys, boards_np, legals_np).

        Chooses the candidate batch size that minimizes waste:
        - padding waste when s >= n: s - n
        - leftover waste when s < n: n - s

        Mode flags:
        - always_pad: prefer s >= n when ties exist
        - never_pad: prefer s <= n when ties exist
        """

        cands = self.batch_candidates
        if max_size is not None:
            cands = [s for s in cands if s <= max_size]

        if not cands:
            return None

        n = len(self.boards)
        if n == 0:
            return None

        best_s = None
        best_waste = None

        for s in cands:
            if s >= n:
                waste = s - n
            else:
                waste = n - s

            if best_waste is None or waste < best_waste:
                best_waste = waste
                best_s = s
                continue

            if waste != best_waste:
                continue

            if always_pad and best_s < n and s >= n:
                best_s = s
                continue

            if never_pad and best_s > n and s <= n:
                best_s = s
                continue

            if not always_pad and not never_pad:
                cur_pads = best_s >= n
                new_pads = s >= n
                if cur_pads and not new_pads:
                    best_s = s

        needed = best_s

        if n >= needed:
            take = needed
            pad = 0
        else:
            take = n
            pad = needed - take

        real_keys = self.keys[:take]
        boards_np = np.stack(self.boards[:take], axis=0)
        legals_np = np.stack(self.legals[:take], axis=0)

        out_keys = list(real_keys)

        if pad:
            out_keys += [KEY_PAD] * pad
            pad_shape = (pad,) + boards_np.shape[1:]
            pad_boards = np.zeros(pad_shape, dtype=boards_np.dtype)
            pad_legals = np.zeros((pad,) + legals_np.shape[1:], dtype=legals_np.dtype)
            boards_np = np.concatenate([boards_np, pad_boards], axis=0)
            legals_np = np.concatenate([legals_np, pad_legals], axis=0)

        self.currently_batched_keys.difference_update(real_keys)
        self.keys = self.keys[take:]
        self.boards = self.boards[take:]
        self.legals = self.legals[take:]

        return (out_keys, boards_np, legals_np)


class TensorFlowThread(object):
    """Async TF predictor + raw_cache writer, with hard pause for safe VRAM teardown."""

    def __init__(self, cfg, infer, max_inflight=None):
        self.infer = infer
        self.cfg = cfg

        if max_inflight is None:
            max_inflight = cfg.max_tf_inflight
        
        self.batch_q = Queue(maxsize=max_inflight)
        self.stop_ev = threading.Event()
        self.t = None
        self.err = None

        self.prediction_times = []
        self.wait_times = []
        self.n_batches = 0
        
        self.preds_lock = threading.Lock()
        self.last_preds_cached = 0
        self.preds_ev = threading.Event()

        self.pause_ev = threading.Event()
        self.paused_ack_ev = threading.Event()
        self.infer_lock = threading.Lock()
        self.held_batch = None

    def start(self):
        if self.t is not None:
            return
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def close(self):
        self.stop_ev.set()

        try:
            self.batch_q.put_nowait(("__poke__", None))
        except Exception:
            pass

        if self.t is not None:
            self.t.join()
        self.t = None

        if self.err is not None:
            raise self.err

    def pause(self, blocking=True):
        if self.stop_ev.is_set():
            return True

        self.pause_ev.set()

        try:
            self.batch_q.put_nowait(("__poke__", None))
        except Exception:
            pass

        if not blocking:
            return True

        while not self.paused_ack_ev.is_set():
            if self.err is not None:
                raise self.err
            if self.stop_ev.is_set():
                return True
            time.sleep(0.002)

        with self.infer_lock:
            pass

        time.sleep(1.0)
        return True

    def unpause(self):
        self.paused_ack_ev.clear()
        self.pause_ev.clear()

        try:
            self.batch_q.put_nowait(("__poke__", None))
        except Exception:
            pass

        return True

    def submit(self, batch, block=True):
        if self.err is not None:
            raise self.err

        if self.stop_ev.is_set() or self.pause_ev.is_set():
            return False

        if not block:
            try:
                self.batch_q.put_nowait(batch)
                return True
            except Exception:
                return False

        self.batch_q.put(batch)
        return True

    def run(self):
        while not self.stop_ev.is_set():
            if self.pause_ev.is_set():
                self.paused_ack_ev.set()
                time.sleep(0.002)
                continue

            try:
                if self.held_batch is not None:
                    batch = self.held_batch
                    self.held_batch = None
                else:
                    t_wait0 = time.time()
                    batch = self.batch_q.get(timeout=0.01)
                    self.wait_times.append(time.time() - t_wait0)

                if not batch:
                    continue

                if isinstance(batch, tuple) and batch[0] == "__poke__":
                    continue

                if self.pause_ev.is_set():
                    self.held_batch = batch
                    self.paused_ack_ev.set()
                    continue

                keys, boards_np, legals_np = batch

                t0 = time.time()
                with self.infer_lock:
                    infer = self.infer

                if infer is None:
                    continue

                probs_np, vals_np = infer((boards_np, legals_np))

                to_raw_cache = []
                for i, k in enumerate(keys):
                    if k == KEY_PAD:
                        continue

                    v = np.asarray(vals_np[i]).reshape(())
                    p = np.asarray(probs_np[i], dtype=np.float32)
                    to_raw_cache.append((k, v, p))

                if to_raw_cache:
                    raw_cache_bulk_insert(to_raw_cache)
                    with self.preds_lock:
                        self.last_preds_cached += 1
                        self.preds_ev.set()

                self.prediction_times.append(time.time() - t0)
                self.n_batches += 1

            except Empty:
                continue

            except Exception as e:
                self.err = e
                self.stop_ev.set()
                return

    def stats(self):
        wt = self.wait_times
        pt = self.prediction_times

        last_wait = wt[-1] if wt else 0.0
        last_pred = pt[-1] if pt else 0.0

        mean_wait = np.mean(wt) if wt else 0.0
        mean_pred = np.mean(pt) if pt else 0.0

        self.wait_times.clear()
        self.prediction_times.clear()

        return {
            "n_batches": self.n_batches,
            "qsize": self.batch_q.qsize(),
            "last_wait_s": last_wait,
            "last_pred_s": last_pred,
            "mean_wait_s": mean_wait,
            "mean_pred_s": mean_pred,
            "paused": self.pause_ev.is_set()
        }



