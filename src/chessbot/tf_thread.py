import time
import threading
from queue import Queue, Empty
from collections import deque

import numpy as np
from pyfastchess import raw_cache_bulk_insert


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
        """ Take a micro batch from MCTS, check for uniqueness and add to batch lists"""
        
        for item in micro_batch:
            self.total_seen += 1
            # key (hash), board, legal moves
            k, b, l = item

            # no dupes allowed
            if k in self.currently_batched_keys:
                self.duplicate_removed += 1
                continue

            self.currently_batched_keys.add(k)
            self.keys.append(k)
            self.boards.append(np.asarray(b, dtype=np.int32))
            self.legals.append(np.asarray(l, dtype=np.int32))

    def pop_batch(self, max_size=None):
        """ pop a batch from lists and return np.arrays ready for preds"""

        cands = self.batch_candidates
        if max_size is not None:
            # respect optional upper bound on batch size
            cands = [s for s in cands if s <= max_size]

        # few saftey checks
        if not cands:
            return None

        n = len(self.boards)
        if n == 0:
            return None

        # choose smallest cand >= n, else clamp to largest cand
        max_c = cands[-1]
        if n >= max_c:
            needed = max_c
        else:
            needed = cands[-1]
            for s in cands:
                if s >= n:
                    needed = s
                    break

        # actual items removed from the queue
        take = min(n, needed)  
        out_keys = self.keys[:take]
        boards_np = np.stack(self.boards[:take], axis=0)
        legals_np = np.stack(self.legals[:take], axis=0)

        pad = needed - take  # pad up to the chosen candidate size
        if pad:
            out_keys += [-9999] * pad  # sentinel keys for padded rows
            pad_shape = (pad,) + boards_np.shape[1:]
            pad_boards = np.zeros(pad_shape, dtype=boards_np.dtype)
            pad_legals = np.zeros((pad,) + legals_np.shape[1:], dtype=legals_np.dtype)
            boards_np = np.concatenate([boards_np, pad_boards], axis=0)
            legals_np = np.concatenate([legals_np, pad_legals], axis=0)

        # remove real keys from the "in batch" set and drop them from queues
        self.currently_batched_keys.difference_update(out_keys)
        self.keys = self.keys[take:]
        self.boards = self.boards[take:]
        self.legals = self.legals[take:]

        return (out_keys, boards_np, legals_np, pad)
        

class TensorFlowThread(object):
    """ Async TF predictor + raw_cache writer. """

    def __init__(self, infer, cfg, max_inflight=2):
        self.infer = infer
        self.cfg = cfg

        self.batch_q = Queue(maxsize=max_inflight)
        self.stop_ev = threading.Event()
        self.t = None
        self.err = None

        self.prediction_times = []
        self.wait_times = []
        self.n_batches = 0

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

    def submit(self, batch, block=True):
        """
        batch is (keys, boards_np, legals_np, pad)
        block=True provides backpressure when max_inflight is hit.
        """
        if self.err is not None:
            raise self.err

        if self.stop_ev.is_set():
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
            try:
                t_wait0 = time.time()
                batch = self.batch_q.get(timeout=0.01)
                self.wait_times.append(time.time() - t_wait0)

                if not batch:
                    continue

                if isinstance(batch, tuple) and batch[0] == "__poke__":
                    continue

                keys, boards_np, legals_np, pad = batch

                t0 = time.time()
                probs_np, vals_np = self.infer((boards_np, legals_np))

                to_raw_cache = []
                for i, k in enumerate(keys):
                    if k == -9999:
                        continue

                    v = np.asarray(vals_np[i]).reshape(()) # scalar
                    p = np.asarray(probs_np[i], dtype=np.float32) # (4288,)
                    to_raw_cache.append((k, v, p))

                if to_raw_cache:
                    raw_cache_bulk_insert(to_raw_cache)

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

        return {
            "n_batches": self.n_batches,
            "n_rows": self.n_rows,
            "qsize": self.batch_q.qsize(),
            "last_wait_s": last_wait,
            "last_pred_s": last_pred,
        }

