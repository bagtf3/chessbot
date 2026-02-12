import time
import threading
from queue import Queue, Empty
from collections import deque

import numpy as np
from pyfastchess import raw_cache_bulk_insert


_now = time.time


class ForceRequest(object):
    def __init__(self, blocking):
        self.blocking = blocking
        self.ev = threading.Event()
        self.err = None


class TensorFlowThread(object):
    """
    Async TF predictor + raw_cache writer.

    Enqueue (zobrist, board_np, legal_np) items.
    Worker de-dupes by zobrist, batches when enough uniques, runs infer(),
    and writes (zobrist, value, policy) rows via raw_cache_bulk_insert().

    force_predict(blocking=True/False) triggers a predict of whatever is
    currently buffered. If blocking=True, waits for that forced prediction
    to complete.
    """

    def __init__(self, infer, cfg):
        self.infer = infer
        self.cfg = cfg

        self.req_q = Queue()
        self.stop_ev = threading.Event()
        self.t = None
        self.err = None

        self.lock = threading.Lock()

        self.target_unique = self.cfg.fwd_batch
        self.fwd_batch = self.cfg.fwd_batch

        self.buf = {}
        self.buf_order = deque()
        self.force_reqs = deque()

        self.batch_candidates = self.create_batch_candidates()

    def create_batch_candidates(self):
        cfg = self.cfg

        batch_candidates = set()
        batch_candidates.add(cfg.fwd_batch)

        bs = 8
        while bs <= cfg.fwd_batch:
            batch_candidates.add(bs)
            bs *= 2

        if len(batch_candidates) >= 2 and cfg.fwd_batch >= 128:
            sbc = sorted(batch_candidates)
            lo = sbc[-2]
            hi = sbc[-1]
            mid = lo + (hi - lo) // 2
            sbc.append(mid)
            return sorted(set(sbc))

        return sorted(set(batch_candidates))

    def start(self):
        if self.t is not None:
            return
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def close(self):
        self.stop_ev.set()
        self.req_q.put(("__poke__", None))

        if self.t is not None:
            self.t.join()
        self.t = None

        if self.err is not None:
            raise self.err

    def submit_many(self, items):
        """
        items: iterable of (zobrist, board_np, legal_np)
        Non-blocking enqueue.
        """
        if not items:
            return
        self.req_q.put(("items", items))

    def set_target_unique(self, n_unique):
        """
        Update batch threshold.

        Requirement: if lowered below current buffered uniques, worker should
        run without needing any new items submitted. We satisfy that by poking
        the worker via the queue.
        """
        with self.lock:
            self.target_unique = n_unique
        self.req_q.put(("__poke__", None))

    def force_predict(self, blocking=True):
        """
        Force a prediction of whatever is currently buffered.

        blocking=False returns immediately.
        blocking=True waits until that forced prediction completes.
        """
        fr = ForceRequest(blocking=blocking)

        with self.lock:
            self.force_reqs.append(fr)

        self.req_q.put(("__poke__", None))

        if not blocking:
            return True

        fr.ev.wait()
        if fr.err is not None:
            raise fr.err
        return True

    def run(self):
        try:
            while not self.stop_ev.is_set():
                self.drain_messages()

                batch_items, should_force = self.try_pop_batch()
                if batch_items:
                    self.predict_and_insert(batch_items)

                if should_force:
                    self.finish_force_reqs_ok()

                if not batch_items and not should_force:
                    try:
                        self.req_q.get(timeout=0.001)
                    except Empty:
                        pass

        except Exception as e:
            self.err = e
            self.finish_force_reqs_err(e)
            self.stop_ev.set()

    def drain_messages(self):
        while True:
            try:
                msg, payload = self.req_q.get_nowait()
            except Empty:
                break

            if msg != "items":
                continue

            for item in payload:
                k = item[0]
                with self.lock:
                    if k in self.buf:
                        continue
                    self.buf[k] = item
                    self.buf_order.append(k)

    def try_pop_batch(self):
        with self.lock:
            should_force = bool(self.force_reqs)
            have = len(self.buf)
            need = self.target_unique

            if not should_force and have < need:
                return [], False

            if should_force:
                take = have
            else:
                take = min(have, need)

            if not take:
                return [], should_force

            out = []
            for _ in range(take):
                k = self.buf_order.popleft()
                item = self.buf.pop(k, None)
                if item is not None:
                    out.append(item)

            return out, should_force

    def finish_force_reqs_ok(self):
        with self.lock:
            while self.force_reqs:
                fr = self.force_reqs.popleft()
                fr.err = None
                fr.ev.set()

    def finish_force_reqs_err(self, err):
        with self.lock:
            while self.force_reqs:
                fr = self.force_reqs.popleft()
                fr.err = err
                fr.ev.set()

    def predict_and_insert(self, preds_batch):
        boards = []
        legals = []
        keys = []

        for item in preds_batch:
            keys.append(item[0])
            boards.append(np.asarray(item[1], dtype=np.int32))
            legals.append(np.asarray(item[2], dtype=np.int32))

        boards_np = np.stack(boards, axis=0)
        legals_np = np.stack(legals, axis=0)

        target_bs = self.fwd_batch
        B = boards_np.shape[0]

        if B < target_bs:
            new_target = None
            for bs in self.batch_candidates:
                if bs >= B:
                    new_target = bs
                    break

            if new_target is None:
                new_target = target_bs

            pad = max(0, new_target - B)
            if pad:
                pad_boards = np.zeros((pad,) + boards_np.shape[1:], dtype=np.int32)
                pad_legals = np.zeros((pad, legals_np.shape[1]), dtype=np.int32)

                boards_np_p = np.concatenate([boards_np, pad_boards], axis=0)
                legals_np_p = np.concatenate([legals_np, pad_legals], axis=0)

                probs_np_p, vals_np_p = self.infer((boards_np_p, legals_np_p))
                probs_np = probs_np_p[:B]
                vals_np = vals_np_p[:B]
            else:
                probs_np, vals_np = self.infer((boards_np, legals_np))
        else:
            probs_np, vals_np = self.infer((boards_np, legals_np))

        to_raw_cache = []
        for i, k in enumerate(keys):
            v = np.asarray(vals_np[i]).reshape(())
            p = np.asarray(probs_np[i], dtype=np.float32)
            to_raw_cache.append((k, v, p))

        raw_cache_bulk_insert(to_raw_cache)


    def submit(self, mini_batch):
        self.req_q.put(mini_batch)

    def enqueue(self, mini_batch):
        if not mini_batch:
            return
        
        for mb in mini_batch:
            key, board_np, legal_np = mb
            if key in self.unique_keys:
                continue

            self.unique_keys.add(key)
            self.keys.append(key)
            self.boards.append(np.asarray(board_np, dtype=np.int32))
            self.legals.append(np.asarray(legal_np, dtype=np.int32))

    def force_predict(self, blocking=False):
        self.force_predict_flag = True

        if not blocking:
            return True

        while self.force_predict_flag:
            if self.err is not None:
                raise self.err
            if self.stop_ev.is_set():
                break
            time.sleep(0.001)
        
        return True

    def run(self):
        while not self.stop_ev.is_set():
            try:
                try:
                    mb = self.req_q.get(timeout=0.01)
                except Empty:
                    mb = None
                
                if mb is not None:
                    self.enqueue(mb)
                
                if self.force_predict_flag:
                    self.predict_and_push()
                    self.force_predict_flag = False
                    continue

                if len(self.keys) >= self.unique_key_trigger:
                    self.predict_and_push()

            except Exception as e:
                self.err = e
                self.res_q.put(("__error__", e))
                self.force_predict_flag = False
                self.stop_ev.set()
                return