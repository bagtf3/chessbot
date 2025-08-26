import time, threading, queue
from concurrent.futures import Future
import tensorflow as tf

class TFBatchedEvaluator:
    def __init__(self, model, encode_state_fn, encode_state_moves_fn=None,
                 max_batch=128, flush_ms=3, jit=True):
        """
        model: Keras model or wrapper exposing:
            - value(states) -> [B] or [B,1]
            - q_values(states, from_ids, to_ids, promo_ids, seg_ids) -> [L_total]
        encode_state_fn(boards) -> tf.Tensor [B,...] on GPU
        encode_state_moves_fn(packs=[(board, moves)]) ->
            (states[B,...], from_ids[L], to_ids[L], promo_ids[L], seg_ids[L])
        """
        self.model = model
        self.encode_state = encode_state_fn
        self.encode_state_moves = encode_state_moves_fn
        self.max_batch = max_batch
        self.flush_ms = flush_ms

        # Queues for jobs
        self.q_val = queue.Queue()
        self.q_qsa = queue.Queue()

        # Optional XLA/JIT wrap for speed
        if jit:
            self._tf_value = tf.function(self._tf_value, jit_compile=True)
            if self.encode_state_moves is not None:
                self._tf_qsa = tf.function(self._tf_qsa, jit_compile=True)

        self._stop = threading.Event()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    # ---- Public APIs (producers) ----
    def eval_state(self, board):
        fut = Future()
        self.q_val.put((board, fut))
        return fut

    def eval_state_moves(self, board, moves):
        assert self.encode_state_moves is not None, "No action-conditioned encoder provided"
        fut = Future()
        self.q_qsa.put((board, moves, fut))
        return fut

    # ---- Worker thread ----
    def stop(self):
        self._stop.set()
        self._t.join(timeout=1)

    def _worker(self):
        last = time.time()
        while not self._stop.is_set():
            jobs_val, jobs_qsa = [], []
            # Gather up to max_batch or until flush_ms elapses
            while (len(jobs_val) + len(jobs_qsa)) < self.max_batch:
                timeout = max(0.0, self.flush_ms/1000.0 - (time.time()-last))
                got = False
                try:
                    jobs_val.append(self.q_val.get(timeout=timeout))
                    got = True
                except queue.Empty:
                    pass
                if (len(jobs_val) + len(jobs_qsa)) >= self.max_batch:
                    break
                try:
                    jobs_qsa.append(self.q_qsa.get_nowait())
                    got = True
                except queue.Empty:
                    pass
                if not got:
                    break

            if not jobs_val and not jobs_qsa:
                time.sleep(self.flush_ms/1000.0)
                last = time.time()
                continue

            if jobs_val:
                boards, futs = zip(*jobs_val)
                states = self.encode_state(list(boards))            # tf.Tensor on GPU
                vals = self._tf_value(states)                       # [B]
                vals = vals.numpy().tolist()
                for v, fut in zip(vals, futs):
                    if not fut.set_running_or_notify_cancel(): continue
                    fut.set_result(float(v))

            if jobs_qsa:
                packs = [(b, list(mv)) for (b, mv, _) in jobs_qsa]
                states, from_ids, to_ids, promo_ids, seg_ids = self.encode_state_moves(packs)
                q_flat = self._tf_qsa(states, from_ids, to_ids, promo_ids, seg_ids)  # [L_total]
                q_flat = q_flat.numpy()
                # split back per request using seg_ids counts
                import numpy as np
                seg = seg_ids.numpy()
                counts = np.bincount(seg)
                cursor = 0
                for idx, (_, _, fut) in enumerate(jobs_qsa):
                    L = counts[idx] if idx < len(counts) else 0
                    q_vec = q_flat[cursor:cursor+L]
                    cursor += L
                    if not fut.set_running_or_notify_cancel(): continue
                    fut.set_result(q_vec)

            last = time.time()

    # ---- TF-wrapped inference ----
    @tf.function(jit_compile=True)
    def _tf_value(self, states):
        v = self.model.value(states)             # [B] or [B,1]
        if len(v.shape) == 2 and v.shape[-1] == 1:
            v = tf.squeeze(v, axis=-1)
        return v

    @tf.function(jit_compile=True)
    def _tf_qsa(self, states, from_ids, to_ids, promo_ids, seg_ids):
        return self.model.q_values(states, from_ids, to_ids, promo_ids, seg_ids)  # [L_total]


class MyChessModel(tf.keras.Model):
    def __init__(self, ...):
        super().__init__()
        # build your encoder/transformer here…

    @tf.function(jit_compile=True)
    def value(self, states):
        """
        states: [B, ...] (e.g., [B, 73, 8, 8] or [B, 64, d_model])
        returns: [B] or [B,1], value in [-1,1] (WHITE POV or node POV—just be consistent)
        """
        # ... encode -> pooled -> dense(1, activation='tanh')
        return v

    @tf.function(jit_compile=True)
    def q_values(self, states, from_ids, to_ids, promo_ids, seg_ids):
        """
        states: [B,...]
        from_ids/to_ids/promo_ids: [L_total] int32
        seg_ids: [L_total] int32, mapping each move to its source state index in 0..B-1
        returns: [L_total] float32, Q(s,a) or Δeval(s,a)
        """
        # Example pattern:
        # 1) Encode states once: z_global [B,d], per-square H [B,64,d]
        # 2) Gather H[from], H[to] via tf.gather with batch dims using seg_ids
        # 3) Build move embeddings m = E_from+E_to+E_promo + H_from + H_to
        # 4) Score with bilinear or small MLP: q = (W z)[seg_ids] · m  or MLP([z||m])
        return q_flat  # [L_total]
    

def encode_state_fn(boards):
    """
    boards: list of python-chess Board
    returns: tf.Tensor on GPU, e.g. [B, 73, 8, 8] float32
    """
    planes = build_planes_batch_np(boards)   # or build directly with tf.numpy_function
    x = tf.convert_to_tensor(planes, dtype=tf.float32)
    return tf.identity(x)  # lands on GPU under default device

def encode_state_moves_fn(packs):
    """
    packs: list of (board, moves_list)
    returns: (states[B,...], from_ids[L], to_ids[L], promo_ids[L], seg_ids[L]) as tf.Tensors (int32 for ids)
    """
    boards, moves_lists = zip(*packs)
    states = encode_state_fn(list(boards))
    froms, tos, promos, segs = [], [], [], []
    for i, (b, mlist) in enumerate(zip(boards, moves_lists)):
        for m in mlist:
            froms.append(m.from_square)
            tos.append(m.to_square)
            promos.append(m.promotion or 0)
            segs.append(i)
    from_ids  = tf.convert_to_tensor(froms,  dtype=tf.int32)
    to_ids    = tf.convert_to_tensor(tos,    dtype=tf.int32)
    promo_ids = tf.convert_to_tensor(promos, dtype=tf.int32)
    seg_ids   = tf.convert_to_tensor(segs,   dtype=tf.int32)
    return states, from_ids, to_ids, promo_ids, seg_ids


be = TFBatchedEvaluator(model=my_model,
                        encode_state_fn=encode_state_fn,
                        encode_state_moves_fn=encode_state_moves_fn,
                        max_batch=128, flush_ms=3, jit=True)

pending = []

def run_simulation(root):
    path, leaf = select_to_leaf(root)   # your selection
    fut = be.eval_state(leaf.board)     # enqueue
    pending.append((leaf, path, fut))

    # Periodically drain completed futures and backup
    drain_completed()

def drain_completed():
    global pending
    still = []
    for leaf, path, fut in pending:
        if fut.done():
            v = fut.result()            # value in node POV (or flip here)
            backup(path, v)             # your negate-per-ply backup
        else:
            still.append((leaf, path, fut))
    pending = still

