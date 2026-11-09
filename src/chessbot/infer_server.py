"""
xc0 inference server: one process owns the TRT sessions for the selfplay
fleet, so the fleet pays for one CUDA context instead of one per worker.

Each worker has a shared-memory slot (its encoded batch in, logits and wdl
out) and a one-way pipe each way. Only (seq, n) and the echoed seq cross the
pipes; a worker drops any reply whose seq is not its own, which is what makes
a slot safe to hand to a respawned worker. Sessions run on their own threads
(ORT releases the GIL), so one session's copies and launch overhead overlap
another's compute.

Control from the runner, as dicts on ctrl:
  {"cmd": "load", "onnx", "model_name", "trt_dir"}   sessions up, ready_ev set
  {"cmd": "release"}   sessions dropped and VRAM freed, release_ev set
  {"cmd": "stop"}
Requests that arrive while released wait in the job queue for the next load.
"""
import gc
import os
import queue
import struct
import sys
import threading
import traceback
from multiprocessing import shared_memory
from multiprocessing.connection import wait

import numpy as np

from chessbot.infer_ort_trt import ENC_SHAPE, enc_dummy, make_trt_session

OUT_NAMES = ["policy_logits", "value_out"]
ORT_DTYPES = {"tensor(float16)": "float16", "tensor(float)": "float32"}
REQ = struct.Struct("<II")
REP = struct.Struct("<I")


def load_sessions(cmd, spec):
    sessions = [make_trt_session(cmd["onnx"], cmd["model_name"], cmd["trt_dir"],
                                 spec["max_bs"], spec["encoding"])
                for _ in range(spec["n_threads"])]
    dummy = enc_dummy(spec["max_bs"], spec["encoding"])
    for sess in sessions:
        sess.run(OUT_NAMES, {"enc_in": dummy})
    return sessions


def make_slots(n_slots, spec, sess):
    """One shared block per array per slot, shaped from the loaded model."""
    outs = {o.name: o for o in sess.get_outputs()}
    max_bs = spec["max_bs"]
    shapes = {
        "enc": ((max_bs, int(ENC_SHAPE[spec["encoding"]])), "int64"),
        "pol": ((max_bs, outs["policy_logits"].shape[1]),
                ORT_DTYPES[outs["policy_logits"].type]),
        "wdl": ((max_bs, 3), ORT_DTYPES[outs["value_out"].type]),
    }
    blocks, slots, descs = [], [], []
    for _ in range(n_slots):
        views, names = {}, {}
        for key, (shape, dt) in shapes.items():
            size = int(np.prod(shape)) * np.dtype(dt).itemsize
            block = shared_memory.SharedMemory(create=True, size=size)
            blocks.append(block)
            names[key] = block.name
            views[key] = np.ndarray(shape, dtype=dt, buffer=block.buf)
        slots.append(views)
        descs.append({"shm": names, "shapes": shapes})
    return blocks, slots, descs


def server_main(spec, first_load, reqs, reps, ctrl, ready_q, ready_ev,
                release_ev):
    """spec: max_bs, encoding, n_threads. first_load: the initial load cmd."""
    jobs = queue.Queue()
    sessions, threads = [], []
    blocks, slots = [], []
    # one job in flight per slot: a respawned worker's first request queues
    # behind its dead predecessor's, so the two never write the slot at once
    lk = threading.Lock()
    busy, waiting = set(), {}

    def submit(i, seq, n):
        with lk:
            if i in busy:
                waiting[i] = (seq, n)
                return
            busy.add(i)
        jobs.put((i, seq, n))

    def done(i):
        with lk:
            nxt = waiting.pop(i, None)
            if nxt is None:
                busy.discard(i)
                return
        jobs.put((i, *nxt))

    def serve(sess):
        try:
            while True:
                job = jobs.get()
                if job is None:
                    return
                i, seq, n = job
                s = slots[i]
                p, v = sess.run(OUT_NAMES, {"enc_in": s["enc"][:n]})
                s["pol"][:n] = p
                s["wdl"][:n] = v
                try:
                    reps[i].send_bytes(REP.pack(seq))
                except OSError:
                    pass
                done(i)
        except Exception:
            # a worker blocked on this reply would wait forever; dying lets
            # the runner restart the server and respawn the fleet
            traceback.print_exc()
            sys.stdout.flush()
            os._exit(1)

    def start(cmd):
        nonlocal sessions, threads
        sessions = load_sessions(cmd, spec)
        threads = [threading.Thread(target=serve, args=(s,), daemon=True)
                   for s in sessions]
        for t in threads:
            t.start()
        release_ev.clear()
        ready_ev.set()
        print(f"[server] ready: {len(sessions)} sessions, "
              f"{os.path.basename(cmd['onnx'])}", flush=True)

    def drop():
        nonlocal sessions, threads
        ready_ev.clear()
        # sentinels queue behind any pending jobs, so those finish first
        for _ in threads:
            jobs.put(None)
        for t in threads:
            t.join()
        sessions, threads = [], []
        gc.collect()
        release_ev.set()
        print("[server] released", flush=True)

    try:
        start(first_load)
        blocks, slots, descs = make_slots(len(reqs), spec, sessions[0])
        mb = sum(b.size for b in blocks) / 2**20
        print(f"[server] {len(reqs)} slots, {mb:.1f} MB shared", flush=True)
        ready_q.put(descs)

        index = {r: i for i, r in enumerate(reqs)}
        live = list(reqs)
        while True:
            for conn in wait(live + [ctrl]):
                if conn is ctrl:
                    cmd = ctrl.recv()
                    if cmd["cmd"] == "stop":
                        return
                    if cmd["cmd"] == "release" and sessions:
                        drop()
                    elif cmd["cmd"] == "load":
                        if sessions:
                            drop()
                        start(cmd)
                    continue
                try:
                    seq, n = REQ.unpack(conn.recv_bytes())
                except (EOFError, OSError):
                    # the runner keeps the worker end for respawns, so this
                    # only happens once the runner itself is gone
                    live.remove(conn)
                    continue
                submit(index[conn], seq, n)
    except Exception:
        traceback.print_exc()
    finally:
        for b in blocks:
            b.close()
            b.unlink()
        sys.stdout.flush()
        os._exit(0)


def make_remote_infer(remote):
    """Drop-in for the local infer: same inputs, same float32 logits and
    softmaxed wdl. Only the session.run moves to the server."""
    shm = {k: shared_memory.SharedMemory(name=n)
           for k, n in remote["shm"].items()}
    views = {k: np.ndarray(shape, dtype=dt, buffer=shm[k].buf)
             for k, (shape, dt) in remote["shapes"].items()}
    enc, pol, wdl = views["enc"], views["pol"], views["wdl"]
    req, rep = remote["req"], remote["rep"]
    # random start, so a fresh worker never collides with its predecessor's seq
    seq = int.from_bytes(os.urandom(4), "little")

    def infer(pair):
        nonlocal seq
        x = pair[0]
        n = len(x)
        enc[:n] = x
        seq = (seq + 1) & 0xFFFFFFFF
        req.send_bytes(REQ.pack(seq, n))
        # a reply meant for a dead predecessor on this slot is skipped
        while REP.unpack(rep.recv_bytes())[0] != seq:
            pass
        logits = pol[:n].astype(np.float32)
        wdl_raw = wdl[:n].astype(np.float32)
        e = np.exp(wdl_raw - wdl_raw.max(axis=-1, keepdims=True))
        return logits, e / e.sum(axis=-1, keepdims=True)

    # the views die with these handles
    infer.shm = shm
    return infer
