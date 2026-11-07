"""
Inference-server proof of concept on real selfplay.
Usage: python infer_server_poc.py <run_tag> --mode local|server
           [--workers N] [--games-at-once N] [--macro-batch N]
           [--server-threads N] [--warmup MIN] [--minutes MIN]

Both modes play the run's real selfplay: GameGenerator specs from its
config.yaml, child_looper-style GameLooper workers (MCTS, caches, encoding,
SF for vs_stockfish games on worker 0) and fat game pkls. They differ only in
where inference runs:
  local   every worker builds its own TRT session, as selfplay does
  server  one process owns the TRT sessions (--server-threads of them, one
          CUDA context); each worker writes its batch into its own
          shared-memory slot and signals over a pipe

The looper is untouched: each worker patches the make_ort_trt_infer the looper
imports. Nothing is written to the run dir: the ONNX and its TRT cache are
copied to POC_DIR and games land there. No rescore, no retrain. Stop the
selfplay run first; both modes need the GPU and SF CPU to themselves.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing import shared_memory
from multiprocessing.connection import wait

import numpy as np

from chessbot import SP_DIR
from chessbot.game_utils import GameGenerator
from chessbot.infer_ort_trt import (
    ENC_SHAPE, enc_dummy, make_trt_session, selfplay_trt_paths,
)
from chessbot.selfplay_runner import (
    GAME_QUEUE_TARGET, SF_QUEUE_TARGET, drain_queue, parse_paths,
)

POC_DIR = os.path.join(os.path.dirname(SP_DIR), "infer_poc")
STATS_EVERY_S = 10.0
PRINT_EVERY_S = 60.0
SERVER_READY_TIMEOUT_S = 900
OUT_NAMES = ["policy_logits", "value_out"]
ORT_DTYPES = {"tensor(float16)": "float16", "tensor(float)": "float32"}


def onnx_prefix(onnx_path, model_name):
    """Same content-hash prefix make_trt_session uses for its engine cache."""
    h = hashlib.sha256()
    with open(onnx_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return f"{model_name}_{h.hexdigest()[:12]}"


def copy_trt_cache(cfg, dst_dir):
    """The run's ONNX plus its current engine, profile and timing files."""
    src_dir, model_name, onnx_path = selfplay_trt_paths(cfg)
    os.makedirs(dst_dir, exist_ok=True)
    prefix = onnx_prefix(onnx_path, model_name)
    for f in os.listdir(src_dir):
        if not (f == os.path.basename(onnx_path) or f.startswith(prefix)
                or f.endswith(".timing")):
            continue
        src, dst = os.path.join(src_dir, f), os.path.join(dst_dir, f)
        if (not os.path.exists(dst)
                or os.path.getmtime(src) > os.path.getmtime(dst)):
            shutil.copy2(src, dst)
    return os.path.join(dst_dir, os.path.basename(onnx_path)), model_name


def warm_trt(onnx_path, model_name, trt_dir, max_bs, encoding):
    """Load (or compile, once) the engine for max_bs so workers never race
    each other to build it."""
    sess = make_trt_session(onnx_path, model_name, trt_dir, max_bs, encoding)
    sess.run(OUT_NAMES, {"enc_in": enc_dummy(max_bs, encoding)})


class Meter:
    """Cumulative per-worker counters, pushed to the parent every
    STATS_EVERY_S. The parent diffs two snapshots for a window."""

    def __init__(self, src, stats_q):
        self.src, self.q = src, stats_q
        self.c = {"moves": 0, "leaves": 0, "calls": 0, "positions": 0,
                  "infer_s": 0.0}
        self.last = time.monotonic()

    def wrap(self, infer):
        def timed(pair):
            t0 = time.perf_counter()
            out = infer(pair)
            self.c["infer_s"] += time.perf_counter() - t0
            self.c["calls"] += 1
            self.c["positions"] += len(pair[0])
            self.maybe_push()
            return out
        return timed

    def hook(self, looper):
        for meter, key in ((looper.mps, "moves"), (looper.lps, "leaves")):
            orig = meter.tick

            def tick(n=1, orig=orig, key=key):
                self.c[key] += n
                return orig(n)
            meter.tick = tick

    def maybe_push(self, force=False):
        now = time.monotonic()
        if force or now - self.last >= STATS_EVERY_S:
            self.q.put({"src": self.src, "pid": os.getpid(), "t": time.time(),
                        **self.c})
            self.last = now


def make_remote_infer(remote):
    """Drop-in for the local infer: same inputs, same float32 logits and
    softmaxed wdl. Only the session.run moves to the server."""
    shm = {k: shared_memory.SharedMemory(name=n)
           for k, n in remote["shm"].items()}
    enc = np.ndarray(remote["enc_shape"], dtype=np.int64, buffer=shm["enc"].buf)
    pol = np.ndarray(remote["pol_shape"], dtype=remote["pol_dtype"],
                     buffer=shm["pol"].buf)
    wdl = np.ndarray(remote["wdl_shape"], dtype=remote["wdl_dtype"],
                     buffer=shm["wdl"].buf)
    req, rep = remote["req"], remote["rep"]

    def infer(pair):
        x = pair[0]
        n = len(x)
        enc[:n] = x
        req.send_bytes(n.to_bytes(4, "little"))
        rep.recv_bytes()
        logits = pol[:n].astype(np.float32)
        wdl_raw = wdl[:n].astype(np.float32)
        e = np.exp(wdl_raw - wdl_raw.max(axis=-1, keepdims=True))
        return logits, e / e.sum(axis=-1, keepdims=True)

    # the views die with these handles
    infer.shm = shm
    return infer


def poc_worker(cfg, stop_ev, recent_q, telem_q, msg_q, game_q, sf_q,
               stats_q, remote):
    """child_looper with the infer swapped (server) or only timed (local)."""
    import chessbot.looper as lp

    meter = Meter(cfg.id, stats_q)
    real = lp.make_ort_trt_infer

    def patched(*args, **kwargs):
        if remote is None:
            sess, infer = real(*args, **kwargs)
        else:
            sess, infer = None, make_remote_infer(remote)
        return sess, meter.wrap(infer)

    lp.make_ort_trt_infer = patched
    with lp.init_selfplay(cfg, recent_q, telem_q, msg_q,
                          game_q=game_q, sf_q=sf_q) as looper:
        meter.hook(looper)
        looper.run(stop_ev)
    meter.maybe_push(force=True)
    # flush the final stats, then hard-exit: ORT/TRT teardown and unread
    # queue feeders can hang a normal interpreter exit
    stats_q.close()
    stats_q.join_thread()
    sys.stdout.flush()
    os._exit(0)


def server_main(onnx_path, model_name, trt_dir, max_bs, encoding, reqs, reps,
                ready_q, stats_q, stop_ev, n_threads):
    """n_threads TRT sessions in one CUDA context, each on its own stream, so
    one session's copies and launch overhead overlap another's compute. The
    main thread reads (n) off the request pipes and queues the job; a
    session thread runs it and answers on that worker's reply pipe. The
    batch itself is already in the worker's slot."""
    sessions = [make_trt_session(onnx_path, model_name, trt_dir, max_bs,
                                 encoding) for _ in range(n_threads)]
    for sess in sessions:
        sess.run(OUT_NAMES, {"enc_in": enc_dummy(max_bs, encoding)})
    outs = {o.name: o for o in sessions[0].get_outputs()}
    p_dim = outs["policy_logits"].shape[1]
    pol_dtype = ORT_DTYPES[outs["policy_logits"].type]
    wdl_dtype = ORT_DTYPES[outs["value_out"].type]
    enc_shape = (max_bs, int(ENC_SHAPE[encoding]))
    pol_shape, wdl_shape = (max_bs, p_dim), (max_bs, 3)

    blocks, slots, descs = [], [], []
    for _ in reqs:
        views, names = [], {}
        for key, shape, dt in (("enc", enc_shape, "int64"),
                               ("pol", pol_shape, pol_dtype),
                               ("wdl", wdl_shape, wdl_dtype)):
            size = int(np.prod(shape)) * np.dtype(dt).itemsize
            block = shared_memory.SharedMemory(create=True, size=size)
            blocks.append(block)
            names[key] = block.name
            views.append(np.ndarray(shape, dtype=dt, buffer=block.buf))
        slots.append(views)
        descs.append({"shm": names, "enc_shape": enc_shape,
                      "pol_shape": pol_shape, "pol_dtype": pol_dtype,
                      "wdl_shape": wdl_shape, "wdl_dtype": wdl_dtype})
    mb = sum(b.size for b in blocks) / 2**20
    print(f"[server] ready: {len(reqs)} slots, {mb:.1f} MB shared, "
          f"{n_threads} sessions, policy {p_dim} {pol_dtype}", flush=True)
    ready_q.put(descs)

    c = {"calls": 0, "positions": 0, "run_s": 0.0, "copy_s": 0.0,
         "threads": n_threads}
    lock = threading.Lock()
    jobs = queue.Queue()

    def serve(sess):
        while True:
            job = jobs.get()
            if job is None:
                return
            i, n = job
            enc, pol, wdl = slots[i]
            t0 = time.perf_counter()
            p, v = sess.run(OUT_NAMES, {"enc_in": enc[:n]})
            t1 = time.perf_counter()
            pol[:n] = p
            wdl[:n] = v
            t2 = time.perf_counter()
            with lock:
                c["run_s"] += t1 - t0
                c["copy_s"] += t2 - t1
                c["calls"] += 1
                c["positions"] += n
            try:
                reps[i].send_bytes(b"\x01")
            except OSError:
                pass

    threads = [threading.Thread(target=serve, args=(s,), daemon=True)
               for s in sessions]
    for t in threads:
        t.start()

    def push_stats():
        with lock:
            snap = dict(c)
        stats_q.put({"src": "server", "pid": os.getpid(), "t": time.time(),
                     **snap})

    index = {r: i for i, r in enumerate(reqs)}
    live = list(reqs)
    last_push = time.monotonic()
    try:
        while live and not stop_ev.is_set():
            for r in wait(live, timeout=1.0):
                try:
                    n = int.from_bytes(r.recv_bytes(), "little")
                except (EOFError, OSError):
                    live.remove(r)
                    continue
                jobs.put((index[r], n))
            now = time.monotonic()
            if now - last_push >= STATS_EVERY_S:
                push_stats()
                last_push = now
    finally:
        for _ in threads:
            jobs.put(None)
        for t in threads:
            t.join(timeout=10)
        push_stats()
        for b in blocks:
            b.close()
            b.unlink()
        stats_q.close()
        stats_q.join_thread()
        sys.stdout.flush()
        os._exit(0)


class GpuSampler(threading.Thread):
    """nvidia-smi once a second: (t, util %, MiB used)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.rows = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-lms", "1000"],
            stdout=subprocess.PIPE, text=True)

    def run(self):
        for line in self.proc.stdout:
            try:
                util, mem = (float(s) for s in line.split(","))
            except ValueError:
                continue
            self.rows.append((time.time(), util, mem))

    def window(self, t0, t1):
        return [(u, m) for t, u, m in self.rows if t0 <= t < t1]

    def stop(self):
        self.proc.terminate()


def ram_by_pid(pids):
    """(rss, private) MB per pid; private is the commit charge on Windows."""
    import psutil
    out = {}
    for name, pid in pids.items():
        try:
            mi = psutil.Process(pid).memory_info()
            out[name] = (mi.rss / 2**20, getattr(mi, "private", mi.vms) / 2**20)
        except psutil.Error:
            pass
    return out


def diff(a, b, key):
    return b.get(key, 0) - a.get(key, 0)


def window_summary(start, end, dt, n_workers, gpu, ram, games):
    w = [s for s in end if s != "server"]
    tot = {k: sum(diff(start.get(s, {}), end[s], k) for s in w)
           for k in ("moves", "leaves", "calls", "positions", "infer_s")}
    out = {
        "window_s": dt,
        "games_finished": games,
        "moves_per_s": tot["moves"] / dt,
        "leaves_per_s": tot["leaves"] / dt,
        "nn_pos_per_s": tot["positions"] / dt,
        "calls_per_s": tot["calls"] / dt,
        "pos_per_call": tot["positions"] / max(1, tot["calls"]),
        "worker_infer_frac": tot["infer_s"] / (dt * max(1, n_workers)),
        "worker_ms_per_call": 1000 * tot["infer_s"] / max(1, tot["calls"]),
    }
    if "server" in end:
        a, b = start.get("server", {}), end["server"]
        calls = max(1, diff(a, b, "calls"))
        # per session thread, so 1.0 means every session always running
        out["server_busy_frac"] = diff(a, b, "run_s") / (dt * b.get("threads", 1))
        out["server_run_ms_per_call"] = 1000 * diff(a, b, "run_s") / calls
        out["server_copy_ms_per_call"] = 1000 * diff(a, b, "copy_s") / calls
    if gpu:
        u = np.array([g[0] for g in gpu])
        m = np.array([g[1] for g in gpu])
        out.update({"gpu_util_mean": float(u.mean()),
                    "gpu_util_p10": float(np.percentile(u, 10)),
                    "gpu_util_ge95_frac": float((u >= 95).mean()),
                    "vram_mib_max": float(m.max())})
    out["ram_mb"] = {k: {"rss": round(r), "private": round(p)}
                     for k, (r, p) in ram.items()}
    out["ram_private_total_mb"] = round(sum(p for _, p in ram.values()))
    return out


def print_tick(elapsed_min, phase, s):
    """One compact block per PRINT_EVERY_S, same fields as the summary."""
    line = (f"[poc] {elapsed_min:5.1f} min {phase:<7} | games {s['games_finished']}"
            f" | mps {s['moves_per_s']:.1f} | lps {s['leaves_per_s']:,.0f}"
            f" | nn/s {s['nn_pos_per_s']:,.0f} | pos/call {s['pos_per_call']:.0f}"
            f" | wait {s['worker_ms_per_call']:.1f} ms"
            f" ({100 * s['worker_infer_frac']:.0f}% of worker time)")
    print(line)
    extra = []
    if "server_busy_frac" in s:
        extra.append(f"server busy {100 * s['server_busy_frac']:.0f}%"
                     f" run {s['server_run_ms_per_call']:.1f} ms"
                     f" copy {s['server_copy_ms_per_call']:.2f} ms")
    if "gpu_util_mean" in s:
        extra.append(f"gpu {s['gpu_util_mean']:.0f}% mean,"
                     f" {100 * s['gpu_util_ge95_frac']:.0f}% of secs >=95,"
                     f" vram {s['vram_mib_max']:,.0f} MiB")
    if s["ram_private_total_mb"]:
        extra.append(f"ram private {s['ram_private_total_mb']:,} MB")
    if extra:
        print("[poc]                 | " + " | ".join(extra))
    sys.stdout.flush()


def print_summary(mode, s):
    print(f"\n[poc] ===== {mode} | {s['window_s'] / 60:.1f} min window =====")
    for k, v in s.items():
        if k == "ram_mb":
            for name, r in v.items():
                print(f"[poc]   ram {name:<8} rss {r['rss']:>6} MB  "
                      f"private {r['private']:>6} MB")
        elif isinstance(v, float):
            print(f"[poc] {k:<26} {v:,.3f}")
        else:
            print(f"[poc] {k:<26} {v}")
    sys.stdout.flush()


def main(a):
    cfg, _, _ = parse_paths(a.run_tag)
    if cfg.encoding_type not in ("xc0", "xc0h"):
        raise RuntimeError(f"poc covers xc0/xc0h only, got {cfg.encoding_type}")
    cfg.n_workers = a.workers or cfg.n_workers
    cfg.games_at_once = a.games_at_once or cfg.games_at_once
    cfg.macro_batch = a.macro_batch or cfg.macro_batch

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        POC_DIR, f"{a.mode}_w{cfg.n_workers}_g{cfg.games_at_once}"
                 f"_mb{cfg.macro_batch}_{stamp}")
    trt_dir = os.path.join(POC_DIR, "trt_cache")
    onnx_path, model_name = copy_trt_cache(cfg, trt_dir)
    cfg.inference_backend = "ort_trt"
    cfg.model_path = onnx_path
    cfg.trt_model_name = model_name
    cfg.trt_cache = trt_dir
    cfg.game_dir = os.path.join(out_dir, "game_logs")
    os.makedirs(cfg.game_dir, exist_ok=True)

    print(f"[poc] {a.mode}: {cfg.n_workers} workers x {cfg.games_at_once} "
          f"games, macro_batch {cfg.macro_batch}, warmup {a.warmup} min, "
          f"window {a.minutes} min -> {out_dir}", flush=True)

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=warm_trt, args=(onnx_path, model_name, trt_dir,
                                           cfg.macro_batch, cfg.encoding_type))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"TRT warm failed, exitcode={p.exitcode}")

    gen = GameGenerator(cfg)
    game_q, sf_q = ctx.Queue(), ctx.Queue()
    recent_q, telem_q, stats_q = ctx.Queue(), ctx.Queue(), ctx.Queue()

    def top_up(target):
        while game_q.qsize() < target:
            spec = gen.next_game()
            while spec.meta.get("vs_stockfish") and sf_q.qsize() >= SF_QUEUE_TARGET:
                spec = gen.next_game()
            (sf_q if spec.meta.get("vs_stockfish") else game_q).put(spec)

    top_up(cfg.n_workers * cfg.games_at_once + GAME_QUEUE_TARGET)

    server, server_stop, remotes = None, None, [None] * cfg.n_workers
    if a.mode == "server":
        # one-way pipes: (reader, writer) for requests and for replies
        req_pipes = [ctx.Pipe(duplex=False) for _ in range(cfg.n_workers)]
        rep_pipes = [ctx.Pipe(duplex=False) for _ in range(cfg.n_workers)]
        ready_q, server_stop = ctx.Queue(), ctx.Event()
        server = ctx.Process(
            target=server_main,
            args=(onnx_path, model_name, trt_dir, cfg.macro_batch,
                  cfg.encoding_type, [r for r, _ in req_pipes],
                  [w for _, w in rep_pipes], ready_q, stats_q, server_stop,
                  a.server_threads),
            daemon=True)
        server.start()
        descs = ready_q.get(timeout=SERVER_READY_TIMEOUT_S)
        remotes = [{**d, "req": rq[1], "rep": rp[0]}
                   for d, rq, rp in zip(descs, req_pipes, rep_pipes)]

    workers = []
    for i in range(cfg.n_workers):
        c = cfg.copy()
        c.sampleable = None
        c.id = f"w{i}"
        stop_ev = ctx.Event()
        w = ctx.Process(
            target=poc_worker,
            args=(c, stop_ev, recent_q, telem_q, ctx.Queue(), game_q,
                  sf_q if i == 0 else None, stats_q, remotes[i]),
            daemon=True)
        w.start()
        workers.append((w, stop_ev))

    gpu = GpuSampler()
    gpu.start()
    latest, pids = {}, {"main": os.getpid()}
    if server is not None:
        pids["server"] = server.pid
    t_start = time.time()
    t_measure = t_start + a.warmup * 60
    t_end = t_measure + a.minutes * 60
    snap0, games, games0, ram = None, 0, 0, {}
    last_print, tick_snap, tick_games = time.time(), {}, 0
    try:
        while time.time() < t_end:
            top_up(GAME_QUEUE_TARGET)
            drain_queue(telem_q)
            games += len(drain_queue(recent_q))
            for msg in drain_queue(stats_q):
                latest[msg["src"]] = msg
                if msg["src"] != "server":
                    pids[msg["src"]] = msg["pid"]
            now = time.time()
            if snap0 is None and now >= t_measure:
                snap0 = {k: dict(v) for k, v in latest.items()}
                games0 = games
                print("[poc] warmup done, measuring", flush=True)
            if snap0 is not None and not ram and now >= t_measure + a.minutes * 30:
                ram = ram_by_pid(pids)
            if now - last_print >= PRINT_EVERY_S and latest:
                s = window_summary(
                    tick_snap, latest, now - last_print, cfg.n_workers,
                    gpu.window(last_print, now), ram_by_pid(pids),
                    games - tick_games)
                phase = "warmup" if snap0 is None else "measure"
                print_tick((now - t_start) / 60, phase, s)
                last_print, tick_games = now, games
                tick_snap = {k: dict(v) for k, v in latest.items()}
            if not any(w.is_alive() for w, _ in workers):
                print("[poc] all workers exited", flush=True)
                break
            time.sleep(1.0)

        for msg in drain_queue(stats_q):
            latest[msg["src"]] = msg
        if snap0 is not None:
            dt = time.time() - t_measure
            s = window_summary(snap0, latest, dt, cfg.n_workers,
                               gpu.window(t_measure, time.time()), ram,
                               games - games0)
            s.update({"mode": a.mode, "workers": cfg.n_workers,
                      "server_threads": a.server_threads if server else None,
                      "games_at_once": cfg.games_at_once,
                      "macro_batch": cfg.macro_batch})
            print_summary(a.mode, s)
            with open(os.path.join(out_dir, "summary.json"), "w") as f:
                json.dump(s, f, indent=2)
    except BaseException:
        # the hard exit below would otherwise swallow it
        traceback.print_exc()
    finally:
        gpu.stop()
        for _, stop_ev in workers:
            stop_ev.set()
        for w, _ in workers:
            w.join(timeout=60)
            if w.is_alive():
                w.terminate()
        if server is not None:
            server_stop.set()
            server.join(timeout=30)
            if server.is_alive():
                server.terminate()
        # unread specs on game_q would block the queue feeder at exit forever
        for q in (game_q, sf_q, recent_q, telem_q, stats_q):
            q.cancel_join_thread()
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tag")
    ap.add_argument("--mode", choices=("local", "server"), required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--games-at-once", type=int, default=0)
    ap.add_argument("--macro-batch", type=int, default=0)
    ap.add_argument("--server-threads", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--minutes", type=float, default=15.0)
    sys.exit(main(ap.parse_args()))
