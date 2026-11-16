"""Search-controller trainer: fits the K5L6 controller (search_controller_model.md)
to the trajectories of one or more collections (DEFAULT_COLLECTIONS when none
are named). Val and test are drawn at startup from whole held-out shards;
training streams the rest through an async shuffle pool.

python -m chessbot.experimentation.train_controller [collection dir or name ...]
"""
import argparse
import atexit
import csv
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chessbot.experimentation.collector import (
    BOARD_FEATURE_NAMES, CHECK_IN_FEATURE_NAMES, COLLECTIONS_DIR, FEATURE_VERSION,
    NN_FEATURE_NAMES, SCHEMA_VERSION, iter_records, iter_shard, shard_paths)
from chessbot.experimentation.runner import sf_value
from chessbot.utils import compact_count

WIDTH = 128
CONTEXT_DIM, CHECKIN_DIM = 48, 64
HEADS = ["stability", "quality"]
D6 = [1, 1, 2, 2, 4, 4]
REF_LR, REF_BATCH = 1e-3, 32
# padded batches queued ahead of the GPU
PREFETCH = 8
# fixed val and test sets, drawn at startup from whole shards set aside;
# each check scores a fresh random VAL_SAMPLE of the fixed val set
VAL_SIZE, VAL_BATCH = 20480, 1024
VAL_SAMPLE = 10240
# training shuffle pool: topped up to POOL_HIGH trajectories whenever it
# drains below POOL_LOW; LOAD_PROCS processes read shards
POOL_HIGH, POOL_LOW = 80_000, 60_000
# corpus fields carried into the predictions parquet for looking cases up
META_POS_FIELDS = ["fen", "game_id", "abs_ply", "run_tag", "best_move", "best_cp",
                   "played_move", "played_cp", "cpl_depth", "parentQ_white", "parent_p_draw"]
CONTEXT_NAMES = list(NN_FEATURE_NAMES) + list(BOARD_FEATURE_NAMES)
PROGRESS_COLUMNS = ["step", "epoch", "train_bce", "val_bce", "val_stab_bce", "val_qual_bce",
                    "val_stab_auc", "val_qual_auc", "lr", "new_best", "stalled",
                    "lr_cut", "elapsed_s", "traj", "checkins"]
# trajectories are cut after the last check-in at or under MAX_SIMS root
# visits; the model is causal, so earlier outputs are unchanged
MAX_SIMS = 16_000
LOAD_PROCS = 4
# heads trained, evaluated and used by the stop rules (stability, quality);
# set from --heads
ACTIVE_HEADS = (True, True)
# held-out shards are read until they hold val + test + HOLD_SPARE, so a
# prior can be fitted on training trajectories before the pool fills
HOLD_SPARE = 5000
# outputs go to the first
DEFAULT_COLLECTIONS = ["ctrl_collect_150k_20261115_104609",
                       "ctrl_collect_100k2_20261115_214410",
                       "ctrl_collect_100k_20261114_230557",
                       "ctrl_collect_26k_20261114_184650"]
# mixers: {one-based conv block: SwiGLU hidden width}, applied after that block
SPECS = {"M": {"k": 5, "dilations": D6, "mixers": {2: 128, 4: 128}},
         "L": {"k": 5, "dilations": D6, "mixers": {2: 256, 4: 256, 6: 256}}}
# end-of-run fire table: the rule is sims >= FIRE_FLOOR and both heads >= threshold
FIRE_FLOOR = 200
FIRE_THRESHOLDS = (0.9, 0.95, 0.98, 0.99)
# TSL rises over xc0's move counted as inaccuracies / blunders (ESL 0.05 / 0.12)
TSL_BARS = (50, 120)
PCT, INT, ONE, SIGNED = (lambda v: f"{100 * v:.1f}", lambda v: f"{v:.0f}",
                         lambda v: f"{v:.1f}", lambda v: f"{v:+.1f}")
# goal-2 gate tables, one per naive stop point: xc0 wants to play its
# most-visited move at the floor, but waits for the controller to agree
GATE_FLOORS = (400, 1200, 2400)
# printed per floor as two tables: what happened to the move, then its cost
GATE_MOVE_COLUMNS = (("wait", "wait%", 7, PCT), ("fallback", "fallback%", 11, PCT),
                     ("same", "same%", 7, PCT), ("equiv", "equiv%", 8, PCT),
                     ("better", "better%", 9, PCT), ("worse", "worse%", 8, PCT),
                     ("no_help", "no help%", 10, PCT))
GATE_COST_COLUMNS = (("naive_tsl", "naive TSL", 11, ONE),
                     ("gated_tsl", "gated TSL", 11, ONE), ("dtsl", "dTSL", 7, SIGNED),
                     ("naive_sf", "naive sf%", 11, PCT), ("gated_sf", "gated sf%", 11, PCT),
                     ("wasted", "wasted", 8, INT))
# printed as two tables: sims, then move quality
FIRE_SIMS_COLUMNS = (("stop", "stop%", 7, PCT), ("sims", "sims@thresh", 13, INT),
                     ("saved", "sims saved", 12, INT), ("ext", "ext%", 7, PCT),
                     ("unfire", "unfire%", 9, PCT), ("total", "sims total", 12, INT))
FIRE_QUALITY_COLUMNS = (("tsl_tot", "TSL tot", 9, ONE), ("cpl_tot", "CPL tot", 9, ONE),
                        ("tsl", "TSL", 7, ONE), ("cpl", "CPL", 7, ONE),
                        ("xc0", "xc0%", 7, PCT), ("sf", "sf%", 7, PCT),
                        ("dtsl", "dTSL", 7, SIGNED), ("dcpl", "dCPL", 7, SIGNED),
                        *((f"d{b}", f"dTSL>{b}", 10, PCT) for b in TSL_BARS))


class InputScale(nn.Module):
    """Learnable per-feature scale and bias on top of the fixed empirical
    scales; folds into the next Linear at export."""

    def __init__(self, n):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return x * self.scale + self.bias


class SwiGLU(nn.Module):
    def __init__(self, hidden, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.up = nn.Linear(WIDTH, hidden)
        self.gate = nn.Linear(WIDTH, hidden)
        self.down = nn.Linear(hidden, WIDTH)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        return x + self.drop(self.down(F.silu(self.up(h)) * self.gate(h)))


class TemporalBlock(nn.Module):
    def __init__(self, k, d, dropout=0.0):
        super().__init__()
        self.k, self.d = k, d
        self.conv = nn.Conv1d(WIDTH, WIDTH, k, dilation=d)
        self.norm = nn.LayerNorm(WIDTH)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gamma, beta):
        y = F.pad(x.transpose(1, 2), ((self.k - 1) * self.d, 0))
        y = self.conv(y).transpose(1, 2)
        return x + self.drop(F.silu(self.norm(y) * (1 + gamma) + beta))


class Controller(nn.Module):
    """Dropout sits on the conv block and mixer residual branches and on the
    pooled features feeding the heads."""

    def __init__(self, k, dilations, mixers, dropout=0.0):
        super().__init__()
        self.context_scale = InputScale(CONTEXT_DIM)
        self.checkin_scale = InputScale(CHECKIN_DIM)
        self.input_proj = nn.Linear(CHECKIN_DIM, WIDTH)
        self.position_proj = nn.Sequential(nn.Linear(CONTEXT_DIM, WIDTH), nn.SiLU(),
                                           SwiGLU(256))
        self.film_proj = nn.Linear(WIDTH, len(dilations) * 2 * WIDTH)
        self.blocks = nn.ModuleList([TemporalBlock(k, d, dropout) for d in dilations])
        self.mixers = nn.ModuleDict({str(i - 1): SwiGLU(h, dropout)
                                     for i, h in mixers.items()})
        self.output_proj = nn.Linear(3 * WIDTH, WIDTH)
        self.head_drop = nn.Dropout(dropout)
        self.stability_head = nn.Linear(WIDTH, 1)
        self.quality_head = nn.Linear(WIDTH, 1)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, context, checkins):
        """context (B, 48), checkins (B, T, 64) -> logits (B, T, 2). Each
        check-in's logits see only it and the check-ins before it, as the
        cached step graph does."""
        p = self.position_proj(self.context_scale(context))
        film = self.film_proj(p).view(-1, len(self.blocks), 2, 1, WIDTH)
        x = self.input_proj(self.checkin_scale(checkins))
        for i, block in enumerate(self.blocks):
            x = block(x, film[:, i, 0], film[:, i, 1])
            if str(i) in self.mixers:
                x = self.mixers[str(i)](x)
        t = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        mean = x.cumsum(1) / t.view(1, -1, 1)
        h = F.silu(self.output_proj(torch.cat([x, mean, p.unsqueeze(1).expand_as(x)], -1)))
        h = self.head_drop(h)
        return torch.cat([self.stability_head(h), self.quality_head(h)], -1)


def load_metadata(coll_dir):
    with open(os.path.join(coll_dir, "meta.json")) as f:
        meta = json.load(f)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "context_feature_names": list(NN_FEATURE_NAMES) + list(BOARD_FEATURE_NAMES),
        "checkin_feature_names": list(CHECK_IN_FEATURE_NAMES),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Unsupported collection {key}: {meta.get(key)!r}")
    return meta


def validate_record(r):
    """Reject malformed inputs before tensors reach the model."""
    tid = r.get("trajectory_id", "<missing>")
    if not isinstance(tid, str) or tid == "<missing>":
        raise ValueError("Each trajectory needs a string trajectory_id")
    ck, ctx = np.asarray(r["checkins"]), np.asarray(r["context"])
    tg, va = np.asarray(r["targets"]), np.asarray(r["target_valid"])
    if ck.ndim != 2:
        raise ValueError(f"{tid}: checkins must be a matrix")
    n = len(ck)
    expected = {"context": (CONTEXT_DIM,), "checkins": (n, CHECKIN_DIM),
                "targets": (n, 2), "target_valid": (n, 2)}
    for key, shape in expected.items():
        a = np.asarray(r[key])
        if a.shape != shape or n == 0:
            raise ValueError(f"{tid}: invalid {key} shape {a.shape}")
        if key == "target_valid":
            if a.dtype != np.bool_:
                raise ValueError(f"{tid}: target_valid must be boolean")
        elif a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f"{tid}: {key} must contain finite float32 values")
    if not va.any() or not np.isin(tg[va], [0, 1]).all():
        raise ValueError(f"{tid}: needs valid binary targets")
    visits = np.asarray(r["root_visits"])
    if (visits.shape != (n,) or not np.issubdtype(visits.dtype, np.integer)
            or visits[0] <= 0 or not (np.diff(visits) > 0).all()):
        raise ValueError(f"{tid}: invalid root_visits")
    if len(r["checkpoint_move"]) != n:
        raise ValueError(f"{tid}: checkpoint_move length mismatch")
    if r.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{tid}: unsupported schema_version")
    pos = r.get("position", {})
    for key in ("run_tag", "game_id"):
        if not isinstance(pos.get(key), str) or not pos[key].strip():
            raise ValueError(f"{tid}: needs position.{key}")
    return ck, ctx, tg, va


def checkin_losses(teacher, moves):
    """(ESL, CPL) of each check-in's move against the teacher's best, or None
    where either eval is missing."""
    base = teacher.get("baseline")
    evals = teacher.get("evals", {})
    out = []
    for m in moves:
        ev = evals.get(m)
        out.append(None if base is None or ev is None else
                   (0.5 * max(0.0, sf_value(base) - sf_value(ev)),
                    max(0, base.cp_stm - ev.cp_stm)))
    return out


def trajectory_meta(r):
    """Who the position is, for looking cases up: its corpus fields, the
    teacher's verdict and the search's own settings. Small and flat, so it
    rides along with every trajectory into the predictions parquet."""
    pos, teacher = r["position"], r.get("teacher", {})
    base = teacher.get("baseline")
    return {**{k: pos.get(k) for k in META_POS_FIELDS},
            "effort_bin": r.get("effort_bin"),
            "teacher_depth": teacher.get("depth"),
            "teacher_best": teacher.get("baseline_move"),
            "teacher_best_cp": base.cp_stm if base is not None else None,
            "teacher_best_mate": base.mate_stm if base is not None else None,
            "final_sims": r.get("final_sims"),
            "stratum_params": json.dumps(r.get("search", {}).get("stratum_params", {}))}


def compact(r, collection):
    """The validated parts of one record that training and the stop replays
    use: the SF evals become per-check-in losses and the rest is dropped.
    Check-ins past MAX_SIMS root visits are cut."""
    ck, ctx, tg, va = validate_record(r)
    visits = r["root_visits"]
    keep = max(1, int(np.searchsorted(visits, MAX_SIMS, side="right")))
    ck, tg, va, visits = ck[:keep], tg[:keep], va[:keep], visits[:keep]
    moves = [sys.intern(m) for m in r["checkpoint_move"][:keep]]
    d = {"context": ctx, "checkins": ck, "targets": tg, "valid": va,
         "trajectory_id": r["trajectory_id"],
         "root_visits": visits, "checkpoint_move": moves,
         "losses": checkin_losses(r.get("teacher", {}), moves),
         "min_decision": r.get("search", {}).get("min_decision", 0),
         "xc0_cap": r["search"]["initial_ceiling"], "collection": collection,
         "stratum": r["position"].get("stratum"), "end_reason": r.get("end_reason"),
         "meta": trajectory_meta(r)}
    # xc0's own stage-1 decision: a dominance lock, or the initial ceiling.
    # The proof always holds at the ceiling, so only a decision short of it
    # is a lock.
    ref = r["stages"][0]["reference"]
    d["xc0_k"] = ref["checkin"]
    d["xc0_locked"] = (ref["how"] == "lock"
                       and int(r["root_visits"][ref["checkin"]]) < d["xc0_cap"])
    # where xc0 stops in a game: at the lock, else at the initial ceiling
    d["xc0_stop"] = (int(r["root_visits"][ref["checkin"]]) if d["xc0_locked"]
                     else d["xc0_cap"])
    # equivalence sets: xc0's at its stage-1 decision, SF's for the position
    d["xc0_set"] = set(ref["xc0_set"])
    d["sf_set"] = set(r.get("teacher", {}).get("expert_set", ()))
    return d


def collection_name(path):
    return os.path.basename(os.path.normpath(path))


def load_collection(coll_dir, metadata=None):
    """Every trajectory in one collection, compacted as records stream in."""
    if metadata is None:
        metadata = load_metadata(coll_dir)
    name = collection_name(coll_dir)
    return [compact(r, name) for r in iter_records(coll_dir)]


def read_shard(path):
    """One shard, compacted; a loader process target."""
    name = collection_name(os.path.dirname(path))
    return [compact(r, name) for r in iter_shard(path)]


def bounded_reads(pool, paths, ahead):
    """read_shard results in path order with at most `ahead` shards in
    flight, so finished shards never pile up in memory."""
    pending, it = [], iter(paths)
    for p in itertools.islice(it, ahead):
        pending.append(pool.apply_async(read_shard, (p,)))
    while pending:
        recs = pending.pop(0).get()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(pool.apply_async(read_shard, (nxt,)))
        yield recs


def hold_out(files, sizes, spare, rng, procs):
    """Shuffle the shards, read them in that order until they hold
    sum(sizes) + spare trajectories, then draw each holdout at random from
    them. Returns (holdouts, leftovers, held shard files, bytes per
    trajectory); the leftovers go to training."""
    order = [files[i] for i in rng.permutation(len(files))]
    need = sum(sizes) + spare
    held, used = [], []
    with mp.get_context("spawn").Pool(procs) as pool:
        for path, recs in zip(order, bounded_reads(pool, order, procs)):
            held += recs
            used.append(path)
            if len(held) >= need:
                break
    if len(held) < need:
        raise ValueError(f"Only {len(held)} trajectories for {need} held out")
    held = [held[i] for i in rng.permutation(len(held))]
    out, start = [], 0
    for n in sizes:
        out.append(held[start:start + n])
        start += n
    per_traj = sum(os.path.getsize(p) for p in used) / len(held)
    return out, held[start:], used, per_traj


class TrajectoryPool:
    """Async draining shuffle buffer over the training shards, after
    bootstrap_model_async_sparse_pkl's AsyncShardPool. Loader processes read
    shards, in a fresh random order every pass, into a pool topped up to
    POOL_HIGH whenever it drains below POOL_LOW; extra trajectories (held-out
    shard leftovers) join every pass. A prefetch thread draws each batch at
    random from the pool and removes it, so a pass serves every trajectory
    once, and stages padded batches PREFETCH ahead."""

    def __init__(self, files, extra, size, seed, procs):
        self.files, self.extra, self.size, self.procs = files, extra, size, procs
        # one generator per thread: numpy generators are not thread-safe
        self.order_rng, self.draw_rng = (np.random.default_rng(s) for s in
                                         np.random.SeedSequence(seed).spawn(2))
        self.pool, self.passes = [], 0
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.ready = queue.Queue(PREFETCH)
        self.err = None
        for target in (self.loader, self.prefetch):
            threading.Thread(target=target, daemon=True).start()

    def stream(self):
        with mp.get_context("spawn").Pool(self.procs) as pool:
            while not self.stop_ev.is_set():
                order = [self.files[i] for i in self.order_rng.permutation(len(self.files))]
                if self.extra:
                    yield list(self.extra)
                for recs in bounded_reads(pool, order, self.procs):
                    if self.stop_ev.is_set():
                        return
                    yield recs
                self.passes += 1

    def loader(self):
        try:
            filling = True
            for recs in self.stream():
                while not self.stop_ev.is_set():
                    with self.lock:
                        n = len(self.pool)
                    if filling and n >= POOL_HIGH:
                        filling = False
                    if not filling and n < POOL_LOW:
                        filling = True
                    if filling:
                        break
                    time.sleep(0.05)
                with self.lock:
                    self.pool += recs
        except BaseException as e:
            self.err = e

    def prefetch(self):
        started = False
        while not self.stop_ev.is_set():
            with self.lock:
                n = len(self.pool)
                go = n >= (self.size if started else POOL_LOW) and self.err is None
                if go:
                    idx = sorted(self.draw_rng.choice(n, self.size, replace=False).tolist(),
                                 reverse=True)
                    chunk = []
                    for i in idx:
                        chunk.append(self.pool[i])
                        self.pool[i] = self.pool[-1]
                        self.pool.pop()
            if self.err is not None:
                self.ready.put(self.err)
                return
            if not go:
                time.sleep(0.02)
                continue
            started = True
            n_ck = sum(len(d["checkins"]) for d in chunk)
            self.ready.put((len(chunk), n_ck, pad_arrays(chunk)))

    def get(self):
        got = self.ready.get()
        if isinstance(got, BaseException):
            raise got
        return got

    def stop(self):
        self.stop_ev.set()


def batcher_main(files, extra, size, seed, procs, out_q, stop_ev):
    """Process target: runs the TrajectoryPool and forwards its padded numpy
    batches, so shard reads and padding never hold the trainer's GIL. Each
    batch carries the pool size and the batches staged behind it."""
    pool = TrajectoryPool(files, extra, size, seed, procs)
    try:
        while not stop_ev.is_set():
            item = (*pool.get(), len(pool.pool), pool.ready.qsize())
            while not stop_ev.is_set():
                try:
                    out_q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue
    except BaseException as e:
        out_q.put(RuntimeError(f"batcher failed: {e!r}"))
    finally:
        pool.stop()


class BatchStream:
    """The trainer's side of batcher_main: a separate process builds the
    batches, and get() returns them as tensors on device. ready and pool_size
    are as of the last get: prebuilt batches waiting in both queues, and
    trajectories in the shuffle pool."""

    def __init__(self, files, extra, size, seed, procs, device):
        ctx = mp.get_context("spawn")
        self.device = device
        self.ready, self.pool_size = 0, 0
        self.q, self.stop_ev = ctx.Queue(PREFETCH), ctx.Event()
        # not a daemon: it runs its own pool of loader processes
        self.proc = ctx.Process(target=batcher_main, args=(
            files, extra, size, seed, procs, self.q, self.stop_ev))
        self.proc.start()
        # runs before multiprocessing joins its children, so a crash never hangs
        atexit.register(self.stop)

    def get(self):
        got = self.q.get()
        if isinstance(got, BaseException):
            raise got
        n_traj, n_ck, arrays, self.pool_size, staged = got
        self.ready = self.q.qsize() + staged
        return n_traj, n_ck, tuple(torch.from_numpy(a).to(self.device) for a in arrays)

    def stop(self):
        self.stop_ev.set()
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()


def batch_indices(data, size, rng=None):
    """Trajectory indices per batch, shuffled when rng is given."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    idx = rng.permutation(len(data)) if rng is not None else np.arange(len(data))
    return [idx[i:i + size] for i in range(0, len(idx), size)]


def pad_arrays(chunk):
    """(ctx, ck, tg, va) numpy arrays; padding goes after the last check-in,
    so causal outputs at real check-ins never see it."""
    t = max(len(d["checkins"]) for d in chunk)
    ck = np.zeros((len(chunk), t, CHECKIN_DIM), np.float32)
    tg = np.zeros((len(chunk), t, 2), np.float32)
    va = np.zeros((len(chunk), t, 2), bool)
    for b, d in enumerate(chunk):
        n = len(d["checkins"])
        ck[b, :n], tg[b, :n], va[b, :n] = d["checkins"], d["targets"], d["valid"]
    ctx = np.stack([d["context"] for d in chunk]).astype(np.float32)
    return ctx, ck, tg, va


def pad_batch(data, idx, pin):
    """pad_arrays as CPU tensors, pinned when pin."""
    tensors = tuple(torch.from_numpy(a) for a in pad_arrays([data[j] for j in idx]))
    return tuple(x.pin_memory() for x in tensors) if pin else tensors


def batches(data, size, device, rng=None, with_indices=False):
    """Padded batches, built PREFETCH ahead on a background thread so the
    GPU never waits on padding."""
    groups = batch_indices(data, size, rng)
    pin = str(device).startswith("cuda")
    q = queue.Queue(PREFETCH)

    def fill():
        try:
            for idx in groups:
                q.put((idx, pad_batch(data, idx, pin)))
            q.put(None)
        except BaseException as e:
            q.put(e)

    threading.Thread(target=fill, daemon=True).start()
    while (got := q.get()) is not None:
        if isinstance(got, BaseException):
            raise got
        idx, cpu = got
        tensors = tuple(x.to(device, non_blocking=pin) for x in cpu)
        yield (idx, tensors) if with_indices else tensors


def masked_bce(logits, tg, va):
    """Per-head contributions to equal-trajectory BCE, shape (2,).

    Average valid check-ins per head, then available heads per trajectory,
    then trajectories. Summing the result gives the scalar training loss.
    Trajectories with no valid targets contribute no weight.
    """
    loss = F.binary_cross_entropy_with_logits(logits, tg, reduction="none")
    counts = va.sum(dim=1)
    head_mean = loss.masked_fill(~va, 0.0).sum(dim=1) / counts.clamp(min=1)
    available = counts > 0
    head_weights = available / available.sum(dim=1, keepdim=True).clamp(min=1)
    usable = available.any(dim=1)
    return (head_mean * head_weights).sum(dim=0) / usable.sum().clamp(min=1)


def auc(y, p):
    pos = y > 0.5
    # float: Windows numpy sums bools to int32, and n1 * n0 overflows it
    n1, n0 = float(pos.sum()), float((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    r = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bce_np(y, p, weights=None):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=weights))


def at_sims(d, sims):
    """Index of the first check-in at or past sims, else the last. Up to the
    initial ceiling its move is xc0's most-visited move at sims: a trajectory
    that ended sooner had locked its leader through that ceiling."""
    return min(int(np.searchsorted(d["root_visits"], sims)), len(d["root_visits"]) - 1)


def move_loss(d, k):
    """(ESL, CPL) of check-in k's move against the teacher's best, or None."""
    return d["losses"][k]


def mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def fire_replay(predictions, threshold):
    """Goal-1 replay of one threshold against xc0's 4k move (its lock or its
    4k leader, always the same move). Fire at the first check-in at FIRE_FLOOR
    sims or more, up to 4k, where both heads reach threshold. Check-ins past 4k
    exist only where SF rejected the move, so the replay never uses them.
    xc0 stops at its dominance lock, else at 4k. stop% and sims total (fire,
    else xc0's stop) cover all trajectories; the rest cover the fired ones.
    sims saved = xc0's stop - the fire, negative when the fire extends past a
    lock. unfire% = a later check-in up to 4k drops below threshold. TSL tot /
    CPL tot, over all trajectories, score the move played first past the post:
    the fired move if the fire comes at or before xc0's stop, else xc0's move.
    TSL = ESL x1000. xc0% / sf% = the fired move is in xc0's stage-1
    equivalence set / SF's equivalence set."""
    sims, saved, ext, unfire, match, sf = [], [], [], [], [], []
    tsl, cpl, dtsl, dcpl, total, played = [], [], [], [], [], []
    for d, prob in predictions:
        cap_k = at_sims(d, d["xc0_cap"])
        ok = ((prob[:cap_k + 1] >= threshold).all(1)
              & (d["root_visits"][:cap_k + 1] >= FIRE_FLOOR))
        hits = np.flatnonzero(ok)
        first = int(hits[0]) if len(hits) else None
        if first is None or d["root_visits"][first] > d["xc0_stop"]:
            first = d["xc0_k"]
        played.append(move_loss(d, first))
        if not len(hits):
            total.append(d["xc0_stop"])
            continue
        k = int(hits[0])
        sims.append(int(d["root_visits"][k]))
        total.append(sims[-1])
        saved.append(d["xc0_stop"] - sims[-1])
        ext.append(d["xc0_locked"] and sims[-1] > d["xc0_stop"])
        unfire.append(not ok[k:].all())
        match.append(d["checkpoint_move"][k] in d["xc0_set"])
        sf.append(d["checkpoint_move"][k] in d["sf_set"])
        mine, ref = move_loss(d, k), move_loss(d, d["xc0_k"])
        if mine is not None:
            tsl.append(1000 * mine[0])
            cpl.append(mine[1])
            if ref is not None:
                dtsl.append(1000 * (mine[0] - ref[0]))
                dcpl.append(mine[1] - ref[1])
    dtsl = np.array(dtsl)
    played = [x for x in played if x is not None]
    return {"stop": len(sims) / len(predictions), "sims": mean(sims),
            "saved": mean(saved), "ext": mean(ext), "total": mean(total),
            "tsl_tot": mean([1000 * e for e, _ in played]),
            "cpl_tot": mean([c for _, c in played]),
            "unfire": mean(unfire),
            "tsl": mean(tsl), "cpl": mean(cpl), "xc0": mean(match), "sf": mean(sf),
            "dtsl": mean(dtsl), "dcpl": mean(dcpl),
            **{f"d{b}": mean(dtsl > b) for b in TSL_BARS}}


def xc0_reference(predictions):
    """xc0 alone: stop% = lock rate, sims at the lock, sims total = its stop
    (lock, else 4k); TSL / CPL / sf% of its move; all over all trajectories."""
    ds = [d for d, _ in predictions]
    locked = [d for d in ds if d["xc0_locked"]]
    losses = [x for x in (move_loss(d, d["xc0_k"]) for d in ds) if x is not None]
    return {"stop": len(locked) / len(ds), "sims": mean([d["xc0_stop"] for d in locked]),
            "total": mean([d["xc0_stop"] for d in ds]),
            "tsl": mean([1000 * e for e, _ in losses]), "cpl": mean([c for _, c in losses]),
            "tsl_tot": mean([1000 * e for e, _ in losses]),
            "cpl_tot": mean([c for _, c in losses]),
            "sf": mean([d["checkpoint_move"][d["xc0_k"]] in d["sf_set"] for d in ds])}


def gate_replay(predictions, floor, threshold):
    """Goal-2 replay: xc0 would play its most-visited move at the first
    check-in at floor sims or more (naive). Gated, it waits from there for the
    first check-in up to 4k where both heads reach threshold and plays that
    move, else the move at 4k (or the last check-in if the trajectory ended
    sooner). The lock is ignored and nothing is ever saved: wasted = gated
    sims - naive sims, 0 when the controller agrees at the floor.

    same / equiv / better / worse partition the labelled trajectories: equiv
    = a different move, both SF-equivalent; better / worse by TSL otherwise.
    no help = of the naive moves outside SF's set, the share still outside
    after the wait."""
    wait, fallback, wasted, bucket, no_help = [], [], [], [], []
    naive_tsl, gated_tsl, naive_sf, gated_sf = [], [], [], []
    for d, prob in predictions:
        moves, visits = d["checkpoint_move"], d["root_visits"]
        s_k, cap_k = at_sims(d, floor), at_sims(d, d["xc0_cap"])
        s_k = min(s_k, cap_k)
        hits = np.flatnonzero((prob[s_k:cap_k + 1] >= threshold).all(1))
        k = s_k + int(hits[0]) if len(hits) else cap_k
        wait.append(k != s_k or not len(hits))
        fallback.append(not len(hits))
        wasted.append(int(visits[k]) - int(visits[s_k]))
        naive, gated = move_loss(d, s_k), move_loss(d, k)
        n_in, g_in = moves[s_k] in d["sf_set"], moves[k] in d["sf_set"]
        naive_sf.append(n_in)
        gated_sf.append(g_in)
        if not n_in:
            no_help.append(not g_in)
        if naive is None or gated is None:
            continue
        naive_tsl.append(1000 * naive[0])
        gated_tsl.append(1000 * gated[0])
        if moves[k] == moves[s_k]:
            bucket.append("same")
        elif (n_in and g_in) or gated[0] == naive[0]:
            bucket.append("equiv")
        else:
            bucket.append("better" if gated[0] < naive[0] else "worse")
    bucket = np.array(bucket)
    return {"wait": mean(wait), "fallback": mean(fallback),
            **{b: mean(bucket == b) for b in ("same", "equiv", "better", "worse")},
            "no_help": mean(no_help), "naive_tsl": mean(naive_tsl),
            "gated_tsl": mean(gated_tsl),
            "dtsl": mean(np.array(gated_tsl) - np.array(naive_tsl)),
            "naive_sf": mean(naive_sf), "gated_sf": mean(gated_sf),
            "wasted": mean(wasted)}


def full_search_reference(predictions):
    """TSL and sf% of the move at 4k (or the last check-in), lock ignored."""
    ds = [d for d, _ in predictions]
    ks = [at_sims(d, d["xc0_cap"]) for d in ds]
    losses = [x for x in (move_loss(d, k) for d, k in zip(ds, ks)) if x is not None]
    return (mean([1000 * e for e, _ in losses]),
            mean([d["checkpoint_move"][k] in d["sf_set"] for d, k in zip(ds, ks)]))


def trajectory_rows(predictions, split):
    """One row per trajectory with both heads and everything a stop rule
    replay needs per check-in, so rules can be replayed without the model
    or the collection. Per-check-in columns are equal-length lists."""
    rows = []
    for d, prob in predictions:
        losses = [move_loss(d, k) for k in range(len(prob))]
        moves = list(d["checkpoint_move"])
        rows.append({
            "trajectory_id": d["trajectory_id"], "split": split,
            "collection": d["collection"], "stratum": d["stratum"],
            "end_reason": d["end_reason"], "xc0_cap": d["xc0_cap"],
            "xc0_k": d["xc0_k"], "xc0_stop": d["xc0_stop"],
            "xc0_locked": d["xc0_locked"], "xc0_move": moves[d["xc0_k"]],
            "root_visits": d["root_visits"].tolist(), "move": moves,
            "stab_prob": prob[:, 0].tolist(), "qual_prob": prob[:, 1].tolist(),
            "stab_target": d["targets"][:, 0].tolist(),
            "qual_target": d["targets"][:, 1].tolist(),
            "stab_valid": d["valid"][:, 0].tolist(),
            "qual_valid": d["valid"][:, 1].tolist(),
            "tsl": [1000 * x[0] if x else None for x in losses],
            "cpl": [x[1] if x else None for x in losses],
            "in_sf": [m in d["sf_set"] for m in moves],
            "in_xc0": [m in d["xc0_set"] for m in moves],
            **d["meta"], **context_columns(d)})
    return rows


def context_columns(d):
    """The 48 context features as ctx_<name> columns."""
    return {f"ctx_{n}": float(v) for n, v in zip(CONTEXT_NAMES, d["context"])}


def save_predictions(path, rows):
    import pandas as pd
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[ctrl] {len(rows)} trajectory predictions -> {path}")


def report(predictions, label):
    """Print the goal-1 fire tables and goal-2 gate tables for predictions;
    returns them for the metrics json."""
    reference = xc0_reference(predictions)
    fires = {t: fire_replay(predictions, t) for t in FIRE_THRESHOLDS}
    print(f"\n[ctrl] fire table on {label}: fires at the first check-in with sims >= "
          f"{FIRE_FLOOR} and both heads >= threshold, up to 4k")
    print("[ctrl] target = xc0's move at its stop (dominance lock, else 4k). stop% "
          "and sims total over all traj, the rest over fired traj")
    print("[ctrl] sims saved = xc0 stop - fire. ext% = fires past a lock. sims "
          "total = fire, else xc0 stop")
    print("[ctrl] TSL tot / CPL tot = first past the post over all traj: the fired "
          "move if it fires by xc0's stop, else xc0's")
    print("[ctrl] TSL = ESL x1000. xc0% / sf% = fired move in xc0's / SF's "
          "equivalence set")
    print("[ctrl] dTSL / dCPL = fired move minus xc0's move, dTSL>b = % of fires "
          "worse than xc0 by more than b")
    print("[ctrl] xc0 lock row: stop% = lock rate, sims at the lock, sims total "
          "and TSL / CPL / sf% over all traj")
    rows = [("xc0 lock", reference)] + [(f"{t:g}", r) for t, r in fires.items()]
    print()
    print_fire_table(rows, FIRE_SIMS_COLUMNS)
    print()
    print_fire_table(rows, FIRE_QUALITY_COLUMNS)

    gates = {s: {t: gate_replay(predictions, s, t) for t in FIRE_THRESHOLDS}
             for s in GATE_FLOORS}
    full_tsl, full_sf = full_search_reference(predictions)
    print(f"\n[ctrl] gate tables on {label}: naive = xc0's most-visited move at S. "
          "gated = wait from S for both heads >= threshold,")
    print("[ctrl] else the move at 4k (or the last check-in). Lock ignored. wait% = "
          "no agreement at S, fallback% = none by 4k")
    print("[ctrl] same / equiv / better / worse sum to 100: equiv = different move, "
          "both SF-equivalent")
    print("[ctrl] no help% = of naive moves outside SF's set, % still outside. "
          "wasted = gated sims - naive sims")
    for s, table in gates.items():
        rows = [(f"{t:g}", r) for t, r in table.items()]
        print(f"\nS = {s}   4k ref: TSL {full_tsl:.1f}  sf% {100 * full_sf:.1f}")
        print_fire_table(rows, GATE_MOVE_COLUMNS)
        print()
        print_fire_table(rows, GATE_COST_COLUMNS)
    return {"fire_floor": FIRE_FLOOR, "xc0_lock": reference,
            "fire_table": [{"threshold": t, **r} for t, r in fires.items()],
            "full_search": {"tsl": full_tsl, "sf": full_sf},
            "gate_tables": [{"floor": s, "threshold": t, **r}
                            for s, table in gates.items() for t, r in table.items()]}


def print_fire_table(rows, columns):
    """rows: [(label, row dict)]; a column a row lacks prints as --."""
    print(f"{'threshold':<10}" + "".join(f"{name:>{w}}" for _, name, w, _ in columns))
    for label, row in rows:
        print(f"{label:<10}" + "".join(
            f"{fmt(row[key]) if row.get(key) is not None else '--':>{w}}"
            for key, _, w, fmt in columns))


def mask_heads(va):
    """va with the heads left out of ACTIVE_HEADS marked unlabelled."""
    if all(ACTIVE_HEADS):
        return va
    va = va.clone()
    va[..., [h for h, on in enumerate(ACTIVE_HEADS) if not on]] = False
    return va


@torch.no_grad()
def predict(model, data, size, device):
    """(trajectory, per-check-in head probabilities) for every trajectory. A
    head left out of ACTIVE_HEADS reads 1.0, so stop rules see only the
    trained heads."""
    model.eval()
    out = []
    for idx, (ctx, ck, tg, va) in batches(data, size, device, with_indices=True):
        prob = torch.sigmoid(model(ctx, ck).float()).cpu().numpy()
        prob[..., [h for h, on in enumerate(ACTIVE_HEADS) if not on]] = 1.0
        out += [(data[j], prob[row, :len(data[j]["checkins"])])
                for row, j in enumerate(idx)]
    return out


@torch.no_grad()
def evaluate(model, data, size, device, prior):
    """Trajectory-weighted selection objective plus labelled diagnostic views."""
    model.eval()
    ys, ps, ws = [[], []], [[], []], [[], []]
    objective = np.zeros(2, np.float64)
    usable_total = 0
    for ctx, ck, tg, va in batches(data, size, device):
        va = mask_heads(va)
        logits = model(ctx, ck).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite validation logits")
        usable = int(va.any(dim=(1, 2)).sum().item())
        objective += masked_bce(logits, tg, va).cpu().numpy() * usable
        usable_total += usable
        prob = torch.sigmoid(logits)
        counts = va.sum(dim=1)
        active_heads = (counts > 0).sum(dim=1, keepdim=True).clamp(min=1)
        weights = 1.0 / (counts.clamp(min=1) * active_heads)
        for h in range(2):
            m = va[..., h]
            ys[h].append(tg[..., h][m].cpu().numpy())
            ps[h].append(prob[..., h][m].cpu().numpy())
            ws[h].append(weights[:, None, h].expand_as(m)[m].cpu().numpy())
    heads, baseline_objective = {}, 0.0
    for h in range(2):
        if not ACTIVE_HEADS[h]:
            continue
        y, p = np.concatenate(ys[h]), np.concatenate(ps[h])
        weights = np.concatenate(ws[h])
        baseline = bce_np(y, np.full_like(y, prior[h]), weights)
        baseline_objective += baseline * weights.sum() / usable_total
        heads[HEADS[h]] = {
            "trajectory_bce": bce_np(y, p, weights), "trajectory_baseline_bce": baseline,
            "trajectory_accuracy": float(np.average((p > .5) == (y > .5), weights=weights)),
            "checkpoint_bce": bce_np(y, p), "checkpoint_auc": float(auc(y, p)),
            "checkpoint_accuracy": float(((p > .5) == (y > .5)).mean()),
        }
    return {"loss": float(objective.sum() / usable_total),
            "baseline_loss": float(baseline_objective), "heads": heads}


def pos_rate(data):
    """Constant predictors fitted with the same trajectory/head weighting."""
    numerator, denominator = np.zeros(2), np.zeros(2)
    for d in data:
        counts = d["valid"].sum(0)
        active = counts > 0
        weights = active / max(1, active.sum())
        means = (d["targets"] * d["valid"]).sum(0) / np.maximum(counts, 1)
        numerator += means * weights
        denominator += weights
    if (denominator == 0).any():
        raise ValueError("Training data needs labels for both heads")
    return (numerator / denominator).tolist()


def param_groups(model, wd):
    """No weight decay on the input scales, norms and biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        skip = p.ndim < 2 or name.startswith(("context_scale", "checkin_scale"))
        (no_decay if skip else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


class Tee:
    """Writes to the console and a log file at once."""

    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()


def log_console(path):
    """Copy everything the run prints, errors included, to path."""
    f = open(path, "w", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, f), Tee(sys.stderr, f)
    print(f"[ctrl] logging to {path}")


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def main():
    ap = argparse.ArgumentParser()
    # one or more collection dirs or names; outputs go to the first
    ap.add_argument("collections", nargs="*", default=DEFAULT_COLLECTIONS)
    ap.add_argument("--spec", default="M", choices=list(SPECS))
    # cap only: training ends when val stalls at the LR floor
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    # None: REF_LR scaled by sqrt(batch / REF_BATCH)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.03)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--val-size", type=int, default=VAL_SIZE)
    # held out like val; scored once, by the best checkpoint, for the final tables
    ap.add_argument("--test-size", type=int, default=VAL_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--warmup", type=float, default=1.0, help="epochs of linear warmup")
    # after warmup, lr-patience checks without a new best val cut the LR by
    # lr-factor; stalling again after lr-cuts cuts (at the floor) stops
    ap.add_argument("--lr-patience", type=int, default=6)
    ap.add_argument("--lr-factor", type=float, default=0.5)
    ap.add_argument("--lr-cuts", type=int, default=4)
    # quality: the stability loss is off and stop rules use quality alone;
    # outputs are named controller_<spec>_qual*
    ap.add_argument("--heads", default="both", choices=["both", "quality"])
    args = ap.parse_args()
    global ACTIVE_HEADS
    ACTIVE_HEADS = (args.heads == "both", True)
    # every output of a run is named controller_<tag>.*, e.g. M_quality_only_1024
    tag = (f"{args.spec}_{'quality_only' if args.heads == 'quality' else 'two_head'}"
           f"_{args.batch}")
    if (args.batch <= 0 or args.epochs <= 0 or args.eval_every <= 0
            or args.lr_patience <= 0 or args.lr_cuts < 0 or args.warmup < 0
            or not 0 < args.lr_factor < 1 or args.val_size <= 0 or args.test_size <= 0):
        ap.error("batch/epochs/eval-every/val-size/test-size must be positive, "
                 "lr-factor between zero and one")
    if args.lr is None:
        args.lr = REF_LR * (args.batch / REF_BATCH) ** 0.5
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.wd) or args.wd < 0:
        ap.error("lr must be finite and positive; wd must be finite and nonnegative")

    dirs = [c if os.path.isdir(c) else os.path.join(COLLECTIONS_DIR, c)
            for c in args.collections]
    coll_dir = dirs[0]
    log_console(os.path.join(coll_dir, f"controller_{tag}.log"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    metadata = [load_metadata(d) for d in dirs]
    files = [p for d in dirs for p in shard_paths(d)]
    t_load = time.time()
    (val, test), extra, held_files, per_traj = hold_out(
        files, (args.val_size, args.test_size), HOLD_SPARE,
        np.random.default_rng(args.seed + 1), LOAD_PROCS)
    train_files = [p for p in files if p not in set(held_files)]
    # training size is estimated from shard bytes; epochs are approximate
    n_train = len(extra) + round(sum(os.path.getsize(p) for p in train_files) / per_traj)
    names = {collection_name(d) for d in dirs}
    for name in sorted(names):
        print(f"[ctrl] {name}  val {sum(t['collection'] == name for t in val)}  "
              f"test {sum(t['collection'] == name for t in test)}")
    print(f"[ctrl] held out {len(held_files)} of {len(files)} shards in "
          f"{time.time() - t_load:.0f}s: val {len(val)}  test {len(test)}  "
          f"to train {len(extra)}   train ~{n_train:,} traj")
    prior = pos_rate(extra)
    print(f"[ctrl] positive rate (held-out spares)  stability {prior[0]:.3f}  "
          f"quality {prior[1]:.3f}")
    train_pool = BatchStream(train_files, extra, args.batch, args.seed, LOAD_PROCS,
                             device)

    model = Controller(**SPECS[args.spec], dropout=args.dropout).to(device)
    print(f"[ctrl] K5L6-{args.spec}  {sum(p.numel() for p in model.parameters()):,} "
          f"params  device {device}")
    steps_per_epoch = -(-n_train // args.batch)
    steps = args.epochs * steps_per_epoch
    warmup_steps = round(args.warmup * steps_per_epoch)
    lr_floor = args.lr * args.lr_factor ** args.lr_cuts
    print(f"[ctrl] AdamW  batch {args.batch}  lr {args.lr:.2e}  wd {args.wd}  "
          f"dropout {args.dropout}  max epochs {args.epochs}  eval every {args.eval_every}")
    print(f"[ctrl] LR  warmup {warmup_steps} steps, then x{args.lr_factor:g} after "
          f"{args.lr_patience} checks with no new best val, floor {lr_floor:.2e}")
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr)
    guess = evaluate(model, val, VAL_BATCH, device, prior)["baseline_loss"]
    print(f"[ctrl] guessing baseline val BCE {guess:.4f}")

    out_path = os.path.join(coll_dir, f"controller_{tag}.pt")
    # train is every other trajectory in the training collections
    split_manifest = {"method": "trajectories drawn at random from whole held-out shards",
                      "val_size": args.val_size, "test_size": args.test_size,
                      "seed": args.seed + 1, "collections": sorted(names),
                      "held_out_shards": [os.path.relpath(p, COLLECTIONS_DIR)
                                          for p in held_files],
                      "validation": [d["trajectory_id"] for d in val],
                      "test": [d["trajectory_id"] for d in test]}
    best_scales, best_metrics = {}, None
    best, best_step = float("inf"), 0
    print("[ctrl] BCE is trajectory-weighted; AUC is checkpoint-weighted")
    print("[ctrl] val drives the LR: * new best, d best = val - prior best")
    print("[ctrl] pat = checks since the last new best, ** = LR cut or stop at floor")
    print(f"{'step':>7} {'ep':>6} {'train':>7} {'val':>7}  {'d best':>8} {'pat':>3}   | "
          f"{'stab auc':>8}  {'qual auc':>8}  | {'s':>4} {'tot s':>6} {'traj':>6} "
          f"{'check-ins':>9} {'ready':>6} {'pool':>7}")
    best_auc = [0.0, 0.0]
    # one row per check, appended as it happens, so a killed run keeps its curve
    progress_path = os.path.join(coll_dir, f"controller_{tag}_progress.csv")
    with open(progress_path, "w", newline="") as f:
        csv.writer(f).writerow(PROGRESS_COLUMNS)
    step, tot, n, t0 = 0, 0.0, 0, time.time()
    run_t0, tot_traj, tot_ck = t0, 0, 0
    lr, cuts, stalled = args.lr, 0, 0
    val_rng = np.random.default_rng(args.seed + 2)
    model.train()
    while step < steps:
        n_traj, n_ck, (ctx, ck, tg, va) = train_pool.get()
        tot_traj += n_traj
        tot_ck += n_ck
        loss = masked_bce(model(ctx, ck), tg, mask_heads(va)).sum()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step < warmup_steps:
            set_lr(opt, lr * (step + 1) / warmup_steps)
        opt.step()
        step += 1
        tot, n = tot + loss.item() * len(ctx), n + len(ctx)
        if step % args.eval_every and step != steps:
            continue
        pick = val_rng.choice(len(val), min(VAL_SAMPLE, len(val)), replace=False)
        metrics = evaluate(model, [val[j] for j in pick], VAL_BATCH, device, prior)
        # an inactive head has no metrics: {} here, -- in the row, blank in the csv
        stability, quality = (metrics["heads"].get(h, {}) for h in HEADS)
        new_best = metrics["loss"] < best
        delta = (f"{metrics['loss'] - best:>+8.4f}" if math.isfinite(best)
                 else f"{'--':>8}")
        stalled = 0 if new_best or step <= warmup_steps else stalled + 1
        flip = stalled >= args.lr_patience
        aucs = ""
        for h, head in enumerate((stability, quality)):
            a = head.get("checkpoint_auc")
            if a is None:
                aucs += f"{'--':>8}  "
                continue
            aucs += f"{a:>8.3f}{'*' if a > best_auc[h] else ' '} "
            best_auc[h] = max(best_auc[h], a)
        print(f"{step:>7} {step / steps_per_epoch:>6.2f} {tot / n:>7.4f} "
              f"{metrics['loss']:>7.4f}{'*' if new_best else ' '} {delta} "
              f"{stalled:>3}{'**' if flip else '  '} | "
              f"{aucs}| {time.time() - t0:>4.0f} "
              f"{time.time() - run_t0:>6.0f} {compact_count(tot_traj):>6} "
              f"{compact_count(tot_ck):>9} "
              f"{train_pool.ready:>3}/{2 * PREFETCH:<2} "
              f"{compact_count(train_pool.pool_size):>7}",
              flush=True)
        with open(progress_path, "a", newline="") as f:
            csv.writer(f).writerow([
                step, round(step / steps_per_epoch, 4), tot / n, metrics["loss"],
                stability.get("trajectory_bce"), quality.get("trajectory_bce"),
                stability.get("checkpoint_auc"), quality.get("checkpoint_auc"),
                opt.param_groups[0]["lr"], int(new_best), stalled, int(flip),
                round(time.time() - run_t0, 1), tot_traj, tot_ck])
        if new_best:
            best, best_step, best_metrics = metrics["loss"], step, metrics
            best_scales = {name: {
                "scale_min": float(getattr(model, name).scale.min().item()),
                "scale_max": float(getattr(model, name).scale.max().item()),
                "bias_min": float(getattr(model, name).bias.min().item()),
                "bias_max": float(getattr(model, name).bias.max().item())}
                for name in ("context_scale", "checkin_scale")}
            torch.save({"model": model.state_dict(), "spec": args.spec, "step": step,
                        "model_config": SPECS[args.spec], "val": metrics,
                        "prior": prior, "training_config": vars(args),
                        "collection_metadata": metadata, "split": split_manifest,
                        "loss_rule": "mean checkpoints/head, mean available heads, mean trajectories",
                        "input_scale_stats": best_scales}, out_path)
        if flip:
            if cuts == args.lr_cuts:
                print(f"[ctrl] val stalled at the LR floor, stopping at step {step}")
                break
            cuts, stalled = cuts + 1, 0
            print(f"[ctrl] val stalled {args.lr_patience} checks: lr {lr:.2e} -> "
                  f"{lr * args.lr_factor:.2e}  ({cuts}/{args.lr_cuts})")
            lr *= args.lr_factor
            set_lr(opt, lr)
        model.train()
        tot, n, t0 = 0.0, 0, time.time()
    train_pool.stop()

    for name, s in best_scales.items():
        print(f"[ctrl] best {name:<13}  scale {s['scale_min']:.2f} .. {s['scale_max']:.2f}"
              f"   bias {s['bias_min']:+.2f} .. {s['bias_max']:+.2f}")
    print(f"[ctrl] best trajectory-weighted val BCE {best:.4f} at step {best_step} "
          f"-> {out_path}")

    model.load_state_dict(torch.load(out_path, weights_only=False)["model"])
    test_metrics = evaluate(model, test, VAL_BATCH, device, prior)
    aucs = "  ".join(f"{label} auc {test_metrics['heads'][h]['checkpoint_auc']:.3f}"
                     for label, h in (("stab", HEADS[0]), ("qual", HEADS[1]))
                     if h in test_metrics["heads"])
    print(f"\n[ctrl] test ({len(test)} traj, best checkpoint)  BCE "
          f"{test_metrics['loss']:.4f}  guess {test_metrics['baseline_loss']:.4f}  {aucs}")
    predictions = predict(model, test, VAL_BATCH, device)
    tables = report(predictions, "test")
    save_predictions(os.path.join(coll_dir, f"controller_{tag}_predictions.parquet"),
                     trajectory_rows(predictions, "test"))
    with open(os.path.join(coll_dir, f"controller_{tag}_metrics.json"), "w") as f:
        json.dump({"step": best_step, "validation": best_metrics, "test": test_metrics,
                   **tables}, f, indent=2)


if __name__ == "__main__":
    main()
