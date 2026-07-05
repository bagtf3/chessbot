"""build_ab_bakeoff_data.py -- N independent processes for the lc0-vs-xc0
encoding bake-off gatherer (default 3 lc0 + 3 xc0).

Each lc0 process reads a disjoint set of lc0 chunk folders; each xc0 process
reads a disjoint set of run tags. Every process is single-threaded and
sequential over its own slice (no cross-process overlap) with its own async
writer thread, and writes its own .tfrecord.gz shards straight into --out-dir.
All processes write the identical record schema, so a later shuffle pass mixes
lc0- and xc0-sourced records together:

    lc0_board  (112, 8, 8) float16  rule50 plane already /99 (no train-time math)
    xc0_board  (64,)       int16
    policy     (1858,)     float32  compact xc0 domain, uniform-blend 0.001, no clip
    wdl        (3,)        float32  0.5*z_stm + 0.5*search_wdl
    legal      (1858,)     uint8

Threads can't parallelize this workload -- pyfastchess bindings hold the GIL
for every call and pickle.load mostly does too (measured: 2 reader threads
gave zero wall-clock speedup over 1). Processes give real parallelism instead:
all sources write toward one shared --target, a combined lc0+xc0 write count
(mp.Value, incremented under a lock by whichever process just wrote a record).
The instant the combined count crosses --target, whichever process noticed
sets a shared stop event and all of them stop wherever they are. There is no
requirement they contribute evenly -- 20M lc0 / 1M xc0 is a fine outcome.

Usage:
    python -m scripts.data.build_ab_bakeoff_data
"""
import argparse
import gzip
import multiprocessing as mp
import os
import pickle
import queue
import random
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import pyfastchess as pf

from xerces_training.chunkparser import ChunkParser

from chessbot import SP_DIR
from chessbot.review import load_game_index, ANALYZE_PKL

DEFAULT_LC0_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\lc0"
DEFAULT_OUT_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\encoding_test\staging"

TARGET_TOTAL = 21_000_000   # combined lc0+xc0 write count; all stop once this is hit
LC0_PROCS = 3
XC0_PROCS = 3
RECORDS_PER_FILE  = 10_240
WRITER_QUEUE      = 6        # shard slices queued for async gzip (~230MB each)
PROGRESS_EVERY    = 20_000   # per-proc report cadence (well under the 200k MILESTONE)
MILESTONE         = 200_000  # combined-written step at which the aggregate log prints

UNIFORM_BLEND = 0.001
Z_BLEND       = 0.5
LC0_DRAW_DROP = 0.5

GAME_CPL_MAX = 20   # game-level average CPL filter
PLY_CPL_MAX  = 15   # per-move CPL filter

LC0_HISTORY_PLIES = 7   # lc0 board = current + 7 prior positions
SKIP_SCENARIOS    = {"blunder_replay"}                # short, half-SF, unfair to warm
NO_WARM_SCENARIOS = {"startpos", "piece_odds", "piece_training"}  # real root; blank history genuine

RUN_TAGS = [
    "16m_precond_run1", "16m_precond_run0", "val_test",
    "precond_run5", "precond_run4", "precond_run3", "precond_run2", "precond_run1",
    "cfiw_wdl_run2", "cfiw_wdl_run1",
]

SL_IDX  = np.nonzero(pf.build_sometimes_legal_mask().astype(bool))[0]  # (1858,)
N_1858  = len(SL_IDX)

OPTIONS = tf.io.TFRecordOptions(compression_type="GZIP")


def to_wdl(x):
    x = float(x)
    return np.array([max(x, 0.0), 1.0 - abs(x), max(-x, 0.0)], dtype=np.float32)


def blend_policy(policy_1858, legal_1858):
    n_legal = float(legal_1858.sum())
    if n_legal <= 0:
        return policy_1858
    pol = (1.0 - UNIFORM_BLEND) * policy_1858 + UNIFORM_BLEND * (legal_1858 / n_legal)
    s = pol.sum()
    return pol / s if s > 0 else pol


def partition(items, n):
    """Round-robin split into n disjoint lists (some may be empty)."""
    return [items[i::n] for i in range(n)]


def fmt_eta(sec):
    if sec != sec or sec == float("inf"):
        return "??"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def bytes_feature(b):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))


def make_example(lc0_board, xc0_board, policy, wdl, legal):
    feat = {
        "lc0_board": bytes_feature(lc0_board.astype(np.float16).tobytes()),
        "xc0_board": bytes_feature(xc0_board.astype(np.int16).tobytes()),
        "policy": bytes_feature(policy.astype(np.float32).tobytes()),
        "legal": bytes_feature(legal.astype(np.uint8).tobytes()),
        "wdl": tf.train.Feature(float_list=tf.train.FloatList(value=wdl.tolist())),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()


# ---------------------------------------------------------------------------
# lc0 source (single process, sequential, no python-chess)
# ---------------------------------------------------------------------------

def list_lc0_subdirs(lc0_dir):
    out = []
    for d in sorted(os.listdir(lc0_dir)):
        full = os.path.join(lc0_dir, d)
        if os.path.isdir(full):
            out.append(full)
    return out


def chunks_in_subdirs(subdirs):
    chunks = []
    for full in subdirs:
        gz = sorted(f for f in os.listdir(full)
                    if f.endswith(".gz") and not f.endswith(".tfrecord.gz"))
        chunks.extend(os.path.join(full, f) for f in gz)
    return chunks


def cp_convert(parser, rec):
    return parser.inner.convert_v6_to_tuple(rec, emit_lc0=True)


def lc0_stream(subdirs, stop_evt, name):
    chunks = chunks_in_subdirs(subdirs)
    random.shuffle(chunks)
    print(f"[{name}] {len(chunks):,} chunk files from {len(subdirs)} dirs", flush=True)
    parser = ChunkParser(chunks, workers=0, batch_size=1,
                         draw_drop_rate=LC0_DRAW_DROP,
                         diff_focus_min=1.0, diff_focus_slope=0.0)
    for rec in parser.inner.sequential_gen():
        if stop_evt.is_set():
            return
        tokens, mask4288, pol4288, result_wdl, search_wdl, lc0 = cp_convert(parser, rec)
        lc0 = np.asarray(lc0)                                   # already rule50 = count/99
        policy = pol4288[SL_IDX]                                # 4288 -> compact 1858
        legal = mask4288[SL_IDX].astype(np.uint8)
        s = policy.sum()
        if s <= 0:
            continue
        policy = policy / s
        wdl = (Z_BLEND * result_wdl + (1.0 - Z_BLEND) * search_wdl).astype(np.float32)
        yield lc0, np.asarray(tokens), blend_policy(policy, legal), wdl, legal


# ---------------------------------------------------------------------------
# xc0 source (single process, sequential, no python-chess)
# ---------------------------------------------------------------------------

def load_xc0_games_for_tag(rt):
    """Return (shuffled games list, {game_id: sf_sub_df}) for a single run tag.

    One tag's SF frame at a time keeps peak RAM bounded -- tags are drained and
    freed in order rather than all held resident.
    """
    rd = os.path.join(SP_DIR, rt)
    ag = load_game_index(rd)
    ag = [a for a in ag if isinstance(a.get("pkl_file"), str)
          and os.path.exists(a["pkl_file"])]
    for g in ag:
        g["run_tag"] = rt

    with open(os.path.join(rd, ANALYZE_PKL), "rb") as fh:
        prev = pickle.load(fh)
    df_all = prev["df_all"]
    sf_by_gid = {gid: sub for gid, sub in df_all.groupby("game_id", sort=False)}
    df_all["clipped_loss"] = np.clip(df_all["loss"], -1000, 1000)
    cpl = df_all.groupby("game_id")["clipped_loss"].mean()
    dm = prev["df_means"].copy()
    dm["overall_cpl"] = dm.game_id.map(cpl)
    dm["run_tag"] = rt

    meta = pd.DataFrame(ag).drop_duplicates("game_id")
    if "overall_cpl" in meta.columns:
        del meta["overall_cpl"]
    meta = meta.query("plies >= 10").merge(
        dm[["game_id", "run_tag", "overall_cpl"]], on=["game_id", "run_tag"])
    meta = meta.query(f"overall_cpl <= {GAME_CPL_MAX}").copy()

    games = meta.to_dict("records")
    random.shuffle(games)
    print(f"[xc0] {rt}: {len(games):,} games after cpl<= {GAME_CPL_MAX} filter", flush=True)
    return games, sf_by_gid


def load_log(path):
    if path.lower().endswith(".pkl.gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def align_sf(moves, sf_rows):
    """Sequential played_move matching, ply -> row index into sf_rows."""
    sf_rows = sf_rows.copy()
    sf_rows["move_num_int"] = sf_rows["move_num"].astype(int)
    sf_rows = sf_rows.sort_values("move_num_int", kind="stable").reset_index(drop=True)
    by_ply = {}
    j = 0
    for ridx, r in sf_rows.iterrows():
        uci = r.get("played_move", "")
        if not uci:
            continue
        for k in range(j, len(moves)):
            if moves[k] == uci:
                by_ply[k] = ridx
                j = k + 1
                break
    return sf_rows, by_ply


def setup_board(scenario, log, moves, start_fen):
    """Return (board at start_fen, record_from ply) with correct lc0 history.

    push_uci is unchecked (an illegal uci is applied blindly and can corrupt the
    C++ board), so each warm move is verified against legal_moves() first, then
    the reconstruction is confirmed with an exact fen() match; either miss falls
    back to the bare fen.

    - NO_WARM_SCENARIOS: real/doctored root with no prior moves; blank early
      history planes are the genuine lc0 view -> record from ply 0.
    - otherwise: replay history_uci from startpos to reconstruct start_fen with a
      full move stack, so every recorded ply carries its true history. If it does
      not reproduce start_fen, fall back to the bare fen and skip the first
      LC0_HISTORY_PLIES plies rather than feed lc0 falsely-blank planes.
    """
    if scenario in NO_WARM_SCENARIOS:
        return pf.Board(start_fen), 0

    history = log.get("history_uci") or moves
    prefix_len = len(history) - len(moves)
    if prefix_len >= 0:
        board = pf.Board()
        legal_warm = True
        for u in history[:prefix_len]:
            if u not in board.legal_moves():
                legal_warm = False
                break
            board.push_uci(u)
        if legal_warm and board.fen() == start_fen:
            return board, 0

    return pf.Board(start_fen), LC0_HISTORY_PLIES


def iter_xc0_game_records(game, sf_df):
    """Yield (lc0, tokens, policy, wdl, legal) tuples, sequential, pf-only.

    Skips stockfish-played plies and plies with per-move CPL > PLY_CPL_MAX;
    every other ply with MCTS candidate_moves becomes a record.
    """
    log = load_log(game["pkl_file"])
    scenario = log.get("scenario") or game.get("scenario", "")
    if scenario in SKIP_SCENARIOS:
        return
    moves = log.get("moves_played", [])
    if len(moves) < 10:
        return

    start_fen = log.get("start_fen")
    tree_data = {int(k): v for k, v in log.get("tree_search_data", {}).items()}
    result = log.get("result")
    vs_sf = log.get("vs_stockfish", False)
    sf_color = log.get("stockfish_color", None)

    sf_rows, sf_by_ply = (None, {})
    if sf_df is not None and len(sf_df):
        sf_rows, sf_by_ply = align_sf(moves, sf_df)

    board, record_from = setup_board(scenario, log, moves, start_fen)
    for ply, move_played in enumerate(moves):
        if ply < record_from:
            board.push_uci(move_played)
            continue
        is_white_move = board.side_to_move() == "w"
        is_sf_move = vs_sf and sf_color is not None and (
            (is_white_move and sf_color) or (not is_white_move and not sf_color))
        if is_sf_move:
            board.push_uci(move_played)
            continue

        idx = sf_by_ply.get(ply)
        if idx is None:
            board.push_uci(move_played)
            continue
        loss = sf_rows.iloc[idx]["loss"]
        if loss is None or loss > PLY_CPL_MAX:
            board.push_uci(move_played)
            continue

        node = tree_data.get(ply)
        cms = node.get("candidate_moves") if node else None
        if not cms:
            board.push_uci(move_played)
            continue

        visits = [[c["uci"], c["visits"]] for c in cms]
        visited = set(u for u, _ in visits)
        lms = board.legal_moves()
        for u in lms:
            if u not in visited:
                visited.add(u)
                visits.append([u, 1])

        counts = np.array([v for _, v in visits], dtype=np.float32)
        s = counts.sum()
        if s <= 0:
            board.push_uci(move_played)
            continue
        pi = counts / s

        visited_ucis = [u for u, _ in visits]
        indices = board.moves_to_indices(visited_ucis)   # compact 1858, covers all legal moves

        policy = np.zeros(N_1858, dtype=np.float32)
        legal = np.zeros(N_1858, dtype=np.uint8)
        for ci, prob in zip(indices, pi):
            policy[ci] += prob
            legal[ci] = 1

        wdl_node = node.get("best_wdl")
        search_wdl = (np.array(wdl_node, dtype=np.float32) if wdl_node is not None
                      else to_wdl(0.0))
        if result > 0:
            z = 1 if is_white_move else -1
        elif result < 0:
            z = -1 if is_white_move else 1
        else:
            z = 0
        wdl = (Z_BLEND * to_wdl(z) + (1.0 - Z_BLEND) * search_wdl).astype(np.float32)
        policy = blend_policy(policy, legal)

        tokens = np.asarray(board.encode_64_tokens(), dtype=np.int16)
        lc0 = np.asarray(board.lc0_features(), dtype=np.float32)
        lc0[109] /= 99.0
        lc0 = lc0.astype(np.float16)

        yield lc0, tokens, policy, wdl, legal

        board.push_uci(move_played)


def xc0_stream(run_tags, stop_evt):
    for rt in run_tags:
        if stop_evt.is_set():
            return
        games, sf_by_gid = load_xc0_games_for_tag(rt)
        for game in games:
            if stop_evt.is_set():
                return
            sf_df = sf_by_gid.get(game["game_id"])
            for rec in iter_xc0_game_records(game, sf_df):
                yield rec
        games = sf_by_gid = None


# ---------------------------------------------------------------------------
# per-process runner: async writer thread + progress/done reporting
# ---------------------------------------------------------------------------

def run_source(name, source, record_gen, out_dir, target, shared_total, stop_evt,
               log_q, prefix):
    """shared_total is an mp.Value shared across both processes -- the stop
    condition is the COMBINED write count across lc0+xc0, not either source's
    own count. Whichever process's increment crosses `target` sets stop_evt;
    the other notices on its next loop iteration and stops wherever it is."""
    write_q = queue.Queue(WRITER_QUEUE)
    stats = {"shards": 0}

    def writer_fn():
        shard_id = 0
        while True:
            records = write_q.get()
            if records is None:
                break
            path = os.path.join(out_dir, f"{prefix}_{shard_id:05d}.tfrecord.gz")
            with tf.io.TFRecordWriter(path, OPTIONS) as w:
                for r in records:
                    w.write(make_example(*r))
            shard_id += 1
            stats["shards"] = shard_id

    wt = threading.Thread(target=writer_fn, name=f"{name}-writer")
    wt.start()

    begin = time.time()
    buffer = []
    total = 0
    last_report = 0
    reason = "source_exhausted"

    for rec in record_gen:
        if stop_evt.is_set():
            reason = "stopped_by_peer"
            break
        buffer.append(rec)
        total += 1
        if len(buffer) >= RECORDS_PER_FILE:
            write_q.put(buffer)
            buffer = []

        with shared_total.get_lock():
            shared_total.value += 1
            combined = shared_total.value

        if total - last_report >= PROGRESS_EVERY:
            last_report = total
            log_q.put({"name": name, "source": source, "kind": "progress",
                       "total": total})
        if combined >= target:
            reason = "target_reached"
            stop_evt.set()
            break

    if reason == "source_exhausted" and stop_evt.is_set():
        # generator returned empty because it saw stop_evt internally, before
        # this loop got a chance to observe it directly
        reason = "stopped_by_peer"

    if buffer:
        write_q.put(buffer)
    write_q.put(None)
    wt.join()

    elapsed = time.time() - begin
    log_q.put({"name": name, "source": source, "kind": "done", "total": total,
               "shards": stats["shards"], "elapsed": elapsed, "reason": reason})


def lc0_worker(name, subdirs, out_dir, target, shared_total, stop_evt, log_q, prefix):
    os.makedirs(out_dir, exist_ok=True)
    run_source(name, "lc0", lc0_stream(subdirs, stop_evt, name), out_dir, target,
               shared_total, stop_evt, log_q, prefix=prefix)


def xc0_worker(name, run_tags, out_dir, target, shared_total, stop_evt, log_q, prefix):
    os.makedirs(out_dir, exist_ok=True)
    run_source(name, "xc0", xc0_stream(run_tags, stop_evt), out_dir, target,
               shared_total, stop_evt, log_q, prefix=prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tags", nargs="+", default=RUN_TAGS)
    ap.add_argument("--lc0-dir", default=DEFAULT_LC0_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--target", type=int, default=TARGET_TOTAL)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    stop_evt = ctx.Event()
    log_q = ctx.Queue()
    shared_total = ctx.Value("q", 0)

    subdir_groups = partition(list_lc0_subdirs(args.lc0_dir), LC0_PROCS)
    tag_groups = partition(list(args.run_tags), XC0_PROCS)

    procs = {}
    src_of = {}
    for i in range(LC0_PROCS):
        name = f"lc0-{i}"
        src_of[name] = "lc0"
        procs[name] = ctx.Process(
            target=lc0_worker, name=f"{name}-proc",
            args=(name, subdir_groups[i], args.out_dir, args.target,
                  shared_total, stop_evt, log_q, f"ab_{name}"))
    for i in range(XC0_PROCS):
        name = f"xc0-{i}"
        src_of[name] = "xc0"
        procs[name] = ctx.Process(
            target=xc0_worker, name=f"{name}-proc",
            args=(name, tag_groups[i], args.out_dir, args.target,
                  shared_total, stop_evt, log_q, f"ab_{name}"))

    begin = time.time()
    for p in procs.values():
        p.start()

    proc_total = {name: 0 for name in procs}
    done = {name: False for name in procs}
    next_ms = MILESTONE
    last_t, last_comb, last_lc0, last_xc0 = begin, 0, 0, 0

    while not all(done.values()):
        try:
            msg = log_q.get(timeout=1.0)
        except queue.Empty:
            for name, p in procs.items():
                if not done[name] and not p.is_alive():
                    print(f"[{name}] exited without a done message (crash?)", flush=True)
                    done[name] = True
            continue

        proc_total[msg["name"]] = msg["total"]
        if msg["kind"] == "done":
            done[msg["name"]] = True
            print(f"[{msg['name']}] DONE total={msg['total']:,}  "
                  f"shards={msg['shards']}  elapsed={msg['elapsed']:.0f}s  "
                  f"reason={msg['reason']}", flush=True)

        lc0_t = sum(v for n, v in proc_total.items() if src_of[n] == "lc0")
        xc0_t = sum(v for n, v in proc_total.items() if src_of[n] == "xc0")
        combined = lc0_t + xc0_t
        if combined >= next_ms:
            now = time.time()
            dt = max(now - last_t, 1e-9)
            rate = (combined - last_comb) / dt
            rlc0 = (lc0_t - last_lc0) / dt
            rxc0 = (xc0_t - last_xc0) / dt
            eta = (args.target - combined) / rate if rate > 0 else float("inf")
            print(f"[{combined:>10,}] lc0 {lc0_t:>9,}  xc0 {xc0_t:>9,} | "
                  f"tot {rate:,.0f}/s  lc0 {rlc0:,.0f}/s  xc0 {rxc0:,.0f}/s | "
                  f"ETA {fmt_eta(eta)}", flush=True)
            last_t, last_comb, last_lc0, last_xc0 = now, combined, lc0_t, xc0_t
            next_ms = (combined // MILESTONE + 1) * MILESTONE

    for name, p in procs.items():
        p.join(timeout=30)
        if p.is_alive():
            print(f"[{name}] still alive after join timeout, terminating", flush=True)
            p.terminate()

    lc0_final = sum(v for n, v in proc_total.items() if src_of[n] == "lc0")
    xc0_final = sum(v for n, v in proc_total.items() if src_of[n] == "xc0")
    elapsed = time.time() - begin
    print(f"\n[all done] elapsed={elapsed:.0f}s  total={lc0_final + xc0_final:,}  "
          f"lc0={lc0_final:,}  xc0={xc0_final:,}  out={args.out_dir}")
    print(f"[next] shuffle for final mix:\n"
          f"  python scripts/data/shuffle_xc0.py --in-dir {args.out_dir} "
          f"--out-dir {os.path.dirname(args.out_dir)}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
