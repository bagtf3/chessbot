"""build_bootstrap_records.py -- Build xc0h-K6 pretraining tfrecords.

Two sources run independently to 20M records each, 3 workers per source:
  lc0   V6 binary chunks -> v6_planes_to_xc0h (direct from bits)
  xc0   selfplay game logs -> board.history_tokens(K=6) via pyfastchess

Schema per record:
  xc0h_board  int16[393]    K=6 history frames + meta
  policy      float32[1858] normalized visit distribution
  legal       uint8[1858]
  wdl         float32[3]    0.5*z + 0.5*search_wdl

Usage:
    python scripts/data/build_bootstrap_records.py
"""
import argparse
import gzip
import multiprocessing as mp
import os
import pickle
import queue
import random
import struct
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import pyfastchess as pf

from xerces_training.chunkparser import ChunkParser, V6_STRUCT_STRING
from xerces_training.parse_utils import v6_planes_to_xc0h, get_policy_vector

from chessbot import SP_DIR
from chessbot.review import load_game_index, ANALYZE_PKL

DEFAULT_LC0_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\lc0"
DEFAULT_OUT_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\xc0hK6_070526\staging"

RUN_TAGS = [
    "16m_precond_run2", "16m_precond_run1", "16m_precond_run0",
    "precond_run5", "precond_run4", "precond_run3", "precond_run2", "precond_run1",
    "cfiw_wdl_run2", "cfiw_wdl_run1",
]

TARGET_PER_SOURCE = 20_000_000
LC0_WORKERS      = 3
XC0_WORKERS      = 4
RECORDS_PER_FILE = 10_240
Z_BLEND          = 0.5
LC0_DRAW_DROP    = 0.5
GAME_CPL_MAX     = 20
MOVE_CPL_MAX     = 15
K                = 6

SKIP_SCENARIOS    = {"blunder_replay"}
NO_WARM_SCENARIOS = {"startpos", "piece_odds", "piece_training"}

SL_IDX   = np.nonzero(pf.build_sometimes_legal_mask().astype(bool))[0]
N_1858   = len(SL_IDX)
OPTIONS  = tf.io.TFRecordOptions(compression_type="GZIP")
V6_STRUC = struct.Struct(V6_STRUCT_STRING)


def bytes_feature(b):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))


def make_example(xc0h_board, policy, wdl):
    feat = {
        "xc0h_board": bytes_feature(xc0h_board.tobytes()),
        "policy":     bytes_feature(policy.tobytes()),
        "wdl":        tf.train.Feature(float_list=tf.train.FloatList(value=wdl.tolist())),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()


def to_wdl(z):
    z = float(z)
    return np.array([max(z, 0.0), 1.0 - abs(z), max(-z, 0.0)], dtype=np.float32)


def partition(items, n):
    return [items[i::n] for i in range(n)]


# ---------------------------------------------------------------------------
# lc0 source
# ---------------------------------------------------------------------------

def lc0_stream(chunks, stop_evt):
    parser = ChunkParser(chunks, workers=0, batch_size=1,
                         draw_drop_rate=LC0_DRAW_DROP,
                         diff_focus_min=1.0, diff_focus_slope=0.0)
    dummy = np.zeros(64, dtype=np.int64)
    for rec in parser.inner.sequential_gen():
        if stop_evt.is_set():
            return
        (ver, ifmt, probs_bytes, planes_bytes,
         us_ooo, us_oo, them_ooo, them_oo, stm, rule50,
         inv, dep,
         root_q, best_q, root_d, best_d, root_m, best_m, pl,
         result_q, result_d, pq, pd, pm, oq, od, om,
         visits, pidx, bidx, r1, r2, r3, r4) = V6_STRUC.unpack(rec)

        probs = np.frombuffer(probs_bytes, dtype=np.float32)
        pol4288, _ = get_policy_vector(probs, stm, us_ooo, us_oo, dummy)
        policy = pol4288[SL_IDX]
        s = policy.sum()
        if s <= 0:
            continue
        policy = policy / s

        result_wdl = np.array([0.5*(1-result_d+result_q), result_d, 0.5*(1-result_d-result_q)],
                               dtype=np.float32)
        search_wdl = np.array([0.5*(1-best_d+best_q), best_d, 0.5*(1-best_d-best_q)],
                               dtype=np.float32)
        wdl = (Z_BLEND * result_wdl + (1.0 - Z_BLEND) * search_wdl).astype(np.float32)

        xc0h = v6_planes_to_xc0h(planes_bytes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50, K)
        yield xc0h, policy, wdl


def lc0_worker(name, chunks, out_dir, target, shared_total, stop_evt, log_q, prefix):
    os.makedirs(out_dir, exist_ok=True)
    random.shuffle(chunks)
    run_source(name, lc0_stream(chunks, stop_evt),
               out_dir, target, shared_total, stop_evt, log_q, prefix)
    os._exit(0)


# ---------------------------------------------------------------------------
# xc0 source
# ---------------------------------------------------------------------------

def load_log(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def load_xc0_games(rt):
    rd = os.path.join(SP_DIR, rt)
    analyze_path = os.path.join(rd, ANALYZE_PKL)
    if not os.path.exists(analyze_path):
        print(f"[xc0] {rt}: no analyze pkl, skipping", flush=True)
        return [], {}

    games = load_game_index(rd)
    games = [g for g in games if isinstance(g.get("pkl_file"), str)
             and os.path.exists(g["pkl_file"])]

    with open(analyze_path, "rb") as fh:
        prev = pickle.load(fh)
    df_all = prev["df_all"]
    df_all = df_all.copy()
    df_all["clipped_loss"] = np.clip(df_all["loss"], -1000, 1000)
    game_cpl = df_all.groupby("game_id")["clipped_loss"].mean()
    good_gids = set(game_cpl[game_cpl <= GAME_CPL_MAX].index)

    games = [g for g in games if g["game_id"] in good_gids]
    sf_by_gid = {gid: sub for gid, sub in df_all.groupby("game_id", sort=False)}

    random.shuffle(games)
    print(f"[xc0] {rt}: {len(games):,} games after CPL<={GAME_CPL_MAX} filter", flush=True)
    return games, sf_by_gid


def align_sf(moves, sf_rows):
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


def iter_xc0_game(game, sf_df):
    log = load_log(game["pkl_file"])
    scenario = log.get("scenario") or game.get("scenario", "")
    if scenario in SKIP_SCENARIOS:
        return

    moves = log.get("moves_played", [])
    if len(moves) < 10:
        return

    start_fen = log.get("start_fen")
    tree_data = {int(k): v for k, v in log.get("tree_search_data", {}).items()}
    result    = log.get("result", 0)
    vs_sf     = log.get("vs_stockfish", False)
    sf_color  = log.get("stockfish_color", None)

    sf_rows, sf_by_ply = (None, {})
    if sf_df is not None and len(sf_df):
        sf_rows, sf_by_ply = align_sf(moves, sf_df)

    # warm history for opening scenarios; startpos/piece_odds/piece_training start fresh
    if scenario in NO_WARM_SCENARIOS:
        board = pf.Board(start_fen)
    else:
        history = log.get("history_uci") or moves
        prefix_len = len(history) - len(moves)
        board = pf.Board()
        warmed = False
        if prefix_len >= 0:
            ok = True
            for u in history[:prefix_len]:
                if u not in board.legal_moves():
                    ok = False
                    break
                board.push_uci(u)
            if ok and board.fen() == start_fen:
                warmed = True
        if not warmed:
            board = pf.Board(start_fen)

    for ply, move_played in enumerate(moves):
        is_white = board.side_to_move() == "w"
        is_sf_move = vs_sf and sf_color is not None and (
            (is_white and sf_color) or (not is_white and not sf_color))
        if is_sf_move:
            board.push_uci(move_played)
            continue

        idx = sf_by_ply.get(ply)
        if idx is None:
            board.push_uci(move_played)
            continue
        loss = sf_rows.iloc[idx]["loss"]
        if loss is None or loss > MOVE_CPL_MAX:
            board.push_uci(move_played)
            continue

        node = tree_data.get(ply)
        cms  = node.get("candidate_moves") if node else None
        if not cms:
            board.push_uci(move_played)
            continue

        visits_map = {c["uci"]: c["visits"] for c in cms}
        for u in board.legal_moves():
            if u not in visits_map:
                visits_map[u] = 1

        ucis   = list(visits_map.keys())
        counts = np.array([visits_map[u] for u in ucis], dtype=np.float32)
        s = counts.sum()
        if s <= 0:
            board.push_uci(move_played)
            continue
        pi = counts / s

        indices = board.moves_to_indices(ucis)
        policy  = np.zeros(N_1858, dtype=np.float32)
        for ci, prob in zip(indices, pi):
            policy[ci] += prob

        wdl_node = node.get("best_wdl")
        if wdl_node is not None:
            search_wdl = np.array(wdl_node, dtype=np.float32)
            if not is_white:
                search_wdl = search_wdl[[2, 1, 0]]
        else:
            search_wdl = to_wdl(0.0)
        z = (1 if is_white else -1) if result > 0 else (
            (-1 if is_white else 1) if result < 0 else 0)
        wdl = (Z_BLEND * to_wdl(z) + (1.0 - Z_BLEND) * search_wdl).astype(np.float32)

        xc0h = np.asarray(board.history_tokens(K), dtype=np.int16)
        yield xc0h, policy, wdl

        board.push_uci(move_played)


def xc0_stream(run_tag, stop_evt):
    games, sf_by_gid = load_xc0_games(run_tag)
    for game in games:
        if stop_evt.is_set():
            return
        sf_df = sf_by_gid.get(game["game_id"])
        yield from iter_xc0_game(game, sf_df)


def xc0_worker(name, run_tag, out_dir, target, shared_total, stop_evt, log_q, prefix):
    os.makedirs(out_dir, exist_ok=True)
    run_source(name, xc0_stream(run_tag, stop_evt),
               out_dir, target, shared_total, stop_evt, log_q, prefix)
    os._exit(0)


# ---------------------------------------------------------------------------
# shared runner: async shard writer + progress reporting
# ---------------------------------------------------------------------------

def run_source(name, record_gen, out_dir, target, shared_total, stop_evt, log_q, prefix):
    write_q = queue.Queue(6)
    stats   = {"shards": 0}

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

    begin      = time.time()
    buffer     = []
    total      = 0
    last_report = 0
    reason     = "source_exhausted"

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
        if total - last_report >= 10_000:
            last_report = total
            log_q.put({"name": name, "kind": "progress"})
        if combined >= target:
            reason = "target_reached"
            stop_evt.set()
            break

    if buffer:
        write_q.put(buffer)
    write_q.put(None)
    wt.join()

    elapsed = time.time() - begin
    log_q.put({"name": name, "kind": "done", "total": total,
               "shards": stats["shards"], "elapsed": elapsed, "reason": reason})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc0-dir",    default=DEFAULT_LC0_DIR)
    ap.add_argument("--out-dir",    default=DEFAULT_OUT_DIR)
    ap.add_argument("--run-tags",   nargs="+", default=RUN_TAGS)
    ap.add_argument("--lc0-target", type=int, default=TARGET_PER_SOURCE)
    ap.add_argument("--xc0-target", type=int, default=TARGET_PER_SOURCE)
    ap.add_argument("--workers",    type=int, default=LC0_WORKERS + XC0_WORKERS,
                    help="total concurrent workers of any type")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # gather lc0 chunks
    lc0_chunks = []
    for d in sorted(os.listdir(args.lc0_dir)):
        full = os.path.join(args.lc0_dir, d)
        if not os.path.isdir(full):
            continue
        lc0_chunks.extend(
            os.path.join(full, f) for f in os.listdir(full)
            if f.endswith(".gz") and not f.endswith(".tfrecord.gz"))
    print(f"[scan] {len(lc0_chunks):,} lc0 chunk files")
    print(f"[scan] {len(args.run_tags)} xc0 run tags: {args.run_tags}")

    ctx = mp.get_context("spawn")
    lc0_stop = ctx.Event()
    xc0_stop = ctx.Event()
    lc0_ctr  = ctx.Value("q", 0)
    xc0_ctr  = ctx.Value("q", 0)
    log_q    = ctx.Queue()

    # pending job list: supervisor pops one at a time and spawns a worker
    pending = (
        [("lc0", s) for s in partition(lc0_chunks, LC0_WORKERS)] +
        [("xc0", rt) for rt in args.run_tags]
    )

    active = {}   # name -> Process
    seq    = [0]  # monotonic worker id

    def spawn_next():
        while pending:
            kind, item = pending[0]
            stop = lc0_stop if kind == "lc0" else xc0_stop
            if stop.is_set():
                pending.pop(0)
                continue
            pending.pop(0)
            i      = seq[0]; seq[0] += 1
            name   = f"{kind}-{i}"
            ctr    = lc0_ctr if kind == "lc0" else xc0_ctr
            tgt    = args.lc0_target if kind == "lc0" else args.xc0_target
            fn     = lc0_worker if kind == "lc0" else xc0_worker
            p = ctx.Process(target=fn,
                            args=(name, item, args.out_dir, tgt,
                                  ctr, stop, log_q, f"{kind}_{i}"))
            p.start()
            active[name] = p
            print(f"[supervisor] spawned {name}  "
                  f"active={len(active)}  pending={len(pending)}", flush=True)
            return True
        return False

    begin = time.time()
    for _ in range(args.workers):
        if not spawn_next():
            break
    print(f"[main] {len(active)} workers running  "
          f"lc0_target={args.lc0_target:,}  xc0_target={args.xc0_target:,}", flush=True)

    last_combined = 0
    REPORT_EVERY  = 50_000

    while active:
        try:
            msg = log_q.get(timeout=2.0)
        except queue.Empty:
            for name, p in list(active.items()):
                if not p.is_alive():
                    print(f"[{name}] exited without done msg", flush=True)
                    del active[name]
                    spawn_next()
            continue

        if msg["kind"] == "done":
            active.pop(msg["name"], None)
            print(f"[{msg['name']}] done  total={msg['total']:,}  "
                  f"shards={msg['shards']}  elapsed={msg['elapsed']:.0f}s  "
                  f"reason={msg['reason']}", flush=True)
            spawn_next()
        elif msg["kind"] == "progress":
            with lc0_ctr.get_lock():
                lc0_n = lc0_ctr.value
            with xc0_ctr.get_lock():
                xc0_n = xc0_ctr.value
            combined = lc0_n + xc0_n
            if combined - last_combined >= REPORT_EVERY:
                last_combined = combined
                elapsed = time.time() - begin
                lc0_rate   = lc0_n / max(elapsed, 1e-9)
                xc0_rate   = xc0_n / max(elapsed, 1e-9)
                total_rate = combined / max(elapsed, 1e-9)
                pct = combined / (args.lc0_target + args.xc0_target) * 100
                print(f"[{combined:,}]  "
                      f"lc0={lc0_n:,} ({lc0_rate:,.0f}/s)  "
                      f"xc0={xc0_n:,} ({xc0_rate:,.0f}/s)  "
                      f"total={total_rate:,.0f}/s  {pct:.1f}%  "
                      f"active={len(active)}  pending={len(pending)}", flush=True)

    for p in active.values():
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    with lc0_ctr.get_lock():
        lc0_final = lc0_ctr.value
    with xc0_ctr.get_lock():
        xc0_final = xc0_ctr.value
    elapsed = time.time() - begin
    print(f"\n[done] elapsed={elapsed:.0f}s  lc0={lc0_final:,}  xc0={xc0_final:,}  "
          f"out={args.out_dir}")
    os._exit(0)


if __name__ == "__main__":
    mp.freeze_support()
    main()
