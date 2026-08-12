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

from xerces_training.uci_to_idx import uci_to_idx as UCI_TO_IDX

from chessbot import SP_DIR
from chessbot.review import load_game_index, ANALYZE_PKL
from chessbot.lc0_utils import lc0_logits_to_xc0_batch
from chessbot.utils import calc_entropy

DEFAULT_LC0_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\lc0"
DEFAULT_OUT_DIR = r"C:\Users\Bryan\Data\chessbot_data\training_data\xc0hK6_071826\staging"

# RUN_TAGS = [
#     "16m_precond_run2", "16m_precond_run1", "16m_precond_run0",
#     "16m_xc0hK6_run4", "16m_xc0hK6_run3", "16m_xc0hK6_run2", "16m_xc0hK6_run1",
#     "precond_run5", "precond_run4", "precond_run3", "precond_run2", "precond_run1",
#     "cfiw_wdl_run2", "cfiw_wdl_run1",
#     "val_test",
# ]

RUN_TAGS = [
    "16m_deep_pretrain_run3", "16m_deep_pretrain_run2", "16m_deep_pretrain_run1",
    "cfiw_pretrained_run5", "cfiw_pretrained_run6", "cfiw_pretrained_run7",
]

TARGET_PER_SOURCE = 20_000_000
LC0_JOBS         = 6
MAX_CONCURRENT_XC0 = 4
RECORDS_PER_FILE = 10_240
Z_BLEND          = 0.5
LC0_DRAW_DROP    = 0.4
DEDUPE_FLOOR     = 0.05
GAME_CPL_MAX     = 20
MOVE_CPL_MAX     = 10
K                = 6

COLLAR_THRESHOLD_CP     = 350
COLLAR_N_CONSEC         = 7
COLLAR_REVERSAL_N_CONSEC = 3

TSCALE_TARGET    = 0.7
TSCALE_MARGIN    = 0.05
TSCALE_MAX_ITERS = 5
TSCALE_LO        = 0.2
TSCALE_HI        = 1.0

SKIP_SCENARIOS     = {"blunder_replay"}
NO_WARM_SCENARIOS  = {"startpos", "piece_odds", "piece_training"}
LC0_FULL_SCENARIOS = {"paired_validation", "UHO"}
LC0_DISTILL_BATCH  = 64  # must match cfg.lc0_distill_batch_size used at selfplay time --
                         # TRT's cache key bakes in opt/max profile shapes, so a mismatched
                         # batch size here silently forces a full recompile every run

SL_IDX   = np.nonzero(pf.build_sometimes_legal_mask().astype(bool))[0]
N_1858   = len(SL_IDX)
OPTIONS  = tf.io.TFRecordOptions(compression_type="GZIP")
V6_STRUC = struct.Struct(V6_STRUCT_STRING)


def resume_shard_id(out_dir, prefix):
    """Next shard_id to write for this prefix, so reruns append instead of
    overwriting whatever's already in out_dir (same approach shuffle_xc0.py
    uses for its own shard files)."""
    marker = prefix + "_"
    existing = [f for f in os.listdir(out_dir)
                if f.startswith(marker) and f.endswith(".tfrecord.gz")]
    if not existing:
        return 0
    nums = []
    for name in existing:
        stem = name[:-len(".tfrecord.gz")]
        try:
            nums.append(int(stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 0


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


MISSING_WDL_WARN_EVERY = 1_000
missing_wdl_ctr = [0]


def detect_has_wdl(games):
    """Called once at worker startup: loads games in order until it finds one
    with tree_search_data, checks for best_wdl on any node, and returns that
    as a fixed answer for the whole run. No re-checking once the worker is
    into its per-position loop."""
    for g in games:
        if not os.path.exists(g["pkl_file"]):
            continue
        log = load_log(g["pkl_file"])
        tree_data = log.get("tree_search_data", {})
        if tree_data:
            return any(n.get("best_wdl") is not None for n in tree_data.values())
    return False


def warn_missing_wdl(run_tag):
    missing_wdl_ctr[0] += 1
    n = missing_wdl_ctr[0]
    if n == 1 or n % MISSING_WDL_WARN_EVERY == 0:
        print(f"[xc0] WARNING: {run_tag} node missing best_wdl on a run detected as "
              f"WDL-head -- skipping xc0 record (count={n:,})", flush=True)


def maybe_temp_scale(pi):
    pi = pi.astype(np.float64)
    ne = calc_entropy(pi, normed_only=True)
    if ne <= TSCALE_TARGET:
        return pi.astype(np.float32)
    lo, hi = TSCALE_LO, TSCALE_HI
    best, best_dist = (lo + hi) / 2, float('inf')
    for _ in range(TSCALE_MAX_ITERS):
        mid    = (lo + hi) / 2
        scaled = pi ** (1.0 / mid)
        scaled /= scaled.sum()
        ne_mid = calc_entropy(scaled, normed_only=True)
        if ne_mid < ne:
            dist = abs(ne_mid - TSCALE_TARGET)
            if dist < best_dist:
                best, best_dist = mid, dist
            if dist < TSCALE_MARGIN:
                break
        if ne_mid < TSCALE_TARGET:
            lo = mid
        else:
            hi = mid
    scaled = pi ** (1.0 / best)
    return (scaled / scaled.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# lc0 distillation batcher
# ---------------------------------------------------------------------------

def lc0_table_index(feat):
    return (int(feat[104, 0, 0]) | int(feat[105, 0, 0])) + 2 * int(feat[108, 0, 0])


class Lc0Batcher:
    def __init__(self, sess, batch_size=256):
        self.sess         = sess
        self.batch_size   = batch_size
        self.pending      = []   # (features_uint8, lc0_idx, xc0_idx)
        self.results      = []   # (lc0_wdl, xc0_policy_1858)
        self.input_name   = sess.get_inputs()[0].name
        self.output_names = [o.name for o in sess.get_outputs()]

    def submit(self, features_uint8, board):
        ti      = lc0_table_index(features_uint8)
        ucis    = board.legal_moves()
        lc0_idx = np.array([UCI_TO_IDX[ti][u.rstrip('n')] for u in ucis], dtype=np.int32)
        xc0_idx = np.array(board.moves_to_indices(ucis), dtype=np.int32)
        self.pending.append((features_uint8, lc0_idx, xc0_idx))
        if len(self.pending) >= self.batch_size:
            self._run_batch()

    def _run_batch(self):
        batch, self.pending = self.pending, []
        feats = np.stack([p[0] for p in batch]).astype(np.float32)
        feats[:, 109] /= 99.0
        outs   = self.sess.run(self.output_names, {self.input_name: feats})
        logits, wdl_batch = outs[0], outs[1]
        policies = lc0_logits_to_xc0_batch(logits, [p[1] for p in batch], [p[2] for p in batch])
        for i, pol in enumerate(policies):
            self.results.append((np.array(wdl_batch[i], dtype=np.float32), pol))

    def flush_and_drain(self):
        if self.pending:
            self._run_batch()
        out, self.results = self.results, []
        return out


# ---------------------------------------------------------------------------
# lc0 source
# ---------------------------------------------------------------------------

def lc0_stream(chunks, stop_evt):
    parser = ChunkParser(chunks, workers=0, batch_size=1,
                         draw_drop_rate=LC0_DRAW_DROP,
                         diff_focus_min=1.0, diff_focus_slope=0.0)
    dummy = np.zeros(64, dtype=np.int64)
    prev_pl  = -1
    game_ply = 0
    for rec in parser.inner.sequential_gen():
        if stop_evt.is_set():
            return
        (ver, ifmt, probs_bytes, planes_bytes,
         us_ooo, us_oo, them_ooo, them_oo, stm, rule50,
         inv, dep,
         root_q, best_q, root_d, best_d, root_m, best_m, pl,
         result_q, result_d, pq, pd, pm, oq, od, om,
         visits, pidx, bidx, r1, r2, r3, r4) = V6_STRUC.unpack(rec)

        if pl > prev_pl:
            game_ply = 0
        else:
            game_ply += 1
        prev_pl = pl

        # Linear 5% at ply 0 -> 100% at ply 15, in even 6.3pt steps. The old
        # form was (ply+1)*0.05 under a `game_ply < 15` guard, which left ply
        # 14 at 75% and then jumped straight to 100% -- a 25pt cliff right
        # where the opening ends. No guard needed now; the ramp reaches 1.0.
        if random.random() >= min(1.0, 0.05 + 0.95 * game_ply / 15.0):
            continue

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

_first_disk_read = [True]


def load_log(path):
    if _first_disk_read[0]:
        _first_disk_read[0] = False
        print(f"[disk] first game read from disk at {time.strftime('%H:%M:%S')}", flush=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def load_xc0_games(rt):
    rd = os.path.join(SP_DIR, rt)
    analyze_path = os.path.join(rd, ANALYZE_PKL)
    if not os.path.exists(analyze_path):
        print(f"[xc0] {rt}: no analyze pkl, skipping", flush=True)
        return [], {}

    games = [g for g in load_game_index(rd) if isinstance(g.get("pkl_file"), str)]

    with open(analyze_path, "rb") as fh:
        prev = pickle.load(fh)
    df_all = prev["df_all"]
    sf_by_gid = {gid: sub for gid, sub in df_all.groupby("game_id", sort=False)}

    random.shuffle(games)
    print(f"[xc0] {rt}: {len(games):,} games", flush=True)
    return games, sf_by_gid


def game_cpl_mean(sf_df):
    if sf_df is None or not len(sf_df):
        return None
    return float(np.clip(sf_df["loss"], -1000, 1000).mean())


def collar_is_noisy(sf_df, result):
    """Detect a decisive-then-reversed swing: one side sustains a >350cp edge
    for 7+ consecutive plies, then either doesn't win the game or gives the
    edge back (crosses to a losing eval for that side) for 3+ straight plies."""
    if sf_df is None or not len(sf_df) or "best_absolute" not in sf_df.columns:
        return False

    evals = sf_df.sort_values("move_num")["best_absolute"].to_numpy()

    holder      = None
    holder_start = None
    run_sign    = 0
    run_count   = 0

    for i, ev in enumerate(evals):
        if ev > COLLAR_THRESHOLD_CP:
            sign = 1
        elif ev < -COLLAR_THRESHOLD_CP:
            sign = -1
        else:
            sign = 0

        if sign != 0 and sign == run_sign:
            run_count += 1
        elif sign != 0:
            run_sign, run_count = sign, 1
        else:
            run_sign, run_count = 0, 0

        if run_count >= COLLAR_N_CONSEC:
            holder, holder_start = ('white' if run_sign == 1 else 'black'), i
            break

    if holder is None:
        return False

    holder_result = 1.0 if holder == 'white' else -1.0
    if float(result) != holder_result:
        return True

    below_count = 0
    for ev in evals[holder_start:]:
        holder_pov = ev if holder == 'white' else -ev
        if holder_pov < 0:
            below_count += 1
            if below_count >= COLLAR_REVERSAL_N_CONSEC:
                return True
        else:
            below_count = 0

    return False


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


def absolute_ply(board):
    """True ply of the position, from the FEN move counters.

    Not board.game_ply(): that is a since-construction counter, so a board
    built from a mid-game FEN reports 0 no matter how deep the position is.
    """
    return (board.fullmove_number() - 1) * 2 + (0 if board.white_to_move() else 1)


def iter_xc0_game(game, sf_df, run_tag, has_wdl, seen=None, lc0_batcher=None, xc0_eligible=True):
    if not os.path.exists(game["pkl_file"]):
        return
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

    lc0_pending  = []   # xc0h boards queued for lc0 inference this game

    for ply, move_played in enumerate(moves):
        is_white   = board.white_to_move()
        is_sf_move = vs_sf and sf_color is not None and (
            (is_white and sf_color) or (not is_white and not sf_color))

        made_xc0      = False
        dedup_skipped = False

        if not is_sf_move and xc0_eligible:
            idx  = sf_by_ply.get(ply)
            loss = sf_rows.iloc[idx]["loss"] if idx is not None else None
            if loss is not None and loss <= MOVE_CPL_MAX:
                node = tree_data.get(ply)
                cms  = node.get("candidate_moves") if node else None
                if cms:
                    take = True
                    if ply < 20 and seen is not None:
                        key  = board.hash()
                        n    = seen.get(key, 0)
                        take = random.random() < max(1.0 - n / 20.0, DEDUPE_FLOOR)
                        if take:
                            seen[key] = n + 1
                        else:
                            dedup_skipped = True
                    if take:
                        visits_map = {c["uci"]: c["visits"] for c in cms}
                        for u in board.legal_moves():
                            if u not in visits_map:
                                visits_map[u] = 1
                        ucis   = list(visits_map.keys())
                        counts = np.array([visits_map[u] for u in ucis], dtype=np.float32)
                        s      = counts.sum()
                        if s > 0:
                            pi = counts / s
                            if loss == 0 and len(ucis) > 5:
                                pi = maybe_temp_scale(pi)
                            indices = board.moves_to_indices(ucis)
                            policy  = np.zeros(N_1858, dtype=np.float32)
                            for ci, prob in zip(indices, pi):
                                policy[ci] += prob
                            wdl_node = node.get("best_wdl")
                            if wdl_node is not None:
                                search_wdl = np.array(wdl_node, dtype=np.float32)[
                                    [2, 1, 0] if not is_white else [0, 1, 2]]
                            elif has_wdl:
                                warn_missing_wdl(run_tag)
                                search_wdl = None
                            else:
                                search_wdl = to_wdl(node.get("Q_stm", 0.0))

                            if search_wdl is not None:
                                z   = (1 if is_white else -1) if result > 0 else (
                                      (-1 if is_white else 1) if result < 0 else 0)
                                wdl = (Z_BLEND * to_wdl(z) + (1.0 - Z_BLEND) * search_wdl).astype(np.float32)
                                xc0h = np.asarray(board.history_tokens(K), dtype=np.int16)
                                yield ('xc0', xc0h, policy, wdl)
                                made_xc0 = True

        # absolute ply, not the loop index: `ply` counts moves played from the
        # game's start_fen, so on a pre-opened game ply 0 is already deep into
        # the game and was being downsampled as if it were an opening.
        gply = absolute_ply(board)
        lc0d_accept = (1.0 if scenario in LC0_FULL_SCENARIOS
                       else min(1.0, 0.05 + gply * 0.95 / 16))
        if not made_xc0 and not dedup_skipped and lc0_batcher is not None and random.random() < lc0d_accept:
            xc0h = np.asarray(board.history_tokens(K), dtype=np.int16)
            lc0_batcher.submit(board.lc0_features(), board)
            lc0_pending.append(xc0h)

        board.push_uci(move_played)

    if lc0_pending:
        for xc0h, (lc0_wdl, lc0_pol) in zip(lc0_pending, lc0_batcher.flush_and_drain()):
            yield ('lc0d', xc0h, lc0_pol, lc0_wdl)


def xc0_stream(run_tag, stop_evt, games, sf_by_gid, has_wdl,
              cpl_hard_skip=None, use_collar_gate=False, lc0_batcher=None, stats=None):
    if stats is None:
        stats = {}
    stats.update(games_hard_skipped=0, games_collar_skipped=0, games_accepted=0,
                xc0_pos=0, lc0d_pos=0)

    seen = {}
    for game in games:
        if stop_evt.is_set():
            return
        if game.get("scenario") in SKIP_SCENARIOS:
            continue
        sf_df    = sf_by_gid.get(game["game_id"])
        game_cpl = game_cpl_mean(sf_df)

        if cpl_hard_skip is not None and game_cpl is not None and game_cpl > cpl_hard_skip:
            stats['games_hard_skipped'] += 1
            continue

        if use_collar_gate and collar_is_noisy(sf_df, game.get("result", 0)):
            stats['games_collar_skipped'] += 1
            continue

        xc0_eligible = game_cpl is not None and game_cpl <= GAME_CPL_MAX
        if not xc0_eligible and lc0_batcher is None:
            continue

        stats['games_accepted'] += 1
        for item in iter_xc0_game(game, sf_df, run_tag, has_wdl, seen, lc0_batcher, xc0_eligible):
            stats['xc0_pos' if item[0] == 'xc0' else 'lc0d_pos'] += 1
            yield item


def xc0_worker(name, run_tag, out_dir, target, shared_total, lc0d_ctr,
              cpl_hard_skip, use_collar_gate, stop_evt, log_q, prefix):
    os.makedirs(out_dir, exist_ok=True)

    # resolved once at worker startup, never re-checked in the per-position loop
    games, sf_by_gid = load_xc0_games(run_tag)
    has_wdl = detect_has_wdl(games)
    print(f"[{name}] {run_tag}: detected {'WDL' if has_wdl else 'scalar-Q'} value head "
          f"-- {'native best_wdl' if has_wdl else 'naive Q_stm mapping'} will be used", flush=True)

    lc0_batcher   = None
    lc0_model     = os.getenv('LC0_DISTILL_MODEL', '')
    lc0_trt_cache = os.getenv('LC0_DISTILL_TRT_CACHE', '')
    if lc0_model and lc0_trt_cache:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        from chessbot.lc0_utils import make_lc0_trt_session
        lc0_onnx = os.path.join(lc0_trt_cache, f'{lc0_model}.onnx')
        sess = make_lc0_trt_session(lc0_onnx, lc0_model, lc0_trt_cache,
                                    opt_batch=LC0_DISTILL_BATCH, max_batch=LC0_DISTILL_BATCH * 2)
        active = sess.get_providers()[0]
        if active != 'TensorrtExecutionProvider':
            raise RuntimeError(f"Lc0Batcher TRT failed, got {active}. Check CUDA/TRT DLLs in PATH.")
        lc0_batcher = Lc0Batcher(sess, batch_size=LC0_DISTILL_BATCH)
        print(f"[{name}] Lc0Batcher: TRT batch_size={LC0_DISTILL_BATCH}", flush=True)

    stats   = {}
    xc0_gen = xc0_stream(run_tag, stop_evt, games, sf_by_gid, has_wdl,
                        cpl_hard_skip, use_collar_gate, lc0_batcher, stats)
    run_source_dual(name, xc0_gen,
                    out_dir, target, shared_total, lc0d_ctr, stop_evt, log_q,
                    xc0_prefix=prefix, lc0d_prefix=f"lc0d_{name}")

    print(f"[{name}] {run_tag} summary  "
          f"xc0_pos={stats.get('xc0_pos', 0):,}  lc0d_pos={stats.get('lc0d_pos', 0):,}  "
          f"games_accepted={stats.get('games_accepted', 0):,}  "
          f"hard_skipped={stats.get('games_hard_skipped', 0):,}  "
          f"collar_skipped={stats.get('games_collar_skipped', 0):,}", flush=True)
    os._exit(0)


# ---------------------------------------------------------------------------
# shared runner: async shard writer + progress reporting
# ---------------------------------------------------------------------------

def run_source_dual(name, record_gen, out_dir, target, shared_total, lc0d_ctr, stop_evt, log_q,
                    xc0_prefix, lc0d_prefix):
    """Like run_source but routes tagged ('xc0'|'lc0d', xc0h, pol, wdl) records to separate shards.
    Only xc0 records count toward the shared target."""
    xc0_wq  = queue.Queue(6)
    lc0d_wq = queue.Queue(6)
    stats   = {"xc0_shards": 0, "lc0d_shards": 0}

    def writer_fn(wq, prefix, shard_key):
        shard_id = resume_shard_id(out_dir, prefix)
        while True:
            records = wq.get()
            if records is None:
                break
            path = os.path.join(out_dir, f"{prefix}_{shard_id:05d}.tfrecord.gz")
            with tf.io.TFRecordWriter(path, OPTIONS) as w:
                for xc0h, pol, wdl in records:
                    w.write(make_example(xc0h, pol, wdl))
            shard_id += 1
            stats[shard_key] = shard_id

    xc0_wt  = threading.Thread(target=writer_fn, args=(xc0_wq,  xc0_prefix,  "xc0_shards"))
    lc0d_wt = threading.Thread(target=writer_fn, args=(lc0d_wq, lc0d_prefix, "lc0d_shards"))
    xc0_wt.start()
    lc0d_wt.start()

    begin       = time.time()
    xc0_buf     = []
    lc0d_buf    = []
    total       = 0
    total_all   = 0
    last_report = 0
    reason      = "source_exhausted"

    for kind, xc0h, pol, wdl in record_gen:
        if stop_evt.is_set():
            reason = "stopped_by_peer"
            break
        total_all += 1
        if kind == 'xc0':
            xc0_buf.append((xc0h, pol, wdl))
            total += 1
            if len(xc0_buf) >= RECORDS_PER_FILE:
                xc0_wq.put(xc0_buf)
                xc0_buf = []
            with shared_total.get_lock():
                shared_total.value += 1
                combined = shared_total.value
            if combined >= target:
                reason = "target_reached"
                stop_evt.set()
                break
        else:
            lc0d_buf.append((xc0h, pol, wdl))
            if len(lc0d_buf) >= RECORDS_PER_FILE:
                lc0d_wq.put(lc0d_buf)
                lc0d_buf = []
            with lc0d_ctr.get_lock():
                lc0d_ctr.value += 1
            with shared_total.get_lock():
                shared_total.value += 1
                combined = shared_total.value
            if combined >= target:
                reason = "target_reached"
                stop_evt.set()
                break
        if total_all - last_report >= 100_000:
            last_report = total_all
            log_q.put({"name": name, "kind": "progress"})

    for buf, wq in [(xc0_buf, xc0_wq), (lc0d_buf, lc0d_wq)]:
        if buf:
            wq.put(buf)
    xc0_wq.put(None)
    lc0d_wq.put(None)
    xc0_wt.join()
    lc0d_wt.join()

    elapsed = time.time() - begin
    log_q.put({"name": name, "kind": "done", "total": total,
               "shards": stats["xc0_shards"] + stats["lc0d_shards"],
               "elapsed": elapsed, "reason": reason})


def run_source(name, record_gen, out_dir, target, shared_total, stop_evt, log_q, prefix):
    write_q = queue.Queue(6)
    stats   = {"shards": 0}

    def writer_fn():
        shard_id = resume_shard_id(out_dir, prefix)
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
    ap.add_argument("--workers",    type=int, default=6,
                    help="total concurrent workers of any type")
    ap.add_argument("--xc0-only",  action="store_true",
                    help="skip the raw lc0 V6-chunk source entirely; only process xc0 selfplay runs")
    ap.add_argument("--cpl-hard-skip", type=float, default=None,
                    help="hard-skip a game entirely (xc0 and lc0d both) if its mean clipped "
                         "CPL exceeds this value")
    ap.add_argument("--use-collar-gate", action="store_true",
                    help="hard-skip a game entirely if a >350cp edge held for 7+ plies is "
                         "either not converted into a win or gives back to a losing eval "
                         "for 3+ consecutive plies")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # pre-compile lc0d TRT engine in the supervisor so workers never race to build it
    lc0_model     = os.getenv('LC0_DISTILL_MODEL', '')
    lc0_trt_cache = os.getenv('LC0_DISTILL_TRT_CACHE', '')
    if lc0_model and lc0_trt_cache:
        from chessbot.lc0_utils import make_lc0_trt_session
        lc0_onnx = os.path.join(lc0_trt_cache, f'{lc0_model}.onnx')
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        print(f"[main] pre-compiling lc0d TRT engine ({lc0_model})...", flush=True)
        _sess = make_lc0_trt_session(lc0_onnx, lc0_model, lc0_trt_cache,
                                     opt_batch=LC0_DISTILL_BATCH, max_batch=LC0_DISTILL_BATCH * 2)
        del _sess
        print(f"[main] lc0d TRT engine ready", flush=True)

    # gather lc0 chunks
    lc0_chunks = []
    if not args.xc0_only:
        for d in sorted(os.listdir(args.lc0_dir)):
            full = os.path.join(args.lc0_dir, d)
            if not os.path.isdir(full):
                continue
            lc0_chunks.extend(
                os.path.join(full, f) for f in os.listdir(full)
                if f.endswith(".gz") and not f.endswith(".tfrecord.gz"))
        print(f"[scan] {len(lc0_chunks):,} lc0 chunk files")
    else:
        print(f"[scan] --xc0-only: skipping raw lc0 source")
    print(f"[scan] {len(args.run_tags)} xc0 run tags: {args.run_tags}")

    ctx = mp.get_context("spawn")
    lc0_stop = ctx.Event()
    xc0_stop = ctx.Event()
    lc0_ctr  = ctx.Value("q", 0)
    xc0_ctr  = ctx.Value("q", 0)
    lc0d_ctr = ctx.Value("q", 0)
    log_q    = ctx.Queue()

    # pending job list: supervisor pops one at a time and spawns a worker
    pending = (
        [("xc0", rt) for rt in args.run_tags] +
        ([] if args.xc0_only else [("lc0", s) for s in partition(lc0_chunks, LC0_JOBS)])
    )

    active = {}   # name -> Process
    seq    = [0]  # monotonic worker id

    def spawn_next():
        active_lc0 = sum(1 for n in active if n.startswith("lc0-"))
        active_xc0 = sum(1 for n in active if n.startswith("xc0-"))
        for idx in range(len(pending)):
            kind, item = pending[idx]
            stop = lc0_stop if kind == "lc0" else xc0_stop
            if stop.is_set():
                pending.pop(idx)
                return spawn_next()
            if kind == "lc0" and active_lc0 >= args.workers - active_xc0:
                continue
            if kind == "xc0" and active_xc0 >= MAX_CONCURRENT_XC0:
                continue
            pending.pop(idx)
            i      = seq[0]; seq[0] += 1
            name   = f"{kind}-{i}"
            ctr    = lc0_ctr if kind == "lc0" else xc0_ctr
            tgt    = args.lc0_target if kind == "lc0" else args.xc0_target
            fn     = lc0_worker if kind == "lc0" else xc0_worker
            extra  = () if kind == "lc0" else (lc0d_ctr, args.cpl_hard_skip, args.use_collar_gate)
            p = ctx.Process(target=fn,
                            args=(name, item, args.out_dir, tgt,
                                  ctr, *extra, stop, log_q, f"{kind}_{i}"))
            p.start()
            active[name] = p
            print(f"[supervisor] spawned {name}  "
                  f"active={len(active)}  lc0={active_lc0+int(kind=='lc0')}  "
                  f"xc0={active_xc0+int(kind=='xc0')}  pending={len(pending)}", flush=True)
            return True
        return False

    begin = time.time()
    for _ in range(args.workers):
        if not spawn_next():
            break
    print(f"[main] {len(active)} workers running  "
          f"lc0_target={args.lc0_target:,}  xc0_target={args.xc0_target:,}", flush=True)

    last_combined = 0
    REPORT_EVERY  = 100_000

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
                lc0_n     = lc0_ctr.value
            with xc0_ctr.get_lock():
                xc0_combo = xc0_ctr.value   # xc0_pure + lc0d combined
            with lc0d_ctr.get_lock():
                lc0d_n    = lc0d_ctr.value
            xc0_n    = xc0_combo - lc0d_n
            combined = lc0_n + xc0_combo    # no double-count
            if combined - last_combined >= REPORT_EVERY:
                last_combined = combined
                elapsed = time.time() - begin
                lc0_rate   = lc0_n   / max(elapsed, 1e-9)
                xc0_rate   = xc0_n   / max(elapsed, 1e-9)
                lc0d_rate  = lc0d_n  / max(elapsed, 1e-9)
                total_rate = combined / max(elapsed, 1e-9)
                total_target = args.xc0_target if args.xc0_only else args.lc0_target + args.xc0_target
                pct = combined / total_target * 100
                print(f"[{combined:,}]  "
                      f"lc0={lc0_n:,} ({lc0_rate:,.0f}/s)  "
                      f"xc0={xc0_n:,} ({xc0_rate:,.0f}/s)  "
                      f"lc0d={lc0d_n:,} ({lc0d_rate:,.0f}/s)  "
                      f"total={total_rate:,.0f}/s  {pct:.1f}%  "
                      f"active={len(active)}  pending={len(pending)}", flush=True)

    for p in active.values():
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    with lc0_ctr.get_lock():
        lc0_final   = lc0_ctr.value
    with xc0_ctr.get_lock():
        xc0_combo   = xc0_ctr.value
    with lc0d_ctr.get_lock():
        lc0d_final  = lc0d_ctr.value
    xc0_final = xc0_combo - lc0d_final
    elapsed = time.time() - begin
    print(f"\n[done] elapsed={elapsed:.0f}s  "
          f"lc0={lc0_final:,}  xc0={xc0_final:,}  lc0d={lc0d_final:,}  "
          f"out={args.out_dir}")
    os._exit(0)


if __name__ == "__main__":
    mp.freeze_support()
    main()
