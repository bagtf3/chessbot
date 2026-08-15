import gc
import gzip
import hashlib
import json
import math
import os
import pickle
import queue
import random
import time

import chess
import numpy as np
import pandas as pd
import yaml

from chessbot.config import Config
from chessbot.game_utils import GameSpec
from chessbot.looper import init_selfplay
from chessbot.mcts_utils import ChessGame
from chessbot.rescore import SFRescoreThread
import chessbot.utils as cbu

VALIDATION_CONFIG_FILENAME = "validation_config.yaml"
HISTORY_FILENAME = "validation_history.jsonl"
V = "[validation]"
DEFAULT_START_DEPTH = 10

PROBE_CONFIG_ENV = "BLUNDER_REPLAY_CONFIG"
PROBE_POOL_ENV = "BLUNDER_POSITIONS"
PROBE_DIRNAME = "blunder_probe"
PROBE_HISTORY_FILENAME = "blunder_probe_history.jsonl"
PROBE_RETIRE_STREAK = 2  # consecutive best-move hits before a position retires
PROBE_MIN_CACHE_DEPTH = 17  # an eval is locked in, and reusable, at this depth
PROBE_EQUIV_CPL = 12  # tighter than the rescorer's equiv band; this is a test
PROBE_MS_STEP = 20  # more SF time next visit when a search misses that depth
PROBE_MS_MAX_MULT = 2  # ...up to this multiple of the baseline budget
P = "[probe]"

SF_TABLE_DEFAULT = [
  {"depth": 1,  "elo": 1575, "name": "SF17d1"},
  {"depth": 2,  "elo": 1745, "name": "SF17d2"},
  {"depth": 3,  "elo": 1837, "name": "SF17d3"},
  {"depth": 4,  "elo": 1984, "name": "SF17d4"},
  {"depth": 5,  "elo": 2092, "name": "SF17d5"},
  {"depth": 6,  "elo": 2245, "name": "SF17d6"},
  {"depth": 7,  "elo": 2376, "name": "SF17d7"},
  {"depth": 8,  "elo": 2567, "name": "SF17d8"},
  {"depth": 9,  "elo": 2759, "name": "SF17d9"},
  {"depth": 10, "elo": 2912, "name": "SF17d10"},
  {"depth": 11, "elo": 3035, "name": "SF17d11"},
  {"depth": 12, "elo": 3137, "name": "SF17d12"},
  {"depth": 13, "elo": 3223, "name": "SF17d13"},
  {"depth": 14, "elo": 3297, "name": "SF17d14"},
  {"depth": 15, "elo": 3363, "name": "SF17d15"},
  {"depth": 16, "elo": 3422, "name": "SF17d16"},
  {"depth": 17, "elo": 3477, "name": "SF17d17"},
  {"depth": 18, "elo": 3529, "name": "SF17d18"},
  {"depth": 20, "elo": 3625, "name": "SF17d20"},
  {"depth": 22, "elo": 3716, "name": "SF17d22"},
  {"depth": 23, "elo": 3761, "name": "SF17d23"},
]

def create_validation_config(cfg, yaml_file=None):
    vcfg = cfg.copy()
    vcfg.sf_table = SF_TABLE_DEFAULT
    vcfg.sf_index = next(
        (i for i, row in enumerate(vcfg.sf_table) if row["depth"] >= DEFAULT_START_DEPTH),
        len(vcfg.sf_table) - 1,
    )
    vcfg.sf_depth = vcfg.sf_table[vcfg.sf_index]['depth']
    vcfg.sf_elo = vcfg.sf_table[vcfg.sf_index]['elo']
    vcfg.consec_over_50 = 0
    vcfg.init_paths()

    if yaml_file is None:
        v_yaml = os.path.join(vcfg.run_dir, VALIDATION_CONFIG_FILENAME)
    else:
        v_yaml = yaml_file
    
    if os.path.exists(v_yaml):
        vcfg.update_via(v_yaml)
        print(f"{V} config updated with local yaml")
    else:
        print(f"{V} no local yaml config found.")

    prev_last = find_last_history_entry_for_run(
        vcfg.run_dir, selfplay_dir=vcfg.selfplay_dir,
        previous_run_tag=vcfg.previous_run_tag
    )

    if prev_last is not None:
        vcfg = continue_depth_from_previous_cfg(vcfg, prev_last)

    format_and_print_validation_info(vcfg, prev_last)
    
    return vcfg


def create_probe_config(cfg, yaml_file=None):
    """
    Config for the blunder-replay probe worker.

    Search params come from the frozen yaml so the metric stays comparable
    across model epochs. Everything else (model path, TRT cache, macro batch)
    is inherited from the live config, so the probe reuses the engine the
    selfplay workers already compiled instead of building its own.
    """
    pcfg = cfg.copy()
    p_yaml = yaml_file or os.environ.get(PROBE_CONFIG_ENV, "")
    if not p_yaml or not os.path.exists(p_yaml):
        raise RuntimeError(f"{P} no frozen probe config at {p_yaml!r}")

    pcfg.update_via(p_yaml)
    pcfg.init_paths()

    # batch shape is a perf knob, not a search knob: keep the live values so
    # the compiled TRT engine still fits
    pcfg.macro_batch = cfg.macro_batch
    pcfg.micro_batch = cfg.micro_batch

    # These describe the model under test, not the search, so they are always
    # the live ones. model_path especially: after prepare_trt it points at the
    # compiled ONNX, but init_paths derives it back to the .ts from run_tag,
    # so restore it last or the probe recompiles its own engine.
    pcfg.encoding_type = cfg.encoding_type
    pcfg.history_K = cfg.history_K
    pcfg.inference_backend = cfg.inference_backend
    pcfg.model_path = cfg.model_path
    pcfg.trt_model_name = cfg.trt_model_name
    pcfg.trt_cache = cfg.trt_cache

    # one move per position, no adjudication, no opponent
    pcfg.probe_mode = True
    pcfg.max_game_length = 0
    pcfg.play_vs_sf_prob = 0.0
    pcfg.use_syzygy = False
    pcfg.allow_resignation = False
    pcfg.use_eval_draw = False
    pcfg.use_material_diff = False

    # no randomness, ever
    pcfg.add_root_noise = False
    pcfg.sample_moves = False
    pcfg.sampleable = {}

    return pcfg


def probe_dir(cfg):
    d = os.path.join(cfg.run_dir, PROBE_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def last_epoch_in_probe_history(path):
    last = None
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            epoch = json.loads(line).get("n_retrains")
            if epoch is not None:
                last = epoch

    return last


def last_probe_epoch(run_dir, selfplay_dir=None, previous_run_tag=None):
    """
    Epoch of the last probe on record, or None if there are none.

    Read once at startup so the cadence survives a restart: with an empty
    history the probe fires on the first round, and after that it waits its
    full interval no matter how many times the process gets bounced. A row
    only lands here once analysis finished, so a probe that died mid-run
    correctly leaves no trace and gets retried.

    A fresh run has no history of its own, so it falls back to the run it
    continues from -- epochs run continuously across a run switch, and the
    position pool is shared, so the cadence should carry over rather than
    restart. Same fallback the validation ladder uses for its SF depth.
    """
    last = last_epoch_in_probe_history(
        os.path.join(run_dir, PROBE_HISTORY_FILENAME)
    )
    if last is not None:
        return last

    if previous_run_tag and selfplay_dir:
        prev = os.path.abspath(os.path.join(selfplay_dir, previous_run_tag))
        last = last_epoch_in_probe_history(
            os.path.join(prev, PROBE_HISTORY_FILENAME)
        )
        if last is not None:
            print(f"{P} continuing cadence from {previous_run_tag} "
                  f"(last probe at epoch {last})")

    return last


def probe_out_path(cfg, n_retrains):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"probe_e{n_retrains}_{stamp}.pkl.gz"
    return os.path.join(probe_dir(cfg), name)


def probe_pool_path(pool_path=None):
    path = pool_path or os.environ.get(PROBE_POOL_ENV, "")
    if not path:
        raise RuntimeError(f"{P} {PROBE_POOL_ENV} is not set")
    return path


def load_probe_pool(pool_path=None):
    """
    The scanned blunder positions. This file is the probe's long-term memory,
    not a static input: every analysis pass writes back what Stockfish learned
    about the position and what the model played there.

    Keyed by a stable hash of the FEN so records survive being appended to,
    re-sampled, or merged from another scan.
    """
    path = probe_pool_path(pool_path)
    if not os.path.exists(path):
        raise RuntimeError(f"{P} no position pool at {path!r}")

    with gzip.open(path, "rb") as f:
        raw = pickle.load(f)

    pool = {}
    for rec in raw:
        rec["pos_id"] = hashlib.sha1(rec["fen"].encode()).hexdigest()[:16]
        pool[rec["pos_id"]] = rec

    if len(pool) != len(raw):
        print(f"{P} pool: dropped {len(raw) - len(pool)} duplicate positions")

    return list(pool.values())


def save_probe_pool(pool, pool_path=None):
    """
    Write the pool back atomically. A probe worker may be reading this file
    while an earlier probe's analysis writes it, and a torn read would take
    down the worker.
    """
    path = probe_pool_path(pool_path)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def probe_specs(cfg, pool, n_positions, seed=0):
    """
    GameSpecs that replay each position from STARTPOS.

    Rebuilding the whole move path (rather than seeding a FEN) keeps
    board.history_size() honest, so the tree lands on the same sims-schedule
    entry the position had in the original game.

    The original error travels in the meta, so a probe file carries its own
    answer key and can be read without the pool it was sampled from.
    """
    n_all = len(pool)
    pool = [r for r in pool if not r.get("retired")]
    if len(pool) != n_all:
        print(f"{P} skipping {n_all - len(pool)} retired positions")

    if n_positions and n_positions < len(pool):
        pool = random.Random(seed).sample(pool, n_positions)

    startpos = chess.Board().fen()
    specs = []
    for rec in pool:
        meta = {
            "pos_id": rec["pos_id"],
            "src_run_tag": rec["run_tag"],
            "src_game_id": rec["game_id"],
            "src_move_num": rec["move_num"],
            "fen": rec["fen"],
            "orig_move": rec["played_move"],
            "orig_best": rec["sf_best_move"],
            "orig_cpl": rec["cpl"],
            "scenario": "blunder_probe",
            "vs_stockfish": False,
            "stockfish_is_white": False,
        }
        specs.append(GameSpec(
            fen=startpos, moves=list(rec["uci_path"]), meta=meta, cfg=cfg,
        ))

    print(f"{P} {len(specs)} positions queued (seed {seed})")
    return specs


def run_probe(cfg, n_positions, seed, out_path, stop_ev=None, msg_q=None):
    """
    Play one move in each sampled position and write a single file.

    Recents and telemetry go to a local queue nobody reads: probe games are a
    validation metric, so they must not reach the game index or the games/hour
    counters. msg_q is the parent's, so pause/unpause still work.
    """
    specs = probe_specs(cfg, load_probe_pool(), n_positions, seed)

    game_q = queue.Queue()
    for spec in specs:
        game_q.put(spec)

    sink = queue.Queue()
    started = time.time()
    with init_selfplay(cfg, sink, sink, msg_q or queue.Queue(),
                       game_queue=game_q) as looper:
        looper.stop_at_empty = True
        looper.run(stop_ev)
        n = looper.write_probe_records(out_path)

    dt = time.time() - started
    short = os.path.join(*out_path.split(os.sep)[-2:])
    print(f"{P} played {n} positions in {dt / 60:.1f} min "
          f"({n / max(dt, 1e-9):.1f} pos/s) -> {short}")
    return n


def cached_deep_score(src, move_uci, min_depth=PROBE_MIN_CACHE_DEPTH):
    """
    Deep score for move_uci from the pool's own cache, or None if unknown.

    Every analysis pass learns two moves for a position: pass 1 names the best
    move and its eval, pass 2 evaluates whatever the model played. Once both
    the best move and this move are on record deep enough, the position needs
    no Stockfish at all.

    Reusability is judged on the depth actually reached, not the time budget
    that was aimed at it -- the same 350ms buys wildly different depths in a
    quiet endgame and a sharp middlegame, and it is the depth that says
    whether the eval is worth trusting.
    """
    evals = src.get("deep_evals") or {}
    depths = src.get("deep_depths") or {}
    best = src.get("deep_best_move")
    if best is None or move_uci not in evals:
        return None

    if (depths.get(best) or 0) < min_depth:
        return None

    if (depths.get(move_uci) or 0) < min_depth:
        return None

    return {
        "deep_best_move": best,
        "deep_best_cp": src["deep_best_cp"],
        "deep_played_cp": evals[move_uci],
        "deep_cpl": src["deep_best_cp"] - evals[move_uci],
        "deep_best_depth": depths.get(best),
        "deep_depth": depths.get(move_uci),
        "from_cache": True,
    }


def ratchet_store(src, move_uci, cp, depth):
    """
    Record a move's eval only when it beats the depth already on file.

    Depth only ever goes up for a position, so a mini-cache entry is never
    replaced by a shallower answer -- a later 320ms search that returns d16
    must not overwrite a d18 result an earlier pass already earned.
    """
    if depth is None:
        return False

    depths = src.setdefault("deep_depths", {})
    if depth <= (depths.get(move_uci) or 0):
        return False

    src.setdefault("deep_evals", {})[move_uci] = cp
    depths[move_uci] = depth
    return True


def probe_board(src):
    """
    Rebuild the position by pushing the whole path from STARTPOS.

    Seeding a bare FEN would hand Stockfish a board with an empty move stack,
    and python-chess sends the stack to the engine as the position command --
    so without it SF cannot see that a move repeats. The rescorer pushes moves
    for the same reason.
    """
    board = chess.Board()
    for uci in src["uci_path"]:
        board.push_uci(uci)
    return board


def run_deep_sf(live, todo, boards, movetime_ms, max_depth, n_threads):
    """Two-pass SF over the live[i] for i in todo. Returns {i: result}."""
    if not todo:
        return {}

    game_q = queue.Queue()
    res_q = queue.Queue()
    acfg = Config()
    acfg.rescore_movetime_ms = movetime_ms

    threads = []
    for _ in range(n_threads):
        th = SFRescoreThread(game_q, res_q, acfg)
        th.max_depth = max_depth
        th.start()
        threads.append(th)

    chunk = 32
    n_chunks = 0
    for start in range(0, len(todo), chunk):
        positions = []
        for i in todo[start:start + chunk]:
            positions.append((i, boards[i], live[i]["probe_move"]))
        game_q.put((n_chunks, positions))
        n_chunks += 1

    scored = {}
    started = time.time()
    for done in range(1, n_chunks + 1):
        gid, results = res_q.get()
        if gid == "__error__":
            for th in threads:
                th.close()
            raise results
        for r in results:
            scored[r["ply_idx"]] = r
        rate = len(scored) / max(time.time() - started, 1e-9)
        print(f"{P} deep SF @{movetime_ms}ms {len(scored)}/{len(todo)} "
              f"({rate:.1f} pos/s)", end="\r")

    for th in threads:
        th.close()
    print()

    return scored


def analyse_probe_file(raw_path, movetime_ms=300, max_depth=20, n_threads=1,
                       n_retrains=None, pool_path=None,
                       min_cache_depth=PROBE_MIN_CACHE_DEPTH,
                       equiv_cpl=PROBE_EQUIV_CPL):
    """
    Deep SF pass over a finished probe file, then fold everything learned back
    into the position pool.

    Only moves the pool has never seen at this budget reach Stockfish, so a
    position that keeps coming up gets cheaper every time. One SF thread by
    default: this runs alongside live selfplay workers and the rescorer, and
    more than one starves them.

    Runs as its own process: the probe worker has already exited and freed the
    GPU, and this is pure Stockfish time that must not block the orchestrator.
    """
    with gzip.open(raw_path, "rb") as f:
        records = pickle.load(f)

    pool = load_probe_pool(pool_path)
    by_id = {r["pos_id"]: r for r in pool}

    live = [r for r in records
            if r.get("probe_move") and r["pos_id"] in by_id]

    todo = []
    for i, rec in enumerate(live):
        src = by_id[rec["pos_id"]]
        hit = cached_deep_score(src, rec["probe_move"], min_cache_depth)
        if hit is None:
            todo.append(i)
        else:
            rec.update(hit)

    budgets = {}
    for i in todo:
        src = by_id[live[i]["pos_id"]]
        budgets.setdefault(src.get("sf_ms") or movetime_ms, []).append(i)

    print(f"{P} {len(live)} moves: {len(live) - len(todo)} known from pool "
          f"at d{min_cache_depth}+, {len(todo)} to Stockfish "
          f"(d{max_depth} x {n_threads}, budgets "
          f"{ {ms: len(v) for ms, v in sorted(budgets.items())} })")

    # one batch per budget: a thread's movetime is shared, so positions on
    # different budgets cannot be interleaved in the same batch
    scored = {}
    for ms in sorted(budgets):
        idx = budgets[ms]
        boards = {i: probe_board(by_id[live[i]["pos_id"]]) for i in idx}
        scored.update(
            run_deep_sf(live, idx, boards, ms, max_depth, n_threads)
        )
        for i in idx:
            live[i]["deep_ms"] = ms

    for i in todo:
        r = scored.get(i)
        if r is None:
            continue
        rec = live[i]
        rec["deep_best_move"] = r["best_uci"]
        rec["deep_best_cp"] = r["best_cp"]
        rec["deep_played_cp"] = r["played_cp"]
        rec["deep_cpl"] = r["best_cp"] - r["played_cp"]
        rec["deep_best_depth"] = r["best_depth"]
        rec["deep_depth"] = r["depth"]
        rec["from_cache"] = False

    rows = []
    n_retired = 0
    for rec in live:
        if "deep_cpl" not in rec:
            continue

        src = by_id[rec["pos_id"]]
        found_best = rec["probe_move"] == rec["deep_best_move"]

        if not rec["from_cache"]:
            # both moves this search resolved, each kept only if it went
            # deeper than whatever that move already had on file
            ratchet_store(src, rec["deep_best_move"], rec["deep_best_cp"],
                          rec["deep_best_depth"])
            ratchet_store(src, rec["probe_move"], rec["deep_played_cp"],
                          rec["deep_depth"])

            depths = src.get("deep_depths") or {}
            prev_best = src.get("deep_best_move")
            if (rec["deep_best_depth"] or 0) >= (depths.get(prev_best) or 0):
                src["deep_best_move"] = rec["deep_best_move"]
                src["deep_best_cp"] = rec["deep_best_cp"]

            # if this search still did not lock in min_cache_depth, buy more
            # time for the next visit: rerunning the same budget on the same
            # position lands in the same place and never gets there
            locked = min(rec["deep_best_depth"] or 0, rec["deep_depth"] or 0)
            if locked < min_cache_depth:
                src["sf_ms"] = min(rec["deep_ms"] + PROBE_MS_STEP,
                                   movetime_ms * PROBE_MS_MAX_MULT)

        src.setdefault("probes", []).append({
            "epoch": n_retrains,
            "move": rec["probe_move"],
            "cpl": rec["deep_cpl"],
            "found_best": found_best,
        })

        # a position the model has solved PROBE_RETIRE_STREAK times running
        # has nothing left to say; retire it so the sample keeps its teeth
        src["streak"] = src.get("streak", 0) + 1 if found_best else 0
        if src["streak"] >= PROBE_RETIRE_STREAK:
            src["retired"] = True
            n_retired += 1

        rows.append({
            "pos_id": rec["pos_id"],
            "src_run_tag": rec["src_run_tag"],
            "orig_move": rec["orig_move"],
            "orig_cpl": rec["orig_cpl"],
            "probe_move": rec["probe_move"],
            "probe_cpl": rec["deep_cpl"],
            "deep_best": rec["deep_best_move"],
            "same_move": rec["probe_move"] == rec["orig_move"],
            "found_best": found_best,
            "found_equiv": rec["deep_cpl"] <= equiv_cpl,
            "best_depth": rec["deep_best_depth"],
            "probe_depth": rec["deep_depth"],
            "from_cache": rec["from_cache"],
            "streak": src["streak"],
            "sims": rec["search"].get("sims"),
            "stop_reason": rec["search"].get("stop_reason"),
        })

    with gzip.open(raw_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_probe_pool(pool, pool_path)
    n_live = sum(1 for r in pool if not r.get("retired"))
    print(f"{P} pool updated: {n_retired} newly retired, {n_live} still live")

    df = pd.DataFrame(rows)
    if not len(df):
        print(f"{P} nothing scored")
        return None

    df.to_csv(raw_path.replace(".pkl.gz", ".csv"), index=False)
    summary = {
        "ts": int(time.time()),
        "n_retrains": n_retrains,
        "n_positions": len(df),
        "orig_cpl": df["orig_cpl"].mean(),
        "probe_cpl": df["probe_cpl"].mean(),
        "improved": (df["probe_cpl"] < df["orig_cpl"]).mean(),
        "worsened": (df["probe_cpl"] > df["orig_cpl"]).mean(),
        "same_move": df["same_move"].mean(),
        "found_best": df["found_best"].mean(),
        "found_equiv": df["found_equiv"].mean(),
        "best_depth": df["best_depth"].mean(),
        "probe_depth": df["probe_depth"].mean(),
        "cache_hit": df["from_cache"].mean(),
        "n_retired": n_retired,
        "n_pool_live": n_live,
        "file": os.path.basename(raw_path),
    }

    hist = os.path.join(os.path.dirname(os.path.dirname(raw_path)),
                        PROBE_HISTORY_FILENAME)
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, default=float) + "\n")

    print(f"{P} n={summary['n_positions']} "
          f"cpl {summary['orig_cpl']:.1f} -> {summary['probe_cpl']:.1f}  "
          f"improved {summary['improved']:.1%}  "
          f"same_move {summary['same_move']:.1%}  "
          f"found_best {summary['found_best']:.1%}  "
          f"found_equiv {summary['found_equiv']:.1%}  "
          f"depth {summary['best_depth']:.1f}  "
          f"cache_hit {summary['cache_hit']:.1%}")
    return summary


def find_last_history_entry_for_run(run_dir, selfplay_dir=None, previous_run_tag=None):
    """
    Return the last JSON object from run_dir/HISTORY_FILENAME. If absent and
    previous_run_tag is given, try previous_run_tag under selfplay_dir.
    """
    hist_path = os.path.join(run_dir, HISTORY_FILENAME)

    def last_from_path(p):
        last_obj = None
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    last_obj = json.loads(s)
                except Exception:
                    continue
        return last_obj

    last = last_from_path(hist_path)
    if last is not None:
        # found in current run
        return last

    # fallback to previous run if provided
    if previous_run_tag and selfplay_dir:
        prev_run_dir = os.path.abspath(os.path.join(selfplay_dir, previous_run_tag))
        prev_hist = os.path.join(prev_run_dir, HISTORY_FILENAME)
        prev_last = last_from_path(prev_hist)
        if prev_last is not None:
            # helpful debug print like the old class did
            print(f"{V} using last history from: {previous_run_tag}")
            return prev_last

    print(f"{V} no validation history found.")
    return None


def continue_depth_from_previous_cfg(cfg, last_entry):
    """
    If last_entry contains a depth or depth_index, set cfg.sf_index to
    the closest matching index in cfg.sf_table. Leaves cfg unchanged if
    nothing usable found.
    """
    prev_depth = last_entry.get("depth", 0)
    prev_bumped = last_entry["bumped"]
    target_depth = prev_depth + (1 if prev_bumped else 0)

    n_rows = len(cfg.sf_table)
    new_index = n_rows - 1
    for i, row in enumerate(cfg.sf_table):
        if row["depth"] >= target_depth:
            new_index = i
            break

    cfg.sf_index = new_index
    cfg.sf_depth = cfg.sf_table[cfg.sf_index]['depth']
    cfg.sf_elo = cfg.sf_table[cfg.sf_index]['elo']
    cfg.consec_over_50 = last_entry['consec_over_50']
    return cfg


def format_and_print_validation_info(cfg, prev_last):
    """
    Print the two status lines (SF line, Xerces line) using fields on cfg.
    """
    if prev_last is not None:
        sfd = prev_last.get('depth', 0)
        sfe = prev_last['sf_elo']
        msc = prev_last['score']
        melo = prev_last['model_elo']
        
        print(
            f"{V} prev run: SF depth: {sfd} SF elo: {sfe} | "
            f"Xerces elo (score): {melo} ({msc})"
        )

    sfd = cfg.sf_depth
    sfe = cfg.sf_elo
    over50 = cfg.consec_over_50
    print(f"{V} curr run: SF depth: {sfd} SF elo: {sfe} | Consec over 0.5: {over50}")


def paired_validation_games(cfg):
    """
    Return paired ChessGame instances for Stockfish validation.

    For each i in range(n):
      - sample a UHO board using GameGenerator
      - clone it
      - create two ChessGame objects where stockfish plays one as white
        and the other as black
    """
    gen = cbu.GameGenerator(cfg)

    games = []
    n = cfg.n_games // 2
    for i in range(n):
        board_white, _ = gen.new_board(game_type="UHO")
        board_black = board_white.clone()

        meta_w = {
            "vs_stockfish": True,
            "stockfish_is_white": True,
            "scenario": "paired_validation"
        }

        meta_b = {
            "vs_stockfish": True,
            "stockfish_is_white": False,
            "scenario": "paired_validation"
        }

        cg_w = ChessGame(board=board_white, meta=meta_w, cfg=cfg)
        cg_b = ChessGame(board=board_black, meta=meta_b, cfg=cfg)

        games.append(cg_w)
        games.append(cg_b)

    return games


def estimate_model_elo(sf_elo, score):
    """
    Estimate model Elo given opponent (sf) Elo and observed score in [0,1].
    Uses the standard Elo expectation inversion:
      E = 1 / (1 + 10^{(R_op - R_model)/400})
    Solving for R_model gives:
      R_model = R_op - 400 * log10(1/E - 1)
    Handles edge cases by clamping score into (eps, 1-eps).
    """
    eps = 1e-6
    s = min(max(score, eps), 1.0 - eps)
    # invert expectation to rating
    return sf_elo - 400.0 * math.log10((1.0 / s) - 1.0)


def build_validation_summary(looper):
    """
    Build validation summary and apply the two-consecutive >50% bump rule.

    Updates the in-memory cfg (ValidationConfig instance) but does not
    write a validation_config file. Only history JSONL is appended.
    """
    recent = looper.recent_games
    sf_games = [g for g in recent if g.get("vs_stockfish")]
    n = len(sf_games)
    if n == 0:
        raise RuntimeError("no stockfish games in looper.recent_games")

    wins = 0
    draws = 0

    for g in sf_games:
        res = g.get("result")
        sf_white = g.get("stockfish_is_white", g.get("stockfish_color", False))
        if res == 0:
            draws += 1
            continue
        if res > 0:
            if not sf_white:
                wins += 1
        elif res < 0:
            if sf_white:
                wins += 1

    score = (wins + 0.5 * draws) / n

    cfg_dict = looper.config.to_dict()
    sf_elo = cfg_dict['sf_elo']
    table = cfg_dict["sf_table"]
    index = cfg_dict["sf_index"]
    entry = table[index]

    model_elo = estimate_model_elo(sf_elo, score)

    # minimal gating: require at least min_games to count this run
    min_games = 30
    run_counts = (n >= min_games) and (score > 0.5)

    # update consecutive counter
    consec = int(cfg_dict.get("consec_over_50", 0))
    if run_counts:
        consec = consec + 1
    else:
        consec = 0

    bumped = False
    action = "none"

    dominant = (n >= min_games) and (score > 0.8)

    if dominant or consec >= 2:
        if index < (len(table) - 1):
            bumped = True
            action = "bumped_depth_dominant" if dominant else "bumped_depth"
            consec = 0
        else:
            action = "at_max_depth"

    # Do NOT persist the validation config file. Only append history.
    summary = {
        "ts": int(time.time()),
        "run_tag": cfg_dict.get("run_tag"),
        "n_games": n,
        "wins": wins,
        "draws": draws,
        "score": np.round(score, 4),
        "sf_elo": sf_elo,
        "model_elo": np.round(model_elo, 2),
        "should_count": bool(run_counts),
        "consec_over_50": int(consec),
        "bumped": bool(bumped),
        "action": action,
        "depth": int(table[index]["depth"]),
        "depth_index": int(index)
    }

    losses = n - wins - draws
    print(f"[sf results] W/D/L {wins}/{draws}/{losses} score: {score:.3f}")
    print(f"[sf results] sf_elo {sf_elo}  model_elo {model_elo:.1f}  bumped {bumped}")

    # append to JSONL history (this will create the file if needed)
    append_validation_summary(looper.config.run_dir, summary)
    return summary



def append_validation_summary(run_dir, summary):
    """
    Append JSONL history and atomically write latest JSON.
    """
    os.makedirs(run_dir, exist_ok=True)

    hist_path = os.path.join(run_dir, HISTORY_FILENAME)

    with open(hist_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
