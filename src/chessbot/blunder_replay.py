"""
Shared blunder-replay pool primitives and candidate rules.

Split out from validation.py (which stays focused on the paired-validation
ladder vs Stockfish) so both scan_blunder_positions.py and rescore.py's live
candidate ingestion share one rule definition and one pool format, without
rescore.py needing to import anything from validation.py -- which imports
SFRescoreThread from rescore.py, a cycle.
"""
import gzip
import json
import os
import pickle
import random
import time

import chess
import numpy as np

from chessbot.game_utils import GameSpec
from chessbot.game_utils import short_fen

POOL = "[pool]"
P = "[probe]"

BLUNDER_POOL_ENV = "BLUNDER_POSITIONS"
BLUNDER_REPLAY_CONFIG_ENV = "BLUNDER_REPLAY_CONFIG"
BLUNDER_REPLAY_DIRNAME = "blunder_probe"
BLUNDER_REPLAY_HISTORY_FILENAME = "blunder_probe_history.jsonl"
BLUNDER_REPLAY_MIN_CACHE_DEPTH = 17
BLUNDER_REPLAY_EQUIV_CPL = 12
PROBE_MS_STEP = 20
PROBE_MS_MAX_MULT = 2
# above this many live positions, stop admitting new candidates -- just drop
# them, don't evict anything to make room. More eviction pressure is coming
# later; for now the pool simply stops growing.
BLUNDER_POOL_MAX_SIZE = 128_000
# spread this many replay positions across each selfplay round, metered
# against real games queued/completions so far -- not a retrain-cadence flood
BLUNDER_REPLAY_PER_ROUND = 4096

# candidate rules: V = best_cp, P = played_cp, cpl = V - P (post equiv/mate
# adjustment), all STM-POV. Z_stm is the result from the mover's own
# perspective, since xc0 plays both sides in selfplay.
A_ABS_V = 200
A_MIN_CPL = 60
B_MIN_V = 120
B_MAX_P = 50
G_MIN_V = 300
G_MAX_P = 100
F_MIN_V = 1200
F_MAX_P = 1000

BLUNDER_SCENARIOS = ("startpos", "UHO", "pre_opened", "pre_opened_mini")


def is_blunder_candidate(v, p, cpl, z_stm):
    """
    A/B/G3/F candidate rules, computed once. Bitwise operators throughout
    (not `and`/`or`), so this works unchanged on plain scalars or pandas
    Series/numpy arrays.

    Returns (is_candidate, label): a bool (or bool Series/array) plus the
    matched class label ("A"/"B"/"G3"/"F", or None) -- both derived from the
    same rule evaluation, so a caller that only wants one doesn't pay for
    computing the other twice. Priority F > A > B > G3 when more than one
    rule matches: rarest and most severe first, so a missed-mate-and-lost
    position labels F rather than whatever else it also happens to satisfy.

    A is restricted to plies the mover actually lost: with prod SF only
    reaching d14-15, an "error" the mover still converted or drew out of is
    more likely measurement noise than a real one. B/G3/F fire on loss or
    draw.
    """
    lost = z_stm == -1
    lost_or_drew = z_stm <= 0
    hit_a = lost & (abs(v) <= A_ABS_V) & (cpl >= A_MIN_CPL)
    hit_b = lost_or_drew & (v >= B_MIN_V) & (p < B_MAX_P)
    hit_g3 = lost_or_drew & (v >= G_MIN_V) & (p < G_MAX_P)
    hit_f = lost_or_drew & (v >= F_MIN_V) & (p < F_MAX_P)

    is_candidate = hit_a | hit_b | hit_g3 | hit_f
    label = np.select(
        [hit_f, hit_a, hit_b, hit_g3], ["F", "A", "B", "G3"], default=None
    )

    if np.ndim(label) == 0:
        label = label.item()

    return is_candidate, label


def probe_pool_path(pool_path=None):
    path = pool_path or os.environ.get(BLUNDER_POOL_ENV, "")
    if not path:
        raise RuntimeError(f"{POOL} {BLUNDER_POOL_ENV} is not set")
    return path


def load_probe_pool(pool_path=None):
    """
    The scanned blunder positions -- a live document, not a static input:
    every analysis pass and every rescored game writes back what it learned.
    Stored at rest as a dict keyed by short_fen.
    """
    path = probe_pool_path(pool_path)
    if not os.path.exists(path):
        raise RuntimeError(f"{POOL} no position pool at {path!r}")

    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_probe_pool(pool, pool_path=None):
    """
    Write the pool back atomically. Readers/writers overlap in time (probe
    workers, the live rescorer, analysis passes), and a torn read would take
    one of them down.
    """
    path = probe_pool_path(pool_path)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def evicted_path(pool_path=None):
    """Per-pool recovery log: the pkl's own name, suffixed instead of
    shared, so the two pools' recovery trails never mingle."""
    return probe_pool_path(pool_path).replace(".pkl.gz", "_evicted.jsonl")


def evict_position(pool, key, reason, n_retrains, evicted_file):
    """
    Pop a position out of the live pool and leave a flat, self-contained
    recovery record behind -- enough to rebuild and replay the position
    later without the pool it came from.
    """
    src = pool.pop(key)
    record = {
        "short_fen": key,
        "fen": src.get("fen"),
        "uci_path": src.get("uci_path"),
        "run_tag": src.get("run_tag"),
        "game_id": src.get("game_id"),
        "move_num": src.get("move_num"),
        "start_ply": src.get("start_ply"),
        "model_epoch": src.get("model_epoch"),
        "scenario": src.get("scenario"),
        "evicted_epoch": n_retrains,
        "evicted_ts": int(time.time()),
        "reason": reason,
        "final_probe": (src.get("probes") or [None])[-1],
    }
    with open(evicted_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=float) + "\n")


def ratchet_store(src, move_uci, cp, depth):
    """
    Record a move's eval only when it beats the depth already on file.
    Depth only ever goes up for a position, so a later shallower result
    never overwrites an earlier deeper one.
    """
    if depth is None:
        return False

    depths = src.setdefault("deep_depths", {})
    if depth <= (depths.get(move_uci) or 0):
        return False

    src.setdefault("deep_evals", {})[move_uci] = cp
    depths[move_uci] = depth
    return True


def cached_deep_score(src, move_uci, min_depth=BLUNDER_REPLAY_MIN_CACHE_DEPTH):
    """
    Deep score for move_uci from the pool's own cache, or None if unknown.
    Reusable only once both the best move and this move are on file at
    min_depth+ -- the depth actually reached, not the time budget aimed at
    it, since the same movetime buys wildly different depths position to
    position.
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


def probe_board(src):
    """
    Rebuild the position by pushing the whole path from STARTPOS. Seeding a
    bare FEN would hand Stockfish a board with an empty move stack, and
    python-chess sends the stack to the engine as the position command --
    so without it SF cannot see that a move repeats.
    """
    board = chess.Board()
    for uci in src["uci_path"]:
        board.push_uci(uci)
    return board


def create_blunder_replay_config(cfg, yaml_file=None):
    """
    Config for the blunder-replay probe worker.

    Search params come from the frozen yaml so the metric stays comparable
    across model epochs. Everything else (model path, TRT cache, macro batch)
    is inherited from the live config, so the probe reuses the engine the
    selfplay workers already compiled instead of building its own.
    """
    pcfg = cfg.copy()
    p_yaml = yaml_file or os.environ.get(BLUNDER_REPLAY_CONFIG_ENV, "")
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


def blunder_replay_dir(cfg):
    d = os.path.join(cfg.run_dir, BLUNDER_REPLAY_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def last_epoch_in_blunder_replay_history(path):
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


def last_blunder_replay_epoch(run_dir, selfplay_dir=None, previous_run_tag=None):
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
    last = last_epoch_in_blunder_replay_history(
        os.path.join(run_dir, BLUNDER_REPLAY_HISTORY_FILENAME)
    )
    if last is not None:
        return last

    if previous_run_tag and selfplay_dir:
        prev = os.path.abspath(os.path.join(selfplay_dir, previous_run_tag))
        last = last_epoch_in_blunder_replay_history(
            os.path.join(prev, BLUNDER_REPLAY_HISTORY_FILENAME)
        )
        if last is not None:
            print(f"{P} continuing cadence from {previous_run_tag} "
                  f"(last probe at epoch {last})")

    return last


def blunder_replay_specs(cfg, pool, n_positions, seed=0):
    """
    GameSpecs that replay each position from STARTPOS.

    Rebuilding the whole move path (rather than seeding a FEN) keeps
    board.history_size() honest, so the tree lands on the same sims-schedule
    entry the position had in the original game.

    The original error travels in the meta, so a probe file carries its own
    answer key and can be read without the pool it was sampled from. Nothing
    to filter here any more -- a position that has earned eviction is simply
    not in the pool dict.
    """
    keys = list(pool.keys())
    if n_positions and n_positions < len(keys):
        keys = random.Random(seed).sample(keys, n_positions)

    startpos = chess.Board().fen()
    specs = []
    for key in keys:
        rec = pool[key]
        meta = {
            "short_fen": key,
            "src_run_tag": rec["run_tag"],
            "src_game_id": rec["game_id"],
            "src_move_num": rec["move_num"],
            "fen": rec["fen"],
            "orig_move": rec["played_move"],
            "orig_best": rec["sf_best_move"],
            "orig_cpl": rec["cpl"],
            "scenario": "blunder_replay_probe",
            "vs_stockfish": False,
            "stockfish_is_white": False,
        }
        specs.append(GameSpec(
            fen=startpos, moves=list(rec["uci_path"]), meta=meta, cfg=cfg,
        ))

    print(f"{P} {len(specs)} positions queued (seed {seed})")
    return specs


