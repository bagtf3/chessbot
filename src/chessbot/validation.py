import json
import math
import os
import time
import numpy as np

from pyfastchess import Board as fastboard
from chessbot.config import Config
from chessbot.mcts_utils import ChessGame
import chessbot.utils as cbu

VALIDATION_CONFIG_FILENAME = "validation_config.json"
HISTORY_FILENAME = "validation_history.jsonl"

SF_TABLE_DEFAULT = [
  {"depth": 1,  "elo": 1614, "name": "SF14d1"},
  {"depth": 3,  "elo": 1783, "name": "SF14d3"},
  {"depth": 4,  "elo": 1853, "name": "SF14d4"},
  {"depth": 5,  "elo": 1922, "name": "SF14d5"},
  {"depth": 6,  "elo": 2002, "name": "SF14d6"},
  {"depth": 7,  "elo": 2107, "name": "SF14d7"},
  {"depth": 9,  "elo": 2336, "name": "SF14d9"},
  {"depth": 12, "elo": 2644, "name": "SF14d12"},
  {"depth": 14, "elo": 2727, "name": "SF14d14"},
  {"depth": 16, "elo": 2810, "name": "SF14d16"},
  {"depth": 18, "elo": 2847, "name": "SF14d18"},
  {"depth": 21, "elo": 2903, "name": "SF14d21"},
  {"depth": 25, "elo": 2965, "name": "SF14d25"},
  {"depth": 30, "elo": 3043, "name": "SF14d30"},
]

class ValidationConfig(Config):
    """
    Config used for validation runs. Hard overrides the knobs that should
    be different for a validation job so you cannot accidentally train
    with validation settings.
    """

    is_validation_run = True

    n_games = 64
    games_at_once = 64
    micro_batch = 2
    fwd_batch = 128

    # disable noisy exploration during validation
    add_root_noise = False
    sample_moves = False
    dirichlet_eps = 0.0
    dirichlet_alpha = 0.0

    # simulation schedule: keep stable sims for evaluation
    sf_move_sims = 5
    sims_floor = 1000
    sims_ceiling = 1600
    target_delta = 300
    es_check_every = 50

    run_post_hoc = True
    mine_bonus_data = True
    
    max_game_length = 300
    min_game_length = 1
    material_diff_cutoff = 100
    material_diff_cutoff_span = 1000
    use_syzygy = True

    def __init__(self):
        """
        Initialize ValidationConfig and ensure validation_config.json contains the
        depth-indexed sf_table and progress counters.

        This does not force UCI_LimitStrength. The engine invocation should use the
        current depth (cfg.sf_depth) when running validation games.
        """
        super().__init__()

        path = os.path.join(self.run_dir, VALIDATION_CONFIG_FILENAME)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                disk = json.load(fh)

            disk_table = disk['sf_table']
            disk_index = disk['sf_index']
            disk_consec = disk['consec_over_50']

            self.sf_table = disk_table
            self.sf_index = int(disk_index)
            self.consec_over_50 = int(disk_consec)

        else:
            # create initial config file using defaults
            disk_out = {
                "sf_table": SF_TABLE_DEFAULT,
                "sf_index": 0,
                "consec_over_50": 0,
                "history": [],
            }
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(disk_out, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)

            self.sf_table = SF_TABLE_DEFAULT
            self.sf_index = 0
            self.consec_over_50 = 0

        # convenience fields for the current depth entry
        entry = self.sf_table[self.sf_index]
        self.sf_depth = entry["depth"]
        self.sf_elo = entry["elo"]

        # validation report
        # Read last history JSONL entry (if present)
        xerces_elo_last = None
        last_score = None
        hist_path = os.path.join(self.run_dir, HISTORY_FILENAME)
        if os.path.exists(hist_path):
            last_obj = None
            with open(hist_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    last_obj = json.loads(s)
            if last_obj is not None:
                xerces_elo_last = last_obj.get("model_elo")
                last_score = last_obj.get("score")

        # Next depth entry (if any)
        if self.sf_index < (len(self.sf_table) - 1):
            next_entry = self.sf_table[self.sf_index + 1]
        else:
            next_entry = None

        # Determine cusp/streak states (only meaningful if there is a next depth)
        on_cusp = (int(self.consec_over_50) == 1) and (next_entry is not None)
        on_streak = (int(self.consec_over_50) >= 1) and (next_entry is not None)

        # Format SF line
        if next_entry is not None:
            next_str = f"Next depth: {next_entry['depth']} ({next_entry['elo']})"
        else:
            next_str = "Next depth: <MAX>"

        sf_line = (
            f"[validation test] SF depth: {self.sf_depth}  SF Elo: {self.sf_elo}  "
            f"{next_str}"
        )

        # Format Xerces/model line without using try/except
        if xerces_elo_last is None:
            xerces_str = "N/A"
        elif isinstance(xerces_elo_last, (int, float)):
            xerces_str = f"{xerces_elo_last:.1f}"
        else:
            xerces_str = str(xerces_elo_last)

        if last_score is None:
            score_str = "N/A"
        elif isinstance(last_score, (int, float)):
            score_str = f"{last_score:.3f}"
        else:
            score_str = str(last_score)

        model_line = (
            f"[validation test] Xerces Elo last test: {xerces_str}  "
            f"Score last test: {score_str}  consec_over_50: {self.consec_over_50}"
        )

        print(sf_line)
        print(model_line)

def paired_validation_games(cfg):
    """
    Return paired ChessGame instances for Stockfish validation.

    For each i in range(n):
      - pick a random pre-opened premove path index
      - build two identical fastboard positions from that index
      - create two ChessGame objects where stockfish plays one as white
        and the other as black
    """

    games = []
    n = cfg.n_games // 2
    indices = [i for i in range(len(cbu.PATHS))]
    selected = np.random.choice(indices, n, replace=False)
    
    for i, s in enumerate(selected):
        # get two separate fastboard instances from same premoved path
        # always play 1 pair from startpos
        if i == 0:
            board_white = fastboard()
            board_black = fastboard()
        else:
            board_white = cbu.get_pre_opened_game(index=s)
            board_black = cbu.get_pre_opened_game(index=s)

        meta_w = {"vs_stockfish": True, "stockfish_is_white": True,
                  "scenario": "paired_validation"}
        
        meta_b = {"vs_stockfish": True, "stockfish_is_white": False,
                  "scenario": "paired_validation"}

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

    Assumes looper.config is the strict dict loaded from validation JSON.
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

    cfg = looper.config.to_dict()
    table = cfg["sf_table"]
    index = cfg["sf_index"]
    entry = table[index]
    sf_elo = entry["elo"]

    model_elo = estimate_model_elo(sf_elo, score)

    # minimal gating: require at least min_games to count this run
    min_games = 30
    run_counts = (n >= min_games) and (score > 0.5)

    # update consecutive counter and bump if two in a row
    consec = cfg["consec_over_50"]
    if run_counts:
        consec = consec + 1
    else:
        consec = 0
    cfg["consec_over_50"] = consec

    bumped = False
    action = "none"
    if consec >= 2:
        if cfg["sf_index"] < (len(table) - 1):
            cfg["sf_index"] = cfg["sf_index"] + 1
            new_entry = table[cfg["sf_index"]]
            cfg["sf_depth"] = new_entry["depth"]
            cfg["sf_elo"] = new_entry["elo"]
            cfg["consec_over_50"] = 0
            bumped = True
            action = "bumped_depth"
        else:
            # at max depth row, reset the counter but do not advance
            cfg["consec_over_50"] = 0
            action = "at_max_depth"

    # persist the updated config (replace old file)
    save_validation_config(cfg["run_dir"], cfg)

    summary = {
        "ts": time.time(),
        "run_tag": cfg["run_tag"],
        "run_dir": cfg["run_dir"],
        "n_games": n,
        "wins": wins,
        "draws": draws,
        "score": score,
        "sf_elo": sf_elo,
        "model_elo": model_elo,
        "should_count": run_counts,
        "consec_over_50": cfg["consec_over_50"],
        "bumped": bumped,
        "action": action
    }

    losses = n - wins - draws
    print(
        f"[sf results] W/D/L {wins}/{draws}/{losses}  "
        f"sf_elo {sf_elo}  model_elo {model_elo:.1f}  bumped {bumped}"
    )

    return summary


def load_validation_config(run_dir, start_elo=None):
    path = os.path.join(run_dir, VALIDATION_CONFIG_FILENAME)

    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    # sf_table must be a non-empty list and entries must contain depth and elo
    table = cfg["sf_table"]
    index = cfg["sf_index"]
    entry = table[index]

    # derive convenience fields from the authoritative table entry
    cfg["sf_depth"] = entry["depth"]
    cfg["sf_elo"] = entry["elo"]

    return cfg


def save_validation_config(run_dir, cfg):
    """ Write validation_config.json atomically. """
    path = os.path.join(run_dir, VALIDATION_CONFIG_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def append_validation_summary(run_dir, summary):
    """
    Append JSONL history and atomically write latest JSON.
    """
    os.makedirs(run_dir, exist_ok=True)

    hist_path = os.path.join(run_dir, HISTORY_FILENAME)

    with open(hist_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
