import json
import math
import os
import time
import numpy as np
import yaml

from chessbot.mcts_utils import ChessGame
import chessbot.utils as cbu

VALIDATION_CONFIG_FILENAME = "validation_config.yaml"
HISTORY_FILENAME = "validation_history.jsonl"
V = "[validation]"

SF_TABLE_DEFAULT = [
  {"depth": 1,  "elo": 1614, "name": "SF14d1"},
  {"depth": 2,  "elo": 1699, "name": "SF14d2"},
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
  {"depth": 30, "elo": 3043, "name": "SF14d30"}
]


def create_validation_config(cfg, yaml_file=None):
    vcfg = cfg.copy()
    vcfg.sf_table = SF_TABLE_DEFAULT
    vcfg.sf_index = 0
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

    prev_index = last_entry.get("depth_index", 0)
    prev_bumped = last_entry["bumped"]

    n_rows = len(cfg.sf_table)
    if prev_bumped:
        new_index = min(n_rows-1, prev_index + 1)
    else:
        new_index = prev_index
    
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

    if consec >= 2:
        if index < (len(table) - 1):
            bumped = True
            action = "bumped_depth"
            consec = 0
        else:
            # at max depth row do not advance
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
