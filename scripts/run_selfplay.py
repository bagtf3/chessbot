import os, pickle, random
import pathlib, json
import time, gc
import sys, subprocess

_now = time.time

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")

import chess

from pyfastchess import terminal_value_white_pov, raw_cache_bulk_insert
from pyfastchess import raw_cache_clear, priors_cache_clear, priors_cache_stats

from chessbot import ENDGAME_LOC, SF_LOC, SP_DIR, MODEL_DIR
from chessbot.model import load_model, save_model, make_conv_infer
from chessbot.mcts_utils import MCTSTree, ChessGame
from chessbot.config import Config
from chessbot.validation import (
    build_validation_summary, paired_validation_games, create_validation_config
)    

import chessbot.utils as cbu
from chessbot.utils import rnd, RateMeter, softmax, GameGenerator
from chessbot.review import start_post_hoc_server, stop_post_hoc_server

from chessbot.looper import GameLooper, init_selfplay


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python looper.py <run_tag>")
        sys.exit(1)

    run_tag = sys.argv[1]
    run_dir = os.path.join(SP_DIR, run_tag)

    if not os.path.isdir(run_dir):
        print(f"[error] run dir not found: {run_dir}")
        sys.exit(1)

    # find run config yaml
    yaml_path = None
    for nm in ("config.yaml", "config.yml"):
        p = os.path.join(run_dir, nm)
        if os.path.exists(p):
            yaml_path = p
            break

    if yaml_path is None:
        print(f"[error] no config.yaml or config.yml found in {run_dir}")
        sys.exit(1)

    # load base config and validation configs
    cfg = Config.from_yaml(yaml_path, init=True)

    # validation yaml path
    val_yaml_path = os.path.join(cfg.run_dir, "validation_config.yaml")

    phs = None
    start = _now()
    n_games, run_num = 0, 1
    left_over_training_queue = []
    try:
        while run_num <= cfg.n_rounds:
            if n_games:
                elapsed = _now() - start
                rt = cbu.format_time(elapsed)
                gph = 3600 * n_games / elapsed
                print(f"[main loop] Time: {rt} ",
                      f"Games: {n_games} ({gph:.1f}/hr)")

            is_validation = (run_num % cfg.validation_every) == 0

            # prepare a working config for this iteration
            #working_cfg = cfg.copy()
            working_cfg = Config.from_yaml(yaml_path, init=True)
            if is_validation:
                if os.path.exists(val_yaml_path):
                    working_cfg = create_validation_config(cfg, val_yaml_path)
                else:
                    err = f"[validation] local yaml not not found."
                    raise Exception(err)

            # toggle syzygy and other stuff on non val runs
            else:
                working_cfg.use_syzygy = (working_cfg.use_syzygy) & bool(run_num % 2)

            # start post-hoc server if requested (use working config)
            if phs is None and working_cfg.run_post_hoc and working_cfg.run_dir:
                phs = start_post_hoc_server(working_cfg)

            # init selfplay with the working config (returns looper + used cfg)
            looper = init_selfplay(config=working_cfg)
            
            looper.training_queue += left_over_training_queue
            # run the games for this iteration
            # looper.run() completes up to a retrain.
            # return to clear scope due to mem leaks
            finished = looper.run(run_num)
            while not finished:
                # spawn a fresh looper (fresh TF/ infer) and continue
                looper = looper.spawn_next_looper(clear_caches=True)
                gc.collect()
                finished = looper.run(run_num)

            # might be some positions not trained, carry them over
            left_over_training_queue = looper.training_queue
            run_num += 1

            # if it was a validation iteration, record its summary
            if is_validation:
                summary = build_validation_summary(looper)

            if is_validation and working_cfg is not cfg:
                # fully reload the base config from disk (replaces cfg)
                cfg = Config.from_yaml(yaml_path, init=True)

            n_games += looper.games_finished

    finally:
        if phs is not None:
            stop_post_hoc_server(phs, timeout=10)