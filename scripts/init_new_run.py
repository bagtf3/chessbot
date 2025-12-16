#!/usr/bin/env python3
"""
init_run.py

Usage:
  python init_run.py <run_tag> [--clone <existing_run_tag>]

Creates run_dir, game_logs, writes config.yaml and config.json, and
initializes validation_config.json (via ValidationConfig). If --clone
is given, copies model and validation_config from the source run dir.
"""

import os
import sys
import argparse
import json
import shutil
import yaml

from chessbot.config import Config
from chessbot.validation import ValidationConfig


def atomic_write(path, obj, fmt="json"):
    tmp = path + ".tmp"
    if fmt == "json":
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=False)
    else:
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(obj, fh, sort_keys=False,
                           default_flow_style=False)
    os.replace(tmp, path)


def write_config(path, obj, fmt="yaml"):
    tmp = path + ".tmp"
    if fmt == "json":
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=False)
    else:
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(obj, fh, sort_keys=False,
                           default_flow_style=False)
    os.replace(tmp, path)


def find_model_in_dir(d):
    h5s = [os.path.join(d, f) for f in os.listdir(d)
           if f.endswith(".h5")]
    if not h5s:
        return None
    h5s.sort(key=os.path.getmtime, reverse=True)
    return h5s[0]


def copy_from_clone(src_tag, dest_cfg):
    src_dir = os.path.abspath(os.path.join(dest_cfg.selfplay_dir, src_tag))
    if not os.path.isdir(src_dir):
        print(f"[clone] source run not found: {src_dir}")
        sys.exit(1)

    src_model = os.path.join(src_dir, f"{src_tag}_model.h5")
    if not os.path.exists(src_model):
        src_model = find_model_in_dir(src_dir)

    if src_model and os.path.exists(src_model):
        dest_model = dest_cfg.model_path
        print(f"[clone] copying model {src_model} -> {dest_model}")
        shutil.copy2(src_model, dest_model)
    else:
        print("[clone] no source model found; skipping model copy")

    src_val = os.path.join(src_dir, "validation_config.json")
    if os.path.exists(src_val):
        dst_val = os.path.join(dest_cfg.run_dir, "validation_config.json")
        print(f"[clone] copying validation config {src_val} -> {dst_val}")
        shutil.copy2(src_val, dst_val)
    else:
        print("[clone] no validation_config.json in source; skipping")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_tag", help="new run tag (e.g. new_run_test)")
    p.add_argument("--clone", help="existing run tag to clone model/configs")
    args = p.parse_args()

    run_tag = args.run_tag
    clone_tag = args.clone

    Config.run_tag = run_tag
    if clone_tag:
        Config.previous_run_tag = clone_tag

    cfg = Config()

    if clone_tag:
        cfg.previous_run_tag = clone_tag

    cfg.init_paths()

    cfg_dict = cfg.to_dict()
    cfg_yaml_path = os.path.join(cfg.run_dir, "config.yaml")
    cfg_json_path = os.path.join(cfg.run_dir, "config.json")

    write_config(cfg_yaml_path, cfg_dict, fmt="yaml")
    print(f"[init] wrote config.yaml -> {cfg_yaml_path}")

    write_config(cfg_json_path, cfg_dict, fmt="json")
    print(f"[init] wrote config.json -> {cfg_json_path}")

    if clone_tag:
        copy_from_clone(clone_tag, cfg)

    val_cfg = ValidationConfig()
    val_json_path = os.path.join(cfg.run_dir, "validation_config.json")
    if os.path.exists(val_json_path):
        print(f"[init] validation_config present -> {val_json_path}")
    else:
        atomic_write(val_json_path, val_cfg.to_dict(), fmt="json")
        print(f"[init] wrote fallback validation_config -> {val_json_path}")

    print(f"[init] run_dir ready: {cfg.run_dir}")
    print("Next steps:")
    print(f"  cd {cfg.run_dir}")
    print("  edit config.yaml and validation_config.json as desired")
    print("  run your looper / training scripts")


if __name__ == "__main__":
    main()
