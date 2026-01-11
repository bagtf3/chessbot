"""
init_new_run.py

Usage:
  python init_new_run.py <run_tag> [--clone <existing_run_tag>]

Simple: create run_dir, write config.yaml and validation_config.yaml.
If --clone is given, copy those files from the clone run dir and set
the new run's init_model to the cloned model (copied into the new run dir).
"""

import os
import sys
import argparse
import yaml
import shutil
from pathlib import Path

def write_yaml(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False, default_flow_style=False)
    os.replace(tmp, path)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_model_in_dir(d):
    # prefer exact-named model <tag>_model.h5 handled by caller,
    # else pick newest .h5
    p = Path(d)
    h5s = sorted(p.glob("*.h5"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not h5s:
        return None
    return str(h5s[0])


def replace_yaml_values_inplace(path, run_tag, init_model, prev_run_tag=None):
    lines = []
    replaced_prev = False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.lstrip().startswith("run_tag:"):
                indent = line[:len(line) - len(line.lstrip())]
                lines.append(f"{indent}run_tag: {run_tag}\n")
            elif line.lstrip().startswith("init_model:"):
                indent = line[:len(line) - len(line.lstrip())]
                lines.append(f"{indent}init_model: {init_model}\n")
            elif line.lstrip().startswith("previous_run_tag:"):
                if prev_run_tag is not None:
                    indent = line[:len(line) - len(line.lstrip())]
                    lines.append(f"{indent}previous_run_tag: {prev_run_tag}\n")
                    replaced_prev = True
            else:
                lines.append(line)

    if prev_run_tag is not None and not replaced_prev:
        # append previous_run_tag if it wasn't present
        lines.append(f"\nprevious_run_tag: {prev_run_tag}\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_tag")
    parser.add_argument("--clone", dest="clone_tag", default=None)
    parser.add_argument("--selfplay_dir", dest="selfplay_dir", default="runs")
    args = parser.parse_args()

    run_tag = args.run_tag
    clone_tag = args.clone_tag
    cfg = argparse.Namespace()
    cfg.selfplay_dir = args.selfplay_dir
    cfg.init_model = None

    dest_dir = os.path.abspath(os.path.join(cfg.selfplay_dir, run_tag))
    if os.path.exists(dest_dir):
        print(f"[init] run_dir already exists: {dest_dir}")
        sys.exit(1)

    os.makedirs(dest_dir, exist_ok=False)

    cfg_yaml_dst = os.path.join(dest_dir, "config.yaml")
    val_yaml_dst = os.path.join(dest_dir, "validation_config.yaml")

    if clone_tag:
        src_dir = os.path.abspath(os.path.join(cfg.selfplay_dir, clone_tag))
        if not os.path.isdir(src_dir):
            print(f"[clone] source run not found: {src_dir}")
            sys.exit(1)

        # find source model: prefer exact-named <clone_tag>_model.h5 else newest .h5
        src_model = os.path.join(src_dir, f"{clone_tag}_model.h5")
        if not os.path.exists(src_model):
            src_model = find_model_in_dir(src_dir)

        # if found, copy into new run_dir as <run_tag>_model.h5 and set cfg.init_model
        if src_model and os.path.exists(src_model):
            dest_model = os.path.join(dest_dir, f"{run_tag}_model.h5")
            shutil.copy2(src_model, dest_model)
            cfg.init_model = dest_model
            print(f"[clone] copied model {src_model} -> {dest_model}")
        else:
            # no source model found; leave cfg.init_model as default (may be configured)
            print("[clone] no model found in source; using default init_model in config")

        # copy config.yaml if present; update run_tag and init_model in the copy
        src_cfg = os.path.join(src_dir, "config.yaml")
        if os.path.exists(src_cfg):
            shutil.copy2(src_cfg, cfg_yaml_dst)
            replace_yaml_values_inplace(
                cfg_yaml_dst, run_tag, cfg.init_model,
                prev_run_tag=clone_tag
            )
            print(f"[clone] copied config.yaml from {clone_tag}")
        else:
            # write minimal config (init_model already possibly set above)
            minimal = {
                "run_tag": run_tag,
                "selfplay_dir": cfg.selfplay_dir,
                "init_model": cfg.init_model
            }
            write_yaml(cfg_yaml_dst, minimal)
            write_yaml(val_yaml_dst, {"is_validation": True})
            print("[clone] wrote minimal config and validation_config")

        # copy training_config.yaml if present; update run_tag and init_model in the copy
        src_train_cfg = os.path.join(src_dir, "training_config.yaml")
        if os.path.exists(src_train_cfg):
            train_cfg_dst = os.path.join(dest_dir, "training_config.yaml")
            shutil.copy2(src_train_cfg, train_cfg_dst)
            # mirror the same inplace replacement behavior as for config.yaml
            replace_yaml_values_inplace(
                train_cfg_dst, run_tag, cfg.init_model,
                prev_run_tag=clone_tag
            )
            print(f"[clone] copied training_config.yaml from {clone_tag}")

    else:
        # no clone: write minimal config + validation config
        write_yaml(cfg_yaml_dst, {
            "run_tag": run_tag,
            "selfplay_dir": cfg.selfplay_dir,
            "init_model": cfg.init_model
        })
        write_yaml(val_yaml_dst, {"is_validation":True})
        print("[init] wrote minimal config.yaml and validation_config.yaml")

    print(f"[init] run_dir ready: {dest_dir}")
    print(f"  edit {cfg_yaml_dst} and {val_yaml_dst} as needed")


if __name__ == "__main__":
    main()
