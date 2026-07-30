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

from chessbot.config import Config


def write_yaml(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False, default_flow_style=False)
    os.replace(tmp, path)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_model_in_dir(d):
    p = Path(d)
    for ext in ("*.pt", "*.ts", "*.h5"):
        matches = sorted(p.glob(ext), key=lambda x: x.stat().st_mtime, reverse=True)
        if matches:
            return str(matches[0])
    return None


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
    p = argparse.ArgumentParser()
    p.add_argument("run_tag")
    p.add_argument("--clone", default=None)
    p.add_argument("--ignore-eval-progress", action="store_true")
    args = p.parse_args()

    run_tag = args.run_tag
    clone_tag = args.clone

    # set run_tag so Config.init_paths will compute the right paths
    Config.run_tag = run_tag
    cfg = Config()
    cfg.init_paths()

    dest_dir = cfg.run_dir
    os.makedirs(dest_dir, exist_ok=True)

    cfg_yaml_dst = os.path.join(dest_dir, "config.yaml")
    val_yaml_dst = os.path.join(dest_dir, "validation_config.yaml")

    if clone_tag:
        src_dir = os.path.abspath(os.path.join(cfg.selfplay_dir, clone_tag))
        if not os.path.isdir(src_dir):
            print(f"[clone] source run not found: {src_dir}")
            sys.exit(1)

        # find source model: prefer exact-named <clone_tag>_model.{ext} else newest
        src_model = None
        for ext in ("pt", "ts", "h5"):
            candidate = os.path.join(src_dir, f"{clone_tag}_model.{ext}")
            if os.path.exists(candidate):
                src_model = candidate
                break
        if src_model is None:
            src_model = find_model_in_dir(src_dir)

        # if found, copy into new run_dir preserving extension and set cfg.init_model
        if src_model and os.path.exists(src_model):
            src_ext = Path(src_model).suffix
            if src_ext in (".pt", ".ts"):
                # copy only the canonical {clone_tag}_model.* files;
                # ignore *_backup.* stale copies (would otherwise collide
                # on the single destination filename and silently win)
                pt_files = [
                    f for f in (
                        list(Path(src_dir).glob("*.pt")) +
                        list(Path(src_dir).glob("*.ts"))
                    )
                    if f.stem == f"{clone_tag}_model"
                ]
                cfg.init_model = None
                for f in pt_files:
                    dst = os.path.join(dest_dir, f"{run_tag}_model{f.suffix}")
                    shutil.copy2(f, dst)
                    print(f"[clone] copied model {f} -> {dst}")
                    if f.suffix == ".pt":
                        cfg.init_model = dst
                if cfg.init_model is None:
                    cfg.init_model = os.path.join(dest_dir, f"{run_tag}_model{src_ext}")
            else:
                dest_model = os.path.join(dest_dir, f"{run_tag}_model{src_ext}")
                shutil.copy2(src_model, dest_model)
                cfg.init_model = dest_model
                print(f"[clone] copied model {src_model} -> {dest_model}")
        else:
            # no source model found; leave cfg.init_model as default (may be configured)
            print("[clone] no model found in source; using default init_model in config")

        # copy train_ckpts (retrain optimizer state) if present, renaming the
        # file(s) so the stem matches the new run's model name
        src_ckpts_dir = os.path.join(src_dir, "train_ckpts")
        if os.path.isdir(src_ckpts_dir):
            dst_ckpts_dir = os.path.join(dest_dir, "train_ckpts")
            os.makedirs(dst_ckpts_dir, exist_ok=True)
            old_stem = f"{clone_tag}_model"
            new_stem = f"{run_tag}_model"
            copied_any = False
            for f in Path(src_ckpts_dir).iterdir():
                if not f.is_file():
                    continue
                name = f.name
                if name.startswith(old_stem):
                    name = new_stem + name[len(old_stem):]
                dst = os.path.join(dst_ckpts_dir, name)
                shutil.copy2(f, dst)
                copied_any = True
            if copied_any:
                print(f"[clone] copied train_ckpts from {clone_tag}, renamed to match {new_stem}")

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
            print("[init] wrote minimal config.yaml")

        # copy validation_config.yaml if present (no modification)
        src_val = os.path.join(src_dir, "validation_config.yaml")
        if os.path.exists(src_val):
            shutil.copy2(src_val, val_yaml_dst)
            print(f"[clone] copied validation_config.yaml from {clone_tag}")
        else:
            write_yaml(val_yaml_dst, {"is_validation":True})
            print("[init] wrote minimal validation_config.yaml (no src found)")

        # copy replay_buffer/ and primary_buffer/ dirs if present
        for buf_name in ("replay_buffer", "primary_buffer"):
            src_buf = os.path.join(src_dir, buf_name)
            if os.path.isdir(src_buf):
                dst_buf = os.path.join(dest_dir, buf_name)
                shutil.copytree(src_buf, dst_buf, dirs_exist_ok=True)
                print(f"[clone] copied {buf_name}/ from {clone_tag}")

        # copy TRT builder cache files (trt_cache/ for selfplay, val_trt/ for
        # validation). Only .profile and .timing -- .engine/.onnx are keyed to
        # the old model_name + weight-content-hash and would just be dead
        # weight under the new run_tag. .timing in particular is a
        # GPU-architecture-keyed builder tactic cache (not model-specific),
        # so reusing it avoids re-profiling tactics from scratch on a fresh
        # run_tag; .profile only pays off if trt_model_name ends up matching.
        for trt_name in ("trt_cache", "val_trt"):
            src_trt = os.path.join(src_dir, trt_name)
            if os.path.isdir(src_trt):
                dst_trt = os.path.join(dest_dir, trt_name)
                os.makedirs(dst_trt, exist_ok=True)
                copied = 0
                for f in os.listdir(src_trt):
                    if f.endswith(".profile") or f.endswith(".timing"):
                        shutil.copy2(os.path.join(src_trt, f), os.path.join(dst_trt, f))
                        copied += 1
                if copied:
                    print(f"[clone] copied {copied} TRT profile/timing cache "
                          f"file(s) from {clone_tag}/{trt_name}")

        # move remaining_untrained.pkl if present
        src_remaining = os.path.join(src_dir, "remaining_untrained.pkl")
        if os.path.exists(src_remaining):
            dst_remaining = os.path.join(dest_dir, "remaining_untrained.pkl")
            shutil.move(src_remaining, dst_remaining)
            print(f"[clone] moved remaining_untrained.pkl from {clone_tag}")

        # move sf_cache.pkl.gz if present
        src_sf_cache = os.path.join(src_dir, "sf_cache.pkl.gz")
        if os.path.exists(src_sf_cache):
            dst_sf_cache = os.path.join(dest_dir, "sf_cache.pkl.gz")
            shutil.move(src_sf_cache, dst_sf_cache)
            print(f"[clone] moved sf_cache.pkl.gz from {clone_tag}")

        # copy eval_progress.csv unless suppressed
        if not args.ignore_eval_progress:
            src_eval = os.path.join(src_dir, "eval_progress.csv")
            if os.path.exists(src_eval):
                dst_eval = os.path.join(dest_dir, "eval_progress.csv")
                shutil.copy2(src_eval, dst_eval)
                print(f"[clone] copied eval_progress.csv from {clone_tag}")
    else:
        # not cloning: write minimal files
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
