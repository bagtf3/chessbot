"""Temporary val_test* runs: fork a run, test against the fork, delete it.

A val_test run is a full copy of a source run (model, train_ckpts, buffers,
live buffer, game logs and indexes, analysis pickles, csvs, npz, configs)
re-pointed at itself, so experiments there never read from or write to the
source. The source is snapshotted before the copy and checked after it: it
must come out byte-for-byte unmodified. val_test* runs are temporary by
design: create, test, learn, delete.

Usage:
  python scripts/val_test.py create <source_run_tag> [--name val_test]
  python scripts/val_test.py delete [--name val_test] --yes
"""
import argparse
import json
import os
import pickle
import shutil
import sys

from chessbot.config import Config

PREFIX = "val_test"
CONFIG_FILES = ("config.yaml", "validation_config.yaml")
INDEX_FILES = ("game_index.json", "game_index_staging.json")
LC0_BACKLOG = "lc0_backlog.pkl"
# top-level files scanned for paths still pointing at the source; game logs
# carry only per-game provenance, so they are left alone
SCAN_SUFFIXES = (".json", ".jsonl", ".yaml", ".csv", ".pkl", ".txt")


def runs_dir():
    return Config.selfplay_dir


def check_name(name):
    if not name.startswith(PREFIX):
        sys.exit(f"[val_test] refusing: {name!r} is not a {PREFIX}* run tag")


def snapshot(root):
    """{relative path: (size, mtime_ns)} for every file under root."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


def path_forms(path):
    """The ways a run dir appears inside files: raw and JSON-escaped."""
    raw = path.encode()
    return [json.dumps(path)[1:-1].encode(), raw]


def rename_model_files(dst, src_tag, name):
    renamed = []
    for sub in ("", "train_ckpts"):
        d = os.path.join(dst, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith(f"{src_tag}_model"):
                new = name + f[len(src_tag):]
                os.rename(os.path.join(d, f), os.path.join(d, new))
                renamed.append(os.path.join(sub, new))
    return renamed


def fix_config(dst, name):
    """run_tag and init_model, edited line by line so the rest of the yaml is
    untouched."""
    p = os.path.join(dst, "config.yaml")
    lines = open(p, encoding="utf-8").read().split("\n")
    model = os.path.join(dst, f"{name}_model.pt")
    for i, line in enumerate(lines):
        if line.startswith("run_tag:"):
            lines[i] = f"run_tag: {name}"
        elif line.startswith("init_model:"):
            lines[i] = f"init_model: {model}"
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def repoint_files(dst, src_dir, files):
    """Replace the source run dir with dst inside the given files."""
    counts = {}
    for name in files:
        p = os.path.join(dst, name)
        if not os.path.exists(p):
            continue
        data = open(p, "rb").read()
        n = 0
        for old, new in zip(path_forms(src_dir), path_forms(dst)):
            n += data.count(old)
            data = data.replace(old, new)
        with open(p, "wb") as f:
            f.write(data)
        counts[name] = n
    return counts


def fix_lc0_backlog(dst):
    """Queued lc0 probes carry a pickled lc0 replay config of the source run;
    the runner reloads them as they are, so they get dst's."""
    p = os.path.join(dst, LC0_BACKLOG)
    if not os.path.exists(p):
        return 0
    from chessbot.lc0_replay import create_lc0_replay_config
    lc0_cfg = create_lc0_replay_config(Config.from_yaml(os.path.join(dst, "config.yaml")))
    with open(p, "rb") as f:
        specs = pickle.load(f)
    specs = [s._replace(cfg=lc0_cfg) for s in specs]
    with open(p, "wb") as f:
        pickle.dump(specs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return len(specs)


def leftover_refs(dst, src_dir):
    """Top-level files that still mention the source run dir."""
    hits = []
    for f in sorted(os.listdir(dst)):
        p = os.path.join(dst, f)
        if os.path.isfile(p) and f.endswith(SCAN_SUFFIXES):
            data = open(p, "rb").read()
            if any(form in data for form in path_forms(src_dir)):
                hits.append(f)
    return hits


def create(src_tag, name):
    check_name(name)
    src = os.path.join(runs_dir(), src_tag)
    dst = os.path.join(runs_dir(), name)
    if not os.path.isfile(os.path.join(src, "config.yaml")):
        sys.exit(f"[val_test] no run at {src}")
    if not os.path.isfile(os.path.join(src, f"{src_tag}_model.pt")):
        sys.exit(f"[val_test] {src} has no {src_tag}_model.pt")
    if os.path.exists(dst):
        sys.exit(f"[val_test] {dst} already exists; delete it first")

    before = snapshot(src)
    print(f"[val_test] copying {src} -> {dst} ({len(before)} files)", flush=True)
    shutil.copytree(src, dst, copy_function=shutil.copy2)

    renamed = rename_model_files(dst, src_tag, name)
    fix_config(dst, name)
    counts = repoint_files(dst, src, INDEX_FILES)
    n_backlog = fix_lc0_backlog(dst)

    cfg = Config.from_yaml(os.path.join(dst, "config.yaml"))
    if cfg.run_tag != name or os.path.dirname(cfg.model_path) != dst:
        sys.exit(f"[val_test] config check failed: run_tag {cfg.run_tag}, "
                 f"model_path {cfg.model_path}")
    if snapshot(src) != before:
        sys.exit(f"[val_test] SOURCE CHANGED during the copy: {src}")

    print(f"[val_test] renamed: {', '.join(renamed) or 'nothing'}")
    print(f"[val_test] config: run_tag {cfg.run_tag}, init_model {name}_model.pt, "
          f"game_dir {cfg.game_dir}")
    for f, n in counts.items():
        print(f"[val_test] {f}: {n} source paths re-pointed")
    print(f"[val_test] {LC0_BACKLOG}: {n_backlog} probes given {name}'s lc0 config")
    left = leftover_refs(dst, src)
    if left:
        print(f"[val_test] WARNING: still mention {src}: {', '.join(left)}")
    print(f"[val_test] source {src_tag} verified unchanged ({len(before)} files)")


def delete(name, yes):
    check_name(name)
    dst = os.path.join(runs_dir(), name)
    if os.path.dirname(os.path.abspath(dst)) != os.path.abspath(runs_dir()):
        sys.exit(f"[val_test] refusing: {dst} is not directly under {runs_dir()}")
    if not os.path.isdir(dst):
        sys.exit(f"[val_test] nothing at {dst}")
    n = len(snapshot(dst))
    if not yes:
        sys.exit(f"[val_test] would delete {dst} ({n} files); rerun with --yes")
    shutil.rmtree(dst)
    print(f"[val_test] deleted {dst} ({n} files)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("source")
    c.add_argument("--name", default=PREFIX)
    d = sub.add_parser("delete")
    d.add_argument("--name", default=PREFIX)
    d.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    if args.cmd == "create":
        create(args.source, args.name)
    else:
        delete(args.name, args.yes)
