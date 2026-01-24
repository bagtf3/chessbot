import os
import json
import pickle

import numpy as np
import pandas as pd

from chessbot.rescore import load_game_index
from chessbot.rescore import ANALYZE_PKL


CPL_MIN = 50
SP_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs"


def derive_pkl_path_from_json(json_path):
    if not json_path.lower().endswith(".json"):
        raise ValueError(f"json path does not end with .json: {json_path}")
    return json_path[:-5] + ".pkl"


def read_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_pickle_file(pkl_path, obj):
    parent = os.path.dirname(pkl_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp = pkl_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, pkl_path)


def atomic_write_jsonl(path, records):
    bak = path + ".bak"
    tmp = path + ".tmp"

    os.replace(path, bak)
    with open(tmp, "w", encoding="utf-8") as out:
        for a in records:
            out.write(json.dumps(a, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

    return bak


def build_delete_set(run_dir, cpl_min):
    analyze_path = os.path.join(run_dir, ANALYZE_PKL)
    with open(analyze_path, "rb") as f:
        prev_run = pickle.load(f)

    df_all = prev_run["df_all"]
    df_means = prev_run["df_means"]

    df_all["clipped_loss"] = np.clip(df_all["loss"], -1000, 1000)
    clipped_cpl = df_all.groupby("game_id")["clipped_loss"].mean()
    df_means["overall_cpl"] = df_means.game_id.map(clipped_cpl)

    if "plies" in df_means.columns:
        to_delete = df_means.query("not ((plies >= 10) and (overall_cpl <= @cpl_min))")
    else:
        to_delete = df_means.query("overall_cpl > @cpl_min")

    return set(to_delete["game_id"]), clipped_cpl


def normalize_game_paths(a):
    pkl_path = a.get("pkl_file", None)
    json_path = a.get("json_file", None)

    if pkl_path is not None and not isinstance(pkl_path, str):
        raise ValueError(f"pkl_file is not a string: {pkl_path}")

    if json_path is not None and not isinstance(json_path, str):
        raise ValueError(f"json_file is not a string: {json_path}")

    return pkl_path, json_path


def check_for_pkl_collisions(all_games):
    owner = {}
    for a in all_games:
        gid = a.get("game_id", None)
        pkl_path, json_path = normalize_game_paths(a)

        paths = []
        if pkl_path:
            paths.append(pkl_path)
        if json_path:
            paths.append(derive_pkl_path_from_json(json_path))

        for p in paths:
            prev = owner.get(p, None)
            if prev is None:
                owner[p] = gid
            elif prev != gid:
                raise ValueError(
                    "two different game_id values map to the same pkl path: "
                    f"{p} -> {prev} and {gid}"
                )


def process_run(run_dir, cpl_min, dry_run):
    all_games = load_game_index(run_dir)
    to_delete_set, clipped_cpl = build_delete_set(run_dir, cpl_min)

    all_games_df = pd.DataFrame(all_games)
    all_games_df["overall_cpl"] = all_games_df["game_id"].map(clipped_cpl)
    all_games = all_games_df.to_dict(orient="records")

    check_for_pkl_collisions(all_games)

    stats = {
        "n_total": 0,
        "n_delete_marked": 0,
        "n_deleted_files": 0,
        "n_kept": 0,
        "n_converted": 0,
        "n_json_removed": 0,
        "n_pkl_already": 0,
        "n_missing": 0,
    }

    for a in all_games:
        stats["n_total"] += 1

        gid = a.get("game_id", None)
        if gid is None:
            raise ValueError("record missing game_id")

        pkl_path, json_path = normalize_game_paths(a)
        is_delete = gid in to_delete_set
        if is_delete:
            stats["n_delete_marked"] += 1

        json_exists = bool(json_path) and os.path.exists(json_path)
        pkl_exists = bool(pkl_path) and os.path.exists(pkl_path)

        derived_pkl = None
        if json_path:
            derived_pkl = derive_pkl_path_from_json(json_path)

        derived_exists = bool(derived_pkl) and os.path.exists(derived_pkl)

        if is_delete:
            if not dry_run:
                if pkl_exists:
                    os.remove(pkl_path)
                    stats["n_deleted_files"] += 1
                if derived_exists and (derived_pkl != pkl_path):
                    os.remove(derived_pkl)
                    stats["n_deleted_files"] += 1
                if json_exists:
                    os.remove(json_path)
                    stats["n_deleted_files"] += 1

            a["pkl_file"] = None
            if "json_file" in a:
                a.pop("json_file")
            continue

        keep_pkl = None
        if pkl_exists:
            keep_pkl = pkl_path
            stats["n_pkl_already"] += 1
        elif derived_exists:
            keep_pkl = derived_pkl
            stats["n_pkl_already"] += 1
        elif json_exists:
            keep_pkl = derived_pkl
            if not dry_run:
                obj = read_json_file(json_path)
                write_pickle_file(keep_pkl, obj)
                os.remove(json_path)
            stats["n_converted"] += 1
        else:
            stats["n_missing"] += 1

        a["pkl_file"] = keep_pkl
        if "json_file" in a:
            a.pop("json_file")
            stats["n_json_removed"] += 1

        if keep_pkl is not None:
            stats["n_kept"] += 1

    for a in all_games:
        if "json_file" in a:
            raise RuntimeError("post-condition violated: json_file key still present")
        if "pkl_file" not in a:
            raise RuntimeError("post-condition violated: missing pkl_file key")

    if not dry_run:
        for a in all_games:
            p = a["pkl_file"]
            if p is None:
                continue
            if not os.path.exists(p):
                raise RuntimeError(f"post-condition violated: missing pkl on disk: {p}")

        index_path = os.path.join(run_dir, "game_index.json")
        bak = atomic_write_jsonl(index_path, all_games)
        return all_games, stats, bak

    return all_games, stats, None


def main(dry_run=True):
    skip = ["cf_10x256x5_low_sims3", "cf_10x256x5_low_sims2"]
    run_tags = [rt for rt in os.listdir(SP_DIR) if rt not in skip]

    total_deleted = 0
    total_converted = 0
    total_kept = 0
    total_missing = 0

    for rt in run_tags:
        print("=" * 60)
        print(f"[clean up] starting JSON->PKL + delete for {rt}")
        print(f"[dry_run] {dry_run}")
        print("=" * 60)

        run_dir = os.path.join(SP_DIR, rt)
        _, stats, bak = process_run(run_dir, CPL_MIN, dry_run=dry_run)

        total_deleted += stats["n_delete_marked"]
        total_converted += stats["n_converted"]
        total_kept += stats["n_kept"]
        total_missing += stats["n_missing"]

        print(f"total records          : {stats['n_total']}")
        print(f"marked for delete      : {stats['n_delete_marked']}")
        print(f"kept (pkl present)     : {stats['n_kept']}")
        print(f"converted json->pkl    : {stats['n_converted']}")
        print(f"json keys removed      : {stats['n_json_removed']}")
        print(f"pkl already existed    : {stats['n_pkl_already']}")
        print(f"missing (no files)     : {stats['n_missing']}")
        if bak:
            print(f"game_index backup      : {bak}")
        print()

    print("=" * 60)
    print("summary")
    print("=" * 60)
    print(f"total marked delete    : {total_deleted}")
    print(f"total kept             : {total_kept}")
    print(f"total converted        : {total_converted}")
    print(f"total missing          : {total_missing}")


main(dry_run=True)
