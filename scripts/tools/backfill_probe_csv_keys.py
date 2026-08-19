"""
One-time: add a short_fen column to every existing probe_e*.csv, derived
from the pre_migration pool backups (the only place pos_id -> fen still
exists -- the CSVs themselves never stored the raw fen, only the hash).

Every on-disk probe CSV predates the short_fen refactor (analyse_probe_file
now writes short_fen directly going forward), so this is a one-time backfill,
not something that runs again.
"""
import glob
import gzip
import hashlib
import pickle

import pandas as pd

from chessbot.blunder_replay import short_fen

BACKUP_GLOB = "C:/Users/Bryan/Data/chessbot_data/*.pre_migration.pkl.gz"
CSV_GLOB = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/*/blunder_probe/probe_e*.csv"


def build_pos_id_map():
    pos_id_to_short_fen = {}
    collisions = 0
    for path in glob.glob(BACKUP_GLOB):
        with gzip.open(path, "rb") as f:
            old_list = pickle.load(f)
        for rec in old_list:
            pos_id = rec.get("pos_id") or hashlib.sha1(
                rec["fen"].encode()).hexdigest()[:16]
            sf = short_fen(rec["fen"])
            if pos_id in pos_id_to_short_fen and pos_id_to_short_fen[pos_id] != sf:
                collisions += 1
                continue
            pos_id_to_short_fen[pos_id] = sf
        print(f"[backfill] {path}: {len(old_list)} records")
    print(f"[backfill] pos_id -> short_fen map: {len(pos_id_to_short_fen)} "
          f"entries, {collisions} cross-backup collisions skipped")
    return pos_id_to_short_fen


def main():
    id_map = build_pos_id_map()

    paths = sorted(glob.glob(CSV_GLOB))
    print(f"[backfill] {len(paths)} probe CSVs found")

    n_fixed = 0
    n_dropped = 0
    for path in paths:
        df = pd.read_csv(path)
        if "short_fen" in df.columns:
            continue
        if "pos_id" not in df.columns:
            print(f"[backfill] {path}: no pos_id or short_fen column, skipping")
            continue

        df["short_fen"] = df["pos_id"].map(id_map)
        missing = df["short_fen"].isna().sum()
        if missing:
            n_dropped += missing
            df = df.dropna(subset=["short_fen"])

        df = df.drop(columns=["pos_id"])
        df.to_csv(path, index=False)
        n_fixed += 1
        print(f"[backfill] {path}: {len(df)} rows, {missing} unmapped dropped")

    print(f"[backfill] rewrote {n_fixed} CSVs, {n_dropped} rows dropped "
          f"(pos_id not found in either backup)")


if __name__ == "__main__":
    main()
