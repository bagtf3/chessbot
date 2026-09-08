"""Per-game keep/exclude lists for the pretrain shard builder, one worker per
run_tag, up to N at a time. Runs entirely off df_all.parquet + game_index.json
(never opens the per-game pkl.gz).

Phase 1: exclude games whose mean CPL > --game-cpl-max.
Phase 2: collar filter on the survivors (a >350cp edge held 7+ plies that then
         doesn't win or reverses for 3+ plies -> noisy -> exclude).

Writes <out-dir>/<run_tag>.json = {run_tag, game_cpl_max, n_games, exclude, keep}.
Resumes: run_tags with an existing json are skipped unless --force.

Usage:
    python scripts/data/scan_game_filters_mp.py [--workers 7] [--game-cpl-max 50]
        [--out-dir DIR] [--force] [--run_tags TAG ...]
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import queue
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pqf

from chessbot import SP_DIR

OUT_DIR_DEFAULT = r"C:\Users\Bryan\Data\chessbot_data\training_data\pretrain_filter_lists"
COLLAR_CP, COLLAR_N, COLLAR_REV = 350, 7, 3

RUN_TAGS = [
    "16m_precond_run0", "16m_precond_run1", "16m_precond_run2",
    "16m_xc0hK6_run1", "16m_xc0hK6_run2", "16m_xc0hK6_run3", "16m_xc0hK6_run4",
    "18m_10c6t_d256_pretrain", "18m_10c6t_d256_selfplay0", "18m_10c6t_d256_selfplay1",
    "18m_10c6t_SWA_selfplay0", "18m_10c6t_SWA_selfplay1", "18m_10c6t_SWA_selfplay2",
    "18m_10c6t_SWA_selfplay3", "18m_10c6t_SWA_selfplay4", "18m_10c6t_SWA_selfplay5",
    "18m_10c6t_SWA_selfplay6", "18m_6c4t_pretrain_run0", "18m_6c4t_selfplay_run0",
]


def collar_is_noisy(evals, result):
    holder = None; holder_start = None; run_sign = 0; run_count = 0
    for i, ev in enumerate(evals):
        sign = 1 if ev > COLLAR_CP else (-1 if ev < -COLLAR_CP else 0)
        if sign != 0 and sign == run_sign:
            run_count += 1
        elif sign != 0:
            run_sign, run_count = sign, 1
        else:
            run_sign, run_count = 0, 0
        if run_count >= COLLAR_N:
            holder, holder_start = ('white' if run_sign == 1 else 'black'), i
            break
    if holder is None:
        return False
    if float(result) != (1.0 if holder == 'white' else -1.0):
        return True
    below = 0
    for ev in evals[holder_start:]:
        pov = ev if holder == 'white' else -ev
        if pov < 0:
            below += 1
            if below >= COLLAR_REV:
                return True
        else:
            below = 0
    return False


def load_results(rd):
    p = os.path.join(rd, "game_index.json")
    out = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    gid = r.get("game_id")
                    if gid is not None:
                        out[gid] = r.get("result", 0)
                except Exception:
                    pass
    return out


def load_df(rd):
    """CPL column is 'loss' on some runs, 'delta' on others; normalize to 'loss'."""
    pq = os.path.join(rd, "df_all.parquet")
    if os.path.exists(pq):
        names = pqf.ParquetFile(pq).schema.names
        cpl = "loss" if "loss" in names else "delta"
        df = pd.read_parquet(pq, columns=["game_id", "move_num", cpl, "best_absolute"])
        return df.rename(columns={cpl: "loss"}) if cpl != "loss" else df
    comb = os.path.join(rd, "analyze_results_combined.pkl")
    if os.path.exists(comb):
        with open(comb, "rb") as f:
            df = pickle.load(f)["df_all"]
        if "loss" not in df.columns and "delta" in df.columns:
            df = df.rename(columns={"delta": "loss"})
        return df[["game_id", "move_num", "loss", "best_absolute"]]
    return None


def scan_one(run_tag, gcm, out_dir):
    rd = os.path.join(SP_DIR, run_tag)
    df = load_df(rd)
    if df is None:
        return {"run_tag": run_tag, "skipped": "no sf_df"}
    res = load_results(rd)

    mean_cpl = df.groupby("game_id")["loss"].apply(
        lambda s: float(np.clip(s.values, -1000, 1000).mean()))
    n_games = len(mean_cpl)
    cpl_excl = set(mean_cpl.index[mean_cpl > gcm])
    survivors = set(mean_cpl.index) - cpl_excl

    df = df.sort_values(["game_id", "move_num"], kind="stable")
    collar_excl = set()
    for gid, sub in df.groupby("game_id", sort=False):
        if gid not in survivors:
            continue
        if collar_is_noisy(sub["best_absolute"].to_numpy(), res.get(gid, 0)):
            collar_excl.add(gid)

    exclude = sorted(cpl_excl | collar_excl)
    keep = sorted(survivors - collar_excl)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{run_tag}.json"), "w") as f:
        json.dump({"run_tag": run_tag, "game_cpl_max": gcm, "n_games": n_games,
                   "exclude": exclude, "keep": keep}, f)
    return {"run_tag": run_tag, "n_games": n_games, "n_cpl": len(cpl_excl),
            "n_collar": len(collar_excl), "n_exclude": len(exclude), "n_keep": len(keep)}


def worker(run_tag, gcm, out_dir):
    try:
        scan_one(run_tag, gcm, out_dir)
    except Exception as e:
        with open(os.path.join(out_dir, f"{run_tag}.error"), "w") as f:
            f.write(str(e))
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--game-cpl-max", type=float, default=50.0, dest="gcm")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT, dest="out_dir")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--run_tags", nargs="+", default=None)
    args = ap.parse_args()

    tags = args.run_tags or RUN_TAGS
    os.makedirs(args.out_dir, exist_ok=True)

    todo = []
    for t in tags:
        if not args.force and os.path.exists(os.path.join(args.out_dir, f"{t}.json")):
            print(f"[skip] {t} (list exists)", flush=True)
            continue
        if load_df_exists(t):
            todo.append(t)
        else:
            print(f"[skip] {t} (no sf_df)", flush=True)
    print(f"[main] {len(todo)} run_tags to scan, {args.workers} at a time, "
          f"game_cpl_max={args.gcm}", flush=True)

    active = {}                 # tag -> (Process, start_time)
    pending = list(todo)
    done = [0]
    t0 = time.time()

    def launch(tag):
        p = mp.Process(target=worker, args=(tag, args.gcm, args.out_dir))
        p.start()
        active[tag] = (p, time.time())

    def report(tag, dt):
        done[0] += 1
        jp = os.path.join(args.out_dir, f"{tag}.json")
        ep = os.path.join(args.out_dir, f"{tag}.error")
        if os.path.exists(jp):
            with open(jp) as f:
                d = json.load(f)
            print(f"[{done[0]}] {tag}: games={d['n_games']:,} "
                  f"EXCLUDE={len(d['exclude']):,} KEEP={len(d['keep']):,}  {dt:.0f}s", flush=True)
        elif os.path.exists(ep):
            with open(ep) as f:
                print(f"[{done[0]}] {tag}: ERROR {f.read().strip()}  {dt:.0f}s", flush=True)
        else:
            print(f"[{done[0]}] {tag}: no output  {dt:.0f}s", flush=True)

    for _ in range(min(args.workers, len(pending))):
        launch(pending.pop(0))

    # gate on process liveness, not a queue message: workers os._exit right
    # after writing their json, which can drop any final queue message
    while active or pending:
        for tag, (p, st) in list(active.items()):
            if not p.is_alive():
                p.join()
                del active[tag]
                report(tag, time.time() - st)
                if pending:
                    launch(pending.pop(0))
        time.sleep(0.5)

    print(f"\n[done] scanned in {time.time()-t0:.0f}s -> {args.out_dir}", flush=True)


def load_df_exists(run_tag):
    rd = os.path.join(SP_DIR, run_tag)
    return (os.path.exists(os.path.join(rd, "df_all.parquet"))
            or os.path.exists(os.path.join(rd, "analyze_results_combined.pkl")))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
