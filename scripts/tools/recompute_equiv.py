"""
Reapply the dynamic equiv band to a converted run's df_all/df_means parquet.

Idempotent: the original best_cp is always reconstructed first, so running
this twice, or on a run that already has the band, is a no-op.

  python recompute_equiv.py <run_tag> [<run_tag> ...] [--all] [--dry-run]
"""
import argparse
import os
import sys

import pandas as pd

from chessbot import SP_DIR
from chessbot.config import Config

FRAMES = ("df_all.parquet", "df_means.parquet")
MISSED_MATE_CAP = 100


def original_best_cp(df):
    """Undo the swap branch, which set best_cp = played_cp but left delta as
    the original difference. best_cp == played_cp with a nonzero delta is that
    branch's unique signature -- an unswapped row there would have delta 0."""
    swapped = (df["best_cp"] == df["played_cp"]) & (df["delta"] != 0)
    return df["best_cp"].astype("int64").mask(
        swapped, df["played_cp"].astype("int64") + df["delta"].astype("int64")
    ), swapped


def recompute(df, equiv_min, equiv_max):
    """Returns (delta, best_cp, played_best_move, n_swapped, n_capped)."""
    best_cp, _ = original_best_cp(df)
    played_cp = df["played_cp"].astype("int64")

    band = (best_cp.abs() * 0.1).clip(lower=equiv_min, upper=equiv_max)
    raw = best_cp - played_cp
    delta = raw.where(raw.abs() > band, 0)

    # xerces beat SF: the played move becomes the best one on record
    swap = delta <= -band
    new_best_cp = best_cp.mask(swap, played_cp)

    # a row already capped at 100 was a missed mate; Z_stm is not in the frame
    # so the rule cannot be re-evaluated, only preserved
    capped = ((df["loss"] == MISSED_MATE_CAP)
              & (df["best_cp"] >= 1200) & (df["played_cp"] >= 500)
              & (raw != MISSED_MATE_CAP))
    delta = delta.mask(capped, MISSED_MATE_CAP)

    return (delta.astype(df["delta"].dtype),
            new_best_cp.astype(df["best_cp"].dtype),
            delta <= 0, int(swap.sum()), int(capped.sum()))


def process(tag, equiv_min, equiv_max, dry_run):
    run_dir = os.path.join(SP_DIR, tag)
    if not os.path.exists(os.path.join(run_dir, FRAMES[0])):
        print(f"[{tag}] no df_all.parquet, skipping", flush=True)
        return

    for name in FRAMES:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            print(f"[{tag}] {name} missing, skipping it", flush=True)
            continue

        df = pd.read_parquet(path)
        delta, best_cp, pbm, n_swap, n_capped = recompute(
            df, equiv_min, equiv_max)
        changed = int((delta != df["delta"]).sum())
        old_bmr = df["played_best_move"].mean()

        print(f"[{tag}] {name:<17} rows {len(df):>9,} | changed {changed:>9,} "
              f"| cpl {df['loss'].mean():6.2f} -> {delta.mean():6.2f} "
              f"| bmr {old_bmr:5.3f} -> {pbm.mean():5.3f} "
              f"| swap {n_swap:,} capped {n_capped:,}", flush=True)

        if dry_run:
            continue
        df["delta"] = delta
        df["loss"] = delta
        df["best_cp"] = best_cp
        df["played_best_move"] = pbm
        tmp = path + ".tmp"
        df.to_parquet(tmp, index=True)
        os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tags", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        tags = sorted(
            d for d in os.listdir(SP_DIR)
            if os.path.exists(os.path.join(SP_DIR, d, FRAMES[0]))
        )
    else:
        tags = args.run_tags

    if not tags:
        print("nothing to do -- pass run tags or --all", flush=True)
        sys.exit(1)

    cfg = Config()
    dry = "  [DRY RUN]" if args.dry_run else ""
    print(f"equiv band: clip(|best_cp| * 0.1, {cfg.rescore_equiv_min}, "
          f"{cfg.rescore_equiv_max}){dry}\n", flush=True)

    for tag in tags:
        process(tag, cfg.rescore_equiv_min, cfg.rescore_equiv_max, args.dry_run)


if __name__ == "__main__":
    main()
