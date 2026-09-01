"""Convert one run's analyze_results_combined.pkl to parquet and drop the pkl.

df_all is ~99% of that file, and most of its bulk is repeated strings: game_id,
stop_reason and the three move columns hold a few thousand distinct values
across millions of rows. Parquet dictionary-encodes them on disk, so the pandas
dtypes stay exactly as they were -- no categoricals, no groupby surprises.
Measured 635 MB -> 109 MB on 18m_10c6t_d256_pretrain, and reads back 4x faster.

df_all is the only required frame. Older pkls also carry df_means (a
one-row-per-game slice) and small summary/results dicts; df_means converts
alongside when present and any other keys go to analyze_meta.pkl.

ONLY run this on a cold run. rescore.py, build_bootstrap_records.py,
clean_training_data.py, eval_selfplay.py, mine_search.py and review_cli.py all
read the pkl path directly and will not find it afterwards.

  python scripts/tools/convert_analyze_to_parquet.py <run_tag> [--dry-run]
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

BASE = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs"
ANALYZE_PKL = "analyze_results_combined.pkl"
META_PKL = "analyze_meta.pkl"
FRAMES = {"df_all": "df_all.parquet", "df_means": "df_means.parquet"}
REQUIRED = ("df_all",)


def shrink(df):
    """int64 -> smallest safe int, float64 -> float32. Strings left alone."""
    out = df.copy()
    for c in out.columns:
        k = out[c].dtype.kind
        if k in "iu":
            out[c] = pd.to_numeric(out[c], downcast="integer")
        elif k == "f":
            out[c] = out[c].astype("float32")
    return out


def col_same(x, y, tol):
    """Compare one column by value. Dtype width may differ after downcast."""
    if x.dtype.kind == "f" or y.dtype.kind == "f":
        return np.allclose(x.to_numpy("float64"), y.to_numpy("float64"),
                           rtol=tol, atol=tol, equal_nan=True)

    if x.dtype.kind in "iub" and y.dtype.kind in "iub":
        return np.array_equal(x.to_numpy("int64"), y.to_numpy("int64"))

    # Old runs carry tuple columns such as sf_wdl. Parquet stores those as
    # lists and hands them back as ndarrays, so compare the numbers rather
    # than the container type.
    xv, yv = x.to_numpy(), y.to_numpy()
    sample = next((v for v in xv[:1000] if not is_null(v)), None)
    if isinstance(sample, (tuple, list, np.ndarray)):
        # nulls sit alongside the sequences, so mask them off before stacking
        xm = np.fromiter((is_null(v) for v in xv), bool, len(xv))
        ym = np.fromiter((is_null(v) for v in yv), bool, len(yv))
        if not np.array_equal(xm, ym):
            return False
        xs = np.stack(xv[~xm])
        ys = np.stack(yv[~ym])
        if tol:
            return np.allclose(xs, ys, rtol=tol, atol=tol, equal_nan=True)
        return np.array_equal(xs, ys)

    return x.equals(y)


def is_null(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def same(a, b, tol=0.0, skip_object=False):
    """Column-wise equality, NaN == NaN, with tolerance for floats."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    cols = a.columns
    if skip_object:
        cols = [c for c in cols if a[c].dtype.kind not in "OS"]
    return all(col_same(a[c], b[c], tol) for c in cols)


def convert(run_dir, dry_run):
    pkl = os.path.join(run_dir, ANALYZE_PKL)

    if not os.path.exists(pkl):
        if os.path.exists(os.path.join(run_dir, FRAMES["df_all"])):
            print("already converted, nothing to do")
            return 0
        raise SystemExit(f"no {ANALYZE_PKL} and no parquet in {run_dir}")

    before = os.path.getsize(pkl)
    print(f"loading {pkl}  ({before / 1e6:.1f} MB)")
    with open(pkl, "rb") as f:
        blob = pickle.load(f)

    missing = [k for k in REQUIRED if k not in blob]
    if missing:
        raise SystemExit(f"pkl is missing {missing}, refusing to convert")

    frames = [k for k in FRAMES if k in blob]
    outs = {k: os.path.join(run_dir, FRAMES[k]) for k in frames}

    small = {}
    for key in frames:
        df = blob[key]
        s = shrink(df)
        # downcast must not change a single value, floats within float32 eps.
        # shrink only touches numeric kinds, so object columns cannot differ.
        if not same(df, s, tol=1e-6, skip_object=True):
            raise SystemExit(f"{key}: downcast changed values, aborting")
        small[key] = s
        print(f"  {key}: {df.shape[0]} rows x {df.shape[1]} cols")

    if dry_run:
        print("[dry-run] would write parquet + meta and delete the pkl")
        return 0

    after = 0
    for key, path in outs.items():
        # keep the index: df_means is indexed by position into df_all, so
        # dropping it would break the game -> ply linkage
        small[key].to_parquet(path, compression="zstd", index=True)
        after += os.path.getsize(path)

    # verify what landed on disk before anything is deleted. On failure the
    # parquet is removed too, so a half-converted run cannot look converted.
    def bail(msg):
        for p in outs.values():
            if os.path.exists(p):
                os.remove(p)
        raise SystemExit(f"{msg}, pkl kept and parquet removed")

    for key, path in outs.items():
        back = pd.read_parquet(path)
        if not same(small[key], back):
            bail(f"{key}: parquet read-back mismatch")
        changed = [c for c in small[key].columns
                   if small[key][c].dtype != back[c].dtype]
        # object columns holding tuples come back as ndarrays; values already
        # verified above, so only flag a change in the numeric columns
        changed = [c for c in changed if small[key][c].dtype.kind != "O"]
        if changed:
            bail(f"{key}: dtypes changed on read-back: {changed}")
    print("verified: parquet round-trips by value")

    meta = {k: v for k, v in blob.items() if k not in FRAMES}
    if meta:
        meta_path = os.path.join(run_dir, META_PKL)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f, protocol=5)
        after += os.path.getsize(meta_path)
        print(f"kept {sorted(meta)} in {META_PKL}")

    os.remove(pkl)
    print(f"removed {ANALYZE_PKL}")
    print(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB "
          f"({before / max(after, 1):.1f}x, saved {(before - after) / 1e6:.1f} MB)")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_tag", help="folder name under selfplay_runs")
    p.add_argument("--base", default=BASE)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_dir = os.path.join(args.base, args.run_tag)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"not a directory: {run_dir}")
    raise SystemExit(convert(run_dir, args.dry_run))


if __name__ == "__main__":
    main()
