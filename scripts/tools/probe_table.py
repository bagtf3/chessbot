"""Blunder replay probe summary from the per-round probe_e*.csv files.

Rows from val_test* source runs and positions more than MAX_AGE epochs past
their admission are left out, matching what the pool keeps. Attempt numbers
count every probe on disk for a position, so the oldest round on disk (*)
also counts positions whose first probe predates the CSVs as first attempts.

Epoch numbers are the continuous retrain counter, so run_tags from one
lineage stack into one set of tables.

  python probe_table.py --run_tags m137_8c8t_selfplay9
  python probe_table.py --run_tags m137_8c8t_selfplay8 m137_8c8t_selfplay9
"""

import argparse
import glob
import os
import re

import pandas as pd

from chessbot import SP_DIR

MAX_AGE = 100
AGE_BINS = [0, 7, 10, 20, 40, MAX_AGE + 1]
AGE_LABELS = ["<=7", "8-10", "11-20", "21-40", f"41-{MAX_AGE}"]
MAX_ATTEMPT_ROW = 4


def load_probes(run_tags):
    frames = []
    for tag in run_tags:
        pattern = os.path.join(SP_DIR, tag, "blunder_probe", "probe_e*.csv")
        for path in glob.glob(pattern):
            d = pd.read_csv(path)
            d["epoch"] = int(re.search(r"_e(\d+)_", path).group(1))
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).sort_values("epoch", kind="stable")
    d["attempt"] = d.groupby("short_fen").cumcount() + 1
    d["age"] = d.epoch - d.added_epoch
    keep = ~d.src_run_tag.astype(str).str.startswith("val_test") & (d.age <= MAX_AGE)
    d = d[keep].copy()
    d["improved"] = d.probe_cpl < d.orig_cpl
    return d


def pct(x):
    return f"{100 * x.mean():.1f}%" if len(x) else "-"


def cpl(x):
    if not len(x):
        return "-"
    a, b = x.orig_cpl.mean(), x.probe_cpl.mean()
    return f"{a:.1f} -> {b:.1f} ({b - a:+.1f})"


def trim_rows(disp, keep_last=19):
    """Collapse the middle to one ellipsis row: first, ..., last 19."""
    if len(disp) <= keep_last + 1:
        return disp
    gap = pd.DataFrame([{c: "..." for c in disp.columns}])
    kept = [disp.iloc[[0]], gap, disp.iloc[-keep_last:]]
    return pd.concat(kept, ignore_index=True)


def print_centered(title, disp):
    cols = list(disp.columns)
    widths = {c: max(len(c), disp[c].astype(str).str.len().max()) for c in cols}
    print(title)
    print(" | ".join(c.center(widths[c]) for c in cols))
    for _, row in disp.iterrows():
        print(" | ".join(str(row[c]).center(widths[c]) for c in cols))
    print()


def round_table(d):
    first_epoch = d.epoch.min()
    rows = []
    for e, g in d.groupby("epoch"):
        f, r = g[g.attempt == 1], g[g.attempt > 1]
        rows.append({
            "round": f"{e}*" if e == first_epoch else str(e),
            "n": str(len(g)),
            "first n": str(len(f)),
            "first cpl": cpl(f),
            "first improved": pct(f.improved),
            "first equiv": pct(f.found_equiv),
            "re-probe n": str(len(r)),
            "re-probe cpl": cpl(r),
            "re-probe equiv": pct(r.found_equiv),
            "all equiv": pct(g.found_equiv),
            "cache": pct(g.from_cache),
        })
    return trim_rows(pd.DataFrame(rows))


def attempt_table(d):
    key = d.attempt.clip(upper=MAX_ATTEMPT_ROW)
    rows = []
    for a, g in d.groupby(key):
        rows.append({
            "attempt": f"{a}+" if a == MAX_ATTEMPT_ROW else str(a),
            "n": str(len(g)),
            "cpl": cpl(g),
            "improved": pct(g.improved),
            "equiv": pct(g.found_equiv),
        })
    return pd.DataFrame(rows)


def age_table(first):
    bins = pd.cut(first.age, AGE_BINS, labels=AGE_LABELS)
    rows = []
    for b, g in first.groupby(bins, observed=True):
        rows.append({
            "age": str(b),
            "n": str(len(g)),
            "cpl": cpl(g),
            "improved": pct(g.improved),
            "equiv": pct(g.found_equiv),
        })
    return pd.DataFrame(rows)


def cohort_table(d):
    rows = []
    for e, g in d[d.attempt == 1].groupby("epoch"):
        fens = set(g.short_fen)
        later = d[d.short_fen.isin(fens) & (d.epoch >= e)]
        cleared = later[later.found_equiv == 1].short_fen.nunique()
        reprobed = later[later.epoch > e].short_fen.nunique()
        rows.append({
            "round": str(e),
            "n": str(len(fens)),
            "fixed first try": pct(g.found_equiv),
            "fixed ever": f"{100 * cleared / len(fens):.1f}%",
            "re-probed": str(reprobed),
        })
    return trim_rows(pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tags", nargs="+", required=True)
    args = ap.parse_args()

    d = load_probes(args.run_tags)
    if not len(d):
        print("no probes found")
        return

    first = d[(d.attempt == 1) & (d.epoch > d.epoch.min())]
    print_centered("per round (* oldest round on disk):", round_table(d))
    print_centered("by attempt:", attempt_table(d))
    print_centered("first attempts by age at probe (oldest round excluded):",
                   age_table(first))
    print_centered("first-attempt cohorts:", cohort_table(d))


if __name__ == "__main__":
    main()
