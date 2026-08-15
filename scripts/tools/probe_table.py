"""Per-probe summary table, slimmed and with a same-move vs different-move
CPL split baked in. Footer row is mean +/- 95% CI across the printed probes
(not a per-position CI -- each row is already an aggregate over ~2048).

Reads <SP_DIR>/<run_tag>/blunder_probe/ for each run_tag given. Epoch numbers
are the continuous retrain counter, so multiple run_tags from one lineage
(e.g. a run and the one it continued from) stack into a single table sorted
by epoch.

  python probe_table.py --run_tags 18m_10c6t_SWA_selfplay1
  python probe_table.py --run_tags 18m_10c6t_SWA_selfplay0 18m_10c6t_SWA_selfplay1
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from chessbot import SP_DIR


def load_rows(run_tags):
    rows = []
    for tag in run_tags:
        run_dir = os.path.join(SP_DIR, tag)
        hist_path = os.path.join(run_dir, "blunder_probe_history.jsonl")
        if not os.path.exists(hist_path):
            print(f"[skip] {tag}: no blunder_probe_history.jsonl")
            continue

        history = {}
        with open(hist_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                history[row["n_retrains"]] = row

        pattern = os.path.join(run_dir, "blunder_probe", "probe_e*.csv")
        for path in sorted(glob.glob(pattern),
                           key=lambda p: int(re.search(r"_e(\d+)_", p).group(1))):
            e = int(re.search(r"_e(\d+)_", path).group(1))
            if e not in history:
                continue
            d = pd.read_csv(path)
            dm = d[~d.same_move]
            sm = d[d.same_move]
            h = history[e]

            rows.append({
                "epoch": e,
                "n": len(d),
                "cpl_a": d.orig_cpl.mean(), "cpl_b": d.probe_cpl.mean(),
                "dm_a": dm.orig_cpl.mean(), "dm_b": dm.probe_cpl.mean(),
                "sm_a": sm.orig_cpl.mean(), "sm_b": sm.probe_cpl.mean(),
                "improved": (d.probe_cpl < d.orig_cpl).mean(),
                "same": d.same_move.mean(),
                "best": d.found_best.mean(),
                "equiv": d.found_equiv.mean(),
                "same & equiv": (d.same_move & d.found_equiv).mean(),
                "depth": d.best_depth.mean(),
                "cache": h["cache_hit"],
                "ret": h["n_retired"],
            })
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def pct(x):
    return f"{100 * x:.1f}%"


def ci95(x):
    n = len(x)
    if n < 2:
        return float("nan")
    return 1.96 * np.std(x, ddof=1) / np.sqrt(n)


def build_table(raw):
    PCT_COLS = ["improved", "same", "best", "equiv", "same & equiv", "cache"]

    disp = pd.DataFrame({
        "epoch": raw["epoch"].astype(str),
        "n": raw["n"].astype(str),
        "cpl": [f"{a:.1f} -> {b:.1f}" for a, b in zip(raw.cpl_a, raw.cpl_b)],
        "cpl (diff move)": [f"{a:.1f} -> {b:.1f}"
                            for a, b in zip(raw.dm_a, raw.dm_b)],
        "cpl (same move)": [f"{a:.1f} -> {b:.1f}"
                            for a, b in zip(raw.sm_a, raw.sm_b)],
        "depth": raw["depth"].round(1).astype(str),
        "ret": raw["ret"].astype(str),
    })
    for c in PCT_COLS:
        disp[c] = raw[c].map(pct)
    disp = disp[["epoch", "n", "cpl", "cpl (diff move)", "cpl (same move)",
                 "improved", "same", "best", "equiv", "same & equiv", "depth",
                 "cache", "ret"]]

    footer = {c: "" for c in disp.columns}
    footer["epoch"] = "mean+/-CI"
    for a_col, b_col, out_col in (("cpl_a", "cpl_b", "cpl"),
                                  ("dm_a", "dm_b", "cpl (diff move)"),
                                  ("sm_a", "sm_b", "cpl (same move)")):
        delta = raw[b_col] - raw[a_col]
        footer[out_col] = f"{delta.mean():+.1f}+/-{ci95(delta):.1f}"
    disp.loc[len(disp)] = footer
    return disp


def print_centered(disp):
    cols = list(disp.columns)
    widths = {c: max(len(c), disp[c].str.len().max()) for c in cols}
    lines = [" | ".join(c.center(widths[c]) for c in cols)]
    for _, row in disp.iterrows():
        lines.append(" | ".join(row[c].center(widths[c]) for c in cols))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tags", nargs="+", required=True)
    args = ap.parse_args()

    raw = load_rows(args.run_tags)
    if not len(raw):
        print("no probes found")
        return
    print_centered(build_table(raw))


if __name__ == "__main__":
    main()
