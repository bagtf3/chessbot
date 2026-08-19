"""
Epoch x all uplift, split by move-changed and by improved/not-improved.

Sourced from the per-epoch probe_e*.csv files on disk, not the live pool:
the pool's own probes[] only survives for positions that haven't been
evicted, so it silently loses history the moment a position is solved.
The CSVs are the permanent record -- every probe that ever ran, regardless
of what later happened to that position in the pool.

Scoped to --run_tags like probe_table.py: epoch is only meaningful within
one model lineage's continuous retrain counter, so globbing every run dir
on disk would silently mix epochs from unrelated model families.

  python pairwise.py --run_tags 18m_10c6t_SWA_selfplay1
  python pairwise.py --run_tags 18m_10c6t_SWA_selfplay0 18m_10c6t_SWA_selfplay1
"""
import argparse
import glob
import itertools
import os
import re

import numpy as np
import pandas as pd

from chessbot import SP_DIR
from chessbot.blunder_replay import BRP_EQUIV_CPL

from probe_table import trim_rows

ap = argparse.ArgumentParser()
ap.add_argument("--run_tags", nargs="+", required=True)
args = ap.parse_args()

seen = {}
for tag in args.run_tags:
    pattern = os.path.join(SP_DIR, tag, "blunder_probe", "probe_e*.csv")
    for path in sorted(glob.glob(pattern)):
        epoch = int(re.search(r"_e(\d+)_", path).group(1))
        df = pd.read_csv(path)
        for row in df.itertuples(index=False):
            seen.setdefault(row.short_fen, {})[epoch] = {
                'move': row.probe_move, 'cpl': row.probe_cpl,
            }

epochs = sorted({e for v in seen.values() for e in v})
print(f'positions with any probe : {len(seen)}')
print(f'epochs                   : {epochs}')
print(f'probed 2+ times          : {sum(1 for v in seen.values() if len(v) > 1)}\n')


def pairs_for(a, b):
    return [(v[a], v[b]) for v in seen.values() if a in v and b in v]


def ci95(x):
    n = len(x)
    if n < 2:
        return float('nan')
    return 1.96 * np.std(x, ddof=1) / np.sqrt(n)


def arrow(before, after):
    return f'{before.mean():.1f} -> {after.mean():.1f}'


def delta_ci(before, after):
    d = after - before
    return f'{d.mean():+.1f} +/- {ci95(d):.1f}'


def pct_plain(mask):
    return f'{100 * mask.mean():.1f}%'


def pct_ci(mask):
    return f'{100 * mask.mean():.1f}% +/- {100 * ci95(mask.astype(float)):.1f}%'


def row(label, pairs, summary=False):
    n = len(pairs)
    before = np.array([x['cpl'] for x, y in pairs], dtype=float)
    after = np.array([y['cpl'] for x, y in pairs], dtype=float)
    diff_mv = np.array([x['move'] != y['move'] for x, y in pairs])
    improved = after < before
    equiv = after <= BRP_EQUIV_CPL

    equiv_cell = pct_plain(equiv)

    if summary:
        cpl_cell = delta_ci(before, after)
        cpl_diff_cell = (delta_ci(before[diff_mv], after[diff_mv])
                         if diff_mv.any() else '')
        diff_cell = pct_plain(diff_mv)
        improved_cell = pct_plain(improved)
        cpl_imp_cell = (delta_ci(before[improved], after[improved])
                        if improved.any() else '')
        cpl_not_imp_cell = (delta_ci(before[~improved], after[~improved])
                            if (~improved).any() else '')
    else:
        cpl_cell = arrow(before, after)
        cpl_diff_cell = arrow(before[diff_mv], after[diff_mv]) if diff_mv.any() else ''
        diff_cell = pct_plain(diff_mv)
        improved_cell = pct_plain(improved)
        cpl_imp_cell = arrow(before[improved], after[improved]) if improved.any() else ''
        cpl_not_imp_cell = (arrow(before[~improved], after[~improved])
                            if (~improved).any() else '')

    return {
        'split': label, 'n': str(n), 'cpl': cpl_cell, 'diff': diff_cell,
        'cpl (diff)': cpl_diff_cell, 'improved': improved_cell,
        'equiv': equiv_cell,
        'cpl (improved)': cpl_imp_cell, 'cpl (!improved)': cpl_not_imp_cell,
    }


rows = []
for a in epochs[:-1]:
    pooled = [p for b in epochs if b > a for p in pairs_for(a, b)]
    if pooled:
        rows.append(row(f'{a} x all', pooled))

# collapse the middle of the per-epoch block before the summaries are
# appended -- those span every pair, so nothing trimmed here is lost
rows = trim_rows(pd.DataFrame(rows)).to_dict('records')

everything = [p for a, b in itertools.combinations(epochs, 2)
              for p in pairs_for(a, b)]
rows.append(row('all x all', everything))
rows.append(row('mean +/- CI', everything, summary=True))

cols = ['split', 'n', 'cpl', 'diff', 'cpl (diff)', 'improved',
       'equiv', 'cpl (improved)', 'cpl (!improved)']
widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
print(' | '.join(c.center(widths[c]) for c in cols))
for r in rows:
    print(' | '.join(r[c].center(widths[c]) for c in cols))
