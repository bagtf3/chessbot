"""Post-retrain validation reporting: per-slice metrics, the eval_progress
csvs and validation_latest.png."""
import os

import numpy as np
import pandas as pd

from chessbot.replay_buffer import SEED_SOURCE, SF_SOURCE
from chessbot.utils import policy_metric_means

PRETRAIN_CONTINUED_CSV = "eval_progress_pretrain_continued.csv"
LC0_PROGRESS_CSV = "eval_progress_lc0.csv"
SF_PROGRESS_CSV = "eval_progress_sf.csv"
# pretraining validated every 20 epochs; the continued file keeps that step
PRETRAIN_EPOCH_STEP = 20

# Validation provenance buckets. seaborn "deep" blue/orange/green, keyed by
# name so a source that is absent this round leaves the others' colors alone.
SOURCE_COLORS = {'xc0': '#4C72B0', 'lc0': '#DD8452', 'historic': '#55A868',
                 'sf': '#C44E52'}
SOURCE_ORDER = ('xc0', 'lc0', 'sf', 'historic')


def plot_source(source):
    """Collapse a record's provenance tag to a plot bucket. SEED_SOURCE and
    the untagged tfrec.gz records are both pretrain-distribution data, so both
    read as historic. Every teacher record is tagged plain 'lc0'."""
    if source is None or source == SEED_SOURCE:
        return 'historic'
    if source.startswith('lc0'):
        return 'lc0'
    if source == SF_SOURCE:
        return 'sf'
    return 'xc0'


def migrate_pretrain_progress(run_dir):
    """First-run split of the unified progress csv.

    Pretraining writes its validation rows straight into eval_progress.csv
    (bootstrap's unified mode). Selfplay wants that file to hold only its own
    rows, counting from 0, with the pretraining axis carried on separately in
    PRETRAIN_CONTINUED_CSV -- which the historic validation stream then keeps
    extending by PRETRAIN_EPOCH_STEP.

    Call once at startup, before next_model_epoch reads the counter. No-op
    once PRETRAIN_CONTINUED_CSV exists, so it runs exactly one time per run.
    Everything in eval_progress.csv at that moment is pretraining output --
    selfplay has not written a row yet -- so the whole file moves.
    """
    progress = os.path.join(run_dir, "eval_progress.csv")
    historic = os.path.join(run_dir, PRETRAIN_CONTINUED_CSV)

    if os.path.exists(historic) or not os.path.exists(progress):
        return

    df = pd.read_csv(progress)
    if not len(df):
        return

    df.to_csv(historic, index=False)
    # keep the schema, drop the rows: next_model_epoch then returns 0 and the
    # selfplay counter starts clean
    df.iloc[0:0].to_csv(progress, index=False)
    lo, hi = int(df.model_epoch.min()), int(df.model_epoch.max())
    print(f"[metrics] moved {len(df)} pretrain rows (epoch {lo}..{hi}) -> "
          f"{PRETRAIN_CONTINUED_CSV}; eval_progress.csv restarts at 0")


def next_pretrain_epoch(path, step=PRETRAIN_EPOCH_STEP):
    """Next model_epoch for the pretrain-continued file: one step past the
    last row. Pretraining validated every `step` epochs and this file just
    keeps that axis going."""
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path)
    if not len(df) or 'model_epoch' not in df.columns:
        return 0
    return int(df['model_epoch'].max()) + step


def report_validation(epoch, progress_csv_path, slices, train_stats=None):
    """Console lines, csv rows and validation_latest.png from the retrain
    worker's validation slices (retrain_worker.validate_slice), keyed
    primary / historic / lc0 / sf. train_stats: {'train_loss': ...,
    'gn_mean': ...} from the retrain that just finished. Validation is
    measured on the model going INTO that retrain, matching how pretraining
    pairs an eval with the loss of the epoch it belongs to."""
    if not len(slices['primary']['wdl_pred']) and not len(slices['historic']['wdl_pred']):
        print("[metrics] no validation samples, skipping")
        return

    eps = 1e-7

    def metrics_block(g):
        if not len(g['wdl_pred']):
            return None
        pp = np.array(g['wdl_pred'], dtype=np.float64)
        pt = np.array(g['wdl_true'], dtype=np.float64)
        pp = np.clip(pp, eps, 1 - eps)
        pp /= pp.sum(axis=1, keepdims=True)

        # scalar value for correlation: expected score from WDL
        pred_v = pp[:, 0] - pp[:, 2]
        true_v = pt[:, 0] - pt[:, 2]
        ce = -np.mean(np.sum(pt * np.log(pp), axis=1))
        # entropy of the target: the floor ce can't go below
        ent = -np.mean(np.sum(pt * np.log(np.clip(pt, eps, 1.0)), axis=1))
        mse = np.mean((pred_v - true_v) ** 2)
        corr = np.corrcoef(pred_v, true_v)[0, 1] if len(pred_v) > 1 else 0.0

        # Same metrics pretraining validates with, on the same target-support
        # mask it uses (bootstrap's mstack = pstack > 0), so every policy
        # column lines up across the pretrain/selfplay seam.
        out = policy_metric_means(g['policy'])
        out.update({'ce': ce, 'mse': mse, 'corr': corr,
                    'value_target_ent': ent, 'value_kl': ce - ent,
                    # > 0: predicts more draws / a higher STM score than
                    # the targets
                    'draw_bias': float(np.mean(pp[:, 1] - pt[:, 1])),
                    'value_bias': float(np.mean(pred_v - true_v)),
                    'n': len(g['wdl_pred'])})
        # mass on the #1 entry of the TARGET distribution -- whichever
        # source produced the record (xc0 visits, lc0 output, historic)
        out['avg_top_prob_target'] = float(np.mean(g['top_prob_target']))
        return out

    pb, hb, lb, sb = (metrics_block(slices[k])
                      for k in ('primary', 'historic', 'lc0', 'sf'))

    # internal key -> csv column. Names match the pretraining schema
    # (bootstrap's UNIFIED_COLS) so both files stack on the same axes.
    col_map = {
        'mse': 'value_mse', 'corr': 'value_corr', 'ce': 'value_ce',
        'value_target_ent': 'value_target_ent', 'value_kl': 'value_kl',
        'draw_bias': 'draw_bias', 'value_bias': 'value_bias',
        'policy_ce': 'policy_ce', 'uniform_ce': 'uniform_ce', 'ce_gain': 'ce_gain',
        'policy_target_ent': 'policy_target_ent', 'policy_kl': 'policy_kl',
        'ce_gain_frac': 'ce_gain_frac',
        'top1_exact': 'top1_exact',
        'top1_mass': 'top1_mass', 'top3_mass': 'top3_mass',
        'top3_mass_target': 'top3_mass_target',
        'n': 'n_samples',
        'mass_on_legal': 'mass_on_legal',
        'avg_top_prob': 'avg_top_prob',
        'avg_top_prob_target': 'avg_top_prob_target',
    }

    def save_csv(path, b, row_epoch):
        if not b:
            return
        row = {'model_epoch': row_epoch}
        for k, col in col_map.items():
            if k in b:
                row[col] = round(b[k], 4)
        if 'policy_ce' in b and 'ce' in b:
            row['chess_loss'] = round(b['policy_ce'] + 4.0 * b['ce'], 4)
        if train_stats:
            row.update(train_stats)

        if not os.path.exists(path):
            pd.DataFrame([row]).round(4).to_csv(path, index=False)
            return

        # positional append rather than concat: a one-row frame whose
        # pretrain-only columns are all-NA is exactly the case pandas
        # deprecated the dtype inference for, and it warned every retrain.
        all_df = pd.read_csv(path)
        cols = list(dict.fromkeys(list(all_df.columns) + list(row)))
        all_df = all_df.reindex(columns=cols)
        all_df.loc[len(all_df)] = [row.get(c, np.nan) for c in cols]
        all_df.round(4).to_csv(path, index=False)

    run_dir = os.path.dirname(progress_csv_path)
    historic_csv = os.path.join(run_dir, PRETRAIN_CONTINUED_CSV)
    # the historic stream continues pretraining's own epoch axis, which
    # advanced PRETRAIN_EPOCH_STEP at a time rather than the retrain counter,
    # so the two halves of the curve share a scale. An empty slice (historic
    # draw off, no lc0/sf records) writes no row.
    writes = [
        (progress_csv_path, pb, epoch),
        (historic_csv, hb, next_pretrain_epoch(historic_csv)),
        (os.path.join(run_dir, LC0_PROGRESS_CSV), lb, epoch),
        (os.path.join(run_dir, SF_PROGRESS_CSV), sb, epoch),
    ]
    for path, b, row_epoch in writes:
        save_csv(path, b, row_epoch)
    print("[retrain] all CSVs saved")

    parts = [g for g in slices.values() if len(g['wdl_pred'])]
    combined = {
        'wdl_pred': np.concatenate([g['wdl_pred'] for g in parts]),
        'wdl_true': np.concatenate([g['wdl_true'] for g in parts]),
        'source': [s for g in parts for s in g['source']],
    }
    for k in (1, 3):
        combined[f'top{k}_mass'] = np.concatenate(
            [g['policy'][f'top{k}_mass'] for g in parts])
    save_validation_plots(
        epoch, os.path.join(run_dir, 'validation_latest.png'), combined)


def save_validation_plots(epoch, out_path, g):
    """Single 5-panel figure over the combined validation slices, split by
    provenance wherever a split is meaningful."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not len(g['wdl_pred']):
        return

    sqrt3_2 = np.sqrt(3) / 2
    max_scatter = 5000
    eps = 1e-9

    pp = np.clip(np.array(g['wdl_pred'], dtype=np.float64), eps, 1.0)
    pp /= pp.sum(axis=1, keepdims=True)
    pt = np.array(g['wdl_true'], dtype=np.float64)
    src = np.array([plot_source(s) for s in g['source']])

    pred_v = pp[:, 0] - pp[:, 2]
    true_v = pt[:, 0] - pt[:, 2]
    per_ce = -np.sum(pt * np.log(pp), axis=1)
    mean_ce = np.mean(per_ce)
    mse = np.mean((pred_v - true_v) ** 2)
    corr = np.corrcoef(pred_v, true_v)[0, 1] if len(pred_v) > 1 else 0.0

    present = [s for s in SOURCE_ORDER if (src == s).any()]
    n = len(pred_v)
    width = 0.8 / max(len(present), 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Epoch {epoch}: validation  n={n}", fontsize=12)

    # one shared source legend -- per-axes legends collide with the
    # rotated bar value labels on the grouped panels
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=SOURCE_COLORS[s], label=s)
                 for s in present],
        loc='upper right', ncol=len(present), fontsize=9, framealpha=0.9,
    )

    def grouped(ax, labels, per_source, fmt, title, ylabel, ylim=None):
        xpos = np.arange(len(labels))
        span = max(abs(v) for vals in per_source.values() for v in vals) or 1.0
        pad = span * 0.03
        for i, s in enumerate(present):
            off = (i - (len(present) - 1) / 2) * width
            ax.bar(xpos + off, per_source[s], width=width * 0.9,
                   color=SOURCE_COLORS[s], label=s)
            for x, v in zip(xpos + off, per_source[s]):
                ax.text(x, v + (pad if v >= 0 else -pad), fmt.format(v),
                        ha='center', va='bottom' if v >= 0 else 'top',
                        fontsize=7, rotation=90)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is None:
            ax.margins(y=0.30)
        else:
            ax.set_ylim(*ylim)

    # largest source first so it does not paint over the smaller ones
    per_plot = max_scatter // max(len(present), 1)
    counts = {s: int((src == s).sum()) for s in present}
    for s in sorted(present, key=lambda s: -counts[s]):
        m = np.flatnonzero(src == s)
        take = np.random.choice(m, min(per_plot, len(m)), replace=False)
        axes[0, 0].scatter(true_v[take], pred_v[take], s=4, alpha=0.18,
                           c=SOURCE_COLORS[s])
    axes[0, 0].set_xlabel("target value (W-L)")
    axes[0, 0].set_ylabel("pred value (W-L)")
    axes[0, 0].set_title(f"value scatter  MSE={mse:.3f}  r={corr:.3f}")
    # drawn largest-first so big sources do not bury small ones, but
    # legended in canonical order with opaque proxy markers -- the live
    # handles inherit the scatter alpha and are unreadable at this size
    axes[0, 0].legend(
        handles=[plt.Line2D([], [], marker='o', linestyle='', markersize=6,
                            color=SOURCE_COLORS[s], label=f"{s} (n={counts[s]})")
                 for s in present],
        loc='upper left', fontsize=8, framealpha=0.9)

    masses = {s: [float(np.mean(g[f'top{k}_mass'][src == s])) for k in (1, 3)]
              for s in present}
    grouped(axes[0, 1], ['top1', 'top3'], masses, '{:.3f}',
            'policy top-k mass by source', 'mass', ylim=(0, 1.0))

    idx = np.random.choice(n, min(max_scatter, n), replace=False)
    tx = 0.5 * pp[idx, 0] + pp[idx, 1]
    ty = sqrt3_2 * pp[idx, 0]
    axes[0, 2].plot([0.5, 1.0, 0.0, 0.5], [sqrt3_2, 0.0, 0.0, sqrt3_2], 'k-', lw=0.8)
    sc = axes[0, 2].scatter(
        tx, ty, s=4, alpha=0.4,
        c=per_ce[idx], cmap='RdYlGn_r', vmin=0, vmax=2.0,
    )
    cbar = fig.colorbar(sc, ax=axes[0, 2], shrink=0.8)
    thresholds = np.arange(0, 2.01, 0.25)
    cdf_vals = [np.mean(per_ce <= t) for t in thresholds]
    cbar.set_ticks(thresholds)
    cbar.set_ticklabels([f'{t:.2f}  ({v:.2f})' for t, v in zip(thresholds, cdf_vals)])
    axes[0, 2].text(0.5, sqrt3_2 + 0.03, 'W', ha='center', va='bottom', fontsize=9)
    axes[0, 2].text(1.03, -0.03, 'D', ha='left', va='top', fontsize=9)
    axes[0, 2].text(-0.03, -0.03, 'L', ha='right', va='top', fontsize=9)
    axes[0, 2].set_aspect('equal')
    axes[0, 2].axis('off')
    axes[0, 2].set_title(f'WDL ternary  mean CE={mean_ce:.3f}', pad=18)

    wdl_labels = ['W', 'D', 'L']
    bias = {s: np.mean(pp[src == s] - pt[src == s], axis=0) for s in present}
    ce_comp = {
        s: -np.mean(pt[src == s] * np.log(pp[src == s]), axis=0)
        for s in present
    }

    grouped(axes[1, 0], wdl_labels, bias, '{:+.4f}',
            'WDL bias by source', 'mean(pred - target)')
    axes[1, 0].axhline(0, color='black', lw=0.8)

    grouped(axes[1, 1], wdl_labels, ce_comp, '{:.4f}',
            f'CE by component  total={mean_ce:.3f}',
            'mean CE contribution (nats)')

    axes[1, 2].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
