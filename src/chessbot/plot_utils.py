import numpy as np
import pandas as pd


def moving_average_pd(arr, window=15):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().values


def plot_training_progress(metrics_history, epoch=None, ma_max=50):
    """
    metrics_history: pd.DataFrame or dict-like with columns used below.
    Each panel shows raw data as a pale line and MA as a bold line.
    """
    import matplotlib.pyplot as plt

    df = metrics_history.copy()

    def col_vals(name):
        if name in df.columns:
            return df[name].tolist()
        return []

    if epoch is None:
        if "model_epoch" in df.columns and len(df):
            epoch = int(max(df["model_epoch"]))
        else:
            epoch = len(df)

    hide_first = 10
    if epoch < 12:
        return

    ma_window = min(ma_max, max(3, int(epoch * 0.2)))
    if ma_window % 2 == 0:
        ma_window += 1

    N = len(df)
    x_full = np.arange(N)
    start = hide_first
    xs = x_full[start:]

    def plot_panel(ax, raw, label, color=None, title=None):
        """Scatter raw points + MA line. raw is a full-length np array."""
        ma = moving_average_pd(raw, window=ma_window)[start:]
        raw_seg = raw[start:]
        kw = dict(color=color) if color else {}
        if raw_seg.size:
            ax.plot(xs, raw_seg, alpha=0.25, lw=0.8, label=label, **kw)
        if ma.size:
            ax.plot(xs, ma, lw=2, label=f"MA{ma_window}", **kw)
        if title:
            ax.set_title(title)
        ax.legend(fontsize=8)
        return ma

    # ---- figure 1: 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # policy CE
    raw = np.array(col_vals("policy_ce") or [])
    if raw.size:
        plot_panel(axes[0, 0], raw, "policy_ce", title="policy CE (nats)")

    # CE gain vs uniform
    raw = np.array(col_vals("ce_gain") or [])
    if raw.size:
        plot_panel(axes[0, 1], raw, "ce_gain", title="CE gain vs uniform")

    # value MSE (+optional value CE on twin axis)
    ax = axes[1, 0]
    raw_mse = np.array(col_vals("value_mse") or [])
    raw_ce  = np.array(col_vals("value_ce")  or [])
    have_ce = raw_ce.size > 0 and not np.all(np.isnan(raw_ce))

    if have_ce:
        ma_mse = moving_average_pd(raw_mse, window=ma_window)[start:] if raw_mse.size else np.array([])
        raw_mse_seg = raw_mse[start:] if raw_mse.size else np.array([])
        l_handles = []
        if raw_mse_seg.size:
            ax.plot(xs, raw_mse_seg, alpha=0.25, lw=0.8, color="tab:blue")
        if ma_mse.size:
            l1, = ax.plot(xs, ma_mse, color="tab:blue", lw=2, label=f"MSE MA{ma_window}")
            l_handles.append(l1)
        ax.set_ylabel("value MSE", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")

        ax_ce = ax.twinx()
        ma_ce = moving_average_pd(raw_ce, window=ma_window)[start:]
        raw_ce_seg = raw_ce[start:]
        if raw_ce_seg.size:
            ax_ce.plot(xs, raw_ce_seg, alpha=0.25, lw=0.8, color="tab:orange")
        if ma_ce.size:
            l2, = ax_ce.plot(xs, ma_ce, color="tab:orange", lw=2, label=f"CE MA{ma_window}")
            l_handles.append(l2)
        ax_ce.set_ylabel("value CE", color="tab:orange")
        ax_ce.tick_params(axis="y", labelcolor="tab:orange")
        ax.set_title("value MSE / CE")
        if l_handles:
            ax.legend(handles=l_handles, fontsize=8)
    else:
        if raw_mse.size:
            plot_panel(ax, raw_mse, "mse", title="value MSE")

    # value corr
    raw = np.array(col_vals("value_corr") or [])
    if raw.size:
        plot_panel(axes[1, 1], raw, "corr", title="value corr")

    plt.tight_layout()

    # ---- figure 2: 1x3 ----
    fig2, axs = plt.subplots(1, 3, figsize=(15, 4))

    # top1 / top3 / top5
    ax = axs[0]
    t1 = np.array(col_vals("top1_mass"))
    t3 = np.array(col_vals("top3_mass"))
    t5 = np.array(col_vals("top5_mass"))
    any_top = any(a.size for a in (t1, t3, t5))
    if not any_top:
        ax.text(0.5, 0.5, "no top-k data", ha="center", va="center")
        ax.set_axis_off()
    else:
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for arr, label, c in zip((t1, t3, t5), ("top1", "top3", "top5"), colors):
            if arr.size:
                x = np.arange(len(arr))
                ax.plot(x, arr, alpha=0.25, lw=0.8, color=c)
                ma = moving_average_pd(arr, window=ma_window)
                ax.plot(np.arange(len(ma)), ma, lw=2, color=c, label=f"{label} MA{ma_window}")
        ax.set_title("mean top-k mass")
        ax.legend(fontsize=8)

    # mass_on_legal
    ax = axs[1]
    mol = np.array(col_vals("mass_on_legal"))
    if not mol.size:
        ax.text(0.5, 0.5, "missing: mass_on_legal", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(mol))
        ax.plot(x, mol, alpha=0.25, lw=0.8, label="mass_on_legal")
        ma_mol = moving_average_pd(mol, window=ma_window)
        ax.plot(x, ma_mol, lw=2, label=f"MA{ma_window}")
        ax.set_title("mass_on_legal")
        ax.legend(fontsize=8)

    # avg_top_prob & top1_exact
    ax = axs[2]
    avg_tp  = np.array(col_vals("avg_top_prob"))
    t1_exact = np.array(col_vals("top1_exact"))
    if not avg_tp.size and not t1_exact.size:
        ax.text(0.5, 0.5, "missing: avg_top_prob / top1_exact", ha="center", va="center")
        ax.set_axis_off()
    else:
        plotted = False
        for arr, label, c in (
            (avg_tp,   "avg_max_prob", "tab:blue"),
            (t1_exact, "top1_exact",   "tab:orange"),
        ):
            if arr.size:
                x = np.arange(len(arr))
                ax.plot(x, arr, alpha=0.25, lw=0.8, color=c)
                ma = moving_average_pd(arr, window=ma_window)
                ax.plot(np.arange(len(ma)), ma, lw=2, color=c, label=f"{label} MA{ma_window}")
                plotted = True
        if plotted:
            ax.set_title("avg_max_prob & top1_exact")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_axis_off()

    plt.tight_layout()

    plt.show()


def plot_validation(
    epoch,
    plot_path,
    nn_vals_stm,
    target_ys,
    sf_cps,
    result_zs,
    val_mse,
    val_corr,
    val_ce,
    nn_wdl_arr,
    tgt_wdl_arr,
    wdl_bias,
    ce_components,
    max_scatter=5000,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_pts = len(nn_vals_stm)
    idx = (np.random.choice(n_pts, max_scatter, replace=False)
           if n_pts > max_scatter else np.arange(n_pts))

    xv   = nn_vals_stm[idx]
    ytgt = target_ys[idx]
    ycp  = sf_cps[idx]
    zc   = result_zs[idx]

    z_palette = {1.0: "#3cb371", 0.0: "#ffd700", -1.0: "#ff8c00"}
    c_all = [z_palette.get(round(float(z)), "#999999") for z in zc]
    legend = [
        mpatches.Patch(color="#3cb371", label="win"),
        mpatches.Patch(color="#ffd700", label="draw"),
        mpatches.Patch(color="#ff8c00", label="loss"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].scatter(ytgt, xv, s=4, alpha=0.3, c='steelblue')
    axes[0, 0].set_xlabel("target Y")
    axes[0, 0].set_ylabel("NN value (STM-POV)")
    axes[0, 0].set_title(
        f"Epoch {epoch}: NN value vs target  MSE={val_mse:.2f} r={val_corr:.2f}")

    valid2 = ~np.isnan(ycp)
    xv2, ycp2 = xv[valid2], ycp[valid2]
    ycp2_tanh = np.tanh(ycp2 * (np.arctanh(0.8) / 500.0))
    mse2  = np.mean((xv2 - ycp2_tanh) ** 2) if len(xv2) > 1 else float('nan')
    corr2 = np.corrcoef(xv2, ycp2)[0, 1]    if len(xv2) > 1 else float('nan')
    c2 = [c_all[i] for i in range(len(xv)) if valid2[i]]
    axes[0, 1].scatter(ycp2, xv2, s=4, alpha=0.3, c=c2)
    axes[0, 1].set_xlabel("SF CP (STM-POV)")
    axes[0, 1].set_ylabel("NN value (STM-POV)")
    axes[0, 1].set_title(f"NN value vs SF centipawns  MSE={mse2:.2f} r={corr2:.2f}")
    axes[0, 1].legend(handles=legend, fontsize=8)

    sqrt3_2 = np.sqrt(3) / 2
    has_wdl = nn_wdl_arr is not None and len(nn_wdl_arr) > 0

    if has_wdl:
        per_ce_full = -np.sum(tgt_wdl_arr * np.log(np.clip(nn_wdl_arr, 1e-7, 1.0)), axis=1)
        n_tern = len(nn_wdl_arr)
        idx_t = (np.random.choice(n_tern, max_scatter, replace=False)
                 if n_tern > max_scatter else np.arange(n_tern))
        nn_s = nn_wdl_arr[idx_t]
        tx = 0.5 * nn_s[:, 0] + nn_s[:, 1]
        ty = sqrt3_2 * nn_s[:, 0]
        axes[0, 2].plot([0.5, 1.0, 0.0, 0.5], [sqrt3_2, 0.0, 0.0, sqrt3_2], 'k-', lw=0.8)
        sc3 = axes[0, 2].scatter(tx, ty, s=4, alpha=0.4, c=per_ce_full[idx_t],
                                 cmap='RdYlGn_r', vmin=0, vmax=2.0)
        cbar = fig.colorbar(sc3, ax=axes[0, 2], shrink=0.8)
        thresholds = np.arange(0, 2.01, 0.25)
        cdf_vals = [np.mean(per_ce_full <= t) for t in thresholds]
        cbar.set_ticks(thresholds)
        cbar.set_ticklabels([f'{t:.2f}  ({v:.2f})' for t, v in zip(thresholds, cdf_vals)])
        axes[0, 2].text(0.5, sqrt3_2 + 0.03, 'W', ha='center', va='bottom', fontsize=9)
        axes[0, 2].text(1.03, -0.03, 'D', ha='left', va='top', fontsize=9)
        axes[0, 2].text(-0.03, -0.03, 'L', ha='right', va='top', fontsize=9)
        axes[0, 2].set_aspect('equal')
        axes[0, 2].axis('off')
        axes[0, 2].set_title(f'nn WDL ternary (STM)  mean CE={val_ce:.3f}')

        labels = ['W', 'D', 'L']
        colors = ['#3cb371', '#ffd700', '#ff8c00']
        bias_vals = [float(wdl_bias[i]) for i in range(3)]
        bar_colors = ['#d73027' if v > 0 else '#4575b4' for v in bias_vals]
        bars = axes[1, 0].bar(labels, bias_vals, color=bar_colors, width=0.5)
        axes[1, 0].axhline(0, color='black', lw=0.8)
        axes[1, 0].set_ylabel('mean(nn - target)')
        axes[1, 0].set_title('WDL bias (red=over, blue=under)')
        for bar, v in zip(bars, bias_vals):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                v + (0.001 if v >= 0 else -0.003),
                f'{v:+.4f}', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=9)

        ce_vals = [float(ce_components[i]) for i in range(3)]
        axes[1, 1].bar(labels, ce_vals, color=colors, width=0.5)
        axes[1, 1].set_ylabel('mean CE contribution (nats)')
        axes[1, 1].set_title(f'CE by component  total={val_ce:.3f}')
        for i, v in enumerate(ce_vals):
            axes[1, 1].text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    else:
        for ax in (axes[0, 2], axes[1, 0], axes[1, 1]):
            ax.set_visible(False)

    axes[1, 2].set_visible(False)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
