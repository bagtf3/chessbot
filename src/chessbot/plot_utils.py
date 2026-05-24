import numpy as np


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
