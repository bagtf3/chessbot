"""Per-bin search-effort bin key, shared by the offline builder and the looper.

A game samples an alpha; each ply's sims floor and ceiling are multiplied by
the scale for its (phase, |parentQ|, parent_p_draw) bin. The looper loads
effort_scale_sweep.json once and looks up bin_key(...) per move.
"""
Q_EDGES   = (0.0, 0.1, 0.2, 0.5, 1.0)
D_EDGES   = (0.0, 0.3, 0.6, 0.7, 1.0)
PLY_EDGES = (16, 31, 61, 101)


def phase_of(ply, ply_edges=PLY_EDGES):
    for i, e in enumerate(ply_edges):
        if ply < e:
            return i
    return len(ply_edges)


def band_label(x, edges):
    for i in range(len(edges) - 1):
        if x < edges[i + 1]:
            return f"[{edges[i]:.1f},{edges[i + 1]:.1f})"
    return f"[{edges[-2]:.1f},{edges[-1]:.1f})"


def bin_key(ply, q_white, p_draw,
            q_edges=Q_EDGES, d_edges=D_EDGES, ply_edges=PLY_EDGES):
    return (f"{phase_of(ply, ply_edges)}|{band_label(abs(q_white), q_edges)}|"
            f"{band_label(p_draw, d_edges)}")
