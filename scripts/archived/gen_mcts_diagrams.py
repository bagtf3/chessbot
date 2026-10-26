"""Generate MCTS explanation diagrams for docs."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import chess
import chess.svg
from pathlib import Path

OUT = Path(__file__).parent.parent / "docs" / "images"

BG       = "#1a1a2e"
SURFACE  = "#16213e"
NODE     = "#2d2d44"
SEL      = "#6d28d9"   # purple - selected path
NEW      = "#065f46"   # green  - new/expanded node
EVAL_CLR = "#92400e"   # amber  - being evaluated
EDGE     = "#4b4b6e"
EDGE_SEL = "#a78bfa"
TXT      = "#e2e8f0"
TXT_DIM  = "#94a3b8"
ACCENT   = "#a78bfa"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "text.color":       TXT,
    "font.family":      "monospace",
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def node(ax, x, y, label, n, q, color=NODE, r=0.38, alpha=1.0):
    c = plt.Circle((x, y), r, color=color, zorder=3, alpha=alpha,
                   linewidth=1.5, ec="#ffffff18")
    ax.add_patch(c)
    ax.text(x, y + 0.13, label,  ha="center", va="center",
            fontsize=9, color=TXT, fontweight="bold", zorder=5)
    ax.text(x, y - 0.08, f"N={n}", ha="center", va="center",
            fontsize=7.5, color=TXT_DIM, zorder=5)
    ax.text(x, y - 0.24, f"Q={q:+.2f}", ha="center", va="center",
            fontsize=7.5, color=TXT_DIM, zorder=5)


def edge(ax, x1, y1, x2, y2, color=EDGE, lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12),
                zorder=2)


def label_box(ax, x, y, text, color=ACCENT):
    ax.text(x, y, text, ha="center", va="center", fontsize=8,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=color, lw=1.2),
            zorder=6)


def setup_ax(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Figure 1 — SELECT: walk the tree following best PUCT
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor(BG)
setup_ax(ax, (-0.5, 6.5), (-0.3, 4.2))

ax.set_title("MCTS — Step 1: Select", color=TXT, fontsize=12,
             fontweight="bold", pad=10)

# tree layout  (x, y, label, N, Q)
nodes = {
    "root": (3.0, 3.6, "root", 120, +0.12),
    "e4":   (1.5, 2.4, "e4",   68, +0.18),
    "d4":   (3.0, 2.4, "d4",   41, +0.08),
    "Nf3":  (4.5, 2.4, "Nf3",  11, +0.03),
    "e4c5": (0.8, 1.1, "c5",   42, +0.22),
    "e4e5": (2.2, 1.1, "e5",   26, +0.11),
    "d4d5": (3.0, 1.1, "d5",   41, +0.08),
}

# edges — default color
edge(ax, 3.0, 3.6, 1.5, 2.4)
edge(ax, 3.0, 3.6, 3.0, 2.4)
edge(ax, 3.0, 3.6, 4.5, 2.4)
edge(ax, 1.5, 2.4, 0.8, 1.1)
edge(ax, 1.5, 2.4, 2.2, 1.1)
edge(ax, 3.0, 2.4, 3.0, 1.1)

# selected path highlighted
edge(ax, 3.0, 3.6, 1.5, 2.4, color=EDGE_SEL, lw=2.8)
edge(ax, 1.5, 2.4, 0.8, 1.1, color=EDGE_SEL, lw=2.8)

# draw nodes
for k, (x, y, lbl, n, q) in nodes.items():
    if k in ("root", "e4", "e4c5"):
        node(ax, x, y, lbl, n, q, color=SEL)
    else:
        node(ax, x, y, lbl, n, q)

# annotations
label_box(ax, 5.6, 3.6, "highest\nPUCT", ACCENT)
label_box(ax, 5.6, 2.4, "still high\nPUCT", ACCENT)
label_box(ax, 0.8, 0.3, "leaf reached\n→ expand next", "#34d399")

ax.text(0.2, 4.05,
        "Follow the path of highest PUCT scores until a leaf is reached.",
        fontsize=8, color=TXT_DIM, style="italic")

plt.tight_layout()
plt.savefig(OUT / "mcts_select.png", dpi=130, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("saved mcts_select.png")


# ---------------------------------------------------------------------------
# Figure 2 — EXPAND + EVALUATE: add a new node and call the network
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor(BG)
setup_ax(ax, (-0.5, 6.5), (-0.8, 4.2))

ax.set_title("MCTS — Step 2: Expand & Evaluate", color=TXT, fontsize=12,
             fontweight="bold", pad=10)

# same tree structure, faded non-selected branches
edge(ax, 3.0, 3.6, 1.5, 2.4, color=EDGE_SEL, lw=2.8)
edge(ax, 3.0, 3.6, 3.0, 2.4, color=EDGE, lw=1.2)
edge(ax, 3.0, 3.6, 4.5, 2.4, color=EDGE, lw=1.2)
edge(ax, 1.5, 2.4, 0.8, 1.1, color=EDGE_SEL, lw=2.8)
edge(ax, 1.5, 2.4, 2.2, 1.1, color=EDGE, lw=1.2)
edge(ax, 3.0, 2.4, 3.0, 1.1, color=EDGE, lw=1.2)

# new child edges from the leaf
edge(ax, 0.8, 1.1, -0.1, -0.1, color="#34d399", lw=2.2)
edge(ax, 0.8, 1.1,  0.8, -0.1, color="#34d399", lw=2.2)
edge(ax, 0.8, 1.1,  1.7, -0.1, color="#34d399", lw=2.2)

node(ax, 3.0, 3.6, "root", 120, +0.12, color=SEL)
node(ax, 1.5, 2.4, "e4",    68, +0.18, color=SEL)
node(ax, 3.0, 2.4, "d4",    41, +0.08, alpha=0.45)
node(ax, 4.5, 2.4, "Nf3",   11, +0.03, alpha=0.45)
node(ax, 0.8, 1.1, "e4/c5", 42, +0.22, color=SEL)
node(ax, 2.2, 1.1, "e5",    26, +0.11, alpha=0.45)
node(ax, 3.0, 1.1, "d5",    41, +0.08, alpha=0.45)

# new leaf nodes
for xn, lbl in [(-0.1, "Nf3"), (0.8, "d4"), (1.7, "Nc6")]:
    node(ax, xn, -0.1, lbl, 0, 0.0, color=NEW)

# NN eval callout
ax.annotate("",
            xy=(0.8, -0.1), xytext=(0.8, -0.6),
            arrowprops=dict(arrowstyle="<-", color="#fbbf24", lw=2))
ax.text(0.8, -0.78,
        "NN → policy + value\n(W 52%, D 31%, L 17%)",
        ha="center", va="center", fontsize=8, color="#fbbf24",
        bbox=dict(boxstyle="round,pad=0.35", fc=BG, ec="#fbbf24", lw=1.2))

ax.text(0.2, 4.05,
        "Expand the leaf. Run the neural network to get policy priors and a WDL value.",
        fontsize=8, color=TXT_DIM, style="italic")

plt.tight_layout()
plt.savefig(OUT / "mcts_expand.png", dpi=130, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("saved mcts_expand.png")


# ---------------------------------------------------------------------------
# Figure 3 — BACKUP: propagate value up the selected path
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor(BG)
setup_ax(ax, (-0.5, 6.5), (-0.3, 4.2))

ax.set_title("MCTS — Step 3: Back Up", color=TXT, fontsize=12,
             fontweight="bold", pad=10)

edge(ax, 3.0, 3.6, 3.0, 2.4, color=EDGE)
edge(ax, 3.0, 3.6, 4.5, 2.4, color=EDGE)
edge(ax, 1.5, 2.4, 2.2, 1.1, color=EDGE)
edge(ax, 3.0, 2.4, 3.0, 1.1, color=EDGE)

# backup path — reversed arrow direction to show value flowing up
for (x1, y1, x2, y2) in [(0.8, 1.1, 1.5, 2.4), (1.5, 2.4, 3.0, 3.6)]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#fbbf24",
                                lw=2.8, mutation_scale=14), zorder=2)

edge(ax, 3.0, 3.6, 1.5, 2.4, color=EDGE_SEL, lw=1.6)
edge(ax, 1.5, 2.4, 0.8, 1.1, color=EDGE_SEL, lw=1.6)

# nodes — show updated N/Q after backup
node(ax, 3.0, 3.6, "root", 121, +0.13, color=SEL)
node(ax, 1.5, 2.4, "e4",    69, +0.19, color=SEL)
node(ax, 3.0, 2.4, "d4",    41, +0.08)
node(ax, 4.5, 2.4, "Nf3",   11, +0.03)
node(ax, 0.8, 1.1, "e4/c5", 43, +0.23, color=SEL)
node(ax, 2.2, 1.1, "e5",    26, +0.11)
node(ax, 3.0, 1.1, "d5",    41, +0.08)

# value labels on the path
for (x, y, val) in [(0.8, 0.45, "+0.52"), (2.05, 1.75, "+0.52"), (3.6, 3.05, "+0.52")]:
    ax.text(x, y, val, fontsize=8, color="#fbbf24", fontweight="bold",
            ha="center", zorder=6)

label_box(ax, 5.4, 1.8, "value flips\neach ply\n(chess alternates)", "#f87171")

ax.text(0.2, 4.05,
        "Propagate the evaluated value back through every node on the selected path.",
        fontsize=8, color=TXT_DIM, style="italic")

plt.tight_layout()
plt.savefig(OUT / "mcts_backup.png", dpi=130, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("saved mcts_backup.png")
