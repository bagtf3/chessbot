"""Generate policy head illustration — boards showing top moves with probabilities."""
import chess
import chess.svg
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent.parent / "docs" / "images"

BG      = "#1a1a2e"
PURPLE  = "#a78bfa"
GREEN   = "#34d399"
DIM     = "#94a3b8"
TXT     = "#e2e8f0"
BAR_W   = "#6d28d9"
BAR_B   = "#065f46"

BOARD_COLORS = {
    "square light":          "#2d2d44",
    "square dark":           "#1a1a2e",
    "square light lastmove": "#4c1d95",
    "square dark lastmove":  "#4c1d95",
    "margin":                "#1a1a2e",
    "coord":                 "#94a3b8",
}

# Row 1: white's options from starting position
WHITE_MOVES = [
    ("e2e4", 0.28, "e4"),
    ("d2d4", 0.22, "d4"),
    ("g1f3", 0.12, "Nf3"),
    ("c2c4", 0.10, "c4"),
]

# Row 2: black's responses after 1.e4
BLACK_MOVES = [
    ("e7e5", 0.35, "e5"),
    ("c7c5", 0.30, "c5"),
    ("e7e6", 0.15, "e6"),
    ("c7c6", 0.10, "c6"),
]


def board_to_img(board: chess.Board, uci: str, size: int = 260,
                 arrow_color: str = PURPLE) -> np.ndarray:
    move = chess.Move.from_uci(uci)
    svg = chess.svg.board(
        board,
        arrows=[chess.svg.Arrow(move.from_square, move.to_square,
                                color=arrow_color)],
        colors=BOARD_COLORS,
        size=size,
        coordinates=True,
        orientation=chess.WHITE,
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    return mpimg.imread(BytesIO(png))


def draw_row(axes_board, axes_bar, moves, board, bar_color, row_label):
    for col, (uci, prob, label) in enumerate(moves):
        img = board_to_img(board, uci, arrow_color=bar_color)

        ax_b = axes_board[col]
        ax_b.imshow(img)
        ax_b.axis("off")
        ax_b.set_facecolor(BG)
        ax_b.set_title(label, color=TXT, fontsize=12, fontweight="bold",
                       fontfamily="monospace", pad=5)

        ax_bar = axes_bar[col]
        ax_bar.set_facecolor(BG)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(0, 1)
        ax_bar.axis("off")
        ax_bar.barh(0.5, 1.0, height=0.6, color="#2d2d44", left=0)
        ax_bar.barh(0.5, prob, height=0.6, color=bar_color, left=0)
        ax_bar.text(0.5, 0.5, f"{int(prob*100)}%",
                    ha="center", va="center", fontsize=11,
                    color=TXT, fontweight="bold", fontfamily="monospace",
                    zorder=5)

    # row label on the left
    axes_board[0].text(-0.12, 0.5, row_label, transform=axes_board[0].transAxes,
                       ha="center", va="center", fontsize=8.5, color=DIM,
                       fontfamily="monospace", rotation=90)


# layout: board1, bar1, spacer, board2, bar2
fig, axes = plt.subplots(
    5, 4, figsize=(11, 12),
    gridspec_kw={"height_ratios": [4, 0.8, 0.3, 4, 0.8], "hspace": 0.06}
)
fig.patch.set_facecolor(BG)

# spacer row
for ax in axes[2]:
    ax.set_visible(False)

# row 1 — white to move, starting position
start = chess.Board()
draw_row(axes[0], axes[1], WHITE_MOVES, start, BAR_W, "root")

# row 2 — black to move, after 1.e4
after_e4 = chess.Board()
after_e4.push_uci("e2e4")
draw_row(axes[3], axes[4], BLACK_MOVES, after_e4, GREEN, "after 1.e4")

# divider line
line_y = (axes[1][0].get_position().y0 + axes[3][0].get_position().y1) / 2
fig.add_artist(plt.Line2D([0.07, 0.97], [line_y, line_y],
                           transform=fig.transFigure,
                           color="#3d3d5c", linewidth=1.2, linestyle="--"))

fig.suptitle("Policy head: probability distribution over legal moves",
             color=DIM, fontsize=10, fontfamily="monospace", y=0.98)

plt.savefig(OUT / "policy_diagram.png", dpi=130, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("saved policy_diagram.png")
