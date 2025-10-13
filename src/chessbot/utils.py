import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import math, random, time, pickle
from time import time as _now

import tensorflow as tf
import matplotlib.pyplot as plt

import chess
import chess.engine
import chess.svg
from IPython.display import SVG, display, clear_output

from pyfastchess import Board as fastboard

from chessbot import SF_LOC
from chessbot import features as ft

from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


uci_path_path =  r"C:/Users/Bryan/Data/chessbot_data/uci_paths3000.pkl"  
with open(uci_path_path, "rb") as f:
    paths = pickle.load(f)
    

def rnd(x, n):
    return np.round(x, n)


def show_board(board, flipped=False, sleep=0.1):
    clear_output(wait=True)
    display(SVG(chess.svg.board(board=board, flipped=flipped)))
    time.sleep(sleep)
    
    
def mirror_board(board):
    return board.mirror()


def mirror_move(mv: chess.Move) -> chess.Move:
    """Mirror a move through the board center (promotion piece is unchanged)."""
    return chess.Move(
        chess.square_mirror(mv.from_square),
        chess.square_mirror(mv.to_square),
        promotion=mv.promotion
    )


def make_random_move(board):
    if isinstance(board, chess.Board):
        moves = list(board.legal_moves)
        board.push(random.choice(moves))
        return board
    
    if isinstance(board, fastboard):
        moves = list(board.legal_moves())
        board.push_uci(random.choice(moves))
        return board    


def get_all_board_features(board):
    all_feats = ft.all_king_exposure_features(board)
    all_feats.update(ft.all_piece_features(board))
    
    wt, blk = ft.get_piece_value_sum(board)
    all_feats['material'] = [wt, blk]
    
    return all_feats


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    y = np.exp(x)
    s = float(np.sum(y))
    return y / s if s > 0 else np.full_like(y, 1.0 / len(y))


def ensure_df(df_or_dicts):
    if isinstance(df_or_dicts, pd.DataFrame):
        return df_or_dicts.copy()
    return pd.DataFrame(df_or_dicts).copy()


def plot_sf_simple(df):
    """
    Line plot over epochs showing:
      - wins (total)
      - draws (total)
      - draws as White
      - draws as Black
      - average game length (plies) on a secondary axis
    """
    df = ensure_df(df).sort_values("epoch").reset_index(drop=True)

    ep = df["epoch"].astype(int).values
    wins_total   = df.get("wins",         0).fillna(0).astype(int).values
    draws_total  = df.get("draws",        0).fillna(0).astype(int).values
    white_draws  = df.get("white_draws",  0).fillna(0).astype(int).values
    black_draws  = df.get("black_draws",  0).fillna(0).astype(int).values
    avg_ply      = df.get("avg_ply",    np.nan).values / 5

    fig, ax1 = plt.subplots(figsize=(10,4))

    # counts (left axis)
    ax1.plot(ep, wins_total,  lw=1.5, label="Wins (total)")
    ax1.plot(ep, draws_total, lw=1.5, label="Draws (total)")
    ax1.plot(ep, white_draws, lw=1.2, alpha=0.9, label="Draws as White")
    ax1.plot(ep, black_draws, lw=1.2, alpha=0.9, label="Draws as Black")
    ax1.plot(ep, avg_ply,     lw=1.5, color='gray', label="Avg plies / 5")
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Counts")
    ax1.grid(True, alpha=0.3)

    # combined legend
    l1, lab1 = ax1.get_legend_handles_labels()
    ax1.legend(l1, lab1, loc="upper left")

    plt.title("SF600 exhibitions: wins/draws and avg game length")
    plt.tight_layout()
    plt.show()
    

def plot_training_progress(all_evals, max_cols=4, save_path=None):
    """
    Plots training/eval metrics for each model output in a grid.
    Adapts rows/cols automatically, up to max_cols wide.

    Parameters
    ----------
    all_evals : pd.DataFrame
        DataFrame containing eval metrics with columns as outputs.
    max_cols : int, optional
        Max number of columns in the plot grid (default 4).
    save_path : str or Path, optional
        If provided, saves the plot image to this file location.
    """
    cols = list(all_evals.columns)

    # Always keep "value" last for consistency
    important = ['value']
    cols = [c for c in cols if c not in important] + important

    n_plots = len(cols)
    n_cols = min(max_cols, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        y = all_evals[col].values
        ax.plot(y, label=col)
        # moving average (MA15)
        ma = pd.Series(y).rolling(15, min_periods=1).mean().values
        ax.plot(ma, lw=2, alpha=0.6, label=f"{col} (MA15)")
        ax.set_title(col)
        ax.legend()

    # Hide unused axes
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    

def plot_pred_vs_true_grid(model, preds, y_true_dict, save_path=None):
    names = list(model.output_names)

    chunk_size = 9
    n_chunks = math.ceil(len(names) / chunk_size)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_names = names[start:end]

        fig, axes = plt.subplots(3, 3, figsize=(14, 8))
        axes = axes.flatten()

        for ax, name in zip(axes, chunk_names):
            if name not in y_true_dict:
                ax.set_visible(False)
                continue

            y_pred = np.asarray(preds[name])
            y_true = np.asarray(y_true_dict[name])

            # squeeze singleton dims
            if y_pred.ndim > 1 and y_pred.shape[-1] == 1:
                y_pred = y_pred.reshape(-1)
            if y_true.ndim > 1 and y_true.shape[-1] == 1 and name == "value":
                y_true = y_true.reshape(-1)

            if name == "value":
                # regression scatter
                yp = y_pred.reshape(-1)
                yt = y_true.reshape(-1)
                n = min(len(yp), len(yt))
                yp, yt = yp[:n], yt[:n]
                ax.scatter(yt, yp, s=8, alpha=0.5)
                lo = float(min(yt.min(), yp.min()))
                hi = float(max(yt.max(), yp.max()))
                ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
                ax.set_title("value")
                ax.set_xlabel("True")
                ax.set_ylabel("Pred")
                ax.grid(True, alpha=0.3)
                continue

            # classification heads: handle soft or sparse y_true
            pred_classes = y_pred.argmax(axis=1)

            if y_true.ndim == 2:
                true_classes = y_true.argmax(axis=1)
            else:
                true_classes = y_true.reshape(-1)

            pred_classes = pred_classes.reshape(-1)
            true_classes = true_classes.reshape(-1)

            n = min(len(true_classes), len(pred_classes))
            ax.scatter(true_classes[:n], pred_classes[:n], s=5, alpha=0.5)
            ax.set_title(name)
            ax.set_xlabel("True class")
            ax.set_ylabel("Pred class")
            ax.grid(True, alpha=0.3)

        for i in range(len(chunk_names), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        # optional: name the window per chunk
        try:
            fig.canvas.manager.set_window_title(f"Pred vs True [{start}:{end}]")
        except Exception:
            pass
        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show(block=False)


def top_k_accuracy(y_true, y_pred, k=3):
    """
    Computes Top-K accuracy for classification heads.
    y_true : (N,) int labels
    y_pred : (N, C) logits or probs
    k      : how many top guesses to consider
    """
    # convert logits -> probs
    probs = tf.nn.softmax(y_pred, axis=-1).numpy()
    # indices of top-k per sample
    topk = np.argpartition(-probs, k, axis=1)[:, :k]
    # check if true label is in top-k
    correct = [y_true[i] in topk[i] for i in range(len(y_true))]
    return np.mean(correct)


def _to_sparse_labels(y):
    y = np.asarray(y)
    if y.ndim == 2:  # one-hot / soft
        return y.argmax(axis=1).astype(np.int64)
    return y.reshape(-1).astype(np.int64)


def _topk_from_logits(y_true_sparse, y_pred_logits, k=1):
    y_pred = np.asarray(y_pred_logits)
    topk = np.argpartition(-y_pred, kth=min(k, y_pred.shape[1]-1), axis=1)[:, :k]
    # count hits
    hits = (topk == y_true_sparse[:, None]).any(axis=1)
    return float(hits.mean())


def score_game_data(model, X, Y_batch, save_path=None):
    raw_preds = model.predict(X, batch_size=256, verbose=0)
    preds = {name: raw_preds[i] for i, name in enumerate(model.output_names)}

    # Plot overview grid
    plot_pred_vs_true_grid(model, preds, Y_batch, save_path=save_path)

    # Keras evaluate -> dataframe row
    cols = ['total_loss'] + model.output_names
    eval_df = pd.DataFrame(model.evaluate(X, Y_batch, verbose=0), index=cols).T

    # ----- Value head metrics -----
    yt_val = np.asarray(Y_batch['value']).reshape(-1)
    yp_val = np.asarray(preds['value']).reshape(-1)

    val_mse = float(np.mean((yt_val - yp_val) ** 2))

    # safe correlation (avoid NaNs if std=0)
    yt_std = yt_val.std()
    yp_std = yp_val.std()
    if yt_val.size > 1 and yt_std > 0 and yp_std > 0:
        corr = float(np.corrcoef(yt_val, yp_val)[0, 1])
    else:
        corr = 0.0

    # ----- Classification metrics -----
    cls_metrics = {}
    for head in [n for n in model.output_names if n != "value"]:
        y_true_sparse = _to_sparse_labels(Y_batch[head])
        y_pred_sparse = np.asarray(preds[head]).argmax(axis=1)
        acc = float((y_true_sparse == y_pred_sparse).mean())
        cls_metrics[head] = acc

    print("\n=== Extra Metrics ===")
    print(f"Value : MSE={val_mse:.4f}, Corr={corr:.3f}")

    return eval_df


def log_and_plot_sf(intra_training_summaries, show=True, save_path=None):
    """
    Pretty-print the latest Stockfish summary and plot simple trends across retrains.

    Args:
        intra_training_summaries: {step: {"summary": {...}, "games": [...]}, ...}
        show: if True, plt.show() the figures (ignored if save_path is given)
        save_path: if set, save figures as f"{save_path}_cpl.png" and f"{save_path}_bmr.png"

    Prints:
        Latest step's games, CPL (overall/white/black), and best-move rates.
    Plots (if >=2 steps exist):
        1) Overall mean CPL vs step
        2) Overall best-move rate (%) vs step
    """
    its = intra_training_summaries
    
    if not its:
        print("No SF analysis yet.")
        return

    steps = sorted(its.keys())
    latest_step = steps[-1]
    s = its[latest_step]["summary"]

    # Pretty print latest
    print("\n=== Stockfish Analysis (latest) ===")
    print(f"Retrain step: {latest_step}")
    print(f"Games analyzed: {int(s['games'])}")
    print(f"Overall mean CPL: {float(s['avg_overall_mean_cpl']):.3f}")
    print(f"  - White mean CPL: {float(s['avg_white_mean_cpl']):.3f}")
    print(f"  - Black mean CPL: {float(s['avg_black_mean_cpl']):.3f}")
    print(f"Overall best-move rate: {float(s['avg_overall_best_move_rate'])*100:.1f}%")
    print(f"  - White best-move rate: {float(s['avg_best_move_rate_white'])*100:.1f}%")
    print(f"  - Black best-move rate: {float(s['avg_best_move_rate_black'])*100:.1f}%")
    print("===================================\n")

    # Need >=2 points to plot a trend
    if len(steps) < 2:
        return

    # Build series
    overall_cpl = [its[k]["summary"]["avg_overall_mean_cpl"] for k in steps]
    overall_bmr_pct = [
        its[k]["summary"]["avg_overall_best_move_rate"] * 100.0 for k in steps
    ]

    # 1) Overall mean CPL trend
    fig1 = plt.figure(figsize=(6.4, 3.6))
    plt.plot(steps, overall_cpl, marker="o")
    plt.title("Overall Mean CPL vs Retrain Step")
    plt.xlabel("Retrain step")
    plt.ylabel("Mean CPL (lower is better)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        fig1.savefig(f"{save_path}_cpl.png", dpi=150)
    
    if show:
        plt.show()
    plt.close(fig1)

    # 2) Overall best-move rate (%) trend
    fig2 = plt.figure(figsize=(6.4, 3.6))
    plt.plot(steps, overall_bmr_pct, marker="o")
    plt.title("Overall Best-Move Rate vs Retrain Step")
    plt.xlabel("Retrain step")
    plt.ylabel("Best-move rate (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        fig2.savefig(f"{save_path}_bmr.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig2)


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.3f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.2f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.2f}s"

##    
## board/ position generation
##

OPENING_BOOK = {
    "Ruy Lopez, Morphy Defense": [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"
    ],
    "Italian Game (Giuoco Piano)": [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"
    ],
    "Scotch Game": [
        "e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "g8f6"
    ],
    "Sicilian Defense, Najdorf": [
        "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"
    ],
    "Sicilian Defense, Dragon": [
        "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"
    ],
    "French Defense, Classical": [
        "e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7"
    ],
    "Caro-Kann, Advance": [
        "e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5", "c2c4", "e7e6"
    ],
    "Caro-Kann, Classical": [
        "e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"
    ],
    "Queen's Gambit Declined": [
        "d2d4", "d7d5", "c2c4", "e7e6", "g1f3", "g8f6", "b1c3", "c7c6"
    ],
    "Queen's Gambit Accepted": [
        "d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6"
    ],
    "Slav Defense": [
        "d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4"
    ],
    "Nimzo-Indian Defense": [
        "d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"
    ],
    "King's Indian Defense": [
        "d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"
    ],
    "Grünfeld Defense": [
        "d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"
    ],
    "London System": [
        "d2d4", "d7d5", "c1f4", "g8f6", "e2e3", "c7c5", "c2c3", "b8c6"
    ],
    "English Opening, Four Knights": [
        "c2c4", "e7e5", "g1f3", "b8c6", "g2g3", "g8f6", "f1g2", "f8c5"
    ],
    "English Opening, Symmetrical": [
        "c2c4", "c7c5", "g1f3", "g8f6", "d2d4", "c5d4", "f3d4", "b8c6"
    ],
    "Scandinavian Defense": [
        "e2e4", "d7d5", "e4d5", "d8d5", "g1f3", "c8g4", "f1e2", "g4f3"
    ],
    "Pirc Defense": [
        "e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f2f4", "f8g7"
    ],
    "Modern Defense": [
        "e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "d7d6", "f2f4", "c7c5"
    ],
    # Ultra-canonical stubs
    "Double King Pawn (e4 e5)": ["e2e4", "e7e5"],
    "Double Queen Pawn (d4 d5)": ["d2d4", "d7d5"],
    "Sicilian Defense Stub (e4 c5)": ["e2e4", "c7c5"],
    "French Defense Stub (e4 e6)": ["e2e4", "e7e6"],
    "Caro-Kann Stub (e4 c6)": ["e2e4", "c7c6"],
    "Pirc Stub (e4 d6)": ["e2e4", "d7d6"],
    "Modern Stub (e4 g6)": ["e2e4", "g7g6"],
    "English Opening Stub (c4)": ["c2c4"],
    "Reti Stub (Nf3)": ["g1f3"],
    "Indian Defense Stub (d4 Nf6)": ["d2d4", "g8f6"],
    "Dutch Stub (d4 f5)": ["d2d4", "f7f5"],
    "Benoni Stub (d4 c5)": ["d2d4", "c7c5"],
    "Catalan Stub (d4 Nf6 c4 e6 g3)": ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"],
    "London Stub (d4 d5 Bf4)": ["d2d4", "d7d5", "c1f4"],
    "King’s Indian Stub (d4 Nf6 c4 g6)": ["d2d4", "g8f6", "c2c4"],
    "Grünfeld Stub (d4 Nf6 c4 g6 Nc3 d5)": ["d2d4", "g8f6", "c2c4", "g7g6"]
}


def get_opening(name=None):
    """
    Return a board set up in a chosen or random opening.
    """
    if name is None:
        name = random.choice(list(OPENING_BOOK.keys()))
    moves = OPENING_BOOK[name]
    
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return name, board


def get_all_openings():
    names, boards = [], []
    for O in OPENING_BOOK.keys():
        names.append(O)
        n, b = get_opening(O)
        boards.append(b)
    
    return names, boards


def random_init(plies=5, python_chess=False):
    b = fastboard()
    p = 0
    counter = 0
    limit = int(3*plies)
    while p < plies:
        # safety in case it gets caught in some weird corner
        counter += 1
        if counter > limit:
            # really bad luck, just try again
            return random_init(plies=plies)
        
        moves = b.legal_moves()
        while not len(moves):
            b.unmake()
            moves = b.legal_moves()
            p -= 1
            if p == 0:
                # more bad luck, startover
                return random_init(plies=plies)
        
        b.push_uci(random.choice(moves))
        p += 1
    
    if python_chess:
        b = chess.Board(b.fen())
    
    return b


def greedy_sf_tree_paths(n_positions=2000, multipv=4, max_depth=7, eval_thresh=150):
    """
    Return a list of UCI move lists (paths) from STARTPOS via greedy BFS.
    Includes STARTPOS as [] and all intermediate paths until n_positions reached.

    Args:
        n_positions : target number of positions (approx; includes STARTPOS)
        multipv     : how many top moves to expand per node
        max_depth   : Stockfish search depth for analysis
        eval_thresh : only expand moves within (best_cp - cp) <= eval_thresh
        sf_path     : path to stockfish binary; if None, uses chess.engine default

    Returns:
        paths : list[list[str]] such as:
            [ [],
              ['e2e4'], ['d2d4'], ['c2c4'], ['g1f3'],
              ['e2e4','d7d5'], ... ]
    """
    
    def short_fen(fen):
        return " ".join(fen.split(" ")[:4])
    
    # Seed: STARTPOS plus four common first moves
    start = chess.Board()
    seed_sans = ["e4", "d4", "Nf3", "c4"]

    paths = []                 # output paths
    seen = set()               # short-FEN dedup
    q = deque()                # queue of (board, path)

    # Add STARTPOS
    paths.append([])                       # []
    seen.add(short_fen(start.fen()))

    # Enqueue seeds
    for san in seed_sans:
        mv = start.parse_san(san)
        b2 = start.copy(); b2.push(mv)
        q.append( (b2, [mv.uci()]) )

    eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)

    try:
        while q and len(paths) < n_positions:
            board, path = q.popleft()
            key = short_fen(board.fen())
            if key in seen:
                continue

            seen.add(key)
            paths.append(path)

            if board.is_game_over():
                continue

            info = eng.analyse(
                board, chess.engine.Limit(depth=max_depth), multipv=multipv
            )
            if not info:
                continue
            
            best_cp = info[0]["score"].pov(board.turn).score(mate_score=1500)
            if best_cp is None:
                best_cp = 0

            for d in info:
                sc = d["score"].pov(board.turn).score(mate_score=1500)
                if sc is None:
                    continue
                if best_cp - sc > eval_thresh:
                    continue
                if "pv" not in d or not d["pv"]:
                    continue
                mv = d["pv"][0]
                b2 = board.copy()
                b2.push(mv)
                q.append( (b2, path + [mv.uci()]) )
    finally:
        eng.quit()

    return paths

    
def get_pre_opened_game():
    b = fastboard()
    moves_to_play = random.choice(paths)
    for mtp in moves_to_play:
        b.push_uci(mtp)
    return b


def random_board_setup(pieces, wk=None, bk=None, queens=True, pyfast=True):
    """
    Make a legal endgame-like position with exactly `pieces` total pieces.
    Pieces are drawn from a bag proportional to a real starting set:
      per color: 8P, 2N, 2B, 2R, 1Q (plus the king already placed)
    Constraints:
      - pawns never on 1st/8th rank for their color
      - never exceed real caps per piece type per color
      - board is valid and not immediately game-over
      - if ensure_move, side to move has at least one legal move
    """
    if pieces < 6:
        raise ValueError("pieces must be >= 6")
    if pieces > 32:
        raise ValueError("pieces must be <= 32")

    # Max counts per color, mirroring real chess
    cap = {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
    }
    
    if not queens:
        cap[chess.QUEEN] = 0
    
    pool_piece_types = [
        chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN
    ]
    colors = [chess.WHITE, chess.BLACK]

    # Squares where pawns are allowed
    pawn_ok_squares = set([i for i in range(8, 56)])

    # Weighted draw from the remaining bag
    def draw_piece_type_and_color(rem_white, rem_black):
        bag = []
        # fill bag with counts so probability ∝ remaining allowed
        for pt in pool_piece_types:
            for _ in range(rem_white[pt]):
                bag.append((pt, chess.WHITE))
            for _ in range(rem_black[pt]):
                bag.append((pt, chess.BLACK))
        if not bag:
            return None
        return random.choice(bag)

    # Place a non-pawn on any empty square; pawn on allowed rank squares
    def place_piece(board, piece_type, color):
        empties = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]
        if not empties:
            return False

        if piece_type == chess.PAWN:
            candidates = [sq for sq in empties if sq in pawn_ok_squares]
        else:
            candidates = empties

        if not candidates:
            return False

        sq = random.choice(candidates)
        board.set_piece_at(sq, chess.Piece(piece_type, color))
        return True

    # Try until we get a valid, non-terminal position
    for _ in range(1000):
        board = chess.Board(None)

        # Kings first (any distinct squares is fine; validity checked later)
        
        wk = random.choice(chess.SQUARES) if wk is None else wk
        while True:
            bk = random.choice(chess.SQUARES) if bk is None else bk
            if bk != wk:
                break
            # prevents spinning forever
            bk= None

        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

        # Remaining counts per color
        rem_white = dict(cap)
        rem_black = dict(cap)

        # How many more to place
        need = pieces - 2
        ok_build = True

        while need > 0 and ok_build:
            choice = draw_piece_type_and_color(rem_white, rem_black)
            if choice is None:
                ok_build = False
                break

            pt, color = choice
            placed = place_piece(board, pt, color)
            if not placed:
                # If we failed to place this kind, remove this option once
                # by temporarily decrementing and continue; if it hits zero
                # it won't be drawn again.
                if color == chess.WHITE:
                    if rem_white[pt] > 0:
                        rem_white[pt] -= 1
                else:
                    if rem_black[pt] > 0:
                        rem_black[pt] -= 1
                continue

            # Successful placement consumes from that color's pool
            if color == chess.WHITE:
                rem_white[pt] -= 1
            else:
                rem_black[pt] -= 1
            need -= 1

        if not ok_build or need != 0:
            continue

        # Randomize side to move
        board.turn = random.choice(colors)

        # Final validity checks
        if not board.is_valid():
            # try to flip the turn
            board.turn = not board.turn
            if not board.is_valid():
                continue
        
        if board.is_game_over():
            continue
        if board.legal_moves.count() == 0:
            continue
        
        return fastboard(board.fen()) if pyfast else board
    
    #if we made it here, it couldnt work
    print("Unable to find valid board. Loosening requirements...")
    next_pieces = max(6, pieces-1)
    return random_board_setup(next_pieces, wk=None, bk=None, queens=False, pyfast=pyfast)


def make_piece_odds_board():
    b = chess.Board()
    meta = {"scenario": "piece_odds", "removed": {"white": [], "black": []}}

    remove_from = random.choice([chess.WHITE, chess.BLACK])
    removable = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    to_remove = random.choice(removable)
    need_to_remove = np.random.randint(1, 4) if to_remove == chess.PAWN else 1

    color_str = "white" if remove_from == chess.WHITE else "black"

    for _ in range(need_to_remove):
        # recompute candidates fresh each time (so we never hit an empty square)
        pool = [sq for sq, pc in b.piece_map().items()
                if pc.color == remove_from and pc.piece_type == to_remove]
        if not pool:
            break  # nothing left of this type/color. bail

        sq = random.choice(pool)
        pc = b.remove_piece_at(sq)
        meta["removed"][color_str].append(pc.symbol())

    return fastboard(b.fen()), meta


def make_piece_training_board():
    fens = {
        "rooks": "rrrrkrrr/pppppppp/8/8/8/8/PPPPPPPP/RRRRKRRR w - - 0 1",
        "bishops": "bbbbkbbb/pppppppp/8/8/8/8/PPPPPPPP/BBBBKBBB w - - 0 1",
        "knights": "nnnnknnn/pppppppp/8/8/8/8/PPPPPPPP/NNNNKNNN w - - 0 1",
        "only_pawns": "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
        "b_vs_k":"bbbbkbbb/pppppppp/8/8/8/8/PPPPPPPP/NNNNKNNN w - - 0 1",
        "extra_queen": 'qnb1kbnq/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1'
    }
    
    pick = random.choice(list(fens.keys()))
    board = chess.Board(fens[pick])
    if np.random.uniform() < 0.5:
        board = board.mirror()
    return fastboard(board.fen()), {"scenario": pick}
        

def score_pov_cp(pov_score, white_to_move, mate_cp):
    s = pov_score.white() if white_to_move else pov_score.black()
    return float(s.score(mate_score=mate_cp))


def evaluate_game_sf(moves_uci, start_fen=None, depth=8, mate_cp=1500):
    board = chess.Board() if start_fen is None else chess.Board(start_fen)
    white_cpl, black_cpl = [], []
    best_moves = [0, 0]

    # diag counters
    drops = {
        "no_best_score": 0,
        "mv_illegal": 0,
        "no_played_score": 0,
        "engine_err": 0,
    }

    limit = chess.engine.Limit(depth=int(depth))
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
        for idx, u in enumerate(moves_uci):
            if board.is_game_over():
                break

            mv = None
            try:
                mv = chess.Move.from_uci(u)
            except Exception:
                mv = None

            try:
                root_for_best = board.copy(stack=False)
                root_for_play = board.copy(stack=False)
                white_to_move = root_for_best.turn

                bm = eng.play(root_for_best, limit=limit)
                best_uci = bm.move.uci() if (bm and bm.move) else None

                best_info = eng.analyse(
                    root_for_best, limit=limit,
                    info=chess.engine.INFO_SCORE
                )
                best_sc = best_info.get("score")
                best_cp = score_pov_cp(best_sc, white_to_move, mate_cp)

                played_cp = None
                if mv is not None and mv in root_for_play.legal_moves:
                    played_info = eng.analyse(
                        root_for_play, limit=limit,
                        root_moves=[mv], info=chess.engine.INFO_SCORE
                    )
                    played_sc = played_info.get("score")
                    played_cp = score_pov_cp(played_sc, white_to_move, mate_cp)
                else:
                    drops["mv_illegal"] += 1

                if best_uci is not None and best_uci == u:
                    best_moves[1 if white_to_move else 0] += 1

                if best_cp is None:
                    drops["no_best_score"] += 1
                if played_cp is None:
                    drops["no_played_score"] += 1

                if (best_cp is not None) and (played_cp is not None):
                    cpl = max(0.0, float(best_cp) - float(played_cp))
                    (white_cpl if white_to_move else black_cpl).append(cpl)

            except (chess.engine.EngineError, ValueError):
                drops["engine_err"] += 1

            # only push legal parsed moves
            if mv is not None and mv in board.legal_moves:
                board.push(mv)
            else:
                # stop if input game is desynced
                break

    # summaries (unchanged)
    def summary(arr):
        if not arr:
            return {"mean_cpl": 0.0, "max_cpl": 0.0, "n": 0}
        n = len(arr)
        return {
            "mean_cpl": round(sum(arr) / n, 3),
            "max_cpl": round(max(arr), 3),
            "n": n,
        }

    white = summary(white_cpl)
    black = summary(black_cpl)
    all_cpl = white_cpl + black_cpl
    overall = round(sum(all_cpl) / len(all_cpl), 3) if all_cpl else 0.0

    total_white_plies = max(1, white["n"])
    total_black_plies = max(1, black["n"])
    bm_rate_white = round(best_moves[1] / total_white_plies, 3)
    bm_rate_black = round(best_moves[0] / total_black_plies, 3)
    bm_overall = round(
        (best_moves[1] + best_moves[0]) /
        (total_white_plies + total_black_plies), 3
    )

    out = {
        "plies": white["n"] + black["n"],
        "white": white,
        "black": black,
        "overall_mean_cpl": overall,
        "best_move_rate_white": bm_rate_white,
        "best_move_rate_black": bm_rate_black,
        "overall_best_move_rate": bm_overall,
        "drops": drops,  # <-- diagnostics
    }
    return out


def evaluate_many_games(games, depth=12, workers=4, mate_cp=1500):
    def worker(moves, fen):
        return evaluate_game_sf(
            moves_uci=list(moves), start_fen=fen, depth=depth, mate_cp=mate_cp)

    N = len(games)
    results = [None] * N

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = {}
        for i, g in enumerate(games):
            futs[ex.submit(worker, g["moves"], g["start_fen"])] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = None

    # keep successful in-order
    results = [r for r in results if r is not None]

    # results is a list of per-game dicts from evaluate_game_sf
    if not results:
        return {"summary": {
            "games": 0,
            "avg_white_mean_cpl": 0.0,
            "avg_black_mean_cpl": 0.0,
            "avg_overall_mean_cpl": 0.0,
            "avg_best_move_rate_white": 0.0,
            "avg_best_move_rate_black": 0.0,
            "avg_overall_best_move_rate": 0.0,
        }, "games": []}
    
    w_means = [r["white"]["mean_cpl"] for r in results]
    b_means = [r["black"]["mean_cpl"] for r in results]
    o_means = [r["overall_mean_cpl"] for r in results]
    
    w_best = [r["best_move_rate_white"] for r in results]
    b_best = [r["best_move_rate_black"] for r in results]
    o_best = [r["overall_best_move_rate"] for r in results]
    
    def avg(xs): 
        return round(sum(xs) / len(xs), 3) if xs else 0.0
    
    summary = {
        "games": len(results),
        "avg_white_mean_cpl": avg(w_means),
        "avg_black_mean_cpl": avg(b_means),
        "avg_overall_mean_cpl": avg(o_means),
        "avg_best_move_rate_white": avg(w_best),
        "avg_best_move_rate_black": avg(b_best),
        "avg_overall_best_move_rate": avg(o_best),
    }
    
    return {"summary": summary, "games": results}


class RateMeter(object):
    def __init__(self, name, interval_s=120.0):
        self.name = name
        self.interval_s = float(interval_s)
        self.t0 = _now()
        self.t_last = self.t0
        self.total = 0
        self.last_total = 0

    def tick(self, n=1):
        self.total += int(n)

    def rate(self):
        # instantaneous rate since last maybe_report (or since start if never called)
        t = _now()
        dtime = t - self.t_last
        dtotal = self.total - self.last_total

        # update for next time
        self.t_last = t
        self.last_total = self.total

        if dtime <= 0:
            return 0.0
        return dtotal / dtime

    def maybe_report(self, extra=""):
        t = _now()
        dt = t - self.t_last
        if dt < self.interval_s:
            return None
        
        rate = self.rate()
        msg = f"{self.name}: {rate:.1f}/s  (total {self.total})"
        if extra:
            msg += f"  {extra}"
        print(msg)
        return rate


def summarize_recent_games(recent, result_is_bot_pov=True):
    """
    Per-(scenario,sf_bucket) stats (unchanged) + bot-vs-SF W/L/D totals.
    For the SF totals we assume `result` is WHITE-POV:
      r > 0 => white won, r < 0 => black won, r == 0 => draw
    This matches your own description for detecting bot wins vs SF.
    """
    def sf_bucket(vs_sf, sf_flag):
        if not vs_sf:
            return "none"
        return "white" if sf_flag else "black"

    stats = defaultdict(lambda: {"N": 0, "W": 0, "L": 0, "D": 0, "plies_sum": 0})

    # bot vs Stockfish totals only (what you care about)
    sf_overall = {"N": 0, "W": 0, "L": 0, "D": 0, "plies_sum": 0}

    for g in recent:
        scenario = g.get("scenario", "unknown")
        bucket = sf_bucket(g.get("vs_stockfish", False), g.get("stockfish_color"))
        key = (scenario, bucket)

        r = g.get("result", 0.0)

        # scenario table: keep your existing semantics (bot-POV by default)
        r_for_table = r
        if not result_is_bot_pov:
            sf_is_white = bool(g.get("stockfish_color"))
            bot_is_white = not sf_is_white
            r_for_table = r if bot_is_white else -r

        s = stats[key]
        s["N"] += 1
        plies = int(g.get("plies", 0))
        s["plies_sum"] += plies
        if r_for_table > 0:
            s["W"] += 1
        elif r_for_table < 0:
            s["L"] += 1
        else:
            s["D"] += 1

        # bot vs SF: count bot wins by checking if SF lost (WHITE-POV logic)
        if g.get("vs_stockfish", False):
            sf_overall["N"] += 1
            sf_overall["plies_sum"] += plies

            sf_is_white = bool(g.get("stockfish_color"))
            if r == 0:
                sf_overall["D"] += 1
            else:
                bot_won = (r < 0) if sf_is_white else (r > 0)
                if bot_won:
                    sf_overall["W"] += 1
                else:
                    sf_overall["L"] += 1

    bucket_order = {"white": 0, "black": 1, "none": 2}
    rows = sorted(stats.items(),
                  key=lambda kv: (kv[0][0], bucket_order.get(kv[0][1], 99)))

    return stats, rows, sf_overall


def print_recent_summary(recent, window=500, result_is_bot_pov=True):
    """
    Pretty-print the scenario table and the bot-vs-Stockfish W/L/D line,
    plus a wins-by-scenario breakdown (SF games only).
    """
    recent = recent[-window:]
    stats, rows, sf_overall = summarize_recent_games(
        recent, result_is_bot_pov=result_is_bot_pov
    )

    # scenario table (unchanged formatting)
    print(
        f"{'scenario':<20} {'sf':<6} {'N':>4} "
        f"{'W':>4} {'L':>4} {'D':>4}   {'avg_plies':>10}"
    )
    print("-" * 60)
    for (scenario, bucket), s in rows:
        avg = (s["plies_sum"] / s["N"]) if s["N"] else 0.0
        print(
            f"{scenario:<20} {bucket:<6} {s['N']:>4} "
            f"{s['W']:>4} {s['L']:>4} {s['D']:>4}  "
            f"{avg:>10.1f}"
        )
    print("-" * 60)

    # bot vs Stockfish summary (90-char lines)
    def pct(n, d): return (n / d) if d else 0.0
    total = sf_overall["N"]
    w, l, d = sf_overall["W"], sf_overall["L"], sf_overall["D"]
    win = pct(w, total)
    avg = (sf_overall["plies_sum"] / total) if total else 0.0
    print(
        f"Total SF games: {total:>4}  W/L/D={w}/{l}/{d}  ",
        f"win_rate={win:.1%}  avg_plies={avg:.1f}"
    )

    # compute counts: scenario -> wins (combine colors)
    wins_by_scenario = defaultdict(int)
    for g in recent:
        if not g.get("vs_stockfish", False):
            continue
        r = g.get("result", 0.0)
        if r == 0:
            continue
        sf_is_white = g.get("stockfish_color")
        bot_won = (r < 0) if sf_is_white else (r > 0)
        if bot_won:
            scenario = g.get("scenario", "unknown")
            wins_by_scenario[scenario] += 1

    if wins_by_scenario:
        print("\nWins by scenario (SF games only):")
        # sort by descending wins, then alphabetically
        for s, c in sorted(wins_by_scenario.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {s:<30} {c}")
    print("~" * 60)