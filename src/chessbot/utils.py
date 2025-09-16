import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import math, random, time, pickle
            
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()

import chess
import chess.engine
import chess.svg
from IPython.display import SVG, display, clear_output

from chessbot import SF_LOC
from chessbot import features as ft

from collections import deque
from pyfastchess import Board as fastboard


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
    # assume its
    else:


def random_init(halfmoves=5):
    board = chess.Board()
    for _ in range(halfmoves):
        moves = list(board.legal_moves)
        # try again 
        if not len(moves):
            return random_init(halfmoves=halfmoves)
        
        board = make_random_move(board)
        
        # also try again
        if board.is_game_over():
            return random_init(halfmoves=halfmoves)
        
    return board


def fb_random_init(plies=5):
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

        b.push_uci(random.choice(moves))
        p += 1
    return b


def plot_training_progress(all_evals, max_cols=4):
    """
    Plots training/eval metrics for each model output in a grid.
    Adapts rows/cols automatically, up to max_cols wide.
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
        # moving average (MA5)
        ma = pd.Series(y).rolling(15, min_periods=1).mean().values
        ax.plot(ma, lw=2, alpha=0.6, label=f"{col} (MA5)")
        ax.set_title(col)
        ax.legend()

    # Hide any unused axes
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_pred_vs_true_grid(model, preds, y_true_dict):
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
        plt.show()


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


def score_game_data(model, X, Y_batch):
    raw_preds = model.predict(X, batch_size=256, verbose=0)
    preds = {name: raw_preds[i] for i, name in enumerate(model.output_names)}

    # Plot overview grid
    plot_pred_vs_true_grid(model, preds, Y_batch)

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


## board/ position generation
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


def random_endgame_board(max_tries=1000):
    """
    Create a random board with exactly 5 pieces:
    - 2 kings (one white, one black)
    - 3 random other pieces (any type, any color, random squares)
    
    Keeps retrying until a legal board is found.
    """
    piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

    for _ in range(max_tries):
        board = chess.Board.empty()
        board.turn = random.choice([chess.WHITE, chess.BLACK])

        # Place the two kings first
        king_squares = random.sample(chess.SQUARES, 2)
        board.set_piece_at(king_squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(king_squares[1], chess.Piece(chess.KING, chess.BLACK))
        occupied = set(king_squares)

        # Place 3 random other pieces
        for _ in range(3):
            sq = random.choice([s for s in chess.SQUARES if s not in occupied])
            occupied.add(sq)
            piece_type = random.choice(piece_types)
            color = random.choice([chess.WHITE, chess.BLACK])
            board.set_piece_at(sq, chess.Piece(piece_type, color))

        # Validate
        if board.is_valid():
            return board

    raise ValueError("Could not generate a valid endgame board in given tries")


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

    eng_kwargs = {}
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


uci_path_path =  r"C:/Users/Bryan/Data/chessbot_data/uci_paths3000.pkl"  
with open(uci_path_path, "rb") as f:
    paths = pickle.load(f)
    
def get_pre_opened_game():
    b = fastboard()
    moves_to_play = random.choice(paths)
    for mtp in moves_to_play:
        b.push_uci(mtp)
    return b


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
    
#%%    


from concurrent.futures import ThreadPoolExecutor, as_completed


def score_pov_cp(pov_score, white_to_move, mate_cp):
    s = pov_score.white() if white_to_move else pov_score.black()
    return float(s.score(mate_score=mate_cp))


def evaluate_game_vs_sf(moves_uci, depth=12, mate_cp=1500):
    board = chess.Board()
    white_cpl, black_cpl = [], []
    best_moves = [0, 0]
    
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
        limit = chess.engine.Limit(depth=int(depth))

        for u in moves_uci:
            if board.is_game_over():
                break
            
            mv = chess.Move.from_uci(u)
            white_to_move = board.turn

            best = eng.analyse(board, limit=limit)
            best_cp = score_pov_cp(best["score"], white_to_move, mate_cp)
            best_uci = str(best['pv'][0])
            
            if best_uci == u:
                best_moves[white_to_move] += 1
                
            played = eng.analyse(board, limit=limit, root_moves=[mv])
            played_cp = score_pov_cp(played["score"], white_to_move, mate_cp)

            cpl = max(0.0, best_cp - played_cp)
            (white_cpl if white_to_move else black_cpl).append(cpl)
            board.push(mv)

    def summary(arr):
        if not arr:
            return {"mean_cpl": 0.0, "max_cpl": 0.0, "n": 0}
        a = np.asarray(arr, dtype=np.float32)
        return {
            "mean_cpl": a.mean().round(3), "max_cpl": a.max().round(3), "n": int(a.size)
        }

    white = summary(white_cpl)
    black = summary(black_cpl)
    overall = float(np.mean(white_cpl + black_cpl)) if (white_cpl or black_cpl) else 0.0

    return {
        "plies": len(white_cpl) + len(black_cpl),
        "white": white, "black": black, "overall_mean_cpl": np.round(overall, 3),
        "best_moves_white": best_moves[1], "best_moves_black": best_moves[0],
        "overall_best_moves": sum(best_moves)
    }


def evaluate_many_games(games_moves_uci, depth=12, workers=4, mate_cp=1500):
    def worker(moves):
        return evaluate_game_vs_sf(moves, depth=depth, mate_cp=mate_cp)

    results = []
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = {ex.submit(worker, g): i for i, g in enumerate(games_moves_uci)}
        for fut in as_completed(futs):
            results.append(fut.result())

    if not results:
        return {"games": [], "summary": {}}

    w_means = [r["white"]["mean_cpl"] for r in results]
    b_means = [r["black"]["mean_cpl"] for r in results]
    o_means = [r["overall_mean_cpl"] for r in results]

    summary = {
        "games": len(results),
        "avg_white_mean_cpl": float(np.mean(w_means)),
        "avg_black_mean_cpl": float(np.mean(b_means)),
        "avg_overall_mean_cpl": float(np.mean(o_means)),
    }
    return {"games": results, "summary": summary}



