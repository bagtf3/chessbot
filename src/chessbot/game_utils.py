import io
import os
import pickle
import random
import re
from collections import deque, namedtuple

import chess
import chess.pgn
import numpy as np

from pyfastchess import Board as fastboard


with open(os.getenv("PRE_OPENED_UCI_PATHS", ""), "rb") as f:
    PATHS = pickle.load(f)

with open(os.getenv("PRE_OPENED_UCI_PATHS_MINI", ""), "rb") as f:
    MINI_PATHS = pickle.load(f)

with open(os.getenv("UHO_PGN_PATH", ""), "r", encoding="utf-8", errors="replace") as f:
    PGN_TEXT = f.read()


# GameSpec carries everything a worker needs to start a game.
# fen   : starting position FEN
# moves : UCI moves to push after constructing the board (preserves move history)
# meta  : scenario metadata dict
# cfg   : Config instance to use for this game
GameSpec = namedtuple("GameSpec", ["fen", "moves", "meta", "cfg"])

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# dedup window for generated openings; the source corpora are finite so
# this saturates, and an unbounded set just wastes retries on duplicates
USED_FENS_CAP = 50000
DEDUP_TYPES = frozenset({
    "UHO", "random_init", "random_middle_game", "random_endgame", "pre_opened"
})


def reconcile_game_boards(start_fen, history_uci, moves_played):
    """
    Build both board representations for a game log, preserving full move
    history (lc0 history planes, repetition state) when it can be recovered.

    moves_played only covers the moves scored in this segment (e.g. the
    model's own moves this game); history_uci is the full move list from
    true STARTPOS. Scenarios seeded partway through an opening (UHO,
    pre_opened) make these diverge by a prefix of unscored moves. Replay
    that prefix from STARTPOS and only trust it if it lands exactly on
    start_fen -- otherwise fall back to start_fen alone: correct position,
    just without history.

    Returns (board_ch, b_fast). Never raises -- but a prefix that fails to
    reconcile is a real data anomaly (history_uci and start_fen should
    always agree for a well-formed log), not a normal case, so that path
    prints a warning instead of silently degrading. The "no history to
    replay" case (prefix_len <= 0) is normal and stays silent.
    """
    board_ch = chess.Board(start_fen)
    b_fast = fastboard(start_fen)

    if not history_uci:
        return board_ch, b_fast

    prefix_len = len(history_uci) - len(moves_played or [])
    if prefix_len <= 0:
        return board_ch, b_fast

    prefix = history_uci[:prefix_len]

    def warn(reason):
        print(f"[game_utils] reconcile_game_boards: {reason}, "
              f"falling back to start_fen-only for {start_fen!r}")
        return board_ch, b_fast

    warm_fast = fastboard()
    for uci in prefix:
        if uci not in warm_fast.legal_moves():
            return warn(f"illegal move {uci!r} in history_uci prefix")
        warm_fast.push_uci(uci)

    if warm_fast.fen() != start_fen:
        return warn(
            f"history_uci prefix landed on {warm_fast.fen()!r}, "
            f"expected start_fen"
        )

    warm_ch = chess.Board()
    try:
        for uci in prefix:
            warm_ch.push_uci(uci)
    except ValueError as e:
        return warn(f"python-chess rejected prefix move: {e}")

    return warm_ch, warm_fast


def random_backrow_fen():
    pieces = ["K", "Q", "R", "R", "B", "B", "N", "N"]
    random.shuffle(pieces)
    white_back = "".join(pieces)
    return f"{white_back.lower()}/pppppppp/8/8/8/8/PPPPPPPP/{white_back} w - - 0 1"


def shuffled_backrow_fen(white_pool, black_pool, fix_king=False):
    w = [p for p in white_pool if p != "K"]
    b = [p for p in black_pool if p != "K"]
    random.shuffle(w)
    random.shuffle(b)
    if fix_king:
        w.insert(4, "K")
        b.insert(4, "K")
    else:
        w.append("K"); random.shuffle(w)
        b.append("K"); random.shuffle(b)
    return f"{to_fen_rank(b).lower()}/pppppppp/8/8/8/8/PPPPPPPP/{to_fen_rank(w)} w - - 0 1"


def to_fen_rank(pieces):
    result, empty = "", 0
    for p in pieces:
        if p == "1":
            empty += 1
        else:
            if empty:
                result += str(empty)
                empty = 0
            result += p
    if empty:
        result += str(empty)
    return result


def make_rook_for_piece_and_pawn_fen():
    # Standard backrow: R N B Q K B N R (indices 0-7)
    # Pick one minor piece square, replace with extra rook, remove pawn in front
    sq = random.choice([1, 2, 5, 6])  # b1=N, c1=B, f1=B, g1=N
    w_back = ["R", "N", "B", "Q", "K", "B", "N", "R"]
    w_back[sq] = "R"
    w_pawns = list("PPPPPPPP")
    w_pawns[sq] = "1"
    return (
        f"rnbqkbnr/pppppppp/8/8/8/8/{''.join(w_pawns)}/{''.join(w_back)} w - - 0 1"
    )


def random_piece_training_fen():
    return random.choice([
        "1bkq1qnb/pppppppp/8/8/8/8/PPPPPPPP/RKRQN1BN w - - 0 1",
        "1bn1qrbk/pppppppp/8/8/8/8/P1PPPPPP/RRNNQK2 w - - 0 1",
        "1bnbqkn1/pppppppp/8/8/8/8/PPPPPPPP/KR1QB1R1 w - - 0 1",
        "1k2qqbb/ppppp1pp/8/8/8/8/PPPPPPPP/R1K1NBQR w - - 0 1",
        "1knq1rbq/pppppppp/8/8/8/8/PPPPPPPP/1QR1NQKB w - - 0 1",
        "1nnqrk1r/pppppppp/8/8/8/8/PPPPPPPP/QR1KNN1R w - - 0 1",
        "1nrkbrqn/pppppppp/8/8/8/8/PPPPPPPP/1KR2RQQ w - - 0 1",
        "1qk1bbnq/pppppp1p/8/8/8/8/PPPPPPPP/BQNNK1RB w - - 0 1",
        "1qq1rnkn/pppppppp/8/8/8/8/PPPPPPPP/R1KBRBQB w - - 0 1",
        "1qqn1brk/p1pppppp/8/8/8/8/PPPPPPPP/2KBQQBB w - - 0 1",
        "1rk2qqr/pppppppp/8/8/8/8/PPPPPPPP/RRRBNKBN w - - 0 1",
        "1rqr1nkq/pppppppp/8/8/8/8/PPPPPP1P/NNNNKQRR w - - 0 1",
        "2qqbqk1/pppppppp/8/8/8/8/PPPPPPPP/RRQ1BBRK w - - 0 1",
        "3qqbkn/pppppppp/8/8/8/8/P1PPPPPP/BRN1RQK1 w - - 0 1",
        "b1bqn1kn/ppppp1pp/8/8/8/8/PPPPPPPP/QRB2RK1 w - - 0 1",
        "b1kr1rnq/pppppppp/8/8/8/8/PPPPPPPP/KB1RQN1R w - - 0 1",
        "b1qq1nnk/pppppppp/8/8/8/8/PPPPPPPP/QBQNB2K w - - 0 1",
        "bbk1bbrr/ppp1pp1p/8/8/8/8/PPPPPPPP/NN1RBBKR w - - 0 1",
        "bbqq1bkb/pppppppp/8/8/8/8/P1PPPPPP/1BNQQBKR w - - 0 1",
        "bbrkb1rn/pppppp1p/8/8/8/8/PPPPPPPP/RBNKR1NN w - - 0 1",
        "bk1nqqb1/pppppppp/8/8/8/8/PPPPPPPP/BRRNQNK1 w - - 0 1",
        "bk1rnrqb/pppppppp/8/8/8/8/PPPPPP1P/RNKRNQ1R w - - 0 1",
        "bkbbnnq1/pppppppp/8/8/8/8/PPPPPPPP/B1QBBBNK w - - 0 1",
        "bknrbqnn/pppppppp/8/8/8/8/PPPPPP1P/BNRKBBQB w - - 0 1",
        "bnbk1bqq/pppppppp/8/8/8/8/PPPPPP1P/KRQQ1NBN w - - 0 1",
        "bnkq1nnq/pppppp1p/8/8/8/8/PPPPPPPP/1QKBQNR1 w - - 0 1",
        "bnkqq1rn/pppppppp/8/8/8/8/PPPPPPPP/QBRK1NQN w - - 0 1",
        "bnrqbkb1/pppppppp/8/8/8/8/P1PPPPPP/BB1QBNKR w - - 0 1",
        "bq1qbbnk/pppppppp/8/8/8/8/P1PPPPPP/RQBN1NQK w - - 0 1",
        "bqb1qkb1/pppppppp/8/8/8/8/PPPPPPPP/1QBKQNB1 w - - 0 1",
        "br1k2rq/pppppppp/8/8/8/8/PPPPPPPP/1RQKR2B w - - 0 1",
        "kn1qqrb1/pppppppp/8/8/8/8/PPPPPPPP/KQNR1QN1 w - - 0 1",
        "kqr1b1nr/pppppppp/8/8/8/8/PPPPPPPP/RNRRRNK1 w - - 0 1",
        "krnbq1rq/pp1ppppp/8/8/8/8/PPPPPPPP/NRRK1QNQ w - - 0 1",
        "krqb1b1n/pppppppp/8/8/8/8/P1PPPPPP/2BN1QQK w - - 0 1",
        "n1q1q1bk/pppppppp/8/8/8/8/PPPPPP1P/BRKBBBBN w - - 0 1",
        "nnqk1bq1/pppppp1p/8/8/8/8/PPPPPPPP/NQRNKB1N w - - 0 1",
        "nnrbbq1k/pppppppp/8/8/8/8/PPPPPPPP/BNKQQ2N w - - 0 1",
        "nqb1bqnk/p1pppppp/8/8/8/8/PPPPPPPP/BNKNRNQB w - - 0 1",
        "nqnbknnr/pppppppp/8/8/8/8/PPPPPPPP/BKR1NQRB w - - 0 1",
        "nrbbn1kb/pppppppp/8/8/8/8/PPPPPPPP/BN1RBNBK w - - 0 1",
        "q1bnrqkb/pppppppp/8/8/8/8/P1PPPPPP/QRRNQ1KN w - - 0 1",
        "q1kqb1r1/pppppppp/8/8/8/8/PPPPPPPP/QK1Q1RB1 w - - 0 1",
        "qb1bkqbb/pppppppp/8/8/8/8/PPPPPP1P/BQKNB1QR w - - 0 1",
        "qbbrk1n1/pppppppp/8/8/8/8/PPPPPPPP/QB2RBKN w - - 0 1",
        "qn1nb1nk/pppppppp/8/8/8/8/PPPPPPPP/1RBBBKNB w - - 0 1",
        "qr1q1rkn/p1pppppp/8/8/8/8/PPPPPPPP/NRKBNBQN w - - 0 1",
        "rbknbrr1/pppppppp/8/8/8/8/P1PPPPPP/RR1RKNNR w - - 0 1",
        "rnkb1rr1/pppppppp/8/8/8/8/PPPPPPPP/R1BNRR1K w - - 0 1",
        "rnn2rqk/p1ppp1pp/8/8/8/8/PPPPPPPP/1NKRNNRR w - - 0 1",
    ])


def get_pre_opened_game(index=None, mini=False):
    """Returns (start_fen, moves_list)."""
    path_list = MINI_PATHS if mini else PATHS
    if index is None:
        moves_to_play = random.choice(path_list)
    else:
        try:
            moves_to_play = path_list[index]
        except Exception:
            print(
                f"No premove path found for {index}!",
                f"please choose 0 - {len(path_list)-1}.",
                "Selecting random premove path"
            )
            moves_to_play = random.choice(path_list)
    return STARTPOS_FEN, list(moves_to_play)


def random_init(plies=5):
    """Returns (start_fen, moves_played)."""
    moves_played = []
    b = fastboard()
    p = 0
    counter = 0
    limit = int(3 * plies)
    while p < plies:
        counter += 1
        if counter > limit:
            return random_init(plies=plies)
        moves = b.legal_moves()
        while not len(moves):
            b.unmake()
            moves_played.pop()
            moves = b.legal_moves()
            p -= 1
            if p == 0:
                return random_init(plies=plies)
        mv = random.choice(moves)
        b.push_uci(mv)
        moves_played.append(mv)
        p += 1
    return STARTPOS_FEN, moves_played


def random_board_setup(pieces, wk=None, bk=None, queens=True):
    """Returns a chess.Board (python-chess) for the generated position."""
    if pieces < 6:
        raise ValueError("pieces must be >= 6")
    if pieces > 32:
        raise ValueError("pieces must be <= 32")

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
    pawn_ok_squares = set(range(8, 56))

    def draw_piece_type_and_color(rem_white, rem_black):
        bag = []
        for pt in pool_piece_types:
            for _ in range(rem_white[pt]):
                bag.append((pt, chess.WHITE))
            for _ in range(rem_black[pt]):
                bag.append((pt, chess.BLACK))
        if not bag:
            return None
        return random.choice(bag)

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

    for _ in range(1000):
        board = chess.Board(None)
        wk = random.choice(chess.SQUARES) if wk is None else wk
        while True:
            bk = random.choice(chess.SQUARES) if bk is None else bk
            if bk != wk:
                break
            bk = None
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        rem_white = dict(cap)
        rem_black = dict(cap)
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
                if color == chess.WHITE:
                    if rem_white[pt] > 0:
                        rem_white[pt] -= 1
                else:
                    if rem_black[pt] > 0:
                        rem_black[pt] -= 1
                continue
            if color == chess.WHITE:
                rem_white[pt] -= 1
            else:
                rem_black[pt] -= 1
            need -= 1
        if not ok_build or need != 0:
            continue
        board.turn = random.choice(colors)
        if not board.is_valid():
            board.turn = not board.turn
            if not board.is_valid():
                continue
        if board.is_game_over():
            continue
        if board.legal_moves.count() == 0:
            continue
        return board

    print("Unable to find valid board. Loosening requirements...")
    return random_board_setup(max(6, pieces - 1), wk=None, bk=None, queens=False)


def make_piece_odds_board(allow_remove_heavies=True):
    """allow_remove_heavies=False keeps queens and rooks on: a self-play game
    at those odds is decided before it starts and teaches little."""
    b = chess.Board()
    meta = {"scenario": "piece_odds", "removed": {"white": [], "black": []}}
    remove_from = random.choice([chess.WHITE, chess.BLACK])
    removable = [chess.BISHOP, chess.KNIGHT, chess.PAWN]
    if allow_remove_heavies:
        removable = [chess.QUEEN, chess.ROOK] + removable
    to_remove = random.choice(removable)
    need_to_remove = np.random.randint(1, 4) if to_remove == chess.PAWN else 1
    color_str = "white" if remove_from == chess.WHITE else "black"
    for _ in range(need_to_remove):
        pool = [sq for sq, pc in b.piece_map().items()
                if pc.color == remove_from and pc.piece_type == to_remove]
        if not pool:
            break
        sq = random.choice(pool)
        pc = b.remove_piece_at(sq)
        meta["removed"][color_str].append(pc.symbol())
    return b.fen(), meta


def remove_pawn(fen, color, file):
    col = file
    parts = fen.split()
    ranks = parts[0].split('/')
    rank_idx, pawn_char = (1, 'p') if color == 'black' else (6, 'P')
    expanded = []
    for c in ranks[rank_idx]:
        if c.isdigit():
            expanded.extend([None] * int(c))
        else:
            expanded.append(c)
    if expanded[col] != pawn_char:
        print(f"warning: no {color} pawn on file {file} in {ranks[rank_idx]!r}")
        return fen
    expanded[col] = None
    new_rank, empty = '', 0
    for sq in expanded:
        if sq is None:
            empty += 1
        else:
            if empty:
                new_rank += str(empty)
                empty = 0
            new_rank += sq
    if empty:
        new_rank += str(empty)
    ranks[rank_idx] = new_rank
    parts[0] = '/'.join(ranks)
    return ' '.join(parts)


def make_b_vs_k_fen():
    fen = shuffled_backrow_fen(
        ["K", "Q", "R", "B", "B", "B", "B", "1"],
        ["K", "Q", "R", "N", "N", "N", "N", "N"],
        fix_king=True,
    )
    back = []
    for c in fen.split('/')[0]:
        if c.isdigit():
            back.extend([None] * int(c))
        else:
            back.append(c)
    eligible = [i for i, p in enumerate(back) if p in ('k', 'n')]
    return remove_pawn(fen, 'black', random.choice(eligible))


def make_piece_training_board():
    pick = random.choice([
        "rooks_vs_standard", "b_vs_k", "extra_queen",
        "random_backrow", "random_backrow_with_replacement", "rook_for_piece_and_pawn",
    ])
    if pick == "rooks_vs_standard":
        fen = "rnbqkbnr/{}/8/8/8/8/PPPPPPPP/RRRRKRRR w - - 0 1".format(random.choice([
            "1p1ppppp", "1pp1pppp", "1ppp1ppp", "1pppp1pp", "1ppppp1p", "1pppppp1",
            "2pppppp",
            "p1pp1ppp", "p1ppp1pp", "p1pppp1p", "p1ppppp1", "p2ppppp",
            "pp1p1ppp", "pp1pp1pp", "pp1ppp1p", "pp1pppp1", "pp2pppp",
            "ppp1p1pp", "ppp1pp1p", "ppp1ppp1", "ppp2ppp",
            "pppp1p1p", "pppp1pp1", "pppp2pp",
            "ppppp1p1", "ppppp2p",
            "pppppp2",
        ]))
    elif pick == "b_vs_k":
        fen = make_b_vs_k_fen()
    elif pick == "extra_queen":
        fen = random.choice([
            "qnb1kbnq/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
            "qnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
            "1nbqkbnq/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
        ])
    elif pick == "random_backrow":
        fen = random_backrow_fen()
    elif pick == "random_backrow_with_replacement":
        fen = random_piece_training_fen()
    else:
        fen = random.choice([
            "rnbqkbnr/pppppppp/8/8/8/8/1PPPPP1P/RNBQKBRR w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/P1PPPP1P/RNBQKBRR w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/P1PPPPPP/RRBQKBNR w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PP1PPP1P/RNBQKBRR w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PP1PPPPP/RNRQKBNR w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPP1PP1P/RNBQKBRR w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPP1PP/RNBQKRNR w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPP2/RNBQKBRR w - - 0 1",
        ])
    board = chess.Board(fen)
    if np.random.uniform() < 0.5:
        board = board.mirror()
    board.turn = chess.WHITE
    return board.fen(), {"scenario": f"pt_{pick}"}


class UhoPgnSampler:
    def __init__(self, pgn_text):
        self.pgn_text = pgn_text
        self.uho_event_start_re = re.compile(r'(?m)^\[Event "')
        self.starts = [
            m.start()
            for m in self.uho_event_start_re.finditer(pgn_text)
        ]
        if not self.starts:
            raise ValueError("No games found (no [Event at line start).")

    def sample(self):
        """Returns (start_fen, moves_list)."""
        k = random.randrange(len(self.starts))
        a = self.starts[k]
        b = self.starts[k + 1] if k + 1 < len(self.starts) else len(self.pgn_text)
        chunk = self.pgn_text[a:b]
        blank = chunk.find("\n\n")
        moves_blob = chunk if blank == -1 else chunk[blank + 2:]
        game = chess.pgn.read_game(io.StringIO(moves_blob))
        if game is None:
            return STARTPOS_FEN, []
        moves = [mv.uci() for mv in game.mainline_moves()]
        return STARTPOS_FEN, moves


def short_fen(fen):
    return " ".join(fen.split(" ")[:4])


def resolve_cfg(cfg):
    """Return a copy of cfg with sampleable params replaced by sampled scalars."""
    resolved = cfg.copy()
    resolved.update({param: random.choice(options) for param, options in cfg.sampleable.items()})
    resolved.sampleable = {}
    return resolved


class GameGenerator:
    def __init__(self, cfg):
        self.config = cfg
        # bounded: the source corpora (UHO pgn, pre_opened pkl path lists)
        # are finite, so over a long run this set saturates and the 20-attempt
        # retry below burns 20 board rebuilds per game only to return a
        # duplicate anyway. FIFO eviction keeps dedup useful over a window.
        self.used_fens = set()
        self.used_fens_order = deque()
        self.uho_sampler = UhoPgnSampler(PGN_TEXT)

        self.games_queued = 0

    @property
    def game_types(self):
        """Read live, not cached at construction: the runner re-points
        .config after every retrain so a game_probs edit takes effect
        mid-run without a restart."""
        return list(self.config.game_probs.keys())

    def generate(self, game_type):
        """Returns (fen, moves, meta). fen is the starting position; moves are
        pushed on top of it to reach the game start position."""
        if game_type == "pre_opened":
            fen, moves = get_pre_opened_game()
            meta = {"scenario": "pre_opened"}

        elif game_type == "pre_opened_mini":
            fen, moves = get_pre_opened_game(mini=True)
            meta = {"scenario": "pre_opened_mini"}

        elif game_type == "UHO":
            fen, moves = self.uho_sampler.sample()
            meta = {"scenario": "UHO"}

        elif game_type == "random_init":
            plies = 2 * np.random.randint(1, 5)
            fen, moves = random_init(plies)
            meta = {"scenario": "random_init", "start_plies": plies}

        elif game_type == "random_middle_game":
            plies = np.random.randint(20, 31)
            fen, moves = random_init(plies)
            meta = {"scenario": "random_middle_game", "start_plies": plies}

        elif game_type == "random_endgame":
            pieces = np.random.randint(8, 14)
            wk = np.random.randint(0, 33)
            bk = np.random.randint(33, 64)
            board = random_board_setup(pieces, wk, bk, queens=False)
            fen, moves = board.fen(), []
            meta = {"scenario": "random_endgame", "pieces": pieces}

        elif game_type == "piece_odds":
            fen, meta = make_piece_odds_board(allow_remove_heavies=False)
            moves = []

        elif game_type == "piece_training":
            fen, meta = make_piece_training_board()
            meta["scenario_subtype"] = meta.pop("scenario")
            meta["scenario"] = "piece_training"
            moves = []
        
        elif game_type == "startpos":
            fen, moves = STARTPOS_FEN, []
            meta = {"scenario": "startpos"}

        else:
            raise ValueError(f"unhandled game_type {game_type}")

        # verify the final position has legal moves
        b = fastboard(fen)
        for mv in moves:
            b.push_uci(mv)
        if not b.legal_moves():
            return self.generate(game_type)

        return fen, moves, meta

    def new_board(self, game_type=None):
        """Returns (board, meta) — compatibility API for misc scripts."""
        if game_type is None:
            types, probs = zip(*self.config.game_probs.items())
            game_type = random.choices(types, weights=probs, k=1)[0]
        fen, moves, meta = self.generate(game_type)
        b = fastboard(fen)
        for mv in moves:
            b.push_uci(mv)
        return b, meta

    def next_game(self, game_type=None, cfg=None):
        if cfg is None:
            cfg = self.config
        if game_type is None:
            types, probs = zip(*cfg.game_probs.items())
            game_type = random.choices(types, weights=probs, k=1)[0]
        elif game_type not in self.game_types:
            raise ValueError(
                f"Unknown game type {game_type}. Pick one of {self.game_types}"
            )

        fen, moves, meta = self.generate(game_type)

        if game_type in DEDUP_TYPES:
            b = fastboard(fen)
            for mv in moves:
                b.push_uci(mv)
            key = short_fen(b.fen())
            attempts = 0
            while key in self.used_fens and attempts < 20:
                fen, moves, meta = self.generate(game_type)
                b = fastboard(fen)
                for mv in moves:
                    b.push_uci(mv)
                key = short_fen(b.fen())
                attempts += 1
            self.remember_fen(key)

        # only paired validation plays SF, but ChessGame reads both keys
        meta["vs_stockfish"] = False
        meta["stockfish_is_white"] = False
        self.games_queued += 1
        return GameSpec(fen=fen, moves=moves, meta=meta, cfg=resolve_cfg(cfg))

    def remember_fen(self, key):
        """FIFO-bounded dedup memory."""
        if key in self.used_fens:
            return
        self.used_fens.add(key)
        self.used_fens_order.append(key)
        while len(self.used_fens_order) > USED_FENS_CAP:
            self.used_fens.discard(self.used_fens_order.popleft())

    def next_blunder_replays(self, pool, n_to_queue, current_epoch=None):
        """Sample n_to_queue probes from the live pool. Metering and logging
        live at the selfplay_runner callsites. current_epoch gates
        the last_seen cooldown and the max-fails age-out."""
        from chessbot.blunder_replay import (
            create_blunder_replay_config, blunder_replay_specs,
        )

        if pool is None or n_to_queue <= 0:
            return []

        replay_cfg = create_blunder_replay_config(self.config)

        specs = blunder_replay_specs(
            replay_cfg, pool, n_to_queue,
            current_epoch=current_epoch
        )

        return specs

    def validation_games(self, cfg=None, n_pairs=None):
        """n_pairs positions, each played from both sides -- 2*n_pairs specs.
        Defaults to cfg.n_games // 2 for the old round-shaped caller, where
        the whole round was the batch."""
        if cfg is None:
            cfg = self.config
        n = n_pairs if n_pairs is not None else cfg.n_games // 2

        val_probs = cfg.validation_game_probs
        types = list(val_probs.keys())
        probs = list(val_probs.values())
        total = sum(probs)
        normalized = [p / total for p in probs]

        # distribute n pairs proportionally, remainder goes to UHO
        counts = [int(np.floor(p * n)) for p in normalized]
        remainder = n - sum(counts)
        if remainder > 0:
            uho_idx = types.index("UHO") if "UHO" in types else 0
            counts[uho_idx] += remainder

        # build shuffled list of game types to attempt
        plan = []
        for t, c in zip(types, counts):
            plan.extend([t] * c)
        random.shuffle(plan)

        seen = set()
        specs = []

        def try_generate_unique(game_type, max_attempts=50):
            for _ in range(max_attempts):
                fen, moves, meta = self.generate(game_type)
                b = fastboard(fen)
                for mv in moves:
                    b.push_uci(mv)
                key = short_fen(b.fen())
                if key not in seen:
                    seen.add(key)
                    return fen, moves, key
            return None

        for game_type in plan:
            result = try_generate_unique(game_type)
            if result is None and game_type != "UHO":
                result = try_generate_unique("UHO")
            if result is None:
                print(f"[validation_games] warning: could not find unique position for {game_type}, skipping")
                continue
            fen, moves, _ = result
            cfg_black = resolve_cfg(cfg)
            cfg_black.contempt_fight_c = 0.0
            specs.append(GameSpec(
                fen=fen, moves=moves,
                meta={"vs_stockfish": True, "stockfish_is_white": True,
                      "scenario": "paired_validation"},
                cfg=cfg_black,
            ))
            specs.append(GameSpec(
                fen=fen, moves=list(moves),
                meta={"vs_stockfish": True, "stockfish_is_white": False,
                      "scenario": "paired_validation"},
                cfg=resolve_cfg(cfg),
            ))
        return specs
