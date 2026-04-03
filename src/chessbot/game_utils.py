import io
import os
import pickle
import random
import re
from collections import namedtuple

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

DEDUP_TYPES = frozenset({
    "UHO", "random_init", "random_middle_game", "random_endgame", "pre_opened"
})


def random_backrow_fen():
    pieces = ["K", "Q", "R", "R", "B", "B", "N", "N"]
    random.shuffle(pieces)
    white_back = "".join(pieces)
    random.shuffle(pieces)
    black_back = "".join(pieces).lower()
    return f"{black_back}/pppppppp/8/8/8/8/PPPPPPPP/{white_back} w - - 0 1"


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


def make_piece_odds_board():
    b = chess.Board()
    meta = {"scenario": "piece_odds", "removed": {"white": [], "black": []}}
    remove_from = random.choice([chess.WHITE, chess.BLACK])
    removable = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
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


def make_piece_training_board():
    fens = {
        "rooks": "rrrrkrrr/pppppppp/8/8/8/8/PPPPPPPP/RRRRKRRR w - - 0 1",
        "bishops": "bbbbkbbb/pppppppp/8/8/8/8/PPPPPPPP/BBBBKBBB w - - 0 1",
        "knights": "nnnnknnn/pppppppp/8/8/8/8/PPPPPPPP/NNNNKNNN w - - 0 1",
        "b_vs_k": "bbbbkbbb/pppppppp/8/8/8/8/PPPPPPPP/NNNNKNNN w - - 0 1",
        "extra_queen": "qnb1kbnq/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
        "random_backrow": random_backrow_fen()
    }
    pick = random.choice(list(fens.keys()))
    board = chess.Board(fens[pick])
    if np.random.uniform() < 0.5:
        board = board.mirror()
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
        self.game_types = list(cfg.game_probs.keys())
        self.sf_count = 0
        self.used_fens = set()
        self.uho_sampler = UhoPgnSampler(PGN_TEXT)

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
            fen, meta = make_piece_odds_board()
            moves = []

        elif game_type == "piece_training":
            fen, meta = make_piece_training_board()
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

    def assign_sf(self, meta):
        cfg = self.config
        meta["vs_stockfish"] = False
        meta["stockfish_is_white"] = False
        if cfg.play_vs_sf_prob <= 0.0:
            return
        if meta.get("scenario") in cfg.sf_exclude:
            return
        if np.random.uniform() > cfg.play_vs_sf_prob:
            return
        self.sf_count += 1
        meta["vs_stockfish"] = True
        meta["stockfish_is_white"] = bool(self.sf_count % 2)

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
            self.used_fens.add(key)

        self.assign_sf(meta)
        return GameSpec(fen=fen, moves=moves, meta=meta, cfg=resolve_cfg(cfg))

    def validation_games(self, cfg=None):
        if cfg is None:
            cfg = self.config
        n = cfg.n_games * max(1, cfg.n_workers) // 2  # number of pairs needed, across all workers

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
            specs.append(GameSpec(
                fen=fen, moves=moves,
                meta={"vs_stockfish": True, "stockfish_is_white": True,
                      "scenario": "paired_validation"},
                cfg=cfg,
            ))
            specs.append(GameSpec(
                fen=fen, moves=list(moves),
                meta={"vs_stockfish": True, "stockfish_is_white": False,
                      "scenario": "paired_validation"},
                cfg=cfg,
            ))
        return specs
