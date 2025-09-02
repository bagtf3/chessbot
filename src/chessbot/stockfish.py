from chessbot import SF_LOC, START_POS_FEN, START_POS_KEY
import subprocess
from collections import defaultdict
import time

class SFProcess:
    def __init__(self, path=SF_LOC):
        self.path = path
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # Init handshake
        self.send("uci")
        self.read_block(wait_token="uciok")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.proc:
            try:
                self.send("quit")
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
            self.proc = None

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def readline(self):
        return self.proc.stdout.readline().strip()

    def read_block(self, wait_token=None, timeout=1.0):
        """Read all lines currently available. 
        If wait_token is set, stop once that line appears."""
        lines = []
        start = time.time()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                if wait_token is None:
                    break
                if time.time() - start > timeout:
                    break
                continue
            line = line.strip()
            lines.append(line)
            if wait_token and wait_token in line:
                break
        return lines

LGL_MV_STOPWORDS = set(['Fen:', 'Key:', 'Chec', 'info', 'Node'])

class SFBoard:
    def __init__(self, sf_proc, fen=None):
        self.sf_proc = sf_proc
        self.move_history = []
        self.legal_moves = []
        
        if fen is None:
            self.current_fen = START_POS_FEN
        
        else:
            self.current_fen = fen
            
        self.fen_history = [self.current_fen]
    
    def get_info(self, fen=None):
        info_fen = self.current_fen if fen is None else fen
        
        self.sf_proc.send(f"position fen {info_fen}\nd")
        lines = self.sf_proc.read_block(wait_token="Checkers:")
        fen, key, checkers = [
            l.split(":")[1].strip() for l in lines if l[:3] in ['Fen', 'Key', 'Che']
        ]
        
        return fen, key, checkers
        
    def get_legal_moves(self, fen=None):
        if fen is None:
            if self.legal_moves:
                return self.legal_moves
            info_fen = self.current_fen
        else:
            info_fen = fen
        
        self.sf_proc.send(f"position fen {info_fen}\ngo perft 1")
        lines = self.sf_proc.read_block(wait_token="Nodes searched")
        moves = []
        for l in lines:
            if l[:4] in LGL_MV_STOPWORDS:
                continue
            if ":" in l:
                mv = l.split(":")[0].strip()
                moves.append(mv)
        
        self.legal_moves = moves
        return moves

    def push(self, move):
        self.move_history.append(move)
        
        self.sf_proc.send(f"position {self.current_fen} moves {move}\nd")
        lines = self.sf_proc.read_block(wait_token="Checkers:")
        fen, key, checkers = [
            l.split(":")[1].strip() for l in lines if l[:3] in ['Fen', 'Key', 'Che']
        ]
        
        self.current_fen = fen
        self.fen_history.append(fen)

    def turn(self):
        side = self.current_fen.split()[1]
        return True if side == "w" else False

    def castling_rights(self):
        rights = self.current_fen.split()[2]
        if rights == "-":
            return set()
        return set(rights)

    def is_game_over(self, claim_draw=False):
        moves = self.get_legal_moves()
        if len(moves) == 0:
            return True

        # check for simple draw conditions
        if claim_draw:
            # threefold repetition
            for key, count in self.zobrist_counter.items():
                if count >= 3:
                    return True
                
            # fifty-move rule
            halfmove = int(self.current_fen.split()[4])
            if halfmove >= 100:
                return True

        return False

#%%
startpos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
double_check = "4k3/2N5/8/8/8/8/8/4R1K1 b - - 0 1"
stalemate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
insufficient = "8/8/8/8/8/4K3/8/4k3 w - - 0 1"
white_mated = '5rk1/2pn1ppp/8/p1QNp3/4P3/3PKBP1/1r6/6q1 w - - 6 33'

with SFProcess() as sf_proc:
    board = SFBoard(sf_proc, double_check)


#%%

import chess
from chessbot.utils import random_init

fens = [random_init(rep % 8).fen() for rep in range(5000)]

def test_pychess_fen_read(fens):
    start = time.time()
    for rep in range(4):
        out = []
        for f in fens:
            out.append(chess.Board(f))
    stop = time.time()
    return stop-start, out


def test_sf_fen_read(fens):
    with SFProcess() as sf_proc:
        start = time.time()
        for rep in range(4):    
            out = []
            for f in fens:
                out.append(SFBoard(sf_proc, fen=f))
    stop = time.time()
    return stop-start, out

py_time, py_boards = test_pychess_fen_read(fens)
sf_time, sf_boards = test_sf_fen_read(fens)

print(f"pychess time: {py_time: <.3}, sf_time: {sf_time: <.3}")
print()
def test_pychess_move_gen(fens):
    start = time.time()
    n = 0
    for f in fens:
        b = chess.Board(f)
        moves = list(b.legal_moves)
        n += len(moves)
    stop = time.time()
    return stop-start, n


def test_sf_move_gen(fens):
    with SFProcess() as sf_proc:
        start = time.time()
        n = 0
        for f in fens:
            b = SFBoard(sf_proc, fen=f)
            moves = b.get_legal_moves()
            n += len(moves)
    stop = time.time()
    return stop-start, n

py_time, py_moves = test_pychess_move_gen(fens)
sf_time, sf_moves = test_sf_move_gen(fens)

print(f"pychess time: {py_time: <.3}, sf_time: {sf_time: <.3}")
print(f"pychess moves: {py_moves}, sf_moves: {sf_moves}")
print()
import random
def test_pychess_move_push(fens):
    start = time.time()
    n = 0
    for f in fens:
        b = chess.Board(f)
        moves = list(b.legal_moves)
        b.push(random.choice(moves))
    stop = time.time()
    return stop-start


def test_sf_move_push(fens):
    with SFProcess() as sf_proc:
        start = time.time()
        for f in fens:
            b = SFBoard(sf_proc, fen=f)
            moves = b.get_legal_moves()
            b.push(random.choice(moves))
    stop = time.time()
    return stop-start

py_time = test_pychess_move_push(fens)
sf_time = test_sf_move_push(fens)

print(f"pychess time: {py_time: <.3}, sf_time: {sf_time: <.3}")
print()
#%%            


def test_perft(sf_proc, fens):
    start = time.time()
    total_moves = 0
    for fen in fens:
        sf_proc.send(f"position fen {fen}")
        sf_proc.send("go perft 1")
        lines = sf_proc.read_block(wait_token="Nodes searched")
        for l in lines:
            if ":" in l and not l.startswith("Nodes"):
                total_moves += 1
    stop = time.time()
    return stop - start, total_moves


def test_multipv(sf_proc, fens):
    # set multipv big enough to cover max legal moves (~218)
    sf_proc.send("setoption name MultiPV value 256\n")

    start = time.time()
    total_moves = 0
    for fen in fens:
        sf_proc.send(f"position fen {fen}\ngo depth 1")
        lines = sf_proc.read_block(wait_token="bestmove")
        for l in lines:
            if "multipv" in l and "pv" in l:
                total_moves += 1
    stop = time.time()
    return stop - start, total_moves


TEST_FENS = fens
with SFProcess(path=SF_LOC) as sf:
    t1, n1 = test_perft(sf, TEST_FENS)
    print(f"perft 1: {t1:.3f}s, moves counted: {n1}")

    t2, n2 = test_multipv(sf, TEST_FENS)
    print(f"depth 1 multipv: {t2:.3f}s, moves counted: {n2}")


#%%
import chess
import chess.engine

STOCKFISH_PATH = SF_LOC
N_GAMES = 150                   # number of self-play games
MAX_MOVES = 80                 # max moves per game
DEPTH = 8                      # SF depth (tune for speed/quality)
OUTFILE = "C:/Users/Bryan/Data/chessbot_data/train.txt"

def board_to_words(board):
    """Tokens to describe the board like fastText wants."""
    for s, p in board.piece_map().items():
        yield f"{chess.SQUARE_NAMES[s]}{p.symbol()}"
    if board.castling_rights & chess.BB_H1: yield "H1-C"
    if board.castling_rights & chess.BB_H8: yield "H8-C"
    if board.castling_rights & chess.BB_A1: yield "A1-C"
    if board.castling_rights & chess.BB_A8: yield "A8-C"
    yield "WhiteTurn" if board.turn else "BlackTurn"

def mirror_move(move):
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      move.promotion)

def prepare_example(board, move):
    """Turn board+move into fastText line: tokens ... __label__uci"""
    if board.turn == chess.WHITE:
        string = " ".join(board_to_words(board))
        uci_move = move.uci()
    else:
        string = " ".join(board_to_words(board.mirror()))
        uci_move = mirror_move(move).uci()
    return f"{string} __label__{uci_move}"

def generate_selfplay(n_games=10, depth=8, max_moves=80, outfile="train.txt"):
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine, open(outfile, "w") as f:
        for g in range(n_games):
            board = chess.Board()
            moves = 0
            while not board.is_game_over() and moves < max_moves:
                result = engine.play(board, chess.engine.Limit(depth=depth))
                if result.move is None:
                    break
                # write training example
                line = prepare_example(board, result.move)
                f.write(line + "\n")
                # play move
                board.push(result.move)
                moves += 1
            print(f"Game {g+1}/{n_games} finished in {moves} moves.")
    print(f"Saved training data to {outfile}")


generate_selfplay(N_GAMES, DEPTH, MAX_MOVES, OUTFILE)



import fasttext

model = fasttext.train_supervised(OUTFILE, epoch=8, lr=0.1, wordNgrams=2, dim=64)
model_file = "C:/Users/Bryan/Data/chessbot_data/models/chess_model.bin"
model.save_model(model_file)


import chess
import fasttext

# Load trained model
model = fasttext.load_model(model_file)

def get_priors(board, model, k=-1):
    # Encode board as text tokens
    if board.turn == chess.WHITE:
        string = " ".join(board_to_words(board))
    else:
        string = " ".join(board_to_words(board.mirror()))

    # Predict (k=-1 = return all labels)
    labels, probs = model.predict(string, k=k)

    # Convert to dict {uci_move: prob}
    priors = {}
    for label, prob in zip(labels, probs):
        move = label.replace("__label__", "")
        uci = chess.Move.from_uci(move)
        if uci in list(board.legal_moves):
            priors[uci.uci()] = prob

    # Normalize to sum = 1
    total = sum([v for k, v in priors.items()])
    if total > 0:
        priors = {m: p/total for m, p in priors.items()}

    return priors


# --- Try it on starting position ---
board = random_init(7)
board = chess.Board()
priors = get_priors(board, model, k=-1)

print("Top priors from fastText:")
for mv, prob in sorted(priors.items(), key=lambda x: -x[1])[:5]:
    print(mv, f"{prob:.3f}")




