import os, json, random
from tqdm import tqdm
import chess
import chess.engine
import numpy as np

from chessbot import SF_LOC
from chessbot.utils import random_init, make_random_move, get_all_openings

PIECE_TO_ID = {
    None: 0,                # empty square
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

# black pieces offset by +6
def square_to_id(piece):
    if piece is None:
        return 0
    base = PIECE_TO_ID[piece.piece_type]
    return base if piece.color == chess.WHITE else base + 6


def board_to_tokens(board: chess.Board):
    tokens = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        tokens.append(square_to_id(piece))
    # extra tokens
    tokens.append(int(board.turn))
    tokens.append(int(board.has_kingside_castling_rights(chess.WHITE)))
    tokens.append(int(board.has_queenside_castling_rights(chess.WHITE)))
    tokens.append(int(board.has_kingside_castling_rights(chess.BLACK)))
    tokens.append(int(board.has_queenside_castling_rights(chess.BLACK)))
    ep = board.ep_square if board.ep_square else 0
    tokens.append(ep % 64)  # encode en passant square (0 if none)
    return np.array(tokens, dtype=np.int32)


def analyze(engine, board, **kwargs):
    info = engine.analyse(
        board, multipv=len(list(board.legal_moves)),
        limit=chess.engine.Limit(**kwargs),
        info=chess.engine.Info.ALL
    )
    
    return info
    

def collect_pos_data(engine, board, depth):
    to_score = board if board.turn else board.mirror()
    lm = list(to_score.legal_moves)
    
    tsf = to_score.fen()
    out = {'fen': tsf, 'moves': [], 'tokens': board_to_tokens(to_score).tolist()}
    
    info = analyze(engine, to_score, depth=depth)
    best_cp = info[0]['score'].white().score(mate_score=2500)
    worst_cp = info[-1]['score'].white().score(mate_score=2500)
    
    all_uci = set()
    for i in info:
        uci = str(i['pv'][0])
        all_uci.add(uci)
        this_cp = i['score'].white().score(mate_score=2500)
        cp_loss = best_cp - this_cp
        out['moves'].append({"uci": uci, "cp": this_cp, "cp_loss": cp_loss})
        
    # multipv misses some sometimes
    not_found = [l for l in lm if str(l) not in all_uci]
    if not_found:
        # theyre the worst moves, trimmed in alpha beta
        for nf in not_found:
            uci = str(nf)
            out['moves'].append({"uci":uci, "cp":worst_cp, "cp_loss":best_cp-worst_cp})
            
    return out


def run_game(board, engine, depth, pos_per_game):
    pos = []
    while not board.is_game_over() and len(pos) < pos_per_game:
        # job the game forward a few moves
        n_moves = random.randint(2, 12)
        for n in range(n_moves):
            try:
                if np.random.random() <= 0.1:
                    board = make_random_move(board)
                    
                else:
                    result = engine.play(board, limit=chess.engine.Limit(time=0.005))
                    board.push(result.move)
                    
                if board.is_game_over():
                    break
                
            # if anything went wrong just return, try again
            except:
                return pos
        
        # need at least 2 moves to get good data
        if len(list(board.legal_moves)) < 2:
            continue
        
        pos.append(collect_pos_data(engine, board, depth))
    
    return pos


def count_existing(outfile):
    if not os.path.exists(outfile):
        return 0
    with open(outfile, "r") as f:
        return sum(1 for _ in f)


def append_positions(outfile, positions):
    with open(outfile, "a") as f:
        for pos in positions:
            f.write(json.dumps(pos) + "\n")


def generate_eval_dataset(n_positions, pos_per_game, depth, outfile):
    listify = lambda o: o.tolist() if hasattr(o, "tolist") else o
    # Ensure file exists
    if not os.path.exists(outfile):
        print(f"{outfile} does not exist. Creating...")
        with open(outfile, "w") as f:
            pass

    existing = count_existing(outfile)
    remaining = n_positions - existing
    if remaining <= 0:
        print(f"{outfile} already has {existing} positions (>= target). Nothing to do.")
        return

    print(f"Resuming: {existing}/{n_positions} positions already exist. "
          f"Need {remaining} more.")

    pbar = tqdm(total=n_positions, initial=existing, desc="Generating positions")

    names, boards = get_all_openings()

    # Seed with opening positions if starting fresh
    if existing == 0:
        with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
            opening_positions = [collect_pos_data(engine, b, depth) for b in boards]
        append_positions(outfile, opening_positions)
        existing += len(opening_positions)
        pbar.update(len(opening_positions))

    # Build pool of boards
    all_boards = boards * 12
    for i in range(160):
        all_boards.append(random_init(1 + i % 5))
    random.shuffle(all_boards)

    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine, open(outfile, "a") as f:
        for board in all_boards:
            new_pos = run_game(board, engine, depth, pos_per_game)
            for pos in new_pos:
                f.write(json.dumps(pos, default=listify) + "\n")
                f.flush()
                os.fsync(f.fileno())  # force write to disk
                existing += 1
                pbar.update(1)

                if existing >= n_positions:
                    break
            if existing >= n_positions:
                break

    pbar.close()
    print(f"Saved {existing} total positions to {outfile}")


def inspect_eval_dataset(path, n=5):
    """
    Load and inspect a few records from the eval dataset.
    """
    positions = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            try:
                positions.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line {i}: {e}")
            if len(positions) >= n:
                break

    print(f"Loaded {len(positions)} example positions from {path}")
    for i, pos in enumerate(positions):
        print(f"\n--- Position {i} ---")
        print("FEN:", pos["fen"])
        print("Num moves:", len(pos["moves"]))
        if pos["moves"]:
            print("Example move:", pos["moves"][0])
        print("Tokens (len={}): {}".format(len(pos["tokens"]), pos["tokens"][:20]))


def verify_eval_file(path, max_print=5):
    total = 0
    valid = 0
    bad_lines = []

    with open(path, "r") as f:
        for i, line in enumerate(f):
            total += 1
            try:
                pos = json.loads(line)
                fen = pos.get("fen", None)
                if fen is None:
                    bad_lines.append((i, "missing fen"))
                    continue
                try:
                    chess.Board(fen)  # will raise if invalid
                    valid += 1
                except Exception as e:
                    bad_lines.append((i, f"invalid fen: {fen}"))
            except json.JSONDecodeError as e:
                bad_lines.append((i, f"json error: {e}"))

    print(f"File: {path}")
    print(f"Total lines: {total}")
    print(f"Valid positions: {valid}")
    print(f"Invalid/missing: {total - valid}")

    if bad_lines:
        print("\nExamples of bad lines:")
        for i, msg in bad_lines[:max_print]:
            print(f"  Line {i}: {msg}")


if __name__ == '__main__':
    DATA_DIR = "C:/Users/Bryan/Data/chessbot_data/training_data"
    outfile = os.path.join(DATA_DIR, "tf_model_pos_eval_data.jsonl")
    generate_eval_dataset(5000, 10, 12, outfile)
    
    inspect_eval_dataset(outfile, n=5)
    verify_eval_file(outfile, max_print=5)




