"""Build a cutechess-cli PGN opening book from UHO and pre_opened sources."""
import io
import pickle
import random
import chess
import chess.pgn

UHO_PGN = "C:/Users/Bryan/Data/chessbot_data/opening_books/UHO_XXL_2022_+100_+129.pgn"
PRE_OPENED_PKL = "C:/Users/Bryan/Data/chessbot_data/pre_opened_uci_paths_quota_4to8.pkl"
OUT_PATH = "C:/Users/Bryan/Data/chessbot_data/cutechess/opening_book.pgn"

N_UHO = 10000
N_PRE = 3000
SEED = 42

random.seed(SEED)


def uci_path_to_pgn(moves):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    for uci in moves:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    game.headers["Event"] = "opening"
    game.headers["White"] = "?"
    game.headers["Black"] = "?"
    game.headers["Result"] = "*"
    return game


def sample_uho(path, n):
    print(f"Scanning UHO PGN...")
    games = []
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
            if len(games) % 50000 == 0:
                print(f"  read {len(games)}...")
    print(f"Total UHO games: {len(games)}, sampling {n}")
    sampled = random.sample(games, n)
    # Strip result and comments, keep only moves
    cleaned = []
    for g in sampled:
        board = chess.Board()
        new_game = chess.pgn.Game()
        new_game.headers["Event"] = "opening"
        new_game.headers["White"] = "?"
        new_game.headers["Black"] = "?"
        new_game.headers["Result"] = "*"
        node = new_game
        for move in g.mainline_moves():
            node = node.add_variation(move)
        cleaned.append(new_game)
    return cleaned


def main():
    with open(PRE_OPENED_PKL, "rb") as f:
        pre_data = pickle.load(f)

    pre_sampled = random.sample(pre_data, N_PRE)
    pre_games = [uci_path_to_pgn(path) for path in pre_sampled]
    print(f"Pre-opened: {len(pre_games)} games")

    uho_games = sample_uho(UHO_PGN, N_UHO)

    all_games = uho_games + pre_games
    random.shuffle(all_games)
    print(f"Total games: {len(all_games)}, writing to {OUT_PATH}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for game in all_games:
            print(game, file=f)
            print(file=f)

    print("Done.")


if __name__ == "__main__":
    main()
