import json, sys, os
import pandas as pd

import chess
import chess.engine
from chessbot import SF_LOC
from chessbot.utils import rnd, score_pov_cp, show_board


RUN_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_selfplay"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # First, try regular JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: parse line-by-line (JSONL / concatenated objects)
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items
    

def load_game_index(path=None):
    if path is None:
        path = os.path.join(RUN_DIR, "game_index.json")
        
    return load_json(path)


def summarize_games(games):
    rows = []
    for g in games:
        vs_sf = bool(g.get("started_vs_stockfish"))
        sf_col = g.get("stockfish_color")
        sf_color = "white" if sf_col is True else ("black" if sf_col is False else None)

        rows.append({
            "scenario": g.get("scenario"),
            "result": g.get("result"),
            "plies": g.get("plies"),
            "vs_sf": vs_sf,
            "beat_sf": g.get("beat_sf") if vs_sf else None,
            "sf_color": sf_color,
        })

    df = pd.DataFrame(rows)

    grp = df.groupby(["scenario", "sf_color"], dropna=False)

    out = grp.agg(
        n_games=("result", "size"),
        avg_result=("result", "mean"),
        avg_plies=("plies", "mean"),
        n_vs_sf=("vs_sf", "sum"),
        n_selfplay=("vs_sf", lambda s: int((~s).sum())),
        n_beat_sf=("beat_sf", lambda s: int((s == True).sum())),
    ).reset_index()

    out["beat_sf_rate"] = out.apply(
        lambda r: (r["n_beat_sf"] / r["n_vs_sf"]) if r["n_vs_sf"] else None, axis=1
    )

    return out


def fetch_games(games, **kwargs):
    """
    Filter a list[dict] of games by key=value conditions.
    - If a value is callable, it will be used as a predicate on the field.
      e.g., fetch_games(games, plies=lambda x: x and x > 40)
    - Otherwise, equality is used.
    Missing keys fail the match.
    """
    def match(g):
        for k, v in kwargs.items():
            if callable(v):
                if not v(g.get(k)):
                    return False
            else:
                if g.get(k, object()) != v:
                    return False
        return True

    return [g for g in games if match(g)]


games = load_game_index()
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
    for g in games:
        if g['started_vs_stockfish']:
            if g['stockfish_color']:
                g['white'] = 'stockfish'
                g['black'] = 'chessbot'
            else:
                g['white'] = 'chessbot'
                g['black'] = 'stockfish'
        else:
            g['white'] = 'chessbot'
            g['black'] = 'chessbot'
                
        full_json = load_json(g['json_file'])
        board = chess.Board(full_json['start_fen'])
        info = eng.analyse(
            board, limit=chess.engine.Limit(depth=15),
            info=chess.engine.INFO_SCORE
        )
        cp = score_pov_cp(info['score'], board.turn, mate_cp=1500)
        g['starting_cp'] = cp


def step_game(board, moves_uci, eng, flipped=False, depth=15, mate_cp=1500):
    """
    Controls:
      <enter>  next move
      b        back one move
      s        eval (always White POV)
      g <n>    jump to ply n (0-based)
      q        quit
    """
    def who_moved(ply_idx, sf_color):
        if sf_color is None:
            return "Move"
        return "SF" if (sf_color and ply_idx % 2 == 0) or ((not sf_color) and ply_idx % 2 == 1) else "Bot"

    def eval_white_pov():
        info = eng.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_SCORE)
        sc = info["score"].pov(chess.WHITE)  # always White POV
        if sc.is_mate():
            return f"mate {sc.mate()}"
        return f"{sc.score() if sc.score() is not None else 0} cp"

    def redraw():
        show_board(board, flipped=flipped)
        print(f"ply={len(board.move_stack)} | eval(White POV)={eval_white_pov()}")

    ply = 0
    redraw()

    while True:
        cmd = input("[enter]=next, b=back, s=eval, g <n>=goto, q=quit > ").strip()
        if cmd == "":
            if ply >= len(moves_uci):
                print("end of game")
                continue
            u = moves_uci[ply]
            mv = chess.Move.from_uci(u)
            label = who_moved(ply, flipped)  # flipped == stockfish_color
            print(f"{label} plays {board.san(mv)}")
            board.push(mv)
            ply += 1
            redraw()
            continue

        if cmd == "b":
            if board.move_stack:
                board.pop()
                ply -= 1
            redraw()
            continue

        if cmd == "s":
            print(f"eval(White POV)={eval_white_pov()}")
            continue

        if cmd.startswith("g "):
            n = int(cmd.split()[1])
            n = max(0, min(n, len(moves_uci)))
            while board.move_stack:
                board.pop()
            for i in range(n):
                board.push_uci(moves_uci[i])
            ply = n
            redraw()
            continue

        if cmd == "q":
            break

        print("unknown command")


wins = [g for g in games if g.get('beat_sf', False)]
for g in wins:
    print(g['scenario'], g['white'], g['black'], g['starting_cp'], g['started_vs_stockfish'])
    
    
to_analyze = fetch_games(games, beat_sf=True, scenario="random_init")[0]
game_data = load_json(to_analyze['json_file'])
board = chess.Board(game_data['start_fen'])
move_data = game_data['tree_search_data']
move_list = game_data['moves_played']

# usage (fits your vars):
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
    step_game(board, move_list, eng, flipped=game_data.get('stockfish_color'), depth=15)











