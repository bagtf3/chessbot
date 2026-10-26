from chessbot import SF_LOC  # do not change or rename this
from chessbot.utils import format_time
from chessbot.game_utils import get_pre_opened_game
import chess
import chess.engine
import chess.pgn
import random
import numpy as np
import time
import weakref

# User knobs
ABS_TIME_LIMIT = 1.0
DEPTHS = [1, 3, 5, 8, 11, 14, 17, 21, 25]

if 'ENGINE_LIST' not in globals():
    ENGINE_LIST = []


def finalize_engine(eng):
    # only quit if this engine is still registered (single control point)
    if eng in ENGINE_LIST:
        ENGINE_LIST.remove(eng)
        eng.close()


def kill_engines():
    while ENGINE_LIST:
        e = ENGINE_LIST.pop()
        e.close()
        del e
        

class SFPlayer():
    def __init__(self, depth):
        self.depth = depth
        self.name = f"SF14d{depth}"
        self.current_elo = 2350 + np.random.random()
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.highest_rated_win = 0
        self.highest_rated_win_opp = ""

        self.engine = chess.engine.SimpleEngine.popen_uci(SF_LOC)
        threads = 1 if depth <= 6 else 2
        self.engine.configure({"Hash": 256, "Threads": threads})
        ENGINE_LIST.append(self.engine)
        
        self.limit = chess.engine.Limit(time=ABS_TIME_LIMIT, depth=self.depth)

        # register finalizer: called when this Player is GC'd
        self.finalizer = weakref.finalize(self, finalize_engine, self.engine)

    def __str__(self):
        return f"<Class Player>: {self.name}"

    def __repr__(self):
        return self.__str__()

    def get_limit(self):
        return self.limit
    
    def update_time_limit(self, time):
        self.limit = chess.engine.Limit(time=time, depth=self.depth)
        
    def get_engine(self):
        return self.engine

    def play(self, board):
        return self.engine.play(board, self.get_limit())
    

class SFPlayerRandomK(SFPlayer):
    def __init__(self, depth, k=5):
        super().__init__(depth)
        self.name = self.name = f"SF14d{depth}K{k}"
        self.k = k

    def play(self, board):
        """
        Analyse with multipv=self.k, sample one PV weighted by its cp
        (from side-to-move perspective), and return a minimal PlayResult-
        like object with .move and .info so it plugs into existing loops.
        """
        ana = self.engine.analyse(board, self.get_limit(), multipv=self.k)
        cps, pvs = [], []
        for entry in ana:
            pv = entry.get("pv")
            if not pv:
                continue
            score = entry.get("score")
            if score is None:
                cp = 0
            else:
                if score.is_mate():
                    m = score.mate()
                    cp = 100000 if m > 0 else -100000
                else:
                    cp = score.pov(board.turn).score() or 0
            pvs.append(pv)
            cps.append(cp)

        min_cp = min(cps)
        weights = [c - min_cp + 1 for c in cps]
        chosen_pv = random.choices(pvs, weights=weights, k=1)[0]
        chosen_move = chosen_pv[0]

        class Result:
            pass

        r = Result()
        r.move = chosen_move
        r.info = {"multipv": self.k, "weights": weights}
        return r


def play_pair(player1, player2, fen):
    eng1 = player1.engine
    eng2 = player2.engine

    limit1 = player1.get_limit()
    limit2 = player2.get_limit()

    pgns = []
    results = []

    for rep in range(2):
        eng1_is_white = (rep == 0)

        # starting board
        board = chess.Board(fen)

        # build PGN game and attach starting position if non-standard
        game = chess.pgn.Game()
        game.setup(board)
        game.headers["FEN"] = board.fen()
        game.headers["SetUp"] = "1"
        game.headers["Event"] = "Elo Depth test"
        game.headers["Site"] = "mybox"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["Round"] = str(rep + 1)

        if eng1_is_white:
            game.headers["White"] = player1.name
            game.headers["Black"] = player2.name
        else:
            game.headers["White"] = player2.name
            game.headers["Black"] = player1.name

        # play
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                engine = eng1 if eng1_is_white else eng2
                limit = limit1 if eng1_is_white else limit2
            else:
                engine = eng2 if eng1_is_white else eng1
                limit = limit2 if eng1_is_white else limit1

            res = engine.play(board, limit)
            board.push(res.move)

        result = board.result()
        results.append([game.headers["White"], game.headers["Black"], result])
        game.headers["Result"] = result

        exporter = chess.pgn.StringExporter(
            headers=True, variations=True, comments=False
        )
        pgn_text = game.accept(exporter)
        if not pgn_text.strip().endswith(result):
            pgn_text = pgn_text.strip() + "\n\n" + result + "\n"
        pgns.append(pgn_text)

    return pgns, results


def update_round(players, games, K=32):
    """
    players: list of Player objects
    games: iterable of (white_name, black_name, result) where result is
           "1-0", "0-1", or "1/2-1/2"
    K: learning rate for this round
    returns: the same list 'players' with updated fields
    """
    name_to_player = {p.name: p for p in players}

    for white, black, result in games:
        pw = name_to_player[white]
        pb = name_to_player[black]

        Ra = pw.current_elo
        Rb = pb.current_elo

        # expected score for white
        Ea = 1.0 / (1.0 + 10 ** ((Rb - Ra) / 400.0))

        # determine actual score from White POV and winner/loser names
        if result == "1-0":
            Sa = 1.0
            winner_name = white
            loser_name = black
        elif result == "0-1":
            Sa = 0.0
            winner_name = black
            loser_name = white
        else:
            Sa = 0.5
            winner_name = None
            loser_name = None

        # record win/draw/loss counts BEFORE Elo change
        if winner_name is not None:
            if winner_name == pw.name:
                pw.wins += 1
                pb.losses += 1
            else:
                pb.wins += 1
                pw.losses += 1
        else:
            pw.draws += 1
            pb.draws += 1

        # if there's a winner, update highest_rated_win using opponent's
        # rating prior to any Elo modification
        if winner_name is not None:
            winner = name_to_player[winner_name]
            loser = name_to_player[loser_name]
            beat_rating = loser.current_elo
            if beat_rating > winner.highest_rated_win:
                winner.highest_rated_win = beat_rating
                winner.highest_rated_win_opp = loser.name

        # Elo update (symmetrical). Use Sa and Ea computed above.
        dR = K * (Sa - Ea)
        pw.current_elo = Ra + dR
        pb.current_elo = Rb - dR
        
    players.sort(key=lambda p: p.current_elo)
    return players


def run_round(players):
    all_res = []
    all_pgns = []
    start = time.time()

    # snapshot ranking low -> high
    n = len(players)

    for i in range(1, n-1):
        indices = [[i-1, i], [i, i+1]]
        for a, b in indices:
            p1 = players[a]
            p2 = players[b]

            fen = get_pre_opened_game().fen()
            pgns, results = play_pair(p1, p2, fen)

            all_pgns += pgns
            all_res += results

    stop = time.time()
    print(f"Round Completed in {format_time(stop - start)}")

    return all_res, all_pgns


def print_standings(players):
    """
    Pretty-print player standings.
    Shows: Name, Elo (2 dp), W D L, top beaten rating, top beaten opponent.
    """
    name_w = 16
    elo_w = 8
    wdl_w = 3   # each of W/D/L
    beat_w = 7
    opp_w = 18

    hdr = (
        f"{'Name':<{name_w}} "
        f"{'Elo':>{elo_w}}   "
        f"{'W':>{wdl_w}} {'D':>{wdl_w}} {'L':>{wdl_w}}   "
        f"{'TopBeat':>{beat_w}}   {'Opponent':<{opp_w}}"
    )
    sep = "-" * (name_w + elo_w + wdl_w*3 + beat_w + opp_w + 18)

    print(hdr)
    print(sep)

    # show highest-first for readability (does not mutate original list)
    for p in sorted(players, key=lambda x: x.current_elo, reverse=True):
        top = p.highest_rated_win
        top_str = f"{int(top)}" if top else "-"
        opp = p.highest_rated_win_opp or "-"

        line = (
            f"{p.name:<{name_w}} "
            f"{p.current_elo:>{elo_w}.2f}   "
            f"{p.wins:>{wdl_w}} {p.draws:>{wdl_w}} {p.losses:>{wdl_w}}   "
            f"{top_str:>{beat_w}}   {opp:<{opp_w}}"
        )
        print(line)
    print()


def update_time_limits(players, new_time):
    for p in players:
        p.update_time_limit(new_time)

        
def reset_best_wins(players):
    for p in players:
        p.highest_rated_win = 0
        p.highest_rated_win_opp = ""
    
        
#%%
players = [SFPlayerRandomK(1, 5)]
players += [SFPlayer(d) for d in DEPTHS]
K = 64

tl = 0.05
update_time_limits(players, tl)
all_res, all_pgns = [], []
for r in range(45):
    print(f" starting round {r} ".center(76, "="))
    if r == 0:
        print_standings(players)
        update_time_limits(players, tl)
    elif r == 4:
        reset_best_wins(players)
            
    elif r > 4:
        tl += 0.1
        update_time_limits(players, min(tl, ABS_TIME_LIMIT))
        
        if r % 3 == 0:
            K = max(16, K / 2)
            print(f"[update] K={K}")
                
    round_res, round_pgns = run_round(players)
    all_res += round_res; all_pgns += round_pgns
    players = update_round(players, round_res, K=K)
    print_standings(players)


kill_engines()
while players:
    p = players.pop()
    del p
        
            
