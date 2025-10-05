import time
import random
from uuid import uuid4
from copy import deepcopy
import numpy as np
import chess

from pyfastchess import Board as FastBoard, Evaluator, terminal_value_white_pov
from chessbot.psqt import build_weights, jostle_weights
from chessbot.absearch import GAUnit
from chessbot.utils import show_board  # optional

# ========== CONFIG ==========
POP_SIZE = 16
DEPTH = 1
QPLY = 3
QCAPTURES = 5
N_EPOCHS = 10
MAX_MOVES_PER_GAME = 150
VERBOSE = True
# ============================


# helpers
def make_contestant(weights, name=None, created_epoch=0):
    return {
        "id": str(uuid4())[:8],
        "name": name or str(uuid4())[:6],
        "weights": deepcopy(weights),
        "created_epoch": created_epoch,
        "age": 0,
        # global accumulators
        "points": 0.0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "wins_with_black": 0,
        "games_played": 0,
    }


def build_engine_for(contestant):
    ev = Evaluator()
    ev.configure(contestant["weights"])
    g = GAUnit(ev, depth=DEPTH)
    g.max_qply = QPLY
    g.qcaptures = QCAPTURES
    return g  # return GAUnit (contains evaluator + TT)


def combine_weights_mask(wA, wB, seed):
    """Create a child weight dict by taking entries from wA or wB according to a random mask.
       Works elementwise for numpy arrays in the dict.
    """
    rng = np.random.RandomState(seed)
    child = {}
    for k in wA:
        a = wA[k]
        b = wB[k]
        # assume both are numpy arrays or scalars; handle scalars first
        if np.isscalar(a):
            # choose from A or B at random
            child[k] = a if rng.rand() < 0.5 else b
        else:
            arr_a = np.array(a)
            arr_b = np.array(b)
            # create boolean mask same shape
            mask = rng.rand(*arr_a.shape) < 0.5
            out = np.where(mask, arr_a, arr_b)
            child[k] = out.astype(arr_a.dtype)
    return child


def canonical_result_from_board(b):
    reason, raw_res = b.is_game_over()
    tv = terminal_value_white_pov(b)
    if tv is None:
        # ongoing — should not be called if game not over
        return None, reason
    if tv == 0:
        return "1/2-1/2", reason
    return ("1-0", reason) if tv == 1.0 else ("0-1", reason)


def update_stats(result, white_c, black_c):
    """Update per-contestant stats from canonical result '1-0'/'0-1'/'1/2-1/2'"""
    if result == "1-0":
        w_pts, b_pts = 1.0, 0.0
    elif result == "0-1":
        w_pts, b_pts = 0.0, 1.0
    else:
        w_pts, b_pts = 0.5, 0.5

    white_c["points"] += w_pts
    black_c["points"] += b_pts
    white_c["games_played"] += 1
    black_c["games_played"] += 1

    if w_pts == 1.0:
        white_c["wins"] += 1
    elif w_pts == 0.5:
        white_c["draws"] += 1
    else:
        white_c["losses"] += 1

    if b_pts == 1.0:
        black_c["wins"] += 1
        black_c["wins_with_black"] += 1
    elif b_pts == 0.5:
        black_c["draws"] += 1
    else:
        black_c["losses"] += 1

# initial population
def initial_population(seed=0):
    base = build_weights(zeros=True)
    boost = build_weights(zeros=False)

    pop = []
    # keep two canonical bases A and B
    pop.append(make_contestant(base, name="A", created_epoch=0))
    pop.append(make_contestant(base, name="B", created_epoch=0))
    # two jostles from base and boost
    aj1 = make_contestant(
        jostle_weights(base, delta=5, seed=seed+1), name="AJ1", created_epoch=0)
    pop.append(aj1)
    
    bj1 = make_contestant(
        jostle_weights(boost, delta=5, seed=seed+2), name="BJ1", created_epoch=0)
    pop.append(bj1)
    
    # fill remainder with jostled variants (mix from base & boost)
    i = 0
    while len(pop) < POP_SIZE:
        if i % 2 == 0:
            w = jostle_weights(base, delta=6 + (i % 4), seed=seed + 10 + i)
            name = f"AJ{i//2+2}"
        else:
            w = jostle_weights(boost, delta=6 + (i % 4), seed=seed + 10 + i)
            name = f"BJ{i//2+2}"
        pop.append(make_contestant(w, name=name, created_epoch=0))
        i += 1
    return pop


# all-vs-all schedule (startpos only)
def build_all_pairs(pop_size):
    pairs = []
    for i in range(pop_size):
        for j in range(i+1, pop_size):
            pairs.append((i, j))
    # for each pair produce two games: (i white, j black) and (j white, i black)
    games = []
    for (i, j) in pairs:
        games.append((i, j))
        games.append((j, i))
    return games


#  play epoch with pseudo-parallel stepping 
def run_epoch_pseudoparallel(population, epoch_index, seed=0, verbose=True):
    # Build engines: one GAUnit per contestant (TT preserved across games)
    engines = [build_engine_for(c) for c in population]

    # Create games list from startpos pairs
    pairs = build_all_pairs(len(population))  # list of (white_idx, black_idx)
    games = []
    for gid, (wi, bi) in enumerate(pairs):
        game = {
            "id": gid,
            "board": FastBoard(),
            "white": wi,
            "black": bi,
            "finished": False,
            "result": None,
            "reason": None,
            "moves": [],
            "move_count": 0,
        }
        games.append(game)

    total_games = len(games)
    if verbose:
        print(
            f"Epoch {epoch_index}: {len(population)} ",
            f"contestants -> {total_games} games (pseudo-parallel)"
        )
    
    loop_iter = 0
    while True:
        loop_iter += 1
        any_active = False
        shown=False
        for g in games:
            if g["finished"]:
                continue
            any_active = True
            b = g["board"]
            # if game already terminal, finalize
            if b.is_game_over()[0] != "none":
                r, reason = canonical_result_from_board(b)
                g["finished"] = True
                g["result"], g["reason"] = r, reason
                update_stats(r, population[g["white"]], population[g["black"]])
                continue

            # side to move
            stm = b.side_to_move()  # 'w' or 'b'
            mover_idx = g["white"] if stm == "w" else g["black"]
            mover = engines[mover_idx]
            # ask engine to pick a move (search uses board and returns UCI)
            best, score, info = mover.search(b, max_depth=DEPTH, verbose=False)

            # push move onto this game's board
            b.push_uci(best)
            if not shown:
                show_board(chess.Board(b.fen()))
                shown = True
            
            g["moves"].append(best)
            g["move_count"] += 1

            # check termination immediately
            if b.is_game_over()[0] != "none" or g["move_count"] >= MAX_MOVES_PER_GAME:
                r, reason = canonical_result_from_board(b)
                g["finished"] = True
                g["result"], g["reason"] = r, reason
                update_stats(r, population[g["white"]], population[g["black"]])
                continue

        if not any_active:
            break

    # all games finished
    # produce per-epoch standings (points, wins, wins_with_black, random tiebreak)
    for c in population:
        # age update
        c["age"] = epoch_index - c["created_epoch"]

    standings = sorted(
        population,
        key=lambda c: (-c["points"], -c["wins"], -c["wins_with_black"], random.random()))
    
    if verbose:
        print("\nEpoch standings (top 8):")
        for i, s in enumerate(standings[:8], 1):
            print(
                f"{i:2}. {s['name']:16} pts={s['points']:.1f} ",
                f"w={s['wins']:2} wb={s['wins_with_black']:2} ",
                f"games={s['games_played']} age={s['age']}")
            
    return standings, games

# selection & reseed
def make_next_generation(population, standings, epoch,
                         top_pct=25, jostle_delta_base=8,
                         meiosis_jostle_delta=3):
    """
    Generalized generation builder.
    - population / standings: lists of contestant dicts (same order as before)
    - epoch: integer epoch number (used for seeding/names)
    - top_pct: percentage (int) of population used as parent pool (e.g. 25)
    Returns next_pop list (same length as population).
    """
    N = len(population)
    assert N >= 16 and N % 4 == 0, "pop must be >=16 and multiple of 4"
    rng = random.Random(epoch)

    # parent pool = top top_pct% of standings (must be at least 4)
    pool_size = max(4, int(N * top_pct // 100))
    # ensure pool_size is multiple of 1 (we'll accept any >=4)
    parent_pool = standings[:pool_size]

    chunk = N // 4   # number per category (kept/jostle/meiosis/meiosis+jostle)
    next_pop = []

    # 1) keep top chunk unchanged (elite)
    # take the top chunk from the standings (not just parent_pool)
    for p in standings[:chunk]:
        next_pop.append(make_contestant(p["weights"],
                                       name=p["name"],
                                       created_epoch=p["created_epoch"]))

    # 2) chunk jostled variants sampled from parent_pool (with replacement)
    for i in range(chunk):
        parent = parent_pool[i % pool_size]  # cycle deterministically
        delta = jostle_delta_base + (epoch % 3)
        child_w = jostle_weights(parent["weights"],
                                 delta=delta, seed=epoch*100 + i + 1)
        name = f"{parent['name'][:3]}J{epoch}"
        next_pop.append(make_contestant(child_w, name=name,
                                        created_epoch=epoch))

    # 3) chunk meiosis children from random pairs in parent_pool
    for i in range(chunk):
        a = rng.choice(parent_pool)
        b = rng.choice(parent_pool)
        child_w = combine_weights_mask(a["weights"], b["weights"],
                                       seed=epoch*1000 + i)
        name = f"M{a['name'][:2]}{b['name'][:2]}{epoch}"
        next_pop.append(make_contestant(child_w, name=name,
                                        created_epoch=epoch))

    # 4) chunk meiosis children + slight jostle (do not use group3 kids)
    for i in range(chunk):
        a = rng.choice(parent_pool)
        b = rng.choice(parent_pool)
        child_w = combine_weights_mask(a["weights"], b["weights"],
                                       seed=epoch*2000 + i)
        child_w = jostle_weights(child_w, delta=meiosis_jostle_delta,
                                 seed=epoch*3000 + i)
        name = f"MJ{a['name'][:2]}{b['name'][:2]}{epoch}"
        next_pop.append(make_contestant(child_w, name=name,
                                        created_epoch=epoch))

    # safety check
    assert len(next_pop) == N, f"next_pop size {len(next_pop)} != {N}"
    return next_pop


# top-level evolution
def run_evolution_all_vs_all(n_epochs=N_EPOCHS, seed=0, verbose=VERBOSE):
    # initial population
    population = initial_population(seed=seed)
    history = []

    for epoch in range(1, n_epochs + 1):
        # zero out per-epoch stats
        for c in population:
            c["points"] = 0.0
            c["wins"] = 0
            c["draws"] = 0
            c["losses"] = 0
            c["wins_with_black"] = 0
            c["games_played"] = 0

        if verbose:
            print("\n" + "="*60)
            print(f"RUNNING EPOCH {epoch}")
            print("="*60)

        standings, games = run_epoch_pseudoparallel(
            population, epoch, seed=seed, verbose=verbose)
        
        history.append({"epoch": epoch, "standings": standings, "games": games})

        # select next generation
        population = make_next_generation(population, standings, epoch)

    return {"population": population, "history": history}


if __name__ == "__main__":
    t0 = time.time()
    res = run_evolution_all_vs_all(n_epochs=N_EPOCHS, seed=42, verbose=True)
    print("Elapsed:", time.time() - t0)
    # final active
    print("\nFINAL TOP 8:")
    sorted_res = sorted(res["population"], key=lambda x: (-x["points"], -x["wins"]))[:8]
    for i, c in enumerate(sorted_res):
        print(i+1, c["name"], c["points"], c["wins"], c["wins_with_black"])


