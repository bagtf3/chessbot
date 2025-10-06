# tournament.py (seedless, jostle/meiosis inlined as make_new_generation)
import time, os, random
from uuid import uuid4
from copy import deepcopy
import numpy as np
import chess
import pickle

from pyfastchess import Board as FastBoard, Evaluator, terminal_value_white_pov
from chessbot.psqt import build_weights, jostle_weights
from chessbot.absearch import GAUnit
from chessbot.utils import show_board

# ========== CONFIG ========== #
POP_SIZE = 16
DEPTH = 1
QPLY = 3
QCAPTURES = 5
N_EPOCHS = 10
MAX_MOVES_PER_GAME = 200
MAT_CUTOFF = 7
MAT_PLYS = 7
VERBOSE = True
SAVE_DIR = "C:/Users/Bryan/Data/chessbot_data/GA_results"
# =========================== #


def save_pickle(res):
    os.makedirs(SAVE_DIR, exist_ok=True)
    fname = "GA_p16_e50_res_history.pkl"
    path = os.path.join(SAVE_DIR, fname)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

        
# ELO helpers
def _expected_score(rA, rB):
    """Expected score for A vs B."""
    return 1.0 / (1.0 + 10 ** ((rB - rA) / 400.0))


def _k_for(contestant):
    """Simple K policy (can be replaced by your preferred rule)."""
    # new players: faster learning
    if contestant.get("lifetime_games_played", 0) < 30:
        return 40
    r = contestant.get("rating", 1500)
    if r >= 2400:
        return 10
    return 20


def _score_from_result_for_white(result):
    """Canonical result is '1-0', '0-1', or '1/2-1/2' from white's pov."""
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    return 0.5


def apply_elo_update(white_c, black_c, result, gid=None, epoch=None):
    """
    Update in-place ratings for two contestants from a single finished game.
    Appends a small rating-history event to each contestant under 'rating_history'.
    """
    # ensure rating fields exist
    if "rating" not in white_c:
        white_c["rating"] = 1500.0
        
    if "rating" not in black_c:
        black_c["rating"] = 1500.0
        
    rW_old = float(white_c["rating"])
    rB_old = float(black_c["rating"])
    eW = _expected_score(rW_old, rB_old)
    eB = 1.0 - eW

    sW = _score_from_result_for_white(result)
    sB = 1.0 - sW

    kW = _k_for(white_c)
    kB = _k_for(black_c)

    rW_new = rW_old + kW * (sW - eW)
    rB_new = rB_old + kB * (sB - eB)

    # write back
    white_c["rating"] = rW_new
    black_c["rating"] = rB_new

    # append small history for bookkeeping
    for C, old, new, opp, s in (
        (white_c, rW_old, rW_new, black_c.get("id"), sW),
        (black_c, rB_old, rB_new, white_c.get("id"), sB),
    ):
        hist = C.setdefault("rating_history", [])
        hist.append({
            "epoch": epoch,
            "game_id": gid,
            "opponent": opp,
            "old": round(old, 2),
            "new": round(new, 2),
            "score": s,
            "k": _k_for(C),
        })


def process_game_ratings(population, games, epoch):
    """
    Walk finished games and update Elo/rating for both participants.
    population is the list (indexable) of contestants; games is the list returned
    by run_epoch_pseudoparallel.
    """
    # games entries have keys: 'white', 'black', 'result', 'id' (per earlier code)
    for g in games:
        if not g.get("finished"):
            continue
        res = g.get("result")
        if res is None:
            continue
        wi = g["white"]
        bi = g["black"]
        gid = g.get("id")
        apply_elo_update(population[wi], population[bi], res, gid=gid, epoch=epoch)


# helpers
# auto-incrementing short names
NEXT_CONTESTANT_SEQ = 1


def next_contestant_name(prefix="C"):
    """Return a short sequential name like C0001, C0002, ..."""
    global NEXT_CONTESTANT_SEQ
    name = f"{prefix}{NEXT_CONTESTANT_SEQ:04d}"
    NEXT_CONTESTANT_SEQ += 1
    return name


def make_contestant(weights, name=None, created_epoch=0, cid=None, lineage=None):
    """
    Create a contestant dict. If name is None, assign a short sequential name.
    lineage is a list of lineage events (dictionaries). contestant_id if provided
    will be used, else random short id generated.
    """
    cid = cid or str(uuid4())[:8]
    nm = name or next_contestant_name()
    return {
        "id": cid,
        "name": nm,
        "weights": deepcopy(weights),
        "created_epoch": created_epoch,
        "age": 0,
        # per-epoch accumulators
        "points": 0.0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "wins_with_black": 0,
        "games_played": 0,
        # lifetime accumulators
        "lifetime_points": 0.0,
        "lifetime_wins": 0,
        "lifetime_draws": 0,
        "lifetime_losses": 0,
        "lifetime_wins_with_black": 0,
        "lifetime_games_played": 0,
        # pedigree
        "lineage": lineage or [{"op": "init", "parents": [], "epoch": created_epoch}],
        "rating": 1500.0,
        "rating_history": []
    }


def build_engine_for(contestant):
    ev = Evaluator()
    ev.configure(contestant["weights"])
    g = GAUnit(ev, depth=DEPTH)
    g.max_qply = QPLY
    g.qcaptures = QCAPTURES
    return g  # return GAUnit (contains evaluator + TT)


def combine_weights_mask(wA, wB):
    """
    Create a child weight dict by taking entries from wA or wB according
    to a random mask. Works elementwise for numpy arrays in the dict.
    Uses non-seeded RNG (no external seed passed).
    """
    rng = np.random.RandomState()  # non-deterministic by default
    child = {}
    for k in wA:
        a = wA[k]
        b = wB[k]
        # assume both are numpy arrays or scalars; handle scalars first
        if np.isscalar(a):
            child[k] = a if rng.rand() < 0.5 else b
        else:
            arr_a = np.array(a)
            arr_b = np.array(b)
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
    """Update per-contestant stats from canonical result."""
    if result == "1-0":
        w_pts, b_pts = 1.0, 0.0
    elif result == "0-1":
        w_pts, b_pts = 0.0, 1.0
    else:
        w_pts, b_pts = 0.5, 0.5

    # per-epoch
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

    # lifetime updates
    white_c["lifetime_points"] += w_pts
    black_c["lifetime_points"] += b_pts
    white_c["lifetime_games_played"] += 1
    black_c["lifetime_games_played"] += 1

    if w_pts == 1.0:
        white_c["lifetime_wins"] += 1
    elif w_pts == 0.5:
        white_c["lifetime_draws"] += 1
    else:
        white_c["lifetime_losses"] += 1

    if b_pts == 1.0:
        black_c["lifetime_wins"] += 1
        black_c["lifetime_wins_with_black"] += 1
    elif b_pts == 0.5:
        black_c["lifetime_draws"] += 1
    else:
        black_c["lifetime_losses"] += 1


# initial population (no seed)
def initial_population():
    base = build_weights(zeros=True)
    boost = build_weights(zeros=False)

    pop = []
    # keep two canonical bases A and B
    pop.append(make_contestant(base, name="A", created_epoch=0))
    pop.append(make_contestant(base, name="B", created_epoch=0))
    # two jostles from base and boost
    aj1 = make_contestant(jostle_weights(base, delta=5), name="AJ1", created_epoch=0)
    pop.append(aj1)

    bj1 = make_contestant(jostle_weights(boost, delta=5), name="BJ1", created_epoch=0)
    pop.append(bj1)

    # fill remainder with jostled variants (mix from base & boost)
    i = 0
    while len(pop) < POP_SIZE:
        if i % 2 == 0:
            w = jostle_weights(base, delta=6 + (i % 4))
            name = f"AJ{i//2+2}"
        else:
            w = jostle_weights(boost, delta=6 + (i % 4))
            name = f"BJ{i//2+2}"
        pop.append(make_contestant(w, name=name, created_epoch=0))
        i += 1
    return pop


# all-vs-all schedule (startpos only)
def build_all_pairs(pop_size):
    pairs = []
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            pairs.append((i, j))
    # for each pair produce two games: (i white, j black) and (j white, i black)
    games = []
    for (i, j) in pairs:
        games.append((i, j))
        games.append((j, i))
    return games


def check_termination(g, b, population):
    if b.is_game_over()[0] != "none":
        r, reason = canonical_result_from_board(b)

    elif g["move_count"] >= MAX_MOVES_PER_GAME:
        r = "1/2-1/2"
        reason = f"Move Limit ({MAX_MOVES_PER_GAME}) Reached"

    elif g["mat_cutoff_count"] >= MAT_PLYS:
        r = "1-0" if b.material_count() > 0 else "0-1"
        reason = f"Material Cutoff {MAT_CUTOFF} for {MAT_PLYS} plies"

    else:
        r = None

    if r is not None:
        g["finished"] = True
        g["result"], g["reason"] = r, reason
        update_stats(r, population[g["white"]], population[g["black"]])
        return True
    return False


# play epoch with pseudo-parallel stepping (no seed param)
def run_epoch_pseudoparallel(population, epoch_index, verbose=True):
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
            "mat_cutoff_count": 0,
            "mat_cutoff_owner": None,
        }
        games.append(game)
        
    random.shuffle(games)
    total_games = len(games)
    if verbose:
        print(
            f"Epoch {epoch_index}: {len(population)} ",
            f"contestants -> {total_games} games (pseudo-parallel)",
        )

    loop_iter = 0
    while True:
        loop_iter += 1
        any_active = False
        shown = False
        for g in games:
            if g["finished"]:
                continue
            any_active = True
            b = g["board"]

            # if game already terminal, finalize
            if check_termination(g, b, population):
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
            
            mat = b.material_count()
            if abs(mat) >= MAT_CUTOFF:
                # owner is the side that has the advantage now
                owner_idx = g["white"] if mat > 0 else g["black"]
                if g["mat_cutoff_owner"] == owner_idx:
                    g["mat_cutoff_count"] += 1
                else:
                    g["mat_cutoff_owner"] = owner_idx
                    g["mat_cutoff_count"] = 1
            else:
                # reset when advantage falls below the cutoff
                g["mat_cutoff_owner"] = None
                g["mat_cutoff_count"] = 0

            # check termination immediately
            if check_termination(g, b, population):
                continue

        if not any_active:
            break

    # all games finished
    # produce per-epoch standings (points, wins, wins_with_black, random tiebreak)
    for c in population:
        # age update
        c["age"] = epoch_index - c["created_epoch"]
    
    process_game_ratings(population, games, epoch_index)
    
    standings = sorted(
        population,
        key=lambda c: (-c["points"], -c["wins"], -c["wins_with_black"])
    )
    
    if verbose:
        print("\nEpoch standings (top 8):")
        for i, s in enumerate(standings[:8], 1):
            elo = s.get("rating", 1500.0)
            print(
                f"{i:2}. {s['name']:12} pts={s['points']:.1f} ",
                f"elo={elo:7.1f} ",
                f"w={s['wins']:2} wb={s['wins_with_black']:2} ",
                f"games={s['games_played']} age={s['age']}",
            )


    return standings, games


def make_new_generation(population, standings, epoch,
                        top_pct=25, jostle_delta_base=8,
                        meiosis_jostle_delta=3):
    """
    Build next population, preserving controls 'A' and 'B' if present.
    This version ensures the returned population size equals N exactly.
    """
    N = len(population)
    assert N >= 16 and N % 4 == 0, "pop must be >=16 and multiple of 4"
    rng = random.Random(epoch)

    # parent pool
    pool_size = max(4, int(N * top_pct // 100))
    parent_pool = standings[:pool_size]

    # find controls and clone them forward first
    next_pop = []
    added = set()
    controls = []
    for ctrl_name in ("A", "B"):
        ctrl = next((c for c in population if c.get("name") == ctrl_name), None)
        if ctrl:
            controls.append(ctrl)

    # clone controls
    for c in controls:
        clone = deepcopy(c)
        clone["points"] = 0.0
        clone["wins"] = clone["draws"] = clone["losses"] = 0
        clone["wins_with_black"] = 0
        clone["games_played"] = 0
        clone["lineage"].append({"o": "ctrl", "p": [c["id"]], "e": epoch})
        next_pop.append(clone)
        added.add(clone["id"])

    n_controls = len(controls)
    slots_left = N - n_controls
    # distribute slots_left across 4 buckets as evenly as possible
    base = slots_left // 4
    rem = slots_left % 4
    # assign extra slot to the earliest buckets (elite first)
    counts = [base + (1 if i < rem else 0) for i in range(4)]
    # counts -> [elite_count, jostle_count, meiosis_count, mj_count]
    elite_count, jostle_count, meiosis_count, mj_count = counts

    # 1) elite clones (skip already added controls)
    elites_added = 0
    for p in standings:
        if elites_added >= elite_count:
            break
        if p["id"] in added:
            continue
        new_p = deepcopy(p)
        new_p["points"] = 0.0
        new_p["wins"] = new_p["draws"] = new_p["losses"] = 0
        new_p["wins_with_black"] = 0
        new_p["games_played"] = 0
        new_p["lineage"].append({"o": "cl", "p": [p["id"]], "e": epoch})
        next_pop.append(new_p)
        added.add(new_p["id"])
        elites_added += 1

    # 2) jostled variants sampled from parent_pool (with replacement)
    for i in range(jostle_count):
        parent = parent_pool[i % pool_size]
        d = jostle_delta_base + (epoch % 3)
        w = jostle_weights(parent["weights"], delta=d)
        child = make_contestant(
            w, created_epoch=epoch, cid=str(uuid4())[:8],
            lineage=[{"o": "j", "p": [parent["id"]], "e": epoch, "d": d}])
        
        next_pop.append(child)
        added.add(child["id"])

    # 3) meiosis children from random pairs in parent_pool
    for i in range(meiosis_count):
        a = rng.choice(parent_pool)
        b = rng.choice(parent_pool)
        w = combine_weights_mask(a["weights"], b["weights"])
        child = make_contestant(
            w, created_epoch=epoch, cid=str(uuid4())[:8],
            lineage=[{"o": "m", "p": [a["id"], b["id"]], "e": epoch}])
        
        next_pop.append(child)
        added.add(child["id"])

    # 4) meiosis children + slight jostle
    for i in range(mj_count):
        a = rng.choice(parent_pool)
        b = rng.choice(parent_pool)
        w = combine_weights_mask(a["weights"], b["weights"])
        w = jostle_weights(w, delta=meiosis_jostle_delta)
        child = make_contestant(w, created_epoch=epoch,
                                cid=str(uuid4())[:8],
                                lineage=[{"o": "mj", "p": [a["id"], b["id"]],
                                          "e": epoch, "d": meiosis_jostle_delta}])
        next_pop.append(child)
        added.add(child["id"])

    # Safety/fill: if still short (rare), fill from top standings skipping added ids
    idx = 0
    while len(next_pop) < N:
        if idx >= len(standings):
            # fallback: jostle an arbitrary parent
            any_parent = rng.choice(population)
            w = jostle_weights(any_parent["weights"], delta=jostle_delta_base)
            child = make_contestant(
                w, created_epoch=epoch, cid=str(uuid4())[:8],
                lineage=[{"o": "fj", "p": [any_parent["id"]], "e": epoch}])
            
            next_pop.append(child)
            added.add(child["id"])
            continue
        cand = standings[idx]; idx += 1
        if cand["id"] in added:
            continue
        cl = deepcopy(cand)
        cl["points"] = 0.0
        cl["wins"] = cl["draws"] = cl["losses"] = 0
        cl["wins_with_black"] = 0
        cl["games_played"] = 0
        cl["lineage"].append({"o": "fc", "p": [cand["id"]], "e": epoch})
        next_pop.append(cl)
        added.add(cl["id"])

    assert len(next_pop) == N, f"next_pop size {len(next_pop)} != {N}"
    return next_pop



# top-level evolution (no seed)
def run_evolution_all_vs_all(n_epochs=N_EPOCHS, verbose=VERBOSE):
    # initial population
    population = initial_population()
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
            print("\n" + "=" * 60)
            print(f"RUNNING EPOCH {epoch}")
            print("=" * 60)

        standings, games = run_epoch_pseudoparallel(population, epoch, verbose=verbose)
        history.append({"epoch": epoch, "standings": standings, "games": games})

        # select next generation (seedless)
        if epoch < n_epochs:
            population = make_new_generation(population, standings, epoch)
            
    return {"population": population, "history": history}



if __name__ == "__main__":
    t0 = time.time()
    res = run_evolution_all_vs_all(n_epochs=N_EPOCHS, verbose=True)
    save_pickle(res['population'])
    print("Elapsed:", time.time() - t0)
    
    # final active
    print("\nFINAL TOP 16 (by ELO):")
    sorted_res = sorted(
        res["population"],
        key=lambda x: (-x.get("rating", 1500.0), -x["lifetime_points"],
                       -x["lifetime_wins"], -x["lifetime_wins_with_black"])
    )[:16]
    
    hdr = f"{'RANK':>4} {'NAME':^12} {'ELO':>7} {'LP':>6} {'LW':>4} {'LWB':>4} {'GAMES':>6}"
    print(hdr)
    print("-" * len(hdr))
    for i, c in enumerate(sorted_res, 1):
        print(
            f"{i:4d} {c['name']:^12} {c.get('rating', 1500.0):7.1f} "
            f"{c['lifetime_points']:6.1f} {c['lifetime_wins']:4d} "
            f"{c['lifetime_wins_with_black']:4d} {c['lifetime_games_played']:6d}"
        )

