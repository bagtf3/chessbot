"""Mine a stratified, weighted position corpus from selfplay runs.

Each position is one xc0 ply and lands in exactly one stratum: the first in
MIX order whose rule it matches. weight = stratum population / samples drawn,
so a weighted read reproduces the source mix. The refutation stratum needs
per-move tree data, which only reviewable games keep, so it is mined from
their logs; its population outside the scanned games is unknown and its
weight is NaN.

Every record is complete on its own: the df_all row, the logged ply, the game
header, and (reviewable games only) the full candidate table and PV. Records
carry init_fen + uci_path (the full line from the game's first position),
which is what xc0h history needs.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import pickle
from collections import Counter

import chess
import numpy as np
import pandas as pd

RUNS_DIR = r"D:\Data\chessbot_data\selfplay_runs"
CORPUS_DIR = r"D:\Data\chessbot_data\experiments\corpora"
ANALYZE_PKL = "analyze_results_combined.pkl"
DF_ALL_PARQUET = "df_all.parquet"
INDEX_FILE = "game_index.json"

CP_MAX = 1200
PEGGED_Q = 0.9
HIGH_QUANTILE = 0.90
CPL_QUANTILE = 0.95
MAX_PER_GAME = 10

REFUT_MIN_SIMS = 1000
REFUT_MIN_VISITS = 200
REFUT_MIN_FRAC = 0.15
REFUT_MAX_VS = 0.10
REFUT_MIN_PICK_VS = 0.5
REFUT_MAX_GAMES = 600

N_WORKERS = 4
OVERSAMPLE = 1.1
# drawn evenly from every ply, ahead of and apart from the rule strata
RANDOM = "random"

MIX = {
    "mate_found": 1000,
    "refutation": 1500,
    "high_cpl": 1500,
    "kl_ce": 1000,
    "high_kl": 1000,
    "high_ce": 1000,
    "regular": 3000,
}

# 75% random; the hard 25% keeps c26k_close's hard proportions plus mates
MIX_RANDOM_HARD = {
    RANDOM: 75000,
    "mate_found": 1500,
    "refutation": 3550,
    "high_cpl": 5725,
    "kl_ce": 4275,
    "high_kl": 4975,
    "high_ce": 4975,
}

PLY_COLS = ["game_id", "move_num", "stm", "played_move", "most_visited_move",
            "best_move", "best_absolute", "played_absolute", "delta",
            "played_best_move", "kl", "ce", "parentQ_white", "parent_p_draw",
            "sims", "stop_reason"]

# logged ply fields copied onto each record; candidate_moves and pv only
# exist on reviewable games
LOG_FIELDS = ["abs_ply", "sel_method", "xc0_move", "cpl", "cpl_depth",
              "best_wdl", "root_wdl_nn", "wdl_target", "z_wdl", "kl_value",
              "sf_value", "pv_len", "gap_block_sims", "gap_block_leader",
              "time", "sims_floor", "sims_ceiling", "sims_min", "sims_max",
              "avg_depth", "max_depth",
              "children_visited", "total_children", "candidate_moves", "pv",
              "pv_played"]

HEADER_FIELDS = ["scenario", "model_epoch", "result", "end_reason", "plies",
                 "vs_stockfish", "stockfish_color", "sf_depth", "start_type",
                 "reviewable", "c_puct", "sims_floor", "sims_ceiling",
                 "dirichlet_alpha", "dirichlet_eps", "fpu_reduction",
                 "contempt_fight_c", "use_syzygy", "allow_resignation"]


def load_plies(run_tag):
    """df_all for one run, from parquet when converted, else the pkl.
    best_cp / played_cp (STM-POV) are rebuilt from the white-POV absolutes,
    since the combined pkl drops them."""
    run_dir = os.path.join(RUNS_DIR, run_tag)
    pq = os.path.join(run_dir, DF_ALL_PARQUET)
    if os.path.exists(pq):
        df = pd.read_parquet(pq, columns=PLY_COLS)
    else:
        with open(os.path.join(run_dir, ANALYZE_PKL), "rb") as f:
            df = pickle.load(f)["df_all"][PLY_COLS]
    df = df.reset_index(drop=True)
    sign = np.where(df.stm, 1, -1)
    df["best_cp"] = (df.best_absolute * sign).clip(-CP_MAX, CP_MAX)
    df["played_cp"] = (df.played_absolute * sign).clip(-CP_MAX, CP_MAX)
    df["run_tag"] = run_tag
    return df


def stratum_masks(df):
    """Rule per df-based stratum, in MIX order. Refutation is mined apart."""
    sign = np.where(df.stm, 1.0, -1.0)
    q_stm = df.parentQ_white * sign
    kl_hi = df.kl >= df.kl.quantile(HIGH_QUANTILE)
    ce_hi = df.ce >= df.ce.quantile(HIGH_QUANTILE)
    mate = (df.best_cp.abs() >= CP_MAX) & (np.sign(df.best_cp) * q_stm < PEGGED_Q)
    return {
        "mate_found": mate,
        "high_cpl": df.delta >= df.delta.quantile(CPL_QUANTILE),
        "kl_ce": kl_hi & ce_hi,
        "high_kl": kl_hi,
        "high_ce": ce_hi,
    }


def ply_keys(game_ids, move_nums):
    return pd.Series(game_ids).astype(str) + ":" + pd.Series(move_nums).astype(str)


def assign_strata(df, taken):
    """Label each ply with its first matching stratum. Plies in taken (mined
    as refutations) are labelled "refutation" and left out of the rest."""
    df["stratum"] = "regular"
    is_taken = ply_keys(df.game_id, df.move_num).isin(taken).values
    open_ = ~is_taken
    for name, mask in stratum_masks(df).items():
        hit = open_ & mask.fillna(False).values
        df.loc[hit, "stratum"] = name
        open_ &= ~hit
    df.loc[is_taken, "stratum"] = "refutation"
    return df


def sample_capped(rows, n, rng):
    """Up to n rows, at most MAX_PER_GAME per game."""
    rows = rows.sample(frac=1.0, random_state=int(rng.integers(2**31)))
    rows = rows[rows.groupby("game_id").cumcount() < MAX_PER_GAME]
    return rows.head(n)


def load_headers(run_tags, game_ids=None, filt=None):
    """Index headers by game_id, streamed; only wanted ones are kept."""
    out = {}
    for run in run_tags:
        with open(os.path.join(RUNS_DIR, run, INDEX_FILE)) as f:
            for line in f:
                if not line.strip():
                    continue
                h = json.loads(line)
                if game_ids is not None and h["game_id"] not in game_ids:
                    continue
                if filt is not None and not filt(h):
                    continue
                out[h["game_id"]] = h
    return out


def load_log_plies(h):
    try:
        with gzip.open(h["pkl_file"], "rb") as f:
            return pickle.load(f)["plies"]
    except (OSError, EOFError, pickle.UnpicklingError, KeyError):
        return None


def position_path(h, move_num):
    """init_fen and the uci line to the position before ply move_num."""
    history = h.get("history_uci") or []
    offset = len(history) - h["plies"]
    return h.get("init_fen") or chess.STARTING_FEN, list(history[:offset + move_num])


def check_position(init_fen, path, stm, played):
    """Replay the line; the side to move and the played move must agree."""
    board = chess.Board(init_fen)
    try:
        for uci in path:
            board.push_uci(uci)
        ok = board.turn == bool(stm) and chess.Move.from_uci(played) in board.legal_moves
    except ValueError:
        return None
    return board.fen() if ok else None


def refutation_row(p):
    """Refuted leader: the top raw prior drew heavy visits, then its
    visit_share collapsed while a different pick took over."""
    cms = p.get("candidate_moves")
    # older logs have no raw prior (P carries noise), so they give none
    if not cms or p.get("sel_method") == "stockfish" or "P_raw" not in cms[0]:
        return None
    sims = sum(c["visits"] for c in cms)
    if sims < REFUT_MIN_SIMS:
        return None
    pick = p.get("xc0_move") or p.get("move_played")
    top = max(cms, key=lambda c: c["P_raw"])
    win = next((c for c in cms if c["uci"] == pick), None)
    if win is None or top["uci"] == pick:
        return None
    if top["visits"] < max(REFUT_MIN_VISITS, REFUT_MIN_FRAC * sims):
        return None
    if top["visit_share"] > REFUT_MAX_VS or win["visit_share"] < REFUT_MIN_PICK_VS:
        return None
    return {"refuted_move": top["uci"], "refuted_N": top["visits"],
            "refuted_P": top["P_raw"]}


def scan_refutations(h):
    """Refutation rows in one game's log, or None when the log is unreadable."""
    plies = load_log_plies(h)
    if plies is None:
        return None
    out = []
    for i, p in enumerate(plies):
        row = refutation_row(p)
        if row is not None:
            out.append({"game_id": h["game_id"], "move_num": i, **row})
    return out


def find_refutations(run_tags, rng, max_games, pool):
    """Scan reviewable logs (non-paired first) for refuted leaders."""
    heads = list(load_headers(run_tags, filt=lambda h: h.get("reviewable")).values())
    rng.shuffle(heads)
    heads.sort(key=lambda h: h["scenario"] == "paired_validation")
    found, scanned = [], 0
    for rows in pool.imap_unordered(scan_refutations, heads[:max_games], chunksize=8):
        if rows is not None:
            scanned += 1
            found += rows
    return pd.DataFrame(found), scanned


def build_record(r, h, p):
    init_fen, path = position_path(h, r["move_num"])
    fen = check_position(init_fen, path, r["stm"], r["played_move"])
    if fen is None:
        return None
    rec = dict(r)
    rec.update({f"log_{k}" if k in rec else k: p.get(k) for k in LOG_FIELDS})
    rec.update({f"game_{k}": h.get(k) for k in HEADER_FIELDS})
    rec.update({"fen": fen, "init_fen": init_fen, "uci_path": path,
                "pkl_file": h.get("pkl_file")})
    # xc0's own deterministic pick: xc0_move is set only when the played
    # move was sampled away from it
    rec["sampled"] = p.get("sel_method") == "sampled"
    rec["xc0_pick"] = p.get("xc0_move") or r["played_move"]
    return rec


def build_game(job):
    """(records, drop counts) for one game's picked rows."""
    h, rows = job
    dropped = Counter()
    if h is None:
        dropped["no_header"] += len(rows)
        return [], dropped
    plies = load_log_plies(h)
    if plies is None:
        dropped["no_log"] += len(rows)
        return [], dropped
    recs = []
    for r in rows:
        p = plies[r["move_num"]] if r["move_num"] < len(plies) else {}
        if not p.get("abs_ply") and p.get("abs_ply") != 0:
            dropped["no_ply"] += 1
            continue
        rec = build_record(r, h, p)
        if rec is None:
            dropped["bad_path"] += 1
            continue
        recs.append(rec)
    return recs, dropped


def build_corpus(name, run_tags, mix=MIX, seed=0, exclude=(), workers=N_WORKERS,
                 refut_max_games=REFUT_MAX_GAMES):
    """exclude: corpus names whose plies (game_id, move_num) are dropped before
    sampling and whose positions (short_fen) are dropped after the build.
    Each stratum is oversampled by OVERSAMPLE, then trimmed to its mix count
    after the replay, exclusion and dedup drops."""
    rng = np.random.default_rng(seed)
    log(f"[corpus] loading plies: {', '.join(run_tags)}")
    df = pd.concat([load_plies(r) for r in run_tags], ignore_index=True)
    df = df[df.stop_reason != "Only 1 legal move"].reset_index(drop=True)
    skip, skip_plies = set(), set()
    for c in exclude:
        prior = pd.read_pickle(os.path.join(CORPUS_DIR, f"{c}.pkl.gz"))
        skip |= set(prior.short_fen)
        skip_plies |= set(ply_keys(prior.game_id, prior.move_num))
    n_plies = len(df)
    df = df[~ply_keys(df.game_id, df.move_num).isin(skip_plies).values]
    df = df.reset_index(drop=True)
    log(f"[corpus] {len(df):,} plies after dropping {n_plies - len(df):,} used plies, "
        f"{len(skip):,} excluded positions")

    with mp.get_context("spawn").Pool(workers) as pool:
        refut, scanned = pd.DataFrame(), 0
        if "refutation" in mix:
            refut, scanned = find_refutations(run_tags, rng, refut_max_games, pool)
            log(f"[corpus] refutations: {len(refut):,} in {scanned:,} games")
        taken = set(ply_keys(refut.game_id, refut.move_num)) if len(refut) else set()
        df = assign_strata(df, taken)
        pools = df.stratum.value_counts().to_dict()
        if len(refut):
            df = df.merge(refut, on=["game_id", "move_num"], how="left")

        picks = []
        if RANDOM in mix:
            rows = sample_capped(df, int(mix[RANDOM] * OVERSAMPLE), rng)
            picks.append(rows.assign(stratum=RANDOM))
            pools[RANDOM] = len(df)
            df = df.drop(rows.index)
        for stratum, n in mix.items():
            if stratum != RANDOM:
                picks.append(sample_capped(df[df.stratum == stratum],
                                           int(n * OVERSAMPLE), rng))
        picks = pd.concat(picks, ignore_index=True)

        heads = load_headers(run_tags, game_ids=set(picks.game_id))
        jobs = [(heads.get(gid), rows.to_dict("records"))
                for gid, rows in picks.groupby("game_id")]
        log(f"[corpus] building {len(picks):,} records from {len(jobs):,} games")
        recs, dropped = [], Counter()
        for i, (r, d) in enumerate(pool.imap_unordered(build_game, jobs, chunksize=16)):
            recs += r
            dropped.update(d)
            if (i + 1) % 5000 == 0:
                log(f"[corpus]   {i + 1:,}/{len(jobs):,} games")

    corpus = pd.DataFrame(recs).sort_values(["game_id", "move_num", "stratum"])
    corpus = corpus.sample(frac=1.0, random_state=seed)
    corpus["short_fen"] = corpus.fen.str.rsplit(" ", n=2).str[0]
    n_built = len(corpus)
    corpus = corpus[~corpus.short_fen.isin(skip)]
    n_excluded = n_built - len(corpus)
    n_before = len(corpus)
    corpus = corpus.drop_duplicates("short_fen")
    n_dups = n_before - len(corpus)
    corpus = corpus[corpus.groupby("stratum").cumcount() < corpus.stratum.map(mix)]
    counts = corpus.stratum.value_counts()
    pop = {s: (np.nan if s == "refutation" else pools.get(s, 0)) for s in counts.index}
    corpus["stratum_pop"] = corpus.stratum.map(pop)
    corpus["weight"] = corpus.stratum_pop / corpus.stratum.map(counts)
    corpus = corpus.reset_index(drop=True)

    os.makedirs(CORPUS_DIR, exist_ok=True)
    corpus.to_pickle(os.path.join(CORPUS_DIR, f"{name}.pkl.gz"))
    meta = {"name": name, "run_tags": run_tags, "seed": seed, "mix": mix,
            "exclude": list(exclude), "oversample": OVERSAMPLE,
            "pools": pools, "refut_games_scanned": scanned,
            "refut_found": len(refut), "dropped": dict(dropped),
            "excluded": n_excluded, "dup_fens": n_dups, "n": len(corpus),
            "counts": counts.to_dict()}
    with open(os.path.join(CORPUS_DIR, f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return corpus, meta


def log(msg):
    print(msg, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--runs", nargs="+",
                    default=[f"m137_8c8t_selfplay{i}" for i in (7, 8, 9, 10)])
    ap.add_argument("--exclude", nargs="*", default=["c26k_close"])
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--refut-games", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    # multiplies every MIX_RANDOM_HARD count
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    mix = {s: round(n * args.scale) for s, n in MIX_RANDOM_HARD.items()}
    _, meta = build_corpus(args.name, args.runs, mix, args.seed,
                           args.exclude, args.workers, args.refut_games)
    log(json.dumps(meta, indent=1, default=str))
