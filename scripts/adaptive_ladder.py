import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from chessbot import CUTECHESS_CLI_LOC, LC0_LOC, SF_LOC, ENDGAME_LOC

CC_PATH = CUTECHESS_CLI_LOC
SF_PATH = SF_LOC
LC0_PATH = LC0_LOC
OPENING_BOOK = r"C:\Users\Bryan\Data\chessbot_data\cutechess\opening_book.pgn"

OUT_DIR = Path(r"C:\Users\Bryan\Data\chessbot_data\ratings_ladder")
RUNS_DIR = OUT_DIR / "runs"
ALL_GAMES_PATH = OUT_DIR / "all_games.pgn"
PAIRINGS_PATH = OUT_DIR / "pairings.jsonl"
STATE_PATH = OUT_DIR / "state.json"
RATINGS_PATH = OUT_DIR / "ratings.txt"
RATINGS_CANDIDATES_PATH = OUT_DIR / "ratings_candidates.txt"
RATINGS_LADDERED_PATH = OUT_DIR / "ratings_laddered.txt"

ORDO_PATH = r"C:\Users\Bryan\Data\chessbot_data\cutechess\ordo-1.2.6-win\ordo-win64.exe"
ORDO_ANCHOR = "SF_VLTC"
ORDO_ANCHOR_RATING = 3776

OPENING_FORMAT = "pgn"
LC0_BACKEND = "cuda-fp16"
LC0_BACKEND_OPTIONS = "gpu=0"
LC0_WEIGHTS = r"c:\Users\Bryan\Data\chessbot_data\lc0\models\t1-256x10-distilled-swa-2432500.pb.gz"
LC0_MINIBATCH_SIZE = "128"
LC0_MAX_PREFETCH = "37"
LC0_CPUCT = "1.589"
LC0_CPUCT_FACTOR = "3.973"
LC0_CPUCT_BASE = "45669"
LC0_FPU_VALUE = "0.2643"
LC0_POLICY_TEMPERATURE = "1.179"
LC0_SMART_PRUNING_FACTOR = "2.0"
LC0_SMART_PRUNING_MIN_BATCHES = "100"

STC = "10+0.1"
MTC = "30+0.3"
LTC = "60+0.6"
VLTC = "120+1"
VVLTC = "240+2"
TIMEMARGIN = 500

TC_NAMES = {"STC": STC, "MTC": MTC, "LTC": LTC, "VLTC": VLTC, "VVLTC": VVLTC}
OPPONENT_DEFAULT_TC = STC

A = 0.3
B = 0.7

GAMES_PER_PAIRING = 8
CONCURRENCY = 2
MAX_MATCHES = 5

USE_TB_ADJUDICATION = True
SYZYGY_PATH = ENDGAME_LOC
SYZYGY_PIECES = "5"

SF_THREADS = "1"
SF_HASH = "128"
MOVE_OVERHEAD_MS = "0"

CANDIDATES1 = [
    "SF_d10",
    "Lc0_n200",
    "Lc0_n400",
    "SF_t50",
    "Lc0_n800",
    "SF_t100",
    "Lc0_n1600",
    "SF_t150",
    "Lc0_MTC",
    "SF_t250",
    "SF_STC",
    "SF_MTC",
    "Lc0_VLTC",
    "SF_LTC",
    "SF_VLTC"
]

LADDER_RANKS = [
    "SF_d8",
    "SF_d9",
    "SF_d10",
    "SF_d11",
    "SF_d12",
    "SF_d13",
    "SF_d14",
    "SF_d15",
    "SF_d16",
    "SF_d17",
    "SF_d18",
    "SF_STC",
    "SF_d20",
    "Lc0_VLTC",
    "SF_MTC",
    "SF_d23",
    "SF_LTC",
    "SF_VLTC"
]

START_RANK_NAME = "SF_d9"
START_RANK = LADDER_RANKS.index(START_RANK_NAME)
MIN_RANK = 0
MAX_RANK = len(LADDER_RANKS) - 1

TERM_LABELS = {
    "Draw (repetition)":          "Draw rep",
    "Draw (insufficient material)": "Draw insuff",
    "Draw (adjudication)":        "Draw adj",
    "Resign (adjudication)":      "Resign adj",
    "Draw (50-move)":             "Draw 50",
    "Loss on Time":               "Flag",
}


def parse_candidate_config(name):
    if name.startswith("Lc0_"):
        kind, suffix = "lc0", name[4:]
    elif name.startswith("SF_"):
        kind, suffix = "sf", name[3:]
    else:
        raise ValueError(f"Unknown candidate prefix: {name}")

    if suffix in TC_NAMES:
        return {"kind": kind, "tc": TC_NAMES[suffix]}
    elif suffix.startswith("t"):
        return {"kind": kind, "limit": "st",
                "value": str(int(suffix[1:]) / 1000)}
    elif suffix.startswith("n"):
        return {"kind": kind, "limit": "nodes", "value": suffix[1:]}
    elif suffix.startswith("d"):
        return {"kind": kind, "limit": "depth", "value": suffix[1:]}
    else:
        raise ValueError(f"Unknown candidate suffix: {suffix}")


def get_candidate_tc(candidate):
    config = parse_candidate_config(candidate)
    return config.get("tc", VLTC)


def get_opponent_tc(rank):
    opponent = LADDER_RANKS[rank]
    if opponent == "SF_d30":
        return VVLTC
    config = parse_candidate_config(opponent)
    if "tc" in config:
        return config["tc"]
    if config.get("limit") == "depth":
        d = int(config["value"])
        if d < 16:
            return STC
        if d < 21:
            return LTC
        if d < 26:
            return VLTC
        return VVLTC
    return STC


def main():
    make_dirs()
    state = load_state()

    for candidate in CANDIDATES1:
        if candidate in state["completed_candidates"]:
            continue
        run_candidate(candidate, state)
        state["completed_candidates"].append(candidate)
        save_state(state)
        run_ordo()

    print("Done.")


def run_ordo():
    if not ALL_GAMES_PATH.exists():
        return
    cmd = [
        ORDO_PATH, "-q",
        "-a", str(ORDO_ANCHOR_RATING),
        "-A", ORDO_ANCHOR,
        "-W", "-s", "100",
        "-p", str(ALL_GAMES_PATH),
        "-o", str(RATINGS_PATH),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"ordo failed: {proc.stderr.strip()}")
        return
    split_ratings()


def split_ratings():
    if not RATINGS_PATH.exists():
        return
    candidates_set = set(CANDIDATES1)
    ladder_set = set(LADDER_RANKS)
    lines = RATINGS_PATH.read_text(encoding="utf-8").splitlines()

    header, data, footer = [], [], []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header.append(line)
        elif stripped[0].isdigit():
            data.append(line)
        else:
            footer.append(line)

    def write_filtered(path, player_set):
        rows = [ln for ln in data if ln.split()[1] in player_set]
        out = header[:]
        for i, ln in enumerate(rows, 1):
            parts = ln.split(None, 1)
            out.append(f"{i:4d} {parts[1]}")
        out.extend(footer)
        path.write_text("\n".join(out) + "\n", encoding="utf-8")

    write_filtered(RATINGS_CANDIDATES_PATH, candidates_set)
    write_filtered(RATINGS_LADDERED_PATH, ladder_set)


def make_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)


def load_state():
    if STATE_PATH.exists():
        with STATE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "completed_candidates": [],
        "next_start_rank": START_RANK,
        "completed_pairings": {},
        "candidates": {},
    }


def save_state(state):
    with STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def run_candidate(candidate, state):
    start_rank = clamp_rank(state["next_start_rank"])
    candidate_state = state["candidates"].setdefault(candidate, {})
    candidate_state.setdefault("results", {})

    needed = {
        r for r in [start_rank]
        if not is_skipped(candidate, LADDER_RANKS[r])
    }
    if not needed:
        needed = {clamp_rank(start_rank - 2)}
    seen = set()
    highest = 0
    matches_run = 0

    print("")
    print(f"Candidate: {candidate}")
    print(f"Start rank: {LADDER_RANKS[start_rank]}")

    while True:
        todo = needed - seen
        if not todo or matches_run >= MAX_MATCHES:
            break

        rank = choose_next_rank(todo, start_rank)
        seen.add(rank)
        opponent = LADDER_RANKS[rank]
        pairing_id = make_pairing_id(candidate, rank)

        if pairing_id in state["completed_pairings"]:
            result = state["completed_pairings"][pairing_id]
        else:
            result = run_pairing(candidate, rank)
            state["completed_pairings"][pairing_id] = result
            candidate_state["results"][str(rank)] = result
            append_pairing_record(result)
            append_all_games(result["pgn_path"])
            save_state(state)
            matches_run += 1

        highest = max(highest, rank)
        print_result(candidate, opponent, result)
        print()
        add_needed_ranks(needed, seen, rank, result, candidate)

    print_candidate_summary(candidate, candidate_state["results"])

    winning_ranks = [
        int(r) for r, res in candidate_state["results"].items()
        if res["score"] >= 0.5
    ]
    if winning_ranks:
        next_rank = clamp_rank(max(winning_ranks))
    else:
        next_rank = START_RANK

    state["next_start_rank"] = next_rank
    msg = f" Finished {candidate}. Next start rank: {LADDER_RANKS[next_rank]} "
    print(msg.center(80, "="))
    save_state(state)


def choose_next_rank(todo, start_rank):
    ranks = list(todo)
    ranks.sort(key=lambda r: (abs(r - start_rank), r))
    return ranks[0]


def is_skipped(candidate, opponent):
    return candidate == opponent


def add_needed_ranks(needed, seen, rank, result, candidate):
    score = result["score"]
    wins = result["wins"]
    losses = result["losses"]
    games = result["games"]

    if wins >= games - 1 or score > B:
        add_needed_rank(needed, seen, rank + 1, candidate)
    elif losses >= games - 1 or wins <= 1 or score < A:
        add_needed_rank(needed, seen, rank - 1, candidate)
    else:
        add_needed_rank(needed, seen, rank + 1, candidate)
        if score < 0.5:
            add_needed_rank(needed, seen, rank - 1, candidate)


def add_needed_rank(needed, seen, rank, candidate):
    if rank < MIN_RANK or rank > MAX_RANK:
        return
    if rank in seen:
        return
    if is_skipped(candidate, LADDER_RANKS[rank]):
        return
    needed.add(rank)


def clamp_rank(rank):
    return max(MIN_RANK, min(MAX_RANK, rank))


def run_pairing(candidate, rank):
    opponent = LADDER_RANKS[rank]
    pairing_id = make_pairing_id(candidate, rank)
    run_dir = RUNS_DIR / pairing_id
    run_dir.mkdir(exist_ok=True)

    pgn_path = run_dir / "games.pgn"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    command_path = run_dir / "command.txt"
    result_path = run_dir / "result.json"

    cmd = build_command(candidate, rank, pgn_path)

    games_before = count_pgn_games(pgn_path)
    start_time = now_text()
    t0 = time.time()
    write_command(command_path, cmd)

    proc = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - t0
    end_time = now_text()

    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    pgn_data = analyze_pgn_games(pgn_path, candidate, skip=games_before)
    wins = pgn_data["wins"]
    draws = pgn_data["draws"]
    losses = pgn_data["losses"]
    games = wins + draws + losses
    score = (wins + 0.5 * draws) / games if games else 0

    candidate_stats = pgn_data["player_stats"].get(candidate, empty_stats())
    opponent_stats = pgn_data["player_stats"].get(opponent, empty_stats())

    result = {
        "pairing_id": pairing_id,
        "candidate": candidate,
        "opponent": opponent,
        "rank": rank,
        "candidate_config": parse_candidate_config(candidate),
        "opponent_config": parse_candidate_config(opponent),
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "terminations": pgn_data["terminations"],
        "avg_length_win": pgn_data["avg_length_win"],
        "avg_length_draw": pgn_data["avg_length_draw"],
        "avg_length_loss": pgn_data["avg_length_loss"],
        "candidate_stats": candidate_stats,
        "opponent_stats": opponent_stats,
        "pgn_path": str(pgn_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": cmd,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": round(elapsed, 1),
        "return_code": proc.returncode,
    }

    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def get_concurrency(candidate):
    return CONCURRENCY


def build_command(candidate, rank, pgn_path):
    opponent = LADDER_RANKS[rank]
    c_tc = get_candidate_tc(candidate)
    o_tc = get_opponent_tc(rank)

    cmd = [CC_PATH]
    cmd += build_engine(
        candidate, parse_candidate_config(candidate), c_tc, TIMEMARGIN
    )
    cmd += build_engine(
        opponent, parse_candidate_config(opponent), o_tc, TIMEMARGIN
    )

    cmd += [
        "-resign", "movecount=7", "score=800",
        "-draw", "movenumber=40", "movecount=8", "score=8",
        "-games", str(GAMES_PER_PAIRING),
        "-repeat",
        "-concurrency", str(get_concurrency(candidate)),
        "-openings",
        f"file={OPENING_BOOK}",
        f"format={OPENING_FORMAT}",
        "order=random",
    ]
    if USE_TB_ADJUDICATION:
        cmd += ["-tb", str(SYZYGY_PATH), "-tbpieces", SYZYGY_PIECES]
    cmd += [
        "-pgnout", str(pgn_path),
        "-ratinginterval", "1",
    ]

    return cmd


def build_engine(name, config, tc, margin):
    kind = config["kind"]
    cmd_path = LC0_PATH if kind == "lc0" else SF_PATH

    args = ["-engine", f"name={name}", f"cmd={cmd_path}", "proto=uci"]

    if "limit" in config:
        args.append(f"{config['limit']}={config['value']}")

    args.append(f"tc={tc}")
    args.append(f"timemargin={margin}")

    if kind == "lc0":
        args.append(f"option.Backend={LC0_BACKEND}")
        args.append(f"option.BackendOptions={LC0_BACKEND_OPTIONS}")
        args.append(f"option.WeightsFile={LC0_WEIGHTS}")
        args.append(f"option.MoveOverheadMs={MOVE_OVERHEAD_MS}")
        args.append(f"option.MinibatchSize={LC0_MINIBATCH_SIZE}")
        args.append(f"option.MaxPrefetch={LC0_MAX_PREFETCH}")
        args.append(f"option.CPuct={LC0_CPUCT}")
        args.append(f"option.CPuctFactor={LC0_CPUCT_FACTOR}")
        args.append(f"option.CPuctBase={LC0_CPUCT_BASE}")
        args.append(f"option.FpuValue={LC0_FPU_VALUE}")
        args.append(f"option.PolicyTemperature={LC0_POLICY_TEMPERATURE}")
        args.append(f"option.SmartPruningFactor={LC0_SMART_PRUNING_FACTOR}")
        args.append(
            f"option.SmartPruningMinimumBatches={LC0_SMART_PRUNING_MIN_BATCHES}"
        )
    else:
        args += [
            f"option.Threads={SF_THREADS}",
            f"option.Hash={SF_HASH}",
        ]

    return args


def empty_stats():
    return {
        "avg_depth": None,
        "avg_time_ms": None,
        "moves_sampled": 0,
    }


def parse_pgn_move_stats(game, white_name, black_name):
    lines = game.splitlines()
    in_header = True
    movetext_lines = []
    for line in lines:
        if in_header and line.strip() == "":
            in_header = False
            continue
        if not in_header:
            movetext_lines.append(line)
    movetext = " ".join(movetext_lines)

    comments = re.findall(r'\{([^}]+)\}', movetext)
    comment_pat = re.compile(
        r'^[+-]?\d*\.?\d+/(\d+)\s+([\d.]+)s$'
    )

    white_depths, white_times = [], []
    black_depths, black_times = [], []

    for i, comment in enumerate(comments):
        m = comment_pat.match(comment.strip())
        if not m:
            continue
        depth = int(m.group(1))
        t_ms = round(float(m.group(2)) * 1000, 1)
        if i % 2 == 0:
            white_depths.append(depth)
            white_times.append(t_ms)
        else:
            black_depths.append(depth)
            black_times.append(t_ms)

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else None

    return {
        white_name: {
            "avg_depth": avg(white_depths),
            "avg_time_ms": avg(white_times),
            "moves_sampled": len(white_depths),
        },
        black_name: {
            "avg_depth": avg(black_depths),
            "avg_time_ms": avg(black_times),
            "moves_sampled": len(black_depths),
        },
    }


def compact_num(x):
    if x is None:
        return "-"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x / 1_000:.1f}k"
    return f"{x:.0f}"


def fmt_stats(s):
    if not s or s.get("moves_sampled", 0) == 0:
        return None
    parts = []
    if s.get("avg_depth") is not None:
        parts.append(f"d={s['avg_depth']}")
    if s.get("avg_time_ms") is not None:
        parts.append(f"time={s['avg_time_ms']:.0f}ms")
    return " ".join(parts) if parts else None


def shorten_term(label):
    return TERM_LABELS.get(label, label)


def print_result(candidate, opponent, result):
    score = result["score"]
    wins = result["wins"]
    draws = result["draws"]
    losses = result["losses"]
    elapsed = result.get("elapsed_seconds", 0)
    games = wins + draws + losses
    games_per_hr = (games / elapsed * 3600) if elapsed > 0 else 0
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(
        f"{candidate} vs {opponent}: "
        f"{wins}-{draws}-{losses}, score={score:.3f} | "
        f"{mins}m{secs:02d}s, {games_per_hr:.0f} games/hr"
    )
    cs = fmt_stats(result.get("candidate_stats"))
    os_ = fmt_stats(result.get("opponent_stats"))
    if cs:
        print(f"  cand: {cs}")
    if os_:
        print(f"  opp : {os_}")


def print_candidate_summary(candidate, results):
    if not results:
        return
    print(f"\n{candidate} summary")

    all_terms = {}
    engine_depths = {}
    engine_times = {}
    opponent_wdl = {}
    total_wins = total_draws = total_losses = 0

    for rank_str in sorted(results, key=lambda x: int(x)):
        r = results[rank_str]
        rank = int(rank_str)
        opponent = LADDER_RANKS[rank]

        for k, v in r.get("terminations", {}).items():
            all_terms[k] = all_terms.get(k, 0) + v

        w, d, l = r["wins"], r["draws"], r["losses"]
        total_wins += w
        total_draws += d
        total_losses += l
        opponent_wdl[opponent] = (w, d, l)

        for name, stats_key in [
            (candidate, "candidate_stats"),
            (opponent, "opponent_stats"),
        ]:
            s = r.get(stats_key, {})
            if not s or s.get("moves_sampled", 0) == 0:
                continue
            if name not in engine_depths:
                engine_depths[name] = []
                engine_times[name] = []
            if s.get("avg_depth") is not None:
                engine_depths[name].append(s["avg_depth"])
            if s.get("avg_time_ms") is not None:
                engine_times[name].append(s["avg_time_ms"])

    if all_terms:
        term_parts = "  ".join(
            f"{shorten_term(k)}:{v}" for k, v in sorted(all_terms.items())
        )
        print(f"Terms: {term_parts}")

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else None

    all_engines = [candidate] + [
        LADDER_RANKS[int(r)] for r in sorted(results, key=lambda x: int(x))
    ]
    seen = set()
    rows = []
    for name in all_engines:
        if name in seen:
            continue
        seen.add(name)
        depths = engine_depths.get(name, [])
        times = engine_times.get(name, [])
        d = avg(depths)
        t = avg(times)
        wdl = (total_wins, total_draws, total_losses) if name == candidate else opponent_wdl.get(name)
        if d is not None or t is not None:
            rows.append((name, d, t, wdl))

    if rows:
        col1 = max(len(r[0]) for r in rows) + 1
        t_width = max(
            len(f"time={r[2]:.0f}ms") if r[2] is not None else 0
            for r in rows
        )
        for name, d, t, wdl in rows:
            label = f"{name}:".ljust(col1)
            d_str = f"d={d}".ljust(10) if d is not None else " " * 10
            t_str = f"time={t:.0f}ms".ljust(t_width) if t is not None else " " * t_width
            wdl_str = f"  WDL={wdl[0]:2d}/{wdl[1]:2d}/{wdl[2]:2d}" if wdl else ""
            print(f"  {label}  {d_str}  {t_str}{wdl_str}")
    print()


def count_pgn_games(pgn_path):
    return len(split_pgn_games(pgn_path))


def parse_termination(game, result):
    comments = re.findall(r'\{([^}]+)\}', game)
    if not comments:
        return "Unknown"
    reason = comments[-1].strip()
    low = reason.lower()
    if "mates" in low:
        return "Checkmate"
    if "resigns" in low or "resign" in low:
        return "Resign"
    if "loses on time" in low or "loss on time" in low:
        return "Loss on Time"
    if result == "1/2-1/2":
        if "repetition" in low:
            return "Draw (repetition)"
        if "50-move" in low or "fifty" in low:
            return "Draw (50-move)"
        if "insufficient" in low:
            return "Draw (insufficient material)"
        if "stalemate" in low:
            return "Draw (stalemate)"
        if "adjudication" in low:
            return "Draw (adjudication)"
        return f"Draw ({reason})"
    if "adjudication" in low:
        return "Resign (adjudication)"
    return reason


def parse_game_length(game):
    lines = game.splitlines()
    movetext = " ".join(l for l in lines if not l.startswith("["))
    numbers = re.findall(r"(\d+)\.", movetext)
    return int(numbers[-1]) if numbers else 0


def analyze_pgn_games(pgn_path, candidate, skip=0):
    games = split_pgn_games(pgn_path)[skip:]
    wins = draws = losses = 0
    win_lengths = []
    draw_lengths = []
    loss_lengths = []
    terminations = {}
    player_depths = {}
    player_times = {}

    for game in games:
        tags = parse_tags(game)
        result = tags.get("Result")
        white = tags.get("White")
        black = tags.get("Black")

        if result not in ["1-0", "0-1", "1/2-1/2"]:
            continue
        if candidate not in [white, black]:
            continue

        if result == "1/2-1/2":
            outcome = "draw"
            draws += 1
        elif (
            (result == "1-0" and white == candidate) or
            (result == "0-1" and black == candidate)
        ):
            outcome = "win"
            wins += 1
        else:
            outcome = "loss"
            losses += 1

        length = parse_game_length(game)
        if outcome == "win":
            win_lengths.append(length)
        elif outcome == "draw":
            draw_lengths.append(length)
        else:
            loss_lengths.append(length)

        reason = parse_termination(game, result)
        terminations[reason] = terminations.get(reason, 0) + 1

        move_stats = parse_pgn_move_stats(game, white, black)
        for player, s in move_stats.items():
            if s["moves_sampled"] == 0:
                continue
            if player not in player_depths:
                player_depths[player] = []
                player_times[player] = []
            player_depths[player].append(s["avg_depth"])
            player_times[player].append(s["avg_time_ms"])

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else None

    def agg_stats(player):
        depths = [d for d in player_depths.get(player, []) if d]
        times = [t for t in player_times.get(player, []) if t]
        return {
            "avg_depth": avg(depths),
            "avg_time_ms": avg(times),
            "moves_sampled": len(depths),
        }

    all_players = set(player_depths) | set(player_times)
    player_stats = {p: agg_stats(p) for p in all_players}

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "terminations": terminations,
        "avg_length_win": avg(win_lengths),
        "avg_length_draw": avg(draw_lengths),
        "avg_length_loss": avg(loss_lengths),
        "player_stats": player_stats,
    }


def split_pgn_games(pgn_path):
    if not pgn_path.exists():
        return []

    text = pgn_path.read_text(encoding="utf-8", errors="replace")
    chunks = []
    current = []

    for line in text.splitlines():
        if line.startswith("[Event ") and current:
            chunks.append("\n".join(current))
            current = []
        current.append(line)

    if current:
        chunks.append("\n".join(current))

    return chunks


def parse_tags(game):
    tags = {}
    for line in game.splitlines():
        if not line.startswith("["):
            continue
        end = line.find(" ")
        if end < 0:
            continue
        key = line[1:end]
        value_start = line.find('"')
        value_end = line.rfind('"')
        if value_start < 0 or value_end <= value_start:
            continue
        tags[key] = line[value_start + 1:value_end]
    return tags


def append_pairing_record(result):
    with PAIRINGS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")


def append_all_games(pgn_path):
    pgn_path = Path(pgn_path)
    if not pgn_path.exists():
        return
    with ALL_GAMES_PATH.open("a", encoding="utf-8") as out_handle:
        with pgn_path.open(
            "r", encoding="utf-8", errors="replace"
        ) as in_handle:
            shutil.copyfileobj(in_handle, out_handle)
        out_handle.write("\n\n")


def make_pairing_id(candidate, rank):
    return f"{candidate}_vs_{LADDER_RANKS[rank]}"


def write_command(path, cmd):
    text = " ".join([quote_arg(arg) for arg in cmd])
    path.write_text(text + "\n", encoding="utf-8")


def quote_arg(arg):
    if " " in arg:
        return f'"{arg}"'
    return arg


def now_text():
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
