import argparse
import glob
import gzip
import os
import pickle
import random

from pyfastchess import Board as fastboard

from chessbot import SP_DIR
from chessbot.config import Config
from chessbot.game_utils import GameGenerator

from . import paths


def latest_run_tag(prefix="16m_precond_run"):
    candidates = [
        d for d in os.listdir(SP_DIR)
        if d.startswith(prefix) and os.path.isdir(os.path.join(SP_DIR, d))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda d: int(d[len(prefix):]))
    return candidates[-1]


def sample_generator_fixtures(cfg, n, seed=0):
    gen = GameGenerator(cfg)
    rng = random.Random(seed)
    game_types = list(cfg.game_probs.keys())
    out = []
    for _ in range(n):
        gtype = rng.choice(game_types)
        board, meta = gen.new_board(gtype)
        out.append({
            "fen": board.fen(),
            "source": f"generator:{gtype}",
            "ply": board.history_size(),
        })
    return out


def sample_gamelog_fixtures(run_tag, n_games, target_plies, seed=0):
    game_dir = os.path.join(SP_DIR, run_tag, "game_logs")
    pkls = glob.glob(os.path.join(game_dir, "*_log.pkl.gz"))
    if not pkls:
        return []

    rng = random.Random(seed)
    chosen = rng.sample(pkls, min(n_games, len(pkls)))

    out = []
    for path in chosen:
        with gzip.open(path, "rb") as f:
            res = pickle.load(f)

        start_fen = res.get("start_fen")
        history = res.get("history_uci") or res.get("moves_played")
        if not start_fen or not history:
            continue

        game_id = os.path.basename(path).replace("_log.pkl.gz", "")
        for target_ply in target_plies:
            if len(history) < target_ply:
                continue
            board = fastboard(start_fen)
            for mv in history[:target_ply]:
                board.push_uci(mv)
            out.append({
                "fen": board.fen(),
                "source": f"gamelog:{game_id}:ply{target_ply}",
                "ply": board.history_size(),
            })
    return out


def build_fixture_set(cfg=None, n_generator=40, n_from_logs=20, run_tag=None,
                       target_plies=(20, 60, 100), seed=0):
    cfg = cfg or Config()
    fixtures = sample_generator_fixtures(cfg, n_generator, seed=seed)

    run_tag = run_tag or latest_run_tag()
    if run_tag and n_from_logs > 0:
        n_games = max(1, n_from_logs // max(1, len(target_plies)))
        fixtures += sample_gamelog_fixtures(
            run_tag, n_games, target_plies, seed=seed
        )
    return fixtures


def save_fixture_set(fixtures, path):
    paths.save_json(path, fixtures)


def load_fixture_set(path):
    return paths.load_json(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-generator", type=int, default=40)
    ap.add_argument("--n-from-logs", type=int, default=20)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="default")
    args = ap.parse_args()

    fixtures = build_fixture_set(
        n_generator=args.n_generator,
        n_from_logs=args.n_from_logs,
        run_tag=args.run_tag,
        seed=args.seed,
    )
    out_path = paths.FIXTURES_DIR / f"{args.name}.json"
    save_fixture_set(fixtures, out_path)
    print(f"wrote {len(fixtures)} fixtures to {out_path}")


if __name__ == "__main__":
    main()
