"""Position-reconstruction and game-length invariants for blunder replay.

Two things lc0_BRP is load-bearing on: that replaying a pool record's
uci_path from STARTPOS lands on exactly the position the record was scored
at, and that max_game_length = N yields exactly N + 1 moves.
"""
import os
import random

import chess
import pytest

from pyfastchess import Board as fastboard

from chessbot.config import Config
from chessbot.game_utils import reconcile_game_boards, short_fen
from chessbot.mcts_utils import ChessGame

POOL_DIR = r"C:\Users\Bryan\Data\chessbot_data"
POOLS = {
    "selfplay": os.path.join(POOL_DIR, "blunder_positions_selfplay.pkl.gz"),
    "raw": os.path.join(POOL_DIR, "blunder_positions_raw.pkl.gz"),
}
PER_SCENARIO = 40


def load_pool(path):
    import gzip
    import pickle
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def sample_records(path):
    """Up to PER_SCENARIO records per scenario, deterministic."""
    if not os.path.exists(path):
        return []
    pool = load_pool(path)
    by_scenario = {}
    for key, rec in pool.items():
        by_scenario.setdefault(rec.get("scenario"), []).append((key, rec))
    out = []
    rng = random.Random(0)
    for scenario in sorted(by_scenario, key=lambda s: str(s)):
        group = by_scenario[scenario]
        group.sort(key=lambda kv: kv[0])
        if len(group) > PER_SCENARIO:
            group = rng.sample(group, PER_SCENARIO)
        out.extend(group)
    return out


def pool_cases():
    cases = []
    for name, path in POOLS.items():
        for key, rec in sample_records(path):
            cases.append(pytest.param(
                key, rec, id=f"{name}-{rec.get('scenario')}-{key[:24]}"))
    return cases


CASES = pool_cases()

if not CASES:
    pytest.skip("no blunder pools on this machine", allow_module_level=True)


@pytest.mark.parametrize("key,rec", CASES)
def test_uci_path_reaches_the_scored_position(key, rec):
    """STARTPOS + uci_path must land on the record's own fen and key.

    If it does not, the replay probes a different position than the one SF
    scored, and every probe_cpl in the pool is measured against the wrong
    board.
    """
    b = fastboard()
    for uci in rec["uci_path"]:
        assert uci in b.legal_moves(), f"illegal {uci} replaying {key}"
        b.push_uci(uci)

    assert b.fen() == rec["fen"]
    assert short_fen(b.fen()) == key


@pytest.mark.parametrize("key,rec", CASES)
def test_both_reconstruction_paths_agree(key, rec, capsys):
    """start_fen alone and startpos + history must reach the same board.

    reconcile_game_boards degrades to start_fen-only on a mismatch and only
    prints, so a silent divergence would cost the lc0 history planes without
    failing anything.
    """
    board_ch, b_fast = reconcile_game_boards(
        start_fen=rec["fen"],
        history_uci=list(rec["uci_path"]),
        moves_played=[],
    )

    assert "falling back" not in capsys.readouterr().out
    assert b_fast.fen() == rec["fen"]
    assert board_ch.fen() == rec["fen"]
    assert board_ch.fen() == chess.Board(rec["fen"]).fen()
    assert b_fast.history_size() == len(rec["uci_path"])


@pytest.mark.parametrize("max_game_length", [0, 1, 2, 3])
def test_max_game_length_plays_one_move_too_many(max_game_length):
    """check_for_terminal tests plies > max_game_length, so N gives N + 1.

    lc0_BRP wants exactly 3 plies and therefore sets 2. Only a comment
    documents that off-by-one today.
    """
    cfg = Config()
    cfg.max_game_length = max_game_length
    cfg.add_root_noise = False
    cfg.sims_ceiling = 8

    meta = {"vs_stockfish": False, "stockfish_is_white": False}
    game = ChessGame(fastboard(), meta, cfg)

    played = 0
    for _ in range(max_game_length + 5):
        mv = sorted(game.board.legal_moves())[0]
        terminal = game.push_move(mv, "test", mv)
        played += 1
        if terminal:
            break

    assert terminal
    assert game.end_reason == "max_length"
    assert played == max_game_length + 1
    assert game.plies == max_game_length + 1
