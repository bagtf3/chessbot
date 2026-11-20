"""BRP target invariants and STM-POV correctness for the training targets.

The BRP tests pin structure (sums, coverage). The POV tests exercise all four
combinations of side-to-move and who is winning, by mirroring: a position and
its colour-mirror are the same position to a STM-relative encoder, so every
target the rescorer builds for one must come out bit-identical to the other.
"""
from collections import defaultdict

import chess
import numpy as np
import pytest

from pyfastchess import Board as fastboard

from chessbot import BLACK_WINNING_WHITE_MOVE, WHITE_WINNING_WHITE_MOVE
from chessbot.config import Config
from chessbot.replay_buffer import densify_policy
from chessbot.rescore import Rescorer, blend_wdl, blend_wdl_stm, z_to_wdl


def test_native_stm_wdl_winning_mover():
    # A winning Black mover has the same STM target as a winning White mover.
    target = blend_wdl_stm(1, [0.8, 0.1, 0.1])
    np.testing.assert_allclose(target, [0.9, 0.05, 0.05])
    assert target.dtype == np.float32


def test_native_stm_wdl_missing_search():
    np.testing.assert_array_equal(blend_wdl_stm(-1, None), [0.0, 0.0, 1.0])

# both are white-to-move; mirroring each covers the black-to-move half, so the
# two of them span all four (side-to-move x who-is-winning) combinations
STM_WINNING_FEN = WHITE_WINNING_WHITE_MOVE.fen()
STM_LOSING_FEN = BLACK_WINNING_WHITE_MOVE.fen()

# carrier position for the blend-shape tests; the eval is injected, not read
BLEND_FEN = STM_WINNING_FEN


class FakeOpeningCounts:
    def key(self, board, x):
        return None

    def bump(self, key):
        pass


class FakeBuffer:
    def __init__(self):
        self.entries = []

    def append(self, record, key=None):
        self.entries.append((record, key))


def make_rescorer():
    """A Rescorer carrying only what the target-building methods touch."""
    r = Rescorer.__new__(Rescorer)
    r.config = Config()
    r.opening_counts = FakeOpeningCounts()
    r.live_buffer = FakeBuffer()
    r.kl_q50 = 0.0
    r.kl_q80 = 0.0
    r.kl_value_q80 = 0.0
    r.sample_counts = defaultdict(int)
    r.sample_counts_window = defaultdict(int)
    return r


def mirror_uci(uci):
    m = chess.Move.from_uci(uci)
    return chess.Move(chess.square_mirror(m.from_square),
                      chess.square_mirror(m.to_square), m.promotion).uci()


VISITS = (300, 120, 40, 10)


def make_ply(fen, ucis=None, visits=VISITS, best_wdl=None):
    """A ply_state shaped the way finalize_game leaves it for the BRP path.

    ucis is explicit for the mirror tests: sorting each board's own legal
    moves would hand the two boards non-corresponding candidate sets.
    """
    board = fastboard(fen)
    if ucis is None:
        ucis = sorted(board.legal_moves())[:len(visits)]
    cm = [
        {'uci': u, 'visits': int(v), 'P': float(v) / sum(visits)}
        for u, v in zip(ucis, visits)
    ]
    return {
        'board': board,
        'turn': board.white_to_move(),
        'tr': {'candidate_moves': cm, 'best_wdl': best_wdl},
    }


def sole_entry(r):
    (record, _), = r.live_buffer.entries
    x, _, policy, Y, vwht, pwht, source = record
    return x, densify_policy(policy), np.asarray(Y, dtype=np.float32), source


def run_blend(fen, best_wdl, ucis=None):
    r = make_rescorer()
    ply = make_ply(fen, ucis=ucis, best_wdl=best_wdl)
    ok = r.add_blunder_replay_training_example(ply, tsl=0.0)
    assert ok
    return ply['board'], sole_entry(r)


def test_blend_policy_and_wdl_are_distributions():
    board, (x, policy, Y, source) = run_blend(
        BLEND_FEN, best_wdl=[0.4, 0.4, 0.2])

    assert source == 'xc0'
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert Y.sum() == pytest.approx(1.0, abs=1e-5)
    assert (policy >= 0).all()
    assert (Y >= 0).all()


def test_every_legal_move_carries_mass():
    """Every legal move gets a 1-visit floor, so none may end at zero."""
    fen = BLEND_FEN
    board, (x, policy, Y, source) = run_blend(fen, best_wdl=[0.4, 0.4, 0.2])

    legal = sorted(fastboard(fen).legal_moves())
    idx = fastboard(fen).moves_to_indices(legal)
    assert len(legal) == len(set(idx))
    assert (policy[list(idx)] > 0).all()
    assert np.count_nonzero(policy) == len(legal)


def test_value_target_is_the_search_wdl_unblended():
    """The value target is the search's own WDL."""
    best_wdl = [0.1, 0.3, 0.6]
    # BLEND_FEN is white to move, so this white-POV wdl is already STM-POV
    expected = np.array(best_wdl, dtype=np.float32)
    board, (x, policy, Y, source) = run_blend(BLEND_FEN, best_wdl=best_wdl)
    np.testing.assert_allclose(Y, expected, atol=1e-6)


POV_CASES = [(STM_WINNING_FEN, True), (STM_LOSING_FEN, False)]


def mirrored_case(fen, stm_winning):
    """(white-to-move, mirror) pair for one 'is STM winning' setting.

    best_wdl is white-POV, best_cp is STM-POV. Mirroring flips who is to move,
    so white-POV flips while STM-POV does not -- which is exactly the axis
    every POV bug lives on. Moves are mirrored, never re-derived, so both
    sides carry the same candidate set and the same SF best.
    """
    stm_wdl = [0.85, 0.10, 0.05] if stm_winning else [0.05, 0.10, 0.85]
    best_cp = 600 if stm_winning else -600

    board_w = chess.Board(fen)
    assert board_w.turn is chess.WHITE
    board_b = board_w.mirror()

    white_fen, black_fen = board_w.fen(), board_b.fen()
    ucis_w = sorted(fastboard(white_fen).legal_moves())
    cands_w = ucis_w[:len(VISITS)]
    best_w = ucis_w[-1]

    # white to move: white-POV == STM-POV. black to move: flip it.
    return [
        (white_fen, list(stm_wdl), best_cp, cands_w, best_w),
        (black_fen, list(reversed(stm_wdl)), best_cp,
         [mirror_uci(u) for u in cands_w], mirror_uci(best_w)),
    ]


@pytest.mark.parametrize("fen,stm_winning", POV_CASES)
def test_blend_targets_are_stm_pov(fen, stm_winning):
    """All four combos: the mirrored pair must produce identical targets, and
    the WDL must read from the mover's side."""
    results = []
    for pos_fen, wdl_white_pov, best_cp, cands, best_uci in \
            mirrored_case(fen, stm_winning):
        board, (x, policy, Y, source) = run_blend(
            pos_fen, best_wdl=wdl_white_pov, ucis=cands)
        results.append((np.asarray(x), policy, Y))

    (x_w, pol_w, Y_w), (x_b, pol_b, Y_b) = results

    assert np.array_equal(x_w, x_b)
    np.testing.assert_allclose(pol_w, pol_b, atol=1e-6)
    np.testing.assert_allclose(Y_w, Y_b, atol=1e-6)

    if stm_winning:
        assert Y_w[0] > Y_w[2]
    else:
        assert Y_w[2] > Y_w[0]


@pytest.mark.parametrize("stm_winning", [True, False])
def test_blend_wdl_is_stm_pov(stm_winning):
    """The normal selfplay target. z_stm is already STM-POV; best_wdl is not.

    Both members of a mirrored pair see the same z_stm and the same STM-POV
    search WDL, so blend_wdl must return the same thing for each.
    """
    stm_wdl = [0.85, 0.10, 0.05] if stm_winning else [0.05, 0.10, 0.85]
    z = 1.0 if stm_winning else -1.0

    Y_w = blend_wdl(z, list(stm_wdl), True)
    Y_b = blend_wdl(z, list(reversed(stm_wdl)), False)

    np.testing.assert_allclose(Y_w, Y_b, atol=1e-6)
    assert Y_w.sum() == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(
        Y_w, 0.5 * z_to_wdl(z) + 0.5 * np.array(stm_wdl), atol=1e-6)

    if stm_winning:
        assert Y_w[0] > Y_w[2]
    else:
        assert Y_w[2] > Y_w[0]
