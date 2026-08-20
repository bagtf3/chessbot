"""Blend invariants and STM-POV correctness for the training targets.

The blend tests pin structure (sums, coverage, monotonicity), not the alpha
endpoints -- those are being tuned. The POV tests exercise all four
combinations of side-to-move and who is winning, by mirroring: a position and
its colour-mirror are the same position to a STM-relative encoder, so every
target the rescorer builds for one must come out bit-identical to the other.
"""
import chess
import numpy as np
import pytest

from pyfastchess import Board as fastboard

from chessbot import BLACK_WINNING_WHITE_MOVE, WHITE_WINNING_WHITE_MOVE
from chessbot.config import Config
from chessbot.replay_buffer import densify_policy
from chessbot.rescore import Rescorer, blend_wdl, z_to_wdl
from chessbot.utils import cp_to_value_tanh, scalar_to_wdl

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


def run_blend(fen, best_uci, best_cp, probe_cpl, best_wdl, ucis=None):
    r = make_rescorer()
    ply = make_ply(fen, ucis=ucis, best_wdl=best_wdl)
    ok = r.add_blunder_replay_training_example(
        ply, epoch=1, best_uci=best_uci, best_cp=best_cp, probe_cpl=probe_cpl)
    assert ok
    return ply['board'], sole_entry(r)


def sf_best_of(fen, avoid_first=True):
    """A legal move that is not the top-visited candidate, so SF and xc0
    disagree and the blend has something to move."""
    board = fastboard(fen)
    ucis = sorted(board.legal_moves())
    return ucis[-1] if avoid_first else ucis[0]


CPLS = [26, 50, 80, 110, 140, 150]


def test_blend_policy_and_wdl_are_distributions():
    board, (x, policy, Y, source) = run_blend(
        BLEND_FEN, sf_best_of(BLEND_FEN), best_cp=120, probe_cpl=90,
        best_wdl=[0.4, 0.4, 0.2])

    assert source == 'xc0_replay_blend'
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert Y.sum() == pytest.approx(1.0, abs=1e-5)
    assert (policy >= 0).all()
    assert (Y >= 0).all()


def test_every_legal_move_carries_mass():
    """The SF pointmass puts 1 visit on every legal move before piling the
    remainder on SF's best, so no legal move may end at zero."""
    fen = BLEND_FEN
    board, (x, policy, Y, source) = run_blend(
        fen, sf_best_of(fen), best_cp=120, probe_cpl=140,
        best_wdl=[0.4, 0.4, 0.2])

    legal = sorted(fastboard(fen).legal_moves())
    idx = fastboard(fen).moves_to_indices(legal)
    assert len(legal) == len(set(idx))
    assert (policy[list(idx)] > 0).all()
    assert np.count_nonzero(policy) == len(legal)


def test_blend_weight_is_monotone_in_cpl():
    """alpha is the only thing cpl drives: the worse xc0's move, the more the
    target must lean on SF, on both the policy and the value side."""
    fen = BLEND_FEN
    best_uci = sf_best_of(fen)
    best_cp = 400
    best_wdl = [0.1, 0.3, 0.6]

    sf_mass = []
    sf_pull = []
    for cpl in CPLS:
        board, (x, policy, Y, source) = run_blend(
            fen, best_uci, best_cp, cpl, best_wdl)
        idx = board.moves_to_indices([best_uci])[0]
        sf_mass.append(float(policy[idx]))
        wdl_sf = scalar_to_wdl(cp_to_value_tanh(best_cp, mid_cp=200.0))
        sf_pull.append(float(np.abs(Y - wdl_sf).sum()))

    assert sf_mass == sorted(sf_mass)
    assert sf_pull == sorted(sf_pull, reverse=True)


def test_blend_stays_between_the_two_endpoints():
    """A convex mix with alpha in [0, 1] can never leave the interval spanned
    by the pure-xc0 and pure-SF targets. Pins the alpha range without pinning
    its endpoints."""
    fen = BLEND_FEN
    best_uci = sf_best_of(fen)
    best_cp = 400
    best_wdl = [0.1, 0.3, 0.6]
    wdl_sf = scalar_to_wdl(cp_to_value_tanh(best_cp, mid_cp=200.0))
    wdl_xc0 = np.array(best_wdl, dtype=np.float32)

    for cpl in CPLS:
        board, (x, policy, Y, source) = run_blend(
            fen, best_uci, best_cp, cpl, best_wdl)
        lo = np.minimum(wdl_sf, wdl_xc0)
        hi = np.maximum(wdl_sf, wdl_xc0)
        assert (Y >= lo - 1e-6).all()
        assert (Y <= hi + 1e-6).all()


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
            pos_fen, best_uci, best_cp, probe_cpl=100,
            best_wdl=wdl_white_pov, ucis=cands)
        results.append((np.asarray(x), policy, Y))

    (x_w, pol_w, Y_w), (x_b, pol_b, Y_b) = results

    assert np.array_equal(x_w, x_b)
    np.testing.assert_allclose(pol_w, pol_b, atol=1e-6)
    np.testing.assert_allclose(Y_w, Y_b, atol=1e-6)

    if stm_winning:
        assert Y_w[0] > Y_w[2]
    else:
        assert Y_w[2] > Y_w[0]


@pytest.mark.parametrize("fen,stm_winning", POV_CASES)
def test_equiv_targets_are_stm_pov(fen, stm_winning):
    """Same four combos through the equiv path, which uses its own flip."""
    results = []
    for pos_fen, wdl_white_pov, _, cands, _best in \
            mirrored_case(fen, stm_winning):
        r = make_rescorer()
        ply = make_ply(pos_fen, ucis=cands, best_wdl=wdl_white_pov)
        assert r.add_equiv_replay_training_example(ply)
        x, policy, Y, source = sole_entry(r)
        assert source == 'xc0_replay_equiv'
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
