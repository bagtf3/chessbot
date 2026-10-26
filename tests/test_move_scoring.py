"""Rules of the three scorers, and the port off the old inline block.

The inline version is reproduced here verbatim as `legacy_loss` and the
rescore scorer is checked against it across a grid, so the refactor is pinned
to the behaviour it replaced rather than to a restatement of it.
"""
import pytest

from chessbot.config import Config
from chessbot.move_scoring import make_scorer

CFG = Config()
BLUNDER_CP = CFG.rescore_blunder_cp_winner


def legacy_loss(best_cp, played_cp, z_stm, blunder_cp):
    """The pre-refactor block from rescore.py, unchanged."""
    EQUIV = min(max(abs(best_cp) * 0.1, CFG.rescore_equiv_min),
                CFG.rescore_equiv_max)
    delta = best_cp - played_cp
    swapped = False
    if abs(delta) <= EQUIV:
        delta = 0
    elif delta <= -EQUIV:
        best_cp = played_cp
        swapped = True
    loss = delta
    missed = (best_cp >= 1200 and played_cp >= 500 and z_stm > 0
              and delta >= blunder_cp)
    if missed:
        loss = min(100, loss)
    return loss, swapped, missed, EQUIV


GRID = [
    (b, p, z)
    for b in (0, 30, 120, 400, 1200, 1500, 2500)
    for p in (-800, -100, 0, 25, 300, 500, 1190, 1450, 2500)
    for z in (-1.0, 0.0, 1.0)
]


@pytest.mark.parametrize("best_cp,played_cp,z_stm", GRID)
def test_rescore_scorer_matches_the_old_inline_block(best_cp, played_cp, z_stm):
    want_loss, want_swap, want_mate, want_equiv = legacy_loss(
        best_cp, played_cp, z_stm, BLUNDER_CP)

    ms = make_scorer('rescore', CFG).score(
        'e2e4', best_cp, 'd2d4', played_cp,
        blunder_cp=BLUNDER_CP, z_stm=z_stm)

    assert ms.loss == want_loss
    assert ms.swapped == want_swap
    assert ms.missed_mate == want_mate
    assert ms.equiv == want_equiv
    assert ms.best_uci == ('d2d4' if want_swap else 'e2e4')


def test_brp_scorer_is_a_raw_delta():
    """probe_cpl drives eviction, the ratchet and escalation, so it must stay
    exactly best_cp - played_cp with no band and no cap."""
    s = make_scorer('brp')
    for best_cp, played_cp in ((300, 290), (1500, 600), (0, 200), (50, 50)):
        ms = s.score('e2e4', best_cp, 'd2d4', played_cp,
                     blunder_cp=BLUNDER_CP, z_stm=1.0)
        assert ms.loss == best_cp - played_cp
        assert not ms.swapped
        assert not ms.missed_mate
        assert ms.equiv == 0.0


def test_lc0_matches_rescore_wherever_the_outcome_is_not_consulted():
    """Same rules by construction; the only divergence is the missed-mate
    test, which lc0 evaluates without a game result."""
    rs, lc = make_scorer('rescore', CFG), make_scorer('lc0', CFG)
    for best_cp, played_cp in ((400, 380), (900, 300), (120, 400), (60, 55)):
        a = rs.score('e2e4', best_cp, 'd2d4', played_cp,
                     blunder_cp=BLUNDER_CP, z_stm=1.0)
        b = lc.score('e2e4', best_cp, 'd2d4', played_cp,
                     blunder_cp=BLUNDER_CP)
        assert (a.loss, a.swapped, a.equiv) == (b.loss, b.swapped, b.equiv)


def test_lc0_caps_a_missed_mate_without_a_game_result():
    """z_stm is None for a probe position; out_cp >= 1200 stands in for it."""
    rs, lc = make_scorer('rescore', CFG), make_scorer('lc0', CFG)
    args = ('e2e4', 2000, 'd2d4', 700)

    assert lc.score(*args, blunder_cp=BLUNDER_CP).missed_mate
    assert lc.score(*args, blunder_cp=BLUNDER_CP).loss == 100
    # the same inputs without an outcome are not a missed mate for rescore
    assert not rs.score(*args, blunder_cp=BLUNDER_CP).missed_mate


def test_equiv_band_scales_with_the_position_eval():
    """10% of the eval, clamped to [equiv_min, equiv_max] -- so it widens with
    the eval but saturates rather than growing without bound."""
    s = make_scorer('rescore', CFG)
    tight = s.score('e2e4', 50, 'd2d4', 20, blunder_cp=BLUNDER_CP, z_stm=0.0)
    wide = s.score('e2e4', 2000, 'd2d4', 1970,
                   blunder_cp=BLUNDER_CP, z_stm=0.0)

    assert tight.equiv == CFG.rescore_equiv_min
    assert wide.equiv == CFG.rescore_equiv_max
    # the same 30cp gap: inside the band at eval 2000, outside it at eval 50
    assert wide.loss == 0
    assert tight.loss == 30

    # clamped at the top rather than scaling on to 10% of 2000
    assert s.score('e2e4', 2000, 'd2d4', 1900,
                   blunder_cp=BLUNDER_CP, z_stm=0.0).loss == 100


def test_swap_adopts_the_played_move_as_the_bar():
    s = make_scorer('rescore', CFG)
    ms = s.score('e2e4', 100, 'd2d4', 600, blunder_cp=BLUNDER_CP, z_stm=1.0)
    assert ms.swapped
    assert ms.best_uci == 'd2d4'
    assert ms.best_cp == 600
    assert ms.loss < 0


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_scorer('nonsense', CFG)
