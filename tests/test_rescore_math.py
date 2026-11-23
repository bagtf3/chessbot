"""STM-POV correctness for the training targets.

The POV test covers both sides of a mirrored pair: a position and its
colour-mirror are the same position to a STM-relative encoder, so every
target built for one must come out identical to the other.
"""
import numpy as np
import pytest

from chessbot.rescore import blend_wdl, blend_wdl_stm, z_to_wdl


def test_native_stm_wdl_winning_mover():
    # A winning Black mover has the same STM target as a winning White mover.
    target = blend_wdl_stm(1, [0.8, 0.1, 0.1])
    np.testing.assert_allclose(target, [0.9, 0.05, 0.05])
    assert target.dtype == np.float32


def test_native_stm_wdl_missing_search():
    np.testing.assert_array_equal(blend_wdl_stm(-1, None), [0.0, 0.0, 1.0])


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
