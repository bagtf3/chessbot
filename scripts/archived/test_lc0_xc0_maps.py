"""Test lc0_logits_to_xc0_batch per-position mapping."""
import numpy as np
import pyfastchess
import chess
from xerces_training.uci_to_idx import uci_to_idx as UCI_TO_IDX
from chessbot.lc0_utils import lc0_logits_to_xc0_batch, Lc0Board
from chessbot.rescore import lc0_table_index


def get_indices(board, features_uint8):
    """Compute (lc0_idx, xc0_idx) arrays for a board position."""
    ti   = lc0_table_index(features_uint8)
    ucis = board.legal_moves()
    lc0_idx = np.array([UCI_TO_IDX[ti][u.rstrip('n')] for u in ucis], dtype=np.int32)
    xc0_idx = np.array(board.moves_to_indices(ucis), dtype=np.int32)
    return lc0_idx, xc0_idx, ucis


def test_startpos_row_sum():
    board = pyfastchess.Board()
    feats = np.array(board.lc0_features())
    lc0_idx, xc0_idx, ucis = get_indices(board, feats)

    # uniform logits over legal moves only
    logits = np.full((1, 1858), -1e9, dtype=np.float32)
    logits[0, lc0_idx] = 0.0

    result = lc0_logits_to_xc0_batch(logits, [lc0_idx], [xc0_idx])
    row_sum = result[0].sum()
    print(f"  startpos: {len(ucis)} legal moves, row_sum={row_sum:.6f}")
    assert abs(row_sum - 1.0) < 1e-5, f"row sum {row_sum} != 1.0"
    # each legal move should have equal probability
    probs_at_legal = result[0, xc0_idx]
    expected = 1.0 / len(ucis)
    assert np.allclose(probs_at_legal, expected, atol=1e-5), "non-uniform probs"
    print("  PASS: startpos row sum and uniform probs")


def test_queen_promo():
    """a7a8q should land on correct XC0 slot; a7a8n knight promo goes to a7a8 in LC0."""
    fen = "8/P7/8/8/8/8/8/K6k w - - 0 1"
    board = pyfastchess.Board()
    board = pyfastchess.Board(fen)
    feats = np.array(board.lc0_features())
    lc0_idx, xc0_idx, ucis = get_indices(board, feats)

    print(f"  promo position legal moves: {ucis}")
    print(f"  lc0_idx: {lc0_idx}, xc0_idx: {xc0_idx}")

    # put all weight on a7a8q
    logits = np.full((1, 1858), -1e9, dtype=np.float32)
    logits[0, lc0_idx] = 0.0
    uci_list = list(ucis)
    q_pos = uci_list.index('a7a8q')
    logits[0, lc0_idx] = -10.0
    logits[0, lc0_idx[q_pos]] = 0.0

    result = lc0_logits_to_xc0_batch(logits, [lc0_idx], [xc0_idx])
    row_sum = result[0].sum()
    assert abs(row_sum - 1.0) < 1e-4, f"row sum {row_sum}"
    top_xc0 = int(result[0].argmax())
    assert top_xc0 == xc0_idx[q_pos], f"top slot {top_xc0} != expected {xc0_idx[q_pos]}"
    print(f"  PASS: queen promo -> xc0 slot {top_xc0}")


def test_knight_promo_lc0_key():
    """a7a8n from pyfastchess maps to 'a7a8' key in UCI_TO_IDX (rstrip n)."""
    fen = "8/P7/8/8/8/8/8/K6k w - - 0 1"
    board = pyfastchess.Board()
    board = pyfastchess.Board(fen)
    feats = np.array(board.lc0_features())
    ti    = lc0_table_index(feats)
    ucis  = board.legal_moves()

    for u in ucis:
        key = u.rstrip('n')
        assert key in UCI_TO_IDX[ti], f"'{key}' (from '{u}') not in UCI_TO_IDX[{ti}]"
    print(f"  PASS: all {len(ucis)} legal move UCIs resolve in UCI_TO_IDX")


def test_batch_mixed():
    """Batch of 8 positions — row sums all ~1.0."""
    n = 8
    boards, lc0_idx_list, xc0_idx_list = [], [], []
    logits = np.zeros((n, 1858), dtype=np.float32)

    for i in range(n):
        b = pyfastchess.Board()
        feats = np.array(b.lc0_features())
        li, xi, ucis = get_indices(b, feats)
        lc0_idx_list.append(li)
        xc0_idx_list.append(xi)
        logits[i, li] = np.random.randn(len(li)).astype(np.float32)

    result = lc0_logits_to_xc0_batch(logits, lc0_idx_list, xc0_idx_list)
    sums = result.sum(axis=1)
    print(f"  batch row sums: min={sums.min():.6f} max={sums.max():.6f}")
    assert np.allclose(sums, 1.0, atol=1e-5), f"row sums off: {sums}"
    print("  PASS: batch row sums")


if __name__ == "__main__":
    print("test_startpos_row_sum")
    test_startpos_row_sum()
    print("test_queen_promo")
    test_queen_promo()
    print("test_knight_promo_lc0_key")
    test_knight_promo_lc0_key()
    print("test_batch_mixed")
    test_batch_mixed()
    print("\nAll tests passed.")
