import numpy as np
from copy import deepcopy

# piece order: 0=P, 1=N, 2=B, 3=R, 4=Q, 5=K
PSQT_BASE = np.array([
    # PAWN (64)
    [0,0,0,0,0,0,0,0,
     5,10,10,-20,-20,10,10,5,
     5,-5,-10,0,0,-10,-5,5,
     0,0,0,20,20,0,0,0,
     5,5,10,25,25,10,5,5,
     10,10,20,30,30,20,10,10,
     50,50,50,50,50,50,50,50,
     0,0,0,0,0,0,0,0],
    # KNIGHT
    [-50,-40,-30,-30,-30,-30,-40,-50,
     -40,-20,0,0,0,0,-20,-40,
     -30,0,10,15,15,10,0,-30,
     -30,5,15,20,20,15,5,-30,
     -30,0,15,20,20,15,0,-30,
     -30,5,10,15,15,10,5,-30,
     -40,-20,0,5,5,0,-20,-40,
     -50,-40,-30,-30,-30,-30,-40,-50],
    # BISHOP
    [-20,-10,-10,-10,-10,-10,-10,-20,
     -10,5,0,0,0,0,5,-10,
     -10,10,10,10,10,10,10,-10,
     -10,0,10,10,10,10,0,-10,
     -10,5,5,10,10,5,5,-10,
     -10,0,5,10,10,5,0,-10,
     -10,0,0,0,0,0,0,-10,
     -20,-10,-10,-10,-10,-10,-10,-20],
    # ROOK
    [0,0,0,5,5,0,0,0,
     -5,0,0,0,0,0,0,-5,
     -5,0,0,0,0,0,0,-5,
     -5,0,0,0,0,0,0,-5,
     -5,0,0,0,0,0,0,-5,
     -5,0,0,0,0,0,0,-5,
     5,10,10,10,10,10,10,5,
     0,0,0,0,0,0,0,0],
    # QUEEN
    [-20,-10,-10,-5,-5,-10,-10,-20,
     -10,0,0,0,0,0,0,-10,
     -10,0,5,5,5,5,0,-10,
     -5,0,5,5,5,5,0,-5,
     0,0,5,5,5,5,0,-5,
     -10,5,5,5,5,5,0,-10,
     -10,0,5,0,0,0,0,-10,
     -20,-10,-10,-5,-5,-10,-10,-20],
    # KING
    [20,30,10,0,0,10,30,20,
     20,20,0,0,0,0,20,20,
     -10,-20,-20,-20,-20,-20,-20,-10,
     20,-30,-30,-40,-40,-30,-30,-20,
     -30,-40,-40,-50,-50,-40,-40,-30,
     -30,-40,-40,-50,-50,-40,-40,-30,
     -30,-40,-40,-50,-50,-40,-40,-30,
     -30,-40,-40,-50,-50,-40,-40,-30]
], dtype=np.int32)


def build_psqt_buckets_from_base(psqt_base):
    """
    psqt_base: np.array shape (6,64)
    returns buckets shaped (4, 6*64) as int32 (same format evaluator expects)
    """
    base = np.array(psqt_base, dtype=np.int32, copy=True)  # (6,64)
    buckets = np.zeros((4, 6, 64), dtype=np.int32)
    # buckets 0..2 reuse base
    buckets[0] = base.copy()
    buckets[1] = base.copy()
    buckets[2] = base.copy()

    # bucket 3 = endgame tweak (same as earlier logic)
    eg = base.copy()
    # pawns: reward advanced pawns (white-perspective)
    pawn_idx = 0
    for sq in range(64):
        rank = (sq // 8) + 1  # 1..8
        if rank >= 6:
            eg[pawn_idx, sq] += 30
        elif rank == 5:
            eg[pawn_idx, sq] += 12
        elif rank == 4:
            eg[pawn_idx, sq] += 6

    # king: encourage centralization in endgame
    king_idx = 5
    files = [3, 4]   # d,e
    ranks = [3, 4]   # 4th,5th (0-based 3,4)
    for r in ranks:
        for f in files:
            eg[king_idx, r*8 + f] += 40

    # Slightly tweak heavy pieces
    for pidx in (3, 4):  # rook, queen
        eg[pidx, :] += 6

    buckets[3] = eg
    # Flatten to (4, 6*64)
    return buckets.reshape((4, 6*64)).astype(np.int32)


def build_weights(zeros=False):
    psqt_arr = build_psqt_buckets_from_base(PSQT_BASE)  # shape (4,384)

    mobility_weights = [0, 4, 6, 2, 1, 0]  # pawn..king
    
    tactical_weights = [
        # pawn (attacked_by_lower, defended, hanging)
        -10,  4,  -40,
        # knight
        -35, 10, -120,
        # bishop
        -30, 10, -110,
        # rook
        -45, 10, -150,
        # queen
        -90, 20, -350,
        # king
        0, 20, 0
    ]
    
    king_weights = [0, 0, 0]
    stm_bias = 0 if zeros else 30
    global_scale = 100
    
    if zeros:
        mobility_weights = [0 for w in mobility_weights]
        tactical_weights = [0 for w in tactical_weights]
        king_weights = [0 for w in king_weights]
        
    return {
        "psqt": psqt_arr,  # (4,384) int32
        "mobility_weights": np.array(mobility_weights, dtype=np.int32),
        "tactical_weights": np.array(tactical_weights, dtype=np.int32),
        "king_weights": np.array(king_weights, dtype=np.int32),
        "stm_bias": int(stm_bias),
        "global_scale": int(global_scale),
    }


def mirror_index(sq):
    """mirror file across a<->h for 0..63 index"""
    file = sq % 8
    rank = sq // 8
    mirror_file = 7 - file
    return rank * 8 + mirror_file


def jostle_weights(weights=None, delta=5, seed=None):
    """
    Return a new weights dict with each integer jostled by randint(-delta, delta).
    Enforces left-right symmetry for PSQT by only jostling half of each 64-square
    table and mirroring the changes.
    - weights: existing weights dict (if None, build_weights() will be used)
    - delta: integer jitter amplitude (applies to all integer fields uniformly)
    - seed: optional RNG seed for reproducibility
    """
    
    rng = np.random.RandomState(seed)
    if weights is None:
        weights = build_weights()

    w = deepcopy(weights)  # avoid mutating caller

    # PSQT: shape may be (4,384) or similar
    ps = np.array(w["psqt"], dtype=np.int32, copy=True).reshape((4, 6, 64))

    # jostle half the squares (files 0..3) and mirror to files 7..4
    for b in range(ps.shape[0]):      # buckets 0..3
        for p in range(ps.shape[1]):  # pieces 0..5
            for sq in range(64):
                f = sq % 8
                # operate only on left half (files 0..3)
                if f <= 3:
                    msq = mirror_index(sq)
                    delta_val = int(rng.randint(-delta, delta + 1))
                    # add same delta to both mirror squares to preserve symmetry
                    ps[b, p, sq] = int(ps[b, p, sq]) + delta_val
                    ps[b, p, msq] = int(ps[b, p, msq]) + delta_val

    w["psqt"] = ps.reshape((4, 6*64)).astype(np.int32)

    # other arrays: add uniform jitter elementwise
    for key in ("mobility_weights", "tactical_weights", "king_weights"):
        arr = np.array(w[key], dtype=np.int32, copy=True)
        jitter = rng.randint(-delta, delta + 1, size=arr.shape)
        arr = (arr + jitter).astype(np.int32)
        w[key] = arr

    # scalars
    w["stm_bias"] = int(w.get("stm_bias", 0) + rng.randint(-delta, delta + 1))
    w["global_scale"] = int(w.get("global_scale", 100) + rng.randint(-delta, delta + 1))

    return w

