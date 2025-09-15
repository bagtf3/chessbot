import chess
import chess.engine
import numpy as np


PIECE_TYPES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
]

SOFT_VALUE = {
    chess.PAWN:0.10, chess.KNIGHT:0.30, chess.BISHOP:0.30,
    chess.ROOK:0.50, chess.QUEEN:0.90, chess.KING:1.00
}

# Precompute once
LOOKUP_RANKS = np.array([-(1 + p // 8) for p in range(64)], dtype=np.int8)
LOOKUP_FILES = np.array([p % 8 for p in range(64)], dtype=np.int8)

def get_rank_file(pieces):
    pieces = np.asarray(pieces, dtype=np.int32)
    return LOOKUP_RANKS[pieces], LOOKUP_FILES[pieces]


def get_board_state(board):
    state = np.zeros((64, 25), dtype=np.int8)

    for color in [chess.WHITE, chess.BLACK]:
        for piece in range(1, 7):
            channel = piece-1 if color else piece-1+6
            attack_channel = channel + 12

            squares = np.fromiter(board.pieces(piece, color), dtype=np.int32)
            state[squares, channel] = 1

            if squares.size > 0:
                for sq in squares:
                    attacked = np.fromiter(board.attacks(sq), dtype=np.int32)
                    state[attacked, attack_channel] += 1
    
    # some meta data
    castling = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ] * 2
    ep_vec = np.zeros(8, dtype=np.int8)
    if board.ep_square:
        ep_vec[:] = -1
        ep_vec[board.ep_square % 8] = 1

    meta = np.array([castling, ep_vec]).reshape(16)
    state[:, -1] = np.tile(meta, 4)

    state = state.reshape(8, 8, 25)    
    state =  np.flipud(state)
    return state

# Directions for sliders (N, NE, E, SE, S, SW, W, NW)
DIRS = [(+1,0),(+1,+1),(0,+1),(-1,+1),(-1,0),(-1,-1),(0,-1),(+1,-1)]
KNIGHT = [(-2,-1),(-2,+1),(-1,+2),(+1,+2),(+2,+1),(+2,-1),(+1,-2),(-1,-2)]
PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]  # Queen handled via slides

def rank_file(sq):
    return divmod(sq, 8)


def is_inside(r,f):
    return 0 <= r < 8 and 0 <= f < 8


def dir_index(dr, df):
    sdr, sdf = np.sign(dr), np.sign(df)
    for i,(rr,ff) in enumerate(DIRS):
        if (np.sign(rr), np.sign(ff)) == (sdr, sdf) and (rr or ff):
            return i
    return None


def knight_index(dr, df):
    for i,(rr,ff) in enumerate(KNIGHT):
        if (dr, df) == (rr, ff): return i
    return None


def lane_index(fr, ff, tr, tf, white_to_move):
    df = (tf - ff)
    if white_to_move:
        if df == 0: return 0
        return 1 if df == -1 else 2
    else:
        if df == 0: return 0
        return 1 if df == +1 else 2


def move_to_8x8x73(move: chess.Move, board: chess.Board):
    fr, ff = rank_file(move.from_square) # from rank, from file
    tr, tf = rank_file(move.to_square) # to rank, to file
    dr, df = (tr - fr), (tf - ff) # delta rank, delta file

    # Knight planes 56..63
    if (abs(dr), abs(df)) in {(1,2),(2,1)}:
        k = knight_index(dr, df)
        return fr, ff, 56 + k

    # Underpromotions planes 64..72 (Queen promo uses slides)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        lane = lane_index(fr, ff, tr, tf, board.turn)       # 0..2
        kind = PROMO_PIECES.index(move.promotion)            # N,B,R -> 0..2
        return fr, ff, 64 + lane*3 + kind

    # Sliding planes 0..55 (includes king 1-step, queen, and queen-promo)
    if dr == 0 or df == 0 or abs(dr) == abs(df):
        dist = max(abs(dr), abs(df))                         # 1..7
        d = dir_index(dr, df)                               # 0..7
        return fr, ff, d*7 + (dist-1)

    raise ValueError("Unsupported move geometry")


def idx_to_move_8x8x73(fr, ff, plane, board: chess.Board) -> chess.Move:
    from_sq = fr*8 + ff

    if 0 <= plane <= 55:  # sliding
        d = plane // 7
        dist = (plane % 7) + 1
        dr, df = DIRS[d]
        tr, tf = fr + dr*dist, ff + df*dist
        if not is_inside(tr, tf): raise ValueError("Off-board slide")
        to_sq = tr*8 + tf
        promo = None
        # if a pawn reaches last rank via slide, treat as Q-promo
        if board.piece_type_at(from_sq) == chess.PAWN:
            if (board.turn and tr == 7) or ((not board.turn) and tr == 0):
                promo = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promo)

    if 56 <= plane <= 63:  # knight
        k = plane - 56
        dr, df = KNIGHT[k]
        tr, tf = fr + dr, ff + df
        if not is_inside(tr, tf): raise ValueError("Off-board knight")
        return chess.Move(from_sq, tr*8 + tf)

    if 64 <= plane <= 72:  # underpromotions (N,B,R)
        sub = plane - 64
        lane, kind = divmod(sub, 3)   # lane 0..2, piece 0..2
        if board.turn:  # White forward
            dr, df = (1, 0) if lane == 0 else (1, -1 if lane == 1 else 1)
            if lane == 2: df = +1
        else:            # Black forward
            dr, df = (-1, 0) if lane == 0 else (-1, +1 if lane == 1 else -1)
            if lane == 2: df = -1
        tr, tf = fr + dr, ff + df
        if not is_inside(tr, tf): raise ValueError("Off-board promo lane")
        promo = PROMO_PIECES[kind]    # N,B,R
        return chess.Move(from_sq, tr*8 + tf, promotion=promo)

    raise ValueError("Bad plane index")


def legal_mask_8x8x73(board: chess.Board):
    mask = np.zeros((8,8,73), dtype=bool)
    for mv in board.legal_moves:
        fr, ff, pl = move_to_8x8x73(mv, board)
        mask[fr, ff, pl] = True
    return mask


PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
]


def piece_to_move_target(board, Yp, eps=1e-6):
    """Return a length-6 soft label over [P,N,B,R,Q,K] from Yp."""
    t = np.zeros(6, dtype=np.float32)
    flat = Yp.reshape(-1)
    nz = np.nonzero(flat)[0]
    if nz.size == 0:
        # fallback: uniform over piece types that have at least one legal move
        types_present = set()
        for mv in board.legal_moves:
            pt = board.piece_type_at(mv.from_square)
            if pt: types_present.add(pt)
        idxs = [PIECE_ORDER.index(pt) for pt in types_present] or [0]
        t[idxs] = 1.0 / len(idxs)
        return t

    for ii in nz:
        fr, ff, pl = np.unravel_index(ii, (8,8,73))
        mv = idx_to_move_8x8x73(fr, ff, pl, board)
        pt = board.piece_type_at(mv.from_square)
        if pt:
            t[PIECE_ORDER.index(pt)] += flat[ii]

    s = t.sum()
    if s > 0:
        t /= s
    else:
        t[:] = 1.0 / 6.0
    # small floor to avoid exact zeros if you like:
    # t = (t + eps); t /= t.sum()
    return t
    
    
def score_to_cp_white(board_score):
    # always look from whites perspective
    from_white = board_score.white()
    
    #check for mates
    if from_white.score() is None:
        return np.clip(from_white.score(mate_score=16), -10, 10) / 10
        
    else:
        return np.clip(from_white.score() / 1000, -0.95, 0.95)


def pawns_to_winprob(pawns, k_pawns=4.0):
    return 1.0 / (1.0 + np.exp(-k_pawns * pawns))


def winprob_to_policy(winprobs, temperature=1.0, eps=1e-7):
    p = np.clip(np.asarray(winprobs, dtype=np.float32), eps, 1.0 - eps)
    logits = np.log(p) - np.log1p(-p)
    logits /= float(temperature)
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def build_training_targets_8x8x73(board, info_list, k_pawns=150.0, temp=2.0):
    """
    Returns:
      Y_policy: (8,8,73) float32 with a prob dist over the PV-first moves from info_list
      y_value:  (1,) float32 win prob for the *position* from top line
    Assumes White to move (no mirroring).
    """
    # Collect top move from each multipv line (dedup in case SF repeats a move)
    move_to_cp = {}
    for e in info_list:
        if 'pv' not in e or not e['pv']:
            continue
        mv = e['pv'][0]
        if mv not in board.legal_moves:
            continue
        cp = score_to_cp_white(e['score'])  # White POV
        # Keep the *best* cp per move if duplicated
        move_to_cp[mv] = max(move_to_cp.get(mv, -1e7), cp)

    if not move_to_cp:
        # fallback: uniform over legal moves
        Y = np.zeros((8,8,73), dtype=np.float32)
        legal = list(board.legal_moves)
        if not legal:
            return Y, np.array([0.5], dtype=np.float32)
        
        idxs = [move_to_8x8x73(m, board) for m in legal]
        p = np.full(len(idxs), 1.0/len(idxs), dtype=np.float32)
        frs, ffs, pls = np.array(idxs, dtype=int).T
        Y[frs, ffs, pls] = p
        return Y, np.array([0.5], dtype=np.float32)

    # Convert per-move CP -> STM winprob -> soft policy
    # (STM is White here by assumption)
    moves, cps = zip(*move_to_cp.items())
    winprobs = pawns_to_winprob(np.array(cps, dtype=np.float32), k_pawns=k_pawns)
    #probs = winprob_to_policy(winprobs, temperature=1.0)

    # Scatter into 8x8x73
    Y = np.zeros((8,8,73), dtype=np.float32)
    idxs = [move_to_8x8x73(m, board) for m in moves]
    frs, ffs, pls = np.array(idxs, dtype=int).T
    Y[frs, ffs, pls] = winprobs

    # Position value target from the TOP line
    top_cp = score_to_cp_white(info_list[0]['score'])
    y_value = pawns_to_winprob(top_cp, k_pawns=k_pawns).astype(np.float32)
    return {'policy_logits': Y, "value": np.array([y_value], dtype=np.float32)}


def encode_worker(fen):
    board = chess.Board(fen)
    return fen, get_board_state(board)



def move_to_labels(board, move):
    from_idx = move.from_square         # 0..63
    to_idx   = move.to_square           # 0..63

    # Piece to move: 0..5 (pawn=0 .. king=5)
    piece = board.piece_type_at(move.from_square)
    piece_idx = piece - 1

    # Promotion: 0=None, 1=Knight, 2=Bishop, 3=Rook, 4=Queen
    promo_idx = 0
    if move.promotion is not None:
        promo_map = {
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK:   3,
            chess.QUEEN:  4,
        }
        promo_idx = promo_map[move.promotion]

    return from_idx, to_idx, piece_idx, promo_idx


def get_legal_labels(board, moves):
    N = len(moves)
    f, t, pc, pr = np.zeros(N, int), np.zeros(N, int), np.zeros(N, int), np.zeros(N, int)
    
    for i, mv in enumerate(moves):
        f[i], t[i], pc[i], pr[i] = move_to_labels(board, mv)
    return {"from":f, "to":t, "piece":pc, "promo": pr}


def compute_move_priors(model_outputs, legal_labels, weights=None):
    """
    Compute move priors given model outputs and legal move labels.
    Returns a (N_moves,) array of probabilities.
    """
    if weights is None:
        weights = {"from":1.0, "to":1.0, "piece":0.5, "promo":0.1}

    N = len(legal_labels["from"])
    
    # --- Safety guards ---
    if N == 0:
        # No legal moves: return empty priors
        return np.array([], dtype=np.float32)
    if N == 1:
        # Only one move: forced
        return np.array([1.0], dtype=np.float32)

    # Weighted log-likelihood accumulation
    priors_best = np.zeros(N, dtype=np.float32)
    for factor in ["from","to","piece","promo"]:
        idxs = legal_labels[factor]
        if len(idxs) == 0:
            continue
        p_best = model_outputs[f"best_{factor}"][idxs]
        p_best = np.clip(p_best, 1e-9, 1.0)
        priors_best += weights[factor] * np.log(p_best)

    # Convert back from log-space to probabilities
    temperature = 1.0
    priors_best = np.exp((priors_best - np.max(priors_best)) / temperature)
    priors_best /= priors_best.sum()

    return priors_best


def sf_entry_to_value(entry, cp_cap=1500.0, mate_decay_per_ply=0.02, mate_min=0.90):
    """
    Map Stockfish entries to [-1, 1] with linear CP scaling and clipping.
    - CP: clip at +/- cp_cap, scale linearly.
    - Mate: sign * score where score is ~1.0 for mate-in-1, then slightly decays by ply.
    """
    mate = entry.get("Mate")
    if mate is None and entry.get("type") == "mate":
        mate = entry.get("value")

    if mate is not None:
        sign = 1.0 if mate > 0 else -1.0
        dist = abs(mate)
        score = 1.0 - mate_decay_per_ply * max(0, dist - 1)
        score = max(mate_min, min(1.0, score))
        return sign * score

    cp = entry.get("Centipawn")
    if cp is None:
        cp = entry.get("value", 0)
    cp = 0 if cp is None else cp
    cp = max(-cp_cap, min(cp_cap, float(cp)))
    return cp / cp_cap


def sf_top_moves_to_values(entries, cp_cap=1500.0, mate_decay_per_ply=0.02, mate_min=0.90):
    out = []
    for e in entries:
        out.append(sf_entry_to_value(
            e, cp_cap=cp_cap, mate_decay_per_ply=mate_decay_per_ply, mate_min=mate_min
        ))
        
    return out


def values_to_priors(vals, temp=0.15, mix=0.0, floor=1e-12):
    """
    Turn value scores in [-1,1] into probabilities via softmax.
    temp: lower = sharper (e.g., 0.10..0.25). 
    mix:  blend with uniform to keep some exploration (0..1).
    """
    v = np.asarray(vals, dtype=np.float64)
    z = v / float(temp)
    z -= z.max()              # numerical stability
    p = np.exp(z)
    p /= p.sum()

    if floor:
        p = np.maximum(p, floor)
        p /= p.sum()

    if mix:
        n = p.size
        p = (1.0 - mix) * p + mix * (1.0 / n)
        p /= p.sum()
    return p


