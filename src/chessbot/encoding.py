import chess
import numpy as np

PIECE_TYPES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
]

SOFT_VALUE = {
    chess.PAWN:0.10, chess.KNIGHT:0.30, chess.BISHOP:0.30,
    chess.ROOK:0.50, chess.QUEEN:0.90, chess.KING:1.00
}

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Returns [8,8,N] float32 planes. Order:
      0..5: my pieces (P,N,B,R,Q,K)
      6..11: opp pieces
      12: my_attacks
      13: opp_attacks
      14: my_value (soft piece values summed per square)
      15: opp_value
      16: my_castle_k
      17: my_castle_q
      18: opp_castle_k
      19: opp_castle_q
      20: en_passant (1 at EP square else 0)
      21: halfmove_clock (scalar plane, normalized /100)
      22: legal_move_count (scalar plane, normalized /60)
    """
    
    planes = np.zeros((8, 8, 23), dtype=np.float32)

    # --- piece planes & value planes
    my_value = np.zeros((8,8), dtype=np.float32)
    opp_value = np.zeros((8,8), dtype=np.float32)

    for sq, pc in board.piece_map().items():
        r, f = divmod(sq, 8)
        mine = (pc.color == True)
        base = 0 if mine else 6
        pt = pc.piece_type
        planes[r, f, base + (pt - 1)] = 1.0
        
        if mine:
            my_value[r, f]  += SOFT_VALUE[pt]
        else:
            opp_value[r, f] += SOFT_VALUE[pt]

    planes[:, :, 14] = my_value
    planes[:, :, 15] = opp_value

    # attack planes (pseudo-legal attacks; include pinned pieces)
    my_att = np.zeros((8,8), dtype=np.float32)
    opp_att = np.zeros((8,8), dtype=np.float32)
    for sq, pc in board.piece_map().items():
        r, f = divmod(sq, 8)
        for tgt in board.attacks(sq):
            tr, tf = divmod(tgt, 8)
            if pc.color == True:
                my_att[tr, tf] = 1.0
            else:
                opp_att[tr, tf] = 1.0
    
    planes[:,:,12] = my_att
    planes[:,:,13] = opp_att

    # castling rights
    planes[:,:,16] = 1.0 if board.has_kingside_castling_rights(True) else 0.0
    planes[:,:,17] = 1.0 if board.has_queenside_castling_rights(True) else 0.0
    planes[:,:,18] = 1.0 if board.has_kingside_castling_rights(False) else 0.0
    planes[:,:,19] = 1.0 if board.has_queenside_castling_rights(False) else 0.0

    # en passant square
    if board.ep_square is not None:
        er, ef = divmod(board.ep_square, 8)
        planes[er, ef, 20] = 1.0

    # scalar planes (normalized)
    planes[:,:,21] = min(board.halfmove_clock, 100) / 100.0
    legal_n = len(list(board.legal_moves))
    planes[:,:,22] = min(legal_n, 60) / 60.0

    return planes

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


def score_to_cp_white(score: chess.engine.PovScore) -> int:
    """Return centipawns from White's POV. Handle mates by mapping to large cp."""
    s = score.white()
    if s.is_mate():
        ply = abs(s.mate())
        sign = 1 if s.mate() > 0 else -1
        # map 'mate in ply' to a huge cp with small ply discount
        return sign * (10000 - 10*ply)
    return s.cp


def pawns_to_winprob(pawns, k_pawns=2.0):
    # pawns: array of STM-centric pawn evals (positive good for STM)
    x = np.asarray(pawns, dtype=np.float32) / float(k_pawns)
    # stable sigmoid
    pos = np.exp(-np.clip(x, None, 50))
    return 1.0 / (1.0 + pos)


def winprob_to_policy(winprobs, temperature=1.0, eps=1e-7):
    p = np.clip(np.asarray(winprobs, dtype=np.float32), eps, 1.0 - eps)
    logits = np.log(p) - np.log1p(-p)          # log-odds
    logits /= float(temperature)
    logits -= logits.max()                     # stabilize
    w = np.exp(logits)
    return w / w.sum()


def build_training_targets_8x8x73(board, info_list, k_pawns=250.0, temp=1.0):
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
        move_to_cp[mv] = max(move_to_cp.get(mv, -1e9), cp)

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
    probs = winprob_to_policy(winprobs, temperature=1.0)

    # Scatter into 8x8x73
    Y = np.zeros((8,8,73), dtype=np.float32)
    idxs = [move_to_8x8x73(m, board) for m in moves]
    frs, ffs, pls = np.array(idxs, dtype=int).T
    Y[frs, ffs, pls] = probs

    # Position value target from the TOP line
    top_cp = score_to_cp_white(info_list[0]['score'])
    y_value = pawns_to_winprob(top_cp, k_pawns=k_pawns).astype(np.float32)
    return Y, np.array([y_value], dtype=np.float32)




