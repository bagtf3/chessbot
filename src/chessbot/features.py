import chess

VAL = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 10
}

NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5:"queen", 6:"king"}
COLORS = {1:"white", True: 'white', 0:"black", False: 'black'}


def attacked_by_lower_value(board, color, square):
    """
    Returns True if the piece of given color on `square` is attacked by
    an enemy piece of strictly lower value.
    """
    piece = board.piece_at(square)
    if not piece or piece.color != color:
        return False  # no such piece there
    
    my_val = VAL[piece.piece_type]
    opp_color = not color
    
    for attacker_sq in board.attackers(opp_color, square):
        attacker_piece = board.piece_at(attacker_sq)
        if attacker_piece and VAL[attacker_piece.piece_type] < my_val:
            return True
    return False


def attackers(board, color, sq):
    return board.attackers(color, sq)


def defended(board, color, sq):
    # defended = at least one friendly piece attacks this square
    return bool(attackers(board, color, sq))


def en_prise(board, color, sq):
    # en prise = opponent can capture this square (piece could be taken)
    return bool(attackers(board, not color, sq))


def is_undefended(board, color, sq):
    return not defended(board, color, sq)


def is_hanging(board, color, sq):
    return is_undefended(board, color, sq) and en_prise(board, color, sq)


def piece_features(piece, board, color):
    sqs = list(board.pieces(piece, color))
    
    features = [
        len(sqs), sum([1*is_undefended(board, color, sq) for sq in sqs]),
        sum([1*en_prise(board, color, sq) for sq in sqs]),
        sum([1*is_hanging(board, color, sq) for sq in sqs]),
        sum([1*attacked_by_lower_value(board, color, sq) for sq in sqs])
    ]
    
    return features        


def queen_features(board, color):
    return piece_features(5, board, color)


def all_piece_features(board):
    functions = {
        "undefended": is_undefended,
        "hanging": is_hanging,
        "en_prise": en_prise,
        "attacked_by_lower_value": attacked_by_lower_value
    }
    
    feats = {}    
    for piece in [1, 2, 3, 4, 5]:
        blk_sqs = list(board.pieces(piece, 0))
        wt_sqs = list(board.pieces(piece, 1))
        
        for f in functions:
            if piece == 1 and f == 'attacked_by_lower_value':
                continue
        
            key = f"{NAMES[piece]}_{f}"
            
            white = sum([1*functions[f](board, 1, sq) for sq in wt_sqs])
            black = sum([1*functions[f](board, 0, sq) for sq in blk_sqs])
            
            # convert queen stuff to binary
            if piece == 5:
                white = min(white, 1)
                black = min(black, 1)
                
            feats[key] = [white, black]
            
    return feats


def king_square(board, color):
    return board.king(color)


def king_ring(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    ring = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            nf, nr = f + df, r + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                ring.append(chess.square(nf, nr))
    return ring


def piece_squares(board, color, piece_type):
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type == piece_type:
            yield sq


def count_pawn_shield(board, color, ksq):
    # crude: count friendly pawns on king file and adjacent files,
    # one and two ranks forward from king (depending on color).
    f = chess.square_file(ksq)
    r = chess.square_rank(ksq)
    direction = 1 if color == chess.WHITE else -1
    files = [x for x in (f - 1, f, f + 1) if 0 <= x <= 7]
    ranks = [r + direction, r + 2 * direction]
    cnt = 0
    for ff in files:
        for rr in ranks:
            if 0 <= rr <= 7:
                sq = chess.square(ff, rr)
                p = board.piece_at(sq)
                if p and p.color == color and p.piece_type == chess.PAWN:
                    cnt += 1
    return cnt  # 0..6 typical


def queenization_exposure(board, color, ksq):
    new_board = board.copy()
    king_attacks = new_board.attacks(ksq)
    new_board.set_piece_at(ksq, chess.Piece(5, color))
    queen_attacks = new_board.attacks(ksq)
    return len(queen_attacks) - len(king_attacks)


def ring_pressure(board, color, ksq):
    this_kings_ring = king_ring(ksq)
    pressure = sum(1 for sq in this_kings_ring if board.is_attacked_by(not color, sq))
    return pressure


def escape_squares(board, color, ksq):
    moves = []
    for m in board.legal_moves:
        if m.from_square == ksq:
            moves.append(m)
    return len(moves)


def all_king_exposure_features(board):
    wt_ksq = king_square(board, 1)
    blk_ksq = king_square(board, 0)
    
    functions = {
        "king_ray_exposure": queenization_exposure,
        "king_ring_pressure": ring_pressure,
        "king_pawn_shield": count_pawn_shield,
        "king_escape_square":escape_squares
    }
    
    feats = {}
    for f in functions:
        wt = functions[f](board, 1, wt_ksq)
        blk = functions[f](board, 0, blk_ksq)
        feats[f] = [wt, blk]
    
    return feats


def get_piece_value_sum(board):
    piece_map = board.piece_map()
    white_sum = 0
    black_sum = 0
    for sq, pc in piece_map.items():
        if pc.color:
            white_sum += VAL[pc.piece_type]
        else:
            black_sum += VAL[pc.piece_type]
    
    return white_sum / 10, black_sum / 10

