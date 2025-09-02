SF_LOC = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
ENDGAME_LOC = "C:/Users/Bryan/Data/chessbot_data/endgame_tables"

import chess
WHITE_WINNING_WHITE_MOVE = chess.Board("rn5N/p2p3p/b2k3n/5p2/1p2P3/8/PPPP1PPP/RNBQKB1R w KQ - 1 12")
WHITE_WINNING_BLACK_MOVE = chess.Board("2kr1b1r/1b1nn2p/1pp2p2/p3p3/2P1BP2/1P3N2/PB1P2PP/RN1Q1RK1 b - - 1 14")
BLACK_WINNING_WHITE_MOVE = chess.Board("r1bqkbnr/ppp2pp1/2n1p3/3p3p/3P4/2P1P3/PP1B1PPP/4KB2 w kq - 1 5")
BLACK_WINNING_BLACK_MOVE = chess.Board("r1bqk2r/p1p2pbp/1pnp1np1/8/2B1PB2/2P2NP1/P1PQ1P1P/R4RK1 b kq - 0 10")
WHITE_WINS_IN_3_WHITE_MOVE = chess.Board("5k2/1R6/2p3p1/4PpP1/p2P1P2/8/PPP3P1/RNBQKB2 w Q - 0 13")
WHITE_WINS_IN_4_BLACK_MOVE = chess.Board("k7/nb1Q4/1p6/1Pp1N2p/2P5/2N4P/P2P2P1/R4RK1 b - - 0 27")
BLACK_WINS_IN_2_WHITE_MOVE = chess.Board("r3k2r/ppp2ppp/5n2/4p1bP/1P1nK1b1/P2P4/5P2/8 w kq - 3 14")
BLACK_WINS_IN_3_BLACK_MOVE = chess.Board("rn2k2r/ppp2pp1/8/2b2K1p/2B5/1P1P1PP1/P1P5/2q5 b kq - 3 22")
WHITE_HAS_LOST = chess.Board("5rk1/2pn1ppp/8/p1QNp3/4P3/3PKBP1/1r6/6q1 w - - 6 33")
BLACK_HAS_LOST = chess.Board("r4knr/3n1Qpp/1pp3B1/p2p4/1b1P4/2N4P/PPP2PP1/R1B2RK1 b - - 0 13")

START_POS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
START_POS_KEY = "8F8F01D4562F59FB"