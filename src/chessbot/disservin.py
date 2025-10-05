import time
import chess
import chess.polyglot
from sys import stdout

MAX_PLY = 60
CHECK_RATE = 256

VALUE_INFINITE = 32001
VALUE_NONE = 32002
VALUE_MATE = 32000
VALUE_MATE_IN_PLY = VALUE_MATE - MAX_PLY
VALUE_MATED_IN_PLY = -VALUE_MATE_IN_PLY

# Theres no TB support but it useful for other people who port this to another language to respect the TB value ranges
VALUE_TB_WIN = VALUE_MATE_IN_PLY
VALUE_TB_LOSS = -VALUE_TB_WIN
VALUE_TB_WIN_IN_MAX_PLY = VALUE_TB_WIN - MAX_PLY
VALUE_TB_LOSS_IN_MAX_PLY = -VALUE_TB_WIN_IN_MAX_PLY


# least significant bit
def lsb(x: int) -> int:
    return (x & -x).bit_length() - 1


def poplsb(x: int) -> int:
    x &= x - 1
    return x


def mate_in(ply: int) -> int:
    return VALUE_MATE - ply


def mated_in(ply: int) -> int:
    return ply - VALUE_MATE


class Limits:
    def __init__(self, nodes, depth, time):
        self.limited = {"nodes": nodes, "depth": depth, "time": time}

# fmt: off
piece_values = {
    None: 0,
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000,
}

"""
These PSQT are taken from https://www.chessprogramming.org/Simplified_Evaluation_Function
and are more or less widely used in beginner chess engines.
"""
psqt_values = {
    chess.PAWN: [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, -20, -20, 10, 10,  5,
    5, -5, -10,  0,  0, -10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT: [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
    ],
    chess.BISHOP: [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
    ],
    chess.ROOK: [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.QUEEN: [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
    ],
    chess.KING: [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
    ],
}


class Evaluation:
    @staticmethod
    def eval_side(board: chess.Board, color: chess.Color) -> int:
        occupied = board.occupied_co[color]

        material = 0
        psqt = 0

        # loop over all set bits
        while occupied:
            # find the least significant bit
            square = lsb(occupied)

            piece = board.piece_type_at(square)

            # add material
            material += piece_values[piece]

            # add piece square table value
            psqt += (
                list(reversed(psqt_values[piece]))[square]
                if color == chess.BLACK
                else psqt_values[piece][square]
            )

            # remove lsb
            occupied = poplsb(occupied)

        return material + psqt

    @staticmethod
    def evaluate(board: chess.Board) -> int:
        return Evaluation.eval_side(board, chess.WHITE) - Evaluation.eval_side(
            board, chess.BLACK
        )


from enum import Enum

Flag = Enum("Flag", ["NONEBOUND", "UPPERBOUND", "LOWERBOUND", "EXACTBOUND"])


class TEntry:
    def __init__(self) -> None:
        self.key = 0
        self.depth = 0
        self.flag = Flag.NONEBOUND
        self.score = VALUE_NONE
        self.move = chess.Move.null()


class TranspositionTable:
    def __init__(self) -> None:
        # Higher values take rather long to initialize
        self.tt_size = 2**19 - 1
        self.transposition_table = [TEntry() for _ in range(self.tt_size)]

    # Calculate "array" index
    def ttIndex(self, key: int) -> int:
        return key % self.tt_size

    # store an entry in the TT
    def storeEntry(
        self, key: int, depth: int, flag: Flag, score: int, move: chess.Move, ply: int
    ) -> None:
        index = self.ttIndex(key)
        entry = self.transposition_table[index]

        # Replacement schema
        if entry.key != key or entry.move != move:
            entry.move = move

        if entry.key != key or flag == Flag.EXACTBOUND or depth + 4 > entry.depth:
            entry.depth = depth
            entry.score = self.scoreToTT(score, ply)
            entry.key = key
            entry.flag = flag

        # self.transposition_table[index] = entry

    def probeEntry(self, key: int) -> TEntry:
        index = self.ttIndex(key)
        entry = self.transposition_table[index]

        return entry

    # if we want to save correct mate scores we have to adjust the distance
    def scoreToTT(self, s: int, plies: int) -> int:
        if s >= VALUE_TB_WIN_IN_MAX_PLY:
            return s + plies
        else:
            if s <= VALUE_TB_LOSS_IN_MAX_PLY:
                return s - plies
            else:
                return s

    # undo the previous adjustment
    def scoreFromTT(self, s: int, plies: int) -> int:
        if s >= VALUE_TB_WIN_IN_MAX_PLY:
            return s - plies
        else:
            if s <= VALUE_TB_LOSS_IN_MAX_PLY:
                return s + plies
            else:
                return s
            
class Search:
    def __init__(self, board: chess.Board) -> None:
        self.board = board

        # This is our transposition table, it stores positions
        # it is one of the most important parts of a chess engine.
        # It stores results of previously performed searches and it
        # allows to skip parts of the search tree and order moves.
        self.transposition_table = TranspositionTable()

        self.pvLength = [0] * MAX_PLY
        # This is our principal variation table, it stores the best
        # moves for each depth. It is used to print the best line
        # after the search is completed.
        self.pvTable = [[chess.Move.null()] * MAX_PLY for _ in range(MAX_PLY)]

        # Total nodes searched
        self.nodes = 0

        # Current limits for the search
        self.limit = Limits(0, MAX_PLY, 0)

        # True when the search is stopped/aborted
        self.stop = False

        # Time checking is expensive and we dont want to do it every node
        self.checks = CHECK_RATE

        # Keeps track of the zobrist hashes encountered during the search
        # Used to efficiently detect repetitions
        self.hashHistory: list[int] = []

        # History Table
        # Indexed by [color][from][to]
        self.htable = [[[0 for x in range(64)] for y in range(64)] for z in range(2)]

    def qsearch(self, alpha: int, beta: int, ply: int) -> int:
        """
        Quiescence Search, this is a special search that only searches
        captures and checks. It is needed to avoid the horizon effect.
        We will continue to search until we reach a quiet position.
        """
        if self.stop or self.checkTime():
            return 0

        # Dont search higher than MAX_PLY
        if ply >= MAX_PLY:
            return Evaluation.evaluate(self.board)

        # staticEval
        bestValue = Evaluation.evaluate(self.board)

        if bestValue >= beta:
            return bestValue

        if bestValue > alpha:
            alpha = bestValue

        # Sort the moves, the highest score should come first,
        # to reduce the size of the search tree
        moves = sorted(
            self.board.generate_legal_captures(),
            key=lambda move: self.scoreQMove(move),
            reverse=True,
        )

        # Loop over all legal captures
        for move in moves:
            self.nodes += 1

            captured = self.board.piece_type_at(move.to_square)

            # Delta Pruning
            if (
                piece_values[captured] + 400 + bestValue < alpha
                and not move.promotion
            ):
                continue

            # Make move
            self.board.push(move)

            score = -self.qsearch(-beta, -alpha, ply + 1)

            # Unmake move
            self.board.pop()

            # We found a new best value
            if score > bestValue:
                bestValue = score

                if score > alpha:
                    alpha = score

                    if score >= beta:
                        break

        return bestValue

    def absearch(self, alpha: int, beta: int, depth: int, ply: int) -> int:
        """
        Alpha Beta Search, this is the main search function.
        It searches the tree recursively and returns the best score.
        This function will be called with increasing depth until
        the time limit is reached or the maximum depth is reached.
        """
        if self.checkTime():
            return 0

        # Dont search higher than MAX_PLY
        if ply >= MAX_PLY:
            return Evaluation.evaluate(self.board)

        self.pvLength[ply] = ply
        RootNode = ply == 0
        hashKey = self.getHash()

        if not RootNode:
            if self.isRepetition(hashKey):
                # slight draw bias
                return -5

            if self.board.halfmove_clock >= 100:
                return 0

            # Mate distance pruning
            alpha = max(alpha, mated_in(ply))
            beta = min(beta, mate_in(ply + 1))
            if alpha >= beta:
                return alpha

        # Jump into qsearch
        if depth <= 0:
            return self.qsearch(alpha, beta, ply)

        # Transposition Table probing
        tte = self.transposition_table.probeEntry(hashKey)
        ttHit = hashKey == tte.key
        ttMove = tte.move if ttHit else chess.Move.null()

        # Adjust score
        ttScore = (
            self.transposition_table.scoreFromTT(tte.score, ply)
            if ttHit
            else VALUE_NONE
        )

        if not RootNode and tte.depth >= depth and ttHit:
            if tte.flag == Flag.LOWERBOUND:
                alpha = max(alpha, ttScore)
            elif tte.flag == Flag.UPPERBOUND:
                beta = min(beta, ttScore)

            if alpha >= beta:
                return ttScore

        inCheck = self.board.is_check()

        # Null move pruning
        if depth >= 3 and not inCheck:
            self.board.push(chess.Move.null())

            score = -self.absearch(-beta, -beta + 1, depth - 2, ply + 1)

            self.board.pop()

            if score >= beta:
                if score >= VALUE_TB_WIN_IN_MAX_PLY:
                    score = beta

                return score

        oldAlpha = alpha
        bestScore = -VALUE_INFINITE
        bestMove = chess.Move.null()
        madeMoves = 0

        # Sort the moves, the highest score should come first
        # The ttMove should be first one searched, incase we have a hit
        moves = sorted(
            self.board.legal_moves,
            key=lambda move: self.scoreMove(move, ttMove),
            reverse=True,
        )

        for move in moves:
            madeMoves += 1
            self.nodes += 1

            # Make move
            self.board.push(move)
            self.hashHistory.append(hashKey)

            # Search
            score = -self.absearch(-beta, -alpha, depth - 1, ply + 1)

            # Unmake move
            self.board.pop()
            self.hashHistory.pop()

            if score > bestScore:
                bestScore = score
                bestMove = move

                # update PV
                self.pvTable[ply][ply] = move

                for i in range(ply + 1, self.pvLength[ply + 1]):
                    self.pvTable[ply][i] = self.pvTable[ply + 1][i]

                self.pvLength[ply] = self.pvLength[ply + 1]

                if score > alpha:
                    # update alpha!
                    alpha = score

                    if score >= beta:
                        # update history
                        if not self.board.is_capture(move):
                            bonus = depth * depth
                            hhBonus = (
                                bonus
                                - self.htable[self.board.turn][move.from_square][
                                    move.to_square
                                ]
                                * abs(bonus)
                                / 16384
                            )

                            self.htable[self.board.turn][move.from_square][
                                move.to_square
                            ] += hhBonus
                        break

        # No moves were played so its checkmate or stalemate
        # Instead checking if the position is a checkmate during evaluate we can do it here
        # and save computation time
        if madeMoves == 0:
            if inCheck:
                return mated_in(ply)
            else:
                return 0

        # Calculate bound and save position in TT
        bound = Flag.NONEBOUND

        if bestScore >= beta:
            bound = Flag.LOWERBOUND
        else:
            if alpha != oldAlpha:
                bound = Flag.EXACTBOUND
            else:
                bound = Flag.UPPERBOUND

        if not self.checkTime():
            # Store in TT
            self.transposition_table.storeEntry(
                hashKey, depth, bound, bestScore, bestMove, ply
            )

        return bestScore

    def iterativeDeepening(self) -> None:
        """
        Iterative Deepening, this will call the absearch function
        with increasing depth until the time limit is reached or
        the maximum depth is reached.
        """
        self.nodes = 0

        score = -VALUE_INFINITE
        bestmove = chess.Move.null()

        # Start measuring time
        self.t0 = time.time_ns()

        # Iterative Deepening Loop
        for d in range(1, self.limit.limited["depth"] + 1):
            score = self.absearch(-VALUE_INFINITE, VALUE_INFINITE, d, 0)

            # Dont use completed depths result
            if self.stop or self.checkTime(True):
                break

            # Save bestmove
            bestmove = self.pvTable[0][0]

            # print info
            now = time.time_ns()
            stdout.write(self.stats(d, score, now - self.t0) + "\n")
            stdout.flush()

        # last attempt to get a bestmove
        if bestmove == chess.Move.null():
            bestmove = self.pvTable[0][0]

        # print bestmove, as per UCI Protocol
        stdout.write("bestmove " + str(bestmove) + "\n")
        stdout.flush()

    # Detect a repetition
    def isRepetition(self, key: int, draw: int = 1) -> bool:
        count = 0
        size = len(self.hashHistory)

        for i in range(size - 1, -1, -2):
            if i >= size - self.board.halfmove_clock:
                if self.hashHistory[i] == key:
                    count += 1
                if count == draw:
                    return True

        return False

    # Most Valuable Victim - Least Valuable Aggressor
    def mvvlva(self, move: chess.Move) -> int:
        mvvlva: list[list[int]] = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 105, 104, 103, 102, 101, 100],
            [0, 205, 204, 203, 202, 201, 200],
            [0, 305, 304, 303, 302, 301, 300],
            [0, 405, 404, 403, 402, 401, 400],
            [0, 505, 504, 503, 502, 501, 500],
            [0, 605, 604, 603, 602, 601, 600],
        ]

        from_square = move.from_square
        to_square = move.to_square
        attacker = self.board.piece_type_at(from_square)
        victim = self.board.piece_type_at(to_square)

        # En passant
        if victim is None:
            victim = 1
        return mvvlva[victim][attacker]

    # assign a score to moves in qsearch
    def scoreQMove(self, move: chess.Move) -> int:
        return self.mvvlva(move)

    # assign a score to normal moves
    def scoreMove(self, move: chess.Move, ttMove: chess.Move) -> int:
        if move == ttMove:
            return 1_000_000
        elif self.board.is_capture(move):
            # make sure captures are ordered higher than quiets
            return 32_000 + self.mvvlva(move)
        return self.htable[self.board.turn][move.from_square][move.to_square]

    def getHash(self) -> int:
        return chess.polyglot.zobrist_hash(self.board)

    def checkTime(self, iter: bool = False) -> bool:
        if self.stop:
            return True

        if (
            self.limit.limited["nodes"] != 0
            and self.nodes >= self.limit.limited["nodes"]
        ):
            return True

        if self.checks > 0 and not iter:
            self.checks -= 1
            return False

        self.checks = CHECK_RATE

        if self.limit.limited["time"] == 0:
            return False

        timeNow = time.time_ns()
        if (timeNow - self.t0) / 1_000_000 > self.limit.limited["time"]:
            return True

        return False

    # Build PV
    def getPV(self) -> str:
        pv = ""

        for i in range(0, self.pvLength[0]):
            pv += " " + str(self.pvTable[0][i])

        return pv

    # Convert mate scores
    def convert_score(self, score: int) -> str:
        if score >= VALUE_MATE_IN_PLY:
            return "mate " + str(
                ((VALUE_MATE - score) // 2) + ((VALUE_MATE - score) & 1)
            )
        elif score <= VALUE_MATED_IN_PLY:
            return "mate " + str(
                -((VALUE_MATE + score) // 2) + ((VALUE_MATE + score) & 1)
            )
        else:
            return "cp " + str(score)

    # Print UCI Info
    def stats(self, depth: int, score: int, time: int) -> str:
        time_in_ms = int(time / 1_000_000)
        time_in_seconds = max(1, time_in_ms / 1_000)
        info = (
            "info depth "
            + str(depth)
            + " score "
            + str(self.convert_score(score))
            + " nodes "
            + str(self.nodes)
            + " nps "
            + str(int(self.nodes / time_in_seconds))
            + " time "
            + str(round(time / 1_000_000))
            + " pv"
            + self.getPV()
        )
        return info

    # Reset search stuff
    def reset(self) -> None:
        self.pvLength[0] = 0
        self.nodes = 0
        self.t0 = 0
        self.stop = False
        self.checks = CHECK_RATE
        self.hashHistory = []
        self.htable = [[[0 for x in range(64)] for y in range(64)] for z in range(2)]


