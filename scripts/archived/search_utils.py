from enum import Enum

# Helpers
MAX_PLY = 60
CHECK_RATE = 256

VALUE_INFINITE = 32001
VALUE_NONE = 32002
VALUE_MATE = 32000
VALUE_MATE_IN_PLY = VALUE_MATE - MAX_PLY
VALUE_MATED_IN_PLY = -VALUE_MATE_IN_PLY

bound_flag = Enum("Flag", ["NONEBOUND", "UPPERBOUND", "LOWERBOUND", "EXACTBOUND"])


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

        
class TranspositionTable:
    """
    Lightweight TT. Uses power-of-two table, two-slot bucket, simple
    replace-by-depth/age. Stores tuples per slot, no TEntry class.
    """

    def __init__(self, bits=19):
        self.size = 1 << bits
        self.mask = self.size - 1
        self.table = [None] * self.size
        self.age = 0

    def tt_index(self, key):
        return key & self.mask

    def probe_entry(self, key):
        i = self.tt_index(key)
        s = self.table[i]
        # hit
        if s and s[0] == key:
            return {
                "key": s[0], "depth": s[1], "flag": s[2],
                "score": s[3], "move_uci": s[4]
            }
        
        s2 = self.table[(i + 1) & self.mask]
        # try the next one down it no hit
        if s2 and s2[0] == key:
            return {
                "key": s2[0], "depth": s2[1], "flag": s2[2],
                "score": s2[3], "move_uci": s2[4]
            }
        
        # both missed
        return None

    def store_entry(self, key, depth, flag, score, move, ply):
        """
        key: 64-bit zobrist-like key
        depth: search depth
        flag: bound_flag enum value
        score: raw score (centipawns or mate encoding)
        move: move UCI string
        ply: current search ply (for mate distance encoding)
        """
        i = self.tt_index(key)
        i2 = (i + 1) & self.mask

        s = self.table[i]
        s2 = self.table[i2]

        self.age += 1
        adj = self.score_to_TT(score, ply)
        entry = (key, depth, flag, adj, move, self.age)

        # empty slot
        if s is None:
            self.table[i] = entry
            return

        # same key in primary slot -> replace per rules
        if s[0] == key:
            # update move if changed
            self.table[i] = self.replace_if_better(s, entry)
            return

        # empty secondary -> place there
        if s2 is None:
            self.table[i2] = entry
            return

        # same key in secondary -> replace per rules
        if s2[0] == key:
            self.table[i2] = self.replace_if_better(s2, entry)
            return

        # both occupied, evict shallower or older
        d0 = s[1]
        d1 = s2[1]
        if d0 < d1:
            self.table[i] = entry
            return
        if d1 < d0:
            self.table[i2] = entry
            return

        # depths equal, evict older
        if s[5] <= s2[5]:
            self.table[i] = entry
        else:
            self.table[i2] = entry

    def replace_if_better(self, old, new):
        # old and new are tuples (key, depth, flag, score, move, age)
        # follow original logic: replace if flag is exact or new depth
        # deeper (depth + 2 > old.depth) or different move
        if new[2] == bound_flag.EXACTBOUND:
            return new
        if new[1] + 2 > old[1]:
            return new
        # if move changed, update move but keep other fields
        if old[4] != new[4]:
            return (old[0], old[1], new[2], new[3], new[4], new[5])
        # otherwise keep old
        return old

    def score_to_TT(self, s, plies):
        # adjust mate distances so mate scores encode distance-to-mate
        # positive mate (win) threshold
        if s >= VALUE_MATE_IN_PLY:
            return s + plies
        # negative mate (loss) threshold
        if s <= -VALUE_MATE_IN_PLY:
            return s - plies
        return s

    def score_from_TT(self, s, plies):
        if s >= VALUE_MATE_IN_PLY:
            return s - plies
        if s <= -VALUE_MATE_IN_PLY:
            return s + plies
        return s

    def clear(self):
        self.table = [None] * self.size
        self.age = 0
