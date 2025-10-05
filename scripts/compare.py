import chess
from chessbot.disservin import Search as DSearch
from pyfastchess import Board as FastBoard, Evaluator
from chessbot.psqt import build_weights
from chessbot.absearch import GAUnit
from chessbot.utils import show_board, random_init


def make_gaunit(depth=5, time=20):
    ev = Evaluator()
    ev.configure(build_weights(zeros=False))
    g = GAUnit(ev, depth=depth, time=time)
    g.max_qply = 5
    g.qcaptures = 16
    return g


rb = random_init(15)

board = chess.Board(rb.fen())
show_board(board, flipped=True)
search = DSearch(board)

search.limit.limited["depth"] = 3
search.iterativeDeepening()

my_search = make_gaunit()
fb = FastBoard(board.fen())
best, score, info = my_search.search(fb, max_depth=7, verbose=True)
print(fb.san(best), score)



# prepare qopts dict to pass to C++


# some of these positions are wicked
boards = [random_init(10 + (i % 10)) for i in range(10000)]

ev = Evaluator()
ev.configure(build_weights(zeros=False))

qopts = {"max_qply": 4, "max_qcaptures": 16, "time_limit_ms": 2}
start = time.time()
for b in boards:
    score, qstats = b.qsearch(-32_000, 32_000, ev, qopts)
stop = time.time()
tot = stop - start
print(f"Total Time: {tot:.4f}")
print(f"Boards/sec: {len(boards)/tot:.4f}")

qopts = {"max_qply": 3, "max_qcaptures": 8, "time_limit_ms": 2}
start = time.time()
for b in boards:
    score, qstats = b.qsearch(-32_000, 32_000, ev, qopts)
stop = time.time()
tot = stop - start
print(f"Total Time: {tot:.4f}")
print(f"Boards/sec: {len(boards)/tot:.4f}")

qopts = {"max_qply": 2, "max_qcaptures": 3, "time_limit_ms": 2}
start = time.time()
for b in boards:
    score, qstats = b.qsearch(-32_000, 32_000, ev, qopts)
stop = time.time()
tot = stop - start
print(f"Total Time: {tot:.4f}")
print(f"Boards/sec: {len(boards)/tot:.4f}")
#%%
import chess
from pyfastchess import Board as FastBoard, Evaluator
from chessbot.psqt import build_weights
from chessbot.absearch import GAUnit
from math import isclose

def symmetry_check(fen, depths=(0,1,2,3), qplys=(0,1,2,3,4), zeros=True):
    ev = Evaluator(); ev.configure(build_weights(zeros=zeros))
    print("fen:", fen)
    for d in depths:
        for q in qplys:
            # make GAUnit configured the way you actually run it:
            u = GAUnit(ev, depth=d)
            u.max_qply = q
            fb_w = FastBoard(fen)         # white to move
            fb_b = FastBoard(fen)
            # flip side
            cb = chess.Board(fen)
            cb.turn = not cb.turn
            fb_b = FastBoard(cb.fen())

            sw, pvw = u.absearch_root(fb_w, d, -32000, 32000)
            sb, pvb = u.absearch_root(fb_b, d, -32000, 32000)

            # we expect sb ≈ -sw
            ok = isclose(sb, -sw, rel_tol=1e-6, abs_tol=1)
            print(f"d={d} q={q}  sw={sw:6} sb={sb:6}  sym_ok={ok}")
        print()
    print("done")

# run on startpos and on a few suspect fens
symmetry_check(chess.STARTING_FEN, depths=range(0,4), qplys=range(0,5))


from chessbot.search_utils import TranspositionTable

def leaf_diagnostics(fen, d, q):
    ev = Evaluator(); ev.configure(build_weights(zeros=False))
    u = GAUnit(ev, depth=d)
    u.max_qply = q

    fb = FastBoard(fen)
    # 1) static evaluate_itemized (white POV)
    item = ev.evaluate_itemized(fb)
    static = ev.evaluate(fb)
    print("static total:", static, "itemized:", item)

    # 2) qsearch directly
    qopts = {"max_qply": getattr(u, "max_qply", None),
             "max_qcaptures": getattr(u, "max_qcaptures", None)}
    
    score, qstats = fb.qsearch(-32000, 32000, ev, qopts)
    print("qsearch score:", score, "qstats:", qstats)

    # 3) call absearch which will call qsearch at depth <= 0
    s = u.absearch(fb, 0, -32000, 32000, 0)
    print("absearch(depth=0) ->", s)

    # 4) TT probe/store: test storing then probing on flipped stm
    tt = TranspositionTable()
    key = fb.hash()
    tt.store_entry(key, depth=1, flag=0, score=1234, move="0000", ply=0)
    peek = tt.probe_entry(key)
    print("TT probe same stm:", peek)
    # flip side and check TT reuse (if your TT implementation stores stm it should differ)
    cb = chess.Board(fb.fen()); cb.turn = not cb.turn
    fb2 = FastBoard(cb.fen())
    peek2 = tt.probe_entry(fb2.hash())
    print("TT probe flipped stm:", peek2)

leaf_diagnostics(chess.STARTING_FEN, d=0, q=3)


class NullTT:
    def probe_entry(self, key): return None
    def store_entry(self, *a, **kw): pass
    def clear(self): pass

# in your GAUnit instance:
u = GAUnit(ev, depth=2)
u.tt = NullTT()
# run quick symmetry_check above with u and compare results.

def symmetry_check(fen, depths=(0,1,2,3), qplys=(0,1,2,3,4), zeros=True):
    ev = Evaluator(); ev.configure(build_weights(zeros=zeros))
    print("fen:", fen)
    for d in depths:
        for q in qplys:
            # make GAUnit configured the way you actually run it:
            u = GAUnit(ev, depth=d)
            u.tt = NullTT()
            u.max_qply = q
            fb_w = FastBoard(fen)         # white to move
            fb_b = FastBoard(fen)
            # flip side
            cb = chess.Board(fen)
            cb.turn = not cb.turn
            fb_b = FastBoard(cb.fen())

            sw, pvw = u.absearch_root(fb_w, d, -32000, 32000)
            sb, pvb = u.absearch_root(fb_b, d, -32000, 32000)

            # we expect sb ≈ -sw
            ok = isclose(sb, -sw, rel_tol=1e-6, abs_tol=1)
            print(f"d={d} q={q}  sw={sw:6} sb={sb:6}  sym_ok={ok}")
        print()
    print("done")

# run on startpos and on a few suspect fens
symmetry_check(chess.STARTING_FEN, depths=range(0,4), qplys=range(0,5))