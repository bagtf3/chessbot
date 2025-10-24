import time, os, random
import numpy as np

from pyfastchess import Board as FastBoard, Evaluator, terminal_value_white_pov
from chessbot.psqt import build_weights, jostle_weights
from chessbot.utils import random_init


import chess


ev = Evaluator()
ev.configure(build_weights())
qopts = {'max_qply': 32, 'qcaptures':2048, "time_limit_ms": 3}

from chessbot import *
from chessbot.utils import *
from pprint import pprint

b = FastBoard(BLACK_WINS_IN_2_WHITE_MOVE.fen())
cb = chess.Board(b.fen())

print(f"{b.fen()}, fens agree?: {b.fen() == cb.fen()}")
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
    info = eng.analyse(
        cb, info=chess.engine.INFO_ALL, limit=chess.engine.Limit(depth=20)
    )
    
pprint(info)
pv = info['pv']
in_check = cb.is_check()
for p in pv:
    qflag = False
    if in_check:
        print(f"{str(p)} is an evasion.")
        qflag = True
        
    if cb.is_capture(p):
        print(f"{str(p)} is a capture.")
        qflag = True
        
    if cb.gives_check(p):
        print(f"{str(p)} gives_check.")
        qflag = True
    
    if not qflag:
        print(f"{str(p)} is a quiet move. qsearch ends.")
        break
        
    cb.push(p)
    in_check = cb.is_check()
    if cb.is_game_over():
        print("result", cb.result())

qopts = {'max_qply': 7, 'qcaptures':64, "time_limit_ms": 32}
s, qs = b.qsearch(-32000, 32000, ev, qopts)
print(s, qs)


#push the first move in the sequence
b.push_uci(str(pv[0]))
s, qs = b.qsearch(-32000, 32000, ev, qopts)
print(s, qs)
# now it finds it
