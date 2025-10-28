import time, os, random
import numpy as np
import pandas as pd

from pyfastchess import Board as FastBoard, Evaluator, terminal_value_white_pov
from chessbot.psqt import build_weights, jostle_weights
from chessbot.utils import random_init
import chess

CLIP_MAX = 2000
MATE_CP = 2100

def score_clipped(x):
    return np.clip(x.score(mate_score=MATE_CP), -CLIP_MAX, CLIP_MAX)


def score_cp_white_pov(pov_score, clipped=True):
    scr = pov_score.white()
    return score_clipped(scr) if clipped else scr.score(mate_score=MATE_CP)


def sf_eval(b):
    cb = chess.Board(b.fen())
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
        info = eng.analyse(
            cb, info=chess.engine.INFO_ALL, limit=chess.engine.Limit(depth=10)
        )
    
    score = score_cp_white_pov(info['score'])
    return score
    
from chessbot import *
from chessbot.utils import *
from pprint import pprint
from chessbot.utils import random_init, random_board_setup

data = []
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
    for i in range(5000):
        if i % 1000 == 0:
            print("=== generating board", i, "===")
        if i % 2:
            wk = np.random.randint(0, 8)
            bk = 63 - np.random.randint(0, 8)
            #b = random_board_setup(12, wk, bk, queens=True)
            b = random_init(5 + i%30)
            
        else:
            b = random_init(10 + i%30)
        try:
            cb = chess.Board(b.fen())
        
            info = eng.analyse(
                cb, info=chess.engine.INFO_ALL, limit=chess.engine.Limit(depth=10)
            )
            
            score = score_cp_white_pov(info['score'])
            
            data.append((b, score))
        except:
            continue

res = []
res2 = []

ev = Evaluator()
ev.configure(build_weights())
qopts = {'max_qply': 7, 'qcaptures':24, "time_limit_ms": 4}
qopts2 = {'max_qply':5, 'qcaptures':24, "time_limit_ms": 3}

for b, score in data:
    s, qs = b.qsearch(-32_000, 32_000, ev, qopts) 
    s2, qs2 = b.qsearch(-32_000, 32_000, ev, qopts2) 
    res.append([score, s] + list(qs.values()))
    res2.append([score, s2] + list(qs2.values()))
        
df = pd.DataFrame(res, columns=['sf', 'ev'] + list(qs.keys()))
df2 = pd.DataFrame(res2, columns=['sf', 'ev'] + list(qs.keys()))
df.ev = np.clip(df.ev, -CLIP_MAX, CLIP_MAX)
df2.ev = np.clip(df2.ev, -CLIP_MAX, CLIP_MAX)

print("df", df.sf.corr(df.ev))
print("df2", df2.sf.corr(df2.ev))
print("df", df.time_used_ms.sum()/len(df), "ms per sample")
print("df2", df2.time_used_ms.sum()/len(df2), "ms per sample")


if __name__ == '__main__':
    b = FastBoard("r7/1ppk4/3p2q1/p2N1bnR/2Bn4/3P2P1/PPP2PP1/4RK2 w - - 1 2")
    s2, qs2 = b.qsearch(-32000, 32000, ev, qopts2)
    print(s2, qs2)
    print(sf_eval(b))
    print(ev.evaluate(b))
