from chessbot.utils import random_init
from chessbot.looper import ChessGame

from pyfastchess import Board, prior_engine_details


game = ChessGame(board=Board())
tree = game.tree


import time
start = time.time()
nn, nt, nc = tree.collect_many_leaves(12000, 5200)
stop = time.time()

print(f"{stop-start:<.3}")
print(nn, nt, nc)

tree.root_child_visits()



pv = tree.principal_variation()
for p in pv:
    print(p.uci, p.visits, p.Q)
    
tree.root_child_visits()


batch = tree.pending_encoded(5)
len(batch)

z, p = batch[0]


import pyfastchess as pf
print(pf.prior_engine_details())          # shows current prior config (dict)


pf.configure_prior_engine({"anytime_uniform_mix": 0.6, 'use_prior_boosts': True, "fish_tacos": "1$"})
print(pf.prior_engine_details())


from chessbot.utils import random_init

b = random_init(25)

import numpy as np
import pyfastchess as pf

# assume pf.create_prior_engine() or pf.ensure_prior_engine() already called
# and you have a Board instance named `b` in scope.

legal = b.legal_moves()

# discover sizes (best-effort; adjust if your engine expects different shapes)
def _discover_dims(board):
    n_from = getattr(board, "num_squares", None) or getattr(board, "num_squares_", None) or 64
    n_to   = n_from
    n_piece = getattr(board, "num_piece_types", None) or getattr(board, "piece_type_count", None) or 6
    n_promo = getattr(board, "num_promo_types", None) or getattr(board, "promo_count", None) or 5
    return int(n_from), int(n_to), int(n_piece), int(n_promo)

n_from, n_to, n_piece, n_promo = _discover_dims(b)
print("using dims => from:", n_from, "to:", n_to, "piece:", n_piece, "promo:", n_promo)
print("len(legal) =", len(legal))

# create random raw logits (float32). These are factorized heads — no softmax here.
p_from = np.random.randn(n_from).astype(np.float32)
p_to   = np.random.randn(n_to).astype(np.float32)
p_piece= np.random.randn(n_piece).astype(np.float32)
p_promo= np.random.randn(n_promo).astype(np.float32)

try:
    priors = pf.prior_engine_build(b, legal, p_from, p_to, p_piece, p_promo)
except Exception as exc:
    print("prior_engine_build threw:", exc)
    print("Debug info: p_from.shape=", p_from.shape, "p_to.shape=", p_to.shape,
          "p_piece.shape=", p_piece.shape, "p_promo.shape=", p_promo.shape)
    raise

print("received", len(priors), "priors. sample (first 10):")
for uci, prob in priors[:10]:
    print(uci, prob)




import numpy as np, pyfastchess as pf
pf.ensure_prior_engine()
pf.raw_cache_clear()
batch = []
for i in range(5):
    k = 100000 + i
    batch.append((k,
                  np.random.uniform(),
                  np.random.randn(64).astype(np.float32),
                  np.random.randn(64).astype(np.float32),
                  np.random.randn(6).astype(np.float32),
                  np.random.randn(5).astype(np.float32)))
pf.raw_cache_bulk_insert(batch)
print(pf.raw_cache_stats())


import numpy as np, pyfastchess as pf

pf.ensure_prior_engine()
pf.raw_cache_clear()

key = 55555
v = 0.0
arr1 = np.zeros(64, dtype=np.float32)     # all zeros
arr2 = np.ones(64, dtype=np.float32)      # all ones
p_piece = np.arange(6, dtype=np.float32)
p_promo = np.arange(5, dtype=np.float32)

pf.raw_cache_bulk_insert([(key, v, arr1, arr1, p_piece, p_promo)])   # insert first
print("after 1st insert:", pf.raw_cache_stats())

pf.raw_cache_bulk_insert([(key, v, arr2, arr2, p_piece, p_promo)])   # overwrite same key
print("after overwrite:", pf.raw_cache_stats())

# Expect size unchanged (should be 1)

#####


import numpy as np, time, pyfastchess as pf

pf.ensure_prior_engine()
pf.raw_cache_clear()

N = 25000
batch = []
for i in range(N):
    k = 200000 + i
    v = np.random.uniform()
    pf_from = np.random.randn(64).astype(np.float32)
    pf_to   = np.random.randn(64).astype(np.float32)
    pf_piece= np.random.randn(6).astype(np.float32)
    pf_promo= np.random.randn(5).astype(np.float32)
    batch.append((k, v, pf_from, pf_to, pf_piece, pf_promo))

t0 = time.perf_counter()
pf.raw_cache_bulk_insert(batch)
t1 = time.perf_counter()
print(f"Inserted {N} entries in {t1-t0:.3f}s -> {N/(t1-t0):.0f} inserts/sec")
print("stats:", pf.raw_cache_stats())


####
import numpy as np, time, pyfastchess as pf

pf.ensure_prior_engine()
pf.raw_cache_clear()

M = 17000
batch = []
for i in range(M):
    k = 300000 + i
    v = 0.0
    pf_from = np.full(64, float(i % 256), dtype=np.float32)  # deterministic-ish
    pf_to   = np.full(64, float((i+7) % 256), dtype=np.float32)
    pf_piece= np.full(6, float(i % 6), dtype=np.float32)
    pf_promo= np.full(5, float(i % 5), dtype=np.float32)
    batch.append((k, v, pf_from, pf_to, pf_piece, pf_promo))

pf.raw_cache_bulk_insert(batch)
s = pf.raw_cache_stats()
print("after heavy insert:", s)
# Expect s['evictions'] > 0 and s['size'] <= s['capacity']
assert s["evictions"] > 0
assert s["size"] <= s["capacity"]
print("eviction test passed.")
######
import numpy as np, pyfastchess as pf

pf.ensure_prior_engine()

for round in range(5):
    pf.raw_cache_clear()
    batch = []
    for i in range(200):
        k = 400000 + round*1000 + i
        batch.append((k,
                      0.25, 
                      np.random.randn(64).astype(np.float32),
                      np.random.randn(64).astype(np.float32),
                      np.random.randn(6).astype(np.float32),
                      np.random.randn(5).astype(np.float32)))
    pf.raw_cache_bulk_insert(batch)
    print(f"round {round}: ", pf.raw_cache_stats())
    
    
#%%

if __name__ == '__main__':
    from chessbot.utils import random_init
    from chessbot.looper import ChessGame
    
    from pyfastchess import Board, prior_engine_details, raw_cache_clear, priors_cache_clear
    
    
    game = ChessGame(board=Board())
    tree = game.tree
    
    
    import time
    raw_cache_clear()
    priors_cache_clear()
    time.sleep(1.0)
    start = time.time()
    nn, nt, nc = tree.collect_many_leaves(200, 500)
    stop = time.time()
    print(f"collected 200 pending in {stop-start:<.3}")
    pn = tree.pending_nodes_
        
    batch = []
    for p in pn:
        k = p.zobrist
        v = 0.0
        pf_from = np.full(64, float(i % 256), dtype=np.float32)  # deterministic-ish
        pf_to   = np.full(64, float((i+7) % 256), dtype=np.float32)
        pf_piece= np.full(6, float(i % 6), dtype=np.float32)
        pf_promo= np.full(5, float(i % 5), dtype=np.float32)
        batch.append((k, v, pf_from, pf_to, pf_piece, pf_promo))
    
    pf.raw_cache_bulk_insert(batch)
    print(len(tree.pending_nodes_))
    start = time.time()
    tree.resolve_pending()
    while len(tree.pending_nodes_):
        pass
    stop = time.time()
    print(f"applied 200 pending in {stop-start:<.3}")
    
    
    game = ChessGame(board=Board())
    tree = game.tree
    
    raw_cache_clear()
    priors_cache_clear()
    time.sleep(1.0)
    
    start = time.time()
    nn, nt, nc = tree.collect_many_leaves(5000, 500)
    stop = time.time()
    print(f"collected 10000 pending in {stop-start:<.3}")
    pn = tree.pending_nodes_
        
    batch = []
    for p in pn:
        k = p.zobrist
        v = 0.0
        pf_from = np.full(64, float(i % 256), dtype=np.float32)  # deterministic-ish
        pf_to   = np.full(64, float((i+7) % 256), dtype=np.float32)
        pf_piece= np.full(6, float(i % 6), dtype=np.float32)
        pf_promo= np.full(5, float(i % 5), dtype=np.float32)
        batch.append((k, v, pf_from, pf_to, pf_piece, pf_promo))
    
    raw_cache_clear()
    priors_cache_clear()
    from pyfastchess import raw_cache_stats
    uniq = set([z for z, b in tree.pending_encoded()])
    rcs = raw_cache_stats()
    n = len(uniq)
    start = time.time()
    pf.raw_cache_bulk_insert(batch)
    while rcs['size'] < n:
        rcs = raw_cache_stats()
    stop = time.time()
    print(f"batch appened {n} raw pending in {stop-start:<.5}")
    

    
    start = time.time()
    tree.resolve_pending()
    while len(tree.pending_nodes_):
        pass
    stop = time.time()
    print(f"applied {n} pending in {stop-start:<.3}")


