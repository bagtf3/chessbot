from chessbot.utils import sf_eval, random_init, mirror_move
from chessbot import MODEL_DIR, SF_LOC
from chessbot.model import MaskedPolicyModel
from chessbot.review import make_fake_visits, make_training_sample
import chess, chess.engine

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import matplotlib.pyplot as plt

#%%
eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
eng.configure({"Threads": 1, "Hash": 64})
#%%

model_loc = MODEL_DIR + "conv_stm_pov_init.h5"
model = MaskedPolicyModel.from_saved(model_loc)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    policy_loss_weight=0.0,
    value_loss_weight=50.0
)


#%%

bonus_pool = []  # persistent across epochs
counter = 0
flips = 0
for epoch in range(1500):
    counter += 1
    if counter > 15:
        flips += 1
        counter = 0
        if flips % 2 == 0:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                policy_loss_weight=0.0,
                value_loss_weight=50.0
            )
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                policy_loss_weight=0.01,
                value_loss_weight=50.0
            )
    
    X, M, Y, P = [], [], [], []
    for rep in range(3000):
        moves = 8 + rep % 60
        rb = random_init(moves)
        cb = chess.Board(rb.fen())
        cb_turn = cb.turn
        if not cb_turn:
            cb = cb.mirror()
            
        sf_val, best = sf_eval(cb, depth=8, engine=eng)
        lms = rb.legal_moves()
        
        if not cb_turn:
            if not best:
                continue
            mv = chess.Move.from_uci(best)
            best = str(mirror_move(mv))
            
        if not best:
            x = rb.stacked_planes_stm_pov(1)
            mask = rb.legal_move_mask()
            policy = np.zeros_like(mask)
        else:
            visits = make_fake_visits(best, lms, ratio_best=25)
            tup = make_training_sample(rb, sf_val, visits)
            x, mask, policy = tup[0], tup[1], tup[2]

        X.append(x); M.append(mask); P.append(policy); Y.append(sf_val)

    Xstack = np.stack(X, axis=0) 
    Mstack = np.stack(M, axis=0)
    Ystack = np.stack(Y, axis=0)
    Pstack = np.stack(P, axis=0)
    
    order = np.argsort(Ystack)  # ascending
    n = order.size
    
    # if epoch % 40 == 0:
    #     # bottom 1024
    #     sel = order[:1024]
    # elif epoch % 4 == 1:
    #     # top 1024
    #     sel = order[-1024:]
    # if epoch % 2 == 0:
    #     # middle 1024
    #     start = (n - 1024) // 2
    #     sel = order[start:start+1024]
    # else:
    #     # tester: mixed/random 1024 from the whole set
    sel = np.random.choice(np.arange(n), size=2500, replace=False)

    Xsel = Xstack[sel]; Msel = Mstack[sel]; Psel = Pstack[sel]; Ysel = Ystack[sel]

    # extras -> bonus pool
    extras = np.setdiff1d(np.arange(Xstack.shape[0]), sel)
    for i in extras:
        bonus_pool.append((Xstack[i], Mstack[i], Pstack[i], Ystack[i]))

    # main eval/fit on selected 1024
    Ydict = {"value": Ysel.astype(np.float32), "policy": Psel}
    preds = model.predict([Xsel, Msel], verbose=0)
    value_preds = preds[1].ravel(); targets = np.asarray(Ysel).ravel()
    mse = np.mean((value_preds - targets)**2); corr = np.corrcoef(value_preds, targets)[0,1]
    print(f"epoch: {epoch} mse: {mse:.4f} corr: {corr:.4f}")
    print(f"mean of Y: {Ysel.mean():.3f}, mean of preds: {value_preds.mean():.3f}")
    plt.scatter(targets, value_preds, s=6)
    plt.plot([-1, 1], [-1, 1], linestyle="--", color="red", alpha=0.6)
    plt.xlim(-1, 1); plt.ylim(-1, 1); plt.gca()
    plt.xlabel("target"); plt.ylabel("pred"); plt.title("pred vs target"); plt.show()

    model.fit([Xsel, Msel], Ydict, epochs=5, batch_size=512, verbose=1)

    # if bonus pool full, train on first 1024 from it and drop them
    if len(bonus_pool) >= 1024:
        print("Draining bonus pool")
        bp = bonus_pool[:1024]
        Xb = np.stack([t[0] for t in bp], axis=0)
        Mb = np.stack([t[1] for t in bp], axis=0)
        Pb = np.stack([t[2] for t in bp], axis=0)
        Yb = np.stack([t[3] for t in bp], axis=0)
        model.fit(
            [Xb, Mb], {"value": Yb.astype(np.float32), "policy": Pb},
            epochs=3, batch_size=512, verbose=1
        )
        
        bonus_pool = bonus_pool[1024:]


model.save(MODEL_DIR + "conv_stm_pov_pre_trained.h5")
