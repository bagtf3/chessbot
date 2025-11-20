from chessbot import MODEL_DIR, SF_LOC

from chessbot.utils import sf_eval, random_init, format_time
from chessbot.utils import GameGenerator, batch_policy_metrics, print_validation

from chessbot.review import make_fake_visits
import chess, chess.engine

from chessbot.model import build_conv_64pv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

MODEL_NAME = 'conv_64_token_12M'

model, opt, loss_weights, loss_dict = build_conv_64pv(
    d_model_embed=128, vocab_size=21, filters=196,
    n_conv=16, proj_dim=96, name=MODEL_NAME
)

model.summary()
model.save(MODEL_DIR + f"{MODEL_NAME}_{0}.h5")
#%%

eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
eng.configure({"Threads": 1, "Hash": 64})
    
#%%
metrics_history = {
    "value_mse": [], "value_corr": [], "epoch_time": []
}

weights_schedule = {
    # slow start
    5  : {"policy_logits": 0.1, "value_out": 0.10},
    10 : {"policy_logits": 0.5, "value_out": 2.00},
    20 : {"policy_logits": 1.0, "value_out": 2.00},
    75 : {"policy_logits": 1.5, "value_out": 2.50},
    150: {"policy_logits": 2.0, "value_out": 2.75},
    300: {"policy_logits": 2.5, "value_out": 2.25},
    400: {"policy_logits": 2.0, "value_out": 1.50},
    500: {"policy_logits": 1.5, "value_out": 0.50},
    # post-500 fine-tune taper (MSE maintenance mostly)
    600: {"policy_logits": 1.0, "value_out": 0.50}
}

eps = 1e-12
big_neg = -1e6

def get_sf_depth(epoch):
    sf_schedule = {100: 8, 200: 9, 400: 10, 600: 11, np.inf: 12}
    for e in sorted(sf_schedule.keys()):
        if epoch < e:
            return sf_schedule[e]


def moving_average_pd(arr, window=15):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().values


#%%
epoch=0
model.save(MODEL_DIR + f"{MODEL_NAME}_{epoch}.h5")
begin = time.time()
#%%

from chessbot.config import Config
gg = GameGenerator(Config())

while epoch < 100:
    for k in sorted(weights_schedule.keys()):
        if epoch < k:
            loss_weights = weights_schedule[k]
            break
    
    epoch_start = time.time()
    X, M, Y, P = [], [], [], []
    for rep in range(1536):
        if rep % 10 == 10:
            rb, _ = gg.new_board("random_endgame")
        elif rep % 21 == 20:
            rb, _ = gg.new_board("pre_opened")
        else:
            moves = 2 + rep % 60
            rb = random_init(moves)
        
        x = rb.encode_64_tokens()
        mask = rb.legal_move_mask()
        
        sf_val, best = sf_eval(rb, depth=get_sf_depth(epoch), engine=eng)
        
        if not best:
            continue
        
        lms = rb.legal_moves()
        visits = make_fake_visits(best, lms, ratio_best=51)
        
        counts = np.array([x[1] for x in visits], dtype=np.float32)
        s = counts.sum()
        pi = (counts / s) if s > 0.0 else None
        
        # get indices from C++
        indices = rb.moves_to_indices(lms)  # list of ints (0..4288)
        policy = np.zeros(64 * 67, dtype=np.float32)

        # accumulate probs into flattened policy
        for idx, p in zip(indices, pi):
            policy[idx] += p

        X.append(x); M.append(mask); P.append(policy); Y.append(sf_val)

    Xstack = np.stack(X, axis=0) 
    Mstack = np.stack(M, axis=0)
    Ystack = np.stack(Y, axis=0)
    Pstack = np.stack(P, axis=0)
    
    # pred validate and train
    preds = model.predict(Xstack, verbose=0)
    
    value_preds = preds[1].ravel(); targets = np.asarray(Ystack).ravel()
    plt.scatter(targets, value_preds, s=6)
    plt.plot([-1, 1], [-1, 1], linestyle="--", color="red", alpha=0.6)
    plt.xlim(-1, 1); plt.ylim(-1, 1); plt.gca()
    plt.xlabel("target"); plt.ylabel("pred"); plt.title("pred vs target"); plt.show()
    
    policy_logits = preds[0]  # (B,4096)
    value_preds = preds[1].ravel()  # (B,)
    
    policy_stats = batch_policy_metrics(policy_logits, Pstack, Mstack)
    
    targets = np.asarray(Ystack).ravel()
    value_mse = np.mean((value_preds - targets) ** 2)
    value_corr = np.corrcoef(value_preds, targets)[0, 1]
    
    for k, v in policy_stats.items():
        if k not in metrics_history:
            metrics_history[k] = []
        
        metrics_history[k].append(v)
    
    metrics_history['value_mse'].append(value_mse)
    metrics_history['value_corr'].append(value_corr)
    
    # build print dict and call the printer
    print_metrics = {"value_mse": value_mse, "value_corr": value_corr}
    print_metrics.update(policy_stats)
    print_validation(epoch, print_metrics)

    hide_first = max(int(0.1 * epoch), 5)
    if (epoch > hide_first) and (epoch % 5 == 0):
        ma_window = min(max(3, int(epoch * 0.2)), 15)
        if ma_window % 2 == 0:
            ma_window += 1
    
        x = np.arange(len(metrics_history["policy_ce"]))
        start = hide_first
        xs = x[start:]
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
        # policy CE
        ax = axes[0, 0]
        raw = np.array(metrics_history["policy_ce"])[start:]
        ma = moving_average_pd(metrics_history["policy_ce"], window=ma_window)[start:]
        if raw.size:
            ax.plot(xs, raw, label="policy_ce", alpha=0.6, lw=1)
        if ma.size:
            ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
        ax.set_title("policy CE (nats)")
        ax.legend()
    
        # CE gain vs uniform
        ax = axes[0, 1]
        raw = np.array(metrics_history["ce_gain"])[start:]
        ma = moving_average_pd(metrics_history["ce_gain"], window=ma_window)[start:]
        if raw.size:
            ax.plot(xs, raw, label="ce_gain", alpha=0.6, lw=1)
        if ma.size:
            ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
        ax.set_title("CE gain vs uniform")
        ax.legend()
    
        # value MSE
        ax = axes[1, 0]
        raw = np.array(metrics_history["value_mse"])[start:]
        ma = moving_average_pd(metrics_history["value_mse"], window=ma_window)[start:]
        if raw.size:
            ax.plot(xs, raw, label="mse", alpha=0.6, lw=1)
        if ma.size:
            ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
        ax.set_title("value MSE")
        ax.legend()
    
        # value corr
        ax = axes[1, 1]
        raw = np.array(metrics_history["value_corr"])[start:]
        ma = moving_average_pd(metrics_history["value_corr"], window=ma_window)[start:]
        if raw.size:
            ax.plot(xs, raw, label="corr", alpha=0.6, lw=1)
        if ma.size:
            ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
        ax.set_title("value corr")
        ax.legend()
    
        plt.tight_layout()
        plt.show()

    # main eval/fit on selected 1024
    Ydict = {"value_out": Ystack.astype(np.float32), "policy_logits": Pstack}
    weights = np.ones_like(Ystack)
    s_wts = {k: weights*loss_weights[k] for k in Ydict.keys()}
    print("-"*100)
    history = model.fit(
        Xstack, Ydict, epochs=4, batch_size=512, verbose=0, sample_weight=s_wts
    )
    
    rows = []
    for m, v in history.history.items():
        name = "total" if m == "loss" else m.replace("_loss", "")
        start = v[0]; end = v[-1]
        delta = start - end
        mark = "*" if delta < 0 else "+"
        rows.append((name, start, end, delta, mark))
    
    name_w = max(len(r[0]) for r in rows)
    num_w = 8   # width for numbers (including decimal point)
    ETAG = f"[epoch {epoch:4d}]"
    fmt = (f"{ETAG} [model fit] "
           f"{{name:<{name_w}}} : value: {{start:{num_w}.4f}} -> "
           f"{{end:{num_w}.4f}}  delta: {{delta:{num_w}.4f}} {{mark}}")

    for name, start, end, delta, mark in rows:
        print(fmt.format(name=name, start=start, end=end, delta=delta, mark=mark))
    
    epoch += 1
    if epoch in [10, 75, 100, 200, 500, 1000]:
        model.save(MODEL_DIR + f"{MODEL_NAME}_{epoch}.h5")
        with open(MODEL_DIR + f"{MODEL_NAME}_train_metrics_{epoch}.pkl", "wb") as f:
            pickle.dump(metrics_history, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    epoch_time = time.time() - epoch_start
    metrics_history['epoch_time'].append(epoch_time)
    e_time = format_time(epoch_time)
    runtime = format_time(time.time() - begin)
    avg_epoch = format_time(np.mean(metrics_history['epoch_time']))
    
    print("-"*100)
    print(f"[time check] last epoch: {e_time}, avg epoch: {avg_epoch},  "
          f"total runtime: {runtime}")
    print()
    
model.save(MODEL_DIR + f"{MODEL_NAME}_{epoch}.h5")
with open(MODEL_DIR + f"{MODEL_NAME}_train_metrics_{epoch}.pkl", "wb") as f:
    pickle.dump(metrics_history, f, protocol=pickle.HIGHEST_PROTOCOL)
#%%
from chessbot.model import make_conv_infer
import time
import numpy as np
import pandas as pd
import gc

def make_random_batch(B, vocab_size=64, move_space=4288, legal_prob=0.02):
    """Create random enc (B,64) and legal mask (B,4288) with >=1 legal move per row."""
    enc = np.random.randint(0, vocab_size, size=(B, 64), dtype=np.int32)
    # sparse random legal mask
    legal = (np.random.rand(B, move_space) < legal_prob).astype(np.int32)
    # ensure at least one legal slot per row
    idx = np.random.randint(0, move_space, size=(B,))
    legal[np.arange(B), idx] = 1
    return enc, legal

def speed_test_infer(
        infer, batch_sizes=(32, 64, 128, 256, 512, 1024), reps=20,
        warmup=3, rng_seed=12345, verbose=True):
    """
    Run timing for given batch sizes.

    Parameters
    - infer: your inference function `infer((enc_np, legal_np)) -> (probs, val)`
    - batch_sizes: iterable of ints
    - reps: number of timed repetitions per batch size
    - warmup: number of warmup calls before timing
    - rng_seed: reproducible random generator seed (optional)
    - verbose: print progress

    Returns
    - pandas.DataFrame with columns: batch, total_s, per_batch_s, samples_per_s, median_batch_s, std_batch_s
    """
    np.random.seed(rng_seed)
    results = []

    for B in batch_sizes:
        enc_np, legal_np = make_random_batch(B)
        # GC + optional small pause to stabilise memory effects
        gc.collect()

        if verbose:
            print(f"Warmup: batch={B}, warmup={warmup} ...")

        # warmup runs (un-timed)
        for _ in range(warmup):
            _ = infer((enc_np, legal_np))

        # timed runs: store per-run times to compute median/std if wanted
        times = []
        if verbose:
            print(f"Timing: batch={B}, reps={reps} ...")

        for r in range(reps):
            t0 = time.perf_counter()
            _ = infer((enc_np, legal_np))
            t1 = time.perf_counter()
            times.append(t1 - t0)

        total = sum(times)
        per_batch = total / len(times)
        samples_per_s = B / per_batch if per_batch > 0 else float("inf")

        results.append({
            "batch": B,
            "total_s": total,
            "per_batch_s": per_batch,
            "samples_per_s": samples_per_s,
            "median_batch_s": float(np.median(times)),
            "std_batch_s": float(np.std(times, ddof=1))
        })

        if verbose:
            print(f" -> B={B:4d} | per-batch {per_batch:.6f}s | {samples_per_s:8.1f} samples/s")

    df = pd.DataFrame(results).sort_values("batch")
    print("\nSummary:")
    print(df.to_string(index=False))
    return df

infer = make_conv_infer(model, max_bs=1024, min_p=0.001, max_p=0.65, temp=1.0)
df = speed_test_infer(
    infer, batch_sizes=(32, 64, 128, 256, 512, 1024),
    reps=20, warmup=3, verbose=True
)
