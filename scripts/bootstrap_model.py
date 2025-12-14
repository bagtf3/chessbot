import os
from chessbot import MODEL_DIR
from chessbot.model import load_model
from chessbot.utils import batch_policy_metrics, print_validation, format_time
from chessbot.review import GameViewer, load_game_index

from collections import defaultdict
import random

from chessbot.model import build_conv_flat_64x67

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

bl = 12
fl = 512
MODEL_NAME = f'conv64_{bl}x{fl}'
    
# model, opt, loss_weights, loss_dict = build_conv_flat_64x67(
#     d_model_embed=128, vocab_size=21, filters=fl,
#     n_blocks=bl, name=MODEL_NAME
# )

# model.summary()
#model.save(MODEL_DIR + f"{MODEL_NAME}_{0}.h5")
#model = load_model(MODEL_DIR + "conv_12x512SE_0.h5")
#%%
metrics_history = {
    "value_mse": [], "value_corr": [], "epoch_time": []
}

weights_schedule = {
    # slow start
    3  : {"policy_logits": 1.0, "value_out": 1.00},
    5  : {"policy_logits": 1.0, "value_out": 1.50},
    20 : {"policy_logits": 1.5, "value_out": 2.00},
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


epoch, idx = 0, 0
MAX_EPOCH = 5000
draw_rate = 0.5
begin = time.time()
#%%
SP_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs"
run_tags = [
    'conv_1000_selfplay_phase3','conv_1000_selfplay_phase4',
    "conv_net_flat_12blocks_run0",
    "conv_net_flat_run1", "conv_net_flat_run2"
]

all_games = []
for rt in run_tags:
    rd = os.path.join(SP_DIR, rt)
    all_games += load_game_index(rd)

random.shuffle(all_games)
thresh = 10000
while idx < len(all_games) and epoch < MAX_EPOCH:
    for k in sorted(weights_schedule.keys()):
        if epoch < k:
            loss_weights = weights_schedule[k]
            break
    
    epoch_start = time.time()
    buffer = defaultdict(list)
    while len(buffer['Z']) < thresh:
        if idx >= len(all_games):
            break
        
        game = all_games[idx]
        idx += 1
        gv = GameViewer(game['json_file'], sf_df=None)
        
        if gv.result == 0:
            if np.random.random() > draw_rate:
                continue
        if len(gv.moves_uci) < 10:
            continue
            
        X, M, P, Z, V, R = gv.generate_training_data(sf_skip=False)
        if X:
            buffer['X'] += X
            buffer['M'] += M
            buffer['P'] += P
            buffer['Z'] += Z
            buffer['V'] += V
            buffer['R'] += R
    
    y = [0.5*a + 0.5*b for a, b in zip(buffer['Z'], buffer['V'])]
    Xstack = np.stack(buffer['X'], axis=0) 
    Mstack = np.stack(buffer['M'], axis=0)
    Pstack = np.stack(buffer['P'], axis=0)
    Ystack = np.stack(y, axis=0)
    
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
        Xstack, Ydict, epochs=2, batch_size=256, verbose=0, sample_weight=s_wts
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
    #if epoch in [10, 75, 100, 200, 500, 1000]:
    #    model.save(MODEL_DIR + f"{MODEL_NAME}_{epoch}.h5")
    #    with open(MODEL_DIR + f"{MODEL_NAME}_train_metrics_{epoch}.pkl", "wb") as f:
    #        pickle.dump(metrics_history, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    epoch_time = time.time() - epoch_start
    metrics_history['epoch_time'].append(epoch_time)
    e_time = format_time(epoch_time)
    runtime = format_time(time.time() - begin)
    avg_epoch = format_time(np.mean(metrics_history['epoch_time']))
    
    print("-"*100)
    print(f"[time check] last epoch: {e_time}, avg epoch: {avg_epoch},  "
          f"total runtime: {runtime}")
    print()
    
model.save(MODEL_DIR + f"{MODEL_NAME}_bootstrapped_{epoch}.h5")
#with open(MODEL_DIR + f"{MODEL_NAME}_train_metrics_{epoch}.pkl", "wb") as f:
#    pickle.dump(metrics_history, f, protocol=pickle.HIGHEST_PROTOCOL)
