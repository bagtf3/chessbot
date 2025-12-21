import os, pickle
from chessbot import MODEL_DIR
#from chessbot.model import load_model
from chessbot.utils import batch_policy_metrics, print_validation, format_time
from chessbot.review import GameViewer, load_game_index, ANALYZE_PKL
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

bl = 12
fl = 296
MODEL_NAME = f'conv64_{bl}x{fl}'
from chessbot.model import build_conv_flat_64x67SE
model, opt, loss_weights, loss_dict = build_conv_flat_64x67SE(
    d_model_embed=128, vocab_size=21, filters=fl,
    n_blocks=bl, name=MODEL_NAME
)

model.summary()
model.save(MODEL_DIR + f"{MODEL_NAME}_init.h5")
#%%
metrics_history = {"value_mse": [], "value_corr": [], "epoch_time": []}

eps = 1e-12
big_neg = -1e6

def moving_average_pd(arr, window=15):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().values

#%%
SP_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs"
run_tags = [
    'conv_1000_selfplay_phase3','conv_1000_selfplay_phase4',
    "conv_net_flat_12blocks_run0",
    "conv_net_flat_run1", "conv_net_flat_run2"
]

all_games = []
df_list = []
for rt in run_tags:
    rd = os.path.join(SP_DIR, rt)
    all_games += load_game_index(rd)

    pkl = os.path.join(rd, ANALYZE_PKL)
    with open(pkl, "rb") as f:
        prev_run = pickle.load(f)
    
    df_all = prev_run['df_all']
    df_means = prev_run['df_means']
    
    # tidy up CPL
    df_all['clipped_loss'] = np.clip(df_all['loss'], -1000, 1000)
    clipped_cpl = df_all.groupby("game_id")['clipped_loss'].mean()

    df_means['overall_cpl'] = df_means.game_id.map(clipped_cpl)
    df_means['run_tag'] = rt
    df_list.append(df_means)
    del df_all
    del df_means

df_trim = pd.concat(df_list).drop_duplicates(['game_id']).sort_values("ts")
df_trim = df_trim.query("scenario != 'random_endgame'").copy()
df_trim = df_trim.query("scenario != 'paired_validation'").copy()

meta = pd.DataFrame(all_games)
meta = meta.query("plies >= 10")
exclude = ['random_endgame', 'paired_validation']
meta = meta.query("scenario != @exclude")

meta = meta.merge(df_trim[['game_id', 'run_tag', 'overall_cpl']], on='game_id')
meta = meta.query("overall_cpl <= 50")

training_games = meta['json_file'].to_list()

random.shuffle(training_games)


def append_to_buffer(buf, X, M, P, Z, V):
    y = [0.5 * a + 0.5 * b for a, b in zip(Z, V)]
    buf += list(zip(X, M, P, y))


def top_up_sliding_buffer(
    buf,
    training_games,
    idx,
    buffer_size,
    draw_rate=0.5,
):
    while len(buf) < buffer_size and idx < len(training_games):
        game = training_games[idx]
        idx += 1

        gv = GameViewer(game, sf_df=None)

        if gv.result == 0:
            if np.random.random() > draw_rate:
                continue

        if len(gv.moves_uci) < 10:
            continue

        X, M, P, Z, V, R = gv.generate_training_data(sf_skip=False)
        if not X:
            continue

        append_to_buffer(buf, X, M, P, Z, V)

    return idx


def sample_from_sliding_buffer(buf, epoch_size):
    n = len(buf)
    if n == 0:
        return None

    k = epoch_size if n >= epoch_size else n

    Xs = []
    Ms = []
    Ps = []
    Ys = []

    for _ in range(k):
        j = np.random.randint(len(buf))
        x, m, p, y = buf[j]
        Xs.append(x)
        Ms.append(m)
        Ps.append(p)
        Ys.append(y)

        buf[j] = buf[-1]
        buf.pop()

    Xb = np.stack(Xs, axis=0)
    Mb = np.stack(Ms, axis=0)
    Pb = np.stack(Ps, axis=0)
    Yb = np.stack(Ys, axis=0)

    return Xb, Mb, Pb, Yb


buffer = []
batch_size = 512
epoch_size = 20*batch_size
buffer_size = 5 * epoch_size
epoch, idx = 0, 0
MAX_EPOCH = 250
draw_rate = 0.5
begin = time.time()
loss_weights = {"policy_logits": 2.0, "value_out": 2.00}

idx = top_up_sliding_buffer(
    buffer,
    training_games,
    idx,
    buffer_size=buffer_size,
    draw_rate=draw_rate
)

#%%
while idx < len(training_games) and epoch <= MAX_EPOCH:
    if epoch > 10:
        loss_weights = {"policy_logits": 1.0, "value_out": 1.0}
    if epoch > 100:
        loss_weights = {"policy_logits": 0.5, "value_out": 0.5}
    if epoch > 150:
        loss_weights = {"policy_logits": 0.25, "value_out": 0.25}
    if epoch > 200:
        loss_weights = {"policy_logits": 0.12, "value_out": 0.12}
        
    epoch_start = time.time()
    
    idx = top_up_sliding_buffer(
        buffer,
        training_games,
        idx,
        buffer_size=buffer_size,
        draw_rate=draw_rate
    )

    
    batch = sample_from_sliding_buffer(buffer, epoch_size)
    if batch is None:
        break

    Xstack, Mstack, Pstack, Ystack = batch
    
    # pred validate and train
    preds = model.predict(Xstack, verbose=0, batch_size=batch_size)
    
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
        Xstack, Ydict, epochs=1, batch_size=batch_size, verbose=0, sample_weight=s_wts
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
    if epoch % 25 == 0:    
        model.save(MODEL_DIR + f"{MODEL_NAME}_bootstrapped_latest.h5")
# when done
model.save(MODEL_DIR + f"{MODEL_NAME}_bootstrapped.h5")
#with open(MODEL_DIR + f"{MODEL_NAME}_train_metrics_{epoch}.pkl", "wb") as f:
#    pickle.dump(metrics_history, f, protocol=pickle.HIGHEST_PROTOCOL)
