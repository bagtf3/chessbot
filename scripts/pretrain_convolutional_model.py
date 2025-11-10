from chessbot import MODEL_DIR, SF_LOC

from chessbot.utils import sf_eval, random_init, mirror_move, format_time
from chessbot.utils import GameGenerator
from chessbot.model import save_transformer_model as save_custom_model

from chessbot.review import make_fake_visits
import chess, chess.engine

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)


def build_conv_64pv(
    d_model_embed=128,
    vocab_size=21,
    filters=256,
    n_conv=12,
    proj_dim=64,
    name="conv_64pv",
):
    DE, FL, VS = d_model_embed, filters, vocab_size
    
    # inputs
    enc_in = Input(shape=(64,), dtype="int32", name="enc_in")

    # embedding -> (B,64,128) -> reshape (B,8,8,128)
    tok_emb = layers.Embedding(input_dim=VS, output_dim=DE, name="token_emb")(enc_in)
    
    x = layers.Reshape((8, 8, DE), name="to_2d")(tok_emb)

    # initial conv to filters
    x = layers.Conv2D(FL, 3, padding="same", activation="relu", name="conv_init")(x)

    # main conv stack (repeated 3x3 convs)
    for i in range(n_conv):
        y = layers.Conv2D(FL, 3, padding="same", activation=None, name=f"conv_{i}")(x)
        y = layers.LeakyReLU(alpha=0.01, name=f"lrelu_{i}")(y)
        
        # residual
        x = layers.Add(name=f"res_{i}")([x, y])

    # expand to 512 then mix
    x = layers.Conv2D(512, 1, padding="same", activation="relu", name="conv_expand")(x)
    x = layers.Conv2D(512, 3, padding="same", activation="relu", name="conv_mix")(x)

    # project back to FL (optional)
    x = layers.Conv2D(FL, 1, padding="same", activation="relu", name="conv_project")(x)

    # flatten to (B,64,channels)
    #b, h, w, c = None, 8, 8, FL
    feat = layers.Reshape((64, FL), name="to_64_x")(x)

    # pairwise policy: from/to projections + biases -> 64x64 -> 4096
    from_proj = layers.Dense(proj_dim, name="from_proj")(feat)
    to_proj = layers.Dense(proj_dim, name="to_proj")(feat)

    # from_proj, to_proj: (B, 64, F)
    pair_scores = tf.matmul(from_proj, to_proj, transpose_b=True)  # -> (B,64,64)

    # per-from and per-to biases
    b_from = layers.Dense(1, name="b_from")(from_proj)   # (B,64,1)
    # b_from is already (B,64,1) after Dense
    # ensure shape is explicit (B,64,1)
    b_from = layers.Reshape((64, 1), name="b_from_reshape")(b_from)

    b_to = layers.Dense(1, name="b_to")(to_proj)        # (B,64,1)
    # want (B,1,64) - use Permute to avoid lambdas
    b_to = layers.Permute((2, 1), name="b_to_permute")(b_to)  # -> (B,1,64)

    pair_scores = layers.Add(name="add_biases")([pair_scores, b_from, b_to])
    policy_logits = layers.Reshape((4096,), name="policy_logits")(pair_scores)

    # value head
    v_pool = layers.GlobalAveragePooling1D(name="v_gap")(feat)
    v = layers.Dense(FL // 2, activation='relu', name="v_fc1")(v_pool)
    v = layers.Dense(FL // 4, activation='relu', name="v_fc2")(v)
    value_out = layers.Dense(1, activation="tanh", name="value_out")(v)

    model = Model(inputs=[enc_in], outputs=[policy_logits, value_out], name=name)

    # compile: losses, weights, optimizer
    loss_weights = {"policy_logits": 1.0, "value_out": 2.0}
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out": "mse",
    }
    
    model.compile(optimizer=opt, loss=loss_dict, loss_weights=loss_weights)
    
    return model, opt, loss_weights, loss_dict

model, opt, loss_weights, loss_dict = build_conv_64pv()
model.summary()
#model.load_weights("C:/Users/Bryan/Data/chessbot_data/models/conv_embedded_pt1000.h5")
#model.save("C:/Users/Bryan/Data/chessbot_data/models/conv_64_token_pt1000.h5")
#%%

weights_schedule = {
    0:   {"policy_logits": 1.50, "value_out": 3.00},
    200: {"policy_logits": 2.00, "value_out": 2.75},
    300: {"policy_logits": 2.50, "value_out": 2.25},
    400: {"policy_logits": 2.00, "value_out": 1.50},
    500: {"policy_logits": 1.50, "value_out": 0.50},
    # post-500 fine-tune taper (MSE maintenance mostly)
    600: {"policy_logits": 1.00, "value_out": 0.50}
}


def weights_for_epoch(epoch, schedule):
    """Return (policy_w, value_w) for the largest key <= epoch."""
    keys = sorted(schedule.keys())
    chosen = keys[0]
    for k in keys:
        if k <= epoch:
            chosen = k
        else:
            break
    s = schedule[chosen]
    return s["policy_logits"], s["value_out"]


def apply_weights_for_epoch(epoch, model, opt, loss_dict):
    p_w, v_w = weights_for_epoch(epoch, weights_schedule)
    model.compile(optimizer=opt, loss=loss_dict, loss_weights=loss_weights)
    print(f"[epoch {epoch:4d}] [weight update] policy={p_w} value={v_w}")
    return model


model = apply_weights_for_epoch(0, model, opt, loss_dict)
#%%
eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
eng.configure({"Threads": 1, "Hash": 64})
    
#%%
metrics_history = {
    "policy_ce": [], "ce_uniform": [], "ce_gain": [],
    "exp_prob_model": [], "exp_prob_uniform": [],
    "value_mse": [], "value_corr": [], "epoch_time": []
}

eps = 1e-12
big_neg = -1e6

def get_sf_depth(epoch):
    sf_schedule = {100: 8, 200: 9, 400: 10, 600: 11, np.inf: 12}
    for e in sorted(sf_schedule.keys()):
        if epoch < e:
            return sf_schedule[e]
        

def stable_softmax(logits):
    # logits: (B,4096) float32
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=1, keepdims=True)
    return e / (s + 1e-20)


def moving_average(arr, window=15):
    if len(arr) < 1:
        return np.array([])
    w = np.ones(window) / window
    return np.convolve(arr, w, mode="valid")


def moving_average_pd(arr, window=15):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().values


def batch_policy_metrics(logits, labels, mask):
    masked_logits = np.where(mask > 0.5, logits, big_neg)
    probs = stable_softmax(masked_logits)

    per_ce = -np.sum(labels * np.log(probs + eps), axis=1)
    policy_ce = per_ce.mean()

    legal_counts = mask.sum(axis=1, keepdims=True)
    uniform = mask / (legal_counts + eps)
    per_ce_uniform = -np.sum(labels * np.log(uniform + eps), axis=1)
    uniform_ce = per_ce_uniform.mean()

    exp_prob_model = np.sum(labels * probs, axis=1).mean()
    exp_prob_uniform = np.sum(labels * uniform, axis=1).mean()

    true_best = labels.argmax(axis=1)
    model_best = probs.argmax(axis=1)

    top1_exact = (model_best == true_best).mean()

    top_probs = probs[np.arange(probs.shape[0]), model_best]
    avg_top_prob = top_probs.mean()

    prob_on_true_per = probs[np.arange(probs.shape[0]), true_best]
    prob_on_true = prob_on_true_per.mean()

    support_mask = labels > 0
    support_mask[np.arange(support_mask.shape[0]), true_best] = False
    prob_on_others_per = (probs * support_mask).sum(axis=1)
    prob_on_others = prob_on_others_per.mean()

    top1_in_support = (labels[np.arange(labels.shape[0]), model_best] > 0).mean()

    return {
        "policy_ce": policy_ce,
        "uniform_ce": uniform_ce,
        "ce_gain": (uniform_ce - policy_ce),
        "exp_prob_model": exp_prob_model,
        "exp_prob_uniform": exp_prob_uniform,
        "top1_in_support": top1_in_support,
        "top1_exact": top1_exact,
        "avg_top_prob": avg_top_prob,
        "prob_on_true": prob_on_true,
        "prob_on_others": prob_on_others
    }


def print_validation(epoch, stats):
    keys = [
        "val_mse", "val_corr",
        "policy_ce", "uniform_ce", "ce_gain",
        "top1_exact", "avg_top_prob",
        "prob_on_true", "prob_on_others"
    ]
    name_w = max(len(k) for k in keys)
    num_w = 8
    fmt_num = f"{{value:{num_w}.4f}}"
    def pair(k, v):
        return f"{k:<{name_w}}: {fmt_num.format(value=v)}"
    ratio = stats.get("prob_on_true", 0.0) / (stats.get("prob_on_others", 0.0) + eps)

    print(f"[epoch {epoch:4d}] [validation] {pair('val_mse', stats['val_mse'])}  "
          f"{pair('val_corr', stats['val_corr'])}")
    print(f"[epoch {epoch:4d}] [validation] {pair('policy_ce', stats['policy_ce'])}  "
          f"{pair('uniform_ce', stats['uniform_ce'])}  {pair('ce_gain', stats['ce_gain'])}")
    print(f"[epoch {epoch:4d}] [validation] {pair('top1_exact', stats['top1_exact'])}  "
          f"{pair('avg_top_prob', stats['avg_top_prob'])}")
    print(f"[epoch {epoch:4d}] [validation] {pair('prob_on_true', stats['prob_on_true'])}  "
          f"{pair('true ratio', ratio)}")

#%%
epoch=0
begin = time.time()
#%%
def mm(uci):
    return str(mirror_move(chess.Move.from_uci(uci)))

from chessbot.config import Config
gg = GameGenerator(Config())

while epoch <= 1000:
    if epoch in weights_schedule.keys():
        model = apply_weights_for_epoch(epoch, model)
    
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
        
        if rb.side_to_move() == 'b':
            visits = [[mm(u), c] for u, c in visits]
        
        counts = np.array([x[1] for x in visits], dtype=np.float32)
        s = counts.sum()
        pi = (counts / s) if s > 0.0 else None
        
        # get indices from C++
        indices = rb.moves_to_indices(lms)  # list of ints (0..4095)
        policy = np.zeros(64 * 64, dtype=np.float32)

        # accumulate probs into flattened policy
        for idx, p in zip(indices, pi):
            policy[idx] += p

        X.append(x); M.append(mask); P.append(policy); Y.append(sf_val)

    Xstack = np.stack(X, axis=0) 
    Mstack = np.stack(M, axis=0)
    Ystack = np.stack(Y, axis=0)
    Pstack = np.stack(P, axis=0)
    
    # pred validate and train
    preds = model.predict([Xstack, Mstack], verbose=0)
    
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
    
    # append metrics (use same key names batch_policy_metrics returns)
    metrics_history.setdefault("top1_exact", []).append(policy_stats["top1_exact"])
    metrics_history.setdefault("avg_top_prob", []).append(policy_stats["avg_top_prob"])
    metrics_history.setdefault("prob_on_true", []).append(policy_stats["prob_on_true"])
    
    # policy CE / uniform CE / gains (handle old key names as fallback)
    metrics_history.setdefault("policy_ce", []).append(policy_stats.get("policy_ce"))
    metrics_history.setdefault("ce_uniform", []).append(policy_stats.get("uniform_ce"))
    metrics_history.setdefault("ce_gain", []).append(policy_stats.get("ce_gain"))
    
    metrics_history.setdefault("exp_prob_model", []).append(
        policy_stats.get("exp_prob_model"))
    
    metrics_history.setdefault("exp_prob_uniform", []).append(
        policy_stats.get("exp_prob_uniform"))
    
    metrics_history.setdefault("value_mse", []).append(value_mse)
    metrics_history.setdefault("value_corr", []).append(value_corr)
    
    # build print dict and call the printer
    print_metrics = {"val_mse": value_mse, "val_corr": value_corr}
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
    print("-"*100)
    history = model.fit([Xstack, Mstack], Ydict, epochs=4, batch_size=512, verbose=0)
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
    if epoch in [0, 100, 200, 500, 1000] == 0:
        model_name = "conv_embedded_v2"
        model.save(MODEL_DIR + f"{model_name}_{epoch}.h5")
        with open(MODEL_DIR + f"{model_name}_train_metrics_{epoch}.pkl", "wb") as f:
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
#%%


