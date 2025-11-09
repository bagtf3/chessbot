from chessbot.utils import sf_eval, random_init, mirror_move, format_time
from chessbot import MODEL_DIR, SF_LOC
from chessbot.review import make_fake_visits, make_training_sample
import chess, chess.engine

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# token ids
BASE = 9
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING_NO_CASTLE = 6
KING_KS_ONLY = 7
KING_QS_ONLY = 8
KING_BOTH = 9
EP_POSSIBLE = 19


def get_king_type(color, b):
    ksq = chess.E1 if color == chess.WHITE else chess.E8
    if b.piece_at(ksq) != chess.Piece(chess.KING, color):
        return KING_NO_CASTLE
    
    else:
        ks = b.has_kingside_castling_rights(color)
        qs = b.has_queenside_castling_rights(color)
        if ks & qs:
            return KING_BOTH
        if ks:
            return KING_KS_ONLY
        if qs:
            return KING_QS_ONLY
    return KING_NO_CASTLE

    
def board_to_64_tokens(board):
    """
    board: python-chess Board-like object
    returns: np.int16 array shape (64,) of token ids, STM made white
    """
    
    arr = np.zeros(64, dtype = np.int16)
    b = board if board.turn else board.mirror()
    
    wht_king = get_king_type(chess.WHITE, b)
    blk_king = get_king_type(chess.BLACK, b)
    
    if b.ep_square is not None:
        arr[b.ep_square] = EP_POSSIBLE
    
    pm = b.piece_map()
    for sq, p in pm.items():
        base = 0 if p.color else BASE
        
        if p.piece_type != chess.KING:
            arr[sq] = base + int(p.piece_type)
        
        else:
            king_val = wht_king if p.color else blk_king
            arr[sq] = base + king_val
            
    return arr


def build_transformer_64pv(
    d_model=256, n_heads=8, n_layers=5, ff_dim=None, proj_dim=48,
    dropout=0.05, vocab_size=21, name="transformer_64pv"
):
    if ff_dim is None:
        ff_dim = d_model * 2

    enc_in = Input(shape=(64,), dtype="int32", name="enc_in")
    legal_in = Input(shape=(4096,), dtype="int32", name="legal_mask")

    token_emb = layers.Embedding(
        input_dim=vocab_size, output_dim=d_model, name="token_emb"
    )(enc_in)

    pos_indices = tf.range(start=0, limit=64, delta=1)
    
    pos_emb = layers.Embedding(
        input_dim=64, output_dim=d_model, name="pos_emb")(pos_indices)
    
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    x = token_emb + pos_emb

    for i in range(n_layers):
        # pre-LN transformer block
        ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x)
        att = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            name=f"mha_{i}")(ln1, ln1, ln1)
        
        att = layers.Dropout(dropout)(att)
        x = layers.Add()([x, att])
        
        ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x)
        ff = layers.Dense(ff_dim, activation="gelu", name=f"ff1_{i}")(ln2)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(d_model, name=f"ff2_{i}")(ff)
        ff = layers.Dropout(dropout)(ff)
        x = layers.Add()([x, ff])

    square_feats = x  # (B,64,d)

    def legal_summaries(mask_tensor, **kwargs):
        # mask_tensor may be int32; cast to float32 first
        m = tf.reshape(tf.cast(mask_tensor, tf.float32), (-1, 64, 64))
        from_count = tf.reduce_sum(m, axis=2)
        from_has = tf.cast(from_count > 0.0, tf.float32)
        from_count_norm = tf.math.log1p(from_count) / tf.math.log(64.0)
        return tf.expand_dims(from_has, axis=2), tf.expand_dims(
            from_count_norm, axis=2
        )


    from_has, from_count_norm = layers.Lambda(
        legal_summaries, name="from_legal_feats")(legal_in)

    # append legal features to each square token
    from_has_tiled = tf.cast(from_has, square_feats.dtype)
    from_count_tiled = tf.cast(from_count_norm, square_feats.dtype)
    square_aug = layers.Concatenate(axis=2, name="square_aug")(
        [square_feats, from_has_tiled, from_count_tiled]
    )

    from_proj = layers.Dense(proj_dim, name="from_proj")(square_aug)
    to_proj = layers.Dense(proj_dim, name="to_proj")(square_aug)

    pair_scores = layers.Lambda(
        lambda t: tf.einsum('bif,bjf->bij', t[0], t[1]),
        name="pair_scores")([from_proj, to_proj]
    )

    # per-from and per-to biases
    b_from = layers.Dense(1, name="b_from")(from_proj)
    b_to = layers.Dense(1, name="b_to")(to_proj)
    b_from = tf.squeeze(b_from, axis=2)[:, :, None]
    b_to = tf.squeeze(b_to, axis=2)[:, None, :]

    pair_scores = pair_scores + b_from + b_to

    policy_logits_raw = layers.Reshape((4096,), name="policy_logits_raw")(pair_scores)

    # hard mask inside model (do masking in float32 to avoid fp16 issues)
    def apply_mask(args):
        logits, mask = args
        logits32 = tf.cast(logits, tf.float32)
        mask32 = tf.cast(mask, tf.float32)
        big_neg = tf.constant(-1e6, dtype=tf.float32)
        masked = tf.where(mask32 > 0.5, logits32, big_neg)
        return tf.cast(masked, logits.dtype)

    policy_logits = layers.Lambda(apply_mask, name="policy_logits")(
        [policy_logits_raw, legal_in]
    )

    # value head
    v_pool = layers.GlobalAveragePooling1D(name="v_gap")(x)
    v = layers.Dense(d_model // 2, activation="relu", name="v_fc1")(v_pool)
    v = layers.Dense(d_model // 4, activation="relu", name="v_fc2")(v)
    value_out = layers.Dense(1, activation="tanh", name="value_out")(v)

    model = Model(inputs=[enc_in, legal_in],
                  outputs=[policy_logits, value_out],
                  name=name)
    return model


# model config (tweak as you like)
model = build_transformer_64pv(
    d_model=288, n_heads=12, n_layers=12, ff_dim=None, proj_dim=64, dropout=0.05,
    vocab_size=21, name="t64pv"
)

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
policy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# compile: policy loss maps to first output, value loss to second
model.compile(
    optimizer=opt,
    loss={"policy_logits": policy_loss, "value_out": "mse"},
    loss_weights={"policy_logits": 1.0, "value_out": 5.0},
)

model.summary()  # prints layer table
#model.save(MODEL_DIR + "transformer_large_init.h5")

#%%
eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
eng.configure({"Threads": 1, "Hash": 64})
#%%
# load weights saved inside the .h5 file
#model.load_weights(MODEL_DIR + "transformer_large_init.h5")


weights_schedule = {
    0:   {"policy_logits": 1.00, "value_out": 5.00},
    50:  {"policy_logits": 1.20, "value_out": 3.50},
    100: {"policy_logits": 1.40, "value_out": 3.25},
    150: {"policy_logits": 1.60, "value_out": 3.00},
    200: {"policy_logits": 1.80, "value_out": 2.75},
    250: {"policy_logits": 2.00, "value_out": 2.50},
    300: {"policy_logits": 2.20, "value_out": 2.25},
    350: {"policy_logits": 2.00, "value_out": 2.00},
    400: {"policy_logits": 2.00, "value_out": 1.50},
    450: {"policy_logits": 1.80, "value_out": 1.00},
    500: {"policy_logits": 1.80, "value_out": 0.50},
    # post-500 fine-tune taper (MSE maintenance mostly)
    600: {"policy_logits": 1.60, "value_out": 0.50},
    800: {"policy_logits": 1.50, "value_out": 0.50},
    1000:{"policy_logits": 1.00, "value_out": 0.50},
}

model.compile(
    optimizer=opt,
    loss={"policy_logits": policy_loss, "value_out": "mse"},
    loss_weights=weights_schedule[0]
)


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



def apply_weights_for_epoch(epoch, model, opt, policy_loss):
    p_w, v_w = weights_for_epoch(epoch, weights_schedule)
    model.compile(
        optimizer=opt,
        loss={"policy_logits": policy_loss, "value_out": "mse"},
        loss_weights={"policy_logits": p_w, "value_out": v_w},
    )
    
    print(f"[epoch {epoch:4d}] [weight update] policy={p_w} value={v_w}")
    return model
    
#%%
epoch=0
begin = time.time()

metrics_history = {
    "policy_ce": [], "ce_uniform": [], "ce_gain": [],
    "exp_prob_model": [], "exp_prob_uniform": [],
    "value_mse": [], "value_corr": [], "epoch_time": []
}

eps = 1e-12
big_neg = -1e6

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
    return s.rolling(window, center=False, min_periods=1).mean().values


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
while epoch <= 1000:
    if epoch in weights_schedule.keys():
        model = apply_weights_for_epoch(epoch, model, opt, policy_loss)
    
    epoch_start = time.time()
    X, M, Y, P = [], [], [], []
    for rep in range(1536):
        moves = 2 + rep % 60
        rb = random_init(moves)
        cb = chess.Board(rb.fen())
        cb_turn = cb.turn
        if not cb_turn:
            cb = cb.mirror()
        
        x = board_to_64_tokens(cb)
        sf_val, best = sf_eval(cb, depth=12, engine=eng)
        
        if not best:
            continue
        
        lms = rb.legal_moves()
        
        if not cb_turn:
            mv = chess.Move.from_uci(best)
            best = str(mirror_move(mv))

        visits = make_fake_visits(best, lms, ratio_best=51)
        tup = make_training_sample(rb, sf_val, visits)
        mask, policy = tup[1], tup[2]
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
          

    hide_first = max(int(0.1*epoch), 5)       # drop first N raw points for visualization
    if (epoch > hide_first) and (epoch % 5 == 0):
        ma_window = min(int(epoch*0.2), 15)
        x = np.arange(len(metrics_history["policy_ce"]))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
        # policy CE
        ax = axes[0, 0]
        if len(x) > hide_first:
            ax.plot(x[hide_first:], metrics_history["policy_ce"][hide_first:],
                    label="policy_ce", alpha=0.6)
        ma = moving_average_pd(metrics_history["policy_ce"], window=ma_window)[hide_first:]
        if ma.size:
            # align MA to center of window
            offset = (ma_window - 1) // 2
            ax.plot(np.arange(len(ma)) + offset, ma, label=f"MA{ma_window}", lw=2)
        ax.set_title("policy CE (nats)")
        ax.legend()
    
        # CE gain vs uniform
        ax = axes[0, 1]
        if len(x) > hide_first:
            ax.plot(x[hide_first:], metrics_history["ce_gain"][hide_first:],
                    label="ce_gain", alpha=0.6)
        ma = moving_average_pd(metrics_history["ce_gain"], window=ma_window)[hide_first:]
        if ma.size:
            offset = (ma_window - 1) // 2
            ax.plot(np.arange(len(ma)) + offset, ma, label=f"MA{ma_window}", lw=2)
        ax.set_title("CE gain vs uniform")
        ax.legend()
    
        # value mse
        ax = axes[1, 0]
        if len(x) > hide_first:
            ax.plot(x[hide_first:], metrics_history["value_mse"][hide_first:],
                    label="mse", alpha=0.6)
        ma = moving_average_pd(metrics_history["value_mse"], window=ma_window)[hide_first:]
        if ma.size:
            offset = (ma_window - 1) // 2
            ax.plot(np.arange(len(ma)) + offset, ma, label=f"MA{ma_window}", lw=2)
        ax.set_title("value MSE")
        ax.legend()
    
        # value corr
        ax = axes[1, 1]
        if len(x) > hide_first:
            ax.plot(x[hide_first:], metrics_history["value_corr"][hide_first:],
                    label="corr", alpha=0.6)
            
        ma = moving_average_pd(metrics_history["value_corr"], window=ma_window)[hide_first:]
        if ma.size:
            offset = (ma_window - 1) // 2
            ax.plot(np.arange(len(ma)) + offset, ma, label=f"MA{ma_window}", lw=2)
        ax.set_title("value corr")
        ax.legend()
    
        plt.tight_layout()
        plt.show()

    # main eval/fit on selected 1024
    Ydict = {"value_out": Ystack.astype(np.float32), "policy_logits": Pstack}
    print("-"*100)
    history = model.fit([Xstack, Mstack], Ydict, epochs=4, batch_size=128, verbose=0)
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
        
    if epoch % 100 == 0:
        model.save(MODEL_DIR + f"transformer_large_pretrained{epoch}.h5")
    
    epoch_time = time.time() - epoch_start
    metrics_history['epoch_time'].append(epoch_time)
    e_time = format_time(epoch_time)
    runtime = format_time(time.time() - begin)
    avg_epoch = format_time(np.mean(metrics_history['epoch_time']))
    
    print("-"*100)
    print(f"[time check] last epoch: {e_time}, avg epoch: {avg_epoch},  "
          f"total runtime: {runtime}")
    print()
    epoch += 1
