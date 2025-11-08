from chessbot.utils import sf_eval, random_init, mirror_move
from chessbot import MODEL_DIR, SF_LOC
from chessbot.model import MaskedPolicyModel
from chessbot.review import make_fake_visits, make_training_sample
import chess, chess.engine

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import matplotlib.pyplot as plt

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
    loss_weights={"policy_logits": 0.5, "value_out": 5.0},
)

model.summary()  # prints layer table
model.save(MODEL_DIR + "transformer_large_init.h5")

#%%
eng = chess.engine.SimpleEngine.popen_uci(SF_LOC)
eng.configure({"Threads": 1, "Hash": 64})
#%%
# load weights saved inside the .h5 file
#model.load_weights(MODEL_DIR + "transformer_model_init.h5")

model.compile(
    optimizer=opt,
    loss={"policy_logits": policy_loss, "value_out": "mse"},
    loss_weights={"policy_logits": 1.0, "value_out": 5.0},
)


#%%
metrics_history = {
    "policy_ce": [], "ce_uniform": [], "ce_gain": [],
    "exp_prob_model": [], "exp_prob_uniform": [],
    "value_mse": [], "value_corr": []
}

eps = 1e-12
big_neg = -1e6

def stable_softmax(logits):
    # logits: (B,4096) float32
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=1, keepdims=True)
    return e / (s + 1e-20)

def batch_policy_metrics(logits, labels, mask):
    logits = logits.astype(np.float32)
    mask = mask.astype(np.float32)
    labels = labels.astype(np.float32)

    assert (mask.sum(axis=1) > 0).all()

    masked_logits = np.where(mask > 0.5, logits, big_neg).astype(np.float32)
    probs = stable_softmax(masked_logits)

    per_ce = -np.sum(labels * np.log(probs + eps), axis=1)
    ce_model = per_ce.mean()

    legal_counts = mask.sum(axis=1, keepdims=True)
    uniform = (mask / (legal_counts + eps)).astype(np.float32)
    per_ce_uniform = -np.sum(labels * np.log(uniform + eps), axis=1)
    ce_uniform = per_ce_uniform.mean()

    exp_model = np.sum(labels * probs, axis=1).mean()
    exp_uniform = np.sum(labels * uniform, axis=1).mean()

    # new: exact top1 (model argmax equals true best index)
    true_best = labels.argmax(axis=1)
    model_best = probs.argmax(axis=1)
    top1_exact = float((model_best == true_best).mean())

    # average of model's top probability (how concentrated model is)
    top_probs = probs[np.arange(probs.shape[0]), model_best]
    avg_top_prob = float(top_probs.mean())

    # model prob on the true best index (useful with soft labels)
    prob_on_true = probs[np.arange(probs.shape[0]), true_best].mean()

    # old metrics retained
    top1_in_support = float(np.array([
        1.0 if labels[i, model_best[i]] > 0.0 else 0.0
        for i in range(labels.shape[0])
    ]).mean())

    return {
        "ce_model": float(ce_model),
        "ce_uniform": float(ce_uniform),
        "ce_gain": float(ce_uniform - ce_model),
        "exp_model": float(exp_model),
        "exp_uniform": float(exp_uniform),
        "top1_in_support": top1_in_support,
        "top1_exact": top1_exact,
        "avg_top_prob": avg_top_prob,
        "prob_on_true": float(prob_on_true)
    }

def moving_average(arr, window=15):
    if len(arr) < 1:
        return np.array([])
    w = np.ones(window) / window
    return np.convolve(arr, w, mode="valid")


for epoch in range(500):
    X, M, Y, P = [], [], [], []
    for rep in range(1536):
        moves = 8 + rep % 60
        rb = random_init(moves)
        cb = chess.Board(rb.fen())
        cb_turn = cb.turn
        if not cb_turn:
            cb = cb.mirror()
        
        x = board_to_64_tokens(cb)
        sf_val, best = sf_eval(cb, depth=8, engine=eng)
        
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
    policy_logits = preds[0]  # (B,4096)
    value_preds = preds[1].ravel()  # (B,)
    
    policy_stats = batch_policy_metrics(policy_logits, Pstack, Mstack)
    
    targets = np.asarray(Ystack).ravel()
    value_mse = float(np.mean((value_preds - targets) ** 2))
    value_corr = float(np.corrcoef(value_preds, targets)[0, 1])
    
    metrics_history.setdefault("top1_exact", []).append(policy_stats["top1_exact"])
    metrics_history.setdefault("avg_top_prob", []).append(policy_stats["avg_top_prob"])
    metrics_history.setdefault("prob_on_true", []).append(policy_stats["prob_on_true"])
    metrics_history["policy_ce"].append(policy_stats["ce_model"])
    metrics_history["ce_uniform"].append(policy_stats["ce_uniform"])
    metrics_history["ce_gain"].append(policy_stats["ce_gain"])
    metrics_history["exp_prob_model"].append(policy_stats["exp_model"])
    metrics_history["exp_prob_uniform"].append(policy_stats["exp_uniform"])
    metrics_history["value_mse"].append(value_mse)
    metrics_history["value_corr"].append(value_corr)
    
    print(f"[epoch {epoch:4d}] policy_ce: {policy_stats['ce_model']:.4f} "
          f"uniform_ce: {policy_stats['ce_uniform']:.4f} gain: "
          f"{policy_stats['ce_gain']:.4f}")
    print(f"[epoch {epoch:4d}] val_mse: {value_mse:.4f} val_corr"" {value_corr:.4f}")
    print(f"[epoch {epoch:4d}] top1_exact: {policy_stats['top1_exact']:.3f} "
          f"avg_top_prob: {policy_stats['avg_top_prob']:.3f} "
          f"prob_on_true: {policy_stats['prob_on_true']:.3f}")
    
    # every 5 epochs show plot with MA15
    if epoch % 5 == 0:
        x = np.arange(len(metrics_history["policy_ce"]))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        # policy CE
        ax = axes[0, 0]
        ax.plot(x, metrics_history["policy_ce"], label="policy_ce", alpha=0.6)
        ma = moving_average(metrics_history["policy_ce"], window=15)
        if ma.size:
            ax.plot(np.arange(len(ma)) + 14, ma, label="MA15", linewidth=2)
        ax.set_title("policy CE (nats)")
        ax.legend()
        # CE gain
        ax = axes[0, 1]
        ax.plot(x, metrics_history["ce_gain"], label="ce_gain", alpha=0.6)
        ma = moving_average(metrics_history["ce_gain"], window=15)
        if ma.size:
            ax.plot(np.arange(len(ma)) + 14, ma, label="MA15", linewidth=2)
        ax.set_title("CE gain vs uniform")
        ax.legend()
        # value mse
        ax = axes[1, 0]
        ax.plot(x, metrics_history["value_mse"], label="mse", alpha=0.6)
        ma = moving_average(metrics_history["value_mse"], window=15)
        if ma.size:
            ax.plot(np.arange(len(ma)) + 14, ma, label="MA15", linewidth=2)
        ax.set_title("value MSE")
        ax.legend()
        # value corr
        ax = axes[1, 1]
        ax.plot(x, metrics_history["value_corr"], label="corr", alpha=0.6)
        ma = moving_average(metrics_history["value_corr"], window=15)
        if ma.size:
            ax.plot(np.arange(len(ma)) + 14, ma, label="MA15", linewidth=2)
        ax.set_title("value corr")
        ax.legend()
        plt.tight_layout()
        plt.show()

    # main eval/fit on selected 1024
    Ydict = {"value_out": Ystack.astype(np.float32), "policy_logits": Pstack}
    preds = model.predict([Xstack, Mstack], verbose=0)
    value_preds = preds[1].ravel(); targets = np.asarray(Ystack).ravel()
    mse = np.mean((value_preds - targets)**2) 
    corr = np.corrcoef(value_preds, targets)[0,1]
    print(f"epoch: {epoch} mse: {mse:.4f} corr: {corr:.4f}")
    print(f"mean of Y: {Ystack.mean():.3f}, mean of preds: {value_preds.mean():.3f}")
    plt.scatter(targets, value_preds, s=6)
    plt.plot([-1, 1], [-1, 1], linestyle="--", color="red", alpha=0.6)
    plt.xlim(-1, 1); plt.ylim(-1, 1); plt.gca()
    plt.xlabel("target"); plt.ylabel("pred"); plt.title("pred vs target"); plt.show()

    model.fit([Xstack, Mstack], Ydict, epochs=4, batch_size=96, verbose=1)

model.save(MODEL_DIR + "transformer_model_pretrained0.h5")
