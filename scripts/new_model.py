import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import time
import numpy as np

BIG_NEG = -1e9
EPS = 1e-9

def build_core_model():
    # Inputs
    board_in = Input(shape=(8,8,28), name="board")
    legal_in = Input(shape=(4096,), name="legal_mask")

    # Bigger conv trunk (~5M total params for the model)
    x = layers.Conv2D(384, 3, padding="same", activation="relu", name="conv1")(board_in)   # 28 -> 384
    x = layers.Conv2D(512, 3, padding="same", activation="relu", name="conv2")(x)         # 384 -> 512
    x = layers.Conv2D(576, 3, padding="same", activation="relu", name="conv3")(x)         # 512 -> 576
    x = layers.Conv2D(576, 1, padding="same", activation="relu", name="conv3_1x1")(x)     # 576 -> 576

    # policy head -> (8,8,64) -> flatten to 4096 logits
    p = layers.Conv2D(64, 1, padding="same", activation=None, name="policy_conv1x1")(x)
    policy_logits = layers.Reshape((4096,), name="policy_logits")(p)

    # value head: GAP -> small MLP -> tanh*10
    v = layers.GlobalAveragePooling2D(name="gap")(x)   # shape (B,576)
    v = layers.Dense(128, activation="relu", name="value_fc1")(v)
    v_raw = layers.Dense(1, activation="tanh", name="value_tanh")(v)
    v_scaled = layers.Lambda(lambda t: t * 10.0, name="value_out")(v_raw)

    core = Model(inputs=[board_in, legal_in], outputs=[policy_logits, v_scaled], name="core_model_v5m")
    return core

class MaskedPolicyModel(tf.keras.Model):
    """
    Wraps the core functional model and overrides train_step/test_step so we can
    use the legal_mask input in the loss calculation.
    Expected training call:
      model.fit(x=(boards, legal_masks),
                y={"policy": policy_targets, "value": value_targets}, ...)
    """
    def __init__(self, core):
        super().__init__(name="masked_policy_model")
        self.core = core
    
    def call(self, inputs, training=False):
        return self.core(inputs, training=training)
    
    def compile(self, optimizer, policy_loss_weight=1.0, value_loss_weight=1.0, **kwargs):
        super().compile(**kwargs)
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        # metrics
        self.policy_loss_tracker = tf.keras.metrics.Mean(name="policy_loss")
        self.value_loss_tracker  = tf.keras.metrics.Mean(name="value_loss")
        self.total_loss_tracker  = tf.keras.metrics.Mean(name="loss")
    
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.policy_loss_tracker, self.value_loss_tracker]
    
    def train_step(self, data):
        # data: (x, y) where x=(boards, legal_mask), y={"policy": labels, "value": v_labels}
        x, y = data
        boards, legal_mask = x
        labels = y["policy"]
        values = y["value"]
        
        with tf.GradientTape() as tape:
            logits, v_pred = self.core([boards, legal_mask], training=True)
            bsize = tf.shape(logits)[0]
            logits_flat = tf.reshape(logits, [bsize, -1])   # (B,4096)
            labels_flat = tf.reshape(labels, [bsize, -1])
            mask_flat   = tf.cast(tf.reshape(legal_mask, [bsize, -1]), logits_flat.dtype)
            
            # (1) mask logits so softmax only over legal moves
            masked_logits = tf.where(mask_flat > 0.5, logits_flat, tf.ones_like(logits_flat) * BIG_NEG)
            logp = tf.nn.log_softmax(masked_logits, axis=1)
            
            # normalize labels per-sample (if labels are raw visits)
            label_sums = tf.reduce_sum(labels_flat, axis=1, keepdims=True)
            labels_norm = labels_flat / (label_sums + EPS)
            
            # per-sample CE; ignore samples with no labels
            per_elem_ce = - labels_norm * logp
            per_sample_ce = tf.reduce_sum(per_elem_ce, axis=1)
            valid_mask = tf.squeeze(label_sums > EPS, axis=1)
            policy_loss = tf.reduce_sum(tf.where(valid_mask, per_sample_ce, tf.zeros_like(per_sample_ce))) / (tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + EPS)
            
            # value loss (MSE on scaled outputs)
            value_loss = tf.reduce_mean(tf.square(v_pred - values))
            
            total_loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss
        
        grads = tape.gradient(total_loss, self.core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.core.trainable_variables))
        
        # update metrics
        self.policy_loss_tracker.update_state(policy_loss)
        self.value_loss_tracker.update_state(value_loss)
        self.total_loss_tracker.update_state(total_loss)
        
        return {"loss": self.total_loss_tracker.result(), "policy_loss": self.policy_loss_tracker.result(), "value_loss": self.value_loss_tracker.result()}
    
    def test_step(self, data):
        x, y = data
        boards, legal_mask = x
        labels = y["policy"]
        values = y["value"]
        
        logits, v_pred = self.core([boards, legal_mask], training=False)
        bsize = tf.shape(logits)[0]
        logits_flat = tf.reshape(logits, [bsize, -1])
        labels_flat = tf.reshape(labels, [bsize, -1])
        mask_flat   = tf.cast(tf.reshape(legal_mask, [bsize, -1]), logits_flat.dtype)
        
        masked_logits = tf.where(mask_flat > 0.5, logits_flat, tf.ones_like(logits_flat) * BIG_NEG)
        logp = tf.nn.log_softmax(masked_logits, axis=1)
        label_sums = tf.reduce_sum(labels_flat, axis=1, keepdims=True)
        labels_norm = labels_flat / (label_sums + EPS)
        per_elem_ce = - labels_norm * logp
        per_sample_ce = tf.reduce_sum(per_elem_ce, axis=1)
        valid_mask = tf.squeeze(label_sums > EPS, axis=1)
        policy_loss = tf.reduce_sum(tf.where(valid_mask, per_sample_ce, tf.zeros_like(per_sample_ce))) / (tf.reduce_sum(tf.cast(valid_mask, tf.float32)) + EPS)
        value_loss = tf.reduce_mean(tf.square(v_pred - values))
        total_loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss
        
        self.policy_loss_tracker.update_state(policy_loss)
        self.value_loss_tracker.update_state(value_loss)
        self.total_loss_tracker.update_state(total_loss)
        
        return {"loss": self.total_loss_tracker.result(), "policy_loss": self.policy_loss_tracker.result(), "value_loss": self.value_loss_tracker.result()}

# build & show the core model
core = build_core_model()
core.summary()  # prints layer shapes and param counts (run locally)

# wrap it
model = MaskedPolicyModel(core)
# compile (optimizer required)
model.compile(optimizer=tf.keras.optimizers.Adam(3e-4))

# usage example (synthetic batch)
B = 8
boards = np.random.randn(B,8,8,28).astype(np.float32)
legal_mask = np.zeros((B,4096), dtype=np.float32)
labels = np.zeros((B,4096), dtype=np.float32)
for i in range(B):
    k = np.random.randint(6, 51)
    idx = np.random.choice(4096, size=k, replace=False)
    legal_mask[i, idx] = 1.0
    visits = np.random.rand(k).astype(np.float32)
    visits /= (visits.sum() + 1e-9)
    labels[i, idx] = visits
values = (np.random.rand(B,1).astype(np.float32) * 20.0) - 10.0  # -10..10

# train step (fit style)
hist = model.fit(x=(boards, legal_mask), y={"policy": labels, "value": values}, epochs=1, batch_size=B)
print("Done one epoch (poof).")

# Inference helpers you will want:
def masked_softmax_from_logits(logits, legal_mask, temperature=1.0):
    """
    logits: np or tf array shape (B,4096) or (B,8,8,64) flattened to (B,4096)
    legal_mask: same shape (B,4096)
    returns: probabilities over legal moves (B,4096) where illegal positions have 0 prob.
    """
    logits = tf.reshape(logits, [tf.shape(logits)[0], -1])
    mask = tf.cast(tf.reshape(legal_mask, [tf.shape(logits)[0], -1]), logits.dtype)
    if temperature != 1.0:
        logits = logits / tf.cast(temperature, logits.dtype)
    masked = tf.where(mask > 0.5, logits, tf.ones_like(logits) * BIG_NEG)
    probs = tf.nn.softmax(masked, axis=1)
    # zero out rows with no legal moves
    has_any = tf.reduce_sum(mask, axis=1) > 0.5
    probs = probs * tf.cast(tf.expand_dims(has_any, axis=1), probs.dtype)
    return probs
#%%

BIG_NEG = -1e9
EPS = 1e-12

# core: Keras model returning (logits(B,4096), value(B,1))

@tf.function(input_signature=[
    tf.TensorSpec([None, 8, 8, 28], tf.float32),
    tf.TensorSpec([None, 4096], tf.int32),
    tf.TensorSpec([], tf.float32),  # min_p
    tf.TensorSpec([], tf.float32),  # max_p
    tf.TensorSpec([], tf.float32)   # temp
])
def infer_with_postproc(board, legal_mask, min_p, max_p, temp):
    """
    Returns: probs (B,4096) and value (B,1).
    - no top-k, no greedy. Always returns full probs vector (masked, clamped, renormed).
    - min_p/max_p and temp are runtime scalars (change without rebuilding).
    """
    # model forward (stays on device)
    logits, value = core([board, legal_mask], training=False)   # logits shape (B,4096)

    logits = tf.reshape(logits, [tf.shape(logits)[0], -1])
    mask = tf.cast(tf.reshape(legal_mask, [tf.shape(logits)[0], -1]), logits.dtype)

    # mask illegal logits
    masked_logits = tf.where(mask > 0.5, logits, tf.ones_like(logits) * tf.cast(BIG_NEG, logits.dtype))

    # softmax over legal moves
    scaled = masked_logits / tf.cast(temp, masked_logits.dtype)
    probs = tf.nn.softmax(scaled, axis=1)
    probs = probs * mask  # zero illegal slots

    # clamp + renormalize (fast)
    min_p = tf.cast(min_p, probs.dtype)
    max_p = tf.cast(max_p, probs.dtype)
    
    # clipped only on legal slots, illegal remain zero
    clipped_legal = tf.where(mask > 0.5, tf.clip_by_value(probs, clip_value_min=min_p, clip_value_max=max_p), tf.zeros_like(probs))
    s = tf.reduce_sum(clipped_legal, axis=1, keepdims=True)
    valid = s > EPS
    probs = tf.where(valid, clipped_legal / (s + (1.0 - tf.cast(valid, probs.dtype))), tf.zeros_like(clipped_legal))

    return probs, value


# ---------- compiled forward-only TF function ----------
@tf.function(input_signature=[
    tf.TensorSpec([None, 8, 8, 28], tf.float32),
    tf.TensorSpec([None, 4096], tf.int32)
])
def fwd_only(board, legal_mask):
    logits, value = core([board, legal_mask], training=False)
    return logits, value

# ---------- prepare inputs (reuse your arrays `boards` and `legal_mask`) ----------
# ensure numpy arrays exist in your session (you created them earlier)
boards_np = np.asarray(boards, dtype=np.float32)            # (B,8,8,28)
legal_np  = np.asarray(legal_mask, dtype=np.int32)         # (B,4096)

# create TF tensors once (keep on device)
board_tf = tf.constant(boards_np, dtype=tf.float32)
legal_tf = tf.constant(legal_np, dtype=tf.int32)

# warm up / compile both graphs (first call compiles)
_ = fwd_only(board_tf, legal_tf)
_ = infer_with_postproc(board_tf, legal_tf,
                        tf.constant(0.001, tf.float32),
                        tf.constant(0.35,  tf.float32),
                        tf.constant(1.0,   tf.float32))

# ---------- quick correctness checks ----------
min_p = 0.001; max_p = 0.35; temp = 1.0

probs_tf, val_tf = infer_with_postproc(board_tf, legal_tf,
                                       tf.constant(min_p, tf.float32),
                                       tf.constant(max_p, tf.float32),
                                       tf.constant(temp, tf.float32))
probs_np = probs_tf.numpy()
val_np = val_tf.numpy()

# sums over legal moves (should be ~1 for rows that have legal moves)
sums = probs_np.sum(axis=1)
print("per-sample sum min/mean/max:", sums.min(), sums.mean(), sums.max())

# ensure illegal indices have near-zero mass
illegal_np = 1 - legal_np
max_illegal = (probs_np * illegal_np).max()
print("max prob on illegal indices (should be ~0):", max_illegal)

# check clip bounds on legal indices (report min/max among legal entries)
legal_probs = probs_np[legal_np.astype(bool)].flatten()
print("legal probs min/mean/max:", legal_probs.min(), legal_probs.mean(), legal_probs.max())

# ---------- helper: numpy postprocess (same algorithm as TF) ----------
def numpy_postprocess_from_logits(logits_np, legal_np, min_p=0.001, max_p=0.35, temp=1.0):
    # logits_np: (B,4096), legal_np: (B,4096) ints 0/1
    BIG_NEG = -1e9
    EPS = 1e-12
    logits = logits_np.astype(np.float64)
    mask = (legal_np > 0).astype(np.float64)
    masked = np.where(mask > 0, logits / temp, BIG_NEG)

    # stable softmax over legal slots only
    row_max = masked.max(axis=1, keepdims=True)
    exp = np.exp(masked - row_max) * mask
    sumexp = exp.sum(axis=1, keepdims=True)
    has_any = (sumexp > 0)
    probs = np.where(has_any, exp / (sumexp + EPS), 0.0)

    # clamp only on legal slots, illegal stay zero
    clipped = np.where(mask > 0, np.clip(probs, min_p, max_p), 0.0)

    # renormalize across legal slots
    s = clipped.sum(axis=1, keepdims=True)
    valid = (s > EPS)
    probs_out = np.where(valid, clipped / (s + (~valid).astype(np.float64)), 0.0)

    return probs_out

#%%
# ---------- timing benchmarks ----------
reps = 50

# 1) forward-only (TF) -> block via .numpy() (includes transfer)
t0 = time.perf_counter()
for _ in range(reps):
    logits_tf, value_tf = fwd_only(board_tf, legal_tf)
    _ = logits_tf.numpy()
    _ = value_tf.numpy()
t1 = time.perf_counter()
t_fwd_only_ms = (t1 - t0) / reps * 1000.0

# 2) forward + postprocess all in TF on GPU (infer_with_postproc) -> block via .numpy()
t0 = time.perf_counter()
for _ in range(reps):
    probs_tf, value_tf = infer_with_postproc(board_tf, legal_tf,
                                             tf.constant(min_p, tf.float32),
                                             tf.constant(max_p, tf.float32),
                                             tf.constant(temp, tf.float32))
    _ = probs_tf.numpy()
    _ = value_tf.numpy()
t1 = time.perf_counter()
t_fwd_post_tf_ms = (t1 - t0) / reps * 1000.0

# 3) forward in TF, then postprocess in NumPy (host)
t0 = time.perf_counter()
for _ in range(reps):
    logits_tf, value_tf = fwd_only(board_tf, legal_tf)
    logits_np = logits_tf.numpy()   # transfer to host (blocks)
    value_np = value_tf.numpy()
    probs_np = numpy_postprocess_from_logits(logits_np, legal_np, min_p=min_p, max_p=max_p, temp=temp)
t1 = time.perf_counter()
t_fwd_post_np_ms = (t1 - t0) / reps * 1000.0

# ---------- report ----------
print(f"avg ms per batch (B={boards_np.shape[0]}):")
print(f"  1) fwd only (TF)         : {t_fwd_only_ms:.3f} ms")
print(f"  2) fwd + post (TF on GPU): {t_fwd_post_tf_ms:.3f} ms")
print(f"  3) fwd + post (numpy)   : {t_fwd_post_np_ms:.3f} ms")

# sanity: compare TF-post vs NumPy-post (they may differ numerically but should be close)
probs_tf_after, _ = infer_with_postproc(board_tf, legal_tf,
                                       tf.constant(min_p, tf.float32),
                                       tf.constant(max_p, tf.float32),
                                       tf.constant(temp, tf.float32))
probs_tf_np = probs_tf_after.numpy()
probs_np_post = numpy_postprocess_from_logits(fwd_only(board_tf, legal_tf)[0].numpy(), legal_np, min_p=min_p, max_p=max_p, temp=temp)

# show per-sample L1 difference between TF-probs and NumPy-probs
l1 = np.abs(probs_tf_np - probs_np_post).sum(axis=1)
print("per-sample L1 diff TF_post vs NumPy_post min/mean/max:", l1.min(), l1.mean(), l1.max())

#%%
import time, math, numpy as np, tensorflow as tf, traceback

# --- set desired big batch size here ---
BATCH_SIZE = 1024  # <- change to the larger batch size you want to test

# --- build batch by repeating your existing small arrays ---
base_boards = np.asarray(boards, dtype=np.float32)
base_legal  = np.asarray(legal_mask, dtype=np.int32)
B0 = base_boards.shape[0]
reps = math.ceil(BATCH_SIZE / B0)
boards_big_np = np.repeat(base_boards, reps, axis=0)[:BATCH_SIZE]
legal_big_np  = np.repeat(base_legal,  reps, axis=0)[:BATCH_SIZE]

# push to device once
boards_big_tf = tf.constant(boards_big_np, dtype=tf.float32)
legal_big_tf  = tf.constant(legal_big_np,  dtype=tf.int32)

min_p = 0.001; max_p = 0.35; temp = 1.0
reps_time = 50

# warm up / compile
_ = fwd_only(boards_big_tf, legal_big_tf)
_ = infer_with_postproc(boards_big_tf, legal_big_tf,
                        tf.constant(min_p, tf.float32),
                        tf.constant(max_p, tf.float32),
                        tf.constant(temp, tf.float32))

try:
    # 1) forward-only (TF) -> block via .numpy()
    t0 = time.perf_counter()
    for _ in range(reps_time):
        logits_tf, value_tf = fwd_only(boards_big_tf, legal_big_tf)
        _ = logits_tf.numpy()
        _ = value_tf.numpy()
    t1 = time.perf_counter()
    t_fwd_only_ms = (t1 - t0) / reps_time * 1000.0

    # 2) forward + postprocess all in TF on GPU
    t0 = time.perf_counter()
    for _ in range(reps_time):
        probs_tf, value_tf = infer_with_postproc(boards_big_tf, legal_big_tf,
                                                 tf.constant(min_p, tf.float32),
                                                 tf.constant(max_p, tf.float32),
                                                 tf.constant(temp, tf.float32))
        _ = probs_tf.numpy()
        _ = value_tf.numpy()
    t1 = time.perf_counter()
    t_fwd_post_tf_ms = (t1 - t0) / reps_time * 1000.0

    # # 3) forward in TF, then postprocess in NumPy (host)
    # t0 = time.perf_counter()
    # for _ in range(reps_time):
    #     logits_tf, value_tf = fwd_only(boards_big_tf, legal_big_tf)
    #     logits_np = logits_tf.numpy()
    #     value_np = value_tf.numpy()
    #     probs_np = numpy_postprocess_from_logits(logits_np, legal_big_np, min_p=min_p, max_p=max_p, temp=temp)
    # t1 = time.perf_counter()
    # t_fwd_post_np_ms = (t1 - t0) / reps_time * 1000.0

    # report
    print(f"\nBatch size = {BATCH_SIZE}, reps = {reps_time}")
    print(f"  1) fwd only (TF)         : {t_fwd_only_ms:.3f} ms per batch, {t_fwd_only_ms/BATCH_SIZE:.6f} ms/sample, {BATCH_SIZE*1000.0/t_fwd_only_ms:8.2f} samples/s")
    print(f"  2) fwd + post (TF on GPU): {t_fwd_post_tf_ms:.3f} ms per batch, {t_fwd_post_tf_ms/BATCH_SIZE:.6f} ms/sample, {BATCH_SIZE*1000.0/t_fwd_post_tf_ms:8.2f} samples/s")
    #print(f"  3) fwd + post (numpy)   : {t_fwd_post_np_ms:.3f} ms per batch, {t_fwd_post_np_ms/BATCH_SIZE:.6f} ms/sample, {BATCH_SIZE*1000.0/t_fwd_post_np_ms:8.2f} samples/s")

    # quick correctness check (TF vs NumPy postproc)
    # probs_tf_after, _ = infer_with_postproc(boards_big_tf, legal_big_tf,
    #                                        tf.constant(min_p, tf.float32),
    #                                        tf.constant(max_p, tf.float32),
    #                                        tf.constant(temp, tf.float32))
    # probs_tf_np = probs_tf_after.numpy()
    # probs_np_post = numpy_postprocess_from_logits(fwd_only(boards_big_tf, legal_big_tf)[0].numpy(), legal_big_np, min_p=min_p, max_p=max_p, temp=temp)
    # l1 = np.abs(probs_tf_np - probs_np_post).sum(axis=1)
    # print("L1 diff TF_post vs NumPy_post min/mean/max:", l1.min(), l1.mean(), l1.max())

except tf.errors.ResourceExhaustedError as e:
    print("OOM on this batch size:", BATCH_SIZE)
    traceback.print_exc()
except Exception as e:
    print("Error during test:")
    traceback.print_exc()