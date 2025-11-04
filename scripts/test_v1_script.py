from chessbot.model import MaskedPolicyModel
from chessbot import MODEL_DIR
from chessbot.mcts_utils import MCTSTree
from chessbot.config import Config
from chessbot.utils import random_init

import numpy as np
import tensorflow as tf

model_loc = MODEL_DIR + "conv_stm_pov_test.h5"
model = MaskedPolicyModel.from_saved(model_loc)

infer = model.make_infer(max_bs=1024)

# train
boards = []
legals = []
values = []
policy = []

for rep in range(2048):
    b = random_init(5 + rep % 30)
    boards.append(b.stacked_planes_stm_pov(1))
    
    legal = b.legal_move_mask()
    legals.append(legal)
    
    n_legal = len(b.legal_moves())
    if n_legal:
        policy.append(legal / n_legal)
    else:
        policy.append(np.zeros_like(legal))
    
    
    v = np.clip(b.material_count() / 9, -1, 1)
    if b.side_to_move() == 'b':
        v = -v
        
    values.append(v)

BIG_NEG = -1e9
EPS = 1e-12

# --- 1) prepare arrays ---
boards_np = np.stack(boards).astype(np.float32)   # -> (N,8,8,29)
legals_np = np.stack(legals).astype(np.int32)     # -> (N,4096)
policy_np = np.stack(policy).astype(np.float32)   # -> (N,4096)
values_np = np.reshape(np.stack(values).astype(np.float32), (-1,1))  # -> (N,1)

print("shapes:", boards_np.shape, legals_np.shape, policy_np.shape, values_np.shape)

# optional quick sanity: any zero-label rows?
label_sums = policy_np.sum(axis=1)
print("zero-labels:", (label_sums <= EPS).sum())

# --- 2) train/test split ---
N = boards_np.shape[0]
val_frac = 0.05
n_val = max(1, int(N * val_frac))
idx = np.arange(N)
np.random.shuffle(idx)
val_idx = idx[:n_val]
train_idx = idx[n_val:]

train_x = (boards_np[train_idx], legals_np[train_idx])
train_y = {"policy": policy_np[train_idx], "value": values_np[train_idx]}
val_x = (boards_np[val_idx], legals_np[val_idx])
val_y = {"policy": policy_np[val_idx], "value": values_np[val_idx]}

# --- 3) compile & train ---
# compile if not already compiled; adjust LR/epochs/batch as you like
try:
    # if already compiled this is a no-op; else compile with defaults
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  policy_loss_weight=1.0, value_loss_weight=1.0)
except Exception:
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  policy_loss_weight=1.0, value_loss_weight=1.0)

EPOCHS = 3
BATCH = 64
model.fit(x=train_x, y=train_y,
          validation_data=(val_x, val_y),
          epochs=EPOCHS, batch_size=BATCH, verbose=2)

# --- 4) make fwd callable (uses your make_fwd method) ---
infer = model.make_infer(max_bs=1024)

# --- 5) run on validation set and postprocess preds (numpy) ---
boards_v, legals_v = val_x
probs_np, vals_np = infer((boards_v, legals_v), min_p=0.001, max_p=0.35, temp=1.0)


# --- 6) compute simple metrics: policy CE, value MSE, top1 match ---
# mask-out samples with zero labels
label_sums_v = val_y["policy"].sum(axis=1)
valid_mask = label_sums_v > EPS

# per-sample CE
log_p = np.log(probs_v + 1e-12)
per_sample_ce = - np.sum(val_y["policy"] * log_p, axis=1)
policy_ce_mean = np.mean(per_sample_ce[valid_mask]) if valid_mask.any() else np.nan

# value mse
value_mse = np.mean((vals_pred.flatten() - val_y["value"].flatten())**2)

# top-1 match rate (argmax of true vs pred, restricted to legal moves)
true_arg = np.argmax(val_y["policy"], axis=1)
pred_arg = np.argmax(probs_v, axis=1)
top1_match = np.mean((true_arg == pred_arg)[valid_mask]) if valid_mask.any() else np.nan

print("VAL samples:", boards_v.shape[0])
print(f"policy CE mean (valid samples): {policy_ce_mean:.6f}")
print(f"value MSE: {value_mse:.6f}")
print(f"top1 match (valid samples): {top1_match:.4%}")

# --- 8) get a fresh fwd from the saved model (optional test reload) ---
# del model; reload simple
# del model
# model = MaskedPolicyModel.from_saved(SAVE_LOC)
# fwd = model.make_fwd(max_bs=1024, warm_shapes=(64,256))






