"""
Gradient tape experiment: measure gradient flow through policy/value heads
and into the backbone trunk. Loads real training data from
debug_training_snapshot.pkl (written by retrain_worker after assembly).

Runs three passes: combined, policy-only, value-only.

Usage:
  python grad_tape_experiment.py [model_path]

Defaults to val_test model if no path given.
"""
import os, pickle, re, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from collections import defaultdict

tf.keras.mixed_precision.set_global_policy("float32")

VAL_TEST_DIR  = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\val_test"
SNAPSHOT_PATH = os.path.join(VAL_TEST_DIR, "debug_training_snapshot.pkl")

parser = argparse.ArgumentParser()
parser.add_argument("model_path", nargs="?",
                    default=os.path.join(VAL_TEST_DIR, "val_test_model.h5"),
                    help="path to .h5 model (default: val_test)")
args = parser.parse_args()

with open(SNAPSHOT_PATH, "rb") as f:
    snap = pickle.load(f)

BATCH = 256
X     = snap["X"][:BATCH].astype(np.int32)
P     = snap["P"][:BATCH].astype(np.float32)
Y     = snap["Y_value"][:BATCH].astype(np.float32)
vwht  = snap["vwht"][:BATCH].astype(np.float32)
pwht  = snap["pwht"][:BATCH].astype(np.float32)

print(f"Loaded {len(X)} real samples from snapshot (batch={BATCH})")
print(f"  X: {X.shape}  policy: {P.shape}  value: {Y.shape}")
print(f"  policy target entropy: {(-P * np.log(P + 1e-9)).sum(axis=1).mean():.3f} nats  "
      f"(random 4288-way = {np.log(4288):.2f})")
print(f"  value Y dist: {Y.mean(axis=0).round(3)}  (win/draw/loss)")
print(f"  pwht mean={pwht.mean():.3f}  vwht mean={vwht.mean():.3f}")
print(f"\nModel: {args.model_path}")
print()

model = tf.keras.models.load_model(args.model_path, compile=False)

X_tf    = tf.constant(X)
P_tf    = tf.constant(P)
Y_tf    = tf.constant(Y)
vwht_tf = tf.constant(vwht)
pwht_tf = tf.constant(pwht)


def group(name):
    n = name.lower()
    m = re.match(r".*(b\d+)_(mha|c1|c2|ln)", n)
    if m: return f"{m.group(1)}_{m.group(2)}"
    if "token_emb"     in n: return "embed"
    if "attn_w"        in n: return "value_attn_pool"
    if "v_ln"          in n: return "value_attn_pool"
    if "v_fc1"         in n: return "v_fc1"
    if "v_fc2"         in n: return "v_fc2"
    if "value_out"     in n: return "value_out"
    if "value_mix"     in n: return "value_mix"
    if "pol_from_fc1"  in n: return "pol_from_fc1"
    if "pol_from_fc2"  in n: return "pol_from_fc2"
    if "pol_from_proj" in n: return "pol_from_proj"
    if "pol_to_fc1"    in n: return "pol_to_fc1"
    if "pol_to_fc2"    in n: return "pol_to_fc2"
    if "pol_to_proj"   in n: return "pol_to_proj"
    if "pol_promo"     in n: return "pol_promo"
    return "other"


def collect_grads(grads):
    groups = defaultdict(lambda: {"n": 0, "gnorm_sq": 0.0, "gmean_sum": 0.0, "wmean_sum": 0.0})
    for v, g in zip(model.trainable_variables, grads):
        if g is None: continue
        gf  = tf.cast(g, tf.float32)
        wf  = tf.cast(v, tf.float32)
        gm  = float(tf.reduce_mean(tf.abs(gf)))
        gn  = float(tf.norm(gf))
        wm  = float(tf.reduce_mean(tf.abs(wf)))
        n   = g.shape.num_elements()
        d   = groups[group(v.name)]
        d["n"]         += n
        d["gnorm_sq"]  += gn**2
        d["gmean_sum"] += gm * n
        d["wmean_sum"] += wm * n
    return groups


def backbone_ref(groups):
    keys = [k for k in groups if re.match(r"b\d+_(c1|c2)$", k)]
    return np.mean([groups[k]["gmean_sum"] / groups[k]["n"] for k in keys]) if keys else 1.0


# three passes
with tf.GradientTape() as tape:
    pol, val = model(X_tf, training=False)
    pol = tf.cast(pol, tf.float32); val = tf.cast(val, tf.float32)
    lp = tf.reduce_mean(pwht_tf * tf.keras.losses.categorical_crossentropy(P_tf, pol, from_logits=True))
    lv = tf.reduce_mean(vwht_tf * tf.keras.losses.categorical_crossentropy(Y_tf, val, from_logits=True))
    loss_combined = lp + lv
grads_combined = tape.gradient(loss_combined, model.trainable_variables)

with tf.GradientTape() as tape:
    pol, _ = model(X_tf, training=False)
    pol = tf.cast(pol, tf.float32)
    loss_pol_only = tf.reduce_mean(pwht_tf * tf.keras.losses.categorical_crossentropy(P_tf, pol, from_logits=True))
grads_pol = tape.gradient(loss_pol_only, model.trainable_variables)

with tf.GradientTape() as tape:
    _, val = model(X_tf, training=False)
    val = tf.cast(val, tf.float32)
    loss_val_only = tf.reduce_mean(vwht_tf * tf.keras.losses.categorical_crossentropy(Y_tf, val, from_logits=True))
grads_val = tape.gradient(loss_val_only, model.trainable_variables)

print(f"loss_pol = {float(lp):.4f}   loss_val = {float(lv):.4f}   total = {float(loss_combined):.4f}")
print()

g_combined = collect_grads(grads_combined)
g_pol      = collect_grads(grads_pol)
g_val      = collect_grads(grads_val)

ref_c = backbone_ref(g_combined)
ref_p = backbone_ref(g_pol)
ref_v = backbone_ref(g_val)

BLOCK_ORDER = [f"b{i}_{k}" for i in range(10) for k in ("mha", "c1", "c2", "ln")]
HEAD_ORDER  = ["embed", "value_mix", "value_attn_pool", "v_fc1", "v_fc2", "value_out",
               "pol_from_fc1", "pol_from_fc2", "pol_from_proj",
               "pol_to_fc1", "pol_to_fc2", "pol_to_proj", "pol_promo", "other"]

hdr = (f"{'Group':<22} {'combined':>11} {'pol_only':>11} {'val_only':>11}"
       f"  {'pol%':>6}  {'val%':>6}  {'|w|_mean':>10}  {'g/w':>8}")
print(hdr)
print("-" * len(hdr))
for k in BLOCK_ORDER + HEAD_ORDER:
    if k not in g_combined: continue
    d    = g_combined[k]
    gm_c = d["gmean_sum"] / d["n"]
    wm_c = d["wmean_sum"] / d["n"]
    gm_p = g_pol[k]["gmean_sum"] / g_pol[k]["n"] if k in g_pol else 0.0
    gm_v = g_val[k]["gmean_sum"] / g_val[k]["n"] if k in g_val else 0.0
    total = gm_p + gm_v
    pp  = 100 * gm_p / total if total > 0 else 0.0
    vp  = 100 * gm_v / total if total > 0 else 0.0
    gw  = gm_c / wm_c if wm_c > 0 else float("inf")
    sep = "  ---" if k == "embed" else ""
    print(f"{k:<22} {gm_c:>11.3e} {gm_p:>11.3e} {gm_v:>11.3e}"
          f"  {pp:>5.1f}%  {vp:>5.1f}%  {wm_c:>10.3e}  {gw:>8.4f}{sep}")

print()
print(f"backbone conv ref — combined: {ref_c:.3e}  pol: {ref_p:.3e}  val: {ref_v:.3e}")
