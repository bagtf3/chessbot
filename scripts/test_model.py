import time, numpy as np, tensorflow as tf
from chessbot.model import load_model

# configure your two models here
model_path  = 'C:/Users/Bryan/Data/chessbot_data/models/conv_small_test.h5'
model_path2 = 'C:/Users/Bryan/Data/chessbot_data/models/conv_model_big_v1000.h5'

# load both
model1 = load_model(model_path)
model2 = load_model(model_path2)

print("TF:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("\nModel1 params:", model1.count_params(), "name:", model1.name)
print("Model2 params:", model2.count_params(), "name:", model2.name)

# --- compile graph forwards (one per model) ---
def make_fwd(model):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 8, 8, 70], dtype=tf.float32)])
    def fwd(x):
        return model(x, training=False)
    # prime once to avoid first-call trace cost
    _ = fwd(tf.zeros([1,8,8,70], tf.float32))
    return fwd

fwd1 = make_fwd(model1)
fwd2 = make_fwd(model2)

# --- timing helper to force GPU sync (so timings are real) ---
def run_once(fwd, x):
    y = fwd(x)
    # force device sync by reducing and materializing to host
    if isinstance(y, (list, tuple)):
        s = [tf.reduce_sum(t) for t in y]
        _ = [t.numpy() for t in s]
    else:
        _ = tf.reduce_sum(y).numpy()

def bench_model(name, fwd, device='/GPU:0', batches=(32,64,128,256,512,1024), warmup=3, reps=20):
    print(f"\n=== {name} @ {device} ===")
    with tf.device(device):
        # per-shape warmups (stabilize autotune)
        for B in batches:
            _ = fwd(tf.zeros([B,8,8,70], tf.float32))
        for _ in range(warmup):
            _ = fwd(tf.zeros([64,8,8,70], tf.float32))

        for B in batches:
            # re-warm this exact shape
            for _ in range(warmup):
                _ = fwd(tf.zeros([B,8,8,70], tf.float32))
            # timed reps
            t0 = time.perf_counter()
            for _ in range(reps):
                x = tf.random.normal([B,8,8,70], dtype=tf.float32)
                run_once(fwd, x)  # includes sync
            t1 = time.perf_counter()

            avg_ms = (t1 - t0) * 1000.0 / reps
            per_sample_ms = avg_ms / B
            sps = (B * reps) / (t1 - t0)
            print(f"B={B:<4}  avg_run={avg_ms:7.2f} ms   per_sample={per_sample_ms:6.3f} ms   throughput={int(sps):>7d} samp/s")

# --- run both on GPU ---
bench_model(f"{model1.name} ({model_path})",  fwd1, device='/GPU:0')
bench_model(f"{model2.name} ({model_path2})", fwd2, device='/GPU:0')


from chessbot.model import make_conv_model

model = make_conv_model("test", width=128, n_blocks=4)
model.summary()
#model.save("C:/Users/Bryan/Data/chessbot_data/models/conv_small_init.h5")


import time, numpy as np, tensorflow as tf

# assumes `model` already exists in your session

# --- compile a graph forward (fast path) ---
@tf.function(input_signature=[tf.TensorSpec([None, 8, 8, 70], tf.float32)])
def fwd(x):
    return model(x, training=False)

# warm up once so the graph is traced
_ = fwd(tf.zeros([1,8,8,70], tf.float32))

def sync_outputs(y):
    # force device sync by reducing to scalars and materializing to host
    if isinstance(y, (list, tuple)):
        vals = [tf.reduce_sum(t) for t in y]
        _ = [v.numpy() for v in vals]
    else:
        _ = tf.reduce_sum(y).numpy()

def bench_once(B, reps_predict, reps_fwd, device='/GPU:0'):
    with tf.device(device):
        # prepare random input once (numpy → one big H2D per call)
        X = np.random.randn(B, 8, 8, 70).astype(np.float32)

        # --- predict() path ---
        # per-shape warmup
        _ = model.predict(X, batch_size=B, verbose=0)
        t0 = time.perf_counter()
        for _ in range(reps_predict):
            _ = model.predict(X, batch_size=B, verbose=0)  # returns numpy (syncs GPU)
        t1 = time.perf_counter()
        ms_pred = (t1 - t0) * 1000.0 / reps_predict
        sps_pred = B * reps_predict / (t1 - t0)

        # --- fwd() path ---
        x_gpu = tf.convert_to_tensor(X)  # single H2D copy
        # per-shape warmup
        _ = fwd(x_gpu)
        # timed reps
        t2 = time.perf_counter()
        for _ in range(reps_fwd):
            y = fwd(x_gpu)
            sync_outputs(y)  # ensure kernels finished
        t3 = time.perf_counter()
        ms_fwd = (t3 - t2) * 1000.0 / reps_fwd
        sps_fwd = B * reps_fwd / (t3 - t2)

    return ms_pred, sps_pred, ms_fwd, sps_fwd

def run_bench(device='/GPU:0'):
    batches = [32, 64, 128, 256, 512, 1024, 2048]
    # keep runtime reasonable at big B
    reps_predict_map = {32:30, 64:30, 128:20, 256:15, 512:10, 1024:5, 2048:3}
    reps_fwd_map     = {32:50, 64:50, 128:30, 256:20, 512:12, 1024:6, 2048:4}

    print(f"\n=== {model.name} @ {device} ===")
    print(f"{'B':>6} | {'predict ms':>10} | {'predict s/s':>11} | {'fwd ms':>10} | {'fwd s/s':>10} | speedup")
    print("-"*70)
    for B in batches:
        ms_pred, sps_pred, ms_fwd, sps_fwd = bench_once(
            B,
            reps_predict=reps_predict_map[B],
            reps_fwd=reps_fwd_map[B],
            device=device
        )
        speed = ms_pred / ms_fwd if ms_fwd > 0 else float('inf')
        print(f"{B:6d} | {ms_pred:10.2f} | {int(sps_pred):11d} | {ms_fwd:10.2f} | {int(sps_fwd):10d} | {speed:6.2f}×")

# run on GPU (change device to '/CPU:0' if you want CPU numbers too)
run_bench('/GPU:0')
