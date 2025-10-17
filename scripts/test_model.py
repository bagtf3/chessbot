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

