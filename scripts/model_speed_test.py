import time
import numpy as np
import tensorflow as tf
from chessbot.model import load_model

def make_compiled_forward(model):
    """
    Return a tf.function-wrapped forward that accepts int32 token inputs
    of shape (batch, 64).
    """
    @tf.function(input_signature=[
        tf.TensorSpec([None, 64], tf.int32)
    ], experimental_compile=True)   
    def f(x):
        return model(x, training=False)
    return f


def speed_test_forward(model, compiled_fn,
                       batch_sizes=(32, 64, 128, 256, 512),
                       n_warmup=10,
                       n_iters=50,
                       vocab_size=21):
    """
    Run warmup then timed inferences for each batch size.
    Prints avg latency (ms) and throughput (samples/sec).
    """
    # quick check
    if not callable(compiled_fn):
        raise ValueError("compiled_fn must be a callable tf.function")

    for b in batch_sizes:
        # random token input (int32)
        inp_np = np.random.randint(0, vocab_size, size=(b, 64)).astype("int32")
        inp = tf.constant(inp_np)

        # warmup
        for _ in range(n_warmup):
            out = compiled_fn(inp)
            # force sync to include device time
            _ = out[0].numpy()

        # timed loop
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = compiled_fn(inp)
            _ = out[0].numpy()
        t1 = time.perf_counter()

        total = t1 - t0
        avg_s = total / n_iters
        avg_ms = avg_s * 1000.0
        throughput = b / avg_s

        print(f"batch={b:<4}  avg_latency={avg_ms:7.2f} ms  "
              f"throughput={throughput:8.1f} samples/s")
        

models = [
    'C:/Users/Bryan/Data/chessbot_data/models/conformer_10x256x5_init.h5',
    'C:/Users/Bryan/Data/chessbot_data/models/conv_lr_mha_442x256x3_init.h5',
    'C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_9x296_selfplay/conv_9x296_selfplay_model.h5',
    'C:/Users/Bryan/Data/chessbot_data/models/conv_lr_mha_642x256x3_init.h5',
    'C:/Users/Bryan/Data/chessbot_data/models/conv_mha_lr_643x256x2_init.h5'
]

for model_file in models:
    name = model_file.split("/")[-1]
    print(" Starting test for {} ".format(name).center(60, "#"))

    model = load_model(model_file)
    fwd = make_compiled_forward(model)
    speed_test_forward(model, fwd)
    print()
    
    del model, fwd
    import gc; gc.collect()
    tf.keras.backend.clear_session()

