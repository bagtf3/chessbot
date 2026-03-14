import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("compute policy:", mixed_precision.global_policy())

import gc
import os
import queue
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chessbot.model import set_loss_weights
from chessbot.utils import batch_policy_metrics, format_time, print_validation


TFREC_DIR  = "C:/Users/Bryan/Data/chessbot_data/bootstrap_tfrecords"
RUN_DIR    = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/val_test_multi"
MODEL_DIR  = "C:/Users/Bryan/Data/chessbot_data/models"

batch_size = 128
epoch_size = 10240
shuffle_buffer = 64000
steps_per_epoch = epoch_size // batch_size
MAX_EPOCH = 900
PLOT_EVERY = 10
LR = 1e-4
MODEL_DEFS = [
    #"16m-pure-conv", # dead
    #"16m-film", # solid
    #"16m-concat-fusion", # dead
    #"16m-conformer", # dead
    #"16m-gated-ctx", #solid
    "16m-frankenformer-interweaved"
]

LW_SCHEDULE = {
    0: {"policy_logits": 0.1, "value_out": 0.1},
    1: {"policy_logits": 0.75, "value_out": 1.5},
    10: {"policy_logits": 1.5, "value_out": 3.0},
    160: {"policy_logits": 1.0, "value_out": 2.25},
    250: {"policy_logits": 0.75, "value_out": 1.5},
    480: {"policy_logits": 0.65, "value_out": 1.25},
    640: {"policy_logits": 0.5, "value_out": 1.0},
    700: {"policy_logits": 0.25, "value_out": 0.5}
}

def load_tf_model(path, lr=LR):
    model = keras.models.load_model(path)

    # print + save summary
    print("\n" + "=" * 80)
    print(f"MODEL SUMMARY: {path}")
    print("=" * 80)

    # always build a fresh optimizer at the requested LR
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = mixed_precision.LossScaleOptimizer(opt)

    model._default_opt = opt
    model._default_loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        ),
        "value_out": tf.keras.losses.MeanSquaredError(),
    }

    return model


model_states = []
for name in MODEL_DEFS:
    run_model_file  = os.path.join(RUN_DIR,   f"{name}_model.h5")
    src_model_file  = os.path.join(MODEL_DIR, f"{name}_model.h5")
    progress_file   = os.path.join(RUN_DIR,   f"{name}_eval_progress.csv")

    print(f"[init] {name}: loading fresh from MODEL_DIR")
    model_states.append({
        "name":            name,
        "load_from":       src_model_file,
        "model_file":      run_model_file,
        "progress_file":   progress_file,
        "model":           None,
        "eval_df":         None,
        "metrics_history": {"value_mse": [], "value_corr": []},
        "current_lw":      None,
        "epoch":           0,
    })


feature_spec = {
    "enc_in": tf.io.FixedLenFeature([], tf.string),
    "mask": tf.io.FixedLenFeature([], tf.string),
    "policy_logits": tf.io.FixedLenFeature([], tf.string),
    "value_out": tf.io.FixedLenFeature([], tf.float32),
    "weight": tf.io.FixedLenFeature([], tf.float32),
}


def list_tfrecord_files(tfrec_dir):
    files = [
        os.path.join(tfrec_dir, x)
        for x in os.listdir(tfrec_dir)
        if x.endswith(".tfrecord.gz")
    ]
    files.sort()
    return files


def parse_record(raw):
    feat = tf.io.parse_single_example(raw, feature_spec)
    enc_in = tf.cast(
        tf.io.parse_tensor(feat["enc_in"], out_type=tf.int16),
        tf.int32,
    )
    mask = tf.io.parse_tensor(feat["mask"], out_type=tf.int32)
    policy = tf.io.parse_tensor(feat["policy_logits"], out_type=tf.float32)
    value = tf.reshape(feat["value_out"], [1])
    weight = feat["weight"]

    return (
        {"enc_in": enc_in, "mask": mask},
        {"policy_logits": policy, "value_out": value},
        {"policy_logits": weight, "value_out": weight},
    )


def make_batch_stream(tfrec_dir, batch_size, shuffle_buffer):
    file_list = list_tfrecord_files(tfrec_dir)
    if not file_list:
        raise RuntimeError(f"No .tfrecord.gz files in {tfrec_dir}")

    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.shuffle(len(file_list), reshuffle_each_iteration=True)
    ds = ds.repeat()

    ds = ds.interleave(
        lambda path: tf.data.TFRecordDataset(
            path,
            compression_type="GZIP",
        ),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class EpochBufferWorker(threading.Thread):

    def __init__(self, batch_stream, steps_per_epoch, max_ready=2):
        super().__init__(daemon=True)
        self.batch_stream = batch_stream
        self.steps_per_epoch = steps_per_epoch
        self.ready = queue.Queue(maxsize=max_ready)
        self.stop_event = threading.Event()
        self.error = None

    def stop(self):
        self.stop_event.set()

    def get(self):
        item = self.ready.get()
        if isinstance(item, Exception):
            raise item
        return item

    def run(self):
        try:
            it = iter(self.batch_stream)

            while not self.stop_event.is_set():
                x_parts = []
                m_parts = []
                p_parts = []
                y_parts = []
                w_parts = []

                for _ in range(self.steps_per_epoch):
                    if self.stop_event.is_set():
                        return

                    inp, out, wt = next(it)
                    x_parts.append(inp["enc_in"])
                    m_parts.append(inp["mask"])
                    p_parts.append(out["policy_logits"])
                    y_parts.append(out["value_out"])
                    w_parts.append(wt["policy_logits"])

                bundle = {
                    "enc_in": tf.concat(x_parts, axis=0),
                    "mask": tf.concat(m_parts, axis=0),
                    "policy_logits": tf.concat(p_parts, axis=0),
                    "value_out": tf.concat(y_parts, axis=0),
                    "weight": tf.concat(w_parts, axis=0),
                }

                self.ready.put(bundle)

        except Exception as exc:
            self.error = exc
            try:
                self.ready.put(exc, timeout=1.0)
            except queue.Full:
                pass


def build_model_inputs(model, bundle):
    inputs = {}

    for name in model.input_names:
        if name == "enc_in":
            inputs["enc_in"] = bundle["enc_in"]
        elif name == "mask":
            inputs["mask"] = bundle["mask"]

    return inputs


def fit_epoch(model, bundle, batch_size, lw):
    x = {"enc_in": bundle["enc_in"]}
    if "mask" in model.input_names and "mask" in bundle:
        x["mask"] = bundle["mask"]

    y = {
        "policy_logits": bundle["policy_logits"],
        "value_out": bundle["value_out"],
    }

    sample_weight = {
        "policy_logits": tf.ones_like(bundle["weight"]) * lw["policy_logits"],
        "value_out": bundle["weight"] * lw["value_out"],
    }

    return model.fit(
        x=x,
        y=y,
        sample_weight=sample_weight,
        batch_size=batch_size,
        epochs=1,
        verbose=0,
        shuffle=False,
    )

def moving_average_pd(arr, window=15):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().values


def do_eval(ms, bundle, batch_size):
    name = ms["name"]
    model = ms["model"]
    epoch = ms["epoch"]

    preds = model.predict(
        build_model_inputs(model, bundle),
        verbose=0,
        batch_size=64,
    )

    policy_logits = preds[0]
    value_preds = preds[1].ravel()

    pstack = bundle["policy_logits"].numpy()
    ystack = bundle["value_out"].numpy().ravel()
    mstack = bundle["mask"].numpy()

    policy_stats = batch_policy_metrics(policy_logits, pstack, mstack)
    value_mse = np.mean((value_preds - ystack) ** 2)
    value_corr = np.corrcoef(value_preds, ystack)[0, 1]

    mh = ms["metrics_history"]
    for k, v in policy_stats.items():
        if k not in mh:
            mh[k] = []
        mh[k].append(v)

    mh["value_mse"].append(value_mse)
    mh["value_corr"].append(value_corr)

    latest_row = pd.DataFrame(mh).iloc[[-1], :]
    n_samples = bundle["enc_in"].shape[0]
    if epoch > 0:
        n_samples *= PLOT_EVERY
    latest_row["n_samples"] = n_samples

    eval_df = ms["eval_df"]
    if eval_df is None:
        latest_row["model_epoch"] = 0
        eval_df = latest_row.copy()
    else:
        latest_row["model_epoch"] = eval_df.tail(1)["model_epoch"].item() + 1
        latest_row = latest_row.reindex(columns=eval_df.columns)
        eval_df = pd.concat([eval_df, latest_row])

    eval_df.round(4).to_csv(ms["progress_file"], index=False)
    ms["eval_df"] = eval_df

    print(f"[eval] {name}")
    print_metrics = {"value_mse": value_mse, "value_corr": value_corr}
    print_metrics.update(policy_stats)
    print_validation(epoch, print_metrics)

    retrains = eval_df.tail(1)["model_epoch"].item()
    hide_first = max(int(0.1 * retrains), 5)
    ma_window = min(max(3, int(retrains * 0.2)), 15)
    if ma_window % 2 == 0:
        ma_window += 1

    x = np.arange(len(eval_df["policy_ce"]))
    xs = x[hide_first:]

    if len(xs) > hide_first:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"{name}  —  epoch {epoch}")

        for ax, key, title in [
            (axes[0, 0], "policy_ce",  "policy CE (nats)"),
            (axes[0, 1], "ce_gain",    "CE gain vs uniform"),
            (axes[0, 2], "value_mse",  "value MSE"),
            (axes[1, 0], "value_corr", "value corr"),
            (axes[1, 1], "top1_exact", "top1 exact"),
        ]:
            raw = np.array(eval_df[key])[hide_first:]
            ma  = moving_average_pd(eval_df[key], window=ma_window)[hide_first:]
            if raw.size:
                ax.plot(xs, raw, alpha=0.6, lw=1, label=key)
            if ma.size:
                ax.plot(xs, ma, lw=2, label=f"MA{ma_window}")
            ax.set_title(title)
            ax.legend()

        # value scatter
        ax_sc = axes[1, 2]
        ax_sc.scatter(ystack, value_preds, s=2, alpha=0.3)
        lims = [
            min(ystack.min(), value_preds.min()), max(ystack.max(), value_preds.max())
        ]
        
        ax_sc.plot(lims, lims, "r--", lw=1)
        ax_sc.set_xlabel("target")
        ax_sc.set_ylabel("pred")
        ax_sc.set_title(f"value scatter  (r={value_corr:.3f})")

        fig.tight_layout()
        plt.show()
        plt.close(fig)
        plt.close("all")


n_files = len(list_tfrecord_files(TFREC_DIR))
print(
    f"[tfrec] files={n_files}  shuffle_buffer={shuffle_buffer}  "
    f"epoch_size={epoch_size}  steps_per_epoch={steps_per_epoch}  "
    f"batch_size={batch_size}"
)

try:
    for ms in model_states:
        name = ms["name"]

        print(f"\n{'#' * 72}")
        print(f"  Training: {name}  ".center(72, "#"))
        print(f"{'#' * 72}\n")

        ms["model"] = load_tf_model(ms["load_from"])
        ms["current_lw"] = None
        ms["metrics_history"] = {"value_mse": [], "value_corr": []}
        ms["epoch"] = 0

        # compile once with unit loss weights — LW schedule via sample_weight scaling
        set_loss_weights(
            ms["model"],
            {"policy_logits": 1.0, "value_out": 1.0},
            jit=False,
        )

        batch_stream = make_batch_stream(
            TFREC_DIR,
            batch_size,
            shuffle_buffer,
        )
        worker = EpochBufferWorker(batch_stream, steps_per_epoch, max_ready=2)
        worker.start()

        epoch = 0
        epoch_time_list = []
        begin = time.time()

        t_fetch = 0.0
        t_eval = 0.0
        t_fit = 0.0
        total_samples  = 0
        window_samples = 0

        while epoch <= MAX_EPOCH:
            ep = ms["epoch"]
            do_plot = (ep % PLOT_EVERY == 0)
            epoch_start = time.time()

            target_lw = LW_SCHEDULE[max([k for k in LW_SCHEDULE if k <= ep])]
            if target_lw is not ms["current_lw"]:
                ms["current_lw"] = target_lw
                lw_str = "  ".join([f"{k}={v}" for k, v in target_lw.items()])
                print(f"[lw update  ] {name} epoch {ep}: {lw_str}")

            t0 = time.time()
            bundle = worker.get()
            t_fetch += time.time() - t0

            if do_plot:
                t0 = time.time()
                do_eval(ms, bundle, batch_size)
                t_eval += time.time() - t0

            print("-" * 89)
            t0 = time.time()
            hist = fit_epoch(ms["model"], bundle, batch_size, target_lw)
            t_fit += time.time() - t0

            n_this = bundle["enc_in"].shape[0]
            total_samples  += n_this
            window_samples += n_this

            h = hist.history
            p_loss = h.get("policy_logits_loss", [float("nan")])[0]
            v_loss = h.get("value_out_loss", [float("nan")])[0]
            t_loss = h.get("loss", [float("nan")])[0]

            print(
                f"[epoch {ep:4d}] [{name}] "
                f"policy_loss: {p_loss:.4f}  "
                f"value_loss: {v_loss:.4f}  "
                f"total: {t_loss:.4f}  "
                f"samples: {total_samples:,}"
            )

            ms["epoch"] += 1
            epoch += 1

            if ep > 0 and ep % 25 == 0:
                ms["model"].save(ms["model_file"])
                print(f"[save] {name} → {ms['model_file']}")

            if ep > 0 and ep % 50 == 0:
                del ms["model"]
                ms["model"] = None
                tf.keras.backend.clear_session()
                gc.collect()
                print(f"[reload] {name} session cleared at epoch {ep}, reloading...")
                ms["model"] = load_tf_model(ms["model_file"])
                set_loss_weights(
                    ms["model"],
                    {"policy_logits": 1.0, "value_out": 1.0},
                    jit=False,
                )
                ms["current_lw"] = None  # force LW schedule re-apply next iter
                print(f"[reload] {name} ready, resuming from epoch {ep + 1}")

            epoch_time = time.time() - epoch_start
            epoch_time_list.append(epoch_time)

            print("-" * 89)
            print(
                f"[time check] last: {format_time(epoch_time)}  "
                f"avg: {format_time(np.mean(epoch_time_list))}  "
                f"total: {format_time(time.time() - begin)}"
            )

            if do_plot and epoch > 0:
                n = PLOT_EVERY
                print(
                    f"[timing/{n}ep] fetch: {t_fetch:.2f}s  "
                    f"fit: {t_fit:.2f}s  "
                    f"eval: {t_eval:.2f}s  "
                    f"samples_window: {window_samples:,}  "
                    f"samples_total: {total_samples:,}"
                )
                t_fetch = 0.0
                t_eval = 0.0
                t_fit = 0.0
                window_samples = 0

            print()

        worker.stop()
        ms["model"].save(ms["model_file"])
        print(f"[save] final {name} → {ms['model_file']}")

        del ms["model"]
        ms["model"] = None
        tf.keras.backend.clear_session()
        gc.collect()

except KeyboardInterrupt:
    print("\n[bootstrap] KeyboardInterrupt")