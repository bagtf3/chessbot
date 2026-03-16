"""
retrain_worker.py

- run_dir contains:
  - pending_training/  (directory containing .tfrecord.gz shard files)
  - config.yaml        (the run config, loaded via Config.from_yaml)
- each shard is a gzipped TFRecord with fields:
  enc_in, mask, policy_logits, value_out, value_weight, policy_weight
- model path is taken from the loaded config (single) or cfg.multiplex_models (multi)
- in multiplex mode: shards are loaded once, each model is trained sequentially
- after all models are trained, shard files are deleted
"""

import argparse
import os, sys
import time
import gc

import numpy as np
import pandas as pd

import tensorflow as tf

from chessbot.model import load_model
from chessbot.config import Config
import chessbot.utils as cbu


TFREC_FEATURE_SPEC = {
    "enc_in":        tf.io.FixedLenFeature([], tf.string),
    "mask":          tf.io.FixedLenFeature([], tf.string),
    "policy_logits": tf.io.FixedLenFeature([], tf.string),
    "value_out":     tf.io.FixedLenFeature([], tf.float32),
    # old bootstrap files have a single "weight" field; new rescore files split into two
    "weight":        tf.io.FixedLenFeature([], tf.float32, default_value=1.0),
    "value_weight":  tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
    "policy_weight": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
}


def list_pending_shards(pending_dir):
    if not os.path.isdir(pending_dir):
        return []

    fns = [fn for fn in os.listdir(pending_dir)
           if fn.endswith(".tfrecord") or fn.endswith(".tfrecord.gz")]
    fns.sort()
    return [os.path.join(pending_dir, fn) for fn in fns]


def list_pending_shards_pkl(pending_dir):
    if not os.path.isdir(pending_dir):
        return []

    fns = sorted(fn for fn in os.listdir(pending_dir) if fn.endswith(".pkl"))
    return [os.path.join(pending_dir, fn) for fn in fns]


def load_shards_pkl(paths):
    import pickle
    X_list, M_list, P_list, Y_list, vwht_list, pwht_list = [], [], [], [], [], []
    loaded = []

    for path in paths:
        try:
            with open(path, "rb") as f:
                chunk = pickle.load(f)
            for x, mask, policy, Y, vwht, pwht in chunk:
                X_list.append(x)
                M_list.append(mask)
                P_list.append(policy)
                Y_list.append(Y)
                vwht_list.append(vwht)
                pwht_list.append(pwht)
            loaded.append(path)
        except Exception as e:
            print("[retrain] failed reading pkl shard (skipped):", path, e)

    return (X_list, M_list, P_list, Y_list, vwht_list, pwht_list), loaded


def load_shards(paths, tries=3, sleep_s=0.25):
    """Load bootstrap tfrecord shards. Returns lists of TF tensors."""
    X_list, M_list, P_list, Y_list, vwht_list, pwht_list = [], [], [], [], [], []
    loaded = []

    for path in paths:
        ok = False
        compression = "GZIP" if path.endswith(".gz") else ""
        for _ in range(tries):
            try:
                dataset = tf.data.TFRecordDataset(path, compression_type=compression)
                for raw in dataset:
                    feat = tf.io.parse_single_example(raw, TFREC_FEATURE_SPEC)
                    X_list.append(
                        tf.io.parse_tensor(feat["enc_in"], out_type=tf.int16)
                        .numpy().astype(np.int32))
                    M_list.append(
                        tf.io.parse_tensor(feat["mask"], out_type=tf.int32)
                        .numpy().astype(np.int32))
                    P_list.append(
                        tf.io.parse_tensor(feat["policy_logits"], out_type=tf.float32)
                        .numpy().astype(np.float32))
                    vw = feat["value_weight"].numpy()
                    if vw < 0:  # old single-weight format
                        w = feat["weight"].numpy().astype(np.float32)
                        vwht_list.append(w)
                        pwht_list.append(w)
                    else:
                        vwht_list.append(vw.astype(np.float32))
                        pwht_list.append(
                            feat["policy_weight"].numpy().astype(np.float32))
                    Y_list.append(feat["value_out"].numpy().astype(np.float32))
                loaded.append(path)
                ok = True
                break
            except Exception:
                time.sleep(sleep_s)

        if not ok:
            print("[retrain] failed reading shard (skipped):", path)
            try:
                os.remove(path)
                print("[retrain] deleted bad shard:", path)
            except Exception as e:
                print("[retrain] failed deleting bad shard:", path, e)

    return (X_list, M_list, P_list, Y_list, vwht_list, pwht_list), loaded


def export_tf_to_onnx(model_path, onnx_path):
    import tf2onnx
    model = load_model(model_path)
    input_sig = [tf.TensorSpec([None, 64], tf.int32, name="enc_in")]
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_sig, opset=17)
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"[retrain] tf→onnx export → {onnx_path}")


def delete_files(paths):
    removed = 0
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                removed += 1
        except Exception as e:
            print("[retrain] failed deleting:", path, "err:", e)
    return removed


def print_fit_history(history, epoch, label=""):
    if history is None:
        return

    h = getattr(history, "history", None)
    if not h:
        return

    rows = []
    for m, v in h.items():
        if not v:
            continue

        name = "total" if m == "loss" else m.replace("_loss", "")
        start = v[0]
        end = v[-1]
        delta = start - end
        mark = "*" if delta < 0 else "+"

        rows.append((name, start, end, delta, mark))

    if not rows:
        return

    name_w = max([len(r[0]) for r in rows])
    num_w = 8
    etag = f"[epoch {epoch:4d}]{(' ' + label) if label else ''}"
    fmt = (
        f"{etag} [model fit] "
        f"{{name:<{name_w}}} : value: {{start:{num_w}.4f}} -> "
        f"{{end:{num_w}.4f}}  delta: {{delta:{num_w}.4f}} {{mark}}"
    )

    for name, start, end, delta, mark in rows:
        print(fmt.format(
            name=name, start=start, end=end, delta=delta, mark=mark
        ))


def enforce_gpu_or_die(max_tries=5, sleep_s=1.0):
    tries = 0
    gpus = []
    while tries < max_tries:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            break
        time.sleep(sleep_s)
        tries += 1

    if not gpus:
        raise RuntimeError("TensorFlow sees no GPU. Refusing to run on CPU.")

    gpu = gpus[0]
    tf.config.set_visible_devices(gpu, "GPU")

    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("GPU was present but no logical GPU is active.")


def retrain_one_model(model_path, X, M, Y, s_wts, cfg, epoch, args, label=""):
    """Load, recompile, fit, and save a single model. Cleans up GPU memory after."""
    tag = f"[retrain{(' ' + label) if label else ''}]"
    print(f"{tag} loading {model_path}")
    model = load_model(model_path)

    # recompile: explicit head weights + fresh LR
    opt_src = getattr(model, '_default_opt', None)
    if isinstance(opt_src, tf.keras.mixed_precision.LossScaleOptimizer):
        base_cls = type(opt_src.inner_optimizer)
        base_cfg = opt_src.inner_optimizer.get_config()
    elif opt_src is not None:
        base_cls = type(opt_src)
        base_cfg = opt_src.get_config()
    else:
        base_cls = tf.keras.optimizers.Adam
        base_cfg = {}

    base_cfg['learning_rate'] = cfg.learning_rate
    inner_opt = base_cls.from_config(base_cfg)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(inner_opt)

    loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out": "mse",
    }
    head_weights = {"policy_logits": 1.0, "value_out": 1.0}
    model.compile(optimizer=opt, loss=loss_dict, loss_weights=head_weights)
    model._default_opt = opt
    model._default_loss_dict = loss_dict

    if not args.skip_plots:
        if os.path.exists(cfg.progress_csv_path):
            all_evals = pd.read_csv(cfg.progress_csv_path)
        else:
            all_evals = pd.DataFrame()

        plt_file = os.path.join(cfg.run_dir, "true_vs_pred_plot_latest.png")
        eval_df = cbu.score_game_data(model, X, M, Y, epoch, save_path=plt_file)
        all_evals = pd.concat([all_evals, eval_df])
        all_evals.round(5).to_csv(cfg.progress_csv_path, index=False)

        if len(all_evals) and len(all_evals) % 5 == 0:
            cbu.plot_training_progress(
                all_evals, epoch=epoch, save_path=cfg.progress_plot_path
            )

    history = model.fit(
        {"enc_in": X}, Y, epochs=args.epochs, batch_size=args.batch_size,
        verbose=0, sample_weight=s_wts, shuffle=True
    )

    print_fit_history(history, epoch, label=label)

    bak_path = model_path.replace(".h5", "_backup.h5")
    if os.path.exists(model_path):
        try:
            os.replace(model_path, bak_path)
            print(f"{tag} backed up existing model")
        except Exception as e:
            print(f"{tag} failed to backup existing model:", e)

    model.save(model_path)
    print(f"{tag} retraining complete for epoch {epoch}")

    del model
    tf.keras.backend.clear_session()
    gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="run directory")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--skip-plots", action="store_true",
                   help="skip all plots and CSV progress saves")
    args = p.parse_args()

    run_dir = args.run_dir
    config_file = os.path.join(run_dir, "config.yaml")
    cfg = Config.from_yaml(config_file)

    if cfg.retrain_backend == "pytorch":
        # give the GPU entirely to pytorch; TF only reads tfrecords on CPU
        tf.config.set_visible_devices([], 'GPU')
        from chessbot.train_pytorch import enforce_pytorch_gpu_or_die
        enforce_pytorch_gpu_or_die()
    else:
        enforce_gpu_or_die(max_tries=5, sleep_s=1.0)

    pending_dir = os.path.join(run_dir, "pending_training")
    pkl_paths = list_pending_shards_pkl(pending_dir)
    tfrec_paths = list_pending_shards(pending_dir)

    if pkl_paths:
        lists, loaded_shards = load_shards_pkl(pkl_paths)
        print(f"[retrain] loaded {len(loaded_shards)} pkl shards, "
              f"samples={len(lists[0])}")
    elif tfrec_paths:
        lists, loaded_shards = load_shards(tfrec_paths)
        print(f"[retrain] loaded {len(loaded_shards)} tfrecord shards, "
              f"samples={len(lists[0])}")
    else:
        print("[retrain] no shards found in:", pending_dir)
        return 0

    X_list, M_list, P_list, Y_list, vwht_list, pwht_list = lists

    if not X_list:
        print("[retrain] no training samples after loading shards")
        return 0

    X       = np.stack(X_list).astype(np.int32)
    M       = np.stack(M_list).astype(np.int32)
    P       = np.stack(P_list).astype(np.float32)
    Y_value = np.array(Y_list,    dtype=np.float32)
    vwht    = np.array(vwht_list, dtype=np.float32)
    pwht    = np.array(pwht_list, dtype=np.float32)

    valid = ~np.isnan(Y_value)
    n_invalid = int((~valid).sum())
    if n_invalid > 0:
        print(f"[retrain] {n_invalid} nan values found in Y, removing")
    idx     = np.random.permutation(valid.sum())
    X       = X[valid][idx]
    M       = M[valid][idx]
    P       = P[valid][idx]
    Y_value = Y_value[valid][idx]
    vwht    = vwht[valid][idx]
    pwht    = pwht[valid][idx]

    Y     = {"value_out": Y_value, "policy_logits": P}
    s_wts = {"value_out": vwht,    "policy_logits": pwht}

    draw_vwht = cfg.value_loss_weight * cfg.draw_value_scale
    kl_str = (
        f"KL boost ×{cfg.KL_weight_boost} when KL>{cfg.KL_boost_threshold}"
        if cfg.KL_weight_boost != 1.0 else "KL boost disabled"
    )
    print(f"[retrain] weights  lr={cfg.learning_rate}  "
          f"head policy=1.0  head value=1.0")
    print(f"[retrain] weights  sample policy={cfg.policy_loss_weight}  "
          f"sample value={cfg.value_loss_weight} (draw: {draw_vwht:.4f})")
    print(f"[retrain] weights  {kl_str}")
    for n, w in zip(['vwht', 'pwht'], [vwht, pwht]):
        mn, me, mx = float(w.min()), float(w.mean()), float(w.max())
        print(f"[retrain] {n}  min={mn:.4f}  mean={me:.4f}  max={mx:.4f}")

    if os.path.exists(cfg.progress_csv_path):
        n_retrains = len(pd.read_csv(cfg.progress_csv_path))
    else:
        n_retrains = 0
    epoch = n_retrains

    if cfg.retrain_backend == "pytorch":
        from chessbot.train_pytorch import load_pt_model, train_pt_model, save_pt_model
        model, arch = load_pt_model(cfg.pytorch_model_path)
        train_pt_model(model, X, M, P, Y_value, vwht, pwht, cfg, args)
        save_pt_model(model, cfg.pytorch_model_path, arch)
        print(f"[retrain] pytorch checkpoint saved → {cfg.pytorch_model_path}")
    else:
        model_paths = cfg.multiplex_models or [cfg.model_path]
        for model_path in model_paths:
            label = os.path.basename(model_path) if cfg.multiplex_models else ""
            retrain_one_model(model_path, X, M, Y, s_wts, cfg, epoch, args, label=label)

    removed = delete_files(loaded_shards)
    print(f"[retrain] deleted {removed} shard files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
