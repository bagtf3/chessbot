"""
retrain_worker.py

- run_dir contains:
  - pending_training/  (directory containing .pkl shard files)
  - config.yaml        (the run config, loaded via Config.from_yaml)
- each shard is a pickle file with arrays: enc_in, mask (ignored), policy_logits, value_out, value_weight, policy_weight
- legacy .tfrecord/.tfrecord.gz shards are still supported as a fallback
- model path is taken from cfg.model_path
- after training, shard files are deleted
"""

import argparse
import os, sys
import time
import gc

import numpy as np

from chessbot.config import Config


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
    X_list, P_list, Y_list, vwht_list, pwht_list = [], [], [], [], []
    loaded = []

    for path in paths:
        try:
            with open(path, "rb") as f:
                chunk = pickle.load(f)
            for x, mask, policy, Y, vwht, pwht, *_ in chunk:
                X_list.append(x)
                P_list.append(policy)
                Y_list.append(Y)
                vwht_list.append(vwht)
                pwht_list.append(pwht)
            loaded.append(path)
        except Exception as e:
            print("[retrain] failed reading pkl shard (skipped):", path, e)

    return (X_list, P_list, Y_list, vwht_list, pwht_list), loaded


def load_shards(paths, tries=3, sleep_s=0.25):
    """Load bootstrap tfrecord shards. Returns lists of TF tensors."""
    import tensorflow as tf
    tfrec_feature_spec = {
        "enc_in":        tf.io.FixedLenFeature([], tf.string),
        "mask":          tf.io.FixedLenFeature([], tf.string),
        "policy_logits": tf.io.FixedLenFeature([], tf.string),
        "value_out":     tf.io.FixedLenFeature([], tf.float32),
        "weight":        tf.io.FixedLenFeature([], tf.float32, default_value=1.0),
        "value_weight":  tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
        "policy_weight": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
    }
    X_list, P_list, Y_list, vwht_list, pwht_list = [], [], [], [], []
    loaded = []

    for path in paths:
        ok = False
        compression = "GZIP" if path.endswith(".gz") else ""
        for _ in range(tries):
            try:
                dataset = tf.data.TFRecordDataset(path, compression_type=compression)
                for raw in dataset:
                    feat = tf.io.parse_single_example(raw, tfrec_feature_spec)
                    X_list.append(
                        tf.io.parse_tensor(feat["enc_in"], out_type=tf.int16)
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

    return (X_list, P_list, Y_list, vwht_list, pwht_list), loaded


def export_tf_to_onnx(model_path, onnx_path):
    import tensorflow as tf
    import tf2onnx
    model = tf.keras.models.load_model(model_path, compile=False)
    input_sig = [tf.TensorSpec([None, 64], tf.int32, name="enc_in")]
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_sig, opset=17)
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"[retrain] tf->onnx export -> {onnx_path}")


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
    import tensorflow as tf
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


def retrain_one_model(model_path, X, Y, s_wts, cfg, epoch, args, label="", timings=None):
    """Load, recompile, fit, and save a single model. Cleans up GPU memory after."""
    import tensorflow as tf
    if timings is None:
        timings = {}
    tag = f"[retrain{(' ' + label) if label else ''}]"
    short_model = os.path.join(
        os.path.basename(os.path.dirname(model_path)),
        os.path.basename(model_path))
    print(f"{tag} loading {short_model}")

    t0 = time.time()
    model = tf.keras.models.load_model(model_path, compile=False)

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

    base_cfg['learning_rate'] = cfg.learning_rate / 1.5
    base_cfg['beta_2'] = cfg.adam_beta2
    inner_opt = base_cls.from_config(base_cfg)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(inner_opt)

    def fp32_ce(y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(
            tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), from_logits=True)

    loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    }
    head_weights = {
        "policy_logits": cfg.policy_loss_weight,
        "value_out": cfg.value_loss_weight

    }
    model.compile(optimizer=opt, loss=loss_dict, loss_weights=head_weights)
    model._default_opt = opt
    model._default_loss_dict = loss_dict

    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    train_ckpts_dir = os.path.join(cfg.run_dir, "train_ckpts", model_stem)
    os.makedirs(train_ckpts_dir, exist_ok=True)
    epoch_var = tf.Variable(epoch, trainable=False, dtype=tf.int64)
    tf_ckpt   = tf.train.Checkpoint(optimizer=inner_opt, epoch=epoch_var)
    manager   = tf.train.CheckpointManager(tf_ckpt, train_ckpts_dir, max_to_keep=2)
    if manager.latest_checkpoint:
        tf_ckpt.restore(manager.latest_checkpoint)
        ckpt = manager.latest_checkpoint
        short_ckpt = os.path.join(
            os.path.basename(os.path.dirname(ckpt)),
            os.path.basename(ckpt))
        print(f"{tag} restored optimizer state from {short_ckpt}")
    else:
        print(f"{tag} no prior optimizer checkpoint - starting fresh")
    timings['load_model'] = timings.get('load_model', 0.0) + (time.time() - t0)

    t0 = time.time()
    history = model.fit(
        {"enc_in": X}, Y, epochs=1, batch_size=args.batch_size,
        verbose=0, sample_weight=s_wts, shuffle=True
    )
    timings['fit'] = timings.get('fit', 0.0) + (time.time() - t0)
    print_fit_history(history, epoch, label=label)

    t0 = time.time()
    bak_path = model_path.replace(".h5", "_backup.h5")
    if os.path.exists(model_path):
        try:
            os.replace(model_path, bak_path)
            print(f"{tag} backed up existing model")
        except Exception as e:
            print(f"{tag} failed to backup existing model:", e)

    model.save(model_path)
    epoch_var.assign(epoch)
    manager.save()
    ckpt = manager.latest_checkpoint
    short_ckpt = os.path.join(
        os.path.basename(os.path.dirname(ckpt)),
        os.path.basename(ckpt))
    print(f"{tag} retraining complete for epoch {epoch}  optimizer -> {short_ckpt}")
    timings['save'] = timings.get('save', 0.0) + (time.time() - t0)

    del model
    tf.keras.backend.clear_session()
    gc.collect()


def print_timings(timings):
    parts = []
    for key, label in [('load_shards', 'shards'), ('fit', 'fit'), ('trt_compile', 'trt'), ('total', 'total')]:
        if key in timings:
            parts.append(f"{label}={timings[key]:.1f}s")
    print(f"[retrain] {' '.join(parts)}")


def save_predictions_pkl(model, X_raw, pred_pkl_path, batch_size=256):
    import pickle
    import torch
    import torch.nn.functional as F
    device = torch.device("cuda")
    model = model.to(device).half().eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(X_raw), batch_size):
            xb = torch.from_numpy(X_raw[i:i + batch_size]).long().to(device)
            pol_logits, val_out = model(xb)
            pol_np = pol_logits.float().cpu().numpy()
            wdl_np = F.softmax(val_out.float(), dim=-1).cpu().numpy()
            for j in range(len(pol_np)):
                results.append((wdl_np[j].tolist(), pol_np[j].tolist()))
    with open(pred_pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[retrain] saved {len(results)} predictions -> {os.path.basename(pred_pkl_path)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="run directory")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epoch", type=int, default=0, help="retrain epoch number for logging")
    p.add_argument("--pred-pkl-path", default=None, help="path to save pre-retrain predictions")
    args = p.parse_args()

    t_total = time.time()
    timings = {}

    run_dir = args.run_dir
    config_file = os.path.join(run_dir, "config.yaml")
    cfg = Config.from_yaml(config_file)

    t0 = time.time()
    if cfg.retrain_backend == "pt_eager":
        from chessbot.train_pytorch import enforce_pytorch_gpu_or_die
        enforce_pytorch_gpu_or_die()
    else:
        enforce_gpu_or_die(max_tries=5, sleep_s=1.0)
    timings['backend_init'] = time.time() - t0

    pending_dir = os.path.join(run_dir, "pending_training")

    t0 = time.time()
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

    X_list, P_list, Y_list, vwht_list, pwht_list = lists

    if not X_list:
        print("[retrain] no training samples after loading shards")
        return 0

    # lc0 planes stay raw uint8 here (rule50 /99 + float happens in retrain_pt at
    # the model-input boundary); xc0 tokens are int32.
    if cfg.encoding_type == "lc0":
        X   = np.stack(X_list)
    else:
        X   = np.stack(X_list).astype(np.int32)
    P       = np.stack(P_list).astype(np.float32)
    Y_value = np.array(Y_list,    dtype=np.float32)
    vwht    = np.array(vwht_list, dtype=np.float32)
    pwht    = np.array(pwht_list, dtype=np.float32)
    timings['load_shards'] = time.time() - t0

    from chessbot.train_pytorch import load_pt_model, retrain_pt
    model, arch = load_pt_model(cfg.model_path)

    if args.pred_pkl_path:
        save_predictions_pkl(model, X, args.pred_pkl_path, batch_size=args.batch_size)

    valid = ~np.isnan(Y_value).any(axis=1) if Y_value.ndim == 2 else ~np.isnan(Y_value)
    n_invalid = int((~valid).sum())
    if n_invalid > 0:
        print(f"[retrain] {n_invalid} nan values found in Y, removing")
    idx     = np.random.permutation(valid.sum())
    X       = X[valid][idx]
    P       = P[valid][idx]
    Y_value = Y_value[valid][idx]
    vwht    = vwht[valid][idx]
    pwht    = pwht[valid][idx]

    Y     = {"value_out": Y_value, "policy_logits": P}
    s_wts = {"value_out": vwht,    "policy_logits": pwht}

    kl_str = (
        f"KL boost x{cfg.KL_weight_boost} when KL>{cfg.KL_boost_threshold}"
        if cfg.KL_weight_boost != 1.0 else "KL boost disabled"
    )
    draw_vwht = cfg.draw_value_scale
    print(f"[retrain] weights  lr={cfg.learning_rate}  "
          f"head policy={cfg.policy_loss_weight}  head value={cfg.value_loss_weight}")
    print(f"[retrain] weights  sample policy=1.0  "
          f"sample value=1.0 (draw: {draw_vwht:.4f})")
    print(f"[retrain] weights  {kl_str}")
    for n, w in zip(['vwht', 'pwht'], [vwht, pwht]):
        mn, me, mx = float(w.min()), float(w.mean()), float(w.max())
        print(f"[retrain] {n}  min={mn:.4f}  mean={me:.4f}  max={mx:.4f}")

    epoch = args.epoch

    print(f"[retrain] loss weights  value={cfg.value_loss_weight:.4f}  "
          f"policy={cfg.policy_loss_weight:.4f}  "
          f"ratio={cfg.value_loss_weight / cfg.policy_loss_weight:.4f}:1")

    retrain_pt(
        cfg.model_path, X, P, Y_value, vwht, pwht, cfg, epoch, args,
        label="", timings=timings, model=model, arch=arch,
    )

    removed = delete_files(loaded_shards)
    print(f"[retrain] deleted {removed} shard files")

    timings['total'] = time.time() - t_total
    print_timings(timings)

    return 0


if __name__ == "__main__":
    sys.exit(main())
