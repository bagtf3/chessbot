"""
retrain_worker.py

- run_dir contains:
  - pending_training/  (directory containing shard .pkl files)
  - config.yaml        (the run config, loaded via Config.from_yaml)
- each shard is a list of tuples:
  (x, mask, policy, Y, vwht, pwht)
- model path is taken from the loaded config (single) or cfg.multiplex_models (multi)
- in multiplex mode: shards are loaded once, each model is trained sequentially
- after all models are trained, shard files are deleted
"""

import argparse
import os, sys
import time
import pickle
import gc

import numpy as np
import pandas as pd

import tensorflow as tf

from chessbot.model import load_model
from chessbot.config import Config
import chessbot.utils as cbu


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def list_pending_shards(pending_dir):
    if not os.path.isdir(pending_dir):
        return []

    fns = [fn for fn in os.listdir(pending_dir) if fn.endswith(".pkl")]
    fns = [fn for fn in fns if not fn.endswith(".tmp.pkl")]
    fns.sort()
    return [os.path.join(pending_dir, fn) for fn in fns]


def load_shards(paths, tries=3, sleep_s=0.25):
    combined = []
    loaded = []

    for path in paths:
        ok = False
        for _ in range(tries):
            try:
                items = load_pickle(path)
                if items:
                    combined += list(items)
                loaded.append(path)
                ok = True
                break
            except Exception:
                time.sleep(sleep_s)

        if not ok:
            print("[retrain] failed reading shard (skipped):", path)

            if path.lower().endswith(".pkl"):
                try:
                    os.remove(path)
                    print("[retrain] deleted bad shard:", path)
                except Exception as e:
                    print("[retrain] failed deleting bad shard:", path, e)

    return combined, loaded


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

    # ── recompile: explicit head weights + fresh LR ───────────────────────────
    _opt_src = getattr(model, '_default_opt', None)
    if isinstance(_opt_src, tf.keras.mixed_precision.LossScaleOptimizer):
        _base_cls = type(_opt_src.inner_optimizer)
        _base_cfg = _opt_src.inner_optimizer.get_config()
    elif _opt_src is not None:
        _base_cls = type(_opt_src)
        _base_cfg = _opt_src.get_config()
    else:
        _base_cls = tf.keras.optimizers.Adam
        _base_cfg = {}

    _base_cfg['learning_rate'] = cfg.learning_rate
    _inner = _base_cls.from_config(_base_cfg)
    _opt = tf.keras.mixed_precision.LossScaleOptimizer(_inner)

    _loss_dict = {
        "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        "value_out": "mse",
    }
    _head_weights = {"policy_logits": 1.0, "value_out": 1.0}
    model.compile(optimizer=_opt, loss=_loss_dict, loss_weights=_head_weights)
    model._default_opt = _opt
    model._default_loss_dict = _loss_dict

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
        X, Y, epochs=args.epochs, batch_size=args.batch_size,
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

    # make sure we are running on the GPU not CPU
    enforce_gpu_or_die(max_tries=5, sleep_s=1.0)

    run_dir = args.run_dir

    # load config from yaml and start looking for training shards
    config_file = os.path.join(run_dir, "config.yaml")
    cfg = Config.from_yaml(config_file)

    pending_dir = os.path.join(run_dir, "pending_training")
    shard_paths = list_pending_shards(pending_dir)
    if not shard_paths:
        print("[retrain] no shards found in:", pending_dir)
        return 0

    combined, loaded_shards = load_shards(shard_paths)
    print(f"[retrain] loaded {len(loaded_shards)} shards, samples={len(combined)}")

    if not combined:
        print("[retrain] no training samples after loading shards")
        return 0

    idx = np.random.permutation(len(combined))
    combined = [combined[i] for i in idx]

    X_list = []
    P_list = []
    mask_list = []
    Y_list = []
    vwht_list = []
    pwht_list = []

    for x, mask, policy, y, vwht, pwht in combined:
        # quick nan check, should never happen
        if np.isnan(y):
            print("[retrain] nan found in y value")
            continue

        X_list.append(x)
        P_list.append(policy)
        mask_list.append(mask)
        Y_list.append(y)
        vwht_list.append(vwht)
        pwht_list.append(pwht)

    X = np.asarray(X_list, dtype=np.int32)
    P = np.stack(P_list, axis=0).astype(np.float32)
    M = np.stack(mask_list, axis=0).astype(np.int32)
    Y_value = np.asarray(Y_list, dtype=np.float32)
    vwht = np.asarray(vwht_list, dtype=np.float32)
    pwht = np.asarray(pwht_list, dtype=np.float32)

    Y = {"value_out": Y_value, "policy_logits": P}
    s_wts = {"value_out": vwht, "policy_logits": pwht}

    # ── weight summary (printed once, applies to all models) ─────────────────
    draw_vwht = cfg.value_loss_weight * cfg.draw_value_scale
    kl_str = (
        f"KL boost ×{cfg.KL_weight_boost} when KL>{cfg.KL_boost_threshold}"
        if cfg.KL_weight_boost != 1.0 else "KL boost disabled"
    )
    print(f"[retrain] weights  lr={cfg.learning_rate}  head policy=1.0  head value=1.0")
    print(f"[retrain] weights  sample policy={cfg.policy_loss_weight}  "
          f"sample value={cfg.value_loss_weight} (draw: {draw_vwht:.4f})")
    print(f"[retrain] weights  {kl_str}")
    for n, w in zip(['vwht', 'pwht'], [vwht, pwht]):
        print(f"[retrain] {n}  min={w.min():.4f}  mean={w.mean():.4f}  max={w.max():.4f}")

    if os.path.exists(cfg.progress_csv_path):
        n_retrains = len(pd.read_csv(cfg.progress_csv_path))
    else:
        n_retrains = 0
    epoch = n_retrains

    # ── train: single model or multiplex loop ────────────────────────────────
    model_paths = cfg.multiplex_models if cfg.multiplex_models else [cfg.model_path]
    for i, model_path in enumerate(model_paths):
        label = os.path.basename(model_path) if cfg.multiplex_models else ""
        retrain_one_model(model_path, X, M, Y, s_wts, cfg, epoch, args, label=label)

    removed = delete_files(loaded_shards)
    print(f"[retrain] deleted {removed} shard files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
