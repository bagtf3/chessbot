# chessbot/retrain_worker.py

import os, pickle, gc
import numpy as np
import pandas as pd
import tensorflow as tf

from chessbot.model import load_model
import chessbot.utils as cbu


def retrain_worker_main(run_dir, batch_file):
    """
    One-shot retrain worker.

    - run_dir: same run_dir as self-play
    - batch_file: pickle created by GameLooper with:
        {"combined": list_of_examples, "epoch": int, "cfg_dict": ...}
    """

    cfg = Config()
    cfg.run_dir = run_dir

    with open(batch_file, "rb") as f:
        payload = pickle.load(f)

    combined = payload["combined"]
    epoch = payload["epoch"]

    # Unpack examples, same as trigger_retrain
    X_list = []
    P_list = []
    mask_list = []
    Z_list = []
    Z_taper_list = []
    Vwq_list = []
    is_add_flag = []

    n_main = payload.get("n_main", len(combined))

    for i, (x, mask, policy, z_stm, vwq, z_tapered) in enumerate(combined):
        X_list.append(x)
        P_list.append(policy)
        mask_list.append(mask)
        Z_list.append(z_stm)
        Vwq_list.append(vwq)
        Z_taper_list.append(z_tapered)
        is_add_flag.append(i >= n_main)

    X = np.asarray(X_list, dtype=np.float32)
    P = np.stack(P_list, axis=0).astype(np.float32)
    M = np.stack(mask_list, axis=0).astype(np.int16)
    Z = np.asarray(Z_list, dtype=np.float32)
    Vwq = np.asarray(Vwq_list, dtype=np.float32)
    Z_taper_arr = np.asarray(Z_taper_list, dtype=np.float32)
    is_add = np.asarray(is_add_flag, dtype=np.bool_)

    # Blend value targets (same as your trigger_retrain)
    wz = cfg.target_y_weights.get("z", 0.25)
    wz_taper = cfg.target_y_weights.get("z_taper", 0.5)
    w_vwq = cfg.target_y_weights.get("vwq", 0.25)
    Y_value = wz*Z + wz_taper*Z_taper_arr + w_vwq*Vwq

    draw_mask = Z == 0
    played_by_winner = Z == 1
    if is_add.any():
        Y_value[is_add] = Vwq[is_add]
        draw_mask[is_add] = False
        played_by_winner[is_add] = True

    Y_value = np.clip(Y_value, -1.0, 1.0)

    weights = np.ones_like(Y_value, dtype=np.float32)
    lw = cfg.loss_weights

    # draw downweight + renorm (your logic)
    v_wts = lw["value_out"]*weights
    wts_before = v_wts.sum()
    v_wts[draw_mask] *= max(cfg.draw_weight, 0.001)
    wts_after = v_wts.sum()

    wt_lost = wts_before - wts_after
    n_non_draws = (~draw_mask).sum()
    if wt_lost > 0 and n_non_draws > 0:
        v_wts[~draw_mask] += wt_lost/n_non_draws

    p_wts_win = lw["policy_winner"]*weights
    p_wts_lose = lw["policy_loser"]*weights
    p_wts = np.where(played_by_winner, p_wts_win, p_wts_lose)

    Y = {"value_out": Y_value, "policy_logits": P}
    s_wts = {"value_out": v_wts, "policy_logits": p_wts}

    # Load model fresh in this worker
    model = load_model(cfg.model_path)

    # Eval + logging: same as your trigger_retrain
    plt_file = os.path.join(run_dir, "true_vs_pred_plot_latest.png")
    eval_df = cbu.score_game_data(model, X, M, Y, epoch, save_path=plt_file)
    if os.path.exists(cfg.progress_csv_path):
        prev = pd.read_csv(cfg.progress_csv_path)
        all_evals = pd.concat([prev, eval_df])
    else:
        all_evals = eval_df
    all_evals.round(5).to_csv(cfg.progress_csv_path, index=False)

    if len(all_evals) and len(all_evals) % 4 == 0:
        prog_plt_file = cfg.progress_plot_path
        cbu.plot_training_progress(all_evals, epoch=epoch, save_path=prog_plt_file)

    # Actual fit (same settings you like)
    history = model.fit(
        X, Y, epochs=3, batch_size=256, verbose=0,
        sample_weight=s_wts, shuffle=True
    )

    rows = []
    for m, v in history.history.items():
        name = "total" if m == "loss" else m.replace("_loss", "")
        start = v[0]
        end = v[-1]
        delta = start - end
        mark = "*" if delta < 0 else "+"
        rows.append((name, start, end, delta, mark))

    name_w = max(len(r[0]) for r in rows)
    num_w = 8
    fmt = (f"[epoch {epoch:4d}] [model fit]  "
           f"{{name:<{name_w}}} : value: {{start:{num_w}.4f}} -> "
           f"{{end:{num_w}.4f}}  delta: {{delta:{num_w}.4f}} {{mark}}")
    for name, start, end, delta, mark in rows:
        print(fmt.format(name=name, start=start, end=end, delta=delta, mark=mark))

    # Save updated weights back to same path
    model.save(cfg.model_path)

    del model, history
    gc.collect()
