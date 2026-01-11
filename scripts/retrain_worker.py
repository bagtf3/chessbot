"""
retrain_worker.py

Expectations:
- run_dir contains:
  - pending_retrain.pkl   (list of (x, mask, policy, z_stm, vwq, z_tapered))
  - config.pkl           (a pickled dict produced by Config.to_dict())
- model path is taken from the config dict (cfg['model_path'])
- script writes model to same path atomically
"""

import argparse
import os, time
import pickle
import sys

import numpy as np
import pandas as pd

from chessbot.model import load_model, save_model
import chessbot.utils as cbu


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="run directory")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    run_dir = args.run_dir
    infile = os.path.join(run_dir, "pending_retrain.pkl")
    config_file = os.path.join(run_dir, "config.pkl")

    cfg = load_pickle(config_file)
    train = load_pickle(infile) # expect list
    
    # check for additional data pkl and load+remove if present
    add_pkl = os.path.join(cfg['run_dir'], "additional_training_data.pkl")
    additional = []
    
    if cfg.get("mine_bonus_data", False):
        if os.path.exists(add_pkl):
            # may hit an unlucky access deny if during a write.
            tries = 0
            while tries < 3:
                try:
                    additional = load_pickle(add_pkl)
                    # remove immediately so nothing is re-read later
                    os.remove(add_pkl)
                    n_add = len(additional)
                    print(f"[retrain] found {n_add} additional training samples")
                    break
                except:
                    # if we get an error, wait a bit and try again
                    time.sleep(0.5)
                    tries += 1
        
        if not additional:
            print("[retrain] no additional data found at", add_pkl)
    
    # combined list: existing queue first, additional appended
    combined = train + additional
    n_main = len(train)

    # Unpack examples
    X_list = []
    P_list = []        # flattened 4288 policy vectors
    mask_list = []     # derived legal-mask (0/1)
    Z_list = []
    Z_taper_list = []
    Vwq_list = []
    is_add_flag = []

    for i, (x, mask, policy, z_stm, vwq, z_tapered) in enumerate(combined):
        X_list.append(x)
        P_list.append(policy)
        mask_list.append(mask)
        Z_list.append(z_stm)
        Vwq_list.append(vwq)
        Z_taper_list.append(z_tapered)
        is_add_flag.append(i >= n_main)  # True for additional samples

    # stack arrays
    X = np.asarray(X_list, dtype=np.int32)               # (N, 64) expected
    P = np.stack(P_list, axis=0).astype(np.float32)      # (N, 4288)
    M = np.stack(mask_list, axis=0).astype(np.int32)     # (N, 4288)
    Z = np.asarray(Z_list, dtype=np.float32)
    Vwq = np.asarray(Vwq_list, dtype=np.float32)
    Z_taper_arr = np.asarray(Z_taper_list, dtype=np.float32)
    is_add = np.asarray(is_add_flag, dtype=np.bool_)

    # optionally blend (possibly tapered) outcome with visit-weighted Q
    wz = cfg['target_y_weights'].get("z", 0.25)
    wz_taper = cfg['target_y_weights'].get("z_taper", 0.5)
    w_vwq = cfg['target_y_weights'].get("vwq", 0.25)
    Y_value = wz*Z + wz_taper*Z_taper_arr + w_vwq*Vwq

    # define a few masks here then ensure additional
    # samples are not blended or downweighted
    draw_mask = Z == 0
    played_by_winner = Z == 1
    if is_add.any():
        Y_value[is_add] = Vwq[is_add]
        draw_mask[is_add] = False
        played_by_winner[is_add] = True
    
    Y_value = np.clip(Y_value, -1.0, 1.0)

    # initial per-sample weights (ones)
    weights = np.ones_like(Y_value, dtype=np.float32)
    lw = cfg['loss_weights']
    
    # downweight draws so value head doesnt collapse
    v_wts = lw['value_out']*weights
    wts_before = v_wts.sum()
    v_wts[draw_mask] *= max(cfg['draw_weight'], 0.001)
    wts_after = v_wts.sum()

    # redistribute to not change overall loss weight
    wt_lost = wts_before - wts_after
    n_non_draws = (~draw_mask).sum()
    if wt_lost > 0 and n_non_draws > 0:
        v_wts[~draw_mask] += wt_lost/n_non_draws
    
    # policy weights vary depend on outcome
    p_wts_win = lw['policy_winner']*weights
    p_wts_lose = lw['policy_loser']*weights
    p_wts = np.where(played_by_winner, p_wts_win, p_wts_lose)
    
    # assemble final Y dict and sample_weight mapping for TF fit
    Y = {"value_out": Y_value, "policy_logits": P}
    s_wts = {'value_out': v_wts, "policy_logits": p_wts}

    # untrained model
    model_path = cfg.get("model_path")
    model = load_model(model_path)

    # previous trains
    if os.path.exists(cfg['progress_csv_path']):
        all_evals = pd.read_csv(cfg['progress_csv_path'])
        n_retrains = len(all_evals)
    else:
        all_evals = pd.DataFrame()
        n_retrains = cfg['n_retrains']

    # evaluation and logging
    plt_file = os.path.join(cfg['run_dir'], "true_vs_pred_plot_latest.png")
    epoch = n_retrains
    eval_df = cbu.score_game_data(model, X, M, Y, epoch, save_path=plt_file)
    all_evals = pd.concat([all_evals, eval_df])
    all_evals.round(5).to_csv(cfg['progress_csv_path'], index=False)

    if len(all_evals) and len(all_evals) % 5 == 0:
        prog_plt_file = cfg['progress_plot_path']
        cbu.plot_training_progress(all_evals, epoch=epoch, save_path=prog_plt_file)

    # fit and save new model: X is a list/tuple matching model inputs (planes, mask)
    history = model.fit(
        X, Y, epochs=args.epochs, batch_size=args.batch_size,
        verbose=0, sample_weight=s_wts, shuffle=True
    )

    rows = []
    for m, v in history.history.items():
        name = "total" if m == "loss" else m.replace("_loss", "")
        start = v[0]; end = v[-1]
        delta = start - end
        mark = "*" if delta < 0 else "+"
        rows.append((name, start, end, delta, mark))
    
    name_w = max(len(r[0]) for r in rows)
    num_w = 8   # width for numbers
    fmt = (f"[epoch {epoch:4d}] [model fit]  "
        f"{{name:<{name_w}}} : value: {{start:{num_w}.4f}} -> "
        f"{{end:{num_w}.4f}}  delta: {{delta:{num_w}.4f}} {{mark}}")

    for name, start, end, delta, mark in rows:
        print(fmt.format(name=name, start=start, end=end, delta=delta, mark=mark))
    
    # checkpoint new weights, move existing model to .bak
    model_path = cfg['model_path']
    bak_path = model_path + ".bak"

    if os.path.exists(model_path):
        try:
            os.replace(model_path, bak_path)
            print(f"[retrain] backed up existing model")
        except Exception as e:
            print(f"[retrain] failed to backup existing model: {e}")

    # save new model to model_path
    model.save(model_path)
    print("[retrain] retraining complete for epoch", epoch)
    return 0


if __name__ == "__main__":
    sys.exit(main())


