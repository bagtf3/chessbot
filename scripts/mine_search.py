import os
import pickle

import pandas as pd
import numpy as np
import chess

from chessbot.utils import calc_entropy, softmax
from chessbot.review import load_json, load_game_index
from chessbot.review import combine_analysis_staging, ANALYZE_PKL

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def game_phase(board):
    # count non-pawn non-king pieces to detect endgame
    pieces = board.piece_map().values()
    pk = set([chess.PAWN, chess.KING])
    nonpawn_count = sum([1 for p in pieces if p.piece_type not in pk])
                        
    if nonpawn_count <= 4:
        return "endgame"
    if board.fullmove_number < 20:
        return "opening"
    return "middlegame"

# helper to print top-k features by score
def print_top_k(names, scores, k=12):
    order = np.argsort(-np.array(scores))
    for i in order[:k]:
        print(f"  {names[i]:<25} {scores[i]:.5f}")
    print()

# wrapper to run classification probe
def run_classification(target_name):
    print(f"\n=== Classification probe: {target_name} ===")
    y = Y[target_name].astype(int).values
    # skip if trivial
    if len(np.unique(y)) < 2:
        print("  target is constant; skipping")
        return

    Xtr, Xte, ytr, yte = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    # scaler = StandardScaler().fit(Xtr)
    # Xtr_s = scaler.transform(Xtr)
    # Xte_s = scaler.transform(Xte)

    # # # mutual information ranking
    # # mi = mutual_info_classif(Xtr_s, ytr, random_state=1)
    # # print("Top features by mutual info:")
    # # print_top_k(feat_names, mi, k=12)

    # logistic baseline
    # log = LogisticRegression(max_iter=400).fit(Xtr_s, ytr)
    # probs = log.predict_proba(Xte_s)[:, 1]
    # preds = (probs >= 0.5).astype(int)

    # print("Logistic metrics:")
    # print(f"  acc={accuracy_score(yte, preds):.3f} "
    #       f"prec={precision_score(yte, preds, zero_division=0):.3f} "
    #       f"rec={recall_score(yte, preds, zero_division=0):.3f} "
    #       f"auc={roc_auc_score(yte, probs):.3f}")

    # coefs = np.abs(log.coef_[0])
    # print("Top logistic coef magnitudes:")
    # print_top_k(feat_names, coefs, k=12)

    # small decision tree for rules
    # more conservative tree
    clf = DecisionTreeClassifier(
        max_depth=6,                # cap depth
        min_samples_split=200,      # require many samples before a split
        min_samples_leaf=100,       # require this many in any leaf
        min_impurity_decrease=1e-4, # require a non-trivial info gain
        random_state=1
    )
    clf.fit(Xtr, ytr)
    print("leaves", clf.get_n_leaves(), "depth", clf.get_depth())
    print(export_text(clf, feature_names=feat_names))
    
    print("Feature importances:")
    print_top_k(feat_names, clf.feature_importances_, k=12)


# wrapper to run regression probe
def run_regression(target_name):
    print(f"\n=== Regression probe: {target_name} ===")
    y = Y[target_name].values.astype(float)
    if np.all(y == y[0]):
        print("  target is constant; skipping")
        return

    Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=0.2,
                                          random_state=42)
    #scaler = StandardScaler().fit(Xtr)
    #Xtr_s = scaler.transform(Xtr)
    #Xte_s = scaler.transform(Xte)

    # mutual info regression ranking
    #mi = mutual_info_regression(Xtr_s, ytr, random_state=1)
    #print("Top features by mutual info (reg):")
    #rint_top_k(feat_names, mi, k=12)

    # small tree regressor for interpretability
    dt = DecisionTreeRegressor(max_depth=4).fit(Xtr, ytr)
    preds = dt.predict(Xte)
    mae = mean_absolute_error(yte, preds)
    rmse = mean_squared_error(yte, preds)
    r2 = r2_score(yte, preds)

    print(f"Tree reg metrics: MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    print("Decision tree rules (depth=4):")
    print(export_text(dt, feature_names=feat_names, show_weights=True))
    print("Feature importances:")
    print_top_k(feat_names, dt.feature_importances_, k=12)


run_dir = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/conv_1000_selfplay_phase2"
_ = combine_analysis_staging(run_dir)
all_games = load_game_index(run_dir)

pkl = os.path.join(run_dir, ANALYZE_PKL)
with open(pkl, "rb") as f:
    prev_run = pickle.load(f)
    
    
d = prev_run['df_all']
d['move_num'] = d['move_num'].astype(str)
scored_games = set(d.game_id.unique())
scored = [g for g in all_games if g['game_id'] in scored_games]

#x_list = []
#y_list = []

#%%
for i, game in enumerate(scored):
    if i % 50 == 0:
        print(f"mining {i}/{len(scored)}")
    
    gid = game['game_id']
    if game['plies'] < 10:
        print(f"skipping {gid}, too short")
        continue
    
    game_data = load_json(game['json_file'])
    df = d.loc[d.game_id == gid]
    if not len(df):
        print("no df found for", gid)
        continue
    
    dfi = df.set_index(['move_num', 'played_move'])
    
    tree_data = game_data['tree_search_data']
    vs_sf = game_data['vs_stockfish']
    sf_color = game_data['stockfish_is_white']
    
    board = chess.Board(game_data['start_fen'])    
    for mn, mv in enumerate(game_data['moves_played']):
        stm = board.turn
        if vs_sf and (stm == sf_color):
            continue
        
        uci_move = chess.Move.from_uci(mv)
        mn = str(mn)
        if (mn, mv) not in dfi.index:
            board.push(uci_move)
            continue
        
        row = dfi.loc[(mn, mv)]
        if isinstance(row, pd.DataFrame):
            print(mn, mv, gid, row.shape)
            raise Exception("OMG")
        
        search = tree_data.get(mn, False)
        if not search:
            board.push(uci_move)
            continue
        
        # if we made it here, we have good data
        this_y = [row['played_best_move'], row['in_top3'], row['loss']]
        y_list.append(this_y)
        
        cm = search['candidate_moves']
        cm = sorted(cm, key=lambda x: x['visits'], reverse=True)
        visits = [c['visits'] for c in cm]
        priors = [p['P'] for p in cm]
        
        visits_arr = np.array(visits, dtype=float)
        total = visits_arr.sum()
        n_cands = visits_arr.size
    
        # entropy / normalized entropy for visits and priors
        ent_vis, norm_vis = calc_entropy(visits_arr)
        ent_p, norm_p = calc_entropy(np.array(priors, dtype=float))
    
        # normalize Q so higher == better for side-to-move (stm)
        qs_raw = np.array([c.get("Q", 0.0) for c in cm], dtype=float)
        if board.turn == chess.BLACK:
            qs_stm = -qs_raw
        else:
            qs_stm = qs_raw
    
        q_probs = softmax(qs_stm)
        ent_q, norm_q = calc_entropy(q_probs)
    
        # U -> nonneg
        u_vals = np.array([c.get("U", 0.0) for c in cm], dtype=float)
        if u_vals.size:
            u_nonneg = u_vals - u_vals.min() if u_vals.min() < 0 else u_vals
            ent_u, norm_u = calc_entropy(u_nonneg)
        else:
            ent_u, norm_u = 0, 0
    
        # basic visit shares and gaps
        top_vis = visits_arr[0] if n_cands > 0 else 0
        second_vis = visits_arr[1] if n_cands > 1 else 0
        top_share = top_vis / max(1, total)
        top2_share = (top_vis + second_vis) / max(1, total)
        visit_margin = top_vis - second_vis
        nonzero_frac = np.count_nonzero(visits_arr) / max(1, n_cands)
    
        # Q / P top values and margins (Q in stm-space)
        top_q = qs_stm[0] if n_cands > 0 else 0
        second_q = qs_stm[1] if n_cands > 1 else 0
        q_margin = top_q - second_q
    
        priors_arr = np.array(priors, dtype=float)
        top_p = priors_arr[0] if n_cands > 0 else 0
        second_p = priors_arr[1] if n_cands > 1 else 0
        p_margin = top_p - second_p
    
        phase = game_phase(board)
        is_opening = (phase == "opening")
        is_middlegame = (phase == "middlegame")
        is_endgame = (phase == "endgame")
    
        priors_arr = np.array(priors, dtype=float)
        top_prior = priors_arr[0] if n_cands > 0 else 0
        max_prior = priors_arr.max() if n_cands > 0 else 0
        top_has_highest_prior = (top_prior >= (max_prior - 1e-12))
    
        # Q in stm-space already in qs_stm
        top_q = qs_stm[0] if n_cands > 0 else 0
        max_q = qs_stm.max() if n_cands > 0 else 0
        top_has_highest_q = (top_q >= (max_q - 1e-12))
        q_delta = top_q - max_q  # <=0 if some other move has larger Q
    
        features = {
            "n_cands": n_cands,
            "nonzero_frac": nonzero_frac,
            "top_share": top_share,
            "top2_share": top2_share,
            "visit_margin": visit_margin,
            "vis_norm": norm_vis,
            "p_norm": norm_p,
            "q_norm": norm_q,
            "u_norm": norm_u,
            "top_q": top_q,
            "top_p": top_prior,
            "q_margin": q_margin,
            "p_margin": p_margin,
            "top_has_highest_prior": int(bool(top_has_highest_prior)),
            "top_has_highest_q": int(bool(top_has_highest_q)),
            "q_delta": q_delta,
            "is_opening": int(bool(is_opening)),
            "is_middlegame": int(bool(is_middlegame)),
            "is_endgame": int(bool(is_endgame)),
        }
        
        x_list.append(features)
        board.push(uci_move)

#%%

# column names for targets
# y_cols = ["played_best", "in_top3", "cpl"]

# # build dataframes
# X = pd.DataFrame(x_list)
# Y = pd.DataFrame(y_list, columns=y_cols)

# # align sizes
# n = min(len(X), len(Y))
# X = X.iloc[:n].reset_index(drop=True)
# Y = Y.iloc[:n].reset_index(drop=True)

# # fill na
# X = X.fillna(0)
# Y = Y.fillna(0)

# Y['cpl'] = np.maximum(1, Y['cpl'])
# Y['under20_cpl'] = 1*(Y['cpl'] <= 20)
# # feature names



# save these DFs
# path = "C:/Users/Bryan/Data/chessbot_data/training_data/"
# X.to_parquet(path+"best_move_data_X.parquet", index=False)
# Y.to_parquet(path+"best_move_data_Y.parquet", index=False)

path = "C:/Users/Bryan/Data/chessbot_data/training_data/"
X = pd.read_parquet(path+"best_move_data_X.parquet")
Y = pd.read_parquet(path+"best_move_data_Y.parquet")
X = X.drop(columns=['top2_share', 'top_q', 'nonzero_frac', 'p_margin', 'is_opening'])
X = X.drop(columns=['top_has_highest_q', 'top_has_highest_prior'])
feat_names = list(X.columns)

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss
import numpy as np
import math
import time
import gc
import pandas as pd
from statistics import mean, median, stdev

def train_small_rf(X_vals, y, n_estimators=32, max_depth=4, random_state=1,
                   feature_names=None):
    # ensure DataFrame with stable column names
    if not isinstance(X_vals, pd.DataFrame):
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_vals.shape[1])]
        X_vals = pd.DataFrame(X_vals, columns=feature_names)
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=2
    )
    
    clf.fit(X_vals, y)
    return clf


# run and show
clf = train_small_rf(X, Y['played_best'].astype(int).values)
#%%
pred = None

import treelite.sklearn as tlskl
import tl2cgen as tl2

try:
    del pred
finally:
    model = tlskl.import_model(clf)
    libpath = "C:/Users/Bryan/Data/chessbot_data/models/sim_decision-v1.dll"
    tl2.export_lib(model, toolchain="msvc", libpath=libpath)
    pred = tl2.Predictor(libpath, nthread=1, verbose=False)
    
row = X.iloc[0:1].to_numpy(dtype=np.float32)
dmat = tl2.DMatrix(row)

for _ in range(10): pred.predict(dmat)
t0 = time.perf_counter()
for _ in range(2000): pred.predict(dmat)
dt = time.perf_counter() - t0
print("ms_per_call", dt / 2000 * 1000, "tps", 2000 / dt)


rows_df = X.iloc[:64]                  # keep df to avoid sklearn warnings
rows = rows_df.to_numpy(dtype=np.float32)
dmat = tl2.DMatrix(rows)

tre_preds = pred.predict(dmat)         # treelite often returns (n,1,k)
if tre_preds.ndim == 3 and tre_preds.shape[1] == 1:
    tre_preds = tre_preds[:, 0, :]     # squeeze the middle axis

sk_preds = clf.predict_proba(rows_df)  # pass DataFrame to avoid warning

print("shape tre", tre_preds.shape, "shape sk", sk_preds.shape)
print("max abs diff", np.max(np.abs(tre_preds - sk_preds)))
print("sample tre[0]", tre_preds[0], "sk[0]", sk_preds[0])


import os
import json
import tempfile
import tl2cgen as tl2
import numpy as np
import time

# save compiled lib (you already have a compiled file from export_lib)
# place lib at models/model-v1.so (or .dll/.dylib)
# save metadata next to it as models/model-v1.json
    
def write_model_artifact(libpath, feature_names, classes):
    base = os.path.splitext(libpath)[0]
    meta_path = base + ".json"
    meta = {
        "feature_names": list(feature_names),
        "classes": list([str(c) for c in classes]),
        "dtype": "float32"
    }
    
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, separators=(",", ":"))
        
    return libpath, meta_path


def load_predictor_and_meta(libpath):
    base = os.path.splitext(libpath)[0]
    meta_path = base + ".json"
    if not os.path.exists(libpath):
        raise RuntimeError("lib not found: " + libpath)
    if not os.path.exists(meta_path):
        raise RuntimeError("meta not found: " + meta_path)
    pred = tl2.Predictor(libpath, nthread=1)
    with open(meta_path, "r") as fh:
        meta = json.load(fh)
    return pred, meta

def predict_row(pred, meta, row_df):
    # enforce order + dtype, return numpy probs/score
    row = row_df[meta["feature_names"]].to_numpy(dtype=np.float32)
    dmat = tl2.DMatrix(row)
    out = pred.predict(dmat)
    if out.ndim == 3 and out.shape[1] == 1:
        out = out[:, 0, :]
    return out[0]    # single-row -> 1D array

# example usage:
write_model_artifact(libpath, clf.feature_names_in_, clf.classes_)
pred, meta = load_predictor_and_meta(libpath)
probs = predict_row(pred, meta, X.iloc[[69]])
#%%
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
PERM_REPEATS = 15
PERM_VAL_SIZE = 10000
CI_MULT = 2
MIN_IMPORTANCE = 1e-6

def feat_names_from_X(X):
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return [f"f{i}" for i in range(X.shape[1])]

def mdi_importances(clf, X):
    names = feat_names_from_X(X)
    mdi = pd.Series(clf.feature_importances_, index=names)
    return mdi.sort_values(ascending=False)

def per_tree_stats(clf, X):
    names = feat_names_from_X(X)
    imps = np.stack([est.feature_importances_ for est in clf.estimators_])
    mean = imps.mean(axis=0)
    std = imps.std(axis=0)
    df = pd.DataFrame({"mean": mean, "std": std}, index=names)
    return df.sort_values("mean", ascending=False)

def permutation_importances(clf, X, y):
    # use a sampled holdout from X/y for speed
    if PERM_VAL_SIZE < len(X):
        idx = pd.Series(X.index).sample(n=PERM_VAL_SIZE, random_state=RANDOM_STATE)
        Xp = X.loc[idx.values]
        yp = y.loc[idx.values]
    else:
        Xp, yp = X, y
    try:
        clf.set_params(n_jobs=1)
    except Exception:
        pass
    res = permutation_importance(clf, Xp, yp,
                                 n_repeats=PERM_REPEATS,
                                 random_state=RANDOM_STATE,
                                 n_jobs=-1)
    names = feat_names_from_X(X)
    df = pd.DataFrame({"mean": res.importances_mean,
                       "std": res.importances_std}, index=names)
    df["lower_ci"] = df["mean"] - CI_MULT * df["std"]
    df = df.sort_values("mean", ascending=False)
    df["keep_candidate"] = df["lower_ci"] > MIN_IMPORTANCE
    return df

# run
print("MDI (top 20)")
print(mdi_importances(clf, X).head(20))

print("\nPer-tree mean ± std (top 20)")
print(per_tree_stats(clf, X).head(20))

print("\nPermutation importances (holdout sample):")
p_df = permutation_importances(clf, X, Y['played_best'])
print(p_df.head(40))
meta['feature_names']

['n_cands',
 'top_share',
 'visit_margin',
 'vis_norm',
 'p_norm',
 'q_norm',
 'u_norm',
 'top_p',
 'q_margin',
 'q_delta',
 'is_middlegame',
 'is_endgame']




from chessbot.config import Config
from chessbot.utils import random_init
from chessbot.mcts_utils import MCTSTree
from chessbot.utils import calc_entropy

tree = MCTSTree(random_init(5), Config())


tree.collect_many_leaves(250, 250)
tree.sim_decision_model = tl2.Predictor(libpath, nthread=1)
vec, dmat, probs = tree.get_sim_decision_probs()


