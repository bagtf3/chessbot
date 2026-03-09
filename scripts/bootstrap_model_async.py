import os, pickle, random, threading, time
from chessbot import MODEL_DIR, SP_DIR
from chessbot.utils import batch_policy_metrics, print_validation, format_time
from chessbot.review import GameViewer, load_game_index, ANALYZE_PKL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chessbot.model import set_loss_weights
from tensorflow import keras
import tensorflow as tf


# ── model + paths ─────────────────────────────────────────────────────────────

def load_model(model_loc):
    return keras.models.load_model(model_loc)

model_file_init = (
    'C:/Users/Bryan/Data/chessbot_data/selfplay_runs'
    '/val_test_film/val_test_film_model.h5'
)
model = load_model(model_file_init)
# set attrs required by set_loss_weights (not saved by speed-test script)
model._default_opt = model.optimizer
model._default_loss_dict = {
    "policy_logits": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    "value_out":     tf.keras.losses.MeanSquaredError(),
}
run_tag  = "val_test_film"
run_dir  = os.path.join(SP_DIR, run_tag)
model_file    = os.path.join(run_dir, run_tag + "_model.h5")
progress_file = os.path.join(run_dir, "eval_progress.csv")

metrics_history = {"value_mse": [], "value_corr": []}
epoch_time_list = []


os.makedirs(run_dir, exist_ok=True)
if not os.path.exists(model_file):
    model.save(model_file)
#%%
# ── game metadata ─────────────────────────────────────────────────────────────

run_tags = ["cf_10x256x5_full_sims3", "cf_10x256x5_full_sims4", "cf_10x256x5_deep_sims1"]

def check(a):
    if isinstance(a.get('pkl_file'), str):
        return os.path.exists(a['pkl_file'])
    return False

all_games, df_list = [], []
for rt in run_tags:
    rd = os.path.join(SP_DIR, rt)
    ag = load_game_index(rd)
    all_games += [a for a in ag if check(a)]

    pkl = os.path.join(rd, ANALYZE_PKL)
    with open(pkl, "rb") as f:
        prev_run = pickle.load(f)

    df_all   = prev_run['df_all']
    df_means = prev_run['df_means']
    df_all['clipped_loss'] = np.clip(df_all['loss'], -1000, 1000)
    clipped_cpl = df_all.groupby("game_id")['clipped_loss'].mean()
    df_means['overall_cpl'] = df_means.game_id.map(clipped_cpl)
    df_means['run_tag'] = rt
    df_list.append(df_means)
    del df_all, df_means

df_trim = pd.concat(df_list).drop_duplicates(['game_id']).sort_values("ts")
df_trim = df_trim.query("scenario != 'random_endgame'").copy()

meta = pd.DataFrame(all_games).drop_duplicates("game_id")
if "overall_cpl" in meta.columns:
    del meta['overall_cpl']
meta = meta.query("plies >= 10")
meta = meta.merge(df_trim[['game_id', 'run_tag', 'overall_cpl']], on='game_id')
meta = meta.query("overall_cpl <= 30")

training_games = meta['pkl_file'].to_list()
recycle_games  = meta.query("overall_cpl <= 25")['pkl_file'].to_list()
print(f"[bootstrap] {len(training_games)} training games, "
      f"{len(recycle_games)} recycle games")
random.shuffle(training_games)


# ── async buffer filler ───────────────────────────────────────────────────────

class BufferFiller:
    """
    Background thread that loads games and fills the buffer up to buffer_size.
    Sleeps when full; the main thread samples from it while the GPU trains.
    """

    def __init__(self, training_games, recycle_games, buffer_size, draw_rate=0.5):
        self.games         = list(training_games)
        self.recycle_games = list(recycle_games)
        self.buffer_size   = buffer_size
        self.draw_rate     = draw_rate

        self.buffer  = []
        self.lock    = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._fill_loop, daemon=True)

        self.idx             = 0
        self.n_loaded        = 0
        self.n_skipped_draw  = 0
        self.n_skipped_short = 0
        self.n_skipped_empty = 0
        self.recycles        = 0

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def buf_len(self):
        with self.lock:
            return len(self.buffer)

    def _recycle(self):
        self.games = list(self.recycle_games)
        random.shuffle(self.games)
        self.idx = 0
        self.recycles += 1
        print(f"\n[filler] recycled (recycle #{self.recycles})")

    def _fill_loop(self):
        while not self._stop.is_set():
            if self.buf_len() >= self.buffer_size:
                time.sleep(0.05)
                continue

            if self.idx >= len(self.games):
                self._recycle()

            game = self.games[self.idx]
            self.idx += 1

            gv = GameViewer(game, sf_df=None)

            if gv.result == 0:
                if random.random() > self.draw_rate:
                    self.n_skipped_draw += 1
                    continue

            if len(gv.moves_uci) < 10:
                self.n_skipped_short += 1
                continue

            X, M, P, Z, V, R = gv.generate_training_data(
                sf_skip=False, check_boost=0, capture_boost=0
            )
            if not X:
                self.n_skipped_empty += 1
                continue

            y = [0.99 * a + 0.1 * b for a, b in zip(Z, V)]
            w = 0.5 if gv.result == 0 else 1.0
            items = list(zip(X, M, P, y, [w] * len(X)))
            self.n_loaded += 1

            with self.lock:
                self.buffer += items

    def sample(self, epoch_size):
        with self.lock:
            n = len(self.buffer)
            if n == 0:
                return None
            k = min(epoch_size, n)
            random.shuffle(self.buffer)
            sel          = self.buffer[:k]
            self.buffer  = self.buffer[k:]

        Xs, Ms, Ps, Ys, Ws = zip(*sel)
        return (np.stack(Xs), np.stack(Ms), np.stack(Ps),
                np.stack(Ys), np.array(Ws, dtype=np.float32))

    def report(self):
        print(
            f"[filler] loaded={self.n_loaded:,}  "
            f"skip_draw={self.n_skipped_draw}  skip_short={self.n_skipped_short}  "
            f"skip_empty={self.n_skipped_empty}  buf={self.buf_len():,}  "
            f"idx={self.idx}/{len(self.games)}  recycles={self.recycles}"
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def moving_average_pd(arr, window=15):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().values


# ── training setup ────────────────────────────────────────────────────────────

batch_size  = 512
epoch_size  = 10240
buffer_size = 6 * epoch_size
MAX_EPOCH   = 1000
draw_rate   = 0.25
PLOT_EVERY  = 5

# loss weight schedule — acts as a hacky LR decay via effective loss scale
# keyed by epoch at which the new weights take effect
LW_SCHEDULE = {
    0:   {"policy_logits": 1.5,  "value_out": 2.25},
    100: {"policy_logits": 1.0,  "value_out": 1.5},
    200: {"policy_logits": 0.8,  "value_out": 1.2},
    400: {"policy_logits": 0.65, "value_out": 0.975},
    600: {"policy_logits": 0.45, "value_out": 0.675},
    800: {"policy_logits": 0.35, "value_out": 0.525},
}
_current_lw = None   # track applied weights to avoid redundant recompiles

eval_df = pd.read_csv(progress_file) if os.path.exists(progress_file) else None

filler = BufferFiller(training_games, recycle_games, buffer_size, draw_rate)
filler.start()

# wait for initial buffer fill before training starts
print(f"[bootstrap] Filling buffer ({buffer_size:,} samples)...")
while filler.buf_len() < buffer_size:
    time.sleep(1.0)
    print(f"\r[bootstrap] Buffer: {filler.buf_len():,} / {buffer_size:,}",
          end="", flush=True)
print()
filler.report()

# ── training loop ─────────────────────────────────────────────────────────────
begin = time.time()
epoch = 0
#%%
try:
    while epoch <= MAX_EPOCH:
        # apply loss weight schedule — recompile only on transitions
        target_lw = LW_SCHEDULE[max(k for k in LW_SCHEDULE if k <= epoch)]
        if target_lw is not _current_lw:
            set_loss_weights(model, target_lw)
            _current_lw = target_lw
            lw_str = "  ".join(f"{k}={v}" for k, v in target_lw.items())
            print(f"[lw update  ] epoch {epoch}: {lw_str}")

        epoch_start = time.time()
        
        if filler.buf_len() < 3*epoch_size:
            print("[bootstrap] Buffer low, waiting for filler...")
            time.sleep(1.0)
            continue
            
        # if filler hasn't caught up yet, wait
        batch = filler.sample(epoch_size)
        if batch is None:
            print("[bootstrap] Buffer empty, waiting for filler...")
            time.sleep(1.0)
            continue

        Xstack, Mstack, Pstack, Ystack, Wstack = batch

        if epoch % PLOT_EVERY == 0:
            preds = model.predict(Xstack, verbose=0, batch_size=batch_size)

            value_preds = preds[1].ravel()
            targets     = np.asarray(Ystack).ravel()
            plt.scatter(targets, value_preds, s=6)
            plt.plot([-1, 1], [-1, 1], linestyle="--", color="red", alpha=0.6)
            plt.xlim(-1, 1); plt.ylim(-1, 1)
            plt.xlabel("target"); plt.ylabel("pred")
            plt.title("pred vs target"); plt.show()

            policy_logits = preds[0]
            value_preds   = preds[1].ravel()

            policy_stats = batch_policy_metrics(policy_logits, Pstack, Mstack)
            value_mse    = np.mean((value_preds - targets) ** 2)
            value_corr   = np.corrcoef(value_preds, targets)[0, 1]

            for k, v in policy_stats.items():
                if k not in metrics_history:
                    metrics_history[k] = []
                metrics_history[k].append(v)
            metrics_history['value_mse'].append(value_mse)
            metrics_history['value_corr'].append(value_corr)

            latest_row = pd.DataFrame(metrics_history).iloc[[-1], :]
            n_samples  = Xstack.shape[0]
            if epoch > 0:
                n_samples *= PLOT_EVERY
            latest_row['n_samples'] = n_samples

            if os.path.exists(progress_file):
                eval_df = pd.read_csv(progress_file)
            if eval_df is None:
                latest_row['model_epoch'] = 0
                eval_df = latest_row.copy()
            else:
                latest_row['model_epoch'] = eval_df.tail(1)["model_epoch"].item() + 1
                latest_row = latest_row.reindex(columns=eval_df.columns)
                eval_df = pd.concat([eval_df, latest_row])
            eval_df.to_csv(progress_file, index=False)

            print_metrics = {"value_mse": value_mse, "value_corr": value_corr}
            print_metrics.update(policy_stats)
            print_validation(epoch, print_metrics)
            filler.report()

            retrains   = eval_df.tail(1)['model_epoch'].item()
            hide_first = max(int(0.1 * retrains), 5)
            ma_window  = min(max(3, int(retrains * 0.2)), 15)
            if ma_window % 2 == 0:
                ma_window += 1

            x  = np.arange(len(eval_df["policy_ce"]))
            xs = x[hide_first:]
            if len(xs) > hide_first:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                for ax, key, title in [
                    (axes[0, 0], "policy_ce",  "policy CE (nats)"),
                    (axes[0, 1], "ce_gain",     "CE gain vs uniform"),
                    (axes[1, 0], "value_mse",   "value MSE"),
                    (axes[1, 1], "value_corr",  "value corr"),
                ]:
                    raw = np.array(eval_df[key])[hide_first:]
                    ma  = moving_average_pd(eval_df[key], window=ma_window)[hide_first:]
                    if raw.size:
                        ax.plot(xs, raw, alpha=0.6, lw=1, label=key)
                    if ma.size:
                        ax.plot(xs, ma, lw=2, label=f"MA{ma_window}")
                    ax.set_title(title); ax.legend()

                plt.tight_layout(); plt.show()

        # ── train ─────────────────────────────────────────────────────────────
        print("-" * 89)
        hist = model.fit(
            Xstack,
            [Pstack, Ystack.astype(np.float32).reshape(-1, 1)],
            sample_weight=Wstack,
            batch_size=batch_size,
            epochs=1,
            verbose=0,
        )
        h      = hist.history
        p_loss = h.get("policy_logits_loss", [float("nan")])[0]
        v_loss = h.get("value_out_loss",     [float("nan")])[0]
        t_loss = h.get("loss",               [float("nan")])[0]
        print(f"[epoch {epoch:4d}] "
              f"policy_loss: {p_loss:.4f}  "
              f"value_loss: {v_loss:.4f}  "
              f"total: {t_loss:.4f}")

        epoch += 1
        epoch_time = time.time() - epoch_start
        epoch_time_list.append(epoch_time)
        print("-" * 89)
        print(f"[time check] last: {format_time(epoch_time)}  "
              f"avg: {format_time(np.mean(epoch_time_list))}  "
              f"total: {format_time(time.time() - begin)}  "
              f"buf: {filler.buf_len():,}")
        print()

        if epoch > 0 and epoch % 25 == 0:
            model.save(model_file)

finally:
    filler.stop()

model.save(model_file)
