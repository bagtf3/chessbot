"""bootstrap_model_async_tfrec_pt.py

PyTorch training on TFRecord data.  Structurally identical to
bootstrap_model_async_tfrec.py — same loop, same LW schedule, same plots,
same timing output.  Only training and eval are PyTorch.

TF is imported but restricted to CPU so it never touches the GPU.
PyTorch owns CUDA entirely.

Pre-saved .pt source models must exist in MODEL_DIR before running
(same requirement as the TF version for .h5).

Run from the repo root or from scripts/:
    python scripts/bootstrap_model_async_tfrec_pt.py
"""

import os

# Keep TF off the GPU before it initialises its CUDA context.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import gc
import queue
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from chessbot.utils import batch_policy_metrics, format_time, print_validation


# ---------------------------------------------------------------------------
# PT model builders  (self-contained copy; keep in sync with model_variant_speed_test.py)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 21
SEQ_LEN    = 64


def _make_ln2d(ch):
    class Ln2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(ch))
            self.b = nn.Parameter(torch.zeros(ch))
        def forward(self, x):
            c = x - x.mean(1, keepdim=True)
            return c * torch.rsqrt((c*c).mean(1, keepdim=True) + 1e-5) \
                   * self.w.view(1,-1,1,1) + self.b.view(1,-1,1,1)
    return Ln2d()


def _pt_attn_pool(x_seq, linear):
    w = torch.softmax(linear(x_seq), dim=1)
    return (x_seq * w).sum(dim=1)


def build_pt_transformer(cfg):
    de    = cfg["d_embed"]
    tl    = cfg["transformer_layers"]
    nh    = cfg["num_heads"]
    fd    = cfg["ff_dim"]
    dr    = cfg["dropout"]
    C_pol = 256  # spatial policy head width

    class TxLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1   = nn.LayerNorm(de)
            self.attn  = nn.MultiheadAttention(de, nh, dropout=0.0, batch_first=True)
            self.drop1 = nn.Dropout(dr)
            self.ln2   = nn.LayerNorm(de)
            self.ff1   = nn.Linear(de, fd)
            self.drop2 = nn.Dropout(dr)
            self.ff2   = nn.Linear(fd, de)

        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop1(h)
            return x + self.ff2(self.drop2(F.gelu(self.ff1(self.ln2(x)))))

    class PosReinject(nn.Module):
        def __init__(self):
            super().__init__()
            self.mix = nn.Parameter(torch.zeros(de))

        def forward(self, x, pos):
            gate = torch.sigmoid(self.mix).view(1, 1, de)
            return x + pos * gate

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb         = nn.Embedding(VOCAB_SIZE, de)
            self.pos         = nn.Embedding(64, de)
            self.txs         = nn.ModuleList([TxLayer() for _ in range(tl)])
            self.reinj2      = PosReinject()
            self.reinj4      = PosReinject()
            self.reinj_final = PosReinject()
            # spatial policy head: conv blocks ground per-square logits in local 8×8 context
            self.pol_mix  = nn.Conv2d(de, C_pol, 1, bias=False)
            self.pol_ln0  = _make_ln2d(C_pol)
            self.pol_c1a  = nn.Conv2d(C_pol, C_pol, 3, padding=1, bias=False)
            self.pol_c1b  = nn.Conv2d(C_pol, C_pol, 3, padding=1, bias=False)
            self.pol_ln1  = _make_ln2d(C_pol)
            self.pol_c2a  = nn.Conv2d(C_pol, C_pol, 3, padding=1, bias=False)
            self.pol_c2b  = nn.Conv2d(C_pol, C_pol, 3, padding=1, bias=False)
            self.pol_ln2  = _make_ln2d(C_pol)
            self.pol_out  = nn.Conv2d(C_pol, 67, 1)
            # value head (unchanged)
            self.ap   = nn.Linear(de, 1)
            self.vfc1 = nn.Linear(de, 256)
            self.vfc2 = nn.Linear(256, 128)
            self.vout = nn.Linear(128, 1)

        def forward(self, t):
            B   = t.shape[0]
            pos = self.pos(torch.arange(64, device=t.device)).unsqueeze(0)
            x   = self.emb(t) + pos
            for i, tx in enumerate(self.txs):
                x = tx(x)
                if i == 1:
                    x = self.reinj2(x, pos)
                if i == 3:
                    x = self.reinj4(x, pos)

            # value reads from sequence space (B, 64, de) before spatial reshape
            v = _pt_attn_pool(x, self.ap)
            v = F.relu(self.vfc1(v))
            v = F.relu(self.vfc2(v))

            # spatial policy head: conv blocks ground per-square logits in local 8×8 context
            x = self.reinj_final(x, pos)
            x = x.reshape(B, 8, 8, de).permute(0, 3, 1, 2).contiguous()
            x = F.leaky_relu(self.pol_ln0(self.pol_mix(x)), 0.01)
            r = x; x = F.leaky_relu(r + self.pol_ln1(self.pol_c1b(F.leaky_relu(self.pol_c1a(x), 0.01))), 0.01)
            r = x; x = F.leaky_relu(r + self.pol_ln2(self.pol_c2b(F.leaky_relu(self.pol_c2a(x), 0.01))), 0.01)
            pol = self.pol_out(x).permute(0, 2, 3, 1).reshape(B, SEQ_LEN * 67)

            return pol, torch.tanh(self.vout(v))

    m = M()
    print(f"  PT params: {sum([p.numel() for p in m.parameters()]):,}")
    return m


def build_pt_transformer_16m(cfg):
    """Pure transformer with steady positional redrip every tl//3 layers."""
    de    = cfg["d_embed"]
    tl    = cfg["transformer_layers"]
    nh    = cfg["num_heads"]
    fd    = cfg["ff_dim"]
    dr    = cfg["dropout"]
    reinject_every = max(1, tl // 3)  # for tl=8: reinject at i=3,6
    C_pol = 192  # spatial policy head width

    class TxLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1   = nn.LayerNorm(de)
            self.attn  = nn.MultiheadAttention(de, nh, dropout=0.0, batch_first=True)
            self.drop1 = nn.Dropout(dr)
            self.ln2   = nn.LayerNorm(de)
            self.ff1   = nn.Linear(de, fd)
            self.drop2 = nn.Dropout(dr)
            self.ff2   = nn.Linear(fd, de)

        def forward(self, x):
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = x + self.drop1(h)
            return x + self.ff2(self.drop2(F.gelu(self.ff1(self.ln2(x)))))

    class PosReinject(nn.Module):
        def __init__(self):
            super().__init__()
            self.mix = nn.Parameter(torch.zeros(de))

        def forward(self, x, pos):
            gate = torch.sigmoid(self.mix).view(1, 1, de)
            return x + pos * gate

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb         = nn.Embedding(VOCAB_SIZE, de)
            self.pos         = nn.Embedding(64, de)
            self.txs         = nn.ModuleList([TxLayer() for _ in range(tl)])
            self.reinj_final = PosReinject()
            # spatial policy head: conv blocks ground per-square logits in local 8×8 context
            self.pol_mix  = nn.Conv2d(de, C_pol, 1, bias=False)
            self.pol_ln0  = _make_ln2d(C_pol)
            self.pol_c1a  = nn.Conv2d(C_pol, C_pol, 3, padding=1, bias=False)
            self.pol_c1b  = nn.Conv2d(C_pol, C_pol, 3, padding=1, bias=False)
            self.pol_ln1  = _make_ln2d(C_pol)
            self.pol_c2a  = nn.Conv2d(C_pol, C_pol, 3, padding=1, bias=False)
            self.pol_c2b  = nn.Conv2d(C_pol, C_pol, 3, padding=1, bias=False)
            self.pol_ln2  = _make_ln2d(C_pol)
            self.pol_out  = nn.Conv2d(C_pol, 67, 1)
            # value head (unchanged)
            self.ap   = nn.Linear(de, 1)
            self.vfc1 = nn.Linear(de, 256)
            self.vfc2 = nn.Linear(256, 128)
            self.vout = nn.Linear(128, 1)

        def forward(self, t):
            B   = t.shape[0]
            pos = self.pos(torch.arange(64, device=t.device)).unsqueeze(0)
            x   = self.emb(t) + pos
            for i, tx in enumerate(self.txs):
                if i > 0 and i % reinject_every == 0:
                    x = x + pos  # steady redrip
                x = tx(x)

            # value reads from sequence space (B, 64, de) before spatial reshape
            v = _pt_attn_pool(x, self.ap)
            v = F.relu(self.vfc1(v))
            v = F.relu(self.vfc2(v))

            # spatial policy head: conv blocks ground per-square logits in local 8×8 context
            x = self.reinj_final(x, pos)
            x = x.reshape(B, 8, 8, de).permute(0, 3, 1, 2).contiguous()
            x = F.leaky_relu(self.pol_ln0(self.pol_mix(x)), 0.01)
            r = x; x = F.leaky_relu(r + self.pol_ln1(self.pol_c1b(F.leaky_relu(self.pol_c1a(x), 0.01))), 0.01)
            r = x; x = F.leaky_relu(r + self.pol_ln2(self.pol_c2b(F.leaky_relu(self.pol_c2a(x), 0.01))), 0.01)
            pol = self.pol_out(x).permute(0, 2, 3, 1).reshape(B, SEQ_LEN * 67)

            return pol, torch.tanh(self.vout(v))

    m = M()
    print(f"  PT params: {sum([p.numel() for p in m.parameters()]):,}")
    return m


VARIANTS = {
    "10m-transformer": dict(
        transformer=True,
        d_embed=256, transformer_layers=12, num_heads=8, ff_dim=1024, dropout=0.05,
    ),
    "16m-transformer": dict(
        transformer=True,
        d_embed=384, transformer_layers=8, num_heads=8, ff_dim=1536, dropout=0.05,
    ),
}

PT_BUILDERS = {
    "10m-transformer": build_pt_transformer,
    "16m-transformer": build_pt_transformer_16m,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TFREC_DIR = "C:/Users/Bryan/Data/chessbot_data/bootstrap_tfrecords"
RUN_DIR   = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/val_test_multi"
MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"

batch_size      = 512
epoch_size      = batch_size * 20
shuffle_buffer  = 64000
steps_per_epoch = epoch_size // batch_size
MAX_EPOCH       = 730
PLOT_EVERY      = 10
LR = 4e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DEFS = [
    "16m-transformer",
    "10m-transformer",
]

LW_SCHEDULE = {
    0: {"policy_logits": 0.1, "value_out": 0.1},
    1: {"policy_logits": 0.75, "value_out": 1.5},
    10: {"policy_logits": 1.5, "value_out": 3.0},
    160: {"policy_logits": 1.0, "value_out": 2.25},
    250: {"policy_logits": 1.0, "value_out": 1.5},
    480: {"policy_logits": 0.65, "value_out": 1.25},
    640: {"policy_logits": 0.5, "value_out": 1.0}
}


# ---------------------------------------------------------------------------
# TFRecord data pipeline  (TF on CPU — identical to the TF bootstrap)
# ---------------------------------------------------------------------------

feature_spec = {
    "enc_in":        tf.io.FixedLenFeature([], tf.string),
    "mask":          tf.io.FixedLenFeature([], tf.string),
    "policy_logits": tf.io.FixedLenFeature([], tf.string),
    "value_out":     tf.io.FixedLenFeature([], tf.float32),
    "weight":        tf.io.FixedLenFeature([], tf.float32),
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
    feat   = tf.io.parse_single_example(raw, feature_spec)
    enc_in = tf.cast(
        tf.io.parse_tensor(feat["enc_in"], out_type=tf.int16),
        tf.int32,
    )
    mask   = tf.io.parse_tensor(feat["mask"],          out_type=tf.int32)
    policy = tf.io.parse_tensor(feat["policy_logits"], out_type=tf.float32)
    value  = tf.reshape(feat["value_out"], [1])
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
        lambda path: tf.data.TFRecordDataset(path, compression_type="GZIP"),
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
    """Background thread that pre-fetches one epoch of TF tensors (CPU)."""

    def __init__(self, batch_stream, steps_per_epoch, max_ready=2):
        super().__init__(daemon=True)
        self.batch_stream    = batch_stream
        self.steps_per_epoch = steps_per_epoch
        self.ready           = queue.Queue(maxsize=max_ready)
        self.stop_event      = threading.Event()
        self.error           = None

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
                x_parts, m_parts, p_parts, y_parts, w_parts = [], [], [], [], []
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
                    "enc_in":        tf.concat(x_parts, axis=0),
                    "mask":          tf.concat(m_parts, axis=0),
                    "policy_logits": tf.concat(p_parts, axis=0),
                    "value_out":     tf.concat(y_parts, axis=0),
                    "weight":        tf.concat(w_parts, axis=0),
                }
                self.ready.put(bundle)

        except Exception as exc:
            self.error = exc
            try:
                self.ready.put(exc, timeout=1.0)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# PT model load / save  (mirrors load_tf_model + model.save)
# ---------------------------------------------------------------------------

def load_pt_model(path, name, device, lr=LR):
    """Load a saved .pt checkpoint.  Rebuilds model from VARIANTS/PT_BUILDERS,
    then restores weights + optimizer + scaler state.
    lr is always forced onto all param groups after loading so the checkpoint
    LR never silently overrides the script-level LR."""
    cfg    = VARIANTS[name]
    model  = PT_BUILDERS[name](cfg).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler("cuda")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    # force the current LR regardless of what the checkpoint stored
    for pg in opt.param_groups:
        pg["lr"] = lr

    return model, opt, scaler


def save_pt_model(ms):
    torch.save({
        "model":     ms["model"].state_dict(),
        "optimizer": ms["optimizer"].state_dict(),
        "scaler":    ms["scaler"].state_dict(),
        "epoch":     ms["epoch"],
    }, ms["model_file"])
    print(f"[save] {ms['name']} → {ms['model_file']}")


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def fit_epoch(ms, bundle, batch_size, lw, device):
    """One epoch of PT training over the pre-fetched bundle.

    Replicates TF sample_weight semantics:
        total_loss = lw_pol * mean(ce_i)  +  lw_val * mean(mse_i * w_i)

    Returns (policy_loss, value_loss, total_loss) averaged over mini-batches.
    """
    model  = ms["model"]
    opt    = ms["optimizer"]
    scaler = ms["scaler"]

    enc      = torch.as_tensor(bundle["enc_in"].numpy(),        dtype=torch.long).to(device)
    policy_t = torch.as_tensor(bundle["policy_logits"].numpy(), dtype=torch.float32).to(device)
    value_t  = torch.as_tensor(bundle["value_out"].numpy(),     dtype=torch.float32).to(device)
    w        = torch.as_tensor(bundle["weight"].numpy(),        dtype=torch.float32).to(device)

    n    = enc.shape[0]
    perm = torch.randperm(n, device=device)

    model.train()
    total_p = total_v = total = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        x  = enc[idx]
        pt = policy_t[idx]   # (B, 4288) soft targets
        vt = value_t[idx]    # (B, 1)
        wi = w[idx]          # (B,)

        opt.zero_grad(set_to_none=True)

        with autocast("cuda"):
            pol, val = model(x)
            # soft-label CE — PyTorch >= 1.10 supports float targets natively
            p_loss = F.cross_entropy(pol, pt) * lw["policy_logits"]
            v_loss = (
                F.mse_loss(val.squeeze(-1), vt.squeeze(-1), reduction="none") * wi
            ).mean() * lw["value_out"]
            loss = p_loss + v_loss

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        total_p += p_loss.item()
        total_v += v_loss.item()
        total   += loss.item()
        n_batches += 1

    return total_p / n_batches, total_v / n_batches, total / n_batches


# ---------------------------------------------------------------------------
# Eval + plotting
# ---------------------------------------------------------------------------

def moving_average_pd(arr, window=15):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().values


def do_eval(ms, bundle, batch_size, device):
    model = ms["model"]
    name  = ms["name"]
    epoch = ms["epoch"]

    enc = torch.as_tensor(bundle["enc_in"].numpy(), dtype=torch.long).to(device)

    model.eval()
    all_pol, all_val = [], []
    with torch.no_grad(), autocast("cuda"):
        for start in range(0, enc.shape[0], batch_size):
            pol, val = model(enc[start:start + batch_size])
            all_pol.append(pol.float().cpu().numpy())
            all_val.append(val.float().cpu().numpy())

    policy_logits = np.concatenate(all_pol, axis=0)
    value_preds   = np.concatenate(all_val, axis=0).ravel()

    pstack = bundle["policy_logits"].numpy()
    ystack = bundle["value_out"].numpy().ravel()
    mstack = bundle["mask"].numpy()

    policy_stats = batch_policy_metrics(policy_logits, pstack, mstack)
    value_mse    = np.mean((value_preds - ystack) ** 2)
    value_corr   = np.corrcoef(value_preds, ystack)[0, 1]

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

    retrains   = eval_df.tail(1)["model_epoch"].item()
    hide_first = max(int(0.1 * retrains), 5)
    ma_window  = min(max(3, int(retrains * 0.2)), 15)
    if ma_window % 2 == 0:
        ma_window += 1

    x  = np.arange(len(eval_df["policy_ce"]))
    xs = x[hide_first:]

    if len(xs) > hide_first:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"{name}  —  epoch {epoch}  [PT]")

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
            min(ystack.min(), value_preds.min()),
            max(ystack.max(), value_preds.max()),
        ]
        ax_sc.plot(lims, lims, "r--", lw=1)
        ax_sc.set_xlabel("target")
        ax_sc.set_ylabel("pred")
        ax_sc.set_title(f"value scatter  (r={value_corr:.3f})")

        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

model_states = []
for name in MODEL_DEFS:
    src_model_file = os.path.join(MODEL_DIR, f"{name}_pt_model.pt")
    run_model_file = os.path.join(RUN_DIR,   f"{name}_pt_model.pt")
    progress_file  = os.path.join(RUN_DIR,   f"{name}_pt_eval_progress.csv")

    print(f"[init] {name}: loading fresh from MODEL_DIR")
    model_states.append({
        "name":            name,
        "load_from":       src_model_file,
        "model_file":      run_model_file,
        "progress_file":   progress_file,
        "model":           None,
        "optimizer":       None,
        "scaler":          None,
        "eval_df":         None,
        "metrics_history": {"value_mse": [], "value_corr": []},
        "current_lw":      None,
        "epoch":           0,
    })

n_files = len(list_tfrecord_files(TFREC_DIR))
print(
    f"[tfrec] files={n_files}  shuffle_buffer={shuffle_buffer}  "
    f"epoch_size={epoch_size}  steps_per_epoch={steps_per_epoch}  "
    f"batch_size={batch_size}"
)
print(f"[device] {DEVICE}")

try:
    for ms in model_states:
        name = ms["name"]

        print(f"\n{'#' * 72}")
        print(f"  Training: {name}  [PyTorch]  ".center(72, "#"))
        print(f"{'#' * 72}\n")

        ms["model"], ms["optimizer"], ms["scaler"] = load_pt_model(
            ms["load_from"], name, DEVICE
        )
        ms["current_lw"]      = None
        ms["metrics_history"] = {"value_mse": [], "value_corr": []}
        ms["epoch"]           = 0

        batch_stream = make_batch_stream(TFREC_DIR, batch_size, shuffle_buffer)
        worker = EpochBufferWorker(batch_stream, steps_per_epoch, max_ready=2)
        worker.start()

        epoch           = 0
        epoch_time_list = []
        begin           = time.time()
        t_fetch = t_eval = t_fit = 0.0
        total_samples = window_samples = 0

        while epoch <= MAX_EPOCH:
            ep      = ms["epoch"]
            do_plot = (ep % PLOT_EVERY == 0)
            epoch_start = time.time()

            target_lw = LW_SCHEDULE[max(k for k in LW_SCHEDULE if k <= ep)]
            if target_lw is not ms["current_lw"]:
                ms["current_lw"] = target_lw
                lw_str = "  ".join(f"{k}={v}" for k, v in target_lw.items())
                print(f"[lw update  ] {name} epoch {ep}: {lw_str}")

            t0 = time.time()
            bundle = worker.get()
            t_fetch += time.time() - t0

            if do_plot:
                t0 = time.time()
                do_eval(ms, bundle, batch_size, DEVICE)
                t_eval += time.time() - t0

            print("-" * 89)
            t0 = time.time()
            p_loss, v_loss, t_loss = fit_epoch(ms, bundle, batch_size, target_lw, DEVICE)
            t_fit += time.time() - t0

            n_this          = bundle["enc_in"].shape[0]
            total_samples  += n_this
            window_samples += n_this

            print(
                f"[epoch {ep:4d}] [{name}] "
                f"policy_loss: {p_loss:.4f}  "
                f"value_loss: {v_loss:.4f}  "
                f"total: {t_loss:.4f}  "
                f"samples: {total_samples:,}"
            )

            ms["epoch"] += 1
            epoch        += 1

            if ep > 0 and ep % 25 == 0:
                save_pt_model(ms)

            if ep > 0 and ep % 50 == 0:
                save_pt_model(ms)
                del ms["model"], ms["optimizer"], ms["scaler"]
                ms["model"] = ms["optimizer"] = ms["scaler"] = None
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[reload] {name} VRAM cleared at epoch {ep}, reloading...")
                ms["model"], ms["optimizer"], ms["scaler"] = load_pt_model(
                    ms["model_file"], name, DEVICE
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
                t_fetch = t_eval = t_fit = 0.0
                window_samples = 0

            print()

        worker.stop()
        save_pt_model(ms)

        del ms["model"], ms["optimizer"], ms["scaler"]
        ms["model"] = ms["optimizer"] = ms["scaler"] = None
        gc.collect()
        torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n[bootstrap_pt] KeyboardInterrupt")
