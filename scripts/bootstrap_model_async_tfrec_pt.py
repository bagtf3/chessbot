"""bootstrap_model_async_tfrec_pt.py

PyTorch training on TFRecord data.  Structurally identical to
bootstrap_model_async_tfrec.py — same loop, same LW schedule, same plots,
same timing output.  Only training and eval are PyTorch.

TF is imported but restricted to CPU so it never touches the GPU.
PyTorch owns CUDA entirely.

Pre-saved .pt source models must exist in MODEL_DIR before running.

Run from the repo root or from scripts/:
    python scripts/bootstrap_model_async_tfrec_pt.py
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import gc
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from chessbot.pretrain import (
    BATCH_SIZE, EPOCH_SIZE, SHUFFLE_BUFFER, VAL_SHUFFLE_BUFFER,
    STEPS_PER_EPOCH, PLOT_EVERY, DEFAULT_LR, DEFAULT_MAX_EPOCH,
    lw_for_epoch,
    list_tfrecord_files, split_train_val,
    make_dataset, EpochBufferThread,
    save_plot,
)
from chessbot import SP_DIR, MODEL_DIR
from chessbot.utils import batch_policy_metrics, format_time, print_validation


# ---------------------------------------------------------------------------
# PT model builders
# (self-contained; keep in sync with model_variant_speed_test.py)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 21
SEQ_LEN    = 64


def make_ln2d(ch):
    class Ln2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(ch))
            self.b = nn.Parameter(torch.zeros(ch))

        def forward(self, x):
            c = x - x.mean(1, keepdim=True)
            return (
                c * torch.rsqrt((c * c).mean(1, keepdim=True) + 1e-5)
                * self.w.view(1, -1, 1, 1) + self.b.view(1, -1, 1, 1)
            )
    return Ln2d()


def pt_attn_pool(x_seq, linear):
    w = torch.softmax(linear(x_seq), dim=1)
    return (x_seq * w).sum(dim=1)


def build_pt_transformer_16m(cfg):
    """Pure transformer with steady pos drip.
    Tapered policy head: 256->192->67."""
    de = cfg["d_embed"]
    tl = cfg["transformer_layers"]
    nh = cfg["num_heads"]
    fd = cfg["ff_dim"]
    dr = cfg["dropout"]

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

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb      = nn.Embedding(VOCAB_SIZE, de)
            self.pos      = nn.Embedding(64, de)
            self.txs      = nn.ModuleList([TxLayer() for _ in range(tl)])
            self.pol_mix  = nn.Conv2d(de, 256, 1, bias=False)
            self.pol_ln0  = make_ln2d(256)
            self.pol_c1a  = nn.Conv2d(256, 256, 3, padding=1, bias=False)
            self.pol_c1b  = nn.Conv2d(256, 256, 3, padding=1, bias=False)
            self.pol_ln1  = make_ln2d(256)
            self.pol_down = nn.Conv2d(256, 192, 1, bias=False)
            self.pol_lnd  = make_ln2d(192)
            self.pol_c2a  = nn.Conv2d(192, 192, 3, padding=1, bias=False)
            self.pol_c2b  = nn.Conv2d(192, 192, 3, padding=1, bias=False)
            self.pol_ln2  = make_ln2d(192)
            self.pol_out  = nn.Conv2d(192, 67, 1)
            self.ap   = nn.Linear(de, 1)
            self.vfc1 = nn.Linear(de, 256)
            self.vfc2 = nn.Linear(256, 128)
            self.vout = nn.Linear(128, 1)

        def forward(self, t):
            B   = t.shape[0]
            pos = self.pos(torch.arange(64, device=t.device)).unsqueeze(0)
            x   = self.emb(t) + pos
            for i, tx in enumerate(self.txs):
                if i > 0:
                    x = x + 0.1 * pos
                x = tx(x)

            v = pt_attn_pool(x, self.ap)
            v = F.gelu(self.vfc1(v))
            v = F.gelu(self.vfc2(v))

            x = x + 0.2 * pos
            x = x.reshape(B, 8, 8, de).permute(0, 3, 1, 2).contiguous()
            x = F.leaky_relu(self.pol_ln0(self.pol_mix(x)), 0.01)
            r = x
            x = F.leaky_relu(
                r + self.pol_ln1(self.pol_c1b(F.leaky_relu(self.pol_c1a(x), 0.01))),
                0.01,
            )
            x = F.leaky_relu(self.pol_lnd(self.pol_down(x)), 0.01)
            r = x
            x = F.leaky_relu(
                r + self.pol_ln2(self.pol_c2b(F.leaky_relu(self.pol_c2a(x), 0.01))),
                0.01,
            )
            pol = self.pol_out(x).permute(0, 2, 3, 1).reshape(B, SEQ_LEN * 67)
            return pol, torch.tanh(self.vout(v))

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_conformer_interweaved(cfg):
    """Interweaved conformer: 10 x [MHA -> Conv -> Conv].
    Graduated pos injection: block 0=1.0, 2=0.1, 4=0.05, 6=0.025, others=none.
    """
    de = cfg["d_embed"]
    cf = cfg["conv_filters"]
    cb = cfg["conv_blocks"]
    nh = cfg["num_heads"]
    dr = cfg["dropout"]

    POS_SCALES = {0: 1.0, 2: 0.1, 4: 0.05, 6: 0.025}

    class Block(nn.Module):
        def __init__(self, pos_scale):
            super().__init__()
            self.pos_scale = pos_scale
            self.ln1  = nn.LayerNorm(cf)
            self.attn = nn.MultiheadAttention(cf, nh, dropout=0.0, batch_first=True)
            self.drop1 = nn.Dropout(dr)
            self.c1   = nn.Conv2d(cf, cf, 3, padding=1, bias=False)
            self.lr1  = nn.LeakyReLU(0.01, True)
            self.c2   = nn.Conv2d(cf, cf, 3, padding=1, bias=False)
            self.ln2  = make_ln2d(cf)

        def forward(self, x, pos):
            b = x.shape[0]
            s = x.permute(0, 2, 3, 1).reshape(b, 64, cf)
            if self.pos_scale != 0.0:
                s = s + self.pos_scale * pos
            r = s
            n = self.ln1(s)
            h, _ = self.attn(n, n, n, need_weights=False)
            s = r + self.drop1(h)
            x = s.reshape(b, 8, 8, cf).permute(0, 3, 1, 2).contiguous()
            r = x
            h = self.lr1(self.c1(x))
            h = self.ln2(self.c2(h))
            return F.leaky_relu(r + h, 0.01)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb    = nn.Embedding(VOCAB_SIZE, de)
            self.proj   = nn.Conv2d(de, cf, 1, bias=False) if de != cf else None
            self.lnp    = make_ln2d(cf) if de != cf else None
            self.pos    = nn.Embedding(64, cf)
            self.blocks = nn.ModuleList(
                [Block(POS_SCALES.get(i, 0.0)) for i in range(cb)]
            )
            self.mix  = nn.Conv2d(cf, cf, 1, bias=False)
            self.lnm  = make_ln2d(cf)
            self.pol  = nn.Conv2d(cf, 67, 1)
            self.v_ln = nn.LayerNorm(cf)
            self.ap   = nn.Linear(cf, 1)
            self.vfc1 = nn.Linear(cf, 256)
            self.vfc2 = nn.Linear(256, 128)
            self.vout = nn.Linear(128, 1)

        def forward(self, t):
            b = t.shape[0]
            x = self.emb(t).reshape(b, 8, 8, de).permute(0, 3, 1, 2).contiguous()
            if self.proj is not None:
                x = F.leaky_relu(self.lnp(self.proj(x)), 0.01)
            pos = self.pos(torch.arange(64, device=t.device)).unsqueeze(0)
            for block in self.blocks:
                x = block(x, pos)
            x = F.leaky_relu(self.lnm(self.mix(x)), 0.01)
            pol = self.pol(x).permute(0, 2, 3, 1).reshape(b, SEQ_LEN * 67)
            s = x.permute(0, 2, 3, 1).reshape(b, 64, cf)
            s = self.v_ln(s)
            v = pt_attn_pool(s, self.ap)
            v = F.relu(self.vfc1(v))
            v = F.relu(self.vfc2(v))
            return pol, torch.tanh(self.vout(v))

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


def build_pt_conformer_transheavy(cfg):
    """Alternating ConvBlock (MHA+Conv+Conv) and TxBlock (MHA+FF), odd total.
    Pattern for 11 blocks: Conv Tx Conv Tx Conv Tx Conv Tx Conv Tx Conv.
    Pos: full inject at start; taper-drip before each TxBlock.
    """
    cf = cfg["conv_filters"]
    nb = cfg["n_blocks"]
    nh = cfg["num_heads"]
    fd = cfg["ff_dim"]
    dr = cfg["dropout"]

    TX_POS = [0.1, 0.05, 0.025, 0.0, 0.0, 0.0]

    class ConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1   = nn.LayerNorm(cf)
            self.attn  = nn.MultiheadAttention(cf, nh, dropout=0.0, batch_first=True)
            self.drop1 = nn.Dropout(dr)
            self.c1    = nn.Conv2d(cf, cf, 3, padding=1, bias=False)
            self.lr1   = nn.LeakyReLU(0.01, True)
            self.c2    = nn.Conv2d(cf, cf, 3, padding=1, bias=False)
            self.ln2   = make_ln2d(cf)

        def forward(self, s):
            b = s.shape[0]
            r = s
            n = self.ln1(s)
            h, _ = self.attn(n, n, n, need_weights=False)
            s = r + self.drop1(h)
            x = s.reshape(b, 8, 8, cf).permute(0, 3, 1, 2).contiguous()
            r = x
            h = self.lr1(self.c1(x))
            h = self.ln2(self.c2(h))
            x = F.leaky_relu(r + h, 0.01)
            return x.permute(0, 2, 3, 1).reshape(b, 64, cf)

    class TxBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1   = nn.LayerNorm(cf)
            self.attn  = nn.MultiheadAttention(cf, nh, dropout=0.0, batch_first=True)
            self.drop1 = nn.Dropout(dr)
            self.ln2   = nn.LayerNorm(cf)
            self.ff1   = nn.Linear(cf, fd)
            self.drop2 = nn.Dropout(dr)
            self.ff2   = nn.Linear(fd, cf)

        def forward(self, x):
            r = x
            n = self.ln1(x)
            h, _ = self.attn(n, n, n, need_weights=False)
            x = r + self.drop1(h)
            return x + self.ff2(self.drop2(F.gelu(self.ff1(self.ln2(x)))))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb    = nn.Embedding(VOCAB_SIZE, cf)
            self.pos    = nn.Embedding(64, cf)
            self.blocks = nn.ModuleList(
                [ConvBlock() if i % 2 == 0 else TxBlock() for i in range(nb)]
            )
            self.mix  = nn.Conv2d(cf, cf, 1, bias=False)
            self.lnm  = make_ln2d(cf)
            self.pol  = nn.Conv2d(cf, 67, 1)
            self.v_ln = nn.LayerNorm(cf)
            self.ap   = nn.Linear(cf, 1)
            self.vfc1 = nn.Linear(cf, 256)
            self.vfc2 = nn.Linear(256, 128)
            self.vout = nn.Linear(128, 1)

        def forward(self, t):
            b = t.shape[0]
            pos = self.pos(torch.arange(64, device=t.device)).unsqueeze(0)
            s   = self.emb(t) + pos

            tx_idx = 0
            for block in self.blocks:
                if isinstance(block, TxBlock):
                    scale = TX_POS[tx_idx] if tx_idx < len(TX_POS) else 0.0
                    if scale != 0.0:
                        s = s + scale * pos
                    tx_idx += 1
                s = block(s)

            x = s.reshape(b, 8, 8, cf).permute(0, 3, 1, 2).contiguous()
            x = F.leaky_relu(self.lnm(self.mix(x)), 0.01)
            pol = self.pol(x).permute(0, 2, 3, 1).reshape(b, SEQ_LEN * 67)
            s = x.permute(0, 2, 3, 1).reshape(b, 64, cf)
            s = self.v_ln(s)
            v = pt_attn_pool(s, self.ap)
            v = F.relu(self.vfc1(v))
            v = F.relu(self.vfc2(v))
            return pol, torch.tanh(self.vout(v))

    m = M()
    print(f"  PT params: {sum(p.numel() for p in m.parameters()):,}")
    return m


VARIANTS = {
    "16m-transformer": dict(
        transformer=True,
        d_embed=384, transformer_layers=8, num_heads=8, ff_dim=1536, dropout=0.05,
    ),
    "16m-conformer-interweaved": dict(
        conformer_interweaved=True,
        d_embed=256, conv_filters=256, conv_blocks=10,
        num_heads=8, ff_dim=1024, dropout=0.05,
    ),
    "13m-conformer-transheavy": dict(
        conformer_transheavy=True,
        conv_filters=256, n_blocks=11,
        num_heads=8, ff_dim=1024, dropout=0.05,
    ),
}

PT_BUILDERS = {
    "16m-transformer":          build_pt_transformer_16m,
    "16m-conformer-interweaved": build_pt_conformer_interweaved,
    "13m-conformer-transheavy":  build_pt_conformer_transheavy,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TFREC_DIR = os.getenv("BOOTSTRAP_TFREC_DIR", "")
RUN_DIR   = os.path.join(SP_DIR, "val_test_multi")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DEFS = [
    "16m-transformer",
]


# ---------------------------------------------------------------------------
# PT model load / save
# ---------------------------------------------------------------------------

def load_pt_model(path, name, device, lr=DEFAULT_LR):
    """Load a saved .pt checkpoint.  Rebuilds model from VARIANTS/PT_BUILDERS,
    then restores weights + optimizer + scaler state.
    LR is always forced onto all param groups after loading."""
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

    for pg in opt.param_groups:
        pg["lr"] = lr

    return model, opt, scaler


def save_pt_model(ms):
    torch.save({
        "model":     ms["model"].state_dict(),
        "optimizer": ms["optimizer"].state_dict(),
        "scaler":    ms["scaler"].state_dict(),
        "epoch":     ms["epoch"],
        "arch":      ms["name"],
    }, ms["model_file"])
    print(f"[save] {ms['name']} -> {ms['model_file']}")


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def fit_epoch(ms, bundle, lw, device):
    """One epoch of PT training over the pre-fetched bundle."""
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

    for start in range(0, n, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        x   = enc[idx]
        pt  = policy_t[idx]
        vt  = value_t[idx]
        wi  = w[idx]

        opt.zero_grad(set_to_none=True)

        with autocast("cuda"):
            pol, val = model(x)
            p_loss = F.cross_entropy(pol, pt) * lw["policy_logits"]
            v_loss = (
                F.mse_loss(val.squeeze(-1), vt.squeeze(-1), reduction="none") * wi
            ).mean() * lw["value_out"]
            loss = p_loss + v_loss

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        total_p   += p_loss.item()
        total_v   += v_loss.item()
        total     += loss.item()
        n_batches += 1

    return total_p / n_batches, total_v / n_batches, total / n_batches


# ---------------------------------------------------------------------------
# Eval + plotting
# ---------------------------------------------------------------------------

def do_eval(ms, bundle, device):
    model = ms["model"]
    name  = ms["name"]
    epoch = ms["epoch"]

    enc = torch.as_tensor(bundle["enc_in"].numpy(), dtype=torch.long).to(device)

    model.eval()
    all_pol, all_val = [], []
    with torch.no_grad(), autocast("cuda"):
        for start in range(0, enc.shape[0], BATCH_SIZE):
            pol, val = model(enc[start:start + BATCH_SIZE])
            all_pol.append(pol.float().cpu().numpy())
            all_val.append(val.float().cpu().numpy())

    pol_preds  = np.concatenate(all_pol, axis=0)
    val_preds  = np.concatenate(all_val, axis=0).ravel()
    pstack     = bundle["policy_logits"].numpy()
    ystack     = bundle["value_out"].numpy().ravel()
    mstack     = bundle["mask"].numpy()

    pol_stats  = batch_policy_metrics(pol_preds, pstack, mstack)
    value_mse  = float(np.mean((val_preds - ystack) ** 2))
    value_corr = float(np.corrcoef(val_preds, ystack)[0, 1])

    mh = ms["metrics_history"]
    for k, v in pol_stats.items():
        mh.setdefault(k, []).append(v)
    mh["value_mse"].append(value_mse)
    mh["value_corr"].append(value_corr)

    eval_df  = ms["eval_df"]
    new_row  = pd.DataFrame(mh).iloc[[-1], :]
    n_samp   = bundle["enc_in"].shape[0] * (PLOT_EVERY if epoch > 0 else 1)
    new_row["n_samples"] = n_samp
    if eval_df is None:
        new_row["model_epoch"] = 0
        eval_df = new_row.copy()
    else:
        new_row["model_epoch"] = eval_df.tail(1)["model_epoch"].item() + 1
        new_row = new_row.reindex(columns=eval_df.columns)
        eval_df = pd.concat([eval_df, new_row])

    eval_df.round(4).to_csv(ms["progress_file"], index=False)
    ms["eval_df"] = eval_df

    print(f"[eval] {name}")
    print_validation(epoch, {"value_mse": value_mse, "value_corr": value_corr,
                              **pol_stats})

    save_plot(eval_df, f"{name}  [PT]", epoch, ms["plot_file"], ystack, val_preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

model_states = []
for name in MODEL_DEFS:
    src_model_file = os.path.join(MODEL_DIR, f"{name}_pt_model.pt")
    run_model_file = os.path.join(RUN_DIR,   f"{name}_pt_model.pt")
    print(f"[init] {name}: loading fresh from MODEL_DIR")
    model_states.append({
        "name":            name,
        "load_from":       src_model_file,
        "model_file":      run_model_file,
        "progress_file":   os.path.join(RUN_DIR, f"{name}_pt_eval_progress.csv"),
        "plot_file":       os.path.join(RUN_DIR, f"{name}_pt_plot.png"),
        "model":           None,
        "optimizer":       None,
        "scaler":          None,
        "eval_df":         None,
        "metrics_history": {"value_mse": [], "value_corr": []},
        "current_lw":      None,
        "epoch":           0,
    })

all_files = list_tfrecord_files(TFREC_DIR)
if not all_files:
    raise RuntimeError(f"No .tfrecord.gz files in {TFREC_DIR}")

train_files, val_files = split_train_val(all_files)
print(
    f"[tfrec] train={len(train_files)}  val={len(val_files)}"
    f"  shuffle_buffer={SHUFFLE_BUFFER}  val_shuffle_buffer={VAL_SHUFFLE_BUFFER}"
    f"  epoch_size={EPOCH_SIZE}  steps_per_epoch={STEPS_PER_EPOCH}"
    f"  batch_size={BATCH_SIZE}"
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

        worker = EpochBufferThread(
            make_dataset(train_files, SHUFFLE_BUFFER), STEPS_PER_EPOCH, max_ready=2
        )
        val_worker = EpochBufferThread(
            make_dataset(val_files, VAL_SHUFFLE_BUFFER), STEPS_PER_EPOCH, max_ready=1
        )
        worker.start()
        val_worker.start()

        epoch           = 0
        epoch_time_list = []
        begin           = time.time()
        t_fetch = t_eval = t_fit = 0.0
        total_samples = window_samples = 0

        while epoch <= DEFAULT_MAX_EPOCH:
            ep          = ms["epoch"]
            do_plot     = (ep % PLOT_EVERY == 0)
            epoch_start = time.time()

            target_lw = lw_for_epoch(ep)
            if target_lw is not ms["current_lw"]:
                ms["current_lw"] = target_lw
                lw_str = "  ".join(f"{k}={v}" for k, v in target_lw.items())
                print(f"[lw update] {name} epoch {ep}: {lw_str}")

            print("-" * 89)

            if do_plot:
                t0 = time.time()
                val_bundle = val_worker.get()
                do_eval(ms, val_bundle, DEVICE)
                t_eval += time.time() - t0

            t0 = time.time()
            bundle = worker.get()
            t_fetch += time.time() - t0

            t0 = time.time()
            p_loss, v_loss, t_loss = fit_epoch(ms, bundle, target_lw, DEVICE)
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
                ms["current_lw"] = None
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
                print(
                    f"[timing/{PLOT_EVERY}ep] fetch: {t_fetch:.2f}s  "
                    f"fit: {t_fit:.2f}s  eval: {t_eval:.2f}s  "
                    f"samples_window: {window_samples:,}  "
                    f"samples_total: {total_samples:,}"
                )
                t_fetch = t_eval = t_fit = 0.0
                window_samples = 0

            print()

        worker.stop()
        val_worker.stop()
        save_pt_model(ms)

        del ms["model"], ms["optimizer"], ms["scaler"]
        ms["model"] = ms["optimizer"] = ms["scaler"] = None
        gc.collect()
        torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n[bootstrap_pt] KeyboardInterrupt")
