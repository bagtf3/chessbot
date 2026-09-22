#!/usr/bin/env python3
"""bakeoff_speedtest_10c6t.py

Speed test for the bake-off block variants scaled to a 10c6t trunk, random
weights.  PT eager (fp16) vs ORT+TensorRT (fp16) across batch sizes, reusing the
harness in bakeoff_speedtest.py.

Layers match bakeoff_4c2t_d256 (plain ConvBlock, ExpandyConvBlock, gated
expandy, FF 4x, SwiGLU 4x); gated-expandy and SwiGLU use lean copies without
the training-only EMA buffers / product hooks.

    m1     10 plain conv                    6x MHA + FF 4x
    m7     10 plain conv                    6x MHA + SwiGLU 4x
    m11    10 expandy conv                  6x MHA + FF 4x
    m13    4 plain + 6 gated-expandy        6x MHA + FF 4x
    m11-7  10 expandy conv                  6x MHA + SwiGLU 4x
    m13-7  4 plain + 6 gated-expandy        3x FF 4x, then 3x SwiGLU 4x
    m13-1-7-5c10t        5 gated-expandy    5x FF 4x, then 5x SwiGLU 4x
    m11-1-6c12t-d256     6 expandy          12x FF 4x
    m13-1-7-6c12t-d256   6 gated-expandy    8x FF 4x, then 4x SwiGLU 4x
    m1-13-1-7-8c8t-d256  4 plain + 4 gated  4x FF 4x, then 4x SwiGLU 4x
    m13-7-8c8t           model.py m13-7-8c8t-d256 (wdl3, GELU-MLP smartgate)
    m13-7-8c8t-L1point5  m13-7-8c8t, L1point5Norm everywhere, unclamped smartgate
    m11-1-8c10t          8 expandy          10x FF 4x
    m11-1-7-8c8t         8 expandy          4x FF 4x, then 4x SwiGLU 4x
    m11-13-1-7-6c10t-d256  2 expandy + 4 gated  6x FF 4x, then 4x SwiGLU 4x  (2x m11-13-1-7-3c5t)

Results also go to a CSV in --out-dir.
"""
import os
import sys
import csv
import gc
import glob
import hashlib
import time
import argparse
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from bakeoff_speedtest import (
    TRT_CACHE, DEFAULT_OUT_DIR, verify_outputs, make_pt_eager_infer,
    export_to_onnx, make_trt_infer, compare_pt_vs_export, speed_test,
    print_summary_table,
)
from bakeoff_4c2t_d256 import (
    BakeoffModel, L1Norm, Attention, ConvBlock, ExpandyConvBlock, TxBlockFF,
    D, HEADS, EPS, TOKENS, PDH, FP16_CLIP,
)

from chessbot.model import VARIANTS, build_pt_expandy_swiglu, xc0h_stem_forward

N_PLAIN = 4
N_VARIANT = 6
N_TX = 6
SWIGLU_H = 4 * D

MODELS_TO_TEST = ["m1", "m7", "m11", "m13"]
DEFAULT_BATCH_SIZES = [32, 64, 128, 256]

MASTER_CSV = "speedtest_10c6t_master.csv"
CSV_FIELDS = ["model", "params", "backend", "batch", "latency_ms",
              "samples_per_s", "run"]
SYNC_TIMEOUT = 360
YES_WORDS = {"y", "yes", "ye", "yeah", "yep", "yup", "sure", "ok", "okay", "do it"}
# param counts for run CSVs written before the params column existed
PARAMS_FALLBACK = {
    "m1": 19980155, "m7": 21559163, "m11": 16716155, "m11-7": 19600763,
}


class RMSNormExport(nn.Module):
    """nn.RMSNorm equivalent written in exportable ops (aten::rms_norm has no
    ONNX symbolic)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(x.dtype)


class L1point5Norm(L1Norm):
    """x / mean(|x|^1.5)^(2/3); same clamp, scale and bias as L1Norm."""

    def forward(self, x):
        x = x.clamp(-250.0, 250.0)
        a = x.abs()
        y = x / ((a * a.sqrt()).mean(-1, keepdim=True).pow(2.0 / 3.0) + EPS)
        return y * self.scale.to(x.dtype) + self.b.to(x.dtype)


def swap_l1_norms(module):
    """Replace every L1Norm under module with an L1point5Norm of the same dim."""
    for name, child in module.named_children():
        if type(child).__name__ == "L1Norm":
            setattr(module, name, L1point5Norm(child.scale.numel()))
        else:
            swap_l1_norms(child)


class GatedExpandyConvBlockLean(nn.Module):
    """GatedExpandyConvBlock (m13) without EMA buffers."""

    def __init__(self, groups=4):
        super().__init__()
        self.n1 = L1Norm(D)
        self.c1 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p1 = nn.Conv2d(2 * D, D, 1, bias=False)
        self.c_gate = nn.Conv2d(D, D, 1)
        self.c2 = nn.Conv2d(D, 2 * D, 3, padding=1, groups=groups)
        self.p2 = nn.Conv2d(2 * D, D, 1, bias=False)

    def forward(self, x):
        h = self.n1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.p1(F.leaky_relu(self.c1(h), 0.05))
        gate = torch.sigmoid(self.c_gate(h))
        h = self.p2(F.leaky_relu(self.c2(h), 0.05))
        return x + gate * h


class TxBlockSwiGLULean(nn.Module):
    """TxBlockSwiGLU (m7) without the last_product hook."""

    def __init__(self, hidden=SWIGLU_H):
        super().__init__()
        self.n1 = L1Norm(D)
        self.attn = Attention(D, HEADS)
        self.n2 = L1Norm(D)
        self.gate = nn.Linear(D, hidden)
        self.up = nn.Linear(D, hidden)
        self.down = nn.Linear(hidden, D, bias=False)

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n)
        n = self.n2(x)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))


MODEL_SPECS = {
    "m1": dict(name="m1_10c6t_conv-van_tx-van", conv="vanilla", tx="vanilla"),
    "m7": dict(name="m7_10c6t_conv-van_tx-swiglu4x", conv="vanilla", tx="swiglu"),
    "m11": dict(
        name="m11_10c6t_conv-expandywide10_tx-van", conv="expandy_all", tx="vanilla"
    ),
    "m13": dict(
        name="m13_10c6t_conv-gated-expandy_tx-van", conv="gated_expandy", tx="vanilla"
    ),
    "m11-7": dict(
        name="m11-7_10c6t_conv-expandy_tx-swiglu4x", conv="expandy_all", tx="swiglu"
    ),
    "m13-7": dict(
        name="m13-7_10c6t_conv-gated-expandy_tx-3ff3swiglu4x",
        conv="gated_expandy", tx="ff3_swiglu3"
    ),
    "m13-1-7-5c10t": dict(
        name="m13-1-7-5c10t_conv-5gated-expandy_tx-5ff5swiglu4x",
        conv="gated_expandy_5", tx="ff5_swiglu5"
    ),
    "m11-1-6c12t-d256": dict(
        name="m11-1-6c12t-d256_conv-6expandy_tx-12ff4x",
        conv="expandy_6", tx="vanilla_12"
    ),
    "m13-1-7-6c12t-d256": dict(
        name="m13-1-7-6c12t-d256_conv-6gated-expandy_tx-8ff4swiglu4x",
        conv="gated_expandy_6", tx="ff8_swiglu4"
    ),
    "m1-13-1-7-8c8t-d256": dict(
        name="m1-13-1-7-8c8t-d256_conv-4plain4gated-expandy_tx-4ff4swiglu4x",
        conv="gated_expandy_4", tx="ff4_swiglu4"
    ),
    "m13-7-8c8t": dict(
        name="m13-7-8c8t-d256_modelpy", build="m137"
    ),
    "m13-7-8c8t-L1point5": dict(
        name="m13-7-8c8t-L1point5_modelpy", build="m137_l1p5"
    ),
    "m11-1-8c10t": dict(
        name="m11-1-8c10t_conv-8expandy_tx-10ff4x",
        conv="expandy_8", tx="vanilla_10"
    ),
    "m11-1-7-8c8t": dict(
        name="m11-1-7-8c8t_conv-8expandy_tx-4ff4swiglu4x",
        conv="expandy_8", tx="ff4_swiglu4"
    ),
    "m11-13-1-7-6c10t-d256": dict(
        name="m11-13-1-7-6c10t-d256_conv-2expandy4gated-expandy_tx-6ff4swiglu4x",
        conv="expandy_2_gated_expandy_4", tx="ff6_swiglu4"
    ),
}

# conv_mode -> (n plain, n expandy, n gated expandy), applied in that order
CONV_COUNTS = {
    "vanilla": (N_PLAIN + N_VARIANT, 0, 0),
    "expandy_all": (0, N_PLAIN + N_VARIANT, 0),
    "expandy_6": (0, 6, 0),
    "expandy_8": (0, 8, 0),
    "expandy_2_gated_expandy_4": (0, 2, 4),
    "gated_expandy": (N_PLAIN, 0, N_VARIANT),
    "gated_expandy_4": (N_PLAIN, 0, 4),
    "gated_expandy_5": (0, 0, 5),
    "gated_expandy_6": (0, 0, 6),
    "gated_expandy_2plain_6": (2, 0, 6),
}

# tx_mode -> (n plain FF, n SwiGLU), FF blocks first
TX_COUNTS = {
    "vanilla": (N_TX, 0),
    "vanilla_10": (10, 0),
    "vanilla_12": (12, 0),
    "swiglu": (0, N_TX),
    "ff3_swiglu3": (3, 3),
    "ff4_swiglu4": (4, 4),
    "ff5_swiglu5": (5, 5),
    "ff6_swiglu4": (6, 4),
    "ff8_swiglu4": (8, 4),
}


class Bakeoff10c6t(BakeoffModel):
    """BakeoffModel stem/heads with the trunk swapped for the 10c6t-scale conv and
    tx stacks named by conv_mode / tx_mode."""

    def __init__(self, conv_mode, tx_mode):
        super().__init__(conv_mode="vanilla", tx_mode="vanilla")
        self.conv_mode = conv_mode
        self.tx_mode = tx_mode
        self.wdl_norm = RMSNormExport(D)
        if conv_mode not in CONV_COUNTS:
            raise ValueError(f"conv_mode={conv_mode}")
        if tx_mode not in TX_COUNTS:
            raise ValueError(f"tx_mode={tx_mode}")
        n_plain, n_exp, n_gexp = CONV_COUNTS[conv_mode]
        self.convs = nn.ModuleList([ConvBlock() for _ in range(n_plain)])
        self.expandy_convs = nn.ModuleList([ExpandyConvBlock() for _ in range(n_exp)])
        self.gated_expandy_convs = nn.ModuleList(
            [GatedExpandyConvBlockLean() for _ in range(n_gexp)])
        n_ff, n_sw = TX_COUNTS[tx_mode]
        self.blocks = nn.ModuleList(
            [TxBlockFF() for _ in range(n_ff)]
            + [TxBlockSwiGLULean() for _ in range(n_sw)])


M137_CFG = VARIANTS["m13-7-8c8t-d256"]


def build_m137():
    return build_pt_expandy_swiglu(M137_CFG)


def build_m137_l1p5():
    """model.py m13-7-8c8t-d256 with L1point5Norm everywhere and the smartgate
    soft clamps and +/-250 clips removed."""
    cf, k = M137_CFG["conv_filters"], M137_CFG["xc0h_K"]

    class M137L1p5(type(build_m137())):
        def forward(self, x_in, range_scan_active=True):
            B = x_in.shape[0]
            x = xc0h_stem_forward(self, x_in, cf, k)
            for blk in self.convs:
                x = blk(x)
            board = x.permute(0, 2, 3, 1).reshape(B, TOKENS, cf)
            x = board + self.pos_scale * self.pos.unsqueeze(0)
            x = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)
            for blk in self.blocks:
                x = blk(x, range_scan_active)
            x = self.trunk(x)
            w = self.wdl_ow(F.gelu(self.wdl_hw(x[:, 64, :])))
            d = self.wdl_od(F.gelu(self.wdl_hd(x[:, 65, :])))
            l = self.wdl_ol(F.gelu(self.wdl_hl(x[:, 66, :])))
            wdl = torch.cat([w, d, l], dim=-1)

            from_set = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 70:71, :]], dim=1)
            to_set = torch.cat([x[:, :64, :], x[:, 68:70, :], x[:, 71:72, :]], dim=1)
            f_proj = from_set + F.gelu(self.from_proj(from_set))
            fn = self.from_norm(f_proj)
            f_base = f_proj[:, :64, :] + self.from_mha(fn[:, :64, :], fn)
            fv = f_base + self.from_out(f_base)
            t_proj = to_set + F.gelu(self.to_proj(to_set))
            tn = self.to_norm(t_proj)
            t_base = t_proj[:, :64, :] + self.to_mha(tn[:, :64, :], tn)
            tv = t_base + self.to_out(t_base)
            dots_raw = torch.bmm(
                self.from_dot_norm(fv), self.to_dot_norm(tv).transpose(1, 2))
            dots_raw = dots_raw.clamp(-FP16_CLIP, FP16_CLIP)
            self.last_dot_raw_pen = self.dot_raw_banger(dots_raw, range_scan_active)
            dots_full = dots_raw * (PDH ** -0.5)
            dots = dots_full.reshape(B, 64 * 64)
            dots_sub = dots_full[:, 48:56, 56:64]
            pf = self.promo_from(fv[:, 48:56, :])
            pt = self.promo_to(tv[:, 56:64, :])
            promo = (dots_sub[..., None] + pf[:, :, None, :] + pt[:, None, :, :]
                     ).permute(0, 3, 2, 1).reshape(B, 192)
            policy = torch.cat([dots, promo], dim=-1)[:, self.sl_idx]

            gi = torch.cat([x[:, 67, :], x[:, 68, :]], dim=-1)
            down = self.gate_ff2(F.gelu(self.gate_ff1(gi)))
            g = self.gate_norm(gi + down)
            gate_act = F.logsigmoid(self.gate_out(g))
            self.last_gate_act_pen = self.gate_act_banger(gate_act, range_scan_active)
            return policy + gate_act, wdl

    m = M137L1p5()
    swap_l1_norms(m)
    return m


MODEL_BUILDERS = {"m137": build_m137, "m137_l1p5": build_m137_l1p5}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="dir the results CSV is written to")
    p.add_argument("--models", nargs="+", default=MODELS_TO_TEST, metavar="ID")
    p.add_argument("--batch-sizes", nargs="+", type=int,
                   default=DEFAULT_BATCH_SIZES, metavar="B")
    p.add_argument("--skip-trt", action="store_true")
    p.add_argument("--fresh", action="store_true",
                   help="before each TRT run, remove that model's cached ONNX "
                        "export and compiled TensorRT engine")
    p.add_argument("--sync", action="store_true",
                   help="after the run, offer to write this run's rows into "
                        "the master CSV (overwrites those models)")
    p.add_argument("--build-master", action="store_true",
                   help="rebuild the master CSV from every speedtest_10c6t_*.csv "
                        "in --out-dir (latest run per model wins) and exit")
    return p.parse_args()


def write_csv(all_results, param_counts, batch_sizes, path, run):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_FIELDS)
        for label, res in all_results.items():
            model, backend = label.split("  ", 1)
            for b in batch_sizes:
                r = res.get(b)
                if r is not None:
                    w.writerow([model, param_counts[model], backend, b,
                                r["latency_ms"], r["throughput"], run])
    print(f"\nresults -> {path}")


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def run_stamp(path):
    return os.path.basename(path)[len("speedtest_10c6t_"):-len(".csv")]


def sync_master(run_csv, master_csv):
    """Replace every model present in run_csv with run_csv's rows in master_csv."""
    new = read_rows(run_csv)
    models = {r["model"] for r in new}
    old = read_rows(master_csv) if os.path.exists(master_csv) else []
    kept = [r for r in old if r["model"] not in models]
    write_rows(kept + new, master_csv)
    print(f"master <- {sorted(models)}  ({master_csv})")


def build_master(out_dir, master_csv):
    """Compile every run CSV in out_dir; per model, the newest run's rows win.
    Runs that predate the params column get it from the known counts."""
    paths = sorted(glob.glob(os.path.join(out_dir, "speedtest_10c6t_*.csv")))
    latest = {}
    for p in paths:
        stamp = run_stamp(p)
        for r in read_rows(p):
            r.setdefault("params", PARAMS_FALLBACK.get(r["model"], ""))
            r.setdefault("run", stamp)
            latest.setdefault(r["model"], {})
            if stamp >= latest[r["model"]].get("run", ""):
                if latest[r["model"]].get("run") != stamp:
                    latest[r["model"]] = {"run": stamp, "rows": []}
                latest[r["model"]]["rows"].append(r)
    rows = [r for m in sorted(latest) for r in latest[m]["rows"]]
    write_rows(rows, master_csv)
    for m in sorted(latest):
        print(f"  {m:<8} run {latest[m]['run']}  rows {len(latest[m]['rows'])}")
    print(f"master -> {master_csv}")


def timed_input(prompt, timeout):
    """input() that returns None if nothing is entered within timeout seconds."""
    box = []
    t = threading.Thread(target=lambda: box.append(input(prompt)), daemon=True)
    t.start()
    t.join(timeout)
    return box[0] if box else None


def wants_sync(answer):
    return answer is not None and answer.strip().lower() in YES_WORDS


def clear_model_trt_cache(onnx_path):
    """Remove an ONNX export and the engine derived from its current contents."""
    if not os.path.exists(onnx_path):
        return

    digest = hashlib.sha256()
    with open(onnx_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    prefix = f"bake_{digest.hexdigest()[:12]}"
    for engine_path in glob.glob(os.path.join(TRT_CACHE, f"{prefix}_*.engine")):
        os.remove(engine_path)
        print(f"  [fresh] removed engine: {engine_path}", flush=True)

    os.remove(onnx_path)
    print(f"  [fresh] removed ONNX:   {onnx_path}", flush=True)


def main():
    args = parse_args()
    master_csv = os.path.join(args.out_dir, MASTER_CSV)
    if args.build_master:
        build_master(args.out_dir, master_csv)
        return

    bs = args.batch_sizes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(TRT_CACHE, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"speedtest_10c6t_{stamp}.csv")

    print(f"PyTorch device: {device}", flush=True)
    print(f"TRT cache:      {TRT_CACHE}", flush=True)

    all_results = {}
    param_counts = {}
    for mid in args.models:
        spec = MODEL_SPECS[mid]
        print(f"\n{'='*60}", flush=True)
        if "build" in spec:
            model = MODEL_BUILDERS[spec["build"]]()
        else:
            model = Bakeoff10c6t(spec["conv"], spec["tx"])
        params = sum(p.numel() for p in model.parameters())
        param_counts[mid] = params
        print(f"  {mid}: {spec['name']}  params {params:,}", flush=True)

        ok, pt_p, pt_v, enc_np = verify_outputs(model, device, mid)
        if not ok:
            print(f"  [SKIP] {mid}: output verification failed", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        print(f"\n  Building {mid} PT eager ...", flush=True)
        eager = make_pt_eager_infer(model, device)
        lbl = f"{mid}  PT eager"
        all_results[lbl] = speed_test(lbl + " [fp16]", eager, bs)
        del eager

        if not args.skip_trt:
            lbl = f"{mid}  ORT TRT"
            onnx_path = os.path.join(TRT_CACHE, f"{spec['name']}_{params}.onnx")
            if args.fresh:
                clear_model_trt_cache(onnx_path)
            if not os.path.exists(onnx_path):
                export_to_onnx(model, device, onnx_path)
            else:
                print(f"  [TRT] ONNX cached: {onnx_path}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            trt_fn, trt_sess = make_trt_infer(onnx_path, TRT_CACHE, max_bs=max(bs))
            compare_pt_vs_export(pt_p, pt_v, trt_fn, enc_np, mid)
            all_results[lbl] = speed_test(lbl + " [fp16]", trt_fn, bs)
            del trt_sess
        else:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        write_csv(all_results, param_counts, bs, csv_path, stamp)

    print_summary_table(all_results, bs)
    print("\nDone.", flush=True)

    if args.sync and all_results:
        ans = timed_input(
            f"\nDo you want to sync CSV results into {MASTER_CSV}? "
            f"[y/N, {SYNC_TIMEOUT}s] ", SYNC_TIMEOUT)
        if wants_sync(ans):
            sync_master(csv_path, master_csv)
        else:
            print("not synced" if ans is not None else "\nno input, not synced")


if __name__ == "__main__":
    main()
