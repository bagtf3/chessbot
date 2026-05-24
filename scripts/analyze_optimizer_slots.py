"""
Analyze Adam m/v optimizer slots from a TF checkpoint or Keras .h5 file.
Reports dead/frozen weights, gradient activity, and m/v coherence by layer depth.

m/v ratio = |m| / sqrt(v + eps) per element — bounded ~[0,1]:
  ~1.0  gradients are consistent in sign (coherent learning signal)
  ~0.0  gradients are oscillating/cancelling (noise or non-stationarity)

TF checkpoint layer map (conformer-interweaved 16m, val_test_multi):
  0: embedding
  1..50: backbone — 10 super-blocks each [norm, attn, norm, conv, conv]
    attn at: 2,7,12,17,22,27,32,37,42,47
    conv at: 4,5,9,10,14,15,19,20,24,25,29,30,34,35,39,40,44,45,49,50
    norm at: 1,3,6,8,11,13,16,18,21,23,26,28,31,33,36,38,41,43,46,48
  51+: projection / heads

H5 layer map (val_test, semantic names):
  token_emb, b0..b9 (each: mha, c1, c2, ln1, ln2), then policy/value heads
"""
import argparse
import re
import numpy as np
import tensorflow as tf
import h5py

EPS = 1e-7   # matches Adam default epsilon

# coherence thresholds for |m|/sqrt(v) ratio
RATIO_HI  = 0.5   # "coherent"  — gradient direction is stable
RATIO_LO  = 0.1   # "incoherent" — gradient is mostly noise / sign-flipping

# --- TF checkpoint layer mapping ---

ATTN_IDX  = {2,7,12,17,22,27,32,37,42,47}
CONV_IDX  = {4,5,9,10,14,15,19,20,24,25,29,30,34,35,39,40,44,45,49,50}
NORM_IDX  = {1,3,6,8,11,13,16,18,21,23,26,28,31,33,36,38,41,43,46,48}
ATTN_BLOCK = {a: i for i, a in enumerate(sorted(ATTN_IDX))}
CONV_BLOCK = {c: i//2 for i, c in enumerate(sorted(CONV_IDX))}


def ckpt_sem_group(name):
    m = re.search(r"layer_with_weights-(\d+)", name)
    if not m:
        return "other"
    idx = int(m.group(1))
    if idx == 0:
        return "embed"
    if idx in ATTN_IDX:
        return f"blk{ATTN_BLOCK[idx]:02d}_attn"
    if idx in CONV_IDX:
        return f"blk{CONV_BLOCK[idx]:02d}_conv"
    if idx in NORM_IDX:
        return None
    if idx == 51:
        return "head_proj1x1_256"
    if idx == 53:
        return "head_squeeze_64"
    if idx == 61:
        return "head_policy_conv"
    if idx == 56:
        return "head_value_out"
    if idx in {57, 58, 59, 60}:
        return "head_policy_ff"
    if idx in {62, 63, 66, 67, 68}:
        return "head_value_ff"
    if idx == 65:
        return "head_policy_out"
    return f"head_other_{idx}"


def h5_sem_group(name):
    prefix = name.split("/")[0]
    m = re.match(r"b(\d+)_(mha|c1|c2|ln1|ln2)", prefix)
    if m:
        blk, kind = int(m.group(1)), m.group(2)
        if kind == "mha":
            return f"blk{blk:02d}_attn"
        if kind in ("c1", "c2"):
            return f"blk{blk:02d}_conv"
        return None
    if prefix == "token_emb":
        return "embed"
    if prefix == "attn_w":
        return "embed"
    if prefix.startswith("pol_"):
        return "head_policy"
    if prefix.startswith("v_") or prefix.startswith("value"):
        return "head_value"
    return "head_other"


# --- shared ---

def depth_key(g):
    m = re.match(r"blk(\d+)_(attn|conv)", g)
    if m:
        return (int(m.group(1)), 0 if m.group(2) == "attn" else 1)
    return (99, g)


def ratio_arr(m_arr, v_arr):
    return np.abs(m_arr) / (np.sqrt(np.abs(v_arr)) + EPS)


def accumulate(groups, grp, v, m, dead_thresh, frozen_thresh):
    n    = v.size
    dead   = int(np.sum(v < dead_thresh))
    frozen = int(np.sum(v < frozen_thresh))
    r      = ratio_arr(m, v)
    if grp not in groups:
        groups[grp] = {
            "n": 0, "dead": 0, "frozen": 0,
            "v_sum": 0.0, "v_max": 0.0, "m_sum": 0.0,
            # ratio buckets
            "r_sum": 0.0,
            "r_hi": 0,    # count with ratio >= RATIO_HI
            "r_lo": 0,    # count with ratio <  RATIO_LO
            "r_vals": [],  # keep flat arrays for percentile later
        }
    g = groups[grp]
    g["n"]      += n
    g["dead"]   += dead
    g["frozen"] += frozen
    g["v_sum"]  += float(np.mean(v)) * n
    g["v_max"]   = max(g["v_max"], float(np.max(v)))
    g["m_sum"]  += float(np.mean(np.abs(m))) * n
    g["r_sum"]  += float(np.mean(r)) * n
    g["r_hi"]   += int(np.sum(r >= RATIO_HI))
    g["r_lo"]   += int(np.sum(r <  RATIO_LO))
    g["r_vals"].append(r.ravel())
    return n, dead, frozen


def print_table(groups, total_n, total_dead, total_frozen, dead_thresh, frozen_thresh):
    # main table
    hdr = (f"{'Group':<26} {'Params':>10} {'Dead%':>7} {'Frz%':>7}"
           f" {'v_mean':>10} {'|m|_mean':>10}"
           f" {'r_mean':>8} {'r>=.5%':>8} {'r<.1%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for grp in sorted(groups, key=depth_key):
        g = groups[grp]
        dp  = 100 * g["dead"]   / g["n"]
        fp  = 100 * g["frozen"] / g["n"]
        vm  = g["v_sum"] / g["n"]
        mm  = g["m_sum"] / g["n"]
        rm  = g["r_sum"] / g["n"]
        rhi = 100 * g["r_hi"] / g["n"]
        rlo = 100 * g["r_lo"] / g["n"]
        print(f"{grp:<26} {g['n']:>10,} {dp:>6.1f}% {fp:>6.1f}%"
              f" {vm:>10.2e} {mm:>10.2e}"
              f" {rm:>8.3f} {rhi:>7.1f}% {rlo:>7.1f}%")
    print("-" * len(hdr))
    dp = 100 * total_dead   / total_n
    fp = 100 * total_frozen / total_n
    print(f"{'TOTAL':<26} {total_n:>10,} {dp:>6.1f}% {fp:>6.1f}%")
    print()
    print(f"dead_thresh={dead_thresh:.0e}  frozen_thresh={frozen_thresh:.0e}"
          f"  coherent=r>={RATIO_HI}  incoherent=r<{RATIO_LO}")

    # ratio percentile breakdown per group
    print()
    print("=== m/v ratio percentiles (p10 / p50 / p90) ===")
    phdr = f"{'Group':<26} {'p10':>8} {'p50':>8} {'p90':>8}  interpretation"
    print(phdr)
    print("-" * len(phdr))
    for grp in sorted(groups, key=depth_key):
        g = groups[grp]
        all_r = np.concatenate(g["r_vals"])
        p10, p50, p90 = np.percentile(all_r, [10, 50, 90])
        if p50 >= RATIO_HI:
            note = "coherent"
        elif p50 >= RATIO_LO:
            note = "mixed"
        else:
            note = "incoherent / sign-flipping"
        print(f"{grp:<26} {p10:>8.3f} {p50:>8.3f} {p90:>8.3f}  {note}")


def _print_key_bias(rows):
    if not rows:
        return
    print()
    print("=== Attention key/bias (structural dead weight) ===")
    for blk, vm, dead_pct, mabs, ratio_mean in sorted(rows):
        print(f"  blk{blk:02d}  v_mean={vm:.2e}  dead={dead_pct:.0f}%"
              f"  |m|={mabs:.2e}  r={ratio_mean:.3f}")


# --- TF checkpoint ---

def analyze_ckpt(path, dead_thresh, frozen_thresh):
    reader = tf.train.load_checkpoint(path)
    shapes = reader.get_variable_to_shape_map()
    m_slots, v_slots = {}, {}
    for name in shapes:
        if "OPTIMIZER_SLOT" not in name:
            continue
        tensor = reader.get_tensor(name).astype(np.float32)
        key = re.sub(r"/\.OPTIMIZER_SLOT/.*", "", name)
        if re.search(r"/m/\.ATTRIBUTES", name):
            m_slots[key] = tensor
        elif re.search(r"/v/\.ATTRIBUTES", name):
            v_slots[key] = tensor

    common = sorted(set(m_slots) & set(v_slots))
    groups = {}
    total_n = total_dead = total_frozen = 0
    key_bias_rows = []

    for key in common:
        grp = ckpt_sem_group(key)
        if grp is None:
            continue
        v, m = v_slots[key], m_slots[key]
        n, d, f = accumulate(groups, grp, v, m, dead_thresh, frozen_thresh)
        total_n += n; total_dead += d; total_frozen += f
        if "_key_dense/bias" in key or "/key/bias" in key:
            idx_m = re.search(r"weights-(\d+)", key)
            idx = int(idx_m.group(1)) if idx_m else -1
            blk = ATTN_BLOCK.get(idx, "?")
            r = ratio_arr(m, v)
            key_bias_rows.append((blk, float(np.mean(v)),
                                  100*float(np.mean(v < dead_thresh)),
                                  float(np.mean(np.abs(m))),
                                  float(np.mean(r))))

    print_table(groups, total_n, total_dead, total_frozen, dead_thresh, frozen_thresh)
    _print_key_bias(key_bias_rows)


# --- H5 ---

def analyze_h5(path, dead_thresh, frozen_thresh):
    f = h5py.File(path, "r")
    opt = f["optimizer_weights"]
    all_paths = []
    opt.visititems(lambda n, o: all_paths.append(n) if isinstance(o, h5py.Dataset) else None)

    m_slots, v_slots = {}, {}
    for p in all_paths:
        p_clean = p[:-2] if p.endswith(":0") else p
        rel = re.sub(r"^cond_\d+/Adam/", "", p_clean)
        if rel.endswith("/m"):
            m_slots[rel[:-2]] = opt[p][()].astype(np.float32)
        elif rel.endswith("/v"):
            v_slots[rel[:-2]] = opt[p][()].astype(np.float32)

    common = sorted(set(m_slots) & set(v_slots))
    groups = {}
    total_n = total_dead = total_frozen = 0
    key_bias_rows = []

    for key in common:
        grp = h5_sem_group(key)
        if grp is None:
            continue
        v, m = v_slots[key], m_slots[key]
        n, d, fz = accumulate(groups, grp, v, m, dead_thresh, frozen_thresh)
        total_n += n; total_dead += d; total_frozen += fz
        if re.search(r"mha/key/bias$", key):
            bm = re.match(r"b(\d+)_mha", key)
            blk = int(bm.group(1)) if bm else "?"
            r = ratio_arr(m, v)
            key_bias_rows.append((blk, float(np.mean(v)),
                                  100*float(np.mean(v < dead_thresh)),
                                  float(np.mean(np.abs(m))),
                                  float(np.mean(r))))

    print_table(groups, total_n, total_dead, total_frozen, dead_thresh, frozen_thresh)
    _print_key_bias(key_bias_rows)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="TF checkpoint prefix or .h5 file")
    parser.add_argument("--dead",   type=float, default=1e-10)
    parser.add_argument("--frozen", type=float, default=1e-8)
    args = parser.parse_args()
    if args.path.endswith(".h5"):
        analyze_h5(args.path, args.dead, args.frozen)
    else:
        analyze_ckpt(args.path, args.dead, args.frozen)
