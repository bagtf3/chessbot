import os
import sys
import math
import h5py
import numpy as np


def fmt_num(x):
    if isinstance(x, (int, np.integer)):
        return f"{x}"
    if not np.isfinite(x):
        return str(x)
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 1000 or ax < 0.001:
        return f"{x:.4e}"
    return f"{x:.6f}"


def fmt_shape(shape):
    return "(" + ", ".join([str(v) for v in shape]) + ")"


def safe_read_stats(ds):
    arr = ds[()]
    arr = np.asarray(arr)

    if arr.size == 0:
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "abs_mean": np.nan,
            "l2": np.nan,
            "zero_frac": np.nan,
        }

    if np.issubdtype(arr.dtype, np.number):
        flat = arr.reshape(-1)
        zero_frac = np.mean(np.abs(flat) < 1e-12)
        stats = {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "count": arr.size,
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "abs_mean": float(np.mean(np.abs(flat))),
            "l2": float(np.linalg.norm(flat)),
            "zero_frac": float(zero_frac),
        }
        return stats

    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "count": arr.size,
        "mean": np.nan,
        "std": np.nan,
        "min": np.nan,
        "max": np.nan,
        "abs_mean": np.nan,
        "l2": np.nan,
        "zero_frac": np.nan,
    }


def hist_summary(arr, bins=21):
    arr = np.asarray(arr)
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return []

    flat = arr.reshape(-1)
    mn = float(np.min(flat))
    mx = float(np.max(flat))
    if mn == mx:
        return [f"all values identical: {fmt_num(mn)}"]

    counts, edges = np.histogram(flat, bins=bins)
    lines = []
    peak = max(counts.max(), 1)

    for i in range(len(counts)):
        left = edges[i]
        right = edges[i + 1]
        c = counts[i]
        bar_n = max(1, int(math.ceil(40 * c / peak))) if c > 0 else 0
        bar = "#" * bar_n
        lines.append(
            f"[{fmt_num(left)}, {fmt_num(right)}]  {c:>10}  {bar}"
        )

    return lines


def parent_layer_name(path):
    parts = path.strip("/").split("/")
    if len(parts) <= 1:
        return "<root>"
    return parts[-2]


def tensor_role_name(path):
    parts = path.strip("/").split("/")
    if not parts:
        return path
    return parts[-1]


def is_numeric_dataset(ds):
    try:
        return np.issubdtype(ds.dtype, np.number)
    except Exception:
        return False


def get_channel_norms(arr):
    arr = np.asarray(arr)
    if arr.ndim == 4:
        out_norm = np.linalg.norm(arr, axis=(0, 1, 2))
        in_norm = np.linalg.norm(arr, axis=(0, 1, 3))
        return out_norm, in_norm
    if arr.ndim == 2:
        out_norm = np.linalg.norm(arr, axis=0)
        in_norm = np.linalg.norm(arr, axis=1)
        return out_norm, in_norm
    return None, None


def summarize_norms(vals):
    vals = np.asarray(vals)
    if vals.size == 0:
        return "empty"
    dead = int(np.sum(vals < 1e-8))
    return (
        f"n={vals.size} "
        f"min={fmt_num(vals.min())} "
        f"p10={fmt_num(np.percentile(vals, 10))} "
        f"med={fmt_num(np.median(vals))} "
        f"p90={fmt_num(np.percentile(vals, 90))} "
        f"max={fmt_num(vals.max())} "
        f"dead<{fmt_num(1e-8)}={dead}"
    )


def collect_datasets(h5_file):
    items = []

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            items.append((name, obj))

    h5_file.visititems(walk)
    return items


def write_report(model_path, out_path):
    with h5py.File(model_path, "r") as f:
        datasets = collect_datasets(f)

        total_params = 0
        layer_rows = []
        tensor_rows = []
        layer_map = {}

        for name, ds in datasets:
            stats = safe_read_stats(ds)
            total_params += stats["count"]

            row = {
                "path": name,
                "layer": parent_layer_name(name),
                "role": tensor_role_name(name),
                "shape": stats["shape"],
                "dtype": stats["dtype"],
                "count": stats["count"],
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "abs_mean": stats["abs_mean"],
                "l2": stats["l2"],
                "zero_frac": stats["zero_frac"],
            }
            tensor_rows.append(row)

            layer = row["layer"]
            if layer not in layer_map:
                layer_map[layer] = {
                    "params": 0,
                    "tensors": 0,
                    "l2_sq": 0.0,
                    "std_weighted_sum": 0.0,
                    "count_weighted_sum": 0,
                }

            layer_map[layer]["params"] += row["count"]
            layer_map[layer]["tensors"] += 1

            if np.isfinite(row["l2"]):
                layer_map[layer]["l2_sq"] += row["l2"] ** 2

            if np.isfinite(row["std"]):
                layer_map[layer]["std_weighted_sum"] += row["std"] * row["count"]
                layer_map[layer]["count_weighted_sum"] += row["count"]

        for layer, vals in layer_map.items():
            layer_rows.append(
                {
                    "layer": layer,
                    "params": vals["params"],
                    "tensors": vals["tensors"],
                    "l2": math.sqrt(vals["l2_sq"]),
                    "avg_std": (
                        vals["std_weighted_sum"] / vals["count_weighted_sum"]
                        if vals["count_weighted_sum"] > 0
                        else np.nan
                    ),
                }
            )

        layer_rows = sorted(layer_rows, key=lambda x: x["params"], reverse=True)
        tensor_rows = sorted(tensor_rows, key=lambda x: x["count"], reverse=True)

        lines = []

        lines.append(f"model_path: {model_path}")
        lines.append(f"file_size_bytes: {os.path.getsize(model_path)}")
        lines.append(f"dataset_count: {len(datasets)}")
        lines.append(f"total_params: {total_params}")
        lines.append("")

        lines.append("FULL DATASET LIST")
        lines.append("")
        for row in sorted(tensor_rows, key=lambda x: x["path"]):
            lines.append(
                f"{row['path']}"
                f"  shape={fmt_shape(row['shape'])}"
                f"  dtype={row['dtype']}"
                f"  count={row['count']}"
            )

        lines.append("")
        lines.append("TOP 60 LAYERS BY PARAM COUNT")
        lines.append("")
        for row in layer_rows[:60]:
            lines.append(
                f"{row['layer']}"
                f"  params={row['params']}"
                f"  tensors={row['tensors']}"
                f"  l2={fmt_num(row['l2'])}"
                f"  avg_std={fmt_num(row['avg_std'])}"
            )

        lines.append("")
        lines.append("TOP 120 TENSORS BY PARAM COUNT")
        lines.append("")
        for row in tensor_rows[:120]:
            lines.append(
                f"{row['path']}"
                f"  shape={fmt_shape(row['shape'])}"
                f"  count={row['count']}"
                f"  mean={fmt_num(row['mean'])}"
                f"  std={fmt_num(row['std'])}"
                f"  min={fmt_num(row['min'])}"
                f"  max={fmt_num(row['max'])}"
                f"  abs_mean={fmt_num(row['abs_mean'])}"
                f"  l2={fmt_num(row['l2'])}"
                f"  zero_frac={fmt_num(row['zero_frac'])}"
            )

        lines.append("")
        lines.append("EMBEDDING TENSORS")
        lines.append("")
        for name, ds in datasets:
            low = name.lower()
            if "emb" not in low:
                continue
            stats = safe_read_stats(ds)
            lines.append(
                f"{name}"
                f"  shape={fmt_shape(stats['shape'])}"
                f"  mean={fmt_num(stats['mean'])}"
                f"  std={fmt_num(stats['std'])}"
                f"  min={fmt_num(stats['min'])}"
                f"  max={fmt_num(stats['max'])}"
                f"  l2={fmt_num(stats['l2'])}"
            )

        lines.append("")
        lines.append("ATTENTION TENSORS")
        lines.append("")
        for name, ds in datasets:
            low = name.lower()
            if "mha" not in low and "attention" not in low:
                continue
            stats = safe_read_stats(ds)
            lines.append(
                f"{name}"
                f"  shape={fmt_shape(stats['shape'])}"
                f"  mean={fmt_num(stats['mean'])}"
                f"  std={fmt_num(stats['std'])}"
                f"  min={fmt_num(stats['min'])}"
                f"  max={fmt_num(stats['max'])}"
                f"  l2={fmt_num(stats['l2'])}"
            )

        lines.append("")
        lines.append("CONV AND DENSE CHANNEL NORM SUMMARY")
        lines.append("")
        for name, ds in datasets:
            low = name.lower()
            if not is_numeric_dataset(ds):
                continue
            if "kernel" not in low and "embeddings" not in low:
                continue

            arr = ds[()]
            out_norm, in_norm = get_channel_norms(arr)
            if out_norm is None:
                continue

            lines.append(f"{name}")
            lines.append(f"  out_norms: {summarize_norms(out_norm)}")
            lines.append(f"  in_norms : {summarize_norms(in_norm)}")

        pos_rows = []
        conv_rows = []
        mha_rows = []

        for name, ds in datasets:
            low = name.lower()
            stats = safe_read_stats(ds)
            if "pos" in low and "emb" in low:
                pos_rows.append((name, stats))
            if "c1" in low or "c2" in low or "conv" in low:
                conv_rows.append((name, stats))
            if "mha" in low or "attention" in low:
                mha_rows.append((name, stats))

        lines.append("")
        lines.append("POS VS CONV STD COMPARISON")
        lines.append("")
        if pos_rows and conv_rows:
            conv_std_vals = [
                x[1]["std"] for x in conv_rows if np.isfinite(x[1]["std"])
            ]
            pos_std_vals = [
                x[1]["std"] for x in pos_rows if np.isfinite(x[1]["std"])
            ]
            if conv_std_vals and pos_std_vals:
                conv_std = float(np.mean(conv_std_vals))
                pos_std = float(np.mean(pos_std_vals))
                ratio = pos_std / conv_std if conv_std != 0 else np.nan
                lines.append(f"avg_pos_std : {fmt_num(pos_std)}")
                lines.append(f"avg_conv_std: {fmt_num(conv_std)}")
                lines.append(f"ratio       : {fmt_num(ratio)}")
            else:
                lines.append("could not compute")
        else:
            lines.append("missing pos or conv tensors")

        lines.append("")
        lines.append("SUSPECTLY SMALL OR LARGE STDS")
        lines.append("")
        for row in tensor_rows:
            s = row["std"]
            if not np.isfinite(s):
                continue
            if s < 1e-5 or s > 1.0:
                lines.append(
                    f"{row['path']}"
                    f"  shape={fmt_shape(row['shape'])}"
                    f"  std={fmt_num(s)}"
                    f"  mean={fmt_num(row['mean'])}"
                    f"  l2={fmt_num(row['l2'])}"
                )

        lines.append("")
        lines.append("HISTOGRAMS FOR KEY TENSORS")
        lines.append("")
        hist_keys = []

        for name, ds in datasets:
            low = name.lower()
            if "pos_emb" in low and "emb" in low:
                hist_keys.append(name)
            if "token_emb" in low and "emb" in low:
                hist_keys.append(name)
            if "b0_c1" in low and "kernel" in low:
                hist_keys.append(name)
            if "b0_c2" in low and "kernel" in low:
                hist_keys.append(name)
            if "b0_mha" in low and "kernel" in low:
                hist_keys.append(name)
            if "policy_conv" in low and "kernel" in low:
                hist_keys.append(name)

        hist_keys = sorted(list(set(hist_keys)))
        for name in hist_keys:
            arr = f[name][()]
            lines.append(name)
            for line in hist_summary(arr, bins=21):
                lines.append("  " + line)
            lines.append("")

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def main():
    if len(sys.argv) < 2:
        print("usage: python inspect_h5_weights.py model.h5 [report.txt]")
        return

    model_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        out_path = stem + "_weight_report.txt"

    write_report(model_path, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
