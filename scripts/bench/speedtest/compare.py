import argparse
from pathlib import Path

from . import paths

METRICS = [
    ("sims_per_sec", "sims/s", "{:.0f}", True),
    ("nodes_per_sec", "nodes/s", "{:.0f}", True),
    ("us_per_sim", "us/sim", "{:.2f}", False),
    ("rebuild_us_per_sim", "rebuild us/sim", "{:.2f}", False),
    ("rebuild_sims_per_sec", "rebuild sims/s", "{:.0f}", True),
    ("puct_per_sim", "puct/sim", "{:.2f}", None),
    ("pruned_per_sim", "pruned/sim", "{:.3f}", None),
]


def load_result(run_dir):
    run_dir = Path(run_dir)
    path = run_dir / "result.json" if run_dir.is_dir() else run_dir
    return paths.load_json(path)


def metrics_block(result):
    """Prefer per-sweep stats (carries sd); fall back to old-format runs."""
    if "stats" in result:
        return result["stats"]

    out = {}
    s = result.get("summary", {})
    out.update(s.get("overall", {}))
    for k, v in s.get("rebuild", {}).items():
        out["rebuild_" + k] = v
    return out


def pct(a, b):
    return ((b - a) / a * 100.0) if a else float("nan")


def significance(a, b, sd_a, sd_b, n_a, n_b):
    """Flag whether a delta clears the run-to-run noise of both builds.

    Not a p-value -- a blunt 2-sigma check on the difference of means, which
    is what we actually need to answer 'is this real or is this jitter'.
    n must come from the two runs being compared: borrowing it from another
    column shrinks the standard error and invents significance.
    """
    if not sd_a and not sd_b:
        return ""
    if n_a < 2 or n_b < 2:
        return ""
    se = ((sd_a ** 2) / n_a + (sd_b ** 2) / n_b) ** 0.5
    if se <= 0:
        return ""
    return "yes" if abs(b - a) >= 2.0 * se else "no"


def compare(dirs, labels=None):
    results = [load_result(d) for d in dirs]
    blocks = [metrics_block(r) for r in results]
    if labels is None:
        labels = [Path(d).name[:22] for d in dirs]

    rows = []
    for key, name, fmt, higher_better in METRICS:
        if not any(key in b for b in blocks):
            continue
        vals = [b.get(key) for b in blocks]
        if any(v is None for v in vals):
            continue

        sds = [b.get(key + "_sd", 0.0) for b in blocks]
        sweeps = [r.get("sweeps", 1) for r in results]

        rows.append({
            "name": name, "fmt": fmt, "vals": vals, "sds": sds,
            "higher_better": higher_better,
            "vs_first": pct(vals[0], vals[-1]),
            "vs_prev": pct(vals[-2], vals[-1]) if len(vals) > 1 else None,
            "sig": significance(vals[-2], vals[-1], sds[-2], sds[-1],
                                sweeps[-2], sweeps[-1])
            if len(vals) > 1 else "",
        })
    return rows, labels


def print_table(rows, labels):
    w = 15
    head = f"{'metric':<16}" + "".join(f"{l[:w]:>{w}}" for l in labels)
    if len(labels) > 1:
        head += f"{'vs first':>10}{'vs prev':>9}{'real?':>7}"
    print(head)
    print("-" * len(head))

    for r in rows:
        line = f"{r['name']:<16}"
        for v, sd in zip(r["vals"], r["sds"]):
            cell = r["fmt"].format(v)
            if sd:
                cell += "+/-" + r["fmt"].format(sd)
            line += f"{cell:>{w}}"
        if len(labels) > 1:
            hb = r["higher_better"]
            mark = " "
            if hb is not None and r["vs_prev"] == r["vs_prev"]:
                if abs(r["vs_prev"]) >= 0.05:
                    good = r["vs_prev"] > 0 if hb else r["vs_prev"] < 0
                    mark = "+" if good else "-"
            line += (f"{r['vs_first']:>9.1f}%"
                     f"{r['vs_prev']:>8.1f}%{mark}{r['sig']:>6}")
        print(line)

    print("\n'real?' = delta vs prev clears 2 sigma of both builds' sweep "
          "spread.\npuct/sim and pruned/sim should hold flat unless search "
          "behaviour changed.")


def main():
    ap = argparse.ArgumentParser(
        description="Compare bench runs. Pass 3 dirs for OG / prev / current.")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--labels", default=None,
                    help="comma-separated column labels")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else None
    rows, labels = compare(args.dirs, labels)
    print_table(rows, labels)


if __name__ == "__main__":
    main()
