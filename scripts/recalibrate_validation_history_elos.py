"""
Recalibrate historical validation ELO fields after a Stockfish depth-table update.

Reads SF_TABLE_DEFAULT from validation.py (already updated in-repo), then rewrites
each target run's validation_history.jsonl so that sf_elo and model_elo match the
new table. Originals are preserved in sf_elo_original / model_elo_original.

Usage:
    python scripts/recalibrate_validation_history_elos.py            # dry-run all tags
    python scripts/recalibrate_validation_history_elos.py --apply    # write changes
    python scripts/recalibrate_validation_history_elos.py --tags precond_run1 precond_run2
    python scripts/recalibrate_validation_history_elos.py --test     # run tests
"""

import argparse
import ast
import json
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime

TARGET_TAGS = [
    "16m_precond_run0",
    "precond_run1",
    "precond_run2",
    "precond_run3",
    "precond_run4",
    "precond_run5",
    "cfiw_pretrained_run1",
    "cfiw_pretrained_run2",
    "cfiw_pretrained_run3",
    "cfiw_pretrained_run4",
    "cfiw_pretrained_run5",
    "cfiw_pretrained_run6",
    "cfiw_pretrained_run7",
    "cfiw_wdl_run1",
    "cfiw_wdl_run2",
]

HISTORY_FILENAME = "validation_history.jsonl"
DEFAULT_SELFPLAY_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs"


def load_sf_depth_table(repo_root):
    """
    Parse SF_TABLE_DEFAULT from validation.py without importing the module.
    Returns (depth_map, raw_table) where depth_map is {depth_int: elo_int}.
    First occurrence wins for duplicate depths (e.g. the depth-5 duplicate).
    Fails loudly if the table is missing or unparseable.
    """
    validation_py = os.path.join(repo_root, "src", "chessbot", "validation.py")
    if not os.path.exists(validation_py):
        raise FileNotFoundError(f"validation.py not found at: {validation_py}")

    with open(validation_py, "r", encoding="utf-8") as fh:
        source = fh.read()

    m = re.search(r"SF_TABLE_DEFAULT\s*=\s*(\[.*?\])", source, re.DOTALL)
    if not m:
        raise ValueError("SF_TABLE_DEFAULT not found in validation.py")

    table = ast.literal_eval(m.group(1))
    if not table:
        raise ValueError("SF_TABLE_DEFAULT is empty in validation.py")

    depth_map = {}
    for entry in table:
        d = int(entry["depth"])
        if d not in depth_map:
            depth_map[d] = int(entry["elo"])

    return depth_map, table


def find_validation_histories(tags, selfplay_dir):
    """Return list of (tag, path) for existing validation_history.jsonl files."""
    found = []
    for tag in tags:
        path = os.path.join(selfplay_dir, tag, HISTORY_FILENAME)
        if os.path.exists(path):
            found.append((tag, path))
    return found


def parse_jsonl(path):
    """Parse JSONL file, return list of dicts. Raises on invalid JSON."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i} in {path}: {e}")
    return records


def is_fixed_depth_sf_record(record):
    """
    Return True for fixed-depth Stockfish validation rows.
    These have a positive integer 'depth' and an 'sf_elo' field.
    """
    return (
        isinstance(record.get("depth"), int)
        and record["depth"] > 0
        and "sf_elo" in record
    )


def estimate_model_elo(sf_elo, score):
    """
    Same formula as validation.py:estimate_model_elo.
    R_model = R_op - 400 * log10(1/s - 1), clamped score to (eps, 1-eps).
    """
    eps = 1e-6
    s = min(max(score, eps), 1.0 - eps)
    return sf_elo - 400.0 * math.log10((1.0 / s) - 1.0)


def recompute_record(record, depth_map):
    """
    Return an updated copy of record with new sf_elo and model_elo.
    Saves originals to sf_elo_original / model_elo_original on first call only.
    Raises ValueError if the record's depth is not in depth_map.
    """
    depth = int(record["depth"])
    if depth not in depth_map:
        raise ValueError(
            f"Depth {depth} not in updated SF table. "
            f"Available: {sorted(depth_map.keys())}"
        )

    new_sf_elo = depth_map[depth]

    wins = record.get("wins", 0)
    draws = record.get("draws", 0)
    n_games = record.get("n_games", 0)
    if n_games > 0:
        score = (wins + 0.5 * draws) / n_games
    else:
        score = float(record.get("score", 0.5))

    new_model_elo = round(estimate_model_elo(new_sf_elo, score), 2)

    updated = dict(record)

    # Preserve originals only on first application
    if "sf_elo_original" not in updated:
        updated["sf_elo_original"] = record["sf_elo"]
    if "model_elo_original" not in updated:
        updated["model_elo_original"] = record["model_elo"]

    updated["sf_elo"] = new_sf_elo
    updated["model_elo"] = new_model_elo

    return updated


def backup_file(path):
    """Create a timestamped backup. Returns backup path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak_{ts}"
    shutil.copy2(path, backup_path)
    return backup_path


def write_jsonl_atomic(path, records):
    """
    Write records as JSONL to a temp file, verify all lines parse cleanly,
    then atomically replace the original via os.replace.
    """
    dir_ = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Verify round-trip
        with open(tmp_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    json.loads(s)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Verification failed at line {i}: {e}")

        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def process_file(tag, path, depth_map, apply=False):
    """
    Process one validation history file.
    Returns a summary dict with counts, examples, and backup path.
    """
    records = parse_jsonl(path)

    updated_records = []
    n_updated = 0
    n_skipped = 0
    examples = []

    for record in records:
        if not is_fixed_depth_sf_record(record):
            updated_records.append(record)
            n_skipped += 1
            continue

        updated = recompute_record(record, depth_map)

        if updated == record:
            # Already up to date — no change
            updated_records.append(record)
            n_skipped += 1
        else:
            if len(examples) < 3:
                examples.append({
                    "depth": record["depth"],
                    "old_sf_elo": record["sf_elo"],
                    "new_sf_elo": updated["sf_elo"],
                    "old_model_elo": record["model_elo"],
                    "new_model_elo": updated["model_elo"],
                })
            updated_records.append(updated)
            n_updated += 1

    backup_path = None
    if apply and n_updated > 0:
        backup_path = backup_file(path)
        write_jsonl_atomic(path, updated_records)

    return {
        "tag": tag,
        "path": path,
        "n_records": len(records),
        "n_updated": n_updated,
        "n_skipped": n_skipped,
        "backup_path": backup_path,
        "examples": examples,
    }


def run_tests():
    """Self-contained tests. Returns True if all pass."""
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} {detail}")
            failed += 1

    print("Running tests...")

    # 1. ELO shift is purely additive: model_elo shifts by exactly (new_sf - old_sf)
    for score in (0.5, 0.75, 0.3, 0.99):
        old_sf, new_sf = 2586, 2759
        delta_sf = new_sf - old_sf
        delta_model = estimate_model_elo(new_sf, score) - estimate_model_elo(old_sf, score)
        check(
            f"ELO shift additive at score={score}",
            abs(delta_model - delta_sf) < 1e-9,
            f"delta_model={delta_model:.6f} delta_sf={delta_sf}",
        )

    # 2. Explicit example from task spec: score -> +150 elo over opponent
    #    if score implies +150 elo over old_sf=2586, model_elo should be 2736
    old_sf = 2586
    target_delta = 150.0
    # score that gives +150 over 2586
    score_for_150 = 1.0 / (1.0 + 10 ** (-target_delta / 400.0))
    m_old = estimate_model_elo(old_sf, score_for_150)
    m_new = estimate_model_elo(2759, score_for_150)
    check(
        "Spec example: 2586->2759 shifts model by 173",
        abs(m_new - m_old - (2759 - 2586)) < 1e-9,
    )

    # 3. Idempotency: recompute_record twice gives same result, originals not nested
    depth_map = {10: 2912}
    rec = {
        "ts": 1000, "run_tag": "test_run", "n_games": 200,
        "wins": 110, "draws": 40, "score": 0.65,
        "sf_elo": 2689, "model_elo": 2800.0,
        "should_count": True, "consec_over_50": 1,
        "bumped": False, "action": "none",
        "depth": 10, "depth_index": 5,
    }
    r1 = recompute_record(rec, depth_map)
    r2 = recompute_record(r1, depth_map)
    check("Idempotent: sf_elo_original unchanged", r2["sf_elo_original"] == r1["sf_elo_original"])
    check("Idempotent: model_elo_original unchanged", r2["model_elo_original"] == r1["model_elo_original"])
    check("Idempotent: sf_elo unchanged", r1["sf_elo"] == r2["sf_elo"])
    check("Idempotent: model_elo unchanged", r1["model_elo"] == r2["model_elo"])
    check("Idempotent: r1 == r2", r1 == r2)

    # 4. Process-file idempotency via temp JSONL
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, HISTORY_FILENAME)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        res1 = process_file("t", p, depth_map, apply=True)
        recs_after_1 = parse_jsonl(p)
        res2 = process_file("t", p, depth_map, apply=True)
        recs_after_2 = parse_jsonl(p)
        check("Process-file first run: 1 updated", res1["n_updated"] == 1)
        check("Process-file second run: 0 updated", res2["n_updated"] == 0)
        check("Process-file second run: records unchanged", recs_after_1 == recs_after_2)

    # 5. Backup created in apply mode
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, HISTORY_FILENAME)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        res = process_file("t", p, depth_map, apply=True)
        check("Backup created", res["backup_path"] is not None and os.path.exists(res["backup_path"]))
        check("Backup is readable JSONL", len(parse_jsonl(res["backup_path"])) == 1)
        check("Backup contains original sf_elo", parse_jsonl(res["backup_path"])[0]["sf_elo"] == 2689)

    # 6. Skip non-SF record
    non_sf = {"ts": 9999, "note": "lc0 match", "score": 0.5}
    check("Non-SF record: not is_fixed_depth_sf_record", not is_fixed_depth_sf_record(non_sf))
    check("Fixed-depth record: is_fixed_depth_sf_record", is_fixed_depth_sf_record(rec))

    # 7. Missing depth raises, not silently skipped
    depth_map_partial = {5: 2092}
    try:
        recompute_record(rec, depth_map_partial)
        check("Missing depth raises ValueError", False, "(no exception raised)")
    except ValueError:
        check("Missing depth raises ValueError", True)

    # 8. Atomic write correctness
    with tempfile.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "test.jsonl")
        recs_in = [{"a": 1, "b": 2}, {"x": "hello"}]
        write_jsonl_atomic(p, recs_in)
        recs_out = parse_jsonl(p)
        check("Atomic write round-trips correctly", recs_out == recs_in)

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Recalibrate validation ELOs after SF depth table update."
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write changes (default is dry-run)",
    )
    parser.add_argument(
        "--tags", "--tag", nargs="+", default=None, metavar="TAG",
        help="Process only these run tags (default: all target tags)",
    )
    parser.add_argument(
        "--root", default=None,
        help="Repo root (default: parent of this script's directory)",
    )
    parser.add_argument(
        "--selfplay-dir", default=DEFAULT_SELFPLAY_DIR, metavar="DIR",
        help=f"Directory containing run-tag subdirs (default: {DEFAULT_SELFPLAY_DIR})",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run tests and exit",
    )
    args = parser.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)

    # Resolve repo root
    if args.root:
        repo_root = os.path.abspath(args.root)
    else:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Load the authoritative SF table
    print(f"Loading SF depth table from: {repo_root}")
    try:
        depth_map, table = load_sf_depth_table(repo_root)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    unique_depths = sorted(depth_map)
    print(f"SF table loaded: {len(table)} entries, {len(depth_map)} unique depths")
    print(f"  Depth range: {unique_depths[0]} - {unique_depths[-1]}")
    print(f"  ELO range:   {depth_map[unique_depths[0]]} - {depth_map[unique_depths[-1]]}")

    tags = args.tags if args.tags else TARGET_TAGS

    # Find existing history files
    found = find_validation_histories(tags, args.selfplay_dir)
    missing_tags = sorted(set(tags) - {t for t, _ in found})

    print(f"\nFound {len(found)} validation history file(s)  ({len(missing_tags)} tag(s) not on disk)")
    if missing_tags:
        print(f"  Not found (skipped): {missing_tags}")

    if not found:
        print("Nothing to process.")
        sys.exit(0)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\nMode: {mode}")
    print("=" * 64)

    total_records = 0
    total_updated = 0
    total_skipped = 0

    for tag, path in found:
        try:
            result = process_file(tag, path, depth_map, apply=args.apply)
        except Exception as e:
            print(f"ERROR processing {tag}: {e}", file=sys.stderr)
            sys.exit(1)

        total_records += result["n_records"]
        total_updated += result["n_updated"]
        total_skipped += result["n_skipped"]

        status = "updated" if result["n_updated"] > 0 else "no changes"
        print(
            f"{tag:30s}  {result['n_records']:3d} records  "
            f"{result['n_updated']:3d} updated  {result['n_skipped']:3d} skipped"
            f"  [{status}]"
        )
        if result["backup_path"]:
            print(f"  backup -> {result['backup_path']}")
        for ex in result["examples"]:
            print(
                f"  depth={ex['depth']:2d}:  sf_elo {ex['old_sf_elo']:4d} -> {ex['new_sf_elo']:4d}  |"
                f"  model_elo {ex['old_model_elo']:8.2f} -> {ex['new_model_elo']:8.2f}"
            )

    print("=" * 64)
    print(
        f"Total: {total_records} records scanned, "
        f"{total_updated} updated, {total_skipped} skipped"
    )
    if not args.apply and total_updated > 0:
        print("(dry-run -- pass --apply to write changes)")


if __name__ == "__main__":
    main()
