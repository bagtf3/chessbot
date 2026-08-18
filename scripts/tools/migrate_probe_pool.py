"""
One-time conversion: old pos_id/list-keyed blunder pool -> short_fen/dict.

Run once per pool file. The original is backed up alongside the live path
(<path>.pre_migration.pkl.gz) rather than deleted -- hard-delete only after
confirming probe_table.py/pairwise.py give consistent results against the
migrated file.

Per record:
  - key becomes short_fen(fen), dropping pos_id entirely
  - candidate_moves and the TREE_COLS (Q_stm, avg_depth, best_wdl,
    children_visited, max_depth, selection_method, sims, stop_reason,
    total_children) are dropped -- recoverable from analyze_results_combined
    .pkl via run_tag+game_id+move_num if ever needed, not worth carrying in
    a pool meant to stay lean
  - kl/ce dropped for the same reason (also absent from the new live
    rescore.py ingestion path, so this keeps old and new records the same
    shape going forward)
  - z_stm/xc0_result collapse into a single z_stm field (they were always
    the same value for a given origin -- xc0 has one fixed color for the
    whole game in the validation pool, so game-level xc0_result already IS
    that game's z_stm)
  - retired=True / streak dropped; probes[] backfilled with found_equiv and
    same_move (both derivable from the already-stored cpl and move vs the
    record's own played_move), and eviction is applied retroactively: a
    record whose trailing probe history would already qualify for eviction
    under the new rules (or that was already retired under the old ones)
    goes straight to the evicted recovery log instead of the live pool, so
    day-one state matches what the new rules would have produced all along
  - times_seen starts at 1 -- multiplicity before this migration isn't
    reconstructable from a flat list, only bumped going forward
  - scenario cannot be recovered for pre-existing records (the field didn't
    exist yet when they were scanned) and is left unset

On a short_fen collision between two old records (possible now that
halfmove/fullmove counters are dropped from the key), the richer one wins:
deeper max cached SF depth, then more probe history.
"""
import argparse
import gzip
import glob
import json
import pickle
import time

from chessbot.blunder_replay import (
    BRP_EQUIV_CPL, evicted_path, save_probe_pool, short_fen,
)

SELFPLAY_RUNS_GLOB = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/*"
# legacy threshold in effect when this one-time migration ran -- decoupled
# from blunder_replay.py's current live eviction policy, which no longer
# has a streak concept
LEGACY_EVICT_STREAK = 2


def convert_record(old):
    z_stm = old.get('z_stm')
    if z_stm is None or z_stm != z_stm:  # NaN check without importing numpy
        z_stm = old.get('xc0_result')

    new = {
        'fen': old['fen'],
        'uci_path': old['uci_path'],
        'start_ply': old.get('start_ply'),
        'run_tag': old.get('run_tag'),
        'game_id': old.get('game_id'),
        'move_num': old.get('move_num'),
        'model_epoch': old.get('model_epoch'),
        'scenario': old.get('scenario'),
        'z_stm': z_stm,
        'played_move': old.get('played_move'),
        'sf_best_move': old.get('sf_best_move'),
        'best_cp': old.get('best_cp'),
        'played_cp': old.get('played_cp'),
        'cpl': old.get('cpl'),
        'times_seen': 1,
    }

    if old.get('deep_evals'):
        new['deep_evals'] = old['deep_evals']
    if old.get('deep_depths'):
        new['deep_depths'] = old['deep_depths']
    if 'deep_best_move' in old:
        new['deep_best_move'] = old['deep_best_move']
    if 'deep_best_cp' in old:
        new['deep_best_cp'] = old['deep_best_cp']
    if 'sf_ms' in old:
        new['sf_ms'] = old['sf_ms']

    probes = []
    for p in (old.get('probes') or []):
        pp = dict(p)
        pp.setdefault('found_equiv', pp.get('cpl', 1e9) <= BRP_EQUIV_CPL)
        pp.setdefault('same_move', pp.get('move') == old.get('played_move'))
        probes.append(pp)
    if probes:
        new['probes'] = probes

    return new


def trailing_equiv_streak(probes):
    streak = 0
    for p in probes:
        streak = streak + 1 if p.get('found_equiv') else 0
    return streak


def eviction_reason(old, probes):
    if old.get('retired'):
        return 'legacy_retired'
    if not probes:
        return None
    last = probes[-1]
    if last.get('same_move') and last.get('found_equiv'):
        return 'same_equiv'
    if trailing_equiv_streak(probes) >= LEGACY_EVICT_STREAK:
        return 'streak'
    return None


def richness(rec):
    depth = max((rec.get('deep_depths') or {}).values(), default=0)
    return (depth, len(rec.get('probes') or []))


def migrate_pool(old_list):
    new_pool = {}
    evicted_records = []
    collisions = 0

    for old in old_list:
        new = convert_record(old)
        probes = new.get('probes') or []
        reason = eviction_reason(old, probes)
        key = short_fen(new['fen'])

        if reason is not None:
            evicted_records.append({
                'short_fen': key,
                'fen': new['fen'],
                'uci_path': new['uci_path'],
                'run_tag': new['run_tag'],
                'game_id': new['game_id'],
                'move_num': new['move_num'],
                'start_ply': new['start_ply'],
                'model_epoch': new['model_epoch'],
                'scenario': new['scenario'],
                'evicted_epoch': probes[-1]['epoch'] if probes else None,
                'evicted_ts': int(time.time()),
                'reason': reason,
                'final_probe': probes[-1] if probes else None,
            })
            continue

        if key in new_pool:
            collisions += 1
            if richness(new) > richness(new_pool[key]):
                new_pool[key] = new
            continue

        new_pool[key] = new

    return new_pool, evicted_records, collisions


def fix_probe_history_key(run_dir):
    """n_retired -> n_evicted, one-time rename to match analyse_probe_file's
    new summary key. Returns True if the file was rewritten."""
    path = f"{run_dir}/blunder_probe_history.jsonl"
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return False

    changed = False
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "n_retired" in row:
            row["n_evicted"] = row.pop("n_retired")
            changed = True
        out.append(json.dumps(row, default=float))

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_path")
    args = ap.parse_args()

    with gzip.open(args.pool_path, "rb") as f:
        old_list = pickle.load(f)

    new_pool, evicted_records, collisions = migrate_pool(old_list)

    backup_path = args.pool_path.replace(".pkl.gz", ".pre_migration.pkl.gz")
    with gzip.open(backup_path, "wb") as f:
        pickle.dump(old_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[migrate] backed up {len(old_list)} old-format records -> "
          f"{backup_path}")

    save_probe_pool(new_pool, args.pool_path)

    evfile = evicted_path(args.pool_path)
    if evicted_records:
        with open(evfile, "a", encoding="utf-8") as f:
            for rec in evicted_records:
                f.write(json.dumps(rec, default=float) + "\n")

    print(f"[migrate] {len(old_list)} in -> {len(new_pool)} live, "
          f"{len(evicted_records)} evicted-on-conversion, "
          f"{collisions} short_fen collisions resolved")
    assert len(new_pool) + len(evicted_records) + collisions == len(old_list)

    n_fixed = 0
    for run_dir in glob.glob(SELFPLAY_RUNS_GLOB):
        if fix_probe_history_key(run_dir):
            n_fixed += 1
    print(f"[migrate] rewrote n_retired -> n_evicted in "
          f"{n_fixed} blunder_probe_history.jsonl file(s)")


if __name__ == "__main__":
    main()
