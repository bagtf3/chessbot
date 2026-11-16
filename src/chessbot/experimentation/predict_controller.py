"""Run a trained controller over one or more collections: save one row per
trajectory (both heads at every check-in, plus the fields a stop rule replay
needs), then print the fire and gate tables on the trajectories the checkpoint
never trained or validated on (its test split, plus anything it never saw).

python -m chessbot.experimentation.predict_controller <checkpoint .pt> <collection> [...]
"""
import argparse
import json
import os

import torch

from chessbot.experimentation import train_controller
from chessbot.experimentation.collector import COLLECTIONS_DIR
from chessbot.experimentation.train_controller import (
    VAL_BATCH, Controller, load_collection, load_metadata, predict, read_shard, report,
    save_predictions, trajectory_rows)

SPLIT_LABELS = {"train": "train", "validation": "val", "test": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    # one or more collection dirs or names; not used with --test-only
    ap.add_argument("collections", nargs="*")
    # default: <checkpoint>_on_<first collection>.parquet next to the checkpoint,
    # or <checkpoint>_predictions.parquet with --test-only
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # score only the checkpoint's own test set, read from its held-out shards,
    # as its training run's final tables would have
    ap.add_argument("--test-only", action="store_true")
    args = ap.parse_args()
    if not args.test_only and not args.collections:
        ap.error("name at least one collection, or pass --test-only")

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    # a quality-only checkpoint's stability head is untrained: stop rules skip it
    heads = ckpt.get("training_config", {}).get("heads", "both")
    train_controller.ACTIVE_HEADS = (heads == "both", True)
    model = Controller(**ckpt["model_config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    split = ckpt.get("split", {})
    seen = {tid: SPLIT_LABELS[s] for s in SPLIT_LABELS for tid in split.get(s, [])}
    # checkpoints that list no train IDs trained on the rest of their collections
    trained_on = set(split.get("collections", ()))
    print(f"[ctrl] {args.checkpoint}  spec {ckpt['spec']}  step {ckpt.get('step')}  "
          f"device {args.device}")

    if args.test_only:
        test_ids = set(split["test"])
        data = [t for p in split["held_out_shards"]
                for t in read_shard(os.path.join(COLLECTIONS_DIR, p))
                if t["trajectory_id"] in test_ids]
        print(f"[ctrl] test {len(data)} of {len(test_ids)} traj from "
              f"{len(split['held_out_shards'])} held-out shards")
        preds = predict(model, data, VAL_BATCH, args.device)
        out = args.out or os.path.splitext(args.checkpoint)[0] + "_predictions.parquet"
        save_predictions(out, trajectory_rows(preds, "test"))
        save_tables(out, report(preds, "test"))
        return

    dirs = [c if os.path.isdir(c) else os.path.join(COLLECTIONS_DIR, c)
            for c in args.collections]
    groups = {}
    for d in dirs:
        data = load_collection(d, load_metadata(d))
        for (t, prob) in predict(model, data, VAL_BATCH, args.device):
            label = seen.get(t["trajectory_id"],
                             "train" if t["collection"] in trained_on else "unseen")
            groups.setdefault(label, []).append((t, prob))
        print(f"[ctrl] {d}  {len(data)} traj")
    print("[ctrl] " + "  ".join(f"{k} {len(v)}" for k, v in sorted(groups.items())))

    out = args.out or (os.path.splitext(args.checkpoint)[0]
                       + f"_on_{os.path.basename(os.path.normpath(dirs[0]))}.parquet")
    save_predictions(out, [row for label, preds in groups.items()
                           for row in trajectory_rows(preds, label)])
    held = groups.get("test", []) + groups.get("unseen", [])
    if held:
        save_tables(out, report(held, "held-out (test + unseen)"))


def save_tables(parquet_path, tables):
    """The printed fire and gate tables, as <parquet stem>_tables.json."""
    path = os.path.splitext(parquet_path)[0] + "_tables.json"
    with open(path, "w") as f:
        json.dump(tables, f, indent=2)
    print(f"[ctrl] tables -> {path}")


if __name__ == "__main__":
    main()
