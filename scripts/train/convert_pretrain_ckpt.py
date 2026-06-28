"""Convert a pretraining checkpoint to selfplay .pt/.ts model files.

Usage:
    python scripts/convert_pretrain_ckpt.py <src_ckpt> <run_tag>

Example:
    python scripts/convert_pretrain_ckpt.py \
        C:/Users/Bryan/Data/chessbot_data/selfplay_runs/val_test_multi/16m-precond-smartgate_pt_ckpt2999.pt \
        16m_precond_run0
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessbot.model import PT_BUILDERS, VARIANTS
from chessbot.train_pytorch import save_pt_model
from chessbot import SP_DIR


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    src_path = sys.argv[1]
    run_tag  = sys.argv[2]
    run_dir  = os.path.join(SP_DIR, run_tag)

    print(f"[convert] loading {src_path}")
    obj = torch.load(src_path, map_location="cpu")

    if isinstance(obj, dict) and "model" in obj:
        state_dict = obj["model"]
        arch       = obj.get("arch", None)
    else:
        raise ValueError(f"Unrecognized checkpoint format: expected dict with 'model' key")

    if arch is None or arch not in PT_BUILDERS:
        raise ValueError(f"Unknown or missing arch '{arch}' in checkpoint (known: {list(PT_BUILDERS)})")

    print(f"[convert] arch={arch}, rebuilding model ...")
    model = PT_BUILDERS[arch](VARIANTS[arch])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[convert] WARNING missing keys: {missing}")
    if unexpected:
        print(f"[convert] WARNING unexpected keys: {unexpected}")

    ts_path = os.path.join(run_dir, f"{run_tag}_model.ts")
    print(f"[convert] saving -> {ts_path}  (+ companion .pt)")
    save_pt_model(model, ts_path, arch=arch)
    print("[convert] done.")


if __name__ == "__main__":
    main()
