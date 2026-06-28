import sys, pickle, gc
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\Bryan\repos\chessbot\src")

SNAPSHOT = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\precond_run2\grad_analysis_snapshot.pkl"
PT_PATH  = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\16m_precond_run0\16m_precond_run0_model.pt"
BATCH    = 2048
DEVICE   = torch.device("cuda")

import pyfastchess
from chessbot.model import PT_BUILDERS, VARIANTS
sl_idx = np.where(pyfastchess.build_sometimes_legal_mask())[0]  # (1858,) indices into 4288
up_mask = sl_idx > 4095                          # underpromo slots within the 1858
print(f"sl_idx: {sl_idx.shape[0]}  underpromo (4288 idx > 4095): {int(up_mask.sum())}")

with open(SNAPSHOT, "rb") as f:
    data = pickle.load(f)
N = len(data)
tokens_np = np.stack([data[i][0] for i in range(N)]).astype(np.int32)  # (N, 64)
pol4288_stored = np.stack([data[i][2] for i in range(N)]).astype(np.float32)  # stored NN out
print(f"samples: {N}")

# did the real test set ever USE an underpromo slot (legal move there)?
up_active = pol4288_stored[:, sl_idx[up_mask]]      # (N, 66) stored logits at underpromo
samples_with_up = int((np.abs(up_active) > 1e-6).any(axis=1).sum())
print(f"samples with nonzero stored signal on an underpromo slot: {samples_with_up}")


def run(policy_1858):
    obj  = torch.load(PT_PATH, map_location=DEVICE)
    arch = obj.get("arch", "16m-precond-smartgate")
    model = PT_BUILDERS[arch](VARIANTS[arch], policy_1858=policy_1858).to(DEVICE).eval()
    model.load_state_dict(obj["model"], strict=False)
    outs = []
    with torch.no_grad():
        for s in range(0, N, BATCH):
            tt = torch.from_numpy(tokens_np[s:s+BATCH]).long().to(DEVICE)
            pol, _ = model(tt)
            outs.append(pol.float().cpu().numpy())
    del model, obj; gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


pol4288 = run(False)   # (N, 4288)
print(f"Phase A done: {pol4288.shape}")
pol1858 = run(True)    # (N, 1858)
print(f"Phase B done: {pol1858.shape}")

expected = pol4288[:, sl_idx]            # (N, 1858)
diff = np.abs(pol1858 - expected)
up_diff = diff[:, up_mask]               # (N, 66)

print(f"\nALL 1858 slots   max diff: {diff.max():.2e}  mean: {diff.mean():.2e}")
print(f"UNDERPROMO slots max diff: {up_diff.max():.2e}  mean: {up_diff.mean():.2e}")
print(f"underpromo logit range in 1858 output: "
      f"[{pol1858[:, up_mask].min():.3f}, {pol1858[:, up_mask].max():.3f}]")

ok = diff.max() < 1e-4 and samples_with_up > 0
print("\nPASS" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
