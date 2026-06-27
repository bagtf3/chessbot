"""
POC: verify that the modified model (no scatter, 1858 output) produces policy values
identical to gathering the 1858 sometimes-legal slots from the original 4288 output.

Old path: .ts model -> (B, 4288) -> gather sl_idx -> (B, 1858)
New path: .pt weights loaded into modified arch -> (B, 1858) directly

Both should be equal within float16 precision.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pyfastchess

sys.path.insert(0, "C:/Users/Bryan/repos/chessbot/src")

RUN_DIR = "C:/Users/Bryan/Data/chessbot_data/selfplay_runs/16m_precond_run0"
PT_PATH = f"{RUN_DIR}/16m_precond_run0_model.pt"

# sl_idx: positions in 4288 space that are sometimes-legal
sl_mask = pyfastchess.build_sometimes_legal_mask()
sl_idx = torch.from_numpy(sl_mask).bool().nonzero(as_tuple=True)[0]
print(f"sometimes-legal slots: {sl_idx.shape[0]}")

# test positions
POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",   # startpos
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",   # italian
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # endgame
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
]

def encode_position(fen):
    board = pyfastchess.Board()
    # push moves to reach position from FEN
    # pyfastchess.Board() accepts FEN via set_fen if available, else use uci path
    # use encode_64_tokens directly after setting up board
    # Board constructor takes fen string
    board = pyfastchess.Board(fen)
    tokens = board.encode_64_tokens()
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, 64]

from chessbot.model import PT_BUILDERS, VARIANTS
obj = torch.load(PT_PATH, map_location="cpu")
arch = obj.get("arch", "16m-precond-smartgate")

# old model: same weights, policy_1858=False -> (B, 4288)
print("loading .pt as old arch (4288 output)...")
old_model = PT_BUILDERS[arch](VARIANTS[arch], policy_1858=False)
old_model.load_state_dict(obj["model"], strict=False)
old_model.eval()

# new model: same weights, policy_1858=True -> (B, 1858)
print("loading .pt as new arch (1858 output)...")
new_model = PT_BUILDERS[arch](VARIANTS[arch], policy_1858=True)
new_model.load_state_dict(obj["model"], strict=False)
new_model.eval()
print(f"  arch: {arch}")

print()
print(f"{'position':<12} {'max_absdiff':>12} {'mean_absdiff':>13} {'match':>8}")
print("-" * 50)

all_match = True
with torch.no_grad():
    for i, fen in enumerate(POSITIONS):
        tokens = encode_position(fen)

        old_out = old_model(tokens)
        if isinstance(old_out, (tuple, list)):
            pol_4288, _ = old_out
        else:
            pol_4288 = old_out
        pol_4288 = pol_4288.float()

        # gather the 1858 from 4288 output
        expected_1858 = pol_4288[:, sl_idx]  # [1, 1858]

        new_out = new_model(tokens)
        if isinstance(new_out, (tuple, list)):
            pol_1858, _ = new_out
        else:
            pol_1858 = new_out
        pol_1858 = pol_1858.float()

        assert pol_1858.shape == (1, 1858), f"bad shape: {pol_1858.shape}"

        diff = (expected_1858 - pol_1858).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        match = max_diff < 1e-3  # float16 tolerance
        all_match = all_match and match

        label = f"pos_{i}"
        print(f"{label:<12} {max_diff:>12.6f} {mean_diff:>13.6f} {'OK' if match else 'FAIL':>8}")

print()
if all_match:
    print("PASS: 1858 outputs match gathered 4288 slices on all positions")
else:
    print("FAIL: mismatch detected")

# sanity: new model output shape
tokens = encode_position(POSITIONS[0])
with torch.no_grad():
    pol, wdl = new_model(tokens)
print(f"\npolicy shape: {tuple(pol.shape)}  (expected (1, 1858))")
print(f"wdl shape:    {tuple(wdl.shape)}  (expected (1, 3))")
print(f"wdl (softmax): {F.softmax(wdl.float(), dim=-1).squeeze().tolist()}")
