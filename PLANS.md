# Xerces Sprint Plans

## 1. Spatial helper planes in xc0h stem (next architecture version)

Add three fixed spatial planes alongside the existing ones-plane, giving the conv
pre-blocks explicit coordinate information from the very first layer.

### Planes
- **Checkerboard** `(rank + file) % 2 * 2 - 1` — alternating ±1, light/dark squares
- **Rank gradient** `rank / 7 * 2 - 1` — −1 at rank 1, +1 at rank 8 (toward opponent)
- **File gradient** `file / 7 * 2 - 1` — −1 at file A, +1 at file H

### Design
All four planes (ones + 3 spatial) are appended **after** `xc0h_proj_ln`, not fed
through it. LN would subtract the constant/gradient value into each square's per-channel
mean and destroy the spatial variation — the whole point. This is why the existing
ones-plane lives post-LN.

Projection shrinks from `CF - 1` to `CF - 4`; the 4 recovered channels hold the
spatial planes. Total output remains `[B, 64, CF]`.

Each plane gets its own learnable scale (init 1.0), same pattern as `xc0h_ones_scale`.
Convs can attenuate any plane toward zero if the geometry isn't useful; cost of ignoring
is near-zero (near-zero weights in 4 input channels).

### Implementation in model.py
In `__init__`, after the existing ones-scale:
```python
self.xc0h_proj    = nn.Linear(xc0h_in_ch, CF - 4)   # was CF - 1
self.xc0h_proj_ln = nn.LayerNorm(CF - 4)             # was CF - 1
self.xc0h_ones_scale    = nn.Parameter(torch.tensor(1.0))
self.xc0h_checker_scale = nn.Parameter(torch.tensor(1.0))
self.xc0h_rank_scale    = nn.Parameter(torch.tensor(1.0))
self.xc0h_file_scale    = nn.Parameter(torch.tensor(1.0))

# precompute all 4 planes as a single [1, 64, 4] buffer — store once, summon in forward
sqs = torch.arange(64).float()
ranks_sq = sqs // 8
files_sq = sqs % 8
self.register_buffer("xc0h_spatial", torch.stack([
    torch.ones(64),                              # ones-plane
    ((ranks_sq + files_sq) % 2) * 2 - 1,        # checkerboard
    ranks_sq / 7.0 * 2 - 1,                     # rank gradient
    files_sq / 7.0 * 2 - 1,                     # file gradient
], dim=-1).unsqueeze(0))                         # [1, 64, 4]
```

In `forward`:
```python
projected = self.xc0h_proj_ln(self.xc0h_proj(fused))   # [B, 64, CF-4]
scales  = torch.stack([self.xc0h_ones_scale, self.xc0h_checker_scale,
                       self.xc0h_rank_scale, self.xc0h_file_scale])
spatial = self.xc0h_spatial.to(projected.dtype).expand(B, -1, -1) * scales  # [B, 64, 4]
x = torch.cat([projected, spatial], dim=-1)             # [B, 64, CF]
```

### Weight compatibility warning
`xc0h_proj` changes shape from `(CF-1, xc0h_in_ch)` to `(CF-4, xc0h_in_ch)` —
**this breaks loading existing xc0h checkpoints with `strict=True`**. Do this only
at the start of a new run, never mid-run. Migration path for warm-starting from an
existing checkpoint: load with `strict=False`, proj + proj_ln reinit randomly,
all other weights (transformer, heads, embeddings) transfer cleanly.

## 2. Byte-encode xc0 records for faster parsing

xc0 tfrecords currently store `xc0h_board` as a raw bytes blob and `policy` as
float32 bytes, parsed by TF's proto machinery at training time. lc0's V6 binary
format parses much faster because the struct layout is fixed and tight — no
proto overhead, direct struct.unpack.

Goal: define a fixed binary layout for xc0h-K6 records analogous to lc0's V6
struct, write a matching parser in `xerces_training/chunkparser.py` or a new
`xc0_chunkparser.py`, and update `build_bootstrap_records.py` and the training
loader to use it. Expected benefit: meaningfully faster data throughput at
training time, especially with many workers. Design the struct so K is fixed at
build time (e.g. K=6 → `int16[393] + float32[1858] + float32[3]` = tight pack)
and the file format uses simple gzip-compressed binary chunks with a small
fixed-size header, mirroring lc0's approach.
