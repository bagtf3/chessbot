# Xerces Sprint Plans

## Byte-encode xc0 records for faster parsing

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
