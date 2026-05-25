"""
build_grad_snapshot.py — Extract N records from pretrain tfrecords into a pkl.

Reads a random sample of .tfrecord.gz files, applies the same preprocessing
as pretrain.py (uniform blend + clip + renormalize), and saves records as a
pkl for reuse in gradient analysis.

Saved format: list of tuples (enc_in, mask, policy, Y_wdl, vwht, pwht)
matching the pending_training pkl shard format.  Also saves a .npz with
stacked arrays for fast loading by analyze_pt_weights.py.

Usage:
  python scripts/build_grad_snapshot.py [--tfrec-dir PATH] [--n 20480] [--out PATH]
"""
import argparse
import os
import pickle
import numpy as np

TFREC_DIR   = r"C:\Users\Bryan\Data\chessbot_data\training_data\wdl"
N_DEFAULT   = 20480
OUT_DEFAULT = r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs\val_test\grad_analysis_snapshot.pkl"

UNIFORM_BLEND   = 0.05
POLICY_MAX_CLIP = 0.6


def main():
    import tensorflow as tf

    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrec-dir", default=TFREC_DIR)
    parser.add_argument("--n",         type=int, default=N_DEFAULT)
    parser.add_argument("--out",       default=OUT_DEFAULT)
    args = parser.parse_args()

    files = sorted(
        os.path.join(args.tfrec_dir, f)
        for f in os.listdir(args.tfrec_dir)
        if f.endswith(".tfrecord.gz")
    )
    rng   = np.random.default_rng(42)
    files = rng.permutation(files).tolist()
    print(f"Found {len(files)} tfrecord files, extracting {args.n} records...")

    feature_spec = {
        "enc_in":        tf.io.FixedLenFeature([], tf.string),
        "mask":          tf.io.FixedLenFeature([], tf.string),
        "policy_logits": tf.io.FixedLenFeature([], tf.string),
        "value_out":     tf.io.FixedLenFeature([3], tf.float32),
        "weight":        tf.io.FixedLenFeature([], tf.float32),
    }

    records = []
    for filepath in files:
        if len(records) >= args.n:
            break
        try:
            ds = tf.data.TFRecordDataset(filepath, compression_type="GZIP")
            for raw in ds:
                if len(records) >= args.n:
                    break
                feat   = tf.io.parse_single_example(raw, feature_spec)
                enc_in = tf.cast(
                    tf.io.parse_tensor(feat["enc_in"], out_type=tf.int16), tf.int32
                ).numpy().astype(np.int32)
                mask   = tf.io.parse_tensor(feat["mask"], out_type=tf.int32).numpy()
                policy = tf.io.parse_tensor(
                    feat["policy_logits"], out_type=tf.float32
                ).numpy().astype(np.float32)
                value  = feat["value_out"].numpy().astype(np.float32)
                weight = float(feat["weight"].numpy())

                mask_f  = mask.astype(np.float32)
                n_legal = mask_f.sum()
                if n_legal > 0:
                    policy = (
                        (1.0 - UNIFORM_BLEND) * policy
                        + UNIFORM_BLEND * (mask_f / n_legal)
                    )
                policy = np.minimum(policy, POLICY_MAX_CLIP)
                s = policy.sum()
                if s > 0:
                    policy /= s

                records.append((enc_in, mask, policy, value, weight, weight))
        except Exception as e:
            print(f"  skipped {os.path.basename(filepath)}: {e}")

    if len(records) < args.n:
        print(f"  WARNING: only got {len(records)} records (wanted {args.n})")

    idx     = rng.permutation(len(records))
    records = [records[i] for i in idx]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(records, fh, protocol=4)
    print(f"Saved {len(records)} records -> {args.out}")

    x0, m0, p0, v0, vw0, pw0 = records[0]
    n_legal = int((m0 > 0).sum())
    ent     = float(-(p0 * np.log(p0 + 1e-9)).sum())
    print(f"Sample:  enc_in={x0.shape}  mask={m0.shape}  policy={p0.shape}  value={v0.shape}")
    print(f"  policy: entropy={ent:.3f} nats  legal_moves={n_legal}  "
          f"max={p0.max():.3f}  sum={p0.sum():.4f}")
    print(f"  value (WDL)={v0.round(3)}  weight={vw0:.3f}")

    X    = np.stack([r[0] for r in records])
    P    = np.stack([r[2] for r in records])
    Y    = np.stack([r[3] for r in records])
    vwht = np.array([r[4] for r in records], dtype=np.float32)
    pwht = np.array([r[5] for r in records], dtype=np.float32)
    print(f"\nAggregate stats over {len(records)} records:")
    print(f"  policy entropy: mean={float(-(P * np.log(P + 1e-9)).sum(1).mean()):.3f} nats")
    print(f"  value W/D/L:    {Y.mean(0).round(3)}")
    print(f"  weight:         mean={vwht.mean():.3f}  min={vwht.min():.3f}  max={vwht.max():.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
