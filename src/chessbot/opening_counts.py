"""Repetition weighting for opening positions.

Selfplay revisits the same handful of openings constantly. Over a 32,957-game
scan of 18m_10c6t_d256_selfplay1, 14 distinct positions carried 10.4% of every
record at ply <= 15, and startpos alone appeared 18,842 times. Training on that
distribution spends capacity relearning the same first four moves.

The fix is a weight, not a filter. Every record is kept; records whose position
has been seen often carry proportionally less sample weight. Mass is restored
per shard by apply_repeat_weights, so the total training signal is unchanged --
it moves off the hot openings and onto everything else.

Counts live in one flat float32 table indexed by a hash of the encoded board's
current-position frame, x[:64]: a single-row count-min sketch. At 1 << 25 slots
(128 MiB) and ~315k distinct openings per run, occupancy stays under 1%; roughly
0.5% of positions share a slot with another, almost always two singletons
colliding into a 0.95 weight. Not worth mitigating.

Keying on x rather than the zobrist hash is deliberate: it makes the slot
recoverable from a bare record. Records that reach a shard without a key --
remaining_untrained reloaded after a restart, seeder output -- are hashed on the
fly at write time and weighted correctly, instead of silently escaping at 1.0.
crc32 and not hash(): the builtin is salted per process, so it would hand out
different slots after every restart and quietly invalidate the saved table.

Only positions at absolute_ply <= MAX_TRACKED_PLY are bumped. That is a true
ply count off the FEN move counters, so UHO games (which open at ply 16)
contribute nothing and are never downweighted, and pre_opened games are tracked
only across the part of their opening inside the window. The gate applies at
bump time, where a board exists; lookups need no gate, because a position that
was never bumped reads back count 0 and weight 1.0 on its own.

Only xerces-played positions bump the table. lc0 records -- distilled main
positions and PV children alike -- neither bump it nor get reweighted by it:
they are appended REWEIGHT_INELIGIBLE at every append site. An lc0 record is a
second, independent observation of a position, so counting it would penalise
the position for being studied, and reweighting it would compound the
suppression already applied to the xc0 record it accompanies.

Counts decay by COUNTS_DECAY_FACTOR on every shard write: half-life 138 shards
(~10,800 games), steady state 200x the per-shard arrival rate. WEIGHT_BANDS is
calibrated against that scale. Raising the decay does not merely lengthen
memory, it multiplies every count and slides positions into harsher bands --
so tune aggression by editing the table, not the decay.
"""
import os
import zlib
from collections import Counter

import numpy as np

TABLE_BITS = 25
TABLE_SIZE = 1 << TABLE_BITS
KEY_MASK = TABLE_SIZE - 1

# Sentinel for "this record must never be reweighted", as distinct from None,
# which means "no key was recorded, hash the record instead". Marked at the
# append site: positions past the ply gate, and every lc0 record without
# exception. Weighting and shard inclusion are separate concerns -- an
# ineligible record is sampled into shards exactly like any other, it just
# keeps the weights it arrived with.
REWEIGHT_INELIGIBLE = -1

# Length of the current-position frame at the head of every token encoding.
POS_FRAME = 64

COUNTS_DECAY_FACTOR = 0.995
MAX_TRACKED_PLY = 15

# (inclusive upper bound on floor(count), weight). Anything above the last
# bound gets FLOOR_WEIGHT.
WEIGHT_BANDS = [
    (1, 1.00), (2, 0.95), (3, 0.88), (4, 0.82), (5, 0.75), (6, 0.70),
    (7, 0.65), (9, 0.60), (12, 0.55), (20, 0.50), (30, 0.45), (60, 0.35),
    (100, 0.25), (500, 0.10),
]
FLOOR_WEIGHT = 0.05

BAND_BOUNDS = np.array([b for b, _ in WEIGHT_BANDS], dtype=np.float32)
BAND_WEIGHTS = np.array(
    [w for _, w in WEIGHT_BANDS] + [FLOOR_WEIGHT], dtype=np.float32)

COUNTS_FILE = "opening_counts.npz"


def absolute_ply(board):
    """True ply of the position, from the FEN move counters.

    Not board.game_ply(): that is a since-construction counter, so a board
    built from a mid-game FEN reports 0 no matter how deep the position is.
    """
    return ((board.fullmove_number() - 1) * 2
            + (0 if board.white_to_move() else 1))


def slot_from_x(x):
    """Table slot from an encoded board's current-position frame.

    crc32 measures like an ideal hash here (28 collisions against 24 expected
    over 39,919 real positions) and costs 5 ms per 10,240-record shard, against
    the ~1 s that shard already spends in gzip.
    """
    return zlib.crc32(np.asarray(x)[:POS_FRAME].tobytes()) & KEY_MASK


def band_weights(counts):
    """floor(count) -> weight, vectorised.

    Floor first: counts are decayed floats, and banding a raw float would drop
    values between integer bounds into the wrong band.
    """
    c = np.floor(np.asarray(counts, dtype=np.float32))
    return BAND_WEIGHTS[np.searchsorted(BAND_BOUNDS, c, side='left')]


class OpeningCounts:
    """Decaying visit counts for opening positions, persisted across runs."""

    def __init__(self, path=None):
        self.counts = np.zeros(TABLE_SIZE, dtype=np.float32)
        self.path = path
        self.n_bumps = 0
        self.n_decays = 0

    def key(self, board, x):
        """Slot for a position we hold a board for, or REWEIGHT_INELIGIBLE
        past the gate.

        Needs both: the board carries the ply, x carries the identity the slot
        is derived from, so a record can be re-keyed later without one.
        """
        if absolute_ply(board) > MAX_TRACKED_PLY:
            return REWEIGHT_INELIGIBLE
        return slot_from_x(x)

    def bump(self, key):
        if key is None or key == REWEIGHT_INELIGIBLE:
            return
        self.counts[key] += 1.0
        self.n_bumps += 1

    def weights(self, keys, records):
        """Per-record weights, plus the eligibility mask they were computed
        under. REWEIGHT_INELIGIBLE -> weight 1.0 and eligible False; a None key
        means the record arrived without one (reloaded carryover, seeder
        output) and is hashed from its own x here rather than escaping
        unweighted."""
        out = np.ones(len(keys), dtype=np.float32)
        elig = np.zeros(len(keys), dtype=bool)
        idx, slots = [], []
        for i, k in enumerate(keys):
            if k == REWEIGHT_INELIGIBLE:
                continue
            elig[i] = True
            idx.append(i)
            slots.append(slot_from_x(records[i][0]) if k is None else k)
        if idx:
            out[idx] = band_weights(self.counts[np.array(slots, dtype=np.int64)])
        return out, elig

    def decay(self, factor=COUNTS_DECAY_FACTOR):
        """Plain full-table multiply, ~16 ms at 1 << 25. Masking to nonzeros
        is 4.5x slower -- the gather costs more than the multiply saves."""
        self.counts *= factor
        self.n_decays += 1

    def occupancy(self):
        return int(np.count_nonzero(self.counts))

    def export_state(self):
        """Sparse snapshot, for the in-memory reload path."""
        nz = np.flatnonzero(self.counts)
        return {
            'bits': TABLE_BITS,
            'idx': nz.astype(np.int64),
            'vals': self.counts[nz].copy(),
            'n_bumps': self.n_bumps,
            'n_decays': self.n_decays,
        }

    def import_state(self, state):
        bits = int(state['bits'])
        if bits != TABLE_BITS:
            raise ValueError(
                f"opening counts built for TABLE_BITS={bits}, "
                f"this build uses {TABLE_BITS}")
        self.counts[:] = 0.0
        self.counts[np.asarray(state['idx'], dtype=np.int64)] = state['vals']
        self.n_bumps = int(state['n_bumps'])
        self.n_decays = int(state['n_decays'])

    def save(self, path=None):
        """Written sparse: 128 MiB of table is ~315k nonzeros, a couple of MB
        on disk, and compressing the dense array instead costs seconds."""
        path = path or self.path
        if path is None:
            raise ValueError("OpeningCounts.save needs a path")
        state = self.export_state()
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **state)
        os.replace(tmp, path)
        return path

    def load(self, path=None):
        """Restore from disk. Returns False when there is nothing to load, so
        a fresh run starts cold -- the table refills within a few thousand
        games and needs no seeding."""
        path = path or self.path
        if path is None or not os.path.exists(path):
            return False
        with np.load(path) as z:
            self.import_state({k: z[k] for k in z.files})
        return True


def reweight_and_rectify(records, w, elig=None):
    """Scale vwht and pwht by w on eligible records, then restore the eligible
    subset's own sums.

    Rectifying inside the eligible subset is what keeps an ineligible record
    genuinely untouched: it neither sheds mass nor absorbs mass shed by others,
    so it leaves carrying exactly the weights it arrived with. Total shard mass
    is preserved either way, since the eligible mass is restored in full.

    vwht and pwht are rectified independently against their own pre-weight sums;
    they do not share a scale (lc0 records already sit at pwht 0.5, blunder plies
    at 3.0), so one target cannot serve both.
    """
    if elig is None:
        elig = np.ones(len(records), dtype=bool)
    if not (elig & (w < 1.0)).any():
        return records

    v = np.array([r[4] for r in records], dtype=np.float64)
    p = np.array([r[5] for r in records], dtype=np.float64)
    v_target, p_target = v[elig].sum(), p[elig].sum()
    v[elig] *= w[elig]
    p[elig] *= w[elig]
    if v[elig].sum() > 0:
        v[elig] *= v_target / v[elig].sum()
    if p[elig].sum() > 0:
        p[elig] *= p_target / p[elig].sum()

    return [(r[0], r[1], r[2], r[3], float(v[i]), float(p[i]), r[6])
            for i, r in enumerate(records)]


def apply_repeat_weights(chunk, keys, counts):
    """Downweight repeated openings inside one shard, then restore the shard's
    mass. Records marked REWEIGHT_INELIGIBLE at append time pass through
    untouched.

    Rectifying per shard rather than globally is what makes the scheme compose:
    each shard leaves with exactly the mass it arrived with, so any subset of
    shards a retrain draws is rectified too, and shuffle_rewrite_primary stirring
    records between shards perturbs that only to ~0.1%.
    """
    w, elig = counts.weights(keys, chunk)
    return reweight_and_rectify(chunk, w, elig)


# Historic tfrec shards never pass through LiveBuffer, so they arrive at a flat
# 1.0 while selfplay openings have been suppressed -- which would leave historic
# carrying more startpos mass than primary despite being the smaller source.
# build_bootstrap_records already thins repeats by dropping them, so what is
# left needs only a light touch: measured on a real 8-shard draw, 99.67% of
# records sit at 1.00 and total mass shed is 0.12%.
HISTORIC_DUP_BOUNDS = np.array([2, 4, 10])
HISTORIC_DUP_WEIGHTS = np.array([1.00, 0.85, 0.66, 0.33], dtype=np.float32)


def historic_dup_weights(records):
    """Weight each record by how often its position repeats in this batch.

    Keyed on x[:64], the current-position frame of the xc0h encoding, which is
    the only position handle a tfrec record carries -- there is no board and no
    ply counter to reach for, so this cannot share the zobrist counter above.
    Counting within the loaded batch rather than globally keeps it self-
    contained and makes the rectification exact over the same set.
    """
    keys = [r[0][:64].tobytes() for r in records]
    counts = Counter(keys)
    n = np.array([counts[k] for k in keys])
    return HISTORIC_DUP_WEIGHTS[
        np.searchsorted(HISTORIC_DUP_BOUNDS, n, side='left')]


def apply_historic_dup_weights(records):
    """Light repeat-suppression for the historic draw, mass-preserving.

    Historic only. Replay is left alone deliberately: it mixes tfrec seeds with
    aged primary shards that already carry opening weights, so reweighting it
    would compound them, and its tfrec share ages out on its own.
    """
    return reweight_and_rectify(records, historic_dup_weights(records))
