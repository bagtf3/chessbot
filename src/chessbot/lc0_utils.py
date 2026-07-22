# Board encoding adapted from lczero-tools (MIT license, abandoned ~2019).
# Original source: https://github.com/so-much-meta/lczero_tools
# UCI-to-index tables vendored into xerces_training (same origin).
# Credit: Leela Chess Zero contributors and the lczero_tools authors.

import collections
import os
import struct

import chess
import numpy as np

from xerces_training.uci_to_idx import IDX_TO_UCI, uci_to_idx as UCI_TO_IDX
from xerces_training.parse_utils import uci_to_xerces_index


def lc0_table_index(features_uint8):
    """Derive LC0->XC0 table index (0-3) from suffix planes of lc0_features() output."""
    us_ooo = int(features_uint8[104, 0, 0])
    us_oo  = int(features_uint8[105, 0, 0])
    stm    = int(features_uint8[108, 0, 0])  # 0=white, 1=black
    return (us_ooo | us_oo) + 2 * stm


HistData = collections.namedtuple(
    'HistData', 'piece_bytes rep us_oo us_ooo them_oo them_ooo stm rule50')
PACK_Q = struct.Struct('>Q').pack


class Lc0Board:
    """LC0-compatible board encoder producing (112, 8, 8) float32 feature planes.

    Maintains up to 8 steps of move history for repetition detection and
    the history planes used by LC0/t1-family models.
    Always encode from STM perspective (us = side to move).
    """

    def __init__(self, board: chess.Board = None):
        self.board = board.copy(stack=False) if board else chess.Board()
        self.hist  = []
        self.tctr  = collections.Counter()
        self.record()

    @classmethod
    def from_fen(cls, fen: str):
        return cls(chess.Board(fen))

    def push_uci(self, uci: str):
        self.board.push(chess.Move.from_uci(uci))
        self.record()

    def push(self, move: chess.Move):
        self.board.push(move)
        self.record()

    def record(self):
        b   = self.board
        key = b._transposition_key()
        self.tctr.update((key,))
        stm = b.turn
        pm  = b.pieces_mask
        raw = (
            b''.join(PACK_Q(pm(pt, chess.WHITE)) for pt in range(1, 7)) +
            b''.join(PACK_Q(pm(pt, chess.BLACK)) for pt in range(1, 7))
        )
        cr = b.castling_rights
        if stm:
            us_ooo, us_oo     = (cr >> chess.A1) & 1, (cr >> chess.H1) & 1
            them_ooo, them_oo = (cr >> chess.A8) & 1, (cr >> chess.H8) & 1
        else:
            us_ooo, us_oo     = (cr >> chess.A8) & 1, (cr >> chess.H8) & 1
            them_ooo, them_oo = (cr >> chess.A1) & 1, (cr >> chess.H1) & 1
        self.hist.append(HistData(
            piece_bytes=raw, rep=int(self.tctr[key] > 1),
            us_oo=us_oo, us_ooo=us_ooo,
            them_oo=them_oo, them_ooo=them_ooo,
            stm=0 if stm else 1,
            rule50=b.halfmove_clock,
        ))

    @staticmethod
    def decode_pieces(raw: bytes, is_black: bool) -> np.ndarray:
        bits   = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        planes = bits[::-1].reshape(12, 8, 8)[::-1].astype(np.float32)
        if is_black:
            planes = planes.reshape(2, 6, 8, 8)[::-1, :, ::-1].reshape(12, 8, 8).copy()
        return planes

    def features(self) -> np.ndarray:
        """Return (112, 8, 8) float32 suitable for LC0 model input."""
        cur      = self.hist[-1]
        is_black = bool(cur.stm)
        chunks   = []
        for h in self.hist[-1:-9:-1]:
            chunks.append(self.decode_pieces(h.piece_bytes, is_black))
            chunks.append(np.full((1, 8, 8), float(h.rep), dtype=np.float32))
        filled = min(len(self.hist), 8)
        if filled < 8:
            chunks.append(np.zeros(((8 - filled) * 13, 8, 8), dtype=np.float32))
        for val in (cur.us_ooo, cur.us_oo, cur.them_ooo, cur.them_oo, cur.stm):
            chunks.append(np.full((1, 8, 8), float(val), dtype=np.float32))
        chunks.append(np.full((1, 8, 8), cur.rule50 / 99.0, dtype=np.float32))
        chunks.append(np.zeros((1, 8, 8), dtype=np.float32))
        chunks.append(np.ones((1, 8, 8),  dtype=np.float32))
        return np.concatenate(chunks, axis=0)

    def policy_indices(self, moves=None) -> list:
        """UCI moves -> LC0 1858-dim policy indices. Defaults to legal moves."""
        if moves is None:
            moves = [m.uci() for m in self.board.legal_moves]
        cur = self.hist[-1]
        tbl = UCI_TO_IDX[(cur.us_ooo | cur.us_oo) + 2 * cur.stm]
        return [tbl[m.rstrip('n')] for m in moves]

    def table_index(self) -> int:
        """Index 0-3 selecting the correct IDX_TO_UCI / UCI_TO_IDX table."""
        cur = self.hist[-1]
        return (cur.us_ooo | cur.us_oo) + 2 * cur.stm


def short_fen(board: chess.Board) -> str:
    """4-field FEN (position, turn, castling, ep) — no clocks."""
    return ' '.join(board.fen().split()[:4])


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def lc0_logits_to_xc0_batch(logits, lc0_idx_list, xc0_idx_list):
    """Convert LC0 policy logits to XC0 1858-domain probability arrays.

    logits:       (N, 1858) float32 — raw policy logits from LC0/ORT model
    lc0_idx_list: list of N int arrays — LC0 indices for each position's legal moves
    xc0_idx_list: list of N int arrays — corresponding XC0 1858-domain indices

    Returns (N, 1858) float32 with softmaxed probabilities in XC0 slots.
    """
    n      = len(logits)
    result = np.zeros((n, 1858), dtype=np.float32)
    for i in range(n):
        lc0_idx = lc0_idx_list[i]
        xc0_idx = xc0_idx_list[i]
        if len(lc0_idx) == 0:
            continue
        raw   = logits[i, lc0_idx].astype(np.float64)
        e     = np.exp(raw - raw.max())
        probs = (e / e.sum()).astype(np.float32)
        result[i, xc0_idx] = probs
    return result


def make_lc0_infer(sess):
    """Wrap an LC0 ORT session into an infer(pair) callable.

    pair[0]: (B, 112, 8, 8) uint8 features from get_all_lc0_features().
    Returns (policy_logits[B,1858], wdl_probs[B,3]).
    Policy indices are in LC0 order (castling at rook slots, promos at lc0 promo tail).
    """
    input_name   = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    def infer(pair):
        feat_u8  = pair[0]                            # (B, 112, 8, 8) uint8
        feat_f32 = feat_u8.astype(np.float32)
        feat_f32[:, 109] /= 99.0                      # normalise rule50
        outs   = sess.run(output_names, {input_name: feat_f32})
        logits = outs[0].astype(np.float32)           # (B, 1858)
        wdl    = outs[1].astype(np.float32)           # (B, 3) already softmaxed
        return logits, wdl

    return infer


def make_lc0_trt_session(onnx_path: str, model_name: str, trt_cache: str,
                          opt_batch: int = 256, max_batch: int = 512):
    """Create an ORT session for an LC0 model with TRT EP.

    Always attempts TRT — compiles and caches a new engine if none exists for
    the sha256[:12] prefix of the ONNX source. First run for a new model will
    be slow while the engine compiles.
    """
    import hashlib
    import tensorrt
    import onnxruntime as ort

    prepped = os.path.join(trt_cache, f'{model_name}.onnx')
    src     = prepped if os.path.exists(prepped) else onnx_path
    h = hashlib.sha256()
    with open(src, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    prefix  = f'{model_name}_{h.hexdigest()[:12]}'
    engines = [f for f in os.listdir(trt_cache)
               if f.startswith(prefix) and f.endswith('.engine')]

    if engines:
        print(f'TRT engine: {engines[0]}')
    else:
        print(f'No TRT engine for prefix {prefix!r}, will compile now (this may take a few minutes)...')

    trt_opts = {
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path':   trt_cache,
        'trt_engine_cache_prefix': prefix,
        'trt_fp16_enable':         True,
        'trt_timing_cache_enable': True,
        'trt_timing_cache_path':   trt_cache,
        'trt_profile_min_shapes':  f'/input/planes:1x112x8x8',
        'trt_profile_opt_shapes':  f'/input/planes:{opt_batch}x112x8x8',
        'trt_profile_max_shapes':  f'/input/planes:{max_batch}x112x8x8',
    }
    ort.set_default_logger_severity(3)
    providers = [('TensorrtExecutionProvider', trt_opts),
                 'CUDAExecutionProvider', 'CPUExecutionProvider']
    sess   = ort.InferenceSession(src, providers=providers)
    active = sess.get_providers()[0]
    if active != 'TensorrtExecutionProvider':
        print(f'WARNING: expected TRT but got {active}')
    else:
        print(f'Provider: TensorrtExecutionProvider')
    return sess
