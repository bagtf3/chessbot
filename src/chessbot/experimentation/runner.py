"""SF-label helpers and the xc0 inference server launcher shared by the
search-controller collector and trainer.

SfEvals pickled by those tools name this module, so it keeps its name.
"""
import os
import pickle
from typing import NamedTuple, Optional

import chess
import numpy as np
from pyfastchess import Board as fastboard

from chessbot.experimentation.corpus import CP_MAX
from chessbot.utils import cp_to_value_tanh

SERVER_WAIT_S = 300.0
GOOD_CP = 10
# a pick also counts as SF-equivalent within this expected score loss of the
# best, unless the best is a forced mate for the side to move
GOOD_ESL = 0.02


class SfEval(NamedTuple):
    """SF's eval of one move played from one position.

    Every value is from the point of view of the side to move in that
    position -- the player choosing the move -- so higher is always better
    for them and the loss of a pick is best.cp_stm - pick.cp_stm.
    """
    cp_stm: int                # clipped to +-CP_MAX; a forced mate is +-CP_MAX
    mate_stm: Optional[int]    # None, or moves to mate: +N they mate, -N they get mated
    depth: Optional[int]
    source: str                # "corpus" (rescore data) or "sf" (evaluated here)


def mate_sign(ev):
    """+1 / -1 when ev is a forced mate for / against the side to move, else
    0. Corpus evals record no mates, so a cp at +-CP_MAX counts as one."""
    if ev.mate_stm is not None:
        return 1 if ev.mate_stm > 0 else -1
    if abs(ev.cp_stm) >= CP_MAX:
        return 1 if ev.cp_stm > 0 else -1
    return 0


def sf_value(ev):
    """ev as a -1..1 value: a forced mate is exactly +-1, anything else goes
    through cp_to_value_tanh."""
    m = mate_sign(ev)
    return float(m) if m else float(cp_to_value_tanh(ev.cp_stm))


def is_equiv(best, ev):
    """Whether pick eval ev is as good as best: within GOOD_CP of it (or
    better), or within GOOD_ESL expected score loss on sf_value. A forced mate for the side to move only accepts the cp rule,
    so missing a mate is never equivalent."""
    if best.cp_stm - ev.cp_stm <= GOOD_CP:
        return True
    if mate_sign(best) > 0:
        return False
    return 0.5 * max(0.0, sf_value(best) - sf_value(ev)) <= GOOD_ESL


def sf_eval_from_score(score, depth):
    """SfEval from a python-chess PovScore of the position's side to move."""
    rel = score.relative
    mate = rel.mate()
    if mate is not None:
        cp = CP_MAX if mate > 0 else -CP_MAX
    else:
        cp = int(np.clip(rel.score(), -CP_MAX, CP_MAX))
    return SfEval(cp_stm=cp, mate_stm=mate, depth=depth, source="sf")


def position_boards(pos):
    """The position as a pyfastchess board (history for xc0h) and a
    python-chess board (repetitions for SF)."""
    fb = fastboard(pos["init_fen"])
    cb = chess.Board(pos["init_fen"])
    for uci in pos["uci_path"]:
        fb.push_uci(uci)
        cb.push_uci(uci)
    return fb, cb


def start_server(ctx, cfg, n_slots, n_threads):
    """One xc0 inference server on the run's current engine, with n_slots
    worker slots. Mirrors SelfplayRunner.start_xc0_server."""
    from chessbot.infer_ort_trt import selfplay_trt_paths
    from chessbot.infer_server import server_main

    trt_dir, model_name, onnx_path = selfplay_trt_paths(cfg)
    load = {"cmd": "load", "onnx": onnx_path, "model_name": model_name,
            "trt_dir": trt_dir}
    reqs = [ctx.Pipe(duplex=False) for _ in range(n_slots)]
    reps = [ctx.Pipe(duplex=False) for _ in range(n_slots)]
    ctrl_r, ctrl_w = ctx.Pipe(duplex=False)
    ready_q = ctx.Queue()
    ready_ev, release_ev = ctx.Event(), ctx.Event()
    spec = {"max_bs": cfg.macro_batch, "encoding": cfg.encoding_type,
            "n_threads": n_threads}
    p = ctx.Process(target=server_main,
                    args=(spec, load, [r for r, _ in reqs], [w for _, w in reps],
                          ctrl_r, ready_q, ready_ev, release_ev),
                    daemon=True)
    p.start()
    for r, _ in reqs:
        r.close()
    for _, w in reps:
        w.close()
    ctrl_r.close()

    descs = ready_q.get(timeout=SERVER_WAIT_S)
    slots = [{**d, "req": rq[1], "rep": rp[0]}
             for d, rq, rp in zip(descs, reqs, reps)]
    print(f"[exp] xc0 server up: {n_threads} sessions, {n_slots} slots, "
          f"{os.path.basename(onnx_path)}", flush=True)
    return p, ctrl_w, slots


class Unpickler(pickle.Unpickler):
    """Reads SfEvals pickled by a launcher that ran as __main__."""

    def find_class(self, module, name):
        if name == "SfEval":
            return SfEval
        return super().find_class(module, name)
