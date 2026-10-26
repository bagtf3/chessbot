#!/usr/bin/env python3
"""Xerces UCI chess engine."""

import argparse
import os
import sys
import threading
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from pyfastchess import Board
from chessbot.mcts_utils import MCTSTree
from chessbot.model import load_model, make_conv_infer
from chessbot.tf_thread import Batcher, TensorFlowThread

ENGINE_NAME = "Xerces"
ENGINE_AUTHOR = "Bryan Goggin"


def build_batch_candidates(min_batch, fwd_batch):
    sizes = {min_batch, fwd_batch}
    bs = min_batch
    while bs <= fwd_batch:
        sizes.add(bs)
        bs *= 2
    if len(sizes) >= 2 and fwd_batch >= 128:
        s = sorted(sizes)
        sizes.add(s[-2] + (s[-1] - s[-2]) // 2)
    return sorted(sizes)


class Cfg:
    """Duck-typed config consumed by MCTSTree."""
    c_puct = 2.0
    sims_floor = 400
    sims_ceiling = 800
    pruning_factor = 1.2
    uniform_eps = 0.05
    prior_clip_max = 0.65
    vscale = 0.9
    fpu_reduction = 0.1
    contempt_flip_q = 0.0
    contempt_fight_c = 0.0
    contempt_save_c = 0.0
    qema_span = 40
    qdelta_span = 100
    add_root_noise = False
    dirichlet_eps = 0.25
    dirichlet_alpha = 0.3
    reuse_tree = True
    sample_moves = False
    robust_only_above = 400
    es_check_every = 100
    min_top_visits = 300
    min_delta = 100
    use_robust = True
    es_jsd_thresh = 0.05
    es_jsd_n_stable = 3
    es_jsd_min_delta = 100
    jsd_min_sims = 600


class XercesUCI:
    def __init__(self):
        self.model_path = os.getenv("XERCES_MODEL", "")
        self.sims_floor = 400
        self.sims_ceiling = 800
        self.c_puct = 2.0
        self.micro_batch = 8
        self.fwd_batch = 32
        self.min_batch = 4
        self.max_inflight = 2
        self.uniform_eps = 0.05
        self.prior_clip_max = 0.65
        self.vscale = 0.9
        self.pruning_factor = 1.2
        self.fpu_reduction = 0.1

        self.board = None
        self.tree = None
        self.infer = None
        self.batcher = None
        self.tf_thread = None
        self.model_loaded = False

        self.stop_event = threading.Event()
        self.search_thread = None

    def send(self, line):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def uci_info(self, msg):
        self.send(f"info string {msg}")

    def setup_inference(self):
        if self.tf_thread is not None:
            self.tf_thread.close()
            self.tf_thread = None

        if not self.model_path:
            self.uci_info("WARNING: ModelPath not set")
            self.model_loaded = True
            return

        self.uci_info(f"loading model: {self.model_path}")
        model = load_model(self.model_path)
        self.infer = make_conv_infer(model, max_bs=self.fwd_batch, vscale=self.vscale)

        candidates = build_batch_candidates(self.min_batch, self.fwd_batch)
        self.batcher = Batcher(None, candidates)
        self.tf_thread = TensorFlowThread(None, self.infer, max_inflight=self.max_inflight)
        self.tf_thread.start()

        for bs in candidates:
            enc = (np.random.rand(bs, 64) * 18).astype(np.int32)
            self.infer((enc,))

        self.uci_info("model ready")
        self.model_loaded = True

    def make_tree_cfg(self):
        cfg = Cfg()
        cfg.c_puct = self.c_puct
        cfg.sims_floor = self.sims_floor
        cfg.sims_ceiling = self.sims_ceiling
        cfg.pruning_factor = self.pruning_factor
        cfg.uniform_eps = self.uniform_eps
        cfg.prior_clip_max = self.prior_clip_max
        cfg.vscale = self.vscale
        cfg.fpu_reduction = self.fpu_reduction
        cfg.robust_only_above = self.sims_floor
        return cfg

    def build_tree(self, board):
        self.board = board
        self.tree = MCTSTree(board, self.make_tree_cfg())

    def handle_uci(self):
        self.send(f"id name {ENGINE_NAME}")
        self.send(f"id author {ENGINE_AUTHOR}")
        self.send("option name ModelPath type string default")
        self.send(f"option name SimsCeiling type spin default {self.sims_ceiling} min 1 max 1000000")
        self.send(f"option name SimsFloor type spin default {self.sims_floor} min 1 max 100000")
        self.send(f"option name CPuct type string default {self.c_puct}")
        self.send(f"option name MicroBatch type spin default {self.micro_batch} min 1 max 512")
        self.send(f"option name FwdBatch type spin default {self.fwd_batch} min 1 max 512")
        self.send(f"option name Vscale type string default {self.vscale}")
        self.send(f"option name FpuReduction type string default {self.fpu_reduction}")
        self.send("uciok")

    def handle_isready(self):
        if not self.model_loaded:
            self.setup_inference()
        self.send("readyok")

    def handle_ucinewgame(self):
        self.tree = None
        self.board = None

    def handle_setoption(self, tokens):
        try:
            ni = tokens.index("name")
            vi = tokens.index("value")
        except ValueError:
            return
        name = "".join(tokens[ni + 1:vi]).lower()
        value = " ".join(tokens[vi + 1:]).strip()

        options = {
            "modelpath":    ("model_path",    str),
            "simscelling":  ("sims_ceiling",  int),
            "simsfloor":    ("sims_floor",    int),
            "cpuct":        ("c_puct",        float),
            "microbatch":   ("micro_batch",   int),
            "fwdbatch":     ("fwd_batch",     int),
            "vscale":       ("vscale",        float),
            "uniformeps":   ("uniform_eps",   float),
            "priorclipmax": ("prior_clip_max", float),
            "fpureduction": ("fpu_reduction", float),
        }
        if name not in options:
            return
        attr, typ = options[name]
        setattr(self, attr, typ(value))
        if name == "modelpath":
            self.model_loaded = False

    def handle_position(self, tokens):
        if not tokens:
            return
        startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        if tokens[0] == "startpos":
            fen, rest = startpos, tokens[1:]
        elif tokens[0] == "fen":
            try:
                mi = tokens.index("moves")
                fen, rest = " ".join(tokens[1:mi]), tokens[mi:]
            except ValueError:
                fen, rest = " ".join(tokens[1:]), []
        else:
            return
        moves = rest[1:] if rest and rest[0] == "moves" else []
        board = Board(fen)
        for mv in moves:
            board.push_uci(mv)
        self.build_tree(board)

    def handle_go(self, tokens):
        if self.board is None or self.tree is None:
            self.send("bestmove 0000")
            return

        params = {}
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in ("wtime", "btime", "winc", "binc", "movestogo",
                     "movetime", "depth", "nodes", "mate"):
                if i + 1 < len(tokens):
                    try:
                        params[t] = int(tokens[i + 1])
                    except ValueError:
                        pass
                    i += 2
                else:
                    i += 1
            elif t == "infinite":
                params["infinite"] = True
                i += 1
            else:
                i += 1

        budget_ms = self.compute_time_budget(params)
        if "nodes" in params:
            n = params["nodes"]
            self.tree.sims_ceiling = n
            self.tree.set_sim_budget(float(n))

        self.stop_event.clear()
        self.search_thread = threading.Thread(
            target=self.search_worker,
            args=(budget_ms,),
            daemon=True,
        )
        self.search_thread.start()

    def compute_time_budget(self, params):
        if params.get("infinite"):
            return None
        if "movetime" in params:
            return max(50, params["movetime"] - 20)
        stm = self.board.side_to_move() if self.board else "w"
        my_time = params.get("wtime" if stm == "w" else "btime")
        if my_time is None:
            return None
        inc = params.get("winc" if stm == "w" else "binc", 0)
        movestogo = params.get("movestogo", 30)
        budget = my_time / movestogo + inc * 0.8
        return int(max(50, min(budget, my_time / 4)))

    def emit_info(self, elapsed):
        tree = self.tree
        board = self.board
        if tree is None or board is None:
            return
        sims = tree.sims_completed_this_move
        sps = sims / max(elapsed, 1e-9)
        sign = 1 if board.side_to_move() == "w" else -1
        _, details = tree.robust_selection_criteria(5, 100)
        if not details:
            return
        cp = int(details[0].Q * sign * 100)
        depth = max(1, tree.depth_stats()[1])
        mv, _, _ = tree.best()
        self.send(
            f"info depth {depth} score cp {cp} nodes {sims}"
            f" nps {int(sps)} time {int(elapsed * 1000)} pv {mv or '0000'}"
        )

    def search_worker(self, budget_ms):
        tree = self.tree
        board = self.board
        batcher = self.batcher
        tf_thread = self.tf_thread

        micro_batch = self.micro_batch
        max_fastpath = max(micro_batch * 4, 64)
        max_unresolved = micro_batch * 4

        tree._es_tripped = False
        tree._es_last_checked_at = tree.sims_completed_this_move
        tree.sim_stop_reason = ""
        tree.es_checks.clear()

        t0 = time.time()
        deadline = (t0 + budget_ms / 1000) if budget_ms is not None else None
        last_info_t = t0

        while not self.stop_event.is_set():
            if tree.stop_simulating():
                break
            if deadline is not None and time.time() >= deadline:
                break

            tree.resolve_inflight()

            if tree.count_unresolved() >= max_unresolved:
                time.sleep(0.0005)
                continue

            res = tree.collect_many_leaves(micro_batch, max_fastpath)
            tree.sims_completed_this_move += (
                res.count_new + res.count_terminal + res.count_cached
            )

            if res.count_new > 0 and batcher is not None:
                batcher.submit(tree.pending_encoded_64_tokens())

            if batcher is not None and tf_thread is not None:
                batch = batcher.pop_batch(never_pad=True, patience=3)
                if batch is not None:
                    tf_thread.submit(batch)

            now = time.time()
            if now - last_info_t >= 1.0:
                self.emit_info(now - t0)
                last_info_t = now

        for _ in range(200):
            if tree.count_unresolved() == 0:
                break
            tree.resolve_inflight()
            time.sleep(0.001)

        elapsed = time.time() - t0
        self.emit_info(elapsed)
        mv, _, _ = tree.best()
        self.send(f"bestmove {mv or '0000'}")

    def handle_stop(self):
        self.stop_event.set()
        if self.search_thread is not None:
            self.search_thread.join(timeout=2.0)
            self.search_thread = None

    def run(self):
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            cmd = tokens[0]
            if cmd == "uci":
                self.handle_uci()
            elif cmd == "isready":
                self.handle_isready()
            elif cmd == "ucinewgame":
                self.handle_ucinewgame()
            elif cmd == "setoption":
                self.handle_setoption(tokens[1:])
            elif cmd == "position":
                self.handle_position(tokens[1:])
            elif cmd == "go":
                self.handle_go(tokens[1:])
            elif cmd == "stop":
                self.handle_stop()
            elif cmd == "quit":
                self.handle_stop()
                if self.tf_thread is not None:
                    self.tf_thread.close()
                break
            elif cmd == "d":
                if self.board:
                    self.uci_info(self.board.fen())

        if self.search_thread is not None:
            self.search_thread.join(timeout=30.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--config")
    args, _ = parser.parse_known_args()

    engine = XercesUCI()

    if args.model:
        engine.model_path = args.model

    if args.config:
        from chessbot.config import Config
        ext = os.path.splitext(args.config)[1].lower()
        cfg = (Config.from_yaml(args.config) if ext in (".yaml", ".yml")
               else Config.from_json(args.config))
        for attr in ("sims_floor", "sims_ceiling", "c_puct", "vscale",
                     "uniform_eps", "prior_clip_max", "pruning_factor", "fpu_reduction"):
            if hasattr(cfg, attr):
                setattr(engine, attr, getattr(cfg, attr))

    engine.run()


if __name__ == "__main__":
    main()
