#!/usr/bin/env python3
"""
xerces.py — UCI interface for the Xerces chess engine.

Usage:
    python xerces.py [--model PATH] [--config PATH]

Communicates with a GUI via stdin/stdout using the UCI protocol.
Model: PyTorch checkpoint (load_pt_model / make_pt_infer).
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pyfastchess import Board, raw_cache_bulk_insert
from chessbot.mcts_utils import MCTSTree
from chessbot.train_pytorch import load_pt_model, make_pt_infer

ENGINE_NAME = "Xerces"
ENGINE_AUTHOR = "Bryan Goggin"


class _Cfg:
    """Minimal duck-typed config for MCTSTree."""
    c_puct = 2.0
    sims_floor = 400
    sims_ceiling = 800
    pruning_factor = 1.2
    uniform_eps = 0.05
    prior_clip_max = 0.65
    add_root_noise = False
    dirichlet_eps = 0.25
    dirichlet_alpha = 0.3
    reuse_tree = True
    use_u_attn = True
    sample_moves = False
    robust_only_above = 999999   # disable sampling; always most_visited
    es_check_every = 100
    min_top_visits = 300
    min_delta = 100
    use_robust = True
    es_jsd_thresh = 0.05
    es_jsd_n_stable = 3
    es_jsd_min_delta = 100
    jsd_min_sims = 600
    move_sample_temp_range = (0.000001, 1.25)


class XercesUCI:
    def __init__(self):
        # setoption-settable params (mirrors _Cfg defaults + model path)
        self.model_path = os.getenv("XERCES_MODEL", "")
        self.sims_floor = 400
        self.sims_ceiling = 800
        self.c_puct = 2.0
        self.micro_batch = 8
        self.uniform_eps = 0.05
        self.prior_clip_max = 0.65
        self.pruning_factor = 1.2
        self.vscale = 0.9
        self.use_u_attn = True

        # runtime state
        self.board: Board | None = None
        self.tree: MCTSTree | None = None
        self.infer_fn = None
        self._model_loaded = False

        # search thread control
        self._stop_event = threading.Event()
        self._search_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ I/O

    def send(self, line: str):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def info(self, msg: str):
        self.send(f"info string {msg}")

    # --------------------------------------------------------- Model loading

    def _load_model(self):
        if not self.model_path:
            self.info("WARNING: ModelPath not set; engine will not evaluate positions")
            self._model_loaded = True
            return
        self.info(f"loading model: {self.model_path}")
        model, arch = load_pt_model(self.model_path)
        _, self.infer_fn = make_pt_infer(model, max_bs=256, vscale=self.vscale)
        self.info(f"model ready  arch={arch}")
        self._model_loaded = True

    # --------------------------------------------------------- Tree creation

    def _make_cfg(self) -> _Cfg:
        cfg = _Cfg()
        cfg.c_puct = float(self.c_puct)
        cfg.sims_floor = int(self.sims_floor)
        cfg.sims_ceiling = int(self.sims_ceiling)
        cfg.pruning_factor = float(self.pruning_factor)
        cfg.uniform_eps = float(self.uniform_eps)
        cfg.prior_clip_max = float(self.prior_clip_max)
        cfg.use_u_attn = bool(self.use_u_attn)
        return cfg

    def _build_tree(self, board: Board):
        self.board = board
        self.tree = MCTSTree(board, self._make_cfg())

    # ---------------------------------------------------- UCI command handlers

    def handle_uci(self):
        self.send(f"id name {ENGINE_NAME}")
        self.send(f"id author {ENGINE_AUTHOR}")
        self.send("option name ModelPath type string default")
        self.send(f"option name SimsCeiling type spin default {self.sims_ceiling} min 1 max 1000000")
        self.send(f"option name SimsFloor   type spin default {self.sims_floor}   min 1 max 100000")
        self.send(f"option name CPuct       type string default {self.c_puct}")
        self.send(f"option name MicroBatch  type spin default {self.micro_batch}  min 1 max 512")
        self.send(f"option name Vscale      type string default {self.vscale}")
        self.send("uciok")

    def handle_isready(self):
        if not self._model_loaded:
            self._load_model()
        self.send("readyok")

    def handle_ucinewgame(self):
        self.tree = None
        self.board = None

    def handle_setoption(self, tokens: list[str]):
        # setoption name <name> value <value>
        try:
            ni = tokens.index("name")
            vi = tokens.index("value")
        except ValueError:
            return
        name = " ".join(tokens[ni + 1 : vi]).strip().lower().replace(" ", "")
        value = " ".join(tokens[vi + 1 :]).strip()

        _map = {
            "modelpath":    ("model_path",    str),
            "simscelling":  ("sims_ceiling",  int),
            "simsfloor":    ("sims_floor",    int),
            "cpuct":        ("c_puct",        float),
            "microbatch":   ("micro_batch",   int),
            "vscale":       ("vscale",        float),
            "uniformeps":   ("uniform_eps",   float),
            "priorclipmmax":("prior_clip_max",float),
        }
        if name in _map:
            attr, typ = _map[name]
            setattr(self, attr, typ(value))
            if name == "modelpath":
                self._model_loaded = False  # will reload on next isready

    def handle_position(self, tokens: list[str]):
        if not tokens:
            return

        startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        if tokens[0] == "startpos":
            fen = startpos_fen
            rest = tokens[1:]
        elif tokens[0] == "fen":
            try:
                mi = tokens.index("moves")
                fen = " ".join(tokens[1:mi])
                rest = tokens[mi:]
            except ValueError:
                fen = " ".join(tokens[1:])
                rest = []
        else:
            return

        move_list = rest[1:] if rest and rest[0] == "moves" else []

        board = Board(fen)
        for mv in move_list:
            board.push_uci(mv)

        self._build_tree(board)

    def handle_go(self, tokens: list[str]):
        if self.board is None or self.tree is None:
            self.send("bestmove 0000")
            return

        params: dict = {}
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

        time_budget_ms = self._compute_time_budget(params)

        if "nodes" in params:
            n = params["nodes"]
            self.tree.sims_ceiling = n
            self.tree.set_sim_budget(float(n))

        self._stop_event.clear()
        self._search_thread = threading.Thread(
            target=self._search_worker,
            args=(time_budget_ms, params.get("infinite", False)),
            daemon=True,
        )
        self._search_thread.start()

    def _compute_time_budget(self, params: dict) -> int | None:
        """Return milliseconds to search, or None if infinite."""
        if params.get("infinite"):
            return None
        if "movetime" in params:
            return max(50, params["movetime"] - 20)

        stm = self.board.side_to_move() if self.board else "w"
        my_time = params.get("wtime" if stm == "w" else "btime")
        if my_time is None:
            return None  # no time info → sim ceiling governs

        inc = params.get("winc" if stm == "w" else "binc", 0)
        movestogo = params.get("movestogo", 30)

        # 1/movestogo of remaining time, plus most of the increment
        budget = my_time / movestogo + inc * 0.8
        # never burn more than 1/4 of the clock; always at least 50 ms
        budget = max(50, min(budget, my_time / 4))
        return int(budget)

    def _search_worker(self, time_budget_ms: int | None, infinite: bool):
        tree = self.tree
        board = self.board
        infer = self.infer_fn

        micro_batch = self.micro_batch
        max_fastpath = max(micro_batch * 4, 64)
        max_unresolved = micro_batch * 4

        # reset early-stop state so go always starts fresh
        tree._es_tripped = False
        tree._es_last_checked_at = tree.sims_completed_this_move
        tree.sim_stop_reason = ""
        tree.es_checks.clear()

        t0 = time.time()
        deadline = (t0 + time_budget_ms / 1000.0) if time_budget_ms is not None else None

        def _over_time():
            return deadline is not None and time.time() >= deadline

        def _infer_and_cache(pending):
            boards_np = np.stack(
                [np.asarray(b, dtype=np.int32) for _, b in pending], axis=0
            )
            logits, vals = infer((boards_np,))
            entries = []
            for idx, (k, _) in enumerate(pending):
                v = float(np.asarray(vals[idx]).reshape(()))
                p = np.asarray(logits[idx], dtype=np.float32)
                entries.append((k, v, p))
            raw_cache_bulk_insert(entries)

        while not self._stop_event.is_set() and not _over_time():
            if tree.stop_simulating():
                break

            tree.resolve_inflight()

            if tree.count_unresolved() >= max_unresolved:
                time.sleep(0.0005)
                continue

            res = tree.collect_many_leaves(micro_batch, max_fastpath)
            n_leafs = res.count_new + res.count_terminal + res.count_cached
            tree.sims_completed_this_move += n_leafs

            if res.count_new > 0 and infer is not None:
                pending = tree.pending_encoded_64_tokens()
                if pending:
                    _infer_and_cache(pending)

        # drain any nodes still in flight
        for _ in range(200):
            if tree.count_unresolved() == 0:
                break
            pending = tree.pending_encoded_64_tokens()
            if pending and infer is not None:
                _infer_and_cache(pending)
            tree.resolve_inflight()
            time.sleep(0.001)

        elapsed = time.time() - t0
        sims = tree.sims_completed_this_move
        sps = sims / max(elapsed, 1e-9)

        mv, _method, _ = tree.best()
        if not mv:
            mv = "0000"

        # emit one info line
        stm = board.side_to_move()
        sign = 1 if stm == "w" else -1
        _, details = tree.robust_selection_criteria(5, 100)
        if details:
            q_stm = details[0].Q * sign
            cp = int(q_stm * 100)
            depth = max(1, tree.depth_stats()[1])
            self.send(
                f"info depth {depth} score cp {cp} nodes {sims} "
                f"nps {int(sps)} time {int(elapsed * 1000)} pv {mv}"
            )

        self.send(f"bestmove {mv}")

    def handle_stop(self):
        self._stop_event.set()
        if self._search_thread is not None:
            self._search_thread.join(timeout=2.0)
            self._search_thread = None

    # --------------------------------------------------------------- Main loop

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
                break
            elif cmd == "d":
                if self.board:
                    self.info(self.board.fen())


def main():
    parser = argparse.ArgumentParser(description="Xerces UCI chess engine")
    parser.add_argument("--model", help="PyTorch model checkpoint path")
    parser.add_argument("--config", help="YAML or JSON config file")
    args, _ = parser.parse_known_args()

    engine = XercesUCI()

    if args.model:
        engine.model_path = args.model

    if args.config:
        from chessbot.config import Config
        ext = os.path.splitext(args.config)[1].lower()
        cfg = Config.from_yaml(args.config) if ext in (".yaml", ".yml") else Config.from_json(args.config)
        engine.sims_floor = cfg.sims_floor
        engine.sims_ceiling = cfg.sims_ceiling
        engine.c_puct = cfg.c_puct
        engine.vscale = cfg.vscale
        engine.uniform_eps = cfg.uniform_eps
        engine.prior_clip_max = cfg.prior_clip_max
        engine.pruning_factor = cfg.pruning_factor

    engine.run()


if __name__ == "__main__":
    main()
