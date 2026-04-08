import os, time
import numpy as np

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

from pyfastchess import Board
from chessbot.mcts_utils import MCTSTree
from chessbot.model import load_model, make_conv_infer
from chessbot.tf_thread import Batcher, TensorFlowThread


class EngineConfig:
    model_path = ""

    # MCTS
    c_puct = 2.5
    sims_floor = 1000
    sims_ceiling = 1600
    pruning_factor = 1.2

    # early stop
    es_check_every = 200
    min_top_visits = 400
    min_delta = 300
    use_robust = True
    robust_only_above = 1000
    es_jsd_thresh = 0.05
    es_jsd_n_stable = 3
    es_jsd_min_delta = 100

    # inference / batching
    micro_batch = 8
    fwd_batch = 32
    min_batch = 1
    max_inflight = 2
    max_unresolved = 16
    
    # policy shaping (must match what model was trained with)
    uniform_eps = 0.1
    prior_clip_max = 0.3
    vscale = 0.9
    
    add_root_noise = False
    sample_moves = False
    
    def setoption(self, name, value):
        if not hasattr(self, name):
            raise AttributeError(f"unknown option: {name}")
        setattr(self, name, value)


def batch_candidates(min_batch, fwd_batch):
    sizes = {min_batch, fwd_batch}
    bs = min_batch
    while bs <= fwd_batch:
        sizes.add(bs)
        bs *= 2
    return sorted(sizes)


class XercesEngine:
    def __init__(self, config):
        self.config = config
        self.tree = None
        self.board = None
        self.setup_inference()

    def setup_inference(self):
        cfg = self.config
        print(f"[xerces] loading model: {cfg.model_path}")
        self.model = load_model(cfg.model_path)
        self.infer = make_conv_infer(
            self.model,
            max_bs=cfg.fwd_batch,
            uniform_eps=cfg.uniform_eps,
            prior_clip_max=cfg.prior_clip_max,
            vscale=cfg.vscale,
        )
        candidates = batch_candidates(cfg.min_batch, cfg.fwd_batch)
        self.batcher = Batcher(cfg, candidates)
        self.tf_thread = TensorFlowThread(
            cfg, self.infer, max_inflight=cfg.max_inflight
        )
        self.tf_thread.start()
        self.warmup(candidates)

    def warmup(self, candidates):
        print("[xerces] warming up GPU...")
        for bs in candidates:
            for _ in range(3):
                enc = (np.random.rand(bs, 64) * 18).astype(np.int32)
                mask = (np.random.rand(bs, 4288) < 0.02).astype(np.int32)
                self.infer((enc, mask))
        print("[xerces] ready")

    def setoption(self, name, value):
        self.config.setoption(name, value)

    def set_position(self, fen, moves=None):
        board = Board(fen)
        if moves:
            for mv in moves:
                board.push_uci(mv)
        self.board = board
        self.tree = MCTSTree(board, self.config)

    def go(self, sims=None, keep_tree=True):
        if self.tree is None:
            raise RuntimeError("call set_position first")

        cfg = self.config
        tree = self.tree

        if sims is not None:
            tree.sims_ceiling = sims
            tree.set_sim_budget(float(sims))

        # reset early stop state; use current sim count as last-checked baseline
        # so we don't immediately re-trip on a resumed search
        tree._es_tripped = False
        tree._es_last_checked_at = tree.sims_completed_this_move
        tree.sim_stop_reason = ""
        tree.es_checks.clear()

        batcher = self.batcher
        tf_thread = self.tf_thread
        micro_batch = cfg.micro_batch
        max_fastpath = max(micro_batch * 4, 64)
        max_unresolved = cfg.max_unresolved

        t0 = time.time()

        while not tree.stop_simulating():
            tree.resolve_inflight()

            if tree.count_unresolved() >= max_unresolved:
                time.sleep(0.0005)
                continue

            res = tree.collect_many_leaves(micro_batch, max_fastpath)
            n_new = res.count_new
            n_leafs = n_new + res.count_terminal + res.count_cached
            tree.sims_completed_this_move += n_leafs

            if n_new > 0:
                pending = tree.pending_encoded_64_tokens()
                batcher.submit(pending)

            batch = batcher.pop_batch(never_pad=True)
            if batch is not None:
                tf_thread.submit(batch)

        for _ in range(100):
            if tree.count_unresolved() == 0:
                break
            tree.resolve_inflight()
            time.sleep(0.001)

        elapsed = time.time() - t0
        self.print_results(elapsed)

        if not keep_tree:
            self.tree = None
            self.board = None

    def print_results(self, elapsed):
        tree = self.tree
        sims = tree.sims_completed_this_move
        sps = round(sims / max(elapsed, 1e-9), 2)

        mv, method, _ = tree.best()
        rsc_map, details = tree.robust_selection_criteria(5, 100)
        rsc_map = rsc_map or {}

        white_to_move = self.board.side_to_move() == 'w'
        sign = 1 if white_to_move else -1

        stop = tree.sim_stop_reason or "ceiling"
        stm_label = "white" if white_to_move else "black"

        print(f"\nfen: {self.board.fen()}")
        print(
            f"sims: {sims}  sims/s: {sps:.2f}  "
            f"method: {method}  stop: {stop}  time: {elapsed:.2f}s  "
            f"stm: {stm_label}"
        )

        if details:
            q_white = details[0].Q
            q_stm = q_white * sign
            vwq = tree.visit_weighted_Q()
            print(f"Q(stm): {q_stm:+.3f}  Q(white): {q_white:+.3f}  Q(vw): {vwq:+.3f}")

        total_n = sum([d.N for d in details]) if details else 1

        header = (
            f"  {'move':<6}  {'san':<8}  {'N':>6}  {'N%':>6}"
            f"  {'P':>7}  {'Q(stm)':>8}  {'RSC':>7}"
        )
        print(f"\n{header}")
        for d in (details or [])[:10]:
            pct = 100.0 * d.N / max(total_n, 1)
            q_stm = d.Q * sign
            rsc_v = rsc_map.get(d.uci, 0.0)
            san = self.board.san(d.uci)
            print(
                f"  {d.uci:<6}  {san:<8}  {d.N:>6}  {pct:>5.1f}%"
                f"  {d.prior:>7.3f}  {q_stm:>+8.3f}  {rsc_v:>7.3f}"
            )

        best_san = self.board.san(mv) if mv else ""
        print(f"\nbest: {mv}  ({best_san})\n")


# --- usage example ---

MODEL_PATH = os.getenv("SP_DIR", "") + "cfiw_16m_low_sims1/cfiw_16m_low_sims1_model.h5"

cfg = EngineConfig()
cfg.model_path = MODEL_PATH

xerces = XercesEngine(cfg)
#%%
START = "2k5/2p3R1/ppn5/3N1p2/1P1Pp3/2P1P3/P4q1r/2K3R1 w - - 1 2"
xerces.set_position(START)
xerces.go(6400)
# xerces.go(3200)   # resume same tree with higher ceiling
# xerces.set_position(START, moves=["e2e4", "e7e5"])
# xerces.go(1600, keep_tree=False)
