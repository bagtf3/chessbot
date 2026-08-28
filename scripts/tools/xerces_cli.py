#!/usr/bin/env python3
"""
scripts/xerces_cli.py  --  Xerces engine interactive analysis demo.

Launches a browser board window with live candidate-move arrows.
Telemetry and engine comparisons print in the terminal.

Usage:
    python scripts/xerces_cli.py
    python scripts/xerces_cli.py --model path/to/model.h5
    python scripts/xerces_cli.py --config path/to/config.yaml
    python scripts/xerces_cli.py --sims 1600 --port 8765
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import chess
import chess.engine
import chess.svg
import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from pyfastchess import Board, priors_cache_clear
from chessbot.mcts_utils import MCTSTree
from chessbot.model import load_model, make_conv_infer
from chessbot.tf_thread import Batcher, TensorFlowThread
from chessbot.engines import lc0_analyze
from chessbot.utils import random_init

SF_LOC = os.getenv("SF_LOC", "")
LC0_LOC = os.getenv("LC0_LOC", "")

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
GRAY   = "\033[90m"

ARROW_COLORS = ["#00c853", "#2196f3", "#ff6d00", "#9c27b0", "#607d8b"]


class Cfg:
    """Duck-typed config consumed by MCTSTree."""
    c_puct           = 2.0
    sims_floor       = 0
    sims_ceiling     = 800
    pruning_factor   = 1.2
    uniform_eps      = 0.05
    prior_clip_max   = 0.65
    vscale           = 0.9
    fpu_reduction    = 0.1
    contempt_zero_q  = 0.0
    contempt_full_q  = 0.5
    contempt_fight_c = 0.0
    qema_span        = 40
    qdelta_span      = 100
    add_root_noise   = False
    dirichlet_eps    = 0.25
    dirichlet_alpha  = 0.3
    reuse_tree       = True
    sample_moves     = False
    use_robust       = True
    es_check_every   = 100
    es_tier1_consec       = 3
    es_tier1_jsd_thresh   = 0.005
    es_tier2_consec       = 5
    es_tier2_jsd_thresh   = 0.0003


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


def q_color(q):
    if q > 0.03:
        return GREEN
    if q < -0.03:
        return RED
    return YELLOW


def make_bar(frac, width=10, filled="=", empty="."):
    n = max(0, min(width, int(round(frac * width))))
    return "[" + filled * n + empty * (width - n) + "]"


def eval_bar_str(q_white, width=30):
    mid = width // 2
    frac = (q_white + 1.0) / 2.0
    pos = int(round(frac * width))
    bar = list("." * width)
    for i in range(min(pos, width)):
        bar[i] = "="
    bar_str = "".join(bar)
    if mid < len(bar_str):
        bar_str = bar_str[:mid] + "|" + bar_str[mid + 1:]
    return f"Black [{bar_str}] White"


def make_board_svg(fen, top_moves, flipped=False, size=480):
    cboard = chess.Board(fen)
    arrows = []
    max_n = top_moves[0]["N"] if top_moves else 1

    for i, m in enumerate(reversed(top_moves[:5])):
        rank = len(top_moves[:5]) - 1 - i
        try:
            move = chess.Move.from_uci(m["uci"])
        except Exception:
            continue
        share_ratio = m["N"] / max(max_n, 1)
        alpha = max(0.25, share_ratio)
        hex_col = ARROW_COLORS[rank]
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        color = f"rgba({r},{g},{b},{alpha:.2f})"
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=color))

    return chess.svg.board(cboard, arrows=arrows, flipped=flipped, size=size)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Xerces Engine</title>
<style>
  :root{--bg:#1a1a2e;--panel:#16213e;--accent:#4fc3f7;--text:#e0e0e0}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Consolas',monospace;padding:20px}
  h1{color:var(--accent);font-size:1.3em;margin-bottom:16px;letter-spacing:2px}
  .layout{display:flex;gap:28px;align-items:flex-start;flex-wrap:wrap}
  .board-wrap svg{display:block;width:480px;height:480px}
  .info{flex:1;min-width:300px}
  .section{background:var(--panel);border-radius:8px;padding:14px;margin-bottom:12px}
  .label{color:#888;font-size:.72em;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
  .status-line{font-size:1em;margin-bottom:4px}
  .speed-line{color:#888;font-size:.78em;margin-top:4px}
  .eval-bar{height:18px;background:#2a2a2a;border-radius:4px;overflow:hidden;position:relative;margin:8px 0}
  .eval-fill{position:absolute;right:0;top:0;bottom:0;background:#f0d9b5;transition:width .3s}
  .eval-labels{font-size:.78em;display:flex;justify-content:space-between;color:#888}
  .eval-q{margin-top:6px;font-size:.9em}
  .move-row{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:.88em}
  .move-num{color:#888;width:14px;text-align:right;flex-shrink:0}
  .move-san{width:52px;font-weight:bold;flex-shrink:0}
  .bar-wrap{flex:1;background:#2a2a2a;height:11px;border-radius:3px;overflow:hidden}
  .bar-fill{height:100%;border-radius:3px;transition:width .3s}
  .move-pct{width:44px;text-align:right;color:#888;font-size:.8em;flex-shrink:0}
  .move-q{width:56px;text-align:right;flex-shrink:0}
  .positive{color:#81c784}.negative{color:#e57373}.neutral{color:#ffd54f}
  .engine-tag{color:var(--accent);margin-right:6px;font-weight:bold}
  .cmp-row{font-size:.85em;margin:5px 0}
  #dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
  .dot-on{background:#ffd54f;animation:pulse 1s infinite}
  .dot-off{background:#81c784}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
</style>
</head>
<body>
<h1>XERCES ENGINE</h1>
<div class="layout">
  <div class="board-wrap"><div id="board">Loading...</div></div>
  <div class="info">
    <div class="section">
      <div class="label">Status</div>
      <div class="status-line"><span id="dot"></span><span id="status-txt">--</span></div>
      <div id="side-txt" style="color:#888;font-size:.85em;margin-top:2px"></div>
      <div class="speed-line" id="speed-txt"></div>
    </div>
    <div class="section">
      <div class="label">Evaluation</div>
      <div id="eval-wrap">--</div>
    </div>
    <div class="section">
      <div class="label">Top Moves</div>
      <div id="moves-wrap">--</div>
    </div>
    <div class="section" id="cmp-section" style="display:none">
      <div class="label">Engine Comparison</div>
      <div id="cmp-wrap"></div>
    </div>
  </div>
</div>
<script>
const COLORS=["#00c853","#2196f3","#ff6d00","#9c27b0","#607d8b"];
function qcls(q){return q>.03?'positive':q<-.03?'negative':'neutral'}
function qstr(q){return(q>=0?'+':'')+q.toFixed(3)}

function render(d){
  const dot=document.getElementById('dot');
  const stxt=document.getElementById('status-txt');
  if(d.status==='analyzing'){
    dot.className='dot-on';
    const pct=d.sims_target>0?Math.round(100*d.sims/d.sims_target):0;
    stxt.textContent=`Analyzing...  ${d.sims.toLocaleString()} / ${d.sims_target.toLocaleString()} sims  (${pct}%)`;
  } else {
    dot.className='dot-off';
    stxt.textContent=`Done  —  ${d.sims.toLocaleString()} sims  in  ${(d.elapsed||0).toFixed(1)}s`;
  }
  const side=d.to_move==='White'?'White to move':'Black to move';
  document.getElementById('side-txt').textContent=side+(d.stop_reason?'  |  stop: '+d.stop_reason:'');
  document.getElementById('speed-txt').textContent=
    d.sims_per_sec>0?`${Math.round(d.sims_per_sec).toLocaleString()} sims/s  |  ${Math.round(d.preds_per_sec).toLocaleString()} preds/s  |  depth avg=${(d.depth_avg||0).toFixed(1)} max=${d.depth_max||0}`:'';

  if(d.q_white!=null){
    const pct=Math.round(100*(d.q_white+1)/2);
    const wdl=d.wdl?`WDL&nbsp;&nbsp;${Math.round(d.wdl[0]*100)}% W / ${Math.round(d.wdl[1]*100)}% D / ${Math.round(d.wdl[2]*100)}% L`:'';
    document.getElementById('eval-wrap').innerHTML=
      `<div class="eval-bar"><div class="eval-fill" style="width:${pct}%"></div></div>
       <div class="eval-labels"><span>Black</span><span>White</span></div>
       <div class="eval-q">Q(white) <span class="${qcls(d.q_white)}">${qstr(d.q_white)}</span>&nbsp;&nbsp;${wdl}</div>`;
  } else {
    document.getElementById('eval-wrap').textContent='--';
  }

  if(d.top_moves&&d.top_moves.length){
    const maxN=d.top_moves[0].N||1;
    document.getElementById('moves-wrap').innerHTML=d.top_moves.map((m,i)=>{
      const bp=Math.round(100*m.N/maxN);
      return `<div class="move-row">
        <div class="move-num">${i+1}</div>
        <div class="move-san">${m.san}</div>
        <div class="bar-wrap"><div class="bar-fill" style="width:${bp}%;background:${COLORS[i]}"></div></div>
        <div class="move-pct">${m.share_pct.toFixed(1)}%</div>
        <div class="move-q ${qcls(m.Q)}">${qstr(m.Q)}</div>
      </div>`;
    }).join('');
  } else {
    document.getElementById('moves-wrap').textContent='--';
  }

  const cmpSec=document.getElementById('cmp-section');
  const cmpWrap=document.getElementById('cmp-wrap');
  if((d.sf_moves&&d.sf_moves.length)||(d.lc0_moves&&d.lc0_moves.length)){
    cmpSec.style.display='';
    let html='';
    if(d.sf_moves&&d.sf_moves.length){
      html+=`<div class="cmp-row"><span class="engine-tag">SF</span>`+
        d.sf_moves.map(m=>`${m.san}<span class="${qcls(m.cp/300)}"> (${m.cp>=0?'+':''}${m.cp}cp)</span>`).join('  ')+'</div>';
    }
    if(d.lc0_moves&&d.lc0_moves.length){
      html+=`<div class="cmp-row"><span class="engine-tag">LC0</span>`+
        d.lc0_moves.map(m=>`${m.san}<span class="${qcls(m.Q)}"> (${qstr(m.Q)})</span>`).join('  ')+'</div>';
    }
    cmpWrap.innerHTML=html;
  } else {
    cmpSec.style.display='none';
  }
}

function tick(){
  fetch('/board.svg?t='+Date.now()).then(r=>r.text()).then(s=>{document.getElementById('board').innerHTML=s}).catch(()=>{});
  fetch('/info.json?t='+Date.now()).then(r=>r.json()).then(render).catch(()=>{});
}
setInterval(tick,500);
tick();
</script>
</body>
</html>"""


class AnalysisState:
    def __init__(self):
        self.lock = threading.Lock()
        self.svg = chess.svg.board(chess.Board(), size=480)
        self.info = {"status": "idle", "sims": 0, "sims_target": 0}

    def update(self, svg=None, info=None):
        with self.lock:
            if svg is not None:
                self.svg = svg
            if info is not None:
                self.info = info

    def get_svg(self):
        with self.lock:
            return self.svg

    def get_info(self):
        with self.lock:
            return dict(self.info)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                body = HTML_TEMPLATE.encode("utf-8")
                self.respond(200, "text/html; charset=utf-8", body)
            elif path == "/board.svg":
                body = state.get_svg().encode("utf-8")
                self.respond(200, "image/svg+xml", body)
            elif path == "/info.json":
                body = json.dumps(state.get_info()).encode("utf-8")
                self.respond(200, "application/json", body)
            else:
                self.respond(404, "text/plain", b"not found")

        def respond(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


TEST_POSITIONS = [
    "r4rk1/ppN3pp/4p3/3p2q1/4n3/4PQ1P/P5P1/2R2RK1 w - - 1 2",
    "8/8/8/6pk/5p2/5P2/4K3/8 b - - 1 3",
    "r6r/p3R1pp/n1p5/1p1p4/k7/2BB3P/2P2P2/5K2 w - - 1 2",
    "q2k1b2/2r1ppp1/3p4/1B3r2/5B2/8/PPPQ1P2/2K4R w - - 2 2",
    "8/8/5R2/p3k1K1/2r3P1/8/5P1P/8 b - - 1 1",
]


class XercesCLI:
    def __init__(self, model_path, cfg_overrides=None, sims=6400, time_budget=float("inf"),
                 port=8765, micro_batch=64, fwd_batch=128, min_batch=4):
        self.sims = sims
        self.time_budget = time_budget
        self.fen = chess.STARTING_FEN
        self.stop_event = threading.Event()

        self.micro_batch = micro_batch
        self.fwd_batch = fwd_batch
        self.min_batch = min_batch

        self.sf_moves = []
        self.lc0_moves = []
        self.last_info = {}
        self.total_preds = 0
        self.t_analysis_start = 0.0

        self.state = AnalysisState()
        self.cfg_overrides = cfg_overrides or {}

        self._start_server(port)
        self._load_model(model_path)
        webbrowser.open(f"http://localhost:{port}/")

    @property
    def stm_flipped(self):
        return chess.Board(self.fen).turn == chess.BLACK

    def _start_server(self, port):
        handler = make_handler(self.state)
        server = HTTPServer(("localhost", port), handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"  Board display: http://localhost:{port}/")

    def _load_model(self, model_path):
        print(f"  Loading model: {model_path}")
        model = load_model(model_path)
        self.infer = make_conv_infer(model, max_bs=self.fwd_batch, vscale=0.9)

        candidates = build_batch_candidates(self.min_batch, self.fwd_batch)
        self.batcher = Batcher(None, candidates)
        self.tf_thread = TensorFlowThread(None, self.infer, max_inflight=2)
        self.tf_thread.start()

        print("  Warming up XLA...")
        for bs in candidates:
            enc = (np.random.rand(bs, 64) * 18).astype(np.int32)
            self.infer((enc,))
        print("  Model ready.\n")

    def make_cfg(self, sims):
        cfg = Cfg()
        cfg.sims_ceiling = sims
        for k, v in self.cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def get_top_moves(self, tree, cboard, n=5):
        _, details = tree.robust_selection_criteria(n + 2, 1)
        if not details:
            return []
        total = max(1, sum(d.N for d in details))
        sign = 1 if cboard.turn == chess.WHITE else -1
        rows = []
        for d in details[:n]:
            if d.N == 0:
                break
            try:
                san = cboard.san(chess.Move.from_uci(d.uci))
            except Exception:
                san = d.uci
            rows.append({
                "uci": d.uci,
                "san": san,
                "N": d.N,
                "share": d.N / total,
                "share_pct": 100.0 * d.N / total,
                "Q": float(d.Q) * sign,
                "P": float(d.prior),
            })
        return rows

    def update_display(self, tree, sims, elapsed, status="analyzing"):
        cboard = chess.Board(self.fen)
        top_moves = self.get_top_moves(tree, cboard)

        q_white = None
        wdl = None
        depth_avg, depth_max = 0.0, 0
        try:
            root = tree.root()
            if root.is_expanded:
                q_white = float(root.Q)
            depth_avg, depth_max = tree.depth_stats()
        except Exception:
            pass

        _, details = tree.robust_selection_criteria(1, 1)
        if details:
            d0 = details[0]
            if cboard.turn == chess.WHITE:
                wdl = [float(d0.win), float(d0.draw), float(d0.loss)]
            else:
                wdl = [float(d0.loss), float(d0.draw), float(d0.win)]

        sps = sims / max(elapsed, 1e-6)
        pps = self.total_preds / max(elapsed, 1e-6)

        svg = make_board_svg(self.fen, top_moves, flipped=self.stm_flipped)
        info = {
            "status": status,
            "sims": sims,
            "sims_target": self.sims,
            "elapsed": elapsed,
            "sims_per_sec": sps,
            "preds_per_sec": pps,
            "to_move": "White" if cboard.turn == chess.WHITE else "Black",
            "stop_reason": tree.sim_stop_reason,
            "q_white": q_white,
            "wdl": wdl,
            "top_moves": top_moves,
            "sf_moves": self.sf_moves,
            "lc0_moves": self.lc0_moves,
            "depth_avg": float(depth_avg),
            "depth_max": int(depth_max),
        }
        self.state.update(svg=svg, info=info)
        self.last_info = info
        self._print_status(top_moves, sims, elapsed, sps, pps, q_white, status,
                           depth_avg, depth_max)

    def _print_status(self, top_moves, sims, elapsed, sps, pps, q_white, status,
                      depth_avg, depth_max):
        os.system("cls" if os.name == "nt" else "clear")
        print(f"\n  {BOLD}XERCES ENGINE{RESET}  |  {self.fen[:40]}...")

        cboard = chess.Board(self.fen)
        side = "White" if cboard.turn == chess.WHITE else "Black"
        q_str = f"{q_white:+.3f}" if q_white is not None else "  ---"
        print(f"  {side} to move   Q(white) = {q_color(q_white or 0)}{q_str}{RESET}")

        if q_white is not None:
            print(f"  {eval_bar_str(q_white)}")

        print()

        if status == "analyzing":
            pct = 100 * sims / max(1, self.sims)
            print(
                f"  {CYAN}[analyzing]{RESET}  {sims:,}/{self.sims:,} ({pct:.0f}%)  |  "
                f"{elapsed:.1f}s  |  {sps:.0f} sims/s  |  {pps:.0f} preds/s"
            )
        else:
            print(
                f"  {GREEN}[done]{RESET}  {sims:,} sims  in  {elapsed:.1f}s  |  "
                f"{sps:.0f} sims/s  |  {pps:.0f} preds/s"
            )
        print(f"  depth avg={depth_avg:.1f}  max={depth_max}")
        print()

        if top_moves:
            max_n = top_moves[0]["N"] or 1
            print(f"  {'#':<3} {'SAN':<7} {'N':>6}  {'share':>6}  {'[bar      ]':^12}  {'Q(stm)':>7}  {'P':>6}")
            print("  " + "-" * 62)
            for i, m in enumerate(top_moves, 1):
                frac = m["N"] / max_n
                bar = make_bar(frac, width=10)
                qc = q_color(m["Q"])
                print(
                    f"  {i:<3} {m['san']:<7} {m['N']:>6}  {m['share']:>5.1%}  {bar}  "
                    f"{qc}{m['Q']:>+7.3f}{RESET}  {m['P']:>6.3f}"
                )
        else:
            print("  (no moves yet)")

        if self.sf_moves or self.lc0_moves:
            print()
            if self.sf_moves:
                parts = "  ".join(
                    f"{m['san']}({m['cp']:+d}cp)" for m in self.sf_moves[:3]
                )
                print(f"  {CYAN}SF {RESET}  {parts}")
            if self.lc0_moves:
                parts = "  ".join(
                    f"{m['san']}({m['Q']:+.3f})" for m in self.lc0_moves[:3]
                )
                print(f"  {CYAN}LC0{RESET}  {parts}")

        print()

    REBUILD_CHECKPOINTS = [500, 2000]

    def _run_sims(self, tree, board, target_sims, t0, time_budget):
        """Run sims on tree until target_sims or time_budget. Returns when done or stop_event set."""
        mb = self.micro_batch
        max_fp = max(mb * 4, 1000)
        max_unresolved = mb * 4
        last_update = time.time()
        update_interval = 0.4

        while not self.stop_event.is_set():
            elapsed = time.time() - t0
            if tree.sims_completed_this_move >= target_sims or elapsed >= time_budget:
                break
            tree.resolve_inflight()
            if tree.count_unresolved() >= max_unresolved:
                time.sleep(0.0005)
                continue

            res = tree.collect_many_leaves(mb, max_fp)
            tree.sims_completed_this_move += (
                res.count_new + res.count_terminal + res.count_cached
            )

            if res.count_new > 0:
                self.batcher.submit(tree.pending_encoded_64_tokens())
                self.total_preds += res.count_new

            batch = self.batcher.pop_batch(never_pad=True, patience=3)
            if batch is not None:
                self.tf_thread.submit(batch)

            now = time.time()
            if now - last_update >= update_interval:
                self.update_display(tree, tree.sims_completed_this_move, now - t0, status="analyzing")
                last_update = now

        for _ in range(200):
            if tree.count_unresolved() == 0:
                break
            tree.resolve_inflight()
            time.sleep(0.001)

    def _fresh_tree(self, board, sims_budget):
        cfg = self.make_cfg(sims_budget)
        tree = MCTSTree(board, cfg)
        tree.sim_stop_reason = ""
        return tree

    def do_analyze(self, sims_override=None, time_override=None):
        sims_budget = sims_override if sims_override is not None else self.sims
        time_budget = time_override if time_override is not None else self.time_budget

        cboard = chess.Board(self.fen)
        if not list(cboard.legal_moves):
            print("  Position is terminal.")
            return

        self.sf_moves = []
        self.lc0_moves = []
        self.total_preds = 0
        self.stop_event.clear()

        priors_cache_clear()
        board = Board(self.fen)
        tree = self._fresh_tree(board, sims_budget)

        t0 = time.time()

        for checkpoint in self.REBUILD_CHECKPOINTS:
            if self.stop_event.is_set() or checkpoint >= sims_budget:
                break
            self._run_sims(tree, board, checkpoint, t0, time_budget)
            if self.stop_event.is_set() or time.time() - t0 >= time_budget:
                break
            tree = self._fresh_tree(board, sims_budget)
            self._run_sims(tree, board, checkpoint, t0, time_budget)

        # run to budget with no further resets
        self._run_sims(tree, board, sims_budget, t0, time_budget)

        elapsed = time.time() - t0
        sims = tree.sims_completed_this_move
        self.update_display(tree, sims, elapsed, status="done")

        top = self.last_info.get("top_moves", [])
        if top:
            best = top[0]
            print(f"  Best: {best['san']} ({best['uci']})   Q={best['Q']:+.3f}   visits={best['share']:.1%}")
            print(f"  Stop: {tree.sim_stop_reason or 'ceiling'}")

    def do_sf(self, depth=16):
        if not SF_LOC:
            print("  SF_LOC not set.")
            return
        print(f"  Running Stockfish depth={depth}...")
        cboard = chess.Board(self.fen)
        sign = 1 if cboard.turn == chess.WHITE else -1
        xc_map = {m["uci"]: m for m in self.last_info.get("top_moves", [])}

        with chess.engine.SimpleEngine.popen_uci(SF_LOC) as eng:
            infos = eng.analyse(cboard, chess.engine.Limit(depth=depth, time=30.0), multipv=5)
        if not isinstance(infos, list):
            infos = [infos]

        self.sf_moves = []
        print(f"\n  {CYAN}Stockfish depth={depth}{RESET}  (STM-POV cp)")
        print(f"  {'#':<3} {'SAN':<7} {'cp':>6}  {'Q(xrc)':>7}")
        print("  " + "-" * 36)
        for info in infos:
            pv = info.get("pv", [])
            mv = pv[0].uci() if pv else None
            score = info.get("score")
            cp_w = score.white().score(mate_score=1500) if score else None
            cp_stm = int(cp_w * sign) if cp_w is not None else None
            try:
                san = cboard.san(chess.Move.from_uci(mv)) if mv else "?"
            except Exception:
                san = mv or "?"
            xc = xc_map.get(mv)
            xc_str = f"{xc['Q']:+.3f}" if xc else "   ---"
            cp_str = f"{cp_stm:+d}" if cp_stm is not None else "mate"
            qc = q_color((cp_stm or 0) / 300)
            print(f"  {info.get('multipv','?'):<3} {san:<7} {qc}{cp_str:>6}{RESET}cp  {xc_str:>7}")
            if mv:
                self.sf_moves.append({"san": san, "uci": mv, "cp": cp_stm or 0})

        self.state.update(info={**self.last_info, "sf_moves": self.sf_moves})
        print()

    def do_lc0(self, nodes=4000):
        if not LC0_LOC:
            print("  LC0_LOC not set.")
            return
        print(f"  Running LC0 ({nodes} nodes)...")
        cboard = chess.Board(self.fen)
        sign = 1 if cboard.turn == chess.WHITE else -1
        castle_norm = {"e1h1": "e1g1", "e1a1": "e1c1", "e8h8": "e8g8", "e8a8": "e8c8"}
        xc_map = {m["uci"]: m for m in self.last_info.get("top_moves", [])}

        rows = lc0_analyze(cboard, nodes=nodes)
        for r in rows:
            r["move"] = castle_norm.get(r["move"], r["move"])

        self.lc0_moves = []
        print(f"\n  {CYAN}LC0 ({nodes} nodes){RESET}  (STM-POV Q)")
        print(f"  {'#':<3} {'SAN':<7} {'N':>6}  {'Q(lc0)':>7}  {'Q(xrc)':>7}  {'D':>5}")
        print("  " + "-" * 50)
        for i, r in enumerate(rows[:5], 1):
            mv = r["move"]
            try:
                san = cboard.san(chess.Move.from_uci(mv))
            except Exception:
                san = mv
            n = r.get("N", 0)
            q_lc0 = r.get("Q", 0.0)
            d = r.get("D", 0.0)
            xc = xc_map.get(mv)
            xc_str = f"{xc['Q']:+.3f}" if xc else "   ---"
            qc = q_color(q_lc0)
            print(f"  {i:<3} {san:<7} {n:>6}  {qc}{q_lc0:>+7.3f}{RESET}  {xc_str:>7}  {d:>5.3f}")
            self.lc0_moves.append({"san": san, "uci": mv, "Q": q_lc0, "N": n})

        self.state.update(info={**self.last_info, "lc0_moves": self.lc0_moves})
        print()

    def do_set_position(self, args):
        tokens = args.split()
        if not tokens:
            print("  Usage: pos startpos [moves ...]  |  pos fen <FEN> [moves ...]")
            return
        try:
            if tokens[0] == "startpos":
                b = chess.Board()
                moves = tokens[2:] if len(tokens) > 2 and tokens[1] == "moves" else []
            elif tokens[0] == "testpos":
                idx = int(tokens[1]) - 1 if len(tokens) > 1 and tokens[1].isdigit() else None
                if idx is None or not (0 <= idx < len(TEST_POSITIONS)):
                    print(f"  Usage: pos testpos <1-{len(TEST_POSITIONS)}>")
                    return
                b = chess.Board(TEST_POSITIONS[idx])
                moves = []
            elif tokens[0] == "random":
                plies = int(tokens[1]) if len(tokens) > 1 else 10
                b = chess.Board(random_init(plies=plies).fen())
                moves = []
            elif tokens[0] == "fen":
                rest = tokens[1:]
                mi = rest.index("moves") if "moves" in rest else len(rest)
                b = chess.Board(" ".join(rest[:mi]))
                moves = rest[mi + 1:]
            else:
                print("  Usage: pos startpos [moves ...]  |  pos fen <FEN> [moves ...]  |  pos random [plies]")
                return
            for mv in moves:
                b.push_uci(mv)
            self.fen = b.fen()
        except Exception as e:
            print(f"  Error: {e}")
            return

        self.sf_moves = []
        self.lc0_moves = []
        self.last_info = {}
        side = "White" if chess.Board(self.fen).turn == chess.WHITE else "Black"
        self.state.update(svg=chess.svg.board(chess.Board(self.fen), flipped=self.stm_flipped, size=480))
        print(f"  Position set. {side} to move.")

    def do_move(self, mv_str):
        try:
            b = chess.Board(self.fen)
            # accept SAN or UCI
            try:
                move = b.parse_san(mv_str)
            except Exception:
                move = chess.Move.from_uci(mv_str)
            b.push(move)
            self.fen = b.fen()
            self.sf_moves = []
            self.lc0_moves = []
            self.last_info = {}
            side = "White" if b.turn == chess.WHITE else "Black"
            self.state.update(svg=chess.svg.board(b, flipped=self.stm_flipped, size=480))
            print(f"  Played {move.uci()}. {side} to move.")
        except Exception as e:
            print(f"  Illegal move: {e}")

    # name -> (attr_or_key, type, description)
    OPTIONS = {
        "Sims":         ("sims",        int,   "default sim budget"),
        "TimeBudget":   ("time_budget", float, "default time budget (seconds)"),
        "MicroBatch":   ("micro_batch", int,   "leaves collected per MCTS step"),
        "CPuct":        ("c_puct",      float, "exploration constant"),
        "UniformEps":   ("uniform_eps", float, "prior smoothing epsilon"),
        "PriorClipMax": ("prior_clip_max", float, "prior clip ceiling"),
    }
    CFG_OPTS = {"CPuct", "UniformEps", "PriorClipMax"}
    CFG_KEYS = {"CPuct": "c_puct", "UniformEps": "uniform_eps", "PriorClipMax": "prior_clip_max"}

    def show_uci_options(self):
        print(f"\n  {BOLD}Options{RESET}  (set with: setoption name <Name> value <val>)\n")
        for name, (attr, typ, desc) in self.OPTIONS.items():
            if name in self.CFG_OPTS:
                cfg = self.make_cfg(self.sims)
                cur = self.cfg_overrides.get(self.CFG_KEYS[name], getattr(cfg, self.CFG_KEYS[name]))
            else:
                cur = getattr(self, attr)
            tname = "spin" if typ is int else "string"
            print(f"  option name {name:<14} type {tname:<7} current {cur}   # {desc}")
        print()

    def handle_setoption(self, tokens):
        try:
            ni = tokens.index("name")
            vi = tokens.index("value")
        except ValueError:
            print("  Usage: setoption name <Name> value <val>")
            return
        name = "".join(tokens[ni + 1:vi])
        value = " ".join(tokens[vi + 1:])
        canon = next((k for k in self.OPTIONS if k.lower() == name.lower()), None)
        if canon is None:
            print(f"  Unknown option '{name}'. Type 'uci' to list options.")
            return
        _, typ, _ = self.OPTIONS[canon]
        try:
            val = typ(value)
        except ValueError:
            print(f"  Bad value for {canon}: '{value}'")
            return
        if canon in self.CFG_OPTS:
            self.cfg_overrides[self.CFG_KEYS[canon]] = val
        else:
            setattr(self, self.OPTIONS[canon][0], val)
        print(f"  {canon} = {val}")

    def show_help(self):
        t = f"{self.time_budget:.0f}s" if self.time_budget != float("inf") else "inf"
        print(f"""
  {BOLD}Commands:{RESET}
    pos startpos [moves e2e4 ...]     set position from start
    pos fen <FEN> [moves ...]         set position from FEN
    pos testpos <1-5>                 load a test position
    pos random [plies]                set a random position (default 10 plies)
    go [sims [time]]                  analyze; stops at sims OR time (s)
                                      defaults: sims={self.sims}  time={t}
    sf [depth D]                      Stockfish comparison  (default d16)
    lc0 [nodes N]                     LC0 comparison  (default 4000 nodes)
    move <san|uci>                    play a move  (e.g. e2e4  or  Nf3)
    setoption name <Name> value <V>   update a tunable option
    uci                               list all options with current values
    fen                               print current FEN
    clear                             clear the priors cache
    stop                              stop ongoing analysis
    quit / exit                       exit
""")

    def run(self):
        print(f"\n  {BOLD}XERCES ENGINE CLI{RESET}  -- type 'help' for commands\n")
        while True:
            try:
                cmd = input("xerces> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not cmd:
                continue
            parts = cmd.split()
            verb = parts[0].lower()
            rest = cmd[len(verb):].strip()

            if verb in ("quit", "exit", "q"):
                break
            elif verb in ("help", "h", "?"):
                self.show_help()
            elif verb == "fen":
                print(f"  {self.fen}")
            elif verb in ("pos", "position"):
                self.do_set_position(rest)
            elif verb in ("go", "analyze", "analyse"):
                sims_ovr = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                time_ovr = float(parts[2]) if len(parts) > 2 else None
                self.do_analyze(sims_override=sims_ovr, time_override=time_ovr)
            elif verb == "sf":
                depth = 16
                if len(parts) > 1 and parts[1].isdigit():
                    depth = int(parts[1])
                elif len(parts) > 2 and parts[1] == "depth" and parts[2].isdigit():
                    depth = int(parts[2])
                self.do_sf(depth=depth)
            elif verb == "lc0":
                nodes = 4000
                if len(parts) > 1 and parts[1].isdigit():
                    nodes = int(parts[1])
                self.do_lc0(nodes=nodes)
            elif verb == "move":
                if len(parts) > 1:
                    self.do_move(parts[1])
                else:
                    print("  Usage: move <san|uci>")
            elif verb == "setoption":
                self.handle_setoption(parts[1:])
            elif verb == "uci":
                self.show_uci_options()
            elif verb == "stop":
                self.stop_event.set()
            elif verb in ("clear", "clearcache"):
                priors_cache_clear()
                print("  Cache cleared.")
            else:
                # try as move (SAN or UCI)
                try:
                    self.do_move(cmd)
                except Exception:
                    print(f"  Unknown command: {verb}  (type 'help')")

        if self.tf_thread is not None:
            self.tf_thread.close()


def main():
    parser = argparse.ArgumentParser(description="Xerces engine interactive analysis demo")
    parser.add_argument("--model", default=None, help="Path to model .h5")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--sims", type=int, default=None, help="Default sim budget")
    parser.add_argument("--port", type=int, default=None, help="HTTP display port")
    args = parser.parse_args()

    model_path = args.model or os.getenv("XERCES_MODEL", "")
    config_path = args.config or os.getenv("XERCES_CONFIG", "")
    port = args.port or int(os.getenv("XERCES_CLI_PORT", "8765"))

    if not model_path or not os.path.exists(model_path):
        print(f"Error: model not found at '{model_path}'. Set XERCES_MODEL in .env or use --model.")
        sys.exit(1)

    cfg_overrides = {}
    sims = args.sims or 6400
    if config_path and os.path.exists(config_path):
        from chessbot.config import Config
        ext = os.path.splitext(config_path)[1].lower()
        cfg = (Config.from_yaml(config_path) if ext in (".yaml", ".yml")
               else Config.from_json(config_path))
        for attr in ("c_puct", "uniform_eps", "prior_clip_max", "vscale",
                     "fpu_reduction", "pruning_factor",
                     "contempt_zero_q", "contempt_full_q", "contempt_fight_c"):
            if hasattr(cfg, attr):
                cfg_overrides[attr] = getattr(cfg, attr)
        if args.sims is None and hasattr(cfg, "sims_ceiling"):
            sc = cfg.sims_ceiling
            if isinstance(sc, dict):
                sims = max(sc.values())
            elif isinstance(sc, int):
                sims = sc

    cli = XercesCLI(model_path, cfg_overrides=cfg_overrides, sims=sims, port=port)
    cli.run()


if __name__ == "__main__":
    main()
