#!/usr/bin/env python3
"""
scripts/review_cli.py  --  Game review with browser-based attention/attribution.

Usage:
    python scripts/review_cli.py \
        --log  path/to/game_log.pkl \
        --pt   path/to/model.pt \
        [--sf-df path/to/analyze_results_combined.pkl] \
        [--port 8766]

Browser shortcuts: ArrowRight/Space=forward  ArrowLeft=back  A=attn  P=pv
"""
import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import chess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pyfastchess import Board
from chessbot.model import PT_BUILDERS, VARIANTS
from chessbot.review import GameViewer

POLICY_COLORS = ["#00c853", "#2196f3", "#ff6d00"]
PIECE_GLYPHS = {
    0: "", 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q",
    6: "K", 7: "K", 8: "K", 9: "K",
    10: "p", 11: "n", 12: "b", 13: "r", 14: "q",
    15: "k", 16: "k", 17: "k", 18: "k", 19: "e",
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Xerces Review</title>
<style>
  :root{--bg:#1a1a2e;--panel:#16213e;--accent:#4fc3f7;--text:#e0e0e0}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Consolas',monospace;padding:20px}
  h1{color:var(--accent);font-size:1.2em;margin-bottom:8px;letter-spacing:2px}
  #ply{color:#888;font-size:.8em;margin-bottom:8px;min-height:1.2em}
  .ctrl{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
  button{background:var(--panel);border:1px solid #2a4a6a;color:var(--text);
    padding:5px 13px;cursor:pointer;font-family:inherit;font-size:.88em;border-radius:3px}
  button:hover{background:#1e3a5a}
  button.on{border-color:var(--accent);color:var(--accent)}
  #cmd{background:var(--panel);border:1px solid #2a4a6a;color:var(--text);
    padding:5px 10px;font-family:inherit;font-size:.88em;border-radius:3px;width:220px}
  #term{height:270px;overflow-y:auto;background:#06060f;border:1px solid #1a2a3a;
    padding:10px 12px;font-size:11.5px;white-space:pre;border-radius:3px;
    margin-bottom:14px;color:#c8d8e4;line-height:1.45}
  #img-wrap{margin-top:4px}
  #img-wrap img{width:100%;border-radius:3px;display:block}
  #placeholder{background:var(--panel);border:1px dashed #2a4a6a;border-radius:3px;
    height:140px;display:flex;align-items:center;justify-content:center;
    color:#555;font-size:.85em}
  #busy{display:none;color:var(--accent);font-size:.8em;margin-left:6px}
</style>
</head>
<body>
<h1>XERCES REVIEW</h1>
<div id="ply">Loading...</div>
<div class="ctrl">
  <button onclick="cmd('prev')">&#8592; Back</button>
  <button onclick="cmd('next')">Forward &#8594;</button>
  <button onclick="cmd('pv')">PV</button>
  <button id="btn-attn" onclick="cmd('attn')">Attn</button>
  <button onclick="cmd('sf')">SF</button>
  <button id="btn-auto" onclick="toggleAuto()">Auto</button>
  <input id="cmd" type="text" placeholder="goto N  /  cpl N  /  kl N  /  b5  ...">
  <button onclick="sendInput()">Run</button>
  <span id="busy">analyzing...</span>
</div>
<div id="term"></div>
<div id="img-wrap">
  <div id="placeholder">Press Attn (or A) to run position analysis</div>
</div>
<script>
let autoAttn = false;

function toggleAuto() {
  autoAttn = !autoAttn;
  document.getElementById('btn-auto').classList.toggle('on', autoAttn);
}

async function cmd(c) {
  if (c === 'attn') document.getElementById('busy').style.display = 'inline';
  try {
    const r = await fetch('/cmd', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cmd: c})
    });
    const d = await r.json();
    if (d.output) appendTerm(d.output);
    if (d.ply_info) document.getElementById('ply').textContent = d.ply_info;
    if (d.attn_changed) refreshImg();
    if (d.state_changed && autoAttn) cmd('attn');
  } finally {
    document.getElementById('busy').style.display = 'none';
  }
}

function appendTerm(t) {
  const el = document.getElementById('term');
  el.textContent += t + '\\n';
  el.scrollTop = el.scrollHeight;
}

function refreshImg() {
  const wrap = document.getElementById('img-wrap');
  const ph = document.getElementById('placeholder');
  if (ph) ph.remove();
  let img = wrap.querySelector('img');
  if (!img) { img = document.createElement('img'); wrap.appendChild(img); }
  img.src = '/analysis.png?t=' + Date.now();
}

function sendInput() {
  const v = document.getElementById('cmd').value.trim();
  if (v) cmd(v);
  document.getElementById('cmd').value = '';
}

document.getElementById('cmd').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendInput();
});

document.addEventListener('keydown', e => {
  if (document.activeElement === document.getElementById('cmd')) return;
  if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); cmd('next'); }
  else if (e.key === 'ArrowLeft')               { e.preventDefault(); cmd('prev'); }
  else if (e.key === 'a') cmd('attn');
  else if (e.key === 'p') cmd('pv');
});

cmd('show');
</script>
</body>
</html>"""


class ReviewState:
    def __init__(self):
        self._lock = threading.Lock()
        self.png_bytes = None

    def set_png(self, data):
        with self._lock:
            self.png_bytes = data

    def get_png(self):
        with self._lock:
            return self.png_bytes


class PositionAnalyzer:
    def __init__(self, pt_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(pt_path, map_location="cpu")
        sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        arch = ckpt.get("arch", "13m-precond-conformer") if isinstance(ckpt, dict) else "13m-precond-conformer"
        model = PT_BUILDERS[arch](VARIANTS[arch])
        model.load_state_dict(sd, strict=True)
        self.model = model.to(self.device).eval()
        print(f"  PT model: {arch}  {sum(p.numel() for p in model.parameters()):,} params  [{self.device}]")

    def run(self, fen):
        """Encode position, capture all attention maps + 4 attribution passes. Returns data dict."""
        b      = Board(fen)
        enc    = b.encode_64_tokens()
        mask   = b.legal_move_mask().astype(bool)
        X      = torch.from_numpy(enc.astype(np.int64)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).to(self.device)

        model = self.model
        ph, vh = model.policy_head, model.value_head

        tx_attns   = {}
        head_attns = {}
        x_final_h  = [None]

        # patch TxBlocks to capture per-head attention weights
        orig_tx = {}
        for i, blk in enumerate(model.blocks):
            orig_tx[i] = blk.forward
            def _make_tx(idx, blk):
                def patched(x):
                    r = x
                    n = blk.ln1(x)
                    h, aw = blk.attn(n, n, n, need_weights=True, average_attn_weights=False)
                    # aw: (B, H, 68, 68) with batch_first=True
                    tx_attns[idx] = aw[0].detach().cpu().numpy()
                    x = r + blk.drop(h)
                    return x + blk.ff2(F.gelu(blk.ff1(blk.ln2(x))))
                return patched
            blk.forward = _make_tx(i, blk)

        # patch value head
        orig_vh = vh.forward
        def patched_vh(x):
            xp = F.gelu(vh.proj(vh.ln(x)))
            q  = vh.query.expand(xp.shape[0], -1, -1)
            h, aw = vh.mha(q, xp, xp, need_weights=True, average_attn_weights=False)
            head_attns["value"] = aw[0].detach().cpu().numpy()  # (H, 1, 68)
            return vh.out(h.squeeze(1))
        vh.forward = patched_vh

        # patch policy head
        orig_ph = ph.forward
        def patched_ph(x):
            B, N, C = x.shape
            s   = ph.ln(x)
            f   = F.gelu(ph.from_proj(s))
            fn  = ph.from_ln(f)
            fh, aw_f = ph.from_mha(fn[:, :64], fn, fn, need_weights=True, average_attn_weights=False)
            head_attns["from"] = aw_f[0].detach().cpu().numpy()  # (H, 64, 68)
            fv  = ph.from_out(f[:, :64] + fh)
            t   = F.gelu(ph.to_proj(s))
            tn  = ph.to_ln(t)
            th, aw_t = ph.to_mha(tn[:, :64], tn, tn, need_weights=True, average_attn_weights=False)
            head_attns["to"] = aw_t[0].detach().cpu().numpy()  # (H, 64, 68)
            tv   = ph.to_out(t[:, :64] + th)
            dots = torch.bmm(fv.float(), tv.float().transpose(1, 2)).mul(ph.scale).reshape(-1, 64*64)
            xp2  = x[:, :64].reshape(B, 8, 8, C).permute(0, 3, 1, 2).contiguous()
            p    = F.leaky_relu(ph.promo_mln(ph.promo_mix(xp2)), 0.02)
            sk   = p
            p    = F.leaky_relu(ph.promo_ln(ph.promo_c1(p)), 0.02)
            promo = ph.promo_out(p + sk).permute(0, 2, 3, 1).reshape(-1, 192).float()
            logits = torch.cat([dots, promo], dim=1)
            logits = logits.masked_fill(~ph.sometimes_legal, -1e9)
            return logits.to(x.dtype)
        ph.forward = patched_ph

        # hook to capture x_final (output of last transformer block)
        def cap_hook(module, inp, out):
            x_final_h[0] = out
            out.retain_grad()
        handle = model.blocks[-1].register_forward_hook(cap_hook)

        try:
            with torch.enable_grad():
                pol, val = model(X)
                x_final  = x_final_h[0]  # (1, 68, D)

                # top-3 policy moves
                pol_masked = pol[0].masked_fill(~mask_t, -1e9)
                probs      = torch.softmax(pol_masked.float(), dim=-1)
                top3_idx   = probs.topk(3).indices.cpu().tolist()
                legal_ucis = b.legal_moves()
                idx_to_uci = {idx: uci for uci, idx in zip(legal_ucis, b.moves_to_indices(legal_ucis))}
                top3 = [(idx_to_uci.get(i, "?"), float(probs[i])) for i in top3_idx]

                # 4 attribution passes: W, D, L, policy-best
                attrs  = {}
                targets = [("W", val[0, 0]), ("D", val[0, 1]), ("L", val[0, 2])]
                for name, tgt in targets:
                    model.zero_grad()
                    if x_final.grad is not None:
                        x_final.grad.zero_()
                    tgt.backward(retain_graph=True)
                    attrs[name] = (x_final * x_final.grad).sum(-1)[0].detach().cpu().numpy()

                best_idx = int(pol_masked.argmax())
                model.zero_grad()
                if x_final.grad is not None:
                    x_final.grad.zero_()
                pol[0, best_idx].backward()
                attrs["policy"] = (x_final * x_final.grad).sum(-1)[0].detach().cpu().numpy()

        finally:
            handle.remove()
            for i, blk in enumerate(model.blocks):
                blk.forward = orig_tx[i]
            vh.forward = orig_vh
            ph.forward = orig_ph

        return {
            "enc": enc, "fen": fen, "top3": top3,
            "tx_attns": tx_attns, "head_attns": head_attns, "attrs": attrs,
        }

    def make_figure(self, data):
        """Generate 3x4 matplotlib figure. Returns PNG bytes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        enc        = data["enc"]
        fen        = data["fen"]
        top3       = data["top3"]
        tx_attns   = data["tx_attns"]
        head_attns = data["head_attns"]
        attrs      = data["attrs"]
        uniform    = 1.0 / 68

        fig, axes = plt.subplots(3, 4, figsize=(22, 16))
        fig.patch.set_facecolor("#0a0a12")
        for ax in axes.flat:
            ax.set_facecolor("#12121e")
            for sp in ax.spines.values():
                sp.set_edgecolor("#2a2a3a")
            ax.tick_params(colors="#555")

        # ── [0,0] board + policy overlays ────────────────────────────
        ax = axes[0, 0]
        enc_grid = enc.reshape(8, 8)  # [rank_idx][file_idx], rank_idx 0 = rank 1

        checker = np.array([[(r + f) % 2 for f in range(8)] for r in range(8)], float)
        ax.imshow(checker, cmap="Greys", vmin=0, vmax=2.5, aspect="equal",
                  extent=[-0.5, 7.5, -0.5, 7.5], origin="lower", alpha=0.35)

        b_chess   = chess.Board(fen)
        stm_white = b_chess.turn == chess.WHITE
        for i, (uci, prob) in enumerate(top3):
            try:
                mv = chess.Move.from_uci(uci)
            except Exception:
                continue
            ff = chess.square_file(mv.from_square)
            fr = chess.square_rank(mv.from_square)
            tf = chess.square_file(mv.to_square)
            tr = chess.square_rank(mv.to_square)
            # enc is STM-as-white; flip coords if black to move
            if not stm_white:
                ff = 7 - ff; fr = 7 - fr
                tf = 7 - tf; tr = 7 - tr
            alpha = max(0.25, prob * 2.0)
            c = POLICY_COLORS[i]
            ax.add_patch(mpatches.Rectangle((ff-.5, fr-.5), 1, 1, linewidth=0,
                                             facecolor=c, alpha=alpha * 0.4))
            ax.add_patch(mpatches.Rectangle((tf-.5, tr-.5), 1, 1, linewidth=2,
                                             edgecolor=c, facecolor=c, alpha=alpha * 0.65))

        for rank in range(8):
            for f in range(8):
                tok   = int(enc_grid[rank, f])
                glyph = PIECE_GLYPHS.get(tok, "")
                if not glyph:
                    continue
                color = "#f0f0f8" if 1 <= tok <= 9 else "#8888aa"
                ax.text(f, rank, glyph, ha="center", va="center", fontsize=12,
                        fontweight="bold", color=color)

        for i, (uci, prob) in enumerate(top3):
            try:
                san = b_chess.san(chess.Move.from_uci(uci))
            except Exception:
                san = uci
            ax.text(0.02, 0.98 - i * 0.09, f"{i+1}. {san}  {prob:.3f}",
                    transform=ax.transAxes, fontsize=8, color=POLICY_COLORS[i],
                    va="top", bbox=dict(facecolor="#0a0a12", alpha=0.7, pad=2, linewidth=0))

        ax.set_xlim(-0.5, 7.5); ax.set_ylim(-0.5, 7.5)
        ax.set_xticks(range(8)); ax.set_xticklabels(list("abcdefgh"), fontsize=7, color="#666")
        ax.set_yticks(range(8)); ax.set_yticklabels([str(r) for r in range(1, 9)], fontsize=7, color="#666")
        ax.set_title("Position  (STM = white)", fontsize=9, color="#99aacc")

        # ── shared helpers ────────────────────────────────────────────
        def key_avg_68(aw_np):
            """(H, Nq, Nk) -> (68,) mean key-attention received, ratio vs uniform."""
            return aw_np.mean(axis=(0, 1)) / uniform

        def draw_grid(ax, data68, title, cmap, vmin, vmax, fmt=".2f"):
            """9x8 heatmap: global tokens row on top, ranks 8..1 below."""
            disp = np.full((9, 8), np.nan)
            disp[0, 2] = data68[64]; disp[0, 3] = data68[65]
            disp[0, 4] = data68[66]; disp[0, 5] = data68[67]
            board = data68[:64].reshape(8, 8)  # [rank_idx][file_idx]
            for r in range(8):
                disp[1 + (7 - r), :] = board[r, :]

            im = ax.imshow(disp, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
            ax.set_title(title, fontsize=8.5, color="#99aacc")
            ax.set_xticks(range(8)); ax.set_xticklabels(list("abcdefgh"), fontsize=6, color="#666")
            ax.set_yticks(range(9))
            ax.set_yticklabels(["glbl"] + [str(r) for r in range(8, 0, -1)], fontsize=6, color="#666")

            mid  = (vmin + vmax) / 2
            span = max(vmax - vmin, 1e-8)
            for rank in range(8):
                for f in range(8):
                    v = disp[1 + (7 - rank), f]
                    if np.isnan(v):
                        continue
                    tok   = int(enc_grid[rank, f])
                    glyph = PIECE_GLYPHS.get(tok, "")
                    val_s = format(v, fmt)
                    txt   = f"{glyph}\n{val_s}" if glyph else val_s
                    tc    = "#dde" if abs(v - mid) > span * 0.3 else "#222"
                    ax.text(f, 1 + (7-rank), txt, ha="center", va="center", fontsize=5.5, color=tc)

            for gi, col in enumerate([2, 3, 4, 5]):
                v = disp[0, col]
                if not np.isnan(v):
                    tc  = "#dde" if abs(v - mid) > span * 0.3 else "#222"
                    ax.text(col, 0, f"g{gi}\n{format(v, fmt)}", ha="center", va="center",
                            fontsize=5.5, color=tc)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ── row 0: head attention maps ────────────────────────────────
        for col, key, title in [
            (1, "value", "Value attn  (key recv, ratio/uniform)"),
            (2, "from",  "Pol from-branch  (ratio/uniform)"),
            (3, "to",    "Pol to-branch  (ratio/uniform)"),
        ]:
            if key in head_attns:
                draw_grid(axes[0, col], key_avg_68(head_attns[key]), title, "RdYlGn", 0.5, 2.0)

        # ── row 1: transformer block attention ────────────────────────
        for i in range(4):
            if i in tx_attns:
                draw_grid(axes[1, i], key_avg_68(tx_attns[i]),
                          f"tx{i:02d}  (key recv, ratio/uniform)", "RdYlGn", 0.5, 2.0)

        # ── row 2: attributions (x * grad).sum(D) ────────────────────
        attr_titles = [
            ("W",      "Win attribution  (x·grad)"),
            ("D",      "Draw attribution  (x·grad)"),
            ("L",      "Loss attribution  (x·grad)"),
            ("policy", "Policy attribution  (x·grad)"),
        ]
        for col, (key, title) in enumerate(attr_titles):
            if key in attrs:
                a  = attrs[key]
                mv = max(float(np.abs(a).max()), 1e-8)
                draw_grid(axes[2, col], a, title, "RdBu_r", -mv, mv, fmt="+.3f")

        plt.tight_layout(pad=1.0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()


def make_handler(state, cli):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.respond(200, "text/html; charset=utf-8", HTML_TEMPLATE.encode())
            elif path == "/analysis.png":
                png = state.get_png()
                if png:
                    self.respond(200, "image/png", png)
                else:
                    self.respond(204, "image/png", b"")
            else:
                self.respond(404, "text/plain", b"not found")

        def do_POST(self):
            n    = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            out, sc, ac, pi = cli.handle_cmd(body.get("cmd", ""))
            resp = json.dumps({"output": out, "state_changed": sc, "attn_changed": ac, "ply_info": pi})
            self.respond(200, "application/json", resp.encode())

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


class ReviewCLI:
    def __init__(self, log_path, pt_path, sf_df_path=None, port=8766):
        self.state    = ReviewState()
        self.analyzer = PositionAnalyzer(pt_path)
        self._lock    = threading.Lock()

        sf_df = None
        if sf_df_path and os.path.exists(sf_df_path):
            with open(sf_df_path, "rb") as f:
                sf_df = pickle.load(f).get("df_all")

        self.viewer = GameViewer(log_path, sf_df=sf_df)
        self._start_server(port)
        webbrowser.open(f"http://localhost:{port}/")
        print(f"  Review: http://localhost:{port}/")

    def _start_server(self, port):
        handler = make_handler(self.state, self)
        server  = HTTPServer(("localhost", port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def handle_cmd(self, cmd_str):
        """Dispatch command; return (output, state_changed, attn_changed, ply_info)."""
        cmd_str = cmd_str.strip()
        cmd_lo  = cmd_str.lower()
        buf     = io.StringIO()
        sc = ac = False

        with self._lock, contextlib.redirect_stdout(buf):
            if cmd_lo in ("next", "forward", ""):
                self.viewer.next(); self.viewer.show_moves(); sc = True
            elif cmd_lo in ("prev", "back", "b"):
                self.viewer.prev(); self.viewer.show_moves(); sc = True
            elif cmd_lo.startswith("b") and cmd_lo[1:].isdigit():
                self.viewer.goto(max(0, self.viewer.ply - int(cmd_lo[1:])))
                self.viewer.show_moves(); sc = True
            elif cmd_lo == "show":
                self.viewer.show_moves()
            elif cmd_lo == "pv":
                self.viewer.show_pv()
            elif cmd_lo.startswith("sf"):
                self.viewer.show_sf_overlay(cmd_lo)
            elif cmd_lo == "attn":
                try:
                    data = self.analyzer.run(self.viewer.board.fen())
                    self.state.set_png(self.analyzer.make_figure(data))
                    ac = True
                    b_ch = chess.Board(self.viewer.board.fen())
                    print("NN top-3 policy:")
                    for i, (uci, prob) in enumerate(data["top3"], 1):
                        try:
                            san = b_ch.san(chess.Move.from_uci(uci))
                        except Exception:
                            san = uci
                        print(f"  {i}. {san:<8}  {prob:.4f}")
                except Exception as e:
                    print(f"[attn error] {e}")
            elif cmd_lo.startswith("goto"):
                parts = cmd_lo.split()
                if len(parts) > 1 and parts[1].isdigit():
                    self.viewer.goto(int(parts[1])); self.viewer.show_moves(); sc = True
                else:
                    print("Usage: goto N")
            elif cmd_lo.startswith("cpl"):
                try:
                    self.viewer.seek_cpl(float(cmd_lo.replace(">", " ").split()[-1]))
                    self.viewer.show_moves(); sc = True
                except (ValueError, IndexError):
                    print("Usage: cpl N")
            elif cmd_lo.startswith("kl"):
                try:
                    self.viewer.seek_kl(float(cmd_lo.replace(">", " ").split()[-1]))
                    self.viewer.show_moves(); sc = True
                except (ValueError, IndexError):
                    print("Usage: kl N")
            elif cmd_lo.startswith("visits"):
                parts = cmd_str.split()
                if parts[-1].isdigit():
                    self.viewer.show_moves(top_n=int(parts[-1]))
                else:
                    self.viewer.show_visits(cmd_str.replace("visits", "").strip())
            else:
                print(f"Unknown command: {cmd_str!r}  (next/prev/pv/attn/sf/goto/cpl/kl/visits/b<N>)")

        n     = len(self.viewer.moves_uci)
        pi    = (f"Ply {self.viewer.ply}/{n}  |  "
                 f"{self.viewer.log.get('scenario', '?')}  |  "
                 f"result: {self.viewer.result}")
        return buf.getvalue(), sc, ac, pi

    def run(self):
        print("\n  XERCES REVIEW CLI  --  browser open  |  q to quit\n")
        # print initial state to browser terminal
        self.handle_cmd("show")
        while True:
            try:
                cmd = input("review> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd.lower() in ("q", "quit", "exit"):
                break
            out, _, _, _ = self.handle_cmd(cmd)
            if out:
                print(out, end="")


def main():
    p = argparse.ArgumentParser(description="Xerces game review with browser analysis")
    p.add_argument("--log",   required=True, help="game log .pkl path")
    p.add_argument("--pt",    required=True, help="PT model .pt path")
    p.add_argument("--sf-df", default=None,  help="analyze_results_combined.pkl (optional)")
    p.add_argument("--port",  type=int, default=8766)
    args = p.parse_args()

    for path, label in [(args.log, "log"), (args.pt, "pt")]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}"); sys.exit(1)

    ReviewCLI(args.log, args.pt, sf_df_path=args.sf_df, port=args.port).run()


if __name__ == "__main__":
    main()
