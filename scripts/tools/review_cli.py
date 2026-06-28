#!/usr/bin/env python3
"""
scripts/review_cli.py  --  Game review with browser-based attention/attribution.

Usage:
    python scripts/review_cli.py --run-tag precond_run1 [--port 8766]

Data root defaults to C:/Users/Bryan/Data/chessbot_data/selfplay_runs/
Override with env var XERCES_DATA_DIR.

Browser shortcuts (in review mode): ArrowRight/Space=forward  ArrowLeft=back  A=attn  P=pv
"""
import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import queue as _queue
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

DATA_ROOT = os.environ.get(
    "XERCES_DATA_DIR",
    r"C:\Users\Bryan\Data\chessbot_data\selfplay_runs"
)

POLICY_COLORS = ["#00c853", "#2196f3", "#ff6d00"]
PIECE_GLYPHS = {
    0: "", 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q",
    6: "K", 7: "K", 8: "K", 9: "K",
    10: "p", 11: "n", 12: "b", 13: "r", 14: "q",
    15: "k", 16: "k", 17: "k", 18: "k", 19: "e",
}
PIECE_UNICODE = {
    1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕",
    6: "♔", 7: "♔", 8: "♔", 9: "♔",
    10: "♟", 11: "♞", 12: "♝", 13: "♜", 14: "♛",
    15: "♚", 16: "♚", 17: "♚", 18: "♚",
}
CHESS_LIGHT = "#f0d9b5"
CHESS_DARK  = "#b58863"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Xerces Review</title>
<style>
  :root{--bg:#1a1a2e;--panel:#16213e;--accent:#4fc3f7;--text:#e0e0e0;--border:#2a4a6a}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Consolas',monospace;padding:20px}
  h1{color:var(--accent);font-size:1.15em;margin-bottom:12px;letter-spacing:2px}
  button{background:var(--panel);border:1px solid var(--border);color:var(--text);
    padding:5px 13px;cursor:pointer;font-family:inherit;font-size:.88em;border-radius:3px}
  button:hover{background:#1e3a5a} button.on{border-color:var(--accent);color:var(--accent)}
  button:disabled{opacity:.4;cursor:default}
  input,select{background:var(--panel);border:1px solid var(--border);color:var(--text);
    padding:5px 9px;font-family:inherit;font-size:.85em;border-radius:3px}

  /* ── SELECTOR ── */
  .frow{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
  #game-count{color:#666;font-size:.8em;margin-left:4px}
  #table-wrap{height:480px;overflow-y:auto;border:1px solid var(--border);border-radius:3px;margin-bottom:10px}
  #gtable{width:100%;border-collapse:collapse;font-size:.84em}
  #gtable thead{position:sticky;top:0;background:#0e1428;z-index:1}
  #gtable th{padding:7px 10px;text-align:left;color:#7ab;cursor:pointer;user-select:none;
    border-bottom:1px solid var(--border);white-space:nowrap}
  #gtable th:hover{color:var(--accent)}
  #gtable th.sorted{color:var(--accent)}
  #gtable td{padding:5px 10px;border-bottom:1px solid #141828;white-space:nowrap}
  #gtable tr:hover td{background:#1a2a3e;cursor:pointer}
  #gtable tr.sel td{background:#0d2a4a}
  .r-W{color:#81c784;font-weight:bold} .r-D{color:#ffd54f;font-weight:bold}
  .r-B{color:#e57373;font-weight:bold}
  .gid{color:#555;font-size:.78em}

  /* ── REVIEW ── */
  html,body{width:100%;height:100%;overflow:hidden}
  body{padding:12px}
  #review{height:calc(100vh - 24px)}
  #review h1{height:28px;margin-bottom:6px}
  #ply{color:#888;font-size:.8em;height:18px;margin-bottom:6px}
  .ctrl{display:flex;gap:6px;align-items:center;flex-wrap:wrap;height:32px;margin-bottom:8px}
  #cmd{width:220px}
  #review-body{display:grid;grid-template-columns:minmax(740px,1.55fr) minmax(340px,.7fr);
    gap:6px;height:calc(100vh - 108px);min-height:0}
  #img-wrap{min-width:0;min-height:0;overflow:hidden;background:#090914;
    border:1px solid #1a2a3a;border-radius:4px;display:flex;
    align-items:center;justify-content:center}
  #img-wrap img{width:100%;height:100%;object-fit:contain;border-radius:3px;display:block}
  #review-right{min-width:0;min-height:0;display:grid;
    grid-template-rows:minmax(280px,.72fr) minmax(220px,.48fr);gap:8px}
  #board-img{width:100%;height:100%;object-fit:contain;background:#090914;
    border:1px solid #1a2a3a;border-radius:4px;display:block}
  #term{min-height:0;overflow-y:auto;background:#06060f;border:1px solid #1a2a3a;
    padding:8px 12px;font-size:11px;white-space:pre;border-radius:3px;
    color:#c8d8e4;line-height:1.35}
  #placeholder{height:100%;width:100%;display:flex;align-items:center;
    justify-content:center;color:#555;font-size:.85em}
  #busy{display:none;color:var(--accent);font-size:.8em;margin-left:4px}
</style>
</head>
<body>

<!-- ════ SELECTOR PANEL ════ -->
<div id="selector">
  <h1 id="sel-title">XERCES REVIEW</h1>
  <div class="frow">
    <input id="f-text"     placeholder="Search..."  oninput="applyFilter()" style="width:180px">
    <select id="f-scenario" onchange="applyFilter()"><option value="">All scenarios</option></select>
    <select id="f-result"   onchange="applyFilter()">
      <option value="">All results</option>
      <option value="W">W wins</option>
      <option value="D">Draw</option>
      <option value="B">B wins</option>
    </select>
    <select id="f-sf" onchange="applyFilter()">
      <option value="">All SF</option>
      <option value="none">No SF</option>
      <option value="White">SF=White</option>
      <option value="Black">SF=Black</option>
    </select>
    <label style="font-size:.85em;display:flex;align-items:center;gap:4px">
      <input type="checkbox" id="f-beat" onchange="applyFilter()"> Beat SF only
    </label>
    <span id="game-count"></span>
  </div>
  <div id="table-wrap">
    <table id="gtable">
      <thead><tr>
        <th onclick="sortBy('_idx')">#</th>
        <th onclick="sortBy('ts')">TS</th>
        <th onclick="sortBy('scenario')">Scenario</th>
        <th onclick="sortBy('result')">Result</th>
        <th onclick="sortBy('sf')">SF Color</th>
        <th onclick="sortBy('plies')">Plies</th>
        <th onclick="sortBy('beat_sf')">Beat SF</th>
        <th>Game ID</th>
      </tr></thead>
      <tbody id="gtbody"></tbody>
    </table>
  </div>
  <button id="load-btn" disabled onclick="loadGame()">Load Game</button>
</div>

<!-- ════ REVIEW PANEL ════ -->
<div id="review" style="display:none">
  <h1><span id="rev-title-text">XERCES REVIEW</span> <button onclick="backToSelector()" style="font-size:.7em;margin-left:10px">Change Game</button></h1>
  <div id="ply">--</div>
  <div class="ctrl">
    <button onclick="cmd('prev')">&#8592; Back</button>
    <button onclick="cmd('next')">Forward &#8594;</button>
    <button onclick="cmd('pv')">PV</button>
    <button id="btn-attn" onclick="cmd('attn')">Attn</button>
    <button onclick="cmd('sf')">SF</button>
    <button id="btn-auto" class="on" onclick="toggleAuto()">Auto</button>
    <button onclick="reloadModel()">Reload Model</button>
    <button onclick="cmd('compare')">Compare</button>
    <input id="cmd" type="text" placeholder="goto N  /  cpl N  /  kl N  /  b5  ...">
    <span id="busy">analyzing...</span>
  </div>
  <div id="review-body">
    <div id="img-wrap"><div id="placeholder">loading...</div></div>
    <div id="review-right">
      <img id="board-img" src="" alt="board">
      <div id="term"></div>
    </div>
  </div>
</div>

<script>
let games=[], filtered=[], selId=null, sortCol='_idx', sortAsc=false, autoAttn=true;

/* ── selector ── */
fetch('/games.json').then(r=>r.json()).then(data=>{
  games = data;
  document.getElementById('sel-title').textContent = 'XERCES REVIEW  --  ' + (data[0]&&data[0]._run_tag||'');
  const scens = [...new Set(data.map(g=>g.scenario).filter(Boolean))].sort();
  const sel = document.getElementById('f-scenario');
  scens.forEach(s=>{ const o=document.createElement('option'); o.value=s; o.textContent=s; sel.appendChild(o); });
  applyFilter();
});

function getResult(g){ const r=g.result; return r>0?'W':r<0?'B':'D'; }
function getSF(g){
  if(!g.vs_stockfish) return 'none';
  return g.stockfish_color===true?'White':'Black';
}
function getSFDisplay(g){ const s=getSF(g); return s==='none'?'--':s; }

function applyFilter(){
  const txt  = document.getElementById('f-text').value.toLowerCase();
  const scen = document.getElementById('f-scenario').value;
  const res  = document.getElementById('f-result').value;
  const sf   = document.getElementById('f-sf').value;
  const beat = document.getElementById('f-beat').checked;
  filtered = games.filter(g=>{
    if(scen && g.scenario!==scen) return false;
    if(res  && getResult(g)!==res) return false;
    if(sf   && getSF(g)!==sf) return false;
    if(beat && !g.beat_sf) return false;
    if(txt){
      const hay=(g.scenario+' '+getResult(g)+' '+getSFDisplay(g)+' '+(g.game_id||'')).toLowerCase();
      if(!hay.includes(txt)) return false;
    }
    return true;
  });
  doSort(sortCol, false);
}

function sortBy(col){
  if(sortCol===col) sortAsc=!sortAsc; else{sortCol=col;sortAsc=col==='_idx'?false:true;}
  doSort(sortCol,false);
  document.querySelectorAll('#gtable th').forEach(th=>th.classList.remove('sorted'));
}

function doSort(col,toggle){
  const num=['_idx','plies','ts'];
  filtered.sort((a,b)=>{
    let va,vb;
    if(col==='result'){va=getResult(a);vb=getResult(b);}
    else if(col==='sf'){va=getSF(a);vb=getSF(b);}
    else if(col==='beat_sf'){va=a.beat_sf?1:0;vb=b.beat_sf?1:0;}
    else{va=a[col]??'';vb=b[col]??'';}
    if(num.includes(col)) return sortAsc?(Number(va)-Number(vb)):(Number(vb)-Number(va));
    const c=String(va).localeCompare(String(vb));
    return sortAsc?c:-c;
  });
  renderTable();
}

function renderTable(){
  const tbody=document.getElementById('gtbody');
  document.getElementById('game-count').textContent=filtered.length+' games';
  const show=filtered.slice(0,2000);
  tbody.innerHTML=show.map(g=>{
    const r=getResult(g), sf=getSFDisplay(g), beat=g.beat_sf?'yes':'';
    const sel=g.game_id===selId?' class="sel"':'';
    const ts=g.ts?new Date(g.ts*1000).toLocaleString('en-US',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}):'?';
    return `<tr${sel} onclick="selectGame('${g.game_id}')">
      <td>${g._idx+1}</td>
      <td style="color:#666;font-size:.8em">${ts}</td>
      <td>${g.scenario||'?'}</td>
      <td class="r-${r}">${r}</td>
      <td>${sf}</td>
      <td>${g.plies||'?'}</td>
      <td style="color:#81c784">${beat}</td>
      <td class="gid">${(g.game_id||'').slice(0,16)}...</td>
    </tr>`;
  }).join('');
  if(filtered.length>2000)
    tbody.innerHTML+=`<tr><td colspan="8" style="color:#555;text-align:center;padding:8px">`+
      `showing first 2000 of ${filtered.length} -- use filters to narrow</td></tr>`;
}

function selectGame(id){
  selId=id;
  document.getElementById('load-btn').disabled=false;
  document.querySelectorAll('#gtbody tr').forEach(tr=>{
    tr.classList.toggle('sel', tr.cells[7]&&tr.cells[7].textContent===id.slice(0,16)+'...');
  });
}

async function loadGame(){
  if(!selId) return;
  document.getElementById('load-btn').disabled=true;
  document.getElementById('load-btn').textContent='Loading...';
  const r=await fetch('/load-game',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({game_id:selId})});
  const d=await r.json();
  document.getElementById('load-btn').disabled=false;
  document.getElementById('load-btn').textContent='Load Game';
  if(d.ok){
    document.getElementById('selector').style.display='none';
    document.getElementById('review').style.display='block';
    document.getElementById('rev-title-text').textContent='XERCES REVIEW -- '+d.game_label;
    document.getElementById('ply').textContent=d.ply_info||'';
    // reset terminal and image
    document.getElementById('term').textContent='';
    document.getElementById('board-img').src='';
    const iw=document.getElementById('img-wrap');
    const existing=iw.querySelector('img');
    if(!existing) iw.innerHTML='<div id="placeholder">loading...</div>';
    if(d.output) appendTerm(d.output);
    if(d.attn_ready) refreshImg();
  } else {
    alert('Failed to load: '+(d.error||'unknown'));
  }
}

function backToSelector(){
  document.getElementById('review').style.display='none';
  document.getElementById('selector').style.display='block';
}

/* ── review ── */
function toggleAuto(){autoAttn=!autoAttn;document.getElementById('btn-auto').classList.toggle('on',autoAttn);}

async function reloadModel(){
  document.getElementById('busy').style.display='inline';
  try{
    const r=await fetch('/reload-model',{method:'POST',
      headers:{'Content-Type':'application/json'},body:'{}'});
    const d=await r.json();
    if(d.output) appendTerm(d.output);
  }finally{document.getElementById('busy').style.display='none';}
}

async function cmd(c){
  document.getElementById('busy').style.display='inline';
  try{
    const r=await fetch('/cmd',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({cmd:c, auto_attn:autoAttn})});
    const d=await r.json();
    if(d.output) appendTerm(d.output);
    if(d.ply_info) document.getElementById('ply').textContent=d.ply_info;
    if(d.attn_changed) refreshImg();
  }finally{document.getElementById('busy').style.display='none';}
}

function appendTerm(t){const el=document.getElementById('term');el.textContent+=t+'\n';el.scrollTop=el.scrollHeight;}

function refreshImg(){
  const wrap=document.getElementById('img-wrap');
  const ph=document.getElementById('placeholder');
  if(ph) ph.remove();
  let img=wrap.querySelector('img');
  if(!img){img=document.createElement('img');wrap.appendChild(img);}
  img.src='/analysis.png?t='+Date.now();
  document.getElementById('board-img').src='/board.svg?t='+Date.now();
}

let _ver=null;
setInterval(async()=>{
  try{const r=await fetch('/version');const d=await r.json();
    if(_ver===null){_ver=d.v;return;}if(d.v!==_ver)location.reload();}
  catch(e){}
},2000);

function sendInput(){const v=document.getElementById('cmd').value.trim();if(v)cmd(v);document.getElementById('cmd').value='';}
document.getElementById('cmd').addEventListener('keydown',e=>{if(e.key==='Enter')sendInput();});
document.addEventListener('keydown',e=>{
  if(document.getElementById('review').style.display==='none') return;
  if(document.activeElement===document.getElementById('cmd')) return;
  if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();cmd('next');}
  else if(e.key==='ArrowLeft'){e.preventDefault();cmd('prev');}
  else if(e.key==='a') cmd('attn');
  else if(e.key==='p') cmd('pv');
});
</script>
</body>
</html>"""


class ReviewState:
    def __init__(self):
        self._lock = threading.Lock()
        self.png_bytes = None
        self.svg_bytes = None

    def set_png(self, data):
        with self._lock:
            self.png_bytes = data

    def get_png(self):
        with self._lock:
            return self.png_bytes

    def set_svg(self, data):
        with self._lock:
            self.svg_bytes = data

    def get_svg(self):
        with self._lock:
            return self.svg_bytes


class PositionAnalyzer:
    def __init__(self, pt_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(pt_path, map_location="cpu")
        sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        arch = ckpt.get("arch", "13m-precond-conformer") if isinstance(ckpt, dict) else "13m-precond-conformer"
        model = PT_BUILDERS[arch](VARIANTS[arch])
        model.load_state_dict(sd, strict=True)
        self.model = model.to(self.device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  PT model: {arch}  {n_params:,} params  [{self.device}]")

    def run(self, fen):
        b      = Board(fen)
        enc    = b.encode_64_tokens()
        mask   = np.array(b.legal_move_mask(), dtype=bool)
        X      = torch.from_numpy(enc.astype(np.int64)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).to(self.device)

        model = self.model
        ph, vh = model.policy_head, model.value_head
        tx_attns, head_attns = {}, {}

        orig_tx = {}
        for i, blk in enumerate(model.blocks):
            orig_tx[i] = blk.forward
            def _make(idx, blk):
                def f(x):
                    r = x; n = blk.ln1(x)
                    h, aw = blk.attn(n, n, n, need_weights=True, average_attn_weights=False)
                    tx_attns[idx] = aw[0].detach().cpu().numpy()
                    x2 = r + blk.drop(h)
                    return x2 + blk.ff2(F.gelu(blk.ff1(blk.ln2(x2))))
                return f
            blk.forward = _make(i, blk)

        orig_vh = vh.forward
        def patched_vh(x):
            xp = F.gelu(vh.proj(vh.ln(x)))
            q  = vh.query.expand(xp.shape[0], -1, -1)
            h, aw = vh.mha(q, xp, xp, need_weights=True, average_attn_weights=False)
            head_attns["value"] = aw[0].detach().cpu().numpy()
            return vh.out(h.squeeze(1))
        vh.forward = patched_vh

        orig_ph = ph.forward
        def patched_ph(x):
            B, N, C = x.shape
            s = ph.ln(x)
            f = F.gelu(ph.from_proj(s)); fn = ph.from_ln(f)
            fh, aw_f = ph.from_mha(fn[:,:64], fn, fn, need_weights=True, average_attn_weights=False)
            head_attns["from"] = aw_f[0].detach().cpu().numpy()
            fv = ph.from_out(f[:,:64] + fh)
            t  = F.gelu(ph.to_proj(s)); tn = ph.to_ln(t)
            th, aw_t = ph.to_mha(tn[:,:64], tn, tn, need_weights=True, average_attn_weights=False)
            head_attns["to"] = aw_t[0].detach().cpu().numpy()
            tv    = ph.to_out(t[:,:64] + th)
            dots  = torch.bmm(fv.float(), tv.float().transpose(1,2)).mul(ph.scale).reshape(-1,64*64)
            xp2   = x[:,:64].reshape(B,8,8,C).permute(0,3,1,2).contiguous()
            p     = F.leaky_relu(ph.promo_mln(ph.promo_mix(xp2)), 0.02)
            sk    = p
            p     = F.leaky_relu(ph.promo_ln(ph.promo_c1(p)), 0.02)
            promo = ph.promo_out(p+sk).permute(0,2,3,1).reshape(-1,192).float()
            logits = torch.cat([dots, promo], dim=1).masked_fill(~ph.sometimes_legal, -1e9)
            return logits.to(x.dtype)
        ph.forward = patched_ph

        wdl_stm = None
        try:
            with torch.no_grad():
                pol, val = model(X)
                wdl_stm    = torch.softmax(val[0].float(), dim=-1).cpu().numpy()
                pol_masked = pol[0].masked_fill(~mask_t, -1e9)
                probs      = torch.softmax(pol_masked.float(), dim=-1)
                legal_ucis = b.legal_moves()
                move_idxs  = b.moves_to_indices(legal_ucis)
                ranked = sorted(
                    zip(legal_ucis, [float(probs[i]) for i in move_idxs]),
                    key=lambda x: -x[1]
                )
                top3 = ranked[:3]
                top5 = ranked[:5]
        finally:
            for i, blk in enumerate(model.blocks): blk.forward = orig_tx[i]
            vh.forward = orig_vh
            ph.forward = orig_ph

        return {"enc": enc, "fen": fen, "top3": top3, "top5": top5,
                "wdl_stm": wdl_stm, "tx_attns": tx_attns, "head_attns": head_attns}

    def make_board_svg(self, data, orientation):
        fen  = data["fen"]
        top3 = data["top3"]
        b_ch = chess.Board(fen)
        colors = ["#00c853cc", "#2196f3cc", "#ff6d00cc"]
        arrows = []
        for i, (uci, _) in enumerate(top3):
            try:
                mv = chess.Move.from_uci(uci)
                arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color=colors[i]))
            except Exception:
                continue
        svg = chess.svg.board(b_ch, orientation=orientation, arrows=arrows, size=640)
        return svg.encode("utf-8")

    def make_figure(self, data, orientation):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tx_attns   = data["tx_attns"]
        head_attns = data["head_attns"]
        uniform    = 1.0 / 68

        def key_avg(aw): return aw.mean(axis=(0, 1)) / uniform

        all_avgs = [key_avg(aw) for aw in tx_attns.values()]
        all_avgs += [key_avg(head_attns[k]) for k in ("value", "from", "to") if k in head_attns]
        attn_sum = np.mean(all_avgs, axis=0) if all_avgs else np.zeros(68)

        panels = [
            (head_attns.get("value"), "value"),    # row 0
            (head_attns.get("from"),  "pol-from"),
            (head_attns.get("to"),    "pol-to"),
            (tx_attns.get(0),         "tx00"),     # row 1
            (tx_attns.get(1),         "tx01"),
            (None,                    ""),
            (tx_attns.get(2),         "tx02"),     # row 2
            (tx_attns.get(3),         "tx03"),
            (None,                    ""),
        ]

        fig, axes = plt.subplots(3, 3, figsize=(12, 13))
        fig.patch.set_facecolor("#0a0a12")
        for ax in axes.flat:
            ax.set_facecolor("#12121e")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
            ax.tick_params(colors="#555")

        top3  = data.get("top3", [])
        top5  = data.get("top5", top3)

        fen = data.get("fen", "")
        is_black_stm = fen.split()[1] == "b" if " " in fen else False
        _stm_xor = np.arange(64, dtype=int) ^ 56  # token i -> abs sq i^56 for black

        def stm_to_abs(d64):
            if not is_black_stm:
                return d64
            abs_d64 = np.empty(64)
            abs_d64[_stm_xor] = d64
            return abs_d64

        def sq_to_xy(sq):
            r, f = sq // 8, sq % 8
            if orientation == chess.WHITE:
                return f, (7 - r) + 1
            else:
                return 7 - f, r + 1

        def board_display(d68):
            board = stm_to_abs(d68[:64]).reshape(8, 8)
            if orientation == chess.WHITE:
                return board[::-1, :], list("abcdefgh"), [str(r) for r in range(8, 0, -1)]
            else:
                return board[:, ::-1], list("hgfedcba"), [str(r) for r in range(1, 9)]

        def draw_move_arrows(ax, moves, colors):
            for i, (uci, _) in enumerate(moves[:3]):
                try:
                    mv     = chess.Move.from_uci(uci)
                    fx, fy = sq_to_xy(mv.from_square)
                    tx, ty = sq_to_xy(mv.to_square)
                    color  = colors[i]
                    ax.annotate("", xy=(tx, ty), xytext=(fx, fy),
                                arrowprops=dict(arrowstyle="-|>", color="black",
                                                lw=5.5, mutation_scale=22))
                    ax.annotate("", xy=(tx, ty), xytext=(fx, fy),
                                arrowprops=dict(arrowstyle="-|>", color=color,
                                                lw=3.0, mutation_scale=18))
                except Exception:
                    pass

        def draw_grid(ax, d68, title):
            board, files, ranks = board_display(d68)
            disp = np.full((9, 8), np.nan)
            disp[0, 2] = d68[64]; disp[0, 3] = d68[65]
            disp[0, 4] = d68[66]; disp[0, 5] = d68[67]
            disp[1:, :] = board
            im = ax.imshow(disp, vmin=0.5, vmax=2.0, cmap="RdYlGn", aspect="equal")
            ax.set_title(title, fontsize=7.5, color="#99aacc")
            ax.set_xticks(range(8)); ax.set_xticklabels(files, fontsize=5, color="#666")
            ax.set_yticks(range(9))
            ax.set_yticklabels(["g"] + ranks, fontsize=5, color="#666")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if title in ("pol-from", "pol-to", "tx03"):
                draw_move_arrows(ax, top3, POLICY_COLORS)

        for idx, (aw, title) in enumerate(panels):
            row, col = divmod(idx, 3)
            ax = axes[row, col]
            if title == "":
                ax.set_visible(False)
            else:
                d68 = key_avg(aw) if aw is not None else attn_sum
                draw_grid(ax, d68, title)

        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color=POLICY_COLORS[0], label="#1"),
            mpatches.Patch(color=POLICY_COLORS[1], label="#2"),
            mpatches.Patch(color=POLICY_COLORS[2], label="#3"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   fontsize=7, framealpha=0.3, edgecolor="#2a2a3a",
                   facecolor="#12121e", labelcolor="#c8d8e4",
                   title="arrow key", title_fontsize=6.5)
        plt.tight_layout(pad=0.5, rect=[0, 0.035, 1, 1])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()


def make_handler(state, cli):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.respond(200, "text/html; charset=utf-8", HTML_TEMPLATE.encode())
            elif path == "/games.json":
                self.respond(200, "application/json", json.dumps(cli.games).encode())
            elif path == "/analysis.png":
                png = state.get_png()
                if png: self.respond(200, "image/png", png)
                else:   self.respond(204, "image/png", b"")
            elif path == "/board.svg":
                svg = state.get_svg()
                if svg: self.respond(200, "image/svg+xml", svg)
                else:   self.respond(204, "image/svg+xml", b"")
            else:
                self.respond(404, "text/plain", b"not found")

        def do_POST(self):
            n    = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            path = self.path.split("?")[0]
            if path == "/load-game":
                resp = cli.load_game(body.get("game_id", ""))
                self.respond(200, "application/json", json.dumps(resp).encode())
            elif path == "/cmd":
                out, sc, ac, pi = cli.handle_cmd(body.get("cmd", ""), body.get("auto_attn", False))
                resp = json.dumps({"output":out,"state_changed":sc,"attn_changed":ac,"ply_info":pi})
                self.respond(200, "application/json", resp.encode())
            elif path == "/reload-model":
                resp = cli.reload_model()
                self.respond(200, "application/json", json.dumps(resp).encode())
            elif path == "/version":
                v = str(int(os.path.getmtime(__file__) * 1000))
                self.respond(200, "application/json", json.dumps({"v": v}).encode())
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


def get_review_orientation(meta):
    if meta.get("vs_stockfish"):
        return chess.BLACK if meta.get("stockfish_color") is True else chess.WHITE
    result = meta.get("result", 0)
    if result > 0: return chess.WHITE
    if result < 0: return chess.BLACK
    return chess.WHITE


class ReviewCLI:
    def __init__(self, run_tag, port=8766):
        self.run_tag    = run_tag
        self.run_dir    = os.path.join(DATA_ROOT, run_tag)
        self.state      = ReviewState()
        self.viewer     = None
        self.orientation  = chess.WHITE
        self._lock        = threading.Lock()
        self._cache_lock  = threading.Lock()
        self._model_lock  = threading.Lock()
        self.prefetch_gen = 0
        self.prefetch_q   = _queue.Queue()

        # load PT model
        pt_path = os.path.join(self.run_dir, f"{run_tag}_model.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"PT model not found: {pt_path}")
        self.pt_path  = pt_path
        self.analyzer = PositionAnalyzer(pt_path)

        # load sf_df if available
        self.sf_df = None
        sf_pkl = os.path.join(self.run_dir, "analyze_results_combined.pkl")
        if os.path.exists(sf_pkl):
            with open(sf_pkl, "rb") as f:
                self.sf_df = pickle.load(f).get("df_all")
            print(f"  SF analysis loaded: {len(self.sf_df)} rows")

        self.analysis_cache = {}

        # load game index (JSONL)
        self.games = self._load_game_index()
        print(f"  Game index: {len(self.games)} games")

        self._start_server(port)
        threading.Thread(target=self._prefetch_worker, daemon=True).start()
        webbrowser.open(f"http://localhost:{port}/")
        print(f"  Review: http://localhost:{port}/")

    def _load_game_index(self):
        path = os.path.join(self.run_dir, "game_index.json")
        games = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                    g["_run_tag"] = self.run_tag
                    games.append(g)
                except json.JSONDecodeError:
                    continue

        if self.sf_df is not None and "game_id" in self.sf_df.columns:
            valid = set(self.sf_df["game_id"].unique())
            games = [g for g in games if g.get("game_id") in valid]

        for i, g in enumerate(games):
            g["_idx"] = i
        return games

    def run_analysis_cached(self, fen, orientation):
        key = (fen, orientation)
        with self._cache_lock:
            cached = self.analysis_cache.get(key)
        if cached is None:
            with self._model_lock:
                with self._cache_lock:
                    cached = self.analysis_cache.get(key)
                if cached is None:
                    data   = self.analyzer.run(fen)
                    png    = self.analyzer.make_figure(data, orientation)
                    svg    = self.analyzer.make_board_svg(data, orientation)
                    cached = {"data": data, "png": png, "svg": svg}
                    with self._cache_lock:
                        self.analysis_cache[key] = cached
        self.state.set_png(cached["png"])
        self.state.set_svg(cached["svg"])
        return cached["data"]

    def _prefetch_worker(self):
        while True:
            gen, fen, orientation = self.prefetch_q.get()
            if gen != self.prefetch_gen:
                continue
            key = (fen, orientation)
            with self._cache_lock:
                if key in self.analysis_cache:
                    continue
            try:
                with self._model_lock:
                    if gen != self.prefetch_gen:
                        continue
                    with self._cache_lock:
                        if key in self.analysis_cache:
                            continue
                    data   = self.analyzer.run(fen)
                    png    = self.analyzer.make_figure(data, orientation)
                    svg    = self.analyzer.make_board_svg(data, orientation)
                    with self._cache_lock:
                        if gen == self.prefetch_gen:
                            self.analysis_cache[key] = {"data": data, "png": png, "svg": svg}
            except Exception:
                pass

    def _enqueue_lookahead(self, current_ply, moves_uci, orientation):
        self.prefetch_gen += 1
        gen = self.prefetch_gen
        try:
            b = chess.Board(self.viewer.board.fen())
            for k in range(1, 6):
                idx = current_ply + k - 1
                if idx >= len(moves_uci):
                    break
                b.push(chess.Move.from_uci(moves_uci[idx]))
                fen = b.fen()
                key = (fen, orientation)
                with self._cache_lock:
                    already = key in self.analysis_cache
                if not already:
                    self.prefetch_q.put((gen, fen, orientation))
        except Exception:
            pass

    def _game_by_id(self, game_id):
        return next((g for g in self.games if g.get("game_id") == game_id), None)

    def load_game(self, game_id):
        meta = self._game_by_id(game_id)
        if not meta:
            return {"ok": False, "error": f"game_id not found: {game_id}"}
        pkl_path = meta.get("pkl_file", "")
        if not os.path.exists(pkl_path):
            return {"ok": False, "error": f"log file missing: {pkl_path}"}
        try:
            with self._lock:
                self.viewer = GameViewer(pkl_path, sf_df=self.sf_df)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    self.viewer.show_moves()
                output = buf.getvalue()
            r = meta.get("result", 0)
            res = "W" if r > 0 else "B" if r < 0 else "D"
            sf_col = ""
            if meta.get("vs_stockfish"):
                sf_col = "  SF=" + ("White" if meta.get("stockfish_color") else "Black")
            label = f"{meta.get('scenario','?')}  {res}{sf_col}  {meta.get('plies','?')}pl"
            n = len(self.viewer.moves_uci)
            pi = f"Ply 0/{n}  |  {label}"
            self.orientation = get_review_orientation(meta)
            attn_ready = False
            try:
                self.run_analysis_cached(self.viewer.board.fen(), self.orientation)
                attn_ready = True
            except Exception as e:
                print(f"[attn error on load] {e}")
            self._enqueue_lookahead(self.viewer.ply, self.viewer.moves_uci, self.orientation)
            return {"ok": True, "output": output, "game_label": label,
                    "ply_info": pi, "attn_ready": attn_ready}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def reload_model(self):
        with self._model_lock:
            self.analyzer = PositionAnalyzer(self.pt_path)
        with self._cache_lock:
            self.analysis_cache.clear()
        return {"ok": True, "output": "Model reloaded. Cache cleared."}

    def handle_cmd(self, cmd_str, auto_attn=False):
        if self.viewer is None:
            return ("No game loaded.", False, False, "--")
        cmd_str = cmd_str.strip(); cmd_lo = cmd_str.lower()
        buf = io.StringIO(); sc = ac = False

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
                    fen  = self.viewer.board.fen()
                    data = self.run_analysis_cached(fen, self.orientation)
                    ac = True
                    b_ch = chess.Board(fen)
                    print("NN top-3 policy:")
                    for i, (uci, prob) in enumerate(data["top3"], 1):
                        try:    san = b_ch.san(chess.Move.from_uci(uci))
                        except: san = uci
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
                    self.viewer.seek_cpl(float(cmd_lo.replace(">","").split()[-1]))
                    self.viewer.show_moves(); sc = True
                except (ValueError, IndexError):
                    print("Usage: cpl N")
            elif cmd_lo.startswith("kl"):
                try:
                    self.viewer.seek_kl(float(cmd_lo.replace(">","").split()[-1]))
                    self.viewer.show_moves(); sc = True
                except (ValueError, IndexError):
                    print("Usage: kl N")
            elif cmd_lo.startswith("visits"):
                parts = cmd_str.split()
                if parts[-1].isdigit():
                    self.viewer.show_moves(top_n=int(parts[-1]))
                else:
                    self.viewer.show_visits(cmd_str.replace("visits","").strip())
            elif cmd_lo == "compare":
                try:
                    fen    = self.viewer.board.fen()
                    b_ch   = chess.Board(fen)
                    is_wh  = self.viewer.turn()
                    data   = self.run_analysis_cached(fen, self.orientation)
                    ac     = True
                    node, _, _ = self.viewer.node_who_chosen()
                    print("=== Compare: stored vs current model ===")
                    wdl = data.get("wdl_stm")
                    stored_wdl = (node or {}).get("nn_wdl")
                    if wdl is not None:
                        cur_w = list(wdl) if is_wh else [wdl[2], wdl[1], wdl[0]]
                        if stored_wdl is not None:
                            sw = stored_wdl
                            dw = [cur_w[i] - sw[i] for i in range(3)]
                            print(f"  {'WDL(W)':<8}  {'curr':>5}  {'stor':>5}  {'d':>6}")
                            for lbl, c, s, d in zip("WDL", cur_w, sw, dw):
                                print(f"  {lbl:<8}  {c:>5.2f}  {s:>5.2f}  {d:>+6.2f}")
                        else:
                            print(f"  WDL(W-pov): current=[{cur_w[0]:.2f} {cur_w[1]:.2f} {cur_w[2]:.2f}]  stored=n/a")
                    cands = (node.get("candidate_moves") or []) if node else []
                    stored_p = {c["uci"]: c.get("P", 0.0) for c in cands}
                    print()
                    print(f"  {'move':<8}  {'curr':>5}  {'stor':>5}  {'d':>6}")
                    for i, (uci, prob) in enumerate(data.get("top5", data.get("top3", [])), 1):
                        try:    san = b_ch.san(chess.Move.from_uci(uci))
                        except: san = uci
                        sp = stored_p.get(uci)
                        if sp is not None:
                            print(f"  {san:<8}  {prob:>5.2f}  {sp:>5.2f}  {prob-sp:>+6.2f}")
                        else:
                            print(f"  {san:<8}  {prob:>5.2f}  {'n/a':>5}  {'n/a':>6}")
                except Exception as e:
                    print(f"[compare error] {e}")
            else:
                print(f"Unknown: {cmd_str!r}  (next/prev/pv/attn/sf/goto/cpl/kl/visits/compare/b<N>)")

            if sc and auto_attn and not ac:
                try:
                    self.run_analysis_cached(self.viewer.board.fen(), self.orientation)
                    ac = True
                except Exception as e:
                    print(f"[attn error] {e}")

            if sc:
                self._enqueue_lookahead(
                    self.viewer.ply, self.viewer.moves_uci, self.orientation
                )

        n  = len(self.viewer.moves_uci)
        pi = f"Ply {self.viewer.ply}/{n}  |  {self.viewer.log.get('scenario','?')}  result: {self.viewer.result}"
        return buf.getvalue(), sc, ac, pi

    def _start_server(self, port):
        handler = make_handler(self.state, self)
        server  = HTTPServer(("localhost", port), handler)
        def _handle_error(req, addr):
            exc = sys.exc_info()[1]
            if not isinstance(exc, (ConnectionAbortedError, BrokenPipeError,
                                    ConnectionResetError)):
                import traceback; traceback.print_exc()
        server.handle_error = _handle_error
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def run(self):
        print(f"\n  {self.run_tag}  --  browser open  |  q to quit\n")
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
    p = argparse.ArgumentParser(description="Xerces game review")
    p.add_argument("--run-tag", required=True, help="Run tag (e.g. precond_run1)")
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()

    run_dir = os.path.join(DATA_ROOT, args.run_tag)
    if not os.path.isdir(run_dir):
        print(f"Error: run directory not found: {run_dir}")
        print(f"  (set XERCES_DATA_DIR env var if your data root differs from default)")
        sys.exit(1)

    ReviewCLI(args.run_tag, port=args.port).run()


if __name__ == "__main__":
    main()
