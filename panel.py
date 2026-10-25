"""
kfpaper admin panel — a secured web GUI with full control over the paper bot.
View status/positions/trades/equity; edit config + strategy params; pause/resume; flatten; reset.
Auth: HTTP Basic (KF_PANEL_USER / KF_PANEL_PASS). Bind to localhost only — reach via SSH tunnel.
Reads the bot's SQLite ledger (WAL, read-only) and the shared config.json. The bot hot-reloads
config each loop; one-shot commands go through the control table.
"""
from __future__ import annotations

import os
import secrets
import sqlite3

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from config import load_config, save_config, SCALAR_FIELDS
from strategy import DEFAULT_PARAMS
from risk import DEFAULT_RISK
from strategies import REGISTRY

DB_PATH = os.getenv("KF_DB", "data/paper.db")
USER = os.getenv("KF_PANEL_USER", "admin")
PASS = os.getenv("KF_PANEL_PASS", "change-me")
POLL_HINT = int(os.getenv("KF_POLL_SEC", "60"))

app = FastAPI(title="kfpaper admin")
security = HTTPBasic()


def auth(creds: HTTPBasicCredentials = Depends(security)):
    ok = secrets.compare_digest(creds.username, USER) and secrets.compare_digest(creds.password, PASS)
    if not ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad credentials",
                            {"WWW-Authenticate": "Basic"})
    return True


def _db():
    c = sqlite3.connect(DB_PATH, timeout=5.0)
    c.execute("PRAGMA busy_timeout=5000")
    c.row_factory = sqlite3.Row
    return c


def _rows(sql, args=()):
    try:
        with _db() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]
    except Exception:
        return []


@app.get("/api/status")
def status_api(_=Depends(auth)):
    cfg = load_config()
    eq = _rows("SELECT ts,balance,equity,open_positions FROM equity ORDER BY rowid DESC LIMIT 1")
    pos = _rows("SELECT symbol,side,entry,qty,notional,leverage,stop,opened_ts,bars FROM positions")
    ctl = {r["k"]: r["v"] for r in _rows("SELECT k,v FROM control")}
    peak = _rows("SELECT MAX(equity) AS p FROM equity")
    last = eq[0] if eq else {}
    gross = sum(p.get("notional", 0) for p in pos)
    return {"config": cfg, "last": last, "positions": pos, "gross": gross,
            "halted": ctl.get("halted") == "1", "halt_reason": ctl.get("halt_reason", ""),
            "peak": (peak[0]["p"] if peak else None),
            "strategies": list(REGISTRY.keys()),
            "start_balance": cfg["start_balance"], "poll_sec": cfg.get("poll_sec", POLL_HINT)}


@app.get("/api/trades")
def trades_api(limit: int = 50, _=Depends(auth)):
    return _rows("SELECT ts,symbol,side,action,price,qty,fee,pnl,balance,reason FROM trades ORDER BY id DESC LIMIT ?",
                 (min(limit, 500),))


@app.get("/api/equity")
def equity_api(limit: int = 400, _=Depends(auth)):
    return list(reversed(_rows("SELECT ts,equity FROM equity ORDER BY rowid DESC LIMIT ?", (min(limit, 2000),))))


@app.post("/api/config")
async def set_config(req: Request, _=Depends(auth)):
    body = await req.json()
    cfg = load_config()
    syms = body.get("symbols")
    if isinstance(syms, str):
        syms = [s.strip() for s in syms.split(",") if s.strip()]
    if isinstance(syms, list) and syms:
        cfg["symbols"] = syms
    if body.get("strategy") in REGISTRY:
        cfg["strategy"] = body["strategy"]
    if isinstance(body.get("risk"), dict):
        rr = dict(DEFAULT_RISK)
        for k, v in body["risk"].items():
            if k in DEFAULT_RISK:
                try:
                    rr[k] = float(v)
                except (ValueError, TypeError):
                    raise HTTPException(400, f"bad risk {k}")
        cfg["risk"] = rr
    for k, typ in SCALAR_FIELDS.items():
        if k in body and body[k] is not None and body[k] != "":
            try:
                cfg[k] = (str(body[k]).lower() in ("1", "true", "yes", "on")) if typ is bool else typ(body[k])
            except (ValueError, TypeError):
                raise HTTPException(400, f"bad value for {k}")
    if isinstance(body.get("params"), dict):
        p = dict(DEFAULT_PARAMS)
        for k, v in body["params"].items():
            if k in DEFAULT_PARAMS:
                try:
                    p[k] = float(v)
                except (ValueError, TypeError):
                    raise HTTPException(400, f"bad param {k}")
        cfg["params"] = p
    save_config(cfg)
    return JSONResponse({"ok": True, "config": cfg})


@app.post("/api/control")
async def control(req: Request, _=Depends(auth)):
    action = (await req.json()).get("action", "")
    if action in ("pause", "resume"):
        cfg = load_config(); cfg["trading_enabled"] = (action == "resume"); save_config(cfg)
        return {"ok": True, "trading_enabled": cfg["trading_enabled"]}
    if action in ("flatten", "reset"):
        with _db() as c:
            c.execute("INSERT INTO control(k,v) VALUES('cmd',?) ON CONFLICT(k) DO UPDATE SET v=?", (action, action))
            c.commit()
        return {"ok": True, "queued": action}
    raise HTTPException(400, "unknown action")


@app.get("/", response_class=HTMLResponse)
def index(_=Depends(auth)):
    return HTML


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>kfpaper admin</title>
<style>
:root{--bg:#0e1116;--card:#171c24;--line:#262d38;--fg:#e6edf3;--mut:#8b98a8;--grn:#2ea043;--red:#f85149;--acc:#388bfd}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mut);font-size:12px;margin-bottom:16px}
.grid{display:grid;gap:12px}.tiles{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.v{font-size:22px;font-weight:650;margin-top:2px}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600}
.on{background:rgba(46,160,67,.15);color:var(--grn)}.off{background:rgba(248,81,73,.15);color:var(--red)}
button{background:#21262d;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--acc)}button.danger:hover{border-color:var(--red);color:var(--red)}
table{width:100%;border-collapse:collapse;font-size:12.5px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600}.pos{color:var(--grn)}.neg{color:var(--red)}
label{display:block;font-size:11px;color:var(--mut);margin:6px 0 2px}
input,select{width:100%;background:#0e1116;border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:6px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}
.mut{color:var(--mut);font-size:12px}h3{margin:0 0 10px;font-size:14px}
canvas{width:100%;height:90px;display:block}.sp{height:10px}
</style></head><body><div class="wrap">
<h1>kfpaper — Kraken Futures paper bot</h1>
<div class="sub">PAPER ONLY · no API keys · cannot place real orders &nbsp;|&nbsp; <span id="alive" class="pill off">checking…</span></div>

<div class="grid tiles">
  <div class="card"><div class="k">Equity</div><div class="v" id="equity">–</div></div>
  <div class="card"><div class="k">Balance</div><div class="v" id="balance">–</div></div>
  <div class="card"><div class="k">P&L vs start</div><div class="v" id="pnl">–</div></div>
  <div class="card"><div class="k">Open positions</div><div class="v" id="nopos">–</div></div>
  <div class="card"><div class="k">Gross exposure</div><div class="v" id="gross">–</div></div>
</div>

<div class="sp"></div>
<div class="card" id="haltbanner" style="display:none;border-color:var(--red);color:var(--red)"></div>

<div class="sp"></div>
<div class="card"><canvas id="eqchart" width="960" height="90"></canvas><div class="mut">equity curve (recent)</div></div>

<div class="sp"></div>
<div class="card">
  <h3>Controls</h3>
  <div class="row" style="grid-template-columns:repeat(auto-fit,minmax(110px,1fr))">
    <button onclick="ctl('resume')">▶ Resume</button>
    <button onclick="ctl('pause')">⏸ Pause</button>
    <button class="danger" onclick="if(confirm('Close ALL paper positions now?'))ctl('flatten')">⏹ Flatten all</button>
    <button class="danger" onclick="if(confirm('Wipe paper account to start balance? Clears trades/equity.'))ctl('reset')">↺ Reset account</button>
  </div>
</div>

<div class="sp"></div>
<div class="card">
  <h3>Open positions</h3>
  <table id="postbl"><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Qty</th><th>Notional</th><th>Lev</th><th>Stop</th><th>Bars</th></tr></thead><tbody></tbody></table>
  <div class="mut" id="posempty" style="display:none">flat — no open positions</div>
</div>

<div class="sp"></div>
<div class="card">
  <h3>Configuration <span class="mut">(applied live on the bot's next loop)</span></h3>
  <div class="row">
    <div><label>Symbols (comma-sep)</label><input id="symbols"></div>
    <div><label>Timeframe</label><select id="timeframe"><option>5m</option><option>15m</option><option>30m</option><option>1h</option><option>4h</option><option>1d</option></select></div>
    <div><label>Leverage</label><input id="leverage" type="number" step="0.5"></div>
    <div><label>Start balance ($)</label><input id="start_balance" type="number" step="10"></div>
    <div><label>Taker fee (frac)</label><input id="taker_fee" type="number" step="0.0001"></div>
    <div><label>Max position %</label><input id="max_position_pct" type="number" step="0.05"></div>
    <div><label>Allow short</label><select id="allow_short"><option value="true">yes</option><option value="false">no</option></select></div>
    <div><label>Poll (sec)</label><input id="poll_sec" type="number" step="5"></div>
    <div><label>Strategy</label><select id="strategy"></select></div>
  </div>
  <h3 style="margin-top:14px">Risk management <span class="mut">(liquidation-safe leverage + circuit breakers)</span></h3>
  <div class="row" id="risk"></div>
  <h3 style="margin-top:14px">Strategy parameters</h3>
  <div class="row" id="params"></div>
  <div class="sp"></div>
  <button onclick="saveCfg()">💾 Save configuration</button> <span class="mut" id="savemsg"></span>
</div>

<div class="sp"></div>
<div class="card">
  <h3>Recent trades</h3>
  <table id="trtbl"><thead><tr><th>Time (UTC)</th><th>Symbol</th><th>Action</th><th>Price</th><th>Fee</th><th>P&L</th><th>Balance</th><th>Reason</th></tr></thead><tbody></tbody></table>
</div>
</div>

<script>
const $=id=>document.getElementById(id);
const f2=n=>(n==null?'–':Number(n).toFixed(2));
async function get(u){const r=await fetch(u);if(!r.ok)throw 0;return r.json();}
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json();}

let CFG=null,STRATS=[];
async function refresh(){
  try{
    const s=await get('/api/status');CFG=s.config;STRATS=s.strategies||[];
    const last=s.last||{};
    $('equity').textContent='$'+f2(last.equity);
    $('balance').textContent='$'+f2(last.balance);
    const pnl=last.equity!=null?last.equity-s.start_balance:null;
    const pe=$('pnl');pe.textContent=(pnl==null?'–':(pnl>=0?'+':'')+f2(pnl));pe.className='v '+(pnl>=0?'pos':'neg');
    $('nopos').textContent=s.positions.length;
    $('gross').textContent=s.gross!=null?'$'+f2(s.gross):'–';
    const hb=$('haltbanner');
    if(s.halted){hb.style.display='block';hb.textContent='⛔ TRADING HALTED by risk engine — '+s.halt_reason+'. Existing positions still managed; new entries blocked.';}
    else{hb.style.display='none';}
    // alive if last snapshot younger than 3 polls
    const fresh=last.ts&&((Date.now()-new Date(last.ts+'Z').getTime())/1000 < 3*(s.poll_sec||60));
    const tog=CFG.trading_enabled;
    $('alive').className='pill '+(fresh?'on':'off');
    $('alive').textContent=(fresh?'bot live':'bot stale')+' · '+(tog?'trading':'PAUSED');
    // positions
    const tb=$('postbl').querySelector('tbody');tb.innerHTML='';
    s.positions.forEach(p=>{tb.insertAdjacentHTML('beforeend',
      `<tr><td>${p.symbol}</td><td class="${p.side=='long'?'pos':'neg'}">${p.side}</td><td>${f2(p.entry)}</td><td>${Number(p.qty).toFixed(4)}</td><td>${f2(p.notional)}</td><td>${p.leverage}</td><td>${f2(p.stop)}</td><td>${p.bars}</td></tr>`);});
    $('posempty').style.display=s.positions.length?'none':'block';
    if(document.activeElement.tagName!='INPUT'&&document.activeElement.tagName!='SELECT')fillCfg();
    drawEq(await get('/api/equity?limit=400'));
    const tr=await get('/api/trades?limit=40');const tt=$('trtbl').querySelector('tbody');tt.innerHTML='';
    tr.forEach(t=>{const c=t.pnl>0?'pos':(t.pnl<0?'neg':'');tt.insertAdjacentHTML('beforeend',
      `<tr><td>${(t.ts||'').replace('T',' ').slice(0,19)}</td><td>${t.symbol}</td><td>${t.action}</td><td>${f2(t.price)}</td><td>${f2(t.fee)}</td><td class="${c}">${f2(t.pnl)}</td><td>${f2(t.balance)}</td><td class="mut">${t.reason||''}</td></tr>`);});
  }catch(e){$('alive').className='pill off';$('alive').textContent='no data';}
}
function buildInputs(container,obj,prefix){
  const c=$(container);if(c.dataset.built)return;c.innerHTML='';
  Object.keys(obj).forEach(k=>c.insertAdjacentHTML('beforeend',`<div><label>${k}</label><input id="${prefix}${k}" type="number" step="0.001"></div>`));
  c.dataset.built='1';
}
function fillCfg(){
  if(!CFG)return;
  $('symbols').value=CFG.symbols.join(',');['timeframe','leverage','start_balance','taker_fee','max_position_pct','poll_sec'].forEach(k=>$(k).value=CFG[k]);
  $('allow_short').value=String(CFG.allow_short);
  const ss=$('strategy');if(!ss.dataset.built){ss.innerHTML='';STRATS.forEach(n=>ss.insertAdjacentHTML('beforeend',`<option value="${n}">${n}</option>`));ss.dataset.built='1';}
  if(document.activeElement!=ss)ss.value=CFG.strategy;
  buildInputs('risk',CFG.risk,'r_');buildInputs('params',CFG.params,'p_');
  Object.keys(CFG.risk).forEach(k=>{const el=$('r_'+k);if(el&&document.activeElement!=el)el.value=CFG.risk[k];});
  Object.keys(CFG.params).forEach(k=>{const el=$('p_'+k);if(el&&document.activeElement!=el)el.value=CFG.params[k];});
}
async function saveCfg(){
  const body={symbols:$('symbols').value,allow_short:$('allow_short').value,strategy:$('strategy').value,params:{},risk:{}};
  ['timeframe','leverage','start_balance','taker_fee','max_position_pct','poll_sec'].forEach(k=>body[k]=$(k).value);
  Object.keys(CFG.params).forEach(k=>body.params[k]=$('p_'+k).value);
  Object.keys(CFG.risk).forEach(k=>body.risk[k]=$('r_'+k).value);
  const r=await post('/api/config',body);$('savemsg').textContent=r.ok?'saved ✓':'error';setTimeout(()=>$('savemsg').textContent='',2500);refresh();
}
async function ctl(a){await post('/api/control',{action:a});refresh();}
function drawEq(data){
  const c=$('eqchart'),x=c.getContext('2d');const W=c.width,H=c.height;x.clearRect(0,0,W,H);
  if(!data||data.length<2)return;const vs=data.map(d=>d.equity);const mn=Math.min(...vs),mx=Math.max(...vs),rng=(mx-mn)||1;
  x.beginPath();data.forEach((d,i)=>{const px=i/(data.length-1)*W,py=H-((d.equity-mn)/rng)*(H-8)-4;i?x.lineTo(px,py):x.moveTo(px,py);});
  x.strokeStyle=vs[vs.length-1]>=vs[0]?'#2ea043':'#f85149';x.lineWidth=1.5;x.stroke();
}
refresh();setInterval(refresh,15000);
</script></body></html>"""
