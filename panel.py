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
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from config import load_config, save_config, SCALAR_FIELDS
from strategy import DEFAULT_PARAMS
from risk import DEFAULT_RISK
from strategies import REGISTRY
from backtest import run_backtest, run_portfolio_backtest
from mlcommon import model_status

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
    pos = _rows("SELECT symbol,side,entry,mark,qty,notional,leverage,stop,opened_ts,bars FROM positions")
    ctl = {r["k"]: r["v"] for r in _rows("SELECT k,v FROM control")}
    peak = _rows("SELECT MAX(equity) AS p FROM equity")
    last = eq[0] if eq else {}
    gross = sum(p.get("notional", 0) for p in pos)
    try:
        from sentiment import current_fng
        fng = current_fng()
    except Exception:
        fng = None
    return {"config": cfg, "last": last, "positions": pos, "gross": gross,
            "halted": ctl.get("halted") == "1", "halt_reason": ctl.get("halt_reason", ""),
            "peak": (peak[0]["p"] if peak else None), "fng": fng,
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


@app.post("/api/backtest")
async def backtest_api(req: Request, _=Depends(auth)):
    """Replay the CURRENT saved params/risk over historical candles (no DB writes)."""
    body = await req.json()
    cfg = load_config()
    symbol = body.get("symbol") or cfg["symbols"][0]
    timeframe = body.get("timeframe") or cfg["timeframe"]
    strategy = body.get("strategy") if body.get("strategy") in REGISTRY else cfg["strategy"]
    try:
        bars = int(body.get("bars") or 1500)
        start_balance = float(body.get("start_balance") or cfg["start_balance"])
    except (ValueError, TypeError):
        raise HTTPException(400, "bad bars/start_balance")
    try:
        res = await run_in_threadpool(
            run_backtest, symbol, timeframe, strategy, cfg["params"], cfg["risk"],
            start_balance, cfg["taker_fee"], bars)
    except Exception as e:
        raise HTTPException(400, f"backtest failed: {e}")
    return res


@app.post("/api/portfolio_backtest")
async def portfolio_backtest_api(req: Request, _=Depends(auth)):
    """Replay ALL configured symbols on one shared account with funding — mirrors the live bot."""
    body = await req.json()
    cfg = load_config()
    timeframe = body.get("timeframe") or cfg["timeframe"]
    strategy = body.get("strategy") if body.get("strategy") in REGISTRY else cfg["strategy"]
    try:
        bars = int(body.get("bars") or 2000)
        start_balance = float(body.get("start_balance") or cfg["start_balance"])
    except (ValueError, TypeError):
        raise HTTPException(400, "bad bars/start_balance")
    try:
        res = await run_in_threadpool(
            run_portfolio_backtest, cfg["symbols"], timeframe, strategy, cfg["params"],
            cfg["risk"], start_balance, cfg["taker_fee"], bars)
    except Exception as e:
        raise HTTPException(400, f"portfolio backtest failed: {e}")
    return res


@app.get("/api/optimize")
def optimize_get(_=Depends(auth)):
    from optimize import load_best
    return {"best": load_best()}


@app.post("/api/optimize")
async def optimize_post(req: Request, _=Depends(auth)):
    """Run an Optuna study (recommend-only) — walk-forward portfolio objective + holdout check."""
    body = await req.json()
    cfg = load_config()
    tf = body.get("timeframe") or cfg["timeframe"]
    strategy = body.get("strategy") if body.get("strategy") in REGISTRY else cfg["strategy"]
    try:
        trials = max(20, min(int(body.get("trials") or 150), 1000))
    except (ValueError, TypeError):
        raise HTTPException(400, "bad trials")

    def _run():
        from optimize import optimize
        params = {**DEFAULT_PARAMS, **cfg.get("params", {})}
        risk_cfg = {**DEFAULT_RISK, **cfg.get("risk", {})}
        return optimize(cfg["symbols"], tf, strategy, params, risk_cfg,
                        cfg["start_balance"], cfg["taker_fee"], trials)

    try:
        rec = await run_in_threadpool(_run)
    except Exception as e:
        raise HTTPException(400, f"optimize failed: {e}")
    return {"ok": True, "best": rec}


@app.get("/api/autoapply")
def autoapply_get(_=Depends(auth)):
    from autoapply import load_status
    return {"status": load_status(), "enabled": bool(load_config().get("autopilot_apply"))}


@app.get("/api/model")
def model_api(_=Depends(auth)):
    """Status of the trained ML models for the configured symbols at the active timeframe."""
    cfg = load_config()
    tf = cfg["timeframe"]
    out = []
    for sym in cfg["symbols"]:
        for mode in ("meta", "dir"):
            st = model_status(sym, tf, mode)
            out.append(st or {"symbol": sym, "timeframe": tf, "mode": mode, "trained_at": None})
    return {"timeframe": tf, "models": out}


@app.post("/api/mltrain")
async def mltrain_api(req: Request, _=Depends(auth)):
    """Backfill history + (re)train both models for the given symbols. Runs in a threadpool."""
    body = await req.json()
    cfg = load_config()
    symbols = body.get("symbols") or cfg["symbols"]
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]
    tf = body.get("timeframe") or cfg["timeframe"]
    try:
        horizon = int(body.get("horizon") or cfg["params"].get("ml_horizon", 24))
    except (ValueError, TypeError):
        raise HTTPException(400, "bad horizon")

    def _run():
        import mldata
        import mltrain
        try:
            import sentiment
            sentiment.backfill_fng()
        except Exception:
            pass
        params = {**DEFAULT_PARAMS, **cfg.get("params", {})}
        risk_cfg = {**DEFAULT_RISK, **cfg.get("risk", {})}
        results = []
        for sym in symbols:
            mldata.backfill(sym, tf, 20000)
            results.append(mltrain.train_symbol(sym, tf, horizon, params, risk_cfg,
                                                cfg["taker_fee"], cfg["start_balance"], mode="both"))
        return results

    try:
        res = await run_in_threadpool(_run)
    except Exception as e:
        raise HTTPException(400, f"train failed: {e}")
    return {"ok": True, "results": res}


@app.get("/api/recommendation")
def recommendation_api(_=Depends(auth)):
    """Latest auto-pilot recommendation (recommend-only; written by autopilot.py)."""
    from autopilot import load_recommendation
    return {"recommendation": load_recommendation()}


@app.post("/api/autopilot")
async def autopilot_api(_=Depends(auth)):
    """Run the auto-pilot now: backfill + retrain + OOS-rank all strategies. Recommend-only."""
    cfg = load_config()

    def _run():
        from autopilot import run_autopilot
        params = {**DEFAULT_PARAMS, **cfg.get("params", {})}
        risk_cfg = {**DEFAULT_RISK, **cfg.get("risk", {})}
        horizon = int(params.get("ml_horizon", 24))
        return run_autopilot(cfg["symbols"], cfg["timeframe"], horizon, params, risk_cfg,
                             cfg["taker_fee"], cfg["start_balance"], cfg["strategy"])

    try:
        rec = await run_in_threadpool(_run)
    except Exception as e:
        raise HTTPException(400, f"autopilot failed: {e}")
    return {"ok": True, "recommendation": rec}


@app.get("/", response_class=HTMLResponse)
def index(_=Depends(auth)):
    return HTML


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>kfpaper</title>
<style>
:root{
  --bg:#0b0e13; --plane:#0e1116; --card:#141a22; --card2:#171f29; --line:#232c38;
  --fg:#e6edf3; --fg2:#aeb9c6; --mut:#7d8a9a;
  --acc:#3987e5; --pos:#2ea043; --posdim:rgba(46,160,67,.14);
  --neg:#f05d57; --negdim:rgba(240,93,87,.14);
  --s1:#3987e5; --s2:#199e70; --s3:#d95926;    /* categorical series (validated, dark) */
  --radius:12px; --shadow:0 1px 0 rgba(255,255,255,.02),0 8px 24px rgba(0,0,0,.28);
}
*{box-sizing:border-box}
body{margin:0;background:
   radial-gradient(1100px 500px at 80% -10%,rgba(57,135,229,.08),transparent 60%),var(--bg);
   color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--acc)}
code{background:#0e141b;border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:12px}
.wrap{max-width:1120px;margin:0 auto;padding:0 16px 48px}
/* header */
header{position:sticky;top:0;z-index:20;background:rgba(11,14,19,.82);backdrop-filter:blur(10px);
  border-bottom:1px solid var(--line)}
.hbar{max-width:1120px;margin:0 auto;padding:12px 16px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.brand{font-weight:750;letter-spacing:.2px;font-size:16px;display:flex;align-items:center;gap:9px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--mut);box-shadow:0 0 0 4px rgba(125,138,154,.14)}
.dot.live{background:var(--pos);box-shadow:0 0 0 4px var(--posdim)}
.dot.bad{background:var(--neg);box-shadow:0 0 0 4px var(--negdim)}
.hstat{margin-left:auto;display:flex;gap:18px;align-items:center}
.hstat .l{color:var(--mut);font-size:10.5px;text-transform:uppercase;letter-spacing:.06em}
.hstat .n{font-weight:700;font-size:15px;font-variant-numeric:tabular-nums}
.pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:11.5px;font-weight:600;border:1px solid var(--line)}
.pill.on{color:var(--pos)}.pill.off{color:var(--neg)}
/* tabs */
nav.tabs{display:flex;gap:4px;max-width:1120px;margin:0 auto;padding:0 10px;overflow-x:auto}
.tab{padding:11px 15px;color:var(--fg2);cursor:pointer;border:none;background:none;font-size:13.5px;font-weight:600;
  border-bottom:2px solid transparent;white-space:nowrap}
.tab:hover{color:var(--fg)}
.tab.active{color:var(--fg);border-bottom-color:var(--acc)}
/* cards + layout */
.sp{height:16px}.sp.s{height:10px}
.grid{display:grid;gap:14px}
.tiles{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.cols{grid-template-columns:1.6fr 1fr}
@media(max-width:820px){.cols{grid-template-columns:1fr}}
.card{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
  border-radius:var(--radius);padding:16px;box-shadow:var(--shadow)}
.card h3{margin:0 0 3px;font-size:14px}
.card .cap{color:var(--mut);font-size:12px;margin:0 0 12px}
.tile{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
  border-radius:var(--radius);padding:13px 14px;box-shadow:var(--shadow)}
.tile .k{color:var(--mut);font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;display:flex;align-items:center;gap:5px}
.tile .v{font-size:24px;font-weight:720;margin-top:3px;font-variant-numeric:tabular-nums;letter-spacing:-.3px}
.tile .d{font-size:11.5px;color:var(--mut);margin-top:1px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.muted{color:var(--mut)}
/* callout */
.note{background:rgba(57,135,229,.06);border:1px solid rgba(57,135,229,.22);border-radius:10px;
  padding:12px 14px;color:var(--fg2);font-size:12.8px;line-height:1.6}
.note b{color:var(--fg)}
.note .s1{color:var(--s1);font-weight:700}.note .s2{color:var(--s2);font-weight:700}.note .s3{color:var(--s3);font-weight:700}
/* buttons + inputs */
button{background:#1b2531;color:var(--fg);border:1px solid var(--line);border-radius:9px;padding:9px 14px;cursor:pointer;
  font-size:13px;font-weight:600;transition:.12s}
button:hover{border-color:var(--acc);background:#1f2a38}
button:disabled{opacity:.5;cursor:default}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
button.primary:hover{filter:brightness(1.08)}
button.danger:hover{border-color:var(--neg);color:var(--neg);background:rgba(240,93,87,.08)}
label{display:block;font-size:11px;color:var(--mut);margin:0 0 3px;font-weight:600}
input,select{width:100%;background:#0b1118;border:1px solid var(--line);color:var(--fg);border-radius:8px;padding:8px 9px;font-size:13px}
input:focus,select:focus{outline:none;border-color:var(--acc)}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:10px}
/* tables */
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:8px 9px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
.badge{display:inline-block;padding:1px 7px;border-radius:6px;font-size:11px;font-weight:700}
.badge.long{background:var(--posdim);color:var(--pos)}.badge.short{background:var(--negdim);color:var(--neg)}
.tag{font-size:10.5px;color:var(--mut);border:1px solid var(--line);border-radius:5px;padding:1px 6px}
/* chart */
.chartwrap{position:relative}
canvas{width:100%;display:block}
.tip{position:absolute;pointer-events:none;display:none;background:#0b1118;border:1px solid var(--line);
  border-radius:8px;padding:7px 9px;font-size:11.5px;box-shadow:0 6px 18px rgba(0,0,0,.4);z-index:5;min-width:120px}
.tip .tt{color:var(--mut);margin-bottom:3px}
.tip .tr{display:flex;align-items:center;gap:6px;justify-content:space-between}
.tip .sw{width:9px;height:9px;border-radius:2px;display:inline-block}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--fg2);margin-top:8px}
.legend .it{display:flex;align-items:center;gap:7px}
.legend .sw{width:11px;height:3px;border-radius:2px;display:inline-block}
.help{cursor:help;color:var(--mut);border:1px solid var(--line);border-radius:50%;width:14px;height:14px;
  display:inline-flex;align-items:center;justify-content:center;font-size:9px;font-weight:700}
.verdict{font-size:13px;padding:10px 12px;border-radius:9px;margin-top:4px;border:1px solid var(--line)}
.hidden{display:none}
.foot{color:var(--mut);font-size:11.5px;text-align:center;margin-top:26px}
</style></head><body>

<header>
  <div class="hbar">
    <div class="brand"><span class="dot" id="dot"></span> kfpaper <span class="tag">PAPER · no real orders</span></div>
    <div class="hstat">
      <div><div class="l">Equity</div><div class="n" id="h_eq">–</div></div>
      <div><div class="l">P&amp;L</div><div class="n" id="h_pnl">–</div></div>
      <span class="pill off" id="alive">checking…</span>
    </div>
  </div>
  <nav class="tabs">
    <button class="tab active" data-tab="dash" onclick="tab('dash')">Dashboard</button>
    <button class="tab" data-tab="bt" onclick="tab('bt')">Backtest</button>
    <button class="tab" data-tab="ml" onclick="tab('ml')">ML Lab</button>
    <button class="tab" data-tab="cfg" onclick="tab('cfg')">Settings</button>
    <button class="tab" data-tab="trades" onclick="tab('trades')">Trades</button>
  </nav>
</header>

<div class="wrap">

<!-- ================= DASHBOARD ================= -->
<section id="tab-dash">
  <div class="sp"></div>
  <div class="grid tiles">
    <div class="tile"><div class="k">Equity</div><div class="v" id="equity">–</div><div class="d">balance + unrealized</div></div>
    <div class="tile"><div class="k">Balance</div><div class="v" id="balance">–</div><div class="d">realized cash</div></div>
    <div class="tile"><div class="k">P&amp;L vs start</div><div class="v" id="pnl">–</div><div class="d" id="pnlpct">—</div></div>
    <div class="tile"><div class="k">Open positions</div><div class="v" id="nopos">–</div><div class="d">live now</div></div>
    <div class="tile"><div class="k">Gross exposure</div><div class="v" id="gross">–</div><div class="d" id="grossx">—</div></div>
    <div class="tile"><div class="k">Market sentiment</div><div class="v" id="fng">–</div><div class="d" id="fnglabel">Fear &amp; Greed</div></div>
  </div>
  <div class="sp"></div>
  <div class="card neg hidden" id="haltbanner" style="border-color:var(--neg);color:var(--neg)"></div>
  <div class="grid cols">
    <div class="card">
      <h3>Equity curve</h3><p class="cap">live account value over recent snapshots</p>
      <div class="chartwrap"><canvas id="eqchart" height="220"></canvas><div class="tip" id="eqchart_tip"></div></div>
    </div>
    <div class="card">
      <h3>Controls</h3><p class="cap">act on the live paper account</p>
      <div class="grid" style="grid-template-columns:1fr 1fr;gap:10px">
        <button onclick="ctl('resume')">▶ Resume</button>
        <button onclick="ctl('pause')">⏸ Pause</button>
        <button class="danger" onclick="if(confirm('Close ALL paper positions now?'))ctl('flatten')">⏹ Flatten all</button>
        <button class="danger" onclick="if(confirm('Wipe paper account to start balance? Clears trades/equity.'))ctl('reset')">↺ Reset</button>
      </div>
      <div class="sp s"></div>
      <div class="note">Changes apply on the bot's next loop. <b>Flatten</b> closes open trades; <b>Reset</b> wipes the account to the start balance.</div>
    </div>
  </div>
  <div class="sp"></div>
  <div class="card">
    <h3>Open positions</h3><p class="cap">live trades the bot is currently managing</p>
    <table id="postbl"><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Price</th><th>Qty</th><th>Notional</th><th>Lev</th><th>Stop</th><th>Bars</th></tr></thead><tbody></tbody></table>
    <div class="muted" id="posempty" style="display:none;padding:10px 2px">flat — no open positions</div>
  </div>
</section>

<!-- ================= BACKTEST ================= -->
<section id="tab-bt" class="hidden">
  <div class="sp"></div>
  <div class="note">
    <b>What a backtest does.</b> It replays a strategy <b>bar-by-bar over real historical candles</b>, using the
    <b>exact same</b> entry, stop, trailing, time-exit, liquidation and position-sizing rules as the live bot — so it
    estimates how that strategy would have performed. <b>Compare all</b> runs every strategy on the same candles so you
    can see which wins. It uses your <b>current saved</b> risk + params and never touches the live account.
  </div>
  <div class="sp"></div>
  <div class="card">
    <h3>Run a backtest</h3><p class="cap">pick a market + window, then run one strategy or compare them all</p>
    <div class="row">
      <div><label>Symbol</label><select id="bt_symbol"></select></div>
      <div><label>Timeframe</label><select id="bt_tf"><option>5m</option><option>15m</option><option>30m</option><option>1h</option><option>4h</option><option>1d</option></select></div>
      <div><label>Strategy (single run)</label><select id="bt_strat"></select></div>
      <div><label>Candles</label><input id="bt_bars" type="number" step="100" value="1500"></div>
      <div><label>Start balance ($)</label><input id="bt_bal" type="number" step="10" value="1000"></div>
    </div>
    <div class="sp s"></div>
    <button class="primary" onclick="runBacktest()">▶ Run single</button>
    <button onclick="compareAll()">⚖ Compare all strategies</button>
    <button onclick="runPortfolio()" title="All symbols, one shared account, with funding — mirrors the live bot">🧪 Portfolio (matches live)</button>
    <span class="muted" id="bt_msg"></span>
  </div>
  <div class="sp"></div>
  <div id="bt_results" class="hidden">
    <div class="card">
      <h3 id="bt_title">Result</h3><p class="cap" id="bt_period"></p>
      <div class="chartwrap"><canvas id="bt_chart" height="230"></canvas><div class="tip" id="bt_chart_tip"></div></div>
      <div class="legend" id="bt_legend"></div>
    </div>
    <div class="sp"></div>
    <div id="bt_single">
      <div class="grid tiles" id="bt_tiles"></div>
      <div class="sp"></div>
      <div class="verdict" id="bt_verdict"></div>
    </div>
    <div id="bt_compare" class="hidden card">
      <h3>Strategy comparison <span class="muted" style="font-weight:400">(same market, same window)</span></h3>
      <table id="bt_cmptbl"><thead><tr><th>Strategy</th><th>Return</th><th>Trades</th><th>Win %</th><th>Profit factor</th><th>Max DD</th><th>Sharpe</th></tr></thead><tbody></tbody></table>
    </div>
    <div class="sp"></div>
    <div class="card">
      <p class="cap" style="margin:0 0 8px">What the numbers mean</p>
      <div class="row" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;font-size:12px;color:var(--fg2)">
        <div><b>Total return</b> — % change of the account over the window.</div>
        <div><b>Trades</b> — number of closed trades (too many = fee drag).</div>
        <div><b>Win rate</b> — % of trades that made money.</div>
        <div><b>Profit factor</b> — gross profit ÷ gross loss. &gt;1 profitable, &gt;1.5 good.</div>
        <div><b>Max drawdown</b> — worst peak-to-trough drop. Smaller = safer.</div>
        <div><b>Sharpe (ann.)</b> — return per unit of risk. &gt;1 good, &gt;2 strong.</div>
      </div>
    </div>
  </div>
</section>

<!-- ================= ML LAB ================= -->
<section id="tab-ml" class="hidden">
  <div class="sp"></div>
  <div class="note">
    <b>Two model types.</b>
    <span class="s2">ml_meta</span> (meta-labeling): the rule-based ensemble proposes a trade and a
    <b>LightGBM</b> model decides whether it's worth taking and how big — a smart filter.
    <span class="s3">ml_dir</span> (direct): the model predicts up/flat/down itself.
    Metrics below are <b>out-of-sample</b> — measured on recent candles the model never trained on, so they're an honest
    read on edge. Models are per <b>(symbol, timeframe)</b>: train at the timeframe you intend to trade, then compare in
    the <b>Backtest</b> tab (strategies <code>ml_meta</code> / <code>ml_dir</code>).
  </div>
  <div class="sp"></div>
  <div class="card">
    <h3>Models <span class="muted" style="font-weight:400" id="ml_tf"></span></h3>
    <p class="cap">status + out-of-sample performance for each configured symbol</p>
    <table id="mltbl"><thead><tr><th>Symbol</th><th>Model</th><th>Trained (UTC)</th><th>Samples</th><th>OOS return</th><th>OOS PF</th><th>OOS maxDD</th><th>Skill</th></tr></thead><tbody></tbody></table>
    <div class="muted" id="mlempty" style="display:none;padding:10px 2px">no models yet for this timeframe — click “Backfill &amp; train”.</div>
    <div class="sp s"></div>
    <button class="primary" onclick="runTrain()">🧠 Backfill &amp; train (both models)</button>
    <span class="muted" id="ml_msg"></span>
    <div class="sp s"></div>
    <div class="note"><b>Skill</b> = ROC-AUC for ml_meta / accuracy for ml_dir. 0.50 means “no better than a coin flip”; higher is better. Training downloads history + trains both models for every configured symbol — it can take a minute.</div>
  </div>
  <div class="sp"></div>
  <div class="card">
    <h3>Auto-pilot <span class="muted" style="font-weight:400">· recommend-only</span></h3>
    <p class="cap">backfills, retrains, and ranks every strategy out-of-sample across your symbols — then recommends. It never switches the live bot on its own; you click Apply.</p>
    <div id="rec_body" class="muted">no recommendation yet — run it, or wait for the weekly schedule.</div>
    <div class="sp s"></div>
    <button class="primary" onclick="runAutopilot()">🤖 Run auto-pilot now</button>
    <button id="rec_apply" class="hidden" onclick="applyRec()">✔ Apply recommended</button>
    <span class="muted" id="ap_msg"></span>
  </div>
  <div class="sp"></div>
  <div class="card">
    <h3>Optimizer <span class="muted" style="font-weight:400">· Optuna · recommend-only</span></h3>
    <p class="cap">searches decision/risk params for a durable edge, scored by the portfolio backtest across out-of-sample folds. The verdict is the <b>holdout</b> column (data Optuna never saw) — trust a result only if holdout ≈ study.</p>
    <div class="row">
      <div><label>Timeframe</label><select id="op_tf"><option>5m</option><option>15m</option><option>30m</option><option>1h</option><option>4h</option><option>1d</option></select></div>
      <div><label>Strategy</label><select id="op_strat"></select></div>
      <div><label>Trials</label><input id="op_trials" type="number" step="10" value="150"></div>
    </div>
    <div class="sp s"></div>
    <div id="op_body" class="muted">no optimization yet — run it (a few minutes for 150 trials).</div>
    <div class="sp s"></div>
    <button class="primary" onclick="runOptimize()">⚙ Run optimizer</button>
    <button id="op_apply" class="hidden" onclick="applyOpt()">✔ Apply best params</button>
    <span class="muted" id="op_msg"></span>
    <div class="sp s"></div>
    <div id="aa_body" class="note" style="display:none"></div>
    <div class="sp s"></div>
    <button id="aa_toggle" onclick="toggleAutoApply()">Auto-apply: …</button>
    <span class="muted">runs nightly after training · strict holdout guards · paper only</span>
  </div>
</section>

<!-- ================= SETTINGS ================= -->
<section id="tab-cfg" class="hidden">
  <div class="sp"></div>
  <div class="card">
    <h3>Configuration <span class="muted" style="font-weight:400">(applied live on the bot's next loop)</span></h3>
    <p class="cap">the active strategy, market list, and account settings</p>
    <div class="row">
      <div><label>Symbols (comma-sep)</label><input id="symbols"></div>
      <div><label>Timeframe</label><select id="timeframe"><option>5m</option><option>15m</option><option>30m</option><option>1h</option><option>4h</option><option>1d</option></select></div>
      <div><label>Strategy</label><select id="strategy"></select></div>
      <div><label>Leverage</label><input id="leverage" type="number" step="0.5"></div>
      <div><label>Start balance ($)</label><input id="start_balance" type="number" step="10"></div>
      <div><label>Taker fee (frac)</label><input id="taker_fee" type="number" step="0.0001"></div>
      <div><label>Max position %</label><input id="max_position_pct" type="number" step="0.05"></div>
      <div><label>Allow short</label><select id="allow_short"><option value="true">yes</option><option value="false">no</option></select></div>
      <div><label>Poll (sec)</label><input id="poll_sec" type="number" step="5"></div>
    </div>
    <h3 style="margin-top:16px">Risk management <span class="muted" style="font-weight:400">(liquidation-safe leverage + circuit breakers)</span></h3>
    <div class="row" id="risk"></div>
    <h3 style="margin-top:16px">Strategy parameters</h3>
    <div class="row" id="params"></div>
    <div class="sp s"></div>
    <button class="primary" onclick="saveCfg()">💾 Save configuration</button> <span class="muted" id="savemsg"></span>
  </div>
</section>

<!-- ================= TRADES ================= -->
<section id="tab-trades" class="hidden">
  <div class="sp"></div>
  <div class="card">
    <h3>Recent trades</h3><p class="cap">latest fills on the paper account</p>
    <table id="trtbl"><thead><tr><th>Time (UTC)</th><th>Symbol</th><th>Action</th><th>Price</th><th>Fee</th><th>P&amp;L</th><th>Balance</th><th>Reason</th></tr></thead><tbody></tbody></table>
  </div>
</section>

<div class="foot">kfpaper · paper-trading control panel · data from your VPS over Tailscale</div>
</div>

<script>
const $=id=>document.getElementById(id);
const f2=n=>(n==null?'–':Number(n).toFixed(2));
const sign=n=>(n>=0?'+':'');
async function get(u){const r=await fetch(u);if(!r.ok)throw new Error(r.status);return r.json();}
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const j=await r.json().catch(()=>({}));if(!r.ok)j._err=(j&&j.detail)||r.status;return j;}

let CFG=null,STRATS=[],curTab='dash';
const SCOLOR={ensemble:getCss('--s1'),ml_meta:getCss('--s2'),ml_dir:getCss('--s3'),breakout:getCss('--mut')};
function getCss(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function tab(name){curTab=name;
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===name));
  ['dash','bt','ml','cfg','trades'].forEach(n=>$('tab-'+n).classList.toggle('hidden',n!==name));
  Object.keys(CHARTS).forEach(id=>render(id,null));   // canvases need a resize-aware redraw when shown
}

/* ---------- chart engine (multi-series line + crosshair tooltip) ---------- */
const CHARTS={};
function fit(c){const r=c.getBoundingClientRect();const dpr=window.devicePixelRatio||1;
  c.width=Math.max(300,r.width*dpr);c.height=(c.getAttribute('height')||220)*dpr;
  const x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);return {W:r.width,H:+(c.getAttribute('height')||220)};}
function setChart(id,series,start){const c=$(id);if(!c)return;CHARTS[id]={series,start};
  if(!c._m){c._m=1;
    c.addEventListener('mousemove',e=>{const r=c.getBoundingClientRect();render(id,(e.clientX-r.left)/r.width);});
    c.addEventListener('mouseleave',()=>render(id,null));
  }
  render(id,null);}
function render(id,hx){
  const c=$(id),st=CHARTS[id];if(!c||!st||c.offsetParent===null)return;
  const {W,H}=fit(c),x=c.getContext('2d');const pad=10,bottom=H-pad,top=pad;
  const all=[];st.series.forEach(s=>s.points.forEach(p=>all.push(p.equity)));
  if(st.start!=null)all.push(st.start);
  if(all.length<2)return;
  let mn=Math.min(...all),mx=Math.max(...all);const rng=(mx-mn)||1;mn-=rng*.06;mx+=rng*.06;
  const Y=v=>bottom-(v-mn)/(mx-mn)*(bottom-top);
  const len=Math.max(...st.series.map(s=>s.points.length));
  const X=i=>len<2?0:i/(len-1)*W;
  x.clearRect(0,0,W,H);
  // baseline at start value
  if(st.start!=null){x.setLineDash([4,4]);x.strokeStyle=getCss('--line');x.lineWidth=1;
    x.beginPath();x.moveTo(0,Y(st.start));x.lineTo(W,Y(st.start));x.stroke();x.setLineDash([]);}
  // area fill for single series
  if(st.series.length===1){const s=st.series[0],col=s.color;
    const g=x.createLinearGradient(0,top,0,bottom);g.addColorStop(0,col+'44');g.addColorStop(1,col+'05');
    x.beginPath();s.points.forEach((p,i)=>{const px=X(i),py=Y(p.equity);i?x.lineTo(px,py):x.moveTo(px,py);});
    x.lineTo(X(s.points.length-1),bottom);x.lineTo(0,bottom);x.closePath();x.fillStyle=g;x.fill();}
  // lines
  st.series.forEach(s=>{x.beginPath();s.points.forEach((p,i)=>{const px=X(i),py=Y(p.equity);i?x.lineTo(px,py):x.moveTo(px,py);});
    x.strokeStyle=s.color;x.lineWidth=2;x.lineJoin='round';x.stroke();});
  // crosshair + tooltip
  const tip=$(id+'_tip');
  if(hx==null){if(tip)tip.style.display='none';return;}
  const idx=Math.round(hx*(len-1)),px=X(idx);
  x.strokeStyle=getCss('--mut');x.globalAlpha=.5;x.beginPath();x.moveTo(px,top);x.lineTo(px,bottom);x.stroke();x.globalAlpha=1;
  let rows='';st.series.forEach(s=>{const p=s.points[Math.min(idx,s.points.length-1)];if(!p)return;
    x.fillStyle=s.color;x.beginPath();x.arc(px,Y(p.equity),3.5,0,7);x.fill();
    const chg=st.start?((p.equity/st.start-1)*100):null;
    rows+=`<div class="tr"><span><span class="sw" style="background:${s.color}"></span>${s.name}</span>`+
          `<b>$${f2(p.equity)}${chg!=null?' <span class="muted">'+sign(chg)+chg.toFixed(1)+'%</span>':''}</b></div>`;});
  const t=(st.series[0].points[Math.min(idx,st.series[0].points.length-1)]||{}).t||'';
  if(tip){tip.innerHTML=`<div class="tt">${(t+'').replace('T',' ').slice(0,16)}</div>${rows}`;
    tip.style.display='block';const left=Math.min(Math.max(px+12,4),W-tip.offsetWidth-4);
    tip.style.left=left+'px';tip.style.top=top+'px';}
}

/* ---------- dashboard refresh ---------- */
async function refresh(){
  try{
    const s=await get('/api/status');CFG=s.config;STRATS=s.strategies||[];
    const last=s.last||{};
    $('equity').textContent='$'+f2(last.equity);$('h_eq').textContent='$'+f2(last.equity);
    $('balance').textContent='$'+f2(last.balance);
    const pnl=last.equity!=null?last.equity-s.start_balance:null;
    const ppct=pnl!=null&&s.start_balance?(pnl/s.start_balance*100):null;
    [$('pnl'),$('h_pnl')].forEach(e=>{e.textContent=pnl==null?'–':sign(pnl)+'$'+f2(Math.abs(pnl));e.className=e.className.replace(/\b(pos|neg)\b/g,'')+' '+(pnl>=0?'pos':'neg');});
    $('pnlpct').textContent=ppct==null?'—':sign(ppct)+ppct.toFixed(2)+'%  of start';
    $('nopos').textContent=s.positions.length;
    $('gross').textContent=s.gross!=null?'$'+f2(s.gross):'–';
    $('grossx').textContent=(s.gross&&last.equity)?(s.gross/last.equity).toFixed(2)+'× equity':'—';
    if(s.fng){const fv=s.fng.value;$('fng').textContent=fv;$('fng').className='v '+(fv<45?'neg':(fv>55?'pos':''));$('fnglabel').textContent=s.fng.label+' (0–100)';}
    else{$('fng').textContent='–';$('fnglabel').textContent='run training to load';}
    const hb=$('haltbanner');
    if(s.halted){hb.classList.remove('hidden');hb.textContent='⛔ TRADING HALTED by risk engine — '+s.halt_reason+'. Open positions still managed; new entries blocked.';}
    else hb.classList.add('hidden');
    const fresh=last.ts&&((Date.now()-new Date(last.ts+'Z').getTime())/1000 < 3*(s.poll_sec||60));
    $('dot').className='dot '+(fresh?'live':'bad');
    $('alive').className='pill '+(fresh?'on':'off');
    $('alive').textContent=(fresh?'● live':'● stale')+' · '+(CFG.trading_enabled?'trading':'paused');
    // positions
    const tb=$('postbl').querySelector('tbody');tb.innerHTML='';
    s.positions.forEach(p=>{const up=(p.mark-p.entry)*(p.side=='long'?1:-1);
      tb.insertAdjacentHTML('beforeend',
      `<tr><td><b>${p.symbol}</b></td><td><span class="badge ${p.side}">${p.side}</span></td><td>${f2(p.entry)}</td>`+
      `<td class="${p.mark==null?'':(up>=0?'pos':'neg')}">${f2(p.mark)}</td><td>${Number(p.qty).toFixed(4)}</td>`+
      `<td>$${f2(p.notional)}</td><td>${p.leverage}×</td><td>${f2(p.stop)}</td><td>${p.bars}</td></tr>`);});
    $('posempty').style.display=s.positions.length?'none':'block';
    if(!['INPUT','SELECT'].includes(document.activeElement.tagName))fillCfg();
    const eq=await get('/api/equity?limit=400');
    setChart('eqchart',[{name:'equity',color:getCss('--acc'),points:eq}],eq.length?eq[0].equity:null);
    const tr=await get('/api/trades?limit=50');const tt=$('trtbl').querySelector('tbody');tt.innerHTML='';
    tr.forEach(t=>{const c=t.pnl>0?'pos':(t.pnl<0?'neg':'');tt.insertAdjacentHTML('beforeend',
      `<tr><td>${(t.ts||'').replace('T',' ').slice(0,19)}</td><td>${t.symbol}</td><td>${t.action}</td><td>${f2(t.price)}</td><td>${f2(t.fee)}</td><td class="${c}">${sign(t.pnl)}${f2(t.pnl)}</td><td>${f2(t.balance)}</td><td class="muted">${t.reason||''}</td></tr>`);});
    try{renderModels(await get('/api/model'));}catch(_){}
  }catch(e){$('alive').className='pill off';$('alive').textContent='● no data';$('dot').className='dot bad';}
}

/* ---------- settings ---------- */
function buildInputs(container,obj,prefix){const c=$(container);if(c.dataset.built)return;c.innerHTML='';
  Object.keys(obj).forEach(k=>c.insertAdjacentHTML('beforeend',`<div><label>${k}</label><input id="${prefix}${k}" type="number" step="0.001"></div>`));c.dataset.built='1';}
function fillCfg(){if(!CFG)return;
  $('symbols').value=CFG.symbols.join(',');['timeframe','leverage','start_balance','taker_fee','max_position_pct','poll_sec'].forEach(k=>$(k).value=CFG[k]);
  $('allow_short').value=String(CFG.allow_short);
  const ss=$('strategy');if(!ss.dataset.built){ss.innerHTML='';STRATS.forEach(n=>ss.insertAdjacentHTML('beforeend',`<option value="${n}">${n}</option>`));ss.dataset.built='1';}
  if(document.activeElement!=ss)ss.value=CFG.strategy;
  const bsy=$('bt_symbol');if(!bsy.dataset.built){bsy.innerHTML='';CFG.symbols.forEach(n=>bsy.insertAdjacentHTML('beforeend',`<option value="${n}">${n}</option>`));bsy.dataset.built='1';}
  const bst=$('bt_strat');if(!bst.dataset.built){bst.innerHTML='';STRATS.forEach(n=>bst.insertAdjacentHTML('beforeend',`<option value="${n}">${n}</option>`));bst.value=CFG.strategy;bst.dataset.built='1';}
  const btf=$('bt_tf');if(!btf.dataset.built){btf.value=CFG.timeframe;btf.dataset.built='1';}
  const bbal=$('bt_bal');if(!bbal.dataset.built){bbal.value=CFG.start_balance;bbal.dataset.built='1';}
  const otf=$('op_tf');if(otf&&!otf.dataset.built){otf.value=CFG.timeframe;otf.dataset.built='1';}
  const ost=$('op_strat');if(ost&&!ost.dataset.built){ost.innerHTML='';STRATS.forEach(n=>ost.insertAdjacentHTML('beforeend',`<option value="${n}">${n}</option>`));ost.value=CFG.strategy;ost.dataset.built='1';}
  buildInputs('risk',CFG.risk,'r_');buildInputs('params',CFG.params,'p_');
  Object.keys(CFG.risk).forEach(k=>{const el=$('r_'+k);if(el&&document.activeElement!=el)el.value=CFG.risk[k];});
  Object.keys(CFG.params).forEach(k=>{const el=$('p_'+k);if(el&&document.activeElement!=el)el.value=CFG.params[k];});}
async function saveCfg(){const body={symbols:$('symbols').value,allow_short:$('allow_short').value,strategy:$('strategy').value,params:{},risk:{}};
  ['timeframe','leverage','start_balance','taker_fee','max_position_pct','poll_sec'].forEach(k=>body[k]=$(k).value);
  Object.keys(CFG.params).forEach(k=>body.params[k]=$('p_'+k).value);Object.keys(CFG.risk).forEach(k=>body.risk[k]=$('r_'+k).value);
  const r=await post('/api/config',body);$('savemsg').textContent=r.ok?'saved ✓':('error: '+(r._err||''));setTimeout(()=>$('savemsg').textContent='',2800);refresh();}
async function ctl(a){await post('/api/control',{action:a});refresh();}

/* ---------- backtest ---------- */
const tile=(k,v,cls,d)=>`<div class="tile"><div class="k">${k}</div><div class="v ${cls||''}">${v}</div><div class="d">${d||''}</div></div>`;
function btForm(){return {symbol:$('bt_symbol').value,timeframe:$('bt_tf').value,bars:$('bt_bars').value,start_balance:$('bt_bal').value};}
function renderSingle(r,title,extraNote){
  const s=r.summary,ret=s.total_return_pct,col=SCOLOR[s.strategy]||getCss('--acc');
  $('bt_compare').classList.add('hidden');$('bt_single').classList.remove('hidden');
  $('bt_title').textContent=title;
  $('bt_period').textContent=`${s.symbol} · ${s.timeframe} · ${s.candles} bars · ${(s.period_start||'').slice(0,10)} → ${(s.period_end||'').slice(0,10)}`+(s.funding_modeled?' · funding modeled':'');
  setChart('bt_chart',[{name:s.strategy,color:col,points:r.equity_curve}],s.start_balance);
  $('bt_legend').innerHTML=`<div class="it"><span class="sw" style="background:${col}"></span>${s.strategy}</div>`;
  $('bt_tiles').innerHTML=[
    tile('Total return',sign(ret)+ret+'%',ret>=0?'pos':'neg','vs start'),
    tile('Final equity','$'+f2(s.final_equity)),
    tile('Trades',s.n_trades,'',s.n_trades>300?'high — fee drag':''),
    tile('Win rate',s.win_rate_pct==null?'–':s.win_rate_pct+'%'),
    tile('Profit factor',s.profit_factor==null?'–':s.profit_factor,s.profit_factor>=1?'pos':'neg',s.profit_factor>=1.5?'good':''),
    tile('Max drawdown',s.max_drawdown_pct+'%','neg','worst drop'),
    tile('Sharpe (ann.)',s.sharpe==null?'–':s.sharpe,s.sharpe>=1?'pos':'',''),
    tile('Avg trade',s.avg_trade==null?'–':'$'+f2(s.avg_trade),s.avg_trade>=0?'pos':'neg'),
  ].join('');
  $('bt_verdict').innerHTML=(extraNote?extraNote+'<br>':'')+verdict(s);
  $('bt_verdict').style.borderColor=(ret>=0?'var(--pos)':'var(--neg)');
  $('bt_results').classList.remove('hidden');
}
async function runBacktest(){const btn=event&&event.target;if(btn)btn.disabled=true;$('bt_msg').textContent='fetching candles + simulating…';
  try{
    const r=await post('/api/backtest',{...btForm(),strategy:$('bt_strat').value});
    if(!r||!r.summary){$('bt_msg').textContent='error: '+(r._err||'no result');if(btn)btn.disabled=false;return;}
    renderSingle(r,'Single run · '+r.summary.strategy);$('bt_msg').textContent='done ✓';
  }catch(e){$('bt_msg').textContent='error';}
  if(btn)btn.disabled=false;}
async function runPortfolio(){const btn=event&&event.target;if(btn)btn.disabled=true;
  $('bt_msg').textContent='simulating all symbols on one account (with funding)…';
  try{
    const r=await post('/api/portfolio_backtest',{timeframe:$('bt_tf').value,bars:$('bt_bars').value,start_balance:$('bt_bal').value});
    if(!r||!r.summary){$('bt_msg').textContent='error: '+(r._err||'no result');if(btn)btn.disabled=false;return;}
    renderSingle(r,'Portfolio · matches live','<b>This mirrors the live bot:</b> all symbols, one shared account, risk caps, and funding — the number to trust, not a single-symbol run.');
    $('bt_msg').textContent='done ✓';
  }catch(e){$('bt_msg').textContent='error';}
  if(btn)btn.disabled=false;}
function verdict(s){
  const good=s.profit_factor!=null&&s.profit_factor>=1.3&&s.total_return_pct>0;
  const ok=s.total_return_pct>0;
  const msg=good?'✅ Profitable with a healthy profit factor over this window. Still validate on other windows before trusting it.'
    :ok?'🟡 Positive return but a thin profit factor — fragile; test more windows.'
    :'🔻 Lost money over this window. On the current params this strategy had no edge here.';
  return `<b>Verdict:</b> ${msg}`;}
async function compareAll(){const btn=event&&event.target;if(btn)btn.disabled=true;
  const names=STRATS.filter(n=>['ensemble','ml_meta','ml_dir','breakout'].includes(n));
  $('bt_msg').textContent='running '+names.length+' strategies…';
  try{
    const rs=await Promise.all(names.map(n=>post('/api/backtest',{...btForm(),strategy:n}).catch(()=>null)));
    const ok=rs.map((r,i)=>({n:names[i],r})).filter(o=>o.r&&o.r.summary);
    if(!ok.length){$('bt_msg').textContent='error: no results';if(btn)btn.disabled=false;return;}
    $('bt_single').classList.add('hidden');$('bt_compare').classList.remove('hidden');
    const first=ok[0].r.summary;
    $('bt_title').textContent='Compare · '+first.symbol;
    $('bt_period').textContent=`${first.symbol} · ${first.timeframe} · ${first.candles} candles · ${(first.period_start||'').slice(0,10)} → ${(first.period_end||'').slice(0,10)}`;
    const series=ok.map(o=>({name:o.n,color:SCOLOR[o.n]||getCss('--mut'),points:o.r.equity_curve}));
    setChart('bt_chart',series,first.start_balance);
    $('bt_legend').innerHTML=series.map(s=>`<div class="it"><span class="sw" style="background:${s.color}"></span>${s.name}</div>`).join('');
    // table sorted by return desc
    ok.sort((a,b)=>b.r.summary.total_return_pct-a.r.summary.total_return_pct);
    const tb=$('bt_cmptbl').querySelector('tbody');tb.innerHTML='';
    ok.forEach((o,i)=>{const s=o.r.summary,ret=s.total_return_pct;
      tb.insertAdjacentHTML('beforeend',`<tr><td><span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${SCOLOR[o.n]||getCss('--mut')};margin-right:6px"></span><b>${o.n}</b>${i===0?' <span class="tag">best</span>':''}</td>`+
        `<td class="${ret>=0?'pos':'neg'}">${sign(ret)}${ret}%</td><td>${s.n_trades}</td><td>${s.win_rate_pct==null?'–':s.win_rate_pct+'%'}</td>`+
        `<td class="${s.profit_factor>=1?'pos':'neg'}">${s.profit_factor==null?'–':s.profit_factor}</td><td class="neg">${s.max_drawdown_pct}%</td><td>${s.sharpe==null?'–':s.sharpe}</td></tr>`);});
    $('bt_results').classList.remove('hidden');$('bt_msg').textContent='done ✓';
  }catch(e){$('bt_msg').textContent='error';}
  if(btn)btn.disabled=false;}

/* ---------- ML lab ---------- */
function renderModels(d){$('ml_tf').textContent='· timeframe '+(d.timeframe||'');
  const tb=$('mltbl').querySelector('tbody');tb.innerHTML='';let any=false;
  (d.models||[]).forEach(m=>{const o=m.oos||{};const trained=m.trained_at?m.trained_at.replace('T',' ').slice(0,16):'–';
    if(m.trained_at)any=true;const ret=o.total_return_pct;
    const skill=('auc'in o&&o.auc!=null)?o.auc:(('accuracy'in o)?o.accuracy:null);
    const mcol=m.mode==='meta'?getCss('--s2'):getCss('--s3');
    tb.insertAdjacentHTML('beforeend',
      `<tr><td><b>${m.symbol}</b></td><td><span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${mcol};margin-right:6px"></span>ml_${m.mode}</td>`+
      `<td class="muted">${trained}</td><td>${m.n_samples||'–'}</td>`+
      `<td class="${ret>=0?'pos':'neg'}">${ret==null?'–':sign(ret)+ret+'%'}</td>`+
      `<td>${o.profit_factor==null?'–':o.profit_factor}</td><td class="neg">${o.max_drawdown_pct==null?'–':o.max_drawdown_pct+'%'}</td>`+
      `<td class="${skill!=null&&skill>0.5?'pos':'muted'}">${skill==null?'–':skill}</td></tr>`);});
  $('mltbl').style.display=any?'table':'none';$('mlempty').style.display=any?'none':'block';}
async function runTrain(){const btn=event&&event.target;if(btn)btn.disabled=true;$('ml_msg').textContent='backfilling + training… this can take a minute';
  try{const r=await post('/api/mltrain',{});
    if(r&&r.ok){$('ml_msg').textContent='done ✓';renderModels(await get('/api/model'));}
    else $('ml_msg').textContent='error: '+(r._err||'failed');
  }catch(e){$('ml_msg').textContent='error';}if(btn)btn.disabled=false;}

let REC=null;
function renderRec(rec){REC=rec;const b=$('rec_body');
  if(!rec){b.className='muted';b.textContent='no recommendation yet — run it, or wait for the weekly schedule.';$('rec_apply').classList.add('hidden');return;}
  b.className='';
  const rows=(rec.ranking||[]).map((r,i)=>`<tr><td><b>${r.strategy}</b>${i===0?' <span class="tag">top</span>':''}</td>`+
    `<td class="${r.avg_return>=0?'pos':'neg'}">${sign(r.avg_return)}${r.avg_return}%</td><td>${r.avg_pf==null?'–':r.avg_pf}</td><td class="neg">${r.avg_maxdd}%</td></tr>`).join('');
  b.innerHTML=`<div class="muted" style="margin-bottom:8px">generated ${(rec.generated_at||'').replace('T',' ').slice(0,16)} UTC · tf ${rec.timeframe} · OOS avg across symbols</div>`+
    `<table><thead><tr><th>Strategy</th><th>Avg return</th><th>Avg PF</th><th>Avg maxDD</th></tr></thead><tbody>${rows}</tbody></table>`+
    `<div class="verdict" style="margin-top:12px;border-color:${rec.change_suggested?'var(--acc)':'var(--line)'}">`+
    `<b>Current:</b> <code>${rec.current_strategy}</code> &nbsp; <b>Recommended:</b> <code>${rec.recommended}</code><br>${rec.note}</div>`;
  const canApply=rec.recommended&&rec.recommended!==rec.current_strategy;
  $('rec_apply').classList.toggle('hidden',!canApply);
  if(canApply)$('rec_apply').textContent='✔ Apply '+rec.recommended;}
async function runAutopilot(){const btn=event&&event.target;if(btn)btn.disabled=true;
  $('ap_msg').textContent='backfilling + retraining + ranking… this can take a couple minutes';
  try{const r=await post('/api/autopilot',{});
    if(r&&r.ok){$('ap_msg').textContent='done ✓';renderRec(r.recommendation);renderModels(await get('/api/model'));}
    else $('ap_msg').textContent='error: '+(r._err||'failed');
  }catch(e){$('ap_msg').textContent='error';}if(btn)btn.disabled=false;}
async function applyRec(){if(!REC||!REC.recommended)return;
  if(!confirm('Switch the LIVE paper bot to '+REC.recommended+'? (takes effect next loop)'))return;
  const btn=event&&event.target;if(btn)btn.disabled=true;$('ap_msg').textContent='applying…';
  const r=await post('/api/config',{strategy:REC.recommended});
  $('ap_msg').textContent=r.ok?('live strategy → '+REC.recommended+' ✓'):('error: '+(r._err||''));
  refresh();if(btn)btn.disabled=false;}

let OPT=null;
function hcell(o,k,suf){const v=o&&o[k];return v==null?'–':(v+(suf||''));}
function renderOpt(b){OPT=b;const body=$('op_body');
  if(!b){body.className='muted';body.textContent='no optimization yet — run it.';$('op_apply').classList.add('hidden');return;}
  body.className='';
  const hb=b.holdout_best||{},hc=b.holdout_current||{};
  const warn=b.overfit_warning?`<div class="verdict" style="border-color:var(--neg);margin-bottom:10px">⚠ <b>Overfit:</b> great in-study but negative on the holdout Optuna never saw — <b>do not apply.</b></div>`:'';
  const agree=(hb.return!=null&&hb.return>0&&!b.overfit_warning);
  body.innerHTML=warn+
    `<div class="muted" style="margin-bottom:8px">${(b.generated_at||'').replace('T',' ').slice(0,16)} UTC · ${b.strategy} · ${b.timeframe} · ${b.n_trials} trials · study median Calmar ${b.study_metric_median_calmar}</div>`+
    `<table><thead><tr><th>On the untouched holdout</th><th>Return</th><th>PF</th><th>Max DD</th><th>Trades</th></tr></thead><tbody>`+
    `<tr><td><b>Best found</b></td><td class="${hb.return>=0?'pos':'neg'}">${hcell(hb,'return','%')}</td><td>${hcell(hb,'pf')}</td><td class="neg">${hcell(hb,'maxdd','%')}</td><td>${hcell(hb,'trades')}</td></tr>`+
    `<tr><td class="muted">Current config</td><td class="${hc.return>=0?'pos':'neg'}">${hcell(hc,'return','%')}</td><td>${hcell(hc,'pf')}</td><td class="neg">${hcell(hc,'maxdd','%')}</td><td>${hcell(hc,'trades')}</td></tr>`+
    `</tbody></table>`+
    `<div class="verdict" style="margin-top:12px;border-color:${agree?'var(--pos)':'var(--line)'}">`+
    (agree?'✅ Holds up out-of-sample. Worth applying and watching live.':'🟡 Holdout is weak — the search did not find a robust edge. Applying is not recommended.')+'</div>'+
    `<div class="muted" style="margin-top:8px;font-size:11.5px">best: lev ${b.best_risk&&b.best_risk.max_leverage} · enter_thr ${b.best_params&&b.best_params.enter_thr} · stop_atr ${b.best_params&&b.best_params.stop_atr} · max_hold ${b.best_params&&b.best_params.max_hold_bars}</div>`;
  $('op_apply').classList.toggle('hidden',!(hb.return!=null));
  $('op_apply').textContent=agree?'✔ Apply best params':'✔ Apply anyway';}
async function runOptimize(){const btn=event&&event.target;if(btn)btn.disabled=true;
  $('op_msg').textContent='searching… this can take a few minutes';
  try{const r=await post('/api/optimize',{timeframe:$('op_tf').value,strategy:$('op_strat').value,trials:$('op_trials').value});
    if(r&&r.ok){$('op_msg').textContent='done ✓';renderOpt(r.best);}
    else $('op_msg').textContent='error: '+(r._err||'failed');
  }catch(e){$('op_msg').textContent='error';}if(btn)btn.disabled=false;}
async function applyOpt(){if(!OPT||!OPT.best_params)return;
  if(!confirm('Apply the optimized params + risk to the LIVE bot ('+OPT.strategy+' @ '+OPT.timeframe+')? Takes effect next loop.'))return;
  const btn=event&&event.target;if(btn)btn.disabled=true;$('op_msg').textContent='applying…';
  const r=await post('/api/config',{strategy:OPT.strategy,timeframe:OPT.timeframe,params:OPT.best_params,risk:OPT.best_risk});
  $('op_msg').textContent=r.ok?'applied to live ✓':('error: '+(r._err||''));refresh();if(btn)btn.disabled=false;}

let AA_ENABLED=false;
function renderAuto(d){AA_ENABLED=!!d.enabled;
  $('aa_toggle').textContent='Auto-apply: '+(AA_ENABLED?'ON':'OFF');
  $('aa_toggle').className=AA_ENABLED?'primary':'';
  const s=d.status,b=$('aa_body');
  if(!s){b.style.display='none';return;}
  b.style.display='block';
  const dec=s.decision,col=dec==='applied'?'var(--pos)':(dec==='would_apply'?'var(--acc)':'var(--line)');
  b.style.borderColor=col;
  b.innerHTML=`<b>Last nightly check:</b> ${(s.generated_at||'').replace('T',' ').slice(0,16)} UTC · `+
    `<b>${dec.replace('_',' ').toUpperCase()}</b><br>${s.reason||''}`+
    (s.applied_at?`<br><span class="muted">last applied: ${s.applied_at.replace('T',' ').slice(0,16)} UTC</span>`:'');}
async function toggleAutoApply(){const btn=event&&event.target;if(btn)btn.disabled=true;
  const r=await post('/api/config',{autopilot_apply:(!AA_ENABLED)});
  if(r.ok){AA_ENABLED=!AA_ENABLED;try{renderAuto(await get('/api/autoapply'));}catch(_){}}
  if(btn)btn.disabled=false;}

window.addEventListener('resize',()=>Object.keys(CHARTS).forEach(id=>render(id,null)));
tab('dash');refresh();setInterval(refresh,15000);
try{get('/api/recommendation').then(d=>renderRec(d.recommendation)).catch(()=>{});}catch(_){}
try{get('/api/optimize').then(d=>renderOpt(d.best)).catch(()=>{});}catch(_){}
try{get('/api/autoapply').then(d=>renderAuto(d)).catch(()=>{});}catch(_){}
</script></body></html>"""
