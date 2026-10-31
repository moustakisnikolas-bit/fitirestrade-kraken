"""
Phase 5/7 — market & derivatives-positioning sentiment.

Three FREE, HISTORICAL (therefore backtestable) sentiment sources, strongest first for a futures bot:

  1. FUNDING RATE (per-symbol, ~hourly, ccxt.krakenfutures.fetch_funding_rate_history) — the cost
     longs pay shorts. Positive = crowd is long (bullish positioning, squeeze risk at extremes);
     negative = crowd is short. The sharpest, most specific crowd-positioning signal we have.
  2. SPOT-PERP BASIS (per-symbol, perp_close/spot_close-1 via ccxt.kraken spot OHLCV) — the perp
     premium. Positive = perp bid over spot = bullish leverage demand; negative = discount/bearish.
  3. FEAR & GREED (market-wide, daily, alternative.me) — slow risk-appetite backdrop.

Open interest is intentionally excluded: ccxt.krakenfutures exposes no (historical) OI, so it could
not be backtested, and we never promote an un-validated signal.

Everything is stored in history.db and composited (per symbol, per bar) into a bullish score in
[-1,1] by `build_sentiment_columns`, consumed by the `ensemble_sent` strategy (strategies.py). Live
freshness comes from `topup()` (cheap, TTL-throttled); the nightly refresh does the deep backfill.
Funding/basis use ccxt; F&G uses only the stdlib.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.request

import pandas as pd

HISTORY_DB = os.path.join(os.getenv("KF_DATA_DIR", "data"), "history.db")
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"

# minutes per timeframe (spot basis is sampled on the trading timeframe)
_TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
           "4h": 240, "6h": 360, "12h": 720, "1d": 1440}

# perp symbol (SOL/USD:USD) -> kraken spot symbol (SOL/USD)
def _spot_symbol(perp: str) -> str:
    return perp.split(":")[0]


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(HISTORY_DB), exist_ok=True)
    c = sqlite3.connect(HISTORY_DB, timeout=10.0)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS fng(ts INTEGER PRIMARY KEY, value REAL)")
    c.execute("CREATE TABLE IF NOT EXISTS funding(symbol TEXT, ts INTEGER, rate REAL, PRIMARY KEY(symbol, ts))")
    c.execute("CREATE TABLE IF NOT EXISTS spot(symbol TEXT, timeframe TEXT, ts INTEGER, close REAL, "
              "PRIMARY KEY(symbol, timeframe, ts))")
    return c


def backfill_fng() -> int:
    """Fetch the full Fear & Greed history (one small request) and store it. Idempotent."""
    req = urllib.request.Request(FNG_URL, headers={"User-Agent": "kfpaper/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    rows = [(int(d["timestamp"]), float(d["value"])) for d in data.get("data", []) if d.get("value")]
    if rows:
        with _db() as c:
            c.executemany("INSERT OR REPLACE INTO fng(ts,value) VALUES(?,?)", rows)
            c.commit()
    with _db() as c:
        return int(c.execute("SELECT COUNT(*) FROM fng").fetchone()[0])


def load_fng_series() -> pd.Series:
    """Daily Fear & Greed value (0..100), indexed by tz-aware UTC timestamp, ascending."""
    try:
        with _db() as c:
            rows = c.execute("SELECT ts,value FROM fng ORDER BY ts").fetchall()
    except Exception:
        rows = []
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows], unit="s", utc=True)
    return pd.Series([r[1] for r in rows], index=idx)


def align_fng(index) -> pd.Series:
    """Forward-fill the daily F&G onto an arbitrary (intraday) candle index, normalised to [-1,1]
    (0 = neutral/50). Missing (before history / unavailable) -> 0.0 so it is safe as a feature."""
    s = load_fng_series()
    if s.empty:
        return pd.Series(0.0, index=index)
    aligned = s.reindex(s.index.union(index)).sort_index().ffill().reindex(index)
    return ((aligned - 50.0) / 50.0).fillna(0.0)


def current_fng() -> dict | None:
    """Latest reading for the dashboard: {value, label} or None."""
    s = load_fng_series()
    if s.empty:
        return None
    v = float(s.iloc[-1])
    label = ("extreme fear" if v < 25 else "fear" if v < 45 else "neutral"
             if v < 55 else "greed" if v < 75 else "extreme greed")
    return {"value": round(v), "label": label}


# --------------------------------------------------------------------------- funding rate
def backfill_funding(symbols: list[str], lookback_days: int = 120, exchange=None) -> dict:
    """Page funding-rate history forward from `lookback_days` ago into history.db. Idempotent."""
    import ccxt
    ex = exchange or ccxt.krakenfutures({"enableRateLimit": True})
    ex.load_markets()
    out = {}
    since0 = ex.milliseconds() - int(lookback_days) * 86400_000
    for sym in symbols:
        if sym not in ex.markets:
            out[sym] = 0; continue
        since, seen = since0, 0
        with _db() as c:
            while True:
                try:
                    page = ex.fetch_funding_rate_history(sym, since=since, limit=500)
                except Exception as e:
                    print(f"  funding {sym} error: {e}"); break
                page = [r for r in page if r.get("timestamp") and r.get("fundingRate") is not None]
                if not page:
                    break
                rows = [(sym, int(r["timestamp"]), float(r["fundingRate"])) for r in page]
                c.executemany("INSERT OR REPLACE INTO funding(symbol,ts,rate) VALUES(?,?,?)", rows)
                c.commit()
                seen += len(rows)
                nxt = int(page[-1]["timestamp"]) + 1
                if nxt <= since:
                    break
                since = nxt
                if len(page) < 500:
                    break
                time.sleep(max(ex.rateLimit, 200) / 1000.0)
            out[sym] = int(c.execute("SELECT COUNT(*) FROM funding WHERE symbol=?", (sym,)).fetchone()[0])
    return out


def load_funding(symbol: str) -> pd.Series:
    """Funding rate per symbol, indexed by tz-aware UTC timestamp, ascending."""
    try:
        with _db() as c:
            rows = c.execute("SELECT ts,rate FROM funding WHERE symbol=? ORDER BY ts", (symbol,)).fetchall()
    except Exception:
        rows = []
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
    return pd.Series([r[1] for r in rows], index=idx)


# --------------------------------------------------------------------------- spot (for basis)
def backfill_spot(symbols: list[str], timeframe: str, max_candles: int = 8000, exchange=None) -> dict:
    """Backfill Kraken SPOT close prices (for the perp-spot basis) into history.db, paging backward."""
    import ccxt
    ex = exchange or ccxt.kraken({"enableRateLimit": True})
    ex.load_markets()
    step = _TF_MIN.get(timeframe, 15) * 60_000
    out = {}
    for perp in symbols:
        s = _spot_symbol(perp)
        if s not in ex.markets:
            out[perp] = 0; continue
        with _db() as c:
            def n():
                return int(c.execute("SELECT COUNT(*) FROM spot WHERE symbol=? AND timeframe=?",
                                     (perp, timeframe)).fetchone()[0])
            def store(page):
                rows = [(perp, timeframe, int(r[0]), float(r[4])) for r in page]
                if rows:
                    c.executemany("INSERT OR REPLACE INTO spot(symbol,timeframe,ts,close) VALUES(?,?,?,?)", rows)
                    c.commit()
            try:
                store(ex.fetch_ohlcv(s, timeframe, limit=720))
            except Exception as e:
                print(f"  spot {s} newest error: {e}")
            row = c.execute("SELECT MIN(ts) FROM spot WHERE symbol=? AND timeframe=?", (perp, timeframe)).fetchone()
            oldest = int(row[0]) if row and row[0] is not None else ex.milliseconds()
            while n() < max_candles:
                since = oldest - 720 * step
                try:
                    page = ex.fetch_ohlcv(s, timeframe, since=since, limit=720)
                except Exception as e:
                    print(f"  spot {s} error: {e}"); break
                page = [r for r in page if r[0] < oldest]
                if not page:
                    break
                store(page)
                no = min(int(r[0]) for r in page)
                if no >= oldest:
                    break
                oldest = no
                time.sleep(max(ex.rateLimit, 200) / 1000.0)
            out[perp] = n()
    return out


def load_spot(symbol: str, timeframe: str) -> pd.Series:
    """Spot close series for a perp symbol, UTC-indexed ascending."""
    try:
        with _db() as c:
            rows = c.execute("SELECT ts,close FROM spot WHERE symbol=? AND timeframe=? ORDER BY ts",
                             (symbol, timeframe)).fetchall()
    except Exception:
        rows = []
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
    return pd.Series([r[1] for r in rows], index=idx)


# --------------------------------------------------------------------------- live top-up (throttled)
_LAST_TOPUP: dict = {}


def topup(symbols: list[str], timeframe: str, kf_exchange=None, ttl_sec: int = 300) -> None:
    """Cheap, TTL-throttled refresh of the newest funding/spot/F&G so the LIVE bot reads fresh
    sentiment from history.db (same store the backtester reads). Fully best-effort: any failure is
    swallowed and leaves the strategy to degrade to neutral. Safe to call every loop."""
    now = time.time()
    if now - _LAST_TOPUP.get("t", 0) < ttl_sec:
        return
    _LAST_TOPUP["t"] = now
    # funding (newest page per symbol)
    try:
        ex = kf_exchange
        if ex is None:
            import ccxt; ex = ccxt.krakenfutures({"enableRateLimit": True}); ex.load_markets()
        for sym in symbols:
            try:
                page = ex.fetch_funding_rate_history(sym, limit=100)
                rows = [(sym, int(r["timestamp"]), float(r["fundingRate"]))
                        for r in page if r.get("timestamp") and r.get("fundingRate") is not None]
                if rows:
                    with _db() as c:
                        c.executemany("INSERT OR REPLACE INTO funding(symbol,ts,rate) VALUES(?,?,?)", rows)
                        c.commit()
            except Exception:
                pass
    except Exception:
        pass
    # spot (newest page per symbol) — kraken spot is a separate exchange
    try:
        import ccxt
        sx = ccxt.kraken({"enableRateLimit": True}); sx.load_markets()
        for perp in symbols:
            try:
                page = sx.fetch_ohlcv(_spot_symbol(perp), timeframe, limit=200)
                rows = [(perp, timeframe, int(r[0]), float(r[4])) for r in page]
                if rows:
                    with _db() as c:
                        c.executemany("INSERT OR REPLACE INTO spot(symbol,timeframe,ts,close) VALUES(?,?,?,?)", rows)
                        c.commit()
            except Exception:
                pass
    except Exception:
        pass
    # F&G (daily; the backfill endpoint is tiny)
    try:
        backfill_fng()
    except Exception:
        pass


# --------------------------------------------------------------------------- composite
def _zscore(s: pd.Series, win: int) -> pd.Series:
    win = max(5, int(win))
    m = s.rolling(win, min_periods=max(5, win // 3)).mean()
    sd = s.rolling(win, min_periods=max(5, win // 3)).std(ddof=0)
    return ((s - m) / (sd + 1e-12)).fillna(0.0)


def build_sentiment_columns(symbol: str | None, timeframe: str | None, index,
                            close, p: dict | None = None) -> pd.DataFrame:
    """Per-bar sentiment aligned to `index` (the candle times). Columns:
      funding, funding_z, basis, basis_z, fng_n, sent_score  (sent_score in [-1,1], + = bullish).
    Robust: missing source -> that component is neutral (0); never raises."""
    import numpy as np
    p = p or {}
    win = int(p.get("sent_z_win", 96))
    out = pd.DataFrame(index=index)

    # funding: ~hourly, forward-fill onto the (intraday) candle index
    fser = load_funding(symbol) if symbol else pd.Series(dtype=float)
    if not fser.empty:
        fa = fser.reindex(fser.index.union(index)).sort_index().ffill().reindex(index)
    else:
        fa = pd.Series(0.0, index=index)
    out["funding"] = fa.fillna(0.0)
    out["funding_z"] = _zscore(out["funding"], win)

    # basis = perp_close / spot_close - 1 (spot ffilled onto index)
    cser = pd.Series(list(close), index=index).astype(float)
    sser = load_spot(symbol, timeframe) if (symbol and timeframe) else pd.Series(dtype=float)
    if not sser.empty:
        sa = sser.reindex(sser.index.union(index)).sort_index().ffill().reindex(index)
        basis = (cser / sa - 1.0)
    else:
        basis = pd.Series(0.0, index=index)
    out["basis"] = basis.fillna(0.0)
    out["basis_z"] = _zscore(out["basis"], win)

    # F&G already normalised to [-1,1]
    out["fng_n"] = align_fng(index)

    wf = float(p.get("sent_w_funding", 0.5))
    wb = float(p.get("sent_w_basis", 0.3))
    wg = float(p.get("sent_w_fng", 0.2))
    tot = (wf + wb + wg) or 1.0
    raw = (wf * np.tanh(out["funding_z"] / 2.0)
           + wb * np.tanh(out["basis_z"] / 2.0)
           + wg * out["fng_n"])
    out["sent_score"] = (raw / tot).clip(-1.0, 1.0).fillna(0.0)
    return out


if __name__ == "__main__":
    import sys
    from config import load_config
    cfg = load_config()
    syms, tf = cfg["symbols"], cfg["timeframe"]
    n = backfill_fng()
    print(f"Fear & Greed daily points: {n} | current: {current_fng()}")
    print("funding:", backfill_funding(syms, lookback_days=int(sys.argv[1]) if len(sys.argv) > 1 else 120))
    print("spot   :", backfill_spot(syms, tf))
