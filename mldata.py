"""
Phase 0 — historical OHLCV pipeline.

Backfills deep candle history from Kraken Futures (public, no keys) into data/history.db so the
ML training (mltrain.py) has enough samples. Idempotent + incremental. Also the feasibility
probe: the CLI prints how many candles we actually have per (symbol, timeframe).

Stored in a SEPARATE SQLite file from the live ledger (paper.db) to avoid any contention with
the running bot.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time

import ccxt
import pandas as pd

DATA_DIR = os.getenv("KF_DATA_DIR", "data")
HISTORY_DB = os.path.join(DATA_DIR, "history.db")
PAGE_LIMIT = 720  # candles per request

# timeframe -> milliseconds
_TF_MS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
          "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200, "1d": 86400}


def _tf_ms(timeframe: str) -> int:
    if timeframe not in _TF_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return _TF_MS[timeframe] * 1000


def _db() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    c = sqlite3.connect(HISTORY_DB, timeout=10.0)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS ohlcv(
        symbol TEXT, timeframe TEXT, ts INTEGER,
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY(symbol, timeframe, ts))""")
    return c


def _last_ts(c: sqlite3.Connection, symbol: str, timeframe: str) -> int | None:
    row = c.execute("SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?",
                    (symbol, timeframe)).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def count(symbol: str, timeframe: str) -> int:
    with _db() as c:
        row = c.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?",
                        (symbol, timeframe)).fetchone()
        return int(row[0]) if row else 0


def backfill(symbol: str, timeframe: str, max_candles: int = 20000,
             exchange: ccxt.Exchange | None = None) -> int:
    """Grab the newest candles, then page BACKWARD until the exchange runs out of history or
    max_candles is reached. Backward paging works for every timeframe (coarse TFs whose full
    max_candles span predates the market's listing no longer start on an empty page). Idempotent
    via INSERT OR REPLACE; the newest fetch each run also tops up recent candles. Returns the
    total rows stored for this (symbol, tf)."""
    ex = exchange or ccxt.krakenfutures({"enableRateLimit": True})
    ex.load_markets()
    if symbol not in ex.markets:
        raise ValueError(f"unknown symbol: {symbol}")
    step = _tf_ms(timeframe)

    def _store(c, page) -> int:
        rows = [(symbol, timeframe, int(r[0]), float(r[1]), float(r[2]),
                 float(r[3]), float(r[4]), float(r[5])) for r in page]
        if rows:
            c.executemany("""INSERT OR REPLACE INTO ohlcv
                (symbol,timeframe,ts,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?,?)""", rows)
            c.commit()
        return len(rows)

    with _db() as c:
        try:
            newest = ex.fetch_ohlcv(symbol, timeframe, limit=PAGE_LIMIT)   # most-recent page
        except Exception as e:
            print(f"  newest fetch error: {e}")
            newest = []
        _store(c, newest)
        row = c.execute("SELECT MIN(ts) FROM ohlcv WHERE symbol=? AND timeframe=?",
                        (symbol, timeframe)).fetchone()
        oldest = int(row[0]) if row and row[0] is not None else ex.milliseconds()

        while count(symbol, timeframe) < max_candles:
            since = oldest - PAGE_LIMIT * step
            try:
                page = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=PAGE_LIMIT)
            except Exception as e:
                print(f"  fetch error @ since={since}: {e}; stopping")
                break
            page = [r for r in page if r[0] < oldest]        # only candles older than we have
            if not page:
                break                                        # exchange has no more history
            _store(c, page)
            new_oldest = min(int(r[0]) for r in page)
            if new_oldest >= oldest:                         # no backward progress -> done
                break
            oldest = new_oldest
            time.sleep(max(ex.rateLimit, 200) / 1000.0)

    return count(symbol, timeframe)


def load_history(symbol: str, timeframe: str) -> pd.DataFrame:
    """Return a UTC-indexed OHLCV frame (same shape as backtest.fetch_ohlcv)."""
    with _db() as c:
        rows = c.execute(
            "SELECT ts,open,high,low,close,volume FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts",
            (symbol, timeframe)).fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop(columns=["ts"])


def _fmt_ts(ms: int | None) -> str:
    return pd.to_datetime(ms, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M") if ms else "-"


def main():
    ap = argparse.ArgumentParser(description="Backfill Kraken Futures OHLCV history")
    ap.add_argument("--symbol", help="single symbol, e.g. SOL/USD:USD")
    ap.add_argument("--timeframe", help="e.g. 1h, 5m")
    ap.add_argument("--max", type=int, default=20000, help="target candle count (cold start)")
    args = ap.parse_args()

    if args.symbol and args.timeframe:
        symbols, tf = [args.symbol], args.timeframe
    else:
        from config import load_config
        cfg = load_config()
        symbols, tf = cfg["symbols"], (args.timeframe or cfg["timeframe"])

    ex = ccxt.krakenfutures({"enableRateLimit": True})
    print(f"Backfilling {symbols} @ {tf} (target {args.max}) -> {HISTORY_DB}\n")
    print(f"{'symbol':<16} {'tf':<4} {'rows':>7}  {'first':<16}  {'last':<16}")
    for sym in symbols:
        try:
            backfill(sym, tf, args.max, exchange=ex)
        except Exception as e:
            print(f"{sym:<16} {tf:<4}  ERROR: {e}")
            continue
        with _db() as c:
            n = count(sym, tf)
            lo = c.execute("SELECT MIN(ts) FROM ohlcv WHERE symbol=? AND timeframe=?", (sym, tf)).fetchone()[0]
            hi = c.execute("SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?", (sym, tf)).fetchone()[0]
        print(f"{sym:<16} {tf:<4} {n:>7}  {_fmt_ts(lo):<16}  {_fmt_ts(hi):<16}")


if __name__ == "__main__":
    main()
