"""
Phase 5 — market sentiment features.

Primary source: the Crypto Fear & Greed Index (alternative.me) — FREE, with full daily history, so
it is backtestable (unlike live-only news/social feeds). Market-wide (BTC-centric) 0..100:
0 = extreme fear, 100 = extreme greed. Stored in history.db and exposed as a model feature
(normalised to [-1,1]) plus a current reading for the dashboard.

Uses only the stdlib (urllib) — no new dependency, so no image rebuild.
"""
from __future__ import annotations

import json
import os
import sqlite3
import urllib.request

import pandas as pd

HISTORY_DB = os.path.join(os.getenv("KF_DATA_DIR", "data"), "history.db")
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(HISTORY_DB), exist_ok=True)
    c = sqlite3.connect(HISTORY_DB, timeout=10.0)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS fng(ts INTEGER PRIMARY KEY, value REAL)")
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


if __name__ == "__main__":
    n = backfill_fng()
    cur = current_fng()
    print(f"Fear & Greed history stored: {n} daily points | current: {cur}")
