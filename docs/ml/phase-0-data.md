# Phase 0 — Historical data pipeline (`mldata.py`)

## Goal
Backfill and store deep historical OHLCV so training has enough samples. This phase is the
**feasibility gate**: Kraken Futures perps are relatively young and the API caps history, so the
first run must report how many candles we can actually get per symbol/timeframe.

## Storage schema — `data/history.db` (SQLite, stdlib only)
```sql
CREATE TABLE IF NOT EXISTS ohlcv(
  symbol    TEXT,
  timeframe TEXT,
  ts        INTEGER,   -- epoch ms, candle open time
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY(symbol, timeframe, ts)
);
```
`INSERT OR REPLACE` makes backfill idempotent; the PK dedupes overlapping pages.

## Public interface
```python
def backfill(symbol: str, timeframe: str, max_candles: int = 20000) -> int
    # Paginate ccxt.krakenfutures.fetch_ohlcv(symbol, tf, since=cursor, limit=...)
    # advancing `since` by the last page's final ts + 1 tf-interval, until no
    # progress or max_candles reached. Returns total rows now stored.

def load_history(symbol: str, timeframe: str) -> pandas.DataFrame
    # DataFrame indexed by UTC DatetimeIndex, columns open/high/low/close/volume
    # (same shape as bot.to_df / backtest.fetch_ohlcv output) for reuse downstream.

def count(symbol: str, timeframe: str) -> int
```

## Pagination details
- ccxt `krakenfutures.fetch_ohlcv` supports `since` (ms) + `limit`. Kraken Futures serves candles
  from the charts API; a single call returns up to ~`limit` (we request 720) and history depth is
  finite.
- Loop: start `since = None` (latest), then walk **backwards** is awkward with this API, so instead
  walk **forwards** from an early `since` (e.g. now − max_candles·tf) and page until caught up to
  now. Stop when a page returns < 2 new rows (no progress) or we reach `max_candles`.
- Respect rate limits (`enableRateLimit: True`); small sleep between pages.
- Incremental re-run: start `since = last_stored_ts + 1` so only new candles are fetched.

## CLI
```
python mldata.py                       # backfill all config symbols at config timeframe
python mldata.py --symbol SOL/USD:USD --timeframe 1h --max 20000
```
Prints a table of `symbol | timeframe | rows | first_ts | last_ts` — **this is the feasibility
readout**.

## Feasibility thresholds (rule of thumb)
- Triple-barrier training wants a few thousand labeled samples minimum. Target **≥ 5,000 candles**
  per (symbol, tf); **≥ 10,000** is comfortable.
- 1h: 10,000 candles ≈ 14 months. 5m: 10,000 ≈ 35 days. Shorter TFs accumulate faster but are
  noisier and fee-sensitive.

## Feasibility result (probed 2026-10-26)
`krakenfutures` 1h returns **20,000 candles per symbol back to 2024-07-15** (SOL/AVAX/LINK) — ~2.3
years, well above the ≥10k comfortable threshold. **Data is not a constraint**; the spot fallback
below is unnecessary for 1h. 5m can be backfilled the same way when needed (shallower calendar span).

## Fallback if perp history is sparse
- Supplement features from Kraken **spot** via `ccxt.kraken` (deeper history), mapping
  `SOL/USD:USD` → `SOL/USD`. Caveat: spot ≠ perp (no funding, slightly different prices), so spot
  is used only to extend *feature* history for training, never to simulate perp fills.
- Secondary option: accumulate forward over time (the live bot already polls; we can log candles).

## Edge cases
- Symbol not in `ex.markets` → raise clear error.
- Empty/short responses → stop paging, keep what we have.
- Clock/timezone: store epoch ms (UTC); loader converts to tz-aware UTC index.
- DB write under the live bot: separate DB file (`history.db`), so no contention with `paper.db`.

## Test steps
1. `docker exec -i kfpaper_bot python mldata.py --symbol SOL/USD:USD --timeframe 1h --max 20000`
2. Confirm it prints rows ≥ target and `data/history.db` exists in the volume.
3. `load_history` returns a clean, monotonic, de-duplicated frame.
