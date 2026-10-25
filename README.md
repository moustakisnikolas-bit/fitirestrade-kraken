# kfpaper — Kraken Futures paper bot (standalone)

A **brand-new, fully isolated** paper-trading bot for **Kraken Futures** perpetuals. It has
**no API keys and no live-order code path** — it cannot place a real trade. It is deliberately
separate from the `autotrade` project: its own image, Docker network (`kfpaper_net`), volume
(`kfpaper_data`), and SQLite ledger. It does not read or touch the new-listing bot or any
autotrade container, DB, or config.

## What it does
- Polls public Kraken-Futures OHLCV for SOL/AVAX/LINK perps.
- Runs a pluggable, regime-aware ensemble strategy (`strategy.py`): mean-reversion + momentum,
  weighted by an ADX trend/range regime detector.
- Paper-trades long/short through `paper_engine.py`: Kraken-Futures taker fee (~0.05%), hourly
  funding accrual, leverage, approximate liquidation. All state in `data/paper.db`.

## Honest status
Backtests (see the autotrade `feat/kraken-ensemble` branch) showed this strategy class has **no
demonstrated edge** on these coins after fees, and Kraken funding on them is ~0. This bot is a
**safe sandbox** to test and auto-tune ideas with zero money at risk — not an income engine.
Real money stays out until a strategy clears out-of-sample backtests AND walk-forward paper.

## Run
```bash
docker compose up -d --build
docker logs -f kfpaper_bot
```

## Inspect the ledger
```bash
docker run --rm -v kfpaper_data:/d keinos/sqlite3 \
  sqlite3 /d/paper.db "SELECT ts,symbol,action,price,pnl,balance FROM trades ORDER BY id DESC LIMIT 20;"
```

## Config (env in docker-compose.yml)
`KF_SYMBOLS, KF_TIMEFRAME, KF_LEVERAGE, KF_START_BALANCE, KF_POLL_SEC, KF_TAKER_FEE,
KF_MAX_POSITION_PCT, KF_ALLOW_SHORT`.
