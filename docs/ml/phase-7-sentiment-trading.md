# Phase 7 — Specialized sentiment trading (derivatives positioning)

## What and why
Generic "sentiment" for crypto is usually social/news scraping — live-only, un-backtestable, noisy.
For a **futures** bot the sharpest, FREE, and (crucially) **historical/backtestable** sentiment is
**derivatives positioning**. We use three sources, strongest first, composited per symbol per bar:

| Signal | Source | Meaning | Coverage |
|--------|--------|---------|----------|
| **Funding rate** | `ccxt.krakenfutures.fetch_funding_rate_history` | longs-pay-shorts; + = crowd long (squeeze risk at extremes) | full (~120d, hourly) |
| **Spot-perp basis** | `perp_close / kraken_spot_close − 1` | perp premium = bullish leverage demand | live-forward (~7d; Kraken caps spot history) |
| **Fear & Greed** | alternative.me (daily) | slow market-wide risk appetite | full (years) |

Open interest is **excluded**: `ccxt.krakenfutures` exposes no historical OI, so it couldn't be
validated — and we never promote an un-validated signal.

## Composite
`sentiment.build_sentiment_columns()` aligns each source to the candle index and builds a bullish
score in `[-1, 1]`:
```
sent_score = ( w_f·tanh(funding_z/2) + w_b·tanh(basis_z/2) + w_g·fng_norm ) / (w_f+w_b+w_g)
```
`funding_z`/`basis_z` are rolling z-scores (`sent_z_win` bars); `tanh` squashes outliers. Weights
default 0.5 / 0.3 / 0.2 (`sent_w_funding/basis/fng`). Missing source → that term is neutral (0).

## How it trades — strategy `ensemble_sent`
A subclass of the ensemble that applies the composite three ways (all only ever make entries
easier-when-confirmed or smaller-when-risky; the hard stop/risk engine are untouched):
1. **TILT** — agreeing sentiment lowers the entry bar: `long_thr = enter_thr − sent_tilt·max(0, sent)`
   (mirror for shorts). Lets a milder ensemble signal fire when positioning confirms it.
2. **FADE** — if funding is crowded in the SAME direction as the trade (`|funding_z| ≥ sent_fade_z`),
   multiply size by `sent_fade_floor` (contrarian squeeze protection).
3. **SIZE** — `conf ×= 1 ± sent_size·|sent_score|` (up if sentiment agrees, down if it disagrees);
   `conf` flows into `RiskEngine.size`.
Optional hard `sent_veto` skips entries whose sentiment strongly opposes. All params live in
`params` (`sent_*`), persist in config.json, tunable from the panel. Degrades to the plain ensemble
if data is missing or `sent_enabled=0`.

## Data freshness
- **Backtest** reads funding/spot/fng from `history.db` (deep backfill via `sentiment.py` CLI).
- **Live** calls `sentiment.topup()` each loop (TTL-throttled, best-effort) to append the newest
  funding/spot/fng to the same `history.db`, so the live strategy and the backtester read the
  identical store. The bot sets `strat.symbol` **before** `prepare()` so per-symbol sentiment loads.

## Honest validation result (2mo, 15m, portfolio)
Sentiment **helps, but does not create an edge**: `ensemble_sent` beat `ensemble` modestly
(−33.1% vs −37.2%, profit factor 0.88 vs 0.85 over the same losing window). The underlying strategy
has no durable edge, and **profit factor stays < 1 in every actively-trading configuration** — so
sentiment slows the bleed and improves win-rate slightly, but cannot by itself make the bot
profitable. It is a genuine edge-*hunting* ingredient (and an ML feature), not a fix. Real
profitability still waits on the nightly optimizer/ML finding a config that clears the holdout guard.

## Related
Equity-curve brake now uses a **rolling** peak (`brake_peak_hours`), not the all-time high, so it
protects during drawdowns without freezing forever — see [phase-6-proactive-guards.md](phase-6-proactive-guards.md).
A key lesson from Phase 6/7: the all-time-peak brake's apparent "profit" came from *halting
trading*; once made recoverable, the no-edge strategy resumes losing. Protection ≠ edge.
