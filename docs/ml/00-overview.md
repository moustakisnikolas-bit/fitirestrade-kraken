# ML forecasting for krakenbot — Overview

## Why
The live paper bot runs the rule-based `ensemble` strategy, which (per its own author) has **no
out-of-sample edge on majors**. We are adding machine-learning forecasting as **selectable,
validated strategies** to improve *risk-adjusted* P&L — realistically a higher profit factor and
lower drawdown, not magical returns. Everything runs **paper-only**; real money is a separate,
later gate.

## Where it plugs in (existing architecture, unchanged contracts)
```
bot.py loop ─ every closed bar ─────────────────────────────────────────────
   strat.prepare(df)            # add indicator/feature columns
   direction,stop,meta = strat.signal(row, price)   # ("long"|"short"|"flat", stop, meta)
   RiskEngine.size()/admit()    # owns position sizing + exposure caps
   PaperFutures.open/close      # simulated fills in SQLite
backtest.py  ─ replays the SAME strat.signal + RiskEngine + stop/trail/liq math ── OOS harness
```
A strategy only proposes **direction + stop**; the risk engine owns sizing. So an ML model is
just a new `Strategy` subclass — **no change to execution or risk core**.

## What we build (all phases, paper-only)
| Phase | File(s) | Deliverable |
|-------|---------|-------------|
| 0 | `mldata.py` | Historical candle backfill → `data/history.db`; feasibility probe |
| 1 | `mltrain.py` | Features + triple-barrier labels; train **two** LightGBM models; walk-forward OOS eval |
| 2 | `strategies.py`, `strategy.py`, `risk.py`, `bot.py`, `backtest.py` | `ml_meta` + `ml_dir` strategies; confidence-scaled sizing; new params |
| 3 | `panel.py` | `/api/model`, `/api/mltrain`; "ML model" card; A/B via existing Backtest card |

## The two models (A/B)
- **`ml_meta` (meta-labeling):** keep `ensemble` as the primary signal; ML predicts **P(this
  trade wins)** and acts as a *take/skip filter + size multiplier*. Easier to find an edge;
  targets win-rate/drawdown directly.
- **`ml_dir` (direct direction):** ML itself predicts up/flat/down over horizon H and proposes
  the trade with an ATR stop. Simpler concept, harder to beat fees.

Both are compared against `ensemble` in the backtester on identical data.

## Validation methodology (non-negotiable)
- **Triple-barrier labeling** (López de Prado): a sample's outcome = which of {profit target,
  stop, time horizon} is hit first — the same rules the live bot uses.
- **Walk-forward / purged CV with embargo:** train on past, test on strictly-future bars, with an
  `H`-bar gap purged between train and test so no label overlaps the test window (no leakage).
- **Judge only OOS.** In-sample metrics are ignored. Every model reports OOS AUC/precision **and**
  an OOS P&L backtest vs the `ensemble` baseline.

## Go-live gates
1. **Paper active-strategy gate:** switch the paper bot to `ml_meta`/`ml_dir` only if its OOS
   backtest beats `ensemble` on **profit factor** and **max drawdown**.
2. **Real-money gate (future, out of scope here):** only after the chosen strategy stays
   profitable over sustained *live paper* trading.
"No edge found" is a valid, money-saving outcome.

## Data flow & persistence
- Candles: `data/history.db` (SQLite). Models: `data/models/<symbol>_<tf>_<mode>.pkl` (joblib).
- Both live in the `kfpaper_data` Docker volume (`/app/data`) → survive image rebuilds.

## Glossary
- **OOS** — out-of-sample (data the model never trained on).
- **Triple-barrier** — label by first-touched of profit / stop / time barriers.
- **Embargo/purge** — bars removed between train & test so forward-looking labels can't leak.
- **Meta-labeling** — ML decides whether to *take* a primary signal, and how big.
- **Profit factor (PF)** — gross profit / gross loss. **Sharpe** — annualized return/vol.
