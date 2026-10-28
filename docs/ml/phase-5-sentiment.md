# Phase 5 — Market sentiment (`sentiment.py`)

## Goal
Add real market sentiment as a model feature, in a way that is **free** and **backtestable** (so it
can be validated OOS, not just believed).

## Source choice
- **Crypto Fear & Greed Index** (alternative.me) — free, **full daily history**, market-wide
  (BTC-centric) 0..100. Backtestable because history exists. **Chosen.**
- Funding rate — already modeled as a *cost* in the backtester; historical per-bar funding isn't
  cleanly available, so it is not added as a signal feature in v1.
- Social / news / LLM sentiment — **rejected for now**: paid and/or no clean history, so it can't be
  honestly backtested. Live-only experiments can come later.

Stdlib only (`urllib`) → **no new dependency, no image rebuild**.

## Data
`history.db` table `fng(ts INTEGER PRIMARY KEY, value REAL)` (daily). `backfill_fng()` pulls the full
history in one small request (idempotent). `current_fng()` → `{value, label}` for the dashboard.

## Feature wiring
`mlcommon.build_features()` adds an `fng` column via `sentiment.align_fng(index)`:
- daily value forward-filled onto each (intraday) candle, normalised to **[-1, 1]** (0 = neutral/50),
- **0.0 when absent** (before history / fetch fails) → always safe.
`"fng"` is appended to `mlcommon.FEATURES`.

**Backward compatible:** models store their own `features` list and predict against it, so models
trained before this change keep working unchanged; only **newly trained** models use `fng`. Retrain
(ML Lab → Backfill & train, or auto-pilot) to let the models actually use sentiment.

Scope note: `fng` feeds the **ML** strategies (`ml_meta`/`ml_dir`, which use `build_features`). The
rule-based `ensemble` uses `compute_indicators` directly and is unaffected; a hard sentiment *gate*
for it is a possible later addition (left out of v1 to avoid disturbing tuned params).

## Panel
- `/api/status` returns `fng` (`{value,label}`); the dashboard shows a **Market sentiment** tile
  (red in fear, green in greed).
- `/api/mltrain` refreshes `backfill_fng()` before training so model data is current.
- A small daily cron keeps the live reading fresh.

## Usage
1. Backfill once (`python sentiment.py` or via ML Lab training).
2. Retrain the ML models for your timeframe so they include `fng`.
3. Compare `ml_meta` with vs without sentiment in the **Portfolio backtest** / Optimizer — keep it
   only if it improves the **out-of-sample / holdout** result. Don't assume it helps.

## Verification
- `python sentiment.py` prints stored point count + current reading.
- `build_features(df, params)` has an `fng` column; `"fng"` in `FEATURES`.
- Dashboard shows the sentiment tile; a freshly trained model's `features` list includes `fng`.
