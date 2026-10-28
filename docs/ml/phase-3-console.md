# Phase 3 — Admin console integration (`panel.py`)

## Endpoints
- `GET /api/model` — status of both models (`meta`, `dir`) for each configured symbol at the
  **active timeframe**, read from the `.pkl` metadata via `mlcommon.model_status` (trained_at,
  n_samples, OOS metrics). Symbols with no model report `trained_at: null`.
- `POST /api/mltrain` — body `{symbols?, timeframe?, horizon?}` (all optional; default to config).
  Runs `mldata.backfill` + `mltrain.train_symbol(mode="both")` for each symbol inside
  `run_in_threadpool` (keeps the panel responsive), returns the A/B `results`. Heavy imports
  (`lightgbm`, `sklearn`) are lazy — pulled in only inside the handler, not at panel startup.

## UI — "ML models" card (between Backtest and Open positions)
- A status table (symbol × {meta, dir}): trained timestamp, samples, OOS return / PF / maxDD,
  AUC(meta) or accuracy(dir). Populated every refresh from `GET /api/model`.
- A **"Backfill & retrain (both models)"** button → `POST /api/mltrain` (shows a "can take a
  minute" note, then refreshes the table).
- A/B needs **no extra plumbing**: the existing Backtest card's Strategy dropdown already lists
  `ml_meta` and `ml_dir` (auto-built from `REGISTRY`), so all three strategies compare directly on
  identical candles.

## Important: timeframe coupling
Models are per `(symbol, timeframe)`. `/api/model` and the retrain button use the **active config
timeframe**. If the bot runs `5m` but models were trained on `1h`, the card shows "no models" for
`5m` and the `ml_*` strategies stay flat live until trained at the running timeframe. Train at the
timeframe you intend to trade.

## Go-live (paper) workflow
1. Set the bot's timeframe to the one you'll trade; click **Backfill & retrain**.
2. In the Backtest card, compare `ensemble` vs `ml_meta` vs `ml_dir` on that symbol/timeframe.
3. Only if `ml_meta`/`ml_dir` beats `ensemble` on **profit factor** and **max drawdown**, set it as
   the active strategy in Configuration and Save.
4. Monitor live paper P&L. Real money remains a separate, later decision.

## Test steps
- Load the panel → "ML models" card renders (empty until trained).
- Click retrain → table fills with OOS metrics.
- `/api/backtest` with `strategy=ml_meta` returns a summary + curve (verified live).
