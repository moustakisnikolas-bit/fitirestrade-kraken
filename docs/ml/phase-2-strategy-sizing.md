# Phase 2 — ML strategies + confidence sizing

## Files touched
`strategies.py` (new strategies), `strategy.py` (new params), `risk.py` (sizing),
`bot.py` + `backtest.py` (pass symbol/timeframe + confidence). Shared logic in `mlcommon.py`.

## Per-symbol model loading (interface addition)
Models are per `(symbol, timeframe)`, but a `Strategy` instance is shared across symbols and
`signal(row, price)` has no symbol argument. Minimal, backward-compatible fix: the base
`Strategy` now carries `self.symbol` / `self.timeframe`, which the callers set before use:
- `bot.py`: `strat.timeframe = cfg["timeframe"]` once per loop; `strat.symbol = sym` right before
  `strat.signal(...)`.
- `backtest.py`: `strat.symbol`/`strat.timeframe` set once (single-symbol run).
Rule-based strategies ignore these fields, so nothing else changes.

## Strategies (registered in `REGISTRY`)
- `_MLBase` — `prepare` = `mlcommon.build_features`; lazy `load_model(symbol, tf, mode)`;
  **safe degradation**: missing model → `flat` + one-time warning (never crashes the bot).
- `ml_meta` (`MetaStrategy`) — `ensemble_decide` is primary; take only if `P(win) ≥ ml_threshold`;
  `meta={"p_win","conf","regime"}`, direction + ATR stop identical to the ensemble.
- `ml_dir` (`DirStrategy`) — direct model; direction = argmax of [flat, up, down]; take only if
  `class ≠ flat and max_prob ≥ ml_threshold`; ATR stop; `meta={"p","conf","cls"}`.

## Confidence-scaled sizing
- `mlcommon.size_mult(conf, threshold, enabled)` → multiplier in **[0.5, 1.5]**
  (0.5 at `conf==threshold`, 1.5 at `conf==1.0`; `1.0` when disabled).
- `RiskEngine.size(equity, entry, stop, size_mult=1.0)` scales the **risk budget** by that
  multiplier. Crucially, notional is **still hard-capped by the liquidation-safe leverage**, so
  confidence sizing can never make a position unsafe.
- `bot.try_enter` and `backtest._enter` read `meta["conf"]` and pass it through. Default `1.0`
  keeps `ensemble`/`breakout` behaviour byte-for-byte unchanged.

## New config params (`DEFAULT_PARAMS` in `strategy.py`)
- `ml_threshold` (0.55) — min model confidence to trade.
- `ml_horizon` (24) — label/forecast horizon in bars (passed to training).
- `ml_size_scaling` (1.0 = on) — toggle confidence sizing.
These appear automatically in the panel's params editor and `config.json`. They are NOT window
lengths, so the Phase-1 int-coercion in `compute_indicators` leaves them as floats.

## Edge cases
- Warm-up / NaN features → `flat`.
- No trained model for the active `(symbol, timeframe)` → `flat` + warn once; bot keeps running.
- `mlcommon.load_model` is mtime-cached → a retrain is picked up without a restart.

## Test steps
- Backtest card: run `ensemble`, `ml_meta`, `ml_dir` on the same symbol/timeframe → compare.
- With `ml_size_scaling=0`, confirm identical sizing to a flat-confidence run.
- Switch live strategy to `ml_meta` with no model → bot logs the warning and trades nothing.
