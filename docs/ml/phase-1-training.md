# Phase 1 — Training & validation (`mltrain.py`, `mlcommon.py`)

## Goal
From stored candles, build features + triple-barrier labels, train **two** LightGBM models
(`meta`, `dir`), validate strictly out-of-sample, and persist model bundles the live strategies
load. Output an **A/B table**: `ensemble` vs `ml_meta` vs `ml_dir`.

## Shared features — `mlcommon.build_features(df, params)`
Returns one frame carrying raw indicator columns (for `ensemble_decide` + ATR stop) AND the
engineered model columns. `mlcommon.FEATURES` is the exact model input list:
`rsi, adx, roc, bb_pct, zscore, ema_ratio, atr_pct, ret_1, ret_3, ret_6, vol_6, dist_bo, hour`.
This single builder is reused by training and by the live `MLStrategy`, guaranteeing identical
features in both places (no train/serve skew).

## Labels (triple-barrier, first-touch over horizon H = `ml_horizon`)
- **meta** — one sample per bar where `ensemble_decide` is non-HOLD. Risk distance
  `dist = stop_atr·ATR`; stop barrier = ensemble stop; profit barrier = `entry ± ml_rr·dist`.
  `y = 1` if the **profit** barrier is touched before the **stop** within H bars, else `0`.
  (Standard meta-labeling: "does the primary signal win?")
- **dir** — one sample per bar. Symmetric barriers `entry ± ml_bk·ATR`. `y = up(1)` if the upper
  barrier is hit first, `down(2)` if lower first, `flat(0)` if neither within H.

## Models
LightGBM (`meta` = binary classifier, `dir` = 3-class). Conservative params to resist overfit:
`n_estimators=300, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
min_child_samples=50, reg_lambda=1.0, random_state=42`.

## Validation — contiguous walk-forward holdout with embargo
- Split at `cut = N·(1 − test_frac)` (default `test_frac=0.3`): **train = early bars, test = last
  bars** (future only — no shuffling).
- **Embargo/purge:** training samples whose label horizon could overlap the test region
  (`bar_index > cut − H`) are dropped, so no forward-looking label leaks into training.
- **Classification metrics** on the holdout: meta → ROC-AUC + precision@threshold; dir → accuracy
  + up/down precision.
- **OOS P&L A/B:** bar-walk the holdout three ways via the **exact live engine**
  (`backtest._Sim/_manage/_enter/_metrics`): `ensemble` (baseline), `ml_meta` (ensemble gated by
  P(win)≥threshold), `ml_dir` (model direction). Reports return %, trades, win rate, profit
  factor, max drawdown, Sharpe — the headline comparison.

## Persistence — `data/models/<symbol>_<tf>_<mode>.pkl` (joblib)
Bundle dict: `{model, features, mode, symbol, timeframe, horizon, rr/bk, oos{...}, trained_at,
n_samples, train_start, train_end}`. Loaded by `mlcommon.load_model` (mtime-cached).

## CLI
```
python mltrain.py --symbol SOL/USD:USD --timeframe 1h --horizon 24 --mode both
python mltrain.py --mode both            # all config symbols at config timeframe
```
Prints, per symbol, the classification metrics and the OOS A/B P&L table, then writes bundles.

## Edge cases
- Too few samples after warmup/purge → skip with a clear message (no model written).
- Rows with NaN features are excluded from training and treated as `flat` in the sim.
- Deterministic (`random_state=42`) so reruns match.

## Test steps
1. Ensure Phase 0 ran (`data/history.db` populated).
2. `docker exec kfpaper_bot python mltrain.py --mode both` → inspect OOS A/B table.
3. Confirm `data/models/*_meta.pkl` and `*_dir.pkl` exist with sane metadata.
