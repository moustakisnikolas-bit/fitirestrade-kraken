# Phase 4 — Optuna parameter optimization (`optimize.py`)

## Goal
Systematically search strategy/risk parameters for a configuration with a **durable, out-of-sample
edge**, scored by the **portfolio backtest** (all symbols, one account, fees + funding) — the same
thing the live bot does. Recommend-only: the optimizer proposes; a human applies.

> The danger is not Optuna — it's overfitting. Trying hundreds of configs and keeping the best
> **in-sample** number is how you build a bot that backtests at +300% and loses live. Every rule
> below exists to prevent that.

## Installation
- Add to `requirements.txt`: `optuna==4.1.0` (pulls SQLAlchemy, alembic, colorlog — modest).
- **Needs one image rebuild** (new dependency). Watch disk per the disk-safety rule; prune is off
  the table (daemon-wide), so just confirm headroom before the build.
- Study storage: SQLite at `data/optuna.db` (persists in the `kfpaper_data` volume → resumable,
  inspectable). Small file.

## Correct usage — the anti-overfitting contract (non-negotiable)
1. **Score only out-of-sample.** The objective runs **walk-forward**: split the history into K
   expanding folds; for each, the config is scored on the **held-out future** slice, with an
   `H`-bar **embargo/purge** between in-sample and out-of-sample (reuse the pattern from
   `mltrain.py`). Never score on data used to pick the config.
2. **A final untouched holdout.** Reserve the most recent slice (e.g. last 15%) that Optuna
   **never sees**. After the study, re-run the single best config on it. If it collapses there,
   it's overfit — reject it. This is the real verdict, printed alongside the study result.
3. **Optimize a risk-adjusted, robust metric — not raw return.** Objective =
   **median across folds of Calmar** (`OOS return / |max drawdown|`), i.e. the *median* (not mean)
   so one lucky fold can't win. Penalize degenerate configs: if a fold makes
   `< min_trades` (e.g. 10) or `max_dd < -40%`, score that fold as a large negative. Raw return as
   the objective is banned.
4. **Costs are already in the objective** — the portfolio backtest includes taker fees and funding,
   so Optuna is rewarded for trading *less and better*, not more.
5. **Small search space, lots of data.** We have 20k candles; keep the space tight to keep the
   data-per-parameter ratio high. Prefer the **decision/risk knobs** (below) over indicator
   lengths — and note indicator lengths force a feature recompute per trial (slow), whereas
   thresholds/stops/risk do **not** when features are cached, so those are the cheap, high-impact
   search.
6. **Determinism + budget.** `TPESampler(seed=42)`, `MedianPruner`, `n_trials` ≈ 150–300.
7. **Stability over spikes.** Report `optuna.importance` and prefer a config on a *plateau* of good
   neighbours, not an isolated spike (a spike is usually noise).

## Search space (first version)
| Param | Range | Note |
|---|---|---|
| `enter_thr` | 0.2 – 0.6 | higher = fewer, stronger entries (less fee drag) |
| `exit_thr` | -0.4 – 0.0 | |
| `stop_atr` | 1.5 – 4.0 | stop distance |
| `trail` | 0.005 – 0.03 | |
| `profit_lock` | 0.005 – 0.03 | |
| `max_hold_bars` | 24 – 192 | time-stop |
| `adx_trend` | 15 – 35 | regime cut |
| risk `max_leverage` | 2 – 10 | **big fee/funding lever** |
| risk `risk_pct_per_trade` | 0.005 – 0.02 | |
| (ml) `ml_threshold` | 0.5 – 0.7 | only when optimizing an `ml_*` strategy |

Indicator lengths (`ema_*`, `rsi_len`, …) kept fixed in v1 (separate, coarser search later).

## Files
- **`optimize.py`**
  - `load_frames(symbols, tf, limit)` → prepare feature frames once (reused across folds within a
    trial for threshold-only params).
  - `walkforward_score(params, risk) -> float` → K folds via a sliced portfolio simulation
    (refactor the core loop of `backtest.run_portfolio_backtest` into
    `_simulate_portfolio(frames, strats, params, risk, start_i, end_i, funding)` so folds don't
    re-fetch candles).
  - `objective(trial)` → suggest params, call `walkforward_score`, return the robust metric.
  - `optimize(symbols, tf, strategy, n_trials)` → run study (SQLite storage), take best, run the
    **holdout** check, write `data/optuna_best.json`:
    `{strategy, timeframe, best_params, best_risk, study_metric, folds[], holdout{...},
      param_importance, generated_at, baseline_current{...}}`.
  - CLI: `python optimize.py --timeframe 15m --strategy ensemble --trials 200`.
- **`backtest.py`** — extract `_simulate_portfolio(...)` (shared by `run_portfolio_backtest` and the
  optimizer) taking an explicit `[start_i, end_i)` bar range and preloaded frames.
- **`panel.py`**
  - `POST /api/optimize` → run in `run_in_threadpool` (or background job); returns best + holdout.
  - `GET /api/optimize` → read `data/optuna_best.json`.
  - **Optimizer card** (ML Lab): search button, progress/trial count, best params table, study-OOS
    vs **holdout** metrics side by side, `importance` bars, and **Apply best** (writes params+risk
    via `/api/config`). Recommend-only; never auto-applies.

## Usage flow
1. Confirm disk headroom → add `optuna`, rebuild once.
2. `Optimize` on your live timeframe (15m) for `ensemble` (and later `ml_meta`).
3. Read the result: trust it **only if the holdout metric agrees with the study metric**. If study
   says +good and holdout says negative → overfit, discard.
4. If it holds up, click **Apply best**, then watch live track the portfolio backtest.

## Honest expectations
Optuna finds the best config *that the strategy family can express*. If the ensemble has no real
edge on these markets, the honest outcome is "best config is still ~flat after costs" — a valid,
money-saving result. Optuna makes the search rigorous; it cannot manufacture an edge that isn't
there. The win it most often delivers here: **fewer trades + lower leverage** that cut the fee/funding
drain.

## Verification
- `import optuna` works in-container; `data/optuna.db` created.
- `python optimize.py --trials 20` runs, writes `optuna_best.json` with folds + holdout.
- Holdout metric present and sane; Apply writes config; live picks it up next loop.
- Re-run is reproducible (seeded).
