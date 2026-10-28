"""
Phase 4 — Optuna parameter optimization (recommend-only).

Searches decision/risk params for a configuration with a DURABLE, out-of-sample edge, scored by the
PORTFOLIO backtest (all symbols, one account, fees + funding). Anti-overfitting by construction:
  * candles reused once; indicator LENGTHS are NOT searched (so frames stay valid across trials)
  * each trial is scored on K disjoint OOS fold-windows; the objective is the MEDIAN Calmar across
    them (one lucky fold can't win), with penalties for too-few-trades / huge drawdowns
  * a final HOLDOUT window Optuna never sees validates the best config — that's the real verdict

Writes data/optuna_best.json. Never changes the live config (the panel's Apply does, on a click).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone

from backtest import load_portfolio_frames, simulate_portfolio, fetch_funding_rates

BEST_PATH = os.path.join(os.getenv("KF_DATA_DIR", "data"), "optuna_best.json")
WARMUP = 60
HOLDOUT_FRAC = 0.15
N_FOLDS = 4
MIN_TRADES = 8


def _calmar(summary: dict) -> float:
    ret = summary.get("total_return_pct") or 0.0
    dd = summary.get("max_drawdown_pct")
    if not dd:                      # no drawdown recorded -> judge on return alone
        return ret
    return ret / abs(dd)


def _folds(n: int):
    """K disjoint OOS windows across the study region [WARMUP, holdout_start)."""
    holdout_start = int(n * (1 - HOLDOUT_FRAC))
    lo, hi = WARMUP, holdout_start
    if hi - lo < N_FOLDS * 50:
        return [(lo, hi)], holdout_start        # too short -> single fold
    width = (hi - lo) // N_FOLDS
    return [(lo + k * width, lo + (k + 1) * width) for k in range(N_FOLDS)], holdout_start


def _score(frames, common, strategy, params, risk_cfg, tf, start_balance, fee, rates, folds) -> float:
    cals = []
    for a, b in folds:
        summ, _ = simulate_portfolio(frames, common, strategy, params, risk_cfg, tf,
                                     start_balance, fee, rates, a, b)
        if summ["n_trades"] < MIN_TRADES or (summ["max_drawdown_pct"] or 0) <= -40:
            cals.append(-10.0)                  # penalise degenerate / blown-up configs
        else:
            cals.append(_calmar(summ))
    return statistics.median(cals) if cals else -10.0


def optimize(symbols, timeframe, strategy, base_params, base_risk, start_balance, fee,
             n_trials=200, limit=5000) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    frames, common = load_portfolio_frames(symbols, timeframe, strategy, base_params, limit)
    rates = fetch_funding_rates(symbols)
    n = len(common)
    folds, holdout_start = _folds(n)
    is_ml = strategy in ("ml_meta", "ml_dir")

    def objective(trial):
        p = dict(base_params)
        p["enter_thr"] = trial.suggest_float("enter_thr", 0.20, 0.60)
        p["exit_thr"] = trial.suggest_float("exit_thr", -0.40, 0.0)
        p["stop_atr"] = trial.suggest_float("stop_atr", 1.5, 4.0)
        p["trail"] = trial.suggest_float("trail", 0.005, 0.03)
        p["profit_lock"] = trial.suggest_float("profit_lock", 0.005, 0.03)
        p["max_hold_bars"] = trial.suggest_int("max_hold_bars", 24, 192)
        p["adx_trend"] = trial.suggest_float("adx_trend", 15.0, 35.0)
        if is_ml:
            p["ml_threshold"] = trial.suggest_float("ml_threshold", 0.50, 0.70)
        r = dict(base_risk)
        r["max_leverage"] = trial.suggest_float("max_leverage", 2.0, 10.0)
        r["risk_pct_per_trade"] = trial.suggest_float("risk_pct_per_trade", 0.005, 0.02)
        return _score(frames, common, strategy, p, r, timeframe, start_balance, fee, rates, folds)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    bp = study.best_params
    best_params = dict(base_params)
    for k in ("enter_thr", "exit_thr", "stop_atr", "trail", "profit_lock", "max_hold_bars", "adx_trend"):
        best_params[k] = bp[k]
    if is_ml:
        best_params["ml_threshold"] = bp["ml_threshold"]
    best_risk = dict(base_risk)
    best_risk["max_leverage"] = bp["max_leverage"]
    best_risk["risk_pct_per_trade"] = bp["risk_pct_per_trade"]

    def _slice(params, risk, a, b):
        s, _ = simulate_portfolio(frames, common, strategy, params, risk, timeframe,
                                  start_balance, fee, rates, a, b)
        return {"return": s["total_return_pct"], "pf": s["profit_factor"],
                "maxdd": s["max_drawdown_pct"], "sharpe": s["sharpe"], "trades": s["n_trades"]}

    # the real verdict: best config on the untouched holdout, vs the current config on the same holdout
    holdout_best = _slice(best_params, best_risk, holdout_start, n)
    holdout_base = _slice(base_params, base_risk, holdout_start, n)
    try:
        import optuna.importance as imp
        importance = {k: round(v, 3) for k, v in imp.get_param_importances(study).items()}
    except Exception:
        importance = {}

    rec = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timeframe": timeframe, "strategy": strategy, "symbols": symbols,
        "n_trials": n_trials, "candles": n, "study_metric_median_calmar": round(study.best_value, 3),
        "best_params": best_params, "best_risk": best_risk, "searched": bp,
        "holdout_best": holdout_best, "holdout_current": holdout_base,
        "holdout_period": [common[holdout_start].isoformat(), common[n - 1].isoformat()],
        "overfit_warning": bool((holdout_best.get("return") or 0) <= 0 and study.best_value > 0),
        "param_importance": importance,
    }
    os.makedirs(os.path.dirname(BEST_PATH), exist_ok=True)
    tmp = BEST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, BEST_PATH)
    return rec


def load_best() -> dict | None:
    try:
        with open(BEST_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Optuna optimize strategy/risk params")
    ap.add_argument("--timeframe")
    ap.add_argument("--strategy")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()

    from config import load_config
    from strategy import DEFAULT_PARAMS
    from risk import DEFAULT_RISK
    cfg = load_config()
    params = {**DEFAULT_PARAMS, **cfg.get("params", {})}
    risk_cfg = {**DEFAULT_RISK, **cfg.get("risk", {})}
    tf = args.timeframe or cfg["timeframe"]
    strat = args.strategy or cfg["strategy"]

    rec = optimize(cfg["symbols"], tf, strat, params, risk_cfg,
                   cfg["start_balance"], cfg["taker_fee"], args.trials, args.limit)
    print(f"\n=== OPTUNA {rec['generated_at']} | {strat} {tf} | {rec['n_trials']} trials ===")
    print(f"  study median Calmar (OOS folds): {rec['study_metric_median_calmar']}")
    hb, hc = rec["holdout_best"], rec["holdout_current"]
    print(f"  HOLDOUT (untouched)   best: ret {hb['return']}%  PF {hb['pf']}  maxDD {hb['maxdd']}%  trades {hb['trades']}")
    print(f"  HOLDOUT (untouched) current: ret {hc['return']}%  PF {hc['pf']}  maxDD {hc['maxdd']}%  trades {hc['trades']}")
    if rec["overfit_warning"]:
        print("  ** OVERFIT WARNING: great in-study but negative on holdout — do NOT apply. **")
    print(f"  best params: {rec['best_params']}")
    print(f"  best risk  : {rec['best_risk']}")


if __name__ == "__main__":
    main()
