"""
Phase 1 — train + validate the two LightGBM models (meta-labeling and direct direction).

Reuses: mldata.load_history (candles), mlcommon.build_features/FEATURES (features),
strategy.ensemble_decide (meta primary signal + dir reference), and the live engine from
backtest.py (_Sim/_manage/_enter/_metrics) for an honest out-of-sample P&L A/B.

Validation is a contiguous walk-forward holdout (train=past, test=future) with an H-bar embargo
so no forward-looking label leaks into training. Judge only the OOS numbers.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from mldata import load_history
from mlcommon import build_features, feature_matrix, FEATURES, model_path, size_mult
from strategy import DEFAULT_PARAMS, ensemble_decide
from risk import DEFAULT_RISK
from backtest import _Sim, _manage, _enter, _metrics

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, subsample=0.8,
                  colsample_bytree=0.8, min_child_samples=50, reg_lambda=1.0,
                  random_state=42, n_jobs=2, verbosity=-1)   # capped: shared VPS w/ autotrade

TEST_FRAC = 0.30


def _first_touch(highs, lows, i, up, dn, H, N) -> int:
    """+1 if the UP level is touched before the DOWN level within H bars after i, -1 if DOWN
    first, 0 if neither. Ties within a bar resolve to DOWN (conservative)."""
    end = min(i + H, N - 1)
    for j in range(i + 1, end + 1):
        if lows[j] <= dn:
            return -1
        if highs[j] >= up:
            return +1
    return 0


def _atr_at(frame, i, price) -> float:
    a = frame["atr"].iat[i]
    return float(a) if not (a is None or np.isnan(a)) else price * 0.03


def build_samples(frame: pd.DataFrame, params: dict, horizon: int, rr: float, bk: float):
    """Return (meta, dir) sample tuples: each (bar_indices, X, y)."""
    N = len(frame)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    X = feature_matrix(frame)
    valid = X.notna().all(axis=1).to_numpy()
    stop_atr = params["stop_atr"]

    m_idx, m_X, m_y = [], [], []
    d_idx, d_X, d_y = [], [], []
    for i in range(N):
        if not valid[i] or i + 1 >= N:
            continue
        price = close[i]
        atr = _atr_at(frame, i, price)

        # ---- direction model: symmetric triple-barrier on every valid bar ----
        b = bk * atr
        ft = _first_touch(highs, lows, i, price + b, price - b, horizon, N)
        d_idx.append(i); d_X.append(X.iloc[i].to_numpy()); d_y.append(0 if ft == 0 else (1 if ft > 0 else 2))

        # ---- meta model: only where the ensemble actually proposes a trade ----
        dec = ensemble_decide(frame.iloc[i], params)
        if dec.direction == "HOLD":
            continue
        dist = stop_atr * atr
        if dec.direction == "BUY":        # long: profit above, stop below -> win if UP first
            win = _first_touch(highs, lows, i, price + rr * dist, price - dist, horizon, N) == +1
        else:                              # short: stop above, target below -> win if DOWN first
            win = _first_touch(highs, lows, i, price + dist, price - rr * dist, horizon, N) == -1
        m_idx.append(i); m_X.append(X.iloc[i].to_numpy()); m_y.append(1 if win else 0)

    return (np.array(m_idx), np.array(m_X), np.array(m_y)), \
           (np.array(d_idx), np.array(d_X), np.array(d_y))


def _simulate(frame, params, risk_cfg, start_balance, taker_fee, timeframe, decide, start_i):
    """Bar-walk [start_i, N) via the live engine; `decide(i)->(direction, stop|None, conf)`."""
    from risk import RiskEngine
    risk = RiskEngine(risk_cfg)
    sim = _Sim(start_balance, taker_fee, risk.r["maint_margin"])
    sym = "BT"
    curve, N = [], len(frame)
    for i in range(start_i, N):
        row = frame.iloc[i]
        price = float(row["close"])
        if np.isnan(price):
            continue
        closed = _manage(sim, sym, row, price, params)
        direction, stop, conf = decide(i)
        if not (closed or direction == "flat" or stop is None):
            _enter(sim, risk, sym, direction, stop, price, sim.equity({sym: price}), conf)
        curve.append((frame.index[i].isoformat(), sim.equity({sym: price})))
    if sim.positions and curve:
        sim.close(sym, float(frame.iloc[-1]["close"]), "end")
    return _metrics(curve, sim.trades, start_balance, timeframe)


def train_symbol(symbol, timeframe, horizon, params, risk_cfg, taker_fee,
                 start_balance=1000.0, rr=1.5, bk=1.0, mode="both") -> dict:
    df = load_history(symbol, timeframe)
    if len(df) < 500:
        raise ValueError(f"not enough history for {symbol} {timeframe}: {len(df)} candles "
                         f"(run mldata.py first)")
    frame = build_features(df, params, symbol, timeframe)
    N = len(frame)
    (mi, mX, my), (di, dX, dy) = build_samples(frame, params, horizon, rr, bk)

    cut = int(N * (1 - TEST_FRAC))
    purge = cut - horizon                      # embargo: train labels can't peek into test
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

    results = {"symbol": symbol, "timeframe": timeframe, "horizon": horizon}
    X_all = feature_matrix(frame).fillna(0.0).to_numpy()
    valid = feature_matrix(frame).notna().all(axis=1).to_numpy()

    def save_bundle(model, name, oos, nsamp):
        os.makedirs(os.path.dirname(model_path(symbol, timeframe, name)), exist_ok=True)
        import joblib
        joblib.dump({"model": model, "features": FEATURES, "mode": name, "symbol": symbol,
                     "timeframe": timeframe, "horizon": horizon, "rr": rr, "bk": bk,
                     "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "n_samples": int(nsamp), "oos": oos,
                     "train_start": frame.index[0].isoformat(),
                     "train_end": frame.index[cut].isoformat()},
                    model_path(symbol, timeframe, name))

    thr = params.get("ml_threshold", 0.55)
    scal = bool(params.get("ml_size_scaling", True))

    # baseline ensemble over the holdout (same for both A/Bs)
    def ens_decide(i):
        row = frame.iloc[i]
        dec = ensemble_decide(row, params)
        atr = _atr_at(frame, i, float(row["close"]))
        if dec.direction == "BUY":
            return "long", float(row["close"]) - params["stop_atr"] * atr, 1.0
        if dec.direction == "SELL":
            return "short", float(row["close"]) + params["stop_atr"] * atr, 1.0
        return "flat", None, 1.0
    results["ensemble_oos"] = _simulate(frame, params, risk_cfg, start_balance, taker_fee,
                                        timeframe, ens_decide, cut)

    # ---- META ----
    if mode in ("meta", "both") and len(mi):
        tr = mi < purge
        te = mi >= cut
        if tr.sum() > 100 and te.sum() > 20:
            mdl = LGBMClassifier(**LGB_PARAMS).fit(mX[tr], my[tr])
            pte = mdl.predict_proba(mX[te])[:, 1]
            auc = roc_auc_score(my[te], pte) if len(set(my[te])) > 1 else None
            prec = precision_score(my[te], (pte >= thr).astype(int), zero_division=0)
            proba_full = mdl.predict_proba(X_all)[:, 1]

            def meta_decide(i):
                row = frame.iloc[i]
                dec = ensemble_decide(row, params)
                if dec.direction == "HOLD" or not valid[i]:
                    return "flat", None, 1.0
                p = proba_full[i]
                if p < thr:
                    return "flat", None, 1.0
                atr = _atr_at(frame, i, float(row["close"]))
                conf = size_mult(p, thr, scal)
                if dec.direction == "BUY":
                    return "long", float(row["close"]) - params["stop_atr"] * atr, conf
                return "short", float(row["close"]) + params["stop_atr"] * atr, conf
            pnl = _simulate(frame, params, risk_cfg, start_balance, taker_fee, timeframe, meta_decide, cut)
            oos = {"auc": round(auc, 3) if auc else None, "precision": round(prec, 3),
                   "n_test": int(te.sum()), **pnl}
            results["meta_oos"] = oos
            save_bundle(mdl, "meta", oos, len(mi))

    # ---- DIR ----
    if mode in ("dir", "both") and len(di):
        tr = di < purge
        te = di >= cut
        if tr.sum() > 100 and te.sum() > 20:
            mdl = LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced",
                                 **LGB_PARAMS).fit(dX[tr], dy[tr])
            pte = mdl.predict(dX[te])
            acc = accuracy_score(dy[te], pte)
            proba_full = mdl.predict_proba(X_all)              # (N,3): [flat,up,down]

            def dir_decide(i):
                if not valid[i]:
                    return "flat", None, 1.0
                pr = proba_full[i]
                cls = int(np.argmax(pr)); conf_raw = float(pr[cls])
                if cls == 0 or conf_raw < thr:
                    return "flat", None, 1.0
                row = frame.iloc[i]
                atr = _atr_at(frame, i, float(row["close"]))
                conf = size_mult(conf_raw, thr, scal)
                if cls == 1:
                    return "long", float(row["close"]) - params["stop_atr"] * atr, conf
                return "short", float(row["close"]) + params["stop_atr"] * atr, conf
            pnl = _simulate(frame, params, risk_cfg, start_balance, taker_fee, timeframe, dir_decide, cut)
            oos = {"accuracy": round(acc, 3), "n_test": int(te.sum()), **pnl}
            results["dir_oos"] = oos
            save_bundle(mdl, "dir", oos, len(di))

    return results


def _print(res: dict):
    print(f"\n==== {res['symbol']} {res['timeframe']} (H={res['horizon']}) ====")
    def line(name, r):
        if not r:
            print(f"  {name:<9} (no result)"); return
        print(f"  {name:<9} ret {r.get('total_return_pct'):>7}%  trades {r.get('n_trades'):>4}  "
              f"win {r.get('win_rate_pct')}%  PF {r.get('profit_factor')}  "
              f"maxDD {r.get('max_drawdown_pct')}%  Sharpe {r.get('sharpe')}"
              + (f"  AUC {r.get('auc')}" if 'auc' in r else '')
              + (f"  acc {r.get('accuracy')}" if 'accuracy' in r else ''))
    print("  --- OOS A/B (held-out future bars) ---")
    line("ensemble", res.get("ensemble_oos"))
    line("ml_meta", res.get("meta_oos"))
    line("ml_dir", res.get("dir_oos"))


def main():
    ap = argparse.ArgumentParser(description="Train ML meta/dir models")
    ap.add_argument("--symbol")
    ap.add_argument("--timeframe")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--mode", choices=["meta", "dir", "both"], default="both")
    args = ap.parse_args()

    from config import load_config
    cfg = load_config()
    params = {**DEFAULT_PARAMS, **cfg.get("params", {})}
    risk_cfg = {**DEFAULT_RISK, **cfg.get("risk", {})}
    taker_fee = cfg.get("taker_fee", 0.0005)
    start_balance = cfg.get("start_balance", 1000.0)
    symbols = [args.symbol] if args.symbol else cfg["symbols"]
    tf = args.timeframe or cfg["timeframe"]

    for sym in symbols:
        try:
            res = train_symbol(sym, tf, args.horizon, params, risk_cfg, taker_fee,
                               start_balance, mode=args.mode)
            _print(res)
        except Exception as e:
            print(f"\n==== {sym} {tf} ====\n  ERROR: {e}")


if __name__ == "__main__":
    main()
