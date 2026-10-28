"""
Auto-pilot (recommend-only).

Weekly job: backfill history, retrain both ML models, and read their out-of-sample A/B results
(from mltrain.train_symbol) to RANK ensemble vs ml_meta vs ml_dir across all configured symbols.
Writes a recommendation to data/recommendation.json and prints a summary. It deliberately does
NOT change the live strategy — the panel surfaces the recommendation and a human clicks "Apply".

Run manually: `python autopilot.py`   (scheduled weekly via cron on the VPS)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from config import load_config
from strategy import DEFAULT_PARAMS
from risk import DEFAULT_RISK
from mltrain import train_symbol

REC_PATH = os.path.join(os.getenv("KF_DATA_DIR", "data"), "recommendation.json")

# OOS metric key in train_symbol output -> strategy name
_STRATS = {"ensemble_oos": "ensemble", "meta_oos": "ml_meta", "dir_oos": "ml_dir"}


def run_autopilot(symbols, timeframe, horizon, params, risk_cfg, taker_fee,
                  start_balance, current_strategy) -> dict:
    import mldata
    per_symbol, agg = [], {}
    for sym in symbols:
        mldata.backfill(sym, timeframe, 20000)
        res = train_symbol(sym, timeframe, horizon, params, risk_cfg, taker_fee,
                           start_balance, mode="both")
        row = {"symbol": sym}
        for key, strat in _STRATS.items():
            o = res.get(key)
            if o:
                row[strat] = {"return": o.get("total_return_pct"), "pf": o.get("profit_factor"),
                              "maxdd": o.get("max_drawdown_pct"), "sharpe": o.get("sharpe")}
                agg.setdefault(strat, []).append(o)
        per_symbol.append(row)

    ranking = []
    for strat, lst in agg.items():
        rets = [x.get("total_return_pct") or 0 for x in lst]
        pfs = [x.get("profit_factor") for x in lst if x.get("profit_factor") is not None]
        dds = [x.get("max_drawdown_pct") or 0 for x in lst]
        ranking.append({
            "strategy": strat, "n": len(lst),
            "avg_return": round(sum(rets) / len(rets), 2),
            "avg_pf": round(sum(pfs) / len(pfs), 2) if pfs else None,
            "avg_maxdd": round(sum(dds) / len(dds), 2),
        })
    # rank by average OOS return, then by profit factor
    ranking.sort(key=lambda r: (r["avg_return"], r["avg_pf"] or 0), reverse=True)

    recommended = ranking[0]["strategy"] if ranking else current_strategy
    cur = next((r for r in ranking if r["strategy"] == current_strategy), None)
    top = ranking[0] if ranking else None
    # guard: only *recommend a change* if the leader clearly beats current on BOTH return & drawdown
    beats = bool(top and cur and top["strategy"] != current_strategy
                 and top["avg_return"] > cur["avg_return"]
                 and top["avg_maxdd"] >= cur["avg_maxdd"])   # maxdd is negative; >= = shallower
    if not beats:
        recommended = current_strategy
        note = f"Keep {current_strategy}: no challenger beats it on BOTH OOS return and drawdown."
    else:
        note = (f"Consider switching {current_strategy} -> {top['strategy']}: better OOS return "
                f"({top['avg_return']}% vs {cur['avg_return']}%) and drawdown "
                f"({top['avg_maxdd']}% vs {cur['avg_maxdd']}%). Recommend-only — apply manually.")

    rec = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "timeframe": timeframe, "horizon": horizon,
           "current_strategy": current_strategy, "recommended": recommended,
           "change_suggested": beats, "note": note,
           "ranking": ranking, "per_symbol": per_symbol}

    os.makedirs(os.path.dirname(REC_PATH), exist_ok=True)
    tmp = REC_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, REC_PATH)
    return rec


def load_recommendation() -> dict | None:
    try:
        with open(REC_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    cfg = load_config()
    params = {**DEFAULT_PARAMS, **cfg.get("params", {})}
    risk_cfg = {**DEFAULT_RISK, **cfg.get("risk", {})}
    horizon = int(params.get("ml_horizon", 24))
    rec = run_autopilot(cfg["symbols"], cfg["timeframe"], horizon, params, risk_cfg,
                        cfg["taker_fee"], cfg["start_balance"], cfg["strategy"])
    print(f"\n=== AUTO-PILOT {rec['generated_at']} | tf={rec['timeframe']} ===")
    print("  OOS ranking (avg across symbols):")
    for r in rec["ranking"]:
        print(f"    {r['strategy']:<9} return {r['avg_return']:>7}%  PF {r['avg_pf']}  maxDD {r['avg_maxdd']}%")
    print(f"  current: {rec['current_strategy']} | recommended: {rec['recommended']}")
    print(f"  {rec['note']}")


if __name__ == "__main__":
    main()
