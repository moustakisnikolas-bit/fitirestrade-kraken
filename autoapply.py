"""
Auto-apply (STRICT, guarded) — the autonomous half of the optimizer.

Runs nightly after training: optimize the live strategy's params/risk on the live timeframe, then
AUTO-UPDATE the live (paper) config **only if** the winner clears every guard on the untouched
holdout Optuna never saw. Otherwise it keeps the current config. Paper only; every decision logged.

Guards (ALL must hold to switch). The PRIMARY test is statistical: a beta-adjusted alpha t-stat on
the untouched holdout, so we only ever adopt a config whose edge is real (not market exposure, not
curve-fit). t > 2 ~ significant at the 95% level.
  * holdout best alpha_t    >= ALPHA_T_MIN        (statistically significant market-neutral edge)
  * holdout best alpha_t    >  holdout current alpha_t   (a genuine improvement)
  * holdout best drawdown shallower-or-equal to current (maxdd is negative)
  * holdout best trades     >= MIN_TRADES          (enough sample to trust the t-stat)
  * resulting leverage <= LEVERAGE_CEILING
  * cooldown: at most one switch per COOLDOWN_HOURS
Note: return-positive is intentionally NOT required — a market-neutral config with significant
positive alpha but a small negative return in a down-market holdout is a WIN (it beat the market),
and alpha_t already captures that correctly.
Controlled by config flag `autopilot_apply`; disabled -> dry-run (decides + logs, never writes).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")

from config import load_config, save_config
from strategy import DEFAULT_PARAMS
from risk import DEFAULT_RISK

STATUS_PATH = os.path.join(os.getenv("KF_DATA_DIR", "data"), "autoapply_status.json")
MIN_TRADES = 10
LEVERAGE_CEILING = 10.0
COOLDOWN_HOURS = 20.0
ALPHA_T_MIN = 2.0           # adopt only configs with a statistically significant holdout alpha


def load_status() -> dict | None:
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def _hours_since(iso: str | None) -> float:
    if not iso:
        return 1e9
    try:
        t = datetime.fromisoformat(iso)
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0
    except Exception:
        return 1e9


def run_autoapply(n_trials: int = 120) -> dict:
    from optimize import optimize
    cfg = load_config()
    enabled = bool(cfg.get("autopilot_apply"))
    params = {**DEFAULT_PARAMS, **cfg.get("params", {})}
    risk_cfg = {**DEFAULT_RISK, **cfg.get("risk", {})}

    rec = optimize(cfg["symbols"], cfg["timeframe"], cfg["strategy"], params, risk_cfg,
                   cfg["start_balance"], cfg["taker_fee"], n_trials)
    hb, hc = rec["holdout_best"], rec["holdout_current"]

    prev = load_status()
    cooled = _hours_since(prev.get("applied_at") if prev else None) >= COOLDOWN_HOURS

    checks = {
        "alpha_significant": (hb.get("alpha_t") or -1e9) >= ALPHA_T_MIN,
        "alpha_beats_current": (hb.get("alpha_t") or -1e9) > (hc.get("alpha_t") or -1e9),
        "drawdown_not_worse": (hb.get("maxdd") or -1e9) >= (hc.get("maxdd") or -1e9),
        "enough_trades": (hb.get("trades") or 0) >= MIN_TRADES,
        "leverage_ok": float(rec["best_risk"].get("max_leverage", 0)) <= LEVERAGE_CEILING,
        "cooldown_ok": cooled,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    passed = all(checks.values())

    decision, reason, applied_at = "kept", "", (prev or {}).get("applied_at")
    if passed and enabled:
        new = load_config()                      # re-read to avoid clobbering a concurrent edit
        new["params"] = rec["best_params"]
        new["risk"] = rec["best_risk"]
        save_config(new)
        decision = "applied"
        applied_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        reason = (f"holdout alpha_t {hc.get('alpha_t')}->{hb.get('alpha_t')} (>= {ALPHA_T_MIN}), "
                  f"ret {hc.get('return')}%->{hb.get('return')}%, DD {hc.get('maxdd')}%->{hb.get('maxdd')}% "
                  f"— significant-alpha guards passed")
    elif passed and not enabled:
        decision, reason = "would_apply", "guards passed but autopilot_apply is OFF (dry-run)"
    else:
        failed = [k for k, v in checks.items() if not v]
        reason = "guards failed: " + ", ".join(failed)

    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "enabled": enabled, "decision": decision, "reason": reason,
        "strategy": cfg["strategy"], "timeframe": cfg["timeframe"],
        "checks": checks, "holdout_best": hb, "holdout_current": hc,
        "overfit_warning": rec.get("overfit_warning"),
        "best_params": rec["best_params"], "best_risk": rec["best_risk"],
        "applied_at": applied_at,
    }
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2, default=_json_default)
    os.replace(tmp, STATUS_PATH)
    return status


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Strict guarded auto-apply of optimizer results")
    ap.add_argument("--trials", type=int, default=120)
    args = ap.parse_args()
    s = run_autoapply(args.trials)
    print(f"\n=== AUTO-APPLY {s['generated_at']} | {s['strategy']} {s['timeframe']} | enabled={s['enabled']} ===")
    print(f"  holdout current: alpha_t {s['holdout_current'].get('alpha_t')}  ret {s['holdout_current'].get('return')}%  DD {s['holdout_current'].get('maxdd')}%")
    print(f"  holdout best   : alpha_t {s['holdout_best'].get('alpha_t')}  ret {s['holdout_best'].get('return')}%  DD {s['holdout_best'].get('maxdd')}%  trades {s['holdout_best'].get('trades')}")
    print(f"  checks: {s['checks']}")
    print(f"  DECISION: {s['decision'].upper()} — {s['reason']}")


if __name__ == "__main__":
    main()
