"""
Risk engine — the core of the futures platform. Strategy-agnostic. Its job is to make leverage
SAFE: size positions by risk (not by gut), pick a leverage low enough that the stop is always hit
BEFORE liquidation, cap total exposure + concurrency, and halt trading on daily-loss / drawdown
breaches. A strategy only ever proposes a direction + stop; the risk engine decides the size.
"""
from __future__ import annotations

DEFAULT_RISK = {
    "risk_pct_per_trade": 0.01,     # risk 1% of equity between entry and stop
    "max_leverage": 3.0,            # hard cap on per-position leverage
    "max_concurrent": 2,            # max simultaneous open positions
    "max_total_exposure_pct": 1.5,  # sum(notional)/equity gross cap
    "daily_loss_limit_pct": 0.05,   # halt NEW entries if down 5% on the day
    "max_drawdown_pct": 0.20,       # halt NEW entries if down 20% from peak equity
    "min_stop_dist_pct": 0.003,     # reject entries whose stop is < 0.3% away (noise)
    "liq_buffer": 0.5,              # stop must sit within 50% of the distance to liquidation
    "maint_margin": 0.005,          # exchange maintenance margin (Kraken ~0.5%)
    # PROACTIVE GUARDS (see docs/ml/phase-6-proactive-guards.md)
    "equity_brake": True,           # #2 graded de-risk as drawdown-from-peak deepens
    "brake_soft_dd": 0.06,          #    start shrinking NEW-entry size at 6% below peak
    "brake_pause_dd": 0.12,         #    PAUSE new entries entirely at 12% below peak
    "brake_floor": 0.3,             #    smallest size multiplier inside the taper zone
    "brake_peak_hours": 120.0,      #    drawdown is measured vs a ROLLING peak over this window
                                    #    (not the all-time high), so the brake releases once equity
                                    #    stabilises instead of freezing forever. 0 = all-time peak.
    "max_same_direction": 1,        # #4 correlation cap: max concurrent same-direction positions
}


class RiskEngine:
    def __init__(self, rcfg: dict | None = None):
        self.r = {**DEFAULT_RISK, **(rcfg or {})}

    def liq_safe_leverage(self, stop_dist: float) -> float:
        """Largest leverage such that liquidation (~1/L - maint away) is FARTHER than the stop,
        with a safety buffer. Guarantees the stop fires before a liquidation can."""
        denom = stop_dist / max(self.r["liq_buffer"], 1e-9) + self.r["maint_margin"]
        cap = 1.0 / denom if denom > 0 else self.r["max_leverage"]
        return max(1.0, min(self.r["max_leverage"], cap))

    def size(self, equity: float, entry: float, stop: float, size_mult: float = 1.0) -> tuple[float, float, str]:
        """Return (notional, leverage, reject_reason). notional=0 means 'do not open'.
        size_mult (default 1.0) scales the risk budget by model confidence; the notional is still
        hard-capped by the liquidation-safe leverage, so sizing never becomes unsafe."""
        if equity <= 0 or entry <= 0:
            return 0.0, 0.0, "no equity"
        dist = abs(entry - stop) / entry
        if dist < self.r["min_stop_dist_pct"]:
            return 0.0, 0.0, f"stop too tight ({dist*100:.2f}%)"
        lev = self.liq_safe_leverage(dist)
        risk_budget = equity * self.r["risk_pct_per_trade"] * max(0.0, size_mult)
        notional = min(risk_budget / dist, equity * lev)       # risk-based, capped by safe leverage
        if notional <= 0:
            return 0.0, 0.0, "zero notional"
        return notional, lev, ""

    def entry_scale(self, equity: float, peak: float | None) -> tuple[float, bool]:
        """#2 Equity-curve brake. Return (size_mult, pause). As equity falls below its peak we
        PROACTIVELY shrink new-entry size (graded), and once the drawdown reaches brake_pause_dd we
        pause new entries entirely — long before the hard max_drawdown halt. Flat curve => (1.0, False)."""
        if not self.r.get("equity_brake", True) or not peak or peak <= 0 or equity <= 0:
            return 1.0, False
        dd = max(0.0, 1.0 - equity / peak)                 # drawdown fraction from peak (>=0)
        soft = self.r.get("brake_soft_dd", 0.06)
        pause = self.r.get("brake_pause_dd", 0.12)
        floor = self.r.get("brake_floor", 0.3)
        if dd >= pause:
            return 0.0, True
        if dd <= soft:
            return 1.0, False
        frac = (dd - soft) / max(pause - soft, 1e-9)        # 0 at soft -> 1 at pause
        return float(1.0 - frac * (1.0 - floor)), False

    def admit(self, positions: dict, equity: float, notional: float,
              direction: str | None = None) -> tuple[float, str]:
        """Return the allowed notional (CLAMPED to fit the exposure cap), or 0 with a reason.
        We trim an oversized position to fit rather than refusing to trade at all."""
        if len(positions) >= self.r["max_concurrent"]:
            return 0.0, "max concurrent positions"
        if equity <= 0:
            return 0.0, "no equity"
        # #4 correlation cap: our majors move together, so N same-direction positions are really one
        # oversized bet. Cap concurrent same-direction exposure.
        msd = int(self.r.get("max_same_direction", 0) or 0)
        if direction and msd:
            same = sum(1 for p in positions.values() if p.get("side") == direction)
            if same >= msd:
                return 0.0, f"correlation cap ({direction} x{same})"
        gross = sum(p["notional"] for p in positions.values())
        headroom = equity * self.r["max_total_exposure_pct"] - gross
        if headroom <= equity * 0.02:                      # need at least 2% equity of room to bother
            return 0.0, f"exposure cap reached ({gross/equity:.2f}x)"
        return min(notional, headroom), ""                 # clamp to fit

    def halted(self, day_open: float | None, peak: float | None, equity: float) -> tuple[bool, str]:
        if day_open and (equity / day_open - 1) <= -self.r["daily_loss_limit_pct"]:
            return True, f"daily loss limit ({(equity/day_open-1)*100:.1f}%)"
        if peak and (equity / peak - 1) <= -self.r["max_drawdown_pct"]:
            return True, f"max drawdown ({(equity/peak-1)*100:.1f}%)"
        return False, ""
