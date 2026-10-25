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

    def size(self, equity: float, entry: float, stop: float) -> tuple[float, float, str]:
        """Return (notional, leverage, reject_reason). notional=0 means 'do not open'."""
        if equity <= 0 or entry <= 0:
            return 0.0, 0.0, "no equity"
        dist = abs(entry - stop) / entry
        if dist < self.r["min_stop_dist_pct"]:
            return 0.0, 0.0, f"stop too tight ({dist*100:.2f}%)"
        lev = self.liq_safe_leverage(dist)
        risk_budget = equity * self.r["risk_pct_per_trade"]
        notional = min(risk_budget / dist, equity * lev)       # risk-based, capped by safe leverage
        if notional <= 0:
            return 0.0, 0.0, "zero notional"
        return notional, lev, ""

    def admit(self, positions: dict, equity: float, notional: float) -> tuple[float, str]:
        """Return the allowed notional (CLAMPED to fit the exposure cap), or 0 with a reason.
        We trim an oversized position to fit rather than refusing to trade at all."""
        if len(positions) >= self.r["max_concurrent"]:
            return 0.0, "max concurrent positions"
        if equity <= 0:
            return 0.0, "no equity"
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
