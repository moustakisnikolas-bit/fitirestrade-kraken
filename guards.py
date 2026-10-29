"""
Proactive guards — the "be smart, don't bleed" layer.

Pure, side-effect-free functions shared by BOTH the live bot and the backtester (same principle as
strategy.py), so a backtest predicts exactly how the guards would behave live. Two of the four
guards live here (they need the indicator row); the other two live in risk.py (they need equity /
open positions):

  #1 REGIME FILTER      regime_entry_ok()  — refuse NEW entries when the market is a bad place to
                        trade at all: too-dead or too-wild volatility (ATR/price out of band), or
                        deep chop (ADX below floor) with no breakout. Staying flat in a no-edge
                        regime is the single biggest loss-avoider for a strategy with no durable edge.
  #3 VOL-TARGET SIZING  vol_target_mult()  — shrink size when realised volatility is above target
                        (ATR/price). Only ever REDUCES size (capped at 1.0); it folds into the
                        existing `conf` size multiplier, so no execution-layer change is needed.

Guards #2 (equity-curve brake) and #4 (correlation cap) are RiskEngine.entry_scale() / admit().

All thresholds come from the strategy `params` dict (keys prefixed `guard_`), so they persist in
config.json and are tunable from the panel like any other param. Everything degrades safely: a
missing/NaN indicator means "don't block / don't scale", never a crash.
"""
from __future__ import annotations

import pandas as pd


def _num(v):
    """float(v) or None if missing/NaN — guards must never raise on a warmup row."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def regime_entry_ok(row: pd.Series, price: float, p: dict) -> tuple[bool, str]:
    """#1 Regime filter. Return (allowed, reason). reason is '' when allowed.
    Blocks NEW entries in regimes where a no-edge strategy just pays fees:
      * ATR/price below `guard_atr_floor`  -> market too dead (whipsaw/no follow-through)
      * ATR/price above `guard_atr_ceiling`-> market too wild (stop-outs; vol-target also shrinks)
      * ADX below `guard_adx_min`          -> deep chop, UNLESS price prints a fresh breakout
    Only ever BLOCKS; it never forces a trade."""
    if not float(p.get("guard_regime", 1.0)):
        return True, ""
    atr = _num(row.get("atr"))
    if atr is not None and price and price > 0:
        atr_pct = atr / price
        if atr_pct < float(p.get("guard_atr_floor", 0.003)):
            return False, f"dead vol {atr_pct*100:.2f}%"
        if atr_pct > float(p.get("guard_atr_ceiling", 0.08)):
            return False, f"wild vol {atr_pct*100:.2f}%"
    adx = _num(row.get("adx"))
    if adx is not None and adx < float(p.get("guard_adx_min", 18.0)):
        bo = _num(row.get("breakout_high"))
        is_breakout = bo is not None and price and price > bo
        if not is_breakout:
            return False, f"chop adx {adx:.0f}"
    return True, ""


def vol_target_mult(row: pd.Series, price: float, p: dict) -> float:
    """#3 Volatility-target sizing. Return a size multiplier in [guard_vol_floor, 1.0].
    When realised vol (ATR/price) exceeds the target, size down proportionally so each trade risks
    a steadier slice of equity; when vol is at/below target, leave size untouched (never inflate)."""
    if not float(p.get("guard_vol_target", 1.0)):
        return 1.0
    atr = _num(row.get("atr"))
    if atr is None or not price or price <= 0:
        return 1.0
    atr_pct = atr / price
    if atr_pct <= 0:
        return 1.0
    target = float(p.get("guard_atr_target", 0.02))
    floor = float(p.get("guard_vol_floor", 0.35))
    return float(min(1.0, max(floor, target / atr_pct)))
