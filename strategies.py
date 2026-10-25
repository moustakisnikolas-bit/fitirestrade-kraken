"""
Pluggable strategy layer. A strategy ONLY proposes a direction + a stop price; the risk engine
owns sizing/leverage. This keeps strategies simple and swappable from the admin panel, and lets
us add/validate new ones without touching execution or risk.

A strategy implements:
  * prepare(df)      -> df with the indicator columns it needs (incl. 'atr')
  * signal(row,price) -> (direction, stop_price|None, meta)   direction in {long, short, flat}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy import compute_indicators, ensemble_decide, DEFAULT_PARAMS


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


class Strategy:
    name = "base"

    def __init__(self, params: dict):
        self.p = {**DEFAULT_PARAMS, **(params or {})}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def signal(self, row: pd.Series, price: float):
        raise NotImplementedError


class EnsembleStrategy(Strategy):
    """Regime-aware mean-reversion + momentum ensemble (NOTE: proven no OOS edge on majors;
    kept as a baseline / template)."""
    name = "ensemble"

    def prepare(self, df):
        return compute_indicators(df, self.p)

    def signal(self, row, price):
        dec = ensemble_decide(row, self.p)
        atr = row["atr"] if not pd.isna(row.get("atr")) else price * 0.03
        if dec.direction == "BUY":
            return "long", price - self.p["stop_atr"] * atr, {"score": dec.score, "regime": dec.regime}
        if dec.direction == "SELL":
            return "short", price + self.p["stop_atr"] * atr, {"score": dec.score, "regime": dec.regime}
        return "flat", None, {"score": dec.score, "regime": dec.regime}


class BreakoutStrategy(Strategy):
    """Donchian channel breakout (turtle-style). Long on break of the N-bar high, short on the
    N-bar low. (Also showed no OOS edge on majors — real use is for inefficient/new coins.)"""
    name = "breakout"

    def prepare(self, df):
        d = df.copy()
        d["atr"] = _atr(d)
        n = int(self.p.get("breakout_entry", 20))
        d["don_hi"] = d["high"].rolling(n).max().shift(1)
        d["don_lo"] = d["low"].rolling(n).min().shift(1)
        return d

    def signal(self, row, price):
        atr = row["atr"] if not pd.isna(row.get("atr")) else price * 0.03
        s = self.p.get("stop_atr", 3.0)
        if not pd.isna(row.get("don_hi")) and price > row["don_hi"]:
            return "long", price - s * atr, {"breakout": "up"}
        if not pd.isna(row.get("don_lo")) and price < row["don_lo"]:
            return "short", price + s * atr, {"breakout": "down"}
        return "flat", None, {}


REGISTRY = {s.name: s for s in (EnsembleStrategy, BreakoutStrategy)}


def get_strategy(name: str, params: dict) -> Strategy:
    return REGISTRY.get(name, EnsembleStrategy)(params)
