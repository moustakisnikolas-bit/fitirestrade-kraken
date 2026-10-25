"""
Kraken ensemble strategy — SELF-CONTAINED, deliberately SEPARATE from the ML analyst
(same principle as strategies/newlisting.py). It must NEVER import or mutate the shared
XGBoost feature vector (app/agents/xgb_scorer.py) — the live OKX models depend on that
10-feature layout, and breaking it would break real-money trading.

"All of the above", done right (regime-aware ensemble):
  * MEAN-REVERSION sub-signal  — buy statistically cheap, sell statistically rich
                                 (z-score of price vs rolling mean, RSI extreme, Bollinger %b).
  * MOMENTUM sub-signal        — ride established trends (EMA fast/slow, ROC, breakout of the
                                 recent high), its strength scaled by trend strength (ADX).
  * REGIME DETECTOR (ADX)      — decides WHICH sub-signal to trust right now instead of letting
                                 them fight: trending market -> weight momentum, ranging -> weight
                                 mean-reversion. This is what makes it an ensemble and not a coin-flip.

The live bot (added after backtest validation) will auto-train the blend weights + thresholds on
its OWN trade outcomes, writing to its OWN tables (kraken_ens_*), so the learning loop never touches
the live OKX path. These pure functions are imported by BOTH the backtest and the live tick so the
two can never diverge.

Spot, LONG-ONLY (a small Kraken spot account cannot short): a BUY opens/holds, a SELL or the trailing
stop closes. No leverage, no liquidation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:                      # pandas fallbacks keep it runnable on the laptop/CI
    HAS_TALIB = False


DEFAULT_PARAMS: dict = {
    # indicator lengths
    "ema_fast": 12,
    "ema_slow": 50,
    "rsi_len": 14,
    "bb_len": 20,
    "bb_k": 2.0,
    "atr_len": 14,
    "adx_len": 14,
    "roc_len": 10,
    "z_len": 20,
    "breakout_len": 20,          # lookback for the momentum breakout high
    # regime
    "adx_trend": 25.0,           # ADX >= this => trending regime, else ranging
    # mean-reversion thresholds
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "bb_buy": 0.10,              # %b below this = statistically cheap
    "bb_sell": 0.90,             # %b above this = statistically rich
    "z_buy": -1.5,               # z-score below this = cheap
    "z_sell": 1.5,
    # decision thresholds on the net ensemble score in [-1, +1]
    "enter_thr": 0.35,           # net score >= this => BUY (enter/hold)
    "exit_thr": -0.20,           # net score <= this => SELL (signal exit)
    # risk management (mirrors newlisting.py's proven profit-lock + trail idea)
    "stop_atr": 2.5,             # initial stop = entry - stop_atr * ATR
    "profit_lock": 0.015,        # once +1.5% in profit, never give the gain fully back
    "trail": 0.012,              # trail 1.2% from the favourable extreme after profit_lock hit
    "max_hold_bars": 96,         # hard time-stop (e.g. 96 * 1h = 4 days) to free capital
}


@dataclass
class SubSignal:
    direction: str               # "BUY" | "SELL" | "HOLD"
    strength: float              # 0.0 .. 1.0
    reason: str = ""


@dataclass
class Decision:
    direction: str               # "BUY" | "SELL" | "HOLD"
    score: float                 # net ensemble score in [-1, +1]
    regime: str                  # "trend" | "range"
    reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- indicators
def compute_indicators(df: pd.DataFrame, p: dict | None = None) -> pd.DataFrame:
    """Add indicator columns to an OHLCV frame (columns: open, high, low, close, volume).
    Returns a copy; rows with insufficient look-back carry NaN and are skipped by the caller."""
    p = {**DEFAULT_PARAMS, **(p or {})}
    out = df.copy()
    close, high, low = out["close"].to_numpy(float), out["high"].to_numpy(float), out["low"].to_numpy(float)

    if HAS_TALIB:
        out["ema_fast"] = talib.EMA(close, p["ema_fast"])
        out["ema_slow"] = talib.EMA(close, p["ema_slow"])
        out["rsi"] = talib.RSI(close, p["rsi_len"])
        out["atr"] = talib.ATR(high, low, close, p["atr_len"])
        out["adx"] = talib.ADX(high, low, close, p["adx_len"])
        out["roc"] = talib.ROC(close, p["roc_len"])
    else:
        s = pd.Series(close, index=out.index)
        out["ema_fast"] = s.ewm(span=p["ema_fast"], adjust=False).mean()
        out["ema_slow"] = s.ewm(span=p["ema_slow"], adjust=False).mean()
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(p["rsi_len"]).mean()
        loss = (-delta.clip(upper=0)).rolling(p["rsi_len"]).mean()
        out["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-12))
        tr = pd.concat([(out["high"] - out["low"]),
                        (out["high"] - s.shift()).abs(),
                        (out["low"] - s.shift()).abs()], axis=1).max(axis=1)
        out["atr"] = tr.rolling(p["atr_len"]).mean()
        out["adx"] = _adx_fallback(out, p["adx_len"])
        out["roc"] = s.pct_change(p["roc_len"]) * 100.0

    mid = out["close"].rolling(p["bb_len"]).mean()
    std = out["close"].rolling(p["bb_len"]).std(ddof=0)
    upper, lower = mid + p["bb_k"] * std, mid - p["bb_k"] * std
    out["bb_pct"] = (out["close"] - lower) / (upper - lower + 1e-12)

    zmean = out["close"].rolling(p["z_len"]).mean()
    zstd = out["close"].rolling(p["z_len"]).std(ddof=0)
    out["zscore"] = (out["close"] - zmean) / (zstd + 1e-12)

    out["breakout_high"] = out["high"].rolling(p["breakout_len"]).max().shift(1)
    return out


def _adx_fallback(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder-ish ADX without talib (good enough for regime gating, not for display)."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).mean() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / (atr + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.rolling(n).mean()


# --------------------------------------------------------------------------- sub-signals
def mean_reversion_signal(row: pd.Series, p: dict) -> SubSignal:
    """Cheap => BUY, rich => SELL. Strength grows with how extreme the stretch is."""
    votes, strengths, why = [], [], []
    if row["bb_pct"] <= p["bb_buy"]:
        votes.append(1); strengths.append(min(1.0, (p["bb_buy"] - row["bb_pct"]) / 0.1 + 0.5)); why.append(f"%b {row['bb_pct']:.2f} low")
    elif row["bb_pct"] >= p["bb_sell"]:
        votes.append(-1); strengths.append(min(1.0, (row["bb_pct"] - p["bb_sell"]) / 0.1 + 0.5)); why.append(f"%b {row['bb_pct']:.2f} high")
    if row["rsi"] <= p["rsi_oversold"]:
        votes.append(1); strengths.append(min(1.0, (p["rsi_oversold"] - row["rsi"]) / 15 + 0.5)); why.append(f"RSI {row['rsi']:.0f} oversold")
    elif row["rsi"] >= p["rsi_overbought"]:
        votes.append(-1); strengths.append(min(1.0, (row["rsi"] - p["rsi_overbought"]) / 15 + 0.5)); why.append(f"RSI {row['rsi']:.0f} overbought")
    if row["zscore"] <= p["z_buy"]:
        votes.append(1); strengths.append(min(1.0, abs(row["zscore"] - p["z_buy"]) / 1.5 + 0.4)); why.append(f"z {row['zscore']:.1f} cheap")
    elif row["zscore"] >= p["z_sell"]:
        votes.append(-1); strengths.append(min(1.0, abs(row["zscore"] - p["z_sell"]) / 1.5 + 0.4)); why.append(f"z {row['zscore']:.1f} rich")
    if not votes:
        return SubSignal("HOLD", 0.0, "MR neutral")
    net = np.sign(sum(votes))
    strength = float(np.mean(strengths)) * (abs(sum(votes)) / len(votes))
    return SubSignal("BUY" if net > 0 else "SELL", round(min(1.0, strength), 3), "; ".join(why))


def momentum_signal(row: pd.Series, p: dict) -> SubSignal:
    """Trend-following: up-trend + positive ROC + breakout => BUY; mirror => SELL.
    Strength scales with trend strength (ADX) so weak 'trends' don't get oversized conviction."""
    trend_up = row["ema_fast"] > row["ema_slow"]
    roc_up = row["roc"] > 0
    breakout = row["close"] > row["breakout_high"] if not np.isnan(row["breakout_high"]) else False
    adx_scale = float(np.clip((row["adx"] - 15) / 25, 0.0, 1.0))   # 0 at ADX15 -> 1 at ADX40
    if trend_up and roc_up:
        strength = (0.5 + 0.5 * breakout) * (0.4 + 0.6 * adx_scale)
        return SubSignal("BUY", round(min(1.0, strength), 3),
                         f"up-trend roc{row['roc']:.1f}{' +breakout' if breakout else ''} adx{row['adx']:.0f}")
    if (not trend_up) and (not roc_up):
        strength = 0.5 * (0.4 + 0.6 * adx_scale)
        return SubSignal("SELL", round(min(1.0, strength), 3), f"down-trend roc{row['roc']:.1f} adx{row['adx']:.0f}")
    return SubSignal("HOLD", 0.0, "mixed trend")


def ensemble_decide(row: pd.Series, p: dict | None = None, weights: dict | None = None) -> Decision:
    """Regime-aware blend. In a TREND, momentum carries most weight; in a RANGE, mean-reversion does.
    `weights` lets the live bot override the default blend with what it has LEARNED from its own trades."""
    p = {**DEFAULT_PARAMS, **(p or {})}
    if any(pd.isna(row.get(c)) for c in ("ema_slow", "rsi", "adx", "bb_pct", "zscore")):
        return Decision("HOLD", 0.0, "warmup", ["insufficient history"])

    regime = "trend" if row["adx"] >= p["adx_trend"] else "range"
    w = weights or ({"momentum": 0.70, "mean_rev": 0.30} if regime == "trend"
                    else {"momentum": 0.30, "mean_rev": 0.70})

    mr = mean_reversion_signal(row, p)
    mo = momentum_signal(row, p)

    def signed(sig: SubSignal) -> float:
        return {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}[sig.direction] * sig.strength

    score = w["mean_rev"] * signed(mr) + w["momentum"] * signed(mo)
    reasons = [f"[{regime}]", f"MR({mr.direction} {mr.strength}): {mr.reason}",
               f"MO({mo.direction} {mo.strength}): {mo.reason}"]

    if score >= p["enter_thr"]:
        direction = "BUY"
    elif score <= p["exit_thr"]:
        direction = "SELL"
    else:
        direction = "HOLD"
    return Decision(direction, round(float(score), 3), regime, reasons)
