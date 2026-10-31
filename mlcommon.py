"""
Shared ML helpers used by BOTH training (mltrain.py) and the live strategies (strategies.py).

Kept deliberately light: imports only pandas/numpy/joblib + strategy.compute_indicators. It does
NOT import lightgbm or sklearn at module load, so importing it from the bot is cheap; lightgbm is
pulled in implicitly only when a saved model is unpickled (joblib.load) for prediction.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from strategy import compute_indicators

MODELS_DIR = os.path.join(os.getenv("KF_DATA_DIR", "data"), "models")

# Feature columns fed to the model. Derived from compute_indicators() output + light engineering
# + market sentiment (Fear & Greed). Models store their OWN feature list, so adding one here is
# backward-compatible: old models keep predicting on their saved list; only newly trained models
# use the extended set.
FEATURES = [
    "rsi", "adx", "roc", "bb_pct", "zscore",
    "ema_ratio", "atr_pct", "ret_1", "ret_3", "ret_6", "vol_6", "dist_bo", "hour", "fng",
    # candle / chart-pattern features (classic TA the model can learn to weight)
    "body_pct", "upper_wick", "lower_wick", "engulf", "wick_sig",
    "trend_struct", "range_pos", "vol_z",
    # derivatives-positioning sentiment — the signals that actually drive PERP direction and that
    # pure TA can't see (funding crowding, spot-perp basis, composite). Require symbol+timeframe.
    "funding_z", "basis_z", "sent_score",
]


def build_features(df: pd.DataFrame, params: dict,
                   symbol: str | None = None, timeframe: str | None = None) -> pd.DataFrame:
    """Return a frame carrying BOTH the raw indicator columns (needed by ensemble_decide and for
    the ATR stop) AND the engineered FEATURES columns, on the same index as the input bars.
    When symbol/timeframe are given, the derivatives-sentiment features are filled from history.db
    (via sentiment.build_sentiment_columns); otherwise they default to neutral 0 so the matrix is
    always complete and old symbol-agnostic callers keep working."""
    ind = compute_indicators(df, params)
    close = ind["close"]
    out = ind.copy()
    out["ema_ratio"] = ind["ema_fast"] / ind["ema_slow"] - 1.0
    out["atr_pct"] = ind["atr"] / close
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["vol_6"] = close.pct_change().rolling(6).std()
    out["dist_bo"] = close / ind["breakout_high"] - 1.0
    out["hour"] = out.index.hour.astype(float)

    # ---- candle / chart-pattern features (hand-coded classic TA; captures reversals + drops/growth)
    o, h, l = ind["open"], ind["high"], ind["low"]
    rng = (h - l) + 1e-9
    body_top = np.maximum(o, close)
    body_bot = np.minimum(o, close)
    out["body_pct"] = (close - o) / rng               # signed body size (bull/bear momentum of bar)
    out["upper_wick"] = (h - body_top) / rng
    out["lower_wick"] = (body_bot - l) / rng
    prev_o, prev_c = o.shift(), close.shift()
    bull_eng = (close > o) & (prev_c < prev_o) & (close >= prev_o) & (o <= prev_c)
    bear_eng = (close < o) & (prev_c > prev_o) & (close <= prev_o) & (o >= prev_c)
    out["engulf"] = bull_eng.astype(float) - bear_eng.astype(float)          # +1 bullish / -1 bearish
    hammer = (out["lower_wick"] > 0.5) & (out["body_pct"].abs() < 0.3)
    star = (out["upper_wick"] > 0.5) & (out["body_pct"].abs() < 0.3)
    out["wick_sig"] = hammer.astype(float) - star.astype(float)             # hammer vs shooting-star
    hh = h > h.rolling(10).max().shift(1)
    ll = l < l.rolling(10).min().shift(1)
    out["trend_struct"] = hh.astype(float) - ll.astype(float)               # breakout up / breakdown
    rhigh, rlow = h.rolling(20).max(), l.rolling(20).min()
    out["range_pos"] = ((close - rlow) / (rhigh - rlow + 1e-9) * 2 - 1)     # -1 range low .. +1 high
    vmean = ind["volume"].rolling(20).mean()
    vstd = ind["volume"].rolling(20).std(ddof=0)
    out["vol_z"] = ((ind["volume"] - vmean) / (vstd + 1e-9)).clip(-4, 4)    # volume spike z-score

    try:
        from sentiment import align_fng
        out["fng"] = align_fng(out.index)       # market Fear & Greed, normalised [-1,1], 0 if absent
    except Exception:
        out["fng"] = 0.0

    # ---- derivatives-positioning sentiment (funding crowding / spot-perp basis / composite) ----
    for c in ("funding_z", "basis_z", "sent_score"):
        out[c] = 0.0
    if symbol:
        try:
            from sentiment import build_sentiment_columns
            sc = build_sentiment_columns(symbol, timeframe, out.index, out["close"], params)
            for c in ("funding_z", "basis_z", "sent_score"):
                out[c] = sc[c].to_numpy()
        except Exception:
            pass
    return out


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The model's X: just the FEATURES columns (callers drop NaN rows themselves)."""
    return frame[FEATURES]


def size_mult(conf: float, threshold: float, enabled: bool = True) -> float:
    """Map a model confidence in [threshold, 1] to a position-size multiplier in [0.5, 1.5].
    At conf == threshold -> 0.5x; at conf == 1.0 -> 1.5x. Returns 1.0 when disabled."""
    if not enabled:
        return 1.0
    span = max(1.0 - threshold, 1e-6)
    return float(min(1.5, max(0.5, 0.5 + (conf - threshold) / span)))


def _safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def model_path(symbol: str, timeframe: str, mode: str) -> str:
    return os.path.join(MODELS_DIR, f"{_safe_name(symbol)}_{timeframe}_{mode}.pkl")


_cache: dict = {}  # (symbol,tf,mode) -> (mtime, obj)


def load_model(symbol: str, timeframe: str, mode: str):
    """Load a saved model bundle (dict) or None if not trained. Cached by file mtime so a retrain
    is picked up without a restart."""
    p = model_path(symbol, timeframe, mode)
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return None
    hit = _cache.get((symbol, timeframe, mode))
    if hit and hit[0] == mtime:
        return hit[1]
    import joblib  # local import: only needed when a model actually exists
    obj = joblib.load(p)
    _cache[(symbol, timeframe, mode)] = (mtime, obj)
    return obj


def model_status(symbol: str, timeframe: str, mode: str) -> dict | None:
    """Lightweight status for the panel: metadata only (no heavy objects echoed)."""
    obj = load_model(symbol, timeframe, mode)
    if not obj:
        return None
    return {
        "symbol": symbol, "timeframe": timeframe, "mode": mode,
        "trained_at": obj.get("trained_at"),
        "n_samples": obj.get("n_samples"),
        "train_start": obj.get("train_start"),
        "train_end": obj.get("train_end"),
        "oos": obj.get("oos", {}),
    }
