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
from mlcommon import build_features, load_model, size_mult
from guards import regime_entry_ok, vol_target_mult


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


class Strategy:
    name = "base"

    def __init__(self, params: dict):
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        # the bot/backtester set these before calling so per-symbol models can be loaded
        self.symbol: str | None = None
        self.timeframe: str | None = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _signal(self, row: pd.Series, price: float):
        """Raw strategy decision -> (direction, stop|None, meta). Subclasses implement this."""
        raise NotImplementedError

    def signal(self, row: pd.Series, price: float):
        """Public entry point: raw signal + PROACTIVE GUARDS (#1 regime filter, #3 vol-target
        sizing) applied uniformly to every strategy. Guards only ever block an entry or shrink its
        size — they never create or flip a trade. Shared by the live bot and the backtester."""
        direction, stop, meta = self._signal(row, price)
        if direction not in ("long", "short"):
            return direction, stop, meta
        ok, why = regime_entry_ok(row, price, self.p)
        if not ok:
            m = dict(meta or {}); m["blocked"] = why
            return "flat", None, m
        vm = vol_target_mult(row, price, self.p)
        if vm != 1.0:
            m = dict(meta or {})
            m["conf"] = float(m.get("conf", 1.0)) * vm
            m["vol_mult"] = round(vm, 3)
            meta = m
        return direction, stop, meta


class EnsembleStrategy(Strategy):
    """Regime-aware mean-reversion + momentum ensemble (NOTE: proven no OOS edge on majors;
    kept as a baseline / template)."""
    name = "ensemble"

    def prepare(self, df):
        return compute_indicators(df, self.p)

    def _signal(self, row, price):
        dec = ensemble_decide(row, self.p)
        atr = row["atr"] if not pd.isna(row.get("atr")) else price * 0.03
        if dec.direction == "BUY":
            return "long", price - self.p["stop_atr"] * atr, {"score": dec.score, "regime": dec.regime}
        if dec.direction == "SELL":
            return "short", price + self.p["stop_atr"] * atr, {"score": dec.score, "regime": dec.regime}
        return "flat", None, {"score": dec.score, "regime": dec.regime}


class SentEnsembleStrategy(EnsembleStrategy):
    """Ensemble + SPECIALIZED SENTIMENT overlay (funding-rate crowding, spot-perp basis, fear&greed).
    Sentiment does three things, each validated via backtest before go-live (docs/ml/phase-7):
      * TILT   — when sentiment agrees with the trade, lower the entry bar so a milder ensemble
                 signal can trigger (sentiment helps you enter); when it disagrees, keep the bar.
      * FADE   — when funding is extremely crowded in the SAME direction as the trade (squeeze
                 risk), shrink size (or veto). Contrarian at the extremes.
      * SIZE   — scale confidence up when sentiment agrees, down when it disagrees.
    Falls back to the plain ensemble if sentiment data is missing (columns default to neutral 0)."""
    name = "ensemble_sent"

    def prepare(self, df):
        d = compute_indicators(df, self.p)
        cols = ("funding", "funding_z", "basis", "basis_z", "fng_n", "sent_score")
        try:
            import sentiment as S
            sc = S.build_sentiment_columns(self.symbol, self.timeframe, d.index, d["close"], self.p)
            for c in cols:
                d[c] = sc[c].to_numpy()
        except Exception:
            for c in cols:
                d[c] = 0.0
        return d

    def _signal(self, row, price):
        if not float(self.p.get("sent_enabled", 1.0)):
            return super()._signal(row, price)
        dec = ensemble_decide(row, self.p)
        ss = float(row.get("sent_score") or 0.0)      # bullish score in [-1,1]
        fz = float(row.get("funding_z") or 0.0)        # funding crowding z-score
        tilt = float(self.p.get("sent_tilt", 0.15))
        long_thr = self.p["enter_thr"] - tilt * max(0.0, ss)     # bullish sentiment -> easier long
        short_thr = self.p["exit_thr"] + tilt * max(0.0, -ss)    # bearish sentiment -> easier short
        if dec.score >= long_thr:
            d = "long"
        elif dec.score <= short_thr:
            d = "short"
        else:
            return "flat", None, {"score": dec.score, "sent": round(ss, 2), "regime": dec.regime}
        veto = float(self.p.get("sent_veto", 0.0))
        if veto > 0 and ((d == "long" and ss <= -veto) or (d == "short" and ss >= veto)):
            return "flat", None, {"score": dec.score, "sent": round(ss, 2), "veto": True}
        conf = 1.0
        if (d == "long" and fz >= self.p.get("sent_fade_z", 2.0)) or \
           (d == "short" and fz <= -self.p.get("sent_fade_z", 2.0)):
            conf *= float(self.p.get("sent_fade_floor", 0.4))    # crowded same-direction -> fade
        size = float(self.p.get("sent_size", 0.3))
        agree = (ss > 0 and d == "long") or (ss < 0 and d == "short")
        conf *= (1.0 + size * abs(ss)) if agree else max(0.2, 1.0 - size * abs(ss))
        atr = row["atr"] if not pd.isna(row.get("atr")) else price * 0.03
        meta = {"score": round(dec.score, 3), "sent": round(ss, 2), "fz": round(fz, 2),
                "conf": round(conf, 3), "regime": dec.regime}
        if d == "long":
            return "long", price - self.p["stop_atr"] * atr, meta
        return "short", price + self.p["stop_atr"] * atr, meta


class FundingFadeStrategy(SentEnsembleStrategy):
    """#3 Structural edge: fade crowded funding. Very negative funding z (shorts crowded, longs get
    PAID the funding) -> go long; very positive (longs crowded, shorts paid) -> go short. A documented
    crypto-perp carry/mean-reversion effect, independent of TA. Reuses SentEnsembleStrategy.prepare()
    for the funding_z column. ATR stop; guards (regime/vol) still apply via base.signal."""
    name = "funding_fade"

    def _signal(self, row, price):
        fz = float(row.get("funding_z") or 0.0)
        z = float(self.p.get("ff_z", 1.5))
        follow = float(self.p.get("ff_follow", 0.0)) >= 1.0   # 0=fade crowd, 1=follow crowd/trend
        atr = row["atr"] if not pd.isna(row.get("atr")) else price * 0.03
        s = self.p.get("stop_atr", 2.5)
        long_cond = (fz >= z) if follow else (fz <= -z)
        short_cond = (fz <= -z) if follow else (fz >= z)
        if long_cond:
            return "long", price - s * atr, {"fz": round(fz, 2)}
        if short_cond:
            return "short", price + s * atr, {"fz": round(fz, 2)}
        return "flat", None, {"fz": round(fz, 2)}


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

    def _signal(self, row, price):
        atr = row["atr"] if not pd.isna(row.get("atr")) else price * 0.03
        s = self.p.get("stop_atr", 3.0)
        if not pd.isna(row.get("don_hi")) and price > row["don_hi"]:
            return "long", price - s * atr, {"breakout": "up"}
        if not pd.isna(row.get("don_lo")) and price < row["don_lo"]:
            return "short", price + s * atr, {"breakout": "down"}
        return "flat", None, {}


class _MLBase(Strategy):
    """Shared base for ML strategies. Loads a per-(symbol, timeframe) LightGBM model lazily; if
    no model is trained yet it stays flat (and warns once) so the bot never crashes."""
    name = "ml_base"
    mode = "meta"

    def __init__(self, params: dict):
        super().__init__(params)
        self._warned = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df, self.p, self.symbol, self.timeframe)

    def _model(self):
        if not self.symbol or not self.timeframe:
            return None
        return load_model(self.symbol, self.timeframe, self.mode)

    def _feats(self, m, row):
        """Return the feature vector in the model's order, or None if any value is missing."""
        vals = []
        for f in m["features"]:
            v = row.get(f)
            if v is None or pd.isna(v):
                return None
            vals.append(float(v))
        return [vals]

    def _flat(self, why):
        if not self._warned:
            print(f"  [{self.name}] {self.symbol} {self.timeframe}: {why}", flush=True)
            self._warned = True
        return "flat", None, {"ml": why}

    def _atr(self, row, price):
        a = row.get("atr")
        return float(a) if (a is not None and not pd.isna(a)) else price * 0.03


class MetaStrategy(_MLBase):
    """Meta-labeling: the ensemble proposes; the model decides whether to take it + how big."""
    name = "ml_meta"
    mode = "meta"

    def _signal(self, row, price):
        m = self._model()
        if m is None:
            return self._flat("no meta model (train it)")
        dec = ensemble_decide(row, self.p)
        if dec.direction == "HOLD":
            return "flat", None, {}
        X = self._feats(m, row)
        if X is None:
            return "flat", None, {}
        prob = float(m["model"].predict_proba(X)[0][1])
        thr = self.p.get("ml_threshold", 0.55)
        if prob < thr:
            return "flat", None, {"p_win": round(prob, 3)}
        conf = size_mult(prob, thr, bool(self.p.get("ml_size_scaling", 1.0)))
        atr = self._atr(row, price)
        meta = {"p_win": round(prob, 3), "conf": conf, "regime": dec.regime}
        if dec.direction == "BUY":
            return "long", price - self.p["stop_atr"] * atr, meta
        return "short", price + self.p["stop_atr"] * atr, meta


class DirStrategy(_MLBase):
    """Direct direction: the model predicts up/flat/down and proposes the trade itself."""
    name = "ml_dir"
    mode = "dir"

    def _signal(self, row, price):
        m = self._model()
        if m is None:
            return self._flat("no dir model (train it)")
        X = self._feats(m, row)
        if X is None:
            return "flat", None, {}
        pr = m["model"].predict_proba(X)[0]   # [flat, up, down]
        cls = int(np.argmax(pr))
        conf_raw = float(pr[cls])
        thr = self.p.get("ml_threshold", 0.55)
        if cls == 0 or conf_raw < thr:
            return "flat", None, {"p": round(conf_raw, 3)}
        conf = size_mult(conf_raw, thr, bool(self.p.get("ml_size_scaling", 1.0)))
        atr = self._atr(row, price)
        meta = {"p": round(conf_raw, 3), "conf": conf, "cls": ("up" if cls == 1 else "down")}
        if cls == 1:
            return "long", price - self.p["stop_atr"] * atr, meta
        return "short", price + self.p["stop_atr"] * atr, meta


REGISTRY = {s.name: s for s in (EnsembleStrategy, SentEnsembleStrategy, FundingFadeStrategy,
                                BreakoutStrategy, MetaStrategy, DirStrategy)}


def get_strategy(name: str, params: dict) -> Strategy:
    return REGISTRY.get(name, EnsembleStrategy)(params)
