"""
Historical backtester for the paper platform.

It deliberately reuses the SAME pieces the live bot uses so a backtest predicts how the
live bot would actually have behaved:
  * the selected strategy's prepare()/signal()  (strategies.py / strategy.py)
  * the RiskEngine for sizing + admission        (risk.py)
  * the exact fill / trailing-stop / time-stop / liquidation math from paper_engine.py
    and the manage_open()/try_enter() flow from bot.py — re-implemented here in memory.

No database, no control table, no side effects beyond fetching public OHLCV candles.
Single-symbol: it simulates one market at a time (the live concurrency cap is irrelevant
to a one-symbol sweep, and per-symbol is the honest way to read an edge).
"""
from __future__ import annotations

import math

import ccxt
import pandas as pd

from risk import RiskEngine
from strategies import get_strategy

# minutes per timeframe, for annualising the Sharpe ratio
_TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
           "4h": 240, "6h": 360, "12h": 720, "1d": 1440}


def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Prefer the deep stored history (data/history.db, backfilled by mldata.py); fall back to a
    live ccxt pull (a single request, ~2000 candles max) when nothing is stored yet."""
    try:
        from mldata import load_history
        h = load_history(symbol, timeframe)
    except Exception:
        h = None
    if h is not None and len(h) >= 120:
        return h.tail(int(limit)).copy()

    ex = ccxt.krakenfutures({"enableRateLimit": True})
    ex.load_markets()
    if symbol not in ex.markets:
        raise ValueError(f"unknown symbol: {symbol}")
    raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    if not raw or len(raw) < 60:
        raise ValueError(f"not enough candles returned ({len(raw) if raw else 0})")
    df = pd.DataFrame(raw, columns=["t", "open", "high", "low", "close", "volume"]).astype(float)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.drop(columns=["t"])


class _Sim:
    """In-memory single-account mirror of PaperFutures (same fee/pnl/liquidation math)."""

    def __init__(self, start_balance: float, taker_fee: float, maint: float):
        self.balance = start_balance
        self.fee = taker_fee
        self.maint = maint
        self.positions: dict = {}
        self.trades: list = []

    @staticmethod
    def _dir(side: str) -> int:
        return 1 if side == "long" else -1

    def unrealized(self, sym: str, price: float) -> float:
        p = self.positions.get(sym)
        return 0.0 if not p else p["qty"] * (price - p["entry"]) * self._dir(p["side"])

    def equity(self, price_map: dict) -> float:
        eq = self.balance
        for s, p in self.positions.items():
            if s in price_map:
                eq += self.unrealized(s, price_map[s])
        return eq

    def open(self, sym, side, price, notional, lev, stop):
        if sym in self.positions or notional <= 0 or price <= 0:
            return
        qty = notional / price
        self.balance -= notional * self.fee
        self.positions[sym] = dict(side=side, entry=price, qty=qty, notional=notional,
                                   leverage=lev, stop=stop, extreme=price, bars=0)
        self.trades.append(dict(action="OPEN", side=side, price=price, pnl=0.0, reason=""))

    def close(self, sym, price, reason) -> float:
        p = self.positions.get(sym)
        if not p:
            return 0.0
        pnl = p["qty"] * (price - p["entry"]) * self._dir(p["side"])
        fee = p["qty"] * price * self.fee
        net = pnl - fee
        self.balance += net
        del self.positions[sym]
        self.trades.append(dict(action="CLOSE", side=p["side"], price=price, pnl=net, reason=reason))
        return net

    def liquidated(self, sym, price) -> bool:
        p = self.positions.get(sym)
        if not p:
            return False
        margin = p["notional"] / p["leverage"]
        if self.unrealized(sym, price) <= -(margin - self.maint * p["notional"]):
            self.close(sym, price, "LIQUIDATION")
            return True
        return False


def _manage(sim: _Sim, sym, row, price, p) -> bool:
    """Mirror of bot.manage_open: trailing lock, stop, time-stop, liquidation."""
    pos = sim.positions.get(sym)
    if not pos:
        return False
    pos["bars"] += 1
    is_long = pos["side"] == "long"
    pos["extreme"] = max(pos["extreme"], row["high"]) if is_long else min(pos["extreme"], row["low"])
    if is_long and pos["extreme"] >= pos["entry"] * (1 + p["profit_lock"]):
        pos["stop"] = max(pos["stop"], pos["extreme"] * (1 - p["trail"]))
    elif (not is_long) and pos["extreme"] <= pos["entry"] * (1 - p["profit_lock"]):
        pos["stop"] = min(pos["stop"], pos["extreme"] * (1 + p["trail"]))
    if sim.liquidated(sym, price):
        return True
    hit = (price <= pos["stop"]) if is_long else (price >= pos["stop"])
    if hit:
        sim.close(sym, price, "stop/trail")
        return True
    if pos["bars"] >= p["max_hold_bars"]:
        sim.close(sym, price, "time-stop")
        return True
    return False


def _enter(sim: _Sim, risk: RiskEngine, sym, direction, stop, price, equity, conf: float = 1.0,
           peak: float | None = None):
    """Mirror of bot.try_enter: equity brake, flip, risk-size (conf*scale), corr+exposure admit, open."""
    pos = sim.positions.get(sym)
    if pos and pos["side"] != direction:
        sim.close(sym, price, f"flip->{direction}")
        pos = None
    if pos:
        return
    scale, pause = risk.entry_scale(equity, peak)           # #2 equity-curve brake
    if pause:
        return
    notional, lev, why = risk.size(equity, price, stop, conf * scale)
    if notional <= 0:
        return
    notional, reason = risk.admit(sim.positions, equity, notional, direction)  # exposure + corr cap
    if notional <= 0:
        return
    sim.open(sym, direction, price, notional, round(lev, 2), stop)


def _apply_funding(sim: _Sim, sym, price, rate_hourly, hours):
    """Mirror of PaperFutures.apply_funding (longs pay when rate>0), scaled to the bar's hours."""
    p = sim.positions.get(sym)
    if not p or not rate_hourly:
        return
    notional_now = p["qty"] * price
    sim.balance += (-sim._dir(p["side"])) * rate_hourly * hours * notional_now


def fetch_funding_rates(symbols) -> dict:
    """Current funding rate per symbol, used as a constant proxy over the backtest window — the
    same rate source the live bot reads (ex.fetch_funding_rate), so the backtest matches live's
    own funding convention rather than trying to reconstruct history."""
    out = {}
    try:
        ex = ccxt.krakenfutures({"enableRateLimit": True})
        ex.load_markets()
        for s in symbols:
            try:
                out[s] = float(ex.fetch_funding_rate(s).get("fundingRate") or 0.0)
            except Exception:
                out[s] = 0.0
    except Exception:
        pass
    return out


def _metrics(curve: list, trades: list, start_balance: float, timeframe: str) -> dict:
    closes = [t for t in trades if t["action"] == "CLOSE"]
    pnls = [t["pnl"] for t in closes]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    final_eq = curve[-1][1] if curve else start_balance

    # max drawdown off the equity curve
    peak, max_dd = -1e18, 0.0
    for _, eq in curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = min(max_dd, eq / peak - 1)

    # annualised Sharpe from per-bar equity returns
    sharpe = None
    if len(curve) > 2:
        eqs = [e for _, e in curve]
        rets = [eqs[i] / eqs[i - 1] - 1 for i in range(1, len(eqs)) if eqs[i - 1] > 0]
        if len(rets) > 1:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            sd = math.sqrt(var)
            if sd > 0:
                ppy = (365 * 24 * 60) / _TF_MIN.get(timeframe, 60)
                sharpe = (mean / sd) * math.sqrt(ppy)

    return {
        "final_equity": round(final_eq, 2),
        "total_return_pct": round((final_eq / start_balance - 1) * 100, 2),
        "n_trades": len(closes),
        "win_rate_pct": round(100 * len(wins) / len(closes), 1) if closes else None,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
        "avg_trade": round(sum(pnls) / len(pnls), 2) if pnls else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
    }


def run_backtest(symbol: str, timeframe: str, strategy: str, params: dict, risk_cfg: dict,
                 start_balance: float, taker_fee: float, limit: int = 1500,
                 funding: bool = True) -> dict:
    limit = max(120, min(int(limit), 20000))   # stored history supports deep backtests
    df = fetch_ohlcv(symbol, timeframe, limit)
    strat = get_strategy(strategy, params)
    strat.symbol = symbol
    strat.timeframe = timeframe
    ind = strat.prepare(df)

    risk = RiskEngine(risk_cfg)
    sim = _Sim(start_balance, taker_fee, risk.r["maint_margin"])
    tf_hours = _TF_MIN.get(timeframe, 60) / 60.0
    rate = fetch_funding_rates([symbol]).get(symbol, 0.0) if funding else 0.0

    # skip the warm-up span where indicators are still NaN
    win_keys = ("ema_slow", "bb_len", "z_len", "breakout_len", "atr_len",
                "adx_len", "roc_len", "rsi_len", "ema_fast")
    warmup = int(max([params.get(k, 0) or 0 for k in win_keys] + [50])) + 5

    curve: list = []
    peak = start_balance                      # running equity peak for the #2 equity-curve brake
    n = len(ind)
    for i in range(min(warmup, n), n):
        row = ind.iloc[i]
        price = float(row["close"])
        if math.isnan(price):
            continue
        _apply_funding(sim, symbol, price, rate, tf_hours)
        closed = _manage(sim, symbol, row, price, params)
        eq_now = sim.equity({symbol: price})
        peak = max(peak, eq_now)
        try:
            direction, stop, meta = strat.signal(row, price)
        except Exception:
            direction, stop, meta = "flat", None, {}
        if not (closed or direction == "flat" or stop is None):
            conf = float((meta or {}).get("conf", 1.0))
            _enter(sim, risk, symbol, direction, stop, price, eq_now, conf, peak)
        curve.append((ind.index[i].isoformat(), sim.equity({symbol: price})))

    # mark-to-market: force-close whatever is open at the final price (so metrics are complete)
    if sim.positions and curve:
        sim.close(symbol, float(ind.iloc[-1]["close"]), "backtest-end")

    summary = _metrics(curve, sim.trades, start_balance, timeframe)
    summary.update(symbol=symbol, timeframe=timeframe, strategy=strategy,
                   candles=n, start_balance=start_balance,
                   period_start=curve[0][0] if curve else None,
                   period_end=curve[-1][0] if curve else None)

    # down-sample the curve for the chart (<= ~400 points)
    step = max(1, len(curve) // 400)
    equity_curve = [{"ts": t, "equity": round(e, 2)} for t, e in curve[::step]]

    return {"summary": summary, "equity_curve": equity_curve, "n_closed": summary["n_trades"]}


def load_portfolio_frames(symbols: list, timeframe: str, strategy: str, params: dict,
                          limit: int = 2000):
    """Fetch + prepare every symbol ONCE, aligned to a common timeline (intersection of bar times).
    Indicator columns depend only on indicator-LENGTH params, so frames can be reused across trials
    that vary only decision/risk params. Returns (frames: {sym->DataFrame}, common_index)."""
    limit = max(120, min(int(limit), 20000))
    frames = {}
    for sym in symbols:
        try:
            df = fetch_ohlcv(sym, timeframe, limit)
        except Exception:
            continue
        st = get_strategy(strategy, params)
        st.symbol = sym
        st.timeframe = timeframe
        frames[sym] = st.prepare(df)
    if not frames:
        raise ValueError("no candle history for the configured symbols")
    common = None
    for f in frames.values():
        common = f.index if common is None else common.intersection(f.index)
    common = common.sort_values()
    frames = {s: f.reindex(common) for s, f in frames.items()}
    return frames, common


def simulate_portfolio(frames: dict, common, strategy: str, params: dict, risk_cfg: dict,
                       timeframe: str, start_balance: float, taker_fee: float,
                       rates: dict | None = None, i0: int = 0, i1: int | None = None):
    """Replay all symbols on ONE shared account over common[i0:i1] — the live bot's loop. Builds
    fresh strategy objects from `params` (cheap; no re-prepare). Returns (summary, curve)."""
    i1 = len(common) if i1 is None else i1
    rates = rates or {}
    tf_hours = _TF_MIN.get(timeframe, 60) / 60.0
    strats = {}
    for sym in frames:
        st = get_strategy(strategy, params)
        st.symbol = sym
        st.timeframe = timeframe
        strats[sym] = st
    risk = RiskEngine(risk_cfg)
    sim = _Sim(start_balance, taker_fee, risk.r["maint_margin"])

    curve, syms = [], list(frames)
    peak = start_balance                      # running equity peak for the #2 equity-curve brake
    for i in range(max(i0, 0), i1):
        pricemap = {}
        for sym in syms:
            price = float(frames[sym]["close"].iat[i])
            if not math.isnan(price):
                pricemap[sym] = price
        eq_now = sim.equity(pricemap) if pricemap else sim.balance
        peak = max(peak, eq_now)
        for sym in list(pricemap):
            row = frames[sym].iloc[i]
            price = pricemap[sym]
            _apply_funding(sim, sym, price, rates.get(sym, 0.0), tf_hours)
            closed = _manage(sim, sym, row, price, params)
            try:
                direction, stop, meta = strats[sym].signal(row, price)
            except Exception:
                direction, stop, meta = "flat", None, {}
            if not (closed or direction == "flat" or stop is None):
                conf = float((meta or {}).get("conf", 1.0))
                _enter(sim, risk, sym, direction, stop, price, sim.equity(pricemap), conf, peak)
        if pricemap:
            curve.append((common[i].isoformat(), sim.equity(pricemap)))
    if sim.positions and i1 > i0:
        for sym in list(sim.positions):
            sim.close(sym, float(frames[sym]["close"].iat[i1 - 1]), "backtest-end")
    return _metrics(curve, sim.trades, start_balance, timeframe), curve


def run_portfolio_backtest(symbols: list, timeframe: str, strategy: str, params: dict,
                           risk_cfg: dict, start_balance: float, taker_fee: float,
                           limit: int = 2000, funding: bool = True) -> dict:
    """Closest thing to live: ALL symbols on ONE account with risk caps + funding — mirrors bot.py."""
    frames, common = load_portfolio_frames(symbols, timeframe, strategy, params, limit)
    rates = fetch_funding_rates(symbols) if funding else {}
    summary, curve = simulate_portfolio(frames, common, strategy, params, risk_cfg, timeframe,
                                        start_balance, taker_fee, rates)
    summary.update(symbol="+".join(symbols), timeframe=timeframe, strategy=strategy,
                   candles=len(common), start_balance=start_balance, funding_modeled=bool(funding),
                   period_start=curve[0][0] if curve else None,
                   period_end=curve[-1][0] if curve else None)
    step = max(1, len(curve) // 400)
    equity_curve = [{"ts": t, "equity": round(e, 2)} for t, e in curve[::step]]
    return {"summary": summary, "equity_curve": equity_curve, "n_closed": summary["n_trades"]}
