"""
Kraken Futures PAPER bot — standalone, isolated, no API keys, cannot place real orders.

Pipeline per loop: reload config (hot) -> poll public Kraken-Futures OHLCV -> honor panel commands
(pause/flatten/reset) -> manage open positions (trailing/stop/time-stop) -> check RISK circuit
breakers -> on each newly CLOSED bar ask the selected STRATEGY for a direction+stop -> RISK ENGINE
sizes it (liquidation-safe leverage) -> PaperFutures executes + accrues hourly funding.
Strategies only propose direction+stop; the risk engine owns sizing. All simulated in SQLite.
"""
from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone

import ccxt
import pandas as pd

from paper_engine import PaperFutures
from risk import RiskEngine
from strategies import get_strategy
from config import ensure_config, load_config, CONFIG_PATH


def log(*a):
    print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["t", "open", "high", "low", "close", "volume"]).astype(float)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.drop(columns=["t"])


def manage_open(broker, sym, row, price, p) -> bool:
    pos = broker.positions.get(sym)
    if not pos:
        return False
    pos["bars"] += 1
    is_long = pos["side"] == "long"
    pos["extreme"] = max(pos["extreme"], row["high"]) if is_long else min(pos["extreme"], row["low"])
    if is_long and pos["extreme"] >= pos["entry"] * (1 + p["profit_lock"]):
        pos["stop"] = max(pos["stop"], pos["extreme"] * (1 - p["trail"]))
    elif (not is_long) and pos["extreme"] <= pos["entry"] * (1 - p["profit_lock"]):
        pos["stop"] = min(pos["stop"], pos["extreme"] * (1 + p["trail"]))
    broker._persist_position(sym)
    if broker.liquidated(sym, price):
        log(f"  {sym} LIQUIDATED @ {price:.4f}"); return True
    hit = (price <= pos["stop"]) if is_long else (price >= pos["stop"])
    if hit:
        log(f"  {sym} CLOSE {pos['side']} @ {price:.4f} stop -> pnl {broker.close(sym, price, 'stop/trail'):+.2f}")
        return True
    if pos["bars"] >= p["max_hold_bars"]:
        log(f"  {sym} CLOSE {pos['side']} @ {price:.4f} time -> pnl {broker.close(sym, price, 'time-stop'):+.2f}")
        return True
    return False


def try_enter(broker, risk, sym, direction, stop, price, equity, meta, peak=None):
    """Strategy proposed direction+stop; risk engine sizes + gates; broker opens."""
    pos = broker.positions.get(sym)
    if pos and pos["side"] != direction:
        broker.close(sym, price, f"flip->{direction}")
        pos = None
    if pos:
        return
    scale, pause = risk.entry_scale(equity, peak)           # #2 equity-curve brake
    if pause:
        log(f"  {sym} entry paused: equity brake (drawdown)"); return
    conf = float((meta or {}).get("conf", 1.0)) if isinstance(meta, dict) else 1.0
    notional, lev, why = risk.size(equity, price, stop, conf * scale)
    if notional <= 0:
        log(f"  {sym} entry rejected: {why}"); return
    notional, reason = risk.admit(broker.positions, equity, notional, direction)  # exposure + corr cap
    if notional <= 0:
        log(f"  {sym} entry blocked: {reason}"); return
    broker.open(sym, direction, price, notional, round(lev, 2), stop, f"{meta} lev={lev:.1f}")
    log(f"  {sym} OPEN {direction} @ {price:.4f} notional {notional:.2f} lev {lev:.1f} stop {stop:.4f} | {meta}")


def main():
    cfg = ensure_config()
    log("=" * 74)
    log(f"Kraken Futures PAPER platform | symbols={cfg['symbols']} tf={cfg['timeframe']} "
        f"strategy={cfg['strategy']} start=${cfg['start_balance']}")
    log(f"RISK: risk/trade={cfg['risk']['risk_pct_per_trade']*100:.1f}% maxLev={cfg['risk']['max_leverage']} "
        f"maxConc={cfg['risk']['max_concurrent']} dayLoss={cfg['risk']['daily_loss_limit_pct']*100:.0f}% "
        f"maxDD={cfg['risk']['max_drawdown_pct']*100:.0f}%")
    log("PAPER ONLY — no API keys, no real orders possible. Config:", CONFIG_PATH)
    ex = ccxt.krakenfutures({"enableRateLimit": True})
    ex.load_markets()
    broker = PaperFutures("data/paper.db", cfg["start_balance"], cfg["taker_fee"])
    last_bar: dict[str, pd.Timestamp] = {}
    last_funding_hour: dict[str, int] = {}

    while True:
        try:
            cfg = load_config()
            broker.taker_fee = cfg["taker_fee"]
            p = cfg["params"]
            risk = RiskEngine(cfg["risk"])
            strat = get_strategy(cfg["strategy"], p)
            strat.timeframe = cfg["timeframe"]

            try:                                       # keep live sentiment fresh (TTL-throttled)
                import sentiment as _sent
                _sent.topup(cfg["symbols"], cfg["timeframe"], ex)
            except Exception:
                pass

            prices, rows = {}, {}
            for sym in cfg["symbols"]:
                if sym not in ex.markets:
                    continue
                raw = ex.fetch_ohlcv(sym, cfg["timeframe"], limit=300)
                if not raw or len(raw) < 60:
                    continue
                strat.symbol = sym                      # set BEFORE prepare (sentiment needs it)
                ind = strat.prepare(to_df(raw).iloc[:-1])
                rows[sym] = ind.iloc[-1]
                prices[sym] = float(ind.iloc[-1]["close"])

            # panel one-shot commands
            cmd = broker.get_control("cmd")
            if cmd == "flatten":
                for s in list(broker.positions):
                    if s in prices:
                        log(f"  {s} FLATTEN (panel) -> pnl {broker.close(s, prices[s], 'panel-flatten'):+.2f}")
                broker.set_control("cmd", "")
            elif cmd == "reset":
                broker.reset(cfg["start_balance"]); last_bar.clear()
                log("  RESET (panel): paper account wiped"); broker.set_control("cmd", "")

            equity = broker.equity(prices) if prices else broker.balance
            halted, hreason = risk.halted(broker.day_open_equity(), broker.peak_equity(), equity)
            broker.set_control("halted", "1" if halted else "0")
            broker.set_control("halt_reason", hreason)

            for sym, row in rows.items():
                price = prices[sym]
                try:
                    hr = datetime.now(timezone.utc).hour
                    if last_funding_hour.get(sym) != hr and sym in broker.positions:
                        broker.apply_funding(sym, float(ex.fetch_funding_rate(sym).get("fundingRate") or 0.0), price)
                        last_funding_hour[sym] = hr
                except Exception:
                    pass

                closed = manage_open(broker, sym, row, price, p)
                bar_t = row.name
                if last_bar.get(sym) != bar_t:
                    last_bar[sym] = bar_t
                    strat.symbol = sym
                    direction, stop, meta = strat.signal(row, price)
                    block = (not cfg["trading_enabled"]) or halted or closed or direction == "flat" or stop is None
                    if not block:
                        peak = broker.recent_peak(risk.r.get("brake_peak_hours", 120.0))
                        try_enter(broker, risk, sym, direction, stop, price, broker.equity(prices),
                                  meta, peak)
                    tag = "" if cfg["trading_enabled"] else " (PAUSED)"
                    tag += f" (HALTED:{hreason})" if halted else ""
                    log(f"  {sym} bar {bar_t:%m-%d %H:%M} px {price:.4f} -> {direction}{tag} {meta}")

            if prices:
                eq = broker.snapshot(prices)
                opens = ",".join(f"{s}:{v['side']}@{v['leverage']}x" for s, v in broker.positions.items()) or "flat"
                gross = sum(v["notional"] for v in broker.positions.values())
                log(f"equity ${eq:.2f} bal ${broker.balance:.2f} gross ${gross:.0f} "
                    f"({gross/max(eq,1e-9):.2f}x) | {opens}{' | HALTED '+hreason if halted else ''}"
                    f"{'' if cfg['trading_enabled'] else ' | PAUSED'}")
        except Exception as e:
            log("ERROR", str(e)); traceback.print_exc()
        time.sleep(load_config().get("poll_sec", 60))


if __name__ == "__main__":
    main()
