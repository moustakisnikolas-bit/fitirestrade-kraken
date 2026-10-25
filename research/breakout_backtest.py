"""
Honest test of a VOLATILITY/TREND BREAKOUT system (turtle-style) on very volatile coins —
the strategy family actually designed to profit from big directional moves. Futures, long+short.

  * Entry: close breaks above the ENTRY_N-bar high (long) / below the ENTRY_N-bar low (short).
  * Exit:  opposite EXIT_N-bar channel, OR an ATR trailing stop (whichever first).
  * Costs: Kraken-Futures taker fee per side. Returns are UNLEVERED %; leverage would scale both
    the return and the drawdown (and add liquidation), so unlevered % is the honest edge measure.
  * Honesty controls: IN-SAMPLE (70%) vs OUT-OF-SAMPLE (30%); vs buy&hold; and vs a RANDOM-ENTRY
    system matched to the same trade frequency (avg of seeds). Breakout is only real if OOS it beats
    BOTH buy&hold AND the random control.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import requests

# small set of genuinely high-volatility, liquid perps (Binance USDT proxy for price action)
SYMBOLS = {"DOGE": "DOGEUSDT", "SOL": "SOLUSDT", "AVAX": "AVAXUSDT",
           "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "ADA": "ADAUSDT"}
TIMEFRAME = "4h"
MONTHS = 36
ENTRY_N = 20
EXIT_N = 10
ATR_STOP = 3.0
FEE = 0.0005
OOS_FRAC = 0.30


def fetch_binance(symbol, timeframe, months):
    end = int(time.time() * 1000); start = end - months * 30 * 24 * 3600 * 1000
    rows, cur = [], start
    while cur < end:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&startTime={cur}&limit=1000"
        for _ in range(6):
            r = requests.get(url, timeout=30)
            if r.status_code in (429, 418):
                time.sleep(float(r.headers.get("Retry-After", 2)) + 1); continue
            r.raise_for_status(); break
        kl = r.json()
        if not kl:
            break
        rows += kl; cur = kl[-1][0] + 1
        if len(kl) < 1000:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["t","o","h","l","c","v","a","b","d","e","f","g"])
    df = df[["t","o","h","l","c","v"]].astype(float)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}).drop(columns=["t"])


def atr(df, n=14):
    tr = pd.concat([(df["high"]-df["low"]),
                    (df["high"]-df["close"].shift()).abs(),
                    (df["low"]-df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def backtest(df, signals):
    """signals: array of +1 (go/stay long bias), -1 (short), 0 (no new entry) per bar.
    Event-driven: one position at a time, ATR trailing stop + opposite channel exit handled by caller
    via the 'exit_long'/'exit_short' boolean arrays embedded in df."""
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    a = df["atr"].to_numpy(float)
    exit_long = df["exit_long"].to_numpy(bool); exit_short = df["exit_short"].to_numpy(bool)
    pos = 0; entry = 0.0; stop = 0.0; extreme = 0.0
    eq = 1.0; peak = 1.0; mdd = 0.0; trades = []
    for i in range(len(df) - 1):
        price = close[i]
        if pos != 0:
            if pos > 0:
                extreme = max(extreme, high[i]); stop = max(stop, extreme - ATR_STOP * a[i])
                hit = low[i] <= stop or exit_long[i]
            else:
                extreme = min(extreme, low[i]); stop = min(stop, extreme + ATR_STOP * a[i])
                hit = high[i] >= stop or exit_short[i]
            if hit:
                exit_px = close[i + 1]
                r = (exit_px / entry - 1.0) * pos - 2 * FEE
                eq *= (1 + r); trades.append(r); pos = 0
        if pos == 0 and signals[i] != 0 and not np.isnan(a[i]):
            pos = int(signals[i]); entry = close[i + 1]; extreme = entry
            stop = entry - ATR_STOP * a[i] if pos > 0 else entry + ATR_STOP * a[i]
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    wins = [t for t in trades if t > 0]
    return {"trades": len(trades), "net": eq - 1,
            "win": (len(wins) / len(trades)) if trades else 0, "mdd": mdd}


def build(df):
    df = df.copy()
    df["atr"] = atr(df)
    hi = df["high"].rolling(ENTRY_N).max().shift(1); lo = df["low"].rolling(ENTRY_N).min().shift(1)
    df["long_sig"] = (df["close"] > hi)
    df["short_sig"] = (df["close"] < lo)
    df["exit_long"] = df["close"] < df["low"].rolling(EXIT_N).min().shift(1)
    df["exit_short"] = df["close"] > df["high"].rolling(EXIT_N).max().shift(1)
    sig = np.where(df["long_sig"], 1, np.where(df["short_sig"], -1, 0))
    return df, sig


def random_control(df, n_trades, seeds=6):
    nets = []
    base = df.copy()
    for s in range(seeds):
        rng = np.random.default_rng(s)
        idx = np.zeros(len(df), dtype=int)
        pick = rng.choice(len(df) - 2, size=min(n_trades, len(df) - 2), replace=False)
        idx[pick] = rng.choice([-1, 1], size=len(pick))
        nets.append(backtest(base, idx)["net"])
    return float(np.mean(nets))


def main():
    print(f"Breakout (turtle {ENTRY_N}/{EXIT_N}, ATRstop {ATR_STOP}) | {TIMEFRAME} {MONTHS}mo | "
          f"fee {FEE*100:.3f}%/side | long+short | unlevered %")
    print("=" * 100)
    print(f"{'coin':5s} {'OOS trades':>10s} {'OOS net':>9s} {'buy&hold':>9s} {'random':>9s} {'win':>5s} {'maxDD':>7s}  verdict")
    print("-" * 100)
    agg = {"net": [], "bh": [], "rnd": []}
    for name, sym in SYMBOLS.items():
        try:
            df = fetch_binance(sym, TIMEFRAME, MONTHS)
        except Exception as e:
            print(f"{name:5s} fetch fail {str(e)[:50]}"); continue
        if len(df) < 800:
            print(f"{name:5s} only {len(df)} bars"); continue
        df, sig = build(df)
        cut = int(len(df) * (1 - OOS_FRAC))
        oos = df.iloc[cut:].reset_index(drop=True); oos_sig = sig[cut:]
        m = backtest(oos, oos_sig)
        bh = float(df["close"].iloc[-1] / df["close"].iloc[cut] - 1)
        rnd = random_control(oos, m["trades"])
        v = "EDGE?" if (m["net"] > bh and m["net"] > rnd and m["net"] > 0) else "no edge"
        print(f"{name:5s} {m['trades']:10d} {m['net']*100:8.1f}% {bh*100:8.1f}% {rnd*100:8.1f}% "
              f"{m['win']*100:4.0f}% {m['mdd']*100:6.1f}%  {v}")
        agg["net"].append(m["net"]); agg["bh"].append(bh); agg["rnd"].append(rnd)
    if agg["net"]:
        print("=" * 100)
        print(f"PORTFOLIO OOS avg | breakout {np.mean(agg['net'])*100:.1f}%  "
              f"buy&hold {np.mean(agg['bh'])*100:.1f}%  random {np.mean(agg['rnd'])*100:.1f}%")
        print("Real edge only if breakout clearly beats BOTH buy&hold AND random-entry.")


if __name__ == "__main__":
    main()
