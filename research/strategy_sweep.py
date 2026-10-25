"""
Strategy search engine. Backtests many strategy families x parameter grids x timeframes across a
basket of volatile perps, ranks by OUT-OF-SAMPLE P&L (never in-sample), and prints the top 5 with
buy&hold + random-entry baselines so we can see whether a winner is REAL or just least-bad.

Futures, long+short, ATR trailing stop, Kraken-Futures taker fee per side. Unlevered % returns.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import requests

COINS = {"DOGE": "DOGEUSDT", "SOL": "SOLUSDT", "AVAX": "AVAXUSDT",
         "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "ADA": "ADAUSDT"}
TF_MONTHS = {"1h": 12, "4h": 36, "1d": 72}
FEE = 0.0005
STOP_ATR = 3.0
OOS_FRAC = 0.30


def fetch(symbol, tf, months):
    end = int(time.time() * 1000); start = end - months * 30 * 24 * 3600 * 1000
    rows, cur = [], start
    while cur < end:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&startTime={cur}&limit=1000"
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
    df = pd.DataFrame(rows, columns=["t","o","h","l","c","v","a","b","d","e","f","g"])[["t","o","h","l","c","v"]].astype(float)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}).drop(columns=["t"])


def indicators(df):
    d = df.copy(); c = d["close"]
    tr = pd.concat([(d["high"]-d["low"]), (d["high"]-c.shift()).abs(), (d["low"]-c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    dl = c.diff(); g = dl.clip(lower=0).rolling(14).mean(); l = (-dl.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100/(1 + g/(l+1e-12))
    e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
    d["macd"] = e12 - e26; d["macd_sig"] = d["macd"].ewm(span=9,adjust=False).mean()
    return d


def signals(d, fam, p):
    c = d["close"]; z = np.zeros(len(d))
    if fam == "donchian":
        hi = d["high"].rolling(p["N"]).max().shift(1); lo = d["low"].rolling(p["N"]).min().shift(1)
        return np.where(c > hi, 1, np.where(c < lo, -1, 0))
    if fam == "ema_cross":
        ef = c.ewm(span=p["f"],adjust=False).mean(); es = c.ewm(span=p["s"],adjust=False).mean()
        return np.where(ef > es, 1, -1)
    if fam == "rsi_mr":
        return np.where(d["rsi"] < p["os"], 1, np.where(d["rsi"] > p["ob"], -1, 0))
    if fam == "boll_mr":
        mid = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
        return np.where(c < mid - p["k"]*sd, 1, np.where(c > mid + p["k"]*sd, -1, 0))
    if fam == "roc_mom":
        r = c.pct_change(p["n"]) * 100
        return np.where(r > p["thr"], 1, np.where(r < -p["thr"], -1, 0))
    if fam == "macd":
        return np.where(d["macd"] > d["macd_sig"], 1, -1)
    if fam == "volbrk":
        a = d["atr"]
        return np.where(c > c.shift() + p["k"]*a, 1, np.where(c < c.shift() - p["k"]*a, -1, 0))
    return z


def bt(d, sig):
    c = d["close"].to_numpy(float); h = d["high"].to_numpy(float); lo = d["low"].to_numpy(float); a = d["atr"].to_numpy(float)
    pos = 0; entry = stop = extreme = 0.0; eq = 1.0; peak = 1.0; mdd = 0.0; trades = []
    for i in range(len(c) - 1):
        if pos != 0:
            if pos > 0:
                extreme = max(extreme, h[i]); stop = max(stop, extreme - STOP_ATR*a[i]); hit = lo[i] <= stop
            else:
                extreme = min(extreme, lo[i]); stop = min(stop, extreme + STOP_ATR*a[i]); hit = h[i] >= stop
            if hit or sig[i] == -pos:
                r = (c[i+1]/entry - 1)*pos - 2*FEE; eq *= 1 + r; trades.append(r); pos = 0
        if pos == 0 and sig[i] != 0 and not np.isnan(a[i]):
            pos = int(sig[i]); entry = c[i+1]; extreme = entry
            stop = entry - STOP_ATR*a[i] if pos > 0 else entry + STOP_ATR*a[i]
        peak = max(peak, eq); mdd = min(mdd, eq/peak - 1)
    wins = [t for t in trades if t > 0]
    return {"net": eq-1, "trades": len(trades), "win": (len(wins)/len(trades) if trades else 0), "mdd": mdd}


def random_net(d, n_trades, seeds=5):
    nets = []
    for s in range(seeds):
        rng = np.random.default_rng(s); sig = np.zeros(len(d), dtype=int)
        if len(d) > 3 and n_trades > 0:
            pick = rng.choice(len(d)-2, size=min(n_trades, len(d)-2), replace=False)
            sig[pick] = rng.choice([-1, 1], size=len(pick))
        nets.append(bt(d, sig)["net"])
    return float(np.mean(nets))


GRID = []
for N in (10, 20, 30, 55): GRID.append((f"donchian{N}", "donchian", {"N": N}))
for f, s in ((10,30),(20,50),(50,200),(12,26)): GRID.append((f"ema{f}/{s}", "ema_cross", {"f": f, "s": s}))
for os_, ob in ((30,70),(25,75),(20,80)): GRID.append((f"rsi{os_}/{ob}", "rsi_mr", {"os": os_, "ob": ob}))
for k in (2.0, 2.5): GRID.append((f"boll{k}", "boll_mr", {"k": k}))
for n, thr in ((6,0),(6,2),(12,0),(12,2)): GRID.append((f"roc{n}>{thr}", "roc_mom", {"n": n, "thr": thr}))
GRID.append(("macd", "macd", {}))
for k in (1.0, 1.5): GRID.append((f"volbrk{k}", "volbrk", {"k": k}))


def main():
    print(f"Strategy sweep | coins={list(COINS)} | fee={FEE*100:.3f}%/side | stopATR={STOP_ATR} | OOS={int(OOS_FRAC*100)}%")
    print(f"{len(GRID)} configs x {len(TF_MONTHS)} timeframes x {len(COINS)} coins = {len(GRID)*len(TF_MONTHS)*len(COINS)} backtests\n")
    data = {}
    for tf, months in TF_MONTHS.items():
        for name, sym in COINS.items():
            try:
                df = fetch(sym, tf, months)
                if len(df) >= 400:
                    data[(tf, name)] = indicators(df)
            except Exception as e:
                print(f"fetch fail {name} {tf}: {str(e)[:50]}")

    base = {}  # (tf) -> (bh, rnd) portfolio avg over coins
    for tf in TF_MONTHS:
        bhs, rnds = [], []
        for name in COINS:
            if (tf, name) not in data:
                continue
            d = data[(tf, name)]; cut = int(len(d)*(1-OOS_FRAC)); oos = d.iloc[cut:]
            bhs.append(float(oos["close"].iloc[-1]/oos["close"].iloc[0]-1))
            rnds.append(random_net(oos.reset_index(drop=True), 40))
        base[tf] = (np.mean(bhs) if bhs else 0, np.mean(rnds) if rnds else 0)

    results = []
    for name, fam, p in GRID:
        for tf in TF_MONTHS:
            nets, trs, wins, mdds = [], [], [], []
            for coin in COINS:
                if (tf, coin) not in data:
                    continue
                d = data[(tf, coin)]; cut = int(len(d)*(1-OOS_FRAC))
                sig = signals(d, fam, p)
                m = bt(d.iloc[cut:].reset_index(drop=True), sig[cut:])
                nets.append(m["net"]); trs.append(m["trades"]); wins.append(m["win"]); mdds.append(m["mdd"])
            if nets:
                results.append({"name": name, "tf": tf, "net": float(np.mean(nets)),
                                "trades": int(np.mean(trs)), "win": float(np.mean(wins)),
                                "mdd": float(np.mean(mdds))})

    results.sort(key=lambda r: -r["net"])
    print("=" * 104)
    print("TOP 10 BY OUT-OF-SAMPLE P&L (portfolio avg across coins)")
    print(f"{'#':>2} {'strategy':14s} {'tf':3s} {'OOS P&L':>9s} {'vs B&H':>8s} {'vs rand':>8s} {'trades':>7s} {'win':>5s} {'maxDD':>7s}  real?")
    print("-" * 104)
    for i, r in enumerate(results[:10], 1):
        bh, rnd = base[r["tf"]]
        real = "YES" if (r["net"] > bh and r["net"] > rnd and r["net"] > 0) else "no"
        print(f"{i:>2} {r['name']:14s} {r['tf']:3s} {r['net']*100:8.1f}% {(r['net']-bh)*100:7.1f}% "
              f"{(r['net']-rnd)*100:7.1f}% {r['trades']:7d} {r['win']*100:4.0f}% {r['mdd']*100:6.1f}%  {real}")
    print("-" * 104)
    for tf in TF_MONTHS:
        print(f"baseline {tf}: buy&hold {base[tf][0]*100:7.1f}%   random {base[tf][1]*100:7.1f}%")
    print("\nTOP 5 (ranked by OOS P&L):")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. {r['name']} @ {r['tf']}  ->  OOS P&L {r['net']*100:.1f}%")


if __name__ == "__main__":
    main()
