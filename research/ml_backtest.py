"""
Honest test: does an XGBoost model on price/TA features add any out-of-sample edge on
SOL/AVAX/LINK? This is the gate BEFORE building the daily-retrain + agent infrastructure.

Method (no leakage):
  * Features: multi-horizon returns, volatility, RSI, MACD hist, Bollinger %b, ATR%, ADX, ROC,
    volume z-score, EMA distance  (+ BTC cross-move).
  * Label: sign of forward return over H bars.
  * Fit XGBoost ONLY on the first 70% (in-sample); predict on the last 30% (out-of-sample).
  * Trade OOS: long if P(up)>0.55, short if <0.45 (optional), else flat; Kraken-Futures taker
    fee charged on every position change. Compare to buy & hold.
  * Sanity: also report a SHUFFLED-label model (should be ~0 edge; if the real model isn't clearly
    better than shuffled, the 'edge' is noise).
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

SYMBOLS = {"SOL": "SOLUSDT", "AVAX": "AVAXUSDT", "LINK": "LINKUSDT"}
TIMEFRAME = "4h"
MONTHS = 36
HORIZON = 6                 # predict direction over next 6 bars (~1 day on 4h)
LONG_THR, SHORT_THR = 0.55, 0.45
ALLOW_SHORT = True
FEE = 0.0005                # Kraken Futures taker per side
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


def features(df, btc_close=None):
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    for n in (1, 3, 6, 12, 24):
        f[f"ret{n}"] = c.pct_change(n)
    f["vol12"] = c.pct_change().rolling(12).std()
    f["vol24"] = c.pct_change().rolling(24).std()
    delta = c.diff()
    g = delta.clip(lower=0).rolling(14).mean(); l = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + g / (l + 1e-12))
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26; f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    mid = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
    f["bb_pct"] = (c - (mid - 2 * sd)) / (4 * sd + 1e-12)
    tr = pd.concat([(df["high"]-df["low"]), (df["high"]-c.shift()).abs(), (df["low"]-c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    up = df["high"].diff(); dn = -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.rolling(14).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(14).mean() / (atr + 1e-12)
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(14).mean() / (atr + 1e-12)
    f["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)).rolling(14).mean()
    f["roc10"] = c.pct_change(10) * 100
    f["vol_z"] = (df["volume"] - df["volume"].rolling(30).mean()) / (df["volume"].rolling(30).std(ddof=0) + 1e-12)
    f["ema50_dist"] = c / c.ewm(span=50, adjust=False).mean() - 1
    if btc_close is not None:
        b = btc_close.reindex(df.index).ffill()
        f["btc_ret6"] = b.pct_change(6)
        f["btc_ret24"] = b.pct_change(24)
    return f


def run(name, df, btc):
    f = features(df, btc)
    fwd = df["close"].shift(-HORIZON) / df["close"] - 1.0
    y = (fwd > 0).astype(int)
    data = f.join(y.rename("y")).dropna()
    X = data.drop(columns=["y"]); yv = data["y"].to_numpy()
    rets = df["close"].pct_change().reindex(data.index).fillna(0).to_numpy()  # per-bar realized return

    cut = int(len(data) * (1 - OOS_FRAC))
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr = yv[:cut]

    def trade(proba):
        pos = np.where(proba > LONG_THR, 1, np.where((proba < SHORT_THR) & ALLOW_SHORT, -1, 0))
        pos_prev = np.concatenate([[0], pos[:-1]])
        r_oos = rets[cut:]
        strat = pos_prev * r_oos - FEE * np.abs(pos - pos_prev)
        eq = float(np.prod(1 + strat) - 1)
        changes = int(np.sum(np.abs(pos - pos_prev) > 0))
        return eq, changes, pos

    mdl = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="logloss", n_jobs=2)
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    eq, ch, pos = trade(p)
    acc = float(((p > 0.5).astype(int) == yv[cut:]).mean())

    # shuffled-label control
    rng = np.random.default_rng(0)
    yshuf = ytr.copy(); rng.shuffle(yshuf)
    ms = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, eval_metric="logloss", n_jobs=2)
    ms.fit(Xtr, yshuf)
    eq_s, _, _ = trade(ms.predict_proba(Xte)[:, 1])

    bh = float(df["close"].reindex(data.index).iloc[-1] / df["close"].reindex(data.index).iloc[cut] - 1)
    imp = sorted(zip(X.columns, mdl.feature_importances_), key=lambda t: -t[1])[:5]
    print(f"\n{name}: OOS bars={len(Xte)} dir-acc={acc*100:.1f}% trades={ch}")
    print(f"   ML strategy net = {eq*100:7.1f}%   shuffled-ctrl = {eq_s*100:7.1f}%   buy&hold = {bh*100:7.1f}%")
    print(f"   top features: {', '.join(f'{k}={v:.2f}' for k,v in imp)}")
    return eq, eq_s, bh


def main():
    print(f"XGBoost ML edge test | {TIMEFRAME} | {MONTHS}mo | horizon={HORIZON} | fee={FEE*100:.3f}%/side | OOS={int(OOS_FRAC*100)}%")
    print("=" * 96)
    btc = fetch_binance("BTCUSDT", TIMEFRAME, MONTHS)["close"]
    mls, shs, bhs = [], [], []
    for name, sym in SYMBOLS.items():
        df = fetch_binance(sym, TIMEFRAME, MONTHS)
        if len(df) < 500:
            print(f"{name}: insufficient data"); continue
        eq, eq_s, bh = run(name, df, btc)
        mls.append(eq); shs.append(eq_s); bhs.append(bh)
    if mls:
        print("\n" + "=" * 96)
        print(f"PORTFOLIO OOS avg | ML = {np.mean(mls)*100:.1f}%   shuffled = {np.mean(shs)*100:.1f}%   buy&hold = {np.mean(bhs)*100:.1f}%")
        print("Verdict: ML is only real if it clearly beats BOTH the shuffled control AND buy&hold.")


if __name__ == "__main__":
    main()
