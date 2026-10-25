"""
Kraken Futures DATA RECON (read-only, public). Answers: can we even backtest a funding/OI-aware
strategy on Kraken perps? Probes symbols, funding-rate history depth + levels, OHLCV depth, and
open interest for the SOL/AVAX/LINK perpetuals. No API keys, no orders.
"""
import statistics as st
from datetime import datetime, timezone

import ccxt

WANT_BASES = {"SOL", "AVAX", "LINK"}


def ts(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat() if ms else "?"


def main():
    ex = ccxt.krakenfutures({"enableRateLimit": True})
    markets = ex.load_markets()

    perps = []
    for m in markets.values():
        if m.get("swap") and m.get("linear") and m.get("base") in WANT_BASES and m.get("quote") in ("USD", "USDT", "USDC"):
            perps.append(m)
    print(f"krakenfutures markets loaded: {len(markets)} | matching SOL/AVAX/LINK linear perps: {len(perps)}")
    for m in perps:
        print(f"  {m['base']:5s} symbol={m['symbol']:18s} id={m['id']:16s} quote={m['quote']} "
              f"contractSize={m.get('contractSize')} min_amt={m.get('limits',{}).get('amount',{}).get('min')}")

    for m in perps:
        sym = m["symbol"]
        print(f"\n===== {m['base']}  ({sym}) =====")
        # funding rate history
        try:
            fh = ex.fetch_funding_rate_history(sym, limit=1000)
            if fh:
                rates = [x["fundingRate"] for x in fh if x.get("fundingRate") is not None]
                per_yr = st.mean(rates) * 3 * 365 * 100 if rates else 0   # 8h funding -> 3/day
                print(f"  funding history: {len(fh)} pts  {ts(fh[0]['timestamp'])}→{ts(fh[-1]['timestamp'])}")
                if rates:
                    print(f"     per-interval rate  min={min(rates)*100:.4f}%  mean={st.mean(rates)*100:.4f}%  "
                          f"max={max(rates)*100:.4f}%   (~{per_yr:.1f}%/yr if held short, before fees)")
            else:
                print("  funding history: EMPTY")
        except Exception as e:
            print(f"  funding history: ERROR {str(e)[:80]}")
        # current funding
        try:
            fr = ex.fetch_funding_rate(sym)
            print(f"  current funding: {fr.get('fundingRate')}  interval={fr.get('interval')}")
        except Exception as e:
            print(f"  current funding: ERROR {str(e)[:60]}")
        # OHLCV depth
        for tf in ("1h", "4h", "1d"):
            try:
                o = ex.fetch_ohlcv(sym, tf, limit=5000)
                print(f"  OHLCV {tf}: {len(o)} bars  {ts(o[0][0]) if o else '?'}→{ts(o[-1][0]) if o else '?'}")
            except Exception as e:
                print(f"  OHLCV {tf}: ERROR {str(e)[:60]}")
        # open interest
        try:
            oi = ex.fetch_open_interest(sym)
            print(f"  open interest: {oi.get('openInterestAmount')} (value={oi.get('openInterestValue')})")
        except Exception as e:
            print(f"  open interest: not available ({str(e)[:50]})")


if __name__ == "__main__":
    main()
