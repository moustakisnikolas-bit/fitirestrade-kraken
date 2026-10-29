# Phase 6 — Proactive guards: don't bleed when there's no edge

## Why
The live paper bot has no *durable* edge (holdouts confirm it). Against a no-edge strategy the
fastest win is not a better forecast — it's **not trading when conditions are bad** and
**cutting risk the moment the equity curve rolls over**, instead of waiting for the hard stop.

These four guards are *loss-avoiders*, not alpha. They can only **block an entry** or **shrink its
size** — they never create a trade, flip a direction, or move a stop. So the worst they can do to a
*winning* configuration is leave a little on the table; the best they can do to a *losing* one is
stop the bleed. That asymmetry is the whole point.

All four are shared by the **live bot** and the **backtester** (same code paths), so a backtest
predicts exactly how they behave live. All degrade safely: a missing/NaN indicator or a flat equity
curve means "don't block / don't scale", never a crash.

## The four guards

| # | Guard | Lives in | Needs | What it does |
|---|-------|----------|-------|--------------|
| 1 | Regime filter | `guards.regime_entry_ok` | indicator row | Refuse NEW entries in bad-to-trade regimes |
| 2 | Equity-curve brake | `risk.RiskEngine.entry_scale` | equity, peak | Shrink then pause new entries as drawdown deepens |
| 3 | Vol-target sizing | `guards.vol_target_mult` | indicator row | Shrink size when volatility is above target |
| 4 | Correlation cap | `risk.RiskEngine.admit` | open positions | Cap concurrent **same-direction** positions |

### #1 Regime filter — `regime_entry_ok(row, price, p) -> (ok, reason)`
Blocks a new entry when the market is a bad place to trade *at all*:
- **ATR/price < `guard_atr_floor`** (default 0.3%) → dead market: whipsaws, no follow-through.
- **ATR/price > `guard_atr_ceiling`** (default 8%) → too wild: stop-outs dominate.
- **ADX < `guard_adx_min`** (default 18) → deep chop, **unless** price prints a fresh breakout
  (`close > breakout_high`), which is the one trade worth taking in a low-ADX tape.

Applied inside `Strategy.signal` *after* the raw decision, so it covers every strategy
(`ensemble`, `breakout`, `ml_meta`, `ml_dir`) identically. A blocked entry returns
`("flat", None, {"blocked": reason})`.

### #2 Equity-curve brake — `entry_scale(equity, peak) -> (size_mult, pause)`
Graded de-risk off drawdown-from-peak `dd = 1 - equity/peak`:
- `dd <= brake_soft_dd` (6%): full size (`1.0`).
- `brake_soft_dd < dd < brake_pause_dd`: linear taper from `1.0` down to `brake_floor` (0.3).
- `dd >= brake_pause_dd` (12%): **pause** — no new entries at all.

This fires **long before** the hard `max_drawdown_pct` halt (20%), so the bot starts protecting
capital while the drawdown is still small and recoverable. The multiplier folds into the risk
budget (`risk.size(..., conf * scale)`); existing positions are untouched (their own stops manage
them). `peak` is `broker.peak_equity()` live, and a running peak in the backtester.

### #3 Vol-target sizing — `vol_target_mult(row, price, p) -> mult`
`mult = clip(guard_atr_target / (ATR/price), guard_vol_floor, 1.0)`. When realised volatility is
above target (default 2%), size down proportionally; at/below target, leave size alone (**never
inflate** — capped at 1.0). Folds into `conf` so it composes with the model-confidence multiplier
and the equity brake.

### #4 Correlation cap — `admit(positions, equity, notional, direction)`
Our universe (SOL/AVAX/LINK …) is highly correlated — two longs are really one oversized bet that
all loses together. `max_same_direction` (default 1) caps concurrent **same-direction** positions.
Opposite-direction positions are still allowed (they hedge). Independent of the gross
`max_concurrent` / exposure caps.

## Config (tunable from the panel)
Regime + vol-target keys live in `params` (`strategy.DEFAULT_PARAMS`); brake + correlation keys live
in `risk` (`risk.DEFAULT_RISK`). Both dicts already persist in `config.json` and hot-reload each loop.

```
params.guard_regime       1.0     # #1 on/off          params.guard_vol_target  1.0   # #3 on/off
params.guard_adx_min      18.0    #    chop floor       params.guard_atr_target  0.02  #    target vol
params.guard_atr_floor    0.003   #    dead floor       params.guard_vol_floor   0.35  #    min size mult
params.guard_atr_ceiling  0.08    #    wild ceiling
risk.equity_brake         true    # #2 on/off           risk.max_same_direction  1     # #4 corr cap
risk.brake_soft_dd        0.06    #    start cutting
risk.brake_pause_dd       0.12    #    pause entries
risk.brake_floor          0.30    #    min size mult
```

Set any `guard_*`/`equity_brake` to `0`/`false` to disable that guard; everything else keeps working.

## Validation
Because guards only block/shrink, validate the way we validate everything else: a **portfolio
backtest on stored history, guards ON vs OFF**, over the same window. The guards are a win if they
**reduce max drawdown** without materially hurting return (and usually they lift profit factor by
cutting the worst fee-only chop trades). "No change" is possible if the window had no bad regimes —
that's fine. Tune `guard_adx_min` / `brake_*` from that A/B, never by gut.

## Safe-degradation checklist
- Warmup row (NaN ATR/ADX) → `regime_entry_ok` returns allowed, `vol_target_mult` returns 1.0.
- No peak yet / flat curve → `entry_scale` returns `(1.0, False)`.
- Guard disabled via config → that guard is a no-op; the others still run.
- Backtester and live bot call the identical functions, so they can never diverge.
