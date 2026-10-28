"""
Paper futures broker — pure accounting, no network, no keys. Simulates Kraken-perp-style
trading: taker fees, hourly funding accrual, leverage, and an approximate liquidation.
State is persisted to SQLite (WAL mode so the admin panel can read concurrently) so the bot
survives restarts and the ledger is inspectable.

This is PAPER ONLY. There is no code path that can touch a real exchange account.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperFutures:
    def __init__(self, db_path: str, start_balance: float, taker_fee: float, maint_margin: float = 0.005):
        self.db = sqlite3.connect(db_path, timeout=5.0)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.taker_fee = taker_fee
        self.maint = maint_margin
        self._init_schema()
        self.balance = self._get_float("balance", start_balance)
        self.positions: dict[str, dict] = self._load_positions()

    # ------------------------------------------------------------------ schema / state
    def _init_schema(self):
        c = self.db
        c.execute("CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS control(k TEXT PRIMARY KEY, v TEXT)")
        c.execute("""CREATE TABLE IF NOT EXISTS positions(
            symbol TEXT PRIMARY KEY, side TEXT, entry REAL, qty REAL, notional REAL,
            leverage REAL, stop REAL, extreme REAL, opened_ts TEXT, bars INTEGER)""")
        # migration: latest mark (current) price per open position
        try:
            c.execute("ALTER TABLE positions ADD COLUMN mark REAL")
        except sqlite3.OperationalError:
            pass
        c.execute("""CREATE TABLE IF NOT EXISTS trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, side TEXT, action TEXT,
            price REAL, qty REAL, notional REAL, fee REAL, pnl REAL, balance REAL, reason TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS equity(
            ts TEXT, balance REAL, equity REAL, open_positions INTEGER)""")
        c.commit()

    def _get_float(self, k: str, default: float) -> float:
        row = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO kv(k, v) VALUES(?, ?)", (k, default)); self.db.commit()
            return default
        return float(row[0])

    def _set_float(self, k: str, v: float):
        self.db.execute("INSERT INTO kv(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=?", (k, v, v))
        self.db.commit()

    def _load_positions(self) -> dict:
        out = {}
        for r in self.db.execute("SELECT symbol,side,entry,qty,notional,leverage,stop,extreme,opened_ts,bars FROM positions"):
            out[r[0]] = dict(side=r[1], entry=r[2], qty=r[3], notional=r[4], leverage=r[5],
                             stop=r[6], extreme=r[7], opened_ts=r[8], bars=r[9])
        return out

    def _persist_position(self, sym: str):
        p = self.positions.get(sym)
        if p is None:
            self.db.execute("DELETE FROM positions WHERE symbol=?", (sym,))
        else:
            self.db.execute("""INSERT INTO positions(symbol,side,entry,qty,notional,leverage,stop,extreme,opened_ts,bars)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol) DO UPDATE SET side=excluded.side,entry=excluded.entry,qty=excluded.qty,
                notional=excluded.notional,leverage=excluded.leverage,stop=excluded.stop,
                extreme=excluded.extreme,opened_ts=excluded.opened_ts,bars=excluded.bars""",
                (sym, p["side"], p["entry"], p["qty"], p["notional"], p["leverage"],
                 p["stop"], p["extreme"], p["opened_ts"], p["bars"]))
        self.db.commit()

    # ------------------------------------------------------------------ control (panel <-> bot)
    def get_control(self, k: str) -> str:
        row = self.db.execute("SELECT v FROM control WHERE k=?", (k,)).fetchone()
        return row[0] if row else ""

    def set_control(self, k: str, v: str):
        self.db.execute("INSERT INTO control(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=?", (k, v, v))
        self.db.commit()

    # ------------------------------------------------------------------ accounting
    @staticmethod
    def _dir(side: str) -> int:
        return 1 if side == "long" else -1

    def unrealized(self, sym: str, price: float) -> float:
        p = self.positions.get(sym)
        if not p:
            return 0.0
        return p["qty"] * (price - p["entry"]) * self._dir(p["side"])

    def equity(self, prices: dict[str, float]) -> float:
        eq = self.balance
        for sym, p in self.positions.items():
            if sym in prices:
                eq += self.unrealized(sym, prices[sym])
        return eq

    def _log_trade(self, sym, side, action, price, qty, notional, fee, pnl, reason):
        self.db.execute("""INSERT INTO trades(ts,symbol,side,action,price,qty,notional,fee,pnl,balance,reason)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (_now(), sym, side, action, price, qty, notional, fee, pnl, self.balance, reason))
        self.db.commit()

    # ------------------------------------------------------------------ ops
    def open(self, sym: str, side: str, price: float, notional: float, leverage: float,
             stop: float, reason: str):
        if sym in self.positions or notional <= 0 or price <= 0:
            return
        qty = notional / price
        fee = notional * self.taker_fee
        self.balance -= fee
        self.positions[sym] = dict(side=side, entry=price, qty=qty, notional=notional,
                                   leverage=leverage, stop=stop, extreme=price,
                                   opened_ts=_now(), bars=0)
        self._set_float("balance", self.balance)
        self._persist_position(sym)
        self.db.execute("UPDATE positions SET mark=? WHERE symbol=?", (price, sym))
        self.db.commit()
        self._log_trade(sym, side, "OPEN", price, qty, notional, fee, 0.0, reason)

    def close(self, sym: str, price: float, reason: str) -> float:
        p = self.positions.get(sym)
        if not p:
            return 0.0
        pnl = p["qty"] * (price - p["entry"]) * self._dir(p["side"])
        fee = p["qty"] * price * self.taker_fee
        self.balance += pnl - fee
        self._set_float("balance", self.balance)
        del self.positions[sym]
        self._persist_position(sym)
        self._log_trade(sym, p["side"], "CLOSE", price, p["qty"], p["qty"] * price, fee, pnl - fee, reason)
        return pnl - fee

    def apply_funding(self, sym: str, rate: float, price: float):
        """Hourly funding. Longs pay when rate>0; shorts receive. Charged on current notional."""
        p = self.positions.get(sym)
        if not p or not rate:
            return
        notional_now = p["qty"] * price
        self.balance += (-self._dir(p["side"])) * rate * notional_now
        self._set_float("balance", self.balance)

    def liquidated(self, sym: str, price: float) -> bool:
        """Approximate: position is liquidated when the loss eats the posted margin (minus maint)."""
        p = self.positions.get(sym)
        if not p:
            return False
        margin = p["notional"] / p["leverage"]
        if self.unrealized(sym, price) <= -(margin - self.maint * p["notional"]):
            self.close(sym, price, "LIQUIDATION")
            return True
        return False

    def reset(self, start_balance: float):
        """Wipe the paper account back to a clean slate (panel 'reset' command)."""
        self.positions = {}
        for t in ("positions", "trades", "equity"):
            self.db.execute(f"DELETE FROM {t}")
        self.balance = start_balance
        self._set_float("balance", start_balance)
        self.db.commit()
        self._log_trade("-", "-", "RESET", 0.0, 0.0, 0.0, 0.0, 0.0, f"reset to ${start_balance}")

    def day_open_equity(self) -> float | None:
        today = datetime.now(timezone.utc).date().isoformat()
        row = self.db.execute("SELECT equity FROM equity WHERE ts>=? ORDER BY rowid ASC LIMIT 1",
                              (today,)).fetchone()
        return float(row[0]) if row else None

    def peak_equity(self) -> float | None:
        row = self.db.execute("SELECT MAX(equity) FROM equity").fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def snapshot(self, prices: dict[str, float]):
        eq = self.equity(prices)
        for sym, px in prices.items():
            if sym in self.positions:
                self.db.execute("UPDATE positions SET mark=? WHERE symbol=?", (px, sym))
        self.db.execute("INSERT INTO equity(ts,balance,equity,open_positions) VALUES(?,?,?,?)",
                        (_now(), self.balance, eq, len(self.positions)))
        self.db.commit()
        return eq
