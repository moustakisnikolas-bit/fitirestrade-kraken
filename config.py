"""
Shared runtime config for the bot and the admin panel. Stored as data/config.json so the panel
can edit it and the bot hot-reloads it every loop (no container restart needed for most changes).
First run seeds it from KF_* environment defaults + the strategy's DEFAULT_PARAMS.
"""
from __future__ import annotations

import json
import os
from threading import Lock

from strategy import DEFAULT_PARAMS
from risk import DEFAULT_RISK

DATA_DIR = os.getenv("KF_DATA_DIR", "data")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
_lock = Lock()

# fields the panel is allowed to edit (everything except params, handled separately)
SCALAR_FIELDS = {
    "timeframe": str, "leverage": float, "start_balance": float, "taker_fee": float,
    "max_position_pct": float, "allow_short": bool, "poll_sec": int, "trading_enabled": bool,
    "autopilot_apply": bool,   # nightly auto-apply of optimizer results (strict-guarded)
}


def _env_defaults() -> dict:
    return {
        "symbols": [s.strip() for s in os.getenv("KF_SYMBOLS", "SOL/USD:USD,AVAX/USD:USD,LINK/USD:USD").split(",") if s.strip()],
        "timeframe": os.getenv("KF_TIMEFRAME", "1h"),
        "leverage": float(os.getenv("KF_LEVERAGE", "2")),
        "start_balance": float(os.getenv("KF_START_BALANCE", "200")),
        "taker_fee": float(os.getenv("KF_TAKER_FEE", "0.0005")),
        "max_position_pct": float(os.getenv("KF_MAX_POSITION_PCT", "0.5")),
        "allow_short": os.getenv("KF_ALLOW_SHORT", "true").lower() == "true",
        "poll_sec": int(os.getenv("KF_POLL_SEC", "60")),
        "trading_enabled": True,
        "autopilot_apply": os.getenv("KF_AUTOPILOT_APPLY", "false").lower() == "true",
        "strategy": os.getenv("KF_STRATEGY", "ensemble"),
        "params": dict(DEFAULT_PARAMS),
        "risk": dict(DEFAULT_RISK),
    }


def load_config() -> dict:
    with _lock:
        cfg = _env_defaults()
        if os.path.exists(CONFIG_PATH):
            try:
                saved = json.load(open(CONFIG_PATH))
                for k, v in saved.items():
                    if k == "params" and isinstance(v, dict):
                        p = dict(DEFAULT_PARAMS); p.update(v); cfg["params"] = p
                    elif k == "risk" and isinstance(v, dict):
                        rr = dict(DEFAULT_RISK); rr.update(v); cfg["risk"] = rr
                    elif k in cfg:
                        cfg[k] = v
            except Exception:
                pass
        return cfg


def save_config(cfg: dict) -> None:
    with _lock:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, CONFIG_PATH)


def ensure_config() -> dict:
    cfg = load_config()
    if not os.path.exists(CONFIG_PATH):
        save_config(cfg)
    return cfg
