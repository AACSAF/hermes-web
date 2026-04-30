"""MT5 data fetching — communicates with mt5_daemon.py on Windows."""

import json
import subprocess
import threading
import time
import logging
from typing import Optional

from . import database as db

logger = logging.getLogger(__name__)

MT5_DAEMON_SCRIPT = r"C:\Users\Administrator\mt5_daemon.py"


class MT5Daemon:
    """Persistent MT5 daemon process."""

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()
        self._ready = False

    def start(self) -> dict:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return {"ok": True, "already_running": True}
            try:
                self._proc = subprocess.Popen(
                    ["powershell.exe", "-Command", f'python "{MT5_DAEMON_SCRIPT}"'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                line = self._proc.stdout.readline().decode("utf-8", errors="replace").strip()
                resp = json.loads(line)
                if resp.get("ready"):
                    self._ready = True
                    logger.info("MT5 daemon started, pid=%s", self._proc.pid)
                    return {"ok": True}
                return {"ok": False, "error": f"Not ready: {resp}"}
            except Exception as e:
                return {"ok": False, "error": str(e)}

    def call(self, action: str, **kwargs) -> dict:
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                self._ready = False
                return {"ok": False, "error": "Daemon not running"}
            try:
                cmd = {"action": action, **kwargs}
                self._proc.stdin.write((json.dumps(cmd) + "\n").encode())
                self._proc.stdin.flush()
                line = self._proc.stdout.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    return {"ok": False, "error": "Empty response"}
                return json.loads(line)
            except Exception as e:
                logger.exception("MT5 daemon call failed")
                self._ready = False
                return {"ok": False, "error": str(e)}

    def is_ready(self) -> bool:
        return self._ready and self._proc is not None and self._proc.poll() is None

    def shutdown(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(b'{"action":"shutdown"}\n')
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
                self._ready = False


# Singleton
daemon = MT5Daemon()


def check_mt5_running() -> bool:
    """Check if terminal64.exe is running on Windows."""
    try:
        result = subprocess.run(
            ["powershell.exe", "-Command",
             "Get-Process -Name 'terminal64' -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count"],
            capture_output=True, timeout=10,
        )
        output = result.stdout.decode("utf-8", errors="replace").strip()
        return int(output) > 0 if output.isdigit() else False
    except Exception:
        return False


def ensure_daemon() -> bool:
    """Ensure daemon is running. Returns True if ready."""
    if daemon.is_ready():
        return True
    if not check_mt5_running():
        return False
    result = daemon.start()
    return result.get("ok", False)


# ── Candle fetching ────────────────────────────────────────────────

# Timeframe mapping for MT5 copy_rates
TF_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408, "W1": 32769, "MN1": 49153,
}


def fetch_candles(symbol: str, timeframe: str, count: int = 200) -> list[dict]:
    """Fetch candles from MT5 daemon and store in DB.

    The daemon must implement a 'candles' action that calls mt5.copy_rates_from_pos().
    """
    if not ensure_daemon():
        logger.error("MT5 daemon not available")
        return []

    # Ask daemon for candles
    result = daemon.call("candles", symbol=symbol, timeframe=timeframe, count=count)
    if not result.get("ok"):
        logger.error("Failed to fetch candles: %s", result.get("error"))
        return []

    candles = result.get("candles", [])
    if not candles:
        return []

    # Store in DB
    conn = db.get_conn()
    for c in candles:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO candles (symbol, timeframe, time, open, high, low, close, volume, spread)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (symbol, timeframe, c["time"], c["open"], c["high"], c["low"], c["close"],
                 c.get("tick_volume", c.get("volume", 0)), c.get("spread", 0))
            )
        except Exception as e:
            logger.warning("Failed to insert candle: %s", e)
    conn.commit()

    return candles


def get_candles(symbol: str, timeframe: str, limit: int = 200) -> list[dict]:
    """Get candles from DB (or fetch if empty)."""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT * FROM candles WHERE symbol=? AND timeframe=? ORDER BY time DESC LIMIT ?",
        (symbol, timeframe, limit)
    ).fetchall()

    if rows:
        return [dict(r) for r in reversed(rows)]

    # Fetch from MT5
    fetch_candles(symbol, timeframe, count=limit)
    rows = conn.execute(
        "SELECT * FROM candles WHERE symbol=? AND timeframe=? ORDER BY time DESC LIMIT ?",
        (symbol, timeframe, limit)
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_closes(symbol: str, timeframe: str, limit: int = 200) -> list[float]:
    """Get close prices as a flat list (newest last)."""
    candles = get_candles(symbol, timeframe, limit)
    return [c["close"] for c in candles]


def get_ohlcv(symbol: str, timeframe: str, limit: int = 200) -> dict:
    """Get OHLCV as dict of lists."""
    candles = get_candles(symbol, timeframe, limit)
    return {
        "time": [c["time"] for c in candles],
        "open": [c["open"] for c in candles],
        "high": [c["high"] for c in candles],
        "low": [c["low"] for c in candles],
        "close": [c["close"] for c in candles],
        "volume": [c["volume"] for c in candles],
    }


# ── Order execution ────────────────────────────────────────────────

def place_order(symbol: str, direction: str, lot_size: float,
                sl: float = 0, tp: float = 0, comment: str = "") -> dict:
    """Place a market order via MT5 daemon.

    Args:
        symbol: e.g. "XAUUSD"
        direction: "BUY" or "SELL"
        lot_size: e.g. 0.01
        sl: stop loss price (0 = none)
        tp: take profit price (0 = none)
        comment: order comment

    Returns:
        {"ok": True, "ticket": ..., ...} or {"ok": False, "error": ...}
    """
    if not ensure_daemon():
        return {"ok": False, "error": "MT5 daemon not available"}

    result = daemon.call("order_send",
                         symbol=symbol,
                         direction=direction,
                         lot_size=lot_size,
                         sl=sl, tp=tp,
                         comment=comment)
    return result


def close_position(ticket: int) -> dict:
    """Close a position by ticket."""
    if not ensure_daemon():
        return {"ok": False, "error": "MT5 daemon not available"}
    return daemon.call("close_position", ticket=ticket)


def modify_position(ticket: int, sl: float = None, tp: float = None) -> dict:
    """Modify SL/TP of an existing position."""
    if not ensure_daemon():
        return {"ok": False, "error": "MT5 daemon not available"}
    kwargs = {"ticket": ticket}
    if sl is not None:
        kwargs["sl"] = sl
    if tp is not None:
        kwargs["tp"] = tp
    return daemon.call("modify_position", **kwargs)
