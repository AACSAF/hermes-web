"""Database layer for the trading system. SQLite with WAL mode."""

import sqlite3
import json
import time
import threading
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "trading.db"

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    """Thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), timeout=10)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


@contextmanager
def tx():
    """Transaction context manager."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Create all tables if they don't exist."""
    conn = get_conn()
    conn.executescript("""
    -- K-line data (OHLCV)
    CREATE TABLE IF NOT EXISTS candles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,       -- M1, M5, M15, H1, H4, D1
        time INTEGER NOT NULL,         -- unix timestamp (seconds)
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL DEFAULT 0,
        spread REAL DEFAULT 0,
        UNIQUE(symbol, timeframe, time)
    );
    CREATE INDEX IF NOT EXISTS idx_candles_sym_tf_time ON candles(symbol, timeframe, time);

    -- Trade signals (AI-generated, before execution)
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,       -- BUY / SELL / FLAT
        confidence REAL NOT NULL,      -- 0-100
        entry_price REAL,
        stop_loss REAL,
        take_profit REAL,
        lot_size REAL,
        reason TEXT,                   -- AI reasoning
        indicators TEXT,               -- JSON: {rsi: 32, macd: "bullish", ...}
        risk_pct REAL,                 -- % of account at risk
        mode TEXT DEFAULT 'observe',   -- observe / semi_auto / auto
        status TEXT DEFAULT 'pending', -- pending / approved / rejected / executed / expired
        created_at REAL NOT NULL,
        expires_at REAL,
        executed_at REAL,
        deal_ticket INTEGER,           -- MT5 deal ticket if executed
        result TEXT,                   -- win / loss / breakeven / cancelled
        pnl REAL,
        notes TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
    CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol, created_at);

    -- Trade journal (completed trades with full context)
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER REFERENCES signals(id),
        ticket INTEGER,               -- MT5 position ticket
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,       -- BUY / SELL
        lot_size REAL NOT NULL,
        entry_price REAL NOT NULL,
        entry_time REAL NOT NULL,
        exit_price REAL,
        exit_time REAL,
        stop_loss REAL,
        take_profit REAL,
        swap REAL DEFAULT 0,
        commission REAL DEFAULT 0,
        profit REAL DEFAULT 0,
        max_favorable REAL DEFAULT 0,  -- max favorable excursion (pips)
        max_adverse REAL DEFAULT 0,    -- max adverse excursion (pips)
        hold_duration REAL,            -- seconds held
        exit_reason TEXT,              -- tp_hit / sl_hit / manual / signal / trailing
        context TEXT,                  -- JSON: full market context at entry
        lesson TEXT                    -- AI-generated post-trade lesson
    );
    CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol, entry_time);
    CREATE INDEX IF NOT EXISTS idx_trades_result ON trades(profit);

    -- Strategy parameters (evolving)
    CREATE TABLE IF NOT EXISTS strategy_params (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,     -- e.g. "rsi_oversold", "sl_atr_mult"
        value TEXT NOT NULL,           -- JSON value
        value_type TEXT NOT NULL,      -- float / int / bool / str
        description TEXT,
        min_val REAL,
        max_val REAL,
        step REAL,                     -- for evolution
        updated_at REAL NOT NULL,
        generation INTEGER DEFAULT 0   -- evolution generation counter
    );

    -- Evolution history (parameter changes and their performance)
    CREATE TABLE IF NOT EXISTS evolution_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        generation INTEGER NOT NULL,
        params_snapshot TEXT NOT NULL,  -- JSON: all params at this point
        trades_count INTEGER,
        win_rate REAL,
        avg_profit REAL,
        max_drawdown REAL,
        profit_factor REAL,
        sharpe_ratio REAL,
        notes TEXT,
        created_at REAL NOT NULL
    );

    -- Rules library (learned from experience)
    CREATE TABLE IF NOT EXISTS rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL,        -- entry / exit / risk / market / time
        rule TEXT NOT NULL,
        source TEXT,                   -- learned / manual / review
        confidence REAL DEFAULT 50,    -- how well validated (0-100)
        times_applied INTEGER DEFAULT 0,
        times_correct INTEGER DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );

    -- Daily review reports
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL UNIQUE,     -- YYYY-MM-DD
        trades_count INTEGER,
        wins INTEGER,
        losses INTEGER,
        total_pnl REAL,
        max_drawdown REAL,
        best_trade TEXT,               -- JSON
        worst_trade TEXT,              -- JSON
        summary TEXT,                  -- AI-generated summary
        action_items TEXT,             -- JSON list of things to improve
        created_at REAL NOT NULL
    );

    -- System state (key-value for scheduler etc.)
    CREATE TABLE IF NOT EXISTS kv_store (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    """)
    conn.commit()

    # Initialize default strategy parameters if empty
    count = conn.execute("SELECT COUNT(*) FROM strategy_params").fetchone()[0]
    if count == 0:
        _init_default_params(conn)
    conn.commit()


def _init_default_params(conn):
    """Seed default strategy parameters."""
    now = time.time()
    defaults = [
        ("rsi_period", "14", "int", "RSI calculation period", 5, 50, 1),
        ("rsi_oversold", "30", "float", "RSI oversold threshold", 10, 45, 1),
        ("rsi_overbought", "70", "float", "RSI overbought threshold", 55, 90, 1),
        ("macd_fast", "12", "int", "MACD fast EMA period", 5, 30, 1),
        ("macd_slow", "26", "int", "MACD slow EMA period", 15, 60, 1),
        ("macd_signal", "9", "int", "MACD signal period", 3, 20, 1),
        ("bb_period", "20", "int", "Bollinger Bands period", 10, 50, 1),
        ("bb_std", "2.0", "float", "Bollinger Bands std dev", 1.0, 3.5, 0.1),
        ("atr_period", "14", "int", "ATR period for volatility", 5, 50, 1),
        ("sl_atr_mult", "1.5", "float", "Stop loss = ATR * this multiplier", 0.5, 5.0, 0.1),
        ("tp_atr_mult", "2.5", "float", "Take profit = ATR * this multiplier", 1.0, 8.0, 0.1),
        ("risk_per_trade", "1.0", "float", "Risk per trade as % of equity", 0.1, 5.0, 0.1),
        ("max_positions", "5", "int", "Max simultaneous open positions", 1, 20, 1),
        ("max_daily_loss_pct", "3.0", "float", "Max daily loss % before halt", 1.0, 10.0, 0.5),
        ("min_confidence", "65", "float", "Minimum confidence to generate signal", 40, 90, 5),
        ("trailing_stop_atr", "1.0", "float", "Trailing stop distance in ATR multiples", 0.5, 3.0, 0.1),
        ("cooldown_loss_count", "3", "int", "Consecutive losses before cooldown", 2, 10, 1),
        ("cooldown_minutes", "120", "int", "Cooldown duration in minutes", 30, 480, 30),
        ("trend_ma_period", "50", "int", "Trend filter MA period", 20, 200, 10),
        ("trend_confirm_tf", "H1", "str", "Timeframe for trend confirmation", None, None, None),
        ("entry_tf", "M15", "str", "Primary entry timeframe", None, None, None),
        ("analysis_tf", "H1", "str", "Analysis timeframe", None, None, None),
    ]
    for name, val, vtype, desc, mn, mx, step in defaults:
        conn.execute(
            "INSERT INTO strategy_params (name, value, value_type, description, min_val, max_val, step, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (name, val, vtype, desc, mn, mx, step, now),
        )


def get_param(name: str, default=None):
    """Get a strategy parameter value."""
    conn = get_conn()
    row = conn.execute("SELECT value, value_type FROM strategy_params WHERE name=?", (name,)).fetchone()
    if not row:
        return default
    v = row["value"]
    t = row["value_type"]
    if t == "int":
        return int(v)
    elif t == "float":
        return float(v)
    elif t == "bool":
        return v.lower() in ("true", "1", "yes")
    return v


def set_param(name: str, value):
    """Update a strategy parameter."""
    conn = get_conn()
    conn.execute("UPDATE strategy_params SET value=?, updated_at=? WHERE name=?",
                 (str(value), time.time(), name))
    conn.commit()


def get_all_params() -> dict:
    """Get all strategy parameters as a dict."""
    conn = get_conn()
    rows = conn.execute("SELECT name, value, value_type FROM strategy_params").fetchall()
    result = {}
    for r in rows:
        v = r["value"]
        t = r["value_type"]
        if t == "int":
            result[r["name"]] = int(v)
        elif t == "float":
            result[r["name"]] = float(v)
        elif t == "bool":
            result[r["name"]] = v.lower() in ("true", "1", "yes")
        else:
            result[r["name"]] = v
    return result


def kv_get(key: str, default=None):
    conn = get_conn()
    row = conn.execute("SELECT value FROM kv_store WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def kv_set(key: str, value: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?,?,?)",
        (key, value, time.time()),
    )
    conn.commit()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
    params = get_all_params()
    print(f"Default parameters: {len(params)} entries")
    for k, v in params.items():
        print(f"  {k}: {v}")
