#!/usr/bin/env python3
"""Hermes Web Dashboard — Chat + MT5 Trading + AI Trading System.

Serves on http://localhost:9090
"""

import asyncio
import json
import logging
import os
import sys
import uuid
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

# Add hermes-agent to sys.path
HERMES_AGENT_DIR = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(HERMES_AGENT_DIR))

# Load .env before importing hermes modules
from hermes_constants import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv

load_hermes_dotenv(hermes_home=get_hermes_home(), project_env=HERMES_AGENT_DIR / ".env")

from aiohttp import web

# ─── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hermes-web")

# ─── Trading System Init ───────────────────────────────────────────
from trading import database as db
from trading import mt5_data
from trading import engine
from trading import risk
from trading import review
from trading import evolution
from trading.scheduler import scheduler

db.init_db()
logger.info("Trading database initialized")

# ─── Chat Session Store ────────────────────────────────────────────
sessions: dict[str, dict] = {}
MAX_HISTORY = 50


def get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    sid = session_id or str(uuid.uuid4())[:8]
    sessions[sid] = {"history": [], "created": datetime.now().isoformat()}
    return sid, sessions[sid]


# ─── AIAgent Management ────────────────────────────────────────────
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from run_agent import AIAgent
        from cli import load_cli_config

        config = load_cli_config()
        model_cfg = config.get("model", {})
        api_key = os.environ.get("XIAOMI_API_KEY", "")

        _agent = AIAgent(
            base_url=model_cfg.get("base_url"),
            provider=model_cfg.get("provider"),
            model=model_cfg.get("default", ""),
            api_key=api_key,
            platform="api_server",
            skip_context_files=True,
            quiet_mode=True,
        )
        logger.info("AIAgent initialized: model=%s provider=%s",
                     model_cfg.get("default"), model_cfg.get("provider"))
    return _agent


# ─── MT5 Daemon (backward compat with old frontend) ───────────────

class MT5DaemonCompat:
    """Wrapper that delegates to mt5_data.daemon."""
    def is_ready(self):
        return mt5_data.daemon.is_ready()
    def start(self):
        return mt5_data.daemon.start()
    def call(self, action, **kwargs):
        return mt5_data.daemon.call(action, **kwargs)
    def shutdown(self):
        mt5_data.daemon.shutdown()

mt5_daemon = MT5DaemonCompat()


def check_mt5_process() -> bool:
    return mt5_data.check_mt5_running()


# ─── Request Handlers ──────────────────────────────────────────────

async def handle_index(request):
    html_path = Path(__file__).parent / "index.html"
    return web.FileResponse(html_path)


# ── Chat API ───────────────────────────────────────────────────────

async def handle_chat(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    user_msg = (data.get("message") or "").strip()
    session_id = data.get("session_id")

    if not user_msg:
        return web.json_response({"error": "Empty message"}, status=400)

    sid, session = get_or_create_session(session_id)

    try:
        agent = get_agent()
        loop = asyncio.get_event_loop()
        history = [{"role": m["role"], "content": m["content"]} for m in session["history"]]

        def run_chat():
            return agent.run_conversation(
                user_message=user_msg,
                conversation_history=history if history else None,
            )

        result = await loop.run_in_executor(None, run_chat)

        response_text = ""
        if isinstance(result, dict):
            response_text = result.get("final_response", "")
        elif result:
            response_text = str(result)

        if not response_text:
            response_text = "(无响应)"

        session["history"].append({"role": "user", "content": user_msg})
        session["history"].append({"role": "assistant", "content": response_text})

        if len(session["history"]) > MAX_HISTORY * 2:
            session["history"] = session["history"][-(MAX_HISTORY * 2):]

        return web.json_response({
            "session_id": sid,
            "response": response_text,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.exception("Chat error")
        return web.json_response({"error": str(e)}, status=500)


async def handle_sessions(request):
    return web.json_response({
        sid: {"created": s["created"], "turns": len(s["history"]) // 2}
        for sid, s in sessions.items()
    })


async def handle_new_session(request):
    sid, session = get_or_create_session(None)
    return web.json_response({"session_id": sid, "created": session["created"]})


async def handle_delete_session(request):
    sid = request.match_info["session_id"]
    if sid in sessions:
        del sessions[sid]
        return web.json_response({"ok": True})
    return web.json_response({"error": "Not found"}, status=404)


# ── MT5 API (existing, backward compatible) ────────────────────────

async def handle_mt5_status(request):
    loop = asyncio.get_event_loop()
    running = await loop.run_in_executor(None, check_mt5_process)
    if mt5_daemon.is_ready():
        result = mt5_daemon.call("connect")
        result["running"] = True
        result["daemon"] = True
        return web.json_response(result)
    return web.json_response({"ok": False, "running": running, "daemon": False})


async def handle_mt5_connect(request):
    loop = asyncio.get_event_loop()
    logger.info("MT5 connect: checking process...")
    running = await loop.run_in_executor(None, check_mt5_process)
    logger.info("MT5 process running: %s", running)
    if not running:
        return web.json_response({"ok": False, "error": "MT5 客户端未运行，请先在 Windows 上启动 MetaTrader 5"})
    if not mt5_daemon.is_ready():
        logger.info("MT5 connect: starting daemon...")
        result = await loop.run_in_executor(None, mt5_daemon.start)
        if not result.get("ok"):
            return web.json_response(result)
    logger.info("MT5 connect: querying account info...")
    result = mt5_daemon.call("connect")
    logger.info("MT5 connect: result ok=%s", result.get("ok"))
    return web.json_response(result)


async def handle_mt5_tick(request):
    if not mt5_daemon.is_ready():
        return web.json_response({"ok": False, "error": "Daemon not connected"})
    return web.json_response(mt5_daemon.call("tick"))


async def handle_mt5_positions(request):
    if not mt5_daemon.is_ready():
        return web.json_response({"ok": False, "error": "Daemon not connected"})
    return web.json_response(mt5_daemon.call("positions"))


async def handle_mt5_orders(request):
    if not mt5_daemon.is_ready():
        return web.json_response({"ok": False, "error": "Daemon not connected"})
    return web.json_response(mt5_daemon.call("orders"))


async def handle_mt5_symbols(request):
    if not mt5_daemon.is_ready():
        return web.json_response({"ok": False, "error": "Daemon not connected"})
    return web.json_response(mt5_daemon.call("symbols"))


async def handle_mt5_ticker(request):
    symbol = request.match_info["symbol"]
    if not mt5_daemon.is_ready():
        return web.json_response({"ok": False, "error": "Daemon not connected"})
    return web.json_response(mt5_daemon.call("ticker", symbol=symbol))


async def handle_mt5_disconnect(request):
    mt5_daemon.shutdown()
    return web.json_response({"ok": True})


# ── Trading Engine API ─────────────────────────────────────────────

async def handle_trading_analyze(request):
    """GET /api/trading/analyze/{symbol} — full market analysis."""
    symbol = request.match_info["symbol"]
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, engine.analyze_market, symbol)
    return web.json_response(result)


async def handle_trading_signal(request):
    """POST /api/trading/signal — generate a trade signal (observe mode)."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    symbol = data.get("symbol", "XAUUSD")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, engine.generate_signal, symbol)
    return web.json_response(result)


async def handle_trading_signals(request):
    """GET /api/trading/signals — list open signals."""
    signals = engine.get_open_signals()
    return web.json_response({"ok": True, "signals": signals, "count": len(signals)})


async def handle_trading_execute(request):
    """POST /api/trading/execute — execute/approve a signal."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    signal_id = data.get("signal_id")
    if not signal_id:
        return web.json_response({"error": "Missing signal_id"}, status=400)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, engine.execute_signal, signal_id)
    return web.json_response(result)


async def handle_trading_approve(request):
    """POST /api/trading/approve — approve a signal for execution."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    signal_id = data.get("signal_id")
    if not signal_id:
        return web.json_response({"error": "Missing signal_id"}, status=400)
    conn = db.get_conn()
    conn.execute("UPDATE signals SET status='approved' WHERE id=? AND status='pending'", (signal_id,))
    conn.commit()
    return web.json_response({"ok": True})


async def handle_trading_reject(request):
    """POST /api/trading/reject — reject a signal."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    signal_id = data.get("signal_id")
    if not signal_id:
        return web.json_response({"error": "Missing signal_id"}, status=400)
    conn = db.get_conn()
    conn.execute("UPDATE signals SET status='rejected' WHERE id=? AND status IN ('pending','approved')", (signal_id,))
    conn.commit()
    return web.json_response({"ok": True})


# ── Trades History API ─────────────────────────────────────────────

async def handle_trading_history(request):
    """GET /api/trading/history — trade history."""
    limit = int(request.query.get("limit", "50"))
    trades = engine.get_trade_history(limit)
    return web.json_response({"ok": True, "trades": trades, "count": len(trades)})


# ── Review API ─────────────────────────────────────────────────────

async def handle_review_daily(request):
    """GET /api/review/daily?date=YYYY-MM-DD — daily review."""
    date_str = request.query.get("date")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, review.generate_daily_review, date_str)
    return web.json_response(result)


async def handle_review_weekly(request):
    """GET /api/review/weekly — weekly summary."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, review.get_weekly_summary)
    return web.json_response(result)


async def handle_review_history(request):
    """GET /api/review/history — list of past reviews."""
    conn = db.get_conn()
    rows = conn.execute("SELECT * FROM reviews ORDER BY date DESC LIMIT 30").fetchall()
    return web.json_response({"ok": True, "reviews": [dict(r) for r in rows]})


# ── Evolution API ──────────────────────────────────────────────────

async def handle_evolution_params(request):
    """GET /api/evolution/params — current strategy parameters."""
    params = db.get_all_params()
    conn = db.get_conn()
    rows = conn.execute("SELECT name, value, value_type, description, min_val, max_val, step FROM strategy_params").fetchall()
    return web.json_response({"ok": True, "params": [dict(r) for r in rows]})


async def handle_evolution_update(request):
    """POST /api/evolution/params — update a parameter."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    name = data.get("name")
    value = data.get("value")
    if not name or value is None:
        return web.json_response({"error": "Missing name/value"}, status=400)
    db.set_param(name, value)
    return web.json_response({"ok": True})


async def handle_evolution_run(request):
    """POST /api/evolution/run — trigger evolution."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, evolution.evolve_params)
    return web.json_response(result)


async def handle_evolution_history(request):
    """GET /api/evolution/history — evolution log."""
    history = evolution.get_evolution_history()
    return web.json_response({"ok": True, "history": history})


async def handle_evolution_rules(request):
    """GET /api/evolution/rules — trading rules."""
    category = request.query.get("category")
    rules = evolution.get_rules(category)
    return web.json_response({"ok": True, "rules": rules})


# ── Scheduler API ──────────────────────────────────────────────────

async def handle_scheduler_status(request):
    """GET /api/scheduler/status — scheduler status."""
    return web.json_response({"ok": True, **scheduler.get_status()})


async def handle_scheduler_start(request):
    """POST /api/scheduler/start — start the trading scheduler."""
    # Ensure MT5 daemon
    if not mt5_daemon.is_ready():
        loop = asyncio.get_event_loop()
        running = await loop.run_in_executor(None, check_mt5_process)
        if not running:
            return web.json_response({"ok": False, "error": "MT5 未运行"})
        await loop.run_in_executor(None, mt5_daemon.start)

    scheduler.start()
    return web.json_response({"ok": True, "message": "调度器已启动"})


async def handle_scheduler_stop(request):
    """POST /api/scheduler/stop — stop the scheduler."""
    scheduler.stop()
    return web.json_response({"ok": True, "message": "调度器已停止"})


async def handle_scheduler_config(request):
    """POST /api/scheduler/config — update scheduler config."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    scheduler.configure(**data)
    return web.json_response({"ok": True, "config": scheduler._config})


async def handle_scheduler_run_cycle(request):
    """POST /api/scheduler/cycle — manually run one analysis cycle."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, scheduler.run_cycle)
    return web.json_response({"ok": True, "result": result})


async def handle_health(request):
    return web.json_response({
        "status": "ok",
        "sessions": len(sessions),
        "mt5_daemon": mt5_daemon.is_ready(),
        "scheduler": scheduler.get_status().get("state", "idle"),
    })


# ─── App Setup ─────────────────────────────────────────────────────

def create_app():
    app = web.Application(client_max_size=10 * 1024 * 1024)

    app.router.add_get("/", handle_index)
    app.router.add_get("/index.html", handle_index)

    # Chat
    app.router.add_post("/api/chat", handle_chat)
    app.router.add_get("/api/sessions", handle_sessions)
    app.router.add_post("/api/sessions/new", handle_new_session)
    app.router.add_delete("/api/sessions/{session_id}", handle_delete_session)

    # MT5
    app.router.add_get("/api/mt5/status", handle_mt5_status)
    app.router.add_post("/api/mt5/connect", handle_mt5_connect)
    app.router.add_post("/api/mt5/disconnect", handle_mt5_disconnect)
    app.router.add_get("/api/mt5/positions", handle_mt5_positions)
    app.router.add_get("/api/mt5/orders", handle_mt5_orders)
    app.router.add_get("/api/mt5/symbols", handle_mt5_symbols)
    app.router.add_get("/api/mt5/ticker/{symbol}", handle_mt5_ticker)
    app.router.add_get("/api/mt5/tick", handle_mt5_tick)

    # Trading Engine
    app.router.add_get("/api/trading/analyze/{symbol}", handle_trading_analyze)
    app.router.add_post("/api/trading/signal", handle_trading_signal)
    app.router.add_get("/api/trading/signals", handle_trading_signals)
    app.router.add_post("/api/trading/execute", handle_trading_execute)
    app.router.add_post("/api/trading/approve", handle_trading_approve)
    app.router.add_post("/api/trading/reject", handle_trading_reject)
    app.router.add_get("/api/trading/history", handle_trading_history)

    # Review
    app.router.add_get("/api/review/daily", handle_review_daily)
    app.router.add_get("/api/review/weekly", handle_review_weekly)
    app.router.add_get("/api/review/history", handle_review_history)

    # Evolution
    app.router.add_get("/api/evolution/params", handle_evolution_params)
    app.router.add_post("/api/evolution/params", handle_evolution_update)
    app.router.add_post("/api/evolution/run", handle_evolution_run)
    app.router.add_get("/api/evolution/history", handle_evolution_history)
    app.router.add_get("/api/evolution/rules", handle_evolution_rules)

    # Scheduler
    app.router.add_get("/api/scheduler/status", handle_scheduler_status)
    app.router.add_post("/api/scheduler/start", handle_scheduler_start)
    app.router.add_post("/api/scheduler/stop", handle_scheduler_stop)
    app.router.add_post("/api/scheduler/config", handle_scheduler_config)
    app.router.add_post("/api/scheduler/cycle", handle_scheduler_run_cycle)

    # Health
    app.router.add_get("/api/health", handle_health)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9090))
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    logging.getLogger().handlers[0].flush = lambda: sys.stderr.flush()

    logger.info("Starting Hermes Web Dashboard on http://localhost:%d", port)
    web.run_app(create_app(), host="0.0.0.0", port=port, print=lambda msg: logger.info(msg))
