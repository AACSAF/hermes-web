#!/usr/bin/env python3
"""Hermes Web Dashboard — Chat + MT5 Trading Panel.

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


# ─── MT5 Daemon ────────────────────────────────────────────────────
MT5_DAEMON_SCRIPT = r"C:\Users\Administrator\mt5_daemon.py"

class MT5Daemon:
    """Persistent MT5 daemon process on Windows side. Single init, sub-ms queries."""

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
                    ["powershell.exe", "-Command",
                     f'python "{MT5_DAEMON_SCRIPT}"'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                # Read the "ready" line
                line = self._proc.stdout.readline().decode("utf-8", errors="replace").strip()
                resp = json.loads(line)
                if resp.get("ready"):
                    self._ready = True
                    logger.info("MT5 daemon started, pid=%s", self._proc.pid)
                    return {"ok": True, "started": True}
                else:
                    return {"ok": False, "error": f"Daemon not ready: {resp}"}
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
                    return {"ok": False, "error": "Empty response from daemon"}
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


mt5_daemon = MT5Daemon()


def check_mt5_process() -> bool:
    """Check if MT5 terminal64.exe is running on Windows."""
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


# ── MT5 API ────────────────────────────────────────────────────────

async def handle_mt5_status(request):
    """GET /api/mt5/status — check MT5 process + daemon."""
    loop = asyncio.get_event_loop()
    running = await loop.run_in_executor(None, check_mt5_process)
    if mt5_daemon.is_ready():
        result = mt5_daemon.call("connect")
        result["running"] = True
        result["daemon"] = True
        return web.json_response(result)
    return web.json_response({"ok": False, "running": running, "daemon": False})


async def handle_mt5_connect(request):
    """POST /api/mt5/connect — start daemon + get account info."""
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
    """GET /api/mt5/tick — all-in-one refresh: account + positions + orders."""
    if not mt5_daemon.is_ready():
        return web.json_response({"ok": False, "error": "Daemon not connected"})
    return web.json_response(mt5_daemon.call("tick"))


async def handle_mt5_positions(request):
    """GET /api/mt5/positions — open positions."""
    if not mt5_daemon.is_ready():
        return web.json_response({"ok": False, "error": "Daemon not connected"})
    return web.json_response(mt5_daemon.call("positions"))


async def handle_mt5_orders(request):
    """GET /api/mt5/orders — pending orders."""
    if not mt5_daemon.is_ready():
        return web.json_response({"ok": False, "error": "Daemon not connected"})
    return web.json_response(mt5_daemon.call("orders"))


async def handle_mt5_symbols(request):
    """GET /api/mt5/symbols — available symbols."""
    if not mt5_daemon.is_ready():
        return web.json_response({"ok": False, "error": "Daemon not connected"})
    return web.json_response(mt5_daemon.call("symbols"))


async def handle_mt5_ticker(request):
    """GET /api/mt5/ticker/{symbol} — current price."""
    symbol = request.match_info["symbol"]
    if not mt5_daemon.is_ready():
        return web.json_response({"ok": False, "error": "Daemon not connected"})
    return web.json_response(mt5_daemon.call("ticker", symbol=symbol))


async def handle_mt5_disconnect(request):
    """POST /api/mt5/disconnect — shutdown daemon."""
    mt5_daemon.shutdown()
    return web.json_response({"ok": True})


async def handle_health(request):
    return web.json_response({"status": "ok", "sessions": len(sessions), "mt5_daemon": mt5_daemon.is_ready()})


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

    app.router.add_get("/api/health", handle_health)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9090))
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    logging.getLogger().handlers[0].flush = lambda: sys.stderr.flush()

    logger.info("Starting Hermes Web Dashboard on http://localhost:%d", port)
    web.run_app(create_app(), host="0.0.0.0", port=port, print=lambda msg: logger.info(msg))
