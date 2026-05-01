"""Scheduler — orchestrates data collection, signal generation, and trade management."""

import time
import threading
import logging
import json
from . import database as db
from . import mt5_data
from . import engine
from . import review
from . import evolution
from . import risk
from . import agents as trading_agents

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Background scheduler for the trading system."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._config = {
            "symbols": ["XAUUSD"],
            "analysis_interval": 300,      # seconds between analysis cycles (5min default)
            "candle_interval": 300,        # seconds between candle fetches
            "review_hour": 0,              # hour to run daily review (0-23, UTC)
            "evolution_interval": 86400,   # seconds between evolution cycles
            "trade_sync_interval": 30,     # seconds between trade syncs
            "trailing_check_interval": 15, # seconds between trailing stop checks
            "mode": "committee",           # "single" (old) or "committee" (multi-agent)
            "auto_execute": False,         # auto-execute approved signals
            "auto_approve_confidence": 70, # auto-approve if confidence >= this
            "last_committee_result": None, # cache last committee result
        }
        self._last_candle_fetch = 0
        self._last_analysis = 0
        self._last_review_date = ""
        self._last_evolution = 0
        self._last_trade_sync = 0
        self._last_trailing_check = 0
        self._status = {"state": "idle", "last_cycle": 0, "cycles": 0}

    def configure(self, **kwargs):
        """Update scheduler configuration."""
        for k, v in kwargs.items():
            if k in self._config:
                self._config[k] = v
        logger.info("Scheduler config updated: %s", self._config)

    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Trading scheduler started")

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Trading scheduler stopped")

    def get_status(self) -> dict:
        """Get current scheduler status."""
        return {
            **self._status,
            "config": self._config,
            "running": self._running,
            "mt5_daemon": mt5_data.daemon.is_ready(),
            "last_candle_fetch": self._last_candle_fetch,
            "last_analysis": self._last_analysis,
            "last_review": self._last_review_date,
        }

    def run_cycle(self) -> dict:
        """Manually trigger one analysis cycle (for testing)."""
        return self._do_analysis_cycle()

    def _loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                now = time.time()
                self._status["state"] = "running"

                # 1. Ensure MT5 daemon (non-blocking — continue analysis even if MT5 down)
                mt5_ready = mt5_data.daemon.is_ready()
                if not mt5_ready:
                    mt5_ready = mt5_data.ensure_daemon()
                    if not mt5_ready:
                        logger.warning("MT5 daemon not available")
                        self._status["mt5_ready"] = False
                    else:
                        self._status["mt5_ready"] = True
                else:
                    self._status["mt5_ready"] = True

                # 2. Skip analysis if already have open position (max 1)
                has_position = False
                if mt5_ready:
                    try:
                        positions_result = mt5_data.daemon.call("positions")
                        pos_list = positions_result.get("positions", []) if positions_result.get("ok") else []
                        has_position = len(pos_list) > 0
                        self._status["open_positions"] = len(pos_list)
                    except Exception:
                        pass

                if has_position:
                    self._status["state"] = "holding"
                    self._status["skip_reason"] = "持仓中，跳过分析"
                    # Still do trade sync and trailing stops
                    if now - self._last_trade_sync >= self._config["trade_sync_interval"]:
                        engine.sync_trades()
                        self._last_trade_sync = now
                    if now - self._last_trailing_check >= self._config["trailing_check_interval"]:
                        self._do_trailing_stops()
                        self._last_trailing_check = now
                    # Calculate next analysis time
                    self._status["next_analysis_at"] = self._last_analysis + self._config["analysis_interval"]
                    time.sleep(5)
                    continue

                # 3. Fetch candles periodically
                if mt5_ready and now - self._last_candle_fetch >= self._config["candle_interval"]:
                    self._do_candle_fetch()
                    self._last_candle_fetch = now

                # 4. Analysis cycle
                interval = self._config["analysis_interval"]
                time_since_last = now - self._last_analysis
                self._status["next_analysis_at"] = self._last_analysis + interval
                self._status["countdown"] = max(0, int(interval - time_since_last))

                if time_since_last >= interval and not self._status.get("analyzing"):
                    # Run analysis in a thread so it doesn't block the loop
                    self._status["analyzing"] = True
                    t = threading.Thread(target=self._run_analysis_threadsafe, daemon=True)
                    t.start()

                # 5. Trade sync
                if mt5_ready and now - self._last_trade_sync >= self._config["trade_sync_interval"]:
                    engine.sync_trades()
                    self._last_trade_sync = now

                # 6. Trailing stop check
                if mt5_ready and now - self._last_trailing_check >= self._config["trailing_check_interval"]:
                    self._do_trailing_stops()
                    self._last_trailing_check = now

                # 7. Daily review
                today = time.strftime("%Y-%m-%d")
                if today != self._last_review_date and time.localtime().tm_hour >= self._config["review_hour"]:
                    result = review.generate_daily_review()
                    if result.get("ok"):
                        logger.info("Daily review generated: %s", result.get("summary", "")[:100])
                    self._last_review_date = today

                    # Evolution after review
                    if now - self._last_evolution >= self._config["evolution_interval"]:
                        evo = evolution.evolve_params()
                        logger.info("Evolution: gen=%s reason=%s", evo.get("generation"), evo.get("reason"))
                        self._last_evolution = now

                self._status["last_cycle"] = now
                self._status["cycles"] += 1
                self._status["state"] = "idle"

            except Exception as e:
                logger.exception("Scheduler error")
                self._status["state"] = f"error: {e}"

            # Sleep between iterations
            time.sleep(5)

    def _do_candle_fetch(self):
        """Fetch latest candles for all configured symbols."""
        for symbol in self._config["symbols"]:
            for tf in ["M15", "H1", "H4"]:
                try:
                    candles = mt5_data.fetch_candles(symbol, tf, count=200)
                    if candles:
                        logger.debug("Fetched %d candles for %s %s", len(candles), symbol, tf)
                except Exception as e:
                    logger.warning("Candle fetch failed for %s %s: %s", symbol, tf, e)

    def _run_analysis_threadsafe(self):
        """Run analysis in a background thread, update status when done."""
        try:
            self._do_analysis_cycle()
        except Exception as e:
            logger.exception("Background analysis failed")
        finally:
            self._last_analysis = time.time()
            self._status["analyzing"] = False
            interval = self._config["analysis_interval"]
            self._status["countdown"] = interval
            self._status["next_analysis_at"] = time.time() + interval

    def _do_analysis_cycle(self) -> dict:
        """Run analysis and generate signals for all symbols."""
        results = {}
        for symbol in self._config["symbols"]:
            try:
                if self._config.get("mode") == "committee":
                    result = self._do_committee_analysis(symbol)
                else:
                    result = engine.generate_signal(symbol)
                results[symbol] = result

                # Auto-approve and auto-execute
                if result.get("ok") and result.get("direction") in ("BUY", "SELL"):
                    confidence = result.get("confidence", 0)
                    if confidence >= self._config.get("auto_approve_confidence", 70):
                        signal_id = result.get("signal_id")
                        if signal_id:
                            conn = db.get_conn()
                            conn.execute("UPDATE signals SET status='approved' WHERE id=? AND status='pending'", (signal_id,))
                            conn.commit()
                            logger.info("Auto-approved signal %s (conf=%d)", signal_id, confidence)

                            if self._config.get("auto_execute"):
                                exec_result = engine.execute_signal(signal_id)
                                result["executed"] = exec_result
                                logger.info("Auto-executed signal %s: %s", signal_id, exec_result)

                    logger.info("Signal: %s %s conf=%d sl=%.2f tp=%.2f",
                                symbol, result.get("direction"), result.get("confidence", 0),
                                result.get("stop_loss", 0), result.get("take_profit", 0))
            except Exception as e:
                logger.exception("Analysis failed for %s", symbol)
                results[symbol] = {"ok": False, "error": str(e)}

        self._status["last_signals"] = results
        return results

    def _do_committee_analysis(self, symbol: str) -> dict:
        """Run multi-agent committee analysis for a symbol."""
        # Collect market data
        market_data = engine.collect_market_data(symbol)
        if not market_data.get("ok"):
            return market_data

        # Run committee
        result = trading_agents.run_committee(market_data, symbol)
        self._config["last_committee_result"] = result

        decision = result.get("decision", {})

        # Store signal if BUY/SELL
        signal_id = None
        if decision.get("direction") in ("BUY", "SELL") and decision.get("confidence", 0) >= 60:
            conn = db.get_conn()
            now = time.time()
            cursor = conn.execute(
                """INSERT INTO signals (symbol, direction, confidence, entry_price, stop_loss, take_profit,
                   lot_size, reason, indicators, risk_pct, mode, status, created_at, expires_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, decision["direction"], decision.get("confidence", 60),
                 decision.get("entry_price", market_data["price"]),
                 decision.get("stop_loss", 0), decision.get("take_profit", 0),
                 decision.get("lot_size_suggestion", 0.01),
                 decision.get("reasoning", ""),
                 json.dumps(market_data["indicators"]),
                 1.0, "committee",
                 "pending", now, now + 600)
            )
            conn.commit()
            signal_id = cursor.lastrowid

        return {
            "ok": True,
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": decision.get("direction", "HOLD"),
            "confidence": decision.get("confidence", 0),
            "entry_price": decision.get("entry_price", 0),
            "stop_loss": decision.get("stop_loss", 0),
            "take_profit": decision.get("take_profit", 0),
            "reasons": decision.get("reasoning", ""),
            "mode": "committee",
            "analyst_count": len(result.get("analyst_opinions", [])),
            "total_latency_ms": result.get("total_latency_ms", 0),
        }

    def _do_trailing_stops(self):
        """Check and update trailing stops for open positions."""
        params = db.get_all_params()
        result = mt5_data.daemon.call("positions")
        if not result.get("ok"):
            return

        for pos in result.get("positions", []):
            symbol = pos["symbol"]
            entry = pos["price_open"]
            current = pos["price_current"]
            direction = "BUY" if pos["type"] == 0 else "SELL"
            ticket = pos["ticket"]
            current_sl = pos.get("sl", 0)

            # Get ATR for trailing distance
            closes = mt5_data.get_closes(symbol, params.get("entry_tf", "M15"), 20)
            if len(closes) < 14:
                continue

            from . import indicators as ind
            ohlcv = mt5_data.get_ohlcv(symbol, params.get("entry_tf", "M15"), 20)
            atr_vals = ind.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], params.get("atr_period", 14))
            current_atr = atr_vals[-1] if atr_vals and atr_vals[-1] is not None else 0
            if current_atr <= 0:
                continue

            trail = risk.calculate_trailing_stop(entry, current, direction, current_atr, params)
            if trail["should_trail"]:
                new_sl = trail["new_sl"]
                # Only move SL in favorable direction
                if direction == "BUY" and (current_sl == 0 or new_sl > current_sl):
                    mt5_data.modify_position(ticket, sl=new_sl)
                    logger.info("Trailing stop: %s #%d SL -> %.5f", symbol, ticket, new_sl)
                elif direction == "SELL" and (current_sl == 0 or new_sl < current_sl):
                    mt5_data.modify_position(ticket, sl=new_sl)
                    logger.info("Trailing stop: %s #%d SL -> %.5f", symbol, ticket, new_sl)


# Singleton
scheduler = TradingScheduler()
