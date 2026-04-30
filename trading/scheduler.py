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

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Background scheduler for the trading system."""

    def __init__(self):
        self._running = False
        self._thread = None
        self._config = {
            "symbols": ["XAUUSD"],
            "analysis_interval": 60,      # seconds between analysis cycles
            "candle_interval": 300,       # seconds between candle fetches
            "review_hour": 0,             # hour to run daily review (0-23, UTC)
            "evolution_interval": 86400,  # seconds between evolution cycles
            "trade_sync_interval": 30,    # seconds between trade syncs
            "trailing_check_interval": 15, # seconds between trailing stop checks
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

                # 1. Ensure MT5 daemon
                if not mt5_data.daemon.is_ready():
                    if not mt5_data.ensure_daemon():
                        logger.warning("MT5 daemon not available, sleeping 30s")
                        time.sleep(30)
                        continue

                # 2. Fetch candles periodically
                if now - self._last_candle_fetch >= self._config["candle_interval"]:
                    self._do_candle_fetch()
                    self._last_candle_fetch = now

                # 3. Analysis cycle
                if now - self._last_analysis >= self._config["analysis_interval"]:
                    self._do_analysis_cycle()
                    self._last_analysis = now

                # 4. Trade sync
                if now - self._last_trade_sync >= self._config["trade_sync_interval"]:
                    engine.sync_trades()
                    self._last_trade_sync = now

                # 5. Trailing stop check
                if now - self._last_trailing_check >= self._config["trailing_check_interval"]:
                    self._do_trailing_stops()
                    self._last_trailing_check = now

                # 6. Daily review
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

    def _do_analysis_cycle(self) -> dict:
        """Run analysis and generate signals for all symbols."""
        results = {}
        for symbol in self._config["symbols"]:
            try:
                result = engine.generate_signal(symbol)
                results[symbol] = result
                if result.get("signal_id"):
                    logger.info("Signal: %s %s conf=%d sl=%.2f tp=%.2f",
                                symbol, result.get("direction"), result.get("confidence"),
                                result.get("stop_loss", 0), result.get("take_profit", 0))
            except Exception as e:
                logger.exception("Analysis failed for %s", symbol)
                results[symbol] = {"ok": False, "error": str(e)}

        self._status["last_signals"] = results
        return results

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
