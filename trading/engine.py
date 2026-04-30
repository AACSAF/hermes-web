"""Trading engine — signal generation, analysis, and trade management."""

import json
import time
import logging
from typing import Optional
from . import database as db
from . import indicators as ind
from . import risk
from . import mt5_data

logger = logging.getLogger(__name__)


# ── Market Analysis ────────────────────────────────────────────────

def analyze_market(symbol: str) -> dict:
    """Full market analysis for a symbol. Returns structured analysis."""
    params = db.get_all_params()

    entry_tf = params.get("entry_tf", "M15")
    analysis_tf = params.get("analysis_tf", "H1")

    # Fetch candle data
    ohlcv_entry = mt5_data.get_ohlcv(symbol, entry_tf, limit=200)
    ohlcv_analysis = mt5_data.get_ohlcv(symbol, analysis_tf, limit=200)

    if not ohlcv_entry["close"] or len(ohlcv_entry["close"]) < 50:
        return {"ok": False, "error": f"{symbol} {entry_tf} 数据不足"}

    closes_e = ohlcv_entry["close"]
    highs_e = ohlcv_entry["high"]
    lows_e = ohlcv_entry["low"]
    opens_e = ohlcv_entry["open"]

    closes_a = ohlcv_analysis["close"]

    # ── Indicators on entry timeframe ──
    rsi_values = ind.rsi(closes_e, params.get("rsi_period", 14))
    macd_data = ind.macd(closes_e, params.get("macd_fast", 12), params.get("macd_slow", 26), params.get("macd_signal", 9))
    bb_data = ind.bollinger_bands(closes_e, params.get("bb_period", 20), params.get("bb_std", 2.0))
    atr_values = ind.atr(highs_e, lows_e, closes_e, params.get("atr_period", 14))
    stoch = ind.stochastic(highs_e, lows_e, closes_e)
    sr = ind.support_resistance(closes_e)

    current_rsi = rsi_values[-1] if rsi_values[-1] is not None else 50
    current_macd = macd_data["macd"][-1] if macd_data["macd"][-1] is not None else 0
    current_signal = macd_data["signal"][-1] if macd_data["signal"][-1] is not None else 0
    current_hist = macd_data["histogram"][-1] if macd_data["histogram"][-1] is not None else 0
    prev_hist = macd_data["histogram"][-2] if len(macd_data["histogram"]) > 1 and macd_data["histogram"][-2] is not None else 0
    current_atr = atr_values[-1] if atr_values[-1] is not None else 0
    current_bb_upper = bb_data["upper"][-1]
    current_bb_lower = bb_data["lower"][-1]
    current_bb_mid = bb_data["middle"][-1]
    current_stoch_k = stoch["k"][-1] if stoch["k"][-1] is not None else 50
    current_stoch_d = stoch["d"][-1] if stoch["d"][-1] is not None else 50

    # ── Trend on analysis timeframe ──
    trend_ma = params.get("trend_ma_period", 50)
    trend = ind.trend_direction(closes_a, trend_ma) if len(closes_a) >= trend_ma else "FLAT"

    # ── Candle patterns ──
    patterns = ind.candle_pattern(opens_e, highs_e, lows_e, closes_e)

    # ── Build indicator summary ──
    indicators = {
        "rsi": round(current_rsi, 1),
        "macd_line": round(current_macd, 5),
        "macd_signal": round(current_signal, 5),
        "macd_histogram": round(current_hist, 5),
        "macd_cross": "bullish" if current_hist > 0 and prev_hist <= 0 else ("bearish" if current_hist < 0 and prev_hist >= 0 else "none"),
        "bb_upper": round(current_bb_upper, 5) if current_bb_upper else None,
        "bb_lower": round(current_bb_lower, 5) if current_bb_lower else None,
        "bb_middle": round(current_bb_mid, 5) if current_bb_mid else None,
        "bb_position": "above_upper" if closes_e[-1] > (current_bb_upper or 0) else ("below_lower" if closes_e[-1] < (current_bb_lower or 0) else "inside"),
        "atr": round(current_atr, 5),
        "stoch_k": round(current_stoch_k, 1),
        "stoch_d": round(current_stoch_d, 1),
        "trend_h1": trend,
        "candle_patterns": patterns,
        "support_levels": sr["supports"],
        "resistance_levels": sr["resistances"],
        "current_price": closes_e[-1],
    }

    # ── Score signals ──
    bullish_score = 0
    bearish_score = 0
    reasons_bull = []
    reasons_bear = []

    # RSI
    rsi_os = params.get("rsi_oversold", 30)
    rsi_ob = params.get("rsi_overbought", 70)
    if current_rsi < rsi_os:
        bullish_score += 20
        reasons_bull.append(f"RSI 超卖({current_rsi:.0f}<{rsi_os})")
    elif current_rsi > rsi_ob:
        bearish_score += 20
        reasons_bear.append(f"RSI 超买({current_rsi:.0f}>{rsi_ob})")

    # MACD crossover
    if indicators["macd_cross"] == "bullish":
        bullish_score += 25
        reasons_bull.append("MACD 金叉")
    elif indicators["macd_cross"] == "bearish":
        bearish_score += 25
        reasons_bear.append("MACD 死叉")

    # MACD histogram momentum
    if current_hist > 0 and current_hist > prev_hist:
        bullish_score += 10
        reasons_bull.append("MACD 动能增强")
    elif current_hist < 0 and current_hist < prev_hist:
        bearish_score += 10
        reasons_bear.append("MACD 动能增强(空)")

    # Bollinger Bands
    if indicators["bb_position"] == "below_lower":
        bullish_score += 15
        reasons_bull.append("价格触及布林带下轨")
    elif indicators["bb_position"] == "above_upper":
        bearish_score += 15
        reasons_bear.append("价格触及布林带上轨")

    # Stochastic
    if current_stoch_k < 20 and current_stoch_d < 20:
        bullish_score += 10
        reasons_bull.append(f"随机指标超卖(K={current_stoch_k:.0f})")
    elif current_stoch_k > 80 and current_stoch_d > 80:
        bearish_score += 10
        reasons_bear.append(f"随机指标超买(K={current_stoch_k:.0f})")

    # Trend filter (strong weight)
    if trend == "UP":
        bullish_score += 20
        reasons_bull.append(f"H1上升趋势")
    elif trend == "DOWN":
        bearish_score += 20
        reasons_bear.append(f"H1下降趋势")

    # Candle patterns
    for p in patterns:
        if "BULLISH" in p:
            bullish_score += 15
            reasons_bull.append(f"K线形态: {p}")
        elif "BEARISH" in p:
            bearish_score += 15
            reasons_bear.append(f"K线形态: {p}")

    # ── Determine direction ──
    if bullish_score > bearish_score and bullish_score >= params.get("min_confidence", 65):
        direction = "BUY"
        confidence = min(bullish_score, 100)
        reasons = reasons_bull
    elif bearish_score > bullish_score and bearish_score >= params.get("min_confidence", 65):
        direction = "SELL"
        confidence = min(bearish_score, 100)
        reasons = reasons_bear
    else:
        direction = "FLAT"
        confidence = max(bullish_score, bearish_score)
        reasons = ["信号不足，建议观望"]

    # ── Calculate SL/TP ──
    sl_tp = risk.calculate_sl_tp(closes_e[-1], direction, current_atr, params) if direction != "FLAT" and current_atr > 0 else {}

    return {
        "ok": True,
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "entry_price": closes_e[-1],
        "stop_loss": sl_tp.get("sl", 0),
        "take_profit": sl_tp.get("tp", 0),
        "sl_distance": sl_tp.get("sl_distance", 0),
        "tp_distance": sl_tp.get("tp_distance", 0),
        "reasons": reasons,
        "indicators": indicators,
        "trend": trend,
        "atr": current_atr,
    }


# ── Signal Generation ──────────────────────────────────────────────

def generate_signal(symbol: str) -> dict:
    """Analyze market and generate a trade signal (observe mode — no execution)."""
    analysis = analyze_market(symbol)
    if not analysis.get("ok"):
        return analysis

    direction = analysis["direction"]
    if direction == "FLAT":
        return {"ok": True, "signal": None, "reason": "信号不足，观望"}

    # Get account info for lot sizing
    account = mt5_data.daemon.call("connect")
    if not account.get("ok"):
        return {"ok": False, "error": "无法获取账户信息"}

    equity = account["account"]["equity"]
    params = db.get_all_params()

    # Calculate lot size
    sl_pips = analysis["sl_distance"] / risk.get_pip_size(symbol)
    pip_val = risk.get_pip_value(symbol)
    lot_size = risk.calculate_lot_size(equity, params.get("risk_per_trade", 1.0), sl_pips, pip_val)

    # Risk check
    positions = mt5_data.daemon.call("positions")
    pos_list = positions.get("positions", []) if positions.get("ok") else []
    risk_check = risk.check_trade_allowed(symbol, direction, equity, pos_list)

    # Store signal
    conn = db.get_conn()
    now = time.time()
    cursor = conn.execute(
        """INSERT INTO signals (symbol, direction, confidence, entry_price, stop_loss, take_profit,
           lot_size, reason, indicators, risk_pct, mode, status, created_at, expires_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (symbol, direction, analysis["confidence"], analysis["entry_price"],
         analysis["stop_loss"], analysis["take_profit"], lot_size,
         json.dumps(analysis["reasons"], ensure_ascii=False),
         json.dumps(analysis["indicators"], ensure_ascii=False),
         params.get("risk_per_trade", 1.0),
         "observe", "pending" if risk_check["allowed"] else "rejected",
         now, now + 300)  # Signal expires in 5 minutes
    )
    conn.commit()

    signal_id = cursor.lastrowid

    return {
        "ok": True,
        "signal_id": signal_id,
        "symbol": symbol,
        "direction": direction,
        "confidence": analysis["confidence"],
        "entry_price": analysis["entry_price"],
        "stop_loss": analysis["stop_loss"],
        "take_profit": analysis["take_profit"],
        "lot_size": lot_size,
        "reasons": analysis["reasons"],
        "risk_check": risk_check,
        "mode": "observe",
    }


# ── Trade Execution ────────────────────────────────────────────────

def execute_signal(signal_id: int) -> dict:
    """Execute a signal (semi-auto or auto mode)."""
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM signals WHERE id=?", (signal_id,)).fetchone()
    if not row:
        return {"ok": False, "error": "信号不存在"}
    if row["status"] not in ("pending", "approved"):
        return {"ok": False, "error": f"信号状态为 {row['status']}，无法执行"}

    # Check expiry
    if row["expires_at"] and time.time() > row["expires_at"]:
        conn.execute("UPDATE signals SET status='expired' WHERE id=?", (signal_id,))
        conn.commit()
        return {"ok": False, "error": "信号已过期"}

    # Execute via MT5
    result = mt5_data.place_order(
        symbol=row["symbol"],
        direction=row["direction"],
        lot_size=row["lot_size"],
        sl=row["stop_loss"],
        tp=row["take_profit"],
        comment=f"hermes_{signal_id}"
    )

    if result.get("ok"):
        conn.execute(
            "UPDATE signals SET status='executed', executed_at=?, deal_ticket=? WHERE id=?",
            (time.time(), result.get("deal", 0), signal_id)
        )
        conn.commit()
        return {"ok": True, "ticket": result.get("ticket"), "price": result.get("price")}
    else:
        conn.execute(
            "UPDATE signals SET status='failed', notes=? WHERE id=?",
            (result.get("error", ""), signal_id)
        )
        conn.commit()
        return {"ok": False, "error": result.get("error", "执行失败")}


# ── Trade Tracking ─────────────────────────────────────────────────

def sync_trades():
    """Sync open positions with the trades table. Record new entries and detect exits."""
    conn = db.get_conn()

    # Get current positions from MT5
    result = mt5_data.daemon.call("positions")
    if not result.get("ok"):
        return

    current_tickets = set()
    for pos in result.get("positions", []):
        ticket = pos["ticket"]
        current_tickets.add(ticket)

        # Check if already tracked
        existing = conn.execute("SELECT id FROM trades WHERE ticket=?", (ticket,)).fetchone()
        if not existing:
            # New trade — record it
            conn.execute(
                """INSERT INTO trades (ticket, symbol, direction, lot_size, entry_price, entry_time,
                   stop_loss, take_profit, context)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (ticket, pos["symbol"], "BUY" if pos["type"] == 0 else "SELL",
                 pos["volume"], pos["price_open"], pos["time"],
                 pos.get("sl", 0), pos.get("tp", 0),
                 json.dumps({"magic": pos.get("magic"), "comment": pos.get("comment")}))
            )

        # Update MFE/MAE (max favorable/adverse excursion)
        trade = conn.execute("SELECT id, entry_price, max_favorable, max_adverse FROM trades WHERE ticket=?", (ticket,)).fetchone()
        if trade:
            current = pos["price_current"]
            entry = trade["entry_price"]
            direction = pos["type"]  # 0=BUY, 1=SELL
            if direction == 0:
                favorable = current - entry
                adverse = entry - current
            else:
                favorable = entry - current
                adverse = current - entry
            if favorable > trade["max_favorable"]:
                conn.execute("UPDATE trades SET max_favorable=? WHERE id=?", (favorable, trade["id"]))
            if adverse > trade["max_adverse"]:
                conn.execute("UPDATE trades SET max_adverse=? WHERE id=?", (adverse, trade["id"]))

    # Detect closed trades (were in trades table but no longer open)
    tracked = conn.execute("SELECT * FROM trades WHERE exit_time IS NULL").fetchall()
    for t in tracked:
        if t["ticket"] not in current_tickets:
            # Position closed — find the exit info from deal history
            # For now, mark with current data
            conn.execute(
                """UPDATE trades SET exit_time=?, exit_reason='sync_detected',
                   profit=?, hold_duration=?
                   WHERE id=?""",
                (time.time(), 0, time.time() - t["entry_time"], t["id"])
            )

    conn.commit()


def get_open_signals() -> list:
    """Get all pending/approved signals."""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT * FROM signals WHERE status IN ('pending','approved') ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_trade_history(limit: int = 50) -> list:
    """Get recent trade history."""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]
