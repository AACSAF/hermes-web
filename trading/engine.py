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


def collect_market_data(symbol: str) -> dict:
    """Collect raw market data for AI analysis (no scoring)."""
    params = db.get_all_params()
    entry_tf = params.get("entry_tf", "M15")
    analysis_tf = params.get("analysis_tf", "H1")

    ohlcv_entry = mt5_data.get_ohlcv(symbol, entry_tf, limit=200)
    ohlcv_analysis = mt5_data.get_ohlcv(symbol, analysis_tf, limit=200)

    if not ohlcv_entry["close"] or len(ohlcv_entry["close"]) < 50:
        return {"ok": False, "error": f"{symbol} {entry_tf} 数据不足"}

    closes_e = ohlcv_entry["close"]
    highs_e = ohlcv_entry["high"]
    lows_e = ohlcv_entry["low"]
    opens_e = ohlcv_entry["open"]
    closes_a = ohlcv_analysis["close"]

    rsi_values = ind.rsi(closes_e, params.get("rsi_period", 14))
    macd_data = ind.macd(closes_e, params.get("macd_fast", 12), params.get("macd_slow", 26), params.get("macd_signal", 9))
    bb_data = ind.bollinger_bands(closes_e, params.get("bb_period", 20), params.get("bb_std", 2.0))
    atr_values = ind.atr(highs_e, lows_e, closes_e, params.get("atr_period", 14))
    stoch = ind.stochastic(highs_e, lows_e, closes_e)
    sr = ind.support_resistance(closes_e)
    patterns = ind.candle_pattern(opens_e, highs_e, lows_e, closes_e)
    trend = ind.trend_direction(closes_a, params.get("trend_ma_period", 50)) if len(closes_a) >= params.get("trend_ma_period", 50) else "FLAT"

    current_atr = atr_values[-1] if atr_values and atr_values[-1] is not None else 0
    current_rsi = rsi_values[-1] if rsi_values[-1] is not None else 50
    current_hist = macd_data["histogram"][-1] if macd_data["histogram"][-1] is not None else 0
    prev_hist = macd_data["histogram"][-2] if len(macd_data["histogram"]) > 1 and macd_data["histogram"][-2] is not None else 0

    macd_cross = "bullish" if current_hist > 0 and prev_hist <= 0 else ("bearish" if current_hist < 0 and prev_hist >= 0 else "none")
    bb_pos = "above_upper" if closes_e[-1] > (bb_data["upper"][-1] or 0) else ("below_lower" if closes_e[-1] < (bb_data["lower"][-1] or 0) else "inside")

    # Recent candles summary (last 10)
    recent_candles = []
    for i in range(-10, 0):
        recent_candles.append({
            "time": ohlcv_entry["time"][i],
            "open": round(opens_e[i], 5),
            "high": round(highs_e[i], 5),
            "low": round(lows_e[i], 5),
            "close": round(closes_e[i], 5),
        })

    # Get recent trades for context
    conn = db.get_conn()
    recent_trades = conn.execute(
        "SELECT symbol, direction, profit, entry_price, exit_price FROM trades ORDER BY entry_time DESC LIMIT 10"
    ).fetchall()

    # Get learned rules
    conn_rules = conn.execute(
        "SELECT category, rule, confidence FROM rules WHERE confidence > 30 ORDER BY confidence DESC LIMIT 10"
    ).fetchall()

    # Account info
    account_result = mt5_data.daemon.call("connect")
    account_info = account_result.get("account", {}) if account_result.get("ok") else {}

    return {
        "ok": True,
        "symbol": symbol,
        "price": closes_e[-1],
        "entry_tf": entry_tf,
        "analysis_tf": analysis_tf,
        "indicators": {
            "rsi": round(current_rsi, 1),
            "macd_histogram": round(current_hist, 5),
            "macd_cross": macd_cross,
            "bb_position": bb_pos,
            "bb_upper": round(bb_data["upper"][-1], 5) if bb_data["upper"][-1] else None,
            "bb_lower": round(bb_data["lower"][-1], 5) if bb_data["lower"][-1] else None,
            "bb_middle": round(bb_data["middle"][-1], 5) if bb_data["middle"][-1] else None,
            "stoch_k": round(stoch["k"][-1], 1) if stoch["k"][-1] is not None else 50,
            "stoch_d": round(stoch["d"][-1], 1) if stoch["d"][-1] is not None else 50,
            "atr": round(current_atr, 5),
        },
        "trend": trend,
        "candle_patterns": patterns,
        "support_resistance": sr,
        "recent_candles": recent_candles,
        "recent_trades": [dict(t) for t in recent_trades] if recent_trades else [],
        "rules": [dict(r) for r in conn_rules] if conn_rules else [],
        "account": {
            "balance": account_info.get("balance", 0),
            "equity": account_info.get("equity", 0),
            "margin_free": account_info.get("margin_free", 0),
            "leverage": account_info.get("leverage", 0),
        },
    }


def build_ai_prompt(symbol: str, data: dict) -> str:
    """Build a market analysis prompt for the AI."""
    ind = data["indicators"]
    sr = data["support_resistance"]
    account = data["account"]

    # Format recent candles
    candles_str = ""
    for c in data["recent_candles"][-5:]:
        direction = "↑" if c["close"] > c["open"] else "↓" if c["close"] < c["open"] else "→"
        candles_str += f"  {direction} O:{c['open']} H:{c['high']} L:{c['low']} C:{c['close']}\n"

    # Format support/resistance
    sr_str = ""
    if sr.get("supports"):
        sr_str += "支撑位: " + ", ".join(f"{s['price']:.2f}(强度{s['strength']})" for s in sr["supports"][:3]) + "\n"
    if sr.get("resistances"):
        sr_str += "阻力位: " + ", ".join(f"{r['price']:.2f}(强度{r['strength']})" for r in sr["resistances"][:3]) + "\n"

    # Format recent trades
    trades_str = ""
    if data["recent_trades"]:
        wins = sum(1 for t in data["recent_trades"] if (t.get("profit") or 0) > 0)
        losses = sum(1 for t in data["recent_trades"] if (t.get("profit") or 0) < 0)
        trades_str = f"近期交易: {wins}胜{losses}负\n"

    # Format rules
    rules_str = ""
    if data["rules"]:
        rules_str = "已学习的交易规则:\n"
        for r in data["rules"][:5]:
            rules_str += f"  - [{r['category']}] {r['rule']} (可信度:{r['confidence']:.0f}%)\n"

    # Inject learned knowledge from reviewer agent
    from . import agents as _agents
    learned = _agents.get_learned_knowledge()
    knowledge_str = ""
    if learned:
        knowledge_str = f"""

## AI 复盘系统已学到的经验（必须参考）
{learned}
"""

    prompt = f"""你是一位专业的量化交易分析师。请根据以下市场数据，对 {symbol} 做出交易决策。

## 当前市场状态
- 品种: {symbol}
- 当前价格: {data['price']}
- {data['entry_tf']} 时间框架趋势: {data['trend']}
- 分析时间框架: {data['analysis_tf']}

## 技术指标
- RSI(14): {ind['rsi']}
- MACD 柱状图: {ind['macd_histogram']} (交叉: {ind['macd_cross']})
- 布林带位置: {ind['bb_position']} (上:{ind['bb_upper']} 中:{ind['bb_middle']} 下:{ind['bb_lower']})
- 随机指标: K={ind['stoch_k']} D={ind['stoch_d']}
- ATR: {ind['atr']}

## K线形态
{', '.join(data['candle_patterns']) if data['candle_patterns'] else '无明显形态'}

## 最近5根K线
{candles_str}

## 支撑阻力
{sr_str}

## 账户状态
- 余额: ${account['balance']:.2f}
- 净值: ${account['equity']:.2f}
- 可用保证金: ${account['margin_free']:.2f}
- 杠杆: 1:{account['leverage']}

{trades_str}
{rules_str}
{knowledge_str}

## 要求
请综合分析以上所有信息，给出你的交易决策。不要机械地套用指标阈值，而是像一个有经验的交易员一样，综合考虑：
1. 当前市场结构（趋势、震荡、突破等）
2. 多个指标的共振或背离
3. K线形态的含义
4. 支撑阻力的可靠性
5. 当前账户状态和风险

请用以下JSON格式回复（只回复JSON，不要其他文字）：
{{
  "direction": "BUY" 或 "SELL" 或 "FLAT",
  "confidence": 0-100的整数,
  "reasoning": "你的详细分析逻辑（中文，2-3句话）",
  "entry_price": 入场价,
  "stop_loss": 止损价,
  "take_profit": 止盈价,
  "lot_size_suggestion": 建议手数,
  "key_factors": ["关键因素1", "关键因素2"],
  "risk_note": "风险提示"
}}"""

    return prompt


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
            # New trade — record it with current timestamp
            entry_time = pos.get("time", 0)
            # MT5 time might be in seconds since epoch or milliseconds
            if entry_time > 1e12:
                entry_time = entry_time / 1000  # convert ms to seconds
            conn.execute(
                """INSERT INTO trades (ticket, symbol, direction, lot_size, entry_price, entry_time,
                   stop_loss, take_profit, context)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (ticket, pos["symbol"], "BUY" if pos["type"] == 0 else "SELL",
                 pos["volume"], pos["price_open"], entry_time,
                 pos.get("sl", 0), pos.get("tp", 0),
                 json.dumps({"magic": pos.get("magic"), "comment": pos.get("comment")}))
            )

        # Update live profit and MFE/MAE
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

            # Update live profit from MT5
            mt5_profit = pos.get("profit", 0)
            conn.execute("UPDATE trades SET profit=? WHERE id=?", (mt5_profit, trade["id"]))

    # Detect closed trades (were in trades table but no longer open)
    tracked = conn.execute("SELECT * FROM trades WHERE exit_time IS NULL").fetchall()
    for t in tracked:
        if t["ticket"] not in current_tickets:
            # Position closed — get final profit from MT5 deal history
            final_profit = 0
            try:
                deals = mt5_data.daemon.call("deals", ticket=t["ticket"])
                if deals.get("ok") and deals.get("deals"):
                    # Sum profit from all deals for this position
                    for deal in deals["deals"]:
                        final_profit += deal.get("profit", 0)
            except Exception:
                pass

            # If we couldn't get deal profit, use last known profit from trades table
            if final_profit == 0 and t["profit"]:
                final_profit = t["profit"]

            entry_time = t["entry_time"] or time.time()
            hold_duration = max(0, time.time() - entry_time)

            conn.execute(
                """UPDATE trades SET exit_time=?, exit_reason='sync_detected',
                   profit=?, hold_duration=?
                   WHERE id=?""",
                (time.time(), final_profit, hold_duration, t["id"])
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
