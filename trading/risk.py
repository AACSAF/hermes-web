"""Risk management module. Enforces position sizing, drawdown limits, and cooldowns."""

import time
from . import database as db


def calculate_lot_size(equity: float, risk_pct: float, sl_pips: float, pip_value: float = 10.0) -> float:
    """Calculate position size based on account risk.

    Args:
        equity: Account equity
        risk_pct: Risk per trade as % of equity (e.g. 1.0 = 1%)
        sl_pips: Stop loss distance in pips
        pip_value: Value per pip per standard lot (default 10 for forex, varies by instrument)

    Returns:
        Recommended lot size (rounded to 0.01)
    """
    if sl_pips <= 0 or equity <= 0:
        return 0.0

    risk_amount = equity * (risk_pct / 100.0)
    lots = risk_amount / (sl_pips * pip_value)

    # Round down to nearest 0.01
    lots = max(0.01, round(lots * 100) / 100)
    return lots


def get_pip_value(symbol: str) -> float:
    """Get approximate pip value per standard lot for common instruments."""
    symbol = symbol.upper()
    # Metals
    if "XAU" in symbol:
        return 1.0  # $1 per 0.01 move per 0.01 lot, but for 1 lot = $100 per $1 move
    if "XAG" in symbol:
        return 50.0
    # JPY pairs
    if "JPY" in symbol:
        return 6.67  # approximate
    # Crypto
    if "BTC" in symbol:
        return 1.0
    # Default forex
    return 10.0


def get_pip_size(symbol: str) -> float:
    """Get pip size (price movement per pip) for the instrument."""
    symbol = symbol.upper()
    if "XAU" in symbol:
        return 0.01
    if "JPY" in symbol:
        return 0.01
    if "BTC" in symbol:
        return 1.0
    return 0.0001


def check_trade_allowed(symbol: str, direction: str, equity: float, current_positions: list) -> dict:
    """Pre-trade risk check. Returns {allowed: bool, reason: str}."""
    params = db.get_all_params()
    conn = db.get_conn()

    # 1. Max positions check
    max_pos = params.get("max_positions", 5)
    if len(current_positions) >= max_pos:
        return {"allowed": False, "reason": f"已达最大持仓数 {max_pos}"}

    # 2. Same symbol same direction check
    for pos in current_positions:
        if pos.get("symbol") == symbol:
            pos_type = pos.get("type", -1)
            if (direction == "BUY" and pos_type == 0) or (direction == "SELL" and pos_type == 1):
                return {"allowed": False, "reason": f"{symbol} 已有同方向持仓"}

    # 3. Daily loss check
    max_daily_loss = params.get("max_daily_loss_pct", 3.0)
    today_start = int(time.time()) - (int(time.time()) % 86400)
    row = conn.execute(
        "SELECT COALESCE(SUM(profit), 0) as daily_pnl FROM trades WHERE exit_time >= ?",
        (today_start,)
    ).fetchone()
    daily_pnl = row["daily_pnl"] if row else 0
    if equity > 0 and (daily_pnl / equity * 100) < -max_daily_loss:
        return {"allowed": False, "reason": f"今日亏损已达 {abs(daily_pnl/equity*100):.1f}%，超过限制 {max_daily_loss}%"}

    # 4. Consecutive loss cooldown check
    cooldown_count = params.get("cooldown_loss_count", 3)
    cooldown_minutes = params.get("cooldown_minutes", 120)

    recent_trades = conn.execute(
        "SELECT profit FROM trades ORDER BY exit_time DESC LIMIT ?",
        (cooldown_count,)
    ).fetchall()

    if len(recent_trades) >= cooldown_count:
        all_losses = all(t["profit"] < 0 for t in recent_trades)
        if all_losses:
            last_trade_time = conn.execute(
                "SELECT exit_time FROM trades ORDER BY exit_time DESC LIMIT 1"
            ).fetchone()
            if last_trade_time:
                elapsed = time.time() - last_trade_time["exit_time"]
                remaining = cooldown_minutes * 60 - elapsed
                if remaining > 0:
                    return {"allowed": False, "reason": f"连续亏损 {cooldown_count} 次，冷却中，还需等待 {remaining/60:.0f} 分钟"}

    # 5. Margin level check (if available from MT5)
    # This is checked at the MT5 level

    return {"allowed": True, "reason": "通过风控检查"}


def calculate_sl_tp(price: float, direction: str, atr_value: float, params: dict = None) -> dict:
    """Calculate stop loss and take profit based on ATR.

    Returns: {sl: float, tp: float, sl_pips: float}
    """
    if params is None:
        params = db.get_all_params()

    sl_mult = params.get("sl_atr_mult", 1.5)
    tp_mult = params.get("tp_atr_mult", 2.5)

    sl_distance = atr_value * sl_mult
    tp_distance = atr_value * tp_mult

    if direction == "BUY":
        sl = price - sl_distance
        tp = price + tp_distance
    else:
        sl = price + sl_distance
        tp = price - tp_distance

    return {
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "sl_distance": round(sl_distance, 5),
        "tp_distance": round(tp_distance, 5),
    }


def calculate_trailing_stop(entry_price: float, current_price: float, direction: str,
                            atr_value: float, params: dict = None) -> dict:
    """Calculate trailing stop level.

    Returns: {new_sl: float or None, should_trail: bool}
    """
    if params is None:
        params = db.get_all_params()

    trail_atr = params.get("trailing_stop_atr", 1.0)
    trail_distance = atr_value * trail_atr

    if direction == "BUY":
        # Only trail upward
        new_sl = current_price - trail_distance
        return {"new_sl": round(new_sl, 5), "should_trail": new_sl > entry_price}
    else:
        new_sl = current_price + trail_distance
        return {"new_sl": round(new_sl, 5), "should_trail": new_sl < entry_price}
