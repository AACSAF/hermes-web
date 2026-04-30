"""Review system — daily/weekly trade analysis and reporting."""

import json
import time
import logging
from datetime import datetime, timedelta
from . import database as db

logger = logging.getLogger(__name__)


def generate_daily_review(date_str: str = None) -> dict:
    """Generate a daily review report.

    Args:
        date_str: YYYY-MM-DD format, defaults to today
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Parse date range
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start_ts = dt.timestamp()
    end_ts = (dt + timedelta(days=1)).timestamp()

    conn = db.get_conn()

    # Get trades for this day
    trades = conn.execute(
        """SELECT * FROM trades WHERE entry_time >= ? AND entry_time < ?
           ORDER BY entry_time""",
        (start_ts, end_ts)
    ).fetchall()

    trades = [dict(t) for t in trades]
    count = len(trades)

    if count == 0:
        return {
            "ok": True,
            "date": date_str,
            "trades_count": 0,
            "summary": "今日无交易",
            "action_items": [],
        }

    wins = sum(1 for t in trades if (t.get("profit") or 0) > 0)
    losses = sum(1 for t in trades if (t.get("profit") or 0) < 0)
    breakevens = count - wins - losses
    total_pnl = sum(t.get("profit") or 0 for t in trades)
    win_rate = (wins / count * 100) if count > 0 else 0

    # Best and worst trades
    best = max(trades, key=lambda t: t.get("profit") or 0)
    worst = min(trades, key=lambda t: t.get("profit") or 0)

    # Max drawdown (running peak-to-trough)
    running = 0
    peak = 0
    max_dd = 0
    for t in sorted(trades, key=lambda x: x.get("entry_time", 0)):
        running += t.get("profit") or 0
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    # Average metrics
    profitable = [t for t in trades if (t.get("profit") or 0) > 0]
    losing = [t for t in trades if (t.get("profit") or 0) < 0]
    avg_win = sum(t["profit"] for t in profitable) / len(profitable) if profitable else 0
    avg_loss = sum(t["profit"] for t in losing) / len(losing) if losing else 0
    profit_factor = abs(sum(t["profit"] for t in profitable) / sum(t["profit"] for t in losing)) if losing and sum(t["profit"] for t in losing) != 0 else float('inf')

    # Average hold duration
    durations = [t.get("hold_duration") or 0 for t in trades if t.get("hold_duration")]
    avg_duration = sum(durations) / len(durations) if durations else 0

    # Symbol breakdown
    symbols = {}
    for t in trades:
        sym = t.get("symbol", "?")
        if sym not in symbols:
            symbols[sym] = {"count": 0, "pnl": 0, "wins": 0}
        symbols[sym]["count"] += 1
        symbols[sym]["pnl"] += t.get("profit") or 0
        if (t.get("profit") or 0) > 0:
            symbols[sym]["wins"] += 1

    # Direction breakdown
    buy_trades = [t for t in trades if t.get("direction") == "BUY"]
    sell_trades = [t for t in trades if t.get("direction") == "SELL"]
    buy_pnl = sum(t.get("profit") or 0 for t in buy_trades)
    sell_pnl = sum(t.get("profit") or 0 for t in sell_trades)

    # Generate action items
    action_items = []

    if win_rate < 40:
        action_items.append("胜率低于40%，考虑提高信号信心阈值或减少交易频率")
    if profit_factor < 1.0:
        action_items.append("盈亏比不足1.0，考虑扩大止盈或缩小止损")
    if max_dd > total_pnl * 0.5 and max_dd > 10:
        action_items.append(f"最大回撤${max_dd:.2f}过高，考虑降低单笔风险比例")
    if abs(avg_loss) > avg_win * 2 and avg_win > 0:
        action_items.append("平均亏损远大于平均盈利，检查止损设置是否过宽")

    for sym, data in symbols.items():
        if data["count"] >= 3 and data["wins"] / data["count"] < 0.3:
            action_items.append(f"{sym}胜率过低({data['wins']}/{data['count']})，考虑暂停该品种交易")

    if count > 20:
        action_items.append("今日交易次数过多，检查是否过度交易")

    # Build summary
    summary_parts = [
        f"📊 {date_str} 交易复盘",
        f"",
        f"交易 {count} 笔 | 胜 {wins} / 负 {losses} / 平 {breakevens}",
        f"胜率 {win_rate:.1f}% | 盈亏比 {profit_factor:.2f}",
        f"总盈亏: ${total_pnl:+.2f}",
        f"最大回撤: ${max_dd:.2f}",
        f"",
        f"做多 {len(buy_trades)} 笔 (${buy_pnl:+.2f}) | 做空 {len(sell_trades)} 笔 (${sell_pnl:+.2f})",
        f"平均持仓 {avg_duration/60:.0f} 分钟",
    ]

    if symbols:
        summary_parts.append("")
        summary_parts.append("品种明细:")
        for sym, data in sorted(symbols.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = data["wins"] / data["count"] * 100 if data["count"] > 0 else 0
            summary_parts.append(f"  {sym}: {data['count']}笔 胜率{wr:.0f}% 盈亏${data['pnl']:+.2f}")

    summary = "\n".join(summary_parts)

    # Store review
    conn.execute(
        """INSERT OR REPLACE INTO reviews (date, trades_count, wins, losses, total_pnl,
           max_drawdown, best_trade, worst_trade, summary, action_items, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (date_str, count, wins, losses, total_pnl, max_dd,
         json.dumps({"symbol": best.get("symbol"), "profit": best.get("profit"), "direction": best.get("direction")}),
         json.dumps({"symbol": worst.get("symbol"), "profit": worst.get("profit"), "direction": worst.get("direction")}),
         summary, json.dumps(action_items, ensure_ascii=False), time.time())
    )
    conn.commit()

    return {
        "ok": True,
        "date": date_str,
        "trades_count": count,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "summary": summary,
        "action_items": action_items,
        "symbols": symbols,
    }


def get_review(date_str: str) -> dict:
    """Get a stored review by date."""
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM reviews WHERE date=?", (date_str,)).fetchone()
    if not row:
        return {"ok": False, "error": f"没有 {date_str} 的复盘记录"}
    return {"ok": True, **dict(row)}


def get_weekly_summary() -> dict:
    """Aggregate last 7 days of reviews."""
    conn = db.get_conn()
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT * FROM reviews WHERE date >= ? ORDER BY date", (seven_days_ago,)
    ).fetchall()

    if not rows:
        return {"ok": True, "message": "过去7天没有交易记录"}

    reviews = [dict(r) for r in rows]
    total_trades = sum(r["trades_count"] for r in reviews)
    total_wins = sum(r["wins"] for r in reviews)
    total_losses = sum(r["losses"] for r in reviews)
    total_pnl = sum(r["total_pnl"] for r in reviews)
    max_dd = max(r["max_drawdown"] for r in reviews)
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    # Collect all action items
    all_actions = []
    for r in reviews:
        if r["action_items"]:
            try:
                all_actions.extend(json.loads(r["action_items"]))
            except json.JSONDecodeError:
                pass

    return {
        "ok": True,
        "period": f"{seven_days_ago} ~ {datetime.now().strftime('%Y-%m-%d')}",
        "trading_days": len(reviews),
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "action_items": list(set(all_actions))[:10],
    }
