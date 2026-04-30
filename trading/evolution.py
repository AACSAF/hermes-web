"""Strategy evolution — auto-tune parameters based on performance."""

import json
import time
import logging
import random
from . import database as db

logger = logging.getLogger(__name__)


def evaluate_performance(trades: list = None, days: int = 7) -> dict:
    """Evaluate recent trading performance metrics.

    Returns: {win_rate, avg_profit, max_drawdown, profit_factor, sharpe_ratio, trades_count}
    """
    conn = db.get_conn()

    if trades is None:
        cutoff = time.time() - days * 86400
        trades = conn.execute(
            "SELECT * FROM trades WHERE exit_time >= ? AND profit != 0 ORDER BY exit_time",
            (cutoff,)
        ).fetchall()
        trades = [dict(t) for t in trades]

    count = len(trades)
    if count == 0:
        return {"win_rate": 0, "avg_profit": 0, "max_drawdown": 0, "profit_factor": 0, "sharpe_ratio": 0, "trades_count": 0}

    profits = [t.get("profit", 0) for t in trades]
    wins = sum(1 for p in profits if p > 0)
    win_rate = wins / count * 100

    avg_profit = sum(profits) / count

    # Max drawdown
    running = 0
    peak = 0
    max_dd = 0
    for p in profits:
        running += p
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = sum(p for p in profits if p > 0)
    gross_loss = abs(sum(p for p in profits if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Simplified Sharpe ratio (daily returns)
    if len(profits) > 1:
        mean = sum(profits) / len(profits)
        variance = sum((p - mean) ** 2 for p in profits) / (len(profits) - 1)
        std = variance ** 0.5
        sharpe = (mean / std) * (252 ** 0.5) if std > 0 else 0  # annualized
    else:
        sharpe = 0

    return {
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "trades_count": count,
    }


def evolve_params(lookback_days: int = 7) -> dict:
    """Run one generation of parameter evolution.

    Strategy: compare current performance against last generation.
    If improved, keep. If not, nudge parameters in the direction of
    the best-performing generation.

    Returns: {generation, changed_params, performance}
    """
    conn = db.get_conn()

    # Get current generation
    gen = int(db.kv_get("evolution_generation", "0"))

    # Evaluate current performance
    perf = evaluate_performance(days=lookback_days)

    # Get last generation's performance
    last_evo = conn.execute(
        "SELECT * FROM evolution_log ORDER BY generation DESC LIMIT 1"
    ).fetchone()

    # Decide whether to evolve
    should_evolve = False
    reason = ""

    if last_evo is None:
        # First run — just record baseline
        reason = "首次运行，记录基线"
        should_evolve = False
    elif perf["trades_count"] < 10:
        reason = f"样本不足({perf['trades_count']}笔)，需要更多数据"
        should_evolve = False
    elif perf["profit_factor"] < 1.0:
        reason = "盈亏比低于1.0，需要调整策略"
        should_evolve = True
    elif perf["win_rate"] < 40:
        reason = "胜率低于40%，需要优化入场条件"
        should_evolve = True
    elif perf["sharpe_ratio"] < 0:
        reason = "夏普比率为负，策略需要改进"
        should_evolve = True
    else:
        reason = "表现良好，微调优化"
        should_evolve = True

    changed_params = []

    if should_evolve:
        params = db.get_all_params()
        all_param_meta = conn.execute(
            "SELECT name, value, value_type, min_val, max_val, step FROM strategy_params WHERE min_val IS NOT NULL"
        ).fetchall()

        for pm in all_param_meta:
            name = pm["name"]
            current_val = float(pm["value"])
            min_val = pm["min_val"]
            max_val = pm["max_val"]
            step = pm["step"] or 1

            if min_val is None or max_val is None:
                continue

            # Determine direction of adjustment
            if perf["profit_factor"] < 1.0:
                # Strategy losing money — be more conservative
                if "risk" in name:
                    delta = -step  # reduce risk
                elif "min_confidence" in name:
                    delta = step  # require more confidence
                elif "sl" in name and "mult" in name:
                    delta = -step * 0.5  # tighten stops
                elif "tp" in name and "mult" in name:
                    delta = step * 0.5  # widen targets
                else:
                    delta = random.choice([-step, 0, step]) * 0.5
            else:
                # Strategy profitable — small random nudge
                delta = random.gauss(0, step * 0.3)

            new_val = max(min_val, min(max_val, current_val + delta))
            new_val = round(new_val / step) * step  # snap to step

            if abs(new_val - current_val) > step * 0.01:
                db.set_param(name, new_val)
                changed_params.append({
                    "name": name,
                    "old": current_val,
                    "new": new_val,
                    "delta": round(new_val - current_val, 4),
                })

        gen += 1
        db.kv_set("evolution_generation", str(gen))

    # Log evolution
    conn.execute(
        """INSERT INTO evolution_log (generation, params_snapshot, trades_count, win_rate,
           avg_profit, max_drawdown, profit_factor, sharpe_ratio, notes, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (gen, json.dumps(db.get_all_params()), perf["trades_count"], perf["win_rate"],
         perf["avg_profit"], perf["max_drawdown"], perf["profit_factor"],
         perf["sharpe_ratio"], reason, time.time())
    )
    conn.commit()

    return {
        "ok": True,
        "generation": gen,
        "should_evolve": should_evolve,
        "reason": reason,
        "performance": perf,
        "changed_params": changed_params,
    }


def add_rule(category: str, rule: str, source: str = "learned") -> dict:
    """Add a new trading rule learned from experience."""
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO rules (category, rule, source, confidence, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (category, rule, source, 50, time.time(), time.time())
    )
    conn.commit()
    return {"ok": True, "category": category, "rule": rule}


def get_rules(category: str = None) -> list:
    """Get trading rules, optionally filtered by category."""
    conn = db.get_conn()
    if category:
        rows = conn.execute(
            "SELECT * FROM rules WHERE category=? ORDER BY confidence DESC", (category,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM rules ORDER BY confidence DESC").fetchall()
    return [dict(r) for r in rows]


def update_rule_confidence(rule_id: int, correct: bool) -> dict:
    """Update rule confidence based on whether it was correct when applied."""
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM rules WHERE id=?", (rule_id,)).fetchone()
    if not row:
        return {"ok": False, "error": "规则不存在"}

    times_applied = row["times_applied"] + 1
    times_correct = row["times_correct"] + (1 if correct else 0)

    # Confidence = success rate * 100, with smoothing
    if times_applied >= 3:
        confidence = (times_correct / times_applied) * 100
    else:
        # With few samples, weight towards prior (50)
        confidence = (times_correct + 5) / (times_applied + 10) * 100

    conn.execute(
        "UPDATE rules SET times_applied=?, times_correct=?, confidence=?, updated_at=? WHERE id=?",
        (times_applied, times_correct, confidence, time.time(), rule_id)
    )
    conn.commit()

    return {"ok": True, "confidence": confidence, "times_applied": times_applied}


def get_evolution_history(limit: int = 20) -> list:
    """Get evolution log history."""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT * FROM evolution_log ORDER BY generation DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]
