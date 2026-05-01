"""Multi-Agent Trading System — committee-based decision making with self-evolution."""

import json
import time
import logging
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from . import database as db

logger = logging.getLogger(__name__)

# ── Agent Role Definitions ─────────────────────────────────────────

AGENT_ROLES = {
    "bull_analyst": {
        "name": "🐂 多头分析师",
        "description": "专注寻找买入机会，分析支撑位和看涨信号",
        "frequency": "high",       # high / medium / low
        "default_model": "mimo",
        "system_prompt": """你是一位专业的多头交易分析师。你的职责是：
1. 专注于寻找买入机会和看涨信号
2. 分析支撑位、反转形态、底部结构
3. 从多头角度解读技术指标
4. 即使市场偏空，也要找到潜在的反弹机会

注意：你要客观分析，不是盲目看涨。如果确实没有买入机会，可以说"无买入机会"。
用中文回复。""",
    },
    "bear_analyst": {
        "name": "🐻 空头分析师",
        "description": "专注寻找卖出/做空机会，分析压力位和看跌信号",
        "frequency": "high",
        "default_model": "mimo",
        "system_prompt": """你是一位专业的空头交易分析师。你的职责是：
1. 专注于寻找卖出/做空机会和看跌信号
2. 分析压力位、顶部结构、下跌动能
3. 从空头角度解读技术指标
4. 即使市场偏多，也要找到潜在的回调风险

注意：你要客观分析，不是盲目看跌。如果确实没有做空机会，可以说"无做空机会"。
用中文回复。""",
    },
    "risk_officer": {
        "name": "⚖️ 风控官",
        "description": "评估风险，计算仓位，设定止损止盈",
        "frequency": "high",
        "default_model": "mimo",
        "system_prompt": """你是一位严格的风险控制官。你的职责是：
1. 评估每笔交易的风险/收益比
2. 计算合理的仓位大小
3. 设定止损和止盈价位
4. 检查账户状态，防止过度交易
5. 如果风险太高，直接否决交易

你的原则：宁可错过机会，也不能承受过大风险。盈亏比低于1.5:1的交易一律否决。
用中文回复。""",
    },
    "macro_observer": {
        "name": "🌍 宏观观察员",
        "description": "关注宏观经济和市场情绪，提供大局观",
        "frequency": "low",        # 低频 — 不是每次都需要
        "default_model": "deepseek",
        "system_prompt": """你是一位宏观经济分析师。你的职责是：
1. 分析当前市场的大趋势和宏观环境
2. 关注重要经济数据、央行政策、地缘政治
3. 评估市场情绪（恐惧/贪婪指数）
4. 提供中长期趋势判断

你的判断会影响整体仓位策略。用中文回复。""",
    },
    "pattern_hunter": {
        "name": "🔍 形态猎手",
        "description": "专注K线形态、图表结构、量价关系",
        "frequency": "medium",     # 中频
        "default_model": "mimo",
        "system_prompt": """你是一位技术形态专家。你的职责是：
1. 识别K线形态（锤子线、吞没、十字星等）
2. 分析图表结构（头肩顶底、三角形、旗形等）
3. 量价关系分析
4. 支撑阻力位的可靠性评估

你只关注形态和结构，不预测方向。用中文回复。""",
    },
    "arbiter": {
        "name": "🧑‍⚖️ 仲裁官",
        "description": "综合所有分析师意见，做出最终决策",
        "frequency": "high",
        "default_model": "deepseek",   # 仲裁用更强的模型
        "system_prompt": """你是交易委员会的最终仲裁者。你会收到多位分析师的意见，你的职责是：
1. 综合考虑所有分析师的观点
2. 评估各方论据的说服力
3. 做出最终的交易决策：BUY / SELL / HOLD
4. 如果各方分歧太大或信号不明确，选择HOLD

决策原则：
- 至少2位分析师达成一致才考虑开仓
- 风控官否决的交易一律不执行
- 置信度低于60的信号不执行

用以下JSON格式回复（只回复JSON）：
{
  "direction": "BUY" 或 "SELL" 或 "HOLD",
  "confidence": 0-100的整数,
  "entry_price": 入场价,
  "stop_loss": 止损价,
  "take_profit": 止盈价,
  "lot_size_suggestion": 建议手数,
  "reasoning": "综合分析逻辑（2-3句话）",
  "dissenting_view": "反对意见（如有）",
  "risk_note": "风险提示"
}""",
    },
    "reviewer": {
        "name": "📊 复盘师",
        "description": "回顾历史交易，总结经验教训，优化策略",
        "frequency": "low",        # 低频 — 每天或每几天一次
        "default_model": "deepseek",
        "system_prompt": """你是一位交易复盘分析师。你的职责是：
1. 分析过去一段时间的交易记录
2. 找出成功和失败的模式
3. 识别需要改进的地方
4. 提出具体可执行的策略优化建议
5. 发现新的交易规则

输出格式（JSON）：
{
  "overall_assessment": "整体评价",
  "win_patterns": ["成功的模式1", ...],
  "loss_patterns": ["失败的模式1", ...],
  "new_rules": [
    {"category": "entry|exit|risk|market|time", "rule": "具体规则"}
  ],
  "param_adjustments": [
    {"param_name": "参数名", "current_value": 10, "suggested_value": 12, "reason": "原因"}
  ],
  "priority_actions": ["最重要的改进项1", "最重要的改进项2"]
}""",
    },
}

# Model provider configs
MODEL_PROVIDERS = {
    "mimo": {
        "name": "MiMo",
        "base_url": "https://api.xiaomi.com/v1",
        "default_model": "MiMo-v2.5-pro",
        "description": "小米 MiMo 模型",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "description": "DeepSeek V3",
    },
    "minimax": {
        "name": "MiniMax",
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "MiniMax-Text-01",
        "description": "MiniMax 模型",
    },
}


def _call_llm(base_url: str, api_key: str, model: str, system_prompt: str, user_prompt: str, timeout: int = 60) -> str:
    """Call an LLM via OpenAI-compatible API."""
    import urllib.request
    import urllib.error

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }).encode("utf-8")

    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("LLM call failed: %s %s", url, e)
        return json.dumps({"error": str(e)})


def get_agent_config(agent_id: str) -> dict:
    """Get agent config from DB, with defaults from AGENT_ROLES."""
    role = AGENT_ROLES.get(agent_id, {})
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM agent_configs WHERE agent_id=?", (agent_id,)).fetchone()

    if row:
        return {
            "agent_id": agent_id,
            "name": row["name"] or role.get("name", agent_id),
            "description": role.get("description", ""),
            "enabled": bool(row["enabled"]),
            "provider": row["provider"] or role.get("default_model", "mimo"),
            "api_key": row["api_key"] or "",
            "model": row["model"] or "",
            "frequency": row["frequency"] or role.get("frequency", "medium"),
            "system_prompt": row["system_prompt"] or role.get("system_prompt", ""),
            "weight": row["weight"] if row["weight"] is not None else 1.0,
        }
    else:
        return {
            "agent_id": agent_id,
            "name": role.get("name", agent_id),
            "description": role.get("description", ""),
            "enabled": True,
            "provider": role.get("default_model", "mimo"),
            "api_key": "",
            "model": "",
            "frequency": role.get("frequency", "medium"),
            "system_prompt": role.get("system_prompt", ""),
            "weight": 1.0,
        }


def get_all_agent_configs() -> list:
    """Get all agent configs."""
    configs = []
    for agent_id in AGENT_ROLES:
        configs.append(get_agent_config(agent_id))
    return configs


def update_agent_config(agent_id: str, updates: dict) -> dict:
    """Update an agent's configuration."""
    conn = db.get_conn()
    existing = conn.execute("SELECT * FROM agent_configs WHERE agent_id=?", (agent_id,)).fetchone()

    if existing:
        sets = []
        vals = []
        for key in ("name", "enabled", "provider", "api_key", "model", "frequency", "system_prompt", "weight"):
            if key in updates:
                sets.append(f"{key}=?")
                vals.append(updates[key])
        if sets:
            sets.append("updated_at=?")
            vals.append(time.time())
            vals.append(agent_id)
            conn.execute(f"UPDATE agent_configs SET {', '.join(sets)} WHERE agent_id=?", vals)
    else:
        role = AGENT_ROLES.get(agent_id, {})
        conn.execute(
            """INSERT INTO agent_configs (agent_id, name, enabled, provider, api_key, model, frequency, system_prompt, weight, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (agent_id,
             updates.get("name", role.get("name", agent_id)),
             updates.get("enabled", 1),
             updates.get("provider", role.get("default_model", "mimo")),
             updates.get("api_key", ""),
             updates.get("model", ""),
             updates.get("frequency", role.get("frequency", "medium")),
             updates.get("system_prompt", role.get("system_prompt", "")),
             updates.get("weight", 1.0),
             time.time())
        )
    conn.commit()
    return {"ok": True}


def _get_api_key_for_provider(provider: str) -> str:
    """Resolve API key: first check DB global keys, then env vars."""
    conn = db.get_conn()
    row = conn.execute("SELECT value FROM kv_store WHERE key=?", (f"api_key_{provider}",)).fetchone()
    if row and row["value"]:
        return row["value"]

    env_map = {
        "mimo": "XIAOMI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "minimax": "MINIMAX_API_KEY",
    }
    env_var = env_map.get(provider, "")
    return os.environ.get(env_var, "")


def _resolve_agent_call(agent_id: str, user_prompt: str) -> dict:
    """Make a single agent call. Returns {agent_id, name, response, latency_ms, error}."""
    config = get_agent_config(agent_id)
    t0 = time.time()

    if not config["enabled"]:
        return {"agent_id": agent_id, "name": config["name"], "response": None, "latency_ms": 0, "error": "disabled"}

    provider = config["provider"]
    provider_info = MODEL_PROVIDERS.get(provider, MODEL_PROVIDERS["mimo"])

    # Resolve API key: agent-specific > provider global > env
    api_key = config["api_key"] or _get_api_key_for_provider(provider)
    if not api_key:
        return {"agent_id": agent_id, "name": config["name"], "response": None,
                "latency_ms": 0, "error": f"未配置 {provider_info['name']} API Key"}

    base_url = provider_info["base_url"]
    model = config["model"] or provider_info["default_model"]
    system_prompt = config["system_prompt"]

    try:
        response = _call_llm(base_url, api_key, model, system_prompt, user_prompt, timeout=90)
        latency = int((time.time() - t0) * 1000)
        return {"agent_id": agent_id, "name": config["name"], "response": response,
                "latency_ms": latency, "error": None, "weight": config["weight"]}
    except Exception as e:
        latency = int((time.time() - t0) * 1000)
        return {"agent_id": agent_id, "name": config["name"], "response": None,
                "latency_ms": latency, "error": str(e)}


def run_committee(market_data: dict, symbol: str = "XAUUSD") -> dict:
    """Run the full trading committee: parallel analysts → arbiter.

    Flow:
    1. High-frequency agents (bull, bear, risk, pattern) run in parallel
    2. Low-frequency agents (macro) run if conditions warrant
    3. Arbiter synthesizes all opinions into final decision
    """
    from . import engine

    prompt = engine.build_ai_prompt(symbol, market_data)

    # Phase 1: Run analyst agents in parallel
    high_freq = ["bull_analyst", "bear_analyst", "risk_officer", "pattern_hunter"]
    medium_freq = ["macro_observer"]  # only on certain conditions

    # Decide which agents to include
    agents_to_run = list(high_freq)

    # Include macro observer if enough trades or significant market events
    macro_config = get_agent_config("macro_observer")
    if macro_config["enabled"]:
        agents_to_run.append("macro_observer")

    # Run all analysts in parallel
    analyst_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_resolve_agent_call, agent_id, prompt): agent_id
            for agent_id in agents_to_run
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                analyst_results.append(result)
                # Log call stats
                _log_agent_call(result)
            except Exception as e:
                agent_id = futures[future]
                analyst_results.append({
                    "agent_id": agent_id, "name": AGENT_ROLES.get(agent_id, {}).get("name", agent_id),
                    "response": None, "latency_ms": 0, "error": str(e)
                })

    # Phase 2: Build arbiter prompt from analyst opinions
    opinions = []
    for r in analyst_results:
        if r.get("response") and not r.get("error"):
            opinions.append(f"### {r['name']}:\n{r['response']}")
        elif r.get("error"):
            opinions.append(f"### {r['name']}: ❌ 错误 — {r['error']}")

    arbiter_prompt = f"""以下是交易委员会各位分析师对 {symbol} 的分析意见：

{chr(10).join(opinions)}

---

请综合以上所有意见，做出最终交易决策。"""

    # Phase 3: Arbiter makes final decision
    arbiter_config = get_agent_config("arbiter")
    arbiter_result = _resolve_agent_call("arbiter", arbiter_prompt)
    _log_agent_call(arbiter_result)

    # Parse arbiter's decision
    decision = {"direction": "HOLD", "confidence": 0, "reasoning": "解析失败"}
    if arbiter_result.get("response") and not arbiter_result.get("error"):
        try:
            json_match = re.search(r'\{[^{}]*\}', arbiter_result["response"], re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
        except Exception:
            decision = {"direction": "HOLD", "confidence": 0, "reasoning": arbiter_result["response"][:200]}

    return {
        "ok": True,
        "symbol": symbol,
        "mode": "committee",
        "decision": decision,
        "analyst_opinions": [
            {"agent_id": r["agent_id"], "name": r["name"],
             "response": r.get("response", ""), "error": r.get("error"),
             "latency_ms": r.get("latency_ms", 0)}
            for r in analyst_results
        ],
        "arbiter": {
            "response": arbiter_result.get("response", ""),
            "error": arbiter_result.get("error"),
            "latency_ms": arbiter_result.get("latency_ms", 0),
        },
        "total_latency_ms": sum(r.get("latency_ms", 0) for r in analyst_results) + arbiter_result.get("latency_ms", 0),
    }


def run_single_agent(agent_id: str, prompt: str) -> dict:
    """Run a single agent with a given prompt (for manual testing)."""
    result = _resolve_agent_call(agent_id, prompt)
    _log_agent_call(result)
    return {"ok": not bool(result.get("error")), **result}


# ── Call Logging ────────────────────────────────────────────────────

def _log_agent_call(result: dict):
    """Log agent call stats to DB for performance tracking."""
    conn = db.get_conn()
    conn.execute(
        """INSERT INTO agent_call_log (agent_id, latency_ms, has_error, created_at)
           VALUES (?,?,?,?)""",
        (result.get("agent_id", ""),
         result.get("latency_ms", 0),
         1 if result.get("error") else 0,
         time.time())
    )
    conn.commit()


def get_agent_stats(agent_id: str = None, days: int = 7) -> list:
    """Get agent call statistics."""
    conn = db.get_conn()
    cutoff = time.time() - days * 86400

    if agent_id:
        rows = conn.execute(
            """SELECT agent_id,
                      COUNT(*) as total_calls,
                      AVG(latency_ms) as avg_latency,
                      SUM(CASE WHEN has_error=1 THEN 1 ELSE 0 END) as errors
               FROM agent_call_log
               WHERE agent_id=? AND created_at>=?
               GROUP BY agent_id""",
            (agent_id, cutoff)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT agent_id,
                      COUNT(*) as total_calls,
                      AVG(latency_ms) as avg_latency,
                      SUM(CASE WHEN has_error=1 THEN 1 ELSE 0 END) as errors
               FROM agent_call_log
               WHERE created_at>=?
               GROUP BY agent_id
               ORDER BY total_calls DESC""",
            (cutoff,)
        ).fetchall()
    return [dict(r) for r in rows]
