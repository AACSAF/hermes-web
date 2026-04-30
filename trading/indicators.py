"""Technical analysis indicators. Pure numpy/pandas-free for minimal deps."""

import math
from typing import Optional


def sma(closes: list[float], period: int) -> list[Optional[float]]:
    """Simple Moving Average."""
    result = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        result[i] = sum(closes[i - period + 1:i + 1]) / period
    return result


def ema(closes: list[float], period: int) -> list[Optional[float]]:
    """Exponential Moving Average."""
    if len(closes) < period:
        return [None] * len(closes)
    result = [None] * len(closes)
    # Seed with SMA
    result[period - 1] = sum(closes[:period]) / period
    k = 2.0 / (period + 1)
    for i in range(period, len(closes)):
        result[i] = closes[i] * k + result[i - 1] * (1 - k)
    return result


def rsi(closes: list[float], period: int = 14) -> list[Optional[float]]:
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return [None] * len(closes)

    result = [None] * len(closes)
    gains = []
    losses = []

    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    # First average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    # Subsequent values (Wilder's smoothing)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return result


def macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD (line, signal, histogram)."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    macd_line = [None] * len(closes)
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    # Signal line = EMA of MACD line
    macd_values = [v for v in macd_line if v is not None]
    sig = ema(macd_values, signal) if len(macd_values) >= signal else [None] * len(macd_values)

    # Map back
    signal_line = [None] * len(closes)
    histogram = [None] * len(closes)
    j = 0
    for i in range(len(closes)):
        if macd_line[i] is not None:
            if j < len(sig):
                signal_line[i] = sig[j]
                if sig[j] is not None:
                    histogram[i] = macd_line[i] - sig[j]
            j += 1

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def bollinger_bands(closes: list[float], period: int = 20, std_dev: float = 2.0) -> dict:
    """Bollinger Bands (upper, middle, lower)."""
    middle = sma(closes, period)
    upper = [None] * len(closes)
    lower = [None] * len(closes)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        avg = middle[i]
        variance = sum((x - avg) ** 2 for x in window) / period
        sd = math.sqrt(variance)
        upper[i] = avg + std_dev * sd
        lower[i] = avg - std_dev * sd

    return {"upper": upper, "middle": middle, "lower": lower}


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[Optional[float]]:
    """Average True Range."""
    if len(closes) < 2:
        return [None] * len(closes)

    tr = [None] * len(closes)
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    result = [None] * len(closes)
    if len(tr) < period:
        return result

    # First ATR = simple average
    valid_tr = [v for v in tr[:period] if v is not None]
    if valid_tr:
        result[period - 1] = sum(valid_tr) / period
        # Wilder's smoothing
        for i in range(period, len(closes)):
            if tr[i] is not None and result[i - 1] is not None:
                result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


def stochastic(highs: list[float], lows: list[float], closes: list[float],
               k_period: int = 14, d_period: int = 3) -> dict:
    """Stochastic Oscillator (%K, %D)."""
    k_values = [None] * len(closes)
    for i in range(k_period - 1, len(closes)):
        window_h = highs[i - k_period + 1:i + 1]
        window_l = lows[i - k_period + 1:i + 1]
        hh = max(window_h)
        ll = min(window_l)
        if hh != ll:
            k_values[i] = 100.0 * (closes[i] - ll) / (hh - ll)
        else:
            k_values[i] = 50.0

    # %D = SMA of %K
    k_valid = [v if v is not None else 0 for v in k_values]
    d_values = sma(k_valid, d_period)
    # Fix: only keep where we have valid K
    for i in range(len(d_values)):
        if k_values[i] is None:
            d_values[i] = None

    return {"k": k_values, "d": d_values}


def support_resistance(closes: list[float], lookback: int = 50, threshold_pct: float = 0.5) -> dict:
    """Find support and resistance levels from recent price action."""
    if len(closes) < lookback:
        lookback = len(closes)

    recent = closes[-lookback:]
    levels = []

    for i in range(2, len(recent) - 2):
        # Local minima (support)
        if recent[i] <= recent[i-1] and recent[i] <= recent[i-2] and \
           recent[i] <= recent[i+1] and recent[i] <= recent[i+2]:
            levels.append({"price": recent[i], "type": "support", "strength": 1})

        # Local maxima (resistance)
        if recent[i] >= recent[i-1] and recent[i] >= recent[i-2] and \
           recent[i] >= recent[i+1] and recent[i] >= recent[i+2]:
            levels.append({"price": recent[i], "type": "resistance", "strength": 1})

    # Cluster nearby levels
    clustered = []
    levels.sort(key=lambda x: x["price"])
    for level in levels:
        merged = False
        for cl in clustered:
            if abs(level["price"] - cl["price"]) / cl["price"] * 100 < threshold_pct:
                cl["price"] = (cl["price"] * cl["strength"] + level["price"]) / (cl["strength"] + 1)
                cl["strength"] += 1
                merged = True
                break
        if not merged:
            clustered.append(level.copy())

    supports = sorted([l for l in clustered if l["type"] == "support"], key=lambda x: x["price"], reverse=True)
    resistances = sorted([l for l in clustered if l["type"] == "resistance"], key=lambda x: x["price"])

    return {"supports": supports[:5], "resistances": resistances[:5]}


def trend_direction(closes: list[float], ma_period: int = 50) -> str:
    """Determine trend direction from MA. Returns 'UP', 'DOWN', or 'FLAT'."""
    ma = sma(closes, ma_period)
    if ma[-1] is None or len(closes) < 2:
        return "FLAT"

    current = closes[-1]
    current_ma = ma[-1]
    prev_ma = ma[-2] if ma[-2] is not None else current_ma

    if current > current_ma and current_ma > prev_ma:
        return "UP"
    elif current < current_ma and current_ma < prev_ma:
        return "DOWN"
    return "FLAT"


def candle_pattern(opens: list[float], highs: list[float], lows: list[float], closes: list[float]) -> list[str]:
    """Identify common candlestick patterns from the last few candles."""
    patterns = []
    n = len(closes)
    if n < 3:
        return patterns

    # Last candle
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    total_range = h - l if h != l else 0.0001

    # Doji
    if body / total_range < 0.1:
        patterns.append("DOJI")

    # Hammer (bullish reversal)
    if lower_wick > 2 * body and upper_wick < body * 0.5 and c > o:
        patterns.append("HAMMER_BULLISH")

    # Shooting star (bearish reversal)
    if upper_wick > 2 * body and lower_wick < body * 0.5 and c < o:
        patterns.append("SHOOTING_STAR_BEARISH")

    # Engulfing
    if n >= 2:
        po, pc = opens[-2], closes[-2]
        # Bullish engulfing
        if pc < po and c > o and c > po and o < pc:
            patterns.append("ENGULFING_BULLISH")
        # Bearish engulfing
        if pc > po and c < o and c < po and o > pc:
            patterns.append("ENGULFING_BEARISH")

    # Three white soldiers (bullish)
    if n >= 3:
        if all(closes[-i] > opens[-i] for i in range(1, 4)):
            if closes[-1] > closes[-2] > closes[-3]:
                patterns.append("THREE_WHITE_SOLDIERS")

    # Three black crows (bearish)
    if n >= 3:
        if all(closes[-i] < opens[-i] for i in range(1, 4)):
            if closes[-1] < closes[-2] < closes[-3]:
                patterns.append("THREE_BLACK_CROWS")

    return patterns
