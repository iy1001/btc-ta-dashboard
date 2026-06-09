"""Signal engine: regime-aware multi-indicator scoring."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any


def detect_regime(df: pd.DataFrame) -> Dict[str, Any]:
    """Classify market regime from ADX and BB width."""
    if len(df) < 50:
        return {"regime": "unknown", "volatility": "normal", "adx": 0, "trending": False}

    adx = float(df["ADX"].iloc[-1]) if "ADX" in df.columns and pd.notna(df["ADX"].iloc[-1]) else 0
    bb_w = df["BB_Width"].tail(50) if "BB_Width" in df.columns else pd.Series([0])
    cw = float(bb_w.iloc[-1]) if len(bb_w) > 0 else 0
    aw = float(bb_w.mean()) if len(bb_w) > 0 else 1

    vol = "high" if aw > 0 and cw > aw * 1.3 else "low" if aw > 0 and cw < aw * 0.7 else "normal"
    trending = adx > 25

    if trending:
        dmp = float(df["DMP"].iloc[-1]) if "DMP" in df.columns else 0
        dmn = float(df["DMN"].iloc[-1]) if "DMN" in df.columns else 0
        regime = "uptrend" if dmp > dmn else "downtrend"
    else:
        regime = "ranging"

    return {"regime": regime, "volatility": vol, "adx": round(adx, 1), "trending": trending}


def detect_divergence(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect RSI divergence over 10 and 20 bar lookback."""
    if len(df) < 30:
        return []
    divs = []
    tail = df.tail(30)
    prices = tail["close"].values
    rsis = tail["RSI"].values

    for lb in [10, 20]:
        if len(prices) < lb + 5:
            continue
        p0, p1 = prices[-1], prices[-lb]
        r0, r1 = rsis[-1], rsis[-lb]

        if p0 > p1 and r0 < r1 and abs(r0 - r1) > 3:
            divs.append({
                "type": "bearish", "indicator": "RSI", "lookback": lb,
                "strength": "strong" if abs(r0 - r1) > 8 else "moderate",
                "desc": f"RSI顶背离({lb}): 价高({p0:.0f}>{p1:.0f}) RSI低({r0:.1f}<{r1:.1f})",
            })
        elif p0 < p1 and r0 > r1 and abs(r0 - r1) > 3:
            divs.append({
                "type": "bullish", "indicator": "RSI", "lookback": lb,
                "strength": "strong" if abs(r0 - r1) > 8 else "moderate",
                "desc": f"RSI底背离({lb}): 价低({p0:.0f}<{p1:.0f}) RSI高({r0:.1f}>{r1:.1f})",
            })
    return divs


def generate_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """Regime-aware signal engine."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    score = 0
    regime = detect_regime(df)
    divs = detect_divergence(df)

    # Regime-adaptive weights
    if regime["regime"] == "ranging":
        tw, mw, vw = 3, 5, 3
    elif regime["regime"] in ("uptrend", "downtrend"):
        tw, mw, vw = 5, 3, 2
    else:
        tw, mw, vw = 4, 4, 2

    # Trend
    ts = 0
    if pd.notna(last.get("SMA_10")) and pd.notna(last.get("SMA_50")):
        ts += 2 if last["SMA_10"] > last["SMA_50"] else -2
    if pd.notna(last.get("SMA_20")) and pd.notna(prev.get("SMA_20")) and pd.notna(last.get("SMA_50")):
        if prev["SMA_20"] <= prev["SMA_50"] and last["SMA_20"] > last["SMA_50"]:
            ts += 3; signals.append("🟢 SMA20金叉SMA50")
        elif prev["SMA_20"] >= prev["SMA_50"] and last["SMA_20"] < last["SMA_50"]:
            ts -= 3; signals.append("🔴 SMA20死叉SMA50")
    score += ts * tw

    # Momentum
    ms = 0
    rsi = float(last["RSI"])
    if rsi < 30: ms += 3; signals.append("🟢 RSI超卖(<30)")
    elif rsi > 70: ms -= 3; signals.append("🔴 RSI超买(>70)")
    elif rsi < 40: ms += 1
    elif rsi > 60: ms -= 1

    if pd.notna(last.get("MACD")) and pd.notna(last.get("MACD_Signal")):
        mv, msig = float(last["MACD"]), float(last["MACD_Signal"])
        mp, msp = float(prev["MACD"]), float(prev["MACD_Signal"])
        if mv > msig and mp <= msp:
            ms += 3; signals.append("🟢 MACD金叉")
        elif mv < msig and mp >= msp:
            ms -= 3; signals.append("🔴 MACD死叉")
    score += ms * mw

    # Divergence
    for d in divs:
        w = -4 if d["type"] == "bearish" else 4
        score += w
        signals.append(d["desc"])

    # Volatility
    vs = 0
    close = float(last["close"])
    if pd.notna(last.get("BB_Upper")) and pd.notna(last.get("BB_Lower")):
        if close > float(last["BB_Upper"]): vs -= 1; signals.append("🔴 突破布林上轨")
        elif close < float(last["BB_Lower"]): vs += 1; signals.append("🟢 跌破布林下轨")
    vr = float(last["Volume_Ratio"]) if pd.notna(last.get("Volume_Ratio")) else 1
    if vr > 2: signals.append(f"📊 放量{vr:.1f}x")
    elif vr < 0.5: signals.append(f"📊 缩量{vr:.1f}x")
    score += vs * vw

    # Regime bias
    if regime["regime"] == "uptrend": score += 1
    elif regime["regime"] == "downtrend": score -= 1

    # Final
    if score >= 8:
        sug, st = "买入", "强烈" if score >= 14 else "温和"
    elif score <= -8:
        sug, st = "卖出", "强烈" if score <= -14 else "温和"
    else:
        sug, st = "观望", "中性"

    return {
        "score": score, "suggestion": sug, "strength": st,
        "signals": signals, "regime": regime, "divergences": divs,
        "price": round(close, 2), "rsi": round(rsi, 1), "vol_ratio": round(vr, 2),
    }
