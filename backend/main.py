"""
BTC Technical Analysis Backend v0.2
Focus: Pure technical analysis with multi-timeframe signal aggregation
"""
import json
import urllib.request
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="BTC TA Analyzer", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BINANCE_URL = "https://api.binance.com/api/v3"
INTERVALS = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}


# ── Data ──

def fetch_klines(symbol="BTCUSDT", interval="1h", limit=500):
    url = f"{BINANCE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Hermes-BTC-TA/0.2"})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read())
    rows = []
    for k in data:
        rows.append({
            "timestamp": k[0],
            "datetime": datetime.fromtimestamp(k[0] / 1000),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
            "close": float(k[4]), "volume": float(k[5]),
        })
    return rows


# ── Indicators ──

def calc_indicators(df: pd.DataFrame):
    """Full suite of technical indicators."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Trend ──
    df["SMA_10"] = ta.sma(c, length=10)
    df["SMA_20"] = ta.sma(c, length=20)
    df["SMA_50"] = ta.sma(c, length=50)
    df["SMA_200"] = ta.sma(c, length=200)
    df["EMA_12"] = ta.ema(c, length=12)
    df["EMA_26"] = ta.ema(c, length=26)

    # Ichimoku Cloud
    ichimoku = ta.ichimoku(h, l, c)
    if ichimoku is not None:
        ichimoku_df = ichimoku[0] if isinstance(ichimoku, tuple) else ichimoku
        for col in ichimoku_df.columns:
            if col in df.columns:
                df[col] = ichimoku_df[col].values if len(ichimoku_df) == len(df) else None
    
    # ── Momentum ──
    df["RSI"] = ta.rsi(c, length=14)
    df["Stoch_K"] = ta.stoch(h, l, c)["STOCHk_14_3_3"] if ta.stoch(h, l, c) is not None else None
    df["Stoch_D"] = ta.stoch(h, l, c)["STOCHd_14_3_3"] if ta.stoch(h, l, c) is not None else None

    # MACD
    macd = ta.macd(c)
    if macd is not None:
        cols = list(macd.columns)
        df["MACD"] = macd[cols[0]]
        df["MACD_Signal"] = macd[cols[2]]
        df["MACD_Hist"] = macd[cols[1]]

    # ADX (trend strength)
    adx = ta.adx(h, l, c)
    if adx is not None:
        cols = list(adx.columns)
        df["ADX"] = adx[cols[0]]
        df["DMP"] = adx[cols[1]]
        df["DMN"] = adx[cols[2]]

    # ── Volatility ──
    bb = ta.bbands(c, length=20, std=2)
    if bb is not None:
        bb_cols = list(bb.columns)
        df["BB_Upper"] = bb[bb_cols[2]]
        df["BB_Mid"] = bb[bb_cols[1]]
        df["BB_Lower"] = bb[bb_cols[0]]
        df["BB_Width"] = (bb[bb_cols[2]] - bb[bb_cols[0]]) / bb[bb_cols[1]] * 100

    df["ATR"] = ta.atr(h, l, c, length=14)
    df["ATR_Pct"] = df["ATR"] / c * 100

    # ── Volume ──
    df["Volume_SMA_20"] = ta.sma(v, length=20)
    df["Volume_Ratio"] = v / df["Volume_SMA_20"]
    df["OBV"] = ta.obv(c, v)
    df["MFI"] = ta.mfi(h, l, c, v, length=14) if ta.mfi(h, l, c, v, length=14) is not None else None

    # ── Price Action ──
    # Candlestick patterns
    df["Doji"] = abs(c - (h + l) / 2) <= (h - l) * 0.1
    df["Bullish_Engulf"] = (df["open"].shift(1) > df["close"].shift(1)) & (c > df["open"].shift(1)) & (df["open"] < df["close"].shift(1))
    df["Bearish_Engulf"] = (df["close"].shift(1) > df["open"].shift(1)) & (c < df["open"].shift(1)) & (df["open"] > df["close"].shift(1))
    df["Hammer"] = (h - l > 3 * abs(c - df["open"])) & ((c - l) / (h - l) > 0.9) & (abs(c - df["open"]) / (h - l) < 0.3)
    df["Shooting_Star"] = (h - l > 3 * abs(c - df["open"])) & ((h - c) / (h - l) > 0.9) & (abs(c - df["open"]) / (h - l) < 0.3)

    # Support/Resistance (simple pivot points)
    df["Pivot"] = (h + l + c) / 3
    df["R1"] = 2 * df["Pivot"] - l
    df["S1"] = 2 * df["Pivot"] - h

    return df


# ── Signal Engine ──

def generate_signal(df: pd.DataFrame) -> dict:
    """Multi-indicator signal engine with weighted scoring."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    score = 0
    weights = {"trend": 0, "momentum": 0, "volatility": 0, "volume": 0, "pattern": 0}
    details = {}

    # ═══ TREND (weight: 5) ═══
    trend_score = 0

    # SMA alignment: 10 > 20 > 50 = bullish
    if last["SMA_10"] > last["SMA_20"] > last["SMA_50"]:
        trend_score += 2
        signals.append("🟢 均线多头排列")
    elif last["SMA_10"] < last["SMA_20"] < last["SMA_50"]:
        trend_score -= 2
        signals.append("🔴 均线空头排列")

    # SMA crossover
    if prev["SMA_20"] <= prev["SMA_50"] and last["SMA_20"] > last["SMA_50"]:
        trend_score += 3
        signals.append("🟢 SMA20金叉SMA50")
    elif prev["SMA_20"] >= prev["SMA_50"] and last["SMA_20"] < last["SMA_50"]:
        trend_score -= 3
        signals.append("🔴 SMA20死叉SMA50")

    # Price vs SMA200
    if pd.notna(last["SMA_200"]):
        if last["close"] > last["SMA_200"]:
            trend_score += 1
        else:
            trend_score -= 1

    # ADX
    if pd.notna(last.get("ADX")):
        if last["ADX"] > 25:
            if last["DMP"] > last["DMN"]:
                trend_score += 2
                signals.append(f"🟢 强趋势上涨(ADX:{last['ADX']:.0f})")
            else:
                trend_score -= 2
                signals.append(f"🔴 强趋势下跌(ADX:{last['ADX']:.0f})")
        elif last["ADX"] < 20:
            signals.append(f"⚪ 震荡市场(ADX:{last['ADX']:.0f})")

    score += trend_score * 5
    weights["trend"] = trend_score
    details["trend"] = f"趋势得分: {trend_score}"

    # ═══ MOMENTUM (weight: 4) ═══
    mom_score = 0

    # RSI
    rsi = last["RSI"]
    if rsi < 30:
        mom_score += 3
        signals.append("🟢 RSI超卖区(<30)")
    elif rsi > 70:
        mom_score -= 3
        signals.append("🔴 RSI超买区(>70)")
    elif rsi < 40:
        mom_score += 1
    elif rsi > 60:
        mom_score -= 1

    # RSI divergence (simple)
    rsi_5ago = df["RSI"].iloc[-6] if len(df) > 5 else rsi
    price_5ago = df["close"].iloc[-6] if len(df) > 5 else last["close"]
    if last["close"] > price_5ago and rsi < rsi_5ago:
        mom_score -= 2
        signals.append("🔴 RSI顶背离(看跌)")
    elif last["close"] < price_5ago and rsi > rsi_5ago:
        mom_score += 2
        signals.append("🟢 RSI底背离(看涨)")

    # MACD
    macd_val = last["MACD"]
    macd_sig = last["MACD_Signal"]
    macd_prev = prev["MACD"]
    macd_sig_prev = prev["MACD_Signal"]
    if macd_val > macd_sig and macd_prev <= macd_sig_prev:
        mom_score += 3
        signals.append("🟢 MACD金叉")
    elif macd_val < macd_sig and macd_prev >= macd_sig_prev:
        mom_score -= 3
        signals.append("🔴 MACD死叉")

    # MACD histogram turning
    if last["MACD_Hist"] > 0 and prev["MACD_Hist"] <= 0:
        mom_score += 1
        signals.append("🟢 MACD柱转正")
    elif last["MACD_Hist"] < 0 and prev["MACD_Hist"] >= 0:
        mom_score -= 1
        signals.append("🔴 MACD柱转负")

    # Stochastic
    if pd.notna(last.get("Stoch_K")):
        sk, sd = last["Stoch_K"], last["Stoch_D"]
        if sk < 20 and sd < 20:
            mom_score += 2
            signals.append(f"🟢 随机指标超卖(K:{sk:.0f})")
        elif sk > 80 and sd > 80:
            mom_score -= 2
            signals.append(f"🔴 随机指标超买(K:{sk:.0f})")

    score += mom_score * 4
    weights["momentum"] = mom_score
    details["momentum"] = f"动量得分: {mom_score}"

    # ═══ VOLATILITY (weight: 2) ═══
    vol_score = 0

    # Bollinger Band position
    close = last["close"]
    bb_upper = last["BB_Upper"]
    bb_lower = last["BB_Lower"]
    bb_width = last.get("BB_Width", 0)
    
    if close > bb_upper:
        vol_score -= 1
        signals.append("🔴 价格突破布林上轨")
    elif close < bb_lower:
        vol_score += 1
        signals.append("🟢 价格跌破布林下轨")

    # Bollinger Band squeeze (low volatility预示突破)
    if bb_width < bb_width * 0.8:  # Simplified
        signals.append("⚪ 布林带收缩(变盘前兆)")

    score += vol_score * 2
    weights["volatility"] = vol_score
    details["volatility"] = f"波动得分: {vol_score}"

    # ═══ VOLUME (weight: 2) ═══
    vol_score2 = 0
    vol_ratio = last["Volume_Ratio"] if pd.notna(last["Volume_Ratio"]) else 1
    
    if vol_ratio > 2 and score > 0:
        vol_score2 += 1
        signals.append(f"📊 放量{vol_ratio:.1f}x(确认上涨)")
    elif vol_ratio > 2 and score < 0:
        vol_score2 -= 1
        signals.append(f"📊 放量{vol_ratio:.1f}x(确认下跌)")
    elif vol_ratio < 0.5:
        signals.append(f"📊 缩量{vol_ratio:.1f}x")

    # MFI
    if pd.notna(last.get("MFI")):
        if last["MFI"] < 20:
            vol_score2 += 1
            signals.append(f"🟢 MFI超卖({last['MFI']:.0f})")
        elif last["MFI"] > 80:
            vol_score2 -= 1
            signals.append(f"🔴 MFI超买({last['MFI']:.0f})")

    score += vol_score2 * 2
    weights["volume"] = vol_score2
    details["volume"] = f"成交量得分: {vol_score2}"

    # ═══ CANDLESTICK PATTERNS (weight: 1) ═══
    pat_score = 0
    if last.get("Bullish_Engulf"):
        pat_score += 2
        signals.append("🟢 看涨吞没形态")
    if last.get("Bearish_Engulf"):
        pat_score -= 2
        signals.append("🔴 看跌吞没形态")
    if last.get("Hammer"):
        pat_score += 1
        signals.append("🟢 锤子线(潜在反转)")
    if last.get("Shooting_Star"):
        pat_score -= 1
        signals.append("🔴 射击之星(潜在反转)")
    if last.get("Doji"):
        signals.append("⚪ 十字星(犹豫信号)")

    score += pat_score * 1
    weights["pattern"] = pat_score
    details["pattern"] = f"形态得分: {pat_score}"

    # ═══ FINAL ═══
    max_possible = 5 * 5 + 4 * 3 + 2 * 1 + 2 * 1 + 1 * 2  # max theoretical
    confidence = min(abs(score) / 20 * 100, 100) if max_possible > 0 else 0

    if score >= 6:
        suggestion = "买入"
        strength = "强烈" if score >= 12 else "温和"
    elif score <= -6:
        suggestion = "卖出"
        strength = "强烈" if score <= -12 else "温和"
    else:
        suggestion = "观望"
        strength = "中性"

    # Key levels
    key_levels = {
        "support_1": round(last.get("S1", last["close"] * 0.97), 2),
        "resistance_1": round(last.get("R1", last["close"] * 1.03), 2),
        "pivot": round(last.get("Pivot", last["close"]), 2),
    }

    return {
        "score": score,
        "max_score": max_possible,
        "confidence": round(confidence, 1),
        "strength": strength,
        "suggestion": suggestion,
        "signals": signals,
        "weights": weights,
        "details": details,
        "price": round(last["close"], 2),
        "rsi": round(rsi, 1),
        "adx": round(last.get("ADX", 0), 1) if pd.notna(last.get("ADX")) else None,
        "macd": round(macd_val, 2),
        "bb_position": "上轨之上" if close > bb_upper else "下轨之下" if close < bb_lower else "轨道内",
        "bb_width": round(bb_width, 2) if pd.notna(bb_width) else 0,
        "vol_ratio": round(vol_ratio, 2),
        "atr_pct": round(last.get("ATR_Pct", 0), 2),
        "key_levels": key_levels,
    }


# ── Multi-timeframe Aggregation ──

def aggregate_timeframes():
    """Fetch data from all timeframes and aggregate signals."""
    results = {}
    for name, interval in INTERVALS.items():
        raw = fetch_klines("BTCUSDT", interval, 200)
        df = pd.DataFrame(raw)
        df = calc_indicators(df)
        signal = generate_signal(df)
        results[name] = {
            "signal": signal["suggestion"],
            "score": signal["score"],
            "strength": signal["strength"],
            "price": signal["price"],
            "rsi": signal["rsi"],
            "adx": signal["adx"],
            "confidence": signal["confidence"],
            "atr_pct": signal["atr_pct"],
            "signals": signal["signals"][:3],
        }

    # Aggregate: weighted by timeframe importance
    weights_tf = {"15m": 1, "1h": 2, "4h": 3, "1d": 4}
    total_score = sum(results[tf]["score"] * weights_tf[tf] for tf in results)
    total_weight = sum(weights_tf[tf] for tf in results)
    agg_score = total_score / total_weight if total_weight > 0 else 0

    if agg_score >= 5:
        agg_suggestion = "买入"
    elif agg_score <= -5:
        agg_suggestion = "卖出"
    else:
        agg_suggestion = "观望"

    return {
        "aggregated_score": round(agg_score, 1),
        "aggregated_suggestion": agg_suggestion,
        "timeframes": results,
        "current_price": results.get("1h", {}).get("price"),
    }


# ── API Endpoints ──

@app.get("/api/btc/overview")
def get_overview(interval: str = Query("1h", regex="^(15m|1h|4h|1d)$"), limit: int = Query(200, le=500)):
    try:
        raw = fetch_klines("BTCUSDT", interval, limit)
        df = pd.DataFrame(raw)
        df = calc_indicators(df)
        signal = generate_signal(df)

        # Chart data
        tail = df.tail(120)
        candles = [{"time": int(r["timestamp"] / 1000), "open": round(r["open"], 2),
                     "high": round(r["high"], 2), "low": round(r["low"], 2), "close": round(r["close"], 2)}
                   for _, r in tail.iterrows()]

        indicators = {}
        for col in ["SMA_10", "SMA_20", "SMA_50", "BB_Upper", "BB_Mid", "BB_Lower"]:
            if col in df.columns:
                indicators[col] = [
                    {"time": int(r["timestamp"] / 1000), "value": round(r[col], 2)}
                    for _, r in tail.iterrows() if pd.notna(r[col])
                ]

        rsi_data = [{"time": int(r["timestamp"] / 1000), "value": round(r["RSI"], 1)}
                    for _, r in tail.iterrows() if pd.notna(r["RSI"])]
        macd_data = [{"time": int(r["timestamp"] / 1000), "macd": round(r["MACD"], 2),
                       "signal": round(r["MACD_Signal"], 2), "histogram": round(r["MACD_Hist"], 2)}
                     for _, r in tail.iterrows() if pd.notna(r["MACD"])]
        
        # ADX data
        adx_data = None
        if "ADX" in df.columns:
            adx_data = [{"time": int(r["timestamp"] / 1000), "adx": round(r["ADX"], 1),
                          "dmp": round(r["DMP"], 1), "dmn": round(r["DMN"], 1)}
                        for _, r in tail.iterrows() if pd.notna(r.get("ADX"))]

        last_row = df.iloc[-1]
        change_24h = round((last_row["close"] - tail.iloc[0]["close"]) / tail.iloc[0]["close"] * 100, 2)

        # Multi-timeframe
        multi = aggregate_timeframes()

        return {
            "success": True,
            "data": {
                "signal": signal,
                "multi_timeframe": multi,
                "current_price": round(last_row["close"], 2),
                "change_pct": change_24h,
                "high_24h": round(df.tail(96)["high"].max(), 2),
                "low_24h": round(df.tail(96)["low"].min(), 2),
                "volume_24h": round(df.tail(96)["volume"].sum(), 0),
                "candles": candles,
                "indicators": indicators,
                "rsi_data": rsi_data,
                "macd_data": macd_data,
                "adx_data": adx_data,
                "interval": interval,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/btc/multi")
def get_multi_timeframe():
    """Get aggregated signal across all timeframes."""
    try:
        return {"success": True, "data": aggregate_timeframes()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/btc/indicators")
def get_indicators(interval: str = Query("1h", regex="^(15m|1h|4h|1d)$")):
    """Get raw indicator values for the latest candle."""
    try:
        raw = fetch_klines("BTCUSDT", interval, 200)
        df = pd.DataFrame(raw)
        df = calc_indicators(df)
        last = df.iloc[-1]
        
        ind = {}
        for col in ["RSI", "MACD", "MACD_Signal", "MACD_Hist", "ADX", "DMP", "DMN",
                     "ATR", "ATR_Pct", "BB_Upper", "BB_Mid", "BB_Lower", "BB_Width",
                     "SMA_10", "SMA_20", "SMA_50", "SMA_200", "Stoch_K", "Stoch_D",
                     "MFI", "OBV", "Volume_Ratio"]:
            if col in df.columns and pd.notna(last.get(col)):
                ind[col] = round(float(last[col]), 2) if isinstance(last[col], (int, float, np.floating)) else str(last[col])
        
        return {"success": True, "data": {"price": round(last["close"], 2), "indicators": ind, "interval": interval}}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/btc/analysis")
def get_llm_analysis(interval: str = Query("1h", regex="^(15m|1h|4h|1d)$")):
    try:
        raw = fetch_klines("BTCUSDT", interval, 200)
        df = pd.DataFrame(raw)
        df = calc_indicators(df)
        signal = generate_signal(df)
        last = df.iloc[-1]
        multi = aggregate_timeframes()

        # Build comprehensive prompt
        prompt = f"""你是一个专业的加密货币技术分析师。请基于以下BTC({interval})技术指标，给出交易分析和建议。

## 当前价格: ${last['close']:.2f}
24h涨跌: {multi['timeframes']['1d']['score']}

## 多周期聚合信号
{' | '.join(f'{tf}: {multi["timeframes"][tf]["signal"]}({multi["timeframes"][tf]["score"]:+d})' for tf in multi['timeframes'])}
聚合判断: {multi['aggregated_suggestion']} (得分: {multi['aggregated_score']:+})

## 关键指标
- RSI(14): {last['RSI']:.1f}
- MACD: {last['MACD']:.2f} (Signal: {last['MACD_Signal']:.2f})
- ADX: {last.get('ADX', 0):.1f} (趋势强度)
- 布林带宽度: {last.get('BB_Width', 0):.1f}%
- ATR(波动率): {last.get('ATR_Pct', 0):.2f}%
- 成交量比: {signal['vol_ratio']:.2f}x
- 随机指标K/D: {last.get('Stoch_K', 0):.0f}/{last.get('Stoch_D', 0):.0f}

## 均线
- SMA20: ${last['SMA_20']:.0f} | SMA50: ${last['SMA_50']:.0f}
- 价格相对SMA200: {'上方' if pd.notna(last.get('SMA_200')) and last['close'] > last['SMA_200'] else '下方'}

## 信号输出
{chr(10).join(signal['signals'])}

请输出：
1. **趋势判断**（一句话）
2. **关键价位**（支撑/阻力）
3. **交易建议**（具体到入场区间、止损位）
4. **风险提示**

简洁中文，300字以内。"""

        # Call DeepSeek
        import subprocess, os
        result = subprocess.run(["bash", "-c", "source ~/.hermes/.env 2>/dev/null; echo \"$DEEPSEEK_API_KEY\""],
                               capture_output=True, text=True)
        api_key = result.stdout.strip()
        
        if not api_key:
            api_key = os.environ.get("STEPFUN_API_KEY", "")
        if not api_key:
            return {"success": False, "error": "No API key"}

        req_data = json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.3,
        }).encode()

        req = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=req_data,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        analysis = result["choices"][0]["message"]["content"]

        return {"success": True, "data": {"analysis": analysis, "signal": signal, "multi": multi, "interval": interval}}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.2.0"}

# Serve frontend
import os
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
