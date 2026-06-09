"""
BTC Technical Analysis Backend
Data: Binance public API (no key needed)
TA: pandas-ta
Analysis: LLM via Hermes provider
"""
import json
import urllib.request
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="BTC TA Analyzer", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BINANCE_URL = "https://api.binance.com/api/v3"


# ── Data ──────────────────────────────────────────────────────────

def fetch_klines(symbol="BTCUSDT", interval="1h", limit=200):
    """Fetch candlestick data from Binance public API."""
    url = f"{BINANCE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Hermes-BTC-TA/0.1"})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read())

    rows = []
    for k in data:
        rows.append({
            "timestamp": k[0],
            "datetime": datetime.fromtimestamp(k[0] / 1000).isoformat(),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return rows


def calc_indicators(df: pd.DataFrame):
    """Calculate technical analysis indicators."""
    # Moving Averages
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["EMA_12"] = ta.ema(df["close"], length=12)
    df["EMA_26"] = ta.ema(df["close"], length=26)

    # RSI
    df["RSI"] = ta.rsi(df["close"], length=14)

    # MACD
    macd = ta.macd(df["close"])
    if macd is not None:
        macd_cols = list(macd.columns)
        df["MACD"] = macd[macd_cols[0]]      # MACD_*
        df["MACD_Signal"] = macd[macd_cols[2]]  # MACDs_*
        df["MACD_Hist"] = macd[macd_cols[1]]    # MACDh_*

    # Bollinger Bands
    bb = ta.bbands(df["close"])
    if bb is not None:
        # Find the correct column names dynamically
        bb_cols = list(bb.columns)
        df["BB_Upper"] = bb[bb_cols[2]]  # BBU_*
        df["BB_Mid"] = bb[bb_cols[1]]    # BBM_*
        df["BB_Lower"] = bb[bb_cols[0]]  # BBL_*

    # ATR (volatility)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Volume SMA
    df["Volume_SMA_20"] = ta.sma(df["volume"], length=20)

    return df


def generate_signal(df: pd.DataFrame) -> dict:
    """Generate trading signal based on TA indicators."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    score = 0

    # RSI
    rsi = last["RSI"]
    if rsi < 30:
        signals.append("🟢 RSI超卖(<30)")
        score += 2
    elif rsi > 70:
        signals.append("🔴 RSI超买(>70)")
        score -= 2
    elif rsi < 40:
        signals.append("🟢 RSI偏低(<40)")
        score += 1
    elif rsi > 60:
        signals.append("🔴 RSI偏高(>60)")
        score -= 1

    # MACD
    macd_val = last["MACD"]
    macd_sig = last["MACD_Signal"]
    macd_prev = prev["MACD"]
    if macd_val > macd_sig and macd_prev <= prev["MACD_Signal"]:
        signals.append("🟢 MACD金叉")
        score += 2
    elif macd_val < macd_sig and macd_prev >= prev["MACD_Signal"]:
        signals.append("🔴 MACD死叉")
        score -= 2
    
    # SMA crossover
    if prev["SMA_20"] <= prev["SMA_50"] and last["SMA_20"] > last["SMA_50"]:
        signals.append("🟢 SMA20上穿SMA50(金叉)")
        score += 2
    elif prev["SMA_20"] >= prev["SMA_50"] and last["SMA_20"] < last["SMA_50"]:
        signals.append("🔴 SMA20下穿SMA50(死叉)")
        score -= 2

    # Bollinger Bands
    close = last["close"]
    bb_upper = last["BB_Upper"]
    bb_lower = last["BB_Lower"]
    if close > bb_upper:
        signals.append("🔴 价格突破布林上轨(超买)")
        score -= 1
    elif close < bb_lower:
        signals.append("🟢 价格跌破布林下轨(超卖)")
        score += 1

    # Volume
    vol_ratio = last["volume"] / last["Volume_SMA_20"] if last["Volume_SMA_20"] else 1
    if vol_ratio > 2:
        signals.append(f"📊 放量{vol_ratio:.1f}x")
    elif vol_ratio < 0.5:
        signals.append(f"📊 缩量{vol_ratio:.1f}x")

    # Final suggestion
    if score >= 3:
        suggestion = "买入"
        strength = "强烈" if score >= 5 else "温和"
    elif score <= -3:
        suggestion = "卖出"
        strength = "强烈" if score <= -5 else "温和"
    else:
        suggestion = "观望"
        strength = "中性"

    return {
        "score": score,
        "strength": strength,
        "suggestion": suggestion,
        "signals": signals,
        "price": last["close"],
        "rsi": round(rsi, 1),
        "macd": round(macd_val, 2),
        "bb_position": "上轨之上" if close > bb_upper else "下轨之下" if close < bb_lower else "轨道内",
        "vol_ratio": round(vol_ratio, 2),
    }


# ── API Endpoints ─────────────────────────────────────────────────

@app.get("/api/btc/overview")
def get_overview(interval: str = Query("1h", regex="^(15m|1h|4h|1d)$"), limit: int = Query(200, le=500)):
    """Get BTC price data, TA indicators, and trading signal."""
    try:
        raw = fetch_klines("BTCUSDT", interval, limit)
        df = pd.DataFrame(raw)
        df = calc_indicators(df)
        signal = generate_signal(df)
        
        # Latest candles for chart (last 100)
        candles = []
        for _, row in df.tail(100).iterrows():
            candles.append({
                "time": int(row["timestamp"] / 1000),
                "open": round(row["open"], 2),
                "high": round(row["high"], 2),
                "low": round(row["low"], 2),
                "close": round(row["close"], 2),
            })

        # Indicator series for overlay
        indicators = {}
        for col in ["SMA_20", "SMA_50", "BB_Upper", "BB_Mid", "BB_Lower"]:
            if col in df.columns:
                indicators[col] = [
                    {"time": int(row["timestamp"] / 1000), "value": round(row[col], 2)}
                    for _, row in df.tail(100).iterrows()
                    if pd.notna(row[col])
                ]

        # Sub charts (RSI, MACD)
        rsi_data = [
            {"time": int(row["timestamp"] / 1000), "value": round(row["RSI"], 1)}
            for _, row in df.tail(100).iterrows() if pd.notna(row["RSI"])
        ]
        macd_data = [
            {
                "time": int(row["timestamp"] / 1000),
                "macd": round(row["MACD"], 2),
                "signal": round(row["MACD_Signal"], 2),
                "histogram": round(row["MACD_Hist"], 2),
            }
            for _, row in df.tail(100).iterrows() if pd.notna(row["MACD"])
        ]

        last_row = df.iloc[-1]
        return {
            "success": True,
            "data": {
                "signal": signal,
                "current_price": round(last_row["close"], 2),
                "change_24h_pct": round((last_row["close"] - df.iloc[-97]["close"]) / df.iloc[-97]["close"] * 100, 2) if len(df) > 96 else 0,
                "high_24h": round(df.tail(96)["high"].max(), 2),
                "low_24h": round(df.tail(96)["low"].min(), 2),
                "volume_24h": round(df.tail(96)["volume"].sum(), 0),
                "candles": candles,
                "indicators": indicators,
                "rsi_data": rsi_data,
                "macd_data": macd_data,
                "interval": interval,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/btc/analysis")
def get_llm_analysis(interval: str = Query("1h", regex="^(15m|1h|4h|1d)$"), model: str = "deepseek-chat"):
    """Get LLM-powered analysis of BTC with trading advice."""
    try:
        raw = fetch_klines("BTCUSDT", interval, 200)
        df = pd.DataFrame(raw)
        df = calc_indicators(df)
        signal = generate_signal(df)
        last = df.iloc[-1]

        # Build analysis prompt
        prompt = f"""你是一个专业的加密货币技术分析师。请基于以下BTC({interval}级别)技术指标数据，给出交易分析和建议：

## 当前价格: ${last['close']:.2f}

## 技术指标
- RSI(14): {last['RSI']:.1f}
- MACD: {last['MACD']:.2f} (Signal: {last['MACD_Signal']:.2f})
- SMA20: ${last['SMA_20']:.2f}
- SMA50: ${last['SMA_50']:.2f}
- 布林带上轨: ${last['BB_Upper']:.2f}
- 布林带下轨: ${last['BB_Lower']:.2f}
- ATR(波动率): {last['ATR']:.2f}
- 成交量/均量比: {signal['vol_ratio']:.2f}x

## 信号: {signal['suggestion']} (强度: {signal['strength']}, Score: {signal['score']})
## 触发信号: {', '.join(signal['signals'])}

请输出：
1. 当前趋势判断（一句话）
2. 关键支撑位和阻力位
3. 明确的交易建议（买入/卖出/观望，含价格区间和仓位管理）
4. 风险提示（如果适用）

输出格式：简洁中文，Markdown，300字以内。"""

        # Call LLM via Hermes-compatible API
        llm_url = "https://api.stepfun.com/v1/chat/completions"
        with open("/root/.hermes/.env") as f:
            env = {}
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    env[k] = v
        stepfun_key = env.get("STEPFUN_API_KEY", "")
        deepseek_key = env.get("DEEPSEEK_API_KEY", "")

        api_key = deepseek_key or stepfun_key
        if not api_key:
            return {"success": False, "error": "No LLM API key available"}

        req_data = json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.3,
        }).encode()

        req = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=req_data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        analysis_text = result["choices"][0]["message"]["content"]

        return {
            "success": True,
            "data": {
                "analysis": analysis_text,
                "signal": signal,
                "model": model,
                "interval": interval,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

# Serve frontend static files
import os
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
