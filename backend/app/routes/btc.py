"""BTC TA routes."""
import pandas as pd
from fastapi import APIRouter, Query
from app.services.data.provider import fetch_klines, fetch_depth, invalidate_cache
from app.services.indicators.calculator import calc_all, calc_orderflow, to_native
from app.services.signal.engine import generate_signal, detect_regime, detect_divergence

router = APIRouter(prefix="/api/btc", tags=["btc"])


@router.get("/overview")
def get_overview(interval: str = Query("1h", regex="^(15m|1h|4h|1d)$"), limit: int = Query(200, le=500)):
    try:
        raw = fetch_klines("BTCUSDT", interval, limit)
        df = pd.DataFrame(raw)
        df = calc_all(df)
        df = calc_orderflow(df)
        signal = generate_signal(df)
        regi = detect_regime(df)
        divs = detect_divergence(df)

        tail = df.tail(120)
        candles = [{"time": int(r["timestamp"]/1000), "o": round(r["open"],2),
                     "h": round(r["high"],2), "l": round(r["low"],2), "c": round(r["close"],2)}
                   for _, r in tail.iterrows()]

        indicators = {}
        for col in ["SMA_20","SMA_50","BB_Upper","BB_Mid","BB_Lower","VWAP"]:
            if col in df.columns:
                indicators[col] = [{"time": int(r["timestamp"]/1000), "v": round(r[col],2)}
                                   for _, r in tail.iterrows() if pd.notna(r[col])]

        rsi_data = [{"time": int(r["timestamp"]/1000), "v": round(r["RSI"],1)}
                    for _, r in tail.iterrows() if pd.notna(r["RSI"])]
        macd_data = [{"time": int(r["timestamp"]/1000), "macd": round(r["MACD"],2),
                       "sig": round(r["MACD_Signal"],2), "hist": round(r["MACD_Hist"],2)}
                     for _, r in tail.iterrows() if pd.notna(r["MACD"])]

        last_r = df.iloc[-1]
        chg = round((last_r["close"] - tail.iloc[0]["close"]) / tail.iloc[0]["close"] * 100, 2) if len(tail) > 0 else 0

        # Multi-timeframe
        multi_tfs = {}
        for tf_name, tf_int in {"15m":"15m","1h":"1h","4h":"4h","1d":"1d"}.items():
            tf_raw = fetch_klines("BTCUSDT", tf_int, 100)
            tf_df = pd.DataFrame(tf_raw)
            tf_df = calc_all(tf_df)
            tf_sig = generate_signal(tf_df)
            multi_tfs[tf_name] = {
                "signal": tf_sig["suggestion"], "score": tf_sig["score"],
                "rsi": tf_sig["rsi"], "confidence": min(abs(tf_sig["score"])/20*100, 100),
            }

        w = {"15m":1,"1h":2,"4h":3,"1d":4}
        agg_score = sum(multi_tfs[t]["score"] * w[t] for t in multi_tfs) / sum(w.values())
        agg_sug = "买入" if agg_score >= 5 else "卖出" if agg_score <= -5 else "观望"

        return to_native({"success": True, "data": {
            "signal": signal,
            "current_price": round(last_r["close"], 2), "change_pct": chg,
            "high_24h": round(df.tail(96)["high"].max(), 2) if len(df) > 96 else 0,
            "low_24h": round(df.tail(96)["low"].min(), 2) if len(df) > 96 else 0,
            "volume_24h": round(df.tail(96)["volume"].sum(), 0) if len(df) > 96 else 0,
            "candles": candles, "indicators": indicators,
            "rsi_data": rsi_data, "macd_data": macd_data,
            "multi_timeframe": {"aggregated_score": round(agg_score, 1),
                                 "aggregated_suggestion": agg_sug,
                                 "timeframes": multi_tfs,
                                 "current_price": round(last_r["close"], 2)},
            "interval": interval,
        }})
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/orderflow")
def get_orderflow():
    try:
        raw = fetch_klines("BTCUSDT", "1h", 48)
        df = pd.DataFrame(raw)
        df = calc_orderflow(df)
        depth = fetch_depth()
        last = df.iloc[-1]
        return {"success": True, "data": {
            "depth_imbalance": depth["imbalance"],
            "bid_volume": depth["bids"], "ask_volume": depth["asks"],
            "cvd": float(last["CVD"]),
            "delta_pct": float(last["Delta_Pct"]),
            "buy_pct": float(last["Buy_Pct"]),
        }}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/multi")
def get_multi():
    try:
        tfs = {}
        for name, intv in {"15m":"15m","1h":"1h","4h":"4h","1d":"1d"}.items():
            raw = fetch_klines("BTCUSDT", intv, 100)
            df = pd.DataFrame(raw)
            df = calc_all(df)
            sig = generate_signal(df)
            tfs[name] = {"signal": sig["suggestion"], "score": sig["score"], "rsi": sig["rsi"]}
        w = {"15m":1,"1h":2,"4h":3,"1d":4}
        agg = sum(tfs[t]["score"] * w[t] for t in tfs) / sum(w.values())
        sug = "买入" if agg >= 5 else "卖出" if agg <= -5 else "观望"
        return {"success": True, "data": {
            "aggregated_score": round(agg, 1), "aggregated_suggestion": sug, "timeframes": tfs,
        }}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/cache/clear")
def clear_cache():
    invalidate_cache()
    return {"success": True, "message": "Cache cleared"}
