"""Technical indicators calculator. All functions take a DataFrame, return enriched DataFrame."""
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Optional


def calc_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full suite of indicators."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Trend ──
    df["SMA_20"] = ta.sma(c, length=20)
    df["SMA_50"] = ta.sma(c, length=50)
    df["EMA_12"] = ta.ema(c, length=12)
    df["EMA_26"] = ta.ema(c, length=26)

    # ── Momentum ──
    df["RSI"] = ta.rsi(c, length=14)

    # MACD
    macd = ta.macd(c)
    if macd is not None:
        cols = list(macd.columns)
        df["MACD"] = macd[cols[0]]
        df["MACD_Signal"] = macd[cols[2]]
        df["MACD_Hist"] = macd[cols[1]]

    # ADX
    adx = ta.adx(h, l, c)
    if adx is not None:
        cols = list(adx.columns)
        df["ADX"] = adx[cols[0]]
        df["DMP"] = adx[cols[1]]
        df["DMN"] = adx[cols[2]]

    # ── Volatility ──
    bb = ta.bbands(c, length=20, std=2)
    if bb is not None:
        cols = list(bb.columns)
        df["BB_Upper"] = bb[cols[2]]
        df["BB_Mid"] = bb[cols[1]]
        df["BB_Lower"] = bb[cols[0]]
        df["BB_Width"] = (bb[cols[2]] - bb[cols[0]]) / bb[cols[1]] * 100

    df["ATR"] = ta.atr(h, l, c, length=14)

    # ── Volume ──
    df["Volume_SMA_20"] = ta.sma(v, length=20)
    df["Volume_Ratio"] = v / df["Volume_SMA_20"].replace(0, np.nan)

    # VWAP
    tp = (h + l + c) / 3
    df["VWAP"] = (tp * v).cumsum() / v.cumsum()

    return df


def calc_orderflow(df: pd.DataFrame) -> pd.DataFrame:
    """Order flow: delta, CVD."""
    df["Delta"] = df["taker_buy_quote"] - (df["quote_vol"] - df["taker_buy_quote"])
    df["CVD"] = df["Delta"].cumsum()
    df["Delta_Pct"] = df["Delta"] / df["quote_vol"].replace(0, np.nan) * 100
    df["Buy_Pct"] = df["taker_buy_quote"] / df["quote_vol"].replace(0, np.nan) * 100
    return df


def to_native(obj):
    """Recursively convert numpy types to native Python."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series,)):
        return to_native(obj.tolist())
    return obj
