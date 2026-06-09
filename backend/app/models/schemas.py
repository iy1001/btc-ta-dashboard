"""Pydantic models / API contracts for BTC TA."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ── Data Models ──

class Kline(BaseModel):
    timestamp: int
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_vol: float
    trades: int
    taker_buy_vol: float
    taker_buy_quote: float


class Candle(BaseModel):
    time: int
    o: float
    h: float
    l: float
    c: float


class IndicatorLine(BaseModel):
    time: int
    value: float


class MACDPoint(BaseModel):
    time: int
    macd: float
    sig: float
    hist: float


class Signal(BaseModel):
    score: int
    max_score: int
    confidence: float
    strength: str
    suggestion: str
    signals: List[str]
    regime: Dict[str, Any]
    divergences: List[Dict[str, Any]]
    price: float
    rsi: float
    vol_ratio: float


class MultiTimeframe(BaseModel):
    aggregated_score: float
    aggregated_suggestion: str
    timeframes: Dict[str, Dict[str, Any]]
    current_price: Optional[float] = None


class OrderFlow(BaseModel):
    depth_imbalance: float
    bid_volume: float
    ask_volume: float
    cvd: float
    delta_pct: float
    buy_pct: float


# ── Response Models ──

class APIResponse(BaseModel):
    success: bool = True
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OverviewData(BaseModel):
    signal: Signal
    multi_timeframe: MultiTimeframe
    current_price: float
    change_pct: float
    high_24h: float
    low_24h: float
    volume_24h: float
    candles: List[Candle]
    indicators: Dict[str, List[IndicatorLine]]
    rsi_data: List[IndicatorLine]
    macd_data: List[MACDPoint]
    adx_data: Optional[List[Dict[str, Any]]] = None
    interval: str


class OverviewResponse(BaseModel):
    success: bool = True
    data: OverviewData
