"""Data provider: Binance API with in-memory caching."""
import json
import time
import urllib.request
from datetime import datetime
from typing import List, Optional, Dict, Any
from app.core.config import settings


# ── Simple TTL Cache ──

_cache: Dict[str, Any] = {}


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < settings.cache_ttl_seconds:
        return entry["data"]
    return None


def _cache_set(key: str, data: Any):
    _cache[key] = {"data": data, "ts": time.time()}


# ── Fetcher ──

def _fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": settings.user_agent})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 200) -> List[dict]:
    """Fetch klines with caching."""
    cache_key = f"klines:{symbol}:{interval}:{limit}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    url = f"{settings.binance_url}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    raw = _fetch_json(url)

    rows = []
    for k in raw:
        rows.append({
            "timestamp": k[0],
            "datetime": datetime.fromtimestamp(k[0] / 1000),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
            "close": float(k[4]), "volume": float(k[5]),
            "quote_vol": float(k[7]), "trades": int(k[8]),
            "taker_buy_vol": float(k[9]), "taker_buy_quote": float(k[10]),
        })
    _cache_set(cache_key, rows)
    return rows


def fetch_depth(symbol: str = "BTCUSDT", limit: int = 20) -> dict:
    """Fetch order book depth."""
    cache_key = f"depth:{symbol}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    url = f"{settings.binance_url}/depth?symbol={symbol}&limit={limit}"
    data = _fetch_json(url)
    bids = sum(float(b[1]) for b in data["bids"])
    asks = sum(float(a[1]) for a in data["asks"])
    result = {
        "bids": round(bids, 4),
        "asks": round(asks, 4),
        "imbalance": round((bids - asks) / (bids + asks) * 100, 2) if (bids + asks) > 0 else 0,
    }
    _cache_set(cache_key, result)
    return result


def invalidate_cache():
    """Clear all cached data."""
    _cache.clear()
