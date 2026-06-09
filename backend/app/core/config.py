"""App configuration."""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Settings:
    binance_url: str = "https://api.binance.com/api/v3"
    cache_ttl_seconds: int = 60  # Cache klines for 60s
    default_limit: int = 200
    max_limit: int = 500
    user_agent: str = "BTC-TA/1.0"
    intervals: Dict[str, str] = field(default_factory=lambda: {
        "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"
    })


settings = Settings()
