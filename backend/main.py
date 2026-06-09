#!/usr/bin/env python3
"""BTC Technical Analysis Dashboard — Entry point."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.api import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
