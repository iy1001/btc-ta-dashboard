"""FastAPI app factory."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.routes.btc import router as btc_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="BTC Technical Analysis Dashboard",
        version="1.0.0",
        description="Real-time BTC technical analysis with multi-timeframe signal aggregation, order flow, and regime detection.",
        docs_url="/docs",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(btc_router)

    # Serve frontend static files
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
    frontend_dir = os.path.normpath(frontend_dir)
    if os.path.exists(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app
