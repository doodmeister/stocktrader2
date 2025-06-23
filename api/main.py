"""
FastAPI application entry point for StockTrader backend.

This module sets up the FastAPI application with all necessary middleware,
routers, and configuration for the stock trading analysis platform.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from api.routers import market_data_enhanced as market_data, analysis, health
from api.dependencies import verify_core_modules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting StockTrader API...")
    
    # Verify core modules are working
    try:
        verify_core_modules()
        logger.info("✅ Core modules verified successfully")
    except Exception as e:
        logger.error(f"❌ Core module verification failed: {e}")
        raise
    
    logger.info("🎉 StockTrader API started successfully")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down StockTrader API...")


# Create FastAPI application
app = FastAPI(
    title="StockTrader API",
    description="Advanced stock trading analysis platform with technical indicators, pattern recognition, and AI-powered insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(market_data.router, prefix="/api/v1/market-data", tags=["market-data"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "StockTrader API",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "Market data download and storage",
            "Technical indicator analysis (10+ indicators)",
            "Candlestick pattern detection (18+ patterns)",
            "OpenAI-powered market analysis",
            "Complete analysis pipeline"
        ],
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
