"""
Pydantic models for the StockTrader API.

This package contains all request and response models used by the API endpoints.
"""

from api.models.market_data import (
    MarketDataRequest,
    MarketDataResponse,
    LoadCSVRequest,
    LoadCSVResponse,
    MarketDataInfo,
    ErrorResponse
)

from api.models.analysis import (
    AnalysisType,
    TechnicalIndicatorRequest,
    PatternDetectionRequest,
    ChatGPTAnalysisRequest,
    CompleteAnalysisRequest,
    IndicatorResult,
    DetectedPattern,
    TechnicalIndicatorResponse,
    PatternDetectionResponse,
    ChatGPTAnalysisResponse,
    CompleteAnalysisResponse
)

__all__ = [
    # Market Data Models
    "MarketDataRequest",
    "MarketDataResponse", 
    "LoadCSVRequest",
    "LoadCSVResponse",
    "MarketDataInfo",
    "ErrorResponse",
    
    # Analysis Models
    "AnalysisType",
    "TechnicalIndicatorRequest",
    "PatternDetectionRequest", 
    "ChatGPTAnalysisRequest",
    "CompleteAnalysisRequest",
    "IndicatorResult",
    "DetectedPattern",
    "TechnicalIndicatorResponse",
    "PatternDetectionResponse",
    "ChatGPTAnalysisResponse",
    "CompleteAnalysisResponse"
]
