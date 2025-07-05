"""
Pydantic models for market data endpoints.

This module defines request and response models for market data operations
including downloading, loading, and storing stock data.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
import pandas as pd


class MarketDataRequest(BaseModel):
    """Request model for downloading market data."""
    
    symbol: str = Field(..., description="Stock symbol (e.g., 'AAPL', 'MSFT')")
    period: str = Field(default="1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")
    interval: str = Field(default="1d", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)")
    save_csv: bool = Field(default=True, description="Whether to save data as CSV file")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Symbol is required")
        return v.strip().upper()
    
    @validator('period')
    def validate_period(cls, v):
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if v not in valid_periods:
            raise ValueError(f"Period must be one of: {', '.join(valid_periods)}")
        return v
    
    @validator('interval')
    def validate_interval(cls, v):
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if v not in valid_intervals:
            raise ValueError(f"Interval must be one of: {', '.join(valid_intervals)}")
        return v


class MarketDataResponse(BaseModel):
    """Response model for market data operations."""
    
    symbol: str
    period: str
    interval: str
    start_date: str
    end_date: str
    total_records: int
    csv_file_path: Optional[str] = None
    data_preview: List[Dict[str, Any]] = Field(description="First 5 rows of data")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            date: lambda d: d.isoformat(),
        }


class LoadCSVRequest(BaseModel):
    """Request model for loading CSV data."""
    
    file_path: str = Field(..., description="Path to CSV file to load")
    symbol: Optional[str] = Field(None, description="Symbol for validation (optional)")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("File path is required")
        return v.strip()


class LoadCSVResponse(BaseModel):
    """Response model for loaded CSV data."""
    
    file_path: str
    symbol: Optional[str]
    total_records: int
    start_date: str
    end_date: str
    columns: List[str]
    data_preview: List[Dict[str, Any]]
    file_size_mb: float


class MarketDataInfo(BaseModel):
    """Model for market data information."""
    
    symbol: str
    current_price: Optional[float] = None
    previous_close: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


class TechnicalAnalysisRequest(BaseModel):
    """Request model for technical analysis."""
    
    symbol: str = Field(..., description="Stock symbol to analyze")
    csv_file: Optional[str] = Field(None, description="CSV file path if analyzing existing data")
    indicators: Optional[List[str]] = Field(
        default=["rsi", "macd", "bollinger_bands", "sma", "ema"],
        description="List of technical indicators to calculate"
    )
    
    # RSI parameters
    rsi_period: int = Field(default=14, description="RSI period")
    
    # MACD parameters
    macd_fast: int = Field(default=12, description="MACD fast period")
    macd_slow: int = Field(default=26, description="MACD slow period")
    macd_signal: int = Field(default=9, description="MACD signal period")
    
    # Bollinger Bands parameters
    bb_period: int = Field(default=20, description="Bollinger Bands period")
    bb_std: float = Field(default=2.0, description="Bollinger Bands standard deviation")
    
    # Moving average parameters
    sma_period: int = Field(default=20, description="Simple Moving Average period")
    ema_period: int = Field(default=20, description="Exponential Moving Average period")


class TechnicalIndicatorData(BaseModel):
    """Individual technical indicator result."""
    
    name: str = Field(..., description="Indicator name")
    current_value: Optional[float] = Field(None, description="Current indicator value")
    signal: Optional[str] = Field(None, description="Buy/Sell/Hold signal")
    values: Optional[List[float]] = Field(None, description="Time series values (last 10 periods)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional indicator metadata")


class TechnicalAnalysisResponse(BaseModel):
    """Response model for technical analysis."""
    
    status: str = Field(..., description="Analysis status")
    symbol: str = Field(..., description="Analyzed symbol")
    analysis_time: datetime = Field(..., description="When analysis was performed")
    
    # Overall signal
    composite_signal: Optional[float] = Field(None, description="Composite signal (-1 to 1, bearish to bullish)")
    signal_strength: Optional[str] = Field(None, description="Signal strength (Weak/Medium/Strong)")
    
    # Individual indicators
    indicators: Dict[str, TechnicalIndicatorData] = Field(..., description="Technical indicator results")
    
    # Summary statistics
    data_info: Dict[str, Any] = Field(..., description="Data information (rows, date range, etc.)")
    
    # Market context
    current_price: Optional[float] = Field(None, description="Current/latest close price")
    price_change: Optional[float] = Field(None, description="Price change from previous period")
    price_change_percent: Optional[float] = Field(None, description="Price change percentage")


class PatternDetectionRequest(BaseModel):
    """Request model for pattern detection."""
    
    symbol: str = Field(..., description="Stock symbol to analyze")
    csv_file: Optional[str] = Field(None, description="CSV file path if analyzing existing data")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold (0.0-1.0)")
    pattern_types: Optional[List[str]] = Field(
        default=None,
        description="Specific patterns to detect (if None, detect all)"
    )


class PatternResult(BaseModel):
    """Individual pattern detection result."""
    
    pattern_name: str = Field(..., description="Pattern name")
    confidence: float = Field(..., description="Detection confidence (0.0-1.0)")
    pattern_type: str = Field(..., description="Pattern type (bullish/bearish/neutral)")
    detection_date: str = Field(..., description="Date when pattern was detected")
    description: str = Field(..., description="Pattern description")


class PatternDetectionResponse(BaseModel):
    """Response model for pattern detection."""
    
    status: str = Field(..., description="Detection status")
    symbol: str = Field(..., description="Analyzed symbol")
    analysis_time: datetime = Field(..., description="When analysis was performed")
    
    # Patterns found
    patterns_detected: List[PatternResult] = Field(..., description="Detected patterns")
    total_patterns: int = Field(..., description="Total number of patterns found")
    
    # Summary by type
    bullish_patterns: int = Field(default=0, description="Number of bullish patterns")
    bearish_patterns: int = Field(default=0, description="Number of bearish patterns")
    neutral_patterns: int = Field(default=0, description="Number of neutral patterns")
    
    # Data info
    data_info: Dict[str, Any] = Field(..., description="Data information")
