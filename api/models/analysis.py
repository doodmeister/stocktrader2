"""
Pydantic models for analysis endpoints.

This module defines request and response models for technical analysis,
pattern detection, and OpenAI integration operations.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class AnalysisType(str, Enum):
    """Types of analysis available."""
    TECHNICAL_INDICATORS = "technical_indicators"
    PATTERN_DETECTION = "pattern_detection"
    CHATGPT_ANALYSIS = "chatgpt_analysis"
    COMPLETE_ANALYSIS = "complete_analysis"


class TechnicalIndicatorRequest(BaseModel):
    """Request model for technical indicator analysis."""
    
    symbol: str = Field(..., description="Stock symbol")
    data_source: str = Field(..., description="Data source: 'csv' or 'download'")
    csv_file_path: Optional[str] = Field(None, description="Path to CSV file (if data_source is 'csv')")
    period: Optional[str] = Field("1y", description="Period for download (if data_source is 'download')")
    
    # Indicator configuration
    include_indicators: List[str] = Field(
        default=[
            "rsi", "macd", "bollinger_bands", "sma", "ema", 
            "stochastic", "williams_r", "cci", "vwap", "obv", "adx"
        ],
        description="List of indicators to calculate"
    )
    
    # RSI settings
    rsi_period: int = Field(default=14, description="RSI calculation period")
    
    # MACD settings
    macd_fast: int = Field(default=12, description="MACD fast period")
    macd_slow: int = Field(default=26, description="MACD slow period")
    macd_signal: int = Field(default=9, description="MACD signal period")
    
    # Bollinger Bands settings
    bb_period: int = Field(default=20, description="Bollinger Bands period")
    bb_std_dev: float = Field(default=2.0, description="Bollinger Bands standard deviation")
    
    # Moving Averages settings
    sma_periods: List[int] = Field(default=[20, 50, 200], description="SMA periods")
    ema_periods: List[int] = Field(default=[12, 26, 50], description="EMA periods")
    
    @validator('include_indicators')
    def validate_indicators(cls, v):
        valid_indicators = [
            "rsi", "macd", "bollinger_bands", "sma", "ema",
            "stochastic", "williams_r", "cci", "vwap", "obv", "adx"
        ]
        for indicator in v:
            if indicator not in valid_indicators:
                raise ValueError(f"Invalid indicator: {indicator}. Valid options: {valid_indicators}")
        return v


class PatternDetectionRequest(BaseModel):
    """Request model for candlestick pattern detection."""
    
    symbol: str = Field(..., description="Stock symbol")
    data_source: str = Field(..., description="Data source: 'csv' or 'download'")
    csv_file_path: Optional[str] = Field(None, description="Path to CSV file (if data_source is 'csv')")
    period: Optional[str] = Field("1y", description="Period for download (if data_source is 'download')")
    
    # Pattern detection configuration
    min_confidence: float = Field(default=0.7, description="Minimum confidence threshold for patterns")
    include_patterns: Optional[List[str]] = Field(None, description="Specific patterns to detect (None for all)")
    recent_only: bool = Field(default=True, description="Only return patterns from recent data")
    lookback_days: int = Field(default=30, description="Days to look back for recent patterns")


class DetectedPattern(BaseModel):
    """Represents a single detected candlestick pattern instance."""
    date: str = Field(..., description="The date the pattern was detected.")
    pattern_name: str = Field(..., description="Name of the detected pattern.")
    confidence: float = Field(..., description="Confidence score of the detection.")
    pattern_type: str = Field(..., description="Type of pattern (e.g., bullish_reversal).")
    description: str = Field(..., description="Description of the pattern.")


class PatternDetectionResponse(BaseModel):
    """Response model for candlestick pattern detection."""
    symbol: str = Field(..., description="Stock symbol analyzed.")
    total_patterns_found: int = Field(..., description="Total number of pattern instances found.")
    patterns: List[DetectedPattern] = Field(..., description="List of detected pattern instances.")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    pattern_summary: dict = Field(..., description="Summary of detected patterns (bullish, bearish, neutral, total)")


class ChatGPTAnalysisRequest(BaseModel):
    """Request model for ChatGPT analysis."""
    
    symbol: str = Field(..., description="Stock symbol")
    analysis_data: Dict[str, Any] = Field(..., description="Compiled analysis data")
    
    # Analysis preferences
    analysis_focus: List[str] = Field(
        default=["technical", "patterns", "trends", "recommendations"],
        description="Areas to focus the analysis on"
    )
    market_context: Optional[str] = Field(None, description="Additional market context")
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance: conservative, moderate, aggressive")
    investment_horizon: str = Field(default="medium", description="Investment horizon: short, medium, long")
    
    @validator('analysis_focus')
    def validate_focus(cls, v):
        valid_focus = ["technical", "patterns", "trends", "recommendations", "risk", "sentiment"]
        for focus in v:
            if focus not in valid_focus:
                raise ValueError(f"Invalid focus area: {focus}. Valid options: {valid_focus}")
        return v
    
    @validator('risk_tolerance')
    def validate_risk_tolerance(cls, v):
        if v not in ["conservative", "moderate", "aggressive"]:
            raise ValueError("Risk tolerance must be: conservative, moderate, or aggressive")
        return v
    
    @validator('investment_horizon')
    def validate_investment_horizon(cls, v):
        if v not in ["short", "medium", "long"]:
            raise ValueError("Investment horizon must be: short, medium, or long")
        return v


class CompleteAnalysisRequest(BaseModel):
    """Request model for complete analysis pipeline."""
    
    symbol: str = Field(..., description="Stock symbol")
    data_source: str = Field(default="download", description="Data source: 'csv' or 'download'")
    csv_file_path: Optional[str] = Field(None, description="Path to CSV file (if data_source is 'csv')")
    period: str = Field(default="1y", description="Period for download (if data_source is 'download')")
    
    # Include all analysis types
    include_technical_indicators: bool = Field(default=True, description="Include technical indicators")
    include_pattern_detection: bool = Field(default=True, description="Include pattern detection")
    include_chatgpt_analysis: bool = Field(default=True, description="Include ChatGPT analysis")
    
    # Configuration for sub-analyses
    technical_config: Optional[TechnicalIndicatorRequest] = Field(None, description="Technical indicator configuration")
    pattern_config: Optional[PatternDetectionRequest] = Field(None, description="Pattern detection configuration")
    chatgpt_config: Optional[ChatGPTAnalysisRequest] = Field(None, description="ChatGPT analysis configuration")


class IndicatorResult(BaseModel):
    """Model for individual technical indicator result."""
    
    name: str = Field(..., description="Indicator name")
    current_value: Optional[Union[float, Dict[str, float]]] = Field(None, description="Current indicator value")
    signal: Optional[str] = Field(None, description="Signal: bullish, bearish, neutral")
    strength: Optional[float] = Field(None, description="Signal strength (0-1)")
    data: List[Dict[str, Any]] = Field(default=[], description="Historical indicator data")


class TechnicalIndicatorResponse(BaseModel):
    """Response model for technical indicator analysis."""
    
    symbol: str
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    data_period: str
    total_records: int
    
    indicators: List[IndicatorResult] = Field(..., description="Technical indicator results")
    overall_signal: str = Field(..., description="Overall signal: bullish, bearish, neutral")
    signal_strength: float = Field(..., description="Overall signal strength (0-1)")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


class ChatGPTAnalysisResponse(BaseModel):
    """Response model for ChatGPT analysis."""
    
    symbol: str
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    
    analysis: str = Field(..., description="ChatGPT analysis text")
    key_insights: List[str] = Field(..., description="Key insights extracted")
    recommendations: List[str] = Field(..., description="Trading recommendations")
    risk_assessment: str = Field(..., description="Risk assessment")
    confidence_level: str = Field(..., description="AI confidence level")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


class CompleteAnalysisResponse(BaseModel):
    """Response model for complete analysis pipeline."""
    
    symbol: str
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    
    market_data: Optional[Dict[str, Any]] = Field(None, description="Market data summary")
    technical_indicators: Optional[TechnicalIndicatorResponse] = Field(None, description="Technical analysis results")
    pattern_detection: Optional[PatternDetectionResponse] = Field(None, description="Pattern detection results")
    chatgpt_analysis: Optional[ChatGPTAnalysisResponse] = Field(None, description="ChatGPT analysis results")
    
    # Overall summary
    overall_signal: str = Field(..., description="Combined overall signal")
    confidence_score: float = Field(..., description="Overall confidence score (0-1)")
    execution_time_seconds: float = Field(..., description="Total analysis execution time")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }
