"""
Pydantic models for data validation in the StockTrader Bot.

This module contains validated data structures for financial data,
market data points, and API request/response models.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from .validation_config import ValidationConfig


class SymbolRequest(BaseModel):
    """Request model for symbol validation."""
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        extra='forbid'
    )
    
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol to validate")
    exchange: Optional[str] = Field(None, description="Exchange code (optional)")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol_format(cls, v: str) -> str:
        """Validate symbol format and convert to uppercase."""
        v_upper = v.upper()
        if not ValidationConfig.SYMBOL_PATTERN.match(v_upper):
            raise ValueError(f"Symbol '{v}' does not match required pattern")
        return v_upper
    
    @field_validator('exchange')
    @classmethod
    def validate_exchange(cls, v: Optional[str]) -> Optional[str]:
        """Validate exchange code if provided."""
        if v is not None:
            v_upper = v.upper()
            if v_upper not in ValidationConfig.SUPPORTED_EXCHANGES:
                raise ValueError(f"Exchange '{v}' is not supported")
            return v_upper
        return v


class MarketDataRequest(BaseModel):
    """Request model for market data retrieval."""
    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)
    
    symbol: str = Field(..., min_length=1, max_length=10)
    interval: str = Field(..., description="Time interval (1m, 5m, 1h, 1d, etc.)")
    period: str = Field(..., description="Data period (1d, 5d, 1mo, 1y, etc.)")
    start_date: Optional[datetime] = Field(None, description="Start date for historical data")
    end_date: Optional[datetime] = Field(None, description="End date for historical data")
    include_indicators: bool = Field(False, description="Whether to include technical indicators")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        v_upper = v.upper()
        if not ValidationConfig.SYMBOL_PATTERN.match(v_upper):
            raise ValueError(ValidationConfig.ERROR_MESSAGES['invalid_symbol'].format(symbol=v))
        return v_upper
    
    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v: str) -> str:
        """Validate time interval."""
        if not ValidationConfig.validate_interval(v):
            raise ValueError(ValidationConfig.ERROR_MESSAGES['invalid_interval'].format(interval=v))
        return v
    
    @model_validator(mode='after')
    def validate_date_range(self) -> 'MarketDataRequest':
        """Validate date range consistency."""
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                raise ValueError("Start date must be before end date")
        return self


class FinancialData(BaseModel):
    """Validated financial data structure."""
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        extra='forbid'
    )
    
    symbol: str = Field(..., min_length=1, max_length=10)
    price: float = Field(..., gt=0, le=ValidationConfig.MAX_PRICE)
    volume: int = Field(..., ge=0, le=ValidationConfig.MAX_VOLUME)
    timestamp: datetime = Field(...)
    exchange: Optional[str] = Field(None)
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol_format(cls, v: str) -> str:
        """Validate symbol format using configuration pattern."""
        v_upper = v.upper()
        if not ValidationConfig.SYMBOL_PATTERN.match(v_upper):
            raise ValueError(f"Symbol '{v}' does not match required pattern")
        return v_upper


class MarketDataPoint(BaseModel):
    """Validated OHLCV market data point."""
    model_config = ConfigDict(validate_assignment=True)
    
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    timestamp: datetime = Field(..., description="Data timestamp")
    symbol: Optional[str] = Field(None, description="Stock symbol")
    
    @model_validator(mode='after')
    def validate_ohlc_relationships(self) -> 'MarketDataPoint':
        """Validate OHLC price relationships."""
        # High should be the maximum price
        if self.high < max(self.open, self.low, self.close):
            raise ValueError(f"High price ({self.high}) must be >= max(open, low, close)")
        
        # Low should be the minimum price
        if self.low > min(self.open, self.high, self.close):
            raise ValueError(f"Low price ({self.low}) must be <= min(open, high, close)")
        
        return self


class TechnicalIndicatorData(BaseModel):
    """Validated technical indicator data point."""
    model_config = ConfigDict(validate_assignment=True)
    
    timestamp: datetime = Field(..., description="Data timestamp")
    symbol: str = Field(..., description="Stock symbol")
    indicator_name: str = Field(..., description="Name of the technical indicator")
    value: Optional[float] = Field(None, description="Indicator value (can be None for initial periods)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional indicator metadata")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper()
    
    @field_validator('indicator_name')
    @classmethod
    def validate_indicator_name(cls, v: str) -> str:
        """Validate indicator name format."""
        # Allow alphanumeric, underscore, and dash
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Indicator name must contain only alphanumeric characters, underscores, and dashes")
        return v.upper()


class DataFrameValidationRequest(BaseModel):
    """Request model for DataFrame validation."""
    model_config = ConfigDict(validate_assignment=True)
    
    required_columns: Optional[List[str]] = Field(None, description="Required column names")
    check_ohlcv: bool = Field(True, description="Whether to validate OHLCV relationships")
    min_rows: Optional[int] = Field(1, description="Minimum required rows")
    max_rows: Optional[int] = Field(None, description="Maximum allowed rows")
    detect_anomalies: bool = Field(False, description="Whether to detect anomalies")
    anomaly_detection_level: Optional[str] = Field("basic", description="Anomaly detection level")
    max_null_percentage: float = Field(0.1, description="Maximum allowed null percentage")
    
    @field_validator('anomaly_detection_level')
    @classmethod
    def validate_anomaly_level(cls, v: Optional[str]) -> Optional[str]:
        """Validate anomaly detection level."""
        if v is not None and v not in ['basic', 'advanced']:
            raise ValueError("Anomaly detection level must be 'basic' or 'advanced'")
        return v
    
    @field_validator('max_null_percentage')
    @classmethod
    def validate_null_percentage(cls, v: float) -> float:
        """Validate null percentage range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Max null percentage must be between 0.0 and 1.0")
        return v


class IndicatorCalculationRequest(BaseModel):
    """Request model for technical indicator calculations."""
    model_config = ConfigDict(validate_assignment=True)
    
    symbol: str = Field(..., description="Stock symbol")
    indicators: List[str] = Field(..., description="List of indicators to calculate")
    period: int = Field(14, description="Default period for indicators")
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Custom parameters for indicators")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        v_upper = v.upper()
        if not ValidationConfig.SYMBOL_PATTERN.match(v_upper):
            raise ValueError(f"Invalid symbol format: {v}")
        return v_upper
    
    @field_validator('indicators')
    @classmethod
    def validate_indicators(cls, v: List[str]) -> List[str]:
        """Validate indicator names."""
        if not v:
            raise ValueError("At least one indicator must be specified")
        
        # Normalize indicator names
        normalized = [indicator.upper() for indicator in v]
        
        # Check for supported indicators (this could be expanded)
        supported = {'RSI', 'MACD', 'SMA', 'EMA', 'BOLLINGER_BANDS', 'ATR', 'ADX', 'STOCHASTIC', 'WILLIAMS_R', 'CCI', 'VWAP', 'OBV'}
        unsupported = set(normalized) - supported
        if unsupported:
            raise ValueError(f"Unsupported indicators: {list(unsupported)}")
        
        return normalized
    
    @field_validator('period')
    @classmethod
    def validate_period(cls, v: int) -> int:
        """Validate indicator period."""
        if not 1 <= v <= 200:
            raise ValueError("Period must be between 1 and 200")
        return v
