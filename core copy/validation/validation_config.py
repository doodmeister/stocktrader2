"""
Validation configuration for the StockTrader Bot.

This module contains centralized configuration for all validation operations
including market data validation, symbol validation, and API limits.
"""
import re
from typing import Dict, Set


class ValidationConfig:
    """Central configuration for validation parameters and thresholds."""
    
    # Symbol validation patterns
    SYMBOL_MIN_LENGTH = 1
    SYMBOL_MAX_LENGTH = 10
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.-]{1,10}$')
    SYMBOL_REQUIRED_ALPHA = True
    
    # Supported stock exchanges (for symbol validation)
    SUPPORTED_EXCHANGES = {
        'NYSE', 'NASDAQ', 'AMEX', 'TSX', 'LSE', 'JSE'
    }
    
    # Date validation intervals (in days) - Yahoo Finance API limits
    INTERVAL_LIMITS = {
        "1m": 7,      # 7 days max for 1-minute data
        "2m": 60,     # 60 days max for 2-minute data
        "5m": 60,     # 60 days max for 5-minute data
        "15m": 60,    # 60 days max for 15-minute data
        "30m": 60,    # 60 days max for 30-minute data
        "60m": 730,   # ~2 years max for 1-hour data
        "1h": 730,    # ~2 years max for hourly data
        "1d": 36500,  # ~100 years (essentially unlimited) for daily data
        "5d": 36500,  # No limit for 5-day data
        "1wk": 36500, # No limit for weekly data
        "1mo": 36500, # No limit for monthly data
        "3mo": 36500  # No limit for quarterly data
    }
    
    # Valid time intervals for market data requests
    VALID_INTERVALS = set(INTERVAL_LIMITS.keys())
    
    # Cache and API settings for FastAPI backend
    SYMBOL_CACHE_TTL = 4 * 60 * 60  # 4 hours in seconds
    VALIDATION_CACHE_SIZE = 1000
    API_TIMEOUT = 30  # Increased for complex indicator calculations
    MAX_API_CALLS_PER_BATCH = 15
    REQUEST_RATE_LIMIT = 100  # requests per minute
    
    # Data validation thresholds
    MIN_DATASET_SIZE = 10
    MAX_DATASET_SIZE = 100_000  # Prevent memory issues
    MAX_NULL_PERCENTAGE = 0.05  # 5% max null values
    MAX_INVALID_OHLC_PERCENTAGE = 0.05  # 5% max invalid OHLC relationships
    
    # Special handling for technical indicator columns (higher null tolerance)
    INDICATOR_NULL_TOLERANCE = 0.7  # 70% for rolling indicators
    INDICATOR_KEYWORDS = {
        'rsi', 'sma', 'ema', 'macd', 'bb', 'bollinger', 'atr', 'adx', 
        'stochastic', 'stoch', 'williams', 'cci', 'vwap', 'obv', 'roc'
    }
    
    # Financial data validation ranges
    MIN_PRICE = 0.01
    MAX_PRICE = 1_000_000.0
    MIN_VOLUME = 0
    MAX_VOLUME = 1_000_000_000_000  # 1 trillion
    MIN_QUANTITY = 0
    MAX_QUANTITY = 1_000_000_000
    MIN_PERCENTAGE = 0.0
    MAX_PERCENTAGE = 1.0
    
    # OHLCV column definitions
    DEFAULT_OHLCV_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    REQUIRED_PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close']
    OPTIONAL_COLUMNS = ['Volume', 'Adj Close', 'Dividends', 'Stock Splits']
    
    # Security and input validation
    MAX_INPUT_LENGTH = 1000
    DANGEROUS_CHARS_PATTERN = re.compile(r'[<>"\'\x00-\x1f\x7f-\x9f]')
    
    # API validation for FastAPI integration
    MAX_SYMBOLS_PER_REQUEST = 50
    MAX_PERIOD_DAYS = 3650  # ~10 years maximum lookback
    MIN_PERIOD_DAYS = 1
    
    # Error message templates
    ERROR_MESSAGES = {
        'invalid_symbol': 'Symbol "{symbol}" does not match required pattern',
        'invalid_interval': 'Interval "{interval}" is not supported',
        'invalid_period': 'Period exceeds maximum allowed for interval "{interval}"',
        'missing_columns': 'Required columns missing: {columns}',
        'invalid_ohlc': 'OHLC data validation failed: {details}',
        'data_too_large': 'Dataset size ({size}) exceeds maximum allowed ({max_size})',
        'insufficient_data': 'Dataset size ({size}) below minimum required ({min_size})'
    }
    
    @classmethod
    def is_indicator_column(cls, column_name: str) -> bool:
        """Check if a column name represents a technical indicator."""
        column_lower = column_name.lower()
        return any(keyword in column_lower for keyword in cls.INDICATOR_KEYWORDS)
    
    @classmethod
    def get_null_tolerance(cls, column_name: str) -> float:
        """Get the appropriate null tolerance for a column."""
        return cls.INDICATOR_NULL_TOLERANCE if cls.is_indicator_column(column_name) else cls.MAX_NULL_PERCENTAGE
    
    @classmethod
    def validate_interval(cls, interval: str) -> bool:
        """Validate if an interval is supported."""
        return interval in cls.VALID_INTERVALS
    
    @classmethod
    def get_max_period_for_interval(cls, interval: str) -> int:
        """Get maximum period in days for a given interval."""
        return cls.INTERVAL_LIMITS.get(interval, 365)  # Default to 1 year
