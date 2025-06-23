"""
FastAPI dependencies for the StockTrader API.

This module provides common dependencies used across the API endpoints,
including authentication, validation, and core module verification.
"""

from fastapi import HTTPException
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)


def verify_core_modules() -> bool:
    """
    Verify that all core StockTrader modules are importable and functional.
    
    Returns:
        bool: True if all modules are working
        
    Raises:
        HTTPException: If any core module fails to import or initialize
    """
    try:
        # Test core data validation
        from core.data_validator import validate_file_path, validate_dataframe
        
        # Test technical indicators
        from core.technical_indicators import TechnicalIndicators
        
        # Test pattern recognition
        from patterns.orchestrator import CandlestickPatterns
        from patterns.factory import create_pattern_detector
        
        # Test feature engineering
        from train.feature_engineering import compute_technical_features
        
        # Test security framework
        from security.authentication import create_jwt_token, get_api_credentials
        from security.authorization import create_user_context, Role
        
        logger.info("✅ All core modules imported successfully")
        return True
        
    except ImportError as e:
        error_msg = f"Failed to import core module: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Core module verification failed: {e}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


def get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment variables.
    
    Returns:
        str: OpenAI API key
        
    Raises:
        HTTPException: If API key is not configured
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )
    return api_key


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize stock symbol.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        str: Normalized symbol
        
    Raises:
        HTTPException: If symbol is invalid
    """
    if not symbol or len(symbol.strip()) == 0:
        raise HTTPException(status_code=400, detail="Stock symbol is required")
    
    # Normalize symbol (uppercase, remove whitespace)
    normalized = symbol.strip().upper()
    
    # Basic validation (alphanumeric and common characters)
    if not normalized.replace(".", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid stock symbol format")
    
    if len(normalized) > 10:
        raise HTTPException(status_code=400, detail="Stock symbol too long")
    
    return normalized


def validate_period(period: str) -> str:
    """
    Validate time period for stock data.
    
    Args:
        period: Time period (e.g., '1y', '6mo', '3mo')
        
    Returns:
        str: Validated period
        
    Raises:
        HTTPException: If period is invalid
    """
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    
    if period not in valid_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period. Must be one of: {', '.join(valid_periods)}"
        )
    
    return period


def validate_interval(interval: str) -> str:
    """
    Validate time interval for stock data.
    
    Args:
        interval: Time interval (e.g., '1d', '1h', '5m')
        
    Returns:
        str: Validated interval
        
    Raises:
        HTTPException: If interval is invalid
    """
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    if interval not in valid_intervals:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interval. Must be one of: {', '.join(valid_intervals)}"
        )
    
    return interval


def ensure_data_directory() -> str:
    """
    Ensure the data directory exists for storing CSV files.
    
    Returns:
        str: Path to data directory
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    return data_dir
