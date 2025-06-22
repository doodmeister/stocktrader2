"""
core/indicators/stochastic.py

Stochastic Oscillator calculation.
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Tuple
from utils.logger import setup_logger
from .base import IndicatorError, validate_input

logger = setup_logger(__name__)

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    ta = None  # Explicitly set to None for type checking
    logger.warning("pandas_ta not found - using fallback for Stochastic calculations")

@validate_input(['high', 'low', 'close'], check_ohlcv_coherence=True)
def calculate_stochastic(df: DataFrame, k_period: int = 14, d_period: int = 3, 
                        high_col: str = 'high', low_col: str = 'low', 
                        close_col: str = 'close') -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Args:
        df: DataFrame with OHLC data
        k_period: Period for %K calculation
        d_period: Period for %D (SMA of %K)
        high_col: Name of high price column
        low_col: Name of low price column
        close_col: Name of close price column
        
    Returns:
        tuple: (%K values, %D values)
    """
    try:
        if k_period < 1:
            raise ValueError("K period must be positive")
        if d_period < 1:
            raise ValueError("D period must be positive")
            
        if TA_AVAILABLE and ta is not None:
            stoch = ta.stoch(df[high_col], df[low_col], df[close_col], 
                           k=k_period, d=d_period)
            if stoch is None or stoch.empty:
                raise ValueError("pandas_ta.stoch returned None or empty DataFrame")
            
            # Extract %K and %D columns
            k_col = f'STOCHk_{k_period}_{d_period}_{d_period}'
            d_col = f'STOCHd_{k_period}_{d_period}_{d_period}'
            
            if k_col not in stoch.columns or d_col not in stoch.columns:
                # Try alternative column names
                k_cols = [col for col in stoch.columns if 'k' in col.lower() or 'K' in col]
                d_cols = [col for col in stoch.columns if 'd' in col.lower() or 'D' in col]
                
                if not k_cols or not d_cols:
                    raise KeyError(f"Stochastic columns not found. Available: {stoch.columns}")
                
                k_col = k_cols[0]
                d_col = d_cols[0]
            
            return stoch[k_col], stoch[d_col]
        else:
            # Fallback implementation
            lowest_low = df[low_col].rolling(window=k_period, min_periods=1).min()
            highest_high = df[high_col].rolling(window=k_period, min_periods=1).max()
            
            # Calculate %K
            k_percent = 100 * ((df[close_col] - lowest_low) / 
                             (highest_high - lowest_low))
            k_percent = k_percent.fillna(50)  # Handle division by zero
            
            # Calculate %D (SMA of %K)
            d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
            
            return k_percent, d_percent
            
    except Exception as e:
        logger.error(f"Stochastic calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate Stochastic: {e}") from e
