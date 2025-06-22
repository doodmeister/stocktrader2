"""
core/indicators/williams_r.py

Williams %R calculation.
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
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
    logger.warning("pandas_ta not found - using fallback for Williams %R calculations")

@validate_input(['high', 'low', 'close'], check_ohlcv_coherence=True)
def calculate_williams_r(df: DataFrame, length: int = 14, 
                        high_col: str = 'high', low_col: str = 'low', 
                        close_col: str = 'close') -> pd.Series:
    """
    Calculate Williams %R oscillator.
    
    Args:
        df: DataFrame with OHLC data
        length: Period for Williams %R calculation
        high_col: Name of high price column
        low_col: Name of low price column
        close_col: Name of close price column
        
    Returns:
        pd.Series: Williams %R values (-100 to 0)
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
            
        if TA_AVAILABLE and ta is not None:
            willr = ta.willr(df[high_col], df[low_col], df[close_col], length=length)
            if willr is None:
                raise ValueError("pandas_ta.willr returned None")
            return willr
        else:
            # Fallback implementation
            highest_high = df[high_col].rolling(window=length, min_periods=1).max()
            lowest_low = df[low_col].rolling(window=length, min_periods=1).min()
            
            # Williams %R formula: -100 * (Highest High - Close) / (Highest High - Lowest Low)
            willr = -100 * ((highest_high - df[close_col]) / 
                           (highest_high - lowest_low))
            
            # Handle division by zero
            willr = willr.fillna(-50)
            
            return willr.clip(-100, 0)
            
    except Exception as e:
        logger.error(f"Williams %R calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate Williams %R: {e}") from e
