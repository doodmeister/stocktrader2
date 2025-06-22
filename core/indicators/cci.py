"""
core/indicators/cci.py

Commodity Channel Index (CCI) calculation.
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
    logger.warning("pandas_ta not found - using fallback for CCI calculations")

@validate_input(['high', 'low', 'close'], check_ohlcv_coherence=True)
def calculate_cci(df: DataFrame, length: int = 20, constant: float = 0.015,
                 high_col: str = 'high', low_col: str = 'low', 
                 close_col: str = 'close') -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).
    
    Args:
        df: DataFrame with OHLC data
        length: Period for CCI calculation
        constant: Constant for CCI calculation (typically 0.015)
        high_col: Name of high price column
        low_col: Name of low price column
        close_col: Name of close price column
        
    Returns:
        pd.Series: CCI values
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
        if constant <= 0:
            raise ValueError("Constant must be positive")
            
        if TA_AVAILABLE and ta is not None:
            cci = ta.cci(df[high_col], df[low_col], df[close_col], 
                        length=length, c=constant)
            if cci is None:
                raise ValueError("pandas_ta.cci returned None")
            return cci
        else:
            # Fallback implementation
            # Calculate Typical Price (TP)
            tp = (df[high_col] + df[low_col] + df[close_col]) / 3
            
            # Calculate Simple Moving Average of TP
            sma_tp = tp.rolling(window=length, min_periods=1).mean()
            
            # Calculate Mean Absolute Deviation
            mad = tp.rolling(window=length, min_periods=1).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )
            
            # Calculate CCI
            cci = (tp - sma_tp) / (constant * mad)
            
            # Handle division by zero
            cci = cci.fillna(0)
            
            return cci
            
    except Exception as e:
        logger.error(f"CCI calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate CCI: {e}") from e
