"""
core/indicators/vwap.py

Volume Weighted Average Price (VWAP) calculation.
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
    logger.warning("pandas_ta not found - using fallback for VWAP calculations")

@validate_input(['high', 'low', 'close', 'volume'], check_ohlcv_coherence=True)
def calculate_vwap(df: DataFrame, anchor: str = 'D',
                  high_col: str = 'high', low_col: str = 'low', 
                  close_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        df: DataFrame with OHLCV data
        anchor: Anchor period ('D' for daily, 'W' for weekly, etc.)
        high_col: Name of high price column
        low_col: Name of low price column
        close_col: Name of close price column
        volume_col: Name of volume column
        
    Returns:
        pd.Series: VWAP values
    """
    try:
        if TA_AVAILABLE and ta is not None:
            # Check if we have a proper datetime index for pandas_ta
            if hasattr(df.index, 'to_period'):
                vwap = ta.vwap(df[high_col], df[low_col], df[close_col], 
                              df[volume_col], anchor=anchor)
                if vwap is not None:
                    return vwap
            
            # If pandas_ta fails or index is not datetime, fall through to manual calculation
        
        # Fallback implementation (always works regardless of index type)
        # Calculate Typical Price
        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        
        # Calculate Price * Volume
        pv = typical_price * df[volume_col]
        
        # Simple cumulative VWAP (works with any index type)
        vwap = pv.cumsum() / df[volume_col].cumsum()
        
        return vwap
            
    except Exception as e:
        logger.error(f"VWAP calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate VWAP: {e}") from e

@validate_input(['close', 'volume'])
def calculate_obv(df: DataFrame, close_col: str = 'close', 
                 volume_col: str = 'volume') -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        df: DataFrame with price and volume data
        close_col: Name of close price column
        volume_col: Name of volume column
        
    Returns:
        pd.Series: OBV values
    """
    try:
        if TA_AVAILABLE and ta is not None:
            obv = ta.obv(df[close_col], df[volume_col])
            if obv is None:
                raise ValueError("pandas_ta.obv returned None")
            return obv
        else:
            # Fallback implementation
            price_change = df[close_col].diff()
              # If price goes up, add volume; if down, subtract volume; if same, add 0
            volume_direction = np.where(price_change.astype(float) > 0, df[volume_col],
                                      np.where(price_change.astype(float) < 0, -df[volume_col], 0))
            
            # Cumulative sum
            obv = pd.Series(volume_direction, index=df.index).cumsum()
            
            return obv
            
    except Exception as e:
        logger.error(f"OBV calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate OBV: {e}") from e
