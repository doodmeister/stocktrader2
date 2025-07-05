"""
core/indicators/adx.py

Average Directional Index (ADX) calculation.
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
    logger.warning("pandas_ta not found - using fallback for ADX calculations")

@validate_input(['high', 'low', 'close'], check_ohlcv_coherence=True)
def calculate_adx(df: DataFrame, length: int = 14,
                 high_col: str = 'high', low_col: str = 'low', 
                 close_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX) and Directional Indicators (+DI, -DI).
    
    Args:
        df: DataFrame with OHLC data
        length: Period for ADX calculation
        high_col: Name of high price column
        low_col: Name of low price column
        close_col: Name of close price column
        
    Returns:
        tuple: (ADX values, +DI values, -DI values)
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
            
        if TA_AVAILABLE and ta is not None:
            adx_data = ta.adx(df[high_col], df[low_col], df[close_col], length=length)
            if adx_data is None or adx_data.empty:
                raise ValueError("pandas_ta.adx returned None or empty DataFrame")
            
            # Extract ADX, +DI, -DI columns
            adx_col = f'ADX_{length}'
            dmp_col = f'DMP_{length}'  # +DI
            dmn_col = f'DMN_{length}'  # -DI
            
            if not all(col in adx_data.columns for col in [adx_col, dmp_col, dmn_col]):
                raise KeyError(f"ADX columns not found. Available: {adx_data.columns}")
            
            return adx_data[adx_col], adx_data[dmp_col], adx_data[dmn_col]
        else:
            # Fallback implementation
            # Calculate True Range (TR)
            high_low = df[high_col] - df[low_col]
            high_close = (df[high_col] - df[close_col].shift()).abs()
            low_close = (df[low_col] - df[close_col].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            plus_dm = df[high_col] - df[high_col].shift()
            minus_dm = df[low_col].shift() - df[low_col]
            
            # Apply directional movement rules
            plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
            minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
            
            plus_dm = pd.Series(plus_dm, index=df.index)
            minus_dm = pd.Series(minus_dm, index=df.index)
            
            # Calculate smoothed True Range and Directional Movements
            atr = tr.rolling(window=length, min_periods=1).mean()
            plus_di = 100 * (plus_dm.rolling(window=length, min_periods=1).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=length, min_periods=1).mean() / atr)
            
            # Calculate ADX
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
            dx = dx.fillna(0)  # Handle division by zero
            adx = dx.rolling(window=length, min_periods=1).mean()
            
            return adx, plus_di, minus_di
            
    except Exception as e:
        logger.error(f"ADX calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate ADX: {e}") from e
