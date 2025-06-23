"""
core/technical_indicators.py

Core technical indicator calculations for financial analysis.

This module provides optimized, centralized implementations of technical indicators
with proper error handling, validation, and fallback implementations.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Union, List
from utils.logger import setup_logger

# Import from the new sub-package
from core.indicators.base import IndicatorError, validate_input
from core.indicators.rsi import calculate_rsi
from core.indicators.macd import calculate_macd
from core.indicators.bollinger_bands import calculate_bollinger_bands
from core.indicators.stochastic import calculate_stochastic
from core.indicators.williams_r import calculate_williams_r
from core.indicators.cci import calculate_cci
from core.indicators.vwap import calculate_vwap, calculate_obv
from core.indicators.adx import calculate_adx


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
    logger.warning("pandas_ta not found - some indicators might use fallback implementations or be unavailable if not part of the new structure")

@validate_input(['high', 'low', 'close'], check_ohlcv_coherence=True)
def calculate_atr(df: DataFrame, length: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with OHLC data
        length: Period for ATR calculation
        
    Returns:
        pd.Series: ATR values    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
            
        if TA_AVAILABLE and ta is not None:
            atr = ta.atr(df['high'], df['low'], df['close'], length=length)
            if atr is None:
                atr = pd.Series([np.nan] * len(df), index=df.index)
        else:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            # Ensure all are pandas Series before concat
            tr = pd.concat([
                high_low.rename("high_low"),
                high_close.rename("high_close"),
                low_close.rename("low_close")
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=length, min_periods=1).mean()
            
        return atr
    except Exception as e:
        logger.error(f"ATR calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate ATR: {e}") from e

@validate_input(['close'])
def calculate_sma(df: DataFrame, length: int = 20, close_col: str = 'close') -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        df: DataFrame with price data
        length: Period for SMA calculation
        close_col: Name of close price column
        
    Returns:
        pd.Series: SMA values
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
            
        return df[close_col].rolling(window=length, min_periods=1).mean()
    except Exception as e:
        logger.error(f"SMA calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate SMA: {e}") from e

@validate_input(['close'])
def calculate_ema(df: DataFrame, length: int = 20, close_col: str = 'close') -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        df: DataFrame with price data
        length: Period for EMA calculation
        close_col: Name of close price column
        
    Returns:
        pd.Series: EMA values
    """
    try:
        if length < 1:
            raise ValueError("Length must be positive")
            
        return df[close_col].ewm(span=length, adjust=False).mean()
    except Exception as e:
        logger.error(f"EMA calculation failed: {e}")
        raise IndicatorError(f"Failed to calculate EMA: {e}") from e

class TechnicalIndicators:
    """
    Main class for technical indicator calculations.
    
    This class provides a unified interface to all technical indicators
    and handles OHLCV data validation and processing.
    """
    
    def __init__(self, data: DataFrame):
        """
        Initialize with OHLCV data.
        
        Args:
            data: DataFrame with OHLC data (columns: Open, High, Low, Close, Volume)
        """
        # Normalize column names to lowercase for consistency
        self.data = data.copy()
        if not self.data.empty:
            # Create a mapping of potential column names to our standard names
            column_mapping = {}
            for col in self.data.columns:
                col_lower = col.lower()
                if col_lower in ['open', 'o']:
                    column_mapping[col] = 'open'
                elif col_lower in ['high', 'h']:
                    column_mapping[col] = 'high'
                elif col_lower in ['low', 'l']:
                    column_mapping[col] = 'low'
                elif col_lower in ['close', 'c']:
                    column_mapping[col] = 'close'
                elif col_lower in ['volume', 'vol', 'v']:
                    column_mapping[col] = 'volume'
            
            # Apply the mapping
            self.data = self.data.rename(columns=column_mapping)
        
        logger.debug(f"TechnicalIndicators initialized with {len(self.data)} rows")    
    # Momentum Indicators
    def calculate_rsi(self, length: int = 14, close_col: str = 'close') -> pd.Series:
        """Calculate RSI using the modular implementation."""
        return calculate_rsi(self.data, length, close_col)
    
    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3, smoothing: int = 3,
                           high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> tuple:
        """Calculate Stochastic Oscillator."""
        return calculate_stochastic(self.data, k_period, d_period, high_col, low_col, close_col)
    
    def calculate_williams_r(self, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        return calculate_williams_r(self.data, period)
    
    def calculate_cci(self, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        return calculate_cci(self.data, period)
    
    # Trend Indicators
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9, close_col: str = 'close') -> tuple:
        """Calculate MACD."""
        return calculate_macd(self.data, fast, slow, signal, close_col)
    
    def calculate_adx(self, period: int = 14) -> tuple:
        """Calculate ADX with +DI and -DI."""
        return calculate_adx(self.data, period)
    
    def calculate_atr(self, length: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        return calculate_atr(self.data, length)
    
    def calculate_sma(self, length: int = 20, close_col: str = 'close') -> pd.Series:
        """Calculate Simple Moving Average."""
        return calculate_sma(self.data, length, close_col)
    
    def calculate_ema(self, length: int = 20, close_col: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average."""
        return calculate_ema(self.data, length, close_col)
    
    # Volatility Indicators
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0, close_col: str = 'close') -> tuple:
        """Calculate Bollinger Bands."""
        return calculate_bollinger_bands(self.data, period, std_dev, close_col)
    
    # Volume Indicators
    def calculate_vwap(self) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        return calculate_vwap(self.data)
    
    def calculate_obv(self) -> pd.Series:
        """Calculate On-Balance Volume."""
        return calculate_obv(self.data)
    
    def get_all_indicators(self) -> dict:
        """
        Calculate all available indicators and return as a dictionary.
        
        Returns:
            dict: Dictionary containing all calculated indicators
        """
        indicators = {}
        
        try:
            # Momentum indicators
            indicators['rsi'] = self.calculate_rsi()
            indicators['stoch_k'], indicators['stoch_d'] = self.calculate_stochastic()
            indicators['williams_r'] = self.calculate_williams_r()
            indicators['cci'] = self.calculate_cci()
            
            # Trend indicators
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.calculate_macd()
            indicators['adx'], indicators['plus_di'], indicators['minus_di'] = self.calculate_adx()
            indicators['atr'] = self.calculate_atr()
            indicators['sma_20'] = self.calculate_sma(20)
            indicators['ema_20'] = self.calculate_ema(20)
            
            # Volatility indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'], indicators['bb_bandwidth'] = self.calculate_bollinger_bands()
            
            # Volume indicators
            if 'volume' in self.data.columns:
                indicators['vwap'] = self.calculate_vwap()
                indicators['obv'] = self.calculate_obv()
            
            logger.info(f"Calculated {len(indicators)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise IndicatorError(f"Failed to calculate indicators: {e}") from e
        
        return indicators
