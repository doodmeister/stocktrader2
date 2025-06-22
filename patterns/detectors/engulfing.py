"""
patterns/detectors/engulfing.py

Bullish Engulfing pattern detector implementation.
"""

import pandas as pd
from ..base import PatternDetector, PatternResult, PatternType, PatternStrength, validate_dataframe


class BullishEngulfingPattern(PatternDetector):
    """
    Bullish Engulfing pattern detector.
    
    A bullish engulfing pattern is a two-candle reversal pattern where:
    - First candle is bearish (red)
    - Second candle is bullish (green) and completely engulfs the first candle's body
    - Indicates potential upward reversal
    """
    
    def __init__(self):
        super().__init__(
            name="Bullish Engulfing",
            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=2
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """
        Detect Bullish Engulfing pattern.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            PatternResult with detection results
        """
        first, second = df.iloc[-2], df.iloc[-1]
        
        # First candle must be bearish
        first_bearish = first['Close'] < first['Open']
        
        # Second candle must be bullish  
        second_bullish = second['Close'] > second['Open']
        
        # Second candle must engulf first candle's body
        engulfs = (second['Open'] < first['Close'] and 
                  second['Close'] > first['Open'])
        
        detected = first_bearish and second_bullish and engulfs
        
        if detected:
            # Calculate confidence based on how much the second candle engulfs the first
            first_body = abs(first['Open'] - first['Close'])
            second_body = abs(second['Open'] - second['Close']) 
            engulfing_ratio = second_body / first_body if first_body > 0 else 1.0
            confidence = min(0.9, 0.6 + min(0.3, engulfing_ratio - 1.0))
        else:
            confidence = 0.0
            
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Bullish reversal pattern where second candle engulfs first bearish candle",
            min_rows_required=self.min_rows
        )
