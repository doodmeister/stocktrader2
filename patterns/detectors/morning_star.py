"""
patterns/detectors/morning_star.py

Morning Star pattern detector implementation.
"""

import pandas as pd
from ..base import PatternDetector, PatternResult, PatternType, PatternStrength, validate_dataframe


class MorningStarPattern(PatternDetector):
    """
    Morning Star pattern detector.
    
    A Morning Star is a three-candle bullish reversal pattern:
    - First candle: Bearish (red) with significant body
    - Second candle: Small body (indecision), gaps down from first
    - Third candle: Bullish (green), closes above midpoint of first candle
    """
    
    def __init__(self):
        super().__init__(
            name="Morning Star",
            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=3
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """
        Detect Morning Star pattern.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            PatternResult with detection results
        """
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        # First candle must be bearish with decent body
        first_bearish = first['Close'] < first['Open']
        first_body = abs(first['Open'] - first['Close'])
        
        # Second candle should be small (indecision)
        second_body = abs(second['Open'] - second['Close'])
        second_range = second['High'] - second['Low']
        second_small = second_body < first_body * 0.5  # Body less than half of first
        
        # Third candle must be bullish
        third_bullish = third['Close'] > third['Open']
        
        # Third candle should close above first candle's midpoint
        first_midpoint = (first['Open'] + first['Close']) / 2
        third_closes_high = third['Close'] > first_midpoint
        
        # Check for gaps (optional but strengthens pattern)
        gap_down = second['High'] < first['Low']
        gap_up = third['Low'] > second['High']
        
        detected = (first_bearish and second_small and third_bullish and third_closes_high)
        
        if detected:
            # Calculate confidence based on pattern quality
            confidence = 0.6  # Base confidence
            
            # Bonus for gaps
            if gap_down:
                confidence += 0.1
            if gap_up:
                confidence += 0.1
                
            # Bonus for strong third candle
            third_body = abs(third['Open'] - third['Close'])
            if third_body > first_body * 0.5:
                confidence += 0.1
                
            confidence = min(0.9, confidence)
        else:
            confidence = 0.0
            
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Three-candle bullish reversal: bearish, small indecision, bullish",
            min_rows_required=self.min_rows
        )
