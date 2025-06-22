"""
patterns/detectors/bearish_patterns.py

Collection of bearish reversal pattern detectors.
"""

import pandas as pd
from ..base import PatternDetector, PatternResult, PatternType, PatternStrength, validate_dataframe


def _is_long_body(row) -> bool:
    """Helper function to determine if a candle has a long body (≥60% of range)."""
    body = abs(row['Close'] - row['Open'])
    total_range = row['High'] - row['Low']
    return total_range > 0 and body >= total_range * 0.6


class BearishEngulfingPattern(PatternDetector):
    """
    Bearish Engulfing pattern detector.
    
    A two-candle bearish reversal pattern where:
    - First candle is bullish
    - Second candle is bearish and completely engulfs the first candle
    - Second candle opens above first candle's close
    - Second candle closes below first candle's open
    """
    
    def __init__(self):
        super().__init__(
            name="Bearish Engulfing",
            pattern_type=PatternType.BEARISH_REVERSAL,
            min_rows=2
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Bearish Engulfing pattern."""
        prev, last = df.iloc[-2], df.iloc[-1]
        
        detected = (
            prev['Close'] > prev['Open'] and  # First candle bullish
            last['Close'] < last['Open'] and  # Second candle bearish
            last['Open'] > prev['Close'] and  # Opens above first close
            last['Close'] < prev['Open']      # Closes below first open
        )
        
        confidence = 0.8 if detected else 0.0
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Two-candle bearish reversal where larger bearish candle engulfs previous bullish candle",
            min_rows_required=self.min_rows
        )


class EveningStarPattern(PatternDetector):
    """
    Evening Star pattern detector.
    
    A three-candle bearish reversal pattern where:
    - First candle is strongly bullish
    - Second candle is small (star) that gaps up
    - Third candle is strongly bearish and closes below first candle's midpoint
    """
    
    def __init__(self):
        super().__init__(
            name="Evening Star",
            pattern_type=PatternType.BEARISH_REVERSAL,
            min_rows=3
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Evening Star pattern."""
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        detected = (
            first['Close'] > first['Open'] and  # First candle bullish
            abs(second['Close'] - second['Open']) < abs(first['Close'] - first['Open']) * 0.3 and  # Small star
            third['Close'] < third['Open'] and  # Third candle bearish
            third['Close'] < (first['Open'] + first['Close']) / 2  # Closes below first midpoint
        )
        
        confidence = 0.85 if detected else 0.0
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Three-candle bearish reversal: bullish, small star, strong bearish",
            min_rows_required=self.min_rows
        )


class ThreeBlackCrowsPattern(PatternDetector):
    """
    Three Black Crows pattern detector.
    
    A three-candle bearish reversal pattern where:
    - Three consecutive bearish candles with long bodies
    - Each candle opens within the previous candle's body
    - Each candle closes lower than the previous candle
    """
    
    def __init__(self):
        super().__init__(
            name="Three Black Crows",
            pattern_type=PatternType.BEARISH_REVERSAL,
            min_rows=3
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Three Black Crows pattern."""
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        detected = (
            first['Close'] < first['Open'] and  # All bearish
            second['Close'] < second['Open'] and
            third['Close'] < third['Open'] and
            second['Open'] < first['Open'] and  # Progressive lower opens
            third['Open'] < second['Open'] and
            second['Close'] < first['Close'] and  # Progressive lower closes
            third['Close'] < second['Close'] and
            _is_long_body(first) and  # All long bodies
            _is_long_body(second) and
            _is_long_body(third)
        )
        
        confidence = 0.8 if detected else 0.0
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Three consecutive bearish candles with progressive lower opens and closes (all long bodies)",
            min_rows_required=self.min_rows
        )


class BearishHaramiPattern(PatternDetector):
    """
    Bearish Harami pattern detector.
    
    A two-candle bearish reversal pattern where:
    - First candle is bullish with a large body
    - Second candle is bearish and contained within first candle's body
    - Indicates potential bearish reversal
    """
    
    def __init__(self):
        super().__init__(
            name="Bearish Harami",
            pattern_type=PatternType.BEARISH_REVERSAL,
            min_rows=2
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Bearish Harami pattern."""
        prev, last = df.iloc[-2], df.iloc[-1]
        
        detected = (
            prev['Close'] > prev['Open'] and  # First candle bullish
            last['Open'] < prev['Close'] and  # Second contained within first
            last['Close'] > prev['Open'] and
            last['Close'] < last['Open']      # Second candle bearish
        )
        
        confidence = 0.65 if detected else 0.0
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Two-candle pattern where small bearish candle is contained within previous bullish candle",
            min_rows_required=self.min_rows
        )


class UpsideGapTwoCrowsPattern(PatternDetector):
    """
    Upside Gap Two Crows pattern detector.
    
    A three-candle bearish reversal pattern where:
    - First candle is long bullish
    - Second candle is small bearish that gaps up from first
    - Third candle is bearish, opens within second body and closes within first body
    """
    
    def __init__(self):
        super().__init__(
            name="Upside Gap Two Crows",
            pattern_type=PatternType.BEARISH_REVERSAL,
            min_rows=3
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Upside Gap Two Crows pattern."""
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        # Candle 1: Long bullish candle
        c1_bullish = first['Close'] > first['Open']
        c1_body = abs(first['Close'] - first['Open'])
        
        # Candle 2: Small bearish candle that gaps up from c1
        c2_bearish = second['Close'] < second['Open']
        c2_body = abs(second['Close'] - second['Open'])
        c2_small = c2_body < 0.6 * c1_body
        c2_gap_up = second['Open'] > first['High']
        
        # Candle 3: Bearish, opens within c2 body and closes within c1 body
        c3_bearish = third['Close'] < third['Open']
        c3_opens_within_c2 = min(second['Open'], second['Close']) < third['Open'] < max(second['Open'], second['Close'])
        c3_closes_within_c1 = min(first['Open'], first['Close']) < third['Close'] < max(first['Open'], first['Close'])
        
        detected = (
            c1_bullish and
            c2_bearish and
            c2_gap_up and
            c2_small and
            c3_bearish and
            c3_opens_within_c2 and
            c3_closes_within_c1
        )
        
        # Confidence calculation
        if detected:
            gap_size = second['Open'] - first['High']
            gap_quality = min(gap_size / c1_body, 1.0)
            confidence = round(0.6 + 0.4 * gap_quality, 3)
        else:
            confidence = 0.0
        
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Bearish reversal pattern: Bullish candle, then two small bearish candles with an upside gap and closing into first candle's body.",
            min_rows_required=self.min_rows
        )
