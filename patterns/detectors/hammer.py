"""
patterns/detectors/hammer.py

Hammer pattern detector implementation.
"""

import pandas as pd
from ..base import PatternDetector, PatternResult, PatternType, PatternStrength, validate_dataframe


class HammerPattern(PatternDetector):
    """
    Hammer pattern detector with enhanced validation.
    
    The Hammer is a bullish reversal pattern characterized by:
    - Small body at the top of the trading range
    - Long lower shadow (>= 2x body size) 
    - Little to no upper shadow
    - Usually appears after a downtrend
    """
    
    def __init__(self):
        super().__init__(
            name="Hammer",            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=1
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """
        Detect Hammer pattern with confidence scoring.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            PatternResult with detection results
        """
        # Get price data using normalized column access
        price_data = self._get_price_data(df, -1)
        
        # Extract OHLC values
        open_price = price_data.get('open', 0)
        high_price = price_data.get('high', 0)
        low_price = price_data.get('low', 0)
        close_price = price_data.get('close', 0)
        
        body = abs(close_price - open_price)
        lower_wick = min(open_price, close_price) - low_price
        upper_wick = high_price - max(open_price, close_price)
        
        if body == 0:  # Avoid division by zero
            confidence = 0.0
            detected = False
            strength = PatternStrength.WEAK
        else:
            # Calculate confidence based on pattern quality
            lower_wick_ratio = lower_wick / body if body > 0 else 0
            upper_wick_ratio = upper_wick / body if body > 0 else 0
            
            # Core pattern requirements
            has_long_lower_wick = lower_wick_ratio >= 2.0
            has_small_upper_wick = upper_wick_ratio <= 1.0
            
            detected = has_long_lower_wick and has_small_upper_wick
            
            if detected:
                # Calculate confidence (0.5 to 1.0 for valid patterns)
                wick_quality = min(lower_wick_ratio / 3.0, 1.0)  # Normalize to max 1.0
                shadow_quality = max(0.0, 1.0 - upper_wick_ratio)
                confidence = 0.5 + 0.5 * (wick_quality * 0.7 + shadow_quality * 0.3)
                strength = self._determine_strength(confidence)
            else:
                confidence = 0.0
                strength = PatternStrength.WEAK
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Bullish reversal pattern with long lower shadow and small body at top",
            min_rows_required=self.min_rows
        )
