"""
patterns/detectors/doji.py

Doji pattern detector implementation.
"""

import pandas as pd
from ..base import PatternDetector, PatternResult, PatternType, PatternStrength, validate_dataframe


class DojiPattern(PatternDetector):
    """
    Doji pattern detector.
    
    A Doji is an indecision pattern where the opening and closing prices
    are virtually the same, indicating a balance between buyers and sellers.
    The pattern is characterized by:
    - Open and close prices are nearly equal (within 10% of the total range)
    - Can have long upper and/or lower shadows
    """
    
    def __init__(self):
        super().__init__(
            name="Doji",
            pattern_type=PatternType.INDECISION,
            min_rows=1
        )
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """
        Detect Doji pattern.
        
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
        
        # Calculate body size relative to total range
        body_size = abs(open_price - close_price)
        total_range = high_price - low_price
        
        if total_range == 0:
            detected = False
            confidence = 0.0
        else:
            # Doji if body is <= 10% of total range
            body_ratio = body_size / total_range
            detected = body_ratio <= 0.1
            
            if detected:
                # Higher confidence for smaller bodies
                confidence = max(0.5, 1.0 - (body_ratio * 5))  # Maps 0.1 -> 0.5, 0.0 -> 1.0
            else:
                confidence = 0.0
        
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Indecision pattern where open and close prices are nearly equal",
            min_rows_required=self.min_rows
        )
