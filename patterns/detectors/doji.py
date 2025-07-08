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
        Detect Doji pattern across the entire DataFrame.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            PatternResult with all detection points
        """
        detection_points = []
        overall_detected = False
        max_confidence = 0.0
        max_strength = PatternStrength.WEAK
        
        # Ensure we have a date column or index for detection points
        date_col = None
        if 'Date' in df.columns:
            date_col = 'Date'
        elif 'date' in df.columns:
            date_col = 'date'
        elif isinstance(df.index, pd.DatetimeIndex):
            date_col = df.index
        
        # Scan each row in the DataFrame
        for i in range(len(df)):
            # Get price data for current row
            price_data = self._get_price_data(df, i)
            
            # Extract OHLC values
            open_price = price_data.get('open', 0)
            high_price = price_data.get('high', 0)
            low_price = price_data.get('low', 0)
            close_price = price_data.get('close', 0)
            
            # Calculate body size relative to total range
            body_size = abs(open_price - close_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                continue
                
            # Doji if body is <= 10% of total range
            body_ratio = body_size / total_range
            detected = body_ratio <= 0.1
            
            if detected:
                # Higher confidence for smaller bodies
                confidence = max(0.5, 1.0 - (body_ratio * 5))  # Maps 0.1 -> 0.5, 0.0 -> 1.0
                strength = self._determine_strength(confidence)
                
                # Get date for this detection point
                if date_col is not None:
                    if isinstance(date_col, pd.Index):
                        detection_date = str(date_col[i])
                    else:
                        detection_date = str(df.iloc[i][date_col])
                else:
                    detection_date = f"Row {i}"
                
                detection_points.append({
                    'date': detection_date,
                    'confidence': confidence,
                    'strength': strength.name,
                    'start_index': i,
                    'details': {
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'body_size': body_size,
                        'total_range': total_range,
                        'body_ratio': body_ratio
                    }
                })
                
                overall_detected = True
                if confidence > max_confidence:
                    max_confidence = confidence
                    max_strength = strength
        
        return PatternResult(
            name=self.name,
            detected=overall_detected,
            confidence=max_confidence,
            pattern_type=self.pattern_type,
            strength=max_strength,
            description="Indecision pattern where open and close prices are nearly equal",
            min_rows_required=self.min_rows,
            detection_points=detection_points
        )
