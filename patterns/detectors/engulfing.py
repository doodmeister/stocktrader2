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
        Detect Bullish Engulfing pattern across the entire DataFrame.
        
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
        
        # Need at least 2 candles for engulfing pattern
        if len(df) < 2:
            return PatternResult(
                name=self.name,
                detected=False,
                confidence=0.0,
                pattern_type=self.pattern_type,
                strength=PatternStrength.WEAK,
                description="Bullish reversal pattern where second candle engulfs first bearish candle",
                min_rows_required=self.min_rows,
                detection_points=[]
            )
        
        # Scan each pair of consecutive candles
        for i in range(1, len(df)):
            first_data = self._get_price_data(df, i-1)
            second_data = self._get_price_data(df, i)
            
            # First candle must be bearish
            first_bearish = first_data['close'] < first_data['open']
            
            # Second candle must be bullish  
            second_bullish = second_data['close'] > second_data['open']
            
            # Second candle must engulf first candle's body
            engulfs = (second_data['open'] < first_data['close'] and 
                      second_data['close'] > first_data['open'])
            
            detected = first_bearish and second_bullish and engulfs
            
            if detected:
                # Calculate confidence based on how much the second candle engulfs the first
                first_body = abs(first_data['open'] - first_data['close'])
                second_body = abs(second_data['open'] - second_data['close']) 
                engulfing_ratio = second_body / first_body if first_body > 0 else 1.0
                confidence = min(0.9, 0.6 + min(0.3, engulfing_ratio - 1.0))
                strength = self._determine_strength(confidence)
                
                # Get date for this detection point (use second candle's date)
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
                    'details': {
                        'first_candle': {
                            'open': first_data['open'],
                            'high': first_data['high'],
                            'low': first_data['low'],
                            'close': first_data['close']
                        },
                        'second_candle': {
                            'open': second_data['open'],
                            'high': second_data['high'],
                            'low': second_data['low'],
                            'close': second_data['close']
                        },
                        'engulfing_ratio': engulfing_ratio,
                        'first_body': first_body,
                        'second_body': second_body
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
            description="Bullish reversal pattern where second candle engulfs first bearish candle",
            min_rows_required=self.min_rows,
            detection_points=detection_points
        )
