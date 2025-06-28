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
        Detect Morning Star pattern across the entire DataFrame.
        
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
        
        # Need at least 3 candles for morning star pattern
        if len(df) < 3:
            return PatternResult(
                name=self.name,
                detected=False,
                confidence=0.0,
                pattern_type=self.pattern_type,
                strength=PatternStrength.WEAK,
                description="Three-candle bullish reversal: bearish, small indecision, bullish",
                min_rows_required=self.min_rows,
                detection_points=[]
            )
        
        # Scan each triplet of consecutive candles
        for i in range(2, len(df)):
            first_data = self._get_price_data(df, i-2)
            second_data = self._get_price_data(df, i-1)
            third_data = self._get_price_data(df, i)
            
            # First candle must be bearish with decent body
            first_bearish = first_data['close'] < first_data['open']
            first_body = abs(first_data['open'] - first_data['close'])
            
            # Second candle should be small (indecision)
            second_body = abs(second_data['open'] - second_data['close'])
            second_small = second_body < first_body * 0.5  # Body less than half of first
            
            # Third candle must be bullish
            third_bullish = third_data['close'] > third_data['open']
            
            # Third candle should close above first candle's midpoint
            first_midpoint = (first_data['open'] + first_data['close']) / 2
            third_closes_high = third_data['close'] > first_midpoint
            
            # Check for gaps (optional but strengthens pattern)
            gap_down = second_data['high'] < first_data['low']
            gap_up = third_data['low'] > second_data['high']
            
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
                third_body = abs(third_data['open'] - third_data['close'])
                if third_body > first_body * 0.5:
                    confidence += 0.1
                    
                confidence = min(0.9, confidence)
                strength = self._determine_strength(confidence)
                
                # Get date for this detection point (use third candle's date)
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
                        'third_candle': {
                            'open': third_data['open'],
                            'high': third_data['high'],
                            'low': third_data['low'],
                            'close': third_data['close']
                        },
                        'first_body': first_body,
                        'second_body': second_body,
                        'third_body': third_body,
                        'gap_down': gap_down,
                        'gap_up': gap_up,
                        'first_midpoint': first_midpoint
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
            description="Three-candle bullish reversal: bearish, small indecision, bullish",
            min_rows_required=self.min_rows,
            detection_points=detection_points
        )
