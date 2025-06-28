"""
patterns/detectors/bullish_patterns.py

Collection of bullish reversal pattern detectors.
"""

import pandas as pd
from ..base import PatternDetector, PatternResult, PatternType, PatternStrength, validate_dataframe


class PiercingPattern(PatternDetector):
    """
    Piercing Pattern detector.
    
    A two-candle bullish reversal pattern where:
    - First candle is bearish with significant body
    - Second candle opens below first candle's close
    - Second candle closes above midpoint of first candle's body
    - Second candle is bullish
    """
    
    def __init__(self):
        super().__init__(
            name="Piercing Pattern",
            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=2
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Piercing Pattern across the entire DataFrame."""
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
        
        # Need at least 2 candles
        if len(df) < 2:
            return PatternResult(
                name=self.name,
                detected=False,
                confidence=0.0,
                pattern_type=self.pattern_type,
                strength=PatternStrength.WEAK,
                description="Two-candle bullish reversal where second candle pierces into first",
                min_rows_required=self.min_rows,
                detection_points=[]
            )
        
        for i in range(1, len(df)):
            prev_data = self._get_price_data(df, i-1)
            last_data = self._get_price_data(df, i)
            midpoint = (prev_data['open'] + prev_data['close']) / 2
            
            detected = (
                prev_data['close'] < prev_data['open'] and  # First candle bearish
                last_data['open'] < prev_data['close'] and  # Gap down opening
                last_data['close'] > midpoint and      # Closes above midpoint
                last_data['close'] > last_data['open']      # Second candle bullish
            )
            
            if detected:
                confidence = 0.7
                strength = self._determine_strength(confidence)
                
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
                        'prev_candle': {
                            'open': prev_data['open'],
                            'high': prev_data['high'],
                            'low': prev_data['low'],
                            'close': prev_data['close']
                        },
                        'current_candle': {
                            'open': last_data['open'],
                            'high': last_data['high'],
                            'low': last_data['low'],
                            'close': last_data['close']
                        },
                        'midpoint': midpoint
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
            description="Two-candle bullish reversal where second candle pierces into first",
            min_rows_required=self.min_rows,
            detection_points=detection_points
        )


class BullishHaramiPattern(PatternDetector):
    """
    Bullish Harami Pattern detector.
    
    A two-candle pattern where:
    - First candle is bearish with large body
    - Second candle is bullish with smaller body
    - Second candle's body is completely within first candle's body
    """
    
    def __init__(self):
        super().__init__(
            name="Bullish Harami",
            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=2
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Bullish Harami Pattern."""
        prev, last = df.iloc[-2], df.iloc[-1]
        
        # First candle bearish, second candle bullish
        first_bearish = prev['Close'] < prev['Open']
        second_bullish = last['Close'] > last['Open']
        
        # Second candle contained within first candle's body
        contained = (last['Open'] > prev['Close'] and 
                    last['Close'] < prev['Open'])
        
        detected = first_bearish and second_bullish and contained
        
        if detected:
            # Calculate confidence based on relative sizes
            first_body = abs(prev['Open'] - prev['Close'])
            second_body = abs(last['Open'] - last['Close'])
            size_ratio = second_body / first_body if first_body > 0 else 0
            confidence = max(0.6, 0.8 - size_ratio)  # Smaller second candle = higher confidence
        else:
            confidence = 0.0
            
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Two-candle pattern with small bullish candle inside large bearish candle",
            min_rows_required=self.min_rows
        )


class ThreeWhiteSoldiersPattern(PatternDetector):
    """
    Three White Soldiers Pattern detector.
    
    A three-candle bullish continuation pattern with:
    - Three consecutive bullish candles
    - Each candle opens within previous candle's body
    - Each candle closes higher than the previous
    - Limited upper shadows (strong buying pressure)
    """
    
    def __init__(self):
        super().__init__(
            name="Three White Soldiers",
            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=3
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Three White Soldiers Pattern."""
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        # All three candles must be bullish
        all_bullish = (
            first['Close'] > first['Open'] and
            second['Close'] > second['Open'] and
            third['Close'] > third['Open']
        )
        
        # Progressive closes (each higher than previous)
        progressive_closes = (
            second['Close'] > first['Close'] and
            third['Close'] > second['Close']
        )
        
        # Opens within previous body
        opens_within_body = (
            first['Open'] < second['Open'] < first['Close'] and
            second['Open'] < third['Open'] < second['Close']
        )
        
        detected = all_bullish and progressive_closes and opens_within_body
        
        if detected:
            # Check for limited upper shadows (strength indicator)
            first_upper = first['High'] - first['Close']
            second_upper = second['High'] - second['Close'] 
            third_upper = third['High'] - third['Close']
            
            first_body = first['Close'] - first['Open']
            second_body = second['Close'] - second['Open']
            third_body = third['Close'] - third['Open']
            
            avg_upper_ratio = (first_upper/first_body + second_upper/second_body + third_upper/third_body) / 3
            confidence = max(0.7, 0.9 - avg_upper_ratio)
        else:
            confidence = 0.0
            
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Three consecutive bullish candles with progressive opens and closes",
            min_rows_required=self.min_rows
        )


class InvertedHammerPattern(PatternDetector):
    """
    Inverted Hammer Pattern detector.
    
    Similar to hammer but with long upper shadow:
    - Small body at bottom of trading range
    - Long upper shadow (>= 2x body size)
    - Little to no lower shadow
    - Bullish reversal pattern when found at bottom of downtrend
    """
    
    def __init__(self):
        super().__init__(
            name="Inverted Hammer",
            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=1
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Inverted Hammer Pattern."""
        row = df.iloc[-1]
        
        body = abs(row['Close'] - row['Open'])
        upper_wick = row['High'] - max(row['Open'], row['Close'])
        lower_wick = min(row['Open'], row['Close']) - row['Low']
        
        if body == 0:
            confidence = 0.0
            detected = False
        else:
            # Calculate ratios
            upper_wick_ratio = upper_wick / body if body > 0 else 0
            lower_wick_ratio = lower_wick / body if body > 0 else 0
            
            # Core pattern requirements
            has_long_upper_wick = upper_wick_ratio >= 2.0
            has_small_lower_wick = lower_wick_ratio <= 1.0
            
            detected = has_long_upper_wick and has_small_lower_wick
            
            if detected:
                # Calculate confidence
                wick_quality = min(upper_wick_ratio / 3.0, 1.0)
                shadow_quality = max(0.0, 1.0 - lower_wick_ratio)
                confidence = 0.5 + 0.5 * (wick_quality * 0.7 + shadow_quality * 0.3)
            else:
                confidence = 0.0
        
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Bullish reversal pattern with long upper shadow and small body at bottom",
            min_rows_required=self.min_rows
        )


class MorningDojiStarPattern(PatternDetector):
    """
    Morning Doji Star Pattern detector.
    
    A three-candle pattern similar to Morning Star but with Doji:
    - First candle: Bearish with significant body
    - Second candle: Doji (indecision) that gaps down
    - Third candle: Bullish that closes above first candle's midpoint
    """
    
    def __init__(self):
        super().__init__(
            name="Morning Doji Star",
            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=3
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Morning Doji Star Pattern."""
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        # First candle bearish
        first_bearish = first['Close'] < first['Open']
        
        # Second candle is Doji
        second_body = abs(second['Open'] - second['Close'])
        second_range = second['High'] - second['Low']
        is_doji = second_body <= (second_range * 0.1) if second_range > 0 else False
        
        # Third candle bullish
        third_bullish = third['Close'] > third['Open']
        
        # Third closes above first's midpoint
        first_midpoint = (first['Open'] + first['Close']) / 2
        third_closes_high = third['Close'] > first_midpoint
        
        detected = first_bearish and is_doji and third_bullish and third_closes_high
        
        confidence = 0.85 if detected else 0.0
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Morning Star pattern with Doji as the middle candle",
            min_rows_required=self.min_rows
        )


class BullishAbandonedBabyPattern(PatternDetector):
    """
    Bullish Abandoned Baby pattern detector.
    
    A rare three-candle bullish reversal pattern where:
    - First candle is bearish
    - Second candle is a Doji that gaps down from first candle
    - Third candle is bullish and gaps up from the Doji
    - The Doji is isolated (gaps on both sides)
    """
    
    def __init__(self):
        super().__init__(
            name="Bullish Abandoned Baby",
            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=3
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Bullish Abandoned Baby pattern."""
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        detected = (
            first['Close'] < first['Open'] and  # First candle bearish
            abs(second['Open'] - second['Close']) < (second['High'] - second['Low']) * 0.1 and  # Doji
            second['Low'] > first['High'] and  # Gap down from first
            third['Open'] > second['High'] and  # Gap up from second
            third['Close'] > third['Open']  # Third candle bullish
        )
        
        confidence = 0.9 if detected else 0.0
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Rare three-candle reversal with isolated Doji in the middle",
            min_rows_required=self.min_rows
        )


class BullishBeltHoldPattern(PatternDetector):
    """
    Bullish Belt Hold pattern detector.
    
    A single-candle bullish reversal pattern where:
    - Strong bullish candle that opens at the low
    - The candle has a long body with minimal upper shadow
    - Indicates strong bullish momentum
    """
    
    def __init__(self):
        super().__init__(
            name="Bullish Belt Hold",
            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=1
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Bullish Belt Hold pattern."""
        row = df.iloc[-1]
        
        detected = (
            row['Close'] > row['Open'] and  # Bullish candle
            row['Open'] == row['Low'] and  # Opens at the low
            (row['Close'] - row['Open']) > (row['High'] - row['Close']) * 0.5  # Strong body
        )
        
        confidence = 0.7 if detected else 0.0
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Bullish candle opening at the low with strong upward momentum",
            min_rows_required=self.min_rows
        )


class ThreeInsideUpPattern(PatternDetector):
    """
    Three Inside Up pattern detector.
    
    A three-candle bullish reversal pattern where:
    - First candle is bearish
    - Second candle is a bullish Harami (contained within first candle)
    - Third candle is bullish and closes above second candle's close
    """
    
    def __init__(self):
        super().__init__(
            name="Three Inside Up",
            pattern_type=PatternType.BULLISH_REVERSAL,
            min_rows=3
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Three Inside Up pattern."""
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        detected = (
            first['Close'] < first['Open'] and  # First candle bearish
            second['Open'] > first['Close'] and  # Harami setup
            second['Close'] < first['Open'] and  # Contained within first
            second['Close'] > second['Open'] and  # Second candle bullish
            third['Close'] > second['Close']  # Confirmation
        )
        
        confidence = 0.75 if detected else 0.0
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Three-candle pattern: Bullish Harami followed by confirmation candle",
            min_rows_required=self.min_rows
        )


class RisingWindowPattern(PatternDetector):
    """
    Rising Window pattern detector.
    
    A two-candle bullish continuation pattern where:
    - There's a gap up between two candles
    - Previous candle's high is below current candle's low
    - Indicates bullish continuation
    """
    
    def __init__(self):
        super().__init__(
            name="Rising Window",
            pattern_type=PatternType.CONTINUATION,  # Bullish continuation
            min_rows=2
        )
    
    @validate_dataframe
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Rising Window pattern."""
        prev, last = df.iloc[-2], df.iloc[-1]
        
        detected = prev['High'] < last['Low']  # Gap up
        confidence = 0.65 if detected else 0.0
        strength = self._determine_strength(confidence)
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=strength,
            description="Gap up pattern indicating bullish continuation",
            min_rows_required=self.min_rows
        )
