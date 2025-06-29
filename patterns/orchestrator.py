"""
patterns/orchestrator.py

Main orchestrator for candlestick pattern detection.
Manages multiple pattern detectors and provides unified interface.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading

from .base import PatternDetector, PatternResult, PatternDetectionError, logger
from .detectors import (
    HammerPattern, BullishEngulfingPattern, MorningStarPattern, DojiPattern,
    PiercingPattern, BullishHaramiPattern, ThreeWhiteSoldiersPattern,
    InvertedHammerPattern, MorningDojiStarPattern, BullishAbandonedBabyPattern,
    BullishBeltHoldPattern, ThreeInsideUpPattern, RisingWindowPattern,
    BearishEngulfingPattern, EveningStarPattern, ThreeBlackCrowsPattern,
    BearishHaramiPattern, UpsideGapTwoCrowsPattern
)


class CandlestickPatterns:
    """
    Main orchestrator for candlestick pattern detection.
    
    Manages multiple pattern detectors and provides a unified interface
    for detecting patterns in OHLCV data with performance optimization.
    """
    
    def __init__(self, confidence_threshold: float = 0.7, enable_parallel: bool = True):
        """
        Initialize the pattern detection system.
        
        Args:
            confidence_threshold: Minimum confidence for pattern detection
            enable_parallel: Whether to enable parallel pattern detection
        """
        self.confidence_threshold = confidence_threshold
        self.enable_parallel = enable_parallel
        self._cache = {}
        self._cache_lock = threading.RLock()
          # Initialize pattern detectors
        self.detectors: Dict[str, PatternDetector] = {
            # Basic patterns
            'hammer': HammerPattern(),
            'bullish_engulfing': BullishEngulfingPattern(),
            'morning_star': MorningStarPattern(),
            'doji': DojiPattern(),
            
            # Bullish patterns
            'piercing_pattern': PiercingPattern(),
            'bullish_harami': BullishHaramiPattern(),
            'three_white_soldiers': ThreeWhiteSoldiersPattern(),
            'inverted_hammer': InvertedHammerPattern(),
            'morning_doji_star': MorningDojiStarPattern(),
            'bullish_abandoned_baby': BullishAbandonedBabyPattern(),
            'bullish_belt_hold': BullishBeltHoldPattern(),
            'three_inside_up': ThreeInsideUpPattern(),
            'rising_window': RisingWindowPattern(),
            
            # Bearish patterns
            'bearish_engulfing': BearishEngulfingPattern(),
            'evening_star': EveningStarPattern(),
            'three_black_crows': ThreeBlackCrowsPattern(),
            'bearish_harami': BearishHaramiPattern(),
            'upside_gap_two_crows': UpsideGapTwoCrowsPattern(),
        }
        
        logger.info(f"Initialized CandlestickPatterns with {len(self.detectors)} detectors")
    
    def get_pattern_occurrences(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detects all pattern occurrences and returns them as a list.

        This method relies on the individual detectors populating the 
        `detection_points` field in the PatternResult.

        Args:
            df: OHLCV DataFrame.

        Returns:
            A list of dictionaries, each representing a single pattern occurrence.
        """
        all_occurrences = []
        detection_results = self.detect_all_patterns(df)

        for name, result in detection_results.items():
            if result.detected and result.detection_points:
                for point in result.detection_points:
                    occurrence = {
                        "date": point.get("date"),
                        "pattern_name": name,
                        "name": result.name,  # for frontend compatibility
                        "signal": result.pattern_type.value,  # for frontend compatibility
                        "confidence": result.confidence,
                        "pattern_type": result.pattern_type.value,
                        "description": result.description,
                        # Try to provide start_index if available in detection point
                        "start_index": point.get("start_index") if "start_index" in point else None
                    }
                    all_occurrences.append(occurrence)
        
        # Sort by date, assuming 'date' is a datetime object or string that sorts correctly
        if all_occurrences:
            all_occurrences.sort(key=lambda x: x['date'], reverse=True)
            
        return all_occurrences

    def detect_all_patterns(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """
        Detect all patterns in the given DataFrame.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary mapping pattern names to PatternResult objects
        """
        if self.enable_parallel:
            return self._detect_parallel(df)
        else:
            return self._detect_sequential(df)
    
    def _detect_sequential(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Detect patterns sequentially."""
        results = {}
        
        for name, detector in self.detectors.items():
            try:
                result = detector.detect(df)
                results[name] = result
                
                if result.detected and result.confidence >= self.confidence_threshold:
                    logger.debug(f"Detected {name} pattern with confidence {result.confidence:.2f}")
                    
            except Exception as e:
                logger.error(f"Error detecting {name} pattern: {e}")
                # Create a failed result
                results[name] = PatternResult(
                    name=detector.name,
                    detected=False,
                    confidence=0.0,
                    pattern_type=detector.pattern_type,
                    strength=detector._determine_strength(0.0),
                    description=f"Detection failed: {str(e)}",
                    min_rows_required=detector.min_rows
                )
        
        return results
    
    def _detect_parallel(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Detect patterns in parallel."""
        results = {}
        
        def detect_pattern(name_detector_pair):
            name, detector = name_detector_pair
            try:
                result = detector.detect(df)
                return name, result
            except Exception as e:
                logger.error(f"Error detecting {name} pattern: {e}")
                return name, PatternResult(
                    name=detector.name,
                    detected=False,
                    confidence=0.0,
                    pattern_type=detector.pattern_type,
                    strength=detector._determine_strength(0.0),
                    description=f"Detection failed: {str(e)}",
                    min_rows_required=detector.min_rows
                )
        
        # Use ThreadPoolExecutor for I/O bound pattern detection
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_results = executor.map(detect_pattern, self.detectors.items())
            
            for name, result in future_results:
                results[name] = result
                
                if result.detected and result.confidence >= self.confidence_threshold:
                    logger.debug(f"Detected {name} pattern with confidence {result.confidence:.2f}")
        
        return results
    
    def detect_specific_pattern(self, df: pd.DataFrame, pattern_name: str) -> Optional[PatternResult]:
        """
        Detect a specific pattern.
        
        Args:
            df: OHLCV DataFrame
            pattern_name: Name of the pattern to detect
            
        Returns:
            PatternResult if pattern exists, None otherwise
        """
        if pattern_name not in self.detectors:
            raise PatternDetectionError(f"Unknown pattern: {pattern_name}")
        
        return self.detectors[pattern_name].detect(df)
    
    def get_pattern_names(self) -> List[str]:
        """Get list of available pattern names."""
        return list(self.detectors.keys())
    
    def get_detector_by_name(self, name: str) -> Optional[PatternDetector]:
        """Get detector by name."""
        return self.detectors.get(name)
    
    def add_detector(self, name: str, detector: PatternDetector) -> None:
        """
        Add a custom pattern detector.
        
        Args:
            name: Unique name for the detector
            detector: PatternDetector instance
        """
        if name in self.detectors:
            logger.warning(f"Overriding existing detector: {name}")
        
        self.detectors[name] = detector
        logger.info(f"Added detector: {name}")
    
    def remove_detector(self, name: str) -> bool:
        """
        Remove a pattern detector.
        
        Args:
            name: Name of the detector to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.detectors:
            del self.detectors[name]
            logger.info(f"Removed detector: {name}")
            return True
        return False
    
    def get_bullish_patterns(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Get only bullish patterns."""
        all_results = self.detect_all_patterns(df)
        return {
            name: result for name, result in all_results.items() 
            if result.pattern_type.value == "bullish_reversal" and result.detected
        }
    
    def get_bearish_patterns(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Get only bearish patterns.""" 
        all_results = self.detect_all_patterns(df)
        return {
            name: result for name, result in all_results.items()
            if result.pattern_type.value == "bearish_reversal" and result.detected
        }
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        with self._cache_lock:
            self._cache.clear()
            logger.debug("Pattern detection cache cleared")
