"""
Patterns package for candlestick pattern detection.

This package provides comprehensive candlestick pattern detection
functionality for the StockTrader application with modular architecture.
"""

# Base classes and types
from .base import (
    PatternType, 
    PatternStrength, 
    PatternResult, 
    PatternDetector,
    PatternDetectionError,
    DataValidationError,
    PatternConfigurationError,
    validate_dataframe,
    performance_monitor
)

# Main orchestrator
from .orchestrator import CandlestickPatterns

# Factory functions  
from .factory import create_pattern_detector, get_available_patterns, create_custom_detector

# Individual detectors
from .detectors import (
    HammerPattern, BullishEngulfingPattern, MorningStarPattern, DojiPattern,
    # Bullish patterns
    PiercingPattern, BullishHaramiPattern, ThreeWhiteSoldiersPattern, 
    InvertedHammerPattern, MorningDojiStarPattern, BullishAbandonedBabyPattern,
    BullishBeltHoldPattern, ThreeInsideUpPattern, RisingWindowPattern,
    # Bearish patterns
    BearishEngulfingPattern, EveningStarPattern, ThreeBlackCrowsPattern,
    BearishHaramiPattern, UpsideGapTwoCrowsPattern
)

__all__ = [
    # Base classes and types
    'PatternType',
    'PatternStrength', 
    'PatternResult',
    'PatternDetector',
    'PatternDetectionError',
    'DataValidationError',
    'PatternConfigurationError',
    'validate_dataframe',
    'performance_monitor',
    
    # Main classes
    'CandlestickPatterns',
    
    # Factory functions
    'create_pattern_detector',
    'get_available_patterns', 
    'create_custom_detector',
    
    # Individual detectors
    'HammerPattern',
    'BullishEngulfingPattern',
    'MorningStarPattern', 
    'DojiPattern',
    # Bullish patterns
    'PiercingPattern',
    'BullishHaramiPattern',
    'ThreeWhiteSoldiersPattern',
    'InvertedHammerPattern',
    'MorningDojiStarPattern',
    'BullishAbandonedBabyPattern',
    'BullishBeltHoldPattern',
    'ThreeInsideUpPattern',
    'RisingWindowPattern',
    # Bearish patterns
    'BearishEngulfingPattern',
    'EveningStarPattern',
    'ThreeBlackCrowsPattern',
    'BearishHaramiPattern',
    'UpsideGapTwoCrowsPattern'
]
