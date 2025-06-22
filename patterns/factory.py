"""
patterns/factory.py

Factory functions for creating pattern detectors and orchestrators.
"""

from typing import List, Optional
from .orchestrator import CandlestickPatterns
from .base import PatternDetector


def create_pattern_detector(
    confidence_threshold: float = 0.7, 
    enable_parallel: bool = True,
    pattern_names: Optional[List[str]] = None
) -> CandlestickPatterns:
    """
    Factory function to create a configured CandlestickPatterns detector.
    
    Args:
        confidence_threshold: Minimum confidence for pattern detection
        enable_parallel: Whether to enable parallel pattern detection
        pattern_names: List of specific patterns to include (None for all)
        
    Returns:
        Configured CandlestickPatterns instance
    """
    orchestrator = CandlestickPatterns(
        confidence_threshold=confidence_threshold,
        enable_parallel=enable_parallel
    )
    
    # If specific patterns requested, remove others
    if pattern_names is not None:
        all_patterns = list(orchestrator.detectors.keys())
        for pattern in all_patterns:
            if pattern not in pattern_names:
                orchestrator.remove_detector(pattern)
    
    return orchestrator


def get_available_patterns() -> List[str]:
    """
    Get list of all available pattern names.
    
    Returns:
        List of pattern names
    """
    # Create a temporary orchestrator to get pattern names
    temp_orchestrator = CandlestickPatterns()
    return temp_orchestrator.get_pattern_names()


def create_custom_detector(
    patterns: List[PatternDetector],
    confidence_threshold: float = 0.7,
    enable_parallel: bool = True
) -> CandlestickPatterns:
    """
    Create an orchestrator with custom pattern detectors.
    
    Args:
        patterns: List of PatternDetector instances
        confidence_threshold: Minimum confidence threshold
        enable_parallel: Whether to enable parallel processing
        
    Returns:
        CandlestickPatterns with custom detectors
    """
    orchestrator = CandlestickPatterns(
        confidence_threshold=confidence_threshold,
        enable_parallel=enable_parallel
    )
    
    # Clear default detectors and add custom ones
    orchestrator.detectors.clear()
    
    for pattern in patterns:
        orchestrator.add_detector(pattern.name.lower().replace(' ', '_'), pattern)
    
    return orchestrator
