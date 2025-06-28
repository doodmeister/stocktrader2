"""
patterns/base.py

Base classes, enums, exceptions, and validation decorators for candlestick pattern detection.
Provides the foundation for the modular pattern detection system.
"""

import functools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Callable, Optional, Any
import pandas as pd

from utils.logger import setup_logger
from core.validation.dataframe_validation_logic import validate_financial_dataframe

logger = setup_logger(__name__)


class PatternType(Enum):
    """Enumeration of pattern types for better categorization."""
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    CONTINUATION = "continuation"
    INDECISION = "indecision"


class PatternStrength(Enum):
    """Pattern reliability strength indicators."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass(frozen=True)
class PatternResult:
    """Immutable result object for pattern detection."""
    name: str
    detected: bool
    confidence: float  # 0.0 to 1.0
    pattern_type: PatternType
    strength: PatternStrength
    description: str
    min_rows_required: int
    detection_points: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Validate confidence range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


class PatternDetectionError(Exception):
    """Custom exception for pattern detection errors."""
    
    def __init__(self, message: str, pattern_name: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.pattern_name = pattern_name
        self.error_code = error_code
        self.message = message


class DataValidationError(PatternDetectionError):
    """Specific exception for data validation issues."""
    pass


class PatternConfigurationError(PatternDetectionError):
    """Exception for pattern configuration issues."""
    pass


def validate_dataframe(func: Callable) -> Callable:
    """
    Enhanced decorator for comprehensive DataFrame validation using centralized validation system.
    
    Features:
    - Comprehensive OHLC relationship validation
    - Statistical anomaly detection for better pattern reliability
    - Performance optimization with caching
    - Rich error reporting and metadata
    - Configurable validation levels
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Handle both static functions and instance methods
        if len(args) >= 1:
            # For instance methods, df is the second argument (after self)
            if hasattr(args[0], '__class__') and len(args) >= 2:
                df = args[1]  # Instance method: args[0] is self, args[1] is df
            else:
                df = args[0]  # Static function: args[0] is df
        elif 'df' in kwargs:
            df = kwargs['df']
        else:
            raise DataValidationError(
                "No DataFrame parameter found",
                error_code="MISSING_DATAFRAME_PARAM"
            )
          # Use centralized validation system for comprehensive validation
        try:
            # Normalize column names to lowercase for consistent validation
            normalized_df = df.copy()
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['open', 'high', 'low', 'close', 'volume']:
                    column_mapping[col] = col_lower
            
            if column_mapping:
                normalized_df = normalized_df.rename(columns=column_mapping)
                logger.debug(f"Normalized columns for pattern detection: {column_mapping}")
                
                # Now create a DataFrame with both normalized and expected column names
                # This ensures pattern detectors can access both 'close' and 'Close'
                for original, normalized in column_mapping.items():
                    if normalized in ['open', 'high', 'low', 'close']:
                        # Add capitalized version for backward compatibility
                        capitalized = normalized.capitalize()
                        if capitalized not in normalized_df.columns:
                            normalized_df[capitalized] = normalized_df[normalized]
            
            # Determine minimum rows required for this pattern
            min_rows_required = 1  # Default minimum for patterns
            if len(args) >= 1 and hasattr(args[0], '__class__') and hasattr(args[0], 'min_rows'):
                min_rows_required = args[0].min_rows
            elif len(args) >= 1 and hasattr(args[0], 'min_rows'):
                min_rows_required = args[0].min_rows
            
            logger.debug(f"Running pattern validation with min_rows={min_rows_required}")
            
            # Use lowercase columns for validation
            validation_result = validate_financial_dataframe(
                normalized_df, 
                required_columns=['open', 'high', 'low', 'close'],
                check_ohlcv=True,
                detect_anomalies=True,
                min_rows=min_rows_required
            )
            
            if not validation_result.is_valid:
                error_message = "Validation failed"
                if validation_result.errors:
                    error_message = "; ".join(validation_result.errors)
                logger.error(f"Pattern validation failed: {error_message}")
                raise DataValidationError(
                    f"DataFrame validation failed: {error_message}",
                    error_code="VALIDATION_FAILED"
                )
            
            logger.debug("Centralized validation passed for pattern detection")
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise  # Re-raise our custom validation errors
            else:
                logger.error(f"Validation error: {e}")
                raise DataValidationError(
                    f"Validation error: {str(e)}",
                    error_code="VALIDATION_ERROR"
                )
        
        # Call the original function with normalized DataFrame that has both column formats
        if len(args) >= 1 and hasattr(args[0], '__class__') and len(args) >= 2:
            # Instance method: replace df argument
            new_args = (args[0], normalized_df) + args[2:]
            return func(*new_args, **kwargs)
        elif 'df' in kwargs:
            kwargs['df'] = normalized_df
            return func(*args, **kwargs)
        else:
            # First argument is df
            new_args = (normalized_df,) + args[1:]
            return func(*new_args, **kwargs)
    
    return wrapper


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor pattern detection performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            if execution_time > 0.1:  # Log slow operations
                logger.warning(f"{func.__name__} took {execution_time:.3f}s to execute")
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper


def validate_pattern_data(df: pd.DataFrame, enable_statistical_analysis: bool = True) -> Dict[str, Any]:
    """
    Convenience function for direct centralized validation of pattern data.
    
    Args:
        df: DataFrame to validate for pattern detection
        enable_statistical_analysis: Whether to perform statistical anomaly detection
        
    Returns:
        Dict containing validation results and enhanced metadata
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        logger.debug("Running centralized pattern data validation")
        validation_result = validate_financial_dataframe(
            df,
            required_columns=['Open', 'High', 'Low', 'Close'],
            check_ohlcv=True,
            detect_anomalies=enable_statistical_analysis
        )
        
        if not validation_result.is_valid:
            error_message = "Pattern data validation failed"
            if validation_result.errors:
                error_message = "; ".join(validation_result.errors)
            logger.error(f"Pattern data validation failed: {error_message}")
            raise DataValidationError(
                f"Pattern data validation failed: {error_message}",
                error_code="PATTERN_VALIDATION_FAILED"
            )
        
        # Extract metadata
        validation_metadata = {
            'is_valid': validation_result.is_valid,
            'row_count': len(df),
            'column_count': len(df.columns),
        }
        
        logger.debug("Centralized pattern data validation completed successfully")
        return validation_metadata
        
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise  # Re-raise validation errors
        else:
            logger.error(f"Pattern validation error: {e}")
            raise DataValidationError(
                f"Pattern validation error: {str(e)}",
                error_code="PATTERN_VALIDATION_ERROR"
            )


class PatternDetector(ABC):
    """Abstract base class for pattern detectors."""
    
    def __init__(self, name: str, pattern_type: PatternType, min_rows: int = 3):
        """
        Initialize pattern detector.
        
        Args:
            name: Pattern name
            pattern_type: Type of pattern (bullish, bearish, etc.)
            min_rows: Minimum rows required for detection
        """
        self.name = name
        self.pattern_type = pattern_type
        self.min_rows = min_rows
    
    def _get_ohlc_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get mapping of OHLC columns handling both cases.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict mapping standard names to actual column names
        """
        # Create case-insensitive mapping
        col_mapping = {}
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        # Map standard OHLC names to actual column names
        standard_cols = ['open', 'high', 'low', 'close', 'volume']
        for std_col in standard_cols:
            if std_col in df_columns_lower:
                col_mapping[std_col] = df_columns_lower[std_col]
        
        return col_mapping
    
    def _get_price_data(self, df: pd.DataFrame, row_index: int = -1) -> Dict[str, float]:
        """
        Get OHLC price data for a specific row with normalized column access.
        
        Args:
            df: DataFrame with OHLC data
            row_index: Row index to extract data from (default: last row)
            
        Returns:
            Dict with price data using standard lowercase keys
        """
        col_mapping = self._get_ohlc_columns(df)
        row = df.iloc[row_index]
        
        price_data = {}
        for std_col, actual_col in col_mapping.items():
            if actual_col in df.columns:
                price_data[std_col] = row[actual_col]
        
        return price_data
    
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """
        Detect pattern in the given DataFrame.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            PatternResult with detection results
        """
        pass
    
    def _calculate_confidence(self, conditions_met: int, total_conditions: int) -> float:
        """Calculate confidence based on conditions met."""
        if total_conditions == 0:
            return 0.0
        return min(1.0, conditions_met / total_conditions)
    
    def _determine_strength(self, confidence: float) -> PatternStrength:
        """Determine pattern strength based on confidence."""
        if confidence >= 0.9:
            return PatternStrength.VERY_STRONG
        elif confidence >= 0.7:
            return PatternStrength.STRONG
        elif confidence >= 0.5:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK
