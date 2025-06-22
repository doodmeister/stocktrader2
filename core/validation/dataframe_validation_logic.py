"""
DataFrame validation logic for the StockTrader Bot.

This module provides comprehensive validation logic for financial data
including OHLCV validation, anomaly detection, and data quality checks.
"""
import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .validation_config import ValidationConfig
from .validation_results import DataFrameValidationResult

# Configure logging
logger = logging.getLogger(__name__)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class DataFrameValidator:
    """
    Comprehensive DataFrame validator for financial market data.
    
    Provides validation for OHLCV data, technical indicators, anomaly detection,
    and data quality checks. Designed for integration with FastAPI backend.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the DataFrame validator.
        
        Args:
            config: Optional validation configuration. Uses default if None.
        """
        self.config = config or ValidationConfig()
        self.logger = logger
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        check_ohlcv: bool = True,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        detect_anomalies: bool = False,
        anomaly_detection_level: str = "basic",
        max_null_percentage: float = 0.05,
        symbol: Optional[str] = None
    ) -> DataFrameValidationResult:
        """
        Perform comprehensive DataFrame validation.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            check_ohlcv: Whether to validate OHLCV relationships
            min_rows: Minimum required rows (uses config default if None)
            max_rows: Maximum allowed rows (uses config default if None)
            detect_anomalies: Whether to detect anomalies
            anomaly_detection_level: Level of anomaly detection ('basic' or 'advanced')
            max_null_percentage: Maximum allowed null percentage
            symbol: Optional symbol for context in error messages
            
        Returns:
            DataFrameValidationResult with comprehensive validation results
        """
        start_time = time.time()
          # Initialize result with all required fields
        result = DataFrameValidationResult(
            is_valid=True,
            errors=None,
            warnings=[],
            details=None,
            dataframe_shape=df.shape,
            validation_timestamp=datetime.now(),
            error_details=None,
            validated_data=None,
            missing_columns=[],
            duplicate_columns=[],
            extra_columns=[],
            nan_counts=None,
            null_percentages=None,
            data_types=None,
            validation_time_seconds=None,
            rows_validated=None,
            columns_validated=None,
            ohlc_validation_passed=None,
            invalid_ohlc_rows=None,
            anomalies_detected=None,
            anomaly_details=None,
            failed_rows_sample=None
        )
        
        try:
            # Basic DataFrame checks
            self._validate_basic_structure(df, result, min_rows, max_rows)
            
            # Column validation
            if required_columns:
                self._validate_columns(df, required_columns, result)
            
            # Data quality checks
            self._validate_data_quality(df, result, max_null_percentage)
            
            # Data type validation
            self._validate_data_types(df, result)
            
            # OHLCV specific validation
            if check_ohlcv:
                self._validate_ohlcv_data(df, result)
            
            # Anomaly detection
            if detect_anomalies:
                self._detect_anomalies(df, result, anomaly_detection_level)
            
            # Set validation metrics
            result.validation_time_seconds = time.time() - start_time
            result.rows_validated = len(df)
            result.columns_validated = len(df.columns)
            
            # Set validated data if successful
            if result.is_valid:
                result.validated_data = df.copy()
                self.logger.info(f"DataFrame validation passed for shape {df.shape}")
            else:
                self.logger.warning(f"DataFrame validation failed: {result.get_summary()}")
                
        except Exception as e:
            result.add_error(f"Validation failed with exception: {str(e)}")
            self.logger.error(f"DataFrame validation error: {str(e)}", exc_info=True)
        
        return result
    
    def _validate_basic_structure(
        self,
        df: pd.DataFrame,
        result: DataFrameValidationResult,
        min_rows: Optional[int],
        max_rows: Optional[int]
    ) -> None:
        """Validate basic DataFrame structure."""
        # Check if DataFrame is empty
        if df.empty:
            result.add_error("DataFrame is empty")
            return
        
        # Check minimum rows
        min_required = min_rows or self.config.MIN_DATASET_SIZE
        if len(df) < min_required:
            result.add_error(
                self.config.ERROR_MESSAGES['insufficient_data'].format(
                    size=len(df), min_size=min_required
                )
            )
        
        # Check maximum rows
        max_allowed = max_rows or self.config.MAX_DATASET_SIZE
        if len(df) > max_allowed:
            result.add_error(
                self.config.ERROR_MESSAGES['data_too_large'].format(
                    size=len(df), max_size=max_allowed
                )
            )
          # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            result.duplicate_columns = duplicate_cols
            result.add_error(f"Duplicate columns found: {duplicate_cols}")
    
    def _validate_columns(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        result: DataFrameValidationResult
    ) -> None:
        """Validate required columns are present (case-insensitive)."""
        # Create case-insensitive mappings
        df_columns_lower = {col.lower(): col for col in df.columns}
        required_lower = {col.lower(): col for col in required_columns}
        
        # Check for missing columns (case-insensitive)
        missing = []
        for req_col in required_columns:
            if req_col.lower() not in df_columns_lower:
                missing.append(req_col)
        
        if missing:
            result.missing_columns = missing
            result.add_error(
                self.config.ERROR_MESSAGES['missing_columns'].format(
                    columns=missing
                )
            )
        
        # Identify extra columns (columns not in required list)
        # Only consider columns as "extra" if they don't match any required column (case-insensitive)
        extra = []
        for df_col in df.columns:
            if df_col.lower() not in required_lower:
                extra.append(df_col)
        
        if extra:
            result.extra_columns = extra
            result.add_warning(f"Extra columns present: {extra}")
    
    def _validate_data_quality(
        self,
        df: pd.DataFrame,
        result: DataFrameValidationResult,
        max_null_percentage: float
    ) -> None:
        """Validate data quality metrics."""
        # Calculate null counts and percentages
        nan_counts = df.isnull().sum().to_dict()
        null_percentages = (df.isnull().sum() / len(df)).to_dict()
        
        result.nan_counts = {col: int(count) for col, count in nan_counts.items()}
        result.null_percentages = null_percentages
        
        # Check null percentage thresholds
        for col, null_pct in null_percentages.items():
            # Use higher tolerance for indicator columns
            threshold = (
                self.config.INDICATOR_NULL_TOLERANCE 
                if self.config.is_indicator_column(col)
                else max_null_percentage
            )
            
            if null_pct > threshold:
                result.add_error(
                    f"Column '{col}' has {null_pct:.2%} null values, "
                    f"exceeding threshold of {threshold:.2%}"                )        
        # Check for completely empty columns
        empty_cols = [col for col, count in nan_counts.items() if count == len(df)]
        if empty_cols:
            result.add_error(f"Completely empty columns: {empty_cols}")
    
    def _validate_data_types(
        self,
        df: pd.DataFrame,
        result: DataFrameValidationResult
    ) -> None:
        """Validate and record data types."""
        result.data_types = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
        
        # Check for object columns that should be numeric (case-insensitive)
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        price_columns_lower = [col.lower() for col in price_columns]
        
        # Create mapping from lowercase to actual column names
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        for price_col in price_columns:
            price_col_lower = price_col.lower()
            actual_col = df_columns_lower.get(price_col_lower)
            
            if actual_col and df[actual_col].dtype == 'object':
                # Try to detect if this should be numeric
                non_null_sample = df[actual_col].dropna().head(10)
                if not non_null_sample.empty:
                    try:
                        pd.to_numeric(non_null_sample)
                        result.add_warning(
                            f"Column '{actual_col}' has object dtype but appears to contain numeric data"                        )
                    except (ValueError, TypeError):
                        pass  # It's legitimately non-numeric
    
    def _validate_ohlcv_data(
        self,
        df: pd.DataFrame,
        result: DataFrameValidationResult
    ) -> None:
        """Validate OHLCV relationships and constraints (case-insensitive)."""
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        
        # Create case-insensitive mapping
        df_columns_lower = {col.lower(): col for col in df.columns}
        actual_ohlc_columns = []
        missing_ohlc = []
        
        for ohlc_col in ohlc_columns:
            actual_col = df_columns_lower.get(ohlc_col.lower())
            if actual_col:
                actual_ohlc_columns.append(actual_col)
            else:
                missing_ohlc.append(ohlc_col)
        
        if missing_ohlc:
            result.add_warning(f"OHLC validation skipped - missing columns: {missing_ohlc}")
            result.ohlc_validation_passed = None
            return
        
        invalid_rows = []
        
        try:
            # Convert to numeric and handle any conversion errors
            ohlc_data = df[actual_ohlc_columns].apply(pd.to_numeric, errors='coerce')
            
            for idx, row in ohlc_data.iterrows():
                open_price, high_price, low_price, close_price = row
                
                # Skip rows with NaN values
                if pd.isna([open_price, high_price, low_price, close_price]).any():
                    continue
                
                # Validate OHLC relationships
                violations = []
                
                # High should be >= all other prices
                if high_price < max(open_price, low_price, close_price):
                    violations.append(f"High ({high_price}) < max(O/L/C)")
                
                # Low should be <= all other prices
                if low_price > min(open_price, high_price, close_price):
                    violations.append(f"Low ({low_price}) > min(O/H/C)")
                
                # Check for negative prices
                if any(price < 0 for price in [open_price, high_price, low_price, close_price]):
                    violations.append("Negative price detected")
                
                # Check for zero prices (usually invalid for stocks)
                if any(price == 0 for price in [open_price, high_price, low_price, close_price]):
                    violations.append("Zero price detected")
                
                if violations:
                    invalid_rows.append(idx)
                    result.add_error(f"Row {idx} OHLC violation: {'; '.join(violations)}")
              # Volume validation if present (case-insensitive)
            volume_col = df_columns_lower.get('volume')
            if volume_col:
                volume_data = pd.to_numeric(df[volume_col], errors='coerce')
                negative_volume = volume_data < 0
                if negative_volume.any():
                    neg_indices = df.index[negative_volume].tolist()
                    result.add_error(f"Negative volume at rows: {neg_indices}")
                    invalid_rows.extend(neg_indices)
            
            # Set OHLC validation results
            result.ohlc_validation_passed = len(invalid_rows) == 0
            result.invalid_ohlc_rows = invalid_rows
            
            # Check if percentage of invalid rows exceeds threshold
            if len(invalid_rows) > 0:
                invalid_percentage = len(invalid_rows) / len(df)
                if invalid_percentage > self.config.MAX_INVALID_OHLC_PERCENTAGE:
                    result.add_error(
                        f"Invalid OHLC percentage ({invalid_percentage:.2%}) exceeds "
                        f"threshold ({self.config.MAX_INVALID_OHLC_PERCENTAGE:.2%})"
                    )
                else:
                    result.add_warning(
                        f"Found {len(invalid_rows)} rows with OHLC violations "
                        f"({invalid_percentage:.2%} of data)"
                    )
        
        except Exception as e:
            result.add_error(f"OHLC validation failed: {str(e)}")
            result.ohlc_validation_passed = False
    
    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        result: DataFrameValidationResult,
        detection_level: str = "basic"
    ) -> None:
        """Detect anomalies in the data."""
        anomalies = {}
        total_anomalies = 0
        
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                col_data = df[col].dropna()
                if len(col_data) < 10:  # Need sufficient data for anomaly detection
                    continue
                
                col_anomalies = []
                
                if detection_level == "basic":
                    # Simple outlier detection using IQR
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                    if outliers:
                        col_anomalies.extend(outliers)
                
                elif detection_level == "advanced":
                    # More sophisticated anomaly detection
                    # Z-score method
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    z_outliers = df[z_scores > 3].index.tolist()
                    col_anomalies.extend(z_outliers)                    # Modified Z-score method (using median)
                    median_val = float(col_data.median())
                    mad_val = float(np.median(np.abs(col_data - median_val)))
                    if mad_val != 0:
                        modified_z_scores = np.abs(0.6745 * (col_data - median_val) / mad_val)
                        mad_outliers = col_data.index[modified_z_scores > 3.5].tolist()
                        col_anomalies.extend(mad_outliers)
                
                # Remove duplicates and add to results
                col_anomalies = list(set(col_anomalies))
                if col_anomalies:
                    anomalies[col] = col_anomalies
                    total_anomalies += len(col_anomalies)
            
            # Set anomaly detection results
            result.anomalies_detected = total_anomalies
            result.anomaly_details = anomalies
            
            if total_anomalies > 0:
                anomaly_percentage = total_anomalies / len(df)
                if anomaly_percentage > 0.1:  # More than 10% anomalies
                    result.add_warning(
                        f"High anomaly rate detected: {anomaly_percentage:.2%} "
                        f"({total_anomalies} anomalies)"
                    )
                else:
                    result.add_warning(f"Detected {total_anomalies} potential anomalies")
        
        except Exception as e:
            result.add_warning(f"Anomaly detection failed: {str(e)}")
            result.anomalies_detected = None
    
    def validate_ohlcv_row(
        self,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single OHLCV row.
        
        Args:
            open_price: Opening price
            high_price: High price
            low_price: Low price
            close_price: Closing price
            volume: Optional volume
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check for negative or zero prices
        prices = [open_price, high_price, low_price, close_price]
        if any(price <= 0 for price in prices):
            errors.append("Prices must be positive")
        
        # Check OHLC relationships
        if high_price < max(open_price, low_price, close_price):
            errors.append(f"High ({high_price}) must be >= max(open, low, close)")
        
        if low_price > min(open_price, high_price, close_price):
            errors.append(f"Low ({low_price}) must be <= min(open, high, close)")
        
        # Validate volume if provided
        if volume is not None and volume < 0:
            errors.append("Volume cannot be negative")
        
        return len(errors) == 0, errors
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        fill_missing: bool = False,
        interpolation_method: str = "linear"
    ) -> pd.DataFrame:
        """
        Clean DataFrame by removing/fixing common issues.
        
        Args:
            df: DataFrame to clean
            remove_duplicates: Whether to remove duplicate rows
            fill_missing: Whether to fill missing values
            interpolation_method: Method for filling missing values
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()
        
        try:
            # Remove duplicate rows
            if remove_duplicates:
                initial_rows = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                removed_rows = initial_rows - len(cleaned_df)
                if removed_rows > 0:
                    self.logger.info(f"Removed {removed_rows} duplicate rows")
            
            # Fill missing values
            if fill_missing:
                numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_columns:
                    if cleaned_df[col].isnull().any():
                        if interpolation_method == "forward_fill":
                            cleaned_df[col] = cleaned_df[col].ffill()
                        elif interpolation_method == "backward_fill":
                            cleaned_df[col] = cleaned_df[col].bfill()
                        elif interpolation_method == "linear":
                            cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
                        elif interpolation_method == "mean":
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                        elif interpolation_method == "median":
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            
            return cleaned_df
        
        except Exception as e:
            self.logger.error(f"DataFrame cleaning failed: {str(e)}")
            return df  # Return original if cleaning fails


# Convenience functions for direct use
def validate_financial_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    check_ohlcv: bool = True,
    **kwargs
) -> DataFrameValidationResult:
    """
    Convenience function for validating financial DataFrames.
    
    Args:
        df: DataFrame to validate
        required_columns: Required column names
        check_ohlcv: Whether to validate OHLCV relationships
        **kwargs: Additional validation parameters
        
    Returns:
        DataFrameValidationResult
    """
    validator = DataFrameValidator()
    return validator.validate_dataframe(
        df=df,
        required_columns=required_columns,
        check_ohlcv=check_ohlcv,
        **kwargs
    )


def validate_ohlcv_dataframe(df: pd.DataFrame) -> DataFrameValidationResult:
    """
    Convenience function specifically for OHLCV data validation.
    Handles both capitalized and lowercase column names.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrameValidationResult
    """
    # Try both capitalized and lowercase column names
    df_columns_lower = {col.lower(): col for col in df.columns}
    
    # Determine which format to use based on what's available
    capitalized_columns = ['Open', 'High', 'Low', 'Close']
    lowercase_columns = ['open', 'high', 'low', 'close']
    
    # Check if we have capitalized columns
    has_capitalized = all(col in df.columns for col in capitalized_columns)
    has_lowercase = all(col.lower() in df_columns_lower for col in capitalized_columns)
    
    if has_capitalized:
        required_columns = capitalized_columns
    elif has_lowercase:
        required_columns = lowercase_columns
    else:
        # Use capitalized as default for error reporting
        required_columns = capitalized_columns
    
    return validate_financial_dataframe(
        df=df,
        required_columns=required_columns,
        check_ohlcv=True,
        detect_anomalies=True
    )


def quick_validate(df: pd.DataFrame) -> bool:
    """
    Quick validation that returns only boolean result.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if validation passes, False otherwise
    """
    result = validate_financial_dataframe(df, check_ohlcv=False)
    return result.is_valid
