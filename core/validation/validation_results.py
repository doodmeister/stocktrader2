"""
Validation result models for the StockTrader Bot.

This module contains Pydantic models for validation results that are
optimized for FastAPI responses and comprehensive error reporting.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
import pandas as pd


class ValidationResult(BaseModel):
    """Base validation result model."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    is_valid: bool = Field(..., description="Whether the validation passed")
    errors: Optional[List[str]] = Field(None, description="List of error messages")
    warnings: List[str] = Field(default_factory=list, description="List of warning messages")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional validation details")
    validation_timestamp: datetime = Field(default_factory=datetime.now, description="When validation was performed")
    
    def add_error(self, error_message: str) -> None:
        """Add an error message and mark validation as failed."""
        self.is_valid = False
        if self.errors is None:
            self.errors = []
        self.errors.append(error_message)
    
    def add_warning(self, warning_message: str) -> None:
        """Add a warning message if it doesn't already exist."""
        if warning_message not in self.warnings:
            self.warnings.append(warning_message)
    
    def has_errors(self) -> bool:
        """Check if validation has any errors."""
        return self.errors is not None and len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if validation has any warnings."""
        return len(self.warnings) > 0
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the validation result."""
        status = "PASSED" if self.is_valid else "FAILED"
        error_count = len(self.errors) if self.errors else 0
        warning_count = len(self.warnings)
        return f"Validation {status}: {error_count} errors, {warning_count} warnings"


class DataFrameValidationResult(ValidationResult):
    """Enhanced DataFrame validation result with detailed error reporting."""
    
    # Error tracking
    error_details: Optional[Dict[Union[int, str], List[str]]] = Field(
        None, 
        description="Detailed errors mapped by row index or column name"
    )
    
    # Data information
    validated_data: Optional[pd.DataFrame] = Field(
        None, 
        description="The validated DataFrame (if validation passed)"
    )
    dataframe_shape: Optional[Tuple[int, int]] = Field(
        None, 
        description="Shape of the DataFrame (rows, columns)"
    )
    
    # Column analysis
    missing_columns: List[str] = Field(
        default_factory=list, 
        description="List of required columns that are missing"
    )
    duplicate_columns: List[str] = Field(
        default_factory=list, 
        description="List of duplicate column names"
    )
    extra_columns: List[str] = Field(
        default_factory=list, 
        description="List of extra columns not in requirements"
    )
    
    # Data quality metrics
    nan_counts: Optional[Dict[str, int]] = Field(
        None, 
        description="Count of NaN values per column"
    )
    null_percentages: Optional[Dict[str, float]] = Field(
        None, 
        description="Percentage of null values per column"
    )
    data_types: Optional[Dict[str, str]] = Field(
        None, 
        description="Data types of each column"
    )
    
    # Validation metrics
    validation_time_seconds: Optional[float] = Field(
        None, 
        description="Time taken for validation in seconds"
    )
    rows_validated: Optional[int] = Field(
        None, 
        description="Number of rows that were validated"
    )
    columns_validated: Optional[int] = Field(
        None, 
        description="Number of columns that were validated"
    )
    
    # OHLC-specific validation
    ohlc_validation_passed: Optional[bool] = Field(
        None, 
        description="Whether OHLC relationship validation passed"
    )
    invalid_ohlc_rows: Optional[List[int]] = Field(
        None, 
        description="Row indices with invalid OHLC relationships"
    )
    
    # Anomaly detection results
    anomalies_detected: Optional[int] = Field(
        None, 
        description="Number of anomalies detected"
    )
    anomaly_details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Detailed anomaly detection results"
    )
    
    # Sample of failed data
    failed_rows_sample: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Sample of rows that failed validation"
    )
    
    @field_validator("validated_data")
    @classmethod
    def validate_dataframe(cls, value):
        """Validate the DataFrame field."""
        if value is not None and not isinstance(value, pd.DataFrame):
            raise ValueError("validated_data must be a pandas DataFrame")
        return value
    
    def add_column_error(self, column: str, error_message: str) -> None:
        """Add an error for a specific column."""
        if self.error_details is None:
            self.error_details = {}
        if column not in self.error_details:
            self.error_details[column] = []
        self.error_details[column].append(error_message)
        self.add_error(f"Column '{column}': {error_message}")
    
    def add_row_error(self, row_index: int, error_message: str) -> None:
        """Add an error for a specific row."""
        if self.error_details is None:
            self.error_details = {}
        if row_index not in self.error_details:
            self.error_details[row_index] = []
        self.error_details[row_index].append(error_message)
        self.add_error(f"Row {row_index}: {error_message}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a comprehensive error summary."""
        return {
            "total_errors": len(self.errors) if self.errors else 0,
            "total_warnings": len(self.warnings),
            "column_errors": len([k for k in (self.error_details or {}).keys() if isinstance(k, str)]),
            "row_errors": len([k for k in (self.error_details or {}).keys() if isinstance(k, int)]),
            "missing_columns": len(self.missing_columns),
            "duplicate_columns": len(self.duplicate_columns),
            "validation_passed": self.is_valid
        }


class SymbolValidationResult(ValidationResult):
    """Result for stock symbol validation."""
    
    symbol: Optional[str] = Field(None, description="The validated symbol")
    normalized_symbol: Optional[str] = Field(None, description="The normalized symbol format")
    exchange: Optional[str] = Field(None, description="The identified exchange")
    is_active: Optional[bool] = Field(None, description="Whether the symbol is actively traded")
    market_cap: Optional[float] = Field(None, description="Market capitalization if available")
    
    def get_validation_info(self) -> Dict[str, Any]:
        """Get symbol validation information."""
        return {
            "original_symbol": self.symbol,
            "normalized_symbol": self.normalized_symbol,
            "exchange": self.exchange,
            "is_active": self.is_active,
            "validation_passed": self.is_valid
        }


class TechnicalIndicatorValidationResult(ValidationResult):
    """Result for technical indicator calculation validation."""
    
    indicator_name: Optional[str] = Field(None, description="Name of the indicator")
    calculation_time_seconds: Optional[float] = Field(None, description="Time taken to calculate")
    data_points_processed: Optional[int] = Field(None, description="Number of data points processed")
    null_values_count: Optional[int] = Field(None, description="Number of null values in result")
    null_percentage: Optional[float] = Field(None, description="Percentage of null values")
    value_range: Optional[Tuple[float, float]] = Field(None, description="Min and max values")
    
    def is_indicator_valid(self) -> bool:
        """Check if indicator calculation is valid based on null percentage."""
        if self.null_percentage is None:
            return self.is_valid
        # Allow higher null percentage for indicators due to rolling calculations
        return self.is_valid and self.null_percentage <= 0.7  # 70% tolerance


class BatchValidationResult(ValidationResult):
    """Result for batch validation operations."""
    
    total_items: int = Field(..., description="Total number of items processed")
    successful_items: int = Field(..., description="Number of successfully validated items")
    failed_items: int = Field(..., description="Number of failed validations")
    individual_results: List[ValidationResult] = Field(
        default_factory=list, 
        description="Individual validation results"
    )
    processing_time_seconds: Optional[float] = Field(
        None, 
        description="Total processing time"
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of batch validation."""
        if self.total_items == 0:
            return 0.0
        return self.successful_items / self.total_items
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get batch validation summary."""
        return {
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
            "processing_time_seconds": self.processing_time_seconds,
            "overall_validation_passed": self.is_valid
        }
