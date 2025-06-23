"""
Market Data Service for the StockTrader API.

This service provides market data downloading, CSV storage, and data management
functionality using the existing utils.data_downloader module.
"""

import os
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

from utils.data_downloader import (
    download_stock_data,
    fetch_daily_ohlcv,
    save_to_csv,
    clear_cache
)
from core.data_validator import validate_symbol
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MarketDataService:
    """
    Service for handling market data operations including downloading,
    storage, and retrieval of stock data.
    """
    
    def __init__(self, data_directory: str = "data"):
        """
        Initialize the MarketDataService.
        
        Args:
            data_directory: Directory to store CSV files
        """
        self.data_directory = data_directory
        self.csv_directory = os.path.join(data_directory, "csv")
        os.makedirs(self.csv_directory, exist_ok=True)
        logger.info(f"MarketDataService initialized with data directory: {self.data_directory}")
    
    def download_and_save_stock_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, date],
        end_date: Union[str, date],
        interval: str = "1d",
        save_csv: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download stock data and optionally save to CSV files.
        
        Args:
            symbols: Single symbol or list of symbols to download
            start_date: Start date for data (YYYY-MM-DD or date object)
            end_date: End date for data (YYYY-MM-DD or date object)
            interval: Data interval (1d, 1h, etc.)
            save_csv: Whether to save data to CSV files
            
        Returns:
            Dictionary of symbol -> DataFrame
            
        Raises:
            ValueError: If invalid parameters or no data returned
        """
        # Normalize symbols to list
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Normalize dates
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        logger.info(f"Downloading data for {symbols} from {start_date} to {end_date}")
        
        # Download data using existing data_downloader
        data_dict = download_stock_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if not data_dict:
            raise ValueError(f"No data returned for symbols: {symbols}")
        
        # Save to CSV if requested
        if save_csv:
            for symbol, df in data_dict.items():
                csv_path = self._get_csv_path(symbol, start_date, end_date, interval)
                save_to_csv(df, csv_path)
                logger.info(f"Saved {symbol} data to {csv_path}")
        
        return data_dict
    
    def load_stock_data_from_csv(
        self,
        symbol: str,
        start_date: Union[str, date],
        end_date: Union[str, date],
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Load stock data from existing CSV file.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            
        Returns:
            DataFrame with stock data or None if not found
        """
        # Normalize dates
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        csv_path = self._get_csv_path(symbol, start_date, end_date, interval)
        
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {symbol} data from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            return None
    
    def get_or_download_stock_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, date],
        end_date: Union[str, date],
        interval: str = "1d",
        force_download: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Get stock data, trying CSV first, then downloading if needed.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            force_download: Force download even if CSV exists
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        # Normalize symbols to list
        if isinstance(symbols, str):
            symbols = [symbols]
        
        result = {}
        symbols_to_download = []
        
        if not force_download:
            # Try to load from CSV first
            for symbol in symbols:
                df = self.load_stock_data_from_csv(symbol, start_date, end_date, interval)
                if df is not None:
                    result[symbol] = df
                else:
                    symbols_to_download.append(symbol)
        else:
            symbols_to_download = symbols
        
        # Download missing symbols
        if symbols_to_download:
            downloaded_data = self.download_and_save_stock_data(
                symbols=symbols_to_download,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                save_csv=True
            )
            result.update(downloaded_data)
        
        return result
    
    def validate_stock_symbol(self, symbol: str) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate a stock symbol using the core validator.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = validate_symbol(symbol)
        validated_symbol = symbol
        if validation_result.details and validation_result.details.get('validated_symbol') is not None:
            validated_symbol = validation_result.details.get('validated_symbol')
        # Ensure validated_symbol is a string, not None
        if validated_symbol is None:
            validated_symbol = ""
        errors = validation_result.errors if validation_result.errors is not None else []
        # Ensure errors is a list of strings
        if not isinstance(errors, list):
            errors = [str(errors)]
        return {
            "is_valid": validation_result.is_valid,
            "symbol": symbol,
            "validated_symbol": validated_symbol,
            "errors": errors
        }
    
    def list_available_data(self) -> List[Dict[str, str]]:
        """
        List all available CSV data files.
        
        Returns:
            List of dictionaries with file information
        """
        csv_files = []
        
        if not os.path.exists(self.csv_directory):
            return csv_files
        
        for filename in os.listdir(self.csv_directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.csv_directory, filename)
                file_info = {
                    "filename": filename,
                    "path": file_path,
                    "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
                csv_files.append(file_info)
        
        return csv_files
    
    def clear_data_cache(self) -> None:
        """Clear the data cache."""
        clear_cache()
        logger.info("Data cache cleared")
    
    def _get_csv_path(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str
    ) -> str:
        """
        Generate CSV file path for a symbol and date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Full path to CSV file
        """
        filename = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{interval}.csv"
        return os.path.join(self.csv_directory, filename)
    
    def get_latest_available_date(self, symbol: str) -> Optional[date]:
        """
        Get the latest available date for a symbol from CSV files.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest available date or None if no data found
        """
        csv_files = self.list_available_data()
        symbol_files = [f for f in csv_files if f["filename"].startswith(f"{symbol}_")]
        
        if not symbol_files:
            return None
        
        latest_date = None
        for file_info in symbol_files:
            try:
                df = pd.read_csv(file_info["path"], index_col=0, parse_dates=True)
                if not df.empty:
                    file_latest = df.index.max().date()
                    if latest_date is None or file_latest > latest_date:
                        latest_date = file_latest
            except Exception as e:
                logger.warning(f"Error reading {file_info['filename']}: {e}")
        
        return latest_date
