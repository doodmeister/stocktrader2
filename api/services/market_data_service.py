"""
Market data service for downloading, storing, and loading stock data.

This service handles downloading stock data from Yahoo Finance, saving it as CSV files,
and loading existing CSV files for analysis.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

from core.data_validator import validate_dataframe, validate_file_path
from api.models.market_data import MarketDataResponse, LoadCSVResponse, MarketDataInfo

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for managing market data operations."""
    
    def __init__(self, data_directory: str = "data"):
        """
        Initialize the market data service.
        
        Args:
            data_directory: Directory to store CSV files
        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(exist_ok=True)
        logger.info(f"MarketDataService initialized with data directory: {self.data_directory}")
    
    def download_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        save_csv: bool = True
    ) -> MarketDataResponse:
        """
        Download stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            save_csv: Whether to save data as CSV file
            
        Returns:
            MarketDataResponse: Downloaded data information
            
        Raises:
            Exception: If download fails or data is invalid
        """
        try:
            logger.info(f"Downloading data for {symbol}, period: {period}, interval: {interval}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Download data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Validate the downloaded data
            try:
                validate_dataframe(data)
            except Exception as e:
                logger.warning(f"Data validation warning: {e}")
            
            # Generate CSV file path
            csv_file_path = None
            if save_csv:
                csv_file_path = self._generate_csv_filename(symbol, period, interval)
                full_path = self.data_directory / csv_file_path
                
                # Save to CSV
                data.to_csv(full_path, index=False)
                logger.info(f"Data saved to: {full_path}")
            
            # Prepare data preview (first 5 rows)
            preview_data = [
                {str(k): v for k, v in row.items()} for row in data.head().to_dict('records')
            ]
            
            # Convert datetime objects to strings for JSON serialization
            for row in preview_data:
                for key, value in row.items():
                    if pd.isna(value):
                        row[key] = None
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        row[key] = value.isoformat()
                    elif isinstance(value, (pd.Timedelta, timedelta)):
                        row[key] = str(value)
            
            return MarketDataResponse(
                symbol=symbol,
                period=period,
                interval=interval,
                start_date=data['Date'].iloc[0].isoformat() if not data.empty else "",
                end_date=data['Date'].iloc[-1].isoformat() if not data.empty else "",
                total_records=len(data),
                csv_file_path=csv_file_path,
                data_preview=preview_data,
            )
            
        except Exception as e:
            logger.error(f"Failed to download data for {symbol}: {e}")
            raise Exception(f"Failed to download stock data: {str(e)}")
    
    def load_csv_data(self, file_path: str, symbol: Optional[str] = None) -> LoadCSVResponse:
        """
        Load stock data from CSV file.
        
        Args:
            file_path: Path to CSV file
            symbol: Optional symbol for validation
            
        Returns:
            LoadCSVResponse: Loaded data information
            
        Raises:
            Exception: If file loading fails or data is invalid
        """
        try:
            # Validate file path
            validate_file_path(file_path)
            
            full_path = Path(file_path)
            if not full_path.exists():
                # Try relative to data directory
                full_path = self.data_directory / file_path
                if not full_path.exists():
                    raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            logger.info(f"Loading CSV data from: {full_path}")
            
            # Load CSV data
            data = pd.read_csv(full_path)
            
            if data.empty:
                raise ValueError("CSV file contains no data")
            
            # Validate the loaded data
            try:
                validate_dataframe(data)
            except Exception as e:
                logger.warning(f"Data validation warning: {e}")
            
            # Get file size
            file_size_mb = full_path.stat().st_size / (1024 * 1024)
            
            # Prepare data preview (first 5 rows)
            preview_data = [
                {str(k): v for k, v in row.items()} for row in data.head().to_dict('records')
            ]
            
            # Convert datetime objects to strings for JSON serialization
            for row in preview_data:
                for key, value in row.items():
                    if pd.isna(value):
                        row[key] = None
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        row[key] = value.isoformat()
                    elif isinstance(value, (pd.Timedelta, timedelta)):
                        row[key] = str(value)
            
            # Try to detect date column and get date range
            date_col = self._detect_date_column(data)
            start_date = ""
            end_date = ""
            
            if date_col:
                try:
                    dates = pd.to_datetime(data[date_col])
                    start_date = dates.min().isoformat()
                    end_date = dates.max().isoformat()
                except Exception as e:
                    logger.warning(f"Failed to parse dates: {e}")
            
            return LoadCSVResponse(
                file_path=str(full_path),
                symbol=symbol,
                total_records=len(data),
                start_date=start_date,
                end_date=end_date,
                columns=list(data.columns),
                data_preview=preview_data,
                file_size_mb=round(file_size_mb, 2)
            )
            
        except Exception as e:
            logger.error(f"Failed to load CSV data from {file_path}: {e}")
            raise Exception(f"Failed to load CSV data: {str(e)}")
    
    def get_stock_info(self, symbol: str) -> MarketDataInfo:
        """
        Get current stock information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            MarketDataInfo: Current stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return MarketDataInfo(
                symbol=symbol,
                current_price=info.get('currentPrice'),
                previous_close=info.get('previousClose'),
                change=info.get('currentPrice', 0) - info.get('previousClose', 0) if info.get('currentPrice') and info.get('previousClose') else None,
                change_percent=((info.get('currentPrice', 0) - info.get('previousClose', 0)) / info.get('previousClose', 1)) * 100 if info.get('currentPrice') and info.get('previousClose') else None,
                volume=info.get('volume'),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                dividend_yield=info.get('dividendYield')
            )
            
        except Exception as e:
            logger.warning(f"Failed to get stock info for {symbol}: {e}")
            return MarketDataInfo(symbol=symbol)
    
    def list_csv_files(self) -> List[Dict[str, Any]]:
        """
        List all CSV files in the data directory.
        
        Returns:
            List of CSV file information
        """
        csv_files = []
        
        for file_path in self.data_directory.glob("*.csv"):
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                csv_files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_mb": round(file_size_mb, 2),
                    "modified": modified_time.isoformat(),
                    "symbol": self._extract_symbol_from_filename(file_path.name)
                })
            except Exception as e:
                logger.warning(f"Failed to get info for {file_path}: {e}")
        
        return sorted(csv_files, key=lambda x: x["modified"], reverse=True)
    
    def _generate_csv_filename(self, symbol: str, period: str, interval: str) -> str:
        """Generate CSV filename for stock data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{symbol}_{period}_{interval}_{timestamp}.csv"
    
    def _detect_date_column(self, data: pd.DataFrame) -> Optional[str]:
        """Detect date column in DataFrame."""
        possible_date_columns = ['Date', 'date', 'Datetime', 'datetime', 'timestamp', 'Time', 'time']
        
        for col in possible_date_columns:
            if col in data.columns:
                return col
        
        # Check if index is datetime
        if isinstance(data.index, pd.DatetimeIndex):
            index_name = data.index.name
            if isinstance(index_name, str):
                return index_name
            return "Date"
        
        return None
    
    def _extract_symbol_from_filename(self, filename: str) -> Optional[str]:
        """Extract symbol from CSV filename."""
        try:
            # Assume format: SYMBOL_period_interval_timestamp.csv
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 1:
                return parts[0]
        except Exception:
            pass
        return None
