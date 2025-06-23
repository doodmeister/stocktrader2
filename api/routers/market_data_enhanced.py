"""
Enhanced Market Data endpoints for downloading, storing, and analyzing stock data.

This module provides REST API endpoints for market data operations using
the enhanced MarketDataService and existing data_downloader functionality.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime, timedelta
import logging
from pydantic import BaseModel

from api.services.market_data_service_enhanced import MarketDataService
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Initialize enhanced market data service
market_service = MarketDataService()


class DownloadRequest(BaseModel):
    """Request model for downloading stock data."""
    symbols: Union[str, List[str]]
    start_date: str
    end_date: str
    interval: str = "1d"
    save_csv: bool = True


class DataResponse(BaseModel):
    """Response model for stock data operations."""
    status: str
    message: str
    symbols: List[str]
    data_info: Dict[str, Any]


@router.post("/download")
async def download_stock_data(request: DownloadRequest) -> DataResponse:
    """
    Download stock data from Yahoo Finance and save as CSV.
    
    Args:
        request: Download request with symbols and date range
        
    Returns:
        DataResponse: Information about downloaded data
        
    Raises:
        HTTPException: If download fails
    """
    try:
        logger.info(f"Download request: {request.symbols} from {request.start_date} to {request.end_date}")
        
        # Download data
        data_dict = market_service.download_and_save_stock_data(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
            save_csv=request.save_csv
        )
        
        # Prepare response
        symbols_list = request.symbols if isinstance(request.symbols, list) else [request.symbols]
        data_info = {}
        
        for symbol in symbols_list:
            if symbol in data_dict:
                df = data_dict[symbol]
                data_info[symbol] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "date_range": {
                        "start": df.index.min().strftime("%Y-%m-%d") if not df.empty else None,
                        "end": df.index.max().strftime("%Y-%m-%d") if not df.empty else None
                    },
                    "latest_close": float(df['close'].iloc[-1]) if 'close' in df.columns and not df.empty else None
                }
            else:
                data_info[symbol] = {"error": "No data retrieved"}
        
        return DataResponse(
            status="success",
            message=f"Downloaded data for {len(data_dict)} symbols",
            symbols=list(data_dict.keys()),
            data_info=data_info
        )
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/load/{symbol}")
async def load_stock_data(
    symbol: str,
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    interval: str = Query("1d", description="Data interval")
) -> DataResponse:
    """
    Load stock data from existing CSV file.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval
        
    Returns:
        DataResponse: Information about loaded data
        
    Raises:
        HTTPException: If file not found or loading fails
    """
    try:
        logger.info(f"Load request: {symbol} from {start_date} to {end_date}")
        
        # Load data from CSV
        df = market_service.load_stock_data_from_csv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if df is None:
            raise HTTPException(status_code=404, detail=f"No CSV data found for {symbol}")
        
        # Prepare response
        data_info = {
            symbol: {
                "rows": len(df),
                "columns": list(df.columns),
                "date_range": {
                    "start": df.index.min().strftime("%Y-%m-%d") if not df.empty else None,
                    "end": df.index.max().strftime("%Y-%m-%d") if not df.empty else None
                },
                "latest_close": float(df['close'].iloc[-1]) if 'close' in df.columns and not df.empty else None
            }
        }
        
        return DataResponse(
            status="success",
            message=f"Loaded data for {symbol}",
            symbols=[symbol],
            data_info=data_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")


@router.get("/get-or-download/{symbol}")
async def get_or_download_stock_data(
    symbol: str,
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    interval: str = Query("1d", description="Data interval"),
    force_download: bool = Query(False, description="Force download even if CSV exists")
) -> DataResponse:
    """
    Get stock data, trying CSV first, then downloading if needed.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval
        force_download: Force download even if CSV exists
        
    Returns:
        DataResponse: Information about retrieved data
    """
    try:
        logger.info(f"Get-or-download request: {symbol} from {start_date} to {end_date}")
        
        # Get data (CSV first, then download)
        data_dict = market_service.get_or_download_stock_data(
            symbols=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            force_download=force_download
        )
        
        if not data_dict or symbol not in data_dict:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        df = data_dict[symbol]
        
        # Prepare response
        data_info = {
            symbol: {
                "rows": len(df),
                "columns": list(df.columns),
                "date_range": {
                    "start": df.index.min().strftime("%Y-%m-%d") if not df.empty else None,
                    "end": df.index.max().strftime("%Y-%m-%d") if not df.empty else None
                },
                "latest_close": float(df['close'].iloc[-1]) if 'close' in df.columns and not df.empty else None
            }
        }
        
        return DataResponse(
            status="success",
            message=f"Retrieved data for {symbol}",
            symbols=[symbol],
            data_info=data_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get-or-download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Get-or-download failed: {str(e)}")


@router.get("/validate/{symbol}")
async def validate_symbol(symbol: str) -> Dict[str, Any]:
    """
    Validate a stock symbol.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        Validation results
    """
    try:
        validation_result = market_service.validate_stock_symbol(symbol)
        return validation_result
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/list-files")
async def list_available_data() -> Dict[str, Any]:
    """
    List all available CSV data files.
    
    Returns:
        List of available data files with metadata
    """
    try:
        files = market_service.list_available_data()
        
        return {
            "status": "success",
            "total_files": len(files),
            "files": files
        }
        
    except Exception as e:
        logger.error(f"List files failed: {e}")
        raise HTTPException(status_code=500, detail=f"List files failed: {str(e)}")


@router.post("/clear-cache")
async def clear_data_cache() -> Dict[str, str]:
    """
    Clear the data cache.
    
    Returns:
        Cache clear status
    """
    try:
        market_service.clear_data_cache()
        return {
            "status": "success",
            "message": "Data cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Clear cache failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clear cache failed: {str(e)}")


@router.get("/latest-date/{symbol}")
async def get_latest_available_date(symbol: str) -> Dict[str, Any]:
    """
    Get the latest available date for a symbol.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Latest available date information
    """
    try:
        latest_date = market_service.get_latest_available_date(symbol)
        
        return {
            "status": "success",
            "symbol": symbol,
            "latest_date": latest_date.strftime("%Y-%m-%d") if latest_date else None,
            "has_data": latest_date is not None
        }
        
    except Exception as e:
        logger.error(f"Get latest date failed: {e}")
        raise HTTPException(status_code=500, detail=f"Get latest date failed: {str(e)}")


@router.get("/health")
async def market_data_health() -> Dict[str, Any]:
    """
    Health check for market data service.
    
    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "market_data",
        "data_directory": market_service.data_directory,
        "csv_directory": market_service.csv_directory,
        "available_files": len(market_service.list_available_data())
    }
