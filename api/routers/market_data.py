"""
Market data endpoints for downloading, storing, and loading stock data.

This module provides REST API endpoints for market data operations including
downloading from Yahoo Finance, saving as CSV, and loading existing files.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from api.models.market_data import (
    MarketDataRequest,
    MarketDataResponse,
    LoadCSVRequest,
    LoadCSVResponse,
    MarketDataInfo,
    ErrorResponse
)
from api.services.market_data_service import MarketDataService
from api.dependencies import (
    validate_symbol,
    validate_period,
    validate_interval,
    ensure_data_directory
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize market data service
market_service = MarketDataService()


@router.post("/download", response_model=MarketDataResponse)
async def download_stock_data(
    request: MarketDataRequest,
    _: str = Depends(ensure_data_directory)
) -> MarketDataResponse:
    """
    Download stock data from Yahoo Finance and optionally save as CSV.
    
    Args:
        request: Market data download request
        
    Returns:
        MarketDataResponse: Downloaded data information
        
    Raises:
        HTTPException: If download fails
    """
    try:
        logger.info(f"Downloading stock data for {request.symbol}")
        
        # Validate inputs using dependencies
        symbol = validate_symbol(request.symbol)
        period = validate_period(request.period)
        interval = validate_interval(request.interval)
        
        # Download data
        response = market_service.download_stock_data(
            symbol=symbol,
            period=period,
            interval=interval,
            save_csv=request.save_csv
        )
        
        logger.info(f"Successfully downloaded {response.total_records} records for {symbol}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to download stock data: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/load-csv", response_model=LoadCSVResponse)
async def load_csv_data(request: LoadCSVRequest) -> LoadCSVResponse:
    """
    Load stock data from CSV file.
    
    Args:
        request: CSV loading request
        
    Returns:
        LoadCSVResponse: Loaded data information
        
    Raises:
        HTTPException: If loading fails
    """
    try:
        logger.info(f"Loading CSV data from {request.file_path}")
        
        # Validate symbol if provided
        symbol = None
        if request.symbol:
            symbol = validate_symbol(request.symbol)
        
        # Load CSV data
        response = market_service.load_csv_data(
            file_path=request.file_path,
            symbol=symbol
        )
        
        logger.info(f"Successfully loaded {response.total_records} records from CSV")
        return response
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail="CSV file not found")
    except Exception as e:
        logger.error(f"Failed to load CSV data: {e}")
        raise HTTPException(status_code=500, detail=f"CSV loading failed: {str(e)}")


@router.get("/info/{symbol}", response_model=MarketDataInfo)
async def get_stock_info(symbol: str) -> MarketDataInfo:
    """
    Get current stock information.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        MarketDataInfo: Current stock information
    """
    try:
        # Validate symbol
        validated_symbol = validate_symbol(symbol)
        
        # Get stock info
        info = market_service.get_stock_info(validated_symbol)
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get stock info for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stock info: {str(e)}")


@router.get("/csv-files", response_model=List[Dict[str, Any]])
async def list_csv_files() -> List[Dict[str, Any]]:
    """
    List all CSV files in the data directory.
    
    Returns:
        List of CSV file information
    """
    try:
        files = market_service.list_csv_files()
        return files
        
    except Exception as e:
        logger.error(f"Failed to list CSV files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.delete("/csv-files/{filename}")
async def delete_csv_file(filename: str) -> Dict[str, str]:
    """
    Delete a CSV file from the data directory.
    
    Args:
        filename: Name of CSV file to delete
        
    Returns:
        Confirmation message
        
    Raises:
        HTTPException: If deletion fails
    """
    try:
        from pathlib import Path
        
        # Validate filename (basic security check)
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = Path("data") / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not file_path.suffix.lower() == '.csv':
            raise HTTPException(status_code=400, detail="Only CSV files can be deleted")
        
        file_path.unlink()
        logger.info(f"Deleted CSV file: {filename}")
        
        return {"message": f"File {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.get("/validate/{symbol}")
async def validate_stock_symbol(symbol: str) -> Dict[str, Any]:
    """
    Validate if a stock symbol exists and is tradeable.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        Validation result
    """
    try:
        # Validate symbol format
        validated_symbol = validate_symbol(symbol)
        
        # Try to get basic info to validate symbol exists
        info = market_service.get_stock_info(validated_symbol)
        
        # Check if we got meaningful data
        is_valid = (
            info.current_price is not None or 
            info.previous_close is not None or
            info.market_cap is not None
        )
        
        return {
            "symbol": validated_symbol,
            "is_valid": is_valid,
            "info": info.dict() if is_valid else None
        }
        
    except Exception as e:
        logger.warning(f"Symbol validation failed for {symbol}: {e}")
        return {
            "symbol": symbol,
            "is_valid": False,
            "error": str(e)
        }
