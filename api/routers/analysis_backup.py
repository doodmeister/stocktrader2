"""
Analysis endpoints for technical indicators, pattern detection, and AI analysis.

This module provides REST API endpoints for the complete analysis pipeline
including technical indicators, candlestick patterns, and OpenAI integration.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response Models
class TechnicalAnalysisRequest(BaseModel):
    """Request model for technical analysis."""
    symbol: str
    csv_file: Optional[str] = None
    indicators: Optional[List[str]] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    sma_period: int = 20
    ema_period: int = 20

class TechnicalIndicatorData(BaseModel):
    """Data model for individual technical indicator."""
    name: str
    current_value: Optional[float] = None
    signal: str = "Hold"
    values: List[Optional[float]] = []
    metadata: Dict[str, Any] = {}

class TechnicalAnalysisResponse(BaseModel):
    """Response model for technical analysis."""
    status: str
    symbol: str
    analysis_time: datetime
    composite_signal: float
    signal_strength: str
    indicators: Dict[str, TechnicalIndicatorData]
    data_info: Dict[str, Any]
    current_price: float
    price_change: float
    price_change_percent: float


# Utility functions
def clean_nan_values(value: Any) -> Any:
    """
    Clean NaN values from various data types for JSON serialization.
    
    Args:
        value: Value to clean (float, list, dict, etc.)
        
    Returns:
        Cleaned value with NaN replaced by None
    """
    if value is None:
        return None
    
    # Handle scalar values first
    if isinstance(value, (int, str, bool)):
        return value
    
    # Handle numpy/pandas scalar values
    if isinstance(value, (np.integer, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value) if isinstance(value, np.floating) else int(value)
    
    # Handle regular float
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    
    # Handle lists
    if isinstance(value, list):
        return [clean_nan_values(item) for item in value]
    
    # Handle dictionaries
    if isinstance(value, dict):
        return {k: clean_nan_values(v) for k, v in value.items()}
    
    # For pandas scalars, try to check if it's NaN
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    
    return value


async def _load_stock_data(symbol: str, csv_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load stock data either from specified CSV file or latest downloaded data.
    
    Args:
        symbol: Stock symbol
        csv_file: Optional CSV file path
        
    Returns:
        DataFrame with OHLCV data or None if not found
    """
    try:
        if csv_file and os.path.exists(csv_file):
            # Load from specific CSV file
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # Ensure column names are standardized
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume',
                'Open': 'Open', 'High': 'High', 'Low': 'Low',
                'Close': 'Close', 'Volume': 'Volume'
            }
            
            data = data.rename(columns=column_mapping)
            return data
            
        else:
            # Find latest CSV file for the symbol
            data_dir = "data/csv"
            if not os.path.exists(data_dir):
                return None
                
            # Look for CSV files matching the symbol
            csv_files = [f for f in os.listdir(data_dir) if f.startswith(f"{symbol}_") and f.endswith(".csv")]
            
            if not csv_files:
                return None
                
            # Use the most recent file (assuming filename contains date)
            latest_file = sorted(csv_files)[-1]
            file_path = os.path.join(data_dir, latest_file)
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Ensure column names are standardized
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume',
                'Open': 'Open', 'High': 'High', 'Low': 'Low',
                'Close': 'Close', 'Volume': 'Volume'
            }
            
            data = data.rename(columns=column_mapping)
            return data
            
    except Exception as e:
        logger.error(f"Failed to load data for {symbol}: {e}")
        return None


@router.get("/health")
async def analysis_health() -> Dict[str, Any]:
    """
    Health check for analysis services.
    
    Returns:
        Analysis services status
    """
    return {
        "status": "healthy",
        "services": {
            "technical_indicators": "available",
            "pattern_detection": "available", 
            "openai_integration": "available",
            "complete_analysis": "available"
        }
    }


@router.post("/test")
async def test_analysis() -> Dict[str, Any]:
    """
    Test endpoint for analysis functionality.
    
    Returns:
        Test results
    """
    try:
        # First try a simple test without imports
        return {
            "status": "success",
            "message": "Analysis endpoint is working",
            "server": "fastapi",
            "test": "basic"
        }
        
    except Exception as e:
        logger.error(f"Analysis test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis test failed: {str(e)}")


@router.post("/test-imports")
async def test_imports() -> Dict[str, Any]:
    """
    Test endpoint for testing imports.
    
    Returns:
        Import test results
    """
    try:
        # Test core module imports
        from core.technical_indicators import TechnicalIndicators
        from patterns.orchestrator import CandlestickPatterns
        
        return {
            "status": "success",
            "message": "Analysis modules loaded successfully",
            "modules": {
                "technical_indicators": "loaded",
                "pattern_detection": "loaded"
            }
        }
        
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Import test failed: {str(e)}")


# TODO: Add complete analysis endpoints
# - Technical indicators analysis
# - Pattern detection analysis  
# - OpenAI integration
# - Complete analysis pipeline


@router.post("/technical-indicators", response_model=TechnicalAnalysisResponse)
async def analyze_technical_indicators(request: TechnicalAnalysisRequest) -> TechnicalAnalysisResponse:
    """
    Perform technical indicators analysis on stock data.
    
    Args:
        request: Technical analysis request parameters
        
    Returns:
        Technical analysis results with indicators and signals
    """
    try:
        # Import analysis modules
        from core.technical_indicators import TechnicalIndicators
        from utils.technicals.analysis import TechnicalAnalysis
          # Load data (either from CSV or latest downloaded data)
        data = await _load_stock_data(request.symbol, request.csv_file)
        
        if data is None or data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for symbol {request.symbol}"
            )
        
        # Check if we have sufficient data for at least one requested indicator
        requested_indicators_min_rows = []
        if request.indicators:
            if "rsi" in request.indicators:
                requested_indicators_min_rows.append(request.rsi_period)
            if "macd" in request.indicators:
                requested_indicators_min_rows.append(request.macd_slow + request.macd_signal)
            if "bollinger_bands" in request.indicators:
                requested_indicators_min_rows.append(request.bb_period)
            if "sma" in request.indicators:
                requested_indicators_min_rows.append(request.sma_period)
            if "ema" in request.indicators:
                requested_indicators_min_rows.append(request.ema_period)
        
        # Only fail if we can't calculate ANY of the requested indicators
        if requested_indicators_min_rows and len(data) < min(requested_indicators_min_rows):
            min_possible = min(requested_indicators_min_rows)
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for any requested indicators. Need at least {min_possible} rows for the smallest indicator, got {len(data)}. Try downloading more historical data."
            )
        
        # Initialize technical indicators
        ti = TechnicalIndicators(data)
        ta = TechnicalAnalysis(data)
        
        indicators = {}
        skipped_indicators = []  # Track indicators skipped due to insufficient data
        
        # Calculate requested indicators with individual data sufficiency checks
        logger.info(f"Starting indicator calculations for {request.symbol}")
        logger.info(f"Requested indicators: {request.indicators}")
        logger.info(f"Data shape: {data.shape}, TI data shape: {ti.data.shape}")
        
        if request.indicators and "rsi" in request.indicators and len(data) >= request.rsi_period:
            logger.info(f"Calculating RSI with period {request.rsi_period}")
            logger.info(f"Data has {len(data)} rows, need {request.rsi_period}")
            try:
                logger.info("About to call ti.calculate_rsi()")
                rsi_values = ti.calculate_rsi(length=request.rsi_period)
                logger.info(f"RSI values calculated, type: {type(rsi_values)}, length: {len(rsi_values) if hasattr(rsi_values, '__len__') else 'N/A'}")
                
                # Ensure rsi_values is a Series and get the last value safely
                if isinstance(rsi_values, pd.Series) and len(rsi_values) > 0:
                    last_rsi = rsi_values.iloc[-1]
                    current_rsi = float(last_rsi) if pd.notna(last_rsi) else None
                    logger.info(f"Current RSI: {current_rsi}")
                else:
                    current_rsi = None
                    logger.warning(f"RSI values not a Series or empty: {type(rsi_values)}")
                
                # Generate RSI signal
                rsi_signal = "Hold"
                if current_rsi is not None:
                    if current_rsi > 70:
                        rsi_signal = "Sell"
                    elif current_rsi < 30:
                        rsi_signal = "Buy"
                
                logger.info(f"Creating TechnicalIndicatorData with current_rsi={current_rsi}, signal={rsi_signal}")
                indicators["rsi"] = TechnicalIndicatorData(
                    name="RSI",
                    current_value=clean_nan_values(current_rsi),
                    signal=rsi_signal,
                    values=clean_nan_values(rsi_values.tail(10).tolist()),
                    metadata={"period": request.rsi_period, "overbought": 70, "oversold": 30}
                )
                logger.info(f"RSI indicator added successfully: {indicators['rsi']}")
            except Exception as e:
                logger.error(f"RSI calculation failed for {request.symbol}: {e}")
                logger.error(f"Exception type: {type(e)}")
                logger.error(f"Data columns available: {list(data.columns)}")
                logger.error(f"TI data columns available: {list(ti.data.columns)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
        elif request.indicators and "rsi" in request.indicators:
            skipped_indicators.append(f"RSI (need {request.rsi_period} rows, got {len(data)})")
            logger.warning(f"Insufficient data for RSI: need {request.rsi_period} rows, got {len(data)}")
        else:
            logger.info("RSI not requested or not in indicators list")
        
        if request.indicators and "macd" in request.indicators and len(data) >= (request.macd_slow + request.macd_signal):
            try:
                macd_line, signal_line, histogram = ti.calculate_macd(
                    fast=request.macd_fast,
                    slow=request.macd_slow,
                    signal=request.macd_signal
                )
                
                # Safely extract current values
                current_macd = None
                current_signal = None
                current_histogram = None
                
                if isinstance(macd_line, pd.Series) and len(macd_line) > 0:
                    last_macd = macd_line.iloc[-1]
                    current_macd = float(last_macd) if pd.notna(last_macd) else None
                
                if isinstance(signal_line, pd.Series) and len(signal_line) > 0:
                    last_signal = signal_line.iloc[-1]
                    current_signal = float(last_signal) if pd.notna(last_signal) else None
                
                if isinstance(histogram, pd.Series) and len(histogram) > 0:
                    last_histogram = histogram.iloc[-1]
                    current_histogram = float(last_histogram) if pd.notna(last_histogram) else None
                
                # Generate MACD signal
                macd_signal = "Hold"
                if current_macd is not None and current_signal is not None:
                    if current_macd > current_signal:
                        macd_signal = "Buy"
                    else:
                        macd_signal = "Sell"
                
                indicators["macd"] = TechnicalIndicatorData(
                    name="MACD",
                    current_value=clean_nan_values(current_macd),
                    signal=macd_signal,
                    values=clean_nan_values(macd_line.tail(10).tolist()),
                    metadata={
                        "signal_line": clean_nan_values(current_signal),
                        "histogram": clean_nan_values(current_histogram),
                        "fast": request.macd_fast,
                        "slow": request.macd_slow,
                        "signal_period": request.macd_signal
                    }
                )
            except Exception as e:
                logger.warning(f"MACD calculation failed for {request.symbol}: {e}")
        elif request.indicators and "macd" in request.indicators:
            skipped_indicators.append(f"MACD (need {request.macd_slow + request.macd_signal} rows, got {len(data)})")
            logger.warning(f"Insufficient data for MACD: need {request.macd_slow + request.macd_signal} rows, got {len(data)}")
        
        if request.indicators and "bollinger_bands" in request.indicators and len(data) >= request.bb_period:
            try:
                upper_band, middle_band, lower_band = ti.calculate_bollinger_bands(
                    period=request.bb_period,
                    std_dev=request.bb_std
                )
                
                current_price = float(data['Close'].iloc[-1])
                
                # Safely extract current band values
                current_upper = None
                current_lower = None
                current_middle = None
                
                if isinstance(upper_band, pd.Series) and len(upper_band) > 0:
                    last_upper = upper_band.iloc[-1]
                    current_upper = float(last_upper) if pd.notna(last_upper) else None
                
                if isinstance(lower_band, pd.Series) and len(lower_band) > 0:
                    last_lower = lower_band.iloc[-1]
                    current_lower = float(last_lower) if pd.notna(last_lower) else None
                
                if isinstance(middle_band, pd.Series) and len(middle_band) > 0:
                    last_middle = middle_band.iloc[-1]
                    current_middle = float(last_middle) if pd.notna(last_middle) else None
                
                # Generate Bollinger Bands signal
                bb_signal = "Hold"
                if current_upper is not None and current_lower is not None:
                    if current_price > current_upper:
                        bb_signal = "Sell"
                    elif current_price < current_lower:
                        bb_signal = "Buy"
                
                # Calculate bandwidth safely
                bandwidth = None
                if current_upper is not None and current_lower is not None and current_middle is not None and current_middle != 0:
                    bandwidth = (current_upper - current_lower) / current_middle
                
                # Clean NaN values from arrays
                upper_values_clean = clean_nan_values(upper_band.tail(10).tolist())
                middle_values_clean = clean_nan_values(middle_band.tail(10).tolist())
                lower_values_clean = clean_nan_values(lower_band.tail(10).tolist())
                main_values_clean = clean_nan_values(middle_band.tail(10).tolist())
                
                indicators["bollinger_bands"] = TechnicalIndicatorData(
                    name="Bollinger Bands",
                    current_value=clean_nan_values(current_middle),
                    signal=bb_signal,
                    values=main_values_clean,
                    metadata={
                        "upper_band": clean_nan_values(current_upper),
                        "lower_band": clean_nan_values(current_lower),
                        "upper_values": upper_values_clean,
                        "middle_values": middle_values_clean,
                        "lower_values": lower_values_clean,
                        "period": request.bb_period,
                        "std_dev": request.bb_std,
                        "bandwidth": clean_nan_values(bandwidth)
                    }
                )
            except Exception as e:
                logger.warning(f"Bollinger Bands calculation failed for {request.symbol}: {e}")
        elif request.indicators and "bollinger_bands" in request.indicators:
            skipped_indicators.append(f"Bollinger Bands (need {request.bb_period} rows, got {len(data)})")
            logger.warning(f"Insufficient data for Bollinger Bands: need {request.bb_period} rows, got {len(data)}")
        
        if request.indicators and "sma" in request.indicators and len(data) >= request.sma_period:
            try:
                sma_values = ti.calculate_sma(length=request.sma_period)
                
                # Safely extract current SMA value
                current_sma = None
                if isinstance(sma_values, pd.Series) and len(sma_values) > 0:
                    last_sma = sma_values.iloc[-1]
                    current_sma = float(last_sma) if pd.notna(last_sma) else None
                
                current_price = float(data['Close'].iloc[-1])
                  # Generate SMA signal
                sma_signal = "Hold"
                if current_sma is not None:
                    sma_signal = "Buy" if current_price > current_sma else "Sell"
                
                indicators["sma"] = TechnicalIndicatorData(
                    name="Simple Moving Average",
                    current_value=clean_nan_values(current_sma),
                    signal=sma_signal,
                    values=clean_nan_values(sma_values.tail(10).tolist()),
                    metadata={"period": request.sma_period}
                )
            except Exception as e:
                logger.warning(f"SMA calculation failed for {request.symbol}: {e}")
        elif request.indicators and "sma" in request.indicators:
            skipped_indicators.append(f"SMA (need {request.sma_period} rows, got {len(data)})")
            logger.warning(f"Insufficient data for SMA: need {request.sma_period} rows, got {len(data)}")
        
        if request.indicators and "ema" in request.indicators and len(data) >= request.ema_period:
            try:
                ema_values = ti.calculate_ema(length=request.ema_period)
                
                # Safely extract current EMA value
                current_ema = None
                if isinstance(ema_values, pd.Series) and len(ema_values) > 0:
                    last_ema = ema_values.iloc[-1]
                    current_ema = float(last_ema) if pd.notna(last_ema) else None
                
                current_price = float(data['Close'].iloc[-1])
                  # Generate EMA signal
                ema_signal = "Hold"
                if current_ema is not None:
                    ema_signal = "Buy" if current_price > current_ema else "Sell"
                
                indicators["ema"] = TechnicalIndicatorData(
                    name="Exponential Moving Average",
                    current_value=clean_nan_values(current_ema),
                    signal=ema_signal,
                    values=clean_nan_values(ema_values.tail(10).tolist()),
                    metadata={"period": request.ema_period}
                )
            except Exception as e:
                logger.warning(f"EMA calculation failed for {request.symbol}: {e}")
        elif request.indicators and "ema" in request.indicators:
            skipped_indicators.append(f"EMA (need {request.ema_period} rows, got {len(data)})")
            logger.warning(f"Insufficient data for EMA: need {request.ema_period} rows, got {len(data)}")
        
        # Calculate composite signal using TechnicalAnalysis
        composite_result = ta.evaluate(
            rsi_period=request.rsi_period,
            macd_fast=request.macd_fast,
            macd_slow=request.macd_slow,
            macd_signal=request.macd_signal,
            bb_period=request.bb_period,
            bb_std=request.bb_std
        )
        
        composite_signal = composite_result[0] if composite_result[0] is not None else 0.0
        
        # Determine signal strength
        signal_strength = "Weak"
        if abs(composite_signal) > 0.6:
            signal_strength = "Strong"
        elif abs(composite_signal) > 0.3:
            signal_strength = "Medium"
        
        # Calculate price change
        current_price = float(data['Close'].iloc[-1])
        previous_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
        price_change = current_price - previous_price
        price_change_percent = (price_change / previous_price * 100) if previous_price != 0 else 0.0
        
        # Data info
        data_info = {
            "rows": len(data),
            "columns": list(data.columns),
            "date_range": {
                "start": str(data.index[0].date()) if hasattr(data.index[0], 'date') else str(data.index[0]),
                "end": str(data.index[-1].date()) if hasattr(data.index[-1], 'date') else str(data.index[-1])
            },
            "skipped_indicators": skipped_indicators
        }
        
        return TechnicalAnalysisResponse(
            status="success",
            symbol=request.symbol,
            analysis_time=datetime.now(),
            composite_signal=clean_nan_values(composite_signal),
            signal_strength=signal_strength,
            indicators=indicators,
            data_info=data_info,
            current_price=clean_nan_values(current_price),
            price_change=clean_nan_values(price_change),
            price_change_percent=clean_nan_values(price_change_percent)
        )
        
    except Exception as e:
        logger.error(f"Technical analysis failed for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Technical analysis failed: {str(e)}"
        )


@router.post("/patterns", response_model=PatternDetectionResponse)
async def detect_patterns(request: PatternDetectionRequest) -> PatternDetectionResponse:
    """
    Detect candlestick patterns in stock data.
    
    Args:
        request: Pattern detection request parameters
        
    Returns:
        Pattern detection results
    """
    try:
        # Import pattern detection modules
        from patterns.orchestrator import CandlestickPatterns
        
        # Load data
        data = await _load_stock_data(request.symbol, request.csv_file)
        
        if data is None or data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol {request.symbol}"
            )
        
        # Initialize pattern detector
        detector = CandlestickPatterns(confidence_threshold=request.confidence_threshold)
        
        # Detect patterns
        if request.pattern_types:
            # Detect specific patterns (this would need implementation in the orchestrator)
            pattern_results = detector.detect_all_patterns(data)
        else:
            # Detect all patterns
            pattern_results = detector.detect_all_patterns(data)
          # Convert results to API format
        patterns_detected = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for pattern_name, result in pattern_results.items():
            # Only include detected patterns that meet threshold
            if not result.detected or result.confidence < request.confidence_threshold:
                continue
                
            # Determine pattern type from pattern_type enum
            if result.pattern_type.name.lower() == "bullish":
                pattern_type = "bullish"
                bullish_count += 1
            elif result.pattern_type.name.lower() == "bearish":
                pattern_type = "bearish" 
                bearish_count += 1
            else:
                pattern_type = "neutral"
                neutral_count += 1
            
            patterns_detected.append(PatternResult(
                pattern_name=result.name,
                confidence=result.confidence,
                pattern_type=pattern_type,
                detection_date=str(data.index[-1].date() if hasattr(data.index[-1], 'date') else data.index[-1]),
                description=result.description
            ))
        
        # Data info
        data_info = {
            "rows": len(data),
            "columns": list(data.columns),
            "date_range": {
                "start": str(data.index[0].date()) if hasattr(data.index[0], 'date') else str(data.index[0]),
                "end": str(data.index[-1].date()) if hasattr(data.index[-1], 'date') else str(data.index[-1])
            }
        }
        
        return PatternDetectionResponse(
            status="success",
            symbol=request.symbol,
            analysis_time=datetime.now(),
            patterns_detected=patterns_detected,
            total_patterns=len(patterns_detected),
            bullish_patterns=bullish_count,
            bearish_patterns=bearish_count,
            neutral_patterns=neutral_count,
            data_info=data_info
        )
        
    except Exception as e:
        logger.error(f"Pattern detection failed for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pattern detection failed: {str(e)}"
        )


@router.post("/test-data-loading")
async def test_data_loading() -> Dict[str, Any]:
    """Test data loading for debugging."""
    try:
        symbol = "CAT"
        data = await _load_stock_data(symbol)
        
        if data is None:
            return {"status": "error", "message": "No data found"}
        
        return {
            "status": "success",
            "symbol": symbol,
            "data_shape": data.shape,
            "columns": list(data.columns),
            "has_required_columns": all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']),
            "sample_data": data.head(2).to_dict()
        }
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/test-indicators")
async def test_indicators() -> Dict[str, Any]:
    """Test technical indicators for debugging."""
    try:
        # Load data
        symbol = "CAT"
        data = await _load_stock_data(symbol)
        
        if data is None:
            return {"status": "error", "message": "No data found"}
        
        # Test TechnicalIndicators initialization
        from core.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators(data)
        
        # Test simple RSI calculation
        rsi_values = ti.calculate_rsi()
        current_rsi = float(rsi_values.iloc[-1]) if not pd.isna(rsi_values.iloc[-1]) else None
        
        return {
            "status": "success",
            "symbol": symbol,
            "data_shape": data.shape,
            "rsi_calculation": "success",
            "current_rsi": current_rsi,
            "rsi_series_length": len(rsi_values)
        }
        
    except Exception as e:
        logger.error(f"Indicators test failed: {e}")
        return {"status": "error", "message": str(e), "type": type(e).__name__}


@router.post("/debug-columns")
async def debug_columns() -> Dict[str, Any]:
    """Debug endpoint to check column mapping in TechnicalIndicators."""
    try:
        # Load data
        symbol = "QQQ"
        csv_file = "c:\\dev\\stocktrader2\\data\\csv\\QQQ_data.csv"
        data = await _load_stock_data(symbol, csv_file)
        
        if data is None:
            return {"status": "error", "message": "No data found"}
        
        # Test TechnicalIndicators initialization
        from core.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators(data)
        
        # Check columns before and after
        original_columns = list(data.columns)
        ti_columns = list(ti.data.columns)
        
        # Try to calculate RSI with debug info
        rsi_error = None
        try:
            rsi_values = ti.calculate_rsi(length=14)
            rsi_success = True
            current_rsi = float(rsi_values.iloc[-1]) if not pd.isna(rsi_values.iloc[-1]) else None
        except Exception as e:
            rsi_success = False
            current_rsi = None
            rsi_error = str(e)
        
        result = {
            "status": "success",
            "symbol": symbol,
            "data_shape": data.shape,
            "original_columns": original_columns,
            "ti_columns": ti_columns,
            "rsi_calculation": {
                "success": rsi_success,
                "current_rsi": current_rsi
            }
        }
        
        if not rsi_success:
            result["rsi_calculation"]["error"] = rsi_error
        
        return result
        
    except Exception as e:
        logger.error(f"Debug columns test failed: {e}")
        return {"status": "error", "message": str(e), "type": type(e).__name__}


@router.post("/debug-rsi-simple")
async def debug_rsi_simple() -> Dict[str, Any]:
    """Debug RSI creation step by step."""
    try:
        # Load data
        symbol = "QQQ"
        csv_file = "c:\\dev\\stocktrader2\\data\\csv\\QQQ_data.csv"
        data = await _load_stock_data(symbol, csv_file)
        
        if data is None:
            return {"status": "error", "message": "No data found"}
        
        # Test TechnicalIndicators initialization
        from core.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators(data)
        
        # Calculate RSI
        rsi_values = ti.calculate_rsi(length=14)
        
        # Test TechnicalIndicatorData creation step by step
        if isinstance(rsi_values, pd.Series) and len(rsi_values) > 0:
            last_rsi = rsi_values.iloc[-1]
            current_rsi = float(last_rsi) if pd.notna(last_rsi) else None
            
            # Test the clean_nan_values step by step
            try:
                rsi_list = rsi_values.tail(10).tolist()
                
                # Test clean_nan_values on current_rsi
                cleaned_current = clean_nan_values(current_rsi)
                
                # Test clean_nan_values on rsi_list
                cleaned_values = clean_nan_values(rsi_list)
                
                return {
                    "status": "debug_step3",
                    "current_rsi": current_rsi,
                    "cleaned_current": cleaned_current,
                    "rsi_list_length": len(rsi_list),
                    "cleaned_values_length": len(cleaned_values),
                    "cleaned_values_sample": cleaned_values[:3] if len(cleaned_values) > 0 else []
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": "clean_nan_values failed",
                    "exception": str(e),
                    "current_rsi": current_rsi
                }
        else:
            return {"status": "error", "message": "RSI values not a Series or empty"}
        
    except Exception as e:
        return {"status": "error", "message": str(e), "type": type(e).__name__}


async def _load_stock_data(symbol: str, csv_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load stock data either from specified CSV file or latest downloaded data.
    
    Args:
        symbol: Stock symbol
        csv_file: Optional CSV file path
        
    Returns:
        DataFrame with OHLCV data or None if not found
    """
    try:
        if csv_file and os.path.exists(csv_file):
            # Load from specific CSV file
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # Ensure column names are standardized
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume',
                'Open': 'Open', 'High': 'High', 'Low': 'Low',
                'Close': 'Close', 'Volume': 'Volume'
            }
            
            data = data.rename(columns=column_mapping)
            return data
            
        else:
            # Find latest CSV file for the symbol
            data_dir = "data/csv"
            if not os.path.exists(data_dir):
                return None
                
            # Look for CSV files matching the symbol
            csv_files = [f for f in os.listdir(data_dir) if f.startswith(f"{symbol}_") and f.endswith(".csv")]
            
            if not csv_files:
                return None
                
            # Use the most recent file (assuming filename contains date)
            latest_file = sorted(csv_files)[-1]
            file_path = os.path.join(data_dir, latest_file)
            
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Ensure column names are standardized
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume',
                'Open': 'Open', 'High': 'High', 'Low': 'Low',
                'Close': 'Close', 'Volume': 'Volume'
            }
            
            data = data.rename(columns=column_mapping)
            return data
            
    except Exception as e:
        logger.error(f"Failed to load data for {symbol}: {e}")
        return None

# Utility functions for data cleaning
def clean_nan_values(value: Any) -> Any:
    """
    Clean NaN values from various data types for JSON serialization.
    
    Args:
        value: Value to clean (float, list, dict, etc.)
        
    Returns:
        Cleaned value with NaN replaced by None
    """
    if pd.isna(value):
        return None
    elif isinstance(value, list):
        return [clean_nan_values(item) for item in value]
    elif isinstance(value, dict):
        return {k: clean_nan_values(v) for k, v in value.items()}
    elif isinstance(value, (np.floating, float)) and (np.isnan(value) or np.isinf(value)):
        return None
    else:
        return value
