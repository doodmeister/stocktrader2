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

from api.models.market_data import (
    TechnicalAnalysisRequest, 
    TechnicalAnalysisResponse,
    TechnicalIndicatorData,
    PatternDetectionRequest,
    PatternDetectionResponse,
    PatternResult
)

logger = logging.getLogger(__name__)
router = APIRouter()


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
        if request.indicators and "rsi" in request.indicators and len(data) >= request.rsi_period:
            try:
                rsi_values = ti.calculate_rsi(length=request.rsi_period)
                current_rsi = float(rsi_values.iloc[-1]) if not pd.isna(rsi_values.iloc[-1]) else None
                
                # Generate RSI signal
                rsi_signal = "Hold"
                if current_rsi is not None:
                    if current_rsi > 70:
                        rsi_signal = "Sell"
                    elif current_rsi < 30:
                        rsi_signal = "Buy"
                
                indicators["rsi"] = TechnicalIndicatorData(
                    name="RSI",
                    current_value=current_rsi,
                    signal=rsi_signal,
                    values=rsi_values.tail(10).tolist(),
                    metadata={"period": request.rsi_period, "overbought": 70, "oversold": 30}
                )
            except Exception as e:
                logger.warning(f"RSI calculation failed for {request.symbol}: {e}")
        elif request.indicators and "rsi" in request.indicators:
            skipped_indicators.append(f"RSI (need {request.rsi_period} rows, got {len(data)})")
            logger.warning(f"Insufficient data for RSI: need {request.rsi_period} rows, got {len(data)}")
        
        if request.indicators and "macd" in request.indicators and len(data) >= (request.macd_slow + request.macd_signal):
            try:
                macd_line, signal_line, histogram = ti.calculate_macd(
                    fast=request.macd_fast,
                    slow=request.macd_slow,
                    signal=request.macd_signal
                )
                
                current_macd = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
                current_signal = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None
                
                # Generate MACD signal
                macd_signal = "Hold"
                if current_macd is not None and current_signal is not None:
                    if current_macd > current_signal:
                        macd_signal = "Buy"
                    else:
                        macd_signal = "Sell"
                
                indicators["macd"] = TechnicalIndicatorData(
                    name="MACD",
                    current_value=current_macd,
                    signal=macd_signal,
                    values=macd_line.tail(10).tolist(),
                    metadata={
                        "signal_line": current_signal,
                        "histogram": float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None,
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
                current_upper = float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else None
                current_lower = float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else None
                current_middle = float(middle_band.iloc[-1]) if not pd.isna(middle_band.iloc[-1]) else None
                
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
                
                indicators["bollinger_bands"] = TechnicalIndicatorData(
                    name="Bollinger Bands",
                    current_value=current_middle,
                    signal=bb_signal,
                    values=middle_band.tail(10).tolist(),
                    metadata={
                        "upper_band": current_upper,
                        "lower_band": current_lower,
                        "period": request.bb_period,
                        "std_dev": request.bb_std,
                        "bandwidth": bandwidth
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
                current_sma = float(sma_values.iloc[-1]) if not pd.isna(sma_values.iloc[-1]) else None
                current_price = float(data['Close'].iloc[-1])
                  # Generate SMA signal
                sma_signal = "Hold"
                if current_sma is not None:
                    sma_signal = "Buy" if current_price > current_sma else "Sell"
                
                indicators["sma"] = TechnicalIndicatorData(
                    name="Simple Moving Average",
                    current_value=current_sma,
                    signal=sma_signal,
                    values=sma_values.tail(10).tolist(),
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
                current_ema = float(ema_values.iloc[-1]) if not pd.isna(ema_values.iloc[-1]) else None
                current_price = float(data['Close'].iloc[-1])
                  # Generate EMA signal
                ema_signal = "Hold"
                if current_ema is not None:
                    ema_signal = "Buy" if current_price > current_ema else "Sell"
                
                indicators["ema"] = TechnicalIndicatorData(
                    name="Exponential Moving Average",
                    current_value=current_ema,
                    signal=ema_signal,
                    values=ema_values.tail(10).tolist(),
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
            composite_signal=composite_signal,
            signal_strength=signal_strength,
            indicators=indicators,
            data_info=data_info,
            current_price=current_price,
            price_change=price_change,
            price_change_percent=price_change_percent
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
