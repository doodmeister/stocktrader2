"""
FastAPI router for analysis endpoints - SIMPLIFIED VERSION.

This module provides REST API endpoints for technical analysis.
Focus on RSI indicator first to ensure the basic structure works.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, Any
import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

# Import models from the models package
from api.models.analysis import (
    TechnicalIndicatorRequest,
    TechnicalIndicatorResponse,
    IndicatorResult,
    PatternDetectionRequest,
    PatternDetectionResponse,
    DetectedPattern
)

# Core functionality imports - use core.indicators for all technical indicators
from core.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_stochastic,
    calculate_williams_r,
    calculate_cci,
    calculate_vwap,
    calculate_obv,
    calculate_adx
)
# Import pattern detection
from patterns.orchestrator import CandlestickPatterns

from core.data_validator import DataValidator

# Import SMA/EMA from technical_indicators (they aren't in core.indicators yet)
from core.technical_indicators import calculate_sma, calculate_ema

# Pattern detection imports

# Additional model imports

# Utility imports
from utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)

# Initialize router
router = APIRouter(tags=["analysis"])

# DATA_PATH for CSV loading
DATA_PATH = "data/csv"

def _find_symbol_csv(symbol: str) -> Optional[str]:
    """
    Find the most recent CSV file for a given symbol.
    Files are expected to be named like: SYMBOL_YYYYMMDD_YYYYMMDD_interval.csv
    """
    pattern = os.path.join(DATA_PATH, f"{symbol}_*.csv")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        return None
    
    # Sort files by modification time (most recent first)
    matching_files.sort(key=os.path.getmtime, reverse=True)
    return matching_files[0]


def clean_nan_values(data: Any) -> Any:
    """
    Clean NaN values from data for JSON serialization.
    """
    try:
        if isinstance(data, dict):
            return {k: clean_nan_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_nan_values(item) for item in data]
        elif isinstance(data, (pd.Series, np.ndarray)):
            if hasattr(data, 'tolist'):
                data_list = data.tolist()
            else:
                data_list = list(data)
            return [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x for x in data_list]
        elif isinstance(data, (int, float)):
            if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
                return None
            return data
        else:
            return data
    except Exception as e:
        logger.error(f"Error cleaning NaN values: {e}")
        return None


def _load_stock_data(symbol: str, data_source: str = "csv", csv_file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load stock data from CSV.
    """
    try:
        if data_source == "csv":
            if csv_file_path:
                file_path = csv_file_path
            else:
                # Try to find a CSV file for this symbol
                file_path = _find_symbol_csv(symbol)
                if not file_path:
                    # Fallback to the old naming pattern
                    file_path = os.path.join(DATA_PATH, f"{symbol}.csv")
            
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"CSV file not found for symbol '{symbol}'. Available files in {DATA_PATH}: {[f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]}"
                )
            
            df = pd.read_csv(file_path)
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required columns: {missing_columns}"
                )
            
            # Convert date column if it exists
            datetime_columns = ['datetime', 'date', 'timestamp']
            date_column = None
            for col in datetime_columns:
                if col in df.columns:
                    date_column = col
                    break
            
            if date_column:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
            elif df.index.name not in datetime_columns:
                if not isinstance(df.index, pd.DatetimeIndex):
                    # If no datetime column found, create a simple range index but log it
                    logger.warning(f"No datetime column found in CSV for {symbol}. Using range index.")
                    df.index = pd.RangeIndex(len(df))
        else:
            raise HTTPException(
                status_code=400,
                detail="Only CSV data source supported in this simplified version"
            )
        
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No data available for symbol: {symbol}"
            )
        
        return df
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load stock data: {str(e)}"
        )


@router.post("/pattern-detection", response_model=PatternDetectionResponse)
async def detect_patterns(request: PatternDetectionRequest):
    """
    Detects candlestick patterns in stock data.
    """
    try:
        logger.info(f"Received pattern detection request: {request.model_dump_json(indent=2)}")
        
        df = _load_stock_data(
            request.symbol,
            request.data_source,
            request.csv_file_path
        )
        
        logger.info(f"Loaded data shape: {df.shape}, columns: {list(df.columns)}")

        detector = CandlestickPatterns(confidence_threshold=request.min_confidence)
        logger.info(f"Created detector with confidence threshold: {request.min_confidence}")
        
        # This method needs to be created in the orchestrator
        # It should return a list of all occurrences
        occurrences = detector.get_pattern_occurrences(df)
        logger.info(f"Raw occurrences found: {len(occurrences)}")
        
        if occurrences:
            logger.info(f"Sample occurrence: {occurrences[0]}")

        # Filter by date if requested, but NOT for CSV data source
        if request.data_source != 'csv' and request.recent_only and request.lookback_days > 0:
            logger.info(f"Applying date filter for the last {request.lookback_days} days.")
            # Ensure the date column is in datetime format for comparison
            try:
                # Assuming 'date' is a string 'YYYY-MM-DD' or similar
                cutoff_date = datetime.now() - timedelta(days=request.lookback_days)
                occurrences = [
                    p for p in occurrences 
                    if pd.to_datetime(p['date']) >= cutoff_date
                ]
                logger.info(f"After date filtering: {len(occurrences)} occurrences")
            except Exception as e:
                logger.error(f"Error during date filtering: {e}. Skipping.")
        elif request.data_source == 'csv':
            logger.info("Skipping date filtering because data source is CSV.")
        else:
            logger.info("No date filtering applied (recent_only=False or lookback_days=0).")

        # Filter by pattern name if requested
        if request.include_patterns:
            occurrences = [
                p for p in occurrences
                if p['pattern_name'] in request.include_patterns
            ]
            logger.info(f"After pattern name filtering: {len(occurrences)} occurrences")

        # Convert to DetectedPattern objects
        try:
            detected_patterns = [DetectedPattern(**p) for p in occurrences]
            logger.info(f"Successfully converted {len(detected_patterns)} DetectedPattern objects")
        except Exception as conversion_error:
            logger.error(f"Error converting to DetectedPattern: {conversion_error}")
            logger.error(f"Sample problematic occurrence: {occurrences[0] if occurrences else 'None'}")
            raise

        # Compute pattern summary (always include, even if empty)
        bullish = 0
        bearish = 0
        neutral = 0
        for p in detected_patterns:
            pt = getattr(p, 'pattern_type', '').lower()
            if 'bullish' in pt:
                bullish += 1
            elif 'bearish' in pt:
                bearish += 1
            else:
                neutral += 1
        pattern_summary = {
            'total_patterns': len(detected_patterns),
            'bullish_patterns': bullish,
            'bearish_patterns': bearish,
            'neutral_patterns': neutral
        }

        return PatternDetectionResponse(
            symbol=request.symbol,
            total_patterns_found=len(detected_patterns),
            patterns=detected_patterns,
            analysis_timestamp=datetime.utcnow(),
            pattern_summary=pattern_summary
        )
    except Exception as e:
        logger.error(f"Error in pattern detection for {request.symbol}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to perform pattern detection: {e}")


@router.post("/technical-indicators", response_model=TechnicalIndicatorResponse)
async def analyze_technical_indicators(request: TechnicalIndicatorRequest):
    """
    Analyze technical indicators for a stock.
    
    configuration and returns structured results with signals and data.
    """
    try:
        logger.info(f"Processing technical indicators for {request.symbol}")
        
        # Load data
        df = _load_stock_data(
            request.symbol,
            request.data_source,
            request.csv_file_path
        )
        
        indicators = []
        
        # Calculate each requested indicator
        for indicator_name in request.include_indicators:
            try:
                if indicator_name == "rsi":
                    rsi_data = calculate_rsi(df, length=request.rsi_period)
                    current_rsi = rsi_data.iloc[-1] if not rsi_data.empty else None
                    
                    # Determine signal
                    signal = "neutral"
                    strength = 0.5
                    if current_rsi is not None and not np.isnan(current_rsi):
                        if current_rsi > 70:
                            signal = "bearish"
                            strength = min((current_rsi - 70) / 30, 1.0)
                        elif current_rsi < 30:
                            signal = "bullish"
                            strength = min((30 - current_rsi) / 30, 1.0)
                    
                    indicators.append(IndicatorResult(
                        name="RSI",
                        current_value=current_rsi,
                        signal=signal,
                        strength=strength,
                        data=clean_nan_values([{
                            "date": str(df.index[idx]) if isinstance(df.index, pd.DatetimeIndex) and idx < len(df) else f"Point {idx + 1}",
                            "value": val
                        } for idx, val in enumerate(rsi_data) if pd.notna(val)])
                    ))
                
                elif indicator_name == "macd":
                    macd_line, signal_line, histogram = calculate_macd(
                        df,
                        fast=request.macd_fast,
                        slow=request.macd_slow,
                        signal=request.macd_signal
                    )
                    
                    current_macd = macd_line.iloc[-1] if not macd_line.empty else None
                    current_signal = signal_line.iloc[-1] if not signal_line.empty else None
                    current_histogram = histogram.iloc[-1] if not histogram.empty else None
                    
                    # Determine signal
                    signal = "neutral"
                    strength = 0.5
                    if current_macd is not None and current_signal is not None:
                        if not np.isnan(current_macd) and not np.isnan(current_signal):
                            if current_macd > current_signal:
                                signal = "bullish"
                                strength = min(abs(current_macd - current_signal) / abs(current_signal), 1.0) if current_signal != 0 else 0.7
                            elif current_macd < current_signal:
                                signal = "bearish"
                                strength = min(abs(current_macd - current_signal) / abs(current_signal), 1.0) if current_signal != 0 else 0.7
                    
                    # Clean current values for JSON serialization
                    clean_macd = None if current_macd is None or np.isnan(current_macd) else float(current_macd)
                    clean_signal = None if current_signal is None or np.isnan(current_signal) else float(current_signal)
                    clean_histogram = None if current_histogram is None or np.isnan(current_histogram) else float(current_histogram)
                    
                    indicators.append(IndicatorResult(
                        name="MACD",
                        current_value={
                            "macd": clean_macd if clean_macd is not None else 0.0,
                            "signal": clean_signal if clean_signal is not None else 0.0,
                            "histogram": clean_histogram if clean_histogram is not None else 0.0
                        },
                        signal=signal,
                        strength=strength,
                        data=clean_nan_values([{
                            "date": str(df.index[idx]) if isinstance(df.index, pd.DatetimeIndex) and idx < len(df) else f"Point {idx + 1}",
                            "macd": macd_val,
                            "signal": signal_val,
                            "histogram": hist_val
                        } for idx, (macd_val, signal_val, hist_val) in enumerate(zip(macd_line, signal_line, histogram))])
                    ))
                
                elif indicator_name == "bollinger_bands":
                    # Fix: calculate_bollinger_bands returns tuple of (upper, middle, lower)
                    upper_band, middle_band, lower_band = calculate_bollinger_bands(
                        df,
                        length=request.bb_period,
                        std=request.bb_std_dev
                    )
                    
                    current_upper = upper_band.iloc[-1] if not upper_band.empty else None
                    current_lower = lower_band.iloc[-1] if not lower_band.empty else None
                    current_middle = middle_band.iloc[-1] if not middle_band.empty else None
                    current_price = df['close'].iloc[-1]
                    
                    # Determine signal
                    signal = "neutral"
                    strength = 0.5
                    if all(x is not None and not np.isnan(x) for x in [current_upper, current_lower, current_price]):
                        if current_price > current_upper:
                            signal = "bearish"
                            strength = 0.8
                        elif current_price < current_lower:
                            signal = "bullish"
                            strength = 0.8
                    
                    indicators.append(IndicatorResult(
                        name="Bollinger Bands",
                        current_value={
                            "upper_band": float(current_upper) if current_upper is not None and not np.isnan(current_upper) else 0.0,
                            "middle_band": float(current_middle) if current_middle is not None and not np.isnan(current_middle) else 0.0,
                            "lower_band": float(current_lower) if current_lower is not None and not np.isnan(current_lower) else 0.0,
                            "current_price": float(current_price) if not np.isnan(current_price) else 0.0
                        },
                        signal=signal,
                        strength=strength,
                        data=clean_nan_values([{
                            "date": str(df.index[idx]) if isinstance(df.index, pd.DatetimeIndex) and idx < len(df) else f"Point {idx + 1}",
                            "upper_band": float(upper_val) if pd.notna(upper_val) else None,
                            "middle_band": float(middle_val) if pd.notna(middle_val) else None,
                            "lower_band": float(lower_val) if pd.notna(lower_val) else None
                        } for idx, (upper_val, middle_val, lower_val) in enumerate(zip(upper_band, middle_band, lower_band))])
                    ))
                
                elif indicator_name == "sma":
                    for period in request.sma_periods:
                        sma_data = calculate_sma(df, length=period)
                        current_sma = sma_data.iloc[-1] if not sma_data.empty else None
                        current_price = df['close'].iloc[-1]
                        
                        signal = "neutral"
                        strength = 0.5
                        if current_sma is not None and not np.isnan(current_sma):
                            if current_price > current_sma:
                                signal = "bullish"
                                strength = 0.6
                            elif current_price < current_sma:
                                signal = "bearish"
                                strength = 0.6
                        
                        indicators.append(IndicatorResult(
                            name=f"SMA_{period}",
                            current_value=current_sma,
                            signal=signal,
                            strength=strength,
                            data=clean_nan_values([{
                                "date": str(idx),
                                "value": val
                            } for idx, val in sma_data.items() if pd.notna(val)])
                        ))
                
                elif indicator_name == "ema":
                    for period in request.ema_periods:
                        ema_data = calculate_ema(df, length=period)
                        current_ema = ema_data.iloc[-1] if not ema_data.empty else None
                        current_price = df['close'].iloc[-1]
                        
                        signal = "neutral"
                        strength = 0.5
                        if current_ema is not None and not np.isnan(current_ema):
                            if current_price > current_ema:
                                signal = "bullish"
                                strength = 0.6
                            elif current_price < current_ema:
                                signal = "bearish"
                                strength = 0.6
                        
                        indicators.append(IndicatorResult(
                            name=f"EMA_{period}",
                            current_value=current_ema,
                            signal=signal,
                            strength=strength,
                            data=clean_nan_values([{
                                "date": str(idx),
                                "value": val
                            } for idx, val in ema_data.items() if pd.notna(val)])
                        ))
                
                elif indicator_name == "stochastic":
                    # Fix: calculate_stochastic returns tuple of (k_series, d_series)
                    k_series, d_series = calculate_stochastic(df)
                    current_k = k_series.iloc[-1] if not k_series.empty else None
                    current_d = d_series.iloc[-1] if not d_series.empty else None
                    
                    signal = "neutral"
                    strength = 0.5
                    if current_k is not None and not np.isnan(current_k):
                        if current_k > 80:
                            signal = "bearish"
                            strength = 0.7
                        elif current_k < 20:
                            signal = "bullish"
                            strength = 0.7
                    
                    indicators.append(IndicatorResult(
                        name="Stochastic",
                        current_value={
                            "k": float(current_k) if current_k is not None and not np.isnan(current_k) else 0.0,
                            "d": float(current_d) if current_d is not None and not np.isnan(current_d) else 0.0
                        },
                        signal=signal,
                        strength=strength,
                        data=clean_nan_values([{
                            "date": str(idx),
                            "k": float(k_val) if pd.notna(k_val) else None,
                            "d": float(d_val) if pd.notna(d_val) else None
                        } for idx, (k_val, d_val) in enumerate(zip(k_series, d_series))])
                    ))
                
                elif indicator_name == "williams_r":
                    wr_data = calculate_williams_r(df)
                    current_wr = wr_data.iloc[-1] if not wr_data.empty else None
                    
                    signal = "neutral"
                    strength = 0.5
                    if current_wr is not None and not np.isnan(current_wr):
                        if current_wr > -20:
                            signal = "bearish"
                            strength = 0.7
                        elif current_wr < -80:
                            signal = "bullish"
                            strength = 0.7
                    
                    indicators.append(IndicatorResult(
                        name="Williams %R",
                        current_value=current_wr,
                        signal=signal,
                        strength=strength,
                        data=clean_nan_values([{
                            "date": str(idx),
                            "value": val
                        } for idx, val in wr_data.items() if pd.notna(val)])
                    ))
                
                elif indicator_name == "cci":
                    cci_data = calculate_cci(df)
                    current_cci = cci_data.iloc[-1] if not cci_data.empty else None
                    
                    signal = "neutral"
                    strength = 0.5
                    if current_cci is not None and not np.isnan(current_cci):
                        if current_cci > 100:
                            signal = "bearish"
                            strength = min(abs(current_cci - 100) / 100, 1.0)
                        elif current_cci < -100:
                            signal = "bullish"
                            strength = min(abs(current_cci + 100) / 100, 1.0)
                    
                    indicators.append(IndicatorResult(
                        name="CCI",
                        current_value=current_cci,
                        signal=signal,
                        strength=strength,
                        data=clean_nan_values([{
                            "date": str(idx),
                            "value": val
                        } for idx, val in cci_data.items() if pd.notna(val)])
                    ))
                
                elif indicator_name == "vwap":
                    vwap_data = calculate_vwap(df)
                    current_vwap = vwap_data.iloc[-1] if not vwap_data.empty else None
                    current_price = df['close'].iloc[-1]
                    
                    signal = "neutral"
                    strength = 0.5
                    if current_vwap is not None and not np.isnan(current_vwap):
                        if current_price > current_vwap:
                            signal = "bullish"
                            strength = 0.6
                        elif current_price < current_vwap:
                            signal = "bearish"
                            strength = 0.6
                    
                    indicators.append(IndicatorResult(
                        name="VWAP",
                        current_value=current_vwap,
                        signal=signal,
                        strength=strength,
                        data=clean_nan_values([{
                            "date": str(idx),
                            "value": val
                        } for idx, val in vwap_data.items() if pd.notna(val)])
                    ))
                
                elif indicator_name == "obv":
                    obv_data = calculate_obv(df)
                    current_obv = obv_data.iloc[-1] if not obv_data.empty else None
                    
                    # OBV signal is typically based on trend
                    signal = "neutral"
                    strength = 0.5
                    if len(obv_data) > 5:
                        recent_trend = obv_data.iloc[-5:].diff().mean()
                        if recent_trend > 0:
                            signal = "bullish"
                            strength = 0.6
                        elif recent_trend < 0:
                            signal = "bearish"
                            strength = 0.6
                    
                    indicators.append(IndicatorResult(
                        name="OBV",
                        current_value=current_obv,
                        signal=signal,
                        strength=strength,
                        data=clean_nan_values([{
                            "date": str(idx),
                            "value": val
                        } for idx, val in obv_data.items() if pd.notna(val)])
                    ))
                
                elif indicator_name == "adx":
                    # Fix: calculate_adx returns tuple of (adx_series, plus_di_series, minus_di_series)
                    adx_series, plus_di_series, minus_di_series = calculate_adx(df)
                    current_adx = adx_series.iloc[-1] if not adx_series.empty else None
                    current_plus_di = plus_di_series.iloc[-1] if not plus_di_series.empty else None
                    current_minus_di = minus_di_series.iloc[-1] if not minus_di_series.empty else None
                    
                    signal = "neutral"
                    strength = 0.5
                    if (current_adx is not None and not np.isnan(current_adx) and 
                        current_plus_di is not None and not np.isnan(current_plus_di) and
                        current_minus_di is not None and not np.isnan(current_minus_di)):
                        
                        if current_adx > 25:  # Strong trend
                            if current_plus_di > current_minus_di:
                                signal = "bullish"
                                strength = min(current_adx / 50, 1.0)
                            else:
                                signal = "bearish"
                                strength = min(current_adx / 50, 1.0)
                    
                    indicators.append(IndicatorResult(
                        name="ADX",
                        current_value={
                            "adx": float(current_adx) if current_adx is not None and not np.isnan(current_adx) else 0.0,
                            "plus_di": float(current_plus_di) if current_plus_di is not None and not np.isnan(current_plus_di) else 0.0,
                            "minus_di": float(current_minus_di) if current_minus_di is not None and not np.isnan(current_minus_di) else 0.0
                        },
                        signal=signal,
                        strength=strength,
                        data=clean_nan_values([{
                            "date": str(idx),
                            "adx": float(adx_val) if pd.notna(adx_val) else None,
                            "plus_di": float(plus_di_val) if pd.notna(plus_di_val) else None,
                            "minus_di": float(minus_di_val) if pd.notna(minus_di_val) else None
                        } for idx, (adx_val, plus_di_val, minus_di_val) in enumerate(zip(adx_series, plus_di_series, minus_di_series))])
                    ))
                
                else:
                    logger.warning(f"Unknown indicator: {indicator_name}")
                    
            except Exception as e:
                logger.error(f"Error calculating {indicator_name}: {e}")
        
        # Calculate overall signal
        bullish_signals = sum(1 for ind in indicators if ind.signal == "bullish")
        bearish_signals = sum(1 for ind in indicators if ind.signal == "bearish")
        total_signals = len(indicators)
        
        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
            signal_strength = bullish_signals / total_signals if total_signals > 0 else 0.5
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
            signal_strength = bearish_signals / total_signals if total_signals > 0 else 0.5
        else:
            overall_signal = "neutral"
            signal_strength = 0.5
        
        return TechnicalIndicatorResponse(
            symbol=request.symbol,
            analysis_timestamp=datetime.now(),
            data_period=request.period or "N/A",
            total_records=len(df),
            indicators=indicators,
            overall_signal=overall_signal,
            signal_strength=signal_strength
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform technical analysis: {e}")


@router.get("/test/rsi/{symbol}")
async def test_rsi(symbol: str):
    """Simple test endpoint for RSI calculation."""
    try:
        df = _load_stock_data(symbol, "csv")
        rsi_data = calculate_rsi(df, length=14)
        return {"symbol": symbol, "rsi": clean_nan_values(rsi_data.tolist())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}
