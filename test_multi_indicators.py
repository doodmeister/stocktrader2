#!/usr/bin/env python3
"""
Test script for multi-indicator analysis endpoint
"""

import sys
import os
sys.path.append(os.getcwd())

import asyncio
from api.models.analysis import TechnicalIndicatorRequest
from api.routers.analysis import analyze_technical_indicators

async def test_multi_indicators():
    """Test the technical indicators endpoint with multiple indicators."""
    
    # Create request with multiple indicators
    request = TechnicalIndicatorRequest(
        symbol="CAT",
        data_source="csv", 
        csv_file_path="data/csv/CAT_20250526_20250626_60m.csv",
        period="1y",
        include_indicators=["rsi", "macd", "bollinger_bands", "stochastic", "williams_r"],
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bb_period=20,
        bb_std_dev=2.0
    )
    
    try:
        # Call the endpoint
        result = await analyze_technical_indicators(request)
        
        print("✅ Multi-indicator analysis test successful!")
        print(f"Symbol: {result.symbol}")
        print(f"Indicators returned: {len(result.indicators)}")
        print(f"Overall signal: {result.overall_signal}")
        print(f"Signal strength: {result.signal_strength}")
        print("\nIndicator details:")
        for indicator in result.indicators:
            print(f"- {indicator.name}: {indicator.signal} (strength: {indicator.strength})")
            if isinstance(indicator.current_value, dict):
                print(f"  Current values: {indicator.current_value}")
            else:
                print(f"  Current value: {indicator.current_value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-indicator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_multi_indicators())
    sys.exit(0 if success else 1)
