#!/usr/bin/env python3
"""
Test script for the analysis endpoint
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.routers.analysis import analyze_technical_indicators
from api.models.analysis import TechnicalIndicatorRequest


async def test_analysis_endpoint():
    """Test the technical indicators endpoint"""
    
    # Create a test request
    request = TechnicalIndicatorRequest(
        symbol="CAT",
        data_source="csv",
        csv_file_path="data/csv/CAT_20250526_20250626_60m.csv",  # Use specific file
        include_indicators=["rsi", "macd"],
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9
    )
    
    try:
        # Call the endpoint
        result = await analyze_technical_indicators(request)
        print("✅ Analysis endpoint test successful!")
        print(f"Symbol: {result.symbol}")
        print(f"Indicators returned: {len(result.indicators)}")
        print(f"Overall signal: {result.overall_signal}")
        print(f"Signal strength: {result.signal_strength}")
        
        # Print each indicator
        for indicator in result.indicators:
            print(f"- {indicator.name}: {indicator.signal} (strength: {indicator.strength})")
            
        return True
        
    except Exception as e:
        print(f"❌ Analysis endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_analysis_endpoint())
    sys.exit(0 if success else 1)
