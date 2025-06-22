#!/usr/bin/env python3
"""
Quick test to verify the base.py imports are fixed.
"""

import pandas as pd
from core.indicators.base import validate_indicator_data

def test_base_import_fix():
    """Test that the imports in base.py are working correctly."""
    # Create a simple test DataFrame
    df = pd.DataFrame({
        'close': [100.0, 101.0, 102.0, 101.5, 103.0]
    })
    
    try:
        # Test basic validation
        validate_indicator_data(df, ['close'])
        print("✅ Basic validation passed")
        
        # Test OHLCV validation
        ohlcv_df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [101.0, 102.0, 103.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [100.5, 101.5, 102.5]
        })
        
        validate_indicator_data(ohlcv_df, ['Open', 'High', 'Low', 'Close'], check_ohlcv_coherence=True)
        print("✅ OHLCV validation passed")
        
        print("✅ All imports and functions working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_base_import_fix()
