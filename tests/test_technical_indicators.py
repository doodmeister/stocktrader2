#!/usr/bin/env python3
"""Test import of TechnicalIndicators"""

try:
    from core.technical_indicators import TechnicalIndicators
    print("Success: TechnicalIndicators imported successfully")
    
    # Test with sample data
    import pandas as pd
    import numpy as np
    
    # Create sample OHLCV data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 50),
        'High': np.random.uniform(105, 115, 50), 
        'Low': np.random.uniform(95, 105, 50),
        'Close': np.random.uniform(100, 110, 50),
        'Volume': np.random.randint(1000, 10000, 50)
    }, index=dates)
    
    # Test TechnicalIndicators class
    ti = TechnicalIndicators(data)
    print(f"TechnicalIndicators initialized with {len(ti.data)} rows")
    
    # Test a few indicators
    rsi = ti.calculate_rsi()
    print(f"RSI calculated: {len(rsi)} values")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
