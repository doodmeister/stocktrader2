#!/usr/bin/env python3
"""
Test script to verify imports work correctly from API context
"""

import sys
import os

# Add the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Testing imports...")

try:
    from core.technical_indicators import TechnicalIndicators
    print("✅ TechnicalIndicators imported successfully")
except Exception as e:
    print(f"❌ TechnicalIndicators import failed: {e}")

try:
    from patterns.orchestrator import CandlestickPatterns
    print("✅ CandlestickPatterns imported successfully")
except Exception as e:
    print(f"❌ CandlestickPatterns import failed: {e}")

try:
    # Test creating instances
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 50),
        'High': np.random.uniform(105, 115, 50), 
        'Low': np.random.uniform(95, 105, 50),
        'Close': np.random.uniform(100, 110, 50),
        'Volume': np.random.randint(1000, 10000, 50)
    }, index=dates)
    
    ti = TechnicalIndicators(data)
    print("✅ TechnicalIndicators instance created successfully")
    
    detector = CandlestickPatterns()
    print("✅ CandlestickPatterns instance created successfully")
    
    print("🎉 All imports and instantiation successful!")
    
except Exception as e:
    print(f"❌ Instance creation failed: {e}")
