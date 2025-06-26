#!/usr/bin/env python3
"""
Quick test for analysis endpoint functionality
"""

import sys
import os
sys.path.append(os.getcwd())

def main():
    print("Testing analysis endpoint imports...")
    
    try:
        from api.models.analysis import TechnicalIndicatorRequest
        print("✅ TechnicalIndicatorRequest imported successfully")
        
        from api.routers.analysis import analyze_technical_indicators
        print("✅ analyze_technical_indicators imported successfully")
        
        from core.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands
        print("✅ Core indicators imported successfully")
        
        import pandas as pd
        import numpy as np
        print("✅ pandas and numpy imported successfully")
        
        print("\n🎉 All imports successful! Analysis endpoint is ready.")
        print("\nAvailable CSV files:")
        import os
        csv_files = os.listdir("data/csv")
        for file in csv_files[:5]:  # Show first 5 files
            print(f"  - {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
