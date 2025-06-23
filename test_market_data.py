#!/usr/bin/env python3
"""
Test script to verify yfinance market data download functionality
"""

import sys
from datetime import date, timedelta
from utils.data_downloader import download_stock_data, fetch_daily_ohlcv
from api.services.market_data_service_enhanced import MarketDataService

def test_yfinance_basic():
    """Test basic yfinance functionality"""
    print("Testing yfinance basic functionality...")
    
    # Test single symbol download
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Test with AAPL
        print(f"Downloading AAPL data from {start_date} to {end_date}")
        data = download_stock_data(
            symbols=["AAPL"],
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        if data:
            print(f"✓ Successfully downloaded data for {list(data.keys())}")
            for symbol, df in data.items():
                print(f"  {symbol}: {len(df)} rows, columns: {list(df.columns)}")
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
                print(f"  Sample data:\n{df.head(2)}")
        else:
            print("✗ No data returned")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        return False
    
    return True

def test_market_service():
    """Test the enhanced market data service"""
    print("\nTesting MarketDataService...")
    
    try:
        service = MarketDataService(data_directory="data/test")
        
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        print(f"Downloading MSFT data from {start_date} to {end_date}")
        data = service.download_and_save_stock_data(
            symbols="MSFT",
            start_date=start_date,
            end_date=end_date,
            save_csv=True
        )
        
        if data:
            print(f"✓ Successfully downloaded via service: {list(data.keys())}")
            for symbol, df in data.items():
                print(f"  {symbol}: {len(df)} rows")
        else:
            print("✗ No data returned from service")
            return False
            
    except Exception as e:
        print(f"✗ Error with market service: {e}")
        return False
    
    return True

def test_multiple_symbols():
    """Test downloading multiple symbols"""
    print("\nTesting multiple symbol download...")
    
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=5)
        
        symbols = ["AAPL", "GOOGL", "TSLA"]
        print(f"Downloading {symbols} from {start_date} to {end_date}")
        
        data = download_stock_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        if data:
            print(f"✓ Successfully downloaded: {list(data.keys())}")
            for symbol, df in data.items():
                print(f"  {symbol}: {len(df)} rows")
        else:
            print("✗ No data returned for multiple symbols")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading multiple symbols: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Market Data Download Test")
    print("=" * 40)
    
    # Run tests
    test1_passed = test_yfinance_basic()
    test2_passed = test_market_service()
    test3_passed = test_multiple_symbols()
    
    print("\n" + "=" * 40)
    print("TEST RESULTS:")
    print(f"Basic yfinance test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Market service test: {'PASS' if test2_passed else 'FAIL'}")
    print(f"Multiple symbols test: {'PASS' if test3_passed else 'FAIL'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    sys.exit(0 if all_passed else 1)
