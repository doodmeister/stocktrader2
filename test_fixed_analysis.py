#!/usr/bin/env python3
"""
Test script for the fixed analysis endpoint with proper file name handling
"""

import requests
import json
import os
from glob import glob

# Check available CSV files first
csv_dir = "data/csv"
if os.path.exists(csv_dir):
    csv_files = glob(os.path.join(csv_dir, "*.csv"))
    print("Available CSV files:")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")
    
    # Extract symbols from filenames
    symbols = []
    for f in csv_files:
        filename = os.path.basename(f)
        symbol = filename.split('_')[0]  # Get the symbol part before the first underscore
        if symbol not in symbols:
            symbols.append(symbol)
    
    print(f"\nAvailable symbols: {symbols}")
else:
    print(f"CSV directory {csv_dir} not found")
    exit(1)

# Test the endpoint with an available symbol
if symbols:
    test_symbol = symbols[0]  # Use the first available symbol
    print(f"\nTesting with symbol: {test_symbol}")
    
    # Test the RSI endpoint first
    print("\n1. Testing RSI endpoint...")
    try:
        response = requests.get(f"http://localhost:8000/api/analysis/test/rsi/{test_symbol}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"RSI calculation successful. Last RSI value: {data.get('rsi', 'N/A')}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
    
    # Test the multi-indicator endpoint
    print("\n2. Testing multi-indicator endpoint...")
    try:
        payload = {
            "symbol": test_symbol,
            "indicators": ["rsi", "macd"],
            "data_source": "csv"
        }
        
        response = requests.post(
            "http://localhost:8000/api/analysis/technical-indicators",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Multi-indicator analysis successful!")
            print(f"Indicators computed: {list(data.get('indicators', {}).keys())}")
            
            # Show sample RSI values
            if 'rsi' in data.get('indicators', {}):
                rsi_values = data['indicators']['rsi']['values']
                print(f"Sample RSI values: {rsi_values[-5:]}")  # Last 5 values
                
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
    
    print("\n3. Testing with an invalid symbol...")
    try:
        response = requests.get("http://localhost:8000/api/analysis/test/rsi/INVALID")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
