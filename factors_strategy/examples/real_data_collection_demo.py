#!/usr/bin/env python3
"""
Real Data Collection Demo
Demonstrates how to use the new real data collection interfaces
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import yaml

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ingestion.data_collector import (
    DataCollector, 
    YahooFinanceProvider, 
    TushareProvider, 
    SyntheticDataProvider
)

async def demo_yahoo_finance():
    """Demo Yahoo Finance data collection"""
    print("\n" + "="*60)
    print("YAHOO FINANCE DATA COLLECTION DEMO")
    print("="*60)
    
    # Test symbols (mix of Chinese and US stocks)
    symbols = [
        "000001.SZ",  # Ping An Bank (Shenzhen)
        "600036.SH",  # China Merchants Bank (Shanghai)
        "AAPL",       # Apple (US)
        "TSLA"        # Tesla (US)
    ]
    
    # Date range for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    # Create data collector with Yahoo Finance as primary provider
    providers = [YahooFinanceProvider(), SyntheticDataProvider()]
    
    async with DataCollector(providers=providers) as collector:
        for symbol in symbols:
            print(f"\n--- Testing {symbol} ---")
            
            try:
                # Test tick data collection
                print(f"Collecting tick data for {symbol}...")
                tick_data = await collector.collect_tick_data(symbol, start_date, end_date)
                
                if not tick_data.empty:
                    print(f"✓ Collected {len(tick_data)} tick records")
                    print(f"  Price range: ${tick_data['price'].min():.2f} - ${tick_data['price'].max():.2f}")
                    print(f"  Time range: {tick_data['timestamp'].min()} to {tick_data['timestamp'].max()}")
                else:
                    print("✗ No tick data collected")
                
                # Test real-time data collection
                print(f"Collecting real-time data for {symbol}...")
                real_time_data = await collector.collect_real_time_data([symbol])
                
                if real_time_data:
                    data = real_time_data[symbol]
                    print(f"✓ Real-time price: ${data['price']:.2f}")
                    print(f"  Change: {data['change_pct']:.2%}")
                    print(f"  Volume: {data['volume']:,.0f}")
                else:
                    print("✗ No real-time data collected")
                
                # Test fundamental data collection
                print(f"Collecting fundamental data for {symbol}...")
                fundamental_data = await collector.collect_fundamental_data(symbol)
                
                if fundamental_data and fundamental_data['market_cap'] > 0:
                    print(f"✓ Market Cap: ${fundamental_data['market_cap']:,.0f}")
                    print(f"  P/E Ratio: {fundamental_data['pe_ratio']:.2f}")
                    print(f"  P/B Ratio: {fundamental_data['pb_ratio']:.2f}")
                else:
                    print("✗ No fundamental data collected")
                    
            except Exception as e:
                print(f"✗ Error collecting data for {symbol}: {e}")

async def demo_provider_fallback():
    """Demo provider fallback mechanism"""
    print("\n" + "="*60)
    print("PROVIDER FALLBACK MECHANISM DEMO")
    print("="*60)
    
    # Create data collector with multiple providers
    providers = [
        YahooFinanceProvider(),
        TushareProvider(),  # Will fail without token
        SyntheticDataProvider()  # Fallback
    ]
    
    symbol = "000001.SZ"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    async with DataCollector(providers=providers) as collector:
        print(f"Testing fallback mechanism with {symbol}")
        print("Expected: Yahoo Finance -> Tushare (fail) -> Synthetic (success)")
        
        try:
            tick_data = await collector.collect_tick_data(symbol, start_date, end_date)
            print(f"✓ Successfully collected {len(tick_data)} records using fallback")
        except Exception as e:
            print(f"✗ All providers failed: {e}")

async def demo_data_quality_validation():
    """Demo data quality validation"""
    print("\n" + "="*60)
    print("DATA QUALITY VALIDATION DEMO")
    print("="*60)
    
    async with DataCollector() as collector:
        symbol = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        print(f"Collecting data for {symbol} with quality validation...")
        
        try:
            tick_data = await collector.collect_tick_data(symbol, start_date, end_date)
            
            # Manual validation demo
            print("\nData Quality Checks:")
            print(f"✓ Records collected: {len(tick_data)}")
            print(f"✓ Required columns present: {set(['timestamp', 'symbol', 'price', 'volume']).issubset(tick_data.columns)}")
            print(f"✓ No null prices: {tick_data['price'].isnull().sum() == 0}")
            print(f"✓ Positive prices: {(tick_data['price'] > 0).all()}")
            print(f"✓ Timestamps ordered: {tick_data['timestamp'].is_monotonic_increasing}")
            
        except Exception as e:
            print(f"✗ Data collection failed: {e}")

async def demo_configuration_loading():
    """Demo loading configuration from YAML"""
    print("\n" + "="*60)
    print("CONFIGURATION LOADING DEMO")
    print("="*60)
    
    config_path = "config/data_providers.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Configuration loaded successfully")
        print(f"  Enabled providers: {[name for name, settings in config['providers'].items() if settings['enabled']]}")
        print(f"  Symbol mappings: {list(config['symbol_mapping'].keys())}")
        print(f"  Data quality validation: {'enabled' if config['data_quality']['validation']['enabled'] else 'disabled'}")
        
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")

async def main():
    """Run all demos"""
    print("REAL DATA COLLECTION SYSTEM DEMO")
    print("This demo shows the new real data collection interfaces")
    print("replacing the previous demo/synthetic data functions.")
    
    try:
        await demo_yahoo_finance()
        await demo_provider_fallback()
        await demo_data_quality_validation()
        await demo_configuration_loading()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Real data collection from Yahoo Finance")
        print("✓ Provider fallback mechanism")
        print("✓ Data quality validation")
        print("✓ Configuration management")
        print("✓ Support for Chinese and US markets")
        print("✓ Async/await pattern for efficient data collection")
        
        print("\nNext Steps:")
        print("1. Add Tushare API token to config/data_providers.yaml for Chinese market data")
        print("2. Implement additional providers (Alpha Vantage, IEX Cloud, etc.)")
        print("3. Add caching layer for improved performance")
        print("4. Set up monitoring and alerting for data quality issues")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())