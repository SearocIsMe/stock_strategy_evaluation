#!/usr/bin/env python
"""
Test script to verify Tushare order book data collection
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.ingestion.data_collector import TushareProvider


async def test_tushare_order_book():
    """Test Tushare order book data collection"""
    
    # Check if token is set
    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        print("❌ TUSHARE_TOKEN environment variable not set!")
        print("Please run: export TUSHARE_TOKEN='your_token_here'")
        return
    
    print(f"✅ Using Tushare token: {token[:4]}...{token[-4:]}")
    
    # Initialize provider
    provider = TushareProvider(token)
    
    # Test symbols
    test_symbols = [
        '000001.SZ',  # 平安银行
        '600519.SH',  # 贵州茅台
        '000002.SZ',  # 万科A
    ]
    
    # Date range (recent dates for better data availability)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    print(f"\n📊 Testing order book data collection")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        
        try:
            # Get order book data
            order_book_data = await provider.get_order_book_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if order_book_data.empty:
                print(f"⚠️  No order book data returned for {symbol}")
            else:
                print(f"✅ Retrieved {len(order_book_data)} order book records")
                
                # Display sample data
                print("\n📋 Sample order book data (first record):")
                first_row = order_book_data.iloc[0]
                
                # Show bid levels
                print("\n  Bid Levels:")
                for i in range(1, 6):  # Show first 5 levels
                    bid_price = first_row.get(f'bid_price_{i}', 'N/A')
                    bid_volume = first_row.get(f'bid_volume_{i}', 'N/A')
                    print(f"    Level {i}: Price={bid_price}, Volume={bid_volume}")
                
                # Show ask levels
                print("\n  Ask Levels:")
                for i in range(1, 6):  # Show first 5 levels
                    ask_price = first_row.get(f'ask_price_{i}', 'N/A')
                    ask_volume = first_row.get(f'ask_volume_{i}', 'N/A')
                    print(f"    Level {i}: Price={ask_price}, Volume={ask_volume}")
                
                # Show summary
                print(f"\n  Mid Price: {first_row.get('mid_price', 'N/A')}")
                print(f"  Spread: {first_row.get('spread', 'N/A')}")
                print(f"  Total Bid Volume: {first_row.get('total_bid_volume', 'N/A')}")
                print(f"  Total Ask Volume: {first_row.get('total_ask_volume', 'N/A')}")
                
                # Check data quality
                print("\n  Data Quality Check:")
                has_all_levels = all(f'bid_price_{i}' in order_book_data.columns for i in range(1, 11))
                print(f"    Has all 10 levels: {'✅' if has_all_levels else '❌'}")
                
                non_zero_bids = (order_book_data['bid_price_1'] > 0).sum()
                print(f"    Non-zero bid prices: {non_zero_bids}/{len(order_book_data)}")
                
                valid_spreads = (order_book_data['spread'] > 0).sum()
                print(f"    Valid spreads: {valid_spreads}/{len(order_book_data)}")
                
        except Exception as e:
            print(f"❌ Error collecting order book data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Order book data collection test completed")


async def test_api_methods():
    """Test different Tushare API methods to see what works"""
    
    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        print("❌ TUSHARE_TOKEN not set!")
        return
    
    print("\n🧪 Testing different Tushare API methods...")
    print("=" * 60)
    
    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()
    
    test_symbol = '600519.SH'
    test_date = datetime.now().strftime('%Y%m%d')
    
    # Test 1: realtime_quote (new primary method)
    print("\n1. Testing realtime_quote API...")
    try:
        df = pro.realtime_quote(ts_code=test_symbol, src='sina')
        print(f"✅ realtime_quote returned {len(df)} records")
        if not df.empty:
            print(f"   Columns: {', '.join(df.columns[:20])}...")
            # Show bid/ask columns
            bid_ask_cols = [col for col in df.columns if 'bid' in col or 'ask' in col]
            print(f"   Bid/Ask columns: {', '.join(bid_ask_cols)}")
            # Show sample data
            if 'bid1' in df.columns:
                print(f"   Sample: bid1={df['bid1'].iloc[0]}, ask1={df['ask1'].iloc[0]}")
    except Exception as e:
        print(f"❌ realtime_quote failed: {e}")
    
    # Test 2: daily with extended fields
    print("\n2. Testing daily API...")
    try:
        df = pro.daily(ts_code=test_symbol, trade_date=test_date)
        print(f"✅ daily returned {len(df)} records")
        if not df.empty:
            print(f"   Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"❌ daily failed: {e}")
    
    # Test 3: tick API (alternative method)
    print("\n3. Testing tick API...")
    try:
        df = pro.tick(
            ts_code=test_symbol,
            start_date=test_date + ' 09:00:00',
            end_date=test_date + ' 15:30:00'
        )
        print(f"✅ tick returned {len(df)} records")
        if not df.empty:
            print(f"   Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"❌ tick failed: {e}")
    
    # Test 4: get_realtime_quotes (basic API - alternative)
    print("\n4. Testing get_realtime_quotes (basic API)...")
    try:
        df = ts.get_realtime_quotes(test_symbol.split('.')[0])
        print(f"✅ get_realtime_quotes returned {len(df)} records")
        if not df.empty:
            print(f"   Columns: {', '.join(df.columns[:10])}...")
            # Show bid/ask columns
            bid_ask_cols = [col for col in df.columns if 'b' in col or 'a' in col]
            print(f"   Bid/Ask columns: {', '.join(bid_ask_cols[:10])}...")
    except Exception as e:
        print(f"❌ get_realtime_quotes failed: {e}")


def main():
    """Main entry point"""
    print("🚀 Tushare Order Book Data Collection Test")
    print("=" * 60)
    
    # Run order book test
    asyncio.run(test_tushare_order_book())
    
    # Run API method tests
    asyncio.run(test_api_methods())


if __name__ == "__main__":
    main()