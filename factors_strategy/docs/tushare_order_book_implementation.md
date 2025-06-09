# Tushare Order Book Data Implementation

## Overview

This document describes the implementation of order book data collection using Tushare API.

## API Methods Used

### 1. **realtime_quote** (Primary Method)
```python
df = pro.realtime_quote(
    ts_code='600519.SH',
    src='sina'  # Use sina as data source
)
```

This API provides:
- 5 levels of bid prices and volumes (bid1-bid5, bid1_vol-bid5_vol)
- 5 levels of ask prices and volumes (ask1-ask5, ask1_vol-ask5_vol)
- Real-time market data
- Works for both current and historical dates

### 2. **tick** (Alternative Method)
```python
df = pro.tick(
    ts_code='600519.SH',
    start_date='20250607 09:00:00',
    end_date='20250607 15:30:00'
)
```

This provides:
- Tick-level data with bid/ask information
- More granular time series data
- Used when realtime_quote fails

### 3. **daily** (Fallback)
```python
df = pro.daily(ts_code='600519.SH', trade_date='20250607')
```

When Level 2 data is not available, we use daily data to generate synthetic order book levels.

### 4. **get_realtime_quotes** (Basic API Alternative)
```python
df = ts.get_realtime_quotes('600519')  # Without exchange suffix
```

This provides:
- Alternative real-time data source
- Different data format but includes bid/ask information

## Data Format Conversion

The implementation handles multiple data formats:

### From realtime_quote API (Primary):
- `bid1`, `bid2`, ... `bid5` → `bid_price_1`, `bid_price_2`, ... `bid_price_5`
- `bid1_vol`, `bid2_vol`, ... → `bid_volume_1`, `bid_volume_2`, ...
- `ask1`, `ask2`, ... `ask5` → `ask_price_1`, `ask_price_2`, ... `ask_price_5`
- `ask1_vol`, `ask2_vol`, ... → `ask_volume_1`, `ask_volume_2`, ...
- Levels 6-10 are synthetically generated based on level 5 spread

### From tick API:
- `bid`, `ask` → `bid_price_1`, `ask_price_1`
- `bid_vol`, `ask_vol` → `bid_volume_1`, `ask_volume_1`
- Levels 2-10 are synthetically generated

### Synthetic Generation:
When only 5 levels are available:
- Uses level 5 spread to extrapolate levels 6-10
- Volume decreases by 20% for each additional level
- Spread increases by 50% of base spread for each level

When only daily data is available:
- Uses 0.1% of price as base spread
- Generates all 10 levels synthetically
- Volume distributed based on daily volume

## Error Handling

The implementation includes robust error handling:

1. **Primary Method Failure**: Falls back to daily data
2. **Missing Permissions**: Uses alternative APIs that don't require Level 2 access
3. **No Data Available**: Generates synthetic order book from price data

## Testing

Run the test script to verify functionality:
```bash
# Set your Tushare token
export TUSHARE_TOKEN="your_token_here"

# Run the test
python factors_strategy/test_tushare_order_book.py
```

## Common Issues and Solutions

### 1. "请指定正确的接口名" Error
**Cause**: The `stock_quote` API method may not be available or requires special permissions
**Solution**: Now using `realtime_quote` as the primary method which is more widely available

### 2. Missing Level 2 Data
**Cause**: Insufficient Tushare permissions
**Solution**: The implementation automatically falls back to daily data or realtime quotes

### 3. Empty Order Book
**Cause**: No trading data for the specified date (weekend/holiday)
**Solution**: Check if the date is a trading day

## Performance Considerations

1. **API Rate Limits**: 
   - Be aware of Tushare rate limits
   - The parallel processing respects these limits

2. **Data Volume**:
   - Each order book record contains 10 bid/ask levels
   - Storage requirements: ~100 fields per record

3. **Caching**:
   - Consider caching order book data as it doesn't change for historical dates
   - Real-time data should not be cached

## Future Improvements

1. **WebSocket Support**: For real-time order book updates
2. **Compression**: Store order book data in compressed format
3. **Delta Updates**: Only store changes in order book levels
4. **Market Depth Analysis**: Add order book imbalance calculations