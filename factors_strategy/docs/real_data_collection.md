# Real Data Collection System

This document describes the new real data collection system that replaces the previous demo/synthetic data functions with actual market data providers.

## Overview

The system implements a provider-based architecture with automatic fallback mechanisms to ensure reliable data collection from multiple sources.

## Architecture

### Provider Pattern
```
DataCollector
├── YahooFinanceProvider (Primary)
├── TushareProvider (Chinese markets)
└── SyntheticDataProvider (Fallback)
```

### Key Components

1. **DataProvider (Abstract Base Class)**
   - Defines the interface for all data providers
   - Methods: `get_tick_data()`, `get_order_book_data()`, `get_real_time_data()`

2. **YahooFinanceProvider**
   - Primary provider for global market data
   - Supports: Tick data (1-min), Real-time data, Fundamental data
   - Converts Chinese symbols (.SH → .SS, .SZ → .SZ)

3. **TushareProvider**
   - Specialized for Chinese market data
   - Requires API token (currently placeholder implementation)
   - Supports: All data types including order book

4. **SyntheticDataProvider**
   - Fallback provider using the original demo functions
   - Generates realistic synthetic data for testing

5. **DataCollector**
   - Orchestrates multiple providers with fallback logic
   - Implements data quality validation
   - Provides unified interface for data collection

## Usage Examples

### Basic Usage
```python
from data.ingestion.data_collector import DataCollector
from datetime import datetime, timedelta

async def collect_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    async with DataCollector() as collector:
        # Collect tick data
        tick_data = await collector.collect_tick_data("AAPL", start_date, end_date)
        
        # Collect real-time data
        real_time = await collector.collect_real_time_data(["AAPL", "TSLA"])
        
        # Collect fundamental data
        fundamentals = await collector.collect_fundamental_data("AAPL")
```

### Custom Provider Configuration
```python
from data.ingestion.data_collector import (
    DataCollector, YahooFinanceProvider, SyntheticDataProvider
)

# Use only specific providers
providers = [YahooFinanceProvider(), SyntheticDataProvider()]
async with DataCollector(providers=providers) as collector:
    data = await collector.collect_tick_data("000001.SZ", start_date, end_date)
```

### With Tushare Token
```python
# Configure Tushare provider with API token
async with DataCollector(tushare_token="your_token_here") as collector:
    data = await collector.collect_tick_data("000001.SZ", start_date, end_date)
```

## Supported Markets

### Chinese Markets
- **Shanghai Stock Exchange (.SH)**: Converted to .SS for Yahoo Finance
- **Shenzhen Stock Exchange (.SZ)**: Native support
- **Market Hours**: 09:30 - 15:00 Asia/Shanghai timezone

### Global Markets
- **US Markets**: NASDAQ, NYSE (AAPL, TSLA, etc.)
- **Other Markets**: Any market supported by Yahoo Finance

## Data Types

### Tick Data
- **Frequency**: 1-minute intervals
- **Fields**: timestamp, symbol, price, volume, turnover, bid/ask arrays
- **Source**: Yahoo Finance historical data

### Order Book Data
- **Levels**: 10 bid/ask levels
- **Fields**: bid_price_1-10, ask_price_1-10, bid_volume_1-10, ask_volume_1-10
- **Source**: Tushare (when available), Synthetic (fallback)

### Real-time Data
- **Fields**: current price, volume, bid/ask, change percentage
- **Source**: Yahoo Finance info API

### Fundamental Data
- **Fields**: market_cap, pe_ratio, pb_ratio, roe, debt_to_equity, etc.
- **Source**: Yahoo Finance info API

## Fallback Mechanism

The system tries providers in priority order:

1. **YahooFinanceProvider** (Priority 1)
   - Attempts to fetch real market data
   - Handles symbol conversion for Chinese stocks

2. **TushareProvider** (Priority 2)
   - Specialized Chinese market data
   - Requires API token configuration

3. **SyntheticDataProvider** (Priority 99)
   - Always succeeds with realistic synthetic data
   - Maintains original demo functionality

If a provider fails or returns empty data, the system automatically tries the next provider.

## Data Quality Validation

The system includes comprehensive data quality checks:

### Automatic Validation
- **Required Columns**: Ensures all necessary fields are present
- **Null Value Checks**: Validates critical fields are not null
- **Timestamp Ordering**: Verifies chronological order
- **Price Range Validation**: Checks for realistic price values

### Manual Validation
```python
# Validate data quality manually
is_valid = collector.validate_data_quality(tick_data, 'tick_data')
```

## Configuration

### Provider Configuration (`config/data_providers.yaml`)
```yaml
providers:
  yahoo_finance:
    enabled: true
    priority: 1
    rate_limits:
      requests_per_minute: 2000
  
  tushare:
    enabled: false
    api_token: null  # Set your token here
    priority: 2
```

### Symbol Mapping
```yaml
symbol_mapping:
  shanghai:
    suffix: ".SH"
    yahoo_suffix: ".SS"
    timezone: "Asia/Shanghai"
```

## Error Handling

### Provider Failures
- Individual provider failures don't stop data collection
- System logs errors and tries next provider
- Only fails if all providers fail

### Network Issues
- Configurable timeouts and retries
- Exponential backoff for rate limiting
- Graceful degradation to synthetic data

## Performance Considerations

### Rate Limiting
- Yahoo Finance: 2000 requests/minute
- Tushare: 200 requests/minute (with token)
- Built-in rate limiting and retry logic

### Caching (Future Enhancement)
- Configurable TTL for different data types
- Reduces API calls and improves performance
- Currently not implemented but designed for easy addition

## Migration from Demo Functions

### Before (Demo Functions)
```python
# Old demo functions
tick_data = collector._generate_sample_tick_data(symbol, start_date, end_date)
order_book = collector._generate_sample_order_book_data(symbol, start_date, end_date)
```

### After (Real Data Collection)
```python
# New real data collection with fallback
tick_data = await collector.collect_tick_data(symbol, start_date, end_date)
order_book = await collector.collect_order_book_data(symbol, start_date, end_date)
```

### Key Changes
1. **Async/Await**: All data collection is now asynchronous
2. **Provider System**: Multiple data sources with automatic fallback
3. **Real Data**: Actual market data from Yahoo Finance and other providers
4. **Data Validation**: Built-in quality checks and validation
5. **Configuration**: YAML-based provider configuration
6. **Error Handling**: Robust error handling and retry mechanisms

## Testing

Run the demo script to test the new system:
```bash
python examples/real_data_collection_demo.py
```

This will demonstrate:
- Real data collection from Yahoo Finance
- Provider fallback mechanism
- Data quality validation
- Configuration loading

## Future Enhancements

1. **Additional Providers**
   - Alpha Vantage
   - IEX Cloud
   - Polygon.io
   - Quandl

2. **Caching Layer**
   - Redis-based caching
   - Configurable TTL
   - Cache invalidation strategies

3. **Real-time Streaming**
   - WebSocket connections
   - Live market data feeds
   - Event-driven updates

4. **Enhanced Tushare Integration**
   - Complete API implementation
   - Chinese market specialization
   - Order book data support

5. **Monitoring and Alerting**
   - Data quality monitoring
   - Provider health checks
   - Performance metrics