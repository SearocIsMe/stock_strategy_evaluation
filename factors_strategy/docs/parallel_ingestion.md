# Parallel Data Ingestion

This document describes the parallel data ingestion feature that significantly improves the performance of tick data collection.

## Overview

The data ingestion pipeline now supports parallel processing of multiple symbols simultaneously, which can dramatically reduce the total time required to collect data for large numbers of stocks.

## Usage

### Environment Setup

If using Tushare data provider, set the API token as an environment variable:

```bash
# Linux/Mac
export TUSHARE_TOKEN="your_tushare_api_token_here"

# Windows
set TUSHARE_TOKEN=your_tushare_api_token_here
```

### Command Line

Use the `--workers` parameter to specify the number of parallel workers:

```bash
# Single worker (default, sequential processing)
python -m data.ingestion.run --symbols ALL --start-date 2025-06-05 --end-date 2025-06-07 --providers-config ./config/data_providers.yaml

# 5 parallel workers
python -m data.ingestion.run --symbols ALL --start-date 2025-06-05 --end-date 2025-06-07 --providers-config ./config/data_providers.yaml --workers 5

# 10 parallel workers for maximum speed
python -m data.ingestion.run --symbols ALL --start-date 2025-06-05 --end-date 2025-06-07 --providers-config ./config/data_providers.yaml --workers 10
```

### Specific Symbols with Parallel Processing

```bash
# Process specific symbols with 3 workers
python -m data.ingestion.run --symbols 000001.SZ 000002.SZ 600000.SH 600519.SH --start-date 2025-06-05 --end-date 2025-06-07 --workers 3
```

## Performance Considerations

### Choosing the Right Number of Workers

1. **Network Bandwidth**: More workers mean more concurrent API requests. Ensure your network can handle the load.

2. **API Rate Limits**: Check the rate limits in `config/data_providers.yaml`:
   - Tushare: 200 requests/minute
   - Yahoo Finance: 2000 requests/minute

3. **System Resources**: Each worker consumes memory and CPU. Monitor system resources when using many workers.

4. **Database Performance**: ClickHouse can handle high write throughput, but ensure your database server has sufficient resources.

### Recommended Worker Counts

- **Small batches (< 10 symbols)**: 1-3 workers
- **Medium batches (10-50 symbols)**: 3-5 workers
- **Large batches (50-200 symbols)**: 5-10 workers
- **All symbols (200+ symbols)**: 10-20 workers

## Implementation Details

### Concurrency Control

The implementation uses Python's `asyncio` with a semaphore to control the maximum number of concurrent operations:

```python
self.semaphore = Semaphore(max_workers)

async def _process_symbol_with_semaphore(self, symbol, ...):
    async with self.semaphore:
        await self._process_symbol(symbol, ...)
```

### Error Handling

- Each symbol is processed independently
- Errors in one symbol don't affect others
- Failed symbols are logged and reported at the end
- The pipeline continues even if some symbols fail

### Progress Monitoring

The logs show:
- Total number of symbols to process
- Number of parallel workers being used
- Progress for each symbol as it's processed
- Summary of successful/failed symbols at the end

## Testing Performance

Use the provided test script to benchmark different worker counts:

```bash
python factors_strategy/test_parallel_ingestion.py
```

This will test with 1, 3, and 5 workers and show the performance improvement.

## Example Output

```
Starting data ingestion for 150 symbols
Using 10 parallel workers
Symbols: ['000001.SZ', '000002.SZ', '000858.SZ', '000725.SZ', '000776.SZ', '600000.SH', '600036.SH', '600519.SH', '600887.SH', '600276.SH']...
Date range: 2025-06-05 00:00:00 to 2025-06-07 00:00:00
Data types: ['tick_data', 'order_book', 'factors']

Processing symbol: 000001.SZ
Processing symbol: 000002.SZ
Processing symbol: 000858.SZ
...
[Multiple symbols processed in parallel]
...

Data ingestion completed. Successful: 148/150
```

## Troubleshooting

### High Memory Usage

If you experience high memory usage with many workers:
1. Reduce the number of workers
2. Process symbols in smaller batches
3. Monitor memory usage with `htop` or similar tools

### API Rate Limit Errors

If you see rate limit errors:
1. Reduce the number of workers
2. Add delays between requests (modify the data collector)
3. Use multiple API keys if supported by the provider

### Database Connection Errors

If you see database connection errors:
1. Check ClickHouse server status
2. Ensure the database can handle concurrent connections
3. Check connection pool settings in the database configuration

## Future Improvements

1. **Dynamic Worker Adjustment**: Automatically adjust workers based on system load
2. **Priority Queue**: Process high-priority symbols first
3. **Resume Capability**: Resume from where it left off if interrupted
4. **Real-time Progress Bar**: Visual progress indicator
5. **Distributed Processing**: Support for multiple machines