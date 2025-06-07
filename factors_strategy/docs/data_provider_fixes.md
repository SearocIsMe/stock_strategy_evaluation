# Data Provider Fixes Documentation

## Overview

This document describes the fixes implemented to address critical bugs in the factors_strategy data collection system:

1. **Enabled Flag Handling**: The `enabled` flag in `data_providers.yaml` was not being respected
2. **Symbol Retrieval**: Symbols were being retrieved from the database instead of API calls
3. **Async Event Loop Error**: Fixed "asyncio.run() cannot be called from a running event loop" error

## Bug Fixes

### 1. Enabled Flag Handling

**Problem**: The `enabled` field in `data_providers.yaml` was being ignored, causing all providers to be initialized regardless of their enabled status.

**Solution**: Modified `DataCollector` class to:
- Load configuration from `data_providers.yaml`
- Filter providers based on `enabled` flag
- Sort providers by priority
- Only initialize enabled providers

**Implementation**:

```python
# In data_collector.py
def _initialize_providers(self):
    """Initialize data providers based on configuration"""
    providers_config = self.config.get('providers', {})
    
    # Sort providers by priority (lower number = higher priority)
    sorted_providers = sorted(
        [(name, cfg) for name, cfg in providers_config.items() if cfg.get('enabled', False)],
        key=lambda x: x[1].get('priority', 999)
    )
    
    for provider_name, provider_config in sorted_providers:
        # Initialize only enabled providers
        ...
```

### 2. Symbol Retrieval from API

**Problem**: Symbols were being retrieved from the `factors` table in the database using a simple SQL query, instead of fetching from data provider APIs.

**Solution**: 
- Added `get_available_symbols()` method to all data providers
- Modified `DataCollector` to aggregate symbols from enabled providers
- Updated `DataIngestionPipeline` to use API-based symbol retrieval with database fallback

**Implementation**:

```python
# In each provider class
async def get_available_symbols(self) -> List[str]:
    """Get list of available symbols from provider"""
    # Provider-specific implementation
    ...

# In DataCollector
async def get_available_symbols(self) -> List[str]:
    """Get available symbols from enabled providers"""
    all_symbols = set()
    
    for provider_name in self.provider_order:
        provider = self.providers[provider_name]
        symbols = await provider.get_available_symbols()
        if symbols:
            all_symbols.update(symbols)
            # Stop at first real provider with symbols
            if provider_name != 'synthetic':
                break
    
    return sorted(list(all_symbols))

# In run.py
def _get_all_symbols(self) -> List[str]:
    """Get all available symbols from data providers via API"""
    # First try API
    symbols = asyncio.run(self.collector.get_available_symbols())
    
    if symbols:
        return symbols
    else:
        # Fallback to database
        return self.reader.get_symbol_universe(active_only=True)
```

## Configuration

The `data_providers.yaml` file now properly controls which providers are active:

```yaml
providers:
  yahoo_finance:
    enabled: false  # Will not be initialized
    priority: 1
    
  tushare:
    enabled: true   # Will be initialized
    priority: 2
    api_token: your_token_here
    
  synthetic:
    enabled: true   # Will be initialized as fallback
    priority: 99
```

## Testing

Run the test script to verify the fixes:

```bash
cd factors_strategy
python test_data_provider_fixes.py
```

The test script verifies:
1. Only enabled providers are initialized
2. Symbols are retrieved from API calls
3. Provider fallback mechanism works correctly
4. Configuration can be dynamically modified

## Usage Examples

### 1. Enable/Disable Providers

Edit `config/data_providers.yaml`:

```yaml
providers:
  yahoo_finance:
    enabled: true  # Enable Yahoo Finance
  tushare:
    enabled: false # Disable Tushare
```

### 2. Run Data Ingestion with API Symbol Retrieval

```bash
# Fetch all symbols from enabled providers
python data/ingestion/run.py --symbols ALL

# Or specify symbols manually
python data/ingestion/run.py --symbols 000001.SZ 600000.SH
```

### 3. Custom Provider Configuration

```python
# Use custom config file
collector = DataCollector("path/to/custom_providers.yaml")

# Get symbols from API
symbols = await collector.get_available_symbols()
```

## Benefits

1. **Flexibility**: Enable/disable providers without code changes
2. **API-First**: Symbols come from real-time API data, not stale database records
3. **Fallback Support**: Graceful degradation when providers fail
4. **Priority-Based**: Providers are tried in order of priority
5. **Configuration-Driven**: All settings in YAML files

### 3. Async Event Loop Error Fix

**Problem**: When running the data ingestion pipeline, the error "asyncio.run() cannot be called from a running event loop" occurred because `asyncio.run()` was being called inside an already running async context.

**Solution**:
- Changed `_get_all_symbols()` method to be async
- Replaced `asyncio.run()` with `await` when calling async methods
- Updated all callers to use `await` properly

**Implementation**:

```python
# In run.py - Before (incorrect)
def _get_all_symbols(self) -> List[str]:
    symbols = asyncio.run(self.collector.get_available_symbols())  # Error!

# After (correct)
async def _get_all_symbols(self) -> List[str]:
    symbols = await self.collector.get_available_symbols()  # Works!

# Update caller
symbols = await self._get_all_symbols()  # Use await
```

## Testing

Run the test scripts to verify all fixes:

```bash
# Test provider fixes
cd factors_strategy
python test_data_provider_fixes.py

# Test async fix
python test_async_fix.py
```

## Future Enhancements

1. Add more data providers (Alpha Vantage, IEX Cloud, etc.)
2. Implement provider health checks
3. Add caching for symbol lists
4. Support for provider-specific symbol filters
5. Real-time provider status monitoring