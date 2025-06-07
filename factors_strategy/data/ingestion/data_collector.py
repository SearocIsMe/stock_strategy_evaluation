"""
Data Collector Module
Handles collection of market data from various sources
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
import pytz
import yfinance as yf
import requests
from abc import ABC, abstractmethod
import yaml
from pathlib import Path

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Thread-%(thread)d] [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def get_tick_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get tick data from provider"""
        pass
    
    @abstractmethod
    async def get_order_book_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get order book data from provider"""
        pass
    
    @abstractmethod
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data from provider"""
        pass
    
    @abstractmethod
    async def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from provider"""
        pass


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider"""
    
    def __init__(self):
        self.session = None
    
    async def get_tick_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get tick data from Yahoo Finance"""
        try:
            # Convert Chinese symbols to Yahoo Finance format
            yahoo_symbol = self._convert_symbol_to_yahoo(symbol)
            
            # Yahoo Finance doesn't provide tick data, so we'll use 1-minute data
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(start=start_date.date(), end=end_date.date(), interval="1m")
            
            if data.empty:
                logger.warning(f"No data found for {symbol} from Yahoo Finance")
                return pd.DataFrame()
            
            # Convert to our tick data format
            tick_data = self._convert_yahoo_to_tick_format(data, symbol)
            return tick_data
            
        except Exception as e:
            logger.error(f"Failed to get tick data from Yahoo Finance for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_order_book_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Yahoo Finance doesn't provide order book data, return empty DataFrame"""
        logger.warning(f"Yahoo Finance doesn't provide order book data for {symbol}")
        return pd.DataFrame()
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data from Yahoo Finance"""
        try:
            real_time_data = {}
            
            for symbol in symbols:
                yahoo_symbol = self._convert_symbol_to_yahoo(symbol)
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info
                
                # Get current price and other metrics
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                previous_close = info.get('previousClose', current_price)
                change_pct = ((current_price - previous_close) / previous_close) if previous_close else 0
                
                real_time_data[symbol] = {
                    'timestamp': datetime.now(pytz.UTC),
                    'price': current_price,
                    'volume': info.get('volume', 0),
                    'bid_price': info.get('bid', current_price * 0.999),
                    'ask_price': info.get('ask', current_price * 1.001),
                    'change_pct': change_pct,
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0)
                }
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"Failed to get real-time data from Yahoo Finance: {e}")
            return {}
    
    async def get_available_symbols(self) -> List[str]:
        """Get available symbols from Yahoo Finance"""
        try:
            # For Chinese markets, we can use predefined lists or scrape from Yahoo Finance
            # This is a simplified version - in production, you might want to fetch from Yahoo's screener
            chinese_symbols = []
            
            # Shanghai Stock Exchange (SSE) - 600xxx, 601xxx, 603xxx, 605xxx, 688xxx
            # Shenzhen Stock Exchange (SZSE) - 000xxx, 001xxx, 002xxx, 003xxx, 300xxx
            
            # For demo purposes, returning a subset of major Chinese stocks
            major_chinese_stocks = [
                '000001.SZ', '000002.SZ', '000858.SZ', '000725.SZ', '000776.SZ',
                '600000.SH', '600036.SH', '600519.SH', '600887.SH', '600276.SH',
                '002415.SZ', '300059.SZ', '300750.SZ', '002594.SZ'
            ]
            
            return major_chinese_stocks
            
        except Exception as e:
            logger.error(f"Failed to get available symbols from Yahoo Finance: {e}")
            return []
    
    def _convert_symbol_to_yahoo(self, symbol: str) -> str:
        """Convert Chinese stock symbols to Yahoo Finance format"""
        if symbol.endswith('.SH'):
            # Shanghai Stock Exchange
            return symbol.replace('.SH', '.SS')
        elif symbol.endswith('.SZ'):
            # Shenzhen Stock Exchange
            return symbol.replace('.SZ', '.SZ')
        return symbol
    
    def _convert_yahoo_to_tick_format(self, yahoo_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Convert Yahoo Finance data to our tick data format"""
        if yahoo_data.empty:
            return pd.DataFrame()
        
        # Reset index to get timestamp as column
        yahoo_data = yahoo_data.reset_index()
        
        # Convert to UTC if not already
        if yahoo_data['Datetime'].dt.tz is None:
            yahoo_data['Datetime'] = yahoo_data['Datetime'].dt.tz_localize('UTC')
        else:
            yahoo_data['Datetime'] = yahoo_data['Datetime'].dt.tz_convert('UTC')
        
        # Create tick data format
        tick_data = pd.DataFrame({
            'timestamp': yahoo_data['Datetime'],
            'symbol': symbol,
            'price': yahoo_data['Close'],
            'volume': yahoo_data['Volume'],
            'turnover': yahoo_data['Close'] * yahoo_data['Volume'],
            'bid_price': [np.array([price * 0.999]) for price in yahoo_data['Close']],
            'bid_volume': [np.array([vol * 0.3]) for vol in yahoo_data['Volume']],
            'ask_price': [np.array([price * 1.001]) for price in yahoo_data['Close']],
            'ask_volume': [np.array([vol * 0.3]) for vol in yahoo_data['Volume']],
            'trade_direction': np.random.choice([1, -1], len(yahoo_data)),
            'trade_type': 'NORMAL',
            'exchange': 'YAHOO',
            'update_time': yahoo_data['Datetime']
        })
        
        return tick_data


class TushareProvider(DataProvider):
    """Tushare data provider for Chinese stocks"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.session = None
        self.ts = None
        
        if self.token:
            try:
                import tushare as ts
                ts.set_token(self.token)
                self.ts = ts.pro_api()
                logger.info(f"Tushare API initialized successfully: {token}")
            except Exception as e:
                logger.error(f"Failed to initialize Tushare API: {e}")
        
    async def get_tick_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get tick data from Tushare"""
        try:
            if not self.token or not self.ts:
                logger.warning("Tushare token not provided or API not initialized")
                return pd.DataFrame()

            # Convert symbol format for Tushare (e.g., 000001.SZ -> 000001.SZ)
            ts_symbol = symbol
            
            # Get minute-level data from Tushare
            df = self.ts.stk_mins(
                ts_code=ts_symbol,
                freq='1min',
                start_date=start_date.strftime('%Y%m%d %H:%M:%S'),
                end_date=end_date.strftime('%Y%m%d %H:%M:%S')
            )
            
            if df.empty:
                logger.warning(f"No tick data found for {symbol} from Tushare")
                return pd.DataFrame()
            
            # Convert to our tick data format
            tick_data = self._convert_tushare_to_tick_format(df, symbol)
            return tick_data
            
        except Exception as e:
            logger.error(f"Failed to get tick data from Tushare for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_order_book_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get order book data from Tushare"""
        try:
            if not self.token or not self.ts:
                logger.warning("Tushare token not provided or API not initialized")
                return pd.DataFrame()
            
            # Tushare provides level 2 data which includes order book
            # This would require additional Tushare permissions
            logger.info(f"Tushare order book data collection for {symbol} would be implemented here")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get order book data from Tushare for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data from Tushare"""
        try:
            if not self.token or not self.ts:
                logger.warning("Tushare token not provided or API not initialized")
                return {}
            
            real_time_data = {}
            
            # Get real-time quotes from Tushare
            for symbol in symbols:
                try:
                    df = self.ts.realtime_quote(ts_code=symbol)
                    if not df.empty:
                        row = df.iloc[0]
                        real_time_data[symbol] = {
                            'timestamp': datetime.now(pytz.UTC),
                            'price': float(row['price']),
                            'volume': float(row['volume']),
                            'bid_price': float(row['bid']),
                            'ask_price': float(row['ask']),
                            'change_pct': float(row['pct_change']) / 100,
                            'market_cap': float(row.get('total_mv', 0)) * 10000,  # Convert to yuan
                            'pe_ratio': float(row.get('pe', 0))
                        }
                except Exception as e:
                    logger.error(f"Failed to get real-time data for {symbol}: {e}")
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"Failed to get real-time data from Tushare: {e}")
            return {}
    
    async def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from Tushare"""
        try:
            if not self.token or not self.ts:
                logger.warning("Tushare token not provided or API not initialized")
                return []
            
            # Get stock list from Tushare
            df = self.ts.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
            
            if df.empty:
                logger.warning("No symbols found from Tushare")
                return []
            
            # Return list of ts_codes (e.g., 000001.SZ, 600000.SH)
            symbols = df['ts_code'].tolist()
            logger.info(f"Found {len(symbols)} symbols from Tushare")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get available symbols from Tushare: {e}")
            return []
    
    def _convert_tushare_to_tick_format(self, tushare_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Convert Tushare data to our tick data format"""
        if tushare_data.empty:
            return pd.DataFrame()
        
        # Convert Tushare timestamp to datetime
        tushare_data['timestamp'] = pd.to_datetime(tushare_data['trade_time'])
        
        # Ensure UTC timezone
        if tushare_data['timestamp'].dt.tz is None:
            tushare_data['timestamp'] = tushare_data['timestamp'].dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC')
        
        # Create tick data format
        tick_data = pd.DataFrame({
            'timestamp': tushare_data['timestamp'],
            'symbol': symbol,
            'price': tushare_data['close'],
            'volume': tushare_data['vol'],
            'turnover': tushare_data['amount'],
            'bid_price': [np.array([price]) for price in tushare_data['close'] * 0.999],
            'bid_volume': [np.array([vol * 0.3]) for vol in tushare_data['vol']],
            'ask_price': [np.array([price]) for price in tushare_data['close'] * 1.001],
            'ask_volume': [np.array([vol * 0.3]) for vol in tushare_data['vol']],
            'trade_direction': np.where(tushare_data['close'] > tushare_data['open'], 1, -1),
            'trade_type': 'NORMAL',
            'exchange': 'TUSHARE',
            'update_time': tushare_data['timestamp']
        })
        
        return tick_data


class SyntheticDataProvider(DataProvider):
    """Fallback synthetic data provider (original demo functions)"""
    
    async def get_tick_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic tick data"""
        return self._generate_sample_tick_data(symbol, start_date, end_date)
    
    async def get_order_book_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic order book data"""
        return self._generate_sample_order_book_data(symbol, start_date, end_date)
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate synthetic real-time data"""
        real_time_data = {}
        for symbol in symbols:
            real_time_data[symbol] = {
                'timestamp': datetime.now(pytz.UTC),
                'price': 100.0 + np.random.normal(0, 1),
                'volume': np.random.lognormal(8, 1),
                'bid_price': 99.95 + np.random.normal(0, 0.1),
                'ask_price': 100.05 + np.random.normal(0, 0.1),
                'change_pct': np.random.normal(0, 0.02)
            }
        return real_time_data
    
    async def get_available_symbols(self) -> List[str]:
        """Get synthetic list of available symbols"""
        # Return comprehensive list of Chinese stock symbols for testing
        return [
            # Major indices components and blue chips
            '000001.SZ', '000002.SZ', '000858.SZ', '000725.SZ', '000776.SZ',
            '600000.SH', '600036.SH', '600519.SH', '600887.SH', '600276.SH',
            '000858.SZ', '002415.SZ', '300059.SZ', '300750.SZ', '002594.SZ',
            
            # Technology stocks
            '000063.SZ', '002230.SZ', '002241.SZ', '300014.SZ', '300033.SZ',
            '300122.SZ', '300124.SZ', '300136.SZ', '300408.SZ', '300433.SZ',
            
            # Financial sector
            '000001.SZ', '600000.SH', '600015.SH', '600016.SH', '600036.SH',
            '600837.SH', '600886.SH', '601009.SH', '601166.SH', '601169.SH',
            
            # Consumer goods
            '000568.SZ', '000596.SZ', '000858.SZ', '000895.SZ', '002304.SZ',
            '600519.SH', '600887.SH', '600999.SH', '603288.SH', '603369.SH',
            
            # Healthcare & Pharma
            '000661.SZ', '000963.SZ', '002007.SZ', '002022.SZ', '002252.SZ',
            '002422.SZ', '002821.SZ', '300003.SZ', '300015.SZ', '300142.SZ'
        ]
    
    def _generate_sample_tick_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate sample tick data for testing"""
        
        # Generate timestamps during market hours (9:30 AM - 3:00 PM China time)
        china_tz = pytz.timezone('Asia/Shanghai')
        
        # Ensure start_date and end_date are timezone-aware and set to market hours
        if start_date.tzinfo is None:
            # Set to market open (9:30 AM China time)
            start_date = china_tz.localize(start_date.replace(hour=9, minute=30, second=0, microsecond=0))
        if end_date.tzinfo is None:
            # Set to market close (3:00 PM China time)
            end_date = china_tz.localize(end_date.replace(hour=15, minute=0, second=0, microsecond=0))
            
        timestamps = pd.date_range(start=start_date, end=end_date, freq='1min', tz=china_tz)
        # Convert to UTC for storage
        timestamps_utc = timestamps.tz_convert('UTC')
        n_records = len(timestamps_utc)
        
        if n_records == 0:
            return pd.DataFrame()
            
        # Generate realistic Chinese stock prices (typically 1-1000 RMB range)
        # Use symbol to determine base price range
        if symbol.endswith('.SZ') or symbol.endswith('.SH'):
            # Chinese stocks: typical range 5-500 RMB
            base_price = np.random.uniform(8.0, 150.0)  # Realistic Chinese stock price range
        else:
            base_price = np.random.uniform(10.0, 200.0)
            
        # Generate realistic price movements (smaller volatility, round to 2 decimal places)
        price_changes = np.random.normal(0, 0.0005, n_records)  # Reduced volatility
        prices = base_price * np.exp(np.cumsum(price_changes))
        # Round to 2 decimal places like real stock prices
        prices = np.round(prices, 2)
        
        # Generate realistic volumes (round to whole numbers)
        volumes = np.round(np.random.lognormal(mean=8, sigma=1, size=n_records), 0)
        
        # Generate realistic bid/ask spreads (smaller for Chinese stocks)
        spreads = np.random.uniform(0.01, 0.03, n_records)  # Smaller spreads
        bid_prices = np.round(prices - spreads / 2, 2)  # Round to 2 decimal places
        ask_prices = np.round(prices + spreads / 2, 2)   # Round to 2 decimal places
        
        # Create DataFrame with UTC timestamps
        tick_data = pd.DataFrame({
            'timestamp': timestamps_utc,
            'symbol': symbol,
            'price': prices,
            'volume': volumes,
            'turnover': prices * volumes,
            'bid_price': [np.array([bp]) for bp in bid_prices],
            'bid_volume': [np.array([vol * 0.3]) for vol in volumes],
            'ask_price': [np.array([ap]) for ap in ask_prices],
            'ask_volume': [np.array([vol * 0.3]) for vol in volumes],
            'trade_direction': np.random.choice([1, -1], n_records),
            'trade_type': 'NORMAL',
            'exchange': 'SZSE',
            'update_time': timestamps_utc  # Store in UTC
        })
        
        return tick_data
        
    def _generate_sample_order_book_data(self, symbol: str, start_date: datetime,
                                       end_date: datetime) -> pd.DataFrame:
        """Generate sample order book data for testing"""
        
        # Generate timestamps every 5 minutes during market hours (9:30 AM - 3:00 PM China time)
        china_tz = pytz.timezone('Asia/Shanghai')
        
        # Ensure start_date and end_date are timezone-aware and set to market hours
        if start_date.tzinfo is None:
            # Set to market open (9:30 AM China time)
            start_date = china_tz.localize(start_date.replace(hour=9, minute=30, second=0, microsecond=0))
        if end_date.tzinfo is None:
            # Set to market close (3:00 PM China time)
            end_date = china_tz.localize(end_date.replace(hour=15, minute=0, second=0, microsecond=0))
            
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min', tz=china_tz)
        # Convert to UTC for storage
        timestamps_utc = timestamps.tz_convert('UTC')
        n_records = len(timestamps_utc)
        
        if n_records == 0:
            return pd.DataFrame()
            
        # Generate realistic Chinese stock prices
        if symbol.endswith('.SZ') or symbol.endswith('.SH'):
            # Chinese stocks: typical range 5-500 RMB
            base_price = np.random.uniform(8.0, 150.0)
        else:
            base_price = np.random.uniform(10.0, 200.0)
            
        # Generate realistic price movements with proper rounding
        price_changes = np.random.normal(0, 0.0005, n_records)
        mid_prices = base_price * np.exp(np.cumsum(price_changes))
        mid_prices = np.round(mid_prices, 2)  # Round to 2 decimal places
        
        # Generate order book levels with UTC timestamps
        order_book_data = pd.DataFrame({
            'timestamp': timestamps_utc,
            'symbol': symbol,
            'exchange': 'SZSE'
        })
        
        # Generate 10 levels of bid/ask to match schema
        for level in range(1, 11):
            # Bid side - round to 2 decimal places
            bid_offset = level * 0.01
            order_book_data[f'bid_price_{level}'] = np.round(mid_prices - bid_offset, 2)
            order_book_data[f'bid_volume_{level}'] = np.round(np.random.lognormal(6, 1, n_records), 0)  # Round volumes to whole numbers
            
            # Ask side - round to 2 decimal places
            ask_offset = level * 0.01
            order_book_data[f'ask_price_{level}'] = np.round(mid_prices + ask_offset, 2)
            order_book_data[f'ask_volume_{level}'] = np.round(np.random.lognormal(6, 1, n_records), 0)  # Round volumes to whole numbers
            
        # Calculate aggregated metrics
        bid_volumes = [order_book_data[f'bid_volume_{i}'] for i in range(1, 11)]
        ask_volumes = [order_book_data[f'ask_volume_{i}'] for i in range(1, 11)]
        
        order_book_data['total_bid_volume'] = sum(bid_volumes)
        order_book_data['total_ask_volume'] = sum(ask_volumes)
        order_book_data['bid_ask_imbalance'] = (
            order_book_data['total_bid_volume'] - order_book_data['total_ask_volume']
        ) / (order_book_data['total_bid_volume'] + order_book_data['total_ask_volume'])
        
        # Add update_time column in UTC
        order_book_data['update_time'] = timestamps_utc
        
        return order_book_data

class DataCollector:
    """Collects market data from various sources with fallback mechanisms"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize data collector with configuration"""
        self.session = None
        self.providers = {}
        self.provider_order = []
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "data_providers.yaml"
        
        self._load_config(config_path)
        self._initialize_providers()
        
    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = {'providers': {}}
    
    def _initialize_providers(self):
        """Initialize data providers based on configuration"""
        providers_config = self.config.get('providers', {})
        
        # Sort providers by priority (lower number = higher priority)
        sorted_providers = sorted(
            [(name, cfg) for name, cfg in providers_config.items() if cfg.get('enabled', False)],
            key=lambda x: x[1].get('priority', 999)
        )
        
        for provider_name, provider_config in sorted_providers:
            try:
                if provider_name == 'yahoo_finance':
                    provider = YahooFinanceProvider()
                elif provider_name == 'tushare':
                    token = provider_config.get('api_token')
                    provider = TushareProvider(token)
                elif provider_name == 'synthetic':
                    provider = SyntheticDataProvider()
                else:
                    logger.warning(f"Unknown provider type: {provider_name}")
                    continue
                
                self.providers[provider_name] = provider
                self.provider_order.append(provider_name)
                logger.info(f"Initialized provider: {provider_name} (priority: {provider_config.get('priority', 999)})")
                
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")
        
        # If no providers are enabled, use synthetic as fallback
        if not self.providers:
            logger.warning("No providers enabled, using synthetic data provider as fallback")
            self.providers['synthetic'] = SyntheticDataProvider()
            self.provider_order.append('synthetic')
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_available_symbols(self) -> List[str]:
        """Get available symbols from enabled providers"""
        all_symbols = set()
        
        for provider_name in self.provider_order:
            provider = self.providers[provider_name]
            try:
                logger.info(f"Getting available symbols from {provider_name}")
                symbols = await provider.get_available_symbols()
                
                if symbols:
                    all_symbols.update(symbols)
                    logger.info(f"Got {len(symbols)} symbols from {provider_name}")
                    # If we get symbols from a real provider, we can stop
                    if provider_name != 'synthetic':
                        break
                        
            except Exception as e:
                logger.error(f"Failed to get symbols from {provider_name}: {e}")
                continue
        
        return sorted(list(all_symbols))
            
    async def collect_tick_data(self, symbol: str, start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
        """Collect tick data for a symbol using multiple providers with fallback"""
        logger.info(f"Collecting tick data for {symbol} from {start_date} to {end_date}")
        
        for provider_name in self.provider_order:
            provider = self.providers[provider_name]
            try:
                logger.info(f"Trying provider: {provider_name}")
                
                tick_data = await provider.get_tick_data(symbol, start_date, end_date)
                
                if not tick_data.empty:
                    # Validate data quality
                    if self.validate_data_quality(tick_data, 'tick_data'):
                        logger.info(f"Successfully collected {len(tick_data)} tick records for {symbol} from {provider_name}")
                        return tick_data
                    else:
                        logger.warning(f"Data quality validation failed for {provider_name}, trying next provider")
                        continue
                else:
                    logger.warning(f"No data returned from {provider_name}, trying next provider")
                    continue
                    
            except Exception as e:
                logger.error(f"Provider {provider_name} failed for tick data: {e}")
                continue
        
        # If all providers fail, raise an exception
        raise Exception(f"All data providers failed to collect tick data for {symbol}")
            
    async def collect_order_book_data(self, symbol: str, start_date: datetime,
                                    end_date: datetime) -> pd.DataFrame:
        """Collect order book data for a symbol using multiple providers with fallback"""
        logger.info(f"Collecting order book data for {symbol} from {start_date} to {end_date}")
        
        for provider_name in self.provider_order:
            provider = self.providers[provider_name]
            try:
                logger.info(f"Trying provider: {provider_name}")
                
                order_book_data = await provider.get_order_book_data(symbol, start_date, end_date)
                
                if not order_book_data.empty:
                    # Validate data quality
                    if self.validate_data_quality(order_book_data, 'order_book'):
                        logger.info(f"Successfully collected {len(order_book_data)} order book records for {symbol} from {provider_name}")
                        return order_book_data
                    else:
                        logger.warning(f"Data quality validation failed for {provider_name}, trying next provider")
                        continue
                else:
                    logger.warning(f"No data returned from {provider_name}, trying next provider")
                    continue
                    
            except Exception as e:
                logger.error(f"Provider {provider_name} failed for order book data: {e}")
                continue
        
        # If all providers fail, raise an exception
        raise Exception(f"All data providers failed to collect order book data for {symbol}")
            
    async def collect_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect real-time market data using multiple providers with fallback"""
        logger.info(f"Collecting real-time data for {len(symbols)} symbols")
        
        for provider_name in self.provider_order:
            provider = self.providers[provider_name]
            try:
                logger.info(f"Trying provider: {provider_name}")
                
                real_time_data = await provider.get_real_time_data(symbols)
                
                if real_time_data:
                    logger.info(f"Successfully collected real-time data for {len(real_time_data)} symbols from {provider_name}")
                    return real_time_data
                else:
                    logger.warning(f"No real-time data returned from {provider_name}, trying next provider")
                    continue
                    
            except Exception as e:
                logger.error(f"Provider {provider_name} failed for real-time data: {e}")
                continue
        
        # If all providers fail, raise an exception
        raise Exception(f"All data providers failed to collect real-time data for symbols: {symbols}")
            
    async def collect_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Collect fundamental data for a symbol using Yahoo Finance"""
        try:
            logger.info(f"Collecting fundamental data for {symbol}")
            
            # Use Yahoo Finance for fundamental data
            yahoo_symbol = self._convert_symbol_to_yahoo(symbol)
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            fundamental_data = {
                'symbol': symbol,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'roe': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'beta': info.get('beta', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'book_value': info.get('bookValue', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'gross_margin': info.get('grossMargins', 0),
                'ebitda_margin': info.get('ebitdaMargins', 0),
                'revenue_per_share': info.get('revenuePerShare', 0),
                'earnings_per_share': info.get('trailingEps', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'operating_cash_flow': info.get('operatingCashflow', 0),
                'total_cash': info.get('totalCash', 0),
                'total_debt': info.get('totalDebt', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'avg_volume_10d': info.get('averageVolume10days', 0),
                'avg_volume': info.get('averageVolume', 0),
                'shares_short': info.get('sharesShort', 0),
                'short_ratio': info.get('shortRatio', 0),
                'held_by_insiders': info.get('heldPercentInsiders', 0),
                'held_by_institutions': info.get('heldPercentInstitutions', 0)
            }
            
            logger.info(f"Successfully collected fundamental data for {symbol}")
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Failed to collect fundamental data for {symbol}: {e}")
            return {}
            
    def _convert_symbol_to_yahoo(self, symbol: str) -> str:
        """Convert Chinese stock symbols to Yahoo Finance format"""
        if symbol.endswith('.SH'):
            # Shanghai Stock Exchange
            return symbol.replace('.SH', '.SS')
        elif symbol.endswith('.SZ'):
            # Shenzhen Stock Exchange
            return symbol.replace('.SZ', '.SZ')
        return symbol
            
    def validate_data_quality(self, data: pd.DataFrame, data_type: str) -> bool:
        """Validate data quality"""
        try:
            if data.empty:
                return False
                
            # Check for required columns based on data type
            if data_type == 'tick_data':
                required_columns = ['timestamp', 'symbol', 'price', 'volume']
            elif data_type == 'order_book':
                required_columns = ['timestamp', 'symbol', 'bid_price_1', 'ask_price_1']
            else:
                return True
                
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                return False
                
            # Check for null values in critical columns
            if data[required_columns].isnull().any().any():
                logger.warning("Found null values in critical columns")
                return False
                
            # Check for reasonable price ranges
            if 'price' in data.columns:
                if (data['price'] <= 0).any() or (data['price'] > 10000).any():
                    logger.warning("Found unreasonable price values")
                    return False
                    
            # Check for reasonable volume ranges
            if 'volume' in data.columns:
                if (data['volume'] < 0).any():
                    logger.warning("Found negative volume values")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error during data quality validation: {e}")
            return False