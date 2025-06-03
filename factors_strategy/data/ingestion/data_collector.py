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
        
    async def get_tick_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get tick data from Tushare"""
        try:
            if not self.token:
                logger.warning("Tushare token not provided, cannot fetch data")
                return pd.DataFrame()
            
            # Tushare requires specific setup and API calls
            # This is a placeholder for actual Tushare implementation
            logger.info(f"Tushare tick data collection for {symbol} would be implemented here")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get tick data from Tushare for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_order_book_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get order book data from Tushare"""
        try:
            if not self.token:
                logger.warning("Tushare token not provided, cannot fetch data")
                return pd.DataFrame()
            
            # Tushare order book implementation would go here
            logger.info(f"Tushare order book data collection for {symbol} would be implemented here")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get order book data from Tushare for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data from Tushare"""
        try:
            if not self.token:
                logger.warning("Tushare token not provided, cannot fetch data")
                return {}
            
            # Tushare real-time data implementation would go here
            logger.info(f"Tushare real-time data collection for {len(symbols)} symbols would be implemented here")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get real-time data from Tushare: {e}")
            return {}


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
    
    def __init__(self, providers: Optional[List[DataProvider]] = None, tushare_token: Optional[str] = None):
        """Initialize data collector with multiple providers"""
        self.session = None
        
        # Initialize providers with fallback order
        if providers is None:
            self.providers = [
                YahooFinanceProvider(),
                TushareProvider(tushare_token),
                SyntheticDataProvider()  # Fallback to synthetic data
            ]
        else:
            self.providers = providers
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def collect_tick_data(self, symbol: str, start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
        """Collect tick data for a symbol using multiple providers with fallback"""
        logger.info(f"Collecting tick data for {symbol} from {start_date} to {end_date}")
        
        for i, provider in enumerate(self.providers):
            try:
                provider_name = provider.__class__.__name__
                logger.info(f"Trying provider {i+1}/{len(self.providers)}: {provider_name}")
                
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
                logger.error(f"Provider {provider.__class__.__name__} failed for tick data: {e}")
                continue
        
        # If all providers fail, raise an exception
        raise Exception(f"All data providers failed to collect tick data for {symbol}")
            
    async def collect_order_book_data(self, symbol: str, start_date: datetime,
                                    end_date: datetime) -> pd.DataFrame:
        """Collect order book data for a symbol using multiple providers with fallback"""
        logger.info(f"Collecting order book data for {symbol} from {start_date} to {end_date}")
        
        for i, provider in enumerate(self.providers):
            try:
                provider_name = provider.__class__.__name__
                logger.info(f"Trying provider {i+1}/{len(self.providers)}: {provider_name}")
                
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
                logger.error(f"Provider {provider.__class__.__name__} failed for order book data: {e}")
                continue
        
        # If all providers fail, raise an exception
        raise Exception(f"All data providers failed to collect order book data for {symbol}")
            
    async def collect_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect real-time market data using multiple providers with fallback"""
        logger.info(f"Collecting real-time data for {len(symbols)} symbols")
        
        for i, provider in enumerate(self.providers):
            try:
                provider_name = provider.__class__.__name__
                logger.info(f"Trying provider {i+1}/{len(self.providers)}: {provider_name}")
                
                real_time_data = await provider.get_real_time_data(symbols)
                
                if real_time_data:
                    logger.info(f"Successfully collected real-time data for {len(real_time_data)} symbols from {provider_name}")
                    return real_time_data
                else:
                    logger.warning(f"No real-time data returned from {provider_name}, trying next provider")
                    continue
                    
            except Exception as e:
                logger.error(f"Provider {provider.__class__.__name__} failed for real-time data: {e}")
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
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'book_value': info.get('bookValue', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'updated_at': datetime.now(pytz.UTC)
            }
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Failed to collect fundamental data for {symbol}: {e}")
            # Return empty data structure on failure
            return {
                'symbol': symbol,
                'market_cap': 0,
                'pe_ratio': 0,
                'pb_ratio': 0,
                'roe': 0,
                'debt_to_equity': 0,
                'revenue_growth': 0,
                'profit_margin': 0,
                'dividend_yield': 0,
                'enterprise_value': 0,
                'price_to_sales': 0,
                'book_value': 0,
                'earnings_growth': 0,
                'updated_at': datetime.now(pytz.UTC)
            }
    
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
                logger.warning(f"Empty {data_type} dataset")
                return False
                
            # Check for required columns
            required_columns = {
                'tick_data': ['timestamp', 'symbol', 'price', 'volume'],
                'order_book': ['timestamp', 'symbol', 'bid_price_1', 'ask_price_1']
            }
            
            if data_type in required_columns:
                missing_cols = set(required_columns[data_type]) - set(data.columns)
                if missing_cols:
                    logger.error(f"Missing required columns for {data_type}: {missing_cols}")
                    return False
                    
            # Check for null values in critical columns
            if data_type == 'tick_data':
                critical_cols = ['timestamp', 'symbol', 'price']
                null_counts = data[critical_cols].isnull().sum()
                if null_counts.any():
                    logger.warning(f"Null values found in critical columns: {null_counts}")
                    
            # Check timestamp ordering
            if 'timestamp' in data.columns:
                if not data['timestamp'].is_monotonic_increasing:
                    logger.warning(f"Timestamps not in ascending order for {data_type}")
                    
            logger.info(f"Data quality validation passed for {data_type}")
            return True
            
        except Exception as e:
            logger.error(f"Data quality validation failed for {data_type}: {e}")
            return False