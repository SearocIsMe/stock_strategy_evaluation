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

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects market data from various sources"""
    
    def __init__(self):
        """Initialize data collector"""
        self.session = None
        
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
        """Collect tick data for a symbol"""
        try:
            logger.info(f"Collecting tick data for {symbol} from {start_date} to {end_date}")
            
            # For demo purposes, generate sample tick data
            # In production, this would connect to real data sources
            tick_data = self._generate_sample_tick_data(symbol, start_date, end_date)
            
            logger.info(f"Collected {len(tick_data)} tick records for {symbol}")
            return tick_data
            
        except Exception as e:
            logger.error(f"Failed to collect tick data for {symbol}: {e}")
            raise
            
    async def collect_order_book_data(self, symbol: str, start_date: datetime,
                                    end_date: datetime) -> pd.DataFrame:
        """Collect order book data for a symbol"""
        try:
            logger.info(f"Collecting order book data for {symbol} from {start_date} to {end_date}")
            
            # For demo purposes, generate sample order book data
            # In production, this would connect to real data sources
            order_book_data = self._generate_sample_order_book_data(symbol, start_date, end_date)
            
            logger.info(f"Collected {len(order_book_data)} order book records for {symbol}")
            return order_book_data
            
        except Exception as e:
            logger.error(f"Failed to collect order book data for {symbol}: {e}")
            raise
            
    def _generate_sample_tick_data(self, symbol: str, start_date: datetime, 
                                 end_date: datetime) -> pd.DataFrame:
        """Generate sample tick data for testing"""
        
        # Generate timestamps every minute
        timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
        n_records = len(timestamps)
        
        if n_records == 0:
            return pd.DataFrame()
            
        # Generate realistic price movements
        base_price = 100.0
        price_changes = np.random.normal(0, 0.001, n_records)
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        # Generate volumes
        volumes = np.random.lognormal(mean=8, sigma=1, size=n_records)
        
        # Generate bid/ask spreads
        spreads = np.random.uniform(0.01, 0.05, n_records)
        bid_prices = prices - spreads / 2
        ask_prices = prices + spreads / 2
        
        # Create DataFrame
        tick_data = pd.DataFrame({
            'timestamp': timestamps,
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
            'exchange': 'SZSE'
        })
        
        return tick_data
        
    def _generate_sample_order_book_data(self, symbol: str, start_date: datetime,
                                       end_date: datetime) -> pd.DataFrame:
        """Generate sample order book data for testing"""
        
        # Generate timestamps every 5 minutes
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        n_records = len(timestamps)
        
        if n_records == 0:
            return pd.DataFrame()
            
        # Generate realistic price movements
        base_price = 100.0
        price_changes = np.random.normal(0, 0.001, n_records)
        mid_prices = base_price * np.exp(np.cumsum(price_changes))
        
        # Generate order book levels
        order_book_data = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': symbol,
            'exchange': 'SZSE'
        })
        
        # Generate 5 levels of bid/ask
        for level in range(1, 6):
            # Bid side
            bid_offset = level * 0.01
            order_book_data[f'bid_price_{level}'] = mid_prices - bid_offset
            order_book_data[f'bid_volume_{level}'] = np.random.lognormal(6, 1, n_records)
            
            # Ask side
            ask_offset = level * 0.01
            order_book_data[f'ask_price_{level}'] = mid_prices + ask_offset
            order_book_data[f'ask_volume_{level}'] = np.random.lognormal(6, 1, n_records)
            
        # Calculate aggregated metrics
        bid_volumes = [order_book_data[f'bid_volume_{i}'] for i in range(1, 6)]
        ask_volumes = [order_book_data[f'ask_volume_{i}'] for i in range(1, 6)]
        
        order_book_data['total_bid_volume'] = sum(bid_volumes)
        order_book_data['total_ask_volume'] = sum(ask_volumes)
        order_book_data['bid_ask_imbalance'] = (
            order_book_data['total_bid_volume'] - order_book_data['total_ask_volume']
        ) / (order_book_data['total_bid_volume'] + order_book_data['total_ask_volume'])
        
        return order_book_data
        
    async def collect_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect real-time market data"""
        try:
            logger.info(f"Collecting real-time data for {len(symbols)} symbols")
            
            # In production, this would connect to real-time data feeds
            # For now, return sample data
            real_time_data = {}
            
            for symbol in symbols:
                real_time_data[symbol] = {
                    'timestamp': datetime.now(),
                    'price': 100.0 + np.random.normal(0, 1),
                    'volume': np.random.lognormal(8, 1),
                    'bid_price': 99.95 + np.random.normal(0, 0.1),
                    'ask_price': 100.05 + np.random.normal(0, 0.1),
                    'change_pct': np.random.normal(0, 0.02)
                }
                
            return real_time_data
            
        except Exception as e:
            logger.error(f"Failed to collect real-time data: {e}")
            raise
            
    async def collect_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Collect fundamental data for a symbol"""
        try:
            logger.info(f"Collecting fundamental data for {symbol}")
            
            # In production, this would fetch real fundamental data
            # For now, return sample data
            fundamental_data = {
                'symbol': symbol,
                'market_cap': np.random.uniform(1e9, 1e12),
                'pe_ratio': np.random.uniform(5, 50),
                'pb_ratio': np.random.uniform(0.5, 5),
                'roe': np.random.uniform(0.05, 0.25),
                'debt_to_equity': np.random.uniform(0.1, 2.0),
                'revenue_growth': np.random.uniform(-0.1, 0.3),
                'profit_margin': np.random.uniform(0.01, 0.2),
                'dividend_yield': np.random.uniform(0, 0.08),
                'updated_at': datetime.now()
            }
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Failed to collect fundamental data for {symbol}: {e}")
            raise
            
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