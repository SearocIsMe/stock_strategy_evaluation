"""
Data Preprocessor Module
Handles data cleaning, validation, and preprocessing
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from factors.traditional.microstructure_factors import MicrostructureFactors

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Thread-%(thread)d] [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses and cleans market data"""
    
    def __init__(self):
        """Initialize data preprocessor"""
        # Default config for microstructure factors
        default_config = {
            'window_sizes': [10, 30, 60, 300],
            'calculation_method': 'standard'
        }
        self.microstructure_factors = MicrostructureFactors(default_config)
        
    def process_tick_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean tick data"""
        try:
            logger.info(f"Processing {len(raw_data)} tick records")
            
            if raw_data.empty:
                return raw_data
                
            # Make a copy to avoid modifying original data
            data = raw_data.copy()
            
            # Clean and validate data
            data = self._clean_tick_data(data)
            data = self._validate_tick_data(data)
            data = self._enrich_tick_data(data)
            
            logger.info(f"Processed tick data: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to process tick data: {e}")
            raise
            
    def process_order_book_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean order book data"""
        try:
            logger.info(f"Processing {len(raw_data)} order book records")
            
            if raw_data.empty:
                return raw_data
                
            # Make a copy to avoid modifying original data
            data = raw_data.copy()
            
            # Clean and validate data
            data = self._clean_order_book_data(data)
            data = self._validate_order_book_data(data)
            data = self._enrich_order_book_data(data)
            
            logger.info(f"Processed order book data: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to process order book data: {e}")
            raise
            
    def calculate_factors(self, tick_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate factors from tick data"""
        try:
            logger.info(f"Calculating factors for {symbol} from {len(tick_data)} tick records")
            
            if tick_data.empty:
                return pd.DataFrame()
                
            # Calculate microstructure factors
            factors_data = []
            
            # Group by date for daily factor calculation
            tick_data['date'] = tick_data['timestamp'].dt.date
            
            for date, day_data in tick_data.groupby('date'):
                if len(day_data) < 10:  # Skip days with insufficient data
                    continue
                    
                # Calculate various factors
                daily_factors = self._calculate_daily_factors(day_data, symbol, date)
                factors_data.extend(daily_factors)
                
            if not factors_data:
                logger.warning(f"No factors calculated for {symbol}")
                return pd.DataFrame()
                
            factors_df = pd.DataFrame(factors_data)
            logger.info(f"Calculated {len(factors_df)} factor records for {symbol}")
            
            return factors_df
            
        except Exception as e:
            logger.error(f"Failed to calculate factors for {symbol}: {e}")
            raise
            
    def _clean_tick_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean tick data"""
        
        # Remove duplicates
        initial_count = len(data)
        data = data.drop_duplicates(subset=['timestamp', 'symbol'])
        if len(data) < initial_count:
            logger.info(f"Removed {initial_count - len(data)} duplicate records")
            
        # Remove invalid prices (negative or zero)
        data = data[data['price'] > 0]
        data = data[data['volume'] > 0]
        
        # Remove extreme outliers (prices more than 10 standard deviations from mean)
        price_mean = data['price'].mean()
        price_std = data['price'].std()
        price_threshold = 10 * price_std
        
        data = data[
            (data['price'] >= price_mean - price_threshold) &
            (data['price'] <= price_mean + price_threshold)
        ]
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        return data
        
    def _validate_tick_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate tick data"""
        
        # Check for required columns
        required_cols = ['timestamp', 'symbol', 'price', 'volume']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
        # Ensure numeric columns are numeric
        numeric_cols = ['price', 'volume', 'turnover']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
        # Remove rows with NaN values in critical columns
        data = data.dropna(subset=['timestamp', 'symbol', 'price', 'volume'])
        
        return data
        
    def _enrich_tick_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enrich tick data with additional fields"""
        
        # Calculate returns
        data['returns'] = data.groupby('symbol')['price'].pct_change()
        
        # Calculate VWAP
        data['vwap'] = (data['price'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Add time-based features
        data['hour'] = data['timestamp'].dt.hour
        data['minute'] = data['timestamp'].dt.minute
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Calculate rolling statistics
        window = min(20, len(data))
        if window > 1:
            data['price_ma'] = data['price'].rolling(window=window).mean()
            data['volume_ma'] = data['volume'].rolling(window=window).mean()
            data['volatility'] = data['returns'].rolling(window=window).std()
            
        return data
        
    def _clean_order_book_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean order book data"""
        
        # Remove duplicates
        initial_count = len(data)
        data = data.drop_duplicates(subset=['timestamp', 'symbol'])
        if len(data) < initial_count:
            logger.info(f"Removed {initial_count - len(data)} duplicate order book records")
            
        # Remove invalid prices and volumes (10 levels to match schema)
        for level in range(1, 11):
            bid_price_col = f'bid_price_{level}'
            ask_price_col = f'ask_price_{level}'
            bid_vol_col = f'bid_volume_{level}'
            ask_vol_col = f'ask_volume_{level}'
            
            if bid_price_col in data.columns:
                data = data[data[bid_price_col] > 0]
            if ask_price_col in data.columns:
                data = data[data[ask_price_col] > 0]
            if bid_vol_col in data.columns:
                data = data[data[bid_vol_col] > 0]
            if ask_vol_col in data.columns:
                data = data[data[ask_vol_col] > 0]
                
        # Ensure bid prices < ask prices
        if 'bid_price_1' in data.columns and 'ask_price_1' in data.columns:
            data = data[data['bid_price_1'] < data['ask_price_1']]
            
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        return data
        
    def _validate_order_book_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate order book data"""
        
        # Check for required columns
        required_cols = ['timestamp', 'symbol', 'bid_price_1', 'ask_price_1']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
        # Ensure numeric columns are numeric
        for level in range(1, 11):
            for side in ['bid', 'ask']:
                price_col = f'{side}_price_{level}'
                vol_col = f'{side}_volume_{level}'
                
                if price_col in data.columns:
                    data[price_col] = pd.to_numeric(data[price_col], errors='coerce')
                if vol_col in data.columns:
                    data[vol_col] = pd.to_numeric(data[vol_col], errors='coerce')
                    
        return data
        
    def _enrich_order_book_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enrich order book data with additional metrics"""
        
        # Calculate mid price
        if 'bid_price_1' in data.columns and 'ask_price_1' in data.columns:
            data['mid_price'] = (data['bid_price_1'] + data['ask_price_1']) / 2
            data['spread'] = data['ask_price_1'] - data['bid_price_1']
            data['spread_bps'] = (data['spread'] / data['mid_price']) * 10000
            
        # Calculate weighted mid price
        if all(col in data.columns for col in ['bid_price_1', 'ask_price_1', 'bid_volume_1', 'ask_volume_1']):
            total_vol = data['bid_volume_1'] + data['ask_volume_1']
            data['weighted_mid_price'] = (
                (data['bid_price_1'] * data['ask_volume_1'] + data['ask_price_1'] * data['bid_volume_1']) / total_vol
            )
            
        return data
        
    def _calculate_daily_factors(self, day_data: pd.DataFrame, symbol: str, date) -> List[Dict]:
        """Calculate daily factors from tick data"""
        
        factors = []
        
        try:
            # Basic price factors
            open_price = day_data.iloc[0]['price']
            close_price = day_data.iloc[-1]['price']
            high_price = day_data['price'].max()
            low_price = day_data['price'].min()
            
            # Volume factors
            total_volume = day_data['volume'].sum()
            avg_volume = day_data['volume'].mean()
            volume_std = day_data['volume'].std()
            
            # Return factors
            daily_return = (close_price - open_price) / open_price
            
            # Calculate returns if not present
            if 'returns' not in day_data.columns:
                day_data['returns'] = day_data['price'].pct_change()
            
            # Volatility factors
            returns = day_data['returns'].dropna()
            if len(returns) > 1:
                volatility = returns.std()
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
            else:
                volatility = skewness = kurtosis = 0
                
            # VWAP factor
            vwap = (day_data['price'] * day_data['volume']).sum() / day_data['volume'].sum()
            
            # Create factor records with update_time
            current_time = datetime.now()
            factor_records = [
                {
                    'date': date,
                    'timestamp': datetime.combine(date, datetime.min.time()),
                    'symbol': symbol,
                    'factor_name': 'daily_return',
                    'factor_value': daily_return,
                    'factor_category': 'price',
                    'window_size': 1,
                    'calculation_time': 0.001,
                    'version': '1.0',
                    'is_valid': 1,
                    'quality_score': 1.0,
                    'update_time': current_time
                },
                {
                    'date': date,
                    'timestamp': datetime.combine(date, datetime.min.time()),
                    'symbol': symbol,
                    'factor_name': 'volatility',
                    'factor_value': volatility,
                    'factor_category': 'risk',
                    'window_size': len(day_data),
                    'calculation_time': 0.001,
                    'version': '1.0',
                    'is_valid': 1,
                    'quality_score': 1.0,
                    'update_time': current_time
                },
                {
                    'date': date,
                    'timestamp': datetime.combine(date, datetime.min.time()),
                    'symbol': symbol,
                    'factor_name': 'volume_ratio',
                    'factor_value': total_volume / avg_volume if avg_volume > 0 else 0,
                    'factor_category': 'volume',
                    'window_size': len(day_data),
                    'calculation_time': 0.001,
                    'version': '1.0',
                    'is_valid': 1,
                    'quality_score': 1.0,
                    'update_time': current_time
                },
                {
                    'date': date,
                    'timestamp': datetime.combine(date, datetime.min.time()),
                    'symbol': symbol,
                    'factor_name': 'vwap_deviation',
                    'factor_value': (close_price - vwap) / vwap,
                    'factor_category': 'price',
                    'window_size': len(day_data),
                    'calculation_time': 0.001,
                    'version': '1.0',
                    'is_valid': 1,
                    'quality_score': 1.0,
                    'update_time': current_time
                }
            ]
            
            factors.extend(factor_records)
            
        except Exception as e:
            logger.error(f"Failed to calculate daily factors for {symbol} on {date}: {e}")
            
        return factors
        
    def normalize_data(self, data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Normalize numerical data"""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if method == 'zscore':
            data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
        elif method == 'minmax':
            data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].min()) / (data[numeric_cols].max() - data[numeric_cols].min())
        elif method == 'robust':
            data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].median()) / (data[numeric_cols].quantile(0.75) - data[numeric_cols].quantile(0.25))
            
        return data