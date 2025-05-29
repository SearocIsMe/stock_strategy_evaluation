"""
Microstructure Factors
High-frequency factors derived from tick data and order book dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from numba import jit, prange

logger = logging.getLogger(__name__)


class MicrostructureFactors:
    """Calculate microstructure-based factors from tick data"""
    
    def __init__(self, config: Dict):
        """Initialize with factor configuration"""
        self.config = config
        self.window_sizes = config.get('window_sizes', [10, 30, 60, 300])
        
    def calculate_all_factors(self, tick_data: pd.DataFrame, 
                            order_book: pd.DataFrame) -> pd.DataFrame:
        """Calculate all microstructure factors"""
        factors = pd.DataFrame(index=tick_data.index)
        
        # Order flow imbalance
        ofi_factors = self.calculate_order_flow_imbalance(tick_data, order_book)
        factors = pd.concat([factors, ofi_factors], axis=1)
        
        # Bid-ask spread
        spread_factors = self.calculate_bid_ask_spread(order_book)
        factors = pd.concat([factors, spread_factors], axis=1)
        
        # Order book depth ratio
        depth_factors = self.calculate_order_book_depth_ratio(order_book)
        factors = pd.concat([factors, depth_factors], axis=1)
        
        # Price impact
        impact_factors = self.calculate_price_impact(tick_data)
        factors = pd.concat([factors, impact_factors], axis=1)
        
        # Trade intensity
        intensity_factors = self.calculate_trade_intensity(tick_data)
        factors = pd.concat([factors, intensity_factors], axis=1)
        
        # Microstructure noise
        noise_factors = self.calculate_microstructure_noise(tick_data)
        factors = pd.concat([factors, noise_factors], axis=1)
        
        return factors
        
    def calculate_order_flow_imbalance(self, tick_data: pd.DataFrame, 
                                     order_book: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow imbalance (OFI)"""
        factors = pd.DataFrame(index=tick_data.index)
        
        for window in self.window_sizes:
            # Calculate buy/sell volumes
            buy_volume = tick_data['volume'].where(tick_data['trade_direction'] == 1, 0)
            sell_volume = tick_data['volume'].where(tick_data['trade_direction'] == -1, 0)
            
            # Rolling window calculation
            buy_sum = buy_volume.rolling(f'{window}s').sum()
            sell_sum = sell_volume.rolling(f'{window}s').sum()
            
            # Order flow imbalance
            ofi = (buy_sum - sell_sum) / (buy_sum + sell_sum + 1e-10)
            factors[f'ofi_{window}s'] = ofi
            
            # Weighted OFI by price level
            if 'bid_volume' in order_book.columns:
                bid_volume = order_book[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1)
                ask_volume = order_book[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1)
                
                weighted_ofi = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
                factors[f'weighted_ofi_{window}s'] = weighted_ofi.rolling(f'{window}s').mean()
                
        return factors
        
    def calculate_bid_ask_spread(self, order_book: pd.DataFrame) -> pd.DataFrame:
        """Calculate various bid-ask spread measures"""
        factors = pd.DataFrame(index=order_book.index)
        
        # Basic spread
        spread = order_book['ask_price_1'] - order_book['bid_price_1']
        mid_price = (order_book['ask_price_1'] + order_book['bid_price_1']) / 2
        
        # Relative spread
        factors['relative_spread'] = spread / mid_price
        
        # Effective spread (considering depth)
        for level in [1, 3, 5]:
            if f'ask_price_{level}' in order_book.columns:
                weighted_ask = 0
                weighted_bid = 0
                total_ask_vol = 0
                total_bid_vol = 0
                
                for i in range(1, level + 1):
                    weighted_ask += order_book[f'ask_price_{i}'] * order_book[f'ask_volume_{i}']
                    weighted_bid += order_book[f'bid_price_{i}'] * order_book[f'bid_volume_{i}']
                    total_ask_vol += order_book[f'ask_volume_{i}']
                    total_bid_vol += order_book[f'bid_volume_{i}']
                
                effective_ask = weighted_ask / (total_ask_vol + 1e-10)
                effective_bid = weighted_bid / (total_bid_vol + 1e-10)
                effective_spread = (effective_ask - effective_bid) / mid_price
                
                factors[f'effective_spread_l{level}'] = effective_spread
                
        # Time-weighted average spread
        for window in self.window_sizes:
            factors[f'twap_spread_{window}s'] = factors['relative_spread'].rolling(f'{window}s').mean()
            
        return factors
        
    def calculate_order_book_depth_ratio(self, order_book: pd.DataFrame) -> pd.DataFrame:
        """Calculate order book depth ratios"""
        factors = pd.DataFrame(index=order_book.index)
        
        # Depth ratios at different levels
        for level in [1, 3, 5, 10]:
            bid_depth = 0
            ask_depth = 0
            
            for i in range(1, min(level + 1, 11)):  # Max 10 levels
                if f'bid_volume_{i}' in order_book.columns:
                    bid_depth += order_book[f'bid_volume_{i}']
                    ask_depth += order_book[f'ask_volume_{i}']
                    
            depth_ratio = bid_depth / (ask_depth + 1e-10)
            factors[f'depth_ratio_l{level}'] = depth_ratio
            
            # Log depth ratio (more stable)
            factors[f'log_depth_ratio_l{level}'] = np.log(depth_ratio + 1e-10)
            
        # Depth imbalance at best quotes
        best_bid_vol = order_book['bid_volume_1']
        best_ask_vol = order_book['ask_volume_1']
        factors['best_depth_imbalance'] = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol + 1e-10)
        
        # Depth concentration (how much volume is at best quotes)
        total_bid_depth = sum(order_book[f'bid_volume_{i}'] for i in range(1, 6))
        total_ask_depth = sum(order_book[f'ask_volume_{i}'] for i in range(1, 6))
        
        factors['bid_depth_concentration'] = best_bid_vol / (total_bid_depth + 1e-10)
        factors['ask_depth_concentration'] = best_ask_vol / (total_ask_depth + 1e-10)
        
        return factors
        
    @staticmethod
    @jit(nopython=True)
    def _calculate_kyle_lambda(price_changes: np.ndarray, volumes: np.ndarray, 
                              window_size: int) -> np.ndarray:
        """Numba-optimized Kyle's lambda calculation"""
        n = len(price_changes)
        kyle_lambda = np.zeros(n)
        
        for i in prange(window_size, n):
            window_prices = price_changes[i-window_size:i]
            window_volumes = volumes[i-window_size:i]
            
            # Remove zeros
            mask = window_volumes > 0
            if np.sum(mask) > 10:  # Need enough data points
                valid_prices = window_prices[mask]
                valid_volumes = window_volumes[mask]
                
                # Simple regression: price_change = lambda * volume
                numerator = np.sum(valid_prices * valid_volumes)
                denominator = np.sum(valid_volumes * valid_volumes)
                
                if denominator > 0:
                    kyle_lambda[i] = numerator / denominator
                    
        return kyle_lambda
        
    def calculate_price_impact(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price impact measures (Kyle's lambda)"""
        factors = pd.DataFrame(index=tick_data.index)
        
        # Price changes
        price_changes = tick_data['price'].diff()
        volumes = tick_data['volume']
        
        for window in self.window_sizes:
            # Kyle's lambda
            window_ticks = int(window * 2)  # Approximate ticks in window
            kyle_lambda = self._calculate_kyle_lambda(
                price_changes.values, 
                volumes.values, 
                window_ticks
            )
            factors[f'kyle_lambda_{window}s'] = kyle_lambda
            
            # Amihud illiquidity
            returns = tick_data['price'].pct_change()
            amihud = (returns.abs() / (volumes + 1e-10)).rolling(f'{window}s').mean()
            factors[f'amihud_illiq_{window}s'] = amihud
            
            # Temporary vs permanent impact
            future_returns = returns.shift(-window_ticks)
            temp_impact = returns / (volumes + 1e-10)
            perm_impact = future_returns / (volumes + 1e-10)
            
            factors[f'temp_impact_{window}s'] = temp_impact.rolling(f'{window}s').mean()
            factors[f'perm_impact_{window}s'] = perm_impact.rolling(f'{window}s').mean()
            
        return factors
        
    def calculate_trade_intensity(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trade intensity and arrival rate factors"""
        factors = pd.DataFrame(index=tick_data.index)
        
        # Trade counts in windows
        for window in self.window_sizes:
            # Number of trades
            trade_count = tick_data.rolling(f'{window}s').count()['price']
            factors[f'trade_count_{window}s'] = trade_count
            
            # Trade arrival rate (trades per second)
            factors[f'trade_rate_{window}s'] = trade_count / window
            
            # Volume-weighted trade size
            total_volume = tick_data['volume'].rolling(f'{window}s').sum()
            avg_trade_size = total_volume / (trade_count + 1e-10)
            factors[f'avg_trade_size_{window}s'] = avg_trade_size
            
            # Trade size variance
            trade_sizes = tick_data['volume']
            size_variance = trade_sizes.rolling(f'{window}s').std()
            factors[f'trade_size_std_{window}s'] = size_variance
            
        # Trade clustering (autocorrelation of trade arrivals)
        trade_indicator = (tick_data['volume'] > 0).astype(int)
        for lag in [1, 5, 10]:
            factors[f'trade_autocorr_lag{lag}'] = trade_indicator.rolling(300).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
            
        return factors
        
    def calculate_microstructure_noise(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate microstructure noise measures"""
        factors = pd.DataFrame(index=tick_data.index)
        
        # Realized variance at different sampling frequencies
        returns_1s = tick_data['price'].resample('1s').last().pct_change()
        returns_5s = tick_data['price'].resample('5s').last().pct_change()
        returns_10s = tick_data['price'].resample('10s').last().pct_change()
        
        # Variance ratio test for noise
        for window in self.window_sizes:
            rv_1s = returns_1s.rolling(f'{window}s').var()
            rv_5s = returns_5s.rolling(f'{window}s').var()
            rv_10s = returns_10s.rolling(f'{window}s').var()
            
            # Reindex to original frequency
            rv_1s = rv_1s.reindex(tick_data.index, method='ffill')
            rv_5s = rv_5s.reindex(tick_data.index, method='ffill')
            rv_10s = rv_10s.reindex(tick_data.index, method='ffill')
            
            # Variance ratios
            factors[f'var_ratio_5s_1s_{window}s'] = rv_5s / (5 * rv_1s + 1e-10)
            factors[f'var_ratio_10s_1s_{window}s'] = rv_10s / (10 * rv_1s + 1e-10)
            
            # Noise estimate (deviation from 1)
            factors[f'noise_estimate_{window}s'] = np.abs(factors[f'var_ratio_5s_1s_{window}s'] - 1)
            
        return factors