"""
Data Writer Module
Handles writing data to ClickHouse database with optimized batch operations
"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datetime import datetime
from .clickhouse_client import ClickHouseClient

logger = logging.getLogger(__name__)


class DataWriter:
    """Optimized data writer for ClickHouse database"""
    
    def __init__(self, client: ClickHouseClient):
        """Initialize data writer with ClickHouse client"""
        self.client = client
        
    def write_tick_data(self, data: Union[pd.DataFrame, List[Dict]], batch_size: int = 100000) -> None:
        """Write tick data to database"""
        try:
            if isinstance(data, pd.DataFrame):
                self.client.insert_dataframe('tick_data', data, batch_size)
            else:
                # Convert list of dicts to batch insert
                query = """
                INSERT INTO tick_data (
                    timestamp, symbol, price, volume, turnover,
                    bid_price, bid_volume, ask_price, ask_volume,
                    trade_direction, trade_type, exchange, update_time
                ) VALUES
                """
                self._batch_insert(query, data, batch_size)
                
            logger.info(f"Successfully wrote {len(data)} tick data records")
        except Exception as e:
            logger.error(f"Failed to write tick data: {e}")
            raise
            
    def write_order_book(self, data: Union[pd.DataFrame, List[Dict]], batch_size: int = 50000) -> None:
        """Write order book data to database"""
        try:
            if isinstance(data, pd.DataFrame):
                self.client.insert_dataframe('order_book', data, batch_size)
            else:
                query = """
                INSERT INTO order_book (
                    timestamp, symbol,
                    bid_price_1, bid_volume_1, bid_price_2, bid_volume_2,
                    bid_price_3, bid_volume_3, bid_price_4, bid_volume_4,
                    bid_price_5, bid_volume_5, bid_price_6, bid_volume_6,
                    bid_price_7, bid_volume_7, bid_price_8, bid_volume_8,
                    bid_price_9, bid_volume_9, bid_price_10, bid_volume_10,
                    ask_price_1, ask_volume_1, ask_price_2, ask_volume_2,
                    ask_price_3, ask_volume_3, ask_price_4, ask_volume_4,
                    ask_price_5, ask_volume_5, ask_price_6, ask_volume_6,
                    ask_price_7, ask_volume_7, ask_price_8, ask_volume_8,
                    ask_price_9, ask_volume_9, ask_price_10, ask_volume_10,
                    total_bid_volume, total_ask_volume, bid_ask_imbalance,
                    exchange, update_time
                ) VALUES
                """
                self._batch_insert(query, data, batch_size)
                
            logger.info(f"Successfully wrote {len(data)} order book records")
        except Exception as e:
            logger.error(f"Failed to write order book data: {e}")
            raise
            
    def write_factors(self, data: Union[pd.DataFrame, List[Dict]], batch_size: int = 200000) -> None:
        """Write factor data to database"""
        try:
            if isinstance(data, pd.DataFrame):
                self.client.insert_dataframe('factors', data, batch_size)
            else:
                query = """
                INSERT INTO factors (
                    date, timestamp, symbol, factor_name, factor_value,
                    factor_category, window_size, calculation_time,
                    version, is_valid, quality_score
                ) VALUES
                """
                self._batch_insert(query, data, batch_size)
                
            logger.info(f"Successfully wrote {len(data)} factor records")
        except Exception as e:
            logger.error(f"Failed to write factor data: {e}")
            raise
            
    def write_predictions(self, data: Union[pd.DataFrame, List[Dict]], batch_size: int = 10000) -> None:
        """Write prediction data to database"""
        try:
            if isinstance(data, pd.DataFrame):
                self.client.insert_dataframe('predictions', data, batch_size)
            else:
                query = """
                INSERT INTO predictions (
                    prediction_date, prediction_time, symbol,
                    probability_10pct_1d, probability_10pct_3d,
                    expected_return_1d, expected_return_3d,
                    cnn_score, lstm_score, transformer_score,
                    xgboost_score, lightgbm_score, ensemble_score,
                    rank_score, rank_position, prediction_confidence,
                    model_agreement, top_factors, factor_contributions,
                    model_version, is_selected, selection_reason
                ) VALUES
                """
                self._batch_insert(query, data, batch_size)
                
            logger.info(f"Successfully wrote {len(data)} prediction records")
        except Exception as e:
            logger.error(f"Failed to write prediction data: {e}")
            raise
            
    def write_trade_signals(self, data: Union[pd.DataFrame, List[Dict]], batch_size: int = 10000) -> None:
        """Write trade signals to database"""
        try:
            if isinstance(data, pd.DataFrame):
                self.client.insert_dataframe('trade_signals', data, batch_size)
            else:
                query = """
                INSERT INTO trade_signals (
                    signal_date, signal_time, symbol, signal_type,
                    signal_strength, entry_price, target_price,
                    stop_loss_price, suggested_position_size,
                    max_position_value, expected_return, expected_risk,
                    sharpe_ratio, is_active, execution_status
                ) VALUES
                """
                self._batch_insert(query, data, batch_size)
                
            logger.info(f"Successfully wrote {len(data)} trade signal records")
        except Exception as e:
            logger.error(f"Failed to write trade signal data: {e}")
            raise
            
    def write_portfolio_positions(self, data: Union[pd.DataFrame, List[Dict]], batch_size: int = 5000) -> None:
        """Write portfolio positions to database"""
        try:
            if isinstance(data, pd.DataFrame):
                self.client.insert_dataframe('portfolio_positions', data, batch_size)
            else:
                query = """
                INSERT INTO portfolio_positions (
                    date, timestamp, symbol, quantity, avg_price,
                    current_price, market_value, unrealized_pnl,
                    realized_pnl, total_pnl, pnl_percentage,
                    position_weight, sector, beta, correlation_to_market,
                    entry_date, days_held
                ) VALUES
                """
                self._batch_insert(query, data, batch_size)
                
            logger.info(f"Successfully wrote {len(data)} portfolio position records")
        except Exception as e:
            logger.error(f"Failed to write portfolio position data: {e}")
            raise
            
    def write_strategy_performance(self, data: Union[pd.DataFrame, List[Dict]], batch_size: int = 1000) -> None:
        """Write strategy performance data to database"""
        try:
            if isinstance(data, pd.DataFrame):
                self.client.insert_dataframe('strategy_performance', data, batch_size)
            else:
                query = """
                INSERT INTO strategy_performance (
                    date, timestamp, total_value, cash_balance,
                    positions_value, daily_return, cumulative_return,
                    daily_volatility, sharpe_ratio, max_drawdown,
                    current_drawdown, trades_today, win_rate,
                    avg_win, avg_loss, profit_factor, factor_ic_mean,
                    factor_ic_std, top_performing_factors,
                    prediction_accuracy, model_confidence
                ) VALUES
                """
                self._batch_insert(query, data, batch_size)
                
            logger.info(f"Successfully wrote {len(data)} strategy performance records")
        except Exception as e:
            logger.error(f"Failed to write strategy performance data: {e}")
            raise
            
    def update_trade_signal_status(self, signal_id: str, status: str, 
                                 actual_entry_price: Optional[float] = None,
                                 actual_exit_price: Optional[float] = None,
                                 actual_return: Optional[float] = None) -> None:
        """Update trade signal execution status"""
        try:
            update_fields = [f"execution_status = '{status}'"]
            
            if actual_entry_price is not None:
                update_fields.append(f"actual_entry_price = {actual_entry_price}")
            if actual_exit_price is not None:
                update_fields.append(f"actual_exit_price = {actual_exit_price}")
            if actual_return is not None:
                update_fields.append(f"actual_return = {actual_return}")
                
            query = f"""
            ALTER TABLE trade_signals UPDATE 
            {', '.join(update_fields)}
            WHERE signal_time = '{signal_id}'
            """
            
            self.client.execute(query)
            logger.info(f"Updated trade signal {signal_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update trade signal status: {e}")
            raise
            
    def _batch_insert(self, query: str, data: List[Dict], batch_size: int) -> None:
        """Helper method for batch insertion"""
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            self.client.execute_many(query, batch)
            
    def delete_old_data(self, table: str, days_to_keep: int) -> None:
        """Delete old data beyond retention period"""
        try:
            # Determine the date column based on table
            date_column = 'timestamp' if table in ['tick_data', 'order_book'] else 'date'
            
            query = f"""
            ALTER TABLE {table} DELETE 
            WHERE {date_column} < now() - INTERVAL {days_to_keep} DAY
            """
            
            self.client.execute(query)
            logger.info(f"Deleted data older than {days_to_keep} days from {table}")
        except Exception as e:
            logger.error(f"Failed to delete old data from {table}: {e}")
            raise
            
    def optimize_tables(self, tables: Optional[List[str]] = None) -> None:
        """Optimize specified tables or all tables"""
        if tables is None:
            tables = [
                'tick_data', 'order_book', 'factors', 'predictions',
                'trade_signals', 'portfolio_positions', 'strategy_performance'
            ]
            
        for table in tables:
            try:
                self.client.optimize_table(table)
                logger.info(f"Optimized table: {table}")
            except Exception as e:
                logger.warning(f"Failed to optimize table {table}: {e}")