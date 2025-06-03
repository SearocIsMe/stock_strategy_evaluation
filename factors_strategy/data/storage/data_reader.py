"""
Data Reader Module
Handles reading data from ClickHouse database with optimized queries
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
from .clickhouse_client import ClickHouseClient

logger = logging.getLogger(__name__)


class DataReader:
    """Optimized data reader for ClickHouse database"""
    
    def __init__(self, client: ClickHouseClient):
        """Initialize data reader with ClickHouse client"""
        self.client = client
        
    def get_tick_data(self, symbols: Union[str, List[str]], 
                     start_time: datetime, end_time: datetime,
                     limit: Optional[int] = None) -> pd.DataFrame:
        """Get tick data for specified symbols and time range"""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
                
            symbols_str = "', '".join(symbols)
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
            SELECT 
                timestamp, symbol, price, volume, turnover,
                bid_price, bid_volume, ask_price, ask_volume,
                trade_direction, trade_type, mid_price, spread, exchange
            FROM tick_data
            WHERE symbol IN ('{symbols_str}')
            AND timestamp BETWEEN '{start_time}' AND '{end_time}'
            ORDER BY timestamp DESC
            {limit_clause}
            """
            
            return self.client.query_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get tick data: {e}")
            raise
            
    def get_order_book(self, symbols: Union[str, List[str]], 
                      start_time: datetime, end_time: datetime,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """Get order book data for specified symbols and time range"""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
                
            symbols_str = "', '".join(symbols)
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
            SELECT 
                timestamp, symbol,
                bid_price_1, bid_volume_1, bid_price_2, bid_volume_2,
                bid_price_3, bid_volume_3, bid_price_4, bid_volume_4,
                bid_price_5, bid_volume_5,
                ask_price_1, ask_volume_1, ask_price_2, ask_volume_2,
                ask_price_3, ask_volume_3, ask_price_4, ask_volume_4,
                ask_price_5, ask_volume_5,
                total_bid_volume, total_ask_volume, bid_ask_imbalance,
                exchange
            FROM order_book
            WHERE symbol IN ('{symbols_str}')
            AND timestamp BETWEEN '{start_time}' AND '{end_time}'
            ORDER BY timestamp DESC
            {limit_clause}
            """
            
            return self.client.query_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get order book data: {e}")
            raise
            
    def get_factors(self, symbols: Union[str, List[str]], 
                   factor_names: Optional[Union[str, List[str]]] = None,
                   start_date: Optional[datetime] = None, 
                   end_date: Optional[datetime] = None,
                   factor_category: Optional[str] = None) -> pd.DataFrame:
        """Get factor data with optional filtering"""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
                
            symbols_str = "', '".join(symbols)
            conditions = [f"symbol IN ('{symbols_str}')"]
            
            if factor_names:
                if isinstance(factor_names, str):
                    factor_names = [factor_names]
                factor_names_str = "', '".join(factor_names)
                conditions.append(f"factor_name IN ('{factor_names_str}')")
                
            if start_date:
                conditions.append(f"date >= '{start_date.date()}'")
            if end_date:
                conditions.append(f"date <= '{end_date.date()}'")
            if factor_category:
                conditions.append(f"factor_category = '{factor_category}'")
                
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT 
                date, timestamp, symbol, factor_name, factor_value,
                factor_category, window_size, calculation_time,
                version, is_valid, quality_score
            FROM factors
            WHERE {where_clause}
            ORDER BY date DESC, symbol, factor_name
            """
            
            return self.client.query_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get factor data: {e}")
            raise
            
    def get_latest_factors(self, symbols: Union[str, List[str]], 
                          factor_names: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """Get latest factor values for specified symbols"""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
                
            symbols_str = "', '".join(symbols)
            factor_filter = ""
            if factor_names:
                if isinstance(factor_names, str):
                    factor_names = [factor_names]
                factor_names_str = "', '".join(factor_names)
                factor_filter = f"AND factor_name IN ('{factor_names_str}')"
                
            query = f"""
            SELECT 
                symbol, factor_name, factor_value, factor_category,
                date, timestamp, quality_score
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY symbol, factor_name ORDER BY date DESC) as rn
                FROM factors
                WHERE symbol IN ('{symbols_str}')
                {factor_filter}
                AND is_valid = 1
            ) t
            WHERE rn = 1
            ORDER BY symbol, factor_name
            """
            
            return self.client.query_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get latest factors: {e}")
            raise
            
    def get_predictions(self, symbols: Union[str, List[str]], 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       selected_only: bool = False) -> pd.DataFrame:
        """Get prediction data for specified symbols"""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
                
            symbols_str = "', '".join(symbols)
            conditions = [f"symbol IN ('{symbols_str}')"]
            
            if start_date:
                conditions.append(f"prediction_date >= '{start_date.date()}'")
            if end_date:
                conditions.append(f"prediction_date <= '{end_date.date()}'")
            if selected_only:
                conditions.append("is_selected = 1")
                
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT 
                prediction_date, prediction_time, symbol,
                probability_10pct_1d, probability_10pct_3d,
                expected_return_1d, expected_return_3d,
                cnn_score, lstm_score, transformer_score,
                xgboost_score, lightgbm_score, ensemble_score,
                rank_score, rank_position, prediction_confidence,
                model_agreement, top_factors, factor_contributions,
                model_version, is_selected, selection_reason
            FROM predictions
            WHERE {where_clause}
            ORDER BY prediction_date DESC, rank_position ASC
            """
            
            return self.client.query_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            raise
            
    def get_latest_predictions(self, limit: int = 50) -> pd.DataFrame:
        """Get latest top predictions"""
        try:
            query = f"""
            SELECT 
                prediction_date, symbol, ensemble_score, rank_position,
                expected_return_1d, expected_return_3d, prediction_confidence,
                top_factors, is_selected
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY prediction_date DESC) as rn
                FROM predictions
                WHERE prediction_date >= today() - 7
            ) t
            WHERE rn = 1
            ORDER BY rank_position ASC
            LIMIT {limit}
            """
            
            return self.client.query_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get latest predictions: {e}")
            raise
            
    def get_trade_signals(self, symbols: Union[str, List[str]] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         signal_type: Optional[str] = None,
                         active_only: bool = False) -> pd.DataFrame:
        """Get trade signals with optional filtering"""
        try:
            conditions = []
            
            if symbols:
                if isinstance(symbols, str):
                    symbols = [symbols]
                symbols_str = "', '".join(symbols)
                conditions.append(f"symbol IN ('{symbols_str}')")
                
            if start_date:
                conditions.append(f"signal_date >= '{start_date.date()}'")
            if end_date:
                conditions.append(f"signal_date <= '{end_date.date()}'")
            if signal_type:
                conditions.append(f"signal_type = '{signal_type}'")
            if active_only:
                conditions.append("is_active = 1")
                
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            SELECT 
                signal_date, signal_time, symbol, signal_type,
                signal_strength, entry_price, target_price,
                stop_loss_price, suggested_position_size,
                expected_return, expected_risk, sharpe_ratio,
                is_active, execution_status, actual_entry_price,
                actual_exit_price, actual_return
            FROM trade_signals
            {where_clause}
            ORDER BY signal_date DESC, signal_time DESC
            """
            
            return self.client.query_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get trade signals: {e}")
            raise
            
    def get_portfolio_positions(self, date: Optional[datetime] = None,
                              symbols: Union[str, List[str]] = None) -> pd.DataFrame:
        """Get current or historical portfolio positions"""
        try:
            conditions = []
            
            if date:
                conditions.append(f"date = '{date.date()}'")
            else:
                # Get latest positions
                conditions.append("date = (SELECT max(date) FROM portfolio_positions)")
                
            if symbols:
                if isinstance(symbols, str):
                    symbols = [symbols]
                symbols_str = "', '".join(symbols)
                conditions.append(f"symbol IN ('{symbols_str}')")
                
            where_clause = " AND ".join(conditions)
            
            query = f"""
            SELECT 
                date, symbol, quantity, avg_price, current_price,
                market_value, unrealized_pnl, realized_pnl,
                total_pnl, pnl_percentage, position_weight,
                sector, beta, correlation_to_market,
                entry_date, days_held
            FROM portfolio_positions
            WHERE {where_clause}
            ORDER BY market_value DESC
            """
            
            return self.client.query_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get portfolio positions: {e}")
            raise
            
    def get_strategy_performance(self, start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               limit: Optional[int] = None) -> pd.DataFrame:
        """Get strategy performance metrics"""
        try:
            conditions = []
            
            if start_date:
                conditions.append(f"date >= '{start_date.date()}'")
            if end_date:
                conditions.append(f"date <= '{end_date.date()}'")
                
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
            SELECT 
                date, total_value, cash_balance, positions_value,
                daily_return, cumulative_return, daily_volatility,
                sharpe_ratio, max_drawdown, current_drawdown,
                trades_today, win_rate, avg_win, avg_loss,
                profit_factor, factor_ic_mean, factor_ic_std,
                top_performing_factors, prediction_accuracy,
                model_confidence
            FROM strategy_performance
            {where_clause}
            ORDER BY date DESC
            {limit_clause}
            """
            
            return self.client.query_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            raise
            
    def get_factor_statistics(self, factor_names: Union[str, List[str]],
                            lookback_days: int = 30) -> pd.DataFrame:
        """Get factor statistics over specified period"""
        try:
            if isinstance(factor_names, str):
                factor_names = [factor_names]
                
            factor_names_str = "', '".join(factor_names)
            
            query = f"""
            SELECT 
                factor_name,
                count() as observations,
                avg(factor_value) as mean_value,
                stddevPop(factor_value) as std_value,
                min(factor_value) as min_value,
                max(factor_value) as max_value,
                quantile(0.25)(factor_value) as q25,
                quantile(0.5)(factor_value) as median,
                quantile(0.75)(factor_value) as q75,
                avg(quality_score) as avg_quality
            FROM factors
            WHERE factor_name IN ('{factor_names_str}')
            AND date >= today() - {lookback_days}
            AND is_valid = 1
            GROUP BY factor_name
            ORDER BY factor_name
            """
            
            return self.client.query_dataframe(query)
        except Exception as e:
            logger.error(f"Failed to get factor statistics: {e}")
            raise
            
    def get_symbol_universe(self, active_only: bool = True) -> List[str]:
        """Get list of available symbols"""
        try:
            active_filter = "AND date >= today() - 7" if active_only else ""
            
            query = f"""
            SELECT DISTINCT symbol
            FROM factors
            WHERE is_valid = 1
            {active_filter}
            ORDER BY symbol
            """
            
            result = self.client.execute(query)
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to get symbol universe: {e}")
            raise
            
    def get_data_quality_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get data quality report for specified date range"""
        try:
            tables = ['tick_data', 'order_book', 'factors', 'predictions']
            report = {}
            
            for table in tables:
                date_col = 'timestamp' if table in ['tick_data', 'order_book'] else 'date'
                
                query = f"""
                SELECT 
                    count() as total_records,
                    count(DISTINCT symbol) as unique_symbols,
                    min({date_col}) as earliest_date,
                    max({date_col}) as latest_date
                FROM {table}
                WHERE {date_col} BETWEEN '{start_date}' AND '{end_date}'
                """
                
                result = self.client.execute(query)
                if result:
                    report[table] = {
                        'total_records': result[0][0],
                        'unique_symbols': result[0][1],
                        'earliest_date': result[0][2],
                        'latest_date': result[0][3]
                    }
                    
            return report
        except Exception as e:
            logger.error(f"Failed to get data quality report: {e}")
            raise