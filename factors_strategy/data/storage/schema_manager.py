"""
Schema Manager for ClickHouse
Handles table creation and schema management
"""

import logging
from typing import Dict, Any
from pathlib import Path
import yaml
from .clickhouse_client import ClickHouseClient

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manages database schemas and table creation"""
    
    def __init__(self, client: ClickHouseClient, config_path: str = "config/database.yaml"):
        """Initialize schema manager"""
        self.client = client
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load schema configuration"""
        config_file = Path(config_path)
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config['schemas']
        
    def create_all_tables(self):
        """Create all tables defined in configuration"""
        logger.info("Creating all tables...")
        
        # Create tick data table
        self.create_tick_data_table()
        
        # Create order book table
        self.create_order_book_table()
        
        # Create factors table
        self.create_factors_table()
        
        # Create predictions table
        self.create_predictions_table()
        
        # Create additional tables
        self.create_trade_signals_table()
        self.create_portfolio_table()
        self.create_performance_table()
        
        logger.info("All tables created successfully")
        
    def create_tick_data_table(self):
        """Create tick data table"""
        config = self.config['tick_data']
        
        query = f"""
        CREATE TABLE IF NOT EXISTS {config['table_name']}
        (
            timestamp DateTime64(3),
            symbol String,
            price Float64,
            volume Float64,
            turnover Float64,
            bid_price Array(Float64),
            bid_volume Array(Float64),
            ask_price Array(Float64),
            ask_volume Array(Float64),
            trade_direction Int8,
            trade_type String,
            
            -- Calculated fields
            mid_price Float64 MATERIALIZED (bid_price[1] + ask_price[1]) / 2,
            spread Float64 MATERIALIZED ask_price[1] - bid_price[1],
            
            -- Metadata
            exchange String,
            update_time DateTime DEFAULT now()
        )
        ENGINE = {config['engine']}()
        PARTITION BY {config['partition_by']}
        ORDER BY {config['order_by']}
        TTL {config['ttl']}
        SETTINGS index_granularity = 8192
        """
        
        self.client.execute(query)
        logger.info(f"Created table: {config['table_name']}")
        
    def create_order_book_table(self):
        """Create order book snapshot table"""
        config = self.config['order_book']
        
        query = f"""
        CREATE TABLE IF NOT EXISTS {config['table_name']}
        (
            timestamp DateTime64(3),
            symbol String,
            
            -- Bid side (up to 10 levels)
            bid_price_1 Float64,
            bid_volume_1 Float64,
            bid_price_2 Float64,
            bid_volume_2 Float64,
            bid_price_3 Float64,
            bid_volume_3 Float64,
            bid_price_4 Float64,
            bid_volume_4 Float64,
            bid_price_5 Float64,
            bid_volume_5 Float64,
            bid_price_6 Float64,
            bid_volume_6 Float64,
            bid_price_7 Float64,
            bid_volume_7 Float64,
            bid_price_8 Float64,
            bid_volume_8 Float64,
            bid_price_9 Float64,
            bid_volume_9 Float64,
            bid_price_10 Float64,
            bid_volume_10 Float64,
            
            -- Ask side (up to 10 levels)
            ask_price_1 Float64,
            ask_volume_1 Float64,
            ask_price_2 Float64,
            ask_volume_2 Float64,
            ask_price_3 Float64,
            ask_volume_3 Float64,
            ask_price_4 Float64,
            ask_volume_4 Float64,
            ask_price_5 Float64,
            ask_volume_5 Float64,
            ask_price_6 Float64,
            ask_volume_6 Float64,
            ask_price_7 Float64,
            ask_volume_7 Float64,
            ask_price_8 Float64,
            ask_volume_8 Float64,
            ask_price_9 Float64,
            ask_volume_9 Float64,
            ask_price_10 Float64,
            ask_volume_10 Float64,
            
            -- Aggregated metrics
            total_bid_volume Float64,
            total_ask_volume Float64,
            bid_ask_imbalance Float64,
            
            -- Metadata
            exchange String,
            update_time DateTime DEFAULT now()
        )
        ENGINE = {config['engine']}()
        PARTITION BY {config['partition_by']}
        ORDER BY {config['order_by']}
        SETTINGS index_granularity = 8192
        """
        
        self.client.execute(query)
        logger.info(f"Created table: {config['table_name']}")
        
    def create_factors_table(self):
        """Create factors table"""
        config = self.config['factors']
        
        query = f"""
        CREATE TABLE IF NOT EXISTS {config['table_name']}
        (
            date Date,
            timestamp DateTime,
            symbol String,
            factor_name String,
            factor_value Float64,
            factor_category String,
            
            -- Factor metadata
            window_size Int32,
            calculation_time Float32,
            version String,
            
            -- Quality metrics
            is_valid UInt8 DEFAULT 1,
            quality_score Float32,
            
            update_time DateTime DEFAULT now()
        )
        ENGINE = {config['engine']}()
        PARTITION BY {config['partition_by']}
        ORDER BY {config['order_by']}
        SETTINGS index_granularity = 8192
        """
        
        self.client.execute(query)
        logger.info(f"Created table: {config['table_name']}")
        
    def create_predictions_table(self):
        """Create predictions table"""
        config = self.config['predictions']
        
        query = f"""
        CREATE TABLE IF NOT EXISTS {config['table_name']}
        (
            prediction_date Date,
            prediction_time DateTime,
            symbol String,
            
            -- Prediction outputs
            probability_10pct_1d Float64,
            probability_10pct_3d Float64,
            expected_return_1d Float64,
            expected_return_3d Float64,
            
            -- Model scores
            cnn_score Float64,
            lstm_score Float64,
            transformer_score Float64,
            xgboost_score Float64,
            lightgbm_score Float64,
            ensemble_score Float64,
            
            -- Ranking
            rank_score Float64,
            rank_position Int32,
            
            -- Confidence metrics
            prediction_confidence Float64,
            model_agreement Float64,
            
            -- Feature importance
            top_factors Array(String),
            factor_contributions Array(Float64),
            
            -- Metadata
            model_version String,
            is_selected UInt8 DEFAULT 0,
            selection_reason String,
            
            update_time DateTime DEFAULT now()
        )
        ENGINE = {config['engine']}()
        PARTITION BY {config['partition_by']}
        ORDER BY {config['order_by']}
        SETTINGS index_granularity = 8192
        """
        
        self.client.execute(query)
        logger.info(f"Created table: {config['table_name']}")
        
    def create_trade_signals_table(self):
        """Create trade signals table"""
        query = """
        CREATE TABLE IF NOT EXISTS trade_signals
        (
            signal_date Date,
            signal_time DateTime,
            symbol String,
            
            -- Signal details
            signal_type String,  -- 'BUY', 'SELL', 'HOLD'
            signal_strength Float64,
            
            -- Entry/Exit prices
            entry_price Float64,
            target_price Float64,
            stop_loss_price Float64,
            
            -- Position sizing
            suggested_position_size Float64,
            max_position_value Float64,
            
            -- Risk metrics
            expected_return Float64,
            expected_risk Float64,
            sharpe_ratio Float64,
            
            -- Status tracking
            is_active UInt8 DEFAULT 1,
            execution_status String,
            actual_entry_price Float64,
            actual_exit_price Float64,
            actual_return Float64,
            
            update_time DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(signal_date)
        ORDER BY (signal_date, symbol)
        """
        
        self.client.execute(query)
        logger.info("Created table: trade_signals")
        
    def create_portfolio_table(self):
        """Create portfolio positions table"""
        query = """
        CREATE TABLE IF NOT EXISTS portfolio_positions
        (
            date Date,
            timestamp DateTime,
            symbol String,
            
            -- Position details
            quantity Float64,
            avg_price Float64,
            current_price Float64,
            market_value Float64,
            
            -- P&L
            unrealized_pnl Float64,
            realized_pnl Float64,
            total_pnl Float64,
            pnl_percentage Float64,
            
            -- Risk metrics
            position_weight Float64,
            sector String,
            beta Float64,
            correlation_to_market Float64,
            
            -- Holding period
            entry_date DateTime,
            days_held Int32,
            
            update_time DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (date, symbol)
        """
        
        self.client.execute(query)
        logger.info("Created table: portfolio_positions")
        
    def create_performance_table(self):
        """Create strategy performance tracking table"""
        query = """
        CREATE TABLE IF NOT EXISTS strategy_performance
        (
            date Date,
            timestamp DateTime,
            
            -- Portfolio metrics
            total_value Float64,
            cash_balance Float64,
            positions_value Float64,
            
            -- Returns
            daily_return Float64,
            cumulative_return Float64,
            
            -- Risk metrics
            daily_volatility Float64,
            sharpe_ratio Float64,
            max_drawdown Float64,
            current_drawdown Float64,
            
            -- Trading metrics
            trades_today Int32,
            win_rate Float64,
            avg_win Float64,
            avg_loss Float64,
            profit_factor Float64,
            
            -- Factor performance
            factor_ic_mean Float64,
            factor_ic_std Float64,
            top_performing_factors Array(String),
            
            -- Model performance
            prediction_accuracy Float64,
            model_confidence Float64,
            
            update_time DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY date
        """
        
        self.client.execute(query)
        logger.info("Created table: strategy_performance")
        
    def create_indexes(self):
        """Create additional indexes for query optimization"""
        indexes = [
            "ALTER TABLE tick_data ADD INDEX idx_symbol_time (symbol, timestamp) TYPE minmax GRANULARITY 4",
            "ALTER TABLE factors ADD INDEX idx_factor_symbol (factor_name, symbol) TYPE bloom_filter GRANULARITY 1",
            "ALTER TABLE predictions ADD INDEX idx_rank (rank_position) TYPE minmax GRANULARITY 1",
        ]
        
        for index_query in indexes:
            try:
                self.client.execute(index_query)
                logger.info(f"Created index: {index_query.split('ADD INDEX')[1].split('(')[0].strip()}")
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")