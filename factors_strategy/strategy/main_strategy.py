"""
Main Strategy Execution Module
Orchestrates the multi-factor AI stock selection strategy
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
import asyncio

# Import custom modules
from data.storage.clickhouse_client import ClickHouseClient
from data.storage.schema_manager import SchemaManager
from factors.traditional.microstructure_factors import MicrostructureFactors
from factors.ai_generated.llm_factor_generator import LLMFactorGenerator
from models.deep_learning.cnn_model import create_cnn_model, CNNTrainer
from models.ensemble.ensemble_model import EnsembleModel, StackingEnsemble

logger = logging.getLogger(__name__)


class StockSelectionStrategy:
    """Main strategy class for AI-powered stock selection"""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize strategy with configuration"""
        self.config_dir = Path(config_dir)
        self.config = self._load_all_configs()
        
        # Initialize components
        self.db_client = None
        self.factor_calculator = None
        self.llm_generator = None
        self.ensemble_model = None
        self.cnn_model = None
        
        # Strategy state
        self.current_positions = {}
        self.daily_recommendations = []
        self.performance_metrics = {}
        
    def _load_all_configs(self) -> Dict:
        """Load all configuration files"""
        configs = {}
        
        config_files = ['database.yaml', 'model.yaml', 'factors.yaml', 'strategy.yaml']
        for config_file in config_files:
            config_path = self.config_dir / config_file
            with open(config_path, 'r') as f:
                config_name = config_file.replace('.yaml', '')
                configs[config_name] = yaml.safe_load(f)
                
        return configs
        
    def initialize(self):
        """Initialize all strategy components"""
        logger.info("Initializing stock selection strategy...")
        
        # Initialize database
        self._initialize_database()
        
        # Initialize factor calculators
        self._initialize_factors()
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Strategy initialization complete")
        
    def _initialize_database(self):
        """Initialize database connection and schema"""
        logger.info("Initializing database...")
        
        # Create database client
        self.db_client = ClickHouseClient(self.config_dir / "database.yaml")
        
        # Create schema if needed
        schema_manager = SchemaManager(self.db_client, self.config_dir / "database.yaml")
        schema_manager.create_all_tables()
        schema_manager.create_indexes()
        
    def _initialize_factors(self):
        """Initialize factor calculation modules"""
        logger.info("Initializing factor modules...")
        
        # Traditional factors
        self.factor_calculator = MicrostructureFactors(
            self.config['factors']['factors']['microstructure']
        )
        
        # LLM factor generator
        self.llm_generator = LLMFactorGenerator(self.config_dir / "strategy.yaml")
        
    def _initialize_models(self):
        """Initialize ML models"""
        logger.info("Initializing models...")
        
        # Ensemble model
        self.ensemble_model = StackingEnsemble(self.config_dir / "model.yaml")
        
        # CNN model
        self.cnn_model = create_cnn_model(self.config['model'])
        
    async def run_daily_strategy(self, date: datetime) -> List[Dict]:
        """Run the complete daily strategy workflow"""
        logger.info(f"Running strategy for date: {date}")
        
        try:
            # Step 1: Load market data
            tick_data, order_book_data = await self._load_market_data(date)
            
            # Step 2: Calculate traditional factors
            traditional_factors = self._calculate_traditional_factors(
                tick_data, order_book_data
            )
            
            # Step 3: Generate AI factors
            ai_factors = await self._generate_ai_factors(
                tick_data, traditional_factors
            )
            
            # Step 4: Combine all factors
            all_factors = self._combine_factors(traditional_factors, ai_factors)
            
            # Step 5: Generate predictions
            predictions = self._generate_predictions(all_factors)
            
            # Step 6: Select top stocks
            recommendations = self._select_top_stocks(predictions)
            
            # Step 7: Apply risk management
            final_recommendations = self._apply_risk_management(recommendations)
            
            # Step 8: Save results
            await self._save_results(date, final_recommendations)
            
            # Step 9: Update performance metrics
            self._update_performance_metrics(date, final_recommendations)
            
            self.daily_recommendations = final_recommendations
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            raise
            
    async def _load_market_data(self, date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load tick and order book data for the trading day"""
        logger.info(f"Loading market data for {date}")
        
        # Define time range
        start_time = date.replace(hour=9, minute=30, second=0)
        end_time = date.replace(hour=15, minute=0, second=0)
        
        # Load tick data
        tick_query = f"""
        SELECT 
            timestamp,
            symbol,
            price,
            volume,
            turnover,
            bid_price,
            bid_volume,
            ask_price,
            ask_volume,
            trade_direction,
            trade_type
        FROM tick_data
        WHERE timestamp >= '{start_time}'
          AND timestamp <= '{end_time}'
          AND symbol IN (
              SELECT DISTINCT symbol 
              FROM tick_data 
              WHERE date = '{date.date()}'
              GROUP BY symbol
              HAVING sum(volume) > {self.config['factors']['universe']['min_daily_volume']}
          )
        ORDER BY symbol, timestamp
        """
        
        tick_data = self.db_client.query_dataframe(tick_query)
        
        # Load order book data
        orderbook_query = f"""
        SELECT *
        FROM order_book
        WHERE timestamp >= '{start_time}'
          AND timestamp <= '{end_time}'
          AND symbol IN (
              SELECT DISTINCT symbol FROM ({tick_query})
          )
        ORDER BY symbol, timestamp
        """
        
        order_book_data = self.db_client.query_dataframe(orderbook_query)
        
        logger.info(f"Loaded {len(tick_data)} tick records and {len(order_book_data)} order book records")
        
        return tick_data, order_book_data
        
    def _calculate_traditional_factors(self, 
                                     tick_data: pd.DataFrame,
                                     order_book_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate traditional microstructure factors"""
        logger.info("Calculating traditional factors...")
        
        # Group by symbol for factor calculation
        factors_list = []
        
        for symbol in tick_data['symbol'].unique():
            symbol_ticks = tick_data[tick_data['symbol'] == symbol]
            symbol_orderbook = order_book_data[order_book_data['symbol'] == symbol]
            
            # Calculate factors
            symbol_factors = self.factor_calculator.calculate_all_factors(
                symbol_ticks, symbol_orderbook
            )
            symbol_factors['symbol'] = symbol
            
            factors_list.append(symbol_factors)
            
        # Combine all factors
        all_factors = pd.concat(factors_list, ignore_index=True)
        
        logger.info(f"Calculated {len(all_factors.columns)} traditional factors")
        
        return all_factors
        
    async def _generate_ai_factors(self, 
                                  tick_data: pd.DataFrame,
                                  traditional_factors: pd.DataFrame) -> pd.DataFrame:
        """Generate AI-enhanced factors using LLM"""
        logger.info("Generating AI factors...")
        
        # Prepare performance metrics for LLM
        performance_metrics = {
            'factor_ic': self._calculate_factor_ic(traditional_factors),
            'factor_ic_trend': self._calculate_ic_trends(traditional_factors),
            'factor_decay_rates': self._calculate_factor_decay()
        }
        
        # Generate new factors
        new_factors = await self.llm_generator.generate_factors(
            tick_data,
            traditional_factors,
            performance_metrics
        )
        
        # Calculate values for new factors
        ai_factors_df = self._calculate_ai_factor_values(new_factors, tick_data)
        
        logger.info(f"Generated {len(new_factors)} AI factors")
        
        return ai_factors_df
        
    def _calculate_factor_ic(self, factors: pd.DataFrame) -> Dict[str, float]:
        """Calculate Information Coefficient for factors"""
        # Simplified IC calculation
        # In production, would calculate rank correlation with forward returns
        ic_values = {}
        
        for col in factors.columns:
            if col != 'symbol':
                # Placeholder IC value
                ic_values[col] = np.random.uniform(0.01, 0.05)
                
        return ic_values
        
    def _calculate_ic_trends(self, factors: pd.DataFrame) -> Dict[str, float]:
        """Calculate IC trend for factors"""
        # Simplified trend calculation
        trends = {}
        
        for col in factors.columns:
            if col != 'symbol':
                # Placeholder trend value
                trends[col] = np.random.uniform(-0.02, 0.02)
                
        return trends
        
    def _calculate_factor_decay(self) -> Dict[str, List[float]]:
        """Calculate factor decay rates"""
        # Placeholder decay rates
        return {
            '1d': 0.9,
            '3d': 0.7,
            '5d': 0.5,
            '10d': 0.3
        }
        
    def _calculate_ai_factor_values(self, 
                                   generated_factors: List,
                                   tick_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate values for AI-generated factors"""
        # Placeholder implementation
        # In production, would parse and execute factor formulas
        
        ai_factors = pd.DataFrame()
        ai_factors['symbol'] = tick_data['symbol'].unique()
        
        for factor in generated_factors:
            # Simulate factor values
            ai_factors[factor.name] = np.random.randn(len(ai_factors))
            
        return ai_factors
        
    def _combine_factors(self, 
                        traditional_factors: pd.DataFrame,
                        ai_factors: pd.DataFrame) -> pd.DataFrame:
        """Combine traditional and AI factors"""
        logger.info("Combining all factors...")
        
        # Merge on symbol
        combined = pd.merge(
            traditional_factors,
            ai_factors,
            on='symbol',
            how='outer'
        )
        
        # Fill missing values
        combined = combined.fillna(0)
        
        # Normalize factors
        factor_cols = [col for col in combined.columns if col != 'symbol']
        for col in factor_cols:
            # Z-score normalization
            mean = combined[col].mean()
            std = combined[col].std()
            if std > 0:
                combined[col] = (combined[col] - mean) / std
                
        logger.info(f"Combined factors shape: {combined.shape}")
        
        return combined
        
    def _generate_predictions(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using ensemble model"""
        logger.info("Generating predictions...")
        
        # Prepare features
        feature_cols = [col for col in factors.columns if col != 'symbol']
        X = factors[feature_cols]
        
        # Get predictions from ensemble
        predictions = pd.DataFrame()
        predictions['symbol'] = factors['symbol']
        
        # Ensemble predictions
        predictions['ensemble_score'] = self.ensemble_model.predict_proba(X)
        
        # Individual model scores (if available)
        for model_name, model in self.ensemble_model.models.items():
            if hasattr(model, 'predict_proba'):
                predictions[f'{model_name}_score'] = model.predict_proba(X)[:, 1]
                
        # Calculate expected returns
        predictions['expected_return_1d'] = predictions['ensemble_score'] * 0.15  # Simplified
        predictions['expected_return_3d'] = predictions['ensemble_score'] * 0.25
        
        # Confidence metrics
        predictions['prediction_confidence'] = self._calculate_prediction_confidence(predictions)
        
        logger.info(f"Generated predictions for {len(predictions)} stocks")
        
        return predictions
        
    def _calculate_prediction_confidence(self, predictions: pd.DataFrame) -> pd.Series:
        """Calculate confidence score for predictions"""
        # Model agreement
        model_scores = [col for col in predictions.columns if col.endswith('_score')]
        
        if len(model_scores) > 1:
            # Standard deviation of model predictions
            model_std = predictions[model_scores].std(axis=1)
            # Lower std means higher agreement
            confidence = 1 - (model_std / model_std.max())
        else:
            # Single model confidence based on score extremity
            confidence = 2 * np.abs(predictions['ensemble_score'] - 0.5)
            
        return confidence
        
    def _select_top_stocks(self, predictions: pd.DataFrame) -> List[Dict]:
        """Select top stocks based on predictions"""
        logger.info("Selecting top stocks...")
        
        # Apply filters
        filtered = predictions.copy()
        
        # Minimum probability threshold
        min_prob = self.config['strategy']['strategy']['signals']['thresholds']['min_probability']
        filtered = filtered[filtered['ensemble_score'] >= min_prob]
        
        # Minimum expected return
        min_return = self.config['strategy']['strategy']['signals']['thresholds']['min_expected_return']
        filtered = filtered[filtered['expected_return_3d'] >= min_return]
        
        # Sort by ranking score
        filtered['rank_score'] = (
            filtered['ensemble_score'] * 0.4 +
            filtered['expected_return_3d'] * 0.3 +
            filtered['prediction_confidence'] * 0.3
        )
        
        filtered = filtered.sort_values('rank_score', ascending=False)
        
        # Select top N
        top_n = self.config['strategy']['strategy']['trading']['portfolio']['max_stocks']
        selected = filtered.head(top_n)
        
        # Convert to recommendations
        recommendations = []
        for idx, row in selected.iterrows():
            rec = {
                'symbol': row['symbol'],
                'rank_position': idx + 1,
                'ensemble_score': row['ensemble_score'],
                'expected_return_1d': row['expected_return_1d'],
                'expected_return_3d': row['expected_return_3d'],
                'prediction_confidence': row['prediction_confidence'],
                'rank_score': row['rank_score'],
                'position_size': self._calculate_position_size(row)
            }
            recommendations.append(rec)
            
        logger.info(f"Selected {len(recommendations)} stocks")
        
        return recommendations
        
    def _calculate_position_size(self, stock_data: pd.Series) -> float:
        """Calculate position size based on risk parity"""
        # Simplified position sizing
        # In production, would use volatility and correlation
        
        base_size = 1.0 / self.config['strategy']['strategy']['trading']['portfolio']['max_stocks']
        
        # Adjust by confidence
        size_adjustment = 0.5 + 0.5 * stock_data['prediction_confidence']
        
        position_size = base_size * size_adjustment
        
        # Apply limits
        max_size = self.config['strategy']['strategy']['trading']['position_sizing']['max_position_size']
        min_size = self.config['strategy']['strategy']['trading']['position_sizing']['min_position_size']
        
        return np.clip(position_size, min_size, max_size)
        
    def _apply_risk_management(self, recommendations: List[Dict]) -> List[Dict]:
        """Apply risk management rules to recommendations"""
        logger.info("Applying risk management...")
        
        # Check sector concentration
        # In production, would map symbols to sectors
        
        # Check correlation limits
        # In production, would calculate pairwise correlations
        
        # Apply position limits
        total_allocation = sum(rec['position_size'] for rec in recommendations)
        if total_allocation > 1.0:
            # Normalize positions
            for rec in recommendations:
                rec['position_size'] = rec['position_size'] / total_allocation
                
        return recommendations
        
    async def _save_results(self, date: datetime, recommendations: List[Dict]):
        """Save strategy results to database"""
        logger.info("Saving results to database...")
        
        # Prepare data for insertion
        predictions_data = []
        signals_data = []
        
        for rec in recommendations:
            # Predictions table
            pred_record = {
                'prediction_date': date.date(),
                'prediction_time': datetime.now(),
                'symbol': rec['symbol'],
                'probability_10pct_1d': rec['ensemble_score'],
                'probability_10pct_3d': rec['ensemble_score'],
                'expected_return_1d': rec['expected_return_1d'],
                'expected_return_3d': rec['expected_return_3d'],
                'ensemble_score': rec['ensemble_score'],
                'rank_score': rec['rank_score'],
                'rank_position': rec['rank_position'],
                'prediction_confidence': rec['prediction_confidence'],
                'model_version': '1.0.0',
                'is_selected': 1
            }
            predictions_data.append(pred_record)
            
            # Trade signals table
            signal_record = {
                'signal_date': date.date(),
                'signal_time': datetime.now(),
                'symbol': rec['symbol'],
                'signal_type': 'BUY',
                'signal_strength': rec['rank_score'],
                'suggested_position_size': rec['position_size'],
                'expected_return': rec['expected_return_3d'],
                'is_active': 1
            }
            signals_data.append(signal_record)
            
        # Insert into database
        if predictions_data:
            self.db_client.insert_dataframe(
                'predictions',
                pd.DataFrame(predictions_data)
            )
            
        if signals_data:
            self.db_client.insert_dataframe(
                'trade_signals',
                pd.DataFrame(signals_data)
            )
            
        logger.info(f"Saved {len(recommendations)} recommendations")
        
    def _update_performance_metrics(self, date: datetime, recommendations: List[Dict]):
        """Update strategy performance metrics"""
        # In production, would track actual returns and calculate metrics
        self.performance_metrics[date] = {
            'num_recommendations': len(recommendations),
            'avg_confidence': np.mean([r['prediction_confidence'] for r in recommendations]),
            'avg_expected_return': np.mean([r['expected_return_3d'] for r in recommendations])
        }
        
    def get_recommendations(self) -> List[Dict]:
        """Get current recommendations"""
        return self.daily_recommendations
        
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        return self.performance_metrics


async def main():
    """Main execution function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize strategy
    strategy = StockSelectionStrategy()
    strategy.initialize()
    
    # Run for today
    today = datetime.now()
    recommendations = await strategy.run_daily_strategy(today)
    
    # Display results
    logger.info("\n=== Today's Stock Recommendations ===")
    for rec in recommendations:
        logger.info(f"{rec['rank_position']}. {rec['symbol']} - "
                   f"Score: {rec['ensemble_score']:.3f}, "
                   f"Expected 3D Return: {rec['expected_return_3d']:.2%}, "
                   f"Position: {rec['position_size']:.2%}")
    
    return recommendations


if __name__ == "__main__":
    asyncio.run(main())