#!/usr/bin/env python
"""
Model Training Module
Main entry point for training machine learning models
"""

import logging
import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.storage.clickhouse_client import ClickHouseClient
from data.storage.data_reader import DataReader
from models.deep_learning.cnn_model import create_cnn_model, CNNTrainer
from models.ensemble.ensemble_model import EnsembleModel, StackingEnsemble

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Main model training pipeline"""
    
    def __init__(self, config_path: str = "config/database.yaml"):
        """Initialize the training pipeline"""
        self.client = ClickHouseClient(config_path)
        self.reader = DataReader(self.client)
        
    async def run_training(self, symbols: List[str],
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          model_types: List[str] = None) -> None:
        """Run the complete model training pipeline"""
        
        if model_types is None:
            model_types = ['cnn', 'ensemble']
            
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now() - timedelta(days=1)
            
        # Handle "ALL" symbols parameter
        if len(symbols) == 1 and symbols[0].upper() == "ALL":
            logger.info("Fetching all available symbols from database...")
            symbols = self.reader.get_symbol_universe(active_only=True)
            if not symbols:
                logger.info("No symbols found in database, using default symbol list...")
                symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '600519.SH']
            
        logger.info(f"Starting model training for {len(symbols)} symbols")
        logger.info(f"Symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Model types: {model_types}")
        
        try:
            # Prepare training data
            training_data = await self._prepare_training_data(symbols, start_date, end_date)
            
            if training_data.empty:
                logger.error("No training data available")
                return
                
            logger.info(f"Prepared {len(training_data)} training samples")
            
            # Train models
            for model_type in model_types:
                logger.info(f"Training {model_type} model")
                
                if model_type == 'cnn':
                    await self._train_cnn_model(training_data)
                elif model_type == 'ensemble':
                    await self._train_ensemble_model(training_data)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
            
    async def _prepare_training_data(self, symbols: List[str], 
                                   start_date: datetime, 
                                   end_date: datetime) -> pd.DataFrame:
        """Prepare training data from factors and price data"""
        try:
            logger.info("Preparing training data...")
            
            all_data = []
            
            for symbol in symbols:
                # Get factor data
                factors = self.reader.get_factors([symbol], start_date=start_date, end_date=end_date)
                
                if factors.empty:
                    logger.warning(f"No factor data for {symbol}")
                    continue
                    
                # Get price data for labels
                tick_data = self.reader.get_tick_data([symbol], start_date, end_date, limit=10000)
                
                if tick_data.empty:
                    logger.warning(f"No tick data for {symbol}")
                    continue
                    
                # Prepare features and labels
                symbol_data = self._create_features_and_labels(factors, tick_data, symbol)
                
                if not symbol_data.empty:
                    all_data.append(symbol_data)
                    
            if all_data:
                training_data = pd.concat(all_data, ignore_index=True)
                return training_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
            
    def _create_features_and_labels(self, factors: pd.DataFrame, 
                                  tick_data: pd.DataFrame, 
                                  symbol: str) -> pd.DataFrame:
        """Create features and labels for training"""
        try:
            # Pivot factors to create feature matrix
            factor_pivot = factors.pivot_table(
                index=['date', 'symbol'], 
                columns='factor_name', 
                values='factor_value'
            ).reset_index()
            
            # Calculate future returns as labels
            tick_data = tick_data.sort_values('timestamp')
            tick_data['future_return_1d'] = tick_data['price'].pct_change(periods=1).shift(-1)
            tick_data['future_return_3d'] = tick_data['price'].pct_change(periods=3).shift(-3)
            
            # Create binary classification labels (1 if return > 0, 0 otherwise)
            tick_data['label_1d'] = (tick_data['future_return_1d'] > 0).astype(int)
            tick_data['label_3d'] = (tick_data['future_return_3d'] > 0).astype(int)
            
            # Aggregate tick data by date
            daily_prices = tick_data.groupby(tick_data['timestamp'].dt.date).agg({
                'price': 'last',
                'volume': 'sum',
                'future_return_1d': 'last',
                'future_return_3d': 'last',
                'label_1d': 'last',
                'label_3d': 'last'
            }).reset_index()
            daily_prices.rename(columns={'timestamp': 'date'}, inplace=True)
            daily_prices['symbol'] = symbol
            
            # Merge factors with price data
            training_data = pd.merge(factor_pivot, daily_prices, on=['date', 'symbol'], how='inner')
            
            # Remove rows with missing labels
            training_data = training_data.dropna(subset=['label_1d', 'label_3d'])
            
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to create features and labels for {symbol}: {e}")
            return pd.DataFrame()
            
    async def _train_cnn_model(self, training_data: pd.DataFrame):
        """Train CNN model"""
        try:
            logger.info("Training CNN model...")
            
            # Prepare data for CNN
            feature_cols = [col for col in training_data.columns 
                          if col not in ['date', 'symbol', 'price', 'volume', 
                                       'future_return_1d', 'future_return_3d', 
                                       'label_1d', 'label_3d']]
            
            X = training_data[feature_cols].fillna(0).values
            y = training_data['label_1d'].values
            
            if len(X) < 100:
                logger.warning("Insufficient data for CNN training")
                return
                
            # Reshape for CNN (samples, timesteps, features)
            # For simplicity, we'll use a 1D CNN
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Create and train CNN model
            model = create_cnn_model(input_shape=(X.shape[1], 1), num_classes=2)
            trainer = CNNTrainer(model)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            history = trainer.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
            
            # Save model
            model_path = Path("models/saved/cnn_model.pth")
            model_path.parent.mkdir(exist_ok=True)
            trainer.save_model(str(model_path))
            
            logger.info(f"CNN model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to train CNN model: {e}")
            raise
            
    async def _train_ensemble_model(self, training_data: pd.DataFrame):
        """Train ensemble model"""
        try:
            logger.info("Training ensemble model...")
            
            # Prepare data for ensemble
            feature_cols = [col for col in training_data.columns 
                          if col not in ['date', 'symbol', 'price', 'volume', 
                                       'future_return_1d', 'future_return_3d', 
                                       'label_1d', 'label_3d']]
            
            X = training_data[feature_cols].fillna(0).values
            y = training_data['label_1d'].values
            
            if len(X) < 100:
                logger.warning("Insufficient data for ensemble training")
                return
                
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create and train ensemble model
            ensemble = StackingEnsemble()
            ensemble.fit(X_train, y_train)
            
            # Evaluate
            train_score = ensemble.score(X_train, y_train)
            val_score = ensemble.score(X_val, y_val)
            
            logger.info(f"Ensemble model - Train score: {train_score:.4f}, Val score: {val_score:.4f}")
            
            # Save model
            model_path = Path("models/saved/ensemble_model.pkl")
            model_path.parent.mkdir(exist_ok=True)
            ensemble.save_model(str(model_path))
            
            logger.info(f"Ensemble model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to train ensemble model: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run model training pipeline')
    parser.add_argument('--symbols', nargs='+', default=['000001.SZ', '000002.SZ'],
                       help='Stock symbols to train on. Use "ALL" to train on all available symbols from database')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-types', nargs='+', 
                       choices=['cnn', 'ensemble'],
                       default=['cnn', 'ensemble'],
                       help='Types of models to train')
    parser.add_argument('--config', default='config/database.yaml',
                       help='Database configuration file')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
    # Create and run pipeline
    pipeline = ModelTrainingPipeline(args.config)
    
    try:
        asyncio.run(pipeline.run_training(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            model_types=args.model_types
        ))
        logger.info("Model training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Model training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()