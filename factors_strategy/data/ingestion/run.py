#!/usr/bin/env python
"""
Data Ingestion Runner
Main entry point for data collection and ingestion pipeline
"""

import logging
import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.storage.clickhouse_client import ClickHouseClient
from data.storage.data_writer import DataWriter
from data.ingestion.data_collector import DataCollector
from data.ingestion.data_preprocessor import DataPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Main data ingestion pipeline"""
    
    def __init__(self, config_path: str = "config/database.yaml"):
        """Initialize the ingestion pipeline"""
        self.client = ClickHouseClient(config_path)
        self.writer = DataWriter(self.client)
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        
    async def run_ingestion(self, symbols: List[str], 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          data_types: List[str] = None) -> None:
        """Run the complete data ingestion pipeline"""
        
        if data_types is None:
            data_types = ['tick_data', 'order_book', 'factors']
            
        if start_date is None:
            start_date = datetime.now() - timedelta(days=1)
        if end_date is None:
            end_date = datetime.now()
            
        logger.info(f"Starting data ingestion for {len(symbols)} symbols")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Data types: {data_types}")
        
        try:
            for symbol in symbols:
                logger.info(f"Processing symbol: {symbol}")
                
                # Collect tick data
                if 'tick_data' in data_types:
                    await self._ingest_tick_data(symbol, start_date, end_date)
                    
                # Collect order book data
                if 'order_book' in data_types:
                    await self._ingest_order_book_data(symbol, start_date, end_date)
                    
                # Calculate and store factors
                if 'factors' in data_types:
                    await self._ingest_factors(symbol, start_date, end_date)
                    
            logger.info("Data ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
            
    async def _ingest_tick_data(self, symbol: str, start_date: datetime, end_date: datetime):
        """Ingest tick data for a symbol"""
        try:
            logger.info(f"Collecting tick data for {symbol}")
            
            # Collect raw tick data
            raw_data = await self.collector.collect_tick_data(symbol, start_date, end_date)
            
            if raw_data.empty:
                logger.warning(f"No tick data found for {symbol}")
                return
                
            # Preprocess the data
            processed_data = self.preprocessor.process_tick_data(raw_data)
            
            # Write to database
            self.writer.write_tick_data(processed_data)
            
            logger.info(f"Ingested {len(processed_data)} tick records for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to ingest tick data for {symbol}: {e}")
            raise
            
    async def _ingest_order_book_data(self, symbol: str, start_date: datetime, end_date: datetime):
        """Ingest order book data for a symbol"""
        try:
            logger.info(f"Collecting order book data for {symbol}")
            
            # Collect raw order book data
            raw_data = await self.collector.collect_order_book_data(symbol, start_date, end_date)
            
            if raw_data.empty:
                logger.warning(f"No order book data found for {symbol}")
                return
                
            # Preprocess the data
            processed_data = self.preprocessor.process_order_book_data(raw_data)
            
            # Write to database
            self.writer.write_order_book(processed_data)
            
            logger.info(f"Ingested {len(processed_data)} order book records for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to ingest order book data for {symbol}: {e}")
            raise
            
    async def _ingest_factors(self, symbol: str, start_date: datetime, end_date: datetime):
        """Calculate and ingest factors for a symbol"""
        try:
            logger.info(f"Calculating factors for {symbol}")
            
            # Get tick data for factor calculation
            from data.storage.data_reader import DataReader
            reader = DataReader(self.client)
            
            tick_data = reader.get_tick_data([symbol], start_date, end_date)
            
            if tick_data.empty:
                logger.warning(f"No tick data available for factor calculation: {symbol}")
                return
                
            # Calculate factors
            factors = self.preprocessor.calculate_factors(tick_data, symbol)
            
            if factors.empty:
                logger.warning(f"No factors calculated for {symbol}")
                return
                
            # Write factors to database
            self.writer.write_factors(factors)
            
            logger.info(f"Ingested {len(factors)} factor records for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to ingest factors for {symbol}: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run data ingestion pipeline')
    parser.add_argument('--symbols', nargs='+', default=['000001.SZ', '000002.SZ'], 
                       help='Stock symbols to ingest')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-types', nargs='+', 
                       choices=['tick_data', 'order_book', 'factors'],
                       default=['tick_data', 'order_book', 'factors'],
                       help='Types of data to ingest')
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
    pipeline = DataIngestionPipeline(args.config)
    
    try:
        asyncio.run(pipeline.run_ingestion(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=args.data_types
        ))
        logger.info("Data ingestion pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()