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
from data.storage.data_reader import DataReader
from data.ingestion.data_collector import DataCollector
from data.ingestion.data_preprocessor import DataPreprocessor
import yaml

# Setup enhanced logging with line numbers and thread ID
import threading
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Thread-%(thread)d] [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Main data ingestion pipeline"""
    
    def __init__(self, config_path: str = "config/database.yaml",
                 data_providers_config: str = "config/data_providers.yaml"):
        """Initialize the ingestion pipeline"""
        self.client = ClickHouseClient(config_path)
        self.writer = DataWriter(self.client)
        self.reader = DataReader(self.client)
        self.collector = DataCollector(data_providers_config)
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
            
        # Handle "ALL" symbols parameter
        if len(symbols) == 1 and symbols[0].upper() == "ALL":
            logger.info("Fetching all available symbols from database...")
            symbols = await self._get_all_symbols()
            if not symbols:
                logger.info("No symbols found in database, using default symbol list...")
                symbols = self._get_default_symbol_list()
        
        logger.info(f"Starting data ingestion for {len(symbols)} symbols")
        logger.info(f"Symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
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
            
    async def _get_all_symbols(self) -> List[str]:
        """Get all available symbols from data providers via API"""
        try:
            # First try to get symbols from data providers (API)
            logger.info("Fetching available symbols from data providers...")
            symbols = await self.collector.get_available_symbols()
            
            if symbols:
                logger.info(f"Found {len(symbols)} symbols from data providers")
                return symbols
            else:
                # Fallback to database if no symbols from providers
                logger.info("No symbols from providers, checking database...")
                db_symbols = self.reader.get_symbol_universe(active_only=True)
                if db_symbols:
                    logger.info(f"Found {len(db_symbols)} symbols in database")
                    return db_symbols
                else:
                    logger.info("No symbols found in database either")
                    return []
        except Exception as e:
            logger.warning(f"Failed to get symbols: {e}")
            return []
            
    def _get_default_symbol_list(self) -> List[str]:
        """Get default comprehensive symbol list for Chinese stock market"""
        # Comprehensive list of major Chinese stocks
        default_symbols = [
            # Major indices components and blue chips
            '000001.SZ', '000002.SZ', '000858.SZ', '000725.SZ', '000776.SZ',
            '600000.SH', '600036.SH', '600519.SH', '600887.SH', '600276.SH',
            '000858.SZ', '002415.SZ', '300059.SZ', '300750.SZ', '002594.SZ',
            
            # Technology stocks
            '000063.SZ', '002230.SZ', '002241.SZ', '300014.SZ', '300033.SZ',
            '300122.SZ', '300124.SZ', '300136.SZ', '300408.SZ', '300433.SZ',
            '300496.SZ', '300628.SZ', '300661.SZ', '300676.SZ', '300750.SZ',
            
            # Financial sector
            '000001.SZ', '600000.SH', '600015.SH', '600016.SH', '600036.SH',
            '600837.SH', '600886.SH', '601009.SH', '601166.SH', '601169.SH',
            '601288.SH', '601318.SH', '601328.SH', '601398.SH', '601601.SH',
            '601628.SH', '601668.SH', '601688.SH', '601818.SH', '601988.SH',
            
            # Consumer goods
            '000568.SZ', '000596.SZ', '000858.SZ', '000895.SZ', '002304.SZ',
            '600519.SH', '600887.SH', '600999.SH', '603288.SH', '603369.SH',
            
            # Healthcare & Pharma
            '000661.SZ', '000963.SZ', '002007.SZ', '002022.SZ', '002252.SZ',
            '002422.SZ', '002821.SZ', '300003.SZ', '300015.SZ', '300142.SZ',
            '300347.SZ', '300601.SZ', '600085.SH', '600196.SH', '600276.SH',
            
            # Energy & Materials
            '000983.SZ', '002466.SZ', '600028.SH', '600188.SH', '600309.SH',
            '600346.SH', '600362.SH', '600438.SH', '600547.SH', '600585.SH',
            '601088.SH', '601225.SH', '601857.SH', '601899.SH', '603993.SH',
            
            # Industrial & Manufacturing
            '000157.SZ', '000338.SZ', '000625.SZ', '000651.SZ', '000768.SZ',
            '002008.SZ', '002129.SZ', '002202.SZ', '002271.SZ', '002415.SZ',
            '600031.SH', '600104.SH', '600150.SH', '600406.SH', '600482.SH',
            '600585.SH', '600690.SH', '600741.SH', '601766.SH', '601989.SH',
            
            # Real Estate
            '000002.SZ', '000069.SZ', '000656.SZ', '000718.SZ', '001979.SZ',
            '600048.SH', '600340.SH', '600383.SH', '600606.SH', '600649.SH',
            
            # Transportation & Logistics
            '000089.SZ', '000099.SZ', '600004.SH', '600026.SH', '600115.SH',
            '600125.SH', '600320.SH', '600350.SH', '600377.SH', '601006.SH',
            '601111.SH', '601333.SH', '601919.SH', '603885.SH',
            
            # Utilities
            '000027.SZ', '000539.SZ', '000767.SZ', '600886.SH', '600900.SH',
            '601985.SH', '600795.SH', '600011.SH', '600642.SH'
        ]
        
        # Remove duplicates and sort
        unique_symbols = sorted(list(set(default_symbols)))
        logger.info(f"Using default symbol list with {len(unique_symbols)} symbols")
        return unique_symbols


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run data ingestion pipeline')
    parser.add_argument('--symbols', nargs='+', default=['000001.SZ', '000002.SZ'],
                       help='Stock symbols to ingest. Use "ALL" to ingest all available symbols from database or default comprehensive list')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-types', nargs='+', 
                       choices=['tick_data', 'order_book', 'factors'],
                       default=['tick_data', 'order_book', 'factors'],
                       help='Types of data to ingest')
    parser.add_argument('--config', default='config/database.yaml',
                       help='Database configuration file')
    parser.add_argument('--providers-config', default='config/data_providers.yaml',
                       help='Data providers configuration file')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
    # Create and run pipeline
    pipeline = DataIngestionPipeline(args.config, args.providers_config)
    
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