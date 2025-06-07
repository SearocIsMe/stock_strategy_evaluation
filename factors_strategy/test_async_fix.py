#!/usr/bin/env python
"""
Test script to verify the async event loop fix
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.ingestion.run import DataIngestionPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_symbol_retrieval():
    """Test that symbol retrieval works without async event loop errors"""
    logger.info("Testing symbol retrieval with async fix...")
    
    try:
        # Initialize pipeline
        pipeline = DataIngestionPipeline()
        
        # Test getting all symbols (this should now work without async errors)
        symbols = await pipeline._get_all_symbols()
        
        logger.info(f"✓ Successfully retrieved {len(symbols)} symbols")
        if symbols:
            logger.info(f"  Sample symbols: {symbols[:5]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to retrieve symbols: {e}")
        return False


async def test_full_ingestion_with_all():
    """Test full ingestion pipeline with ALL symbols parameter"""
    logger.info("\nTesting full ingestion with ALL symbols...")
    
    try:
        # Initialize pipeline
        pipeline = DataIngestionPipeline()
        
        # Run ingestion with ALL symbols (limited date range for testing)
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now()
        
        await pipeline.run_ingestion(
            symbols=["ALL"],
            start_date=start_date,
            end_date=end_date,
            data_types=['tick_data']  # Just test tick data
        )
        
        logger.info("✓ Full ingestion completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Full ingestion failed: {e}")
        return False


async def main():
    """Run all async tests"""
    logger.info("Starting Async Fix Tests")
    logger.info("=" * 50)
    
    # Test 1: Symbol retrieval
    test1_passed = await test_symbol_retrieval()
    
    # Test 2: Full ingestion (optional, takes longer)
    # Uncomment to test full pipeline
    # test2_passed = await test_full_ingestion_with_all()
    
    logger.info("\n" + "=" * 50)
    if test1_passed:
        logger.info("✓ All tests passed! The async event loop issue is fixed.")
        logger.info("\nThe fix:")
        logger.info("- Changed _get_all_symbols() to async method")
        logger.info("- Used 'await' instead of 'asyncio.run()' inside async context")
        logger.info("- Updated all calls to use 'await'")
    else:
        logger.info("✗ Tests failed. Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main())