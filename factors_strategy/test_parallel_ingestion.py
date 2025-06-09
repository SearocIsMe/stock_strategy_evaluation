#!/usr/bin/env python
"""
Test script to demonstrate parallel data ingestion
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.ingestion.run import DataIngestionPipeline
from datetime import datetime, timedelta


async def test_parallel_ingestion():
    """Test parallel data ingestion with different worker counts"""
    
    # Test symbols
    test_symbols = [
        '000001.SZ', '000002.SZ', '000858.SZ', '000725.SZ', '000776.SZ',
        '600000.SH', '600036.SH', '600519.SH', '600887.SH', '600276.SH'
    ]
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    # Test with different worker counts
    worker_counts = [1, 3, 5]
    
    for workers in worker_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {workers} workers")
        print(f"{'='*60}")
        
        # Create pipeline with specified worker count
        pipeline = DataIngestionPipeline(
            config_path="config/database.yaml",
            data_providers_config="config/data_providers.yaml",
            max_workers=workers
        )
        
        # Measure execution time
        start_time = time.time()
        
        try:
            await pipeline.run_ingestion(
                symbols=test_symbols,
                start_date=start_date,
                end_date=end_date,
                data_types=['tick_data']  # Only test tick data for speed
            )
            
            elapsed_time = time.time() - start_time
            print(f"\nCompleted in {elapsed_time:.2f} seconds with {workers} workers")
            print(f"Average time per symbol: {elapsed_time/len(test_symbols):.2f} seconds")
            
        except Exception as e:
            print(f"Error with {workers} workers: {e}")


async def test_all_symbols():
    """Test parallel ingestion with ALL symbols"""
    
    print(f"\n{'='*60}")
    print("Testing with ALL symbols and 10 workers")
    print(f"{'='*60}")
    
    # Create pipeline with 10 workers
    pipeline = DataIngestionPipeline(
        config_path="config/database.yaml",
        data_providers_config="config/data_providers.yaml",
        max_workers=10
    )
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    # Measure execution time
    start_time = time.time()
    
    try:
        await pipeline.run_ingestion(
            symbols=['ALL'],
            start_date=start_date,
            end_date=end_date,
            data_types=['tick_data']
        )
        
        elapsed_time = time.time() - start_time
        print(f"\nCompleted in {elapsed_time:.2f} seconds with 10 workers")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main entry point"""
    print("Parallel Data Ingestion Performance Test")
    print("========================================")
    
    # Run tests
    asyncio.run(test_parallel_ingestion())
    
    # Uncomment to test with ALL symbols
    # asyncio.run(test_all_symbols())


if __name__ == "__main__":
    main()