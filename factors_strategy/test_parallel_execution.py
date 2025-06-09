#!/usr/bin/env python
"""
Test script to demonstrate actual parallel execution with detailed logging
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


async def simulate_data_collection(symbol: str, delay: float = 2.0):
    """Simulate data collection with a delay"""
    logger.info(f"Starting collection for {symbol}")
    await asyncio.sleep(delay)  # Simulate API call
    logger.info(f"Completed collection for {symbol}")
    return f"Data for {symbol}"


async def test_sequential_processing(symbols):
    """Test sequential processing"""
    logger.info("=== SEQUENTIAL PROCESSING ===")
    start_time = time.time()
    
    results = []
    for symbol in symbols:
        result = await simulate_data_collection(symbol)
        results.append(result)
    
    elapsed = time.time() - start_time
    logger.info(f"Sequential processing took {elapsed:.2f} seconds")
    return results


async def test_parallel_processing(symbols, max_workers=3):
    """Test parallel processing with semaphore"""
    logger.info(f"=== PARALLEL PROCESSING (max_workers={max_workers}) ===")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_with_semaphore(symbol):
        async with semaphore:
            return await simulate_data_collection(symbol)
    
    # Create all tasks
    tasks = [process_with_semaphore(symbol) for symbol in symbols]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    logger.info(f"Parallel processing took {elapsed:.2f} seconds")
    return results


async def test_asyncio_behavior():
    """Demonstrate asyncio's single-threaded concurrent execution"""
    import threading
    
    logger.info("\n=== ASYNCIO CONCURRENCY DEMONSTRATION ===")
    logger.info(f"Main thread ID: {threading.current_thread().ident}")
    
    async def task_with_thread_info(task_id: str, delay: float):
        thread_id = threading.current_thread().ident
        logger.info(f"[{task_id}] Starting - Thread ID: {thread_id}")
        await asyncio.sleep(delay)
        logger.info(f"[{task_id}] Completed - Thread ID: {thread_id}")
        return f"Result from {task_id}"
    
    # Run multiple tasks concurrently
    tasks = [
        task_with_thread_info("Task-A", 1.0),
        task_with_thread_info("Task-B", 1.0),
        task_with_thread_info("Task-C", 1.0),
    ]
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time
    
    logger.info(f"\nAll tasks completed in {elapsed:.2f} seconds")
    logger.info("Note: All tasks run in the same thread but execute concurrently!")
    
    return results


async def main():
    """Main test function"""
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META']
    
    # Test sequential processing
    logger.info("\n" + "="*60)
    await test_sequential_processing(test_symbols)
    
    # Test parallel processing with different worker counts
    logger.info("\n" + "="*60)
    await test_parallel_processing(test_symbols, max_workers=2)
    
    logger.info("\n" + "="*60)
    await test_parallel_processing(test_symbols, max_workers=6)
    
    # Demonstrate asyncio behavior
    logger.info("\n" + "="*60)
    await test_asyncio_behavior()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY:")
    logger.info("- Sequential: 6 symbols × 2 seconds = ~12 seconds")
    logger.info("- Parallel (2 workers): ~6 seconds (2x speedup)")
    logger.info("- Parallel (6 workers): ~2 seconds (6x speedup)")
    logger.info("- All async tasks run in the same thread (asyncio event loop)")
    logger.info("- Concurrency is achieved through cooperative multitasking, not threading")


if __name__ == "__main__":
    asyncio.run(main())