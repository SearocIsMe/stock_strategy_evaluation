#!/usr/bin/env python
"""
Demo script showing the performance improvement with parallel data collection
"""

import subprocess
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def run_data_collection(symbols, workers=1):
    """Run data collection with specified number of workers"""
    
    # Date range for demo
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    # Build command
    cmd = [
        'python', '-m', 'data.ingestion.run',
        '--symbols'] + symbols + [
        '--start-date', start_date.strftime('%Y-%m-%d'),
        '--end-date', end_date.strftime('%Y-%m-%d'),
        '--data-types', 'tick_data',  # Only collect tick data for speed
        '--workers', str(workers)
    ]
    
    print(f"\n{'='*60}")
    print(f"Running with {workers} worker(s)")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Measure execution time
    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Success! Completed in {elapsed_time:.2f} seconds")
            print(f"   Average: {elapsed_time/len(symbols):.2f} seconds per symbol")
        else:
            print(f"❌ Failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error running command: {e}")
        
    return elapsed_time


def main():
    """Main demo function"""
    
    print("🚀 Parallel Data Collection Performance Demo")
    print("=" * 60)
    
    # Test symbols
    test_symbols = [
        '000001.SZ', '000002.SZ', '000858.SZ', '000725.SZ', '000776.SZ',
        '600000.SH', '600036.SH', '600519.SH', '600887.SH', '600276.SH'
    ]
    
    print(f"Testing with {len(test_symbols)} symbols:")
    print(f"{', '.join(test_symbols)}")
    
    # Test with different worker counts
    results = {}
    
    # Sequential (1 worker)
    print("\n" + "="*60)
    print("SEQUENTIAL PROCESSING (Baseline)")
    time_1_worker = run_data_collection(test_symbols, workers=1)
    results[1] = time_1_worker
    
    # Parallel with 5 workers
    print("\n" + "="*60)
    print("PARALLEL PROCESSING (5 Workers)")
    time_5_workers = run_data_collection(test_symbols, workers=5)
    results[5] = time_5_workers
    
    # Parallel with 10 workers
    print("\n" + "="*60)
    print("PARALLEL PROCESSING (10 Workers)")
    time_10_workers = run_data_collection(test_symbols, workers=10)
    results[10] = time_10_workers
    
    # Summary
    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY")
    print("="*60)
    
    baseline = results.get(1, 1)
    
    for workers, elapsed_time in sorted(results.items()):
        speedup = baseline / elapsed_time if elapsed_time > 0 else 0
        print(f"{workers:2d} workers: {elapsed_time:6.2f}s total, "
              f"{elapsed_time/len(test_symbols):5.2f}s per symbol, "
              f"Speedup: {speedup:4.1f}x")
    
    print("\n💡 Recommendations:")
    print("- For < 10 symbols: Use 1-3 workers")
    print("- For 10-50 symbols: Use 3-5 workers")
    print("- For 50+ symbols: Use 5-10 workers")
    print("- Monitor system resources and adjust accordingly")
    
    print("\n📝 Note: Actual speedup depends on:")
    print("- Network bandwidth and latency")
    print("- API rate limits")
    print("- System resources (CPU, memory)")
    print("- Database write performance")


if __name__ == "__main__":
    main()