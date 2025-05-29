#!/usr/bin/env python
"""
Main entry point for the Multi-Factor AI Stock Selection Strategy
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from strategy.main_strategy import StockSelectionStrategy
from visualization.dashboard import StrategyDashboard, create_static_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/strategy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Multi-Factor AI Stock Selection Strategy'
    )
    
    parser.add_argument(
        '--mode',
        choices=['run', 'backtest', 'dashboard', 'report'],
        default='run',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        default='today',
        help='Date to run strategy for (YYYY-MM-DD or "today")'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtesting (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtesting (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with sample data'
    )
    
    parser.add_argument(
        '--dashboard-port',
        type=int,
        default=8050,
        help='Port for dashboard server'
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Configuration directory'
    )
    
    return parser.parse_args()


async def run_strategy(args):
    """Run the strategy for a specific date"""
    # Parse date
    if args.date == 'today':
        target_date = datetime.now()
    else:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    
    logger.info(f"Running strategy for date: {target_date.date()}")
    
    # Initialize strategy
    strategy = StockSelectionStrategy(config_dir=args.config_dir)
    strategy.initialize()
    
    # Run strategy
    try:
        recommendations = await strategy.run_daily_strategy(target_date)
        
        # Display results
        logger.info("\n" + "="*60)
        logger.info("STOCK RECOMMENDATIONS")
        logger.info("="*60)
        
        for i, rec in enumerate(recommendations, 1):
            logger.info(
                f"{i}. {rec['symbol']} | "
                f"Score: {rec['ensemble_score']:.3f} | "
                f"Expected 3D Return: {rec['expected_return_3d']:.2%} | "
                f"Position: {rec['position_size']:.2%}"
            )
        
        logger.info("="*60)
        
        # Generate report
        if not args.test_mode:
            report_path = create_static_report(target_date)
            logger.info(f"Report generated: {report_path}")
            
    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        raise


async def run_backtest(args):
    """Run backtesting"""
    if not args.start_date or not args.end_date:
        logger.error("Start date and end date required for backtesting")
        return
    
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
    
    # Initialize strategy
    strategy = StockSelectionStrategy(config_dir=args.config_dir)
    strategy.initialize()
    
    # Run backtest for each trading day
    current_date = start_date
    all_results = []
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            logger.info(f"Backtesting {current_date.date()}")
            
            try:
                recommendations = await strategy.run_daily_strategy(current_date)
                all_results.append({
                    'date': current_date,
                    'recommendations': recommendations
                })
            except Exception as e:
                logger.error(f"Backtest failed for {current_date.date()}: {e}")
        
        current_date += timedelta(days=1)
    
    # Generate backtest report
    logger.info(f"Backtest completed. Processed {len(all_results)} days")
    
    # Calculate performance metrics
    total_days = len(all_results)
    avg_recommendations = sum(len(r['recommendations']) for r in all_results) / total_days
    
    logger.info(f"Average recommendations per day: {avg_recommendations:.1f}")


def run_dashboard(args):
    """Run the interactive dashboard"""
    logger.info(f"Starting dashboard on port {args.dashboard_port}")
    
    dashboard = StrategyDashboard(config_path=f"{args.config_dir}/strategy.yaml")
    dashboard.run(debug=False, port=args.dashboard_port)


def generate_report(args):
    """Generate a static report"""
    # Parse date
    if args.date == 'today':
        target_date = datetime.now()
    else:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    
    logger.info(f"Generating report for {target_date.date()}")
    
    report_path = create_static_report(target_date)
    logger.info(f"Report saved to: {report_path}")


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("Multi-Factor AI Stock Selection Strategy")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    
    try:
        if args.mode == 'run':
            asyncio.run(run_strategy(args))
        elif args.mode == 'backtest':
            asyncio.run(run_backtest(args))
        elif args.mode == 'dashboard':
            run_dashboard(args)
        elif args.mode == 'report':
            generate_report(args)
            
    except KeyboardInterrupt:
        logger.info("Strategy execution interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()