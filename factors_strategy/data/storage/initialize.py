"""
Database Initialization Script
Sets up the database schema and initial data
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data.storage.clickhouse_client import ClickHouseClient
from data.storage.schema_manager import SchemaManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_database():
    """Initialize the database with all required tables"""
    
    logger.info("Starting database initialization...")
    
    try:
        # Create database client
        client = ClickHouseClient()
        
        # Create schema manager
        schema_manager = SchemaManager(client)
        
        # Create database if not exists
        db_name = client.config['database']
        client.create_database(db_name)
        logger.info(f"Database '{db_name}' created or already exists")
        
        # Create all tables
        schema_manager.create_all_tables()
        
        # Create indexes
        schema_manager.create_indexes()
        
        # Verify tables
        verify_tables(client)
        
        logger.info("Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        if 'client' in locals():
            client.close()


def verify_tables(client: ClickHouseClient):
    """Verify that all tables were created successfully"""
    
    logger.info("Verifying database tables...")
    
    expected_tables = [
        'tick_data',
        'order_book',
        'factors',
        'predictions',
        'trade_signals',
        'portfolio_positions',
        'strategy_performance'
    ]
    
    # Get list of tables
    query = f"SHOW TABLES FROM {client.config['database']}"
    result = client.execute(query)
    
    existing_tables = [row[0] for row in result]
    
    # Check each expected table
    for table in expected_tables:
        if table in existing_tables:
            # Get table size info
            size_info = client.get_table_size(table)
            logger.info(f"✓ Table '{table}' exists - Size: {size_info['size']}, Rows: {size_info['rows']}")
        else:
            logger.error(f"✗ Table '{table}' is missing!")
            
    # Report any unexpected tables
    unexpected_tables = set(existing_tables) - set(expected_tables)
    if unexpected_tables:
        logger.warning(f"Unexpected tables found: {unexpected_tables}")


def create_sample_data(client: ClickHouseClient):
    """Create sample data for testing (optional)"""
    
    logger.info("Creating sample data...")
    
    # Sample tick data
    sample_tick_data = """
    INSERT INTO tick_data (timestamp, symbol, price, volume, turnover, bid_price, bid_volume, ask_price, ask_volume, trade_direction, exchange)
    VALUES
        (now() - interval 1 hour, '000001.SZ', 12.50, 1000000, 12500000, [12.49, 12.48], [50000, 100000], [12.50, 12.51], [60000, 80000], 1, 'SZSE'),
        (now() - interval 1 hour, '000002.SZ', 25.30, 2000000, 50600000, [25.29, 25.28], [80000, 120000], [25.30, 25.31], [90000, 110000], -1, 'SZSE'),
        (now() - interval 1 hour, '600000.SH', 8.75, 1500000, 13125000, [8.74, 8.73], [70000, 90000], [8.75, 8.76], [75000, 85000], 1, 'SSE')
    """
    
    try:
        client.execute(sample_tick_data)
        logger.info("Sample tick data created")
    except Exception as e:
        logger.warning(f"Failed to create sample data: {e}")


def reset_database(client: ClickHouseClient, confirm: bool = False):
    """Reset database by dropping all tables (use with caution!)"""
    
    if not confirm:
        logger.error("Database reset requires confirmation. Set confirm=True to proceed.")
        return
        
    logger.warning("RESETTING DATABASE - This will delete all data!")
    
    # Get list of tables
    query = f"SHOW TABLES FROM {client.config['database']}"
    result = client.execute(query)
    
    tables = [row[0] for row in result]
    
    # Drop each table
    for table in tables:
        logger.info(f"Dropping table: {table}")
        client.drop_table(table)
        
    logger.info("Database reset complete")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Stock Strategy Database")
    parser.add_argument('--reset', action='store_true', help='Reset database (drops all tables)')
    parser.add_argument('--sample-data', action='store_true', help='Create sample data')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing tables')
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Just verify existing tables
        client = ClickHouseClient()
        verify_tables(client)
        client.close()
    elif args.reset:
        # Reset database
        response = input("Are you sure you want to reset the database? This will delete all data! (yes/no): ")
        if response.lower() == 'yes':
            client = ClickHouseClient()
            reset_database(client, confirm=True)
            client.close()
            # Re-initialize
            initialize_database()
        else:
            logger.info("Database reset cancelled")
    else:
        # Normal initialization
        initialize_database()
        
        if args.sample_data:
            client = ClickHouseClient()
            create_sample_data(client)
            client.close()