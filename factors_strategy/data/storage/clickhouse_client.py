"""
ClickHouse Database Client
Manages connections and operations for ClickHouse time-series database
"""

import logging
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
import pandas as pd
from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseError
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class ClickHouseClient:
    """ClickHouse database client with connection pooling and query optimization"""
    
    def __init__(self, config_path: str = "config/database.yaml"):
        """Initialize ClickHouse client with configuration"""
        self.config = self._load_config(config_path)
        self.client = None
        self._connect()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load database configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        return config['database']['clickhouse']
        
    def _connect(self):
        """Establish connection to ClickHouse"""
        try:
            self.client = Client(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                settings={
                    'max_memory_usage': 10000000000,  # 10GB
                    'max_threads': 16,
                    'distributed_product_mode': 'global',
                }
            )
            logger.info(f"Connected to ClickHouse at {self.config['host']}:{self.config['port']}")
        except ClickHouseError as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise
            
    @contextmanager
    def get_client(self):
        """Context manager for database operations"""
        try:
            yield self.client
        except ClickHouseError as e:
            logger.error(f"ClickHouse error: {e}")
            raise
        finally:
            pass  # Connection pooling handled by client
            
    def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute a query and return results"""
        try:
            with self.get_client() as client:
                if params:
                    return client.execute(query, params)
                return client.execute(query)
        except ClickHouseError as e:
            logger.error(f"Query execution failed: {e}")
            raise
            
    def execute_many(self, query: str, data: List[Dict]) -> None:
        """Execute batch insert operations"""
        try:
            with self.get_client() as client:
                client.execute(query, data)
            logger.info(f"Inserted {len(data)} records")
        except ClickHouseError as e:
            logger.error(f"Batch insert failed: {e}")
            raise
            
    def query_dataframe(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute query and return results as pandas DataFrame"""
        try:
            with self.get_client() as client:
                if params:
                    result = client.execute(query, params, with_column_types=True)
                else:
                    result = client.execute(query, with_column_types=True)
                    
                if not result[0]:  # No data
                    return pd.DataFrame()
                    
                columns = [col[0] for col in result[1]]
                return pd.DataFrame(result[0], columns=columns)
        except ClickHouseError as e:
            logger.error(f"DataFrame query failed: {e}")
            raise
            
    def insert_dataframe(self, table: str, df: pd.DataFrame, batch_size: int = 100000) -> None:
        """Insert pandas DataFrame into ClickHouse table"""
        try:
            # Convert DataFrame to list of tuples for insertion
            data = df.to_dict('records')
            
            # Insert in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                query = f"INSERT INTO {table} VALUES"
                self.execute_many(query, batch)
                
            logger.info(f"Inserted {len(df)} records into {table}")
        except Exception as e:
            logger.error(f"DataFrame insert failed: {e}")
            raise
            
    def create_database(self, database: str) -> None:
        """Create database if not exists"""
        query = f"CREATE DATABASE IF NOT EXISTS {database}"
        self.execute(query)
        logger.info(f"Database {database} created or already exists")
        
    def drop_table(self, table: str, if_exists: bool = True) -> None:
        """Drop table"""
        if_exists_clause = "IF EXISTS" if if_exists else ""
        query = f"DROP TABLE {if_exists_clause} {table}"
        self.execute(query)
        logger.info(f"Table {table} dropped")
        
    def optimize_table(self, table: str, partition: Optional[str] = None) -> None:
        """Optimize table for better performance"""
        if partition:
            query = f"OPTIMIZE TABLE {table} PARTITION {partition} FINAL"
        else:
            query = f"OPTIMIZE TABLE {table} FINAL"
        self.execute(query)
        logger.info(f"Table {table} optimized")
        
    def get_table_size(self, table: str) -> Dict[str, Any]:
        """Get table size statistics"""
        query = f"""
        SELECT 
            formatReadableSize(sum(bytes)) as size,
            sum(rows) as rows,
            count() as parts
        FROM system.parts
        WHERE database = '{self.config['database']}' 
        AND table = '{table}'
        AND active
        """
        result = self.execute(query)
        if result:
            return {
                'size': result[0][0],
                'rows': result[0][1],
                'parts': result[0][2]
            }
        return {'size': '0B', 'rows': 0, 'parts': 0}
        
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.disconnect()
            logger.info("ClickHouse connection closed")