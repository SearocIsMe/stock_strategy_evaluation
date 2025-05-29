#!/bin/bash

# Multi-Factor AI Stock Selection Strategy - Backup Script
# This script backs up database data, models, and configurations

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="$PROJECT_DIR/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="stock_strategy_backup_$TIMESTAMP"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

# Load configuration
source "$PROJECT_DIR/config/backup.env" 2>/dev/null || true

# Default values if not set in config
CLICKHOUSE_HOST="${CLICKHOUSE_HOST:-localhost}"
CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-9000}"
CLICKHOUSE_USER="${CLICKHOUSE_USER:-default}"
CLICKHOUSE_PASSWORD="${CLICKHOUSE_PASSWORD:-}"
CLICKHOUSE_DATABASE="${CLICKHOUSE_DATABASE:-stock_strategy}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
COMPRESS_BACKUP="${COMPRESS_BACKUP:-true}"
BACKUP_MODELS="${BACKUP_MODELS:-true}"
BACKUP_CONFIGS="${BACKUP_CONFIGS:-true}"
BACKUP_LOGS="${BACKUP_LOGS:-false}"

# Functions
print_status() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Create backup directory structure
print_status "Creating backup directory: $BACKUP_PATH"
mkdir -p "$BACKUP_PATH"/{database,models,configs,logs,metadata}

# Backup metadata
print_status "Creating backup metadata..."
cat > "$BACKUP_PATH/metadata/backup_info.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "backup_name": "$BACKUP_NAME",
    "project_dir": "$PROJECT_DIR",
    "clickhouse_version": "$(docker exec stock_strategy_clickhouse clickhouse-client --query "SELECT version()" 2>/dev/null || echo "unknown")",
    "backup_components": {
        "database": true,
        "models": $BACKUP_MODELS,
        "configs": $BACKUP_CONFIGS,
        "logs": $BACKUP_LOGS
    }
}
EOF

# Backup ClickHouse database
print_status "Backing up ClickHouse database..."

# Get list of tables
TABLES=$(docker exec stock_strategy_clickhouse clickhouse-client \
    --host "$CLICKHOUSE_HOST" \
    --port "$CLICKHOUSE_PORT" \
    --user "$CLICKHOUSE_USER" \
    --password "$CLICKHOUSE_PASSWORD" \
    --query "SHOW TABLES FROM $CLICKHOUSE_DATABASE" 2>/dev/null)

if [ -z "$TABLES" ]; then
    print_error "No tables found in database $CLICKHOUSE_DATABASE"
else
    # Create database schema backup
    print_status "Backing up database schema..."
    
    for table in $TABLES; do
        print_status "Backing up schema for table: $table"
        
        docker exec stock_strategy_clickhouse clickhouse-client \
            --host "$CLICKHOUSE_HOST" \
            --port "$CLICKHOUSE_PORT" \
            --user "$CLICKHOUSE_USER" \
            --password "$CLICKHOUSE_PASSWORD" \
            --query "SHOW CREATE TABLE $CLICKHOUSE_DATABASE.$table" \
            > "$BACKUP_PATH/database/${table}_schema.sql" 2>/dev/null
    done
    
    # Backup table data
    print_status "Backing up table data..."
    
    for table in $TABLES; do
        print_status "Backing up data for table: $table"
        
        # Get row count
        ROW_COUNT=$(docker exec stock_strategy_clickhouse clickhouse-client \
            --host "$CLICKHOUSE_HOST" \
            --port "$CLICKHOUSE_PORT" \
            --user "$CLICKHOUSE_USER" \
            --password "$CLICKHOUSE_PASSWORD" \
            --query "SELECT count() FROM $CLICKHOUSE_DATABASE.$table" 2>/dev/null)
        
        print_status "Table $table has $ROW_COUNT rows"
        
        # Export data in batches for large tables
        if [ "$ROW_COUNT" -gt 1000000 ]; then
            print_status "Large table detected. Exporting in batches..."
            
            BATCH_SIZE=1000000
            OFFSET=0
            BATCH_NUM=1
            
            while [ $OFFSET -lt $ROW_COUNT ]; do
                print_status "Exporting batch $BATCH_NUM (offset: $OFFSET)..."
                
                docker exec stock_strategy_clickhouse clickhouse-client \
                    --host "$CLICKHOUSE_HOST" \
                    --port "$CLICKHOUSE_PORT" \
                    --user "$CLICKHOUSE_USER" \
                    --password "$CLICKHOUSE_PASSWORD" \
                    --query "SELECT * FROM $CLICKHOUSE_DATABASE.$table LIMIT $BATCH_SIZE OFFSET $OFFSET FORMAT Native" \
                    > "$BACKUP_PATH/database/${table}_data_batch${BATCH_NUM}.native" 2>/dev/null
                
                OFFSET=$((OFFSET + BATCH_SIZE))
                BATCH_NUM=$((BATCH_NUM + 1))
            done
        else
            # Export entire table for smaller tables
            docker exec stock_strategy_clickhouse clickhouse-client \
                --host "$CLICKHOUSE_HOST" \
                --port "$CLICKHOUSE_PORT" \
                --user "$CLICKHOUSE_USER" \
                --password "$CLICKHOUSE_PASSWORD" \
                --query "SELECT * FROM $CLICKHOUSE_DATABASE.$table FORMAT Native" \
                > "$BACKUP_PATH/database/${table}_data.native" 2>/dev/null
        fi
        
        # Also create CSV backup for important tables
        if [[ "$table" == "predictions" || "$table" == "trade_signals" || "$table" == "strategy_performance" ]]; then
            print_status "Creating CSV backup for table: $table"
            
            docker exec stock_strategy_clickhouse clickhouse-client \
                --host "$CLICKHOUSE_HOST" \
                --port "$CLICKHOUSE_PORT" \
                --user "$CLICKHOUSE_USER" \
                --password "$CLICKHOUSE_PASSWORD" \
                --query "SELECT * FROM $CLICKHOUSE_DATABASE.$table FORMAT CSVWithNames" \
                > "$BACKUP_PATH/database/${table}_data.csv" 2>/dev/null
        fi
    done
fi

# Backup models
if [ "$BACKUP_MODELS" = "true" ]; then
    print_status "Backing up trained models..."
    
    if [ -d "$PROJECT_DIR/models/saved" ]; then
        cp -r "$PROJECT_DIR/models/saved" "$BACKUP_PATH/models/"
        print_success "Models backed up successfully"
    else
        print_status "No saved models found"
    fi
    
    # Backup model checkpoints
    if [ -d "$PROJECT_DIR/models/checkpoints" ]; then
        print_status "Backing up model checkpoints..."
        # Only backup latest checkpoint to save space
        find "$PROJECT_DIR/models/checkpoints" -name "*.pth" -type f -mtime -7 \
            -exec cp {} "$BACKUP_PATH/models/checkpoints/" \;
    fi
fi

# Backup configurations
if [ "$BACKUP_CONFIGS" = "true" ]; then
    print_status "Backing up configuration files..."
    
    cp -r "$PROJECT_DIR/config" "$BACKUP_PATH/configs/"
    
    # Remove any sensitive information from configs
    find "$BACKUP_PATH/configs" -name "*.yaml" -type f -exec sed -i 's/password:.*/password: <REDACTED>/g' {} \;
    
    print_success "Configurations backed up successfully"
fi

# Backup logs (optional)
if [ "$BACKUP_LOGS" = "true" ]; then
    print_status "Backing up log files..."
    
    if [ -d "$PROJECT_DIR/logs" ]; then
        # Only backup recent logs (last 7 days)
        find "$PROJECT_DIR/logs" -name "*.log" -type f -mtime -7 \
            -exec cp {} "$BACKUP_PATH/logs/" \;
        print_success "Logs backed up successfully"
    fi
fi

# Create backup verification file
print_status "Creating backup verification file..."

cat > "$BACKUP_PATH/metadata/verification.sh" << 'EOF'
#!/bin/bash
# Backup verification script

echo "Verifying backup integrity..."

# Check database files
DB_FILES=$(find ./database -name "*.native" -o -name "*.sql" | wc -l)
echo "Database files found: $DB_FILES"

# Check model files
if [ -d "./models/saved" ]; then
    MODEL_FILES=$(find ./models -name "*.pkl" -o -name "*.pth" | wc -l)
    echo "Model files found: $MODEL_FILES"
fi

# Check config files
CONFIG_FILES=$(find ./configs -name "*.yaml" | wc -l)
echo "Configuration files found: $CONFIG_FILES"

# Calculate checksums
echo "Calculating checksums..."
find . -type f -exec md5sum {} \; > metadata/checksums.md5

echo "Backup verification complete!"
EOF

chmod +x "$BACKUP_PATH/metadata/verification.sh"

# Compress backup if enabled
if [ "$COMPRESS_BACKUP" = "true" ]; then
    print_status "Compressing backup..."
    
    cd "$BACKUP_DIR"
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
    
    # Calculate compressed size
    COMPRESSED_SIZE=$(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)
    print_success "Backup compressed to ${BACKUP_NAME}.tar.gz (Size: $COMPRESSED_SIZE)"
    
    # Remove uncompressed backup
    rm -rf "$BACKUP_NAME"
    
    FINAL_BACKUP="${BACKUP_NAME}.tar.gz"
else
    FINAL_BACKUP="$BACKUP_NAME"
fi

# Clean up old backups
print_status "Cleaning up old backups (retention: $RETENTION_DAYS days)..."

find "$BACKUP_DIR" -name "stock_strategy_backup_*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete
DELETED_COUNT=$(find "$BACKUP_DIR" -name "stock_strategy_backup_*" -type d -mtime +$RETENTION_DAYS | wc -l)

if [ $DELETED_COUNT -gt 0 ]; then
    find "$BACKUP_DIR" -name "stock_strategy_backup_*" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;
    print_status "Deleted $DELETED_COUNT old backups"
fi

# Upload to cloud storage (optional)
if [ ! -z "$CLOUD_BACKUP_ENABLED" ] && [ "$CLOUD_BACKUP_ENABLED" = "true" ]; then
    print_status "Uploading backup to cloud storage..."
    
    if [ ! -z "$AWS_S3_BUCKET" ]; then
        # Upload to AWS S3
        aws s3 cp "$BACKUP_DIR/$FINAL_BACKUP" "s3://$AWS_S3_BUCKET/backups/$FINAL_BACKUP"
        print_success "Backup uploaded to S3: s3://$AWS_S3_BUCKET/backups/$FINAL_BACKUP"
    fi
    
    if [ ! -z "$AZURE_CONTAINER" ]; then
        # Upload to Azure Blob Storage
        az storage blob upload \
            --container-name "$AZURE_CONTAINER" \
            --file "$BACKUP_DIR/$FINAL_BACKUP" \
            --name "backups/$FINAL_BACKUP"
        print_success "Backup uploaded to Azure: $AZURE_CONTAINER/backups/$FINAL_BACKUP"
    fi
fi

# Send notification (optional)
if [ ! -z "$NOTIFICATION_WEBHOOK" ]; then
    print_status "Sending backup notification..."
    
    BACKUP_SIZE=$(du -h "$BACKUP_DIR/$FINAL_BACKUP" | cut -f1)
    
    curl -X POST "$NOTIFICATION_WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "{
            \"text\": \"Stock Strategy Backup Completed\",
            \"backup_name\": \"$FINAL_BACKUP\",
            \"size\": \"$BACKUP_SIZE\",
            \"timestamp\": \"$TIMESTAMP\"
        }" 2>/dev/null || print_error "Failed to send notification"
fi

# Final summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Backup Completed Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Backup Name: $FINAL_BACKUP"
echo -e "Location: $BACKUP_DIR/$FINAL_BACKUP"
echo -e "Timestamp: $TIMESTAMP"

if [ "$COMPRESS_BACKUP" = "true" ]; then
    echo -e "Size: $COMPRESSED_SIZE (compressed)"
fi

print_success "Backup process completed!"

# Create backup log entry
echo "[$TIMESTAMP] Backup completed: $FINAL_BACKUP" >> "$PROJECT_DIR/logs/backup.log"