#!/bin/bash

# Multi-Factor AI Stock Selection Strategy - Restore Script
# This script restores database data, models, and configurations from backup

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="$PROJECT_DIR/backups"

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

print_warning() {
    echo -e "${BLUE}[WARNING] $1${NC}"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -b, --backup-file FILE    Path to backup file (tar.gz or directory)"
    echo "  -l, --list                List available backups"
    echo "  -d, --database-only       Restore only database"
    echo "  -m, --models-only         Restore only models"
    echo "  -c, --configs-only        Restore only configurations"
    echo "  -f, --force               Force restore without confirmation"
    echo "  -t, --test                Test restore (dry run)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -l                                    # List available backups"
    echo "  $0 -b backup_20240101_120000.tar.gz     # Restore from specific backup"
    echo "  $0 -b backup_20240101_120000 -d         # Restore only database"
}

# Parse command line arguments
BACKUP_FILE=""
LIST_BACKUPS=false
DATABASE_ONLY=false
MODELS_ONLY=false
CONFIGS_ONLY=false
FORCE_RESTORE=false
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--backup-file)
            BACKUP_FILE="$2"
            shift 2
            ;;
        -l|--list)
            LIST_BACKUPS=true
            shift
            ;;
        -d|--database-only)
            DATABASE_ONLY=true
            shift
            ;;
        -m|--models-only)
            MODELS_ONLY=true
            shift
            ;;
        -c|--configs-only)
            CONFIGS_ONLY=true
            shift
            ;;
        -f|--force)
            FORCE_RESTORE=true
            shift
            ;;
        -t|--test)
            TEST_MODE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# List available backups
if [ "$LIST_BACKUPS" = true ]; then
    echo -e "${GREEN}Available backups:${NC}"
    echo ""
    
    # List compressed backups
    if ls "$BACKUP_DIR"/stock_strategy_backup_*.tar.gz 1> /dev/null 2>&1; then
        for backup in "$BACKUP_DIR"/stock_strategy_backup_*.tar.gz; do
            BACKUP_NAME=$(basename "$backup")
            BACKUP_SIZE=$(du -h "$backup" | cut -f1)
            BACKUP_DATE=$(stat -c %y "$backup" 2>/dev/null || stat -f %Sm "$backup" 2>/dev/null)
            echo "  - $BACKUP_NAME (Size: $BACKUP_SIZE, Date: $BACKUP_DATE)"
        done
    fi
    
    # List uncompressed backups
    if ls -d "$BACKUP_DIR"/stock_strategy_backup_*/ 1> /dev/null 2>&1; then
        for backup in "$BACKUP_DIR"/stock_strategy_backup_*/; do
            BACKUP_NAME=$(basename "$backup")
            BACKUP_SIZE=$(du -sh "$backup" | cut -f1)
            echo "  - $BACKUP_NAME/ (Size: $BACKUP_SIZE)"
        done
    fi
    
    exit 0
fi

# Check if backup file is specified
if [ -z "$BACKUP_FILE" ]; then
    print_error "No backup file specified. Use -b option to specify backup file."
    show_usage
    exit 1
fi

# Determine backup path
if [[ "$BACKUP_FILE" = /* ]]; then
    # Absolute path
    BACKUP_PATH="$BACKUP_FILE"
else
    # Relative path or filename
    if [ -f "$BACKUP_DIR/$BACKUP_FILE" ]; then
        BACKUP_PATH="$BACKUP_DIR/$BACKUP_FILE"
    elif [ -d "$BACKUP_DIR/$BACKUP_FILE" ]; then
        BACKUP_PATH="$BACKUP_DIR/$BACKUP_FILE"
    else
        BACKUP_PATH="$BACKUP_FILE"
    fi
fi

# Check if backup exists
if [ ! -e "$BACKUP_PATH" ]; then
    print_error "Backup not found: $BACKUP_PATH"
    exit 1
fi

# Extract backup if compressed
TEMP_DIR=""
if [[ "$BACKUP_PATH" == *.tar.gz ]]; then
    print_status "Extracting backup archive..."
    TEMP_DIR=$(mktemp -d)
    tar -xzf "$BACKUP_PATH" -C "$TEMP_DIR"
    
    # Find the extracted directory
    EXTRACTED_DIR=$(find "$TEMP_DIR" -maxdepth 1 -type d -name "stock_strategy_backup_*" | head -1)
    if [ -z "$EXTRACTED_DIR" ]; then
        print_error "Failed to find extracted backup directory"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
    RESTORE_DIR="$EXTRACTED_DIR"
else
    RESTORE_DIR="$BACKUP_PATH"
fi

# Verify backup structure
print_status "Verifying backup structure..."

if [ ! -f "$RESTORE_DIR/metadata/backup_info.json" ]; then
    print_error "Invalid backup: metadata/backup_info.json not found"
    [ ! -z "$TEMP_DIR" ] && rm -rf "$TEMP_DIR"
    exit 1
fi

# Display backup information
echo -e "\n${GREEN}Backup Information:${NC}"
cat "$RESTORE_DIR/metadata/backup_info.json" | python3 -m json.tool

# Determine what to restore
RESTORE_DATABASE=true
RESTORE_MODELS=true
RESTORE_CONFIGS=true

if [ "$DATABASE_ONLY" = true ]; then
    RESTORE_MODELS=false
    RESTORE_CONFIGS=false
elif [ "$MODELS_ONLY" = true ]; then
    RESTORE_DATABASE=false
    RESTORE_CONFIGS=false
elif [ "$CONFIGS_ONLY" = true ]; then
    RESTORE_DATABASE=false
    RESTORE_MODELS=false
fi

# Confirmation prompt
if [ "$FORCE_RESTORE" = false ] && [ "$TEST_MODE" = false ]; then
    echo -e "\n${YELLOW}This will restore the following components:${NC}"
    [ "$RESTORE_DATABASE" = true ] && echo "  - Database"
    [ "$RESTORE_MODELS" = true ] && echo "  - Models"
    [ "$RESTORE_CONFIGS" = true ] && echo "  - Configurations"
    
    echo -e "\n${RED}WARNING: This will overwrite existing data!${NC}"
    read -p "Are you sure you want to continue? (yes/no): " CONFIRM
    
    if [ "$CONFIRM" != "yes" ]; then
        print_status "Restore cancelled by user"
        [ ! -z "$TEMP_DIR" ] && rm -rf "$TEMP_DIR"
        exit 0
    fi
fi

# Test mode message
if [ "$TEST_MODE" = true ]; then
    echo -e "\n${BLUE}Running in TEST MODE - no changes will be made${NC}"
fi

# Load configuration
source "$PROJECT_DIR/config/backup.env" 2>/dev/null || true

# Default values if not set in config
CLICKHOUSE_HOST="${CLICKHOUSE_HOST:-localhost}"
CLICKHOUSE_PORT="${CLICKHOUSE_PORT:-9000}"
CLICKHOUSE_USER="${CLICKHOUSE_USER:-default}"
CLICKHOUSE_PASSWORD="${CLICKHOUSE_PASSWORD:-}"
CLICKHOUSE_DATABASE="${CLICKHOUSE_DATABASE:-stock_strategy}"

# Restore database
if [ "$RESTORE_DATABASE" = true ] && [ -d "$RESTORE_DIR/database" ]; then
    print_status "Restoring database..."
    
    # Check if ClickHouse is running
    if ! docker exec stock_strategy_clickhouse clickhouse-client --query "SELECT 1" >/dev/null 2>&1; then
        print_error "ClickHouse is not running. Please start it first."
        [ ! -z "$TEMP_DIR" ] && rm -rf "$TEMP_DIR"
        exit 1
    fi
    
    # Create database if not exists
    if [ "$TEST_MODE" = false ]; then
        docker exec stock_strategy_clickhouse clickhouse-client \
            --host "$CLICKHOUSE_HOST" \
            --port "$CLICKHOUSE_PORT" \
            --user "$CLICKHOUSE_USER" \
            --password "$CLICKHOUSE_PASSWORD" \
            --query "CREATE DATABASE IF NOT EXISTS $CLICKHOUSE_DATABASE"
    fi
    
    # Restore tables
    for schema_file in "$RESTORE_DIR"/database/*_schema.sql; do
        if [ -f "$schema_file" ]; then
            TABLE_NAME=$(basename "$schema_file" _schema.sql)
            print_status "Restoring table schema: $TABLE_NAME"
            
            if [ "$TEST_MODE" = false ]; then
                # Drop existing table
                docker exec stock_strategy_clickhouse clickhouse-client \
                    --host "$CLICKHOUSE_HOST" \
                    --port "$CLICKHOUSE_PORT" \
                    --user "$CLICKHOUSE_USER" \
                    --password "$CLICKHOUSE_PASSWORD" \
                    --query "DROP TABLE IF EXISTS $CLICKHOUSE_DATABASE.$TABLE_NAME"
                
                # Create table from schema
                cat "$schema_file" | docker exec -i stock_strategy_clickhouse clickhouse-client \
                    --host "$CLICKHOUSE_HOST" \
                    --port "$CLICKHOUSE_PORT" \
                    --user "$CLICKHOUSE_USER" \
                    --password "$CLICKHOUSE_PASSWORD" \
                    --database "$CLICKHOUSE_DATABASE"
            fi
        fi
    done
    
    # Restore table data
    for data_file in "$RESTORE_DIR"/database/*_data.native; do
        if [ -f "$data_file" ]; then
            TABLE_NAME=$(basename "$data_file" _data.native)
            print_status "Restoring table data: $TABLE_NAME"
            
            if [ "$TEST_MODE" = false ]; then
                cat "$data_file" | docker exec -i stock_strategy_clickhouse clickhouse-client \
                    --host "$CLICKHOUSE_HOST" \
                    --port "$CLICKHOUSE_PORT" \
                    --user "$CLICKHOUSE_USER" \
                    --password "$CLICKHOUSE_PASSWORD" \
                    --query "INSERT INTO $CLICKHOUSE_DATABASE.$TABLE_NAME FORMAT Native"
            fi
        fi
    done
    
    # Restore batched data
    for batch_file in "$RESTORE_DIR"/database/*_data_batch*.native; do
        if [ -f "$batch_file" ]; then
            TABLE_NAME=$(basename "$batch_file" | sed 's/_data_batch[0-9]*.native//')
            BATCH_NUM=$(basename "$batch_file" | sed 's/.*_data_batch\([0-9]*\).native/\1/')
            print_status "Restoring table data batch $BATCH_NUM: $TABLE_NAME"
            
            if [ "$TEST_MODE" = false ]; then
                cat "$batch_file" | docker exec -i stock_strategy_clickhouse clickhouse-client \
                    --host "$CLICKHOUSE_HOST" \
                    --port "$CLICKHOUSE_PORT" \
                    --user "$CLICKHOUSE_USER" \
                    --password "$CLICKHOUSE_PASSWORD" \
                    --query "INSERT INTO $CLICKHOUSE_DATABASE.$TABLE_NAME FORMAT Native"
            fi
        fi
    done
    
    print_success "Database restore completed"
fi

# Restore models
if [ "$RESTORE_MODELS" = true ] && [ -d "$RESTORE_DIR/models" ]; then
    print_status "Restoring models..."
    
    if [ "$TEST_MODE" = false ]; then
        # Backup existing models
        if [ -d "$PROJECT_DIR/models/saved" ]; then
            MODELS_BACKUP="$PROJECT_DIR/models/saved.backup.$(date +%Y%m%d_%H%M%S)"
            mv "$PROJECT_DIR/models/saved" "$MODELS_BACKUP"
            print_status "Existing models backed up to: $MODELS_BACKUP"
        fi
        
        # Restore models
        cp -r "$RESTORE_DIR/models/saved" "$PROJECT_DIR/models/" 2>/dev/null || true
        
        # Restore checkpoints if present
        if [ -d "$RESTORE_DIR/models/checkpoints" ]; then
            mkdir -p "$PROJECT_DIR/models/checkpoints"
            cp -r "$RESTORE_DIR/models/checkpoints/"* "$PROJECT_DIR/models/checkpoints/" 2>/dev/null || true
        fi
    fi
    
    print_success "Models restore completed"
fi

# Restore configurations
if [ "$RESTORE_CONFIGS" = true ] && [ -d "$RESTORE_DIR/configs" ]; then
    print_status "Restoring configurations..."
    
    if [ "$TEST_MODE" = false ]; then
        # Backup existing configs
        CONFIG_BACKUP="$PROJECT_DIR/config.backup.$(date +%Y%m%d_%H%M%S)"
        cp -r "$PROJECT_DIR/config" "$CONFIG_BACKUP"
        print_status "Existing configs backed up to: $CONFIG_BACKUP"
        
        # Restore configs
        cp -r "$RESTORE_DIR/configs/"* "$PROJECT_DIR/config/"
        
        print_warning "Configuration files restored. Please review and update any environment-specific settings."
    fi
    
    print_success "Configurations restore completed"
fi

# Verify restoration
if [ "$TEST_MODE" = false ]; then
    print_status "Verifying restoration..."
    
    # Verify database
    if [ "$RESTORE_DATABASE" = true ]; then
        TABLES=$(docker exec stock_strategy_clickhouse clickhouse-client \
            --host "$CLICKHOUSE_HOST" \
            --port "$CLICKHOUSE_PORT" \
            --user "$CLICKHOUSE_USER" \
            --password "$CLICKHOUSE_PASSWORD" \
            --query "SHOW TABLES FROM $CLICKHOUSE_DATABASE" 2>/dev/null | wc -l)
        
        print_status "Database tables restored: $TABLES"
    fi
    
    # Verify models
    if [ "$RESTORE_MODELS" = true ]; then
        if [ -d "$PROJECT_DIR/models/saved" ]; then
            MODEL_COUNT=$(find "$PROJECT_DIR/models/saved" -name "*.pkl" -o -name "*.pth" | wc -l)
            print_status "Model files restored: $MODEL_COUNT"
        fi
    fi
fi

# Cleanup
if [ ! -z "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

# Final summary
echo -e "\n${GREEN}========================================${NC}"
if [ "$TEST_MODE" = true ]; then
    echo -e "${GREEN}Restore Test Completed!${NC}"
    echo -e "${GREEN}No changes were made.${NC}"
else
    echo -e "${GREEN}Restore Completed Successfully!${NC}"
fi
echo -e "${GREEN}========================================${NC}"

echo -e "\nRestored components:"
[ "$RESTORE_DATABASE" = true ] && echo "  ✓ Database"
[ "$RESTORE_MODELS" = true ] && echo "  ✓ Models"
[ "$RESTORE_CONFIGS" = true ] && echo "  ✓ Configurations"

if [ "$TEST_MODE" = false ]; then
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo "  1. Review restored configuration files"
    echo "  2. Update any environment-specific settings"
    echo "  3. Restart the strategy service"
    echo "  4. Verify system functionality"
fi

print_success "Restore process completed!"

# Create restore log entry
if [ "$TEST_MODE" = false ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Restore completed from: $BACKUP_FILE" >> "$PROJECT_DIR/logs/restore.log"
fi