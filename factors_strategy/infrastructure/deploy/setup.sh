#!/bin/bash

# Multi-Factor AI Stock Selection Strategy - Setup Script
# This script sets up the complete infrastructure for the strategy system

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
PYTHON_MIN_VERSION="3.10"  # Minimum Python version required
VENV_NAME="factor-quant"
CLICKHOUSE_VERSION="23.8"
DOCKER_COMPOSE_FILE="$PROJECT_DIR/infrastructure/deploy/docker-compose.yml"

# Preserve user's Python path when using sudo
if [ -n "$SUDO_USER" ]; then
    USER_PYTHON=$(sudo -u "$SUDO_USER" which python3 2>/dev/null)
    if [ -n "$USER_PYTHON" ]; then
        export PATH="$(dirname "$USER_PYTHON"):$PATH"
    fi
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Multi-Factor AI Stock Selection Strategy${NC}"
echo -e "${GREEN}Infrastructure Setup Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION_INSTALLED=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        
        if [ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$PYTHON_VERSION_INSTALLED" | sort -V | head -n1)" = "$PYTHON_MIN_VERSION" ]; then
            print_success "Python $PYTHON_VERSION_INSTALLED is installed (>= $PYTHON_MIN_VERSION required)"
            return 0
        else
            print_error "Python $PYTHON_VERSION_INSTALLED is installed, but Python $PYTHON_MIN_VERSION or higher is required"
            print_status "Please install Python $PYTHON_MIN_VERSION+ from https://www.python.org/downloads/"
            return 1
        fi
    else
        print_error "Python 3 is not installed. Please install Python $PYTHON_MIN_VERSION or higher."
        return 1
    fi
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! check_python_version; then
    exit 1
fi

if ! command_exists docker; then
    print_error "Docker is not installed. Please install Docker."
    exit 1
fi

# Check Docker permissions
if ! docker ps >/dev/null 2>&1; then
    if [ -n "$SUDO_USER" ]; then
        print_status "Running with sudo. Checking Docker service..."
        if ! systemctl is-active --quiet docker; then
            print_status "Starting Docker service..."
            systemctl start docker
        fi
        # Try again after starting docker
        if ! docker ps >/dev/null 2>&1; then
            print_error "Cannot connect to Docker daemon even with sudo."
            exit 1
        fi
    else
        print_error "Cannot connect to Docker daemon. Permission denied."
        print_status "Please ensure Docker is running and you have proper permissions."
        print_status "Try one of the following:"
        print_status "  1. Add your user to the docker group: sudo usermod -aG docker $USER"
        print_status "  2. Log out and log back in for group changes to take effect"
        print_status "  3. Or run this script with sudo (not recommended for security reasons)"
        exit 1
    fi
fi

if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
    print_error "Docker Compose is not installed. Please install Docker Compose."
    exit 1
fi

print_success "All prerequisites are installed."

# Create project directories
print_status "Creating project directories..."

directories=(
    "data/raw"
    "data/processed"
    "models/saved"
    "models/checkpoints"
    "logs"
    "output/recommendations"
    "output/reports"
    "backups/database"
    "backups/models"
)

for dir in "${directories[@]}"; do
    echo "to create folder: $PROJECT_DIR/$dir"
    mkdir -p "$PROJECT_DIR/$dir"
    print_status "Created directory: $dir"
done

# Setup Python virtual environment
print_status "Setting up Python virtual environment..."

cd "$PROJECT_DIR"

if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv "$VENV_NAME"
    print_success "Created virtual environment: $VENV_NAME"
else
    print_status "Virtual environment already exists."
fi

# Activate virtual environment
if [ -n "$SUDO_USER" ]; then
    # When running with sudo, we need to ensure proper permissions
    chown -R "$SUDO_USER:$SUDO_USER" "$VENV_NAME"
fi
source "$VENV_NAME/bin/activate"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Install additional GPU support if NVIDIA GPU is available
if command_exists nvidia-smi; then
    print_status "NVIDIA GPU detected. Installing CUDA dependencies..."
    # PyTorch with CUDA 12.2 - using latest stable version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
    print_success "Installed PyTorch with CUDA 12.2 support"
else
    print_status "No NVIDIA GPU detected. Using CPU version of PyTorch."
    pip install torch torchvision torchaudio
fi

# Create Docker Compose file for services
print_status "Creating Docker Compose configuration..."

cat > "$DOCKER_COMPOSE_FILE" << 'EOF'
services:
  clickhouse:
    image: clickhouse/clickhouse-server:23.8
    container_name: stock_strategy_clickhouse
    ports:
      - "8123:8123"  # HTTP interface
      - "9000:9000"  # Native interface
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - ./clickhouse-config.xml:/etc/clickhouse-server/config.xml:ro
      - ./clickhouse-users.xml:/etc/clickhouse-server/users.xml:ro
    environment:
      - CLICKHOUSE_DB=stock_strategy
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "clickhouse-client", "--query", "SELECT 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: stock_strategy_redis
    ports:
      - "6380:6380"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: stock_strategy_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: stock_strategy_grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  clickhouse_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: stock_strategy_network
EOF

# Create ClickHouse configuration
print_status "Creating ClickHouse configuration..."

cat > "$PROJECT_DIR/infrastructure/deploy/clickhouse-config.xml" << 'EOF'
<?xml version="1.0"?>
<clickhouse>
    <logger>
        <level>information</level>
        <log>/var/log/clickhouse-server/clickhouse-server.log</log>
        <errorlog>/var/log/clickhouse-server/clickhouse-server.err.log</errorlog>
        <size>1000M</size>
        <count>10</count>
    </logger>

    <http_port>8123</http_port>
    <tcp_port>9000</tcp_port>
    
    <max_connections>4096</max_connections>
    <keep_alive_timeout>3</keep_alive_timeout>
    <max_concurrent_queries>100</max_concurrent_queries>
    
    <uncompressed_cache_size>8589934592</uncompressed_cache_size>
    <mark_cache_size>5368709120</mark_cache_size>
    
    <path>/var/lib/clickhouse/</path>
    <tmp_path>/var/lib/clickhouse/tmp/</tmp_path>
    
    <users_config>users.xml</users_config>
    
    <default_profile>default</default_profile>
    <default_database>stock_strategy</default_database>
    
    <timezone>Asia/Shanghai</timezone>
    
    <merge_tree>
        <max_suspicious_broken_parts>5</max_suspicious_broken_parts>
    </merge_tree>
</clickhouse>
EOF

# Create ClickHouse users configuration
cat > "$PROJECT_DIR/infrastructure/deploy/clickhouse-users.xml" << 'EOF'
<?xml version="1.0"?>
<clickhouse>
    <users>
        <default>
            <password></password>
            <networks>
                <ip>::/0</ip>
            </networks>
            <profile>default</profile>
            <quota>default</quota>
            <access_management>1</access_management>
        </default>
    </users>
    
    <profiles>
        <default>
            <max_memory_usage>10000000000</max_memory_usage>
            <use_uncompressed_cache>0</use_uncompressed_cache>
            <load_balancing>random</load_balancing>
        </default>
    </profiles>
    
    <quotas>
        <default>
            <interval>
                <duration>3600</duration>
                <queries>0</queries>
                <errors>0</errors>
                <result_rows>0</result_rows>
                <read_rows>0</read_rows>
                <execution_time>0</execution_time>
            </interval>
        </default>
    </quotas>
</clickhouse>
EOF

# Create Prometheus configuration
print_status "Creating Prometheus configuration..."

cat > "$PROJECT_DIR/infrastructure/deploy/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'clickhouse'
    static_configs:
      - targets: ['clickhouse:9363']
        
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6380']
        
  - job_name: 'strategy_app'
    static_configs:
      - targets: ['host.docker.internal:8000']
EOF

# Create Grafana datasources configuration
cat > "$PROJECT_DIR/infrastructure/deploy/grafana-datasources.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: ClickHouse
    type: vertamedia-clickhouse-datasource
    access: proxy
    url: http://clickhouse:8123
    jsonData:
      defaultDatabase: stock_strategy
EOF

# Start Docker services
print_status "Starting Docker services..."

cd "$PROJECT_DIR/infrastructure/deploy"

# Use docker compose v2 if available, otherwise fall back to docker-compose
if docker compose version >/dev/null 2>&1; then
    docker compose up -d
else
    docker-compose up -d
fi

# Wait for services to be ready
print_status "Waiting for services to be ready..."

# Wait for ClickHouse
until docker exec stock_strategy_clickhouse clickhouse-client --query "SELECT 1" >/dev/null 2>&1; do
    print_status "Waiting for ClickHouse to be ready..."
    sleep 5
done
print_success "ClickHouse is ready!"

# Wait for Redis
until docker exec stock_strategy_redis redis-cli ping >/dev/null 2>&1; do
    print_status "Waiting for Redis to be ready..."
    sleep 2
done
print_success "Redis is ready!"

# Initialize database schema
print_status "Initializing database schema..."

cd "$PROJECT_DIR"
source "$VENV_NAME/bin/activate"

# Check if the initialize module exists before running it
if [ -f "$PROJECT_DIR/data/storage/initialize.py" ]; then
    python -m data.storage.initialize
else
    print_status "Database initialization script not found. Skipping..."
fi

# Create systemd service (if on Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_status "Creating systemd service..."
    
    sudo tee /etc/systemd/system/stock-strategy.service > /dev/null << EOF
[Unit]
Description=Stock Strategy AI Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=${SUDO_USER:-$USER}
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/$VENV_NAME/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$PROJECT_DIR/$VENV_NAME/bin/python -m strategy.main_strategy
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    print_success "Systemd service created. Use 'sudo systemctl start stock-strategy' to start."
fi

# Create cron job for daily execution
print_status "Setting up cron job for daily execution..."

CRON_CMD="0 16 * * 1-5 cd $PROJECT_DIR && $PROJECT_DIR/$VENV_NAME/bin/python -m strategy.main_strategy >> $PROJECT_DIR/logs/cron.log 2>&1"
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

# Final setup tasks
print_status "Running final setup tasks..."

# Create example configuration files if they don't exist
for config_file in database.yaml model.yaml factors.yaml strategy.yaml; do
    if [ ! -f "$PROJECT_DIR/config/$config_file" ]; then
        print_error "Configuration file missing: config/$config_file"
        print_status "Please create the configuration file before running the strategy."
    fi
done

# Display service URLs
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nService URLs:"
echo -e "  - ClickHouse HTTP: http://localhost:8123"
echo -e "  - Grafana: http://localhost:3000 (admin/admin)"
echo -e "  - Prometheus: http://localhost:9090"
echo -e "  - Redis: localhost:6380"
echo -e "\nPython Version: $(python3 --version)"
echo -e "PyTorch Version: $(python -c 'import torch; print(f"PyTorch {torch.__version__}")')"
echo -e "\nNext steps:"
echo -e "  1. Configure your data sources in config/*.yaml"
echo -e "  2. Run data ingestion: python -m data.ingestion.run"
echo -e "  3. Train models: python -m models.train"
echo -e "  4. Run strategy: python -m strategy.main_strategy"
echo -e "\nFor daily automated runs, the cron job has been set up."
echo -e "Check logs in: $PROJECT_DIR/logs/"

print_success "Infrastructure setup completed successfully!"