# 🚀 Quick Start Guide

## Prerequisites

- **Python 3.12+** (Required - for latest performance features)
- **PyTorch 2.6.0+** (Required - for advanced GPU capabilities)
- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.1+ (optional but strongly recommended)
- At least 32GB RAM
- 500GB+ SSD storage

## 🛠️ Installation

### 1. Verify Python Version
```bash
python --version
# Should output: Python 3.12.x or higher
# If not, install Python 3.12+ from https://www.python.org/downloads/
```

### 2. Clone the Repository
```bash
git clone <repository-url>
cd factors_strategy
```

### 3. Run Setup Script
```bash
chmod +x infrastructure/deploy/setup.sh
./infrastructure/deploy/setup.sh
```

This will:
- Verify Python 3.12+ is installed
- Create Python virtual environment
- Install all dependencies (including PyTorch 2.6.0+)
- Start ClickHouse and other services
- Initialize database schema
- Set up monitoring dashboards

### 4. Configure Your Environment

Copy example configuration files:
```bash
cp config/*.yaml.example config/
```

Edit the configuration files:
- `config/database.yaml` - Database connection settings
- `config/model.yaml` - Model hyperparameters
- `config/factors.yaml` - Factor definitions
- `config/strategy.yaml` - Strategy parameters

### 5. Verify Installation
```bash
python test_setup.py
```

This will check:
- Python version (3.12+)
- PyTorch version (2.6.0+)
- GPU availability
- All required packages
- Database connectivity

## 📊 Running the Strategy

### Daily Execution
```bash
python run_strategy.py
```

### Run with Sample Data (for testing)
```bash
python -m data.storage.initialize --sample-data
python run_strategy.py --test-mode
```

### Start Dashboard
```bash
python run_strategy.py --mode dashboard
```
Then open http://localhost:8050 in your browser.

## 🔧 Common Operations

### Backup Database
```bash
chmod +x infrastructure/backup/backup.sh
./infrastructure/backup/backup.sh
```

### Restore from Backup
```bash
chmod +x infrastructure/backup/restore.sh
./infrastructure/backup/restore.sh -l  # List backups
./infrastructure/backup/restore.sh -b backup_20240101_120000.tar.gz
```

### Monitor System
- Grafana: http://localhost:3000 (admin/admin)
- ClickHouse: http://localhost:8123
- Strategy Dashboard: http://localhost:8050

### Check Logs
```bash
tail -f logs/strategy.log
tail -f logs/factors.log
tail -f logs/models.log
```

## 📈 Training Models

### Train All Models
```bash
python scripts/train_models.py --all
```

### Train Specific Model
```bash
python scripts/train_models.py --model cnn
python scripts/train_models.py --model ensemble
```

### Backtest Strategy
```bash
python run_strategy.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

## 🐛 Troubleshooting

### Python Version Issues
```bash
# Check Python version
python --version

# If < 3.12, install Python 3.12+:
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# macOS (using Homebrew):
brew install python@3.12

# Windows:
# Download from https://www.python.org/downloads/
```

### PyTorch Installation Issues
```bash
# For CUDA 12.1 (recommended for H100):
pip install torch==2.6.0 torchvision==0.20.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch==2.6.0 torchvision==0.20.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Verify installation:
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### ClickHouse Connection Issues
```bash
# Check if ClickHouse is running
docker ps | grep clickhouse

# Restart ClickHouse
docker-compose -f infrastructure/deploy/docker-compose.yml restart clickhouse

# Check logs
docker logs stock_strategy_clickhouse
```

### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch==2.6.0 torchvision==0.20.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

### Memory Issues
```bash
# Increase Docker memory limit
# Edit ~/.docker/daemon.json and add:
{
  "memory": 16384,
  "memory-swap": 32768
}

# Restart Docker
sudo systemctl restart docker
```

## 📚 Next Steps

1. **Configure Data Sources**: Set up connections to your tick data providers
2. **Customize Factors**: Add your own factors in `factors/traditional/`
3. **Train Models**: Use historical data to train the ML models
4. **Backtest**: Validate strategy performance on historical data
5. **Paper Trade**: Run in simulation mode before live trading
6. **Monitor**: Set up alerts and monitoring dashboards

## 🤝 Getting Help

- Check the [documentation](docs/)
- Review [architecture diagrams](docs/system_diagram.md)
- Look at [example notebooks](notebooks/)
- Submit issues on GitHub

## ⚡ Performance Tips

1. **Use SSD Storage**: ClickHouse performs best on fast SSDs
2. **Allocate Sufficient RAM**: At least 16GB for ClickHouse
3. **GPU Acceleration**: Use NVIDIA GPU (H100 recommended) for deep learning models
4. **Parallel Processing**: Adjust worker counts in config files
5. **Data Partitioning**: Ensure proper date-based partitioning

## 🔒 Security Checklist

- [ ] Change default passwords in configuration files
- [ ] Enable TLS for ClickHouse connections
- [ ] Set up firewall rules for exposed ports
- [ ] Configure API authentication
- [ ] Enable audit logging
- [ ] Encrypt sensitive configuration files

## 📊 Sample Commands

### Generate Daily Report
```bash
python scripts/generate_report.py --date today
```

### Export Recommendations
```bash
python scripts/export_recommendations.py --format csv --output recommendations.csv
```

### Analyze Factor Performance
```bash
python scripts/analyze_factors.py --period 30d
```

### Update Models
```bash
python scripts/update_models.py --incremental
```

## 🎯 Production Deployment

For production deployment:

1. Use Kubernetes manifests in `infrastructure/k8s/`
2. Set up proper secrets management
3. Configure horizontal pod autoscaling
4. Enable distributed tracing
5. Set up centralized logging
6. Configure backup automation
7. Implement disaster recovery procedures

See [Production Guide](docs/production.md) for detailed instructions.

## 📋 System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.12+ | 3.12+ |
| PyTorch | 2.6.0+ | 2.6.0+ |
| RAM | 32GB | 64GB+ |
| GPU | None (CPU) | NVIDIA H100 |
| Storage | 500GB SSD | 1TB+ NVMe SSD |
| CPU | 8 cores | 16+ cores |
| OS | Ubuntu 20.04+ | Ubuntu 22.04+ |