# 🚀 Multi-Factor AI Stock Selection Strategy System

## 📊 Overview

This system implements an advanced multi-factor stock selection strategy for the Chinese A-share market, leveraging high-frequency tick data, machine learning models, and LLM-enhanced factor generation to predict stocks with 10%+ gains in the next 1-3 days.

## 🔧 Requirements

- **Python 3.12+** (Required)
- **PyTorch 2.6.0+** (Required)
- **CUDA 12.1+** (Recommended for GPU acceleration)
- **Docker & Docker Compose**
- **32GB+ RAM**
- **500GB+ SSD Storage**

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
│                    (Tick Data, OrderBook)                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                    Data Ingestion Layer                         │
│              (Real-time & Historical Processing)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                  Time-Series Database                           │
│              (ClickHouse / DolphinDB)                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                 Factor Engineering Module                       │
│         ┌─────────────────┬─────────────────┐                   │
│         │ Traditional     │ AI-Generated    │                   │
│         │ Factors         │ Factors (LLM)   │                   │
│         └─────────────────┴─────────────────┘                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                  ML Prediction Models                           │
│         ┌─────────────────┬─────────────────┐                   │
│         │ Deep Learning   │ Ensemble Models │                   │
│         │ (CNN/DNN)       │ (XGBoost/LGBM) │                    │
│         └─────────────────┴─────────────────┘                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                 Stock Ranking & Selection                       │
│                    (Top-N Recommendations)                      │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Key Features

- **High-Frequency Data Processing**: Tick-level data ingestion and processing
- **Dynamic Factor Generation**: AI-powered factor discovery using LLMs
- **Advanced ML Models**: Deep learning models optimized for H100 GPU
- **Automated Infrastructure**: Scripts for deployment, backup, and recovery
- **Configurable Parameters**: YAML-based configuration management
- **Comprehensive Backtesting**: Walk-forward validation and performance metrics

## 📁 Project Structure

```
factors_strategy/
├── config/                 # Configuration files
│   ├── database.yaml      # Database configuration
│   ├── model.yaml         # Model parameters
│   ├── factors.yaml       # Factor definitions
│   └── strategy.yaml      # Strategy parameters
├── data/                  # Data processing modules
│   ├── ingestion/         # Data ingestion scripts
│   ├── storage/           # Database interfaces
│   └── preprocessing/     # Data cleaning and preparation
├── factors/               # Factor engineering
│   ├── traditional/       # Traditional factors
│   ├── ai_generated/      # AI-generated factors
│   └── evaluation/        # Factor evaluation metrics
├── models/                # Machine learning models
│   ├── deep_learning/     # CNN/DNN models
│   ├── ensemble/          # XGBoost/LightGBM
│   └── llm_integration/   # LLM factor generation
├── strategy/              # Strategy implementation
│   ├── ranking/           # Stock ranking algorithms
│   ├── selection/         # Portfolio selection
│   └── risk_management/   # Risk controls
├── backtesting/           # Backtesting framework
│   ├── engine/            # Backtesting engine
│   └── metrics/           # Performance metrics
├── infrastructure/        # Infrastructure automation
│   ├── deploy/            # Deployment scripts
│   ├── backup/            # Backup scripts
│   └── monitoring/        # System monitoring
├── visualization/         # Data visualization
│   └── dashboards/        # Interactive dashboards
└── tests/                 # Unit and integration tests
```

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.12+ installed:
```bash
python --version  # Should show Python 3.12.x or higher
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd factors_strategy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup infrastructure
./infrastructure/deploy/setup.sh
```

### Configuration

1. Copy example configuration files:
```bash
cp config/*.example.yaml config/
```

2. Update configuration files with your settings:
- `database.yaml`: Database connection details
- `model.yaml`: Model hyperparameters
- `factors.yaml`: Factor definitions
- `strategy.yaml`: Strategy parameters

### Running the System

```bash
# Initialize database
python -m data.storage.initialize

# Test system setup
python test_setup.py

# Run data ingestion
python -m data.ingestion.run

# Train models
python -m models.train

# Generate predictions
python -m strategy.predict

# Run backtesting
python -m backtesting.run
```

## 📊 Factor Categories

### Traditional Factors
- **Liquidity Factors**: Order flow imbalance, bid-ask spread
- **Volatility Factors**: Realized volatility, price jumps
- **Momentum Factors**: Price momentum, volume momentum
- **Microstructure Factors**: Trade size distribution, order book depth

### AI-Generated Factors
- **Deep Learning Features**: CNN-extracted patterns from tick data
- **LLM-Enhanced Factors**: Dynamically generated factors with semantic meaning
- **Ensemble Features**: Combined signals from multiple models

## 🤖 Model Architecture

### Deep Learning Models
- **CNN Model**: Captures temporal patterns in tick data
- **LSTM/GRU**: Sequential pattern recognition
- **Transformer**: Attention-based feature extraction

### Ensemble Models
- **XGBoost**: Gradient boosting for tabular features
- **LightGBM**: Fast gradient boosting
- **Random Forest**: Robust ensemble predictions

## 📈 Performance Metrics

- **Information Coefficient (IC)**: Factor predictive power
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Risk metrics
- **Hit Rate**: Prediction accuracy for 10%+ gains

## 🔧 Infrastructure

### Database Schema
- Tick data storage with nanosecond precision
- Factor value storage with versioning
- Prediction history and performance tracking

### Automation Scripts
- `deploy.sh`: Automated deployment
- `backup.sh`: Database backup
- `restore.sh`: System recovery
- `monitor.sh`: Health monitoring

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Architecture Overview](docs/architecture.md)
- [System Diagrams](docs/system_diagram.md)
- [Implementation Summary](docs/implementation_summary.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚡ Performance Requirements

- **Python**: 3.12+ (for latest performance optimizations)
- **PyTorch**: 2.6.0+ (for advanced GPU features)
- **GPU**: NVIDIA H100 recommended for optimal performance
- **Memory**: 32GB minimum, 64GB+ recommended
- **Storage**: NVMe SSD for database performance

## 🔒 Security Features

- Configuration encryption
- API authentication
- Rate limiting
- Audit logging
- Data masking in logs

## 📞 Support

For issues and questions:
- Check the [documentation](docs/)
- Review [common issues](docs/troubleshooting.md)
- Submit an issue on GitHub