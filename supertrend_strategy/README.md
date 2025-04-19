# Supertrend Strategy Evaluation

A quantitative trading strategy backtesting system for A-shares market, combining multiple technical indicators including Supertrend, EMA, Ichimoku Cloud, and more.

## Project Structure

```
supertrend_strategy/
├── config/                  # Configuration files
│   └── strategy_params.yaml # Strategy parameters configuration
├── core/                    # Core components
│   ├── backtest_engine.py   # Backtesting engine
│   ├── data_fetcher.py      # Data retrieval and processing
│   ├── risk_manager.py      # Risk management and position sizing
│   └── signal_generator.py  # Signal generation based on indicators
├── utils/                   # Utility functions
│   └── plotter.py           # Visualization tools
├── results/                 # Backtest results and charts
├── main.py                  # Main entry point
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Features

- **Multi-indicator Strategy**: Combines Supertrend, EMA, Ichimoku Cloud, and volume indicators
- **Comprehensive Backtesting**: Full portfolio simulation with transaction costs
- **Risk Management**: Position sizing, stop-loss, and take-profit mechanisms
- **Performance Analysis**: Detailed metrics including returns, drawdowns, Sharpe ratio
- **Visualization**: Charts for portfolio performance, drawdowns, and trade analysis

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SearocIsMe/stock_strategy_evaluation.git
cd stock_strategy_evaluation
cd supertrend_strategy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Tushare API token:
   - Option 1: Edit the `config/strategy_params.yaml` file and replace the `tushare_token` value
   - Option 2: Set the environment variable:
   ```bash
   export TUSHARE_TOKEN="your_tushare_token_here"
   ```

## Usage

### Running a Backtest

Basic backtest with default parameters:
```bash
python main.py --mode backtest
```

Specify date range:
```bash
python main.py --mode backtest --start_date 2023-01-01 --end_date 2023-12-31
```

Specify stock list:
```bash
python main.py --mode backtest --stock_list path/to/stocklist.txt
```

### Command Line Arguments

- `--config`: Path to configuration file (default: `config/strategy_params.yaml`)
- `--mode`: Running mode (`backtest` or `live`)
- `--start_date`: Backtest start date (YYYY-MM-DD)
- `--end_date`: Backtest end date (YYYY-MM-DD)
- `--output_dir`: Output directory for results (default: `results`)
- `--stock_list`: Path to stock list file
- `--top_n`: Number of stocks to select (default: 10)
- `--verbose`: Show detailed information (default: True)
- `--plot`: Generate charts (default: True)

## Strategy Logic

The strategy combines multiple technical indicators to generate buy and sell signals:

### Buy Signals
1. All three Supertrend indicators show uptrend
2. Price is above EMA144
3. Price is above Ichimoku Cloud
4. Weekly MA conditions are met (13-week MA > 34-week MA)
5. Volume and turnover conditions are satisfied
6. RSI is not in overbought territory

### Sell Signals
1. Price falls below EMA144
2. Price enters the Ichimoku Cloud
3. Any Supertrend indicator turns to downtrend
4. Stop-loss or take-profit levels are reached

## Performance Metrics

The backtest results include:
- Total and annualized returns
- Sharpe ratio and maximum drawdown
- Win rate and profit/loss ratio
- Average holding period
- Comparison to benchmark index

## License

[MIT License](LICENSE)

## Acknowledgements

- [Tushare](https://tushare.pro/) for providing financial data
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization