#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股量化交易策略主程序
结合Supertrend、EMA、Ichimoku云层等多重指标的选股和交易策略
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional

from core.data_fetcher import DataFetcher
from core.signal_generator import SignalGenerator
from core.risk_manager import RiskManager
from core.backtest_engine import BacktestEngine
from utils.plotter import Plotter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('strategy.log', mode='a')
    ]
)
logger = logging.getLogger('Main')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='A股量化交易策略')
    
    parser.add_argument('--config', type=str, default='config/strategy_params.yaml',
                      help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['backtest', 'live'], default='backtest',
                      help='运行模式: backtest=回测, live=实盘')
    parser.add_argument('--start_date', type=str, help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='回测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='输出目录')
    parser.add_argument('--stock_list', type=str, help='股票列表文件路径')
    parser.add_argument('--top_n', type=int, default=10,
                      help='选择的股票数量')
    parser.add_argument('--verbose', action='store_true', default=True,
                      help='是否显示详细信息')
    parser.add_argument('--plot', action='store_true', default=True,
                      help='是否生成图表')
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        sys.exit(1)

def load_stock_list(file_path: str) -> List[str]:
    """从文件加载股票列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stocks = [line.strip() for line in f if line.strip()]
        logger.info(f"从{file_path}加载了{len(stocks)}只股票")
        return stocks
    except Exception as e:
        logger.error(f"加载股票列表失败: {e}")
        return []

def run_backtest(config: dict, args) -> Dict:
    """运行回测"""
    logger.info("初始化回测组件...")
    
    # 初始化组件
    data_fetcher = DataFetcher(args.config)
    signal_generator = SignalGenerator(args.config)
    risk_manager = RiskManager(args.config)
    backtest_engine = BacktestEngine(args.config)
    
    # 设置回测参数
    start_date = args.start_date or config['data']['start_date']
    end_date = args.end_date or config['data']['end_date']
    
    # 加载股票列表
    stock_list = None
    if args.stock_list:
        stock_list = load_stock_list(args.stock_list)
    
    # 运行回测
    logger.info(f"开始回测，从{start_date}到{end_date}...")
    backtest_results = backtest_engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        stock_list=stock_list,
        verbose=args.verbose
    )
    
    # 输出回测结果
    if backtest_results:
        metrics = backtest_results.get('performance_metrics', {})
        logger.info("回测完成，绩效指标:")
        logger.info(f"总收益率: {metrics.get('total_return', 0):.2f}%")
        logger.info(f"年化收益率: {metrics.get('annual_return', 0):.2f}%")
        logger.info(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
        logger.info(f"胜率: {metrics.get('win_rate', 0):.2f}%")
        logger.info(f"总交易次数: {metrics.get('total_trades', 0)}")
        logger.info(f"基准收益率: {metrics.get('benchmark_return', 0):.2f}%")
        
        # 生成图表
        if args.plot:
            logger.info("生成回测报告图表...")
            plotter = Plotter(args.output_dir)
            report_path = plotter.generate_report(backtest_results)
            logger.info(f"回测报告已保存至: {report_path}")
            
            # 输出表现最好的交易
            top_trades = backtest_engine.get_top_performing_trades(5)
            if not top_trades.empty:
                logger.info("表现最好的5笔交易:")
                for i, (_, trade) in enumerate(top_trades.iterrows(), 1):
                    logger.info(f"{i}. {trade['ts_code']}: {trade['profit_pct']:.2f}%, "
                              f"持仓{trade['hold_days']}天, 原因: {trade['reason']}")
            
            # 输出表现最差的交易
            worst_trades = backtest_engine.get_worst_performing_trades(5)
            if not worst_trades.empty:
                logger.info("表现最差的5笔交易:")
                for i, (_, trade) in enumerate(worst_trades.iterrows(), 1):
                    logger.info(f"{i}. {trade['ts_code']}: {trade['profit_pct']:.2f}%, "
                              f"持仓{trade['hold_days']}天, 原因: {trade['reason']}")
    
    return backtest_results

def run_live_trading(config: dict, args):
    """运行实盘交易（示例框架，实际实现需要对接交易接口）"""
    logger.info("实盘交易模式尚未实现")
    logger.info("这需要对接实际的交易接口，如：券商API、交易所API等")
    logger.info("实盘交易流程大致如下:")
    logger.info("1. 初始化交易接口连接")
    logger.info("2. 获取账户信息和当前持仓")
    logger.info("3. 获取实时行情数据")
    logger.info("4. 生成交易信号")
    logger.info("5. 执行交易操作")
    logger.info("6. 更新持仓和账户信息")
    logger.info("7. 记录交易日志")
    logger.info("8. 循环执行步骤3-7")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据模式运行
    if args.mode == 'backtest':
        run_backtest(config, args)
    elif args.mode == 'live':
        run_live_trading(config, args)
    else:
        logger.error(f"不支持的运行模式: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"程序运行出错: {e}")
        sys.exit(1)
