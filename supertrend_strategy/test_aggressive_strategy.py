#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试激进策略的回测脚本
"""

import os
import sys
import logging
from main import run_backtest, load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aggressive_strategy_test.log', mode='a')
    ]
)
logger = logging.getLogger('AggressiveStrategyTest')

def main():
    """主函数"""
    # 设置参数
    class Args:
        config = 'config/strategy_params.yaml'
        strategy = 'aggressive'
        mode = 'backtest'
        start_date = '2023-05-01'
        end_date = '2023-12-31'
        output_dir = 'results'
        stock_list = None
        top_n = 10
        verbose = True
        plot = True
        output_csv = True
    
    args = Args()
    
    # 加载配置，应用指定策略
    config = load_config(args.config, args.strategy)
    
    # 确保使用激进策略
    config['active_strategy'] = 'aggressive'
    logger.info(f"强制设置活动策略为: {config['active_strategy']}")
    
    # 确保策略ID参数也设置为激进策略
    args.strategy = 'aggressive'
    
    # 创建输出目录
    output_dir = f"{args.output_dir}/{args.strategy}"
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    # 运行回测
    logger.info(f"开始测试激进策略回测，从{args.start_date}到{args.end_date}...")
    backtest_results = run_backtest(config, args)
    
    logger.info("测试完成")
    return backtest_results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"程序运行出错: {e}")
        sys.exit(1)