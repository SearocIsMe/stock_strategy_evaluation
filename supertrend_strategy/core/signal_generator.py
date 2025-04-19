import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger('SignalGenerator')

class SignalGenerator:
    """
    信号生成器类
    负责根据技术指标生成买入和卖出信号
    """
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml'):
        """
        初始化信号生成器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def generate_signals(self, stock_data: Dict[str, pd.DataFrame], date: str) -> Dict[str, Dict]:
        """
        为指定日期生成所有股票的信号
        
        Args:
            stock_data: 股票数据字典 {股票代码: 数据DataFrame}
            date: 日期字符串 (YYYYMMDD)
            
        Returns:
            信号字典 {股票代码: {'signal': 信号, 'score': 分数, ...}}
        """
        signals = {}
        total_stocks = len(stock_data)
        valid_stocks = 0
        stocks_with_buy_signal = 0
        stocks_with_sell_signal = 0
        
        logger.info(f"开始为日期 {date} 生成信号，共 {total_stocks} 只股票")
        
        for ts_code, df in stock_data.items():
            # 确保数据不为空且包含指定日期
            if df.empty or date not in df['trade_date'].values:
                continue
            
            # 获取指定日期的数据
            date_idx = df[df['trade_date'] == date].index[0]
            
            # 确保有足够的历史数据
            if date_idx < 20:  # 至少需要20天数据
                continue
            
            valid_stocks += 1
            
            # 获取当天数据
            current_data = df.iloc[date_idx]
            
            # 生成买入信号
            buy_signal = self._generate_buy_signal(df, date_idx)
            if buy_signal == 1:
                stocks_with_buy_signal += 1
            
            # 生成卖出信号
            sell_signal = self._generate_sell_signal(df, date_idx)
            if sell_signal == 1:
                stocks_with_sell_signal += 1
            
            # 计算股票评分
            score = self._calculate_stock_score(df, date_idx)
            
            # 存储信号
            signals[ts_code] = {
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'score': score,
                'price': current_data['close'],
                'date': date,
                'rsi': current_data['rsi'] if 'rsi' in current_data else None,
                'all_uptrend': current_data['all_uptrend'] if 'all_uptrend' in current_data else None,
                'price_above_ema': current_data['price_above_ema'] if 'price_above_ema' in current_data else None,
                'price_above_cloud': current_data['price_above_cloud'] if 'price_above_cloud' in current_data else None,
                'weekly_condition': current_data['weekly_condition'] if 'weekly_condition' in current_data else None,
                'volume_signal': current_data['volume_signal'] if 'volume_signal' in current_data else None,
                'turnover_signal': current_data['turnover_signal'] if 'turnover_signal' in current_data else None,
                'cost_valid': current_data['cost_valid'] if 'cost_valid' in current_data else None
            }
        
        logger.info(f"日期 {date} 信号生成完成:")
        logger.info(f"  - 有效股票: {valid_stocks}/{total_stocks}")
        logger.info(f"  - 买入信号: {stocks_with_buy_signal} 只")
        logger.info(f"  - 卖出信号: {stocks_with_sell_signal} 只")
        
        return signals
    
    def _generate_buy_signal(self, df: pd.DataFrame, idx: int) -> int:
        """
        生成买入信号
        
        Args:
            df: 股票数据
            idx: 当前日期索引
            
        Returns:
            1表示买入，0表示不操作
        """
        # 确保索引有效
        if idx < 0 or idx >= len(df):
            return 0
        
        current = df.iloc[idx]
        
        # 策略1: 三重Supertrend过滤 + EMA144 + Ichimoku云层
        strategy1_signal = 0
        if ('all_uptrend' in current and current['all_uptrend'] == 1 and
            'price_above_ema' in current and current['price_above_ema'] == 1 and
            'price_above_cloud' in current and current['price_above_cloud'] == 1):
            strategy1_signal = 1
        
        # 策略2: 周线均线 + 放量 + 换手率 + 筹码集中度
        strategy2_components = 0
        if 'weekly_condition' in current and current['weekly_condition'] == 1:
            strategy2_components += 1
        if 'volume_signal' in current and current['volume_signal'] == 1:
            strategy2_components += 1
        if 'turnover_signal' in current and current['turnover_signal'] == 1:
            strategy2_components += 1
        if 'cost_valid' in current and current['cost_valid'] == 1:
            strategy2_components += 1
        
        # 策略2信号: 至少满足3个条件
        strategy2_signal = 1 if strategy2_components >= 3 else 0
        
        # 综合信号: 两个策略都满足
        buy_signal = 1 if strategy1_signal == 1 and strategy2_signal == 1 else 0
        
        # RSI过滤: 避免超买
        if 'rsi' in current and current['rsi'] > self.config['rsi']['overbought']:
            buy_signal = 0
        
        return buy_signal
    
    def _generate_sell_signal(self, df: pd.DataFrame, idx: int) -> int:
        """
        生成卖出信号
        
        Args:
            df: 股票数据
            idx: 当前日期索引
            
        Returns:
            1表示卖出，0表示不操作
        """
        # 确保索引有效
        if idx < 1 or idx >= len(df):  # 需要至少一天的历史数据
            return 0
        
        current = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        # 止损条件1: 跌破EMA144
        stop_condition1 = 'price_above_ema' in current and current['price_above_ema'] == 0
        
        # 止损条件2: 进入云层
        stop_condition2 = 'price_above_cloud' in current and current['price_above_cloud'] == 0
        
        # 止损条件3: Supertrend转势
        stop_condition3 = ('all_uptrend' in current and current['all_uptrend'] == 0 and
                          'all_uptrend' in prev and prev['all_uptrend'] == 1)
        
        # 综合止损信号
        sell_signal = 1 if stop_condition1 or stop_condition2 or stop_condition3 else 0
        
        return sell_signal
    
    def _calculate_stock_score(self, df: pd.DataFrame, idx: int) -> float:
        """
        计算股票评分
        
        Args:
            df: 股票数据
            idx: 当前日期索引
            
        Returns:
            股票评分 (0-100)
        """
        # 确保索引有效
        if idx < 0 or idx >= len(df):
            return 0.0
        
        current = df.iloc[idx]
        
        # 基础分数
        base_score = 0.0
        
        # 趋势强度 (40分)
        trend_score = 0.0
        if 'all_uptrend' in current and current['all_uptrend'] == 1:
            trend_score += 15.0
        if 'price_above_ema' in current and current['price_above_ema'] == 1:
            trend_score += 10.0
        if 'price_above_cloud' in current and current['price_above_cloud'] == 1:
            trend_score += 15.0
        
        # 量能评分 (30分)
        volume_score = 0.0
        if 'volume_signal' in current and current['volume_signal'] == 1:
            volume_score += 15.0
        if 'turnover_signal' in current and current['turnover_signal'] == 1:
            volume_score += 15.0
        
        # 筹码评分 (20分)
        chip_score = 0.0
        if 'cost_valid' in current and current['cost_valid'] == 1:
            chip_score += 20.0
        
        # 周线评分 (10分)
        weekly_score = 0.0
        if 'weekly_condition' in current and current['weekly_condition'] == 1:
            weekly_score += 10.0
        
        # 总分
        total_score = trend_score + volume_score + chip_score + weekly_score
        
        # RSI反向调整 (RSI越高，得分越低)
        if 'rsi' in current and not pd.isna(current['rsi']):
            rsi = current['rsi']
            # RSI > 70 开始扣分，最多扣30分
            if rsi > 70:
                rsi_penalty = min(30, (rsi - 70) * 1.5)
                total_score = max(0, total_score - rsi_penalty)
        
        return total_score
    
    def select_stocks(self, signals: Dict[str, Dict], max_stocks: int = 10) -> List[str]:
        """
        根据信号和评分选择股票
        
        Args:
            signals: 信号字典 {股票代码: {'signal': 信号, 'score': 分数, ...}}
            max_stocks: 最大选择数量
            
        Returns:
            选中的股票代码列表
        """
        # 筛选有买入信号的股票
        buy_stocks = {ts_code: data for ts_code, data in signals.items() 
                     if data['buy_signal'] == 1}
        
        # 按评分排序
        sorted_stocks = sorted(buy_stocks.items(), 
                              key=lambda x: x[1]['score'], 
                              reverse=True)
        
        # 选择前N只股票
        selected = [ts_code for ts_code, _ in sorted_stocks[:max_stocks]]
        
        logger.info(f"选股完成，从{len(buy_stocks)}只买入信号股票中选出{len(selected)}只")
        return selected
    
    def get_position_adjustments(self, 
                               current_positions: Dict[str, float],
                               signals: Dict[str, Dict],
                               max_positions: int = 10) -> Tuple[List[str], List[str]]:
        """
        获取仓位调整建议
        
        Args:
            current_positions: 当前持仓 {股票代码: 持仓比例}
            signals: 信号字典
            max_positions: 最大持仓数量
            
        Returns:
            (买入列表, 卖出列表)
        """
        # 获取卖出信号的股票
        to_sell = [ts_code for ts_code in current_positions.keys()
                  if ts_code in signals and signals[ts_code]['sell_signal'] == 1]
        
        # 选择新的买入股票
        available_slots = max_positions - (len(current_positions) - len(to_sell))
        
        if available_slots <= 0:
            return [], to_sell
        
        # 筛选未持仓且有买入信号的股票
        potential_buys = {ts_code: data for ts_code, data in signals.items()
                         if ts_code not in current_positions and data['buy_signal'] == 1}
        
        # 按评分排序
        sorted_buys = sorted(potential_buys.items(),
                           key=lambda x: x[1]['score'],
                           reverse=True)
        
        # 选择前N只股票
        to_buy = [ts_code for ts_code, _ in sorted_buys[:available_slots]]
        
        logger.info(f"仓位调整: 买入{len(to_buy)}只，卖出{len(to_sell)}只")
        return to_buy, to_sell
