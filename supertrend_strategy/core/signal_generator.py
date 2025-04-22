import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
from core.data_fetcher import DataFetcher

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
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml', mode: str = 'backtest'):
        """
        初始化信号生成器
        
        Args:
            config_path: 配置文件路径
            mode: 运行模式，'backtest'或'live'
        """
        # 加载配置
        self.config = self._load_config(config_path)
        self.mode = mode
        
        # 初始化数据获取器，用于获取基本面数据
        self.data_fetcher = DataFetcher(config_path, mode=mode)
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查是否使用新的策略配置格式
            if 'strategies' in config:
                # 获取默认策略或第一个可用策略
                strategy_id = config.get('active_strategy', 'default')
                if strategy_id not in config['strategies']:
                    available_strategies = list(config['strategies'].keys())
                    if available_strategies:
                        strategy_id = available_strategies[0]
                        logger.warning(f"策略 '{strategy_id}' 不存在，使用 '{available_strategies[0]}'")
                    else:
                        logger.warning("配置文件中没有定义任何策略，使用原始配置")
                        return config
                
                # 合并策略配置到主配置
                strategy_config = config['strategies'][strategy_id]
                for key, value in strategy_config.items():
                    config[key] = value
                
                logger.info(f"使用策略配置: {strategy_id}")
            
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def generate_signals(self, stock_data: Dict[str, pd.DataFrame], date: str,
                        current_positions: Dict[str, Dict] = None) -> Dict[str, Dict]:
        """
        为指定日期生成所有股票的信号
        
        Args:
            stock_data: 股票数据字典 {股票代码: 数据DataFrame}
            date: 日期字符串 (YYYYMMDD)
            current_positions: 当前持仓 {股票代码: 持仓信息}，用于卖出信号约束
            
        Returns:
            信号字典 {股票代码: {'signal': 信号, 'score': 分数, ...}}
        """
        signals = {}
        total_stocks = len(stock_data)
        valid_stocks = 0
        stocks_with_buy_signal = 0
        stocks_with_sell_signal = 0
        
        # 初始化当前持仓
        if current_positions is None:
            current_positions = {}
        
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
            
            # 生成卖出信号 - 只有当股票在当前持仓中时才考虑卖出信号
            sell_signal = 0
            if ts_code in current_positions:
                sell_signal = self._generate_sell_signal(df, date_idx)
                if sell_signal == 1:
                    stocks_with_sell_signal += 1
            
            # 计算股票评分
            score = self._calculate_stock_score(df, date_idx)
            
            # 获取基本面数据用于计算ROE/PB
            roe_pb_score = 0
            ema20_deviation_score = 0
            
            # 获取基本面数据
            try:
                # 传递当前日期，以便在回测模式下获取正确的基本面数据
                fundamental_data = self.data_fetcher.get_fundamental_data(ts_code, date=date)
                if not fundamental_data.empty:
                    # 计算ROE/PB比率
                    if 'roe' in fundamental_data.columns and 'pb' in fundamental_data.columns:
                        roe = fundamental_data['roe'].iloc[0]
                        pb = fundamental_data['pb'].iloc[0]
                        
                        if not pd.isna(roe) and not pd.isna(pb) and pb > 0:
                            roe_pb_ratio = roe / pb
                            # ROE/PB比率越高，分数越高 (最高10分)
                            # 一般来说，ROE/PB > 1 是比较好的
                            roe_pb_score = min(10, roe_pb_ratio * 5)
            except Exception as e:
                logger.warning(f"获取股票{ts_code}基本面数据失败: {e}")
            
            # 计算EMA(20)乖离率
            if 'close' in current_data and date_idx >= 20:
                # 计算EMA(20)
                ema20 = df['close'].iloc[date_idx-20:date_idx+1].ewm(span=20, adjust=False).mean().iloc[-1]
                # 计算乖离率
                deviation = abs(current_data['close'] / ema20 - 1) * 100
                # 乖离率越小，分数越高 (最高10分)
                ema20_deviation_score = max(0, 10 - deviation)
            
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
                'cost_valid': current_data['cost_valid'] if 'cost_valid' in current_data else None,
                'cost_valid_ratio': current_data['cost_valid_ratio'] if 'cost_valid_ratio' in current_data else None,
                'price_deviation_ratio': current_data['price_deviation_ratio'] if 'price_deviation_ratio' in current_data else None,
                'ema20_deviation_score': ema20_deviation_score,
                'roe_pb_score': roe_pb_score
            }
        
        logger.info(f"日期 {date} 信号生成完成:")
        logger.info(f"  - 有效股票: {valid_stocks}/{total_stocks}")
        logger.info(f"  - 买入信号: {stocks_with_buy_signal} 只")
        logger.info(f"  - 卖出信号: {stocks_with_sell_signal} 只")
        
        # 打印评分最高的前10只股票
        if signals:
            top_stocks = sorted(signals.items(), key=lambda x: x[1]['score'] if 'score' in x[1] else 0, reverse=True)[:10]
            logger.info(f"评分最高的前10只股票:")
            for i, (ts_code, data) in enumerate(top_stocks, 1):
                # 基本信息
                log_msg = f"  {i}. {ts_code}: 评分={data['score']:.2f}, 买入信号={data['buy_signal']}, 卖出信号={data['sell_signal']}"
                
                # 添加详细指标信息
                indicators = [
                    f"RSI={data['rsi']:.2f}" if data['rsi'] is not None else "RSI=N/A",
                    f"上升趋势={data['all_uptrend']}" if data['all_uptrend'] is not None else "上升趋势=N/A",
                    f"价格>EMA={data['price_above_ema']}" if data['price_above_ema'] is not None else "价格>EMA=N/A",
                    f"价格>云层={data['price_above_cloud']}" if data['price_above_cloud'] is not None else "价格>云层=N/A",
                    f"周线条件={data['weekly_condition']}" if data['weekly_condition'] is not None else "周线条件=N/A",
                    f"成交量信号={data['volume_signal']}" if data['volume_signal'] is not None else "成交量信号=N/A",
                    f"换手率信号={data['turnover_signal']}" if data['turnover_signal'] is not None else "换手率信号=N/A",
                    f"筹码集中比率={data.get('cost_valid_ratio', 0):.2f}" if 'cost_valid_ratio' in data else "筹码集中比率=N/A",
                    f"价格偏离比率={data.get('price_deviation_ratio', 0):.2f}" if 'price_deviation_ratio' in data else "价格偏离比率=N/A",
                    f"EMA20乖离率评分={data.get('ema20_deviation_score', 0):.2f}",
                    f"ROE/PB评分={data.get('roe_pb_score', 0):.2f}"
                ]
                
                # 添加指标信息到日志
                log_msg += "\n    " + ", ".join(indicators)
                logger.info(log_msg)
        
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
        
        # 趋势强度 (35分)
        trend_score = 0.0
        if 'all_uptrend' in current and current['all_uptrend'] == 1:
            trend_score += 12.0
        if 'price_above_ema' in current and current['price_above_ema'] == 1:
            trend_score += 10.0
        if 'price_above_cloud' in current and current['price_above_cloud'] == 1:
            trend_score += 13.0
        
        # 量能评分 (25分)
        volume_score = 0.0
        if 'volume_signal' in current and current['volume_signal'] == 1:
            volume_score += 12.5
        if 'turnover_signal' in current and current['turnover_signal'] == 1:
            volume_score += 12.5
        
        # 筹码评分 (15分) - 使用筹码集中度比率
        chip_score = 0.0
        if 'cost_valid_ratio' in current and not pd.isna(current['cost_valid_ratio']):
            # 筹码集中度比率越高，得分越高 (最高15分)
            # 比率至少为1才能得到满分
            chip_score = min(15.0, current['cost_valid_ratio'] * 15.0)
        elif 'cost_valid' in current and current['cost_valid'] == 1:
            # 兼容旧版本
            chip_score += 15.0
        
        # 周线评分 (5分)
        weekly_score = 0.0
        if 'weekly_condition' in current and current['weekly_condition'] == 1:
            weekly_score += 5.0
        
        # EMA(20)乖离率评分 (10分)
        ema20_deviation_score = 0.0
        if idx >= 20:
            # 计算EMA(20)
            ema20 = df['close'].iloc[idx-20:idx+1].ewm(span=20, adjust=False).mean().iloc[-1]
            # 计算乖离率
            deviation = abs(current['close'] / ema20 - 1) * 100
            # 乖离率越小，分数越高 (最高10分)
            ema20_deviation_score = max(0, 10 - deviation)
        
        # ROE/PB评分 (10分)
        roe_pb_score = 0.0
        # 尝试从基本面数据计算ROE/PB
        if 'roe' in current and 'pb' in current and not pd.isna(current['roe']) and not pd.isna(current['pb']) and current['pb'] > 0:
            roe_pb_ratio = current['roe'] / current['pb']
            roe_pb_score = min(10, roe_pb_ratio * 5)
        
        # 总分
        total_score = trend_score + volume_score + chip_score + weekly_score + ema20_deviation_score + roe_pb_score
        
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
        
        # 按评分排序 - 使用总分、ROE/PB评分和EMA20乖离率作为排序因素
        sorted_stocks = sorted(buy_stocks.items(),
                              key=lambda x: (x[1]['score'],
                                           x[1].get('roe_pb_score', 0),
                                           x[1].get('ema20_deviation_score', 0)),
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
        # 获取卖出信号的股票 - 只考虑当前持仓中的股票
        to_sell = [ts_code for ts_code in current_positions.keys()
                  if ts_code in signals and signals[ts_code]['sell_signal'] == 1]
        
        # 选择新的买入股票
        available_slots = max_positions - (len(current_positions) - len(to_sell))
        
        if available_slots <= 0:
            return [], to_sell
        
        # 筛选未持仓且有买入信号的股票
        potential_buys = {ts_code: data for ts_code, data in signals.items()
                         if ts_code not in current_positions and data['buy_signal'] == 1}
        
        # 按评分排序 - 使用总分、ROE/PB评分和EMA20乖离率作为排序因素
        sorted_buys = sorted(potential_buys.items(),
                           key=lambda x: (x[1]['score'],
                                        x[1].get('roe_pb_score', 0),
                                        x[1].get('ema20_deviation_score', 0)),
                           reverse=True)
        
        # 选择前N只股票
        to_buy = [ts_code for ts_code, _ in sorted_buys[:available_slots]]
        
        logger.info(f"仓位调整: 买入{len(to_buy)}只，卖出{len(to_sell)}只")
        return to_buy, to_sell
