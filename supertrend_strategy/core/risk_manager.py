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
logger = logging.getLogger('RiskManager')

class RiskManager:
    """
    风险管理器类
    负责仓位管理、止盈止损和资金分配
    """
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml'):
        """
        初始化风险管理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化资金和仓位
        self.initial_capital = self.config['capital']['initial']
        self.current_capital = self.initial_capital
        self.max_positions = self.config['capital']['max_positions']
        self.position_sizing_method = self.config['capital']['position_sizing']
        
        # 止盈止损参数
        self.take_profit_pct = self.config['exit']['take_profit'] / 100
        self.stop_loss_pct = self.config['exit']['stop_loss'] / 100
        
        # 激进策略的部分止盈参数
        self.partial_take_profit_pct1 = 0.30  # 30%止盈点
        self.partial_take_profit_pct2 = 0.50  # 50%止盈点
        
        # 持仓记录
        self.positions = {}  # {股票代码: {'entry_price': 价格, 'shares': 数量, 'entry_date': 日期, ...}}
        
        # 交易记录
        self.trades = []  # 交易记录列表
        
        # 部分止盈记录
        self.partial_profit_taken = {}  # {股票代码: {'30%': True/False, '50%': True/False}}
    
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
    
    def calculate_position_size(self, available_capital: float, num_positions: int) -> float:
        """
        计算单个仓位大小
        
        Args:
            available_capital: 可用资金
            num_positions: 计划持仓数量
            
        Returns:
            单个仓位资金
        """
        if num_positions <= 0:
            return 0
        
        # 获取最小持仓比例配置
        min_position_ratio = self.config['evaluation']['position_limits'].get('min_position_ratio', 0.05)
        
        # 计算总资产价值
        total_portfolio_value = self.current_capital
        
        # 计算最小仓位大小 (总资产的最小比例)
        min_position_size = total_portfolio_value * min_position_ratio
        
        # 计算平均分配的仓位大小
        if self.position_sizing_method == 'equal':
            # 等权分配
            avg_position_size = available_capital / num_positions
        elif self.position_sizing_method == 'dynamic':
            # 动态分配 (基于波动率或其他因素)
            # 这里简化为等权分配，实际可以基于波动率或其他因素调整
            avg_position_size = available_capital / num_positions
        else:
            # 默认等权分配
            avg_position_size = available_capital / num_positions
        
        # 返回较大的值，确保仓位不小于最小比例
        return max(avg_position_size, min_position_size)
    
    def open_position(self, ts_code: str, price: float, date: str,
                     available_capital: float = None, signals: Dict = None) -> Dict:
        """
        开仓
        
        Args:
            ts_code: 股票代码
            price: 开仓价格
            date: 开仓日期
            available_capital: 可用资金，默认使用当前资金
            signals: 信号字典，用于获取额外信息
            
        Returns:
            开仓信息字典
        """
        # 检查是否已持仓
        if ts_code in self.positions:
            logger.warning(f"股票{ts_code}已在持仓中，不能重复开仓")
            return {}
        
        # 检查持仓数量是否已达上限
        if len(self.positions) >= self.max_positions:
            logger.warning(f"持仓数量已达上限{self.max_positions}，不能继续开仓")
            return {}
        
        # 计算可用资金
        if available_capital is None:
            # 计算当前可用资金
            invested_capital = sum(pos['capital'] for pos in self.positions.values())
            available_capital = self.current_capital - invested_capital
        
        # 计算仓位大小
        position_capital = self.calculate_position_size(
            available_capital, 
            self.max_positions - len(self.positions)
        )
        
        # 计算可买入股数 (向下取整到100股的整数倍)
        shares = int(position_capital / price / 100) * 100
        
        # 检查是否有足够资金买入至少100股
        if shares < 100:
            logger.warning(f"资金不足，无法买入股票{ts_code}的最小交易单位(100股)")
            return {}
        
        # 计算实际使用资金
        actual_capital = shares * price
        
        # 检查是否满足最小持仓比例要求
        min_position_ratio = self.config['evaluation']['position_limits'].get('min_position_ratio', 0.05)
        total_portfolio_value = self.current_capital
        min_position_size = total_portfolio_value * min_position_ratio
        
        if actual_capital < min_position_size:
            logger.warning(f"股票{ts_code}的持仓金额{actual_capital:.2f}元低于最小持仓比例要求({min_position_ratio*100:.1f}%，即{min_position_size:.2f}元)")
            # 我们仍然允许开仓，但会在CSV中标记为"持仓不足"
        
        # 记录持仓信息
        position = {
            'ts_code': ts_code,
            'entry_price': price,
            'shares': shares,
            'capital': actual_capital,
            'entry_date': date,
            'take_profit_price': price * (1 + self.take_profit_pct),
            'stop_loss_price': price * (1 - self.stop_loss_pct)
        }
        
        # 获取当前策略ID
        strategy_id = self.config.get('active_strategy', 'default')
        
        # 如果是激进策略，添加额外信息
        if strategy_id == 'aggressive' and signals and ts_code in signals:
            # 添加涨停信息
            position['limit_up_valid'] = signals[ts_code].get('limit_up_valid', False)
            
            # 添加月线MACD信息
            position['monthly_macd_golden_cross'] = signals[ts_code].get('monthly_macd_golden_cross', False)
        
        self.positions[ts_code] = position
        
        # 记录交易
        trade = {
            'ts_code': ts_code,
            'date': date,
            'type': 'buy',
            'price': price,
            'shares': shares,
            'value': actual_capital,
            'commission': self._calculate_commission(actual_capital, is_sell=False)
        }
        
        # 如果是激进策略，添加额外信息到交易记录
        if strategy_id == 'aggressive' and signals and ts_code in signals:
            # 添加涨停信息
            trade['limit_up_valid'] = signals[ts_code].get('limit_up_valid', False)
            
            # 添加月线MACD信息
            trade['monthly_macd_golden_cross'] = signals[ts_code].get('monthly_macd_golden_cross', False)
        self.trades.append(trade)
        
        logger.info(f"开仓: {ts_code}, 价格: {price}, 股数: {shares}, 金额: {actual_capital:.2f}")
        return position
    
    def close_position(self, ts_code: str, price: float, date: str, reason: str = 'signal',
                      partial_ratio: float = 1.0) -> Dict:
        """
        平仓
        
        Args:
            ts_code: 股票代码
            price: 平仓价格
            date: 平仓日期
            reason: 平仓原因 ('signal', 'stop_loss', 'take_profit', 'partial_take_profit_30', 'partial_take_profit_50')
            partial_ratio: 平仓比例 (1.0表示全部平仓，小于1.0表示部分平仓)
            
        Returns:
            平仓信息字典
        """
        # 检查是否持仓
        if ts_code not in self.positions:
            logger.warning(f"股票{ts_code}不在持仓中，无法平仓")
            return {}
        
        # 获取持仓信息
        position = self.positions[ts_code]
        
        # 计算平仓股数
        shares_to_close = int(position['shares'] * partial_ratio)
        if shares_to_close < 100:  # 确保至少卖出100股
            shares_to_close = min(100, position['shares'])
        
        # 计算平仓金额
        close_value = shares_to_close * price
        
        # 计算收益
        position_ratio = shares_to_close / position['shares']
        capital_closed = position['capital'] * position_ratio
        profit = close_value - capital_closed
        profit_pct = profit / capital_closed * 100
        
        # 记录交易
        trade = {
            'ts_code': ts_code,
            'date': date,
            'type': 'sell',
            'price': price,
            'shares': shares_to_close,
            'value': close_value,
            'commission': self._calculate_commission(close_value, is_sell=True),
            'profit': profit,
            'profit_pct': profit_pct,
            'hold_days': self._calculate_hold_days(position['entry_date'], date),
            'reason': reason,
            'partial_ratio': partial_ratio
        }
        self.trades.append(trade)
        
        # 更新资金
        self.current_capital += profit
        
        # 更新或移除持仓
        if partial_ratio < 1.0 and position['shares'] - shares_to_close >= 100:
            # 部分平仓，更新持仓
            self.positions[ts_code]['shares'] -= shares_to_close
            self.positions[ts_code]['capital'] -= capital_closed
            
            # 更新部分止盈记录
            if reason == 'partial_take_profit_30':
                self.partial_profit_taken[ts_code]['30%'] = True
            elif reason == 'partial_take_profit_50':
                self.partial_profit_taken[ts_code]['50%'] = True
                
            logger.info(f"部分平仓: {ts_code}, 价格: {price}, 股数: {shares_to_close}, "
                      f"收益: {profit:.2f} ({profit_pct:.2f}%), 原因: {reason}")
        else:
            # 全部平仓，移除持仓
            del self.positions[ts_code]
            # 如果有部分止盈记录，也移除
            if ts_code in self.partial_profit_taken:
                del self.partial_profit_taken[ts_code]
            logger.info(f"平仓: {ts_code}, 价格: {price}, 股数: {shares_to_close}, "
                      f"收益: {profit:.2f} ({profit_pct:.2f}%), 原因: {reason}")
        
        return trade
    
    def _calculate_commission(self, value: float, is_sell: bool = False) -> float:
        """
        计算交易手续费
        
        Args:
            value: 交易金额
            is_sell: 是否为卖出交易
            
        Returns:
            手续费
        """
        # 从配置中获取费率
        commission_rate = self.config['evaluation']['transaction_costs'].get('commission_rate', 0.00025)
        stamp_duty_rate = self.config['evaluation']['transaction_costs'].get('stamp_duty_rate', 0.001)
        min_commission = self.config['evaluation']['transaction_costs'].get('min_commission', 5.0)
        
        # 计算佣金
        commission = value * commission_rate
        
        # 最低佣金
        commission = max(commission, min_commission)
        
        # 如果是卖出交易，加上印花税
        if is_sell:
            stamp_duty = value * stamp_duty_rate
            commission += stamp_duty
        
        return commission
    
    def _calculate_hold_days(self, entry_date: str, exit_date: str) -> int:
        """
        计算持仓天数
        
        Args:
            entry_date: 开仓日期
            exit_date: 平仓日期
            
        Returns:
            持仓天数
        """
        try:
            # 转换日期格式
            if len(entry_date) == 8:  # YYYYMMDD格式
                entry = datetime.strptime(entry_date, '%Y%m%d')
                exit = datetime.strptime(exit_date, '%Y%m%d')
            else:  # YYYY-MM-DD格式
                entry = datetime.strptime(entry_date, '%Y-%m-%d')
                exit = datetime.strptime(exit_date, '%Y-%m-%d')
            
            # 计算天数差
            delta = exit - entry
            return delta.days
        except:
            return 0
    
    def check_stop_conditions(self, ts_code: str, current_price: float, 
                             signals: Dict = None) -> Tuple[bool, str]:
        """
        检查止盈止损条件
        
        Args:
            ts_code: 股票代码
            current_price: 当前价格
            signals: 信号字典 (可选)
            
        Returns:
            (是否平仓, 平仓原因)
        """
        # 检查是否持仓
        if ts_code not in self.positions:
            return False, ''
        
        position = self.positions[ts_code]
        
        # 检查止盈条件
        if current_price >= position['take_profit_price']:
            return True, 'take_profit'
        
        # 检查止损条件
        if current_price <= position['stop_loss_price']:
            return True, 'stop_loss'
        
        # 检查信号止损
        if signals and ts_code in signals and signals[ts_code]['sell_signal'] == 1:
            return True, 'signal'
        
        return False, ''
    
    def update_portfolio(self, date: str, stock_data: Dict[str, pd.DataFrame],
                        signals: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        更新投资组合
        
        Args:
            date: 当前日期
            stock_data: 股票数据字典
            signals: 信号字典
            
        Returns:
            (开仓列表, 平仓列表)
        """
        # 记录操作
        opened_positions = []
        closed_positions = []
        
        # 获取当前策略ID
        strategy_id = self.config.get('active_strategy', 'default')
        
        # 1. 检查现有持仓是否需要平仓
        for ts_code in list(self.positions.keys()):
            # 获取当前价格
            if ts_code in stock_data and not stock_data[ts_code].empty:
                df = stock_data[ts_code]
                current_data = df[df['trade_date'] == date]
                
                if not current_data.empty:
                    current_price = current_data['close'].values[0]
                    position = self.positions[ts_code]
                    
                    # 检查是否是激进策略下的"买入后第二天跌破20日均线"条件
                    if strategy_id == 'aggressive':
                        # 检查是否是买入后的第二天
                        entry_date = position['entry_date']
                        days_held = self._calculate_hold_days(entry_date, date)
                        
                        if days_held == 1:  # 买入后的第二天
                            # 检查是否跌破20日均线
                            if 'ema20' in current_data.columns and current_price < current_data['ema20'].values[0]:
                                # 平仓
                                closed_trade = self.close_position(ts_code, current_price, date, 'day_after_buy_exit')
                                if closed_trade:
                                    closed_positions.append(closed_trade)
                                continue  # 已平仓，跳过后续检查
                    
                    # 检查部分止盈条件 (仅适用于激进策略)
                    if strategy_id == 'aggressive':
                        should_partial_close, partial_ratio, partial_reason = self.check_partial_profit_conditions(ts_code, current_price)
                        
                        if should_partial_close:
                            # 部分平仓
                            closed_trade = self.close_position(ts_code, current_price, date, partial_reason, partial_ratio)
                            if closed_trade:
                                closed_positions.append(closed_trade)
                            continue  # 已部分平仓，跳过后续检查
                    
                    # 检查止盈止损条件
                    should_close, reason = self.check_stop_conditions(ts_code, current_price, signals)
                    
                    if should_close:
                        # 平仓
                        closed_trade = self.close_position(ts_code, current_price, date, reason)
                        if closed_trade:
                            closed_positions.append(closed_trade)
        
        # 2. 计算可用资金
        invested_capital = sum(pos['capital'] for pos in self.positions.values())
        available_capital = self.current_capital - invested_capital
        
        # 3. 计算可开仓数量
        available_slots = self.max_positions - len(self.positions)
        
        # 4. 如果有可用仓位，寻找新的开仓机会
        if available_slots > 0 and available_capital > 0:
            # 筛选有买入信号的股票
            buy_candidates = {ts_code: data for ts_code, data in signals.items()
                             if data['buy_signal'] == 1 and ts_code not in self.positions}
            
            # 按评分排序
            sorted_candidates = sorted(buy_candidates.items(),
                                     key=lambda x: x[1]['score'],
                                     reverse=True)
            
            # 选择前N只股票开仓
            for i, (ts_code, data) in enumerate(sorted_candidates):
                if i >= available_slots:
                    break
                
                # 开仓
                position = self.open_position(
                    ts_code,
                    data['price'],
                    date,
                    available_capital / (available_slots - i),
                    signals  # 传递信号字典
                )
                
                if position:
                    opened_positions.append(position)
                    # 更新可用资金
                    available_capital -= position['capital']
        
        return opened_positions, closed_positions
    
    def get_portfolio_value(self, date: str, stock_data: Dict[str, pd.DataFrame]) -> float:
        """
        计算投资组合价值
        
        Args:
            date: 当前日期
            stock_data: 股票数据字典
            
        Returns:
            投资组合总价值
        """
        # 计算持仓市值
        portfolio_value = 0
        
        for ts_code, position in self.positions.items():
            # 获取当前价格
            if ts_code in stock_data and not stock_data[ts_code].empty:
                df = stock_data[ts_code]
                current_data = df[df['trade_date'] == date]
                
                if not current_data.empty:
                    current_price = current_data['close'].values[0]
                    position_value = position['shares'] * current_price
                    portfolio_value += position_value
        
        # 加上未投入的现金
        invested_capital = sum(pos['capital'] for pos in self.positions.values())
        cash = self.current_capital - invested_capital
        
        total_value = portfolio_value + cash
        
        return total_value
    
    def get_position_summary(self) -> pd.DataFrame:
        """
        获取持仓摘要
        
        Returns:
            持仓摘要DataFrame
        """
        if not self.positions:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(self.positions.values())
        
        # 计算持仓比例
        total_capital = sum(pos['capital'] for pos in self.positions.values())
        if total_capital > 0:
            df['weight'] = df['capital'] / total_capital * 100
        else:
            df['weight'] = 0
        
        return df
    
    def get_trade_summary(self) -> pd.DataFrame:
        """
        获取交易摘要
        
        Returns:
            交易摘要DataFrame
        """
        if not self.trades:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(self.trades)
        
        return df
    
    def get_performance_metrics(self) -> Dict:
        """
        计算绩效指标
        
        Returns:
            绩效指标字典
        """
        if not self.trades:
            return {}
        
        # 转换为DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # 筛选卖出交易
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        if sell_trades.empty:
            return {}
        
        # 计算总收益
        total_profit = sell_trades['profit'].sum()
        total_profit_pct = total_profit / self.initial_capital * 100
        
        # 计算胜率
        winning_trades = sell_trades[sell_trades['profit'] > 0]
        win_rate = len(winning_trades) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
        
        # 计算平均收益
        avg_profit = sell_trades['profit'].mean()
        avg_profit_pct = sell_trades['profit_pct'].mean()
        
        # 计算最大收益和最大亏损
        max_profit = sell_trades['profit'].max()
        max_profit_pct = sell_trades['profit_pct'].max()
        max_loss = sell_trades['profit'].min()
        max_loss_pct = sell_trades['profit_pct'].min()
        
        # 计算平均持仓天数
        avg_hold_days = sell_trades['hold_days'].mean()
        
        # 计算夏普比率 (简化版)
        # 这里需要日收益率数据，暂时返回N/A
        
        metrics = {
            'total_profit': total_profit,
            'total_profit_pct': total_profit_pct,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_profit_pct': avg_profit_pct,
            'max_profit': max_profit,
            'max_profit_pct': max_profit_pct,
            'max_loss': max_loss,
            'max_loss_pct': max_loss_pct,
            'avg_hold_days': avg_hold_days,
            'total_trades': len(sell_trades)
        }
        
        return metrics
    
    def check_partial_profit_conditions(self, ts_code: str, current_price: float) -> Tuple[bool, float, str]:
        """
        检查部分止盈条件 (用于激进策略)
        
        Args:
            ts_code: 股票代码
            current_price: 当前价格
            
        Returns:
            (是否部分平仓, 平仓比例, 平仓原因)
        """
        # 检查是否持仓
        if ts_code not in self.positions:
            return False, 0, ''
        
        # 获取当前策略ID
        strategy_id = self.config.get('active_strategy', 'default')
        
        # 只有激进策略才使用部分止盈
        if strategy_id != 'aggressive':
            return False, 0, ''
        
        # 获取持仓信息
        position = self.positions[ts_code]
        
        # 初始化部分止盈记录
        if ts_code not in self.partial_profit_taken:
            self.partial_profit_taken[ts_code] = {'30%': False, '50%': False}
        
        # 计算当前收益率
        profit_pct = (current_price / position['entry_price'] - 1)
        
        # 检查50%止盈条件 (卖出一半)
        if profit_pct >= self.partial_take_profit_pct2 and not self.partial_profit_taken[ts_code]['50%']:
            return True, 0.5, 'partial_take_profit_50'
        
        # 检查30%止盈条件 (卖出1/3)
        if profit_pct >= self.partial_take_profit_pct1 and not self.partial_profit_taken[ts_code]['30%']:
            return True, 1/3, 'partial_take_profit_30'
        
        return False, 0, ''
    
    def reset(self):
        """重置风险管理器"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.partial_profit_taken = {}
        logger.info("风险管理器已重置")
