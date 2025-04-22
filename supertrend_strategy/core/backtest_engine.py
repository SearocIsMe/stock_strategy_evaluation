import pandas as pd
import numpy as np
import logging
import yaml
import time
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
from tqdm import tqdm

from core.data_fetcher import DataFetcher
from core.signal_generator import SignalGenerator
from core.risk_manager import RiskManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger('BacktestEngine')

class BacktestEngine:
    """
    回测引擎类
    负责协调数据获取、信号生成和风险管理，执行回测
    """
    
    def __init__(self, config_path: str = 'config/strategy_params.yaml'):
        """
        初始化回测引擎
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.data_fetcher = DataFetcher(config_path)
        self.signal_generator = SignalGenerator(config_path)
        self.risk_manager = RiskManager(config_path)
        
        # 回测结果
        self.performance_data = []  # 每日绩效数据
        self.trade_history = []     # 交易历史
        self.daily_positions = {}   # 每日持仓
        
        # 回测参数
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.benchmark = self.config['evaluation']['benchmark']
    
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
    
    def run_backtest(self, start_date: str = None, end_date: str = None,
                    stock_list: List[str] = None, verbose: bool = True,
                    apply_fundamental_filter: bool = True) -> Dict:
        """
        运行回测
        
        Args:
            start_date: 开始日期，默认使用配置文件中的日期
            end_date: 结束日期，默认使用配置文件中的日期
            stock_list: 股票列表，默认使用所有A股
            verbose: 是否显示进度条
            apply_fundamental_filter: 是否应用基本面筛选，默认为True
            
        Returns:
            回测结果字典
        """
        # 设置日期范围
        self.start_date = start_date or self.config['data']['start_date']
        self.end_date = end_date or self.config['data']['end_date']
        
        # 重置风险管理器
        self.risk_manager.reset()
        
        # 清空回测结果
        self.performance_data = []
        self.trade_history = []
        self.daily_positions = {}
        
        # 获取股票列表
        if stock_list is None:
            logger.info("获取A股股票列表...")
            stock_list = self.data_fetcher.get_stock_list()
            
            # 应用基本面筛选
            if apply_fundamental_filter:
                if self.config.get('fundamental'):
                    logger.info("应用基本面筛选条件...")
                    # 传递回测结束日期，确保使用回测期间的基本面数据
                    stock_list = self.data_fetcher.filter_stocks_by_fundamental(stock_list, date=self.end_date)
                    logger.info(f"基本面筛选后剩余股票数量: {len(stock_list)}")
                else:
                    logger.warning("未找到基本面筛选参数配置，将使用所有股票")
        
        # 获取交易日历
        logger.info("获取交易日历...")
        trade_dates = self.data_fetcher.get_trade_dates(self.start_date, self.end_date)
        
        if not trade_dates:
            logger.error("未获取到交易日历，回测终止")
            return {}
            
        # 确保交易日期按升序排序（从旧到新）
        trade_dates = sorted(trade_dates)
        logger.info(f"交易日期已排序，第一个交易日: {trade_dates[0]}，最后一个交易日: {trade_dates[-1]}")
        
        # 获取基准指数数据
        logger.info(f"获取基准指数{self.benchmark}数据...")
        benchmark_data = self.data_fetcher.get_stock_data(self.benchmark, self.start_date, self.end_date)
        
        # 批量获取股票数据
        logger.info(f"获取{len(stock_list)}只股票的历史数据...")
        
        # 为了提高效率，可以分批获取
        batch_size = 50
        all_stock_data = {}
        
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i:i+batch_size]
            batch_data = self.data_fetcher.get_batch_stock_data(batch, self.start_date, self.end_date)
            all_stock_data.update(batch_data)
            
            if verbose:
                logger.info(f"已获取{min(i+batch_size, len(stock_list))}/{len(stock_list)}只股票数据")
        
        # 开始回测
        logger.info(f"开始回测，从{self.start_date}到{self.end_date}...")
        
        # 初始资金
        initial_capital = self.config['capital']['initial']
        logger.info(f"初始资金: {initial_capital:,.2f}元")
        
        # 记录每日表现
        daily_performance = []
        
        # 交易统计
        total_buy_trades = 0
        total_sell_trades = 0
        profitable_trades = 0
        losing_trades = 0
        
        # 遍历每个交易日
        total_days = len(trade_dates)
        logger.info(f"共 {total_days} 个交易日需要回测")
        
        iterator = tqdm(trade_dates) if verbose else trade_dates
        for i, date in enumerate(iterator):
            # 显示进度
            if i % 20 == 0 and not verbose:  # 如果使用tqdm则不需要额外显示进度
                logger.info(f"回测进度: {i}/{total_days} ({i/total_days*100:.1f}%)")
            
            # 生成当日信号 - 传递当前持仓信息
            signals = self.signal_generator.generate_signals(all_stock_data, date, self.risk_manager.positions)
            
            # 更新投资组合
            opened, closed = self.risk_manager.update_portfolio(date, all_stock_data, signals)
            
            # 记录交易
            for trade in opened + closed:
                self.trade_history.append(trade)
            
            # 统计交易
            total_buy_trades += len(opened)
            total_sell_trades += len(closed)
            
            # 记录买入交易
            if opened:
                buy_details = [f"{t['ts_code']}({t['entry_price']:.2f}元)" for t in opened]
                logger.info(f"日期 {date} 买入 {len(opened)} 只股票: {', '.join(buy_details)}")
            
            # 记录卖出交易并统计盈亏
            if closed:
                sell_details = []
                for t in closed:
                    profit_pct = t.get('profit_pct', 0)
                    if profit_pct > 0:
                        profitable_trades += 1
                    else:
                        losing_trades += 1
                    sell_details.append(f"{t['ts_code']}({profit_pct:+.2f}%)")
                
                logger.info(f"日期 {date} 卖出 {len(closed)} 只股票: {', '.join(sell_details)}")
            
            # 计算当日投资组合价值
            portfolio_value = self.risk_manager.get_portfolio_value(date, all_stock_data)
            
            # 计算基准指数表现
            benchmark_value = self._calculate_benchmark_value(benchmark_data, date, initial_capital)
            
            # 记录当日表现
            daily_perf = {
                'date': date,
                'portfolio_value': portfolio_value,
                'benchmark_value': benchmark_value,
                'cash': self.risk_manager.current_capital - sum(pos['capital'] for pos in self.risk_manager.positions.values()),
                'positions': len(self.risk_manager.positions),
                'daily_return': 0.0,  # 将在下一步计算
                'benchmark_return': 0.0  # 将在下一步计算
            }
            daily_performance.append(daily_perf)
            
            # 记录当日持仓
            self.daily_positions[date] = self.risk_manager.get_position_summary()
            
            # 每隔一段时间显示当前持仓和收益情况
            if i % 50 == 0 or i == total_days - 1:
                current_return = (portfolio_value / initial_capital - 1) * 100
                benchmark_current_return = (benchmark_value / initial_capital - 1) * 100
                logger.info(f"当前进度 {i+1}/{total_days} ({(i+1)/total_days*100:.1f}%)")
                logger.info(f"当前资产: {portfolio_value:,.2f}元, 收益率: {current_return:+.2f}%")
                logger.info(f"基准收益: {benchmark_current_return:+.2f}%, 超额收益: {current_return-benchmark_current_return:+.2f}%")
                logger.info(f"当前持仓: {len(self.risk_manager.positions)} 只股票, 现金: {self.risk_manager.current_capital:,.2f}元")
        
        # 计算每日收益率
        daily_performance_df = pd.DataFrame(daily_performance)
        if not daily_performance_df.empty and len(daily_performance_df) > 1:
            daily_performance_df['daily_return'] = daily_performance_df['portfolio_value'].pct_change()
            daily_performance_df['benchmark_return'] = daily_performance_df['benchmark_value'].pct_change()
            
            # 第一天的收益率设为0
            daily_performance_df.loc[0, 'daily_return'] = 0
            daily_performance_df.loc[0, 'benchmark_return'] = 0
        
        self.performance_data = daily_performance_df
        
        # 计算绩效指标
        performance_metrics = self._calculate_performance_metrics()
        
        # 获取交易摘要
        trade_summary = self.risk_manager.get_trade_summary()
        
        # 整合回测结果
        backtest_results = {
            'performance_metrics': performance_metrics,
            'daily_performance': daily_performance_df,
            'trade_summary': trade_summary,
            'final_positions': self.risk_manager.get_position_summary(),
            'trade_history': self.risk_manager.trades
        }
        
        # 统计交易结果
        win_rate = profitable_trades / total_sell_trades * 100 if total_sell_trades > 0 else 0
        
        logger.info("=" * 50)
        logger.info("回测完成")
        logger.info("=" * 50)
        logger.info(f"初始资金: {initial_capital:,.2f}元, 最终资金: {portfolio_value:,.2f}元")
        logger.info(f"总收益率: {(portfolio_value/initial_capital - 1)*100:+.2f}%")
        logger.info(f"基准收益率: {(benchmark_value/initial_capital - 1)*100:+.2f}%")
        logger.info(f"超额收益: {((portfolio_value/initial_capital) - (benchmark_value/initial_capital))*100:+.2f}%")
        logger.info("-" * 50)
        logger.info("交易统计:")
        logger.info(f"总交易次数: {total_buy_trades} 买入, {total_sell_trades} 卖出")
        logger.info(f"盈利交易: {profitable_trades} 次, 亏损交易: {losing_trades} 次")
        logger.info(f"胜率: {win_rate:.2f}%")
        
        # 显示绩效指标
        metrics = performance_metrics
        logger.info("-" * 50)
        logger.info("绩效指标:")
        logger.info(f"年化收益率: {metrics.get('annual_return', 0):+.2f}%")
        logger.info(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
        logger.info(f"波动率: {metrics.get('volatility', 0):.2f}%")
        logger.info(f"平均持仓天数: {metrics.get('avg_hold_days', 0):.1f}天")
        logger.info(f"平均收益率: {metrics.get('avg_profit_pct', 0):+.2f}%")
        
        # 显示基准指数详细信息
        if self.config['evaluation'].get('output_benchmark_details', False):
            logger.info("-" * 50)
            logger.info(f"基准指数 ({self.benchmark}) 详细信息:")
            logger.info(f"基准年化收益率: {metrics.get('benchmark_annual_return', 0):+.2f}%")
            logger.info(f"基准夏普比率: {metrics.get('benchmark_sharpe', 0):.2f}")
            logger.info(f"基准最大回撤: {metrics.get('benchmark_max_drawdown', 0):.2f}%")
            logger.info(f"基准波动率: {metrics.get('benchmark_volatility', 0):.2f}%")
            logger.info(f"相对基准胜率: {metrics.get('win_rate_vs_benchmark', 0):.2f}%")
            logger.info(f"Alpha: {metrics.get('annual_return', 0) - metrics.get('benchmark_annual_return', 0):+.2f}%")
            logger.info(f"Beta: {metrics.get('beta', 0):.2f}")
            
            # 计算相关系数
            if not self.performance_data.empty and len(self.performance_data) > 1:
                correlation = self.performance_data['daily_return'].corr(self.performance_data['benchmark_return'])
                logger.info(f"与基准相关系数: {correlation:.2f}")
        
        logger.info("=" * 50)
        
        return backtest_results
    
    def _calculate_benchmark_value(self, benchmark_data: pd.DataFrame, date: str, 
                                 initial_value: float) -> float:
        """
        计算基准指数价值
        
        Args:
            benchmark_data: 基准指数数据
            date: 当前日期
            initial_value: 初始价值
            
        Returns:
            基准指数当前价值
        """
        if benchmark_data.empty:
            return initial_value
        
        # 确保基准数据按日期升序排序
        benchmark_data = benchmark_data.sort_values('trade_date', ascending=True)
        
        # 获取基准指数在回测开始日的收盘价
        start_idx = benchmark_data[benchmark_data['trade_date'] == self.start_date].index
        if len(start_idx) == 0:
            start_price = benchmark_data['close'].iloc[0]
        else:
            start_price = benchmark_data.loc[start_idx[0], 'close']
        
        # 获取当前日期的收盘价
        current_idx = benchmark_data[benchmark_data['trade_date'] == date].index
        if len(current_idx) == 0:
            return initial_value
        
        current_price = benchmark_data.loc[current_idx[0], 'close']
        
        # 计算价值
        benchmark_value = initial_value * (current_price / start_price)
        
        return benchmark_value
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        计算绩效指标
        
        Returns:
            绩效指标字典
        """
        if self.performance_data.empty:
            return {}
        
        # 提取数据
        portfolio_values = self.performance_data['portfolio_value']
        benchmark_values = self.performance_data['benchmark_value']
        daily_returns = self.performance_data['daily_return']
        benchmark_returns = self.performance_data['benchmark_return']
        
        # 初始和最终价值
        initial_value = portfolio_values.iloc[0]
        final_value = portfolio_values.iloc[-1]
        
        # 总收益率
        total_return = (final_value / initial_value - 1) * 100
        benchmark_return = (benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1) * 100
        
        # 年化收益率
        days = len(self.performance_data)
        annual_return = ((1 + total_return/100) ** (252/days) - 1) * 100
        benchmark_annual_return = ((1 + benchmark_return/100) ** (252/days) - 1) * 100
        
        # 波动率
        volatility = daily_returns.std() * np.sqrt(252) * 100
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100
        
        # 夏普比率
        risk_free_rate = 0.03  # 假设无风险利率为3%
        sharpe_ratio = (annual_return/100 - risk_free_rate) / (volatility/100) if volatility != 0 else 0
        benchmark_sharpe = (benchmark_annual_return/100 - risk_free_rate) / (benchmark_volatility/100) if benchmark_volatility != 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # 基准最大回撤
        benchmark_cum_returns = (1 + benchmark_returns).cumprod()
        benchmark_running_max = benchmark_cum_returns.cummax()
        benchmark_drawdown = (benchmark_cum_returns / benchmark_running_max - 1) * 100
        benchmark_max_drawdown = benchmark_drawdown.min()
        
        # 胜率 (相对于基准)
        win_days = sum(daily_returns > benchmark_returns)
        win_rate = win_days / len(daily_returns) * 100 if len(daily_returns) > 0 else 0
        
        # 计算Beta (系统风险)
        beta = 0
        if benchmark_volatility != 0:
            # 计算协方差
            covariance = daily_returns.cov(benchmark_returns)
            # Beta = 协方差 / 基准方差
            beta = covariance / benchmark_returns.var()
        
        # 交易统计
        trade_metrics = self.risk_manager.get_performance_metrics()
        
        # 合并指标
        metrics = {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'annual_return': annual_return,
            'benchmark_annual_return': benchmark_annual_return,
            'volatility': volatility,
            'benchmark_volatility': benchmark_volatility,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'win_rate_vs_benchmark': win_rate,
            'beta': beta,
            'initial_value': initial_value,
            'final_value': final_value,
            'backtest_days': days
        }
        
        # 添加交易统计
        metrics.update(trade_metrics)
        
        return metrics
    
    def get_daily_signals(self, date: str, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        获取指定日期的所有股票信号
        
        Args:
            date: 日期
            stock_data: 股票数据字典
            
        Returns:
            信号DataFrame
        """
        # 生成信号
        signals = self.signal_generator.generate_signals(stock_data, date)
        
        # 转换为DataFrame
        signals_list = []
        for ts_code, data in signals.items():
            data['ts_code'] = ts_code
            signals_list.append(data)
        
        if not signals_list:
            return pd.DataFrame()
        
        signals_df = pd.DataFrame(signals_list)
        
        # 按评分排序
        if 'score' in signals_df.columns:
            signals_df = signals_df.sort_values('score', ascending=False)
        
        return signals_df
    
    def get_performance_summary(self) -> Dict:
        """
        获取绩效摘要
        
        Returns:
            绩效摘要字典
        """
        if self.performance_data.empty:
            return {}
        
        # 计算绩效指标
        metrics = self._calculate_performance_metrics()
        
        # 提取关键指标
        summary = {
            'total_return': metrics.get('total_return', 0),
            'annual_return': metrics.get('annual_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_trades': metrics.get('total_trades', 0),
            'avg_profit_pct': metrics.get('avg_profit_pct', 0),
            'benchmark_return': metrics.get('benchmark_return', 0),
            'alpha': metrics.get('annual_return', 0) - metrics.get('benchmark_annual_return', 0)
        }
        
        return summary
    
    def get_top_performing_trades(self, n: int = 10) -> pd.DataFrame:
        """
        获取表现最好的交易
        
        Args:
            n: 返回的交易数量
            
        Returns:
            交易DataFrame
        """
        trades_df = self.risk_manager.get_trade_summary()
        
        if trades_df.empty:
            return pd.DataFrame()
        
        # 筛选卖出交易
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        if sell_trades.empty:
            return pd.DataFrame()
        
        # 按收益率排序
        top_trades = sell_trades.sort_values('profit_pct', ascending=False).head(n)
        
        return top_trades
    
    def get_worst_performing_trades(self, n: int = 10) -> pd.DataFrame:
        """
        获取表现最差的交易
        
        Args:
            n: 返回的交易数量
            
        Returns:
            交易DataFrame
        """
        trades_df = self.risk_manager.get_trade_summary()
        
        if trades_df.empty:
            return pd.DataFrame()
        
        # 筛选卖出交易
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        if sell_trades.empty:
            return pd.DataFrame()
        
        # 按收益率排序
        worst_trades = sell_trades.sort_values('profit_pct', ascending=True).head(n)
        
        return worst_trades
