import pandas as pd
import numpy as np
import matplotlib
# Set the backend to 'Agg' to avoid GUI dependencies
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import os
from datetime import datetime
import logging
import platform

# 检测WSL环境
is_wsl = False
if platform.system() == "Linux":
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                is_wsl = True
    except:
        pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger('Plotter')

# 记录环境信息
if is_wsl:
    logger.info("Detected WSL environment, using non-GUI 'Agg' backend for matplotlib")

# 全局变量，用于存储是否使用英文标签
USE_ENGLISH_LABELS = True

# 空行，移除全局字体设置代码

class Plotter:
    """
    绘图工具类
    负责可视化回测结果和生成图表
    """
    
    @staticmethod
    def check_font_for_chinese(font_name):
        """
        检查字体是否能渲染特定的中文字符
        
        Args:
            font_name: 字体名称
            
        Returns:
            bool: 是否支持中文
        """
        try:
            from matplotlib.font_manager import FontProperties
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            # 测试字符，包括常见的中文字符和特定的"投"字和"撤"字
            test_chars = "中文测试投资撤回"
            
            # 创建一个临时图形来测试字体
            fig = plt.figure(figsize=(1, 1))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            
            # 尝试使用指定字体渲染文本
            font_prop = FontProperties(family=font_name)
            ax.text(0.5, 0.5, test_chars, fontproperties=font_prop)
            
            # 渲染图形到内存
            canvas.draw()
            
            # 如果没有抛出异常，则认为字体可以渲染中文
            plt.close(fig)
            return True
        except Exception:
            return False
    
    def __init__(self, output_dir: str = 'results'):
        """
        初始化绘图工具
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置绘图风格
        sns.set_style('whitegrid')
        self.colors = sns.color_palette('Set1')
        
        # 设置中文字体支持
        self.setup_chinese_font()
        
    def setup_chinese_font(self):
        """设置中文字体支持"""
        global USE_ENGLISH_LABELS
        
        try:
            # 尝试多种支持中文的字体，按优先级排序
            cjk_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
                         'Droid Sans Fallback', 'Arial Unicode MS', 'DejaVu Sans']
            
            # 检查matplotlib可用的字体
            from matplotlib.font_manager import fontManager, FontProperties
            available_fonts = [f.name for f in fontManager.ttflist]
            
            # 找到第一个可用的CJK字体
            font_found = False
            for font in cjk_fonts:
                if font in available_fonts and self.check_font_for_chinese(font):
                    plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                    font_found = True
                    logger.info(f"使用字体 '{font}' 显示中文字符")
                    break
            
            # 默认使用英文标签，除非明确找到了支持所有测试字符的中文字体
            USE_ENGLISH_LABELS = True
            
            if font_found:
                # 即使找到了字体，也进行额外测试确保它能渲染所有需要的字符
                try:
                    # 创建一个临时图形来测试更多字符
                    fig = plt.figure(figsize=(1, 1))
                    ax = fig.add_subplot(111)
                    
                    # 测试更多可能出现的中文字符
                    test_text = "投资回撤分析交易策略收益率"
                    font_prop = FontProperties(family=plt.rcParams['font.sans-serif'][0])
                    ax.text(0.5, 0.5, test_text, fontproperties=font_prop)
                    
                    # 渲染到内存
                    fig.canvas.draw()
                    plt.close(fig)
                    
                    # 如果没有抛出异常，可以使用中文标签
                    USE_ENGLISH_LABELS = False
                    logger.info("中文字体测试通过，将使用中文标签")
                except Exception as e:
                    logger.warning(f"中文字体测试失败: {e}，将使用英文标签")
            else:
                # 如果没有找到理想的字体，尝试使用系统默认字体
                plt.rcParams['font.sans-serif'] = ['sans-serif']
                logger.warning("未找到支持中文的字体，将使用英文标签")
            
            plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
            
            # 额外设置，确保字体可以正确渲染
            plt.rcParams['font.family'] = 'sans-serif'
            
        except Exception as e:
            logger.warning(f"设置中文字体时出错: {e}，将使用英文标签")
            USE_ENGLISH_LABELS = True
    
    def plot_portfolio_performance(self, daily_performance: pd.DataFrame, 
                                 title: str = '策略表现', 
                                 save_path: str = None) -> plt.Figure:
        """
        绘制投资组合表现图
        
        Args:
            daily_performance: 每日表现DataFrame
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            matplotlib Figure对象
        """
        if daily_performance.empty:
            logger.warning("无法绘制投资组合表现图：数据为空")
            return None
        
        # 转换日期格式
        daily_performance['date'] = pd.to_datetime(daily_performance['date'])
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制投资组合价值和基准价值
        ax.plot(daily_performance['date'], daily_performance['portfolio_value'], 
               label='策略', color=self.colors[0], linewidth=2)
        ax.plot(daily_performance['date'], daily_performance['benchmark_value'], 
               label='基准', color=self.colors[1], linewidth=2, alpha=0.7)
        
        # 设置x轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加标题和标签
        ax.set_title(title, fontsize=15)
        
        # 根据字体支持情况选择中文或英文标签
        if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
        else:
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('价值', fontsize=12)
            
        ax.legend(fontsize=12)
        
        # 添加收益率标注
        if len(daily_performance) > 0:
            initial_value = daily_performance['portfolio_value'].iloc[0]
            final_value = daily_performance['portfolio_value'].iloc[-1]
            total_return = (final_value / initial_value - 1) * 100
            
            benchmark_initial = daily_performance['benchmark_value'].iloc[0]
            benchmark_final = daily_performance['benchmark_value'].iloc[-1]
            benchmark_return = (benchmark_final / benchmark_initial - 1) * 100
            
            # 根据字体支持情况选择中文或英文标签
            if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
                ax.annotate(f'Strategy Return: {total_return:.2f}%',
                          xy=(0.02, 0.95), xycoords='axes fraction',
                          fontsize=12, color=self.colors[0])
                ax.annotate(f'Benchmark Return: {benchmark_return:.2f}%',
                          xy=(0.02, 0.90), xycoords='axes fraction',
                          fontsize=12, color=self.colors[1])
                ax.annotate(f'Excess Return: {total_return - benchmark_return:.2f}%',
                          xy=(0.02, 0.85), xycoords='axes fraction',
                          fontsize=12, color='green' if total_return > benchmark_return else 'red')
            else:
                ax.annotate(f'策略收益率: {total_return:.2f}%',
                          xy=(0.02, 0.95), xycoords='axes fraction',
                          fontsize=12, color=self.colors[0])
                ax.annotate(f'基准收益率: {benchmark_return:.2f}%',
                          xy=(0.02, 0.90), xycoords='axes fraction',
                          fontsize=12, color=self.colors[1])
                ax.annotate(f'超额收益: {total_return - benchmark_return:.2f}%',
                          xy=(0.02, 0.85), xycoords='axes fraction',
                          fontsize=12, color='green' if total_return > benchmark_return else 'red')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
                logger.info(f"Portfolio performance chart saved to: {save_path}")
            else:
                logger.info(f"投资组合表现图已保存至: {save_path}")
        
        return fig
    
    def plot_drawdown(self, daily_performance: pd.DataFrame,
                    title: str = None,
                    save_path: str = None) -> plt.Figure:
        """
        绘制回撤图
        
        Args:
            daily_performance: 每日表现DataFrame
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            matplotlib Figure对象
        """
        # 根据字体支持情况选择中文或英文标题和警告
        if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
            default_title = 'Drawdown Analysis'
            empty_data_warning = "Cannot plot drawdown: data is empty"
        else:
            default_title = '回撤分析'
            empty_data_warning = "无法绘制回撤图：数据为空"
            
        # 使用提供的标题或默认标题
        title = title or default_title
            
        if daily_performance.empty:
            logger.warning(empty_data_warning)
            return None
        
        # 转换日期格式
        daily_performance['date'] = pd.to_datetime(daily_performance['date'])
        
        # 计算回撤
        # 计算策略回撤
        portfolio_cumulative = (1 + daily_performance['daily_return']).cumprod()
        portfolio_running_max = portfolio_cumulative.cummax()
        portfolio_drawdown = (portfolio_cumulative / portfolio_running_max - 1) * 100
        
        # 计算基准回撤
        benchmark_cumulative = (1 + daily_performance['benchmark_return']).cumprod()
        benchmark_running_max = benchmark_cumulative.cummax()
        benchmark_drawdown = (benchmark_cumulative / benchmark_running_max - 1) * 100
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 根据字体支持情况选择中文或英文标签
        if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
            strategy_label = 'Strategy Drawdown'
            benchmark_label = 'Benchmark Drawdown'
            x_label = 'Date'
            y_label = 'Drawdown (%)'
            strategy_max_dd_label = f'Strategy Max Drawdown: {portfolio_drawdown.min():.2f}%'
            benchmark_max_dd_label = f'Benchmark Max Drawdown: {benchmark_drawdown.min():.2f}%'
        else:
            strategy_label = '策略回撤'
            benchmark_label = '基准回撤'
            x_label = '日期'
            y_label = '回撤 (%)'
            strategy_max_dd_label = f'策略最大回撤: {portfolio_drawdown.min():.2f}%'
            benchmark_max_dd_label = f'基准最大回撤: {benchmark_drawdown.min():.2f}%'
            
        # 绘制回撤
        ax.fill_between(daily_performance['date'], 0, portfolio_drawdown,
                      color=self.colors[0], alpha=0.3, label=strategy_label)
        ax.plot(daily_performance['date'], portfolio_drawdown,
               color=self.colors[0], linewidth=1)
        
        ax.fill_between(daily_performance['date'], 0, benchmark_drawdown,
                      color=self.colors[1], alpha=0.3, label=benchmark_label)
        ax.plot(daily_performance['date'], benchmark_drawdown,
               color=self.colors[1], linewidth=1)
        
        # 设置x轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加标题和标签
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.legend(fontsize=12)
        
        # 添加最大回撤标注
        max_drawdown = portfolio_drawdown.min()
        benchmark_max_drawdown = benchmark_drawdown.min()
        
        ax.annotate(strategy_max_dd_label,
                  xy=(0.02, 0.15), xycoords='axes fraction',
                  fontsize=12, color=self.colors[0])
        ax.annotate(benchmark_max_dd_label,
                  xy=(0.02, 0.10), xycoords='axes fraction',
                  fontsize=12, color=self.colors[1])
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
                logger.info(f"Drawdown chart saved to: {save_path}")
            else:
                logger.info(f"回撤图已保存至: {save_path}")
        
        return fig
    
    def plot_trade_analysis(self, trades_df: pd.DataFrame,
                          title: str = None,
                          save_path: str = None) -> plt.Figure:
        """
        绘制交易分析图
        
        Args:
            trades_df: 交易DataFrame
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            matplotlib Figure对象
        """
        # 根据字体支持情况选择中文或英文标题和警告
        if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
            default_title = 'Trade Analysis'
            empty_data_warning = "Cannot plot trade analysis: data is empty"
        else:
            default_title = '交易分析'
            empty_data_warning = "无法绘制交易分析图：数据为空"
            
        # 使用提供的标题或默认标题
        title = title or default_title
            
        if trades_df.empty:
            logger.warning(empty_data_warning)
            return None
        
        # 筛选卖出交易
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        if sell_trades.empty:
            if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
                logger.warning("Cannot plot trade analysis: no sell trades")
            else:
                logger.warning("无法绘制交易分析图：没有卖出交易")
            return None
        
        # 根据字体支持情况选择中文或英文标签
        if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
            profit_dist_title = 'Profit Distribution'
            profit_label = 'Profit (%)'
            trade_count_label = 'Trade Count'
            hold_time_title = 'Holding Period Distribution'
            hold_days_label = 'Holding Days'
            profit_time_title = 'Profit vs Holding Period'
            exit_reason_title = 'Exit Reason Analysis'
        else:
            profit_dist_title = '收益率分布'
            profit_label = '收益率 (%)'
            trade_count_label = '交易次数'
            hold_time_title = '持仓时间分布'
            hold_days_label = '持仓天数'
            profit_time_title = '收益率与持仓时间关系'
            exit_reason_title = '平仓原因分析'
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 收益分布直方图
        sns.histplot(sell_trades['profit_pct'], bins=20, kde=True, ax=axes[0, 0], color=self.colors[0])
        axes[0, 0].axvline(x=0, color='red', linestyle='--')
        axes[0, 0].set_title(profit_dist_title, fontsize=14)
        axes[0, 0].set_xlabel(profit_label, fontsize=12)
        axes[0, 0].set_ylabel(trade_count_label, fontsize=12)
        
        # 2. 持仓时间分布
        sns.histplot(sell_trades['hold_days'], bins=20, kde=True, ax=axes[0, 1], color=self.colors[1])
        axes[0, 1].set_title(hold_time_title, fontsize=14)
        axes[0, 1].set_xlabel(hold_days_label, fontsize=12)
        axes[0, 1].set_ylabel(trade_count_label, fontsize=12)
        
        # 3. 收益率与持仓时间散点图
        sns.scatterplot(x='hold_days', y='profit_pct', data=sell_trades, ax=axes[1, 0],
                       hue='profit_pct', palette='RdYlGn', size='profit_pct', sizes=(20, 200),
                       legend=False)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title(profit_time_title, fontsize=14)
        axes[1, 0].set_xlabel(hold_days_label, fontsize=12)
        axes[1, 0].set_ylabel(profit_label, fontsize=12)
        
        # 4. 平仓原因分析
        reason_counts = sell_trades['reason'].value_counts()
        
        # 如果使用英文标签，尝试翻译平仓原因
        if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
            # 创建中英文对照字典
            reason_translation = {
                '止损': 'Stop Loss',
                '止盈': 'Take Profit',
                '趋势转变': 'Trend Reversal',
                '跌破EMA': 'Below EMA',
                '跌入云层': 'Into Cloud',
                '其他': 'Other'
            }
            
            # 尝试翻译标签
            translated_index = []
            for reason in reason_counts.index:
                translated_index.append(reason_translation.get(reason, reason))
            
            axes[1, 1].pie(reason_counts, labels=translated_index, autopct='%1.1f%%',
                         colors=sns.color_palette('Set2', len(reason_counts)))
        else:
            axes[1, 1].pie(reason_counts, labels=reason_counts.index, autopct='%1.1f%%',
                         colors=sns.color_palette('Set2', len(reason_counts)))
            
        axes[1, 1].set_title(exit_reason_title, fontsize=14)
        
        # 添加总标题
        plt.suptitle(title, fontsize=16)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
                logger.info(f"Trade analysis chart saved to: {save_path}")
            else:
                logger.info(f"交易分析图已保存至: {save_path}")
        
        return fig
    
    def plot_stock_signals(self, stock_data: pd.DataFrame, ts_code: str,
                         title: str = None,
                         save_path: str = None) -> plt.Figure:
        """
        绘制单只股票的信号和指标
        
        Args:
            stock_data: 股票数据DataFrame
            ts_code: 股票代码
            title: 图表标题，默认为股票代码
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            matplotlib Figure对象
        """
        # 根据字体支持情况选择中文或英文标签
        if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
            empty_data_warning = f"Cannot plot signal chart for stock {ts_code}: data is empty"
            default_title = f"Stock {ts_code} Signal Analysis"
            price_label = "Price"
            close_price_label = "Close Price"
            volume_label = "Volume"
            volume_ma_label = "Volume MA"
            date_label = "Date"
            buy_signal_label = "Buy Signal"
            sell_signal_label = "Sell Signal"
            lead_line1_label = "Lead Line 1"
            lead_line2_label = "Lead Line 2"
        else:
            empty_data_warning = f"无法绘制股票{ts_code}信号图：数据为空"
            default_title = f"股票 {ts_code} 信号分析"
            price_label = "价格"
            close_price_label = "收盘价"
            volume_label = "成交量"
            volume_ma_label = "成交量均线"
            date_label = "日期"
            buy_signal_label = "买入信号"
            sell_signal_label = "卖出信号"
            lead_line1_label = "先行带1"
            lead_line2_label = "先行带2"
            
        if stock_data.empty:
            logger.warning(empty_data_warning)
            return None
        
        # 设置标题
        if title is None:
            title = default_title
        
        # 转换日期格式
        stock_data['date'] = pd.to_datetime(stock_data['trade_date'])
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 1. 价格和指标
        ax1 = axes[0]
        
        # 绘制K线图
        ax1.plot(stock_data['date'], stock_data['close'], label=close_price_label, color='black')
        
        # 绘制EMA
        if 'ema144' in stock_data.columns:
            ax1.plot(stock_data['date'], stock_data['ema144'], label='EMA144',
                   color=self.colors[0], linewidth=1.5)
        
        # 绘制Ichimoku云层
        if 'lead_line1' in stock_data.columns and 'lead_line2' in stock_data.columns:
            ax1.plot(stock_data['date'], stock_data['lead_line1'], label=lead_line1_label,
                   color=self.colors[1], linewidth=1, alpha=0.7)
            ax1.plot(stock_data['date'], stock_data['lead_line2'], label=lead_line2_label,
                   color=self.colors[2], linewidth=1, alpha=0.7)
            
            # 填充云层
            ax1.fill_between(stock_data['date'], stock_data['lead_line1'], stock_data['lead_line2'],
                          where=stock_data['lead_line1'] >= stock_data['lead_line2'],
                          color='green', alpha=0.2)
            ax1.fill_between(stock_data['date'], stock_data['lead_line1'], stock_data['lead_line2'],
                          where=stock_data['lead_line1'] < stock_data['lead_line2'],
                          color='red', alpha=0.2)
        
        # 绘制买入信号
        if 'all_uptrend' in stock_data.columns and 'price_above_ema' in stock_data.columns and 'price_above_cloud' in stock_data.columns:
            buy_signals = stock_data[
                (stock_data['all_uptrend'] == 1) &
                (stock_data['price_above_ema'] == 1) &
                (stock_data['price_above_cloud'] == 1)
            ]
            
            if not buy_signals.empty:
                ax1.scatter(buy_signals['date'], buy_signals['close'],
                          marker='^', color='green', s=100, label=buy_signal_label)
        
        # 绘制卖出信号
        if 'all_uptrend' in stock_data.columns and 'price_above_ema' in stock_data.columns and 'price_above_cloud' in stock_data.columns:
            # 找出趋势转变的点
            stock_data['uptrend_change'] = stock_data['all_uptrend'].diff().fillna(0)
            
            sell_signals = stock_data[
                ((stock_data['price_above_ema'] == 0) |
                 (stock_data['price_above_cloud'] == 0) |
                 (stock_data['uptrend_change'] == -1))
            ]
            
            if not sell_signals.empty:
                ax1.scatter(sell_signals['date'], sell_signals['close'],
                          marker='v', color='red', s=100, label=sell_signal_label)
        
        # 设置x轴格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        # 添加网格和图例
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1.set_title(title, fontsize=15)
        ax1.set_ylabel(price_label, fontsize=12)
        
        # 2. 成交量
        ax2 = axes[1]
        
        # 绘制成交量柱状图
        volume_colors = np.where(stock_data['close'] >= stock_data['close'].shift(1), 'green', 'red')
        ax2.bar(stock_data['date'], stock_data['vol'], color=volume_colors, alpha=0.7)
        
        # 绘制成交量均线
        if 'volume_ma' in stock_data.columns:
            ax2.plot(stock_data['date'], stock_data['volume_ma'], color='blue', linewidth=1.5, label=volume_ma_label)
        
        # 设置x轴格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        # 添加网格和标签
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel(volume_label, fontsize=12)
        ax2.legend(loc='upper left')
        
        # 3. RSI
        ax3 = axes[2]
        
        # 绘制RSI
        if 'rsi' in stock_data.columns:
            ax3.plot(stock_data['date'], stock_data['rsi'], color=self.colors[3], linewidth=1.5)
            
            # 添加超买超卖线
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            
            # 设置y轴范围
            ax3.set_ylim(0, 100)
        
        # 设置x轴格式
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # 添加网格和标签
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel(date_label, fontsize=12)
        ax3.set_ylabel('RSI', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
                logger.info(f"Stock {ts_code} signal chart saved to: {save_path}")
            else:
                logger.info(f"股票{ts_code}信号图已保存至: {save_path}")
        
        return fig
    
    def plot_performance_metrics(self, metrics: Dict,
                              title: str = None,
                              save_path: str = None) -> plt.Figure:
        """
        绘制绩效指标图表
        
        Args:
            metrics: 绩效指标字典
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            matplotlib Figure对象
        """
        # 根据字体支持情况选择中文或英文标签
        if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
            default_title = 'Performance Metrics'
            empty_data_warning = "Cannot plot performance metrics: data is empty"
            value_label = "Value"
            metrics_labels = {
                'total_return': 'Total Return (%)',
                'annual_return': 'Annual Return (%)',
                'sharpe_ratio': 'Sharpe Ratio',
                'max_drawdown': 'Max Drawdown (%)',
                'win_rate': 'Win Rate (%)',
                'profit_loss_ratio': 'Profit/Loss Ratio',
                'avg_hold_days': 'Avg Holding Days',
                'total_trades': 'Total Trades'
            }
            save_message = f"Performance metrics chart saved to: {save_path}"
        else:
            default_title = '绩效指标'
            empty_data_warning = "无法绘制绩效指标图：数据为空"
            value_label = "值"
            metrics_labels = {
                'total_return': '总收益率 (%)',
                'annual_return': '年化收益率 (%)',
                'sharpe_ratio': '夏普比率',
                'max_drawdown': '最大回撤 (%)',
                'win_rate': '胜率 (%)',
                'profit_loss_ratio': '盈亏比',
                'avg_hold_days': '平均持仓天数',
                'total_trades': '总交易次数'
            }
            save_message = f"绩效指标图已保存至: {save_path}"
            
        # 使用提供的标题或默认标题
        title = title or default_title
            
        if not metrics:
            logger.warning(empty_data_warning)
            return None
        
        # 提取关键指标
        key_metrics = {
            metrics_labels['total_return']: metrics.get('total_return', 0),
            metrics_labels['annual_return']: metrics.get('annual_return', 0),
            metrics_labels['sharpe_ratio']: metrics.get('sharpe_ratio', 0),
            metrics_labels['max_drawdown']: metrics.get('max_drawdown', 0),
            metrics_labels['win_rate']: metrics.get('win_rate', 0),
            metrics_labels['profit_loss_ratio']: metrics.get('profit_loss_ratio', 0),
            metrics_labels['avg_hold_days']: metrics.get('avg_hold_days', 0),
            metrics_labels['total_trades']: metrics.get('total_trades', 0)
        }
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制条形图
        metrics_names = list(key_metrics.keys())
        metrics_values = list(key_metrics.values())
        
        # 确定条形图颜色
        colors = []
        for name, value in key_metrics.items():
            if '收益率' in name or '夏普比率' in name or '胜率' in name or '盈亏比' in name:
                colors.append('green' if value > 0 else 'red')
            elif '回撤' in name:
                colors.append('red')
            else:
                colors.append(self.colors[0])
        
        # 绘制条形图
        bars = ax.bar(metrics_names, metrics_values, color=colors)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')
        
        # 设置标题和标签
        ax.set_title(title, fontsize=15)
        ax.set_ylabel(value_label, fontsize=12)
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(save_message)
        
        return fig
        
    def plot_performance_comparison(self, daily_performance: pd.DataFrame,
                                  title: str = None,
                                  save_path: str = None) -> plt.Figure:
        """
        绘制策略与基准收益率对比直方图
        
        Args:
            daily_performance: 每日表现DataFrame
            title: 图表标题
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            matplotlib Figure对象
        """
        # 根据字体支持情况选择中文或英文标题和警告
        if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
            default_title = 'Strategy vs Benchmark Performance'
            empty_data_warning = "Cannot plot performance comparison: data is empty"
            strategy_label = "Strategy"
            benchmark_label = "Benchmark"
            return_label = "Return (%)"
            save_message = f"Performance comparison chart saved to: {save_path}"
        else:
            default_title = '策略与基准收益率对比'
            empty_data_warning = "无法绘制收益率对比图：数据为空"
            strategy_label = "策略"
            benchmark_label = "基准"
            return_label = "收益率 (%)"
            save_message = f"收益率对比图已保存至: {save_path}"
            
        # 使用提供的标题或默认标题
        title = title or default_title
            
        if daily_performance.empty:
            logger.warning(empty_data_warning)
            return None
        
        # 计算总收益率
        if len(daily_performance) > 0:
            initial_value = daily_performance['portfolio_value'].iloc[0]
            final_value = daily_performance['portfolio_value'].iloc[-1]
            total_return = (final_value / initial_value - 1) * 100
            
            benchmark_initial = daily_performance['benchmark_value'].iloc[0]
            benchmark_final = daily_performance['benchmark_value'].iloc[-1]
            benchmark_return = (benchmark_final / benchmark_initial - 1) * 100
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 创建数据
            returns = [total_return, benchmark_return]
            labels = [strategy_label, benchmark_label]
            colors = ['green' if r > 0 else 'red' for r in returns]
            
            # 绘制条形图
            bars = ax.bar(labels, returns, color=colors)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}%',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
            
            # 添加标题和标签
            ax.set_title(title, fontsize=15)
            ax.set_ylabel(return_label, fontsize=12)
            
            # 添加网格
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加零线
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(save_message)
            
            return fig
        
        return None
    
    def generate_report(self, backtest_results: Dict, output_dir: str = None) -> str:
        """
        生成回测报告
        
        Args:
            backtest_results: 回测结果字典
            output_dir: 输出目录，默认使用初始化时设置的目录
            
        Returns:
            报告保存路径
        """
        # 根据字体支持情况选择中文或英文标签
        if 'USE_ENGLISH_LABELS' in globals() and USE_ENGLISH_LABELS:
            empty_data_warning = "Cannot generate backtest report: data is empty"
            portfolio_title = "Portfolio Performance"
            drawdown_title = "Drawdown Analysis"
            comparison_title = "Strategy vs Benchmark Comparison"
            trade_title = "Trade Analysis"
            metrics_title = "Performance Metrics"
            report_message = f"Backtest report generated to directory: {output_dir}"
        else:
            empty_data_warning = "无法生成回测报告：数据为空"
            portfolio_title = "投资组合表现"
            drawdown_title = "回撤分析"
            comparison_title = "策略与基准收益率对比"
            trade_title = "交易分析"
            metrics_title = "绩效指标"
            report_message = f"回测报告已生成至目录: {output_dir}"
            
        if not backtest_results:
            logger.warning(empty_data_warning)
            return ""
        
        # 设置输出目录
        if output_dir is None:
            output_dir = self.output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 提取数据
        metrics = backtest_results.get('performance_metrics', {})
        daily_performance = backtest_results.get('daily_performance', pd.DataFrame())
        trade_summary = backtest_results.get('trade_summary', pd.DataFrame())
        
        # 生成图表
        if not daily_performance.empty:
            # 绘制投资组合表现图
            self.plot_portfolio_performance(
                daily_performance,
                title=portfolio_title,
                save_path=os.path.join(output_dir, f'portfolio_performance_{timestamp}.png')
            )
            
            # 绘制策略与基准收益率对比图
            self.plot_performance_comparison(
                daily_performance,
                title=comparison_title,
                save_path=os.path.join(output_dir, f'performance_comparison_{timestamp}.png')
            )
            
            # 绘制回撤图
            self.plot_drawdown(
                daily_performance,
                title=drawdown_title,
                save_path=os.path.join(output_dir, f'drawdown_{timestamp}.png')
            )
        
        if not trade_summary.empty:
            # 绘制交易分析图
            self.plot_trade_analysis(
                trade_summary,
                title=trade_title,
                save_path=os.path.join(output_dir, f'trade_analysis_{timestamp}.png')
            )
        
        # 绘制绩效指标图
        if metrics:
            self.plot_performance_metrics(
                metrics,
                title=metrics_title,
                save_path=os.path.join(output_dir, f'performance_metrics_{timestamp}.png')
            )
        
        logger.info(report_message)
        return output_dir
