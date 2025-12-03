
# -*- coding: utf-8 -*-
"""
专业级量化交易策略 - 优化版 v1.0
==================================
目标：年化收益率200%+，最大回撤<10%，适配JoinQuant平台
核心特性：
1. 三层风险控制体系（指数熔断+回撤监控+个股止损）
2. 智能择时系统（多信号融合）
3. 动态仓位管理（根据市场环境自适应）
4. 完全避免未来函数
5. 多策略组合（进攻/平衡/防御）

作者：基于9个策略深度分析优化
最后更新：2025-12-02
"""

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import math
from datetime import timedelta, datetime

# ==================== 全局配置 ====================

def initialize(context):
    """策略初始化"""
    # 基础设置
    set_option('avoid_future_data', True)  # 严格避免未来函数
    set_benchmark('000300.XSHG')  # 沪深300基准
    set_option('use_real_price', True)
    
    # 交易成本设置
    set_slippage(FixedSlippage(0.002))  # 0.2%滑点
    set_order_cost(OrderCost(
        open_tax=0, 
        close_tax=0.001,
        open_commission=0.0003, 
        close_commission=0.0003,
        close_today_commission=0, 
        min_commission=5
    ), type='stock')
    
    # 日志设置
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    
    # ========== 核心参数配置 ==========
    
    # 风险控制参数
    g.max_drawdown_threshold = 0.08  # 最大回撤阈值8%（留2%安全边际）
    g.index_fuse_level_1 = 0.05      # 指数熔断一级：5%单日跌幅
    g.index_fuse_level_2 = 0.08      # 指数熔断二级：8%单日跌幅（强制清仓）
    g.emergency_position_ratio = 0.30 # 一级熔断后仓位比例
    
    # 仓位配置（根据市场环境动态调整）
    g.position_config = {
        'STRONG': [0.50, 0.35, 0.15],   # 强势市场：进攻/平衡/防御
        'NEUTRAL': [0.35, 0.45, 0.20],  # 中性市场
        'WEAK': [0.15, 0.35, 0.50]      # 弱势市场
    }
    
    # 策略参数
    g.aggressive_stock_num = 3   # 进攻型持股数量
    g.balanced_stock_num = 3     # 平衡型持股数量
    g.defensive_etf_num = 2      # 防御型ETF数量
    
    # 止损止盈参数
    g.stock_stop_loss = 0.08     # 个股止损8%
    g.stock_take_profit = 1.00   # 个股止盈100%
    g.trailing_stop = 0.15       # 移动止损：从最高点回撤15%
    
    # 择时参数
    g.kdj_params = {'N': 27, 'M1': 9, 'M2': 9}  # KDJ参数（长周期）
    g.rsrs_params = {'N': 18, 'M': 600}         # RSRS参数
    g.ma_period = 20                             # 均线周期
    
    # ETF池（防御和轮动）
    g.defensive_etf_pool = [
        '518880.XSHG',  # 黄金ETF
        '511880.XSHG',  # 银华日利（货币）
        '511090.XSHG',  # 30年国债
    ]
    
    g.rotation_etf_pool = [
        '510180.XSHG',  # 上证180ETF
        '513030.XSHG',  # 德国DAX ETF
        '513100.XSHG',  # 纳指ETF
        '518880.XSHG',  # 黄金ETF
        '510410.XSHG',  # 资源ETF
        '513520.XSHG',  # 日经225ETF
        '159915.XSHE',  # 创业板ETF
        '512480.XSHG',  # 半导体ETF
    ]
    
    # ========== 全局变量初始化 ==========
    
    g.market_status = 'NEUTRAL'           # 市场状态
    g.current_position_ratio = 1.0        # 当前仓位比例
    g.historical_high = context.portfolio.total_value  # 历史最高净值
    g.max_profit_dict = {}                # 记录每只股票的最高收益
    g.hold_days_dict = {}                 # 持仓天数记录
    
    # 策略持仓分类
    g.aggressive_holdings = []   # 进攻型持仓
    g.balanced_holdings = []     # 平衡型持仓
    g.defensive_holdings = []    # 防御型持仓
    
    # 择时信号历史
    g.slope_series = []          # RSRS斜率序列
    g.last_kdj_signal = 'KEEP'   # 上次KDJ信号
    
    # 风控记录
    g.sold_today = set()         # 当日已卖出股票（避免重复交易）
    g.emergency_mode = False     # 紧急模式标志
    
    # ========== 定时任务设置 ==========
    
    # 盘前准备
    run_daily(before_market_open, 'before_open')
    
    # 风险监控（分钟级）
    run_daily(monitor_index_fuse, '9:31')
    run_daily(monitor_index_fuse, '10:00')
    run_daily(monitor_index_fuse, '11:00')
    run_daily(monitor_index_fuse, '13:30')
    run_daily(monitor_index_fuse, '14:00')
    run_daily(monitor_index_fuse, '14:30')
    
    # 回撤监控
    run_daily(monitor_drawdown, '10:15')
    run_daily(monitor_drawdown, '14:15')
    
    # 交易执行
    run_weekly(weekly_rebalance, 1, '9:35')  # 每周一调仓
    
    # 止损止盈
    run_daily(check_stop_loss, '10:30')
    run_daily(check_stop_loss, '14:45')
    
    # 收盘处理
    run_daily(after_market_close, 'after_close')
    
    log.info('=' * 60)
    log.info('策略初始化完成 - 优化版 v1.0')
    log.info(f'初始资金: {context.portfolio.total_value:.2f}')
    log.info('=' * 60)


# ==================== 核心功能模块 ====================

def before_market_open(context):
    """盘前准备：计算择时信号和市场状态"""
    log.info('=' * 60)
    log.info(f'交易日: {context.current_dt.date()}')
    
    # 清空当日卖出记录
    g.sold_today.clear()
    
    # 更新历史最高净值
    if context.portfolio.total_value > g.historical_high:
        g.historical_high = context.portfolio.total_value
    
    # 计算市场择时信号
    calculate_market_status(context)
    
    # 输出状态
    log.info(f'市场状态: {g.market_status}')
    log.info(f'当前仓位比例: {g.current_position_ratio:.1%}')
    log.info(f'账户总资产: {context.portfolio.total_value:.2f}')
    log.info(f'可用资金: {context.portfolio.available_cash:.2f}')
    log.info(f'持仓市值: {context.portfolio.positions_value:.2f}')
    
    # 如果处于紧急模式，检查是否可以解除
    if g.emergency_mode:
        check_emergency_mode_exit(context)


def monitor_index_fuse(context):
    """
    指数熔断监控（核心风控）
    实时监控主要指数的跌幅，触发自动降仓或清仓
    """
    # 获取主要指数列表
    indices = ['000300.XSHG', '000016.XSHG', '000905.XSHG']  # 沪深300、上证50、中证500
    
    try:
        # 获取指数当日开盘价和当前价（使用前一分钟数据避免未来函数）
        safe_time = context.current_dt - timedelta(minutes=1)
        
        for index in indices:
            # 获取今日开盘价
            open_data = attribute_history(index, 1, '1d', ['open'], skip_paused=True, df=False)
            if not open_data or 'open' not in open_data:
                continue
            open_price = open_data['open'][0]
            
            # 获取当前价（前一分钟收盘价）
            current_data = get_price(index, end_date=safe_time, frequency='1m', 
                                    fields=['close'], count=1, skip_paused=True)
            if current_data is None or current_data.empty:
                continue
            current_price = current_data['close'].iloc[-1]
            
            # 计算跌幅
            drop_ratio = (current_price - open_price) / open_price
            
            # 二级熔断：8%跌幅，强制清仓
            if drop_ratio < -g.index_fuse_level_2:
                log.info(f'⚠️⚠️⚠️ {index} 触发二级熔断！跌幅: {drop_ratio:.2%}')
                trigger_emergency_liquidation(context, f'{index}单日暴跌{abs(drop_ratio):.2%}')
                return
            
            # 一级熔断：5%跌幅，降仓至30%
            elif drop_ratio < -g.index_fuse_level_1:
                if g.current_position_ratio > g.emergency_position_ratio:
                    log.info(f'⚠️ {index} 触发一级熔断！跌幅: {drop_ratio:.2%}')
                    reduce_position_to(context, g.emergency_position_ratio, 
                                     f'{index}单日下跌{abs(drop_ratio):.2%}')
                    return
    
    except Exception as e:
        log.error(f'指数熔断监控异常: {str(e)}')


def monitor_drawdown(context):
    """
    回撤监控
    监控策略净值回撤，超过阈值触发降仓
    """
    current_value = context.portfolio.total_value
    drawdown = (current_value - g.historical_high) / g.historical_high
    
    # 记录回撤
    record(drawdown=drawdown * 100)
    
    # 硬阈值：8%回撤触发紧急止损
    if drawdown < -g.max_drawdown_threshold:
        log.info(f'🚨🚨🚨 触发最大回撤阈值！当前回撤: {drawdown:.2%}')
        trigger_emergency_liquidation(context, f'回撤达到{abs(drawdown):.2%}')
    
    # 软阈值：6%回撤触发降仓至50%
    elif drawdown < -0.06 and g.current_position_ratio > 0.5:
        log.info(f'⚠️ 回撤预警！当前回撤: {drawdown:.2%}，降低仓位')
        reduce_position_to(context, 0.5, f'回撤预警{abs(drawdown):.2%}')


def calculate_market_status(context):
    """
    计算市场状态（多信号融合择时）
    综合KDJ、RSRS、均线、成交量等多个指标判断市场环境
    """
    try:
        # 计算各个择时信号
        kdj_signal = calculate_kdj_signal(context)
        rsrs_signal = calculate_rsrs_signal(context)
        ma_signal = calculate_ma_signal(context)
        
        # 信号权重融合
        signal_scores = {
            'BUY': 1.0,
            'KEEP': 0.5,
            'SELL': 0.0
        }
        
        timing_score = (
            signal_scores.get(kdj_signal, 0.5) * 0.4 +   # KDJ权重40%
            signal_scores.get(rsrs_signal, 0.5) * 0.3 +  # RSRS权重30%
            signal_scores.get(ma_signal, 0.5) * 0.3      # MA权重30%
        )
        
        # 根据综合得分判断市场状态
        if timing_score >= 0.65:
            g.market_status = 'STRONG'
        elif timing_score >= 0.35:
            g.market_status = 'NEUTRAL'
        else:
            g.market_status = 'WEAK'
        
        log.info(f'择时信号 - KDJ:{kdj_signal}, RSRS:{rsrs_signal}, MA:{ma_signal}, 综合得分:{timing_score:.2f}')
        
    except Exception as e:
        log.error(f'市场状态计算异常: {str(e)}')
        g.market_status = 'NEUTRAL'  # 异常时默认中性


def calculate_kdj_signal(context, index='399101.XSHE'):
    """计算KDJ长周期择时信号"""
    try:
        N, M1, M2 = g.kdj_params['N'], g.kdj_params['M1'], g.kdj_params['M2']
        
        # 获取历史数据
        df = get_price(index, end_date=context.previous_date, count=N+20, 
                      frequency='daily', fields=['close', 'high', 'low'], 
                      skip_paused=True)
        
        if df is None or len(df) < N+3:
            return 'KEEP'
        
        # 计算RSV
        low_n = df['low'].rolling(window=N, min_periods=1).min()
        high_n = df['high'].rolling(window=N, min_periods=1).max()
        rsv = ((df['close'] - low_n) / (high_n - low_n) * 100).fillna(50)
        
        # 计算K值
        k_values = pd.Series(index=rsv.index, dtype=float)
        k_values.iloc[0] = 50
        for i in range(1, len(rsv)):
            k_values.iloc[i] = (k_values.iloc[i-1] * (M1-1) + rsv.iloc[i]) / M1
        
        # 计算D值
        d_values = pd.Series(index=k_values.index, dtype=float)
        d_values.iloc[0] = 50
        for i in range(1, len(k_values)):
            d_values.iloc[i] = (d_values.iloc[i-1] * (M2-1) + k_values.iloc[i]) / M2
        
        # 计算J值
        j_values = 3 * k_values - 2 * d_values
        
        if len(j_values) < 2:
            return 'KEEP'
        
        last_j = j_values.iloc[-1]
        
        # 信号判断
        if last_j < 0:  # 超卖
            return 'BUY'
        elif last_j > 70:  # 超买
            return 'SELL'
        else:
            return 'KEEP'
            
    except Exception as e:
        log.error(f'KDJ计算异常: {str(e)}')
        return 'KEEP'


def calculate_rsrs_signal(context, index='000300.XSHG'):
    """计算RSRS择时信号"""
    try:
        N, M = g.rsrs_params['N'], g.rsrs_params['M']
        
        # 获取数据
        df = attribute_history(index, N+20, '1d', ['high', 'low'], skip_paused=True)
        
        if df is None or len(df) < N:
            return 'KEEP'
        
        # 计算斜率
        x = df['low'].values[-N:]
        y = df['high'].values[-N:]
        slope = np.polyfit(x, y, 1)[0]
        
        # 更新斜率序列
        g.slope_series.append(slope)
        if len(g.slope_series) > M:
            g.slope_series = g.slope_series[-M:]
        
        # 计算R²
        y_pred = slope * x + np.polyfit(x, y, 1)[1]
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        
        # 计算Z-score
        if len(g.slope_series) >= M:
            mean = np.mean(g.slope_series)
            std = np.std(g.slope_series)
            if std > 0:
                z_score = (slope - mean) / std
                rsrs_score = z_score * r2
                
                if rsrs_score > 0.7:
                    return 'BUY'
                elif rsrs_score < -0.7:
                    return 'SELL'
        
        return 'KEEP'
        
    except Exception as e:
        log.error(f'RSRS计算异常: {str(e)}')
        return 'KEEP'


def calculate_ma_signal(context, index='000300.XSHG'):
    """计算均线信号"""
    try:
        df = attribute_history(index, g.ma_period+5, '1d', ['close'], skip_paused=True)
        
        if df is None or len(df) < g.ma_period:
            return 'KEEP'
        
        ma = df['close'].rolling(window=g.ma_period).mean()
        current_price = df['close'].iloc[-1]
        ma_value = ma.iloc[-1]
        
        # 价格在均线上方视为强势
        if current_price > ma_value * 1.02:
            return 'BUY'
        elif current_price < ma_value * 0.98:
            return 'SELL'
        else:
            return 'KEEP'
            
    except Exception as e:
        log.error(f'MA计算异常: {str(e)}')
        return 'KEEP'


def weekly_rebalance(context):
    """
    每周调仓
    根据市场状态动态分配资金到不同策略
    """
    if g.emergency_mode:
        log.info('处于紧急模式，暂停调仓')
        return
    
    log.info('=' * 60)
    log.info('开始每周调仓')
    
    # 获取目标仓位配置
    allocation = g.position_config[g.market_status]
    log.info(f'目标配置 - 进攻:{allocation[0]:.0%} 平衡:{allocation[1]:.0%} 防御:{allocation[2]:.0%}')
    
    # 计算各策略可用资金
    total_value = context.portfolio.total_value * g.current_position_ratio
    aggressive_value = total_value * allocation[0]
    balanced_value = total_value * allocation[1]
    defensive_value = total_value * allocation[2]
    
    # 执行各策略调仓
    rebalance_aggressive_strategy(context, aggressive_value)
    rebalance_balanced_strategy(context, balanced_value)
    rebalance_defensive_strategy(context, defensive_value)
    
    log.info('调仓完成')
    log.info('=' * 60)


def rebalance_aggressive_strategy(context, target_value):
    """
    进攻型策略调仓
    使用涨停基因+小市值逻辑
    """
    if target_value < 1000:
        # 清仓进攻型持仓
        for stock in g.aggressive_holdings[:]:
            close_position(context, stock, '进攻型策略资金不足')
        return
    
    # 选股：小市值+涨停基因
    stock_list = select_aggressive_stocks(context)
    
    if not stock_list:
        log.info('进攻型策略：无合适标的')
        return
    
    # 调仓：卖出不在目标列表的股票
    for stock in g.aggressive_holdings[:]:
        if stock not in stock_list:
            close_position(context, stock, '进攻型调出')
    
    # 买入新标的
    target_stocks = stock_list[:g.aggressive_stock_num]
    value_per_stock = target_value / g.aggressive_stock_num
    
    for stock in target_stocks:
        if stock not in g.aggressive_holdings:
            open_position(context, stock, value_per_stock, 'aggressive')


def rebalance_balanced_strategy(context, target_value):
    """
    平衡型策略调仓
    使用优化小市值逻辑（国九条+财务质量）
    """
    if target_value < 1000:
        for stock in g.balanced_holdings[:]:
            close_position(context, stock, '平衡型策略资金不足')
        return
    
    # 选股：质量小市值
    stock_list = select_balanced_stocks(context)
    
    if not stock_list:
        log.info('平衡型策略：无合适标的')
        return
    
    # 调仓
    for stock in g.balanced_holdings[:]:
        if stock not in stock_list:
            close_position(context, stock, '平衡型调出')
    
    # 买入
    target_stocks = stock_list[:g.balanced_stock_num]
    value_per_stock = target_value / g.balanced_stock_num
    
    for stock in target_stocks:
        if stock not in g.balanced_holdings:
            open_position(context, stock, value_per_stock, 'balanced')


def rebalance_defensive_strategy(context, target_value):
    """
    防御型策略调仓
    使用ETF动量轮动
    """
    if target_value < 1000:
        for etf in g.defensive_holdings[:]:
            close_position(context, etf, '防御型策略资金不足')
        return
    
    # 选择最强ETF
    best_etfs = select_defensive_etfs(context)
    
    if not best_etfs:
        log.info('防御型策略：无合适ETF')
        return
    
    # 调仓
    for etf in g.defensive_holdings[:]:
        if etf not in best_etfs:
            close_position(context, etf, '防御型调出')
    
    # 买入
    value_per_etf = target_value / min(len(best_etfs), g.defensive_etf_num)
    
    for etf in best_etfs[:g.defensive_etf_num]:
        if etf not in g.defensive_holdings:
            open_position(context, etf, value_per_etf, 'defensive')


# ==================== 选股模块 ====================

def select_aggressive_stocks(context):
    """进攻型选股：小市值+涨停基因"""
    try:
        # 获取小市值股票池
        initial_list = get_index_stocks('399101.XSHE', date=context.previous_date)
        initial_list = filter_basic_stocks(context, initial_list)
        
        # 按市值排序
        q = query(valuation.code, valuation.market_cap).filter(
            valuation.code.in_(initial_list),
            valuation.market_cap.between(10, 100)
        ).order_by(valuation.market_cap.asc()).limit(200)
        
        df = get_fundamentals(q, date=context.previous_date)
        if df is None or df.empty:
            return []
        
        stock_list = df['code'].tolist()
        
        # 价格过滤（使用历史数据）
        prices = history(1, '1d', 'close', stock_list, df=False)
        stock_list = [s for s in stock_list if s in context.portfolio.positions or prices[s][-1] <= 50]
        
        return stock_list[:20]
        
    except Exception as e:
        log.error(f'进攻型选股异常: {str(e)}')
        return []


def select_balanced_stocks(context):
    """平衡型选股：国九条+财务质量"""
    try:
        initial_list = get_index_stocks('399101.XSHE', date=context.previous_date)
        initial_list = filter_basic_stocks(context, initial_list)
        
        # 财务筛选
        q = query(
            valuation.code,
            valuation.market_cap,
            income.net_profit,
            income.operating_revenue
        ).filter(
            valuation.code.in_(initial_list),
            valuation.market_cap.between(10, 100),
            income.net_profit > 2000000,
            income.operating_revenue > 1e8,
            indicator.roe > 0,
            indicator.roa > 0
        ).order_by(valuation.market_cap.asc()).limit(100)
        
        df = get_fundamentals(q, date=context.previous_date)
        if df is None or df.empty:
            return []
        
        stock_list = df['code'].tolist()
        
        # 价格过滤
        prices = history(1, '1d', 'close', stock_list, df=False)
        stock_list = [s for s in stock_list if s in context.portfolio.positions or prices[s][-1] <= 50]
        
        return stock_list[:20]
        
    except Exception as e:
        log.error(f'平衡型选股异常: {str(e)}')
        return []


def select_defensive_etfs(context):
    """防御型选股：ETF动量排序"""
    try:
        etf_scores = []
        
        for etf in g.rotation_etf_pool:
            try:
                # 获取近20日数据
                df = attribute_history(etf, 22, '1d', ['close'], skip_paused=True)
                if df is None or len(df) < 20:
                    continue
                
                # 计算动量得分（简化版）
                y = np.log(df['close'].values)
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                annualized_return = math.exp(slope * 250) - 1
                
                # 计算R²
                y_pred = slope * x + np.polyfit(x, y, 1)[1]
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                
                score = annualized_return * r2
                etf_scores.append((etf, score))
                
            except Exception:
                continue
        
        # 按得分排序
        etf_scores.sort(key=lambda x: x[1], reverse=True)
        return [etf for etf, score in etf_scores if score > 0][:5]
        
    except Exception as e:
        log.error(f'防御型选股异常: {str(e)}')
        return g.defensive_etf_pool[:2]


def filter_basic_stocks(context, stock_list):
    """基础过滤：ST、停牌、次新股等"""
    current_data = get_current_data()
    filtered = []
    
    for stock in stock_list:
        # ST过滤
        if current_data[stock].is_st or 'ST' in current_data[stock].name:
            continue
        # 退市股过滤
        if '退' in current_data[stock].name:
            continue
        # 停牌过滤
        if current_data[stock].paused:
            continue
        # 创业板、科创板、北交所过滤
        if stock.startswith('300') or stock.startswith('688') or stock.startswith('8') or stock.startswith('4'):
            continue
        # 次新股过滤（375天）
        start_date = get_security_info(stock).start_date
        if context.previous_date - start_date < timedelta(days=375):
            continue
        
        filtered.append(stock)
    
    return filtered


def check_stop_loss(context):
    """
    止损止盈检查
    三层止损机制：
    1. 个股止损（-8%）
    2. 个股止盈（+100%）
    3. 移动止损（从最高点回撤15%）
    """
    for stock in list(context.portfolio.positions.keys()):
        if stock in g.sold_today:
            continue
        
        position = context.portfolio.positions[stock]
        avg_cost = position.avg_cost
        
        # 获取当前价格（使用前一日收盘价避免未来函数）
        try:
            price_data = attribute_history(stock, 1, '1d', ['close'], skip_paused=True)
            if price_data is None or price_data.empty:
                continue
            current_price = price_data['close'].iloc[-1]
        except:
            continue
        
        # 计算盈亏比例
        profit_ratio = (current_price - avg_cost) / avg_cost
        
        # 更新最高收益记录
        if stock not in g.max_profit_dict:
            g.max_profit_dict[stock] = profit_ratio
        else:
            g.max_profit_dict[stock] = max(g.max_profit_dict[stock], profit_ratio)
        
        # 止损：-8%
        if profit_ratio < -g.stock_stop_loss:
            log.info(f'⚠️ {stock} 触发止损 ({profit_ratio:.2%})')
            close_position(context, stock, f'止损{profit_ratio:.2%}')
            continue
        
        # 止盈：+100%
        if profit_ratio > g.stock_take_profit:
            log.info(f'🎯 {stock} 触发止盈 ({profit_ratio:.2%})')
            close_position(context, stock, f'止盈{profit_ratio:.2%}')
            continue
        
        # 移动止损：收益>20%后，从最高点回撤15%
        if g.max_profit_dict[stock] > 0.20:
            max_profit = g.max_profit_dict[stock]
            trailing_threshold = max_profit * (1 - g.trailing_stop)
            if profit_ratio < trailing_threshold:
                log.info(f'📉 {stock} 触发移动止损 (最高{max_profit:.2%}→当前{profit_ratio:.2%})')
                close_position(context, stock, f'移动止损{profit_ratio:.2%}')
                continue


def open_position(context, security, value, strategy_type):
    """
    开仓操作
    Args:
        security: 股票/ETF代码
        value: 目标市值
        strategy_type: 策略类型（'aggressive'/'balanced'/'defensive'）
    """
    if value < 1000:
        return False
    
    # 检查是否在当日卖出列表中
    if security in g.sold_today:
        log.info(f'{security} 今日已卖出，不重复买入')
        return False
    
    try:
        # 执行下单
        order_target_value(security, value)
        
        # 更新持仓分类
        if strategy_type == 'aggressive':
            if security not in g.aggressive_holdings:
                g.aggressive_holdings.append(security)
        elif strategy_type == 'balanced':
            if security not in g.balanced_holdings:
                g.balanced_holdings.append(security)
        elif strategy_type == 'defensive':
            if security not in g.defensive_holdings:
                g.defensive_holdings.append(security)
        
        # 初始化持仓记录
        g.hold_days_dict[security] = 0
        g.max_profit_dict[security] = 0.0
        
        log.info(f'✅ 买入 [{strategy_type}] {security}, 目标市值: {value:.2f}')
        return True
        
    except Exception as e:
        log.error(f'买入失败 {security}: {str(e)}')
        return False


def close_position(context, security, reason=''):
    """
    平仓操作
    Args:
        security: 股票/ETF代码
        reason: 卖出原因
    """
    try:
        # 执行平仓
        order_target_value(security, 0)
        
        # 从持仓分类中移除
        if security in g.aggressive_holdings:
            g.aggressive_holdings.remove(security)
        if security in g.balanced_holdings:
            g.balanced_holdings.remove(security)
        if security in g.defensive_holdings:
            g.defensive_holdings.remove(security)
        
        # 清理记录
        if security in g.hold_days_dict:
            del g.hold_days_dict[security]
        if security in g.max_profit_dict:
            del g.max_profit_dict[security]
        
        # 加入当日卖出列表
        g.sold_today.add(security)
        
        log.info(f'🔴 卖出 {security}，原因: {reason}')
        return True
        
    except Exception as e:
        log.error(f'卖出失败 {security}: {str(e)}')
        return False


def trigger_emergency_liquidation(context, reason):
    """
    紧急清仓
    触发条件：指数暴跌8%或回撤超过10%
    """
    log.info('=' * 60)
    log.info(f'🚨🚨🚨 紧急清仓触发！原因: {reason}')
    log.info('=' * 60)
    
    # 设置紧急模式
    g.emergency_mode = True
    g.current_position_ratio = 0.0
    
    # 清空所有持仓
    for stock in list(context.portfolio.positions.keys()):
        close_position(context, stock, f'紧急清仓-{reason}')
    
    # 买入货币基金
    if context.portfolio.available_cash > 1000:
        try:
            order_target_value('511880.XSHG', context.portfolio.available_cash)
            log.info('已转入货币基金避险')
        except:
            log.warning('货币基金买入失败')


def reduce_position_to(context, target_ratio, reason):
    """
    降低仓位至目标比例
    Args:
        target_ratio: 目标仓位比例（0-1）
        reason: 降仓原因
    """
    log.info('=' * 60)
    log.info(f'⚠️ 降仓触发至{target_ratio:.0%}，原因: {reason}')
    log.info('=' * 60)
    
    current_position_value = context.portfolio.positions_value
    total_value = context.portfolio.total_value
    current_ratio = current_position_value / total_value
    
    if current_ratio <= target_ratio:
        log.info(f'当前仓位{current_ratio:.0%}已低于目标{target_ratio:.0%}，无需降仓')
        return
    
    # 计算需要卖出的金额
    reduce_value = current_position_value - total_value * target_ratio
    
    # 优先卖出亏损股和ETF
    positions_list = []
    for stock, pos in context.portfolio.positions.items():
        profit_ratio = (pos.price - pos.avg_cost) / pos.avg_cost
        positions_list.append((stock, pos.value, profit_ratio))
    
    # 按盈亏比排序（亏损的先卖）
    positions_list.sort(key=lambda x: x[2])
    
    sold_value = 0
    for stock, value, profit in positions_list:
        if sold_value >= reduce_value:
            break
        close_position(context, stock, f'降仓-{reason}')
        sold_value += value
    
    # 更新仓位比例
    g.current_position_ratio = target_ratio
    log.info(f'降仓完成，当前仓位: {target_ratio:.0%}')


def check_emergency_mode_exit(context):
    """
    检查是否可以退出紧急模式
    条件：市场状态转为NEUTRAL或STRONG，且连续3日无重大波动
    """
    if not g.emergency_mode:
        return
    
    # 检查市场状态
    if g.market_status == 'STRONG':
        log.info('市场转强，退出紧急模式')
        g.emergency_mode = False
        g.current_position_ratio = 0.7  # 恢复至70%仓位
        return
    elif g.market_status == 'NEUTRAL':
        # 需要额外确认
        try:
            # 检查近3日波动
            indices = ['000300.XSHG', '000016.XSHG']
            stable = True
            for index in indices:
                df = attribute_history(index, 3, '1d', ['close'], skip_paused=True)
                if df is None or len(df) < 3:
                    continue
                max_change = abs((df['close'].max() - df['close'].min()) / df['close'].min())
                if max_change > 0.05:  # 3日波动>5%
                    stable = False
                    break
            
            if stable:
                log.info('市场趋稳，退出紧急模式')
                g.emergency_mode = False
                g.current_position_ratio = 0.5  # 恢复至50%仓位
        except:
            pass


def after_market_close(context):
    """收盘后处理"""
    log.info('=' * 60)
    log.info(f'收盘总结 - {context.current_dt.date()}')
    log.info(f'账户总资产: {context.portfolio.total_value:.2f}')
    log.info(f'当日盈亏: {context.portfolio.total_value - context.portfolio.starting_cash:.2f}')
    log.info(f'当前仓位比例: {g.current_position_ratio:.0%}')
    log.info(f'市场状态: {g.market_status}')
    log.info(f'紧急模式: {g.emergency_mode}')
    
    # 持仓详情
    if context.portfolio.positions:
        log.info('持仓明细:')
        for stock, pos in context.portfolio.positions.items():
            profit = (pos.price - pos.avg_cost) / pos.avg_cost
            strategy_type = 'unknown'
            if stock in g.aggressive_holdings:
                strategy_type = 'aggressive'
            elif stock in g.balanced_holdings:
                strategy_type = 'balanced'
            elif stock in g.defensive_holdings:
                strategy_type = 'defensive'
            log.info(f'  [{strategy_type}] {stock}: {pos.total_amount}股, '
                    f'成本{pos.avg_cost:.2f}, 现价{pos.price:.2f}, '
                    f'盈亏{profit:.2%}')
    
    log.info('=' * 60)


# ==================== 初始化入口 ====================

# JoinQuant要求的初始化函数名
def initialize(context):
    """策略初始化入口"""
    initialize(context)
