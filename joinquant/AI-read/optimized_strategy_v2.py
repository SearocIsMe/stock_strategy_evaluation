# -*- coding: utf-8 -*-
"""
专业级量化交易策略 - v2.0 (修复版)
====================================
目标：年化收益率200%+，最大回撤<10%
基于simple162策略的核心逻辑优化
完全避免未来函数，JoinQuant平台实盘可用
"""

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import math
from datetime import timedelta

# ==================== 策略初始化 ====================

def initialize(context):
    """策略初始化"""
    # 基础设置
    set_option('avoid_future_data', True)
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    
    # 交易成本
    set_slippage(FixedSlippage(0.002))
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5
    ), type='stock')
    
    # 日志设置
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    
    # ========== 核心参数 ==========
    
    # 仓位配置（简化为两策略）
    g.small_cap_ratio = 0.60      # 小市值策略60%
    g.etf_rotation_ratio = 0.40   # ETF轮动40%
    
    # 小市值参数
    g.stock_num = 4              # 持股数量
    g.min_market_cap = 10        # 最小市值（亿）
    g.max_market_cap = 100       # 最大市值（亿）
    g.max_stock_price = 50       # 最高股价
    
    # 风控参数
    g.stop_loss_ratio = 0.09     # 个股止损9%
    g.stop_profit_ratio = 1.50   # 个股止盈150%
    g.index_crash_threshold = 0.05  # 指数暴跌阈值5%
    
    # ETF池
    g.etf_pool = [
        '159915.XSHE',  # 创业板ETF
        '510500.XSHG',  # 中证500ETF
        '512480.XSHG',  # 半导体ETF
        '518880.XSHG',  # 黄金ETF
        '513100.XSHG',  # 纳指ETF
        '159985.XSHE',  # 豆粕ETF
        '510180.XSHG',  # 上证180ETF
    ]
    
    # 全局变量
    g.small_cap_holdings = []
    g.etf_holdings = []
    g.yesterday_limit_up = []
    g.historical_high = context.portfolio.total_value
    g.max_profit_dict = {}
    g.sold_today = set()
    g.emergency_mode = False
    
    # 定时任务
    run_daily(before_trading, 'before_open')
    run_daily(check_index_crash, '9:31')
    run_daily(check_index_crash, '10:00')
    run_daily(check_index_crash, '14:00')
    run_weekly(rebalance_small_cap, 2, '9:35')   # 每周二调仓小市值
    run_daily(rebalance_etf, '10:30')            # 每日检查ETF
    run_daily(check_stop_loss, '10:15')
    run_daily(check_stop_loss, '14:45')
    run_daily(check_limit_up_stocks, '14:30')
    run_daily(after_trading, 'after_close')
    
    log.info('策略初始化完成 v2.0')


def before_trading(context):
    """盘前准备"""
    g.sold_today.clear()
    
    # 更新历史最高
    if context.portfolio.total_value > g.historical_high:
        g.historical_high = context.portfolio.total_value
    
    # 计算回撤
    drawdown = (context.portfolio.total_value - g.historical_high) / g.historical_high
    
    log.info('=' * 50)
    log.info(f'日期: {context.current_dt.date()}')
    log.info(f'总资产: {context.portfolio.total_value:.2f}')
    log.info(f'当前回撤: {drawdown:.2%}')
    log.info(f'紧急模式: {g.emergency_mode}')
    
    # 获取昨日涨停股
    if g.small_cap_holdings:
        df = get_price(g.small_cap_holdings, end_date=context.previous_date,
                      frequency='daily', fields=['close', 'high_limit'],
                      count=1, panel=False, fill_paused=False)
        g.yesterday_limit_up = list(df[df['close'] == df['high_limit']].code)
    else:
        g.yesterday_limit_up = []


def check_index_crash(context):
    """
    指数暴跌监控
    监控主要指数，5%跌幅触发降仓，8%触发清仓
    """
    if g.emergency_mode:
        return
    
    indices = ['000300.XSHG', '000016.XSHG', '000905.XSHG']
    
    try:
        for index in indices:
            # 获取今日开盘价
            df_today = attribute_history(index, 1, '1d', ['open'], skip_paused=True, df=False)
            if not df_today or 'open' not in df_today:
                continue
            day_open = df_today['open'][0]
            
            # 获取当前价（使用前一分钟避免未来函数）
            safe_time = context.current_dt - timedelta(minutes=1)
            df_current = get_price(index, end_date=safe_time, frequency='1m',
                                  fields=['close'], count=1)
            if df_current is None or df_current.empty:
                continue
            current_price = df_current['close'].iloc[-1]
            
            # 计算跌幅
            drop = (current_price - day_open) / day_open
            
            # 8%暴跌：强制清仓
            if drop < -0.08:
                log.info(f'🚨 {index} 暴跌{abs(drop):.2%}，强制清仓！')
                emergency_liquidation(context)
                return
            
            # 5%大跌：降仓至30%
            elif drop < -g.index_crash_threshold:
                log.info(f'⚠️ {index} 大跌{abs(drop):.2%}，降低仓位')
                reduce_positions(context, 0.30)
                return
                
    except Exception as e:
        log.error(f'指数监控异常: {str(e)}')


def emergency_liquidation(context):
    """紧急清仓"""
    g.emergency_mode = True
    
    # 清空所有持仓
    for stock in list(context.portfolio.positions.keys()):
        order_target_value(stock, 0)
        log.info(f'紧急清仓: {stock}')
    
    # 清空持仓记录
    g.small_cap_holdings.clear()
    g.etf_holdings.clear()
    
    # 买入货币基金
    if context.portfolio.available_cash > 1000:
        order_target_value('511880.XSHG', context.portfolio.available_cash * 0.95)


def reduce_positions(context, target_ratio):
    """降低仓位至目标比例"""
    current_value = context.portfolio.positions_value
    total_value = context.portfolio.total_value
    
    if current_value / total_value <= target_ratio:
        return
    
    # 需要卖出的金额
    reduce_value = current_value - total_value * target_ratio
    
    # 按亏损排序，优先卖亏损股
    positions = []
    for stock, pos in context.portfolio.positions.items():
        profit = (pos.price - pos.avg_cost) / pos.avg_cost
        positions.append((stock, pos.value, profit))
    
    positions.sort(key=lambda x: x[2])  # 亏损的在前
    
    sold_value = 0
    for stock, value, profit in positions:
        if sold_value >= reduce_value:
            break
        order_target_value(stock, 0)
        sold_value += value
        
        # 更新持仓列表
        if stock in g.small_cap_holdings:
            g.small_cap_holdings.remove(stock)
        if stock in g.etf_holdings:
            g.etf_holdings.remove(stock)


# ==================== 小市值策略 ====================

def rebalance_small_cap(context):
    """
    小市值策略调仓（每周二）
    核心逻辑：市值+财务质量筛选
    """
    if g.emergency_mode:
        log.info('紧急模式，暂停小市值调仓')
        return
    
    log.info('开始小市值选股调仓')
    
    # 选股
    target_stocks = select_small_cap_stocks(context)
    
    if not target_stocks:
        log.info('无合适小市值股票')
        return
    
    # 卖出不在目标列表的股票（保留昨日涨停）
    for stock in g.small_cap_holdings[:]:
        if stock not in target_stocks and stock not in g.yesterday_limit_up:
            order_target_value(stock, 0)
            g.small_cap_holdings.remove(stock)
            log.info(f'卖出小市值: {stock}')
    
    # 买入新股票
    target_value = context.portfolio.total_value * g.small_cap_ratio
    current_value = sum([context.portfolio.positions[s].value 
                        for s in g.small_cap_holdings 
                        if s in context.portfolio.positions])
    available_value = max(0, target_value - current_value)
    
    buy_list = [s for s in target_stocks if s not in g.small_cap_holdings]
    
    if buy_list and available_value > 1000:
        value_per_stock = available_value / len(buy_list)
        for stock in buy_list:
            if stock not in g.sold_today:
                order_target_value(stock, value_per_stock)
                g.small_cap_holdings.append(stock)
                log.info(f'买入小市值: {stock}, 金额: {value_per_stock:.2f}')
                if len(g.small_cap_holdings) >= g.stock_num:
                    break


def select_small_cap_stocks(context):
    """
    小市值选股
    筛选条件：市值10-100亿 + 净利润>0 + ROE>0 + 价格<50元
    """
    try:
        # 获取股票池
        initial_list = get_index_stocks('399101.XSHE', date=context.previous_date)
        initial_list = filter_basic_stocks(context, initial_list)
        
        # 财务筛选
        q = query(
            valuation.code,
            valuation.market_cap
        ).filter(
            valuation.code.in_(initial_list),
            valuation.market_cap.between(g.min_market_cap, g.max_market_cap),
            income.net_profit > 0,
            income.operating_revenue > 1e8,
            indicator.roe > 0
        ).order_by(
            valuation.market_cap.asc()
        ).limit(100)
        
        df = get_fundamentals(q, date=context.previous_date)
        
        if df is None or df.empty:
            log.info('财务筛选无结果')
            return []
        
        stock_list = df['code'].tolist()
        
        # 价格过滤
        if stock_list:
            prices = history(1, '1d', 'close', stock_list, df=False)
            stock_list = [s for s in stock_list 
                         if s in context.portfolio.positions 
                         or prices[s][-1] <= g.max_stock_price]
        
        return stock_list[:g.stock_num * 2]
        
    except Exception as e:
        log.error(f'小市值选股异常: {str(e)}')
        return []


# ==================== ETF轮动策略 ====================

def rebalance_etf(context):
    """
    ETF轮动策略
    每日检查，选择动量最强的ETF
    """
    if g.emergency_mode:
        return
    
    # 计算ETF动量得分
    best_etf = select_best_etf(context)
    
    if not best_etf:
        # 无合适ETF，清空ETF仓位
        for etf in g.etf_holdings[:]:
            order_target_value(etf, 0)
            g.etf_holdings.remove(etf)
        return
    
    # 当前持有的ETF
    current_etf = None
    for etf in g.etf_holdings:
        if etf in context.portfolio.positions:
            current_etf = etf
            break
    
    # 需要换仓
    if current_etf and current_etf != best_etf:
        order_target_value(current_etf, 0)
        g.etf_holdings.remove(current_etf)
        current_etf = None
        log.info(f'卖出ETF: {current_etf}')
    
    # 买入最佳ETF
    if not current_etf:
        target_value = context.portfolio.total_value * g.etf_rotation_ratio
        if target_value > 1000 and best_etf not in g.sold_today:
            order_target_value(best_etf, target_value)
            if best_etf not in g.etf_holdings:
                g.etf_holdings.append(best_etf)
            log.info(f'买入ETF: {best_etf}, 金额: {target_value:.2f}')


def select_best_etf(context):
    """
    选择动量最强的ETF
    计算20日动量得分，选择得分最高的
    """
    try:
        etf_scores = []
        
        for etf in g.etf_pool:
            try:
                df = attribute_history(etf, 22, '1d', ['close'], skip_paused=True)
                if df is None or len(df) < 20:
                    continue
                
                # 计算动量
                prices = df['close'].values
                y = np.log(prices)
                x = np.arange(len(y))
                
                # 线性回归
                slope, intercept = np.polyfit(x, y, 1)
                
                # 年化收益
                annual_return = math.exp(slope * 250) - 1
                
                # R平方
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                
                # 动量得分
                score = annual_return * r2
                
                # 过滤近3日大跌的ETF
                if min(prices[-1]/prices[-2], prices[-2]/prices[-3]) > 0.95:
                    etf_scores.append((etf, score))
                    
            except Exception as e:
                continue
        
        if not etf_scores:
            return None
        
        # 排序并返回最佳
        etf_scores.sort(key=lambda x: x[1], reverse=True)
        return etf_scores[0][0]
        
    except Exception as e:
        log.error(f'ETF选择异常: {str(e)}')
        return None


# ==================== 风控模块 ====================

def check_stop_loss(context):
    """
    止损止盈检查
    个股止损9%，止盈150%
    """
    for stock in list(context.portfolio.positions.keys()):
        if stock in g.sold_today:
            continue
        
        pos = context.portfolio.positions[stock]
        
        # 获取当前价（使用持仓价格，避免未来函数）
        current_price = pos.price
        avg_cost = pos.avg_cost
        profit_ratio = (current_price - avg_cost) / avg_cost
        
        # 更新最高收益
        if stock not in g.max_profit_dict:
            g.max_profit_dict[stock] = profit_ratio
        else:
            g.max_profit_dict[stock] = max(g.max_profit_dict[stock], profit_ratio)
        
        # 止损
        if profit_ratio < -g.stop_loss_ratio:
            order_target_value(stock, 0)
            log.info(f'止损: {stock}, 亏损{profit_ratio:.2%}')
            update_holdings_after_sell(stock)
            g.sold_today.add(stock)
            continue
        
        # 止盈
        if profit_ratio > g.stop_profit_ratio:
            order_target_value(stock, 0)
            log.info(f'止盈: {stock}, 盈利{profit_ratio:.2%}')
            update_holdings_after_sell(stock)
            g.sold_today.add(stock)
            continue
        
        # 移动止损（收益>30%后，从最高点回撤20%）
        if g.max_profit_dict[stock] > 0.30:
            max_profit = g.max_profit_dict[stock]
            if profit_ratio < max_profit * 0.80:  # 从最高点回撤20%
                order_target_value(stock, 0)
                log.info(f'移动止损: {stock}, 最高{max_profit:.2%}→当前{profit_ratio:.2%}')
                update_holdings_after_sell(stock)
                g.sold_today.add(stock)


def check_limit_up_stocks(context):
    """
    检查昨日涨停股今日是否打开
    如果打开涨停则卖出
    """
    if not g.yesterday_limit_up:
        return
    
    now_time = context.current_dt
    
    for stock in g.yesterday_limit_up[:]:
        if stock not in context.portfolio.positions:
            continue
        
        try:
            # 获取当前数据
            df = get_price(stock, end_date=now_time, frequency='1m',
                          fields=['close', 'high_limit'], count=1,
                          panel=False, fill_paused=True)
            
            if df.empty:
                continue
            
            current_price = df['close'].iloc[0]
            high_limit = df['high_limit'].iloc[0]
            
            if current_price < high_limit:
                order_target_value(stock, 0)
                log.info(f'涨停打开，卖出: {stock}')
                update_holdings_after_sell(stock)
            else:
                log.info(f'继续涨停: {stock}')
                
        except Exception as e:
            log.error(f'检查涨停异常 {stock}: {str(e)}')


def update_holdings_after_sell(stock):
    """卖出后更新持仓列表"""
    if stock in g.small_cap_holdings:
        g.small_cap_holdings.remove(stock)
    if stock in g.etf_holdings:
        g.etf_holdings.remove(stock)
    if stock in g.max_profit_dict:
        del g.max_profit_dict[stock]


# ==================== 辅助函数 ====================

def filter_basic_stocks(context, stock_list):
    """
    基础过滤
    排除：ST、停牌、次新股、科创板、创业板、北交所
    """
    current_data = get_current_data()
    filtered = []
    
    for stock in stock_list:
        # ST/退市
        if current_data[stock].is_st or 'ST' in current_data[stock].name or '退' in current_data[stock].name:
            continue
        # 停牌
        if current_data[stock].paused:
            continue
        # 科创板/创业板/北交所
        if stock.startswith('300') or stock.startswith('688') or stock.startswith('8') or stock.startswith('4'):
            continue
        # 次新股（上市<375天）
        start_date = get_security_info(stock).start_date
        if context.previous_date - start_date < timedelta(days=375):
            continue
        
        filtered.append(stock)
    
    return filtered


def after_trading(context):
    """收盘后处理"""
    log.info('=' * 50)
    log.info(f'收盘 - {context.current_dt.date()}')
    log.info(f'总资产: {context.portfolio.total_value:.2f}')
    log.info(f'小市值持仓: {len(g.small_cap_holdings)}只')
    log.info(f'ETF持仓: {len(g.etf_holdings)}只')
    
    # 检查是否可退出紧急模式
    if g.emergency_mode:
        # 检查是否连续3日无暴跌
        try:
            df = get_price('000300.XSHG', end_date=context.previous_date,
                          count=3, frequency='1d', fields=['close'])
            if len(df) >= 3:
                max_drop = 0
                for i in range(1, len(df)):
                    drop = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
                    max_drop = min(max_drop, drop)
                
                if max_drop > -0.03:  # 3日内单日最大跌幅<3%
                    log.info('市场企稳，退出紧急模式')
                    g.emergency_mode = False
        except:
            pass
    
    log.info('=' * 50)