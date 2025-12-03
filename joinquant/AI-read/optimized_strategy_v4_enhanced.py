# -*- coding: utf-8 -*-
"""
高性能量化策略 - v4.0 增强版
================================
基于v2.0成功框架，增强收益能力
目标：年化180-250%，最大回撤<10%

v2.0表现：年化53.82%，回撤9.31%，夏普2.754 ✅
v4.0改进：提高进攻性，增加交易频率，目标年化200%+

完全避免未来函数，JoinQuant实盘可用
"""

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import math
from datetime import timedelta

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
    
    # 日志
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    
    # ========== 核心参数（提高进攻性）==========
    
    # 仓位分配（更激进）
    g.small_cap_ratio = 0.70   # 小市值提高到70%（v2是60%）
    g.etf_ratio = 0.30         # ETF降至30%
    
    # 小市值参数（增加持股数）
    g.stock_num = 6            # 6只股票（v2是4只）
    g.min_market_cap = 8       # 扩大选股范围（v2是10）
    g.max_market_cap = 120     # 扩大到120亿（v2是100）
    g.max_stock_price = 100    # 提高价格上限（v2是50）
    
    # 风控参数（适度放宽以提高收益）
    g.stop_loss_ratio = 0.10   # 止损10%（v2是9%）
    g.stop_profit_ratio = 2.00 # 止盈200%（v2是150%）
    g.index_crash_threshold = 0.05
    
    # ETF池（增加品种）
    g.etf_pool = [
        '159915.XSHE',  # 创业板ETF
        '510500.XSHG',  # 中证500ETF
        '512480.XSHG',  # 半导体ETF
        '518880.XSHG',  # 黄金ETF
        '513100.XSHG',  # 纳指ETF
        '159985.XSHE',  # 豆粕ETF
        '510180.XSHG',  # 上证180ETF
        '513520.XSHG',  # 日经225ETF
        '159949.XSHE',  # 创业板50
    ]
    
    # 全局变量
    g.small_cap_holdings = []
    g.etf_holdings = []
    g.yesterday_limit_up = []
    g.historical_high = context.portfolio.total_value
    g.max_profit_dict = {}
    g.sold_today = set()
    g.emergency_mode = False
    
    # 定时任务（增加调仓频率）
    run_daily(before_trading, 'before_open')
    run_daily(check_index_crash, '9:31')
    run_daily(check_index_crash, '10:00')
    run_daily(check_index_crash, '14:00')
    run_daily(rebalance_small_cap, '9:35')       # 改为每日调仓（v2是每周）
    run_daily(rebalance_etf, '10:30')
    run_daily(check_stop_loss, '10:15')
    run_daily(check_stop_loss, '14:45')
    run_daily(check_limit_up_stocks, '14:30')
    run_daily(after_trading, 'after_close')
    
    log.info('策略v4.0增强版初始化完成')


def before_trading(context):
    """盘前准备"""
    g.sold_today.clear()
    
    # 更新历史最高
    if context.portfolio.total_value > g.historical_high:
        g.historical_high = context.portfolio.total_value
    
    # 计算回撤
    drawdown = (context.portfolio.total_value - g.historical_high) / g.historical_high
    
    log.info(f'日期:{context.current_dt.date()} 总资产:{context.portfolio.total_value:.0f} 回撤:{drawdown:.2%}')
    
    # 获取昨日涨停股
    all_holdings = g.small_cap_holdings + g.etf_holdings
    if all_holdings:
        df = get_price(all_holdings, end_date=context.previous_date,
                      frequency='daily', fields=['close', 'high_limit'],
                      count=1, panel=False, fill_paused=False)
        g.yesterday_limit_up = list(df[df['close'] == df['high_limit']].code)
    else:
        g.yesterday_limit_up = []


def check_index_crash(context):
    """指数暴跌监控（保留v2逻辑）"""
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
            
            # 获取当前价
            safe_time = context.current_dt - timedelta(minutes=1)
            df_current = get_price(index, end_date=safe_time, frequency='1m',
                                  fields=['close'], count=1)
            if df_current is None or df_current.empty:
                continue
            current_price = df_current['close'].iloc[-1]
            
            # 计算跌幅
            drop = (current_price - day_open) / day_open
            
            # 8%暴跌：清仓
            if drop < -0.08:
                log.info(f'🚨{index}暴跌{abs(drop):.2%}，强制清仓！')
                emergency_liquidation(context)
                return
            
            # 5%大跌：减仓50%
            elif drop < -g.index_crash_threshold:
                log.info(f'⚠️{index}大跌{abs(drop):.2%}，降仓')
                reduce_positions(context, 0.50)
                return
                
    except Exception as e:
        log.error(f'指数监控异常: {str(e)}')


def emergency_liquidation(context):
    """紧急清仓"""
    g.emergency_mode = True
    
    for stock in list(context.portfolio.positions.keys()):
        order_target_value(stock, 0)
        log.info(f'紧急清仓: {stock}')
    
    g.small_cap_holdings.clear()
    g.etf_holdings.clear()
    
    # 买入货币基金
    if context.portfolio.available_cash > 1000:
        order_target_value('511880.XSHG', context.portfolio.available_cash)


def reduce_positions(context, target_ratio):
    """降低仓位"""
    current_value = context.portfolio.positions_value
    total_value = context.portfolio.total_value
    
    if current_value / total_value <= target_ratio:
        return
    
    reduce_value = current_value - total_value * target_ratio
    
    # 按亏损排序
    positions = []
    for stock, pos in context.portfolio.positions.items():
        profit = (pos.price - pos.avg_cost) / pos.avg_cost
        positions.append((stock, pos.value, profit))
    
    positions.sort(key=lambda x: x[2])
    
    sold_value = 0
    for stock, value, profit in positions:
        if sold_value >= reduce_value:
            break
        order_target_value(stock, 0)
        sold_value += value
        
        if stock in g.small_cap_holdings:
            g.small_cap_holdings.remove(stock)
        if stock in g.etf_holdings:
            g.etf_holdings.remove(stock)


# ==================== 小市值策略（增强版）====================

def rebalance_small_cap(context):
    """
    小市值调仓（改为每日，提高灵活性）
    """
    if g.emergency_mode:
        # 检查退出紧急模式
        try:
            df = get_price('000300.XSHG', end_date=context.previous_date,
                          count=3, frequency='1d', fields=['close'])
            if len(df) >= 3:
                max_drop = min([(df['close'].iloc[i] - df['close'].iloc[i-1])/df['close'].iloc[i-1] 
                               for i in range(1, len(df))])
                if max_drop > -0.02:
                    log.info('市场企稳，退出紧急模式')
                    g.emergency_mode = False
                else:
                    return
        except:
            return
    
    # 选股
    target_stocks = select_small_cap_stocks(context)
    
    if not target_stocks:
        log.info('无合适小市值股票')
        return
    
    # 卖出不在目标列表的（保留昨日涨停）
    for stock in g.small_cap_holdings[:]:
        if stock not in target_stocks and stock not in g.yesterday_limit_up:
            order_target_value(stock, 0)
            g.small_cap_holdings.remove(stock)
            log.info(f'卖出小市值: {stock}')
    
    # 计算可买入金额
    target_value = context.portfolio.total_value * g.small_cap_ratio
    current_value = sum([context.portfolio.positions[s].value 
                        for s in g.small_cap_holdings 
                        if s in context.portfolio.positions])
    available_value = max(0, target_value - current_value)
    
    # 买入新股票
    buy_list = [s for s in target_stocks if s not in g.small_cap_holdings and s not in g.sold_today]
    
    if buy_list and available_value > 1000:
        can_buy_num = min(len(buy_list), g.stock_num - len(g.small_cap_holdings))
        if can_buy_num > 0:
            value_per_stock = available_value / can_buy_num
            bought = 0
            for stock in buy_list:
                if bought >= can_buy_num:
                    break
                if value_per_stock > 1000:
                    order_target_value(stock, value_per_stock)
                    g.small_cap_holdings.append(stock)
                    bought += 1
                    log.info(f'买入小市值: {stock}, 金额: {value_per_stock:.0f}')


def select_small_cap_stocks(context):
    """
    小市值选股（放宽条件，增加标的）
    """
    try:
        # 获取深证综指成分股
        initial_list = get_index_stocks('399101.XSHE', date=context.previous_date)
        initial_list = filter_basic_stocks(context, initial_list)
        
        # 财务筛选（放宽条件）
        q = query(
            valuation.code,
            valuation.market_cap,
            income.net_profit
        ).filter(
            valuation.code.in_(initial_list),
            valuation.market_cap.between(g.min_market_cap, g.max_market_cap),
            income.net_profit > 0,  # 仅要求净利润>0
            income.operating_revenue > 5e7  # 降低营收要求（v2是1e8）
        ).order_by(
            valuation.market_cap.asc()
        ).limit(150)  # 增加候选数量（v2是100）
        
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
                         or (s in prices and prices[s][-1] <= g.max_stock_price)]
        
        return stock_list[:g.stock_num * 3]  # 返回更多候选
        
    except Exception as e:
        log.error(f'小市值选股异常: {str(e)}')
        return []


# ==================== ETF轮动策略 ====================

def rebalance_etf(context):
    """ETF轮动策略"""
    if g.emergency_mode:
        return
    
    # 选择最佳ETF
    best_etf = select_best_etf(context)
    
    if not best_etf:
        # 无合适ETF，清空
        for etf in g.etf_holdings[:]:
            order_target_value(etf, 0)
            g.etf_holdings.remove(etf)
        return
    
    # 当前ETF
    current_etf = None
    for etf in g.etf_holdings:
        if etf in context.portfolio.positions:
            current_etf = etf
            break
    
    # 需要换仓
    if current_etf and current_etf != best_etf:
        order_target_value(current_etf, 0)
        g.etf_holdings.remove(current_etf)
        log.info(f'卖出ETF: {current_etf}')
        current_etf = None
    
    # 买入最佳ETF
    if not current_etf and best_etf not in g.sold_today:
        target_value = context.portfolio.total_value * g.etf_ratio
        if target_value > 1000:
            order_target_value(best_etf, target_value)
            if best_etf not in g.etf_holdings:
                g.etf_holdings.append(best_etf)
            log.info(f'买入ETF: {best_etf}, 金额: {target_value:.0f}')


def select_best_etf(context):
    """选择动量最强的ETF"""
    try:
        etf_scores = []
        
        for etf in g.etf_pool:
            try:
                df = attribute_history(etf, 22, '1d', ['close'], skip_paused=True)
                if df is None or len(df) < 20:
                    continue
                
                prices = df['close'].values
                
                # 过滤近3日暴跌的ETF
                if min(prices[-1]/prices[-2], prices[-2]/prices[-3]) < 0.95:
                    continue
                
                # 计算动量得分
                y = np.log(prices)
                x = np.arange(len(y))
                slope, intercept = np.polyfit(x, y, 1)
                
                # 年化收益
                annual_return = math.exp(slope * 250) - 1
                
                # R²
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                
                score = annual_return * r2
                
                if score > 0:  # 只选正收益的ETF
                    etf_scores.append((etf, score))
                    
            except Exception:
                continue
        
        if not etf_scores:
            return None
        
        etf_scores.sort(key=lambda x: x[1], reverse=True)
        return etf_scores[0][0]
        
    except Exception as e:
        log.error(f'ETF选择异常: {str(e)}')
        return None


# ==================== 风控模块 ====================

def check_stop_loss(context):
    """
    止损止盈检查
    使用持仓价格（pos.price）避免未来函数
    """
    for stock in list(context.portfolio.positions.keys()):
        if stock in g.sold_today:
            continue
        
        pos = context.portfolio.positions[stock]
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
        
        # 移动止损（收益>30%后启动）
        if g.max_profit_dict[stock] > 0.30:
            max_profit = g.max_profit_dict[stock]
            trailing_threshold = max_profit * 0.75  # 从最高点回撤25%
            if profit_ratio < trailing_threshold:
                order_target_value(stock, 0)
                log.info(f'移动止损: {stock}, 最高{max_profit:.2%}→当前{profit_ratio:.2%}')
                update_holdings_after_sell(stock)
                g.sold_today.add(stock)


def check_limit_up_stocks(context):
    """检查昨日涨停股今日是否打开"""
    if not g.yesterday_limit_up:
        return
    
    for stock in g.yesterday_limit_up[:]:
        if stock not in context.portfolio.positions:
            continue
        
        try:
            df = get_price(stock, end_date=context.current_dt, frequency='1m',
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
    基础过滤（放宽条件，允许创业板）
    """
    current_data = get_current_data()
    filtered = []
    
    for stock in stock_list:
        # ST/退市/停牌
        if (current_data[stock].is_st or 
            'ST' in current_data[stock].name or
            '退' in current_data[stock].name or
            current_data[stock].paused):
            continue
        
        # 仅排除科创板和北交所（保留创业板！）
        if stock.startswith('688') or stock.startswith('8') or stock.startswith('4'):
            continue
        
        # 次新股
        start_date = get_security_info(stock).start_date
        if context.previous_date - start_date < timedelta(days=375):
            continue
        
        filtered.append(stock)
    
    return filtered


def after_trading(context):
    """收盘总结"""
    log.info('=' * 50)
    log.info(f'收盘:{context.current_dt.date()}')
    log.info(f'总资产:{context.portfolio.total_value:.2f}')
    log.info(f'小市值持仓:{len(g.small_cap_holdings)}只')
    log.info(f'ETF持仓:{len(g.etf_holdings)}只')
    log.info(f'紧急模式:{g.emergency_mode}')
    
    # 退出紧急模式检查
    if g.emergency_mode:
        try:
            df = get_price('000300.XSHG', end_date=context.previous_date,
                          count=5, frequency='1d', fields=['close'])
            if len(df) >= 5:
                max_drop = min([(df['close'].iloc[i] - df['close'].iloc[i-1])/df['close'].iloc[i-1] 
                               for i in range(1, len(df))])
                if max_drop > -0.02:
                    log.info('市场企稳，退出紧急模式')
                    g.emergency_mode = False
        except:
            pass
    
    log.info('=' * 50)