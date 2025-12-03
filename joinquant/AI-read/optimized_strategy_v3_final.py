# -*- coding: utf-8 -*-
"""
高性能量化策略 - v3.0 最终版
================================
基于simple162策略核心逻辑，增强风险控制
目标：年化150-200%+，最大回撤<10%

核心改进：
1. 指数熔断机制（5%降仓，8%清仓）
2. 最大回撤控制（10%硬阈值）
3. 优化小市值选股
4. ETF动量轮动

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
    
    # ========== 策略参数 ==========
    
    # 仓位分配
    g.small_cap_ratio = 0.60   # 小市值60%
    g.etf_ratio = 0.40         # ETF 40%
    
    # 小市值参数
    g.stock_num = 4
    g.stop_loss = 0.09         # 止损9%
    g.stop_profit = 1.50       # 止盈150%
    
    # ETF参数
    g.etf_pool = [
        '159915.XSHE',  # 创业板
        '510500.XSHG',  # 中证500
        '512480.XSHG',  # 半导体
        '518880.XSHG',  # 黄金
        '513100.XSHG',  # 纳指
    ]
    
    # 风控参数
    g.max_drawdown = 0.10      # 最大回撤阈值10%
    g.index_fuse_5 = 0.05      # 指数5%熔断
    g.index_fuse_8 = 0.08      # 指数8%强制清仓
    
    # 全局变量
    g.holdings = []            # 持仓列表
    g.yesterday_hl = []        # 昨日涨停
    g.historical_high = context.portfolio.total_value
    g.emergency = False        # 紧急模式
    g.sold_today = set()
    
    # 定时任务
    run_daily(prepare, '9:05')
    run_daily(monitor_risk, '9:31')
    run_daily(monitor_risk, '10:00')
    run_daily(monitor_risk, '13:30')
    run_daily(monitor_risk, '14:00')
    run_weekly(adjust_small_cap, 2, '9:35')
    run_daily(adjust_etf, '10:30')
    run_daily(stop_loss_check, '10:15')
    run_daily(stop_loss_check, '14:40')
    run_daily(check_limit_up, '14:30')
    run_daily(summary, 'after_close')
    
    log.info('策略v3.0初始化完成')


def prepare(context):
    """盘前准备"""
    g.sold_today.clear()
    
    # 更新最高净值
    if context.portfolio.total_value > g.historical_high:
        g.historical_high = context.portfolio.total_value
    
    # 获取昨日涨停
    g.holdings = list(context.portfolio.positions.keys())
    if g.holdings:
        df = get_price(g.holdings, end_date=context.previous_date,
                      frequency='daily', fields=['close', 'high_limit'],
                      count=1, panel=False, fill_paused=False)
        g.yesterday_hl = list(df[df['close'] == df['high_limit']].code)
    else:
        g.yesterday_hl = []
    
    # 计算回撤
    dd = (context.portfolio.total_value - g.historical_high) / g.historical_high
    log.info(f'日期:{context.current_dt.date()} 总资产:{context.portfolio.total_value:.0f} 回撤:{dd:.2%}')


def monitor_risk(context):
    """风险监控：指数熔断+回撤控制"""
    if g.emergency:
        return
    
    # 回撤监控
    dd = (context.portfolio.total_value - g.historical_high) / g.historical_high
    if dd < -g.max_drawdown:
        log.info(f'🚨触发最大回撤{abs(dd):.2%}，清仓！')
        clear_all(context, '回撤超限')
        return
    
    # 指数熔断监控
    try:
        indices = ['000300.XSHG', '000016.XSHG']
        
        for idx in indices:
            # 今日开盘
            open_data = attribute_history(idx, 1, '1d', ['open'], df=False)
            if not open_data:
                continue
            day_open = open_data['open'][0]
            
            # 当前价（前1分钟）
            safe_time = context.current_dt - timedelta(minutes=1)
            current_df = get_price(idx, end_date=safe_time, frequency='1m',
                                  fields=['close'], count=1)
            if current_df is None or current_df.empty:
                continue
            
            current = current_df['close'].iloc[-1]
            drop = (current - day_open) / day_open
            
            # 8%暴跌：清仓
            if drop < -g.index_fuse_8:
                log.info(f'🚨{idx}暴跌{abs(drop):.2%}，强制清仓！')
                clear_all(context, f'{idx}暴跌')
                return
            
            # 5%大跌：减仓50%
            elif drop < -g.index_fuse_5:
                log.info(f'⚠️{idx}大跌{abs(drop):.2%}，减仓50%')
                reduce_half(context)
                return
                
    except Exception as e:
        log.error(f'风控监控异常:{str(e)}')


def clear_all(context, reason):
    """清仓所有持仓"""
    g.emergency = True
    
    for stock in list(context.portfolio.positions.keys()):
        order_target_value(stock, 0)
    
    g.holdings.clear()
    log.info(f'已清仓，原因:{reason}')
    
    # 买货币基金
    if context.portfolio.available_cash > 1000:
        order_target_value('511880.XSHG', context.portfolio.available_cash * 0.9)


def reduce_half(context):
    """减仓50%"""
    current_val = context.portfolio.positions_value
    target_val = context.portfolio.total_value * 0.5
    
    if current_val <= target_val:
        return
    
    # 计算需卖出金额
    reduce = current_val - target_val
    
    # 按亏损排序
    positions = [(s, p.value, (p.price-p.avg_cost)/p.avg_cost) 
                 for s, p in context.portfolio.positions.items()]
    positions.sort(key=lambda x: x[2])
    
    sold = 0
    for stock, val, pft in positions:
        if sold >= reduce:
            break
        order_target_value(stock, 0)
        sold += val
        if stock in g.holdings:
            g.holdings.remove(stock)


# ==================== 小市值策略 ====================

def adjust_small_cap(context):
    """小市值调仓（每周二）"""
    if g.emergency:
        # 检查是否退出紧急模式
        try:
            df = get_price('000300.XSHG', end_date=context.previous_date,
                          count=5, frequency='1d', fields=['close'])
            if len(df) >= 5:
                max_drop = min([(df['close'].iloc[i] - df['close'].iloc[i-1])/df['close'].iloc[i-1] 
                               for i in range(1, len(df))])
                if max_drop > -0.02:
                    log.info('市场企稳，退出紧急模式')
                    g.emergency = False
                else:
                    log.info('仍在紧急模式')
                    return
        except:
            return
    
    log.info('开始小市值调仓')
    
    # 选股
    stocks = get_small_cap_list(context)
    if not stocks:
        log.info('无合适股票')
        return
    
    # 当前小市值持仓
    sc_holdings = [s for s in g.holdings if s not in g.etf_pool]
    
    # 卖出（保留昨日涨停）
    for stock in sc_holdings[:]:
        if stock not in stocks and stock not in g.yesterday_hl:
            order_target_value(stock, 0)
            g.holdings.remove(stock)
            log.info(f'卖出:{stock}')
    
    # 买入
    target_val = context.portfolio.total_value * g.small_cap_ratio
    current_val = sum([context.portfolio.positions[s].value 
                      for s in sc_holdings if s in context.portfolio.positions])
    avail = max(0, target_val - current_val)
    
    buy_list = [s for s in stocks if s not in sc_holdings and s not in g.sold_today]
    
    if buy_list and avail > 1000:
        val_per = avail / min(len(buy_list), g.stock_num - len(sc_holdings))
        for stock in buy_list:
            if len(sc_holdings) >= g.stock_num:
                break
            if val_per > 1000:
                order_target_value(stock, val_per)
                g.holdings.append(stock)
                sc_holdings.append(stock)
                log.info(f'买入:{stock} 金额:{val_per:.0f}')


def get_small_cap_list(context):
    """
    小市值选股
    深证综指成分股，市值10-100亿，净利润>0，ROE>0，价格<50
    """
    try:
        initial = get_index_stocks('399101.XSHE', date=context.previous_date)
        initial = filter_stocks(context, initial)
        
        q = query(
            valuation.code,
            valuation.market_cap
        ).filter(
            valuation.code.in_(initial),
            valuation.market_cap.between(10, 100),
            income.net_profit > 0,
            indicator.roe > 0
        ).order_by(
            valuation.market_cap.asc()
        ).limit(80)
        
        df = get_fundamentals(q, date=context.previous_date)
        if df is None or df.empty:
            return []
        
        stocks = df['code'].tolist()
        
        # 价格过滤
        if stocks:
            prices = history(1, '1d', 'close', stocks, df=False)
            stocks = [s for s in stocks if prices[s][-1] <= 50]
        
        return stocks[:g.stock_num * 2]
        
    except Exception as e:
        log.error(f'选股异常:{str(e)}')
        return []


# ==================== ETF轮动 ====================

def adjust_etf(context):
    """ETF调仓（每日）"""
    if g.emergency:
        return
    
    best = get_best_etf(context)
    if not best:
        return
    
    # 当前ETF
    current = None
    for etf in g.etf_pool:
        if etf in context.portfolio.positions:
            current = etf
            break
    
    # 需要换仓
    if current and current != best:
        order_target_value(current, 0)
        if current in g.holdings:
            g.holdings.remove(current)
        log.info(f'卖出ETF:{current}')
        current = None
    
    # 买入
    if not current and best not in g.sold_today:
        val = context.portfolio.total_value * g.etf_ratio
        if val > 1000:
            order_target_value(best, val)
            if best not in g.holdings:
                g.holdings.append(best)
            log.info(f'买入ETF:{best} 金额:{val:.0f}')


def get_best_etf(context):
    """选择动量最强ETF"""
    try:
        scores = []
        for etf in g.etf_pool:
            try:
                df = attribute_history(etf, 22, '1d', ['close'])
                if df is None or len(df) < 20:
                    continue
                
                prices = df['close'].values
                
                # 过滤近3日大跌
                if min(prices[-1]/prices[-2], prices[-2]/prices[-3]) < 0.95:
                    continue
                
                # 计算动量
                y = np.log(prices)
                x = np.arange(len(y))
                slope, inter = np.polyfit(x, y, 1)
                
                # 年化收益
                ret = math.exp(slope * 250) - 1
                
                # R2
                y_pred = slope * x + inter
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                
                score = ret * r2
                scores.append((etf, score))
                
            except:
                continue
        
        if not scores:
            return None
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
        
    except Exception as e:
        log.error(f'ETF选择异常:{str(e)}')
        return None


# ==================== 风控 ====================

def stop_loss_check(context):
    """止损止盈"""
    for stock in list(context.portfolio.positions.keys()):
        if stock in g.sold_today:
            continue
        
        pos = context.portfolio.positions[stock]
        pft = (pos.price - pos.avg_cost) / pos.avg_cost
        
        # 止损
        if pft < -g.stop_loss:
            order_target_value(stock, 0)
            if stock in g.holdings:
                g.holdings.remove(stock)
            g.sold_today.add(stock)
            log.info(f'止损:{stock} {pft:.2%}')
        
        # 止盈
        elif pft > g.stop_profit:
            order_target_value(stock, 0)
            if stock in g.holdings:
                g.holdings.remove(stock)
            g.sold_today.add(stock)
            log.info(f'止盈:{stock} {pft:.2%}')


def check_limit_up(context):
    """检查昨日涨停今日打开"""
    if not g.yesterday_hl:
        return
    
    for stock in g.yesterday_hl:
        if stock not in context.portfolio.positions:
            continue
        
        try:
            df = get_price(stock, end_date=context.current_dt,
                          frequency='1m', fields=['close', 'high_limit'],
                          count=1, panel=False, fill_paused=True)
            
            if df.empty:
                continue
            
            if df['close'].iloc[0] < df['high_limit'].iloc[0]:
                order_target_value(stock, 0)
                if stock in g.holdings:
                    g.holdings.remove(stock)
                log.info(f'涨停打开:{stock}')
        except:
            pass


# ==================== 辅助函数 ====================

def filter_stocks(context, stock_list):
    """基础过滤"""
    current_data = get_current_data()
    result = []
    
    for stock in stock_list:
        # ST/退市/停牌
        if (current_data[stock].is_st or 
            'ST' in current_data[stock].name or
            '退' in current_data[stock].name or
            current_data[stock].paused):
            continue
        
        # 创业板/科创板/北交所
        if (stock.startswith('300') or stock.startswith('688') or 
            stock.startswith('8') or stock.startswith('4')):
            continue
        
        # 次新股
        if context.previous_date - get_security_info(stock).start_date < timedelta(days=375):
            continue
        
        result.append(stock)
    
    return result


def summary(context):
    """收盘总结"""
    log.info('=' * 50)
    log.info(f'收盘:{context.current_dt.date()}')
    log.info(f'总资产:{context.portfolio.total_value:.2f}')
    log.info(f'持仓数:{len(context.portfolio.positions)}')
    log.info(f'紧急模式:{g.emergency}')
    
    # 退出紧急模式检查
    if g.emergency:
        try:
            df = get_price('000300.XSHG', end_date=context.previous_date,
                          count=3, frequency='1d', fields=['close'])
            if len(df) >= 3:
                drops = [(df['close'].iloc[i] - df['close'].iloc[i-1])/df['close'].iloc[i-1]
                        for i in range(1, len(df))]
                if all(d > -0.03 for d in drops):
                    log.info('市场企稳，退出紧急模式')
                    g.emergency = False
        except:
            pass
    
    log.info('=' * 50)