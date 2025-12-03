# -*- coding: utf-8 -*-
"""
高性能量化策略 - v6.0 平衡增强版
================================
目标：年化80-120%，最大回撤<10%

核心理念：质量优先的适度进攻
- 基于v2.0成功框架（53.82%年化，9.31%回撤）
- 避免v5.0的致命错误（盲目激进导致1.37%惨败）
- 在保持质量的前提下，适度提升收益能力

关键改进：
1. 持股数量：4只→6只（增加机会但保持质量）
2. 仓位配置：60%→70%（渐进提升）
3. 市值范围：10-100亿→8-120亿（略扩大但避开垃圾区）
4. 保持严格风控：9%止损、150%止盈
5. 保留质量筛选：ROE>5%、营收>5000万

完全避免未来函数
"""

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import math
from datetime import timedelta

def initialize(context):
    """策略初始化"""
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
    
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    
    # ========== v6核心参数（质量+适度进攻）==========
    
    # 仓位分配（适度提升）
    g.small_cap_ratio = 0.70   # 70%小市值（v2是60%，v5是85%）
    g.etf_ratio = 0.30         # 30% ETF（v2是40%，v5是15%）
    
    # 小市值参数（质量+机会平衡）
    g.stock_num = 6            # 6只股票（v2是4只，v5是8只）
    g.min_market_cap = 8       # 8亿起（避开5-10亿垃圾区！）
    g.max_market_cap = 120     # 120亿（略扩大选股池）
    g.max_stock_price = 60     # 60元（适度提高）
    
    # 风控参数（保持v2严格标准）
    g.stop_loss_ratio = 0.09        # 9%止损（v2标准，v5是12%）
    g.stop_profit_ratio = 1.50      # 150%止盈（v2标准，v5是300%）
    g.trailing_stop_ratio = 0.25    # 移动止损25%
    g.max_drawdown_limit = 0.10     # 10%最大回撤
    g.index_crash_threshold = 0.05  # 5%指数熔断（v2标准，v5是6%）
    
    # ETF池（保持多样化）
    g.etf_pool = [
        '512480.XSHG',  # 半导体ETF（高弹性）
        '159915.XSHE',  # 创业板ETF
        '159949.XSHE',  # 创业板50
        '510500.XSHG',  # 中证500ETF
        '513100.XSHG',  # 纳指ETF
        '518880.XSHG',  # 黄金ETF（避险）
    ]
    
    # 全局变量
    g.small_cap_holdings = []
    g.etf_holdings = []
    g.yesterday_limit_up = []
    g.historical_high = context.portfolio.total_value
    g.max_profit_dict = {}
    g.sold_today = set()
    g.emergency_mode = False
    
    # 定时任务（保持v2的稳健节奏）
    run_daily(before_trading, 'before_open')
    run_daily(check_index_crash, '9:31')
    run_daily(check_index_crash, '10:00')
    run_daily(check_index_crash, '14:00')
    run_weekly(rebalance_small_cap, 2, '9:35')  # 每周二调仓（v2节奏）
    run_daily(rebalance_etf, '10:30')
    run_daily(check_stop_loss, '10:15')
    run_daily(check_stop_loss, '14:45')
    run_daily(check_limit_up_stocks, '14:30')
    run_daily(after_trading, 'after_close')
    
    log.info('策略v6.0平衡增强版初始化完成')


def before_trading(context):
    """盘前准备"""
    g.sold_today.clear()
    
    if context.portfolio.total_value > g.historical_high:
        g.historical_high = context.portfolio.total_value
    
    drawdown = (context.portfolio.total_value - g.historical_high) / g.historical_high
    
    log.info(f'[{context.current_dt.date()}] 资产:{context.portfolio.total_value:.0f} 回撤:{drawdown:.2%}')
    
    # 获取昨日涨停
    all_holdings = g.small_cap_holdings + g.etf_holdings
    if all_holdings:
        df = get_price(all_holdings, end_date=context.previous_date,
                      frequency='daily', fields=['close', 'high_limit'],
                      count=1, panel=False, fill_paused=False)
        g.yesterday_limit_up = list(df[df['close'] == df['high_limit']].code)
    else:
        g.yesterday_limit_up = []


def check_index_crash(context):
    """指数监控（v2严格标准）"""
    if g.emergency_mode:
        return
    
    # 回撤监控
    dd = (context.portfolio.total_value - g.historical_high) / g.historical_high
    if dd < -g.max_drawdown_limit:
        log.info(f'🚨回撤{abs(dd):.2%}超限，清仓！')
        emergency_liquidation(context)
        return
    
    # 指数监控
    indices = ['000300.XSHG', '000016.XSHG', '000905.XSHG']
    
    try:
        for index in indices:
            df_open = attribute_history(index, 1, '1d', ['open'], df=False)
            if not df_open or 'open' not in df_open:
                continue
            day_open = df_open['open'][0]
            
            safe_time = context.current_dt - timedelta(minutes=1)
            df_cur = get_price(index, end_date=safe_time, frequency='1m',
                              fields=['close'], count=1)
            if df_cur is None or df_cur.empty:
                continue
            current = df_cur['close'].iloc[-1]
            
            drop = (current - day_open) / day_open
            
            # 8%暴跌：清仓
            if drop < -0.08:
                log.info(f'🚨{index}暴跌{abs(drop):.2%}！')
                emergency_liquidation(context)
                return
            
            # 5%大跌：减仓50%（v2标准）
            elif drop < -g.index_crash_threshold:
                log.info(f'⚠️{index}跌{abs(drop):.2%}，降仓')
                reduce_positions(context, 0.50)
                return
                
    except Exception as e:
        log.error(f'监控异常: {str(e)}')


def emergency_liquidation(context):
    """紧急清仓"""
    g.emergency_mode = True
    
    for stock in list(context.portfolio.positions.keys()):
        order_target_value(stock, 0)
    
    g.small_cap_holdings.clear()
    g.etf_holdings.clear()
    
    if context.portfolio.available_cash > 1000:
        order_target_value('511880.XSHG', context.portfolio.available_cash)


def reduce_positions(context, target_ratio):
    """降仓"""
    current_val = context.portfolio.positions_value
    total_val = context.portfolio.total_value
    
    if current_val / total_val <= target_ratio:
        return
    
    reduce_val = current_val - total_val * target_ratio
    
    positions = [(s, p.value, (p.price-p.avg_cost)/p.avg_cost) 
                 for s, p in context.portfolio.positions.items()]
    positions.sort(key=lambda x: x[2])  # 亏损的先卖
    
    sold = 0
    for stock, val, pft in positions:
        if sold >= reduce_val:
            break
        order_target_value(stock, 0)
        sold += val
        
        if stock in g.small_cap_holdings:
            g.small_cap_holdings.remove(stock)
        if stock in g.etf_holdings:
            g.etf_holdings.remove(stock)


# ==================== 小市值策略（保持质量）====================

def rebalance_small_cap(context):
    """小市值调仓（每周，保持v2节奏）"""
    if g.emergency_mode:
        try:
            df = get_price('000300.XSHG', end_date=context.previous_date,
                          count=3, frequency='1d', fields=['close'])
            if len(df) >= 3:
                drops = [(df['close'].iloc[i]-df['close'].iloc[i-1])/df['close'].iloc[i-1] 
                        for i in range(1, len(df))]
                if all(d > -0.02 for d in drops):
                    g.emergency_mode = False
                    log.info('退出紧急模式')
                else:
                    return
        except:
            return
    
    # 选股
    stocks = select_small_cap_stocks(context)
    
    if not stocks:
        log.info('无合适股票')
        return
    
    # 卖出（保留昨日涨停）
    for stock in g.small_cap_holdings[:]:
        if stock not in stocks and stock not in g.yesterday_limit_up:
            order_target_value(stock, 0)
            g.small_cap_holdings.remove(stock)
    
    # 买入
    target_val = context.portfolio.total_value * g.small_cap_ratio
    current_val = sum([context.portfolio.positions[s].value 
                      for s in g.small_cap_holdings 
                      if s in context.portfolio.positions])
    avail = max(0, target_val - current_val)
    
    buy_list = [s for s in stocks if s not in g.small_cap_holdings and s not in g.sold_today]
    
    if buy_list and avail > 1000:
        can_buy = min(len(buy_list), g.stock_num - len(g.small_cap_holdings))
        if can_buy > 0:
            val_per = avail / can_buy
            bought = 0
            for stock in buy_list:
                if bought >= can_buy:
                    break
                if val_per > 1000:
                    order_target_value(stock, val_per)
                    g.small_cap_holdings.append(stock)
                    bought += 1
                    log.info(f'买入: {stock} {val_per:.0f}元')


def select_small_cap_stocks(context):
    """
    小市值选股（v6质量优先）
    
    关键改进vs v5：
    1. 市值下限8亿（不是5亿！）- 避开垃圾股区
    2. 保留ROE>5%要求 - 质量筛选
    3. 保留营收要求 - 避免空壳公司
    4. limit=120（不是200）- 精选优质
    """
    try:
        initial = get_index_stocks('399101.XSHE', date=context.previous_date)
        initial = filter_basic_stocks(context, initial)
        
        # v6质量筛选（保持v2标准！）
        q = query(
            valuation.code,
            valuation.market_cap,
            indicator.roe,
            income.operating_revenue
        ).filter(
            valuation.code.in_(initial),
            valuation.market_cap.between(g.min_market_cap, g.max_market_cap),
            income.net_profit > 0,           # 盈利
            indicator.roe > 0.05,            # ROE>5%（质量要求！）
            income.operating_revenue > 5e7   # 营收>5000万
        ).order_by(
            valuation.market_cap.asc()
        ).limit(120)  # 精选120支候选
        
        df = get_fundamentals(q, date=context.previous_date)
        
        if df is None or df.empty:
            return []
        
        stocks = df['code'].tolist()
        
        # 价格过滤
        if stocks:
            prices = history(1, '1d', 'close', stocks, df=False)
            stocks = [s for s in stocks if s in context.portfolio.positions or prices[s][-1] <= g.max_stock_price]
        
        return stocks[:g.stock_num * 3]  # 返回18支候选（6只×3）
        
    except Exception as e:
        log.error(f'选股异常: {str(e)}')
        return []


# ==================== ETF轮动 ====================

def rebalance_etf(context):
    """ETF轮动"""
    if g.emergency_mode:
        return
    
    best = select_best_etf(context)
    if not best:
        return
    
    current = None
    for etf in g.etf_holdings:
        if etf in context.portfolio.positions:
            current = etf
            break
    
    if current and current != best:
        order_target_value(current, 0)
        g.etf_holdings.remove(current)
        current = None
    
    if not current and best not in g.sold_today:
        val = context.portfolio.total_value * g.etf_ratio
        if val > 1000:
            order_target_value(best, val)
            if best not in g.etf_holdings:
                g.etf_holdings.append(best)
            log.info(f'买入ETF: {best} {val:.0f}元')


def select_best_etf(context):
    """选择动量最强ETF"""
    try:
        scores = []
        for etf in g.etf_pool:
            try:
                df = attribute_history(etf, 22, '1d', ['close'])
                if df is None or len(df) < 20:
                    continue
                
                prices = df['close'].values
                
                # 过滤暴跌
                if min(prices[-1]/prices[-2], prices[-2]/prices[-3]) < 0.95:
                    continue
                
                # 动量
                y = np.log(prices)
                x = np.arange(len(y))
                slope, inter = np.polyfit(x, y, 1)
                ret = math.exp(slope * 250) - 1
                
                # R2
                y_pred = slope * x + inter
                r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2)
                
                score = ret * r2
                if score > 0:
                    scores.append((etf, score))
            except:
                continue
        
        if not scores:
            return None
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
    except:
        return None


# ==================== 风控（v2严格标准）====================

def check_stop_loss(context):
    """止损止盈（保持v2的9%止损、150%止盈）"""
    for stock in list(context.portfolio.positions.keys()):
        if stock in g.sold_today:
            continue
        
        pos = context.portfolio.positions[stock]
        pft = (pos.price - pos.avg_cost) / pos.avg_cost
        
        # 更新最高收益
        if stock not in g.max_profit_dict:
            g.max_profit_dict[stock] = pft
        else:
            g.max_profit_dict[stock] = max(g.max_profit_dict[stock], pft)
        
        # 止损9%（v2标准）
        if pft < -g.stop_loss_ratio:
            order_target_value(stock, 0)
            update_holdings(stock)
            g.sold_today.add(stock)
            log.info(f'止损: {stock} {pft:.2%}')
        
        # 止盈150%（v2标准，现实可达）
        elif pft > g.stop_profit_ratio:
            order_target_value(stock, 0)
            update_holdings(stock)
            g.sold_today.add(stock)
            log.info(f'止盈: {stock} {pft:.2%}')
        
        # 移动止损（收益>30%后，回撤25%）
        elif g.max_profit_dict[stock] > 0.30:
            max_pft = g.max_profit_dict[stock]
            if pft < max_pft * (1 - g.trailing_stop_ratio):
                order_target_value(stock, 0)
                update_holdings(stock)
                g.sold_today.add(stock)
                log.info(f'移动止损: {stock} {max_pft:.2%}→{pft:.2%}')


def check_limit_up_stocks(context):
    """检查涨停打开"""
    if not g.yesterday_limit_up:
        return
    
    for stock in g.yesterday_limit_up:
        if stock not in context.portfolio.positions:
            continue
        
        try:
            df = get_price(stock, end_date=context.current_dt, frequency='1m',
                          fields=['close', 'high_limit'], count=1,
                          panel=False, fill_paused=True)
            
            if not df.empty and df['close'].iloc[0] < df['high_limit'].iloc[0]:
                order_target_value(stock, 0)
                update_holdings(stock)
                log.info(f'涨停打开: {stock}')
        except:
            pass


def update_holdings(stock):
    """更新持仓列表"""
    if stock in g.small_cap_holdings:
        g.small_cap_holdings.remove(stock)
    if stock in g.etf_holdings:
        g.etf_holdings.remove(stock)
    if stock in g.max_profit_dict:
        del g.max_profit_dict[stock]


# ==================== 辅助函数 ====================

def filter_basic_stocks(context, stock_list):
    """基础过滤（保留创业板300，避免科创北交）"""
    current_data = get_current_data()
    result = []
    
    for stock in stock_list:
        if (current_data[stock].is_st or 
            'ST' in current_data[stock].name or
            '退' in current_data[stock].name or
            current_data[stock].paused):
            continue
        
        # 仅排除科创板和北交所（保留创业板！）
        if stock.startswith('688') or stock.startswith('8') or stock.startswith('4'):
            continue
        
        # 次新股
        if context.previous_date - get_security_info(stock).start_date < timedelta(days=375):
            continue
        
        result.append(stock)
    
    return result


def after_trading(context):
    """收盘总结"""
    log.info('=' * 40)
    log.info(f'[{context.current_dt.date()}] 资产:{context.portfolio.total_value:.0f}')
    log.info(f'小市值:{len(g.small_cap_holdings)}只 ETF:{len(g.etf_holdings)}只')
    
    # 退出紧急模式检查
    if g.emergency_mode:
        try:
            df = get_price('000300.XSHG', end_date=context.previous_date,
                          count=5, frequency='1d', fields=['close'])
            if len(df) >= 5:
                drops = [(df['close'].iloc[i]-df['close'].iloc[i-1])/df['close'].iloc[i-1] 
                        for i in range(1, len(df))]
                if all(d > -0.02 for d in drops):
                    g.emergency_mode = False
                    log.info('退出紧急模式')
        except:
            pass
    
    log.info('=' * 40)