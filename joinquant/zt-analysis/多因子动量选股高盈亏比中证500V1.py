# 克隆自聚宽文章：https://www.joinquant.com/post/63587
# 标题：【实盘数据分享-停更】鹰击长空打板策略
# 作者：实测狂魔~王子玉

# 克隆自聚宽文章：https://www.joinquant.com/post/61144
# 标题：中大盘股,动能选股,ATR吊灯止损,年化300%以上
# 作者：Q641924483

# 克隆自聚宽文章：https://www.joinquant.com/post/61126
# 标题：多因子动量选股高盈亏比中证500V1
# 作者：Wjz820711

# 标题：多因子动量选股，高盈亏比中证500V1
from jqdata import *
import numpy as np
import pandas as pd
import talib as ta
import requests
import json
import time
from datetime import datetime, timedelta
import os

def initialize(context):
    # 初始化全局变量
    g.trade_stats = {
        'total_trades': 0, 
        'win_trades': 0, 
        'loss_trades': 0, 
        'total_profit': 0, 
        'max_profit': 0, 
        'max_loss': 0
    }
    
    # 数据重试机制参数
    g.data_retry_count = 3  # 数据获取重试次数
    g.data_retry_interval = 3  # 重试间隔(秒)
    
    # 创建当日卖出集合
    g.sold_today = set()
    log.info("策略全局变量初始化完成")
    
    # 设置基准
    set_benchmark('399905.XSHE')
    
    # 设置佣金和滑点
    set_order_cost(OrderCost(
        open_tax=0, 
        close_tax=0.001,
        open_commission=0.0003, 
        close_commission=0.0003,
        close_today_commission=0, 
        min_commission=5
    ), type='stock')
    
    set_slippage(FixedSlippage(0.01))
    
    # 策略核心参数
    g.atr_period = 14  # ATR波动率周期
    g.trend_filter_period = 50  # 趋势过滤均线周期
    g.entry_threshold = 1.2  # 突破阈值
    g.max_hold_days = 20  # 最大持有天数
    g.stop_loss_ratio = 0.8  # 止损ATR倍数
    g.profit_ratio = 3.0  # 止盈ATR倍数
    g.position_ratio = 0.3  # 单票仓位比例
    
    # 初始化交易时间表
    g.trade_days = get_trade_days(
        start_date=context.run_params.start_date,
        end_date=context.run_params.end_date
    )
    
    # 每日运行 - 调整交易时间为开盘后10分钟
    run_daily(before_market_open, time='before_open', reference_security='399905.XSHE')
    run_daily(trade_routine, time='9:25', reference_security='399905.XSHE')  # 延迟10分钟避开开盘高峰
    run_daily(after_trading_end, time='after_close', reference_security='399905.XSHE')

def safe_get_price(stock, count, end_date, frequency, fields):
    """带重试机制的安全数据获取函数"""
    for attempt in range(g.data_retry_count):
        try:
            data = get_price(
                stock, 
                count=count, 
                end_date=end_date,
                frequency=frequency, 
                fields=fields
            )
            if not data.empty:
                return data
            log.warning(f"尝试 {attempt+1}/{g.data_retry_count}: {stock} 数据为空")
        except Exception as e:
            log.warning(f"尝试 {attempt+1}/{g.data_retry_count}: {stock} 数据获取异常: {str(e)}")
        
        # 等待重试
        time.sleep(g.data_retry_interval)
    
    # 终极回退方案：使用前一日数据
    log.error(f"无法获取 {stock} 数据，使用前一日数据替代")
    return get_price(
        stock, 
        count=count, 
        end_date=end_date - timedelta(days=1),
        frequency=frequency, 
        fields=fields
    )

def safe_get_current_price(stock, context):
    """带重试机制的实时价格获取"""
    for attempt in range(g.data_retry_count):
        try:
            current_data = get_current_data()[stock]
            current_price = current_data.last_price
            
            # 验证价格有效性
            if current_price > 0 and current_data.low_limit < current_price < current_data.high_limit:
                return current_price
            log.warning(f"尝试 {attempt+1}/{g.data_retry_count}: {stock} 价格异常({current_price})")
        except Exception as e:
            log.warning(f"尝试 {attempt+1}/{g.data_retry_count}: {stock} 实时数据异常: {str(e)}")
        
        # 等待重试
        time.sleep(g.data_retry_interval)
    
    # 终极回退方案：使用持仓平均成本
    if stock in context.portfolio.positions:
        log.error(f"使用持仓成本作为 {stock} 的当前价格")
        return context.portfolio.positions[stock].avg_cost
    else:
        log.error(f"无法获取 {stock} 的有效价格，使用0")
        return 0

def before_market_open(context):
    """开盘前运行"""
    log.info("=" * 50)
    log.info(f"交易日: {context.current_dt}")
    log.info(f"账户总资产: {context.portfolio.total_value:.2f}元")
    log.info(f"可用资金: {context.portfolio.available_cash:.2f}元")
    log.info(f"持仓市值: {context.portfolio.positions_value:.2f}元")
    
    # 数据连通性检查
    test_stock = '000001.XSHE'
    test_data = safe_get_price(
        test_stock, 
        count=1, 
        end_date=context.current_dt,
        frequency='minute',
        fields=['close']
    )
    log.info(f"数据健康检查: 测试股票{test_stock}获取{'成功' if not test_data.empty else '失败'}")
    
    # 持仓概览
    if len(context.portfolio.positions) > 0:
        log.info("当前持仓:")
        for stock, pos in context.portfolio.positions.items():
            stock_name = get_security_info(stock).display_name
            profit = (pos.price - pos.avg_cost) * pos.total_amount
            profit_ratio = (pos.price / pos.avg_cost - 1) * 100
            log.info(f"{stock_name}({stock}): {pos.total_amount}股，成本: {pos.avg_cost:.2f}，现价: {pos.price:.2f}，盈亏: {profit:.2f}元({profit_ratio:.2f}%)")
    else:
        log.info("当前无持仓")
    
    log.info("=" * 50)

def trade_routine(context):
    """交易主逻辑"""
    # 获取当前日期索引
    current_date = context.current_dt.date()
    idx = list(g.trade_days).index(current_date)
    
    # 清仓条件: 大盘趋势破位
    if idx > 50 and not market_trend_filter(context):
        for stock in list(context.portfolio.positions.keys()):
            close_position(context, stock, "大盘趋势破位")
    
    # 持仓管理
    manage_existing_positions(context, current_date)
    
    # 开仓逻辑
    if len(context.portfolio.positions) < 5:  # 最大持仓数
        open_new_positions(context, idx)

def market_trend_filter(context):
    """大盘趋势过滤器"""
    log.info("===== 检查大盘趋势 =====")
    # 获取沪深300指数数据
    bench = '399905.XSHE'
    prices = safe_get_price(
        bench, 
        count=100, 
        end_date=context.current_dt,
        frequency='daily', 
        fields=['close']
    )
    
    # 计算趋势指标
    ma50 = ta.MA(prices['close'], timeperiod=50)[-1]
    current_close = prices['close'][-1]
    
    # 价格在50日均线上方为上升趋势
    result = current_close > ma50 * 1.02
    log.info(f"大盘趋势结果: 指数{bench}当前价{current_close:.2f} 50日均线{ma50:.2f} 满足条件? {result}")
    return result

def manage_existing_positions(context, current_date):
    """持仓管理: 止损/止盈/时间止损"""
    for stock, pos in context.portfolio.positions.items():
        # 获取持仓数据
        avg_cost = pos.avg_cost
        
        # 使用安全价格获取函数
        current_price = safe_get_current_price(stock, context)
        
        # 计算波动率止损
        atr = calc_atr(stock, context, g.atr_period)
        stop_loss_price = avg_cost - g.stop_loss_ratio * atr
        take_profit_price = avg_cost + g.profit_ratio * atr
        
        # 止损逻辑
        if current_price <= stop_loss_price:
            close_position(context, stock, f"止损触发(ATR={atr:.2f})")
            continue
        
        # 止盈逻辑
        if current_price >= take_profit_price:
            close_position(context, stock, f"止盈触发(ATR={atr:.2f})")
            continue
        
        # 时间止损
        hold_days = (current_date - pos.init_time.date()).days
        if hold_days >= g.max_hold_days:
            close_position(context, stock, "持有时间到期")
            continue

def open_new_positions(context, day_idx):
    """开仓逻辑"""
    log.info("===== 开始选股 =====")
    log.info(f"当前可用资金: {context.portfolio.available_cash:.2f}元")
    
    # 获取候选股票池
    candidate_stocks = get_stock_pool(context)
    log.info(f"初始候选股票数量: {len(candidate_stocks)}")
    log.info(f"今日禁买列表: {list(g.sold_today)}")
    
    # 过滤今日已卖出的股票
    candidate_stocks = [s for s in candidate_stocks if s not in g.sold_today]
    log.info(f"过滤后候选股数量: {len(candidate_stocks)}")
    
    if not candidate_stocks:
        log.info("无合格候选股票，跳过开仓")
        return
    
    # 计算候选股指标
    results = []
    for stock in candidate_stocks:
        # 获取价格数据 - 使用安全获取函数
        prices = safe_get_price(
            stock, 
            count=100, 
            end_date=context.current_dt,
            frequency='daily', 
            fields=['high', 'low', 'close', 'volume']
        )
        
        # 计算技术指标
        atr = calc_atr_from_prices(prices, g.atr_period)
        ma20 = ta.MA(prices['close'], timeperiod=20)[-1]
        adx = ta.ADX(prices['high'], prices['low'], prices['close'], timeperiod=14)[-1]
        volume_ma = ta.MA(prices['volume'], timeperiod=20)[-1]
        
        # 过滤条件
        if prices['close'][-1] < ma20 * 1.05:
            continue  # 价格在20日均线附近
        if adx < 25:
            continue  # 趋势强度不足
        if prices['volume'][-1] < volume_ma * 1.2:
            continue  # 量能不足
        
        # 计算突破强度
        breakout_strength = (prices['close'][-1] - max(prices['high'][-20:-1])) / atr
        if breakout_strength > g.entry_threshold:
            results.append((stock, breakout_strength, atr, prices['close'][-1]))
            log.info(f"√ {stock} 突破强度: {breakout_strength:.2f} (需 > {g.entry_threshold})")
    
    # 按突破强度排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    if not results:
        log.info("无突破强度达标的股票")
        return
    
    # 计算每只股票的目标仓位(基于总资产)
    total_value = context.portfolio.total_value
    target_per_stock = total_value * g.position_ratio
    
    # 开仓
    stocks_to_buy = min(5, len(results))  # 最多买5只
    for i, (stock, strength, atr, current_price) in enumerate(results[:stocks_to_buy]):
        # 计算目标数量
        target_amount = int(target_per_stock / current_price)
        
        # 确保数量是100的整数倍(A股交易规则)
        target_amount = (target_amount // 100) * 100
        
        # 检查可用现金是否足够
        if target_amount <= 0:
            continue
        
        # 计算所需资金
        required_cash = target_amount * current_price * 1.003  # 包含交易费用
        
        # 如果现金不足，调整购买数量
        if required_cash > context.portfolio.available_cash:
            # 计算最大可买数量
            max_possible = int(context.portfolio.available_cash / (current_price * 1.003))
            max_possible = (max_possible // 100) * 100  # 调整为100的整数倍
            if max_possible < 100:
                continue
            target_amount = max_possible
        
        # 发送开仓通知
        stock_name = get_security_info(stock).display_name
        actual_cash = target_amount * current_price
        
        # 添加买入日志
        cash_per_stock = target_amount * current_price
        log.info(f"[买入] {stock_name}({stock})，数量: {target_amount}股，价格: {current_price:.2f}元，金额: {cash_per_stock:.2f}元")
        
        # 执行交易
        order(stock, target_amount)
        log.info(f"开仓 {stock}，数量: {target_amount}，仓位: {actual_cash / total_value * 100:.1f}%")
        
        # 更新交易统计
        g.trade_stats['total_trades'] += 1

def get_stock_pool(context):
    """获取候选股票池"""
    # 过滤ST/*ST/停牌/涨跌停
    prev_date = context.current_dt - timedelta(days=1)
    stocks = get_index_stocks('399905.XSHE', date=prev_date)
    current_data = get_current_data()
    
    return [
        s for s in stocks 
        if not (
            s.startswith('688') or  # 排除科创板
            current_data[s].is_st or
            current_data[s].paused or
            current_data[s].day_open >= current_data[s].high_limit or
            current_data[s].day_open <= current_data[s].low_limit
        )
    ]

def calc_atr(stock, context, period):
    """计算ATR"""
    prices = safe_get_price(
        stock, 
        count=period + 20, 
        end_date=context.current_dt,
        frequency='daily', 
        fields=['high', 'low', 'close']
    )
    return calc_atr_from_prices(prices, period)

def calc_atr_from_prices(prices, period):
    """根据价格数据计算ATR"""
    high = np.array(prices['high'])
    low = np.array(prices['low'])
    close = np.array(prices['close'])
    
    # 计算真实波幅(TR)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    return np.mean(tr[-period:])

def close_position(context, stock, reason):
    """平仓操作"""
    # 获取持仓信息
    pos = context.portfolio.positions[stock]
    avg_cost = pos.avg_cost
    
    # 使用安全价格获取函数
    current_price = safe_get_current_price(stock, context)
    
    profit = (current_price - avg_cost) * pos.total_amount
    profit_ratio = (current_price / avg_cost - 1) * 100
    
    # 添加卖出日志
    cash_per_stock = pos.total_amount * current_price
    stock_name = get_security_info(stock).display_name
    log.info(f"[卖出] {stock_name}({stock})，数量: {pos.total_amount}股，价格: {current_price:.2f}元，金额: {cash_per_stock:.2f}元")
    log.info(f"平仓原因: {reason}，持仓盈亏: {profit:.2f}元({profit_ratio:.2f}%)")
    
    # 更新交易统计
    g.trade_stats['total_profit'] += profit
    
    if profit > 0:
        g.trade_stats['win_trades'] += 1
        if profit > g.trade_stats['max_profit']:
            g.trade_stats['max_profit'] = profit
    else:
        g.trade_stats['loss_trades'] += 1
        if profit < g.trade_stats['max_loss']:
            g.trade_stats['max_loss'] = profit
    
    g.sold_today.add(stock)  # 将卖出的股票加入今日禁止列表
    
    # 执行平仓
    order_target_value(stock, 0)

def after_trading_end(context):
    """收盘后运行"""
    log.info("=" * 50)
    log.info(f"收盘总结 {context.current_dt}")
    log.info(f"账户总资产: {context.portfolio.total_value:.2f}元")
    log.info(f"日盈亏: {context.portfolio.total_value - context.portfolio.starting_cash:.2f}元")
    
    # 添加防御性检查: 确保trade_stats存在
    if not hasattr(g, 'trade_stats'):
        log.warning("检测到trade_stats未初始化，正在创建空统计")
        g.trade_stats = {
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'total_profit': 0,
            'max_profit': 0,
            'max_loss': 0
        }
    
    # 交易统计
    if g.trade_stats['total_trades'] > 0:
        win_rate = g.trade_stats['win_trades'] / g.trade_stats['total_trades'] * 100
        log.info("交易统计: 总交易%d次，盈利%d次，亏损%d次，胜率: %.2f%%" % (
            g.trade_stats['total_trades'], 
            g.trade_stats['win_trades'],
            g.trade_stats['loss_trades'], 
            win_rate
        ))
        log.info("总盈亏: %.2f元, 最大盈利: %.2f元, 最大亏损: %.2f元" % (
            g.trade_stats['total_profit'], 
            g.trade_stats['max_profit'],
            g.trade_stats['max_loss']
        ))
    else:
        log.info("当日无交易记录")
    
    log.info("=" * 50)
    
    # 清空今日卖出列表
    if hasattr(g, 'sold_today'):
        g.sold_today.clear()
    else:
        log.warning("sold_today未初始化，创建空集合")
        g.sold_today = set()