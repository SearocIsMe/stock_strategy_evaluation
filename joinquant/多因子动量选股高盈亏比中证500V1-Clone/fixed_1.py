# 克隆自聚宽文章：https://www.joinquant.com/post/63688
# 标题：高股息低杠杆小市值轮动策略优化
# 作者：lglgl

# 克隆自聚宽文章：https://www.joinquant.com/post/61144
# 标题：中大盘股,动能选股,ATR吊灯止损,年化300%以上
# 作者：Q641924483

# 克隆自聚宽文章：https://www.joinquant.com/post/61126
# 标题：多因子动量选股高盈亏比中证500V1
# 作者：Wjz820711

# 标题：多因子动量选股，高盈亏比中证500V1 - 无未来函数修复版
from jqdata import *
import numpy as np
import pandas as pd
import talib as ta
import requests
import json
import time
from datetime import datetime, timedelta
import os

import myqmt_sql

# ⭐ 在这里设置这个策略的分类标签（写入 trade.fenlei）
myqmt_sql.FENLEI = '中证500V1_无未来函数版'      

from myqmt_sql import (
    order_zzy as order,
    order_target_zzy as order_target,
    order_value_zzy as order_value,
    order_target_value_zzy as order_target_value
)


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
    log.info("策略全局变量初始化完成 - 无未来函数版本")
    
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
    
    # 每日运行 - 调整交易时间为避开开盘高峰
    run_daily(before_market_open, time='before_open', reference_security='399905.XSHE')
    run_daily(trade_routine_fixed, time='9:40', reference_security='399905.XSHE')  # 延迟避开开盘高峰
    run_daily(after_trading_end_fixed, time='after_close', reference_security='399905.XSHE')

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
    
    # 终极回退方案：使用更早的数据
    log.error(f"无法获取 {stock} 数据，使用更早数据替代")
    return get_price(
        stock, 
        count=count, 
        end_date=end_date - timedelta(days=2),
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
        log.error(f"无法获取 {stock} 的有效价格，使用前收盘价")
    prev_data = safe_get_price(
        stock, 
        count=1, 
        end_date=context.current_dt - timedelta(days=1),
        frequency='daily',
        fields=['close']
    )
    return prev_data['close'].iloc[0] if not prev_data.empty else 0

def before_market_open(context):
    """开盘前运行"""
    log.info("=" * 50)
    log.info(f"交易日: {context.current_dt}")
    log.info(f"账户总资产: {context.portfolio.total_value:.2f}元")
    log.info(f"可用资金: {context.portfolio.available_cash:.2f}元")
    log.info(f"持仓市值: {context.portfolio.positions_value:.2f}元")
    
    # 数据连通性检查 - 使用前一交易日数据
    test_stock = '000001.XSHE'
    prev_date = context.current_dt - timedelta(days=1)
    test_data = safe_get_price(
        test_stock, 
        count=1, 
        end_date=prev_date,
        frequency='daily',
        fields=['close']
    )
    log.info(f"数据健康检查: 测试股票{test_stock}获取{'成功' if not test_data.empty else '失败'}")
    
    # 持仓概览 - 使用前一交易日数据
    if len(context.portfolio.positions) > 0:
        log.info("当前持仓:")
        for stock, pos in context.portfolio.positions.items():
            stock_name = get_security_info(stock).display_name
            # 使用前一交易日收盘价计算
            prev_close = test_data['close'].iloc[0] if not test_data.empty else 0
            
            # 获取前一交易日价格来显示盈亏
            prev_price_data = safe_get_price(
                stock, 
                count=1, 
                end_date=prev_date,
                frequency='daily',
                fields=['close']
            )
            
            if not prev_price_data.empty:
                profit = (prev_price_data['close'].iloc[0] - pos.avg_cost) * pos.total_amount
            profit_ratio = (prev_price_data['close'].iloc[0] / pos.avg_cost - 1) * 100
            log.info(f"{stock_name}({stock}): {pos.total_amount}股，成本: {pos.avg_cost:.2f}，前收: {prev_price_data['close'].iloc[0]:.2f}，盈亏: {profit:.2f}元({profit_ratio:.2f}%)")
        else:
            log.info(f"{stock_name}({stock}): {pos.total_amount}股，成本: {pos.avg_cost:.2f}")
    
    log.info("=" * 50)

def trade_routine_fixed(context):
    """修复版交易主逻辑 - 无未来函数"""
    current_date = context.current_dt.date()
    
    # 获取交易日索引（用于历史数据参考）
    if hasattr(g, 'trade_days'):
        idx = list(g.trade_days).index(current_date)
    else:
        idx = 0
    
    # 清仓条件检查（使用历史数据）
    if idx > 50:
        if not market_trend_filter_fixed(context):
            log.info("大盘趋势破位，执行清仓")
            for stock in list(context.portfolio.positions.keys()):
                close_position_fixed(context, stock, "大盘趋势破位")
    
    # 持仓管理（使用修复版）
    manage_existing_positions_fixed(context, current_date)
    
    # 开仓逻辑（在持仓管理后）
    if len(context.portfolio.positions) < 3:
        open_new_positions_fixed(context, idx)
    else:
        log.info("达到最大持仓限制，不开新仓")

def market_trend_filter_fixed(context):
    """修复版大盘趋势过滤器 - 使用历史数据"""
    log.info("===== 检查大盘趋势(修复版) =====")
    
    # 使用前一日数据 - 关键修复！
    prev_date = context.current_dt - timedelta(days=1)
    
    bench = '399905.XSHE'
    prices = safe_get_price(
        bench, 
        count=100, 
        end_date=prev_date,  # 不使用当日数据！
        frequency='daily', 
        fields=['close']
    )
    
    if prices.empty or len(prices) < 50:
        log.warning("基准指数数据不足，默认通过趋势过滤")
        return True
    
    # 计算趋势指标（基于历史数据）
    ma50 = ta.MA(prices['close'], timeperiod=50)
    
    if len(ma50) == 0:
        return True
        
    current_close = prices['close'].iloc[-1]  # 前一日收盘价
    ma50_value = ma50.iloc[-1] if hasattr(ma50, 'iloc') else ma50[-1]
    
    # 使用更保守的阈值
    result = current_close > ma50_value * 0.98  # 略微放宽条件
    
    log.info(f"大盘趋势结果(修复版): 指数{bench}")
    log.info(f"  前收盘价: {current_close:.2f}")
    log.info(f"  50日均线: {ma50_value:.2f}")
    log.info(f"  满足条件? {result}")
    
    return result

def manage_existing_positions_fixed(context, current_date):
    """修复版持仓管理: 使用严格的历史数据避免未来函数"""
    log.info("===== 开始持仓管理(修复版) =====")
    
    if len(context.portfolio.positions) == 0:
        log.info("当前无持仓，跳过持仓管理")
        return
    
    positions_to_close = []
    
    for stock, pos in context.portfolio.positions.items():
        stock_name = get_security_info(stock).display_name
        log.info(f"检查持仓: {stock_name}({stock})")
        
        # 基础持仓信息
        avg_cost = pos.avg_cost
        hold_days = (current_date - pos.init_time.date()).days
        
        try:
            # 获取历史价格数据 - 关键：使用前一日数据
            prev_date = context.current_dt - timedelta(days=1)
            
            # 获取足够的历史数据用于计算
            price_data = safe_get_price(
                stock, 
                count=g.atr_period + 30,  # 多取一些数据确保计算准确
                end_date=prev_date,  # 使用前一交易日数据！
                frequency='daily', 
                fields=['high', 'low', 'close']
            )
            
            if price_data.empty or len(price_data) < g.atr_period + 1:
                log.warning(f"{stock} 数据不足，跳过处理")
                continue
            
            # 获取前一交易日的收盘价作为决策基准
            prev_close = price_data['close'].iloc[-1] if hasattr(price_data['close'], 'iloc') else price_data['close'].values[-1]
            
            # 计算ATR - 使用历史数据
            atr = calc_atr_from_prices_fixed(price_data, g.atr_period)
            
            if atr <= 0:
                log.warning(f"{stock} ATR计算异常，跳过处理")
            continue
            
            # 止损止盈计算（使用前一交易日收盘价）
            stop_loss_price = avg_cost - g.stop_loss_ratio * atr
            take_profit_price = avg_cost + g.profit_ratio * atr
            
            # 检查止损条件
            if prev_close <= stop_loss_price:
                positions_to_close.append((
                    stock, 
                    f"止损触发(前收:{prev_close:.2f} <= 止损价:{stop_loss_price:.2f}, ATR:{atr:.2f})"
                ))
                log.info(f"  → 止损条件满足: {prev_close:.2f} <= {stop_loss_price:.2f}")
                continue
            
            # 检查止盈条件  
            if prev_close >= take_profit_price:
                positions_to_close.append((
                    stock,
                    f"止盈触发(前收:{prev_close:.2f} >= 止盈价:{take_profit_price:.2f}, ATR:{atr:.2f})"
                ))
                log.info(f"  → 止盈条件满足: {prev_close:.2f} >= {take_profit_price:.2f}")
                continue
            
            # 时间止损检查
            if hold_days >= g.max_hold_days:
                positions_to_close.append((
                    stock,
                    f"持有时间到期({hold_days}天 >= {g.max_hold_days}天)"
                ))
                log.info(f"  → 时间止损: 持有{hold_days}天")
                continue
            
            # 如果所有条件都不满足，记录持仓状态
            current_profit_ratio = (prev_close / avg_cost - 1) * 100
            log.info(f"  → 继续持有，盈亏: {current_profit_ratio:.2f}%")
            
        except Exception as e:
            log.error(f"处理持仓{stock}时发生异常: {str(e)}")
            continue
    
    # 执行平仓操作（放到循环外，避免在迭代中修改字典）
    for stock, reason in positions_to_close:
        close_position_fixed(context, stock, reason)

def calc_atr_from_prices_fixed(prices, period):
    """修复版ATR计算，严格避免未来数据"""
    if len(prices) < period + 1:
        return 0
        
    high = np.array(prices['high'])
    low = np.array(prices['low'])
    close = np.array(prices['close'])
    
    # 确保不包含任何未来信息
    tr_values = []
    for i in range(1, len(prices)):
        tr1 = high[i] - low[i]  # 当日最高-最低
        tr2 = abs(high[i] - close[i-1])  # 当日最高-前收
        tr3 = abs(low[i] - close[i-1])   # 当日最低-前收
        tr = max(tr1, tr2, tr3)
        tr_values.append(tr)
    
    # 返回最后一个完整的period周期ATR
    if len(tr_values) >= period:
        return np.mean(tr_values[-period:])
    else:
        return np.mean(tr_values) if tr_values else 0

def get_stock_pool_fixed(context):
    """修复版候选股票池 - 无未来函数"""
    # 使用前一个交易日的数据
    prev_date = context.current_dt - timedelta(days=1)
    
    # 获取前一日的中证500成分股
    stocks = get_index_stocks('399905.XSHE', date=prev_date)
    current_data = get_current_data()
    
    filtered_stocks = []
    for stock in stocks:
        try:
            # 只使用历史数据进行过滤
            if current_data[stock].is_st or current_data[stock].paused:
                continue
                
            # 排除科创板和创业板（可选）
            if stock.startswith('688'):
                continue
                
            # 获取前一日价格数据进行基础过滤
            prev_day_data = safe_get_price(
                stock, 
                count=2, 
                end_date=prev_date,
                frequency='daily', 
                fields=['close']
            )
            
            if prev_day_data.empty or len(prev_day_data) < 2:
                continue
                
            # 进一步过滤：价格不能太低，交易量不能太小
            if len(prev_day_data) >= 2:
                prev_close = prev_day_data['close'].iloc[-1]
                # 排除价格异常和流动性差的股票
            if prev_close < 3 or prev_close > 200:
                continue
                
            filtered_stocks.append(stock)
            
        except Exception as e:
            log.warning(f"过滤{stock}时出错: {str(e)}")
            continue
    
    log.info(f"股票池过滤后数量: {len(filtered_stocks)}")
    return filtered_stocks

def open_new_positions_fixed(context, day_idx):
    """修复版开仓逻辑 - 无未来函数"""
    log.info("===== 开始选股(修复版) =====")
    log.info(f"当前可用资金: {context.portfolio.available_cash:.2f}元")
    
    # 获取候选股票池（修复版）
    candidate_stocks = get_stock_pool_fixed(context)
    log.info(f"候选股票数量: {len(candidate_stocks)}")
    
    if not candidate_stocks:
        log.info("无合格候选股票，跳过开仓")
        return
    
    # 进一步过滤今日已卖出的股票
    if hasattr(g, 'sold_today'):
        candidate_stocks = [s for s in candidate_stocks if s not in g.sold_today)
    
    if not candidate_stocks:
        log.info("今日已卖出所有候选股票，无法开仓")
        return
    
    # 计算指标（使用历史数据）
    results = []
    prev_date = context.current_dt - timedelta(days=1)
    
    for stock in candidate_stocks:
        try:
            # 获取价格数据 - 使用前一日数据
            prices = safe_get_price(
                stock, 
                count=100, 
                end_date=prev_date,  # 关键修复！
                frequency='daily', 
                fields=['high', 'low', 'close', 'volume']
            )
            
            if prices.empty or len(prices) < 50:
                continue
                
            # 基础过滤：使用前一日收盘价判断
            prev_close = prices['close'].iloc[-1] if hasattr(prices['close'], 'iloc') else prices['close'].values[-1]
            
            # 计算技术指标（基于历史数据）
            ma20 = ta.MA(prices['close'], timeperiod=20)
            if len(ma20) == 0:
                continue
                
            ma20_value = ma20.iloc[-1] if hasattr(ma20, 'iloc') else ma20[-1]
            adx = ta.ADX(prices['high'], prices['low'], prices['close'], timeperiod=14)
            if len(adx) == 0:
                continue
                
            adx_value = adx.iloc[-1] if hasattr(adx, 'iloc') else adx[-1]
            volume_ma = ta.MA(prices['volume'], timeperiod=20)
            if len(volume_ma) == 0:
                continue
                
            volume_ma_value = volume_ma.iloc[-1] if hasattr(volume_ma, 'iloc') else volume_ma[-1]
            
            # 过滤条件（使用历史数据）
            if prev_close < ma20_value * 1.02:  # 略微放宽条件
                continue
                
            if adx_value < 20:  # 降低阈值
                continue
                
            if prices['volume'].iloc[-1] < volume_ma_value * 1.1:  # 降低量能要求
                continue
            
            # 计算突破强度（基于完全历史数据）
            if len(prices) < 21:
                continue
                
            historical_highs = prices['high'][-21:-1]  # 排除最后一天，使用历史数据
            historical_highs_data = historical_highs if hasattr(historical_highs, 'values') else historical_highs.values
            if len(historical_highs_data) == 0:
                continue
                
            previous_max = max(historical_highs_data)
            
            atr = calc_atr_from_prices_fixed(prices, g.atr_period)
            
            if atr <= 0:
                continue
                
            breakout_strength = (prev_close - previous_max) / atr
            
            if breakout_strength > g.entry_threshold:
                results.append((stock, breakout_strength, atr, prev_close))
                log.info(f"√ {stock} 符合条件，突破强度: {breakout_strength:.2f}")
        
        except Exception as e:
            log.warning(f"分析{stock}时出错: {str(e)}")
            continue
    
    # 按突破强度排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    if not results:
        log.info("无突破强度达标的股票")
        return
    
    # 计算每只股票的目标仓位(基于总资产)
    total_value = context.portfolio.total_value
    target_per_stock = total_value * g.position_ratio
    
    # 开仓
    stocks_to_buy = min(3 - len(context.portfolio.positions), len(results))
    
    if stocks_to_buy < 1:
        log.info(f"持仓数已满{len(context.portfolio.positions)}，无法再开仓")
        return
    
    for i, (stock, strength, atr, current_price) in enumerate(results[:stocks_to_buy]):
        # 计算目标数量
        target_amount = int(target_per_stock / current_price)
        
        # 确保数量是100的整数倍(A股交易规则）
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
        log.info(f"[买入修复版] {stock_name}({stock})，数量: {target_amount}股，价格: {current_price:.2f}元，金额: {cash_per_stock:.2f}元")
        
        # 执行交易
        order(stock, target_amount)
        log.info(f"开仓 {stock}，数量: {target_amount}，仓位: {actual_cash / total_value * 100:.1f}%")
        
        # 更新交易统计
        g.trade_stats['total_trades'] += 1

def close_position_fixed(context, stock, reason):
    """修复版平仓操作，避免未来函数"""
    if stock not in context.portfolio.positions:
        log.warning(f"尝试平仓不存在的持仓: {stock}")
        return
    
    pos = context.portfolio.positions[stock]
    avg_cost = pos.avg_cost
    
    try:
        # 使用当日开盘数据执行平仓（这是实时可用的）
        current_data = get_current_data()[stock]
        execute_price = current_data.day_open
        
        # 验证执行价格的合理性
        if execute_price <= 0 or execute_price > current_data.high_limit:
            log.warning(f"{stock} 开盘价{execute_price:.2f}异常，使用前收盘价")
            prev_data = safe_get_price(
                stock, 
                count=1, 
                end_date=context.current_dt - timedelta(days=1),
                frequency='daily',
                fields=['close']
            )
            execute_price = prev_data['close'].iloc[0] if not prev_data.empty else avg_cost
        
        # 计算盈亏（基于执行价格）
        profit = (execute_price - avg_cost) * pos.total_amount
        profit_ratio = (execute_price / avg_cost - 1) * 100
        
        stock_name = get_security_info(stock).display_name
        log.info(f"[卖出修复版] {stock_name}({stock})")
        log.info(f"  执行价格: {execute_price:.2f}, 成本: {avg_cost:.2f}")
        log.info(f"  平仓原因: {reason}")
        log.info(f"  持仓盈亏: {profit:.2f}元({profit_ratio:.2f}%)")
        
        # 更新交易统计
        if hasattr(g, 'trade_stats'):
            g.trade_stats['total_profit'] += profit
            
            if profit > 0:
                g.trade_stats['win_trades'] += 1
                if profit > g.trade_stats['max_profit']:
                    g.trade_stats['max_profit'] = profit
            else:
                g.trade_stats['loss_trades'] += 1
                if profit < g.trade_stats['max_loss']:
                    g.trade_stats['max_loss'] = profit
        
        # 添加到禁止买入列表
        if hasattr(g, 'sold_today'):
            g.sold_today.add(stock)
        else:
            g.sold_today = set([stock])
        
        # 执行平仓
        order_target_value(stock, 0)
        
    except Exception as e:
        log.error(f"平仓操作失败 {stock}: {str(e)}")

def after_trading_end_fixed(context):
    """修复版收盘后运行"""
    log.info("=" * 50)
    log.info(f"收盘总结(修复版) {context.current_dt}")
    log.info(f"账户总资产: {context.portfolio.total_value:.2f}元")
    
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
        g.sold_today = set()
        
# 兼容原有函数调用，但使用修复版逻辑
def calc_atr(stock, context, period):
    """兼容原有函数调用，内部使用修复版逻辑"""
    prev_date = context.current_dt - timedelta(days=1)
    prices = safe_get_price(
        stock, 
        count=period + 20, 
        end_date=prev_date,  # 使用历史数据
        frequency='daily', 
        fields=['high', 'low', 'close']
    )
    return calc_atr_from_prices_fixed(prices, period)

