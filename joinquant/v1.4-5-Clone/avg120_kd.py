# 导入聚宽量化平台相关库
import jqdata
import pandas as pd
import numpy as np

def initialize(context):
    """
    初始化函数，设定基准、股票池、参数等
    """
    # 设定沪深300作为基准
    g.benchmark = '000300.XSHG'
    set_benchmark(g.benchmark)
    
    # 全局变量
    g.stock_pool = []  # 股票池
    g.selected_stocks = []  # 选出的股票
    g.hold_days = {}  # 持仓天数记录
    
    # 均线参数
    g.ma_short = 120  # 短期均线
    g.ma_long = 240   # 长期均线
    
    # KD指标参数
    g.kd_period = 9   # KD周期
    
    # 风险控制参数
    g.max_position_ratio = 0.2  # 单只股票最大仓位比例
    g.max_positions = 5         # 最大持仓股票数量
    g.stop_loss_rate = 0.05     # 止损比例5%
    g.take_profit_rate = 0.15   # 止盈比例15%
    
    # 设置每1分钟运行一次
    run_daily(trade, time='every_bar')
    
    # 记录日志
    log.info("策略初始化完成")

def filter_stocks(context):
    """
    第一步：筛选满足均线条件的股票
    条件：120日均线在240日均线之上，且两条均线都向上
    """
    all_stocks = get_index_stocks('000300.XSHG')  # 获取沪深300成分股
    
    filtered_stocks = []
    
    for stock in all_stocks:
        try:
            # 获取股票基本信息，剔除ST股和次新股
            stock_info = get_security_info(stock)
            if stock_info.display_name.find('ST') != -1 or stock_info.display_name.find('*ST') != -1:
                continue
            
            # 计算上市天数，剔除上市不足60天的次新股（约3个月）
            list_days = (context.current_dt.date() - stock_info.start_date).days
            if list_days < 60:
                continue
            
            # 获取历史数据计算均线
            hist_data = get_price(stock, count=g.ma_long+20, end_date=context.current_dt, 
                                frequency='1d', fields=['close'])
            
            if len(hist_data) < g.ma_long:
                continue
            
            close_prices = hist_data['close']
            
            # 计算120日和240日均线
            ma_120 = close_prices.rolling(window=g.ma_short).mean()
            ma_240 = close_prices.rolling(window=g.ma_long).mean()
            
            # 检查条件：120日线在240日线之上，且两条均线都向上
            if (ma_120.iloc[-1] > ma_240.iloc[-1] and 
                ma_120.iloc[-1] > ma_120.iloc[-2] and 
                ma_240.iloc[-1] > ma_240.iloc[-2]):
                filtered_stocks.append(stock)
                
        except Exception as e:
            log.warn("处理股票{}时出错: {}".format(stock, str(e)))
            continue
    
    return filtered_stocks

def filter_by_weekly_kd(context, stocks):
    """
    第二步：用周线KD金叉进一步筛选股票
    """
    kd_filtered_stocks = []
    
    for stock in stocks:
        try:
            # 获取周线数据计算KD指标
            weekly_data = get_price(stock, count=g.kd_period*3, end_date=context.current_dt, 
                                  frequency='5d', fields=['high', 'low', 'close'])
            
            if len(weekly_data) < g.kd_period:
                continue
            
            high_prices = weekly_data['high']
            low_prices = weekly_data['low']
            close_prices = weekly_data['close']
            
            # 计算KD指标
            lowest_low = low_prices.rolling(window=g.kd_period).min()
            highest_high = high_prices.rolling(window=g.kd_period).max()
            
            rsv = (close_prices - lowest_low) / (highest_high - lowest_low) * 100
            k_values = rsv.ewm(span=3).mean()
            d_values = k_values.ewm(span=3).mean()
            
            # 检查KD金叉：K线上穿D线
            if (k_values.iloc[-1] > d_values.iloc[-1] and 
                k_values.iloc[-2] <= d_values.iloc[-2]):
                kd_filtered_stocks.append(stock)
                
        except Exception as e:
            log.warn("计算股票{}周线KD时出错: {}".format(stock, str(e)))
            continue
    
    return kd_filtered_stocks

def check_daily_kd_golden_cross(context, stock):
    """
    第三步：检查日线KD金叉买入信号
    """
    try:
        # 获取日线数据计算KD指标
        daily_data = get_price(stock, count=g.kd_period*3, end_date=context.current_dt, 
                             frequency='1d', fields=['high', 'low', 'close'])
        
        if len(daily_data) < g.kd_period:
            return False
        
        high_prices = daily_data['high']
        low_prices = daily_data['low']
        close_prices = daily_data['close']
        
        # 计算KD指标
        lowest_low = low_prices.rolling(window=g.kd_period).min()
        highest_high = high_prices.rolling(window=g.kd_period).max()
        
        rsv = (close_prices - lowest_low) / (highest_high - lowest_low) * 100
        k_values = rsv.ewm(span=3).mean()
        d_values = k_values.ewm(span=3).mean()
        
        # 检查KD金叉：K线上穿D线
        return (k_values.iloc[-1] > d_values.iloc[-1] and 
                k_values.iloc[-2] <= d_values.iloc[-2])
        
    except Exception as e:
        log.warn("检查股票{}日线KD时出错: {}".format(stock, str(e)))
        return False

def check_buy_timing_rules(context, stock):
    """
    检查买卖时机规则
    """
    try:
        current_data = get_current_data()
        
        # 规则4：低开超过3个点，15分钟内不翻红，不下单
        today_data = get_price(stock, count=1, end_date=context.current_dt, 
                             frequency='1m', fields=['open', 'close'])
        
        if len(today_data) > 0:
            open_price = today_data['open'].iloc[-1]
            current_price = current_data[stock].last_price
            
            if open_price > 0 and (open_price - current_data[stock].prev_close) / current_data[stock].prev_close < -0.03:
                # 低开超过3%，检查是否翻红
                if current_price <= open_price:
                    return False
        
        return True
        
    except Exception as e:
        log.warn("检查买卖时机时出错: {}".format(str(e)))
        return True

def risk_management(context, stock):
    """
    风险管理：止损止盈检查
    """
    if stock not in context.portfolio.positions:
        return 'hold'
    
    position = context.portfolio.positions[stock]
    current_price = get_current_data()[stock].last_price
    cost_price = position.avg_cost
    
    # 计算盈亏比例
    profit_rate = (current_price - cost_price) / cost_price
    
    # 止损检查
    if profit_rate <= -g.stop_loss_rate:
        return 'sell'
    
    # 止盈检查
    if profit_rate >= g.take_profit_rate:
        return 'sell'
    
    # 规则6：连涨三天，减仓一半
    hist_data = get_price(stock, count=4, end_date=context.current_dt, 
                         frequency='1d', fields=['close'])
    
    if len(hist_data) >= 4:
        closes = hist_data['close']
        if all(closes.iloc[i] > closes.iloc[i-1] for i in range(1, 4)):
            return 'reduce'
    
    return 'hold'

def get_current_position_count(context):
    """
    获取当前持仓股票数量
    """
    return len([stock for stock in context.portfolio.positions.keys() 
                if context.portfolio.positions[stock].total_amount > 0])

def get_available_position_slots(context):
    """
    获取可用的持仓位置数量
    """
    current_positions = get_current_position_count(context)
    return max(0, g.max_positions - current_positions)

def trade(context):
    """
    主交易函数，每分钟运行一次
    """
    # 只在交易时间运行
    if not is_trading_time(context):
        return
    
    # 每天开盘时重新选股
    if context.current_dt.hour == 9 and context.current_dt.minute == 30:
        log.info("开始每日选股...")
        
        # 第一步：均线筛选
        ma_filtered = filter_stocks(context)
        log.info("均线筛选后股票数量: {}".format(len(ma_filtered)))
        
        # 第二步：周线KD筛选
        kd_filtered = filter_by_weekly_kd(context, ma_filtered)
        log.info("周线KD筛选后股票数量: {}".format(len(kd_filtered)))
        
        g.selected_stocks = kd_filtered
        g.stock_pool = kd_filtered  # 正确使用g.stock_pool变量
    
    # 处理现有持仓
    positions_to_remove = []
    for stock in list(context.portfolio.positions.keys()):
        if context.portfolio.positions[stock].total_amount <= 0:
            positions_to_remove.append(stock)
            continue
            
        action = risk_management(context, stock)
        
        if action == 'sell':
            # 清仓卖出
            order_target_value(stock, 0)
            log.info("执行止损/止盈，卖出股票: {}".format(stock))
            positions_to_remove.append(stock)
            
        elif action == 'reduce':
            # 减仓一半
            current_value = context.portfolio.positions[stock].value
            order_target_value(stock, current_value * 0.5)
            log.info("连涨三天减仓一半: {}".format(stock))
    
    # 更新持仓列表，移除已清仓的股票
    for stock in positions_to_remove:
        if stock in context.portfolio.positions:
            del context.portfolio.positions[stock]
    
    # 检查买入信号
    available_cash = context.portfolio.cash * 0.95  # 保留5%现金
    available_slots = get_available_position_slots(context)  # 获取可用持仓位置
    
    if available_slots <= 0:
        return  # 持仓已满，不买入新股票
    
    # 使用g.stock_pool变量进行遍历
    for stock in g.stock_pool:
        # 检查是否已经在持仓中
        if stock in context.portfolio.positions and context.portfolio.positions[stock].total_amount > 0:
            continue
            
        # 检查持仓数量限制
        if get_current_position_count(context) >= g.max_positions:
            break
        
        # 检查单只股票仓位限制
        max_single_position_value = context.portfolio.total_value * g.max_position_ratio
        if max_single_position_value <= 0:
            break
        
        # 第三步：检查日线KD金叉买入信号
        if check_daily_kd_golden_cross(context, stock):
            # 检查买卖时机规则
            if check_buy_timing_rules(context, stock):
                # 计算买入金额（不超过单只股票最大仓位和可用资金的较小值）
                buy_value = min(available_cash, max_single_position_value)
                
                if buy_value > 0:
                    order_target_value(stock, buy_value)
                    log.info("买入股票: {}, 金额: {:.2f}".format(stock, buy_value))
                    
                    # 记录买入时间
                    g.hold_days[stock] = context.current_dt.date()
                    
                    available_cash -= buy_value
                    available_slots -= 1
                    
                    if available_slots <= 0:
                        break

def is_trading_time(context):
    """
    检查是否为交易时间
    """
    current_time = context.current_dt.time()
    
    # A股交易时间：上午9:30-11:30，下午13:00-15:00
    morning_start = pd.Timestamp('09:30:00').time()
    morning_end = pd.Timestamp('11:30:00').time()
    afternoon_start = pd.Timestamp('13:00:00').time()
    afternoon_end = pd.Timestamp('15:00:00').time()
    
    return ((morning_start <= current_time <= morning_end) or 
            (afternoon_start <= current_time <= afternoon_end))

def after_trading_end(context):
    """
    收盘后运行函数，用于记录和统计
    """
    # 计算当日收益
    total_value = context.portfolio.total_value
    daily_return = (total_value - context.portfolio.starting_cash) / context.portfolio.starting_cash
    
    log.info("当日总资产: {:.2f}, 累计收益率: {:.2%}".format(total_value, daily_return))
    
    # 记录持仓情况
    if len(context.portfolio.positions) > 0:
        log.info("当前持仓:")
        for stock, position in context.portfolio.positions.items():
            if position.total_amount > 0:  # 只显示有效持仓
                profit_rate = (position.price - position.avg_cost) / position.avg_cost
                log.info("{}: {:.0f}股, 市值: {:.2f}, 盈亏: {:.2%}".format(
                    stock, position.total_amount, position.value, profit_rate))

# 策略说明：
# 本策略基于用户提供的三重过滤系统：
# 1. 均线过滤：120日均线在240日均线之上，且两条均线都向上
# 2. KD过滤：周线KD出现金叉
# 3. 买入信号：日线KD出现金叉时买入
# 
# 同时结合了用户提供的买卖时机规则进行风险控制
# 策略每1分钟运行一次，实时把握买卖时机
