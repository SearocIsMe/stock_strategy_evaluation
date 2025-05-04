# coding=utf-8
from jqdata import *
import numpy as np
import pandas as pd
import talib
import datetime

'''
策略说明：
本策略是将Supertrend策略移植到聚宽(JoinQuant)平台的实现
策略逻辑：
1. 使用三重Supertrend指标过滤 + EMA144 + Ichimoku云层
2. 周线均线 + 放量 + 换手率 + 筹码集中度
3. RSI过滤避免超买
4. 止盈止损机制
'''

# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('初始化策略开始')
    
    # 股票类每笔交易时的手续费是：买入时佣金万分之2.5，卖出时佣金万分之2.5，印花税千分之1
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.00025, close_commission=0.00025, close_today_commission=0, min_commission=5), type='stock')
    
    # 设定滑点
    set_slippage(PriceRelatedSlippage(0.00246))
    
    # 初始化全局变量
    g.stock_num = 10  # 持仓股票数量
    g.days = 0  # 交易日计数器
    g.refresh_rate = 5  # 调仓频率，表示调仓间隔天数
    
    # Supertrend参数
    g.st1_factor = 3
    g.st1_period = 12
    g.st2_factor = 1
    g.st2_period = 10
    g.st3_factor = 2
    g.st3_period = 11
    
    # EMA参数
    g.ema_length = 144
    
    # Ichimoku云层参数
    g.conversion_periods = 9
    g.base_periods = 26
    g.lagging_span_periods = 52
    
    # 周线均线参数
    g.ma1_period = 8
    g.ma2_period = 21
    
    # 量能参数
    g.volume_multiplier = 1.3
    g.turnover_threshold = 3.0
    g.volume_ma_period = 20
    
    # 筹码集中度参数
    g.chip_concentration_threshold = 30
    
    # RSI参数
    g.rsi_period = 14
    g.rsi_overbought = 60
    
    # 止盈止损参数
    g.take_profit = 0.15  # 15%止盈
    g.stop_loss = 0.10    # 10%止损
    
    # 设置交易时间
    run_daily(market_open, time='open', reference_security='000300.XSHG')
    run_daily(market_close, time='close', reference_security='000300.XSHG')
    
    # 设置定时运行函数
    run_daily(daily_operation, time='9:30', reference_security='000300.XSHG')
    
    # 记录持仓信息
    g.positions = {}  # 记录持仓信息 {股票代码: {'entry_price': 价格, 'entry_date': 日期}}
    
    log.info('初始化策略完成')

# 开盘时运行函数
def market_open(context):
    pass

# 收盘时运行函数
def market_close(context):
    # 记录每日收盘后的持仓信息
    record_position_info(context)

# 每日运行函数
def daily_operation(context):
    # 计数器加1
    g.days += 1
    
    # 如果不是调仓日，不进行操作
    if g.days % g.refresh_rate != 0:
        return
    
    log.info('今日进行调仓操作')
    
    # 获取持仓股票
    current_positions = context.portfolio.positions
    
    # 检查现有持仓是否需要卖出
    check_sell_signals(context, current_positions)
    
    # 如果有空仓位，选择新的股票买入
    available_slots = g.stock_num - len(context.portfolio.positions)
    if available_slots > 0:
        buy_stocks(context, available_slots)

# 检查卖出信号
def check_sell_signals(context, positions):
    for stock in list(positions.keys()):
        if positions[stock].closeable_amount <= 0:
            continue
            
        # 获取股票数据
        stock_data = get_stock_data(context, stock, 60)
        if stock_data is None or len(stock_data) < 60:
            continue
            
        current_price = stock_data['close'][-1]
        position = positions[stock]
        
        # 检查止盈止损条件
        if stock in g.positions:
            entry_price = g.positions[stock]['entry_price']
            profit_pct = (current_price / entry_price) - 1
            
            # 止盈
            if profit_pct >= g.take_profit:
                log.info(f"股票{stock}达到止盈条件，收益率: {profit_pct*100:.2f}%")
                order_target(stock, 0)
                if stock in g.positions:
                    del g.positions[stock]
                continue
                
            # 止损
            if profit_pct <= -g.stop_loss:
                log.info(f"股票{stock}达到止损条件，收益率: {profit_pct*100:.2f}%")
                order_target(stock, 0)
                if stock in g.positions:
                    del g.positions[stock]
                continue
        
        # 检查技术指标卖出信号
        sell_signal = generate_sell_signal(stock_data)
        if sell_signal:
            log.info(f"股票{stock}产生卖出信号")
            order_target(stock, 0)
            if stock in g.positions:
                del g.positions[stock]

# 选择股票并买入
def buy_stocks(context, available_slots):
    if available_slots <= 0:
        return
        
    # 获取股票池
    stocks = get_filtered_stocks(context)
    
    # 计算每只股票的信号和评分
    stock_signals = {}
    for stock in stocks:
        # 获取股票数据
        stock_data = get_stock_data(context, stock, 60)
        if stock_data is None or len(stock_data) < 60:
            continue
            
        # 生成买入信号
        buy_signal, score = generate_buy_signal(stock_data)
        
        if buy_signal:
            stock_signals[stock] = score
    
    # 按评分排序
    sorted_stocks = sorted(stock_signals.items(), key=lambda x: x[1], reverse=True)
    
    # 计算每只股票的买入金额
    available_cash = context.portfolio.cash
    per_stock_cash = available_cash / available_slots
    
    # 买入评分最高的股票
    for i, (stock, score) in enumerate(sorted_stocks):
        if i >= available_slots:
            break
            
        # 获取当前价格
        current_price = get_price(stock, count=1, frequency='daily', fields=['close'])['close'][0]
        
        # 计算可买入股数
        shares = int(per_stock_cash / current_price / 100) * 100
        if shares >= 100:  # 确保至少买入100股
            log.info(f"买入股票{stock}，评分: {score:.2f}，价格: {current_price:.2f}，股数: {shares}")
            order(stock, shares)
            
            # 记录买入信息
            g.positions[stock] = {
                'entry_price': current_price,
                'entry_date': context.current_dt.strftime('%Y-%m-%d')
            }

# 获取过滤后的股票池
def get_filtered_stocks(context):
    # 获取所有A股
    stocks = list(get_all_securities(['stock']).index)
    
    # 过滤ST股票和新上市股票
    stocks = filter_st_stocks(stocks)
    stocks = filter_new_stocks(context, stocks, 60)
    
    # 过滤基本面
    stocks = filter_by_fundamentals(context, stocks)
    
    return stocks

# 过滤ST股票
def filter_st_stocks(stocks):
    current_data = get_current_data()
    return [stock for stock in stocks if not current_data[stock].is_st and not current_data[stock].paused]

# 过滤新上市股票
def filter_new_stocks(context, stocks, days):
    return [stock for stock in stocks if (context.current_dt.date() - get_security_info(stock).start_date).days > days]

# 基本面筛选
def filter_by_fundamentals(context, stocks):
    # 获取最新财务数据
    q = query(
        valuation.code,
        valuation.pe_ratio,
        valuation.pb_ratio,
        indicator.gross_profit_margin,
        indicator.roe,
        indicator.roa
    ).filter(
        valuation.code.in_(stocks),
        indicator.gross_profit_margin > 20,  # 毛利率 > 20%
        indicator.roe > 5,                   # ROE > 5%
        valuation.pe_ratio < 100,            # PE < 100
        valuation.pb_ratio < 14              # PB < 14
    )
    
    df = get_fundamentals(q, date=context.current_dt.date())
    return list(df['code'])

# 获取股票数据
def get_stock_data(context, stock, days):
    end_date = context.current_dt
    start_date = end_date - datetime.timedelta(days=days*2)  # 获取更多数据用于计算指标
    
    try:
        # 获取日线数据
        daily_data = get_price(stock, start_date=start_date, end_date=end_date, frequency='daily', 
                              fields=['open', 'high', 'low', 'close', 'volume', 'money', 'factor'])
        
        if daily_data is None or len(daily_data) < days:
            return None
            
        # 计算技术指标
        daily_data = calculate_indicators(context, daily_data, stock)
        
        # 获取周线数据
        weekly_data = get_price(stock, start_date=start_date, end_date=end_date, frequency='W',
                                fields=['open', 'high', 'low', 'close', 'volume'])
        
        if weekly_data is not None and len(weekly_data) > g.ma2_period:
            # 计算周线指标
            daily_data = calculate_weekly_indicators(daily_data, weekly_data)
        
        return daily_data
    except Exception as e:
        log.error(f"获取股票{stock}数据失败: {e}")
        return None

# 计算技术指标
def calculate_indicators(context, df, stock):
    # 计算Supertrend指标
    df = calculate_supertrend(df)
    
    # 计算EMA指标
    df['ema144'] = talib.EMA(df['close'].values, timeperiod=g.ema_length)
    df['price_above_ema'] = np.where(df['close'] > df['ema144'], 1, 0)
    
    # 计算Ichimoku云层指标
    df = calculate_ichimoku(df)
    
    # 计算RSI指标
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=g.rsi_period)
    
    # 计算成交量指标
    df['volume_ma'] = talib.SMA(df['volume'].values, timeperiod=g.volume_ma_period)
    df['volume_signal'] = np.where(df['volume'] >= df['volume_ma'] * g.volume_multiplier, 1, 0)
    
    # 计算换手率
    if 'money' in df.columns:
        # 获取流通市值
        try:
            market_cap = get_fundamentals(query(valuation.circulating_market_cap).filter(valuation.code == stock), date=context.current_dt.date())['circulating_market_cap'][0] * 100000000
            df['turnover_rate'] = df['money'] / market_cap * 100
            df['turnover_signal'] = np.where(df['turnover_rate'] >= g.turnover_threshold, 1, 0)
        except:
            df['turnover_rate'] = 0
            df['turnover_signal'] = 0
    else:
        df['turnover_rate'] = 0
        df['turnover_signal'] = 0
    
    # 计算筹码集中度（简化版）
    df = calculate_chip_concentration(df)
    
    return df

# 计算Supertrend指标
def calculate_supertrend(df):
    # 计算Supertrend 1
    df['st1'], df['st1_dir'] = supertrend(df, g.st1_factor, g.st1_period)
    df['uptrend1'] = np.where(df['st1_dir'] < 0, 1, 0)
    
    # 计算Supertrend 2
    df['st2'], df['st2_dir'] = supertrend(df, g.st2_factor, g.st2_period)
    df['uptrend2'] = np.where(df['st2_dir'] < 0, 1, 0)
    
    # 计算Supertrend 3
    df['st3'], df['st3_dir'] = supertrend(df, g.st3_factor, g.st3_period)
    df['uptrend3'] = np.where(df['st3_dir'] < 0, 1, 0)
    
    # 计算综合上升趋势
    df['all_uptrend'] = np.where(
        (df['uptrend1'] == 1) & (df['uptrend2'] == 1) & (df['uptrend3'] == 1),
        1, 0
    )
    
    return df

# Supertrend计算函数
def supertrend(df, factor, period):
    # 计算ATR
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
    atr = pd.Series(tr).rolling(period).mean().values
    
    # 计算上下轨
    hl2 = (high + low) / 2
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)
    
    # 计算Supertrend
    supertrend = np.zeros_like(close)
    direction = np.zeros_like(close)
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
            direction[i] = -1  # 上升趋势
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
            direction[i] = 1   # 下降趋势
    
    return pd.Series(supertrend), pd.Series(direction)

# 计算Ichimoku云层指标
def calculate_ichimoku(df):
    # 计算转换线 (Conversion Line)
    high9 = df['high'].rolling(window=g.conversion_periods).max()
    low9 = df['low'].rolling(window=g.conversion_periods).min()
    df['conversion_line'] = (high9 + low9) / 2
    
    # 计算基准线 (Base Line)
    high26 = df['high'].rolling(window=g.base_periods).max()
    low26 = df['low'].rolling(window=g.base_periods).min()
    df['base_line'] = (high26 + low26) / 2
    
    # 计算先行带1 (Leading Span A)
    df['lead_line1'] = ((df['conversion_line'] + df['base_line']) / 2).shift(g.base_periods)
    
    # 计算先行带2 (Leading Span B)
    high52 = df['high'].rolling(window=g.lagging_span_periods).max()
    low52 = df['low'].rolling(window=g.lagging_span_periods).min()
    df['lead_line2'] = ((high52 + low52) / 2).shift(g.base_periods)
    
    # 计算延迟线 (Lagging Span)
    df['lagging_span'] = df['close'].shift(-g.base_periods)
    
    # 价格是否在云层上方
    df['price_above_cloud'] = np.where(
        (df['close'] > df['lead_line1']) & (df['close'] > df['lead_line2']),
        1, 0
    )
    
    return df

# 计算周线指标
def calculate_weekly_indicators(daily_df, weekly_df):
    # 计算周线EMA
    weekly_df['ma1'] = talib.EMA(weekly_df['close'].values, timeperiod=g.ma1_period)
    weekly_df['ma2'] = talib.EMA(weekly_df['close'].values, timeperiod=g.ma2_period)
    
    # 获取最新的周线数据
    latest_ma1 = weekly_df['ma1'].iloc[-1]
    latest_ma2 = weekly_df['ma2'].iloc[-1]
    latest_close = weekly_df['close'].iloc[-1]
    
    # 判断周线条件
    weekly_condition = 1 if latest_close > latest_ma1 and latest_ma1 > latest_ma2 else 0
    
    # 将周线条件应用到日线数据
    daily_df['weekly_condition'] = weekly_condition
    
    return daily_df

# 计算筹码集中度（简化版）
def calculate_chip_concentration(df):
    # 初始化筹码集中度列
    df['concentration_ratio'] = np.nan
    df['dominant_price'] = np.nan
    df['cost_valid'] = 0
    df['cost_valid_ratio'] = 0
    
    # 计算成交量加权平均价格（VWAP）作为主力成本
    window_size = 20  # 使用固定窗口大小简化计算
    
    for i in range(window_size, len(df)):
        # 获取历史成交量数据
        volume_array = df['volume'].iloc[:i].values
        
        # 计算总成交量和密集区成交量
        sorted_volumes = np.sort(volume_array)
        start_idx = max(0, len(sorted_volumes) - window_size)
        
        sum_total = np.sum(sorted_volumes)
        sum_dense = np.sum(sorted_volumes[start_idx:])
        
        # 计算筹码集中度
        if sum_total > 0 and sum_dense > 0:
            concentration_ratio = min(100, (sum_dense / (sum_total + 1e-6)) * 100)
            df.iloc[i, df.columns.get_loc('concentration_ratio')] = concentration_ratio
        
        # 计算主力成本价（VWAP）
        window = df.iloc[:i]
        total_vol = window['volume'].sum()
        if total_vol > 0:
            weighted_sum = (window['close'] * window['volume']).sum()
            dominant_price = weighted_sum / total_vol
            df.iloc[i, df.columns.get_loc('dominant_price')] = dominant_price
            
            # 计算价格偏离度
            prev_close = df['close'].iloc[i-1]
            if not np.isnan(dominant_price) and dominant_price > 0:
                price_deviation = abs(prev_close - dominant_price) / dominant_price * 100
                
                # 判断筹码集中度是否有效
                if concentration_ratio >= g.chip_concentration_threshold and price_deviation < (100 - g.chip_concentration_threshold):
                    df.iloc[i, df.columns.get_loc('cost_valid')] = 1
                    df.iloc[i, df.columns.get_loc('cost_valid_ratio')] = concentration_ratio / g.chip_concentration_threshold
    
    return df

# 生成买入信号
def generate_buy_signal(df):
    if len(df) < 30:  # 确保有足够的数据
        return False, 0
    
    # 获取最新数据
    current = df.iloc[-1]
    
    # 策略1: 三重Supertrend过滤 + EMA144 + Ichimoku云层
    strategy1_signal = 0
    if ('all_uptrend' in current and current['all_uptrend'] == 1 and
        'price_above_ema' in current and current['price_above_ema'] == 1 and
        'price_above_cloud' in current and current['price_above_cloud'] == 1):
        strategy1_signal = 1
    
    # 策略2: 周线均线 + 放量 + 换手率 + 筹码集中度
    strategy2_components = 0
    if 'weekly_condition' in current and current['weekly_condition'] == 1:
        strategy2_components += 1
    if 'volume_signal' in current and current['volume_signal'] == 1:
        strategy2_components += 1
    if 'turnover_signal' in current and current['turnover_signal'] == 1:
        strategy2_components += 1
    if 'cost_valid' in current and current['cost_valid'] == 1:
        strategy2_components += 1
    
    # 策略2信号: 至少满足3个条件
    strategy2_signal = 1 if strategy2_components >= 3 else 0
    
    # 综合信号: 两个策略都满足
    buy_signal = 1 if strategy1_signal == 1 and strategy2_signal == 1 else 0
    
    # RSI过滤: 避免超买
    if 'rsi' in current and current['rsi'] > g.rsi_overbought:
        buy_signal = 0
    
    # 计算评分
    score = calculate_score(current)
    
    return buy_signal == 1, score

# 生成卖出信号
def generate_sell_signal(df):
    if len(df) < 2:  # 确保有足够的数据
        return False
    
    # 获取最新数据和前一天数据
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 止损条件1: 跌破EMA144
    stop_condition1 = 'price_above_ema' in current and current['price_above_ema'] == 0
    
    # 止损条件2: 进入云层
    stop_condition2 = 'price_above_cloud' in current and current['price_above_cloud'] == 0
    
    # 止损条件3: Supertrend转势
    stop_condition3 = ('all_uptrend' in current and current['all_uptrend'] == 0 and
                      'all_uptrend' in prev and prev['all_uptrend'] == 1)
    
    # 综合止损信号
    sell_signal = stop_condition1 or stop_condition2 or stop_condition3
    
    return sell_signal

# 计算股票评分
def calculate_score(current):
    # 基础分数
    base_score = 0.0
    
    # 趋势强度 (35分)
    trend_score = 0.0
    if 'all_uptrend' in current and current['all_uptrend'] == 1:
        trend_score += 12.0
    if 'price_above_ema' in current and current['price_above_ema'] == 1:
        trend_score += 10.0
    if 'price_above_cloud' in current and current['price_above_cloud'] == 1:
        trend_score += 13.0
    
    # 量能评分 (25分)
    volume_score = 0.0
    if 'volume_signal' in current and current['volume_signal'] == 1:
        volume_score += 12.5
    if 'turnover_signal' in current and current['turnover_signal'] == 1:
        volume_score += 12.5
    
    # 筹码评分 (15分)
    chip_score = 0.0
    if 'cost_valid_ratio' in current and not pd.isna(current['cost_valid_ratio']):
        # 筹码集中度比率越高，得分越高 (最高15分)
        chip_score = min(15, current['cost_valid_ratio'] * 15)
    
    # 周线评分 (15分)
    weekly_score = 0.0
    if 'weekly_condition' in current and current['weekly_condition'] == 1:
        weekly_score = 15.0
    
    # RSI评分 (10分) - RSI越低，得分越高
    rsi_score = 0.0
    if 'rsi' in current and not pd.isna(current['rsi']):
        # RSI在30-50之间得分最高
        if current['rsi'] < 30:
            rsi_score = 5.0
        elif current['rsi'] < 50:
            rsi_score = 10.0
        elif current['rsi'] < g.rsi_overbought:
            rsi_score = 5.0
    
    # 总分 (100分)
    total_score = trend_score + volume_score + chip_score + weekly_score + rsi_score
    
    return total_score

# 记录持仓信息
def record_position_info(context):
    # 记录持仓数量
    record(position_count=len(context.portfolio.positions))
    
    # 记录仓位比例
    record(position_ratio=context.portfolio.positions_value / context.portfolio.total_value)
    
    # 记录收益率
    record(return_rate=(context.portfolio.total_value / context.portfolio.starting_cash - 1) * 100)