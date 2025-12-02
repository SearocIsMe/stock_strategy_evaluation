# 克隆自聚宽文章：https://www.joinquant.com/post/62613
# 标题：六一中路策略，短线悟道经验，欢迎大家优化
# 作者：铜江公园

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from datetime import timedelta
from jqlib.technical_analysis import *
from jqdata import *

# --------------------【情绪周期整合】开始：新增情绪指标计算模块 --------------------
def get_market_sentiment_indicators(context, check_date=None):
    """
    计算市场情绪核心指标。
    该函数应在每日开盘前（如 before_trading_start）调用，计算前一交易日的情绪数据。

    :param context: 策略上下文
    :param check_date: 要计算指标的日期，格式为 'YYYY-MM-DD'。如果为 None，则默认为 context.previous_date。
    :return: 包含五大核心情绪指标的字典。
    """
    if check_date is None:
        check_date = context.previous_date

    # 1. 获取所有A股列表，并排除ST、*ST、北交所、科创板及当日上市新股
    all_stocks = get_all_securities(['stock'], date=check_date)
    all_stocks = all_stocks[all_stocks.start_date < check_date] # 过滤掉当天或未来上市的
    all_stocks = all_stocks[~all_stocks['display_name'].str.contains('ST')]
    all_stocks = all_stocks[~all_stocks['display_name'].str.contains('退')]
    all_stocks = all_stocks[~all_stocks.index.str.startswith('688')] # 排除科创板
    all_stocks = all_stocks[~all_stocks.index.str.startswith('8')] # 排除北交所
    
    stock_list = list(all_stocks.index)

    # 获取前两个交易日的数据用于计算
    df_prices = get_price(stock_list, end_date=check_date, count=2, frequency='daily', 
                        fields=['open', 'close', 'high', 'low', 'high_limit', 'low_limit', 'volume', 'money'], 
                        skip_paused=False, fq='pre', panel=False)
    
    if df_prices.empty:
        return {
            '最高连板高度': 0, '涨停家数': 0, '昨日涨停表现': 0.0, '炸板率': 0.0, '跌停家数': 0
        }

    today_data = df_prices[df_prices.time.dt.date == pd.to_datetime(check_date).date()].set_index('code')
    today_data = today_data[today_data['volume'] > 0]

    yesterday_data = df_prices[df_prices.time.dt.date < pd.to_datetime(check_date).date()]
    if not yesterday_data.empty:
        yesterday_data = yesterday_data.set_index('code')
    
    limit_up_stocks = today_data[today_data['close'] >= today_data['high_limit']]
    limit_up_count = len(limit_up_stocks)

    limit_down_stocks = today_data[today_data['close'] <= today_data['low_limit']]
    limit_down_count = len(limit_down_stocks)

    touched_limit_up = today_data[today_data['high'] >= today_data['high_limit']]
    blown_board_stocks = touched_limit_up[touched_limit_up['close'] < touched_limit_up['high_limit']]
    blown_board_rate = len(blown_board_stocks) / len(touched_limit_up) if len(touched_limit_up) > 0 else 0.0

    yesterday_limit_up_performance = 0.0
    if not yesterday_data.empty:
        yesterday_limit_up_stocks = yesterday_data[yesterday_data['close'] >= yesterday_data['high_limit']]
        today_performance_stocks = today_data.loc[today_data.index.isin(yesterday_limit_up_stocks.index)]
        if not today_performance_stocks.empty:
            yesterday_closes = yesterday_data.loc[today_performance_stocks.index]['close']
            today_opens = today_performance_stocks['open']
            performance = (today_opens - yesterday_closes) / yesterday_closes
            yesterday_limit_up_performance = performance.mean()

    highest_board = 0
    if not limit_up_stocks.empty:
        limit_up_list = list(limit_up_stocks.index)
        # 【修正】使用 get_price 替代 history，因为需要获取多个字段
        history_prices_df = get_price(security=limit_up_list, count=15, end_date=check_date, frequency='1d', 
                                    fields=['close', 'high_limit'], skip_paused=True, panel=False, fq='pre')
        
        for stock in limit_up_list:
            current_board = 1 # 当天已涨停，所以从1开始
            
            # 筛选出该股票的历史数据并按时间降序排列
            stock_history = history_prices_df[history_prices_df['code'] == stock].sort_values('time', ascending=False)
            
            # 从昨天（stock_history的第二行）开始往前检查是否连续涨停
            for idx, row in stock_history.iloc[1:].iterrows():
                if row['close'] >= row['high_limit']:
                    current_board += 1
                else:
                    break # 一旦中断，就停止计数
            
            if current_board > highest_board:
                highest_board = current_board

    return {
        '最高连板高度': highest_board, '涨停家数': limit_up_count, '昨日涨停表现': yesterday_limit_up_performance,
        '炸板率': blown_board_rate, '跌停家数': limit_down_count
    }

def determine_market_phase(indicators):
    """根据情绪指标判断市场阶段"""
    if indicators['炸板率'] > 0.40 and indicators['跌停家数'] > 10:
        return '退潮期'
    if indicators['最高连板高度'] <= 3 and indicators['跌停家数'] > indicators['涨停家数']:
        return '冰点期'
    if indicators['炸板率'] > 0.35 or (indicators['跌停家数'] > 5 and indicators['最高连板高度'] > 6):
        return '高潮期'
    if indicators['最高连板高度'] >= 5 and indicators['昨日涨停表现'] > 0.02:
        return '发酵期'
    if indicators['最高连板高度'] > 3 and indicators['涨停家数'] < 80 and indicators['昨日涨停表现'] > 0.01:
        return '启动期'
    return '震荡期' # 默认状态
# --------------------【情绪周期整合】结束：新增情绪指标计算模块 --------------------


def initialize(context):
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_slippage(FixedSlippage(0.0001))
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.0001, close_commission=0.0001,
                            close_today_commission=0, min_commission=1), type='stock')
    #----------------Settings--------------------------------------------------/
    g.stock_num=4
    g.today_list=[]
    g.down=0.4
    g.today_bought_stocks=set()
    g.today_sold_stocks=set()
    g.last_total_value = 0
    g.sentiment_indicators = {} # 【情绪周期整合】初始化情绪指标字典
    g.market_phase = '未知'      # 【情绪周期整合】初始化市场阶段
    #----------------Settings--------------------------------------------------/
    run_daily(before_trading_start, time="09:00") # 【情绪周期整合】新增开盘前运行函数
    run_daily(perpare,time="09:26")
    run_daily(buy,time="09:27")
    run_daily(sell,time='13:00')
    run_daily(sell,time='14:55')
    run_daily(dieting, time="every_bar")
    run_daily(daily_trading_report, time="15:05")

    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('order', 'warning')

# 【情绪周期整合】新增开盘前运行函数，用于计算情绪指标
def before_trading_start(context):
    g.sentiment_indicators = get_market_sentiment_indicators(context)
    g.market_phase = determine_market_phase(g.sentiment_indicators)
    
    log.info(f"--- 市场情绪判断【{g.market_phase}】 ---")
    log.info(f"最高连板: {g.sentiment_indicators['最高连板高度']}板, 涨停: {g.sentiment_indicators['涨停家数']}家, 跌停: {g.sentiment_indicators['跌停家数']}家")
    log.info(f"昨日涨停表现: {g.sentiment_indicators['昨日涨停表现']:.2%}, 炸板率: {g.sentiment_indicators['炸板率']:.2%}")
    log.info("---------------------------------")
    
def perpare(context):#筛选
    g.dieting=[]
    current_data = get_current_data()
    g.yesterday_high_dict = {}
    g.today_list=[]
    g.today_bought_stocks=set()
    g.today_sold_stocks=set()
    stk_list=get_st(context)
    singal=today_is_between(context)
    if singal==True:
        print(f'筛选前：{len(stk_list)}')
        stk_list=GJT_filter_stocks(stk_list)
        print(f'筛选后：{len(stk_list)}')
        
    stk_list=filter_stocks(context,stk_list)
    if len(stk_list)==0:
        return
    stk_list=rzq_list(context,stk_list)
    if len(stk_list)==0:
        return

    df = get_price(
        stk_list, end_date=context.previous_date, frequency='daily',
        fields=['close'], count=1, panel=False, fill_paused=False, skip_paused=True
    ).set_index('code')
    
    open_now_values = []
    for s in stk_list:
        try:
            open_now_values.append(current_data[s].day_open)
        except KeyError as e:
            print(f"警告: 股票 {s} 的数据不可用, 错误: {e}")
            open_now_values.append(None)
    
    df['open_now'] = open_now_values
    df = df.dropna(subset=['open_now'])
    df = df[(df['open_now'] / df['close']) < 1.01]
    df = df[(df['open_now'] / df['close']) > 0.95]
    
    stk_list = list(df.index)
    hold_list = list(context.portfolio.positions)
    stk_list = list(set(stk_list) - set(hold_list))
    
    if len(stk_list) == 0:
        return
    
    df_val = get_valuation(
        stk_list, start_date=context.previous_date, end_date=context.previous_date,
        fields=['turnover_ratio', 'market_cap', 'circulating_market_cap']
    )
    
    df.index = df.index.astype(str)
    df_val['code'] = df_val['code'].astype(str)
    df_combined = pd.merge(df.reset_index(), df_val, on='code')
    df_combined['factor'] = df_combined['turnover_ratio'] * (df_combined['open_now'] / df_combined['close'])
    df_sorted = df_combined.sort_values(by='factor', ascending=False)
    g.today_list = list(df_sorted['code'])
    log.info(f"股池数: {len(stk_list)}")

def sell(context):
    # 【情绪周期整合】退潮期无条件清仓
    if g.market_phase == '退潮期':
        log.info(f"情绪周期为【{g.market_phase}】，执行清仓操作。")
        for s in list(context.portfolio.positions.keys()):
            pos = context.portfolio.positions[s]
            if pos.closeable_amount > 0:
                order = order_target_value(s, 0)
                if order and order.filled > 0:
                    g.today_sold_stocks.add(s)
        return # 清仓后直接返回，不执行后续卖出逻辑

    hold_list = list(context.portfolio.positions)
    if not hold_list:
        return
        
    current_data = get_current_data()
    yesterday = context.previous_date
    
    sellable_list = []
    for s in hold_list:
        pos = context.portfolio.positions[s]
        if s not in g.today_bought_stocks and pos.closeable_amount > 0:
            sellable_list.append(s)
    
    if not sellable_list:
        log.info('没有可卖出的股票（全部为当日买入）')
        return
    
    hist_data = get_price(
        sellable_list, end_date=yesterday, frequency='daily',
        fields=['close'], count=8, panel=False
    )
    
    ma7_data = hist_data.groupby('code')['close'].apply(lambda x: x.rolling(7).mean().iloc[-1]).to_dict()
    
    df_history = get_price(
        sellable_list, end_date=yesterday, frequency='daily',
        fields=['close', 'high_limit'], count=1, panel=False
    )
    
    df_history['avg_cost'] = [context.portfolio.positions[s].avg_cost for s in sellable_list]
    df_history['price'] = [context.portfolio.positions[s].price for s in sellable_list]
    df_history['high_limit'] = [current_data[s].high_limit for s in sellable_list]
    df_history['low_limit'] = [current_data[s].low_limit for s in sellable_list]
    df_history['last_price'] = [current_data[s].last_price for s in sellable_list]
    df_history['ma7'] = [ma7_data.get(s, 0) for s in sellable_list]
    df_history['closeable_amount'] = [context.portfolio.positions[s].closeable_amount for s in sellable_list]
    
    cond1 = (df_history['last_price'] != df_history['high_limit'])
    cond2_1 = df_history['last_price'] < df_history['ma7']
    ret_matrix = (df_history['price'] / df_history['avg_cost'] - 1) * 100
    cond2_2 = ret_matrix > 0
    cond2_3 = (df_history['close'] == df_history['high_limit'])
    
    sell_condition = cond1 & (cond2_1 | cond2_2 | cond2_3)
    
    sell_list = df_history[
        sell_condition & 
        (df_history['last_price'] > df_history['low_limit']) &
        (df_history['closeable_amount'] > 0)
    ].code.tolist()
    
    for s in sell_list:
        pos = context.portfolio.positions[s]
        if pos.closeable_amount <= 0 or current_data[s].paused:
            continue
        
        order = order_target_value(s, 0)
        if order and order.filled > 0:
            g.today_sold_stocks.add(s)
            log.info(f'卖出 {s} | 成本价:{pos.avg_cost:.2f} 现价:{pos.price:.2f} 可卖:{pos.closeable_amount}股 成交:{order.filled}股')
        else:
            log.warn(f'卖出失败 {s} | 成本价:{pos.avg_cost:.2f} 现价:{pos.price:.2f} 可卖:{pos.closeable_amount}股')
        print('-'*50)

def buy(context):
    # 【情绪周期整合】根据情绪周期调整买入行为
    if g.market_phase in ['退潮期']:
        log.info(f"情绪周期为【{g.market_phase}】，今日不进行任何买入操作。")
        return
    
    target=filter_stocks_by_b_s(context,g.today_list)
    
    hold_list = list(context.portfolio.positions)
    
    # 【情绪周期整合】根据情绪周期调整持仓数量
    if g.market_phase == '冰点期':
        log.info("情绪周期为【冰点期】，执行小仓位试错，最多持仓1只。")
        adjusted_stock_num = 1
    else:
        adjusted_stock_num = g.stock_num

    num = adjusted_stock_num - len(hold_list)
    if num <= 0:
        return
        
    target=[x for x in target if x not in hold_list][:num]
    if len(target) > 0:
        value=context.portfolio.available_cash
        cash_per_stock = value / num
        current_data = get_current_data()
        for stock in target:
            if current_data[stock].paused or \
            current_data[stock].last_price==current_data[stock].low_limit or \
            current_data[stock].last_price==current_data[stock].high_limit:
                continue
            order = order_value(stock, cash_per_stock)
            if order and order.filled > 0:
                g.today_bought_stocks.add(stock)
                log.info (f"买入 {stock} 成交:{order.filled}股")
                
#----------------函数群--------------------------------------------------/    
def filter_stocks_by_b_s(context,stock_list):
    date= context.current_dt.strftime("%Y-%m-%d")
    valid_stocks = []
    for stock in stock_list:
        auction_df = get_call_auction(stock, start_date=date, end_date=date)
        if auction_df is None or auction_df.empty:
            continue
        auction_df = auction_df.assign(
            sellmoney=lambda df: df['a1_p']*df['a1_v'] + df['a2_p']*df['a2_v'] + df['a3_p']*df['a3_v'] + df['a4_p']*df['a4_v'] + df['a5_p']*df['a5_v'],
            buymoney=lambda df: df['b1_p']*df['b1_v'] + df['b2_p']*df['b2_v'] + df['b3_p']*df['b3_v'] + df['b4_p']*df['b4_v'] + df['b5_p']*df['b5_v']
        ).assign(b_s=lambda df: (df['buymoney'] - df['sellmoney']) / df['sellmoney'])
        if not auction_df.empty and auction_df['b_s'].iloc[0] > 0:
            valid_stocks.append(stock)
    return valid_stocks

def today_is_between(context):
    today = context.current_dt.strftime('%m-%d')
    return ('01-15' <= today <= '01-31') or ('04-15' <= today <= '04-30') or ('12-15' <= today <= '12-31')

def get_st(context):
    stocks = get_index_stocks('399101.XSHE', date=context.previous_date)
    st_data = get_extras('is_st', stocks, count=1, end_date=context.previous_date).T
    st_data.columns = ['is_st']
    return st_data[st_data['is_st'] == False].index.tolist()

def get_shifted_date(date, days, days_type='T'):
    d_date = transform_date(date, 'd')
    yesterday = d_date + dt.timedelta(-1)
    if days_type == 'N':
        return str(yesterday + dt.timedelta(days+1))
    if days_type == 'T':
        all_trade_days = [i.strftime('%Y-%m-%d') for i in list(get_all_trade_days())]
        if str(yesterday) in all_trade_days:
            return all_trade_days[all_trade_days.index(str(yesterday)) + days + 1]
        else:
            for i in range(100):
                last_trade_date = yesterday - dt.timedelta(i)
                if str(last_trade_date) in all_trade_days:
                    return all_trade_days[all_trade_days.index(str(last_trade_date)) + days + 1]
    return None

def transform_date(date, date_type):
    if isinstance(date, str):
        dt_date = dt.datetime.strptime(date, '%Y-%m-%d')
    elif isinstance(date, dt.datetime):
        dt_date = date
    elif isinstance(date, dt.date):
        dt_date = dt.datetime.combine(date, dt.time())
    dct = {'str': dt_date.strftime('%Y-%m-%d'), 'dt': dt_date, 'd': dt_date.date()}
    return dct[date_type]

def get_ever_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close','high','high_limit'], count=1, panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()
    return list(df[df['close'] != df['high_limit']].code)

def get_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close','low','high_limit'], count=1, panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()
    return list(df[df['close'] == df['high_limit']].code)

def rzq_list(context,initial_list): 
    date = transform_date(context.previous_date, 'str')
    date_1=get_shifted_date(date, -1, 'T')
    h1_list = get_ever_hl_stock(initial_list, date)
    elements_to_remove = get_hl_stock(initial_list, date_1)
    return [stock for stock in h1_list if stock in elements_to_remove]
    
def filter_stocks(context, stocks):
    yesterday = context.previous_date
    df = get_price(stocks, count=11, frequency='1d', fields=['close', 'low', 'volume'], end_date=yesterday, panel=False).reset_index()
    valid_stocks = []
    for code, group in df.groupby('code'):
        if len(group) < 11: continue
        group = group.copy()
        group['ma10'] = group['close'].rolling(10).mean()
        group['prev_low'] = group['low'].shift(1)
        group['prev_volume'] = group['volume'].shift(1)
        last_row = group.iloc[-1]
        if not pd.isna(last_row[['ma10', 'prev_low', 'prev_volume']]).any() and \
        last_row['close'] > last_row['prev_low'] and \
        last_row['close'] > last_row['ma10'] and \
        last_row['volume'] > last_row['prev_volume'] and \
        last_row['volume'] < 10 * last_row['prev_volume'] and \
        last_row['close'] > 1:
            valid_stocks.append(code)
    return valid_stocks

def GJT_filter_stocks(stocks):
    q = query(
        valuation.code, income.np_parent_company_owners, income.net_profit,
        income.operating_revenue
    ).filter(
        valuation.code.in_(stocks), income.np_parent_company_owners > 0,
        income.net_profit > 0, income.operating_revenue > 1e8,
        indicator.roe>0, indicator.roa>0
    )
    return list(get_fundamentals(q).code)

def dieting(context):
    current_data = get_current_data()
    for s in list(context.portfolio.positions) :
        if s not in g.dieting:
            dtj=current_data[s].low_limit
            zxj=current_data[s].last_price
            if zxj==dtj and (context.portfolio.positions[s].closeable_amount != 0):
                if s not in g.dieting:
                    g.dieting.append(s)
    g.dieting=list(set(g.dieting))
    if len(g.dieting)>0:
        for s in g.dieting[:]:
            dtj=current_data[s].low_limit
            zxj=current_data[s].last_price
            if zxj>dtj:
                pos = context.portfolio.positions[s]
                closeable_amount = pos.closeable_amount
                if closeable_amount <= 0:
                    log.warn(f'跌停打开跳过卖出 {s}: 可卖数量不足: {closeable_amount}')
                    g.dieting.remove(s)
                    continue
                if current_data[s].paused:
                    log.warn(f'跌停打开跳过卖出 {s}: 股票停牌')
                    continue
                order = order_target_value(s, 0)
                if order and order.filled > 0:
                    g.today_sold_stocks.add(s)
                    log.info(f"跌停打开卖出:{s} 成交:{order.filled}股")
                else:
                    log.warn(f"跌停打开卖出失败:{s} 可卖:{closeable_amount}股")
                g.dieting.remove(s)

def daily_trading_report(context):
    try:
        current_date = context.current_dt.strftime('%Y-%m-%d')
        total_value = context.portfolio.total_value
        cash = context.portfolio.cash
        
        daily_return = total_value - g.last_total_value if g.last_total_value > 0 else 0
        daily_return_pct = (daily_return / g.last_total_value * 100) if g.last_total_value > 0 else 0
        g.last_total_value = total_value
        
        log.info(f"📊【{current_date} 交易报告】| 情绪周期:【{g.market_phase}】") # 【情绪周期整合】在报告中加入情绪阶段
        log.info(f"💰 总资产: {total_value:,.2f}元 | 现金: {cash:,.2f}元")
        log.info(f"📈 当日收益: {daily_return:+,.2f}元 ({daily_return_pct:+.2f}%)")
        
        log.info(f"📈【当日买入】{len(g.today_bought_stocks)}只: {', '.join(g.today_bought_stocks) or '无'}")
        log.info(f"📉【当日卖出】{len(g.today_sold_stocks)}只: {', '.join(g.today_sold_stocks) or '无'}")
        
        positions_info = []
        for code, pos in context.portfolio.positions.items():
            if pos.total_amount > 0:
                stock_name = get_security_info(code).display_name
                profit_loss = (pos.price - pos.avg_cost) * pos.total_amount
                profit_loss_pct = (pos.price / pos.avg_cost - 1) * 100 if pos.avg_cost > 0 else 0
                positions_info.append(f"  {stock_name}({code}): {pos.total_amount}股 | 成本:{pos.avg_cost:.2f} | 现价:{pos.price:.2f} | 市值:{pos.value:,.2f}元 | 盈亏:{profit_loss:+,.2f}元 ({profit_loss_pct:+.2f}%)")
        
        if positions_info:
            log.info(f"📋【持仓详情】共{len(positions_info)}只股票")
            for info in positions_info:
                log.info(info)
        else:
            log.info("📋【持仓详情】当前无持仓")
        
        if g.dieting:
            log.info(f"⚠️【跌停监控】{len(g.dieting)}只股票跌停: {', '.join(g.dieting)}")
        else:
            log.info("✅【跌停监控】无跌停股票")
        
        log.info("=" * 80)
        
    except Exception as e:
        log.error(f"生成每日交易报告失败: {e}")