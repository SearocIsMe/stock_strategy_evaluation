# 克隆自聚宽文章：https://www.joinquant.com/post/63587
# 标题：【实盘数据分享】鹰击长空打板策略
# 作者：实测狂魔~王子玉

from jqdata import *         # 聚宽数据模块，提供基础数据API
from jqfactor import *       # 聚宽因子模块，提供因子计算功能
from jqlib.technical_analysis import *  # 聚宽技术分析模块
import datetime as dt        # 日期时间处理
import pandas as pd          # 数据处理和分析
from datetime import datetime
from datetime import timedelta

import newqmt_sql

# ⭐ 在这里设置这个策略的分类标签（写入 trade.fenlei）
newqmt_sql.FENLEI = 'eagles'      

from newqmt_sql import (
    order_zzy as order,
    order_target_zzy as order_target,
    order_value_zzy as order_value,
    order_target_value_zzy as order_target_value
)


def initialize(context):
    # 设置策略基本参数
    set_option('use_real_price', True)  # 使用真实价格，而非复权价格
    log.set_level('system', 'error')    # 设置日志级别，只显示错误信息
    set_option('avoid_future_data', True)  # 避免使用未来数据
    
    # 设置定时任务
    run_daily(get_stock_list, '9:10')         # 9:10 获取选股列表
    run_daily(buy, '09:26')                   # 9:26 执行买入操作（集合竞价结束后）
    run_daily(sell_heavy_turnover, time='10:00')  # 10:00 执行高位放量卖出
    run_daily(sell_am, time='11:25')          # 11:25 上午收盘前执行止盈
    run_daily(sell_pm, time='13:20')          # 14:50 下午收盘前执行止盈/止损


def get_stock_list(context): 

    # 获取交易日期
    date = context.previous_date  # 前一个交易日
    date_2, date_1, date = get_trade_days(end_date=date, count=3)  # 获取最近3个交易日
    
    # 获取初始股票池
    initial_list = prepare_stock_list(date)
    
    # 获取不同日期的涨停股票列表
    hl0_list = get_hl_stock(initial_list, date)       # 昨日涨停股票
    hl1_list = get_ever_hl_stock(initial_list, date_1)  # 前日曾涨停股票
    hl2_list = get_ever_hl_stock(initial_list, date_2)  # 前前日曾涨停股票
    
    # 一进二策略：昨日涨停且前两日未涨停的股票
    elements_to_remove = set(hl1_list + hl2_list)  # 合并前两日涨停股票，用于快速查找
    g.gap_up = [stock for stock in hl0_list if stock not in elements_to_remove]  # 昨日涨停且前两日未涨停
    
    # 首板低开策略：昨日首次涨停的股票
    g.gap_down = [s for s in hl0_list if s not in hl1_list]  # 昨日涨停但前日未涨停
    
    # 弱转强策略：昨日曾涨停但收盘未涨停，且前日未涨停的股票
    h1_list = get_ever_hl_stock2(initial_list, date)  # 昨日曾涨停但收盘未涨停的股票
    elements_to_remove = get_hl_stock(initial_list, date_1)  # 前日涨停的股票
    g.reversal = [stock for stock in h1_list if stock not in elements_to_remove]  # 昨日曾涨停但收盘未涨停，且前日未涨停
    

def buy(context):
 
    # 初始化股票列表
    qualified_stocks = []  
    gk_stocks = []         
    dk_stocks = []         
    rzq_stocks = []        
    
    # 获取当前市场数据和时间
    current_data = get_current_data()
    date_now = context.current_dt.strftime("%Y-%m-%d")
    mid_time1 = '09:15:00'
    end_times1 = '09:26:00'
    start = date_now + mid_time1
    end = date_now + end_times1
    
    
    for s in g.gap_up:
        # 条件一：筛选均价、成交金额、市值、换手率
        prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
        # 计算均价增长值，要求大于7%
        avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][0] * 1.1 - 1
        # 成交金额要求在5.5亿到20亿之间
        if avg_price_increase_value < 0.07 or prev_day_data['money'][0] < 5.5e8 or prev_day_data['money'][0] > 20e8:
            continue
            
        # 条件二：筛选市值，要求总市值>75亿，流通市值<520亿
        turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date, 
                                          fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
        if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < 75 or turnover_ratio_data['circulating_market_cap'][0] > 520:
            continue
            
        # 条件三：排除左压（上涨时未放量）的股票
        if rise_low_volume(s, context):
            continue
            
        # 条件四：检查集合竞价数据，要求高开且开盘比例在特定范围内
        auction_data = get_call_auction(s, start_date=date_now, end_date=date_now, fields=['time', 'volume', 'current'])
        # 集合竞价成交量要求大于昨日成交量的3%
        if auction_data.empty or auction_data['volume'][0] / prev_day_data['volume'][-1] < 0.03:
            continue
        # 开盘价相对于昨日涨停价的比例要求在1-1.06之间
        current_ratio = auction_data['current'][0] / (current_data[s].high_limit/1.1)
        if current_ratio <= 1 or current_ratio >= 1.06:
            continue
            
        # 如果股票满足所有条件，则添加到列表中
        gk_stocks.append(s)
        qualified_stocks.append(s)
    
    
    date = transform_date(context.previous_date, 'str')
    
    if g.gap_down:
        stock_list = g.gap_down
        
        # 条件一：筛选相对位置，要求在60日内的相对位置<=0.5
        rpd = get_relative_position_df(stock_list, date, 60)
        rpd = rpd[rpd['rp'] <= 0.5]  # 相对位置低于0.5的股票
        stock_list = list(rpd.index)

        # 条件二：筛选低开幅度，要求低开3%-4.5%
        if len(stock_list) != 0:
            df = get_price(stock_list, end_date=date, frequency='daily', fields=['close'], 
                          count=1, panel=False, fill_paused=False, skip_paused=True).set_index('code')
            df['open_pct'] = [current_data[s].day_open/df.loc[s, 'close'] for s in stock_list]
            df = df[(0.955 <= df['open_pct']) & (df['open_pct'] <= 0.97)]  # 低开3%-4.5%
            stock_list = list(df.index)

        # 条件三：筛选成交金额，要求大于1亿
        for s in stock_list:
            prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            if prev_day_data['money'][0] >= 1e8:
                dk_stocks.append(s)
                qualified_stocks.append(s)
        
    
    for s in g.reversal:
        # 条件一：过滤前3天涨幅超过20%的股票
        price_data = attribute_history(s, 4, '1d', fields=['close'], skip_paused=True)
        if len(price_data) < 4:
            continue
        increase_ratio = (price_data['close'][-1] - price_data['close'][0]) / price_data['close'][0]
        if increase_ratio > 0.20:
            continue
        
        # 条件二：过滤前一日收盘价小于开盘价5%以上的股票（大幅低走）
        prev_day_data = attribute_history(s, 1, '1d', fields=['open', 'close'], skip_paused=True)
        if len(prev_day_data) < 1:
            continue
        open_close_ratio = (prev_day_data['close'][0] - prev_day_data['open'][0]) / prev_day_data['open'][0]
        if open_close_ratio < -0.05:
            continue
        
        # 条件三：筛选均价和成交金额
        prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
        avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][0] - 1
        # 均价增长值>-4%，成交金额在3亿到19亿之间
        if avg_price_increase_value < -0.04 or prev_day_data['money'][0] < 3e8 or prev_day_data['money'][0] > 19e8:
            continue
            
        # 条件四：筛选市值，要求总市值>75亿，流通市值<520亿
        turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date, 
                                          fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
        if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < 75 or turnover_ratio_data['circulating_market_cap'][0] > 520:
            continue

        # 条件五：排除左压（上涨时未放量）的股票
        if rise_low_volume(s, context):
            continue
            
        # 条件六：检查集合竞价数据
        auction_data = get_call_auction(s, start_date=date_now, end_date=date_now, fields=['time', 'volume', 'current'])
        # 集合竞价成交量要求大于昨日成交量的3%
        if auction_data.empty or auction_data['volume'][0] / prev_day_data['volume'][-1] < 0.03:
            continue
        # 开盘价相对于昨日涨停价的比例要求在0.98-1.09之间
        current_ratio = auction_data['current'][0] / (current_data[s].high_limit/1.1)
        if current_ratio <= 0.98 or current_ratio >= 1.09:
            continue
            
        # 如果股票满足所有条件，则添加到列表中
        rzq_stocks.append(s)
        qualified_stocks.append(s)
    
    
    # 仅当有符合条件的股票且可用现金占总资产比例>30%时执行买入
    if len(qualified_stocks) != 0 and context.portfolio.available_cash/context.portfolio.total_value > 0.3:
        # 计算每只股票的买入金额，平均分配可用资金
        value = context.portfolio.available_cash / len(qualified_stocks)
        for s in qualified_stocks:
            # 确保有足够资金买入至少100股
            if context.portfolio.available_cash/current_data[s].last_price > 100: 
                # 以开盘价买入
                order_value(s, value, MarketOrderStyle(current_data[s].day_open))
                

# ================================================
# 日期处理相关函数
# ================================================
def transform_date(date, date_type):

    if type(date) == str:
        str_date = date
        dt_date = dt.datetime.strptime(date, '%Y-%m-%d')
        d_date = dt_date.date()
    elif type(date) == dt.datetime:
        str_date = date.strftime('%Y-%m-%d')
        dt_date = date
        d_date = dt_date.date()
    elif type(date) == dt.date:
        str_date = date.strftime('%Y-%m-%d')
        dt_date = dt.datetime.strptime(str_date, '%Y-%m-%d')
        d_date = date
    dct = {'str': str_date, 'dt': dt_date, 'd': d_date}
    return dct[date_type]

def get_shifted_date(date, days, days_type='T'):

    # 获取上一个自然日
    d_date = transform_date(date, 'd')
    yesterday = d_date + dt.timedelta(-1)
    
    # 按自然日平移
    if days_type == 'N':
        shifted_date = yesterday + dt.timedelta(days+1)
    
    # 按交易日平移
    if days_type == 'T':
        all_trade_days = [i.strftime('%Y-%m-%d') for i in list(get_all_trade_days())]
        
        # 如果上一个自然日是交易日，根据其在交易日列表中的index计算平移后的交易日        
        if str(yesterday) in all_trade_days:
            shifted_date = all_trade_days[all_trade_days.index(str(yesterday)) + days + 1]
        # 否则，从上一个自然日向前数，先找到最近一个交易日，再开始平移
        else:
            for i in range(100):
                last_trade_date = yesterday - dt.timedelta(i)
                if str(last_trade_date) in all_trade_days:
                    shifted_date = all_trade_days[all_trade_days.index(str(last_trade_date)) + days + 1]
                    break
    return str(shifted_date)



# ================================================
# 股票过滤函数
# ================================================
def filter_new_stock(initial_list, date, days=50):
    d_date = transform_date(date, 'd')
    return [stock for stock in initial_list if d_date - get_security_info(stock).start_date > dt.timedelta(days=days)]

def filter_st_paused_stock(initial_list):
    current_data = get_current_data()
    # 使用列表推导式结合any()函数，筛选出符合条件的股票
    return [stock for stock in initial_list 
            if not any([
                current_data[stock].is_st,          # 排除ST股
                current_data[stock].paused,         # 排除停牌股
                '退' in current_data[stock].name    # 排除名称中含'退'字的股票，避免退市股
            ])]

def filter_kcbj_stock(initial_list):
    return [stock for stock in initial_list if stock[:2] in (('60','00','30'))]

def filter_st_stock(initial_list, date):
    str_date = transform_date(date, 'str')
    # 如果当前日期不是交易日，则使用前一个交易日
    if get_shifted_date(str_date, 0, 'N') != get_shifted_date(str_date, 0, 'T'):
        str_date = get_shifted_date(str_date, -1, 'T')
    # 获取股票的ST状态
    df = get_extras('is_st', initial_list, start_date=str_date, end_date=str_date, df=True)
    df = df.T
    df.columns = ['is_st']
    # 过滤掉ST股票
    df = df[df['is_st'] == False]
    filter_list = list(df.index)
    return filter_list

def filter_paused_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['paused'], count=1, panel=False, fill_paused=True)
    # 过滤掉停牌股票（paused=0表示未停牌）
    df = df[df['paused'] == 0]
    paused_list = list(df.code)
    return paused_list

def filter_extreme_limit_stock(context, stock_list, date):
    tmp = []
    for stock in stock_list:
        df = get_price(stock, end_date=date, frequency='daily', fields=['low','high_limit'], count=1, panel=False)
        # 如果最低价小于涨停价，说明不是一字涨停
        if df.iloc[0,0] < df.iloc[0,1]:
            tmp.append(stock)
    return tmp



def prepare_stock_list(date): 
    # 获取所有A股
    initial_list = get_all_securities('stock', date).index.tolist()
    # 过滤掉科创板
    initial_list = filter_kcbj_stock(initial_list)
    # 过滤掉新股
    initial_list = filter_new_stock(initial_list, date)
    # 过滤掉ST股和停牌股
    initial_list = filter_st_paused_stock(initial_list)
    return initial_list

def rise_low_volume(s, context):
    # 获取股票的历史高价和成交量数据
    hist = attribute_history(s, 106, '1d', fields=['high','volume'], skip_paused=True, df=False)
    high_prices = hist['high'][:102]
    prev_high = high_prices[-1]  # 最近一日的高价
    
    # 寻找前面最近一次高于当前高价的日期，计算中间的天数
    zyts_0 = next((i-1 for i, high in enumerate(high_prices[-3::-1], 2) if high >= prev_high), 100)
    zyts = zyts_0 + 5  # 增加5天作为观察期
    
    # 如果当前成交量小于观察期内最大成交量的90%，则认为存在左压
    if hist['volume'][-1] <= max(hist['volume'][-zyts:-1]) * 0.9:
        return True
    return False

def get_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close','high_limit'], count=1, panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    # 筛选收盘价等于涨停价的股票
    df = df[df['close'] == df['high_limit']]
    hl_list = list(df.code)
    return hl_list
    
def get_ever_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['high','high_limit'], count=1, panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    # 筛选最高价等于涨停价的股票
    df = df[df['high'] == df['high_limit']]
    hl_list = list(df.code)
    return hl_list

def get_ever_hl_stock2(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close','high','high_limit'], count=1, panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    cd1 = df['high'] == df['high_limit']  # 条件1：最高价等于涨停价（曾经涨停）
    cd2 = df['close'] != df['high_limit']  # 条件2：收盘价不等于涨停价（收盘未涨停）
    df = df[cd1 & cd2]  # 同时满足两个条件
    hl_list = list(df.code)
    return hl_list

# ================================================
# 涨停统计和指数涨幅计算函数
# ================================================
def get_hl_count_df(hl_list, date, watch_days):
    # 获取watch_days的数据
    df = get_price(hl_list, end_date=date, frequency='daily', fields=['close','high_limit','low'], count=watch_days, panel=False, fill_paused=False, skip_paused=False)
    df.index = df.code
    
    # 计算涨停与一字涨停数，一字涨停定义为最低价等于涨停价
    hl_count_list = []
    extreme_hl_count_list = []
    for stock in hl_list:
        df_sub = df.loc[stock]
        # 计算收盘涨停的天数
        hl_days = df_sub[df_sub.close==df_sub.high_limit].high_limit.count()
        # 计算一字涨停的天数（最低价等于涨停价）
        extreme_hl_days = df_sub[df_sub.low==df_sub.high_limit].high_limit.count()
        hl_count_list.append(hl_days)
        extreme_hl_count_list.append(extreme_hl_days)
    
    # 创建DataFrame记录结果
    df = pd.DataFrame(index=hl_list, data={'count':hl_count_list, 'extreme_count':extreme_hl_count_list})
    return df

def get_continue_count_df(hl_list, date, watch_days):
    df = pd.DataFrame()
    # 从2板开始，逐步检查到watch_days板
    for d in range(2, watch_days+1):
        # 获取d天内的涨停次数
        HLC = get_hl_count_df(hl_list, date, d)
        # 筛选出涨停次数等于d的股票（即连续d天涨停）
        CHLC = HLC[HLC['count'] == d]
        df = df.append(CHLC)
    
    # 处理可能的重复记录，保留最大连板数
    stock_list = list(set(df.index))
    ccd = pd.DataFrame()
    for s in stock_list:
        tmp = df.loc[[s]]
        if len(tmp) > 1:  # 如果一只股票有多条记录
            M = tmp['count'].max()  # 取最大连板数
            tmp = tmp[tmp['count'] == M]  # 只保留最大连板数的记录
        ccd = ccd.append(tmp)
    
    # 按连板数降序排列
    if len(ccd) != 0:
        ccd = ccd.sort_values(by='count', ascending=False)    
    return ccd

def get_index_increase_ratio(index_code, context):
    # 获取指数昨天和前天的收盘价
    close_prices = attribute_history(index_code, 2, '1d', fields=['close'], skip_paused=True)
    if len(close_prices) < 2:
        return 0  # 如果数据不足，返回0
    
    day_before_yesterday_close = close_prices['close'][0]  # 前天收盘价
    yesterday_close = close_prices['close'][1]  # 昨天收盘价
    
    # 计算涨幅
    increase_ratio = (yesterday_close - day_before_yesterday_close) / day_before_yesterday_close
    return increase_ratio

# ================================================
# 卖出函数
# ================================================
def sell_am(context):
    # 基础信息
    date = transform_date(context.previous_date, 'str')
    current_data = get_current_data()
    
    # 遍历所有持仓股票
    for s in list(context.portfolio.positions):
        # 条件：有可卖出的仓位 且 未涨停 且 有盈利
        if ((context.portfolio.positions[s].closeable_amount != 0) and 
            (current_data[s].last_price < current_data[s].high_limit) and 
            (current_data[s].last_price > 1*context.portfolio.positions[s].avg_cost)):
            
            order_target_value(s, 0)  # 清仓
            
                
def sell_pm(context):
    # 基础信息
    date = transform_date(context.previous_date, 'str')
    current_data = get_current_data()
    
    # 遍历所有持仓股票
    for s in list(context.portfolio.positions):
        # 计算5日均线
        close_data2 = attribute_history(s, 4, '1d', ['close'])
        M4 = close_data2['close'].mean()  # 前4日均价
        MA5 = (M4*4 + current_data[s].last_price)/5  # 5日均线
        
        # 条件1：止盈 - 有可卖出的仓位 且 未涨停 且 有盈利
        if ((context.portfolio.positions[s].closeable_amount != 0) and 
            (current_data[s].last_price < current_data[s].high_limit) and 
            (current_data[s].last_price > 1*context.portfolio.positions[s].avg_cost)):
            
            order_target_value(s, 0)  # 清仓
            
        
        # 条件2：止损 - 有可卖出的仓位 且 价格跌破5日均线
        elif ((context.portfolio.positions[s].closeable_amount != 0) and 
              (current_data[s].last_price < MA5)):
            
            order_target_value( s, 0)  # 清仓
            
                           
# 首版低开策略代码                
def filter_new_stock2(initial_list, date, days=250):
    d_date = transform_date(date, 'd')
    return [stock for stock in initial_list if d_date - get_security_info(stock).start_date > dt.timedelta(days=days)]
    
    
# 每日初始股票池
def prepare_stock_list2(date): 
    initial_list = get_all_securities('stock', date).index.tolist()
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_new_stock2(initial_list, date)
    initial_list = filter_st_stock(initial_list, date)
    initial_list = filter_paused_stock(initial_list, date)
    return initial_list    
    
# 计算股票处于一段时间内相对位置
def get_relative_position_df(stock_list, date, watch_days):
    if len(stock_list) != 0:
        df = get_price(stock_list, end_date=date, fields=['high', 'low', 'close'], count=watch_days, fill_paused=False, skip_paused=False, panel=False).dropna()
        close = df.groupby('code').apply(lambda df: df.iloc[-1,-1])
        high = df.groupby('code').apply(lambda df: df['high'].max())
        low = df.groupby('code').apply(lambda df: df['low'].min())
        result = pd.DataFrame()
        result['rp'] = (close-low) / (high-low)
        return result
    else:
        return pd.DataFrame(columns=['rp'])
        
# 计算股票最高位处于一段时间内相对位置        
def get_relative_position_df2(stock_list, date, watch_days):
    if len(stock_list) != 0:
        df = get_price(stock_list, end_date=date, fields=['high', 'low', 'close'], count=watch_days, fill_paused=False, skip_paused=False, panel=False).dropna()
        yestoday_high = df.groupby('code').apply(lambda df: df.iloc[-1,-3])
        high = df.groupby('code').apply(lambda df: df['high'].max())
        low = df.groupby('code').apply(lambda df: df['low'].min())
        result = pd.DataFrame()
        result['rp'] = (yestoday_high-low) / (high-low)
        return result
    else:
        return pd.DataFrame(columns=['rp'])
        
        
def sell_heavy_turnover(context):
    date = transform_date(context.previous_date, 'str')
    current_data = get_current_data()
    
    # 计算相对位置
    rpd = get_relative_position_df2(list(context.portfolio.positions), date, 60)
    rpd = rpd[rpd['rp'] >= 0.85]  # 筛选处于高位的股票(相对位置>=0.85)
    stock_list = list(rpd.index)
    
    for s in stock_list:
        df = get_price(s, end_date=date, fields=['high', 'low', 'close', 'open', 'volume', 'high_limit'], count=2, fill_paused=False, skip_paused=False, panel=False)
        pullback_ratio = (df['close'][-1] - df['high'][-1]) / df['high'][-1]  # 回调比例
        vol_ratio = df['volume'][-1] / df['volume'][-2]  # 成交量比例
        close_pos = (df['close'][-1] - min(df['high'][-1], df['low'][-1])) / abs(df['high'][-1] - df['low'][-1])  # 收盘位置
        
        # 如果处于高位，成交量放大，回调位置低于黄金分割点，当天在跌, 则卖出
        if ((context.portfolio.positions[s].closeable_amount != 0) and (vol_ratio >= 0.5) and (close_pos <= 0.65) and (current_data[s].last_price < df['close'][-1])):
            order_target_value(s, 0)  # 清仓
            


