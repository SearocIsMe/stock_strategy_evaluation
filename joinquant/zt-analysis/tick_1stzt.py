# 克隆自聚宽文章：https://www.joinquant.com/post/63340
# 标题：tick级别首板，非会员就不要clone了
# 作者：成都小宝总

from jqdata import *
from jqfactor import *
import json
import pandas as pd
import redis
import hashlib
import hmac
import requests
import base64
import time
from datetime import datetime


################################### 初始化设置 #############################################
def initialize(context):
    set_option('use_real_price', True)
    log.set_level('system', 'error')
    set_option("match_by_signal", True) # # 强制撮合，仅支持限价单。使用限价单进行委托时将不对委托价格和成交数量进行任何检查而直接成交
    g.stock_num = 2
    g.base_stock_num = 2  # 基础最大持仓数，根据市场环境动态调整
    g.push = False
    g.day_round = 0
    g.youxian=[]
    g.jianting=[]
    g.all_remove=[]
    g.sotck_data_hongpanlv=dict()
    g.sotck_data_yijialv=dict()
    g.max_hold_days = 5  # 最大持仓天数
    # 持仓信息字典：{stock_code: {'buy_date': date, 'zt_price': float}}
    # buy_date: 买入日期，用于计算持仓天数
    # zt_price: 买入当天的涨停价，作为止损线
    g.position_info = dict()
   
    log.info(g.push)
    set_option('avoid_future_data', True)

def handle_tick(context, tick):
    current_data = get_current_data()
    time_now = context.current_dt.strftime('%H:%M:%S')
    
    # === 实时止损：跌破买入当天涨停价且跌破10日均线则卖出 ===
    for s in list(context.portfolio.positions):
        pos = context.portfolio.positions[s]
        if pos.closeable_amount == 0:
            continue
        if s in g.position_info:
            zt_price = g.position_info[s]['zt_price']
            # 跌破涨停价，且跌破10日均线，才止损；未跌破MA10则继续持有
            if current_data[s].last_price < zt_price:
                ma10 = g.ma10_data.get(s, None)
                if ma10 is None or current_data[s].last_price < ma10:
                    order_target_value(s, 0)
                    profit_rate = current_data[s].last_price / pos.avg_cost - 1
                    log.info("跌破涨停价且跌破MA10止损卖出%s, 涨停价:%.2f, MA10:%.2f, 现价:%.2f, 盈亏:%.2f%%",
                             current_data[s].name, zt_price, ma10 or 0, current_data[s].last_price, profit_rate*100)
                    del g.position_info[s]
                    continue
    
    # 持仓已满则不再买入
    if (g.stock_num - len(context.portfolio.positions)) == 0:
        return
    # 交易窗口：09:30 - 11:00
    if time_now >= '11:00:00' or time_now < '09:30:00':
        return
    if tick.code in g.today_tick or tick.code in g.remove_list:
        return
    # 跌破昨收价（涨幅为负）则排除
    if tick.current < g.stock_data[tick.code] / 1.1:
        g.remove_list.append(tick.code)
        return
    # 买入条件：价格接近涨停价（距涨停价0.05元内）且不低于开盘价
    if tick.current >= g.stock_data[tick.code] - 0.05 and tick.current >= current_data[tick.code].day_open and tick.code not in list(context.portfolio.positions):
        value = context.portfolio.available_cash / (g.stock_num - len(context.portfolio.positions))
        if value > 100:
            order_value(tick.code, value)
            g.today_tick[tick.code] = 1
            # 记录买入信息：买入日期和当天涨停价
            g.position_info[tick.code] = {
                'buy_date': context.current_dt.date(),
                'zt_price': g.stock_data[tick.code]  # 买入当天的涨停价作为止损线
            }
            log.info("买入%s, 价格:%.2f, 涨停价止损线:%.2f", current_data[tick.code].name, tick.current, g.stock_data[tick.code])
    
        
def after_code_changed(context):
    g.push = False
    g.day_buy=0
    
    unschedule_all() # 取消所有定时运行

    run_daily(prepare,  time='09:27')
    g.target_list = []
    g.remove_list = []
    g.init_pre=1

    # 每日14:50检查持仓：5日到期卖出 或 跌破涨停价止损
    run_daily(sell, time='14:50', reference_security='000300.XSHG')

    
def saixuan(context):
    now = context.current_dt
    current_data = get_current_data()

    for s in g.yizhi:
        df_panel_all = get_price(
                        s,
                        count=50,
                        end_date=now,
                        frequency='minute',
                        fields=['open','high','low','close','high_limit','money','volume']
                    )
        if current_data[s].day_open*1.06<df_panel_all['close'][-1]<current_data[s].high_limit:
            if df_panel_all['high'].max()==current_data[s].high_limit:
                continue
            subscribe(s, 'tick')
            g.stock_data[s]=current_data[s].high_limit
            log.info("监听%s",current_data[s].name)
            
def prepare(context):

    unsubscribe_all()
    g.today_tick = dict()
    g.today_tick_zhaban = dict()
    g.youxian=[]
    g.sotck_data=dict()
    g.sotck_data_open=dict()
    g.day_buy=0
    g.buy_complited = False
    g.today = 0
    g.stock_data=dict()
    g.remove_list=[]
    g.jianting=[]
    
    # === 预计算持仓股的10日均线（用于止损判断）===
    g.ma10_data = dict()
    held_stocks = list(context.portfolio.positions.keys())
    if held_stocks:
        try:
            for s in held_stocks:
                if s in g.position_info:
                    ma10_df = get_price(s, end_date=context.previous_date, frequency='daily',
                                        fields=['close'], count=10, skip_paused=True)
                    if len(ma10_df) >= 10:
                        g.ma10_data[s] = ma10_df['close'].mean()
        except:
            pass
    
    # === 市场环境过滤：指数在5日均线之下时减仓操作 ===
    g.market_weak = False
    index_code = '000300.XSHG'  # 沪深300作为市场基准
    index_data = get_price(index_code, end_date=context.previous_date, frequency='daily',
                           fields=['close'], count=5, skip_paused=True)
    if len(index_data) >= 5:
        ma5 = index_data['close'].mean()
        current_close = index_data['close'].iloc[-1]
        if current_close < ma5:
            g.market_weak = True
            g.stock_num = max(1, g.base_stock_num - 1)  # 弱市减少持仓
            log.info("市场弱势（沪深300收于%.2f < MA5 %.2f），最大持仓数降为%d", current_close, ma5, g.stock_num)
        else:
            g.stock_num = g.base_stock_num
    else:
        g.stock_num = g.base_stock_num
    
    #获取今日股票
    g.yizhi = prepare_stock_list(context)
    current_data = get_current_data()
    for s in g.yizhi:
        subscribe(s, 'tick')
        g.stock_data[s]=current_data[s].high_limit
        log.info("监听%s",current_data[s].name)
     
    log.info("今日监控股票列表%s",g.yizhi)

# 每日初始股票池
def prepare_stock_list8(context):
    today = context.current_dt.date()
    yesterday = context.previous_date
    initial_list = set_stockpool(context)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_paused_stock(initial_list, today)
    #initial_list = filter_new_stock(initial_list, today)
    #initial_list=get_hl_stock(initial_list,yesterday,10)
    ZHANGTING1=get_hl_stock(initial_list,yesterday,1)
    #ZHANGTING2=get_hl_stock(initial_list,yesterday-timedelta(days=1),1)
    #intersection = list(set(ZHANGTING1) & set(ZHANGTING2))
    res=[]
    cur = get_current_data()
    for s in ZHANGTING1:
        if cur[s].day_open<= cur[s].high_limit/1.1*1.08  and cur[s].day_open!= cur[s].high_limit:
             res.append(s)
    return res

    
## 定义股票池
def set_stockpool(context):
    yesterday = context.previous_date
    initial_list = get_all_securities('stock', yesterday).index.tolist()
    return initial_list

def gen_sign(secret):# 拼接时间戳以及签名校验
    timestamp = int(time.time())

    string_to_sign = '{}\n{}'.format(timestamp, secret)
    # 使用 HMAC-SHA256 进行加密
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    # 对结果进行 base64 编码
    sign = base64.b64encode(hmac_code).decode('utf-8')
    return sign
##################################  交易函数群 ##################################
def buy(context):
    log.info("buy")
    #今日完毕
    if g.buy_complited or g.today ==5:
        return
    current_data = get_current_data()
    time_now = context.current_dt.strftime('%H:%M:%S')
    if time_now>='14:30:00' or time_now<'09:32:00':
        return
    now = context.current_dt
    zeroToday = now - datetime.timedelta(hours=now.hour, minutes=now.minute, seconds=now.second,microseconds=now.microsecond)
    lastToday = zeroToday + datetime.timedelta(hours=9, minutes=30, seconds=00)
    endToday = zeroToday + datetime.timedelta(hours=9, minutes=32, seconds=00)
    #昨日涨停护具
    qualified_stocks = g.youxian+g.jianting
    log.info("股票池%s",qualified_stocks)
    for stock in qualified_stocks:
        if stock in g.remove_list or stock in g.all_remove:
            continue
        if current_data[stock].day_open==current_data[stock].high_limit:
            g.remove_list.append(stock)
            continue
        
        if get_current_data()[stock].last_price<get_current_data()[stock].high_limit/1.1:
            log.info(stock + "跌3个点，remove")
            g.remove_list.append(stock)
            continue
        if time_now<'14:30:00':
            time_difference = now - lastToday
            minutes_since_target = int(time_difference.total_seconds() / 60)
            df_panel_all = get_price(
                        stock,
                        start_date=lastToday,
                        end_date=now,
                        frequency='minute',
                        fields=['open','high','low','close','high_limit','money','volume']
                    )
            zhangfu_ratio = df_panel_all['close'][-1]/(df_panel_all['high_limit'][-1]/1.1)
            if zhangfu_ratio > 1.6 and stock not in g.youxian:
               
                g.youxian.append(stock)
            #and g.sotck_data_hongpanlv[stock] > 0.4 and g.sotck_data_yijialv[stock]>0.5  
            if zhangfu_ratio >= 1.09 and stock not in list(context.portfolio.positions):
                cur_count = (g.stock_num-len(context.portfolio.positions))
                if cur_count ==0:
                    g.buy_complited = True
                    return
                value = context.portfolio.available_cash / (g.stock_num-len(context.portfolio.positions))
                print(value)
                if value/current_data[stock].last_price>100:
                    order_value(stock, value)
                    print('买入' + stock+'->'+current_data[stock].name)
           
def jianyi(context,stock):
    turnover_ratio_data=get_valuation(stock, start_date=context.previous_date, end_date=context.previous_date, fields=['turnover_ratio', 'market_cap','circulating_market_cap'])
    shizhi=turnover_ratio_data['market_cap'][0]
    if shizhi>=50 and shizhi < 150:
        return "->注意:权重高，可以多买点"
    return ""
def sell(context):
    """每日14:50检查持仓，执行5日持有期规则：
    1. 持仓超过5天 → 卖出（无论盈亏）
    2. 跌破买入当天涨停价且跌破10日均线 → 止损卖出
    3. 涨停股继续持有（不卖）
    """
    current_data = get_current_data()
    today = context.current_dt.date()
    
    for s in list(context.portfolio.positions):
        pos = context.portfolio.positions[s]
        if pos.closeable_amount == 0:
            continue
        
        # 涨停股继续持有
        if current_data[s].last_price >= current_data[s].high_limit:
            continue
        
        # 检查持仓信息
        if s not in g.position_info:
            # 没有持仓记录（可能是之前买入的），补录并卖出
            order_target_value(s, 0)
            log.info("无持仓记录，卖出%s", current_data[s].name)
            continue
        
        buy_date = g.position_info[s]['buy_date']
        zt_price = g.position_info[s]['zt_price']
        hold_days = (today - buy_date).days
        profit_rate = current_data[s].last_price / pos.avg_cost - 1
        
        # 持仓超过5天，卖出
        if hold_days >= g.max_hold_days:
            order_target_value(s, 0)
            log.info("持有%d天到期卖出%s, 盈亏:%.2f%%", hold_days, current_data[s].name, profit_rate*100)
            del g.position_info[s]
            continue
        
        # 跌破买入当天涨停价，且跌破10日均线，才止损；未跌破MA10则继续持有
        if current_data[s].last_price < zt_price:
            ma10 = g.ma10_data.get(s, None)
            if ma10 is None or current_data[s].last_price < ma10:
                order_target_value(s, 0)
                log.info("跌破涨停价且跌破MA10止损卖出%s, 涨停价:%.2f, MA10:%.2f, 现价:%.2f, 盈亏:%.2f%%",
                         current_data[s].name, zt_price, ma10 or 0, current_data[s].last_price, profit_rate*100)
                del g.position_info[s]
                continue
                
   
def calculate_lb(stocks,end_date,count):
    ss_info=get_price(stocks, end_date=end_date, frequency='daily', fields=['close','high_limit'], count=count, panel=False, fill_paused=False, skip_paused=False)
    ss_info['zt']=np.where((ss_info['high_limit']==ss_info['close']),1,0)
    zt_stocks=ss_info[ss_info['zt']==1].code.tolist()
    ss_info=ss_info[ss_info['code'].isin(zt_stocks)]

    def func(group):
        group['lb']=0
        lb_counter=0
        for i in group.index:
            if group.loc[i, 'zt'] == 1:
                lb_counter += 1
                group.loc[i, 'lb'] = lb_counter
            else:
                lb_counter = 0
                group.loc[i, 'lb'] = lb_counter
        return group
    
    ss_info=ss_info.groupby(by='code').apply(func)
    ss_info=ss_info[ss_info.lb>0]
    df =  ss_info[['time','code','lb']]
    return df
    
def nextday(time):
    dates= get_all_trade_days()
# 转换为 datetime 对象
    dates = np.array([np.datetime64(date) for date in dates])
# 指定日期
    specified_date = np.datetime64(time[:10])
# 查找下一个日期
    next_date = dates[dates > specified_date][0]
    return str(next_date)    
def hongpanlv(context,df2_code,df):
    g.jianting=[]
    for s in df2_code:
        if s in g.sotck_data_hongpanlv:
            continue
        #打2版后第二天的红盘概==率
        df2=df[(df['code']==s) ]['time']
        if len(df2)<10:
            continue
        if df2[:-1].empty:
            g.sotck_data[s]='涨停基因较差'
            continue
        #print("========")
       # print(s)
        #print(df[(df['code']==s) ])
        count_high = 0
        count_all=0
        lianban=0
        yijia=0
        #print(df2)
        for times in df2[:-1]:
            count_all+=1
            #print(times)
            ss_info=get_price(s, end_date=nextday(str(times)), frequency='daily', fields=['close','high_limit','high','pre_close'], count=1, panel=False, fill_paused=False, skip_paused=False)
                #print(ss_info)
            if ss_info['close'][-1] > ss_info['pre_close'][-1]:
                count_high+=1
            if ss_info['high_limit'][-1]==ss_info['close'][-1]:
                lianban+=1
            if ss_info['close'][-1]/ss_info['pre_close'][-1]>1.05:
                yijia+=1
      
        ratio = float(count_high) / float(count_all)
        lianbanlv = float(lianban) / float(count_all)
        yijia5 = float(yijia) / float(count_all)
        g.sotck_data_hongpanlv[s]=round(ratio,2)
        g.sotck_data_yijialv[s]=round(yijia5,2)
        if round(ratio,2)<0.5 or round(yijia5,2)<0.5:
            g.all_remove.append(s)
        else:
            g.jianting.append(s)
        '''
        print('次日红盘率'+str(round(ratio,2)))
    print('连板率'+str(round(lianbanlv,2)))
    print('溢价5%的概率'+str(round(yijia5,2)))
        '''
        g.sotck_data[s]='最近200个交易日【\n涨停次数'+str(count_all)+'\n次日红盘5%次数'+str(yijia)+'\n次日红盘率'+str(round(ratio,2)*100)+'%\n'+'连板率'+str(round(lianbanlv,2)*100)+'%\n'+'次日溢价5%的概率'+str(round(yijia5,2)*100)+'%】'
#        log.info(sotck_data)

 # 每日初始股票池
def prepare_stock_list_not_zhangting(context):
    
    log.info("初始化连板数据")
    today = context.current_dt.date()
    yesterday = context.previous_date
    initial_list = set_stockpool(context)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_paused_stock(initial_list, today)
    initial_list = filter_new_stock(initial_list, today)
    initial_list=get_hl_stock(initial_list,yesterday,1)
    cur = get_current_data()
    df2_code=[]
    initial_list = get_hl_not_stock(initial_list, yesterday,1)
    for s in initial_list:
        if cur[s].day_open==cur[s].high_limit:
            df2_code.append(s)
    g.jianting = df2_code
    return df2_code

# 每日初始股票池
def prepare_stock_list(context):
    today = context.current_dt.date()
    yesterday = context.previous_date
    # 获取初始股票池
    initial_list = set_stockpool(context)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_paused_stock(initial_list, today)
    initial_list = filter_new_stock(initial_list, today)
    # 筛选昨日涨停股（close == high_limit）
    initial_list = get_hl_stock(initial_list, yesterday, 1)
    
    # === 核心过滤：只保留首板股，排除连板股 ===
    # 策略逻辑：昨日首板 → 今日一进二 → 明日三板卖
    # 如果前天也涨停，说明昨日是连板（二板及以上），不是首板，必须排除
    day_before_yesterday = get_trade_days(end_date=yesterday, count=2)[0]  # 前一个交易日
    lianban_list = get_hl_stock(initial_list, day_before_yesterday, 1)  # 前天也涨停的=连板股
    initial_list = [s for s in initial_list if s not in lianban_list]  # 只保留首板股
    log.info("昨日首板股数量:%d, 排除连板股数量:%d", len(initial_list), len(lianban_list))
    
    # === 排除昨日一字板股（开盘即涨停，次日追买风险大）===
    if initial_list:
        yiziban_df = get_price(initial_list, end_date=yesterday, frequency='daily',
                               fields=['open', 'high_limit'], count=1, panel=False, fill_paused=False, skip_paused=True)
        yiziban_list = yiziban_df[yiziban_df['open'] == yiziban_df['high_limit']]['code'].tolist()
        initial_list = [s for s in initial_list if s not in yiziban_list]
        log.info("排除昨日一字板股数量:%d, 剩余首板股数量:%d", len(yiziban_list), len(initial_list))
    
    # === 排除5日内涨幅超过23%的股票（短期涨幅过大，追高风险大）===
    if initial_list:
        gain_df = get_price(initial_list, end_date=yesterday, frequency='daily',
                            fields=['close'], count=5, panel=False, fill_paused=False, skip_paused=True)
        # 计算每只股票5日涨幅
        gain_list = []
        for s in initial_list:
            stock_df = gain_df[gain_df['code'] == s]
            if len(stock_df) >= 2:
                gain_rate = (stock_df['close'].iloc[-1] / stock_df['close'].iloc[0]) - 1
                if gain_rate > 0.23:
                    gain_list.append(s)
        initial_list = [s for s in initial_list if s not in gain_list]
        log.info("排除5日涨幅超23%%股数量:%d, 剩余首板股数量:%d", len(gain_list), len(initial_list))
    
    # 从首板股中，筛选今日高开但未一字涨停的股票
    # 条件1：开盘价 > 昨收*1.05（高开5%以上）
    # 条件2：开盘价 < 涨停价（未一字板）
    # 条件3：开盘价 < 涨停价*0.98（排除开盘价太接近涨停价的，追高风险大）
    # 条件4：昨日换手率 > 2%（排除流动性差的股票）
    cur = get_current_data()
    hl_list = []
    for s in initial_list:
        pre_close = cur[s].high_limit / 1.1  # 昨收价
        day_open = cur[s].day_open
        # 高开5%以上且未一字涨停
        if day_open > pre_close * 1.05 and day_open < cur[s].high_limit:
            # 排除开盘价太接近涨停价的（距涨停不到2%，追高风险大）
            if day_open > cur[s].high_limit * 0.98:
                continue
            hl_list.append(s)
    # 换手率过滤：排除昨日换手率过低的股票
    if hl_list:
        try:
            turnover_data = get_valuation(hl_list, end_date=yesterday, start_date=yesterday,
                                          fields=['turnover_ratio'])
            low_turnover = turnover_data[turnover_data['turnover_ratio'] < 2.0].index.tolist()
            hl_list = [s for s in hl_list if s not in low_turnover]
        except:
            pass  # 如果获取换手率失败，不过滤
    return hl_list

def get_hl_stock_dangtian(stock_list,d):
    if not stock_list:return []
    h_s = get_price(stock_list, frequency='1d',end_date=d, fields=['close', 'high', 'pre_close'],
                  count=1, panel=False, fill_paused=False, skip_paused=True
                  ).query('high>=pre_close*1.06').groupby('code').size()
    return h_s.index.tolist()
###################################  其它函数群 ##################################
def get_low_price(context,stock):
    h_s = get_price(stock, end_date=context.current_dt, frequency='daily', fields=['pre_close', 'open', 'paused'],
                    count=1, panel=False, fill_paused=False, skip_paused=True
                    )
    if h_s['pre_close'][-1]*0.98 > h_s['open'][-1]:
        return True

# 昨日飞低收或者涨停的不要
def get_hl_not_stock(stock_list, date1,days):
    if not stock_list:return []
    h_s = get_price(stock_list, end_date=date1, frequency='daily', fields=['open','high','low_limit','low','close', 'high_limit'],
                  count=days, panel=False, fill_paused=False, skip_paused=True
                  ).query('high!=high_limit').groupby('code').size()
    return h_s.index.tolist()
# 筛选出某一日涨停的股票
def get_hl_stock(stock_list, date1,days):
    if not stock_list:return []
    h_s = get_price(stock_list, end_date=date1, frequency='daily', fields=['close', 'high_limit'],
                  count=days, panel=False, fill_paused=False, skip_paused=True
                  ).query('close==high_limit').groupby('code').size()
    return h_s.index.tolist()

# 过滤函数
def filter_new_stock(initial_list, date, days=50):
    return [stock for stock in initial_list if get_security_info(stock).start_date < date - timedelta(days=days)]

def filter_st_paused_stock(initial_list, date):
    current_data = get_current_data()
    return [stock for stock in initial_list if not (
            current_data[stock].is_st or
            current_data[stock].paused or
            '退' in current_data[stock].name)]

    
def gen_sign(secret):# 拼接时间戳以及签名校验
    timestamp = int(time.time())

    string_to_sign = '{}\n{}'.format(timestamp, secret)
    # 使用 HMAC-SHA256 进行加密
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    # 对结果进行 base64 编码
    sign = base64.b64encode(hmac_code).decode('utf-8')
    return sign
    

def filter_kcbj_stock(initial_list):
    # 排除北交所（4/8开头）和科创板（68开头），保留创业板（3开头）
    return [stock for stock in initial_list if stock[0] != '4' and stock[0] != '8' and stock[:2] != '68']
