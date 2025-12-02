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

import newqmt_sql

# ⭐ 在这里设置这个策略的分类标签（写入 trade.fenlei）
newqmt_sql.FENLEI = 'tick-origin'      # 或 '中长线趋势策略B' 等

from newqmt_sql import (
    order_zzy as order,
    order_target_zzy as order_target,
    order_value_zzy as order_value,
    order_target_value_zzy as order_target_value
)

################################### 初始化设置 #############################################
def initialize(context):
    set_option('use_real_price', True)
    log.set_level('system', 'error')
    set_option("match_by_signal", True) # # 强制撮合，仅支持限价单。使用限价单进行委托时将不对委托价格和成交数量进行任何检查而直接成交
    g.stock_num = 2
    g.push = False
    g.day_round = 0
    g.youxian=[]
    g.jianting=[]
    g.all_remove=[]
    g.sotck_data_hongpanlv=dict()
    g.sotck_data_yijialv=dict()
    
    # Redis连接配置（使用您提供的凭证）
   
    log.info(g.push)
    #set_option('avoid_future_data', True)
    g.WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/15c8e896-5cb4-40a4-ba69-b2a36d4d4cdf"
    g.WEBHOOK_SECRET = "IG3VNZ51o81qYbiTZqZX5N9f"
# 创建Redis连接
def create_redis_connection(REDIS_CONFIG):
    return redis.Redis(**REDIS_CONFIG)

def handle_tick(context, tick):
    current_data = get_current_data()
    if  (g.stock_num-len(context.portfolio.positions))==0:
        return
    time_now = context.current_dt.strftime('%H:%M:%S')
    if time_now>='10:30:00' or time_now<'09:30:00':
        return
    if tick.code in g.today_tick or tick.code in g.remove_list:
        return
    if tick.current <g.stock_data[tick.code]/1.1:
        g.remove_list.append(tick.code)
    if  tick.current>= g.stock_data[tick.code]-0.05 and  tick.current>= current_data[tick.code].day_open and tick.code not in list(context.portfolio.positions) :
        value = context.portfolio.available_cash / (g.stock_num-len(context.portfolio.positions))
        if value > 100:
            order_value(tick.code,value)
            #publisher(tick.code,g.stock_data[tick.code],100)
            g.today_tick[tick.code] = 1
            print("买入"+get_current_data()[tick.code].name)
    
''' ====================== 发布者代码 ====================== '''
def publisher(c,p,a):
    pub = g.redis_client
    CHANNEL_NAME="trading_data"
    
    print("[发布者] 已连接到Redis，开始发送数据...")
    
    # 模拟发布3条交易数据
    data = {
            'code': c,
            'price': p,
            'amount': a
        }
        
        # 发布JSON格式的消息
    pub.publish(CHANNEL_NAME, json.dumps(data))
    print("[发布者] 数据发送完成")      
        
def after_code_changed(context):
    g.push = False
    g.day_buy=0
    
    unschedule_all() # 取消所有定时运行
    # run_daily(get_stock_list, '9:05')
   # run_daily(buy,  time='every_bar')
    #昨日涨停
   #run_daily(saixuan,  time='10:20')
   # run_daily(saixuan,  time='09:50')

    run_daily(prepare,  time='09:27')
    g.target_list = []
    g.remove_list = []
    g.init_pre=1
    #g.WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/14b7908e-c04e-4fec-b4e9-78f587cbcc5a"
   # g.WEBHOOK_SECRET = "0MJBTfGQkP4exShpVpn7af"
    run_daily(sell, time='11:25', reference_security='000300.XSHG')
    run_daily(sell, time='14:50', reference_security='000300.XSHG')
    #initialize(context)
def saixuan(context):
    now = context.current_dt
    current_data = get_current_data()
    #zeroToday = now - datetime.timedelta(hours=now.hour, minutes=now.minute, seconds=now.second,microseconds=now.microsecond)
   # lastToday = zeroToday + datetime.timedelta(hours=9, minutes=30, seconds=00)
   # endToday = zeroToday + datetime.timedelta(hours=9, minutes=32, seconds=00)
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
    REDIS_CONFIG = {
        'host': 'redis-11679.c292.ap-southeast-1-1.ec2.redns.redis-cloud.com',
        'port': 11679,
        'decode_responses': True,
        #'username': 'default',
        'password': 'ObL0E7RbFHLgG9MjyeXVaBPZYrzavgj5'
    }
   # g.redis_client=create_redis_connection(REDIS_CONFIG)
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
    #排除昨日涨停
    g.remove_list=[]
    g.jianting=['000008.XSHE', '000066.XSHE', '000099.XSHE', '000158.XSHE', '000566.XSHE', '000595.XSHE', '000605.XSHE', '000608.XSHE', '000702.XSHE', '000712.XSHE', '000716.XSHE', '000717.XSHE', '000795.XSHE', '000801.XSHE', '000810.XSHE', '000833.XSHE', '000859.XSHE', '000880.XSHE', '000903.XSHE', '000917.XSHE', '000953.XSHE', '000981.XSHE', '001209.XSHE', '001229.XSHE', '001298.XSHE', '001379.XSHE', '001696.XSHE', '002085.XSHE', '002095.XSHE', '002103.XSHE', '002131.XSHE', '002134.XSHE', '002146.XSHE', '002178.XSHE', '002181.XSHE', '002199.XSHE', '002232.XSHE', '002277.XSHE', '002285.XSHE', '002298.XSHE', '002305.XSHE', '002347.XSHE', '002348.XSHE', '002403.XSHE', '002423.XSHE', '002455.XSHE', '002514.XSHE', '002526.XSHE', '002583.XSHE', '002593.XSHE', '002611.XSHE', '002628.XSHE', '002640.XSHE', '002654.XSHE', '002670.XSHE', '002725.XSHE', '002769.XSHE', '002820.XSHE', '002823.XSHE', '002857.XSHE', '002862.XSHE', '002869.XSHE', '003026.XSHE', '600171.XSHG', '600198.XSHG', '600207.XSHG', '600243.XSHG', '600292.XSHG', '600410.XSHG', '600439.XSHG', '600463.XSHG', '600501.XSHG', '600593.XSHG', '600619.XSHG', '600622.XSHG', '600624.XSHG', '600635.XSHG', '600650.XSHG', '600653.XSHG', '600676.XSHG', '600678.XSHG', '600793.XSHG', '600811.XSHG', '600817.XSHG', '600839.XSHG', '600841.XSHG', '600889.XSHG', '600979.XSHG', '600990.XSHG', '601727.XSHG', '601933.XSHG', '603006.XSHG', '603021.XSHG', '603038.XSHG', '603106.XSHG', '603278.XSHG', '603499.XSHG', '603533.XSHG', '603580.XSHG', '603656.XSHG', '603657.XSHG', '603662.XSHG', '603666.XSHG', '603679.XSHG', '603716.XSHG', '603739.XSHG', '603803.XSHG', '603859.XSHG', '603988.XSHG', '605180.XSHG', '605198.XSHG', '000056.XSHE', '000062.XSHE', '000066.XSHE', '000533.XSHE', '000536.XSHE', '000566.XSHE', '000573.XSHE', '000605.XSHE', '000620.XSHE', '000665.XSHE', '000677.XSHE', '000681.XSHE', '000702.XSHE', '000712.XSHE', '000716.XSHE', '000717.XSHE', '000759.XSHE', '000785.XSHE', '000795.XSHE', '000801.XSHE', '000810.XSHE', '000818.XSHE', '000833.XSHE', '000856.XSHE', '000859.XSHE', '000880.XSHE', '000903.XSHE', '000958.XSHE', '000965.XSHE', '000981.XSHE', '001209.XSHE', '001379.XSHE', '001696.XSHE', '002036.XSHE', '002065.XSHE', '002094.XSHE', '002095.XSHE', '002103.XSHE', '002122.XSHE', '002123.XSHE', '002131.XSHE', '002146.XSHE', '002164.XSHE', '002175.XSHE', '002178.XSHE', '002181.XSHE', '002184.XSHE', '002208.XSHE', '002232.XSHE', '002265.XSHE', '002276.XSHE', '002277.XSHE', '002278.XSHE', '002285.XSHE', '002290.XSHE', '002347.XSHE', '002369.XSHE', '002403.XSHE', '002423.XSHE', '002514.XSHE', '002526.XSHE', '002527.XSHE', '002570.XSHE', '002580.XSHE', '002583.XSHE', '002593.XSHE', '002611.XSHE', '002628.XSHE', '002633.XSHE', '002640.XSHE', '002654.XSHE', '002670.XSHE', '002681.XSHE', '002691.XSHE', '002725.XSHE', '002741.XSHE', '002767.XSHE', '002820.XSHE', '002823.XSHE', '002851.XSHE', '002862.XSHE', '002869.XSHE', '002881.XSHE', '002912.XSHE', '003026.XSHE', '600120.XSHG', '600126.XSHG', '600171.XSHG', '600172.XSHG', '600198.XSHG', '600203.XSHG', '600292.XSHG', '600327.XSHG', '600386.XSHG', '600410.XSHG', '600439.XSHG', '600463.XSHG', '600481.XSHG', '600501.XSHG', '600539.XSHG', '600579.XSHG', '600589.XSHG', '600592.XSHG', '600593.XSHG', '600602.XSHG', '600619.XSHG', '600622.XSHG', '600624.XSHG', '600629.XSHG', '600635.XSHG', '600650.XSHG', '600653.XSHG', '600676.XSHG', '600678.XSHG', '600679.XSHG', '600693.XSHG', '600714.XSHG', '600719.XSHG', '600743.XSHG', '600793.XSHG', '600800.XSHG', '600817.XSHG', '600824.XSHG', '600825.XSHG', '600839.XSHG', '600865.XSHG', '600881.XSHG', '600889.XSHG', '600936.XSHG', '600979.XSHG', '601086.XSHG', '601162.XSHG', '601177.XSHG', '601727.XSHG', '601933.XSHG', '603004.XSHG', '603038.XSHG', '603039.XSHG', '603063.XSHG', '603086.XSHG', '603106.XSHG', '603110.XSHG', '603278.XSHG', '603300.XSHG', '603366.XSHG', '603583.XSHG', '603586.XSHG', '603626.XSHG', '603662.XSHG', '603667.XSHG', '603677.XSHG', '603739.XSHG', '603776.XSHG', '603777.XSHG', '603803.XSHG', '603881.XSHG', '603883.XSHG', '603928.XSHG', '603949.XSHG', '603955.XSHG', '605033.XSHG', '605069.XSHG', '605100.XSHG', '605179.XSHG', '605258.XSHG', '605398.XSHG', '605488.XSHG']
    #获取今日股票
    g.yizhi = prepare_stock_list(context)
    current_data = get_current_data()
    for s in g.yizhi:
        subscribe(s, 'tick')
        g.stock_data[s]=current_data[s].high_limit
        log.info("监听%s",current_data[s].name)
    #prepare_stock_list_not_zhangting(context)
   # g.jianting=get_hl_not_stock(g.jianting,context.previous_date,1)
    
    #g.target_list =prepare_stock_list(context)
   # log.info("今日监控股票列表%s",g.yizhi)

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
                    feishu(stock,"买入")
                    print('买入' + stock+'->'+current_data[stock].name)
           
def jianyi(context,stock):
    turnover_ratio_data=get_valuation(stock, start_date=context.previous_date, end_date=context.previous_date, fields=['turnover_ratio', 'market_cap','circulating_market_cap'])
    shizhi=turnover_ratio_data['market_cap'][0]
    if shizhi>=50 and shizhi < 150:
        return "->注意:权重高，可以多买点"
    return ""
def sell(context):
    stime = context.current_dt.strftime("%H%M")
    current_data = get_current_data()

    # 根据时间执行不同的卖出策略
    if stime == '1125':
        for s in list(context.portfolio.positions):  #上午有利润就跑
            if ((context.portfolio.positions[s].closeable_amount != 0) and (current_data[s].last_price < current_data[s].high_limit) and (current_data[s].last_price > 1*context.portfolio.positions[s].avg_cost)):#avg_cost当前持仓成本
                order_target_value(s, 0)
                 
    elif stime == '1450':
        for s in list(context.portfolio.positions):
            if ((context.portfolio.positions[s].closeable_amount != 0) and (current_data[s].last_price < current_data[s].high_limit)):#closeable_amount可卖出的仓位
                order_target_value(s, 0)
                
   
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
    #initial_list = get_hl_not_stock(initial_list, yesterday,1)
    for s in initial_list:
        if cur[s].day_open==cur[s].high_limit:
            df2_code.append(s)
    #g.jianting=df2_code
    return df2_code

# 每日初始股票池
def prepare_stock_list(context):
    today = context.current_dt.date()
    yesterday = context.previous_date
    initial_list = set_stockpool(context)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_paused_stock(initial_list, today)
    initial_list = filter_new_stock(initial_list, today)
    initial_list=get_hl_stock(initial_list,yesterday,1)
    cur = get_current_data()
    hl_list=[]
   # hl_list = get_hl_stock_dangtian(initial_list,today)     # 昨日涨停
    for s in initial_list:
        if cur[s].day_open > cur[s].high_limit/1.1*1.08 and cur[s].day_open<cur[s].high_limit:
            hl_list.append(s)
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
    
def feishu_msg(stock,buy,msg):
    if g.push==False:
        return
    current_data = get_current_data()
    params = {
        "timestamp": int(time.time()),
        "sign": gen_sign(g.WEBHOOK_SECRET),
        "msg_type": "text",
        "content": {"text": msg+buy+'->' + stock+'->'+current_data[stock].name},
    }
    resp = requests.post(g.WEBHOOK_URL, json=params)

def feishu(stock,buy):
    if g.push==False:
        return
    current_data = get_current_data()
    params = {
        "timestamp": int(time.time()),
        "sign": gen_sign(g.WEBHOOK_SECRET),
        "msg_type": "text",
        "content": {"text": '【首板】'+buy+'->' + stock+'->'+current_data[stock].name},
    }
    resp = requests.post(g.WEBHOOK_URL, json=params)
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
    return [stock for stock in initial_list if stock[0] != '4'  and stock[0] != '8' and stock[:2] != '68' and stock[0] != '3']  #and stock[0] != '3'
