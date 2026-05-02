from jqdata import *
from jqfactor import *
import json
import pandas as pd
import numpy as np
import redis
import hashlib
import hmac
import requests
import base64
import time
from datetime import datetime, timedelta
from jqlib.technical_analysis import *

import newqmt_sql

# ⭐ 在这里设置这个策略的分类标签（写入 trade.fenlei）
newqmt_sql.FENLEI = 'test-tick'      # 或 '中长线趋势策略B' 等

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
    set_option("match_by_signal", True) # 强制撮合，仅支持限价单。使用限价单进行委托时将不对委托价格和成交数量进行任何检查而直接成交
    g.stock_num = 2
    g.push = False
    g.day_round = 0
    g.youxian=[]
    g.jianting=[]
    g.all_remove=[]
    g.sotck_data_hongpanlv=dict()
    g.sotck_data_yijialv=dict()
    
    # Redis连接配置（使用您提供的凭证）
    g.REDIS_CONFIG = {
        'host': 'redis-11679.c292.ap-southeast-1-1.ec2.redns.redis-cloud.com',
        'port': 11679,
        'decode_responses': True,
        'password': 'ObL0E7RbFHLgG9MjyeXVaBPZYrzavgj5'
    }
    
    # 初始化变量
    g.redis_client = None
    g.today_tick = dict()
    g.today_tick_zhaban = dict()
    g.sotck_data = dict()
    g.sotck_data_open = dict()
    g.day_buy = 0
    g.buy_complited = False
    g.today = 0
    g.stock_data = dict()
    g.remove_list = []
    g.yizhi = []
    
    log.info(g.push)
    g.WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/15c8e896-5cb4-40a4-ba69-b2a36d4d4cdf"
    g.WEBHOOK_SECRET = "IG3VNZ51o81qYbiTZqZX5N9f"
    
    # 设置定时函数
    schedule_function(prepare, time_rule=market_open(minute=1))
    schedule_function(sell, time_rule=market_close(minute=-5))
    schedule_function(sell, time_rule=market_close(minute=-95))
    
# 创建Redis连接
def create_redis_connection(REDIS_CONFIG):
    try:
        return redis.Redis(**REDIS_CONFIG)
    except:
        return None
    
################################### 主要函数群 ##################################
def prepare(context):
    """每日准备函数"""
    current_date = context.current_dt.date()
    
    # 重置每日变量
    g.today_tick = dict()
    g.today_tick_zhaban = dict()
    g.youxian = []
    g.sotck_data = dict()
    g.sotck_data_open = dict()
    g.day_buy = 0
    g.buy_complited = False
    g.today = 0
    g.stock_data = dict()
    g.remove_list = []
    
    # 不需要调用 unsubscribe_all()，使用新的订阅管理方式
    # 获取今日股票池
    g.yizhi = prepare_stock_list(context)
    current_data = get_current_data()
    
    log.info(f"今日监控股票数量: {len(g.yizhi)}")
    
    # 为每个股票设置监听，但只在tick级别策略中使用subscribe
    try:
        # 只在支持tick订阅的环境中执行
        for s in g.yizhi:
            if hasattr(context, 'subscribe'):
                subscribe(s, 'tick')
            g.stock_data[s] = current_data[s].high_limit
            log.info("监听 %s - %s" % (current_data[s].name, s))
    except Exception as e:
        log.warning(f"订阅股票时出错: {e}")
        # 在不支持tick订阅的环境中，只记录信息

def saixuan(context):
    """筛选函数"""
    now = context.current_dt
    current_data = get_current_data()
    
    for s in g.yizhi:
        try:
            df_panel_all = get_price(
                s,
                count=50,
                end_date=now,
                frequency='minute',
                fields=['open','high','low','close','high_limit','money','volume']
            )
            
            if current_data[s].day_open * 1.06 < df_panel_all['close'][-1] < current_data[s].high_limit:
                if df_panel_all['high'].max() == current_data[s].high_limit:
                    continue
                
                # 只在支持的环境中订阅
                try:
                    subscribe(s, 'tick')
                except:
                    pass
                    
                g.stock_data[s] = current_data[s].high_limit
                log.info("筛选后监听 %s", current_data[s].name)
        except Exception as e:
            log.warning(f"计算{s}数据时出错: {e}")
            continue

def buy(context):
    """买入函数"""
    if g.buy_complited or g.today == 5 or len(context.portfolio.positions) >= g.stock_num:
        return
        
    current_data = get_current_data()
    time_now = context.current_dt.strftime('%H:%M:%S')
    
    # 交易时间检查
    if time_now >= '14:30:00' or time_now < '09:32:00':
        return
        
    qualified_stocks = g.youxian + g.jianting
    log.info("股票池数量: %d", len(qualified_stocks))
    
    for stock in qualified_stocks:
        # 跳过已被移除或已持有的股票
        if (stock in g.remove_list or stock in g.all_remove or 
            stock in list(context.portfolio.positions)):
            continue
            
        # 检查开盘即涨停
        if current_data[stock].day_open == current_data[stock].high_limit:
            g.remove_list.append(stock)
            continue
            
        # 检查跌幅过大
        if get_current_data()[stock].last_price < get_current_data()[stock].high_limit / 1.1:
            log.info("%s 跌3个点，remove" % stock)
            g.remove_list.append(stock)
            continue
            
        # 计算涨幅比例
        try:
            now = context.current_dt
            zeroToday = now - timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)
            lastToday = zeroToday + timedelta(hours=9, minutes=30)
            
            df_panel_all = get_price(
                stock,
                start_date=lastToday,
                end_date=now,
                frequency='minute',
                fields=['open','high','low','close','high_limit','money','volume']
            )
            
            zhangfu_ratio = df_panel_all['close'][-1] / (df_panel_all['high_limit'][-1] / 1.1)
            
            # 优先股处理
            if zhangfu_ratio > 1.6 and stock not in g.youxian:
                g.youxian.append(stock)
                
            # 买入条件
            if (zhangfu_ratio >= 1.09 and 
                context.portfolio.available_cash > 0 and
                (g.stock_num - len(context.portfolio.positions)) > 0):
                
                value = context.portfolio.available_cash / (g.stock_num - len(context.portfolio.positions))
                
                if value / current_data[stock].last_price > 100:
                    order_value(stock, value)
                    feishu(stock, "买入")
                    log.info('买入 %s -> %s' % (stock, current_data[stock].name))
                    
                    # 检查是否完成买入
                    if len(context.portfolio.positions) >= g.stock_num:
                        g.buy_complited = True
                        return
                        
        except Exception as e:
            log.warning(f"计算{stock}买入条件时出错: {e}")
            continue

def sell(context):
    """卖出函数"""
    current_data = get_current_data()
    time_str = context.current_dt.strftime("%H%M")
    
    # 上午卖出：有利润就跑
    if time_str == '1125':
        for s in list(context.portfolio.positions):
            try:
                pos = context.portfolio.positions[s]
                if (pos.closeable_amount > 0 and 
                    current_data[s].last_price < current_data[s].high_limit and
                    current_data[s].last_price > (1.00 * pos.avg_cost)):
                    
                    order_target_value(s, 0)
                    log.info(f"上午卖出 {s} - {current_data[s].name}")
                    
            except Exception as e:
                log.warning(f"卖出{s}时出错: {e}")
                continue
                
    # 下午卖出：避免收盘风险
    elif time_str == '1450':
        for s in list(context.portfolio.positions):
            try:
                pos = context.portfolio.positions[s]
                if pos.closeable_amount > 0 and current_data[s].last_price < current_data[s].high_limit:
                    order_target_value(s, 0)
                    log.info(f"下午卖出 {s} - {current_data[s].name}")
                    
            except Exception as e:
                log.warning(f"卖出{s}时出错: {e}")
                continue

################################### 过滤函数群 ##################################
# 新增过滤函数
def filter_negative_pe_stock(initial_list, date):
    """过滤掉市盈率小于0的股票"""
    filtered_list = []
    batch_size = 100  # 分批处理，避免一次查询过多
    
    for i in range(0, len(initial_list), batch_size):
        batch_stocks = initial_list[i:i+batch_size]
        
        try:
            # 批量查询市盈率
            q = query(
                valuation.code, 
                valuation.pe_ratio
            ).filter(
                valuation.code.in_(batch_stocks)
            )
            
            df = get_fundamentals(q, date=date)
            
            if df.empty:
                # 没有数据，暂时保留
                filtered_list.extend(batch_stocks)
                continue
                
            # 创建市盈率字典
            pe_dict = dict(zip(df['code'], df['pe_ratio']))
            
            for stock in batch_stocks:
                if stock in pe_dict:
                    pe_ratio = pe_dict[stock]
                    if pe_ratio is not None and pe_ratio >= 0:
                        filtered_list.append(stock)
                    else:
                        stock_name = get_security_info(stock).display_name
                        log.info(f"过滤股票{stock}({stock_name})：市盈率{pe_ratio} < 0")
                else:
                    # 没有市盈率数据，保守保留
                    filtered_list.append(stock)
                    
        except Exception as e:
            log.warning(f"批量查询市盈率时出错: {e}")
            # 出错时暂时保留该批次股票
            filtered_list.extend(batch_stocks)
    
    return filtered_list

def filter_consecutive_limit_up_stock(initial_list, context, days=3):
    """过滤掉相对于当日，倒退3天连续涨停的股票"""
    filtered_list = []
    yesterday = context.previous_date
    
    # 分批处理提高效率
    batch_size = 50
    
    for i in range(0, len(initial_list), batch_size):
        batch_stocks = initial_list[i:i+batch_size]
        
        try:
            # 批量获取价格数据
            price_data = get_price(
                batch_stocks,
                end_date=yesterday,
                frequency='daily',
                fields=['close', 'high_limit'],
                count=days,
                panel=False,
                skip_paused=True
            )
            
            if price_data.empty:
                filtered_list.extend(batch_stocks)
                continue
            
            # 按股票分组
            for stock in batch_stocks:
                stock_data = price_data[price_data['code'] == stock]
                
                if stock_data.empty or len(stock_data) < days:
                    # 数据不足，保守保留
                    filtered_list.append(stock)
                    continue
                
                # 检查是否连续涨停
                is_consecutive_limit_up = True
                for _, row in stock_data.iterrows():
                    if row['close'] != row['high_limit']:
                        is_consecutive_limit_up = False
                        break
                
                if is_consecutive_limit_up:
                    stock_info = get_security_info(stock)
                    log.info(f"过滤股票{stock}({stock_info.display_name})：连续{days}天涨停")
                else:
                    filtered_list.append(stock)
                    
        except Exception as e:
            log.warning(f"批量检查连续涨停时出错: {e}")
            # 出错时暂时保留
            filtered_list.extend(batch_stocks)
    
    return filtered_list

################################### 股票筛选函数 ##################################
def prepare_stock_list(context):
    """主要股票筛选函数 - 包含新过滤条件"""
    today = context.current_dt.date()
    yesterday = context.previous_date
    
    # 1. 获取初始股票池
    initial_list = set_stockpool(context)
    
    # 2. 基础过滤
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_paused_stock(initial_list, today)
    initial_list = filter_new_stock(initial_list, today)
    
    # 3. 新增过滤条件
    initial_list = filter_negative_pe_stock(initial_list, yesterday)
    initial_list = filter_consecutive_limit_up_stock(initial_list, context, days=3)
    
    # 4. 筛选昨日涨停的股票
    initial_list = get_hl_stock(initial_list, yesterday, 1)
    
    # 5. 筛选今日开盘合理的股票
    cur = get_current_data()
    hl_list = []
    
    for s in initial_list:
        try:
            if (s in cur and 
                cur[s].day_open > cur[s].high_limit / 1.1 * 1.08 and 
                cur[s].day_open < cur[s].high_limit):
                hl_list.append(s)
        except:
            continue
    
    log.info(f"最终筛选后股票数量: {len(hl_list)}")
    return hl_list

def prepare_stock_list8(context):
    """备用股票筛选函数"""
    today = context.current_dt.date()
    yesterday = context.previous_date
    
    initial_list = set_stockpool(context)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_paused_stock(initial_list, today)
    
    # 新增过滤条件
    initial_list = filter_negative_pe_stock(initial_list, yesterday)
    initial_list = filter_consecutive_limit_up_stock(initial_list, context, days=3)
    
    ZHANGTING1 = get_hl_stock(initial_list, yesterday, 1)
    res = []
    cur = get_current_data()
    
    for s in ZHANGTING1:
        try:
            if (s in cur and 
                cur[s].day_open <= cur[s].high_limit / 1.1 * 1.08 and 
                cur[s].day_open != cur[s].high_limit):
                res.append(s)
        except:
            continue
    
    return res

################################### 辅助函数群 ##################################
def set_stockpool(context):
    """定义股票池"""
    yesterday = context.previous_date
    try:
        return get_all_securities('stock', yesterday).index.tolist()
    except:
        return []

def get_hl_stock(stock_list, date1, days):
    """筛选出某一日涨停的股票"""
    if not stock_list:
        return []
    
    try:
        # 分批处理
        batch_size = 100
        result = []
        
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i:i+batch_size]
            
            h_s = get_price(
                batch,
                end_date=date1,
                frequency='daily',
                fields=['close', 'high_limit'],
                count=days,
                panel=False,
                skip_paused=True
            )
            
            if h_s.empty:
                continue
                
            # 筛选涨停股票
            limit_up_stocks = h_s[h_s['close'] == h_s['high_limit']]['code'].unique()
            result.extend(limit_up_stocks)
        
        return list(set(result))  # 去重
        
    except Exception as e:
        log.warning(f"筛选涨停股时出错: {e}")
        return []

def filter_kcbj_stock(initial_list):
    """过滤科创板、创业板等特殊板块"""
    return [stock for stock in initial_list 
            if not (stock.startswith('688') or 
                   stock.startswith('300') or 
                   stock.startswith('4') or 
                   stock.startswith('8'))]

def filter_st_paused_stock(initial_list, date):
    """过滤ST、停牌、退市股"""
    current_data = get_current_data()
    filtered_list = []
    
    for stock in initial_list:
        try:
            if stock in current_data:
                security = current_data[stock]
                if (not security.is_st and 
                    not security.paused and 
                    '退' not in security.name):
                    filtered_list.append(stock)
        except:
            continue
    
    return filtered_list

def filter_new_stock(initial_list, date, days=60):
    """过滤新股"""
    filtered_list = []
    
    for stock in initial_list:
        try:
            info = get_security_info(stock)
            # 确保上市时间足够长
            if info and info.start_date:
                # 确保有足够的交易日
                all_trade_days = get_all_trade_days()
                start_timestamp = pd.Timestamp(info.start_date)
                if sum(1 for trade_day in all_trade_days if trade_day >= start_timestamp) > days:
                    filtered_list.append(stock)
        except:
            continue
    
    return filtered_list

################################### 工具函数 ##################################
def gen_sign(secret):
    """生成飞书签名"""
    timestamp = int(time.time())
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"), 
        digestmod=hashlib.sha256
    ).digest()
    sign = base64.b64encode(hmac_code).decode('utf-8')
    return sign

def feishu(stock, action):
    """发送飞书通知"""
    if not g.push:
        return
    
    try:
        current_data = get_current_data()
        stock_name = current_data[stock].name if stock in current_data else stock
        
        params = {
            "timestamp": int(time.time()),
            "sign": gen_sign(g.WEBHOOK_SECRET),
            "msg_type": "text",
            "content": {
                "text": f"【首板】{action} -> {stock} -> {stock_name}"
            },
        }
        requests.post(g.WEBHOOK_URL, json=params, timeout=5)
    except Exception as e:
        log.warning(f"发送飞书通知失败: {e}")

def handle_tick(context, tick):
    """Tick处理函数（如果需要）"""
    current_data = get_current_data()
    
    # 检查是否已满仓或不在交易时间
    if (len(context.portfolio.positions) >= g.stock_num or
        tick.current is None or tick.code is None):
        return
        
    time_now = context.current_dt.strftime('%H:%M:%S')
    if time_now >= '10:30:00' or time_now < '09:30:00':
        return
        
    if (tick.code in g.today_tick or 
        tick.code in g.remove_list or 
        tick.code not in g.stock_data):
        return
        
    # 检查价格条件
    target_price = g.stock_data[tick.code]
    if target_price is None or tick.current is None:
        return
        
    if tick.current < target_price / 1.1:
        g.remove_list.append(tick.code)
        return
        
    if (tick.current >= target_price - 0.05 and 
        tick.current >= current_data[tick.code].day_open and 
        tick.code not in list(context.portfolio.positions)):
        
        cash_available = context.portfolio.available_cash
        positions_available = g.stock_num - len(context.portfolio.positions)
        
        if positions_available > 0 and cash_available > 100:
            value = cash_available / positions_available
            if value > 100:
                order_value(tick.code, value)
                g.today_tick[tick.code] = 1
                log.info("买入 %s" % get_current_data()[tick.code].name)
