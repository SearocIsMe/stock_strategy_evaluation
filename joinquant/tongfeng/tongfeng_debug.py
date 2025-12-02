# 克隆自聚宽文章：https://www.joinquant.com/post/62097
# 标题：桐峰小市值-第三版-邮件发送
# 作者：蚂蚁量化

#### 克隆自聚宽文章：https://www.joinquant.com/post/58646
#### 标题：实盘前的debug让收益翻倍(十年回测近千倍)
#### 作者：蚂蚁量化

#### 克隆自聚宽文章：https://www.joinquant.com/post/58633
#### 标题：准备实盘，帮忙看看是否可以？
#### 作者：桐峰


# -*- coding: utf-8 -*-
"""
量化交易策略：小市值选股+动态止损系统
核心逻辑：
1. 每月从中小板指筛选市值5-300亿的股票池
2. 每周调仓持有市值最小的4只股票
3. 实施组合止损策略（个股止损+大盘止损）
4. 特殊处理涨停板股票
5. 每年1月和4月空仓
"""

# 聚宽平台API
from jqdata import *  
import numpy as np  
import pandas as pd  
from datetime import time, timedelta  

import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time


# ========== 全局参数配置 ==========
BENCHMARK = '000001.XSHG'  # 上证指数作为基准
MARKET_INDEX = '399101.XSHE'  # 中小板指作为选股范围
EMPTY_MONTHS = [1, 4]  # 1月和4月空仓
CASH_ETF = '511880.XSHG'  # 货币ETF用于空仓时现金管理

# 止损策略类型常量
STOPLOSS_SINGLE = 1   # 仅个股止损
STOPLOSS_MARKET = 2   # 仅大盘止损 
STOPLOSS_COMBINED = 3 # 复合止损策略（默认）


def initialize(context):
    """策略初始化函数，由聚宽框架自动调用"""
    # 防止未来函数
    set_option('avoid_future_data', True)
    # 设置基准收益率
    set_benchmark(BENCHMARK)
    # 使用真实价格回测
    set_option('use_real_price', True)
    # 设置滑点（单边0.3%）
    set_slippage(FixedSlippage(3/1000))
    # 设置交易成本（股票万2.5，卖出印花税0.1%）
    set_order_cost(
        OrderCost(
            open_tax=0,                # 买入印花税
            close_tax=0.001,           # 卖出印花税
            open_commission=2.5/10000, # 买入佣金
            close_commission=2.5/10000,# 卖出佣金
            close_today_commission=0,  # 平今佣金（股票无）
            min_commission=5           # 最低佣金
        ),
        type='stock'
    )
    
    # 设置日志级别
    log.set_level('order', 'error')   # 订单日志只报错
    log.set_level('system', 'error')  # 系统日志只报错
    log.set_level('strategy', 'debug') # 策略日志显示debug信息

def after_code_changed(context):
    unschedule_all()
    # ========== 初始化全局变量 ==========
    g.trading_signal = True       # 当日是否可交易
    g.run_stoploss = True         # 是否运行止损逻辑
    g.yesterday_HL_list = []      # 昨日涨停股票列表
    g.target_list = []            # 本周目标股票池
    g.pass_months = EMPTY_MONTHS  # 空仓月份配置
    g.limitup_stocks = []         # 当日涨停股票列表
    g.target_stock_count = 4      # 目标持仓数量
    g.sell_reason = ''            # 卖出原因记录（用于日志）
    g.stoploss_strategy = STOPLOSS_COMBINED  # 使用复合止损策略
    g.stoploss_limit = 0.06       # 个股止损阈值6%
    g.stoploss_market = 0.05      # 大盘止损阈值5%
    g.etf = CASH_ETF              # 现金管理ETF代码
    g.email_content = ''          # 邮件内容
    
    # 初始执行选股
    filter_monthly(context)
    # ========== 定时任务设置 ==========
    run_monthly(filter_monthly, 1, '9:00')      # 每月1号9点选股
    run_daily(prepare_stock_list, 'before_open')       # 每日开盘前准备
    run_daily(trade_afternoon, '14:00')         # 下午交易时段
    run_daily(sell_stocks, '10:00')             # 上午执行止损
    run_daily(close_account, '14:50')           # 收盘前清理仓位
    run_daily(final_report, '15:10')           # 收盘后报告
    run_weekly(weekly_adjustment, 2, '10:00')   # 每周二调仓

def prepare_stock_list(context):
    """每日开盘前准备数据"""
    # 获取当前持仓列表
    hold_list = list(context.portfolio.positions.keys())
    g.limitup_stocks = []  # 重置当日涨停列表
    g.email_content = f"\n\n============今天是 {context.current_dt.strftime('%Y-%m-%d')} ============"    # "重置邮件内容"

    if hold_list:
        # 获取持仓股昨日收盘价和涨停价
        price_df = get_price(
            hold_list,
            end_date=context.previous_date,  # 使用前一日数据
            frequency='daily',
            fields=['close', 'high_limit'], # 需要收盘价和涨停价
            count=1,
            panel=False,
            fill_paused=False
        )
        # 筛选昨日收盘价等于涨停价的股票
        g.yesterday_HL_list = price_df[price_df['close'] == price_df['high_limit']]['code'].tolist()
    else:
        g.yesterday_HL_list = []

    # 检查当日是否可交易（非空仓月份）
    g.trading_signal = today_is_tradable(context)
    g.email_content += f"\n 昨日涨停, 下午破板将卖出的股票: {g.yesterday_HL_list}"
    g.email_content += f"\n 今日 {'不是' if g.trading_signal else '是'} 空仓月份{g.pass_months}"
    send_email_weixin(context,g.email_content)
    
def filter_monthly(context):
    """月度选股：从中小板指筛选小市值股票"""
    # 构建查询：选择中小板指成分股，市值5-300亿，按市值升序排列
    q = query(
        valuation.code,
    ).filter(
        valuation.code.in_(get_index_stocks(MARKET_INDEX)),
        valuation.market_cap.between(5, 300)  # 市值单位：亿元
    ).order_by(
        valuation.market_cap.asc()  # 小市值优先
    )
    # 获取基本面数据
    fund_df = get_fundamentals(q)
    # 取市值最小的N*20只股票（N为目标持仓数）
    g.month_scope = fund_df['code'].head(g.target_stock_count * 20).tolist()
    
def get_stock_list(context):
    """从月度股票池筛选最终候选股票"""
    # 先进行基础过滤（剔除ST、新股等）
    filtered_stocks = filter_stocks(context, g.month_scope)

    # 再次查询市值数据
    q = query(
        valuation.code,
        valuation.market_cap
    ).filter(
        valuation.code.in_(filtered_stocks),
        valuation.market_cap.between(5, 300)
    ).order_by(
        valuation.market_cap.asc()  # 仍然按市值排序
    )
    fund_df = get_fundamentals(q)
    # 取市值最小的N*3只作为候选（N为目标持仓数）
    candidate_stocks = fund_df['code'].head(g.target_stock_count * 3).tolist()
    return candidate_stocks

def weekly_adjustment(context):
    hold_list = list(context.portfolio.positions.keys())
    """每周调仓逻辑"""
    if not g.trading_signal:
        # 空仓月份直接买入货币ETF
        g.email_content += f"\n空仓月份({g.pass_months}), 买入{g.etf}"
        buy_security(context, [g.etf])
        return

    # 获取本周目标股票池
    g.target_list = get_stock_list(context)
    g.email_content += (f"\n本周股票池有:{len(g.target_list)}只股票")
    
    # 构建卖出列表（需同时满足三个条件）：
    # 1. 不在本周目标前N名
    # 2. 昨日未涨停（给予涨停股额外持有机会）
    # 3. 未停牌
    current_data = get_current_data()
    sell_list = [
        stock for stock in hold_list
        if stock not in g.target_list[:g.target_stock_count] and 
           stock not in g.yesterday_HL_list and 
           not current_data[stock].paused
    ]
    
    # 执行卖出
    for stock in sell_list:
        if current_data[stock].paused: continue  # 跳过停牌股
        g.email_content += (f"\n卖出 {stock}")
        order_target_value(stock, 0)    

    
    # 计算需要买入的数量
    to_buy_num = g.target_stock_count - len(context.portfolio.positions)
    # 构建买入列表（需同时满足三个条件）：
    # 1. 在目标池中
    # 2. 当前未持有
    # 3. 昨日未涨停（避免追高）
    to_buy = [x for x in g.target_list 
             if x not in context.portfolio.positions.keys() and 
                x not in g.yesterday_HL_list][:to_buy_num]
    buy_security(context, to_buy)
    
    send_email_weixin(context,g.email_content)
    

def check_limit_up(context):
    """检查昨日涨停股今日是否开板"""
    if not g.yesterday_HL_list:
        return
        
    current_data = get_current_data()
    for stock in g.yesterday_HL_list:
        current_close = current_data[stock].last_price
        high_limit = current_data[stock].high_limit

        if current_close < high_limit:
            # 如果涨停打开则卖出
            if current_data[stock].paused: continue  # 跳过停牌股
            g.email_content += f"\n {stock}涨停打开，执行卖出"
            order_target_value(stock, 0)    
            g.sell_reason = 'limitup'  # 记录卖出原因
            g.limitup_stocks.append(stock)  # 加入当日涨停列表

        else:
            g.email_content += f"\n {stock} 封涨停，继续持有"

def check_remain_amount(context):
    """卖出后剩余资金处理"""
    if not g.sell_reason:  # 无卖出操作直接返回
        return

    hold_list = list(context.portfolio.positions.keys())
    cash = context.portfolio.cash

    if g.sell_reason == 'limitup':
        # 涨停卖出后的资金再投资
        need_buy_count = g.target_stock_count - len(hold_list)
        if need_buy_count > 0:
            # 从目标池排除已涨停股票
            candidates = [s for s in g.target_list 
                         if s not in g.limitup_stocks and 
                            s not in hold_list]
            buy_list = candidates[:need_buy_count]
            g.email_content += f"\n涨停卖出后剩余资金{cash:.2f}元，补仓：{buy_list}"
            buy_security(context, buy_list)
    elif g.sell_reason == 'stoploss':
        # 止损后转货币ETF
        g.email_content += f"\n止损后剩余资金{cash:.2f}元，买入{g.etf}"
        buy_security(context, [g.etf])

    g.sell_reason = ''  # 重置卖出原因

def trade_afternoon(context):
    """下午交易时段操作"""        
    cash1 = context.portfolio.cash   
    if g.trading_signal:
        check_limit_up(context)   # 检查涨停股
        check_remain_amount(context)  # 处理剩余资金
    if context.portfolio.cash!=cash1:
        send_email_weixin(context,g.email_content)

def sell_stocks(context):
    """执行止损策略"""
    if not g.run_stoploss:  # 止损开关检查
        return

    positions = context.portfolio.positions
    if not positions:  # 无持仓直接返回
        return
        
    current_data = get_current_data()
    cash1 = context.portfolio.cash    
    # 个股止损逻辑（复合策略或单独策略）
    if g.stoploss_strategy in (STOPLOSS_SINGLE, STOPLOSS_COMBINED):
        for stock, pos in positions.items():
            current_price = pos.price
            avg_cost = pos.avg_cost
            if current_data[stock].paused: continue  # 跳过停牌股
            
            # 止盈逻辑（收益率≥100%）
            if current_price >= avg_cost * 2:
                order_target_value(stock, 0)
                g.email_content += f"\n{stock} 收益100%，执行止盈"
            # 止损逻辑（亏损≥阈值）
            elif current_price < avg_cost * (1 - g.stoploss_limit):
                order_target_value(stock, 0)
                g.email_content += f"\n{stock} 跌幅达 {int(g.stoploss_limit*100)}%，执行止损"
                g.sell_reason = 'stoploss'  # 记录止损原因

    # 大盘止损逻辑（复合策略或单独策略）
    if g.stoploss_strategy in (STOPLOSS_MARKET, STOPLOSS_COMBINED):
        # 获取中小板指当日涨跌幅
        index_price = get_price(
            MARKET_INDEX,
            end_date=context.previous_date,
            frequency='daily',
            fields=['open', 'close'],
            count=1
        )
        if not index_price.empty:
            # 计算日内涨跌幅（收盘/开盘-1）
            market_down_ratio = (index_price['close'].iloc[0] / index_price['open'].iloc[0]) - 1
            if abs(market_down_ratio) >= g.stoploss_market:
                # 当昨天日内涨跌幅超过g.stoploss_market, 清仓所有非ETF持仓
                for stock in positions.keys():
                    if stock == g.etf: continue
                    order_target_value(stock, 0)
                g.sell_reason = 'stoploss'
                g.email_content += (f"\n市场平均跌幅 {market_down_ratio:.2%}，执行止损")
    if context.portfolio.cash!=cash1:
        send_email_weixin(context,g.email_content)
    

def filter_stocks(context, stock_list):
    """股票过滤器：剔除不符合条件的股票"""
    if not stock_list:
        return []
        
    hold_list = list(context.portfolio.positions.keys())

    current_data = get_current_data()
    # 获取前一分钟收盘价（用于判断涨跌停）
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    filtered = []

    for stock in stock_list:
        data = current_data[stock]
        # 基础过滤条件
        if data.paused: continue                   # 剔除停牌股
        if data.is_st or '退' in data.name:       # 剔除ST/*ST/退市股
            continue  
        if stock.startswith(('30', '68', '8', '4')): # 剔除创业板/科创板等
            continue  
        # 涨跌停过滤（已持仓股不受限）
        if stock not in hold_list and last_prices[stock].iloc[-1] >= data.high_limit:
            continue  # 剔除涨停股
        if stock not in hold_list and last_prices[stock].iloc[-1] <= data.low_limit:
            continue  # 剔除跌停股
        # 次新股过滤（上市不满375天）
        listing_date = get_security_info(stock).start_date
        if (context.previous_date - listing_date).days < 375:
            continue

        filtered.append(stock)
    return filtered

    
def close_account(context):
    """收盘前清理仓位（空仓月份专用）"""
    cash1 = context.portfolio.cash    
    if not g.trading_signal:
        current_data = get_current_data()
        hold_list = list(context.portfolio.positions.keys())
        for stock in hold_list: 
            if current_data[stock].paused: continue  # 跳过停牌股
            if stock == g.etf: continue              # 保留货币ETF
            g.email_content += (f"\n空仓月卖出 {stock}")
            order_target_value(stock, 0)      
    if context.portfolio.cash!=cash1:
        send_email_weixin(context,g.email_content)
            
            
# ========== 以下是工具函数 ==========
def buy_security(context, target_list):
    """按等金额买入股票"""
    current_hold = [pos.security for pos in context.portfolio.positions.values()]
    need_buy = [stock for stock in target_list if stock not in current_hold]
    if not need_buy:
        return
    
    current_data = get_current_data()
    buy_count = len(need_buy)
    cash = context.portfolio.cash
    if cash <= 0 or buy_count <= 0:
        return

    # 等分现金买入
    per_stock_value = cash / buy_count

    for stock in need_buy:
        if current_data[stock].paused: continue  # 跳过停牌股
        g.email_content += (f"\n买入 {stock}，金额 {per_stock_value:.2f} 元")
        order_target_value(stock, per_stock_value)
        # 达到目标持仓数即停止
        if len(context.portfolio.positions) >= g.target_stock_count:
            break

def today_is_tradable(context):
    """检查当日是否交易日（非空仓月份）"""
    return context.current_dt.month not in g.pass_months
    

def final_report(context):
    current_data = get_current_data()
    hold_list = list(context.portfolio.positions.keys())
    g.email_content += f"\n账户持仓:"
    for stock in hold_list:
        qty = context.portfolio.positions[stock].total_amount
        g.email_content += f"\n {stock}, {current_data[stock].name}, {qty} 股"
    g.email_content += f"\n现金: {int(context.portfolio.cash)} 元"
    g.email_content += f"\n盈利: {context.portfolio.total_value/context.portfolio.starting_cash-1:.2f} %"
    g.email_content += f"\n==================报告结束================\n\n"
    
    send_email_weixin(context,g.email_content)
  
#     ===============================================\
#     邮件、微信发送函数 eeeeeeeeeeeeeeeeeeeeeeeeeeee
#     ===============================================\
    
def send_email_weixin(context,content):
    print(content)
    if context.run_params.type == 'sim_trade':
        print('\n- 微信内容发送', context.current_dt)
        send_message(content, channel='weixin')
        
        print('\n- QQ邮箱发送', context.current_dt)
        subject = "小市值交易信号:" + context.current_dt.strftime('%Y-%m-%d %H:%M')
        content = content
        to_email = "xxxxxxxxxxx@qq.com"  # 收件人邮箱
        QQ_email_sending(subject,content,to_email)     
    else:
        print("- 回测时只显示邮件内容，不实际发送邮件")
        
def QQ_email_sending(subject,content,to_email):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.header import Header
    from email.utils import formataddr
    import time

    # 邮件配置
    smtp_server = "smtp.126.com"
    smtp_port = 587
    email_address = "jianghp2000@126.com"   # 发件人QQ邮箱
    password = "HP@750701"            ## 注意，授权码，非密码
    sender_name = "蚂蚁量化"               # 发件人名称
    
    # 创建邮件对象
    msg = MIMEMultipart()
    msg['From'] = formataddr((Header(sender_name, 'utf-8').encode(), email_address))  # 发件人姓名含中文
    msg['To'] = to_email                        # 收件人邮箱
    msg['Subject'] = Header(subject, 'utf-8')  # 主题含中文

    # 邮件正文
    body = content
    msg.attach(MIMEText(body, 'plain', 'utf-8'))  # 明确指定正文编码为 utf-8

    # 发送邮件
    max_retries = 3
    for attempt in range(max_retries):
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  # 启用加密
            server.login(email_address, password)
            server.sendmail(email_address, [to_email], msg.as_string())
            server.quit()
            print("邮件发送成功")
            break
        except Exception as e:
            print(f"发送失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("等待1秒后重新尝试...")
                time.sleep(1)  # 等待1秒再重试
            else:
                print(f"已达到最大重试次数，跳过发送给: {to_email}")




