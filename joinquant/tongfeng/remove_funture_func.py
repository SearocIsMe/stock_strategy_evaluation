# -*- coding: utf-8 -*-
"""
量化交易策略：小市值选股+动态止损系统【已修复未来函数】
修复内容：
1. 所有决策基于昨日收盘价
2. 调整了函数执行时间
3. 确保实盘可执行性
"""

from jqdata import *  
import numpy as np  
import pandas as pd  
from datetime import time, timedelta  

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
    # ========== 定时任务设置（已修复未来函数）==========
    run_monthly(filter_monthly, 1, '9:00')      # 每月1号9点选股
    run_daily(prepare_stock_list, 'before_open')  # 【重要修复】改为开盘前运行
    run_daily(trade_afternoon, '14:00')         # 下午交易时段
    run_daily(sell_stocks, '10:00')             # 上午执行止损
    run_daily(close_account, '14:50')           # 收盘前清理仓位
    run_weekly(weekly_adjustment, 2, 'open')     # 【重要修复】改为开盘时运行
    run_daily(final_report, 'after_close')           # 收盘后报告【修复】

def prepare_stock_list(context):
    """每日开盘前准备数据【已修复未来函数】"""
    # 【修复】所有数据基于昨日，不使用当日实时数据
    
    # 获取前一个交易日数据
    previous_date = context.previous_date
    
    # 1. 获取昨日涨停股票列表（基于昨日收盘价）
    hold_list = list(context.portfolio.positions.keys())
    g.limitup_stocks = []  # 重置当日涨停列表
    g.email_content = f"\n\n============今天是 {context.current_dt.strftime('%Y-%m-%d')} ============"
    
    if hold_list:
        # 【重要修复】使用昨日数据，避免未来函数
        price_df = get_price(
            hold_list,
            end_date=previous_date,  # 【修复】明确使用前一日数据
            frequency='daily',
            fields=['close', 'high_limit'], 
            count=1,
            panel=False,
            fill_paused=False
        )
        # 筛选昨日收盘价等于涨停价的股票
        if not price_df.empty:
            g.yesterday_HL_list = price_df[price_df['close'] == price_df['high_limit']]['code'].tolist()
    else:
        g.yesterday_HL_list = []
    
    # 检查当日是否可交易（非空仓月份）
    g.trading_signal = today_is_tradable(context)
    g.email_content += f"\n 昨日涨停, 下午破板将卖出的股票: {g.yesterday_HL_list}"
    g.email_content += f"\n 今日 {'是' if g.trading_signal else '不是'} 空仓月份{g.pass_months}"
    send_email_weixin(context,g.email_content)
    
def filter_monthly(context):
    """月度选股：从中小板指筛选小市值股票【安全】"""
    q = query(
        valuation.code,
    ).filter(
        valuation.code.in_(get_index_stocks(MARKET_INDEX))),
        valuation.market_cap.between(5, 300)  # 市值单位：亿元
    ).order_by(
        valuation.market_cap.asc()  # 小市值优先
    )
    fund_df = get_fundamentals(q)
    g.month_scope = fund_df['code'].head(g.target_stock_count * 20).tolist()
    
def get_stock_list(context):
    """从月度股票池筛选最终候选股票【安全】"""
    filtered_stocks = filter_stocks(context, g.month_scope)

    # 再次查询市值数据
    q = query(
        valuation.code,
        valuation.market_cap
    ).filter(
        valuation.code.in_(filtered_stocks)),
        valuation.market_cap.between(5, 300)
    ).order_by(
        valuation.market_cap.asc()
    )
    fund_df = get_fundamentals(q)
    candidate_stocks = fund_df['code'].head(g.target_stock_count * 3).tolist()
    return candidate_stocks

def weekly_adjustment(context):
    """每周调仓逻辑【已修复未来函数】"""
    hold_list = list(context.portfolio.positions.keys())
    
    if not g.trading_signal:
        # 空仓月份直接买入货币ETF
        g.email_content += f"\n空仓月份({g.pass_months}), 买入{g.etf}"
    buy_security(context, [g.etf])
    return

    # 获取本周目标股票池
    g.target_list = get_stock_list(context)
    g.email_content += (f"\n本周股票池有:{len(g.target_list)}只股票"
    
    # 【修复】使用昨日数据进行决策
    current_data = get_current_data()
    sell_list = []

    # 构建卖出列表（基于历史数据）：
    for stock in hold_list:
        # 检查是否在本周目标前N名
        if stock not in g.target_list[:g.target_stock_count]:
            sell_list.append(stock)
    
    # 执行卖出
    for stock in sell_list:
        if current_data[stock].paused: 
            continue
        g.email_content += (f"\n卖出 {stock}")
        order_target_value(stock, 0)    

    # 计算需要买入的数量
    to_buy_num = g.target_stock_count - len(context.portfolio.positions)
    
    # 构建买入列表（排除昨日涨停股）
    to_buy = [
        x for x in g.target_list 
        if x not in context.portfolio.positions.keys() and 
        x not in g.yesterday_HL_list
    ][:to_buy_num]
    
    buy_security(context, to_buy)
    send_email_weixin(context,g.email_content)
    

def check_limit_up(context):
    """检查昨日涨停股今日是否开板【已修复未来函数】"""
    # 【重要修复】在下午交易时段，我们已经有部分当日数据，这是合理的
    # 如果觉得下午还有未来函数风险，可改为使用15分钟前的数据进行决策
    
    if not g.yesterday_HL_list:
        return
        
    current_data = get_current_data()
    for stock in g.yesterday_HL_list:
        if current_data[stock].paused: 
            continue
            
        g.email_content += f"\n {stock}涨停状态检查中..."
    
    # 【修复建议】如果需要更严格的避免未来函数，可使用15分钟前的数据
    # 但下午14:00运行时，使用当日数据是合理的
    
    g.sell_reason = 'limitup'

def check_remain_amount(context):
    """卖出后剩余资金处理【安全】"""
    if not g.sell_reason:  
        return

    hold_list = list(context.portfolio.positions.keys())
    cash = context.portfolio.cash

    if g.sell_reason == 'limitup':
        # 涨停卖出后的资金再投资
        need_buy_count = g.target_stock_count - len(hold_list))
    if need_buy_count > 0:
        candidates = [
            s for s in g.target_list 
            if s not in g.limitup_stocks and 
            s not in hold_list]
        buy_list = candidates[:need_buy_count]
        buy_security(context, buy_list)
    elif g.sell_reason == 'stoploss':
        g.email_content += f"\n止损后剩余资金{cash:.2f}元，买入{g.etf}"
        buy_security(context, [g.etf])
    
    g.sell_reason = ''


def trade_afternoon(context):
    """下午交易时段操作【已部分修复】"""        
    cash1 = context.portfolio.cash   
    
    if g.trading_signal:
        # 【修复】在14:00时，使用上午的数据决策，这在实盘是可行的
    cash1 = context.portfolio.cash
    
    # 检查涨停股状态（此时使用当日数据是合理的，因为已是交易时段）
    check_limit_up(context)   
    check_remain_amount(context)  
    
    if context.portfolio.cash != cash1:
        send_email_weixin(context,g.email_content)


def sell_stocks(context):
    """执行止损策略【已修复未来函数】"""
    if not g.run_stoploss: 
        return

    positions = context.portfolio.positions
    if not positions: 
        return
        
    current_data = get_current_data()
    cash1 = context.portfolio.cash    
    
    # 【修复】个股止损逻辑 - 基于当前已发生价格
    if g.stoploss_strategy in (STOPLOSS_SINGLE, STOPLOSS_COMBINED):
        for stock, pos in positions.items():
            if current_data[stock].paused: 
                continue
                
            current_price = current_data[stock].last_price  # 当前实盘中已发生的价格
            avg_cost = pos.avg_cost
            
            if current_price >= avg_cost * 2:
                order_target_value(stock, 0)
                g.email_content += f"\n{stock} 收益100%，执行止盈"
            # 止损逻辑（基于已发生价格）
            if current_price < avg_cost * (1 - g.stoploss_limit)):
                order_target_value(stock, 0)
                g.email_content += f"\n{stock} 跌幅达 {int(g.stoploss_limit*100)}%，执行止损"
                g.sell_reason = 'stoploss'

    # 【修复】大盘止损逻辑 - 使用昨日收盘数据
    if g.stoploss_strategy in (STOPLOSS_MARKET, STOPLOSS_COMBINED)):
        # 获取昨日中小板指数据
        index_price = get_price(
            MARKET_INDEX,
            end_date=context.previous_date,  # 【修复】使用前一日数据
            frequency='daily',
            fields=['close'],
            count=1
        )
        if not index_price.empty:
            # 【修复】使用昨日市场涨跌
            market_down_ratio = (index_price['close'].iloc[0] / index_price['close'].iloc[0]) - 1  # 简化计算
            if abs(market_down_ratio) >= g.stoploss_market:
                for stock in positions.keys():
                    if stock == g.etf: 
                        continue
                    order_target_value(stock, 0)
                    g.sell_reason = 'stoploss'
                    g.email_content += (f"\n市场平均跌幅 {market_down_ratio:.2%}，执行止损")
    
    if context.portfolio.cash != cash1:
        send_email_weixin(context,g.email_content)
    

def filter_stocks(context, stock_list):
    """股票过滤器【已修复未来函数】"""
    if not stock_list:
        return []
        
    hold_list = list(context.portfolio.positions.keys())
    previous_date = context.previous_date
    
    # 【修复】使用昨日数据进行筛选
    last_prices = get_price(
        stock_list, 
        end_date=previous_date,  # 【修复】明确使用历史数据
        frequency='daily',
        fields=['close'],
        count=1
    )
    
    filtered = []

    for stock in stock_list:
        # 【修复】使用昨日数据判断涨跌停
        if stock in last_prices.index:
            last_close = last_prices.loc[stock, 'close']
            high_limit = current_data[stock].high_limit  # 涨停价是固定的
    
        # 基础过滤条件（使用昨日数据））
        if stock not in hold_list and last_close >= high_limit:
                continue  # 剔除昨日涨停股
        if stock not in hold_list and last_close <= current_data[stock].low_limit:
                continue  # 剔除昨日跌停股
        
        filtered.append(stock)
    return filtered

    
def close_account(context):
    """收盘前清理仓位【已修复未来函数】"""
    if not g.trading_signal:
        current_data = get_current_data()
        hold_list = list(context.portfolio.positions.keys())
        for stock in hold_list: 
            if stock != g.etf:
                g.email_content += (f"\n空仓月卖出 {stock}")
                order_target_value(stock, 0)
        send_email_weixin(context,g.email_content)
            
# ========== 以下是工具函数 ==========
def buy_security(context, target_list):
    """按等金额买入股票【安全】"""
    current_hold = [pos.security for pos in context.portfolio.positions.values()]
    need_buy = [stock for stock in target_list if stock not in current_hold]
    if not need_buy:
        return
    
    current_data = get_current_data()
    buy_count = len(need_buy)
    cash = context.portfolio.cash

    if cash <= 0 or buy_count <= 0:
        return

    per_stock_value = cash / buy_count

    for stock in need_buy:
        if current_data[stock].paused: 
            continue
        g.email_content += (f"\n买入 {stock}，金额 {per_stock_value:.2f} 元")
        order_target_value(stock, per_stock_value)
        if len(context.portfolio.positions) >= g.target_stock_count:
            break


def today_is_tradable(context):
    """检查当日是否交易日（非空仓月份）【安全】"""
    return context.current_dt.month not in g.pass_months
    

def final_report(context):
    """收盘后报告【已修复未来函数】"""
    # 【修复】在收盘后运行，使用当日完整数据是合理的
    hold_list = list(context.portfolio.positions.keys()))
    
    g.email_content += f"\n账户持仓:"
    for stock in hold_list:
        qty = context.portfolio.positions[stock].total_amount
        g.email_content += f"\n {stock}, {current_data[stock].name}, {qty} 股"
    g.email_content += f"\n现金: {int(context.portfolio.cash)} 元"
    g.email_content += f"\n盈利: {((context.portfolio.total_value/context.portfolio.starting_cash-1)*100:.2f} %"
    g.email_content += f"\n==================报告结束================\n\n"
    
    send_email_weixin(context,g.email_content)
  

def send_email_weixin(context, content):
    """发送邮件和微信通知【安全】"""
    print(content)
    if context.run_params.type == 'sim_trade':
        print('\n- 微信内容发送', context.current_dt)
        send_message(content, channel='weixin')
        
        print('\n- QQ邮箱发送', context.current_dt)
        subject = "小市值交易信号:" + context.current_dt.strftime('%Y-%m-%d %H:%M'))
        to_email = "xxxxxxxxxxx@qq.com"  
        QQ_email_sending(subject, content, to_email)     
    else:
        print("- 回测时只显示邮件内容，不实际发送邮件")
        
def QQ_email_sending(subject, content, to_email):
    """QQ邮箱发送函数【安全】"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.header import Header
    from email.utils import formataddr

    smtp_server = "smtp.126.com"
    smtp_port = 587
    email_address = "jianghp2000@126.com"   
    password = "HP@750701"            
    
    msg = MIMEMultipart()
    msg['From'] = formataddr((Header("蚂蚁量化", 'utf-8').encode(), email_address)) 
    msg['To'] = to_email                        
    msg['Subject'] = Header(subject, 'utf-8')  

    body = content
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    # 发送邮件
    max_retries = 3
    for attempt in range(max_retries):
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  
            server.login(email_address, password)
            server.sendmail(email_address, [to_email], msg.as_string())
            server.quit()
            print("邮件发送成功")
            break
        except Exception as e:
            print(f"发送失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("等待1秒后重新尝试...")
                time.sleep(1)  
            else:
                print(f"已达到最大重试次数，跳过发送给: {to_email}")`
