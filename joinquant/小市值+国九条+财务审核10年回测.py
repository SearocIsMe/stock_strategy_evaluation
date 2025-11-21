# 克隆自聚宽文章：https://www.joinquant.com/post/63777
# 标题：小市值+国九条+财务审核10年回测，年化89%，回撤25%
# 作者：朵朵2017

# 克隆自聚宽文章https://www.joinquant.com/view/community/detail/f355361860cbdb761d4f99766d974ca0
# 标题：二次优化—厉害了小市值年化115回撤32
# 作者：红杉树2025


# 导入函数库
from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
# 修改导入方式，避免命名冲突
import datetime
# 移除了不存在的 OrderStatus 导入

# 初始化函数
def initialize(context):
    g.signal = ''
    # 开启防未来函数
    set_option('avoid_future_data', True)
    # 设定基准
    set_benchmark('399101.XSHE')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(3/10000))
    # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=1/10000, close_commission=1/10000, close_today_commission=0, min_commission=5), type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'debug')

    # 初始化全局变量 bool
    g.no_trading_today_signal = False  # 是否为可交易日
    g.pass_april = True  # 是否四月空仓
    g.run_stoploss = True  # 是否进行止损
    g.filter_audit = True  # 是否筛选审计意见
    
    # 全局变量list
    g.hold_list = []  # 当前持仓的全部股票
    g.yesterday_HL_list = []  # 记录持仓中昨日涨停的股票
    g.target_list = []
    g.not_buy_again = []

    # 全局变量
    g.stock_num = 3
    g.min_mv = 5
    g.max_mv = 50    
    g.up_price = 100  # 设置股票单价
    g.reason_to_sell = ''
    g.stoploss_strategy = 3  # 1为止损线止损，2为市场趋势止损, 3为联合1、2策略
    g.stoploss_limit = 0.91  # 止损线
    g.stoploss_market = 0.95  # 市场趋势止损参数

    g.HV_control = True  # 新增，Ture是日频判断是否放量，False则不然
    g.HV_duration = 120  # HV_control用，周期可以是240-120-60，默认比例是0.9
    g.HV_ratio = 0.9  # HV_control用
    g.stockL = []
    g.no_trading_buy = ['600036.XSHG', '518880.XSHG', '600900.XSHG', '511010.XSHG', '513100.XSHG', '512820.XSHG']  # 空仓月份持有
    g.no_trading_hold_signal = False
    g.filter_audit = True  # 是否筛选审计意见 

    # 设置交易运行时间
    run_daily(prepare_stock_list, '9:05')
    run_weekly(weekly_adjustment, 2, '10:30')
    run_daily(sell_stocks, time='10:00')  # 止损函数
    run_daily(trade_afternoon, time='14:25')  # 检查持仓中的涨停股是否需要卖出
    run_daily(trade_afternoon, time='14:55')  # 检查持仓中的涨停股是否需要卖出
    run_daily(close_account, '14:50')
    run_weekly(print_position_info, 5, time='15:10')


# 1-1 准备股票池
def prepare_stock_list(context):
    # 获取已持有列表
    g.hold_list = []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)

    # 获取昨日涨停列表
    if g.hold_list != []:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily', fields=['close', 'high_limit', 'low_limit'], count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)
    else:
        g.yesterday_HL_list = []

    # 判断今天是否为账户资金再平衡的日期
    g.no_trading_today_signal = today_is_between(context)


# 1-2 选股模块
def get_stock_list(context):
    final_list = []
    MKT_index = '399101.XSHE'
    initial_list = get_index_stocks(MKT_index)
    initial_list = filter_new_stock(context, initial_list)   # 过滤次新股
    initial_list = filter_kcbj_stock(initial_list)           # 过滤科创北交
    initial_list = filter_st_stock(initial_list)             # 过滤ST股
    initial_list = filter_paused_stock(initial_list)         # 过滤停牌股
    
    """ 
    # 按照流通市值升序排序，取前200
    q = query(valuation.code).filter(valuation.code.in_(initial_list)).order_by(valuation.circulating_market_cap.asc()).limit(210)
    sorted_list = list(get_fundamentals(q).code)
    
    """
        # 按照流通市值升序排序，取前200
    q = query(valuation.code,valuation.circulating_market_cap,  # 总市值 circulating_market_cap/market_cap
        income.np_parent_company_owners,  # 归属于母公司所有者的净利润
        income.net_profit,  # 净利润
        income.operating_revenue  # 营业收入
    ).filter(valuation.code.in_(initial_list),
        valuation.market_cap.between(g.min_mv,g.max_mv),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8,
        indicator.roe>0,
        indicator.roa>0,
    ).order_by(valuation.market_cap.asc()).limit(210)
    sorted_list = list(get_fundamentals(q).code)

    # 剔除前10，只保留第11到200（即后200）
    initial_list = sorted_list[0:]   

    # 过滤涨停和跌停股   
    initial_list1 = filter_limitup_stock(context, initial_list)
    initial_list1 = filter_limitdown_stock(context, initial_list)
    
    # 过滤审计意见
    if g.filter_audit == True:
        initial_list = filter_audit(context,initial_list)    
    
    # 再按总市值升序排列
    q = query(valuation.code, indicator.eps).filter(valuation.code.in_(initial_list1)).order_by(valuation.market_cap.asc())
    df = get_fundamentals(q)
    stock_list = list(df.code)
    final_list = list(df.code)

    stock_list = stock_list[0:100]                # 保留前100
    stock_list = get_stock_industry(stock_list)  # 加入行业信息（如有定义）
    final_list = stock_list[:g.stock_num * 2]
    log.info('今日前10:%s' % final_list)
    return final_list
    
#2.1 筛选审计意见
'''
审计意见类型编码
类型编码 审计意见类型
1 	     无保留
2 	     无保留带解释性说明
3        保留意见
4        拒绝/无法表示意见
5        否定意见
6 	     未经审计
7 	     保留带解释性说明
10 	     经审计（不确定具体意见类型）
11       无保留带持续经营重大不确定性
'''

def filter_audit(context,initial_list):
    # 获取审计意见，近三年内如果有不合格(report_type为3、4、5、7)的审计意见则返回False，否则返回True
    final_list = []
    expection_Audit_list = []
    for stock in initial_list:
        lstd = context.previous_date
        last_year = (lstd.replace(year=lstd.year - 3, month=1, day=1)).strftime('%Y-%m-%d')
        q=query(finance.STK_AUDIT_OPINION.code,finance.STK_AUDIT_OPINION.pub_date,finance.STK_AUDIT_OPINION).filter(
                                finance.STK_AUDIT_OPINION.code==stock,
                                finance.STK_AUDIT_OPINION.pub_date>=last_year,
                                finance.STK_AUDIT_OPINION.pub_date<=context.current_dt,
                                )
        df=finance.run_query(q)
        # print('\n%s'%df)
        values_to_check = [3, 4, 5, 7]
        contains_unwanted_values = df['opinion_type_id'].isin(values_to_check).any()
        if not contains_unwanted_values:
            final_list.append(stock)
        else:
            expection_Audit_list.append(stock)
    print('★★★★ 去除近三年内存在审计问题的%s只 ★★★★'%(len(expection_Audit_list)))
    print('★★★★ 存在审计问题的: %s  '%(expection_Audit_list))
    return  final_list  # 返回剔除审计意见异常后的list
    
"""
#2.1 筛选审计意见
def filter_audit(context, code):
    # 获取审计意见，近三年内如果有不合格(report_type为2、3、4、5)的审计意见则返回False，否则返回True
    lstd = context.previous_date
    last_year = (lstd.replace(year=lstd.year - 3, month=1, day=1)).strftime('%Y-%m-%d')
    q=query(finance.STK_AUDIT_OPINION).filter(finance.STK_AUDIT_OPINION.code==code,finance.STK_AUDIT_OPINION.pub_date>=last_year)
    df=finance.run_query(q)
    df['report_type'] = df['report_type'].astype(str)
    contains_nums = df['report_type'].str.contains(r'2|3|4|5')
    return not contains_nums.any()    
"""

# 1-3 整体调整持仓
def weekly_adjustment(context):
    if g.no_trading_today_signal == False:
        close_no_trading_hold(context)

        # 获取应买入列表
        g.not_buy_again = []
        g.target_list = get_stock_list(context)
        target_list = g.target_list[:g.stock_num * 2]
        log.info(str(target_list))

        # 新增：记录当次调仓卖出的股票
        sold_stocks = []
        # 调仓卖出
        for stock in g.hold_list:
            if (stock not in target_list) and (stock not in g.yesterday_HL_list):
                log.info("卖出[%s]" % (stock))
                position = context.portfolio.positions[stock]
                close_position(position)
                sold_stocks.append(stock)  # 记录卖出的股票
        
        # 打印卖出的股票列表，用于调试
        log.info("本次调仓卖出的股票: %s" % sold_stocks)

        # 调仓买入：剔除刚卖出的股票和已持有的股票
        adjusted_target_list = [
            stock for stock in target_list
            if stock not in sold_stocks and stock not in context.portfolio.positions.keys()
        ]
        
        # 打印调整后的买入列表，用于调试
        log.info("调整后的买入列表: %s" % adjusted_target_list)

        buy_security(context, adjusted_target_list)  # 传入过滤后的列表

        # 记录已买入股票
        for position in list(context.portfolio.positions.values()):
            stock = position.security
            g.not_buy_again.append(stock)


# 1-4 调整昨日涨停股票
def check_limit_up(context):
    now_time = context.current_dt
    if g.yesterday_HL_list != []:
        # 对昨日涨停股票观察到尾盘如不涨停则提前卖出，如果涨停即使不在应买入列表仍暂时持有
        for stock in g.yesterday_HL_list:
            if context.portfolio.positions[stock].closeable_amount > -100:
                current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close', 'high_limit'], skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
                if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                    log.info("[%s]涨停打开，卖出" % (stock))
                    position = context.portfolio.positions[stock]
                    close_position(position)
                    g.reason_to_sell = 'limitup'
                else:
                    log.info("[%s]涨停，继续持有" % (stock))


# 1-5 如果昨天有股票卖出或者买入失败，剩余的金额今天早上买入
def check_remain_amount(context):
    if g.reason_to_sell == 'limitup':  # 判断提前售出原因，如果是涨停售出则次日再次交易，如果是止损售出则不交易
        g.hold_list = []
        for position in list(context.portfolio.positions.values()):
            stock = position.security
            g.hold_list.append(stock)
        if len(g.hold_list) < g.stock_num:
            target_list = get_stock_list(context)
            # 剔除本周一曾买入的股票，不再买入
            target_list = filter_not_buy_again(target_list)
            target_list = target_list[:min(g.stock_num, len(target_list))]
            log.info('有余额可用' + str(round((context.portfolio.cash), 2)) + '元。' + str(target_list))
            buy_security(context, target_list)
            g.reason_to_sell = ''
    else:
        log.info('虽然有余额（' + str(round((context.portfolio.cash), 2)) + '元）可用，但是为止损后余额，下周再交易')
        g.reason_to_sell = ''


# 1-6 下午检查交易
def trade_afternoon(context):
    if g.no_trading_today_signal == False:
        check_limit_up(context)
        if g.HV_control == True:
            check_high_volume(context)
        huanshou(context)
        check_remain_amount(context)


# 1-7 止盈止损
def sell_stocks(context):
    if g.run_stoploss == True:
        if g.stoploss_strategy == 1:
            for stock in context.portfolio.positions.keys():
                # 股票盈利大于等于100%则卖出
                if context.portfolio.positions[stock].price >= context.portfolio.positions[stock].avg_cost * 2:
                    order_target_value(stock, 0)
                    log.debug("收益100%止盈,卖出{}".format(stock))
                # 止损
                elif context.portfolio.positions[stock].price < context.portfolio.positions[stock].avg_cost * g.stoploss_limit:
                    order_target_value(stock, 0)
                    log.debug("收益止损,卖出{}".format(stock))
                    g.reason_to_sell = 'stoploss'
        elif g.stoploss_strategy == 2:
            stock_df = get_price(security=get_index_stocks('399101.XSHE'), end_date=context.previous_date, frequency='daily', fields=['close', 'open'], count=1, panel=False)
            down_ratio = (stock_df['close'] / stock_df['open']).mean()
            if down_ratio < g.stoploss_market:
                g.reason_to_sell = 'stoploss'
                log.debug("大盘惨跌,平均降幅{:.2%}".format(1 - down_ratio))
                for stock in context.portfolio.positions.keys():
                    order_target_value(stock, 0)
        elif g.stoploss_strategy == 3:
            stock_df = get_price(security=get_index_stocks('399101.XSHE'), end_date=context.previous_date, frequency='daily', fields=['close', 'open'], count=1, panel=False)
            down_ratio = (stock_df['close'] / stock_df['open']).mean()
            if down_ratio < g.stoploss_market:
                g.reason_to_sell = 'stoploss'
                log.debug("大盘惨跌,平均降幅{:.2%}".format(1 - down_ratio))
                for stock in context.portfolio.positions.keys():
                    order_target_value(stock, 0)
            else:
                for stock in context.portfolio.positions.keys():
                    if context.portfolio.positions[stock].price < context.portfolio.positions[stock].avg_cost * g.stoploss_limit:
                        order_target_value(stock, 0)
                        log.debug("收益止损,卖出{}".format(stock))
                        g.reason_to_sell = 'stoploss'


# 3-2 调整放量股票
def check_high_volume(context):
    current_data = get_current_data()
    for stock in context.portfolio.positions:
        if current_data[stock].paused == True:
            continue
        if current_data[stock].last_price == current_data[stock].high_limit:
            continue
        if context.portfolio.positions[stock].closeable_amount == 0:
            continue
        df_volume = get_bars(stock, count=g.HV_duration, unit='1d', fields=['volume'], include_now=True, df=True)
        if df_volume['volume'].values[-1] > g.HV_ratio * df_volume['volume'].values.max():
            log.info("[%s]天量，卖出" % stock)
            position = context.portfolio.positions[stock]
            close_position(position)


# 2-1 过滤停牌股票
def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]


# 2-2 过滤ST及其他具有退市标签的股票
def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st
            and 'ST' not in current_data[stock].name
            and '*' not in current_data[stock].name
            and '退' not in current_data[stock].name]


# 2-3 过滤科创北交股票
def filter_kcbj_stock(stock_list):
    for stock in stock_list[:]:
        if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68' or stock[:2] == '30':
            stock_list.remove(stock)
    return stock_list


# 2-4 过滤涨停的股票
def filter_limitup_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] < current_data[stock].high_limit]


# 2-5 过滤跌停的股票
def filter_limitdown_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if (stock in context.portfolio.positions.keys()
                                              or last_prices[stock][-1] > current_data[stock].low_limit)]


# 2-6 过滤次新股
def filter_new_stock(context, stock_list):
    yesterday = context.previous_date
    # 使用 datetime.timedelta 替代原有的错误写法
    return [stock for stock in stock_list if not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=375)]


# 2-6.5 过滤股价
def filter_highprice_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] <= g.up_price]


# 2-7 删除本周一买入的股票
def filter_not_buy_again(stock_list):
    return [stock for stock in stock_list if stock not in g.not_buy_again]


# 获取股票所属行业
def get_stock_industry(stock):
    result = get_industry(security=stock)
    selected_stocks = []
    industry_list = []

    for stock_code, info in result.items():
        industry_name = info['sw_l2']['industry_name']
        if industry_name not in industry_list:
            industry_list.append(industry_name)
            selected_stocks.append(stock_code)
            print(f"行业信息: {industry_name} (股票: {stock_code})")

        # 选取了 10 个不同行业的股票
        if len(industry_list) == 10:
            break
    return selected_stocks
    

# 换手率计算
def huanshoulv(context, stock, is_avg=False):
    if is_avg:
        # 计算平均换手率
        # 使用 datetime.timedelta 替代原有的错误写法
        start_date = context.current_dt - datetime.timedelta(days=20)
        end_date = context.previous_date
        df_volume = get_price(stock, end_date=end_date, frequency='daily', fields=['volume'], count=20)
        df_cap = get_valuation(stock, end_date=end_date, fields=['circulating_cap'], count=1)
        circulating_cap = df_cap['circulating_cap'].iloc[0] if not df_cap.empty else 0
        if circulating_cap == 0:
            return 0.0
        df_volume['turnover_ratio'] = df_volume['volume'] / (circulating_cap * 10000)
        return df_volume['turnover_ratio'].mean()
    else:
        # 计算实时换手率
        date_now = context.current_dt
        df_vol = get_price(stock, start_date=date_now.date(), end_date=date_now, frequency='1m', fields=['volume'],
                           skip_paused=False, fq='pre', panel=True, fill_paused=False)
        volume = df_vol['volume'].sum()
        date_pre = context.previous_date
        df_circulating_cap = get_valuation(stock, end_date=date_pre, fields=['circulating_cap'], count=1)
        circulating_cap = df_circulating_cap['circulating_cap'].iloc[0] if not df_circulating_cap.empty else 0
        if circulating_cap == 0:
            return 0.0
        turnover_ratio = volume / (circulating_cap * 10000)
        return turnover_ratio


# 换手检测
def huanshou(context):
    shrink, expand = 0.003, 0.1
    current_data = get_current_data()
    for stock in context.portfolio.positions:
        if current_data[stock].paused == True:
            continue
        if current_data[stock].last_price >= current_data[stock].high_limit * 0.97:
            continue
        if context.portfolio.positions[stock].closeable_amount == 0:
            continue
        rt = huanshoulv(context, stock, False)
        avg = huanshoulv(context, stock, True)
        if avg == 0:
            continue
        r = rt / avg
        action, icon = '', ''
        if avg < 0.003:
            action, icon = '缩量', '❄️'
        elif rt > expand and r > 2:
            action, icon = '放量', '?'
        if action:
            log.info(f"{action} {stock} {get_security_info(stock).display_name} 换手率:{rt:.2%}→均:{avg:.2%} 倍率:{r:.1f}x {icon}")
            position = context.portfolio.positions[stock]
            close_position(position)
            g.reason_to_sell = 'limitup'


# 3-1 交易模块-自定义下单
def ordertarget_value(security, value):
    if value == 0:
        log.debug("Selling out %s" % (security))
    else:
        log.debug("Order %s to value %f" % (security, value))
    return order_target_value(security, value)


# 3-2 交易模块-开仓
def open_position(security, value):
    order = order_target_value(security, value)
    if order is not None and order.filled > 0:
        return True
    return False


# 3-3 交易模块-平仓
def close_position(position):
    security = position.security
    order = order_target_value(security, 0)  # 可能会因停牌失败
    if order is not None:
        # 修改订单状态判断逻辑，不再使用 OrderStatus
        if order.status == 'filled' and order.filled == order.amount:
            return True
    return False


# 3-4 买入模块
def buy_security(context, target_list, cash=0, buy_number=0):
    # 调仓买入
    position_count = len(context.portfolio.positions)
    target_num = g.stock_num
    if cash == 0:
        cash = context.portfolio.total_value  # cash
    if buy_number == 0:
        buy_number = target_num
    bought_num = 0
    print('---------------------buy_number：%s' % buy_number)
    
    # 计算还可以买入的股票数量
    available_slots = target_num - position_count
    if available_slots <= 0:
        return  # 持仓已满，不执行买入
    
    value = cash / target_num  # 每只股票的买入金额
    remaining_slots = available_slots
    
    for stock in target_list:
        # 跳过已持有的股票
        if context.portfolio.positions.get(stock, None) is not None and context.portfolio.positions[stock].total_amount > 0:
            continue
            
        # 检查是否还有可用仓位
        if remaining_slots <= 0:
            break
            
        if bought_num < buy_number:
            if open_position(stock, value):
                log.info("买入[%s]（%s元）" % (stock, value))
                g.not_buy_again.append(stock)  # 记录已买入的股票
                bought_num += 1
                remaining_slots -= 1  # 减少可用仓位


# 4-1 判断今天是否为四月
def today_is_between(context):
    today = context.current_dt.strftime('%m-%d')
    if g.pass_april is True:
        if (('04-01' <= today) and (today <= '04-30')) or (('01-01' <= today) and (today <= '01-30')):
            return True
        else:
            return False
    else:
        return False


# 4-2 清仓后次日资金可转
def close_account(context):
    if g.no_trading_today_signal == True:
        if len(g.hold_list) != 0 and g.no_trading_hold_signal == False:
            for stock in g.hold_list:
                position = context.portfolio.positions[stock]
                if close_position(position):
                    log.info("卖出[%s]" % (stock))
                else:
                    log.info("卖出[%s]错误！！！！！" % (stock))
            buy_security(context, g.no_trading_buy)
            g.no_trading_hold_signal = True


# 4-3 清仓小市值不交易期间股票
def close_no_trading_hold(context):
    if g.no_trading_hold_signal == True:
        for stock in g.hold_list:
            position = context.portfolio.positions[stock]
            close_position(position)
            log.info("卖出[%s]" % (stock))
        g.no_trading_hold_signal = False


# 1-8 动态调仓代码
def adjust_stock_num(context):
    ma_para = 10  # 设置MA参数
    today = context.previous_date
    # 使用 datetime.timedelta 替代原有的错误写法
    start_date = today - datetime.timedelta(days=ma_para * 2)
    index_df = get_price('399101.XSHE', start_date=start_date, end_date=today, frequency='daily')
    index_df['ma'] = index_df['close'].rolling(window=ma_para).mean()
    last_row = index_df.iloc[-1]
    diff = last_row['close'] - last_row['ma']

    # 根据差值结果返回数字
    result = 3 if diff >= 500 else \
        3 if 200 <= diff < 500 else \
        4 if -200 <= diff < 200 else \
        5 if -500 <= diff < -200 else \
        6
    return result


def print_position_info(context):
    print('———————————————————————————————————')
    for position in list(context.portfolio.positions.values()):
        securities = position.security
        cost = position.avg_cost
        price = position.price
        ret = 100 * (price / cost - 1)
        value = position.value
        amount = position.total_amount
        print('代码:{}'.format(securities))
        print('收益率:{}%'.format(format(ret, '.2f')))
        print('持仓(股):{}'.format(amount))
        print('市值:{}'.format(format(value, '.2f')))
        print('———————————————————————————————————')
    print('余额:{}'.format(format(context.portfolio.cash, '.2f')))
    print('———————————————————————————————————————分割线————————————————————————————————————————')    