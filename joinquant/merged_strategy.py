
# 导入聚宽量化平台相关库
import jqdata
import pandas as pd
import numpy as np
import datetime
from jqfactor import *

def initialize(context):
    """
    初始化函数，设定基准、股票池、参数等
    """
    # 设定中证500作为基准（结合两个策略的基准）
    g.benchmark = '000905.XSHG'
    set_benchmark(g.benchmark)
    
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 设置交易成本
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003, close_today_commission=0, min_commission=5),type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    
    # 全局变量
    g.stock_pool = []  # 股票池
    g.selected_stocks = []  # 选出的股票
    g.hold_days = {}  # 持仓天数记录
    g.hold_list = [] # 当前持仓的全部股票    
    g.yesterday_HL_list = [] # 记录持仓中昨日涨停的股票
    g.no_trading_today_signal = False
    
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
    g.stock_num = 1             # 每组因子选择的股票数量
    
    # 因子参数（来自trade-145.py）
    g.factor_list = [
        (#ARBR-SGAI-NPtTORttm-RPps
            [
                'ARBR', #情绪类因子 ARBR
                'SGAI', #质量类因子 销售管理费用指数
                'net_profit_to_total_operate_revenue_ttm', #质量类因子 净利润与营业总收入之比
                'retained_profit_per_share' #每股指标因子 每股未分配利润
            ],
            [
                -2.3425,
                -694.7936,
                -170.0463,
                -1362.5762
            ]
        ),
        (#P1Y-TPtCR-VOL120
            [
                'Price1Y', #动量类因子 当前股价除以过去一年股价均值再减1
                'total_profit_to_cost_ratio', #质量类因子 成本费用利润率
                'VOL120' #情绪类因子 120日平均换手率
            ],
            [
                -0.0647128120839873,
                -0.006385116279168804,
                -0.0029867925845833217
            ]
        ),
        (#PNF-TPtCR-ITR
            [
                'price_no_fq', #技术指标因子 不复权价格因子
                'total_profit_to_cost_ratio', #质量类因子 成本费用利润率
                'inventory_turnover_rate' #质量类因子 存货周转率
            ],
            [
                -6.123355346008858e-05,
                -0.002579342458393642,
                -2.194257357346814e-06
            ]
        ),
        (#DtA-OCtORR-DAVOL20-PNF-SG
            [
                'debt_to_assets', #风格因子 资产负债率
                'operating_cost_to_operating_revenue_ratio', #质量类因子 销售成本率
                'DAVOL20', #情绪类因子 20日平均换手率与120日平均换手率之比
                'price_no_fq', #技术指标因子 不复权价格因子
                'sales_growth' #风格因子 5年营业收入增长率
            ],
            [
                0.04477354820057883,
                0.021636407482421707,
                -0.01864268317469762,
                -0.0004678118383947827,
                0.02884867440332058
            ]
        ),
        (#TVSTD6-CFpsttm-SR120-NONPttm
            [
                'TVSTD6', #情绪类因子 6日成交金额的标准差
                'cashflow_per_share_ttm', #每股指标因子 每股现金流量净额
                'sharpe_ratio_120', #风险类因子 120日夏普率
                'non_operating_net_profit_ttm' #基础科目及衍生类因子 营业外收支净额TTM
            ],
            [
                -5.394060941494863e-12,
                4.6306072704138405e-05,
                -0.0030567075906980912,
                1.4227113275455325e-12
            ]
        )
    ]
    
    # 设置每1分钟运行一次（实时交易）
    run_daily(trade, time='every_bar')
    # 设置整点运行（整数时刻交易）
    run_daily(integer_hour_trade, time='9:30')
    run_daily(integer_hour_trade, time='10:30')
    run_daily(integer_hour_trade, time='11:30')
    run_daily(integer_hour_trade, time='13:30')
    run_daily(integer_hour_trade, time='14:30')
    # 每日准备股票池
    run_daily(prepare_stock_list, '9:05')
    # 每周调仓
    run_weekly(weekly_adjustment, 1, '9:30')
    # 检查涨停股
    run_daily(check_limit_up, '14:00')
    # 收盘后运行
    run_daily(close_account, '14:30')
    run_daily(print_position_info, '15:10')
    
    # 记录日志
    log.info("合并策略初始化完成")

def prepare_stock_list(context):
    """
    准备股票池
    """
    # 获取已持有列表
    g.hold_list = []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)
    # 获取昨日涨停列表
    if g.hold_list != []:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily', fields=['close','high_limit'], count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)
    else:
        g.yesterday_HL_list = []
    # 判断今天是否为账户资金再平衡的日期
    g.no_trading_today_signal = today_is_between(context, '04-05', '04-30')

def get_stock_list(context):
    """
    选股模块（基于trade-145.py的因子选股）
    """
    # 指定日期防止未来数据
    yesterday = context.previous_date
    # 获取初始列表
    initial_list = get_all_securities('stock', yesterday).index.tolist()
    # Filter new stocks using context to prevent future data issues
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_stock(initial_list)
    initial_list = filter_new_stock(initial_list, context)
    final_list = []
    # MS
    for factor_list, coef_list in g.factor_list:
        factor_values = get_factor_values(initial_list, factor_list, end_date=yesterday, count=1)
        df = pd.DataFrame(index=initial_list, columns=factor_values.keys())
        for i in range(len(factor_list)):
            df[factor_list[i]] = list(factor_values[factor_list[i]].T.iloc[:,0])
        df = df.dropna()
        df['total_score'] = 0
        for i in range(len(factor_list)):
            df['total_score'] += coef_list[i] * df[factor_list[i]]
        df = df.sort_values(by=['total_score'], ascending=False)  # 分数越高即预测未来收益越高，排序默认降序
        complex_factor_list = list(df.index)[:int(0.1 * len(list(df.index)))]
        q = query(valuation.code, valuation.circulating_market_cap, indicator.eps).filter(valuation.code.in_(complex_factor_list)).order_by(valuation.circulating_market_cap.asc())
        df = get_fundamentals(q)
        df = df[df['eps'] > 0]
        lst = list(df.code)
        lst = filter_paused_stock(lst)
        lst = filter_limitup_stock(context, lst)
        lst = filter_limitdown_stock(context, lst)
        lst = lst[:min(g.stock_num, len(lst))]
        for stock in lst:
            if stock not in final_list:
                final_list.append(stock)
    return final_list

def filter_stocks(context):
    """
    第一步：筛选满足均线条件的股票（来自avg120_kd.py）
    条件：120日均线在240日均线之上，且两条均线都向上
    """
    all_stocks = get_index_stocks('000905.XSHG')  # 获取中证500成分股
    
    filtered_stocks = []
    
    for stock in all_stocks:
        try:
            # 获取股票基本信息，剔除ST股和次新股
            stock_info = get_security_info(stock)
            if stock_info.display_name.find('ST') != -1 or stock_info.display_name.find('*ST') != -1:
                continue
            
            # 计算上市天数，剔除上市不足60天的次新股（约3个月）
            list_days = (context.previous_date - stock_info.start_date).days
            if list_days < 60:
                continue
            
            # 获取历史数据计算均线
            hist_data = get_price(stock, count=g.ma_long+20, end_date=context.previous_date,
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
    第二步：用周线KD金叉进一步筛选股票（来自avg120_kd.py）
    """
    kd_filtered_stocks = []
    
    for stock in stocks:
        try:
            # 获取周线数据计算KD指标
            weekly_data = get_price(stock, count=g.kd_period*3, end_date=context.previous_date,
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
    第三步：检查日线KD金叉买入信号（来自avg120_kd.py）
    """
    try:
        # 获取日线数据计算KD指标
        daily_data = get_price(stock, count=g.kd_period*3, end_date=context.previous_date,
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
    检查买卖时机规则（来自avg120_kd.py）
    """
    try:
        current_data = get_current_data()
        
        # 规则4：低开超过3个点，15分钟内不翻红，不下单
        today_data = get_price(stock, count=1, end_date=context.previous_date,
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
    风险管理：止损止盈检查（来自avg120_kd.py）
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
    # 使用前一日数据避免未来函数问题
    hist_data = get_price(stock, count=4, end_date=context.previous_date,
                         frequency='1d', fields=['close'])
    
    if len(hist_data) >= 4:
        closes = hist_data['close']
        if all(closes.iloc[i] > closes.iloc[i-1] for i in range(1, 4)):
            return 'reduce'
    
    return 'hold'

def get_current_position_count(context):
    """
    获取当前持仓股票数量（来自avg120_kd.py）
    """
    return len([stock for stock in context.portfolio.positions.keys() 
                if context.portfolio.positions[stock].total_amount > 0])

def get_available_position_slots(context):
    """
    获取可用的持仓位置数量（来自avg120_kd.py）
    """
    current_positions = get_current_position_count(context)
    return max(0, g.max_positions - current_positions)

def trade(context):
    """
    主交易函数，每分钟运行一次（来自avg120_kd.py，但结合了风险控制）
    """
    # 只在交易时间运行
    if not is_trading_time(context):
        return
    
    # 处理现有持仓的风险管理
    for stock in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[stock]
        if position.total_amount <= 0:
            continue
            
        action = risk_management(context, stock)
        
        if action == 'sell':
            # 清仓卖出
            order_target_value(stock, 0)
            log.info("执行止损/止盈，卖出股票: {}".format(stock))
            
        elif action == 'reduce':
            # 减仓一半
            current_value = position.value
            order_target_value(stock, current_value * 0.5)
            log.info("连涨三天减仓一半: {}".format(stock))
    
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

def integer_hour_trade(context):
    """
    整点交易函数（来自trade-145.py的调仓逻辑）
    """
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

def weekly_adjustment(context):
    """
    每周调仓函数（来自trade-145.py）
    """
    if g.no_trading_today_signal == False:
        # 获取应买入列表
        log.info("call get_stock_list")
        target_list = get_stock_list(context)
        # 调仓卖出
        for stock in g.hold_list:
            if (stock not in target_list) and (stock not in g.yesterday_HL_list):
                log.info("卖出[%s]" % (stock))
                position = safe_get_position(context, stock)
                if position:
                    close_position(position)
                else:
                    log.info("股票[%s]已不在持仓中" % (stock))
            else:
                log.info("已持有[%s]" % (stock))
        # 调仓买入
        # 计算当前持仓数量（只计算有实际持仓的）
        position_count = len([stock for stock in context.portfolio.positions.keys()
                                if context.portfolio.positions[stock].total_amount > 0])
        
        target_num = len(target_list)
        # 确保target_num > position_count且有可用资金
        if target_num > position_count and context.portfolio.cash > 0:
            # 计算每只股票的买入金额
            # 需要买入的股票数量
            need_to_buy = target_num - position_count
            if need_to_buy > 0:
                value = context.portfolio.cash / need_to_buy
                # 买入目标股票
                bought_count = 0
                for stock in target_list:
                    # 检查是否已经持有该股票
                    position = safe_get_position(context, stock)
                    # 如果未持有或持仓为0，则买入
                    if not position or position.total_amount == 0:
                        if open_position(stock, value):
                            bought_count += 1
                            # 如果已达到需要买入的数量，则停止买入
                            if bought_count >= need_to_buy:
                                break

def check_limit_up(context):
    """
    检查涨停股（来自trade-145.py）
    """
    now_time = context.current_dt
    if g.yesterday_HL_list != []:
        # 对昨日涨停股票观察到尾盘如不涨停则提前卖出，如果涨停即使不在应买入列表仍暂时持有
        for stock in g.yesterday_HL_list:
            # 修复bug：检查股票是否仍在持仓中
            if stock not in context.portfolio.positions:
                continue
            current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close','high_limit'], skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
            if current_data.iloc[0,0] < current_data.iloc[0,1]:
                log.info("[%s]涨停打开，卖出" % (stock))
                position = context.portfolio.positions[stock]
                close_position(position)
            else:
                log.info("[%s]涨停，继续持有" % (stock))

def close_account(context):
    """
    清仓函数（来自trade-145.py）
    """
    if g.no_trading_today_signal == True:
        if len(g.hold_list) != 0:
            for stock in g.hold_list:
                # 修复bug：检查股票是否在持仓中
                if stock in context.portfolio.positions:
                    position = context.portfolio.positions[stock]
                    close_position(position)
                    log.info("卖出[%s]" % (stock))

def print_position_info(context):
    """
    打印持仓信息（来自trade-145.py）
    """
    # 打印当天成交记录
    trades = get_trades()
    for _trade in trades.values():
        print('成交记录：'+str(_trade))
    # 打印账户信息
    for position in list(context.portfolio.positions.values()):
        securities=position.security
        cost=position.avg_cost
        price=position.price
        ret=100*(price/cost-1)
        value=position.value
        amount=position.total_amount
        print('代码:{}'.format(securities))
        print('成本价:{}'.format(format(cost,'.2f')))
        print('现价:{}'.format(price))
        print('收益率:{}%'.format(format(ret,'.2f')))
        print('持仓(股):{}'.format(amount))
        print('市值:{}'.format(format(value,'.2f')))
        print('———————————————————————————————————')
    print('———————————————————————————————————————分割线————————————————————————————————————————')

# 辅助函数（来自trade-145.py）
def safe_get_position(context, stock):
    """安全获取持仓信息"""
    if stock in context.portfolio.positions:
        return context.portfolio.positions[stock]
    else:
        return None

def filter_paused_stock(stock_list):
    """过滤停牌股票"""
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]

def filter_st_stock(stock_list):
    """过滤ST及其他具有退市标签的股票"""
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st
            and 'ST' not in current_data[stock].name
            and '*' not in current_data[stock].name
            and '退' not in current_data[stock].name]

def filter_kcbj_stock(stock_list):
    """过滤科创北交股票"""
    for stock in stock_list[:]:
        if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68':
            stock_list.remove(stock)
    return stock_list

def filter_limitup_stock(context, stock_list):
    """过滤涨停的股票"""
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] <    current_data[stock].high_limit]

def filter_limitdown_stock(context, stock_list):
    """过滤跌停的股票"""
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] > current_data[stock].low_limit]

def filter_new_stock(stock_list, context):
    """过滤次新股"""
    # 过滤上市不足60天的次新股（约3个月）
    filtered_list = []
    for stock in stock_list:
        try:
            stock_info = get_security_info(stock)
            list_days = (context.previous_date - stock_info.start_date).days
            if list_days >= 60:
                filtered_list.append(stock)
        except Exception as e:
            log.warn("获取股票{}信息时出错: {}".format(stock, str(e)))
            continue
    return filtered_list

def order_target_value_(security, value):
    """自定义下单"""
    if value == 0:
        log.debug("Selling out %s" % (security))
    else:
        log.debug("Order %s to value %f" % (security, value))
    return order_target_value(security, value)

def open_position(security, value):
    """开仓"""
    order = order_target_value_(security, value)
    if order != None and order.filled > 0:
        return True
    return False

def close_position(position):
    """平仓"""
    security = position.security
    order = order_target_value_(security, 0)  # 可能会因停牌失败
    if order != None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False

def today_is_between(context, start_date, end_date):
    """判断今天是否为账户资金再平衡的日期"""
    today = context.current_dt.strftime('%m-%d')
    if (start_date <= today) and (today <= end_date):
        return True
    else:
        return False

def is_trading_time(context):
    """
    检查是否为交易时间（来自avg120_kd.py）
    """
    current_time = context.current_dt.time()
    
    # A股交易时间：上午9:30-11:30，下午13:00-15:00
    morning_start = pd.Timestamp('09:30:00').time()
    morning_end = pd.Timestamp('11:30:00').time()
    afternoon_start = pd.Timestamp('13:00:00').time()
    afternoon_end = pd.Timestamp('15:00:00').time()
    
    return ((morning_start <= current_time <= morning_end) or
            (afternoon_start <= current_time <= afternoon_end))

# 策略说明：
# 本策略结合了两个策略的优点：
# 1. 使用avg120_kd.py的实时交易算法作为主要交易逻辑，每分钟运行一次
# 2. 使用trade-145.py的整点调仓逻辑，定期重新选股
# 3. 结合了avg120_kd.py的风险控制机制进行实时交易修正
# 4. 使用trade-145.py的因子选股模型进行股票筛选