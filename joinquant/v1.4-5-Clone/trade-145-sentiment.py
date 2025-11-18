# https://www.joinquant.com/view/community/detail/30684f8d65a74eef0d704239f0eec8be?type=1&page=2
# 导入函数库
from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd

# 初始化函数 
def initialize(context):
    # 设定基准
    set_benchmark('000905.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003, close_today_commission=0, min_commission=5),type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    # 初始化全局变量
    g.no_trading_today_signal = False
    g.stock_num = 1
    g.hold_list = [] # 当前持仓的全部股票    
    g.yesterday_HL_list = [] # 记录持仓中昨日涨停的股票
    g.factor_list = [
        (#ARBR-SGAI-NPtTORttm-RPps
            [
                'ARBR', # 情绪类因子 ARBR
                'SGAI', # 质量类因子 销售管理费用指数
                'net_profit_to_total_operate_revenue_ttm', # 质量类因子 净利润与营业总收入之比
                'retained_profit_per_share' # 每股指标因子 每股未分配利润
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
                'Price1Y', # 动量类因子 当前股价除以过去一年股价均值再减1
                'total_profit_to_cost_ratio', # 质量类因子 成本费用利润率
                'VOL120' # 情绪类因子 120日平均换手率
            ],
            [
                -0.0647128120839873,
                -0.006385116279168804,
                -0.0029867925845833217
            ]
        ),
        (#PNF-TPtCR-ITR
            [
                'price_no_fq', # 技术指标因子 不复权价格因子
                'total_profit_to_cost_ratio', # 质量类因子 成本费用利润率
                'inventory_turnover_rate' # 质量类因子 存货周转率
            ],
            [
                -6.123355346008858e-05,
                -0.002579342458393642,
                -2.194257357346814e-06
            ]
        ),
        (#DtA-OCtORR-DAVOL20-PNF-SG
            [
                'debt_to_assets', # 风格因子 资产负债率
                'operating_cost_to_operating_revenue_ratio', # 质量类因子 销售成本率
                'DAVOL20', # 情绪类因子 20日平均换手率与120日平均换手率之比
                'price_no_fq', # 技术指标因子 不复权价格因子
                'sales_growth' # 风格因子 5年营业收入增长率
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
                'TVSTD6', # 情绪类因子 6日成交金额的标准差
                'cashflow_per_share_ttm', # 每股指标因子 每股现金流量净额
                'sharpe_ratio_120', # 风险类因子 120日夏普率
                'non_operating_net_profit_ttm' # 基础科目及衍生类因子 营业外收支净额TTM
            ],
            [
                -5.394060941494863e-12,
                4.6306072704138405e-05,
                -0.0030567075906980912,
                1.4227113275455325e-12
            ]
        )
    ]
    # 大盘风险控制相关变量
    g.yesterday_limit_up_count = 0  # 前日涨停股数量
    g.szzs_current_price = 0        # 上证指数当前价格
    g.szzs_ma5 = 0                  # 上证指数5日均线
    g.szzs_ma10 = 0                 # 上证指数10日均线
    g.risk_control_triggered = False  # 风险控制是否触发
    
    # 设置交易运行时间
    run_daily(prepare_stock_list, '9:05')
    run_daily(check_market_risk, '9:15')  # 新增：大盘风险检查
    run_weekly(weekly_adjustment, 1, '9:30')
    run_daily(check_limit_up, '14:00') # 检查持仓中的涨停股是否需要卖出
    run_daily(close_account, '14:30')
    run_daily(print_position_info, '15:10')

# 新增：大盘风险检查函数
def check_market_risk(context):
    """检查大盘风险并执行相应操作"""
    try:
        # 获取上证指数数据
        szzs = '000001.XSHG'
        # 获取当前价格
        current_price_data = get_price(szzs, end_date=context.current_dt, frequency='1d', fields=['close'], count=1)
        g.szzs_current_price = current_price_data['close'].iloc[0]
        
        # 获取均线数据
        ma_data = get_price(szzs, end_date=context.current_dt, frequency='1d', fields=['close'], count=20)
        g.szzs_ma5 = ma_data['close'].iloc[-5:].mean()
        g.szzs_ma10 = ma_data['close'].iloc[-10:].mean()
        
        # 检查前日涨停股数量（使用昨日数据）
        yesterday = context.previous_date
        all_stocks = list(get_all_securities('stock', yesterday).index)
        limit_up_data = get_price(all_stocks, end_date=yesterday, frequency='daily', fields=['close','high_limit'], count=1)
        if not limit_up_data.empty:
            limit_up_stocks = limit_up_data[limit_up_data['close'] == limit_up_data['high_limit']]
            g.yesterday_limit_up_count = len(limit_up_stocks)
        else:
            g.yesterday_limit_up_count = 0
            
        log.info("大盘风险检查：前日涨停数=%d, 当前价格=%.2f, 5日均线=%.2f, 10日均线=%.2f" % 
                (g.yesterday_limit_up_count, g.szzs_current_price, g.szzs_ma5, g.szzs_ma10))
        
        # 执行风险控制逻辑
        execute_risk_control(context)
        
    except Exception as e:
        log.error("大盘风险检查出错：%s" % str(e))

# 新增：执行风险控制逻辑
def execute_risk_control(context):
    """根据大盘风险条件执行减仓或清仓"""
    condition1 = g.yesterday_limit_up_count < 30  # 前日整个大盘涨停数量不足30只
    condition2 = g.szzs_current_price < g.szzs_ma10  # 当前价格跌破10日均线
    condition3 = g.szzs_current_price < g.szzs_ma5   # 当前价格跌破5日均线
    
    # 统计满足条件的数量
    satisfied_conditions = sum([condition1, condition2, condition3])
    log.info("风险条件：涨停<30只(%s), 破10日均线(%s), 破5日均线(%s), 满足条件数=%d" % 
            (condition1, condition2, condition3, satisfied_conditions))
    
    # 获取当前持仓股票列表（安全方式）
    current_positions = list(context.portfolio.positions.keys())
    
    if satisfied_conditions >= 2:
        if satisfied_conditions == 3:
            # 三个条件都满足，清仓
            log.info("三个风险条件都满足，执行清仓操作")
            clear_all_positions(context, current_positions)
            g.risk_control_triggered = True
        else:
            # 满足两个条件，减仓50%
            log.info("满足两个风险条件，执行50%减仓操作")
            reduce_positions_by_percentage(context, current_positions, 0.5)
            g.risk_control_triggered = True
    elif condition1:  # 仅满足条件1
        log.info("仅满足条件1：前日涨停数<30，执行30%减仓")
        reduce_positions_by_percentage(context, current_positions, 0.3)
        g.risk_control_triggered = True
    elif condition2:  # 仅满足条件2
        log.info("仅满足条件2：跌破10日均线，执行50%清仓且不开新仓")
        reduce_positions_by_percentage(context, current_positions, 0.5)
        g.risk_control_triggered = True
    elif condition3:  # 仅满足条件3
        log.info("仅满足条件3：跌破5日均线，执行30%减仓")
        reduce_positions_by_percentage(context, current_positions, 0.3)
        g.risk_control_triggered = True
    else:
        g.risk_control_triggered = False
        log.info("无风险条件触发，正常交易")

# 新增：清仓所有持仓
def clear_all_positions(context, position_list):
    """清仓所有持仓"""
    for stock in position_list:
        if stock in context.portfolio.positions:
            position = context.portfolio.positions[stock]
            if position.total_amount > 0:
                log.info("清仓股票[%s]" % stock)
                close_position(position)

# 新增：按比例减仓
def reduce_positions_by_percentage(context, position_list, percentage):
    """按比例减仓"""
    for stock in position_list:
        if stock in context.portfolio.positions:
            position = context.portfolio.positions[stock]
            if position.total_amount > 0:
                current_value = position.value
                target_value = current_value * (1 - percentage)
                log.info("减仓股票[%s] %.0f%%，从%.2f减至%.2f" % (stock, percentage*100, current_value, target_value))
                order_target_value_(stock, target_value)

# 1-1 准备股票池
def prepare_stock_list(context):
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

# 1-2 选股模块
def get_stock_list(context):
    # 新增：如果风险控制触发且满足不开新仓条件，则不选股
    if g.risk_control_triggered and g.szzs_current_price < g.szzs_ma10:
        log.info("风险控制触发且跌破10日均线，暂停开新仓")
        return []
    
    # 指定日期防止未来数据
    yesterday = context.previous_date
    today = context.current_dt
    # 获取初始列表
    initial_list = get_all_securities('stock', today).index.tolist()
    initial_list = filter_new_stock(context, initial_list)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_stock(initial_list)
    final_list = []
    # MS
    for factor_list,coef_list in g.factor_list:
        factor_values = get_factor_values(initial_list,factor_list, end_date=yesterday, count=1)
        df = pd.DataFrame(index=initial_list, columns=factor_values.keys())
        for i in range(len(factor_list)):
            df[factor_list[i]] = list(factor_values[factor_list[i]].T.iloc[:,0])
        df = df.dropna()
        df['total_score'] = 0
        for i in range(len(factor_list)):
            df['total_score'] += coef_list[i]*df[factor_list[i]]
        df = df.sort_values(by=['total_score'], ascending=False) # 分数越高即预测未来收益越高，排序默认降序
        complex_factor_list = list(df.index)[:int(0.1*len(list(df.index)))]
        q = query(valuation.code,valuation.circulating_market_cap,indicator.eps).filter(valuation.code.in_(complex_factor_list)).order_by(valuation.circulating_market_cap.asc())
        df = get_fundamentals(q)
        df = df[df['eps']>0]
        lst  = list(df.code)
        lst = filter_paused_stock(lst)
        lst = filter_limitup_stock(context, lst)
        lst = filter_limitdown_stock(context, lst)
        lst = lst[:min(g.stock_num, len(lst))]
        for stock in lst:
            if stock not in final_list:
                final_list.append(stock)
    return final_list

# 1-3 整体调整持仓
def weekly_adjustment(context):
    # 新增：检查是否因为风险控制暂停交易
    if g.risk_control_triggered and g.szzs_current_price < g.szzs_ma10:
        log.info("因风险控制触发且跌破10日均线，暂停调仓买入")
        return
        
    if g.no_trading_today_signal == False:
        # 获取应买入列表
        log.info("call get_stock_list")
        target_list = get_stock_list(context)
        # 调仓卖出
        for stock in g.hold_list:
            if (stock not in target_list) and (stock not in g.yesterday_HL_list):
                log.info("卖出[%s]" % (stock))
                # 修复bug：检查股票是否在持仓中
                if stock in context.portfolio.positions:
                    position = context.portfolio.positions[stock]
                    close_position(position)
                else:
                    log.info("股票[%s]已不在持仓中" % (stock))
            else:
                log.info("已持有[%s]" % (stock))
        # 调仓买入
        position_count = len(context.portfolio.positions)
        
        if position_count < 1:
            log.info("position_count = 0")

        target_num = len(target_list)
        if target_num > position_count and context.portfolio.cash > 0:
            value = context.portfolio.cash / (target_num - position_count)
            for stock in target_list:
                # 修复bug：使用安全的方式检查持仓
                if stock not in context.portfolio.positions or context.portfolio.positions[stock].total_amount == 0:
                    if open_position(stock, value):
                        if len(context.portfolio.positions) == target_num:
                            break

# 1-4 调整昨日涨停股票
def check_limit_up(context):
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
        if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68':
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
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] > current_data[stock].low_limit]

# 2-6 过滤次新股
def filter_new_stock(context,stock_list):
    yesterday = context.previous_date
    return [stock for stock in stock_list if not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=375)]

# 3-1 交易模块-自定义下单
def order_target_value_(security, value):
    if value == 0:
        log.debug("Selling out %s" % (security))
    else:
        log.debug("Order %s to value %f" % (security, value))
    return order_target_value(security, value)

# 3-2 交易模块-开仓
def open_position(security, value):
    order = order_target_value_(security, value)
    if order != None and order.filled > 0:
        return True
    return False

# 3-3 交易模块-平仓
def close_position(position):
    security = position.security
    order = order_target_value_(security, 0)  # 可能会因停牌失败
    if order != None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False

# 4-1 判断今天是否为账户资金再平衡的日期
def today_is_between(context, start_date, end_date):
    today = context.current_dt.strftime('%m-%d')
    if (start_date <= today) and (today <= end_date):
        return True
    else:
        return False

# 4-2 清仓后次日资金可转
def close_account(context):
    if g.no_trading_today_signal == True:
        if len(g.hold_list) != 0:
            for stock in g.hold_list:
                # 修复bug：检查股票是否在持仓中
                if stock in context.portfolio.positions:
                    position = context.portfolio.positions[stock]
                    close_position(position)
                    log.info("卖出[%s]" % (stock))

# 4-3 打印每日持仓信息
def print_position_info(context):
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
    
    # 打印大盘风险控制状态
    log.info("大盘风险控制状态：触发=%s, 前日涨停数=%d, 当前价格=%.2f, 5日均线=%.2f, 10日均线=%.2f" % 
            (g.risk_control_triggered, g.yesterday_limit_up_count, g.szzs_current_price, g.szzs_ma5, g.szzs_ma10))

