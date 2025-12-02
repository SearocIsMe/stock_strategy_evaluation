# 克隆自聚宽文章：https://www.joinquant.com/post/63326
# 标题：2019-2025收益率超45倍！无未来函数！
# 作者：leitingwhat

# 股票+EFT动量轮动模型
from jqdata import *  # 聚宽核心数据接口（获取股票/ETF价格、财务数据等）
from jqfactor import *  # 聚宽因子库（用于因子计算，此处未直接调用但预留）
import numpy as np  # 数值计算库（如线性回归、数组处理）
import pandas as pd  # 数据分析库（处理表格数据，如DataFrame）
from datetime import time, date, datetime, timedelta  # 时间处理库（日期加减、时间判断）
from jqdata import finance  # 聚宽财务数据接口（获取审计意见、利润表等）
import time  # 时间控制库（此处未直接调用，预留）
import json  #  JSON处理库（此处未直接调用，预留）
import math  # 数学计算库（如指数、对数计算）

import newqmt_sql

# ⭐ 在这里设置这个策略的分类标签（写入 trade.fenlei）
newqmt_sql.FENLEI = 'superlogic'      

from newqmt_sql import (
    order_zzy as order,
    order_target_zzy as order_target,
    order_value_zzy as order_value,
    order_target_value_zzy as order_target_value
)

# ==================== 函数定义 ====================
 
def after_code_changed(context):
    """代码变更后的初始化"""
    # 基础设置
    unschedule_all()  # 清空所有定时任务
    set_option('avoid_future_data', True)  # 开启防未来函数
    set_benchmark('399101.XSHE')  # 设定基准
    set_option('use_real_price', True)  # 用真实价格交易
    set_slippage(FixedSlippage(3/10000))  # 设置滑点
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=2.5/10000, 
                           close_commission=2.5/10000, close_today_commission=0, min_commission=5), type='stock')
    
    # 日志设置
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    
    # 全局变量初始化
    g.ref_stock = '000300.XSHG'  # 择时计算的基础数据
    
    # 布尔类型全局变量
    g.trading_signal = True  # 是否为可交易日
    g.run_stoploss = True  # 是否进行止损
    g.filter_audit = False  # 是否筛选审计意见
    g.adjust_num = False  # 是否调整持仓数量
    
    # 列表类型全局变量
    g.hold_list = []  # 当前持仓的全部股票
    g.yesterday_HL_list = []  # 记录持仓中昨日涨停的股票
    g.target_list = []
    g.pass_months = [1, 4]  # 空仓的月份
    g.limitup_stocks = []  # 记录涨停的股票避免再次买入
    g.stoploss_list = []  # 止损卖出列表
    g.other_sale = []  # 其他卖出列表
    
    # 数值类型全局变量
    g.min_mv = 20  # 股票最小市值要求（亿元）
    g.max_mv = 600  # 股票最大市值要求（亿元）
    g.stock_num = 2  # 持股数量
    g.etf_num = 1  # ETF数量
    
    # 止损相关设置
    g.stoploss_strategy = 1  # 1为止损线止损，2为市场趋势止损, 3为联合1、2策略
    g.stoploss_limit = 0.09  # 止损线
    g.stoploss_market = 0.05  # 市场趋势止损参数
    g.highest = 700  # 股票单价上限设置
    
    # ETF相关设置
    g.money_etf = '511880.XSHG'  # 空仓月份持有银华日利ETF
    g.stock_pool = [
        '159928.XSHE',  # 中证消费ETF
        '510300.XSHG',  # 沪深300ETF
        '159949.XSHE',  # 创业板50ETF
        '159783.XSHE',  # 双创ETF
        '518880.XSHG',  # 黄金ETF
        '159561.XSHE',  # 德国ETF
        '159915.XSHE',  # 创业板
        '513100.XSHG',  # 纳指
        '510500.XSHG',  # 中证500ETF
        '159985.XSHE'   # 豆粕etf
    ]
    
    # ETF动量参数设置
    g.momentum_day = 22  # 最新动量参考天数
    g.score_threshold = 0.7  # rsrs标准分指标阈值
    g.N = 18  # 计算最新斜率slope，拟合度r2参考天数
    g.M = 600  # 计算最新标准分zscore，rsrs_score参考天数
    g.mean_day = 30  # 计算结束ma收盘价参考天数
    g.mean_diff_day = 2  # 计算初始ma收盘价参考天数
    g.slope_series = initial_slope_series()  # 初始化斜率序列
    
    # 其他参数设置
    g.sold_stock_record = {}
    g.max_industry_stocks = 2
    g.record_days = 10
    g.last_kdj_signal = 'KEEP'
    g.count_days = 0
    
    # 设置定时任务
    run_daily(prepare_stock_list, '9:05')  # 准备股票池
    run_daily(my_trade, '9:30')  # ETF交易逻辑
    run_daily(trade_afternoon, time='14:00', reference_security='399101.XSHE')  # 下午交易检查
    run_daily(stop_loss, time='09:35')  # 止损检查
    run_daily(stop_loss, time='11:25')  # 止损检查
    run_daily(stop_loss, time='14:55')  # 止损检查
    run_daily(close_account, '10:10')  # 清仓检查
    run_weekly(weekly_adjustment, 1, '9:30')  # 每周调仓
    run_daily(trading_signal, "09:30")  # 交易信号检查
    run_daily(check_signal_change, "09:35")  # 信号变化检查
    
    # 打印初始状态
    log.info('策略初始化完成')
    log.info('账户可用资金：%.2f' % context.portfolio.available_cash)
    
def prepare_stock_list(context):
    """准备股票池，获取已持有列表和昨日涨停列表
    功能：初始化涨停股列表、获取当前持仓、筛选昨日涨停股、确定目标股票列表。"""
    g.limitup_stocks = []  # 全局变量：记录当日打开涨停的股票（避免重复买入）
    g.hold_list = list(context.portfolio.positions.keys())  # 全局变量：当前持仓股票代码列表（从账户持仓中提取）
    
    # 获取昨日涨停列表（仅针对当前持仓股，避免无关股票干扰）
    if g.hold_list:  # 若有持仓，才筛选持仓中的昨日涨停股
        # 调用聚宽get_price接口，获取持仓股前1日的收盘价和涨停价
        df = get_price(
            g.hold_list,  # 标的列表（持仓股）
            end_date=context.previous_date,  # 数据截止日：前一交易日（避免当日数据）
            frequency='daily',  # 数据频率：日线
            fields=['close', 'high_limit'],  # 取2个字段：收盘价、涨停价
            count=1,  # 取1天数据（仅前一交易日）
            panel=False,  # 返回DataFrame格式（而非Panel）
            fill_paused=False  # 停牌日不填充数据（避免无效值）
        )
        # 筛选“收盘价 == 涨停价”的股票：即昨日涨停股
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = df['code'].tolist()  # 全局变量：持仓中昨日涨停股列表
    else:
        g.yesterday_HL_list = []  # 无持仓时，昨日涨停列表为空
    
    # 获取目标股票列表：调用get_stock_list（核心选股函数），取前g.stock_num只（预设持股数）
    g.target_list = get_stock_list(context)[:g.stock_num]
    
    # 注释代码：原本用于记录交易信号到回测图表，此处预留
    # record(signal=g.trading_signal)
 
def get_stock_list(context):
    """主要选股逻辑
    功能：从指数成分股出发，经过基础过滤→基本面筛选→审计意见筛选→近期卖出过滤→行业分散→价格筛选，得到最终股票池。"""
    final_list = []  # 最终股票列表
    MKT_index = '399101.XSHE'  # 基准指数：深证综指（覆盖全市场，获取宽基成分股）
    # 初始列表：指数成分股 → 先经过filter_stocks（基础条件过滤，如排除ST、次新股）
    initial_list = filter_stocks(context, get_index_stocks(MKT_index))
    # 使用前一交易日的日期获取财务数据
    query_date = context.previous_date
    # 第一步：基本面筛选（用聚宽get_fundamentals查询财务数据）
    q = query(
        valuation.code,  # 股票代码
        indicator.eps,  # 每股收益（EPS，判断盈利性）
        valuation.market_cap,  # 市值（控制规模）
        income.np_parent_company_owners,  # 归母公司净利润（盈利真实性）
        income.net_profit,  # 净利润（盈利性）
        income.operating_revenue  # 营业收入（规模与流动性）
    ).filter(
        valuation.code.in_(initial_list),  # 仅从初始列表中选
        indicator.eps > 0,  # EPS>0：盈利为正（排除亏损股）
        valuation.market_cap.between(g.min_mv, g.max_mv),  # 市值在[min_mv, max_mv]亿之间（预设10-500亿）
        income.np_parent_company_owners > 0,  # 归母公司净利润>0（核心盈利为正）
        income.net_profit > 0,  # 净利润>0（整体盈利为正）
        income.operating_revenue > 1e8  # 营收>1亿（排除小市值垃圾股）
    ).order_by(valuation.market_cap.asc()).limit(g.stock_num * 3)  # 按市值升序排（偏好中小盘），取3倍持股数（留缓冲）
    
    df = get_fundamentals(q)  # 执行查询，获取财务数据DataFrame
    
    # 若无符合基本面的股票，返回空列表（后续会切换到ETF）
    if df is None or len(df) == 0:
        log.info('无基本面符合条件的股票')
        return []
    
    # 第二步：审计意见筛选（可选，由g.filter_audit控制，默认关闭）
    if g.filter_audit: 
        before_audit_filter = len(df)  # 筛选前股票数量
        # 对每只股票调用filter_audit（检查近3年审计意见是否合规）
        df['audit'] = df['code'].apply(lambda x: filter_audit(context, x))
        df_audit = df[df['audit'] == True]  # 保留审计合规的股票
        # 日志：记录剔除的问题审计股票数量
        log.info('去除掉了存在审计问题的股票{}只'.format(len(df) - len(df_audit)))
        df = df_audit  # 更新为审计后列表
    
    # 第三步：过滤最近卖出的股票（避免“刚卖又买”的频繁交易）
    final_list = df['code'].tolist()  # 先将审计后股票转为列表
    if hasattr(g, 'sold_stock_record'):  # 若存在“近期卖出记录”全局变量
        # 提取所有近期卖出的股票（遍历每日卖出记录）
        recently_sold_stocks = [stock for date in g.sold_stock_record for stock in g.sold_stock_record[date]]
        # 从最终列表中排除近期卖出的股票
        final_list = [stock for stock in final_list if stock not in recently_sold_stocks]
    
    # 第四步：行业分散（避免单一行业风险，由g.max_industry_stocks控制，默认每行业最多2只）
    industry_info = hyxx(final_list)  # 调用hyxx：获取每只股票的申万一级行业
    final_list = hyxz(final_list, industry_info, g.max_industry_stocks)  # 调用hyxz：按行业分散筛选
    
    # 第五步：价格筛选（排除高价股，由g.highest控制，默认100元）
    if final_list:
        # 调用history接口，获取最终列表股票的最新收盘价（前1交易日）
        last_prices = history(
            1,  # 取1天数据
            unit='1d',  # 日线
            field='close',  # 仅取收盘价
            security_list=final_list,  # 标的列表
            df=False  # 返回字典格式（key=代码，value=价格列表）
        )
        # 保留“持仓股”或“收盘价≤100元”的股票（持仓股不强制价格筛选，避免被动卖出）
        return [stock for stock in final_list if stock in g.hold_list or last_prices[stock][-1] <= g.highest]
    else:
        log.info('无适合股票，买入ETF')
        return []
 
def filter_stocks(context, stock_list):
    """过滤股票的基本条件
    功能：从指数成分股中排除高风险股票（如 ST、停牌、次新股、科创板等）。"""
    current_data = get_current_data()
    filtered_stocks = []
    
    for stock in stock_list:
        # 基本条件过滤
        if current_data[stock].paused:  # 停牌
            continue 
        if current_data[stock].is_st:  # ST 
            continue
        if '退' in current_data[stock].name:  # 退市 
            continue
        if stock.startswith('300') or stock.startswith('688') or stock.startswith('8') or stock.startswith('4') or stock.startswith('9'):
            continue
            
        # 次新股过滤
        start_date = get_security_info(stock).start_date
        if context.previous_date - start_date < timedelta(days=375):
            continue
            
        filtered_stocks.append(stock)
    
    return filtered_stocks
 
def filter_audit(context, code):
    """筛选审计意见
    功能：检查股票近 3 年审计意见是否合规（排除 “保留意见”“否定意见” 等问题意见）。"""
    lstd = context.previous_date  # 基准日期：前一交易日
    last_year = lstd.replace(year=lstd.year - 3, month=1, day=1)  # 近3年起始日：前3年1月1日
    #  query聚宽“审计意见表”（finance.STK_AUDIT_OPINION）
    q = query(
        finance.STK_AUDIT_OPINION.code,  # 股票代码
        finance.STK_AUDIT_OPINION.report_type  # 审计报告类型（1=标准无保留，2=保留，3=否定等）
    ).filter(
        finance.STK_AUDIT_OPINION.code == code,  # 仅查询当前股票
        finance.STK_AUDIT_OPINION.pub_date >= last_year  # 仅查近3年报告
    )
    df = finance.run_query(q)  # 执行财务查询
    
    # 若近3年无审计报告（罕见，如新股），默认合规
    if df is None or len(df) == 0:
        return True
        
    # 转换报告类型为字符串，检查是否包含“2/3/4/5”（问题意见类型）
    df['report_type'] = df['report_type'].astype(str)
    contains_nums = df['report_type'].str.contains(r'2|3|4|5')  # 正则匹配：是否含2/3/4/5
    return not contains_nums.any()  # 无问题意见则返回True（合规），否则False
 
# ==================== 交易逻辑 ====================
def my_trade(context):
    """主要交易逻辑 - Handles ETF trading when stock trading signal is off
    功能：当g.trading_signal=False（不能交易股票）时，9:30 执行 ETF 轮动（选动量最好的 ETF，结合择时信号买卖）。"""
    if not g.trading_signal:  # 仅当“股票交易信号关闭”时执行
        hour = context.current_dt.hour  # 当前时间小时（如9）
        minute = context.current_dt.minute  # 当前时间分钟（如30）
        if hour == 9 and minute == 30:  # 仅在9:30执行（开盘第一时间）
            # 第一步：获取ETF排名（调用get_rank，选动量分数最高的ETF）
            target_etf_rank = get_rank(context, g.stock_pool)  # 返回格式：(ETF代码, 分数)
            if target_etf_rank is None:  # 若无法获取排名（如无数据）
                log.warning("无法获取ETF排名，跳过交易")
                return
                
            target_etf = target_etf_rank[0]  # 提取排名第一的ETF代码
            g.target_list = [target_etf]  # 将目标ETF存入全局变量（后续清仓用）
            
            # 第二步：获取择时信号（调用get_timing_signal，基于RSRS策略）
            timing_signal = get_timing_signal(context, g.ref_stock)  # g.ref_stock=沪深300（000300.XSHG）
            log.info('今日自选ETF及择时信号:{} {}'.format(target_etf, timing_signal))
            
            current_positions = context.portfolio.positions  # 当前账户持仓
            
            # 第三步：根据择时信号执行交易
            if timing_signal == 'SELL':  # 择时信号为“卖出”：清仓所有持仓
                log.info("择时信号为SELL，清仓所有头寸")
                for stock in list(current_positions.keys()):  # 遍历所有持仓
                    close_position(context, stock)  # 调用平仓函数
            elif timing_signal == 'BUY' or timing_signal == 'KEEP':  # 买入/持有信号
                # 先卖出“非目标ETF”的持仓（只保留目标ETF）
                for stock in list(current_positions.keys()):
                    if stock != target_etf:  # 若持仓不是目标ETF
                        log.info("[%s] 已不在目标ETF列表中，卖出" % (stock))
                        close_position(context, stock)  # 平仓

                # 再买入目标ETF（若未持有或持仓为0）
                position_count = len(context.portfolio.positions)  # 当前持仓数量
                cash_available = context.portfolio.available_cash  # 可用现金
                target_etf_position = current_positions.get(target_etf, None)  # 目标ETF的持仓信息

                # 条件：有可用现金 + 未持有目标ETF / 持仓为0
                if cash_available > 0 and (target_etf_position is None or target_etf_position.total_amount == 0):
                    # 计算每只ETF的买入金额：可用现金 / 需买入的ETF数量（g.etf_num=1，即全仓买入）
                    if g.etf_num > position_count:  # 若需买入的ETF数量 > 当前持仓数（即还能买）
                        value = cash_available / (g.etf_num - position_count)  
                        log.info("买入目标ETF [%s]（%s元）" % (target_etf, value))
                        order_target_value(target_etf, value)  # 下单：买入目标市值的ETF

def weekly_adjustment(context):
    """每周调仓逻辑 - Handles stock trading when trading signal is ON
    功能：每周一 9:30 执行（run_weekly(..., 1, '9:30')），当g.trading_signal=True时，卖出非目标股、买入新目标股。"""
    if g.trading_signal:  # 仅当“股票交易信号开启”时执行
        # 第一步：确定卖出/保留列表
        # 卖出列表：持仓不在目标列表 + 非昨日涨停（昨日涨停股暂留，避免错过连续涨停）
        sell_list = [stock for stock in g.hold_list if stock not in g.target_list and stock not in g.yesterday_HL_list]
        # 保留列表：持仓在目标列表 + 昨日涨停（需继续持有的股票）
        hold_list = [stock for stock in g.hold_list if stock in g.target_list or stock in g.yesterday_HL_list]
        log.info("计划卖出(非目标且非昨日涨停):[%s]" % (str(sell_list)))
        log.info("计划保留(目标或昨日涨停):[%s]" % (str(hold_list)))
        
        # 第二步：执行卖出（遍历卖出列表，平仓非保留股）
        current_positions = context.portfolio.positions
        for stock in sell_list:
            if stock in current_positions:  # 若该股票仍在持仓中
                close_position(context, stock)  # 平仓
            
        # 第三步：执行买入（买入目标列表中未持仓的股票）
        buy_list = [stock for stock in g.target_list if stock not in g.hold_list]  # 新买入列表
        log.info("计划买入(新目标):[%s]" % (str(buy_list)))
        # 调用buy_security：买入股票，需买入数量=预设持股数 - 保留股数量（g.stock_num - len(hold_list)）
        buy_security(context, buy_list, g.stock_num - len(hold_list))
 
def buy_security(context, target_list, num_to_buy):
    """买入证券 (Stocks)
    功能：按 “平均分配资金” 原则买入目标股票，处理现金不足、下单异常等情况。"""
    position_count = len(context.portfolio.positions)  # 当前持仓数量
    target_buy_count = num_to_buy  # 需买入的股票数量（由调仓逻辑传入）
    
    # 条件：需买入数量>0 + 有可用现金
    if target_buy_count > 0 and context.portfolio.available_cash > 0:
        # 每只股票的目标买入金额：可用现金 / 需买入数量（平均分配）
        value_per_stock = context.portfolio.available_cash / target_buy_count
        
        bought_count = 0  # 已买入数量计数器
        for stock in target_list:  # 遍历目标买入列表
            if bought_count >= target_buy_count:  # 已买够需买入数量，停止
                break
                
            # 现金不足检查：假设最低买入金额1000元（避免零碎下单）
            if context.portfolio.available_cash < 1000:  
                log.warning("现金不足，无法继续买入 %s" % stock)
                break
            
            # 尝试下单：买入目标市值的股票
            try:
                order_target_value(stock, value_per_stock)  # 聚宽下单接口：目标市值买入
                log.info("买入[%s]（目标价值 %.2f元）" % (stock, value_per_stock))
                bought_count += 1  # 已买入数量+1
            except Exception as e:  # 捕获下单异常（如无权限、停牌）
                log.error("尝试买入 [%s] 时下单失败: %s" % (stock, str(e)))
            
            # 持仓数量检查：若已达预设持股数（g.stock_num），停止
            if len(context.portfolio.positions) >= g.stock_num:
                break
    elif target_buy_count <= 0:  # 无需买入（如持仓已达预设数量）
        log.info("无需买入新的股票。")
 
def close_position(context, security):
    """平仓操作
    功能：将指定标的（股票 / ETF）平仓（目标市值设为 0），并记录卖出记录。"""
    try:
        # 聚宽下单接口：将security的目标市值设为0（即清仓）
        order_target_value(security, 0)
        record_recently_sold_stocks(context, security)  # 调用函数：记录该股票的卖出记录
        return True  # 平仓成功
    except Exception as e:  # 捕获平仓异常（如停牌、跌停无法卖出）
        log.error("平仓操作失败 [%s]: %s" % (security, str(e)))
        return False  # 平仓失败
 
def close_account(context):
    """清仓函数，在交易信号为False时执行
    功能：每日 10:10 执行，当g.trading_signal=False时，清仓非目标 ETF（仅保留货币 ETF 和目标 ETF）。"""
    if not g.trading_signal:  # 仅当“股票交易信号关闭”时执行
        curr_data = get_current_data()  # 获取当前市场数据（停牌、跌停状态）
        current_positions = context.portfolio.positions  # 当前持仓
        
        # 遍历所有持仓（仅当有持仓时）
        if len(g.hold_list) > 0:
            for stock in list(current_positions.keys()):
                # 跳过货币ETF（g.money_etf=银华日利511880.XSHG，空仓时持有）
                if stock == g.money_etf:  
                    continue
                # 跳过目标ETF（my_trade中已确定的目标ETF，需保留）
                if stock in g.target_list:  
                    continue
                # 跳过无数据的标的（罕见情况）
                if stock not in curr_data:  
                    log.warning("无法获取 %s 的当前数据，跳过清仓检查" % stock)
                    continue
                # 跳过停牌股（无法卖出）
                if curr_data[stock].paused:  
                    continue
                # 跳过跌停股（无法卖出）
                if curr_data[stock].last_price <= curr_data[stock].low_limit:  
                    continue
                
                # 执行清仓：非目标ETF、非货币ETF、可卖出的标的
                log.info("交易信号为False，清仓非目标ETF [%s]" % (stock))
                close_position(context, stock)
 
# ==================== 风控逻辑 ====================
def stop_loss(context):
    """止损逻辑
    功能：每日 3 个时间点（9:35、11:25、14:55）执行，分个股止损 / 止盈和市场趋势止损。"""
    if g.run_stoploss:  # 仅当“开启止损”（g.run_stoploss=True）时执行
        current_positions = context.portfolio.positions  # 当前持仓
        
        # ------------ 个股止损/止盈策略（策略1或3）------------
        if g.stoploss_strategy == 1 or g.stoploss_strategy == 3:
            for stock in current_positions.keys():  # 遍历每只持仓股
                position = current_positions[stock]  # 持仓详情
                price = position.price  # 当前股价
                avg_cost = position.avg_cost  # 持仓平均成本
                # 获取前一交易日收盘价
                prev_data = attribute_history(stock, 1, '1d', ['close'])
                if prev_data is not None and len(prev_data) > 0:
                    price = prev_data['close'][-1]  # 使用前一交易日收盘价
                    avg_cost = position.avg_cost
                    
                    if price >= avg_cost * 2:  # 止盈
                        close_position(context, stock)
                    elif price < avg_cost * (1 - g.stoploss_limit):  # 止损
                        close_position(context, stock)
        
        # ------------ 市场趋势止损策略（策略2或3）------------
        if g.stoploss_strategy == 2 or g.stoploss_strategy == 3:
            try:
                # 获取深证综指成分股前1交易日的开盘价和收盘价
                stock_df = get_price(
                    security=get_index_stocks('399101.XSHE'),  # 深证综指成分股
                    end_date=context.previous_date,  # 前一交易日
                    frequency='daily',  # 日线
                    fields=['close', 'open'],  # 收盘价、开盘价
                    count=1,  # 1天数据
                    panel=False  # DataFrame格式
                )
                if stock_df is not None and len(stock_df) > 0:
                    # 计算市场平均跌幅：(1 - 收盘价/开盘价)的均值（即全市场下跌幅度）
                    down_ratio = (1 - stock_df['close'] / stock_df['open']).mean()
                    
                    # 市场止损条件：平均跌幅 ≥ 预设阈值（g.stoploss_market=5%）
                    if down_ratio >= g.stoploss_market:
                        for stock in current_positions.keys():  # 清仓所有持仓
                            g.stoploss_list.append(stock)  # 记录止损
                            log.debug("大盘惨跌,平均降幅{:.2%}".format(down_ratio))
                            close_position(context, stock)  # 平仓
                            g.other_sale.append(stock)  # 记录卖出
            except Exception as e:  # 捕获计算异常
                log.error("市场趋势止损计算失败: %s" % str(e))
 
def check_holdings_decline(context):
    """检查持仓标的的价格变化情况，
    功能：检查是否 90% 以上持仓股下跌超过 1%，若是则触发清仓信号（切换到 ETF）。"""
    positions = context.portfolio.positions  # 当前持仓
    
    if not positions:  # 无持仓时返回False
        return False
    
    stocks = list(positions.keys())  # 持仓股列表
    try:
        # 获取持仓股近2个交易日的收盘价（前2天和前1天，计算涨跌幅）
        price_data = history(
            2,  # 2天数据
            unit='1d',  # 日线
            field='close',  # 收盘价
            security_list=stocks,  # 持仓股
            df=False  # 字典格式
        )
        
        decline_count = 0  # 下跌超过1%的股票数量
        for stock in stocks:
            # 确保该股票有2天价格数据
            if stock in price_data and len(price_data[stock]) >= 2:
                yesterday_price = price_data[stock][0]  # 前2天收盘价
                today_price = price_data[stock][1]  # 前1天收盘价
                # 计算涨跌幅：(今日-昨日)/昨日 < -1%（下跌超过1%）
                if (today_price - yesterday_price) / yesterday_price < -0.01:
                    decline_count += 1
        
        decline_ratio = decline_count / len(stocks)  # 下跌股票占比
        
        # 触发条件：90%以上持仓下跌超过1%
        if decline_ratio >= 0.9:
            g.trading_signal = False  # 关闭股票交易信号（切换到ETF）
            g.count_days = 1  # 清仓计数天数=1
            log.info("触发清仓信号：{:.2%}的持仓股票下跌超过1%".format(decline_ratio))
            return True
    except Exception as e:  # 捕获数据异常
        log.error("检查持仓下跌失败: %s" % str(e))
    
    return False
 
# ==================== 辅助函数 ====================
def record_recently_sold_stocks(context, stock):
    """记录最近卖出的股票，
    功能：记录卖出的股票（排除 ETF），并清理超过g.record_days（10 天）的旧记录，避免刚卖又买。"""
    # Only record sold stocks, not ETFs from g.stock_pool
    # 仅记录股票，不记录ETF（g.stock_pool是ETF列表）
    if stock not in g.stock_pool:
        current_date = context.current_dt.date()  # 当前日期（交易日）
        
        # 若当前日期不在卖出记录中，初始化空列表
        if current_date not in g.sold_stock_record:
            g.sold_stock_record[current_date] = []
        g.sold_stock_record[current_date].append(stock)  # 加入当前卖出的股票
    
        # 清理过期记录：保留近g.record_days（10天）的交易日记录
        # 获取“当前日期-10天”到“当前日期”的所有交易日
        trade_days = get_trade_days(start_date=current_date - timedelta(days=g.record_days), end_date=current_date)
        # 遍历卖出记录的所有日期，删除非近期交易日的记录
        for date_key in list(g.sold_stock_record.keys()):
            if date_key not in trade_days:
                del g.sold_stock_record[date_key]

def hyxx(stocks):
    """获取行业信息
    功能：获取股票的申万一级行业（聚宽get_industry接口），返回 “股票代码→行业名称” 的 Series。"""
    if not stocks:  # 无股票时返回空Series
        return pd.Series()
    # 聚宽接口：获取股票的行业分类（默认申万一级）
    industry_data = get_industry(stocks, date=None)  # date=None表示最新行业
    # 构建Series：key=股票代码，value=申万一级行业名称（无行业则为None）
    return pd.Series({
        stock: industry_data[stock]['sw_l1']['industry_name'] 
        if stock in industry_data and 'sw_l1' in industry_data[stock] 
        else None 
        for stock in stocks
    })

def hyxz(stocks, industry_info, max_industry_stocks):
    """行业分散
    功能：控制每个行业的最大持股数（max_industry_stocks=2），避免单一行业风险"""
    if industry_info is None or len(stocks) == 0:  # 无行业信息或无股票，返回原列表
        return stocks
    counts = {}  # 字典：key=行业名称，value=该行业已选股票数量
    result = []  # 行业分散后的股票列表
    for stock in stocks:
        industry = industry_info[stock]  # 该股票的行业
        if industry is None:  # 无行业分类的股票，直接加入
            result.append(stock)
        # 若该行业已选数量 < 最大限制，加入并计数+1
        elif counts.get(industry, 0) < max_industry_stocks:
            result.append(stock)
            counts[industry] = counts.get(industry, 0) + 1
    return result

def print_position_info(context):
    """打印持仓信息
    功能：输出持仓股的关键信息（代码、成本、现价、收益率、持仓数、市值），用于回测调试。"""
    for position in list(context.portfolio.positions.values()):  # 遍历每只持仓
        security = position.security  # 股票代码
        cost = position.avg_cost  # 平均成本
        price = position.price  # 当前价格
        ret = 100 * (price / cost - 1)  # 收益率（百分比）
        value = position.value  # 持仓市值
        amount = position.total_amount  # 持仓数量
        # 打印详细信息
        log.info('代码:%s, 成本价:%.2f, 现价:%.2f, 收益率:%.2f%%, 持仓:%d, 市值:%.2f' % 
                (security, cost, price, ret, amount, value))
    # 分割线：区分不同时间的持仓日志
    log.info('———————————————————————————————————————分割线————————————————————————————————————————')

# ==================== ETF动量策略相关函数 ====================
def get_rank(context, etf_list):
    """对ETF列表进行排名打分
    功能：对g.stock_pool（预设 ETF 列表）计算动量分数，返回分数最高的 ETF（含分数）。"""
    etf_rank = []  # 列表：存储(ETF代码, 分数)
    for etf in etf_list:  # 遍历每只ETF
        score = get_etf_score(etf)  # 调用get_etf_score：计算ETF动量分数
        if score is not None:  # 若分数有效（无数据时为None）
            etf_rank.append((etf, score))
    
    if not etf_rank:  # 无有效分数的ETF，返回None
        return None
        
    # 按分数降序排序（分数越高，动量越好）
    etf_rank = sorted(etf_rank, key=lambda x: x[1], reverse=True)
    return etf_rank[0]  # 返回排名第一的ETF（代码+分数）

def get_etf_score(etf):
    """计算ETF动量分数
    功能：基于 “线性回归斜率（趋势）+ R 平方（拟合度）” 计算 ETF 动量分数，兼顾 “趋势强度” 和 “稳定性”。"""
    try:
        # 获取ETF近g.momentum_day（20天）的收盘价（复权后）
        data = attribute_history(
            etf,  # ETF代码
            g.momentum_day,  # 20天数据
            '1d',  # 日线
            ['close']  # 仅收盘价
        )
        if data is None or len(data) < g.momentum_day:  # 数据不足20天，返回None
            return None
            
        # 第一步：取收盘价的对数（将指数增长转为线性增长，符合回归假设）
        y = np.log(data['close'])
        # 第二步：构建x轴（时间序列，0,1,2,...,19）
        x = np.arange(len(y))
        
        # 第三步：线性回归拟合（y = slope*x + intercept），仅取斜率（趋势方向与强度）
        slope, _ = np.polyfit(x, y, 1)
        
        # 第四步：计算年化收益率（斜率→日收益率→年化）
        # 日收益率≈slope（因y=ln(price)，Δy≈日收益率），年化= (e^slope)^250 - 1（250个交易日）
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        
        # 第五步：计算R平方（拟合度，0-1，越接近1说明趋势越稳定）
        y_pred = slope * x + _  # 回归预测值
        residuals = y - y_pred  # 残差（实际值-预测值）
        ss_res = np.sum(residuals**2)  # 残差平方和（趋势波动）
        ss_tot = np.sum((y - np.mean(y))** 2)  # 总平方和（整体波动）
        if ss_tot == 0:  # 无波动（罕见），R平方=0
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)  # R平方=1-残差占比（越大越稳定）
        
        # 最终分数：年化收益 × R平方（兼顾收益和稳定性，避免高波动的“伪趋势”）
        return annualized_returns * r_squared
    except Exception as e:  # 捕获计算异常（如无数据）
        log.error("计算ETF分数失败 [%s]: %s" % (etf, str(e)))
        return None

# ==================== 择时信号函数 ====================
def get_ols(x, y):
    """计算OLS回归
    功能：计算 x（自变量）和 y（因变量）的线性回归参数（截距、斜率、R 平方），为 RSRS 策略提供基础。"""
    # 线性回归：y = slope*x + intercept，返回截距和斜率
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept  # 预测值
    residuals = y - y_pred  # 残差
    
    # 计算R平方：处理特殊情况（数据点≤1或无波动）
    if len(y) <= 1 or np.var(y, ddof=1) == 0:
        r2 = 0
    else:
        # R平方=1 - (残差平方和 / (n-1)*y的方差)
        r2 = 1 - (np.sum(residuals**2) / ((len(y) - 1) * np.var(y, ddof=1)))
    return (intercept, slope, r2)  # 返回截距、斜率、R平方

def initial_slope_series():
    """初始化斜率序列
    功能：为g.ref_stock（沪深 300）计算初始的斜率序列（用于后续 Z 分数标准化）。"""
    try:
        # 获取沪深300近g.N+g.M（18+600=618天）的最高价和最低价
        data = attribute_history(
            g.ref_stock,  # 沪深300（000300.XSHG）
            g.N + g.M,  # 18+600=618天（N=18：单段拟合天数，M=600：标准化窗口）
            '1d',  # 日线
            ['high', 'low']  # 最高价、最低价
        )
        if data is None or len(data) < g.N + g.M:  # 数据不足，返回空列表
            return []
        # 计算M个斜率：每N天拟合一次（从第0天到第M-1天，共600个斜率）
        return [get_ols(data.low[i:i+g.N].values, data.high[i:i+g.N].values)[1] for i in range(g.M)]
    except Exception as e:  # 捕获异常
        log.error("初始化斜率序列失败: %s" % str(e))
        return []

def get_zscore(slope_series):
    """计算Z分数
    功能：将斜率序列标准化为 Z 分数，用于判断当前趋势偏离历史的程度。"""
    if len(slope_series) == 0:  # 无序列数据，返回0
        return 0
    mean = np.mean(slope_series)  # 序列均值
    std = np.std(slope_series)  # 序列标准差
    if std == 0:  # 无波动（罕见），返回0
        return 0
    # Z分数=（当前斜率-均值）/ 标准差（>0：强于历史平均，<0：弱于历史平均）
    return (slope_series[-1] - mean) / std

def get_timing_signal(context, stock):
    """获取择时信号
    功能：基于沪深 300 的 “高低价斜率 + Z 分数 + R 平方” 生成择时信号（BUY/SELL/KEEP）。"""
    try:
        # 冗余数据获取（原代码未使用，预留）
        close_data = attribute_history(g.ref_stock, g.mean_day + g.mean_diff_day, '1d', ['close'])
        # 获取沪深300近g.N（18天）的高低价（用于拟合斜率）
        high_low_data = attribute_history(
            g.ref_stock,  # 沪深300
            g.N,  # 18天
            '1d',  # 日线
            ['high', 'low']  # 高低价
        )
        
        # 数据不足时，返回KEEP（观望）
        if close_data is None or high_low_data is None or len(high_low_data) < g.N:
            return "KEEP"

        # 拟合近18天的高低价斜率（RSRS核心：用最低价对最高价回归，斜率反映趋势强度）
        intercept, slope, r2 = get_ols(high_low_data.low.values, high_low_data.high.values)
        g.slope_series.append(slope)  # 将新斜率加入全局斜率序列
        
        # 保持斜率序列长度为g.M（600天）：超过则删除最早的斜率
        if len(g.slope_series) > g.M:
            g.slope_series = g.slope_series[-g.M:]
        
        # 计算RSRS分数：Z分数 × R平方（Z分数反映趋势强度，R平方反映趋势稳定性）
        if len(g.slope_series) < g.M:  # 序列长度不足600天，分数为0
            rsrs_score = 0
        else:
            rsrs_score = get_zscore(g.slope_series[-g.M:]) * r2  # 标准化后乘拟合度
        
        # 生成择时信号：
        if rsrs_score > g.score_threshold:  # 分数>0.7（强趋势）→ BUY（买入）
            return "BUY"
        elif rsrs_score < -g.score_threshold:  # 分数<-0.7（弱趋势）→ SELL（卖出）
            return "SELL"
        else:
            return "KEEP"  # 中间区间→观望
    except Exception as e:  # 捕获异常，返回观望
        log.error("获取择时信号失败: %s" % str(e))
        return "KEEP"

# ==================== 交易信号管理 ====================
def trading_signal(context):
    """更新交易信号
    功能：每日 9:30 执行，调用market_condition判断当前是否适合交易股票，更新全局信号。"""
    if hasattr(g, 'trading_signal'):
        g.previous_trading_signal = g.trading_signal
    else:
        g.previous_trading_signal = True  # 初始默认开启

    # 调用market_condition：基于KDJ判断当前市场是否适合交易股票
    g.trading_signal = market_condition(context)
    
    # 日志：输出当前交易信号状态
    if g.trading_signal:
        log.info("择时信号: True (继续交易)")
    else:
        log.info("择时信号: False (清仓或转ETF)")

def check_signal_change(context):
    """检查信号变化, 若从False变为True, 执行周调仓
    功能：每日 9:35 执行，若信号从 “False（ETF）” 变为 “True（股票）”，立即执行周调仓（无需等周一）。"""
    if not hasattr(g, 'previous_trading_signal'):
        g.previous_trading_signal = g.trading_signal

    # 信号从“关闭”变为“开启”：市场由差转好，立即调仓买股票
    if g.previous_trading_signal == False and g.trading_signal == True:
        log.info("交易信号从 False 变为 True，执行周调仓逻辑...")
        weekly_adjustment(context)  # 调用周调仓函数

# ==================== 市场条件判断 ====================
def market_condition(context):
    """根据长周期KDJ判断市场情况
    功能：核心市场判断逻辑，用长周期 KDJ（27,9,9） 判断市场趋势，决定是否开启股票交易。"""
    try:
        # 第一步：调用calculate_angle_signal获取KDJ市场信号（BUY/SELL/KEEP）
        KDJ_signal = calculate_angle_signal(context)
        
        # 若KDJ信号计算失败，保持当前信号（默认开启）
        if KDJ_signal is None:
            log.warning("KDJ 信号计算失败，保持当前交易信号状态")
            if not hasattr(g, 'trading_signal') or g.trading_signal is None:
                g.trading_signal = True
                g.count_days = 0
                log.info("KDJ 信号失败，设置默认交易信号为True")
        else:
            # 根据KDJ信号更新交易信号：
            if KDJ_signal == 'BUY' or KDJ_signal == 'KEEP':  # KDJ信号好→开启股票交易
                g.trading_signal = True
                g.count_days = 0  # 清仓计数重置为0
            elif KDJ_signal == 'SELL':  # KDJ信号差→关闭股票交易
                g.trading_signal = False
                g.count_days = 1  # 清仓计数=1
        
        # 兜底：若交易信号未定义，默认开启
        if not hasattr(g, 'trading_signal') or g.trading_signal is None:
            g.trading_signal = True
            g.count_days = 0
            log.warning("g.trading_signal 为None，设置默认值为True")
        
        # 强制恢复逻辑：若关闭交易超过15天，强制开启（避免长期空仓错过机会）
        if g.trading_signal == False:
            g.count_days += 1  # 清仓计数+1
            log.info("保持清仓信号，计数天数: %d" % g.count_days)
            
            if g.count_days > 15:  # 超过15天→强制开启，执行调仓
                g.trading_signal = True
                g.count_days = 0
                log.info("计数超过15天，强制触发交易信号")
                weekly_adjustment(context)
        
        return g.trading_signal  # 返回最终交易信号
        
    except Exception as e:  # 捕获异常，默认开启交易
        log.error("早上市场判断出错: %s" % str(e))
        if not hasattr(g, 'trading_signal') or g.trading_signal is None:
            g.trading_signal = True
            g.count_days = 0
            log.info("异常情况下设置默认交易信号为True")
        return g.trading_signal

def calculate_angle_signal(context, stock_code='399101.XSHE'):
    """计算KDJ择时信号
    功能：基于深证综指（399101.XSHE）计算长周期 KDJ（N=27, M1=9, M2=9），生成市场买卖信号。"""
    N = 27  # KDJ参数：RSV计算窗口（27天，长周期更稳定）
    M1 = 9  # K值平滑窗口（9天）
    M2 = 9  # D值平滑窗口（9天）
    
    try:
        # 获取深证综指近N+20（47天）的收盘价、最高价、最低价（确保计算RSV时有足够数据）
        df = get_price(
            stock_code,  # 深证综指
            end_date=context.previous_date,  # 前一交易日
            count=N + 20,  # 47天数据
            frequency='daily',  # 日线
            fields=['close', 'high', 'low'],  # 收盘价、最高价、最低价
            skip_paused=True  # 跳过停牌日
        )
    except Exception as e:  # 捕获数据异常
        log.error("获取数据时出错：%s" % str(e))
        return None
    
    # 数据不足（需至少N+3天，避免计算KDJ时窗口不够）
    if df is None or len(df) < N + 3:
        log.error("数据不足，需要至少%d天数据，当前只有%d天" % (N + 3, len(df) if df is not None else 0))
        return None
    
    # 第一步：计算RSV（未成熟随机值，0-100）
    low_n = df['low'].rolling(window=N, min_periods=1).min()  # N天内最低价（滚动窗口）
    high_n = df['high'].rolling(window=N, min_periods=1).max()  # N天内最高价（滚动窗口）
    # RSV = (收盘价 - N天最低价) / (N天最高价 - N天最低价) * 100（反映当前价格在N天区间的位置）
    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)  # 填充NaN值（如N天内无波动），默认50
    
    # 第二步：计算K值（RSV的指数平滑，初始值50）
    k_values = pd.Series(index=rsv.index, dtype=float)
    k_values.iloc[0] = 50  # 第一个K值=50
    # K(n) = K(n-1)*(M1-1)/M1 + RSV(n)/M1（平滑公式）
    for i in range(1, len(rsv)):
        k_values.iloc[i] = (k_values.iloc[i-1] * (M1 - 1) + rsv.iloc[i]) / M1
    
    # 第三步：计算D值（K值的指数平滑，初始值50）
    d_values = pd.Series(index=k_values.index, dtype=float)
    d_values.iloc[0] = 50  # 第一个D值=50
    # D(n) = D(n-1)*(M2-1)/M2 + K(n)/M2（平滑公式）
    for i in range(1, len(k_values)):
        d_values.iloc[i] = (d_values.iloc[i-1] * (M2 - 1) + k_values.iloc[i]) / M2
    
    # 第四步：计算J值（J=3K-2D，反映K、D的偏离程度）
    j_values = 3 * k_values - 2 * d_values
    
    # 数据不足（至少2个J值才能判断趋势）
    if len(j_values) < 2:
        log.error("KDJ 计算后数据不足")
        return None
    
    # 提取最新2个交易日的KDJ值（当前值和前一日值）
    last_j = j_values.iloc[-1]  # 当前J
    last_k = k_values.iloc[-1]  # 当前K
    last_d = d_values.iloc[-1]  # 当前D
    prev_j = j_values.iloc[-2]  # 前一日J
    prev_k = k_values.iloc[-2]  # 前一日K
    prev_d = d_values.iloc[-2]  # 前一日D
    
    # 排除NaN值（计算异常）
    if (pd.isna(last_j) or pd.isna(last_k) or pd.isna(last_d) or
        pd.isna(prev_j) or pd.isna(prev_k) or pd.isna(prev_d)):
        log.error("KDJ 值包含NaN，无法判断信号")
        return None
    
    # 初始化上一次KDJ信号（默认KEEP）
    if not hasattr(g, 'last_kdj_signal'):
        g.last_kdj_signal = 'KEEP'
    
    # -------------------------- SELL信号条件1：J从超买区回落穿过K、D --------------------------
    if (prev_j > 70 and  # 前一日J>70（超买区）
        prev_j > prev_k and  # 前一日J>K（趋势向上）
        prev_j > prev_d and  # 前一日J>D（趋势向上）
        last_j < prev_j and  # 当前J<前一日J（趋势反转向下）
        last_j <= last_k and  # 当前J≤K（J穿过K向下）
        last_j <= last_d):  # 当前J≤D（J穿过D向下）
        g.last_kdj_signal = 'SELL'
        log.info("KDJ 触发SELL信号 - 前J:%.2f, 前K:%.2f, 前D:%.2f, 当前J:%.2f, 当前K:%.2f, 当前D:%.2f" %
                (prev_j, prev_k, prev_d, last_j, last_k, last_d))
        return 'SELL'
    
    # -------------------------- SELL信号条件2：J持续低于K、D且继续下跌 --------------------------
    elif (prev_j < prev_k and  # 前一日J<K
          prev_j < prev_d and  # 前一日J<D
          last_j < last_k and  # 当前J<K
          last_j < last_d and  # 当前J<D
          last_j < prev_j and  # 当前J<前一日J（继续下跌）
          last_j > 10):  # J>10（未到超卖区，避免过早买入）
        g.last_kdj_signal = 'SELL'
        log.info("KDJ 触发新增SELL信号 - 前J:%.2f, 前K:%.2f, 前D:%.2f, 当前J:%.2f, 当前K:%.2f, 当前D:%.2f" %
                (prev_j, prev_k, prev_d, last_j, last_k, last_d))
        return 'SELL'
    
    # -------------------------- BUY信号条件：J<0（超卖区，市场低估） --------------------------
    elif last_j < 0:
        g.last_kdj_signal = 'BUY'
        log.info("KDJ 触发BUY信号 - 当前J:%.2f, 当前K:%.2f, 当前D:%.2f" % (last_j, last_k, last_d))
        return 'BUY'
    
    # -------------------------- 无信号：保持上一次信号 --------------------------
    else:
        if g.last_kdj_signal == 'BUY':
            return 'KEEP'  # 上一次是BUY，当前无新信号→继续持有
        elif g.last_kdj_signal == 'SELL':
            return 'SELL'  # 上一次是SELL，当前无新信号→继续空仓
        else:
            return 'KEEP'  # 初始状态→观望

# ==================== 其他辅助函数 ====================
def trade_afternoon(context):
    """下午交易检查
    功能：每日 14:00 执行，包含 “昨日涨停股打开检查” 和 “剩余资金补仓检查”。"""
    check_limit_up(context)  # 检查昨日涨停股是否打开涨停
    check_remain_amount(context)  # 检查剩余资金是否需要补仓

def check_limit_up(context):
    if not g.yesterday_HL_list:
        return

    # 安全时间：减去1分钟，避免读取未完成K线
    safe_end_time = context.current_dt - timedelta(minutes=1)

    for stock in g.yesterday_HL_list:
        try:
            current_data = get_price(
                stock,
                end_date=safe_end_time,
                frequency='1m',
                fields=['close', 'high_limit'],
                count=1,
                skip_paused=False,
                fq='pre',
                panel=False,
                fill_paused=True
            )
            if current_data is None or len(current_data) == 0:
                continue

            close_price = current_data['close'].iloc[0]
            high_limit = current_data['high_limit'].iloc[0]

            if close_price < high_limit:  # 已打开涨停
                log.info("[%s] 涨停打开，卖出" % stock)
                close_position(context, stock)
                g.other_sale.append(stock)
                g.limitup_stocks.append(stock)
            else:
                log.info("[%s] 仍涨停，继续持有" % stock)

        except Exception as e:
            log.error("检查涨停状态失败 [%s]: %s" % (stock, str(e)))


def check_remain_amount(context):
    """如果昨天有股票卖出或者买入失败造成空仓，剩余的金额当日买入
    功能：若因 “卖出股票” 或 “止损” 导致持仓不足g.stock_num（3 只），用剩余资金补买目标股票；止损仓位补买货币 ETF"""
    if g.trading_signal:  # 仅当股票交易信号开启时执行
        # 计算需补仓的数量：其他卖出数量 + 止损卖出数量
        addstock_num = len(g.other_sale)
        loss_num = len(g.stoploss_list)
        empty_num = addstock_num + loss_num
        
        # 更新当前持仓列表
        g.hold_list = list(context.portfolio.positions.keys())
        # 若当前持仓数 < 预设持股数（3只）→ 需要补仓
        if len(g.hold_list) < g.stock_num:
            # 核心逻辑：仅用“其他卖出”的仓位补股票，“止损”的仓位补货币ETF（降低风险）
            num_stocks_to_buy = min(addstock_num, g.stock_num - len(g.hold_list))
            # 目标补仓列表：目标股票中排除“打开涨停的股票”（g.limitup_stocks），取前num_stocks_to_buy只
            target_list = [stock for stock in g.target_list if stock not in g.limitup_stocks][:num_stocks_to_buy]
            log.info('有余额可用' + str(round(context.portfolio.available_cash, 2)) + '元。买入' + str(target_list))
            buy_security(context, target_list, len(target_list))  # 补买股票
            
            # 止损仓位补买货币ETF（若有止损卖出）
            if loss_num != 0:
                log.info('有余额可用' + str(round(context.portfolio.available_cash, 2)) + '元。买入货币基金' + str(g.money_etf))
                # 注释代码：原本补买货币ETF，此处预留
                # buy_security(context, [g.money_etf], loss_num)
        
        # 重置卖出列表（避免重复补仓）
        g.stoploss_list = []
        g.other_sale = []

# ==================== 初始化函数 ====================
def initialize(context):
    """聚宽策略初始化函数"""
    after_code_changed(context)
